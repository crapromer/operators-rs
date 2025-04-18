use super::{args::SchemeLayout, Args, MatMul};
use crate::{
    common_cpu::Cpu, type_not_support, ByteOf, LaunchError, QueueAlloc, SchemeError, Workspace,
};
mod archutil;
use core::slice;
use ggml_quants::Q8_0;
use half::f16;
use rayon::prelude::*;
use std::arch::x86_64::*;
use std::mem;
use std::simd::num::SimdFloat;
pub struct Operator;

pub fn vec_dot_q8_0_q8_0_avx2(abs: &[Q8_0], bbs: &[Q8_0]) -> f32 {
    use std::arch::x86_64::*;

    use archutil::x86_64::*;

    debug_assert_eq!(abs.len(), bbs.len());

    unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        for [(abs0, bbs0), (abs1, bbs1)] in abs.iter().zip(bbs).array_chunks::<2>() {
            let d0 = _mm256_set1_ps(abs0.delta.to_f32() * bbs0.delta.to_f32());
            let d1 = _mm256_set1_ps(abs1.delta.to_f32() * bbs1.delta.to_f32());

            let qa0 = _mm256_loadu_si256(abs0.quants.as_ptr() as *const __m256i);
            let qb0 = _mm256_loadu_si256(bbs0.quants.as_ptr() as *const __m256i);

            let qa1 = _mm256_loadu_si256(abs1.quants.as_ptr() as *const __m256i);
            let qb1 = _mm256_loadu_si256(bbs1.quants.as_ptr() as *const __m256i);

            let q0 = mul_sum_i8_pairs_float(qa0, qb0);
            let q1 = mul_sum_i8_pairs_float(qa1, qb1);

            acc0 = _mm256_fmadd_ps(d0, q0, acc0);
            acc1 = _mm256_fmadd_ps(d1, q1, acc1);
        }

        if abs.len() % 2 == 1 {
            let a = abs.last().unwrap_unchecked();
            let b = bbs.last().unwrap_unchecked();

            let d = _mm256_set1_ps(a.delta.to_f32() * b.delta.to_f32());

            let qa = _mm256_loadu_si256(a.quants.as_ptr() as *const __m256i);
            let qb = _mm256_loadu_si256(b.quants.as_ptr() as *const __m256i);

            let q = mul_sum_i8_pairs_float(qa, qb);

            acc0 = _mm256_fmadd_ps(d, q, acc0);
        }

        hsum_float_8(_mm256_add_ps(acc0, acc1))
    }
}

pub fn quantize_f32_q8_0(data: *const f32, ld: usize, rows: usize, columns: usize) -> Vec<Q8_0> {
    //TODO:实现对任意维度的量化
    use std::simd::f32x4;
    let total_len = rows * columns;
    assert!(total_len % 32 == 0);

    let mut bs = Vec::with_capacity(total_len / 32);
    (0..columns as isize).for_each(|c| unsafe {
        let ptr = data.offset(c * ld as isize);
        let ptr_ref = slice::from_raw_parts(ptr, rows);
        for i in (0..rows).step_by(32) {
            let mut vsrc = [f32x4::splat(0.0); 8];
            let mut vasrc = [f32x4::splat(0.0); 8];
            let mut vmax = [f32x4::splat(0.0); 8];

            for j in 0..8 {
                vsrc[j] = f32x4::from_slice(&ptr_ref[i + j * 4..]);
                vasrc[j] = vsrc[j].abs();
            }

            for j in 0..4 {
                vmax[2 * j] = vasrc[2 * j].simd_max(vasrc[2 * j + 1]);
            }
            for j in 0..2 {
                vmax[4 * j] = vmax[4 * j].simd_max(vmax[4 * j + 2]);
            }
            for j in 0..1 {
                vmax[8 * j] = vmax[8 * j].simd_max(vmax[8 * j + 4]);
            }
            let max = vmax[0].reduce_max();

            let d = max / 127.0;
            let vd = f32x4::splat(d);
            let mut qs = [0_i8; 32];

            for j in 0..8 {
                let v = vsrc[j] / vd;
                let vi: std::simd::i32x4 = v.cast();
                qs[4 * j] = vi[0] as i8;
                qs[4 * j + 1] = vi[1] as i8;
                qs[4 * j + 2] = vi[2] as i8;
                qs[4 * j + 3] = vi[3] as i8;
            }

            bs.push(Q8_0 {
                delta: f16::from_f32(d),
                quants: qs,
            });
        }
    });

    bs
}

fn gemm_q8_q8(
    a: *const Q8_0,
    b: *const Q8_0,
    c: *mut f32,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    a_ld: usize,
    b_ld: usize,
    c_ld: usize,
    a_stride: isize,
    b_stride: isize,
    c_stride: isize,
    alpha: f32,
    beta: f32,
) {
    (0..batch as isize).for_each(|i| unsafe {
        let a_ptr = a.offset(i * a_stride);
        let b_ptr = b.offset(i * b_stride);
        let b_vec = quantize_f32_q8_0(b_ptr as *const f32, b_ld as usize, k, n);
        let c_ptr = c.offset(i * c_stride);
        let a_ref = slice::from_raw_parts(a_ptr, m * k / 32);
        let b_ref = b_vec.as_slice();
        let c_ref = slice::from_raw_parts_mut(c_ptr, m * n);
        let work_len = c_ref.len() / 16;
        let chunk_len = 16;
        let _ = crossbeam::scope(|s| {
            c_ref
                .chunks_mut(work_len)
                .enumerate()
                .for_each(|(work_idx, work_buf)| {
                    s.spawn(move |_| {
                        work_buf.chunks_mut(chunk_len).enumerate().for_each(
                            |(chunk_idx, chunk_buf)| {
                                for (i, cval) in chunk_buf.iter_mut().enumerate() {
                                    let elem_idx = work_idx * work_len + chunk_idx * chunk_len + i;
                                    let am = elem_idx % m;
                                    let bn: usize = (elem_idx - am) / m;
                                    let sum = vec_dot_q8_0_q8_0_avx2(
                                        &a_ref[am * a_ld / 32 as usize
                                            ..am * a_ld / 32 as usize + k / 32],
                                        &b_ref[bn * k / 32 as usize..bn * k / 32 as usize + k / 32],
                                    );
                                    *cval = alpha * sum + beta * *cval;
                                }
                            },
                        );
                    });
                });
        });
        // for am in 0..m {
        //     for bn in 0..n {
        //         let mut sum = 0.0f32;
        //         // for block in 0..(k / 32) {
        //         //     // let a_block = &*a_ptr.add(block + am * a_ld as usize / 32);
        //         //     // let b_block = &b_vec[block + bn * k / 32 as usize];

        //         // }
        //         sum = vec_dot_q8_0_q8_0_avx2(
        //             &a_ref[am * a_ld / 32 as usize..am * a_ld / 32 as usize + k / 32],
        //             &b_ref[bn * k / 32 as usize..bn * k / 32 as usize + k / 32],
        //         );
        //         let temp_c_ptr = c_ptr.add(bn * c_ld as usize + am);
        //         *temp_c_ptr = alpha * sum + beta * *temp_c_ptr;
        //     }
        // }
    });
}

fn kongzhuan() {}

impl MatMul<Cpu> for Operator {}

impl crate::Operator for Operator {
    type Hardware = Cpu;
    type TopoNode = Cpu;
    type Args = Args<Cpu>;

    #[inline]
    fn new(_node: &Self::TopoNode) -> Self {
        Self
    }

    fn scheme(
        &mut self,
        _args: &Self::Args,
        _max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        Ok(0)
    }

    fn launch<QA>(
        &self,
        args: &Self::Args,
        _workspace: &mut [ByteOf<Self::Hardware>],
        _queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        let SchemeLayout {
            dt_a,
            dt_b,
            ab_swap,
            a_trans,
            b_trans,
            batch,
            m,
            n,
            k,
            c_stride,
            c_ld,
            a_stride,
            a_ld,
            b_stride,
            b_ld,
        } = args.layout()?;
        let &Args {
            c_base,
            beta,
            a_base,
            b_base,
            alpha,
            ..
        } = args;

        let c = c_base as usize;
        let [a, b] = if ab_swap {
            [b_base, a_base]
        } else {
            [a_base, b_base]
        }
        .map(|ptr| ptr as usize);
        let (lhs_cs, lhs_rs) = if a_trans { (1, a_ld) } else { (a_ld, 1) };
        let (rhs_cs, rhs_rs) = if b_trans { (1, b_ld) } else { (b_ld, 1) };

        macro_rules! gemm {
            ($ty:ty; $alpha:expr, $beta:expr) => {
                (0..batch as isize).for_each(|i| unsafe {
                    gemm::gemm(
                        m,
                        n,
                        k,
                        (c as *mut $ty).offset(i * c_stride),
                        c_ld,
                        1,
                        beta != 0.,
                        (a as *const $ty).offset(i * a_stride),
                        lhs_cs,
                        lhs_rs,
                        (b as *const $ty).offset(i * b_stride),
                        rhs_cs,
                        rhs_rs,
                        $beta,
                        $alpha,
                        false,
                        false,
                        false,
                        gemm::Parallelism::Rayon(0),
                    )
                })
            };
        }
        // let workspace_size = n * k / 32 * 34;
        // let mut workspace = Workspace::new(queue_alloc, workspace, workspace_size as _);

        use digit_layout::types as ty;
        use gemm::f16;
        use ggml_quants::types as qty;
        match (dt_a, dt_b) {
            (ty::F16, ty::F16) => gemm!(f16; f16::from_f32(alpha), f16::from_f32(beta)),
            (ty::F32, ty::F32) => gemm!(f32; alpha, beta),
            (ty::F64, ty::F64) => gemm!(f64; alpha as _, beta as _),
            (ty::F32, qty::Q8_0) => gemm_q8_q8(
                a as *const Q8_0,
                b as *const Q8_0,
                c as *mut f32,
                batch,
                m,
                n,
                k,
                a_ld as usize,
                b_ld as usize,
                c_ld as usize,
                a_stride as isize,
                b_stride as isize,
                c_stride as isize,
                alpha,
                beta,
            ),
            // (ty::F32, qty::Q8_0) => kongzhuan(),
            _ => Err(type_not_support(format!("Unsupported {dt_a},{dt_b}")))?,
        }
        Ok(())
    }
}
