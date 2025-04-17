use super::{args::SchemeLayout, Args, MatMul};
use crate::{common_cpu::Cpu, type_not_support, ByteOf, LaunchError, QueueAlloc, SchemeError, Workspace};

use core::slice;
use ggml_quants::Q8_0;
use half::f16;
use rayon::prelude::*;
use std::arch::x86_64::*;
use std::mem;
use std::simd::num::SimdFloat;
pub struct Operator;

pub fn quantize_f32_q8_0(data: *const f32, dst: *mut Q8_0 ,ld: usize, rows: usize, columns: usize) {
    //TODO:实现对任意维度的量化
    use std::simd::f32x4;
    let total_len = rows * columns;
    assert!(total_len % 32 == 0);

    let mut output_index = 0;
    (0..columns as isize).for_each(|c| unsafe {
        let ptr = data.offset(c * ld as isize);
        let ptr_ref = std::slice::from_raw_parts(ptr, rows);
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

            // Write directly to the output pointer
            *dst.add(output_index) = Q8_0 {
                delta: f16::from_f32(d),
                quants: qs,
            };
            output_index += 1;
        }
    });
}

unsafe fn hsum256_ps(v: __m256) -> f32 {
    let v128 = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
    let v64 = _mm_add_ps(v128, _mm_movehl_ps(v128, v128));
    let v32 = _mm_add_ss(v64, _mm_movehdup_ps(v64));
    _mm_cvtss_f32(v32)
}

fn gemm_q8_q8(
    a: *const Q8_0,
    b: *const Q8_0,
    c: *mut f32,
    workspace: *mut Q8_0,
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
    beta:f32
) {
    (0..batch as isize).for_each(|i| unsafe {
        let a_ptr = a.offset(i * a_stride);
        let b_ptr = b.offset(i * b_stride);
        quantize_f32_q8_0(b as *const f32,workspace, b_ld as usize, k, n);
        let c_ptr = c.offset(i * c_stride);
        for am in 0..m {
            for bn in 0..n {
                let mut sum = _mm256_setzero_ps();
                for block in 0..(k / 32) {
                    let a_block = &*a_ptr.add(block + am * a_ld as usize / 32);
                    let b_block = &*b_ptr.add(block + bn * k / 32 as usize);
                    let a_i8 = _mm256_loadu_si256(a_block.quants.as_ptr() as *const __m256i);
                    let a_i16_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_i8, 0));
                    let a_i16_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_i8, 1));
                    let b_i8 = _mm256_loadu_si256(b_block.quants.as_ptr() as *const __m256i);
                    let b_i16_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_i8, 0));
                    let b_i16_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_i8, 1));
                    let prod_lo = _mm256_madd_epi16(a_i16_lo, b_i16_lo);
                    let prod_hi = _mm256_madd_epi16(a_i16_hi, b_i16_hi);
                    let prod_lo_f = _mm256_cvtepi32_ps(prod_lo);
                    let prod_hi_f = _mm256_cvtepi32_ps(prod_hi);
                    let delta = a_block.delta.to_f32() * b_block.delta.to_f32();
                    let delta_vec = _mm256_set1_ps(delta);

                    let res_lo = _mm256_mul_ps(prod_lo_f, delta_vec);
                    let res_hi = _mm256_mul_ps(prod_hi_f, delta_vec);

                    // 累加到结果
                    sum = _mm256_add_ps(sum, res_lo);
                    sum = _mm256_add_ps(sum, res_hi);
                }
                let sum = hsum256_ps(sum);
                let temp_c_ptr = c_ptr.add(bn * c_ld as usize + am);
                *temp_c_ptr = alpha * sum + beta * *temp_c_ptr;
            }
        }
    });
}

fn kongzhuan(){}

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
        workspace: &mut [ByteOf<Self::Hardware>],
        queue_alloc: &QA,
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
        let workspace_size = n * k /32 * 34;
        let mut workspace = Workspace::new(queue_alloc, workspace, workspace_size as _);

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
                workspace.as_mut_ptr() as *mut Q8_0,
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
