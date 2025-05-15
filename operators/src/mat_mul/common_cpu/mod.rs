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
use std::{cmp, mem};
use std::simd::num::SimdFloat;
use std::sync::atomic::{AtomicUsize, Ordering};
pub struct Operator;
#[inline(always)]
pub unsafe fn sum_i16_pairs_float(x: __m256i) -> __m256 {
    let ones = _mm256_set1_epi16(1);
    let summed_pairs = _mm256_madd_epi16(ones, x);
    _mm256_cvtepi32_ps(summed_pairs)
}
#[inline(always)]
pub unsafe fn mul_sum_us8_pairs_float(ax: __m256i, sy: __m256i) -> __m256 {
    let dot = _mm256_maddubs_epi16(ax, sy);
    sum_i16_pairs_float(dot)
}
#[inline(always)]
pub unsafe fn hsum_float_8(x: __m256) -> f32 {
    let res = _mm256_extractf128_ps(x, 1);
    let res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    let res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    let res = _mm_add_ss(res, _mm_movehdup_ps(res));
    _mm_cvtss_f32(res)
}
#[inline(always)]
pub unsafe fn mul_sum_i8_pairs_float(x: __m256i, y: __m256i) -> __m256 {
    let ax = _mm256_sign_epi8(x, x);
    let sy = _mm256_sign_epi8(y, x);
    mul_sum_us8_pairs_float(ax, sy)
}
#[inline(always)]
pub fn vec_dot_q8_0_q8_0(xs: *const Q8_0, ys: *const Q8_0, n: usize) -> f32 {
    use std::arch::x86_64::*;

    unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        let mut i = 0;
        while i + 3 < n {
            let x0 = xs.add(i);
            let y0 = ys.add(i);
            let d0 = _mm256_set1_ps(f16::to_f32((*x0).delta) * f16::to_f32((*y0).delta));
            let qx0 = _mm256_loadu_si256((*x0).quants.as_ptr() as *const __m256i);
            let qy0 = _mm256_loadu_si256((*y0).quants.as_ptr() as *const __m256i);
            let q0 = mul_sum_i8_pairs_float(qx0, qy0);
            acc0 = _mm256_fmadd_ps(d0, q0, acc0);

            let x1 = xs.add(i + 1);
            let y1 = ys.add(i + 1);
            let d1 = _mm256_set1_ps(f16::to_f32((*x1).delta) * f16::to_f32((*y1).delta));
            let qx1 = _mm256_loadu_si256((*x1).quants.as_ptr() as *const __m256i);
            let qy1 = _mm256_loadu_si256((*y1).quants.as_ptr() as *const __m256i);
            let q1 = mul_sum_i8_pairs_float(qx1, qy1);
            acc1 = _mm256_fmadd_ps(d1, q1, acc1);

            let x2 = xs.add(i + 2);
            let y2 = ys.add(i + 2);
            let d2 = _mm256_set1_ps(f16::to_f32((*x2).delta) * f16::to_f32((*y2).delta));
            let qx2 = _mm256_loadu_si256((*x2).quants.as_ptr() as *const __m256i);
            let qy2 = _mm256_loadu_si256((*y2).quants.as_ptr() as *const __m256i);
            let q2 = mul_sum_i8_pairs_float(qx2, qy2);
            acc2 = _mm256_fmadd_ps(d2, q2, acc2);

            let x3 = xs.add(i + 3);
            let y3 = ys.add(i + 3);
            let d3 = _mm256_set1_ps(f16::to_f32((*x3).delta) * f16::to_f32((*y3).delta));
            let qx3 = _mm256_loadu_si256((*x3).quants.as_ptr() as *const __m256i);
            let qy3 = _mm256_loadu_si256((*y3).quants.as_ptr() as *const __m256i);
            let q3 = mul_sum_i8_pairs_float(qx3, qy3);
            acc3 = _mm256_fmadd_ps(d3, q3, acc3);

            i += 4;
        }

        // 处理尾部不满4的元素
        while i < n {
            let x = xs.add(i);
            let y = ys.add(i);
            let d = _mm256_set1_ps(f16::to_f32((*x).delta) * f16::to_f32((*y).delta));
            let qx = _mm256_loadu_si256((*x).quants.as_ptr() as *const __m256i);
            let qy = _mm256_loadu_si256((*y).quants.as_ptr() as *const __m256i);
            let q = mul_sum_i8_pairs_float(qx, qy);
            acc0 = _mm256_fmadd_ps(d, q, acc0);
            i += 1;
        }

        // 合并4个 accumulator 并做水平求和
        let acc01 = _mm256_add_ps(acc0, acc1);
        let acc23 = _mm256_add_ps(acc2, acc3);
        let acc = _mm256_add_ps(acc01, acc23);
        hsum_float_8(acc)
    }
}
#[inline(always)]
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

#[inline(always)]
unsafe fn vec_dot_f32_f32(
    a_ptr: *const f32,
    b_ptr: *const f32,
    k: usize,
) -> f32 {
    let mut sumv = _mm256_setzero_ps();
    let k_rounded = k - k % 8;

    for ki in (0..k_rounded).step_by(8) {
        let av = _mm256_loadu_ps(a_ptr.add(ki));
        let bv = _mm256_loadu_ps(b_ptr.add(ki));
        sumv = _mm256_fmadd_ps(av, bv, sumv);
    }

    let mut sum_arr = [0.0f32; 8];
    _mm256_storeu_ps(sum_arr.as_mut_ptr(), sumv);
    let mut total = sum_arr.iter().sum::<f32>();

    for ki in k_rounded..k {
        total += *a_ptr.add(ki) * *b_ptr.add(ki);
    }

    total
}
#[inline(always)]
unsafe fn vec_dot_f32_f32_simd(
    a_ptr: *const f32,
    a_stride: usize,
    b_ptr: *const f32,
    k: usize,
) -> f32 {
    let mut sumv = _mm256_setzero_ps();
    let k_rounded = k - k % 8;
    for ki in (0..k_rounded).step_by(8) {
        let mut a_chunk = [0.0f32; 8];

        for i in 0..8 {
            a_chunk[i] = *a_ptr.add((ki + i) * a_stride);
        }

        let av = _mm256_loadu_ps(a_chunk.as_ptr());
        let bv = _mm256_loadu_ps(b_ptr.add(ki));
        sumv = _mm256_fmadd_ps(av, bv, sumv);
    }

    let mut sum_arr = [0.0f32; 8];
    _mm256_storeu_ps(sum_arr.as_mut_ptr(), sumv);
    let mut total = sum_arr.iter().sum::<f32>();

    for ki in k_rounded..k {
        total += *a_ptr.add(ki * a_stride) * *b_ptr.add(ki);
    }

    total
    
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
#[inline(always)]
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
        let a_ptr = a.offset(i * a_stride) as usize;
        let b_ptr = b.offset(i * b_stride) as usize;
        let b_vec = quantize_f32_q8_0(b_ptr as *const f32, b_ld as usize, k, n);
        let b_quant = b_vec.as_ptr() as usize;
        let c_ptr = c.offset(i * c_stride) as usize;


        // let a_ref = slice::from_raw_parts(a_ptr, m * k / 32);
        // let b_ref = b_vec.as_slice();
        // let thread_num = 20;
        // let work_len = m * n / thread_num;
        // let chunk_len = 16;
        // let chunk_num = work_len / chunk_len;
        // (0..thread_num).into_par_iter().for_each(|tid| {
        //     (0..chunk_num).into_par_iter().for_each(|cid| {
        //         (0..chunk_len).into_par_iter().for_each(|i| {
        //             let elem_idx = tid * work_len + cid * chunk_len + i;
        //             let am = elem_idx % m;
        //             let bn = (elem_idx - am) / m;
        //             let sum = vec_dot_q8_0_q8_0_avx2(
        //                 &a_ref[am * a_ld / 32 as usize..am * a_ld / 32 as usize + k / 32],
        //                 &b_ref[bn * k / 32 as usize..bn * k / 32 as usize + k / 32],
        //             );
        //             let temp_c = (c_ptr as *mut f32).add(elem_idx);
        //             *temp_c = alpha * sum + beta * *temp_c;
        //         });
        //     });
        // });


        let thread_num = 32;
        let chunk_size = 32;
        let block_size = 16;
        let m_chunk = (m + chunk_size -1)/chunk_size;
        let n_chunk = (n + chunk_size -1)/chunk_size;
        let m_chunk_elem = (m + m_chunk - 1) / m_chunk;
        let n_chunk_elem = (n + n_chunk - 1) / n_chunk;
        let current_chunk = AtomicUsize::new(0);
        // println!("{},{}===============",m_chunk,n_chunk);
        (0..thread_num).into_par_iter().for_each(|tid|{
            let mut chunk_id = tid;
            loop {
                if chunk_id >= m_chunk * n_chunk {
                    break;
                }
                let ith0 = chunk_id % m_chunk;
                let ith1 = chunk_id / m_chunk;
                let ir0_start = m_chunk_elem * ith0;
                let ir1_start = n_chunk_elem * ith1;
                let ir0_end = cmp::min(ir0_start + m_chunk_elem, m);
                let ir1_end = cmp::min(ir1_start + n_chunk_elem, n);
                // (ir0_start..ir0_end).for_each(|iir0|{
                //     (ir1_start..ir1_end).for_each(|iir1|{
                //         let a = (a_ptr + std::mem::size_of::<Q8_0>() * (iir0 * a_ld/32)) as *const Q8_0;
                //         let b = (b_quant + std::mem::size_of::<Q8_0>() * (iir1 * k/32)) as *const Q8_0;
                //         let c =
                //             (c_ptr + std::mem::size_of::<f32>() * (iir1 * c_ld  + iir0)) as *mut f32;
                //         let sum = vec_dot_q8_0_q8_0(a,b,k/32);
                //         // let sum = vec_dot_q8_0_q8_0_avx2(slice::from_raw_parts(a, k/32),slice::from_raw_parts(b, k/32));
                //         *c = alpha * sum + beta * *c;
                //     });
                // });

                (ir1_start..ir1_end).step_by(block_size).for_each(|iir1|{
                    (ir0_start..ir0_end).step_by(block_size).for_each(|iir0|{
                        (iir1..cmp::min(iir1+block_size,n)).for_each(|bn|{
                            (iir0..cmp::min(iir0+block_size,m)).for_each(|am|{
                                let a = (a_ptr + std::mem::size_of::<Q8_0>() * (am * a_ld/32)) as *const Q8_0;
                                let b = (b_quant + std::mem::size_of::<Q8_0>() * (bn * k/32)) as *const Q8_0;
                                let c =
                                    (c_ptr + std::mem::size_of::<f32>() * (bn * c_ld  + am)) as *mut f32;
                                let sum = vec_dot_q8_0_q8_0(a,b,k/32);
                                *c = alpha * sum + beta * *c;
                            });
                        });

                        
                    });
                });
                chunk_id = current_chunk.fetch_add(1, Ordering::Relaxed);
            }
        });
    });
}

fn gemm_f32_f32(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs_cs: usize,
    lhs_rs: usize,
    rhs_cs: usize,
    rhs_rs: usize,
    dst_cs: usize,
    dst_rs: usize,
    a_stride: isize,
    b_stride: isize,
    c_stride: isize,
    alpha: f32,
    beta: f32,
) {
    (0..batch as isize).for_each(|i| unsafe {
        let a_ptr = a.offset(i * a_stride) as usize;
        let b_ptr = b.offset(i * b_stride) as usize;
        let c_ptr = c.offset(i * c_stride) as usize;
        // let thread_num = 20;
        // let work_len = m * n / thread_num;
        // let chunk_len = 32;
        // let chunk_num = work_len / chunk_len;
        // (0..thread_num).into_par_iter().for_each(|tid| {
        //     (0..chunk_num).into_par_iter().for_each(|cid| {
        //         (0..chunk_len).into_par_iter().for_each(|i| {
        //             let elem_idx = tid * work_len + cid * chunk_len + i;
        //             let am = elem_idx % m;
        //             let bn = (elem_idx - am) / m;
        //             let a = (a_ptr + std::mem::size_of::<f32>() * (am * lhs_rs)) as *const f32;
        //             let b = (b_ptr + std::mem::size_of::<f32>() * (bn * rhs_cs)) as *const f32;
        //             let c =
        //                 (c_ptr + std::mem::size_of::<f32>() * (bn * dst_cs + am * dst_rs)) as *mut f32;
        //             let sum = vec_dot_f32_f32_simd(a, lhs_cs, b, rhs_rs, k);
        //             *c = alpha * sum + beta * *c;
        //         });
        //     });
        // });
        // (0..n).into_par_iter().for_each(|bn| {
        //     (0..m).into_par_iter().for_each(|am| {
        //         let a = (a_ptr + std::mem::size_of::<f32>() * (am * lhs_rs)) as *const f32;
        //         let b = (b_ptr + std::mem::size_of::<f32>() * (bn * rhs_cs)) as *const f32;
        //         let c =
        //             (c_ptr + std::mem::size_of::<f32>() * (bn * dst_cs + am * dst_rs)) as *mut f32;
        //         let sum = vec_dot_f32_f32_simd(a, lhs_cs, b, k);
        //         *c = alpha * sum + beta * *c;
        //     });
        // });

        let thread_num = 32;
        let chunk_size = 32;
        let block_size = 16;
        let m_chunk = (m + chunk_size -1)/chunk_size;
        let n_chunk = (n + chunk_size -1)/chunk_size;
        let m_chunk_elem = (m + m_chunk - 1) / m_chunk;
        let n_chunk_elem = (n + n_chunk - 1) / n_chunk;
        let current_chunk = AtomicUsize::new(0);
        (0..thread_num).into_par_iter().for_each(|tid|{
            let mut chunk_id = tid;
            loop {
                if(chunk_id >= m_chunk * n_chunk){
                    break;
                }
                let ith0 = chunk_id % m_chunk;
                let ith1 = chunk_id / m_chunk;
                let ir0_start = m_chunk_elem * ith0;
                let ir1_start = n_chunk_elem * ith1;
                let ir0_end = cmp::min(ir0_start + m_chunk_elem, m);
                let ir1_end = cmp::min(ir1_start + n_chunk_elem, n);
                (ir0_start..ir0_end).step_by(block_size).for_each(|iir0|{
                    (ir1_start..ir1_end).step_by(block_size).for_each(|iir1|{
                        (iir0..cmp::min(iir0+block_size,m)).for_each(|am|{
                            (iir1..cmp::min(iir1+block_size,n)).for_each(|bn|{
                                let a = (a_ptr + std::mem::size_of::<f32>() * (am * lhs_rs)) as *const f32;
                                let b = (b_ptr + std::mem::size_of::<f32>() * (bn * rhs_cs)) as *const f32;
                                let c =
                                    (c_ptr + std::mem::size_of::<f32>() * (bn * dst_cs + am * dst_rs)) as *mut f32;
                                let sum = vec_dot_f32_f32_simd(a, lhs_cs, b, k);
                                *c = alpha * sum + beta * *c;
                            });
                        });

                        
                    });
                });
                chunk_id = current_chunk.fetch_add(1, Ordering::Relaxed);
            }
        });
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
                        gemm::Parallelism::Rayon(20),
                    )
                })
            };
        }
        // let workspace_size = n * k / 32 * 34;
        // let mut workspace = Workspace::new(queue_alloc, workspace, workspace_size as _);
        use std::time::Instant;
        let start = Instant::now();

        use digit_layout::types as ty;
        use gemm::f16;
        use ggml_quants::types as qty;
        match (dt_a, dt_b) {
            (ty::F16, ty::F16) => gemm!(f16; f16::from_f32(alpha), f16::from_f32(beta)),
            // (ty::F32, ty::F32) => unsafe {
            //     matrixmultiply::sgemm(
            //         m,
            //         k,
            //         n,
            //         alpha,
            //         a as *const f32,
            //         lhs_rs,
            //         lhs_cs,
            //         b as *const f32,
            //         rhs_rs,
            //         rhs_cs,
            //         beta,
            //         c as *mut f32,
            //         c_ld,
            //         1,
            //     )
            // },
            // (ty::F32, ty::F32) => gemm!(f32; alpha, beta),
            // (ty::F32, ty::F32) => kongzhuan(),
            (ty::F32, ty::F32) => gemm_f32_f32(
                a as *const f32,
                b as *const f32,
                c as *mut f32,
                batch,
                m,
                n,
                k,
                lhs_cs as usize,
                lhs_rs as usize,
                rhs_cs as usize,
                rhs_rs as usize,
                c_ld as usize,
                1 as usize,
                a_stride as isize,
                b_stride as isize,
                c_stride as isize,
                alpha,
                beta,
            ),
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
        let duration = start.elapsed().as_micros();
        // println!(
        //     "{}*{} Mat a:(({},{}),({},{})),Mat b:(({},{}),({},{})), Mat c:(({},{}),({},{})) Duration:{:?}/{}",
        //     dt_a.group_size(),dt_b.group_size(),m, k, lhs_rs, lhs_cs, k, n, rhs_rs, rhs_cs, m, n, 1, c_ld, duration,m*n*k/duration as usize
        // );
        Ok(())
    }
}
