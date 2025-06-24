use ggml_quants::Q8_0;
use half::f16;
use std::arch::asm;
use std::arch::x86_64::*;
#[inline(always)]
unsafe fn hsum_float_8(x: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(x, 1);
    let lo = _mm256_castps256_ps128(x);
    let sum = _mm_add_ps(lo, hi);
    let sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    let sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    _mm_cvtss_f32(sum)
}

pub struct qblasq8 {
    k: usize,
    a: *const Q8_0,
    lda: usize,
    b: *const Q8_0,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
    ith: usize,
    nth: usize,
}

impl qblasq8 {
    pub fn new(
        k: usize,
        a: *const Q8_0,
        lda: usize,
        b: *const Q8_0,
        ldb: usize,
        c: *mut f32,
        ldc: usize,
        ith: usize,
        nth: usize,
    ) -> Self {
        Self {
            k,
            a,
            lda,
            b,
            ldb,
            c,
            ldc,
            ith,
            nth,
        }
    }

    pub fn gemm(&mut self, m: usize, n: usize) {
        self.mnpack(0, m, 0, n);
    }

    fn mnpack(&mut self, m0: usize, m: usize, n0: usize, n: usize) {
        if m <= m0 || n <= n0 {
            return;
        }
        let mc;
        let nc;
        if m - m0 >= 4 && n - n0 >= 1 {
            mc = 4;
            nc = 1;
            unsafe {
                self.gemm4x1(m0, m, n0, n);
            }
        } else if m - m0 >= 1 && n - n0 >= 4 {
            mc = 1;
            nc = 4;
            unsafe {
                self.gemm1x4(m0, m, n0, n);
            }
        } else {
            mc = 1;
            nc = 1;
            unsafe {
                self.gemm1x1(m0, m, n0, n);
            }
        }
        let mp = m0 + (m - m0) / mc * mc;
        let np = n0 + (n - n0) / nc * nc;
        self.mnpack(mp, m, n0, np);
        self.mnpack(m0, mp, np, n);
        self.mnpack(mp, m, np, n);
    }

    #[inline(never)]
    unsafe fn gemm1x4(&mut self, m0: usize, m: usize, n0: usize, n: usize) {
        let RM = 1;
        let RN = 4;
        let ytiles = (m - m0) / RM;
        let xtiles = (n - n0) / RN;
        let tiles = ytiles * xtiles;
        let duty = (tiles as f64) / (self.nth as f64).max(1.0);
        let spot = duty * self.ith as f64 + 0.5;
        let start = spot as usize;
        let mut end = (spot + duty) as usize;
        if end > tiles {
            end = tiles;
        }

        for job in start..end {
            let i = m0 + (job / xtiles) * RM;
            let j = n0 + (job % xtiles) * RN;

            let mut c = [_mm256_setzero_ps(); 4];
            let k2 = self.k / 32;

            let bp0 = self.b.add(self.ldb / 32 * (j + 0));
            let bp1 = self.b.add(self.ldb / 32 * (j + 1));
            let bp2 = self.b.add(self.ldb / 32 * (j + 2));
            let bp3 = self.b.add(self.ldb / 32 * (j + 3));
            let ap = self.a.add(self.lda / 32 * i);

            for l in 0..k2 {
                let da0 = f16::to_f32((*ap.add(l)).delta);
                let f = _mm256_loadu_si256(ap.add(l) as *const __m256i);
                let u = _mm256_sign_epi8(f, f);
                for (idx, bp) in [bp0, bp1, bp2, bp3].iter().enumerate() {
                    let db = f16::to_f32((*bp.add(l)).delta);
                    let d = _mm256_set1_ps(db * da0);
                    let e = _mm256_loadu_si256(bp.add(l) as *const __m256i);
                    let s = _mm256_sign_epi8(e, f);
                    let g = mul_sum_signed_i8_pairs_float(u, s);
                    c[idx] = _mm256_fmadd_ps(d, g, c[idx]);
                }
            }

            for s in 0..4 {
                *self.c.add(self.ldc * (j + s) + i) = hsum_float_8(c[s]);
            }
        }
    }
    #[inline(never)]
    unsafe fn gemm4x1(&mut self, m0: usize, m: usize, n0: usize, n: usize) {
        let RM = 4;
        let RN = 1;
        let ytiles = (m - m0) / RM;
        let xtiles = (n - n0) / RN;
        let tiles = ytiles * xtiles;
        let duty = (tiles as f64) / (self.nth as f64).max(1.0);
        let spot = duty * self.ith as f64 + 0.5;
        let start = spot as usize;
        let mut end = (spot + duty) as usize;
        if end > tiles {
            end = tiles;
        }

        for job in start..end {
            let i = m0 + (job / xtiles) * RM;
            let j = n0 + (job % xtiles) * RN;

            let mut c = [_mm256_setzero_ps(); 4];
            let k2 = self.k / 32;

            let ap0 = self.a.add(self.lda / 32 * (i + 0));
            let ap1 = self.a.add(self.lda / 32 * (i + 1));
            let ap2 = self.a.add(self.lda / 32 * (i + 2));
            let ap3 = self.a.add(self.lda / 32 * (i + 3));
            let bp = self.b.add(self.ldb / 32 * j);

            for l in 0..k2 {
                let db0 = f16::to_f32((*bp.add(l)).delta);
                let f = _mm256_loadu_si256(bp.add(l) as *const __m256i);
                let u = _mm256_sign_epi8(f, f);

                for (idx, ap) in [ap0, ap1, ap2, ap3].iter().enumerate() {
                    let da = f16::to_f32((*ap.add(l)).delta);
                    let d = _mm256_set1_ps(da * db0);
                    let e = _mm256_loadu_si256(ap.add(l) as *const __m256i);
                    let s = _mm256_sign_epi8(e, f);
                    let g = mul_sum_signed_i8_pairs_float(u, s);
                    c[idx] = _mm256_fmadd_ps(d, g, c[idx]);
                }
            }

            for r in 0..4 {
                *self.c.add(self.ldc * j + (i + r)) = hsum_float_8(c[r]);
            }
        }
    }
    #[inline(never)]
    unsafe fn gemm1x1(&mut self, m0: usize, m: usize, n0: usize, n: usize) {
        let RM = 1;
        let RN = 1;
        let ytiles = (m - m0) / RM;
        let xtiles = (n - n0) / RN;
        let tiles = ytiles * xtiles;
        let duty = (tiles as f64) / (self.nth as f64).max(1.0);
        let spot = duty * self.ith as f64 + 0.5;
        let start = spot as usize;
        let mut end = (spot + duty) as usize;
        if end > tiles {
            end = tiles;
        }

        for job in start..end {
            let i = m0 + (job / xtiles) * RM;
            let j = n0 + (job % xtiles) * RN;

            let mut c = _mm256_setzero_ps();
            let k2 = self.k / 32;

            let ap = self.a.add(self.lda / 32 * i);
            let bp = self.b.add(self.ldb / 32 * j);

            for l in 0..k2 {
                let d = _mm256_set1_ps(
                    f16::to_f32((*ap.add(l)).delta) * f16::to_f32((*bp.add(l)).delta),
                );
                let e = _mm256_loadu_si256(ap.add(l) as *const __m256i);
                let f = _mm256_loadu_si256(bp.add(l) as *const __m256i);
                let g = mul_sum_i8_pairs_float(e, f);
                c = _mm256_fmadd_ps(d, g, c);
            }

            *self.c.add(self.ldc * j + i) = hsum_float_8(c);
        }
    }
}

/// Rust版 VNNI dot-product for int8 pairs
#[inline(always)]
pub unsafe fn vpdpbusd(x: __m256i, y: __m256i, s: __m256i) -> __m256i {
    // maddubs: pairs of u8 x s8 => i16
    let prod_16 = _mm256_maddubs_epi16(x, y);
    // madd: pairs of i16 => i32 (sum pairs)
    let prod_32 = _mm256_madd_epi16(prod_16, _mm256_set1_epi16(1));
    // accumulate
    _mm256_add_epi32(prod_32, s)
}

/// 等价于 mul_sum_i8_pairs_float
#[inline(always)]
pub unsafe fn mul_sum_i8_pairs_float(x: __m256i, y: __m256i) -> __m256 {
    _mm256_cvtepi32_ps(vpdpbusd(
        _mm256_sign_epi8(x, x), // x unsigned
        _mm256_sign_epi8(y, x), // y signed
        _mm256_setzero_si256(),
    ))
}

/// 等价于 mul_sum_signed_i8_pairs_float
#[inline(always)]
pub unsafe fn mul_sum_signed_i8_pairs_float(u: __m256i, s: __m256i) -> __m256 {
    _mm256_cvtepi32_ps(vpdpbusd(u, s, _mm256_setzero_si256()))
}
