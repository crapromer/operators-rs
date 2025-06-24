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

pub struct qblasf32 {
    k: usize,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
    ith: usize,
    nth: usize,
}

impl qblasf32 {
    pub fn new(
        k: usize,
        a: *const f32,
        lda: usize,
        b: *const f32,
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
        if m - m0 >= 3 && n - n0 >= 4 {
            mc = 3;
            nc = 4;
            unsafe {
                self.gemm3x4(m0, m, n0, n);
            }
        } else if m - m0 >= 4 && n - n0 >= 1 {
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
    unsafe fn gemm3x4(&mut self, m0: usize, m: usize, n0: usize, n: usize) {
        let RM = 3;
        let RN = 4;
        let ytiles = (m - m0) / RM;
        let xtiles = (n - n0) / RN;
        let tiles = ytiles * xtiles;
        let mut duty = (tiles as f64) / (self.nth as f64);
        if duty < 1.0 {
            duty = 1.0;
        }
        let spot = duty * (self.ith as f64) + 0.5;
        let mut end = (spot + duty) as usize;
        let start = spot as usize;
        if end > tiles {
            end = tiles;
        }

        for job in start..end {
            let i = m0 + (job / xtiles) * RM;
            let j = n0 + (job % xtiles) * RN;

            let mut c = [[_mm256_setzero_ps(); 4]; 3];
            for l in (0..self.k).step_by(8) {
                let k0 = _mm256_loadu_ps(self.b.add(self.ldb * (j + 0) + l));
                let k1 = _mm256_loadu_ps(self.b.add(self.ldb * (j + 1) + l));
                let k2 = _mm256_loadu_ps(self.b.add(self.ldb * (j + 2) + l));
                let k3 = _mm256_loadu_ps(self.b.add(self.ldb * (j + 3) + l));

                let a0 = _mm256_loadu_ps(self.a.add(self.lda * (i + 0) + l));
                c[0][0] = _mm256_fmadd_ps(a0, k0, c[0][0]);
                c[0][1] = _mm256_fmadd_ps(a0, k1, c[0][1]);
                c[0][2] = _mm256_fmadd_ps(a0, k2, c[0][2]);
                c[0][3] = _mm256_fmadd_ps(a0, k3, c[0][3]);

                let a1 = _mm256_loadu_ps(self.a.add(self.lda * (i + 1) + l));
                c[1][0] = _mm256_fmadd_ps(a1, k0, c[1][0]);
                c[1][1] = _mm256_fmadd_ps(a1, k1, c[1][1]);
                c[1][2] = _mm256_fmadd_ps(a1, k2, c[1][2]);
                c[1][3] = _mm256_fmadd_ps(a1, k3, c[1][3]);

                let a2 = _mm256_loadu_ps(self.a.add(self.lda * (i + 2) + l));
                c[2][0] = _mm256_fmadd_ps(a2, k0, c[2][0]);
                c[2][1] = _mm256_fmadd_ps(a2, k1, c[2][1]);
                c[2][2] = _mm256_fmadd_ps(a2, k2, c[2][2]);
                c[2][3] = _mm256_fmadd_ps(a2, k3, c[2][3]);
            }
            for r in 0..3 {
                for s in 0..4 {
                    *self.c.add(self.ldc * (j + s) + (i + r)) = hsum_float_8(c[r][s]);
                }
            }
        }
    }
    #[inline(never)]
    unsafe fn gemm1x4(&mut self, m0: usize, m: usize, n0: usize, n: usize) {
        let RM = 1;
        let RN = 4;
        let ytiles = (m - m0) / RM;
        let xtiles = (n - n0) / RN;
        let tiles = ytiles * xtiles;
        let mut duty = (tiles as f64) / (self.nth as f64);
        if duty < 1.0 {
            duty = 1.0;
        }
        let spot = duty * (self.ith as f64) + 0.5;
        let mut end = (spot + duty) as usize;
        let start = spot as usize;
        if end > tiles {
            end = tiles;
        }

        for job in start..end {
            let i = m0 + (job / xtiles) * RM;
            let j = n0 + (job % xtiles) * RN;

            let mut c = [_mm256_setzero_ps(); 4];
            for l in (0..self.k).step_by(8) {
                let a0 = _mm256_loadu_ps(self.a.add(self.lda * i + l));
                let k0 = _mm256_loadu_ps(self.b.add(self.ldb * (j + 0) + l));
                let k1 = _mm256_loadu_ps(self.b.add(self.ldb * (j + 1) + l));
                let k2 = _mm256_loadu_ps(self.b.add(self.ldb * (j + 2) + l));
                let k3 = _mm256_loadu_ps(self.b.add(self.ldb * (j + 3) + l));
                c[0] = _mm256_fmadd_ps(a0, k0, c[0]);
                c[1] = _mm256_fmadd_ps(a0, k1, c[1]);
                c[2] = _mm256_fmadd_ps(a0, k2, c[2]);
                c[3] = _mm256_fmadd_ps(a0, k3, c[3]);
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
        let mut duty = (tiles as f64) / (self.nth as f64);
        if duty < 1.0 {
            duty = 1.0;
        }
        let spot = duty * (self.ith as f64) + 0.5;
        let mut end = (spot + duty) as usize;
        let start = spot as usize;
        if end > tiles {
            end = tiles;
        }

        for job in start..end {
            let i = m0 + (job / xtiles) * RM;
            let j = n0 + (job % xtiles) * RN;

            let mut c = [_mm256_setzero_ps(); 4];
            for l in (0..self.k).step_by(8) {
                let k0 = _mm256_loadu_ps(self.b.add(self.ldb * j + l));
                let a0 = _mm256_loadu_ps(self.a.add(self.lda * (i + 0) + l));
                let a1 = _mm256_loadu_ps(self.a.add(self.lda * (i + 1) + l));
                let a2 = _mm256_loadu_ps(self.a.add(self.lda * (i + 2) + l));
                let a3 = _mm256_loadu_ps(self.a.add(self.lda * (i + 3) + l));
                c[0] = _mm256_fmadd_ps(a0, k0, c[0]);
                c[1] = _mm256_fmadd_ps(a1, k0, c[1]);
                c[2] = _mm256_fmadd_ps(a2, k0, c[2]);
                c[3] = _mm256_fmadd_ps(a3, k0, c[3]);
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
        let mut duty = (tiles as f64) / (self.nth as f64);
        if duty < 1.0 {
            duty = 1.0;
        }
        let spot = duty * (self.ith as f64) + 0.5;
        let mut end = (spot + duty) as usize;
        let start = spot as usize;
        if end > tiles {
            end = tiles;
        }

        for job in start..end {
            let i = m0 + (job / xtiles) * RM;
            let j = n0 + (job % xtiles) * RN;

            let mut c = _mm256_setzero_ps();
            for l in (0..self.k).step_by(8) {
                let a = _mm256_loadu_ps(self.a.add(self.lda * i + l));
                let b = _mm256_loadu_ps(self.b.add(self.ldb * j + l));
                c = _mm256_fmadd_ps(a, b, c);
            }
            *self.c.add(self.ldc * j + i) = hsum_float_8(c);
        }
    }
}
