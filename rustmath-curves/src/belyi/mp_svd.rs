//! High-precision complex one-sided Jacobi SVD (A = U Σ V*), in rug/MPFR.
//!
//! One-sided Jacobi is slower than bidiagonal SVD but far easier to make reliable in
//! arbitrary precision, and it delivers high *relative* accuracy for small singular
//! values and singular vectors (Drmač–Veselić; LAPACK xGEJSV philosophy). This is
//! exactly what the §4 modular-forms computation needs: the small-σ right-singular
//! subspace of A−I is the space S_k(Γ) of modular forms, which Gauss–Jordan pivots
//! could not resolve (they floor at ~10⁻⁴ while the true small σ ~10⁻²⁹).
//!
//! Complex scalars are stored as two `rug::Float`s (`MpC`) rather than `rug::Complex`,
//! which keeps Jacobi rotations allocation-light and exposes real-valued norms
//! directly. Adapted from a design note; degenerate/clustered singular values are
//! reported (individual vectors are then not canonical — use the subspace).

use rug::Float;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SvdError {
    DimensionMismatch,
    WideMatrixUnsupported,
    EmptyMatrix,
    InvalidTolerance,
}

/// A high-precision complex number stored as (re, im).
#[derive(Clone, Debug)]
pub struct MpC {
    pub re: Float,
    pub im: Float,
}

impl MpC {
    pub fn zero(prec: u32) -> Self {
        Self { re: Float::with_val(prec, 0), im: Float::with_val(prec, 0) }
    }
    pub fn one(prec: u32) -> Self {
        Self { re: Float::with_val(prec, 1), im: Float::with_val(prec, 0) }
    }
    pub fn new(re: Float, im: Float) -> Self {
        Self { re, im }
    }
    pub fn from_f64(prec: u32, re: f64, im: f64) -> Self {
        Self { re: Float::with_val(prec, re), im: Float::with_val(prec, im) }
    }
    pub fn conj(&self) -> Self {
        Self { re: self.re.clone(), im: -self.im.clone() }
    }
    pub fn add(&self, rhs: &Self) -> Self {
        Self { re: self.re.clone() + &rhs.re, im: self.im.clone() + &rhs.im }
    }
    pub fn sub(&self, rhs: &Self) -> Self {
        Self { re: self.re.clone() - &rhs.re, im: self.im.clone() - &rhs.im }
    }
    pub fn mul(&self, rhs: &Self) -> Self {
        let re = self.re.clone() * &rhs.re - self.im.clone() * &rhs.im;
        let im = self.re.clone() * &rhs.im + self.im.clone() * &rhs.re;
        Self { re, im }
    }
    /// conj(self) * rhs.
    pub fn conj_mul(&self, rhs: &Self) -> Self {
        let re = self.re.clone() * &rhs.re + self.im.clone() * &rhs.im;
        let im = self.re.clone() * &rhs.im - self.im.clone() * &rhs.re;
        Self { re, im }
    }
    pub fn scale(&self, a: &Float) -> Self {
        Self { re: self.re.clone() * a, im: self.im.clone() * a }
    }
    pub fn div_real(&self, a: &Float) -> Self {
        Self { re: self.re.clone() / a, im: self.im.clone() / a }
    }
    pub fn abs2(&self) -> Float {
        self.re.clone() * &self.re + self.im.clone() * &self.im
    }
    pub fn abs(&self, prec: u32) -> Float {
        let mut x = Float::with_val(prec, self.abs2());
        x.sqrt_mut();
        x
    }
}

/// A dense complex matrix, row-major, at fixed precision.
#[derive(Clone, Debug)]
pub struct MpMatrix {
    pub rows: usize,
    pub cols: usize,
    pub prec: u32,
    pub data: Vec<MpC>,
}

impl MpMatrix {
    pub fn zeros(rows: usize, cols: usize, prec: u32) -> Self {
        Self { rows, cols, prec, data: vec![MpC::zero(prec); rows * cols] }
    }
    pub fn identity(n: usize, prec: u32) -> Self {
        let mut m = Self::zeros(n, n, prec);
        for i in 0..n {
            m.set(i, i, MpC::one(prec));
        }
        m
    }
    pub fn from_row_major(rows: usize, cols: usize, prec: u32, data: Vec<MpC>) -> Result<Self, SvdError> {
        if rows == 0 || cols == 0 {
            return Err(SvdError::EmptyMatrix);
        }
        if data.len() != rows * cols {
            return Err(SvdError::DimensionMismatch);
        }
        Ok(Self { rows, cols, prec, data })
    }
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> &MpC {
        &self.data[i * self.cols + j]
    }
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, z: MpC) {
        self.data[i * self.cols + j] = z;
    }
    pub fn col_norm2(&self, j: usize) -> Float {
        let mut acc = Float::with_val(self.prec, 0);
        for i in 0..self.rows {
            acc += self.get(i, j).abs2();
        }
        acc
    }
    /// ⟨col p, col q⟩ = Σ_i conj(A_ip) A_iq.
    pub fn col_dot(&self, p: usize, q: usize) -> MpC {
        let mut acc = MpC::zero(self.prec);
        for i in 0..self.rows {
            acc = acc.add(&self.get(i, p).conj_mul(self.get(i, q)));
        }
        acc
    }
    /// Right-multiply columns p,q by the 2×2 unitary diag(e,1)·[[c,s],[−s,c]].
    pub fn rotate_cols_complex_jacobi(&mut self, p: usize, q: usize, e: &MpC, c: &Float, s: &Float) {
        for i in 0..self.rows {
            let x = self.get(i, p).clone();
            let y = self.get(i, q).clone();
            let ex = x.mul(e);
            let new_p = ex.scale(c).sub(&y.scale(s));
            let new_q = ex.scale(s).add(&y.scale(c));
            self.set(i, p, new_p);
            self.set(i, q, new_q);
        }
    }
    pub fn normalize_columns_to_unit(&self, sigmas: &[Float]) -> Self {
        let mut u = Self::zeros(self.rows, self.cols, self.prec);
        for j in 0..self.cols {
            if sigmas[j] == 0 {
                continue;
            }
            for i in 0..self.rows {
                u.set(i, j, self.get(i, j).div_real(&sigmas[j]));
            }
        }
        u
    }
    pub fn permute_columns(&self, perm: &[usize]) -> Self {
        let mut out = Self::zeros(self.rows, self.cols, self.prec);
        for (new_j, &old_j) in perm.iter().enumerate() {
            for i in 0..self.rows {
                out.set(i, new_j, self.get(i, old_j).clone());
            }
        }
        out
    }
    /// Frobenius norm ‖A − U Σ V*‖ (validation).
    pub fn residual_norm(&self, u: &MpMatrix, sigmas: &[Float], v: &MpMatrix) -> Float {
        let mut acc = Float::with_val(self.prec, 0);
        for i in 0..self.rows {
            for j in 0..self.cols {
                let mut approx = MpC::zero(self.prec);
                for k in 0..self.cols {
                    let uk = u.get(i, k).scale(&sigmas[k]);
                    approx = approx.add(&uk.mul(&v.get(j, k).conj()));
                }
                acc += self.get(i, j).sub(&approx).abs2();
            }
        }
        acc.sqrt_mut();
        acc
    }
}

#[derive(Clone, Debug)]
pub struct JacobiSvdOptions {
    pub prec: u32,
    pub max_sweeps: usize,
    pub tol: Float,
    pub sort_descending: bool,
    pub cluster_tol: Float,
}

impl JacobiSvdOptions {
    pub fn new(prec: u32, max_sweeps: usize, tol_decimal: &str, cluster_decimal: &str) -> Self {
        let tol = Float::with_val(prec, Float::parse(tol_decimal).expect("valid tol"));
        let cluster_tol = Float::with_val(prec, Float::parse(cluster_decimal).expect("valid cluster tol"));
        Self { prec, max_sweeps, tol, sort_descending: true, cluster_tol }
    }
}

#[derive(Clone, Debug)]
pub struct JacobiSvdResult {
    pub u: MpMatrix,
    pub sigma: Vec<Float>,
    pub v: MpMatrix,
    pub sweeps: usize,
    pub final_offdiag: Float,
    pub clusters: Vec<Vec<usize>>,
}

impl JacobiSvdResult {
    /// Indices of singular values ≤ threshold — the numerical kernel dimension.
    pub fn numerical_nullity_indices(&self, threshold: &Float) -> Vec<usize> {
        self.sigma
            .iter()
            .enumerate()
            .filter_map(|(i, s)| (s <= threshold).then_some(i))
            .collect()
    }
    /// Right singular vectors (columns of V) with σ ≤ threshold: an hp kernel basis of A.
    pub fn right_nullspace_basis(&self, threshold: &Float) -> MpMatrix {
        let idx = self.numerical_nullity_indices(threshold);
        let mut out = MpMatrix::zeros(self.v.rows, idx.len(), self.v.prec);
        for (new_j, &old_j) in idx.iter().enumerate() {
            for i in 0..self.v.rows {
                out.set(i, new_j, self.v.get(i, old_j).clone());
            }
        }
        out
    }
}

/// Thin one-sided complex Jacobi SVD of `a` (requires rows ≥ cols).
pub fn jacobi_svd(a: &MpMatrix, opt: &JacobiSvdOptions) -> Result<JacobiSvdResult, SvdError> {
    if a.rows == 0 || a.cols == 0 {
        return Err(SvdError::EmptyMatrix);
    }
    if a.rows < a.cols {
        return Err(SvdError::WideMatrixUnsupported);
    }
    if opt.tol <= 0 {
        return Err(SvdError::InvalidTolerance);
    }
    let prec = opt.prec;
    let n = a.cols;
    let mut w = a.clone(); // W = A V
    let mut v = MpMatrix::identity(n, prec);
    let mut sweeps_done = 0usize;

    for sweep in 0..opt.max_sweeps {
        let mut changed = false;
        let mut max_off = Float::with_val(prec, 0);
        for p in 0..n {
            for q in (p + 1)..n {
                let app = w.col_norm2(p);
                let aqq = w.col_norm2(q);
                if app == 0 || aqq == 0 {
                    continue;
                }
                let apq = w.col_dot(p, q);
                let beta = apq.abs(prec);
                if beta == 0 {
                    continue;
                }
                let mut denom = Float::with_val(prec, app.clone() * &aqq);
                denom.sqrt_mut();
                let off = beta.clone() / denom;
                if off > max_off {
                    max_off = off.clone();
                }
                if off <= opt.tol {
                    continue;
                }
                let (e, c, s) = jacobi_pair_rotation(&app, &aqq, &apq, prec);
                w.rotate_cols_complex_jacobi(p, q, &e, &c, &s);
                v.rotate_cols_complex_jacobi(p, q, &e, &c, &s);
                changed = true;
            }
        }
        sweeps_done = sweep + 1;
        if !changed || max_off <= opt.tol {
            break;
        }
    }

    let mut sigmas = Vec::with_capacity(n);
    for j in 0..n {
        let mut s = w.col_norm2(j);
        s.sqrt_mut();
        sigmas.push(s);
    }
    let mut u = w.normalize_columns_to_unit(&sigmas);

    if opt.sort_descending {
        let mut perm: Vec<usize> = (0..n).collect();
        perm.sort_by(|&i, &j| sigmas[j].partial_cmp(&sigmas[i]).unwrap());
        sigmas = perm.iter().map(|&i| sigmas[i].clone()).collect();
        u = u.permute_columns(&perm);
        v = v.permute_columns(&perm);
    }

    let final_offdiag = max_normalized_offdiag(&w);
    let clusters = detect_singular_value_clusters(&sigmas, &opt.cluster_tol, prec);
    Ok(JacobiSvdResult { u, sigma: sigmas, v, sweeps: sweeps_done, final_offdiag, clusters })
}

/// Complex Jacobi rotation for the 2×2 Hermitian Gram block [[app, apq],[conj apq, aqq]]:
/// strip the phase of apq, diagonalize the real symmetric block, return diag(e,1)·[[c,s],[−s,c]].
fn jacobi_pair_rotation(app: &Float, aqq: &Float, apq: &MpC, prec: u32) -> (MpC, Float, Float) {
    let beta = apq.abs(prec);
    let e = if beta == 0 { MpC::one(prec) } else { apq.div_real(&beta) };
    let two = Float::with_val(prec, 2);
    let tau = (aqq.clone() - app) / (two * beta.clone());
    let abs_tau = tau.clone().abs();
    let mut root = Float::with_val(prec, tau.clone() * &tau + 1);
    root.sqrt_mut();
    let denom = abs_tau + root;
    let mut t = Float::with_val(prec, 1) / denom;
    if tau < 0 {
        t = -t;
    }
    let mut c = Float::with_val(prec, t.clone() * &t + 1);
    c.sqrt_mut();
    c = Float::with_val(prec, 1) / c;
    let s = c.clone() * t;
    (e, c, s)
}

fn max_normalized_offdiag(w: &MpMatrix) -> Float {
    let prec = w.prec;
    let n = w.cols;
    let mut max_off = Float::with_val(prec, 0);
    for p in 0..n {
        for q in (p + 1)..n {
            let app = w.col_norm2(p);
            let aqq = w.col_norm2(q);
            if app == 0 || aqq == 0 {
                continue;
            }
            let beta = w.col_dot(p, q).abs(prec);
            let mut denom = Float::with_val(prec, app * &aqq);
            denom.sqrt_mut();
            let off = beta / denom;
            if off > max_off {
                max_off = off;
            }
        }
    }
    max_off
}

fn detect_singular_value_clusters(sigmas: &[Float], cluster_tol: &Float, prec: u32) -> Vec<Vec<usize>> {
    if sigmas.is_empty() {
        return Vec::new();
    }
    let mut clusters = Vec::<Vec<usize>>::new();
    let mut current = vec![0usize];
    for i in 1..sigmas.len() {
        let (a, b) = (&sigmas[i - 1], &sigmas[i]);
        let gap = (a.clone() - b).abs();
        let scale = if a > b { a.clone() } else { b.clone() };
        let rel_gap = if scale == 0 { Float::with_val(prec, 0) } else { gap / scale };
        if rel_gap <= *cluster_tol {
            current.push(i);
        } else {
            if current.len() > 1 {
                clusters.push(current);
            }
            current = vec![i];
        }
    }
    if current.len() > 1 {
        clusters.push(current);
    }
    clusters
}

#[cfg(test)]
mod tests {
    use super::*;

    const PREC: u32 = 256;

    fn opts() -> JacobiSvdOptions {
        JacobiSvdOptions::new(PREC, 60, "1e-70", "1e-40")
    }

    // Diagonal matrix: singular values are the |diagonal|, sorted.
    #[test]
    fn svd_diagonal() {
        let data = vec![
            MpC::from_f64(PREC, 3.0, 0.0),
            MpC::zero(PREC),
            MpC::zero(PREC),
            MpC::from_f64(PREC, 0.0, -5.0),
        ];
        let a = MpMatrix::from_row_major(2, 2, PREC, data).unwrap();
        let r = jacobi_svd(&a, &opts()).unwrap();
        assert!((r.sigma[0].to_f64() - 5.0).abs() < 1e-60);
        assert!((r.sigma[1].to_f64() - 3.0).abs() < 1e-60);
        assert!(a.residual_norm(&r.u, &r.sigma, &r.v).to_f64() < 1e-60);
    }

    // The note's 3×2 example: reconstruction residual must be ~0.
    #[test]
    fn svd_reconstructs() {
        let data = vec![
            MpC::from_f64(PREC, 1.0, 0.0),
            MpC::from_f64(PREC, 0.0, 1.0),
            MpC::from_f64(PREC, 2.0, -1.0),
            MpC::from_f64(PREC, 1.0, 0.0),
            MpC::from_f64(PREC, -1.0, 0.5),
            MpC::from_f64(PREC, 0.25, -0.75),
        ];
        let a = MpMatrix::from_row_major(3, 2, PREC, data).unwrap();
        let r = jacobi_svd(&a, &opts()).unwrap();
        assert!(a.residual_norm(&r.u, &r.sigma, &r.v).to_f64() < 1e-60);
        assert!(r.sigma[0] >= r.sigma[1]); // sorted
    }

    // Rank-deficient matrix: exactly one tiny singular value ⇒ numerical nullity 1,
    // and the recovered kernel vector v satisfies A v ≈ 0.
    #[test]
    fn svd_nullspace() {
        // columns c0, c1 with c2 = c0 + 2 c1  ⇒ rank 2, nullity 1 (3×3).
        let c = |re: f64, im: f64| MpC::from_f64(PREC, re, im);
        // rows of [c0 | c1 | c0+2c1]
        let col0 = [c(1.0, 0.0), c(0.0, 1.0), c(2.0, 0.0)];
        let col1 = [c(1.0, 1.0), c(-1.0, 0.0), c(0.0, 1.0)];
        let mut data = Vec::new();
        for i in 0..3 {
            data.push(col0[i].clone());
            data.push(col1[i].clone());
            data.push(col0[i].add(&col1[i].scale(&Float::with_val(PREC, 2))));
        }
        let a = MpMatrix::from_row_major(3, 3, PREC, data).unwrap();
        let r = jacobi_svd(&a, &opts()).unwrap();
        let thr = Float::with_val(PREC, Float::parse("1e-50").unwrap());
        let ker = r.numerical_nullity_indices(&thr);
        assert_eq!(ker.len(), 1, "expected nullity 1, σ = {:?}", r.sigma.iter().map(|s| s.to_f64()).collect::<Vec<_>>());
        // A v ≈ 0 for the kernel vector
        let vb = r.right_nullspace_basis(&thr);
        let mut resid = Float::with_val(PREC, 0);
        for i in 0..3 {
            let mut acc = MpC::zero(PREC);
            for j in 0..3 {
                acc = acc.add(&a.get(i, j).mul(vb.get(j, 0)));
            }
            resid += acc.abs2();
        }
        assert!(resid.to_f64() < 1e-80, "‖A v‖² = {:.2e}", resid.to_f64());
    }
}
