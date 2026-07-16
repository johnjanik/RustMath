//! KMSV §5 (genus 0): assemble the rational Belyi map from the modular forms.
//!
//! Given a basis of S_k(Γ) (from [`super::modular_forms_hp::recover_forms`]), echelonize
//! by w-valuation to get g(w) = w^m + O(w^{m+2e}), h(w) = w^{m+e} + O(w^{m+2e}) (eq 5.5),
//! form the coordinate x(w) on X(Γ) ≅ P¹, and (§5b) match φ_Δ(x(w)) = φ(w) against the
//! exact hypergeometric series to recover the rational map Φ(x). This module is the
//! genus-0 assembly; hp complex power-series arithmetic is done in place on Vec<Complex>.

use super::modular_forms_hp::{atlas_setup, domain_radius_hp_centered, recover_forms_centered, AtlasDumpParams};
use super::mp_svd::{jacobi_svd, JacobiSvdOptions, MpC, MpMatrix};
use super::solve::SolveParams;
use super::triangle_group_hp::TriangleGroupHp;
use rug::{Complex, Float};

/// A certified rational-map fit from [`solve_belyi_map_certified`]: the canonically
/// normalized coefficients plus TWO independent residual estimates for the unit-norm
/// null vector the SVD returned.
pub struct BelyiFit {
    /// P coefficients p_0..p_d, normalized so Q's highest-degree nonzero coeff is 1.
    pub p: Vec<Complex>,
    /// Q coefficients q_0..q_d, with the highest-degree nonzero coeff exactly 1.
    pub q: Vec<Complex>,
    /// σ_min of the linear system (the SVD's claim for min ‖M v‖ over unit v).
    pub sigma_min: Float,
    /// ‖M v‖₂ recomputed directly from the assembled rows at the returned unit null
    /// vector v — the independent cross-check of `sigma_min`.
    pub direct_residual: Float,
}

impl BelyiFit {
    /// The certified fit residual: max(σ_min, direct recomputation) — the honest
    /// upper estimate of the two (they agree to rounding when the SVD is sane).
    pub fn residual(&self) -> Float {
        if self.direct_residual >= self.sigma_min {
            self.direct_residual.clone()
        } else {
            self.sigma_min.clone()
        }
    }
}

/// Failure of the canonical normalization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FitError {
    /// Every Q coefficient of the null vector is below the zero tolerance — the
    /// "map" has no denominator, so no canonical form exists (and no genuine
    /// rational Belyi map was found; do NOT weaken the tolerance to force one).
    DegenerateDenominator,
}

/// Shared assembly + SVD for the φ·Q(x) − P(x) = 0 system. Returns the RAW
/// (unit-norm, un-normalized) null-vector split (p, q), σ_min, and ‖M v‖₂
/// recomputed directly from the assembled matrix at that null vector.
fn solve_belyi_map_core(
    x: &[Complex],
    phi: &[Complex],
    d: usize,
    opt: &JacobiSvdOptions,
    extra_rows: usize,
) -> (Vec<Complex>, Vec<Complex>, Float, Float) {
    let prec = opt.prec;
    let len = x.len();
    // x^i for i = 0..=d
    let mut xpow = vec![vec![Complex::with_val(prec, (1.0, 0.0)); len]];
    for i in 1..=d {
        xpow.push(series_mul(&xpow[i - 1], x, len, prec));
    }
    // φ·x^i
    let phixi: Vec<Vec<Complex>> = (0..=d).map(|i| series_mul(phi, &xpow[i], len, prec)).collect();
    let ncols = 2 * (d + 1); // [q_0..q_d, p_0..p_d]
    let nrows = ncols + extra_rows;
    assert!(len >= nrows, "series length {len} < system rows {nrows}");
    let mut data = Vec::with_capacity(nrows * ncols);
    for n in 0..nrows {
        for i in 0..=d {
            data.push(MpC::new(phixi[i][n].real().clone(), phixi[i][n].imag().clone()));
        }
        for i in 0..=d {
            data.push(MpC::new(
                Float::with_val(prec, -xpow[i][n].real()),
                Float::with_val(prec, -xpow[i][n].imag()),
            ));
        }
    }
    let mat = MpMatrix::from_row_major(nrows, ncols, prec, data).expect("system matrix");
    let svd = jacobi_svd(&mat, opt).expect("svd");
    let last = ncols - 1; // smallest singular value ⇒ null vector
    let sigma_min = svd.sigma[last].clone();
    // direct residual: r = M v at the returned null vector (v is unit-norm)
    let mut r2 = Float::with_val(prec, 0.0);
    for n in 0..nrows {
        let mut acc = MpC::zero(prec);
        for j in 0..ncols {
            acc = acc.add(&mat.get(n, j).mul(svd.v.get(j, last)));
        }
        r2 += acc.abs2();
    }
    let direct = r2.sqrt();
    let take = |i: usize| {
        let z = svd.v.get(i, last);
        Complex::with_val(prec, (z.re.clone(), z.im.clone()))
    };
    let q: Vec<Complex> = (0..=d).map(take).collect();
    let p: Vec<Complex> = (0..=d).map(|i| take(d + 1 + i)).collect();
    (p, q, sigma_min, direct)
}

/// Solve for the rational Belyi map Φ(x) = P(x)/Q(x) (deg ≤ d each) from the power
/// series x(w) and φ(w), via the linear relation φ(w)·Q(x(w)) − P(x(w)) = 0. The
/// coefficient-of-w^n equations form a homogeneous system whose 1-D null space (hp SVD)
/// gives (P, Q) up to scale. Returns (P coeffs p_0..p_d, Q coeffs q_0..q_d) — RAW
/// (unit-norm null vector), exactly as always. The SVD tolerances are the prec-derived
/// ones from [`super::solve`], which at prec = 256 are byte-identical to the historical
/// literals ("1e-70", "1e-40"); the row surplus is the (5,3,3)-tested default 6.
/// Prefer [`solve_belyi_map_certified`], which also returns the fit residual.
pub fn solve_belyi_map(x: &[Complex], phi: &[Complex], d: usize, prec: u32) -> (Vec<Complex>, Vec<Complex>) {
    let opt = JacobiSvdOptions::new(
        prec,
        SolveParams::DEFAULT_MAX_SWEEPS,
        &super::solve::jacobi_tol_decimal(prec),
        &super::solve::cluster_tol_decimal(prec),
    );
    let (p, q, _, _) = solve_belyi_map_core(x, phi, d, &opt, SolveParams::DEFAULT_EXTRA_ROWS);
    (p, q)
}

/// Canonically normalize a (P, Q) pair in place: scale both by 1/q_lead, where
/// q_lead is the highest-degree Q coefficient with |q_i| > `zero_tol`·max|(P,Q)|
/// (the largest magnitude over BOTH slices), making it exactly 1 (mpc division is
/// correctly rounded, so q_lead/q_lead == 1 exactly). `zero_tol` is RELATIVE to the
/// joint (P, Q) magnitude: the selection is then invariant under overall rescaling,
/// which is what makes the operation exactly idempotent — an absolute threshold is
/// not scale-invariant, so dividing by a q_lead with |q_lead| < 1 could push a
/// previously-"zero" higher coefficient over the bar and a second pass would pick a
/// different lead. Referencing the JOINT max (not max|q| alone) keeps the degenerate
/// case honest: a Q that is uniformly at the truncation floor while P carries the
/// mass is a vanishing denominator and must be rejected, not self-normalized. For a
/// unit-norm null vector the relative and absolute readings of 10^-digits differ by
/// at most (2d+2)^{1/2}, immaterial at these tolerances.
pub fn normalize_pq(p: &mut [Complex], q: &mut [Complex], prec: u32, zero_tol: &Float) -> Result<(), FitError> {
    let abs = |z: &Complex| -> Float {
        Float::with_val(prec, Float::with_val(prec, z.real() * z.real()) + Float::with_val(prec, z.imag() * z.imag()))
            .sqrt()
    };
    let joint_max = p
        .iter()
        .chain(q.iter())
        .map(&abs)
        .max_by(|a, b| a.partial_cmp(b).expect("magnitudes are never NaN"))
        .ok_or(FitError::DegenerateDenominator)?;
    let bar = Float::with_val(prec, &joint_max * zero_tol);
    let lead = q
        .iter()
        .rposition(|z| abs(z) > bar)
        .ok_or(FitError::DegenerateDenominator)?;
    let q_lead = q[lead].clone();
    for z in p.iter_mut().chain(q.iter_mut()) {
        *z = Complex::with_val(prec, &*z / &q_lead);
    }
    Ok(())
}

/// [`solve_belyi_map`] with a certified residual and canonical normalization
/// (item A6). The SVD options and row surplus come from `params` (no free-floating
/// literals). Returns the fit with BOTH residual estimates — σ_min from the SVD and
/// ‖M v‖₂ recomputed directly from the assembled rows at the returned unit null
/// vector; [`BelyiFit::residual`] is their max (the honest upper estimate). Both
/// refer to the UNIT-norm coefficient vector, not the normalized one. (P, Q) are
/// then canonically normalized: highest-degree nonzero Q coefficient scaled to
/// exactly 1, "nonzero" meaning above the truncation floor 10^-digits.
pub fn solve_belyi_map_certified(
    x: &[Complex],
    phi: &[Complex],
    d: usize,
    params: &SolveParams,
) -> Result<BelyiFit, FitError> {
    let prec = params.prec_bits;
    let opt = params.svd_options();
    let (mut p, mut q, sigma_min, direct) = solve_belyi_map_core(x, phi, d, &opt, params.extra_rows);
    let zero_tol = Float::with_val(
        prec,
        Float::parse(format!("1e-{}", params.digits)).expect("digits tolerance"),
    );
    normalize_pq(&mut p, &mut q, prec, &zero_tol)?;
    Ok(BelyiFit { p, q, sigma_min, direct_residual: direct })
}

/// Convert an exact `Rational` to a high-precision `Float` (via decimal strings).
fn rat_to_float(r: &rustmath_rationals::Rational, prec: u32) -> Float {
    let num = Float::with_val(prec, Float::parse(r.numerator().to_string()).unwrap());
    let den = Float::with_val(prec, Float::parse(r.denominator().to_string()).unwrap());
    Float::with_val(prec, &num / &den)
}

/// κ (eq 4.13): κ = ((μ−1)/(μ+1))·Γ(2−C)Γ(C−A)Γ(C−B)/(Γ(1−A)Γ(1−B)Γ(C)), with
/// A,B,C the hypergeometric parameters (eq 4.10). Computed in hp via the rug Γ-function.
pub fn kappa(tg: &TriangleGroupHp) -> Float {
    let prec = tg.prec;
    let f = |v: f64| Float::with_val(prec, v);
    let inv = |x: f64| Float::with_val(prec, f(1.0) / f(x));
    let (a, b, c) = (tg.a as f64, tg.b as f64, tg.c as f64);
    let big_a = Float::with_val(prec, f(0.5) * Float::with_val(prec, f(1.0) + inv(a) - inv(b) - inv(c)));
    let big_b = Float::with_val(prec, f(0.5) * Float::with_val(prec, f(1.0) + inv(a) - inv(b) + inv(c)));
    let big_c = Float::with_val(prec, f(1.0) + inv(a));
    let gam = |x: Float| x.gamma();
    let num = gam(Float::with_val(prec, f(2.0) - &big_c))
        * gam(Float::with_val(prec, Float::with_val(prec, &big_c - &big_a)))
        * gam(Float::with_val(prec, &big_c - &big_b));
    let den = gam(Float::with_val(prec, f(1.0) - &big_a))
        * gam(Float::with_val(prec, f(1.0) - &big_b))
        * gam(big_c.clone());
    let ratio = Float::with_val(
        prec,
        Float::with_val(prec, &tg.mu - f(1.0)) / Float::with_val(prec, &tg.mu + f(1.0)),
    );
    Float::with_val(prec, &ratio * Float::with_val(prec, num / den))
}

/// The Δ-uniformizer φ as a power series in w: φ(w) = Σ φ_n (w/κ)^{a·n} (eq 4.16),
/// with the exact rational coefficients φ_n from the hypergeometric expansion scaled by
/// κ^{-p}. Real coefficients when κ is real.
pub fn phi_w(a: i64, b: i64, c: i64, kappa: &Float, prec: u32, len: usize) -> Vec<Complex> {
    let phi_u = super::hypergeometric::phi_in_u(a, b, c, len);
    let inv_kappa = Float::with_val(prec, Float::with_val(prec, 1.0) / kappa);
    let mut kpow = vec![Float::with_val(prec, 1.0); len];
    for p in 1..len {
        kpow[p] = Float::with_val(prec, &kpow[p - 1] * &inv_kappa);
    }
    let mut out = vec![Complex::with_val(prec, (0.0, 0.0)); len];
    for p in 0..len {
        let rf = rat_to_float(phi_u.coeff(p), prec);
        let coeff = Float::with_val(prec, &rf * &kpow[p]);
        out[p] = Complex::with_val(prec, (coeff, 0.0));
    }
    out
}

/// First index with |coeff| above `tol` (the w-valuation); `len` if all tiny.
fn valuation(s: &[Complex], tol: f64) -> usize {
    s.iter()
        .position(|c| {
            let re = c.real().to_f64();
            let im = c.imag().to_f64();
            (re * re + im * im).sqrt() > tol
        })
        .unwrap_or(s.len())
}

/// Truncated product of two power series to `len` terms.
fn series_mul(a: &[Complex], b: &[Complex], len: usize, prec: u32) -> Vec<Complex> {
    let mut out = vec![Complex::with_val(prec, (0.0, 0.0)); len];
    for i in 0..a.len().min(len) {
        for j in 0..b.len().min(len - i) {
            let t = Complex::with_val(prec, &a[i] * &b[j]);
            out[i + j] = Complex::with_val(prec, &out[i + j] + &t);
        }
    }
    out
}

/// Reciprocal 1/s of a unit power series (s[0] ≠ 0), to `len` terms.
fn unit_recip(s: &[Complex], len: usize, prec: u32) -> Vec<Complex> {
    let mut r = vec![Complex::with_val(prec, (0.0, 0.0)); len];
    let s0_inv = Complex::with_val(prec, Complex::with_val(prec, (1.0, 0.0)) / &s[0]);
    r[0] = s0_inv.clone();
    for n in 1..len {
        let mut acc = Complex::with_val(prec, (0.0, 0.0));
        for j in 1..=n {
            if j < s.len() {
                let t = Complex::with_val(prec, &s[j] * &r[n - j]);
                acc = Complex::with_val(prec, &acc + &t);
            }
        }
        r[n] = Complex::with_val(prec, -Complex::with_val(prec, &s0_inv * &acc));
    }
    r
}

/// Power-series quotient num/den (valuation(num) ≥ valuation(den)), to `len` terms.
fn series_div(num: &[Complex], den: &[Complex], len: usize, prec: u32, tol: f64) -> Vec<Complex> {
    let vn = valuation(num, tol);
    let vd = valuation(den, tol);
    assert!(vn >= vd, "series_div: numerator valuation {vn} < denominator {vd}");
    let num_s = &num[vn..];
    let den_s = &den[vd..];
    let inv = unit_recip(den_s, len, prec);
    let q = series_mul(num_s, &inv, len, prec);
    // shift up by (vn - vd)
    let shift = vn - vd;
    let mut out = vec![Complex::with_val(prec, (0.0, 0.0)); len];
    for i in 0..len.saturating_sub(shift) {
        out[i + shift] = q[i].clone();
    }
    out
}

/// Convert recovered forms (coefficient vectors b, f = (1−w)^k Σ b_n w^n) into ordinary
/// power series Σ c_n w^n by multiplying out the (1−w)^k automorphy factor.
pub fn forms_to_series(forms: &[Vec<Complex>], k: i64, prec: u32) -> Vec<Vec<Complex>> {
    let len = forms[0].len();
    // (1−w)^k = Σ_j binom(k,j) (−1)^j w^j
    let mut omw = vec![Complex::with_val(prec, (0.0, 0.0)); len];
    let mut binom = 1i128;
    for j in 0..=(k as usize).min(len - 1) {
        let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
        omw[j] = Complex::with_val(prec, (sign * binom as f64, 0.0));
        binom = binom * (k as i128 - j as i128) / (j as i128 + 1);
    }
    forms.iter().map(|b| series_mul(&omw, b, len, prec)).collect()
}

/// Echelonize a set of power series into reduced row-echelon form by w-valuation:
/// each returned series is monic at a distinct increasing valuation, with that valuation
/// zeroed in all others. For dim d this yields leading valuations 0,1,…,d−1 (when the
/// space is spanned by such). Returns the series sorted by valuation.
pub fn echelonize(mut rows: Vec<Vec<Complex>>, prec: u32, tol: f64) -> Vec<Vec<Complex>> {
    let d = rows.len();
    let len = rows[0].len();
    let mut pivot_of_row = vec![0usize; d];
    let mut done = 0usize;
    while done < d {
        // pick the row (≥ done) of smallest valuation
        let mut best = done;
        let mut best_val = valuation(&rows[done], tol);
        for r in (done + 1)..d {
            let v = valuation(&rows[r], tol);
            if v < best_val {
                best_val = v;
                best = r;
            }
        }
        rows.swap(done, best);
        let piv = best_val;
        pivot_of_row[done] = piv;
        // normalize row `done` to be monic at its pivot
        let lead_inv = Complex::with_val(prec, Complex::with_val(prec, (1.0, 0.0)) / &rows[done][piv]);
        for j in 0..len {
            rows[done][j] = Complex::with_val(prec, &rows[done][j] * &lead_inv);
        }
        // eliminate this pivot column from every other row
        for r in 0..d {
            if r != done {
                let factor = rows[r][piv].clone();
                let fnorm = {
                    let re = factor.real().to_f64();
                    let im = factor.imag().to_f64();
                    (re * re + im * im).sqrt()
                };
                if fnorm > 0.0 {
                    for j in 0..len {
                        let t = Complex::with_val(prec, &factor * &rows[done][j]);
                        rows[r][j] = Complex::with_val(prec, &rows[r][j] - &t);
                    }
                }
            }
        }
        done += 1;
    }
    // sort by valuation
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by_key(|&i| valuation(&rows[i], tol));
    order.into_iter().map(|i| rows[i].clone()).collect()
}

/// The coordinate x(w) = Θ^e·h(w)/(g(w) + c·h(w)) on X(Γ) (eq 5.5), with c the
/// coefficient of w^{m+2e} in h — the `+c·h` cancels the w^{m+e} term so x(w) =
/// (Θw)^e + O(w^{3e}). Here Θ is left as 1 (an overall scale). For the (5,3,3) genus-0
/// case g,h are the valuation-1 and valuation-2 echelon forms (m=1, e=1, so c = [w³]h).
pub fn coordinate_x(echelon: &[Vec<Complex>], m: usize, e: usize, prec: u32, tol: f64) -> Vec<Complex> {
    let len = echelon[0].len();
    let g = echelon.iter().find(|s| valuation(s, tol) == m).expect("valuation-m form");
    let h = echelon.iter().find(|s| valuation(s, tol) == m + e).expect("valuation-(m+e) form");
    let c = h[m + 2 * e].clone(); // coefficient of w^{m+2e} in h
    // denom = g + c·h
    let mut denom = g.clone();
    for j in 0..len {
        let t = Complex::with_val(prec, &c * &h[j]);
        denom[j] = Complex::with_val(prec, &denom[j] + &t);
    }
    series_div(h, &denom, len, prec, tol)
}

// ---------------------------------------------------------------------------
// A3: the end-to-end solve driver + JSON persistence
// ---------------------------------------------------------------------------

/// Everything one solve run needs: the atlas/geometry half plus the A7 binding.
pub struct BelyiSolveSpec {
    /// Geometry: permutations, (a,b,c), rebase, N, weight k, prec, chart center,
    /// compactify knobs. `nlimbs` and `out` are dump-only fields and are ignored
    /// by the solve driver.
    pub atlas: AtlasDumpParams,
    /// The N-vs-precision binding; must agree with `atlas.n` / `atlas.prec`.
    pub params: SolveParams,
    /// Degree bound d for P, Q (the passport degree for a genus-0 Belyi map).
    pub d: usize,
    /// coordinate_x valuation m: g = w^m + O(w^{m+2e}) (eq 5.5; (5,3,3) uses 1).
    pub m: usize,
    /// coordinate_x valuation step e: h = w^{m+e} + … ((5,3,3) uses 1).
    pub e: usize,
    /// w-valuation tolerance for echelonize/coordinate_x (the (5,3,3) N=48 config
    /// uses 1e-25).
    pub echelon_tol: f64,
}

/// One persisted solve result — exactly what [`write_belyi_result`] serializes and
/// [`read_belyi_result`] parses back. Decimal strings in the file carry full
/// precision (mpfr "as many digits as needed to read back identically").
#[derive(Debug, Clone)]
pub struct BelyiPersist {
    pub p: Vec<Complex>,
    pub q: Vec<Complex>,
    /// max(σ_min, direct recomputation) — see [`BelyiFit::residual`].
    pub residual: Float,
    pub sigma_min: Float,
    pub direct_residual: Float,
    pub prec: u32,
    pub big_n: usize,
    pub chart: String,
    pub abc: (u32, u32, u32),
    pub k: i64,
    pub d: usize,
    pub digits: usize,
}

fn io_invalid(msg: String) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, msg)
}

fn io_data(msg: String) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg)
}

/// Drive the FULL §4/§5 chain end-to-end — `recover_forms_centered` →
/// `forms_to_series` → `echelonize` → `coordinate_x` → `solve_belyi_map_certified`
/// — and persist the certified result as JSON at `out` (item A3).
///
/// * The setup (rebase, compactify, chart-center dispatch) is
///   [`atlas_setup`] — verbatim the streamed-harness path — and the sampling is
///   the harness `q = 2N + 8`.
/// * The N-vs-ρ half of the A7 binding is enforced against the MEASURED chart
///   radius before any assembly; an under-truncated run is refused, not degraded.
/// * `phi` must be the length-(N+1) Δ-uniformizer series for `atlas.abc` at
///   `atlas.prec` (see [`phi_w`]); the x-coordinate is the driver's own
///   `coordinate_x` output (unit scale Θ = 1 — the recovered (P, Q) are in that
///   chart's coordinate, which fixes the map up to the x-rescaling absorbed in
///   its coefficients).
/// * NOTE: the Julia parameter-homotopy JSON emitted by
///   [`super::pipeline::assemble_2_12_5_homotopy`] (target file
///   `belyi_2_12_5_result.json`) is a DIFFERENT solve route with a different
///   format; only name a file `belyi_2_12_5_result.json` when actually running
///   the [2,12,5] production solve.
pub fn run_and_persist_belyi(spec: &BelyiSolveSpec, phi: &[Complex], out: &str) -> std::io::Result<()> {
    let a = &spec.atlas;
    let sp = &spec.params;
    if a.n != sp.big_n {
        return Err(io_invalid(format!("atlas.n = {} but SolveParams.big_n = {}", a.n, sp.big_n)));
    }
    if a.prec != sp.prec_bits {
        return Err(io_invalid(format!("atlas.prec = {} but SolveParams.prec_bits = {}", a.prec, sp.prec_bits)));
    }
    if phi.len() != a.n + 1 {
        return Err(io_invalid(format!("phi has {} terms; need N+1 = {}", phi.len(), a.n + 1)));
    }
    let (tg64, tg, cg, ctr) = atlas_setup(a);
    let rho = domain_radius_hp_centered(&cg, &tg, &ctr);
    sp.check_rho(rho.to_f64())
        .map_err(|e| io_invalid(format!("N-vs-ρ binding failed at measured ρ = {:.6}: {e:?}", rho.to_f64())))?;
    let q_samples = 2 * a.n + 8;
    let forms = recover_forms_centered(
        &tg64, &tg, &cg, a.k, a.n, q_samples, &sp.threshold_decimal, &sp.tol_decimal, 1.0, &ctr,
    );
    if forms.len() < 2 {
        return Err(io_data(format!(
            "dim S_{} = {} (< 2): no (g, h) coordinate pair exists at this weight; refusing to continue",
            a.k,
            forms.len()
        )));
    }
    let series = forms_to_series(&forms, a.k, a.prec);
    let ech = echelonize(series, a.prec, spec.echelon_tol);
    let vals: Vec<usize> = ech.iter().map(|s| valuation(s, spec.echelon_tol)).collect();
    if !vals.contains(&spec.m) || !vals.contains(&(spec.m + spec.e)) {
        return Err(io_data(format!(
            "echelon valuations {vals:?} do not contain m = {} and m+e = {}",
            spec.m,
            spec.m + spec.e
        )));
    }
    let x = coordinate_x(&ech, spec.m, spec.e, a.prec, spec.echelon_tol);
    let fit = solve_belyi_map_certified(&x, phi, spec.d, sp)
        .map_err(|e| io_data(format!("certified solve failed: {e:?}")))?;
    let rec = BelyiPersist {
        residual: fit.residual(),
        sigma_min: fit.sigma_min.clone(),
        direct_residual: fit.direct_residual.clone(),
        p: fit.p,
        q: fit.q,
        prec: a.prec,
        big_n: a.n,
        chart: a.center.clone(),
        abc: a.abc,
        k: a.k,
        d: spec.d,
        digits: sp.digits,
    };
    write_belyi_result(&rec, out)
}

/// Full-precision decimal string for an hp Float (mpfr guarantees enough digits
/// that parsing it back at the same precision recovers the identical value —
/// which is what makes the JSON round trip byte-faithful).
fn float_full_decimal(f: &Float) -> String {
    f.to_string_radix(10, None)
}

fn complex_array_json(v: &[Complex]) -> String {
    let items: Vec<String> = v
        .iter()
        .map(|z| format!("[\"{}\", \"{}\"]", float_full_decimal(z.real()), float_full_decimal(z.imag())))
        .collect();
    format!("[{}]", items.join(", "))
}

/// Serialize a [`BelyiPersist`] as JSON. All hp values are full-precision decimal
/// strings; ints are plain JSON numbers.
pub fn write_belyi_result(r: &BelyiPersist, path: &str) -> std::io::Result<()> {
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str(&format!("  \"P\": {},\n", complex_array_json(&r.p)));
    s.push_str(&format!("  \"Q\": {},\n", complex_array_json(&r.q)));
    s.push_str(&format!("  \"residual\": \"{}\",\n", float_full_decimal(&r.residual)));
    s.push_str(&format!("  \"sigma_min\": \"{}\",\n", float_full_decimal(&r.sigma_min)));
    s.push_str(&format!("  \"direct_residual\": \"{}\",\n", float_full_decimal(&r.direct_residual)));
    s.push_str(&format!("  \"prec\": {},\n", r.prec));
    s.push_str(&format!("  \"N\": {},\n", r.big_n));
    s.push_str(&format!("  \"chart\": \"{}\",\n", r.chart));
    s.push_str(&format!("  \"abc\": [{}, {}, {}],\n", r.abc.0, r.abc.1, r.abc.2));
    s.push_str(&format!("  \"k\": {},\n", r.k));
    s.push_str(&format!("  \"d\": {},\n", r.d));
    s.push_str(&format!("  \"digits\": {}\n", r.digits));
    s.push_str("}\n");
    std::fs::write(path, s)
}

/// Minimal JSON value for the reader (no external deps allowed in belyi modules).
#[derive(Debug, Clone)]
enum Json {
    Str(String),
    Num(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

struct JsonParser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> JsonParser<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && (self.b[self.i] as char).is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn expect(&mut self, c: u8) -> Result<(), String> {
        self.ws();
        if self.i < self.b.len() && self.b[self.i] == c {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected '{}' at byte {}", c as char, self.i))
        }
    }
    fn peek(&mut self) -> Option<u8> {
        self.ws();
        self.b.get(self.i).copied()
    }
    fn string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut out = String::new();
        while self.i < self.b.len() {
            let c = self.b[self.i];
            self.i += 1;
            match c {
                b'"' => return Ok(out),
                b'\\' => {
                    let e = *self.b.get(self.i).ok_or("truncated escape")?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        other => return Err(format!("unsupported escape \\{}", other as char)),
                    }
                }
                _ => out.push(c as char),
            }
        }
        Err("unterminated string".into())
    }
    fn value(&mut self) -> Result<Json, String> {
        match self.peek().ok_or("unexpected end of input")? {
            b'"' => Ok(Json::Str(self.string()?)),
            b'[' => {
                self.expect(b'[')?;
                let mut items = Vec::new();
                if self.peek() == Some(b']') {
                    self.i += 1;
                    return Ok(Json::Arr(items));
                }
                loop {
                    items.push(self.value()?);
                    match self.peek() {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Json::Arr(items));
                        }
                        _ => return Err(format!("expected ',' or ']' at byte {}", self.i)),
                    }
                }
            }
            b'{' => {
                self.expect(b'{')?;
                let mut items = Vec::new();
                if self.peek() == Some(b'}') {
                    self.i += 1;
                    return Ok(Json::Obj(items));
                }
                loop {
                    let key = self.string()?;
                    self.expect(b':')?;
                    items.push((key, self.value()?));
                    match self.peek() {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Json::Obj(items));
                        }
                        _ => return Err(format!("expected ',' or '}}' at byte {}", self.i)),
                    }
                }
            }
            c if c == b'-' || c == b'+' || c.is_ascii_digit() => {
                let start = self.i;
                while self.i < self.b.len()
                    && matches!(self.b[self.i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
                {
                    self.i += 1;
                }
                Ok(Json::Num(String::from_utf8_lossy(&self.b[start..self.i]).into_owned()))
            }
            c => Err(format!("unexpected '{}' at byte {}", c as char, self.i)),
        }
    }
}

fn json_get<'v>(obj: &'v [(String, Json)], key: &str) -> Result<&'v Json, String> {
    obj.iter().find(|(k, _)| k == key).map(|(_, v)| v).ok_or(format!("missing key \"{key}\""))
}

fn json_usize(v: &Json) -> Result<usize, String> {
    match v {
        Json::Num(s) => s.parse().map_err(|e| format!("bad integer {s:?}: {e}")),
        _ => Err("expected number".into()),
    }
}

fn json_float(v: &Json, prec: u32) -> Result<Float, String> {
    match v {
        Json::Str(s) => Ok(Float::with_val(prec, Float::parse(s).map_err(|e| format!("bad decimal {s:?}: {e}"))?)),
        _ => Err("expected decimal string".into()),
    }
}

fn json_complex_vec(v: &Json, prec: u32) -> Result<Vec<Complex>, String> {
    let Json::Arr(items) = v else { return Err("expected array".into()) };
    items
        .iter()
        .map(|it| {
            let Json::Arr(pair) = it else { return Err("expected [re, im] pair".into()) };
            if pair.len() != 2 {
                return Err(format!("expected 2 components, got {}", pair.len()));
            }
            let re = json_float(&pair[0], prec)?;
            let im = json_float(&pair[1], prec)?;
            Ok(Complex::with_val(prec, (re, im)))
        })
        .collect()
}

/// Read back a JSON file produced by [`write_belyi_result`]. hp values are parsed
/// at the file's own recorded `prec`.
pub fn read_belyi_result(path: &str) -> std::io::Result<BelyiPersist> {
    let text = std::fs::read_to_string(path)?;
    let parse = || -> Result<BelyiPersist, String> {
        let mut p = JsonParser { b: text.as_bytes(), i: 0 };
        let Json::Obj(obj) = p.value()? else { return Err("top level is not an object".into()) };
        let prec = json_usize(json_get(&obj, "prec")?)? as u32;
        let abc_v = json_get(&obj, "abc")?;
        let Json::Arr(abc_items) = abc_v else { return Err("abc is not an array".into()) };
        if abc_items.len() != 3 {
            return Err(format!("abc has {} entries", abc_items.len()));
        }
        let chart = match json_get(&obj, "chart")? {
            Json::Str(s) => s.clone(),
            _ => return Err("chart is not a string".into()),
        };
        Ok(BelyiPersist {
            p: json_complex_vec(json_get(&obj, "P")?, prec)?,
            q: json_complex_vec(json_get(&obj, "Q")?, prec)?,
            residual: json_float(json_get(&obj, "residual")?, prec)?,
            sigma_min: json_float(json_get(&obj, "sigma_min")?, prec)?,
            direct_residual: json_float(json_get(&obj, "direct_residual")?, prec)?,
            prec,
            big_n: json_usize(json_get(&obj, "N")?)?,
            chart,
            abc: (
                json_usize(&abc_items[0])? as u32,
                json_usize(&abc_items[1])? as u32,
                json_usize(&abc_items[2])? as u32,
            ),
            k: json_usize(json_get(&obj, "k")?)? as i64,
            d: json_usize(json_get(&obj, "d")?)?,
            digits: json_usize(json_get(&obj, "digits")?)?,
        })
    };
    parse().map_err(|e| io_data(format!("{path}: {e}")))
}

/// The [2,12,5] production spec: the M24-passport degree-24 dessin
/// (the campaign permutations), z_a chart, weight k = 4 (the campaign harness
/// default), N = 3000 / prec = 400 bits / 12 digits — the A7-consistent binding
/// for the measured z_a-chart ρ ≈ 0.9906 (see `explore_2_12_5_domain`;
/// 3000·log10(1/0.9906) ≈ 12.3).
///
/// `m`, `e` (the eq-5.5 valuations feeding `coordinate_x`) depend on the weight-4
/// echelon structure, which is only known once the forms are actually recovered —
/// the campaign session must supply them after inspecting the echelon valuations
/// (the driver refuses, with the valuation list in the error, if they are absent).
pub fn belyi_2_12_5_spec(m: usize, e: usize, echelon_tol: f64) -> BelyiSolveSpec {
    let s0 = vec![0, 14, 10, 9, 4, 5, 23, 17, 18, 3, 2, 11, 22, 13, 1, 15, 16, 7, 8, 19, 21, 20, 12, 6];
    let s1 = vec![14, 2, 22, 9, 16, 8, 13, 15, 18, 1, 23, 20, 3, 0, 21, 12, 19, 7, 17, 11, 10, 4, 5, 6];
    let params = SolveParams::new(400, 3000, 12).expect("(prec=400, N=3000, digits=12) satisfies the A7 binding");
    BelyiSolveSpec {
        atlas: AtlasDumpParams {
            s0,
            s1,
            abc: (2, 12, 5),
            base: 0,
            n: 3000,
            k: 4,
            prec: 400,
            nlimbs: 3,
            r_prune: 0.996,
            l_max: 40,
            center: "a".into(),
            coset: 0,
            out: String::new(),
        },
        params,
        d: 24,
        m,
        e,
        echelon_tol,
    }
}

/// PRODUCTION entry point for the [2,12,5] degree-24 solve: builds the spec via
/// [`belyi_2_12_5_spec`], computes κ and the exact Δ-uniformizer φ(w) for
/// (2,12,5) at 400 bits, and runs [`run_and_persist_belyi`].
///
/// This is the CAMPAIGN session's job: at N = 3000 the in-memory assembly is the
/// known OOM risk (see `dump_scaled_ami_streamed`'s notes) and the run takes
/// serious CPU — nothing in this crate's test suite calls it, and no [2,12,5]
/// result is claimed until it has actually been run. Name the output
/// `belyi_2_12_5_result.json` only for the real run (the Julia parameter-homotopy
/// route in `pipeline.rs` targets that same name with a DIFFERENT format).
pub fn run_belyi_2_12_5(m: usize, e: usize, echelon_tol: f64, out: &str) -> std::io::Result<()> {
    let spec = belyi_2_12_5_spec(m, e, echelon_tol);
    let tg = TriangleGroupHp::new(2, 12, 5, spec.params.prec_bits);
    let kap = kappa(&tg);
    let phi = phi_w(2, 12, 5, &kap, spec.params.prec_bits, spec.atlas.n + 1);
    run_and_persist_belyi(&spec, &phi, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belyi::coset_graph::CosetGraph;
    use crate::belyi::modular_forms_hp::recover_forms;
    use crate::belyi::triangle_group::TriangleGroup;
    use crate::belyi::triangle_group_hp::TriangleGroupHp;

    // Validate the forms→coordinate pipeline against the paper's (5,3,3) x(w) (eq 5.10):
    //   x(w) = Θw − (3/2)Θ³w³ + (81/16)Θ⁵w⁵ − (189/16)Θ⁷w⁷ + …
    // The Θ-INDEPENDENT signature: x is odd, and c₅ = (9/4)c₃², c₇ = (7/2)c₃³.
    #[test]
    fn coordinate_x_matches_paper_5_3_3() {
        let prec = 256u32;
        let tg64 = TriangleGroup::new(5, 3, 3);
        let tg = TriangleGroupHp::new(5, 3, 3, prec);
        let s0 = vec![4, 0, 1, 2, 3];
        let s1 = vec![1, 2, 0, 3, 4];
        let mut cg = CosetGraph::build(&tg64, &s0, &s1);
        cg.compactify(&tg64);

        let forms = recover_forms(&tg64, &tg, &cg, 6, 48, 96, "1e-8", "1e-70", 1.0);
        assert_eq!(forms.len(), 3);
        let series = forms_to_series(&forms, 6, prec);
        let ech = echelonize(series, prec, 1e-25);
        // leading valuations should be 0,1,2
        let vals: Vec<usize> = ech.iter().map(|s| valuation(s, 1e-25)).collect();
        assert_eq!(vals, vec![0, 1, 2], "echelon valuations");

        let x = coordinate_x(&ech, 1, 1, prec, 1e-25);
        let cf = |n: usize| -> num_complex::Complex64 {
            num_complex::Complex64::new(x[n].real().to_f64(), x[n].imag().to_f64())
        };
        let (c1, c3, c5, c7) = (cf(1), cf(3), cf(5), cf(7));
        // x is odd: even coefficients vanish
        for n in [0usize, 2, 4, 6, 8] {
            assert!(cf(n).norm() < 1e-12, "x[{n}] should vanish, got {}", cf(n).norm());
        }
        // leading coefficient is nonzero
        assert!(c1.norm() > 1e-6);
        // Θ-independent relations from eq 5.10
        let r5 = (c5 - 2.25 * c3 * c3).norm() / c5.norm();
        let r7 = (c7 - 3.5 * c3 * c3 * c3).norm() / c7.norm();
        assert!(r5 < 1e-10, "c5 = (9/4)c3² failed, rel {r5:.2e}");
        assert!(r7 < 1e-10, "c7 = (7/2)c3³ failed, rel {r7:.2e}");
    }

    // Probe the [2,12,5] domain radius ρ, which sets the power-series truncation N (error
    // ~ρ^N). Finding: ρ ≈ 0.9906 (vs 0.53 for (5,3,3)) ⇒ N ≈ 3000 for 1e-14 accuracy —
    // the naive KMSV power-series route does not scale to this d=24 passport with the
    // z_a-centered Dirichlet domain. See notes in the module header.
    #[test]
    #[ignore]
    fn explore_2_12_5_domain() {
        let s0: Vec<usize> = vec![0, 14, 10, 9, 4, 5, 23, 17, 18, 3, 2, 11, 22, 13, 1, 15, 16, 7, 8, 19, 21, 20, 12, 6];
        let s1: Vec<usize> = vec![14, 2, 22, 9, 16, 8, 13, 15, 18, 1, 23, 20, 3, 0, 21, 12, 19, 7, 17, 11, 10, 4, 5, 6];
        let tg64 = TriangleGroup::new(2, 12, 5);
        let cg = CosetGraph::build(&tg64, &s0, &s1);
        for &rp in &[0.95f64, 0.97, 0.98, 0.99, 0.995, 0.999] {
            let (reached, rho) = cg.probe_domain(&tg64, rp, 40, 20000);
            println!("r_prune={rp}: reached {reached}/24, ρ={rho:.6}");
        }
    }

    // Shared (5,3,3) setup: returns (x_paper = Θ·x₀, φ(w), Θ).
    fn setup_5_3_3(prec: u32) -> (Vec<Complex>, Vec<Complex>, Complex) {
        use rug::float::Constant;
        let tg64 = TriangleGroup::new(5, 3, 3);
        let tg = TriangleGroupHp::new(5, 3, 3, prec);
        let s0 = vec![4, 0, 1, 2, 3];
        let s1 = vec![1, 2, 0, 3, 4];
        let mut cg = CosetGraph::build(&tg64, &s0, &s1);
        cg.compactify(&tg64);
        let forms = recover_forms(&tg64, &tg, &cg, 6, 48, 96, "1e-8", "1e-70", 1.0);
        let ech = echelonize(forms_to_series(&forms, 6, prec), prec, 1e-25);
        let x0 = coordinate_x(&ech, 1, 1, prec, 1e-25);
        let len = x0.len();
        let kap = kappa(&tg);
        let alpha = Float::with_val(prec, Float::with_val(prec, 81.0) / 2.0);
        let root = (alpha.clone().ln() / Float::with_val(prec, 5.0)).exp();
        let pi = Float::with_val(prec, Constant::Pi);
        let ang = Float::with_val(prec, Float::with_val(prec, &pi * 2.0) / 5.0);
        let scale = Float::with_val(prec, Float::with_val(prec, 1.0) / Float::with_val(prec, &root * &kap));
        let theta = Complex::with_val(
            prec,
            (
                Float::with_val(prec, &scale * ang.clone().cos()),
                Float::with_val(prec, &scale * ang.clone().sin()),
            ),
        );
        let xp: Vec<Complex> = x0.iter().map(|z| Complex::with_val(prec, z * &theta)).collect();
        let phi = phi_w(5, 3, 3, &kap, prec, len);
        (xp, phi, theta)
    }

    // The SOLVER (not verifier): recover Φ = P/Q from x(w), φ(w) with no knowledge of the
    // answer, then check it equals the paper's 648x⁵/(324x⁵+405x⁴−120x²+16).
    #[test]
    fn solve_map_recovers_paper_5_3_3() {
        let prec = 256u32;
        let (xp, phi, _theta) = setup_5_3_3(prec);
        let (p, q) = solve_belyi_map(&xp, &phi, 5, prec);
        // normalize by q₅ (leading denom coeff)
        let q5 = q[5].clone();
        let norm = |z: &Complex| -> num_complex::Complex64 {
            let r = Complex::with_val(prec, z / &q5);
            num_complex::Complex64::new(r.real().to_f64(), r.imag().to_f64())
        };
        // expected (÷324): p = [0,0,0,0,0, 648/324=2]; q = [16/324,0,−120/324,0,405/324,1]
        let exp_p = [0.0, 0.0, 0.0, 0.0, 0.0, 2.0];
        let exp_q = [16.0 / 324.0, 0.0, -120.0 / 324.0, 0.0, 405.0 / 324.0, 1.0];
        for i in 0..=5 {
            let dp = (norm(&p[i]) - num_complex::Complex64::new(exp_p[i], 0.0)).norm();
            let dq = (norm(&q[i]) - num_complex::Complex64::new(exp_q[i], 0.0)).norm();
            assert!(dp < 1e-6, "P[{i}] = {} (want {})", norm(&p[i]), exp_p[i]);
            assert!(dq < 1e-6, "Q[{i}] = {} (want {})", norm(&q[i]), exp_q[i]);
        }
    }

    // A6: the certified solver — canonical normalization (leading Q coeff exactly 1),
    // exact idempotence of the normalization, agreement of the two residual estimates,
    // and a residual recomputation done independently in the test.
    #[test]
    fn certified_solve_5_3_3() {
        use crate::belyi::solve::SolveParams;
        let prec = 256u32;
        let (xp, phi, _theta) = setup_5_3_3(prec);
        let sp = SolveParams::new(prec, 48, 13).unwrap();
        let fit = solve_belyi_map_certified(&xp, &phi, 5, &sp).expect("non-degenerate Q");

        // canonical: highest-degree Q coefficient is EXACTLY 1
        assert_eq!(fit.q[5], Complex::with_val(prec, (1.0, 0.0)));
        // normalized coefficients match the paper's map ÷324 with no manual rescale
        let exp_p = [0.0, 0.0, 0.0, 0.0, 0.0, 2.0];
        let exp_q = [16.0 / 324.0, 0.0, -120.0 / 324.0, 0.0, 405.0 / 324.0, 1.0];
        let to64 = |z: &Complex| num_complex::Complex64::new(z.real().to_f64(), z.imag().to_f64());
        for i in 0..=5 {
            let dp = (to64(&fit.p[i]) - num_complex::Complex64::new(exp_p[i], 0.0)).norm();
            let dq = (to64(&fit.q[i]) - num_complex::Complex64::new(exp_q[i], 0.0)).norm();
            assert!(dp < 1e-6, "P[{i}] = {} (want {})", to64(&fit.p[i]), exp_p[i]);
            assert!(dq < 1e-6, "Q[{i}] = {} (want {})", to64(&fit.q[i]), exp_q[i]);
        }

        // idempotence: normalizing an already-normalized pair is the exact identity
        let mut p2 = fit.p.clone();
        let mut q2 = fit.q.clone();
        let zero_tol = Float::with_val(prec, Float::parse("1e-13").unwrap());
        normalize_pq(&mut p2, &mut q2, prec, &zero_tol).unwrap();
        assert!(p2.iter().zip(fit.p.iter()).all(|(a, b)| a == b), "normalize not idempotent on P");
        assert!(q2.iter().zip(fit.q.iter()).all(|(a, b)| a == b), "normalize not idempotent on Q");

        // the two residual estimates are the same mathematical quantity (‖M v‖ at the
        // minimizing unit v) computed two ways; they must agree far beyond doubt
        let s = fit.sigma_min.to_f64();
        let dr = fit.direct_residual.to_f64();
        assert!(s > 0.0 && dr > 0.0);
        assert!((s - dr).abs() <= 1e-6 * s.max(dr), "σ_min={s:.3e} vs direct={dr:.3e}");
        // truncation-limited fit floor: ρ^48 ≈ 5e-14 amplified through the composition
        // (measured 3.80e-11 at this config; 1e-9 gives honest headroom without
        // admitting a broken fit, which sits at O(1e-3) or worse)
        assert!(fit.residual().to_f64() < 1e-9, "residual {:.3e}", fit.residual().to_f64());

        // independent recomputation: rebuild the unit vector norm from the returned
        // normalized (P,Q) (unit up to a phase — norms are phase-invariant) and take
        // the 2-norm of the series coefficients of φ·Q(x) − P(x) over the system rows.
        let len = xp.len();
        let mut xpow = vec![vec![Complex::with_val(prec, (1.0, 0.0)); len]];
        for i in 1..=5 {
            let next = series_mul(&xpow[i - 1], &xp, len, prec);
            xpow.push(next);
        }
        let mut qx = vec![Complex::with_val(prec, (0.0, 0.0)); len];
        let mut px = vec![Complex::with_val(prec, (0.0, 0.0)); len];
        for i in 0..=5 {
            for n in 0..len {
                let tq = Complex::with_val(prec, &fit.q[i] * &xpow[i][n]);
                qx[n] = Complex::with_val(prec, &qx[n] + &tq);
                let tp = Complex::with_val(prec, &fit.p[i] * &xpow[i][n]);
                px[n] = Complex::with_val(prec, &px[n] + &tp);
            }
        }
        let phi_qx = series_mul(&phi, &qx, len, prec);
        let mut nrm2 = Float::with_val(prec, 0.0);
        for z in fit.p.iter().chain(fit.q.iter()) {
            nrm2 += Float::with_val(prec, Float::with_val(prec, z.real() * z.real()) + Float::with_val(prec, z.imag() * z.imag()));
        }
        let nrows = 2 * 6 + sp.extra_rows;
        let mut r2 = Float::with_val(prec, 0.0);
        for n in 0..nrows {
            let rn = Complex::with_val(prec, &phi_qx[n] - &px[n]);
            r2 += Float::with_val(prec, Float::with_val(prec, rn.real() * rn.real()) + Float::with_val(prec, rn.imag() * rn.imag()));
        }
        let recomputed = Float::with_val(prec, r2 / nrm2).sqrt().to_f64();
        assert!(
            (recomputed - dr).abs() <= 1e-6 * dr.max(recomputed),
            "independent residual {recomputed:.3e} vs returned {dr:.3e}"
        );
    }

    // A3: the driver end-to-end on the cheap (5,3,3) N=48 config — the JSON appears,
    // parses back, P/Q are the paper's map (checked through Θ-independent invariants,
    // valid in the driver's unscaled x0 coordinate), the residual sits at the
    // truncation-limited floor, and the round trip is byte-faithful.
    #[test]
    fn run_and_persist_belyi_5_3_3_roundtrip() {
        use crate::belyi::solve::SolveParams;
        let prec = 256u32;
        let params = SolveParams::new(prec, 48, 13).unwrap();
        let atlas = AtlasDumpParams {
            s0: vec![4, 0, 1, 2, 3],
            s1: vec![1, 2, 0, 3, 4],
            abc: (5, 3, 3),
            base: 0,
            n: 48,
            k: 6,
            prec,
            nlimbs: 2,
            // compactify() defaults — the exact geometry the (5,3,3) tests use
            r_prune: 0.95,
            l_max: 18,
            center: "a".into(),
            coset: 0,
            out: String::new(),
        };
        let spec = BelyiSolveSpec { atlas, params, d: 5, m: 1, e: 1, echelon_tol: 1e-25 };
        let tg = TriangleGroupHp::new(5, 3, 3, prec);
        let kap = kappa(&tg);
        let phi = phi_w(5, 3, 3, &kap, prec, 49);
        let dir = std::env::temp_dir();
        let out1 = dir.join("rustmath_belyi_test_5_3_3.json");
        let out2 = dir.join("rustmath_belyi_test_5_3_3_rt.json");
        let out1 = out1.to_str().unwrap();
        let out2 = out2.to_str().unwrap();
        run_and_persist_belyi(&spec, &phi, out1).expect("driver");

        let r = read_belyi_result(out1).expect("read back");
        assert_eq!(r.p.len(), 6, "P non-empty, deg 5");
        assert_eq!(r.q.len(), 6, "Q non-empty, deg 5");
        assert_eq!((r.prec, r.big_n, r.k, r.d, r.digits), (256, 48, 6, 5, 13));
        assert_eq!(r.chart, "a");
        assert_eq!(r.abc, (5, 3, 3));
        // residual small and consistent with the truncation budget: ρ^48 ≈ 5e-14
        // amplified through the composition (measured 1.06e-11 at this config with
        // the driver's q = 2N+8 sampling); 1e-9 is honest headroom while a broken
        // fit sits at O(1e-3)+
        let res = r.residual.to_f64();
        assert!(res > 0.0 && res < 1e-9, "residual {res:.3e}");
        assert!(r.sigma_min.to_f64() <= res && r.direct_residual.to_f64() <= res);
        // canonical normalization survived persistence: leading Q coeff exactly 1
        assert_eq!(r.q[5], Complex::with_val(prec, (1.0, 0.0)));
        // Θ-independent invariants of the canonically normalized degree-5 map
        // (derived exactly from the paper coefficients, python fractions):
        //   p5 = 2,  q2/q4³ = −128/675,  q0/q4⁵ = 4096/253125,
        // with p0..p4 and the odd Q coefficients vanishing. These hold in ANY
        // x-rescaling, hence in the driver's unscaled x0 coordinate.
        let to64 = |z: &Complex| num_complex::Complex64::new(z.real().to_f64(), z.imag().to_f64());
        for i in 0..=4 {
            assert!(to64(&r.p[i]).norm() < 1e-6, "P[{i}] = {} should vanish", to64(&r.p[i]));
        }
        assert!((to64(&r.p[5]) - num_complex::Complex64::new(2.0, 0.0)).norm() < 1e-6, "P[5] = {}", to64(&r.p[5]));
        for i in [1usize, 3] {
            assert!(to64(&r.q[i]).norm() < 1e-6, "Q[{i}] = {} should vanish", to64(&r.q[i]));
        }
        let (q0, q2, q4) = (to64(&r.q[0]), to64(&r.q[2]), to64(&r.q[4]));
        let i1 = q2 / (q4 * q4 * q4);
        let i2 = q0 / (q4 * q4 * q4 * q4 * q4);
        assert!((i1 - num_complex::Complex64::new(-128.0 / 675.0, 0.0)).norm() < 1e-4, "I1 = {i1}");
        assert!((i2 - num_complex::Complex64::new(4096.0 / 253125.0, 0.0)).norm() < 1e-4, "I2 = {i2}");

        // byte-faithful round trip: serializing the parsed record reproduces the file
        write_belyi_result(&r, out2).expect("re-write");
        assert_eq!(
            std::fs::read(out1).unwrap(),
            std::fs::read(out2).unwrap(),
            "JSON round trip is not byte-faithful"
        );
        std::fs::remove_file(out1).ok();
        std::fs::remove_file(out2).ok();
    }

    // A3: the [2,12,5] production entry point is wired with the verified parameters —
    // WITHOUT running it (that is the campaign session's job; N=3000 at 400 bits).
    #[test]
    fn belyi_2_12_5_spec_is_wired() {
        let spec = belyi_2_12_5_spec(1, 1, 1e-25);
        assert_eq!(spec.atlas.abc, (2, 12, 5));
        assert_eq!(spec.atlas.s0.len(), 24);
        assert_eq!(spec.atlas.s1.len(), 24);
        assert_eq!(spec.atlas.k, 4);
        assert_eq!(spec.atlas.center, "a");
        assert_eq!((spec.atlas.r_prune, spec.atlas.l_max), (0.996, 40));
        assert_eq!(spec.d, 24);
        assert_eq!(
            (spec.params.prec_bits, spec.params.big_n, spec.params.digits),
            (400, 3000, 12)
        );
        // the binding holds at the measured z_a-chart ρ ≈ 0.9906
        spec.params.check_rho(0.9906).expect("N = 3000 covers 12 digits");
        // s0 must have order dividing 2 (the 'a' generator of the (2,12,5) triple)
        for i in 0..24 {
            assert_eq!(spec.atlas.s0[spec.atlas.s0[i]], i, "s0 must square to the identity");
        }
    }

    // normalize_pq refuses to fabricate a canonical form when Q has no coefficient
    // above the zero tolerance.
    #[test]
    fn normalize_pq_rejects_degenerate_denominator() {
        let prec = 128u32;
        let mut p = vec![Complex::with_val(prec, (1.0, 0.0)); 3];
        let mut q = vec![Complex::with_val(prec, (1e-30, 0.0)); 3];
        let zero_tol = Float::with_val(prec, Float::parse("1e-13").unwrap());
        assert_eq!(
            normalize_pq(&mut p, &mut q, prec, &zero_tol),
            Err(FitError::DegenerateDenominator)
        );
    }

    // §5b end-to-end: with κ (rug Γ) and Θ = (81/2)^{1/5}·e^{2πi/5}/κ, verify the paper's
    // Belyi map Φ(x) = 648x⁵/(324x⁵+405x⁴−120x²+16) satisfies Φ(Θ·x₀(w)) = φ(w), where
    // φ(w) is the exact hypergeometric Δ-uniformizer. Validates κ, φ(w), and the whole
    // §5 assembly against ground truth.
    #[test]
    fn belyi_map_matches_paper_5_3_3() {
        use rug::float::Constant;
        let prec = 256u32;
        let tg64 = TriangleGroup::new(5, 3, 3);
        let tg = TriangleGroupHp::new(5, 3, 3, prec);
        let s0 = vec![4, 0, 1, 2, 3];
        let s1 = vec![1, 2, 0, 3, 4];
        let mut cg = CosetGraph::build(&tg64, &s0, &s1);
        cg.compactify(&tg64);
        let forms = recover_forms(&tg64, &tg, &cg, 6, 48, 96, "1e-8", "1e-70", 1.0);
        let ech = echelonize(forms_to_series(&forms, 6, prec), prec, 1e-25);
        let x0 = coordinate_x(&ech, 1, 1, prec, 1e-25);
        let len = x0.len();

        let kap = kappa(&tg);
        assert!((kap.to_f64() - 0.37630).abs() < 1e-4, "κ = {} (expected ≈0.3763)", kap.to_f64());

        // Θ = e^{2πi/5} / ((81/2)^{1/5} · κ)   [fixes 40.5·Θ⁵ = κ⁻⁵ at leading order]
        let alpha = Float::with_val(prec, Float::with_val(prec, 81.0) / 2.0);
        let root = (alpha.clone().ln() / Float::with_val(prec, 5.0)).exp();
        let pi = Float::with_val(prec, Constant::Pi);
        let ang = Float::with_val(prec, Float::with_val(prec, &pi * 2.0) / 5.0);
        let scale = Float::with_val(prec, Float::with_val(prec, 1.0) / Float::with_val(prec, &root * &kap));
        let theta = Complex::with_val(
            prec,
            (
                Float::with_val(prec, &scale * ang.clone().cos()),
                Float::with_val(prec, &scale * ang.clone().sin()),
            ),
        );
        // sanity: Θ ≈ 0.3917053 + 1.205545 i (eq 5.9)
        assert!((theta.real().to_f64() - 0.3917053).abs() < 1e-4, "Θ_re = {}", theta.real().to_f64());
        assert!((theta.imag().to_f64() - 1.205545).abs() < 1e-4, "Θ_im = {}", theta.imag().to_f64());

        // x_paper = Θ · x₀
        let xp: Vec<Complex> = x0.iter().map(|z| Complex::with_val(prec, z * &theta)).collect();
        // Φ(xp) = 648 xp⁵ / (324 xp⁵ + 405 xp⁴ − 120 xp² + 16)
        let x2 = series_mul(&xp, &xp, len, prec);
        let x4 = series_mul(&x2, &x2, len, prec);
        let x5 = series_mul(&x4, &xp, len, prec);
        let cc = |re: f64| Complex::with_val(prec, (re, 0.0));
        let mut num = vec![Complex::with_val(prec, (0.0, 0.0)); len];
        let mut den = vec![Complex::with_val(prec, (0.0, 0.0)); len];
        for j in 0..len {
            num[j] = Complex::with_val(prec, &x5[j] * cc(648.0));
            let t324 = Complex::with_val(prec, &x5[j] * cc(324.0));
            let t405 = Complex::with_val(prec, &x4[j] * cc(405.0));
            let t120 = Complex::with_val(prec, &x2[j] * cc(-120.0));
            den[j] = Complex::with_val(prec, Complex::with_val(prec, &t324 + &t405) + &t120);
        }
        den[0] = Complex::with_val(prec, &den[0] + cc(16.0));
        let phi_of_x = series_div(&num, &den, len, prec, 1e-25);

        let phi = phi_w(5, 3, 3, &kap, prec, len);
        // Compare in the accurate low-order range (higher coeffs degrade as the N=48
        // form error ~ρ^N is amplified by the series division and 5th powers).
        let mut worst = 0f64;
        let mut worst_n = 0;
        for n in 0..11 {
            let d = Complex::with_val(prec, &phi_of_x[n] - &phi[n]);
            let m = (d.real().to_f64().powi(2) + d.imag().to_f64().powi(2)).sqrt();
            if m > worst {
                worst = m;
                worst_n = n;
            }
        }
        // ~2e-8 at N=48 (truncation-limited — tightens with N); decisive vs the exact map.
        assert!(worst < 1e-7, "Φ(Θ·x₀(w)) ≠ φ(w): worst coeff diff {worst:.2e} at n={worst_n}");
    }
}
