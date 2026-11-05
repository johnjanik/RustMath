//! Arbitrary-precision polynomial root-finding over [`BigComplex`] via
//! Aberth–Ehrlich simultaneous iteration.
//!
//! PLAN Phase-2 deliverable: reused by number fields, Galois theory, and
//! elliptic curves (period computations).
//!
//! # Method
//!
//! [`aberth_roots`] finds all complex roots of a univariate polynomial
//! `p(x) = a_0 + a_1 x + … + a_n x^n` (coefficients in **ascending** order)
//! with the Aberth–Ehrlich third-order simultaneous iteration
//!
//! ```text
//! z_i ← z_i − N_i / (1 − N_i · Σ_{j≠i} 1/(z_i − z_j)),   N_i = p(z_i)/p'(z_i)
//! ```
//!
//! run Gauss–Seidel style (updated positions are used immediately) at a
//! working precision of `precision + 64` guard bits.
//!
//! **Deterministic initialization**: the `n` starting points are placed on a
//! circle `R·exp(iθ_k)`, where `log2 R` over-approximates the Fujiwara root
//! bound `2·max_k |a_{n−k}/a_n|^{1/k}` (computed exactly from the dyadic
//! exponents of the coefficients) and `θ_k = π(4k+1)/(2n)`. No angle is `0`
//! or `π` and the set is not conjugate-symmetric, so real-axis stalemates for
//! real-coefficient inputs are impossible; the same input always produces the
//! same output.
//!
//! # Certification: what is and is not guaranteed
//!
//! Two stopping criteria, checked each sweep:
//!
//! 1. **Backward error (freeze)**: `|p(z_i)| ≤ 2^{8−wp} · Σ_k |a_k||z_i|^k`
//!    (with `wp = precision + 64` the working precision). This makes `z_i` an
//!    exact root of a polynomial whose coefficients are relatively perturbed
//!    by at most `2^{8−wp}` — indistinguishable from a true root at working
//!    precision, so the point is frozen.
//! 2. **Correction size**: a sweep in which every live correction satisfies
//!    `|w_i| ≤ 2^{−(precision+16)}·(1+|z_i|)` ends the iteration
//!    (`converged = true`). The final sweeps therefore double as Newton
//!    polish for the well-separated roots.
//!
//! Each returned root also carries a **certified forward bound**
//! ([`RootEstimate::error_bound`]): the classical partial-fraction inequality
//! `min_j |z − ζ_j| ≤ n·|p(z)/p'(z)|` guarantees a true root of the input
//! within that distance of `z` (inflated by one target-precision ulp for the
//! final rounding, plus 2⁻⁸ relative headroom). The bound is *evaluated* in
//! `wp`-bit floating point, not ball arithmetic; the 64 guard bits dominate
//! the evaluation rounding by a wide margin, but callers needing a fully
//! rigorous enclosure should re-verify with
//! [`BigComplexBall`](crate::ball::BigComplexBall).
//!
//! **Separation** ([`RootEstimate::isolated`]): set when this root's error
//! disk is disjoint from every other approximation's error disk
//! (`err_i + err_j < |z_i − z_j|` for all `j`). For such roots the certified
//! disk pins down a root well separated from all other approximations.
//!
//! # Multiple roots — honest limitations
//!
//! At an `m`-fold root the iteration converges only linearly and the `m`
//! approximations stagnate on a tiny cluster of radius `≈ 2^{−wp/m}` around
//! the true root (they freeze via the backward-error criterion; their
//! individual `error_bound`s honestly report the larger uncertainty).
//! [`PolynomialRoots::clusters`] groups approximations closer than
//! `2^{−precision/2}` (relative): **multiplicity = cluster size is a
//! heuristic, not a certified multiplicity** — genuinely distinct roots
//! closer than the threshold would be merged. The cluster mean
//! ([`RootCluster::center`]) is typically far more accurate than the members
//! because their errors are close to a scaled sum of `m`-th roots of unity,
//! which cancels, but this is not certified either. Roots exactly at the
//! origin are the exception: they are detected exactly from vanishing low
//! coefficients ([`PolynomialRoots::zero_multiplicity`]) before iterating.
//!
//! Coefficients are rounded once to `wp` bits on entry; for `Integer` /
//! `Rational` inputs wider than `wp` bits, all certificates are relative to
//! those correctly-rounded coefficients.

use crate::bigcomplex::BigComplex;
use rustmath_core::analytic::{ComplexField, RealField};
use rustmath_core::{MathError, Result, Ring};
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use rustmath_reals::bigfloat::BigFloat;

/// Guard bits added to the requested precision for the working precision.
const GUARD_BITS: u64 = 64;

/// Hard cap on Aberth sweeps (each sweep is O(n²)).
const MAX_SWEEPS: usize = 1000;

// ------------------------------------------------------------ coefficients --

/// Coefficient types accepted by [`aberth_roots`].
///
/// Implemented for exact inputs (`Integer`, `Rational`, `i64`, `i32`) and
/// floating inputs (`BigFloat`, `BigComplex`); each is rounded once to the
/// working precision.
pub trait RootCoefficient {
    /// This coefficient as a [`BigComplex`] correctly rounded to `precision`
    /// bits.
    fn to_big_complex(&self, precision: u64) -> BigComplex;
}

impl RootCoefficient for Integer {
    fn to_big_complex(&self, precision: u64) -> BigComplex {
        BigComplex::from_integer(self, precision)
    }
}
impl RootCoefficient for Rational {
    fn to_big_complex(&self, precision: u64) -> BigComplex {
        BigComplex::from_rational(self, precision)
    }
}
impl RootCoefficient for i64 {
    fn to_big_complex(&self, precision: u64) -> BigComplex {
        BigComplex::from_integer(&Integer::from(*self), precision)
    }
}
impl RootCoefficient for i32 {
    fn to_big_complex(&self, precision: u64) -> BigComplex {
        BigComplex::from_integer(&Integer::from(*self), precision)
    }
}
impl RootCoefficient for BigFloat {
    fn to_big_complex(&self, precision: u64) -> BigComplex {
        BigComplex::new(self.with_precision(precision), BigFloat::zero_prec(precision))
    }
}
impl RootCoefficient for BigComplex {
    fn to_big_complex(&self, precision: u64) -> BigComplex {
        ComplexField::with_precision(self, precision)
    }
}

// ------------------------------------------------------------------ results --

/// One approximate root with its certificate.
#[derive(Clone, Debug)]
pub struct RootEstimate {
    /// The approximation, rounded to the requested precision.
    pub value: BigComplex,
    /// Certified upper bound on the distance from `value` to *some* true root
    /// (`n·|p(z)/p'(z)|` + rounding allowance; see module docs). `None` when
    /// `p'` vanished exactly at the approximation, so no bound is available.
    pub error_bound: Option<BigFloat>,
    /// Whether this root's error disk is disjoint from every other
    /// approximation's error disk (see module docs).
    pub isolated: bool,
}

/// A group of approximations that agree to roughly half the requested
/// precision — the honest output for a (suspected) multiple root.
#[derive(Clone, Debug)]
pub struct RootCluster {
    /// Mean of the member approximations, rounded to the requested precision.
    pub center: BigComplex,
    /// Cluster size (heuristic multiplicity, **not certified**), except for
    /// the exact origin cluster whose multiplicity is exact.
    pub multiplicity: usize,
    /// Indices into [`PolynomialRoots::roots`].
    pub indices: Vec<usize>,
}

/// Full output of [`aberth_roots`].
#[derive(Clone, Debug)]
pub struct PolynomialRoots {
    /// All `deg p` approximate roots (exact zero roots first).
    pub roots: Vec<RootEstimate>,
    /// Partition of `roots` into proximity clusters; multiplicities sum to
    /// `roots.len()`.
    pub clusters: Vec<RootCluster>,
    /// Exact multiplicity of the root `0` (from vanishing low coefficients).
    pub zero_multiplicity: usize,
    /// Aberth sweeps performed.
    pub iterations: usize,
    /// `true` iff every root hit a stopping criterion before the sweep cap;
    /// on `false` the results (and their certified bounds) are still returned.
    pub converged: bool,
    /// The precision (bits) the results are rounded to.
    pub precision: u64,
}

// ------------------------------------------------------------------ helpers --

/// `2^e` at the given precision (exact — powers of two are dyadic).
fn pow2(e: i64, prec: u64) -> BigFloat {
    let two = BigFloat::from_integer(&Integer::from(2), prec);
    let mut base = if e >= 0 {
        two.clone()
    } else {
        BigFloat::one_prec(prec) / two
    };
    let mut result = BigFloat::one_prec(prec);
    let mut k = e.unsigned_abs();
    while k > 0 {
        if k & 1 == 1 {
            result = result * base.clone();
        }
        base = base.clone() * base;
        k >>= 1;
    }
    result
}

fn bf_i64(n: i64, prec: u64) -> BigFloat {
    BigFloat::from_integer(&Integer::from(n), prec)
}

/// `floor(log2 |x|)` from the exact dyadic representation (`None` for 0).
fn log2_mag(x: &BigFloat) -> Option<i64> {
    let (mantissa, exponent) = x.mantissa_exponent();
    if mantissa.is_zero() {
        None
    } else {
        Some(exponent + mantissa.bit_length() as i64 - 1)
    }
}

/// `floor(log2 max(|re|, |im|))` of a coefficient (`None` for 0).
fn coeff_mag(c: &BigComplex) -> Option<i64> {
    match (log2_mag(&c.real()), log2_mag(&c.imag())) {
        (None, None) => None,
        (a, b) => Some(a.unwrap_or(i64::MIN).max(b.unwrap_or(i64::MIN))),
    }
}

/// Evaluate `p(z)` and `p'(z)` simultaneously by Horner.
fn eval_p_dp(coeffs: &[BigComplex], z: &BigComplex) -> (BigComplex, BigComplex) {
    let n = coeffs.len() - 1;
    let mut p = coeffs[n].clone();
    let mut dp = BigComplex::zero_prec(z.prec());
    for k in (0..n).rev() {
        dp = dp * z.clone() + p.clone();
        p = p * z.clone() + coeffs[k].clone();
    }
    (p, dp)
}

/// `Σ_k |a_k| r^k` (majorant of `|p|` on `|z| = r`), by Horner.
fn eval_majorant(abs_coeffs: &[BigFloat], r: &BigFloat) -> BigFloat {
    let last = abs_coeffs.len() - 1;
    let mut s = abs_coeffs[last].clone();
    for k in (0..last).rev() {
        s = s * r.clone() + abs_coeffs[k].clone();
    }
    s
}

/// Deterministic escape nudge for the (rare) degenerate configurations:
/// exact collision of two iterates, or `p'(z_i) = 0` with `p(z_i) ≠ 0`.
fn nudged(z: &BigComplex, index: usize, wp: u64) -> BigComplex {
    let t = pow2(-((wp / 3) as i64), wp)
        * (BigFloat::one_prec(wp) + z.abs())
        * bf_i64(index as i64 + 1, wp);
    z.clone() + BigComplex::new(t.clone(), t)
}

// ---------------------------------------------------------------- iteration --

/// Deterministic starting points: circle of radius `2^e ≥` (Fujiwara bound),
/// angles `π(4k+1)/(2n)`.
fn initial_points(coeffs: &[BigComplex], wp: u64) -> Vec<BigComplex> {
    let n = coeffs.len() - 1;
    let mag_lead = coeff_mag(&coeffs[n]).expect("leading coefficient is nonzero");
    let mut log2_bound = f64::NEG_INFINITY;
    for k in 1..=n {
        if let Some(mk) = coeff_mag(&coeffs[n - k]) {
            // |a_{n−k}/a_n|^{1/k} ≤ 2^{(mk − mag_lead + 2)/k}
            let cand = (mk - mag_lead + 2) as f64 / k as f64;
            if cand > log2_bound {
                log2_bound = cand;
            }
        }
    }
    let e = if log2_bound.is_finite() {
        (log2_bound.ceil() as i64 + 1).max(0)
    } else {
        0
    };
    let radius = pow2(e, wp);
    let pi = <BigFloat as RealField>::pi(wp);
    let two_n = bf_i64(2 * n as i64, wp);
    (0..n)
        .map(|k| {
            let theta = pi.clone() * bf_i64(4 * k as i64 + 1, wp) / two_n.clone();
            BigComplex::new(radius.clone() * theta.cos(), radius.clone() * theta.sin())
        })
        .collect()
}

/// Run the Aberth–Ehrlich iteration; returns (final points, sweeps, converged).
fn aberth_iterate(
    coeffs: &[BigComplex],
    precision: u64,
    wp: u64,
) -> (Vec<BigComplex>, usize, bool) {
    let n = coeffs.len() - 1;
    let one = BigFloat::one_prec(wp);
    let one_c = BigComplex::one_prec(wp);
    let abs_coeffs: Vec<BigFloat> = coeffs.iter().map(|a| a.abs()).collect();
    let mut z = initial_points(coeffs, wp);
    // Backward-error freeze threshold and correction-size threshold (docs).
    let eps_back = pow2(8 - wp as i64, wp);
    let eps_corr = pow2(-(precision as i64) - 16, wp);
    let mut frozen = vec![false; n];
    let mut iterations = 0usize;
    let mut converged = false;

    for sweep in 0..MAX_SWEEPS {
        iterations = sweep + 1;
        let mut all_settled = true;
        for i in 0..n {
            if frozen[i] {
                continue;
            }
            let (p, dp) = eval_p_dp(coeffs, &z[i]);
            if p.is_zero() {
                frozen[i] = true;
                continue;
            }
            let p_abs = p.abs();
            let z_abs = z[i].abs();
            if p_abs <= eps_back.clone() * eval_majorant(&abs_coeffs, &z_abs) {
                frozen[i] = true; // at the evaluation noise floor: cannot improve
                continue;
            }
            if dp.is_zero() {
                z[i] = nudged(&z[i], i, wp); // stationary point that is not a root
                all_settled = false;
                continue;
            }
            let newton = p / dp;
            let mut coupling = BigComplex::zero_prec(wp);
            let mut collision = false;
            for j in 0..n {
                if j == i {
                    continue;
                }
                let d = z[i].clone() - z[j].clone();
                if d.is_zero() {
                    collision = true;
                    break;
                }
                coupling = coupling + one_c.clone() / d;
            }
            if collision {
                z[i] = nudged(&z[i], i, wp);
                all_settled = false;
                continue;
            }
            let denom = one_c.clone() - newton.clone() * coupling;
            let w = if denom.is_zero() {
                newton // fall back to a plain Newton step
            } else {
                newton / denom
            };
            let w_abs = w.abs();
            z[i] = z[i].clone() - w;
            if w_abs > eps_corr.clone() * (one.clone() + z[i].abs()) {
                all_settled = false;
            }
        }
        if all_settled {
            converged = true;
            break;
        }
    }
    (z, iterations, converged)
}

// -------------------------------------------------------------- entry point --

/// Find all complex roots of `p(x) = coeffs[0] + coeffs[1]·x + …` (ascending
/// coefficients) to `precision` bits. See the module docs for the algorithm,
/// the certified stopping criteria, and the honest multiple-root story.
///
/// Errors on an empty coefficient slice or the zero polynomial (no
/// well-defined root set); a nonzero constant returns an empty root list.
pub fn aberth_roots<T: RootCoefficient>(coeffs: &[T], precision: u64) -> Result<PolynomialRoots> {
    let precision = precision.max(8);
    let wp = precision + GUARD_BITS;
    let mut c: Vec<BigComplex> = coeffs.iter().map(|a| a.to_big_complex(wp)).collect();
    while c.last().is_some_and(|a| a.is_zero()) {
        c.pop();
    }
    if c.is_empty() {
        return Err(MathError::InvalidArgument(
            "aberth_roots: the zero polynomial has no well-defined root set".to_string(),
        ));
    }
    // Exact zero roots: strip vanishing low coefficients (exact for the
    // dyadic working coefficients).
    let mut zero_multiplicity = 0usize;
    while c[0].is_zero() {
        c.remove(0);
        zero_multiplicity += 1;
    }
    let n = c.len() - 1;

    let (z, iterations, converged) = if n == 0 {
        (Vec::new(), 0, true)
    } else {
        aberth_iterate(&c, precision, wp)
    };

    // Assemble the full root list (exact zero roots first) with certificates,
    // all still at working precision.
    let total = zero_multiplicity + n;
    let mut values: Vec<BigComplex> = Vec::with_capacity(total);
    let mut errors: Vec<Option<BigFloat>> = Vec::with_capacity(total);
    for _ in 0..zero_multiplicity {
        values.push(BigComplex::zero_prec(wp));
        errors.push(Some(BigFloat::zero_prec(wp))); // exact roots
    }
    let n_bf = bf_i64(n as i64, wp);
    let ulp = pow2(-(precision as i64), wp);
    let headroom = BigFloat::one_prec(wp) + pow2(-8, wp);
    for zi in &z {
        let (p, dp) = eval_p_dp(&c, zi);
        let base = if p.is_zero() {
            Some(BigFloat::zero_prec(wp))
        } else if dp.is_zero() {
            None // no Newton-based certificate at a critical point
        } else {
            Some(n_bf.clone() * p.abs() / dp.abs())
        };
        // Certified distance to some true root, inflated for the final
        // rounding to `precision` bits (one ulp) plus 2^-8 relative headroom.
        let err = base.map(|b| (b + zi.abs() * ulp.clone()) * headroom.clone());
        values.push(zi.clone());
        errors.push(err);
    }

    // Pairwise distances for isolation flags and clustering.
    let abs_values: Vec<BigFloat> = values.iter().map(|v| v.abs()).collect();
    let mut isolated = vec![true; total];
    let mut parent: Vec<usize> = (0..total).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    let cluster_eps = pow2(-((precision / 2) as i64), wp);
    let one = BigFloat::one_prec(wp);
    for i in 0..total {
        if errors[i].is_none() {
            isolated[i] = false;
        }
        for j in (i + 1)..total {
            let d = (values[i].clone() - values[j].clone()).abs();
            match (&errors[i], &errors[j]) {
                (Some(ei), Some(ej)) => {
                    if ei.clone() + ej.clone() >= d {
                        isolated[i] = false;
                        isolated[j] = false;
                    }
                }
                _ => {
                    isolated[i] = false;
                    isolated[j] = false;
                }
            }
            let scale = if abs_values[i] >= abs_values[j] {
                abs_values[i].clone()
            } else {
                abs_values[j].clone()
            };
            if d <= cluster_eps.clone() * (one.clone() + scale) {
                let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                if ri != rj {
                    parent[ri] = rj;
                }
            }
        }
    }

    // Materialize clusters (deterministic order: by smallest member index).
    let mut cluster_indices: Vec<Vec<usize>> = Vec::new();
    let mut root_of: Vec<Option<usize>> = vec![None; total];
    for i in 0..total {
        let r = find(&mut parent, i);
        match root_of[r] {
            Some(slot) => cluster_indices[slot].push(i),
            None => {
                root_of[r] = Some(cluster_indices.len());
                cluster_indices.push(vec![i]);
            }
        }
    }
    let clusters: Vec<RootCluster> = cluster_indices
        .into_iter()
        .map(|indices| {
            let mut sum = BigComplex::zero_prec(wp);
            for &i in &indices {
                sum = sum + values[i].clone();
            }
            let m = indices.len();
            let center = sum / BigComplex::from_integer(&Integer::from(m as i64), wp);
            RootCluster {
                center: ComplexField::with_precision(&center, precision),
                multiplicity: m,
                indices,
            }
        })
        .collect();

    let roots: Vec<RootEstimate> = (0..total)
        .map(|i| RootEstimate {
            value: ComplexField::with_precision(&values[i], precision),
            error_bound: errors[i]
                .as_ref()
                .map(|e| e.with_precision(precision)),
            isolated: isolated[i],
        })
        .collect();

    Ok(PolynomialRoots {
        roots,
        clusters,
        zero_multiplicity,
        iterations,
        converged,
        precision,
    })
}

// -------------------------------------------------------------------- tests --

#[cfg(test)]
mod tests {
    use super::*;

    fn bfi(n: i64, prec: u64) -> BigFloat {
        BigFloat::from_integer(&Integer::from(n), prec)
    }

    fn bci(re: i64, im: i64, prec: u64) -> BigComplex {
        BigComplex::new(bfi(re, prec), bfi(im, prec))
    }

    /// Exactly 10^-50 (as a 300-bit float).
    fn tol_1e50() -> BigFloat {
        let q = Rational::new(Integer::from(1), Integer::from(10).pow(50)).unwrap();
        BigFloat::from_rational(&q, 300)
    }

    fn min_dist(result: &PolynomialRoots, target: &BigComplex) -> BigFloat {
        result
            .roots
            .iter()
            .map(|r| (r.value.clone() - target.clone()).abs())
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    #[test]
    fn test_known_factorization_quartic() {
        // (x^2+1)(x-2)(x+3) = x^4 + x^3 - 5x^2 + x - 6  [sympy-verified]
        // roots: 2, -3, i, -i.
        let coeffs: Vec<Integer> = [-6i64, 1, -5, 1, 1].iter().map(|&k| Integer::from(k)).collect();
        let r = aberth_roots(&coeffs, 200).unwrap();
        assert_eq!(r.roots.len(), 4);
        assert!(r.converged, "quartic must converge");
        assert_eq!(r.zero_multiplicity, 0);
        let tol = tol_1e50();
        for expected in [bci(2, 0, 300), bci(-3, 0, 300), bci(0, 1, 300), bci(0, -1, 300)] {
            let d = min_dist(&r, &expected);
            assert!(d < tol, "root {expected} missed: distance {}", d.to_decimal_string(5));
        }
        for est in &r.roots {
            let e = est.error_bound.as_ref().expect("simple roots must carry a bound");
            assert!(e < &tol, "certified bound too large: {}", e.to_decimal_string(5));
            assert!(est.isolated, "all four roots are well separated");
        }
        assert_eq!(r.clusters.len(), 4);
        assert!(r.clusters.iter().all(|c| c.multiplicity == 1));
    }

    #[test]
    fn test_wilkinson_lite_degree_10() {
        // p(x) = prod_{k=1..10} (x-k); ascending coefficients verified with
        // sympy: Poly(prod((x-k)), x).all_coeffs() reversed. mpmath polyroots
        // at 300 bits recovers 1..10 with error 0 (below representation).
        let coeffs: Vec<Integer> = [
            3628800i64, -10628640, 12753576, -8409500, 3416930, -902055, 157773, -18150, 1320,
            -55, 1,
        ]
        .iter()
        .map(|&k| Integer::from(k))
        .collect();
        let r = aberth_roots(&coeffs, 200).unwrap();
        assert_eq!(r.roots.len(), 10);
        assert!(r.converged, "wilkinson-10 must converge (got {} sweeps)", r.iterations);
        let tol = tol_1e50();
        for k in 1..=10i64 {
            let d = min_dist(&r, &bci(k, 0, 300));
            assert!(
                d < tol,
                "root {k} matched only to {} (need < 1e-50 at 200 bits)",
                d.to_decimal_string(5)
            );
        }
        for est in &r.roots {
            let e = est.error_bound.as_ref().expect("simple roots must carry a bound");
            assert!(e < &tol);
            assert!(est.isolated);
        }
        assert_eq!(r.clusters.len(), 10);
    }

    #[test]
    fn test_cyclotomic_phi5() {
        // Phi_5 = x^4+x^3+x^2+x+1; roots e^{2πik/5}, k=1..4. Closed forms
        // (sympy-verified): cos(2π/5) = (√5−1)/4, sin(2π/5) = √(10+2√5)/4,
        // cos(4π/5) = −(√5+1)/4,     sin(4π/5) = √(10−2√5)/4.
        let coeffs: Vec<Integer> = (0..5).map(|_| Integer::from(1)).collect();
        let r = aberth_roots(&coeffs, 200).unwrap();
        assert_eq!(r.roots.len(), 4);
        assert!(r.converged);
        let p = 300;
        let sqrt5 = bfi(5, p).sqrt();
        let four = bfi(4, p);
        let c1 = (sqrt5.clone() - bfi(1, p)) / four.clone();
        let s1 = (bfi(10, p) + bfi(2, p) * sqrt5.clone()).sqrt() / four.clone();
        let c2 = -((sqrt5.clone() + bfi(1, p)) / four.clone());
        let s2 = (bfi(10, p) - bfi(2, p) * sqrt5).sqrt() / four;
        let tol = tol_1e50();
        for expected in [
            BigComplex::new(c1.clone(), s1.clone()),
            BigComplex::new(c1, -s1),
            BigComplex::new(c2.clone(), s2.clone()),
            BigComplex::new(c2, -s2),
        ] {
            let d = min_dist(&r, &expected);
            assert!(d < tol, "phi_5 root missed by {}", d.to_decimal_string(5));
        }
        // all roots on the unit circle
        let one = BigFloat::one_prec(300);
        for est in &r.roots {
            let dev = est.value.abs() - one.clone();
            assert!(rustmath_core::ordering::OrderedRing::abs(&dev) < tol);
            assert!(est.error_bound.as_ref().unwrap() < &tol);
            assert!(est.isolated);
        }
    }

    #[test]
    fn test_rational_coefficients() {
        // (x - 1/2)(x + 1/3) = x^2 - x/6 - 1/6
        let m16 = Rational::new(Integer::from(-1), Integer::from(6)).unwrap();
        let coeffs = vec![m16.clone(), m16, Rational::from_integer(1)];
        let r = aberth_roots(&coeffs, 200).unwrap();
        assert_eq!(r.roots.len(), 2);
        assert!(r.converged);
        let tol = tol_1e50();
        let half = BigComplex::from_rational(&Rational::new(Integer::from(1), Integer::from(2)).unwrap(), 300);
        let mthird = BigComplex::from_rational(&Rational::new(Integer::from(-1), Integer::from(3)).unwrap(), 300);
        assert!(min_dist(&r, &half) < tol);
        assert!(min_dist(&r, &mthird) < tol);
    }

    #[test]
    fn test_exact_zero_roots() {
        // x^4 - x^3 = x^3 (x - 1): zero root of exact multiplicity 3.
        let r = aberth_roots(&[0i64, 0, 0, -1, 1], 200).unwrap();
        assert_eq!(r.zero_multiplicity, 3);
        assert_eq!(r.roots.len(), 4);
        assert!(r.converged);
        let tol = tol_1e50();
        assert!(min_dist(&r, &bci(1, 0, 300)) < tol);
        // the three exact zero roots form one exact cluster
        let zero_cluster = r
            .clusters
            .iter()
            .find(|c| c.center.is_zero())
            .expect("origin cluster");
        assert_eq!(zero_cluster.multiplicity, 3);
        // exact roots carry a zero bound
        for i in 0..3 {
            assert!(r.roots[i].value.is_zero());
            assert!(r.roots[i].error_bound.as_ref().unwrap().is_zero());
        }
    }

    #[test]
    fn test_double_root_cluster_honesty() {
        // (x-1)^2 (x+2) = x^3 - 3x + 2: double root at 1 -> the two
        // approximations stagnate in a tiny cluster around 1 (linear
        // convergence; see module docs), the simple root -2 stays certified.
        let r = aberth_roots(&[2i64, -3, 0, 1], 200).unwrap();
        assert_eq!(r.roots.len(), 3);
        assert!(r.converged, "double-root case should settle via the backward criterion");
        assert_eq!(r.clusters.len(), 2, "expected {{1,1}} cluster + {{-2}}");
        let double = r
            .clusters
            .iter()
            .find(|c| c.multiplicity == 2)
            .expect("multiplicity-2 cluster at 1");
        // members and center agree with 1 to well beyond half working precision
        let q30 = Rational::new(Integer::from(1), Integer::from(10).pow(30)).unwrap();
        let tol30 = BigFloat::from_rational(&q30, 300);
        assert!((double.center.clone() - bci(1, 0, 300)).abs() < tol30);
        for &i in &double.indices {
            assert!((r.roots[i].value.clone() - bci(1, 0, 300)).abs() < tol30);
        }
        // the simple root -2 is fully accurate and isolated
        let tol = tol_1e50();
        assert!(min_dist(&r, &bci(-2, 0, 300)) < tol);
        let simple_cluster = r.clusters.iter().find(|c| c.multiplicity == 1).unwrap();
        let simple = &r.roots[simple_cluster.indices[0]];
        assert!(simple.error_bound.as_ref().unwrap() < &tol);
        assert!(simple.isolated);
    }

    #[test]
    fn test_degenerate_inputs() {
        // zero polynomial / empty input: error
        assert!(aberth_roots::<i64>(&[], 200).is_err());
        assert!(aberth_roots(&[0i64, 0], 200).is_err());
        // nonzero constant: no roots
        let r = aberth_roots(&[5i64], 200).unwrap();
        assert!(r.roots.is_empty());
        assert!(r.clusters.is_empty());
        assert!(r.converged);
        // linear: 2x - 7 -> 7/2
        let r = aberth_roots(&[-7i64, 2], 200).unwrap();
        assert_eq!(r.roots.len(), 1);
        assert!(r.converged);
        let expected = BigComplex::from_rational(
            &Rational::new(Integer::from(7), Integer::from(2)).unwrap(),
            300,
        );
        assert!(min_dist(&r, &expected) < tol_1e50());
    }

    #[test]
    fn test_bigfloat_and_bigcomplex_coefficients() {
        // x^2 - 2 with BigFloat coefficients: roots ±√2.
        let wp = 300;
        let coeffs = vec![bfi(-2, wp), bfi(0, wp), bfi(1, wp)];
        let r = aberth_roots(&coeffs, 200).unwrap();
        let sqrt2 = bfi(2, 300).sqrt();
        let tol = tol_1e50();
        assert!(min_dist(&r, &BigComplex::new(sqrt2.clone(), bfi(0, 300))) < tol);
        assert!(min_dist(&r, &BigComplex::new(-sqrt2, bfi(0, 300))) < tol);
        // (x - i)(x + i) = x^2 + 1 fed as BigComplex coefficients
        let coeffs = vec![bci(1, 0, wp), bci(0, 0, wp), bci(1, 0, wp)];
        let r = aberth_roots(&coeffs, 200).unwrap();
        assert!(min_dist(&r, &bci(0, 1, 300)) < tol);
        assert!(min_dist(&r, &bci(0, -1, 300)) < tol);
    }
}
