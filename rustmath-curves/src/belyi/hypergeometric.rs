//! KMSV §4 (hypergeometric part): the triangle-group uniformizer φ as an exact
//! power series.
//!
//! For Δ = Δ(a,b,c) the quotient X(Δ) = Δ\ℍ is a genus-0 orbifold, and the
//! uniformizer φ : X(Δ) ≅ P¹ (elliptic points ↦ 0,1,∞, an a-to-1 map at w_a = 0)
//! has an explicit expansion around z_a. The functional inverse ψ = φ⁻¹ is a ratio
//! of solutions of the hypergeometric ₂F₁ differential equation (eq 4.9–4.14):
//!
//!   ψ(t)/κ = t^{1/a} · F(A,B,C;t) / F(1+A−C, 1+B−C, 2−C; t),  t = φ(w),
//!
//! with A,B,C rational (eq 4.10) and κ (eq 4.13) a ratio of Γ-values. Reverting and
//! raising to the a-th power gives (eq 4.16)
//!
//!   φ(w) = (w/κ)^a + c₂ (w/κ)^{2a} + …  ∈ ℚ[[(w/κ)^a]]   — RATIONAL in u = w/κ.
//!
//! Since κ factors out, the whole computation is exact rational power-series
//! arithmetic. The transcendental κ is only needed later (§5) to fix the scale.

use rustmath_core::traits::Field;
use rustmath_powerseries::series::PowerSeries;
use rustmath_rationals::Rational;

fn rat(n: i64) -> Rational {
    Rational::from_i64(n)
}
fn ratio(n: i64, d: i64) -> Rational {
    Rational::new(n, d).expect("nonzero denominator")
}

/// The hypergeometric parameters (A, B, C) of eq (4.10) for Δ(a,b,c).
pub fn abc_params(a: i64, b: i64, c: i64) -> (Rational, Rational, Rational) {
    let one = rat(1);
    let half = ratio(1, 2);
    let big_a = half.clone() * (one.clone() + ratio(1, a) - ratio(1, b) - ratio(1, c));
    let big_b = half * (one.clone() + ratio(1, a) - ratio(1, b) + ratio(1, c));
    let big_c = one + ratio(1, a);
    (big_a, big_b, big_c)
}

/// The Gaussian hypergeometric series F(A,B,C;t) = Σ (A)_n(B)_n/((C)_n n!) tⁿ,
/// truncated to `prec` terms, over ℚ (requires C ∉ ℤ_{≤0}).
pub fn hyp_series(
    big_a: &Rational,
    big_b: &Rational,
    big_c: &Rational,
    prec: usize,
) -> PowerSeries<Rational> {
    let mut coeffs = Vec::with_capacity(prec);
    let mut c = rat(1);
    coeffs.push(c.clone());
    for n in 1..prec {
        let nm1 = rat((n - 1) as i64);
        // c_n = c_{n-1} · (A+n-1)(B+n-1) / ((C+n-1)·n)
        let num = (big_a.clone() + nm1.clone()) * (big_b.clone() + nm1.clone());
        let den = (big_c.clone() + nm1) * rat(n as i64);
        c = c.clone() * num / den;
        coeffs.push(c.clone());
    }
    PowerSeries::new(coeffs, prec)
}

/// Series reversion by Newton iteration: given `f` with f(0)=0 and f'(0)≠0, return
/// `g` with f(g(u)) = u to the tracked precision. Uses `compose`/`derivative`/
/// `inverse`, doubling correct terms each step.
pub fn revert(f: &PowerSeries<Rational>) -> PowerSeries<Rational> {
    let prec = f.precision();
    let a1_inv = f.coeff(1).inverse().expect("f'(0) ≠ 0 for reversion");
    // g₀ = u / a₁
    let mut g = PowerSeries::new(vec![rat(0), a1_inv], prec);
    let fp = f.derivative();
    let id = PowerSeries::new(vec![rat(0), rat(1)], prec); // the series u
    let iters = (prec as f64).log2().ceil() as usize + 2;
    for _ in 0..iters {
        // g ← g − (f∘g − u)/(f'∘g)
        let fg = f.compose(&g);
        let num = fg - id.clone();
        let denom = fp.compose(&g).inverse().expect("f'∘g invertible (const ≠ 0)");
        let corr = num * denom;
        g = (g.clone() - corr).truncate(prec);
    }
    g
}

fn pow_series(f: &PowerSeries<Rational>, k: usize) -> PowerSeries<Rational> {
    let prec = f.precision();
    let mut r = PowerSeries::new(vec![rat(1)], prec);
    for _ in 0..k {
        r = r * f.clone();
    }
    r
}

/// The uniformizer φ(w) of Δ(a,b,c) expanded around z_a as a rational power series
/// in `u = w/κ` (eq 4.16): φ = u^a + c₂ u^{2a} + …. Returns the series in `u`
/// (only exponents divisible by a are nonzero). `prec` = number of `u`-terms tracked.
pub fn phi_in_u(a: i64, b: i64, c: i64, prec: usize) -> PowerSeries<Rational> {
    let (big_a, big_b, big_c) = abc_params(a, b, c);
    let one = rat(1);
    // F₁ = F(A,B,C;t),  F₂ʰʸᵖ = F(1+A−C, 1+B−C, 2−C; t)
    let f1 = hyp_series(&big_a, &big_b, &big_c, prec);
    let a2 = one.clone() + big_a.clone() - big_c.clone();
    let b2 = one.clone() + big_b.clone() - big_c.clone();
    let c2 = rat(2) - big_c.clone();
    let f2 = hyp_series(&a2, &b2, &c2, prec);
    // R(t) = F₁/F₂ʰʸᵖ  (so ψ/κ = t^{1/a} R(t))
    let r = f1 * f2.inverse().expect("F₂(0)=1 invertible");
    // u(τ) = τ·R(τ^a): coefficient of τ^{a·n+1} is R_n.
    let mut ucoeffs = vec![rat(0); prec];
    for n in 0..prec {
        let idx = a as usize * n + 1;
        if idx < prec {
            ucoeffs[idx] = r.coeff(n).clone();
        }
    }
    let u_of_tau = PowerSeries::new(ucoeffs, prec);
    // revert to τ(u), then φ = τ(u)^a = t, a series in u = w/κ.
    let tau = revert(&u_of_tau);
    pow_series(&tau, a as usize)
}

/// R(t) = F₁/F₂ʰʸᵖ, i.e. the coefficients of ψ(t)/(κ t^{1/a}) — useful for validation
/// against the paper's intermediate `w/κ = t^{1/a}(1 + t/10 + …)` expansion.
pub fn psi_over_kappa_reduced(a: i64, b: i64, c: i64, prec: usize) -> PowerSeries<Rational> {
    let (big_a, big_b, big_c) = abc_params(a, b, c);
    let one = rat(1);
    let f1 = hyp_series(&big_a, &big_b, &big_c, prec);
    let a2 = one.clone() + big_a.clone() - big_c.clone();
    let b2 = one.clone() + big_b.clone() - big_c.clone();
    let c2 = rat(2) - big_c;
    let f2 = hyp_series(&a2, &b2, &c2, prec);
    f1 * f2.inverse().expect("F₂(0)=1 invertible")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abc_5_3_3() {
        let (a, b, c) = abc_params(5, 3, 3);
        assert_eq!(a, ratio(4, 15));
        assert_eq!(b, ratio(3, 5));
        assert_eq!(c, ratio(6, 5));
    }

    #[test]
    fn abc_2_12_5() {
        let (a, b, c) = abc_params(2, 12, 5);
        assert_eq!(a, ratio(73, 120));
        assert_eq!(b, ratio(97, 120));
        assert_eq!(c, ratio(3, 2));
    }

    // Paper (5,3,3): w/κ = t^{1/5}(1 + t/10 + 3943/89100 t² + 2161/81000 t³ + …).
    #[test]
    fn psi_reduced_5_3_3() {
        let r = psi_over_kappa_reduced(5, 3, 3, 6);
        assert_eq!(*r.coeff(0), rat(1));
        assert_eq!(*r.coeff(1), ratio(1, 10));
        assert_eq!(*r.coeff(2), ratio(3943, 89100));
        assert_eq!(*r.coeff(3), ratio(2161, 81000));
    }

    // Paper (5,3,3), eq near (5.10):
    //   φ(w) = (w/κ)^5 − 1/2 (w/κ)^10 + 637/3564 (w/κ)^15 − 383/7128 (w/κ)^20 + O(w^25).
    #[test]
    fn phi_5_3_3_matches_paper() {
        let phi = phi_in_u(5, 3, 3, 25);
        assert_eq!(*phi.coeff(5), rat(1));
        assert_eq!(*phi.coeff(10), ratio(-1, 2));
        assert_eq!(*phi.coeff(15), ratio(637, 3564));
        assert_eq!(*phi.coeff(20), ratio(-383, 7128));
        // exponents not divisible by a=5 vanish
        for k in [1, 2, 3, 4, 6, 7, 11, 13, 16, 19] {
            assert_eq!(*phi.coeff(k), rat(0), "phi coeff {k} should vanish");
        }
    }

    // Our target: φ for (2,12,5). Leading term is u^2; verify structure (rational,
    // even exponents only) and that it is computable to reasonable depth.
    #[test]
    fn phi_2_12_5_structure() {
        let phi = phi_in_u(2, 12, 5, 20);
        assert_eq!(*phi.coeff(0), rat(0));
        assert_eq!(*phi.coeff(2), rat(1)); // (w/κ)^a leading coefficient is 1
        // a = 2 ⇒ all odd-degree coefficients vanish
        for k in (1..20).step_by(2) {
            assert_eq!(*phi.coeff(k), rat(0), "odd phi coeff {k} must vanish");
        }
        // and at least one higher even coefficient is present (nontrivial series)
        assert_ne!(*phi.coeff(4), rat(0));
    }
}
