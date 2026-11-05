//! The λ-pinned genus-0 degree-24 Belyi system for the `[2,12,5]` class — the
//! genuine solve gap.
//!
//! **Newly authored from the math spec** (Agent C). This module has *no*
//! counterpart in `dessin_engine`; it is built directly from the pinned ansatz in
//! the PGL₂(Q̄) frame that sends the two order-12 points (the whole fibre over
//! `φ = 1`) to `{0, ∞}`:
//!
//! ```text
//!     A(x)² · B(x)  −  λ · R(x)⁵ · S(x)  =  c · x¹²
//! ```
//!
//! * `A` monic deg 8, `B` monic deg 8 — over `φ = 0`, type `2⁸ 1⁸`: `A` holds the
//!   8 double roots, `B` the 8 simple roots.
//! * `R = (x−1)·(x³ + r₂x² + r₁x + r₀)` monic deg 4, `S` monic deg 4 — over
//!   `φ = ∞`, type `5⁴ 1⁴`: `R⁵` gives 4 order-5 poles (one pinned at `x = 1`),
//!   `S` the 4 simple poles.
//! * `λ` a scaling parameter, `c` the constant.
//!
//! Sending the two order-12 points to `{0, ∞}` forces the `φ = 1` polynomial
//! `A²B − λR⁵S` to be the monomial `c·x¹²` (an order-12 zero at `0`; the degree
//! drops to 12, i.e. 12 roots at `∞`).
//!
//! **Unknowns (25):** `a₀..a₇` (8), `b₀..b₇` (8), `r₀,r₁,r₂` (3), `s₀..s₃` (4),
//! `λ` (1), `c` (1). **Equations (25):** `[xⁱ](A²B − λR⁵S − c·x¹²) = 0`,
//! `i = 0..24`.
//!
//! Two products are exposed:
//!
//! 1. [`pinned_system_2_12_5`] — the fixed-target 25-unknown / 25-equation
//!    [`PolySystem`] (for reference / a small Gröbner probe).
//! 2. A **parameter-homotopy** contract: [`psi`] evaluates the 25 coefficients of
//!    `A²B − λR⁵S` at a concrete rational seed of the 24 *solving* variables
//!    (`A,B,R,S,λ`; `c` is absorbed by scaling), and [`p_star`] is the target
//!    parameter vector `coeffs(x¹²) = (0,…,0,1,0,…,0)` (1 at degree 12). A random
//!    seed `z₀` gives free start parameters `p₀ = psi(z₀)`, and the homotopy tracks
//!    `p₀ → p_star`. These feed `rustmath_numerical::homotopy::ParameterHomotopyJob`
//!    in a later phase; no numerical code is called here.

use rustmath_integers::Integer;
use rustmath_polynomials::multivariate::{Monomial, MultivariatePolynomial};
use rustmath_polynomials::poly_system::PolySystem;
use rustmath_rationals::Rational;
use std::collections::BTreeMap;

/// Number of unknowns in the pinned system (`a×8, b×8, r×3, s×4, λ, c`).
pub const NUM_UNKNOWNS: usize = 25;
/// Number of equations (coefficients of `x⁰..x²⁴`).
pub const NUM_EQUATIONS: usize = 25;
/// Number of *solving* variables in the parameter-homotopy view (`c` absorbed):
/// `a×8, b×8, r×3, s×4, λ`.
pub const NUM_SOLVING_VARS: usize = 24;
/// Cover degree (`= deg(A²B) = deg(R⁵S)`).
pub const DEGREE: usize = 24;

// Variable-index layout in the full [`PolySystem`] (25 unknowns).
const A_BASE: usize = 0; //  a0..a7  -> 0..=7
const B_BASE: usize = 8; //  b0..b7  -> 8..=15
const R_BASE: usize = 16; // r0..r2  -> 16..=18
const S_BASE: usize = 19; // s0..s3  -> 19..=22
const LAMBDA: usize = 23;
const C: usize = 24;
const X: usize = 25; // the polynomial variable x (dropped after coefficient extraction)

/// Ordered names of the 25 unknowns, in the [`PolySystem`] variable order.
pub fn unknown_names() -> Vec<String> {
    let mut names = Vec::with_capacity(NUM_UNKNOWNS);
    for i in 0..8 {
        names.push(format!("a{i}"));
    }
    for i in 0..8 {
        names.push(format!("b{i}"));
    }
    for i in 0..3 {
        names.push(format!("r{i}"));
    }
    for i in 0..4 {
        names.push(format!("s{i}"));
    }
    names.push("lambda".into());
    names.push("c".into());
    names
}

/// Ordered names of the 24 *solving* variables (parameter-homotopy view; `c`
/// absorbed by scaling `c ↦ 1`).
pub fn solving_var_names() -> Vec<String> {
    let mut names = unknown_names();
    names.retain(|n| n != "c");
    names
}

// --- symbolic helpers over Integer[unknowns, x] ----------------------------

fn mono(pairs: &[(usize, u32)]) -> Monomial {
    let mut map = BTreeMap::new();
    for &(v, e) in pairs {
        if e > 0 {
            map.insert(v, e);
        }
    }
    Monomial::from_exponents(map)
}

fn poly_pow(base: &MultivariatePolynomial<Integer>, e: u32) -> MultivariatePolynomial<Integer> {
    let mut acc = MultivariatePolynomial::<Integer>::constant(Integer::one());
    for _ in 0..e {
        acc = acc * base.clone();
    }
    acc
}

/// Monic degree-`deg` form `x^deg + Σ_{i<deg} v_{base+i} · x^i`.
fn monic_form(base: usize, deg: usize) -> MultivariatePolynomial<Integer> {
    let mut f = MultivariatePolynomial::<Integer>::zero();
    f.add_term(mono(&[(X, deg as u32)]), Integer::one());
    for i in 0..deg {
        f.add_term(mono(&[(base + i, 1), (X, i as u32)]), Integer::one());
    }
    f
}

/// Build the fixed-target 25-unknown / 25-equation system
/// `[xⁱ](A²B − λR⁵S − c·x¹²) = 0`, `i = 0..24`.
///
/// Variable order matches [`unknown_names`]: `a0..a7, b0..b7, r0..r2, s0..s3,
/// lambda, c`. Equation `i` is the coefficient of `xⁱ`, so
/// `system.polynomials()[i]` is `[xⁱ](A²B − λR⁵S) − (c if i==12 else 0)`.
pub fn pinned_system_2_12_5() -> PolySystem {
    let a = monic_form(A_BASE, 8);
    let b = monic_form(B_BASE, 8);
    let s = monic_form(S_BASE, 4);

    // cubic = x^3 + r2 x^2 + r1 x + r0 ; R = (x - 1) * cubic
    let mut cubic = MultivariatePolynomial::<Integer>::zero();
    cubic.add_term(mono(&[(X, 3)]), Integer::one());
    cubic.add_term(mono(&[(R_BASE + 2, 1), (X, 2)]), Integer::one());
    cubic.add_term(mono(&[(R_BASE + 1, 1), (X, 1)]), Integer::one());
    cubic.add_term(mono(&[(R_BASE, 1)]), Integer::one());
    let mut x_minus_1 = MultivariatePolynomial::<Integer>::zero();
    x_minus_1.add_term(mono(&[(X, 1)]), Integer::one());
    x_minus_1.add_term(mono(&[]), Integer::from(-1));
    let r = x_minus_1 * cubic;

    // A^2 B - lambda * (R^5 S) - c * x^12
    let p = poly_pow(&a, 2) * b;
    let lambda = MultivariatePolynomial::<Integer>::variable(LAMBDA);
    let q = lambda * (poly_pow(&r, 5) * s);
    let mut cx12 = MultivariatePolynomial::<Integer>::zero();
    cx12.add_term(mono(&[(C, 1), (X, 12)]), Integer::one());
    let expr = p - q - cx12;

    // Bucket by x-degree, drop the x variable.
    let mut buckets: Vec<MultivariatePolynomial<Integer>> =
        vec![MultivariatePolynomial::<Integer>::zero(); NUM_EQUATIONS];
    for (m, coeff) in expr.terms() {
        let d = m.exponent(X) as usize;
        debug_assert!(d <= DEGREE, "x-degree {d} exceeds {DEGREE}");
        let mut stripped = BTreeMap::new();
        for (&v, &e) in m.iter_exponents() {
            if v != X {
                stripped.insert(v, e);
            }
        }
        buckets[d].add_term(Monomial::from_exponents(stripped), coeff.clone());
    }

    PolySystem::new(NUM_UNKNOWNS, buckets)
}

// --- parameter-homotopy contract (numeric ψ over Q) ------------------------

fn rat_poly_mul(a: &[Rational], b: &[Rational]) -> Vec<Rational> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut c = vec![Rational::from_i64(0); a.len() + b.len() - 1];
    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            c[i + j] = c[i + j].clone() + ai.clone() * bj.clone();
        }
    }
    c
}

fn rat_poly_pow(a: &[Rational], e: u32) -> Vec<Rational> {
    let mut acc = vec![Rational::from_i64(1)];
    for _ in 0..e {
        acc = rat_poly_mul(&acc, a);
    }
    acc
}

/// The parameter map `ψ`: given a concrete rational seed of the 24 solving
/// variables, return the 25 coefficients (`x⁰..x²⁴`) of `A²B − λR⁵S`.
///
/// This is an *independent* numeric implementation (dense rational convolution),
/// distinct from the symbolic [`pinned_system_2_12_5`]; agreement between the two
/// is the module's cross-check.
///
/// Coefficient conventions (all ascending in `x`, monic leading term implicit):
/// * `a0` = `[a0..a7]` (length 8) ⇒ `A = x⁸ + a₇x⁷ + … + a₀`,
/// * `b0` = `[b0..b7]` (length 8) ⇒ `B = x⁸ + …`,
/// * `r0` = `[r0,r1,r2]` (length 3) ⇒ `R = (x−1)(x³ + r₂x² + r₁x + r₀)`,
/// * `s0` = `[s0..s3]` (length 4) ⇒ `S = x⁴ + …`,
/// * `lambda0` = `λ`.
///
/// # Panics
/// Panics if any input slice has the wrong length.
pub fn psi(
    a0: &[Rational],
    b0: &[Rational],
    r0: &[Rational],
    s0: &[Rational],
    lambda0: &Rational,
) -> Vec<Rational> {
    assert_eq!(a0.len(), 8, "A needs 8 free coefficients a0..a7");
    assert_eq!(b0.len(), 8, "B needs 8 free coefficients b0..b7");
    assert_eq!(r0.len(), 3, "R needs 3 free coefficients r0,r1,r2");
    assert_eq!(s0.len(), 4, "S needs 4 free coefficients s0..s3");

    let one = Rational::from_i64(1);

    // A = x^8 + a7 x^7 + ... + a0  (dense ascending, length 9)
    let mut a = a0.to_vec();
    a.push(one.clone());
    // B = x^8 + ... (length 9)
    let mut b = b0.to_vec();
    b.push(one.clone());
    // S = x^4 + ... (length 5)
    let mut s = s0.to_vec();
    s.push(one.clone());
    // cubic = x^3 + r2 x^2 + r1 x + r0 (length 4); R = (x - 1) * cubic (length 5)
    let cubic = vec![r0[0].clone(), r0[1].clone(), r0[2].clone(), one.clone()];
    let x_minus_1 = vec![Rational::from_i64(-1), one.clone()];
    let r = rat_poly_mul(&x_minus_1, &cubic);

    // P = A^2 B ; Q = lambda * R^5 * S
    let p = rat_poly_mul(&rat_poly_pow(&a, 2), &b);
    let q_unscaled = rat_poly_mul(&rat_poly_pow(&r, 5), &s);
    let q: Vec<Rational> = q_unscaled
        .iter()
        .map(|coef| coef.clone() * lambda0.clone())
        .collect();

    // psi = P - Q, padded to 25 coefficients (x^0 .. x^24)
    (0..NUM_EQUATIONS)
        .map(|k| {
            let pk = p.get(k).cloned().unwrap_or_else(|| Rational::from_i64(0));
            let qk = q.get(k).cloned().unwrap_or_else(|| Rational::from_i64(0));
            pk - qk
        })
        .collect()
}

/// The target parameter vector `p* = coeffs(x¹²) = (0,…,0,1,0,…,0)` (length 25,
/// the `1` at degree 12). The scaling freedom is used to absorb `c ↦ 1`, so the
/// homotopy solves `ψ(A,B,R,S,λ) = p*`.
pub fn p_star() -> Vec<Rational> {
    let mut v = vec![Rational::from_i64(0); NUM_EQUATIONS];
    v[12] = Rational::from_i64(1);
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_core::Ring; // for Rational::is_zero in assertions

    #[test]
    fn system_shape_and_degrees() {
        let sys = pinned_system_2_12_5();
        assert_eq!(sys.num_variables(), NUM_UNKNOWNS);
        assert_eq!(sys.num_equations(), NUM_EQUATIONS);

        // Total degrees: A²B contributes ≤ 3, λR⁵S contributes ≤ 7, c ≤ 1.
        let degs: Vec<usize> = sys.polynomials().iter().map(|p| p.total_degree()).collect();
        assert!(degs.iter().all(|&d| d <= 7), "some equation exceeds degree 7");
        assert_eq!(*degs.iter().max().unwrap(), 7, "max equation degree should be 7");

        // Leading equation x^24: coeff = 1 - lambda, total degree 1.
        assert_eq!(sys.polynomials()[24].total_degree(), 1);
    }

    #[test]
    fn p_star_is_x12_monomial() {
        let ps = p_star();
        assert_eq!(ps.len(), NUM_EQUATIONS);
        for (i, v) in ps.iter().enumerate() {
            if i == 12 {
                assert_eq!(*v, Rational::from_i64(1));
            } else {
                assert!(v.is_zero(), "p_star[{i}] should be zero");
            }
        }
    }

    #[test]
    fn solving_var_count() {
        assert_eq!(solving_var_names().len(), NUM_SOLVING_VARS);
        assert_eq!(unknown_names().len(), NUM_UNKNOWNS);
        assert!(!solving_var_names().iter().any(|n| n == "c"));
    }

    /// The core validation: ψ on a random rational seed reproduces the fixed-target
    /// system's left-hand-side coefficients (evaluated at c = 0), i.e. the numeric
    /// convolution and the symbolic construction agree coefficient-by-coefficient.
    #[test]
    fn psi_reproduces_fixed_target_lhs() {
        let ri = Rational::from_i64;
        // A random rational seed for the 24 solving variables.
        let a0 = [ri(1), ri(-2), ri(3), ri(0), ri(-1), ri(2), ri(1), ri(-3)];
        let b0 = [ri(2), ri(1), ri(-1), ri(3), ri(0), ri(-2), ri(1), ri(1)];
        let r0 = [ri(-1), ri(2), ri(1)];
        let s0 = [ri(3), ri(-2), ri(1), ri(2)];
        let lambda0 = Rational::new(3, 2).unwrap();

        let psi_coeffs = psi(&a0, &b0, &r0, &s0, &lambda0);
        assert_eq!(psi_coeffs.len(), NUM_EQUATIONS);

        // Evaluation point for the symbolic system: unknown order a0..a7, b0..b7,
        // r0..r2, s0..s3, lambda, c — with c = 0 so the system LHS equals ψ.
        let mut point: Vec<Rational> = Vec::with_capacity(NUM_UNKNOWNS);
        point.extend_from_slice(&a0);
        point.extend_from_slice(&b0);
        point.extend_from_slice(&r0);
        point.extend_from_slice(&s0);
        point.push(lambda0.clone());
        point.push(ri(0)); // c = 0

        let sys = pinned_system_2_12_5();
        let residual = sys.evaluate(&point);
        assert_eq!(residual.len(), NUM_EQUATIONS);
        for i in 0..NUM_EQUATIONS {
            assert_eq!(residual[i], psi_coeffs[i], "mismatch at x^{i}");
        }
    }

    /// A sanity check on the homotopy contract: if a seed happens to satisfy
    /// ψ(seed) = p*, then that seed is an exact solution of the fixed-target
    /// system with c = 1. We do not have a closed-form seed here, so we instead
    /// verify the algebraic identity linking ψ, p*, and the system: for any seed,
    /// system.evaluate(seed with c) [i] = ψ(seed)[i] − (c if i==12 else 0).
    #[test]
    fn system_equals_psi_minus_c_x12() {
        let ri = Rational::from_i64;
        let a0 = [ri(0), ri(1), ri(-1), ri(2), ri(1), ri(0), ri(-2), ri(1)];
        let b0 = [ri(1), ri(-1), ri(2), ri(0), ri(1), ri(-1), ri(3), ri(-2)];
        let r0 = [ri(2), ri(-1), ri(3)];
        let s0 = [ri(-1), ri(1), ri(2), ri(-3)];
        let lambda0 = ri(2);
        let c = ri(5);

        let psi_coeffs = psi(&a0, &b0, &r0, &s0, &lambda0);

        let mut point: Vec<Rational> = Vec::new();
        point.extend_from_slice(&a0);
        point.extend_from_slice(&b0);
        point.extend_from_slice(&r0);
        point.extend_from_slice(&s0);
        point.push(lambda0);
        point.push(c.clone());

        let residual = pinned_system_2_12_5().evaluate(&point);
        for i in 0..NUM_EQUATIONS {
            let expected = if i == 12 {
                psi_coeffs[i].clone() - c.clone()
            } else {
                psi_coeffs[i].clone()
            };
            assert_eq!(residual[i], expected, "mismatch at x^{i}");
        }
    }
}
