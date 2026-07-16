//! The **symbolic deleted-sheet resolvent** `R(X, u)` over `Q[u]` — the regular
//! (one-parameter) form of the `1 + (n−1)` fibre split.
//!
//! For a degree-`n` rational map `φ = P/Q` the bivariate numerator
//!
//! ```text
//!     N(X, u) = P(X)·Q(u) − P(u)·Q(X)
//! ```
//!
//! vanishes identically on the diagonal `X = u`, so `(X − u)` divides it
//! **exactly** in `Q[X, u]`. The quotient
//!
//! ```text
//!     R(X, u) = N(X, u) / (X − u)
//! ```
//!
//! is a bivariate polynomial, symmetric in `X ↔ u` (both `N` and `X − u` are
//! antisymmetric), with `deg_X R = deg_u R = n − 1` whenever `P/Q` is a
//! non-constant map of degree `n = max(deg P, deg Q)`. For the degree-24 Belyi
//! map this is the degree-23 object of the regular statement: its Galois group
//! over `Q(u)` is the point stabilizer (M23 for the M24 cover).
//!
//! ## The `u ↔ x₀` pairing (derived from [`super::audit`])
//!
//! The numeric route in [`super::audit::audit_m23_residual`] picks a rational
//! **source** point `x₀`, sets `t₀ = φ(x₀)` ([`super::audit::phi_at`]), forms the
//! degree-24 numerator `f_{t₀}(X) = P(X) − t₀·Q(X)` of `φ − t₀`
//! ([`super::audit::specialize_numerator`], up to primitivization), and strips
//! the linear factor `(X − x₀)`. The symbolic route specializes `u → x₀` here.
//! Since `P(x₀) = t₀·Q(x₀)`,
//!
//! ```text
//!     R(X, x₀) = ( P(X)·Q(x₀) − P(x₀)·Q(X) ) / (X − x₀)
//!              = Q(x₀) · ( P(X) − t₀·Q(X) ) / (X − x₀),
//! ```
//!
//! i.e. **specializing `u → x₀` (a point on the source curve, not a base point)
//! gives `Q(x₀)` times the deleted-sheet fibre polynomial over `t₀ = φ(x₀)`** —
//! the two routes agree up to the nonzero rational scalar `Q(x₀)` (`x₀` not a
//! pole), hence exactly after primitivization. This pairing needs no
//! root-finding: `x₀` is chosen rationally and `t₀ = φ(x₀)` is computed, never
//! the other way round. The cross-check is the acceptance test below.

use rustmath_polynomials::UnivariatePolynomial;
use rustmath_rationals::Rational;

/// The resolvent `R(X, u) ∈ Q[X, u]`, stored as the vector of its `X`-power
/// coefficients: `coeff_x[i]` is the coefficient of `Xⁱ`, a univariate
/// polynomial in `u` (ascending coefficients).
///
/// Invariants established by [`deleted_sheet_resolvent`]:
/// * the synthetic division of `N(X, u)` by `(X − u)` left an **exactly zero**
///   remainder in `Q[u]` (never rounded);
/// * `coeff_x.len() == deg_x + 1` with a nonzero leading `X`-coefficient
///   `coeff_x[deg_x](u) = lc(P)·Q(u) − lc_n(Q)·P(u)` (where `lc_n` pads to
///   degree `n`), and `deg_x = max(deg P, deg Q) − 1`.
#[derive(Debug, Clone)]
pub struct DeletedSheetResolvent {
    coeff_x: Vec<UnivariatePolynomial<Rational>>,
}

impl DeletedSheetResolvent {
    /// The degree in `X` (`= max(deg P, deg Q) − 1`; `23` for the degree-24 map).
    pub fn deg_x(&self) -> usize {
        self.coeff_x.len() - 1
    }

    /// The degree in `u` — equals [`Self::deg_x`] by the `X ↔ u` symmetry.
    pub fn deg_u(&self) -> usize {
        self.coeff_x
            .iter()
            .filter_map(|c| c.degree())
            .max()
            .unwrap_or(0)
    }

    /// All `X`-power coefficients, ascending: entry `i` is `[Xⁱ] R ∈ Q[u]`.
    pub fn coeffs(&self) -> &[UnivariatePolynomial<Rational>] {
        &self.coeff_x
    }

    /// The coefficient of `Xⁱ` as a polynomial in `u`.
    pub fn coeff_of_x(&self, i: usize) -> &UnivariatePolynomial<Rational> {
        &self.coeff_x[i]
    }

    /// Specialize `u → u₀`: the univariate `R(X, u₀) ∈ Q[X]`, ascending, always
    /// of length `deg_x + 1` (**not** trimmed: the leading coefficient
    /// `lc(P)·Q(u₀) − lc_n(Q)·P(u₀)` vanishes exactly when `φ(u₀)` hits the
    /// degree-drop value, e.g. `t₀ = 1/λ` for the pinned Belyi map — callers
    /// screen by degree, as `audit` does).
    ///
    /// When `u₀ = x₀` is a rational source point that is not a pole of `φ`,
    /// `R(X, x₀) = Q(x₀)·(P(X) − φ(x₀)·Q(X))/(X − x₀)` — the deleted-sheet
    /// fibre polynomial over `t₀ = φ(x₀)`, up to the scalar `Q(x₀)` (see the
    /// module docs for the derivation from `audit`).
    pub fn specialize(&self, u0: &Rational) -> Vec<Rational> {
        self.coeff_x.iter().map(|c| c.evaluate(u0)).collect()
    }
}

/// Trim trailing zero coefficients (ascending representation).
fn trim(p: &[Rational]) -> &[Rational] {
    let mut n = p.len();
    while n > 0 && p[n - 1] == Rational::from_i64(0) {
        n -= 1;
    }
    &p[..n]
}

/// Compute the deleted-sheet resolvent `R(X, u) = (P(X)Q(u) − P(u)Q(X))/(X − u)`
/// for `φ = P/Q`, given `p`, `q` as dense **ascending** rational coefficient
/// vectors.
///
/// The coefficient of `Xⁱ` in `N(X, u) = P(X)Q(u) − P(u)Q(X)` is
/// `Nᵢ(u) = pᵢ·Q(u) − qᵢ·P(u)`; the division by `(X − u)` is synthetic division
/// in the `X` variable over `Q[u]`, and the remainder — identically
/// `N(u, u) = 0` when the arithmetic is exact — is **asserted** to be the zero
/// polynomial (an `Err`, never a rounding, if it is not).
///
/// Honest degree contract: `deg_X R = max(deg P, deg Q) − 1` always holds for a
/// non-degenerate input, so for the degree-24 Belyi map the result has
/// `deg_x() == 23`; there is no special-casing by degree. Errors (no resolvent
/// exists as a nonzero polynomial):
/// * `P` or `Q` is the zero polynomial;
/// * `P` and `Q` are proportional (φ constant — `N ≡ 0`, `R ≡ 0`), which
///   includes the case where both are constants.
pub fn deleted_sheet_resolvent(
    p: &[Rational],
    q: &[Rational],
) -> Result<DeletedSheetResolvent, String> {
    let p = trim(p);
    let q = trim(q);
    if p.is_empty() {
        return Err("P is the zero polynomial: φ ≡ 0 is constant, R ≡ 0".into());
    }
    if q.is_empty() {
        return Err("Q is the zero polynomial: φ is undefined".into());
    }
    // n = max(deg P, deg Q) = deg_X N generically; the map degree.
    let n = p.len().max(q.len()) - 1;

    let p_of_u = UnivariatePolynomial::new(p.to_vec());
    let q_of_u = UnivariatePolynomial::new(q.to_vec());
    let zero = Rational::from_i64(0);

    // Nᵢ(u) = pᵢ·Q(u) − qᵢ·P(u), for i = 0..=n.
    let n_coeffs: Vec<UnivariatePolynomial<Rational>> = (0..=n)
        .map(|i| {
            let pi = p.get(i).unwrap_or(&zero);
            let qi = q.get(i).unwrap_or(&zero);
            q_of_u.scalar_mul(pi) - p_of_u.scalar_mul(qi)
        })
        .collect();

    // N_n(u) = lc(P)·Q(u) − lc_n(Q)·P(u) ≡ 0 ⟺ P ∥ Q ⟺ N ≡ 0 ⟺ R ≡ 0.
    if n_coeffs[n].is_zero() {
        return Err(
            "P and Q are proportional: φ is constant and the deleted-sheet \
             resolvent vanishes identically"
                .into(),
        );
    }

    // Synthetic division of Σ Nᵢ(u)·Xⁱ by (X − u) over Q[u]:
    // c_{n−1} = N_n;  c_{k−1} = N_k + u·c_k;  remainder = N_0 + u·c_0.
    let mut quot: Vec<UnivariatePolynomial<Rational>> =
        vec![UnivariatePolynomial::zero(); n];
    let mut carry = n_coeffs[n].clone();
    for k in (0..n).rev() {
        quot[k] = carry.clone();
        carry = n_coeffs[k].clone() + carry.shift(1); // + u·carry
    }
    if !carry.is_zero() {
        // Mathematically impossible for exact input (N(u,u) ≡ 0); reaching this
        // means corrupted input or an arithmetic bug — refuse, never round.
        return Err(format!(
            "(X − u) does not divide P(X)Q(u) − P(u)Q(X) exactly: nonzero \
             remainder of u-degree {:?} — exact-arithmetic invariant violated",
            carry.degree()
        ));
    }

    Ok(DeletedSheetResolvent { coeff_x: quot })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belyi::audit::{phi_at, specialize_numerator};
    use rustmath_integers::Integer;
    use rustmath_polynomials::zx;

    fn ri(n: i64) -> Rational {
        Rational::from_i64(n)
    }
    fn rq(n: i64, d: i64) -> Rational {
        Rational::new(n, d).unwrap()
    }
    fn rvec(v: &[i64]) -> Vec<Rational> {
        v.iter().map(|&n| ri(n)).collect()
    }

    // -- dense ascending rational helpers (test-local; audit's are private) ----

    fn rat_mul(a: &[Rational], b: &[Rational]) -> Vec<Rational> {
        let mut c = vec![ri(0); a.len() + b.len() - 1];
        for (i, ai) in a.iter().enumerate() {
            for (j, bj) in b.iter().enumerate() {
                c[i + j] = c[i + j].clone() + ai.clone() * bj.clone();
            }
        }
        c
    }

    fn rat_pow(a: &[Rational], e: u32) -> Vec<Rational> {
        let mut acc = vec![ri(1)];
        for _ in 0..e {
            acc = rat_mul(&acc, a);
        }
        acc
    }

    fn rat_scale(a: &[Rational], k: &Rational) -> Vec<Rational> {
        a.iter().map(|c| c.clone() * k.clone()).collect()
    }

    /// Synthetic division by (x − x0), as in audit.rs.
    fn div_linear(poly: &[Rational], x0: &Rational) -> (Vec<Rational>, Rational) {
        let n = poly.len();
        let mut q = vec![ri(0); n - 1];
        let mut carry = poly[n - 1].clone();
        for k in (0..n - 1).rev() {
            q[k] = carry.clone();
            carry = poly[k].clone() + x0.clone() * carry;
        }
        (q, carry)
    }

    /// Clear to primitive integer form (positive lc, content 1), as in audit.rs.
    fn clear_primitive(f: &[Rational]) -> Vec<Integer> {
        let mut den = Integer::one();
        for c in f {
            den = den.lcm(c.denominator());
        }
        let ints: Vec<Integer> = f
            .iter()
            .map(|c| {
                let scale = den.clone() / c.denominator().clone();
                c.numerator().clone() * scale
            })
            .collect();
        zx::normalize(&ints)
    }

    /// The bivariate coefficient [Xⁱ uʲ] of the resolvent.
    fn biv_coeff(r: &DeletedSheetResolvent, i: usize, j: usize) -> Rational {
        r.coeffs()
            .get(i)
            .and_then(|c| c.coefficients().get(j).cloned())
            .unwrap_or_else(|| ri(0))
    }

    /// Verify the defining identity (X − u)·R(X, u) == P(X)Q(u) − P(u)Q(X)
    /// coefficient-by-coefficient in Q[u] — a complete exactness proof of the
    /// division for this input. [Xⁱ] of the left side is R_{i−1}(u) − u·Rᵢ(u).
    fn assert_product_identity(r: &DeletedSheetResolvent, p: &[Rational], q: &[Rational]) {
        let p_of_u = UnivariatePolynomial::new(p.to_vec());
        let q_of_u = UnivariatePolynomial::new(q.to_vec());
        let zero_r = ri(0);
        let n = r.deg_x() + 1;
        for i in 0..=n {
            let r_prev = if i > 0 {
                r.coeff_of_x(i - 1).clone()
            } else {
                UnivariatePolynomial::zero()
            };
            let r_cur = if i < n {
                r.coeff_of_x(i).clone()
            } else {
                UnivariatePolynomial::zero()
            };
            let lhs = r_prev - r_cur.shift(1);
            let pi = p.get(i).unwrap_or(&zero_r);
            let qi = q.get(i).unwrap_or(&zero_r);
            let rhs = q_of_u.scalar_mul(pi) - p_of_u.scalar_mul(qi);
            assert!(
                (lhs - rhs).is_zero(),
                "product identity fails at X^{i}"
            );
        }
    }

    fn assert_symmetric(r: &DeletedSheetResolvent) {
        let d = r.deg_x();
        for i in 0..=d {
            for j in 0..=d {
                assert_eq!(
                    biv_coeff(r, i, j),
                    biv_coeff(r, j, i),
                    "R is not symmetric at (X^{i}, u^{j})"
                );
            }
        }
    }

    // ------------------------------------------------------------------
    // Small synthetic map: P = 2X⁴ + 3X³ − X + 5, Q = X³ − 2X² + 7.
    // Full oracle from sympy: expand((P(X)Q(u) − P(u)Q(X))/(X−u)); the
    // division left remainder 0 and (ascending in u, per X-power):
    //   [X⁰] = −7 + 10u + 16u² + 14u³
    //   [X¹] = 10 + 14u + 15u²
    //   [X²] = 16 + 15u − 6u² − 4u³
    //   [X³] = 14      − 4u² + 2u³
    // ------------------------------------------------------------------
    fn small_p() -> Vec<Rational> {
        rvec(&[5, -1, 0, 3, 2])
    }
    fn small_q() -> Vec<Rational> {
        rvec(&[7, 0, -2, 1])
    }

    #[test]
    fn small_example_matches_sympy_oracle_exhaustively() {
        let r = deleted_sheet_resolvent(&small_p(), &small_q()).unwrap();
        assert_eq!(r.deg_x(), 3, "deg_X = max(4, 3) − 1");
        assert_eq!(r.deg_u(), 3);
        let expect: [&[i64]; 4] = [
            &[-7, 10, 16, 14],
            &[10, 14, 15],
            &[16, 15, -6, -4],
            &[14, 0, -4, 2],
        ];
        for (i, exp) in expect.iter().enumerate() {
            assert_eq!(
                r.coeff_of_x(i).coefficients(),
                &rvec(exp)[..],
                "sympy oracle mismatch at X^{i}"
            );
        }
    }

    #[test]
    fn small_example_product_identity_and_symmetry() {
        let r = deleted_sheet_resolvent(&small_p(), &small_q()).unwrap();
        assert_product_identity(&r, &small_p(), &small_q());
        assert_symmetric(&r);
    }

    #[test]
    fn small_example_specialization_matches_numeric_route() {
        // For rational source points x0, R(X, x0) must equal
        // Q(x0)·(P(X) − t0·Q(X))/(X − x0) with t0 = P(x0)/Q(x0); after
        // primitivization the two routes are identical.
        let p = small_p();
        let q = small_q();
        let r = deleted_sheet_resolvent(&p, &q).unwrap();
        let eval = |f: &[Rational], x: &Rational| {
            let mut acc = ri(0);
            for c in f.iter().rev() {
                acc = acc * x.clone() + c.clone();
            }
            acc
        };
        let zero = ri(0);
        for x0 in [ri(2), ri(-1), rq(1, 2), rq(-3, 5), ri(7)] {
            let qx0 = eval(&q, &x0);
            assert!(qx0 != ri(0), "chosen x0 must not be a pole");
            let t0 = eval(&p, &x0) / qx0.clone();
            // numeric route: P − t0·Q, strip (X − x0)
            let num: Vec<Rational> = p
                .iter()
                .zip(q.iter().chain(std::iter::repeat(&zero)))
                .map(|(pi, qi)| pi.clone() - t0.clone() * qi.clone())
                .collect();
            let (quot, rem) = div_linear(&num, &x0);
            assert_eq!(rem, ri(0));
            // symbolic route
            let spec = r.specialize(&x0);
            // exact scalar relation: spec == Q(x0) · quot
            let scaled: Vec<Rational> = quot.iter().map(|c| c.clone() * qx0.clone()).collect();
            assert_eq!(spec, scaled, "R(X, x0) != Q(x0)·(P − t0·Q)/(X − x0) at x0 = {x0:?}");
            assert_eq!(clear_primitive(&spec), clear_primitive(&quot));
        }
    }

    // ------------------------------------------------------------------
    // Degree-24-shaped sparse example:
    //   P = X²⁴ + 3X¹⁷ − 5X³ + 7,  Q = 2X²² − X⁵ + 11.
    // sympy oracle: remainder 0, deg_X = deg_u = 23, symmetric, 118 nonzero
    // terms; the sampled coefficients and the u = 3/2 specialization below
    // were computed independently in sympy.
    // ------------------------------------------------------------------
    fn sparse_p() -> Vec<Rational> {
        let mut p = vec![ri(0); 25];
        p[0] = ri(7);
        p[3] = ri(-5);
        p[17] = ri(3);
        p[24] = ri(1);
        p
    }
    fn sparse_q() -> Vec<Rational> {
        let mut q = vec![ri(0); 23];
        q[0] = ri(11);
        q[5] = ri(-1);
        q[22] = ri(2);
        q
    }

    #[test]
    fn deg24_sparse_degree_symmetry_and_sympy_samples() {
        let r = deleted_sheet_resolvent(&sparse_p(), &sparse_q()).unwrap();
        assert_eq!(r.deg_x(), 23, "degree-24 map gives deg_X = 23");
        assert_eq!(r.deg_u(), 23);
        assert_product_identity(&r, &sparse_p(), &sparse_q());
        assert_symmetric(&r);
        // sympy-sampled bivariate coefficients [X^i u^j]
        let samples: &[(usize, usize, i64)] = &[
            (23, 0, 11),
            (0, 23, 11),
            (22, 1, 11),
            (12, 12, 10),
            (21, 17, -6),
            (4, 0, 7),
            (16, 5, -17),
            (13, 8, -17),
            (2, 14, 33),
            (14, 9, 11),
            (7, 21, -1),
            (1, 3, 7),
            (16, 3, 0),
            (9, 9, 0),
        ];
        for &(i, j, c) in samples {
            assert_eq!(biv_coeff(&r, i, j), ri(c), "sympy oracle mismatch at [X^{i} u^{j}]");
        }
    }

    #[test]
    fn deg24_sparse_specialization_matches_sympy() {
        // sympy: R(X, 3/2) ascending coefficients 0..4 and the X²³ coefficient.
        let r = deleted_sheet_resolvent(&sparse_p(), &sparse_q()).unwrap();
        let spec = r.specialize(&rq(3, 2));
        assert_eq!(spec.len(), 24);
        assert_eq!(spec[0], rq(630883718289, 8388608));
        assert_eq!(spec[1], rq(210294572763, 4194304));
        assert_eq!(spec[2], rq(70098190921, 2097152));
        assert_eq!(spec[3], rq(37839867681, 524288));
        assert_eq!(spec[4], rq(12613289227, 262144));
        assert_eq!(spec[23], rq(31388203033, 2097152));
    }

    // ------------------------------------------------------------------
    // THE ACCEPTANCE GATE: cross-validate against audit.rs's numeric route
    // on the pinned degree-24 Belyi shape. Same concrete (non-solution)
    // 25-vector as audit's own tests; P = A²B, Q = λR⁵S rebuilt with the
    // exact layout of audit::extract (a₀..a₇, b₀..b₇, r₀..r₂, s₀..s₃, λ, c;
    // R = (x−1)·(x³ + r₂x² + r₁x + r₀); A, B, S monic).
    // ------------------------------------------------------------------
    fn sample_coeffs() -> Vec<Rational> {
        let mut v: Vec<Rational> = Vec::new();
        v.extend([1, -2, 3, 0, -1, 2, 1, -3].iter().map(|&n| ri(n))); // a0..a7
        v.extend([2, 1, -1, 3, 0, -2, 1, 1].iter().map(|&n| ri(n))); // b0..b7
        v.extend([-1, 2, 1].iter().map(|&n| ri(n))); // r0..r2
        v.extend([3, -2, 1, 2].iter().map(|&n| ri(n))); // s0..s3
        v.push(rq(3, 2)); // lambda
        v.push(ri(1)); // c
        v
    }

    fn belyi_p_and_q(coeffs: &[Rational]) -> (Vec<Rational>, Vec<Rational>) {
        let one = ri(1);
        let mut a = coeffs[0..8].to_vec();
        a.push(one.clone());
        let mut b = coeffs[8..16].to_vec();
        b.push(one.clone());
        let mut s = coeffs[19..23].to_vec();
        s.push(one.clone());
        let cubic = vec![
            coeffs[16].clone(),
            coeffs[17].clone(),
            coeffs[18].clone(),
            one.clone(),
        ];
        let x_minus_1 = vec![ri(-1), one];
        let r = rat_mul(&x_minus_1, &cubic);
        let p = rat_mul(&rat_pow(&a, 2), &b); // A²B, degree 24, monic
        let q = rat_scale(&rat_mul(&rat_pow(&r, 5), &s), &coeffs[23]); // λR⁵S
        (p, q)
    }

    #[test]
    fn symbolic_route_agrees_with_audit_numeric_route_on_many_x0() {
        let coeffs = sample_coeffs();
        let (p, q) = belyi_p_and_q(&coeffs);
        let resolvent = deleted_sheet_resolvent(&p, &q).unwrap();
        assert_eq!(resolvent.deg_x(), 23, "degree-24 Belyi map: deg_X R = 23");
        assert_eq!(resolvent.deg_u(), 23);

        // Various heights and signs; x0 = 1 (the rational pole, R(1) = 0) excluded.
        let x0s: Vec<Rational> = vec![
            ri(2),
            ri(-2),
            rq(1, 2),
            rq(-1, 2),
            ri(3),
            ri(-3),
            rq(5, 3),
            rq(-7, 2),
            rq(22, 7),
            rq(-11, 4),
            ri(10),
            rq(101, 13),
        ];
        let mut checked = 0usize;
        for x0 in &x0s {
            // numeric route (audit.rs): t0 = φ(x0), f = primitive numerator of
            // φ − t0, strip (X − x0), primitivize.
            let t0 = match phi_at(&coeffs, x0) {
                Some(t) => t,
                None => continue, // pole
            };
            let f_int = specialize_numerator(&coeffs, &t0);
            if zx::degree(&f_int) != 24 {
                continue; // degenerate t0 (= 1/λ); both routes drop together
            }
            let f_rat: Vec<Rational> = f_int
                .iter()
                .map(|c| Rational::from_integer(c.clone()))
                .collect();
            let (quot, rem) = div_linear(&f_rat, x0);
            assert_eq!(rem, ri(0), "x0 must be a root of the specialized numerator");
            let g_num = clear_primitive(&quot);
            assert_eq!(zx::degree(&g_num), 23);

            // symbolic route: specialize u → x0, primitivize.
            let g_sym = clear_primitive(&resolvent.specialize(x0));

            // zx::normalize fixes content 1 and positive leading coefficient, so
            // "equal up to a nonzero rational scalar and sign" becomes equality.
            assert_eq!(g_num, g_sym, "routes disagree at x0 = {x0:?}");
            checked += 1;
        }
        assert!(
            checked >= 10,
            "acceptance gate needs at least 10 cross-checked x0 (got {checked})"
        );
    }

    // ------------------------------------------------------------------
    // Honest degenerate-input handling.
    // ------------------------------------------------------------------
    #[test]
    fn degenerate_inputs_are_errors() {
        let p = small_p();
        // Q = 0
        assert!(deleted_sheet_resolvent(&p, &[]).is_err());
        assert!(deleted_sheet_resolvent(&p, &rvec(&[0, 0])).is_err());
        // P = 0
        assert!(deleted_sheet_resolvent(&rvec(&[0]), &p).is_err());
        // proportional: P = −3·Q (φ constant, R ≡ 0)
        let q3 = rat_scale(&p, &ri(-3));
        assert!(deleted_sheet_resolvent(&q3, &p).is_err());
        // both constants (also proportional)
        assert!(deleted_sheet_resolvent(&rvec(&[5]), &rvec(&[7])).is_err());
        // trailing zeros must not fool the degree logic: pad P with zeros
        let mut padded = p.clone();
        padded.extend([ri(0), ri(0)]);
        let r1 = deleted_sheet_resolvent(&padded, &small_q()).unwrap();
        assert_eq!(r1.deg_x(), 3);
    }
}
