//! Function fields `K = ℚ(t)[x] / (F)` and specialization `t ↦ a ∈ ℚ`.

use crate::ratfunc::RationalFunction;
use rustmath_polynomials::UnivariatePolynomial;
use rustmath_rationals::Rational;
use std::fmt;

/// A polynomial in `x` with coefficients in `ℚ(t)`, i.e. an element of `ℚ(t)[x]`.
pub type FfPoly = UnivariatePolynomial<RationalFunction>;

/// A polynomial in `x` with coefficients in `ℚ`, i.e. an element of `ℚ[x]`.
/// This is what a defining polynomial specializes to.
pub type QxPoly = UnivariatePolynomial<Rational>;

/// Outcome of specializing a defining polynomial `F(t, x)` at `t = a`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Specialization {
    /// `a` is a *good* place: no coefficient pole, degree preserved, and the
    /// specialized polynomial is separable (square-free). Carries `F(a, x)`.
    Good(QxPoly),
    /// `a` is a pole of some coefficient of `F` (denominator vanished).
    Pole,
    /// The leading coefficient vanished at `a`, dropping the degree.
    DegreeDrop(QxPoly),
    /// `F(a, x)` has a repeated root (not separable), e.g. a branch point.
    NotSeparable(QxPoly),
}

impl Specialization {
    /// The specialized polynomial if one was produced (everything except `Pole`).
    pub fn polynomial(&self) -> Option<&QxPoly> {
        match self {
            Specialization::Good(p)
            | Specialization::DegreeDrop(p)
            | Specialization::NotSeparable(p) => Some(p),
            Specialization::Pole => None,
        }
    }

    /// Whether this is a good (degree-preserving, separable) specialization.
    pub fn is_good(&self) -> bool {
        matches!(self, Specialization::Good(_))
    }
}

/// A function field `K = ℚ(t)[x] / (F(t, x))`.
///
/// `F` must be irreducible over `ℚ(t)` for `K` to be a field; the constructor
/// [`FunctionField::new`] checks this (and rejects non-irreducible `F`), while
/// [`FunctionField::new_unchecked`] skips the check for callers that already
/// know `F` is irreducible.
#[derive(Clone)]
pub struct FunctionField {
    /// The defining polynomial `F ∈ ℚ(t)[x]`, kept monic in `x`.
    defining: FfPoly,
}

impl FunctionField {
    /// Build `K = ℚ(t)[x]/(F)`, verifying that `F` is irreducible over `ℚ(t)`.
    ///
    /// Returns `Err` with a message if `F` is constant, the zero polynomial, or
    /// reducible. `F` is made monic in `x` before being stored.
    pub fn new(defining: FfPoly) -> Result<Self, String> {
        match defining.degree() {
            None => return Err("defining polynomial is zero".into()),
            Some(0) => return Err("defining polynomial is constant".into()),
            Some(_) => {}
        }
        let monic = defining.make_monic();
        if !crate::factor::is_irreducible_over_qt(&monic)? {
            return Err("defining polynomial is reducible over ℚ(t)".into());
        }
        Ok(FunctionField { defining: monic })
    }

    /// Build `K` from an `F` already known to be irreducible, skipping the check.
    /// `F` is still made monic in `x`.
    pub fn new_unchecked(defining: FfPoly) -> Self {
        FunctionField {
            defining: defining.make_monic(),
        }
    }

    /// The (monic) defining polynomial `F ∈ ℚ(t)[x]`.
    pub fn defining_polynomial(&self) -> &FfPoly {
        &self.defining
    }

    /// The degree `[K : ℚ(t)] = deg_x F`.
    pub fn degree(&self) -> usize {
        self.defining.degree().unwrap_or(0)
    }

    /// Specialize the defining polynomial at `t = a ∈ ℚ`, classifying the result
    /// (see [`Specialization`]). A `Good` result means `F(a, x) ∈ ℚ[x]` defines a
    /// number field of the same degree whose Galois group is (generically) the
    /// arithmetic monodromy of `K/ℚ(t)` — the workhorse of regular-cover
    /// specialization.
    pub fn specialize(&self, a: &Rational) -> Specialization {
        specialize_poly(&self.defining, a)
    }
}

impl fmt::Display for FunctionField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ℚ(t)[x] / ({})", self.defining)
    }
}

impl fmt::Debug for FunctionField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FunctionField(deg {}, F = {:?})", self.degree(), self.defining)
    }
}

/// Specialize an arbitrary `F ∈ ℚ(t)[x]` at `t = a`, returning the classified
/// outcome. The original (highest) `x`-degree of `F` is used as the reference
/// degree, so a vanishing leading coefficient is reported as `DegreeDrop`.
pub fn specialize_poly(f: &FfPoly, a: &Rational) -> Specialization {
    let orig_deg = match f.degree() {
        Some(d) => d,
        None => return Specialization::Good(QxPoly::zero()),
    };

    // Evaluate each ℚ(t) coefficient at t = a; any pole is fatal.
    let mut q_coeffs: Vec<Rational> = Vec::with_capacity(orig_deg + 1);
    for i in 0..=orig_deg {
        match f.coeff(i).evaluate(a) {
            Some(v) => q_coeffs.push(v),
            None => return Specialization::Pole,
        }
    }

    let specialized = QxPoly::new(q_coeffs);

    // Degree dropped iff the new degree is below the original.
    if specialized.degree() != Some(orig_deg) {
        return Specialization::DegreeDrop(specialized);
    }

    // Separability: square-free over ℚ.
    if !specialized.is_square_free() {
        return Specialization::NotSeparable(specialized);
    }

    Specialization::Good(specialized)
}

/// Convenience: build `F ∈ ℚ(t)[x]` from per-`x`-degree `ℚ(t)` coefficients
/// (constant term first).
pub fn ff_poly_from_coeffs(coeffs: Vec<RationalFunction>) -> FfPoly {
    UnivariatePolynomial::new(coeffs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ratfunc::QtPoly;
    use rustmath_core::Ring;

    fn q(n: i64) -> Rational {
        Rational::from_i64(n)
    }

    /// Build F = x^2 - t^2  ∈ ℚ(t)[x].
    fn x2_minus_t2() -> FfPoly {
        let t = RationalFunction::t();
        let minus_t2 = RationalFunction::zero() - (t.clone() * t);
        ff_poly_from_coeffs(vec![minus_t2, RationalFunction::zero(), RationalFunction::one()])
    }

    #[test]
    fn degree_of_field() {
        // x^2 - (t^3 + t + 1): irreducible (a constant in ℚ(t) is never a square
        // there unless it is one), degree 2.
        let t = RationalFunction::t();
        let c = t.clone() * t.clone() * t.clone() + t.clone() + RationalFunction::one();
        let f = ff_poly_from_coeffs(vec![
            RationalFunction::zero() - c,
            RationalFunction::zero(),
            RationalFunction::one(),
        ]);
        let k = FunctionField::new(f).unwrap();
        assert_eq!(k.degree(), 2);
    }

    #[test]
    fn good_specialization() {
        let f = x2_minus_t2();
        // At t = 3: x^2 - 9, separable, degree 2.
        match specialize_poly(&f, &q(3)) {
            Specialization::Good(p) => {
                assert_eq!(p.degree(), Some(2));
                assert_eq!(p.coefficients(), &[q(-9), q(0), q(1)]);
            }
            other => panic!("expected Good, got {:?}", other),
        }
    }

    #[test]
    fn separability_check_at_branch_point() {
        // x^2 - t^2 at t = 0 is x^2: repeated root, not separable.
        let f = x2_minus_t2();
        match specialize_poly(&f, &q(0)) {
            Specialization::NotSeparable(p) => {
                assert_eq!(p.coefficients(), &[q(0), q(0), q(1)]);
            }
            other => panic!("expected NotSeparable, got {:?}", other),
        }
    }

    #[test]
    fn pole_specialization() {
        // F = x - 1/(t-2): coefficient pole at t = 2.
        let den = UnivariatePolynomial::new(vec![q(-2), q(1)]);
        let one_over = RationalFunction::new(QtPoly::one(), den).unwrap();
        let f = ff_poly_from_coeffs(vec![
            RationalFunction::zero() - one_over,
            RationalFunction::one(),
        ]);
        assert_eq!(specialize_poly(&f, &q(2)), Specialization::Pole);
    }

    #[test]
    fn degree_drop_specialization() {
        // F = t*x^2 + x + 1: leading coefficient t vanishes at t = 0.
        let t = RationalFunction::t();
        let f = ff_poly_from_coeffs(vec![RationalFunction::one(), RationalFunction::one(), t]);
        match specialize_poly(&f, &q(0)) {
            Specialization::DegreeDrop(p) => assert_eq!(p.degree(), Some(1)),
            other => panic!("expected DegreeDrop, got {:?}", other),
        }
    }

    #[test]
    fn matches_direct_computation_at_several_t() {
        // Cross-check specialization of F = x^2 - t^2 against hand evaluation.
        let f = x2_minus_t2();
        for a in [-3i64, -1, 2, 5, 7] {
            let p = specialize_poly(&f, &q(a)).polynomial().unwrap().clone();
            // x^2 - a^2
            assert_eq!(p.coefficients(), &[q(-a * a), q(0), q(1)]);
        }
    }
}
