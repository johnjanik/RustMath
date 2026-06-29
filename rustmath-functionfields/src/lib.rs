//! # rustmath-functionfields
//!
//! Function fields over `ℚ`: the rational function field `ℚ(t)`, finite
//! extensions `K = ℚ(t)[x] / (F)`, factorization of polynomials over `ℚ(t)`, and
//! specialization `t ↦ a ∈ ℚ`.
//!
//! This crate is the foundation for **regular-cover scanning** in the IGP24
//! inverse-Galois pipeline: build a regular extension of `ℚ(t)` realizing a
//! target group, then specialize `t` to rationals to obtain number fields with
//! (generically) the same Galois group. See `docs/inverse_galois_mvp.md`,
//! Appendix A, layer **L2**.
//!
//! ## Design
//!
//! Everything is built on top of `rustmath-polynomials` and `rustmath-rationals`;
//! no polynomial arithmetic is reimplemented.
//!
//! - [`RationalFunction`] is an element of `ℚ(t) = Frac(ℚ[t])`, stored as a
//!   reduced fraction of [`rustmath_polynomials::UnivariatePolynomial<Rational>`].
//!   It implements [`rustmath_core::Field`], so
//!   `UnivariatePolynomial<RationalFunction>` *is* `ℚ(t)[x]` for free, reusing all
//!   of the existing univariate machinery (`div_rem`, `gcd`, derivative, …).
//! - [`FunctionField`] wraps a monic irreducible `F ∈ ℚ(t)[x]` and exposes the
//!   degree, defining polynomial, and specialization.
//! - [`factor_over_qt`] factors `F ∈ ℚ(t)[x]` over `ℚ(t)` by the classical
//!   Trager-style *norm + factor-over-ℚ + recombination* scheme (full method
//!   documented in [`factor`]).
//! - [`specialize_poly`] / [`FunctionField::specialize`] substitute `t = a` and
//!   classify the result (good / pole / degree-drop / non-separable).
//!
//! ## Implemented in this first cut
//!
//! - `ℚ(t)` as a field (reduced fractions, eval at `t = a`, full ring/field ops).
//! - `FunctionField` (degree, defining polynomial, irreducibility-checked ctor).
//! - Squarefree decomposition and factorization over `ℚ(t)` (Trager-style:
//!   specialize to a good place, factor over `ℚ` with the reused Zassenhaus
//!   factorizer, recombine by Lagrange-interpolated trial division — which makes
//!   acceptance *certain*).
//! - Irreducibility test over `ℚ(t)`.
//! - Specialization with separability / degree / pole classification.
//!
//! ## Deferred (documented, not implemented)
//!
//! - **Newton–Puiseux expansion** of roots at a place, and place/valuation
//!   objects (ramification at branch points). The separability classifier in
//!   [`specialize_poly`] already flags branch points; full Puiseux series and a
//!   `Place`/`Valuation` API are the natural next layer.
//! - **Global function fields** over `𝔽_q` (only the characteristic-0 base `ℚ(t)`
//!   is supported here).
//! - **Sub-exponential recombination.** The recombination is worst-case
//!   exponential in the number of local factors (no Hensel-lift/LLL coefficient
//!   bound); fine for the small degrees of regular-cover scanning.
//! - **Trager norm over algebraic base fields** (factoring over `ℚ(t)(α)`); only
//!   the purely transcendental base `ℚ(t)` is handled.

pub mod factor;
pub mod function_field;
pub mod genus;
pub mod places;
pub mod ratfunc;

pub use genus::{branch_radical, disc_x, genus, GenusError};
pub use places::Place;
pub use factor::{factor_over_qt, is_irreducible_over_qt};
pub use function_field::{
    ff_poly_from_coeffs, specialize_poly, FfPoly, FunctionField, QxPoly, Specialization,
};
pub use ratfunc::{QtPoly, RationalFunction};

#[cfg(test)]
mod integration_tests {
    use super::*;
    use rustmath_core::Ring;
    use rustmath_rationals::Rational;

    fn q(n: i64) -> Rational {
        Rational::from_i64(n)
    }

    #[test]
    fn end_to_end_build_factor_specialize() {
        // Build F = x^2 - t^2, recognize it is reducible, factor it, then build a
        // FunctionField from an irreducible piece and specialize it.
        let t = RationalFunction::t();
        let minus_t2 = RationalFunction::zero() - (t.clone() * t.clone());
        let f = ff_poly_from_coeffs(vec![
            minus_t2,
            RationalFunction::zero(),
            RationalFunction::one(),
        ]);

        // Reducible:
        assert!(!is_irreducible_over_qt(&f).unwrap());
        // Building a FunctionField from a reducible F must error.
        assert!(FunctionField::new(f.clone()).is_err());

        // Take an irreducible factor x - t and build K = ℚ(t)[x]/(x - t) ≅ ℚ(t).
        let (_, factors) = factor_over_qt(&f);
        let (g, _) = &factors[0];
        let k = FunctionField::new(g.clone()).unwrap();
        assert_eq!(k.degree(), 1);

        // An honest degree-2 field: x^2 - (t^3 + 1), irreducible over ℚ(t).
        let c = t.clone() * t.clone() * t.clone() + RationalFunction::one();
        let f2 = ff_poly_from_coeffs(vec![
            RationalFunction::zero() - c,
            RationalFunction::zero(),
            RationalFunction::one(),
        ]);
        let k2 = FunctionField::new(f2).unwrap();
        assert_eq!(k2.degree(), 2);

        // Specialize at t = 2: x^2 - 9, separable, degree preserved.
        let spec = k2.specialize(&q(2));
        assert!(spec.is_good());
        let p = spec.polynomial().unwrap();
        // 2^3 + 1 = 9
        assert_eq!(p.coefficients(), &[q(-9), q(0), q(1)]);
    }
}
