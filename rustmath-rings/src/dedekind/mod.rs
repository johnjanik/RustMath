//! Finitely generated modules over Dedekind domains via pseudo-matrices.
//!
//! MAGMA Handbook ch. 55/56 flavor ("Modules over Dedekind domains"), following
//! the algorithmic treatment of Cohen, *Advanced Topics in Computational Number
//! Theory* (GTM 193), ch. 1:
//!
//! * a **pseudo-matrix** is a matrix over the fraction field `K` together with a
//!   fractional ideal per row; the row `(𝔞ᵢ, vᵢ)` contributes the pseudo-element
//!   `𝔞ᵢ·vᵢ`, and the module is `M = Σᵢ 𝔞ᵢ·vᵢ ⊆ Kⁿ`;
//! * [`pseudo::PseudoMatrix::hnf`] is the Hermite-like reduction over a Dedekind
//!   domain (Cohen Algorithm 1.4.7 flavor), producing a **pseudo-basis**
//!   `M = ⊕ᵢ 𝔞ᵢ·vᵢ` with an echelon matrix whose pivots are exactly `1`;
//! * [`pseudo::PseudoMatrix::steinitz_ideal`] returns the Steinitz-class
//!   representative `𝔞₁···𝔞ₖ` (`M ≅ O^{k-1} ⊕ 𝔞₁···𝔞ₖ`);
//! * [`pseudo::PseudoMatrix::elementary_divisors`] computes the elementary
//!   divisor ideals `𝔡₁ | 𝔡₂ | …` of `Oⁿ/M` via determinantal (Fitting) ideals,
//!   which localize correctly over a Dedekind domain;
//! * [`hom`] provides `Hom`/`End` of pseudo-basis (projective) modules and of
//!   torsion modules given by elementary divisors.
//!
//! Two contexts implement the abstract [`DedekindContext`] interface:
//! [`zctx::ZDedekind`] (the PID `ℤ`, ideals = positive rationals — used to
//! cross-check the generic algorithms against the Smith normal form from
//! `rustmath-matrix`) and [`nfctx::NfDedekind`] (the maximal order `O_K` of a
//! number field, backed by the verified ideal arithmetic of
//! `rustmath-numberfields`).
//!
//! # Honesty contract
//!
//! Everything that only needs **ideal arithmetic** (products, inverses, sums,
//! membership, idempotent splittings) is computed exactly. Anything that needs
//! **class-group decisions** is surfaced honestly:
//!
//! * [`Principality::Unresolved`] is *not* a decision — it reports that the
//!   bounded principality search found no generator;
//! * [`IsoDecision::Unresolved`] likewise: two modules of equal rank whose
//!   Steinitz quotient cannot be certified principal stay undecided;
//! * [`pseudo::steinitz_normal_form`] (the explicit `O^{k-1} ⊕ 𝔞`
//!   transformation) returns [`DedekindError::NeedsClassGroup`].

pub mod hom;
pub mod nfctx;
pub mod pseudo;
pub mod zctx;

pub use hom::{end_module, hom_module, torsion_hom_cyclic_factors, HomModule, PseudoHom};
pub use nfctx::{NfDedekind, NfElem};
pub use pseudo::{steinitz_normal_form, ElementaryDivisors, PseudoMatrix};
pub use zctx::{ZDedekind, ZIdeal};

use thiserror::Error;

/// Errors from the Dedekind-module layer.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum DedekindError {
    /// Dimension / shape mismatch in a pseudo-matrix or hom matrix.
    #[error("shape mismatch: {0}")]
    Shape(String),
    /// An idempotent splitting was requested for ideals that are not integral
    /// and coprime.
    #[error("ideals not integral/coprime: {0}")]
    NotCoprime(String),
    /// A quotient construction needs `M ⊆ Oⁿ` and the input is not integral.
    #[error("module not contained in the standard module: {0}")]
    NotIntegral(String),
    /// The requested operation needs class-group data (ideal reduction /
    /// discrete logarithms in `Cl(K)`) that this crate cannot compute.
    #[error("needs class-group data not computable in-crate: {0}")]
    NeedsClassGroup(String),
    /// An internal invariant failed — a bug, never a mathematical statement.
    #[error("internal invariant violated: {0}")]
    Internal(String),
}

/// Outcome of a *bounded* principality search.
///
/// `Principal(g)` is a certificate: `g` has been verified to generate the
/// ideal. `Unresolved` is **not** a decision — the bounded search found no
/// generator, which says nothing about the ideal class without class-group
/// data.
#[derive(Debug, Clone, PartialEq)]
pub enum Principality<E> {
    /// A verified generator: the principal ideal `(g)` equals the input ideal.
    Principal(E),
    /// The bounded search found nothing. UNRESOLVED — not a proof of
    /// non-principality.
    Unresolved,
}

/// Outcome of a module isomorphism test over a Dedekind domain.
///
/// Rank + Steinitz class classify finitely generated projective modules, so
/// `NotIsomorphic` is only ever produced from a *decided* invariant (unequal
/// ranks). Equal ranks with an unresolved Steinitz comparison stay
/// [`IsoDecision::Unresolved`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IsoDecision {
    /// Certified isomorphic (equal rank, Steinitz quotient certified principal).
    Isomorphic,
    /// Certified non-isomorphic (a decided invariant differs).
    NotIsomorphic,
    /// UNRESOLVED: deciding would need class-group data.
    Unresolved,
}

/// The arithmetic interface of a Dedekind domain `O` with fraction field `K`:
/// exact field arithmetic on `K`-elements and exact arithmetic on **nonzero
/// fractional ideals**.
///
/// Implementations must give both `Elem` and `Ideal` *canonical* representations
/// so that derived `PartialEq` is mathematical equality.
pub trait DedekindContext {
    /// An element of the fraction field `K` (canonical representation).
    type Elem: Clone + PartialEq + std::fmt::Debug;
    /// A nonzero fractional ideal of `O` (canonical representation).
    type Ideal: Clone + PartialEq + std::fmt::Debug;

    // ---- field arithmetic on K ----

    /// `0 ∈ K`.
    fn zero(&self) -> Self::Elem;
    /// `1 ∈ K`.
    fn one(&self) -> Self::Elem;
    /// `a + b`.
    fn add(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;
    /// `-a`.
    fn neg(&self, a: &Self::Elem) -> Self::Elem;
    /// `a · b`.
    fn mul(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;
    /// `a⁻¹`; `None` when `a` is not invertible (i.e. `a = 0` for a field).
    fn inv(&self, a: &Self::Elem) -> Option<Self::Elem>;

    /// `a - b`.
    fn sub(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        self.add(a, &self.neg(b))
    }
    /// `a / b`; `None` when `b` is not invertible.
    fn div(&self, a: &Self::Elem, b: &Self::Elem) -> Option<Self::Elem> {
        self.inv(b).map(|bi| self.mul(a, &bi))
    }
    /// Is `a = 0`?
    fn is_zero(&self, a: &Self::Elem) -> bool {
        *a == self.zero()
    }

    // ---- nonzero fractional ideals ----

    /// The unit ideal `O`.
    fn unit_ideal(&self) -> Self::Ideal;
    /// `𝔞·𝔟`.
    fn ideal_mul(&self, a: &Self::Ideal, b: &Self::Ideal) -> Self::Ideal;
    /// The ideal sum `𝔞 + 𝔟` (the "gcd" of the two ideals).
    fn ideal_add(&self, a: &Self::Ideal, b: &Self::Ideal) -> Self::Ideal;
    /// `𝔞⁻¹` (fractional-ideal inverse).
    fn ideal_inv(&self, a: &Self::Ideal) -> Self::Ideal;
    /// The principal fractional ideal `(x)`; `None` iff `x = 0`.
    fn principal_ideal(&self, x: &Self::Elem) -> Option<Self::Ideal>;
    /// Is `x ∈ 𝔞`? (`x = 0` is in every ideal.)
    fn ideal_contains(&self, a: &Self::Ideal, x: &Self::Elem) -> bool;
    /// Is `𝔞 ⊆ 𝔟`?
    fn ideal_subset(&self, a: &Self::Ideal, b: &Self::Ideal) -> bool;

    /// Is `𝔞` integral (`𝔞 ⊆ O`)?
    fn ideal_is_integral(&self, a: &Self::Ideal) -> bool {
        self.ideal_subset(a, &self.unit_ideal())
    }
    /// `𝔞 ∩ 𝔟 = 𝔞·𝔟·(𝔞+𝔟)⁻¹` (the "lcm" of the two ideals).
    fn ideal_intersect(&self, a: &Self::Ideal, b: &Self::Ideal) -> Self::Ideal {
        let prod = self.ideal_mul(a, b);
        self.ideal_mul(&prod, &self.ideal_inv(&self.ideal_add(a, b)))
    }
    /// The scaled ideal `x·𝔞`; `None` iff `x = 0`.
    fn scaled_ideal(&self, x: &Self::Elem, a: &Self::Ideal) -> Option<Self::Ideal> {
        self.principal_ideal(x).map(|p| self.ideal_mul(&p, a))
    }

    /// Idempotent splitting: for **integral, coprime** ideals `𝔠₁ + 𝔠₂ = O`,
    /// return `(u, v)` with `u ∈ 𝔠₁`, `v ∈ 𝔠₂`, `u + v = 1`. Errors with
    /// [`DedekindError::NotCoprime`] when the precondition fails.
    fn idempotents(
        &self,
        c1: &Self::Ideal,
        c2: &Self::Ideal,
    ) -> Result<(Self::Elem, Self::Elem), DedekindError>;

    /// Bounded principality search; see [`Principality`] for the honesty
    /// contract. `Principal(g)` must be verified by the implementation.
    fn principal_generator(&self, a: &Self::Ideal) -> Principality<Self::Elem>;
}

#[cfg(test)]
mod trait_tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn default_ideal_intersect_is_lcm_over_z() {
        let ctx = ZDedekind;
        let a = ZIdeal::from_int(4).unwrap();
        let b = ZIdeal::from_int(6).unwrap();
        // lcm(4, 6) = 12, gcd(4, 6) = 2
        assert_eq!(ctx.ideal_intersect(&a, &b), ZIdeal::from_int(12).unwrap());
        assert_eq!(ctx.ideal_add(&a, &b), ZIdeal::from_int(2).unwrap());
    }

    #[test]
    fn default_scaled_ideal() {
        let ctx = ZDedekind;
        let a = ZIdeal::from_int(6).unwrap();
        let x = Rational::new(1, 2).unwrap();
        assert_eq!(ctx.scaled_ideal(&x, &a).unwrap(), ZIdeal::from_int(3).unwrap());
        assert!(ctx.scaled_ideal(&Rational::from_i64(0), &a).is_none());
    }
}
