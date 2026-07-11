//! Typed rational function field layer: K(x) = Frac(K[x]) with real arithmetic.
//!
//! This module is the honest, fully-typed replacement for the string-typed
//! facades in `function_field::function_field_rational`,
//! `function_field_element`, `function_field::valuation` and the string-keyed
//! parts of `crate::divisor`. Everything here computes: elements are
//! normalized numerator/denominator pairs of `UnivariatePolynomial<K>`,
//! valuations are exact integers obtained by factor multiplicity, and the
//! degree/product laws (`deg div(f) = 0`, `v(fg) = v(f) + v(g)`) are enforced
//! by tests over both `Rational` and `GFp<5>`.
//!
//! # Contents
//!
//! - [`gfp::GFp`]: const-generic prime field GF(p) implementing the
//!   `rustmath_core` `Field` traits with working `zero()`/`one()` (the
//!   `rustmath_finitefields::PrimeField` element type carries a runtime
//!   modulus and its `Ring::zero()` panics, so it cannot serve as a generic
//!   coefficient field).
//! - [`element::RationalFunction`]: an element of K(x), kept in canonical
//!   form (gcd(num, den) = 1, den monic).
//! - [`element::RationalFunctionField`]: the parent field K(x).
//! - [`factor::FactorableConstantField`]: the capability trait for factoring
//!   K[x] into monic irreducibles (implemented for `Rational` and `GFp<P>`),
//!   with a built-in reconstruction check so a wrong factorization is an
//!   `Err`, never a silent lie.
//! - [`place::Place`]: places of K(x) (monic irreducible finite places and
//!   the infinite place), valuations, uniformizers and residue fields
//!   ([`place::ResidueClass`] is the quotient representation K[x]/(p)).
//! - [`divisor::Divisor`]: the free abelian group on places, principal
//!   divisors, effective parts, and exact genus-0 Riemann-Roch spaces L(D).
//!
//! # MAGMA / Sage correspondence
//!
//! MAGMA handbook ch. 41-43 base layer; `sage.rings.function_field` for the
//! rational function field case (genus 0).

pub mod divisor;
pub mod element;
pub mod factor;
pub mod gfp;
pub mod place;

pub use divisor::Divisor;
pub use element::{RationalFunction, RationalFunctionField};
pub use factor::FactorableConstantField;
pub use gfp::GFp;
pub use place::{Place, ResidueClass};
