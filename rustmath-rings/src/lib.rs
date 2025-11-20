//! # RustMath Rings Module
//!
//! This module provides a comprehensive collection of ring structures and their operations,
//! mirroring SageMath's `sage.rings` module. It serves as a central hub for accessing all
//! ring-related functionality in RustMath.
//!
//! ## Overview
//!
//! A ring is a fundamental algebraic structure consisting of a set equipped with two binary
//! operations (addition and multiplication) satisfying specific axioms. This module provides:
//!
//! - **Abstract Base Classes (ABC)**: Traits and type definitions for various ring categories
//! - **Concrete Ring Implementations**: Integer rings, rational fields, finite fields, etc.
//! - **Specialized Ring Structures**: Algebraic closures, asymptotic rings, etc.
//! - **Ring Homomorphisms and Morphisms**: Structure-preserving maps between rings
//!
//! ## Module Structure
//!
//! - `abc`: Abstract base classes defining ring categories (algebraic fields, p-adic rings, etc.)
//! - `algebraic_closure`: Algebraic closures of finite fields
//! - `asymptotic`: Asymptotic expansion rings for analytic combinatorics
//! - Re-exports from existing crates: integers, rationals, finite fields, p-adics, etc.
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::abc::{IntegerRing, RationalField, FiniteFieldTrait};
//! use rustmath_rings::algebraic_closure::AlgebraicClosureFiniteField;
//! ```

pub mod abc;
pub mod algebraic_closure;
pub mod asymptotic;
pub mod category_methods;
pub mod asymptotic_misc;
pub mod big_oh;
pub mod cfinite_sequence;
pub mod constructor;
pub mod derivation;
pub mod derivations_function_field;
pub mod differential;
pub mod divisor;
pub mod fraction_field;
pub mod fraction_field_element;
pub mod fraction_field_fpt;
pub mod function_field_element;
pub mod function_field_element_polymod;
pub mod function_field_element_rational;
pub mod function_field_module;
pub mod generic;
pub mod growth_group;
pub mod growth_group_cartesian;
pub mod homset;
pub mod infinity;
pub mod localization;
pub mod monomials;
pub mod morphism;
pub mod noncommutative_ideals;
pub mod numbers_abc;
pub mod padics;
pub mod pari_ring;
pub mod quotient_ring;
pub mod quotient_ring_element;
pub mod real_interval_absolute;
pub mod real_mpfi;
pub mod residue_field;
pub mod ring;
pub mod ring_extension;
pub mod ring_extension_element;
pub mod ring_extension_morphism;
pub mod term_monoid;
pub mod universal_cyclotomic_field;
pub mod augmented_valuation;
pub mod function_field;
pub mod invariants;
pub mod laurent_series_ring;
pub mod laurent_series_ring_element;
pub mod lazy_series;
pub mod lazy_series_ring;
pub mod multi_power_series_ring;
pub mod multi_power_series_ring_element;
pub mod power_series_pari;
pub mod power_series_poly;
pub mod puiseux_series_ring;
pub mod puiseux_series_ring_element;
pub mod real_arb;
pub mod real_lazy;
pub mod semirings;
pub mod sum_of_squares;
pub mod tate_algebra;
pub mod valuation;
pub mod qqbar;

// Re-export core ring types from other crates
pub use rustmath_integers::Integer;
pub use rustmath_rationals::Rational;
pub use rustmath_reals::Real;
pub use rustmath_complex::Complex;
// pub use rustmath_finitefields::{FiniteField, FiniteFieldElement, GaloisField};
// pub use rustmath_padics::{PAdicNumber, PAdicRing};

// Re-export core traits
pub use rustmath_core::{Ring, CommutativeRing, Field, EuclideanDomain, IntegralDomain};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_imports() {
        // Test that all re-exports are accessible
        use crate::abc::*;
        use crate::algebraic_closure::*;
        use crate::asymptotic::*;
    }
}
