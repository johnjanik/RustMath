//! Drinfeld modules
//!
//! This module provides comprehensive support for Drinfeld modules, corresponding to
//! SageMath's `sage.rings.function_field.drinfeld_modules` package.
//!
//! # Mathematical Overview
//!
//! Drinfeld modules are function field analogues of elliptic curves and abelian varieties,
//! introduced by Vladimir Drinfeld in 1974. They play a fundamental role in:
//!
//! - The Langlands program for function fields
//! - Class field theory of function fields
//! - Arithmetic geometry over finite fields
//! - Cryptographic applications (Drinfeld module cryptography)
//!
//! ## Definition
//!
//! A Drinfeld module of rank r over a field K with Fq ⊆ K is a ring homomorphism:
//!
//! φ: A → K{τ}
//!
//! where:
//! - A is a Dedekind domain (typically Fq[T])
//! - K{τ} is the twisted polynomial ring with τα = α^q·τ for α ∈ Fq
//! - φ(a) has degree r·deg(a) for non-zero a ∈ A
//!
//! ## Example: The Carlitz Module
//!
//! The simplest Drinfeld module is the Carlitz module C: F_q[T] → F_q[T]{τ}:
//!
//! C(T) = T + τ
//!
//! This is the function field analogue of the Gm multiplicative group scheme.
//!
//! # Module Structure
//!
//! - `action`: Actions of Drinfeld modules on mathematical structures
//! - `charzero_drinfeld_module`: Drinfeld modules in characteristic zero
//! - `drinfeld_module`: Base Drinfeld module implementation
//! - `finite_drinfeld_module`: Drinfeld modules over finite fields
//! - `homset`: Homomorphism sets between Drinfeld modules
//! - `morphism`: Morphisms (including isogenies) between Drinfeld modules
//!
//! # Key Concepts
//!
//! ## Rank
//!
//! The rank r of a Drinfeld module φ is defined as:
//! r = deg_τ(φ(T))
//!
//! where T is a generator of the base ring Fq[T].
//!
//! ## Isogenies
//!
//! An isogeny u: φ → ψ is a non-zero morphism with finite kernel:
//! - Degree: deg(u) as a polynomial in τ
//! - Separable: when gcd(deg(u), q) = 1
//! - Kernel size: |ker(u)| = q^deg(u)
//!
//! ## Applications
//!
//! - **Explicit class field theory**: Drinfeld modules generate abelian extensions
//! - **Arithmetic dynamics**: Study of dynamics under φ(a) for a ∈ A
//! - **Cryptography**: Drinfeld module discrete logarithm problem
//! - **Transcendence theory**: Analogues of Schanuel's conjecture
//!
//! # Examples
//!
//! ```ignore
//! use rustmath_rings::function_field::drinfeld_modules::*;
//!
//! // Create the Carlitz module
//! let carlitz = drinfeld_module::DrinfeldModuleFactory::carlitz_module(2);
//! assert_eq!(carlitz.rank(), 1);
//!
//! // Create a rank 2 module over F4
//! let phi = finite_drinfeld_module::DrinfeldModule_finite::new("F4".to_string(), 2);
//! assert_eq!(phi.point_count(1), 16); // q^(r*deg) = 4^(2*1) = 16
//!
//! // Create an endomorphism ring
//! let end_phi = homset::DrinfeldModuleHomset::new("φ".to_string(), "φ".to_string());
//! assert!(end_phi.is_endomorphism_set());
//!
//! // Create an isogeny
//! let isogeny = morphism::DrinfeldModuleMorphism::new("φ".to_string(), "ψ".to_string(), 3);
//! assert_eq!(isogeny.kernel_size(2), 8); // 2^3 = 8
//! ```
//!
//! # References
//!
//! - Drinfeld, V.G. (1974). "Elliptic modules"
//! - Goss, D. (1996). "Basic Structures of Function Field Arithmetic"
//! - Thakur, D.S. (2004). "Function Field Arithmetic"

pub mod action;
pub mod charzero_drinfeld_module;
pub mod drinfeld_module;
pub mod finite_drinfeld_module;
pub mod homset;
pub mod morphism;

// Re-export main types for convenience
pub use action::{DrinfeldModuleAction, HasDrinfeldAction};
pub use charzero_drinfeld_module::{DrinfeldModule, DrinfeldModule_charzero, DrinfeldModule_rational};
pub use drinfeld_module::{DrinfeldModule as BaseDrinfeldModule, DrinfeldModuleFactory};
pub use finite_drinfeld_module::DrinfeldModule_finite;
pub use homset::{DrinfeldModuleHomset, DrinfeldModuleMorphismAction};
pub use morphism::DrinfeldModuleMorphism;

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;
    use rustmath_integers::Integer;

    #[test]
    fn test_module_imports() {
        // Test that all modules are accessible
        let _action: DrinfeldModuleAction<Rational, Integer> =
            DrinfeldModuleAction::new("test".to_string());

        let _charzero: DrinfeldModule_charzero<Rational> =
            DrinfeldModule_charzero::new("Q(T)".to_string(), 1);

        let _finite: DrinfeldModule_finite<Rational, Integer> =
            DrinfeldModule_finite::new("F4".to_string(), 1);

        let _homset: DrinfeldModuleHomset<Rational, Integer> =
            DrinfeldModuleHomset::new("φ".to_string(), "ψ".to_string());

        let _morphism: DrinfeldModuleMorphism<Rational, Integer> =
            DrinfeldModuleMorphism::new("φ".to_string(), "ψ".to_string(), 1);
    }

    #[test]
    fn test_carlitz_module_factory() {
        let carlitz: BaseDrinfeldModule<Rational, Integer> =
            DrinfeldModuleFactory::carlitz_module(2);

        assert_eq!(carlitz.rank(), 1);
        assert_eq!(carlitz.characteristic(), 2);
    }

    #[test]
    fn test_rational_drinfeld_module() {
        let module: DrinfeldModule_rational<Rational> =
            DrinfeldModule_rational::new("Q".to_string());

        assert!(module.is_carlitz());
        assert_eq!(module.generator(), "T");
    }

    #[test]
    fn test_finite_field_isogeny() {
        let morphism: DrinfeldModuleMorphism<Rational, Integer> =
            DrinfeldModuleMorphism::new("φ".to_string(), "ψ".to_string(), 2);

        assert!(morphism.is_isogeny());
        assert_eq!(morphism.kernel_size(3), 9); // 3^2
    }

    #[test]
    fn test_endomorphism_ring() {
        let end_phi: DrinfeldModuleHomset<Rational, Integer> =
            DrinfeldModuleHomset::new("φ".to_string(), "φ".to_string());

        assert!(end_phi.is_endomorphism_set());
        assert!(end_phi.is_nonempty());
    }

    #[test]
    fn test_drinfeld_action() {
        let action: DrinfeldModuleAction<Rational, Integer> =
            DrinfeldModuleAction::new("Carlitz".to_string());

        assert!(action.is_defined());
        assert!(action.domain().contains("Base ring"));
    }

    #[test]
    fn test_workflow() {
        // Simulate a typical workflow with Drinfeld modules

        // 1. Create a finite Drinfeld module
        let phi: DrinfeldModule_finite<Rational, Integer> =
            DrinfeldModule_finite::new("F4".to_string(), 2);

        assert_eq!(phi.rank(), 2);
        assert_eq!(phi.field_size(), 4);

        // 2. Create an endomorphism ring
        let end_phi: DrinfeldModuleHomset<Rational, Integer> =
            DrinfeldModuleHomset::new("φ".to_string(), "φ".to_string());

        assert!(end_phi.is_endomorphism_set());

        // 3. Create an endomorphism
        let frobenius: DrinfeldModuleMorphism<Rational, Integer> =
            DrinfeldModuleMorphism::new("φ".to_string(), "φ".to_string(), 1);

        assert!(frobenius.is_endomorphism());
        assert!(frobenius.is_isogeny());

        // 4. Check point count
        assert_eq!(phi.point_count(1), 16); // 4^(2*1)
    }
}
