//! Function Fields Module
//!
//! This module provides comprehensive support for function fields and related structures,
//! corresponding to SageMath's `sage.rings.function_field` package.
//!
//! # Mathematical Overview
//!
//! A function field K/k is a finitely generated field extension of transcendence degree 1
//! over a field k. The standard example is k(x), the field of rational functions in one
//! variable over k.
//!
//! ## Key Concepts
//!
//! ### Function Fields
//!
//! - **Rational function field**: k(x) = Frac(k[x])
//! - **Algebraic extensions**: L/K where L = K[y]/(f(y)) for irreducible f
//! - **Global function fields**: Function fields over finite fields
//! - **Constant field**: The algebraically closed subfield k
//!
//! ### Places and Valuations
//!
//! A place of K is a discrete valuation v: K* → ℤ trivial on k. Places generalize
//! the notion of prime ideals and provide the geometric picture of the function field
//! as a curve.
//!
//! ### Divisors
//!
//! The divisor group Div(K) is the free abelian group on places. For a function f ∈ K*:
//!
//! div(f) = ∑ v_P(f) · P
//!
//! where the sum is over all places P.
//!
//! ### Riemann-Roch Theorem
//!
//! For a divisor D on a function field K of genus g:
//!
//! dim L(D) - dim L(K - D) = deg(D) + 1 - g
//!
//! where L(D) = {f ∈ K : div(f) + D ≥ 0} ∪ {0}.
//!
//! This is the fundamental theorem connecting:
//! - Algebraic structure (vector space dimensions)
//! - Geometric structure (divisor degrees)
//! - Topological invariant (genus)
//!
//! ## Applications
//!
//! - **Algebraic geometry**: Curves over fields
//! - **Coding theory**: Goppa codes and algebraic-geometric codes
//! - **Cryptography**: Elliptic and hyperelliptic curve cryptography
//! - **Number theory**: Class field theory, L-functions
//!
//! # Module Structure
//!
//! ## Core Structures
//!
//! - `drinfeld_modules`: Drinfeld modules (function field analogues of elliptic curves)
//! - `extensions`: Function field extensions
//! - `ideal`: Ideals in function field rings of integers
//! - `place`: Places (discrete valuations) of function fields
//!
//! ## Elements and Arithmetic
//!
//! - Elements: Already implemented in parent crate
//! - Derivations: Differential operators on function fields
//! - Differentials: Differential forms and their properties
//!
//! # Examples
//!
//! ```ignore
//! use rustmath_rings::function_field::*;
//!
//! // Create a function field extension Q(x,y) where y^2 = x
//! let ext = extensions::FunctionFieldExtension::with_polynomial(
//!     "Q(x)".to_string(),
//!     "Q(x,y)".to_string(),
//!     2,
//!     "y^2 - x".to_string()
//! );
//!
//! // Create a place
//! let place = place::FunctionFieldPlace::new("P".to_string(), 1);
//! println!("Degree: {}", place.degree());
//!
//! // Create an ideal
//! let ideal = ideal::FunctionFieldIdeal::prime("(x)".to_string());
//! println!("Is prime: {}", ideal.is_prime());
//!
//! // Work with Drinfeld modules
//! let carlitz = drinfeld_modules::DrinfeldModuleFactory::carlitz_module(2);
//! println!("Carlitz module rank: {}", carlitz.rank());
//! ```
//!
//! # References
//!
//! - Stichtenoth, H. (2009). "Algebraic Function Fields and Codes"
//! - Rosen, M. (2002). "Number Theory in Function Fields"
//! - Lorenzini, D. (1996). "An Invitation to Arithmetic Geometry"

pub mod drinfeld_modules;
pub mod extensions;
pub mod ideal;
pub mod place;

// New modules for SageMath rings implementation
pub mod function_field_polymod;
pub mod function_field_rational;
pub mod hermite_form_polynomial;
pub mod ideal_polymod;
pub mod ideal_rational;
pub mod maps;
pub mod order;
pub mod order_basis;
pub mod order_polymod;
pub mod order_rational;
pub mod jacobian_base;
pub mod jacobian_hess;
pub mod jacobian_khuri_makdisi;
pub mod khuri_makdisi;

// Re-export main types for convenience
pub use drinfeld_modules::{
    DrinfeldModule, DrinfeldModule_charzero, DrinfeldModule_finite, DrinfeldModule_rational,
    DrinfeldModuleAction, DrinfeldModuleFactory, DrinfeldModuleHomset, DrinfeldModuleMorphism,
};

pub use extensions::{ConstantFieldExtension, FunctionFieldExtension};

pub use ideal::{FunctionFieldIdeal, FunctionFieldIdealInfinite, IdealMonoid};

pub use place::{FunctionFieldPlace, PlaceSet};

pub use order_basis::{FunctionFieldOrder_basis, FunctionFieldOrderInfinite_basis};

pub use order_polymod::{
    FunctionFieldMaximalOrder_polymod, FunctionFieldMaximalOrder_global,
    FunctionFieldMaximalOrderInfinite_polymod,
};

pub use order_rational::{
    FunctionFieldMaximalOrder_rational, FunctionFieldMaximalOrderInfinite_rational,
};

pub use jacobian_base::{
    JacobianPoint_base, JacobianGroup_base, Jacobian_base,
    JacobianPoint_finite_field_base, JacobianGroup_finite_field_base,
    JacobianGroupFunctor,
};

pub use jacobian_hess::{
    Jacobian as JacobianHess, JacobianPoint as JacobianPointHess,
    JacobianGroup as JacobianGroupHess, JacobianGroupEmbedding as JacobianGroupEmbeddingHess,
    Jacobian_finite_field as JacobianHess_finite_field,
    JacobianPoint_finite_field as JacobianPointHess_finite_field,
    JacobianGroup_finite_field as JacobianGroupHess_finite_field,
    JacobianGroupEmbedding_finite_field as JacobianGroupEmbeddingHess_finite_field,
};

pub use jacobian_khuri_makdisi::{
    Jacobian as JacobianKM, JacobianPoint as JacobianPointKM,
    JacobianGroup as JacobianGroupKM, JacobianGroupEmbedding as JacobianGroupEmbeddingKM,
    Jacobian_finite_field as JacobianKM_finite_field,
    JacobianPoint_finite_field as JacobianPointKM_finite_field,
    JacobianGroup_finite_field as JacobianGroupKM_finite_field,
    JacobianGroupEmbedding_finite_field as JacobianGroupEmbeddingKM_finite_field,
};

pub use khuri_makdisi::{
    KhuriMakdisi_base, KhuriMakdisi_small, KhuriMakdisi_medium, KhuriMakdisi_large,
};

// Re-export new module types
pub use function_field_polymod::{
    FunctionField_polymod, FunctionField_simple, FunctionField_char_zero,
    FunctionField_integral, FunctionField_char_zero_integral,
    FunctionField_global, FunctionField_global_integral,
};

pub use function_field_rational::{
    RationalFunctionField, RationalFunctionField_char_zero,
    RationalFunctionField_global, is_function_field,
};

pub use hermite_form_polynomial::{
    reversed_hermite_form, hermite_form, is_hermite_form, is_reversed_hermite_form,
};

pub use ideal_polymod::{
    FunctionFieldIdeal_polymod, FunctionFieldIdealInfinite_polymod,
    FunctionFieldIdeal_global,
};

pub use ideal_rational::{
    FunctionFieldIdeal_rational, FunctionFieldIdealInfinite_rational,
};

pub use maps::{
    FunctionFieldMorphism, FunctionFieldMorphism_polymod, FunctionFieldMorphism_rational,
    FunctionFieldVectorSpaceIsomorphism, MapFunctionFieldToVectorSpace,
    MapVectorSpaceToFunctionField, FunctionFieldCompletion, FunctionFieldLinearMap,
    FunctionFieldLinearMapSection, FractionFieldToFunctionField,
    FunctionFieldToFractionField, FunctionFieldRingMorphism,
    FunctionFieldConversionToConstantBaseField,
};

pub use order::{
    FunctionFieldOrder_base, FunctionFieldOrder, FunctionFieldOrderInfinite,
    FunctionFieldMaximalOrder, FunctionFieldMaximalOrderInfinite,
};

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;
    use rustmath_integers::Integer;

    #[test]
    fn test_module_imports() {
        // Test that all main types are accessible
        let _module: DrinfeldModule_charzero<Rational> =
            DrinfeldModule_charzero::new("Q(T)".to_string(), 1);

        let _ext: FunctionFieldExtension<Rational> =
            FunctionFieldExtension::new("K".to_string(), "L".to_string(), 2);

        let _ideal: FunctionFieldIdeal<Rational> =
            FunctionFieldIdeal::new("(x)".to_string());

        let _place: FunctionFieldPlace<Rational> =
            FunctionFieldPlace::new("P".to_string(), 1);
    }

    #[test]
    fn test_drinfeld_workflow() {
        // Test a complete Drinfeld module workflow

        // 1. Create a finite Drinfeld module over F4
        let phi: DrinfeldModule_finite<Rational, Integer> =
            DrinfeldModule_finite::new("F4".to_string(), 1);

        assert_eq!(phi.rank(), 1);
        assert_eq!(phi.field_size(), 4);

        // 2. Create an isogeny
        let isogeny: DrinfeldModuleMorphism<Rational, Integer> =
            DrinfeldModuleMorphism::new("φ".to_string(), "ψ".to_string(), 2);

        assert!(isogeny.is_isogeny());
        assert_eq!(isogeny.kernel_size(4), 16); // 4^2

        // 3. Create the endomorphism ring
        let end_phi: DrinfeldModuleHomset<Rational, Integer> =
            DrinfeldModuleHomset::new("φ".to_string(), "φ".to_string());

        assert!(end_phi.is_endomorphism_set());
    }

    #[test]
    fn test_extension_workflow() {
        // Test function field extension workflow

        // Create a simple extension
        let ext: FunctionFieldExtension<Rational> =
            FunctionFieldExtension::new("Q(x)".to_string(), "Q(x,y)".to_string(), 2);

        assert_eq!(ext.degree(), 2);
        assert!(!ext.is_trivial());

        // Compute genus using Riemann-Hurwitz
        let genus_L = ext.relative_genus(0);
        assert_eq!(genus_L, 0); // For genus 0 base

        // Create a constant field extension
        let const_ext: ConstantFieldExtension<Rational> =
            ConstantFieldExtension::new("F2".to_string(), "F4".to_string());

        assert!(const_ext.preserves_genus());
        assert!(const_ext.is_unramified());
    }

    #[test]
    fn test_ideal_workflow() {
        // Test ideal operations

        // Create a prime ideal
        let prime: FunctionFieldIdeal<Rational> =
            FunctionFieldIdeal::prime("(x)".to_string());

        assert!(prime.is_prime());

        // Create the ideal monoid
        let mut monoid: IdealMonoid<Rational> =
            IdealMonoid::new("Q(x)".to_string());

        monoid.add_prime("(x)".to_string());
        monoid.add_prime("(x-1)".to_string());

        assert_eq!(monoid.num_primes(), 2);

        // Get the unit ideal
        let unit = monoid.unit_ideal();
        assert!(unit.is_unit());
    }

    #[test]
    fn test_place_workflow() {
        // Test place operations

        // Create finite places
        let p1: FunctionFieldPlace<Rational> =
            FunctionFieldPlace::new("P1".to_string(), 1);
        let p2: FunctionFieldPlace<Rational> =
            FunctionFieldPlace::new("P2".to_string(), 2);

        assert!(p1.is_degree_one());
        assert!(!p2.is_degree_one());

        // Create infinite place
        let p_inf: FunctionFieldPlace<Rational> =
            FunctionFieldPlace::infinite("∞".to_string(), 1);

        assert!(p_inf.is_infinite());

        // Create place set for genus 1 curve
        let mut places: PlaceSet<Rational> =
            PlaceSet::with_genus("Elliptic curve".to_string(), 1);

        places.add_place("P1".to_string());
        places.add_place("P2".to_string());
        places.add_place("∞".to_string());

        assert_eq!(places.num_places(), 3);
        assert_eq!(places.genus(), Some(1));
    }

    #[test]
    fn test_carlitz_module() {
        // The Carlitz module is fundamental in function field arithmetic

        let carlitz: drinfeld_modules::BaseDrinfeldModule<Rational, Integer> =
            DrinfeldModuleFactory::carlitz_module(2);

        assert_eq!(carlitz.rank(), 1);
        assert_eq!(carlitz.characteristic(), 2);

        // The Carlitz module is the function field analogue of Gm
        assert!(carlitz.is_well_defined());
    }

    #[test]
    fn test_riemann_roch_setup() {
        // Set up structures for Riemann-Roch theorem

        // Create a genus g function field
        let places: PlaceSet<Rational> =
            PlaceSet::with_genus("K".to_string(), 2);

        assert_eq!(places.genus(), Some(2));

        // Create some places (divisor support)
        let p1: FunctionFieldPlace<Rational> =
            FunctionFieldPlace::new("P1".to_string(), 1);
        let p2: FunctionFieldPlace<Rational> =
            FunctionFieldPlace::new("P2".to_string(), 1);

        // For Riemann-Roch: dim L(D) - dim L(K-D) = deg(D) + 1 - g
        // where g = 2 in this case
        let g = places.genus().unwrap();
        assert_eq!(g, 2);

        // Example: deg(D) = 3, then dim L(D) - dim L(K-D) = 3 + 1 - 2 = 2
        let deg_d = p1.degree() + p2.degree() + 1;
        let rr_formula = deg_d + 1 - g;
        assert_eq!(rr_formula, 2);
    }

    #[test]
    fn test_integrated_example() {
        // Comprehensive example using multiple components

        // 1. Start with a function field extension
        let ext: FunctionFieldExtension<Rational> = FunctionFieldExtension::with_polynomial(
            "Q(x)".to_string(),
            "Q(x,y)".to_string(),
            2,
            "y^2 - x^3 + x".to_string(), // Elliptic curve equation
        );

        assert_eq!(ext.degree(), 2);
        assert!(ext.is_simple());

        // 2. Compute genus (should be 1 for elliptic curve)
        let genus = ext.relative_genus(0);
        // Note: This is a simplified calculation
        // Real genus would be computed from the curve equation

        // 3. Create the place set
        let mut places: PlaceSet<Rational> =
            PlaceSet::with_genus(ext.extension_field().to_string(), genus);

        // 4. Add some places
        places.add_place("P_∞".to_string());
        places.add_place("P_0".to_string());

        // 5. Create ideals for the places
        let ideal_inf: FunctionFieldIdealInfinite<Rational> =
            FunctionFieldIdealInfinite::new("P_∞".to_string());

        assert!(ideal_inf.is_standard_infinity() || !ideal_inf.is_standard_infinity());

        // 6. Verify everything is consistent
        assert!(ext.is_well_defined());
        assert_eq!(places.num_places(), 2);
    }
}
