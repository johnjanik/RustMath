//! # Ring Category Methods
//!
//! This module provides category-theoretic methods for rings, mirroring SageMath's
//! `sage.categories.rings` module structure. It organizes methods into four main categories:
//!
//! - **ElementMethods**: Methods for ring elements (units, divisibility, inverses)
//! - **ParentMethods**: Methods for ring structures (properties, constructions, ideals)
//! - **MorphismMethods**: Methods for ring homomorphisms (injectivity, extensions)
//! - **SubcategoryMethods**: Methods for subcategories (domains, fields, commutative rings)
//!
//! ## Overview
//!
//! SageMath's category system organizes methods based on their mathematical context.
//! This module provides the Rust equivalent using traits that can be implemented by
//! various ring types.
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::category_methods::{RingElementMethods, RingParentMethods};
//! ```

use rustmath_core::{Ring, CommutativeRing, Field, IntegralDomain, EuclideanDomain, MathError, Result};
use std::fmt::Debug;

// ============================================================================
// ElementMethods - Methods for ring elements
// ============================================================================

/// Methods for ring elements, corresponding to SageMath's `Rings.ElementMethods`
///
/// This trait provides operations on ring elements including:
/// - Testing if an element is a unit
/// - Computing multiplicative inverses
/// - Exact division operations
///
/// Reference: https://doc.sagemath.org/html/en/reference/categories/sage/categories/rings.html#sage.categories.rings.Rings.ElementMethods
pub trait RingElementMethods: Ring {
    /// Check if this element is a unit (has a multiplicative inverse)
    ///
    /// An element `a` in a ring `R` is a unit if there exists `b` in `R`
    /// such that `a * b = b * a = 1`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_rings::category_methods::RingElementMethods;
    ///
    /// // In integers, only ±1 are units
    /// assert!(1i32.is_unit_element());
    /// assert!((-1i32).is_unit_element());
    /// assert!(!2i32.is_unit_element());
    /// ```
    ///
    /// # Default Implementation
    ///
    /// The default implementation only recognizes 1, -1, and 0 (which is not a unit).
    /// Most concrete ring types should override this with specialized logic.
    fn is_unit_element(&self) -> bool {
        self.is_one() || *self == -Self::one()
    }

    /// Return the multiplicative inverse of this element if it is a unit
    ///
    /// Returns an error if the element is not a unit.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_rings::category_methods::RingElementMethods;
    ///
    /// // In integers, only ±1 have inverses
    /// assert_eq!(1i32.inverse_of_unit().unwrap(), 1);
    /// assert_eq!((-1i32).inverse_of_unit().unwrap(), -1);
    /// assert!(2i32.inverse_of_unit().is_err());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `MathError::NotInvertible` if the element is not a unit.
    fn inverse_of_unit(&self) -> Result<Self> {
        if self.is_one() {
            Ok(Self::one())
        } else if *self == -Self::one() {
            Ok(-Self::one())
        } else {
            Err(MathError::NotInvertible)
        }
    }

    /// Divide `self` by `other` if the division is exact (no remainder)
    ///
    /// This method attempts to compute `self / other` and returns an error
    /// if the division is not exact (i.e., if there would be a remainder).
    ///
    /// # Arguments
    ///
    /// * `other` - The divisor
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_rings::category_methods::RingElementMethods;
    ///
    /// assert_eq!(6i32.divide_if_possible(&2).unwrap(), 3);
    /// assert_eq!(6i32.divide_if_possible(&3).unwrap(), 2);
    /// assert!(7i32.divide_if_possible(&2).is_err());  // Not exact
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `MathError::NotDivisible` if the division is not exact or if
    /// attempting to divide by zero.
    fn divide_if_possible(&self, other: &Self) -> Result<Self>
    where
        Self: EuclideanDomain,
    {
        if other.is_zero() {
            return Err(MathError::DivisionByZero);
        }

        let (quotient, remainder) = self.div_rem(other)?;

        if remainder.is_zero() {
            Ok(quotient)
        } else {
            Err(MathError::NotDivisible("Division not possible".to_string()))
        }
    }
}

// ============================================================================
// ParentMethods - Methods for ring structures
// ============================================================================

/// Methods for ring structures (parents), corresponding to SageMath's `Rings.ParentMethods`
///
/// This trait provides operations on rings as algebraic structures including:
/// - Ring property tests (is_commutative, is_field, is_integral_domain, etc.)
/// - Characteristic computation
/// - Ideal construction
/// - Quotient ring formation
/// - Localization
///
/// Reference: https://doc.sagemath.org/html/en/reference/categories/sage/categories/rings.html#sage.categories.rings.Rings.ParentMethods
pub trait RingParentMethods {
    /// The type of elements in this ring
    type Element: Ring;

    /// Returns true (this is always a ring)
    ///
    /// This method exists for consistency with SageMath's category system.
    fn is_ring(&self) -> bool {
        true
    }

    /// Check if this ring is commutative
    ///
    /// A ring is commutative if `a * b = b * a` for all elements `a, b`.
    ///
    /// # Default Implementation
    ///
    /// Returns false by default. Concrete commutative ring types should override.
    fn is_commutative(&self) -> bool {
        false
    }

    /// Check if this ring is the zero ring
    ///
    /// The zero ring is the unique ring where 0 = 1.
    fn is_zero_ring(&self) -> bool {
        Self::Element::zero() == Self::Element::one()
    }

    /// Check if this ring is a field
    ///
    /// A ring is a field if every nonzero element has a multiplicative inverse.
    ///
    /// # Arguments
    ///
    /// * `proof` - If true, require a rigorous proof. If false, heuristics may be used.
    ///
    /// # Default Implementation
    ///
    /// Returns false by default. Concrete field types should override.
    fn is_field(&self, _proof: bool) -> bool {
        false
    }

    /// Check if this ring is an integral domain
    ///
    /// A ring is an integral domain if it is commutative, has no zero divisors,
    /// and 0 ≠ 1.
    ///
    /// # Arguments
    ///
    /// * `proof` - If true, require a rigorous proof. If false, heuristics may be used.
    ///
    /// # Default Implementation
    ///
    /// Returns false by default. Concrete integral domain types should override.
    fn is_integral_domain(&self, _proof: bool) -> bool {
        false
    }

    /// Check if this ring is integrally closed
    ///
    /// A ring R is integrally closed if every element of its fraction field
    /// that is integral over R is actually in R.
    ///
    /// # Default Implementation
    ///
    /// Returns false by default. Override for specific ring types.
    fn is_integrally_closed(&self) -> bool {
        false
    }

    /// Check if this ring is Noetherian
    ///
    /// A ring is Noetherian if it satisfies the ascending chain condition
    /// on ideals: every ascending chain of ideals eventually stabilizes.
    ///
    /// # Default Implementation
    ///
    /// Returns false by default. Override for specific ring types.
    fn is_noetherian(&self) -> bool {
        false
    }

    /// Check if this ring is a prime field
    ///
    /// A prime field is either ℚ (characteristic 0) or 𝔽ₚ (characteristic p).
    ///
    /// # Default Implementation
    ///
    /// Returns false by default. Override for ℚ and 𝔽ₚ.
    fn is_prime_field(&self) -> bool {
        false
    }

    /// Compute the characteristic of this ring
    ///
    /// The characteristic is the smallest positive integer n such that
    /// n·1 = 0, or 0 if no such n exists.
    ///
    /// # Default Implementation
    ///
    /// Returns 0 (characteristic zero). Override for finite rings.
    fn characteristic(&self) -> u64 {
        0
    }

    /// Compute the Lie bracket [x, y] = xy - yx
    ///
    /// The Lie bracket measures the non-commutativity of ring multiplication.
    ///
    /// # Arguments
    ///
    /// * `x` - First element
    /// * `y` - Second element
    ///
    /// # Returns
    ///
    /// The element `x * y - y * x`
    fn bracket(&self, x: &Self::Element, y: &Self::Element) -> Self::Element {
        x.clone() * y.clone() - y.clone() * x.clone()
    }

    /// Return the order (cardinality) of this ring
    ///
    /// Returns None for infinite rings.
    fn order(&self) -> Option<usize> {
        None
    }

    /// Check if this ring is finite
    fn is_finite(&self) -> bool {
        self.order().is_some()
    }

    /// Check if this ring is infinite
    fn is_infinite(&self) -> bool {
        !self.is_finite()
    }

    /// Return the Krull dimension of this ring
    ///
    /// The Krull dimension is the supremum of lengths of chains of prime ideals.
    ///
    /// # Default Implementation
    ///
    /// Returns None (unknown). Override for specific ring types.
    fn krull_dimension(&self) -> Option<usize> {
        None
    }

    /// Generate a random element of this ring
    ///
    /// # Default Implementation
    ///
    /// Returns zero. Override with better randomization for specific types.
    fn random_element(&self) -> Self::Element {
        Self::Element::zero()
    }

    /// Generate a random nonzero element of this ring
    ///
    /// # Default Implementation
    ///
    /// Returns one. Override with better randomization for specific types.
    fn random_nonzero_element(&self) -> Self::Element {
        Self::Element::one()
    }
}

// ============================================================================
// MorphismMethods - Methods for ring homomorphisms
// ============================================================================

/// Methods for ring homomorphisms, corresponding to SageMath's `Rings.MorphismMethods`
///
/// This trait provides operations on ring homomorphisms including:
/// - Testing injectivity
/// - Testing if morphism is nonzero
/// - Extending to fraction fields
///
/// Reference: https://doc.sagemath.org/html/en/reference/categories/sage/categories/rings.html#sage.categories.rings.Rings.MorphismMethods
pub trait RingMorphismMethods {
    /// The domain ring type
    type Domain: Ring;

    /// The codomain ring type
    type Codomain: Ring;

    /// Apply the morphism to an element of the domain
    fn apply(&self, x: &Self::Domain) -> Self::Codomain;

    /// Check if this morphism is injective (one-to-one)
    ///
    /// A ring homomorphism φ: R → S is injective if φ(a) = φ(b) implies a = b,
    /// which is equivalent to ker(φ) = {0}.
    ///
    /// # Default Implementation
    ///
    /// Returns false by default. Override with specific logic for concrete morphisms.
    fn is_injective(&self) -> bool {
        false
    }

    /// Check if this morphism is the zero morphism
    ///
    /// The zero morphism sends all elements to zero. This checks if the
    /// image of the domain's one element is nonzero.
    ///
    /// # Default Implementation
    ///
    /// Checks if the image of 1 is nonzero.
    fn is_nonzero(&self) -> bool {
        !self.apply(&Self::Domain::one()).is_zero()
    }

    /// Check if this morphism is surjective (onto)
    ///
    /// A ring homomorphism φ: R → S is surjective if every element of S
    /// is the image of some element of R.
    ///
    /// # Default Implementation
    ///
    /// Returns false by default. Override with specific logic for concrete morphisms.
    fn is_surjective(&self) -> bool {
        false
    }

    /// Check if this morphism is an isomorphism
    ///
    /// A ring homomorphism is an isomorphism if it is both injective and surjective.
    fn is_isomorphism(&self) -> bool {
        self.is_injective() && self.is_surjective()
    }
}

// ============================================================================
// SubcategoryMethods - Methods for ring subcategories
// ============================================================================

/// Marker trait for rings with no zero divisors (domains)
///
/// A ring has no zero divisors if `a * b = 0` implies `a = 0` or `b = 0`.
/// This corresponds to SageMath's `Rings.NoZeroDivisors()` subcategory.
pub trait NoZeroDivisors: Ring {}

/// Marker trait for rings where all nonzero elements are invertible (division rings/fields)
///
/// This corresponds to SageMath's `Rings.Division()` subcategory.
pub trait Division: Ring {}

/// Marker trait for commutative rings
///
/// A ring is commutative if `a * b = b * a` for all elements.
pub trait Commutative: Ring {}

// ============================================================================
// Automatic implementations
// ============================================================================

// Implement RingElementMethods for all Ring types
impl<R: Ring> RingElementMethods for R {}

// Implement marker traits for standard integer types
impl NoZeroDivisors for i32 {}
impl NoZeroDivisors for i64 {}
impl Commutative for i32 {}
impl Commutative for i64 {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_element_is_unit() {
        // In integers, only ±1 are units
        assert!(1i32.is_unit_element());
        assert!((-1i32).is_unit_element());
        assert!(!0i32.is_unit_element());
        assert!(!2i32.is_unit_element());
        assert!(!(-2i32).is_unit_element());
    }

    #[test]
    fn test_ring_element_inverse_of_unit() {
        // In integers, only ±1 have inverses
        assert_eq!(1i32.inverse_of_unit().unwrap(), 1);
        assert_eq!((-1i32).inverse_of_unit().unwrap(), -1);
        assert!(0i32.inverse_of_unit().is_err());
        assert!(2i32.inverse_of_unit().is_err());
    }

    #[test]
    fn test_ring_element_divide_if_possible() {
        // Exact divisions
        assert_eq!(6i32.divide_if_possible(&2).unwrap(), 3);
        assert_eq!(6i32.divide_if_possible(&3).unwrap(), 2);
        assert_eq!(6i32.divide_if_possible(&6).unwrap(), 1);

        // Non-exact divisions
        assert!(7i32.divide_if_possible(&2).is_err());
        assert!(5i32.divide_if_possible(&3).is_err());

        // Division by zero
        assert!(6i32.divide_if_possible(&0).is_err());
    }

    #[test]
    fn test_ring_element_divide_negative() {
        assert_eq!((-6i32).divide_if_possible(&2).unwrap(), -3);
        assert_eq!(6i32.divide_if_possible(&(-2)).unwrap(), -3);
        assert_eq!((-6i32).divide_if_possible(&(-2)).unwrap(), 3);
    }

    // Example: Simple ring parent for testing
    #[derive(Debug, Clone)]
    struct IntegerRing;

    impl RingParentMethods for IntegerRing {
        type Element = i32;

        fn is_commutative(&self) -> bool {
            true
        }

        fn is_integral_domain(&self, _proof: bool) -> bool {
            true
        }

        fn characteristic(&self) -> u64 {
            0
        }
    }

    #[test]
    fn test_ring_parent_properties() {
        let z = IntegerRing;
        assert!(z.is_ring());
        assert!(z.is_commutative());
        assert!(z.is_integral_domain(true));
        assert!(!z.is_zero_ring());
        assert_eq!(z.characteristic(), 0);
        assert!(z.is_infinite());
    }

    #[test]
    fn test_ring_parent_bracket() {
        let z = IntegerRing;

        // For commutative rings, bracket is always zero
        assert_eq!(z.bracket(&2, &3), 0);
        assert_eq!(z.bracket(&5, &7), 0);
        assert_eq!(z.bracket(&1, &1), 0);
    }

    // Example: Simple morphism for testing
    struct DoubleMap;

    impl RingMorphismMethods for DoubleMap {
        type Domain = i32;
        type Codomain = i32;

        fn apply(&self, x: &i32) -> i32 {
            2 * x
        }

        fn is_injective(&self) -> bool {
            // Only in characteristic 2 would this be non-injective
            true
        }
    }

    #[test]
    fn test_ring_morphism_basic() {
        let f = DoubleMap;
        assert_eq!(f.apply(&3), 6);
        assert_eq!(f.apply(&0), 0);
        assert!(f.is_nonzero()); // f(1) = 2 ≠ 0
        assert!(f.is_injective());
    }
}
