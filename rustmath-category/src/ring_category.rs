//! Ring Category - Category of rings and ring homomorphisms
//!
//! This module implements the category of rings, where:
//! - Objects are rings
//! - Morphisms are ring homomorphisms
//!
//! It provides both the categorical structure and SageMath-style element/parent methods.

use crate::axioms::{Associativity, Commutativity, Distributivity, Identity, Unity, AxiomSet};
use crate::category::Category;
use rustmath_core::{Ring, CommutativeRing, Parent, Result};
use std::fmt;

/// The category of rings
///
/// Objects: Rings (structures with addition and multiplication)
/// Morphisms: Ring homomorphisms (structure-preserving maps)
///
/// # Axioms
/// - Additive group structure (associativity, commutativity, identity, inverse)
/// - Multiplicative associativity
/// - Distributivity: a(b + c) = ab + ac
/// - Unity: multiplicative identity 1
#[derive(Clone, Debug)]
pub struct RingCategory;

impl RingCategory {
    /// Create a new ring category
    pub fn new() -> Self {
        RingCategory
    }
}

impl Default for RingCategory {
    fn default() -> Self {
        Self::new()
    }
}

impl Category for RingCategory {
    fn name(&self) -> &str {
        "Rings"
    }

    fn axioms(&self) -> Vec<&str> {
        vec![
            "closure",
            "associativity",
            "commutativity_additive",
            "identity_additive",
            "inverse_additive",
            "associativity_multiplicative",
            "distributivity",
            "unity",
        ]
    }

    fn description(&self) -> String {
        "Category of rings and ring homomorphisms".to_string()
    }
}

impl fmt::Display for RingCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ring")
    }
}

/// Subcategory of commutative rings
///
/// Rings where multiplication is commutative: ab = ba
#[derive(Clone, Debug)]
pub struct CommutativeRingCategory;

impl CommutativeRingCategory {
    /// Create a new commutative ring category
    pub fn new() -> Self {
        CommutativeRingCategory
    }
}

impl Default for CommutativeRingCategory {
    fn default() -> Self {
        Self::new()
    }
}

impl Category for CommutativeRingCategory {
    fn name(&self) -> &str {
        "CommutativeRings"
    }

    fn super_categories(&self) -> Vec<Box<dyn Category>> {
        vec![Box::new(RingCategory)]
    }

    fn axioms(&self) -> Vec<&str> {
        let mut axioms = RingCategory.axioms();
        axioms.push("commutativity_multiplicative");
        axioms
    }

    fn description(&self) -> String {
        "Category of commutative rings".to_string()
    }
}

impl fmt::Display for CommutativeRingCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CommutativeRing")
    }
}

/// Subcategory of integral domains
///
/// Commutative rings with no zero divisors
#[derive(Clone, Debug)]
pub struct IntegralDomainCategory;

impl IntegralDomainCategory {
    /// Create a new integral domain category
    pub fn new() -> Self {
        IntegralDomainCategory
    }
}

impl Default for IntegralDomainCategory {
    fn default() -> Self {
        Self::new()
    }
}

impl Category for IntegralDomainCategory {
    fn name(&self) -> &str {
        "IntegralDomains"
    }

    fn super_categories(&self) -> Vec<Box<dyn Category>> {
        vec![Box::new(CommutativeRingCategory)]
    }

    fn axioms(&self) -> Vec<&str> {
        let mut axioms = CommutativeRingCategory.axioms();
        axioms.push("no_zero_divisors");
        axioms
    }

    fn description(&self) -> String {
        "Category of integral domains (commutative rings with no zero divisors)".to_string()
    }
}

/// Methods for ring elements (ElementMethods)
///
/// Provides operations that can be performed on ring elements
pub trait RingElementMethods: Ring {
    /// Check if this element is a unit (has multiplicative inverse)
    ///
    /// # Returns
    /// `true` if the element is invertible in the ring
    fn is_unit(&self) -> bool {
        // Default: only 1 and -1 are guaranteed units
        // Subclasses should override for specific rings
        self.is_one() || (*self == -Self::one())
    }

    /// Check if this element is nilpotent
    ///
    /// An element x is nilpotent if x^n = 0 for some positive integer n
    fn is_nilpotent(&self) -> bool {
        // Default: only zero is nilpotent in integral domains
        // Subclasses should override for specific rings
        self.is_zero()
    }

    /// Check if this element is idempotent
    ///
    /// An element x is idempotent if x^2 = x
    fn is_idempotent(&self) -> bool
    where
        Self: Sized,
    {
        let x_squared = self.clone() * self.clone();
        x_squared == *self
    }

    /// Get the multiplicative order of this element
    ///
    /// Returns the smallest positive integer n such that x^n = 1,
    /// or None if no such n exists
    fn multiplicative_order(&self) -> Option<u32> {
        if !self.is_unit() {
            return None;
        }

        let mut current = self.clone();
        let one = Self::one();

        for n in 1..=1000 {
            // Arbitrary limit
            if current == one {
                return Some(n);
            }
            current = current * self.clone();
        }

        None // Order is larger than 1000 or infinite
    }
}

/// Blanket implementation for all rings
impl<R: Ring> RingElementMethods for R {}

/// Methods for ring parent structures (ParentMethods)
///
/// Provides operations on the ring structure itself
pub trait RingParentMethods: Parent {
    /// Get the characteristic of this ring
    ///
    /// The characteristic is the smallest positive integer n such that
    /// n·1 = 0, or 0 if no such n exists.
    fn characteristic(&self) -> u32 {
        0 // Default: characteristic 0
    }

    /// Check if this ring is commutative
    fn is_commutative(&self) -> bool {
        false // Default: not necessarily commutative
    }

    /// Check if this ring is an integral domain
    fn is_integral_domain(&self) -> bool {
        false // Default: not necessarily an integral domain
    }

    /// Check if this ring is a field
    fn is_field(&self) -> bool {
        false // Default: not a field
    }

    /// Check if this ring is finite
    fn is_finite(&self) -> bool {
        self.cardinality().is_some()
    }

    /// Get the axiom set for this ring
    fn axiom_set(&self) -> AxiomSet {
        AxiomSet::ring()
    }
}

/// Subcategory methods for commutative rings
pub trait CommutativeRingParentMethods: RingParentMethods {
    /// Check if this is a principal ideal domain (PID)
    fn is_pid(&self) -> bool {
        false // Default: not necessarily a PID
    }

    /// Check if this is a unique factorization domain (UFD)
    fn is_ufd(&self) -> bool {
        false // Default: not necessarily a UFD
    }

    /// Check if this is a Euclidean domain
    fn is_euclidean(&self) -> bool {
        false // Default: not necessarily Euclidean
    }

    /// Get the axiom set for commutative rings
    fn axiom_set(&self) -> AxiomSet {
        let mut set = AxiomSet::ring();
        set.add_axiom(Commutativity);
        set
    }
}

/// Ring with basis - extends ParentWithBasis for rings
///
/// Rings that have a distinguished basis (like polynomial rings)
pub trait RingWithBasis: Ring + Parent {
    /// Type of basis indices
    type BasisIndex: Clone;

    /// Get a basis element by index
    fn basis_element(&self, index: Self::BasisIndex) -> Self::Element;

    /// Get all basis indices (for finite dimensional rings)
    fn basis_indices(&self) -> Option<Vec<Self::BasisIndex>> {
        None // Default: infinite or not enumerable
    }

    /// Express an element in terms of the basis
    fn coordinates(&self, element: &Self::Element) -> Vec<(Self::BasisIndex, Self)>;
}

/// Standard ring constructions
pub mod constructions {
    use super::*;

    /// Quotient ring R/I for an ideal I
    ///
    /// Elements are cosets r + I
    pub struct QuotientRing<R: Ring> {
        base_ring: R,
        // In a full implementation, would store the ideal
    }

    impl<R: Ring> QuotientRing<R> {
        /// Create a quotient ring R/I
        pub fn new(base_ring: R) -> Self {
            QuotientRing { base_ring }
        }

        /// Get the base ring
        pub fn base_ring(&self) -> &R {
            &self.base_ring
        }
    }

    /// Product ring R × S
    ///
    /// Elements are pairs (r, s)
    pub struct ProductRing<R: Ring, S: Ring> {
        first: R,
        second: S,
    }

    impl<R: Ring, S: Ring> ProductRing<R, S> {
        /// Create a product ring R × S
        pub fn new(first: R, second: S) -> Self {
            ProductRing { first, second }
        }

        /// Get the first component ring
        pub fn first_ring(&self) -> &R {
            &self.first
        }

        /// Get the second component ring
        pub fn second_ring(&self) -> &S {
            &self.second
        }
    }

    /// Matrix ring M_n(R)
    ///
    /// Ring of n×n matrices over R
    pub struct MatrixRing<R: Ring> {
        base_ring: R,
        dimension: usize,
    }

    impl<R: Ring> MatrixRing<R> {
        /// Create a matrix ring M_n(R)
        pub fn new(base_ring: R, dimension: usize) -> Self {
            MatrixRing {
                base_ring,
                dimension,
            }
        }

        /// Get the base ring
        pub fn base_ring(&self) -> &R {
            &self.base_ring
        }

        /// Get the matrix dimension
        pub fn dimension(&self) -> usize {
            self.dimension
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_category_creation() {
        let cat = RingCategory::new();
        assert_eq!(cat.name(), "Rings");
    }

    #[test]
    fn test_ring_category_axioms() {
        let cat = RingCategory;
        let axioms = cat.axioms();
        assert!(axioms.contains(&"associativity"));
        assert!(axioms.contains(&"distributivity"));
        assert!(axioms.contains(&"unity"));
    }

    #[test]
    fn test_ring_category_display() {
        let cat = RingCategory;
        assert_eq!(format!("{}", cat), "Ring");
    }

    #[test]
    fn test_commutative_ring_category() {
        let cat = CommutativeRingCategory::new();
        assert_eq!(cat.name(), "CommutativeRings");

        let axioms = cat.axioms();
        assert!(axioms.contains(&"commutativity_multiplicative"));
    }

    #[test]
    fn test_commutative_ring_is_subcategory() {
        let comm_cat = CommutativeRingCategory;
        let ring_cat = RingCategory;

        assert!(comm_cat.is_subcategory_of(&ring_cat));
    }

    #[test]
    fn test_integral_domain_category() {
        let cat = IntegralDomainCategory::new();
        assert_eq!(cat.name(), "IntegralDomains");

        let axioms = cat.axioms();
        assert!(axioms.contains(&"no_zero_divisors"));
    }

    #[test]
    fn test_integral_domain_is_subcategory() {
        let domain_cat = IntegralDomainCategory;
        let comm_cat = CommutativeRingCategory;
        let ring_cat = RingCategory;

        assert!(domain_cat.is_subcategory_of(&comm_cat));
        assert!(domain_cat.is_subcategory_of(&ring_cat));
    }

    // Test RingElementMethods
    #[test]
    fn test_is_unit_for_integers() {
        assert!(1i32.is_unit());
        assert!((-1i32).is_unit());
        assert!(!2i32.is_unit());
        assert!(!0i32.is_unit());
    }

    #[test]
    fn test_is_nilpotent() {
        assert!(0i32.is_nilpotent());
        assert!(!1i32.is_nilpotent());
        assert!(!2i32.is_nilpotent());
    }

    #[test]
    fn test_is_idempotent() {
        assert!(0i32.is_idempotent()); // 0² = 0
        assert!(1i32.is_idempotent()); // 1² = 1
        assert!(!2i32.is_idempotent()); // 2² = 4 ≠ 2
    }

    #[test]
    fn test_multiplicative_order() {
        assert_eq!(1i32.multiplicative_order(), Some(1));
        assert_eq!((-1i32).multiplicative_order(), Some(2));
        assert_eq!(2i32.multiplicative_order(), None); // Not a unit
    }

    // Test constructions
    #[test]
    fn test_quotient_ring_construction() {
        let qr = constructions::QuotientRing::new(0i32);
        assert_eq!(qr.base_ring(), &0);
    }

    #[test]
    fn test_product_ring_construction() {
        let pr = constructions::ProductRing::new(0i32, 0.0f64);
        assert_eq!(pr.first_ring(), &0);
        assert_eq!(pr.second_ring(), &0.0);
    }

    #[test]
    fn test_matrix_ring_construction() {
        let mr = constructions::MatrixRing::new(0i32, 3);
        assert_eq!(mr.dimension(), 3);
        assert_eq!(mr.base_ring(), &0);
    }

    #[test]
    fn test_ring_category_description() {
        let cat = RingCategory;
        assert!(cat.description().contains("ring"));
    }

    #[test]
    fn test_ring_category_default() {
        let cat1 = RingCategory::new();
        let cat2 = RingCategory::default();
        assert_eq!(cat1.name(), cat2.name());
    }
}
