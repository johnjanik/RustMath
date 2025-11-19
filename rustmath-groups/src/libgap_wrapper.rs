//! LibGAP Wrapper - Base traits for group elements and parents
//!
//! This module provides base abstractions for group elements and their parent structures.
//! In SageMath, these would wrap LibGAP objects, but in RustMath we provide native Rust
//! implementations with similar functionality.
//!
//! # Overview
//!
//! - `ElementLibGAP`: Trait for group elements with comparison and display
//! - `ParentLibGAP`: Trait for parent group structures
//!
//! # Example
//!
//! ```
//! use rustmath_groups::libgap_wrapper::{ElementLibGAP, ParentLibGAP};
//! ```

use std::fmt;
use std::hash::Hash;
use crate::group_traits::GroupElement;

/// Trait for group elements
///
/// This trait provides the base interface for all group elements in RustMath.
/// In SageMath, ElementLibGAP wraps a GAP element object. Here we provide
/// a pure Rust trait-based abstraction.
///
/// # Type Parameters
///
/// This trait is parameterized by the specific element type and provides
/// methods for element comparison, display, and basic operations.
pub trait ElementLibGAP: GroupElement + PartialOrd {
    /// Get a string representation of this element
    fn to_string_repr(&self) -> String {
        format!("{}", self)
    }

    /// Check if this element is less than another
    ///
    /// This provides a canonical ordering for elements, useful for
    /// sorting and canonical form computations.
    fn less_than(&self, other: &Self) -> bool {
        self < other
    }

    /// Get a hash value for this element
    ///
    /// This is useful for storing elements in hash-based collections.
    fn hash_value(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    /// Create a deep copy of this element
    fn deep_copy(&self) -> Self {
        self.clone()
    }
}

/// Trait for parent group structures
///
/// This trait provides the base interface for group parent structures.
/// In SageMath, ParentLibGAP wraps a GAP group object. Here we provide
/// a pure Rust trait-based abstraction.
///
/// A parent structure represents the group itself (not individual elements),
/// containing information about the group structure, generators, and properties.
pub trait ParentLibGAP: Clone + fmt::Debug + fmt::Display {
    /// The element type for this parent group
    type Element: ElementLibGAP;

    /// Get the identity element of the group
    fn identity_element(&self) -> Self::Element;

    /// Get the generators of the group
    ///
    /// Returns a list of group elements that generate the entire group.
    fn generators(&self) -> Vec<Self::Element>;

    /// Get the number of generators
    fn num_generators(&self) -> usize {
        self.generators().len()
    }

    /// Check if an element belongs to this group
    fn contains_element(&self, element: &Self::Element) -> bool;

    /// Get the order (cardinality) of the group
    ///
    /// Returns None for infinite groups, Some(n) for finite groups of order n.
    fn group_order(&self) -> Option<usize>;

    /// Check if the group is finite
    fn is_finite_group(&self) -> bool {
        self.group_order().is_some()
    }

    /// Check if the group is trivial (contains only the identity)
    fn is_trivial_group(&self) -> bool {
        self.group_order() == Some(1)
    }

    /// Get a canonical string representation of this group
    fn canonical_repr(&self) -> String {
        format!("{}", self)
    }

    /// Generate a random element from the group
    ///
    /// For finite groups, returns a uniformly random element.
    /// For infinite groups, this may return a random element from
    /// the generators and their products.
    fn random_element(&self) -> Option<Self::Element> {
        None // Default implementation
    }

    /// Get all elements of the group (for finite groups)
    ///
    /// Returns None for infinite groups or if enumeration is not supported.
    fn all_elements(&self) -> Option<Vec<Self::Element>> {
        None // Default implementation
    }
}

/// A generic wrapper for group elements
///
/// This struct provides a concrete implementation of ElementLibGAP
/// that can wrap any GroupElement type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GenericElementWrapper<E: GroupElement + PartialOrd> {
    element: E,
}

impl<E: GroupElement + PartialOrd> GenericElementWrapper<E> {
    /// Create a new element wrapper
    pub fn new(element: E) -> Self {
        GenericElementWrapper { element }
    }

    /// Get the wrapped element
    pub fn element(&self) -> &E {
        &self.element
    }

    /// Unwrap to get the inner element
    pub fn into_inner(self) -> E {
        self.element
    }
}

impl<E: GroupElement + PartialOrd> fmt::Display for GenericElementWrapper<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.element)
    }
}

impl<E: GroupElement + PartialOrd> PartialOrd for GenericElementWrapper<E> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.element.partial_cmp(&other.element)
    }
}

impl<E: GroupElement + PartialOrd> Ord for GenericElementWrapper<E>
where E: Ord {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.element.cmp(&other.element)
    }
}

impl<E: GroupElement + PartialOrd> GroupElement for GenericElementWrapper<E> {
    fn identity() -> Self {
        GenericElementWrapper::new(E::identity())
    }

    fn inverse(&self) -> Self {
        GenericElementWrapper::new(self.element.inverse())
    }

    fn op(&self, other: &Self) -> Self {
        GenericElementWrapper::new(self.element.op(&other.element))
    }

    fn pow(&self, n: i64) -> Self {
        GenericElementWrapper::new(self.element.pow(n))
    }

    fn order(&self) -> Option<usize> {
        self.element.order()
    }
}

impl<E: GroupElement + PartialOrd> ElementLibGAP for GenericElementWrapper<E> {}

/// A generic parent group wrapper
///
/// This struct provides a concrete implementation of ParentLibGAP
/// that can work with any group element type.
#[derive(Clone, Debug)]
pub struct GenericParentWrapper<E: GroupElement + PartialOrd> {
    generators: Vec<E>,
    order: Option<usize>,
    description: String,
}

impl<E: GroupElement + PartialOrd> GenericParentWrapper<E> {
    /// Create a new parent group from generators
    pub fn new(generators: Vec<E>, order: Option<usize>, description: String) -> Self {
        GenericParentWrapper {
            generators,
            order,
            description,
        }
    }

    /// Create a trivial group (containing only the identity)
    pub fn trivial() -> Self {
        GenericParentWrapper {
            generators: vec![],
            order: Some(1),
            description: "Trivial group".to_string(),
        }
    }
}

impl<E: GroupElement + PartialOrd> fmt::Display for GenericParentWrapper<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description)
    }
}

impl<E: GroupElement + PartialOrd> ParentLibGAP for GenericParentWrapper<E> {
    type Element = GenericElementWrapper<E>;

    fn identity_element(&self) -> Self::Element {
        GenericElementWrapper::new(E::identity())
    }

    fn generators(&self) -> Vec<Self::Element> {
        self.generators
            .iter()
            .map(|g| GenericElementWrapper::new(g.clone()))
            .collect()
    }

    fn contains_element(&self, _element: &Self::Element) -> bool {
        // Default implementation: assume all elements are in the group
        // Subclasses should override this for specific membership tests
        true
    }

    fn group_order(&self) -> Option<usize> {
        self.order
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test element type - integers under addition mod n
    #[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    struct ModInt {
        value: i64,
        modulus: i64,
    }

    impl ModInt {
        fn new(value: i64, modulus: i64) -> Self {
            ModInt {
                value: value.rem_euclid(modulus),
                modulus,
            }
        }
    }

    impl fmt::Display for ModInt {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{} mod {}", self.value, self.modulus)
        }
    }

    impl GroupElement for ModInt {
        fn identity() -> Self {
            ModInt::new(0, 1)
        }

        fn inverse(&self) -> Self {
            ModInt::new(-self.value, self.modulus)
        }

        fn op(&self, other: &Self) -> Self {
            assert_eq!(self.modulus, other.modulus);
            ModInt::new(self.value + other.value, self.modulus)
        }

        fn pow(&self, n: i64) -> Self {
            ModInt::new(self.value * n, self.modulus)
        }

        fn order(&self) -> Option<usize> {
            if self.value == 0 {
                return Some(1);
            }
            for i in 1..=self.modulus {
                if (self.value * i) % self.modulus == 0 {
                    return Some(i as usize);
                }
            }
            None
        }
    }

    #[test]
    fn test_element_wrapper() {
        let elem = GenericElementWrapper::new(ModInt::new(2, 5));
        assert_eq!(elem.element().value, 2);

        let inv = elem.inverse();
        assert_eq!(inv.element().value, 3); // -2 ≡ 3 (mod 5)
    }

    #[test]
    fn test_element_wrapper_op() {
        let a = GenericElementWrapper::new(ModInt::new(2, 5));
        let b = GenericElementWrapper::new(ModInt::new(3, 5));
        let sum = a.op(&b);
        assert_eq!(sum.element().value, 0); // 2 + 3 = 5 ≡ 0 (mod 5)
    }

    #[test]
    fn test_element_wrapper_display() {
        let elem = GenericElementWrapper::new(ModInt::new(3, 7));
        let repr = elem.to_string_repr();
        assert!(repr.contains("3"));
    }

    #[test]
    fn test_element_wrapper_hash() {
        let elem1 = GenericElementWrapper::new(ModInt::new(3, 7));
        let elem2 = GenericElementWrapper::new(ModInt::new(3, 7));
        assert_eq!(elem1.hash_value(), elem2.hash_value());
    }

    #[test]
    fn test_parent_wrapper_trivial() {
        let group: GenericParentWrapper<ModInt> = GenericParentWrapper::trivial();
        assert!(group.is_trivial_group());
        assert_eq!(group.group_order(), Some(1));
    }

    #[test]
    fn test_parent_wrapper_cyclic() {
        let gen = ModInt::new(1, 5);
        let group = GenericParentWrapper::new(
            vec![gen],
            Some(5),
            "Cyclic group of order 5".to_string()
        );

        assert_eq!(group.num_generators(), 1);
        assert!(group.is_finite_group());
        assert_eq!(group.group_order(), Some(5));
    }

    #[test]
    fn test_parent_wrapper_identity() {
        let gen = ModInt::new(1, 7);
        let group = GenericParentWrapper::new(
            vec![gen],
            Some(7),
            "Z/7Z".to_string()
        );

        let id = group.identity_element();
        assert_eq!(id.element().value, 0);
    }

    #[test]
    fn test_parent_wrapper_generators() {
        let gen1 = ModInt::new(1, 10);
        let gen2 = ModInt::new(2, 10);
        let group = GenericParentWrapper::new(
            vec![gen1, gen2],
            Some(10),
            "Test group".to_string()
        );

        let gens = group.generators();
        assert_eq!(gens.len(), 2);
        assert_eq!(gens[0].element().value, 1);
        assert_eq!(gens[1].element().value, 2);
    }

    #[test]
    fn test_element_ordering() {
        let a = GenericElementWrapper::new(ModInt::new(2, 10));
        let b = GenericElementWrapper::new(ModInt::new(5, 10));
        assert!(a.less_than(&b));
        assert!(a < b);
    }

    #[test]
    fn test_element_deep_copy() {
        let a = GenericElementWrapper::new(ModInt::new(3, 7));
        let b = a.deep_copy();
        assert_eq!(a, b);
    }

    #[test]
    fn test_element_power() {
        let a = GenericElementWrapper::new(ModInt::new(2, 7));
        let a_cubed = a.pow(3);
        assert_eq!(a_cubed.element().value, 6); // 2*3 = 6 (mod 7)
    }
}
