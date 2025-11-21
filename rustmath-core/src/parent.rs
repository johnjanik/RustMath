//! Parent Trait - Algebraic Structure Container
//!
//! The Parent trait represents algebraic structures that contain elements,
//! such as rings, groups, modules, and algebras. It provides the interface
//! for creating and validating elements.
//!
//! Corresponds to sage.structure.parent.Parent

use crate::Ring;
use std::fmt::Debug;

/// A Parent is an algebraic structure that contains elements
///
/// Parents represent mathematical structures like rings, groups, modules, etc.
/// They are responsible for:
/// - Creating elements
/// - Validating element membership
/// - Providing structural information (cardinality, generators, etc.)
///
/// # Examples
///
/// A ring of integers, a polynomial ring, or a matrix algebra can all be Parents.
pub trait Parent: Debug + Clone {
    /// The type of elements in this parent
    type Element: Clone + PartialEq;

    /// Check if an element belongs to this parent
    fn contains(&self, element: &Self::Element) -> bool;

    /// Get the zero element (if this is an additive structure)
    fn zero(&self) -> Option<Self::Element> {
        None
    }

    /// Get the one element (if this is a multiplicative structure)
    fn one(&self) -> Option<Self::Element> {
        None
    }

    /// Get the cardinality of this parent (number of elements)
    ///
    /// Returns None for infinite structures
    fn cardinality(&self) -> Option<usize> {
        None
    }

    /// Check if this parent is finite
    fn is_finite(&self) -> bool {
        self.cardinality().is_some()
    }

    /// Get a human-readable name for this parent
    fn name(&self) -> String {
        format!("{:?}", self)
    }
}

/// A parent that has a ring structure
///
/// This extends Parent with ring operations
pub trait RingParent: Parent {
    /// The coefficient ring type
    type Ring: Ring;

    /// Get the base ring
    fn base_ring(&self) -> &Self::Ring;
}

/// A parent that is itself a ring
///
/// For structures like polynomial rings, matrix rings, etc. where
/// the parent structure has ring operations
pub trait ParentAsRing: Parent + Ring {
    // Combines Parent and Ring traits
}

/// A parent with a basis (for modules and algebras)
pub trait ParentWithBasis: Parent {
    /// The type used to index basis elements
    type BasisIndex: Clone + PartialEq;

    /// Get the dimension (rank) of this parent
    fn dimension(&self) -> Option<usize>;

    /// Get the rank (synonym for dimension)
    fn rank(&self) -> Option<usize> {
        self.dimension()
    }

    /// Get a basis element by index
    fn basis_element(&self, index: &Self::BasisIndex) -> Option<Self::Element>;

    /// Get all basis indices
    fn basis_indices(&self) -> Vec<Self::BasisIndex>;
}

/// A parent with generators (for groups, algebras)
pub trait ParentWithGenerators: Parent {
    /// Get the algebra/group generators
    fn generators(&self) -> Vec<Self::Element>;

    /// Get the number of generators
    fn num_generators(&self) -> usize {
        self.generators().len()
    }
}

/// Helper trait for parents that can check equality
pub trait ParentEq: Parent {
    /// Check if two parents are equal (represent the same mathematical structure)
    fn parent_eq(&self, other: &Self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Example: A simple finite set as a Parent
    #[derive(Debug, Clone, PartialEq)]
    struct FiniteSet {
        elements: Vec<i32>,
    }

    impl Parent for FiniteSet {
        type Element = i32;

        fn contains(&self, element: &Self::Element) -> bool {
            self.elements.contains(element)
        }

        fn cardinality(&self) -> Option<usize> {
            Some(self.elements.len())
        }
    }

    #[test]
    fn test_finite_set_parent() {
        let set = FiniteSet {
            elements: vec![1, 2, 3, 4, 5],
        };

        assert!(set.contains(&3));
        assert!(!set.contains(&10));
        assert_eq!(set.cardinality(), Some(5));
        assert!(set.is_finite());
    }

    // Example: A parent with basis
    #[derive(Debug, Clone)]
    struct VectorSpace {
        dimension: usize,
    }

    impl Parent for VectorSpace {
        type Element = Vec<f64>;

        fn contains(&self, element: &Self::Element) -> bool {
            element.len() == self.dimension
        }

        fn zero(&self) -> Option<Self::Element> {
            Some(vec![0.0; self.dimension])
        }

        fn cardinality(&self) -> Option<usize> {
            None // Infinite
        }
    }

    impl ParentWithBasis for VectorSpace {
        type BasisIndex = usize;

        fn dimension(&self) -> Option<usize> {
            Some(self.dimension)
        }

        fn basis_element(&self, index: &Self::BasisIndex) -> Option<Self::Element> {
            if *index < self.dimension {
                let mut vec = vec![0.0; self.dimension];
                vec[*index] = 1.0;
                Some(vec)
            } else {
                None
            }
        }

        fn basis_indices(&self) -> Vec<Self::BasisIndex> {
            (0..self.dimension).collect()
        }
    }

    #[test]
    fn test_vector_space_parent() {
        let v3 = VectorSpace { dimension: 3 };

        assert_eq!(v3.dimension(), Some(3));
        assert!(!v3.is_finite());

        let zero = v3.zero().unwrap();
        assert_eq!(zero, vec![0.0, 0.0, 0.0]);

        let e0 = v3.basis_element(&0).unwrap();
        assert_eq!(e0, vec![1.0, 0.0, 0.0]);

        let e1 = v3.basis_element(&1).unwrap();
        assert_eq!(e1, vec![0.0, 1.0, 0.0]);

        assert!(v3.contains(&vec![1.0, 2.0, 3.0]));
        assert!(!v3.contains(&vec![1.0, 2.0])); // Wrong dimension
    }
}
