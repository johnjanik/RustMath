//! Parent Trait - Algebraic Structure Container
//!
//! The Parent trait represents algebraic structures that contain elements,
//! such as rings, groups, modules, and algebras. It provides the interface
//! for creating and validating elements.
//!
//! Corresponds to sage.structure.parent.Parent
//!
//! # Layering note (decision record, P2-G)
//!
//! [`Parent`] is the substrate for the coercion vocabulary in
//! [`crate::coercion`] ([`Coercible`](crate::coercion::Coercible) /
//! [`Pushout`](crate::coercion::Pushout)). The higher-level, graph-driven
//! coercion system lives in `rustmath-category` (`core_bridge` module), which
//! depends on this crate — never the reverse. Core stays a pure trait
//! vocabulary with no category dependency.

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

    /// The number of elements, when it is known and fits in `usize` (MAGMA `#R`,
    /// Handbook Ch. 17 §17.4.2).
    ///
    /// `Some(n)`: exactly `n` elements. `None` makes **no claim**: the parent may be
    /// infinite, finite with more than `usize::MAX` elements (GF(2^64+13), GR(2^64, 1)), or
    /// of uncomputed size. Never infer infiniteness from `None`; use [`Parent::is_finite`].
    /// Large finite parents expose their exact `Integer` order inherently
    /// (`FiniteField::order`, `GaloisRing::cardinality`, `Zmod::order` in
    /// `rustmath-finitefields`).
    fn cardinality(&self) -> Option<usize> {
        None
    }

    /// MAGMA `IsFinite(R)` (Handbook Ch. 17 §17.4.3: "true if R is known finite; false if
    /// known infinite; error if unknown"): `Some(true)` known finite, `Some(false)` known
    /// infinite, `None` undecided (where MAGMA raises an error).
    ///
    /// The default derives only what [`Parent::cardinality`] proves: `Some(_)` gives
    /// `Some(true)`, otherwise `None`. It never returns `Some(false)`; an implementor that
    /// knows it is infinite (or finite beyond `usize`) overrides this.
    fn is_finite(&self) -> Option<bool> {
        self.cardinality().map(|_| true)
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
        assert_eq!(set.is_finite(), Some(true));
    }

    // A parent that overrides nothing makes no finiteness claim.
    #[derive(Debug, Clone)]
    struct Opaque;

    impl Parent for Opaque {
        type Element = i32;

        fn contains(&self, _element: &Self::Element) -> bool {
            true
        }
    }

    #[test]
    fn test_default_is_finite_makes_no_claim() {
        assert_eq!(Opaque.cardinality(), None);
        assert_eq!(Opaque.is_finite(), None);
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
            None // no claim (see `Parent::cardinality`)
        }

        // R^n over the reals is infinite for n >= 1 and {0} for n = 0.
        fn is_finite(&self) -> Option<bool> {
            Some(self.dimension == 0)
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
        assert_eq!(v3.is_finite(), Some(false));
        assert_eq!(VectorSpace { dimension: 0 }.is_finite(), Some(true));

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
