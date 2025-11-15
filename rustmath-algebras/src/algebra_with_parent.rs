//! Demonstrating Parent trait usage with algebras
//!
//! This module shows how to integrate the Parent trait with our algebra
//! implementations to provide proper parent/element relationships.

use rustmath_core::{Ring, Parent, ParentWithBasis, ParentWithGenerators};
use crate::down_up_algebra::{DownUpAlgebra, Element as DownUpElement, DownUpBasisIndex};

/// DownUpAlgebra with Parent implementation
///
/// This wraps the DownUpAlgebra to provide Parent trait functionality
impl<R: Ring> Parent for DownUpAlgebra<R> {
    type Element = DownUpElement<R>;

    fn contains(&self, _element: &Self::Element) -> bool {
        // All DownUpElements belong to this algebra
        // In a more sophisticated implementation, we could check if the element
        // was actually created from this specific algebra instance
        true
    }

    fn zero(&self) -> Option<Self::Element> {
        Some(self.zero())
    }

    fn one(&self) -> Option<Self::Element> {
        Some(self.one())
    }

    fn cardinality(&self) -> Option<usize> {
        // The down-up algebra is infinite-dimensional
        None
    }

    fn name(&self) -> String {
        format!("Down-Up Algebra DU({}, {}, {})",
                self.alpha(), self.beta(), self.gamma())
    }
}

impl<R: Ring> ParentWithGenerators for DownUpAlgebra<R> {
    fn generators(&self) -> Vec<Self::Element> {
        let (d, u) = self.algebra_generators();
        vec![d, u]
    }

    fn num_generators(&self) -> usize {
        2
    }
}

/// Example: A simple algebra parent with finite basis
#[derive(Debug, Clone)]
pub struct SimpleAlgebra<R: Ring> {
    dimension: usize,
    base_ring: std::marker::PhantomData<R>,
}

impl<R: Ring> SimpleAlgebra<R> {
    pub fn new(dimension: usize) -> Self {
        SimpleAlgebra {
            dimension,
            base_ring: std::marker::PhantomData,
        }
    }
}

/// Element type for SimpleAlgebra
#[derive(Debug, Clone, PartialEq)]
pub struct SimpleElement<R: Ring> {
    pub coefficients: Vec<R>,
}

impl<R: Ring> Parent for SimpleAlgebra<R> {
    type Element = SimpleElement<R>;

    fn contains(&self, element: &Self::Element) -> bool {
        element.coefficients.len() == self.dimension
    }

    fn zero(&self) -> Option<Self::Element> {
        Some(SimpleElement {
            coefficients: vec![R::zero(); self.dimension],
        })
    }

    fn one(&self) -> Option<Self::Element> {
        let mut coeffs = vec![R::zero(); self.dimension];
        if self.dimension > 0 {
            coeffs[0] = R::one();
        }
        Some(SimpleElement { coefficients: coeffs })
    }

    fn cardinality(&self) -> Option<usize> {
        // Infinite if base ring is infinite
        None
    }

    fn is_finite(&self) -> bool {
        false
    }

    fn name(&self) -> String {
        format!("Simple Algebra of dimension {}", self.dimension)
    }
}

impl<R: Ring> ParentWithBasis for SimpleAlgebra<R> {
    type BasisIndex = usize;

    fn dimension(&self) -> Option<usize> {
        Some(self.dimension)
    }

    fn basis_element(&self, index: &Self::BasisIndex) -> Option<Self::Element> {
        if *index < self.dimension {
            let mut coeffs = vec![R::zero(); self.dimension];
            coeffs[*index] = R::one();
            Some(SimpleElement { coefficients: coeffs })
        } else {
            None
        }
    }

    fn basis_indices(&self) -> Vec<Self::BasisIndex> {
        (0..self.dimension).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_down_up_algebra_as_parent() {
        let algebra = DownUpAlgebra::new(
            Integer::from(1),
            Integer::from(0),
            Integer::from(0),
        );

        // Test Parent trait methods
        assert_eq!(algebra.name(), "Down-Up Algebra DU(1, 0, 0)");
        assert!(!algebra.is_finite());
        assert_eq!(algebra.cardinality(), None);

        // Test zero and one (Parent trait returns Option)
        let zero = Parent::zero(&algebra).unwrap();
        assert!(zero.is_zero());

        let one = Parent::one(&algebra).unwrap();
        assert!(!one.is_zero());

        // Test generators
        assert_eq!(algebra.num_generators(), 2);
        let gens = algebra.generators();
        assert_eq!(gens.len(), 2);
    }

    #[test]
    fn test_simple_algebra_as_parent() {
        let algebra: SimpleAlgebra<Integer> = SimpleAlgebra::new(3);

        // Test Parent trait
        assert_eq!(algebra.name(), "Simple Algebra of dimension 3");
        assert!(!algebra.is_finite());

        // Test zero element
        let zero = algebra.zero().unwrap();
        assert!(algebra.contains(&zero));
        assert_eq!(zero.coefficients, vec![Integer::zero(); 3]);

        // Test one element
        let one = algebra.one().unwrap();
        assert!(algebra.contains(&one));
        assert_eq!(one.coefficients[0], Integer::one());

        // Test basis
        assert_eq!(algebra.dimension(), Some(3));
        let b0 = algebra.basis_element(&0).unwrap();
        assert_eq!(b0.coefficients[0], Integer::one());
        assert_eq!(b0.coefficients[1], Integer::zero());

        let b1 = algebra.basis_element(&1).unwrap();
        assert_eq!(b1.coefficients[1], Integer::one());

        // Invalid index
        assert!(algebra.basis_element(&10).is_none());

        // Test basis indices
        let indices = algebra.basis_indices();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_element_membership() {
        let algebra: SimpleAlgebra<Integer> = SimpleAlgebra::new(3);

        // Valid element
        let elem = SimpleElement {
            coefficients: vec![Integer::from(1), Integer::from(2), Integer::from(3)],
        };
        assert!(algebra.contains(&elem));

        // Invalid element (wrong dimension)
        let bad_elem = SimpleElement {
            coefficients: vec![Integer::from(1), Integer::from(2)],
        };
        assert!(!algebra.contains(&bad_elem));
    }
}
