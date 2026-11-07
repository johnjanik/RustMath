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

    /// `Some(1)` over the zero ring; otherwise `None` (no claim, see
    /// [`Parent::cardinality`]; the algebra is then infinite, see `is_finite`).
    fn cardinality(&self) -> Option<usize> {
        if R::zero() == R::one() {
            Some(1)
        } else {
            None
        }
    }

    /// MAGMA `IsFinite` (Ch. 17 §17.4.3). The monomials u^i (du)^j d^k form a basis
    /// (Benkart and Roby, "Down-up algebras", J. Algebra 209 (1998); the crate's normal
    /// form `DownUpBasisIndex`), so over a nonzero base ring the u^i, i >= 0, are distinct
    /// and the algebra is infinite. Over the zero ring it is the zero ring.
    ///
    /// `R::zero()`/`R::one()` panic for parent-carrying element types (Galois rings,
    /// p-adics); `DownUpAlgebra`'s own constructors already call them, so such an algebra
    /// cannot be built in the first place.
    fn is_finite(&self) -> Option<bool> {
        Some(R::zero() == R::one())
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

    /// `Some(1)` for dimension 0; otherwise `None`: the size `#R^dim` depends on the base
    /// ring, which this parent does not carry.
    fn cardinality(&self) -> Option<usize> {
        if self.dimension == 0 {
            Some(1)
        } else {
            None
        }
    }

    /// MAGMA `IsFinite` (Ch. 17 §17.4.3). `R^dim` with `dim >= 1` is finite iff `R` is.
    /// `Some(false)` when the element type fixes characteristic 0 (a nonzero ring of
    /// characteristic 0 is infinite); `None` when only the runtime parent of `R` knows
    /// (for `R = IntegerMod` or `FiniteFieldElement` the algebra is finite, of unknown size).
    fn is_finite(&self) -> Option<bool> {
        if self.dimension == 0 {
            Some(true)
        } else if R::static_characteristic() == Some(0) {
            Some(false)
        } else {
            None
        }
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
        assert_eq!(algebra.is_finite(), Some(false));
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
        assert_eq!(algebra.is_finite(), Some(false));

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

    // MAGMA Ch. 17 §17.4.3 IsFinite: decided only where the element type proves it.
    #[test]
    fn test_simple_algebra_finiteness() {
        use rustmath_polynomials::UnivariatePolynomial;
        use rustmath_rationals::Rational;

        // R^0 = {0}.
        let zero_dim: SimpleAlgebra<Integer> = SimpleAlgebra::new(0);
        assert_eq!(zero_dim.is_finite(), Some(true));
        assert_eq!(zero_dim.cardinality(), Some(1));

        // Q^2 is infinite (GAP: IsFinite(Rationals) -> false).
        let q2: SimpleAlgebra<Rational> = SimpleAlgebra::new(2);
        assert_eq!(q2.is_finite(), Some(false));
        assert_eq!(q2.cardinality(), None);

        // The polynomial element type states no characteristic: undecided, not "infinite".
        let poly2: SimpleAlgebra<UnivariatePolynomial<Integer>> = SimpleAlgebra::new(2);
        assert_eq!(poly2.is_finite(), None);
    }

    #[test]
    fn test_down_up_algebra_over_q_is_infinite() {
        use rustmath_rationals::Rational;
        let algebra = DownUpAlgebra::new(
            Rational::from_integer(1),
            Rational::from_integer(1),
            Rational::from_integer(0),
        );
        assert_eq!(algebra.is_finite(), Some(false));
        assert_eq!(algebra.cardinality(), None);
    }
}
