//! Weyl Lie Conformal Algebra
//!
//! The Weyl algebra is the free R[∂]-module with generators a, b and λ-bracket:
//! [a_λ b] = C (central element)
//!
//! Corresponds to sage.algebras.lie_conformal_algebras.weyl_lie_conformal_algebra

use rustmath_core::Ring;
use crate::lie_conformal_algebra::{LieConformalAlgebra, GeneratorIndex};
use crate::graded_lie_conformal_algebra::{GradedLieConformalAlgebra, Degree};
use crate::lie_conformal_algebra_element::LieConformalAlgebraElement;

/// Weyl Lie conformal algebra
#[derive(Clone)]
pub struct WeylLieConformalAlgebra<R: Ring> {
    base_ring: R,
}

impl<R: Ring + Clone> WeylLieConformalAlgebra<R> {
    pub fn new(base_ring: R) -> Self {
        WeylLieConformalAlgebra { base_ring }
    }
}

pub type WeylLCAElement<R> = LieConformalAlgebraElement<R, GeneratorIndex>;

impl<R: Ring + Clone + From<i64>> LieConformalAlgebra<R> for WeylLieConformalAlgebra<R> {
    type Element = WeylLCAElement<R>;

    fn base_ring(&self) -> &R {
        &self.base_ring
    }

    fn ngens(&self) -> Option<usize> {
        Some(3) // a, b, C
    }

    fn generator(&self, i: usize) -> Option<Self::Element> {
        if i < 3 {
            Some(LieConformalAlgebraElement::from_basis(GeneratorIndex::finite(i)))
        } else {
            None
        }
    }

    fn zero(&self) -> Self::Element {
        LieConformalAlgebraElement::zero()
    }
}

impl<R: Ring + Clone + From<i64>> GradedLieConformalAlgebra<R> for WeylLieConformalAlgebra<R> {
    fn generator_degree(&self, index: usize) -> Option<Degree> {
        match index {
            0 | 1 => Some(Degree::int(1)), // a, b have degree 1
            2 => Some(Degree::int(0)),      // C has degree 0
            _ => None,
        }
    }

    fn degree(&self, _element: &Self::Element) -> Option<Degree> {
        None
    }
}
