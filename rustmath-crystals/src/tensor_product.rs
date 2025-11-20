//! Tensor product of crystals
//!
//! The tensor product of two crystals B₁ ⊗ B₂ is defined with the rule:
//! - e_i(b₁ ⊗ b₂) = e_i(b₁) ⊗ b₂ if φ_i(b₁) ≥ ε_i(b₂)
//! - e_i(b₁ ⊗ b₂) = b₁ ⊗ e_i(b₂) if φ_i(b₁) < ε_i(b₂)
//!
//! Similarly for f_i.

use crate::operators::{Crystal, CrystalElement};
use crate::weight::Weight;
use std::marker::PhantomData;

/// An element in a tensor product crystal
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorElement<E1: CrystalElement, E2: CrystalElement> {
    /// Left component
    pub left: E1,
    /// Right component
    pub right: E2,
}

impl<E1: CrystalElement, E2: CrystalElement> TensorElement<E1, E2> {
    /// Create a new tensor product element
    pub fn new(left: E1, right: E2) -> Self {
        TensorElement { left, right }
    }
}

/// Tensor product of two crystals
pub struct TensorProductCrystal<C1, C2>
where
    C1: Crystal,
    C2: Crystal,
{
    /// Left crystal
    pub left: C1,
    /// Right crystal
    pub right: C2,
}

impl<C1, C2> TensorProductCrystal<C1, C2>
where
    C1: Crystal,
    C2: Crystal,
{
    /// Create a new tensor product crystal
    pub fn new(left: C1, right: C2) -> Self {
        TensorProductCrystal { left, right }
    }
}

impl<C1, C2> Crystal for TensorProductCrystal<C1, C2>
where
    C1: Crystal,
    C2: Crystal,
{
    type Element = TensorElement<C1::Element, C2::Element>;

    fn weight(&self, b: &Self::Element) -> Weight {
        let w1 = self.left.weight(&b.left);
        let w2 = self.right.weight(&b.right);
        &w1 + &w2
    }

    fn e_i(&self, b: &Self::Element, i: usize) -> Option<Self::Element> {
        let phi_left = self.left.phi_i(&b.left, i);
        let eps_right = self.right.epsilon_i(&b.right, i);

        if phi_left >= eps_right {
            // Apply e_i to left component
            self.left.e_i(&b.left, i).map(|new_left| TensorElement {
                left: new_left,
                right: b.right.clone(),
            })
        } else {
            // Apply e_i to right component
            self.right.e_i(&b.right, i).map(|new_right| TensorElement {
                left: b.left.clone(),
                right: new_right,
            })
        }
    }

    fn f_i(&self, b: &Self::Element, i: usize) -> Option<Self::Element> {
        let phi_left = self.left.phi_i(&b.left, i);
        let eps_right = self.right.epsilon_i(&b.right, i);

        if phi_left > eps_right {
            // Apply f_i to left component
            self.left.f_i(&b.left, i).map(|new_left| TensorElement {
                left: new_left,
                right: b.right.clone(),
            })
        } else {
            // Apply f_i to right component
            self.right.f_i(&b.right, i).map(|new_right| TensorElement {
                left: b.left.clone(),
                right: new_right,
            })
        }
    }

    fn elements(&self) -> Vec<Self::Element> {
        let left_elements = self.left.elements();
        let right_elements = self.right.elements();

        let mut result = Vec::new();
        for l in &left_elements {
            for r in &right_elements {
                result.push(TensorElement {
                    left: l.clone(),
                    right: r.clone(),
                });
            }
        }
        result
    }

    fn epsilon_i(&self, b: &Self::Element, i: usize) -> i64 {
        let eps_left = self.left.epsilon_i(&b.left, i);
        let eps_right = self.right.epsilon_i(&b.right, i);
        let phi_left = self.left.phi_i(&b.left, i);

        eps_right.max(eps_left - phi_left)
    }

    fn phi_i(&self, b: &Self::Element, i: usize) -> i64 {
        let phi_left = self.left.phi_i(&b.left, i);
        let phi_right = self.right.phi_i(&b.right, i);
        let eps_right = self.right.epsilon_i(&b.right, i);

        phi_left.max(phi_right + eps_right)
    }
}

/// Helper function to create tensor products
pub fn tensor<C1, C2>(c1: C1, c2: C2) -> TensorProductCrystal<C1, C2>
where
    C1: Crystal,
    C2: Crystal,
{
    TensorProductCrystal::new(c1, c2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::SimpleCrystal;
    use crate::weight::Weight;

    #[test]
    fn test_tensor_product_weight() {
        let c1 = SimpleCrystal {
            rank: 2,
            elements: vec![Weight::new(vec![1, 0]), Weight::new(vec![0, 1])],
        };

        let c2 = SimpleCrystal {
            rank: 2,
            elements: vec![Weight::new(vec![2, 0]), Weight::new(vec![0, 2])],
        };

        let tensor = TensorProductCrystal::new(c1, c2);

        let elem = TensorElement {
            left: Weight::new(vec![1, 0]),
            right: Weight::new(vec![2, 0]),
        };

        let weight = tensor.weight(&elem);
        assert_eq!(weight.coords, vec![3, 0]);
    }

    #[test]
    fn test_tensor_product_elements() {
        let c1 = SimpleCrystal {
            rank: 2,
            elements: vec![Weight::new(vec![1, 0]), Weight::new(vec![0, 1])],
        };

        let c2 = SimpleCrystal {
            rank: 2,
            elements: vec![Weight::new(vec![2, 0])],
        };

        let tensor = TensorProductCrystal::new(c1, c2);
        let elements = tensor.elements();

        // Should have 2 * 1 = 2 elements
        assert_eq!(elements.len(), 2);
    }
}
