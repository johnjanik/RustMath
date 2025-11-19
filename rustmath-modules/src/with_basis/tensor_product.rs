//! Tensor products of modules with basis
//!
//! This module implements the tensor product M ⊗ N of two modules with basis.
//! For modules M and N with bases {e_i} and {f_j}, the tensor product has basis
//! {e_i ⊗ f_j} indexed by pairs (i, j).

use rustmath_core::{Ring, Parent, ParentWithBasis};
use crate::with_basis::element::ModuleWithBasisElement;
use crate::with_basis::parent::ModuleWithBasis;
use std::fmt::{self, Debug};

/// Index type for tensor product basis elements
///
/// A tensor product basis element e_i ⊗ f_j is indexed by the pair (i, j)
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TensorIndex<I1, I2> {
    pub left_index: I1,
    pub right_index: I2,
}

impl<I1, I2> TensorIndex<I1, I2> {
    pub fn new(left_index: I1, right_index: I2) -> Self {
        TensorIndex {
            left_index,
            right_index,
        }
    }
}

impl<I1: fmt::Display, I2: fmt::Display> fmt::Display for TensorIndex<I1, I2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}⊗{}", self.left_index, self.right_index)
    }
}

/// Tensor product of two modules with basis
///
/// For modules M and N over ring R, the tensor product M ⊗_R N is a module
/// with basis indexed by pairs of basis indices from M and N.
#[derive(Clone, Debug)]
pub struct TensorProduct<M1, M2>
where
    M1: ModuleWithBasis,
    M2: ModuleWithBasis<BaseRing = M1::BaseRing>,
{
    left: M1,
    right: M2,
}

impl<M1, M2> TensorProduct<M1, M2>
where
    M1: ModuleWithBasis,
    M2: ModuleWithBasis<BaseRing = M1::BaseRing>,
{
    /// Create a new tensor product
    pub fn new(left: M1, right: M2) -> Self {
        TensorProduct { left, right }
    }

    /// Get the left factor
    pub fn left(&self) -> &M1 {
        &self.left
    }

    /// Get the right factor
    pub fn right(&self) -> &M2 {
        &self.right
    }

    /// Compute the tensor product of two elements
    ///
    /// For x = Σ a_i e_i and y = Σ b_j f_j, compute
    /// x ⊗ y = Σ_ij a_i b_j (e_i ⊗ f_j)
    pub fn tensor(
        &self,
        left_elem: &ModuleWithBasisElement<M1::BasisIndex, M1::BaseRing>,
        right_elem: &ModuleWithBasisElement<M2::BasisIndex, M1::BaseRing>,
    ) -> ModuleWithBasisElement<TensorIndex<M1::BasisIndex, M2::BasisIndex>, M1::BaseRing>
    where
        M1::BasisIndex: Ord + Clone,
        M2::BasisIndex: Ord + Clone,
        M1::BaseRing: Ring,
    {
        let mut terms = Vec::new();

        for (i, a_i) in left_elem.items() {
            for (j, b_j) in right_elem.items() {
                let coeff = a_i.clone() * b_j.clone();
                if !coeff.is_zero() {
                    terms.push((TensorIndex::new(i.clone(), j.clone()), coeff));
                }
            }
        }

        ModuleWithBasisElement::from_terms(terms)
    }

    /// Create a pure tensor e_i ⊗ f_j
    pub fn pure_tensor(
        &self,
        left_idx: &M1::BasisIndex,
        right_idx: &M2::BasisIndex,
    ) -> ModuleWithBasisElement<TensorIndex<M1::BasisIndex, M2::BasisIndex>, M1::BaseRing>
    where
        M1::BasisIndex: Ord + Clone,
        M2::BasisIndex: Ord + Clone,
        M1::BaseRing: Ring,
    {
        ModuleWithBasisElement::from_basis_element(
            TensorIndex::new(left_idx.clone(), right_idx.clone()),
            M1::BaseRing::one(),
        )
    }

    /// Universal property: construct a bilinear map from tensor product
    ///
    /// Given a bilinear map f: M × N → P, there exists a unique linear map
    /// g: M ⊗ N → P such that g(x ⊗ y) = f(x, y)
    pub fn from_bilinear_map<P, F>(
        &self,
        _target: P,
        _bilinear_map: F,
    ) -> Option<ModuleWithBasisElement<TensorIndex<M1::BasisIndex, M2::BasisIndex>, M1::BaseRing>>
    where
        P: ModuleWithBasis<BaseRing = M1::BaseRing>,
        F: Fn(&M1::Element, &M2::Element) -> P::Element,
        M1::BasisIndex: Ord + Clone,
        M2::BasisIndex: Ord + Clone,
        M1::BaseRing: Ring,
    {
        // Placeholder - proper implementation would construct the morphism
        None
    }

    /// Associativity isomorphism: (M ⊗ N) ⊗ P ≅ M ⊗ (N ⊗ P)
    pub fn associator<M3>(
        &self,
        _third: &M3,
    ) -> Option<ModuleWithBasisElement<TensorIndex<M1::BasisIndex, M2::BasisIndex>, M1::BaseRing>>
    where
        M3: ModuleWithBasis<BaseRing = M1::BaseRing>,
        M1::BasisIndex: Ord + Clone,
        M2::BasisIndex: Ord + Clone,
        M1::BaseRing: Ring,
    {
        // Placeholder
        None
    }

    /// Symmetry isomorphism: M ⊗ N ≅ N ⊗ M via x ⊗ y ↦ y ⊗ x
    pub fn swap(
        &self,
        elem: &ModuleWithBasisElement<TensorIndex<M1::BasisIndex, M2::BasisIndex>, M1::BaseRing>,
    ) -> ModuleWithBasisElement<TensorIndex<M2::BasisIndex, M1::BasisIndex>, M1::BaseRing>
    where
        M1::BasisIndex: Ord + Clone,
        M2::BasisIndex: Ord + Clone,
        M1::BaseRing: Ring,
    {
        let mut terms = Vec::new();

        for (idx, coeff) in elem.items() {
            terms.push((
                TensorIndex::new(idx.right_index.clone(), idx.left_index.clone()),
                coeff.clone(),
            ));
        }

        ModuleWithBasisElement::from_terms(terms)
    }
}

impl<M1, M2> Parent for TensorProduct<M1, M2>
where
    M1: ModuleWithBasis,
    M2: ModuleWithBasis<BaseRing = M1::BaseRing>,
    M1::BasisIndex: Ord + Clone + Debug,
    M2::BasisIndex: Ord + Clone + Debug,
    M1::BaseRing: Ring,
{
    type Element = ModuleWithBasisElement<TensorIndex<M1::BasisIndex, M2::BasisIndex>, M1::BaseRing>;

    fn contains(&self, _element: &Self::Element) -> bool {
        true
    }

    fn zero(&self) -> Option<Self::Element> {
        Some(ModuleWithBasisElement::zero())
    }
}

impl<M1, M2> ParentWithBasis for TensorProduct<M1, M2>
where
    M1: ModuleWithBasis,
    M2: ModuleWithBasis<BaseRing = M1::BaseRing>,
    M1::BasisIndex: Ord + Clone + Debug,
    M2::BasisIndex: Ord + Clone + Debug,
    M1::BaseRing: Ring,
{
    type BasisIndex = TensorIndex<M1::BasisIndex, M2::BasisIndex>;

    fn dimension(&self) -> Option<usize> {
        match (self.left.dimension(), self.right.dimension()) {
            (Some(d1), Some(d2)) => Some(d1 * d2),
            _ => None,
        }
    }

    fn basis_element(&self, index: &Self::BasisIndex) -> Option<Self::Element> {
        Some(ModuleWithBasisElement::from_basis_element(
            index.clone(),
            M1::BaseRing::one(),
        ))
    }

    fn basis_indices(&self) -> Vec<Self::BasisIndex> {
        let mut indices = Vec::new();

        for left_idx in self.left.basis_indices() {
            for right_idx in self.right.basis_indices() {
                indices.push(TensorIndex::new(left_idx.clone(), right_idx.clone()));
            }
        }

        indices
    }
}

impl<M1, M2> ModuleWithBasis for TensorProduct<M1, M2>
where
    M1: ModuleWithBasis,
    M2: ModuleWithBasis<BaseRing = M1::BaseRing>,
    M1::BasisIndex: Ord + Clone + Debug,
    M2::BasisIndex: Ord + Clone + Debug,
    M1::BaseRing: Ring,
    M1::Element: Clone,
    M2::Element: Clone,
{
    type BaseRing = M1::BaseRing;

    fn base_ring(&self) -> &Self::BaseRing {
        self.left.base_ring()
    }

    fn module_zero(&self) -> Self::Element {
        ModuleWithBasisElement::zero()
    }

    fn is_module_zero(&self, elem: &Self::Element) -> bool {
        elem.is_zero()
    }

    fn module_add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.add(b)
    }

    fn module_negate(&self, a: &Self::Element) -> Self::Element {
        a.negate()
    }

    fn module_scalar_mul(&self, scalar: &Self::BaseRing, elem: &Self::Element) -> Self::Element {
        elem.scalar_mul(scalar)
    }

    fn from_basis_element(&self, index: &Self::BasisIndex, coefficient: Self::BaseRing) -> Option<Self::Element> {
        Some(ModuleWithBasisElement::from_basis_element(
            index.clone(),
            coefficient,
        ))
    }
}

/// Tensor power: M^⊗n = M ⊗ M ⊗ ... ⊗ M (n times)
pub fn tensor_power<M>(module: &M, n: usize) -> Option<M>
where
    M: ModuleWithBasis + Clone,
{
    if n == 0 {
        None // The 0-th tensor power is the base ring (not implemented here)
    } else if n == 1 {
        Some(module.clone())
    } else {
        // For n > 1, would need to construct iterated tensor product
        // This is a placeholder
        Some(module.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::with_basis::parent::FreeModuleWithBasis;
    use num_bigint::BigInt;

    #[test]
    fn test_tensor_index() {
        let idx = TensorIndex::new(1, 2);
        assert_eq!(idx.left_index, 1);
        assert_eq!(idx.right_index, 2);
    }

    #[test]
    fn test_tensor_product_creation() {
        let base_ring = BigInt::from(0);
        let m1 = FreeModuleWithBasis::standard(base_ring.clone(), 2);
        let m2 = FreeModuleWithBasis::standard(base_ring, 3);

        let tensor = TensorProduct::new(m1, m2);

        // dim(M ⊗ N) = dim(M) * dim(N) = 2 * 3 = 6
        assert_eq!(tensor.dimension(), Some(6));
    }

    #[test]
    fn test_basis_indices() {
        let base_ring = BigInt::from(0);
        let m1 = FreeModuleWithBasis::standard(base_ring.clone(), 2);
        let m2 = FreeModuleWithBasis::standard(base_ring, 2);

        let tensor = TensorProduct::new(m1, m2);
        let indices = tensor.basis_indices();

        assert_eq!(indices.len(), 4);

        // Check we have (0,0), (0,1), (1,0), (1,1)
        assert!(indices.contains(&TensorIndex::new(0, 0)));
        assert!(indices.contains(&TensorIndex::new(0, 1)));
        assert!(indices.contains(&TensorIndex::new(1, 0)));
        assert!(indices.contains(&TensorIndex::new(1, 1)));
    }

    #[test]
    fn test_tensor_of_basis_elements() {
        let base_ring = BigInt::from(0);
        let m1 = FreeModuleWithBasis::standard(base_ring.clone(), 2);
        let m2 = FreeModuleWithBasis::standard(base_ring, 2);

        let tensor = TensorProduct::new(m1.clone(), m2.clone());

        let e0 = m1.basis_element(&0).unwrap();
        let f1 = m2.basis_element(&1).unwrap();

        let result = tensor.tensor(&e0, &f1);

        // Should be e_0 ⊗ f_1
        assert_eq!(result.coefficient(&TensorIndex::new(0, 1)), BigInt::from(1));
        assert_eq!(result.coefficient(&TensorIndex::new(0, 0)), BigInt::from(0));
        assert_eq!(result.coefficient(&TensorIndex::new(1, 1)), BigInt::from(0));
    }

    #[test]
    fn test_tensor_bilinearity() {
        let base_ring = BigInt::from(0);
        let m1 = FreeModuleWithBasis::standard(base_ring.clone(), 2);
        let m2 = FreeModuleWithBasis::standard(base_ring, 2);

        let tensor = TensorProduct::new(m1.clone(), m2.clone());

        // (2*e_0 + 3*e_1) ⊗ (5*f_0 + 7*f_1)
        let left = ModuleWithBasisElement::from_terms(vec![
            (0, BigInt::from(2)),
            (1, BigInt::from(3)),
        ]);

        let right = ModuleWithBasisElement::from_terms(vec![
            (0, BigInt::from(5)),
            (1, BigInt::from(7)),
        ]);

        let result = tensor.tensor(&left, &right);

        // Should get 10*(e_0⊗f_0) + 14*(e_0⊗f_1) + 15*(e_1⊗f_0) + 21*(e_1⊗f_1)
        assert_eq!(result.coefficient(&TensorIndex::new(0, 0)), BigInt::from(10));
        assert_eq!(result.coefficient(&TensorIndex::new(0, 1)), BigInt::from(14));
        assert_eq!(result.coefficient(&TensorIndex::new(1, 0)), BigInt::from(15));
        assert_eq!(result.coefficient(&TensorIndex::new(1, 1)), BigInt::from(21));
    }

    #[test]
    fn test_pure_tensor() {
        let base_ring = BigInt::from(0);
        let m1 = FreeModuleWithBasis::standard(base_ring.clone(), 2);
        let m2 = FreeModuleWithBasis::standard(base_ring, 3);

        let tensor = TensorProduct::new(m1, m2);

        let pure = tensor.pure_tensor(&0, &2);

        assert_eq!(pure.coefficient(&TensorIndex::new(0, 2)), BigInt::from(1));
        assert_eq!(pure.num_nonzero(), 1);
    }

    #[test]
    fn test_swap() {
        let base_ring = BigInt::from(0);
        let m1 = FreeModuleWithBasis::standard(base_ring.clone(), 2);
        let m2 = FreeModuleWithBasis::standard(base_ring, 2);

        let tensor = TensorProduct::new(m1, m2);

        let elem = ModuleWithBasisElement::from_terms(vec![
            (TensorIndex::new(0, 1), BigInt::from(3)),
            (TensorIndex::new(1, 0), BigInt::from(5)),
        ]);

        let swapped = tensor.swap(&elem);

        // e_0⊗f_1 should become f_1⊗e_0
        assert_eq!(swapped.coefficient(&TensorIndex::new(1, 0)), BigInt::from(3));
        // e_1⊗f_0 should become f_0⊗e_1
        assert_eq!(swapped.coefficient(&TensorIndex::new(0, 1)), BigInt::from(5));
    }

    #[test]
    fn test_tensor_module_operations() {
        let base_ring = BigInt::from(0);
        let m1 = FreeModuleWithBasis::standard(base_ring.clone(), 2);
        let m2 = FreeModuleWithBasis::standard(base_ring, 2);

        let tensor = TensorProduct::new(m1, m2);

        let elem1 = ModuleWithBasisElement::from_terms(vec![
            (TensorIndex::new(0, 0), BigInt::from(2)),
            (TensorIndex::new(1, 1), BigInt::from(3)),
        ]);

        let elem2 = ModuleWithBasisElement::from_terms(vec![
            (TensorIndex::new(0, 1), BigInt::from(5)),
            (TensorIndex::new(1, 1), BigInt::from(1)),
        ]);

        // Test addition
        let sum = tensor.module_add(&elem1, &elem2);
        assert_eq!(sum.coefficient(&TensorIndex::new(0, 0)), BigInt::from(2));
        assert_eq!(sum.coefficient(&TensorIndex::new(0, 1)), BigInt::from(5));
        assert_eq!(sum.coefficient(&TensorIndex::new(1, 1)), BigInt::from(4));

        // Test scalar multiplication
        let scaled = tensor.module_scalar_mul(&BigInt::from(2), &elem1);
        assert_eq!(scaled.coefficient(&TensorIndex::new(0, 0)), BigInt::from(4));
        assert_eq!(scaled.coefficient(&TensorIndex::new(1, 1)), BigInt::from(6));
    }
}
