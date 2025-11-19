//! Cartesian products of modules with basis
//!
//! This module implements the Cartesian product (direct sum) of modules with basis.
//! For modules M and N, the Cartesian product M × N has basis (e_i, 0) and (0, f_j)
//! where e_i are basis elements of M and f_j are basis elements of N.

use rustmath_core::{Ring, Parent, ParentWithBasis};
use crate::with_basis::element::ModuleWithBasisElement;
use crate::with_basis::parent::ModuleWithBasis;
use std::fmt::{self, Debug};

/// Index type for Cartesian product basis elements
///
/// Elements are indexed by Left(i) for the first component or Right(j) for the second
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProductIndex<I1, I2> {
    Left(I1),
    Right(I2),
}

impl<I1: fmt::Display, I2: fmt::Display> fmt::Display for ProductIndex<I1, I2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProductIndex::Left(i) => write!(f, "L({})", i),
            ProductIndex::Right(j) => write!(f, "R({})", j),
        }
    }
}

/// Cartesian product of two modules with basis
///
/// The Cartesian product M × N is itself a module with basis indexed by
/// ProductIndex<I1, I2>, where I1 and I2 are the index types of M and N.
#[derive(Clone, Debug)]
pub struct CartesianProduct<M1, M2>
where
    M1: ModuleWithBasis,
    M2: ModuleWithBasis<BaseRing = M1::BaseRing>,
{
    left: M1,
    right: M2,
}

impl<M1, M2> CartesianProduct<M1, M2>
where
    M1: ModuleWithBasis,
    M2: ModuleWithBasis<BaseRing = M1::BaseRing>,
{
    /// Create a new Cartesian product
    pub fn new(left: M1, right: M2) -> Self {
        CartesianProduct { left, right }
    }

    /// Get the left component
    pub fn left(&self) -> &M1 {
        &self.left
    }

    /// Get the right component
    pub fn right(&self) -> &M2 {
        &self.right
    }

    /// Inject an element from the left module
    pub fn inject_left(&self, elem: &M1::Element) -> ModuleWithBasisElement<ProductIndex<M1::BasisIndex, M2::BasisIndex>, M1::BaseRing>
    where
        M1::BasisIndex: Ord + Clone,
        M2::BasisIndex: Ord + Clone,
        M1::BaseRing: Ring,
        M1::Element: Clone,
    {
        // Convert element to have ProductIndex::Left indices
        ModuleWithBasisElement::zero() // Placeholder - needs proper implementation
    }

    /// Inject an element from the right module
    pub fn inject_right(&self, elem: &M2::Element) -> ModuleWithBasisElement<ProductIndex<M1::BasisIndex, M2::BasisIndex>, M1::BaseRing>
    where
        M1::BasisIndex: Ord + Clone,
        M2::BasisIndex: Ord + Clone,
        M1::BaseRing: Ring,
        M2::Element: Clone,
    {
        // Convert element to have ProductIndex::Right indices
        ModuleWithBasisElement::zero() // Placeholder - needs proper implementation
    }

    /// Project to the left component
    pub fn project_left(&self, elem: &ModuleWithBasisElement<ProductIndex<M1::BasisIndex, M2::BasisIndex>, M1::BaseRing>) -> ModuleWithBasisElement<M1::BasisIndex, M1::BaseRing>
    where
        M1::BasisIndex: Ord + Clone,
        M2::BasisIndex: Ord + Clone,
        M1::BaseRing: Ring,
    {
        use std::collections::BTreeMap;
        let mut coeffs = BTreeMap::new();

        for (idx, coeff) in elem.items() {
            if let ProductIndex::Left(i) = idx {
                coeffs.insert(i.clone(), coeff.clone());
            }
        }

        ModuleWithBasisElement::new(coeffs)
    }

    /// Project to the right component
    pub fn project_right(&self, elem: &ModuleWithBasisElement<ProductIndex<M1::BasisIndex, M2::BasisIndex>, M1::BaseRing>) -> ModuleWithBasisElement<M2::BasisIndex, M1::BaseRing>
    where
        M1::BasisIndex: Ord + Clone,
        M2::BasisIndex: Ord + Clone,
        M1::BaseRing: Ring,
    {
        use std::collections::BTreeMap;
        let mut coeffs = BTreeMap::new();

        for (idx, coeff) in elem.items() {
            if let ProductIndex::Right(j) = idx {
                coeffs.insert(j.clone(), coeff.clone());
            }
        }

        ModuleWithBasisElement::new(coeffs)
    }
}

/// Type alias for Cartesian product element
pub type CartesianProductElement<I1, I2, R> = ModuleWithBasisElement<ProductIndex<I1, I2>, R>;

impl<M1, M2> Parent for CartesianProduct<M1, M2>
where
    M1: ModuleWithBasis,
    M2: ModuleWithBasis<BaseRing = M1::BaseRing>,
    M1::BasisIndex: Ord + Clone + Debug,
    M2::BasisIndex: Ord + Clone + Debug,
    M1::BaseRing: Ring,
{
    type Element = CartesianProductElement<M1::BasisIndex, M2::BasisIndex, M1::BaseRing>;

    fn contains(&self, _element: &Self::Element) -> bool {
        true // All product elements are contained
    }

    fn zero(&self) -> Option<Self::Element> {
        Some(ModuleWithBasisElement::zero())
    }
}

impl<M1, M2> ParentWithBasis for CartesianProduct<M1, M2>
where
    M1: ModuleWithBasis,
    M2: ModuleWithBasis<BaseRing = M1::BaseRing>,
    M1::BasisIndex: Ord + Clone + Debug,
    M2::BasisIndex: Ord + Clone + Debug,
    M1::BaseRing: Ring,
{
    type BasisIndex = ProductIndex<M1::BasisIndex, M2::BasisIndex>;

    fn dimension(&self) -> Option<usize> {
        match (self.left.dimension(), self.right.dimension()) {
            (Some(d1), Some(d2)) => Some(d1 + d2),
            _ => None,
        }
    }

    fn basis_element(&self, index: &Self::BasisIndex) -> Option<Self::Element> {
        match index {
            ProductIndex::Left(_) => Some(ModuleWithBasisElement::from_basis_element(
                index.clone(),
                M1::BaseRing::one(),
            )),
            ProductIndex::Right(_) => Some(ModuleWithBasisElement::from_basis_element(
                index.clone(),
                M1::BaseRing::one(),
            )),
        }
    }

    fn basis_indices(&self) -> Vec<Self::BasisIndex> {
        let mut indices = Vec::new();

        for idx in self.left.basis_indices() {
            indices.push(ProductIndex::Left(idx));
        }

        for idx in self.right.basis_indices() {
            indices.push(ProductIndex::Right(idx));
        }

        indices
    }
}

impl<M1, M2> ModuleWithBasis for CartesianProduct<M1, M2>
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::with_basis::parent::FreeModuleWithBasis;
    use num_bigint::BigInt;

    #[test]
    fn test_product_index() {
        let left: ProductIndex<usize, String> = ProductIndex::Left(5);
        let right: ProductIndex<usize, String> = ProductIndex::Right("x".to_string());

        assert!(left < right); // Left indices come before Right indices
    }

    #[test]
    fn test_cartesian_product_creation() {
        let base_ring = BigInt::from(0);
        let m1 = FreeModuleWithBasis::standard(base_ring.clone(), 2);
        let m2 = FreeModuleWithBasis::standard(base_ring, 3);

        let product = CartesianProduct::new(m1, m2);

        assert_eq!(product.dimension(), Some(5)); // 2 + 3
    }

    #[test]
    fn test_basis_indices() {
        let base_ring = BigInt::from(0);
        let m1 = FreeModuleWithBasis::standard(base_ring.clone(), 2);
        let m2 = FreeModuleWithBasis::standard(base_ring, 3);

        let product = CartesianProduct::new(m1, m2);
        let indices = product.basis_indices();

        assert_eq!(indices.len(), 5);

        // Check first two are Left indices
        assert!(matches!(indices[0], ProductIndex::Left(0)));
        assert!(matches!(indices[1], ProductIndex::Left(1)));

        // Check last three are Right indices
        assert!(matches!(indices[2], ProductIndex::Right(0)));
        assert!(matches!(indices[3], ProductIndex::Right(1)));
        assert!(matches!(indices[4], ProductIndex::Right(2)));
    }

    #[test]
    fn test_product_element() {
        let base_ring = BigInt::from(0);
        let m1 = FreeModuleWithBasis::standard(base_ring.clone(), 2);
        let m2 = FreeModuleWithBasis::standard(base_ring, 3);

        let product = CartesianProduct::new(m1, m2);

        // Create element (3, 0) + (0, 5)
        let elem = ModuleWithBasisElement::from_terms(vec![
            (ProductIndex::Left(0), BigInt::from(3)),
            (ProductIndex::Right(1), BigInt::from(5)),
        ]);

        assert_eq!(elem.coefficient(&ProductIndex::Left(0)), BigInt::from(3));
        assert_eq!(elem.coefficient(&ProductIndex::Right(1)), BigInt::from(5));
    }

    #[test]
    fn test_projection() {
        let base_ring = BigInt::from(0);
        let m1 = FreeModuleWithBasis::standard(base_ring.clone(), 2);
        let m2 = FreeModuleWithBasis::standard(base_ring, 3);

        let product = CartesianProduct::new(m1, m2);

        let elem = ModuleWithBasisElement::from_terms(vec![
            (ProductIndex::Left(0), BigInt::from(3)),
            (ProductIndex::Left(1), BigInt::from(2)),
            (ProductIndex::Right(1), BigInt::from(5)),
        ]);

        let left_proj = product.project_left(&elem);
        assert_eq!(left_proj.coefficient(&0), BigInt::from(3));
        assert_eq!(left_proj.coefficient(&1), BigInt::from(2));

        let right_proj = product.project_right(&elem);
        assert_eq!(right_proj.coefficient(&1), BigInt::from(5));
    }
}
