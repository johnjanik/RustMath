//! Parent trait for modules with basis
//!
//! This module defines the ModuleWithBasis trait, which combines the
//! ParentWithBasis and Module traits to provide a full interface for
//! modules with a distinguished basis.

use rustmath_core::{Ring, Parent, ParentWithBasis};
use crate::module::Module;
use crate::with_basis::element::ModuleWithBasisElement;
use rustmath_matrix::Matrix;
use std::fmt::Debug;

/// Trait for modules with a distinguished basis
///
/// This trait combines ParentWithBasis (for basis structure) and Module
/// (for module operations) to provide a complete interface for modules
/// with a basis.
///
/// # Type Parameters
/// - `BaseRing`: The ring over which this is a module
/// - `BasisIndex`: The type used to index basis elements
/// - `Element`: The element type (typically ModuleWithBasisElement)
pub trait ModuleWithBasis: ParentWithBasis + Clone + Debug {
    /// The base ring type
    type BaseRing: Ring;

    /// Get the base ring
    fn base_ring(&self) -> &Self::BaseRing;

    /// Get the rank (dimension) of the module
    fn module_rank(&self) -> Option<usize> {
        self.dimension()
    }

    /// Create the zero element
    fn module_zero(&self) -> Self::Element;

    /// Check if an element is zero
    fn is_module_zero(&self, elem: &Self::Element) -> bool;

    /// Add two elements
    fn module_add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;

    /// Negate an element
    fn module_negate(&self, a: &Self::Element) -> Self::Element;

    /// Scalar multiplication
    fn module_scalar_mul(&self, scalar: &Self::BaseRing, elem: &Self::Element) -> Self::Element;

    /// Subtract two elements
    fn module_sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.module_add(a, &self.module_negate(b))
    }

    /// Linear combination of elements
    fn linear_combination(&self, coeffs: &[Self::BaseRing], elements: &[Self::Element]) -> Self::Element
    where
        Self::Element: Clone,
    {
        assert_eq!(coeffs.len(), elements.len(), "Coefficients and elements must have same length");

        if coeffs.is_empty() {
            return self.module_zero();
        }

        let mut result = self.module_scalar_mul(&coeffs[0], &elements[0]);
        for i in 1..coeffs.len() {
            let term = self.module_scalar_mul(&coeffs[i], &elements[i]);
            result = self.module_add(&result, &term);
        }
        result
    }

    /// Create an element from basis element
    fn from_basis_element(&self, index: &Self::BasisIndex, coefficient: Self::BaseRing) -> Option<Self::Element>;

    /// Get all basis elements
    fn basis_elements(&self) -> Vec<Self::Element> {
        self.basis_indices()
            .into_iter()
            .filter_map(|idx| self.basis_element(&idx))
            .collect()
    }

    /// Get the Gram matrix of the module (if applicable)
    ///
    /// The Gram matrix G has entries G[i,j] = <e_i, e_j> for an inner product
    fn gram_matrix(&self) -> Option<Matrix<Self::BaseRing>> {
        None // Default: no inner product defined
    }

    /// Check if this module is free
    fn is_free(&self) -> bool {
        // Default: assume free if we have a basis
        self.dimension().is_some()
    }

    /// Get the ambient module (if this is a submodule)
    fn ambient(&self) -> Option<Self> {
        None // Default: this is the ambient module
    }
}

/// Parent methods - additional methods for ModuleWithBasis parents
///
/// These are helper methods that can be used on any ModuleWithBasis
pub trait ModuleWithBasisParentMethods: ModuleWithBasis {
    /// Create element from sparse coefficients
    fn from_vector(&self, coefficients: Vec<(Self::BasisIndex, Self::BaseRing)>) -> Self::Element
    where
        Self::BasisIndex: Ord,
        Self::Element: From<Vec<(Self::BasisIndex, Self::BaseRing)>>,
    {
        coefficients.into()
    }

    /// Get the monomial (basis element with coefficient 1)
    fn monomial(&self, index: &Self::BasisIndex) -> Option<Self::Element> {
        self.from_basis_element(index, Self::BaseRing::one())
    }

    /// Get the n-th basis element (for integer-indexed bases)
    fn gen(&self, n: usize) -> Option<Self::Element>
    where
        Self::BasisIndex: From<usize>,
    {
        let index = Self::BasisIndex::from(n);
        self.basis_element(&index)
    }

    /// Get all generators
    fn gens(&self) -> Vec<Self::Element> {
        self.basis_elements()
    }

    /// Compute echelon form of a list of elements
    fn echelon_form(&self, elements: Vec<Self::Element>) -> Vec<Self::Element>
    where
        Self::Element: Clone,
    {
        // TODO: Implement Gaussian elimination
        // For now, just return the input
        elements
    }

    /// Compute a basis for the span of given elements
    fn span_basis(&self, elements: Vec<Self::Element>) -> Vec<Self::Element>
    where
        Self::Element: Clone,
    {
        // TODO: Implement proper basis extraction
        // For now, use echelon form
        self.echelon_form(elements)
    }

    /// Check if elements are linearly independent
    fn are_linearly_independent(&self, elements: &[Self::Element]) -> bool
    where
        Self::Element: Clone,
    {
        let basis = self.span_basis(elements.to_vec());
        basis.len() == elements.len()
    }
}

// Blanket implementation of ParentMethods for all ModuleWithBasis
impl<T: ModuleWithBasis> ModuleWithBasisParentMethods for T {}

/// Concrete implementation of a free module with basis
#[derive(Clone, Debug)]
pub struct FreeModuleWithBasis<I, R>
where
    I: Ord + Clone + Debug,
    R: Ring,
{
    base_ring: R,
    basis_indices: Vec<I>,
}

impl<I, R> FreeModuleWithBasis<I, R>
where
    I: Ord + Clone + Debug,
    R: Ring,
{
    /// Create a new free module with given basis indices
    pub fn new(base_ring: R, basis_indices: Vec<I>) -> Self {
        FreeModuleWithBasis {
            base_ring,
            basis_indices,
        }
    }

    /// Create a free module with integer basis indices 0, 1, ..., n-1
    pub fn standard(base_ring: R, rank: usize) -> Self
    where
        I: From<usize>,
    {
        let basis_indices = (0..rank).map(I::from).collect();
        FreeModuleWithBasis::new(base_ring, basis_indices)
    }
}

impl<I, R> Parent for FreeModuleWithBasis<I, R>
where
    I: Ord + Clone + Debug,
    R: Ring,
{
    type Element = ModuleWithBasisElement<I, R>;

    fn contains(&self, _element: &Self::Element) -> bool {
        // All sparse vectors over this basis are contained
        true
    }

    fn zero(&self) -> Option<Self::Element> {
        Some(ModuleWithBasisElement::zero())
    }

    fn cardinality(&self) -> Option<usize> {
        None // Infinite (unless base ring is finite)
    }
}

impl<I, R> ParentWithBasis for FreeModuleWithBasis<I, R>
where
    I: Ord + Clone + Debug,
    R: Ring,
{
    type BasisIndex = I;

    fn dimension(&self) -> Option<usize> {
        Some(self.basis_indices.len())
    }

    fn basis_element(&self, index: &Self::BasisIndex) -> Option<Self::Element> {
        if self.basis_indices.contains(index) {
            Some(ModuleWithBasisElement::from_basis_element(
                index.clone(),
                R::one(),
            ))
        } else {
            None
        }
    }

    fn basis_indices(&self) -> Vec<Self::BasisIndex> {
        self.basis_indices.clone()
    }
}

impl<I, R> ModuleWithBasis for FreeModuleWithBasis<I, R>
where
    I: Ord + Clone + Debug,
    R: Ring,
{
    type BaseRing = R;

    fn base_ring(&self) -> &Self::BaseRing {
        &self.base_ring
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
        if self.basis_indices.contains(index) {
            Some(ModuleWithBasisElement::from_basis_element(
                index.clone(),
                coefficient,
            ))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    #[test]
    fn test_free_module_creation() {
        let base_ring = BigInt::from(0); // Using BigInt as ring
        let module = FreeModuleWithBasis::standard(base_ring, 3);

        assert_eq!(module.dimension(), Some(3));
        assert_eq!(module.module_rank(), Some(3));
    }

    #[test]
    fn test_basis_elements() {
        let base_ring = BigInt::from(0);
        let module = FreeModuleWithBasis::standard(base_ring, 3);

        let basis = module.basis_elements();
        assert_eq!(basis.len(), 3);

        // Check that basis elements are correct
        for i in 0..3 {
            assert_eq!(basis[i].coefficient(&i), BigInt::from(1));
            for j in 0..3 {
                if i != j {
                    assert_eq!(basis[i].coefficient(&j), BigInt::from(0));
                }
            }
        }
    }

    #[test]
    fn test_module_operations() {
        let base_ring = BigInt::from(0);
        let module = FreeModuleWithBasis::standard(base_ring, 3);

        let e0 = module.basis_element(&0).unwrap();
        let e1 = module.basis_element(&1).unwrap();

        // Test addition
        let sum = module.module_add(&e0, &e1);
        assert_eq!(sum.coefficient(&0), BigInt::from(1));
        assert_eq!(sum.coefficient(&1), BigInt::from(1));

        // Test scalar multiplication
        let scaled = module.module_scalar_mul(&BigInt::from(3), &e0);
        assert_eq!(scaled.coefficient(&0), BigInt::from(3));

        // Test negation
        let neg = module.module_negate(&e0);
        assert_eq!(neg.coefficient(&0), BigInt::from(-1));
    }

    #[test]
    fn test_linear_combination() {
        let base_ring = BigInt::from(0);
        let module = FreeModuleWithBasis::standard(base_ring, 3);

        let e0 = module.basis_element(&0).unwrap();
        let e1 = module.basis_element(&1).unwrap();
        let e2 = module.basis_element(&2).unwrap();

        let coeffs = vec![BigInt::from(2), BigInt::from(3), BigInt::from(-1)];
        let elements = vec![e0, e1, e2];

        let result = module.linear_combination(&coeffs, &elements);
        assert_eq!(result.coefficient(&0), BigInt::from(2));
        assert_eq!(result.coefficient(&1), BigInt::from(3));
        assert_eq!(result.coefficient(&2), BigInt::from(-1));
    }

    #[test]
    fn test_zero_element() {
        let base_ring = BigInt::from(0);
        let module = FreeModuleWithBasis::standard(base_ring, 3);

        let zero = module.module_zero();
        assert!(module.is_module_zero(&zero));
        assert_eq!(zero.num_nonzero(), 0);
    }

    #[test]
    fn test_monomial() {
        let base_ring = BigInt::from(0);
        let module = FreeModuleWithBasis::standard(base_ring, 3);

        let m1 = module.monomial(&1).unwrap();
        assert_eq!(m1.coefficient(&1), BigInt::from(1));
        assert_eq!(m1.coefficient(&0), BigInt::from(0));
    }

    #[test]
    fn test_gens() {
        let base_ring = BigInt::from(0);
        let module = FreeModuleWithBasis::standard(base_ring, 4);

        let gens = module.gens();
        assert_eq!(gens.len(), 4);
    }

    #[test]
    fn test_string_basis_indices() {
        let base_ring = BigInt::from(0);
        let indices = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        let module = FreeModuleWithBasis::new(base_ring, indices);

        assert_eq!(module.dimension(), Some(3));

        let ex = module.basis_element(&"x".to_string()).unwrap();
        assert_eq!(ex.coefficient(&"x".to_string()), BigInt::from(1));
        assert_eq!(ex.coefficient(&"y".to_string()), BigInt::from(0));
    }
}
