//! Cellular Basis
//!
//! Implementation of cellular algebras with a cellular basis structure.
//!
//! A cellular algebra over a commutative ring R has a cell datum (Λ, i, M, C)
//! where Λ is a finite poset. The cellular basis consists of elements
//! {c^λ_st | λ ∈ Λ, s,t ∈ T(λ)} forming a complete basis for the algebra.
//!
//! The structure includes:
//! - A cell poset Λ with ordering ≥
//! - Index sets T(λ) for each λ ∈ Λ
//! - An anti-involution ι: c^λ_st ↦ c^λ_ts
//! - Cell modules C^λ with specific bilinear forms
//!
//! Corresponds to sage.algebras.cellular_basis

use rustmath_core::Ring;
use rustmath_modules::CombinatorialFreeModuleElement;
use std::hash::Hash;
use std::collections::BTreeSet;
use std::marker::PhantomData;

/// Index for a cellular basis element
///
/// Represents a triple (λ, s, t) where:
/// - λ is an element of the cell poset
/// - s, t are indices from the cell module index set T(λ)
#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct CellularIndex<L, T>
where
    L: Ord + Clone,
    T: Ord + Clone,
{
    /// The cell poset element
    pub lambda: L,
    /// First cell module index
    pub s: T,
    /// Second cell module index
    pub t: T,
}

impl<L, T> CellularIndex<L, T>
where
    L: Ord + Clone,
    T: Ord + Clone,
{
    /// Create a new cellular index
    pub fn new(lambda: L, s: T, t: T) -> Self {
        CellularIndex { lambda, s, t }
    }

    /// Apply the anti-involution (swap s and t)
    pub fn anti_involution(&self) -> Self {
        CellularIndex {
            lambda: self.lambda.clone(),
            s: self.t.clone(),
            t: self.s.clone(),
        }
    }
}

/// Cellular basis algebra
///
/// An algebra with a cellular basis structure defined by a cell datum.
/// The basis is indexed by triples (λ, s, t) from the cell poset and
/// associated index sets.
///
/// # Type Parameters
///
/// * `R` - The base ring
/// * `L` - The type for cell poset elements (must be orderable)
/// * `T` - The type for cell module indices (must be orderable)
/// * `CellModuleIndicesFunc` - Function returning indices for each cell
/// * `ProductFunc` - Function computing products on basis elements
///
/// # Examples
///
/// ```
/// use rustmath_algebras::cellular_basis::{CellularBasis, CellularIndex};
/// use rustmath_modules::CombinatorialFreeModuleElement;
/// use rustmath_integers::Integer;
/// use std::collections::BTreeSet;
///
/// // Define cell poset elements
/// let poset = vec![0, 1, 2].into_iter().collect();
///
/// // Define cell module indices function
/// let indices_fn = |lambda: &usize| -> BTreeSet<usize> {
///     (0..=*lambda).collect()
/// };
///
/// // Define product function (simplified)
/// let product_fn = |idx1: &CellularIndex<usize, usize>,
///                   idx2: &CellularIndex<usize, usize>| {
///     CombinatorialFreeModuleElement::from_basis_index(idx1.clone())
/// };
///
/// // Create the cellular basis
/// let cell_basis = CellularBasis::new(poset, indices_fn, product_fn);
/// ```
pub struct CellularBasis<R, L, T, CellModuleIndicesFunc, ProductFunc>
where
    R: Ring,
    L: Ord + Clone + Hash,
    T: Ord + Clone + Hash,
    CellModuleIndicesFunc: Fn(&L) -> BTreeSet<T>,
    ProductFunc: Fn(&CellularIndex<L, T>, &CellularIndex<L, T>)
        -> CombinatorialFreeModuleElement<R, CellularIndex<L, T>>,
{
    /// The cell poset
    cell_poset: BTreeSet<L>,
    /// Function returning cell module indices for each poset element
    cell_module_indices: CellModuleIndicesFunc,
    /// Product on basis elements
    product_on_basis: ProductFunc,
    /// Phantom data
    _phantom: PhantomData<(R, T)>,
}

impl<R, L, T, CellModuleIndicesFunc, ProductFunc>
    CellularBasis<R, L, T, CellModuleIndicesFunc, ProductFunc>
where
    R: Ring,
    L: Ord + Clone + Hash,
    T: Ord + Clone + Hash,
    CellModuleIndicesFunc: Fn(&L) -> BTreeSet<T>,
    ProductFunc: Fn(&CellularIndex<L, T>, &CellularIndex<L, T>)
        -> CombinatorialFreeModuleElement<R, CellularIndex<L, T>>,
{
    /// Create a new cellular basis algebra
    ///
    /// # Arguments
    ///
    /// * `cell_poset` - The cell poset Λ
    /// * `cell_module_indices` - Function returning T(λ) for each λ
    /// * `product_on_basis` - Product function for basis elements
    pub fn new(
        cell_poset: BTreeSet<L>,
        cell_module_indices: CellModuleIndicesFunc,
        product_on_basis: ProductFunc,
    ) -> Self {
        CellularBasis {
            cell_poset,
            cell_module_indices,
            product_on_basis,
            _phantom: PhantomData,
        }
    }

    /// Get the cell poset
    pub fn poset(&self) -> &BTreeSet<L> {
        &self.cell_poset
    }

    /// Get the cell module indices for a given poset element
    ///
    /// # Arguments
    ///
    /// * `lambda` - The poset element
    ///
    /// # Returns
    ///
    /// The set of indices T(λ)
    pub fn indices_for_cell(&self, lambda: &L) -> BTreeSet<T> {
        (self.cell_module_indices)(lambda)
    }

    /// Generate all basis indices
    ///
    /// Returns all valid (λ, s, t) triples for the cellular basis
    pub fn all_basis_indices(&self) -> Vec<CellularIndex<L, T>> {
        let mut indices = Vec::new();

        for lambda in &self.cell_poset {
            let cell_indices = self.indices_for_cell(lambda);
            for s in &cell_indices {
                for t in &cell_indices {
                    indices.push(CellularIndex::new(
                        lambda.clone(),
                        s.clone(),
                        t.clone(),
                    ));
                }
            }
        }

        indices
    }

    /// Create a basis element
    ///
    /// # Arguments
    ///
    /// * `lambda` - Cell poset element
    /// * `s` - First index
    /// * `t` - Second index
    ///
    /// # Returns
    ///
    /// The basis element c^λ_st
    pub fn basis_element(
        &self,
        lambda: L,
        s: T,
        t: T,
    ) -> CombinatorialFreeModuleElement<R, CellularIndex<L, T>> {
        let index = CellularIndex::new(lambda, s, t);
        CombinatorialFreeModuleElement::from_basis_index(index)
    }

    /// Apply the anti-involution to an element
    ///
    /// The anti-involution ι maps c^λ_st to c^λ_ts
    ///
    /// # Arguments
    ///
    /// * `element` - The element to transform
    ///
    /// # Returns
    ///
    /// The image under the anti-involution
    pub fn anti_involution(
        &self,
        element: &CombinatorialFreeModuleElement<R, CellularIndex<L, T>>,
    ) -> CombinatorialFreeModuleElement<R, CellularIndex<L, T>>
    where
        R: Clone,
    {
        let mut result = CombinatorialFreeModuleElement::zero();

        for (index, coeff) in element.iter() {
            let new_index = index.anti_involution();
            let term = CombinatorialFreeModuleElement::monomial(new_index, coeff.clone());
            result = result + term;
        }

        result
    }

    /// Compute the product of two basis elements
    ///
    /// # Arguments
    ///
    /// * `left` - First basis index
    /// * `right` - Second basis index
    ///
    /// # Returns
    ///
    /// The product as a linear combination
    pub fn product_basis(
        &self,
        left: &CellularIndex<L, T>,
        right: &CellularIndex<L, T>,
    ) -> CombinatorialFreeModuleElement<R, CellularIndex<L, T>> {
        (self.product_on_basis)(left, right)
    }

    /// Compute the product of two elements
    ///
    /// # Arguments
    ///
    /// * `left` - First element
    /// * `right` - Second element
    ///
    /// # Returns
    ///
    /// The product element
    pub fn product(
        &self,
        left: &CombinatorialFreeModuleElement<R, CellularIndex<L, T>>,
        right: &CombinatorialFreeModuleElement<R, CellularIndex<L, T>>,
    ) -> CombinatorialFreeModuleElement<R, CellularIndex<L, T>>
    where
        R: Clone,
    {
        let mut result = CombinatorialFreeModuleElement::zero();

        for (left_index, left_coeff) in left.iter() {
            for (right_index, right_coeff) in right.iter() {
                let product_basis = self.product_basis(left_index, right_index);
                let coeff = left_coeff.clone() * right_coeff.clone();
                let term = product_basis.scalar_mul(&coeff);
                result = result + term;
            }
        }

        result
    }

    /// Check if an element is in the cell module C^λ
    ///
    /// An element is in C^λ if all its basis elements have first index λ
    ///
    /// # Arguments
    ///
    /// * `element` - The element to check
    /// * `lambda` - The cell to check membership in
    ///
    /// # Returns
    ///
    /// `true` if element is in C^λ
    pub fn is_in_cell_module(
        &self,
        element: &CombinatorialFreeModuleElement<R, CellularIndex<L, T>>,
        lambda: &L,
    ) -> bool {
        for (index, _coeff) in element.iter() {
            if &index.lambda != lambda {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_cellular_index_creation() {
        let idx = CellularIndex::new(1, 2, 3);
        assert_eq!(idx.lambda, 1);
        assert_eq!(idx.s, 2);
        assert_eq!(idx.t, 3);
    }

    #[test]
    fn test_anti_involution_on_index() {
        let idx = CellularIndex::new(1, 2, 3);
        let inv = idx.anti_involution();

        assert_eq!(inv.lambda, 1);
        assert_eq!(inv.s, 3);
        assert_eq!(inv.t, 2);
    }

    #[test]
    fn test_cellular_basis_creation() {
        let poset: BTreeSet<usize> = vec![0, 1, 2].into_iter().collect();

        let indices_fn = |lambda: &usize| -> BTreeSet<usize> {
            (0..=*lambda).collect()
        };

        let product_fn = |idx1: &CellularIndex<usize, usize>,
                          _idx2: &CellularIndex<usize, usize>| {
            CombinatorialFreeModuleElement::<Integer, CellularIndex<usize, usize>>::from_basis_index(idx1.clone())
        };

        let cell_basis = CellularBasis::new(poset.clone(), indices_fn, product_fn);

        assert_eq!(cell_basis.poset(), &poset);
    }

    #[test]
    fn test_cell_module_indices() {
        let poset: BTreeSet<usize> = vec![0, 1, 2].into_iter().collect();

        let indices_fn = |lambda: &usize| -> BTreeSet<usize> {
            (0..=*lambda).collect()
        };

        let product_fn = |idx1: &CellularIndex<usize, usize>,
                          _idx2: &CellularIndex<usize, usize>| {
            CombinatorialFreeModuleElement::<Integer, CellularIndex<usize, usize>>::from_basis_index(idx1.clone())
        };

        let cell_basis = CellularBasis::new(poset, indices_fn, product_fn);

        let indices_0 = cell_basis.indices_for_cell(&0);
        assert_eq!(indices_0.len(), 1);
        assert!(indices_0.contains(&0));

        let indices_2 = cell_basis.indices_for_cell(&2);
        assert_eq!(indices_2.len(), 3);
        assert!(indices_2.contains(&0));
        assert!(indices_2.contains(&1));
        assert!(indices_2.contains(&2));
    }

    #[test]
    fn test_all_basis_indices() {
        let poset: BTreeSet<usize> = vec![0, 1].into_iter().collect();

        let indices_fn = |lambda: &usize| -> BTreeSet<usize> {
            (0..=*lambda).collect()
        };

        let product_fn = |idx1: &CellularIndex<usize, usize>,
                          _idx2: &CellularIndex<usize, usize>| {
            CombinatorialFreeModuleElement::<Integer, CellularIndex<usize, usize>>::from_basis_index(idx1.clone())
        };

        let cell_basis = CellularBasis::new(poset, indices_fn, product_fn);

        let all_indices = cell_basis.all_basis_indices();

        // For λ=0: (0,0,0) - 1 element
        // For λ=1: (1,0,0), (1,0,1), (1,1,0), (1,1,1) - 4 elements
        // Total: 5 elements
        assert_eq!(all_indices.len(), 5);
    }

    #[test]
    fn test_basis_element_creation() {
        let poset: BTreeSet<usize> = vec![0, 1].into_iter().collect();

        let indices_fn = |lambda: &usize| -> BTreeSet<usize> {
            (0..=*lambda).collect()
        };

        let product_fn = |idx1: &CellularIndex<usize, usize>,
                          _idx2: &CellularIndex<usize, usize>| {
            CombinatorialFreeModuleElement::<Integer, CellularIndex<usize, usize>>::from_basis_index(idx1.clone())
        };

        let cell_basis = CellularBasis::new(poset, indices_fn, product_fn);

        let elem = cell_basis.basis_element(1, 0, 1);

        let expected_idx = CellularIndex::new(1, 0, 1);
        assert_eq!(elem.coefficient(&expected_idx), Integer::from(1));
    }

    #[test]
    fn test_anti_involution_on_element() {
        let poset: BTreeSet<usize> = vec![0, 1].into_iter().collect();

        let indices_fn = |lambda: &usize| -> BTreeSet<usize> {
            (0..=*lambda).collect()
        };

        let product_fn = |idx1: &CellularIndex<usize, usize>,
                          _idx2: &CellularIndex<usize, usize>| {
            CombinatorialFreeModuleElement::<Integer, CellularIndex<usize, usize>>::from_basis_index(idx1.clone())
        };

        let cell_basis = CellularBasis::new(poset, indices_fn, product_fn);

        // Create c^1_01
        let elem = cell_basis.basis_element(1, 0, 1);

        // Apply anti-involution: should give c^1_10
        let inv_elem = cell_basis.anti_involution(&elem);

        let expected_idx = CellularIndex::new(1, 1, 0);
        assert_eq!(inv_elem.coefficient(&expected_idx), Integer::from(1));

        let original_idx = CellularIndex::new(1, 0, 1);
        assert_eq!(inv_elem.coefficient(&original_idx), Integer::from(0));
    }

    #[test]
    fn test_is_in_cell_module() {
        let poset: BTreeSet<usize> = vec![0, 1].into_iter().collect();

        let indices_fn = |lambda: &usize| -> BTreeSet<usize> {
            (0..=*lambda).collect()
        };

        let product_fn = |idx1: &CellularIndex<usize, usize>,
                          _idx2: &CellularIndex<usize, usize>| {
            CombinatorialFreeModuleElement::<Integer, CellularIndex<usize, usize>>::from_basis_index(idx1.clone())
        };

        let cell_basis = CellularBasis::new(poset, indices_fn, product_fn);

        // Element in C^1
        let elem_in_c1 = cell_basis.basis_element(1, 0, 1);
        assert!(cell_basis.is_in_cell_module(&elem_in_c1, &1));
        assert!(!cell_basis.is_in_cell_module(&elem_in_c1, &0));

        // Element in C^0
        let elem_in_c0 = cell_basis.basis_element(0, 0, 0);
        assert!(cell_basis.is_in_cell_module(&elem_in_c0, &0));
        assert!(!cell_basis.is_in_cell_module(&elem_in_c0, &1));
    }
}
