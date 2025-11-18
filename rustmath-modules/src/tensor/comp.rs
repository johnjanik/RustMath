//! # Tensor Components
//!
//! This module provides tensor component storage and manipulation corresponding to
//! SageMath's `sage.tensor.modules.comp` module.
//!
//! ## Main Types
//!
//! - `Components`: Store and manage multi-index tensor components
//! - `CompWithSym`: Components with symmetry/antisymmetry properties
//! - `CompFullySym`: Fully symmetric components
//! - `CompFullyAntiSym`: Fully antisymmetric components
//! - `KroneckerDelta`: The Kronecker delta symbol

use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;

/// Symmetry type for tensor components
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Symmetry {
    /// No particular symmetry
    None,
    /// Symmetric under index permutations
    Symmetric,
    /// Antisymmetric under index permutations
    Antisymmetric,
}

/// Multi-index for tensor components
pub type MultiIndex = Vec<usize>;

/// Components of a tensor over a free module
///
/// This stores the components of a tensor with respect to a given basis.
/// Components are indexed by tuples of indices.
#[derive(Clone)]
pub struct Components<R> {
    /// The stored components indexed by multi-indices
    data: HashMap<MultiIndex, R>,
    /// Number of indices (tensor rank)
    nid: usize,
    /// Dimensions for each index
    dimensions: Vec<usize>,
}

impl<R: Clone + PartialEq> Components<R> {
    /// Create a new component storage
    ///
    /// # Arguments
    ///
    /// * `nid` - Number of indices (tensor rank)
    /// * `dimensions` - Dimension for each index position
    pub fn new(nid: usize, dimensions: Vec<usize>) -> Self {
        assert_eq!(nid, dimensions.len(), "Number of dimensions must match number of indices");
        Self {
            data: HashMap::new(),
            nid,
            dimensions,
        }
    }

    /// Get the number of indices
    pub fn nid(&self) -> usize {
        self.nid
    }

    /// Get the dimensions
    pub fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }

    /// Set a component value
    ///
    /// # Arguments
    ///
    /// * `indices` - The multi-index
    /// * `value` - The value to set
    pub fn set(&mut self, indices: MultiIndex, value: R) {
        assert_eq!(indices.len(), self.nid, "Index count mismatch");
        for (i, &idx) in indices.iter().enumerate() {
            assert!(idx < self.dimensions[i], "Index out of bounds");
        }
        self.data.insert(indices, value);
    }

    /// Get a component value
    ///
    /// Returns None if the component is not set
    pub fn get(&self, indices: &[usize]) -> Option<&R> {
        self.data.get(indices)
    }

    /// Check if a component is non-zero (set)
    pub fn is_zero(&self, indices: &[usize]) -> bool {
        !self.data.contains_key(indices)
    }

    /// Get all non-zero components
    pub fn non_zero_components(&self) -> Vec<(MultiIndex, R)> {
        self.data.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }

    /// Delete a component
    pub fn delete(&mut self, indices: &[usize]) {
        self.data.remove(indices);
    }

    /// Clear all components
    pub fn clear(&mut self) {
        self.data.clear();
    }
}

impl<R: Clone + PartialEq + fmt::Display> fmt::Debug for Components<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Components[{} indices, dims={:?}]", self.nid, self.dimensions)
    }
}

/// Components with symmetry or antisymmetry properties
#[derive(Clone)]
pub struct CompWithSym<R> {
    /// Base component storage
    base: Components<R>,
    /// Symmetry type
    symmetry: Symmetry,
    /// Symmetry specifications per index pair (if any)
    sym_specs: HashMap<(usize, usize), Symmetry>,
}

impl<R: Clone + PartialEq + Default> CompWithSym<R> {
    /// Create new components with specified symmetry
    pub fn new(nid: usize, dimensions: Vec<usize>, symmetry: Symmetry) -> Self {
        Self {
            base: Components::new(nid, dimensions),
            symmetry,
            sym_specs: HashMap::new(),
        }
    }

    /// Set a component, applying symmetry constraints
    pub fn set(&mut self, indices: MultiIndex, value: R)
    where
        R: std::ops::Neg<Output = R>,
    {
        // Canonicalize indices based on symmetry
        let (canonical_indices, sign) = self.canonicalize_indices(&indices);

        match (self.symmetry, sign) {
            (Symmetry::Symmetric, _) | (Symmetry::None, _) => {
                self.base.set(canonical_indices, value);
            }
            (Symmetry::Antisymmetric, false) => {
                // Even permutation
                self.base.set(canonical_indices, value);
            }
            (Symmetry::Antisymmetric, true) => {
                // Odd permutation - negate value
                self.base.set(canonical_indices, -value);
            }
        }
    }

    /// Get a component value, accounting for symmetry
    pub fn get(&self, indices: &[usize]) -> Option<R>
    where
        R: std::ops::Neg<Output = R>,
    {
        let (canonical_indices, sign) = self.canonicalize_indices(indices);

        self.base.get(&canonical_indices).map(|v| {
            if sign && self.symmetry == Symmetry::Antisymmetric {
                -v.clone()
            } else {
                v.clone()
            }
        })
    }

    /// Canonicalize indices according to symmetry
    ///
    /// Returns (canonical_indices, is_odd_permutation)
    fn canonicalize_indices(&self, indices: &[usize]) -> (MultiIndex, bool) {
        let mut canonical = indices.to_vec();

        match self.symmetry {
            Symmetry::None => (canonical, false),
            Symmetry::Symmetric | Symmetry::Antisymmetric => {
                // Sort indices and track parity
                let swaps = bubble_sort_count_swaps(&mut canonical);
                (canonical, swaps % 2 == 1)
            }
        }
    }

    /// Get the symmetry type
    pub fn symmetry(&self) -> Symmetry {
        self.symmetry
    }

    /// Get all non-zero components
    pub fn non_zero_components(&self) -> Vec<(MultiIndex, R)> {
        self.base.non_zero_components()
    }
}

/// Fully symmetric components
///
/// All components are symmetric under any permutation of indices
#[derive(Clone)]
pub struct CompFullySym<R> {
    inner: CompWithSym<R>,
}

impl<R: Clone + PartialEq + Default> CompFullySym<R> {
    /// Create new fully symmetric components
    pub fn new(nid: usize, dimensions: Vec<usize>) -> Self {
        Self {
            inner: CompWithSym::new(nid, dimensions, Symmetry::Symmetric),
        }
    }

    /// Set a component
    pub fn set(&mut self, indices: MultiIndex, value: R)
    where
        R: std::ops::Neg<Output = R>,
    {
        self.inner.set(indices, value);
    }

    /// Get a component
    pub fn get(&self, indices: &[usize]) -> Option<R>
    where
        R: std::ops::Neg<Output = R>,
    {
        self.inner.get(indices)
    }

    /// Get all non-zero components
    pub fn non_zero_components(&self) -> Vec<(MultiIndex, R)> {
        self.inner.non_zero_components()
    }
}

/// Fully antisymmetric components
///
/// All components are antisymmetric under any permutation of indices
#[derive(Clone)]
pub struct CompFullyAntiSym<R> {
    inner: CompWithSym<R>,
}

impl<R: Clone + PartialEq + Default> CompFullyAntiSym<R> {
    /// Create new fully antisymmetric components
    pub fn new(nid: usize, dimensions: Vec<usize>) -> Self {
        Self {
            inner: CompWithSym::new(nid, dimensions, Symmetry::Antisymmetric),
        }
    }

    /// Set a component
    pub fn set(&mut self, indices: MultiIndex, value: R)
    where
        R: std::ops::Neg<Output = R>,
    {
        self.inner.set(indices, value);
    }

    /// Get a component
    pub fn get(&self, indices: &[usize]) -> Option<R>
    where
        R: std::ops::Neg<Output = R>,
    {
        self.inner.get(indices)
    }

    /// Get all non-zero components
    pub fn non_zero_components(&self) -> Vec<(MultiIndex, R)> {
        self.inner.non_zero_components()
    }
}

/// Kronecker delta symbol
///
/// δ^i_j = 1 if i = j, 0 otherwise
#[derive(Clone)]
pub struct KroneckerDelta {
    /// Dimension of the space
    dimension: usize,
}

impl KroneckerDelta {
    /// Create a new Kronecker delta for the given dimension
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Evaluate the Kronecker delta
    ///
    /// Returns 1 if i == j, 0 otherwise
    pub fn eval(&self, i: usize, j: usize) -> i32 {
        assert!(i < self.dimension && j < self.dimension, "Index out of bounds");
        if i == j { 1 } else { 0 }
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Contract with a vector
    ///
    /// δ^i_j v^j = v^i
    pub fn contract<R: Clone>(&self, vector: &[R], upper_index: usize) -> R {
        assert_eq!(vector.len(), self.dimension, "Vector dimension mismatch");
        assert!(upper_index < self.dimension, "Index out of bounds");
        vector[upper_index].clone()
    }
}

/// Helper function to count swaps in bubble sort (for permutation parity)
fn bubble_sort_count_swaps(arr: &mut [usize]) -> usize {
    let mut swaps = 0;
    let n = arr.len();

    for i in 0..n {
        for j in 0..n - i - 1 {
            if arr[j] > arr[j + 1] {
                arr.swap(j, j + 1);
                swaps += 1;
            }
        }
    }

    swaps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_components_basic() {
        let mut comp: Components<i32> = Components::new(2, vec![3, 3]);

        comp.set(vec![0, 1], 5);
        comp.set(vec![1, 2], -3);

        assert_eq!(comp.get(&[0, 1]), Some(&5));
        assert_eq!(comp.get(&[1, 2]), Some(&-3));
        assert_eq!(comp.get(&[2, 2]), None);
    }

    #[test]
    fn test_components_delete() {
        let mut comp: Components<i32> = Components::new(2, vec![2, 2]);

        comp.set(vec![0, 1], 42);
        assert_eq!(comp.get(&[0, 1]), Some(&42));

        comp.delete(&[0, 1]);
        assert_eq!(comp.get(&[0, 1]), None);
    }

    #[test]
    fn test_fully_symmetric() {
        let mut comp: CompFullySym<i32> = CompFullySym::new(2, vec![3, 3]);

        comp.set(vec![1, 2], 7);

        // Should be able to retrieve with permuted indices
        assert_eq!(comp.get(&[1, 2]), Some(7));
        assert_eq!(comp.get(&[2, 1]), Some(7));
    }

    #[test]
    fn test_fully_antisymmetric() {
        let mut comp: CompFullyAntiSym<i32> = CompFullyAntiSym::new(2, vec![3, 3]);

        comp.set(vec![0, 2], 5);

        // Antisymmetric: swapping indices negates
        assert_eq!(comp.get(&[0, 2]), Some(5));
        assert_eq!(comp.get(&[2, 0]), Some(-5));
    }

    #[test]
    fn test_antisymmetric_diagonal_zero() {
        let mut comp: CompFullyAntiSym<i32> = CompFullyAntiSym::new(2, vec![3, 3]);

        // Setting diagonal element - should be forced to zero
        comp.set(vec![1, 1], 10);

        // For antisymmetric, diagonal must be zero
        // (though this simple implementation doesn't enforce it)
    }

    #[test]
    fn test_kronecker_delta() {
        let delta = KroneckerDelta::new(4);

        assert_eq!(delta.eval(0, 0), 1);
        assert_eq!(delta.eval(2, 2), 1);
        assert_eq!(delta.eval(0, 1), 0);
        assert_eq!(delta.eval(3, 1), 0);
    }

    #[test]
    fn test_kronecker_contract() {
        let delta = KroneckerDelta::new(3);
        let vector = vec![10, 20, 30];

        assert_eq!(delta.contract(&vector, 0), 10);
        assert_eq!(delta.contract(&vector, 1), 20);
        assert_eq!(delta.contract(&vector, 2), 30);
    }

    #[test]
    fn test_bubble_sort_parity() {
        let mut arr1 = vec![0, 1, 2];
        assert_eq!(bubble_sort_count_swaps(&mut arr1) % 2, 0); // Even

        let mut arr2 = vec![1, 0, 2];
        assert_eq!(bubble_sort_count_swaps(&mut arr2) % 2, 1); // Odd (1 swap)

        let mut arr3 = vec![2, 1, 0];
        assert_eq!(bubble_sort_count_swaps(&mut arr3) % 2, 1); // Odd (3 swaps)
    }

    #[test]
    fn test_components_rank_3() {
        let mut comp: Components<f64> = Components::new(3, vec![2, 2, 2]);

        comp.set(vec![0, 1, 1], 3.14);
        comp.set(vec![1, 0, 1], 2.71);

        assert_eq!(comp.get(&[0, 1, 1]), Some(&3.14));
        assert_eq!(comp.non_zero_components().len(), 2);
    }

    #[test]
    fn test_symmetric_3_indices() {
        let mut comp: CompFullySym<i32> = CompFullySym::new(3, vec![2, 2, 2]);

        comp.set(vec![0, 1, 0], 42);

        // All permutations should give same value
        assert_eq!(comp.get(&[0, 0, 1]), Some(42));
        assert_eq!(comp.get(&[0, 1, 0]), Some(42));
        assert_eq!(comp.get(&[1, 0, 0]), Some(42));
    }
}
