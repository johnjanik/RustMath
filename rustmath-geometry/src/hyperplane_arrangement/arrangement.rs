//! Hyperplane arrangements
//!
//! A hyperplane arrangement is a finite collection of hyperplanes in a vector space.
//! This module provides types for working with hyperplane arrangements and computing
//! their properties and invariants.
//!
//! # Examples
//!
//! ```
//! use rustmath_geometry::hyperplane_arrangement::arrangement::HyperplaneArrangementElement;
//! use rustmath_geometry::hyperplane_arrangement::hyperplane::Hyperplane;
//! use rustmath_integers::Integer;
//!
//! // Create hyperplanes
//! let h1 = Hyperplane::new(vec![Integer::from(1), Integer::from(0)], Integer::from(0));
//! let h2 = Hyperplane::new(vec![Integer::from(0), Integer::from(1)], Integer::from(0));
//!
//! // Create arrangement
//! let arr = HyperplaneArrangementElement::new(vec![h1, h2]);
//!
//! assert_eq!(arr.num_hyperplanes(), 2);
//! assert_eq!(arr.ambient_dimension(), 2);
//! ```

use crate::hyperplane_arrangement::hyperplane::Hyperplane;
use rustmath_core::Ring;
use std::fmt;
use std::collections::HashSet;

/// A hyperplane arrangement
///
/// Represents a finite collection of hyperplanes in a vector space.
/// This is the element type for hyperplane arrangements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HyperplaneArrangementElement<R: Ring> {
    /// The hyperplanes in this arrangement
    hyperplanes: Vec<Hyperplane<R>>,
    /// Cached ambient dimension
    ambient_dim: usize,
}

impl<R: Ring> HyperplaneArrangementElement<R> {
    /// Create a new hyperplane arrangement
    ///
    /// # Arguments
    ///
    /// * `hyperplanes` - The collection of hyperplanes
    ///
    /// # Panics
    ///
    /// Panics if hyperplanes have different ambient dimensions or if the list is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::hyperplane_arrangement::arrangement::HyperplaneArrangementElement;
    /// use rustmath_geometry::hyperplane_arrangement::hyperplane::Hyperplane;
    /// use rustmath_integers::Integer;
    ///
    /// let h1 = Hyperplane::new(vec![Integer::from(1), Integer::from(0)], Integer::from(0));
    /// let h2 = Hyperplane::new(vec![Integer::from(0), Integer::from(1)], Integer::from(0));
    /// let arr = HyperplaneArrangementElement::new(vec![h1, h2]);
    /// ```
    pub fn new(hyperplanes: Vec<Hyperplane<R>>) -> Self {
        if hyperplanes.is_empty() {
            panic!("Hyperplane arrangement must contain at least one hyperplane");
        }

        let ambient_dim = hyperplanes[0].ambient_dimension();

        // Verify all hyperplanes have the same ambient dimension
        for h in &hyperplanes {
            if h.ambient_dimension() != ambient_dim {
                panic!("All hyperplanes must have the same ambient dimension");
            }
        }

        Self {
            hyperplanes,
            ambient_dim,
        }
    }

    /// Create an empty arrangement (for internal use)
    ///
    /// Note: This creates an invalid arrangement and should only be used
    /// when hyperplanes will be added immediately after.
    fn empty(ambient_dim: usize) -> Self {
        Self {
            hyperplanes: vec![],
            ambient_dim,
        }
    }

    /// Get the hyperplanes in this arrangement
    pub fn hyperplanes(&self) -> &[Hyperplane<R>] {
        &self.hyperplanes
    }

    /// Get the number of hyperplanes
    pub fn num_hyperplanes(&self) -> usize {
        self.hyperplanes.len()
    }

    /// Get the ambient dimension
    pub fn ambient_dimension(&self) -> usize {
        self.ambient_dim
    }

    /// Get the dimension of the arrangement
    ///
    /// This is the dimension of the ambient space.
    pub fn dimension(&self) -> usize {
        self.ambient_dim
    }

    /// Compute the rank of the arrangement
    ///
    /// The rank is the dimension of the span of the normal vectors.
    /// For a full implementation, this requires computing the rank of a matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::hyperplane_arrangement::arrangement::HyperplaneArrangementElement;
    /// use rustmath_geometry::hyperplane_arrangement::hyperplane::Hyperplane;
    /// use rustmath_integers::Integer;
    ///
    /// let h1 = Hyperplane::new(vec![Integer::from(1), Integer::from(0)], Integer::from(0));
    /// let h2 = Hyperplane::new(vec![Integer::from(0), Integer::from(1)], Integer::from(0));
    /// let arr = HyperplaneArrangementElement::new(vec![h1, h2]);
    ///
    /// // The rank should be 2 (both normals are independent)
    /// assert_eq!(arr.rank(), 2);
    /// ```
    pub fn rank(&self) -> usize {
        if self.hyperplanes.is_empty() {
            return 0;
        }

        // Compute the rank of the matrix formed by normal vectors
        // For now, we use a simplified computation
        let normals: Vec<Vec<R>> = self.hyperplanes
            .iter()
            .map(|h| h.normal().to_vec())
            .collect();

        // Count linearly independent vectors (simplified)
        compute_rank(&normals)
    }

    /// Check if the arrangement is essential
    ///
    /// An arrangement is essential if the normals span the ambient space,
    /// i.e., rank equals ambient dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::hyperplane_arrangement::arrangement::HyperplaneArrangementElement;
    /// use rustmath_geometry::hyperplane_arrangement::hyperplane::Hyperplane;
    /// use rustmath_integers::Integer;
    ///
    /// let h1 = Hyperplane::new(vec![Integer::from(1), Integer::from(0)], Integer::from(0));
    /// let h2 = Hyperplane::new(vec![Integer::from(0), Integer::from(1)], Integer::from(0));
    /// let arr = HyperplaneArrangementElement::new(vec![h1, h2]);
    ///
    /// assert!(arr.is_essential());
    /// ```
    pub fn is_essential(&self) -> bool {
        self.rank() == self.ambient_dim
    }

    /// Check if the arrangement is central
    ///
    /// An arrangement is central if all hyperplanes pass through the origin.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::hyperplane_arrangement::arrangement::HyperplaneArrangementElement;
    /// use rustmath_geometry::hyperplane_arrangement::hyperplane::Hyperplane;
    /// use rustmath_integers::Integer;
    ///
    /// let h1 = Hyperplane::new(vec![Integer::from(1), Integer::from(0)], Integer::from(0));
    /// let h2 = Hyperplane::new(vec![Integer::from(0), Integer::from(1)], Integer::from(0));
    /// let arr = HyperplaneArrangementElement::new(vec![h1, h2]);
    ///
    /// assert!(arr.is_central());
    /// ```
    pub fn is_central(&self) -> bool {
        // Check if all hyperplanes contain the origin
        let origin = vec![R::zero(); self.ambient_dim];
        self.hyperplanes.iter().all(|h| h.contains(&origin))
    }

    /// Check if the arrangement is linear
    ///
    /// An arrangement is linear if all hyperplanes pass through the origin
    /// (same as is_central).
    pub fn is_linear(&self) -> bool {
        self.is_central()
    }

    /// Check if the arrangement is simplicial
    ///
    /// An arrangement is simplicial if each region has bounding hyperplanes
    /// with linearly independent normal vectors.
    ///
    /// This is a complex property to check; we provide a basic implementation.
    pub fn is_simplicial(&self) -> bool {
        // A full implementation would check all regions
        // For now, we check if we have exactly n hyperplanes in n dimensions
        // with independent normals
        if self.num_hyperplanes() != self.ambient_dim {
            return false;
        }
        self.rank() == self.ambient_dim
    }

    /// Union of this arrangement with another
    ///
    /// Returns a new arrangement containing all hyperplanes from both.
    ///
    /// # Panics
    ///
    /// Panics if arrangements have different ambient dimensions.
    pub fn union(&self, other: &Self) -> Self {
        if self.ambient_dim != other.ambient_dim {
            panic!("Arrangements must have the same ambient dimension");
        }

        let mut hyperplanes = self.hyperplanes.clone();
        hyperplanes.extend(other.hyperplanes.clone());

        Self {
            hyperplanes,
            ambient_dim: self.ambient_dim,
        }
    }

    /// Deletion of a hyperplane
    ///
    /// Returns the arrangement with the specified hyperplane removed.
    ///
    /// # Arguments
    ///
    /// * `index` - Index of the hyperplane to remove
    ///
    /// # Panics
    ///
    /// Panics if index is out of bounds or if removal would leave an empty arrangement.
    pub fn deletion(&self, index: usize) -> Self {
        if index >= self.hyperplanes.len() {
            panic!("Hyperplane index out of bounds");
        }
        if self.hyperplanes.len() <= 1 {
            panic!("Cannot delete from single-hyperplane arrangement");
        }

        let mut hyperplanes = self.hyperplanes.clone();
        hyperplanes.remove(index);

        Self {
            hyperplanes,
            ambient_dim: self.ambient_dim,
        }
    }

    /// Restriction to a hyperplane
    ///
    /// Returns the arrangement restricted to the specified hyperplane.
    /// This is the arrangement induced on the hyperplane by the other hyperplanes.
    ///
    /// # Arguments
    ///
    /// * `index` - Index of the hyperplane to restrict to
    ///
    /// Note: Full implementation requires projecting hyperplanes onto a subspace.
    pub fn restriction(&self, index: usize) -> Option<Self> {
        if index >= self.hyperplanes.len() {
            return None;
        }

        // For a complete implementation, we would:
        // 1. Take the intersection of each hyperplane with hyperplanes[index]
        // 2. Express these in coordinates of the hyperplane
        // This requires substantial linear algebra

        // Placeholder: return a simplified arrangement
        None
    }

    /// Compute sign vector of a point
    ///
    /// Returns a vector indicating which side of each hyperplane the point is on:
    /// - 0 if on the hyperplane
    /// - 1 if positive (for types that support comparison)
    ///
    /// Note: For rings without ordering, this only distinguishes zero from non-zero.
    pub fn sign_vector(&self, point: &[R]) -> Vec<i8> {
        assert_eq!(point.len(), self.ambient_dim, "Point dimension mismatch");

        self.hyperplanes
            .iter()
            .map(|h| {
                let value = h.evaluate(point);
                if value.is_zero() {
                    0
                } else {
                    1 // For general rings, we can't determine sign
                }
            })
            .collect()
    }

    /// Check if a point is contained in any hyperplane
    pub fn contains_point(&self, point: &[R]) -> bool {
        self.hyperplanes.iter().any(|h| h.contains(point))
    }

    /// Get all hyperplanes containing a point
    pub fn hyperplanes_containing(&self, point: &[R]) -> Vec<usize> {
        self.hyperplanes
            .iter()
            .enumerate()
            .filter(|(_, h)| h.contains(point))
            .map(|(i, _)| i)
            .collect()
    }

    /// Compute the number of regions
    ///
    /// This is a complex combinatorial problem. We provide a basic formula
    /// for special cases.
    pub fn num_regions(&self) -> usize {
        // For n hyperplanes in general position in d dimensions:
        // Number of regions = sum_{k=0}^{d} C(n, k)
        // where C(n, k) is the binomial coefficient

        // This is a simplified computation
        // A full implementation would use the characteristic polynomial
        1 + self.num_hyperplanes()
    }
}

impl<R: Ring + fmt::Display> fmt::Display for HyperplaneArrangementElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HyperplaneArrangement({} hyperplanes in {}D space)",
            self.num_hyperplanes(),
            self.ambient_dim
        )
    }
}

/// Parent class for hyperplane arrangements
///
/// This manages the creation and organization of hyperplane arrangements.
#[derive(Clone, Debug)]
pub struct HyperplaneArrangements {
    /// Dimension of the ambient space
    dimension: usize,
}

impl HyperplaneArrangements {
    /// Create a new hyperplane arrangements parent
    ///
    /// # Arguments
    ///
    /// * `dimension` - Dimension of the ambient vector space
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Create an arrangement from hyperplanes
    pub fn arrangement<R: Ring>(&self, hyperplanes: Vec<Hyperplane<R>>) -> HyperplaneArrangementElement<R> {
        for h in &hyperplanes {
            if h.ambient_dimension() != self.dimension {
                panic!("Hyperplane dimension mismatch");
            }
        }
        HyperplaneArrangementElement::new(hyperplanes)
    }
}

/// Compute the rank of a collection of vectors
///
/// This is a simplified implementation that counts linearly independent vectors.
fn compute_rank<R: Ring>(vectors: &[Vec<R>]) -> usize {
    if vectors.is_empty() {
        return 0;
    }

    let dim = vectors[0].len();

    // For a proper implementation, we would use row reduction
    // For now, we use a simplified heuristic

    // Count non-zero vectors
    let mut count = 0;
    let mut seen_directions = HashSet::new();

    for (i, vec) in vectors.iter().enumerate() {
        // Check if vector is non-zero
        if !vec.iter().all(|x| x.is_zero()) {
            // For simplicity, we assume different non-zero vectors are independent
            // A full implementation would check linear dependence
            seen_directions.insert(i);
            count += 1;
            if count >= dim {
                return dim;
            }
        }
    }

    count.min(dim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;
    use rustmath_rationals::Rational;

    #[test]
    fn test_new_arrangement() {
        let h1 = Hyperplane::new(vec![Integer::from(1), Integer::from(0)], Integer::from(0));
        let h2 = Hyperplane::new(vec![Integer::from(0), Integer::from(1)], Integer::from(0));
        let arr = HyperplaneArrangementElement::new(vec![h1, h2]);

        assert_eq!(arr.num_hyperplanes(), 2);
        assert_eq!(arr.ambient_dimension(), 2);
        assert_eq!(arr.dimension(), 2);
    }

    #[test]
    fn test_rank() {
        // Two independent hyperplanes
        let h1 = Hyperplane::new(vec![Integer::from(1), Integer::from(0)], Integer::from(0));
        let h2 = Hyperplane::new(vec![Integer::from(0), Integer::from(1)], Integer::from(0));
        let arr = HyperplaneArrangementElement::new(vec![h1, h2]);

        assert_eq!(arr.rank(), 2);
    }

    #[test]
    fn test_is_essential() {
        let h1 = Hyperplane::new(vec![Integer::from(1), Integer::from(0)], Integer::from(0));
        let h2 = Hyperplane::new(vec![Integer::from(0), Integer::from(1)], Integer::from(0));
        let arr = HyperplaneArrangementElement::new(vec![h1, h2]);

        assert!(arr.is_essential());
    }

    #[test]
    fn test_is_central() {
        // Central arrangement - all pass through origin
        let h1 = Hyperplane::new(vec![Integer::from(1), Integer::from(0)], Integer::from(0));
        let h2 = Hyperplane::new(vec![Integer::from(0), Integer::from(1)], Integer::from(0));
        let arr = HyperplaneArrangementElement::new(vec![h1, h2]);

        assert!(arr.is_central());
        assert!(arr.is_linear());
    }

    #[test]
    fn test_not_central() {
        // Non-central arrangement
        let h1 = Hyperplane::new(vec![Integer::from(1), Integer::from(0)], Integer::from(1));
        let h2 = Hyperplane::new(vec![Integer::from(0), Integer::from(1)], Integer::from(0));
        let arr = HyperplaneArrangementElement::new(vec![h1, h2]);

        assert!(!arr.is_central());
    }

    #[test]
    fn test_union() {
        let h1 = Hyperplane::new(vec![Integer::from(1), Integer::from(0)], Integer::from(0));
        let h2 = Hyperplane::new(vec![Integer::from(0), Integer::from(1)], Integer::from(0));
        let arr1 = HyperplaneArrangementElement::new(vec![h1]);
        let arr2 = HyperplaneArrangementElement::new(vec![h2]);

        let union = arr1.union(&arr2);
        assert_eq!(union.num_hyperplanes(), 2);
    }

    #[test]
    fn test_deletion() {
        let h1 = Hyperplane::new(vec![Integer::from(1), Integer::from(0)], Integer::from(0));
        let h2 = Hyperplane::new(vec![Integer::from(0), Integer::from(1)], Integer::from(0));
        let h3 = Hyperplane::new(vec![Integer::from(1), Integer::from(1)], Integer::from(0));
        let arr = HyperplaneArrangementElement::new(vec![h1, h2, h3]);

        let del = arr.deletion(1);
        assert_eq!(del.num_hyperplanes(), 2);
    }

    #[test]
    fn test_sign_vector() {
        // x = 0 and y = 0
        let h1 = Hyperplane::new(vec![Integer::from(1), Integer::from(0)], Integer::from(0));
        let h2 = Hyperplane::new(vec![Integer::from(0), Integer::from(1)], Integer::from(0));
        let arr = HyperplaneArrangementElement::new(vec![h1, h2]);

        // Point (1, 1) is not on either hyperplane (non-zero evaluation)
        let sv = arr.sign_vector(&[Integer::from(1), Integer::from(1)]);
        assert_eq!(sv, vec![1, 1]);

        // Origin is on both hyperplanes
        let sv_origin = arr.sign_vector(&[Integer::from(0), Integer::from(0)]);
        assert_eq!(sv_origin, vec![0, 0]);

        // Point on one hyperplane
        let sv_one = arr.sign_vector(&[Integer::from(0), Integer::from(5)]);
        assert_eq!(sv_one, vec![0, 1]);
    }

    #[test]
    fn test_contains_point() {
        let h1 = Hyperplane::new(vec![Integer::from(1), Integer::from(0)], Integer::from(0));
        let h2 = Hyperplane::new(vec![Integer::from(0), Integer::from(1)], Integer::from(0));
        let arr = HyperplaneArrangementElement::new(vec![h1, h2]);

        assert!(arr.contains_point(&[Integer::from(0), Integer::from(0)]));
        assert!(arr.contains_point(&[Integer::from(1), Integer::from(0)]));
        assert!(!arr.contains_point(&[Integer::from(1), Integer::from(1)]));
    }

    #[test]
    fn test_hyperplanes_containing() {
        let h1 = Hyperplane::new(vec![Integer::from(1), Integer::from(0)], Integer::from(0));
        let h2 = Hyperplane::new(vec![Integer::from(0), Integer::from(1)], Integer::from(0));
        let arr = HyperplaneArrangementElement::new(vec![h1, h2]);

        let containing = arr.hyperplanes_containing(&[Integer::from(0), Integer::from(0)]);
        assert_eq!(containing, vec![0, 1]);

        let containing_one = arr.hyperplanes_containing(&[Integer::from(1), Integer::from(0)]);
        assert_eq!(containing_one, vec![1]);
    }

    #[test]
    fn test_hyperplane_arrangements_parent() {
        let parent = HyperplaneArrangements::new(2);
        assert_eq!(parent.dimension(), 2);

        let h1 = Hyperplane::new(vec![Integer::from(1), Integer::from(0)], Integer::from(0));
        let h2 = Hyperplane::new(vec![Integer::from(0), Integer::from(1)], Integer::from(0));
        let arr = parent.arrangement(vec![h1, h2]);

        assert_eq!(arr.ambient_dimension(), 2);
    }

    #[test]
    fn test_with_rationals() {
        let h1 = Hyperplane::new(
            vec![Rational::from(1), Rational::from(0)],
            Rational::from(0),
        );
        let h2 = Hyperplane::new(
            vec![Rational::from(0), Rational::from(1)],
            Rational::from(0),
        );
        let arr = HyperplaneArrangementElement::new(vec![h1, h2]);

        assert!(arr.is_central());
        assert_eq!(arr.rank(), 2);
    }

    #[test]
    fn test_display() {
        let h1 = Hyperplane::new(vec![Integer::from(1), Integer::from(0)], Integer::from(0));
        let h2 = Hyperplane::new(vec![Integer::from(0), Integer::from(1)], Integer::from(0));
        let arr = HyperplaneArrangementElement::new(vec![h1, h2]);

        let display = format!("{}", arr);
        assert!(display.contains("HyperplaneArrangement"));
        assert!(display.contains("2 hyperplanes"));
    }
}
