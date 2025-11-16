//! Collection of points in the same ambient space
//!
//! This module provides the `PointCollection` type for managing collections
//! of points that belong to the same ambient space. It is particularly useful
//! in toric geometry for representing cone rays, polytope vertices, and facet
//! normals.
//!
//! # Examples
//!
//! ```
//! use rustmath_geometry::point_collection::PointCollection;
//! use rustmath_integers::Integer;
//!
//! // Create a collection of 2D points
//! let points = vec![
//!     vec![Integer::from(1), Integer::from(0)],
//!     vec![Integer::from(0), Integer::from(1)],
//!     vec![Integer::from(1), Integer::from(1)],
//! ];
//! let collection = PointCollection::new(2, points);
//!
//! assert_eq!(collection.len(), 3);
//! assert_eq!(collection.dimension(), 2);
//! ```

use rustmath_core::Ring;
use rustmath_matrix::Matrix;
use std::fmt;

/// A collection of points in the same ambient space
///
/// Represents a set of points that all belong to the same vector space or lattice.
/// This is useful for geometric computations where multiple points need to be
/// treated as a unified object.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PointCollection<R: Ring> {
    /// The dimension of the ambient space
    dimension: usize,
    /// The points in the collection
    points: Vec<Vec<R>>,
}

impl<R: Ring> PointCollection<R> {
    /// Create a new point collection
    ///
    /// # Arguments
    ///
    /// * `dimension` - The dimension of the ambient space
    /// * `points` - The points in the collection
    ///
    /// # Panics
    ///
    /// Panics if any point has a dimension different from the specified dimension
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::point_collection::PointCollection;
    /// use rustmath_integers::Integer;
    ///
    /// let points = vec![
    ///     vec![Integer::from(1), Integer::from(0)],
    ///     vec![Integer::from(0), Integer::from(1)],
    /// ];
    /// let collection = PointCollection::new(2, points);
    /// assert_eq!(collection.len(), 2);
    /// ```
    pub fn new(dimension: usize, points: Vec<Vec<R>>) -> Self {
        // Validate that all points have the correct dimension
        for (i, point) in points.iter().enumerate() {
            assert_eq!(
                point.len(),
                dimension,
                "Point {} has dimension {}, expected {}",
                i,
                point.len(),
                dimension
            );
        }

        Self { dimension, points }
    }

    /// Create an empty point collection
    pub fn empty(dimension: usize) -> Self {
        Self {
            dimension,
            points: Vec::new(),
        }
    }

    /// Get the dimension of the ambient space
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the number of points in the collection
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if the collection is empty
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Get a point by index
    pub fn get(&self, index: usize) -> Option<&[R]> {
        self.points.get(index).map(|p| p.as_slice())
    }

    /// Get all points
    pub fn points(&self) -> &[Vec<R>] {
        &self.points
    }

    /// Find the index of a point
    ///
    /// Returns None if the point is not in the collection
    pub fn index(&self, point: &[R]) -> Option<usize> {
        self.points.iter().position(|p| p.as_slice() == point)
    }

    /// Check if a point is in the collection
    pub fn contains(&self, point: &[R]) -> bool {
        self.index(point).is_some()
    }

    /// Add a point to the collection
    ///
    /// # Panics
    ///
    /// Panics if the point has a dimension different from the collection's dimension
    pub fn add_point(&mut self, point: Vec<R>) {
        assert_eq!(
            point.len(),
            self.dimension,
            "Point has dimension {}, expected {}",
            point.len(),
            self.dimension
        );
        self.points.push(point);
    }

    /// Create a matrix where each row is a point
    ///
    /// Returns None if the collection is empty
    pub fn to_row_matrix(&self) -> Option<Matrix<R>> {
        if self.is_empty() {
            return None;
        }

        let mut data = Vec::new();
        for point in &self.points {
            data.extend(point.iter().cloned());
        }

        Matrix::from_vec(self.points.len(), self.dimension, data).ok()
    }

    /// Create a matrix where each column is a point
    ///
    /// Returns None if the collection is empty
    pub fn to_column_matrix(&self) -> Option<Matrix<R>> {
        if self.is_empty() {
            return None;
        }

        let mut data = Vec::new();
        for i in 0..self.dimension {
            for point in &self.points {
                data.push(point[i].clone());
            }
        }

        Matrix::from_vec(self.dimension, self.points.len(), data).ok()
    }

    /// Combine two point collections
    ///
    /// # Panics
    ///
    /// Panics if the collections have different dimensions
    pub fn extend(&mut self, other: &PointCollection<R>) {
        assert_eq!(
            self.dimension, other.dimension,
            "Cannot extend collections with different dimensions"
        );
        self.points.extend(other.points.iter().cloned());
    }

    /// Create a new collection by combining two collections
    ///
    /// # Panics
    ///
    /// Panics if the collections have different dimensions
    pub fn concatenate(&self, other: &PointCollection<R>) -> Self {
        assert_eq!(
            self.dimension, other.dimension,
            "Cannot concatenate collections with different dimensions"
        );

        let mut points = self.points.clone();
        points.extend(other.points.iter().cloned());

        Self {
            dimension: self.dimension,
            points,
        }
    }

    /// Get an iterator over the points
    pub fn iter(&self) -> impl Iterator<Item = &[R]> {
        self.points.iter().map(|p| p.as_slice())
    }

    /// Create a Cartesian product with another collection
    ///
    /// Each point in the result is the concatenation of a point from self
    /// and a point from other.
    pub fn cartesian_product(&self, other: &PointCollection<R>) -> PointCollection<R> {
        let new_dimension = self.dimension + other.dimension;
        let mut points = Vec::new();

        for p1 in &self.points {
            for p2 in &other.points {
                let mut point = p1.clone();
                point.extend(p2.iter().cloned());
                points.push(point);
            }
        }

        PointCollection {
            dimension: new_dimension,
            points,
        }
    }

    /// Check if all points are distinct
    pub fn has_duplicates(&self) -> bool {
        for i in 0..self.points.len() {
            for j in (i + 1)..self.points.len() {
                if self.points[i] == self.points[j] {
                    return true;
                }
            }
        }
        false
    }

    /// Remove duplicate points
    pub fn deduplicate(&mut self) {
        let mut unique_points = Vec::new();
        for point in &self.points {
            if !unique_points.contains(point) {
                unique_points.push(point.clone());
            }
        }
        self.points = unique_points;
    }

    /// Create a new collection with only unique points
    pub fn deduplicated(&self) -> Self {
        let mut result = self.clone();
        result.deduplicate();
        result
    }
}

impl<R: Ring + fmt::Display> fmt::Display for PointCollection<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            write!(f, "[]")
        } else {
            write!(f, "[")?;
            for (i, point) in self.points.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "(")?;
                for (j, coord) in point.iter().enumerate() {
                    if j > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", coord)?;
                }
                write!(f, ")")?;
            }
            write!(f, "]")
        }
    }
}

/// Check if an object is a PointCollection (type guard)
///
/// In Rust, this is mainly for documentation purposes and testing.
pub fn is_point_collection<R: Ring>(_obj: &PointCollection<R>) -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_creation() {
        let points = vec![
            vec![Integer::from(1), Integer::from(0)],
            vec![Integer::from(0), Integer::from(1)],
        ];
        let collection = PointCollection::new(2, points);
        assert_eq!(collection.len(), 2);
        assert_eq!(collection.dimension(), 2);
        assert!(!collection.is_empty());
    }

    #[test]
    fn test_empty() {
        let collection = PointCollection::<Integer>::empty(3);
        assert_eq!(collection.len(), 0);
        assert_eq!(collection.dimension(), 3);
        assert!(collection.is_empty());
    }

    #[test]
    fn test_get() {
        let points = vec![
            vec![Integer::from(1), Integer::from(0)],
            vec![Integer::from(0), Integer::from(1)],
        ];
        let collection = PointCollection::new(2, points);

        let point0 = collection.get(0).unwrap();
        assert_eq!(point0[0], Integer::from(1));
        assert_eq!(point0[1], Integer::from(0));

        assert!(collection.get(2).is_none());
    }

    #[test]
    fn test_index() {
        let points = vec![
            vec![Integer::from(1), Integer::from(0)],
            vec![Integer::from(0), Integer::from(1)],
            vec![Integer::from(1), Integer::from(1)],
        ];
        let collection = PointCollection::new(2, points);

        assert_eq!(
            collection.index(&[Integer::from(0), Integer::from(1)]),
            Some(1)
        );
        assert_eq!(
            collection.index(&[Integer::from(2), Integer::from(2)]),
            None
        );
    }

    #[test]
    fn test_contains() {
        let points = vec![
            vec![Integer::from(1), Integer::from(0)],
            vec![Integer::from(0), Integer::from(1)],
        ];
        let collection = PointCollection::new(2, points);

        assert!(collection.contains(&[Integer::from(1), Integer::from(0)]));
        assert!(!collection.contains(&[Integer::from(2), Integer::from(2)]));
    }

    #[test]
    fn test_add_point() {
        let mut collection = PointCollection::empty(2);
        collection.add_point(vec![Integer::from(1), Integer::from(0)]);
        collection.add_point(vec![Integer::from(0), Integer::from(1)]);

        assert_eq!(collection.len(), 2);
        assert!(collection.contains(&[Integer::from(1), Integer::from(0)]));
    }

    #[test]
    fn test_to_row_matrix() {
        let points = vec![
            vec![Integer::from(1), Integer::from(2)],
            vec![Integer::from(3), Integer::from(4)],
        ];
        let collection = PointCollection::new(2, points);

        let matrix = collection.to_row_matrix().unwrap();
        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 2);
        assert_eq!(matrix.get(0, 0).unwrap(), &Integer::from(1));
        assert_eq!(matrix.get(0, 1).unwrap(), &Integer::from(2));
        assert_eq!(matrix.get(1, 0).unwrap(), &Integer::from(3));
        assert_eq!(matrix.get(1, 1).unwrap(), &Integer::from(4));
    }

    #[test]
    fn test_to_column_matrix() {
        let points = vec![
            vec![Integer::from(1), Integer::from(2)],
            vec![Integer::from(3), Integer::from(4)],
        ];
        let collection = PointCollection::new(2, points);

        let matrix = collection.to_column_matrix().unwrap();
        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 2);
        assert_eq!(matrix.get(0, 0).unwrap(), &Integer::from(1));
        assert_eq!(matrix.get(1, 0).unwrap(), &Integer::from(2));
        assert_eq!(matrix.get(0, 1).unwrap(), &Integer::from(3));
        assert_eq!(matrix.get(1, 1).unwrap(), &Integer::from(4));
    }

    #[test]
    fn test_extend() {
        let mut collection1 = PointCollection::new(
            2,
            vec![vec![Integer::from(1), Integer::from(0)]],
        );
        let collection2 = PointCollection::new(
            2,
            vec![vec![Integer::from(0), Integer::from(1)]],
        );

        collection1.extend(&collection2);
        assert_eq!(collection1.len(), 2);
    }

    #[test]
    fn test_concatenate() {
        let collection1 = PointCollection::new(
            2,
            vec![vec![Integer::from(1), Integer::from(0)]],
        );
        let collection2 = PointCollection::new(
            2,
            vec![vec![Integer::from(0), Integer::from(1)]],
        );

        let combined = collection1.concatenate(&collection2);
        assert_eq!(combined.len(), 2);
        assert_eq!(collection1.len(), 1); // Original unchanged
    }

    #[test]
    fn test_cartesian_product() {
        let collection1 = PointCollection::new(
            2,
            vec![
                vec![Integer::from(1), Integer::from(0)],
                vec![Integer::from(0), Integer::from(1)],
            ],
        );
        let collection2 = PointCollection::new(
            1,
            vec![vec![Integer::from(5)], vec![Integer::from(6)]],
        );

        let product = collection1.cartesian_product(&collection2);
        assert_eq!(product.dimension(), 3);
        assert_eq!(product.len(), 4); // 2 * 2 = 4

        // Check first point: (1, 0, 5)
        assert_eq!(
            product.get(0).unwrap(),
            &[Integer::from(1), Integer::from(0), Integer::from(5)]
        );
    }

    #[test]
    fn test_has_duplicates() {
        let collection_no_dups = PointCollection::new(
            2,
            vec![
                vec![Integer::from(1), Integer::from(0)],
                vec![Integer::from(0), Integer::from(1)],
            ],
        );
        assert!(!collection_no_dups.has_duplicates());

        let collection_with_dups = PointCollection::new(
            2,
            vec![
                vec![Integer::from(1), Integer::from(0)],
                vec![Integer::from(0), Integer::from(1)],
                vec![Integer::from(1), Integer::from(0)], // duplicate
            ],
        );
        assert!(collection_with_dups.has_duplicates());
    }

    #[test]
    fn test_deduplicate() {
        let mut collection = PointCollection::new(
            2,
            vec![
                vec![Integer::from(1), Integer::from(0)],
                vec![Integer::from(0), Integer::from(1)],
                vec![Integer::from(1), Integer::from(0)], // duplicate
                vec![Integer::from(0), Integer::from(1)], // duplicate
            ],
        );

        assert_eq!(collection.len(), 4);
        collection.deduplicate();
        assert_eq!(collection.len(), 2);
        assert!(!collection.has_duplicates());
    }

    #[test]
    fn test_iter() {
        let points = vec![
            vec![Integer::from(1), Integer::from(0)],
            vec![Integer::from(0), Integer::from(1)],
        ];
        let collection = PointCollection::new(2, points);

        let collected: Vec<_> = collection.iter().collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0], &[Integer::from(1), Integer::from(0)]);
    }

    #[test]
    fn test_is_point_collection() {
        let collection: PointCollection<Integer> = PointCollection::empty(2);
        assert!(is_point_collection(&collection));
    }
}
