//! Euclidean Groups
//!
//! This module implements Euclidean groups, which are special affine groups that
//! preserve the Euclidean metric (distances). These groups consist of orthogonal
//! transformations (rotations and reflections) combined with translations.
//!
//! Mathematically, the Euclidean group is: E(V) = SO(V) ⋉ V
//! where SO(V) is the special orthogonal group.
//!
//! Elements are pairs (A, b) where A is an orthogonal matrix and b is a translation vector.

use std::fmt;
use rustmath_matrix::Matrix;
use rustmath_core::Ring;
use crate::affine_group::{AffineGroup, AffineGroupElement};

/// The Euclidean group of dimension n
///
/// A specialized affine group where the linear part must be orthogonal (unitary).
/// This preserves distances and angles in Euclidean space.
#[derive(Clone, Debug)]
pub struct EuclideanGroup<R: Ring> {
    /// The underlying affine group
    affine_group: AffineGroup<R>,
    /// The dimension
    dimension: usize,
}

impl<R: Ring> EuclideanGroup<R> {
    /// Create a new Euclidean group of the specified dimension
    ///
    /// # Arguments
    /// * `dimension` - The dimension n of the Euclidean space
    pub fn new(dimension: usize) -> Self {
        EuclideanGroup {
            affine_group: AffineGroup::new(dimension),
            dimension,
        }
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the identity element
    pub fn identity(&self) -> AffineGroupElement<R>
    where
        R: Clone + From<i32>,
    {
        self.affine_group.identity()
    }

    /// Create a translation element
    pub fn translation(&self, vector: Vec<R>) -> Result<AffineGroupElement<R>, String>
    where
        R: Clone + From<i32>,
    {
        self.affine_group.translation(vector)
    }

    /// Create a Euclidean transformation (rotation/reflection + translation)
    ///
    /// # Arguments
    /// * `matrix` - The orthogonal matrix (must satisfy A^T A = I)
    /// * `vector` - The translation vector
    ///
    /// # Errors
    /// Returns an error if the matrix is not orthogonal
    pub fn element(&self, matrix: Matrix<R>, vector: Vec<R>) -> Result<AffineGroupElement<R>, String>
    where
        R: Clone + std::ops::Add<Output = R> + std::ops::Mul<Output = R> + PartialEq + From<i32>,
    {
        // Check if matrix is orthogonal (A^T * A should equal identity)
        if !is_orthogonal(&matrix) {
            return Err("Matrix must be orthogonal (unitary) for Euclidean group".to_string());
        }

        AffineGroupElement::new(matrix, vector)
    }

    /// Create a rotation element (orthogonal matrix with no translation)
    ///
    /// # Arguments
    /// * `matrix` - The orthogonal matrix
    ///
    /// # Errors
    /// Returns an error if the matrix is not orthogonal
    pub fn rotation(&self, matrix: Matrix<R>) -> Result<AffineGroupElement<R>, String>
    where
        R: Clone + std::ops::Add<Output = R> + std::ops::Mul<Output = R> + PartialEq + From<i32>,
    {
        if !is_orthogonal(&matrix) {
            return Err("Matrix must be orthogonal for rotation".to_string());
        }

        let vector = vec![R::from(0); self.dimension];
        AffineGroupElement::new(matrix, vector)
    }
}

impl<R: Ring + Clone> fmt::Display for EuclideanGroup<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Euclidean group of dimension {}", self.dimension)
    }
}

/// Check if a matrix is orthogonal
///
/// A matrix A is orthogonal if A^T * A = I (identity matrix)
fn is_orthogonal<R>(matrix: &Matrix<R>) -> bool
where
    R: Ring + Clone + std::ops::Add<Output = R> + std::ops::Mul<Output = R> + PartialEq + From<i32>,
{
    if matrix.rows() != matrix.cols() {
        return false;
    }

    let n = matrix.rows();

    // Compute A^T * A
    for i in 0..n {
        for j in 0..n {
            let mut sum = R::from(0);
            for k in 0..n {
                sum = sum + (matrix[(k, i)].clone() * matrix[(k, j)].clone());
            }

            let expected = if i == j { R::from(1) } else { R::from(0) };
            if sum != expected {
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_euclidean_group_creation() {
        let group: EuclideanGroup<Integer> = EuclideanGroup::new(2);
        assert_eq!(group.dimension(), 2);
    }

    #[test]
    fn test_identity() {
        let group: EuclideanGroup<Integer> = EuclideanGroup::new(3);
        let id = group.identity();
        assert_eq!(id.dimension(), 3);
    }

    #[test]
    fn test_translation() {
        let group: EuclideanGroup<Integer> = EuclideanGroup::new(2);
        let trans = group.translation(vec![Integer::from(1), Integer::from(2)]);
        assert!(trans.is_ok());
    }

    #[test]
    fn test_is_orthogonal_identity() {
        let identity: Matrix<Integer> = Matrix::identity(3);
        assert!(is_orthogonal(&identity));
    }

    #[test]
    fn test_is_orthogonal_rotation_2d() {
        // 90-degree rotation in 2D: [[0, -1], [1, 0]]
        // In integer arithmetic, this should be orthogonal
        let matrix = Matrix::from_vec(2, 2, vec![
            Integer::from(0), Integer::from(-1),
            Integer::from(1), Integer::from(0),
        ]).unwrap();

        assert!(is_orthogonal(&matrix));
    }

    #[test]
    fn test_is_orthogonal_reflection() {
        // Reflection across x-axis: [[1, 0], [0, -1]]
        let matrix = Matrix::from_vec(2, 2, vec![
            Integer::from(1), Integer::from(0),
            Integer::from(0), Integer::from(-1),
        ]).unwrap();

        assert!(is_orthogonal(&matrix));
    }

    #[test]
    fn test_not_orthogonal() {
        // Scaling matrix: [[2, 0], [0, 2]]
        let matrix = Matrix::from_vec(2, 2, vec![
            Integer::from(2), Integer::from(0),
            Integer::from(0), Integer::from(2),
        ]).unwrap();

        assert!(!is_orthogonal(&matrix));
    }

    #[test]
    fn test_rotation_element() {
        let group: EuclideanGroup<Integer> = EuclideanGroup::new(2);

        // 90-degree rotation
        let matrix = Matrix::from_vec(2, 2, vec![
            Integer::from(0), Integer::from(-1),
            Integer::from(1), Integer::from(0),
        ]).unwrap();

        let rotation = group.rotation(matrix);
        assert!(rotation.is_ok());
    }

    #[test]
    fn test_non_orthogonal_rejected() {
        let group: EuclideanGroup<Integer> = EuclideanGroup::new(2);

        // Non-orthogonal matrix
        let matrix = Matrix::from_vec(2, 2, vec![
            Integer::from(2), Integer::from(0),
            Integer::from(0), Integer::from(2),
        ]).unwrap();

        let result = group.rotation(matrix);
        assert!(result.is_err());
    }

    #[test]
    fn test_euclidean_element() {
        let group: EuclideanGroup<Integer> = EuclideanGroup::new(2);

        let matrix = Matrix::identity(2);
        let vector = vec![Integer::from(1), Integer::from(2)];

        let elem = group.element(matrix, vector);
        assert!(elem.is_ok());
    }
}
