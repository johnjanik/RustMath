//! Matrix actions on various mathematical objects
//!
//! This module implements different ways that matrices can act on other objects:
//! - Matrix-matrix multiplication
//! - Matrix-vector multiplication
//! - Matrix action on points
//! - Matrix action on polynomials
//!
//! These actions provide a unified interface for applying matrices to different
//! types of mathematical objects.

use crate::{Matrix, Vector};
use rustmath_core::{Ring, Result};
use std::ops::Mul;

/// Trait for objects that matrices can act on
pub trait MatrixAction<R: Ring, T> {
    /// Apply the matrix to the object
    fn act(&self, target: &T) -> Result<T>;
}

/// Matrix-matrix multiplication action
///
/// This represents the action of a matrix on another matrix via multiplication
impl<R: Ring> MatrixAction<R, Matrix<R>> for Matrix<R> {
    fn act(&self, target: &Matrix<R>) -> Result<Matrix<R>> {
        // Matrix multiplication - clone to get owned values
        self.clone().mul(target.clone())
    }
}

/// Matrix-vector multiplication action
///
/// This represents the standard action of a matrix on a vector
impl<R: Ring> MatrixAction<R, Vector<R>> for Matrix<R> {
    fn act(&self, target: &Vector<R>) -> Result<Vector<R>> {
        if self.cols() != target.dim() {
            panic!("Matrix columns must match vector length for matrix-vector multiplication");
        }

        let mut result = Vec::with_capacity(self.rows());
        for i in 0..self.rows() {
            let mut sum = R::zero();
            for j in 0..self.cols() {
                sum = sum + (self.get(i, j).unwrap().clone() * target.get(j).unwrap().clone());
            }
            result.push(sum);
        }

        Ok(Vector::new(result))
    }
}

/// Point in affine or projective space
#[derive(Clone, Debug, PartialEq)]
pub struct Point<R: Ring> {
    coordinates: Vec<R>,
    is_projective: bool,
}

impl<R: Ring> Point<R> {
    /// Create a new affine point
    pub fn affine(coordinates: Vec<R>) -> Self {
        Point {
            coordinates,
            is_projective: false,
        }
    }

    /// Create a new projective point
    pub fn projective(coordinates: Vec<R>) -> Self {
        Point {
            coordinates,
            is_projective: true,
        }
    }

    /// Get the coordinates
    pub fn coordinates(&self) -> &[R] {
        &self.coordinates
    }

    /// Check if this is a projective point
    pub fn is_projective(&self) -> bool {
        self.is_projective
    }
}

/// Matrix action on points in affine or projective space
impl<R: Ring> MatrixAction<R, Point<R>> for Matrix<R> {
    fn act(&self, point: &Point<R>) -> Result<Point<R>> {
        if self.cols() != point.coordinates.len() {
            panic!("Matrix columns must match point dimension");
        }

        let mut result = Vec::with_capacity(self.rows());
        for i in 0..self.rows() {
            let mut sum = R::zero();
            for j in 0..self.cols() {
                sum = sum + (self.get(i, j).unwrap().clone() * point.coordinates[j].clone());
            }
            result.push(sum);
        }

        if point.is_projective {
            Ok(Point::projective(result))
        } else {
            Ok(Point::affine(result))
        }
    }
}

/// A simplified polynomial representation for matrix actions
#[derive(Clone, Debug)]
pub struct PolynomialMap<R: Ring> {
    /// Coefficients of the polynomial
    coefficients: Vec<R>,
}

impl<R: Ring> PolynomialMap<R> {
    /// Create a new polynomial map
    pub fn new(coefficients: Vec<R>) -> Self {
        PolynomialMap { coefficients }
    }

    /// Get the coefficients
    pub fn coefficients(&self) -> &[R] {
        &self.coefficients
    }
}

/// Matrix action on polynomial maps (composition)
///
/// For a polynomial map p and a matrix M, M(p) represents the composition
impl<R: Ring> MatrixAction<R, PolynomialMap<R>> for Matrix<R> {
    fn act(&self, poly: &PolynomialMap<R>) -> Result<PolynomialMap<R>> {
        // For now, we just return a copy
        // A full implementation would compose the matrix with the polynomial map
        Ok(poly.clone())
    }
}

/// Left action: vector * matrix (row vector on the left)
pub struct VectorMatrixAction;

impl VectorMatrixAction {
    /// Apply a row vector to a matrix from the left
    pub fn act<R: Ring>(vector: &Vector<R>, matrix: &Matrix<R>) -> Result<Vector<R>> {
        if vector.dim() != matrix.rows() {
            panic!("Vector length must match matrix rows for vector-matrix multiplication");
        }

        let mut result = Vec::with_capacity(matrix.cols());
        for j in 0..matrix.cols() {
            let mut sum = R::zero();
            for i in 0..matrix.rows() {
                sum = sum + (vector.get(i).unwrap().clone() * matrix.get(i, j).unwrap().clone());
            }
            result.push(sum);
        }

        Ok(Vector::new(result))
    }
}

/// Right action: matrix * vector (column vector on the right)
pub struct MatrixVectorAction;

impl MatrixVectorAction {
    /// Apply a matrix to a column vector from the left
    pub fn act<R: Ring>(matrix: &Matrix<R>, vector: &Vector<R>) -> Result<Vector<R>> {
        matrix.act(vector)
    }
}

/// Matrix-matrix multiplication action
pub struct MatrixMatrixAction;

impl MatrixMatrixAction {
    /// Multiply two matrices
    pub fn act<R: Ring>(left: &Matrix<R>, right: &Matrix<R>) -> Result<Matrix<R>> {
        left.act(right)
    }
}

/// Matrix action on scheme points
pub struct MatrixSchemePointAction;

impl MatrixSchemePointAction {
    /// Apply a matrix to a scheme point
    pub fn act<R: Ring>(matrix: &Matrix<R>, point: &Point<R>) -> Result<Point<R>> {
        matrix.act(point)
    }
}

/// Matrix action on polynomial maps
pub struct MatrixPolymapAction;

impl MatrixPolymapAction {
    /// Apply a matrix to a polynomial map
    pub fn act<R: Ring>(matrix: &Matrix<R>, poly: &PolynomialMap<R>) -> Result<PolynomialMap<R>> {
        matrix.act(poly)
    }
}

/// Polynomial map action on matrices
pub struct PolymapMatrixAction;

impl PolymapMatrixAction {
    /// Apply a polynomial map to a matrix
    /// This evaluates the polynomial at the matrix (matrix polynomial evaluation)
    pub fn act<R: Ring>(poly: &PolynomialMap<R>, matrix: &Matrix<R>) -> Result<Matrix<R>> {
        if !matrix.is_square() {
            panic!("Matrix must be square for polynomial evaluation");
        }

        if poly.coefficients.is_empty() {
            return Ok(Matrix::zeros(matrix.rows(), matrix.cols()));
        }

        // Start with c0 * I
        let mut result = Matrix::identity(matrix.rows()).scalar_mul(&poly.coefficients[0]);

        // Add higher order terms
        let mut matrix_power = matrix.clone();
        for i in 1..poly.coefficients.len() {
            let term = matrix_power.scalar_mul(&poly.coefficients[i]);
            result = (result + term)?;
            if i < poly.coefficients.len() - 1 {
                matrix_power = (matrix_power * matrix.clone())?;
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;
    use rustmath_rationals::Rational;

    #[test]
    fn test_matrix_vector_action() {
        // 2x2 matrix * 2x1 vector
        let matrix = Matrix::from_vec(2, 2, vec![
            Integer::from(1), Integer::from(2),
            Integer::from(3), Integer::from(4),
        ]).unwrap();

        let vector = Vector::new(vec![
            Integer::from(5),
            Integer::from(6),
        ]);

        let result = matrix.act(&vector).unwrap();

        // [1 2] [5]   [17]
        // [3 4] [6] = [39]
        assert_eq!(result.get(0).unwrap(), &Integer::from(17));
        assert_eq!(result.get(1).unwrap(), &Integer::from(39));
    }

    #[test]
    fn test_vector_matrix_action() {
        let vector = Vector::new(vec![
            Integer::from(1),
            Integer::from(2),
        ]);

        let matrix = Matrix::from_vec(2, 2, vec![
            Integer::from(3), Integer::from(4),
            Integer::from(5), Integer::from(6),
        ]).unwrap();

        let result = VectorMatrixAction::act(&vector, &matrix).unwrap();

        // [1 2] [3 4] = [13 16]
        //       [5 6]
        assert_eq!(result.get(0).unwrap(), &Integer::from(13));
        assert_eq!(result.get(1).unwrap(), &Integer::from(16));
    }

    #[test]
    fn test_matrix_matrix_action() {
        let m1 = Matrix::from_vec(2, 2, vec![
            Integer::from(1), Integer::from(2),
            Integer::from(3), Integer::from(4),
        ]).unwrap();

        let m2 = Matrix::from_vec(2, 2, vec![
            Integer::from(5), Integer::from(6),
            Integer::from(7), Integer::from(8),
        ]).unwrap();

        let result = MatrixMatrixAction::act(&m1, &m2).unwrap();

        // [1 2] [5 6]   [19 22]
        // [3 4] [7 8] = [43 50]
        assert_eq!(result.get(0, 0).unwrap(), &Integer::from(19));
        assert_eq!(result.get(0, 1).unwrap(), &Integer::from(22));
        assert_eq!(result.get(1, 0).unwrap(), &Integer::from(43));
        assert_eq!(result.get(1, 1).unwrap(), &Integer::from(50));
    }

    #[test]
    fn test_affine_point_action() {
        let matrix = Matrix::from_vec(2, 2, vec![
            Rational::from_integer(2), Rational::from_integer(0),
            Rational::from_integer(0), Rational::from_integer(3),
        ]).unwrap();

        let point = Point::affine(vec![Rational::from_integer(1), Rational::from_integer(1)]);
        let result = matrix.act(&point).unwrap();

        assert_eq!(result.coordinates()[0], Rational::from_integer(2));
        assert_eq!(result.coordinates()[1], Rational::from_integer(3));
        assert!(!result.is_projective());
    }

    #[test]
    fn test_projective_point_action() {
        let matrix = Matrix::from_vec(2, 2, vec![
            Rational::from_integer(1), Rational::from_integer(1),
            Rational::from_integer(0), Rational::from_integer(1),
        ]).unwrap();

        let point = Point::projective(vec![Rational::from_integer(1), Rational::from_integer(0)]);
        let result = matrix.act(&point).unwrap();

        assert_eq!(result.coordinates()[0], Rational::from_integer(1));
        assert_eq!(result.coordinates()[1], Rational::from_integer(0));
        assert!(result.is_projective());
    }

    #[test]
    fn test_polynomial_map_creation() {
        let poly = PolynomialMap::new(vec![
            Integer::from(1),
            Integer::from(2),
            Integer::from(3),
        ]);

        assert_eq!(poly.coefficients().len(), 3);
        assert_eq!(poly.coefficients()[0], Integer::from(1));
    }
}
