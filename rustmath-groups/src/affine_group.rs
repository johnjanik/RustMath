//! Affine Groups
//!
//! This module implements affine groups and their elements. An affine group consists
//! of all invertible affine transformations on an affine space. These transformations
//! combine linear transformations (via the general linear group GL(V)) with translations.
//!
//! Mathematically, the affine group is a semidirect product: Aff(V) = GL(V) ⋉ V
//!
//! Elements are represented as pairs (A, b) where:
//! - A is an invertible matrix (linear part)
//! - b is a vector (translation part)
//! - The transformation applies as: x ↦ Ax + b

use std::fmt;
use std::ops::Mul;
use rustmath_matrix::Matrix;
use rustmath_core::{Ring, MathError};

/// An element of an affine group
///
/// Represents an affine transformation x ↦ Ax + b where A is an invertible
/// matrix and b is a translation vector.
#[derive(Clone, Debug)]
pub struct AffineGroupElement<R: Ring> {
    /// The invertible matrix component (linear part)
    matrix: Matrix<R>,
    /// The translation vector component
    vector: Vec<R>,
    /// The dimension of the affine space
    dimension: usize,
}

impl<R: Ring> AffineGroupElement<R> {
    /// Create a new affine group element
    ///
    /// # Arguments
    /// * `matrix` - The invertible matrix A (must be square)
    /// * `vector` - The translation vector b
    ///
    /// # Returns
    /// An affine group element representing x ↦ Ax + b
    ///
    /// # Errors
    /// Returns an error if the matrix is not square or if dimensions don't match
    pub fn new(matrix: Matrix<R>, vector: Vec<R>) -> Result<Self, String> {
        if matrix.rows() != matrix.cols() {
            return Err("Matrix must be square".to_string());
        }

        if matrix.rows() != vector.len() {
            return Err(format!(
                "Vector length ({}) must match matrix dimension ({})",
                vector.len(),
                matrix.rows()
            ));
        }

        let dimension = matrix.rows();

        Ok(AffineGroupElement {
            matrix,
            vector,
            dimension,
        })
    }

    /// Get the matrix component A
    pub fn matrix(&self) -> &Matrix<R> {
        &self.matrix
    }

    /// Get the vector component b
    pub fn vector(&self) -> &[R] {
        &self.vector
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Convert to a (n+1) × (n+1) matrix representation
    ///
    /// Returns a matrix of the form:
    /// ```text
    /// [ A | b ]
    /// [ 0 | 1 ]
    /// ```
    pub fn to_matrix(&self) -> Matrix<R>
    where
        R: Clone + From<i32>,
    {
        let n = self.dimension;
        let mut result = Matrix::zero(n + 1, n + 1);

        // Copy the matrix A
        for i in 0..n {
            for j in 0..n {
                result[(i, j)] = self.matrix[(i, j)].clone();
            }
        }

        // Copy the vector b
        for i in 0..n {
            result[(i, n)] = self.vector[i].clone();
        }

        // Set the bottom-right element to 1
        result[(n, n)] = R::from(1);

        result
    }

    /// Compose two affine transformations
    ///
    /// If self represents (A₁, b₁) and other represents (A₂, b₂),
    /// the composition is (A₁A₂, A₁b₂ + b₁)
    pub fn compose(&self, other: &Self) -> Result<Self, String>
    where
        R: Clone + std::ops::Add<Output = R> + std::ops::Mul<Output = R>,
    {
        if self.dimension != other.dimension {
            return Err("Dimensions must match for composition".to_string());
        }

        // Compute A₁ * A₂
        let new_matrix = self.matrix.mul(other.matrix).map_err(|e| e.to_string())?;

        // Compute A₁ * b₂ + b₁
        let mut new_vector = Vec::new();
        for i in 0..self.dimension {
            let mut sum = self.vector[i].clone();
            for j in 0..self.dimension {
                sum = sum + (self.matrix[(i, j)].clone() * other.vector[j].clone());
            }
            new_vector.push(sum);
        }

        Self::new(new_matrix, new_vector)
    }

    /// Apply the affine transformation to a vector
    ///
    /// Computes Ax + b
    pub fn apply(&self, x: &[R]) -> Result<Vec<R>, String>
    where
        R: Clone + std::ops::Add<Output = R> + std::ops::Mul<Output = R>,
    {
        if x.len() != self.dimension {
            return Err(format!(
                "Vector length ({}) must match dimension ({})",
                x.len(),
                self.dimension
            ));
        }

        let mut result = Vec::new();
        for i in 0..self.dimension {
            let mut sum = self.vector[i].clone();
            for j in 0..self.dimension {
                sum = sum + (self.matrix[(i, j)].clone() * x[j].clone());
            }
            result.push(sum);
        }

        Ok(result)
    }

    /// Compute the inverse transformation
    ///
    /// If self represents (A, b), the inverse is (A⁻¹, -A⁻¹b)
    pub fn inverse(&self) -> Result<Self, String>
    where
        R: Clone + std::ops::Add<Output = R> + std::ops::Mul<Output = R> + std::ops::Neg<Output = R> + rustmath_core::Field + From<i32>,
    {
        // Compute A⁻¹
        let inv_matrix_opt = self.matrix.inverse().map_err(|e| e.to_string())?;
        let inv_matrix = inv_matrix_opt.ok_or_else(|| "Matrix is singular and cannot be inverted".to_string())?;

        // Compute -A⁻¹ * b
        let mut new_vector = Vec::new();
        for i in 0..self.dimension {
            let mut sum = R::from(0);
            for j in 0..self.dimension {
                sum = sum + (inv_matrix[(i, j)].clone() * self.vector[j].clone());
            }
            new_vector.push(-sum);
        }

        Self::new(inv_matrix, new_vector)
    }

    /// Create an identity affine transformation
    pub fn identity(dimension: usize) -> Self
    where
        R: Clone + From<i32>,
    {
        let matrix = Matrix::identity(dimension);
        let vector = vec![R::from(0); dimension];

        Self::new(matrix, vector).unwrap()
    }

    /// Create a pure translation (identity matrix with translation vector)
    pub fn translation(vector: Vec<R>) -> Self
    where
        R: Clone + From<i32>,
    {
        let dimension = vector.len();
        let matrix = Matrix::identity(dimension);

        Self::new(matrix, vector).unwrap()
    }

    /// Create a pure linear transformation (no translation)
    pub fn linear(matrix: Matrix<R>) -> Result<Self, String>
    where
        R: Clone + From<i32>,
    {
        if matrix.rows() != matrix.cols() {
            return Err("Matrix must be square".to_string());
        }

        let dimension = matrix.rows();
        let vector = vec![R::from(0); dimension];

        Self::new(matrix, vector)
    }
}

impl<R: Ring + Clone + PartialEq> PartialEq for AffineGroupElement<R> {
    fn eq(&self, other: &Self) -> bool {
        self.dimension == other.dimension &&
        self.matrix == other.matrix &&
        self.vector == other.vector
    }
}

impl<R: Ring + Clone + PartialEq> Eq for AffineGroupElement<R> {}

impl<R: Ring + Clone + From<i32>> Default for AffineGroupElement<R> {
    /// Create a default element (identity transformation in 1D space)
    fn default() -> Self {
        let matrix = Matrix::identity(1);
        let vector = vec![R::from(0)];
        AffineGroupElement::new(matrix, vector).unwrap()
    }
}

impl<R: Ring + Clone + std::ops::Add<Output = R> + std::ops::Mul<Output = R>> Mul for AffineGroupElement<R> {
    type Output = Result<Self, String>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.compose(&rhs)
    }
}

impl<R: Ring + Clone + std::ops::Add<Output = R> + std::ops::Mul<Output = R>> Mul for &AffineGroupElement<R> {
    type Output = Result<AffineGroupElement<R>, String>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.compose(rhs)
    }
}

impl<R: Ring + Clone + fmt::Display> fmt::Display for AffineGroupElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Affine transformation:")?;
        writeln!(f, "Matrix A:")?;
        write!(f, "{}", self.matrix)?;
        writeln!(f, "Vector b:")?;
        write!(f, "[")?;
        for (i, val) in self.vector.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", val)?;
        }
        write!(f, "]")
    }
}

/// The affine group of dimension n
///
/// Consists of all invertible affine transformations on n-dimensional affine space.
/// Elements are pairs (A, b) where A ∈ GL(n) and b ∈ ℝⁿ (or the appropriate field).
#[derive(Clone, Debug)]
pub struct AffineGroup<R: Ring> {
    /// The dimension of the affine space
    dimension: usize,
    /// Phantom data for the ring type
    _phantom: std::marker::PhantomData<R>,
}

impl<R: Ring> AffineGroup<R> {
    /// Create a new affine group of the specified dimension
    ///
    /// # Arguments
    /// * `dimension` - The dimension n of the affine space
    pub fn new(dimension: usize) -> Self {
        AffineGroup {
            dimension,
            _phantom: std::marker::PhantomData,
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
        AffineGroupElement::identity(self.dimension)
    }

    /// Create a translation element
    pub fn translation(&self, vector: Vec<R>) -> Result<AffineGroupElement<R>, String>
    where
        R: Clone + From<i32>,
    {
        if vector.len() != self.dimension {
            return Err(format!(
                "Vector length ({}) must match dimension ({})",
                vector.len(),
                self.dimension
            ));
        }

        Ok(AffineGroupElement::translation(vector))
    }

    /// Create a linear transformation element
    pub fn linear(&self, matrix: Matrix<R>) -> Result<AffineGroupElement<R>, String>
    where
        R: Clone + From<i32>,
    {
        if matrix.rows() != self.dimension || matrix.cols() != self.dimension {
            return Err(format!(
                "Matrix must be {}×{}",
                self.dimension,
                self.dimension
            ));
        }

        AffineGroupElement::linear(matrix)
    }

    /// Create a general affine transformation
    pub fn element(&self, matrix: Matrix<R>, vector: Vec<R>) -> Result<AffineGroupElement<R>, String> {
        if matrix.rows() != self.dimension || matrix.cols() != self.dimension {
            return Err(format!(
                "Matrix must be {}×{}",
                self.dimension,
                self.dimension
            ));
        }

        if vector.len() != self.dimension {
            return Err(format!(
                "Vector length ({}) must match dimension ({})",
                vector.len(),
                self.dimension
            ));
        }

        AffineGroupElement::new(matrix, vector)
    }
}

impl<R: Ring + Clone> fmt::Display for AffineGroup<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Affine group of dimension {}", self.dimension)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_affine_element_creation() {
        let matrix = Matrix::from_vec(2, 2, vec![
            Integer::from(1), Integer::from(0),
            Integer::from(0), Integer::from(1),
        ]).unwrap();
        let vector = vec![Integer::from(1), Integer::from(2)];

        let elem = AffineGroupElement::new(matrix, vector);
        assert!(elem.is_ok());

        let elem = elem.unwrap();
        assert_eq!(elem.dimension(), 2);
    }

    #[test]
    fn test_identity() {
        let id: AffineGroupElement<Integer> = AffineGroupElement::identity(3);
        assert_eq!(id.dimension(), 3);
        assert_eq!(id.vector(), &[Integer::from(0); 3]);
    }

    #[test]
    fn test_translation() {
        let vector = vec![Integer::from(1), Integer::from(2), Integer::from(3)];
        let trans = AffineGroupElement::translation(vector.clone());

        assert_eq!(trans.dimension(), 3);
        assert_eq!(trans.vector(), &vector);
    }

    #[test]
    fn test_linear() {
        let matrix = Matrix::identity(2);
        let linear = AffineGroupElement::<Integer>::linear(matrix);

        assert!(linear.is_ok());
        let linear = linear.unwrap();
        assert_eq!(linear.dimension(), 2);
        assert_eq!(linear.vector(), &[Integer::from(0); 2]);
    }

    #[test]
    fn test_apply() {
        // Create a simple transformation: x ↦ 2x + 1
        let matrix = Matrix::from_vec(2, 2, vec![
            Integer::from(2), Integer::from(0),
            Integer::from(0), Integer::from(2),
        ]).unwrap();
        let vector = vec![Integer::from(1), Integer::from(1)];

        let elem = AffineGroupElement::new(matrix, vector).unwrap();

        let x = vec![Integer::from(1), Integer::from(2)];
        let result = elem.apply(&x).unwrap();

        assert_eq!(result, vec![Integer::from(3), Integer::from(5)]);
    }

    #[test]
    fn test_compose() {
        // First transformation: x ↦ 2x + 1
        let matrix1 = Matrix::from_vec(1, 1, vec![Integer::from(2)]).unwrap();
        let vector1 = vec![Integer::from(1)];
        let elem1 = AffineGroupElement::new(matrix1, vector1).unwrap();

        // Second transformation: x ↦ 3x + 2
        let matrix2 = Matrix::from_vec(1, 1, vec![Integer::from(3)]).unwrap();
        let vector2 = vec![Integer::from(2)];
        let elem2 = AffineGroupElement::new(matrix2, vector2).unwrap();

        // Composition: x ↦ 2(3x + 2) + 1 = 6x + 5
        let comp = elem1.compose(&elem2).unwrap();

        let x = vec![Integer::from(1)];
        let result = comp.apply(&x).unwrap();

        assert_eq!(result, vec![Integer::from(11)]); // 6*1 + 5 = 11
    }

    #[test]
    fn test_affine_group() {
        let group: AffineGroup<Integer> = AffineGroup::new(2);
        assert_eq!(group.dimension(), 2);

        let id = group.identity();
        assert_eq!(id.dimension(), 2);

        let trans = group.translation(vec![Integer::from(1), Integer::from(2)]);
        assert!(trans.is_ok());
    }

    #[test]
    fn test_to_matrix() {
        let matrix = Matrix::from_vec(2, 2, vec![
            Integer::from(1), Integer::from(2),
            Integer::from(3), Integer::from(4),
        ]).unwrap();
        let vector = vec![Integer::from(5), Integer::from(6)];

        let elem = AffineGroupElement::new(matrix, vector).unwrap();
        let mat = elem.to_matrix();

        assert_eq!(mat.rows(), 3);
        assert_eq!(mat.cols(), 3);
        assert_eq!(mat[(0, 0)], Integer::from(1));
        assert_eq!(mat[(0, 1)], Integer::from(2));
        assert_eq!(mat[(0, 2)], Integer::from(5));
        assert_eq!(mat[(1, 0)], Integer::from(3));
        assert_eq!(mat[(1, 1)], Integer::from(4));
        assert_eq!(mat[(1, 2)], Integer::from(6));
        assert_eq!(mat[(2, 0)], Integer::from(0));
        assert_eq!(mat[(2, 1)], Integer::from(0));
        assert_eq!(mat[(2, 2)], Integer::from(1));
    }
}
