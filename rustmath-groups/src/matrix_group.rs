//! Matrix groups - groups of invertible matrices
//!
//! This module implements classical matrix groups such as GL(n), SL(n), O(n), U(n)

use rustmath_core::Field;
use rustmath_matrix::Matrix;
use rustmath_rationals::Rational;
use std::marker::PhantomData;

/// A matrix group - a group of invertible matrices
pub trait MatrixGroup<F: Field> {
    /// Get the dimension n (for n×n matrices)
    fn dimension(&self) -> usize;

    /// Check if a matrix is in the group
    fn contains(&self, matrix: &Matrix<F>) -> bool;

    /// Get the identity element
    fn identity(&self) -> Matrix<F>;
}

/// The general linear group GL(n, F) - all invertible n×n matrices over F
#[derive(Clone, Debug)]
pub struct GLn<F: Field> {
    n: usize,
    _phantom: PhantomData<F>,
}

impl<F: Field> GLn<F> {
    /// Create GL(n, F)
    pub fn new(n: usize) -> Self {
        GLn {
            n,
            _phantom: PhantomData,
        }
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.n
    }
}

impl<F: Field> MatrixGroup<F> for GLn<F> {
    fn dimension(&self) -> usize {
        self.n
    }

    fn contains(&self, matrix: &Matrix<F>) -> bool {
        // Check dimensions
        if matrix.rows() != self.n || matrix.cols() != self.n {
            return false;
        }

        // Check if det != 0 (invertible)
        match matrix.determinant() {
            Ok(det) => det != F::zero(),
            Err(_) => false,
        }
    }

    fn identity(&self) -> Matrix<F> {
        Matrix::identity(self.n)
    }
}

/// The special linear group SL(n, F) - matrices with determinant 1
#[derive(Clone, Debug)]
pub struct SLn<F: Field> {
    n: usize,
    _phantom: PhantomData<F>,
}

impl<F: Field> SLn<F> {
    /// Create SL(n, F)
    pub fn new(n: usize) -> Self {
        SLn {
            n,
            _phantom: PhantomData,
        }
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.n
    }
}

impl<F: Field> MatrixGroup<F> for SLn<F> {
    fn dimension(&self) -> usize {
        self.n
    }

    fn contains(&self, matrix: &Matrix<F>) -> bool {
        // Check dimensions
        if matrix.rows() != self.n || matrix.cols() != self.n {
            return false;
        }

        // Check if det = 1
        match matrix.determinant() {
            Ok(det) => det == F::one(),
            Err(_) => false,
        }
    }

    fn identity(&self) -> Matrix<F> {
        Matrix::identity(self.n)
    }
}

/// Compute the order of GL(n, F_q) for finite field F_q
///
/// |GL(n, q)| = (q^n - 1)(q^n - q)(q^n - q²)...(q^n - q^{n-1})
pub fn gl_order_finite(n: usize, q: usize) -> usize {
    let mut order = 1usize;
    let q_n = q.pow(n as u32); // q^n

    for i in 0..n {
        let q_i = q.pow(i as u32); // q^i
        order *= q_n - q_i;
    }

    order
}

/// Compute the order of SL(n, F_q) for finite field F_q
///
/// |SL(n, q)| = |GL(n, q)| / (q - 1)
pub fn sl_order_finite(n: usize, q: usize) -> usize {
    if q <= 1 {
        return 1;
    }
    gl_order_finite(n, q) / (q - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gln_creation() {
        let gl2 = GLn::<Rational>::new(2);
        assert_eq!(gl2.dimension(), 2);
    }

    #[test]
    fn test_gln_contains_identity() {
        let gl2 = GLn::<Rational>::new(2);
        let id = Matrix::<Rational>::identity(2);

        assert!(gl2.contains(&id));
    }

    #[test]
    fn test_gln_contains_invertible() {
        let gl2 = GLn::<Rational>::new(2);

        // Create a matrix with det = 1
        let one = Rational::new(1, 1).unwrap();
        let zero = Rational::new(0, 1).unwrap();

        let mat = Matrix::from_vec(2, 2, vec![one.clone(), one.clone(), zero.clone(), one.clone()])
            .unwrap();

        assert!(gl2.contains(&mat));
    }

    #[test]
    fn test_gln_not_contains_singular() {
        let gl2 = GLn::<Rational>::new(2);

        // Create a singular matrix (det = 0)
        let one = Rational::new(1, 1).unwrap();

        let mat =
            Matrix::from_vec(2, 2, vec![one.clone(), one.clone(), one.clone(), one.clone()])
                .unwrap();

        // This matrix has det = 0, so not in GL(2)
        assert!(!gl2.contains(&mat));
    }

    #[test]
    fn test_sln_creation() {
        let sl2 = SLn::<Rational>::new(2);
        assert_eq!(sl2.dimension(), 2);
    }

    #[test]
    fn test_sln_contains_identity() {
        let sl2 = SLn::<Rational>::new(2);
        let id = Matrix::<Rational>::identity(2);

        assert!(sl2.contains(&id));
    }

    #[test]
    fn test_sln_contains_det_one() {
        let sl2 = SLn::<Rational>::new(2);

        // Create a matrix with det = 1
        let one = Rational::new(1, 1).unwrap();
        let zero = Rational::new(0, 1).unwrap();

        let mat = Matrix::from_vec(2, 2, vec![one.clone(), one.clone(), zero.clone(), one.clone()])
            .unwrap();

        assert!(sl2.contains(&mat));
    }

    #[test]
    fn test_sln_not_contains_det_not_one() {
        let sl2 = SLn::<Rational>::new(2);

        // Create a matrix with det = 2
        let one = Rational::new(1, 1).unwrap();
        let two = Rational::new(2, 1).unwrap();
        let zero = Rational::new(0, 1).unwrap();

        let mat = Matrix::from_vec(2, 2, vec![two.clone(), zero.clone(), zero.clone(), one.clone()])
            .unwrap();

        // This matrix has det = 2, so not in SL(2)
        assert!(!sl2.contains(&mat));
    }

    #[test]
    fn test_gl_order_finite() {
        // GL(2, 2) has order (4-1)(4-2) = 3*2 = 6
        assert_eq!(gl_order_finite(2, 2), 6);

        // GL(2, 3) has order (9-1)(9-3) = 8*6 = 48
        assert_eq!(gl_order_finite(2, 3), 48);
    }

    #[test]
    fn test_sl_order_finite() {
        // SL(2, 2) has order 6/1 = 6
        assert_eq!(sl_order_finite(2, 2), 6);

        // SL(2, 3) has order 48/2 = 24
        assert_eq!(sl_order_finite(2, 3), 24);
    }
}
