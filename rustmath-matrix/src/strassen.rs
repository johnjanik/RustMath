//! Strassen algorithm for fast matrix multiplication
//!
//! This module implements Strassen's algorithm, a divide-and-conquer approach
//! to matrix multiplication that achieves O(n^2.807) time complexity instead
//! of the standard O(n^3).
//!
//! The algorithm is most efficient for large matrices and switches to standard
//! multiplication for small matrices where the overhead isn't worthwhile.

use crate::Matrix;
use rustmath_core::{MathError, Result, Ring};

/// Threshold for switching from Strassen to naive multiplication
/// Below this size, naive multiplication is faster due to lower overhead
const STRASSEN_THRESHOLD: usize = 64;

/// Multiply two matrices using Strassen's algorithm
///
/// Strassen's algorithm uses a divide-and-conquer approach that reduces
/// the number of recursive multiplications from 8 to 7, achieving
/// O(n^2.807) complexity instead of O(n^3).
///
/// # Arguments
/// * `a` - First matrix (m × n)
/// * `b` - Second matrix (n × p)
///
/// # Returns
/// The product matrix (m × p)
///
/// # Errors
/// Returns an error if matrix dimensions are incompatible
///
/// # Examples
/// ```
/// use rustmath_matrix::{Matrix, strassen::strassen_multiply};
///
/// let a = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
/// let b = Matrix::from_vec(2, 2, vec![5, 6, 7, 8]).unwrap();
/// let c = strassen_multiply(&a, &b).unwrap();
/// ```
///
/// # Algorithm
/// For matrices A and B, partition them into 2×2 block matrices:
/// ```text
/// A = [A11 A12]    B = [B11 B12]
///     [A21 A22]        [B21 B22]
/// ```
///
/// Compute 7 products (instead of 8):
/// - M1 = (A11 + A22)(B11 + B22)
/// - M2 = (A21 + A22)B11
/// - M3 = A11(B12 - B22)
/// - M4 = A22(B21 - B11)
/// - M5 = (A11 + A12)B22
/// - M6 = (A21 - A11)(B11 + B12)
/// - M7 = (A12 - A22)(B21 + B22)
///
/// Then construct the result:
/// ```text
/// C = [C11 C12]
///     [C21 C22]
/// ```
/// where:
/// - C11 = M1 + M4 - M5 + M7
/// - C12 = M3 + M5
/// - C21 = M2 + M4
/// - C22 = M1 - M2 + M3 + M6
pub fn strassen_multiply<R: Ring>(a: &Matrix<R>, b: &Matrix<R>) -> Result<Matrix<R>> {
    // Verify dimensions are compatible
    if a.cols() != b.rows() {
        return Err(MathError::InvalidArgument(format!(
            "Cannot multiply {}×{} matrix with {}×{} matrix",
            a.rows(),
            a.cols(),
            b.rows(),
            b.cols()
        )));
    }

    // For non-square matrices or small matrices, use naive multiplication
    if a.rows() != a.cols()
        || b.rows() != b.cols()
        || a.rows() != b.rows()
        || a.rows() < STRASSEN_THRESHOLD
    {
        return naive_multiply(a, b);
    }

    let n = a.rows();

    // For odd dimensions or small matrices, use naive multiplication
    if n % 2 != 0 || n < STRASSEN_THRESHOLD {
        return naive_multiply(a, b);
    }

    // Partition matrices into 4 submatrices each
    let mid = n / 2;

    let a11 = extract_submatrix(a, 0, 0, mid, mid)?;
    let a12 = extract_submatrix(a, 0, mid, mid, mid)?;
    let a21 = extract_submatrix(a, mid, 0, mid, mid)?;
    let a22 = extract_submatrix(a, mid, mid, mid, mid)?;

    let b11 = extract_submatrix(b, 0, 0, mid, mid)?;
    let b12 = extract_submatrix(b, 0, mid, mid, mid)?;
    let b21 = extract_submatrix(b, mid, 0, mid, mid)?;
    let b22 = extract_submatrix(b, mid, mid, mid, mid)?;

    // Compute the 7 Strassen products
    let m1 = strassen_multiply(&matrix_add(&a11, &a22)?, &matrix_add(&b11, &b22)?)?;
    let m2 = strassen_multiply(&matrix_add(&a21, &a22)?, &b11)?;
    let m3 = strassen_multiply(&a11, &matrix_sub(&b12, &b22)?)?;
    let m4 = strassen_multiply(&a22, &matrix_sub(&b21, &b11)?)?;
    let m5 = strassen_multiply(&matrix_add(&a11, &a12)?, &b22)?;
    let m6 = strassen_multiply(&matrix_sub(&a21, &a11)?, &matrix_add(&b11, &b12)?)?;
    let m7 = strassen_multiply(&matrix_sub(&a12, &a22)?, &matrix_add(&b21, &b22)?)?;

    // Combine the results
    let c11 = matrix_add(&matrix_sub(&matrix_add(&m1, &m4)?, &m5)?, &m7)?;
    let c12 = matrix_add(&m3, &m5)?;
    let c21 = matrix_add(&m2, &m4)?;
    let c22 = matrix_add(&matrix_sub(&matrix_add(&m1, &m3)?, &m2)?, &m6)?;

    // Combine the 4 submatrices into the result
    combine_submatrices(&c11, &c12, &c21, &c22)
}

/// Naive O(n³) matrix multiplication
fn naive_multiply<R: Ring>(a: &Matrix<R>, b: &Matrix<R>) -> Result<Matrix<R>> {
    let m = a.rows();
    let n = a.cols();
    let p = b.cols();

    let mut data = Vec::with_capacity(m * p);

    for i in 0..m {
        for j in 0..p {
            let mut sum = R::zero();
            for k in 0..n {
                sum = sum + a.get(i, k)?.clone() * b.get(k, j)?.clone();
            }
            data.push(sum);
        }
    }

    Matrix::from_vec(m, p, data)
}

/// Extract a submatrix
fn extract_submatrix<R: Ring>(
    m: &Matrix<R>,
    row_start: usize,
    col_start: usize,
    rows: usize,
    cols: usize,
) -> Result<Matrix<R>> {
    let mut data = Vec::with_capacity(rows * cols);

    for i in 0..rows {
        for j in 0..cols {
            data.push(m.get(row_start + i, col_start + j)?.clone());
        }
    }

    Matrix::from_vec(rows, cols, data)
}

/// Combine 4 submatrices into a single matrix
fn combine_submatrices<R: Ring>(
    c11: &Matrix<R>,
    c12: &Matrix<R>,
    c21: &Matrix<R>,
    c22: &Matrix<R>,
) -> Result<Matrix<R>> {
    let n = c11.rows();
    let total_size = 2 * n;
    let mut data = Vec::with_capacity(total_size * total_size);

    // Top half
    for i in 0..n {
        // Top-left block
        for j in 0..n {
            data.push(c11.get(i, j)?.clone());
        }
        // Top-right block
        for j in 0..n {
            data.push(c12.get(i, j)?.clone());
        }
    }

    // Bottom half
    for i in 0..n {
        // Bottom-left block
        for j in 0..n {
            data.push(c21.get(i, j)?.clone());
        }
        // Bottom-right block
        for j in 0..n {
            data.push(c22.get(i, j)?.clone());
        }
    }

    Matrix::from_vec(total_size, total_size, data)
}

/// Add two matrices
fn matrix_add<R: Ring>(a: &Matrix<R>, b: &Matrix<R>) -> Result<Matrix<R>> {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        return Err(MathError::InvalidArgument(
            "Matrices must have same dimensions for addition".to_string(),
        ));
    }

    let mut data = Vec::with_capacity(a.rows() * a.cols());
    for i in 0..a.rows() {
        for j in 0..a.cols() {
            data.push(a.get(i, j)?.clone() + b.get(i, j)?.clone());
        }
    }

    Matrix::from_vec(a.rows(), a.cols(), data)
}

/// Subtract two matrices
fn matrix_sub<R: Ring>(a: &Matrix<R>, b: &Matrix<R>) -> Result<Matrix<R>> {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        return Err(MathError::InvalidArgument(
            "Matrices must have same dimensions for subtraction".to_string(),
        ));
    }

    let mut data = Vec::with_capacity(a.rows() * a.cols());
    for i in 0..a.rows() {
        for j in 0..a.cols() {
            data.push(a.get(i, j)?.clone() - b.get(i, j)?.clone());
        }
    }

    Matrix::from_vec(a.rows(), a.cols(), data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strassen_multiply_2x2() {
        let a = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
        let b = Matrix::from_vec(2, 2, vec![5, 6, 7, 8]).unwrap();
        let c = strassen_multiply(&a, &b).unwrap();

        // Expected: [19, 22, 43, 50]
        assert_eq!(*c.get(0, 0).unwrap(), 19);
        assert_eq!(*c.get(0, 1).unwrap(), 22);
        assert_eq!(*c.get(1, 0).unwrap(), 43);
        assert_eq!(*c.get(1, 1).unwrap(), 50);
    }

    #[test]
    fn test_strassen_multiply_identity() {
        let id = Matrix::identity(4);
        let a = Matrix::from_vec(
            4,
            4,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        )
        .unwrap();

        let result = strassen_multiply(&a, &id).unwrap();

        // Multiplying by identity should give the same matrix
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(result.get(i, j).unwrap(), a.get(i, j).unwrap());
            }
        }
    }

    #[test]
    fn test_strassen_vs_naive_small() {
        let a = Matrix::from_vec(4, 4, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
            .unwrap();
        let b = Matrix::from_vec(
            4,
            4,
            vec![
                16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
            ],
        )
        .unwrap();

        let strassen_result = strassen_multiply(&a, &b).unwrap();
        let naive_result = naive_multiply(&a, &b).unwrap();

        // Results should be identical
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(
                    strassen_result.get(i, j).unwrap(),
                    naive_result.get(i, j).unwrap()
                );
            }
        }
    }

    #[test]
    fn test_strassen_incompatible_dimensions() {
        let a = Matrix::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let b = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();

        let result = strassen_multiply(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_matrix_add_sub() {
        let a = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
        let b = Matrix::from_vec(2, 2, vec![5, 6, 7, 8]).unwrap();

        let sum = matrix_add(&a, &b).unwrap();
        assert_eq!(*sum.get(0, 0).unwrap(), 6);
        assert_eq!(*sum.get(0, 1).unwrap(), 8);
        assert_eq!(*sum.get(1, 0).unwrap(), 10);
        assert_eq!(*sum.get(1, 1).unwrap(), 12);

        let diff = matrix_sub(&b, &a).unwrap();
        assert_eq!(*diff.get(0, 0).unwrap(), 4);
        assert_eq!(*diff.get(0, 1).unwrap(), 4);
        assert_eq!(*diff.get(1, 0).unwrap(), 4);
        assert_eq!(*diff.get(1, 1).unwrap(), 4);
    }

    #[test]
    fn test_extract_and_combine() {
        let m = Matrix::from_vec(4, 4, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
            .unwrap();

        let m11 = extract_submatrix(&m, 0, 0, 2, 2).unwrap();
        let m12 = extract_submatrix(&m, 0, 2, 2, 2).unwrap();
        let m21 = extract_submatrix(&m, 2, 0, 2, 2).unwrap();
        let m22 = extract_submatrix(&m, 2, 2, 2, 2).unwrap();

        let reconstructed = combine_submatrices(&m11, &m12, &m21, &m22).unwrap();

        // Should get the original matrix back
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(reconstructed.get(i, j).unwrap(), m.get(i, j).unwrap());
            }
        }
    }
}
