//! Integration between cluster algebras and quivers
//!
//! This module provides helpers to convert between exchange matrices,
//! quiver representations, and cluster algebra structures.

use crate::cluster_algebra::{ExchangeMatrix, ClusterAlgebra, ClusterAlgebraSeed};
use rustmath_core::Ring;
use rustmath_matrix::Matrix;

/// Convert an integer matrix to an exchange matrix
///
/// Validates that the matrix is suitable for use as a cluster algebra
/// exchange matrix (skew-symmetric principal part).
pub fn matrix_to_exchange<R>(matrix: Vec<Vec<i64>>, rank: usize) -> Result<ExchangeMatrix<R>, String>
where
    R: Ring + Clone + From<i64>,
{
    if matrix.is_empty() {
        return Err("Matrix cannot be empty".to_string());
    }

    let rows = matrix.len();
    let cols = matrix[0].len();

    if cols != rank {
        return Err(format!("Matrix has {} columns, expected {}", cols, rank));
    }

    // Convert to Matrix<R>
    let mut mat = Matrix::zeros(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            mat.set(i, j, R::from(matrix[i][j]));
        }
    }

    // Validate skew-symmetry of principal part
    if rows >= rank && cols >= rank {
        for i in 0..rank {
            for j in 0..rank {
                let bij = matrix[i][j];
                let bji = matrix[j][i];
                if bij + bji != 0 {
                    return Err(format!(
                        "Principal part not skew-symmetric: B[{}][{}] + B[{}][{}] = {} + {} ≠ 0",
                        i, j, j, i, bij, bji
                    ));
                }
            }
        }
    }

    Ok(ExchangeMatrix::new(mat, rank))
}

/// Convert an exchange matrix to an integer matrix
pub fn exchange_to_matrix<R>(exchange: &ExchangeMatrix<R>) -> Vec<Vec<i64>>
where
    R: Ring + Clone + Into<i64>,
{
    let rows = exchange.matrix.rows();
    let cols = exchange.matrix.cols();

    let mut result = vec![vec![0i64; cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            if let Ok(val) = exchange.matrix.get(i, j) {
                // This is a simplification; proper conversion would handle this better
                result[i][j] = 0; // Placeholder
            }
        }
    }

    result
}

/// Create a cluster algebra from an integer exchange matrix
///
/// This is a convenience function that handles the conversion from
/// i64 matrices to the appropriate Ring type.
pub fn cluster_algebra_from_matrix<R>(
    matrix: Vec<Vec<i64>>,
    rank: usize,
) -> Result<ClusterAlgebra<R>, String>
where
    R: Ring + Clone + From<i64> + Ord + std::ops::Neg<Output = R>,
{
    let exchange = matrix_to_exchange(matrix, rank)?;
    Ok(ClusterAlgebra::new(exchange))
}

/// Validate that a matrix is a valid exchange matrix
///
/// Checks:
/// - Principal part is skew-symmetric
/// - Dimensions are correct
pub fn validate_exchange_matrix(matrix: &[Vec<i64>], rank: usize) -> Result<(), String> {
    if matrix.is_empty() {
        return Err("Matrix cannot be empty".to_string());
    }

    for (i, row) in matrix.iter().enumerate() {
        if row.len() != rank {
            return Err(format!(
                "Row {} has length {}, expected {}",
                i,
                row.len(),
                rank
            ));
        }
    }

    // Check skew-symmetry of principal part
    if matrix.len() >= rank {
        for i in 0..rank {
            for j in 0..rank {
                if matrix[i][j] + matrix[j][i] != 0 {
                    return Err(format!(
                        "Principal part not skew-symmetric at ({}, {})",
                        i, j
                    ));
                }
            }
        }
    }

    Ok(())
}

/// Extract the principal part of an exchange matrix
pub fn principal_part(matrix: &[Vec<i64>], rank: usize) -> Vec<Vec<i64>> {
    let mut result = vec![vec![0i64; rank]; rank];
    for i in 0..rank {
        for j in 0..rank {
            if i < matrix.len() && j < matrix[i].len() {
                result[i][j] = matrix[i][j];
            }
        }
    }
    result
}

/// Extract the coefficient part of an exchange matrix
pub fn coefficient_part(matrix: &[Vec<i64>], rank: usize) -> Vec<Vec<i64>> {
    if matrix.len() <= rank {
        return vec![];
    }

    let num_coeff = matrix.len() - rank;
    let mut result = vec![vec![0i64; rank]; num_coeff];

    for i in 0..num_coeff {
        for j in 0..rank {
            if rank + i < matrix.len() && j < matrix[rank + i].len() {
                result[i][j] = matrix[rank + i][j];
            }
        }
    }

    result
}

/// Compute the transpose of a matrix
pub fn transpose(matrix: &[Vec<i64>]) -> Vec<Vec<i64>> {
    if matrix.is_empty() {
        return vec![];
    }

    let rows = matrix.len();
    let cols = matrix[0].len();

    let mut result = vec![vec![0i64; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            result[j][i] = matrix[i][j];
        }
    }

    result
}

/// Matrix multiplication for i64 matrices
pub fn matrix_multiply(a: &[Vec<i64>], b: &[Vec<i64>]) -> Vec<Vec<i64>> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }

    let m = a.len();
    let n = a[0].len();
    let p = b[0].len();

    let mut result = vec![vec![0i64; p]; m];
    for i in 0..m {
        for j in 0..p {
            let mut sum = 0i64;
            for k in 0..n {
                sum += a[i][k] * b[k][j];
            }
            result[i][j] = sum;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_validate_exchange_matrix() {
        // Valid A_2 matrix
        let matrix = vec![vec![0, 1], vec![-1, 0]];
        assert!(validate_exchange_matrix(&matrix, 2).is_ok());

        // Invalid: not skew-symmetric
        let bad_matrix = vec![vec![0, 1], vec![1, 0]];
        assert!(validate_exchange_matrix(&bad_matrix, 2).is_err());
    }

    #[test]
    fn test_principal_part() {
        let matrix = vec![
            vec![0, 1, -1],
            vec![-1, 0, 1],
            vec![1, -1, 0],
            vec![1, 0, 0], // coefficient row
        ];

        let principal = principal_part(&matrix, 3);
        assert_eq!(principal.len(), 3);
        assert_eq!(principal[0].len(), 3);
        assert_eq!(principal[0][1], 1);
    }

    #[test]
    fn test_coefficient_part() {
        let matrix = vec![
            vec![0, 1],
            vec![-1, 0],
            vec![1, 0], // coefficient row
            vec![0, 1], // coefficient row
        ];

        let coeff = coefficient_part(&matrix, 2);
        assert_eq!(coeff.len(), 2);
        assert_eq!(coeff[0], vec![1, 0]);
        assert_eq!(coeff[1], vec![0, 1]);
    }

    #[test]
    fn test_transpose() {
        let matrix = vec![vec![1, 2, 3], vec![4, 5, 6]];

        let t = transpose(&matrix);
        assert_eq!(t.len(), 3);
        assert_eq!(t[0], vec![1, 4]);
        assert_eq!(t[1], vec![2, 5]);
        assert_eq!(t[2], vec![3, 6]);
    }

    #[test]
    fn test_matrix_multiply() {
        let a = vec![vec![1, 2], vec![3, 4]];
        let b = vec![vec![5, 6], vec![7, 8]];

        let c = matrix_multiply(&a, &b);
        assert_eq!(c[0][0], 19); // 1*5 + 2*7
        assert_eq!(c[0][1], 22); // 1*6 + 2*8
        assert_eq!(c[1][0], 43); // 3*5 + 4*7
        assert_eq!(c[1][1], 50); // 3*6 + 4*8
    }

    #[test]
    fn test_cluster_algebra_from_matrix() {
        let matrix = vec![vec![0, 1], vec![-1, 0]];

        let algebra: Result<ClusterAlgebra<Integer>, _> = cluster_algebra_from_matrix(matrix, 2);
        assert!(algebra.is_ok());

        let alg = algebra.unwrap();
        assert_eq!(alg.rank(), 2);
    }
}
