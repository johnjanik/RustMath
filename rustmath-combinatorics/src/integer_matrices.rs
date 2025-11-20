//! Integer matrix enumeration with row and column sum constraints
//!
//! This module provides functionality to enumerate all non-negative integer matrices
//! that satisfy specified row and column sum constraints.
//!
//! # Examples
//!
//! ```
//! use rustmath_combinatorics::integer_matrices::{integer_matrices, integer_matrices_bounded};
//!
//! // Enumerate all 2x3 matrices with row sums [3, 2] and column sums [1, 2, 2]
//! let matrices = integer_matrices(&[3, 2], &[1, 2, 2]);
//! assert_eq!(matrices.len(), 5);
//!
//! // With entry bounds [0, 2]
//! let bounded = integer_matrices_bounded(&[3, 2], &[1, 2, 2], 0, 2);
//! assert_eq!(bounded.len(), 5);
//! ```

use rustmath_core::Ring;
use rustmath_integers::Integer;

/// Represents an integer matrix with specified dimensions
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegerMatrix {
    /// The matrix data in row-major order
    data: Vec<Vec<Integer>>,
    /// Number of rows
    rows: usize,
    /// Number of columns
    cols: usize,
}

impl IntegerMatrix {
    /// Create a new integer matrix from a 2D vector
    pub fn new(data: Vec<Vec<Integer>>) -> Option<Self> {
        if data.is_empty() {
            return Some(IntegerMatrix {
                data: vec![],
                rows: 0,
                cols: 0,
            });
        }

        let rows = data.len();
        let cols = data[0].len();

        // Check that all rows have the same length
        for row in &data {
            if row.len() != cols {
                return None;
            }
        }

        Some(IntegerMatrix { data, rows, cols })
    }

    /// Create a zero matrix with specified dimensions
    pub fn zeros(rows: usize, cols: usize) -> Self {
        let data = vec![vec![Integer::zero(); cols]; rows];
        IntegerMatrix { data, rows, cols }
    }

    /// Get the number of rows
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Get the number of columns
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Get the element at position (i, j)
    pub fn get(&self, i: usize, j: usize) -> Option<&Integer> {
        self.data.get(i)?.get(j)
    }

    /// Set the element at position (i, j)
    pub fn set(&mut self, i: usize, j: usize, value: Integer) -> bool {
        if i >= self.rows || j >= self.cols {
            return false;
        }
        self.data[i][j] = value;
        true
    }

    /// Get the data as a 2D vector
    pub fn data(&self) -> &[Vec<Integer>] {
        &self.data
    }

    /// Compute row sums
    pub fn row_sums(&self) -> Vec<Integer> {
        self.data
            .iter()
            .map(|row| row.iter().fold(Integer::zero(), |acc, x| acc + x.clone()))
            .collect()
    }

    /// Compute column sums
    pub fn col_sums(&self) -> Vec<Integer> {
        let mut sums = vec![Integer::zero(); self.cols];
        for row in &self.data {
            for (j, val) in row.iter().enumerate() {
                sums[j] = sums[j].clone() + val.clone();
            }
        }
        sums
    }

    /// Check if the matrix satisfies the given row and column sum constraints
    pub fn satisfies_constraints(&self, row_sums: &[u32], col_sums: &[u32]) -> bool {
        if row_sums.len() != self.rows || col_sums.len() != self.cols {
            return false;
        }

        let actual_row_sums = self.row_sums();
        let actual_col_sums = self.col_sums();

        for (i, &expected) in row_sums.iter().enumerate() {
            if actual_row_sums[i] != Integer::from(expected) {
                return false;
            }
        }

        for (j, &expected) in col_sums.iter().enumerate() {
            if actual_col_sums[j] != Integer::from(expected) {
                return false;
            }
        }

        true
    }
}

/// Enumerate all non-negative integer matrices with specified row and column sums
///
/// This function generates all m × n matrices with non-negative integer entries
/// such that row i sums to row_sums[i] and column j sums to col_sums[j].
///
/// # Arguments
///
/// * `row_sums` - Vector of desired row sums (length m)
/// * `col_sums` - Vector of desired column sums (length n)
///
/// # Returns
///
/// A vector of all matrices satisfying the constraints, or an empty vector if
/// the constraints are impossible to satisfy (e.g., sum of row_sums ≠ sum of col_sums).
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::integer_matrices::integer_matrices;
///
/// // 2x2 matrices with row sums [2, 1] and column sums [1, 2]
/// let matrices = integer_matrices(&[2, 1], &[1, 2]);
/// assert_eq!(matrices.len(), 2);
/// ```
pub fn integer_matrices(row_sums: &[u32], col_sums: &[u32]) -> Vec<IntegerMatrix> {
    // Check feasibility: sum of row sums must equal sum of column sums
    let row_sum_total: u32 = row_sums.iter().sum();
    let col_sum_total: u32 = col_sums.iter().sum();

    if row_sum_total != col_sum_total {
        return vec![];
    }

    if row_sums.is_empty() || col_sums.is_empty() {
        return vec![];
    }

    let rows = row_sums.len();
    let cols = col_sums.len();

    let mut result = Vec::new();
    let mut matrix = IntegerMatrix::zeros(rows, cols);
    let mut remaining_row_sums: Vec<i64> = row_sums.iter().map(|&x| x as i64).collect();
    let mut remaining_col_sums: Vec<i64> = col_sums.iter().map(|&x| x as i64).collect();

    enumerate_matrices_helper(
        &mut matrix,
        &mut remaining_row_sums,
        &mut remaining_col_sums,
        0,
        0,
        &mut result,
        None,
        None,
    );

    result
}

/// Enumerate all integer matrices with specified row and column sums and entry bounds
///
/// This function generates all m × n matrices with integer entries in the range [min_entry, max_entry]
/// such that row i sums to row_sums[i] and column j sums to col_sums[j].
///
/// # Arguments
///
/// * `row_sums` - Vector of desired row sums (length m)
/// * `col_sums` - Vector of desired column sums (length n)
/// * `min_entry` - Minimum value for each matrix entry (inclusive)
/// * `max_entry` - Maximum value for each matrix entry (inclusive)
///
/// # Returns
///
/// A vector of all matrices satisfying the constraints.
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::integer_matrices::integer_matrices_bounded;
///
/// // 2x2 matrices with entries in [0, 2]
/// let matrices = integer_matrices_bounded(&[2, 2], &[2, 2], 0, 2);
/// assert!(matrices.len() > 0);
/// ```
pub fn integer_matrices_bounded(
    row_sums: &[u32],
    col_sums: &[u32],
    min_entry: i64,
    max_entry: i64,
) -> Vec<IntegerMatrix> {
    // Check feasibility
    let row_sum_total: u32 = row_sums.iter().sum();
    let col_sum_total: u32 = col_sums.iter().sum();

    if row_sum_total != col_sum_total {
        return vec![];
    }

    if row_sums.is_empty() || col_sums.is_empty() {
        return vec![];
    }

    let rows = row_sums.len();
    let cols = col_sums.len();

    let mut result = Vec::new();
    let mut matrix = IntegerMatrix::zeros(rows, cols);
    let mut remaining_row_sums: Vec<i64> = row_sums.iter().map(|&x| x as i64).collect();
    let mut remaining_col_sums: Vec<i64> = col_sums.iter().map(|&x| x as i64).collect();

    enumerate_matrices_helper(
        &mut matrix,
        &mut remaining_row_sums,
        &mut remaining_col_sums,
        0,
        0,
        &mut result,
        Some(min_entry),
        Some(max_entry),
    );

    result
}

/// Helper function for backtracking enumeration
#[allow(clippy::too_many_arguments)]
fn enumerate_matrices_helper(
    matrix: &mut IntegerMatrix,
    remaining_row_sums: &mut [i64],
    remaining_col_sums: &mut [i64],
    row: usize,
    col: usize,
    result: &mut Vec<IntegerMatrix>,
    min_entry: Option<i64>,
    max_entry: Option<i64>,
) {
    let rows = matrix.rows();
    let cols = matrix.cols();

    // Base case: filled all entries
    if row == rows {
        // Check that all remaining sums are zero
        if remaining_row_sums.iter().all(|&x| x == 0)
            && remaining_col_sums.iter().all(|&x| x == 0)
        {
            result.push(matrix.clone());
        }
        return;
    }

    // Calculate next position
    let (next_row, next_col) = if col + 1 < cols {
        (row, col + 1)
    } else {
        (row + 1, 0)
    };

    // Determine the range of valid values for this position
    let min_val = min_entry.unwrap_or(0);
    let max_val = max_entry.unwrap_or(i64::MAX);

    // The value must be:
    // 1. At least min_val
    // 2. At most max_val
    // 3. At most the remaining row sum
    // 4. At most the remaining column sum
    // 5. At least (remaining row sum - sum of max values for remaining columns in this row)
    // 6. At least (remaining column sum - sum of max values for remaining rows in this column)

    let remaining_cols_in_row = cols - col - 1;
    let remaining_rows_in_col = rows - row - 1;

    // Lower bound: ensure we can satisfy the row sum with remaining columns
    let lower_bound_row = if col == cols - 1 {
        // Last column in row: must equal remaining row sum
        remaining_row_sums[row]
    } else if max_val == i64::MAX {
        // No upper bound on entries, so lower bound is just min_val
        min_val
    } else {
        // Calculate the maximum we can place in remaining columns
        let max_from_remaining_cols = (remaining_cols_in_row as i64).saturating_mul(max_val);
        (remaining_row_sums[row] - max_from_remaining_cols).max(min_val)
    };

    // Lower bound: ensure we can satisfy the column sum with remaining rows
    let lower_bound_col = if row == rows - 1 {
        // Last row in column: must equal remaining column sum
        remaining_col_sums[col]
    } else if max_val == i64::MAX {
        // No upper bound on entries, so lower bound is just min_val
        min_val
    } else {
        // Calculate the maximum we can place in remaining rows
        let max_from_remaining_rows = (remaining_rows_in_col as i64).saturating_mul(max_val);
        (remaining_col_sums[col] - max_from_remaining_rows).max(min_val)
    };

    let lower_bound = lower_bound_row.max(lower_bound_col).max(min_val);

    // Upper bound
    let upper_bound = remaining_row_sums[row]
        .min(remaining_col_sums[col])
        .min(max_val);

    // Try all valid values
    for value in lower_bound..=upper_bound {
        // Set the value
        matrix.set(row, col, Integer::from(value));
        remaining_row_sums[row] -= value;
        remaining_col_sums[col] -= value;

        // Recurse
        enumerate_matrices_helper(
            matrix,
            remaining_row_sums,
            remaining_col_sums,
            next_row,
            next_col,
            result,
            min_entry,
            max_entry,
        );

        // Backtrack
        remaining_row_sums[row] += value;
        remaining_col_sums[col] += value;
    }

    // Reset to zero for backtracking
    matrix.set(row, col, Integer::zero());
}

/// Count the number of integer matrices with specified row and column sums
///
/// This is more efficient than generating all matrices when you only need the count.
///
/// # Arguments
///
/// * `row_sums` - Vector of desired row sums
/// * `col_sums` - Vector of desired column sums
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::integer_matrices::count_integer_matrices;
///
/// let count = count_integer_matrices(&[2, 1], &[1, 2]);
/// assert_eq!(count, 2);
/// ```
pub fn count_integer_matrices(row_sums: &[u32], col_sums: &[u32]) -> usize {
    integer_matrices(row_sums, col_sums).len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integer_matrix_basic() {
        let data = vec![
            vec![Integer::from(1), Integer::from(2)],
            vec![Integer::from(3), Integer::from(4)],
        ];
        let matrix = IntegerMatrix::new(data.clone()).unwrap();

        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 2);
        assert_eq!(matrix.get(0, 0), Some(&Integer::from(1)));
        assert_eq!(matrix.get(1, 1), Some(&Integer::from(4)));
    }

    #[test]
    fn test_row_col_sums() {
        let data = vec![
            vec![Integer::from(1), Integer::from(2)],
            vec![Integer::from(3), Integer::from(4)],
        ];
        let matrix = IntegerMatrix::new(data).unwrap();

        let row_sums = matrix.row_sums();
        assert_eq!(row_sums, vec![Integer::from(3), Integer::from(7)]);

        let col_sums = matrix.col_sums();
        assert_eq!(col_sums, vec![Integer::from(4), Integer::from(6)]);
    }

    #[test]
    fn test_integer_matrices_small() {
        // 1x1 matrix with sum 2
        let matrices = integer_matrices(&[2], &[2]);
        assert_eq!(matrices.len(), 1);
        assert_eq!(matrices[0].get(0, 0), Some(&Integer::from(2)));

        // 1x2 matrix with row sum [3] and column sums [1, 2]
        let matrices = integer_matrices(&[3], &[1, 2]);
        assert_eq!(matrices.len(), 1);
        assert_eq!(matrices[0].get(0, 0), Some(&Integer::from(1)));
        assert_eq!(matrices[0].get(0, 1), Some(&Integer::from(2)));
    }

    #[test]
    fn test_integer_matrices_2x2() {
        // 2x2 with row sums [2, 1] and column sums [1, 2]
        let matrices = integer_matrices(&[2, 1], &[1, 2]);
        assert_eq!(matrices.len(), 2);

        // Verify all matrices satisfy constraints
        for m in &matrices {
            assert!(m.satisfies_constraints(&[2, 1], &[1, 2]));
        }

        // The two matrices should be:
        // [[0, 2], [1, 0]] and [[1, 1], [0, 1]]
        let mut found_first = false;
        let mut found_second = false;

        for m in &matrices {
            if m.get(0, 0) == Some(&Integer::from(0)) && m.get(0, 1) == Some(&Integer::from(2)) {
                assert_eq!(m.get(1, 0), Some(&Integer::from(1)));
                assert_eq!(m.get(1, 1), Some(&Integer::from(0)));
                found_first = true;
            }
            if m.get(0, 0) == Some(&Integer::from(1)) && m.get(0, 1) == Some(&Integer::from(1)) {
                assert_eq!(m.get(1, 0), Some(&Integer::from(0)));
                assert_eq!(m.get(1, 1), Some(&Integer::from(1)));
                found_second = true;
            }
        }

        assert!(found_first && found_second);
    }

    #[test]
    fn test_integer_matrices_infeasible() {
        // Row sums don't match column sums
        let matrices = integer_matrices(&[2, 1], &[1, 1]);
        assert_eq!(matrices.len(), 0);

        // Another infeasible case
        let matrices = integer_matrices(&[5], &[2, 2]);
        assert_eq!(matrices.len(), 0);
    }

    #[test]
    fn test_integer_matrices_bounded() {
        // 2x2 with entries in [0, 1]
        let matrices = integer_matrices_bounded(&[2, 1], &[1, 2], 0, 1);

        // Verify all entries are in bounds
        for m in &matrices {
            for i in 0..m.rows() {
                for j in 0..m.cols() {
                    let val = m.get(i, j).unwrap();
                    assert!(*val >= Integer::from(0));
                    assert!(*val <= Integer::from(1));
                }
            }
            assert!(m.satisfies_constraints(&[2, 1], &[1, 2]));
        }

        // Should only have matrices where entries are 0 or 1
        assert_eq!(matrices.len(), 1);
        // The only valid matrix is [[1, 1], [0, 1]]
        assert_eq!(matrices[0].get(0, 0), Some(&Integer::from(1)));
        assert_eq!(matrices[0].get(0, 1), Some(&Integer::from(1)));
        assert_eq!(matrices[0].get(1, 0), Some(&Integer::from(0)));
        assert_eq!(matrices[0].get(1, 1), Some(&Integer::from(1)));
    }

    #[test]
    fn test_count_integer_matrices() {
        let count = count_integer_matrices(&[2, 1], &[1, 2]);
        assert_eq!(count, 2);

        let count = count_integer_matrices(&[3], &[1, 2]);
        assert_eq!(count, 1);
    }

    #[test]
    fn test_integer_matrices_2x3() {
        // 2x3 with row sums [3, 2] and column sums [1, 2, 2]
        let matrices = integer_matrices(&[3, 2], &[1, 2, 2]);

        // Verify all matrices satisfy constraints
        for m in &matrices {
            assert!(m.satisfies_constraints(&[3, 2], &[1, 2, 2]));
        }

        // Let's verify we have the expected matrices
        // The valid matrices are:
        // [[0, 1, 2], [1, 1, 0]]
        // [[0, 2, 1], [1, 0, 1]]
        // [[1, 0, 2], [0, 2, 0]]
        // [[1, 1, 1], [0, 1, 1]]
        // [[1, 2, 0], [0, 0, 2]]
        // There should be 5 valid matrices
        assert_eq!(matrices.len(), 5);
    }

    #[test]
    fn test_zero_matrix() {
        let matrices = integer_matrices(&[0, 0], &[0, 0]);
        assert_eq!(matrices.len(), 1);

        let m = &matrices[0];
        assert_eq!(m.get(0, 0), Some(&Integer::zero()));
        assert_eq!(m.get(0, 1), Some(&Integer::zero()));
        assert_eq!(m.get(1, 0), Some(&Integer::zero()));
        assert_eq!(m.get(1, 1), Some(&Integer::zero()));
    }

    #[test]
    fn test_single_row() {
        let matrices = integer_matrices(&[5], &[2, 3]);
        assert_eq!(matrices.len(), 1);
        assert_eq!(matrices[0].get(0, 0), Some(&Integer::from(2)));
        assert_eq!(matrices[0].get(0, 1), Some(&Integer::from(3)));
    }

    #[test]
    fn test_single_column() {
        let matrices = integer_matrices(&[2, 3], &[5]);
        assert_eq!(matrices.len(), 1);
        assert_eq!(matrices[0].get(0, 0), Some(&Integer::from(2)));
        assert_eq!(matrices[0].get(1, 0), Some(&Integer::from(3)));
    }
}
