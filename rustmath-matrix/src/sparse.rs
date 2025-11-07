//! Sparse matrix representation
//!
//! This module provides sparse matrix data structures for efficient storage
//! and operations on matrices with many zero elements.

use rustmath_core::{Field, MathError, Result};

/// Sparse matrix in Compressed Sparse Row (CSR) format
///
/// Stores only non-zero elements using three arrays:
/// - values: non-zero values
/// - col_indices: column indices of non-zero values
/// - row_ptrs: pointers to where each row starts in values array
///
/// This format is efficient for:
/// - Matrix-vector multiplication
/// - Row slicing
/// - Memory usage when matrix is sparse (< 10% non-zero)
#[derive(Debug, Clone)]
pub struct SparseMatrix<F: Field> {
    /// Number of rows
    rows: usize,
    /// Number of columns
    cols: usize,
    /// Non-zero values (row-major order)
    values: Vec<F>,
    /// Column index for each non-zero value
    col_indices: Vec<usize>,
    /// Row pointers: row_ptrs[i] is the index in values where row i starts
    /// row_ptrs has length rows + 1, with row_ptrs[rows] = values.len()
    row_ptrs: Vec<usize>,
}

impl<F: Field> SparseMatrix<F> {
    /// Create a new sparse matrix from CSR format data
    pub fn from_csr(
        rows: usize,
        cols: usize,
        values: Vec<F>,
        col_indices: Vec<usize>,
        row_ptrs: Vec<usize>,
    ) -> Result<Self> {
        if values.len() != col_indices.len() {
            return Err(MathError::InvalidArgument(
                "values and col_indices must have same length".to_string(),
            ));
        }

        if row_ptrs.len() != rows + 1 {
            return Err(MathError::InvalidArgument(
                "row_ptrs must have length rows + 1".to_string(),
            ));
        }

        if row_ptrs[rows] != values.len() {
            return Err(MathError::InvalidArgument(
                "row_ptrs[rows] must equal values.len()".to_string(),
            ));
        }

        // Validate column indices
        for &col in &col_indices {
            if col >= cols {
                return Err(MathError::InvalidArgument(format!(
                    "Column index {} out of bounds (cols = {})",
                    col, cols
                )));
            }
        }

        Ok(SparseMatrix {
            rows,
            cols,
            values,
            col_indices,
            row_ptrs,
        })
    }

    /// Create a sparse matrix from dense format
    ///
    /// Automatically detects non-zero elements and builds CSR representation.
    pub fn from_dense(rows: usize, cols: usize, data: &[F]) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(MathError::InvalidArgument(
                "Data length must equal rows * cols".to_string(),
            ));
        }

        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = vec![0];

        for i in 0..rows {
            for j in 0..cols {
                let val = &data[i * cols + j];
                if !val.is_zero() {
                    values.push(val.clone());
                    col_indices.push(j);
                }
            }
            row_ptrs.push(values.len());
        }

        Ok(SparseMatrix {
            rows,
            cols,
            values,
            col_indices,
            row_ptrs,
        })
    }

    /// Create a sparse zero matrix
    pub fn zero(rows: usize, cols: usize) -> Self {
        SparseMatrix {
            rows,
            cols,
            values: Vec::new(),
            col_indices: Vec::new(),
            row_ptrs: vec![0; rows + 1],
        }
    }

    /// Create a sparse identity matrix
    pub fn identity(n: usize) -> Self {
        let mut values = Vec::with_capacity(n);
        let mut col_indices = Vec::with_capacity(n);
        let mut row_ptrs = Vec::with_capacity(n + 1);

        for i in 0..n {
            values.push(F::one());
            col_indices.push(i);
            row_ptrs.push(i);
        }
        row_ptrs.push(n);

        SparseMatrix {
            rows: n,
            cols: n,
            values,
            col_indices,
            row_ptrs,
        }
    }

    /// Get the number of rows
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Get the number of columns
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Get the number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Get the sparsity (fraction of zero elements)
    pub fn sparsity(&self) -> f64 {
        let total = self.rows * self.cols;
        if total == 0 {
            return 0.0;
        }
        1.0 - (self.nnz() as f64 / total as f64)
    }

    /// Get element at position (i, j)
    pub fn get(&self, i: usize, j: usize) -> Result<F> {
        if i >= self.rows || j >= self.cols {
            return Err(MathError::InvalidArgument(
                "Index out of bounds".to_string(),
            ));
        }

        let row_start = self.row_ptrs[i];
        let row_end = self.row_ptrs[i + 1];

        // Binary search for the column index
        for idx in row_start..row_end {
            if self.col_indices[idx] == j {
                return Ok(self.values[idx].clone());
            } else if self.col_indices[idx] > j {
                break;
            }
        }

        Ok(F::zero())
    }

    /// Convert to dense matrix representation
    pub fn to_dense(&self) -> Vec<F> {
        let mut dense = vec![F::zero(); self.rows * self.cols];

        for i in 0..self.rows {
            let row_start = self.row_ptrs[i];
            let row_end = self.row_ptrs[i + 1];

            for idx in row_start..row_end {
                let j = self.col_indices[idx];
                dense[i * self.cols + j] = self.values[idx].clone();
            }
        }

        dense
    }

    /// Matrix-vector multiplication: y = A * x
    pub fn matvec(&self, x: &[F]) -> Result<Vec<F>> {
        if x.len() != self.cols {
            return Err(MathError::InvalidArgument(
                "Vector dimension doesn't match matrix columns".to_string(),
            ));
        }

        let mut y = vec![F::zero(); self.rows];

        for i in 0..self.rows {
            let row_start = self.row_ptrs[i];
            let row_end = self.row_ptrs[i + 1];

            let mut sum = F::zero();
            for idx in row_start..row_end {
                let j = self.col_indices[idx];
                sum = sum + self.values[idx].clone() * x[j].clone();
            }
            y[i] = sum;
        }

        Ok(y)
    }

    /// Transpose the sparse matrix
    ///
    /// Converts from CSR to CSC (which is the same as CSR of the transpose)
    pub fn transpose(&self) -> Self {
        let mut new_values = vec![F::zero(); self.values.len()];
        let mut new_col_indices = vec![0; self.col_indices.len()];
        let mut new_row_ptrs = vec![0; self.cols + 1];

        // Count non-zeros per column
        for &col in &self.col_indices {
            new_row_ptrs[col + 1] += 1;
        }

        // Compute cumulative sum
        for i in 1..=self.cols {
            new_row_ptrs[i] += new_row_ptrs[i - 1];
        }

        // Keep track of current position in each new row
        let mut current_pos = new_row_ptrs.clone();

        // Fill in the values
        for i in 0..self.rows {
            let row_start = self.row_ptrs[i];
            let row_end = self.row_ptrs[i + 1];

            for idx in row_start..row_end {
                let j = self.col_indices[idx];
                let dest = current_pos[j];

                new_values[dest] = self.values[idx].clone();
                new_col_indices[dest] = i;

                current_pos[j] += 1;
            }
        }

        SparseMatrix {
            rows: self.cols,
            cols: self.rows,
            values: new_values,
            col_indices: new_col_indices,
            row_ptrs: new_row_ptrs,
        }
    }

    /// Add two sparse matrices
    pub fn add(&self, other: &SparseMatrix<F>) -> Result<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MathError::InvalidArgument(
                "Matrix dimensions must match for addition".to_string(),
            ));
        }

        // Simple implementation: convert to dense, add, convert back
        // A more efficient implementation would merge the sparse representations
        let mut dense = self.to_dense();
        let other_dense = other.to_dense();

        for i in 0..(self.rows * self.cols) {
            dense[i] = dense[i].clone() + other_dense[i].clone();
        }

        SparseMatrix::from_dense(self.rows, self.cols, &dense)
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &F) -> Self {
        let mut new_values = Vec::with_capacity(self.values.len());

        for val in &self.values {
            new_values.push(val.clone() * scalar.clone());
        }

        SparseMatrix {
            rows: self.rows,
            cols: self.cols,
            values: new_values,
            col_indices: self.col_indices.clone(),
            row_ptrs: self.row_ptrs.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_identity() {
        let id: SparseMatrix<i32> = SparseMatrix::identity(3);

        assert_eq!(id.rows(), 3);
        assert_eq!(id.cols(), 3);
        assert_eq!(id.nnz(), 3);

        assert_eq!(id.get(0, 0).unwrap(), 1);
        assert_eq!(id.get(1, 1).unwrap(), 1);
        assert_eq!(id.get(2, 2).unwrap(), 1);
        assert_eq!(id.get(0, 1).unwrap(), 0);
    }

    #[test]
    fn test_from_dense() {
        let dense = vec![1, 0, 2, 0, 0, 0, 3, 0, 4];
        let sparse = SparseMatrix::from_dense(3, 3, &dense).unwrap();

        assert_eq!(sparse.nnz(), 4);
        assert_eq!(sparse.get(0, 0).unwrap(), 1);
        assert_eq!(sparse.get(0, 2).unwrap(), 2);
        assert_eq!(sparse.get(2, 0).unwrap(), 3);
        assert_eq!(sparse.get(2, 2).unwrap(), 4);
    }

    #[test]
    fn test_to_dense() {
        let sparse = SparseMatrix::from_csr(
            2,
            2,
            vec![1, 2, 3, 4],
            vec![0, 1, 0, 1],
            vec![0, 2, 4],
        )
        .unwrap();

        let dense = sparse.to_dense();
        assert_eq!(dense, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_matvec() {
        // Matrix: [1 2]
        //         [3 4]
        let sparse = SparseMatrix::from_dense(2, 2, &[1, 2, 3, 4]).unwrap();
        let x = vec![5, 6];

        // Result should be [1*5 + 2*6, 3*5 + 4*6] = [17, 39]
        let y = sparse.matvec(&x).unwrap();

        assert_eq!(y, vec![17, 39]);
    }

    #[test]
    fn test_transpose() {
        // Matrix: [1 2]
        //         [3 4]
        let sparse = SparseMatrix::from_dense(2, 2, &[1, 2, 3, 4]).unwrap();
        let transposed = sparse.transpose();

        // Transpose should be: [1 3]
        //                      [2 4]
        assert_eq!(transposed.get(0, 0).unwrap(), 1);
        assert_eq!(transposed.get(0, 1).unwrap(), 3);
        assert_eq!(transposed.get(1, 0).unwrap(), 2);
        assert_eq!(transposed.get(1, 1).unwrap(), 4);
    }

    #[test]
    fn test_sparsity() {
        // 3x3 matrix with 2 non-zero elements
        let sparse = SparseMatrix::from_dense(3, 3, &[1, 0, 0, 0, 2, 0, 0, 0, 0]).unwrap();

        // Sparsity should be 7/9 ≈ 0.777
        let sparsity = sparse.sparsity();
        assert!((sparsity - 0.777).abs() < 0.01);
    }
}
