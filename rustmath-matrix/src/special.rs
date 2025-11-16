//! Special matrix constructors
//!
//! This module provides constructors for various special matrices including:
//! - Identity, zero, and ones matrices
//! - Diagonal and block diagonal matrices
//! - Circulant, Toeplitz, and Hankel matrices
//! - Vandermonde and companion matrices
//! - Hilbert and Lehmer matrices
//! - Jordan blocks and elementary matrices

use crate::Matrix;
use rustmath_core::{Field, MathError, Result, Ring};
use rustmath_rationals::Rational;

/// Create an identity matrix of size n × n
///
/// # Arguments
/// * `n` - The dimension of the identity matrix
///
/// # Examples
/// ```
/// use rustmath_matrix::special::identity_matrix;
/// let id = identity_matrix::<i64>(3);
/// assert_eq!(id.rows(), 3);
/// assert_eq!(id.cols(), 3);
/// ```
pub fn identity_matrix<R: Ring>(n: usize) -> Matrix<R> {
    Matrix::identity(n)
}

/// Create a zero matrix of size m × n
///
/// # Arguments
/// * `rows` - Number of rows
/// * `cols` - Number of columns
///
/// # Examples
/// ```
/// use rustmath_matrix::special::zero_matrix;
/// let z = zero_matrix::<i64>(2, 3);
/// assert_eq!(z.rows(), 2);
/// assert_eq!(z.cols(), 3);
/// ```
pub fn zero_matrix<R: Ring>(rows: usize, cols: usize) -> Matrix<R> {
    Matrix::zeros(rows, cols)
}

/// Create a matrix filled with ones of size m × n
///
/// # Arguments
/// * `rows` - Number of rows
/// * `cols` - Number of columns
///
/// # Examples
/// ```
/// use rustmath_matrix::special::ones_matrix;
/// let ones = ones_matrix::<i64>(2, 3);
/// assert_eq!(ones.rows(), 2);
/// assert_eq!(ones.cols(), 3);
/// ```
pub fn ones_matrix<R: Ring>(rows: usize, cols: usize) -> Matrix<R> {
    let data = (0..rows * cols).map(|_| R::one()).collect();
    Matrix::from_vec(rows, cols, data).unwrap()
}

/// Create a diagonal matrix from a vector of diagonal entries
///
/// # Arguments
/// * `diagonal` - Vector of diagonal entries
///
/// # Examples
/// ```
/// use rustmath_matrix::special::diagonal_matrix;
/// let diag = diagonal_matrix(vec![1, 2, 3]);
/// assert_eq!(diag.rows(), 3);
/// assert_eq!(diag.cols(), 3);
/// ```
pub fn diagonal_matrix<R: Ring>(diagonal: Vec<R>) -> Matrix<R> {
    let n = diagonal.len();
    let mut data = Vec::with_capacity(n * n);

    for i in 0..n {
        for j in 0..n {
            if i == j {
                data.push(diagonal[i].clone());
            } else {
                data.push(R::zero());
            }
        }
    }

    Matrix::from_vec(n, n, data).unwrap()
}

/// Create a column matrix (column vector as n × 1 matrix)
///
/// # Arguments
/// * `entries` - Vector of column entries
///
/// # Examples
/// ```
/// use rustmath_matrix::special::column_matrix;
/// let col = column_matrix(vec![1, 2, 3]);
/// assert_eq!(col.rows(), 3);
/// assert_eq!(col.cols(), 1);
/// ```
pub fn column_matrix<R: Ring>(entries: Vec<R>) -> Matrix<R> {
    let rows = entries.len();
    Matrix::from_vec(rows, 1, entries).unwrap()
}

/// Create a Jordan block matrix with eigenvalue on the diagonal and ones on the superdiagonal
///
/// # Arguments
/// * `eigenvalue` - The eigenvalue to place on the diagonal
/// * `n` - Size of the Jordan block
///
/// # Examples
/// ```
/// use rustmath_matrix::special::jordan_block;
/// let j = jordan_block(5, 3);
/// assert_eq!(j.rows(), 3);
/// assert_eq!(j.cols(), 3);
/// ```
pub fn jordan_block<R: Ring>(eigenvalue: R, n: usize) -> Matrix<R> {
    let mut data = Vec::with_capacity(n * n);

    for i in 0..n {
        for j in 0..n {
            if i == j {
                data.push(eigenvalue.clone());
            } else if j == i + 1 {
                data.push(R::one());
            } else {
                data.push(R::zero());
            }
        }
    }

    Matrix::from_vec(n, n, data).unwrap()
}

/// Create a circulant matrix from a vector
///
/// A circulant matrix is a special matrix where each row is a cyclic shift of the previous row.
///
/// # Arguments
/// * `first_row` - First row of the circulant matrix
///
/// # Examples
/// ```
/// use rustmath_matrix::special::circulant;
/// let c = circulant(vec![1, 2, 3]);
/// assert_eq!(c.rows(), 3);
/// assert_eq!(c.cols(), 3);
/// ```
pub fn circulant<R: Ring>(first_row: Vec<R>) -> Matrix<R> {
    let n = first_row.len();
    let mut data = Vec::with_capacity(n * n);

    for i in 0..n {
        for j in 0..n {
            let idx = (j + n - i) % n;
            data.push(first_row[idx].clone());
        }
    }

    Matrix::from_vec(n, n, data).unwrap()
}

/// Create a Toeplitz matrix
///
/// A Toeplitz matrix is constant along diagonals.
///
/// # Arguments
/// * `first_column` - First column of the matrix
/// * `first_row` - First row of the matrix (must share first element with first_column)
///
/// # Examples
/// ```
/// use rustmath_matrix::special::toeplitz;
/// let t = toeplitz(vec![1, 2, 3], vec![1, 4, 5]).unwrap();
/// assert_eq!(t.rows(), 3);
/// assert_eq!(t.cols(), 3);
/// ```
pub fn toeplitz<R: Ring>(first_column: Vec<R>, first_row: Vec<R>) -> Result<Matrix<R>> {
    if first_column.is_empty() || first_row.is_empty() {
        return Err(MathError::InvalidArgument(
            "Toeplitz matrix requires non-empty row and column".to_string(),
        ));
    }

    let m = first_column.len();
    let n = first_row.len();
    let mut data = Vec::with_capacity(m * n);

    for i in 0..m {
        for j in 0..n {
            if j >= i {
                data.push(first_row[j - i].clone());
            } else {
                data.push(first_column[i - j].clone());
            }
        }
    }

    Matrix::from_vec(m, n, data)
}

/// Create a Hankel matrix
///
/// A Hankel matrix is constant along anti-diagonals.
///
/// # Arguments
/// * `first_column` - First column of the matrix
/// * `last_row` - Last row of the matrix (must share last element with first_column if both non-empty)
///
/// # Examples
/// ```
/// use rustmath_matrix::special::hankel;
/// let h = hankel(vec![1, 2, 3], vec![3, 4, 5]).unwrap();
/// assert_eq!(h.rows(), 3);
/// assert_eq!(h.cols(), 3);
/// ```
pub fn hankel<R: Ring>(first_column: Vec<R>, last_row: Vec<R>) -> Result<Matrix<R>> {
    if first_column.is_empty() {
        return Err(MathError::InvalidArgument(
            "Hankel matrix requires non-empty first column".to_string(),
        ));
    }

    let m = first_column.len();
    let n = last_row.len();
    let mut data = Vec::with_capacity(m * n);

    for i in 0..m {
        for j in 0..n {
            let sum = i + j;
            if sum < m {
                data.push(first_column[sum].clone());
            } else {
                let offset = sum - m + 1;
                if offset < n {
                    data.push(last_row[offset].clone());
                } else {
                    data.push(R::zero());
                }
            }
        }
    }

    Matrix::from_vec(m, n, data)
}

/// Create a Vandermonde matrix
///
/// The (i,j)-th entry is v[i]^j where v is the input vector.
///
/// # Arguments
/// * `v` - Vector of values
/// * `ncols` - Number of columns (powers go from 0 to ncols-1)
///
/// # Examples
/// ```
/// use rustmath_matrix::special::vandermonde;
/// let vdm = vandermonde(vec![1, 2, 3], 3);
/// assert_eq!(vdm.rows(), 3);
/// assert_eq!(vdm.cols(), 3);
/// ```
pub fn vandermonde<R: Ring>(v: Vec<R>, ncols: usize) -> Matrix<R> {
    let nrows = v.len();
    let mut data = Vec::with_capacity(nrows * ncols);

    for i in 0..nrows {
        let mut power = R::one();
        for _ in 0..ncols {
            data.push(power.clone());
            power = power * v[i].clone();
        }
    }

    Matrix::from_vec(nrows, ncols, data).unwrap()
}

/// Create a Hilbert matrix
///
/// The (i,j)-th entry is 1/(i+j+1).
///
/// # Arguments
/// * `n` - Dimension of the square Hilbert matrix
///
/// # Examples
/// ```
/// use rustmath_matrix::special::hilbert;
/// use rustmath_rationals::Rational;
/// let h = hilbert::<Rational>(3);
/// assert_eq!(h.rows(), 3);
/// assert_eq!(h.cols(), 3);
/// ```
pub fn hilbert<F: Field + From<Rational>>(n: usize) -> Matrix<F> {
    let mut data = Vec::with_capacity(n * n);

    for i in 0..n {
        for j in 0..n {
            let denominator = (i + j + 1) as i64;
            let rational = Rational::new(1, denominator).unwrap();
            data.push(F::from(rational));
        }
    }

    Matrix::from_vec(n, n, data).unwrap()
}

/// Create a Lehmer matrix
///
/// The (i,j)-th entry is min(i,j)/max(i,j) for 1-indexed i,j.
///
/// # Arguments
/// * `n` - Dimension of the square Lehmer matrix
///
/// # Examples
/// ```
/// use rustmath_matrix::special::lehmer;
/// use rustmath_rationals::Rational;
/// let l = lehmer::<Rational>(3);
/// assert_eq!(l.rows(), 3);
/// assert_eq!(l.cols(), 3);
/// ```
pub fn lehmer<F: Field + From<Rational>>(n: usize) -> Matrix<F> {
    let mut data = Vec::with_capacity(n * n);

    for i in 1..=n {
        for j in 1..=n {
            let min_ij = i.min(j) as i64;
            let max_ij = i.max(j) as i64;
            let rational = Rational::new(min_ij, max_ij).unwrap();
            data.push(F::from(rational));
        }
    }

    Matrix::from_vec(n, n, data).unwrap()
}

/// Create an elementary matrix of type I: row swap
///
/// E(i, j) swaps rows i and j when multiplied on the left.
///
/// # Arguments
/// * `n` - Dimension of the matrix
/// * `i` - First row to swap (0-indexed)
/// * `j` - Second row to swap (0-indexed)
///
/// # Examples
/// ```
/// use rustmath_matrix::special::elementary_matrix_swap;
/// let e = elementary_matrix_swap::<i64>(3, 0, 1).unwrap();
/// assert_eq!(e.rows(), 3);
/// ```
pub fn elementary_matrix_swap<R: Ring>(n: usize, i: usize, j: usize) -> Result<Matrix<R>> {
    if i >= n || j >= n {
        return Err(MathError::InvalidArgument(
            "Row indices must be less than n".to_string(),
        ));
    }

    let mut data = Vec::with_capacity(n * n);

    for row in 0..n {
        for col in 0..n {
            let value = if row == col && row != i && row != j {
                R::one()
            } else if row == i && col == j {
                R::one()
            } else if row == j && col == i {
                R::one()
            } else {
                R::zero()
            };
            data.push(value);
        }
    }

    Matrix::from_vec(n, n, data)
}

/// Create an elementary matrix of type II: row scaling
///
/// E(i, k) scales row i by factor k when multiplied on the left.
///
/// # Arguments
/// * `n` - Dimension of the matrix
/// * `i` - Row to scale (0-indexed)
/// * `k` - Scaling factor
///
/// # Examples
/// ```
/// use rustmath_matrix::special::elementary_matrix_scale;
/// let e = elementary_matrix_scale(3, 1, 5).unwrap();
/// assert_eq!(e.rows(), 3);
/// ```
pub fn elementary_matrix_scale<R: Ring>(n: usize, i: usize, k: R) -> Result<Matrix<R>> {
    if i >= n {
        return Err(MathError::InvalidArgument(
            "Row index must be less than n".to_string(),
        ));
    }

    let mut data = Vec::with_capacity(n * n);

    for row in 0..n {
        for col in 0..n {
            let value = if row == col && row == i {
                k.clone()
            } else if row == col {
                R::one()
            } else {
                R::zero()
            };
            data.push(value);
        }
    }

    Matrix::from_vec(n, n, data)
}

/// Create an elementary matrix of type III: row addition
///
/// E(i, j, k) adds k times row j to row i when multiplied on the left.
///
/// # Arguments
/// * `n` - Dimension of the matrix
/// * `i` - Target row (0-indexed)
/// * `j` - Source row (0-indexed)
/// * `k` - Scaling factor
///
/// # Examples
/// ```
/// use rustmath_matrix::special::elementary_matrix_add;
/// let e = elementary_matrix_add(3, 0, 1, 2).unwrap();
/// assert_eq!(e.rows(), 3);
/// ```
pub fn elementary_matrix_add<R: Ring>(n: usize, i: usize, j: usize, k: R) -> Result<Matrix<R>> {
    if i >= n || j >= n {
        return Err(MathError::InvalidArgument(
            "Row indices must be less than n".to_string(),
        ));
    }

    if i == j {
        return Err(MathError::InvalidArgument(
            "Source and target rows must be different".to_string(),
        ));
    }

    let mut data = Vec::with_capacity(n * n);

    for row in 0..n {
        for col in 0..n {
            let value = if row == col {
                R::one()
            } else if row == i && col == j {
                k.clone()
            } else {
                R::zero()
            };
            data.push(value);
        }
    }

    Matrix::from_vec(n, n, data)
}

/// Create a block diagonal matrix from a list of square matrices
///
/// # Arguments
/// * `blocks` - Vector of square matrices to place on the diagonal
///
/// # Examples
/// ```
/// use rustmath_matrix::special::{block_diagonal_matrix, identity_matrix};
/// let id2 = identity_matrix::<i64>(2);
/// let id3 = identity_matrix::<i64>(3);
/// let bd = block_diagonal_matrix(vec![id2, id3]).unwrap();
/// assert_eq!(bd.rows(), 5);
/// assert_eq!(bd.cols(), 5);
/// ```
pub fn block_diagonal_matrix<R: Ring>(blocks: Vec<Matrix<R>>) -> Result<Matrix<R>> {
    if blocks.is_empty() {
        return Err(MathError::InvalidArgument(
            "At least one block is required".to_string(),
        ));
    }

    // Calculate total size
    let mut total_rows = 0;
    let mut total_cols = 0;
    for block in &blocks {
        total_rows += block.rows();
        total_cols += block.cols();
    }

    let mut data = Vec::with_capacity(total_rows * total_cols);

    for (block_idx, block) in blocks.iter().enumerate() {
        let block_rows = block.rows();
        let block_cols = block.cols();

        // For each row in this block
        for i in 0..block_rows {
            // Add zeros before this block
            for prev_block in &blocks[..block_idx] {
                for _ in 0..prev_block.cols() {
                    data.push(R::zero());
                }
            }

            // Add the block's row
            for j in 0..block_cols {
                data.push(block.get(i, j).unwrap().clone());
            }

            // Add zeros after this block
            for next_block in &blocks[block_idx + 1..] {
                for _ in 0..next_block.cols() {
                    data.push(R::zero());
                }
            }
        }
    }

    Matrix::from_vec(total_rows, total_cols, data)
}

/// Create a block matrix from a 2D array of matrices
///
/// # Arguments
/// * `block_rows` - Number of block rows
/// * `block_cols` - Number of block columns
/// * `blocks` - Vector of matrices in row-major order
///
/// # Examples
/// ```
/// use rustmath_matrix::special::{block_matrix, identity_matrix, zero_matrix};
/// let id = identity_matrix::<i64>(2);
/// let z = zero_matrix::<i64>(2, 2);
/// let bm = block_matrix(2, 2, vec![id.clone(), z.clone(), z.clone(), id.clone()]).unwrap();
/// assert_eq!(bm.rows(), 4);
/// assert_eq!(bm.cols(), 4);
/// ```
pub fn block_matrix<R: Ring>(
    block_rows: usize,
    block_cols: usize,
    blocks: Vec<Matrix<R>>,
) -> Result<Matrix<R>> {
    if blocks.len() != block_rows * block_cols {
        return Err(MathError::InvalidArgument(format!(
            "Expected {} blocks, got {}",
            block_rows * block_cols,
            blocks.len()
        )));
    }

    if blocks.is_empty() {
        return Err(MathError::InvalidArgument(
            "At least one block is required".to_string(),
        ));
    }

    // Check that blocks in the same row have the same height
    // and blocks in the same column have the same width
    let mut row_heights = vec![0; block_rows];
    let mut col_widths = vec![0; block_cols];

    for i in 0..block_rows {
        for j in 0..block_cols {
            let block = &blocks[i * block_cols + j];

            if j == 0 {
                row_heights[i] = block.rows();
            } else if block.rows() != row_heights[i] {
                return Err(MathError::InvalidArgument(format!(
                    "Blocks in row {} have inconsistent heights",
                    i
                )));
            }

            if i == 0 {
                col_widths[j] = block.cols();
            } else if block.cols() != col_widths[j] {
                return Err(MathError::InvalidArgument(format!(
                    "Blocks in column {} have inconsistent widths",
                    j
                )));
            }
        }
    }

    let total_rows: usize = row_heights.iter().sum();
    let total_cols: usize = col_widths.iter().sum();

    let mut data = Vec::with_capacity(total_rows * total_cols);

    for block_row in 0..block_rows {
        let height = row_heights[block_row];

        for i in 0..height {
            for block_col in 0..block_cols {
                let block = &blocks[block_row * block_cols + block_col];
                let width = col_widths[block_col];

                for j in 0..width {
                    data.push(block.get(i, j).unwrap().clone());
                }
            }
        }
    }

    Matrix::from_vec(total_rows, total_cols, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_matrix() {
        let id: Matrix<i64> = identity_matrix(3);
        assert_eq!(id.rows(), 3);
        assert_eq!(id.cols(), 3);
        assert_eq!(*id.get(0, 0).unwrap(), 1);
        assert_eq!(*id.get(1, 1).unwrap(), 1);
        assert_eq!(*id.get(0, 1).unwrap(), 0);
    }

    #[test]
    fn test_zero_matrix() {
        let z: Matrix<i64> = zero_matrix(2, 3);
        assert_eq!(z.rows(), 2);
        assert_eq!(z.cols(), 3);
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(*z.get(i, j).unwrap(), 0);
            }
        }
    }

    #[test]
    fn test_ones_matrix() {
        let ones: Matrix<i64> = ones_matrix(2, 3);
        assert_eq!(ones.rows(), 2);
        assert_eq!(ones.cols(), 3);
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(*ones.get(i, j).unwrap(), 1);
            }
        }
    }

    #[test]
    fn test_diagonal_matrix() {
        let diag = diagonal_matrix(vec![1, 2, 3]);
        assert_eq!(diag.rows(), 3);
        assert_eq!(diag.cols(), 3);
        assert_eq!(*diag.get(0, 0).unwrap(), 1);
        assert_eq!(*diag.get(1, 1).unwrap(), 2);
        assert_eq!(*diag.get(2, 2).unwrap(), 3);
        assert_eq!(*diag.get(0, 1).unwrap(), 0);
    }

    #[test]
    fn test_jordan_block() {
        let j = jordan_block(5, 3);
        assert_eq!(j.rows(), 3);
        assert_eq!(j.cols(), 3);
        assert_eq!(*j.get(0, 0).unwrap(), 5);
        assert_eq!(*j.get(1, 1).unwrap(), 5);
        assert_eq!(*j.get(0, 1).unwrap(), 1);
        assert_eq!(*j.get(1, 2).unwrap(), 1);
        assert_eq!(*j.get(0, 2).unwrap(), 0);
    }

    #[test]
    fn test_circulant() {
        let c = circulant(vec![1, 2, 3]);
        assert_eq!(c.rows(), 3);
        assert_eq!(c.cols(), 3);
        // First row: [1, 2, 3]
        assert_eq!(*c.get(0, 0).unwrap(), 1);
        assert_eq!(*c.get(0, 1).unwrap(), 2);
        assert_eq!(*c.get(0, 2).unwrap(), 3);
        // Second row: [3, 1, 2]
        assert_eq!(*c.get(1, 0).unwrap(), 3);
        assert_eq!(*c.get(1, 1).unwrap(), 1);
        assert_eq!(*c.get(1, 2).unwrap(), 2);
    }

    #[test]
    fn test_toeplitz() {
        let t = toeplitz(vec![1, 2, 3], vec![1, 4, 5]).unwrap();
        assert_eq!(t.rows(), 3);
        assert_eq!(t.cols(), 3);
        assert_eq!(*t.get(0, 0).unwrap(), 1);
        assert_eq!(*t.get(0, 1).unwrap(), 4);
        assert_eq!(*t.get(1, 0).unwrap(), 2);
    }

    #[test]
    fn test_hankel() {
        let h = hankel(vec![1, 2, 3], vec![3, 4, 5]).unwrap();
        assert_eq!(h.rows(), 3);
        assert_eq!(h.cols(), 3);
        assert_eq!(*h.get(0, 0).unwrap(), 1);
        assert_eq!(*h.get(0, 1).unwrap(), 2);
        assert_eq!(*h.get(1, 1).unwrap(), 3);
    }

    #[test]
    fn test_vandermonde() {
        let v = vandermonde(vec![1, 2, 3], 3);
        assert_eq!(v.rows(), 3);
        assert_eq!(v.cols(), 3);
        // First row: [1, 1, 1] (powers of 1)
        assert_eq!(*v.get(0, 0).unwrap(), 1);
        assert_eq!(*v.get(0, 1).unwrap(), 1);
        assert_eq!(*v.get(0, 2).unwrap(), 1);
        // Second row: [1, 2, 4] (powers of 2)
        assert_eq!(*v.get(1, 0).unwrap(), 1);
        assert_eq!(*v.get(1, 1).unwrap(), 2);
        assert_eq!(*v.get(1, 2).unwrap(), 4);
    }

    #[test]
    fn test_hilbert() {
        let h: Matrix<Rational> = hilbert(3);
        assert_eq!(h.rows(), 3);
        assert_eq!(h.cols(), 3);
        assert_eq!(*h.get(0, 0).unwrap(), Rational::new(1, 1).unwrap());
        assert_eq!(*h.get(0, 1).unwrap(), Rational::new(1, 2).unwrap());
        assert_eq!(*h.get(1, 1).unwrap(), Rational::new(1, 3).unwrap());
    }

    #[test]
    fn test_lehmer() {
        let l: Matrix<Rational> = lehmer(3);
        assert_eq!(l.rows(), 3);
        assert_eq!(l.cols(), 3);
        assert_eq!(*l.get(0, 0).unwrap(), Rational::new(1, 1).unwrap());
        assert_eq!(*l.get(0, 1).unwrap(), Rational::new(1, 2).unwrap());
        assert_eq!(*l.get(1, 0).unwrap(), Rational::new(1, 2).unwrap());
    }

    #[test]
    fn test_elementary_matrix_swap() {
        let e: Matrix<i64> = elementary_matrix_swap(3, 0, 1).unwrap();
        assert_eq!(e.rows(), 3);
        assert_eq!(*e.get(0, 0).unwrap(), 0);
        assert_eq!(*e.get(0, 1).unwrap(), 1);
        assert_eq!(*e.get(1, 0).unwrap(), 1);
    }

    #[test]
    fn test_elementary_matrix_scale() {
        let e = elementary_matrix_scale(3, 1, 5).unwrap();
        assert_eq!(e.rows(), 3);
        assert_eq!(*e.get(0, 0).unwrap(), 1);
        assert_eq!(*e.get(1, 1).unwrap(), 5);
        assert_eq!(*e.get(2, 2).unwrap(), 1);
    }

    #[test]
    fn test_elementary_matrix_add() {
        let e = elementary_matrix_add(3, 0, 1, 2).unwrap();
        assert_eq!(e.rows(), 3);
        assert_eq!(*e.get(0, 0).unwrap(), 1);
        assert_eq!(*e.get(0, 1).unwrap(), 2);
        assert_eq!(*e.get(1, 1).unwrap(), 1);
    }

    #[test]
    fn test_block_diagonal_matrix() {
        let id2: Matrix<i64> = identity_matrix(2);
        let id3: Matrix<i64> = identity_matrix(3);
        let bd = block_diagonal_matrix(vec![id2, id3]).unwrap();

        assert_eq!(bd.rows(), 5);
        assert_eq!(bd.cols(), 5);
        assert_eq!(*bd.get(0, 0).unwrap(), 1);
        assert_eq!(*bd.get(1, 1).unwrap(), 1);
        assert_eq!(*bd.get(2, 2).unwrap(), 1);
        assert_eq!(*bd.get(0, 2).unwrap(), 0);
    }

    #[test]
    fn test_block_matrix() {
        let id: Matrix<i64> = identity_matrix(2);
        let z: Matrix<i64> = zero_matrix(2, 2);

        let bm = block_matrix(2, 2, vec![id.clone(), z.clone(), z.clone(), id.clone()]).unwrap();

        assert_eq!(bm.rows(), 4);
        assert_eq!(bm.cols(), 4);
        assert_eq!(*bm.get(0, 0).unwrap(), 1);
        assert_eq!(*bm.get(0, 2).unwrap(), 0);
        assert_eq!(*bm.get(2, 2).unwrap(), 1);
    }

    #[test]
    fn test_column_matrix() {
        let col = column_matrix(vec![1, 2, 3]);
        assert_eq!(col.rows(), 3);
        assert_eq!(col.cols(), 1);
        assert_eq!(*col.get(0, 0).unwrap(), 1);
        assert_eq!(*col.get(1, 0).unwrap(), 2);
        assert_eq!(*col.get(2, 0).unwrap(), 3);
    }
}
