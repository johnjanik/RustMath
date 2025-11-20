//! Matrix operations

use rustmath_core::{MathError, Result, Ring};
use std::fmt;
use std::ops::{Add, Index, IndexMut, Mul, Sub};

/// Generic matrix over a ring R
#[derive(Clone, PartialEq, Debug)]
pub struct Matrix<R: Ring> {
    data: Vec<R>,
    rows: usize,
    cols: usize,
}

impl<R: Ring> Matrix<R> {
    /// Create a new matrix from a flat vector (row-major order)
    pub fn from_vec(rows: usize, cols: usize, data: Vec<R>) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(MathError::InvalidArgument(format!(
                "Data length {} doesn't match dimensions {}x{}",
                data.len(),
                rows,
                cols
            )));
        }

        Ok(Matrix { data, rows, cols })
    }

    /// Create a zero matrix
    pub fn zeros(rows: usize, cols: usize) -> Self {
        let data = (0..rows * cols).map(|_| R::zero()).collect();
        Matrix { data, rows, cols }
    }

    /// Create an identity matrix
    pub fn identity(n: usize) -> Self {
        let mut data = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                data.push(if i == j { R::one() } else { R::zero() });
            }
        }
        Matrix {
            data,
            rows: n,
            cols: n,
        }
    }

    /// Get number of rows
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Get number of columns
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Get element at (i, j)
    pub fn get(&self, i: usize, j: usize) -> Result<&R> {
        if i >= self.rows || j >= self.cols {
            return Err(MathError::InvalidArgument("Index out of bounds".to_string()));
        }
        Ok(&self.data[i * self.cols + j])
    }

    /// Set element at (i, j)
    pub fn set(&mut self, i: usize, j: usize, value: R) -> Result<()> {
        if i >= self.rows || j >= self.cols {
            return Err(MathError::InvalidArgument("Index out of bounds".to_string()));
        }
        self.data[i * self.cols + j] = value;
        Ok(())
    }

    /// Transpose the matrix
    pub fn transpose(&self) -> Self {
        let mut data = Vec::with_capacity(self.rows * self.cols);
        for j in 0..self.cols {
            for i in 0..self.rows {
                data.push(self.data[i * self.cols + j].clone());
            }
        }

        Matrix {
            data,
            rows: self.cols,
            cols: self.rows,
        }
    }

    /// Check if matrix is square
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// Trace (sum of diagonal elements)
    pub fn trace(&self) -> Result<R> {
        if !self.is_square() {
            return Err(MathError::InvalidArgument(
                "Trace is only defined for square matrices".to_string(),
            ));
        }

        let mut sum = R::zero();
        for i in 0..self.rows {
            sum = sum + self.data[i * self.cols + i].clone();
        }
        Ok(sum)
    }

    /// Get the submatrix by removing row i and column j
    fn submatrix(&self, row: usize, col: usize) -> Result<Self> {
        if !self.is_square() || self.rows == 0 {
            return Err(MathError::InvalidArgument(
                "Submatrix requires a non-empty square matrix".to_string(),
            ));
        }

        let n = self.rows - 1;
        let mut data = Vec::with_capacity(n * n);

        for i in 0..self.rows {
            if i == row {
                continue;
            }
            for j in 0..self.cols {
                if j == col {
                    continue;
                }
                data.push(self.data[i * self.cols + j].clone());
            }
        }

        Ok(Matrix {
            data,
            rows: n,
            cols: n,
        })
    }

    /// Calculate determinant using cofactor expansion
    ///
    /// This uses recursive cofactor expansion which is O(n!).
    /// For larger matrices (>4x4), consider using LU decomposition instead.
    pub fn determinant(&self) -> Result<R> {
        if !self.is_square() {
            return Err(MathError::InvalidArgument(
                "Determinant is only defined for square matrices".to_string(),
            ));
        }

        match self.rows {
            0 => Err(MathError::InvalidArgument(
                "Cannot compute determinant of empty matrix".to_string(),
            )),
            1 => Ok(self.data[0].clone()),
            2 => {
                // ad - bc
                let a = self.data[0].clone();
                let b = self.data[1].clone();
                let c = self.data[2].clone();
                let d = self.data[3].clone();
                Ok(a * d - b * c)
            }
            3 => {
                // Sarrus rule for 3x3
                let a = self.data[0].clone();
                let b = self.data[1].clone();
                let c = self.data[2].clone();
                let d = self.data[3].clone();
                let e = self.data[4].clone();
                let f = self.data[5].clone();
                let g = self.data[6].clone();
                let h = self.data[7].clone();
                let i = self.data[8].clone();

                let pos = a.clone() * e.clone() * i.clone()
                    + b.clone() * f.clone() * g.clone()
                    + c.clone() * d.clone() * h.clone();

                let neg = c.clone() * e.clone() * g.clone()
                    + a.clone() * f.clone() * h.clone()
                    + b.clone() * d.clone() * i.clone();

                Ok(pos - neg)
            }
            _ => {
                // Cofactor expansion along first row
                let mut det = R::zero();
                let mut sign = R::one();

                for j in 0..self.cols {
                    let submat = self.submatrix(0, j)?;
                    let cofactor = sign.clone() * self.data[j].clone() * submat.determinant()?;
                    det = det + cofactor;
                    sign = R::zero() - sign; // Flip sign
                }

                Ok(det)
            }
        }
    }

    /// Get a row as a vector
    pub fn row(&self, i: usize) -> Result<Vec<R>> {
        if i >= self.rows {
            return Err(MathError::InvalidArgument("Row index out of bounds".to_string()));
        }

        let start = i * self.cols;
        let end = start + self.cols;
        Ok(self.data[start..end].to_vec())
    }

    /// Get a column as a vector
    pub fn col(&self, j: usize) -> Result<Vec<R>> {
        if j >= self.cols {
            return Err(MathError::InvalidArgument(
                "Column index out of bounds".to_string(),
            ));
        }

        let mut col_data = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            col_data.push(self.data[i * self.cols + j].clone());
        }
        Ok(col_data)
    }

    /// Check if matrix is symmetric (A = A^T)
    pub fn is_symmetric(&self) -> bool {
        if !self.is_square() {
            return false;
        }

        for i in 0..self.rows {
            for j in (i + 1)..self.cols {
                if self.data[i * self.cols + j] != self.data[j * self.cols + i] {
                    return false;
                }
            }
        }
        true
    }

    /// Check if matrix is diagonal
    pub fn is_diagonal(&self) -> bool {
        if !self.is_square() {
            return false;
        }

        for i in 0..self.rows {
            for j in 0..self.cols {
                if i != j && !self.data[i * self.cols + j].is_zero() {
                    return false;
                }
            }
        }
        true
    }

    /// Check if matrix is upper triangular
    pub fn is_upper_triangular(&self) -> bool {
        if !self.is_square() {
            return false;
        }

        for i in 0..self.rows {
            for j in 0..i {
                if !self.data[i * self.cols + j].is_zero() {
                    return false;
                }
            }
        }
        true
    }

    /// Check if matrix is lower triangular
    pub fn is_lower_triangular(&self) -> bool {
        if !self.is_square() {
            return false;
        }

        for i in 0..self.rows {
            for j in (i + 1)..self.cols {
                if !self.data[i * self.cols + j].is_zero() {
                    return false;
                }
            }
        }
        true
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &R) -> Self {
        let data = self.data.iter().map(|x| x.clone() * scalar.clone()).collect();
        Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Matrix power (A^n) for non-negative integer n
    pub fn pow(&self, n: u32) -> Result<Self> {
        if !self.is_square() {
            return Err(MathError::InvalidArgument(
                "Matrix power is only defined for square matrices".to_string(),
            ));
        }

        if n == 0 {
            return Ok(Self::identity(self.rows));
        }

        if n == 1 {
            return Ok(self.clone());
        }

        // Use binary exponentiation
        let mut result = Self::identity(self.rows);
        let mut base = self.clone();
        let mut exp = n;

        while exp > 0 {
            if exp % 2 == 1 {
                result = (result * base.clone())?;
            }
            base = (base.clone() * base)?;
            exp /= 2;
        }

        Ok(result)
    }

    /// Swap two rows
    pub fn swap_rows(&mut self, i: usize, j: usize) -> Result<()> {
        if i >= self.rows || j >= self.rows {
            return Err(MathError::InvalidArgument("Row index out of bounds".to_string()));
        }

        if i == j {
            return Ok(());
        }

        for k in 0..self.cols {
            self.data.swap(i * self.cols + k, j * self.cols + k);
        }

        Ok(())
    }

    /// Scale a row by a scalar
    pub fn scale_row(&mut self, i: usize, scalar: &R) -> Result<()> {
        if i >= self.rows {
            return Err(MathError::InvalidArgument("Row index out of bounds".to_string()));
        }

        for j in 0..self.cols {
            let idx = i * self.cols + j;
            self.data[idx] = self.data[idx].clone() * scalar.clone();
        }

        Ok(())
    }

    /// Add a multiple of row i to row j: row[j] += scalar * row[i]
    pub fn add_row_multiple(&mut self, i: usize, j: usize, scalar: &R) -> Result<()> {
        if i >= self.rows || j >= self.rows {
            return Err(MathError::InvalidArgument("Row index out of bounds".to_string()));
        }

        for k in 0..self.cols {
            let val = self.data[i * self.cols + k].clone() * scalar.clone();
            self.data[j * self.cols + k] = self.data[j * self.cols + k].clone() + val;
        }

        Ok(())
    }

    /// Compute Frobenius norm: sqrt(sum of squares of all elements)
    ///
    /// Only available for matrices over types with numeric conversion
    pub fn frobenius_norm(&self) -> f64
    where
        R: rustmath_core::NumericConversion,
    {
        let sum_of_squares: f64 = self
            .data
            .iter()
            .map(|x| {
                let val = x.to_f64().unwrap_or(0.0);
                val * val
            })
            .sum();

        sum_of_squares.sqrt()
    }

    /// Compute the infinity norm (maximum absolute row sum)
    pub fn infinity_norm(&self) -> f64
    where
        R: rustmath_core::NumericConversion,
    {
        let mut max_sum = 0.0;

        for i in 0..self.rows {
            let mut row_sum = 0.0;
            for j in 0..self.cols {
                let val = self.data[i * self.cols + j].to_f64().unwrap_or(0.0);
                row_sum += val.abs();
            }
            if row_sum > max_sum {
                max_sum = row_sum;
            }
        }

        max_sum
    }

    /// Compute the one norm (maximum absolute column sum)
    pub fn one_norm(&self) -> f64
    where
        R: rustmath_core::NumericConversion,
    {
        let mut max_sum = 0.0;

        for j in 0..self.cols {
            let mut col_sum = 0.0;
            for i in 0..self.rows {
                let val = self.data[i * self.cols + j].to_f64().unwrap_or(0.0);
                col_sum += val.abs();
            }
            if col_sum > max_sum {
                max_sum = col_sum;
            }
        }

        max_sum
    }

    /// Compute the condition number using the infinity norm
    ///
    /// The condition number measures how sensitive the solution to Ax=b is
    /// to perturbations in A and b. A condition number of 1 is ideal.
    /// Returns None if the matrix is singular.
    pub fn condition_number(&self) -> Option<f64>
    where
        R: rustmath_core::NumericConversion + rustmath_core::Field,
    {
        if !self.is_square() {
            return None;
        }

        // cond(A) = ||A|| * ||A^{-1}||
        let norm_a = self.infinity_norm();

        // Try to compute inverse
        let inv = self.inverse().ok()??;
        let norm_inv = inv.infinity_norm();

        Some(norm_a * norm_inv)
    }

    /// Check if matrix is Hermitian (for real matrices, this means symmetric)
    ///
    /// For complex matrices, A is Hermitian if A = A^H (conjugate transpose).
    /// For real matrices, this is equivalent to being symmetric.
    pub fn is_hermitian(&self) -> bool {
        // For real matrices, Hermitian is the same as symmetric
        self.is_symmetric()
    }

    /// Check if matrix is positive definite
    ///
    /// A matrix A is positive definite if:
    /// 1. It is symmetric (or Hermitian)
    /// 2. All eigenvalues are positive
    ///
    /// We use Sylvester's criterion: all leading principal minors are positive.
    /// This is equivalent to checking that all eigenvalues are positive.
    pub fn is_positive_definite(&self) -> bool
    where
        R: rustmath_core::NumericConversion + rustmath_core::Field,
    {
        if !self.is_square() || !self.is_symmetric() {
            return false;
        }

        // Use Sylvester's criterion: all leading principal minors must be positive
        for k in 1..=self.rows {
            // Extract leading k×k submatrix
            let mut submatrix_data = Vec::with_capacity(k * k);
            for i in 0..k {
                for j in 0..k {
                    submatrix_data.push(self.data[i * self.cols + j].clone());
                }
            }

            let submatrix = match Matrix::from_vec(k, k, submatrix_data) {
                Ok(m) => m,
                Err(_) => return false,
            };

            // Compute determinant of submatrix
            let det = match submatrix.determinant() {
                Ok(d) => d,
                Err(_) => return false,
            };

            // Check if determinant is positive
            let det_f64 = match det.to_f64() {
                Some(v) => v,
                None => return false,
            };

            if det_f64 <= 0.0 {
                return false;
            }
        }

        true
    }

    /// Get reference to internal data
    pub fn data(&self) -> &[R] {
        &self.data
    }

    /// Get mutable reference to internal data
    pub fn data_mut(&mut self) -> &mut [R] {
        &mut self.data
    }
}

// Implement Index trait for tuple indexing (immutable)
impl<R: Ring> Index<(usize, usize)> for Matrix<R> {
    type Output = R;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        if i >= self.rows || j >= self.cols {
            panic!("Matrix index out of bounds: ({}, {}) for {}x{} matrix", i, j, self.rows, self.cols);
        }
        &self.data[i * self.cols + j]
    }
}

// Implement IndexMut trait for tuple indexing (mutable)
impl<R: Ring> IndexMut<(usize, usize)> for Matrix<R> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        if i >= self.rows || j >= self.cols {
            panic!("Matrix index out of bounds: ({}, {}) for {}x{} matrix", i, j, self.rows, self.cols);
        }
        let cols = self.cols; // Capture for borrow checker
        &mut self.data[i * cols + j]
    }
}

impl<R: Ring> fmt::Display for Matrix<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "[")?;
        for i in 0..self.rows {
            write!(f, "  [")?;
            for j in 0..self.cols {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self.data[i * self.cols + j])?;
            }
            writeln!(f, "]")?;
        }
        write!(f, "]")
    }
}

// Typesetting implementation for matrices where elements implement MathDisplay
impl<R> rustmath_typesetting::MathDisplay for Matrix<R>
where
    R: Ring + rustmath_typesetting::MathDisplay,
{
    fn math_format(&self, options: &rustmath_typesetting::FormatOptions) -> String {
        use rustmath_typesetting::OutputFormat;

        // Convert each element to string using the element's math_format
        let mut rows_str: Vec<Vec<String>> = Vec::new();
        for i in 0..self.rows {
            let mut row: Vec<String> = Vec::new();
            for j in 0..self.cols {
                let elem = &self.data[i * self.cols + j];
                row.push(elem.math_format(options));
            }
            rows_str.push(row);
        }

        match options.format {
            OutputFormat::LaTeX => {
                rustmath_typesetting::latex::matrix(&rows_str, options.matrix_brackets, options.mode)
            }
            OutputFormat::Unicode => {
                rustmath_typesetting::unicode::matrix(&rows_str, options.matrix_brackets, options)
            }
            OutputFormat::Ascii => {
                rustmath_typesetting::ascii::matrix(&rows_str, options.matrix_brackets, options.mode)
            }
            OutputFormat::Html => {
                rustmath_typesetting::html::matrix(&rows_str, options.matrix_brackets)
            }
            OutputFormat::Plain => {
                // Simple text representation
                let mut result = String::from("[\n");
                for row in rows_str {
                    result.push_str("  [");
                    result.push_str(&row.join(", "));
                    result.push_str("]\n");
                }
                result.push(']');
                result
            }
        }
    }

    fn precedence(&self) -> i32 {
        rustmath_typesetting::utils::precedence::ATOMIC
    }
}

impl<R: Ring> Add for Matrix<R> {
    type Output = Result<Self>;

    fn add(self, other: Self) -> Result<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MathError::InvalidArgument(
                "Matrix dimensions must match for addition".to_string(),
            ));
        }

        let data = self
            .data
            .into_iter()
            .zip(other.data)
            .map(|(a, b)| a + b)
            .collect();

        Ok(Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        })
    }
}

impl<R: Ring> Sub for Matrix<R> {
    type Output = Result<Self>;

    fn sub(self, other: Self) -> Result<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MathError::InvalidArgument(
                "Matrix dimensions must match for subtraction".to_string(),
            ));
        }

        let data = self
            .data
            .into_iter()
            .zip(other.data)
            .map(|(a, b)| a - b)
            .collect();

        Ok(Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        })
    }
}

impl<R: Ring> Mul for Matrix<R> {
    type Output = Result<Self>;

    fn mul(self, other: Self) -> Result<Self> {
        if self.cols != other.rows {
            return Err(MathError::InvalidArgument(format!(
                "Cannot multiply {}x{} matrix with {}x{} matrix",
                self.rows, self.cols, other.rows, other.cols
            )));
        }

        let mut data = Vec::with_capacity(self.rows * other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = R::zero();
                for k in 0..self.cols {
                    let a = self.data[i * self.cols + k].clone();
                    let b = other.data[k * other.cols + j].clone();
                    sum = sum + a * b;
                }
                data.push(sum);
            }
        }

        Ok(Matrix {
            data,
            rows: self.rows,
            cols: other.cols,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation() {
        let m = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 2);
        assert_eq!(*m.get(0, 0).unwrap(), 1);
        assert_eq!(*m.get(1, 1).unwrap(), 4);
    }

    #[test]
    fn test_identity() {
        let id: Matrix<i32> = Matrix::identity(3);
        assert_eq!(*id.get(0, 0).unwrap(), 1);
        assert_eq!(*id.get(1, 1).unwrap(), 1);
        assert_eq!(*id.get(2, 2).unwrap(), 1);
        assert_eq!(*id.get(0, 1).unwrap(), 0);
    }

    #[test]
    fn test_transpose() {
        let m = Matrix::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let mt = m.transpose();
        assert_eq!(mt.rows(), 3);
        assert_eq!(mt.cols(), 2);
        assert_eq!(*mt.get(0, 0).unwrap(), 1);
        assert_eq!(*mt.get(1, 0).unwrap(), 2);
        assert_eq!(*mt.get(2, 1).unwrap(), 6);
    }

    #[test]
    fn test_addition() {
        let a = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
        let b = Matrix::from_vec(2, 2, vec![5, 6, 7, 8]).unwrap();
        let c = (a + b).unwrap();

        assert_eq!(*c.get(0, 0).unwrap(), 6);
        assert_eq!(*c.get(1, 1).unwrap(), 12);
    }

    #[test]
    fn test_multiplication() {
        let a = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
        let b = Matrix::from_vec(2, 2, vec![5, 6, 7, 8]).unwrap();
        let c = (a * b).unwrap();

        // [1 2] [5 6]   [19 22]
        // [3 4] [7 8] = [43 50]
        assert_eq!(*c.get(0, 0).unwrap(), 19);
        assert_eq!(*c.get(0, 1).unwrap(), 22);
        assert_eq!(*c.get(1, 0).unwrap(), 43);
        assert_eq!(*c.get(1, 1).unwrap(), 50);
    }

    #[test]
    fn test_trace() {
        let m = Matrix::from_vec(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        assert_eq!(m.trace().unwrap(), 15); // 1 + 5 + 9 = 15
    }

    #[test]
    fn test_determinant_2x2() {
        let m = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
        // det = 1*4 - 2*3 = -2
        assert_eq!(m.determinant().unwrap(), -2);
    }

    #[test]
    fn test_determinant_3x3() {
        let m = Matrix::from_vec(3, 3, vec![1, 2, 3, 0, 1, 4, 5, 6, 0]).unwrap();
        // Using Sarrus rule: det = 1 + 40 + 0 - 15 - 24 - 0 = 2
        assert_eq!(m.determinant().unwrap(), 1);
    }

    #[test]
    fn test_determinant_identity() {
        let id: Matrix<i32> = Matrix::identity(4);
        assert_eq!(id.determinant().unwrap(), 1);
    }

    #[test]
    fn test_is_symmetric() {
        // Symmetric matrix
        let m = Matrix::from_vec(3, 3, vec![1, 2, 3, 2, 4, 5, 3, 5, 6]).unwrap();
        assert!(m.is_symmetric());

        // Non-symmetric matrix
        let m = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
        assert!(!m.is_symmetric());

        // Identity is symmetric
        let id: Matrix<i32> = Matrix::identity(3);
        assert!(id.is_symmetric());
    }

    #[test]
    fn test_is_diagonal() {
        // Diagonal matrix
        let m = Matrix::from_vec(3, 3, vec![1, 0, 0, 0, 2, 0, 0, 0, 3]).unwrap();
        assert!(m.is_diagonal());

        // Non-diagonal matrix
        let m = Matrix::from_vec(2, 2, vec![1, 2, 0, 4]).unwrap();
        assert!(!m.is_diagonal());

        // Identity is diagonal
        let id: Matrix<i32> = Matrix::identity(3);
        assert!(id.is_diagonal());
    }

    #[test]
    fn test_is_triangular() {
        // Upper triangular
        let upper = Matrix::from_vec(3, 3, vec![1, 2, 3, 0, 4, 5, 0, 0, 6]).unwrap();
        assert!(upper.is_upper_triangular());
        assert!(!upper.is_lower_triangular());

        // Lower triangular
        let lower = Matrix::from_vec(3, 3, vec![1, 0, 0, 2, 3, 0, 4, 5, 6]).unwrap();
        assert!(lower.is_lower_triangular());
        assert!(!lower.is_upper_triangular());

        // Diagonal is both upper and lower triangular
        let diag = Matrix::from_vec(2, 2, vec![1, 0, 0, 2]).unwrap();
        assert!(diag.is_upper_triangular());
        assert!(diag.is_lower_triangular());
    }

    #[test]
    fn test_scalar_mul() {
        let m = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
        let scaled = m.scalar_mul(&3);

        assert_eq!(*scaled.get(0, 0).unwrap(), 3);
        assert_eq!(*scaled.get(0, 1).unwrap(), 6);
        assert_eq!(*scaled.get(1, 0).unwrap(), 9);
        assert_eq!(*scaled.get(1, 1).unwrap(), 12);
    }

    #[test]
    fn test_matrix_pow() {
        let m = Matrix::from_vec(2, 2, vec![1, 1, 0, 1]).unwrap();

        // A^0 = I
        let p0 = m.pow(0).unwrap();
        assert_eq!(p0, Matrix::identity(2));

        // A^1 = A
        let p1 = m.pow(1).unwrap();
        assert_eq!(p1, m);

        // A^2
        let p2 = m.pow(2).unwrap();
        assert_eq!(*p2.get(0, 0).unwrap(), 1);
        assert_eq!(*p2.get(0, 1).unwrap(), 2);
        assert_eq!(*p2.get(1, 0).unwrap(), 0);
        assert_eq!(*p2.get(1, 1).unwrap(), 1);
    }

    #[test]
    fn test_row_operations() {
        let mut m = Matrix::from_vec(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();

        // Test swap_rows
        m.swap_rows(0, 2).unwrap();
        assert_eq!(*m.get(0, 0).unwrap(), 7);
        assert_eq!(*m.get(2, 0).unwrap(), 1);

        // Test scale_row
        m.scale_row(1, &2).unwrap();
        assert_eq!(*m.get(1, 0).unwrap(), 8);
        assert_eq!(*m.get(1, 1).unwrap(), 10);

        // Test add_row_multiple
        let mut m2 = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
        m2.add_row_multiple(0, 1, &2).unwrap(); // row[1] += 2 * row[0]
        assert_eq!(*m2.get(1, 0).unwrap(), 5); // 3 + 2*1
        assert_eq!(*m2.get(1, 1).unwrap(), 8); // 4 + 2*2
    }
}
