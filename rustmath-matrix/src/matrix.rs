//! Matrix operations

use rustmath_core::{MathError, Result, Ring};
use std::fmt;
use std::ops::{Add, Mul, Sub};

/// Generic matrix over a ring R
#[derive(Clone, PartialEq)]
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
}
