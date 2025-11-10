//! Python bindings for Matrix type

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyIndexError};
use rustmath_matrix::Matrix;
use rustmath_integers::Integer;
use rustmath_core::Ring;
use crate::integers::PyInteger;

/// Python wrapper for RustMath Matrix (of integers)
#[pyclass]
#[derive(Clone)]
pub struct PyMatrix {
    pub(crate) inner: Matrix<Integer>,
}

#[pymethods]
impl PyMatrix {
    /// Create a new matrix from a 2D list
    /// Example: PyMatrix.from_list([[1, 2], [3, 4]])
    #[staticmethod]
    fn from_list(data: Vec<Vec<i64>>) -> PyResult<Self> {
        if data.is_empty() {
            return Err(PyValueError::new_err("Matrix cannot be empty"));
        }

        let rows = data.len();
        let cols = data[0].len();

        if cols == 0 {
            return Err(PyValueError::new_err("Matrix rows cannot be empty"));
        }

        // Check all rows have same length
        for (i, row) in data.iter().enumerate() {
            if row.len() != cols {
                return Err(PyValueError::new_err(
                    format!("Row {} has length {}, expected {}", i, row.len(), cols)
                ));
            }
        }

        // Flatten to row-major vector
        let flat: Vec<Integer> = data.into_iter()
            .flat_map(|row| row.into_iter().map(Integer::from))
            .collect();

        Matrix::from_vec(rows, cols, flat)
            .map(|inner| PyMatrix { inner })
            .map_err(|e| PyValueError::new_err(format!("Error creating matrix: {:?}", e)))
    }

    /// Create a zero matrix
    #[staticmethod]
    fn zeros(rows: usize, cols: usize) -> PyMatrix {
        PyMatrix {
            inner: Matrix::zeros(rows, cols),
        }
    }

    /// Create an identity matrix
    #[staticmethod]
    fn identity(n: usize) -> PyMatrix {
        PyMatrix {
            inner: Matrix::identity(n),
        }
    }

    // ========== Accessors ==========

    /// Get number of rows
    fn rows(&self) -> usize {
        self.inner.rows()
    }

    /// Get number of columns
    fn cols(&self) -> usize {
        self.inner.cols()
    }

    /// Get shape as tuple (rows, cols)
    fn shape(&self) -> (usize, usize) {
        (self.inner.rows(), self.inner.cols())
    }

    /// Get element at (i, j)
    fn get(&self, i: usize, j: usize) -> PyResult<PyInteger> {
        self.inner.get(i, j)
            .map(|val| PyInteger { inner: val.clone() })
            .map_err(|e| PyIndexError::new_err(format!("{:?}", e)))
    }

    /// Set element at (i, j)
    fn set(&mut self, i: usize, j: usize, value: i64) -> PyResult<()> {
        self.inner.set(i, j, Integer::from(value))
            .map_err(|e| PyIndexError::new_err(format!("{:?}", e)))
    }

    // ========== Operations ==========

    /// Transpose the matrix
    fn transpose(&self) -> PyMatrix {
        PyMatrix {
            inner: self.inner.transpose(),
        }
    }

    /// Compute determinant
    fn determinant(&self) -> PyResult<PyInteger> {
        self.inner.determinant()
            .map(|det| PyInteger { inner: det })
            .map_err(|e| PyValueError::new_err(format!("Error computing determinant: {:?}", e)))
    }

    /// Compute trace (sum of diagonal elements)
    fn trace(&self) -> PyResult<PyInteger> {
        self.inner.trace()
            .map(|tr| PyInteger { inner: tr })
            .map_err(|e| PyValueError::new_err(format!("Error computing trace: {:?}", e)))
    }

    /// Check if matrix is square
    fn is_square(&self) -> bool {
        self.inner.is_square()
    }

    // ========== Arithmetic Operations ==========

    fn __add__(&self, other: &PyMatrix) -> PyResult<PyMatrix> {
        if self.shape() != other.shape() {
            return Err(PyValueError::new_err(
                format!("Cannot add matrices of different shapes: {:?} and {:?}",
                    self.shape(), other.shape())
            ));
        }
        (self.inner.clone() + other.inner.clone())
            .map(|m| PyMatrix { inner: m })
            .map_err(|e| PyValueError::new_err(format!("Error: {:?}", e)))
    }

    fn __sub__(&self, other: &PyMatrix) -> PyResult<PyMatrix> {
        if self.shape() != other.shape() {
            return Err(PyValueError::new_err(
                format!("Cannot subtract matrices of different shapes: {:?} and {:?}",
                    self.shape(), other.shape())
            ));
        }
        (self.inner.clone() - other.inner.clone())
            .map(|m| PyMatrix { inner: m })
            .map_err(|e| PyValueError::new_err(format!("Error: {:?}", e)))
    }

    fn __mul__(&self, other: &PyMatrix) -> PyResult<PyMatrix> {
        if self.cols() != other.rows() {
            return Err(PyValueError::new_err(
                format!("Cannot multiply: {}x{} by {}x{}",
                    self.rows(), self.cols(), other.rows(), other.cols())
            ));
        }
        (self.inner.clone() * other.inner.clone())
            .map(|m| PyMatrix { inner: m })
            .map_err(|e| PyValueError::new_err(format!("Error: {:?}", e)))
    }

    fn __neg__(&self) -> PyResult<PyMatrix> {
        // Matrix doesn't implement Neg directly, use scalar multiplication
        let neg_one = rustmath_integers::Integer::from(-1);
        let mut result = self.inner.clone();

        // Negate each element
        for i in 0..result.rows() {
            for j in 0..result.cols() {
                if let Ok(val) = result.get(i, j) {
                    let negated = neg_one.clone() * val.clone();
                    let _ = result.set(i, j, negated);
                }
            }
        }

        Ok(PyMatrix { inner: result })
    }

    // ========== Comparison ==========

    fn __eq__(&self, other: &PyMatrix) -> bool {
        self.inner == other.inner
    }

    fn __ne__(&self, other: &PyMatrix) -> bool {
        self.inner != other.inner
    }

    // ========== Conversion ==========

    /// Convert to nested list
    fn to_list(&self) -> Vec<Vec<String>> {
        let mut result = Vec::new();
        for i in 0..self.inner.rows() {
            let mut row = Vec::new();
            for j in 0..self.inner.cols() {
                if let Ok(val) = self.inner.get(i, j) {
                    row.push(format!("{}", val));
                }
            }
            result.push(row);
        }
        result
    }

    // ========== String Representation ==========

    fn __str__(&self) -> String {
        let mut s = String::from("[\n");
        for i in 0..self.inner.rows() {
            s.push_str("  [");
            for j in 0..self.inner.cols() {
                if let Ok(val) = self.inner.get(i, j) {
                    s.push_str(&format!("{}", val));
                    if j < self.inner.cols() - 1 {
                        s.push_str(", ");
                    }
                }
            }
            s.push_str("]");
            if i < self.inner.rows() - 1 {
                s.push_str(",");
            }
            s.push('\n');
        }
        s.push(']');
        s
    }

    fn __repr__(&self) -> String {
        format!("Matrix({}x{})", self.inner.rows(), self.inner.cols())
    }
}
