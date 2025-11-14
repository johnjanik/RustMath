//! Matrix construction arguments
//!
//! This module provides structures for handling various ways to construct matrices,
//! including from lists, sparse entries, dictionary entries, and other formats.

use crate::Matrix;
use rustmath_core::Ring;
use std::collections::HashMap;

/// A single entry in a sparse matrix representation
#[derive(Clone, Debug, PartialEq)]
pub struct SparseEntry<R> {
    /// Row index (0-based)
    pub i: usize,
    /// Column index (0-based)
    pub j: usize,
    /// Entry value
    pub entry: R,
}

impl<R> SparseEntry<R> {
    /// Create a new sparse entry
    pub fn new(i: usize, j: usize, entry: R) -> Self {
        SparseEntry { i, j, entry }
    }
}

/// Arguments for constructing a matrix
///
/// This structure encapsulates all the different ways a matrix can be constructed,
/// providing a unified interface for matrix initialization.
#[derive(Clone)]
pub struct MatrixArgs<R: Ring> {
    /// Number of rows
    pub nrows: Option<usize>,
    /// Number of columns
    pub ncols: Option<usize>,
    /// The base ring/field
    pub base: Option<String>,
    /// Whether the matrix is sparse
    pub sparse: bool,
    /// Dense entries (row-major order)
    pub entries: Option<Vec<R>>,
    /// Sparse entries as a vector
    pub sparse_entries: Option<Vec<SparseEntry<R>>>,
    /// Sparse entries as a dictionary
    pub dict_entries: Option<HashMap<(usize, usize), R>>,
    /// Row keys (for labeled rows)
    pub row_keys: Option<Vec<String>>,
    /// Column keys (for labeled columns)
    pub column_keys: Option<Vec<String>>,
    /// Additional keyword arguments
    pub kwds: HashMap<String, String>,
}

impl<R: Ring> MatrixArgs<R> {
    /// Create a new MatrixArgs with default values
    pub fn new() -> Self {
        MatrixArgs {
            nrows: None,
            ncols: None,
            base: None,
            sparse: false,
            entries: None,
            sparse_entries: None,
            dict_entries: None,
            row_keys: None,
            column_keys: None,
            kwds: HashMap::new(),
        }
    }

    /// Set the number of rows
    pub fn with_nrows(mut self, nrows: usize) -> Self {
        self.nrows = Some(nrows);
        self
    }

    /// Set the number of columns
    pub fn with_ncols(mut self, ncols: usize) -> Self {
        self.ncols = Some(ncols);
        self
    }

    /// Set the base ring
    pub fn with_base(mut self, base: String) -> Self {
        self.base = Some(base);
        self
    }

    /// Set whether the matrix is sparse
    pub fn with_sparse(mut self, sparse: bool) -> Self {
        self.sparse = sparse;
        self
    }

    /// Set dense entries
    pub fn with_entries(mut self, entries: Vec<R>) -> Self {
        self.entries = Some(entries);
        self
    }

    /// Set sparse entries
    pub fn with_sparse_entries(mut self, entries: Vec<SparseEntry<R>>) -> Self {
        self.sparse_entries = Some(entries);
        self
    }

    /// Set dictionary entries
    pub fn with_dict_entries(mut self, entries: HashMap<(usize, usize), R>) -> Self {
        self.dict_entries = Some(entries);
        self
    }

    /// Set row keys
    pub fn with_row_keys(mut self, keys: Vec<String>) -> Self {
        self.row_keys = Some(keys);
        self
    }

    /// Set column keys
    pub fn with_column_keys(mut self, keys: Vec<String>) -> Self {
        self.column_keys = Some(keys);
        self
    }

    /// Add a keyword argument
    pub fn with_kwarg(mut self, key: String, value: String) -> Self {
        self.kwds.insert(key, value);
        self
    }

    /// Build the matrix from the arguments
    ///
    /// # Errors
    ///
    /// Returns an error if the arguments are inconsistent or insufficient
    pub fn build(self) -> Result<Matrix<R>, String> {
        let nrows = self.nrows.ok_or("Number of rows not specified")?;
        let ncols = self.ncols.ok_or("Number of columns not specified")?;

        if let Some(entries) = self.entries {
            // Dense matrix from entries
            if entries.len() != nrows * ncols {
                return Err(format!(
                    "Expected {} entries for {}x{} matrix, got {}",
                    nrows * ncols, nrows, ncols, entries.len()
                ));
            }
            Matrix::from_vec(nrows, ncols, entries)
                .map_err(|e| format!("Failed to create matrix: {:?}", e))
        } else if let Some(sparse_entries) = self.sparse_entries {
            // Sparse matrix from sparse entries
            let mut data = vec![R::zero(); nrows * ncols];
            for entry in sparse_entries {
                if entry.i >= nrows || entry.j >= ncols {
                    return Err(format!(
                        "Sparse entry ({}, {}) out of bounds for {}x{} matrix",
                        entry.i, entry.j, nrows, ncols
                    ));
                }
                data[entry.i * ncols + entry.j] = entry.entry;
            }
            Matrix::from_vec(nrows, ncols, data)
                .map_err(|e| format!("Failed to create matrix: {:?}", e))
        } else if let Some(dict_entries) = self.dict_entries {
            // Sparse matrix from dictionary
            let mut data = vec![R::zero(); nrows * ncols];
            for ((i, j), value) in dict_entries {
                if i >= nrows || j >= ncols {
                    return Err(format!(
                        "Dictionary entry ({}, {}) out of bounds for {}x{} matrix",
                        i, j, nrows, ncols
                    ));
                }
                data[i * ncols + j] = value;
            }
            Matrix::from_vec(nrows, ncols, data)
                .map_err(|e| format!("Failed to create matrix: {:?}", e))
        } else {
            // Zero matrix
            Ok(Matrix::zeros(nrows, ncols))
        }
    }
}

impl<R: Ring> Default for MatrixArgs<R> {
    fn default() -> Self {
        Self::new()
    }
}

/// Initialize a MatrixArgs structure from various inputs
///
/// This is a convenience function that creates a MatrixArgs from common
/// construction patterns.
pub fn matrix_args_init<R: Ring>(
    nrows: usize,
    ncols: usize,
    entries: Option<Vec<R>>,
) -> MatrixArgs<R> {
    let mut args = MatrixArgs::new()
        .with_nrows(nrows)
        .with_ncols(ncols);

    if let Some(e) = entries {
        args = args.with_entries(e);
    }

    args
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;
    use rustmath_rationals::Rational;

    #[test]
    fn test_sparse_entry_creation() {
        let entry = SparseEntry::new(0, 1, Integer::from(5));
        assert_eq!(entry.i, 0);
        assert_eq!(entry.j, 1);
        assert_eq!(entry.entry, Integer::from(5));
    }

    #[test]
    fn test_matrix_args_creation() {
        let args = MatrixArgs::<Integer>::new();
        assert!(args.nrows.is_none());
        assert!(args.ncols.is_none());
        assert!(!args.sparse);
    }

    #[test]
    fn test_matrix_args_builder() {
        let args = MatrixArgs::<Integer>::new()
            .with_nrows(2)
            .with_ncols(3)
            .with_sparse(true);

        assert_eq!(args.nrows, Some(2));
        assert_eq!(args.ncols, Some(3));
        assert!(args.sparse);
    }

    #[test]
    fn test_build_dense_matrix() {
        let entries = vec![
            Integer::from(1), Integer::from(2),
            Integer::from(3), Integer::from(4),
        ];

        let args = MatrixArgs::new()
            .with_nrows(2)
            .with_ncols(2)
            .with_entries(entries);

        let matrix = args.build().unwrap();
        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 2);
        assert_eq!(matrix.get(0, 0).unwrap(), &Integer::from(1));
        assert_eq!(matrix.get(1, 1).unwrap(), &Integer::from(4));
    }

    #[test]
    fn test_build_sparse_matrix() {
        let sparse_entries = vec![
            SparseEntry::new(0, 0, Rational::from_integer(1)),
            SparseEntry::new(1, 1, Rational::from_integer(2)),
        ];

        let args = MatrixArgs::new()
            .with_nrows(2)
            .with_ncols(2)
            .with_sparse_entries(sparse_entries);

        let matrix = args.build().unwrap();
        assert_eq!(matrix.get(0, 0).unwrap(), &Rational::from_integer(1));
        assert_eq!(matrix.get(0, 1).unwrap(), &Rational::from_integer(0));
        assert_eq!(matrix.get(1, 0).unwrap(), &Rational::from_integer(0));
        assert_eq!(matrix.get(1, 1).unwrap(), &Rational::from_integer(2));
    }

    #[test]
    fn test_build_dict_matrix() {
        let mut dict = HashMap::new();
        dict.insert((0, 1), Integer::from(5));
        dict.insert((1, 0), Integer::from(7));

        let args = MatrixArgs::new()
            .with_nrows(2)
            .with_ncols(2)
            .with_dict_entries(dict);

        let matrix = args.build().unwrap();
        assert_eq!(matrix.get(0, 1).unwrap(), &Integer::from(5));
        assert_eq!(matrix.get(1, 0).unwrap(), &Integer::from(7));
    }

    #[test]
    fn test_build_zero_matrix() {
        let args = MatrixArgs::<Integer>::new()
            .with_nrows(3)
            .with_ncols(3);

        let matrix = args.build().unwrap();
        assert_eq!(matrix.rows(), 3);
        assert_eq!(matrix.cols(), 3);

        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(matrix.get(i, j).unwrap(), &Integer::from(0));
            }
        }
    }

    #[test]
    fn test_matrix_args_init() {
        let entries = vec![Integer::from(1), Integer::from(2), Integer::from(3), Integer::from(4)];
        let args = matrix_args_init(2, 2, Some(entries));

        assert_eq!(args.nrows, Some(2));
        assert_eq!(args.ncols, Some(2));
        assert!(args.entries.is_some());
    }

    #[test]
    fn test_invalid_dimensions() {
        let entries = vec![Integer::from(1), Integer::from(2)]; // Only 2 entries

        let args = MatrixArgs::new()
            .with_nrows(2)
            .with_ncols(2)
            .with_entries(entries);

        assert!(args.build().is_err());
    }

    #[test]
    fn test_sparse_entry_out_of_bounds() {
        let sparse_entries = vec![
            SparseEntry::new(5, 5, Integer::from(1)), // Out of bounds
        ];

        let args = MatrixArgs::new()
            .with_nrows(2)
            .with_ncols(2)
            .with_sparse_entries(sparse_entries);

        assert!(args.build().is_err());
    }
}
