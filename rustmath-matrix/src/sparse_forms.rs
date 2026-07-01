//! Sparse matrices over a Euclidean domain, with Smith/Hermite normal forms,
//! elementary divisors and rank (MAGMA ch. 27).
//!
//! MAGMA source: Handbook chapter 27 (Sparse Matrices), §27.3.2 (`Weight`,
//! `Support`, density), §27.6 (sparse ↔ dense bridges), §27.11.2 (`Rank`) and
//! §27.12 (`ElementaryDivisors`, Smith form).
//!
//! The historical [`crate::SparseMatrix`] is fixed to `F: Field` and therefore
//! cannot express integer (Euclidean-domain) normal forms. This module adds a
//! **purely additive**, sibling type [`SparseMatrixED<R>`] over any
//! `R: EuclideanDomain` (Z being the leading example), mirroring the dense
//! `integer_forms` path: it reuses the already-general
//! `Matrix<R: EuclideanDomain>::{smith_normal_form, hermite_normal_form,
//! elementary_divisors}` after a sparse→dense bridge.
//!
//! Representation is coordinate/dictionary-of-keys (`BTreeMap<(row, col), R>`),
//! which keeps only structural non-zeros and gives cheap `Weight`/`Support`.
//! The reduction itself currently densifies before eliminating; a sparse-native
//! Markowitz-pivoted elimination (ch. 27, `[DEJ84]`) is a performance-only
//! refinement left as future work — results are identical either way.

use crate::integer_forms::{HermiteNormalForm, SmithNormalForm};
use crate::Matrix;
use rustmath_core::{EuclideanDomain, Field, MathError, Result, Ring};
use std::collections::BTreeMap;

/// A sparse matrix over a ring `R`, stored as a map from `(row, col)` to its
/// non-zero value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseMatrixED<R: Ring + Ord> {
    rows: usize,
    cols: usize,
    /// Structural non-zeros, keyed by `(row, col)` in row-major order.
    entries: BTreeMap<(usize, usize), R>,
}

impl<R: Ring + Ord> SparseMatrixED<R> {
    /// A sparse zero matrix of the given shape.
    pub fn zero(rows: usize, cols: usize) -> Self {
        SparseMatrixED {
            rows,
            cols,
            entries: BTreeMap::new(),
        }
    }

    /// Build from `(row, col, value)` triples. Zero values and duplicate keys
    /// (later wins) are handled; out-of-range indices error.
    pub fn from_triples(
        rows: usize,
        cols: usize,
        triples: impl IntoIterator<Item = (usize, usize, R)>,
    ) -> Result<Self> {
        let mut entries = BTreeMap::new();
        for (i, j, v) in triples {
            if i >= rows || j >= cols {
                return Err(MathError::InvalidArgument(format!(
                    "entry ({i},{j}) out of range for {rows}x{cols}"
                )));
            }
            if v.is_zero() {
                entries.remove(&(i, j));
            } else {
                entries.insert((i, j), v);
            }
        }
        Ok(SparseMatrixED { rows, cols, entries })
    }

    /// Build from a dense row-major slice.
    pub fn from_dense(rows: usize, cols: usize, data: &[R]) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(MathError::InvalidArgument(
                "data length must equal rows * cols".to_string(),
            ));
        }
        let mut entries = BTreeMap::new();
        for i in 0..rows {
            for j in 0..cols {
                let v = &data[i * cols + j];
                if !v.is_zero() {
                    entries.insert((i, j), v.clone());
                }
            }
        }
        Ok(SparseMatrixED { rows, cols, entries })
    }

    /// Build from a dense [`Matrix`] (ch. 27.6 bridge).
    pub fn from_matrix(m: &Matrix<R>) -> Self {
        // `Matrix` guarantees a consistent shape, so this cannot fail.
        Self::from_dense(m.rows(), m.cols(), m.data()).expect("dense matrix shape is consistent")
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Number of structural non-zeros.
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    /// Entry at `(i, j)` (zero if absent).
    pub fn get(&self, i: usize, j: usize) -> R {
        self.entries.get(&(i, j)).cloned().unwrap_or_else(R::zero)
    }

    /// Set entry `(i, j)`; storing a zero removes it from the structure.
    pub fn set(&mut self, i: usize, j: usize, value: R) -> Result<()> {
        if i >= self.rows || j >= self.cols {
            return Err(MathError::InvalidArgument("index out of range".to_string()));
        }
        if value.is_zero() {
            self.entries.remove(&(i, j));
        } else {
            self.entries.insert((i, j), value);
        }
        Ok(())
    }

    /// The number of non-zero entries in row `i` (MAGMA `Weight` for a row).
    pub fn row_weight(&self, i: usize) -> usize {
        self.entries.range((i, 0)..(i + 1, 0)).count()
    }

    /// The total number of non-zero entries (MAGMA `Weight` for a matrix).
    pub fn weight(&self) -> usize {
        self.nnz()
    }

    /// The set of columns in which row `i` has a non-zero (MAGMA `Support`).
    pub fn row_support(&self, i: usize) -> Vec<usize> {
        self.entries
            .range((i, 0)..(i + 1, 0))
            .map(|(&(_, j), _)| j)
            .collect()
    }

    /// The fraction of entries that are structurally zero.
    pub fn density(&self) -> f64 {
        let total = self.rows * self.cols;
        if total == 0 {
            return 0.0;
        }
        self.nnz() as f64 / total as f64
    }

    /// Convert to a dense [`Matrix`] (ch. 27.6 bridge).
    pub fn to_dense(&self) -> Matrix<R> {
        let mut data = vec![R::zero(); self.rows * self.cols];
        for (&(i, j), v) in &self.entries {
            data[i * self.cols + j] = v.clone();
        }
        Matrix::from_vec(self.rows, self.cols, data).expect("shape is consistent")
    }

    /// Iterate structural non-zeros as `(row, col, &value)` in row-major order.
    pub fn iter_nonzero(&self) -> impl Iterator<Item = (usize, usize, &R)> {
        self.entries.iter().map(|(&(i, j), v)| (i, j, v))
    }
}

impl<R: EuclideanDomain + Ord> SparseMatrixED<R> {
    /// The Smith normal form (diagonal `S = P·A·Q`, `dᵢ | dᵢ₊₁`), generalised to
    /// any Euclidean domain (ch. 27.12). Delegates to the dense path.
    pub fn smith_normal_form(&self) -> Result<SmithNormalForm<R>> {
        self.to_dense().smith_normal_form()
    }

    /// The Hermite normal form `H = U·A` over a Euclidean domain (ch. 27).
    pub fn hermite_normal_form(&self) -> Result<HermiteNormalForm<R>> {
        self.to_dense().hermite_normal_form()
    }

    /// The elementary divisors `[e₁, …, e_d]` (`eᵢ | eᵢ₊₁`), i.e. the non-zero
    /// diagonal of the Smith form (MAGMA `ElementaryDivisors`, ch. 27.12).
    pub fn elementary_divisors(&self) -> Result<Vec<R>> {
        self.to_dense().elementary_divisors()
    }
}

impl<R: EuclideanDomain + Ord> SparseMatrixED<R> {
    /// The rank of the matrix over its field of fractions, computed as the
    /// number of non-zero elementary divisors (MAGMA `Rank`, ch. 27.11.2).
    ///
    /// This is valid for any Euclidean domain (a PID): the number of non-zero
    /// invariant factors of `A` equals `rank(A)` over `Frac(R)`.
    pub fn rank_via_smith(&self) -> Result<usize> {
        Ok(self.elementary_divisors()?.len())
    }
}

impl<F: Field + Ord> SparseMatrixED<F> {
    /// The rank over a field, via dense reduced row echelon rank.
    pub fn rank(&self) -> Result<usize> {
        self.to_dense().rank()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_core::Ring;
    use rustmath_integers::Integer;
    use rustmath_rationals::Rational;

    fn z(n: i64) -> Integer {
        Integer::from(n)
    }

    #[test]
    fn structure_weight_support_density() {
        // [[2,0,0],[0,0,3]]
        let m = SparseMatrixED::from_triples(2, 3, [(0, 0, z(2)), (1, 2, z(3))]).unwrap();
        assert_eq!(m.nnz(), 2);
        assert_eq!(m.weight(), 2);
        assert_eq!(m.row_weight(0), 1);
        assert_eq!(m.row_support(1), vec![2]);
        assert_eq!(m.get(0, 0), z(2));
        assert_eq!(m.get(0, 1), z(0));
        assert!((m.density() - 2.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn dense_bridge_roundtrip() {
        let dense = vec![z(1), z(0), z(0), z(4)];
        let sp = SparseMatrixED::from_dense(2, 2, &dense).unwrap();
        let back = sp.to_dense();
        assert_eq!(back.data(), &dense[..]);
        let sp2 = SparseMatrixED::from_matrix(&back);
        assert_eq!(sp, sp2);
    }

    #[test]
    fn elementary_divisors_over_Z() {
        // [[2,4],[6,8]] has Smith form diag(2, 4): elementary divisors [2, 4].
        let m = SparseMatrixED::from_dense(2, 2, &[z(2), z(4), z(6), z(8)]).unwrap();
        let divs = m.elementary_divisors().unwrap();
        assert_eq!(divs.len(), 2);
        // divisibility d1 | d2
        let (_, r) = divs[1].clone().div_rem(&divs[0]).unwrap();
        assert!(r.is_zero());
        // rank via smith = 2 (both divisors non-zero)
        assert_eq!(m.rank_via_smith().unwrap(), 2);
    }

    #[test]
    fn smith_form_is_diagonal_over_Z() {
        let m = SparseMatrixED::from_dense(2, 2, &[z(2), z(4), z(6), z(8)]).unwrap();
        let snf = m.smith_normal_form().unwrap();
        // off-diagonal entries vanish
        assert!(snf.s.get(0, 1).unwrap().is_zero());
        assert!(snf.s.get(1, 0).unwrap().is_zero());
    }

    #[test]
    fn hermite_form_is_upper_triangular_over_Z() {
        let m = SparseMatrixED::from_dense(2, 2, &[z(2), z(3), z(4), z(5)]).unwrap();
        let hnf = m.hermite_normal_form().unwrap();
        // lower-left entry is eliminated
        assert!(hnf.h.get(1, 0).unwrap().is_zero());
    }

    #[test]
    fn rank_over_field_rational() {
        // rank-1 matrix over Q
        let m = SparseMatrixED::from_dense(
            2,
            2,
            &[
                Rational::from_integer(1),
                Rational::from_integer(2),
                Rational::from_integer(2),
                Rational::from_integer(4),
            ],
        )
        .unwrap();
        assert_eq!(m.rank().unwrap(), 1);
    }
}
