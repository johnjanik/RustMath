//! Linear system solving algorithms
//!
//! MAGMA Handbook ch. 26 §26.8 (digest `rustmath-lava/docs/Chapter_26_Matrices.md:222-225`).
//! MAGMA uses the row convention `V·A = W`. The four forms and their methods here:
//!
//! | MAGMA | Here |
//! |---|---|
//! | `IsConsistent(A, w)`, w a vector | [`Matrix::solve_left`] (`is_some()`, the solution) |
//! | `Solution(A, w)`, and the nullspace | [`Matrix::solve_left_with_kernel`] |
//! | `IsConsistent(A, W)`, W an r×n matrix | [`Matrix::solve_left_matrix`] |
//! | `Solution(A, W)` | [`Matrix::solve_left_matrix_with_kernel`] |
//! | `IsConsistent(A, Q)`, Q a sequence of vectors | [`Matrix::solve_left_sequence`] |
//! | `Solution(A, Q)` | [`Matrix::solve_left_sequence_with_kernel`] |
//!
//! MAGMA's `Solution` raises an error for an inconsistent system; here that is the documented
//! `Ok(None)`. [`Matrix::solve`] and [`Matrix::solve_matrix`] are the column convention
//! `A·X = B` (Sage `solve_right`). Fields only: MAGMA also allows Euclidean rings such as Z,
//! which these methods do not cover.
//!
//! Algorithm: Gauss–Jordan elimination on `[A | B]` (B with k columns) and the Rouché–Capelli
//! criterion (K. Hoffman and R. Kunze, *Linear Algebra*, 2nd ed., §1.4, row-reduced echelon
//! matrices and the solution of `A X = Y`): column j of B is consistent iff
//! `rank [A|b_j] = rank A`, i.e. iff column `n + j` is not a pivot column of `RREF [A|B]`. The
//! left block of `RREF [A|B]` is `RREF A` (the pivot choice in a column depends only on that
//! column), so one elimination decides all k systems: every column is consistent iff no pivot
//! lies in a B column. The particular solution returned sets every free (non-pivot) variable
//! to 0.
//!
//! Cost: one elimination of an m×(n+k) matrix, `O(m·n·(n+k))` field operations, against
//! `O(k·m·n²)` for k separate solves. Over Q, dense, n = k = 30, entries in [-5, 5], debug
//! build: 0.35 s against 1.6 s (the same V). Over Q the entries grow as in any fraction-based
//! elimination (no Dixon lifting).

use crate::Matrix;
use rustmath_core::{Field, MathError, Result};

/// Row echelon form result
pub struct RowEchelonForm<F: Field> {
    /// The matrix in row echelon form
    pub matrix: Matrix<F>,
    /// The pivot columns
    pub pivots: Vec<usize>,
    /// The rank of the matrix
    pub rank: usize,
}

impl<F: Field> Matrix<F> {
    /// Convert to row echelon form using Gaussian elimination
    ///
    /// This modifies the matrix in place and returns information about pivots.
    /// Uses partial pivoting for numerical stability.
    pub fn row_echelon_form(&self) -> Result<RowEchelonForm<F>> {
        let mut result = self.clone();
        let mut pivots = Vec::new();
        let mut current_row = 0;

        for col in 0..result.cols() {
            if current_row >= result.rows() {
                break;
            }

            // Find pivot (largest absolute value in column, starting from current_row)
            let mut pivot_row = current_row;
            let mut max_val = result.data()[current_row * result.cols() + col].clone();

            for row in (current_row + 1)..result.rows() {
                let val = result.data()[row * result.cols() + col].clone();
                // For fields without absolute value, we just use the first non-zero
                if !val.is_zero() && max_val.is_zero() {
                    max_val = val;
                    pivot_row = row;
                }
            }

            // If column is all zeros, skip
            if max_val.is_zero() {
                continue;
            }

            // Swap rows if needed
            if pivot_row != current_row {
                let cols = result.cols();
                for j in 0..cols {
                    let temp = result.data()[current_row * cols + j].clone();
                    let swap_val = result.data()[pivot_row * cols + j].clone();
                    result.data_mut()[current_row * cols + j] = swap_val;
                    result.data_mut()[pivot_row * cols + j] = temp;
                }
            }

            pivots.push(col);

            // Scale pivot row
            let pivot = result.data()[current_row * result.cols() + col].clone();
            if pivot.is_zero() {
                return Err(MathError::DivisionByZero);
            }

            let cols = result.cols();
            for j in col..cols {
                let val = result.data()[current_row * cols + j].clone();
                result.data_mut()[current_row * cols + j] = val / pivot.clone();
            }

            // Eliminate column below pivot
            let rows = result.rows();
            for row in (current_row + 1)..rows {
                let factor = result.data()[row * cols + col].clone();
                for j in col..cols {
                    let pivot_val = result.data()[current_row * cols + j].clone();
                    let current_val = result.data()[row * cols + j].clone();
                    result.data_mut()[row * cols + j] = current_val - factor.clone() * pivot_val;
                }
            }

            current_row += 1;
        }

        Ok(RowEchelonForm {
            rank: pivots.len(),
            pivots,
            matrix: result,
        })
    }

    /// Convert to reduced row echelon form (Gauss-Jordan elimination)
    ///
    /// This produces the unique reduced row echelon form where:
    /// - Each pivot is 1
    /// - Each pivot is the only non-zero entry in its column
    pub fn reduced_row_echelon_form(&self) -> Result<RowEchelonForm<F>> {
        let mut ref_form = self.row_echelon_form()?;

        // Back-substitution to make zeros above pivots
        for i in (0..ref_form.pivots.len()).rev() {
            let pivot_col = ref_form.pivots[i];
            let pivot_row = i;

            // Eliminate above pivot
            let matrix_cols = ref_form.matrix.cols();
            for row in 0..pivot_row {
                let factor = ref_form.matrix.data()[row * matrix_cols + pivot_col].clone();
                if !factor.is_zero() {
                    for j in 0..matrix_cols {
                        let pivot_val = ref_form.matrix.data()[pivot_row * matrix_cols + j].clone();
                        let current_val = ref_form.matrix.data()[row * matrix_cols + j].clone();
                        ref_form.matrix.data_mut()[row * matrix_cols + j] =
                            current_val - factor.clone() * pivot_val;
                    }
                }
            }
        }

        Ok(ref_form)
    }

    /// Compute the rank of the matrix
    pub fn rank(&self) -> Result<usize> {
        let ref_form = self.row_echelon_form()?;
        Ok(ref_form.rank)
    }

    /// Solve the linear system `A·x = b` (column convention; Sage `solve_right`).
    ///
    /// * `Ok(Some(x))` iff the system is consistent. `x` is the particular solution read off
    ///   `RREF [A|b]` with every free variable set to 0; it is the unique solution when
    ///   `rank A = cols`. The full solution set is `x + span(self.kernel())`.
    /// * `Ok(None)` iff the system is inconsistent (`rank [A|b] = rank A + 1`).
    /// * `Err(InvalidArgument)` if `b.len() != rows`.
    ///
    /// Any shape is accepted, including `0×n` (every `x` solves; the zero vector is returned)
    /// and `m×0` (`Some(vec![])` iff `b = 0`). Zero tests are exact comparisons, so the answer
    /// is decided exactly over an exact field; no numerical robustness is claimed for
    /// floating-point `Field` impls.
    pub fn solve(&self, b: &[F]) -> Result<Option<Vec<F>>> {
        let m = self.rows();
        if b.len() != m {
            return Err(MathError::InvalidArgument(
                "Vector length must match number of rows".to_string(),
            ));
        }
        let bm = Matrix::from_vec(m, 1, b.to_vec())?;
        // X is n×1: its data is the solution vector.
        Ok(self.solve_columns(&bm)?.map(|x| x.data().to_vec()))
    }

    /// The shared body of [`Matrix::solve`] and [`Matrix::solve_matrix`]: `A·X = B` for an m×k
    /// matrix `B` (`b.rows() == self.rows()` is the caller's check) by one Gauss–Jordan
    /// elimination of `[A | B]`. `Ok(None)` iff some column of `B` is inconsistent; otherwise
    /// `X` (n×k) with every free variable 0 (module docs).
    fn solve_columns(&self, b: &Matrix<F>) -> Result<Option<Matrix<F>>> {
        let (m, n, k) = (self.rows(), self.cols(), b.cols());
        // Augmented matrix [A | B] (m × (n+k)).
        let w = n + k;
        let mut aug_data = Vec::with_capacity(m * w);
        for i in 0..m {
            aug_data.extend_from_slice(&self.data()[i * n..(i + 1) * n]);
            aug_data.extend_from_slice(&b.data()[i * k..(i + 1) * k]);
        }
        let augmented = Matrix::from_vec(m, w, aug_data)?;
        let rref = augmented.reduced_row_echelon_form()?;

        // Pivots are strictly increasing, so some B column (index >= n) is a pivot column iff
        // the last pivot is one. That happens iff rank [A|b_j] > rank A for some j:
        // inconsistent.
        if rref.pivots.last().is_some_and(|&c| c >= n) {
            return Ok(None);
        }

        // Pivot row r reads x_c + Σ_{free f} R[r][f]·x_f = R[r][n+j] for column j (pivots are
        // normalised to 1 by `row_echelon_form`). With every free x_f = 0 this gives
        // x_c = R[r][n+j]. Rows of index >= rank are zero rows, including their B entries,
        // because no B column is a pivot column. `F::zero()` for the free variables keeps
        // `solve` identical to its earlier one-column implementation.
        let mut x = vec![F::zero(); n * k];
        for (r, &c) in rref.pivots.iter().enumerate() {
            for j in 0..k {
                x[c * k + j] = rref.matrix.data()[r * w + n + j].clone();
            }
        }
        Ok(Some(Matrix::from_vec(n, k, x)?))
    }

    /// Solve `A·X = B` for an m×k matrix `B` (column convention; Sage `solve_right` with a
    /// matrix), by one elimination of `[A | B]` (module docs).
    ///
    /// * `Ok(Some(X))`, `X` n×k, iff every column of `B` is consistent. Column j of `X` is what
    ///   [`Matrix::solve`] returns for column j of `B` (every free variable 0); the full
    ///   solution set is `X` plus any n×k matrix with columns in `span(self.kernel())`.
    /// * `Ok(None)` iff some column of `B` is inconsistent.
    /// * `Err(InvalidArgument)` if `b.rows() != self.rows()`.
    ///
    /// Any shape is accepted (k = 0 gives `Some` of an n×0 matrix). Zero tests are exact
    /// comparisons, so the answer is decided exactly over an exact field; no numerical
    /// robustness is claimed for floating-point `Field` impls.
    pub fn solve_matrix(&self, b: &Matrix<F>) -> Result<Option<Matrix<F>>> {
        if b.rows() != self.rows() {
            return Err(MathError::InvalidArgument(format!(
                "solve_matrix: B has {} rows, A has {}",
                b.rows(),
                self.rows()
            )));
        }
        self.solve_columns(b)
    }

    /// Solve the left linear system `x·A = b` (row convention; equivalently `Aᵀ·xᵀ = bᵀ`).
    ///
    /// MAGMA `IsConsistent(A, w)` (Handbook ch. 26 §26.8) is `A.solve_left(w)?.is_some()`, and
    /// its second return value is the returned `x`. Semantics as in [`Matrix::solve`]:
    /// `Ok(None)` iff inconsistent, otherwise a particular solution with the free variables 0.
    /// `Err(InvalidArgument)` if `b.len() != cols`. For many right-hand sides use the matrix
    /// form [`Matrix::solve_left_matrix`] or the sequence form [`Matrix::solve_left_sequence`]:
    /// they eliminate once for all of them.
    pub fn solve_left(&self, b: &[F]) -> Result<Option<Vec<F>>> {
        if b.len() != self.cols() {
            return Err(MathError::InvalidArgument(
                "Vector length must match number of columns".to_string(),
            ));
        }

        // Transpose and solve
        let at = self.transpose();
        at.solve(b)
    }

    /// MAGMA `Solution(A, w)` and the extra return values of `IsConsistent(A, w)` (Handbook
    /// ch. 26 §26.8): `Ok(Some((v, N)))` with `v·A = w` (the particular solution of
    /// [`Matrix::solve_left`]) and `N` a basis of the left nullspace `{u : u·A = 0}`, so that
    /// the solution set is `v + span(N)`.
    ///
    /// `Ok(None)` iff the system is inconsistent (MAGMA raises an error there; this returns the
    /// documented `None`). `Err(InvalidArgument)` if `b.len() != cols`.
    #[allow(clippy::type_complexity)]
    pub fn solve_left_with_kernel(&self, b: &[F]) -> Result<Option<(Vec<F>, Vec<Vec<F>>)>> {
        match self.solve_left(b)? {
            None => Ok(None),
            Some(v) => {
                let kernel = self.transpose().kernel()?;
                Ok(Some((v, kernel)))
            }
        }
    }

    /// MAGMA `IsConsistent(A, W)` with an r×n matrix `W` (Handbook ch. 26 §26.8): `V·A = W`,
    /// by one elimination of `[Aᵀ | Wᵀ]` (module docs).
    ///
    /// * `Ok(Some(V))`, `V` r×m, iff every row of `W` is consistent. Row i of `V` is what
    ///   [`Matrix::solve_left`] returns for row i of `W` (every free variable 0).
    /// * `Ok(None)` iff some row of `W` is inconsistent.
    /// * `Err(InvalidArgument)` if `w.cols() != self.cols()`.
    ///
    /// Any shape is accepted: r = 0 gives `Some` of a 0×m matrix; m = 0 (A has no rows) gives
    /// `Some` of an r×0 matrix iff `W = 0`; n = 0 gives the r×m zero matrix.
    pub fn solve_left_matrix(&self, w: &Matrix<F>) -> Result<Option<Matrix<F>>> {
        if w.cols() != self.cols() {
            return Err(MathError::InvalidArgument(format!(
                "solve_left_matrix: W has {} columns, A has {}",
                w.cols(),
                self.cols()
            )));
        }
        // V·A = W iff Aᵀ·Vᵀ = Wᵀ.
        Ok(self
            .transpose()
            .solve_columns(&w.transpose())?
            .map(|x| x.transpose()))
    }

    /// MAGMA `Solution(A, W)` and the extra return values of `IsConsistent(A, W)` (Handbook
    /// ch. 26 §26.8): `Ok(Some((V, N)))` with `V·A = W` (the particular solution of
    /// [`Matrix::solve_left_matrix`]) and `N` a basis of the left nullspace `{u : u·A = 0}`, so
    /// that every solution is `V` plus a matrix whose rows lie in `span(N)`.
    ///
    /// `Ok(None)` iff some row is inconsistent (MAGMA raises an error there; this returns the
    /// documented `None`, as [`Matrix::solve_left_with_kernel`] does).
    /// `Err(InvalidArgument)` if `w.cols() != self.cols()`.
    #[allow(clippy::type_complexity)]
    pub fn solve_left_matrix_with_kernel(
        &self,
        w: &Matrix<F>,
    ) -> Result<Option<(Matrix<F>, Vec<Vec<F>>)>> {
        match self.solve_left_matrix(w)? {
            None => Ok(None),
            Some(v) => Ok(Some((v, self.transpose().kernel()?))),
        }
    }

    /// MAGMA `IsConsistent(A, Q)` for a sequence `Q` of vectors of length n (Handbook ch. 26
    /// §26.8): `Ok(Some(V))` with `V[i]·A = Q[i]` for every i, iff all are consistent;
    /// `Ok(None)` otherwise. `V[i]` is what [`Matrix::solve_left`] returns for `Q[i]`. One
    /// elimination for the whole sequence ([`Matrix::solve_left_matrix`] on the matrix with
    /// rows `Q[i]`).
    ///
    /// `Q` empty gives `Ok(Some(vec![]))`. `Err(InvalidArgument)` naming the index if some
    /// `Q[i].len() != self.cols()`.
    pub fn solve_left_sequence(&self, q: &[Vec<F>]) -> Result<Option<Vec<Vec<F>>>> {
        let n = self.cols();
        if let Some(i) = q.iter().position(|qi| qi.len() != n) {
            return Err(MathError::InvalidArgument(format!(
                "solve_left_sequence: Q[{i}] has length {}, A has {n} columns",
                q[i].len()
            )));
        }
        let w = Matrix::from_vec(q.len(), n, q.iter().flatten().cloned().collect())?;
        Ok(self.solve_left_matrix(&w)?.map(|v| {
            let m = v.cols();
            (0..v.rows())
                .map(|i| v.data()[i * m..(i + 1) * m].to_vec())
                .collect()
        }))
    }

    /// MAGMA `Solution(A, Q)` and the extra return values of `IsConsistent(A, Q)` (Handbook
    /// ch. 26 §26.8): `Ok(Some((V, N)))` with `V[i]·A = Q[i]` for every i (as
    /// [`Matrix::solve_left_sequence`]) and `N` a basis of the left nullspace `{u : u·A = 0}`.
    /// `Ok(None)` iff some `Q[i]` is inconsistent (MAGMA raises an error there). Errors as for
    /// [`Matrix::solve_left_sequence`].
    #[allow(clippy::type_complexity)]
    pub fn solve_left_sequence_with_kernel(
        &self,
        q: &[Vec<F>],
    ) -> Result<Option<(Vec<Vec<F>>, Vec<Vec<F>>)>> {
        match self.solve_left_sequence(q)? {
            None => Ok(None),
            Some(v) => Ok(Some((v, self.transpose().kernel()?))),
        }
    }

    /// Compute the kernel (null space) of the matrix
    ///
    /// Returns a basis for the kernel as a vector of column vectors.
    /// The kernel consists of all vectors x such that Ax = 0.
    pub fn kernel(&self) -> Result<Vec<Vec<F>>> {
        let rref = self.reduced_row_echelon_form()?;

        // Identify free variables (columns without pivots)
        let mut free_vars = Vec::new();
        let pivot_cols: std::collections::HashSet<usize> = rref.pivots.iter().copied().collect();

        for col in 0..self.cols() {
            if !pivot_cols.contains(&col) {
                free_vars.push(col);
            }
        }

        // If no free variables, kernel is trivial (just zero vector)
        if free_vars.is_empty() {
            return Ok(Vec::new());
        }

        // Build basis vectors for kernel
        let mut basis = Vec::new();

        for &free_var in &free_vars {
            let mut kernel_vec = vec![F::zero(); self.cols()];
            kernel_vec[free_var] = F::one();

            // For each pivot variable, set it based on the free variable
            for (pivot_row, &pivot_col) in rref.pivots.iter().enumerate() {
                // The value in the rref at (pivot_row, free_var) tells us
                // how much of the pivot variable to use
                let coeff = rref.matrix.get(pivot_row, free_var)?;
                kernel_vec[pivot_col] = F::zero() - coeff.clone();
            }

            basis.push(kernel_vec);
        }

        Ok(basis)
    }

    /// Compute the image (column space) of the matrix
    ///
    /// Returns a basis for the column space as a vector of column vectors.
    /// The image consists of all vectors b such that Ax = b has a solution.
    pub fn image(&self) -> Result<Vec<Vec<F>>> {
        let rref = self.reduced_row_echelon_form()?;

        // The pivot columns of the original matrix form a basis for the column space
        let mut basis = Vec::new();

        for &pivot_col in &rref.pivots {
            let mut col_vec = Vec::with_capacity(self.rows());
            for row in 0..self.rows() {
                col_vec.push(self.get(row, pivot_col)?.clone());
            }
            basis.push(col_vec);
        }

        Ok(basis)
    }

    /// Compute the Moore-Penrose pseudoinverse
    ///
    /// For a matrix A, the pseudoinverse A⁺ satisfies:
    /// - AA⁺A = A
    /// - A⁺AA⁺ = A⁺
    /// - (AA⁺)^T = AA⁺
    /// - (A⁺A)^T = A⁺A
    ///
    /// For full-rank matrices, this uses A⁺ = (A^T A)^{-1} A^T.
    /// Returns None if the computation fails.
    pub fn pseudoinverse(&self) -> Result<Option<Self>>
    where
        F: rustmath_core::NumericConversion,
    {
        let m = self.rows();
        let n = self.cols();

        // For full-rank case (rank = min(m,n))
        if m >= n {
            // Left pseudoinverse: A⁺ = (A^T A)^{-1} A^T
            let at = self.transpose();
            let ata = (at.clone() * self.clone())?;

            // Try to invert A^T A
            if let Some(ata_inv) = ata.inverse()? {
                let pinv = (ata_inv * at)?;
                return Ok(Some(pinv));
            }
        } else {
            // Right pseudoinverse: A⁺ = A^T (A A^T)^{-1}
            let at = self.transpose();
            let aat = (self.clone() * at.clone())?;

            // Try to invert A A^T
            if let Some(aat_inv) = aat.inverse()? {
                let pinv = (at * aat_inv)?;
                return Ok(Some(pinv));
            }
        }

        // If we can't compute using normal equations, return None
        Ok(None)
    }

    /// Compute the inverse of a square matrix
    ///
    /// Returns None if the matrix is singular (non-invertible).
    /// Uses Gauss-Jordan elimination: [A | I] -> [I | A^{-1}]
    pub fn inverse(&self) -> Result<Option<Self>> {
        if !self.is_square() {
            return Err(MathError::InvalidArgument(
                "Only square matrices can be inverted".to_string(),
            ));
        }

        let n = self.rows();

        // Create augmented matrix [A | I]
        let mut aug_data = Vec::with_capacity(n * (2 * n));
        for i in 0..n {
            for j in 0..n {
                aug_data.push(self.data()[i * n + j].clone());
            }
            // Add identity matrix columns
            for j in 0..n {
                aug_data.push(if i == j { F::one() } else { F::zero() });
            }
        }

        let mut augmented = Matrix::from_vec(n, 2 * n, aug_data)?;

        // Apply Gauss-Jordan elimination
        for col in 0..n {
            // Find pivot
            let mut pivot_row = col;
            let mut max_val = augmented.data()[col * augmented.cols() + col].clone();

            for row in (col + 1)..n {
                let val = augmented.data()[row * augmented.cols() + col].clone();
                if !val.is_zero() && max_val.is_zero() {
                    max_val = val;
                    pivot_row = row;
                }
            }

            // Check if matrix is singular
            if max_val.is_zero() {
                return Ok(None);
            }

            // Swap rows if needed
            let aug_cols = augmented.cols();
            if pivot_row != col {
                for j in 0..aug_cols {
                    let temp = augmented.data()[col * aug_cols + j].clone();
                    let swap_val = augmented.data()[pivot_row * aug_cols + j].clone();
                    augmented.data_mut()[col * aug_cols + j] = swap_val;
                    augmented.data_mut()[pivot_row * aug_cols + j] = temp;
                }
            }

            // Scale pivot row
            let pivot = augmented.data()[col * aug_cols + col].clone();
            for j in 0..aug_cols {
                let val = augmented.data()[col * aug_cols + j].clone();
                augmented.data_mut()[col * aug_cols + j] = val / pivot.clone();
            }

            // Eliminate column in all other rows
            for row in 0..n {
                if row == col {
                    continue;
                }
                let factor = augmented.data()[row * aug_cols + col].clone();
                for j in 0..aug_cols {
                    let pivot_val = augmented.data()[col * aug_cols + j].clone();
                    let current_val = augmented.data()[row * aug_cols + j].clone();
                    augmented.data_mut()[row * aug_cols + j] =
                        current_val - factor.clone() * pivot_val;
                }
            }
        }

        // Extract inverse from right half of augmented matrix
        let mut inv_data = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                inv_data.push(augmented.data()[i * augmented.cols() + n + j].clone());
            }
        }

        Ok(Some(Matrix::from_vec(n, n, inv_data)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_row_echelon_form() {
        // [1 2 3]
        // [2 4 5]
        // [3 5 6]
        let m = Matrix::from_vec(
            3,
            3,
            vec![
                Rational::from_integer(1),
                Rational::from_integer(2),
                Rational::from_integer(3),
                Rational::from_integer(2),
                Rational::from_integer(4),
                Rational::from_integer(5),
                Rational::from_integer(3),
                Rational::from_integer(5),
                Rational::from_integer(6),
            ],
        )
        .unwrap();

        let ref_form = m.row_echelon_form().unwrap();
        assert_eq!(ref_form.rank, 3);
    }

    #[test]
    fn test_rank() {
        // Rank 2 matrix
        // [1 2]
        // [2 4]
        let m = Matrix::from_vec(
            2,
            2,
            vec![
                Rational::from_integer(1),
                Rational::from_integer(2),
                Rational::from_integer(2),
                Rational::from_integer(4),
            ],
        )
        .unwrap();

        assert_eq!(m.rank().unwrap(), 1); // Second row is multiple of first
    }

    #[test]
    fn test_solve_simple() {
        // 2x + y = 5
        // x - y = 1
        // Solution: x = 2, y = 1
        let a = Matrix::from_vec(
            2,
            2,
            vec![
                Rational::from_integer(2),
                Rational::from_integer(1),
                Rational::from_integer(1),
                Rational::from_integer(-1),
            ],
        )
        .unwrap();

        let b = vec![Rational::from_integer(5), Rational::from_integer(1)];

        let solution = a.solve(&b).unwrap().expect("Should have solution");

        assert_eq!(solution[0], Rational::from_integer(2));
        assert_eq!(solution[1], Rational::from_integer(1));
    }

    #[test]
    fn test_inverse() {
        // [1 2]
        // [3 4]
        // Inverse: 1/-2 * [4 -2] = [-2   1  ]
        //                  [-3 1]   [3/2 -1/2]
        let m = Matrix::from_vec(
            2,
            2,
            vec![
                Rational::from_integer(1),
                Rational::from_integer(2),
                Rational::from_integer(3),
                Rational::from_integer(4),
            ],
        )
        .unwrap();

        let inv = m.inverse().unwrap().expect("Should be invertible");

        // Verify A * A^{-1} = I
        let product = (m.clone() * inv.clone()).unwrap();
        let identity = Matrix::identity(2);

        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(product.get(i, j).unwrap(), identity.get(i, j).unwrap());
            }
        }

        // Verify A^{-1} * A = I
        let product2 = (inv * m).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(product2.get(i, j).unwrap(), identity.get(i, j).unwrap());
            }
        }
    }

    #[test]
    fn test_inverse_singular() {
        // Singular matrix (second row is multiple of first)
        // [1 2]
        // [2 4]
        let m = Matrix::from_vec(
            2,
            2,
            vec![
                Rational::from_integer(1),
                Rational::from_integer(2),
                Rational::from_integer(2),
                Rational::from_integer(4),
            ],
        )
        .unwrap();

        assert!(m.inverse().unwrap().is_none());
    }

    #[test]
    fn test_kernel() {
        // Matrix with non-trivial kernel:
        // [1 2 3]
        // [2 4 6]
        // Second row is 2 * first row, so rank = 1, kernel dimension = 2
        let m = Matrix::from_vec(
            2,
            3,
            vec![
                Rational::from_integer(1),
                Rational::from_integer(2),
                Rational::from_integer(3),
                Rational::from_integer(2),
                Rational::from_integer(4),
                Rational::from_integer(6),
            ],
        )
        .unwrap();

        let kernel = m.kernel().unwrap();
        assert_eq!(kernel.len(), 2); // Kernel dimension should be 2

        // Verify each basis vector is in the kernel (Av = 0)
        for v in &kernel {
            assert_eq!(v.len(), 3);
            // Compute m * v
            let mut result = vec![Rational::from_integer(0); 2];
            for i in 0..2 {
                for j in 0..3 {
                    result[i] = result[i].clone() + m.get(i, j).unwrap().clone() * v[j].clone();
                }
                assert_eq!(result[i], Rational::from_integer(0));
            }
        }
    }

    #[test]
    fn test_image() {
        // [1 2 3]
        // [2 4 5]
        // [3 6 7]
        // Columns 1 and 3 are linearly independent, column 2 = 2*column 1
        let m = Matrix::from_vec(
            3,
            3,
            vec![
                Rational::from_integer(1),
                Rational::from_integer(2),
                Rational::from_integer(3),
                Rational::from_integer(2),
                Rational::from_integer(4),
                Rational::from_integer(5),
                Rational::from_integer(3),
                Rational::from_integer(6),
                Rational::from_integer(7),
            ],
        )
        .unwrap();

        let image = m.image().unwrap();
        // Rank should be 2 (columns 0 and 2 are basis)
        assert!(image.len() >= 2);
    }

    #[test]
    fn test_solve_left() {
        // Test xA = b where x is unknown
        // A = [1 2]    b = [5 11]
        //     [3 4]
        // We want x such that xA = b
        // x[1,3] = [1 2] should give [1*1+3*2, 1*3+3*4] = [7, 15], not matching
        //          [3 4]
        // x[1,2] = [1 2] gives [1*1+2*3, 1*2+2*4] = [7, 10], not matching
        //          [3 4]
        let a = Matrix::from_vec(
            2,
            2,
            vec![
                Rational::from_integer(1),
                Rational::from_integer(2),
                Rational::from_integer(3),
                Rational::from_integer(4),
            ],
        )
        .unwrap();

        let b = vec![Rational::from_integer(7), Rational::from_integer(10)];

        let solution = a.solve_left(&b).unwrap();
        if let Some(x) = solution {
            // Verify xA = b
            let mut result = vec![Rational::from_integer(0); 2];
            for j in 0..2 {
                for i in 0..2 {
                    result[j] = result[j].clone() + x[i].clone() * a.get(i, j).unwrap().clone();
                }
            }
            assert_eq!(result[0], b[0]);
            assert_eq!(result[1], b[1]);
        }
    }

    // Reference values: PARI/GP 2.15.4. `matinverseimage(A, b)` solves A*X = b and returns
    // `[]~` when there is none; on these inputs its answer is exactly the particular solution
    // with the free variables 0. `matker(A)` is the right kernel.

    fn q(v: i64) -> Rational {
        Rational::from_integer(v)
    }

    fn qv(v: &[i64]) -> Vec<Rational> {
        v.iter().map(|&x| q(x)).collect()
    }

    fn qm(rows: usize, cols: usize, v: &[i64]) -> Matrix<Rational> {
        Matrix::from_vec(rows, cols, qv(v)).unwrap()
    }

    /// A·x for a column vector x.
    fn mat_vec(a: &Matrix<Rational>, x: &[Rational]) -> Vec<Rational> {
        (0..a.rows())
            .map(|i| {
                (0..a.cols()).fold(q(0), |acc, j| {
                    acc + a.get(i, j).unwrap().clone() * x[j].clone()
                })
            })
            .collect()
    }

    /// x·A for a row vector x.
    fn vec_mat(x: &[Rational], a: &Matrix<Rational>) -> Vec<Rational> {
        (0..a.cols())
            .map(|j| {
                (0..a.rows()).fold(q(0), |acc, i| {
                    acc + x[i].clone() * a.get(i, j).unwrap().clone()
                })
            })
            .collect()
    }

    #[test]
    fn solve_inconsistent_singular_square() {
        // The ch026.Solution / ch026.IsConsistent defect: this used to return Some([0, 1]).
        // gp: matrank([1,1;1,1]) = 1, matrank(concat([1,1;1,1],[1,2]~)) = 2,
        //     matinverseimage([1,1;1,1],[1,2]~) = []~
        let a = qm(2, 2, &[1, 1, 1, 1]);
        assert_eq!(a.solve(&qv(&[1, 2])).unwrap(), None);
    }

    #[test]
    fn solve_left_inconsistent() {
        // gp: matinverseimage([1,1;1,1]~,[1,2]~) = []~
        let a = qm(2, 2, &[1, 1, 1, 1]);
        assert_eq!(a.solve_left(&qv(&[1, 2])).unwrap(), None);
        assert_eq!(a.solve_left_with_kernel(&qv(&[1, 2])).unwrap(), None);
    }

    #[test]
    fn solve_consistent_singular() {
        // gp: matinverseimage([1,1;1,1],[1,1]~) = [1, 0]~, matker([1,1;1,1]) = [-1; 1]
        let a = qm(2, 2, &[1, 1, 1, 1]);
        let b = qv(&[1, 1]);
        let x = a.solve(&b).unwrap().expect("consistent");
        assert_eq!(x, qv(&[1, 0]));
        assert_eq!(mat_vec(&a, &x), b);
    }

    #[test]
    fn solve_overdetermined() {
        // gp: matinverseimage([1,2;3,4;5,6],[5,11,17]~) = [1, 2]~;
        //     matinverseimage([1,2;3,4;5,6],[5,11,18]~) = []~ (matrank of [A|b] = 3)
        let a = qm(3, 2, &[1, 2, 3, 4, 5, 6]);
        let b = qv(&[5, 11, 17]);
        let x = a.solve(&b).unwrap().expect("consistent");
        assert_eq!(x, qv(&[1, 2]));
        assert_eq!(mat_vec(&a, &x), b);
        assert_eq!(a.solve(&qv(&[5, 11, 18])).unwrap(), None);
    }

    #[test]
    fn solve_underdetermined() {
        // gp: matinverseimage([1,2,3;4,5,6],[6,15]~) = [0, 3, 0]~,
        //     matker([1,2,3;4,5,6]) = [1; -2; 1]
        let a = qm(2, 3, &[1, 2, 3, 4, 5, 6]);
        let b = qv(&[6, 15]);
        let x = a.solve(&b).unwrap().expect("consistent");
        assert_eq!(x, qv(&[0, 3, 0]));
        assert_eq!(mat_vec(&a, &x), b);
        assert_eq!(a.kernel().unwrap(), vec![qv(&[1, -2, 1])]);
    }

    #[test]
    fn solve_rank2_3x3() {
        // gp: matrank([1,2,3;4,5,6;7,8,9]) = 2; matinverseimage(A,[1,0,0]~) = []~
        //     (matrank of [A|b] = 3); matinverseimage(A,[6,15,24]~) = [0, 3, 0]~
        let a = qm(3, 3, &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(a.solve(&qv(&[1, 0, 0])).unwrap(), None);
        let b = qv(&[6, 15, 24]);
        let x = a.solve(&b).unwrap().expect("consistent");
        assert_eq!(x, qv(&[0, 3, 0]));
        assert_eq!(mat_vec(&a, &x), b);
    }

    #[test]
    fn solve_left_unique() {
        // gp: matsolve([1,2;3,4]~,[7,10]~) = [1, 2]~
        let a = qm(2, 2, &[1, 2, 3, 4]);
        let b = qv(&[7, 10]);
        let x = a.solve_left(&b).unwrap().expect("consistent");
        assert_eq!(x, qv(&[1, 2]));
        assert_eq!(vec_mat(&x, &a), b);
    }

    #[test]
    fn solve_left_with_kernel_basic() {
        // gp: matinverseimage([1,1;1,1]~,[2,2]~) = [2, 0]~, matker([1,1;1,1]~) = [-1; 1]
        let a = qm(2, 2, &[1, 1, 1, 1]);
        let b = qv(&[2, 2]);
        let (v, n) = a.solve_left_with_kernel(&b).unwrap().expect("consistent");
        assert_eq!(v, qv(&[2, 0]));
        assert_eq!(vec_mat(&v, &a), b);
        assert_eq!(n, vec![qv(&[-1, 1])]);
        for u in &n {
            assert_eq!(vec_mat(u, &a), qv(&[0, 0]));
        }
    }

    #[test]
    fn solve_left_with_kernel_3x2() {
        // gp: matinverseimage([1,2;3,4;5,6]~,[9,12]~) = [0, 3, 0]~,
        //     matker([1,2;3,4;5,6]~) = [1; -2; 1], [0,3,0]*[1,2;3,4;5,6] = [9, 12]
        let a = qm(3, 2, &[1, 2, 3, 4, 5, 6]);
        let b = qv(&[9, 12]);
        let (v, n) = a.solve_left_with_kernel(&b).unwrap().expect("consistent");
        assert_eq!(v, qv(&[0, 3, 0]));
        assert_eq!(vec_mat(&v, &a), b);
        assert_eq!(n, vec![qv(&[1, -2, 1])]);
        for u in &n {
            assert_eq!(vec_mat(u, &a), qv(&[0, 0]));
        }
    }

    #[test]
    fn solve_empty_dims() {
        // Definitional: a 0×2 system is solved by every x; a 2×0 system only when b = 0.
        let a = Matrix::<Rational>::from_vec(0, 2, vec![]).unwrap();
        assert_eq!(a.solve(&[]).unwrap(), Some(qv(&[0, 0])));
        let b = Matrix::<Rational>::from_vec(2, 0, vec![]).unwrap();
        assert_eq!(b.solve(&qv(&[0, 0])).unwrap(), Some(vec![]));
        assert_eq!(b.solve(&qv(&[1, 0])).unwrap(), None);
        assert!(b.solve(&qv(&[1])).is_err());
    }

    // ---------------------------------------------------------------------------------------
    // Multi-RHS forms (MAGMA IsConsistent(A, W), IsConsistent(A, Q), Solution(A, W/Q)).
    // ---------------------------------------------------------------------------------------

    /// The rows of a matrix.
    fn rows_of(m: &Matrix<Rational>) -> Vec<Vec<Rational>> {
        (0..m.rows()).map(|i| m.row(i).unwrap()).collect()
    }

    #[test]
    fn solve_left_matrix_two_rhs() {
        // gp: A = [1,2;2,4;0,1]; W = [3,7;1,3]; matinverseimage(A~, W~)~ = [3,0,1; 1,0,1],
        //     [3,0,1;1,0,1]*A = [3,7; 1,3]
        let a = qm(3, 2, &[1, 2, 2, 4, 0, 1]);
        let w = qm(2, 2, &[3, 7, 1, 3]);
        let v = a.solve_left_matrix(&w).unwrap().expect("consistent");
        assert_eq!(v, qm(2, 3, &[3, 0, 1, 1, 0, 1]));
        assert_eq!((v.clone() * a.clone()).unwrap(), w);
        // Row i is what solve_left returns for row i.
        for (vi, wi) in rows_of(&v).iter().zip(rows_of(&w)) {
            assert_eq!(a.solve_left(&wi).unwrap().as_ref(), Some(vi));
        }
    }

    #[test]
    fn solve_left_matrix_with_kernel_two_rhs() {
        // gp: matker([1,2;2,4;0,1]~) = [-2; 1; 0];
        //     ([3,0,1;1,0,1] + [-2,1,0;-2,1,0])*A = [3,7; 1,3]
        let a = qm(3, 2, &[1, 2, 2, 4, 0, 1]);
        let w = qm(2, 2, &[3, 7, 1, 3]);
        let (v, n) = a
            .solve_left_matrix_with_kernel(&w)
            .unwrap()
            .expect("consistent");
        assert_eq!(v, qm(2, 3, &[3, 0, 1, 1, 0, 1]));
        assert_eq!(n, vec![qv(&[-2, 1, 0])]);
        for (vi, wi) in rows_of(&v).iter().zip(rows_of(&w)) {
            let shifted: Vec<Rational> = vi
                .iter()
                .zip(&n[0])
                .map(|(x, y)| x.clone() + y.clone())
                .collect();
            assert_eq!(vec_mat(&shifted, &a), wi);
        }
    }

    #[test]
    fn solve_left_matrix_one_bad_row() {
        // gp: matinverseimage([1,2;2,4]~, [3,6;1,3]~) = [;] (inconsistent);
        //     matinverseimage([1,2;2,4]~, [3,6]~) = [3, 0]~;
        //     matinverseimage([1,2;2,4]~, [1,3]~) = []~
        let a = qm(2, 2, &[1, 2, 2, 4]);
        let w = qm(2, 2, &[3, 6, 1, 3]);
        assert_eq!(a.solve_left_matrix(&w).unwrap(), None);
        assert_eq!(a.solve_left_matrix_with_kernel(&w).unwrap(), None);
        assert_eq!(a.solve_left(&qv(&[3, 6])).unwrap(), Some(qv(&[3, 0])));
        assert_eq!(a.solve_left(&qv(&[1, 3])).unwrap(), None);
    }

    #[test]
    fn solve_left_matrix_unique() {
        // gp: A = [2,1,1;1,3,2;1,0,0]; matdet(A) = -1; W = [4,5,3;1,1,1];
        //     W*A^-1 = [-1,2,4; 1,0,-1] = matinverseimage(A~, W~)~
        let a = qm(3, 3, &[2, 1, 1, 1, 3, 2, 1, 0, 0]);
        let w = qm(2, 3, &[4, 5, 3, 1, 1, 1]);
        let (v, n) = a
            .solve_left_matrix_with_kernel(&w)
            .unwrap()
            .expect("consistent");
        assert_eq!(v, qm(2, 3, &[-1, 2, 4, 1, 0, -1]));
        assert!(n.is_empty());
    }

    #[test]
    fn solve_matrix_column_convention() {
        // gp: matinverseimage([1,2,0;2,4,1], [3,1;7,3]) = [3,1; 0,0; 1,1]
        let a = qm(2, 3, &[1, 2, 0, 2, 4, 1]);
        let b = qm(2, 2, &[3, 1, 7, 3]);
        let x = a.solve_matrix(&b).unwrap().expect("consistent");
        assert_eq!(x, qm(3, 2, &[3, 1, 0, 0, 1, 1]));
        assert_eq!((a.clone() * x.clone()).unwrap(), b);
        // Column j is what solve returns for column j of B.
        assert_eq!(a.solve(&qv(&[3, 7])).unwrap(), Some(qv(&[3, 0, 1])));
        assert_eq!(a.solve(&qv(&[1, 3])).unwrap(), Some(qv(&[1, 0, 1])));
        assert!(a.solve_matrix(&qm(3, 1, &[1, 2, 3])).is_err());
        // gp: matrank([1,1;1,1]) = 1, matinverseimage([1,1;1,1], [1,1;1,2]) = [;]
        let s = qm(2, 2, &[1, 1, 1, 1]);
        assert_eq!(s.solve_matrix(&qm(2, 2, &[1, 1, 1, 2])).unwrap(), None);
    }

    #[test]
    fn solve_left_sequence_matches_rows() {
        // Same systems as solve_left_matrix_two_rhs and solve_left_matrix_one_bad_row.
        let a = qm(3, 2, &[1, 2, 2, 4, 0, 1]);
        let q_seq = vec![qv(&[3, 7]), qv(&[1, 3])];
        let v = a.solve_left_sequence(&q_seq).unwrap().expect("consistent");
        assert_eq!(v, vec![qv(&[3, 0, 1]), qv(&[1, 0, 1])]);
        let (v2, n) = a
            .solve_left_sequence_with_kernel(&q_seq)
            .unwrap()
            .expect("consistent");
        assert_eq!(v2, v);
        assert_eq!(n, vec![qv(&[-2, 1, 0])]);
        let b = qm(2, 2, &[1, 2, 2, 4]);
        assert_eq!(
            b.solve_left_sequence(&[qv(&[3, 6]), qv(&[1, 3])]).unwrap(),
            None
        );
        assert_eq!(
            b.solve_left_sequence_with_kernel(&[qv(&[3, 6]), qv(&[1, 3])])
                .unwrap(),
            None
        );
        // Empty sequence; a vector of the wrong length (index 1).
        assert_eq!(a.solve_left_sequence(&[]).unwrap(), Some(vec![]));
        match a.solve_left_sequence(&[qv(&[3, 7]), qv(&[1])]) {
            Err(MathError::InvalidArgument(msg)) => assert!(msg.contains("Q[1]"), "{msg}"),
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn solve_left_matrix_shapes() {
        // Definitional cases (no reference system needed).
        // W with 2 columns for a 3-column A.
        let a3 = qm(2, 3, &[1, 0, 0, 0, 1, 0]);
        assert!(matches!(
            a3.solve_left_matrix(&qm(1, 2, &[1, 1])),
            Err(MathError::InvalidArgument(_))
        ));
        // A 3×2, W 0×2: V is 0×3.
        let a = qm(3, 2, &[1, 2, 2, 4, 0, 1]);
        let w0 = Matrix::<Rational>::from_vec(0, 2, vec![]).unwrap();
        let v = a.solve_left_matrix(&w0).unwrap().expect("consistent");
        assert_eq!((v.rows(), v.cols()), (0, 3));
        // A 0×2 (m = 0): V is r×0, consistent iff W = 0.
        let a0 = Matrix::<Rational>::from_vec(0, 2, vec![]).unwrap();
        let v = a0
            .solve_left_matrix(&qm(1, 2, &[0, 0]))
            .unwrap()
            .expect("consistent");
        assert_eq!((v.rows(), v.cols()), (1, 0));
        assert_eq!(a0.solve_left_matrix(&qm(1, 2, &[0, 1])).unwrap(), None);
        // A 2×0 (n = 0), W 3×0: V is the 3×2 zero matrix, the left nullspace is all of Q^2.
        let an = Matrix::<Rational>::from_vec(2, 0, vec![]).unwrap();
        let w3 = Matrix::<Rational>::from_vec(3, 0, vec![]).unwrap();
        let (v, n) = an
            .solve_left_matrix_with_kernel(&w3)
            .unwrap()
            .expect("consistent");
        assert_eq!(v, qm(3, 2, &[0, 0, 0, 0, 0, 0]));
        assert_eq!(n.len(), 2);
    }
}
