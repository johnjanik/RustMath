//! Linear system solving algorithms
//!
//! MAGMA Handbook ch. 26 §26.8 (digest `rustmath-lava/docs/Chapter_26_Matrices.md`):
//! `IsConsistent(A, w)` and `Solution(A, w)` use the row convention `v·A = w`; here they are
//! [`Matrix::solve_left`] and [`Matrix::solve_left_with_kernel`]. [`Matrix::solve`] is the
//! column convention `A·x = b` (Sage `solve_right`).
//!
//! Algorithm: Gauss–Jordan elimination on `[A | b]` and the Rouché–Capelli criterion
//! (K. Hoffman and R. Kunze, *Linear Algebra*, 2nd ed., §1.4, row-reduced echelon matrices and
//! the solution of `A X = Y`): the system is consistent iff `rank [A|b] = rank A`, i.e. iff the
//! `b` column is not a pivot column of `RREF [A|b]`. The particular solution returned sets every
//! free (non-pivot) variable to 0.

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
                    result.data_mut()[row * cols + j] =
                        current_val - factor.clone() * pivot_val;
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
                        let pivot_val =
                            ref_form.matrix.data()[pivot_row * matrix_cols + j].clone();
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
        let (m, n) = (self.rows(), self.cols());
        if b.len() != m {
            return Err(MathError::InvalidArgument(
                "Vector length must match number of rows".to_string(),
            ));
        }

        // Create augmented matrix [A | b] (m × (n+1)).
        let mut aug_data = Vec::with_capacity(m * (n + 1));
        for (i, bi) in b.iter().enumerate() {
            aug_data.extend_from_slice(&self.data()[i * n..(i + 1) * n]);
            aug_data.push(bi.clone());
        }
        let augmented = Matrix::from_vec(m, n + 1, aug_data)?;
        let rref = augmented.reduced_row_echelon_form()?;

        // Pivots are strictly increasing, so the b column (index n) is a pivot column iff it
        // is the last pivot. That happens iff rank [A|b] > rank A: inconsistent.
        if rref.pivots.last() == Some(&n) {
            return Ok(None);
        }

        // Pivot row r reads x_c + Σ_{free f} R[r][f]·x_f = R[r][n] (pivots are normalised to 1
        // by `row_echelon_form`). With every free x_f = 0 this gives x_c = R[r][n]. Rows of
        // index >= rank are zero rows, including their b entry, because the b column is not a
        // pivot column.
        let w = n + 1;
        let mut x = vec![F::zero(); n];
        for (r, &c) in rref.pivots.iter().enumerate() {
            x[c] = rref.matrix.data()[r * w + n].clone();
        }
        Ok(Some(x))
    }

    /// Solve the left linear system `x·A = b` (row convention; equivalently `Aᵀ·xᵀ = bᵀ`).
    ///
    /// MAGMA `IsConsistent(A, w)` (Handbook ch. 26 §26.8) is `A.solve_left(w)?.is_some()`, and
    /// its second return value is the returned `x`. Semantics as in [`Matrix::solve`]:
    /// `Ok(None)` iff inconsistent, otherwise a particular solution with the free variables 0.
    /// `Err(InvalidArgument)` if `b.len() != cols`.
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

    /// Compute the kernel (null space) of the matrix
    ///
    /// Returns a basis for the kernel as a vector of column vectors.
    /// The kernel consists of all vectors x such that Ax = 0.
    pub fn kernel(&self) -> Result<Vec<Vec<F>>> {
        let rref = self.reduced_row_echelon_form()?;

        // Identify free variables (columns without pivots)
        let mut free_vars = Vec::new();
        let pivot_cols: std::collections::HashSet<usize> =
            rref.pivots.iter().copied().collect();

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
}
