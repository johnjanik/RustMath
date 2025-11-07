//! Linear system solving algorithms

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

        for col in 0..result.cols {
            if current_row >= result.rows {
                break;
            }

            // Find pivot (largest absolute value in column, starting from current_row)
            let mut pivot_row = current_row;
            let mut max_val = result.data[current_row * result.cols + col].clone();

            for row in (current_row + 1)..result.rows {
                let val = result.data[row * result.cols + col].clone();
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
                for j in 0..result.cols {
                    let temp = result.data[current_row * result.cols + j].clone();
                    result.data[current_row * result.cols + j] =
                        result.data[pivot_row * result.cols + j].clone();
                    result.data[pivot_row * result.cols + j] = temp;
                }
            }

            pivots.push(col);

            // Scale pivot row
            let pivot = result.data[current_row * result.cols + col].clone();
            if pivot.is_zero() {
                return Err(MathError::DivisionByZero);
            }

            for j in col..result.cols {
                let val = result.data[current_row * result.cols + j].clone();
                result.data[current_row * result.cols + j] = val / pivot.clone();
            }

            // Eliminate column below pivot
            for row in (current_row + 1)..result.rows {
                let factor = result.data[row * result.cols + col].clone();
                for j in col..result.cols {
                    let pivot_val = result.data[current_row * result.cols + j].clone();
                    let current_val = result.data[row * result.cols + j].clone();
                    result.data[row * result.cols + j] =
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
            for row in 0..pivot_row {
                let factor = ref_form.matrix.data[row * ref_form.matrix.cols + pivot_col].clone();
                if !factor.is_zero() {
                    for j in 0..ref_form.matrix.cols {
                        let pivot_val =
                            ref_form.matrix.data[pivot_row * ref_form.matrix.cols + j].clone();
                        let current_val = ref_form.matrix.data[row * ref_form.matrix.cols + j].clone();
                        ref_form.matrix.data[row * ref_form.matrix.cols + j] =
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

    /// Solve the linear system Ax = b
    ///
    /// Returns None if the system has no solution or infinitely many solutions.
    /// For a unique solution, the matrix must be square and non-singular.
    pub fn solve(&self, b: &[F]) -> Result<Option<Vec<F>>> {
        if b.len() != self.rows {
            return Err(MathError::InvalidArgument(
                "Vector length must match number of rows".to_string(),
            ));
        }

        // Create augmented matrix [A | b]
        let mut aug_data = Vec::with_capacity(self.rows * (self.cols + 1));
        for i in 0..self.rows {
            for j in 0..self.cols {
                aug_data.push(self.data[i * self.cols + j].clone());
            }
            aug_data.push(b[i].clone());
        }

        let augmented = Matrix {
            data: aug_data,
            rows: self.rows,
            cols: self.cols + 1,
        };

        // Reduce to row echelon form
        let ref_form = augmented.reduced_row_echelon_form()?;

        // Check for inconsistency (row of form [0 0 ... 0 | non-zero])
        for i in ref_form.rank..self.rows {
            let b_val = ref_form.matrix.data[i * ref_form.matrix.cols + self.cols].clone();
            if !b_val.is_zero() {
                return Ok(None); // No solution
            }
        }

        // For now, only handle unique solution case (square matrix, full rank)
        if !self.is_square() || ref_form.rank != self.rows {
            return Ok(None); // Infinitely many solutions or no solution
        }

        // Extract solution from last column
        let mut solution = Vec::with_capacity(self.cols);
        for i in 0..self.cols {
            solution.push(ref_form.matrix.data[i * ref_form.matrix.cols + self.cols].clone());
        }

        Ok(Some(solution))
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

        let n = self.rows;

        // Create augmented matrix [A | I]
        let mut aug_data = Vec::with_capacity(n * (2 * n));
        for i in 0..n {
            for j in 0..n {
                aug_data.push(self.data[i * n + j].clone());
            }
            // Add identity matrix columns
            for j in 0..n {
                aug_data.push(if i == j { F::one() } else { F::zero() });
            }
        }

        let mut augmented = Matrix {
            data: aug_data,
            rows: n,
            cols: 2 * n,
        };

        // Apply Gauss-Jordan elimination
        for col in 0..n {
            // Find pivot
            let mut pivot_row = col;
            let mut max_val = augmented.data[col * augmented.cols + col].clone();

            for row in (col + 1)..n {
                let val = augmented.data[row * augmented.cols + col].clone();
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
            if pivot_row != col {
                for j in 0..augmented.cols {
                    let temp = augmented.data[col * augmented.cols + j].clone();
                    augmented.data[col * augmented.cols + j] =
                        augmented.data[pivot_row * augmented.cols + j].clone();
                    augmented.data[pivot_row * augmented.cols + j] = temp;
                }
            }

            // Scale pivot row
            let pivot = augmented.data[col * augmented.cols + col].clone();
            for j in 0..augmented.cols {
                let val = augmented.data[col * augmented.cols + j].clone();
                augmented.data[col * augmented.cols + j] = val / pivot.clone();
            }

            // Eliminate column in all other rows
            for row in 0..n {
                if row == col {
                    continue;
                }
                let factor = augmented.data[row * augmented.cols + col].clone();
                for j in 0..augmented.cols {
                    let pivot_val = augmented.data[col * augmented.cols + j].clone();
                    let current_val = augmented.data[row * augmented.cols + j].clone();
                    augmented.data[row * augmented.cols + j] =
                        current_val - factor.clone() * pivot_val;
                }
            }
        }

        // Extract inverse from right half of augmented matrix
        let mut inv_data = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                inv_data.push(augmented.data[i * augmented.cols + n + j].clone());
            }
        }

        Ok(Some(Matrix {
            data: inv_data,
            rows: n,
            cols: n,
        }))
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
                Rational::from((1, 1)),
                Rational::from((2, 1)),
                Rational::from((3, 1)),
                Rational::from((2, 1)),
                Rational::from((4, 1)),
                Rational::from((5, 1)),
                Rational::from((3, 1)),
                Rational::from((5, 1)),
                Rational::from((6, 1)),
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
                Rational::from((1, 1)),
                Rational::from((2, 1)),
                Rational::from((2, 1)),
                Rational::from((4, 1)),
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
                Rational::from((2, 1)),
                Rational::from((1, 1)),
                Rational::from((1, 1)),
                Rational::from((-1, 1)),
            ],
        )
        .unwrap();

        let b = vec![Rational::from((5, 1)), Rational::from((1, 1))];

        let solution = a.solve(&b).unwrap().expect("Should have solution");

        assert_eq!(solution[0], Rational::from((2, 1)));
        assert_eq!(solution[1], Rational::from((1, 1)));
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
                Rational::from((1, 1)),
                Rational::from((2, 1)),
                Rational::from((3, 1)),
                Rational::from((4, 1)),
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
                Rational::from((1, 1)),
                Rational::from((2, 1)),
                Rational::from((2, 1)),
                Rational::from((4, 1)),
            ],
        )
        .unwrap();

        assert!(m.inverse().unwrap().is_none());
    }
}
