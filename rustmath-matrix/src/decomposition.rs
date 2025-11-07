//! Matrix decompositions (LU, QR, etc.)

use crate::Matrix;
use rustmath_core::{Field, MathError, Result};

/// LU decomposition result
///
/// Represents A = LU where L is lower triangular with 1s on diagonal,
/// and U is upper triangular.
pub struct LUDecomposition<F: Field> {
    /// Lower triangular matrix with 1s on diagonal
    pub l: Matrix<F>,
    /// Upper triangular matrix
    pub u: Matrix<F>,
}

/// PLU decomposition result
///
/// Represents PA = LU where P is a permutation matrix
pub struct PLUDecomposition<F: Field> {
    /// Permutation matrix
    pub p: Matrix<F>,
    /// Lower triangular matrix with 1s on diagonal
    pub l: Matrix<F>,
    /// Upper triangular matrix
    pub u: Matrix<F>,
    /// Row permutation vector
    pub perm: Vec<usize>,
}

impl<F: Field> Matrix<F> {
    /// Compute LU decomposition without pivoting
    ///
    /// Uses Doolittle's algorithm. May fail if a zero pivot is encountered.
    /// For better numerical stability, use `plu_decomposition` instead.
    pub fn lu_decomposition(&self) -> Result<LUDecomposition<F>> {
        if !self.is_square() {
            return Err(MathError::InvalidArgument(
                "LU decomposition requires a square matrix".to_string(),
            ));
        }

        let n = self.rows;
        let mut l_data = vec![F::zero(); n * n];
        let mut u_data = vec![F::zero(); n * n];

        // Initialize L diagonal to 1
        for i in 0..n {
            l_data[i * n + i] = F::one();
        }

        // Doolittle's algorithm
        for j in 0..n {
            // Upper triangular
            for i in 0..=j {
                let mut sum = F::zero();
                for k in 0..i {
                    sum = sum + l_data[i * n + k].clone() * u_data[k * n + j].clone();
                }
                u_data[i * n + j] = self.data[i * n + j].clone() - sum;
            }

            // Lower triangular
            for i in (j + 1)..n {
                let mut sum = F::zero();
                for k in 0..j {
                    sum = sum + l_data[i * n + k].clone() * u_data[k * n + j].clone();
                }

                let u_jj = u_data[j * n + j].clone();
                if u_jj.is_zero() {
                    return Err(MathError::InvalidArgument(
                        "Zero pivot encountered; matrix may be singular".to_string(),
                    ));
                }

                l_data[i * n + j] = (self.data[i * n + j].clone() - sum) / u_jj;
            }
        }

        Ok(LUDecomposition {
            l: Matrix {
                data: l_data,
                rows: n,
                cols: n,
            },
            u: Matrix {
                data: u_data,
                rows: n,
                cols: n,
            },
        })
    }

    /// Compute PLU decomposition with partial pivoting
    ///
    /// Returns P, L, U such that PA = LU where P is a permutation matrix.
    /// This is more numerically stable than LU without pivoting.
    pub fn plu_decomposition(&self) -> Result<PLUDecomposition<F>> {
        if !self.is_square() {
            return Err(MathError::InvalidArgument(
                "PLU decomposition requires a square matrix".to_string(),
            ));
        }

        let n = self.rows;
        let mut a = self.clone();
        let mut l_data = vec![F::zero(); n * n];
        let mut perm = (0..n).collect::<Vec<_>>();

        // Initialize L diagonal to 1
        for i in 0..n {
            l_data[i * n + i] = F::one();
        }

        for k in 0..n {
            // Find pivot
            let mut max_row = k;
            let mut max_val = a.data[k * n + k].clone();

            for i in (k + 1)..n {
                let val = a.data[i * n + k].clone();
                // In a proper implementation, we'd compare absolute values
                // For now, just find first non-zero if current is zero
                if !val.is_zero() && max_val.is_zero() {
                    max_val = val;
                    max_row = i;
                }
            }

            if max_val.is_zero() {
                return Err(MathError::InvalidArgument(
                    "Matrix is singular; cannot compute PLU decomposition".to_string(),
                ));
            }

            // Swap rows in A and permutation
            if max_row != k {
                for j in 0..n {
                    let temp = a.data[k * n + j].clone();
                    a.data[k * n + j] = a.data[max_row * n + j].clone();
                    a.data[max_row * n + j] = temp;
                }

                // Swap in L (only the part that's been computed)
                for j in 0..k {
                    let temp = l_data[k * n + j].clone();
                    l_data[k * n + j] = l_data[max_row * n + j].clone();
                    l_data[max_row * n + j] = temp;
                }

                perm.swap(k, max_row);
            }

            // Gaussian elimination
            for i in (k + 1)..n {
                let factor = a.data[i * n + k].clone() / a.data[k * n + k].clone();
                l_data[i * n + k] = factor.clone();

                for j in (k + 1)..n {
                    let val = a.data[i * n + j].clone();
                    a.data[i * n + j] = val - factor.clone() * a.data[k * n + j].clone();
                }
                a.data[i * n + k] = F::zero();
            }
        }

        // Create permutation matrix
        let mut p_data = vec![F::zero(); n * n];
        for (i, &p_i) in perm.iter().enumerate() {
            p_data[i * n + p_i] = F::one();
        }

        Ok(PLUDecomposition {
            p: Matrix {
                data: p_data,
                rows: n,
                cols: n,
            },
            l: Matrix {
                data: l_data,
                rows: n,
                cols: n,
            },
            u: a, // A has been transformed into U
            perm,
        })
    }

    /// Compute determinant using LU decomposition
    ///
    /// This is more efficient than cofactor expansion for matrices larger than 3x3.
    /// det(A) = det(P) * det(L) * det(U) = (-1)^{# swaps} * product(diag(U))
    pub fn determinant_lu(&self) -> Result<F> {
        if !self.is_square() {
            return Err(MathError::InvalidArgument(
                "Determinant is only defined for square matrices".to_string(),
            ));
        }

        let plu = self.plu_decomposition()?;

        // Count number of swaps in permutation
        let mut swaps = 0;
        let mut visited = vec![false; self.rows];
        for i in 0..self.rows {
            if !visited[i] {
                let mut j = i;
                let mut cycle_len = 0;
                while !visited[j] {
                    visited[j] = true;
                    j = plu.perm[j];
                    cycle_len += 1;
                }
                if cycle_len > 1 {
                    swaps += cycle_len - 1;
                }
            }
        }

        // det(P) = (-1)^swaps
        let mut det = if swaps % 2 == 0 {
            F::one()
        } else {
            F::zero() - F::one()
        };

        // det(L) = 1 (diagonal is all ones)
        // det(U) = product of diagonal
        for i in 0..self.rows {
            det = det * plu.u.data[i * self.cols + i].clone();
        }

        Ok(det)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_lu_decomposition() {
        // [2 1]
        // [4 3]
        let m = Matrix::from_vec(
            2,
            2,
            vec![
                Rational::from((2, 1)),
                Rational::from((1, 1)),
                Rational::from((4, 1)),
                Rational::from((3, 1)),
            ],
        )
        .unwrap();

        let lu = m.lu_decomposition().unwrap();

        // Verify L has 1s on diagonal
        assert_eq!(*lu.l.get(0, 0).unwrap(), Rational::from((1, 1)));
        assert_eq!(*lu.l.get(1, 1).unwrap(), Rational::from((1, 1)));

        // Verify L is lower triangular
        assert_eq!(*lu.l.get(0, 1).unwrap(), Rational::from((0, 1)));

        // Verify U is upper triangular
        assert_eq!(*lu.u.get(1, 0).unwrap(), Rational::from((0, 1)));

        // Verify A = LU by multiplying back
        let reconstructed = (lu.l * lu.u).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(reconstructed.get(i, j).unwrap(), m.get(i, j).unwrap());
            }
        }
    }

    #[test]
    fn test_plu_decomposition() {
        let m = Matrix::from_vec(
            3,
            3,
            vec![
                Rational::from((2, 1)),
                Rational::from((1, 1)),
                Rational::from((1, 1)),
                Rational::from((4, 1)),
                Rational::from((3, 1)),
                Rational::from((3, 1)),
                Rational::from((8, 1)),
                Rational::from((7, 1)),
                Rational::from((9, 1)),
            ],
        )
        .unwrap();

        let plu = m.plu_decomposition().unwrap();

        // Verify PA = LU
        let pa = (plu.p.clone() * m).unwrap();
        let lu = (plu.l * plu.u).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(pa.get(i, j).unwrap(), lu.get(i, j).unwrap());
            }
        }
    }

    #[test]
    fn test_determinant_lu() {
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

        // det = 1*4 - 2*3 = -2
        let det = m.determinant_lu().unwrap();
        assert_eq!(det, Rational::from((-2, 1)));
    }
}
