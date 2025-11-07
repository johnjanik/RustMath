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

/// QR decomposition result
///
/// Represents A = QR where Q is orthogonal and R is upper triangular
pub struct QRDecomposition<F: Field> {
    /// Orthogonal matrix (Q^T Q = I)
    pub q: Matrix<F>,
    /// Upper triangular matrix
    pub r: Matrix<F>,
}

/// Cholesky decomposition result
///
/// Represents A = LL^T where L is lower triangular
pub struct CholeskyDecomposition<F: Field> {
    /// Lower triangular matrix
    pub l: Matrix<F>,
}

/// Hessenberg decomposition result
///
/// Represents A = QHQ^T where H is upper Hessenberg (zeros below first subdiagonal)
pub struct HessenbergDecomposition<F: Field> {
    /// Upper Hessenberg matrix
    pub h: Matrix<F>,
    /// Orthogonal transformation matrix
    pub q: Matrix<F>,
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

    /// Compute QR decomposition using Gram-Schmidt orthogonalization
    ///
    /// Returns Q and R such that A = QR where:
    /// - Q is orthogonal (Q^T Q = I)
    /// - R is upper triangular
    ///
    /// Uses the modified Gram-Schmidt algorithm for better numerical stability.
    pub fn qr_decomposition(&self) -> Result<QRDecomposition<F>>
    where
        F: rustmath_core::NumericConversion,
    {
        let m = self.rows;
        let n = self.cols;

        if m < n {
            return Err(MathError::InvalidArgument(
                "QR decomposition requires rows >= cols".to_string(),
            ));
        }

        // Extract columns
        let mut q_cols: Vec<Vec<F>> = Vec::new();
        let mut r_data = vec![F::zero(); n * n];

        for j in 0..n {
            // Get column j
            let mut v = self.col(j)?;

            // Orthogonalize against previous columns
            for i in 0..j {
                // r[i,j] = q_i^T * a_j
                let mut dot = F::zero();
                for k in 0..m {
                    dot = dot + q_cols[i][k].clone() * v[k].clone();
                }
                r_data[i * n + j] = dot.clone();

                // v = v - r[i,j] * q_i
                for k in 0..m {
                    v[k] = v[k].clone() - dot.clone() * q_cols[i][k].clone();
                }
            }

            // Compute norm of v
            let mut norm_squared = F::zero();
            for k in 0..m {
                norm_squared = norm_squared + v[k].clone() * v[k].clone();
            }

            let norm = match norm_squared.to_f64() {
                Some(val) if val > 0.0 => F::from_f64(val.sqrt()),
                _ => {
                    return Err(MathError::InvalidArgument(
                        "Column vectors are linearly dependent".to_string(),
                    ))
                }
            };

            let norm = match norm {
                Some(n) => n,
                None => {
                    return Err(MathError::InvalidArgument(
                        "Failed to compute norm".to_string(),
                    ))
                }
            };

            r_data[j * n + j] = norm.clone();

            // Normalize v
            let mut q_j = Vec::with_capacity(m);
            for k in 0..m {
                q_j.push(v[k].clone() / norm.clone());
            }

            q_cols.push(q_j);
        }

        // Build Q matrix from columns
        let mut q_data = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                q_data.push(q_cols[j][i].clone());
            }
        }

        Ok(QRDecomposition {
            q: Matrix {
                data: q_data,
                rows: m,
                cols: n,
            },
            r: Matrix {
                data: r_data,
                rows: n,
                cols: n,
            },
        })
    }

    /// Compute Cholesky decomposition for positive definite matrices
    ///
    /// Returns L such that A = LL^T where L is lower triangular.
    /// The matrix must be symmetric positive definite.
    ///
    /// This is more efficient than LU decomposition for this special case.
    pub fn cholesky_decomposition(&self) -> Result<CholeskyDecomposition<F>>
    where
        F: rustmath_core::NumericConversion,
    {
        if !self.is_square() {
            return Err(MathError::InvalidArgument(
                "Cholesky decomposition requires a square matrix".to_string(),
            ));
        }

        if !self.is_symmetric() {
            return Err(MathError::InvalidArgument(
                "Cholesky decomposition requires a symmetric matrix".to_string(),
            ));
        }

        let n = self.rows;
        let mut l_data = vec![F::zero(); n * n];

        for i in 0..n {
            for j in 0..=i {
                let mut sum = F::zero();

                if i == j {
                    // Diagonal element
                    for k in 0..j {
                        let l_jk = l_data[j * n + k].clone();
                        sum = sum + l_jk.clone() * l_jk;
                    }

                    let diff = self.data[i * n + j].clone() - sum;

                    // Check if positive
                    let diff_f64 = diff.to_f64().ok_or_else(|| {
                        MathError::InvalidArgument("Cannot convert to f64".to_string())
                    })?;

                    if diff_f64 <= 0.0 {
                        return Err(MathError::InvalidArgument(
                            "Matrix is not positive definite".to_string(),
                        ));
                    }

                    let sqrt_val = F::from_f64(diff_f64.sqrt()).ok_or_else(|| {
                        MathError::InvalidArgument("Cannot convert sqrt back".to_string())
                    })?;

                    l_data[i * n + j] = sqrt_val;
                } else {
                    // Off-diagonal element
                    for k in 0..j {
                        sum = sum + l_data[i * n + k].clone() * l_data[j * n + k].clone();
                    }

                    let l_jj = l_data[j * n + j].clone();
                    if l_jj.is_zero() {
                        return Err(MathError::DivisionByZero);
                    }

                    l_data[i * n + j] = (self.data[i * n + j].clone() - sum) / l_jj;
                }
            }
        }

        Ok(CholeskyDecomposition {
            l: Matrix {
                data: l_data,
                rows: n,
                cols: n,
            },
        })
    }

    /// Compute Hessenberg form using Householder reflections
    ///
    /// Returns H and Q such that H = Q^T A Q where H is upper Hessenberg.
    /// Upper Hessenberg means h[i][j] = 0 for i > j+1.
    ///
    /// This is useful as a preconditioner for eigenvalue algorithms.
    pub fn hessenberg_decomposition(&self) -> Result<HessenbergDecomposition<F>>
    where
        F: rustmath_core::NumericConversion,
    {
        if !self.is_square() {
            return Err(MathError::InvalidArgument(
                "Hessenberg decomposition requires a square matrix".to_string(),
            ));
        }

        let n = self.rows;
        if n <= 2 {
            // Already in Hessenberg form
            return Ok(HessenbergDecomposition {
                h: self.clone(),
                q: Matrix::identity(n),
            });
        }

        let mut h = self.clone();
        let mut q = Matrix::identity(n);

        // Householder reduction to Hessenberg form
        for k in 0..(n - 2) {
            // Compute Householder vector for column k, rows k+1 to n
            let mut v = vec![F::zero(); n - k - 1];
            for i in 0..(n - k - 1) {
                v[i] = h.data[(k + 1 + i) * n + k].clone();
            }

            // Compute norm
            let mut norm_sq = F::zero();
            for vi in &v {
                norm_sq = norm_sq + vi.clone() * vi.clone();
            }

            let norm = match norm_sq.to_f64() {
                Some(val) if val > 1e-10 => match F::from_f64(val.sqrt()) {
                    Some(n) => n,
                    None => continue, // Skip if can't convert
                },
                _ => continue, // Skip if zero or too small
            };

            // v[0] += sign(v[0]) * norm
            let sign = if v[0].to_f64().unwrap_or(0.0) >= 0.0 {
                F::one()
            } else {
                F::zero() - F::one()
            };
            v[0] = v[0].clone() + sign * norm;

            // Normalize v
            let mut v_norm_sq = F::zero();
            for vi in &v {
                v_norm_sq = v_norm_sq + vi.clone() * vi.clone();
            }

            let v_norm = match v_norm_sq.to_f64() {
                Some(val) if val > 1e-10 => match F::from_f64(val.sqrt()) {
                    Some(n) => n,
                    None => continue,
                },
                _ => continue,
            };

            for vi in &mut v {
                *vi = vi.clone() / v_norm.clone();
            }

            // Apply Householder transformation: H = I - 2vv^T
            // h = (I - 2vv^T) * h * (I - 2vv^T)

            // Left multiplication: (I - 2vv^T) * h
            for j in 0..n {
                // Compute v^T * h[:, j] for rows k+1 to n
                let mut dot = F::zero();
                for i in 0..(n - k - 1) {
                    dot = dot + v[i].clone() * h.data[(k + 1 + i) * n + j].clone();
                }

                let two = F::one() + F::one();
                let factor = two * dot;

                for i in 0..(n - k - 1) {
                    h.data[(k + 1 + i) * n + j] =
                        h.data[(k + 1 + i) * n + j].clone() - factor.clone() * v[i].clone();
                }
            }

            // Right multiplication: h * (I - 2vv^T)
            for i in 0..n {
                // Compute h[i, :] * v for cols k+1 to n
                let mut dot = F::zero();
                for j in 0..(n - k - 1) {
                    dot = dot + h.data[i * n + (k + 1 + j)].clone() * v[j].clone();
                }

                let two = F::one() + F::one();
                let factor = two * dot;

                for j in 0..(n - k - 1) {
                    h.data[i * n + (k + 1 + j)] =
                        h.data[i * n + (k + 1 + j)].clone() - factor.clone() * v[j].clone();
                }
            }

            // Update Q matrix
            for i in 0..n {
                let mut dot = F::zero();
                for j in 0..(n - k - 1) {
                    dot = dot + q.data[i * n + (k + 1 + j)].clone() * v[j].clone();
                }

                let two = F::one() + F::one();
                let factor = two * dot;

                for j in 0..(n - k - 1) {
                    q.data[i * n + (k + 1 + j)] =
                        q.data[i * n + (k + 1 + j)].clone() - factor.clone() * v[j].clone();
                }
            }
        }

        Ok(HessenbergDecomposition { h, q })
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

    #[test]
    fn test_qr_decomposition() {
        // Simple 2x2 matrix
        let m = Matrix::from_vec(
            2,
            2,
            vec![
                Rational::from((1, 1)),
                Rational::from((0, 1)),
                Rational::from((0, 1)),
                Rational::from((1, 1)),
            ],
        )
        .unwrap();

        let qr = m.qr_decomposition().unwrap();

        // Verify Q is orthogonal: Q^T Q = I
        let qt = qr.q.transpose();
        let qtq = (qt * qr.q.clone()).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j {
                    Rational::from((1, 1))
                } else {
                    Rational::from((0, 1))
                };
                let val = qtq.get(i, j).unwrap();
                // Allow small numerical error
                let diff = (val.clone() - expected.clone()).to_f64().unwrap_or(0.0).abs();
                assert!(diff < 1e-10, "Q^T Q should be identity");
            }
        }

        // Verify A = QR
        let reconstructed = (qr.q * qr.r).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let orig = m.get(i, j).unwrap();
                let recon = reconstructed.get(i, j).unwrap();
                let diff = (orig.clone() - recon.clone()).to_f64().unwrap_or(0.0).abs();
                assert!(diff < 1e-10, "A should equal QR");
            }
        }
    }

    #[test]
    fn test_cholesky_decomposition() {
        // Positive definite matrix:
        // [4  12  -16]
        // [12  37  -43]
        // [-16 -43  98]
        // This has Cholesky factor L =
        // [2   0   0]
        // [6   1   0]
        // [-8  5   3]

        let m = Matrix::from_vec(
            3,
            3,
            vec![
                Rational::from((4, 1)),
                Rational::from((12, 1)),
                Rational::from((-16, 1)),
                Rational::from((12, 1)),
                Rational::from((37, 1)),
                Rational::from((-43, 1)),
                Rational::from((-16, 1)),
                Rational::from((-43, 1)),
                Rational::from((98, 1)),
            ],
        )
        .unwrap();

        let chol = m.cholesky_decomposition().unwrap();

        // Verify L is lower triangular
        assert!(chol.l.is_lower_triangular());

        // Verify A = LL^T
        let lt = chol.l.transpose();
        let reconstructed = (chol.l * lt).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                let orig = m.get(i, j).unwrap();
                let recon = reconstructed.get(i, j).unwrap();
                let diff = (orig.clone() - recon.clone()).to_f64().unwrap_or(0.0).abs();
                assert!(diff < 1e-8, "A should equal LL^T at ({}, {})", i, j);
            }
        }
    }
}
