//! Eigenvalue and eigenvector computation

use crate::Matrix;
use rustmath_core::{Field, MathError, Result};

/// Result of eigenvalue computation
#[derive(Debug, Clone)]
pub struct EigenDecomposition<F: Field> {
    /// Eigenvalues
    pub eigenvalues: Vec<F>,
}

/// Result of eigenvector computation
#[derive(Debug, Clone)]
pub struct Eigenvector<F: Field> {
    /// The eigenvalue
    pub eigenvalue: F,
    /// The eigenvector
    pub eigenvector: Vec<F>,
}

/* // Commented out: Requires from_f64 and other floating-point operations
impl<F: Field> Matrix<F> {
    /// Compute the dominant eigenvalue using power iteration
    ///
    /// Returns the eigenvalue with largest absolute value and its eigenvector.
    /// This is an iterative method that may not converge for all matrices.
    pub fn power_iteration(&self, max_iterations: usize, tolerance: f64) -> Result<Eigenvector<F>>
    where
        F: rustmath_core::NumericConversion,
    {
        if !self.is_square() {
            return Err(MathError::InvalidArgument(
                "Power iteration requires a square matrix".to_string(),
            ));
        }

        let n = self.rows();

        // Start with a random vector (use all ones for determinism)
        let mut v = vec![F::one(); n];

        for _ in 0..max_iterations {
            // Compute A*v
            let mut av = vec![F::zero(); n];
            for i in 0..n {
                let mut sum = F::zero();
                for j in 0..n {
                    sum = sum + self.data()[i * n + j].clone() * v[j].clone();
                }
                av[i] = sum;
            }

            // Compute norm of Av
            let mut norm_sq = F::zero();
            for x in &av {
                norm_sq = norm_sq + x.clone() * x.clone();
            }

            let norm = match norm_sq.to_f64() {
                Some(val) if val > 0.0 => match F::from_f64(val.sqrt()) {
                    Some(n) => n,
                    None => {
                        return Err(MathError::InvalidArgument(
                            "Cannot convert norm".to_string(),
                        ))
                    }
                },
                _ => {
                    return Err(MathError::InvalidArgument(
                        "Vector norm is zero or invalid".to_string(),
                    ))
                }
            };

            // Normalize: v_new = Av / ||Av||
            let mut v_new = Vec::with_capacity(n);
            for x in av {
                v_new.push(x / norm.clone());
            }

            // Check convergence: ||v_new - v|| < tolerance
            let mut diff_sq = F::zero();
            for i in 0..n {
                let d = v_new[i].clone() - v[i].clone();
                diff_sq = diff_sq + d.clone() * d;
            }

            if let Some(diff_val) = diff_sq.to_f64() {
                if diff_val.sqrt() < tolerance {
                    // Converged - compute Rayleigh quotient: eigenvalue = v^T A v
                    let mut eigenval = F::zero();
                    for i in 0..n {
                        let mut av_i = F::zero();
                        for j in 0..n {
                            av_i = av_i + self.data()[i * n + j].clone() * v_new[j].clone();
                        }
                        eigenval = eigenval + v_new[i].clone() * av_i;
                    }

                    return Ok(Eigenvector {
                        eigenvalue: eigenval,
                        eigenvector: v_new,
                    });
                }
            }

            v = v_new;
        }

        Err(MathError::InvalidArgument(
            "Power iteration did not converge".to_string(),
        ))
    }

    /// Compute all eigenvalues using the QR algorithm
    ///
    /// This is an iterative algorithm that diagonalizes the matrix.
    /// Returns approximate eigenvalues (may be complex for real matrices).
    pub fn eigenvalues(&self, max_iterations: usize, tolerance: f64) -> Result<Vec<F>>
    where
        F: rustmath_core::NumericConversion,
    {
        if !self.is_square() {
            return Err(MathError::InvalidArgument(
                "Eigenvalues are only defined for square matrices".to_string(),
            ));
        }

        let n = self.rows();

        // First reduce to Hessenberg form for efficiency
        let hess = self.hessenberg_decomposition()?;
        let mut a = hess.h;

        // QR algorithm
        for _ in 0..max_iterations {
            // Check if already diagonal (within tolerance)
            let mut is_diagonal = true;
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        if let Some(val) = a.data()[i * n + j].to_f64() {
                            if val.abs() > tolerance {
                                is_diagonal = false;
                                break;
                            }
                        }
                    }
                }
                if !is_diagonal {
                    break;
                }
            }

            if is_diagonal {
                break;
            }

            // Perform QR decomposition
            let qr = match a.qr_decomposition() {
                Ok(qr) => qr,
                Err(_) => break, // If QR fails, stop and return current diagonal
            };

            // A_new = R * Q
            a = match qr.r * qr.q {
                Ok(m) => m,
                Err(_) => break,
            };
        }

        // Extract eigenvalues from diagonal
        let mut eigenvalues = Vec::with_capacity(n);
        for i in 0..n {
            eigenvalues.push(a.data()[i * n + i].clone());
        }

        Ok(eigenvalues)
    }

    /// Compute an eigenvector for a given eigenvalue
    ///
    /// Solves (A - λI)v = 0 for v.
    pub fn eigenvector_for_eigenvalue(&self, eigenvalue: &F) -> Result<Vec<F>>
    where
        F: rustmath_core::NumericConversion,
    {
        if !self.is_square() {
            return Err(MathError::InvalidArgument(
                "Eigenvectors are only defined for square matrices".to_string(),
            ));
        }

        let n = self.rows();

        // Compute A - λI
        let mut a_shifted = self.clone();
        for i in 0..n {
            let old_val = a_shifted.data()[i * n + i].clone();
            a_shifted.data_mut()[i * n + i] = old_val - eigenvalue.clone();
        }

        // Find kernel (null space) of A - λI
        let kernel = a_shifted.kernel()?;

        if kernel.is_empty() {
            return Err(MathError::InvalidArgument(
                "No eigenvector found for this eigenvalue".to_string(),
            ));
        }

        // Return the first basis vector of the kernel
        Ok(kernel[0].clone())
    }

    /// Compute all eigenvectors (right eigenvectors)
    ///
    /// Returns a vector of (eigenvalue, eigenvector) pairs.
    pub fn eigenvectors_right(&self, max_iterations: usize, tolerance: f64) -> Result<Vec<Eigenvector<F>>>
    where
        F: rustmath_core::NumericConversion,
    {
        let eigenvalues = self.eigenvalues(max_iterations, tolerance)?;

        let mut result = Vec::new();

        for eigenval in eigenvalues {
            // Try to compute eigenvector for this eigenvalue
            if let Ok(eigenvec) = self.eigenvector_for_eigenvalue(&eigenval) {
                result.push(Eigenvector {
                    eigenvalue: eigenval,
                    eigenvector: eigenvec,
                });
            }
        }

        Ok(result)
    }

    /// Compute left eigenvectors
    ///
    /// Left eigenvectors satisfy v^T A = λ v^T, which is equivalent to
    /// computing right eigenvectors of A^T.
    pub fn eigenvectors_left(&self, max_iterations: usize, tolerance: f64) -> Result<Vec<Eigenvector<F>>>
    where
        F: rustmath_core::NumericConversion,
    {
        let at = self.transpose();
        at.eigenvectors_right(max_iterations, tolerance)
    }

    /// Compute the Jordan canonical form
    ///
    /// Returns J and P such that A = P J P^(-1), where J is in Jordan form.
    ///
    /// Note: This implementation works best for diagonalizable matrices.
    /// For non-diagonalizable matrices, it attempts to compute generalized
    /// eigenvectors but may not always succeed.
    pub fn jordan_form(&self, max_iterations: usize, tolerance: f64) -> Result<JordanForm<F>>
    where
        F: rustmath_core::NumericConversion,
    {
        if !self.is_square() {
            return Err(MathError::InvalidArgument(
                "Jordan form requires a square matrix".to_string(),
            ));
        }

        let n = self.rows();

        // Compute eigenvalues and eigenvectors
        let eigenvalues = self.eigenvalues(max_iterations, tolerance)?;
        let eigenvectors = self.eigenvectors_right(max_iterations, tolerance)?;

        if eigenvectors.is_empty() {
            return Err(MathError::InvalidArgument(
                "No eigenvectors found".to_string(),
            ));
        }

        // Build the Jordan matrix J and transformation matrix P
        let mut j_data = vec![F::zero(); n * n];
        let mut p_data = vec![F::zero(); n * n];

        // For now, assume matrix is diagonalizable or nearly so
        // In a full implementation, we'd compute generalized eigenvectors here

        for (idx, eigvec) in eigenvectors.iter().enumerate().take(n) {
            // Set diagonal entry in J to eigenvalue
            j_data[idx * n + idx] = eigvec.eigenvalue.clone();

            // Set column of P to eigenvector
            for i in 0..n.min(eigvec.eigenvector.len()) {
                p_data[i * n + idx] = eigvec.eigenvector[i].clone();
            }
        }

        // If we don't have enough eigenvectors, try to find generalized eigenvectors
        if eigenvectors.len() < n {
            // Attempt to find generalized eigenvectors for repeated eigenvalues
            let mut used = eigenvectors.len();

            for eigenval in &eigenvalues {
                if used >= n {
                    break;
                }

                // Try to find a generalized eigenvector by solving (A - λI)^k v = 0
                // for increasing k
                for k in 2..=3 {
                    if used >= n {
                        break;
                    }

                    // Compute (A - λI)^k
                    let mut a_shifted = self.clone();
                    for i in 0..n {
                        let old_val = a_shifted.data()[i * n + i].clone();
                        a_shifted.data_mut()[i * n + i] = old_val - eigenval.clone();
                    }

                    let mut power = a_shifted.clone();
                    for _ in 1..k {
                        power = match power.clone() * a_shifted.clone() {
                            Ok(m) => m,
                            Err(_) => break,
                        };
                    }

                    // Find kernel
                    if let Ok(kernel) = power.kernel() {
                        for vec in kernel {
                            if used >= n {
                                break;
                            }

                            // Check if this is linearly independent from what we have
                            let mut is_new = true;
                            for j in 0..used {
                                let mut all_same = true;
                                for i in 0..n.min(vec.len()) {
                                    if (vec[i].clone() - p_data[i * n + j].clone()).is_zero() {
                                        continue;
                                    }
                                    all_same = false;
                                    break;
                                }
                                if all_same {
                                    is_new = false;
                                    break;
                                }
                            }

                            if is_new {
                                // Add to P
                                for i in 0..n.min(vec.len()) {
                                    p_data[i * n + used] = vec[i].clone();
                                }

                                // Set J entry
                                j_data[used * n + used] = eigenval.clone();

                                // Add 1 on the superdiagonal if this is a Jordan block
                                if used > 0 && (j_data[(used - 1) * n + (used - 1)].clone() - eigenval.clone()).is_zero() {
                                    j_data[(used - 1) * n + used] = F::one();
                                }

                                used += 1;
                            }
                        }
                    }
                }
            }
        }

        let j = Matrix::from_vec(n, n, j_data)?;
        let p = Matrix::from_vec(n, n, p_data)?;

        Ok(JordanForm { j, p })
    }
}
*/

/// Jordan canonical form result
#[derive(Debug, Clone)]
pub struct JordanForm<F: Field> {
    /// The Jordan form matrix J (block diagonal with Jordan blocks)
    pub j: Matrix<F>,
    /// The transformation matrix P such that A = P J P^(-1)
    pub p: Matrix<F>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    /* // Commented out: power_iteration method is commented out
    #[test]
    fn test_power_iteration_simple() {
        // Matrix with known eigenvalue 3:
        // [2 1]
        // [1 2]
        // Eigenvalues are 3 and 1
        let m = Matrix::from_vec(
            2,
            2,
            vec![
                Rational::from_integer(2),
                Rational::from_integer(1),
                Rational::from_integer(1),
                Rational::from_integer(2),
            ],
        )
        .unwrap();

        let result = m.power_iteration(100, 1e-6).unwrap();

        // Dominant eigenvalue should be 3
        let eigenval_f64 = result.eigenvalue.to_f64().unwrap();
        assert!((eigenval_f64 - 3.0).abs() < 0.01, "Eigenvalue should be close to 3");
    }

    #[test]
    fn test_eigenvalues_diagonal() {
        // Diagonal matrix has eigenvalues on the diagonal
        let m = Matrix::from_vec(
            3,
            3,
            vec![
                Rational::from_integer(2),
                Rational::from_integer(0),
                Rational::from_integer(0),
                Rational::from_integer(0),
                Rational::from_integer(3),
                Rational::from_integer(0),
                Rational::from_integer(0),
                Rational::from_integer(0),
                Rational::from_integer(5),
            ],
        )
        .unwrap();

        let eigenvals = m.eigenvalues(100, 1e-6).unwrap();

        assert_eq!(eigenvals.len(), 3);

        // Convert to f64 and sort for comparison
        let mut vals: Vec<f64> = eigenvals.iter().map(|v| v.to_f64().unwrap()).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert!((vals[0] - 2.0).abs() < 0.01);
        assert!((vals[1] - 3.0).abs() < 0.01);
        assert!((vals[2] - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_eigenvector_computation() {
        // Identity matrix: eigenvalue 1, any vector is an eigenvector
        let m = Matrix::<Rational>::identity(2);

        let eigenvec = m
            .eigenvector_for_eigenvalue(&Rational::from_integer(1))
            .unwrap();

        assert_eq!(eigenvec.len(), 2);
    }
    */
}
