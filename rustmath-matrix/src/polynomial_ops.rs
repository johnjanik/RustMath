//! Polynomial operations on matrices (characteristic polynomial, minimal polynomial, etc.)

use crate::Matrix;
use rustmath_core::{Field, MathError, Result};
use rustmath_polynomials::UnivariatePolynomial;

impl<F: Field> Matrix<F> {
    /// Compute the characteristic polynomial det(A - λI)
    ///
    /// Uses the Faddeev-LeVerrier algorithm to compute coefficients.
    /// The characteristic polynomial is p(λ) = det(λI - A).
    ///
    /// For an n×n matrix, returns a degree-n polynomial.
    pub fn characteristic_polynomial(&self) -> Result<UnivariatePolynomial<F>>
    where
        F: rustmath_core::NumericConversion,
    {
        if !self.is_square() {
            return Err(MathError::InvalidArgument(
                "Characteristic polynomial is only defined for square matrices".to_string(),
            ));
        }

        let n = self.rows;

        // Special cases for small matrices
        if n == 1 {
            // p(λ) = λ - a₀₀
            return Ok(UnivariatePolynomial::new(vec![
                F::zero() - self.data[0].clone(),
                F::one(),
            ]));
        }

        if n == 2 {
            // p(λ) = λ² - tr(A)λ + det(A)
            let trace = self.trace()?;
            let det = self.determinant()?;
            return Ok(UnivariatePolynomial::new(vec![
                det,
                F::zero() - trace,
                F::one(),
            ]));
        }

        // Faddeev-LeVerrier algorithm for general case
        // p(λ) = λⁿ + c_{n-1}λⁿ⁻¹ + ... + c₁λ + c₀
        let mut coeffs = vec![F::zero(); n + 1];
        coeffs[n] = F::one(); // Leading coefficient

        let mut m = Matrix::identity(n); // M₀ = I
        let mut c = F::zero(); // c_n = 0

        for k in 1..=n {
            // M_k = A * M_{k-1} + c_{n-k+1} * I
            m = (self.clone() * m)?;

            if k > 1 {
                // Add c * I
                for i in 0..n {
                    m.data[i * n + i] = m.data[i * n + i].clone() + c.clone();
                }
            }

            // c_{n-k} = -1/k * tr(M_k)
            let trace = m.trace()?;
            let k_field = F::from_i64(k as i64).ok_or_else(|| {
                MathError::InvalidArgument("Cannot convert k to field element".to_string())
            })?;
            c = F::zero() - (trace / k_field);
            coeffs[n - k] = c.clone();
        }

        Ok(UnivariatePolynomial::new(coeffs))
    }

    /// Compute the minimal polynomial
    ///
    /// The minimal polynomial is the monic polynomial of smallest degree that
    /// annihilates the matrix (i.e., m(A) = 0).
    ///
    /// For each distinct eigenvalue λ, the exponent in the minimal polynomial
    /// equals the size of the largest Jordan block for λ.
    ///
    /// This implementation:
    /// 1. Computes eigenvalues
    /// 2. For each eigenvalue, finds the smallest k where (A - λI)^k has stable nullity
    /// 3. Constructs m(x) = ∏(x - λ_i)^k_i
    pub fn minimal_polynomial(&self) -> Result<UnivariatePolynomial<F>>
    where
        F: rustmath_core::NumericConversion,
    {
        if !self.is_square() {
            return Err(MathError::InvalidArgument(
                "Minimal polynomial is only defined for square matrices".to_string(),
            ));
        }

        let n = self.rows;

        // Special case for 1x1 matrices
        if n == 1 {
            return Ok(UnivariatePolynomial::new(vec![
                F::zero() - self.data[0].clone(),
                F::one(),
            ]));
        }

        // Compute eigenvalues
        let eigenvalues = match self.eigenvalues(100, 1e-10) {
            Ok(vals) => vals,
            Err(_) => {
                // If eigenvalue computation fails, fall back to characteristic polynomial
                return self.characteristic_polynomial();
            }
        };

        if eigenvalues.is_empty() {
            return self.characteristic_polynomial();
        }

        // Group eigenvalues by uniqueness (with tolerance)
        let mut unique_eigenvalues: Vec<(F, usize)> = Vec::new();

        for eval in eigenvalues {
            let mut found = false;

            for (unique_eval, max_power) in &mut unique_eigenvalues {
                // Check if eigenvalues are "equal" (within numerical tolerance)
                let diff = eval.clone() - unique_eval.clone();
                if let Some(diff_f64) = diff.to_f64() {
                    if diff_f64.abs() < 1e-10 {
                        // Same eigenvalue, we'll update the power later
                        found = true;
                        break;
                    }
                }
            }

            if !found {
                // Determine the exponent for this eigenvalue
                let exponent = self.find_minimal_polynomial_exponent(&eval)?;
                unique_eigenvalues.push((eval, exponent));
            }
        }

        // Build minimal polynomial as product of (x - λ_i)^k_i
        let mut result = UnivariatePolynomial::new(vec![F::one()]);

        for (eigenval, exponent) in unique_eigenvalues {
            // Create (x - eigenval)
            let linear_factor = UnivariatePolynomial::new(vec![
                F::zero() - eigenval.clone(),
                F::one(),
            ]);

            // Raise to power exponent
            for _ in 0..exponent {
                result = result * linear_factor.clone();
            }
        }

        Ok(result)
    }

    /// Find the exponent of an eigenvalue in the minimal polynomial
    ///
    /// This equals the size of the largest Jordan block for this eigenvalue.
    /// We find it by computing the nullity of (A - λI)^k for increasing k.
    fn find_minimal_polynomial_exponent(&self, eigenvalue: &F) -> Result<usize>
    where
        F: rustmath_core::NumericConversion,
    {
        let n = self.rows;

        // Compute A - λI
        let mut a_shifted = self.clone();
        for i in 0..n {
            a_shifted.data[i * n + i] = a_shifted.data[i * n + i].clone() - eigenvalue.clone();
        }

        let mut prev_nullity = 0;
        let mut current_matrix = a_shifted.clone();

        // Check powers up to n (since the largest Jordan block can't be larger than n)
        for k in 1..=n {
            // Compute nullity of (A - λI)^k
            let nullity = match current_matrix.kernel() {
                Ok(kernel) => kernel.len(),
                Err(_) => 0,
            };

            // If nullity stabilizes, we've found the exponent
            if k > 1 && nullity == prev_nullity {
                return Ok(k - 1);
            }

            prev_nullity = nullity;

            // Compute next power: (A - λI)^(k+1) = (A - λI)^k * (A - λI)
            if k < n {
                current_matrix = match current_matrix.clone() * a_shifted.clone() {
                    Ok(m) => m,
                    Err(_) => break,
                };
            }
        }

        // If we reach here, the exponent is at most n
        Ok(n.min(prev_nullity.max(1)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_characteristic_polynomial_2x2() {
        // Matrix [1 2]
        //        [3 4]
        // Characteristic polynomial: λ² - 5λ + (4 - 6) = λ² - 5λ - 2
        let m = Matrix::from_vec(
            2,
            2,
            vec![
                Rational::from(1),
                Rational::from(2),
                Rational::from(3),
                Rational::from(4),
            ],
        )
        .unwrap();

        let charpoly = m.characteristic_polynomial().unwrap();

        assert_eq!(charpoly.degree(), 2);

        // Check constant term: det(A) = 1*4 - 2*3 = -2
        let c0 = charpoly.coefficient(0);
        assert_eq!(c0, Rational::from(-2));

        // Check linear term: -tr(A) = -(1+4) = -5
        let c1 = charpoly.coefficient(1);
        assert_eq!(c1, Rational::from(-5));

        // Check leading coefficient
        let c2 = charpoly.coefficient(2);
        assert_eq!(c2, Rational::from(1));
    }

    #[test]
    fn test_characteristic_polynomial_identity() {
        // Identity matrix
        // Characteristic polynomial: (λ - 1)ⁿ
        let m = Matrix::<Rational>::identity(3);
        let charpoly = m.characteristic_polynomial().unwrap();

        assert_eq!(charpoly.degree(), 3);

        // For I₃: p(λ) = (λ - 1)³ = λ³ - 3λ² + 3λ - 1
        assert_eq!(charpoly.coefficient(0), Rational::from(-1)); // constant
        assert_eq!(charpoly.coefficient(1), Rational::from(3)); // λ term
        assert_eq!(charpoly.coefficient(2), Rational::from(-3)); // λ² term
        assert_eq!(charpoly.coefficient(3), Rational::from(1)); // λ³ term
    }

    #[test]
    fn test_characteristic_polynomial_zero() {
        // Zero matrix
        // Characteristic polynomial: λⁿ
        let m = Matrix::<Rational>::zeros(2, 2);
        let charpoly = m.characteristic_polynomial().unwrap();

        assert_eq!(charpoly.degree(), 2);
        assert_eq!(charpoly.coefficient(0), Rational::from(0)); // constant = det = 0
        assert_eq!(charpoly.coefficient(1), Rational::from(0)); // linear = -tr = 0
        assert_eq!(charpoly.coefficient(2), Rational::from(1)); // λ²
    }
}
