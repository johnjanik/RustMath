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

    /// Compute a minimal polynomial (simplified version)
    ///
    /// This is a placeholder - proper implementation requires eigenvalue computation.
    /// For now, returns the characteristic polynomial as an approximation.
    pub fn minimal_polynomial(&self) -> Result<UnivariatePolynomial<F>>
    where
        F: rustmath_core::NumericConversion,
    {
        // TODO: Implement proper minimal polynomial computation
        // The minimal polynomial divides the characteristic polynomial
        self.characteristic_polynomial()
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
