//! Univariate polynomials (polynomials in one variable)

use crate::polynomial::Polynomial;
use rustmath_core::{EuclideanDomain, MathError, Result, Ring};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

/// Univariate polynomial over a ring R
#[derive(Clone, PartialEq, Eq)]
pub struct UnivariatePolynomial<R: Ring> {
    /// Coefficients in increasing degree order: [a0, a1, a2, ...] represents a0 + a1*x + a2*x^2 + ...
    coeffs: Vec<R>,
}

impl<R: Ring> UnivariatePolynomial<R> {
    /// Create a new polynomial from coefficients
    pub fn new(mut coeffs: Vec<R>) -> Self {
        // Remove leading zeros
        while coeffs.len() > 1 && coeffs.last().is_some_and(|c| c.is_zero()) {
            coeffs.pop();
        }

        if coeffs.is_empty() {
            coeffs.push(R::zero());
        }

        UnivariatePolynomial { coeffs }
    }

    /// Create a constant polynomial
    pub fn constant(c: R) -> Self {
        UnivariatePolynomial::new(vec![c])
    }

    /// Create the polynomial x (the variable)
    pub fn var() -> Self {
        UnivariatePolynomial::new(vec![R::zero(), R::one()])
    }

    /// Get the coefficients
    pub fn coefficients(&self) -> &[R] {
        &self.coeffs
    }

    /// Multiply by a scalar
    pub fn scalar_mul(&self, scalar: &R) -> Self {
        if scalar.is_zero() {
            return UnivariatePolynomial::new(vec![R::zero()]);
        }

        let coeffs = self.coeffs.iter().map(|c| c.clone() * scalar.clone()).collect();
        UnivariatePolynomial::new(coeffs)
    }

    /// Shift the polynomial by multiplying by x^n
    pub fn shift(&self, n: usize) -> Self {
        if n == 0 || self.is_zero() {
            return self.clone();
        }

        let mut coeffs = vec![R::zero(); n];
        coeffs.extend_from_slice(&self.coeffs);
        UnivariatePolynomial::new(coeffs)
    }

    /// Polynomial composition: compute p(q(x))
    ///
    /// Substitutes polynomial q into this polynomial p
    pub fn compose(&self, q: &Self) -> Self {
        if self.is_zero() {
            return UnivariatePolynomial::new(vec![R::zero()]);
        }

        // Use Horner's method: p(q) = a_0 + q*(a_1 + q*(a_2 + ...))
        let mut result = UnivariatePolynomial::new(vec![self.coeffs.last().unwrap().clone()]);

        for coeff in self.coeffs.iter().rev().skip(1) {
            // result = result * q + coeff
            result = (result.clone() * q.clone())
                + UnivariatePolynomial::new(vec![coeff.clone()]);
        }

        result
    }

    /// Scale the variable: compute p(c*x) for constant c
    pub fn scale_variable(&self, c: &R) -> Self {
        let mut coeffs = Vec::with_capacity(self.coeffs.len());
        let mut power_of_c = R::one();

        for coeff in &self.coeffs {
            coeffs.push(coeff.clone() * power_of_c.clone());
            power_of_c = power_of_c * c.clone();
        }

        UnivariatePolynomial::new(coeffs)
    }

    /// Translate the polynomial: compute p(x + a)
    pub fn translate(&self, a: &R) -> Self {
        // p(x + a) = p composed with (x + a)
        let x_plus_a = UnivariatePolynomial::new(vec![a.clone(), R::one()]);
        self.compose(&x_plus_a)
    }

    /// Compute the derivative
    pub fn derivative(&self) -> Self {
        if self.coeffs.len() <= 1 {
            return UnivariatePolynomial::new(vec![R::zero()]);
        }

        let coeffs = self.coeffs[1..]
            .iter()
            .enumerate()
            .map(|(i, c)| {
                // Multiply coefficient by (i+1) using repeated addition
                let mut result = R::zero();
                for _ in 0..=(i as u32) {
                    result = result + c.clone();
                }
                result
            })
            .collect();

        UnivariatePolynomial::new(coeffs)
    }

    /// Compute the indefinite integral (antiderivative)
    ///
    /// Returns ∫p(x)dx with constant of integration = 0
    /// For coefficient ring where division by integers is available
    pub fn integrate(&self) -> Self
    where
        R: rustmath_core::NumericConversion,
    {
        if self.is_zero() {
            return UnivariatePolynomial::new(vec![R::zero()]);
        }

        // Add a zero constant term, then divide each coefficient by its new index
        let mut coeffs = vec![R::zero()]; // Constant of integration

        for (i, c) in self.coeffs.iter().enumerate() {
            // Coefficient for x^(i+1) is c/(i+1)
            let divisor = R::from_i64((i + 1) as i64);
            // For integer coefficients, this is an approximation
            // In a proper implementation, we'd need rational coefficients
            let new_coeff = c.clone() / divisor;
            coeffs.push(new_coeff);
        }

        UnivariatePolynomial::new(coeffs)
    }

    /// Compute definite integral from a to b
    ///
    /// ∫[a,b] p(x)dx = P(b) - P(a) where P is the antiderivative
    pub fn definite_integral(&self, a: &R, b: &R) -> R
    where
        R: rustmath_core::NumericConversion,
    {
        let antiderivative = self.integrate();
        antiderivative.eval(b) - antiderivative.eval(a)
    }

    /// Compute polynomial GCD (for polynomials over a field or Euclidean domain)
    ///
    /// # Limitations
    ///
    /// This implementation uses the Euclidean algorithm which requires exact division.
    /// For polynomials with integer coefficients, this may fail or produce incorrect
    /// results when the leading coefficient of the divisor doesn't divide the leading
    /// coefficient of the dividend.
    ///
    /// **TODO**: Implement pseudo-division or subresultant-based GCD algorithm for
    /// polynomials over integers (Z[x]). Until then, GCD over integer polynomials
    /// works reliably only when coefficients divide cleanly.
    pub fn gcd(&self, other: &Self) -> Self
    where
        R: EuclideanDomain,
    {
        let mut a = self.clone();
        let mut b = other.clone();

        while !b.is_zero() {
            let (_, r) = a.div_rem(&b).unwrap();
            a = b;
            b = r;
        }

        // Note: Making polynomial monic (leading coefficient = 1) requires
        // division by the leading coefficient, which needs the Field trait.
        // This is left as a future enhancement.

        a
    }

    /// Compute polynomial LCM (least common multiple)
    ///
    /// For polynomials f and g: lcm(f, g) = (f * g) / gcd(f, g)
    ///
    /// # Limitations
    ///
    /// Same limitations as GCD - works best over fields or when coefficients divide cleanly.
    pub fn lcm(&self, other: &Self) -> Self
    where
        R: EuclideanDomain,
    {
        if self.is_zero() || other.is_zero() {
            return UnivariatePolynomial::new(vec![R::zero()]);
        }

        let g = self.gcd(other);
        let product = self.clone() * other.clone();

        // lcm = (self * other) / gcd
        match product.div_rem(&g) {
            Ok((quotient, _)) => quotient,
            Err(_) => product, // Fallback if division fails
        }
    }

    /// Compute the discriminant of a polynomial
    ///
    /// Currently supports degrees 2 and 3.
    /// Returns None for unsupported degrees or zero polynomial.
    pub fn discriminant(&self) -> Option<R>
    where
        R: rustmath_core::NumericConversion,
    {
        match self.degree()? {
            0 | 1 => Some(R::one()),
            2 => {
                // ax² + bx + c
                // Discriminant = b² - 4ac
                let a = self.coeff(2);
                let b = self.coeff(1);
                let c = self.coeff(0);

                let b_squared = b.clone() * b.clone();
                let four = R::from_i64(4);
                let four_ac = four * a.clone() * c.clone();
                Some(b_squared - four_ac)
            }
            3 => {
                // ax³ + bx² + cx + d
                // Discriminant = 18abcd - 4b³d + b²c² - 4ac³ - 27a²d²
                let a = self.coeff(3);
                let b = self.coeff(2);
                let c = self.coeff(1);
                let d = self.coeff(0);

                let term1 = R::from_i64(18) * a.clone() * b.clone() * c.clone() * d.clone();
                let term2 = R::from_i64(4) * b.clone() * b.clone() * b.clone() * d.clone();
                let term3 = b.clone() * b.clone() * c.clone() * c.clone();
                let term4 = R::from_i64(4) * a.clone() * c.clone() * c.clone() * c.clone();
                let term5 = R::from_i64(27) * a.clone() * a.clone() * d.clone() * d.clone();

                Some(term1 - term2 + term3 - term4 - term5)
            }
            _ => None,
        }
    }

    /// Check if this polynomial is monic (leading coefficient is 1)
    pub fn is_monic(&self) -> bool {
        if let Some(lc) = self.leading_coeff() {
            lc.is_one()
        } else {
            false
        }
    }

    /// Get the content of the polynomial (GCD of all coefficients)
    ///
    /// Only works for coefficients in a Euclidean domain
    pub fn content(&self) -> R
    where
        R: EuclideanDomain,
    {
        if self.coeffs.is_empty() {
            return R::zero();
        }

        let mut gcd = self.coeffs[0].clone();
        for coeff in &self.coeffs[1..] {
            gcd = EuclideanDomain::gcd(&gcd, coeff);
            if gcd.is_one() {
                break;
            }
        }
        gcd
    }

    /// Construct the Sylvester matrix of two polynomials
    ///
    /// For polynomials f of degree m and g of degree n, the Sylvester matrix
    /// is an (m+n) × (m+n) matrix whose determinant is the resultant of f and g.
    ///
    /// Returns the matrix as a vector of rows (Vec<Vec<R>>)
    pub fn sylvester_matrix(&self, other: &Self) -> Vec<Vec<R>> {
        let m = self.degree().unwrap_or(0);
        let n = other.degree().unwrap_or(0);
        let size = m + n;

        let mut matrix = vec![vec![R::zero(); size]; size];

        // First n rows: shifted coefficients of self
        for i in 0..n {
            for j in 0..=m {
                matrix[i][i + j] = self.coeffs.get(m - j).cloned().unwrap_or_else(|| R::zero());
            }
        }

        // Last m rows: shifted coefficients of other
        for i in 0..m {
            for j in 0..=n {
                matrix[n + i][i + j] = other.coeffs.get(n - j).cloned().unwrap_or_else(|| R::zero());
            }
        }

        matrix
    }

    /// Compute the resultant of two polynomials
    ///
    /// The resultant is the determinant of the Sylvester matrix.
    /// It is zero if and only if the polynomials have a common root (over an algebraically closed field).
    ///
    /// # Limitations
    ///
    /// This naive implementation computes the determinant using expansion by minors,
    /// which is O(n!) and only practical for small polynomials (degree < 10).
    /// For larger polynomials, more efficient algorithms should be used.
    pub fn resultant(&self, other: &Self) -> R
    where
        R: rustmath_core::NumericConversion,
    {
        if self.is_zero() || other.is_zero() {
            return R::zero();
        }

        let matrix = self.sylvester_matrix(other);
        Self::determinant_helper(&matrix)
    }

    /// Helper function to compute determinant recursively
    fn determinant_helper(matrix: &[Vec<R>]) -> R
    where
        R: rustmath_core::NumericConversion,
    {
        let n = matrix.len();

        if n == 0 {
            return R::zero();
        }

        if n == 1 {
            return matrix[0][0].clone();
        }

        if n == 2 {
            return matrix[0][0].clone() * matrix[1][1].clone()
                - matrix[0][1].clone() * matrix[1][0].clone();
        }

        // Expansion by first row
        let mut det = R::zero();
        let mut sign = R::one();

        for j in 0..n {
            if !matrix[0][j].is_zero() {
                // Create minor by removing first row and j-th column
                let mut minor = Vec::with_capacity(n - 1);
                for i in 1..n {
                    let mut row = Vec::with_capacity(n - 1);
                    for k in 0..n {
                        if k != j {
                            row.push(matrix[i][k].clone());
                        }
                    }
                    minor.push(row);
                }

                let cofactor = sign.clone() * matrix[0][j].clone() * Self::determinant_helper(&minor);
                det = det + cofactor;
            }

            sign = R::zero() - sign; // Flip sign
        }

        det
    }
}

impl<R: Ring> Polynomial for UnivariatePolynomial<R> {
    type Coeff = R;
    type Var = ();

    fn from_coeffs(coeffs: Vec<R>) -> Self {
        UnivariatePolynomial::new(coeffs)
    }

    fn degree(&self) -> Option<usize> {
        if self.coeffs.len() == 1 && self.coeffs[0].is_zero() {
            None
        } else {
            Some(self.coeffs.len() - 1)
        }
    }

    fn eval(&self, point: &R) -> R {
        // Horner's method for evaluation
        if self.coeffs.is_empty() {
            return R::zero();
        }

        let mut result = self.coeffs.last().unwrap().clone();
        for coeff in self.coeffs.iter().rev().skip(1) {
            result = result * point.clone() + coeff.clone();
        }

        result
    }

    fn coeff(&self, degree: usize) -> &R {
        self.coeffs.get(degree).unwrap_or(&self.coeffs[0])
    }
}

impl<R: Ring> fmt::Display for UnivariatePolynomial<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut first = true;
        for (i, coeff) in self.coeffs.iter().enumerate().rev() {
            if coeff.is_zero() {
                continue;
            }

            if !first {
                write!(f, " + ")?;
            }
            first = false;

            if i == 0 {
                write!(f, "{}", coeff)?;
            } else if i == 1 {
                if coeff.is_one() {
                    write!(f, "x")?;
                } else {
                    write!(f, "{}*x", coeff)?;
                }
            } else if coeff.is_one() {
                write!(f, "x^{}", i)?;
            } else {
                write!(f, "{}*x^{}", coeff, i)?;
            }
        }

        Ok(())
    }
}

impl<R: Ring> fmt::Debug for UnivariatePolynomial<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Poly({:?})", self.coeffs)
    }
}

impl<R: Ring> Add for UnivariatePolynomial<R> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let max_len = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let a = self.coeffs.get(i).cloned().unwrap_or_else(|| R::zero());
            let b = other.coeffs.get(i).cloned().unwrap_or_else(|| R::zero());
            coeffs.push(a + b);
        }

        UnivariatePolynomial::new(coeffs)
    }
}

impl<R: Ring> Sub for UnivariatePolynomial<R> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let max_len = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let a = self.coeffs.get(i).cloned().unwrap_or_else(|| R::zero());
            let b = other.coeffs.get(i).cloned().unwrap_or_else(|| R::zero());
            coeffs.push(a - b);
        }

        UnivariatePolynomial::new(coeffs)
    }
}

impl<R: Ring> Mul for UnivariatePolynomial<R> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return UnivariatePolynomial::new(vec![R::zero()]);
        }

        let result_len = self.coeffs.len() + other.coeffs.len() - 1;
        let mut coeffs = vec![R::zero(); result_len];

        for (i, a) in self.coeffs.iter().enumerate() {
            for (j, b) in other.coeffs.iter().enumerate() {
                coeffs[i + j] = coeffs[i + j].clone() + a.clone() * b.clone();
            }
        }

        UnivariatePolynomial::new(coeffs)
    }
}

impl<R: Ring> Neg for UnivariatePolynomial<R> {
    type Output = Self;

    fn neg(self) -> Self {
        let coeffs = self.coeffs.into_iter().map(|c| -c).collect();
        UnivariatePolynomial::new(coeffs)
    }
}

// Division for polynomials over fields
impl<R: Ring + EuclideanDomain> Div for UnivariatePolynomial<R> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        self.div_rem(&other).unwrap().0
    }
}

impl<R: Ring + EuclideanDomain> Rem for UnivariatePolynomial<R> {
    type Output = Self;

    fn rem(self, other: Self) -> Self {
        self.div_rem(&other).unwrap().1
    }
}

impl<R: Ring> UnivariatePolynomial<R> {
    /// Division with remainder (for polynomials over fields/Euclidean domains)
    pub fn div_rem(&self, divisor: &Self) -> Result<(Self, Self)>
    where
        R: EuclideanDomain,
    {
        if divisor.is_zero() {
            return Err(MathError::DivisionByZero);
        }

        if self.is_zero() {
            return Ok((
                UnivariatePolynomial::new(vec![R::zero()]),
                UnivariatePolynomial::new(vec![R::zero()]),
            ));
        }

        let dividend_deg = self.degree().unwrap();
        let divisor_deg = divisor.degree().unwrap();

        if dividend_deg < divisor_deg {
            return Ok((UnivariatePolynomial::new(vec![R::zero()]), self.clone()));
        }

        let mut quotient = vec![R::zero(); dividend_deg - divisor_deg + 1];
        let mut remainder = self.clone();

        let divisor_lc = divisor.leading_coeff().unwrap().clone();

        while !remainder.is_zero() {
            let remainder_deg = match remainder.degree() {
                Some(d) => d,
                None => break,
            };

            if remainder_deg < divisor_deg {
                break;
            }

            let remainder_lc = remainder.leading_coeff().unwrap().clone();
            let (coeff_quot, _) = remainder_lc.div_rem(&divisor_lc)?;

            let shift = remainder_deg - divisor_deg;
            quotient[shift] = coeff_quot.clone();

            let mut subtrahend_coeffs = vec![R::zero(); shift];
            subtrahend_coeffs.extend(divisor.coeffs.iter().map(|c| c.clone() * coeff_quot.clone()));

            let subtrahend = UnivariatePolynomial::new(subtrahend_coeffs);
            remainder = remainder - subtrahend;
        }

        Ok((
            UnivariatePolynomial::new(quotient),
            remainder,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_creation() {
        let p = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(2),
            Integer::from(3),
        ]);

        assert_eq!(p.degree(), Some(2));
        assert_eq!(p.coeff(0), &Integer::from(1));
        assert_eq!(p.coeff(1), &Integer::from(2));
        assert_eq!(p.coeff(2), &Integer::from(3));
    }

    #[test]
    fn test_eval() {
        // p(x) = 1 + 2x + 3x^2
        let p = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(2),
            Integer::from(3),
        ]);

        // p(2) = 1 + 4 + 12 = 17
        assert_eq!(p.eval(&Integer::from(2)), Integer::from(17));
    }

    #[test]
    fn test_addition() {
        let p1 = UnivariatePolynomial::new(vec![Integer::from(1), Integer::from(2)]);
        let p2 = UnivariatePolynomial::new(vec![Integer::from(3), Integer::from(4)]);

        let sum = p1 + p2;
        assert_eq!(sum.coefficients(), &[Integer::from(4), Integer::from(6)]);
    }

    #[test]
    fn test_multiplication() {
        // (1 + x) * (1 + x) = 1 + 2x + x^2
        let p = UnivariatePolynomial::new(vec![Integer::from(1), Integer::from(1)]);
        let prod = p.clone() * p;

        assert_eq!(
            prod.coefficients(),
            &[Integer::from(1), Integer::from(2), Integer::from(1)]
        );
    }

    #[test]
    fn test_derivative() {
        // p(x) = 1 + 2x + 3x^2
        // p'(x) = 2 + 6x
        let p = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(2),
            Integer::from(3),
        ]);

        let deriv = p.derivative();
        assert_eq!(deriv.coefficients(), &[Integer::from(2), Integer::from(6)]);
    }

    #[test]
    fn test_discriminant() {
        // Quadratic: x^2 - 5x + 6 = (x-2)(x-3)
        // Discriminant = b^2 - 4ac = 25 - 24 = 1
        let p = UnivariatePolynomial::new(vec![
            Integer::from(6),
            Integer::from(-5),
            Integer::from(1),
        ]);
        assert_eq!(p.discriminant(), Some(Integer::from(1)));

        // Quadratic: x^2 + 1 (no real roots)
        // Discriminant = 0 - 4 = -4
        let p = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(0),
            Integer::from(1),
        ]);
        assert_eq!(p.discriminant(), Some(Integer::from(-4)));

        // Linear polynomial: discriminant = 1
        let p = UnivariatePolynomial::new(vec![
            Integer::from(2),
            Integer::from(3),
        ]);
        assert_eq!(p.discriminant(), Some(Integer::from(1)));
    }

    #[test]
    fn test_is_monic() {
        let p = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(2),
            Integer::from(1), // Leading coefficient
        ]);
        assert!(p.is_monic());

        let p = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(2),
            Integer::from(3), // Leading coefficient
        ]);
        assert!(!p.is_monic());
    }

    #[test]
    fn test_content() {
        // 6 + 9x + 12x^2, content = gcd(6, 9, 12) = 3
        let p = UnivariatePolynomial::new(vec![
            Integer::from(6),
            Integer::from(9),
            Integer::from(12),
        ]);
        assert_eq!(p.content(), Integer::from(3));

        // 2 + 4x, content = 2
        let p = UnivariatePolynomial::new(vec![
            Integer::from(2),
            Integer::from(4),
        ]);
        assert_eq!(p.content(), Integer::from(2));
    }

    #[test]
    fn test_compose() {
        // p(x) = x + 1, q(x) = 2x
        // p(q(x)) = 2x + 1
        let p = UnivariatePolynomial::new(vec![Integer::from(1), Integer::from(1)]);
        let q = UnivariatePolynomial::new(vec![Integer::from(0), Integer::from(2)]);

        let composed = p.compose(&q);
        assert_eq!(composed.coefficients(), &[Integer::from(1), Integer::from(2)]);

        // p(x) = x^2, q(x) = x + 1
        // p(q(x)) = (x+1)^2 = x^2 + 2x + 1
        let p = UnivariatePolynomial::new(vec![Integer::from(0), Integer::from(0), Integer::from(1)]);
        let q = UnivariatePolynomial::new(vec![Integer::from(1), Integer::from(1)]);

        let composed = p.compose(&q);
        assert_eq!(
            composed.coefficients(),
            &[Integer::from(1), Integer::from(2), Integer::from(1)]
        );
    }

    #[test]
    fn test_scale_variable() {
        // p(x) = x^2 + 2x + 3
        // p(2x) = 4x^2 + 4x + 3
        let p = UnivariatePolynomial::new(vec![
            Integer::from(3),
            Integer::from(2),
            Integer::from(1),
        ]);

        let scaled = p.scale_variable(&Integer::from(2));
        assert_eq!(
            scaled.coefficients(),
            &[Integer::from(3), Integer::from(4), Integer::from(4)]
        );
    }

    #[test]
    fn test_translate() {
        // p(x) = x^2, p(x + 1) = x^2 + 2x + 1
        let p = UnivariatePolynomial::new(vec![Integer::from(0), Integer::from(0), Integer::from(1)]);
        let translated = p.translate(&Integer::from(1));

        assert_eq!(
            translated.coefficients(),
            &[Integer::from(1), Integer::from(2), Integer::from(1)]
        );
    }
}
