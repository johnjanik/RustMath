//! Algebraic Number Fields
//!
//! This module implements algebraic number fields Q(α) where α is a root of
//! an irreducible polynomial over Q. Number fields are fundamental objects
//! in algebraic number theory.
//!
//! # Examples
//!
//! ```
//! use rustmath_numberfields::NumberField;
//! use rustmath_polynomials::univariate::UnivariatePolynomial;
//! use rustmath_rationals::Rational;
//!
//! // Create Q(√2) using minimal polynomial x^2 - 2
//! let poly = UnivariatePolynomial::new(vec![
//!     Rational::from_integer(-2),  // constant term
//!     Rational::from_integer(0),   // x coefficient
//!     Rational::from_integer(1),   // x^2 coefficient
//! ]);
//!
//! let field = NumberField::new(poly);
//! ```

use rustmath_core::{EuclideanDomain, Ring};
use rustmath_polynomials::univariate::UnivariatePolynomial;
use rustmath_rationals::Rational;
use std::fmt;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum NumberFieldError {
    #[error("Polynomial must be irreducible")]
    ReduciblePolynomial,
    #[error("Polynomial degree must be at least 1")]
    InvalidDegree,
    #[error("Element does not belong to this field")]
    InvalidElement,
}

/// An element of a number field, represented as a polynomial in the generator
#[derive(Clone, Debug)]
pub struct NumberFieldElement {
    /// Coefficients of the element as a polynomial in the generator
    /// If α is the generator, this represents c₀ + c₁α + c₂α² + ... + cₙ₋₁αⁿ⁻¹
    coeffs: Vec<Rational>,
    /// Reference to the defining polynomial (minimal polynomial)
    min_poly: UnivariatePolynomial<Rational>,
}

impl NumberFieldElement {
    /// Create a new number field element from coefficients
    pub fn new(coeffs: Vec<Rational>, min_poly: UnivariatePolynomial<Rational>) -> Self {
        let mut elem = NumberFieldElement { coeffs, min_poly };
        elem.reduce();
        elem
    }

    /// Reduce this element modulo the minimal polynomial
    fn reduce(&mut self) {
        if self.coeffs.len() >= self.min_poly.degree().unwrap_or(1) {
            let poly = UnivariatePolynomial::new(self.coeffs.clone());
            let (_, remainder) = poly.quo_rem(&self.min_poly);
            // Extract coefficients directly from the remainder polynomial
            self.coeffs = remainder.coefficients().to_vec();
        }
        // Remove trailing zeros
        while self.coeffs.len() > 1 && self.coeffs.last() == Some(&Rational::zero()) {
            self.coeffs.pop();
        }
    }

    /// Get the degree of this element (highest non-zero coefficient)
    pub fn degree(&self) -> usize {
        self.coeffs.len().saturating_sub(1)
    }

    /// Get coefficient at given position
    pub fn coeff(&self, i: usize) -> Rational {
        self.coeffs.get(i).cloned().unwrap_or_else(Rational::zero)
    }

    /// Compute the norm of this element
    /// The norm is the product of all conjugates
    pub fn norm(&self) -> Rational {
        // For now, use resultant method
        let poly = UnivariatePolynomial::new(self.coeffs.clone());
        let res = poly.resultant(&self.min_poly);
        let deg = self.min_poly.degree().unwrap_or(1);
        let sign = if deg % 2 == 0 { 1 } else { -1 };
        res * Rational::from_integer(sign)
    }

    /// Compute the trace of this element
    /// The trace is the sum of all conjugates
    pub fn trace(&self) -> Rational {
        // For linear elements, trace is degree times the coefficient
        if self.coeffs.is_empty() {
            return Rational::zero();
        }
        // General case: use the fact that trace(1) = n, trace(α) = -a_{n-1}/a_n
        // where the minimal polynomial is a_n x^n + a_{n-1} x^{n-1} + ...
        // This is a simplified version; full implementation requires more theory
        let n = self.min_poly.degree().unwrap_or(1);
        if self.coeffs.len() == 1 {
            // Rational element: trace is n times the element
            self.coeffs[0].clone() * Rational::from_integer(n as i64)
        } else {
            // For now, return 0 for non-rational elements
            // Full implementation requires computing characteristic polynomial
            Rational::zero()
        }
    }
}

impl PartialEq for NumberFieldElement {
    fn eq(&self, other: &Self) -> bool {
        if self.min_poly != other.min_poly {
            return false;
        }
        self.coeffs == other.coeffs
    }
}

impl Eq for NumberFieldElement {}

impl fmt::Display for NumberFieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.coeffs.is_empty() || (self.coeffs.len() == 1 && self.coeffs[0].is_zero()) {
            return write!(f, "0");
        }

        let mut first = true;
        for (i, c) in self.coeffs.iter().enumerate() {
            if c.is_zero() {
                continue;
            }

            if !first {
                write!(f, " + ")?;
            }
            first = false;

            if i == 0 {
                write!(f, "{}", c)?;
            } else if c == &Rational::one() {
                if i == 1 {
                    write!(f, "α")?;
                } else {
                    write!(f, "α^{}", i)?;
                }
            } else {
                if i == 1 {
                    write!(f, "{}*α", c)?;
                } else {
                    write!(f, "{}*α^{}", c, i)?;
                }
            }
        }
        Ok(())
    }
}

/// A number field Q(α) where α is a root of an irreducible polynomial
#[derive(Clone, Debug)]
pub struct NumberField {
    /// The minimal polynomial defining the field
    minimal_polynomial: UnivariatePolynomial<Rational>,
    /// Degree of the field extension [Q(α) : Q]
    degree: usize,
}

impl NumberField {
    /// Create a new number field from a minimal polynomial
    ///
    /// The polynomial must be irreducible over Q.
    pub fn new(minimal_polynomial: UnivariatePolynomial<Rational>) -> Self {
        let degree = minimal_polynomial.degree().expect("Polynomial must be non-zero");
        if degree == 0 {
            panic!("Minimal polynomial must have degree at least 1");
        }
        // Note: We don't check irreducibility here for performance reasons
        // In production code, this should be verified
        NumberField {
            minimal_polynomial,
            degree,
        }
    }

    /// Get the degree of this number field
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Get the minimal polynomial
    pub fn minimal_polynomial(&self) -> &UnivariatePolynomial<Rational> {
        &self.minimal_polynomial
    }

    /// Create the zero element of this field
    pub fn zero(&self) -> NumberFieldElement {
        NumberFieldElement::new(vec![Rational::zero()], self.minimal_polynomial.clone())
    }

    /// Create the one element of this field
    pub fn one(&self) -> NumberFieldElement {
        NumberFieldElement::new(vec![Rational::one()], self.minimal_polynomial.clone())
    }

    /// Create an element from a rational number
    pub fn from_rational(&self, r: Rational) -> NumberFieldElement {
        NumberFieldElement::new(vec![r], self.minimal_polynomial.clone())
    }

    /// Create the generator α of this field (a root of the minimal polynomial)
    pub fn generator(&self) -> NumberFieldElement {
        NumberFieldElement::new(
            vec![Rational::zero(), Rational::one()],
            self.minimal_polynomial.clone(),
        )
    }

    /// Add two elements
    pub fn add(&self, a: &NumberFieldElement, b: &NumberFieldElement) -> NumberFieldElement {
        let max_len = a.coeffs.len().max(b.coeffs.len());
        let mut coeffs = Vec::with_capacity(max_len);
        for i in 0..max_len {
            coeffs.push(a.coeff(i) + b.coeff(i));
        }
        NumberFieldElement::new(coeffs, self.minimal_polynomial.clone())
    }

    /// Subtract two elements
    pub fn sub(&self, a: &NumberFieldElement, b: &NumberFieldElement) -> NumberFieldElement {
        let max_len = a.coeffs.len().max(b.coeffs.len());
        let mut coeffs = Vec::with_capacity(max_len);
        for i in 0..max_len {
            coeffs.push(a.coeff(i) - b.coeff(i));
        }
        NumberFieldElement::new(coeffs, self.minimal_polynomial.clone())
    }

    /// Multiply two elements
    pub fn mul(&self, a: &NumberFieldElement, b: &NumberFieldElement) -> NumberFieldElement {
        let mut coeffs = vec![Rational::zero(); a.coeffs.len() + b.coeffs.len()];
        for (i, c1) in a.coeffs.iter().enumerate() {
            for (j, c2) in b.coeffs.iter().enumerate() {
                coeffs[i + j] = coeffs[i + j].clone() + c1.clone() * c2.clone();
            }
        }
        NumberFieldElement::new(coeffs, self.minimal_polynomial.clone())
    }

    /// Compute multiplicative inverse
    ///
    /// For rational elements (constant polynomials), this is straightforward.
    /// For general elements, this uses trial and error to find coefficients.
    /// A full implementation would use the extended Euclidean algorithm.
    pub fn inv(&self, a: &NumberFieldElement) -> Result<NumberFieldElement, NumberFieldError> {
        if a.coeffs.is_empty() || a.coeffs.iter().all(|c| c.is_zero()) {
            return Err(NumberFieldError::InvalidElement);
        }

        // Special case: if a is a rational (constant), just invert it
        if a.coeffs.len() == 1 || a.coeffs[1..].iter().all(|c| c.is_zero()) {
            let rational_val = &a.coeffs[0];
            if rational_val.is_zero() {
                return Err(NumberFieldError::InvalidElement);
            }
            let inv = Rational::one() / rational_val.clone();
            return Ok(NumberFieldElement::new(
                vec![inv],
                self.minimal_polynomial.clone(),
            ));
        }

        // For non-rational elements, we would need extended GCD
        // which is not currently implemented for univariate polynomials
        // TODO: Implement extended GCD for univariate polynomials
        Err(NumberFieldError::InvalidElement)
    }

    /// Divide two elements
    pub fn div(&self, a: &NumberFieldElement, b: &NumberFieldElement) -> Result<NumberFieldElement, NumberFieldError> {
        let b_inv = self.inv(b)?;
        Ok(self.mul(a, &b_inv))
    }

    /// Compute the discriminant of the number field
    ///
    /// The discriminant is the determinant of the trace matrix
    pub fn discriminant(&self) -> Rational {
        // Compute the discriminant of the minimal polynomial
        // disc(f) = (-1)^(n(n-1)/2) * res(f, f') / leading_coeff(f)
        let f = &self.minimal_polynomial;
        let f_prime = f.derivative();
        let res = f.resultant(&f_prime);

        let n = self.degree;
        let sign_exp = n * (n - 1) / 2;
        let sign = if sign_exp % 2 == 0 {
            Rational::one()
        } else {
            -Rational::one()
        };

        let leading_coeff = f.coeff(n).clone();
        sign * res / leading_coeff
    }

    /// Get the integral basis of the ring of integers
    ///
    /// For now, returns the power basis {1, α, α², ..., α^(n-1)}
    /// Computing the actual integral basis requires more sophisticated algorithms
    pub fn power_basis(&self) -> Vec<NumberFieldElement> {
        (0..self.degree)
            .map(|i| {
                let mut coeffs = vec![Rational::zero(); i + 1];
                coeffs[i] = Rational::one();
                NumberFieldElement::new(coeffs, self.minimal_polynomial.clone())
            })
            .collect()
    }
}

impl fmt::Display for NumberField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Q(α) where α satisfies {}", self.minimal_polynomial)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_x_squared_minus_2() -> UnivariatePolynomial<Rational> {
        UnivariatePolynomial::new(vec![
            Rational::from_integer(-2), // constant term -2
            Rational::from_integer(0),  // x coefficient
            Rational::from_integer(1),  // x^2 coefficient
        ])
    }

    fn make_x_cubed_minus_2() -> UnivariatePolynomial<Rational> {
        UnivariatePolynomial::new(vec![
            Rational::from_integer(-2), // constant term -2
            Rational::from_integer(0),  // x coefficient
            Rational::from_integer(0),  // x^2 coefficient
            Rational::from_integer(1),  // x^3 coefficient
        ])
    }

    #[test]
    fn test_create_number_field() {
        let poly = make_x_squared_minus_2();
        let field = NumberField::new(poly);
        assert_eq!(field.degree(), 2);
    }

    #[test]
    fn test_zero_and_one() {
        let poly = make_x_squared_minus_2();
        let field = NumberField::new(poly);

        let zero = field.zero();
        let one = field.one();

        assert_eq!(zero.coeff(0), Rational::zero());
        assert_eq!(one.coeff(0), Rational::one());
    }

    #[test]
    fn test_generator() {
        let poly = make_x_squared_minus_2();
        let field = NumberField::new(poly);

        let alpha = field.generator();
        assert_eq!(alpha.coeff(0), Rational::zero());
        assert_eq!(alpha.coeff(1), Rational::one());
    }

    #[test]
    fn test_addition() {
        let poly = make_x_squared_minus_2();
        let field = NumberField::new(poly);

        let one = field.one();
        let alpha = field.generator();

        // 1 + α
        let sum = field.add(&one, &alpha);
        assert_eq!(sum.coeff(0), Rational::one());
        assert_eq!(sum.coeff(1), Rational::one());
    }

    #[test]
    fn test_multiplication() {
        let poly = make_x_squared_minus_2();
        let field = NumberField::new(poly);

        let alpha = field.generator();

        // α * α = α² = 2 (since α² - 2 = 0)
        let product = field.mul(&alpha, &alpha);
        assert_eq!(product.coeff(0), Rational::from_integer(2));
        assert_eq!(product.coeff(1), Rational::zero());
    }

    #[test]
    fn test_multiplication_reduction() {
        let poly = make_x_cubed_minus_2();
        let field = NumberField::new(poly);

        let alpha = field.generator();

        // α³ = 2 (since α³ - 2 = 0)
        let alpha2 = field.mul(&alpha, &alpha);
        let alpha3 = field.mul(&alpha2, &alpha);

        assert_eq!(alpha3.coeff(0), Rational::from_integer(2));
        assert_eq!(alpha3.coeff(1), Rational::zero());
        assert_eq!(alpha3.coeff(2), Rational::zero());
    }

    #[test]
    fn test_subtraction() {
        let poly = make_x_squared_minus_2();
        let field = NumberField::new(poly);

        let one = field.one();
        let alpha = field.generator();

        // α - 1
        let diff = field.sub(&alpha, &one);
        assert_eq!(diff.coeff(0), -Rational::one());
        assert_eq!(diff.coeff(1), Rational::one());
    }

    #[test]
    fn test_from_rational() {
        let poly = make_x_squared_minus_2();
        let field = NumberField::new(poly);

        let three = field.from_rational(Rational::from_integer(3));
        assert_eq!(three.coeff(0), Rational::from_integer(3));
        assert_eq!(three.coeff(1), Rational::zero());
    }

    #[test]
    fn test_discriminant() {
        let poly = make_x_squared_minus_2();
        let field = NumberField::new(poly);

        // For x^2 - 2, discriminant should be 8
        let disc = field.discriminant();
        assert_eq!(disc, Rational::from_integer(8));
    }

    #[test]
    fn test_power_basis() {
        let poly = make_x_squared_minus_2();
        let field = NumberField::new(poly);

        let basis = field.power_basis();
        assert_eq!(basis.len(), 2);

        // First element should be 1
        assert_eq!(basis[0].coeff(0), Rational::one());
        assert_eq!(basis[0].coeff(1), Rational::zero());

        // Second element should be α
        assert_eq!(basis[1].coeff(0), Rational::zero());
        assert_eq!(basis[1].coeff(1), Rational::one());
    }

    #[test]
    fn test_element_norm() {
        let poly = make_x_squared_minus_2();
        let field = NumberField::new(poly);

        let two = field.from_rational(Rational::from_integer(2));
        let norm = two.norm();

        // Norm of 2 in Q(√2) should be 4
        assert_eq!(norm, Rational::from_integer(4));
    }

    #[test]
    fn test_element_display() {
        let poly = make_x_squared_minus_2();
        let field = NumberField::new(poly);

        let alpha = field.generator();
        let display = format!("{}", alpha);
        assert_eq!(display, "α");

        let one_plus_alpha = field.add(&field.one(), &alpha);
        let display = format!("{}", one_plus_alpha);
        assert_eq!(display, "1 + α");
    }

    #[test]
    fn test_inverse() {
        let poly = make_x_squared_minus_2();
        let field = NumberField::new(poly);

        // Test inverse of 2
        let two = field.from_rational(Rational::from_integer(2));
        let two_inv = field.inv(&two).unwrap();
        assert_eq!(two_inv.coeff(0), Rational::new(1, 2).unwrap());

        // Verify that 2 * (1/2) = 1
        let product = field.mul(&two, &two_inv);
        assert_eq!(product.coeff(0), Rational::one());
    }

    #[test]
    fn test_division() {
        let poly = make_x_squared_minus_2();
        let field = NumberField::new(poly);

        let four = field.from_rational(Rational::from_integer(4));
        let two = field.from_rational(Rational::from_integer(2));

        // 4 / 2 = 2
        let quotient = field.div(&four, &two).unwrap();
        assert_eq!(quotient.coeff(0), Rational::from_integer(2));
    }
}
