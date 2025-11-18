//! Rational function field elements
//!
//! This module provides element types for rational function fields k(x) where k is a field.
//! Corresponds to SageMath's `sage.rings.function_field.element_rational`.
//!
//! # Mathematical Background
//!
//! A rational function field k(x) is the field of fractions of the polynomial ring k[x].
//! Elements are rational functions f(x)/g(x) where f, g ∈ k[x] and g ≠ 0.
//!
//! This is the simplest type of function field - a purely transcendental extension of
//! the base field k.
//!
//! # Examples
//!
//! ```ignore
//! use rustmath_rings::function_field_element_rational::*;
//! use rustmath_rationals::Rational;
//!
//! // Create a rational function (x^2 + 1)/(x - 1)
//! let elem = FunctionFieldElement_rational::new(
//!     vec![Rational::from(1), Rational::from(0), Rational::from(1)],  // x^2 + 1
//!     vec![Rational::from(-1), Rational::from(1)],  // x - 1
//! );
//! ```

use rustmath_core::{Ring, Field};
use std::fmt;
use std::marker::PhantomData;

/// Element of a rational function field k(x)
///
/// Represents a rational function f(x)/g(x) where f and g are polynomials over the
/// base field k. This corresponds to SageMath's FunctionFieldElement_rational.
///
/// # Type Parameters
///
/// - `F`: The base field (e.g., ℚ, ℝ, or a finite field)
///
/// # Implementation Note
///
/// Polynomials are represented as coefficient vectors in increasing degree order:
/// [a₀, a₁, a₂, ...] represents a₀ + a₁x + a₂x² + ...
#[derive(Clone, Debug)]
pub struct FunctionFieldElement_rational<F: Field> {
    /// Numerator polynomial coefficients
    numerator: Vec<F>,
    /// Denominator polynomial coefficients
    denominator: Vec<F>,
}

impl<F: Field> FunctionFieldElement_rational<F> {
    /// Create a new rational function element
    ///
    /// # Arguments
    ///
    /// * `num` - Numerator polynomial coefficients [a₀, a₁, ...]
    /// * `den` - Denominator polynomial coefficients [b₀, b₁, ...]
    ///
    /// # Panics
    ///
    /// Panics if denominator is the zero polynomial
    pub fn new(num: Vec<F>, den: Vec<F>) -> Self {
        if den.is_empty() || den.iter().all(|c| c.is_zero()) {
            panic!("Denominator cannot be zero polynomial");
        }

        FunctionFieldElement_rational {
            numerator: num,
            denominator: den,
        }
    }

    /// Create the zero element (0/1)
    pub fn zero() -> Self {
        FunctionFieldElement_rational {
            numerator: vec![F::zero()],
            denominator: vec![F::one()],
        }
    }

    /// Create the one element (1/1)
    pub fn one() -> Self {
        FunctionFieldElement_rational {
            numerator: vec![F::one()],
            denominator: vec![F::one()],
        }
    }

    /// Create the variable x (x/1)
    pub fn variable() -> Self {
        FunctionFieldElement_rational {
            numerator: vec![F::zero(), F::one()],  // 0 + 1*x
            denominator: vec![F::one()],
        }
    }

    /// Check if this element is zero
    pub fn is_zero(&self) -> bool {
        self.numerator.is_empty() || self.numerator.iter().all(|c| c.is_zero())
    }

    /// Check if this element is one
    pub fn is_one(&self) -> bool {
        // Both numerator and denominator should be the constant 1
        self.numerator.len() == 1
            && self.denominator.len() == 1
            && self.numerator[0].is_one()
            && self.denominator[0].is_one()
    }

    /// Get the numerator polynomial
    pub fn numerator(&self) -> &Vec<F> {
        &self.numerator
    }

    /// Get the denominator polynomial
    pub fn denominator(&self) -> &Vec<F> {
        &self.denominator
    }

    /// Get the degree of the numerator
    pub fn numerator_degree(&self) -> usize {
        if self.is_zero() {
            0
        } else {
            self.numerator.len().saturating_sub(1)
        }
    }

    /// Get the degree of the denominator
    pub fn denominator_degree(&self) -> usize {
        self.denominator.len().saturating_sub(1)
    }

    /// Compute the inverse of this rational function
    ///
    /// Returns 1/f = g(x)/f(x) if f ≠ 0, None if f = 0.
    pub fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            Some(FunctionFieldElement_rational {
                numerator: self.denominator.clone(),
                denominator: self.numerator.clone(),
            })
        }
    }

    /// Evaluate this rational function at a point
    ///
    /// Compute f(a) for a given field element a.
    pub fn evaluate(&self, point: &F) -> Option<F> {
        let num_val = self.eval_poly(&self.numerator, point);
        let den_val = self.eval_poly(&self.denominator, point);

        if den_val.is_zero() {
            None  // Division by zero
        } else {
            // Would need division in F, which requires F to have a division operation
            // For now, return None as placeholder
            None
        }
    }

    /// Helper: evaluate a polynomial at a point
    fn eval_poly(&self, coeffs: &[F], point: &F) -> F {
        if coeffs.is_empty() {
            return F::zero();
        }

        let mut result = F::zero();
        let mut power = F::one();

        for coeff in coeffs.iter() {
            result = result + coeff.clone() * power.clone();
            power = power * point.clone();
        }

        result
    }

    /// Compute the derivative d/dx of this rational function
    ///
    /// Uses the quotient rule: (f/g)' = (f'g - fg')/g²
    pub fn derivative(&self) -> Self {
        let f_prime = Self::poly_derivative(&self.numerator);
        let g_prime = Self::poly_derivative(&self.denominator);

        // f'g
        let f_prime_g = Self::poly_mul(&f_prime, &self.denominator);
        // fg'
        let f_g_prime = Self::poly_mul(&self.numerator, &g_prime);
        // f'g - fg'
        let numerator = Self::poly_sub(&f_prime_g, &f_g_prime);
        // g²
        let denominator = Self::poly_mul(&self.denominator, &self.denominator);

        FunctionFieldElement_rational {
            numerator,
            denominator,
        }
    }

    /// Compute derivative of a polynomial
    fn poly_derivative(coeffs: &[F]) -> Vec<F> {
        if coeffs.len() <= 1 {
            return vec![F::zero()];
        }

        let mut result = Vec::new();
        for (i, coeff) in coeffs.iter().enumerate().skip(1) {
            // d/dx(x^i) = i*x^(i-1)
            let mut term = coeff.clone();
            for _ in 0..i {
                term = term + coeff.clone();
            }
            result.push(term);
        }

        if result.is_empty() {
            vec![F::zero()]
        } else {
            result
        }
    }

    /// Multiply two polynomials
    fn poly_mul(a: &[F], b: &[F]) -> Vec<F> {
        if a.is_empty() || b.is_empty() {
            return vec![F::zero()];
        }

        let mut result = vec![F::zero(); a.len() + b.len() - 1];

        for (i, ai) in a.iter().enumerate() {
            for (j, bj) in b.iter().enumerate() {
                result[i + j] = result[i + j].clone() + (ai.clone() * bj.clone());
            }
        }

        result
    }

    /// Subtract two polynomials
    fn poly_sub(a: &[F], b: &[F]) -> Vec<F> {
        let max_len = a.len().max(b.len());
        let mut result = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let ai = a.get(i).cloned().unwrap_or_else(F::zero);
            let bi = b.get(i).cloned().unwrap_or_else(F::zero);
            result.push(ai - bi);
        }

        result
    }

    /// Check if this rational function is integral (i.e., is actually a polynomial)
    ///
    /// A rational function is integral if its denominator is a non-zero constant.
    pub fn is_integral(&self) -> bool {
        self.denominator.len() == 1 && !self.denominator[0].is_zero()
    }

    /// Get the pole and zero divisor information
    ///
    /// Returns (zeros, poles) where zeros come from the numerator and poles from denominator.
    pub fn divisor_info(&self) -> (usize, usize) {
        (self.numerator_degree(), self.denominator_degree())
    }
}

impl<F: Field> PartialEq for FunctionFieldElement_rational<F> {
    fn eq(&self, other: &Self) -> bool {
        // Cross multiply: f/g == h/k iff fk == gh
        let lhs = Self::poly_mul(&self.numerator, &other.denominator);
        let rhs = Self::poly_mul(&other.numerator, &self.denominator);

        if lhs.len() != rhs.len() {
            return false;
        }

        lhs.iter().zip(rhs.iter()).all(|(a, b)| a == b)
    }
}

impl<F: Field> Eq for FunctionFieldElement_rational<F> {}

impl<F: Field> fmt::Display for FunctionFieldElement_rational<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let format_poly = |coeffs: &[F]| -> String {
            if coeffs.is_empty() || coeffs.iter().all(|c| c.is_zero()) {
                return "0".to_string();
            }

            let mut terms = Vec::new();
            for (i, c) in coeffs.iter().enumerate() {
                if !c.is_zero() {
                    let term = match i {
                        0 => format!("{}", c),
                        1 => format!("{}*x", c),
                        _ => format!("{}*x^{}", c, i),
                    };
                    terms.push(term);
                }
            }

            if terms.is_empty() {
                "0".to_string()
            } else {
                terms.join(" + ")
            }
        };

        let num_str = format_poly(&self.numerator);
        let den_str = format_poly(&self.denominator);

        if self.is_integral() {
            write!(f, "{}", num_str)
        } else {
            write!(f, "({})/({})", num_str, den_str)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_creation() {
        let elem = FunctionFieldElement_rational::new(
            vec![Rational::from(1)],
            vec![Rational::from(1)],
        );

        assert_eq!(elem.numerator().len(), 1);
        assert_eq!(elem.denominator().len(), 1);
    }

    #[test]
    #[should_panic(expected = "Denominator cannot be zero polynomial")]
    fn test_zero_denominator() {
        let _ = FunctionFieldElement_rational::<Rational>::new(
            vec![Rational::from(1)],
            vec![],
        );
    }

    #[test]
    fn test_zero_one() {
        let zero = FunctionFieldElement_rational::<Rational>::zero();
        let one = FunctionFieldElement_rational::<Rational>::one();

        assert!(zero.is_zero());
        assert!(!zero.is_one());
        assert!(one.is_one());
        assert!(!one.is_zero());
    }

    #[test]
    fn test_variable() {
        let x = FunctionFieldElement_rational::<Rational>::variable();
        assert_eq!(x.numerator().len(), 2);
        assert_eq!(x.denominator().len(), 1);
    }

    #[test]
    fn test_inverse() {
        let elem = FunctionFieldElement_rational::new(
            vec![Rational::from(2)],
            vec![Rational::from(3)],
        );

        let inv = elem.inverse().unwrap();
        assert_eq!(inv.numerator()[0], Rational::from(3));
        assert_eq!(inv.denominator()[0], Rational::from(2));

        // Zero has no inverse
        let zero = FunctionFieldElement_rational::<Rational>::zero();
        assert!(zero.inverse().is_none());
    }

    #[test]
    fn test_is_integral() {
        // Polynomial (integral)
        let poly = FunctionFieldElement_rational::new(
            vec![Rational::from(1), Rational::from(2)],  // 1 + 2x
            vec![Rational::from(1)],
        );
        assert!(poly.is_integral());

        // Proper rational function (not integral)
        let rat = FunctionFieldElement_rational::new(
            vec![Rational::from(1)],
            vec![Rational::from(1), Rational::from(1)],  // 1/(1+x)
        );
        assert!(!rat.is_integral());
    }

    #[test]
    fn test_degrees() {
        let elem = FunctionFieldElement_rational::new(
            vec![Rational::from(1), Rational::from(2), Rational::from(3)],  // 1 + 2x + 3x^2
            vec![Rational::from(1), Rational::from(1)],  // 1 + x
        );

        assert_eq!(elem.numerator_degree(), 2);
        assert_eq!(elem.denominator_degree(), 1);
    }

    #[test]
    fn test_divisor_info() {
        let elem = FunctionFieldElement_rational::new(
            vec![Rational::from(1), Rational::from(2)],
            vec![Rational::from(1), Rational::from(1)],
        );

        let (zeros, poles) = elem.divisor_info();
        assert_eq!(zeros, 1);
        assert_eq!(poles, 1);
    }

    #[test]
    fn test_equality() {
        // 2/4 should equal 1/2 (after cross multiply)
        let a = FunctionFieldElement_rational::new(
            vec![Rational::from(2)],
            vec![Rational::from(4)],
        );
        let b = FunctionFieldElement_rational::new(
            vec![Rational::from(1)],
            vec![Rational::from(2)],
        );

        assert_eq!(a, b);
    }
}
