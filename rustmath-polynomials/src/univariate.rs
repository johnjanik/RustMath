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

    /// Compute polynomial GCD (for polynomials over a field or Euclidean domain)
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
}
