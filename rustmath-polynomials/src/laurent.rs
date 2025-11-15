//! Laurent polynomials - polynomials with both positive and negative powers
//!
//! A Laurent polynomial is a polynomial that can have negative powers of the variable.
//! It has the form: a_{-n}*x^{-n} + ... + a_{-1}*x^{-1} + a_0 + a_1*x + ... + a_m*x^m

use rustmath_core::Ring;
use std::collections::BTreeMap;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// A Laurent polynomial over a ring R
///
/// Represented as a sparse map from exponents (which can be negative) to coefficients
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LaurentPolynomial<R: Ring> {
    /// Map from exponent to coefficient
    /// Only non-zero coefficients are stored
    terms: BTreeMap<i32, R>,
}

impl<R: Ring> LaurentPolynomial<R> {
    /// Create a new Laurent polynomial
    pub fn new() -> Self {
        LaurentPolynomial {
            terms: BTreeMap::new(),
        }
    }

    /// Create a Laurent polynomial from a monomial c*x^n
    pub fn monomial(coeff: R, power: i32) -> Self {
        if coeff.is_zero() {
            return LaurentPolynomial::new();
        }

        let mut terms = BTreeMap::new();
        terms.insert(power, coeff);
        LaurentPolynomial { terms }
    }

    /// Create a constant Laurent polynomial
    pub fn constant(c: R) -> Self {
        Self::monomial(c, 0)
    }

    /// Create the variable x
    pub fn var() -> Self {
        Self::monomial(R::one(), 1)
    }

    /// Add a term to this polynomial
    pub fn add_term(&mut self, coeff: R, power: i32) {
        if coeff.is_zero() {
            return;
        }

        let entry = self.terms.entry(power).or_insert_with(R::zero);
        *entry = entry.clone() + coeff;

        // Remove if becomes zero
        if entry.is_zero() {
            self.terms.remove(&power);
        }
    }

    /// Get the coefficient of x^n
    pub fn coeff(&self, power: i32) -> R {
        self.terms.get(&power).cloned().unwrap_or_else(R::zero)
    }

    /// Check if this is the zero polynomial
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Get the minimal exponent (lowest power)
    pub fn min_exponent(&self) -> Option<i32> {
        self.terms.keys().next().copied()
    }

    /// Get the maximal exponent (highest power)
    pub fn max_exponent(&self) -> Option<i32> {
        self.terms.keys().next_back().copied()
    }

    /// Get all terms as a vector of (power, coefficient) pairs
    pub fn terms(&self) -> Vec<(i32, R)> {
        self.terms
            .iter()
            .map(|(p, c)| (*p, c.clone()))
            .collect()
    }
}

impl<R: Ring> Default for LaurentPolynomial<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Ring> fmt::Display for LaurentPolynomial<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut first = true;
        // Iterate in descending order of powers
        for (&power, coeff) in self.terms.iter().rev() {
            if !first {
                write!(f, " + ")?;
            }
            first = false;

            if power == 0 {
                write!(f, "{}", coeff)?;
            } else if power == 1 {
                if coeff.is_one() {
                    write!(f, "q")?;
                } else {
                    write!(f, "{}*q", coeff)?;
                }
            } else if power < 0 {
                if coeff.is_one() {
                    write!(f, "q^({})", power)?;
                } else {
                    write!(f, "{}*q^({})", coeff, power)?;
                }
            } else {
                // power > 1
                if coeff.is_one() {
                    write!(f, "q^{}", power)?;
                } else {
                    write!(f, "{}*q^{}", coeff, power)?;
                }
            }
        }
        Ok(())
    }
}

impl<R: Ring> Add for LaurentPolynomial<R> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut result = self.clone();
        for (power, coeff) in other.terms {
            result.add_term(coeff, power);
        }
        result
    }
}

impl<R: Ring> Sub for LaurentPolynomial<R> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + (-other)
    }
}

impl<R: Ring> Neg for LaurentPolynomial<R> {
    type Output = Self;

    fn neg(self) -> Self {
        let mut result = LaurentPolynomial::new();
        for (power, coeff) in self.terms {
            result.add_term(-coeff, power);
        }
        result
    }
}

impl<R: Ring> Mul for LaurentPolynomial<R> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return LaurentPolynomial::new();
        }

        let mut result = LaurentPolynomial::new();

        for (p1, c1) in &self.terms {
            for (p2, c2) in &other.terms {
                let new_power = p1 + p2;
                let new_coeff = c1.clone() * c2.clone();
                result.add_term(new_coeff, new_power);
            }
        }

        result
    }
}

impl<R: Ring> Ring for LaurentPolynomial<R> {
    fn zero() -> Self {
        LaurentPolynomial::new()
    }

    fn one() -> Self {
        LaurentPolynomial::constant(R::one())
    }

    fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    fn is_one(&self) -> bool {
        self.terms.len() == 1 && {
            let (power, coeff) = self.terms.iter().next().unwrap();
            *power == 0 && coeff.is_one()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_monomial() {
        let p = LaurentPolynomial::<Integer>::monomial(Integer::from(3), 2);
        assert_eq!(p.coeff(2), Integer::from(3));
        assert_eq!(p.coeff(0), Integer::zero());
    }

    #[test]
    fn test_negative_power() {
        let p = LaurentPolynomial::<Integer>::monomial(Integer::from(1), -2);
        assert_eq!(p.coeff(-2), Integer::from(1));
        assert_eq!(p.min_exponent(), Some(-2));
        assert_eq!(p.max_exponent(), Some(-2));
    }

    #[test]
    fn test_addition() {
        let p1 = LaurentPolynomial::<Integer>::monomial(Integer::from(1), 2);
        let p2 = LaurentPolynomial::<Integer>::monomial(Integer::from(1), -1);

        let sum = p1 + p2;
        assert_eq!(sum.coeff(2), Integer::from(1));
        assert_eq!(sum.coeff(-1), Integer::from(1));
    }

    #[test]
    fn test_multiplication() {
        // (q^2) * (q^{-1}) = q
        let p1 = LaurentPolynomial::<Integer>::monomial(Integer::from(1), 2);
        let p2 = LaurentPolynomial::<Integer>::monomial(Integer::from(1), -1);

        let prod = p1 * p2;
        assert_eq!(prod.coeff(1), Integer::from(1));
        assert_eq!(prod.coeff(0), Integer::zero());
    }

    #[test]
    fn test_zero() {
        let p = LaurentPolynomial::<Integer>::zero();
        assert!(p.is_zero());
    }

    #[test]
    fn test_one() {
        let p = LaurentPolynomial::<Integer>::one();
        assert!(p.is_one());
        assert_eq!(p.coeff(0), Integer::one());
    }
}
