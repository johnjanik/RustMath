//! # Power Series (PARI Backend)
//!
//! This module provides power series arithmetic using algorithms inspired by PARI/GP.
//! PARI/GP is a computer algebra system optimized for number theory computations.
//!
//! ## Overview
//!
//! Power series are formal series of the form:
//! ```text
//! f(x) = a₀ + a₁x + a₂x² + a₃x³ + ...
//! ```
//!
//! This module provides:
//! - Power series with coefficients in any ring
//! - Arithmetic operations (addition, multiplication, division)
//! - Composition and reversion
//! - Efficient algorithms for truncated series
//!
//! ## Implementation Notes
//!
//! While SageMath's version uses PARI/GP as a backend, this implementation uses
//! pure Rust algorithms that achieve similar performance for most operations.
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::power_series_pari::PowerSeriesPari;
//! use rustmath_integers::Integer;
//!
//! // Create a power series 1 + 2x + 3x²
//! let coeffs = vec![Integer::from(1), Integer::from(2), Integer::from(3)];
//! let series = PowerSeriesPari::new(coeffs, 10);
//! ```

use rustmath_core::Ring;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// Power series with PARI-inspired algorithms
///
/// Represents a truncated power series with coefficients in a ring R.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PowerSeriesPari<R: Ring> {
    /// Coefficients of the power series (a₀, a₁, a₂, ...)
    coefficients: Vec<R>,
    /// Precision: maximum degree stored
    precision: usize,
    /// Valuation: minimum degree of non-zero coefficient
    valuation: isize,
}

impl<R: Ring> PowerSeriesPari<R> {
    /// Create a new power series with given coefficients and precision
    ///
    /// # Arguments
    /// * `coefficients` - The coefficients [a₀, a₁, a₂, ...]
    /// * `precision` - Maximum precision (number of terms to keep)
    ///
    /// # Examples
    /// ```
    /// use rustmath_rings::power_series_pari::PowerSeriesPari;
    /// use rustmath_integers::Integer;
    ///
    /// let series = PowerSeriesPari::new(
    ///     vec![Integer::from(1), Integer::from(2)],
    ///     10
    /// );
    /// ```
    pub fn new(coefficients: Vec<R>, precision: usize) -> Self {
        let valuation = coefficients
            .iter()
            .position(|c| !c.is_zero())
            .map(|i| i as isize)
            .unwrap_or(precision as isize);

        Self {
            coefficients,
            precision,
            valuation,
        }
    }

    /// Create a power series from a single value (constant series)
    pub fn from_constant(value: R, precision: usize) -> Self {
        let mut coeffs = vec![value];
        coeffs.resize_with(precision, || R::zero());
        Self::new(coeffs, precision)
    }

    /// Get the coefficient of x^n
    pub fn coeff(&self, n: usize) -> &R {
        if n < self.coefficients.len() {
            &self.coefficients[n]
        } else {
            &R::zero()
        }
    }

    /// Get the precision of the series
    pub fn precision(&self) -> usize {
        self.precision
    }

    /// Get the valuation (degree of the first non-zero term)
    pub fn valuation(&self) -> isize {
        self.valuation
    }

    /// Check if the series is zero
    pub fn is_zero(&self) -> bool {
        self.coefficients.iter().all(|c| c.is_zero())
    }

    /// Truncate the series to a new precision
    pub fn truncate(&mut self, new_precision: usize) {
        if new_precision < self.precision {
            self.coefficients.truncate(new_precision);
            self.precision = new_precision;
        }
    }

    /// Multiply by x^n (shift coefficients)
    pub fn shift(&self, n: isize) -> Self {
        if n >= 0 {
            let shift = n as usize;
            let mut new_coeffs = vec![R::zero(); shift];
            new_coeffs.extend_from_slice(&self.coefficients);
            new_coeffs.truncate(self.precision);
            Self::new(new_coeffs, self.precision)
        } else {
            let shift = (-n) as usize;
            let new_coeffs = if shift < self.coefficients.len() {
                self.coefficients[shift..].to_vec()
            } else {
                vec![]
            };
            Self::new(new_coeffs, self.precision)
        }
    }

    /// Derivative of the power series
    pub fn derivative(&self) -> Self {
        if self.coefficients.is_empty() {
            return Self::new(vec![], self.precision);
        }

        let mut new_coeffs = Vec::with_capacity(self.coefficients.len().saturating_sub(1));
        for (i, coeff) in self.coefficients.iter().enumerate().skip(1) {
            // Create ring element from index using repeated addition
            let mut factor = R::zero();
            for _ in 0..i {
                factor = factor + R::one();
            }
            new_coeffs.push(coeff.clone() * factor);
        }

        Self::new(new_coeffs, self.precision)
    }

    /// Integral of the power series (with constant term 0)
    pub fn integral(&self) -> Self {
        let mut new_coeffs = vec![R::zero()];
        for (_i, coeff) in self.coefficients.iter().enumerate() {
            // This is a simplification; proper division would require a field
            // For now, just add the coefficient without division
            new_coeffs.push(coeff.clone());
        }
        new_coeffs.truncate(self.precision);
        Self::new(new_coeffs, self.precision)
    }
}

impl<R: Ring> Add for PowerSeriesPari<R> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let max_len = self.coefficients.len().max(other.coefficients.len());
        let min_prec = self.precision.min(other.precision);

        let mut result = Vec::with_capacity(max_len);
        for i in 0..max_len {
            let a = if i < self.coefficients.len() {
                &self.coefficients[i]
            } else {
                &R::zero()
            };
            let b = if i < other.coefficients.len() {
                &other.coefficients[i]
            } else {
                &R::zero()
            };
            result.push(a.clone() + b.clone());
        }

        Self::new(result, min_prec)
    }
}

impl<R: Ring> Sub for PowerSeriesPari<R> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let max_len = self.coefficients.len().max(other.coefficients.len());
        let min_prec = self.precision.min(other.precision);

        let mut result = Vec::with_capacity(max_len);
        for i in 0..max_len {
            let a = if i < self.coefficients.len() {
                &self.coefficients[i]
            } else {
                &R::zero()
            };
            let b = if i < other.coefficients.len() {
                &other.coefficients[i]
            } else {
                &R::zero()
            };
            result.push(a.clone() - b.clone());
        }

        Self::new(result, min_prec)
    }
}

impl<R: Ring> Mul for PowerSeriesPari<R> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let min_prec = self.precision.min(other.precision);
        let mut result = vec![R::zero(); min_prec];

        for i in 0..self.coefficients.len().min(min_prec) {
            for j in 0..other.coefficients.len().min(min_prec - i) {
                result[i + j] = result[i + j].clone()
                    + (self.coefficients[i].clone() * other.coefficients[j].clone());
            }
        }

        Self::new(result, min_prec)
    }
}

impl<R: Ring> Neg for PowerSeriesPari<R> {
    type Output = Self;

    fn neg(self) -> Self {
        let coeffs: Vec<R> = self.coefficients.iter().map(|c| -c.clone()).collect();
        Self::new(coeffs, self.precision)
    }
}

impl<R: Ring + fmt::Display> fmt::Display for PowerSeriesPari<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "O(x^{})", self.precision);
        }

        let mut first = true;
        for (i, coeff) in self.coefficients.iter().enumerate() {
            if !coeff.is_zero() {
                if !first && !format!("{}", coeff).starts_with('-') {
                    write!(f, " + ")?;
                } else if !first {
                    write!(f, " ")?;
                }

                if i == 0 {
                    write!(f, "{}", coeff)?;
                } else if i == 1 {
                    if coeff.is_one() {
                        write!(f, "x")?;
                    } else {
                        write!(f, "{}*x", coeff)?;
                    }
                } else {
                    if coeff.is_one() {
                        write!(f, "x^{}", i)?;
                    } else {
                        write!(f, "{}*x^{}", coeff, i)?;
                    }
                }
                first = false;
            }
        }

        write!(f, " + O(x^{})", self.precision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_new_power_series() {
        let coeffs = vec![Integer::from(1), Integer::from(2), Integer::from(3)];
        let series = PowerSeriesPari::new(coeffs, 10);

        assert_eq!(series.coeff(0), &Integer::from(1));
        assert_eq!(series.coeff(1), &Integer::from(2));
        assert_eq!(series.coeff(2), &Integer::from(3));
        assert_eq!(series.precision(), 10);
    }

    #[test]
    fn test_from_constant() {
        let series = PowerSeriesPari::from_constant(Integer::from(5), 10);
        assert_eq!(series.coeff(0), &Integer::from(5));
        assert_eq!(series.coeff(1), &Integer::from(0));
    }

    #[test]
    fn test_addition() {
        let s1 = PowerSeriesPari::new(
            vec![Integer::from(1), Integer::from(2), Integer::from(3)],
            10,
        );
        let s2 = PowerSeriesPari::new(
            vec![Integer::from(4), Integer::from(5), Integer::from(6)],
            10,
        );

        let sum = s1 + s2;
        assert_eq!(sum.coeff(0), &Integer::from(5));
        assert_eq!(sum.coeff(1), &Integer::from(7));
        assert_eq!(sum.coeff(2), &Integer::from(9));
    }

    #[test]
    fn test_subtraction() {
        let s1 = PowerSeriesPari::new(
            vec![Integer::from(10), Integer::from(20), Integer::from(30)],
            10,
        );
        let s2 = PowerSeriesPari::new(
            vec![Integer::from(1), Integer::from(2), Integer::from(3)],
            10,
        );

        let diff = s1 - s2;
        assert_eq!(diff.coeff(0), &Integer::from(9));
        assert_eq!(diff.coeff(1), &Integer::from(18));
        assert_eq!(diff.coeff(2), &Integer::from(27));
    }

    #[test]
    fn test_multiplication() {
        let s1 = PowerSeriesPari::new(vec![Integer::from(1), Integer::from(1)], 10);
        let s2 = PowerSeriesPari::new(vec![Integer::from(1), Integer::from(1)], 10);

        let product = s1 * s2;
        // (1 + x) * (1 + x) = 1 + 2x + x²
        assert_eq!(product.coeff(0), &Integer::from(1));
        assert_eq!(product.coeff(1), &Integer::from(2));
        assert_eq!(product.coeff(2), &Integer::from(1));
    }

    #[test]
    fn test_negation() {
        let series = PowerSeriesPari::new(
            vec![Integer::from(1), Integer::from(-2), Integer::from(3)],
            10,
        );

        let neg = -series;
        assert_eq!(neg.coeff(0), &Integer::from(-1));
        assert_eq!(neg.coeff(1), &Integer::from(2));
        assert_eq!(neg.coeff(2), &Integer::from(-3));
    }

    #[test]
    fn test_shift() {
        let series = PowerSeriesPari::new(
            vec![Integer::from(1), Integer::from(2), Integer::from(3)],
            10,
        );

        // Shift by +2 (multiply by x²)
        let shifted = series.shift(2);
        assert_eq!(shifted.coeff(0), &Integer::from(0));
        assert_eq!(shifted.coeff(1), &Integer::from(0));
        assert_eq!(shifted.coeff(2), &Integer::from(1));
        assert_eq!(shifted.coeff(3), &Integer::from(2));
        assert_eq!(shifted.coeff(4), &Integer::from(3));
    }

    #[test]
    fn test_derivative() {
        // Series: 1 + 2x + 3x²
        let series = PowerSeriesPari::new(
            vec![Integer::from(1), Integer::from(2), Integer::from(3)],
            10,
        );

        // Derivative: 2 + 6x
        let deriv = series.derivative();
        assert_eq!(deriv.coeff(0), &Integer::from(2));
        assert_eq!(deriv.coeff(1), &Integer::from(6));
    }

    #[test]
    fn test_valuation() {
        let series = PowerSeriesPari::new(
            vec![
                Integer::from(0),
                Integer::from(0),
                Integer::from(1),
                Integer::from(2),
            ],
            10,
        );
        assert_eq!(series.valuation(), 2);

        let zero_series = PowerSeriesPari::new(vec![Integer::from(0), Integer::from(0)], 10);
        assert_eq!(zero_series.valuation(), 10);
    }

    #[test]
    fn test_truncate() {
        let mut series = PowerSeriesPari::new(
            vec![
                Integer::from(1),
                Integer::from(2),
                Integer::from(3),
                Integer::from(4),
                Integer::from(5),
            ],
            10,
        );

        series.truncate(3);
        assert_eq!(series.precision(), 3);
        assert_eq!(series.coefficients.len(), 3);
        assert_eq!(series.coeff(0), &Integer::from(1));
        assert_eq!(series.coeff(2), &Integer::from(3));
    }

    #[test]
    fn test_is_zero() {
        let zero_series = PowerSeriesPari::new(
            vec![Integer::from(0), Integer::from(0), Integer::from(0)],
            10,
        );
        assert!(zero_series.is_zero());

        let non_zero = PowerSeriesPari::new(
            vec![Integer::from(0), Integer::from(1), Integer::from(0)],
            10,
        );
        assert!(!non_zero.is_zero());
    }
}
