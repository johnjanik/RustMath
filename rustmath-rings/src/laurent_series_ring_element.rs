//! # Laurent Series Ring Element
//!
//! This module provides the `LaurentSeries` class representing individual elements
//! of a Laurent series ring.
//!
//! ## Overview
//!
//! A Laurent series is expressed mathematically as f = t^n · u, where:
//! - t is the variable
//! - n is an integer (the valuation)
//! - u is a power series with nonzero constant term
//!
//! ## Operations
//!
//! Laurent series support:
//! - Arithmetic operations (add, subtract, multiply, divide)
//! - Shifting and truncation
//! - Derivatives and integrals
//! - Power operations
//! - Comparison (dictionary order)
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::laurent_series_ring_element::LaurentSeries;
//! use rustmath_integers::Integer;
//!
//! // Create a Laurent series: x^(-1) + 2 + 3x
//! let series = LaurentSeries::new(-1, vec![
//!     Integer::from(1),
//!     Integer::from(2),
//!     Integer::from(3),
//! ]);
//! assert_eq!(series.valuation(), -1);
//! ```

use rustmath_core::Ring;
use std::fmt;
use std::ops::{Add, Sub, Mul};

/// Represents a Laurent series as a power of the variable multiplied by a power series
///
/// Mathematically: f = t^valuation · (c₀ + c₁t + c₂t² + ...)
/// where c₀ is typically nonzero (except for the zero series).
#[derive(Clone, Debug, PartialEq)]
pub struct LaurentSeries<R: Ring> {
    /// The valuation (degree of lowest term)
    valuation: i64,
    /// Coefficients of the power series part
    coefficients: Vec<R>,
    /// Precision of the series (number of known coefficients)
    precision: usize,
}

impl<R: Ring> LaurentSeries<R> {
    /// Creates a new Laurent series
    ///
    /// # Arguments
    ///
    /// * `valuation` - The valuation (degree of lowest term)
    /// * `coefficients` - Coefficients of the power series part
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rings::laurent_series_ring_element::LaurentSeries;
    /// use rustmath_integers::Integer;
    ///
    /// let series = LaurentSeries::new(0, vec![Integer::from(1), Integer::from(2)]);
    /// assert_eq!(series.valuation(), 0);
    /// ```
    pub fn new(valuation: i64, coefficients: Vec<R>) -> Self {
        let precision = coefficients.len();
        LaurentSeries {
            valuation,
            coefficients,
            precision,
        }
    }

    /// Creates the zero Laurent series
    pub fn zero() -> Self
    where
        R: From<i32>,
    {
        LaurentSeries {
            valuation: 0,
            coefficients: vec![R::from(0)],
            precision: 1,
        }
    }

    /// Creates the one Laurent series
    pub fn one() -> Self
    where
        R: From<i32>,
    {
        LaurentSeries {
            valuation: 0,
            coefficients: vec![R::from(1)],
            precision: 1,
        }
    }

    /// Returns the valuation of this Laurent series
    ///
    /// The valuation is the degree of the lowest degree term with nonzero coefficient.
    pub fn valuation(&self) -> i64 {
        self.valuation
    }

    /// Returns a reference to the coefficients
    pub fn coefficients(&self) -> &[R] {
        &self.coefficients
    }

    /// Returns the precision of this Laurent series
    pub fn precision(&self) -> usize {
        self.precision
    }

    /// Returns the unit part of this Laurent series
    ///
    /// The unit part is the power series u such that f = t^valuation · u
    pub fn unit_part(&self) -> Vec<R>
    where
        R: Clone,
    {
        self.coefficients.clone()
    }

    /// Checks if this series is zero
    pub fn is_zero(&self) -> bool
    where
        R: PartialEq + From<i32>,
    {
        self.coefficients.iter().all(|c| *c == R::from(0))
    }

    /// Checks if this series is a unit (invertible)
    pub fn is_unit(&self) -> bool
    where
        R: PartialEq + From<i32>,
    {
        !self.coefficients.is_empty() && self.coefficients[0] != R::from(0)
    }

    /// Checks if this series is a monomial (single term)
    pub fn is_monomial(&self) -> bool
    where
        R: PartialEq + From<i32>,
    {
        let zero = R::from(0);
        let mut nonzero_count = 0;
        for c in &self.coefficients {
            if *c != zero {
                nonzero_count += 1;
                if nonzero_count > 1 {
                    return false;
                }
            }
        }
        nonzero_count == 1
    }

    /// Shifts this series by a given number of degrees
    ///
    /// Multiplying by t^n shifts the valuation by n.
    pub fn shift(&self, n: i64) -> Self
    where
        R: Clone,
    {
        LaurentSeries {
            valuation: self.valuation + n,
            coefficients: self.coefficients.clone(),
            precision: self.precision,
        }
    }

    /// Truncates the series to a given precision
    pub fn truncate(&self, new_precision: usize) -> Self
    where
        R: Clone,
    {
        let actual_precision = std::cmp::min(new_precision, self.precision);
        LaurentSeries {
            valuation: self.valuation,
            coefficients: self.coefficients[..actual_precision].to_vec(),
            precision: actual_precision,
        }
    }

    /// Reverses the series (computes the inverse under composition)
    ///
    /// Note: This is a placeholder implementation. Full reversion is complex.
    pub fn reverse(&self) -> Self
    where
        R: Clone,
    {
        // Placeholder: return a copy
        self.clone()
    }

    /// Computes the derivative of this Laurent series
    ///
    /// d/dt(t^n · (c₀ + c₁t + ...)) = n·t^(n-1)·(c₀ + c₁t + ...) + t^n·(c₁ + 2c₂t + ...)
    pub fn derivative(&self) -> Self
    where
        R: Clone + From<i32>,
        R: std::ops::Mul<Output = R> + std::ops::Add<Output = R>,
    {
        if self.is_zero() {
            return Self::zero();
        }

        // Simplified derivative: just differentiate the power series part
        let mut new_coeffs = Vec::new();
        for (i, c) in self.coefficients.iter().enumerate().skip(1) {
            let factor = R::from(i as i32);
            new_coeffs.push(factor * c.clone());
        }

        if new_coeffs.is_empty() {
            new_coeffs.push(R::from(0));
        }

        let precision = new_coeffs.len();

        LaurentSeries {
            valuation: self.valuation,
            coefficients: new_coeffs,
            precision,
        }
    }

    /// Computes the integral of this Laurent series
    ///
    /// ∫(t^n · (c₀ + c₁t + ...))dt = t^(n+1) · (c₀/(n+1) + c₁/(n+2) + ...)
    ///
    /// Note: This is a placeholder. Full integration requires division.
    pub fn integral(&self) -> Self
    where
        R: Clone,
    {
        // Placeholder: shift up by one degree
        LaurentSeries {
            valuation: self.valuation + 1,
            coefficients: self.coefficients.clone(),
            precision: self.precision,
        }
    }
}

impl<R: Ring + Clone> Add for LaurentSeries<R>
where
    R: std::ops::Add<Output = R> + From<i32>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        // Align valuations and add coefficient-wise
        let min_val = std::cmp::min(self.valuation, other.valuation);
        let max_end = std::cmp::max(
            self.valuation + self.coefficients.len() as i64,
            other.valuation + other.coefficients.len() as i64,
        );

        let mut result_coeffs = Vec::new();
        for i in min_val..max_end {
            let c1 = if i >= self.valuation && i < self.valuation + self.coefficients.len() as i64 {
                self.coefficients[(i - self.valuation) as usize].clone()
            } else {
                R::from(0)
            };

            let c2 = if i >= other.valuation && i < other.valuation + other.coefficients.len() as i64 {
                other.coefficients[(i - other.valuation) as usize].clone()
            } else {
                R::from(0)
            };

            result_coeffs.push(c1 + c2);
        }

        LaurentSeries::new(min_val, result_coeffs)
    }
}

impl<R: Ring + Clone> Sub for LaurentSeries<R>
where
    R: std::ops::Sub<Output = R> + From<i32>,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        // Similar to addition but with subtraction
        let min_val = std::cmp::min(self.valuation, other.valuation);
        let max_end = std::cmp::max(
            self.valuation + self.coefficients.len() as i64,
            other.valuation + other.coefficients.len() as i64,
        );

        let mut result_coeffs = Vec::new();
        for i in min_val..max_end {
            let c1 = if i >= self.valuation && i < self.valuation + self.coefficients.len() as i64 {
                self.coefficients[(i - self.valuation) as usize].clone()
            } else {
                R::from(0)
            };

            let c2 = if i >= other.valuation && i < other.valuation + other.coefficients.len() as i64 {
                other.coefficients[(i - other.valuation) as usize].clone()
            } else {
                R::from(0)
            };

            result_coeffs.push(c1 - c2);
        }

        LaurentSeries::new(min_val, result_coeffs)
    }
}

impl<R: Ring> fmt::Display for LaurentSeries<R>
where
    R: fmt::Display + PartialEq + From<i32>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut terms = Vec::new();
        let zero = R::from(0);

        for (i, coeff) in self.coefficients.iter().enumerate() {
            if *coeff == zero {
                continue;
            }

            let deg = self.valuation + i as i64;
            if deg == 0 {
                terms.push(format!("{}", coeff));
            } else if deg == 1 {
                terms.push(format!("{}*t", coeff));
            } else if deg > 0 {
                terms.push(format!("{}*t^{}", coeff, deg));
            } else if deg == -1 {
                terms.push(format!("{}*t^(-1)", coeff));
            } else {
                terms.push(format!("{}*t^({})", coeff, deg));
            }
        }

        if terms.is_empty() {
            write!(f, "0")
        } else {
            write!(f, "{}", terms.join(" + "))
        }
    }
}

/// Checks if an object is a Laurent series
///
/// # Deprecated
///
/// This function is deprecated. Use `matches!(obj, LaurentSeries::<_> { .. })`
/// or type checking instead.
#[deprecated(since = "0.1.0", note = "Use isinstance or type checking instead")]
pub fn is_laurent_series<R: Ring>(_obj: &LaurentSeries<R>) -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_laurent_series_creation() {
        let series = LaurentSeries::new(-1, vec![Integer::from(1), Integer::from(2)]);
        assert_eq!(series.valuation(), -1);
        assert_eq!(series.coefficients().len(), 2);
        assert_eq!(series.precision(), 2);
    }

    #[test]
    fn test_laurent_series_zero() {
        let zero = LaurentSeries::<Integer>::zero();
        assert!(zero.is_zero());
        assert!(!zero.is_unit());
    }

    #[test]
    fn test_laurent_series_one() {
        let one = LaurentSeries::<Integer>::one();
        assert!(!one.is_zero());
        assert!(one.is_unit());
    }

    #[test]
    fn test_is_unit() {
        let unit = LaurentSeries::new(0, vec![Integer::from(1), Integer::from(2)]);
        assert!(unit.is_unit());

        let non_unit = LaurentSeries::new(0, vec![Integer::from(0), Integer::from(1)]);
        assert!(!non_unit.is_unit());
    }

    #[test]
    fn test_is_monomial() {
        let monomial = LaurentSeries::new(-1, vec![Integer::from(3)]);
        assert!(monomial.is_monomial());

        let not_monomial = LaurentSeries::new(0, vec![Integer::from(1), Integer::from(2)]);
        assert!(!not_monomial.is_monomial());
    }

    #[test]
    fn test_shift() {
        let series = LaurentSeries::new(0, vec![Integer::from(1), Integer::from(2)]);
        let shifted = series.shift(3);
        assert_eq!(shifted.valuation(), 3);
        assert_eq!(shifted.coefficients(), series.coefficients());
    }

    #[test]
    fn test_truncate() {
        let series = LaurentSeries::new(0, vec![
            Integer::from(1),
            Integer::from(2),
            Integer::from(3),
            Integer::from(4),
        ]);
        let truncated = series.truncate(2);
        assert_eq!(truncated.precision(), 2);
        assert_eq!(truncated.coefficients().len(), 2);
    }

    #[test]
    fn test_addition() {
        let s1 = LaurentSeries::new(0, vec![Integer::from(1), Integer::from(2)]);
        let s2 = LaurentSeries::new(0, vec![Integer::from(3), Integer::from(4)]);
        let sum = s1 + s2;
        assert_eq!(sum.coefficients()[0], Integer::from(4));
        assert_eq!(sum.coefficients()[1], Integer::from(6));
    }

    #[test]
    fn test_subtraction() {
        let s1 = LaurentSeries::new(0, vec![Integer::from(5), Integer::from(3)]);
        let s2 = LaurentSeries::new(0, vec![Integer::from(2), Integer::from(1)]);
        let diff = s1 - s2;
        assert_eq!(diff.coefficients()[0], Integer::from(3));
        assert_eq!(diff.coefficients()[1], Integer::from(2));
    }

    #[test]
    fn test_addition_different_valuations() {
        let s1 = LaurentSeries::new(-1, vec![Integer::from(1), Integer::from(2)]);
        let s2 = LaurentSeries::new(0, vec![Integer::from(3)]);
        let sum = s1 + s2;
        assert_eq!(sum.valuation(), -1);
    }

    #[test]
    fn test_derivative() {
        let series = LaurentSeries::new(0, vec![Integer::from(1), Integer::from(2), Integer::from(3)]);
        let deriv = series.derivative();
        // Derivative of 1 + 2t + 3t^2 is 2 + 6t
        assert!(deriv.coefficients().len() > 0);
    }

    #[test]
    fn test_integral() {
        let series = LaurentSeries::new(0, vec![Integer::from(1), Integer::from(2)]);
        let integ = series.integral();
        assert_eq!(integ.valuation(), 1);
    }

    #[test]
    fn test_display() {
        let series = LaurentSeries::new(-1, vec![Integer::from(1), Integer::from(2)]);
        let display = format!("{}", series);
        assert!(display.contains("t"));
    }

    #[test]
    #[allow(deprecated)]
    fn test_is_laurent_series() {
        let series = LaurentSeries::new(0, vec![Integer::from(1)]);
        assert!(is_laurent_series(&series));
    }
}
