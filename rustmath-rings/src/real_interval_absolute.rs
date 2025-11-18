//! # Real Interval Absolute Precision Module
//!
//! Implementation of interval arithmetic with fixed absolute precision, as opposed to
//! relative precision used in standard floating-point interval arithmetic.
//!
//! ## Overview
//!
//! This module provides real intervals where the error bound is a fixed absolute value
//! rather than relative to the magnitude. This is particularly useful for:
//! - Series computations where terms accumulate
//! - Addition/subtraction-heavy algorithms
//! - Situations where uniform absolute error is needed
//!
//! ## Absolute vs Relative Precision
//!
//! **Relative Precision** (standard intervals):
//! - Error proportional to magnitude: x ± ε·|x|
//! - Good for multiplication/division
//! - Precision loss in addition/subtraction
//!
//! **Absolute Precision** (this module):
//! - Fixed error bound: x ± ε (constant)
//! - Excellent for addition/subtraction
//! - May lose precision in multiplication of large numbers
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::real_interval_absolute::{RealIntervalAbsoluteField, shift_floor, shift_ceil};
//!
//! // Create a field with absolute precision 2^(-10)
//! let field = RealIntervalAbsoluteField::new(10);
//! ```
//!
//! ## Implementation Details
//!
//! Intervals are represented using integer arithmetic:
//! - `mantissa`: Left endpoint scaled by 2^precision
//! - `diameter`: Width of interval in same units
//! - Actual interval: [mantissa >> prec, (mantissa + diameter) >> prec]

use rustmath_core::Field;
use std::fmt;
use thiserror::Error;

/// Errors that can occur in absolute precision interval arithmetic
#[derive(Debug, Clone, Error, PartialEq)]
pub enum IntervalError {
    #[error("Invalid precision: must be positive")]
    InvalidPrecision,

    #[error("Division by zero interval")]
    DivisionByZero,

    #[error("Interval contains zero, operation undefined")]
    ContainsZero,
}

/// Field of real intervals with absolute precision
///
/// Represents the field of real numbers with intervals having a fixed absolute
/// error bound of 2^(-absprec).
#[derive(Debug, Clone, PartialEq)]
pub struct RealIntervalAbsoluteField {
    /// Absolute precision (error is 2^(-absprec))
    abs_prec: u32,
}

impl RealIntervalAbsoluteField {
    /// Creates a new absolute precision interval field
    ///
    /// # Arguments
    /// * `abs_prec` - The absolute precision (must be positive)
    ///
    /// # Errors
    /// Returns `InvalidPrecision` if precision is 0
    pub fn new(abs_prec: u32) -> Result<Self, IntervalError> {
        if abs_prec == 0 {
            return Err(IntervalError::InvalidPrecision);
        }
        Ok(RealIntervalAbsoluteField { abs_prec })
    }

    /// Returns the absolute precision
    pub fn precision(&self) -> u32 {
        self.abs_prec
    }

    /// Returns the error bound (2^(-precision))
    pub fn error_bound(&self) -> f64 {
        2.0_f64.powi(-(self.abs_prec as i32))
    }
}

impl Default for RealIntervalAbsoluteField {
    fn default() -> Self {
        // Default to 53 bits of precision (double precision)
        RealIntervalAbsoluteField { abs_prec: 53 }
    }
}

impl fmt::Display for RealIntervalAbsoluteField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Real Interval Field with absolute precision 2^(-{})",
            self.abs_prec
        )
    }
}

/// An element of a real interval absolute field
///
/// Represents an interval [a, b] using:
/// - `mantissa`: Left endpoint × 2^precision
/// - `diameter`: Width × 2^precision
///
/// The actual interval is [mantissa >> prec, (mantissa + diameter) >> prec]
#[derive(Debug, Clone, PartialEq)]
pub struct RealIntervalAbsoluteElement {
    /// Scaled left endpoint
    mantissa: i128,
    /// Scaled width
    diameter: u128,
    /// Reference to parent field
    abs_prec: u32,
}

impl RealIntervalAbsoluteElement {
    /// Creates a new interval element
    pub fn new(mantissa: i128, diameter: u128, abs_prec: u32) -> Self {
        RealIntervalAbsoluteElement {
            mantissa,
            diameter,
            abs_prec,
        }
    }

    /// Creates an interval from a point value
    pub fn from_value(value: f64, abs_prec: u32) -> Self {
        let scale = 2i128.pow(abs_prec);
        let mantissa = (value * scale as f64) as i128;
        RealIntervalAbsoluteElement {
            mantissa,
            diameter: 1, // Minimal uncertainty
            abs_prec,
        }
    }

    /// Returns the left endpoint
    pub fn lower(&self) -> f64 {
        let scale = 2f64.powi(self.abs_prec as i32);
        self.mantissa as f64 / scale
    }

    /// Returns the right endpoint
    pub fn upper(&self) -> f64 {
        let scale = 2f64.powi(self.abs_prec as i32);
        (self.mantissa + self.diameter as i128) as f64 / scale
    }

    /// Returns the midpoint of the interval
    pub fn midpoint(&self) -> f64 {
        (self.lower() + self.upper()) / 2.0
    }

    /// Returns the width/diameter of the interval
    pub fn width(&self) -> f64 {
        let scale = 2f64.powi(self.abs_prec as i32);
        self.diameter as f64 / scale
    }

    /// Checks if the interval contains zero
    pub fn contains_zero(&self) -> bool {
        self.mantissa < 0 && (self.mantissa + self.diameter as i128) > 0
            || self.mantissa == 0
    }

    /// Adds two intervals
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.abs_prec, other.abs_prec);
        RealIntervalAbsoluteElement {
            mantissa: self.mantissa + other.mantissa,
            diameter: self.diameter + other.diameter,
            abs_prec: self.abs_prec,
        }
    }

    /// Subtracts two intervals
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.abs_prec, other.abs_prec);
        RealIntervalAbsoluteElement {
            mantissa: self.mantissa - other.mantissa,
            diameter: self.diameter + other.diameter,
            abs_prec: self.abs_prec,
        }
    }

    /// Negates the interval
    pub fn neg(&self) -> Self {
        RealIntervalAbsoluteElement {
            mantissa: -(self.mantissa + self.diameter as i128),
            diameter: self.diameter,
            abs_prec: self.abs_prec,
        }
    }
}

impl fmt::Display for RealIntervalAbsoluteElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:.6}, {:.6}]", self.lower(), self.upper())
    }
}

/// Factory for creating real interval absolute fields
///
/// Implements the singleton/unique factory pattern to ensure field uniqueness.
#[derive(Debug, Default)]
pub struct Factory;

impl Factory {
    /// Creates a key for the field (the precision value)
    pub fn create_key(abs_prec: u32) -> u32 {
        abs_prec
    }

    /// Creates a field object with the given precision
    pub fn create_object(abs_prec: u32) -> Result<RealIntervalAbsoluteField, IntervalError> {
        RealIntervalAbsoluteField::new(abs_prec)
    }
}

/// Helper for MPFR-style operations on absolute precision intervals
///
/// This bridges absolute precision intervals with relative precision operations
/// by converting to/from standard interval representations.
#[derive(Debug)]
pub struct MpfrOp;

impl MpfrOp {
    /// Applies an MPFR-style operation to an absolute precision interval
    ///
    /// This converts the interval to relative precision, applies the operation,
    /// and converts back to absolute precision.
    pub fn apply<F>(interval: &RealIntervalAbsoluteElement, op: F) -> RealIntervalAbsoluteElement
    where
        F: Fn(f64) -> f64,
    {
        let lower = op(interval.lower());
        let upper = op(interval.upper());
        let mid = (lower + upper) / 2.0;
        let width = (upper - lower).abs();

        let scale = 2i128.pow(interval.abs_prec);
        let mantissa = (mid * scale as f64) as i128;
        let diameter = (width * scale as f64) as u128;

        RealIntervalAbsoluteElement::new(mantissa, diameter, interval.abs_prec)
    }
}

/// Computes floor(x / 2^shift)
///
/// This performs a floor division by a power of 2, useful for precision reduction.
pub fn shift_floor(x: i128, shift: u32) -> i128 {
    if shift == 0 {
        return x;
    }
    if x >= 0 {
        x >> shift
    } else {
        // For negative numbers, we need to round towards negative infinity
        let shifted = x >> shift;
        if (x & ((1i128 << shift) - 1)) != 0 {
            shifted - 1
        } else {
            shifted
        }
    }
}

/// Computes ceil(x / 2^shift)
///
/// This performs a ceiling division by a power of 2, useful for precision reduction.
pub fn shift_ceil(x: i128, shift: u32) -> i128 {
    if shift == 0 {
        return x;
    }
    if x >= 0 {
        // For positive numbers, round towards positive infinity
        let shifted = x >> shift;
        if (x & ((1i128 << shift) - 1)) != 0 {
            shifted + 1
        } else {
            shifted
        }
    } else {
        x >> shift
    }
}

/// Factory function for creating absolute precision interval fields
pub fn real_interval_absolute_field(abs_prec: u32) -> Result<RealIntervalAbsoluteField, IntervalError> {
    Factory::create_object(abs_prec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_creation() {
        let field = RealIntervalAbsoluteField::new(10).unwrap();
        assert_eq!(field.precision(), 10);
        assert!((field.error_bound() - 2.0_f64.powi(-10)).abs() < 1e-10);
    }

    #[test]
    fn test_field_invalid_precision() {
        assert!(RealIntervalAbsoluteField::new(0).is_err());
    }

    #[test]
    fn test_field_display() {
        let field = RealIntervalAbsoluteField::new(20).unwrap();
        assert_eq!(
            format!("{}", field),
            "Real Interval Field with absolute precision 2^(-20)"
        );
    }

    #[test]
    fn test_element_from_value() {
        let elem = RealIntervalAbsoluteElement::from_value(3.14, 10);
        assert!((elem.midpoint() - 3.14).abs() < 0.01);
    }

    #[test]
    fn test_element_contains_zero() {
        let elem1 = RealIntervalAbsoluteElement::new(-100, 200, 10);
        assert!(elem1.contains_zero());

        let elem2 = RealIntervalAbsoluteElement::new(100, 50, 10);
        assert!(!elem2.contains_zero());
    }

    #[test]
    fn test_interval_addition() {
        let a = RealIntervalAbsoluteElement::from_value(1.0, 10);
        let b = RealIntervalAbsoluteElement::from_value(2.0, 10);
        let c = a.add(&b);
        assert!((c.midpoint() - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_interval_subtraction() {
        let a = RealIntervalAbsoluteElement::from_value(5.0, 10);
        let b = RealIntervalAbsoluteElement::from_value(2.0, 10);
        let c = a.sub(&b);
        assert!((c.midpoint() - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_interval_negation() {
        let a = RealIntervalAbsoluteElement::from_value(3.0, 10);
        let b = a.neg();
        assert!((b.midpoint() + 3.0).abs() < 0.01);
    }

    #[test]
    fn test_shift_floor() {
        assert_eq!(shift_floor(100, 2), 25);  // 100 / 4 = 25
        assert_eq!(shift_floor(101, 2), 25);  // floor(101 / 4) = 25
        assert_eq!(shift_floor(-100, 2), -25); // -100 / 4 = -25
        assert_eq!(shift_floor(-99, 2), -25);  // floor(-99 / 4) = -25
    }

    #[test]
    fn test_shift_ceil() {
        assert_eq!(shift_ceil(100, 2), 25);   // 100 / 4 = 25
        assert_eq!(shift_ceil(101, 2), 26);   // ceil(101 / 4) = 26
        assert_eq!(shift_ceil(-100, 2), -25); // -100 / 4 = -25
        assert_eq!(shift_ceil(-101, 2), -25); // ceil(-101 / 4) = -25
    }

    #[test]
    fn test_factory() {
        assert_eq!(Factory::create_key(15), 15);
        let field = Factory::create_object(15).unwrap();
        assert_eq!(field.precision(), 15);
    }

    #[test]
    fn test_real_interval_absolute_field_factory() {
        let field = real_interval_absolute_field(12).unwrap();
        assert_eq!(field.precision(), 12);
    }
}
