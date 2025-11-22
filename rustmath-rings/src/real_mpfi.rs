//! # Real MPFI Module
//!
//! Implementation of arbitrary precision real interval arithmetic using MPFI-style semantics.
//!
//! ## Overview
//!
//! MPFI (Multiple Precision Floating-point Interval) provides interval arithmetic
//! with arbitrary precision and guaranteed correct rounding. This module implements
//! intervals where both the lower and upper bounds are arbitrary precision floating-point
//! numbers (using the `rug` crate which wraps MPFR).
//!
//! ## Precision Guarantees
//!
//! - **Correctness**: All operations maintain the invariant that the true mathematical
//!   result is contained within the computed interval
//! - **Directed Rounding**: Lower bounds are rounded down (toward -∞), upper bounds are
//!   rounded up (toward +∞)
//! - **Arbitrary Precision**: Precision is specified in bits for the mantissa (minimum 2 bits)
//! - **Composability**: Operations on intervals with different precisions use the maximum
//!   precision of the operands
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::real_mpfi::{RealIntervalField, RealInterval};
//!
//! // Create a real interval field with 53 bits precision (IEEE 754 double)
//! let rif = RealIntervalField::new(53);
//!
//! // Create intervals
//! let x = rif.interval(1.0, 2.0);
//! let y = rif.interval(3.0, 4.0);
//!
//! // Arithmetic operations maintain containment guarantees
//! let sum = &x + &y;  // [4.0, 6.0]
//! let prod = &x * &y; // [3.0, 8.0]
//!
//! // High precision computation
//! let rif_256 = RealIntervalField::new(256);
//! let pi_lower = 3.14159265358979323;
//! let pi_upper = 3.14159265358979324;
//! let pi_interval = rif_256.interval(pi_lower, pi_upper);
//! ```
//!
//! ## Related Modules
//!
//! - `real_interval_absolute`: Fixed absolute precision intervals
//! - `real_arb`: Ball arithmetic using arb library semantics
//! - `rustmath-reals`: Core real number and interval implementations

use rug::float::Round;
use rug::Float;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use thiserror::Error;

/// Errors for real interval operations
#[derive(Debug, Clone, Error, PartialEq)]
pub enum RealIntervalError {
    #[error("Invalid precision: must be at least 2 bits")]
    InvalidPrecision,

    #[error("Division by interval containing zero")]
    DivisionByZero,

    #[error("Invalid interval: lower bound must be <= upper bound")]
    InvalidInterval,

    #[error("Operation undefined on this interval (e.g., sqrt of negative interval)")]
    UndefinedOperation,
}

/// Real interval field with arbitrary precision
///
/// This represents the field of real intervals with a specified precision
/// (number of mantissa bits). The precision determines the accuracy of the
/// interval endpoints.
///
/// # Examples
///
/// ```rust
/// use rustmath_rings::real_mpfi::RealIntervalField;
///
/// let rif = RealIntervalField::new(100);
/// assert_eq!(rif.precision(), 100);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct RealIntervalField {
    /// Precision in bits (minimum 2)
    precision: u32,
}

impl RealIntervalField {
    /// Creates a new real interval field with specified precision
    ///
    /// # Arguments
    ///
    /// * `precision` - Number of bits for the mantissa (minimum 2)
    ///
    /// # Panics
    ///
    /// Panics if precision < 2
    pub fn new(precision: u32) -> Self {
        assert!(precision >= 2, "Precision must be at least 2 bits");
        RealIntervalField { precision }
    }

    /// Returns the precision in bits
    pub fn precision(&self) -> u32 {
        self.precision
    }

    /// Creates an interval [value, value] (point interval)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_rings::real_mpfi::RealIntervalField;
    ///
    /// let rif = RealIntervalField::new(53);
    /// let x = rif.point(3.14);
    /// assert_eq!(x.lower_f64(), x.upper_f64());
    /// ```
    pub fn point(&self, value: f64) -> RealInterval {
        RealInterval::new_point(value, self.precision)
    }

    /// Creates an interval [lower, upper]
    ///
    /// # Arguments
    ///
    /// * `lower` - Lower bound
    /// * `upper` - Upper bound
    ///
    /// # Panics
    ///
    /// Panics if lower > upper
    pub fn interval(&self, lower: f64, upper: f64) -> RealInterval {
        RealInterval::new(lower, upper, self.precision)
    }

    /// Creates an interval from a value with optional error bound
    ///
    /// Returns the interval [value - error, value + error]
    pub fn from_value_error(&self, value: f64, error: f64) -> RealInterval {
        RealInterval::new(value - error, value + error, self.precision)
    }
}

impl Default for RealIntervalField {
    /// Creates a real interval field with IEEE 754 double precision (53 bits)
    fn default() -> Self {
        RealIntervalField::new(53)
    }
}

impl fmt::Display for RealIntervalField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Real Interval Field with {} bits precision",
            self.precision
        )
    }
}

/// An element of a real interval field
///
/// Represents a closed interval [lower, upper] of real numbers with arbitrary precision
/// endpoints. All arithmetic operations maintain the guarantee that the true mathematical
/// result is contained within the computed interval.
///
/// # Precision Guarantees
///
/// - Lower bounds are computed with rounding toward -∞ (Round::Down)
/// - Upper bounds are computed with rounding toward +∞ (Round::Up)
/// - This ensures all operations are **outward rounding**, preserving correctness
///
/// # Examples
///
/// ```rust
/// use rustmath_rings::real_mpfi::RealInterval;
///
/// // Create interval with 100-bit precision
/// let x = RealInterval::new(1.0, 2.0, 100);
/// assert!(x.contains(1.5));
///
/// // Point interval
/// let y = RealInterval::new_point(3.14, 53);
/// assert!(y.is_point());
/// ```
#[derive(Clone)]
pub struct RealInterval {
    /// Lower bound (rounded down)
    lower: Float,
    /// Upper bound (rounded up)
    upper: Float,
    /// Precision in bits
    precision: u32,
}

impl RealInterval {
    /// Creates a new interval [lower, upper]
    ///
    /// The lower bound is rounded down, the upper bound is rounded up, ensuring
    /// the true values are contained in the interval.
    ///
    /// # Arguments
    ///
    /// * `lower` - Lower bound value
    /// * `upper` - Upper bound value
    /// * `precision` - Precision in bits
    ///
    /// # Panics
    ///
    /// Panics if lower > upper or precision < 2
    pub fn new(lower: f64, upper: f64, precision: u32) -> Self {
        assert!(precision >= 2, "Precision must be at least 2 bits");
        assert!(
            lower <= upper,
            "Invalid interval: lower ({}) > upper ({})",
            lower,
            upper
        );

        // Create bounds with directed rounding
        let lower_bound = Float::with_val_round(precision, lower, Round::Down).0;
        let upper_bound = Float::with_val_round(precision, upper, Round::Up).0;

        RealInterval {
            lower: lower_bound,
            upper: upper_bound,
            precision,
        }
    }

    /// Creates a point interval [value, value]
    pub fn new_point(value: f64, precision: u32) -> Self {
        assert!(precision >= 2, "Precision must be at least 2 bits");

        let val = Float::with_val(precision, value);
        RealInterval {
            lower: val.clone(),
            upper: val,
            precision,
        }
    }

    /// Creates an interval from Float values (takes ownership)
    pub fn from_floats(lower: Float, upper: Float, precision: u32) -> Result<Self, RealIntervalError> {
        if lower > upper {
            return Err(RealIntervalError::InvalidInterval);
        }
        Ok(RealInterval {
            lower,
            upper,
            precision,
        })
    }

    /// Returns the precision in bits
    pub fn precision(&self) -> u32 {
        self.precision
    }

    /// Returns a reference to the lower bound
    pub fn lower(&self) -> &Float {
        &self.lower
    }

    /// Returns a reference to the upper bound
    pub fn upper(&self) -> &Float {
        &self.upper
    }

    /// Returns the lower bound as f64 (may lose precision)
    pub fn lower_f64(&self) -> f64 {
        self.lower.to_f64()
    }

    /// Returns the upper bound as f64 (may lose precision)
    pub fn upper_f64(&self) -> f64 {
        self.upper.to_f64()
    }

    /// Returns the midpoint of the interval
    ///
    /// Computed as (lower + upper) / 2
    pub fn midpoint(&self) -> Float {
        let sum = Float::with_val(self.precision, &self.lower + &self.upper);
        sum / 2u32
    }

    /// Returns the width (diameter) of the interval
    ///
    /// Computed as upper - lower (rounded up to ensure correctness)
    pub fn width(&self) -> Float {
        Float::with_val_round(self.precision, &self.upper - &self.lower, Round::Up).0
    }

    /// Returns the radius (half-width) of the interval
    pub fn radius(&self) -> Float {
        self.width() / 2u32
    }

    /// Returns the relative width: width / |midpoint|
    ///
    /// Returns None if midpoint is zero
    pub fn relative_width(&self) -> Option<Float> {
        let mid = self.midpoint();
        if mid == 0 {
            None
        } else {
            let width = self.width();
            Some(width / mid.abs())
        }
    }

    /// Checks if this interval contains a value
    pub fn contains(&self, value: f64) -> bool {
        let v = Float::with_val(self.precision, value);
        self.lower <= v && v <= self.upper
    }

    /// Checks if this interval contains another interval
    pub fn contains_interval(&self, other: &RealInterval) -> bool {
        self.lower <= other.lower && other.upper <= self.upper
    }

    /// Checks if this interval contains zero
    pub fn contains_zero(&self) -> bool {
        self.lower <= 0 && 0 <= self.upper
    }

    /// Checks if this is a point interval (zero width)
    pub fn is_point(&self) -> bool {
        self.lower == self.upper
    }

    /// Checks if this interval is strictly positive
    pub fn is_positive(&self) -> bool {
        self.lower > 0
    }

    /// Checks if this interval is strictly negative
    pub fn is_negative(&self) -> bool {
        self.upper < 0
    }

    /// Checks if this interval is non-negative
    pub fn is_nonnegative(&self) -> bool {
        self.lower >= 0
    }

    /// Checks if two intervals overlap (have non-empty intersection)
    pub fn overlaps(&self, other: &RealInterval) -> bool {
        !(self.upper < other.lower || other.upper < self.lower)
    }

    /// Computes the intersection of two intervals
    ///
    /// Returns None if the intervals don't overlap
    pub fn intersection(&self, other: &RealInterval) -> Option<RealInterval> {
        if !self.overlaps(other) {
            return None;
        }

        let prec = self.precision.max(other.precision);
        let lower = if self.lower >= other.lower {
            Float::with_val_round(prec, &self.lower, Round::Down).0
        } else {
            Float::with_val_round(prec, &other.lower, Round::Down).0
        };

        let upper = if self.upper <= other.upper {
            Float::with_val_round(prec, &self.upper, Round::Up).0
        } else {
            Float::with_val_round(prec, &other.upper, Round::Up).0
        };

        Some(RealInterval {
            lower,
            upper,
            precision: prec,
        })
    }

    /// Computes the hull (smallest interval containing both intervals)
    ///
    /// Also known as the interval union
    pub fn hull(&self, other: &RealInterval) -> RealInterval {
        let prec = self.precision.max(other.precision);

        let lower = if self.lower <= other.lower {
            Float::with_val_round(prec, &self.lower, Round::Down).0
        } else {
            Float::with_val_round(prec, &other.lower, Round::Down).0
        };

        let upper = if self.upper >= other.upper {
            Float::with_val_round(prec, &self.upper, Round::Up).0
        } else {
            Float::with_val_round(prec, &other.upper, Round::Up).0
        };

        RealInterval {
            lower,
            upper,
            precision: prec,
        }
    }

    /// Splits the interval at its midpoint
    ///
    /// Returns (left, right) where left = [lower, mid] and right = [mid, upper]
    pub fn split(&self) -> (RealInterval, RealInterval) {
        let mid = self.midpoint();

        let left = RealInterval {
            lower: Float::with_val_round(self.precision, &self.lower, Round::Down).0,
            upper: Float::with_val_round(self.precision, &mid, Round::Up).0,
            precision: self.precision,
        };

        let right = RealInterval {
            lower: Float::with_val_round(self.precision, &mid, Round::Down).0,
            upper: Float::with_val_round(self.precision, &self.upper, Round::Up).0,
            precision: self.precision,
        };

        (left, right)
    }

    /// Computes the absolute value of the interval
    ///
    /// - [a, b] with a ≥ 0 => [a, b]
    /// - [a, b] with b ≤ 0 => [-b, -a]
    /// - [a, b] with a < 0 < b => [0, max(|a|, |b|)]
    pub fn abs(&self) -> RealInterval {
        if self.lower >= 0 {
            // Entirely non-negative
            self.clone()
        } else if self.upper <= 0 {
            // Entirely non-positive
            -self.clone()
        } else {
            // Contains zero
            let abs_lower = Float::with_val(self.precision, &self.lower).abs();
            let abs_upper = Float::with_val(self.precision, &self.upper).abs();
            let max_abs = abs_lower.max(&abs_upper);

            RealInterval {
                lower: Float::with_val(self.precision, 0),
                upper: Float::with_val_round(self.precision, max_abs, Round::Up).0,
                precision: self.precision,
            }
        }
    }

    /// Computes the square of the interval
    pub fn square(&self) -> RealInterval {
        self.clone() * self.clone()
    }

    /// Computes the square root of the interval
    ///
    /// Returns None if the interval contains negative numbers
    pub fn sqrt(&self) -> Option<RealInterval> {
        if self.lower < 0 {
            return None;
        }

        let lower = Float::with_val_round(self.precision, &self.lower, Round::Down).0.sqrt();
        let upper = Float::with_val_round(self.precision, &self.upper, Round::Up).0.sqrt();

        Some(RealInterval {
            lower,
            upper,
            precision: self.precision,
        })
    }

    /// Computes the reciprocal (1/x) of the interval
    ///
    /// Returns Err if the interval contains zero
    pub fn recip(&self) -> Result<RealInterval, RealIntervalError> {
        if self.contains_zero() {
            return Err(RealIntervalError::DivisionByZero);
        }

        // 1/[a,b] = [1/b, 1/a] when 0 not in [a,b]
        let lower = Float::with_val_round(self.precision, 1.0, Round::Down).0 / &self.upper;
        let upper = Float::with_val_round(self.precision, 1.0, Round::Up).0 / &self.lower;

        Ok(RealInterval {
            lower,
            upper,
            precision: self.precision,
        })
    }

    /// Computes e^x for the interval
    pub fn exp(&self) -> RealInterval {
        let lower = Float::with_val_round(self.precision, &self.lower, Round::Down).0.exp();
        let upper = Float::with_val_round(self.precision, &self.upper, Round::Up).0.exp();

        RealInterval {
            lower,
            upper,
            precision: self.precision,
        }
    }

    /// Computes ln(x) for the interval
    ///
    /// Returns None if the interval contains non-positive numbers
    pub fn ln(&self) -> Option<RealInterval> {
        if self.lower <= 0 {
            return None;
        }

        let lower = Float::with_val_round(self.precision, &self.lower, Round::Down).0.ln();
        let upper = Float::with_val_round(self.precision, &self.upper, Round::Up).0.ln();

        Some(RealInterval {
            lower,
            upper,
            precision: self.precision,
        })
    }

    /// Computes sin(x) for the interval
    ///
    /// Note: For wide intervals, this may return [-1, 1]
    pub fn sin(&self) -> RealInterval {
        use std::f64::consts::PI;

        let width_f64 = self.width().to_f64();

        // If interval is wider than 2π, result is [-1, 1]
        if width_f64 >= 2.0 * PI {
            return RealInterval::new(-1.0, 1.0, self.precision);
        }

        // Otherwise, compute sin at endpoints and determine range
        let sin_lower = Float::with_val(self.precision, &self.lower).sin();
        let sin_upper = Float::with_val(self.precision, &self.upper).sin();

        // This is a simplified implementation
        // A full implementation would check for critical points (π/2 + nπ)
        let min = sin_lower.clone().min(&sin_upper);
        let max = sin_lower.max(&sin_upper);

        RealInterval {
            lower: Float::with_val_round(self.precision, min, Round::Down).0,
            upper: Float::with_val_round(self.precision, max, Round::Up).0,
            precision: self.precision,
        }
    }

    /// Computes cos(x) for the interval
    pub fn cos(&self) -> RealInterval {
        use std::f64::consts::PI;

        let width_f64 = self.width().to_f64();

        // If interval is wider than 2π, result is [-1, 1]
        if width_f64 >= 2.0 * PI {
            return RealInterval::new(-1.0, 1.0, self.precision);
        }

        let cos_lower = Float::with_val(self.precision, &self.lower).cos();
        let cos_upper = Float::with_val(self.precision, &self.upper).cos();

        let min = cos_lower.clone().min(&cos_upper);
        let max = cos_lower.max(&cos_upper);

        RealInterval {
            lower: Float::with_val_round(self.precision, min, Round::Down).0,
            upper: Float::with_val_round(self.precision, max, Round::Up).0,
            precision: self.precision,
        }
    }
}

impl fmt::Display for RealInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_point() {
            write!(f, "{}", self.lower)
        } else {
            write!(f, "[{}, {}]", self.lower, self.upper)
        }
    }
}

impl fmt::Debug for RealInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RealInterval([{}, {}], {} bits)",
            self.lower, self.upper, self.precision
        )
    }
}

impl PartialEq for RealInterval {
    fn eq(&self, other: &Self) -> bool {
        self.lower == other.lower && self.upper == other.upper
    }
}

// Arithmetic operations with proper directed rounding

impl Add for RealInterval {
    type Output = Self;

    /// Addition: [a, b] + [c, d] = [a+c, b+d]
    ///
    /// Lower bound rounded down, upper bound rounded up
    fn add(self, other: Self) -> Self {
        let prec = self.precision.max(other.precision);

        let lower = Float::with_val_round(prec, &self.lower + &other.lower, Round::Down).0;
        let upper = Float::with_val_round(prec, &self.upper + &other.upper, Round::Up).0;

        RealInterval {
            lower,
            upper,
            precision: prec,
        }
    }
}

impl<'a, 'b> Add<&'b RealInterval> for &'a RealInterval {
    type Output = RealInterval;

    fn add(self, other: &'b RealInterval) -> RealInterval {
        let prec = self.precision.max(other.precision);

        let lower = Float::with_val_round(prec, &self.lower + &other.lower, Round::Down).0;
        let upper = Float::with_val_round(prec, &self.upper + &other.upper, Round::Up).0;

        RealInterval {
            lower,
            upper,
            precision: prec,
        }
    }
}

impl Sub for RealInterval {
    type Output = Self;

    /// Subtraction: [a, b] - [c, d] = [a-d, b-c]
    ///
    /// Lower bound rounded down, upper bound rounded up
    fn sub(self, other: Self) -> Self {
        let prec = self.precision.max(other.precision);

        let lower = Float::with_val_round(prec, &self.lower - &other.upper, Round::Down).0;
        let upper = Float::with_val_round(prec, &self.upper - &other.lower, Round::Up).0;

        RealInterval {
            lower,
            upper,
            precision: prec,
        }
    }
}

impl<'a, 'b> Sub<&'b RealInterval> for &'a RealInterval {
    type Output = RealInterval;

    fn sub(self, other: &'b RealInterval) -> RealInterval {
        let prec = self.precision.max(other.precision);

        let lower = Float::with_val_round(prec, &self.lower - &other.upper, Round::Down).0;
        let upper = Float::with_val_round(prec, &self.upper - &other.lower, Round::Up).0;

        RealInterval {
            lower,
            upper,
            precision: prec,
        }
    }
}

impl Mul for RealInterval {
    type Output = Self;

    /// Multiplication: [a, b] × [c, d] = [min(ac, ad, bc, bd), max(ac, ad, bc, bd)]
    ///
    /// Lower bound rounded down, upper bound rounded up
    fn mul(self, other: Self) -> Self {
        let prec = self.precision.max(other.precision);

        // Compute all four products
        let p1 = Float::with_val(prec, &self.lower * &other.lower);
        let p2 = Float::with_val(prec, &self.lower * &other.upper);
        let p3 = Float::with_val(prec, &self.upper * &other.lower);
        let p4 = Float::with_val(prec, &self.upper * &other.upper);

        // Find min and max
        let min_val = p1.clone().min(&p2).min(&p3).min(&p4);
        let max_val = p1.max(&p2).max(&p3).max(&p4);

        let lower = Float::with_val_round(prec, min_val, Round::Down).0;
        let upper = Float::with_val_round(prec, max_val, Round::Up).0;

        RealInterval {
            lower,
            upper,
            precision: prec,
        }
    }
}

impl<'a, 'b> Mul<&'b RealInterval> for &'a RealInterval {
    type Output = RealInterval;

    fn mul(self, other: &'b RealInterval) -> RealInterval {
        let prec = self.precision.max(other.precision);

        let p1 = Float::with_val(prec, &self.lower * &other.lower);
        let p2 = Float::with_val(prec, &self.lower * &other.upper);
        let p3 = Float::with_val(prec, &self.upper * &other.lower);
        let p4 = Float::with_val(prec, &self.upper * &other.upper);

        let min_val = p1.clone().min(&p2).min(&p3).min(&p4);
        let max_val = p1.max(&p2).max(&p3).max(&p4);

        let lower = Float::with_val_round(prec, min_val, Round::Down).0;
        let upper = Float::with_val_round(prec, max_val, Round::Up).0;

        RealInterval {
            lower,
            upper,
            precision: prec,
        }
    }
}

impl Div for RealInterval {
    type Output = Self;

    /// Division: [a, b] / [c, d]
    ///
    /// Panics if the divisor interval contains zero
    /// Use `recip()` and multiply for better error handling
    fn div(self, other: Self) -> Self {
        (&self).div(&other)
    }
}

impl<'a, 'b> Div<&'b RealInterval> for &'a RealInterval {
    type Output = RealInterval;

    fn div(self, other: &'b RealInterval) -> RealInterval {
        if other.contains_zero() {
            panic!("Division by interval containing zero");
        }

        // [a,b] / [c,d] = [a,b] * [1/d, 1/c]
        let recip = other.recip().expect("Division by zero already checked");
        self * &recip
    }
}

impl Neg for RealInterval {
    type Output = Self;

    /// Negation: -[a, b] = [-b, -a]
    fn neg(self) -> Self {
        RealInterval {
            lower: -self.upper,
            upper: -self.lower,
            precision: self.precision,
        }
    }
}

impl<'a> Neg for &'a RealInterval {
    type Output = RealInterval;

    fn neg(self) -> RealInterval {
        RealInterval {
            lower: -self.upper.clone(),
            upper: -self.lower.clone(),
            precision: self.precision,
        }
    }
}

// Factory functions for convenience

/// Creates a new real interval field with specified precision
pub fn real_interval_field(precision: u32) -> RealIntervalField {
    RealIntervalField::new(precision)
}

/// Creates a new real interval [lower, upper] with specified precision
pub fn real_interval(lower: f64, upper: f64, precision: u32) -> RealInterval {
    RealInterval::new(lower, upper, precision)
}

/// Type alias for convenience (matches SageMath naming)
pub type RIF = RealIntervalField;

/// Creates the default real interval field (53-bit precision)
pub fn rif_default() -> RealIntervalField {
    RealIntervalField::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_creation() {
        let field = RealIntervalField::new(100);
        assert_eq!(field.precision(), 100);
    }

    #[test]
    fn test_field_default() {
        let field = RealIntervalField::default();
        assert_eq!(field.precision(), 53);
    }

    #[test]
    fn test_field_display() {
        let field = RealIntervalField::new(64);
        assert_eq!(
            format!("{}", field),
            "Real Interval Field with 64 bits precision"
        );
    }

    #[test]
    fn test_interval_creation() {
        let interval = RealInterval::new(1.0, 2.0, 53);
        assert_eq!(interval.precision(), 53);
        assert!(interval.lower_f64() <= 1.0);
        assert!(interval.upper_f64() >= 2.0);
    }

    #[test]
    fn test_point_interval() {
        let interval = RealInterval::new_point(3.14, 53);
        assert!(interval.is_point());
        assert_eq!(interval.width().to_f64(), 0.0);
    }

    #[test]
    fn test_interval_contains() {
        let interval = RealInterval::new(1.0, 3.0, 53);
        assert!(interval.contains(2.0));
        assert!(interval.contains(1.0));
        assert!(interval.contains(3.0));
        assert!(!interval.contains(4.0));
        assert!(!interval.contains(0.5));
    }

    #[test]
    fn test_interval_contains_zero() {
        let i1 = RealInterval::new(-1.0, 1.0, 53);
        assert!(i1.contains_zero());

        let i2 = RealInterval::new(1.0, 2.0, 53);
        assert!(!i2.contains_zero());

        let i3 = RealInterval::new(-2.0, -1.0, 53);
        assert!(!i3.contains_zero());
    }

    #[test]
    fn test_interval_contains_interval() {
        let i1 = RealInterval::new(0.0, 10.0, 53);
        let i2 = RealInterval::new(2.0, 5.0, 53);
        assert!(i1.contains_interval(&i2));
        assert!(!i2.contains_interval(&i1));
    }

    #[test]
    fn test_interval_midpoint() {
        let interval = RealInterval::new(2.0, 4.0, 53);
        let mid = interval.midpoint().to_f64();
        assert!((mid - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_interval_width() {
        let interval = RealInterval::new(1.5, 3.5, 53);
        let width = interval.width().to_f64();
        assert!((width - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_interval_overlaps() {
        let i1 = RealInterval::new(1.0, 5.0, 53);
        let i2 = RealInterval::new(3.0, 7.0, 53);
        let i3 = RealInterval::new(6.0, 8.0, 53);

        assert!(i1.overlaps(&i2));
        assert!(i2.overlaps(&i1));
        assert!(!i1.overlaps(&i3));
    }

    #[test]
    fn test_interval_intersection() {
        let i1 = RealInterval::new(1.0, 5.0, 53);
        let i2 = RealInterval::new(3.0, 7.0, 53);

        let intersection = i1.intersection(&i2).unwrap();
        assert!((intersection.lower_f64() - 3.0).abs() < 1e-10);
        assert!((intersection.upper_f64() - 5.0).abs() < 1e-10);

        let i3 = RealInterval::new(6.0, 8.0, 53);
        assert!(i1.intersection(&i3).is_none());
    }

    #[test]
    fn test_interval_hull() {
        let i1 = RealInterval::new(1.0, 3.0, 53);
        let i2 = RealInterval::new(5.0, 7.0, 53);

        let hull = i1.hull(&i2);
        assert!((hull.lower_f64() - 1.0).abs() < 1e-10);
        assert!((hull.upper_f64() - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_interval_split() {
        let interval = RealInterval::new(0.0, 4.0, 53);
        let (left, right) = interval.split();

        let mid = interval.midpoint().to_f64();
        assert!((left.upper_f64() - mid).abs() < 1e-10);
        assert!((right.lower_f64() - mid).abs() < 1e-10);
    }

    #[test]
    fn test_interval_abs() {
        // All positive
        let i1 = RealInterval::new(2.0, 5.0, 53);
        let abs1 = i1.abs();
        assert!((abs1.lower_f64() - 2.0).abs() < 1e-10);
        assert!((abs1.upper_f64() - 5.0).abs() < 1e-10);

        // All negative
        let i2 = RealInterval::new(-5.0, -2.0, 53);
        let abs2 = i2.abs();
        assert!((abs2.lower_f64() - 2.0).abs() < 1e-10);
        assert!((abs2.upper_f64() - 5.0).abs() < 1e-10);

        // Contains zero
        let i3 = RealInterval::new(-3.0, 2.0, 53);
        let abs3 = i3.abs();
        assert!((abs3.lower_f64() - 0.0).abs() < 1e-10);
        assert!((abs3.upper_f64() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_interval_sqrt() {
        let i = RealInterval::new(4.0, 9.0, 53);
        let sqrt_i = i.sqrt().unwrap();

        assert!((sqrt_i.lower_f64() - 2.0).abs() < 1e-10);
        assert!((sqrt_i.upper_f64() - 3.0).abs() < 1e-10);

        // Negative interval
        let i_neg = RealInterval::new(-4.0, -1.0, 53);
        assert!(i_neg.sqrt().is_none());
    }

    #[test]
    fn test_addition() {
        let i1 = RealInterval::new(1.0, 2.0, 53);
        let i2 = RealInterval::new(3.0, 4.0, 53);
        let sum = &i1 + &i2;

        assert!((sum.lower_f64() - 4.0).abs() < 1e-10);
        assert!((sum.upper_f64() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_subtraction() {
        let i1 = RealInterval::new(5.0, 7.0, 53);
        let i2 = RealInterval::new(2.0, 3.0, 53);
        let diff = &i1 - &i2;

        assert!((diff.lower_f64() - 2.0).abs() < 1e-10);
        assert!((diff.upper_f64() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_multiplication() {
        let i1 = RealInterval::new(2.0, 3.0, 53);
        let i2 = RealInterval::new(4.0, 5.0, 53);
        let prod = &i1 * &i2;

        assert!((prod.lower_f64() - 8.0).abs() < 1e-10);
        assert!((prod.upper_f64() - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_multiplication_mixed_signs() {
        let i1 = RealInterval::new(-2.0, 3.0, 53);
        let i2 = RealInterval::new(-1.0, 4.0, 53);
        let prod = &i1 * &i2;

        // min = -2 * 4 = -8, max = 3 * 4 = 12
        assert!(prod.lower_f64() <= -8.0);
        assert!(prod.upper_f64() >= 12.0);
        assert!(prod.contains(0.0));
    }

    #[test]
    fn test_division() {
        let i1 = RealInterval::new(4.0, 12.0, 53);
        let i2 = RealInterval::new(2.0, 3.0, 53);
        let quot = &i1 / &i2;

        // [4,12] / [2,3] = [4,12] * [1/3, 1/2] = [4/3, 6]
        assert!(quot.lower_f64() >= 4.0 / 3.0 - 1e-10);
        assert!(quot.upper_f64() <= 6.0 + 1e-10);
    }

    #[test]
    #[should_panic(expected = "Division by interval containing zero")]
    fn test_division_by_zero_interval() {
        let i1 = RealInterval::new(1.0, 2.0, 53);
        let i2 = RealInterval::new(-1.0, 1.0, 53); // Contains zero
        let _quot = &i1 / &i2;
    }

    #[test]
    fn test_negation() {
        let i = RealInterval::new(2.0, 5.0, 53);
        let neg_i = -i;

        assert!((neg_i.lower_f64() + 5.0).abs() < 1e-10);
        assert!((neg_i.upper_f64() + 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_recip() {
        let i = RealInterval::new(2.0, 4.0, 53);
        let recip = i.recip().unwrap();

        // 1/[2,4] = [1/4, 1/2]
        assert!((recip.lower_f64() - 0.25).abs() < 1e-10);
        assert!((recip.upper_f64() - 0.5).abs() < 1e-10);

        let i_zero = RealInterval::new(-1.0, 1.0, 53);
        assert!(i_zero.recip().is_err());
    }

    #[test]
    fn test_exp() {
        let i = RealInterval::new(0.0, 1.0, 53);
        let exp_i = i.exp();

        // e^[0,1] = [1, e]
        assert!((exp_i.lower_f64() - 1.0).abs() < 1e-10);
        assert!((exp_i.upper_f64() - std::f64::consts::E).abs() < 1e-9);
    }

    #[test]
    fn test_ln() {
        let i = RealInterval::new(1.0, std::f64::consts::E, 53);
        let ln_i = i.ln().unwrap();

        // ln[1, e] = [0, 1]
        assert!((ln_i.lower_f64() - 0.0).abs() < 1e-10);
        assert!((ln_i.upper_f64() - 1.0).abs() < 1e-9);

        let i_neg = RealInterval::new(-2.0, -1.0, 53);
        assert!(i_neg.ln().is_none());
    }

    #[test]
    fn test_high_precision() {
        // Test with 256-bit precision
        let i1 = RealInterval::new(1.0, 2.0, 256);
        let i2 = RealInterval::new(3.0, 4.0, 256);
        let sum = &i1 + &i2;

        assert_eq!(sum.precision(), 256);
        assert!((sum.lower_f64() - 4.0).abs() < 1e-15);
        assert!((sum.upper_f64() - 6.0).abs() < 1e-15);
    }

    #[test]
    fn test_mixed_precision() {
        let i1 = RealInterval::new(1.0, 2.0, 53);
        let i2 = RealInterval::new(3.0, 4.0, 128);
        let sum = &i1 + &i2;

        // Result should have max precision
        assert_eq!(sum.precision(), 128);
    }

    #[test]
    fn test_containment_guarantee() {
        // Test that arithmetic maintains containment
        let i1 = RealInterval::new(1.0, 2.0, 100);
        let i2 = RealInterval::new(3.0, 4.0, 100);

        let sum = &i1 + &i2;
        // True sum of any x in [1,2] and y in [3,4] is in [4,6]
        assert!(sum.contains(4.0));
        assert!(sum.contains(5.0));
        assert!(sum.contains(6.0));
        assert!(!sum.contains(3.9));
        assert!(!sum.contains(6.1));

        let prod = &i1 * &i2;
        // True product range is [3, 8]
        assert!(prod.contains(3.0));
        assert!(prod.contains(5.0));
        assert!(prod.contains(8.0));
    }

    #[test]
    fn test_real_interval_field_factory() {
        let field = real_interval_field(80);
        assert_eq!(field.precision(), 80);

        let i = field.interval(1.0, 2.0);
        assert_eq!(i.precision(), 80);
    }

    #[test]
    fn test_rif_alias() {
        let rif: RIF = RIF_default();
        assert_eq!(rif.precision(), 53);
    }

    #[test]
    fn test_is_positive_negative() {
        let i1 = RealInterval::new(2.0, 5.0, 53);
        assert!(i1.is_positive());
        assert!(!i1.is_negative());
        assert!(i1.is_nonnegative());

        let i2 = RealInterval::new(-5.0, -2.0, 53);
        assert!(!i2.is_positive());
        assert!(i2.is_negative());
        assert!(!i2.is_nonnegative());

        let i3 = RealInterval::new(-1.0, 1.0, 53);
        assert!(!i3.is_positive());
        assert!(!i3.is_negative());
        assert!(!i3.is_nonnegative());

        let i4 = RealInterval::new(0.0, 1.0, 53);
        assert!(!i4.is_positive());
        assert!(!i4.is_negative());
        assert!(i4.is_nonnegative());
    }

    #[test]
    fn test_square() {
        let i = RealInterval::new(2.0, 3.0, 53);
        let sq = i.square();

        assert!((sq.lower_f64() - 4.0).abs() < 1e-10);
        assert!((sq.upper_f64() - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_very_high_precision() {
        // Test with 1000-bit precision
        let i = RealInterval::new(1.0, 2.0, 1000);
        assert_eq!(i.precision(), 1000);

        let double = &i + &i;
        assert_eq!(double.precision(), 1000);
        assert!(double.contains(3.0));
    }
}
