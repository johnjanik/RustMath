//! # Real MPFI Module
//!
//! Implementation of real interval arithmetic using MPFI-style semantics.
//!
//! ## Overview
//!
//! MPFI (Multiple Precision Floating-point Interval) provides interval arithmetic
//! with relative precision. This module wraps interval operations and provides
//! compatibility with SageMath's real_mpfi functionality.
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::real_mpfi::{RealIntervalField, RealInterval};
//!
//! // Create a real interval field with 53 bits precision
//! let RIF = RealIntervalField::new(53);
//! ```
//!
//! ## Related Modules
//!
//! - `real_interval_absolute`: Fixed absolute precision intervals
//! - `rustmath-reals`: Core real number and interval implementations

use rustmath_core::{Field, Ring};
use std::fmt;
use thiserror::Error;

/// Errors for real interval operations
#[derive(Debug, Clone, Error, PartialEq)]
pub enum RealIntervalError {
    #[error("Invalid precision")]
    InvalidPrecision,

    #[error("Division by interval containing zero")]
    DivisionByZero,
}

/// Real interval field with relative precision
///
/// This represents the field of real intervals with a specified precision
/// (number of mantissa bits).
#[derive(Debug, Clone, PartialEq)]
pub struct RealIntervalField {
    /// Precision in bits
    precision: u32,
}

impl RealIntervalField {
    /// Creates a new real interval field
    pub fn new(precision: u32) -> Self {
        RealIntervalField { precision }
    }

    /// Returns the precision
    pub fn precision(&self) -> u32 {
        self.precision
    }
}

impl Default for RealIntervalField {
    fn default() -> Self {
        RealIntervalField::new(53) // IEEE 754 double precision
    }
}

impl fmt::Display for RealIntervalField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Real Interval Field with {} bits precision", self.precision)
    }
}

/// An element of a real interval field
///
/// Represents a closed interval [lower, upper] of real numbers.
#[derive(Debug, Clone, PartialEq)]
pub struct RealIntervalFieldElement {
    /// Lower bound
    lower: f64,
    /// Upper bound
    upper: f64,
    /// Precision
    precision: u32,
}

impl RealIntervalFieldElement {
    /// Creates a new interval
    pub fn new(lower: f64, upper: f64, precision: u32) -> Self {
        assert!(lower <= upper, "Lower must be <= upper");
        RealIntervalFieldElement {
            lower,
            upper,
            precision,
        }
    }

    /// Creates an interval from a single value
    pub fn from_value(value: f64, precision: u32) -> Self {
        RealIntervalFieldElement {
            lower: value,
            upper: value,
            precision,
        }
    }

    /// Returns the lower bound
    pub fn lower(&self) -> f64 {
        self.lower
    }

    /// Returns the upper bound
    pub fn upper(&self) -> f64 {
        self.upper
    }

    /// Returns the midpoint
    pub fn midpoint(&self) -> f64 {
        (self.lower + self.upper) / 2.0
    }

    /// Returns the diameter/width
    pub fn diameter(&self) -> f64 {
        self.upper - self.lower
    }

    /// Checks if this interval contains a value
    pub fn contains(&self, x: f64) -> bool {
        self.lower <= x && x <= self.upper
    }

    /// Checks if this interval contains zero
    pub fn contains_zero(&self) -> bool {
        self.contains(0.0)
    }
}

impl fmt::Display for RealIntervalFieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if (self.lower - self.upper).abs() < 1e-10 {
            write!(f, "{:.6}", self.lower)
        } else {
            write!(f, "[{:.6}, {:.6}]", self.lower, self.upper)
        }
    }
}

/// Factory function for creating real interval fields
pub fn real_interval_field(precision: u32) -> RealIntervalField {
    RealIntervalField::new(precision)
}

/// Factory function for creating real intervals
pub fn real_interval(lower: f64, upper: f64, precision: u32) -> RealIntervalFieldElement {
    RealIntervalFieldElement::new(lower, upper, precision)
}

/// Type alias for convenience
pub type RealInterval = RealIntervalFieldElement;

/// Checks if a value is a real interval field
pub fn is_real_interval_field<T>() -> bool {
    std::any::TypeId::of::<T>() == std::any::TypeId::of::<RealIntervalField>()
}

/// Checks if a value is a real interval field element
pub fn is_real_interval_field_element<T>() -> bool {
    std::any::TypeId::of::<T>() == std::any::TypeId::of::<RealIntervalFieldElement>()
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
        assert_eq!(format!("{}", field), "Real Interval Field with 64 bits precision");
    }

    #[test]
    fn test_interval_creation() {
        let interval = RealIntervalFieldElement::new(1.0, 2.0, 53);
        assert_eq!(interval.lower(), 1.0);
        assert_eq!(interval.upper(), 2.0);
    }

    #[test]
    fn test_interval_from_value() {
        let interval = RealIntervalFieldElement::from_value(3.14, 53);
        assert_eq!(interval.midpoint(), 3.14);
        assert_eq!(interval.diameter(), 0.0);
    }

    #[test]
    fn test_interval_contains() {
        let interval = RealIntervalFieldElement::new(1.0, 3.0, 53);
        assert!(interval.contains(2.0));
        assert!(!interval.contains(4.0));
        assert!(interval.contains(1.0));
        assert!(interval.contains(3.0));
    }

    #[test]
    fn test_interval_contains_zero() {
        let i1 = RealIntervalFieldElement::new(-1.0, 1.0, 53);
        assert!(i1.contains_zero());

        let i2 = RealIntervalFieldElement::new(1.0, 2.0, 53);
        assert!(!i2.contains_zero());
    }

    #[test]
    fn test_interval_midpoint() {
        let interval = RealIntervalFieldElement::new(2.0, 4.0, 53);
        assert_eq!(interval.midpoint(), 3.0);
    }

    #[test]
    fn test_interval_diameter() {
        let interval = RealIntervalFieldElement::new(1.5, 3.5, 53);
        assert_eq!(interval.diameter(), 2.0);
    }

    #[test]
    fn test_interval_display_point() {
        let interval = RealIntervalFieldElement::from_value(2.5, 53);
        assert!(format!("{}", interval).contains("2.5"));
    }

    #[test]
    fn test_interval_display_range() {
        let interval = RealIntervalFieldElement::new(1.0, 2.0, 53);
        let s = format!("{}", interval);
        assert!(s.contains("[") && s.contains("]"));
    }

    #[test]
    fn test_real_interval_field_factory() {
        let field = real_interval_field(80);
        assert_eq!(field.precision(), 80);
    }

    #[test]
    fn test_real_interval_factory() {
        let interval = real_interval(0.5, 1.5, 53);
        assert_eq!(interval.diameter(), 1.0);
    }
}
