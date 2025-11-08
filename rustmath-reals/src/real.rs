//! Real number implementation with arbitrary precision support

use rustmath_core::{CommutativeRing, Field, MathError, NumericConversion, Result, Ring};
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Real number with configurable precision
///
/// Currently wraps f64 for basic functionality.
/// Future versions will support arbitrary precision arithmetic.
#[derive(Clone, Debug)]
pub struct Real {
    value: f64,
}

impl Real {
    /// Create a new Real from an f64
    pub fn new(value: f64) -> Self {
        Real { value }
    }

    /// Create Real from an integer
    pub fn from_integer(n: &Integer) -> Self {
        Real {
            value: n.to_f64().unwrap_or(f64::NAN),
        }
    }

    /// Create Real from a rational
    pub fn from_rational(r: &Rational) -> Self {
        Real {
            value: r.to_f64().unwrap_or(f64::NAN),
        }
    }

    /// Convert to f64
    pub fn to_f64(&self) -> f64 {
        self.value
    }

    /// Get the absolute value
    pub fn abs(&self) -> Self {
        Real {
            value: self.value.abs(),
        }
    }

    /// Get the floor (greatest integer ≤ self)
    pub fn floor(&self) -> Integer {
        Integer::from(self.value.floor() as i64)
    }

    /// Get the ceiling (smallest integer ≥ self)
    pub fn ceil(&self) -> Integer {
        Integer::from(self.value.ceil() as i64)
    }

    /// Round to nearest integer
    pub fn round(&self) -> Integer {
        Integer::from(self.value.round() as i64)
    }

    /// Check if the value is NaN
    pub fn is_nan(&self) -> bool {
        self.value.is_nan()
    }

    /// Check if the value is infinite
    pub fn is_infinite(&self) -> bool {
        self.value.is_infinite()
    }

    /// Check if the value is finite
    pub fn is_finite(&self) -> bool {
        self.value.is_finite()
    }

    /// Get the sign (-1, 0, or 1)
    pub fn signum(&self) -> Self {
        Real {
            value: self.value.signum(),
        }
    }

    /// Compute self raised to the power of exp
    pub fn pow(&self, exp: &Self) -> Self {
        Real {
            value: self.value.powf(exp.value),
        }
    }

    /// Compute integer power
    pub fn powi(&self, exp: i32) -> Self {
        Real {
            value: self.value.powi(exp),
        }
    }
}

impl fmt::Display for Real {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl From<f64> for Real {
    fn from(value: f64) -> Self {
        Real::new(value)
    }
}

impl From<i32> for Real {
    fn from(value: i32) -> Self {
        Real::new(value as f64)
    }
}

impl From<i64> for Real {
    fn from(value: i64) -> Self {
        Real::new(value as f64)
    }
}

impl Add for Real {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Real {
            value: self.value + other.value,
        }
    }
}

impl Sub for Real {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Real {
            value: self.value - other.value,
        }
    }
}

impl Mul for Real {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Real {
            value: self.value * other.value,
        }
    }
}

impl Div for Real {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Real {
            value: self.value / other.value,
        }
    }
}

impl Neg for Real {
    type Output = Self;

    fn neg(self) -> Self {
        Real { value: -self.value }
    }
}

impl PartialEq for Real {
    fn eq(&self, other: &Self) -> bool {
        (self.value - other.value).abs() < 1e-10
    }
}

impl Ring for Real {
    fn zero() -> Self {
        Real { value: 0.0 }
    }

    fn one() -> Self {
        Real { value: 1.0 }
    }

    fn is_zero(&self) -> bool {
        self.value.abs() < 1e-10
    }

    fn is_one(&self) -> bool {
        (self.value - 1.0).abs() < 1e-10
    }
}

impl CommutativeRing for Real {}

impl Field for Real {
    fn inverse(&self) -> Result<Self> {
        if self.is_zero() {
            Err(MathError::DivisionByZero)
        } else {
            Ok(Real {
                value: 1.0 / self.value,
            })
        }
    }
}

impl NumericConversion for Real {
    fn from_i64(n: i64) -> Self {
        Real {
            value: n as f64,
        }
    }

    fn from_u64(n: u64) -> Self {
        Real {
            value: n as f64,
        }
    }

    fn to_i64(&self) -> Option<i64> {
        if self.value.is_finite() && self.value >= i64::MIN as f64 && self.value <= i64::MAX as f64 {
            Some(self.value as i64)
        } else {
            None
        }
    }

    fn to_u64(&self) -> Option<u64> {
        if self.value.is_finite() && self.value >= 0.0 && self.value <= u64::MAX as f64 {
            Some(self.value as u64)
        } else {
            None
        }
    }

    fn to_usize(&self) -> Option<usize> {
        if self.value.is_finite() && self.value >= 0.0 && self.value <= usize::MAX as f64 {
            Some(self.value as usize)
        } else {
            None
        }
    }

    fn to_f64(&self) -> Option<f64> {
        Some(self.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_ops() {
        let a = Real::from(3.0);
        let b = Real::from(4.0);

        let sum = a.clone() + b.clone();
        assert!((sum.to_f64() - 7.0).abs() < 1e-10);

        let prod = a.clone() * b.clone();
        assert!((prod.to_f64() - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_integer() {
        let n = Integer::from(42);
        let r = Real::from_integer(&n);
        assert!((r.to_f64() - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_rational() {
        let rat = Rational::new(Integer::from(3), Integer::from(4)).unwrap();
        let r = Real::from_rational(&rat);
        assert!((r.to_f64() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_floor_ceil() {
        let r = Real::from(3.7);
        assert_eq!(r.floor(), Integer::from(3));
        assert_eq!(r.ceil(), Integer::from(4));
        assert_eq!(r.round(), Integer::from(4));
    }
}
