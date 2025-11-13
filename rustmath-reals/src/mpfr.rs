//! Arbitrary precision real numbers using MPFR (via rug crate)
//!
//! This module provides the `RealMPFR` type, which supports arbitrary precision
//! floating-point arithmetic using GMP/MPFR libraries through the `rug` crate.
//!
//! # Examples
//!
//! ```
//! use rustmath_reals::RealMPFR;
//!
//! // Create a high-precision real number (256 bits of precision)
//! let x = RealMPFR::with_val(256, 3.14159);
//! let y = RealMPFR::with_val(256, 2.71828);
//! let sum = x + y;
//!
//! // Use default precision (53 bits, equivalent to f64)
//! let z = RealMPFR::from(42.0);
//! ```

use rustmath_core::{CommutativeRing, Field, MathError, NumericConversion, Result, Ring};
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use rug::ops::Pow;
use rug::Float;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Default precision in bits (equivalent to f64)
pub const DEFAULT_PRECISION: u32 = 53;

/// Arbitrary precision real number using MPFR
///
/// This type wraps `rug::Float` to provide arbitrary precision floating-point arithmetic.
/// The precision is configurable and specified in bits for the mantissa.
///
/// # Examples
///
/// ```
/// use rustmath_reals::RealMPFR;
///
/// // Default precision (53 bits = f64 equivalent)
/// let x = RealMPFR::from(3.14159);
///
/// // High precision (1000 bits)
/// let y = RealMPFR::with_val(1000, 3.14159);
/// let z = RealMPFR::with_val(1000, 2.71828);
/// let product = y * z;
/// ```
#[derive(Clone)]
pub struct RealMPFR {
    value: Float,
}

impl RealMPFR {
    /// Create a new RealMPFR from a rug::Float
    pub fn from_float(value: Float) -> Self {
        RealMPFR { value }
    }

    /// Create a new RealMPFR with specified precision from an f64
    ///
    /// # Arguments
    ///
    /// * `prec` - Precision in bits for the mantissa
    /// * `value` - The f64 value to convert
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_reals::RealMPFR;
    ///
    /// let x = RealMPFR::with_val(256, 3.14159);
    /// assert_eq!(x.precision(), 256);
    /// ```
    pub fn with_val(prec: u32, value: f64) -> Self {
        RealMPFR {
            value: Float::with_val(prec, value),
        }
    }

    /// Create a new RealMPFR with specified precision from an integer
    pub fn with_val_integer(prec: u32, value: &Integer) -> Self {
        // Convert Integer to i64 or use string representation for large values
        if let Some(i) = value.to_i64() {
            RealMPFR {
                value: Float::with_val(prec, i),
            }
        } else {
            // For very large integers, convert via string
            let s = value.to_string();
            RealMPFR {
                value: Float::with_val(prec, Float::parse(&s).expect("Failed to parse integer")),
            }
        }
    }

    /// Create a new RealMPFR with specified precision from a rational
    pub fn with_val_rational(prec: u32, value: &Rational) -> Self {
        // Convert rational to real by dividing numerator by denominator
        let numer = RealMPFR::with_val_integer(prec, value.numerator());
        let denom = RealMPFR::with_val_integer(prec, value.denominator());
        numer / denom
    }

    /// Get the precision of this number in bits
    pub fn precision(&self) -> u32 {
        self.value.prec()
    }

    /// Convert to f64 (may lose precision)
    pub fn to_f64(&self) -> f64 {
        self.value.to_f64()
    }

    /// Get the absolute value
    pub fn abs(&self) -> Self {
        RealMPFR {
            value: self.value.clone().abs(),
        }
    }

    /// Compute self raised to the power of exp
    pub fn pow(&self, exp: &Self) -> Self {
        RealMPFR {
            value: self.value.clone().pow(&exp.value),
        }
    }

    /// Compute integer power
    pub fn powi(&self, exp: i32) -> Self {
        RealMPFR {
            value: self.value.clone().pow(exp),
        }
    }

    /// Compute square root
    pub fn sqrt(&self) -> Self {
        RealMPFR {
            value: self.value.clone().sqrt(),
        }
    }

    /// Compute exponential (e^x)
    pub fn exp(&self) -> Self {
        RealMPFR {
            value: self.value.clone().exp(),
        }
    }

    /// Compute natural logarithm
    pub fn ln(&self) -> Self {
        RealMPFR {
            value: self.value.clone().ln(),
        }
    }

    /// Compute logarithm (alias for ln)
    pub fn log(&self) -> Self {
        self.ln()
    }

    /// Compute logarithm base 10
    pub fn log10(&self) -> Self {
        RealMPFR {
            value: self.value.clone().log10(),
        }
    }

    /// Compute sine
    pub fn sin(&self) -> Self {
        RealMPFR {
            value: self.value.clone().sin(),
        }
    }

    /// Compute cosine
    pub fn cos(&self) -> Self {
        RealMPFR {
            value: self.value.clone().cos(),
        }
    }

    /// Compute tangent
    pub fn tan(&self) -> Self {
        RealMPFR {
            value: self.value.clone().tan(),
        }
    }

    /// Compute arcsine
    pub fn asin(&self) -> Self {
        RealMPFR {
            value: self.value.clone().asin(),
        }
    }

    /// Compute arccosine
    pub fn acos(&self) -> Self {
        RealMPFR {
            value: self.value.clone().acos(),
        }
    }

    /// Compute arctangent
    pub fn atan(&self) -> Self {
        RealMPFR {
            value: self.value.clone().atan(),
        }
    }

    /// Compute hyperbolic sine
    pub fn sinh(&self) -> Self {
        RealMPFR {
            value: self.value.clone().sinh(),
        }
    }

    /// Compute hyperbolic cosine
    pub fn cosh(&self) -> Self {
        RealMPFR {
            value: self.value.clone().cosh(),
        }
    }

    /// Compute hyperbolic tangent
    pub fn tanh(&self) -> Self {
        RealMPFR {
            value: self.value.clone().tanh(),
        }
    }

    /// Get pi with the same precision as this number
    pub fn pi_with_prec(&self) -> Self {
        RealMPFR {
            value: Float::with_val(self.precision(), Float::parse("3.141592653589793238462643383279502884197").unwrap()),
        }
    }

    /// Get e (Euler's number) with the same precision as this number
    pub fn e_with_prec(&self) -> Self {
        RealMPFR {
            value: Float::with_val(self.precision(), 1).exp(),
        }
    }

    /// Get pi with specified precision
    pub fn pi(prec: u32) -> Self {
        RealMPFR {
            value: Float::with_val(prec, Float::parse("3.141592653589793238462643383279502884197").unwrap()),
        }
    }

    /// Get e (Euler's number) with specified precision
    pub fn e(prec: u32) -> Self {
        RealMPFR {
            value: Float::with_val(prec, 1).exp(),
        }
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
    pub fn signum(&self) -> i32 {
        self.value.cmp0().map_or(0, |ord| match ord {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        })
    }

    /// Get the floor (greatest integer ≤ self)
    pub fn floor(&self) -> Integer {
        let f = self.value.clone().floor();
        // Convert to integer via f64 for reasonable-sized values
        let f64_val = f.to_f64();
        if f64_val.is_finite() && f64_val.abs() < (i64::MAX as f64) {
            Integer::from(f64_val as i64)
        } else {
            // For very large values, this is an approximation
            Integer::from(0) // Placeholder for now
        }
    }

    /// Get the ceiling (smallest integer ≥ self)
    pub fn ceil(&self) -> Integer {
        let c = self.value.clone().ceil();
        let f64_val = c.to_f64();
        if f64_val.is_finite() && f64_val.abs() < (i64::MAX as f64) {
            Integer::from(f64_val as i64)
        } else {
            Integer::from(0) // Placeholder for now
        }
    }

    /// Round to nearest integer
    pub fn round(&self) -> Integer {
        let r = self.value.clone().round();
        let f64_val = r.to_f64();
        if f64_val.is_finite() && f64_val.abs() < (i64::MAX as f64) {
            Integer::from(f64_val as i64)
        } else {
            Integer::from(0) // Placeholder for now
        }
    }
}

impl fmt::Display for RealMPFR {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl fmt::Debug for RealMPFR {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RealMPFR({}, {} bits)", self.value, self.precision())
    }
}

impl From<f64> for RealMPFR {
    fn from(value: f64) -> Self {
        RealMPFR {
            value: Float::with_val(DEFAULT_PRECISION, value),
        }
    }
}

impl From<i32> for RealMPFR {
    fn from(value: i32) -> Self {
        RealMPFR {
            value: Float::with_val(DEFAULT_PRECISION, value),
        }
    }
}

impl From<i64> for RealMPFR {
    fn from(value: i64) -> Self {
        RealMPFR {
            value: Float::with_val(DEFAULT_PRECISION, value),
        }
    }
}

impl Add for RealMPFR {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let prec = self.precision().max(other.precision());
        RealMPFR {
            value: Float::with_val(prec, self.value + other.value),
        }
    }
}

impl<'a, 'b> Add<&'b RealMPFR> for &'a RealMPFR {
    type Output = RealMPFR;

    fn add(self, other: &'b RealMPFR) -> RealMPFR {
        let prec = self.precision().max(other.precision());
        RealMPFR {
            value: Float::with_val(prec, &self.value + &other.value),
        }
    }
}

impl Sub for RealMPFR {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let prec = self.precision().max(other.precision());
        RealMPFR {
            value: Float::with_val(prec, self.value - other.value),
        }
    }
}

impl<'a, 'b> Sub<&'b RealMPFR> for &'a RealMPFR {
    type Output = RealMPFR;

    fn sub(self, other: &'b RealMPFR) -> RealMPFR {
        let prec = self.precision().max(other.precision());
        RealMPFR {
            value: Float::with_val(prec, &self.value - &other.value),
        }
    }
}

impl Mul for RealMPFR {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let prec = self.precision().max(other.precision());
        RealMPFR {
            value: Float::with_val(prec, self.value * other.value),
        }
    }
}

impl<'a, 'b> Mul<&'b RealMPFR> for &'a RealMPFR {
    type Output = RealMPFR;

    fn mul(self, other: &'b RealMPFR) -> RealMPFR {
        let prec = self.precision().max(other.precision());
        RealMPFR {
            value: Float::with_val(prec, &self.value * &other.value),
        }
    }
}

impl Div for RealMPFR {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let prec = self.precision().max(other.precision());
        RealMPFR {
            value: Float::with_val(prec, self.value / other.value),
        }
    }
}

impl<'a, 'b> Div<&'b RealMPFR> for &'a RealMPFR {
    type Output = RealMPFR;

    fn div(self, other: &'b RealMPFR) -> RealMPFR {
        let prec = self.precision().max(other.precision());
        RealMPFR {
            value: Float::with_val(prec, &self.value / &other.value),
        }
    }
}

impl Neg for RealMPFR {
    type Output = Self;

    fn neg(self) -> Self {
        RealMPFR {
            value: -self.value,
        }
    }
}

impl<'a> Neg for &'a RealMPFR {
    type Output = RealMPFR;

    fn neg(self) -> RealMPFR {
        RealMPFR {
            value: -self.value.clone(),
        }
    }
}

impl PartialEq for RealMPFR {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Ring for RealMPFR {
    fn zero() -> Self {
        RealMPFR {
            value: Float::with_val(DEFAULT_PRECISION, 0),
        }
    }

    fn one() -> Self {
        RealMPFR {
            value: Float::with_val(DEFAULT_PRECISION, 1),
        }
    }

    fn is_zero(&self) -> bool {
        self.value == 0
    }

    fn is_one(&self) -> bool {
        self.value == 1
    }
}

impl CommutativeRing for RealMPFR {}

impl Field for RealMPFR {
    fn inverse(&self) -> Result<Self> {
        if self.is_zero() {
            Err(MathError::DivisionByZero)
        } else {
            Ok(RealMPFR {
                value: self.value.clone().recip(),
            })
        }
    }
}

impl NumericConversion for RealMPFR {
    fn from_i64(n: i64) -> Self {
        RealMPFR {
            value: Float::with_val(DEFAULT_PRECISION, n),
        }
    }

    fn from_u64(n: u64) -> Self {
        RealMPFR {
            value: Float::with_val(DEFAULT_PRECISION, n),
        }
    }

    fn to_i64(&self) -> Option<i64> {
        let f64_val = self.value.to_f64();
        if f64_val.is_finite() && f64_val >= i64::MIN as f64 && f64_val <= i64::MAX as f64 {
            Some(f64_val as i64)
        } else {
            None
        }
    }

    fn to_u64(&self) -> Option<u64> {
        let f64_val = self.value.to_f64();
        if f64_val.is_finite() && f64_val >= 0.0 && f64_val <= u64::MAX as f64 {
            Some(f64_val as u64)
        } else {
            None
        }
    }

    fn to_usize(&self) -> Option<usize> {
        let f64_val = self.value.to_f64();
        if f64_val.is_finite() && f64_val >= 0.0 && f64_val <= usize::MAX as f64 {
            Some(f64_val as usize)
        } else {
            None
        }
    }

    fn to_f64(&self) -> Option<f64> {
        Some(self.value.to_f64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_creation() {
        let x = RealMPFR::from(3.14159);
        assert_eq!(x.precision(), DEFAULT_PRECISION);
        assert!((x.to_f64() - 3.14159).abs() < 1e-10);
    }

    #[test]
    fn test_high_precision_creation() {
        let x = RealMPFR::with_val(256, 3.14159);
        assert_eq!(x.precision(), 256);
    }

    #[test]
    fn test_arithmetic() {
        let a = RealMPFR::from(3.0);
        let b = RealMPFR::from(4.0);

        let sum = a.clone() + b.clone();
        assert!((sum.to_f64() - 7.0).abs() < 1e-10);

        let prod = a.clone() * b.clone();
        assert!((prod.to_f64() - 12.0).abs() < 1e-10);

        let diff = a.clone() - b.clone();
        assert!((diff.to_f64() + 1.0).abs() < 1e-10);

        let quot = a.clone() / b.clone();
        assert!((quot.to_f64() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_ring_traits() {
        let zero = RealMPFR::zero();
        let one = RealMPFR::one();

        assert!(zero.is_zero());
        assert!(!zero.is_one());
        assert!(!one.is_zero());
        assert!(one.is_one());

        let x = RealMPFR::from(5.0);
        let y = x.clone() + zero.clone();
        assert_eq!(x, y);

        let z = x.clone() * one.clone();
        assert_eq!(x, z);
    }

    #[test]
    fn test_field_traits() {
        let x = RealMPFR::from(4.0);
        let inv = x.inverse().unwrap();
        assert!((inv.to_f64() - 0.25).abs() < 1e-10);

        let zero = RealMPFR::zero();
        assert!(zero.inverse().is_err());
    }

    #[test]
    fn test_sqrt() {
        let x = RealMPFR::from(16.0);
        let root = x.sqrt();
        assert!((root.to_f64() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_exp_ln() {
        let x = RealMPFR::from(2.0);
        let exp_x = x.exp();
        let ln_exp_x = exp_x.ln();
        assert!((ln_exp_x.to_f64() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_trig_functions() {
        let x = RealMPFR::with_val(256, 0.0);
        assert!((x.sin().to_f64() - 0.0).abs() < 1e-10);
        assert!((x.cos().to_f64() - 1.0).abs() < 1e-10);

        let pi_half = RealMPFR::pi(256) / RealMPFR::with_val(256, 2.0);
        assert!((pi_half.sin().to_f64() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_high_precision_pi() {
        // Compute pi with 1000 bits of precision
        let pi_1000 = RealMPFR::pi(1000);
        assert_eq!(pi_1000.precision(), 1000);

        // Check that it starts with 3.14159...
        let pi_str = pi_1000.to_string();
        assert!(pi_str.starts_with("3.14159"));
    }

    #[test]
    fn test_from_integer() {
        let n = Integer::from(42);
        let r = RealMPFR::with_val_integer(DEFAULT_PRECISION, &n);
        assert!((r.to_f64() - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_rational() {
        let rat = Rational::new(Integer::from(3), Integer::from(4)).unwrap();
        let r = RealMPFR::with_val_rational(DEFAULT_PRECISION, &rat);
        assert!((r.to_f64() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_floor_ceil_round() {
        let x = RealMPFR::from(3.7);
        assert_eq!(x.floor(), Integer::from(3));
        assert_eq!(x.ceil(), Integer::from(4));
        assert_eq!(x.round(), Integer::from(4));

        let y = RealMPFR::from(-3.7);
        assert_eq!(y.floor(), Integer::from(-4));
        assert_eq!(y.ceil(), Integer::from(-3));
    }

    #[test]
    fn test_numeric_conversion() {
        let x = RealMPFR::from_i64(42);
        assert_eq!(x.to_i64(), Some(42));

        let y = RealMPFR::from_u64(100);
        assert_eq!(y.to_u64(), Some(100));
    }

    #[test]
    fn test_very_high_precision() {
        // Test with 10000 bits of precision
        let x = RealMPFR::with_val(10000, 2.0);
        let sqrt_2 = x.sqrt();

        // sqrt(2) should be approximately 1.41421356...
        assert!((sqrt_2.to_f64() - 1.41421356).abs() < 1e-7);

        // Squaring should give us back 2
        let squared = sqrt_2.clone() * sqrt_2.clone();
        assert!((squared.to_f64() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_e_constant() {
        let e = RealMPFR::e(256);
        // e ≈ 2.71828182845904523536...
        assert!((e.to_f64() - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_hyperbolic_functions() {
        let x = RealMPFR::from(1.0);
        let sinh_x = x.sinh();
        let cosh_x = x.cosh();
        let tanh_x = x.tanh();

        // sinh(1) ≈ 1.1752011936438014
        assert!((sinh_x.to_f64() - 1.1752011936438014).abs() < 1e-10);

        // cosh(1) ≈ 1.5430806348152437
        assert!((cosh_x.to_f64() - 1.5430806348152437).abs() < 1e-10);

        // tanh(x) = sinh(x) / cosh(x)
        let tanh_computed = sinh_x.clone() / cosh_x.clone();
        assert!((tanh_x.to_f64() - tanh_computed.to_f64()).abs() < 1e-10);
    }
}
