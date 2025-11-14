//! Arbitrary precision complex numbers using MPC (via rug crate)
//!
//! This module provides the `ComplexMPFR` type, which supports arbitrary precision
//! complex arithmetic using GMP/MPFR/MPC libraries through the `rug` crate.
//!
//! # Examples
//!
//! ```
//! use rustmath_complex::ComplexMPFR;
//!
//! // Create a high-precision complex number (256 bits of precision)
//! let z = ComplexMPFR::with_val(256, (3.0, 4.0));
//! let w = ComplexMPFR::with_val(256, (1.0, 2.0));
//! let sum = z + w;
//!
//! // Use default precision (53 bits, equivalent to f64)
//! let z_default = ComplexMPFR::from((3.0, 4.0));
//! ```

use rustmath_core::{CommutativeRing, Field, MathError, NumericConversion, Result, Ring};
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use rustmath_reals::RealMPFR;
use rug::ops::Pow;
use rug::Complex as RugComplex;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Default precision in bits (equivalent to f64)
pub const DEFAULT_PRECISION: u32 = 53;

/// Arbitrary precision complex number using MPC
///
/// This type wraps `rug::Complex` to provide arbitrary precision complex arithmetic.
/// The precision is configurable and specified in bits for the mantissa of both
/// real and imaginary parts.
///
/// # Examples
///
/// ```
/// use rustmath_complex::ComplexMPFR;
///
/// // Default precision (53 bits = f64 equivalent)
/// let z = ComplexMPFR::from((3.0, 4.0));
///
/// // High precision (1000 bits)
/// let w = ComplexMPFR::with_val(1000, (3.0, 4.0));
/// let conj = w.conjugate();
/// ```
#[derive(Clone)]
pub struct ComplexMPFR {
    value: RugComplex,
}

impl ComplexMPFR {
    /// Create a new ComplexMPFR from a rug::Complex
    pub fn from_complex(value: RugComplex) -> Self {
        ComplexMPFR { value }
    }

    /// Create a new ComplexMPFR with specified precision from f64 values
    ///
    /// # Arguments
    ///
    /// * `prec` - Precision in bits for the mantissa
    /// * `value` - Tuple of (real, imaginary) f64 values
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_complex::ComplexMPFR;
    ///
    /// let z = ComplexMPFR::with_val(256, (3.0, 4.0));
    /// assert_eq!(z.precision(), 256);
    /// ```
    pub fn with_val(prec: u32, value: (f64, f64)) -> Self {
        ComplexMPFR {
            value: RugComplex::with_val(prec, value),
        }
    }

    /// Create a new ComplexMPFR with specified precision from RealMPFR values
    pub fn with_val_reals(real: RealMPFR, imag: RealMPFR) -> Self {
        let prec = real.precision().max(imag.precision());
        // Extract the underlying rug::Float values
        let real_f64 = real.to_f64();
        let imag_f64 = imag.to_f64();
        ComplexMPFR {
            value: RugComplex::with_val(prec, (real_f64, imag_f64)),
        }
    }

    /// Create a new ComplexMPFR with specified precision from integers
    pub fn with_val_integers(prec: u32, real: &Integer, imag: &Integer) -> Self {
        let real_f64 = if let Some(i) = real.to_i64() {
            i as f64
        } else {
            real.to_string().parse::<f64>().unwrap_or(0.0)
        };
        let imag_f64 = if let Some(i) = imag.to_i64() {
            i as f64
        } else {
            imag.to_string().parse::<f64>().unwrap_or(0.0)
        };
        ComplexMPFR {
            value: RugComplex::with_val(prec, (real_f64, imag_f64)),
        }
    }

    /// Create a new ComplexMPFR with specified precision from rationals
    pub fn with_val_rationals(prec: u32, real: &Rational, imag: &Rational) -> Self {
        let real_mpfr = RealMPFR::with_val_rational(prec, real);
        let imag_mpfr = RealMPFR::with_val_rational(prec, imag);
        Self::with_val_reals(real_mpfr, imag_mpfr)
    }

    /// Get the precision of this number in bits
    pub fn precision(&self) -> u32 {
        self.value.prec()
    }

    /// Get the real part as f64 (may lose precision)
    pub fn real(&self) -> f64 {
        self.value.real().to_f64()
    }

    /// Get the imaginary part as f64 (may lose precision)
    pub fn imag(&self) -> f64 {
        self.value.imag().to_f64()
    }

    /// Get the real part as RealMPFR
    pub fn real_part(&self) -> RealMPFR {
        RealMPFR::from_float(self.value.real().clone())
    }

    /// Get the imaginary part as RealMPFR
    pub fn imag_part(&self) -> RealMPFR {
        RealMPFR::from_float(self.value.imag().clone())
    }

    /// Compute the modulus (absolute value) |z| = √(a² + b²)
    pub fn abs(&self) -> RealMPFR {
        RealMPFR::from_float(self.value.clone().abs())
    }

    /// Compute the squared modulus |z|² = a² + b²
    pub fn norm(&self) -> RealMPFR {
        RealMPFR::from_float(self.value.clone().norm())
    }

    /// Compute the argument (phase angle) in radians
    ///
    /// Returns angle θ where z = r·e^(iθ), θ ∈ (-π, π]
    pub fn arg(&self) -> RealMPFR {
        RealMPFR::from_float(self.value.clone().arg())
    }

    /// Compute the complex conjugate z̄ = a - bi
    pub fn conj(&self) -> Self {
        ComplexMPFR {
            value: self.value.clone().conj(),
        }
    }

    /// Compute the complex conjugate (alias for conj)
    pub fn conjugate(&self) -> Self {
        self.conj()
    }

    /// Compute the reciprocal 1/z
    pub fn recip(&self) -> Result<Self> {
        if self.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        Ok(ComplexMPFR {
            value: self.value.clone().recip(),
        })
    }

    /// Compute the reciprocal (alias for recip)
    pub fn reciprocal(&self) -> Result<Self> {
        self.recip()
    }

    /// Compute the exponential e^z
    ///
    /// e^(a+bi) = e^a · (cos(b) + i·sin(b))
    pub fn exp(&self) -> Self {
        ComplexMPFR {
            value: self.value.clone().exp(),
        }
    }

    /// Compute the natural logarithm ln(z)
    ///
    /// ln(z) = ln|z| + i·arg(z)
    pub fn ln(&self) -> Self {
        ComplexMPFR {
            value: self.value.clone().ln(),
        }
    }

    /// Compute the natural logarithm (alias for ln)
    pub fn log(&self) -> Self {
        self.ln()
    }

    /// Compute z raised to power w: z^w
    ///
    /// z^w = e^(w·ln(z))
    pub fn pow(&self, w: &Self) -> Self {
        ComplexMPFR {
            value: self.value.clone().pow(&w.value),
        }
    }

    /// Compute integer power
    pub fn powi(&self, exp: i32) -> Self {
        ComplexMPFR {
            value: self.value.clone().pow(exp),
        }
    }

    /// Compute the square root
    ///
    /// Returns the principal square root
    pub fn sqrt(&self) -> Self {
        ComplexMPFR {
            value: self.value.clone().sqrt(),
        }
    }

    /// Compute sine: sin(z)
    pub fn sin(&self) -> Self {
        ComplexMPFR {
            value: self.value.clone().sin(),
        }
    }

    /// Compute cosine: cos(z)
    pub fn cos(&self) -> Self {
        ComplexMPFR {
            value: self.value.clone().cos(),
        }
    }

    /// Compute tangent: tan(z)
    pub fn tan(&self) -> Self {
        ComplexMPFR {
            value: self.value.clone().tan(),
        }
    }

    /// Compute arcsine: asin(z)
    pub fn asin(&self) -> Self {
        ComplexMPFR {
            value: self.value.clone().asin(),
        }
    }

    /// Compute arccosine: acos(z)
    pub fn acos(&self) -> Self {
        ComplexMPFR {
            value: self.value.clone().acos(),
        }
    }

    /// Compute arctangent: atan(z)
    pub fn atan(&self) -> Self {
        ComplexMPFR {
            value: self.value.clone().atan(),
        }
    }

    /// Compute hyperbolic sine: sinh(z)
    pub fn sinh(&self) -> Self {
        ComplexMPFR {
            value: self.value.clone().sinh(),
        }
    }

    /// Compute hyperbolic cosine: cosh(z)
    pub fn cosh(&self) -> Self {
        ComplexMPFR {
            value: self.value.clone().cosh(),
        }
    }

    /// Compute hyperbolic tangent: tanh(z)
    pub fn tanh(&self) -> Self {
        ComplexMPFR {
            value: self.value.clone().tanh(),
        }
    }

    /// Compute inverse hyperbolic sine: asinh(z)
    pub fn asinh(&self) -> Self {
        ComplexMPFR {
            value: self.value.clone().asinh(),
        }
    }

    /// Compute inverse hyperbolic cosine: acosh(z)
    pub fn acosh(&self) -> Self {
        ComplexMPFR {
            value: self.value.clone().acosh(),
        }
    }

    /// Compute inverse hyperbolic tangent: atanh(z)
    pub fn atanh(&self) -> Self {
        ComplexMPFR {
            value: self.value.clone().atanh(),
        }
    }

    /// Create zero with specified precision
    pub fn zero_with_prec(prec: u32) -> Self {
        ComplexMPFR {
            value: RugComplex::with_val(prec, (0, 0)),
        }
    }

    /// Create one with specified precision
    pub fn one_with_prec(prec: u32) -> Self {
        ComplexMPFR {
            value: RugComplex::with_val(prec, (1, 0)),
        }
    }

    /// Create the imaginary unit i with specified precision
    pub fn i_with_prec(prec: u32) -> Self {
        ComplexMPFR {
            value: RugComplex::with_val(prec, (0, 1)),
        }
    }

    /// Check if the value is NaN
    pub fn is_nan(&self) -> bool {
        self.value.real().is_nan() || self.value.imag().is_nan()
    }

    /// Check if the value is infinite
    pub fn is_infinite(&self) -> bool {
        self.value.real().is_infinite() || self.value.imag().is_infinite()
    }

    /// Check if the value is finite
    pub fn is_finite(&self) -> bool {
        self.value.real().is_finite() && self.value.imag().is_finite()
    }
}

impl fmt::Display for ComplexMPFR {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let r = self.real();
        let i = self.imag();

        if i >= 0.0 {
            write!(f, "{} + {}i", r, i)
        } else {
            write!(f, "{} - {}i", r, -i)
        }
    }
}

impl fmt::Debug for ComplexMPFR {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ComplexMPFR({}, {} bits)", self.value, self.precision())
    }
}

impl From<(f64, f64)> for ComplexMPFR {
    fn from(value: (f64, f64)) -> Self {
        ComplexMPFR {
            value: RugComplex::with_val(DEFAULT_PRECISION, value),
        }
    }
}

impl From<f64> for ComplexMPFR {
    fn from(value: f64) -> Self {
        ComplexMPFR {
            value: RugComplex::with_val(DEFAULT_PRECISION, (value, 0.0)),
        }
    }
}

impl From<(i32, i32)> for ComplexMPFR {
    fn from(value: (i32, i32)) -> Self {
        ComplexMPFR {
            value: RugComplex::with_val(DEFAULT_PRECISION, value),
        }
    }
}

impl From<i32> for ComplexMPFR {
    fn from(value: i32) -> Self {
        ComplexMPFR {
            value: RugComplex::with_val(DEFAULT_PRECISION, (value, 0)),
        }
    }
}

impl Add for ComplexMPFR {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let prec = self.precision().max(other.precision());
        ComplexMPFR {
            value: RugComplex::with_val(prec, self.value + other.value),
        }
    }
}

impl<'a, 'b> Add<&'b ComplexMPFR> for &'a ComplexMPFR {
    type Output = ComplexMPFR;

    fn add(self, other: &'b ComplexMPFR) -> ComplexMPFR {
        let prec = self.precision().max(other.precision());
        ComplexMPFR {
            value: RugComplex::with_val(prec, &self.value + &other.value),
        }
    }
}

impl Sub for ComplexMPFR {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let prec = self.precision().max(other.precision());
        ComplexMPFR {
            value: RugComplex::with_val(prec, self.value - other.value),
        }
    }
}

impl<'a, 'b> Sub<&'b ComplexMPFR> for &'a ComplexMPFR {
    type Output = ComplexMPFR;

    fn sub(self, other: &'b ComplexMPFR) -> ComplexMPFR {
        let prec = self.precision().max(other.precision());
        ComplexMPFR {
            value: RugComplex::with_val(prec, &self.value - &other.value),
        }
    }
}

impl Mul for ComplexMPFR {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let prec = self.precision().max(other.precision());
        ComplexMPFR {
            value: RugComplex::with_val(prec, self.value * other.value),
        }
    }
}

impl<'a, 'b> Mul<&'b ComplexMPFR> for &'a ComplexMPFR {
    type Output = ComplexMPFR;

    fn mul(self, other: &'b ComplexMPFR) -> ComplexMPFR {
        let prec = self.precision().max(other.precision());
        ComplexMPFR {
            value: RugComplex::with_val(prec, &self.value * &other.value),
        }
    }
}

impl Div for ComplexMPFR {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let prec = self.precision().max(other.precision());
        ComplexMPFR {
            value: RugComplex::with_val(prec, self.value / other.value),
        }
    }
}

impl<'a, 'b> Div<&'b ComplexMPFR> for &'a ComplexMPFR {
    type Output = ComplexMPFR;

    fn div(self, other: &'b ComplexMPFR) -> ComplexMPFR {
        let prec = self.precision().max(other.precision());
        ComplexMPFR {
            value: RugComplex::with_val(prec, &self.value / &other.value),
        }
    }
}

impl Neg for ComplexMPFR {
    type Output = Self;

    fn neg(self) -> Self {
        ComplexMPFR {
            value: -self.value,
        }
    }
}

impl<'a> Neg for &'a ComplexMPFR {
    type Output = ComplexMPFR;

    fn neg(self) -> ComplexMPFR {
        ComplexMPFR {
            value: -self.value.clone(),
        }
    }
}

impl PartialEq for ComplexMPFR {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Ring for ComplexMPFR {
    fn zero() -> Self {
        ComplexMPFR {
            value: RugComplex::with_val(DEFAULT_PRECISION, (0, 0)),
        }
    }

    fn one() -> Self {
        ComplexMPFR {
            value: RugComplex::with_val(DEFAULT_PRECISION, (1, 0)),
        }
    }

    fn is_zero(&self) -> bool {
        self.value.real() == 0 && self.value.imag() == 0
    }

    fn is_one(&self) -> bool {
        self.value.real() == 1 && self.value.imag() == 0
    }
}

impl CommutativeRing for ComplexMPFR {}

impl Field for ComplexMPFR {
    fn inverse(&self) -> Result<Self> {
        self.recip()
    }
}

impl NumericConversion for ComplexMPFR {
    fn from_i64(n: i64) -> Self {
        ComplexMPFR {
            value: RugComplex::with_val(DEFAULT_PRECISION, (n, 0)),
        }
    }

    fn from_u64(n: u64) -> Self {
        ComplexMPFR {
            value: RugComplex::with_val(DEFAULT_PRECISION, (n, 0)),
        }
    }

    fn to_i64(&self) -> Option<i64> {
        // Only convert if imaginary part is zero
        if self.value.imag() != 0 {
            return None;
        }
        let real_f64 = self.value.real().to_f64();
        if real_f64.is_finite() && real_f64 >= i64::MIN as f64 && real_f64 <= i64::MAX as f64 {
            Some(real_f64 as i64)
        } else {
            None
        }
    }

    fn to_u64(&self) -> Option<u64> {
        // Only convert if imaginary part is zero
        if self.value.imag() != 0 {
            return None;
        }
        let real_f64 = self.value.real().to_f64();
        if real_f64.is_finite() && real_f64 >= 0.0 && real_f64 <= u64::MAX as f64 {
            Some(real_f64 as u64)
        } else {
            None
        }
    }

    fn to_usize(&self) -> Option<usize> {
        // Only convert if imaginary part is zero
        if self.value.imag() != 0 {
            return None;
        }
        let real_f64 = self.value.real().to_f64();
        if real_f64.is_finite() && real_f64 >= 0.0 && real_f64 <= usize::MAX as f64 {
            Some(real_f64 as usize)
        } else {
            None
        }
    }

    fn to_f64(&self) -> Option<f64> {
        // Only convert if imaginary part is zero
        if self.value.imag() != 0 {
            return None;
        }
        Some(self.value.real().to_f64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_creation() {
        let z = ComplexMPFR::from((3.0, 4.0));
        assert_eq!(z.precision(), DEFAULT_PRECISION);
        assert!((z.real() - 3.0).abs() < 1e-10);
        assert!((z.imag() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_high_precision_creation() {
        let z = ComplexMPFR::with_val(256, (3.0, 4.0));
        assert_eq!(z.precision(), 256);
    }

    #[test]
    fn test_arithmetic() {
        let z1 = ComplexMPFR::from((3.0, 4.0));
        let z2 = ComplexMPFR::from((1.0, 2.0));

        // Addition: (3+4i) + (1+2i) = 4+6i
        let sum = z1.clone() + z2.clone();
        assert!((sum.real() - 4.0).abs() < 1e-10);
        assert!((sum.imag() - 6.0).abs() < 1e-10);

        // Subtraction: (3+4i) - (1+2i) = 2+2i
        let diff = z1.clone() - z2.clone();
        assert!((diff.real() - 2.0).abs() < 1e-10);
        assert!((diff.imag() - 2.0).abs() < 1e-10);

        // Multiplication: (3+4i)(1+2i) = 3+6i+4i+8i² = 3+10i-8 = -5+10i
        let prod = z1.clone() * z2.clone();
        assert!((prod.real() - (-5.0)).abs() < 1e-10);
        assert!((prod.imag() - 10.0).abs() < 1e-10);

        // Division: (3+4i)/(1+2i)
        let quot = z1.clone() / z2.clone();
        // (3+4i)/(1+2i) = (3+4i)(1-2i)/(1+4) = (3-6i+4i-8i²)/5 = (3-2i+8)/5 = (11-2i)/5 = 2.2-0.4i
        assert!((quot.real() - 2.2).abs() < 1e-10);
        assert!((quot.imag() - (-0.4)).abs() < 1e-10);
    }

    #[test]
    fn test_abs_arg() {
        let z = ComplexMPFR::from((3.0, 4.0));
        let abs_z = z.abs();
        assert!((abs_z.to_f64() - 5.0).abs() < 1e-10);

        let i = ComplexMPFR::from((0.0, 1.0));
        let arg_i = i.arg();
        assert!((arg_i.to_f64() - std::f64::consts::PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_conjugate() {
        let z = ComplexMPFR::from((3.0, 4.0));
        let conj = z.conjugate();

        assert!((conj.real() - 3.0).abs() < 1e-10);
        assert!((conj.imag() - (-4.0)).abs() < 1e-10);
    }

    #[test]
    fn test_exp_ln() {
        let z = ComplexMPFR::from((1.0, 1.0));
        let exp_z = z.exp();
        let ln_exp_z = exp_z.ln();

        // ln(e^z) = z (up to branch cut considerations)
        assert!((ln_exp_z.real() - z.real()).abs() < 1e-10);
        assert!((ln_exp_z.imag() - z.imag()).abs() < 1e-10);
    }

    #[test]
    fn test_sqrt() {
        let z = ComplexMPFR::from((0.0, 4.0)); // 4i
        let sqrt_z = z.sqrt();

        // √(4i) ≈ 1.414 + 1.414i (√2 + √2·i)
        let expected = 2.0_f64.sqrt();
        assert!((sqrt_z.real() - expected).abs() < 1e-10);
        assert!((sqrt_z.imag() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_trig() {
        let z = ComplexMPFR::from((0.0, 0.0));

        let sin_z = z.sin();
        let cos_z = z.cos();

        // sin(0) = 0, cos(0) = 1
        assert!(sin_z.abs().to_f64() < 1e-10);
        assert!((cos_z.real() - 1.0).abs() < 1e-10);
        assert!(cos_z.imag().abs() < 1e-10);
    }

    #[test]
    fn test_ring_traits() {
        let zero = ComplexMPFR::zero();
        let one = ComplexMPFR::one();

        assert!(zero.is_zero());
        assert!(!zero.is_one());
        assert!(!one.is_zero());
        assert!(one.is_one());

        let z = ComplexMPFR::from((5.0, 3.0));
        let z_plus_zero = z.clone() + zero.clone();
        assert_eq!(z, z_plus_zero);

        let z_times_one = z.clone() * one.clone();
        assert_eq!(z, z_times_one);
    }

    #[test]
    fn test_field_traits() {
        let z = ComplexMPFR::from((3.0, 4.0));
        let inv = z.inverse().unwrap();

        // 1/(3+4i) = (3-4i)/(9+16) = (3-4i)/25 = 0.12 - 0.16i
        assert!((inv.real() - 0.12).abs() < 1e-10);
        assert!((inv.imag() - (-0.16)).abs() < 1e-10);

        let zero = ComplexMPFR::zero();
        assert!(zero.inverse().is_err());
    }

    #[test]
    fn test_pow() {
        let z = ComplexMPFR::from((2.0, 0.0));
        let w = ComplexMPFR::from((3.0, 0.0));
        let result = z.pow(&w);

        // 2^3 = 8
        assert!((result.real() - 8.0).abs() < 1e-10);
        assert!(result.imag().abs() < 1e-10);
    }

    #[test]
    fn test_powi() {
        let z = ComplexMPFR::from((1.0, 1.0));
        let z_squared = z.powi(2);

        // (1+i)² = 1+2i+i² = 1+2i-1 = 2i
        assert!(z_squared.real().abs() < 1e-10);
        assert!((z_squared.imag() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_hyperbolic_functions() {
        let z = ComplexMPFR::from((1.0, 0.0));
        let sinh_z = z.sinh();
        let cosh_z = z.cosh();
        let tanh_z = z.tanh();

        // For real z=1, sinh(1) ≈ 1.1752011936438014
        assert!((sinh_z.real() - 1.1752011936438014).abs() < 1e-10);
        assert!(sinh_z.imag().abs() < 1e-10);

        // cosh(1) ≈ 1.5430806348152437
        assert!((cosh_z.real() - 1.5430806348152437).abs() < 1e-10);
        assert!(cosh_z.imag().abs() < 1e-10);

        // tanh(x) = sinh(x) / cosh(x)
        let tanh_computed = sinh_z.clone() / cosh_z.clone();
        assert!((tanh_z.real() - tanh_computed.real()).abs() < 1e-10);
        assert!((tanh_z.imag() - tanh_computed.imag()).abs() < 1e-10);
    }

    #[test]
    fn test_high_precision() {
        // Test with 1000 bits of precision
        let z = ComplexMPFR::with_val(1000, (3.0, 4.0));
        assert_eq!(z.precision(), 1000);

        let w = ComplexMPFR::with_val(1000, (1.0, 2.0));
        let product = z * w;

        // (3+4i)(1+2i) = -5+10i
        assert!((product.real() - (-5.0)).abs() < 1e-10);
        assert!((product.imag() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_numeric_conversion() {
        let z = ComplexMPFR::from_i64(42);
        assert_eq!(z.to_i64(), Some(42));
        assert!(z.imag().abs() < 1e-10);

        let w = ComplexMPFR::from((3.0, 4.0));
        // Cannot convert to i64 if imaginary part is non-zero
        assert_eq!(w.to_i64(), None);
    }

    #[test]
    fn test_norm() {
        let z = ComplexMPFR::from((3.0, 4.0));
        let norm = z.norm();
        // |3+4i|² = 9 + 16 = 25
        assert!((norm.to_f64() - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_trig() {
        let z = ComplexMPFR::from((0.5, 0.0));
        let asin_z = z.asin();
        let sin_asin_z = asin_z.sin();

        // sin(asin(z)) = z
        assert!((sin_asin_z.real() - 0.5).abs() < 1e-10);
        assert!(sin_asin_z.imag().abs() < 1e-10);
    }

    #[test]
    fn test_inverse_hyperbolic() {
        let z = ComplexMPFR::from((2.0, 0.0));
        let asinh_z = z.asinh();
        let sinh_asinh_z = asinh_z.sinh();

        // sinh(asinh(z)) = z
        assert!((sinh_asinh_z.real() - 2.0).abs() < 1e-10);
        assert!(sinh_asinh_z.imag().abs() < 1e-10);
    }
}
