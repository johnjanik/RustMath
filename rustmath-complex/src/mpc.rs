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

    /// Create a new ComplexMPFR from RealMPFR values, at the larger of the
    /// two parts' precisions (lossless).
    pub fn with_val_reals(real: RealMPFR, imag: RealMPFR) -> Self {
        let prec = real.precision().max(imag.precision());
        // Use the underlying rug::Float values directly. (A previous version
        // round-tripped through f64, silently truncating both parts to 53
        // bits — precision-loss bug.)
        ComplexMPFR {
            value: RugComplex::with_val(prec, (real.as_float().clone(), imag.as_float().clone())),
        }
    }

    /// Create a new ComplexMPFR with specified precision from integers
    ///
    /// The conversion is lossless up to the target precision: each part is
    /// transferred to MPFR as a full-width `rug::Integer` and then rounded
    /// once (correctly) to `prec` bits. Integers of any size are handled;
    /// integers of at most `prec` bits are represented exactly. (A previous
    /// version round-tripped through `to_i64() as f64`, which panicked
    /// beyond i64 and silently rounded beyond 2^53 — lossy-integer bug.)
    pub fn with_val_integers(prec: u32, real: &Integer, imag: &Integer) -> Self {
        let real_f = rug::Float::with_val(prec, integer_to_rug(real));
        let imag_f = rug::Float::with_val(prec, integer_to_rug(imag));
        ComplexMPFR {
            value: RugComplex::with_val(prec, (real_f, imag_f)),
        }
    }

    /// Create a new ComplexMPFR with specified precision from rationals
    ///
    /// Each part is computed as an exact-numerator / exact-denominator MPFR
    /// division, so the result is the true value of the rational correctly
    /// rounded once to `prec` bits, for numerators/denominators of any size.
    /// (A previous version routed through `RealMPFR::with_val_rational`,
    /// which truncates >i64 parts and double-rounds — lossy-integer bug.)
    pub fn with_val_rationals(prec: u32, real: &Rational, imag: &Rational) -> Self {
        ComplexMPFR {
            value: RugComplex::with_val(
                prec,
                (rational_to_float(prec, real), rational_to_float(prec, imag)),
            ),
        }
    }

    /// Get the precision of this number in bits
    pub fn precision(&self) -> u32 {
        let (real_prec, _imag_prec) = self.value.prec();
        real_prec
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
        use rug::ops::Pow;
        use rug::Float;
        let prec = self.precision();
        let re = self.value.real().clone();
        let im = self.value.imag().clone();
        let re_sq = Float::with_val(prec, re.pow(2));
        let im_sq = Float::with_val(prec, im.pow(2));
        let sum = Float::with_val(prec, &re_sq + &im_sq);
        let result = Float::with_val(prec, sum.sqrt());
        RealMPFR::from_float(result)
    }

    /// Compute the squared modulus |z|² = a² + b²
    pub fn norm(&self) -> RealMPFR {
        use rug::ops::Pow;
        use rug::Float;
        let prec = self.precision();
        let re = self.value.real().clone();
        let im = self.value.imag().clone();
        let re_sq = Float::with_val(prec, re.pow(2));
        let im_sq = Float::with_val(prec, im.pow(2));
        let result = Float::with_val(prec, &re_sq + &im_sq);
        RealMPFR::from_float(result)
    }

    /// Compute the argument (phase angle) in radians
    ///
    /// Returns angle θ where z = r·e^(iθ), θ ∈ (-π, π]
    pub fn arg(&self) -> RealMPFR {
        use rug::Float;
        let prec = self.precision();
        let re = self.value.real().clone();
        let im = self.value.imag().clone();
        let result = Float::with_val(prec, im.atan2(&re));
        RealMPFR::from_float(result)
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

/// Lossless conversion from a rustmath [`Integer`] (num-bigint backed) to a
/// `rug::Integer`, transferring the little-endian magnitude bytes directly.
/// Exact for integers of any size — no i64/f64 round-trip.
fn integer_to_rug(n: &Integer) -> rug::Integer {
    let (_sign, bytes) = n.as_bigint().to_bytes_le();
    let mag = rug::Integer::from_digits(&bytes, rug::integer::Order::Lsf);
    if n.signum() < 0 {
        -mag
    } else {
        mag
    }
}

/// Convert a rustmath [`Integer`] to a `rug::Float` exactly: the precision is
/// chosen as the integer's significant bit count, so no rounding occurs.
fn integer_to_float_exact(n: &Integer) -> rug::Float {
    let r = integer_to_rug(n);
    let bits = r.significant_bits().max(2);
    rug::Float::with_val(bits, r)
}

/// Convert a rustmath [`Rational`] to a `rug::Float` with a single correct
/// rounding to `prec` bits: numerator and denominator are transferred exactly,
/// and MPFR division rounds the true quotient once to the target precision.
fn rational_to_float(prec: u32, q: &Rational) -> rug::Float {
    let num = integer_to_float_exact(q.numerator());
    let den = integer_to_float_exact(q.denominator());
    rug::Float::with_val(prec, &num / &den)
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
        *self.value.real() == 0 && *self.value.imag() == 0
    }

    fn is_one(&self) -> bool {
        *self.value.real() == 1 && *self.value.imag() == 0
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
        if *self.value.imag() != 0 {
            return None;
        }
        // Exact extraction via rug::Integer, truncating toward zero like the
        // old `as i64` cast did for fractions. (A previous version rounded
        // through f64, which is lossy for exact integers in (2^53, 2^63) and
        // saturated 2^63 to i64::MAX instead of returning None.)
        let (i, _) = self
            .value
            .real()
            .to_integer_round(rug::float::Round::Zero)?;
        i.to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        // Only convert if imaginary part is zero
        if *self.value.imag() != 0 {
            return None;
        }
        let re = self.value.real();
        // Preserve the old contract: any negative real (even one that would
        // truncate to 0) converts to None.
        if *re < 0 {
            return None;
        }
        // Exact extraction (see to_i64): no f64 round-trip.
        let (i, _) = re.to_integer_round(rug::float::Round::Zero)?;
        i.to_u64()
    }

    fn to_usize(&self) -> Option<usize> {
        // Same exact path as to_u64, narrowed to the platform width.
        self.to_u64().and_then(|n| usize::try_from(n).ok())
    }

    fn to_f64(&self) -> Option<f64> {
        // Only convert if imaginary part is zero
        if *self.value.imag() != 0 {
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
    fn test_numeric_conversion_exact_above_2p53() {
        // Regression (F1): to_i64/to_u64/to_usize used to round through f64,
        // corrupting exact integers in (2^53, 2^63).
        let zero = Integer::from(0);

        // 2^62 + 1 is not f64-representable (nearest f64 is 2^62).
        let n: i64 = (1i64 << 62) + 1;
        let z = ComplexMPFR::with_val_integers(128, &Integer::from(n), &zero);
        assert_eq!(NumericConversion::to_i64(&z), Some(n));

        // i64::MAX - 1: the old f64 path rounded to 2^63 and saturated to
        // i64::MAX — off by one.
        let m = i64::MAX - 1;
        let z = ComplexMPFR::with_val_integers(128, &Integer::from(m), &zero);
        assert_eq!(NumericConversion::to_i64(&z), Some(m));

        // 2^63 itself is out of i64 range and must be None (the old path
        // accepted it because `2^63 <= i64::MAX as f64` compares equal, then
        // saturated the cast).
        let p63 = Integer::from(2).pow(63);
        let z = ComplexMPFR::with_val_integers(128, &p63, &zero);
        assert_eq!(NumericConversion::to_i64(&z), None);
        // ... but it is a perfectly good u64/usize (on 64-bit targets).
        let u: u64 = 1u64 << 63;
        assert_eq!(NumericConversion::to_u64(&z), Some(u));
        assert_eq!(NumericConversion::to_usize(&z), usize::try_from(u).ok());

        // u64 near the top, odd (not f64-representable): 2^63 + 3.
        let u: u64 = (1u64 << 63) + 3;
        let z = ComplexMPFR::with_val_integers(128, &Integer::from(u), &zero);
        assert_eq!(NumericConversion::to_u64(&z), Some(u));
        assert_eq!(NumericConversion::to_usize(&z), usize::try_from(u).ok());

        // 2^64 is out of u64 range.
        let p64 = Integer::from(2).pow(64);
        let z = ComplexMPFR::with_val_integers(128, &p64, &zero);
        assert_eq!(NumericConversion::to_u64(&z), None);
        assert_eq!(NumericConversion::to_usize(&z), None);

        // Negative counterpart is exact too.
        let n: i64 = -((1i64 << 61) + 5);
        let z = ComplexMPFR::with_val_integers(128, &Integer::from(n), &zero);
        assert_eq!(NumericConversion::to_i64(&z), Some(n));
        assert_eq!(NumericConversion::to_u64(&z), None);
    }

    #[test]
    fn test_numeric_conversion_truncation_contract_preserved() {
        // Fractions still truncate toward zero (old `as i64` semantics) ...
        let z = ComplexMPFR::with_val(64, (3.75, 0.0));
        assert_eq!(NumericConversion::to_i64(&z), Some(3));
        assert_eq!(NumericConversion::to_u64(&z), Some(3));
        let z = ComplexMPFR::with_val(64, (-3.75, 0.0));
        assert_eq!(NumericConversion::to_i64(&z), Some(-3));
        // ... and negatives (even ones truncating to 0) stay None for u64.
        assert_eq!(NumericConversion::to_u64(&z), None);
        let z = ComplexMPFR::with_val(64, (-0.5, 0.0));
        assert_eq!(NumericConversion::to_u64(&z), None);
        assert_eq!(NumericConversion::to_usize(&z), None);
        // Non-real and non-finite inputs stay None.
        let z = ComplexMPFR::with_val(64, (1.0, 2.0));
        assert_eq!(NumericConversion::to_i64(&z), None);
        let z = ComplexMPFR::with_val(64, (f64::INFINITY, 0.0));
        assert_eq!(NumericConversion::to_i64(&z), None);
        assert_eq!(NumericConversion::to_u64(&z), None);
        let z = ComplexMPFR::with_val(64, (f64::NAN, 0.0));
        assert_eq!(NumericConversion::to_i64(&z), None);
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

    /// Independent conversion path for cross-checking `integer_to_rug`:
    /// decimal-string transfer instead of magnitude-byte transfer.
    fn rug_int_via_string(n: &Integer) -> rug::Integer {
        rug::Integer::from_str_radix(&n.to_string(), 10).unwrap()
    }

    #[test]
    fn test_with_val_integers_beyond_i64_bit_exact() {
        // 2^100 + 3 (101 bits) and -(2^90 + 7): both far beyond i64/f64-exact
        // range. The old to_i64()-based path panicked on these.
        let a = Integer::from(2).pow(100) + Integer::from(3);
        let b = -(Integer::from(2).pow(90) + Integer::from(7));

        let z = ComplexMPFR::with_val_integers(256, &a, &b);
        assert_eq!(z.precision(), 256);

        // Both parts must be exactly the source integers (101/91 bits fit in
        // a 256-bit mantissa with no rounding). Round-trip via an independent
        // decimal-string conversion and demand bit-exact integer equality.
        assert!(z.value.real().is_integer());
        assert!(z.value.imag().is_integer());
        assert_eq!(
            z.value.real().to_integer().unwrap(),
            rug_int_via_string(&a)
        );
        assert_eq!(
            z.value.imag().to_integer().unwrap(),
            rug_int_via_string(&b)
        );
    }

    #[test]
    fn test_with_val_integers_200_bit_round_trip() {
        // A 201-bit integer at 256-bit precision: exact round trip required.
        let a = Integer::from(2).pow(200) + Integer::from(987654321i64);
        let z = ComplexMPFR::with_val_integers(256, &a, &Integer::zero());
        assert_eq!(
            z.value.real().to_integer().unwrap(),
            rug_int_via_string(&a)
        );
        assert!(z.value.imag().is_zero());
    }

    #[test]
    fn test_with_val_integers_small_values_unchanged() {
        // Sanity: small values keep working as before.
        let z = ComplexMPFR::with_val_integers(64, &Integer::from(42), &Integer::from(-17));
        assert_eq!(z.value.real().to_integer().unwrap(), 42);
        assert_eq!(z.value.imag().to_integer().unwrap(), -17);
    }

    #[test]
    fn test_with_val_rationals_dyadic_beyond_i64_bit_exact() {
        // (2^100 + 1) / 2^12 is exactly representable in a 200-bit mantissa
        // (101 significant bits). Demand a bit-exact round trip: multiplying
        // back by 2^12 must recover the >64-bit numerator exactly.
        let num = Integer::from(2).pow(100) + Integer::from(1);
        let re = Rational::new(num.clone(), Integer::from(2).pow(12)).unwrap();
        let im = Rational::new(Integer::zero(), Integer::one()).unwrap();

        let z = ComplexMPFR::with_val_rationals(200, &re, &im);
        assert_eq!(z.precision(), 200);

        let recovered = (z.value.real().clone() << 12u32).to_integer().unwrap();
        assert_eq!(recovered, rug_int_via_string(&num));
        assert!(z.value.imag().is_zero());
    }

    #[test]
    fn test_with_val_rationals_non_dyadic_correctly_rounded() {
        // Non-dyadic rationals with >64-bit numerators and denominators:
        // result must equal the true quotient correctly rounded to 256 bits.
        // Expected values are computed through an independent conversion path
        // (decimal strings -> rug::Integer -> exact Floats -> one division).
        let re_num = Integer::from(2).pow(100) + Integer::from(1);
        let re_den = Integer::from(3);
        let im_num = -(Integer::from(2).pow(80) + Integer::from(9));
        let im_den = Integer::from(2).pow(70) + Integer::from(1);

        let re = Rational::new(re_num.clone(), re_den.clone()).unwrap();
        let im = Rational::new(im_num.clone(), im_den.clone()).unwrap();
        let z = ComplexMPFR::with_val_rationals(256, &re, &im);

        let expect = |n: &Integer, d: &Integer| -> rug::Float {
            let n = rug::Float::with_val(512, rug_int_via_string(n));
            let d = rug::Float::with_val(512, rug_int_via_string(d));
            rug::Float::with_val(256, &n / &d)
        };
        // Bit-exact equality with the correctly rounded true value.
        assert_eq!(*z.value.real(), expect(&re_num, &re_den));
        assert_eq!(*z.value.imag(), expect(&im_num, &im_den));
        assert!(z.value.imag().is_sign_negative());
    }
}
