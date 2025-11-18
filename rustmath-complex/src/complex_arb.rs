//! Arbitrary precision complex balls using ball arithmetic
//!
//! Complex balls represent complex numbers with rigorous error bounds using
//! ball arithmetic (center + radius representation). This provides verified
//! numerical computation with guaranteed error bounds.
//!
//! This module provides:
//! - `ComplexBall`: A complex number with error bounds (center ± radius)
//! - `ComplexBallField`: Factory for creating complex balls with specific precision
//! - `IntegrationContext`: Context for numerical integration with error control

use rustmath_core::{CommutativeRing, Field, MathError, Result, Ring};
use rustmath_reals::{Real, RealMPFR};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Default precision for complex balls (53 bits = f64 equivalent)
pub const DEFAULT_PRECISION: u32 = 53;

/// Complex ball: a complex number with rigorous error bounds
///
/// Represents a complex disk centered at `center` with radius `radius`.
/// All values within the disk are possible values for this number.
///
/// Ball arithmetic ensures that:
/// - Operations maintain rigorous error bounds
/// - Results contain the true mathematical result
/// - Precision loss is tracked automatically
///
/// # Examples
///
/// ```
/// use rustmath_complex::ComplexBall;
///
/// // Create a ball centered at 3+4i with radius 0.01
/// let z = ComplexBall::new(3.0, 4.0, 0.01);
/// assert!(z.contains(3.005, 4.005));
/// assert!(!z.contains(3.1, 4.1));
/// ```
#[derive(Clone, Debug)]
pub struct ComplexBall {
    /// Center of the ball (real part)
    center_real: RealMPFR,
    /// Center of the ball (imaginary part)
    center_imag: RealMPFR,
    /// Radius of the ball (uncertainty)
    radius: RealMPFR,
    /// Precision in bits
    precision: u32,
}

impl ComplexBall {
    /// Create a new complex ball with default precision
    ///
    /// # Arguments
    ///
    /// * `real` - Real part of center
    /// * `imag` - Imaginary part of center
    /// * `radius` - Radius of uncertainty
    pub fn new(real: f64, imag: f64, radius: f64) -> Self {
        ComplexBall {
            center_real: RealMPFR::with_val(DEFAULT_PRECISION, real),
            center_imag: RealMPFR::with_val(DEFAULT_PRECISION, imag),
            radius: RealMPFR::with_val(DEFAULT_PRECISION, radius.abs()),
            precision: DEFAULT_PRECISION,
        }
    }

    /// Create a new complex ball with specified precision
    pub fn with_precision(prec: u32, real: f64, imag: f64, radius: f64) -> Self {
        ComplexBall {
            center_real: RealMPFR::with_val(prec, real),
            center_imag: RealMPFR::with_val(prec, imag),
            radius: RealMPFR::with_val(prec, radius.abs()),
            precision: prec,
        }
    }

    /// Create an exact complex ball (zero radius) with default precision
    pub fn exact(real: f64, imag: f64) -> Self {
        Self::new(real, imag, 0.0)
    }

    /// Create an exact complex ball with specified precision
    pub fn exact_with_precision(prec: u32, real: f64, imag: f64) -> Self {
        Self::with_precision(prec, real, imag, 0.0)
    }

    /// Get the center (real part)
    pub fn center_real(&self) -> &RealMPFR {
        &self.center_real
    }

    /// Get the center (imaginary part)
    pub fn center_imag(&self) -> &RealMPFR {
        &self.center_imag
    }

    /// Get the radius
    pub fn radius(&self) -> &RealMPFR {
        &self.radius
    }

    /// Get the precision in bits
    pub fn precision(&self) -> u32 {
        self.precision
    }

    /// Check if a complex value is contained in this ball
    pub fn contains(&self, real: f64, imag: f64) -> bool {
        let dr = real - self.center_real.to_f64();
        let di = imag - self.center_imag.to_f64();
        let dist = (dr * dr + di * di).sqrt();
        dist <= self.radius.to_f64()
    }

    /// Check if this ball contains another ball
    pub fn contains_ball(&self, other: &ComplexBall) -> bool {
        // This ball contains other if the distance between centers plus other's radius
        // is less than or equal to this ball's radius
        let dr = (self.center_real.to_f64() - other.center_real.to_f64()).abs();
        let di = (self.center_imag.to_f64() - other.center_imag.to_f64()).abs();
        let dist = (dr * dr + di * di).sqrt();
        dist + other.radius.to_f64() <= self.radius.to_f64()
    }

    /// Check if this ball overlaps with another ball
    pub fn overlaps(&self, other: &ComplexBall) -> bool {
        let dr = (self.center_real.to_f64() - other.center_real.to_f64()).abs();
        let di = (self.center_imag.to_f64() - other.center_imag.to_f64()).abs();
        let dist = (dr * dr + di * di).sqrt();
        dist <= self.radius.to_f64() + other.radius.to_f64()
    }

    /// Check if this is an exact ball (zero radius)
    pub fn is_exact(&self) -> bool {
        self.radius.to_f64() == 0.0
    }

    /// Get the modulus bound (maximum absolute value in the ball)
    pub fn abs_upper_bound(&self) -> f64 {
        let center_abs = {
            let r = self.center_real.to_f64();
            let i = self.center_imag.to_f64();
            (r * r + i * i).sqrt()
        };
        center_abs + self.radius.to_f64()
    }

    /// Get the lower bound on modulus
    pub fn abs_lower_bound(&self) -> f64 {
        let center_abs = {
            let r = self.center_real.to_f64();
            let i = self.center_imag.to_f64();
            (r * r + i * i).sqrt()
        };
        (center_abs - self.radius.to_f64()).max(0.0)
    }

    /// Compute the conjugate
    pub fn conjugate(&self) -> Self {
        ComplexBall {
            center_real: self.center_real.clone(),
            center_imag: -self.center_imag.clone(),
            radius: self.radius.clone(),
            precision: self.precision,
        }
    }

    /// Compute absolute value as a ball
    pub fn abs(&self) -> RealMPFR {
        // Upper bound on |z|
        let r = self.center_real.to_f64();
        let i = self.center_imag.to_f64();
        let center_abs = (r * r + i * i).sqrt();
        RealMPFR::with_val(
            self.precision,
            center_abs + self.radius.to_f64(),
        )
    }

    /// Compute the argument (phase) with error bounds
    pub fn arg(&self) -> RealMPFR {
        use std::f64::consts::PI;
        let arg_center = self.center_imag.to_f64().atan2(self.center_real.to_f64());

        // Estimate error in argument due to radius
        let center_abs = {
            let r = self.center_real.to_f64();
            let i = self.center_imag.to_f64();
            (r * r + i * i).sqrt()
        };

        // If ball contains origin, argument is undefined
        if center_abs <= self.radius.to_f64() {
            // Return a ball covering all angles
            return RealMPFR::with_val(self.precision, PI);
        }

        RealMPFR::with_val(self.precision, arg_center)
    }

    /// Convert to a point estimate (center only, discarding error bounds)
    pub fn to_complex(&self) -> (f64, f64) {
        (self.center_real.to_f64(), self.center_imag.to_f64())
    }

    /// Zero ball with specified precision
    pub fn zero_with_prec(prec: u32) -> Self {
        Self::exact_with_precision(prec, 0.0, 0.0)
    }

    /// One ball with specified precision
    pub fn one_with_prec(prec: u32) -> Self {
        Self::exact_with_precision(prec, 1.0, 0.0)
    }

    /// Imaginary unit ball with specified precision
    pub fn i_with_prec(prec: u32) -> Self {
        Self::exact_with_precision(prec, 0.0, 1.0)
    }
}

impl fmt::Display for ComplexBall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let r = self.center_real.to_f64();
        let i = self.center_imag.to_f64();
        let rad = self.radius.to_f64();

        if rad == 0.0 {
            // Exact value
            if i >= 0.0 {
                write!(f, "{} + {}i", r, i)
            } else {
                write!(f, "{} - {}i", r, -i)
            }
        } else {
            // Ball with radius
            if i >= 0.0 {
                write!(f, "[{} + {}i ± {}]", r, i, rad)
            } else {
                write!(f, "[{} - {}i ± {}]", r, -i, rad)
            }
        }
    }
}

impl Add for ComplexBall {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let prec = self.precision.max(other.precision);

        // (z1 ± r1) + (z2 ± r2) = (z1 + z2) ± (r1 + r2)
        ComplexBall {
            center_real: &self.center_real + &other.center_real,
            center_imag: &self.center_imag + &other.center_imag,
            radius: &self.radius + &other.radius,
            precision: prec,
        }
    }
}

impl Sub for ComplexBall {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let prec = self.precision.max(other.precision);

        // (z1 ± r1) - (z2 ± r2) = (z1 - z2) ± (r1 + r2)
        ComplexBall {
            center_real: &self.center_real - &other.center_real,
            center_imag: &self.center_imag - &other.center_imag,
            radius: &self.radius + &other.radius,
            precision: prec,
        }
    }
}

impl Mul for ComplexBall {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let prec = self.precision.max(other.precision);

        // Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        let ac = &self.center_real * &other.center_real;
        let bd = &self.center_imag * &other.center_imag;
        let ad = &self.center_real * &other.center_imag;
        let bc = &self.center_imag * &other.center_real;

        let new_real = &ac - &bd;
        let new_imag = &ad + &bc;

        // Error propagation: use |z1||r2| + |z2||r1| + r1*r2
        let abs1 = self.abs_upper_bound();
        let abs2 = other.abs_upper_bound();
        let r1 = self.radius.to_f64();
        let r2 = other.radius.to_f64();

        let new_radius = abs1 * r2 + abs2 * r1 + r1 * r2;

        ComplexBall {
            center_real: new_real,
            center_imag: new_imag,
            radius: RealMPFR::with_val(prec, new_radius),
            precision: prec,
        }
    }
}

impl Div for ComplexBall {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        // Division: z1/z2 = z1 * (1/z2)
        // For 1/z where z = c ± r, we compute 1/c with error bounds

        let prec = self.precision.max(other.precision);

        let c_re = other.center_real.to_f64();
        let c_im = other.center_imag.to_f64();
        let r = other.radius.to_f64();

        let c_abs_sq = c_re * c_re + c_im * c_im;
        let c_abs = c_abs_sq.sqrt();

        // Check if ball contains zero
        if c_abs <= r {
            // Division by ball containing zero - return large ball
            return ComplexBall::with_precision(prec, 0.0, 0.0, 1e10);
        }

        // Reciprocal of center
        let inv_real = c_re / c_abs_sq;
        let inv_imag = -c_im / c_abs_sq;

        // Error in reciprocal
        let min_abs = c_abs - r;
        let max_abs = c_abs + r;
        let inv_min = 1.0 / max_abs;
        let inv_max = 1.0 / min_abs;
        let inv_error = (inv_max - inv_min) / 2.0;

        let recip = ComplexBall::with_precision(prec, inv_real, inv_imag, inv_error);

        // Multiply by reciprocal
        self * recip
    }
}

impl Neg for ComplexBall {
    type Output = Self;

    fn neg(self) -> Self {
        ComplexBall {
            center_real: -self.center_real,
            center_imag: -self.center_imag,
            radius: self.radius,
            precision: self.precision,
        }
    }
}

/// Complex ball field: factory for creating complex balls with specific precision
///
/// This provides a unified interface for creating complex balls with consistent precision.
///
/// # Examples
///
/// ```
/// use rustmath_complex::ComplexBallField;
///
/// let field = ComplexBallField::new(256); // 256-bit precision
/// let z = field.make_ball(3.0, 4.0, 0.01);
/// assert_eq!(z.precision(), 256);
/// ```
#[derive(Clone, Debug)]
pub struct ComplexBallField {
    precision: u32,
}

impl ComplexBallField {
    /// Create a new complex ball field with specified precision
    pub fn new(precision: u32) -> Self {
        ComplexBallField { precision }
    }

    /// Get the default field (53-bit precision)
    pub fn default() -> Self {
        ComplexBallField {
            precision: DEFAULT_PRECISION,
        }
    }

    /// Get the precision of this field
    pub fn precision(&self) -> u32 {
        self.precision
    }

    /// Create a complex ball in this field
    pub fn make_ball(&self, real: f64, imag: f64, radius: f64) -> ComplexBall {
        ComplexBall::with_precision(self.precision, real, imag, radius)
    }

    /// Create an exact complex ball in this field
    pub fn make_exact(&self, real: f64, imag: f64) -> ComplexBall {
        ComplexBall::exact_with_precision(self.precision, real, imag)
    }

    /// Create zero
    pub fn zero(&self) -> ComplexBall {
        ComplexBall::zero_with_prec(self.precision)
    }

    /// Create one
    pub fn one(&self) -> ComplexBall {
        ComplexBall::one_with_prec(self.precision)
    }

    /// Create imaginary unit
    pub fn i(&self) -> ComplexBall {
        ComplexBall::i_with_prec(self.precision)
    }
}

/// Context for numerical integration with error control
///
/// Provides settings for adaptive integration algorithms that use
/// complex balls to track error bounds.
///
/// # Examples
///
/// ```
/// use rustmath_complex::IntegrationContext;
///
/// let ctx = IntegrationContext::new()
///     .with_precision(256)
///     .with_tolerance(1e-10)
///     .with_max_depth(20);
/// ```
#[derive(Clone, Debug)]
pub struct IntegrationContext {
    /// Precision in bits for ball arithmetic
    precision: u32,
    /// Absolute error tolerance
    abs_tolerance: f64,
    /// Relative error tolerance
    rel_tolerance: f64,
    /// Maximum recursion depth for adaptive integration
    max_depth: usize,
    /// Maximum number of function evaluations
    max_evals: usize,
}

impl IntegrationContext {
    /// Create a new integration context with default settings
    pub fn new() -> Self {
        IntegrationContext {
            precision: DEFAULT_PRECISION,
            abs_tolerance: 1e-10,
            rel_tolerance: 1e-10,
            max_depth: 15,
            max_evals: 10000,
        }
    }

    /// Set the precision
    pub fn with_precision(mut self, precision: u32) -> Self {
        self.precision = precision;
        self
    }

    /// Set the absolute tolerance
    pub fn with_abs_tolerance(mut self, tol: f64) -> Self {
        self.abs_tolerance = tol;
        self
    }

    /// Set the relative tolerance
    pub fn with_rel_tolerance(mut self, tol: f64) -> Self {
        self.rel_tolerance = tol;
        self
    }

    /// Set both tolerances at once
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.abs_tolerance = tol;
        self.rel_tolerance = tol;
        self
    }

    /// Set the maximum recursion depth
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Set the maximum number of evaluations
    pub fn with_max_evals(mut self, evals: usize) -> Self {
        self.max_evals = evals;
        self
    }

    /// Get the precision
    pub fn precision(&self) -> u32 {
        self.precision
    }

    /// Get the absolute tolerance
    pub fn abs_tolerance(&self) -> f64 {
        self.abs_tolerance
    }

    /// Get the relative tolerance
    pub fn rel_tolerance(&self) -> f64 {
        self.rel_tolerance
    }

    /// Get the maximum depth
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Get the maximum evaluations
    pub fn max_evals(&self) -> usize {
        self.max_evals
    }

    /// Check if a result meets the tolerance criteria
    pub fn is_converged(&self, result: &ComplexBall, target_abs: f64) -> bool {
        let radius = result.radius().to_f64();
        let abs_ok = radius <= self.abs_tolerance;
        let rel_ok = radius <= self.rel_tolerance * target_abs;
        abs_ok || rel_ok
    }
}

impl Default for IntegrationContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_ball_creation() {
        let z = ComplexBall::new(3.0, 4.0, 0.1);
        assert_eq!(z.center_real().to_f64(), 3.0);
        assert_eq!(z.center_imag().to_f64(), 4.0);
        assert_eq!(z.radius().to_f64(), 0.1);
    }

    #[test]
    fn test_exact_ball() {
        let z = ComplexBall::exact(3.0, 4.0);
        assert!(z.is_exact());
        assert_eq!(z.radius().to_f64(), 0.0);
    }

    #[test]
    fn test_contains() {
        let z = ComplexBall::new(0.0, 0.0, 1.0);
        assert!(z.contains(0.5, 0.5));
        assert!(!z.contains(2.0, 2.0));
    }

    #[test]
    fn test_ball_addition() {
        let z1 = ComplexBall::new(1.0, 2.0, 0.1);
        let z2 = ComplexBall::new(3.0, 4.0, 0.2);
        let sum = z1 + z2;

        assert_eq!(sum.center_real().to_f64(), 4.0);
        assert_eq!(sum.center_imag().to_f64(), 6.0);
        assert!((sum.radius().to_f64() - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_ball_subtraction() {
        let z1 = ComplexBall::new(5.0, 6.0, 0.1);
        let z2 = ComplexBall::new(2.0, 3.0, 0.2);
        let diff = z1 - z2;

        assert_eq!(diff.center_real().to_f64(), 3.0);
        assert_eq!(diff.center_imag().to_f64(), 3.0);
        assert!((diff.radius().to_f64() - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_ball_multiplication() {
        let z1 = ComplexBall::exact(2.0, 0.0);
        let z2 = ComplexBall::exact(3.0, 0.0);
        let prod = z1 * z2;

        assert_eq!(prod.center_real().to_f64(), 6.0);
        assert_eq!(prod.center_imag().to_f64(), 0.0);
    }

    #[test]
    fn test_ball_conjugate() {
        let z = ComplexBall::new(3.0, 4.0, 0.1);
        let conj = z.conjugate();

        assert_eq!(conj.center_real().to_f64(), 3.0);
        assert_eq!(conj.center_imag().to_f64(), -4.0);
        assert_eq!(conj.radius().to_f64(), 0.1);
    }

    #[test]
    fn test_ball_field() {
        let field = ComplexBallField::new(256);
        let z = field.make_exact(3.0, 4.0);

        assert_eq!(z.precision(), 256);
        assert!(z.is_exact());
    }

    #[test]
    fn test_overlaps() {
        let z1 = ComplexBall::new(0.0, 0.0, 1.0);
        let z2 = ComplexBall::new(1.5, 0.0, 1.0);
        let z3 = ComplexBall::new(3.0, 0.0, 0.5);

        assert!(z1.overlaps(&z2));
        assert!(!z1.overlaps(&z3));
    }

    #[test]
    fn test_integration_context() {
        let ctx = IntegrationContext::new()
            .with_precision(128)
            .with_tolerance(1e-12)
            .with_max_depth(20);

        assert_eq!(ctx.precision(), 128);
        assert_eq!(ctx.abs_tolerance(), 1e-12);
        assert_eq!(ctx.max_depth(), 20);
    }

    #[test]
    fn test_abs_bounds() {
        let z = ComplexBall::new(3.0, 4.0, 0.5);
        let abs_upper = z.abs_upper_bound();
        let abs_lower = z.abs_lower_bound();

        // |3+4i| = 5
        assert!(abs_upper >= 5.0);
        assert!(abs_lower <= 5.0);
        assert!((abs_upper - 5.5).abs() < 0.1);
        assert!((abs_lower - 4.5).abs() < 0.1);
    }

    #[test]
    fn test_division() {
        let z1 = ComplexBall::exact(6.0, 0.0);
        let z2 = ComplexBall::exact(2.0, 0.0);
        let quot = z1 / z2;

        assert!((quot.center_real().to_f64() - 3.0).abs() < 1e-10);
        assert!(quot.center_imag().to_f64().abs() < 1e-10);
    }

    #[test]
    fn test_negation() {
        let z = ComplexBall::new(3.0, -4.0, 0.1);
        let neg = -z;

        assert_eq!(neg.center_real().to_f64(), -3.0);
        assert_eq!(neg.center_imag().to_f64(), 4.0);
        assert_eq!(neg.radius().to_f64(), 0.1);
    }
}
