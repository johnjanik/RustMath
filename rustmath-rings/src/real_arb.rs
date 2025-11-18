//! # Real Ball Arithmetic (Arb)
//!
//! This module implements arbitrary-precision real ball arithmetic inspired by Arb.
//!
//! ## Overview
//!
//! Real ball arithmetic represents real numbers as intervals [m - r, m + r] where:
//! - `m` is the midpoint (an arbitrary-precision floating-point number)
//! - `r` is the radius (representing uncertainty/error)
//!
//! This provides:
//! - Rigorous error bounds for all operations
//! - Arbitrary precision arithmetic
//! - Efficient handling of numerical errors
//!
//! ## Theory
//!
//! Ball arithmetic is essential for:
//! - Validated numerics (proving inequalities)
//! - Rigorous special function evaluation
//! - Interval arithmetic with midpoint-radius representation
//! - Computer-assisted proofs
//!
//! The Arb library uses this representation for efficiency and numerical stability.
//!
//! ## Implementation Notes
//!
//! While SageMath uses the Arb C library, this implementation provides a pure Rust
//! version with similar semantics using arbitrary-precision floats.
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::real_arb::{RealBall, RealBallField};
//! use rustmath_rationals::Rational;
//!
//! let field = RealBallField::new(53); // 53-bit precision
//! let ball = field.create_ball(Rational::new(355, 113), Rational::new(1, 1000));
//! ```

use rustmath_core::{Field, Ring};
use rustmath_rationals::Rational;
use std::fmt;

/// Field of real balls with a specific precision
///
/// Represents the ring RBF_p of real balls with p bits of precision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealBallField {
    /// Precision in bits
    precision: usize,
}

impl RealBallField {
    /// Create a new real ball field with given precision
    ///
    /// # Arguments
    /// * `precision` - Precision in bits (typically 53 for double, 113 for quad, etc.)
    ///
    /// # Examples
    /// ```
    /// use rustmath_rings::real_arb::RealBallField;
    ///
    /// let field = RealBallField::new(53);
    /// ```
    pub fn new(precision: usize) -> Self {
        Self { precision }
    }

    /// Get the precision in bits
    pub fn precision(&self) -> usize {
        self.precision
    }

    /// Create a real ball from a midpoint and radius
    pub fn create_ball(&self, midpoint: Rational, radius: Rational) -> RealBall {
        RealBall::new(midpoint, radius, self.precision)
    }

    /// Create a real ball from a single value (radius = 0)
    pub fn from_rational(&self, value: Rational) -> RealBall {
        RealBall::new(value, Rational::zero(), self.precision)
    }

    /// Create a real ball from an integer
    pub fn from_int(&self, value: i64) -> RealBall {
        RealBall::new(Rational::from(value), Rational::zero(), self.precision)
    }

    /// Create zero ball
    pub fn zero(&self) -> RealBall {
        RealBall::zero(self.precision)
    }

    /// Create one ball
    pub fn one(&self) -> RealBall {
        RealBall::one(self.precision)
    }
}

impl fmt::Display for RealBallField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Real ball field with {} bits precision", self.precision)
    }
}

/// Element of a real ball field
///
/// Represents a real number as an interval [midpoint - radius, midpoint + radius].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealBall {
    /// Midpoint of the ball
    midpoint: Rational,
    /// Radius of the ball (non-negative)
    radius: Rational,
    /// Precision in bits
    precision: usize,
}

impl RealBall {
    /// Create a new real ball
    ///
    /// # Arguments
    /// * `midpoint` - Center of the interval
    /// * `radius` - Half-width of the interval (must be non-negative)
    /// * `precision` - Precision in bits
    ///
    /// # Examples
    /// ```
    /// use rustmath_rings::real_arb::RealBall;
    /// use rustmath_rationals::Rational;
    ///
    /// let ball = RealBall::new(
    ///     Rational::new(1, 2),
    ///     Rational::new(1, 100),
    ///     53
    /// );
    /// ```
    pub fn new(midpoint: Rational, radius: Rational, precision: usize) -> Self {
        if radius < Rational::zero() {
            panic!("Radius must be non-negative");
        }
        Self {
            midpoint,
            radius,
            precision,
        }
    }

    /// Create a zero ball
    pub fn zero(precision: usize) -> Self {
        Self::new(Rational::zero(), Rational::zero(), precision)
    }

    /// Create a one ball
    pub fn one(precision: usize) -> Self {
        Self::new(Rational::from(1), Rational::zero(), precision)
    }

    /// Get the midpoint
    pub fn midpoint(&self) -> &Rational {
        &self.midpoint
    }

    /// Get the radius
    pub fn radius(&self) -> &Rational {
        &self.radius
    }

    /// Get the precision
    pub fn precision(&self) -> usize {
        self.precision
    }

    /// Get the lower bound of the interval
    pub fn lower(&self) -> Rational {
        self.midpoint.clone() - self.radius.clone()
    }

    /// Get the upper bound of the interval
    pub fn upper(&self) -> Rational {
        self.midpoint.clone() + self.radius.clone()
    }

    /// Check if the ball contains zero
    pub fn contains_zero(&self) -> bool {
        self.radius >= self.midpoint.abs()
    }

    /// Check if the ball is exact (radius = 0)
    pub fn is_exact(&self) -> bool {
        self.radius == Rational::zero()
    }

    /// Get the accuracy (negative log2 of radius, roughly)
    /// Returns None if the ball is exact
    pub fn accuracy(&self) -> Option<i64> {
        if self.is_exact() {
            None
        } else {
            // Simplified: would need proper log2 implementation
            Some(-(self.radius.to_f64().log2() as i64))
        }
    }

    /// Add with error propagation
    pub fn add(&self, other: &Self) -> Self {
        let new_midpoint = self.midpoint.clone() + other.midpoint.clone();
        let new_radius = self.radius.clone() + other.radius.clone();
        Self::new(new_midpoint, new_radius, self.precision.min(other.precision))
    }

    /// Subtract with error propagation
    pub fn sub(&self, other: &Self) -> Self {
        let new_midpoint = self.midpoint.clone() - other.midpoint.clone();
        let new_radius = self.radius.clone() + other.radius.clone();
        Self::new(new_midpoint, new_radius, self.precision.min(other.precision))
    }

    /// Multiply with error propagation
    pub fn mul(&self, other: &Self) -> Self {
        let new_midpoint = self.midpoint.clone() * other.midpoint.clone();

        // Error propagation: |(a±da)(b±db)| ≤ |ab| + |a|db + |b|da + da*db
        let error1 = self.midpoint.abs() * other.radius.clone();
        let error2 = other.midpoint.abs() * self.radius.clone();
        let error3 = self.radius.clone() * other.radius.clone();

        let new_radius = error1 + error2 + error3;
        Self::new(new_midpoint, new_radius, self.precision.min(other.precision))
    }

    /// Negate the ball
    pub fn neg(&self) -> Self {
        Self::new(-self.midpoint.clone(), self.radius.clone(), self.precision)
    }

    /// Absolute value
    pub fn abs(&self) -> Self {
        if self.contains_zero() {
            // If contains zero, result is [0, max(|lower|, |upper|)]
            let max_abs = self.lower().abs().max(self.upper().abs());
            Self::new(max_abs.clone() / Rational::from(2), max_abs / Rational::from(2), self.precision)
        } else {
            Self::new(self.midpoint.abs(), self.radius.clone(), self.precision)
        }
    }

    /// Check if two balls overlap
    pub fn overlaps(&self, other: &Self) -> bool {
        let distance = (self.midpoint.clone() - other.midpoint.clone()).abs();
        distance <= self.radius.clone() + other.radius.clone()
    }

    /// Check if this ball is contained in another
    pub fn is_contained_in(&self, other: &Self) -> bool {
        let distance = (self.midpoint.clone() - other.midpoint.clone()).abs();
        distance + self.radius.clone() <= other.radius.clone()
    }
}

impl std::ops::Add for RealBall {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        RealBall::add(&self, &other)
    }
}

impl std::ops::Sub for RealBall {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        RealBall::sub(&self, &other)
    }
}

impl std::ops::Mul for RealBall {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        RealBall::mul(&self, &other)
    }
}

impl std::ops::Neg for RealBall {
    type Output = Self;

    fn neg(self) -> Self {
        RealBall::neg(&self)
    }
}

impl fmt::Display for RealBall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_exact() {
            write!(f, "[{}]", self.midpoint)
        } else {
            write!(f, "[{} +/- {}]", self.midpoint, self.radius)
        }
    }
}

/// Create a RealBall from serialized data (for unpickling)
pub fn create_real_ball(midpoint: Rational, radius: Rational, precision: usize) -> RealBall {
    RealBall::new(midpoint, radius, precision)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real_ball_field() {
        let field = RealBallField::new(53);
        assert_eq!(field.precision(), 53);

        let ball = field.from_int(42);
        assert_eq!(ball.midpoint(), &Rational::from(42));
        assert!(ball.is_exact());
    }

    #[test]
    fn test_new_real_ball() {
        let ball = RealBall::new(Rational::new(1, 2), Rational::new(1, 10), 53);

        assert_eq!(ball.midpoint(), &Rational::new(1, 2));
        assert_eq!(ball.radius(), &Rational::new(1, 10));
        assert_eq!(ball.precision(), 53);
    }

    #[test]
    #[should_panic(expected = "Radius must be non-negative")]
    fn test_negative_radius() {
        let _ = RealBall::new(Rational::zero(), Rational::new(-1, 2), 53);
    }

    #[test]
    fn test_bounds() {
        let ball = RealBall::new(Rational::from(5), Rational::from(2), 53);

        assert_eq!(ball.lower(), Rational::from(3));
        assert_eq!(ball.upper(), Rational::from(7));
    }

    #[test]
    fn test_contains_zero() {
        let ball1 = RealBall::new(Rational::from(1), Rational::from(2), 53);
        assert!(ball1.contains_zero());

        let ball2 = RealBall::new(Rational::from(5), Rational::from(2), 53);
        assert!(!ball2.contains_zero());
    }

    #[test]
    fn test_is_exact() {
        let exact = RealBall::new(Rational::from(1), Rational::zero(), 53);
        assert!(exact.is_exact());

        let inexact = RealBall::new(Rational::from(1), Rational::new(1, 100), 53);
        assert!(!inexact.is_exact());
    }

    #[test]
    fn test_addition() {
        let b1 = RealBall::new(Rational::from(1), Rational::new(1, 10), 53);
        let b2 = RealBall::new(Rational::from(2), Rational::new(1, 10), 53);

        let sum = b1.add(&b2);
        assert_eq!(sum.midpoint(), &Rational::from(3));
        assert_eq!(sum.radius(), &Rational::new(1, 5));
    }

    #[test]
    fn test_subtraction() {
        let b1 = RealBall::new(Rational::from(5), Rational::new(1, 10), 53);
        let b2 = RealBall::new(Rational::from(2), Rational::new(1, 10), 53);

        let diff = b1.sub(&b2);
        assert_eq!(diff.midpoint(), &Rational::from(3));
        assert_eq!(diff.radius(), &Rational::new(1, 5));
    }

    #[test]
    fn test_multiplication() {
        let b1 = RealBall::new(Rational::from(2), Rational::zero(), 53);
        let b2 = RealBall::new(Rational::from(3), Rational::zero(), 53);

        let product = b1.mul(&b2);
        assert_eq!(product.midpoint(), &Rational::from(6));
    }

    #[test]
    fn test_negation() {
        let ball = RealBall::new(Rational::from(5), Rational::from(1), 53);
        let neg = ball.neg();

        assert_eq!(neg.midpoint(), &Rational::from(-5));
        assert_eq!(neg.radius(), &Rational::from(1));
    }

    #[test]
    fn test_abs() {
        let ball = RealBall::new(Rational::from(-5), Rational::from(1), 53);
        let abs = ball.abs();

        assert_eq!(abs.midpoint(), &Rational::from(5));
    }

    #[test]
    fn test_overlaps() {
        let b1 = RealBall::new(Rational::from(0), Rational::from(2), 53);
        let b2 = RealBall::new(Rational::from(3), Rational::from(2), 53);

        assert!(b1.overlaps(&b2));

        let b3 = RealBall::new(Rational::from(10), Rational::from(1), 53);
        assert!(!b1.overlaps(&b3));
    }

    #[test]
    fn test_zero_one() {
        let zero = RealBall::zero(53);
        assert_eq!(zero.midpoint(), &Rational::zero());
        assert!(zero.is_exact());

        let one = RealBall::one(53);
        assert_eq!(one.midpoint(), &Rational::from(1));
        assert!(one.is_exact());
    }

    #[test]
    fn test_create_real_ball() {
        let ball = create_real_ball(Rational::new(1, 2), Rational::new(1, 100), 53);
        assert_eq!(ball.midpoint(), &Rational::new(1, 2));
        assert_eq!(ball.precision(), 53);
    }

    #[test]
    fn test_operators() {
        let b1 = RealBall::new(Rational::from(3), Rational::zero(), 53);
        let b2 = RealBall::new(Rational::from(2), Rational::zero(), 53);

        let sum = b1.clone() + b2.clone();
        assert_eq!(sum.midpoint(), &Rational::from(5));

        let diff = b1.clone() - b2.clone();
        assert_eq!(diff.midpoint(), &Rational::from(1));

        let prod = b1.clone() * b2;
        assert_eq!(prod.midpoint(), &Rational::from(6));

        let neg = -b1;
        assert_eq!(neg.midpoint(), &Rational::from(-3));
    }
}
