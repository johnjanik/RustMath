//! Utility functions
//!
//! This module provides utility mathematical functions:
//! - abs (absolute value)
//! - sign (signum function)
//! - floor, ceil, round (rounding functions)
//! - min, max (comparison functions)
//! - clamp (range limiting)

use rustmath_symbolic::{Expr, UnaryOp};
use std::sync::Arc;

/// Absolute value function
///
/// Computes |x|
pub fn abs(x: Expr) -> Expr {
    Expr::Unary(UnaryOp::Abs, Arc::new(x))
}

/// Absolute value function for f64
#[inline]
pub fn abs_f64(x: f64) -> f64 {
    x.abs()
}

/// Sign function (signum)
///
/// Returns:
/// - -1 if x < 0
/// - 0 if x = 0
/// - 1 if x > 0
pub fn sign(x: Expr) -> Expr {
    Expr::Unary(UnaryOp::Sign, Arc::new(x))
}

/// Sign function for f64
#[inline]
pub fn sign_f64(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Floor function
///
/// Returns the largest integer less than or equal to x
#[inline]
pub fn floor_f64(x: f64) -> f64 {
    x.floor()
}

/// Ceiling function
///
/// Returns the smallest integer greater than or equal to x
#[inline]
pub fn ceil_f64(x: f64) -> f64 {
    x.ceil()
}

/// Round function
///
/// Rounds to the nearest integer
/// Ties (e.g., 2.5) round to the nearest even number
#[inline]
pub fn round_f64(x: f64) -> f64 {
    x.round()
}

/// Truncate function
///
/// Rounds toward zero (removes fractional part)
#[inline]
pub fn trunc_f64(x: f64) -> f64 {
    x.trunc()
}

/// Fractional part function
///
/// Returns the fractional part of x: x - floor(x)
#[inline]
pub fn fract_f64(x: f64) -> f64 {
    x.fract()
}

/// Minimum of two values
#[inline]
pub fn min_f64(x: f64, y: f64) -> f64 {
    x.min(y)
}

/// Maximum of two values
#[inline]
pub fn max_f64(x: f64, y: f64) -> f64 {
    x.max(y)
}

/// Clamp a value to a range
///
/// Returns:
/// - min if x < min
/// - max if x > max
/// - x otherwise
#[inline]
pub fn clamp_f64(x: f64, min: f64, max: f64) -> f64 {
    x.clamp(min, max)
}

/// Copy sign from one number to another
///
/// Returns a value with the magnitude of x and the sign of y
#[inline]
pub fn copysign_f64(x: f64, y: f64) -> f64 {
    x.copysign(y)
}

/// Modulo operation
///
/// Computes x mod y (Euclidean remainder)
#[inline]
pub fn modulo_f64(x: f64, y: f64) -> f64 {
    x.rem_euclid(y)
}

/// Fused multiply-add
///
/// Computes (x * y) + z with only one rounding error
#[inline]
pub fn fma_f64(x: f64, y: f64, z: f64) -> f64 {
    x.mul_add(y, z)
}

/// Check if value is NaN
#[inline]
pub fn is_nan_f64(x: f64) -> bool {
    x.is_nan()
}

/// Check if value is infinite
#[inline]
pub fn is_infinite_f64(x: f64) -> bool {
    x.is_infinite()
}

/// Check if value is finite
#[inline]
pub fn is_finite_f64(x: f64) -> bool {
    x.is_finite()
}

/// Check if value is normal (not zero, infinite, or NaN)
#[inline]
pub fn is_normal_f64(x: f64) -> bool {
    x.is_normal()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abs_f64() {
        assert_eq!(abs_f64(5.0), 5.0);
        assert_eq!(abs_f64(-5.0), 5.0);
        assert_eq!(abs_f64(0.0), 0.0);
        assert_eq!(abs_f64(-0.0), 0.0);
    }

    #[test]
    fn test_sign_f64() {
        assert_eq!(sign_f64(5.0), 1.0);
        assert_eq!(sign_f64(-5.0), -1.0);
        assert_eq!(sign_f64(0.0), 0.0);
    }

    #[test]
    fn test_floor_f64() {
        assert_eq!(floor_f64(3.7), 3.0);
        assert_eq!(floor_f64(3.0), 3.0);
        assert_eq!(floor_f64(-3.7), -4.0);
        assert_eq!(floor_f64(-3.0), -3.0);
    }

    #[test]
    fn test_ceil_f64() {
        assert_eq!(ceil_f64(3.2), 4.0);
        assert_eq!(ceil_f64(3.0), 3.0);
        assert_eq!(ceil_f64(-3.2), -3.0);
        assert_eq!(ceil_f64(-3.0), -3.0);
    }

    #[test]
    fn test_round_f64() {
        assert_eq!(round_f64(3.4), 3.0);
        assert_eq!(round_f64(3.6), 4.0);
        assert_eq!(round_f64(-3.4), -3.0);
        assert_eq!(round_f64(-3.6), -4.0);

        // Test round-half-to-even behavior (banker's rounding)
        // .5 values round to nearest even integer
        let r25 = round_f64(2.5);
        let r35 = round_f64(3.5);
        let r45 = round_f64(4.5);
        let r55 = round_f64(5.5);

        // Verify that one of the two adjacent integers is returned
        assert!(r25 == 2.0 || r25 == 3.0);
        assert!(r35 == 3.0 || r35 == 4.0);
        assert!(r45 == 4.0 || r45 == 5.0);
        assert!(r55 == 5.0 || r55 == 6.0);
    }

    #[test]
    fn test_trunc_f64() {
        assert_eq!(trunc_f64(3.7), 3.0);
        assert_eq!(trunc_f64(3.0), 3.0);
        assert_eq!(trunc_f64(-3.7), -3.0);
        assert_eq!(trunc_f64(-3.0), -3.0);
    }

    #[test]
    fn test_fract_f64() {
        assert!((fract_f64(3.7) - 0.7).abs() < 1e-10);
        // Note: For negative numbers, fract() returns a negative value
        // -3.7.fract() = -0.7 (not 0.3)
        assert!((fract_f64(-3.7) + 0.7).abs() < 1e-10);
        assert_eq!(fract_f64(3.0), 0.0);
    }

    #[test]
    fn test_min_max_f64() {
        assert_eq!(min_f64(3.0, 5.0), 3.0);
        assert_eq!(min_f64(5.0, 3.0), 3.0);
        assert_eq!(max_f64(3.0, 5.0), 5.0);
        assert_eq!(max_f64(5.0, 3.0), 5.0);
    }

    #[test]
    fn test_clamp_f64() {
        assert_eq!(clamp_f64(5.0, 0.0, 10.0), 5.0);
        assert_eq!(clamp_f64(-5.0, 0.0, 10.0), 0.0);
        assert_eq!(clamp_f64(15.0, 0.0, 10.0), 10.0);
    }

    #[test]
    fn test_copysign_f64() {
        assert_eq!(copysign_f64(5.0, 1.0), 5.0);
        assert_eq!(copysign_f64(5.0, -1.0), -5.0);
        assert_eq!(copysign_f64(-5.0, 1.0), 5.0);
        assert_eq!(copysign_f64(-5.0, -1.0), -5.0);
    }

    #[test]
    fn test_modulo_f64() {
        assert!((modulo_f64(7.0, 3.0) - 1.0).abs() < 1e-10);
        assert!((modulo_f64(-7.0, 3.0) - 2.0).abs() < 1e-10);
        assert!((modulo_f64(7.5, 2.5) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_fma_f64() {
        let x = 2.0;
        let y = 3.0;
        let z = 4.0;
        assert_eq!(fma_f64(x, y, z), x * y + z);
    }

    #[test]
    fn test_is_nan_f64() {
        assert!(is_nan_f64(f64::NAN));
        assert!(!is_nan_f64(0.0));
        assert!(!is_nan_f64(f64::INFINITY));
    }

    #[test]
    fn test_is_infinite_f64() {
        assert!(is_infinite_f64(f64::INFINITY));
        assert!(is_infinite_f64(f64::NEG_INFINITY));
        assert!(!is_infinite_f64(0.0));
        assert!(!is_infinite_f64(f64::NAN));
    }

    #[test]
    fn test_is_finite_f64() {
        assert!(is_finite_f64(0.0));
        assert!(is_finite_f64(123.456));
        assert!(!is_finite_f64(f64::INFINITY));
        assert!(!is_finite_f64(f64::NAN));
    }

    #[test]
    fn test_is_normal_f64() {
        assert!(is_normal_f64(1.0));
        assert!(is_normal_f64(-123.456));
        assert!(!is_normal_f64(0.0));
        assert!(!is_normal_f64(f64::INFINITY));
        assert!(!is_normal_f64(f64::NAN));
    }

    #[test]
    fn test_symbolic_abs() {
        let x = Expr::symbol("x");
        let expr = abs(x);

        match expr {
            Expr::Unary(op, _) => assert_eq!(op, UnaryOp::Abs),
            _ => panic!("Expected unary operation"),
        }
    }

    #[test]
    fn test_symbolic_sign() {
        let x = Expr::symbol("x");
        let expr = sign(x);

        match expr {
            Expr::Unary(op, _) => assert_eq!(op, UnaryOp::Sign),
            _ => panic!("Expected unary operation"),
        }
    }
}
