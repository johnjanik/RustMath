//! Trigonometric functions
//!
//! This module provides implementations of trigonometric functions and their inverses:
//! - sin, cos, tan (basic trigonometric functions)
//! - sec, csc, cot (reciprocal trigonometric functions)
//! - arcsin, arccos, arctan (inverse trigonometric functions)

use rustmath_symbolic::{Expr, UnaryOp};

/// Sine function
///
/// Computes sin(x) for a given value or expression.
///
/// # Examples
///
/// ```
/// use rustmath_functions::{sin, sin_f64};
/// use rustmath_symbolic::Expr;
///
/// // Numeric evaluation
/// let result = sin_f64(std::f64::consts::PI / 2.0);
/// assert!((result - 1.0).abs() < 1e-10);
///
/// // Symbolic evaluation
/// let x = Expr::symbol("x");
/// let expr = sin(x);
/// ```
pub fn sin(x: Expr) -> Expr {
    Expr::Unary(UnaryOp::Sin, std::sync::Arc::new(x))
}

/// Sine function for f64
#[inline]
pub fn sin_f64(x: f64) -> f64 {
    x.sin()
}

/// Cosine function
///
/// Computes cos(x) for a given value or expression.
pub fn cos(x: Expr) -> Expr {
    Expr::Unary(UnaryOp::Cos, std::sync::Arc::new(x))
}

/// Cosine function for f64
#[inline]
pub fn cos_f64(x: f64) -> f64 {
    x.cos()
}

/// Tangent function
///
/// Computes tan(x) for a given value or expression.
pub fn tan(x: Expr) -> Expr {
    Expr::Unary(UnaryOp::Tan, std::sync::Arc::new(x))
}

/// Tangent function for f64
#[inline]
pub fn tan_f64(x: f64) -> f64 {
    x.tan()
}

/// Secant function
///
/// Computes sec(x) = 1/cos(x)
pub fn sec(x: Expr) -> Expr {
    let one = Expr::from(1);
    let cos_x = cos(x);
    one / cos_x
}

/// Secant function for f64
#[inline]
pub fn sec_f64(x: f64) -> f64 {
    1.0 / x.cos()
}

/// Cosecant function
///
/// Computes csc(x) = 1/sin(x)
pub fn csc(x: Expr) -> Expr {
    let one = Expr::from(1);
    let sin_x = sin(x);
    one / sin_x
}

/// Cosecant function for f64
#[inline]
pub fn csc_f64(x: f64) -> f64 {
    1.0 / x.sin()
}

/// Cotangent function
///
/// Computes cot(x) = 1/tan(x) = cos(x)/sin(x)
pub fn cot(x: Expr) -> Expr {
    let one = Expr::from(1);
    let tan_x = tan(x);
    one / tan_x
}

/// Cotangent function for f64
#[inline]
pub fn cot_f64(x: f64) -> f64 {
    1.0 / x.tan()
}

/// Arcsine (inverse sine) function
///
/// Computes arcsin(x) for a given value or expression.
/// Domain: [-1, 1]
/// Range: [-π/2, π/2]
pub fn arcsin(x: Expr) -> Expr {
    Expr::Unary(UnaryOp::Arcsin, std::sync::Arc::new(x))
}

/// Arcsine function for f64
#[inline]
pub fn arcsin_f64(x: f64) -> f64 {
    x.asin()
}

/// Arccosine (inverse cosine) function
///
/// Computes arccos(x) for a given value or expression.
/// Domain: [-1, 1]
/// Range: [0, π]
pub fn arccos(x: Expr) -> Expr {
    Expr::Unary(UnaryOp::Arccos, std::sync::Arc::new(x))
}

/// Arccosine function for f64
#[inline]
pub fn arccos_f64(x: f64) -> f64 {
    x.acos()
}

/// Arctangent (inverse tangent) function
///
/// Computes arctan(x) for a given value or expression.
/// Domain: (-∞, ∞)
/// Range: (-π/2, π/2)
pub fn arctan(x: Expr) -> Expr {
    Expr::Unary(UnaryOp::Arctan, std::sync::Arc::new(x))
}

/// Arctangent function for f64
#[inline]
pub fn arctan_f64(x: f64) -> f64 {
    x.atan()
}

/// Two-argument arctangent function
///
/// Computes atan2(y, x) = arctan(y/x) with proper quadrant handling.
/// Returns the angle θ such that x = r*cos(θ) and y = r*sin(θ).
///
/// This function handles all quadrants correctly and avoids division by zero.
#[inline]
pub fn atan2_f64(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_sin_f64() {
        assert!((sin_f64(0.0) - 0.0).abs() < 1e-10);
        assert!((sin_f64(PI / 2.0) - 1.0).abs() < 1e-10);
        assert!((sin_f64(PI) - 0.0).abs() < 1e-10);
        assert!((sin_f64(3.0 * PI / 2.0) + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cos_f64() {
        assert!((cos_f64(0.0) - 1.0).abs() < 1e-10);
        assert!((cos_f64(PI / 2.0) - 0.0).abs() < 1e-10);
        assert!((cos_f64(PI) + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tan_f64() {
        assert!((tan_f64(0.0) - 0.0).abs() < 1e-10);
        assert!((tan_f64(PI / 4.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sec_f64() {
        assert!((sec_f64(0.0) - 1.0).abs() < 1e-10);
        assert!((sec_f64(PI / 3.0) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_csc_f64() {
        assert!((csc_f64(PI / 2.0) - 1.0).abs() < 1e-10);
        assert!((csc_f64(PI / 6.0) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_cot_f64() {
        assert!((cot_f64(PI / 4.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_arcsin_f64() {
        assert!((arcsin_f64(0.0) - 0.0).abs() < 1e-10);
        assert!((arcsin_f64(1.0) - PI / 2.0).abs() < 1e-10);
        assert!((arcsin_f64(-1.0) + PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_arccos_f64() {
        assert!((arccos_f64(1.0) - 0.0).abs() < 1e-10);
        assert!((arccos_f64(0.0) - PI / 2.0).abs() < 1e-10);
        assert!((arccos_f64(-1.0) - PI).abs() < 1e-10);
    }

    #[test]
    fn test_arctan_f64() {
        assert!((arctan_f64(0.0) - 0.0).abs() < 1e-10);
        assert!((arctan_f64(1.0) - PI / 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_atan2_f64() {
        assert!((atan2_f64(1.0, 1.0) - PI / 4.0).abs() < 1e-10);
        assert!((atan2_f64(1.0, 0.0) - PI / 2.0).abs() < 1e-10);
        assert!((atan2_f64(-1.0, -1.0) + 3.0 * PI / 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_symbolic_sin() {
        let x = Expr::symbol("x");
        let expr = sin(x);

        // Check that it's a unary operation
        match expr {
            Expr::Unary(op, _) => assert_eq!(op, UnaryOp::Sin),
            _ => panic!("Expected unary operation"),
        }
    }

    #[test]
    fn test_pythagorean_identity() {
        // sin²(x) + cos²(x) = 1
        let values = [0.0, PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0];

        for x in values {
            let sin_x = sin_f64(x);
            let cos_x = cos_f64(x);
            let result = sin_x * sin_x + cos_x * cos_x;
            assert!((result - 1.0).abs() < 1e-10, "Failed for x = {}", x);
        }
    }
}
