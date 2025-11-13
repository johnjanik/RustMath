//! Power and root functions
//!
//! This module provides implementations of power and root functions:
//! - pow (general power function)
//! - sqrt (square root)
//! - cbrt (cube root)
//! - hypot (hypotenuse function)

use rustmath_symbolic::{Expr, BinaryOp, UnaryOp};
use std::sync::Arc;

/// Power function
///
/// Computes base^exponent
pub fn pow(base: Expr, exponent: Expr) -> Expr {
    Expr::Binary(BinaryOp::Pow, Arc::new(base), Arc::new(exponent))
}

/// Power function for f64
#[inline]
pub fn pow_f64(base: f64, exponent: f64) -> f64 {
    base.powf(exponent)
}

/// Integer power function for f64
///
/// More efficient than powf for integer exponents
#[inline]
pub fn powi_f64(base: f64, exponent: i32) -> f64 {
    base.powi(exponent)
}

/// Square root function
///
/// Computes sqrt(x) = x^(1/2)
/// Domain: [0, ∞)
pub fn sqrt(x: Expr) -> Expr {
    Expr::Unary(UnaryOp::Sqrt, Arc::new(x))
}

/// Square root function for f64
#[inline]
pub fn sqrt_f64(x: f64) -> f64 {
    x.sqrt()
}

/// Cube root function
///
/// Computes cbrt(x) = x^(1/3)
/// Unlike sqrt, cbrt is defined for negative numbers
pub fn cbrt(x: Expr) -> Expr {
    let one_third = Expr::Rational(
        rustmath_rationals::Rational::new(1, 3)
            .expect("1/3 is a valid rational")
    );
    pow(x, one_third)
}

/// Cube root function for f64
#[inline]
pub fn cbrt_f64(x: f64) -> f64 {
    x.cbrt()
}

/// N-th root function
///
/// Computes the n-th root of x: x^(1/n)
pub fn root(x: Expr, n: i64) -> Expr {
    let exponent = Expr::Rational(
        rustmath_rationals::Rational::new(1, n)
            .expect("1/n is a valid rational")
    );
    pow(x, exponent)
}

/// N-th root function for f64
#[inline]
pub fn root_f64(x: f64, n: i32) -> f64 {
    x.powf(1.0 / n as f64)
}

/// Hypotenuse function
///
/// Computes sqrt(x² + y²) without overflow or underflow at intermediate stages
#[inline]
pub fn hypot_f64(x: f64, y: f64) -> f64 {
    x.hypot(y)
}

/// Three-dimensional hypotenuse
///
/// Computes sqrt(x² + y² + z²)
#[inline]
pub fn hypot3_f64(x: f64, y: f64, z: f64) -> f64 {
    (x * x + y * y + z * z).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pow_f64() {
        assert!((pow_f64(2.0, 3.0) - 8.0).abs() < 1e-10);
        assert!((pow_f64(10.0, 2.0) - 100.0).abs() < 1e-10);
        assert!((pow_f64(2.0, 0.5) - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_powi_f64() {
        assert!((powi_f64(2.0, 10) - 1024.0).abs() < 1e-10);
        assert!((powi_f64(3.0, 4) - 81.0).abs() < 1e-10);
        assert!((powi_f64(5.0, 0) - 1.0).abs() < 1e-10);
        assert!((powi_f64(2.0, -3) - 0.125).abs() < 1e-10);
    }

    #[test]
    fn test_sqrt_f64() {
        assert!((sqrt_f64(0.0) - 0.0).abs() < 1e-10);
        assert!((sqrt_f64(1.0) - 1.0).abs() < 1e-10);
        assert!((sqrt_f64(4.0) - 2.0).abs() < 1e-10);
        assert!((sqrt_f64(9.0) - 3.0).abs() < 1e-10);
        assert!((sqrt_f64(2.0) - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_cbrt_f64() {
        assert!((cbrt_f64(0.0) - 0.0).abs() < 1e-10);
        assert!((cbrt_f64(1.0) - 1.0).abs() < 1e-10);
        assert!((cbrt_f64(8.0) - 2.0).abs() < 1e-10);
        assert!((cbrt_f64(27.0) - 3.0).abs() < 1e-10);

        // cbrt works for negative numbers
        assert!((cbrt_f64(-8.0) + 2.0).abs() < 1e-10);
        assert!((cbrt_f64(-27.0) + 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_root_f64() {
        // 4th root of 16 = 2
        assert!((root_f64(16.0, 4) - 2.0).abs() < 1e-10);

        // 5th root of 32 = 2
        assert!((root_f64(32.0, 5) - 2.0).abs() < 1e-10);

        // 10th root of 1024 = 2
        assert!((root_f64(1024.0, 10) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_hypot_f64() {
        // 3-4-5 right triangle
        assert!((hypot_f64(3.0, 4.0) - 5.0).abs() < 1e-10);

        // 5-12-13 right triangle
        assert!((hypot_f64(5.0, 12.0) - 13.0).abs() < 1e-10);

        // Unit circle
        assert!((hypot_f64(1.0, 0.0) - 1.0).abs() < 1e-10);
        assert!((hypot_f64(0.0, 1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hypot3_f64() {
        // 2-3-6 right "triangle" in 3D (2² + 3² + 6² = 4 + 9 + 36 = 49)
        assert!((hypot3_f64(2.0, 3.0, 6.0) - 7.0).abs() < 1e-10);

        // Unit sphere
        let val = 1.0 / 3.0_f64.sqrt();
        assert!((hypot3_f64(val, val, val) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sqrt_cbrt_relationship() {
        // sqrt(cbrt(x)) = x^(1/6)
        let x = 64.0;
        let result = sqrt_f64(cbrt_f64(x));
        let expected = pow_f64(x, 1.0 / 6.0);
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_power_laws() {
        let x = 2.0;
        let a = 3.0;
        let b = 4.0;

        // x^a * x^b = x^(a+b)
        let lhs = pow_f64(x, a) * pow_f64(x, b);
        let rhs = pow_f64(x, a + b);
        assert!((lhs - rhs).abs() < 1e-10);

        // (x^a)^b = x^(a*b)
        let lhs = pow_f64(pow_f64(x, a), b);
        let rhs = pow_f64(x, a * b);
        assert!((lhs - rhs).abs() < 1e-10);

        // x^a / x^b = x^(a-b)
        let lhs = pow_f64(x, a) / pow_f64(x, b);
        let rhs = pow_f64(x, a - b);
        assert!((lhs - rhs).abs() < 1e-10);
    }

    #[test]
    fn test_symbolic_pow() {
        let x = Expr::symbol("x");
        let n = Expr::from(2);
        let expr = pow(x, n);

        match expr {
            Expr::Binary(op, _, _) => assert_eq!(op, BinaryOp::Pow),
            _ => panic!("Expected binary operation"),
        }
    }

    #[test]
    fn test_symbolic_sqrt() {
        let x = Expr::symbol("x");
        let expr = sqrt(x);

        match expr {
            Expr::Unary(op, _) => assert_eq!(op, UnaryOp::Sqrt),
            _ => panic!("Expected unary operation"),
        }
    }
}
