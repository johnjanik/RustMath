//! Exponential and logarithmic functions
//!
//! This module provides implementations of exponential and logarithmic functions:
//! - exp, exp2, exp10 (exponential functions)
//! - log, ln, log2, log10 (logarithmic functions)

use rustmath_symbolic::{Expr, UnaryOp, BinaryOp};
use std::sync::Arc;

/// Natural exponential function
///
/// Computes e^x where e is Euler's number (≈ 2.71828...)
pub fn exp(x: Expr) -> Expr {
    Expr::Unary(UnaryOp::Exp, Arc::new(x))
}

/// Natural exponential function for f64
#[inline]
pub fn exp_f64(x: f64) -> f64 {
    x.exp()
}

/// Base-2 exponential function
///
/// Computes 2^x
pub fn exp2(x: Expr) -> Expr {
    let two = Expr::from(2);
    Expr::Binary(BinaryOp::Pow, Arc::new(two), Arc::new(x))
}

/// Base-2 exponential function for f64
#[inline]
pub fn exp2_f64(x: f64) -> f64 {
    x.exp2()
}

/// Base-10 exponential function
///
/// Computes 10^x
pub fn exp10(x: Expr) -> Expr {
    let ten = Expr::from(10);
    Expr::Binary(BinaryOp::Pow, Arc::new(ten), Arc::new(x))
}

/// Base-10 exponential function for f64
#[inline]
pub fn exp10_f64(x: f64) -> f64 {
    10.0_f64.powf(x)
}

/// Natural logarithm function (base e)
///
/// Computes ln(x) = log_e(x)
/// Domain: (0, ∞)
pub fn ln(x: Expr) -> Expr {
    Expr::Unary(UnaryOp::Log, Arc::new(x))
}

/// Natural logarithm function for f64
#[inline]
pub fn ln_f64(x: f64) -> f64 {
    x.ln()
}

/// Natural logarithm function (alias for ln)
pub fn log(x: Expr) -> Expr {
    ln(x)
}

/// Natural logarithm function for f64 (alias for ln_f64)
#[inline]
pub fn log_f64(x: f64) -> f64 {
    ln_f64(x)
}

/// Base-2 logarithm function
///
/// Computes log_2(x)
pub fn log2(x: Expr) -> Expr {
    // log_2(x) = ln(x) / ln(2)
    let ln_x = ln(x);
    let two = Expr::from(2);
    let ln_2 = ln(two);
    ln_x / ln_2
}

/// Base-2 logarithm function for f64
#[inline]
pub fn log2_f64(x: f64) -> f64 {
    x.log2()
}

/// Base-10 logarithm function
///
/// Computes log_10(x)
pub fn log10(x: Expr) -> Expr {
    // log_10(x) = ln(x) / ln(10)
    let ln_x = ln(x);
    let ten = Expr::from(10);
    let ln_10 = ln(ten);
    ln_x / ln_10
}

/// Base-10 logarithm function for f64
#[inline]
pub fn log10_f64(x: f64) -> f64 {
    x.log10()
}

/// Logarithm with arbitrary base
///
/// Computes log_base(x)
pub fn logb(base: Expr, x: Expr) -> Expr {
    // log_b(x) = ln(x) / ln(b)
    let ln_x = ln(x);
    let ln_b = ln(base);
    ln_x / ln_b
}

/// Logarithm with arbitrary base for f64
#[inline]
pub fn logb_f64(base: f64, x: f64) -> f64 {
    x.ln() / base.ln()
}

/// exp(x) - 1
///
/// Computes e^x - 1 accurately for small values of x
#[inline]
pub fn expm1_f64(x: f64) -> f64 {
    x.exp_m1()
}

/// ln(1 + x)
///
/// Computes ln(1 + x) accurately for small values of x
#[inline]
pub fn ln1p_f64(x: f64) -> f64 {
    x.ln_1p()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{E, LN_2, LN_10};

    #[test]
    fn test_exp_f64() {
        assert!((exp_f64(0.0) - 1.0).abs() < 1e-10);
        assert!((exp_f64(1.0) - E).abs() < 1e-10);
        assert!((exp_f64(2.0) - E * E).abs() < 1e-10);
    }

    #[test]
    fn test_exp2_f64() {
        assert!((exp2_f64(0.0) - 1.0).abs() < 1e-10);
        assert!((exp2_f64(1.0) - 2.0).abs() < 1e-10);
        assert!((exp2_f64(10.0) - 1024.0).abs() < 1e-10);
    }

    #[test]
    fn test_exp10_f64() {
        assert!((exp10_f64(0.0) - 1.0).abs() < 1e-10);
        assert!((exp10_f64(1.0) - 10.0).abs() < 1e-10);
        assert!((exp10_f64(2.0) - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_ln_f64() {
        assert!((ln_f64(1.0) - 0.0).abs() < 1e-10);
        assert!((ln_f64(E) - 1.0).abs() < 1e-10);
        assert!((ln_f64(E * E) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_log2_f64() {
        assert!((log2_f64(1.0) - 0.0).abs() < 1e-10);
        assert!((log2_f64(2.0) - 1.0).abs() < 1e-10);
        assert!((log2_f64(1024.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_log10_f64() {
        assert!((log10_f64(1.0) - 0.0).abs() < 1e-10);
        assert!((log10_f64(10.0) - 1.0).abs() < 1e-10);
        assert!((log10_f64(100.0) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_logb_f64() {
        assert!((logb_f64(2.0, 8.0) - 3.0).abs() < 1e-10);
        assert!((logb_f64(10.0, 1000.0) - 3.0).abs() < 1e-10);
        assert!((logb_f64(5.0, 125.0) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_properties() {
        let values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0];

        for x in values {
            // exp(ln(x)) = x
            assert!((exp_f64(ln_f64(x)) - x).abs() < 1e-10);

            // ln(exp(x)) = x
            assert!((ln_f64(exp_f64(x)) - x).abs() < 1e-10);

            // exp2(log2(x)) = x
            assert!((exp2_f64(log2_f64(x)) - x).abs() < 1e-10);

            // exp10(log10(x)) = x
            assert!((exp10_f64(log10_f64(x)) - x).abs() < 1e-10);
        }
    }

    #[test]
    fn test_logarithm_properties() {
        // log(a * b) = log(a) + log(b)
        let a = 2.0;
        let b = 3.0;
        assert!((ln_f64(a * b) - (ln_f64(a) + ln_f64(b))).abs() < 1e-10);

        // log(a / b) = log(a) - log(b)
        assert!((ln_f64(a / b) - (ln_f64(a) - ln_f64(b))).abs() < 1e-10);

        // log(a^n) = n * log(a)
        let n = 3.0;
        assert!((ln_f64(a.powf(n)) - n * ln_f64(a)).abs() < 1e-10);
    }

    #[test]
    fn test_expm1_accuracy() {
        // For very small x, expm1 is more accurate than exp(x) - 1
        let x = 1e-10;
        let using_expm1 = expm1_f64(x);
        let using_exp = exp_f64(x) - 1.0;

        // Both should be close to x for small x, but expm1 is more accurate
        assert!((using_expm1 - x).abs() < 1e-15);
    }

    #[test]
    fn test_ln1p_accuracy() {
        // For very small x, ln1p is more accurate than ln(1 + x)
        let x = 1e-10;
        let using_ln1p = ln1p_f64(x);

        // ln(1 + x) ≈ x for small x
        assert!((using_ln1p - x).abs() < 1e-15);
    }

    #[test]
    fn test_symbolic_exp() {
        let x = Expr::symbol("x");
        let expr = exp(x);

        match expr {
            Expr::Unary(op, _) => assert_eq!(op, UnaryOp::Exp),
            _ => panic!("Expected unary operation"),
        }
    }

    #[test]
    fn test_symbolic_ln() {
        let x = Expr::symbol("x");
        let expr = ln(x);

        match expr {
            Expr::Unary(op, _) => assert_eq!(op, UnaryOp::Log),
            _ => panic!("Expected unary operation"),
        }
    }
}
