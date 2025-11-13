//! Hyperbolic functions
//!
//! This module provides implementations of hyperbolic functions and their inverses:
//! - sinh, cosh, tanh (hyperbolic functions)
//! - sech, csch, coth (reciprocal hyperbolic functions)
//! - arsinh, arcosh, artanh (inverse hyperbolic functions)

use rustmath_symbolic::{Expr, UnaryOp};
use std::sync::Arc;

/// Hyperbolic sine function
///
/// Computes sinh(x) = (e^x - e^(-x)) / 2
pub fn sinh(x: Expr) -> Expr {
    Expr::Unary(UnaryOp::Sinh, Arc::new(x))
}

/// Hyperbolic sine function for f64
#[inline]
pub fn sinh_f64(x: f64) -> f64 {
    x.sinh()
}

/// Hyperbolic cosine function
///
/// Computes cosh(x) = (e^x + e^(-x)) / 2
pub fn cosh(x: Expr) -> Expr {
    Expr::Unary(UnaryOp::Cosh, Arc::new(x))
}

/// Hyperbolic cosine function for f64
#[inline]
pub fn cosh_f64(x: f64) -> f64 {
    x.cosh()
}

/// Hyperbolic tangent function
///
/// Computes tanh(x) = sinh(x) / cosh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
pub fn tanh(x: Expr) -> Expr {
    Expr::Unary(UnaryOp::Tanh, Arc::new(x))
}

/// Hyperbolic tangent function for f64
#[inline]
pub fn tanh_f64(x: f64) -> f64 {
    x.tanh()
}

/// Hyperbolic secant function
///
/// Computes sech(x) = 1 / cosh(x)
pub fn sech(x: Expr) -> Expr {
    let one = Expr::from(1);
    let cosh_x = cosh(x);
    one / cosh_x
}

/// Hyperbolic secant function for f64
#[inline]
pub fn sech_f64(x: f64) -> f64 {
    1.0 / x.cosh()
}

/// Hyperbolic cosecant function
///
/// Computes csch(x) = 1 / sinh(x)
pub fn csch(x: Expr) -> Expr {
    let one = Expr::from(1);
    let sinh_x = sinh(x);
    one / sinh_x
}

/// Hyperbolic cosecant function for f64
#[inline]
pub fn csch_f64(x: f64) -> f64 {
    1.0 / x.sinh()
}

/// Hyperbolic cotangent function
///
/// Computes coth(x) = cosh(x) / sinh(x) = 1 / tanh(x)
pub fn coth(x: Expr) -> Expr {
    let one = Expr::from(1);
    let tanh_x = tanh(x);
    one / tanh_x
}

/// Hyperbolic cotangent function for f64
#[inline]
pub fn coth_f64(x: f64) -> f64 {
    1.0 / x.tanh()
}

/// Inverse hyperbolic sine function
///
/// Computes arsinh(x) = ln(x + sqrt(x² + 1))
/// Domain: (-∞, ∞)
#[inline]
pub fn arsinh_f64(x: f64) -> f64 {
    x.asinh()
}

/// Inverse hyperbolic cosine function
///
/// Computes arcosh(x) = ln(x + sqrt(x² - 1))
/// Domain: [1, ∞)
#[inline]
pub fn arcosh_f64(x: f64) -> f64 {
    x.acosh()
}

/// Inverse hyperbolic tangent function
///
/// Computes artanh(x) = 0.5 * ln((1 + x) / (1 - x))
/// Domain: (-1, 1)
#[inline]
pub fn artanh_f64(x: f64) -> f64 {
    x.atanh()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sinh_f64() {
        assert!((sinh_f64(0.0) - 0.0).abs() < 1e-10);
        assert!((sinh_f64(1.0) - 1.1752011936438014).abs() < 1e-10);
        assert!((sinh_f64(-1.0) + 1.1752011936438014).abs() < 1e-10);
    }

    #[test]
    fn test_cosh_f64() {
        assert!((cosh_f64(0.0) - 1.0).abs() < 1e-10);
        assert!((cosh_f64(1.0) - 1.5430806348152437).abs() < 1e-10);
        assert!((cosh_f64(-1.0) - 1.5430806348152437).abs() < 1e-10); // cosh is even
    }

    #[test]
    fn test_tanh_f64() {
        assert!((tanh_f64(0.0) - 0.0).abs() < 1e-10);
        assert!((tanh_f64(1.0) - 0.7615941559557649).abs() < 1e-10);
        assert!((tanh_f64(-1.0) + 0.7615941559557649).abs() < 1e-10);

        // tanh approaches ±1 as x → ±∞
        assert!((tanh_f64(10.0) - 1.0).abs() < 1e-8);
        assert!((tanh_f64(-10.0) + 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_sech_f64() {
        assert!((sech_f64(0.0) - 1.0).abs() < 1e-10);
        assert!((sech_f64(1.0) - 1.0 / 1.5430806348152437).abs() < 1e-10);
    }

    #[test]
    fn test_csch_f64() {
        assert!((csch_f64(1.0) - 1.0 / 1.1752011936438014).abs() < 1e-10);
    }

    #[test]
    fn test_coth_f64() {
        assert!((coth_f64(1.0) - 1.0 / 0.7615941559557649).abs() < 1e-10);
    }

    #[test]
    fn test_arsinh_f64() {
        assert!((arsinh_f64(0.0) - 0.0).abs() < 1e-10);
        assert!((arsinh_f64(1.0) - 0.881373587019543).abs() < 1e-10);
        assert!((arsinh_f64(-1.0) + 0.881373587019543).abs() < 1e-10);
    }

    #[test]
    fn test_arcosh_f64() {
        assert!((arcosh_f64(1.0) - 0.0).abs() < 1e-10);
        assert!((arcosh_f64(2.0) - 1.3169578969248166).abs() < 1e-10);
    }

    #[test]
    fn test_artanh_f64() {
        assert!((artanh_f64(0.0) - 0.0).abs() < 1e-10);
        assert!((artanh_f64(0.5) - 0.5493061443340548).abs() < 1e-10);
        assert!((artanh_f64(-0.5) + 0.5493061443340548).abs() < 1e-10);
    }

    #[test]
    fn test_hyperbolic_identity() {
        // cosh²(x) - sinh²(x) = 1
        let values = [0.0, 0.5, 1.0, 2.0, 5.0];

        for x in values {
            let cosh_x = cosh_f64(x);
            let sinh_x = sinh_f64(x);
            let result = cosh_x * cosh_x - sinh_x * sinh_x;
            assert!((result - 1.0).abs() < 1e-10, "Failed for x = {}", x);
        }
    }

    #[test]
    fn test_inverse_properties() {
        let values = [0.0, 0.5, 1.0, 2.0];

        for x in values {
            // sinh(arsinh(x)) = x
            assert!((sinh_f64(arsinh_f64(x)) - x).abs() < 1e-10);

            // arsinh(sinh(x)) = x
            assert!((arsinh_f64(sinh_f64(x)) - x).abs() < 1e-10);
        }

        // Test arcosh (domain: x >= 1)
        let values_cosh = [1.0, 1.5, 2.0, 3.0];
        for x in values_cosh {
            assert!((cosh_f64(arcosh_f64(x)) - x).abs() < 1e-10);
            assert!((arcosh_f64(cosh_f64(x)) - x).abs() < 1e-10);
        }

        // Test artanh (domain: -1 < x < 1)
        let values_tanh = [0.0, 0.3, 0.5, 0.9];
        for x in values_tanh {
            assert!((tanh_f64(artanh_f64(x)) - x).abs() < 1e-10);
            assert!((artanh_f64(tanh_f64(x)) - x).abs() < 1e-10);
        }
    }

    #[test]
    fn test_definitions() {
        let x: f64 = 1.5;

        // sinh(x) = (e^x - e^(-x)) / 2
        let sinh_def = (x.exp() - (-x).exp()) / 2.0;
        assert!((sinh_f64(x) - sinh_def).abs() < 1e-10);

        // cosh(x) = (e^x + e^(-x)) / 2
        let cosh_def = (x.exp() + (-x).exp()) / 2.0;
        assert!((cosh_f64(x) - cosh_def).abs() < 1e-10);

        // tanh(x) = sinh(x) / cosh(x)
        let tanh_def = sinh_f64(x) / cosh_f64(x);
        assert!((tanh_f64(x) - tanh_def).abs() < 1e-10);
    }

    #[test]
    fn test_symbolic_sinh() {
        let x = Expr::symbol("x");
        let expr = sinh(x);

        match expr {
            Expr::Unary(op, _) => assert_eq!(op, UnaryOp::Sinh),
            _ => panic!("Expected unary operation"),
        }
    }
}
