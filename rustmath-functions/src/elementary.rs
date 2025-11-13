//! Elementary functions - unified interface
//!
//! This module provides a unified interface for all elementary mathematical functions.
//! It re-exports functions from specialized modules and provides convenience functions.

// Re-export everything from specialized modules
pub use crate::exponential::*;
pub use crate::hyperbolic::*;
pub use crate::power::*;
pub use crate::trigonometric::*;
pub use crate::utility::*;

/// Evaluate a common mathematical expression using f64
///
/// This is a convenience function for quick numeric evaluation
pub mod eval {
    use super::*;

    /// Evaluate sin(x)
    pub fn sin(x: f64) -> f64 {
        sin_f64(x)
    }

    /// Evaluate cos(x)
    pub fn cos(x: f64) -> f64 {
        cos_f64(x)
    }

    /// Evaluate tan(x)
    pub fn tan(x: f64) -> f64 {
        tan_f64(x)
    }

    /// Evaluate exp(x)
    pub fn exp(x: f64) -> f64 {
        exp_f64(x)
    }

    /// Evaluate ln(x)
    pub fn ln(x: f64) -> f64 {
        ln_f64(x)
    }

    /// Evaluate sqrt(x)
    pub fn sqrt(x: f64) -> f64 {
        sqrt_f64(x)
    }

    /// Evaluate abs(x)
    pub fn abs(x: f64) -> f64 {
        abs_f64(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_comprehensive_evaluation() {
        // Test a complex expression: sin(π/6) * exp(1) + sqrt(4)
        let result = sin_f64(PI / 6.0) * exp_f64(1.0) + sqrt_f64(4.0);

        // sin(π/6) = 0.5, exp(1) = e ≈ 2.71828, sqrt(4) = 2
        // result ≈ 0.5 * 2.71828 + 2 = 1.35914 + 2 = 3.35914
        let expected = 0.5 * std::f64::consts::E + 2.0;
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_euler_identity() {
        // e^(iπ) + 1 = 0
        // Real part: e^0 * cos(π) + 1 = 1 * (-1) + 1 = 0
        let real_part = exp_f64(0.0) * cos_f64(PI) + 1.0;
        assert!(real_part.abs() < 1e-10);

        // Imaginary part: e^0 * sin(π) = 1 * 0 = 0
        let imag_part = exp_f64(0.0) * sin_f64(PI);
        assert!(imag_part.abs() < 1e-10);
    }

    #[test]
    fn test_taylor_approximation() {
        // Test Taylor series: e^x ≈ 1 + x + x²/2! + x³/3! + x⁴/4! for small x
        let x = 0.1;
        let taylor4 = 1.0 + x + pow_f64(x, 2.0) / 2.0
                     + pow_f64(x, 3.0) / 6.0 + pow_f64(x, 4.0) / 24.0;
        let actual = exp_f64(x);

        // 4th order Taylor series should be very accurate for x = 0.1
        assert!((taylor4 - actual).abs() < 1e-7);
    }

    #[test]
    fn test_logarithm_change_of_base() {
        // log_b(x) = ln(x) / ln(b)
        let x = 100.0;
        let base = 10.0;

        let using_formula = ln_f64(x) / ln_f64(base);
        let using_function = log10_f64(x);

        assert!((using_formula - using_function).abs() < 1e-10);
    }
}
