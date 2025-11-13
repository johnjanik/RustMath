//! RustMath Functions - Elementary mathematical functions
//!
//! This crate provides implementations of elementary mathematical functions including:
//! - Trigonometric functions (sin, cos, tan, sec, csc, cot)
//! - Inverse trigonometric functions (arcsin, arccos, arctan)
//! - Hyperbolic functions (sinh, cosh, tanh, sech, csch, coth)
//! - Inverse hyperbolic functions (arsinh, arcosh, artanh)
//! - Exponential and logarithmic functions (exp, log, ln)
//! - Power and root functions (pow, sqrt, cbrt)
//! - Utility functions (abs, sign, floor, ceil, round)
//!
//! Functions support both numeric evaluation (for concrete types like f64, Integer, Rational)
//! and symbolic representation (for use in expression trees).

pub mod elementary;
pub mod exponential;
pub mod hyperbolic;
pub mod power;
pub mod trigonometric;
pub mod utility;

// Re-export commonly used functions
pub use elementary::*;
pub use exponential::*;
pub use hyperbolic::*;
pub use power::*;
pub use trigonometric::*;
pub use utility::*;

use rustmath_symbolic::Expr;

/// Trait for mathematical functions that can be evaluated numerically
pub trait NumericFunction<T> {
    /// Evaluate the function at the given value
    fn eval(&self, x: T) -> T;
}

/// Trait for mathematical functions that have symbolic representations
pub trait SymbolicFunction {
    /// Convert the function applied to an expression into a symbolic expression
    fn to_symbolic(&self, arg: Expr) -> Expr;

    /// Get the symbolic derivative of this function
    fn derivative(&self) -> Box<dyn SymbolicFunction>;
}

/// Mathematical constants
pub mod constants {
    use std::f64::consts as f64_consts;

    /// Pi (π ≈ 3.14159...)
    pub const PI: f64 = f64_consts::PI;

    /// Euler's number (e ≈ 2.71828...)
    pub const E: f64 = f64_consts::E;

    /// Tau (τ = 2π ≈ 6.28318...)
    pub const TAU: f64 = f64_consts::TAU;

    /// Golden ratio (φ ≈ 1.61803...)
    pub const PHI: f64 = 1.618033988749895;

    /// Square root of 2 (√2 ≈ 1.41421...)
    pub const SQRT_2: f64 = f64_consts::SQRT_2;

    /// Natural logarithm of 2 (ln(2) ≈ 0.69315...)
    pub const LN_2: f64 = f64_consts::LN_2;

    /// Natural logarithm of 10 (ln(10) ≈ 2.30259...)
    pub const LN_10: f64 = f64_consts::LN_10;

    /// Base-10 logarithm of e (log₁₀(e) ≈ 0.43429...)
    pub const LOG10_E: f64 = f64_consts::LOG10_E;

    /// Base-2 logarithm of e (log₂(e) ≈ 1.44270...)
    pub const LOG2_E: f64 = f64_consts::LOG2_E;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        use constants::*;

        assert!((PI - 3.14159265358979).abs() < 1e-10);
        assert!((E - 2.71828182845905).abs() < 1e-10);
        assert!((TAU - 2.0 * PI).abs() < 1e-10);
    }
}
