//! Real numbers with configurable precision
//!
//! This module provides real number arithmetic with support for:
//! - Standard f64 precision
//! - Configurable rounding modes
//! - Transcendental functions (sin, cos, exp, log, etc.)
//! - Interval arithmetic for verified computations
//! - Conversions from integers and rationals

pub mod interval;
pub mod real;
pub mod rounding;
pub mod transcendental;

pub use interval::Interval;
pub use real::Real;
pub use rounding::RoundingMode;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_arithmetic() {
        let a = Real::from(3.0);
        let b = Real::from(4.0);
        let c = a.clone() + b.clone();
        assert!((c.to_f64() - 7.0).abs() < 1e-10);
    }
}
