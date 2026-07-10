//! Real numbers with configurable precision
//!
//! This module provides real number arithmetic with support for:
//! - Standard f64 precision (`Real` type)
//! - Arbitrary precision using MPFR (`RealMPFR` type)
//! - Configurable rounding modes
//! - Transcendental functions (sin, cos, exp, log, etc.)
//! - Interval arithmetic for verified computations
//! - Conversions from integers and rationals

pub mod interval;
pub mod mpfr;
pub mod real;
pub mod rounding;
pub mod transcendental;

// Wave 0 (MAGMA port): pure-Rust arbitrary-precision real (additive).
pub mod bigfloat;
// Wave 0 (MAGMA port): wire RealMPFR into the new core traits (additive).
pub mod mpfr_impls;

pub use interval::Interval;
pub use mpfr::{RealMPFR, DEFAULT_PRECISION};
pub use real::{Real, RealField};
pub use rounding::RoundingMode;
// Wave 0: selective re-exports (NB: the *trait* `rustmath_core::analytic::RealField`
// is deliberately not re-exported anywhere — it would collide with the
// `RealField` struct above; import it path-qualified).
pub use bigfloat::{BigFloat, BigFloatField};

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
