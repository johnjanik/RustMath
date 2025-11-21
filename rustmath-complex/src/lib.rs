//! Complex numbers with real and imaginary parts
//!
//! Provides complex number arithmetic over various real number types.
//!
//! This module includes:
//! - `Complex`: Standard precision complex numbers (f64-based)
//! - `ComplexMPFR`: Arbitrary precision complex numbers using MPC/MPFR
//! - `ComplexBall`: Complex ball arithmetic with rigorous error bounds
//! - `ComplexIntervalFieldElement`: Complex interval arithmetic
//! - `ComplexIntervalField`: Factory for complex intervals
//! - MPC morphisms and helper functions

pub mod complex;
pub mod mpc;
pub mod complex_arb;
pub mod complex_interval;
pub mod complex_interval_field;
pub mod complex_mpc_ext;

pub use complex::{Complex, ComplexField};
pub use mpc::{ComplexMPFR, DEFAULT_PRECISION};
pub use complex_arb::{ComplexBall, ComplexBallField, IntegrationContext};
pub use complex_interval::{
    ComplexIntervalFieldElement, create_complex_interval_field_element,
    is_complex_interval_field_element, make_complex_interval_field_element0,
};
pub use complex_interval_field::{complex_interval_field, ComplexIntervalFieldClass};
pub use complex_mpc_ext::{
    CCtoMPC, INTEGERtoMPC, MPCtoMPC, MPFRtoMPC, mpcomplex_field,
    MpcomplexFieldClass, MPComplexNumber, late_import, split_complex_string,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_complex() {
        let z = Complex::new(3.0, 4.0);
        assert!((z.real() - 3.0).abs() < 1e-10);
        assert!((z.imag() - 4.0).abs() < 1e-10);
        assert!((z.abs() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn basic_complex_mpfr() {
        let z = ComplexMPFR::from((3.0, 4.0));
        assert!((z.real() - 3.0).abs() < 1e-10);
        assert!((z.imag() - 4.0).abs() < 1e-10);
        assert!((z.abs().to_f64() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn high_precision_complex() {
        // Test with 256 bits of precision
        let z = ComplexMPFR::with_val(256, (3.0, 4.0));
        assert_eq!(z.precision(), 256);

        let w = ComplexMPFR::with_val(256, (1.0, 2.0));
        let product = z * w;

        // (3+4i)(1+2i) = -5+10i
        assert!((product.real() - (-5.0)).abs() < 1e-10);
        assert!((product.imag() - 10.0).abs() < 1e-10);
    }
}
