//! Special mathematical functions
//!
//! This module provides implementations of special functions commonly used in
//! mathematical computing, including:
//! - Gamma and Beta functions
//! - Riemann Zeta function
//! - Bessel functions
//! - Error functions
//!
//! These implementations use series expansions and asymptotic approximations.

// Unused imports removed
// These modules import what they need

pub mod gamma;
pub mod beta;
pub mod zeta;
pub mod bessel;
pub mod error;

pub use gamma::{gamma, ln_gamma, digamma};
pub use beta::beta;
pub use zeta::zeta;
pub use bessel::{bessel_j, bessel_y};
pub use error::{erf, erfc};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_integers() {
        // Gamma(1) = 0! = 1
        assert!((gamma(1.0) - 1.0).abs() < 1e-10);

        // Gamma(2) = 1! = 1
        assert!((gamma(2.0) - 1.0).abs() < 1e-10);

        // Gamma(3) = 2! = 2
        assert!((gamma(3.0) - 2.0).abs() < 1e-10);

        // Gamma(4) = 3! = 6
        assert!((gamma(4.0) - 6.0).abs() < 1e-10);

        // Gamma(5) = 4! = 24
        assert!((gamma(5.0) - 24.0).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_half() {
        // Gamma(1/2) = sqrt(pi)
        assert!((gamma(0.5) - PI.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_beta_function() {
        // Beta(1, 1) = 1
        assert!((beta(1.0, 1.0) - 1.0).abs() < 1e-10);

        // Beta(2, 3) = 1/12
        assert!((beta(2.0, 3.0) - 1.0/12.0).abs() < 1e-10);
    }

    #[test]
    fn test_zeta_known_values() {
        // Zeta(2) = pi^2/6
        let expected = PI * PI / 6.0;
        assert!((zeta(2.0) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_bessel_j0_zero() {
        // J_0(0) = 1
        assert!((bessel_j(0, 0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_erf_zero() {
        // erf(0) = 0
        assert!(erf(0.0).abs() < 1e-10);
    }

    #[test]
    fn test_erfc_zero() {
        // erfc(0) = 1
        assert!((erfc(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_erf_erfc_complement() {
        // erf(x) + erfc(x) = 1
        for x in &[0.5, 1.0, 1.5, 2.0] {
            assert!((erf(*x) + erfc(*x) - 1.0).abs() < 1e-10);
        }
    }
}
