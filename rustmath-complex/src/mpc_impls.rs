//! Wave 0 (MAGMA port): wire the rug/MPC-backed [`ComplexMPFR`](crate::ComplexMPFR)
//! into the new `rustmath-core` [`ComplexField`] trait.
//!
//! MAGMA source: Handbook chapter 25 (complex fields).
//!
//! `ComplexMPFR` already implements `Ring`/`CommutativeRing`/`Field`; this file
//! adds `IntegralDomain` and the analytic [`ComplexField`] interface (with
//! `Real = RealMPFR`), so the MPC type is first-class alongside the pure-Rust
//! [`BigComplex`](crate::bigcomplex::BigComplex). Purely additive; no new deps.

use crate::mpc::ComplexMPFR;
use rustmath_core::analytic::ComplexField;
use rustmath_core::IntegralDomain;
use rustmath_reals::RealMPFR;

impl IntegralDomain for ComplexMPFR {}

impl ComplexField for ComplexMPFR {
    type Real = RealMPFR;

    fn re(&self) -> RealMPFR {
        self.real_part()
    }
    fn im(&self) -> RealMPFR {
        self.imag_part()
    }
    fn conj(&self) -> Self {
        ComplexMPFR::conj(self)
    }
    fn abs(&self) -> RealMPFR {
        ComplexMPFR::abs(self)
    }
    fn arg(&self) -> RealMPFR {
        ComplexMPFR::arg(self)
    }
    fn from_real_imag(re: RealMPFR, im: RealMPFR) -> Self {
        ComplexMPFR::with_val_reals(re, im)
    }
    fn i(precision: u64) -> Self {
        ComplexMPFR::i_with_prec(precision as u32)
    }
    fn precision(&self) -> u64 {
        ComplexMPFR::precision(self) as u64
    }
    fn sqrt(&self) -> Self {
        ComplexMPFR::sqrt(self)
    }
    fn exp(&self) -> Self {
        ComplexMPFR::exp(self)
    }
    fn ln(&self) -> Self {
        ComplexMPFR::ln(self)
    }
    fn sin(&self) -> Self {
        ComplexMPFR::sin(self)
    }
    fn cos(&self) -> Self {
        ComplexMPFR::cos(self)
    }

    // ---- Wave 0 contract delta ----------------------------------------------

    fn with_precision(&self, precision: u64) -> Self {
        let p = precision.min(u32::MAX as u64) as u32;
        ComplexMPFR::with_val_reals(
            RealMPFR::with_precision(&self.real_part(), p),
            RealMPFR::with_precision(&self.imag_part(), p),
        )
    }
    /// Native MPC pow (overrides the derived principal-branch default): full
    /// MPFR special-value model (`NaN`/`±∞` propagate IEEE-style).
    fn pow(&self, exponent: &Self) -> Self {
        ComplexMPFR::pow(self, exponent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_core::analytic::RealField;

    #[test]
    fn test_complexmpfr_is_complexfield() {
        let z = ComplexMPFR::with_val(200, (3.0, 4.0));
        assert!((RealField::to_f64(&ComplexField::abs(&z)) - 5.0).abs() < 1e-30);
        assert_eq!(ComplexField::precision(&z), 200);
        // i^2 = -1
        let i = <ComplexMPFR as ComplexField>::i(200);
        let m1 = i.clone() * i;
        assert!(RealField::to_f64(&m1.re()).abs() < 1e-40 || (RealField::to_f64(&m1.re()) + 1.0).abs() < 1e-30);
        assert!((RealField::to_f64(&m1.re()) + 1.0).abs() < 1e-30);
    }

    #[test]
    fn test_complexmpfr_contract_delta() {
        // with_precision narrows and widens; the value survives losslessly
        // when it fits (regression guard for the with_val_reals f64 round-trip
        // bug: 2^-100 is invisible at 53 bits).
        let tiny = ComplexMPFR::one_with_prec(300)
            / ComplexMPFR::with_val(300, (2.0, 0.0)).powi(100);
        let z = ComplexMPFR::one_with_prec(300) + tiny;
        let w = ComplexField::with_precision(&z, 200);
        assert_eq!(ComplexField::precision(&w), 200);
        let back = w - ComplexMPFR::one_with_prec(300);
        assert!(
            !RealField::to_f64(&back.re()).is_nan() && RealField::to_f64(&back.re()) > 0.0,
            "1 + 2^-100 must stay distinguishable from 1 at 200 bits"
        );
        // native pow override: (3+4i)^2 = -7 + 24i
        let z = ComplexMPFR::with_val(150, (3.0, 4.0));
        let two = ComplexMPFR::with_val(150, (2.0, 0.0));
        let sq = ComplexField::pow(&z, &two);
        assert!((RealField::to_f64(&sq.re()) + 7.0).abs() < 1e-30);
        assert!((RealField::to_f64(&sq.im()) - 24.0).abs() < 1e-30);
    }

    #[test]
    fn test_with_val_reals_lossless() {
        // Regression: with_val_reals used to round-trip through f64.
        let one = RealMPFR::with_val(300, 1.0);
        let tiny = RealMPFR::with_val(300, 2.0).pow(&RealMPFR::with_val(300, -100.0));
        let re = one + tiny;
        let z = ComplexMPFR::with_val_reals(re.clone(), RealMPFR::with_val(300, 0.0));
        assert_eq!(z.real_part(), re, "with_val_reals must be lossless");
    }

    #[test]
    fn test_complexmpfr_generic_conj() {
        fn norm_via_conj<C: ComplexField>(z: &C) -> C::Real {
            (z.clone() * z.conj()).re()
        }
        let z = ComplexMPFR::with_val(150, (3.0, 4.0));
        // z * conj(z) = |z|^2 = 25 (real)
        assert!((RealField::to_f64(&norm_via_conj(&z)) - 25.0).abs() < 1e-25);
    }
}
