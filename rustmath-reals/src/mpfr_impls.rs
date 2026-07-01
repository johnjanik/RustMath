//! Wave 0 (MAGMA port): wire the rug/MPFR-backed [`RealMPFR`](crate::RealMPFR)
//! into the new `rustmath-core` trait vocabulary.
//!
//! MAGMA source: Handbook chapter 25 (real fields).
//!
//! `RealMPFR` already implements `Ring`/`CommutativeRing`/`Field`; this file
//! adds the *newly introduced* markers and analytic interface so the MPFR type
//! is a first-class [`RealField`] alongside the pure-Rust
//! [`BigFloat`](crate::bigfloat::BigFloat). Purely additive; no new deps
//! (`rug` is already a dependency of this crate).

use crate::mpfr::RealMPFR;
use rustmath_core::analytic::RealField;
use rustmath_core::ordering::{OrderedField, OrderedRing};
use rustmath_core::IntegralDomain;
use std::cmp::Ordering;

impl IntegralDomain for RealMPFR {}

impl PartialOrd for RealMPFR {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Exact sign of the (full-precision) difference.
        match (self.clone() - other.clone()).signum() {
            s if s < 0 => Some(Ordering::Less),
            s if s > 0 => Some(Ordering::Greater),
            _ => Some(Ordering::Equal),
        }
    }
}

impl OrderedRing for RealMPFR {
    fn sign(&self) -> i32 {
        self.signum()
    }
    fn abs(&self) -> Self {
        RealMPFR::abs(self)
    }
}

impl OrderedField for RealMPFR {}

impl RealField for RealMPFR {
    fn precision(&self) -> u64 {
        RealMPFR::precision(self) as u64
    }
    fn from_f64(x: f64, precision: u64) -> Self {
        RealMPFR::with_val(precision as u32, x)
    }
    fn to_f64(&self) -> f64 {
        RealMPFR::to_f64(self)
    }
    fn pi(precision: u64) -> Self {
        RealMPFR::pi(precision as u32)
    }
    fn e(precision: u64) -> Self {
        RealMPFR::e(precision as u32)
    }
    fn sqrt(&self) -> Self {
        RealMPFR::sqrt(self)
    }
    fn exp(&self) -> Self {
        RealMPFR::exp(self)
    }
    fn ln(&self) -> Self {
        RealMPFR::ln(self)
    }
    fn sin(&self) -> Self {
        RealMPFR::sin(self)
    }
    fn cos(&self) -> Self {
        RealMPFR::cos(self)
    }
    fn atan(&self) -> Self {
        RealMPFR::atan(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Generic over the *core* RealField trait: proves RealMPFR is first-class.
    fn generic_hypot<R: RealField>(a: &R, b: &R) -> R {
        (a.clone() * a.clone() + b.clone() * b.clone()).sqrt()
    }

    #[test]
    fn test_realmpfr_is_realfield() {
        let a = RealMPFR::with_val(200, 3.0);
        let b = RealMPFR::with_val(200, 4.0);
        let h = generic_hypot(&a, &b);
        assert!((RealField::to_f64(&h) - 5.0).abs() < 1e-30);
        assert_eq!(RealField::precision(&a), 200);
    }

    #[test]
    fn test_realmpfr_ordering() {
        let a = RealMPFR::with_val(100, 1.0);
        let b = RealMPFR::with_val(100, 2.0);
        assert!(a < b);
        assert_eq!(OrderedRing::sign(&(a - b)), -1);
    }

    #[test]
    fn test_realmpfr_pi() {
        let pi = <RealMPFR as RealField>::pi(200);
        assert!((RealField::to_f64(&pi) - std::f64::consts::PI).abs() < 1e-14);
    }
}
