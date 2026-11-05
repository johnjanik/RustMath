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

    // ---- Wave 0 contract delta ----------------------------------------------

    fn with_precision(&self, precision: u64) -> Self {
        RealMPFR::with_precision(self, precision.min(u32::MAX as u64) as u32)
    }
    fn floor(&self) -> Self {
        // Stay in Float (exact at any magnitude, keeps MPFR specials).
        RealMPFR::from_float(self.as_float().clone().floor())
    }
    fn ceil(&self) -> Self {
        RealMPFR::from_float(self.as_float().clone().ceil())
    }
    fn round(&self) -> Self {
        // rug's round is round-to-nearest, ties away from zero — the contract.
        RealMPFR::from_float(self.as_float().clone().round())
    }
    fn atan2(&self, x: &Self) -> Self {
        // native MPFR atan2 (overrides the derived default)
        RealMPFR::atan2(self, x)
    }
    fn to_decimal_string(&self, digits: usize) -> String {
        RealMPFR::to_decimal_string(self, digits)
    }
    fn from_decimal_str(s: &str, precision: u64) -> rustmath_core::Result<Self> {
        RealMPFR::from_decimal_str(s, precision.min(u32::MAX as u64) as u32)
    }
    /// Native MPFR pow (overrides the derived default): full special-value
    /// model — negative base with non-integral exponent yields NaN, `0^-y`
    /// yields `+∞`, etc.
    fn pow(&self, exponent: &Self) -> Self {
        RealMPFR::pow(self, exponent)
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

    #[test]
    fn test_realmpfr_contract_delta() {
        let x = RealMPFR::with_val(200, 2.5);
        // floor/ceil/round with tie-away-from-zero
        assert_eq!(RealField::to_f64(&RealField::floor(&x)), 2.0);
        assert_eq!(RealField::to_f64(&RealField::ceil(&x)), 3.0);
        assert_eq!(RealField::to_f64(&RealField::round(&x)), 3.0);
        let n = RealMPFR::with_val(200, -2.5);
        assert_eq!(RealField::to_f64(&RealField::round(&n)), -3.0);
        // with_precision narrows and widens
        let w = RealField::with_precision(&x, 64);
        assert_eq!(RealField::precision(&w), 64);
        assert_eq!(RealField::to_f64(&w), 2.5);
        // atan2 quadrants (native override)
        let y = RealMPFR::with_val(100, 1.0);
        let mx = RealMPFR::with_val(100, -1.0);
        let a = RealField::atan2(&y, &mx);
        assert!((RealField::to_f64(&a) - 3.0 * std::f64::consts::FRAC_PI_4).abs() < 1e-14);
        // decimal I/O roundtrip
        let pi = <RealMPFR as RealField>::pi(200);
        let s = RealField::to_decimal_string(&pi, 40);
        let back = <RealMPFR as RealField>::from_decimal_str(&s, 200).unwrap();
        let diff = RealField::to_f64(&OrderedRing::abs(&(pi - back)));
        assert!(diff < 1e-38, "roundtrip diff {diff}");
        assert!(<RealMPFR as RealField>::from_decimal_str("zz", 64).is_err());
        // pow (native override)
        let two = RealMPFR::with_val(100, 2.0);
        let ten = RealMPFR::with_val(100, 10.0);
        assert_eq!(RealField::to_f64(&RealField::pow(&two, &ten)), 1024.0);
    }

    #[test]
    fn test_realmpfr_floor_large_value_exact() {
        // Regression: floor/ceil/round used to return a placeholder 0 for
        // values beyond i64 range.
        let big = RealMPFR::with_val(300, 2.0).pow(&RealMPFR::with_val(300, 100.0));
        let f = RealMPFR::floor(&big); // 2^100 exactly
        let expected = rustmath_integers::Integer::from(2).pow(100);
        assert_eq!(f, expected);
    }
}
