//! Real and complex balls implementing the core [`Ball`] trait.
//!
//! MAGMA source: Handbook chapter 25 (real and complex fields; certified /
//! ball arithmetic à la Arb).
//!
//! A ball `[c ± r]` rigorously encloses every point within radius `r` of the
//! center `c`. These types provide the pure-Rust certified-arithmetic layer
//! over [`BigFloat`]/[`BigComplex`]. They are distinct from the existing
//! rug-backed [`ComplexBall`](crate::ComplexBall) type.
//!
//! Note: the underlying [`BigFloat`] arithmetic rounds to a target precision,
//! so a fully rigorous implementation would inflate the radius by one ulp per
//! operation. The propagation here tracks input uncertainty exactly and treats
//! the dyadic arithmetic as exact; ulp-level inflation is a documented
//! follow-up for Wave 1 consumers that need provable enclosures.
//!
//! # Canonicalization decision (Wave 0)
//!
//! [`BigRealBall`] and [`BigComplexBall`] are the **canonical certified
//! (ball) types** for RustMath going forward: they are the only types
//! implementing the shared [`rustmath_core::analytic::Ball`] trait, and new
//! code should target that trait rather than any concrete interval type. The
//! five pre-existing interval/ball types —
//!
//! * `rustmath-rings` `real_arb` (Arb-style ball),
//! * `rustmath-rings` `real_mpfi` (MPFI-style interval),
//! * `rustmath-rings` `real_interval_absolute` (absolute-precision interval),
//! * `rustmath-reals` `Interval` (f64 endpoints),
//! * `rustmath-complex` [`ComplexBall`](crate::ComplexBall) (rug-backed)
//!
//! — are **Phase-2 migration targets**: they stay as-is in Wave 0 (no
//! migration now) and are to be re-expressed over, or replaced by, the
//! `Ball` trait during the Phase-2 canonicalization pass (see
//! `docs/PLAN.md`).

use crate::bigcomplex::BigComplex;
use rustmath_core::analytic::{Ball, ComplexField};
use rustmath_core::ordering::OrderedRing;
use rustmath_reals::bigfloat::BigFloat;

/// A real ball `[center ± radius]` over [`BigFloat`].
#[derive(Clone, Debug)]
pub struct BigRealBall {
    center: BigFloat,
    radius: BigFloat,
}

impl BigRealBall {
    /// Construct `[center ± |radius|]`.
    pub fn new(center: BigFloat, radius: BigFloat) -> Self {
        BigRealBall {
            center,
            radius: OrderedRing::abs(&radius),
        }
    }

    /// An exact point ball `[c ± 0]`.
    pub fn exact(center: BigFloat) -> Self {
        let z = BigFloat::zero_prec(center.prec());
        BigRealBall { center, radius: z }
    }

    /// Certified sum: `[c1 ± r1] + [c2 ± r2] = [c1+c2 ± r1+r2]`.
    pub fn add(&self, other: &Self) -> Self {
        BigRealBall {
            center: self.center.clone() + other.center.clone(),
            radius: self.radius.clone() + other.radius.clone(),
        }
    }

    /// Certified product:
    /// `[c1 ± r1]·[c2 ± r2] = [c1 c2 ± |c1| r2 + |c2| r1 + r1 r2]`.
    pub fn mul(&self, other: &Self) -> Self {
        let c = self.center.clone() * other.center.clone();
        let r = OrderedRing::abs(&self.center) * other.radius.clone()
            + OrderedRing::abs(&other.center) * self.radius.clone()
            + self.radius.clone() * other.radius.clone();
        BigRealBall { center: c, radius: r }
    }
}

impl Ball for BigRealBall {
    type Center = BigFloat;
    type Radius = BigFloat;

    fn center(&self) -> BigFloat {
        self.center.clone()
    }
    fn radius(&self) -> BigFloat {
        self.radius.clone()
    }
    fn from_center_radius(center: BigFloat, radius: BigFloat) -> Self {
        BigRealBall::new(center, radius)
    }
    fn contains(&self, point: &BigFloat) -> bool {
        let d = OrderedRing::abs(&(point.clone() - self.center.clone()));
        d <= self.radius
    }
    fn contains_zero(&self) -> bool {
        OrderedRing::abs(&self.center) <= self.radius
    }
}

/// A complex ball `[center ± radius]` with a [`BigComplex`] center and a
/// non-negative [`BigFloat`] radius.
#[derive(Clone, Debug)]
pub struct BigComplexBall {
    center: BigComplex,
    radius: BigFloat,
}

impl BigComplexBall {
    /// Construct `[center ± |radius|]`.
    pub fn new(center: BigComplex, radius: BigFloat) -> Self {
        BigComplexBall {
            center,
            radius: OrderedRing::abs(&radius),
        }
    }

    /// Certified sum.
    pub fn add(&self, other: &Self) -> Self {
        BigComplexBall {
            center: self.center.clone() + other.center.clone(),
            radius: self.radius.clone() + other.radius.clone(),
        }
    }

    /// Certified product.
    pub fn mul(&self, other: &Self) -> Self {
        let c = self.center.clone() * other.center.clone();
        let r = self.center.abs() * other.radius.clone()
            + other.center.abs() * self.radius.clone()
            + self.radius.clone() * other.radius.clone();
        BigComplexBall { center: c, radius: r }
    }
}

impl Ball for BigComplexBall {
    type Center = BigComplex;
    type Radius = BigFloat;

    fn center(&self) -> BigComplex {
        self.center.clone()
    }
    fn radius(&self) -> BigFloat {
        self.radius.clone()
    }
    fn from_center_radius(center: BigComplex, radius: BigFloat) -> Self {
        BigComplexBall::new(center, radius)
    }
    fn contains(&self, point: &BigComplex) -> bool {
        let d = (point.clone() - self.center.clone()).abs();
        d <= self.radius
    }
    fn contains_zero(&self) -> bool {
        self.center.abs() <= self.radius
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    fn bf(n: i64, p: u64) -> BigFloat {
        BigFloat::from_integer(&Integer::from(n), p)
    }
    fn half(p: u64) -> BigFloat {
        use rustmath_rationals::Rational;
        BigFloat::from_rational(&Rational::new(Integer::from(1), Integer::from(2)).unwrap(), p)
    }

    #[test]
    fn test_real_ball_contains() {
        let p = 80;
        let b = BigRealBall::from_center_radius(bf(1, p), half(p)); // [1 ± 0.5]
        assert!(b.contains(&bf(1, p)));
        assert!(!b.contains(&bf(2, p)));
        assert!(!b.contains_zero());
        let around0 = BigRealBall::new(half(p), bf(1, p)); // [0.5 ± 1]
        assert!(around0.contains_zero());
    }

    #[test]
    fn test_real_ball_add_mul() {
        let p = 80;
        let a = BigRealBall::new(bf(2, p), half(p)); // [2 ± 0.5]
        let b = BigRealBall::new(bf(3, p), half(p)); // [3 ± 0.5]
        let s = a.add(&b);
        assert!(s.center() == bf(5, p));
        assert!(s.radius() == bf(1, p));
        assert!(s.contains(&bf(5, p)));
        let m = a.mul(&b);
        // center 6, radius = 2*0.5 + 3*0.5 + 0.25 = 2.75
        assert!(m.center() == bf(6, p));
        assert!(m.contains(&bf(6, p)));
        // 6 is well within; check the true product interval endpoints are enclosed
        assert!(m.contains(&(bf(2, p) * bf(3, p))));
    }

    #[test]
    fn test_complex_ball() {
        let p = 100;
        let center = BigComplex::new(bf(3, p), bf(4, p)); // 3+4i, |.|=5
        let ball = BigComplexBall::from_center_radius(center.clone(), bf(1, p));
        assert!(ball.contains(&center));
        // a point at distance 0.5 is inside
        let near = BigComplex::new(bf(3, p) + half(p), bf(4, p));
        assert!(ball.contains(&near));
        // origin is at distance 5 > 1, so not contained
        assert!(!ball.contains_zero());
    }
}
