//! The PID `ℤ` as a Dedekind context: `K = ℚ`, and every nonzero fractional
//! ideal is `qℤ` for a unique rational `q > 0`.
//!
//! This is the reference instantiation: the generic pseudo-matrix algorithms
//! specialize over `ℤ` to classical Hermite/Smith computations, which is what
//! the cross-checks against `rustmath-matrix` in [`crate::dedekind::pseudo`]
//! exploit.

use super::{DedekindContext, DedekindError, Principality};
use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;

/// A nonzero fractional ideal of `ℤ`, canonically represented by its positive
/// rational generator.
#[derive(Debug, Clone, PartialEq)]
pub struct ZIdeal(Rational);

impl ZIdeal {
    /// The ideal `qℤ`; `None` iff `q = 0`.
    pub fn new(q: Rational) -> Option<Self> {
        if q.is_zero() {
            None
        } else {
            Some(ZIdeal(q.abs()))
        }
    }

    /// The ideal `nℤ`; `None` iff `n = 0`.
    pub fn from_int(n: i64) -> Option<Self> {
        Self::new(Rational::from_i64(n))
    }

    /// The canonical (positive) generator.
    pub fn generator(&self) -> &Rational {
        &self.0
    }
}

/// The Dedekind context for `ℤ` (unit struct — all data is in the algorithms).
#[derive(Debug, Clone, Copy, Default)]
pub struct ZDedekind;

impl DedekindContext for ZDedekind {
    type Elem = Rational;
    type Ideal = ZIdeal;

    fn zero(&self) -> Rational {
        Rational::zero()
    }
    fn one(&self) -> Rational {
        Rational::one()
    }
    fn add(&self, a: &Rational, b: &Rational) -> Rational {
        a.clone() + b.clone()
    }
    fn neg(&self, a: &Rational) -> Rational {
        -a.clone()
    }
    fn mul(&self, a: &Rational, b: &Rational) -> Rational {
        a.clone() * b.clone()
    }
    fn inv(&self, a: &Rational) -> Option<Rational> {
        a.reciprocal().ok()
    }

    fn unit_ideal(&self) -> ZIdeal {
        ZIdeal(Rational::one())
    }

    fn ideal_mul(&self, a: &ZIdeal, b: &ZIdeal) -> ZIdeal {
        ZIdeal(a.0.clone() * b.0.clone())
    }

    /// `(a/b) + (c/d) = gcd(ad, cb)/(bd)` for lowest-terms representatives.
    fn ideal_add(&self, a: &ZIdeal, b: &ZIdeal) -> ZIdeal {
        let (an, ad) = (a.0.numerator().clone(), a.0.denominator().clone());
        let (bn, bd) = (b.0.numerator().clone(), b.0.denominator().clone());
        let num = (an * bd.clone()).gcd(&(bn * ad.clone()));
        ZIdeal(
            Rational::new(num, ad * bd).expect("nonzero denominator in ideal gcd"),
        )
    }

    fn ideal_inv(&self, a: &ZIdeal) -> ZIdeal {
        ZIdeal(a.0.reciprocal().expect("nonzero ideal generator"))
    }

    fn principal_ideal(&self, x: &Rational) -> Option<ZIdeal> {
        ZIdeal::new(x.clone())
    }

    fn ideal_contains(&self, a: &ZIdeal, x: &Rational) -> bool {
        (x.clone() / a.0.clone()).is_integer()
    }

    fn ideal_subset(&self, a: &ZIdeal, b: &ZIdeal) -> bool {
        (a.0.clone() / b.0.clone()).is_integer()
    }

    fn idempotents(
        &self,
        c1: &ZIdeal,
        c2: &ZIdeal,
    ) -> Result<(Rational, Rational), DedekindError> {
        if !c1.0.is_integer() || !c2.0.is_integer() {
            return Err(DedekindError::NotCoprime(format!(
                "idempotents need integral ideals, got ({}) and ({})",
                c1.0, c2.0
            )));
        }
        let a: Integer = c1.0.numerator().clone();
        let b: Integer = c2.0.numerator().clone();
        let (g, s, t) = a.extended_gcd(&b);
        if !g.is_one() {
            return Err(DedekindError::NotCoprime(format!(
                "({}) + ({}) = ({}) ≠ (1)",
                a, b, g
            )));
        }
        // u = s·a ∈ (a), v = t·b ∈ (b), u + v = g = 1
        let u = Rational::from_integer(s * a);
        let v = Rational::from_integer(t * b);
        debug_assert_eq!(u.clone() + v.clone(), Rational::one());
        Ok((u, v))
    }

    /// Every ideal of a PID is principal; the canonical generator is always a
    /// certificate, so this never returns `Unresolved`.
    fn principal_generator(&self, a: &ZIdeal) -> Principality<Rational> {
        Principality::Principal(a.0.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ideal_canonical_and_eq() {
        let a = ZIdeal::new(Rational::new(-3, 2).unwrap()).unwrap();
        let b = ZIdeal::new(Rational::new(3, 2).unwrap()).unwrap();
        assert_eq!(a, b);
        assert!(ZIdeal::new(Rational::zero()).is_none());
    }

    #[test]
    fn fractional_gcd() {
        let ctx = ZDedekind;
        // (3/4) + (5/6): common denominator 12 → gcd(9, 10)/12 = 1/12
        let a = ZIdeal::new(Rational::new(3, 4).unwrap()).unwrap();
        let b = ZIdeal::new(Rational::new(5, 6).unwrap()).unwrap();
        assert_eq!(
            ctx.ideal_add(&a, &b),
            ZIdeal::new(Rational::new(1, 12).unwrap()).unwrap()
        );
    }

    #[test]
    fn idempotents_split() {
        let ctx = ZDedekind;
        let a = ZIdeal::from_int(4).unwrap();
        let b = ZIdeal::from_int(9).unwrap();
        let (u, v) = ctx.idempotents(&a, &b).unwrap();
        assert!(ctx.ideal_contains(&a, &u));
        assert!(ctx.ideal_contains(&b, &v));
        assert_eq!(u + v, Rational::one());
        // non-coprime pair must error
        let c = ZIdeal::from_int(6).unwrap();
        assert!(matches!(
            ctx.idempotents(&a, &c),
            Err(DedekindError::NotCoprime(_))
        ));
        // non-integral pair must error
        let half = ZIdeal::new(Rational::new(1, 2).unwrap()).unwrap();
        assert!(matches!(
            ctx.idempotents(&half, &b),
            Err(DedekindError::NotCoprime(_))
        ));
    }

    #[test]
    fn membership_and_subset() {
        let ctx = ZDedekind;
        let a = ZIdeal::new(Rational::new(3, 2).unwrap()).unwrap();
        assert!(ctx.ideal_contains(&a, &Rational::from_i64(3)));
        assert!(ctx.ideal_contains(&a, &Rational::new(-9, 2).unwrap()));
        assert!(!ctx.ideal_contains(&a, &Rational::from_i64(1)));
        assert!(ctx.ideal_contains(&a, &Rational::zero()));
        let b = ZIdeal::from_int(3).unwrap();
        assert!(ctx.ideal_subset(&b, &a)); // 3ℤ ⊆ (3/2)ℤ
        assert!(!ctx.ideal_subset(&a, &b));
        assert!(!ctx.ideal_is_integral(&a));
        assert!(ctx.ideal_is_integral(&b));
    }
}
