//! `rustmath-core` trait adoption for the existing [`PowerSeries`] type.
//!
//! MAGMA source: Handbook Chapter 49 §49.3.3 (ring predicates), §49.4.2
//! (arithmetic), §49.4.3 (equality).  Chapter 49 places power series in the
//! `RngSer`/`RngSerElt` category, i.e. they form a commutative ring (an
//! integral domain when the coefficient ring is a domain).
//!
//! Before this file `PowerSeries<R>` only provided the `std::ops` arithmetic and
//! ad-hoc inherent constructors; it did **not** implement the `rustmath-core`
//! `Ring` tower, so it could not be consumed generically (e.g. as the
//! coefficient ring of a matrix or polynomial).  Closing that gap is the "top
//! integration gap" flagged in `docs/port/build_backlogs.md`.
//!
//! This is a `*_core_impls.rs` file per the worker discipline (trait impls for a
//! pre-existing type live in a new file, not in the type's own module).
//!
//! # Ring predicates (MAGMA Ch. 49 §49.4.4 `IsUnit`, `IsNilpotent`)
//!
//! The hooks `unit_status`, `unit_inverse`, `nilpotent_status`,
//! `multiplicative_order_status` and `static_characteristic` use, for any coefficient ring
//! R with x central (Atiyah and Macdonald, *Introduction to Commutative Algebra*, Ch. 1,
//! Ex. 5 for commutative R; the same argument works in general): f is a unit of R\[\[x\]\]
//! iff f(0) is a unit of R, with inverse g_0 = f_0^-1, g_m = -g_0 Σ_{i=1..m} f_i g_(m-i)
//! (a right inverse, hence the inverse of a unit); the lowest nonzero coefficient a_v of f
//! gives a_v^k as the coefficient of x^(kv) in f^k, so a non-nilpotent a_v makes f
//! non-nilpotent; for commutative R a nilpotent f has nilpotent coefficients (A-M Ch. 1,
//! Ex. 5(ii)).
//!
//! **Weak predicates (a departure from MAGMA).** MAGMA's `eq`, `IsZero`, `IsOne` are strict
//! for series (Ch. 49 §49.4.3-4: identical valuation, precision and coefficients). Here
//! `==`, `is_zero` and `is_one` are the weak versions (MAGMA `IsWeaklyEqual`,
//! `IsWeaklyZero`), as before. The hooks build on them: `O(x^5)` is nilpotent and
//! `1 + O(x^5)` has order 1, statements about the representative. `O(x^0)` (nothing known)
//! decides nothing.
//!
//! **Precondition.** The rules that read a coefficient's `is_zero()` ("the lowest nonzero
//! coefficient") are proofs when that test is exact (Z, Q, Z/m, GF(q), Galois rings,
//! polynomials over those). For a weak coefficient type (the p-adic types, series of
//! series) they hold for the representative only, under the convention that `is_zero` is
//! exact.

use crate::precision::DEFAULT_PRECISION;
use crate::series::PowerSeries;
use rustmath_core::{
    CommutativeRing, IntegralDomain, MathError, MultiplicativeOrder, Result, Ring,
};

/// Equality of truncated series (Chapter 49.1.6 / 49.4.3): two series are equal
/// iff their known coefficients agree up to the smaller of the two precisions
/// (i.e. their difference is `O(x^p)` with `p` the minimum precision).
impl<R: Ring> PartialEq for PowerSeries<R> {
    fn eq(&self, other: &Self) -> bool {
        let p = self.precision().min(other.precision());
        (0..p).all(|i| self.coeff(i) == other.coeff(i))
    }
}

impl<R: Ring> Ring for PowerSeries<R> {
    fn zero() -> Self {
        PowerSeries::zero(DEFAULT_PRECISION)
    }

    fn one() -> Self {
        PowerSeries::constant(R::one(), DEFAULT_PRECISION)
    }

    fn is_zero(&self) -> bool {
        (0..self.precision()).all(|i| self.coeff(i).is_zero())
    }

    fn is_one(&self) -> bool {
        if self.precision() == 0 {
            return false;
        }
        self.coeff(0).is_one() && (1..self.precision()).all(|i| self.coeff(i).is_zero())
    }

    /// `R::static_characteristic()` (R ⊂ R\[\[x\]\]).
    fn static_characteristic() -> Option<u64> {
        R::static_characteristic()
    }

    fn static_is_commutative() -> Option<bool> {
        R::static_is_commutative()
    }

    /// `O(x^0)`: `None` (nothing is known); otherwise `coeff(0).unit_status()`.
    fn unit_status(&self) -> Option<bool> {
        if self.precision() == 0 {
            return None;
        }
        self.coeff(0).unit_status()
    }

    /// The inverse to the same precision, by the recurrence of the module docs (exact
    /// modulo x^p). `O(x^0)`: `Err(NotImplemented)`; a constant term that is a proven
    /// non-unit: `Err(NotInvertible)`; any other error of the constant term's
    /// `unit_inverse` is passed on (undecided).
    fn unit_inverse(&self) -> Result<Self> {
        let p = self.precision();
        if p == 0 {
            return Err(MathError::NotImplemented(
                "PowerSeries::unit_inverse: O(x^0) has no known coefficient".into(),
            ));
        }
        let g0 = self.coeff(0).unit_inverse()?;
        let mut g: Vec<R> = Vec::with_capacity(p);
        g.push(g0.clone());
        for m in 1..p {
            let mut acc: Option<R> = None;
            for i in 1..=m {
                let t = self.coeff(i).clone() * g[m - i].clone();
                acc = Some(match acc {
                    None => t,
                    Some(a) => a + t,
                });
            }
            let sum = acc.expect("m >= 1 terms");
            g.push(-(g0.clone() * sum));
        }
        Ok(PowerSeries::new(g, p))
    }

    /// `O(x^0)`: `None`; weakly zero: `Some(true)` (the convention); the lowest nonzero
    /// known coefficient non-nilpotent: `Some(false)`; for commutative R (P1's
    /// `static_is_commutative`), any known coefficient non-nilpotent: `Some(false)`;
    /// otherwise `None` (never `Some(true)` for a nonzero series: the unknown tail could be
    /// non-nilpotent).
    fn nilpotent_status(&self) -> Option<bool> {
        let p = self.precision();
        if p == 0 {
            return None;
        }
        let Some(lowest) = (0..p).find(|&i| !self.coeff(i).is_zero()) else {
            return Some(true);
        };
        if self.coeff(lowest).nilpotent_status() == Some(false) {
            return Some(false);
        }
        if R::static_is_commutative() == Some(true)
            && (lowest..p).any(|i| self.coeff(i).nilpotent_status() == Some(false))
        {
            return Some(false);
        }
        None
    }

    /// `O(x^0)`: `Undecided`; otherwise the trait's default rules (`1`, `-1`, a proven
    /// non-unit), which now see [`Self::unit_status`].
    fn multiplicative_order_status(&self) -> MultiplicativeOrder {
        if self.precision() == 0 {
            return MultiplicativeOrder::Undecided;
        }
        if self.is_one() {
            MultiplicativeOrder::Finite(1)
        } else if (-self.clone()).is_one() {
            match Self::static_characteristic() {
                Some(c) if c != 1 && c != 2 => MultiplicativeOrder::Finite(2),
                _ => MultiplicativeOrder::Undecided,
            }
        } else if self.unit_status() == Some(false) {
            MultiplicativeOrder::NotAUnit
        } else {
            MultiplicativeOrder::Undecided
        }
    }
}

/// `R[[x]]` is commutative when the coefficient ring `R` is.
impl<R: CommutativeRing> CommutativeRing for PowerSeries<R> {}

/// `R[[x]]` is an integral domain when `R` is (no zero divisors: the product of
/// the lowest-degree terms cannot vanish).
impl<R: IntegralDomain> IntegralDomain for PowerSeries<R> {}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;
    use rustmath_rationals::Rational;

    fn ps(coeffs: &[i64], prec: usize) -> PowerSeries<Integer> {
        PowerSeries::new(coeffs.iter().map(|&c| Integer::from(c)).collect(), prec)
    }

    #[test]
    fn ring_identities() {
        let zero = <PowerSeries<Integer> as Ring>::zero();
        let one = <PowerSeries<Integer> as Ring>::one();
        assert!(zero.is_zero());
        assert!(one.is_one());
        assert!(!zero.is_one());
        assert!(!one.is_zero());

        let f = ps(&[1, 2, 3], 5);
        // f + 0 == f, f * 1 == f (up to min precision)
        assert_eq!(f.clone() + zero.clone(), f);
        assert_eq!(f.clone() * one.clone(), f);
    }

    #[test]
    fn partial_eq_respects_precision() {
        // Same known coefficients, different unknown tails => equal to min prec.
        let a = ps(&[1, 1], 2);
        let b = ps(&[1, 1, 7, 9], 4);
        assert_eq!(a, b); // agree on x^0, x^1; b's x^2,x^3 are beyond a's precision
        let c = ps(&[1, 2], 2);
        assert_ne!(a, c);
    }

    #[test]
    fn generic_ring_consumption() {
        // A function generic over Ring can now take PowerSeries.
        fn sum_of_squares<T: Ring>(x: T) -> T {
            x.clone() * x.clone() + T::one()
        }
        let f = PowerSeries::new(vec![Rational::from_i64(1), Rational::from_i64(1)], 4);
        let g = sum_of_squares(f);
        // (1+x)^2 + 1 = 2 + 2x + x^2
        assert_eq!(g.coeff(0), &Rational::from_i64(2));
        assert_eq!(g.coeff(1), &Rational::from_i64(2));
        assert_eq!(g.coeff(2), &Rational::from_i64(1));
    }

    #[test]
    fn pow_via_trait() {
        let f = PowerSeries::new(vec![Rational::from_i64(1), Rational::from_i64(1)], 6);
        // (1+x)^3 = 1 + 3x + 3x^2 + x^3
        let cube = f.pow(3);
        assert_eq!(cube.coeff(0), &Rational::from_i64(1));
        assert_eq!(cube.coeff(1), &Rational::from_i64(3));
        assert_eq!(cube.coeff(2), &Rational::from_i64(3));
        assert_eq!(cube.coeff(3), &Rational::from_i64(1));
    }
}
