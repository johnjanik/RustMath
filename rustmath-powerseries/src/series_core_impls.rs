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

use crate::precision::DEFAULT_PRECISION;
use crate::series::PowerSeries;
use rustmath_core::{CommutativeRing, IntegralDomain, Ring};

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
