//! Const-generic prime field GF(p) with a working `rustmath_core::Field` impl.
//!
//! The existing finite-field element types in `rustmath_finitefields` carry a
//! runtime modulus, so their static `Ring::zero()` / `Ring::one()` cannot be
//! implemented (they panic). That makes them unusable as the coefficient
//! field of a generic `UnivariatePolynomial<K>`. `GFp<const P: u64>` bakes
//! the modulus into the type, giving an honest `Field` implementation that
//! composes with every generic algorithm in the workspace.
//!
//! Primality of `P` is NOT checkable at compile time here; `P >= 2` is
//! enforced at compile time, and [`GFp::modulus_is_prime`] provides the
//! runtime check. For non-prime `P`, `inverse()` of a zero divisor returns an
//! honest `Err` rather than a wrong answer.

use rustmath_core::{CommutativeRing, EuclideanDomain, Field, IntegralDomain, MathError, Result, Ring};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// An element of GF(p) for the compile-time modulus `P`.
///
/// Invariant: `value < P`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GFp<const P: u64> {
    value: u64,
}

impl<const P: u64> GFp<P> {
    /// Compile-time gate: the modulus must be at least 2.
    const MODULUS_OK: () = assert!(P >= 2, "GFp modulus must be >= 2");

    /// Create an element from a signed integer, reducing mod `P`.
    pub fn new(v: i64) -> Self {
        #[allow(clippy::let_unit_value)]
        let _ = Self::MODULUS_OK;
        // Reduce into [0, P). P >= 2 fits in i128 comfortably.
        let p = P as i128;
        let mut r = (v as i128) % p;
        if r < 0 {
            r += p;
        }
        GFp { value: r as u64 }
    }

    /// Create an element from an unsigned integer, reducing mod `P`.
    pub fn from_u64(v: u64) -> Self {
        #[allow(clippy::let_unit_value)]
        let _ = Self::MODULUS_OK;
        GFp { value: v % P }
    }

    /// The canonical representative in `[0, P)`.
    pub fn value(&self) -> u64 {
        self.value
    }

    /// The modulus `P`.
    pub fn modulus() -> u64 {
        P
    }

    /// Runtime primality check of the modulus (trial division).
    ///
    /// `GFp<P>` is a field if and only if this returns `true`.
    pub fn modulus_is_prime() -> bool {
        if P < 2 {
            return false;
        }
        if P % 2 == 0 {
            return P == 2;
        }
        let mut d: u64 = 3;
        while (d as u128) * (d as u128) <= P as u128 {
            if P % d == 0 {
                return false;
            }
            d += 2;
        }
        true
    }
}

impl<const P: u64> fmt::Display for GFp<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl<const P: u64> Add for GFp<P> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        // u128 arithmetic: safe for any P <= u64::MAX.
        let s = (self.value as u128 + rhs.value as u128) % P as u128;
        GFp { value: s as u64 }
    }
}

impl<const P: u64> Sub for GFp<P> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl<const P: u64> Mul for GFp<P> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let m = (self.value as u128 * rhs.value as u128) % P as u128;
        GFp { value: m as u64 }
    }
}

impl<const P: u64> Neg for GFp<P> {
    type Output = Self;
    fn neg(self) -> Self {
        if self.value == 0 {
            self
        } else {
            GFp { value: P - self.value }
        }
    }
}

impl<const P: u64> Div for GFp<P> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        self * rhs
            .inverse()
            .expect("division by a non-invertible element of GFp")
    }
}

impl<const P: u64> Ring for GFp<P> {
    fn zero() -> Self {
        #[allow(clippy::let_unit_value)]
        let _ = Self::MODULUS_OK;
        GFp { value: 0 }
    }

    fn one() -> Self {
        #[allow(clippy::let_unit_value)]
        let _ = Self::MODULUS_OK;
        GFp { value: 1 }
    }

    fn is_zero(&self) -> bool {
        self.value == 0
    }

    fn is_one(&self) -> bool {
        self.value == 1
    }
}

impl<const P: u64> CommutativeRing for GFp<P> {}

// Honest caveat: Z/PZ is an integral domain (indeed a field) only for prime
// P. `P >= 2` is enforced at compile time; primality is the caller's
// obligation, checkable via `modulus_is_prime()`. `inverse()` never returns a
// wrong value for composite P: it returns an Err for zero divisors.
impl<const P: u64> IntegralDomain for GFp<P> {}

impl<const P: u64> EuclideanDomain for GFp<P> {
    fn norm(&self) -> u64 {
        if self.is_zero() {
            0
        } else {
            1
        }
    }

    fn div_rem(&self, other: &Self) -> Result<(Self, Self)> {
        if other.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        Ok((*self * other.inverse()?, Self::zero()))
    }
}

impl<const P: u64> Field for GFp<P> {
    fn inverse(&self) -> Result<Self> {
        if self.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        // Extended Euclid over i128; P <= u64::MAX fits.
        let (mut old_r, mut r) = (self.value as i128, P as i128);
        let (mut old_s, mut s) = (1i128, 0i128);
        while r != 0 {
            let q = old_r / r;
            (old_r, r) = (r, old_r - q * r);
            (old_s, s) = (s, old_s - q * s);
        }
        if old_r != 1 {
            // Zero divisor: only possible when P is composite.
            return Err(MathError::InvalidArgument(format!(
                "{} is not invertible mod {} (gcd = {}); GFp modulus must be prime",
                self.value, P, old_r
            )));
        }
        let p = P as i128;
        let mut inv = old_s % p;
        if inv < 0 {
            inv += p;
        }
        Ok(GFp { value: inv as u64 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gfp_arithmetic_mod5() {
        // Verified by hand: 2+4=6=1, 3*4=12=2, -2=3, 3-4=-1=4 (mod 5).
        let two = GFp::<5>::new(2);
        let three = GFp::<5>::new(3);
        let four = GFp::<5>::new(4);
        assert_eq!((two + four).value(), 1);
        assert_eq!((three * four).value(), 2);
        assert_eq!((-two).value(), 3);
        assert_eq!((three - four).value(), 4);
    }

    #[test]
    fn test_gfp_negative_reduction() {
        // -7 mod 5 = 3.
        assert_eq!(GFp::<5>::new(-7).value(), 3);
        assert_eq!(GFp::<5>::new(-5).value(), 0);
        assert_eq!(GFp::<5>::from_u64(12).value(), 2);
    }

    #[test]
    fn test_gfp_inverse_all_nonzero_mod7() {
        // Every nonzero element of GF(7) has an inverse with a * a^{-1} = 1.
        for a in 1..7i64 {
            let x = GFp::<7>::new(a);
            let inv = x.inverse().unwrap();
            assert!((x * inv).is_one(), "a={}", a);
        }
        assert!(GFp::<7>::new(0).inverse().is_err());
    }

    #[test]
    fn test_gfp_inverse_mod5_specific() {
        // 3^{-1} = 2 mod 5 since 3*2 = 6 = 1.
        assert_eq!(GFp::<5>::new(3).inverse().unwrap().value(), 2);
    }

    #[test]
    fn test_gfp_fermat_little_theorem_mod5() {
        // a^4 = 1 for all nonzero a in GF(5).
        for a in 1..5i64 {
            assert!(GFp::<5>::new(a).pow(4).is_one());
        }
    }

    #[test]
    fn test_gfp_ring_statics() {
        assert!(GFp::<5>::zero().is_zero());
        assert!(GFp::<5>::one().is_one());
        assert_eq!(GFp::<5>::modulus(), 5);
    }

    #[test]
    fn test_modulus_is_prime() {
        assert!(GFp::<2>::modulus_is_prime());
        assert!(GFp::<5>::modulus_is_prime());
        assert!(GFp::<7>::modulus_is_prime());
        assert!(!GFp::<6>::modulus_is_prime());
        assert!(!GFp::<9>::modulus_is_prime());
    }

    #[test]
    fn test_gfp_composite_modulus_honest_err() {
        // 2 is a zero divisor mod 6: inverse must be an Err, never a value.
        assert!(GFp::<6>::new(2).inverse().is_err());
        // 5 is coprime to 6 so it does have an inverse (unit group of Z/6Z).
        assert!((GFp::<6>::new(5) * GFp::<6>::new(5).inverse().unwrap()).is_one());
    }

    #[test]
    fn test_gfp_div_rem_field_style() {
        let a = GFp::<5>::new(4);
        let b = GFp::<5>::new(3);
        let (q, r) = a.div_rem(&b).unwrap();
        assert!(r.is_zero());
        assert_eq!(q * b, a);
    }
}
