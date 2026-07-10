//! Divisibility / local-ring markers for **non-domain** rings.
//!
//! MAGMA source: Handbook chapter 48 (Galois rings `GR(p^a, d)`).
//!
//! `rustmath-rings` already provides the domain-restricted markers
//! `NoetherianRing`, `DedekindDomain` and `PrincipalIdealDomain` (all of which
//! sit above [`IntegralDomain`](crate::IntegralDomain)). Those cannot describe
//! rings that are **commutative but not domains** — most importantly the Galois
//! rings `GR(p^a, d) = Z_{p^a}[x]/(f)` (and `Z/p^a Z`), which for `a > 1` have
//! zero divisors yet are local principal-ideal rings (finite chain rings).
//!
//! This module adds exactly the markers that are genuinely missing, all sitting
//! directly on [`CommutativeRing`] rather than on `IntegralDomain`. It is purely
//! additive and does **not** duplicate the `rustmath-rings` domain markers.

use crate::CommutativeRing;

/// A commutative **local** ring: it has a unique maximal ideal `m`, so every
/// element is either a unit or lies in `m`.
///
/// Unlike `rustmath-rings::PrincipalIdealDomain`, this does *not* require the
/// ring to be a domain, so Galois rings and `Z/p^a Z` qualify.
pub trait LocalRing: CommutativeRing {
    /// Whether this element is a unit (equivalently, `element ∉ m`).
    fn is_unit(&self) -> bool;

    /// Whether this element lies in the maximal ideal (a non-unit).
    fn is_in_maximal_ideal(&self) -> bool {
        !self.is_unit()
    }
}

/// A (possibly non-domain) **principal ideal ring**: every ideal is principal.
///
/// This is the non-domain analogue of `rustmath-rings::PrincipalIdealDomain`.
/// Marker only.
pub trait PrincipalIdealRing: CommutativeRing {}

/// A **finite chain ring**: a local principal ideal ring whose ideals form a
/// single chain `R ⊋ m ⊋ m² ⊋ … ⊋ m^n = (0)`.
///
/// Examples: `Z/p^k Z` and the Galois rings `GR(p^a, d)`. Every non-zero element
/// factors uniquely as `unit · π^k` where `π` generates `m` and
/// `0 <= k < n` = the nilpotency index.
pub trait FiniteChainRing: LocalRing + PrincipalIdealRing {
    /// The nilpotency index `n` (the length of the ideal chain, i.e. the
    /// smallest `n` with `m^n = 0`).
    fn nilpotency_index(&self) -> u64;

    /// The chain valuation `v(self)`: the largest `k` with `self ∈ m^k`.
    /// By convention `v(0) = n` (the nilpotency index).
    fn chain_valuation(&self) -> u64;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Ring;
    use std::fmt;
    use std::ops::{Add, Mul, Neg, Sub};

    /// `Z/4Z`: the smallest finite chain ring that is not a domain
    /// (`2 · 2 = 0`). Maximal ideal `m = (2)`, nilpotency index `2`.
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Z4(u8);

    impl Z4 {
        fn new(n: i64) -> Self {
            Z4(n.rem_euclid(4) as u8)
        }
    }
    impl fmt::Display for Z4 {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl Add for Z4 {
        type Output = Self;
        fn add(self, o: Self) -> Self {
            Z4((self.0 + o.0) % 4)
        }
    }
    impl Sub for Z4 {
        type Output = Self;
        fn sub(self, o: Self) -> Self {
            Z4((self.0 + 4 - o.0) % 4)
        }
    }
    impl Mul for Z4 {
        type Output = Self;
        fn mul(self, o: Self) -> Self {
            Z4((self.0 * o.0) % 4)
        }
    }
    impl Neg for Z4 {
        type Output = Self;
        fn neg(self) -> Self {
            Z4((4 - self.0) % 4)
        }
    }
    impl Ring for Z4 {
        fn zero() -> Self {
            Z4(0)
        }
        fn one() -> Self {
            Z4(1)
        }
        fn is_zero(&self) -> bool {
            self.0 == 0
        }
        fn is_one(&self) -> bool {
            self.0 == 1
        }
    }
    impl CommutativeRing for Z4 {}
    impl LocalRing for Z4 {
        fn is_unit(&self) -> bool {
            self.0 % 2 == 1 // units of Z/4Z are 1 and 3
        }
    }
    impl PrincipalIdealRing for Z4 {}
    impl FiniteChainRing for Z4 {
        fn nilpotency_index(&self) -> u64 {
            2
        }
        fn chain_valuation(&self) -> u64 {
            match self.0 {
                0 => 2,       // v(0) = n
                2 => 1,       // 2 ∈ m
                _ => 0,       // 1, 3 are units
            }
        }
    }

    #[test]
    fn test_z4_is_not_a_domain() {
        // The whole point: a zero divisor exists.
        assert!((Z4::new(2) * Z4::new(2)).is_zero());
    }

    #[test]
    fn test_local_ring_units() {
        assert!(Z4::new(1).is_unit());
        assert!(Z4::new(3).is_unit());
        assert!(!Z4::new(2).is_unit());
        assert!(Z4::new(2).is_in_maximal_ideal());
        assert!(!Z4::new(0).is_unit());
    }

    #[test]
    fn test_finite_chain_valuation() {
        assert_eq!(Z4::new(1).chain_valuation(), 0);
        assert_eq!(Z4::new(2).chain_valuation(), 1);
        assert_eq!(Z4::new(0).chain_valuation(), 2);
        assert_eq!(Z4::new(3).nilpotency_index(), 2);
    }
}
