//! Local arithmetic: Hilbert symbols `(a,b)_v ∈ {±1}` over `Q`.
//!
//! Ported from `dessin_engine/src/hilbert.rs`
//! (`/home/john/inverse_galois/M23/dessin_engine/src/hilbert.rs`), adapted to
//! RustMath's `rustmath_rationals::Rational` / `rustmath_integers::Integer`
//! foundation types instead of dessin_engine's private `Rat`.
//!
//! `(a,b)_v = 1` iff `z² = a x² + b y²` has a nontrivial `Q_v`-solution. The
//! conic layer reads a genus-0 curve's obstruction as a quaternion class `(a,b)`
//! and asks which places ramify. By Hilbert reciprocity a class ramifies at an
//! even, finite set of places contained in `{∞, 2} ∪ {p ∣ ab}`.

use num_bigint::{BigInt, Sign};
use num_traits::{Signed, ToPrimitive, Zero};
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HilbertError {
    #[error("prime must be 2 or an odd prime")]
    InvalidPrime,
    #[error("zero input not allowed for Hilbert symbol")]
    ZeroInput,
    #[error("integer too large for small-prime modular routine")]
    TooLargeForSmallRoutine,
    #[error("modular inverse does not exist")]
    NoModInverse,
}

/// A place of `Q`: the real place `∞` or a finite place `p`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Place {
    Real,
    Finite(u64),
}

impl std::fmt::Display for Place {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Place::Real => write!(f, "oo"),
            Place::Finite(p) => write!(f, "{p}"),
        }
    }
}

// --- small rational helpers (dessin_engine's `Rat` carried these inherently) ---

pub(crate) fn rat_is_zero(r: &Rational) -> bool {
    r.numerator().is_zero()
}

pub(crate) fn rat_is_negative(r: &Rational) -> bool {
    r.numerator().signum() < 0
}

/// `v_p(num/den) = v_p(num) - v_p(den)`. Undefined at zero; callers reject zero
/// before this (the Hilbert routines check zero inputs first).
pub(crate) fn rat_valuation(r: &Rational, p: u64) -> i64 {
    r.valuation(&Integer::from(p)) as i64
}

/// Unit part of `self` modulo `modulus`, after removing every factor of `p`.
/// Odd Hilbert symbols use `modulus = p`; for `p = 2` use `modulus = 8`.
pub(crate) fn rat_unit_mod(r: &Rational, p: u64, modulus: u64) -> Result<u64, HilbertError> {
    let n = remove_prime_factors(r.numerator().as_bigint().clone(), p);
    let d = remove_prime_factors(r.denominator().as_bigint().clone(), p);

    let n = mod_bigint_nonnegative(&n, modulus);
    let d = mod_bigint_nonnegative(&d, modulus);

    let ni = n.to_i64().ok_or(HilbertError::TooLargeForSmallRoutine)?;
    let di = d.to_i64().ok_or(HilbertError::TooLargeForSmallRoutine)?;

    let inv = mod_inv_i64(di.rem_euclid(modulus as i64), modulus as i64)?;
    Ok((ni.rem_euclid(modulus as i64) * inv).rem_euclid(modulus as i64) as u64)
}

/// Primes dividing numerator or denominator (prototype trial division).
pub(crate) fn rat_support_primes_small(r: &Rational) -> Vec<u64> {
    let mut primes = factor_trial_u64_abs(r.numerator().as_bigint());
    primes.extend(factor_trial_u64_abs(r.denominator().as_bigint()));
    primes.sort_unstable();
    primes.dedup();
    primes
}

/// `(a,b)_place ∈ {+1,-1}`.
pub fn hilbert_symbol(a: &Rational, b: &Rational, place: Place) -> Result<i8, HilbertError> {
    if rat_is_zero(a) || rat_is_zero(b) {
        return Err(HilbertError::ZeroInput);
    }
    match place {
        Place::Real => Ok(if rat_is_negative(a) && rat_is_negative(b) {
            -1
        } else {
            1
        }),
        Place::Finite(2) => hilbert_2(a, b),
        Place::Finite(p) if p % 2 == 1 => hilbert_odd(a, b, p),
        _ => Err(HilbertError::InvalidPrime),
    }
}

fn hilbert_odd(a: &Rational, b: &Rational, p: u64) -> Result<i8, HilbertError> {
    let alpha = rat_valuation(a, p).rem_euclid(2);
    let beta = rat_valuation(b, p).rem_euclid(2);
    let u = rat_unit_mod(a, p, p)?;
    let v = rat_unit_mod(b, p, p)?;

    let mut sign = 1i8;
    // (-1)^{alpha*beta*(p-1)/2}
    if alpha == 1 && beta == 1 && ((p - 1) / 2) % 2 == 1 {
        sign = -sign;
    }
    // (u|p)^beta * (v|p)^alpha
    if beta == 1 {
        sign *= legendre_symbol(u, p)?;
    }
    if alpha == 1 {
        sign *= legendre_symbol(v, p)?;
    }
    Ok(sign)
}

fn hilbert_2(a: &Rational, b: &Rational) -> Result<i8, HilbertError> {
    let alpha = rat_valuation(a, 2).rem_euclid(2) as u64;
    let beta = rat_valuation(b, 2).rem_euclid(2) as u64;
    let u = rat_unit_mod(a, 2, 8)?;
    let v = rat_unit_mod(b, 2, 8)?;

    let exponent = (epsilon_2(u) * epsilon_2(v) + alpha * omega_2(v) + beta * omega_2(u)) % 2;
    Ok(if exponent == 0 { 1 } else { -1 })
}

/// `(u-1)/2 mod 2`, `u` odd mod 8.
fn epsilon_2(u: u64) -> u64 {
    (((u + 7) % 8) / 2) % 2
}

/// `(u^2-1)/8 mod 2`, `u` odd mod 8.
fn omega_2(u: u64) -> u64 {
    ((u * u + 63) / 8) % 2
}

fn legendre_symbol(a: u64, p: u64) -> Result<i8, HilbertError> {
    if p < 3 || p % 2 == 0 {
        return Err(HilbertError::InvalidPrime);
    }
    let a = a % p;
    if a == 0 {
        return Ok(0);
    }
    let r = mod_pow_u64(a, (p - 1) / 2, p);
    if r == 1 {
        Ok(1)
    } else if r == p - 1 {
        Ok(-1)
    } else {
        Err(HilbertError::InvalidPrime)
    }
}

fn mod_pow_u64(base: u64, mut exp: u64, modu: u64) -> u64 {
    let mut acc = 1u128;
    let mut b = (base % modu) as u128;
    let m = modu as u128;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = (acc * b) % m;
        }
        b = (b * b) % m;
        exp >>= 1;
    }
    acc as u64
}

/// The only places where `(a,b)` can ramify: `∞`, `2`, and primes dividing `ab`.
pub fn candidate_places_for_quaternion(a: &Rational, b: &Rational) -> Vec<Place> {
    let mut primes = vec![2u64];
    primes.extend(rat_support_primes_small(a));
    primes.extend(rat_support_primes_small(b));
    primes.sort_unstable();
    primes.dedup();
    let mut places = vec![Place::Real];
    places.extend(primes.into_iter().map(Place::Finite));
    places
}

// --- BigInt helpers (ported from dessin_engine's `rational.rs`) ---

fn remove_prime_factors(mut x: BigInt, p: u64) -> BigInt {
    if x.is_zero() {
        return x;
    }
    let p_big = BigInt::from(p);
    let sign = x.sign();
    x = if sign == Sign::Minus { -x } else { x };
    while (&x % &p_big).is_zero() {
        x /= &p_big;
    }
    if sign == Sign::Minus {
        -x
    } else {
        x
    }
}

fn mod_bigint_nonnegative(x: &BigInt, modulus: u64) -> BigInt {
    let m = BigInt::from(modulus);
    let mut r = x % &m;
    if r.sign() == Sign::Minus {
        r += &m;
    }
    r
}

fn egcd(mut a: i64, mut b: i64) -> (i64, i64, i64) {
    let (mut x0, mut x1) = (1i64, 0i64);
    let (mut y0, mut y1) = (0i64, 1i64);
    while b != 0 {
        let q = a / b;
        (a, b) = (b, a - q * b);
        (x0, x1) = (x1, x0 - q * x1);
        (y0, y1) = (y1, y0 - q * y1);
    }
    (a, x0, y0)
}

fn mod_inv_i64(a: i64, m: i64) -> Result<i64, HilbertError> {
    let (g, x, _) = egcd(a, m);
    if g.abs() != 1 {
        return Err(HilbertError::NoModInverse);
    }
    Ok(x.rem_euclid(m))
}

fn factor_trial_u64_abs(x: &BigInt) -> Vec<u64> {
    // Prototype only. Replace with a real integer-factorization backend.
    let Some(mut n) = x.abs().to_u128() else {
        return vec![];
    };
    let mut out = Vec::new();
    if n == 0 {
        return out;
    }
    if n % 2 == 0 {
        out.push(2);
        while n % 2 == 0 {
            n /= 2;
        }
    }
    let mut p = 3u128;
    while p * p <= n {
        if n % p == 0 {
            out.push(p as u64);
            while n % p == 0 {
                n /= p;
            }
        }
        p += 2;
    }
    if n > 1 {
        out.push(n as u64);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: i64) -> Rational {
        Rational::from_i64(n)
    }

    #[test]
    fn unit_mod_minus_one() {
        assert_eq!(rat_unit_mod(&r(-1), 2, 8).unwrap(), 7);
        assert_eq!(rat_unit_mod(&r(-1), 3, 3).unwrap(), 2);
    }

    #[test]
    fn hamilton_quaternion_ramifies_at_2_and_infinity() {
        let m1 = r(-1);
        let ramified: Vec<Place> = candidate_places_for_quaternion(&m1, &m1)
            .into_iter()
            .filter(|&v| hilbert_symbol(&m1, &m1, v).unwrap() == -1)
            .collect();
        assert!(ramified.contains(&Place::Real));
        assert!(ramified.contains(&Place::Finite(2)));
        assert_eq!(ramified.len(), 2);
    }

    #[test]
    fn split_quaternion_has_no_ramified_candidates() {
        let one = r(1);
        let m1 = r(-1);
        for place in candidate_places_for_quaternion(&one, &m1) {
            assert_eq!(hilbert_symbol(&one, &m1, place).unwrap(), 1);
        }
    }

    #[test]
    fn reciprocity_product_is_one() {
        // For any a,b the product of (a,b)_v over all places is +1.
        let a = Rational::new(6, 35).unwrap();
        let b = r(-10);
        let prod: i8 = candidate_places_for_quaternion(&a, &b)
            .into_iter()
            .map(|v| hilbert_symbol(&a, &b, v).unwrap())
            .product();
        assert_eq!(prod, 1);
    }
}
