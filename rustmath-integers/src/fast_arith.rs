//! Fast arithmetic operations using native types
//!
//! This module provides optimized arithmetic operations for common cases
//! where native integer types (i32, i64, u64) can be used instead of
//! arbitrary precision integers.
//!
//! Key optimizations:
//! - Using native CPU arithmetic instructions
//! - Avoiding heap allocations
//! - SIMD operations where applicable
//! - Efficient prime generation and testing
//!
//! `pow_mod` is MAGMA `Modexp(n, k, m)` (Handbook ch. 18 §18.12.1, digest
//! `rustmath-lava/docs/Chapter_18_Ring_of_Integers.md`) on machine words, with the same
//! conventions as [`crate::Integer::mod_pow`]: binary powering (D. E. Knuth, *The Art of
//! Computer Programming* vol. 2, 3rd ed., §4.6.3), negative exponents through the modular
//! inverse (extended Euclid, §4.5.2). `add_mod` and `mul_mod` follow the same conventions
//! (`Err(DivisionByZero)` for `m = 0`, least non-negative residue modulo `|m|`). `gcd` and
//! `lcm` are MAGMA `Gcd` / `Lcm` (§18.6): non-negative, returned as unsigned words so that
//! `gcd(i32::MIN, 0) = 2^31` and every lcm of two words fit. `is_prime` is the deterministic
//! [`is_prime_u64`] (proven for every `u64`). The small-prime lists and [`prime_range`] come
//! from the segmented sieve [`crate::sieve`].

use crate::primality::is_prime_u64;
use rustmath_core::{MathError, Result};

/// `base^e mod m` for `m >= 2`, `0 <= base < m` and `m <= 2^31`: every product is below
/// `2^62`, so nothing overflows.
fn pow_mod_nonneg_i64(mut base: i64, mut e: u64, m: i64) -> i64 {
    let mut result = 1i64;
    while e > 0 {
        if e & 1 == 1 {
            result = (result * base) % m;
        }
        base = (base * base) % m;
        e >>= 1;
    }
    result
}

/// `base^e mod m` for `m >= 2`, `0 <= base < m` and `m <= 2^63`: every product is below
/// `2^126`, so nothing overflows.
fn pow_mod_nonneg_i128(mut base: i128, mut e: u64, m: i128) -> i128 {
    let mut result = 1i128;
    while e > 0 {
        if e & 1 == 1 {
            result = (result * base) % m;
        }
        base = (base * base) % m;
        e >>= 1;
    }
    result
}

/// `base^e mod m` for `m >= 1` on unsigned 64-bit words. Residues are below `2^64`, so every
/// product is below `2^128` and fits a `u128`.
pub(crate) fn pow_mod_u64(base: u64, mut e: u64, m: u64) -> u64 {
    let m = u128::from(m);
    let mut base = u128::from(base) % m;
    let mut result = 1 % m;
    while e > 0 {
        if e & 1 == 1 {
            result = result * base % m;
        }
        base = base * base % m;
        e >>= 1;
    }
    result as u64
}

/// Euclid on magnitudes (Knuth, TAOCP vol. 2, §4.5.2 Algorithm A); `gcd(0, 0) = 0`.
fn gcd_magnitudes(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

/// Fast arithmetic operations using i32
///
/// Provides optimized operations when working with 32-bit integers
#[derive(Clone, Debug)]
pub struct ArithInt {
    /// Cached values for common operations
    small_primes: Vec<i32>,
}

impl ArithInt {
    /// Create a new ArithInt instance
    pub fn new() -> Self {
        ArithInt {
            small_primes: Self::compute_small_primes(1000),
        }
    }

    /// The primes up to n (segmented sieve, [`crate::sieve::primes_up_to`]).
    fn compute_small_primes(n: usize) -> Vec<i32> {
        crate::sieve::primes_up_to(n as u64)
            .into_iter()
            .map(|p| i32::try_from(p).expect("n <= 1000"))
            .collect()
    }

    /// Modular addition `(a + b) mod |m|`, the least non-negative residue in `[0, |m|)`, with
    /// the conventions of [`Self::pow_mod`] (PARI/GP `lift(Mod(a + b, m))`).
    ///
    /// # Errors
    /// `DivisionByZero` when `m = 0`.
    pub fn add_mod(&self, a: i32, b: i32, m: i32) -> Result<i32> {
        if m == 0 {
            return Err(MathError::DivisionByZero);
        }
        let r = (i64::from(a) + i64::from(b)).rem_euclid(i64::from(m).abs());
        // 0 <= r < |m| <= 2^31, so r <= i32::MAX.
        Ok(i32::try_from(r).expect("a residue below |m| <= 2^31 fits in i32"))
    }

    /// Modular multiplication `(a · b) mod |m|`, the least non-negative residue in `[0, |m|)`,
    /// with the conventions of [`Self::pow_mod`] (PARI/GP `lift(Mod(a * b, m))`). The product
    /// is formed in 64 bits, so every input is exact.
    ///
    /// # Errors
    /// `DivisionByZero` when `m = 0`.
    pub fn mul_mod(&self, a: i32, b: i32, m: i32) -> Result<i32> {
        if m == 0 {
            return Err(MathError::DivisionByZero);
        }
        let r = (i64::from(a) * i64::from(b)).rem_euclid(i64::from(m).abs());
        Ok(i32::try_from(r).expect("a residue below |m| <= 2^31 fits in i32"))
    }

    /// Modular exponentiation `a^b mod |m|` on 32-bit words (MAGMA `Modexp`, Handbook ch. 18
    /// §18.12.1), with the conventions of [`crate::Integer::mod_pow`] (PARI/GP
    /// `lift(Mod(a, m)^b)`). The result is in `[0, |m|)`:
    ///
    /// | input | result |
    /// |---|---|
    /// | `m = 0` | `Err(DivisionByZero)` |
    /// | `\|m\| = 1` | `Ok(0)` for every `a`, `b` (Z/1Z = {0}) |
    /// | `m < 0` | computed modulo `\|m\|` |
    /// | `b >= 0` | `a^b` reduced; `0^0 = 1` |
    /// | `b < 0`, `gcd(a, m) = 1` | `(a^-1)^\|b\|` reduced |
    /// | `b < 0`, not invertible | `Err(NotInvertible)` |
    ///
    /// Works in 64 bits, so every input, including `i32::MIN`, is exact.
    pub fn pow_mod(&self, a: i32, b: i32, m: i32) -> Result<i32> {
        if m == 0 {
            return Err(MathError::DivisionByZero);
        }
        let m64 = i64::from(m).abs(); // <= 2^31
        if m64 == 1 {
            return Ok(0);
        }
        let mut base = i64::from(a).rem_euclid(m64);
        if b < 0 {
            let e = num_integer::Integer::extended_gcd(&base, &m64);
            if e.gcd != 1 {
                return Err(MathError::NotInvertible);
            }
            base = e.x.rem_euclid(m64);
        }
        // The result is in [0, |m|) with |m| <= 2^31, so it fits in i32.
        Ok(pow_mod_nonneg_i64(base, u64::from(b.unsigned_abs()), m64) as i32)
    }

    /// MAGMA `Gcd(a, b)` (Handbook ch. 18 §18.6): non-negative, Euclid on `|a|`, `|b|`.
    /// Total: the result is unsigned, so `gcd(i32::MIN, 0) = 2^31` fits.
    pub fn gcd(&self, a: i32, b: i32) -> u32 {
        let g = gcd_magnitudes(u64::from(a.unsigned_abs()), u64::from(b.unsigned_abs()));
        u32::try_from(g).expect("gcd of two i32 magnitudes is at most 2^31")
    }

    /// MAGMA `Lcm(a, b)` (§18.6): non-negative, `|a| / g · |b| <= 2^62`; 0 when an input is
    /// 0. Total: the result is a `u64`.
    pub fn lcm(&self, a: i32, b: i32) -> u64 {
        if a == 0 || b == 0 {
            return 0;
        }
        let (ua, ub) = (u64::from(a.unsigned_abs()), u64::from(b.unsigned_abs()));
        (ua / gcd_magnitudes(ua, ub)) * ub
    }

    /// Primality of `n` (false for `n < 2`): the deterministic [`is_prime_u64`], a proof for
    /// every input. The old trial division overflowed `i * i` for `n` near `i32::MAX`.
    pub fn is_prime(&self, n: i32) -> bool {
        u64::try_from(n).is_ok_and(is_prime_u64)
    }

    /// Get the cached small primes
    pub fn get_small_primes(&self) -> &[i32] {
        &self.small_primes
    }
}

impl Default for ArithInt {
    fn default() -> Self {
        Self::new()
    }
}

/// Fast arithmetic operations using i64
///
/// Provides optimized operations when working with 64-bit integers
#[derive(Clone, Debug)]
pub struct ArithLLong {
    /// Cached values for common operations
    small_primes: Vec<i64>,
}

impl ArithLLong {
    /// Create a new ArithLLong instance
    pub fn new() -> Self {
        ArithLLong {
            small_primes: Self::compute_small_primes(10000),
        }
    }

    /// The primes up to n (segmented sieve, [`crate::sieve::primes_up_to`]).
    fn compute_small_primes(n: usize) -> Vec<i64> {
        crate::sieve::primes_up_to(n as u64)
            .into_iter()
            .map(|p| i64::try_from(p).expect("n <= 10000"))
            .collect()
    }

    /// Modular addition `(a + b) mod |m|`, the least non-negative residue in `[0, |m|)`, with
    /// the conventions of [`Self::pow_mod`] (PARI/GP `lift(Mod(a + b, m))`). The sum is formed
    /// in 128 bits, so it cannot overflow.
    ///
    /// # Errors
    /// `DivisionByZero` when `m = 0`.
    pub fn add_mod(&self, a: i64, b: i64, m: i64) -> Result<i64> {
        if m == 0 {
            return Err(MathError::DivisionByZero);
        }
        let r = (i128::from(a) + i128::from(b)).rem_euclid(i128::from(m).abs());
        Ok(i64::try_from(r).expect("a residue below |m| <= 2^63 fits in i64"))
    }

    /// Modular multiplication `(a · b) mod |m|`, the least non-negative residue in `[0, |m|)`,
    /// with the conventions of [`Self::pow_mod`] (PARI/GP `lift(Mod(a * b, m))`). The product
    /// is formed in 128 bits, so every input is exact.
    ///
    /// # Errors
    /// `DivisionByZero` when `m = 0`.
    pub fn mul_mod(&self, a: i64, b: i64, m: i64) -> Result<i64> {
        if m == 0 {
            return Err(MathError::DivisionByZero);
        }
        let r = (i128::from(a) * i128::from(b)).rem_euclid(i128::from(m).abs());
        Ok(i64::try_from(r).expect("a residue below |m| <= 2^63 fits in i64"))
    }

    /// Modular exponentiation `a^b mod |m|` on 64-bit words (MAGMA `Modexp`, Handbook ch. 18
    /// §18.12.1), with the conventions of [`crate::Integer::mod_pow`] (PARI/GP
    /// `lift(Mod(a, m)^b)`). The result is in `[0, |m|)`:
    ///
    /// | input | result |
    /// |---|---|
    /// | `m = 0` | `Err(DivisionByZero)` |
    /// | `\|m\| = 1` | `Ok(0)` for every `a`, `b` (Z/1Z = {0}) |
    /// | `m < 0` | computed modulo `\|m\|` |
    /// | `b >= 0` | `a^b` reduced; `0^0 = 1` |
    /// | `b < 0`, `gcd(a, m) = 1` | `(a^-1)^\|b\|` reduced |
    /// | `b < 0`, not invertible | `Err(NotInvertible)` |
    ///
    /// Works in 128 bits, so every input, including `i64::MIN`, is exact.
    pub fn pow_mod(&self, a: i64, b: i64, m: i64) -> Result<i64> {
        if m == 0 {
            return Err(MathError::DivisionByZero);
        }
        let m128 = i128::from(m).abs(); // <= 2^63
        if m128 == 1 {
            return Ok(0);
        }
        let mut base = i128::from(a).rem_euclid(m128);
        if b < 0 {
            let e = num_integer::Integer::extended_gcd(&base, &m128);
            if e.gcd != 1 {
                return Err(MathError::NotInvertible);
            }
            base = e.x.rem_euclid(m128);
        }
        // The result is in [0, |m|) with |m| <= 2^63, so it fits in i64.
        Ok(pow_mod_nonneg_i128(base, b.unsigned_abs(), m128) as i64)
    }

    /// MAGMA `Gcd(a, b)` (Handbook ch. 18 §18.6): non-negative, Euclid on `|a|`, `|b|`.
    /// Total: the result is unsigned, so `gcd(i64::MIN, 0) = 2^63` fits.
    pub fn gcd(&self, a: i64, b: i64) -> u64 {
        gcd_magnitudes(a.unsigned_abs(), b.unsigned_abs())
    }

    /// MAGMA `Lcm(a, b)` (§18.6): non-negative, `|a| / g · |b| <= 2^126`; 0 when an input is
    /// 0. Total: the result is a `u128`.
    pub fn lcm(&self, a: i64, b: i64) -> u128 {
        if a == 0 || b == 0 {
            return 0;
        }
        let (ua, ub) = (a.unsigned_abs(), b.unsigned_abs());
        u128::from(ua / gcd_magnitudes(ua, ub)) * u128::from(ub)
    }

    /// Primality of `n` (false for `n < 2`): the deterministic [`is_prime_u64`], a proof for
    /// every input.
    pub fn is_prime(&self, n: i64) -> bool {
        u64::try_from(n).is_ok_and(is_prime_u64)
    }

    /// Get the cached small primes
    pub fn get_small_primes(&self) -> &[i64] {
        &self.small_primes
    }
}

impl Default for ArithLLong {
    fn default() -> Self {
        Self::new()
    }
}

/// The primes p with `start <= p < end`, ascending (MAGMA `PrimesInInterval(start, end - 1)`),
/// by the segmented sieve [`crate::sieve::primes_in_range`]: exact for every `u64` range.
///
/// # Examples
/// ```
/// use rustmath_integers::fast_arith::prime_range;
///
/// assert_eq!(prime_range(10, 30), vec![11, 13, 17, 19, 23, 29]);
/// ```
pub fn prime_range(start: u64, end: u64) -> Vec<u64> {
    crate::sieve::primes_in_range(start, end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arith_int_modular_ops() {
        let arith = ArithInt::new();

        // Test modular addition
        assert_eq!(arith.add_mod(5, 7, 10), Ok(2));
        assert_eq!(arith.add_mod(8, 9, 10), Ok(7));

        // Test modular multiplication
        assert_eq!(arith.mul_mod(3, 4, 10), Ok(2));
        assert_eq!(arith.mul_mod(7, 8, 10), Ok(6));

        // Test modular exponentiation
        assert_eq!(arith.pow_mod(2, 10, 1000), Ok(24));
        assert_eq!(arith.pow_mod(3, 5, 13), Ok(9));
    }

    #[test]
    fn test_arith_int_gcd_lcm() {
        let arith = ArithInt::new();

        assert_eq!(arith.gcd(48, 18), 6);
        assert_eq!(arith.gcd(100, 50), 50);
        assert_eq!(arith.gcd(17, 19), 1);

        assert_eq!(arith.lcm(4, 6), 12);
        assert_eq!(arith.lcm(21, 6), 42);
    }

    #[test]
    fn test_arith_int_is_prime() {
        let arith = ArithInt::new();

        assert!(arith.is_prime(2));
        assert!(arith.is_prime(3));
        assert!(arith.is_prime(17));
        assert!(arith.is_prime(97));

        assert!(!arith.is_prime(1));
        assert!(!arith.is_prime(4));
        assert!(!arith.is_prime(100));
    }

    #[test]
    fn test_arith_llong_modular_ops() {
        let arith = ArithLLong::new();

        // Test modular addition
        assert_eq!(arith.add_mod(5, 7, 10), Ok(2));
        assert_eq!(arith.add_mod(8, 9, 10), Ok(7));

        // Test modular multiplication (using 128-bit)
        assert_eq!(arith.mul_mod(3, 4, 10), Ok(2));
        assert_eq!(arith.mul_mod(7, 8, 10), Ok(6));

        // Test with larger numbers
        let large = 1_000_000_007i64;
        assert_eq!(arith.mul_mod(large - 1, large - 1, large), Ok(1));
    }

    #[test]
    fn test_arith_llong_pow_mod() {
        let arith = ArithLLong::new();

        assert_eq!(arith.pow_mod(2, 10, 1000), Ok(24));
        assert_eq!(arith.pow_mod(3, 5, 13), Ok(9));

        // Test with larger numbers
        assert_eq!(arith.pow_mod(123456, 789, 1000000007), Ok(182677862));
    }

    #[test]
    fn test_pow_mod_unit_modulus_and_add_mod_overflow() {
        // gp: lift(Mod(5,1)^0) -> 0 (both returned 1)
        assert_eq!(ArithInt::new().pow_mod(5, 0, 1), Ok(0));
        assert_eq!(ArithLLong::new().pow_mod(5, 0, 1), Ok(0));
        // gp: (2*(2^63-2)) % (2^63-1) -> 9223372036854775805 (overflowed)
        assert_eq!(
            ArithLLong::new().add_mod(i64::MAX - 1, i64::MAX - 1, i64::MAX),
            Ok(9_223_372_036_854_775_805)
        );
    }

    #[test]
    fn test_arith_llong_gcd_lcm() {
        let arith = ArithLLong::new();

        assert_eq!(arith.gcd(48, 18), 6);
        assert_eq!(arith.gcd(100, 50), 50);
        assert_eq!(arith.gcd(17, 19), 1);

        assert_eq!(arith.lcm(4, 6), 12);
        assert_eq!(arith.lcm(21, 6), 42);
    }

    #[test]
    fn test_arith_llong_is_prime() {
        let arith = ArithLLong::new();

        assert!(arith.is_prime(2));
        assert!(arith.is_prime(3));
        assert!(arith.is_prime(17));
        assert!(arith.is_prime(97));
        assert!(arith.is_prime(1000000007));

        assert!(!arith.is_prime(1));
        assert!(!arith.is_prime(4));
        assert!(!arith.is_prime(100));
        assert!(!arith.is_prime(1000000008));
    }

    #[test]
    fn test_prime_range_small() {
        let primes = prime_range(10, 30);
        assert_eq!(primes, vec![11, 13, 17, 19, 23, 29]);
    }

    #[test]
    fn test_prime_range_with_start() {
        let primes = prime_range(0, 20);
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19]);
    }

    #[test]
    fn test_prime_range_large() {
        let primes = prime_range(100, 130);
        assert_eq!(primes, vec![101, 103, 107, 109, 113, 127]);
    }

    #[test]
    fn test_prime_range_empty() {
        let primes = prime_range(50, 50);
        assert_eq!(primes, Vec::<u64>::new());

        let primes = prime_range(100, 50);
        assert_eq!(primes, Vec::<u64>::new());
    }

    #[test]
    fn test_prime_range_single_prime() {
        let primes = prime_range(17, 18);
        assert_eq!(primes, vec![17]);
    }

    #[test]
    fn test_pow_mod_u64_full_word() {
        // gp: precprime(2^64) -> 18446744073709551557
        let p = 18_446_744_073_709_551_557u64;
        // gp: lift(Mod(2,2^64-59)^(2^64-60)) -> 1; lift(Mod(3,2^64-59)^(10^18))
        // -> 4014180641660839766; lift(Mod(2^64-2,2^64-1)^(2^64-1)) -> 18446744073709551614
        assert_eq!(pow_mod_u64(2, p - 1, p), 1);
        assert_eq!(
            pow_mod_u64(3, 1_000_000_000_000_000_000, p),
            4_014_180_641_660_839_766
        );
        assert_eq!(pow_mod_u64(u64::MAX - 1, u64::MAX, u64::MAX), u64::MAX - 1);
        assert_eq!(pow_mod_u64(5, 0, 1), 0);
    }
}
