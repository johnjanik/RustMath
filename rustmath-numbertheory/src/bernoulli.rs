//! # Bernoulli Numbers
//!
//! MAGMA: `BernoulliNumber(n)` (Handbook Ch. 25 "Real and Complex Fields" and
//! Ch. 144 "Enumerative Combinatorics"). This module computes Bernoulli numbers
//! both as exact rationals and modulo primes.
//!
//! ## Background
//!
//! Bernoulli numbers B_n are defined by the generating function (the MAGMA convention):
//! ```text
//! t/(e^t - 1) = Σ B_n * t^n / n!
//! ```
//!
//! The first few Bernoulli numbers are:
//! - B_0 = 1
//! - B_1 = -1/2
//! - B_2 = 1/6
//! - B_4 = -1/30
//! - B_6 = 1/42
//!
//! For n > 1 and n odd, B_n = 0.
//!
//! ## Algorithms
//!
//! 1. **Recurrence over Q** ([`bernoulli_number`]): Σ_{k=0}^{n} C(n+1, k) B_k = 0,
//!    exact big-rational arithmetic, memoized.
//! 2. **Power-series recurrence mod p** ([`bernoulli_mod_p`], and
//!    [`bernoulli_mod_p_single`] when p > n + 1): with b_m = B_m/m!, the identity
//!    (t/(e^t - 1)) · ((e^t - 1)/t) = 1 gives b_m = -Σ_{k<m} b_k/(m+1-k)!, which only
//!    needs word-size arithmetic modulo p. O(n²) operations.
//! 3. **Multimodular reconstruction** ([`bernoulli_multimodular`]): the denominator
//!    D_n is known exactly (von Staudt–Clausen), |B_n| < 4·n!/(2π)^n (Euler's formula
//!    |B_n| = 2·n!·ζ(n)/(2π)^n), so D_n·B_n is an integer of known height; it is
//!    computed modulo enough word-size primes and lifted by CRT. This is the strategy
//!    of D. Harvey, "A multimodular algorithm for computing Bernoulli numbers",
//!    Math. Comp. 79 (2010), with the simple O(n²) recurrence (2) as the per-prime
//!    step instead of Harvey's O(p log p) one.
//!
//! The checksum in [`verify_bernoulli_mod_p`] is
//! Σ_{k=0}^{(p-3)/2} 2^{2k}(2k+1) B_{2k} ≡ -2 (mod p), from J. Buhler, R. Crandall,
//! R. Ernvall, T. Metsänkylä, "Irregular primes and cyclotomic invariants to four
//! million", Math. Comp. 61 (1993).
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_numbertheory::bernoulli::{bernoulli_number, bernoulli_mod_p_single};
//! use rustmath_rationals::Rational;
//!
//! // Compute the 6th Bernoulli number: 1/42
//! let b6 = bernoulli_number(6);
//! assert_eq!(b6, Rational::new(1, 42).unwrap());
//!
//! // B_4 = -1/30 and 30 ≡ 2 (mod 7), so B_4 ≡ -2⁻¹ ≡ -4 ≡ 3 (mod 7)
//! let b4_mod_7 = bernoulli_mod_p_single(7, 4);
//! assert_eq!(b4_mod_7, Some(3));
//!
//! // B_6 = 1/42 is not 7-integral (von Staudt–Clausen: 7 - 1 divides 6)
//! assert_eq!(bernoulli_mod_p_single(7, 6), None);
//! ```

use once_cell::sync::Lazy;
use rustmath_core::{NumericConversion, Ring};
use rustmath_integers::crt;
use rustmath_integers::fast_arith::prime_range;
use rustmath_integers::prime::is_prime;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::collections::HashMap;
use std::sync::Mutex;

/// Thread-safe cache for memoizing Bernoulli numbers
static CACHE: Lazy<Mutex<HashMap<usize, Rational>>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// Primes used by [`bernoulli_multimodular`] are taken downward from here, so that
/// products of two residues fit comfortably in a `u64`.
const MULTIMODULAR_PRIME_CEILING: u64 = 1 << 31;

/// Width of the sieve windows used to enumerate those primes.
const PRIME_WINDOW: u64 = 1 << 16;

/// The primes below a ceiling, in decreasing order, sieved window by window.
struct DescendingPrimes {
    hi: u64,
    window: Vec<u64>,
}

impl DescendingPrimes {
    fn below(ceiling: u64) -> Self {
        DescendingPrimes {
            hi: ceiling,
            window: Vec::new(),
        }
    }
}

impl Iterator for DescendingPrimes {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        while self.window.is_empty() {
            if self.hi <= 2 {
                return None;
            }
            let lo = self.hi.saturating_sub(PRIME_WINDOW).max(2);
            self.window = prime_range(lo, self.hi); // ascending
            self.hi = lo;
        }
        self.window.pop()
    }
}

/// Helper function to create a Rational, unwrapping the Result
fn rational(numer: i64, denom: i64) -> Rational {
    Rational::new(numer, denom).expect("Invalid rational number")
}

/// Computes the n-th Bernoulli number using the recursive definition.
///
/// This uses the recurrence relation:
/// ```text
/// Σ_{k=0}^n C(n+1, k) * B_k = 0
/// ```
///
/// which gives:
/// ```text
/// B_n = -1/(n+1) * Σ_{k=0}^{n-1} C(n+1, k) * B_k
/// ```
///
/// Arithmetic is exact (big rationals, big binomial coefficients); results are
/// memoized.
///
/// # Complexity
///
/// O(n³) big-integer operations for the first call (O(n²) terms, each with an
/// O(n) binomial coefficient); later calls with smaller indices hit the cache.
///
/// # Examples
///
/// ```rust
/// use rustmath_numbertheory::bernoulli::bernoulli_number;
/// use rustmath_rationals::Rational;
///
/// assert_eq!(bernoulli_number(0), Rational::from(1));
/// assert_eq!(bernoulli_number(1), Rational::new(-1, 2).unwrap());
/// assert_eq!(bernoulli_number(2), Rational::new(1, 6).unwrap());
/// assert_eq!(bernoulli_number(4), Rational::new(-1, 30).unwrap());
/// assert_eq!(bernoulli_number(12), Rational::new(-691, 2730).unwrap());
/// ```
pub fn bernoulli_number(n: usize) -> Rational {
    // Check cache first
    {
        let cache = CACHE.lock().unwrap();
        if let Some(result) = cache.get(&n) {
            return result.clone();
        }
    }

    // Base cases
    if n == 0 {
        return Rational::from(1);
    }

    if n == 1 {
        return rational(-1, 2);
    }

    // B_n = 0 for odd n > 1
    if n > 1 && n % 2 == 1 {
        return Ring::zero();
    }

    // Compute using recurrence relation
    let mut sum: Rational = Ring::zero();

    for k in 0..n {
        let binom = binomial_coefficient(n + 1, k);
        let b_k = bernoulli_number(k);
        sum = sum + Rational::from_integer(binom) * b_k;
    }

    let result = -sum / Rational::from(n as i64 + 1);

    // Store in cache
    {
        let mut cache = CACHE.lock().unwrap();
        cache.insert(n, result.clone());
    }

    result
}

/// Computes binomial coefficient C(n, k) = n! / (k! * (n-k)!) exactly.
fn binomial_coefficient(n: usize, k: usize) -> Integer {
    if k > n {
        return Integer::zero();
    }

    if k == 0 || k == n {
        return Integer::one();
    }

    let k = k.min(n - k); // Take advantage of symmetry

    // C(n, i+1) = C(n, i) * (n - i) / (i + 1), and the division is exact.
    let mut result = Integer::one();
    for i in 0..k {
        result = result * Integer::from((n - i) as u64) / Integer::from((i + 1) as u64);
    }

    result
}

/// Computes Bernoulli numbers B_0, B_2, ..., B_{2*max_n}.
///
/// Each entry is computed with [`bernoulli_number`] (memoized, so the whole
/// vector costs the same as the largest entry).
///
/// # Arguments
///
/// * `max_n` - Compute Bernoulli numbers up to B_{2*max_n}
///
/// # Returns
///
/// Vector containing [B_0, B_2, B_4, ..., B_{2*max_n}]
///
/// # Examples
///
/// ```rust
/// use rustmath_numbertheory::bernoulli::bernoulli_numbers_vec;
/// use rustmath_rationals::Rational;
///
/// let bernoulli = bernoulli_numbers_vec(3);
/// assert_eq!(bernoulli.len(), 4);
/// assert_eq!(bernoulli[0], Rational::from(1)); // B_0
/// assert_eq!(bernoulli[1], Rational::new(1, 6).unwrap()); // B_2
/// assert_eq!(bernoulli[2], Rational::new(-1, 30).unwrap()); // B_4
/// assert_eq!(bernoulli[3], Rational::new(1, 42).unwrap()); // B_6
/// ```
pub fn bernoulli_numbers_vec(max_n: usize) -> Vec<Rational> {
    let mut result = Vec::with_capacity(max_n + 1);

    for i in 0..=max_n {
        let n = 2 * i;
        result.push(bernoulli_number(n));
    }

    result
}

/// `a * b mod p` without overflow.
fn mul_mod(a: u64, b: u64, p: u64) -> u64 {
    ((a as u128 * b as u128) % p as u128) as u64
}

/// `base^exp mod p` without overflow.
fn pow_mod_u64(base: u64, mut exp: u64, p: u64) -> u64 {
    let mut result = 1 % p;
    let mut base = base % p;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mul_mod(result, base, p);
        }
        base = mul_mod(base, base, p);
        exp >>= 1;
    }
    result
}

/// B_0, B_1, ..., B_{max_n} modulo the prime `p`.
///
/// With b_m = B_m/m!, the coefficient of t^m in
/// (t/(e^t - 1)) · ((e^t - 1)/t) = 1 gives
/// ```text
/// b_m = -Σ_{k=0}^{m-1} b_k / (m+1-k)!      (m ≥ 1),
/// ```
/// so only the inverses of 1!, ..., (max_n+1)! modulo p are needed. B_m = m!·b_m.
///
/// Requires `p` prime and `max_n + 1 < p`: then every j ≤ max_n + 1 is a unit mod p,
/// and (p-1) ∤ m for 2 ≤ m ≤ max_n, so every B_m in range is p-integral.
/// Cost: O(max_n²) word operations.
fn bernoulli_table_mod_p(p: u64, max_n: usize) -> Vec<u64> {
    assert!(
        (max_n as u64) + 1 < p,
        "bernoulli_table_mod_p: need max_n + 1 < p (max_n = {}, p = {})",
        max_n,
        p
    );
    let top = max_n + 1;

    let mut fact = vec![1 % p; top + 1];
    for j in 1..=top {
        fact[j] = mul_mod(fact[j - 1], j as u64, p);
    }
    let mut inv_fact = vec![1 % p; top + 1];
    inv_fact[top] = pow_mod_u64(fact[top], p - 2, p);
    for j in (1..=top).rev() {
        inv_fact[j - 1] = mul_mod(inv_fact[j], j as u64, p);
    }

    // b[k] = B_k / k! (mod p); B_k = 0 for odd k ≥ 3.
    let mut b = vec![0u64; max_n + 1];
    b[0] = 1 % p;
    for m in 1..=max_n {
        if m >= 3 && m % 2 == 1 {
            continue;
        }
        let mut s = 0u64;
        for k in 0..m {
            if k >= 3 && k % 2 == 1 {
                continue;
            }
            s = (s + mul_mod(b[k], inv_fact[m + 1 - k], p)) % p;
        }
        b[m] = (p - s) % p;
    }

    (0..=max_n).map(|m| mul_mod(fact[m], b[m], p)).collect()
}

/// Panics unless `p` is a prime.
fn assert_prime_modulus(p: i64, caller: &str) {
    assert!(
        p >= 2 && is_prime(&Integer::from(p)),
        "{}: modulus {} is not prime",
        caller,
        p
    );
}

/// Computes the n-th Bernoulli number modulo a prime p.
///
/// Returns `None` if B_n is not p-integral (i.e., p divides its denominator; by
/// von Staudt–Clausen this happens for even n ≥ 2 exactly when (p-1) | n, and for
/// n = 1 when p = 2).
///
/// # Algorithm
///
/// - If p > n + 1, the power-series recurrence modulo p is used (word-size
///   arithmetic, O(n²) operations, no big numbers).
/// - Otherwise the exact rational [`bernoulli_number`]`(n)` is computed and reduced
///   modulo p.
///
/// # Arguments
///
/// * `p` - Prime modulus
/// * `n` - Index of Bernoulli number
///
/// # Returns
///
/// `Some(B_n mod p)` (in `0..p`) if B_n is p-integral, `None` otherwise
///
/// # Panics
///
/// Panics if `p` is not a prime (this includes every `p < 2`).
///
/// # Examples
///
/// ```rust
/// use rustmath_numbertheory::bernoulli::bernoulli_mod_p_single;
///
/// // B_4 = -1/30, and 30 ≡ 2 (mod 7), so B_4 ≡ -1/2 ≡ 3 (mod 7)
/// assert_eq!(bernoulli_mod_p_single(7, 4), Some(3));
///
/// // 37 is irregular: it divides the numerator of B_32
/// assert_eq!(bernoulli_mod_p_single(37, 32), Some(0));
/// ```
pub fn bernoulli_mod_p_single(p: i64, n: usize) -> Option<i64> {
    assert_prime_modulus(p, "bernoulli_mod_p_single");

    if n == 0 {
        return Some(1);
    }

    if n == 1 {
        // B_1 = -1/2
        let inv2 = mod_inverse(2, p)?;
        return Some((-inv2).rem_euclid(p));
    }

    // B_n = 0 for odd n > 1
    if n % 2 == 1 {
        return Some(0);
    }

    // von Staudt–Clausen: for even n ≥ 2, p divides the denominator of B_n iff (p-1) | n.
    if n.is_multiple_of(p as usize - 1) {
        return None;
    }

    if (n as u64) + 1 < p as u64 {
        let table = bernoulli_table_mod_p(p as u64, n);
        return Some(table[n] as i64);
    }

    // p ≤ n + 1: reduce the exact value.
    let b_n = bernoulli_number(n);
    let modulus = Integer::from(p);
    let numer = b_n.numerator().modulo(&modulus).to_i64();
    let denom = b_n.denominator().modulo(&modulus).to_i64();
    if gcd(denom, p) != 1 {
        return None; // Not p-integral
    }
    let denom_inv = mod_inverse(denom, p)?;

    Some(((numer as i128 * denom_inv as i128) % p as i128) as i64)
}

/// Computes all Bernoulli numbers B_0, B_2, ..., B_{p-3} modulo a prime p.
///
/// Uses the power-series recurrence modulo p (see the module documentation):
/// O(p²) word operations. All of these B_{2k} are p-integral (von Staudt–Clausen,
/// since p - 1 > p - 3), and p is irregular iff one of them is 0.
///
/// # Arguments
///
/// * `p` - Prime modulus, at least 5
///
/// # Returns
///
/// Vector containing [B_0, B_2, B_4, ..., B_{p-3}] mod p, each in `0..p`
///
/// # Panics
///
/// Panics if `p < 5` or if `p` is not a prime.
///
/// # Examples
///
/// ```rust
/// use rustmath_numbertheory::bernoulli::bernoulli_mod_p;
///
/// let b_mod_7 = bernoulli_mod_p(7);
/// // B_0 = 1, B_2 = 1/6, B_4 = -1/30
/// assert_eq!(b_mod_7, vec![1, 6, 3]);
/// ```
pub fn bernoulli_mod_p(p: i64) -> Vec<i64> {
    assert!(p >= 5, "Prime must be at least 5");
    assert_prime_modulus(p, "bernoulli_mod_p");

    let max_index = (p - 3) as usize;
    bernoulli_table_mod_p(p as u64, max_index)
        .into_iter()
        .step_by(2)
        .map(|b| b as i64)
        .collect()
}

/// Computes Bernoulli number using multi-modular algorithm.
///
/// For even n ≥ 2: the denominator D_n is computed exactly by von Staudt–Clausen,
/// and the integer N = D_n·B_n satisfies |N| < 4·D_n·n!/6^n (from
/// |B_n| = 2·n!·ζ(n)/(2π)^n with ζ(n) < 2 and 2π > 6). N is computed modulo
/// primes p < 2^31 (all > n + 1) with the power-series recurrence, until the product
/// M of the primes exceeds twice that bound, and recovered as the symmetric CRT lift
/// in (-M/2, M/2). The result is exact; no step is heuristic.
///
/// # Complexity
///
/// O(n²) word operations per prime, with about (log₂ n! - n·log₂ 6)/31 primes.
///
/// # Arguments
///
/// * `n` - Index of Bernoulli number
/// * `num_primes` - Minimum number of primes to use. More primes are added
///   automatically until the height bound above is met, so a small value never
///   makes the result inexact.
///
/// # Returns
///
/// The n-th Bernoulli number
///
/// # Panics
///
/// Only for even n ≥ 2 (n = 0, n = 1 and odd n return at once):
///
/// - if n + 1 ≥ 2^30 (the primes used must exceed n + 1 and lie below 2^31);
/// - with "ran out of primes" if more primes are needed than lie in (n + 1, 2^31).
///   This happens when `num_primes` exceeds that count, or when the height bound
///   alone needs them all (n beyond roughly 1.3·10^8).
///
/// # Examples
///
/// ```rust
/// use rustmath_numbertheory::bernoulli::{bernoulli_multimodular, bernoulli_number};
/// use rustmath_rationals::Rational;
///
/// let b6 = bernoulli_multimodular(6, 10);
/// assert_eq!(b6, Rational::new(1, 42).unwrap());
///
/// // B_30 = 8615841276005/14322
/// let b30 = bernoulli_multimodular(30, 1);
/// assert_eq!(b30, Rational::new(8615841276005i64, 14322).unwrap());
/// assert_eq!(b30, bernoulli_number(30));
/// ```
pub fn bernoulli_multimodular(n: usize, num_primes: usize) -> Rational {
    if n == 0 {
        return Rational::from(1);
    }

    if n == 1 {
        return rational(-1, 2);
    }

    // B_n = 0 for odd n > 1
    if n % 2 == 1 {
        return Ring::zero();
    }

    assert!(
        (n as u64) + 1 < MULTIMODULAR_PRIME_CEILING / 2,
        "bernoulli_multimodular: n = {} is too large for word-size primes",
        n
    );

    // Exact denominator (von Staudt–Clausen)
    let denom = bernoulli_denominator(n);

    // Height bound: |D_n · B_n| < 4 · D_n · n! / 6^n  ≤  height
    let mut n_factorial = Integer::one();
    for j in 2..=n {
        n_factorial = n_factorial * Integer::from(j as u64);
    }
    let height = Integer::from(4) * denom.clone() * n_factorial / Integer::from(6).pow(n as u32)
        + Integer::one();
    let target = Integer::from(2) * height;

    // Compute N = D_n · B_n modulo enough primes p with n + 1 < p < 2^31.
    let mut residues = Vec::new();
    let mut moduli = Vec::new();
    let mut modulus = Integer::one();
    let mut primes = DescendingPrimes::below(MULTIMODULAR_PRIME_CEILING);

    while moduli.len() < num_primes.max(1) || modulus <= target {
        let p_word = primes.next().expect("a prime above n + 1");
        let p = Integer::from(p_word);
        assert!(
            (n as u64) + 1 < p_word,
            "bernoulli_multimodular: ran out of primes above n + 1 = {}",
            n + 1
        );

        let b_n = bernoulli_table_mod_p(p_word, n)[n];
        let d = denom.modulo(&p).to_u64().expect("residue below 2^31");
        residues.push(Integer::from(mul_mod(b_n, d, p_word)));
        modulus = modulus * p.clone();
        moduli.push(p);
    }

    // Symmetric lift of the CRT residue to (-M/2, M/2)
    let lifted = chinese_remainder_theorem(&residues, &moduli);
    let numer = if lifted.clone() * Integer::from(2) > modulus {
        lifted - modulus
    } else {
        lifted
    };

    Rational::new(numer, denom).expect("Bernoulli denominator is positive")
}

/// Simple primality test for small numbers.
fn is_prime_simple(n: i64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 {
        return true;
    }
    if n % 2 == 0 {
        return false;
    }

    let sqrt_n = (n as f64).sqrt() as i64;
    for i in (3..=sqrt_n).step_by(2) {
        if n % i == 0 {
            return false;
        }
    }

    true
}

/// Computes the denominator of B_n using von Staudt-Clausen theorem.
///
/// For even n ≥ 2 the denominator is the product of all primes p with (p-1) | n.
/// Also D_0 = 1, D_1 = 2, and D_n = 1 for odd n > 1 (where B_n = 0).
fn bernoulli_denominator(n: usize) -> Integer {
    if n == 0 {
        return Integer::one();
    }

    if n == 1 {
        return Integer::from(2);
    }

    if n % 2 == 1 {
        return Integer::one();
    }

    let mut denom = Integer::one();
    let mut include = |d: usize| {
        if is_prime_simple(d as i64 + 1) {
            denom = denom.clone() * Integer::from(d as u64 + 1);
        }
    };

    // Run over all divisors d of n.
    let mut d = 1usize;
    while d * d <= n {
        if n.is_multiple_of(d) {
            include(d);
            if d * d != n {
                include(n / d);
            }
        }
        d += 1;
    }

    denom
}

/// Chinese Remainder Theorem over pairwise coprime moduli.
///
/// Returns the unique x in [0, Π m_i) with x ≡ a_i (mod m_i), as a big integer.
fn chinese_remainder_theorem(residues: &[Integer], moduli: &[Integer]) -> Integer {
    crt::chinese_remainder_theorem(residues, moduli)
        .expect("CRT needs equally many residues and pairwise coprime positive moduli")
}

/// Computes modular multiplicative inverse using extended Euclidean algorithm.
///
/// Returns Some(x) where a*x ≡ 1 (mod m), or None if gcd(a, m) != 1.
fn mod_inverse(a: i64, m: i64) -> Option<i64> {
    let (g, x, _) = extended_gcd(a, m);

    if g != 1 {
        None
    } else {
        Some(x.rem_euclid(m))
    }
}

/// Extended Euclidean algorithm.
///
/// Returns (gcd, x, y) where gcd = a*x + b*y.
fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if b == 0 {
        return (a, 1, 0);
    }

    let (g, x1, y1) = extended_gcd(b, a % b);
    let x = y1;
    let y = x1 - (a / b) * y1;

    (g, x, y)
}

/// Computes GCD using Euclidean algorithm.
fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());

    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }

    a
}

/// Verifies a list of Bernoulli numbers modulo p using the checksum identity.
///
/// The identity (Buhler–Crandall–Ernvall–Metsänkylä) is:
/// Σ_{k=0}^{(p-3)/2} 2^{2k} (2k+1) B_{2k} ≡ -2 (mod p)
///
/// # Arguments
///
/// * `p` - Odd prime modulus
/// * `bernoulli` - Vector of Bernoulli numbers B_0, B_2, ..., B_{p-3} mod p
///   (exactly (p-1)/2 entries, as returned by [`bernoulli_mod_p`])
///
/// # Returns
///
/// `true` if the checksum passes, `false` otherwise (including when the list does
/// not have (p-1)/2 entries)
///
/// # Panics
///
/// Panics if `p` is not an odd prime. The identity only holds for odd primes, so a
/// composite modulus (or 2) is rejected instead of being checked.
///
/// # Examples
///
/// ```rust
/// use rustmath_numbertheory::bernoulli::{bernoulli_mod_p, verify_bernoulli_mod_p};
///
/// let mut b = bernoulli_mod_p(11);
/// assert!(verify_bernoulli_mod_p(11, &b));
/// b[1] = (b[1] + 1) % 11; // corrupt B_2
/// assert!(!verify_bernoulli_mod_p(11, &b));
/// ```
pub fn verify_bernoulli_mod_p(p: i64, bernoulli: &[i64]) -> bool {
    assert!(
        p >= 3 && is_prime(&Integer::from(p)),
        "verify_bernoulli_mod_p: modulus {} is not an odd prime",
        p
    );
    if bernoulli.len() != ((p - 3) / 2 + 1) as usize {
        return false;
    }

    let modulus = p as i128;
    let mut sum = 0i128;

    for (k, &b_2k) in bernoulli.iter().enumerate() {
        // 2^{2k} (2k + 1) B_{2k}
        let power_of_2 = mod_pow(2, 2 * k as i64, p) as i128;
        let term =
            (power_of_2 * ((2 * k as i128 + 1) % modulus)) % modulus * (b_2k as i128 % modulus);
        sum = (sum + term) % modulus;
    }

    sum.rem_euclid(modulus) == (-2i128).rem_euclid(modulus)
}

/// Modular exponentiation: computes base^exp mod m (for m > 0, exp ≥ 0).
fn mod_pow(base: i64, exp: i64, m: i64) -> i64 {
    let m = m as i128;
    let mut result = 1i128 % m;
    let mut base = (base as i128).rem_euclid(m);
    let mut exp = exp;

    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * base) % m;
        }
        exp >>= 1;
        base = (base * base) % m;
    }

    result as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn big(s: &str) -> Integer {
        Integer::from_decimal_str(s).expect("decimal integer literal")
    }

    /// `b mod p` for a p-integral rational, by big-integer arithmetic.
    fn reduce_mod(b: &Rational, p: i64) -> i64 {
        let m = Integer::from(p);
        let den_inv = b.denominator().modulo(&m).mod_inverse(&m).expect("p-integral");
        (b.numerator().modulo(&m) * den_inv).modulo(&m).to_i64()
    }

    #[test]
    fn test_bernoulli_small() {
        assert_eq!(bernoulli_number(0), Rational::from(1));
        assert_eq!(bernoulli_number(1), rational(-1, 2));
        assert_eq!(bernoulli_number(2), rational(1, 6));
        assert_eq!(bernoulli_number(3), Ring::zero());
        assert_eq!(bernoulli_number(4), rational(-1, 30));
        assert_eq!(bernoulli_number(5), Ring::zero());
        assert_eq!(bernoulli_number(6), rational(1, 42));
        assert_eq!(bernoulli_number(8), rational(-1, 30));
        assert_eq!(bernoulli_number(10), rational(5, 66));
        assert_eq!(bernoulli_number(12), rational(-691, 2730));
    }

    #[test]
    fn test_bernoulli_large_index() {
        // Reference values: Akiyama–Tanigawa over Python fractions (independent
        // algorithm); B_60 also agrees with OEIS A027641/A027642.
        // (Before the binomial fix this overflowed i64 and panicked.)
        let b60 = Rational::new(
            big("-1215233140483755572040304994079820246041491"),
            Integer::from(56786730),
        )
        .unwrap();
        assert_eq!(bernoulli_number(60), b60);

        let b70 = Rational::new(
            big("1505381347333367003803076567377857208511438160235"),
            Integer::from(4686),
        )
        .unwrap();
        assert_eq!(bernoulli_number(70), b70);
    }

    #[test]
    fn test_bernoulli_odd_zero() {
        // All odd Bernoulli numbers > 1 are zero
        for n in (3..20).step_by(2) {
            assert_eq!(bernoulli_number(n), Ring::zero());
        }
    }

    #[test]
    fn test_bernoulli_vec() {
        let vec = bernoulli_numbers_vec(3);
        assert_eq!(vec.len(), 4);
        assert_eq!(vec[0], Rational::from(1)); // B_0
        assert_eq!(vec[1], rational(1, 6)); // B_2
        assert_eq!(vec[2], rational(-1, 30)); // B_4
        assert_eq!(vec[3], rational(1, 42)); // B_6
    }

    #[test]
    fn test_binomial_coefficient() {
        assert_eq!(binomial_coefficient(5, 0), Integer::from(1));
        assert_eq!(binomial_coefficient(5, 1), Integer::from(5));
        assert_eq!(binomial_coefficient(5, 2), Integer::from(10));
        assert_eq!(binomial_coefficient(5, 3), Integer::from(10));
        assert_eq!(binomial_coefficient(5, 4), Integer::from(5));
        assert_eq!(binomial_coefficient(5, 5), Integer::from(1));
        assert_eq!(binomial_coefficient(5, 6), Integer::from(0));
        // Exceeds i64 (Python: math.comb(70, 35))
        assert_eq!(binomial_coefficient(70, 35), big("112186277816662845432"));
    }

    #[test]
    fn test_bernoulli_mod_p_single() {
        // Test with small primes
        assert_eq!(bernoulli_mod_p_single(7, 0), Some(1));

        // B_1 = -1/2 ≡ 3 (mod 7); not 2-integral
        assert_eq!(bernoulli_mod_p_single(7, 1), Some(3));
        assert_eq!(bernoulli_mod_p_single(2, 1), None);

        // B_2 = 1/6, 6^{-1} ≡ 6 (mod 7)
        assert_eq!(bernoulli_mod_p_single(7, 2), Some(6));

        // B_4 = -1/30, 30 ≡ 2 (mod 7), 2^{-1} ≡ 4 (mod 7), -4 ≡ 3 (mod 7)
        assert_eq!(bernoulli_mod_p_single(7, 4), Some(3));

        // Odd index > 1
        assert_eq!(bernoulli_mod_p_single(7, 5), Some(0));

        // von Staudt–Clausen: (p-1) | n  =>  p | denominator
        assert_eq!(bernoulli_mod_p_single(7, 6), None);
        assert_eq!(bernoulli_mod_p_single(2, 4), None);
        assert_eq!(bernoulli_mod_p_single(13, 12), None);

        // Exact-reduction path (p ≤ n + 1); values from Python fractions:
        // B_10 = 5/66 ≡ 4 (mod 7), B_12 = -691/2730 ≡ 1 (mod 11), B_22 ≡ 1 (mod 5)
        assert_eq!(bernoulli_mod_p_single(7, 10), Some(4));
        assert_eq!(bernoulli_mod_p_single(11, 12), Some(1));
        assert_eq!(bernoulli_mod_p_single(5, 22), Some(1));

        // Power-series path (p > n + 1): B_10 ≡ 5 (mod 13)
        assert_eq!(bernoulli_mod_p_single(13, 10), Some(5));

        // Irregular pairs (p, k) with p | numerator(B_k)
        assert_eq!(bernoulli_mod_p_single(37, 32), Some(0));
        assert_eq!(bernoulli_mod_p_single(59, 44), Some(0));
        assert_eq!(bernoulli_mod_p_single(67, 58), Some(0));
    }

    #[test]
    fn test_bernoulli_mod_p_single_agrees_with_exact() {
        for p in [2i64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61] {
            for n in 0..=60usize {
                let b = bernoulli_number(n);
                let expected = if b.denominator().modulo(&Integer::from(p)).is_zero() {
                    None
                } else {
                    Some(reduce_mod(&b, p))
                };
                assert_eq!(bernoulli_mod_p_single(p, n), expected, "B_{} mod {}", n, p);
            }
        }
    }

    #[test]
    #[should_panic(expected = "not prime")]
    fn test_bernoulli_mod_p_single_rejects_composite() {
        bernoulli_mod_p_single(9, 4);
    }

    #[test]
    #[should_panic(expected = "modulus 9 is not prime")]
    fn test_bernoulli_mod_p_rejects_composite() {
        bernoulli_mod_p(9);
    }

    #[test]
    #[should_panic(expected = "too large for word-size primes")]
    fn test_bernoulli_multimodular_rejects_huge_index() {
        // n + 1 = 2^30 + 1 ≥ 2^30: rejected before any work is done.
        bernoulli_multimodular(1 << 30, 1);
    }

    #[test]
    fn test_bernoulli_mod_p() {
        let b_mod_7 = bernoulli_mod_p(7);
        assert_eq!(b_mod_7, vec![1, 6, 3]); // B_0, B_2, B_4

        // p = 37: Python fractions reference; 37 | numerator(B_32)
        let b_mod_37 = bernoulli_mod_p(37);
        assert_eq!(
            b_mod_37,
            vec![1, 31, 16, 15, 16, 4, 17, 32, 22, 31, 15, 15, 17, 12, 29, 2, 0, 2]
        );

        // p = 101 (exact rationals overflowed i64 here before): Python fractions
        // reference; (101, 68) is an irregular pair.
        let b_mod_101 = bernoulli_mod_p(101);
        assert_eq!(
            b_mod_101,
            vec![
                1, 17, 37, 89, 37, 72, 39, 18, 24, 27, 68, 26, 16, 20, 79, 84, 57, 51, 92, 26,
                63, 56, 30, 76, 7, 91, 57, 51, 42, 31, 53, 45, 47, 62, 0, 28, 54, 92, 79, 25,
                86, 15, 39, 10, 84, 29, 11, 51, 2, 76
            ]
        );
    }

    #[test]
    fn test_mod_inverse() {
        assert_eq!(mod_inverse(3, 7), Some(5)); // 3*5 = 15 ≡ 1 (mod 7)
        assert_eq!(mod_inverse(2, 7), Some(4)); // 2*4 = 8 ≡ 1 (mod 7)
        assert_eq!(mod_inverse(6, 7), Some(6)); // 6*6 = 36 ≡ 1 (mod 7)
        assert_eq!(mod_inverse(2, 4), None); // gcd(2, 4) = 2 != 1
    }

    #[test]
    fn test_extended_gcd() {
        let (g, x, y) = extended_gcd(35, 15);
        assert_eq!(g, 5);
        assert_eq!(35 * x + 15 * y, 5);
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(35, 15), 5);
        assert_eq!(gcd(12, 18), 6);
        assert_eq!(gcd(7, 13), 1);
    }

    #[test]
    fn test_chinese_remainder_theorem() {
        // x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7)
        // Solution: x ≡ 23 (mod 105)
        let residues: Vec<Integer> = [2, 3, 2].iter().map(|&r| Integer::from(r)).collect();
        let moduli: Vec<Integer> = [3, 5, 7].iter().map(|&m| Integer::from(m)).collect();
        let result = chinese_remainder_theorem(&residues, &moduli);
        assert_eq!(result.modulo(&Integer::from(3)), Integer::from(2));
        assert_eq!(result.modulo(&Integer::from(5)), Integer::from(3));
        assert_eq!(result.modulo(&Integer::from(7)), Integer::from(2));
        assert_eq!(result, Integer::from(23));
    }

    #[test]
    fn test_mod_pow() {
        assert_eq!(mod_pow(2, 10, 1000), 24); // 2^10 = 1024 ≡ 24 (mod 1000)
        assert_eq!(mod_pow(3, 4, 7), 4); // 3^4 = 81 ≡ 4 (mod 7)
        // No overflow for moduli near 2^62
        let m = (1i64 << 62) - 57;
        assert_eq!(mod_pow(m - 1, 2, m), 1);
    }

    #[test]
    fn test_verify_bernoulli_mod_p() {
        let b_mod_7 = bernoulli_mod_p(7);
        assert!(verify_bernoulli_mod_p(7, &b_mod_7));

        let b_mod_11 = bernoulli_mod_p(11);
        assert!(verify_bernoulli_mod_p(11, &b_mod_11));

        for p in [13i64, 37, 101, 103, 211] {
            assert!(verify_bernoulli_mod_p(p, &bernoulli_mod_p(p)), "checksum mod {}", p);
        }

        // A corrupted entry or a truncated list fails the checksum.
        let mut bad = b_mod_11.clone();
        bad[2] = (bad[2] + 1) % 11;
        assert!(!verify_bernoulli_mod_p(11, &bad));
        assert!(!verify_bernoulli_mod_p(11, &b_mod_11[..4]));

        // Smallest odd prime: the list is [B_0] = [1], and 1 ≡ -2 (mod 3).
        assert!(verify_bernoulli_mod_p(3, &[1]));
        assert!(!verify_bernoulli_mod_p(3, &[0]));
    }

    #[test]
    #[should_panic(expected = "modulus 9 is not an odd prime")]
    fn test_verify_bernoulli_mod_p_rejects_composite() {
        // Σ_k 4^k (2k+1) b_k = 7 ≡ -2 (mod 9), so without the primality check this
        // list of the right length would pass the checksum for the composite 9.
        verify_bernoulli_mod_p(9, &[7, 0, 0, 0]);
    }

    #[test]
    #[should_panic(expected = "modulus 2 is not an odd prime")]
    fn test_verify_bernoulli_mod_p_rejects_two() {
        verify_bernoulli_mod_p(2, &[1]);
    }

    #[test]
    fn test_is_prime_simple() {
        assert!(!is_prime_simple(1));
        assert!(is_prime_simple(2));
        assert!(is_prime_simple(3));
        assert!(!is_prime_simple(4));
        assert!(is_prime_simple(5));
        assert!(is_prime_simple(7));
        assert!(is_prime_simple(11));
        assert!(!is_prime_simple(15));
    }

    #[test]
    fn test_bernoulli_multimodular() {
        // Test that multimodular gives same result as direct computation
        for n in vec![0, 2, 4, 6, 8] {
            let direct = bernoulli_number(n);
            let multi = bernoulli_multimodular(n, 5);
            assert_eq!(direct, multi, "Mismatch for B_{}", n);
        }
        // Beyond the old n ≤ 20 shortcut, with too few primes requested.
        for n in 0..=120usize {
            assert_eq!(bernoulli_multimodular(n, 1), bernoulli_number(n), "Mismatch for B_{}", n);
        }
    }

    #[test]
    fn test_bernoulli_multimodular_reference_values() {
        // B_22 = 854513/138, B_100 = -94598...47711/33330 (OEIS A027641/A027642,
        // re-derived with Akiyama–Tanigawa over Python fractions).
        assert_eq!(bernoulli_multimodular(22, 10), rational(854513, 138));
        let b100 = Rational::new(
            big("-94598037819122125295227433069493721872702841533066936133385696204311395415197247711"),
            Integer::from(33330),
        )
        .unwrap();
        assert_eq!(bernoulli_multimodular(100, 10), b100);
    }

    #[test]
    fn test_bernoulli_denominator() {
        // B_0 has denominator 1
        assert_eq!(bernoulli_denominator(0), Integer::from(1));

        // B_2 has denominator 6 = 2*3 (since 1|2 for both 2-1=1 and 3-1=2)
        assert_eq!(bernoulli_denominator(2), Integer::from(6));

        // B_12 = -691/2730, 2730 = 2·3·5·7·13
        assert_eq!(bernoulli_denominator(12), Integer::from(2730));

        // Primes above 1000 count (1009 - 1 = 1008), and the product exceeds i64.
        assert_eq!(bernoulli_denominator(1008), big("14493038256293268734790"));

        for n in 0..=60usize {
            if n == 1 || n % 2 == 0 {
                assert_eq!(&bernoulli_denominator(n), bernoulli_number(n).denominator(), "D_{}", n);
            }
        }
    }
}
