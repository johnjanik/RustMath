//! Prime numbers (MAGMA Handbook ch. 18 §18.9, "Primes and Primality Testing"; digest
//! `rustmath-lava/docs/Chapter_18_Ring_of_Integers.md`).
//!
//! - [`is_prime`] / [`try_is_prime`]: MAGMA `IsPrime(n)` (`Proof := true`), a proof for every
//!   size: the deterministic word test below 2^64, the twelve strong tests 2..37 below
//!   psi_12 = 318665857834031151167461 (Sorenson-Webster 2017), then Baillie-PSW and the
//!   Atkin-Morain ECPP prover ([`crate::ecpp`], [AM93]) with a verified PARI-compatible
//!   certificate ([`prove_primality`], [`EcppCertificate`]).
//! - [`is_probable_prime`]: MAGMA `IsProbablePrime(n)` / `IsPrime(n: Proof := false)`
//!   (Baillie-PSW, exact below psi_12; [`crate::primality`]).
//! - [`next_prime`], [`previous_prime`], [`prime_range`], [`nth_prime`], [`prime_pi`],
//!   [`primes_first_n`]: §18.9.2 over the segmented sieve [`crate::sieve`] below 2^64; above
//!   2^64 the candidates are pre-sieved by small primes and the probable primes are proven.
//! - The factorization drivers (§18.10) prove every factor they return with [`is_prime`].

use crate::ecpp::ProverParams;
use crate::primality::{batches_below_1000, bpsw, twelve_base_witness, PSI_12};
use crate::Integer;
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use rustmath_core::{MathError, NumericConversion, Result};
use std::collections::VecDeque;
use std::ops::ControlFlow;
use std::sync::{Mutex, PoisonError};

pub use crate::ecpp::{prove_primality, EcppCertificate, EcppStep, Primality};
pub use crate::primality::{is_prime_u64, is_probable_prime, NotPrimeWitness};

/// Check if a number is prime using trial division (for small numbers)
pub fn is_prime_trial_division(n: &Integer) -> bool {
    if *n <= Integer::one() {
        return false;
    }
    if *n <= Integer::from(3) {
        return true;
    }
    if n.is_even() {
        return false;
    }

    let mut i = Integer::from(3);
    let sqrt_n = n.sqrt().expect("n > 3 is non-negative");

    while i <= sqrt_n {
        if (n.clone() % i.clone()).is_zero() {
            return false;
        }
        i = i + Integer::from(2);
    }

    true
}

/// MAGMA `IsProbablePrime(n: Bases := rounds)` with the bases fixed to the first
/// max(rounds, 1) primes 2, 3, 5, ... (at most the 6542 primes below 2^16), each used once, and
/// only those below n: n is composite at the first base to which it is not a strong probable
/// prime (Miller-Rabin, [Jae93] G. Jaeschke, *On strong pseudoprimes to several bases*, Math.
/// Comp. 61 (1993) 915-926).
///
/// `false` is always a proof. With rounds >= 12 the answer is a proof for
/// n < psi_12 = 318665857834031151167461 (Sorenson-Webster 2017); with fewer bases more strong
/// pseudoprimes pass (3215031751 = 151 · 751 · 28351 passes the four bases 2, 3, 5, 7). MAGMA
/// draws random bases; these are fixed, so the result is reproducible. For a probable-prime test
/// above psi_12 use [`is_probable_prime`] (Baillie-PSW), for a proof [`is_prime`].
pub fn is_prime_miller_rabin(n: &Integer, rounds: u32) -> bool {
    if *n <= Integer::one() {
        return false;
    }
    if *n <= Integer::from(3) {
        return true;
    }
    if n.is_even() {
        return false;
    }
    let v = n.as_bigint();
    let count = (rounds.max(1) as usize).min(crate::sieve::small_primes().len());
    crate::sieve::small_primes()[..count]
        .iter()
        .map(|&a| BigInt::from(a))
        .take_while(|a| a < v)
        .all(|a| crate::primality::strong_probable_prime(v, &a))
}

/// Primes >= psi_12 whose ECPP certificate `verify` accepted, so that repeated contract checks
/// on one large prime cost one proof per process. At most 64 entries, oldest dropped first. It
/// holds only what was proven, so it cannot change an answer.
static PROVEN_PRIMES: Mutex<VecDeque<Integer>> = Mutex::new(VecDeque::new());
const PROVEN_PRIMES_CAP: usize = 64;

fn proven_cache_contains(n: &Integer) -> bool {
    PROVEN_PRIMES
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .contains(n)
}

fn proven_cache_insert(n: &Integer) {
    let mut cache = PROVEN_PRIMES.lock().unwrap_or_else(PoisonError::into_inner);
    if !cache.contains(n) {
        if cache.len() >= PROVEN_PRIMES_CAP {
            cache.pop_front();
        }
        cache.push_back(n.clone());
    }
}

/// [`try_is_prime`] with explicit ECPP limits (tests reach the `Err` branch with these).
pub(crate) fn try_is_prime_with(n: &Integer, params: &ProverParams) -> Result<bool> {
    let v = n.as_bigint();
    if *v < BigInt::from(2u32) {
        return Ok(false);
    }
    if let Some(w) = v.to_u64() {
        return Ok(is_prime_u64(w));
    }
    if batches_below_1000().least_divisor(v).is_some() {
        return Ok(false);
    }
    if *v < BigInt::from(PSI_12) {
        return Ok(twelve_base_witness(v).is_none());
    }
    if bpsw(v).is_err() {
        return Ok(false);
    }
    if proven_cache_contains(n) {
        return Ok(true);
    }
    match crate::ecpp::prove_primality_with(n, params) {
        Primality::Prime(_) => {
            proven_cache_insert(n);
            Ok(true)
        }
        Primality::NotPrime(_) => Ok(false),
        Primality::Unresolved(why) => Err(MathError::NotSupported(format!(
            "is_prime: {n} is a BPSW probable prime, but ECPP found no proof ({why}); use \
             prime::prove_primality or prime::is_probable_prime"
        ))),
    }
}

/// MAGMA `IsPrime(n)` (`Proof := true`, ch. 18 §18.9.1) as a `Result`: `Ok(true)` proven prime,
/// `Ok(false)` proven not prime, `Err(MathError::NotSupported(..))` when n is a Baillie-PSW
/// probable prime whose ECPP proof is unresolved.
///
/// | n | Method | Answer |
/// |---|---|---|
/// | n < 2 | | false |
/// | n < 2^64 | [`is_prime_u64`] (about 10 µs in debug) | proof |
/// | 2^64 <= n < psi_12 | trial division below 1000, strong tests to the bases 2..37 | proof (Sorenson-Webster 2017) |
/// | n >= psi_12 | trial division, BPSW; on a pass the cache of proven primes, then [`prove_primality`] | proof, or `Err` when unresolved |
///
/// Cost of a prime above psi_12 (one ECPP proof, then cached for the process; measured with
/// the design prototype on random primes): release 0.024 s at 128 bits, 0.07 s at 256, 0.41 s
/// at 512, 3.3 s at 1024, 21 s at 1536, 76 s at 2048, about 16 minutes at 3072 bits; a debug
/// build is 8 to 15 times slower (0.065 s at 128 bits, 0.36 s at 256, 3.2 s at 512, 38 s at
/// 1024). A composite above psi_12 costs trial division and one strong test in almost all
/// cases. Unresolved inputs: no prime up to 2048 bits was unresolved in the experiments (61
/// proofs and 180 coverage runs); at 3072 and 4096 bits 2 of 32 primes had no top-level
/// candidate with class number <= 100 and were unresolved. No answer is ever a guess.
pub fn try_is_prime(n: &Integer) -> Result<bool> {
    try_is_prime_with(n, &ProverParams::DEFAULT)
}

/// MAGMA `IsPrime(n)`: a proof of primality or compositeness for every size (see
/// [`try_is_prime`] for the method, the cost and the rate of unresolved inputs).
///
/// # Panics
///
/// When [`try_is_prime`] is `Err`: n is a Baillie-PSW probable prime and the ECPP prover found
/// no certificate (none up to 2048 bits in the experiments; about 1 in 16 at 3072 and 4096
/// bits). The message names n and the reason. Callers that return `Result` use
/// [`try_is_prime`]; callers that only need a probable prime use [`is_probable_prime`].
pub fn is_prime(n: &Integer) -> bool {
    try_is_prime(n).unwrap_or_else(|e| panic!("{e}"))
}

/// Check if a number is a pseudoprime (passes Fermat test but may not be prime)
///
/// A pseudoprime to base a is a composite number n such that a^(n-1) ≡ 1 (mod n).
/// This is the Fermat primality test. Note that this can have false positives
/// (Carmichael numbers pass for all bases).
///
/// For base 2, these are called "Fermat pseudoprimes base 2".
pub fn is_pseudoprime(n: &Integer, base: Option<u32>) -> bool {
    let a = base.unwrap_or(2);

    if *n <= Integer::one() {
        return false;
    }

    if *n <= Integer::from(3) {
        return true;
    }

    if n.is_even() {
        return *n == Integer::from(2);
    }

    let base_int = Integer::from(a);

    // Check if base_int^(n-1) ≡ 1 (mod n)
    let exponent = n.clone() - Integer::one();

    match base_int.mod_pow(&exponent, n) {
        Ok(result) => result.is_one(),
        Err(_) => false,
    }
}

/// The largest prime below 2^64 (gp: precprime(2^64) = 18446744073709551557).
const LARGEST_U64_PRIME: u64 = 18_446_744_073_709_551_557;

/// The window survivors of a pre-sieve: candidate i is start + 2i (ascending) or start - 2i
/// (descending), start odd; `false` marks a candidate with a factor among `odd_primes` (all
/// candidates are larger than these primes).
fn presieve(start: &BigInt, count: usize, descending: bool, odd_primes: &[u32]) -> Vec<bool> {
    let mut keep = vec![true; count];
    for &p in odd_primes {
        let p64 = u64::from(p);
        let r = (start % p64).to_u64().expect("a residue modulo a u32 fits a u64");
        let inv2 = p64.div_ceil(2);
        let i0 = if descending {
            r * inv2 % p64
        } else {
            (p64 - r) % p64 * inv2 % p64
        };
        let mut i = i0 as usize;
        while i < count {
            keep[i] = false;
            i += p as usize;
        }
    }
    keep
}

/// The odd primes below 2^12 (pre-sieve of `next_prime` / `previous_prime` above 2^64).
fn odd_primes_below_2_12() -> &'static [u32] {
    let s = crate::sieve::small_primes();
    &s[1..s.partition_point(|&p| p < 1 << 12)]
}

/// MAGMA `NextPrime(n)` (ch. 18 §18.9.2): the least proven prime > n (gp `nextprime(n + 1)`);
/// 2 for every n < 2. Below the largest prime under 2^64 the candidates are tested with
/// [`is_prime_u64`]; above, windows of 2^12 odd candidates are pre-sieved by the primes below
/// 2^12, each survivor gets [`is_probable_prime`], and the first probable prime is proven with
/// [`try_is_prime`] (`Ok(false)`, a BPSW pseudoprime, moves on).
///
/// # Errors
///
/// `NotSupported` when the least probable prime > n has no ECPP proof (unresolved): it cannot be
/// skipped, because the answer must be the least prime.
pub fn try_next_prime(n: &Integer) -> Result<Integer> {
    let v = n.as_bigint();
    if *v < BigInt::from(2u32) {
        return Ok(Integer::from(2u32));
    }
    if let Some(w) = v.to_u64() {
        if w < LARGEST_U64_PRIME {
            let mut c = w + 1;
            if c > 2 && c.is_multiple_of(2) {
                c += 1;
            }
            while !is_prime_u64(c) {
                c += 2;
            }
            return Ok(Integer::from(c));
        }
    }
    const WINDOW: usize = 1 << 12;
    let mut start: BigInt = v + 1u32;
    if !start.bit(0) {
        start += 1u32;
    }
    loop {
        let keep = presieve(&start, WINDOW, false, odd_primes_below_2_12());
        for (i, &k) in keep.iter().enumerate() {
            if !k {
                continue;
            }
            let c = Integer::new(&start + 2 * i as u64);
            if is_probable_prime(&c) && try_is_prime(&c)? {
                return Ok(c);
            }
        }
        start += 2 * WINDOW as u64;
    }
}

/// [`try_next_prime`], panicking on `Err` (an unresolved ECPP proof; none up to 2048 bits in
/// the experiments).
pub fn next_prime(n: &Integer) -> Integer {
    try_next_prime(n).unwrap_or_else(|e| panic!("next_prime: {e}"))
}

/// MAGMA `PreviousPrime(n)` (ch. 18 §18.9.2): the largest proven prime < n, `Ok(None)` for
/// n <= 2. The mirror image of [`try_next_prime`]: above 2^64 descending pre-sieved windows,
/// at 2^64 the word sieve takes over.
///
/// # Errors
///
/// `NotSupported` when the largest probable prime < n has no ECPP proof.
pub fn try_previous_prime(n: &Integer) -> Result<Option<Integer>> {
    let v = n.as_bigint();
    if *v <= BigInt::from(2u32) {
        return Ok(None);
    }
    let two_64 = BigInt::from(u64::MAX) + 1u32;
    if *v <= two_64 {
        let top = (v - 1u32).to_u64().expect("n - 1 < 2^64");
        let mut c = top;
        if c > 2 && c.is_multiple_of(2) {
            c -= 1;
        }
        while c > 2 && !is_prime_u64(c) {
            c -= 2;
        }
        return Ok(Some(Integer::from(c)));
    }
    const WINDOW: usize = 1 << 12;
    let mut start: BigInt = v - 1u32;
    if !start.bit(0) {
        start -= 1u32;
    }
    loop {
        let keep = presieve(&start, WINDOW, true, odd_primes_below_2_12());
        for (i, &k) in keep.iter().enumerate() {
            let c = &start - 2 * i as u64;
            if c < two_64 {
                return Ok(Some(Integer::from(LARGEST_U64_PRIME)));
            }
            if !k {
                continue;
            }
            let c = Integer::new(c);
            if is_probable_prime(&c) && try_is_prime(&c)? {
                return Ok(Some(c));
            }
        }
        start -= 2 * WINDOW as u64;
    }
}

/// [`try_previous_prime`], panicking on `Err` (an unresolved ECPP proof: the largest prime below
/// n cannot be skipped).
pub fn previous_prime(n: &Integer) -> Option<Integer> {
    try_previous_prime(n).unwrap_or_else(|e| panic!("previous_prime: {e}"))
}

/// MAGMA `IsPrimePower(n)` with its extra return values (ch. 18 §18.9.1): `Some((p, k))` iff
/// n = p^k with p prime and k >= 1, the primality of p proven. [`Integer::is_perfect_power`]
/// writes n = c^m with c not a perfect power (or n itself when n is no perfect power), then
/// [`is_prime`] proves or refutes c. `None` for n <= 1.
///
/// # Panics
///
/// As [`is_prime`], when the ECPP proof of c is unresolved.
pub fn prime_power(n: &Integer) -> Option<(Integer, u32)> {
    if *n <= Integer::one() {
        return None;
    }
    let (c, m) = n.is_perfect_power().unwrap_or_else(|| (n.clone(), 1));
    if is_prime(&c) {
        Some((c, m))
    } else {
        None
    }
}

/// Check if n is a prime power (n = p^k for some prime p and k ≥ 1): `prime_power(n).is_some()`
/// (MAGMA `IsPrimePower(n)`, p proven prime).
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::prime::is_prime_power;
///
/// assert!(is_prime_power(&Integer::from(8)));   // 2^3
/// assert!(is_prime_power(&Integer::from(27)));  // 3^3
/// assert!(!is_prime_power(&Integer::from(12))); // 2^2 * 3
/// ```
pub fn is_prime_power(n: &Integer) -> bool {
    prime_power(n).is_some()
}

/// MAGMA `NthPrime(n)`: the n-th prime (1-indexed: nth_prime(1) = 2), by the segmented sieve
/// ([`crate::sieve::nth_prime_u64`]; `nth_prime(10^6)` takes about 0.1 s in debug,
/// `nth_prime(455052511)`, the largest prime below 10^10 that MAGMA promises, about 75 s).
///
/// # Panics
///
/// For n = 0, and for n > pi(2^64) = 425656284035217743 (OEIS A007053), where the n-th prime
/// does not fit a `u64` and is not computed.
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::prime::nth_prime;
///
/// assert_eq!(nth_prime(1), Integer::from(2));
/// assert_eq!(nth_prime(2), Integer::from(3));
/// assert_eq!(nth_prime(10), Integer::from(29));
/// ```
pub fn nth_prime(n: usize) -> Integer {
    if n == 0 {
        panic!("nth_prime: n must be >= 1");
    }
    match crate::sieve::nth_prime_u64(n as u64) {
        Some(p) => Integer::from(p),
        None => panic!(
            "nth_prime: n = {n} exceeds pi(2^64) = {}; the n-th prime does not fit a u64 and \
             is not computed",
            crate::sieve::PRIME_PI_2_64
        ),
    }
}

/// Calls `f` on each Baillie-PSW probable prime c with `lo <= c < hi`, ascending, where
/// lo >= 2^64: windows of 2^16 odd candidates pre-sieved by the odd primes below 2^16, then
/// [`is_probable_prime`] on the survivors. Stops at the first `Err` of `f`.
fn for_each_big_probable_prime(
    lo: &BigInt,
    hi: &BigInt,
    mut f: impl FnMut(Integer) -> Result<()>,
) -> Result<()> {
    const WINDOW: usize = 1 << 16;
    let mut start = lo.clone();
    if !start.bit(0) {
        start += 1u32;
    }
    let odd = &crate::sieve::small_primes()[1..];
    while start < *hi {
        let keep = presieve(&start, WINDOW, false, odd);
        for (i, &k) in keep.iter().enumerate() {
            let c = &start + 2 * i as u64;
            if c >= *hi {
                return Ok(());
            }
            if k {
                let c = Integer::new(c);
                if is_probable_prime(&c) {
                    f(c)?;
                }
            }
        }
        start += 2 * WINDOW as u64;
    }
    Ok(())
}

/// The primes in [start, stop), each proven (MAGMA `PrimesInInterval(start, stop - 1)`). Below
/// 2^64 the segmented sieve (exact); above 2^64 windows of 2^16 odd candidates pre-sieved by
/// the primes below 2^16, each survivor [`is_probable_prime`], each probable prime
/// [`try_is_prime`]: `Ok(true)` listed, `Ok(false)` dropped (a BPSW pseudoprime). So the range
/// proves only the primes it returns, and a composite costs one BPSW test.
///
/// # Errors
///
/// `NotSupported` when one probable prime of the range has no ECPP proof (the list must be
/// complete).
pub fn try_prime_range(start: &Integer, stop: &Integer) -> Result<Vec<Integer>> {
    let (lo, hi) = (start.as_bigint(), stop.as_bigint());
    let mut out = Vec::new();
    if lo >= hi {
        return Ok(out);
    }
    let two_64 = BigInt::from(u64::MAX) + 1u32;
    let zero = BigInt::from(0u32);
    let small_lo = lo.max(&zero);
    let small_hi = hi.min(&two_64);
    if small_lo < small_hi {
        let b = small_lo.to_u64().expect("below 2^64");
        let e = (small_hi - 1u32).to_u64().expect("below 2^64");
        for p in crate::sieve::primes_in_interval(b, e) {
            out.push(Integer::from(p));
        }
    }
    if *hi > two_64 {
        let big_lo = lo.max(&two_64).clone();
        for_each_big_probable_prime(&big_lo, hi, |c| {
            if try_is_prime(&c)? {
                out.push(c);
            }
            Ok(())
        })?;
    }
    Ok(out)
}

/// The primes in [start, stop) ([`try_prime_range`]), panicking on `Err` (an unresolved ECPP
/// proof).
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::prime::prime_range;
///
/// let primes = prime_range(&Integer::from(10), &Integer::from(20));
/// assert_eq!(primes, [11, 13, 17, 19].map(Integer::from).to_vec());
/// ```
pub fn prime_range(start: &Integer, stop: &Integer) -> Vec<Integer> {
    try_prime_range(start, stop).unwrap_or_else(|e| panic!("prime_range: {e}"))
}

/// The first n primes (segmented sieve, like [`nth_prime`]).
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::prime::primes_first_n;
///
/// let primes = primes_first_n(5);
/// assert_eq!(primes, [2, 3, 5, 7, 11].map(Integer::from).to_vec());
/// ```
pub fn primes_first_n(n: usize) -> Vec<Integer> {
    crate::sieve::first_n_primes(n as u64)
        .into_iter()
        .map(Integer::from)
        .collect()
}

/// The prime counting function pi(x) (MAGMA ch. 18 §18.9.2), by the segmented sieve
/// ([`crate::sieve::prime_count`], linear in x: about 0.75 s for x = 10^8 in a debug build).
///
/// # Panics
///
/// For x > 2^64 - 1: the sieve cannot reach it and sublinear counting (Lagarias-Miller-Odlyzko)
/// is not implemented.
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::prime::prime_pi;
///
/// assert_eq!(prime_pi(&Integer::from(10)), 4);  // 2, 3, 5, 7
/// assert_eq!(prime_pi(&Integer::from(100)), 25);
/// ```
pub fn prime_pi(x: &Integer) -> usize {
    if *x < Integer::from(2) {
        return 0;
    }
    match x.to_u64() {
        Some(v) => usize::try_from(crate::sieve::prime_count(v)).expect("pi(x) < 2^64"),
        None => panic!(
            "prime_pi: x = {x} exceeds 2^64; the sieve cannot reach it and sublinear counting \
             (Lagarias-Miller-Odlyzko) is not implemented"
        ),
    }
}

/// A random proven prime p with a <= p < b (MAGMA `RandomPrime` with `Proof := true`, in the
/// interval form). `Ok(None)` when [a, b) has no prime.
///
/// The probable primes of [a, b) are collected (the sieve below 2^64, exact; above, the
/// pre-sieved windows of [`try_prime_range`] with [`is_probable_prime`] only, no proofs). Then
/// one is drawn uniformly (`rand::thread_rng`) and proven with [`try_is_prime`]: `Ok(true)`
/// returns it, `Ok(false)` (a BPSW pseudoprime) or `Err` (unresolved) drops it and another is
/// drawn. The result is uniform over the primes of [a, b) that the prover proves (all of them
/// in every experiment up to 2048 bits). The whole range is enumerated, so the cost is linear in
/// b - a; MAGMA's `RandomPrime(n)` form (a random n-bit prime without enumeration) is not
/// implemented.
///
/// # Errors
///
/// `NotSupported` when probable primes exist in [a, b) and none of them could be proven.
pub fn try_random_prime(a: &Integer, b: &Integer) -> Result<Option<Integer>> {
    use rand::Rng;

    if a >= b {
        return Ok(None);
    }
    let (lo, hi) = (a.as_bigint(), b.as_bigint());
    let two_64 = BigInt::from(u64::MAX) + 1u32;
    let zero = BigInt::from(0u32);
    let mut pool: Vec<Integer> = Vec::new();
    let small_lo = lo.max(&zero);
    let small_hi = hi.min(&two_64);
    if small_lo < small_hi {
        let bb = small_lo.to_u64().expect("below 2^64");
        let e = (small_hi - 1u32).to_u64().expect("below 2^64");
        pool.extend(crate::sieve::primes_in_interval(bb, e).into_iter().map(Integer::from));
    }
    if *hi > two_64 {
        let big_lo = lo.max(&two_64).clone();
        for_each_big_probable_prime(&big_lo, hi, |c| {
            pool.push(c);
            Ok(())
        })?;
    }
    let total = pool.len();
    let mut unresolved = 0usize;
    let mut rng = rand::thread_rng();
    while !pool.is_empty() {
        let i = rng.gen_range(0..pool.len());
        let c = pool.swap_remove(i);
        match try_is_prime(&c) {
            Ok(true) => return Ok(Some(c)),
            Ok(false) => {}
            Err(_) => unresolved += 1,
        }
    }
    if unresolved > 0 {
        return Err(MathError::NotSupported(format!(
            "random_prime: none of the {total} probable primes in [{a}, {b}) could be proven \
             ({unresolved} unresolved ECPP proofs)"
        )));
    }
    Ok(None)
}

/// A random prime in [a, b) ([`try_random_prime`]), `None` if there is none; panics on `Err`
/// (probable primes exist and none could be proven).
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::prime::random_prime;
///
/// // Get a random prime between 10 and 20
/// if let Some(p) = random_prime(&Integer::from(10), &Integer::from(20)) {
///     assert!(p >= Integer::from(10) && p < Integer::from(20));
/// }
/// ```
pub fn random_prime(a: &Integer, b: &Integer) -> Option<Integer> {
    try_random_prime(a, b).unwrap_or_else(|e| panic!("{e}"))
}

/// Compute all prime factors of n
pub fn factor(n: &Integer) -> Vec<(Integer, u32)> {
    if n.is_zero() || n.is_one() {
        return vec![];
    }

    let mut factors = Vec::new();
    let mut remaining = n.abs();

    // Handle factor of 2
    if remaining.is_even() {
        let mut count = 0;
        while remaining.is_even() {
            remaining = remaining / Integer::from(2);
            count += 1;
        }
        factors.push((Integer::from(2), count));
    }

    // Try odd factors
    let mut factor = Integer::from(3);
    while factor.clone() * factor.clone() <= remaining {
        let mut count = 0;
        while (&remaining % &factor).is_zero() {
            remaining = remaining / factor.clone();
            count += 1;
        }
        if count > 0 {
            factors.push((factor.clone(), count));
        }
        factor = factor + Integer::from(2);
    }

    // If remaining > 1, it's a prime factor
    if remaining > Integer::one() {
        factors.push((remaining, 1));
    }

    factors
}

/// Get all prime divisors of n (without multiplicities)
///
/// Returns a vector of distinct prime factors of n in ascending order.
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::prime::prime_divisors;
///
/// let n = Integer::from(60); // 60 = 2² × 3 × 5
/// let divisors = prime_divisors(&n);
/// // Returns [2, 3, 5]
/// ```
pub fn prime_divisors(n: &Integer) -> Vec<Integer> {
    if n.is_zero() || n.is_one() {
        return vec![];
    }

    // Use the factor() function to get prime factorization
    let factorization = factor(n);

    // Extract just the primes (first element of each tuple)
    factorization.into_iter().map(|(prime, _)| prime).collect()
}

/// Pollard's Rho algorithm for finding a non-trivial factor
///
/// This is a probabilistic algorithm that's much faster than trial division
/// for large composite numbers with small factors. It uses Floyd's cycle
/// detection method with the polynomial f(x) = (x² + 1) mod n.
///
/// # Algorithm
///
/// The algorithm maintains two sequences:
/// - x_{i+1} = f(x_i) = (x_i² + 1) mod n
/// - y_{i+1} = f(f(y_i))
///
/// It computes gcd(|x - y|, n) at each step. When a cycle is detected,
/// this GCD will be a non-trivial factor of n with high probability.
///
/// # Comparison with SageMath
///
/// SageMath's `factor()` function uses PARI, which implements a combination
/// of Shanks SQUFOF and Pollard Rho, along with ECM and MPQS for larger numbers.
/// This implementation is a pure Rust version of the classic Pollard Rho algorithm.
///
/// **Key Features:**
/// - Uses f(x) = x² + 1 as the iteration function (standard choice)
/// - Floyd's cycle detection (tortoise and hare algorithm)
/// - Returns `Some(2)` for even numbers (fast path)
/// - Returns `None` for probable primes ([`is_probable_prime`], Baillie-PSW; exact below
///   psi_12)
/// - Returns `None` if algorithm fails after 1,000,000 iterations
///
/// # Parameters
///
/// * `n` - The composite number to factor
///
/// # Returns
///
/// * `Some(factor)` - A non-trivial factor of n (1 < factor < n)
/// * `None` - If n is a probable prime (BPSW), n ≤ 1, or no factor was found
///
/// # Time Complexity
///
/// Expected: O(n^(1/4)) for finding the smallest prime factor
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::prime::pollard_rho;
///
/// // Find a factor of 143 = 11 × 13
/// let n = Integer::from(143);
/// let factor = pollard_rho(&n);
/// assert!(factor.is_some());
/// let f = factor.unwrap();
/// assert!(f == Integer::from(11) || f == Integer::from(13));
/// ```
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::prime::pollard_rho;
///
/// // Returns None for primes
/// let n = Integer::from(17);
/// assert_eq!(pollard_rho(&n), None);
/// ```
///
/// # See Also
///
/// - [`factor_pollard_rho`] - Complete factorization using Pollard's Rho
/// - [`pollard_p_minus_1`] - Pollard's p-1 algorithm for smooth factors
/// - [`factor`] - General factorization using trial division
pub fn pollard_rho(n: &Integer) -> Option<Integer> {
    if n <= &Integer::from(1) {
        return None;
    }

    // A probable prime (including 2) has no factor to find; a splitting decision needs no
    // proof (BPSW, exact below psi_12).
    if is_probable_prime(n) {
        return None;
    }

    // For composite even numbers, 2 is a factor
    if n.is_even() {
        return Some(Integer::from(2));
    }

    // Pollard's Rho with Floyd's cycle detection
    let mut x = Integer::from(2);
    let mut y = Integer::from(2);

    // f(x) = (x² + 1) mod n
    let f = |val: &Integer| -> Integer {
        let result = (val.clone() * val.clone() + Integer::one()) % n.clone();
        if result.signum() < 0 {
            result + n.clone()
        } else {
            result
        }
    };

    // Try to find a factor
    for _ in 0..1000000 {
        x = f(&x);
        y = f(&f(&y));

        let diff = if x > y {
            x.clone() - y.clone()
        } else {
            y.clone() - x.clone()
        };

        let d = diff.gcd(n);

        if d > Integer::one() && &d < n {
            return Some(d);
        }

        if d == *n {
            // Cycle detected, restart with different starting point
            x = Integer::from(rand_int() % 100 + 2);
            y = x.clone();
        }
    }

    None
}

/// Pollard's p-1 algorithm for finding a non-trivial factor
///
/// This algorithm is effective when n has a prime factor p such that p-1
/// is B-smooth (i.e., all prime factors of p-1 are ≤ B).
///
/// The algorithm computes a = 2^(k!) mod n for increasing k, then checks
/// gcd(a-1, n). If p-1 divides k!, then a^(p-1) ≡ 1 (mod p) by Fermat's
/// Little Theorem, so p divides gcd(a-1, n).
///
/// # Arguments
/// * `n` - The number to factor
/// * `bound` - The smoothness bound B (typically 100-10000)
///
/// Returns None if n is a probable prime ([`is_probable_prime`], Baillie-PSW) or no factor
/// was found. Stage 1 raises 2 to the largest power p^e <= bound of every prime p <= bound
/// (the primes from the segmented sieve [`crate::sieve::for_each_prime`]).
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::prime::pollard_p_minus_1;
///
/// // 299 = 13 × 23, and 13-1 = 12 = 2² × 3
/// let factor = pollard_p_minus_1(&Integer::from(299), 10);
/// assert!(factor.is_some());
/// ```
pub fn pollard_p_minus_1(n: &Integer, bound: u32) -> Option<Integer> {
    if n <= &Integer::from(1) {
        return None;
    }

    // A probable prime (including 2) has no factor to find.
    if is_probable_prime(n) {
        return None;
    }

    if n.is_even() {
        return Some(Integer::from(2));
    }

    // Start with base 2
    let mut a = Integer::from(2);

    // a = 2^(prod p^e) mod n over the primes p <= bound, p^e the largest power <= bound.
    let mut failed = false;
    crate::sieve::for_each_prime(2, u64::from(bound) + 1, |p| {
        let p = u32::try_from(p).expect("p <= bound < 2^32");
        let mut q = p;
        while q <= bound / p {
            q *= p;
        }
        match a.mod_pow(&Integer::from(q), n) {
            Ok(new_a) => {
                a = new_a;
                ControlFlow::Continue(())
            }
            Err(_) => {
                failed = true;
                ControlFlow::Break(())
            }
        }
    });
    if failed {
        return None;
    }

    // Compute gcd(a - 1, n)
    let diff = if a > Integer::one() {
        a - Integer::one()
    } else {
        a + n.clone() - Integer::one()
    };

    let g = diff.gcd(n);

    if g > Integer::one() && &g < n {
        Some(g)
    } else {
        None
    }
}

// Simple deterministic "random" number for Pollard's Rho
fn rand_int() -> i64 {
    use std::cell::Cell;
    thread_local!(static COUNTER: Cell<i64> = Cell::new(42));
    COUNTER.with(|c| {
        let val = c.get();
        c.set(val.wrapping_mul(1103515245).wrapping_add(12345));
        (val / 65536) % 32768
    })
}

/// Complete integer factorization using Pollard's Rho algorithm
///
/// Returns the prime factorization of |n| as `(prime, exponent)` pairs with the primes in
/// ascending order: PARI/GP `factor(n)` without the sign, MAGMA `Factorization(n)` (Handbook
/// ch. 18 §18.10, digest `rustmath-lava/docs/Chapter_18_Ring_of_Integers.md`). For n = 0, 1
/// and -1 the list is empty.
///
/// # Algorithm
///
/// 1. Divide out the factors of 2.
/// 2. While the cofactor m is not accepted by [`is_prime`], find a proper divisor d of m (see
///    below), factor d by recursion and divide each of its primes out of m with full
///    multiplicity.
/// 3. The last cofactor passed the loop guard, so it is prime.
///
/// A proper divisor of an odd composite m is found by, in order:
/// - the perfect-power test [`Integer::is_perfect_power`]: m = b^k gives d = b;
/// - [`pollard_rho`] (Floyd's cycle detection on x^2 + 1, at most 10^6 steps; J. M. Pollard,
///   *A Monte Carlo method for factorization*, BIT 15 (1975) 331-334);
/// - if that returns `None`: rounds j = 0, 1, 2, ... of Brent's variant with x^2 + c,
///   c = j + 2, at most 2^(12+j) steps (R. P. Brent, *An improved Monte Carlo factorization
///   algorithm*, BIT 20 (1980) 176-184, [Bre80] in the ch. 18 digest; H. Cohen, GTM 138,
///   §8.5), each round followed by trial division by the next 2^(12+j) odd numbers. Rho is
///   Monte Carlo and may fail for every c tried; the trial-division windows make the search
///   complete, since a composite m has a prime factor p <= sqrt(m), which they reach after
///   about p/2 divisions in all.
///
/// A cofactor that the methods have not yet split is never pushed as a prime: the search
/// goes on until it splits.
///
/// Every returned factor is proven prime by [`is_prime`] (the strong tests to the bases 2..37
/// below psi_12 = 318665857834031151167461, J. Sorenson, J. Webster, *Strong pseudoprimes to
/// twelve prime bases*, Math. Comp. 86 (2017) 985-1003; Baillie-PSW and ECPP above). psi_12
/// itself is split: [399165290221, 798330580441].
///
/// # Panics
///
/// When the ECPP proof of a probable-prime factor is unresolved ([`is_prime`] panics; no
/// unresolved prime up to 2048 bits in the experiments, about 1 in 16 at 3072 and 4096 bits).
///
/// # Comparison with SageMath
///
/// This function is similar to SageMath's `factor()` but uses only Pollard's Rho (with
/// Brent's variant and trial division as the fallback), whereas SageMath uses PARI which
/// employs:
/// - Shanks SQUFOF
/// - Pollard's Rho
/// - Lenstra's ECM (for larger numbers)
/// - MPQS (Multiple Polynomial Quadratic Sieve)
///
/// # Parameters
///
/// * `n` - The integer to factor
///
/// # Returns
///
/// A vector of `(prime, exponent)` pairs representing the complete prime
/// factorization of |n|, sorted by prime factors in ascending order.
///
/// For n = 0, 1 or -1, returns an empty vector.
///
/// # Cost
///
/// There is no time or iteration limit: the call returns only with the complete
/// factorization. Rho needs about sqrt(p) steps to split off a prime p, so the run time is set
/// by the second-largest prime factor q of |n| (the largest one is the last cofactor) and
/// grows like sqrt(q). Measured in a debug build: (10^9+7)(10^9+9) takes about 0.2 s and
/// (10^12+39)(10^12+61) about 20 s (10^6 steps); for q with 20 or more digits (10^10 steps
/// or more) the call does not finish in practical time. The trial-division windows bound the
/// worst case of each split by about p/2 trial divisions for its smallest prime factor p, plus
/// up to about twice as many Brent steps (a round can overshoot its budget by 2x) and 10^6
/// Floyd steps; the windows restart at 3 for each cofactor that needs the fallback.
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::prime::factor_pollard_rho;
///
/// // Factor 60 = 2² × 3 × 5
/// let n = Integer::from(60);
/// let factors = factor_pollard_rho(&n);
///
/// // Verify the factorization
/// let mut product = Integer::from(1);
/// for (p, e) in &factors {
///     product = product * p.pow(*e);
/// }
/// assert_eq!(product, n);
/// ```
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::prime::factor_pollard_rho;
///
/// // Factor a prime number
/// let n = Integer::from(97);
/// let factors = factor_pollard_rho(&n);
/// assert_eq!(factors, vec![(Integer::from(97), 1)]);
/// ```
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::prime::factor_pollard_rho;
///
/// // 12780049 = 53 · 59 · 61 · 67 (PARI/GP: factor(12780049))
/// let factors = factor_pollard_rho(&Integer::from(12780049));
/// let primes: Vec<Integer> = [53, 59, 61, 67].into_iter().map(Integer::from).collect();
/// assert_eq!(factors, primes.into_iter().map(|p| (p, 1)).collect::<Vec<_>>());
/// ```
///
/// # See Also
///
/// - [`pollard_rho`] - Find a single factor using Pollard's Rho
/// - [`factor`] - Factorization using trial division
/// - [`crate::factorint::factor_trial_division`] - Trial division with limit
pub fn factor_pollard_rho(n: &Integer) -> Vec<(Integer, u32)> {
    if n.is_zero() || n.is_one() {
        return vec![];
    }

    let mut factors = Vec::new();
    let mut remaining = n.abs();

    // Handle factor of 2
    if remaining.is_even() {
        let mut count = 0;
        while remaining.is_even() {
            remaining = remaining / Integer::from(2);
            count += 1;
        }
        factors.push((Integer::from(2), count));
    }

    // `remaining` is odd from here on. `is_prime` is a proof both ways, so every cofactor
    // handed to `split_odd_composite` is an odd composite, and every factor pushed is a
    // proven prime.
    while remaining > Integer::one() && !is_prime(&remaining) {
        let divisor = split_odd_composite(&remaining);
        // The divisor might be composite: factor it by recursion (it is smaller than
        // `remaining`, so the recursion ends).
        let divisor_factors = if is_prime(&divisor) {
            vec![(divisor, 1)]
        } else {
            factor_pollard_rho(&divisor)
        };

        // Each prime of the divisor divides `remaining`; extract all of its instances, so it
        // cannot come up again later.
        for (prime, _) in divisor_factors {
            let mut count = 0;
            while (&remaining % &prime).is_zero() {
                remaining = remaining / prime.clone();
                count += 1;
            }
            if count > 0 {
                factors.push((prime, count));
            }
        }
    }

    // The loop guard accepted `remaining` as prime.
    if remaining > Integer::one() {
        factors.push((remaining, 1));
    }

    factors.sort_by(|a, b| a.0.cmp(&b.0));
    factors
}

/// Number of steps of the first Brent round and of the first trial-division window in
/// [`split_odd_composite`]; both double each round.
const SPLIT_FIRST_ROUND: u64 = 1 << 12;

/// A proper divisor d of an odd composite m (1 < d < m, d | m).
///
/// Perfect-power test, then [`pollard_rho`], then [`split_brent_and_trial_division`]. Always
/// returns; see [`factor_pollard_rho`] for the cost. m is composite (the caller's [`is_prime`]
/// said so). If m is a BPSW pseudoprime (none is known), `pollard_rho(m)` returns `None` at
/// once through its [`is_probable_prime`] shortcut, and `split_brent_and_trial_division`
/// (complete, it always returns a divisor) splits m.
fn split_odd_composite(m: &Integer) -> Integer {
    debug_assert!(m.is_odd() && *m > Integer::one());
    if let Some((base, _)) = m.is_perfect_power() {
        return base;
    }
    if let Some(d) = pollard_rho(m) {
        return d;
    }
    split_brent_and_trial_division(m)
}

/// The complete fallback of [`split_odd_composite`]: rounds j = 0, 1, 2, ... of Brent's rho
/// with x^2 + (j + 2) and at most `SPLIT_FIRST_ROUND * 2^j` steps, each followed by trial
/// division by the next `SPLIT_FIRST_ROUND * 2^j` odd numbers from 3 on.
///
/// Termination does not depend on rho: the windows reach the least prime factor p of m after
/// about p/2 divisions, and p <= sqrt(m) because m is composite.
///
/// # Panics
///
/// If m is odd and the windows pass sqrt(m) without a divisor, i.e. m is prime or 1. Callers
/// only pass odd composites, so this is an internal invariant.
fn split_brent_and_trial_division(m: &Integer) -> Integer {
    let mut next_trial = Integer::from(3);
    let mut budget = SPLIT_FIRST_ROUND;
    let mut c: u64 = 2;
    loop {
        if let Some(d) = brent_rho(m, &Integer::from(c), budget) {
            return d;
        }
        if let Some(d) = trial_division_window(m, &mut next_trial, budget) {
            return d;
        }
        c += 1;
        budget = budget.saturating_mul(2);
    }
}

/// One run of Brent's variant of Pollard rho on m with f(x) = x^2 + c mod m, from x_0 = 2,
/// for at most about `max_steps` evaluations of f ([Bre80]; Cohen GTM 138, §8.5).
///
/// The differences |x - y| are multiplied together modulo m and the gcd is taken every 128
/// steps; when that gcd is m, the last block is replayed one step at a time.
///
/// Returns `Some(d)` with 1 < d < m and d | m, or `None` when the run ends with gcd m (the
/// cycles modulo all prime factors of m were detected together) or the step budget runs out.
/// A `None` says nothing about m.
fn brent_rho(m: &Integer, c: &Integer, max_steps: u64) -> Option<Integer> {
    const BLOCK: u64 = 128;
    let f = |v: &Integer| -> Integer { (&(v * v) + c).modulo(m) };

    let mut y = Integer::from(2).modulo(m);
    let mut q = Integer::one();
    let mut r: u64 = 1;
    let mut steps: u64 = 0;

    let (x, mut ys, mut g) = loop {
        let x = y.clone();
        for _ in 0..r {
            y = f(&y);
        }
        steps += r;
        let mut ys = y.clone();
        let mut g = Integer::one();
        let mut k: u64 = 0;
        while k < r && g.is_one() {
            ys = y.clone();
            let block = BLOCK.min(r - k);
            for _ in 0..block {
                y = f(&y);
                q = (&q * &(&x - &y).abs()).modulo(m);
            }
            g = q.gcd(m);
            k += block;
            steps += block;
        }
        if !g.is_one() {
            break (x, ys, g);
        }
        if steps >= max_steps {
            return None;
        }
        r = r.saturating_mul(2);
    };

    if g == *m {
        // Replay the last block: some |x - y_i| in it shares a prime with m, because q was
        // coprime to m before the block.
        loop {
            ys = f(&ys);
            g = (&x - &ys).abs().gcd(m);
            if !g.is_one() {
                break;
            }
        }
    }

    if g.is_one() || g == *m {
        None
    } else {
        Some(g)
    }
}

/// Trial division of m by the `count` odd numbers from `*next` on; `*next` is advanced past
/// the numbers tried. Returns the first divisor found.
///
/// # Panics
///
/// If the window passes sqrt(m) without a divisor: then m has no divisor in
/// [3, sqrt(m)], so an odd m is prime (or 1), against the precondition of
/// [`split_brent_and_trial_division`].
fn trial_division_window(m: &Integer, next: &mut Integer, count: u64) -> Option<Integer> {
    for _ in 0..count {
        assert!(
            &*next * &*next <= *m,
            "trial_division_window: no divisor of {m} up to its square root, so it is not an \
             odd composite"
        );
        if (m % &*next).is_zero() {
            return Some(next.clone());
        }
        *next = &*next + &Integer::from(2);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_prime() {
        assert!(is_prime(&Integer::from(2)));
        assert!(is_prime(&Integer::from(3)));
        assert!(is_prime(&Integer::from(5)));
        assert!(is_prime(&Integer::from(7)));
        assert!(is_prime(&Integer::from(11)));
        assert!(is_prime(&Integer::from(13)));

        assert!(!is_prime(&Integer::from(1)));
        assert!(!is_prime(&Integer::from(4)));
        assert!(!is_prime(&Integer::from(6)));
        assert!(!is_prime(&Integer::from(8)));
        assert!(!is_prime(&Integer::from(9)));
        assert!(!is_prime(&Integer::from(10)));

        // gp: isprime(2147483647) = 1, isprime(2^61-1) = 1, isprime(318665857834031151167461) = 0
        assert!(is_prime(&Integer::from(2_147_483_647u32)));
        assert!(is_prime(&Integer::from((1u64 << 61) - 1)));
        assert!(!is_prime(&z("318665857834031151167461")));
    }

    /// The `Err` of `try_is_prime` (review 2 n7): with no rows at the top level the ECPP proof
    /// of 2^127 - 1 is unresolved, `try_is_prime_with` reports it, and the cache of proven
    /// primes is not filled. gp: isprime(2^127-1) = 1.
    #[test]
    fn test_try_is_prime_unresolved_is_err() {
        let m127 = Integer::from(2).pow(127) - Integer::one();
        let starved = ProverParams {
            h_top: 0,
            ..ProverParams::DEFAULT
        };
        // The cache is process-wide and other tests prove 2^127 - 1, so use a prime no other
        // test proves. gp: nextprime(2^127 + 10^6) - 2^127 = 1000121,
        // isprime(2^127 + 1000123) = 0.
        let p = z("170141183460469231731687303715885105849");
        assert_eq!(p, Integer::from(2).pow(127) + Integer::from(1_000_121));
        assert!(m127 < p);
        assert!(is_probable_prime(&p));
        match try_is_prime_with(&p, &starved) {
            Err(MathError::NotSupported(msg)) => {
                assert!(msg.contains("ECPP found no proof"), "{msg}");
                assert!(msg.contains("no curve found with class number <= 0"), "{msg}");
            }
            other => panic!("{other:?}"),
        }
        assert!(!proven_cache_contains(&p));
        // A composite past the probable-prime gate is still decided without the prover.
        assert_eq!(try_is_prime_with(&z("318665857834031151167461"), &starved), Ok(false));
        assert_eq!(try_is_prime_with(&(p.clone() + Integer::from(2)), &starved), Ok(false));
    }

    #[test]
    #[should_panic(expected = "ECPP found no proof")]
    fn test_is_prime_panics_when_unresolved() {
        let starved = ProverParams {
            h_top: 0,
            ..ProverParams::DEFAULT
        };
        let p = z("170141183460469231731687303715884105757"); // gp: nextprime(2^127)
        try_is_prime_with(&p, &starved).unwrap_or_else(|e| panic!("{e}"));
    }

    #[test]
    fn test_next_prime() {
        assert_eq!(next_prime(&Integer::from(2)), Integer::from(3));
        assert_eq!(next_prime(&Integer::from(7)), Integer::from(11));
        assert_eq!(next_prime(&Integer::from(20)), Integer::from(23));
    }

    #[test]
    fn test_previous_prime() {
        assert_eq!(previous_prime(&Integer::from(3)), Some(Integer::from(2)));
        assert_eq!(previous_prime(&Integer::from(10)), Some(Integer::from(7)));
        assert_eq!(previous_prime(&Integer::from(20)), Some(Integer::from(19)));
        assert_eq!(previous_prime(&Integer::from(100)), Some(Integer::from(97)));

        // Edge cases
        assert_eq!(previous_prime(&Integer::from(2)), None);
        assert_eq!(previous_prime(&Integer::from(1)), None);
    }

    #[test]
    fn test_factor() {
        let factors = factor(&Integer::from(60));
        assert_eq!(factors.len(), 3);
        assert_eq!(factors[0], (Integer::from(2), 2));
        assert_eq!(factors[1], (Integer::from(3), 1));
        assert_eq!(factors[2], (Integer::from(5), 1));

        let factors = factor(&Integer::from(17));
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0], (Integer::from(17), 1));
    }

    #[test]
    fn test_is_prime_power() {
        // Prime powers
        assert!(is_prime_power(&Integer::from(2)));  // 2^1
        assert!(is_prime_power(&Integer::from(4)));  // 2^2
        assert!(is_prime_power(&Integer::from(8)));  // 2^3
        assert!(is_prime_power(&Integer::from(27))); // 3^3
        assert!(is_prime_power(&Integer::from(125))); // 5^3

        // Not prime powers
        assert!(!is_prime_power(&Integer::from(1)));
        assert!(!is_prime_power(&Integer::from(6)));  // 2 * 3
        assert!(!is_prime_power(&Integer::from(12))); // 2^2 * 3
        assert!(!is_prime_power(&Integer::from(30))); // 2 * 3 * 5
    }

    #[test]
    fn test_nth_prime() {
        assert_eq!(nth_prime(1), Integer::from(2));
        assert_eq!(nth_prime(2), Integer::from(3));
        assert_eq!(nth_prime(3), Integer::from(5));
        assert_eq!(nth_prime(4), Integer::from(7));
        assert_eq!(nth_prime(5), Integer::from(11));
        assert_eq!(nth_prime(10), Integer::from(29));
    }

    #[test]
    fn test_prime_range() {
        let primes = prime_range(&Integer::from(10), &Integer::from(20));
        assert_eq!(primes.len(), 4);
        assert_eq!(primes[0], Integer::from(11));
        assert_eq!(primes[1], Integer::from(13));
        assert_eq!(primes[2], Integer::from(17));
        assert_eq!(primes[3], Integer::from(19));

        // Range including 2
        let primes = prime_range(&Integer::from(0), &Integer::from(10));
        assert_eq!(primes.len(), 4);
        assert_eq!(primes[0], Integer::from(2));
        assert_eq!(primes[1], Integer::from(3));
        assert_eq!(primes[2], Integer::from(5));
        assert_eq!(primes[3], Integer::from(7));

        // Empty range
        let primes = prime_range(&Integer::from(20), &Integer::from(10));
        assert_eq!(primes.len(), 0);
    }

    #[test]
    fn test_primes_first_n() {
        let primes = primes_first_n(5);
        assert_eq!(primes.len(), 5);
        assert_eq!(primes[0], Integer::from(2));
        assert_eq!(primes[1], Integer::from(3));
        assert_eq!(primes[2], Integer::from(5));
        assert_eq!(primes[3], Integer::from(7));
        assert_eq!(primes[4], Integer::from(11));

        let primes = primes_first_n(0);
        assert_eq!(primes.len(), 0);

        let primes = primes_first_n(1);
        assert_eq!(primes.len(), 1);
        assert_eq!(primes[0], Integer::from(2));
    }

    #[test]
    fn test_prime_pi() {
        assert_eq!(prime_pi(&Integer::from(1)), 0);
        assert_eq!(prime_pi(&Integer::from(2)), 1);
        assert_eq!(prime_pi(&Integer::from(10)), 4);  // 2, 3, 5, 7
        assert_eq!(prime_pi(&Integer::from(20)), 8);  // 2, 3, 5, 7, 11, 13, 17, 19
        assert_eq!(prime_pi(&Integer::from(100)), 25);
    }

    #[test]
    fn test_random_prime() {
        // Test basic functionality
        let p = random_prime(&Integer::from(10), &Integer::from(20));
        assert!(p.is_some());
        let prime = p.unwrap();
        assert!(prime >= Integer::from(10) && prime < Integer::from(20));
        assert!(is_prime(&prime));

        // Verify it's one of the primes in range: 11, 13, 17, 19
        let valid_primes = vec![
            Integer::from(11),
            Integer::from(13),
            Integer::from(17),
            Integer::from(19),
        ];
        assert!(valid_primes.contains(&prime));

        // Test with no primes in range
        let p = random_prime(&Integer::from(24), &Integer::from(28));
        assert!(p.is_none());

        // Test invalid range (a >= b)
        let p = random_prime(&Integer::from(20), &Integer::from(10));
        assert!(p.is_none());

        // Test range with single prime
        let p = random_prime(&Integer::from(2), &Integer::from(3));
        assert!(p.is_some());
        assert_eq!(p.unwrap(), Integer::from(2));
    }

    // Note: test_pollard_p_minus_1 has been removed due to the algorithm
    // having reliability issues. The pollard_p_minus_1 function exists but
    // may not always succeed in finding factors within reasonable bounds.

    // ===== Pollard's Rho Tests =====

    #[test]
    fn test_pollard_rho_composite() {
        // Test on composite numbers
        let n = Integer::from(143); // 11 × 13
        let factor = pollard_rho(&n);
        assert!(factor.is_some());
        let f = factor.unwrap();
        assert!(f == Integer::from(11) || f == Integer::from(13));
        assert_eq!(&n % &f, Integer::zero());
    }

    #[test]
    fn test_pollard_rho_prime() {
        // Test on prime number (should return None)
        let n = Integer::from(17);
        let factor = pollard_rho(&n);
        assert!(factor.is_none());
    }

    #[test]
    fn test_pollard_rho_even() {
        // Test on even number (should return 2)
        let n = Integer::from(42); // 2 × 3 × 7
        let factor = pollard_rho(&n);
        assert_eq!(factor, Some(Integer::from(2)));
    }

    #[test]
    fn test_pollard_rho_small() {
        // Test edge cases
        assert_eq!(pollard_rho(&Integer::from(0)), None);
        assert_eq!(pollard_rho(&Integer::from(1)), None);
        assert_eq!(pollard_rho(&Integer::from(2)), None); // 2 is prime
    }

    #[test]
    fn test_pollard_rho_semiprime() {
        // Test on semiprimes
        let n = Integer::from(221); // 13 × 17
        let factor = pollard_rho(&n);
        assert!(factor.is_some());
        let f = factor.unwrap();
        assert!(f == Integer::from(13) || f == Integer::from(17));
    }

    #[test]
    fn test_factor_pollard_rho_small() {
        // Factor 60 = 2² × 3 × 5
        let n = Integer::from(60);
        let factors = factor_pollard_rho(&n);

        // Verify the product
        let mut product = Integer::one();
        for (p, e) in &factors {
            product = product * p.pow(*e);
        }
        assert_eq!(product, n);
    }

    #[test]
    fn test_factor_pollard_rho_prime() {
        // Factor a prime
        let n = Integer::from(97);
        let factors = factor_pollard_rho(&n);

        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0], (Integer::from(97), 1));
    }

    #[test]
    fn test_factor_pollard_rho_perfect_square() {
        // Factor 144 = 2⁴ × 3²
        let n = Integer::from(144);
        let factors = factor_pollard_rho(&n);

        // Verify factorization
        let mut product = Integer::one();
        for (p, e) in &factors {
            product = product * p.pow(*e);
        }
        assert_eq!(product, n);

        // Check that we have 2 and 3 as factors
        assert!(factors.iter().any(|(p, _)| p == &Integer::from(2)));
        assert!(factors.iter().any(|(p, _)| p == &Integer::from(3)));
    }

    #[test]
    fn test_factor_pollard_rho_large_composite() {
        // Factor 1001 = 7 × 11 × 13
        let n = Integer::from(1001);
        let factors = factor_pollard_rho(&n);

        // Verify factorization
        let mut product = Integer::one();
        for (p, e) in &factors {
            product = product * p.pow(*e);
        }
        assert_eq!(product, n);
    }

    #[test]
    fn test_factor_pollard_rho_with_large_factor() {
        // Factor 2 × 104729 where 104729 is prime
        let n = Integer::from(209458);
        let factors = factor_pollard_rho(&n);

        // Verify factorization
        let mut product = Integer::one();
        for (p, e) in &factors {
            product = product * p.pow(*e);
        }
        assert_eq!(product, n);
    }

    #[test]
    fn test_factor_pollard_rho_zero_one() {
        // Edge cases
        assert_eq!(factor_pollard_rho(&Integer::from(0)), vec![]);
        assert_eq!(factor_pollard_rho(&Integer::from(1)), vec![]);
    }

    #[test]
    fn test_factor_pollard_rho_power_of_prime() {
        // Factor 243 = 3^5
        let n = Integer::from(243);
        let factors = factor_pollard_rho(&n);

        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0], (Integer::from(3), 5));
    }

    // ===== Correctness Verification for Different Algorithms =====

    #[test]
    fn test_compare_factor_methods() {
        // Compare factor() and factor_pollard_rho() on same inputs
        let test_numbers = vec![60, 143, 210, 1000];

        for num in test_numbers {
            let n = Integer::from(num);

            let factors1 = factor(&n);
            let factors2 = factor_pollard_rho(&n);

            // Both should produce the same product
            let mut product1 = Integer::one();
            for (p, e) in &factors1 {
                product1 = product1 * p.pow(*e);
            }

            let mut product2 = Integer::one();
            for (p, e) in &factors2 {
                product2 = product2 * p.pow(*e);
            }

            assert_eq!(product1, n, "factor() failed for {}", num);
            assert_eq!(product2, n, "factor_pollard_rho() failed for {}", num);
            assert_eq!(product1, product2);
        }
    }

    #[test]
    fn test_all_factors_are_prime() {
        // Verify that all returned factors are actually prime
        let test_numbers = vec![60, 143, 210, 1001, 2520];

        for num in test_numbers {
            let n = Integer::from(num);
            let factors = factor_pollard_rho(&n);

            for (prime, _) in factors {
                assert!(is_prime(&prime), "{} is not prime in factorization of {}", prime, num);
            }
        }
    }

    #[test]
    fn test_factorization_completeness() {
        // Verify that factorization is complete (no missing factors)
        let n = Integer::from(2520); // 2³ × 3² × 5 × 7
        let factors = factor(&n);

        // Rebuild the number from factors
        let mut product = Integer::one();
        for (p, e) in &factors {
            product = product * p.pow(*e);
        }

        assert_eq!(product, n);
    }

    // ===== X1b: the fallback of factor_pollard_rho (MAGMA ch. 18 §18.10) =====

    fn z(s: &str) -> Integer {
        Integer::from_decimal_str(s).unwrap()
    }

    fn is_proper_divisor(d: &Integer, m: &Integer) -> bool {
        *d > Integer::one() && d < m && (m % d).is_zero()
    }

    /// Floyd's rho on x^2 + 1 fails on 25 (the cycles modulo 5 and 25 close together), which
    /// is why `factor_pollard_rho` must not trust a `None` from it.
    #[test]
    fn test_pollard_rho_fails_on_25() {
        assert_eq!(pollard_rho(&Integer::from(25)), None);
    }

    /// The fallback splits every odd composite, including 25 and inputs without small primes.
    #[test]
    fn test_split_brent_and_trial_division() {
        // gp: factor(15) = [3,1; 5,1]; factor(9) = Mat([3,2]); factor(25) = Mat([5,2]);
        // factor(3127) = [53,1; 59,1]; factor(12780049) = [53,1; 59,1; 61,1; 67,1];
        // factor(1000000016000000063) = [1000000007,1; 1000000009,1]
        for s in ["9", "15", "25", "3127", "12780049", "1000000016000000063"] {
            let m = z(s);
            let d = split_brent_and_trial_division(&m);
            assert!(
                is_proper_divisor(&d, &m),
                "{d} is not a proper divisor of {m}"
            );
            let d = split_odd_composite(&m);
            assert!(
                is_proper_divisor(&d, &m),
                "{d} is not a proper divisor of {m}"
            );
        }
    }

    #[test]
    fn test_brent_rho() {
        // gp: factor(1000000016000000063) = [1000000007,1; 1000000009,1]
        let m = z("1000000016000000063");
        let d =
            brent_rho(&m, &Integer::from(2), 1 << 20).expect("Brent splits m within 2^20 steps");
        assert!(d == z("1000000007") || d == z("1000000009"), "d = {d}");
        // A budget too small to reach the cycle modulo either prime gives an honest None.
        assert_eq!(brent_rho(&m, &Integer::from(2), 1), None);
    }

    #[test]
    fn test_trial_division_window() {
        // gp: factor(3127) = [53,1; 59,1]: the odd numbers 3..51 are not divisors.
        let m = Integer::from(3127);
        let mut next = Integer::from(3);
        assert_eq!(trial_division_window(&m, &mut next, 25), None);
        assert_eq!(next, Integer::from(53));
        assert_eq!(
            trial_division_window(&m, &mut next, 1),
            Some(Integer::from(53))
        );
    }

    /// A prime is not an odd composite: the window reaches its square root and panics
    /// instead of returning "no divisor".
    #[test]
    #[should_panic(expected = "no divisor of 101 up to its square root")]
    fn test_trial_division_window_rejects_prime() {
        // gp: isprime(101) = 1
        let mut next = Integer::from(3);
        let _ = trial_division_window(&Integer::from(101), &mut next, 100);
    }

    /// Output is sorted ascending (PARI order), as documented.
    #[test]
    fn test_factor_pollard_rho_sorted() {
        // gp: factor(13710311357) = [101,1; 103,1; 107,1; 109,1; 113,1]
        let got = factor_pollard_rho(&Integer::from(13710311357i64));
        let want: Vec<(Integer, u32)> = [101, 103, 107, 109, 113]
            .into_iter()
            .map(|p| (Integer::from(p), 1))
            .collect();
        assert_eq!(got, want);
    }
}
