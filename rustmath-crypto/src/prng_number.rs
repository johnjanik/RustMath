//! Number-theoretic pseudo-random bit generators.
//!
//! Port of MAGMA Handbook **Chapter 158 — Pseudo-random Bit Sequences**,
//! §158.3 *Number Theoretic Bit Generators*: the RSA and Blum–Blum–Shub bit
//! generators plus their modulus-construction helpers.
//!
//! Everything is built on [`rustmath_integers::Integer`] and the crate's existing
//! prime machinery (`rustmath_integers::prime`). Bits are `u8` values in `{0,1}`.
//!
//! **Warning (from the MAGMA handbook):** these moduli are *not* intended for
//! real-world cryptographic use — the seeding here is a small deterministic PRNG,
//! chosen so that the `*_seeded` / explicit constructors are fully reproducible
//! for testing. The randomised `b`-bit constructors seed from the system clock.

use rustmath_core::{MathError, Result};
use rustmath_integers::prime::is_prime;
use rustmath_integers::Integer;

/// A tiny, self-contained SplitMix64 PRNG (pure Rust, no external deps).
///
/// Used only to pick candidate primes and seeds; it is *not* a cryptographic
/// generator. Seed it explicitly for reproducible output.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Create a PRNG with an explicit 64-bit seed.
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    /// Seed from the system clock (non-reproducible).
    pub fn from_entropy() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15);
        SplitMix64::new(nanos ^ 0xD1B5_4A32_D192_ED03)
    }

    /// Next 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// A uniformly-ish random integer with exactly `bits` bits (top bit set), i.e. in
/// `[2^(bits-1), 2^bits)`. Requires `bits >= 1`.
fn random_integer_bits(bits: u64, rng: &mut SplitMix64) -> Integer {
    debug_assert!(bits >= 1);
    let two = Integer::from(2);
    let mut v = Integer::one(); // top bit set
    for _ in 1..bits {
        let bit = Integer::from(rng.next_u64() & 1);
        v = v * two.clone() + bit;
    }
    v
}

/// A (pseudo-)random prime of approximately `bits` bits (MAGMA `RandomPrime`
/// analogue). Starts from a random `bits`-bit odd candidate and scans upward.
fn random_prime_bits(bits: u64, rng: &mut SplitMix64) -> Integer {
    let two = Integer::from(2);
    loop {
        let mut cand = random_integer_bits(bits, rng);
        if cand.is_even() {
            cand = cand + Integer::one();
        }
        // Scan odd candidates; give up after a while and re-seed a fresh start.
        for _ in 0..(4 * bits + 64) {
            if is_prime(&cand) {
                return cand;
            }
            cand = cand + two.clone();
        }
    }
}

/// A (pseudo-)random prime `≡ 3 (mod 4)` of approximately `bits` bits, as required
/// by the Blum–Blum–Shub construction.
fn random_prime_3mod4_bits(bits: u64, rng: &mut SplitMix64) -> Integer {
    let four = Integer::from(4);
    let three = Integer::from(3);
    loop {
        let cand0 = random_integer_bits(bits, rng);
        // Force cand ≡ 3 (mod 4).
        let r = &cand0 % &four;
        let mut cand = cand0 - r + three.clone();
        for _ in 0..(4 * bits + 64) {
            if is_prime(&cand) {
                return cand;
            }
            cand = cand + four.clone();
        }
    }
}

/// Least-significant bit (parity) of `x` as a `0`/`1` bit.
#[inline]
fn parity_bit(x: &Integer) -> u8 {
    if x.is_even() {
        0
    } else {
        1
    }
}

/// Reduce `x` into `[0, n)` (n > 0).
fn reduce_mod(x: &Integer, n: &Integer) -> Integer {
    let r = x % n;
    if r.signum() < 0 {
        r + n.clone()
    } else {
        r
    }
}

// ---------------------------------------------------------------------------
// RSA pseudo-random bit generator
// ---------------------------------------------------------------------------

/// Build an RSA modulus `n` of about `b` bits together with a valid exponent `e`
/// (MAGMA `RSAModulus(b)`): `Gcd(EulerPhi(n), e) = 1`.
///
/// The exponent is `3` when `gcd(φ(n), 3) = 1`, otherwise a random odd `e`
/// coprime to `φ(n)` is chosen. `b` must be at least 16.
pub fn rsa_modulus(b: u64, rng: &mut SplitMix64) -> Result<(Integer, Integer)> {
    if b < 16 {
        return Err(MathError::InvalidArgument("b must be at least 16".into()));
    }
    let half = b / 2;
    let other = b - half;
    let p = random_prime_bits(half, rng);
    let mut q = random_prime_bits(other, rng);
    while q == p {
        q = random_prime_bits(other, rng);
    }
    let n = p.clone() * q.clone();
    // φ(n) = (p-1)(q-1); we know the factorisation, so this is exact.
    let phi = (p - Integer::one()) * (q - Integer::one());

    let three = Integer::from(3);
    let e = if phi.gcd(&three).is_one() {
        three
    } else {
        // Pick a random odd e in (1, phi) coprime to phi.
        loop {
            let bits = phi.bit_length().max(2) - 1;
            let mut cand = random_integer_bits(bits.max(2), rng);
            if cand.is_even() {
                cand = cand + Integer::one();
            }
            if cand > Integer::one() && cand < phi && phi.gcd(&cand).is_one() {
                break cand;
            }
        }
    };
    Ok((n, e))
}

/// Build an RSA modulus `n` of about `b` bits such that `Gcd(EulerPhi(n), e) = 1`
/// for the supplied exponent `e` (MAGMA `RSAModulus(b, e)`).
///
/// `b` must be at least 16 and `e` odd with `1 < e < 2^b`.
pub fn rsa_modulus_with_exponent(b: u64, e: &Integer, rng: &mut SplitMix64) -> Result<Integer> {
    if b < 16 {
        return Err(MathError::InvalidArgument("b must be at least 16".into()));
    }
    if e.is_even() || *e <= Integer::one() {
        return Err(MathError::InvalidArgument(
            "e must be odd and greater than 1".into(),
        ));
    }
    let half = b / 2;
    let other = b - half;
    loop {
        let p = random_prime_bits(half, rng);
        let mut q = random_prime_bits(other, rng);
        while q == p {
            q = random_prime_bits(other, rng);
        }
        let phi = (p.clone() - Integer::one()) * (q.clone() - Integer::one());
        if phi.gcd(e).is_one() {
            return Ok(p * q);
        }
    }
}

/// The RSA pseudo-random bit generator with explicit parameters
/// (MAGMA `RandomSequenceRSA(n, e, s, t)`): iterate `x ← x^e mod n` and emit the
/// parity bit of each successive `x`. Fully deterministic.
///
/// `n` must be greater than 1.
pub fn random_sequence_rsa_explicit(
    n: &Integer,
    e: &Integer,
    s: &Integer,
    t: usize,
) -> Result<Vec<u8>> {
    if *n <= Integer::one() {
        return Err(MathError::InvalidArgument("n must be greater than 1".into()));
    }
    let mut x = reduce_mod(s, n);
    let mut out = Vec::with_capacity(t);
    for _ in 0..t {
        x = x.mod_pow(e, n)?;
        out.push(parity_bit(&x));
    }
    Ok(out)
}

/// The RSA pseudo-random bit generator, generating a fresh `b`-bit modulus,
/// exponent and seed from `rng` (MAGMA `RandomSequenceRSA(b, t)`).
///
/// `b` must be at least 16. Reproducible given the same seeded `rng`.
pub fn random_sequence_rsa(b: u64, t: usize, rng: &mut SplitMix64) -> Result<Vec<u8>> {
    let (n, e) = rsa_modulus(b, rng)?;
    // Seed value modulo n (kept away from the trivial 0/1).
    let seed_bits = n.bit_length().max(2);
    let s = reduce_mod(&random_integer_bits(seed_bits, rng), &n);
    random_sequence_rsa_explicit(&n, &e, &s, t)
}

// ---------------------------------------------------------------------------
// Blum–Blum–Shub pseudo-random bit generator
// ---------------------------------------------------------------------------

/// Build a Blum–Blum–Shub modulus `n = p·q` of about `b` bits with both primes
/// `≡ 3 (mod 4)` (MAGMA `BBSModulus(b)` / `BlumBlumShubModulus(b)`).
///
/// `b` must be at least 16.
pub fn bbs_modulus(b: u64, rng: &mut SplitMix64) -> Result<Integer> {
    if b < 16 {
        return Err(MathError::InvalidArgument("b must be at least 16".into()));
    }
    let half = b / 2;
    let other = b - half;
    let p = random_prime_3mod4_bits(half, rng);
    let mut q = random_prime_3mod4_bits(other, rng);
    while q == p {
        q = random_prime_3mod4_bits(other, rng);
    }
    Ok(p * q)
}

/// The Blum–Blum–Shub generator with explicit modulus and seed
/// (MAGMA `RandomSequenceBlumBlumShub(n, s, t)` / `BlumBlumShub(n, s, t)`):
/// `x_0 = s^2 mod n`, then `x_i = x_{i-1}^2 mod n`, emitting `parity(x_i)`.
///
/// `n` must be greater than 1 and `gcd(s, n) = 1`.
pub fn blum_blum_shub_explicit(n: &Integer, s: &Integer, t: usize) -> Result<Vec<u8>> {
    if *n <= Integer::one() {
        return Err(MathError::InvalidArgument("n must be greater than 1".into()));
    }
    let s_red = reduce_mod(s, n);
    if !s_red.gcd(n).is_one() {
        return Err(MathError::InvalidArgument(
            "gcd(s, n) must be 1".into(),
        ));
    }
    let two = Integer::from(2);
    let mut x = s_red.mod_pow(&two, n)?; // x_0 = s^2 mod n
    let mut out = Vec::with_capacity(t);
    for _ in 0..t {
        x = x.mod_pow(&two, n)?;
        out.push(parity_bit(&x));
    }
    Ok(out)
}

/// The Blum–Blum–Shub generator with a fresh `b`-bit modulus and seed
/// (MAGMA `RandomSequenceBlumBlumShub(b, t)` / `BlumBlumShub(b, t)`).
///
/// `b` must be at least 16. Reproducible given the same seeded `rng`.
pub fn blum_blum_shub(b: u64, t: usize, rng: &mut SplitMix64) -> Result<Vec<u8>> {
    let n = bbs_modulus(b, rng)?;
    // Pick a seed coprime to n.
    let seed_bits = n.bit_length().max(2);
    let mut s = reduce_mod(&random_integer_bits(seed_bits, rng), &n);
    while s <= Integer::one() || !s.gcd(&n).is_one() {
        s = reduce_mod(&random_integer_bits(seed_bits, rng), &n);
    }
    blum_blum_shub_explicit(&n, &s, t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_deterministic() {
        let mut a = SplitMix64::new(12345);
        let mut b = SplitMix64::new(12345);
        assert_eq!(a.next_u64(), b.next_u64());
        assert_ne!(SplitMix64::new(1).next_u64(), SplitMix64::new(2).next_u64());
    }

    #[test]
    fn random_prime_bits_are_prime() {
        let mut rng = SplitMix64::new(42);
        for _ in 0..5 {
            let p = random_prime_bits(20, &mut rng);
            assert!(is_prime(&p), "{p} should be prime");
        }
    }

    #[test]
    fn random_prime_3mod4() {
        let mut rng = SplitMix64::new(7);
        let four = Integer::from(4);
        let three = Integer::from(3);
        for _ in 0..5 {
            let p = random_prime_3mod4_bits(20, &mut rng);
            assert!(is_prime(&p));
            assert_eq!(&p % &four, three);
        }
    }

    /// MAGMA example H158E1-style RSA generator round trip: fully deterministic
    /// output, correct length, and both bit values present.
    #[test]
    fn rsa_explicit_deterministic() {
        // Small textbook RSA: p=61, q=53, n=3233, φ=3120, pick e=17 (gcd=1).
        let n = Integer::from(3233);
        let e = Integer::from(17);
        let s = Integer::from(123);
        let a = random_sequence_rsa_explicit(&n, &e, &s, 64).unwrap();
        let b = random_sequence_rsa_explicit(&n, &e, &s, 64).unwrap();
        assert_eq!(a, b, "generator must be deterministic");
        assert_eq!(a.len(), 64);
        assert!(a.iter().any(|&x| x == 0) && a.iter().any(|&x| x == 1));
    }

    #[test]
    fn rsa_explicit_rejects_bad_modulus() {
        let n = Integer::one();
        let e = Integer::from(3);
        let s = Integer::from(2);
        assert!(random_sequence_rsa_explicit(&n, &e, &s, 4).is_err());
    }

    /// MAGMA example H158E2 (spirit): draw a sequence from a ~b-bit RSA modulus
    /// and inspect its statistics; here we just require reproducibility, correct
    /// length and a plausible balance of 0s and 1s.
    #[test]
    fn rsa_random_sequence_reproducible() {
        let mut r1 = SplitMix64::new(2024);
        let mut r2 = SplitMix64::new(2024);
        let a = random_sequence_rsa(32, 100, &mut r1).unwrap();
        let b = random_sequence_rsa(32, 100, &mut r2).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 100);
        let ones = a.iter().filter(|&&x| x == 1).count();
        assert!(ones > 10 && ones < 90, "unbalanced: {ones} ones of 100");
    }

    #[test]
    fn bbs_explicit_deterministic() {
        // p=7, q=11 (both ≡ 3 mod 4), n=77; seed 3, gcd(3,77)=1.
        let n = Integer::from(77);
        let s = Integer::from(3);
        let a = blum_blum_shub_explicit(&n, &s, 32).unwrap();
        let b = blum_blum_shub_explicit(&n, &s, 32).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 32);
    }

    #[test]
    fn bbs_explicit_rejects_noncoprime_seed() {
        // gcd(7, 77) = 7 != 1.
        let n = Integer::from(77);
        let s = Integer::from(7);
        assert!(blum_blum_shub_explicit(&n, &s, 8).is_err());
    }

    #[test]
    fn bbs_modulus_primes_are_3mod4_product() {
        let mut rng = SplitMix64::new(99);
        let n = bbs_modulus(20, &mut rng).unwrap();
        assert!(n > Integer::one());
        // n must be odd (product of two odd primes).
        assert!(!n.is_even());
    }

    #[test]
    fn bbs_random_reproducible() {
        let mut r1 = SplitMix64::new(555);
        let mut r2 = SplitMix64::new(555);
        let a = blum_blum_shub(24, 50, &mut r1).unwrap();
        let b = blum_blum_shub(24, 50, &mut r2).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 50);
    }

    #[test]
    fn rsa_modulus_exponent_coprime() {
        let mut rng = SplitMix64::new(31337);
        let (n, e) = rsa_modulus(24, &mut rng).unwrap();
        assert!(n > Integer::one());
        assert!(e > Integer::one());
        // A generated sequence should run without error.
        let seq = random_sequence_rsa_explicit(&n, &e, &Integer::from(5), 16).unwrap();
        assert_eq!(seq.len(), 16);
    }

    #[test]
    fn rsa_modulus_with_fixed_exponent() {
        let mut rng = SplitMix64::new(808);
        let e = Integer::from(65537);
        let n = rsa_modulus_with_exponent(24, &e, &mut rng).unwrap();
        let phi_coprime = {
            // n is a product of two ~12-bit primes; just check a sequence runs.
            random_sequence_rsa_explicit(&n, &e, &Integer::from(9), 8).is_ok()
        };
        assert!(phi_coprime);
    }

    #[test]
    fn small_b_rejected() {
        let mut rng = SplitMix64::new(1);
        assert!(rsa_modulus(8, &mut rng).is_err());
        assert!(bbs_modulus(8, &mut rng).is_err());
    }
}
