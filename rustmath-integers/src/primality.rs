//! Probable-prime tests: the deterministic word test and the Baillie-PSW test (MAGMA Handbook
//! ch. 18 §18.9.1 `IsProbablePrime`, `IsPrime(n: Proof := false)`; digest
//! `rustmath-lava/docs/Chapter_18_Ring_of_Integers.md`). The proving test `IsPrime` is
//! [`crate::prime::is_prime`] (this module, then ECPP in [`crate::ecpp`]).
//!
//! - [`is_prime_u64`]: trial division by the primes below 41, then the strong (Miller-Rabin)
//!   test to the twelve prime bases 2..37. No composite below
//!   psi_12 = 318665857834031151167461 > 2^64 is a strong pseudoprime to all twelve bases
//!   (J. Sorenson, J. Webster, *Strong pseudoprimes to twelve prime bases*, Math. Comp. 86
//!   (2017) 985-1003), so the answer is a proof for every `u64`. The strong test is [Jae93]
//!   G. Jaeschke, *On strong pseudoprimes to several bases*, Math. Comp. 61 (1993) 915-926.
//! - [`is_probable_prime`]: exact below psi_12 (the word test, then the twelve bases); above it
//!   the Baillie-PSW test: a strong probable prime to base 2 and a strong Lucas probable prime
//!   with Selfridge's parameters (method A: the first D in 5, -7, 9, -11, ... with
//!   (D/n) = -1, P = 1, Q = (1 - D)/4). R. Baillie, S. S. Wagstaff Jr., *Lucas pseudoprimes*,
//!   Math. Comp. 35 (1980) 1391-1417; C. Pomerance, J. L. Selfridge, S. S. Wagstaff Jr., *The
//!   pseudoprimes to 25·10^9*, Math. Comp. 35 (1980) 1003-1026; R. Baillie, A. Fiori,
//!   S. S. Wagstaff Jr., *Strengthening the Baillie-PSW primality test*, Math. Comp. 90 (2021)
//!   1931-1955 (no BPSW pseudoprime below 2^64; none is known at all). `false` is always a
//!   proof that n is not prime; `true` above psi_12 is not a proof.
//! - [`NotPrimeWitness`]: why a number is not prime, in a form that [`NotPrimeWitness::confirms`]
//!   re-checks cheaply.

use crate::Integer;
use num_bigint::BigInt;
use num_integer::Integer as _;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::sync::OnceLock;

/// psi_12 = 318665857834031151167461, the least strong pseudoprime to the twelve prime bases
/// 2..37 (Sorenson-Webster 2017). It fits a `u128`.
pub(crate) const PSI_12: u128 = 318_665_857_834_031_151_167_461;

/// The twelve prime bases of the deterministic test.
pub(crate) const TWELVE_BASES: [u32; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

/// Deterministic primality of a 64-bit word: trial division by the primes below 41, then the
/// strong (Miller-Rabin) test to the twelve prime bases 2..37. No composite below
/// psi_12 = 318665857834031151167461 > 2^64 is a strong pseudoprime to all twelve bases
/// (J. Sorenson, J. Webster, *Strong pseudoprimes to twelve prime bases*, Math. Comp. 86
/// (2017) 985-1003), so the answer is a proof for every `u64`. About 10 µs in a debug build.
pub fn is_prime_u64(n: u64) -> bool {
    use crate::fast_arith::pow_mod_u64;
    if n < 2 {
        return false;
    }
    for p in TWELVE_BASES {
        let p = u64::from(p);
        if n.is_multiple_of(p) {
            return n == p;
        }
    }
    if n < 41 * 41 {
        return true; // no prime factor below 41 and n < 41^2
    }
    let r = (n - 1).trailing_zeros();
    let d = (n - 1) >> r;
    'bases: for a in TWELVE_BASES {
        let mut x = pow_mod_u64(u64::from(a), d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..r {
            x = (u128::from(x) * u128::from(x) % u128::from(n)) as u64;
            if x == n - 1 {
                continue 'bases;
            }
        }
        return false;
    }
    true
}

/// Why a number n is not prime. Every variant can be re-checked cheaply with
/// [`NotPrimeWitness::confirms`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NotPrimeWitness {
    /// n < 2.
    BelowTwo,
    /// A divisor d with 1 < d < n.
    Divisor(Integer),
    /// n is odd and not a strong probable prime to this base (a Miller-Rabin witness),
    /// 2 <= base < n.
    MillerRabin(u32),
    /// n is odd, D = 1 (mod 4), (D/n) = -1, gcd(n, 2 D Q) = 1 for Q = (1 - D)/4, and n is not a
    /// strong Lucas probable prime for (P, Q) = (1, Q). Every prime with these conditions
    /// passes (Baillie-Wagstaff 1980), so the witness does not depend on D being Selfridge's
    /// first choice; the prover records Selfridge's D.
    StrongLucas(i64),
    /// n > 1 is a perfect square.
    Square,
}

impl NotPrimeWitness {
    /// Re-runs the check behind the witness; true iff it shows that n is not prime. Never
    /// panics: data that does not meet the variant's conditions gives false.
    pub fn confirms(&self, n: &Integer) -> bool {
        let v = n.as_bigint();
        match self {
            NotPrimeWitness::BelowTwo => *v < BigInt::from(2u32),
            NotPrimeWitness::Divisor(d) => {
                let d = d.as_bigint();
                *d > BigInt::one() && d < v && (v % d).is_zero()
            }
            NotPrimeWitness::MillerRabin(a) => {
                let a = BigInt::from(*a);
                v.is_odd() && a >= BigInt::from(2u32) && a < *v && !strong_probable_prime(v, &a)
            }
            NotPrimeWitness::StrongLucas(d) => {
                let d = *d;
                v.is_odd()
                    && *v > BigInt::one()
                    && d.rem_euclid(4) == 1
                    && jacobi_small(d, v) == -1
                    && matches!(lucas_check(v, d), Ok(false))
            }
            NotPrimeWitness::Square => {
                *v > BigInt::one() && {
                    let r = v.sqrt();
                    &r * &r == *v
                }
            }
        }
    }
}

/// Products of consecutive primes from a list, each product below 2^64, so that the residues of
/// a big number modulo many small primes cost one `BigInt % u64` per product.
pub(crate) struct PrimeBatches {
    batches: Vec<(u64, std::ops::Range<usize>)>,
    primes: &'static [u32],
}

impl PrimeBatches {
    fn new(primes: &'static [u32]) -> Self {
        let mut batches = Vec::new();
        let mut prod: u128 = 1;
        let mut start = 0usize;
        for (i, &p) in primes.iter().enumerate() {
            if prod * u128::from(p) >= 1u128 << 64 {
                batches.push((prod as u64, start..i));
                prod = 1;
                start = i;
            }
            prod *= u128::from(p);
        }
        if start < primes.len() {
            batches.push((prod as u64, start..primes.len()));
        }
        PrimeBatches { batches, primes }
    }

    /// The least prime of the list that divides `n` (n > 0), if any.
    pub(crate) fn least_divisor(&self, n: &BigInt) -> Option<u32> {
        for (prod, range) in &self.batches {
            let r = (n % *prod)
                .to_u64()
                .expect("a residue modulo a u64 fits a u64");
            for &p in &self.primes[range.clone()] {
                if r.is_multiple_of(u64::from(p)) {
                    return Some(p);
                }
            }
        }
        None
    }

    /// m = k q with k the part of m (m > 0) made of the primes of the list: returns (k, q).
    pub(crate) fn strip(&self, m: &BigInt) -> (BigInt, BigInt) {
        let mut q = m.clone();
        let mut k = BigInt::one();
        for (prod, range) in &self.batches {
            let r = (&q % *prod)
                .to_u64()
                .expect("a residue modulo a u64 fits a u64");
            if r != 0 && gcd_u64(r, *prod) == 1 {
                continue;
            }
            for &p in &self.primes[range.clone()] {
                if r.is_multiple_of(u64::from(p)) {
                    let pb = BigInt::from(p);
                    loop {
                        let (qq, rr) = q.div_rem(&pb);
                        if !rr.is_zero() {
                            break;
                        }
                        q = qq;
                        k *= p;
                    }
                }
            }
        }
        (k, q)
    }
}

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// The primes below 1000 in batches (trial division before the strong tests).
pub(crate) fn batches_below_1000() -> &'static PrimeBatches {
    static B: OnceLock<PrimeBatches> = OnceLock::new();
    B.get_or_init(|| {
        let small = crate::sieve::small_primes();
        let end = small.partition_point(|&p| p < 1000);
        PrimeBatches::new(&small[..end])
    })
}

/// The primes below 2^16 in batches (the ECPP prover strips them from the curve orders).
pub(crate) fn batches_below_2_16() -> &'static PrimeBatches {
    static B: OnceLock<PrimeBatches> = OnceLock::new();
    B.get_or_init(|| PrimeBatches::new(crate::sieve::small_primes()))
}

/// Jacobi symbol (a/n) for an odd n > 0 and a word-size a (any sign), by quadratic reciprocity:
/// one `BigInt % u64`, then word arithmetic.
pub(crate) fn jacobi_small(a: i64, n: &BigInt) -> i32 {
    let n_low = n.iter_u64_digits().next().unwrap_or(0);
    let n8 = n_low & 7;
    let mut t = 1;
    if a < 0 && n8 % 4 == 3 {
        t = -t; // (-1/n) = (-1)^((n-1)/2)
    }
    let mut m = a.unsigned_abs();
    if m == 0 {
        return if n.is_one() { 1 } else { 0 };
    }
    let v = m.trailing_zeros();
    m >>= v;
    if v % 2 == 1 && (n8 == 3 || n8 == 5) {
        t = -t; // (2/n) = (-1)^((n^2-1)/8)
    }
    if m == 1 {
        return t;
    }
    if m % 4 == 3 && n8 % 4 == 3 {
        t = -t; // reciprocity for odd m, n
    }
    let r = (n % m).to_u64().expect("a residue modulo a u64 fits a u64");
    t * jacobi_u64(r, m)
}

/// Jacobi symbol (a/n) for odd n > 0 on words.
fn jacobi_u64(mut a: u64, mut n: u64) -> i32 {
    let mut t = 1;
    a %= n;
    while a != 0 {
        let v = a.trailing_zeros();
        a >>= v;
        if v % 2 == 1 && (n % 8 == 3 || n % 8 == 5) {
            t = -t;
        }
        std::mem::swap(&mut a, &mut n);
        if a % 4 == 3 && n % 4 == 3 {
            t = -t;
        }
        a %= n;
    }
    if n == 1 {
        t
    } else {
        0
    }
}

/// Strong probable-prime test of the odd number n > 2 to the base a (Miller-Rabin, [Jae93]):
/// with n - 1 = d 2^s, d odd, true iff a^d = 1 or a^(d 2^r) = -1 (mod n) for some r < s.
/// `x <- x^2` is a multiplication, not a modular power.
pub(crate) fn strong_probable_prime(n: &BigInt, a: &BigInt) -> bool {
    let n_minus_1 = n - 1u32;
    let s = n_minus_1.trailing_zeros().expect("n > 1");
    let d = &n_minus_1 >> s;
    let mut x = a.mod_floor(n).modpow(&d, n);
    if x.is_one() || x == n_minus_1 {
        return true;
    }
    for _ in 1..s {
        x = (&x * &x) % n;
        if x == n_minus_1 {
            return true;
        }
        if x.is_one() {
            return false;
        }
    }
    false
}

/// The steps 3-5 of BPSW for a given D: n odd > 1, D = 1 (mod 4) and (D/n) = -1 (callers
/// ensure both). With Q = (1 - D)/4 and P = 1: `Err(Divisor(g))` when g = gcd(n, |Q|) > 1 (then
/// g < n, because n | Q would give D = 1 (mod n) and (D/n) = 1); otherwise `Ok(true)` iff n is
/// a strong Lucas probable prime: with n + 1 = d 2^s, d odd, U_d = 0 or V_(d 2^r) = 0 for some
/// 0 <= r < s (mod n).
pub(crate) fn lucas_check(n: &BigInt, d: i64) -> Result<bool, NotPrimeWitness> {
    debug_assert!(n.is_odd() && *n > BigInt::one() && d.rem_euclid(4) == 1);
    let q_small = (1 - i128::from(d)) / 4;
    let q = BigInt::from(q_small);
    let g = n.gcd(&q.abs());
    if !g.is_one() {
        return Err(NotPrimeWitness::Divisor(Integer::new(g)));
    }
    let dd = BigInt::from(d).mod_floor(n);
    let qm = q.mod_floor(n);
    let n_plus_1: BigInt = n + 1u32;
    let s = n_plus_1.trailing_zeros().expect("n + 1 > 0");
    let k = &n_plus_1 >> s;
    // Halving modulo n is multiplication by (n + 1)/2.
    let inv2 = &n_plus_1 >> 1u32;
    let mut u = BigInt::one();
    let mut v = BigInt::one(); // V_1 = P = 1
    let mut qk = qm.clone();
    let bits = k.bits();
    for i in (0..bits - 1).rev() {
        // Doubling: U_2k = U_k V_k, V_2k = V_k^2 - 2 Q^k, Q^2k = (Q^k)^2.
        u = (&u * &v) % n;
        v = (&v * &v - &qk - &qk).mod_floor(n);
        qk = (&qk * &qk) % n;
        if k.bit(i) {
            // Increment: U_(k+1) = (P U + V)/2, V_(k+1) = (D U + P V)/2, Q^(k+1) = Q^k Q.
            let nu = ((&u + &v) * &inv2).mod_floor(n);
            let nv = ((&dd * &u + &v) * &inv2).mod_floor(n);
            u = nu;
            v = nv;
            qk = (&qk * &qm) % n;
        }
    }
    if u.is_zero() || v.is_zero() {
        return Ok(true);
    }
    for _ in 1..s {
        v = (&v * &v - &qk - &qk).mod_floor(n);
        qk = (&qk * &qk) % n;
        if v.is_zero() {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Selfridge's D (method A) for an odd non-square n > 1: the first D in 5, -7, 9, -11, ...
/// with (D/n) = -1. A D with (D/n) = 0 and gcd(|D|, n) a proper divisor gives
/// `Err(Divisor(..))`. For a non-square n such a D exists, so the loop ends (in practice after
/// one to three tries).
fn selfridge_d(n: &BigInt) -> Result<i64, NotPrimeWitness> {
    let mut d: i64 = 5;
    loop {
        match jacobi_small(d, n) {
            -1 => return Ok(d),
            0 => {
                let g = n.gcd(&BigInt::from(d.unsigned_abs()));
                if &g < n {
                    return Err(NotPrimeWitness::Divisor(Integer::new(g)));
                }
                // n divides D (only for n <= |D|): keep looking.
            }
            _ => {}
        }
        d = if d > 0 { -d - 2 } else { -d + 2 };
    }
}

/// The Baillie-PSW test of an odd n > 1: `Ok(())` if n is a strong probable prime to base 2 and a
/// strong Lucas probable prime with Selfridge's parameters, else the witness. Correct for every
/// odd n > 1 (a prime always passes); trial division by small primes is the caller's business.
pub(crate) fn bpsw(n: &BigInt) -> Result<(), NotPrimeWitness> {
    debug_assert!(n.is_odd() && *n > BigInt::one());
    if *n == BigInt::from(3u32) {
        return Ok(());
    }
    if !strong_probable_prime(n, &BigInt::from(2u32)) {
        return Err(NotPrimeWitness::MillerRabin(2));
    }
    let r = n.sqrt();
    if &r * &r == *n {
        return Err(NotPrimeWitness::Square);
    }
    let d = selfridge_d(n)?;
    match lucas_check(n, d)? {
        true => Ok(()),
        false => Err(NotPrimeWitness::StrongLucas(d)),
    }
}

/// The first of the twelve bases 2..37 to which the odd n > 37 is not a strong probable prime.
pub(crate) fn twelve_base_witness(n: &BigInt) -> Option<u32> {
    TWELVE_BASES
        .into_iter()
        .find(|&a| !strong_probable_prime(n, &BigInt::from(a)))
}

/// The probable-prime verdict on n >= 2^64 after trial division below 1000: `Ok(())` when n is
/// a probable prime (a proof below psi_12), else the witness.
pub(crate) fn probable_prime_big(n: &BigInt) -> Result<(), NotPrimeWitness> {
    if let Some(p) = batches_below_1000().least_divisor(n) {
        return Err(NotPrimeWitness::Divisor(Integer::from(p)));
    }
    if *n < BigInt::from(PSI_12) {
        return match twelve_base_witness(n) {
            Some(a) => Err(NotPrimeWitness::MillerRabin(a)),
            None => Ok(()),
        };
    }
    bpsw(n)
}

/// MAGMA `IsProbablePrime(n)` / `IsPrime(n: Proof := false)` (ch. 18 §18.9.1). Exact for
/// n < psi_12 = 318665857834031151167461 (n < 2^64 by [`is_prime_u64`], then the strong test to
/// the twelve prime bases 2..37, Sorenson-Webster 2017). Above psi_12 the Baillie-PSW test: a
/// strong probable prime to base 2 and a strong Lucas probable prime with Selfridge's
/// parameters. `false` is always a proof; `true` above psi_12 is not (no counterexample is
/// known; none exists below 2^64, Baillie-Fiori-Wagstaff 2021). Cost: one strong test and one
/// Lucas sequence, about the price of three modular exponentiations.
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::prime::is_probable_prime;
/// // psi_12 = 399165290221 * 798330580441 passes the twelve bases 2..37 but not BPSW.
/// let psi12 = Integer::from_decimal_str("318665857834031151167461").unwrap();
/// assert!(!is_probable_prime(&psi12));
/// assert!(is_probable_prime(&(Integer::from(2).pow(127) - Integer::one())));
/// ```
pub fn is_probable_prime(n: &Integer) -> bool {
    let v = n.as_bigint();
    if v.is_negative() {
        return false;
    }
    if let Some(w) = v.to_u64() {
        return is_prime_u64(w);
    }
    probable_prime_big(v).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn big(s: &str) -> BigInt {
        s.parse().unwrap()
    }

    /// A base-2 strong probable-prime test on words, to enumerate pseudoprimes.
    fn sprp2_u64(n: u64) -> bool {
        use crate::fast_arith::pow_mod_u64;
        let r = (n - 1).trailing_zeros();
        let d = (n - 1) >> r;
        let mut x = pow_mod_u64(2, d, n);
        if x == 1 || x == n - 1 {
            return true;
        }
        for _ in 1..r {
            x = (u128::from(x) * u128::from(x) % u128::from(n)) as u64;
            if x == n - 1 {
                return true;
            }
        }
        false
    }

    #[test]
    fn test_is_prime_u64_matches_sieve() {
        // Every n < 10^5 against the sieve (gp: #primes([0,100000]) -> 9592).
        let primes = crate::sieve::primes_in_range(0, 100_000);
        assert_eq!(primes.len(), 9592);
        let mut next = primes.iter().peekable();
        for n in 0..100_000u64 {
            let sieve = next.peek() == Some(&&n);
            if sieve {
                next.next();
            }
            assert_eq!(is_prime_u64(n), sieve, "is_prime_u64({n})");
        }
    }

    #[test]
    fn test_is_prime_u64_strong_pseudoprimes_and_word_edges() {
        // gp: factor(3215031751) -> 151*751*28351 (strong pseudoprime to the bases 2, 3, 5, 7);
        // factor(3825123056546413051) -> 149491*747451*34233211 (strong pseudoprime to the
        // bases 2..31). Both have no factor below 41, so they reach the Miller-Rabin stage.
        assert!(!is_prime_u64(3_215_031_751));
        assert!(!is_prime_u64(3_825_123_056_546_413_051));
        // gp: isprime(2^61-1) -> 1, precprime(2^64) -> 18446744073709551557, isprime(2^64-1) -> 0
        assert!(is_prime_u64((1 << 61) - 1));
        assert!(is_prime_u64(18_446_744_073_709_551_557));
        assert!(!is_prime_u64(u64::MAX));
    }

    #[test]
    fn jacobi_small_matches_kronecker() {
        // Against the general Kronecker symbol of integer.rs on a grid.
        for n in (1u64..400).step_by(2) {
            let nb = BigInt::from(n);
            for a in -50i64..50 {
                let want = Integer::from(a).kronecker_symbol(&Integer::from(n));
                assert_eq!(jacobi_small(a, &nb), i32::from(want), "({a}/{n})");
            }
        }
        // gp: kronecker(-7, 318665857834031151167461) = -1, kronecker(5, ..) = 1
        let psi = big("318665857834031151167461");
        assert_eq!(jacobi_small(-7, &psi), -1);
        assert_eq!(jacobi_small(5, &psi), 1);
    }

    /// The odd composites below 10^6 that are strong probable primes to base 2 (OEIS A001262:
    /// 46 of them, first 2047, last 983401; gp script `t16_sp2.gp` of the I1 design counts 46)
    /// all fail BPSW. BPSW is correct for every odd n > 1, so these inputs (all with a factor
    /// below 1000) are in its contract.
    #[test]
    fn bpsw_rejects_strong_base2_pseudoprimes() {
        let mut found = Vec::new();
        for n in (3u64..1_000_000).step_by(2) {
            if !is_prime_u64(n) && sprp2_u64(n) {
                found.push(n);
            }
        }
        assert_eq!(found.len(), 46);
        assert_eq!(found[..5], [2047, 3277, 4033, 4681, 8321]);
        assert_eq!(*found.last().unwrap(), 983401);
        for &n in &found {
            assert!(bpsw(&BigInt::from(n)).is_err(), "bpsw({n}) passed");
        }
        // Base-2 strong pseudoprimes without a prime factor below 1000 (review 2 n1; gp:
        // factor(2284453) = 1069*2137, factor(5489641) = 1657*3313, factor(8725753) = 2089*4177,
        // factor(9863461) = 2221*4441, factor(27278026129) = 1657*3313*4969,
        // factor(65700513721) = 2221*4441*6661; each passes the base-2 strong test sp2(n) and
        // ispseudoprime(n) = 0) fail the Lucas part with Selfridge's D.
        for n in [
            2284453u64,
            5489641,
            8725753,
            9863461,
            27278026129,
            65700513721,
        ] {
            assert!(sprp2_u64(n), "{n} is a base-2 strong pseudoprime");
            assert!(batches_below_1000()
                .least_divisor(&BigInt::from(n))
                .is_none());
            match bpsw(&BigInt::from(n)) {
                Err(NotPrimeWitness::StrongLucas(d)) => {
                    assert!(NotPrimeWitness::StrongLucas(d).confirms(&Integer::from(n)));
                }
                other => panic!("bpsw({n}) = {other:?}"),
            }
        }
    }

    /// The odd composites n < 20000 that pass the strong Lucas test with Selfridge's
    /// parameters (method A): enumerate the odd non-square composites, take Selfridge's D, skip
    /// n with (D/n) = 0 for a D before it or gcd(n, Q) > 1. This is the definition of OEIS
    /// A217255, whose terms below 20000 are 5459, 5777, 10877, 16109, 18971.
    #[test]
    fn strong_lucas_pseudoprimes_below_20000() {
        let mut found = Vec::new();
        for n in (3u64..20000).step_by(2) {
            if is_prime_u64(n) {
                continue;
            }
            let nb = BigInt::from(n);
            let r = nb.sqrt();
            if &r * &r == nb {
                continue;
            }
            let Ok(d) = selfridge_d(&nb) else { continue };
            if let Ok(true) = lucas_check(&nb, d) {
                found.push(n);
            }
        }
        assert_eq!(found, vec![5459, 5777, 10877, 16109, 18971]);
    }

    /// Selfridge's D never gets large enough to reach the gcd(n, Q) branch for an n without
    /// prime factors below 1000, so the test calls `lucas_check` with a chosen D.
    /// gp: gcd(1013*1019, (1-4053)/4) = 1013, 4053 % 4 = 1, kronecker(4053, 1032247) = -1.
    #[test]
    fn bpsw_gcd_with_q() {
        let n = BigInt::from(1_032_247u64);
        assert_eq!(jacobi_small(4053, &n), -1);
        assert_eq!(
            lucas_check(&n, 4053),
            Err(NotPrimeWitness::Divisor(Integer::from(1013)))
        );
        // gp: kronecker(-7, psi_12) = -1 and psi_12 fails the strong Lucas test.
        let psi = big("318665857834031151167461");
        assert_eq!(bpsw(&psi), Err(NotPrimeWitness::StrongLucas(-7)));
    }

    #[test]
    fn witness_confirms_checks_its_data() {
        let psi = Integer::from_decimal_str("318665857834031151167461").unwrap();
        // D = -8 is not 1 mod 4, so it is not a valid strong Lucas witness (review 2 n8).
        assert!(!NotPrimeWitness::StrongLucas(-8).confirms(&psi));
        assert!(NotPrimeWitness::StrongLucas(-7).confirms(&psi));
        // gp: isprime(2^127-1) = 1: no witness confirms a prime.
        let m127 = Integer::from(2).pow(127) - Integer::one();
        assert!(!NotPrimeWitness::MillerRabin(2).confirms(&m127));
        assert!(!NotPrimeWitness::StrongLucas(-7).confirms(&m127));
        assert!(!NotPrimeWitness::Divisor(Integer::one()).confirms(&m127));
        assert!(!NotPrimeWitness::Divisor(m127.clone()).confirms(&m127));
        assert!(!NotPrimeWitness::Square.confirms(&m127));
        assert!(!NotPrimeWitness::BelowTwo.confirms(&m127));
        assert!(NotPrimeWitness::Square.confirms(&Integer::from(49)));
        assert!(!NotPrimeWitness::Square.confirms(&Integer::from(1)));
        assert!(NotPrimeWitness::BelowTwo.confirms(&Integer::from(-7)));
        // A base out of range is no witness.
        assert!(!NotPrimeWitness::MillerRabin(1).confirms(&Integer::from(15)));
        assert!(!NotPrimeWitness::MillerRabin(15).confirms(&Integer::from(15)));
        assert!(NotPrimeWitness::MillerRabin(2).confirms(&Integer::from(15)));
    }

    #[test]
    fn strip_and_least_divisor() {
        let b = batches_below_2_16();
        // m = 2^5 * 65521 * (2^61 - 1); gp: isprime(65521), isprime(2^61-1)
        let p61 = BigInt::from((1u64 << 61) - 1);
        let m = BigInt::from(32u32) * 65521u32 * &p61;
        let (k, q) = b.strip(&m);
        assert_eq!(k, BigInt::from(32u64 * 65521));
        assert_eq!(q, p61);
        assert_eq!(
            batches_below_1000().least_divisor(&BigInt::from(997u32 * 991)),
            Some(991)
        );
        assert_eq!(batches_below_1000().least_divisor(&p61), None);
    }
}
