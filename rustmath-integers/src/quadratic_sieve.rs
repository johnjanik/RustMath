//! The self-initialising quadratic sieve (SIQS) for integer factorization.
//!
//! MAGMA Handbook ch. 18 §18.10.3 `MPQS(n)` (Lenstra's multiple polynomial quadratic sieve;
//! SIQS is the same family) and §18.10.1 (`MPQSLimit` of `Factorization`); digest
//! `rustmath-lava/docs/Chapter_18_Ring_of_Integers.md`. References:
//! - C. Pomerance, *The quadratic sieve factoring algorithm*, EUROCRYPT '84, LNCS 209 (1985)
//!   169-182.
//! - R. D. Silverman, *The multiple polynomial quadratic sieve*, Math. Comp. 48 (1987) 329-339.
//! - W. R. Alford, C. Pomerance, *Implementing the self-initializing quadratic sieve on a
//!   distributed network*, World Scientific 1995, 163-174.
//! - S. P. Contini, *Factoring integers with the self-initializing quadratic sieve*, MSc thesis,
//!   University of Georgia, 1997.
//! - R. Crandall, C. Pomerance, *Prime Numbers: A Computational Perspective*, 2nd ed., Springer
//!   2005, §6.1.
//! - D. E. Knuth, TAOCP vol. 2, 3rd ed., §4.5.4 (Knuth-Schroeppel multiplier).
//! - H. Cohen, GTM 138, Algorithm 1.5.1 (Tonelli-Shanks), §10.4.
//!
//! # Algorithm
//!
//! Step 0 settles the easy cases: n = +-1, the factor 2, the odd primes up to 47 and perfect
//! powers. Then, on N = |n|:
//! 1. a Knuth-Schroeppel multiplier k;
//! 2. a factor base of F primes p with (kN/p) != -1 (a prime of the scan that divides N is
//!    returned at once);
//! 3. polynomials Q(x) = (a x + b)^2 - kN = a g(x) with a = q_1 ... q_s a product of factor base
//!    primes and the 2^(s-1) values of b switched by a Gray code (Contini);
//! 4. a log sieve of g(x) over x in [-M, M) by the primes above 30;
//! 5. relations y^2 - kN = +-prod p^e (full) or +-prod p^e L with one large prime L (partial;
//!    two partials with the same L make one matrix row), each checked exactly; repeated |y| are
//!    dropped;
//! 6. a basis of the left nullspace of the parity matrix over GF(2) (dense Gaussian
//!    elimination, at least `extra_relations` vectors);
//! 7. for each vector, X = prod y and Y = sqrt(prod Q) modulo N, and gcd(X - Y, N).
//!
//! # Cost
//!
//! Measured with the defaults of [`QsParams::for_bits`] on semiprimes with two primes of equal
//! size (release build): 6 ms at 20 digits, 77 ms at 40, 0.35 s at 50, 2.4 s at 60, 39 s at 70
//! digits, about 4.5 times more per 5 digits (PARI/GP 2.15.4 `factor`: 2.4 s at 60 and 31 s at
//! 70 digits). A debug build is 5 to 7 times slower. The cost depends on the size of the
//! composite, not of its factors.
//!
//! # Examples
//!
//! ```
//! use rustmath_integers::Integer;
//! use rustmath_integers::quadratic_sieve::quadratic_sieve_factor;
//!
//! // gp: factor(27005010885249061781) = [3141592661, 1; 8595961921, 1]
//! let n = Integer::from_decimal_str("27005010885249061781").unwrap();
//! let d = quadratic_sieve_factor(&n).expect("a composite without small factors");
//! assert!(d == Integer::from(3141592661u64) || d == Integer::from(8595961921u64));
//! ```

mod gf2;
mod siqs;

use crate::factorization::{limited_engine, QsPolicy, SplitPlan};
use crate::prime::{is_probable_prime, POLLARD_RHO_DEFAULT_ITERATIONS};
use crate::Integer;
use num_bigint::BigUint;
use rustmath_core::{MathError, Result};

/// Parameters of one SIQS run (MAGMA ch. 18 §18.10.3 `MPQS`; Contini 1997).
///
/// Memory of a run, with R = F + 1 + extra: the sieve array, 2 M bytes (at most 32 MiB); the
/// factor base, roots and Bainv2 table, about (20 + 4 s) F bytes with s <= 64 a-factors (at most
/// 18 MB); the relations, at most R + 16 F of them (at most 16 F large primes keep a first
/// partial relation, and the run stops at R rows), measured 3.7 to 4.9 F from 20 to 55 digits
/// at about 300 bytes each; the set of a-values drawn, one entry of s indices per accepted
/// a-value, at most ceil(`max_polynomials` / 2^(s-1)) entries (small with the defaults, but a
/// large `max_polynomials` with a small s makes it grow); the dense GF(2) matrix,
/// R (F + 1 + R) / 8 bytes, about F^2 / 4 bytes (0.17 MB at F = 802, the default for a 40-digit
/// n of 131 bits; 3.6 MB at F = 3794 for a 60-digit n of 198 bits; 1.07 GB at the cap
/// F = 65536). Measured peak: 14 MB at 60 digits, 34 MB at 70 digits. At the caps (F = 65536,
/// M = 2^24) about 1.5 GB (estimate: 1.1 million relations of about 350 bytes; a relation
/// grows slowly with n, as |y| has about bits(n)/2 bits). Time of the matrix step:
/// O(R^2 (F + R) / 64) word operations (0.15 s in release at F = 3901 in the prototype, about
/// 12 minutes at the cap). Time of the sieve: at most `max_polynomials` polynomials (checked
/// before every one), each O(M + sum M/p) plus O(F) word operations per candidate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QsParams {
    /// Number of primes in the factor base, 2 included. `MIN_FACTOR_BASE_SIZE ..=
    /// MAX_FACTOR_BASE_SIZE`.
    pub factor_base_size: usize,
    /// Half width M of the sieve interval [-M, M). `MIN_SIEVE_HALF_WIDTH ..=
    /// MAX_SIEVE_HALF_WIDTH`; rounded up to a multiple of 64.
    pub sieve_half_width: usize,
    /// Partial relations with one prime cofactor below min(this * p_max, p_max^2) are kept (the
    /// product saturates), where p_max is the largest factor base prime. >= 1.
    pub large_prime_multiplier: u64,
    /// Bits subtracted from the expected log2 |g(x)| in the sieve threshold. Any value; a large
    /// value only makes more locations candidates (each is checked exactly).
    pub threshold_slack_bits: u8,
    /// Matrix rows beyond F + 1; the nullspace has at least this dimension.
    /// `1 ..= MAX_EXTRA_RELATIONS`.
    pub extra_relations: usize,
    /// Polynomials sieved before the run gives up with [`QsOutcome::OutOfPolynomials`]; the
    /// count is checked before every polynomial, so no more than this many are sieved. >= 1.
    pub max_polynomials: u64,
    /// Seed of the splitmix64 generator that draws the a-factors. Any value.
    pub seed: u64,
}

/// (bits, M, large prime multiplier) of the default table, interpolated linearly.
const DEFAULT_TABLE: [(f64, f64, f64); 13] = [
    (40.0, 4096.0, 10.0),
    (60.0, 8192.0, 20.0),
    (80.0, 16384.0, 30.0),
    (100.0, 16384.0, 40.0),
    (120.0, 32768.0, 40.0),
    (140.0, 32768.0, 50.0),
    (160.0, 32768.0, 60.0),
    (180.0, 32768.0, 70.0),
    (200.0, 65536.0, 80.0),
    (220.0, 65536.0, 90.0),
    (240.0, 65536.0, 100.0),
    (260.0, 98304.0, 100.0),
    (280.0, 131072.0, 100.0),
];

impl QsParams {
    pub const MIN_FACTOR_BASE_SIZE: usize = 30;
    /// The dense matrix takes about F^2 / 4 bytes: 1.08 GB here.
    pub const MAX_FACTOR_BASE_SIZE: usize = 65536;
    pub const MIN_SIEVE_HALF_WIDTH: usize = 64;
    /// The sieve array takes 2 M bytes: 32 MiB here. The default table never exceeds 2^17.
    pub const MAX_SIEVE_HALF_WIDTH: usize = 1 << 24;
    pub const MAX_EXTRA_RELATIONS: usize = 1024;

    /// The defaults for an n of `bits` bits. Defined for every u64 and always valid.
    ///
    /// F = clamp(round(3.4 L(n)^0.27), 60, `MAX_FACTOR_BASE_SIZE`) with L(n) =
    /// exp(sqrt(ln n ln ln n)) and ln n = max(bits, 40) ln 2 (a fit to the measured optimum; the
    /// cap is reached at about 106 digits). M and the large prime multiplier come from a table
    /// in bits from 40 to 280, interpolated linearly, M rounded to a multiple of 64, the end
    /// values outside: M = 4096 .. 131072, multiplier 10 .. 100. Threshold slack 12 bits, 32
    /// extra relations, `max_polynomials` = 1000 F, seed 1.
    pub fn for_bits(bits: u64) -> QsParams {
        let bf = bits.max(40) as f64;
        let ln = bf * std::f64::consts::LN_2;
        let l = (ln * ln.ln()).sqrt().exp();
        // For huge bits the f64 L(n) is infinite and the clamp gives the cap.
        let f = (3.4 * l.powf(0.27))
            .round()
            .clamp(60.0, Self::MAX_FACTOR_BASE_SIZE as f64) as usize;
        let last = DEFAULT_TABLE.len() - 1;
        let b = bf.clamp(DEFAULT_TABLE[0].0, DEFAULT_TABLE[last].0);
        let mut i = 0;
        while i + 2 < DEFAULT_TABLE.len() && DEFAULT_TABLE[i + 1].0 < b {
            i += 1;
        }
        let (b0, m0, l0) = DEFAULT_TABLE[i];
        let (b1, m1, l1) = DEFAULT_TABLE[i + 1];
        let t = ((b - b0) / (b1 - b0)).clamp(0.0, 1.0);
        let lerp = |x: f64, y: f64| x + t * (y - x);
        QsParams {
            factor_base_size: f,
            sieve_half_width: ((lerp(m0, m1) / 64.0).round() * 64.0) as usize,
            large_prime_multiplier: lerp(l0, l1).round() as u64,
            threshold_slack_bits: 12,
            extra_relations: 32,
            max_polynomials: 1000 * f as u64,
            seed: 1,
        }
    }

    /// `Ok(())` iff every field is in its documented range.
    ///
    /// # Errors
    ///
    /// `InvalidArgument` naming the first field out of range and its range.
    pub fn validate(&self) -> Result<()> {
        let bad = |msg: String| Err(MathError::InvalidArgument(format!("QsParams: {msg}")));
        if !(Self::MIN_FACTOR_BASE_SIZE..=Self::MAX_FACTOR_BASE_SIZE)
            .contains(&self.factor_base_size)
        {
            return bad(format!(
                "factor_base_size = {} is outside [{}, {}]",
                self.factor_base_size,
                Self::MIN_FACTOR_BASE_SIZE,
                Self::MAX_FACTOR_BASE_SIZE
            ));
        }
        if !(Self::MIN_SIEVE_HALF_WIDTH..=Self::MAX_SIEVE_HALF_WIDTH)
            .contains(&self.sieve_half_width)
        {
            return bad(format!(
                "sieve_half_width = {} is outside [{}, {}]",
                self.sieve_half_width,
                Self::MIN_SIEVE_HALF_WIDTH,
                Self::MAX_SIEVE_HALF_WIDTH
            ));
        }
        if self.large_prime_multiplier < 1 {
            return bad("large_prime_multiplier = 0 must be at least 1".to_string());
        }
        if !(1..=Self::MAX_EXTRA_RELATIONS).contains(&self.extra_relations) {
            return bad(format!(
                "extra_relations = {} is outside [1, {}]",
                self.extra_relations,
                Self::MAX_EXTRA_RELATIONS
            ));
        }
        if self.max_polynomials < 1 {
            return bad("max_polynomials = 0 must be at least 1".to_string());
        }
        Ok(())
    }
}

/// Result of one sieve run. None of the variants is a statement about the primality of n.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QsOutcome {
    /// A proper divisor d of |n| (1 < d < |n|, d | n), not necessarily prime.
    Factor(Integer),
    /// Every dependency gave gcd 1 or |n|. Always so for a prime (`dependencies: 0` when step 0
    /// or the factor base scan meets |n| itself); probability about 2^-dependencies for a
    /// composite that is not a prime power.
    NoFactor { dependencies: usize },
    /// The polynomial budget or the supply of a-values ran out before `needed` matrix rows were
    /// found; `relations` is the number of rows found. `relations: 0` can also mean that the
    /// a-supply guard fired before any sieving: the factor base has too few primes above 30 to
    /// build an a of the size n needs (or it would need more than 64 a-factors).
    OutOfPolynomials { relations: usize, needed: usize },
}

/// Step 0 on N = |n| >= 1: `Some(Factor(d))` or `Some(NoFactor { dependencies: 0 })` when it
/// decides, `None` when N is odd, N > 47, N has no prime factor <= 47 and is not a perfect
/// power (the precondition of the sieve core).
fn quick_paths(n: &Integer) -> Option<QsOutcome> {
    const ODD_PRIMES_TO_47: [u32; 14] = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
    let no_factor = QsOutcome::NoFactor { dependencies: 0 };
    if n.is_one() {
        return Some(no_factor);
    }
    if n.is_even() {
        return Some(if *n == Integer::from(2) {
            no_factor
        } else {
            QsOutcome::Factor(Integer::from(2))
        });
    }
    for p in ODD_PRIMES_TO_47 {
        let p = Integer::from(p);
        if (n % &p).is_zero() {
            return Some(if *n == p {
                no_factor
            } else {
                QsOutcome::Factor(p)
            });
        }
    }
    // The base of a perfect power is minimal, so 1 < b < N.
    n.is_perfect_power()
        .map(|(base, _)| QsOutcome::Factor(base))
}

fn magnitude(n: &Integer) -> BigUint {
    n.as_bigint().magnitude().clone()
}

/// One factor of n, or `None`.
///
/// Step 0 (n = +-1, the primes up to 47, perfect powers), then `None` when |n| is a probable
/// prime ([`is_probable_prime`], Baillie-PSW, exact below psi_12), then the sieve with
/// [`QsParams::for_bits`] and seeds 1, 2, 3 until it returns a factor. `None` never claims that
/// n is prime: it is also the answer for 0, and (with probability about 2^-96) for a composite
/// on which all three runs failed. The factor need not be prime. Negative n: the divisor is
/// positive.
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::quadratic_sieve::quadratic_sieve_factor;
///
/// // 15347 = 103 · 149 (gp: factor(15347)). A prime factor below the factor base bound is
/// // found by the factor-base scan.
/// let n = Integer::from(15347);
/// assert_eq!(quadratic_sieve_factor(&n), Some(Integer::from(103)));
/// ```
pub fn quadratic_sieve_factor(n: &Integer) -> Option<Integer> {
    if n.is_zero() {
        return None;
    }
    let abs = n.abs();
    if let Some(outcome) = quick_paths(&abs) {
        return match outcome {
            QsOutcome::Factor(d) => Some(d),
            _ => None,
        };
    }
    if is_probable_prime(&abs) {
        return None;
    }
    let big = magnitude(&abs);
    let mut params = QsParams::for_bits(big.bits());
    for seed in 1..=3 {
        params.seed = seed;
        if let (QsOutcome::Factor(d), _) = siqs::run(&big, &params) {
            return Some(d);
        }
    }
    None
}

/// One sieve run with the given parameters, after step 0 and without the probable-prime check
/// (a prime n > 47 runs the sieve and gives `NoFactor`).
///
/// `Ok(NoFactor { dependencies: 0 })` for n = +-1 and n = +-p with p a prime <= 47. MAGMA's
/// `MPQS(n)` requires n > 5 * 10^24 (digest §18.10.3); this function accepts every n: step 0 and
/// the factor base scan settle small n, and the sieve itself works from about 8 digits on.
///
/// # Errors
///
/// `InvalidArgument` for n = 0, and when `params.validate()` fails.
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::quadratic_sieve::{
///     quadratic_sieve_factor_with_params, QsOutcome, QsParams,
/// };
///
/// // gp: factor(270050147102784401) = [314159311, 1; 859596191, 1]
/// let n = Integer::from(270050147102784401u64);
/// let outcome = quadratic_sieve_factor_with_params(&n, &QsParams::for_bits(58)).unwrap();
/// let QsOutcome::Factor(d) = outcome else { panic!("{outcome:?}") };
/// assert!(d == Integer::from(314159311) || d == Integer::from(859596191));
/// ```
pub fn quadratic_sieve_factor_with_params(n: &Integer, params: &QsParams) -> Result<QsOutcome> {
    if n.is_zero() {
        return Err(MathError::InvalidArgument(
            "quadratic_sieve_factor_with_params: n = 0 has no factorization".to_string(),
        ));
    }
    params.validate()?;
    let abs = n.abs();
    if let Some(outcome) = quick_paths(&abs) {
        return Ok(outcome);
    }
    Ok(siqs::run(&magnitude(&abs), params).0)
}

/// Complete factorization of |n| with the sieve: `(prime, exponent)` pairs, primes ascending
/// (PARI/GP `factor(n)` without the sign; MAGMA `Factorization(n)`, ch. 18 §18.10). Empty for
/// 0 and +-1.
///
/// # Algorithm
///
/// The engine of [`crate::factorization::factorization_with_limits`] with trial division by
/// the primes up to 47, Brent rho with 8191 iterations per composite, then the sieve
/// ([`quadratic_sieve_factor_with_params`] with [`QsParams::for_bits`] and seeds 1, 2, 3), and
/// the complete rho fallback if the sieve fails three times. A divisor the sieve returns need
/// not be prime: it goes back to the queue. Every returned factor is proven prime (trial
/// division, exactness below psi_12 = 318665857834031151167461, or a verified ECPP
/// certificate); psi_12 = 399165290221 · 798330580441 is split.
///
/// # Cost
///
/// The sieve's cost depends on the size of the composite, not of its factors (see the module
/// doc: 77 ms at 40 digits, 2.4 s at 60 digits, 39 s at 70 digits in release). Rho with 8191
/// iterations runs first and usually removes prime factors below about 10^7 at once.
///
/// # Panics
///
/// When a factor passed BPSW but its ECPP proof was unresolved (no unresolved prime up to 2048
/// bits in the experiments; about 1 in 16 at 3072 and 4096 bits), with "the factor {q} passed
/// BPSW but ECPP found no proof". [`crate::factorization::factorization_with_limits`] never
/// panics and reports such a factor in `probable_primes`.
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::quadratic_sieve::quadratic_sieve_factor_complete;
///
/// let n = Integer::from(1001); // 7 × 11 × 13
/// let factors = quadratic_sieve_factor_complete(&n);
/// assert_eq!(factors.len(), 3);
///
/// // 12780049 = 53 · 59 · 61 · 67 (PARI/GP: factor(12780049)); Brent rho with 8191
/// // iterations splits it before the sieve is needed.
/// let factors = quadratic_sieve_factor_complete(&Integer::from(12780049));
/// let primes: Vec<Integer> = [53, 59, 61, 67].into_iter().map(Integer::from).collect();
/// assert_eq!(factors, primes.into_iter().map(|p| (p, 1)).collect::<Vec<_>>());
///
/// // gp: factor(270050107561429816771648674403) = [314159265359057, 1; 859596190017779, 1]
/// let n = Integer::from_decimal_str("270050107561429816771648674403").unwrap();
/// let p = Integer::from(314159265359057u64);
/// let q = Integer::from(859596190017779u64);
/// assert_eq!(quadratic_sieve_factor_complete(&n), vec![(p, 1), (q, 1)]);
/// ```
pub fn quadratic_sieve_factor_complete(n: &Integer) -> Vec<(Integer, u32)> {
    let plan = SplitPlan {
        rho_c: Integer::one(),
        rho_x0: Integer::one(),
        rho_steps: POLLARD_RHO_DEFAULT_ITERATIONS,
        qs: QsPolicy::Always,
        complete: true,
    };
    let result = limited_engine(n, 47, &plan, true);
    if let Some((q, _)) = result.probable_primes.first() {
        panic!(
            "quadratic_sieve_factor_complete: the factor {q} passed BPSW but ECPP found no \
             proof; use factorization::factorization_with_limits, which reports it in \
             probable_primes"
        );
    }
    assert!(
        result.composites.is_empty(),
        "quadratic_sieve_factor_complete: the complete plan left a composite"
    );
    result.primes
}

#[cfg(test)]
mod tests {
    use super::*;

    /// HEAD's `test_quadratic_sieve_small`, rewritten: 143 = 11 · 13 is settled by step 0.
    #[test]
    fn test_quadratic_sieve_small() {
        // gp: factor(143) = [11,1; 13,1]
        let n = Integer::from(143);
        assert_eq!(
            quadratic_sieve_factor_with_params(&n, &QsParams::for_bits(8)),
            Ok(QsOutcome::Factor(Integer::from(11)))
        );
    }

    /// HEAD's `test_quadratic_sieve_larger`, rewritten: 1001 = 7 · 11 · 13.
    #[test]
    fn test_quadratic_sieve_larger() {
        // gp: factor(1001) = [7,1; 11,1; 13,1]
        let n = Integer::from(1001);
        assert_eq!(
            quadratic_sieve_factor_with_params(&n, &QsParams::for_bits(10)),
            Ok(QsOutcome::Factor(Integer::from(7)))
        );
    }

    #[test]
    fn test_quadratic_sieve_prime() {
        let p = Integer::from(97);
        let factor = quadratic_sieve_factor(&p);
        assert!(factor.is_none());
    }

    #[test]
    fn test_quadratic_sieve_even() {
        let n = Integer::from(100);
        let factor = quadratic_sieve_factor(&n);
        assert_eq!(factor, Some(Integer::from(2)));
    }

    #[test]
    fn test_complete_factorization() {
        let n = Integer::from(60); // 2² × 3 × 5
        let factors = quadratic_sieve_factor_complete(&n);

        // Verify factorization
        let mut product = Integer::one();
        for (prime, exp) in &factors {
            for _ in 0..*exp {
                product = product * prime.clone();
            }
        }
        assert_eq!(product, n);
    }

    /// 165731 = 53^2 · 59. HEAD's sieve returned the composite 3127 = 53 · 59 here, and the
    /// driver once pushed it as a prime. Now the factor base scan meets 53 first (a property
    /// of the scan order, not of random choices). The path of a composite sieve divisor is
    /// tested by `factorization::tests::engine_requeues_composite_divisors`.
    #[test]
    fn test_complete_factorization_repeated_prime() {
        // gp: factor(165731) = [53,2; 59,1]
        let n = Integer::from(165731);
        assert_eq!(quadratic_sieve_factor(&n), Some(Integer::from(53)));
        assert_eq!(
            quadratic_sieve_factor_complete(&n),
            vec![(Integer::from(53), 2), (Integer::from(59), 1)]
        );
    }

    /// Blocker B-04: HEAD's sieve found no factor of 3127 = 53 · 59, and the driver used to
    /// push 3127 itself as a prime. The factor base scan now finds 53.
    #[test]
    fn test_complete_factorization_3127() {
        // gp: factor(3127) = [53,1; 59,1]; isprime(3127) = 0
        let n = Integer::from(3127);
        assert_eq!(quadratic_sieve_factor(&n), Some(Integer::from(53)));
        assert_eq!(
            quadratic_sieve_factor_complete(&n),
            vec![(Integer::from(53), 1), (Integer::from(59), 1)]
        );
    }
}
