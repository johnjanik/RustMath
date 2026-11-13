//! Factorization under limits: MAGMA's `Factorization(n : ...)` with its third return value.
//!
//! MAGMA Handbook ch. 18 §18.10.1 (`Factorization` with `TrialDivisionLimit`,
//! `PollardRhoLimit`, `ECMLimit`, `MPQSLimit`, `Proof`, and the sequence of unfactored
//! composites it returns), §18.10.3 (`TrialDivision`, `PollardRho`, `MPQS`), §18.10.4
//! (`CoprimeBasis`); digest `rustmath-lava/docs/Chapter_18_Ring_of_Integers.md`. References:
//! - \[BDS93\] E. Bach, J. Driscoll, J. Shallit, *Factor refinement*, J. Algorithms 15 (1993)
//!   199-222 (the gcd-only refinement of the pieces).
//! - \[Bre80\] R. P. Brent, *An improved Monte Carlo factorization algorithm*, BIT 20 (1980)
//!   176-184 (rho).
//! - C. Bays, R. H. Hudson, *The segmented sieve of Eratosthenes and primes in arithmetic
//!   progressions to 10^12*, BIT 17 (1977) 121-127 (trial division over [`crate::sieve`]).
//! - Baillie-PSW and ECPP of [`crate::prime::try_is_prime`] for the proofs.
//!
//! The name avoids MAGMA's `PartialFactorization`, a different intrinsic (§18.10.4).
//!
//! # Relation to MAGMA
//!
//! - MAGMA's cascade is Cunningham lookup, trial division, perfect power, Pollard rho (Brent),
//!   SQUFOF, ECM and MPQS. This engine is trial division, perfect power, Brent rho and the
//!   quadratic sieve ([`crate::quadratic_sieve`]), with `ECMLimit := 0` fixed and no
//!   Cunningham table or SQUFOF. Leaving ECM out is a choice on measured cost, not an
//!   infeasibility: the crate's ECM ([`crate::ecm`], affine Weierstrass arithmetic, a stage 2 of
//!   100 primes) takes 164 ms per curve in release at B1 = 2000 on the 41-digit
//!   p12 * P30 = 10000000000300000000000000031900000000957, and only a few percent of the curves
//!   find the 12-digit factor, while the sieve splits that whole number in about 0.08 s. The
//!   price shows on large composites with a medium-size factor: without ECM such a number costs
//!   the sieve's time for the whole composite, hours to days from about 90 digits (one design
//!   review measured 0.45 s per curve and 3 of 160 curves for 100000000003 * nextprime(10^88),
//!   about 25 s expected with ECM, against an estimated 90 hours of sieving). In this crate a
//!   larger `pollard_rho_limit` is the way to reach factors up to about 12 to 14 digits (rho
//!   needs about sqrt(p) steps for a prime factor p).
//! - `MPQSLimit`: MAGMA's default is infinity, and a value below 25 disables MPQS. Here
//!   `mpqs_limit: None` is infinity, and `Some(x)` with x < 25 disables the sieve. With x >= 25
//!   the sieve takes the composites of at most x decimal digits. MAGMA sends composites below
//!   25 digits to SQUFOF and ECM instead; this engine has neither, so the sieve takes those too
//!   (6 ms at 20 digits in release).
//! - The third return value: MAGMA returns it only when both `ECMLimit` and `MPQSLimit` are
//!   set. `ECMLimit` is fixed at 0 here, so `composites` can be non-empty only when
//!   `mpqs_limit` is `Some`. With `None` the engine does not stop until the factorization is
//!   complete, as MAGMA's default does.
//! - `Proof` (default true) as in MAGMA. `Bases` is not offered: the probable-prime test is
//!   Baillie-PSW, not 20 Miller-Rabin bases.
//!
//! # Examples
//!
//! ```
//! use rustmath_integers::factorization::{factorization_with_limits, FactorizationLimits};
//! use rustmath_integers::Integer;
//!
//! // gp: factor(79839848800024750353128) = [2, 3; 9973, 1; 10007, 1; 100000000000031, 1]
//! let n = Integer::from_decimal_str("79839848800024750353128").unwrap();
//! let limits = FactorizationLimits {
//!     trial_division_limit: 10000,
//!     pollard_rho_limit: 0,
//!     mpqs_limit: Some(0),
//!     proof: true,
//! };
//! let f = factorization_with_limits(&n, &limits);
//! assert_eq!(f.primes, vec![(Integer::from(2), 3), (Integer::from(9973), 1)]);
//! // 10007 > 10000 stays in the unfactored composite, MAGMA's third return value.
//! let rest = Integer::from(1000700000000310217u64);
//! assert_eq!(f.composites, vec![(rest, 1)]);
//! assert_eq!(f.value(), n);
//! ```

use crate::primality::PSI_12;
use crate::prime::{is_probable_prime, rho_rounds, split_odd_composite, try_is_prime};
use crate::quadratic_sieve::{quadratic_sieve_factor_with_params, QsOutcome, QsParams};
use crate::sieve::for_each_prime;
use crate::Integer;
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use std::collections::HashMap;
use std::ops::ControlFlow;

/// A factorization of n computed under limits: MAGMA's factorisation sequence, sign and
/// sequence of unfactored composites (ch. 18 §18.10.1), plus the probable primes whose proof
/// was not obtained.
///
/// Invariants: every base is > 1; the bases of all three lists together are pairwise coprime;
/// each list is sorted by base; for n != 0, sign * prod b^e over the three lists equals n;
/// n = 0 gives sign 0 and empty lists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LimitedFactorization {
    /// -1, 0 or 1.
    pub sign: i8,
    /// (p, e): p proven prime (trial division, decided below psi_12, or a verified ECPP
    /// certificate).
    pub primes: Vec<(Integer, u32)>,
    /// (q, e): q passed Baillie-PSW ([`is_probable_prime`]), q >= psi_12, and q was not proven
    /// prime (`proof` was false, or the proof was unresolved).
    pub probable_primes: Vec<(Integer, u32)>,
    /// (m, e): m proven composite (`is_probable_prime(m)` is false, or the proof found m not
    /// prime) and not split by the enabled methods within their limits.
    pub composites: Vec<(Integer, u32)>,
}

impl LimitedFactorization {
    /// `probable_primes` and `composites` are both empty.
    pub fn is_complete(&self) -> bool {
        self.probable_primes.is_empty() && self.composites.is_empty()
    }

    /// sign * prod b^e over the three lists (n itself; 0 for sign 0).
    pub fn value(&self) -> Integer {
        if self.sign == 0 {
            return Integer::zero();
        }
        let product = self
            .primes
            .iter()
            .chain(&self.probable_primes)
            .chain(&self.composites)
            .fold(Integer::one(), |acc, (b, e)| acc * b.pow(*e));
        if self.sign < 0 {
            -product
        } else {
            product
        }
    }
}

/// The limits of [`factorization_with_limits`]: MAGMA `Factorization(n : TrialDivisionLimit,
/// PollardRhoLimit, MPQSLimit, Proof)` with `ECMLimit := 0`.
///
/// # Cost and memory
///
/// Trial division: one word remainder per prime up to min(limit, isqrt(cofactor)), from the
/// segmented sieve [`crate::sieve::for_each_prime`], called once below 2^40 and once from 2^40
/// up (memory O(1) in the limit: 32 KiB segments and base primes up to 2^20). Beyond 2^16 it
/// stops early once the cofactor passes [`is_probable_prime`]. Measured cost per prime in
/// release on a 30-digit cofactor: 65 ns near 10^7, 86 ns near 10^9, 113 ns near 10^10,
/// 171 ns near 10^11 and 308 ns near 10^12 (the sieve's share grows with the number of base
/// primes), and about 6 µs above 2^40, where the sieve confirms each survivor with
/// [`crate::prime::is_prime_u64`]. A debug build is about 7 times slower. So a limit of 10^9 on
/// a cofactor without small factors costs about 4 s in release, 10^12 about 2.5 hours, 2^40
/// about 2.7 hours, and each further 10^9 above 2^40 about 3.5 minutes.
///
/// Rho: `pollard_rho_limit` iterations per composite, memory O(size of n).
///
/// The sieve: [`QsParams::for_bits`] of the composite: memory 14 MB at 60 digits and 34 MB at
/// 70 digits (measured), at most about 1.5 GB (the caps of [`QsParams`], reached from about 106
/// digits; estimate); time about 2.4 s at 60 digits and 39 s at 70 digits in release, growing
/// about 4.5 times per 5 digits. With `mpqs_limit: None` there is no time limit: a large
/// composite with a medium-size factor costs the sieve's time for the whole composite (no ECM,
/// see the module doc).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FactorizationLimits {
    /// Trial division by the primes up to this bound (MAGMA default 10000; 0 or 1: none). Any
    /// u64.
    pub trial_division_limit: u64,
    /// Brent-rho iterations per composite (MAGMA default 8191; 0: no rho).
    pub pollard_rho_limit: u64,
    /// `None`: the sieve runs on every composite left (MAGMA default infinity) and the result is
    /// complete. `Some(x)` with x < 25: no sieve (MAGMA: "< 25 disables MPQS"). `Some(x)` with
    /// x >= 25: the sieve runs on composites of at most x decimal digits.
    pub mpqs_limit: Option<u32>,
    /// Prove every probable prime >= psi_12 (MAGMA `Proof`, default true).
    pub proof: bool,
}

impl Default for FactorizationLimits {
    /// MAGMA's defaults: 10000, 8191, infinity, `Proof := true`.
    fn default() -> Self {
        FactorizationLimits {
            trial_division_limit: 10000,
            pollard_rho_limit: crate::prime::POLLARD_RHO_DEFAULT_ITERATIONS,
            mpqs_limit: None,
            proof: true,
        }
    }
}

/// MAGMA `Factorization(n : TrialDivisionLimit, PollardRhoLimit, MPQSLimit, Proof)` with
/// `ECMLimit := 0` (ch. 18 §18.10.1; the module doc lists the differences to MAGMA).
///
/// Trial division by the primes up to `trial_division_limit`, then for each composite left: the
/// factor 2, the perfect-power test, Brent rho with `pollard_rho_limit` iterations, and the
/// sieve as `mpqs_limit` allows. With `mpqs_limit: None` a composite on which the sieve fails
/// three times is finished by the complete rho fallback, so the result has no `composites`.
/// Never panics: a factor whose ECPP proof is unresolved is reported in `probable_primes`.
pub fn factorization_with_limits(
    n: &Integer,
    limits: &FactorizationLimits,
) -> LimitedFactorization {
    let qs = match limits.mpqs_limit {
        None => QsPolicy::Always,
        Some(x) if x < 25 => QsPolicy::Never,
        Some(x) => QsPolicy::UpToDigits(x),
    };
    let plan = SplitPlan {
        rho_c: Integer::one(),
        rho_x0: Integer::one(),
        rho_steps: limits.pollard_rho_limit,
        qs,
        complete: limits.mpqs_limit.is_none(),
    };
    limited_engine(n, limits.trial_division_limit, &plan, limits.proof)
}

/// The class of a piece c > 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Class {
    Prime,
    ProbablePrime,
    Composite,
}

/// c > 1. `is_probable_prime(c)` false: `Composite` (a proof). True and c < psi_12: `Prime`
/// (exact there). Else, if `proof`: [`try_is_prime`] (Baillie-PSW, the cache of proven primes,
/// then ECPP): `Ok(true)` `Prime`, `Ok(false)` `Composite`, `Err` (unresolved) `ProbablePrime`;
/// without `proof`: `ProbablePrime`.
pub(crate) fn classify(c: &Integer, proof: bool) -> Class {
    if !is_probable_prime(c) {
        return Class::Composite;
    }
    if *c < Integer::new(BigInt::from(PSI_12)) {
        return Class::Prime;
    }
    if !proof {
        return Class::ProbablePrime;
    }
    match try_is_prime(c) {
        Ok(true) => Class::Prime,
        Ok(false) => Class::Composite,
        Err(_) => Class::ProbablePrime,
    }
}

/// Which composites the sieve may take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QsPolicy {
    Never,
    /// Composites of at most this many decimal digits.
    UpToDigits(u32),
    Always,
}

/// The methods [`split_composite`] may use.
#[derive(Clone, Debug)]
pub(crate) struct SplitPlan {
    pub(crate) rho_c: Integer,
    pub(crate) rho_x0: Integer,
    /// Brent-rho steps per composite (0: no rho).
    pub(crate) rho_steps: u64,
    pub(crate) qs: QsPolicy,
    /// The complete rho fallback ([`split_odd_composite`]) when the other methods fail.
    pub(crate) complete: bool,
}

/// Pieces of the composite c > 1 with c = prod b^k and every 1 < b < c, or `None` (only when
/// `!plan.complete`). In this order: c even gives [(2, v), (c / 2^v, 1)] (the second dropped if
/// 1); a perfect power b^j gives [(b, j)]; [`rho_rounds`] with the plan's c, x0 and steps gives
/// [(d, 1), (c / d, 1)]; then the sieve if `plan.qs` allows c, with [`QsParams::for_bits`] and
/// seeds 1, 2, 3 until it returns a factor; then, if complete, the rho fallback
/// [`split_odd_composite`] (which always succeeds).
pub(crate) fn split_composite(c: &Integer, plan: &SplitPlan) -> Option<Vec<(Integer, u32)>> {
    let one = Integer::one();
    let two = Integer::from(2);
    if c.is_even() {
        let mut v = 0u32;
        let mut r = c.clone();
        while r.is_even() {
            r = &r / &two;
            v += 1;
        }
        let mut out = vec![(two, v)];
        if r > one {
            out.push((r, 1));
        }
        return Some(out);
    }
    if let Some((b, j)) = c.is_perfect_power() {
        return Some(vec![(b, j)]);
    }
    let pair = |d: Integer| Some(vec![(c / &d, 1), (d, 1)]);
    if plan.rho_steps > 0 {
        if let Some(d) = rho_rounds(c, &plan.rho_c, &plan.rho_x0, plan.rho_steps) {
            return pair(d);
        }
    }
    let sieve_allowed = match plan.qs {
        QsPolicy::Never => false,
        QsPolicy::Always => true,
        QsPolicy::UpToDigits(x) => has_at_most_digits(c, x),
    };
    if sieve_allowed {
        let mut params = QsParams::for_bits(c.bits());
        for seed in 1..=3 {
            params.seed = seed;
            if let Ok(QsOutcome::Factor(d)) = quadratic_sieve_factor_with_params(c, &params) {
                return pair(d);
            }
        }
    }
    if plan.complete {
        return pair(split_odd_composite(c));
    }
    None
}

/// True iff 0 < c < 10^d, without forming 10^d unless needed: c <= 0 gives false;
/// bits(c) <= 3d gives true (c < 8^d <= 10^d); bits(c) > 4d gives false (c >= 2^(4d) = 16^d >
/// 10^d; for d = 0 every c > 0 has bits > 0); otherwise c is compared with 10^d exactly (then
/// 10^d has about as many bits as c).
pub(crate) fn has_at_most_digits(c: &Integer, d: u32) -> bool {
    if c.signum() <= 0 {
        return false;
    }
    let bits = c.bits();
    let d64 = u64::from(d);
    if bits <= 3 * d64 {
        return true;
    }
    if bits > 4 * d64 {
        return false;
    }
    *c < Integer::from(10).pow(d)
}

/// Factor refinement [BDS93]: while two bases share g = gcd > 1, replace (a^e, b^f) by
/// ((a/g)^e, (b/g)^f, g^(e+f)) (merged when a = b), dropping 1s. The product is preserved; the
/// sum of log2 of the bases drops by at least 1 per step, so it ends after at most (total bits)
/// steps. Bases <= 1 in the input are dropped.
pub(crate) fn coprime_refine(pieces: Vec<(Integer, u32)>) -> Vec<(Integer, u32)> {
    let one = Integer::one();
    let mut v: Vec<(Integer, u32)> = pieces.into_iter().filter(|(b, _)| *b > one).collect();
    'outer: loop {
        for i in 0..v.len() {
            for j in i + 1..v.len() {
                let g = v[i].0.gcd(&v[j].0);
                if g.is_one() {
                    continue;
                }
                let (b, f) = v.remove(j);
                let (a, e) = v.remove(i);
                let ef = e.checked_add(f).expect("coprime_refine: exponent overflow");
                if a == b {
                    v.push((a, ef));
                } else {
                    for piece in [(&a / &g, e), (&b / &g, f), (g, ef)] {
                        if piece.0 > one {
                            v.push(piece);
                        }
                    }
                }
                continue 'outer;
            }
        }
        return v;
    }
}

/// floor(sqrt(m)) as a u64, saturating at u64::MAX (so m >= 2^128 gives u64::MAX).
fn isqrt_sat(m: &Integer) -> u64 {
    m.sqrt()
        .ok()
        .and_then(|r| r.as_bigint().to_u64())
        .unwrap_or(u64::MAX)
}

/// The state of [`trial_division_stage`] between its two sieve calls.
struct TrialDivision<'a> {
    m: &'a mut Integer,
    bound: u64,
    /// m changed since the last probable-prime test (or was never tested).
    changed: bool,
    stopped: bool,
    visited: u64,
    found: Vec<(Integer, u32)>,
}

impl TrialDivision<'_> {
    fn step(&mut self, p: u64) -> ControlFlow<()> {
        self.visited += 1;
        if p > self.bound {
            self.stopped = true;
            return ControlFlow::Break(());
        }
        if p > 1 << 16 && self.changed {
            self.changed = false;
            if is_probable_prime(self.m) {
                self.stopped = true;
                return ControlFlow::Break(());
            }
        }
        let r = self.m.as_bigint().magnitude() % p;
        if r.to_u64() == Some(0) {
            let pi = Integer::from(p);
            let mut e = 0u32;
            while (&*self.m % &pi).is_zero() {
                *self.m = &*self.m / &pi;
                e += 1;
            }
            self.found.push((pi, e));
            self.changed = true;
            self.bound = self.bound.min(isqrt_sat(self.m));
            if self.m.is_one() {
                self.stopped = true;
                return ControlFlow::Break(());
            }
        }
        ControlFlow::Continue(())
    }
}

/// Trial division of m by the primes p with lo <= p <= limit and p <= isqrt(m), ascending,
/// through two [`for_each_prime`] calls split at 2^40: a call with hi <= 2^40 does no survivor
/// test, and only the second call has survivors that the sieve confirms with
/// [`crate::prime::is_prime_u64`]. Divides each prime found out of m with its exact exponent.
/// Beyond 2^16 it stops early once m passes [`is_probable_prime`] (tested only when m has
/// changed since the last test), so a limit up to 2^16 pays no BPSW test. Returns the primes
/// found and the number of primes passed to the sieve's closure.
///
/// The stage stops only when every prime up to the bound was tried (so m has no prime factor
/// <= min(limit, sqrt(m))), or m = 1, or m passed `is_probable_prime`. No `limit + 1` and no
/// square of the limit is formed.
pub(crate) fn trial_division_stage(
    m: &mut Integer,
    lo: u64,
    limit: u64,
) -> (Vec<(Integer, u32)>, u64) {
    const SPLIT: u64 = 1 << 40;
    if limit < 2 || *m <= Integer::one() {
        return (Vec::new(), 0);
    }
    let bound = limit.min(isqrt_sat(m));
    let mut st = TrialDivision {
        m,
        bound,
        changed: true,
        stopped: false,
        visited: 0,
        found: Vec::new(),
    };
    if lo < SPLIT {
        let hi = st.bound.saturating_add(1).min(SPLIT);
        for_each_prime(lo, hi, |p| st.step(p));
    }
    if !st.stopped && st.bound >= SPLIT {
        let hi = st.bound.saturating_add(1);
        for_each_prime(lo.max(SPLIT), hi, |p| st.step(p));
    }
    (st.found, st.visited)
}

/// The engine behind [`factorization_with_limits`],
/// [`crate::prime::factor_pollard_rho_limited`] and
/// [`crate::quadratic_sieve::quadratic_sieve_factor_complete`].
///
/// 1. n = 0: sign 0 and empty lists. Else sign = signum(n), m = |n|.
/// 2. [`trial_division_stage`] with lo = 2, if `trial_division_limit >= 2`; then m > 1 goes to
///    the queue.
/// 3. Queue of (c, e): [`classify`]; a prime or probable prime is recorded; a composite is split
///    by [`split_composite`] and its pieces (b, k) are queued as (b, k e), or it is recorded as
///    a composite when the plan cannot split it. Every split replaces c by bases below c, so the
///    loop ends.
/// 4. [`coprime_refine`] over all recorded pieces. A piece equal to a recorded base keeps its
///    class; a new piece (a quotient or gcd of recorded composites) is classified and not split
///    again; a composite perfect power b^j among them becomes (b, j e).
/// 5. The three lists are sorted by base.
pub(crate) fn limited_engine(
    n: &Integer,
    trial_division_limit: u64,
    plan: &SplitPlan,
    proof: bool,
) -> LimitedFactorization {
    let mut out = LimitedFactorization {
        sign: n.signum(),
        primes: Vec::new(),
        probable_primes: Vec::new(),
        composites: Vec::new(),
    };
    if n.is_zero() {
        return out;
    }
    let mut m = n.abs();
    let mut recorded: Vec<(Integer, u32)> = Vec::new();
    let mut class_of: HashMap<Integer, Class> = HashMap::new();
    if trial_division_limit >= 2 && m > Integer::one() {
        let (found, _) = trial_division_stage(&mut m, 2, trial_division_limit);
        for (p, e) in found {
            class_of.insert(p.clone(), Class::Prime);
            recorded.push((p, e));
        }
    }
    let mut queue: Vec<(Integer, u32)> = Vec::new();
    if m > Integer::one() {
        queue.push((m, 1));
    }
    while let Some((c, e)) = queue.pop() {
        let class = classify(&c, proof);
        if class == Class::Composite {
            if let Some(pieces) = split_composite(&c, plan) {
                for (b, k) in pieces {
                    let ke = k.checked_mul(e).expect("limited_engine: exponent overflow");
                    queue.push((b, ke));
                }
                continue;
            }
        }
        class_of.insert(c.clone(), class);
        recorded.push((c, e));
    }
    for (b, e) in coprime_refine(recorded) {
        let (b, e, class) = match class_of.get(&b) {
            Some(&class) => (b, e, class),
            None => match classify(&b, proof) {
                Class::Composite => match b.is_perfect_power() {
                    Some((root, j)) => {
                        let je = j.checked_mul(e).expect("limited_engine: exponent overflow");
                        let class = classify(&root, proof);
                        (root, je, class)
                    }
                    None => (b, e, Class::Composite),
                },
                class => (b, e, class),
            },
        };
        match class {
            Class::Prime => out.primes.push((b, e)),
            Class::ProbablePrime => out.probable_primes.push((b, e)),
            Class::Composite => out.composites.push((b, e)),
        }
    }
    out.primes.sort();
    out.probable_primes.sort();
    out.composites.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn z(s: &str) -> Integer {
        Integer::from_decimal_str(s).unwrap_or_else(|| panic!("bad integer {s:?}"))
    }

    fn pieces(v: &[(&str, u32)]) -> Vec<(Integer, u32)> {
        v.iter().map(|&(b, e)| (z(b), e)).collect()
    }

    fn product(v: &[(Integer, u32)]) -> Integer {
        v.iter().fold(Integer::one(), |acc, (b, e)| acc * b.pow(*e))
    }

    fn sorted(mut v: Vec<(Integer, u32)>) -> Vec<(Integer, u32)> {
        v.sort();
        v
    }

    #[test]
    fn coprime_refine_values() {
        // gp: factor(60) = [2,2; 3,1; 5,1]
        assert_eq!(
            sorted(coprime_refine(pieces(&[("6", 1), ("10", 1)]))),
            pieces(&[("2", 2), ("3", 1), ("5", 1)])
        );
        // gp: factor(1380*675*3408*654) = [2,7; 3,6; 5,3; 23,1; 71,1; 109,1]
        let got = sorted(coprime_refine(pieces(&[
            ("1380", 1),
            ("675", 1),
            ("3408", 1),
            ("654", 1),
        ])));
        assert_eq!(
            got,
            pieces(&[
                ("2", 7),
                ("3", 6),
                ("5", 3),
                ("23", 1),
                ("71", 1),
                ("109", 1)
            ])
        );
        assert_eq!(
            coprime_refine(pieces(&[("12", 1), ("12", 2)])),
            pieces(&[("12", 3)])
        );
        assert_eq!(
            coprime_refine(pieces(&[("1", 4), ("7", 2)])),
            pieces(&[("7", 2)])
        );
    }

    #[test]
    fn coprime_refine_random() {
        let small = [2u32, 3, 5, 7, 11, 13];
        let mut state = 11u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        for _ in 0..200 {
            let len = 2 + next() % 5;
            let input: Vec<(Integer, u32)> = (0..len)
                .map(|_| {
                    let mut b = Integer::one();
                    for _ in 0..1 + next() % 4 {
                        b = b * Integer::from(small[next() % small.len()]);
                    }
                    (b, 1 + (next() % 3) as u32)
                })
                .collect();
            let out = coprime_refine(input.clone());
            assert_eq!(product(&out), product(&input), "{input:?}");
            for i in 0..out.len() {
                assert!(out[i].0 > Integer::one());
                for j in i + 1..out.len() {
                    assert!(out[i].0.gcd(&out[j].0).is_one(), "{out:?}");
                }
            }
        }
    }

    /// The stage crosses 2^40 with two sieve calls.
    #[test]
    fn trial_division_crosses_2_40() {
        // gp: precprime(2^40) = 1099511627689; nextprime(2^40) = 1099511627791;
        //     precprime(2^40) * nextprime(2^40) * nextprime(10^29) =
        //     120892581953546433750499900385647336431813123664094681;
        //     c = 0; forprime(p = 2^40 - 2^16, 2^40, c++); c = 2363;
        //     nextprime(nextprime(2^40) + 1) = 1099511627803
        let mut m = z("120892581953546433750499900385647336431813123664094681");
        let (found, visited) =
            trial_division_stage(&mut m, (1 << 40) - (1 << 16), (1 << 40) + (1 << 16));
        assert_eq!(found, pieces(&[("1099511627689", 1), ("1099511627791", 1)]));
        // The early stop fires at the prime after nextprime(2^40): m = P30 passed BPSW.
        assert_eq!(m, z("100000000000000000000000000319"));
        assert_eq!(visited, 2363 + 2);
    }

    #[test]
    fn trial_division_stage_small_limits() {
        // gp: factor(100) = [2,2; 5,2]
        let mut m = Integer::from(100);
        let (found, _) = trial_division_stage(&mut m, 2, u64::MAX);
        assert_eq!(found, pieces(&[("2", 2), ("5", 2)]));
        assert!(m.is_one());
        // A limit below 2 does nothing; 1 has nothing to divide.
        let mut m = Integer::from(100);
        assert_eq!(trial_division_stage(&mut m, 2, 1), (vec![], 0));
        assert_eq!(m, Integer::from(100));
        let mut m = Integer::one();
        assert_eq!(trial_division_stage(&mut m, 2, 1000), (vec![], 0));
        // 2 * P30 with no limit: the 6542 primes below 2^16, then the early stop.
        // gp: primepi(2^16) = 6542
        let mut m = z("200000000000000000000000000638");
        let (found, visited) = trial_division_stage(&mut m, 2, u64::MAX);
        assert_eq!(found, pieces(&[("2", 1)]));
        assert_eq!(m, z("100000000000000000000000000319"));
        assert_eq!(visited, 6543);
    }

    #[test]
    fn has_at_most_digits_edges() {
        assert!(!has_at_most_digits(&Integer::zero(), 5));
        assert!(!has_at_most_digits(&Integer::from(-7), 5));
        assert!(!has_at_most_digits(&Integer::one(), 0));
        assert!(has_at_most_digits(&Integer::from(9), 1));
        assert!(!has_at_most_digits(&Integer::from(10), 1));
        assert!(has_at_most_digits(&z("999999999999999999999999"), 24));
        assert!(!has_at_most_digits(&z("1000000000000000000000000"), 24));
        // gp: #Str(2^80) = 25
        let p80 = Integer::from(2).pow(80);
        assert!(!has_at_most_digits(&p80, 24));
        assert!(has_at_most_digits(&p80, 25));
        // Decided by the bit bound, without forming 10^u32::MAX.
        assert!(has_at_most_digits(&Integer::from(2).pow(4000), u32::MAX));
    }

    fn check_pieces(c: &Integer, got: &[(Integer, u32)]) {
        assert_eq!(product(got), *c, "{c}: {got:?}");
        for (b, _) in got {
            assert!(*b > Integer::one() && b < c, "{c}: {got:?}");
        }
    }

    #[test]
    fn split_composite_plans() {
        let plan = |rho_steps: u64, qs: QsPolicy, complete: bool| SplitPlan {
            rho_c: Integer::one(),
            rho_x0: Integer::one(),
            rho_steps,
            qs,
            complete,
        };
        // The complete fallback (Floyd rho, then Brent rounds with trial-division windows).
        // gp: factor(3127) = [53,1; 59,1]; factor(165731) = [53,2; 59,1];
        //     factor(1000000016000000063) = [1000000007,1; 1000000009,1]
        let fallback = plan(0, QsPolicy::Never, true);
        for s in ["3127", "165731", "1000000016000000063"] {
            let c = z(s);
            check_pieces(&c, &split_composite(&c, &fallback).expect("complete"));
        }
        // The sieve stage on the 40-digit row (rho with 8191 steps does not split it).
        // gp: factor(2700501075613600140300554153516920373243) =
        //     [31415926535897932429, 1; 85959619001776934567, 1]
        let c = z("2700501075613600140300554153516920373243");
        let got = split_composite(&c, &plan(8191, QsPolicy::Always, true)).expect("complete");
        assert_eq!(
            sorted(got),
            pieces(&[("31415926535897932429", 1), ("85959619001776934567", 1)])
        );
        // No method: None. gp: factor(20000000000008900000000000837) =
        //     [100000000000031,1; 200000000000027,1]
        let c = z("20000000000008900000000000837");
        assert_eq!(split_composite(&c, &plan(0, QsPolicy::Never, false)), None);
        // Perfect powers and the factor 2 come first, whatever the plan.
        for p in [plan(0, QsPolicy::Never, false), fallback.clone()] {
            assert_eq!(
                split_composite(&Integer::from(3).pow(40), &p),
                Some(pieces(&[("3", 40)]))
            );
            assert_eq!(
                split_composite(&Integer::from(3127 * 1024), &p),
                Some(pieces(&[("2", 10), ("3127", 1)]))
            );
        }
    }

    /// A divisor of the sieve need not be prime; composite ones go back to the queue.
    #[test]
    fn engine_requeues_composite_divisors() {
        let plan = SplitPlan {
            rho_c: Integer::one(),
            rho_x0: Integer::one(),
            rho_steps: 0,
            qs: QsPolicy::Always,
            complete: true,
        };
        // gp: factor(6000000113000000706000001463) = [1000000007,1; 2000000011,1; 3000000019,1]
        // gp: factor(100000014000001100000085400002989) = [100000007,2; 10000000000000061,1]
        for (n, want) in [
            (
                "6000000113000000706000001463",
                pieces(&[("1000000007", 1), ("2000000011", 1), ("3000000019", 1)]),
            ),
            (
                "100000014000001100000085400002989",
                pieces(&[("100000007", 2), ("10000000000000061", 1)]),
            ),
        ] {
            let got = limited_engine(&z(n), 0, &plan, true);
            assert_eq!(got.primes, want, "{n}");
            assert!(got.is_complete());
            assert_eq!(got.sign, 1);
        }
    }

    #[test]
    fn classify_classes() {
        // gp: isprime(nextprime(10^29)) = 1; psi_12 = 399165290221 * 798330580441
        let p30 = z("100000000000000000000000000319");
        assert_eq!(classify(&p30, true), Class::Prime);
        assert_eq!(classify(&p30, false), Class::ProbablePrime);
        assert_eq!(
            classify(&z("318665857834031151167461"), true),
            Class::Composite
        );
        assert_eq!(classify(&Integer::from(1000000007), false), Class::Prime);
        assert_eq!(classify(&Integer::from(1001), true), Class::Composite);
    }
}
