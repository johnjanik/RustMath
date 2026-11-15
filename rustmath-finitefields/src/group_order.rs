//! Multiplicative orders from the factored group exponent (crate-private).
//!
//! MAGMA Handbook ch. 19 §19.7 `Order(a)` (the multiplicative order of a unit of Z/mZ, 0 for a
//! non-unit) and `IsPrimitive(n)`; ch. 21 §21.4.4 `IsPrimitive(a)` and §21.4.7 `Order(a)`,
//! `FactoredOrder(a)` ("factorisation of #F - 1, then order-from-factorisation"); ch. 48
//! (Galois rings reduce to their residue field). Digests
//! `rustmath-lava/docs/Chapter_19_Integer_Residue_Class_Rings.md`,
//! `Chapter_21_Finite_Fields.md`, `Chapter_48_Galois_Rings.md`. The factorizations come from
//! MAGMA `Factorization(n : TrialDivisionLimit, PollardRhoLimit, MPQSLimit, Proof)` (ch. 18
//! §18.10.1), here `rustmath_integers::factorization_with_limits`.
//!
//! Algorithms and references:
//! - \[Coh93\] H. Cohen, *A Course in Computational Algebraic Number Theory*, GTM 138 (1993),
//!   §1.4.1, Alg. 1.4.3 "Order of an Element": from any multiple `h` of the order, factored,
//!   strip each prime as far as `x^e = 1` allows. Alg. 1.4.4 (the primitive-root test) is the
//!   same loop.
//! - \[Car10\] R. D. Carmichael, *Note on a new number theory function*, Bull. AMS 16 (1910)
//!   232-238: the exponent of (Z/mZ)^* is `λ(m) = lcm λ(p^e)`, `λ(p^e) = p^(e-1)(p - 1)` for an
//!   odd prime, `λ(2) = 1`, `λ(4) = 2`, `λ(2^e) = 2^(e-2)` for `e >= 3`.
//! - \[BLSTW02\] J. Brillhart, D. H. Lehmer, J. L. Selfridge, B. Tuckerman, S. S. Wagstaff Jr.,
//!   *Factorizations of b^n ± 1*, Contemporary Mathematics 22, AMS, 3rd ed. 2002: the algebraic
//!   split `p^n - 1 = ∏_{d | n} Φ_d(p)`, so each cyclotomic value is factored on its own.
//!
//! Honesty: a factorization is used only when it is complete and every base is a proven prime.
//! Alg. 1.4.3 with a composite "prime" `r` would treat `r` as one prime and could return a
//! multiple of the true order. Probable primes or unsplit composites give `Err`, which the
//! `Ring` hooks report as `MultiplicativeOrder::Undecided`.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use rustmath_integers::{factorization_with_limits, FactorizationLimits, Integer};

/// How hard the group exponent is factored and proven.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Effort {
    /// The inherent methods: `FactorizationLimits::default()` (MAGMA's defaults: trial division
    /// to 10^4, 8191 rho steps, the sieve without a digit limit, `proof: true`). Complete unless
    /// an ECPP proof is unresolved. No time bound.
    Complete,
    /// The `Ring` hooks: trial division to `CAPPED_TRIAL_DIVISION`, `CAPPED_RHO_STEPS` Brent-rho
    /// steps per composite, the sieve only on composites of at most `CAPPED_SIEVE_DIGITS`
    /// digits, `proof: false`; then each probable prime of at most `CAPPED_PROOF_BITS` bits is
    /// proven (`rustmath_integers::prime::try_is_prime`, an ECPP proof cached for the process),
    /// and a larger one is not attempted. So one factorization costs at most the trial
    /// division, 8191 rho steps per composite, a sieve on at most 30 digits, and one ECPP
    /// proof of at most 256 bits per prime factor.
    Capped,
}

pub(crate) const CAPPED_TRIAL_DIVISION: u64 = 10_000;
pub(crate) const CAPPED_RHO_STEPS: u64 = 8191;
pub(crate) const CAPPED_SIEVE_DIGITS: u32 = 30;
pub(crate) const CAPPED_PROOF_BITS: u64 = 256;

/// Prime factorization, ascending, every base a proven prime. The empty vector is 1.
pub(crate) type Factored = Vec<(Integer, u32)>;

/// Complete, proven factorizations, keyed by the integer that was factored. Shared by every
/// type and both efforts; cleared at `CACHE_CAP` entries (recomputing is always correct). No
/// lock is held while factoring, and no code panics while holding it (a poisoned lock is
/// recovered).
static CACHE: LazyLock<Mutex<HashMap<Integer, Arc<Factored>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

const CACHE_CAP: usize = 4096;

/// `n >= 1` factored with every base a proven prime, or why not. The `Err` text names the
/// unsplit composites, the probable primes left unproven (Complete: an unresolved ECPP proof;
/// Capped: above 256 bits, or unresolved), or a probable prime that ECPP showed composite
/// (Capped only; that would be a BPSW pseudoprime, and none is known). Proven results go into
/// the process-wide cache, so a capped call can succeed after an exact call factored the same
/// `n`.
pub(crate) fn factor_proven(n: &Integer, effort: Effort) -> Result<Arc<Factored>, String> {
    if let Some(f) = CACHE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(n)
        .cloned()
    {
        return Ok(f);
    }
    let f = Arc::new(factor_proven_uncached(n, effort)?);
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if cache.len() >= CACHE_CAP {
        cache.clear();
    }
    cache.insert(n.clone(), Arc::clone(&f));
    Ok(f)
}

/// The computation of [`factor_proven`] without the cache (never looked up, never stored).
/// The unit tests that assert a capped failure call this one, so a complete factorization
/// cached by another test of the same binary cannot change their outcome.
fn factor_proven_uncached(n: &Integer, effort: Effort) -> Result<Factored, String> {
    if *n < Integer::one() {
        return Err(format!("group_order: {n} is not a positive integer"));
    }
    if n.is_one() {
        return Ok(Vec::new());
    }
    let limits = match effort {
        Effort::Complete => FactorizationLimits::default(),
        Effort::Capped => FactorizationLimits {
            trial_division_limit: CAPPED_TRIAL_DIVISION,
            pollard_rho_limit: CAPPED_RHO_STEPS,
            mpqs_limit: Some(CAPPED_SIEVE_DIGITS),
            proof: false,
        },
    };
    let lf = factorization_with_limits(n, &limits);
    if !lf.composites.is_empty() {
        let list: Vec<String> = lf.composites.iter().map(|(c, _)| c.to_string()).collect();
        return Err(format!(
            "the composite part {} of {n} is not split within the cost cap",
            list.join(", ")
        ));
    }
    let mut primes = lf.primes;
    match effort {
        Effort::Complete => {
            if let Some((q, _)) = lf.probable_primes.first() {
                return Err(format!(
                    "the probable prime {q} dividing {n} has no primality proof (ECPP unresolved)"
                ));
            }
        }
        Effort::Capped => {
            for (q, e) in lf.probable_primes {
                let bits = q.bits();
                if bits > CAPPED_PROOF_BITS {
                    return Err(format!(
                        "the probable prime {q} ({bits} bits) dividing {n} is above the \
                         {CAPPED_PROOF_BITS}-bit proof cap"
                    ));
                }
                match rustmath_integers::prime::try_is_prime(&q) {
                    Ok(true) => primes.push((q, e)),
                    Ok(false) => {
                        return Err(format!(
                            "{q} passed Baillie-PSW but is not prime (a BPSW pseudoprime)"
                        ))
                    }
                    Err(why) => {
                        return Err(format!(
                            "the probable prime {q} dividing {n} has no primality proof ({why})"
                        ))
                    }
                }
            }
        }
    }
    primes.sort();
    Ok(primes)
}

/// Merges `f` into `acc` (both ascending): exponents of a common prime are combined with `op`.
fn merge(acc: &mut Factored, f: &[(Integer, u32)], op: fn(u32, u32) -> u32) {
    for (p, e) in f {
        match acc.binary_search_by(|(q, _)| q.cmp(p)) {
            Ok(i) => acc[i].1 = op(acc[i].1, *e),
            Err(i) => acc.insert(i, (p.clone(), *e)),
        }
    }
}

/// λ(m) factored (m >= 2), combined from `factor_proven(m)` and `factor_proven(p - 1)` for
/// each prime p | m (\[Car10\]; the lcm is the maximum of the exponents).
pub(crate) fn carmichael_lambda(m: &Integer, effort: Effort) -> Result<Factored, String> {
    let two = Integer::from(2);
    let mut lambda: Factored = Vec::new();
    for (p, e) in factor_proven(m, effort)?.iter() {
        let mut part: Factored = Vec::new();
        if *p == two {
            match *e {
                1 => {}
                2 => part.push((two.clone(), 1)),
                _ => part.push((two.clone(), e - 2)),
            }
        } else {
            if *e > 1 {
                part.push((p.clone(), e - 1));
            }
            merge(
                &mut part,
                &factor_proven(&(p.clone() - Integer::one()), effort)?,
                |a, b| a + b,
            );
        }
        merge(&mut lambda, &part, u32::max);
    }
    Ok(lambda)
}

/// The Möbius function of `n >= 1` (trial division; `n` is a field degree).
fn mobius(mut n: usize) -> i32 {
    let mut r = 1;
    let mut d = 2;
    while d * d <= n {
        if n.is_multiple_of(d) {
            n /= d;
            if n.is_multiple_of(d) {
                return 0;
            }
            r = -r;
        }
        d += 1;
    }
    if n > 1 {
        r = -r;
    }
    r
}

/// Φ_d(p) = ∏_{e | d} (p^e - 1)^{μ(d/e)}, by exact integer division.
fn cyclotomic_value(p: &Integer, d: usize) -> Integer {
    let (mut num, mut den) = (Integer::one(), Integer::one());
    for e in (1..=d).filter(|e| d.is_multiple_of(*e)) {
        let t = p.pow(e as u32) - Integer::one();
        match mobius(d / e) {
            1 => num = num * t,
            -1 => den = den * t,
            _ => {}
        }
    }
    let (q, r) = (num.clone() / den.clone(), num % den);
    debug_assert!(r.is_zero(), "Φ_d(p) is an integer");
    q
}

/// p^n - 1 factored through the cyclotomic values Φ_d(p), d | n, each through
/// [`factor_proven`] (\[BLSTW02\]); the pieces can share small primes, so exponents add.
pub(crate) fn field_group_order(p: &Integer, n: usize, effort: Effort) -> Result<Factored, String> {
    let mut acc: Factored = Vec::new();
    for d in (1..=n).filter(|d| n.is_multiple_of(*d)) {
        merge(
            &mut acc,
            &factor_proven(&cyclotomic_value(p, d), effort)?,
            |a, b| a + b,
        );
    }
    Ok(acc)
}

/// The product ∏ p^e.
pub(crate) fn product(f: &[(Integer, u32)]) -> Integer {
    f.iter().fold(Integer::one(), |acc, (p, e)| acc * p.pow(*e))
}

/// \[Coh93\] Alg. 1.4.3: the order of `x`, factored, from a factored multiple `h` of it.
///
/// Invariant: `x^e = 1` after each prime, and after the step for `p_i` the `p_i`-adic
/// valuation of `e` equals that of `ord(x)` (the least `k` with `x^(e' p_i^k) = 1`, `e'` the
/// `p_i`-free part, is `v_{p_i}(ord x)`, because `e'` is a multiple of the `p_i`-free part of
/// `ord x`); later steps do not change it. Correct for every multiple `h` of the order.
///
/// # Panics
/// "group_order: x^h != 1" when `h` is not a multiple of the order; the callers' exponents
/// rule this out.
pub(crate) fn order_from_exponent<T>(
    x: &T,
    h: &[(Integer, u32)],
    pow: impl Fn(&T, &Integer) -> T,
    is_one: impl Fn(&T) -> bool,
) -> Factored {
    let mut e = product(h);
    assert!(is_one(&pow(x, &e)), "group_order: x^h != 1");
    let mut out: Factored = Vec::new();
    for (p, v) in h {
        e = e / p.pow(*v);
        let mut g = pow(x, &e);
        let mut k = 0u32;
        while !is_one(&g) {
            g = pow(&g, p);
            e = e * p.clone();
            k += 1;
        }
        if k > 0 {
            out.push((p.clone(), k));
        }
    }
    out
}

/// The order of a unit `x` of a finite group, factored; every order of the crate goes through
/// it. Shortcuts without factoring: `is_one(x)` gives `vec![]` (order 1); else `minus_one`
/// (the caller says `x = -1` and `-1 != 1`) gives `vec![(2, 1)]`. Otherwise `exponent()`
/// supplies a factored multiple of `ord(x)`, and Alg. 1.4.3 runs. `Err` only from `exponent()`.
pub(crate) fn unit_order<T>(
    x: &T,
    minus_one: bool,
    exponent: impl FnOnce() -> Result<Factored, String>,
    pow: impl Fn(&T, &Integer) -> T,
    is_one: impl Fn(&T) -> bool,
) -> Result<Factored, String> {
    if is_one(x) {
        return Ok(Vec::new());
    }
    if minus_one {
        return Ok(vec![(Integer::from(2), 1)]);
    }
    let h = exponent()?;
    Ok(order_from_exponent(x, &h, pow, is_one))
}

/// The order of the unit `a` mod `m` (`m >= 2`, `0 <= a < m`, `gcd(a, m) = 1`), factored.
/// [`unit_order`] with `minus_one = (a = m - 1 and m > 2)`; the multiple of the order is
/// `m - 1` when `a^(m-1) = 1 (mod m)` (one exponentiation: always for a prime `m`, and for a
/// composite `m` exactly when `a` is a Fermat liar; then `m` is neither factored nor proven
/// prime), else `λ(m)` ([`carmichael_lambda`]). When the factorization of `m - 1` fails
/// and `m` is not a Baillie-PSW probable prime (so `m` is composite and `λ(m) < m - 1`),
/// `λ(m)` is tried before giving up: its factorization can be cheap when `m - 1` is hard.
pub(crate) fn unit_order_mod(a: &Integer, m: &Integer, effort: Effort) -> Result<Factored, String> {
    let one = Integer::one();
    let m_minus_1 = m.clone() - one.clone();
    let minus_one = *a == m_minus_1 && *m > Integer::from(2);
    let pow = |x: &Integer, e: &Integer| x.mod_pow(e, m).expect("m >= 2");
    unit_order(
        a,
        minus_one,
        || {
            if !pow(a, &m_minus_1).is_one() {
                return carmichael_lambda(m, effort);
            }
            match factor_proven(&m_minus_1, effort) {
                Ok(f) => Ok(f.as_ref().clone()),
                Err(why) if !rustmath_integers::prime::is_probable_prime(m) => {
                    carmichael_lambda(m, effort).map_err(|why2| format!("{why}; {why2}"))
                }
                Err(why) => Err(why),
            }
        },
        pow,
        |x| x.is_one(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int(v: i64) -> Integer {
        Integer::from(v)
    }

    fn big(s: &str) -> Integer {
        Integer::from_decimal_str(s).unwrap()
    }

    fn fac(v: &[(i64, u32)]) -> Factored {
        v.iter().map(|&(p, e)| (int(p), e)).collect()
    }

    /// Brute force: the least k >= 1 with a^k = 1 mod m.
    fn brute(a: i64, m: i64) -> i64 {
        let (mut x, mut k) = (a % m, 1);
        while x != 1 {
            x = x * a % m;
            k += 1;
        }
        k
    }

    /// Alg. 1.4.3 from any multiple: 2 mod 7 has order 3, from h = 6, 12 and 36.
    #[test]
    fn order_from_exponent_any_multiple() {
        let pow = |x: &Integer, e: &Integer| x.mod_pow(e, &int(7)).unwrap();
        for h in [
            fac(&[(2, 1), (3, 1)]),
            fac(&[(2, 2), (3, 1)]),
            fac(&[(2, 2), (3, 2)]),
        ] {
            assert_eq!(
                order_from_exponent(&int(2), &h, pow, |x| x.is_one()),
                fac(&[(3, 1)])
            );
        }
    }

    #[test]
    #[should_panic(expected = "group_order: x^h != 1")]
    fn order_from_exponent_checks_the_multiple() {
        let pow = |x: &Integer, e: &Integer| x.mod_pow(e, &int(7)).unwrap();
        order_from_exponent(&int(3), &fac(&[(2, 1)]), pow, |x| x.is_one());
    }

    /// 1 and -1 need no factoring: the exponent closure is never called.
    #[test]
    fn unit_order_shortcuts() {
        let pow = |x: &Integer, e: &Integer| x.mod_pow(e, &int(11)).unwrap();
        let never = || -> Result<Factored, String> { panic!("factored") };
        assert_eq!(
            unit_order(&int(1), false, never, pow, |x| x.is_one()),
            Ok(vec![])
        );
        assert_eq!(
            unit_order(&int(10), true, never, pow, |x| x.is_one()),
            Ok(fac(&[(2, 1)]))
        );
        let failing = || -> Result<Factored, String> { Err("no".to_string()) };
        assert_eq!(
            unit_order(&int(2), false, failing, pow, |x| x.is_one()),
            Err("no".to_string())
        );
    }

    /// gp: znorder(Mod(2,341)) = 10 (Fermat liar: Mod(2,341)^340 = 1), znorder(Mod(3,341)) =
    /// 30 (Mod(3,341)^340 = 56, the λ path), znorder(Mod(2,561)) = 40, znorder(Mod(5,561)) = 80.
    #[test]
    fn unit_order_mod_fermat_and_lambda_paths() {
        assert_eq!(int(2).mod_pow(&int(340), &int(341)).unwrap(), int(1));
        assert_eq!(int(3).mod_pow(&int(340), &int(341)).unwrap(), int(56));
        assert_eq!(
            product(&unit_order_mod(&int(2), &int(341), Effort::Complete).unwrap()),
            int(10)
        );
        assert_eq!(
            product(&unit_order_mod(&int(3), &int(341), Effort::Complete).unwrap()),
            int(30)
        );
        assert_eq!(
            product(&unit_order_mod(&int(2), &int(561), Effort::Capped).unwrap()),
            int(40)
        );
        assert_eq!(
            product(&unit_order_mod(&int(5), &int(561), Effort::Capped).unwrap()),
            int(80)
        );
        // brute force on every unit mod m < 120, both efforts
        for m in 2..120i64 {
            for a in 1..m {
                if int(a).gcd(&int(m)) != int(1) {
                    continue;
                }
                for effort in [Effort::Complete, Effort::Capped] {
                    let f = unit_order_mod(&int(a), &int(m), effort).unwrap();
                    assert_eq!(product(&f), int(brute(a, m)), "{a} mod {m}");
                }
            }
        }
    }

    /// gp: [lcm(10,30), 2^8, lcm([2,10,16])] = [30, 256, 80] for λ(341 = 11·31),
    /// λ(2^10), λ(561 = 3·11·17), and factor(80) = 2^4 * 5; λ(2) = 1, λ(4) = 2, λ(8) = 2.
    #[test]
    fn carmichael_lambda_matches_pari() {
        let lam = |m: i64| product(&carmichael_lambda(&int(m), Effort::Complete).unwrap());
        assert_eq!(lam(341), int(30));
        assert_eq!(lam(1024), int(256));
        assert_eq!(lam(561), int(80));
        assert_eq!(lam(2), int(1));
        assert_eq!(lam(4), int(2));
        assert_eq!(lam(8), int(2));
        assert_eq!(
            carmichael_lambda(&int(561), Effort::Complete).unwrap(),
            fac(&[(2, 4), (5, 1)])
        );
    }

    /// The cyclotomic split multiplies back to p^n - 1, ascending, with proven prime bases.
    /// gp: factor(5^3-1) = 2^2 * 31.
    #[test]
    fn field_group_order_is_p_to_the_n_minus_one() {
        for (p, n) in [(3i64, 40usize), (2, 64), (5, 30), (101, 12), (2, 1), (7, 1)] {
            let f = field_group_order(&int(p), n, Effort::Complete).unwrap();
            assert_eq!(
                product(&f),
                int(p).pow(n as u32) - Integer::one(),
                "{p}^{n}"
            );
            assert!(f.windows(2).all(|w| w[0].0 < w[1].0), "ascending");
        }
        assert_eq!(
            field_group_order(&int(5), 3, Effort::Capped).unwrap(),
            fac(&[(2, 2), (31, 1)])
        );
    }

    /// P43 - 1 = 2 · 43 · C41 with C41 = 100000000000000000039 · 300000000000000000053 =
    /// 30000000000000000017000000000000000002067 (gp: P43 =
    /// 2580000000000000001462000000000000000177763, isprime(P43) = 1, factor(P43-1)). C41 has 41 digits (above the 30-digit sieve cap) and 21-digit factors
    /// (8191 rho steps fail), so the capped effort fails; the complete one succeeds. The
    /// capped assertion uses the uncached function, so test order does not matter.
    #[test]
    fn factor_proven_capped_and_complete_on_p43_minus_one() {
        let n = big("2580000000000000001462000000000000000177762");
        let capped = factor_proven_uncached(&n, Effort::Capped).unwrap_err();
        assert!(
            capped.contains("30000000000000000017000000000000000002067"),
            "the error names the 41-digit composite: {capped}"
        );
        let complete = factor_proven_uncached(&n, Effort::Complete).unwrap();
        assert_eq!(
            complete,
            vec![
                (int(2), 1),
                (int(43), 1),
                (big("100000000000000000039"), 1),
                (big("300000000000000000053"), 1)
            ]
        );
        assert_eq!(product(&complete), n);
    }

    /// m = 7 s with s = 952841793122185456008103211101 prime, and m - 1 = 2 · 3 · P · Q with
    /// P = 1045908801259457, Q = 1062854387785943 (PQ has 31 digits: above the sieve cap, and
    /// its 16-digit factors are out of reach of 8191 rho steps). a =
    /// 6207317463790804616919148922629 has order 6 mod 7 and mod s, so it is a Fermat liar
    /// (6 | m - 1). The capped Fermat path fails; λ(m) = lcm(6, s - 1) is cheap (s - 1 =
    /// 2^2 3^2 5^2 · 48936577 · 112428451 · 192428091677) and decides.
    /// gp: factor(m-1), factor(s-1), znorder(Mod(a,m)) = 6, Mod(a,m)^(m-1) == 1.
    #[test]
    fn capped_fermat_path_falls_back_to_lambda() {
        let m = big("6669892551855298192056722477707");
        let a = big("6207317463790804616919148922629");
        assert!(a
            .mod_pow(&(m.clone() - Integer::one()), &m)
            .unwrap()
            .is_one());
        assert!(factor_proven_uncached(&(m.clone() - Integer::one()), Effort::Capped).is_err());
        assert_eq!(
            unit_order_mod(&a, &m, Effort::Capped).unwrap(),
            fac(&[(2, 1), (3, 1)])
        );
        assert_eq!(
            unit_order_mod(&a, &m, Effort::Complete).unwrap(),
            fac(&[(2, 1), (3, 1)])
        );
    }

    #[test]
    fn factor_proven_small_values() {
        assert_eq!(
            factor_proven(&int(1), Effort::Capped).unwrap().as_ref(),
            &Vec::new()
        );
        assert_eq!(
            factor_proven(&int(360), Effort::Capped).unwrap().as_ref(),
            &fac(&[(2, 3), (3, 2), (5, 1)])
        );
        assert!(factor_proven_uncached(&int(0), Effort::Complete).is_err());
    }
}
