//! The core of the self-initialising quadratic sieve (SIQS): Knuth-Schroeppel multiplier,
//! factor base, polynomials with Gray-code switching, log sieve, relations with the single large
//! prime variation, and the square-root step.
//!
//! MAGMA Handbook ch. 18 §18.10.3 `MPQS` and §18.10.1 `MPQSLimit` (digest
//! `rustmath-lava/docs/Chapter_18_Ring_of_Integers.md`). References:
//! - [Pom85] C. Pomerance, *The quadratic sieve factoring algorithm*, EUROCRYPT '84, LNCS 209
//!   (1985) 169-182.
//! - [Sil87] R. D. Silverman, *The multiple polynomial quadratic sieve*, Math. Comp. 48 (1987)
//!   329-339 (multiple polynomials, threshold, the multiplier function, large primes).
//! - [AP95] W. R. Alford, C. Pomerance, *Implementing the self-initializing quadratic sieve on a
//!   distributed network*, in: Number Theoretic and Algebraic Methods in Computer Science
//!   (Moscow 1993), World Scientific 1995, 163-174.
//! - [Con97] S. P. Contini, *Factoring integers with the self-initializing quadratic sieve*, MSc
//!   thesis, University of Georgia, 1997 (a = q_1 ... q_s, B_l, Gray code, Bainv2 root update).
//! - [CP05] R. Crandall, C. Pomerance, *Prime Numbers: A Computational Perspective*, 2nd ed.,
//!   Springer 2005, §6.1.1 (basic QS, L(n)), §6.1.4 (large primes), §6.1.5 (multiple
//!   polynomials), §6.1.6 (self-initialization).
//! - [Knu97] D. E. Knuth, TAOCP vol. 2, 3rd ed., §4.5.4 (the Knuth-Schroeppel multiplier) and
//!   §3.4.2 Algorithm P (the shuffle that draws the a-factors).
//! - [Coh93] H. Cohen, GTM 138, Algorithm 1.5.1 (Tonelli-Shanks), §10.4 (QS and MPQS).
//!
//! The precondition of [`run`] is step 0 of the driver (`super`): N is odd, N > 47, N has no
//! prime factor <= 47 and is not a perfect power. Then gcd(k, N) = 1 for every multiplier k
//! (its prime factors are <= 47) and kN is never a square.
//!
//! Soundness does not depend on any parameter or on floating point: `Factor(d)` is a factor
//! base prime that divides N or a gcd checked against 1 and N; every relation is checked
//! exactly; the congruence X^2 = Y^2 (mod N) is asserted. Floating point only steers the
//! parameters, the threshold and which valid a-value is drawn.

use super::gf2::left_nullspace;
use super::{QsOutcome, QsParams};
use crate::fast_arith::pow_mod_u64;
use crate::primality::is_prime_u64;
use crate::sieve::{for_each_prime, small_primes};
use crate::Integer;
use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer as _;
use num_traits::{One, ToPrimitive, Zero};
use std::collections::{HashMap, HashSet};
use std::ops::ControlFlow;

/// At most this many first partial relations per factor base prime are stored (16 F in all).
const FIRST_PARTIALS_PER_PRIME: usize = 16;
/// Consecutive repeated a-values after which the supply counts as exhausted.
const MAX_REPEATED_A: u32 = 1000;
/// The a-supply guard: s > 64 is refused (2^(s-1) polynomials must fit a u64, and the Bainv2
/// table has s rows).
const MAX_A_FACTORS: usize = 64;
/// Primes up to this bound are not sieved with (small prime variation) and are not a-factors.
const SMALL_PRIME_CUTOFF: u32 = 30;
/// The factor base scan stops here; every factor base prime fits a u32.
const FACTOR_BASE_SCAN_END: u64 = 1 << 31;

// ------------------------------------------------------------------ word arithmetic

fn mul_mod_u64(a: u64, b: u64, m: u64) -> u64 {
    (u128::from(a) * u128::from(b) % u128::from(m)) as u64
}

/// The inverse of a modulo p, for gcd(a, p) = 1 (extended Euclid on words).
fn inv_mod_u64(a: u64, p: u64) -> u64 {
    let (mut r0, mut r1) = (i128::from(p), i128::from(a % p));
    let (mut t0, mut t1) = (0i128, 1i128);
    while r1 != 0 {
        let q = r0 / r1;
        (r0, r1) = (r1, r0 - q * r1);
        (t0, t1) = (t1, t0 - q * t1);
    }
    assert_eq!(r0, 1, "inv_mod_u64: {a} is not invertible modulo {p}");
    t0.rem_euclid(i128::from(p)) as u64
}

/// Square root of a modulo the odd prime p < 2^63: `Some(r)` with r^2 = a mod p, 0 <= r < p,
/// or `None` exactly when a is a non-residue (Euler's criterion, exact for a prime).
/// Tonelli-Shanks [Coh93, 1.5.1] on u64 words with u128 products.
///
/// # Panics
///
/// With "sqrt_mod_u64: p = {p} is not an odd prime below 2^63" if p is not one
/// ([`is_prime_u64`]).
fn sqrt_mod_u64(a: u64, p: u64) -> Option<u64> {
    assert!(
        p % 2 == 1 && p < 1 << 63 && is_prime_u64(p),
        "sqrt_mod_u64: p = {p} is not an odd prime below 2^63"
    );
    let a = a % p;
    if a == 0 {
        return Some(0);
    }
    if pow_mod_u64(a, (p - 1) / 2, p) != 1 {
        return None;
    }
    if p % 4 == 3 {
        return Some(pow_mod_u64(a, (p + 1) / 4, p));
    }
    let s = (p - 1).trailing_zeros();
    let q = (p - 1) >> s;
    // The least non-residue z: (p - 1)/2 of 1 .. p - 1 are non-residues, so the search ends.
    let mut z = 2u64;
    while pow_mod_u64(z, (p - 1) / 2, p) != p - 1 {
        z += 1;
    }
    let mut m = s;
    let mut c = pow_mod_u64(z, q, p);
    let mut t = pow_mod_u64(a, q, p);
    let mut r = pow_mod_u64(a, q.div_ceil(2), p);
    while t != 1 {
        // The least i with t^(2^i) = 1; i < m because t lies in the subgroup of order 2^(m-1).
        let mut i = 0;
        let mut tt = t;
        while tt != 1 {
            tt = mul_mod_u64(tt, tt, p);
            i += 1;
        }
        assert!(
            i < m,
            "sqrt_mod_u64: Tonelli-Shanks invariant failed for the prime {p}"
        );
        let b = pow_mod_u64(c, 1u64 << (m - i - 1), p);
        m = i;
        c = mul_mod_u64(b, b, p);
        t = mul_mod_u64(t, c, p);
        r = mul_mod_u64(r, b, p);
    }
    Some(r)
}

/// v mod p for a small modulus p.
fn mod_small(v: &BigUint, p: u32) -> u32 {
    let mut r: u64 = 0;
    for d in v.iter_u32_digits().rev() {
        r = ((r << 32) | u64::from(d)) % u64::from(p);
    }
    r as u32
}

/// v mod p in [0, p) for a signed v.
fn mod_small_signed(v: &BigInt, p: u32) -> u32 {
    let r = mod_small(v.magnitude(), p);
    if v.sign() == Sign::Minus && r != 0 {
        p - r
    } else {
        r
    }
}

/// splitmix64 (S. Vigna), the generator of the a-factor draw.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// log2 of v > 0, to about 52 bits of precision (only used for parameters and the threshold).
fn log2_big(v: &BigUint) -> f64 {
    let bits = v.bits();
    if bits <= 52 {
        return (v.to_u64().unwrap_or(1) as f64).log2();
    }
    let shift = bits - 52;
    ((v >> shift).to_u64().unwrap_or(1) as f64).log2() + shift as f64
}

// ------------------------------------------------------------------ step 1: multiplier

/// The Knuth-Schroeppel multiplier [Knu97 §4.5.4, Sil87]: the squarefree k in the list below
/// that maximises f(k) = -ln(k)/2 + c_2(kN) + sum over the first 300 odd primes p of ln(p)/p if
/// p | k and 2 ln(p)/(p - 1) if kN is a non-zero square mod p; c_2 = 2 ln 2 if kN = 1 mod 8,
/// ln 2 if 5 mod 8, ln(2)/2 otherwise (kN even included).
fn multiplier(n: &BigUint) -> u32 {
    const KS: [u32; 30] = [
        1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37,
        38, 39, 41, 42, 43, 47,
    ];
    let ln2 = std::f64::consts::LN_2;
    let odd_primes = &small_primes()[1..301];
    let mut best = (f64::NEG_INFINITY, 1u32);
    for &k in &KS {
        let kn = n * k;
        let mut f = -0.5 * f64::from(k).ln();
        if kn.is_even() {
            f += 0.5 * ln2;
        } else {
            match mod_small(&kn, 8) {
                1 => f += 2.0 * ln2,
                5 => f += ln2,
                _ => f += 0.5 * ln2,
            }
        }
        for &p in odd_primes {
            let r = mod_small(&kn, p);
            let lp = f64::from(p).ln();
            if r == 0 {
                f += lp / f64::from(p);
            } else if pow_mod_u64(u64::from(r), u64::from(p - 1) / 2, u64::from(p)) == 1 {
                f += 2.0 * lp / (f64::from(p) - 1.0);
            }
        }
        if f > best.0 {
            best = (f, k);
        }
    }
    best.1
}

// ------------------------------------------------------------------ step 2: factor base

/// The factor base: primes[0] = 2, then the odd primes p with (kN/p) != -1, in increasing order.
#[derive(Debug)]
struct FactorBase {
    kn: BigUint,
    primes: Vec<u32>,
    /// t_p with t_p^2 = kN mod p (0 for p | k, 1 for p = 2; both unused).
    sqrt_kn: Vec<u32>,
    logp: Vec<u8>,
    divides_k: Vec<bool>,
    /// The first index with p > 30: sieving and the a-factors start there.
    sieve_start: usize,
}

#[derive(Debug)]
enum BaseScan {
    Base(FactorBase),
    /// A prime p < N of the scan divides N.
    Divisor(u32),
    /// The scan met N itself: N is prime.
    IsPrime,
    /// The scan reached 2^31 with fewer than `size` primes.
    Exhausted,
}

/// Step 2: the factor base of `size` primes for kN, from the segmented sieve (memory O(1)).
/// Every prime up to the last one that divides N is found here.
fn factor_base(n: &BigUint, k: u32, size: usize) -> BaseScan {
    let kn: BigUint = n * k;
    let mut primes: Vec<u32> = vec![2];
    let mut sqrt_kn: Vec<u32> = vec![1];
    let mut found: Option<BaseScan> = None;
    if size > 1 {
        for_each_prime(3, FACTOR_BASE_SCAN_END, |p| {
            let p32 = p as u32;
            let r = mod_small(&kn, p32);
            if r == 0 {
                if !k.is_multiple_of(p32) {
                    // p | N.
                    found = Some(if *n == BigUint::from(p32) {
                        BaseScan::IsPrime
                    } else {
                        BaseScan::Divisor(p32)
                    });
                    return ControlFlow::Break(());
                }
                primes.push(p32);
                sqrt_kn.push(0);
            } else if pow_mod_u64(u64::from(r), (p - 1) / 2, p) == 1 {
                // Euler's criterion first, so only the residues pay for the primality check
                // inside `sqrt_mod_u64`.
                let t = sqrt_mod_u64(u64::from(r), p).expect("r is a residue modulo p");
                primes.push(p32);
                sqrt_kn.push(t as u32);
            }
            if primes.len() >= size {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });
    }
    if let Some(outcome) = found {
        return outcome;
    }
    if primes.len() < size {
        return BaseScan::Exhausted;
    }
    let logp = primes
        .iter()
        .map(|&p| f64::from(p).log2().round() as u8)
        .collect();
    let divides_k = primes.iter().map(|&p| k.is_multiple_of(p)).collect();
    let sieve_start = primes
        .iter()
        .position(|&p| p > SMALL_PRIME_CUTOFF)
        .unwrap_or(primes.len());
    BaseScan::Base(FactorBase {
        kn,
        primes,
        sqrt_kn,
        logp,
        divides_k,
        sieve_start,
    })
}

// ------------------------------------------------------------------ step 4: a-values

/// State of the a-factor draw.
#[derive(Debug)]
struct ASelector {
    /// The indices of E whose log2 p is near log2(target)/s (W, a subset of E).
    window: Vec<usize>,
    /// E: the factor base indices with p > 30 and p not dividing k.
    eligible: Vec<usize>,
    /// Number of a-factors.
    s: usize,
    log2_target: f64,
    rng: u64,
    /// The sorted index sets of the a-values drawn so far.
    used: HashSet<Vec<usize>>,
}

/// Sets up the a-factor draw for sieve half width m, or `None` when the a-supply guard of step
/// 4 fails: s > 64, |E| < s or |W| < max(s - 1, 1).
fn a_selector(fb: &FactorBase, m: usize, seed: u64) -> Option<ASelector> {
    let nfb = fb.primes.len();
    let log2_target = (log2_big(&fb.kn) + 1.0) / 2.0 - (m as f64).log2();
    let log2_pmax = f64::from(fb.primes[nfb - 1]).log2();
    let qbits_pref = 11.0f64.min(log2_pmax - 1.0).max(3.0);
    let s = ((log2_target / qbits_pref).round() as usize).max(1);
    let qbits = log2_target / s as f64;
    let log2p = |j: usize| f64::from(fb.primes[j]).log2();
    let start = fb.sieve_start;
    let mut lo = (start..nfb)
        .find(|&j| log2p(j) >= qbits - 1.0)
        .unwrap_or(start.min(nfb - 1));
    let mut hi = (0..nfb)
        .rev()
        .find(|&j| log2p(j) <= qbits + 1.0)
        .unwrap_or(nfb - 1);
    while hi < lo + 4 * s + 8 && (lo > start || hi + 1 < nfb) {
        if lo > start {
            lo -= 1;
        }
        if hi + 1 < nfb {
            hi += 1;
        }
    }
    let window: Vec<usize> = (lo..=hi)
        .filter(|&j| j >= start && !fb.divides_k[j])
        .collect();
    let eligible: Vec<usize> = (start..nfb).filter(|&j| !fb.divides_k[j]).collect();
    if s > MAX_A_FACTORS || eligible.len() < s || window.len() < (s - 1).max(1) {
        return None;
    }
    Some(ASelector {
        window,
        eligible,
        s,
        log2_target,
        rng: seed ^ 0x5151_2023,
        used: HashSet::new(),
    })
}

/// Draws a new a-value: s - 1 indices from a partial Fisher-Yates shuffle of W [Knu97 §3.4.2],
/// then the index of E, not drawn, whose log2 p is closest to log2(target) minus the drawn
/// logs (for s = 1 a uniform entry of W). Returns the sorted index set, or `None` after 1000
/// repeated sets in a row (the supply is exhausted).
fn choose_a(fb: &FactorBase, sel: &mut ASelector) -> Option<Vec<usize>> {
    let mut repeats = 0u32;
    loop {
        let mut idx: Vec<usize> = Vec::with_capacity(sel.s);
        if sel.s == 1 {
            let r = splitmix64(&mut sel.rng) % sel.window.len() as u64;
            idx.push(sel.window[r as usize]);
        } else {
            let mut w = sel.window.clone();
            let mut acc = 0.0;
            for t in 0..sel.s - 1 {
                let r = t + (splitmix64(&mut sel.rng) % (w.len() - t) as u64) as usize;
                w.swap(t, r);
                idx.push(w[t]);
                acc += f64::from(fb.primes[w[t]]).log2();
            }
            let want = sel.log2_target - acc;
            let mut best: Option<(f64, usize)> = None;
            for &j in &sel.eligible {
                if idx.contains(&j) {
                    continue;
                }
                let d = (f64::from(fb.primes[j]).log2() - want).abs();
                if best.is_none_or(|(bd, _)| d < bd) {
                    best = Some((d, j));
                }
            }
            let (_, j) = best.expect("choose_a: |E| >= s was checked by the a-supply guard");
            idx.push(j);
        }
        idx.sort_unstable();
        if sel.used.insert(idx.clone()) {
            return Some(idx);
        }
        repeats += 1;
        if repeats > MAX_REPEATED_A {
            return None;
        }
    }
}

// ------------------------------------------------------------------ step 4: polynomials

/// Q(x) = (a x + b)^2 - kN = a g(x) with g(x) = a x^2 + 2 b x + c [Con97 §4].
#[derive(Debug)]
struct Polynomial {
    a: BigInt,
    /// 2 B_l for l = 0 .. s - 1.
    two_bl: Vec<BigInt>,
    b: BigInt,
    c: BigInt,
    /// The position of b in the Gray-code order, 0 .. 2^(s-1) - 1.
    index: u64,
}

/// Sieve roots shifted to array indices, the Bainv2 table and the a-factor mask; reused for
/// every a-value.
#[derive(Debug)]
struct Roots {
    r1: Vec<u32>,
    r2: Vec<u32>,
    bainv2: Vec<Vec<u32>>,
    is_a_factor: Vec<bool>,
}

impl Roots {
    fn new(nfb: usize, s: usize) -> Roots {
        Roots {
            r1: vec![0; nfb],
            r2: vec![0; nfb],
            bainv2: vec![vec![0; nfb]; s],
            is_a_factor: vec![false; nfb],
        }
    }
}

/// True iff index j is sieved: j >= 1 and p_j does not divide a k.
fn sieved(fb: &FactorBase, roots: &Roots, j: usize) -> bool {
    !(roots.is_a_factor[j] || fb.divides_k[j])
}

/// The first polynomial of the a-value with the given factor indices; fills `roots`.
/// B_l = (a/q_l) gamma_l with gamma_l = t_{q_l} (a/q_l)^-1 mod q_l, replaced by q_l - gamma_l
/// if larger than q_l/2; b = B_1 + ... + B_s, so b^2 = kN mod a.
fn init_polynomial(
    fb: &FactorBase,
    m: usize,
    a_factors: &[usize],
    roots: &mut Roots,
) -> Polynomial {
    let nfb = fb.primes.len();
    roots.is_a_factor.iter_mut().for_each(|x| *x = false);
    let mut a = BigUint::one();
    for &j in a_factors {
        a *= fb.primes[j];
        roots.is_a_factor[j] = true;
    }
    let mut bl: Vec<BigUint> = Vec::with_capacity(a_factors.len());
    for &j in a_factors {
        let q = u64::from(fb.primes[j]);
        let aq = &a / q;
        let aq_mod = u64::from(mod_small(&aq, q as u32));
        let mut gamma = mul_mod_u64(u64::from(fb.sqrt_kn[j]), inv_mod_u64(aq_mod, q), q);
        if gamma > q / 2 {
            gamma = q - gamma;
        }
        bl.push(aq * gamma);
    }
    let mut b = BigInt::zero();
    for x in &bl {
        b += BigInt::from(x.clone());
    }
    let kn = BigInt::from(fb.kn.clone());
    let a_int = BigInt::from(a.clone());
    let num = &b * &b - &kn;
    debug_assert!((&num % &a_int).is_zero());
    let c = num / &a_int;
    for j in 1..nfb {
        if !sieved(fb, roots, j) {
            continue;
        }
        let p = u64::from(fb.primes[j]);
        let ainv = inv_mod_u64(u64::from(mod_small(&a, p as u32)), p);
        for (l, x) in bl.iter().enumerate() {
            roots.bainv2[l][j] = mul_mod_u64(2 * u64::from(mod_small(x, p as u32)), ainv, p) as u32;
        }
        let bm = u64::from(mod_small_signed(&b, p as u32));
        let t = u64::from(fb.sqrt_kn[j]);
        let mm = m as u64 % p;
        let s1 = mul_mod_u64(ainv, (t + p - bm) % p, p);
        let s2 = mul_mod_u64(ainv, (2 * p - t - bm) % p, p);
        roots.r1[j] = ((s1 + mm) % p) as u32;
        roots.r2[j] = ((s2 + mm) % p) as u32;
    }
    Polynomial {
        a: a_int,
        two_bl: bl.into_iter().map(|x| BigInt::from(x << 1u32)).collect(),
        b,
        c,
        index: 0,
    }
}

/// One Gray-code step [Con97 §4.3]: for i = index + 1, l = v_2(i) and e = +1 if (i >> l) = 3
/// mod 4, else -1; b <- b + 2 e B_l and r <- r - e Bainv2[l] mod p for both roots. The 2^(s-1)
/// steps visit all sign patterns of B_1 .. B_{s-1} with B_s fixed.
fn next_polynomial(fb: &FactorBase, kn: &BigInt, poly: &mut Polynomial, roots: &mut Roots) {
    poly.index += 1;
    let i = poly.index;
    let l = i.trailing_zeros() as usize;
    let plus = (i >> l) % 4 == 3;
    if plus {
        poly.b += &poly.two_bl[l];
    } else {
        poly.b -= &poly.two_bl[l];
    }
    for j in 1..fb.primes.len() {
        if !sieved(fb, roots, j) {
            continue;
        }
        let p = fb.primes[j];
        let d = roots.bainv2[l][j];
        let (r1, r2) = (&mut roots.r1[j], &mut roots.r2[j]);
        if plus {
            *r1 = if *r1 >= d { *r1 - d } else { *r1 + p - d };
            *r2 = if *r2 >= d { *r2 - d } else { *r2 + p - d };
        } else {
            *r1 += d;
            if *r1 >= p {
                *r1 -= p;
            }
            *r2 += d;
            if *r2 >= p {
                *r2 -= p;
            }
        }
    }
    poly.c = (&poly.b * &poly.b - kn) / &poly.a;
}

// ------------------------------------------------------------------ step 5: sieve

/// Step 5 [Pom85, CP05 §6.1.1]: fills `buf` (length 2M) with `init`, adds round(log2 p) at the
/// two roots of every sieved prime p > 30, and returns the indices whose byte is >= 128.
fn sieve(fb: &FactorBase, roots: &Roots, init: u8, buf: &mut [u8]) -> Vec<usize> {
    buf.fill(init);
    let len = buf.len();
    for j in fb.sieve_start..fb.primes.len() {
        if !sieved(fb, roots, j) {
            continue;
        }
        let p = fb.primes[j] as usize;
        let lg = fb.logp[j];
        let mut i = roots.r1[j] as usize;
        while i < len {
            buf[i] = buf[i].wrapping_add(lg);
            i += p;
        }
        if roots.r2[j] != roots.r1[j] {
            let mut i = roots.r2[j] as usize;
            while i < len {
                buf[i] = buf[i].wrapping_add(lg);
                i += p;
            }
        }
    }
    let mut hits = Vec::new();
    for (w, chunk) in buf.chunks_exact(8).enumerate() {
        let word = u64::from_le_bytes(chunk.try_into().expect("chunks_exact(8)"));
        if word & 0x8080_8080_8080_8080 == 0 {
            continue;
        }
        for (o, &v) in chunk.iter().enumerate() {
            if v >= 128 {
                hits.push(8 * w + o);
            }
        }
    }
    hits
}

/// The initial byte of the sieve array, 128 - T with T = round(log2 M + (log2 kN - 1)/2 -
/// log2 B_L - slack) clamped to [1, 128]: a location is a candidate iff its byte reaches 128.
/// The byte cannot wrap below 128 for n up to about 85 digits; above, T is clamped and only more
/// candidates appear (each is checked exactly).
fn sieve_init(fb: &FactorBase, m: usize, lp_bound: u64, slack_bits: u8) -> u8 {
    let t = (m as f64).log2() + (log2_big(&fb.kn) - 1.0) / 2.0
        - (lp_bound as f64).log2()
        - f64::from(slack_bits);
    128 - t.clamp(1.0, 128.0).round() as u8
}

/// The large prime bound B_L = min(lp_mult * p_max, p_max^2 - 1) (the product saturates).
fn large_prime_bound(fb: &FactorBase, lp_mult: u64) -> u64 {
    let pmax = u64::from(fb.primes[fb.primes.len() - 1]);
    lp_mult.saturating_mul(pmax).min(pmax * pmax - 1)
}

// ------------------------------------------------------------------ steps 6 and 7: relations

/// y^2 - kN = (-1)^negative * prod p_j^e_j * large_prime, with y = a x + b.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Relation {
    y: BigInt,
    negative: bool,
    /// (factor base index, exponent), ascending, exponents >= 1.
    factors: Vec<(u32, u32)>,
    /// 1 for a full relation, else a prime above p_max and below the large prime bound.
    large_prime: u64,
}

/// Step 6: the relation at array index i, or `None` when g(x) = 0 or its cofactor after the
/// factor base is not 1 and not below `lp_bound`. A prime p not dividing a k divides g(x)
/// exactly when i is one of its two roots, so the root test finds every such prime; a prime of
/// a or k is tested directly. A prime <= p_max outside the factor base cannot divide g, so the
/// cofactor has only prime factors > p_max, and a cofactor below lp_bound < p_max^2 is prime.
///
/// # Panics
///
/// If the exact check y^2 - kN = sign * prod p^e * L fails (an internal invariant).
fn check_candidate(
    fb: &FactorBase,
    poly: &Polynomial,
    roots: &Roots,
    m: usize,
    i: usize,
    lp_bound: u64,
) -> Option<Relation> {
    let x = BigInt::from(i as i64 - m as i64);
    let g: BigInt = (&poly.a * &x + &poly.b * 2u32) * &x + &poly.c;
    if g.is_zero() {
        return None;
    }
    let negative = g.sign() == Sign::Minus;
    let mut v: BigUint = g.magnitude().clone();
    let mut factors: Vec<(u32, u32)> = Vec::new();
    let tz = v.trailing_zeros().unwrap_or(0);
    if tz > 0 {
        v >>= tz;
        factors.push((0, tz as u32));
    }
    for j in 1..fb.primes.len() {
        let p = fb.primes[j];
        let hit = if sieved(fb, roots, j) {
            let im = (i % p as usize) as u32;
            im == roots.r1[j] || im == roots.r2[j]
        } else {
            mod_small(&v, p) == 0
        };
        // Q = a g: every a-factor gets one more.
        let mut e = u32::from(roots.is_a_factor[j]);
        if hit {
            while mod_small(&v, p) == 0 {
                v /= p;
                e += 1;
            }
        }
        if e > 0 {
            factors.push((j as u32, e));
        }
    }
    let large_prime = if v.is_one() {
        1
    } else {
        match v.to_u64() {
            Some(w) if w < lp_bound => w,
            _ => return None,
        }
    };
    let y = &poly.a * &x + &poly.b;
    let mut prod = BigUint::from(large_prime);
    for &(j, e) in &factors {
        prod *= BigUint::from(fb.primes[j as usize]).pow(e);
    }
    let q = if negative {
        -BigInt::from(prod)
    } else {
        BigInt::from(prod)
    };
    assert_eq!(
        &y * &y - BigInt::from(fb.kn.clone()),
        q,
        "check_candidate: the relation check y^2 - kN = Q failed"
    );
    Some(Relation {
        y,
        negative,
        factors,
        large_prime,
    })
}

/// What [`RelationStore::add`] did with a relation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Added {
    /// A full relation, one matrix row.
    Full,
    /// The first partial relation with its large prime; no row yet.
    PartialFirst,
    /// A partial relation whose large prime has a stored first partial: a row {first, this}.
    PartialPaired,
    /// |y| was seen before: dropped, no row and no pairing.
    Duplicate,
    /// A partial relation with a new large prime while `partial_of` has `first_partial_cap`
    /// entries: dropped, and its |y| is not recorded.
    Dropped,
}

/// Step 7: the relations, the matrix rows (lists of relation indices) and the large prime map.
#[derive(Debug)]
struct RelationStore {
    rels: Vec<Relation>,
    rows: Vec<Vec<usize>>,
    partial_of: HashMap<u64, usize>,
    seen_y: HashSet<BigUint>,
    first_partial_cap: usize,
}

impl RelationStore {
    fn new(first_partial_cap: usize) -> RelationStore {
        RelationStore {
            rels: Vec::new(),
            rows: Vec::new(),
            partial_of: HashMap::new(),
            seen_y: HashSet::new(),
            first_partial_cap,
        }
    }

    /// Stores a relation. The order is: duplicate test, cap test, insert |y|, store.
    ///
    /// De-duplication is a correctness step: different polynomials can give the same y (and y,
    /// -y give the same Q). A repeated relation, or a pair row {r, r'} with y' = +-y and the
    /// same large prime, is a dependency by itself whose congruence X = +-Y never splits N, so
    /// it would use up part of the nullspace the matrix guarantees.
    fn add(&mut self, rel: Relation) -> Added {
        if self.seen_y.contains(rel.y.magnitude()) {
            return Added::Duplicate;
        }
        let lp = rel.large_prime;
        if lp != 1
            && !self.partial_of.contains_key(&lp)
            && self.partial_of.len() >= self.first_partial_cap
        {
            return Added::Dropped;
        }
        self.seen_y.insert(rel.y.magnitude().clone());
        let id = self.rels.len();
        self.rels.push(rel);
        if lp == 1 {
            self.rows.push(vec![id]);
            return Added::Full;
        }
        match self.partial_of.get(&lp) {
            Some(&first) => {
                self.rows.push(vec![first, id]);
                Added::PartialPaired
            }
            None => {
                self.partial_of.insert(lp, id);
                Added::PartialFirst
            }
        }
    }
}

// ------------------------------------------------------------------ steps 8 and 9

/// The parity rows of the matrix: column 0 is the sign, column 1 + j the prime p_j.
fn parity_rows(fb: &FactorBase, store: &RelationStore) -> (Vec<Vec<u64>>, usize) {
    let ncols = fb.primes.len() + 1;
    let wc = ncols.div_ceil(64);
    let rows = store
        .rows
        .iter()
        .map(|row| {
            let mut bits = vec![0u64; wc];
            for &id in row {
                let rel = &store.rels[id];
                if rel.negative {
                    bits[0] ^= 1;
                }
                for &(j, e) in &rel.factors {
                    if e % 2 == 1 {
                        let c = j as usize + 1;
                        bits[c / 64] ^= 1u64 << (c % 64);
                    }
                }
            }
            bits
        })
        .collect();
    (rows, ncols)
}

/// Step 9 [CP05 §6.1.1]: for each nullspace vector (a list of matrix rows), X = prod y_i mod N
/// and Y = prod p_j^(E_j/2) prod L^(E_L/2) mod N over the multiset of its relations, then
/// gcd(X - Y, N). Returns the first proper divisor, or `NoFactor` with the number of vectors
/// tried.
///
/// # Panics
///
/// If an exponent sum is odd or X^2 != Y^2 (mod N) (internal invariants: X^2 = prod Q_i =
/// (-1)^s prod p^E prod L^E_L = Y^2 since kN = 0 mod N and s is even).
fn square_roots(
    n: &BigUint,
    fb: &FactorBase,
    store: &RelationStore,
    deps: &[Vec<usize>],
) -> QsOutcome {
    let n_int = BigInt::from(n.clone());
    let nfb = fb.primes.len();
    let mut tried = 0usize;
    for dep in deps {
        tried += 1;
        let mut x = BigInt::one();
        let mut e_sum = vec![0u64; nfb];
        let mut negs = 0u64;
        let mut lps: HashMap<u64, u64> = HashMap::new();
        for &row in dep {
            for &id in &store.rows[row] {
                let rel = &store.rels[id];
                x = (x * &rel.y) % &n_int;
                if rel.negative {
                    negs += 1;
                }
                for &(j, e) in &rel.factors {
                    e_sum[j as usize] += u64::from(e);
                }
                if rel.large_prime > 1 {
                    *lps.entry(rel.large_prime).or_insert(0) += 1;
                }
            }
        }
        assert!(
            negs.is_multiple_of(2)
                && e_sum.iter().all(|&e| e.is_multiple_of(2))
                && lps.values().all(|&e| e.is_multiple_of(2)),
            "square_roots: a nullspace vector has an odd exponent"
        );
        let mut y = BigUint::one();
        for (j, &e) in e_sum.iter().enumerate() {
            if e > 0 {
                y = (y * BigUint::from(fb.primes[j]).modpow(&BigUint::from(e / 2), n)) % n;
            }
        }
        for (&l, &e) in &lps {
            y = (y * BigUint::from(l).modpow(&BigUint::from(e / 2), n)) % n;
        }
        let x = x
            .mod_floor(&n_int)
            .to_biguint()
            .expect("a residue mod N is non-negative");
        assert!(
            (&x * &x) % n == (&y * &y) % n,
            "square_roots: the congruence X^2 = Y^2 (mod N) failed"
        );
        let diff = if x >= y { &x - &y } else { &y - &x };
        let g = diff.gcd(n);
        if !g.is_one() && &g != n {
            return QsOutcome::Factor(Integer::new(BigInt::from(g)));
        }
    }
    QsOutcome::NoFactor {
        dependencies: tried,
    }
}

// ------------------------------------------------------------------ the run

/// Counters of one run: s, the accepted a-values, the polynomials sieved, the dropped partial
/// relations.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct RunStats {
    pub(super) a_factors: usize,
    pub(super) a_values: u64,
    pub(super) polynomials: u64,
    pub(super) dropped_partials: u64,
}

/// Steps 1 to 9 on N with the (validated) parameters. Precondition: N odd, N > 47, no prime
/// factor <= 47, not a perfect power (step 0 of the driver).
pub(super) fn run(n: &BigUint, params: &QsParams) -> (QsOutcome, RunStats) {
    debug_assert!(n.is_odd() && *n > BigUint::from(47u32));
    let mut stats = RunStats::default();
    let needed = params.factor_base_size + 1 + params.extra_relations;
    let k = multiplier(n);
    let fb = match factor_base(n, k, params.factor_base_size) {
        BaseScan::Base(fb) => fb,
        BaseScan::Divisor(p) => return (QsOutcome::Factor(Integer::from(p)), stats),
        BaseScan::IsPrime => return (QsOutcome::NoFactor { dependencies: 0 }, stats),
        BaseScan::Exhausted => {
            return (
                QsOutcome::OutOfPolynomials {
                    relations: 0,
                    needed,
                },
                stats,
            )
        }
    };
    let nfb = fb.primes.len();
    let m = params.sieve_half_width.div_ceil(64) * 64;
    let lp_bound = large_prime_bound(&fb, params.large_prime_multiplier);
    let Some(mut sel) = a_selector(&fb, m, params.seed) else {
        return (
            QsOutcome::OutOfPolynomials {
                relations: 0,
                needed,
            },
            stats,
        );
    };
    stats.a_factors = sel.s;
    let init = sieve_init(&fb, m, lp_bound, params.threshold_slack_bits);
    let kn = BigInt::from(fb.kn.clone());
    let mut store = RelationStore::new(FIRST_PARTIALS_PER_PRIME * nfb);
    let mut roots = Roots::new(nfb, sel.s);
    let mut buf = vec![0u8; 2 * m];
    let polys_per_a = 1u64 << (sel.s - 1);
    let out_of_polynomials = |store: &RelationStore| QsOutcome::OutOfPolynomials {
        relations: store.rows.len(),
        needed,
    };
    'collect: loop {
        if stats.polynomials >= params.max_polynomials {
            return (out_of_polynomials(&store), stats);
        }
        let Some(a_factors) = choose_a(&fb, &mut sel) else {
            return (out_of_polynomials(&store), stats);
        };
        stats.a_values += 1;
        let mut poly = init_polynomial(&fb, m, &a_factors, &mut roots);
        for index in 0..polys_per_a {
            // The budget is checked before every polynomial, the first one of each a included:
            // one a has 2^(s-1) polynomials, and s can be 31 with valid arguments.
            if stats.polynomials >= params.max_polynomials {
                return (out_of_polynomials(&store), stats);
            }
            if index > 0 {
                next_polynomial(&fb, &kn, &mut poly, &mut roots);
            }
            stats.polynomials += 1;
            for i in sieve(&fb, &roots, init, &mut buf) {
                let Some(rel) = check_candidate(&fb, &poly, &roots, m, i, lp_bound) else {
                    continue;
                };
                if store.add(rel) == Added::Dropped {
                    stats.dropped_partials += 1;
                }
                // Stop at R rows: the first R rows are the matrix.
                if store.rows.len() >= needed {
                    break 'collect;
                }
            }
        }
    }
    let (rows, ncols) = parity_rows(&fb, &store);
    let deps = left_nullspace(&rows, ncols);
    (square_roots(n, &fb, &store, &deps), stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn big(s: &str) -> BigUint {
        s.parse().expect("decimal")
    }

    /// gp: d = 30 row of the design table: p = 314159265359057, q = 859596190017779.
    const N30: &str = "270050107561429816771648674403";
    /// gp: d = 40 row: p = 31415926535897932429, q = 85959619001776934567.
    const N40: &str = "2700501075613600140300554153516920373243";
    /// gp: nextprime(10^74) * nextprime(3*10^75), 150 digits, 497 bits.
    const N150: &str = "300000000000000000000000000000000000000000000000000000000000000000000000627700000000000000000000000000000000000000000000000000000000000000000000013869";

    /// Port of HEAD's `test_tonelli_shanks_least_non_residue_above_100`: the least non-residue
    /// of p = 23616331489 is 107.
    #[test]
    fn sqrt_mod_u64_least_non_residue_above_100() {
        // gp: p = 23616331489; isprime(p) = 1; p % 8 = 1; valuation(p-1, 2) = 5;
        // z = 2; while(kronecker(z, p) != -1, z++); z = 107
        let p: u64 = 23616331489;
        // gp: r = lift(sqrt(Mod(2, p))); min(r, p - r) = 4184367042
        // gp: r = lift(sqrt(Mod(1000, p))); min(r, p - r) = 2465952743
        for (a, root) in [(2u64, 4184367042u64), (1000, 2465952743)] {
            let r = sqrt_mod_u64(a, p).expect("a is a residue mod p");
            assert!(r == root || r == p - root, "sqrt({a}) mod {p}: got {r}");
            assert_eq!(mul_mod_u64(r, r, p), a);
        }
        // gp: kronecker(-2, p) = 1, so -2 = p - 2 has a root.
        let r = sqrt_mod_u64(p - 2, p).expect("-2 is a residue mod p");
        assert!(r < p);
        assert_eq!(mul_mod_u64(r, r, p), p - 2);
        // gp: kronecker(107, p) = -1
        assert_eq!(sqrt_mod_u64(107, p), None);
    }

    /// Ports of HEAD's `test_tonelli_shanks` and `test_tonelli_shanks_small_primes`.
    #[test]
    fn sqrt_mod_u64_small_primes_exhaustive() {
        // sqrt(2) mod 7 is 3 or 4 (3^2 = 9 = 2 mod 7).
        let r = sqrt_mod_u64(2, 7).unwrap();
        assert!(r == 3 || r == 4, "got {r}");
        // gp: lift(sqrt(Mod(3, 13))) = 4 (the roots are 4 and 9)
        let r = sqrt_mod_u64(3, 13).unwrap();
        assert!(r == 4 || r == 9, "got {r}");
        // gp: kronecker(3, 41) = -1; issquare(Mod(5, 41)) = 1
        assert_eq!(sqrt_mod_u64(3, 41), None);
        let r = sqrt_mod_u64(5, 41).unwrap();
        assert_eq!(r * r % 41, 5);
        // Brute force: for each odd prime p < 200 and each a, a root exists exactly for the
        // squares mod p.
        for p in (3u64..200).filter(|&p| is_prime_u64(p)) {
            for a in 0..p {
                let is_square = (0..p).any(|x| x * x % p == a);
                match sqrt_mod_u64(a, p) {
                    Some(r) => assert_eq!(r * r % p, a, "p = {p}, a = {a}"),
                    None => assert!(!is_square, "p = {p}: {a} is a square but got None"),
                }
            }
        }
    }

    /// A composite modulus is a contract violation, not a "no square root".
    #[test]
    #[should_panic(expected = "sqrt_mod_u64: p = 15 is not an odd prime below 2^63")]
    fn sqrt_mod_u64_rejects_composite() {
        // gp: isprime(15) = 0
        let _ = sqrt_mod_u64(4, 15);
    }

    #[test]
    fn factor_base_matches_kronecker() {
        // gp: N = 1000000016000000063; select(p -> kronecker(3*N, p) >= 0, primes(100))
        //     (the first 100 primes) = the 52 odd primes below, 3 to 523; with 2 the base has
        //     53 primes.
        let n = big("1000000016000000063");
        let BaseScan::Base(fb) = factor_base(&n, 3, 53) else {
            panic!("no prime <= 523 divides N");
        };
        let want: Vec<u32> = vec![
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 53, 61, 67, 71, 79, 83, 103, 109, 127, 139,
            149, 151, 181, 191, 197, 199, 211, 223, 227, 233, 251, 269, 281, 293, 313, 337, 347,
            349, 353, 359, 383, 389, 397, 401, 409, 419, 431, 433, 457, 509, 523,
        ];
        assert_eq!(fb.primes, want);
        assert!(fb.divides_k[1] && fb.divides_k.iter().filter(|&&d| d).count() == 1);
        // gp: lift(sqrt(Mod(3*N, p))) for p = 5 .. 29, up to sign: [2, 2, 1, 1, 3, 7, 7, 13]
        let t: Vec<u32> = (2..10)
            .map(|j| fb.sqrt_kn[j].min(fb.primes[j] - fb.sqrt_kn[j]))
            .collect();
        assert_eq!(t, vec![2, 2, 1, 1, 3, 7, 7, 13]);
        for j in 2..fb.primes.len() {
            let p = fb.primes[j];
            let t = u64::from(fb.sqrt_kn[j]);
            assert_eq!(
                t * t % u64::from(p),
                u64::from(mod_small(&fb.kn, p)),
                "p = {p}"
            );
        }
        assert_eq!(fb.primes[fb.sieve_start], 31);
        assert_eq!(fb.logp[fb.sieve_start], 5);
    }

    #[test]
    fn factor_base_reports_a_small_prime_factor() {
        // gp: 101 * nextprime(10^29) = 10100000000000000000000000032219
        let n = big("10100000000000000000000000032219");
        assert!(matches!(factor_base(&n, 1, 100), BaseScan::Divisor(101)));
        // N = 101 itself: the scan meets N, which is prime (gp: isprime(101) = 1).
        assert!(matches!(
            factor_base(&big("101"), 1, 100),
            BaseScan::IsPrime
        ));
        let (outcome, _) = run(&big("101"), &QsParams::for_bits(7));
        assert_eq!(outcome, QsOutcome::NoFactor { dependencies: 0 });
    }

    #[test]
    fn a_selector_guard() {
        // N150 with a factor base of 30 primes needs s = 34 a-factors, but only 22 primes of
        // the base are above 30.
        let n = big(N150);
        let k = multiplier(&n);
        let BaseScan::Base(fb) = factor_base(&n, k, 30) else {
            panic!("no small factor");
        };
        let m = QsParams::for_bits(n.bits()).sieve_half_width;
        assert!(a_selector(&fb, m, 1).is_none());
        // The 30-digit row with `for_bits`: 200 draws give sorted, distinct sets of s indices,
        // each a subset of E.
        let n = big(N30);
        let params = QsParams::for_bits(n.bits());
        let k = multiplier(&n);
        let BaseScan::Base(fb) = factor_base(&n, k, params.factor_base_size) else {
            panic!("no small factor");
        };
        let mut sel = a_selector(&fb, params.sieve_half_width, 1).expect("guard passes");
        let s = sel.s;
        let mut seen = HashSet::new();
        for _ in 0..200 {
            let idx = choose_a(&fb, &mut sel).expect("supply not exhausted");
            assert_eq!(idx.len(), s);
            assert!(idx.windows(2).all(|w| w[0] < w[1]), "{idx:?}");
            assert!(idx.iter().all(|j| sel.eligible.contains(j)), "{idx:?}");
            assert!(seen.insert(idx));
        }
    }

    /// The 30-digit row's first a-value: every polynomial of the Gray code has correct roots.
    fn first_a_of(n: &BigUint) -> (FactorBase, usize, Vec<usize>) {
        let params = QsParams::for_bits(n.bits());
        let k = multiplier(n);
        let BaseScan::Base(fb) = factor_base(n, k, params.factor_base_size) else {
            panic!("no small factor");
        };
        let m = params.sieve_half_width;
        let mut sel = a_selector(&fb, m, 1).expect("guard passes");
        let idx = choose_a(&fb, &mut sel).expect("a first a-value");
        (fb, m, idx)
    }

    #[test]
    fn polynomial_roots_survive_gray_code() {
        let n = big(N30);
        let (fb, m, idx) = first_a_of(&n);
        let s = idx.len();
        let mut roots = Roots::new(fb.primes.len(), s);
        let mut poly = init_polynomial(&fb, m, &idx, &mut roots);
        let kn = BigInt::from(fb.kn.clone());
        let mut bs = HashSet::new();
        for index in 0..1u64 << (s - 1) {
            if index > 0 {
                next_polynomial(&fb, &kn, &mut poly, &mut roots);
            }
            assert!(((&poly.b * &poly.b - &kn) % &poly.a).is_zero());
            assert_eq!(&poly.a * &poly.c, &poly.b * &poly.b - &kn);
            assert!(bs.insert(poly.b.magnitude().clone()), "b repeated");
            for j in 1..fb.primes.len() {
                if !sieved(&fb, &roots, j) {
                    continue;
                }
                let p = BigInt::from(fb.primes[j]);
                for r in [roots.r1[j], roots.r2[j]] {
                    let y = &poly.a * BigInt::from(i64::from(r) - m as i64) + &poly.b;
                    assert!(((&y * &y - &kn) % &p).is_zero(), "p = {p}, index {index}");
                }
            }
        }
        assert_eq!(bs.len(), 1 << (s - 1));
    }

    #[test]
    fn relations_are_exact() {
        let n = big(N30);
        let params = QsParams::for_bits(n.bits());
        let (fb, m, idx) = first_a_of(&n);
        let nfb = fb.primes.len();
        let pmax = u64::from(fb.primes[nfb - 1]);
        let lp_bound = large_prime_bound(&fb, params.large_prime_multiplier);
        let init = sieve_init(&fb, m, lp_bound, params.threshold_slack_bits);
        let mut roots = Roots::new(nfb, idx.len());
        let poly = init_polynomial(&fb, m, &idx, &mut roots);
        let mut buf = vec![0u8; 2 * m];
        let mut count = 0;
        for i in sieve(&fb, &roots, init, &mut buf) {
            let Some(rel) = check_candidate(&fb, &poly, &roots, m, i, lp_bound) else {
                continue;
            };
            count += 1;
            // Independent recomputation of y^2 - kN.
            let mut prod = Integer::from(rel.large_prime);
            for &(j, e) in &rel.factors {
                prod = prod * Integer::from(fb.primes[j as usize]).pow(e);
            }
            if rel.negative {
                prod = -prod;
            }
            let y = Integer::new(rel.y.clone());
            assert_eq!(&y * &y - Integer::new(BigInt::from(fb.kn.clone())), prod);
            assert!(
                rel.large_prime == 1
                    || (rel.large_prime > pmax
                        && rel.large_prime < lp_bound
                        && is_prime_u64(rel.large_prime))
            );
        }
        assert!(count > 0, "the first polynomial gives relations");
    }

    fn rel(y: i64, large_prime: u64) -> Relation {
        Relation {
            y: BigInt::from(y),
            negative: false,
            factors: vec![],
            large_prime,
        }
    }

    #[test]
    fn relation_store_drops_repeated_y() {
        let mut store = RelationStore::new(1000);
        assert_eq!(store.add(rel(5, 1)), Added::Full);
        assert_eq!(store.add(rel(-5, 1)), Added::Duplicate);
        assert_eq!(store.rows, vec![vec![0]]);
        assert_eq!(store.add(rel(7, 101)), Added::PartialFirst);
        assert_eq!(store.add(rel(-7, 101)), Added::Duplicate);
        assert_eq!(store.rows.len(), 1);
        assert_eq!(store.add(rel(9, 101)), Added::PartialPaired);
        assert_eq!(store.rows.last(), Some(&vec![1, 2]));
        assert_eq!(store.rels.len(), 3);
    }

    #[test]
    fn relation_store_caps_first_partials() {
        let mut store = RelationStore::new(2);
        assert_eq!(store.add(rel(7, 101)), Added::PartialFirst);
        assert_eq!(store.add(rel(9, 103)), Added::PartialFirst);
        assert_eq!(store.add(rel(11, 107)), Added::Dropped);
        assert_eq!(store.rels.len(), 2);
        assert!(store.rows.is_empty());
        // Pairing still works at the cap.
        assert_eq!(store.add(rel(13, 101)), Added::PartialPaired);
        assert_eq!(store.rows, vec![vec![0, 2]]);
        assert_eq!(store.add(rel(15, 1)), Added::Full);
        // A dropped |y| is not recorded, so the same relation is dropped again, not a duplicate.
        assert_eq!(store.add(rel(11, 107)), Added::Dropped);
        assert_eq!(store.rels.len(), 4);
    }

    /// The first-partial cap never binds on the measured rows; a port that drops partials here
    /// has the cap test in the wrong place.
    #[test]
    fn run_on_grid_rows_drops_no_partials() {
        // gp: the d = 20, 30 and 35 rows of the design table (factor(p*q) == [p,1; q,1]).
        for (n, p, q) in [
            ("27005010885249061781", "3141592661", "8595961921"),
            (N30, "314159265359057", "859596190017779"),
            (
                "27005010756136059108789661003408639",
                "31415926535897999",
                "859596190017769361",
            ),
        ] {
            let n = big(n);
            let (outcome, stats) = run(&n, &QsParams::for_bits(n.bits()));
            let QsOutcome::Factor(d) = outcome else {
                panic!("{n}: {outcome:?}");
            };
            let d = d.as_bigint().magnitude().clone();
            assert!(d == big(p) || d == big(q), "{n}: {d}");
            assert_eq!(stats.dropped_partials, 0, "{n}");
            assert!(stats.polynomials > 0 && stats.a_values > 0);
        }
    }

    #[test]
    fn run_stops_at_the_polynomial_budget() {
        // The 40-digit row (131 bits, gp: #binary(n) = 131): F = 802, needed = 802 + 1 + 32 =
        // 835, s = 5, so each a-value has 16 polynomials.
        let n = big(N40);
        for b in [1u64, 3, 16, 17] {
            let params = QsParams {
                max_polynomials: b,
                ..QsParams::for_bits(131)
            };
            let (outcome, stats) = run(&n, &params);
            match outcome {
                QsOutcome::OutOfPolynomials { relations, needed } => {
                    assert_eq!(needed, 835);
                    assert!(relations < needed);
                }
                other => panic!("budget {b}: {other:?}"),
            }
            assert_eq!(stats.a_factors, 5);
            assert_eq!(stats.polynomials, b);
            assert_eq!(stats.a_values, b.div_ceil(16));
            assert_eq!(stats.dropped_partials, 0);
        }
    }
}
