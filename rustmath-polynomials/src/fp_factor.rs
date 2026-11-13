//! Factorization over `F_p[x]` (distinct-degree + Cantor–Zassenhaus).
//!
//! Ported from the `FpPoly` engine of the p-adelic calculator. Polynomials are
//! little-endian `Vec<i64>` arrays with every coefficient reduced to `[0, p)`.
//! The prime `p` is passed as a parameter, not stored. The caller uses one
//! prime across operations.
//!
//! This is the mod-`p` companion of [`crate::zp_hensel`]: factor `f mod p` here,
//! then Hensel-lift the factors to `Z/p^N Z` there.
//!
//! Specification: MAGMA Handbook ch. 23 *Univariate Polynomial Rings* §23.8.1
//! (`Factorization` over F_q, `DistinctDegreeFactorization`,
//! `EqualDegreeFactorization`, `IsIrreducible`; digest
//! `rustmath-lava/docs/Chapter_23_Univariate_Polynomial_Rings.md`) and ch. 21
//! *Finite Fields* §21.3.5 (`Roots`).
//!
//! Algorithms:
//! - \[CZ81\] D. G. Cantor, H. Zassenhaus, *A new algorithm for factoring
//!   polynomials over finite fields*, Math. Comp. 36 (1981) 587-592: random
//!   equal-degree splitting, the trace map in characteristic 2.
//! - \[vzGG\] J. von zur Gathen, J. Gerhard, *Modern Computer Algebra*, §14.2
//!   (distinct-degree factorization) and §14.3: Algorithm 14.8 (equal-degree
//!   splitting), Theorem 14.9 (a uniform try fails with probability at most
//!   2^(1-r) <= 1/2 on a product of r >= 2 distinct irreducibles of degree d),
//!   Algorithm 14.10 (recursive equal-degree factorization).
//! - \[Rab80\] M. O. Rabin, *Probabilistic algorithms in finite fields*, SIAM J.
//!   Comput. 9 (1980) 273-280: `x^(p^d) - x` is the product of the monic
//!   irreducibles of degree dividing `d`; the equal-degree contract certificate
//!   ([`equal_degree_factor`]) is Rabin's criterion generalised to products.
//! - SplitMix64: G. L. Steele, D. Lea, C. H. Flood, *Fast splittable
//!   pseudorandom number generators*, OOPSLA 2014 (the sampler's generator).
//!
//! Overflow contract: every product formed in `i64` is at most
//! `(p-1) + (p-1)^2` (`mul`: accumulator plus a product of two residues;
//! `div_mod`, `make_monic`, `extended_gcd`, `mod_inv`: at most `(p-1)^2`), so
//! the arithmetic is exact for every prime `p <= MAX_P` ([`MAX_P`] =
//! 3_037_000_499 = `floor(sqrt(i64::MAX))`; `(p-1)^2 + (p-1) > i64::MAX` from
//! 3_037_000_501 on).
//!
//! Checked entries. The factorization entries [`factor`],
//! [`factor_with_multiplicity`], [`squarefree_factor`],
//! [`distinct_degree_factor`] and [`equal_degree_factor`] check first that `p`
//! is a prime `<= MAX_P` ([`check_modulus`], a proof for every `p`), then read
//! the coefficients of `f` modulo `p`. A bad `p` or `f ≡ 0` is a panic that
//! names the entry and the reason. Their outputs are sorted by
//! [`factor_order`], the order of PARI's `factormod`.
//!
//! Unchecked helpers. The arithmetic helpers (`add`, `sub`, `mul`, `div_mod`,
//! `gcd`, `extended_gcd`, `pow_mod`, `make_monic`, `mod_inv`, `derivative_of`,
//! `trim`, `degree`, `is_zero`) run in the inner loops of `zp_hensel`, the
//! Galois sieve, Mac Lane and the number field ideals, so they check nothing.
//! They need a prime `p <= MAX_P` and coefficients in `[0, p)`; otherwise the
//! result is meaningless or an arithmetic panic.

use rustmath_integers::Integer;
use std::cmp::Ordering;

/// Strip trailing zeros, keeping at least one coefficient. The zero polynomial
/// is `[0]`. Canonical form for every returned value.
pub fn trim(p: &[i64]) -> Vec<i64> {
    let mut c = p.to_vec();
    while c.len() > 1 && *c.last().unwrap() == 0 {
        c.pop();
    }
    if c.is_empty() {
        c.push(0);
    }
    c
}

/// True iff `p` is the zero polynomial.
pub fn is_zero(p: &[i64]) -> bool {
    p.iter().all(|&c| c == 0)
}

/// Polynomial degree. Returns `-1` for the zero polynomial.
pub fn degree(p: &[i64]) -> i64 {
    let t = trim(p);
    if is_zero(&t) {
        -1
    } else {
        (t.len() - 1) as i64
    }
}

#[inline]
fn modp(a: i64, p: i64) -> i64 {
    ((a % p) + p) % p
}

/// `a + b mod p`.
pub fn add(a: &[i64], b: &[i64], p: i64) -> Vec<i64> {
    let n = a.len().max(b.len());
    let mut out = vec![0i64; n];
    for i in 0..a.len() {
        out[i] = modp(out[i] + a[i], p);
    }
    for i in 0..b.len() {
        out[i] = modp(out[i] + b[i], p);
    }
    trim(&out)
}

/// `a - b mod p`.
pub fn sub(a: &[i64], b: &[i64], p: i64) -> Vec<i64> {
    let n = a.len().max(b.len());
    let mut out = vec![0i64; n];
    for i in 0..a.len() {
        out[i] = modp(out[i] + a[i], p);
    }
    for i in 0..b.len() {
        out[i] = modp(out[i] - b[i], p);
    }
    trim(&out)
}

/// `a * b mod p`, schoolbook.
pub fn mul(a: &[i64], b: &[i64], p: i64) -> Vec<i64> {
    if is_zero(a) || is_zero(b) {
        return vec![0];
    }
    let mut out = vec![0i64; a.len() + b.len() - 1];
    for i in 0..a.len() {
        if a[i] == 0 {
            continue;
        }
        for j in 0..b.len() {
            if b[j] == 0 {
                continue;
            }
            out[i + j] = modp(out[i + j] + a[i] * b[j], p);
        }
    }
    trim(&out)
}

/// Long division: returns `(q, r)` with `a = q * b + r`, `deg(r) < deg(b)`.
/// Panics if `b` is the zero polynomial or its leading coefficient is not
/// invertible mod `p` (the latter cannot happen for prime `p`).
pub fn div_mod(a: &[i64], b: &[i64], p: i64) -> (Vec<i64>, Vec<i64>) {
    let a_trim = trim(a);
    let b_trim = trim(b);
    assert!(!is_zero(&b_trim), "fp_factor::div_mod: divisor must be non-zero");
    if a_trim.len() < b_trim.len() {
        return (vec![0], a_trim);
    }
    let mut rem = a_trim.clone();
    let b_deg = b_trim.len() - 1;
    let q_len = a_trim.len() - b_trim.len() + 1;
    let mut quot = vec![0i64; q_len];
    let b_lead_inv =
        mod_inv(b_trim[b_deg], p).expect("fp_factor::div_mod: leading coefficient not invertible");
    while rem.len() >= 1 + b_deg && !is_zero(&rem) {
        let lead = rem[rem.len() - 1];
        let coeff = modp(lead * b_lead_inv, p);
        let shift = rem.len() - 1 - b_deg;
        quot[shift] = coeff;
        for i in 0..b_trim.len() {
            rem[shift + i] = modp(rem[shift + i] - coeff * b_trim[i], p);
        }
        rem = trim(&rem);
    }
    (trim(&quot), rem)
}

/// `gcd(a, b)` over `F_p[x]`, returned monic.
pub fn gcd(a: &[i64], b: &[i64], p: i64) -> Vec<i64> {
    let mut x = trim(a);
    let mut y = trim(b);
    while !is_zero(&y) {
        let (_, r) = div_mod(&x, &y, p);
        x = y;
        y = r;
    }
    make_monic(&x, p)
}

/// Extended GCD over `F_p[x]`: returns `(g, s, t)` with `s*a + t*b = g`,
/// `g` monic and `s, t` rescaled to match. Used for the Bezout step of
/// Hensel lifting.
pub fn extended_gcd(a: &[i64], b: &[i64], p: i64) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let (mut old_r, mut r) = (trim(a), trim(b));
    let (mut old_s, mut s) = (vec![1i64], vec![0i64]);
    let (mut old_t, mut t) = (vec![0i64], vec![1i64]);
    while !is_zero(&r) {
        let (q, rem) = div_mod(&old_r, &r, p);
        old_r = r;
        r = rem;
        let new_s = sub(&old_s, &mul(&q, &s, p), p);
        old_s = s;
        s = new_s;
        let new_t = sub(&old_t, &mul(&q, &t, p), p);
        old_t = t;
        t = new_t;
    }
    // Normalize to monic gcd; rescale s, t accordingly.
    if let Some(&lead) = old_r.last() {
        if lead != 0 && lead != 1 {
            if let Some(inv) = mod_inv(lead, p) {
                let g_monic: Vec<i64> = old_r.iter().map(|&c| modp(c * inv, p)).collect();
                let s_monic: Vec<i64> = old_s.iter().map(|&c| modp(c * inv, p)).collect();
                let t_monic: Vec<i64> = old_t.iter().map(|&c| modp(c * inv, p)).collect();
                return (trim(&g_monic), trim(&s_monic), trim(&t_monic));
            }
        }
    }
    (trim(&old_r), trim(&old_s), trim(&old_t))
}

/// `base^exp mod modulus` in `F_p[x]`. `exp` is an arbitrary-precision
/// [`Integer`] since distinct-degree and equal-degree factorization need
/// exponents like `p^d` and `(p^d - 1)/2`, which exceed `i64`.
pub fn pow_mod(base: &[i64], exp: &Integer, modulus: &[i64], p: i64) -> Vec<i64> {
    let zero = Integer::from(0i64);
    let two = Integer::from(2i64);
    let mut result = vec![1i64];
    let mut b = div_mod(base, modulus, p).1;
    let mut e = exp.clone();
    while e > zero {
        if &(&e % &two) == &Integer::from(1i64) {
            result = div_mod(&mul(&result, &b, p), modulus, p).1;
        }
        e = &e / &two;
        if e > zero {
            b = div_mod(&mul(&b, &b, p), modulus, p).1;
        }
    }
    result
}

/// Multiply by the inverse of the leading coefficient so the polynomial is
/// monic. The zero polynomial is returned unchanged.
pub fn make_monic(a: &[i64], p: i64) -> Vec<i64> {
    let t = trim(a);
    if is_zero(&t) {
        return t;
    }
    let lead = *t.last().unwrap();
    if lead == 1 {
        return t;
    }
    match mod_inv(lead, p) {
        Some(inv) => t.iter().map(|&c| modp(c * inv, p)).collect(),
        None => t,
    }
}

/// Modular inverse of an integer in `F_p` via the extended Euclidean
/// algorithm. Returns `None` if `a ≡ 0 (mod p)`.
pub fn mod_inv(a: i64, p: i64) -> Option<i64> {
    let (mut old_r, mut r) = (modp(a, p), p);
    let (mut old_s, mut s) = (1i64, 0i64);
    while r != 0 {
        let q = old_r / r;
        let tmp_r = old_r - q * r;
        old_r = r;
        r = tmp_r;
        let tmp_s = old_s - q * s;
        old_s = s;
        s = tmp_s;
    }
    if old_r != 1 {
        return None;
    }
    Some(modp(old_s, p))
}

/// Formal derivative `Σ i * a_i * x^(i-1) mod p`.
pub fn derivative_of(f: &[i64], p: i64) -> Vec<i64> {
    if f.len() <= 1 {
        return vec![0];
    }
    let mut out = Vec::with_capacity(f.len() - 1);
    for i in 1..f.len() {
        out.push(modp((i as i64) * f[i], p));
    }
    trim(&out)
}

/// The `p`-th root of a polynomial whose formal derivative vanishes over
/// `F_p`: `f' = 0` in characteristic `p` forces every exponent with a nonzero
/// coefficient to be a multiple of `p`, so `f(x) = g(x^p)` — and over the
/// *prime* field the coefficient `p`-th root is the identity (Fermat:
/// `c^p = c`, so `c` is its own `p`-th root). Hence `g[i] = f[p*i]`.
///
/// Panics if some coefficient sits at an exponent not divisible by `p`
/// (i.e. `f' != 0`), which would make the result meaningless.
fn pth_root(f: &[i64], p: i64) -> Vec<i64> {
    let f_t = trim(f);
    let pu = p as usize;
    let mut g = Vec::with_capacity(f_t.len() / pu + 1);
    for (i, &c) in f_t.iter().enumerate() {
        if i % pu == 0 {
            g.push(c);
        } else {
            assert!(
                c == 0,
                "fp_factor::pth_root: input is not a polynomial in x^{p}"
            );
        }
    }
    trim(&g)
}

/// `f` with every coefficient reduced into `[0, p)`, trimmed.
fn reduce(f: &[i64], p: i64) -> Vec<i64> {
    trim(&f.iter().map(|&c| modp(c, p)).collect::<Vec<i64>>())
}

/// `f` read modulo `p` and made monic, for the checked entries: panics
/// `"{who}: f = 0 has no factorization"` when `f ≡ 0 (mod p)`.
fn monic_nonzero(f: &[i64], p: i64, who: &str) -> Vec<i64> {
    let f_m = make_monic(&reduce(f, p), p);
    if is_zero(&f_m) {
        panic!("{who}: f = 0 has no factorization");
    }
    f_m
}

/// `Ok(())` iff the monic, reduced `f` is squarefree over `F_p`: `f != 0`, and
/// `deg f <= 0` or (`f' != 0` and `gcd(f, f') = 1`). Over `F_p` this is exact:
/// `f' = 0` means `f = g(x^p) = g^p`.
fn squarefree_check(f: &[i64], p: i64) -> Result<(), String> {
    if is_zero(f) {
        return Err("f = 0 is not squarefree".to_string());
    }
    if degree(f) <= 0 {
        return Ok(());
    }
    let df = derivative_of(f, p);
    if is_zero(&df) || degree(&gcd(f, &df, p)) != 0 {
        return Err("f is not squarefree".to_string());
    }
    Ok(())
}

/// The prime divisors of `d >= 1` in increasing order, by trial division
/// (`prime_divisors(1)` is empty).
fn prime_divisors(d: i64) -> Vec<i64> {
    let mut out = Vec::new();
    let mut rest = d;
    let mut l = 2;
    while rest > 1 {
        if l * l > rest {
            out.push(rest);
            break;
        }
        if rest % l == 0 {
            out.push(l);
            while rest % l == 0 {
                rest /= l;
            }
        }
        l += 1;
    }
    out
}

/// Radical of `f` over `F_p`: the product of the *distinct* monic irreducible
/// factors of `f`, each exactly once (monic; a nonzero constant gives `[1]`).
/// This is what distinct-degree factorization needs as input. MAGMA
/// `SquarefreeFactorization` (ch. 23 §23.8.1) returns the full decomposition;
/// this entry returns only its radical.
///
/// Characteristic-`p` subtleties handled correctly (both were bugs before):
/// * `f' = 0` means `f(x) = g(x^p)`; over the prime field `g`'s coefficients
///   are `f`'s directly (`pth_root`), and `rad(f) = rad(g)`: recurse.
/// * Even when `f' != 0`, factors whose multiplicity is divisible by `p`
///   survive inside `gcd(f, f')` at *full* multiplicity, so `f / gcd(f, f')`
///   alone misses them (e.g. `(x+1)^3 (x+2)` over `F_3` lost `x+1`). The
///   `p`-power part is split off and recursed on.
///
/// The coefficients of `f` are read modulo `p`.
///
/// # Panics
///
/// `p` is not a prime `<= MAX_P` (checked first, [`check_modulus`]);
/// `f ≡ 0 (mod p)`.
pub fn squarefree_factor(f: &[i64], p: i64) -> Vec<i64> {
    const WHO: &str = "fp_factor::squarefree_factor";
    if let Err(why) = check_modulus(p) {
        panic!("{WHO}: {why}");
    }
    radical(&monic_nonzero(f, p, WHO), p)
}

/// [`squarefree_factor`] without the checks: `p` is a prime `<= MAX_P` and
/// `f` is reduced.
fn radical(f: &[i64], p: i64) -> Vec<i64> {
    let f_m = make_monic(f, p);
    if degree(&f_m) <= 0 {
        return f_m;
    }
    let derivative = derivative_of(&f_m, p);
    if is_zero(&derivative) {
        // f = g(x^p): same distinct irreducible factors as g.
        return radical(&pth_root(&f_m, p), p);
    }
    let g = gcd(&f_m, &derivative, p);
    if degree(&g) == 0 {
        return f_m;
    }
    // With f = prod f_i^{e_i}: gcd(f, f') = prod_{p ∤ e_i} f_i^{e_i - 1}
    //                                     * prod_{p | e_i} f_i^{e_i},
    // so w := f / gcd(f, f') = prod_{p ∤ e_i} f_i (each once).
    let w = make_monic(&div_mod(&f_m, &g, p).0, p);
    // Strip every factor shared with w out of g; what remains is
    // c = prod_{p | e_i} f_i^{e_i}, a perfect p-th power with c' = 0.
    let mut c = g;
    loop {
        let y = gcd(&c, &w, p);
        if degree(&y) <= 0 {
            break;
        }
        c = make_monic(&div_mod(&c, &y, p).0, p);
    }
    if degree(&c) <= 0 {
        return w;
    }
    // rad(f) = w * rad(c); the two parts share no irreducible factors
    // (multiplicities not divisible by p vs. divisible by p).
    mul(&w, &radical(&c, p), p)
}

/// MAGMA `DistinctDegreeFactorization(f)` (ch. 23 §23.8.1; vzGG §14.2). The
/// coefficients of `f` are read modulo `p`. Returns pairs `(d, g_d)`, `d`
/// increasing, where `g_d` is the monic product of the irreducible factors of
/// `f` of degree exactly `d`. A nonzero constant gives `vec![]`.
///
/// # Panics
///
/// `p` is not a prime `<= MAX_P` (checked first, [`check_modulus`]); `f ≡ 0`;
/// `f` is not squarefree (MAGMA requires a squarefree input; on a repeated
/// factor the loop below would attach a wrong degree to a block).
pub fn distinct_degree_factor(f: &[i64], p: i64) -> Vec<(i64, Vec<i64>)> {
    const WHO: &str = "fp_factor::distinct_degree_factor";
    if let Err(why) = check_modulus(p) {
        panic!("{WHO}: {why}");
    }
    let f_m = make_monic(&reduce(f, p), p);
    if let Err(why) = squarefree_check(&f_m, p) {
        panic!("{WHO}: {why}");
    }
    ddf_squarefree(&f_m, p)
}

/// The distinct-degree loop on a monic, reduced, squarefree `f` over a prime
/// `p <= MAX_P` (checked by the caller). After step `d` every factor of degree
/// `<= d` is gone, so the remainder of degree `< 2(d+1)` is irreducible
/// (vzGG §14.2, \[Rab80\]).
fn ddf_squarefree(f_m: &[i64], p: i64) -> Vec<(i64, Vec<i64>)> {
    let mut f_rem = f_m.to_vec();
    let mut result: Vec<(i64, Vec<i64>)> = Vec::new();
    let mut h: Vec<i64> = vec![0, 1]; // x
    let mut d: i64 = 1;
    let p_big = Integer::from(p);
    while degree(&f_rem) >= 2 * d {
        // After d iterations h ≡ x^(p^d) mod f_rem.
        h = pow_mod(&h, &p_big, &f_rem, p);
        let h_minus_x = sub(&h, &[0, 1], p);
        let g = gcd(&h_minus_x, &f_rem, p);
        if degree(&g) > 0 {
            result.push((d, g.clone()));
            f_rem = div_mod(&f_rem, &g, p).0;
            f_rem = make_monic(&f_rem, p);
            h = div_mod(&h, &f_rem, p).1;
        }
        d += 1;
    }
    if degree(&f_rem) > 0 {
        result.push((degree(&f_rem), f_rem));
    }
    result
}

/// Largest modulus the `i64` arithmetic of this module handles exactly: every
/// product formed (`mul`: accumulator + a*b <= (p-1) + (p-1)^2; `div_mod`,
/// `make_monic`, `extended_gcd`, `mod_inv`: (p-1)^2) stays <= `i64::MAX` iff
/// p <= 3_037_000_500. The bound is `floor(sqrt(i64::MAX))` (gp:
/// `sqrtint(2^63-1) = 3037000499`), and 3_037_000_500 is even, so the primes
/// allowed are the same; the largest of them is 3_037_000_493 (gp:
/// `precprime(3037000499)`).
pub const MAX_P: i64 = 3_037_000_499;

/// Uniform vzGG Algorithm 14.8 tries per equal-degree block. On a certified
/// block each fails with probability <= 1/2 (vzGG Theorem 14.9), so all of
/// them fail with probability <= 2^-256.
const LAS_VEGAS_TRIES: usize = 256;

/// The equal-degree recursion runs the contract certificate on a block only
/// after this many failed tries (lazy certificate). On a valid block 32
/// consecutive failures have probability <= 2^-32, so the certificate almost
/// never runs on the path from [`factor`], whose blocks are valid.
const TRIES_BEFORE_CERTIFICATE: usize = 32;

/// SplitMix64 (Steele, Lea, Flood 2014): the state advances by the odd constant
/// `0x9E3779B97F4A7C15` mod 2^64 (period exactly 2^64) and the output mixer is
/// a bijection of `u64`, so no output repeats within 2^64 draws. Seeded from
/// the input, so factorization is reproducible across runs.
struct SplitMix64(u64);

impl SplitMix64 {
    /// The seed of the equal-degree stage, derived from the block, `d` and `p`.
    fn seeded(f_m: &[i64], d: i64, p: i64) -> Self {
        let mut seed: u64 = 0x243F_6A88_85A3_08D3 ^ (d as u64).wrapping_mul(0x100_0193);
        seed ^= (p as u64).wrapping_mul(0x85EB_CA6B);
        for (i, &c) in f_m.iter().enumerate() {
            seed = seed
                .wrapping_mul(0x0100_0000_01B3)
                .wrapping_add((c as u64) ^ (i as u64));
        }
        SplitMix64(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Exactly uniform in `[0, n)` by rejection: with `r = 2^64 mod n`, accept
    /// `v <= u64::MAX - r` (a multiple of `n` many values) and return `v % n`.
    /// Terminates: the `r < n` rejected values each occur once per period of
    /// 2^64 outputs, so at most `r` consecutive draws are rejected.
    fn below(&mut self, n: i64) -> i64 {
        assert!(
            n >= 1,
            "fp_factor::SplitMix64::below: n = {n} must be positive"
        );
        let m = n as u64;
        let r = (u64::MAX % m + 1) % m;
        loop {
            let v = self.next_u64();
            if v <= u64::MAX - r {
                return (v % m) as i64;
            }
        }
    }
}

/// The trace map `T(a) = a + a^2 + a^4 + ... + a^(2^(d-1)) rem f` over `F_2`
/// (\[CZ81\]; vzGG §14.3, even q): on every CRT component `F_2[x]/(f_i)` with
/// `deg f_i = d` it is the absolute trace `F_(2^d) -> F_2`, so
/// `gcd(T(a), f) = prod_{Tr(a_i) = 0} f_i`. `a` is reduced mod `f` first.
fn trace_map(a: &[i64], f: &[i64], d: i64, p: i64) -> Vec<i64> {
    let a0 = div_mod(a, f, p).1;
    let mut sum = a0.clone();
    let mut current = a0;
    for _ in 1..d {
        current = div_mod(&mul(&current, &current, p), f, p).1;
        sum = add(&sum, &current, p);
    }
    sum
}

/// One splitting step of vzGG Algorithm 14.8 on the monic block `f` (degree
/// `n`, a product of distinct irreducibles of degree `d`): step 3, a proper
/// `gcd(a, f)`; then `gcd(T(a), f)` (p = 2, [`trace_map`]) or
/// `gcd(a^e - 1, f)` with `e = (p^d - 1)/2` (odd p). Returns a proper monic
/// divisor of `f`, or `None` when this `a` does not split `f`.
fn split_with(a: &[i64], f: &[i64], d: i64, p: i64, e: &Integer) -> Option<Vec<i64>> {
    let n = degree(f);
    let proper = |g: &[i64]| {
        let dg = degree(g);
        dg > 0 && dg < n
    };
    let g = gcd(a, f, p);
    if proper(&g) {
        return Some(g);
    }
    let t = if p == 2 {
        trace_map(a, f, d, p)
    } else {
        sub(&pow_mod(a, e, f, p), &[1], p)
    };
    let g = gcd(&t, f, p);
    if proper(&g) {
        Some(g)
    } else {
        None
    }
}

/// `Ok(())` iff `p` is a prime `<= MAX_P` (the module's exact-arithmetic
/// bound); otherwise `Err("p = {p} is not a prime <= 3037000499")`.
///
/// Exact: [`rustmath_integers::prime::is_prime_u64`] is a proof of primality
/// or compositeness for every `u64` (strong tests to a deterministic base
/// set), so no composite passes, and every prime `<= MAX_P` does.
pub fn check_modulus(p: i64) -> Result<(), String> {
    if (2..=MAX_P).contains(&p) && rustmath_integers::prime::is_prime_u64(p as u64) {
        Ok(())
    } else {
        Err(format!("p = {p} is not a prime <= {MAX_P}"))
    }
}

/// The order of the factor lists of this module: by degree, then by the
/// coefficients from the top down. This is the order of PARI's `factormod`
/// (and of `berlekamp::factor`); MAGMA's `Factorization` fixes no order.
/// Both arguments are compared after trimming.
pub fn factor_order(a: &[i64], b: &[i64]) -> Ordering {
    let (a, b) = (trim(a), trim(b));
    degree(&a)
        .cmp(&degree(&b))
        .then_with(|| a.iter().rev().cmp(b.iter().rev()))
}

/// `Ok(())` iff `f` (monic, coefficients in `[0, p)`) is a squarefree product
/// of distinct monic irreducibles of degree exactly `d` over `F_p`, with `p` a
/// prime `<= MAX_P`; otherwise the first violated condition.
///
/// Rabin's criterion (\[Rab80\], vzGG §14.2) generalised to products:
/// `x^(p^d) ≡ x (mod f)` iff every irreducible factor of the squarefree `f`
/// has degree dividing `d`, and `gcd(x^(p^(d/l)) - x, f) = 1` for every prime
/// `l | d` excludes the proper divisors of `d`. Cost: `d` Frobenius steps, i.e.
/// about `d log2(p)` modular squarings.
fn equal_degree_contract(f: &[i64], d: i64, p: i64) -> Result<(), String> {
    check_modulus(p)?;
    block_contract(f, d, p)
}

/// The conditions of [`equal_degree_contract`] after the prime check (the
/// caller has already established that `p` is a prime `<= MAX_P`).
///
/// The Frobenius ladder `X_k = x^(p^k) rem f` runs for `k = 1..=d`, but only
/// `X_0`, the current rung and the rungs `X_(d/l)` for the primes `l | d` are
/// kept: memory `(omega(d) + 2)` polynomials of degree `< deg f` instead of
/// `d + 1`.
fn block_contract(f: &[i64], d: i64, p: i64) -> Result<(), String> {
    if d < 1 {
        return Err(format!("d = {d} must be at least 1"));
    }
    if is_zero(f) {
        return Err("f = 0 has no factorization".to_string());
    }
    let n = degree(f);
    if n < 1 {
        return Err("f is constant".to_string());
    }
    if n % d != 0 {
        return Err(format!("d = {d} does not divide deg f = {n}"));
    }
    squarefree_check(f, p)?;
    let primes = prime_divisors(d);
    // The rungs to keep, increasing (d / l decreases as l increases).
    let mut wanted: Vec<i64> = primes.iter().map(|&l| d / l).collect();
    wanted.sort_unstable();
    let p_big = Integer::from(p);
    let x0 = div_mod(&[0, 1], f, p).1;
    let mut kept: Vec<(i64, Vec<i64>)> = Vec::with_capacity(wanted.len());
    let mut current = x0.clone();
    let mut next = 0;
    for k in 1..=d {
        current = pow_mod(&current, &p_big, f, p);
        if next < wanted.len() && wanted[next] == k {
            kept.push((k, current.clone()));
            next += 1;
        }
    }
    if trim(&current) != trim(&x0) {
        return Err(format!(
            "f has an irreducible factor whose degree does not divide {d}"
        ));
    }
    for &l in &primes {
        let k = d / l;
        let xk = &kept
            .iter()
            .find(|(j, _)| *j == k)
            .expect("fp_factor: every rung d/l is kept")
            .1;
        let g = gcd(&sub(xk, &x0, p), f, p);
        if degree(&g) != 0 {
            return Err(format!(
                "f has an irreducible factor of degree dividing {k}"
            ));
        }
    }
    Ok(())
}

/// The equal-degree recursion (vzGG Algorithm 14.10) on the monic block `f`.
///
/// Per block: at most [`LAS_VEGAS_TRIES`] uniform tries of vzGG Algorithm 14.8
/// ([`split_with`]). Before try number [`TRIES_BEFORE_CERTIFICATE`] the block
/// must pass [`equal_degree_contract`] (else panic: the block violates the
/// contract). If all tries fail, panic. No path returns an unsplit block.
/// `n == d` pushes `f` unchecked: the caller guarantees the contract (public
/// entry: eager certificate; [`factor`]: DDF blocks of the radical over a
/// checked prime).
fn edf_rec(f: &[i64], d: i64, p: i64, out: &mut Vec<Vec<i64>>) {
    let n = degree(f);
    if n == d && d >= 1 {
        out.push(f.to_vec());
        return;
    }
    if d < 1 || n < d {
        // Nothing to split: the block violates the contract.
        contract_or_panic(f, d, p);
        unreachable!("fp_factor: a block with d < 1 or deg f < d passed the certificate");
    }
    let mut rng = SplitMix64::seeded(f, d, p);
    let e = if p == 2 {
        Integer::zero()
    } else {
        (&Integer::from(p).pow(d as u32) - &Integer::one()) / Integer::from(2i64)
    };
    let mut found: Option<Vec<i64>> = None;
    for t in 0..LAS_VEGAS_TRIES {
        if t == TRIES_BEFORE_CERTIFICATE {
            contract_or_panic(f, d, p);
        }
        // Uniform a with deg a < n (vzGG 14.8, step 1).
        let a = trim(&(0..n).map(|_| rng.below(p)).collect::<Vec<i64>>());
        if let Some(g) = split_with(&a, f, d, p, &e) {
            found = Some(g);
            break;
        }
    }
    let g = found.unwrap_or_else(|| {
        panic!(
            "fp_factor: no split of a certified equal-degree block in {LAS_VEGAS_TRIES} \
             uniform tries (probability <= 2^-{LAS_VEGAS_TRIES}); the generator is broken"
        )
    });
    let g1 = make_monic(&g, p);
    let g2 = make_monic(&div_mod(f, &g1, p).0, p);
    edf_rec(&g1, d, p, out);
    edf_rec(&g2, d, p, out);
}

/// Panic with the reason if `f` is not a valid equal-degree block.
fn contract_or_panic(f: &[i64], d: i64, p: i64) {
    if let Err(why) = equal_degree_contract(f, d, p) {
        panic!("fp_factor: equal-degree block violates its contract ({why})");
    }
}

/// Equal-degree factorization (MAGMA `EqualDegreeFactorization`, ch. 23
/// §23.8.1; Cantor–Zassenhaus \[CZ81\], vzGG Algorithms 14.8/14.10). Splits a
/// squarefree `f` whose irreducible factors all have degree `d` into those
/// monic factors. For `p = 2` the trace map is used; for odd `p`,
/// `a^((p^d-1)/2) - 1`. The coefficients of `f` are read modulo `p`.
///
/// The contract is **checked** (unlike MAGMA, whose result is "unpredictable
/// if conditions are not met"): before any splitting, `f` must pass a
/// certificate (Rabin's criterion for products, \[Rab80\]) proving that it is a
/// squarefree product of distinct monic irreducibles of degree exactly `d`
/// over `F_p`. The returned list therefore always consists of irreducible
/// factors of degree `d` whose product is `f` made monic, sorted by
/// [`factor_order`] (PARI's `factormod` order).
///
/// A nonzero constant `f` returns `vec![]` (the empty product).
///
/// # Panics
///
/// - `p` is not a prime `<= MAX_P` (checked first; [`MAX_P`] is the module's
///   exact-arithmetic bound);
/// - `f ≡ 0 (mod p)`;
/// - `d < 1`, `d` does not divide `deg f`, `f` is not squarefree, or `f` has an
///   irreducible factor of degree other than `d` (including `deg f == d` with
///   `f` reducible); the message names the violated condition;
/// - the sampler fails 256 consecutive uniform tries on a certified block
///   (probability <= 2^-256 per block: a broken generator); no value is
///   returned in that case.
pub fn equal_degree_factor(f: &[i64], d: i64, p: i64) -> Vec<Vec<i64>> {
    const WHO: &str = "fp_factor::equal_degree_factor";
    if let Err(why) = check_modulus(p) {
        panic!("{WHO}: {why}");
    }
    let f_m = monic_nonzero(f, p, WHO);
    if degree(&f_m) == 0 {
        return Vec::new();
    }
    // Eager certificate (`p` was checked above).
    if let Err(why) = block_contract(&f_m, d, p) {
        panic!("{WHO}: {why}");
    }
    let mut out = Vec::new();
    edf_rec(&f_m, d, p, &mut out);
    out.sort_by(|a, b| factor_order(a, b));
    out
}

/// Full irreducible factorization of `f` in `F_p[x]` (MAGMA `Factorization`
/// over F_p, ch. 23 §23.8.1, without multiplicities). The coefficients of `f`
/// are read modulo `p`. The radical ([`squarefree_factor`]) is computed first
/// so repeated factors are coalesced (multiplicities are *not* tracked: every
/// returned factor is distinct and monic; use [`factor_with_multiplicity`] to
/// recover exponents). Correct on inseparable inputs (`x^2 + 1` over `F_2`,
/// `x^5 - 2` over `F_5`, ...) since the radical handles the characteristic-`p`
/// cases. Then distinct-degree factorization and the equal-degree recursion
/// (uniform Las Vegas tries, with the contract certificate after 32 failures),
/// so a returned factor is always irreducible: no block is ever returned
/// unsplit. A nonzero constant `f` returns `vec![]`.
///
/// The output is sorted by [`factor_order`]: by degree, then by coefficients
/// from the top down, the order of PARI's `factormod`.
///
/// # Panics
///
/// - `p` is not a prime `<= MAX_P` (checked first, [`check_modulus`]);
/// - `f ≡ 0 (mod p)` (the zero polynomial has no factorization);
/// - the sampler fails 256 consecutive uniform tries on a certified block
///   (probability <= 2^-256 per block); no value is returned in that case.
pub fn factor(f: &[i64], p: i64) -> Vec<Vec<i64>> {
    const WHO: &str = "fp_factor::factor";
    if let Err(why) = check_modulus(p) {
        panic!("{WHO}: {why}");
    }
    factor_monic(&monic_nonzero(f, p, WHO), p)
}

/// [`factor`] without the prime check, for callers that have proved `p` a
/// prime `<= MAX_P` themselves (`zassenhaus::choose_prime`: trial division and
/// a loop bound). The coefficients are still read modulo `p`.
///
/// # Panics
///
/// `f ≡ 0 (mod p)`.
pub(crate) fn factor_prime(f: &[i64], p: i64) -> Vec<Vec<i64>> {
    factor_monic(&monic_nonzero(f, p, "fp_factor::factor_prime"), p)
}

/// The factorization of the monic, reduced, nonzero `f_m` over a checked prime.
fn factor_monic(f_m: &[i64], p: i64) -> Vec<Vec<i64>> {
    if degree(f_m) <= 0 {
        return Vec::new();
    }
    let rad = radical(f_m, p);
    let mut out: Vec<Vec<i64>> = Vec::new();
    for (d, gd) in ddf_squarefree(&rad, p) {
        let mut pieces = Vec::new();
        edf_rec(&make_monic(&gd, p), d, p, &mut pieces);
        out.extend(pieces.into_iter().map(|piece| make_monic(&piece, p)));
    }
    out.sort_by(|a, b| factor_order(a, b));
    out
}

/// Irreducible factorization of `f` in `F_p[x]` *with multiplicities*:
/// returns `(g_i, e_i)` pairs with each `g_i` monic irreducible and distinct,
/// such that `prod g_i^{e_i}` equals `f` (read modulo `p`) made monic, in the
/// order of [`factor`]. Multiplicities are found by repeated exact division of
/// `f` by each distinct factor.
///
/// # Panics
///
/// As for [`factor`]: `p` is not a prime `<= MAX_P` (checked first);
/// `f ≡ 0 (mod p)`.
pub fn factor_with_multiplicity(f: &[i64], p: i64) -> Vec<(Vec<i64>, u32)> {
    const WHO: &str = "fp_factor::factor_with_multiplicity";
    if let Err(why) = check_modulus(p) {
        panic!("{WHO}: {why}");
    }
    let f_m = monic_nonzero(f, p, WHO);
    factor_monic(&f_m, p)
        .into_iter()
        .map(|g| {
            let mut mult = 0u32;
            let mut rem = f_m.clone();
            loop {
                let (q, r) = div_mod(&rem, &g, p);
                if !is_zero(&r) {
                    break;
                }
                mult += 1;
                rem = q;
            }
            debug_assert!(mult >= 1, "factor() returned a non-divisor");
            (g, mult)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    // Multiply a list of factors mod p and compare (monic) against expected.
    fn product(factors: &[Vec<i64>], p: i64) -> Vec<i64> {
        let mut acc = vec![1i64];
        for f in factors {
            acc = mul(&acc, f, p);
        }
        make_monic(&acc, p)
    }

    #[test]
    fn test_basic_arith() {
        let p = 7;
        // (x + 1)(x + 2) = x^2 + 3x + 2
        let a = vec![1, 1];
        let b = vec![2, 1];
        assert_eq!(mul(&a, &b, p), vec![2, 3, 1]);
        // division inverts multiplication
        let prod = vec![2, 3, 1];
        let (q, r) = div_mod(&prod, &a, p);
        assert_eq!(q, vec![2, 1]);
        assert!(is_zero(&r));
    }

    #[test]
    fn test_mod_inv() {
        let p = 7;
        for a in 1..7 {
            let inv = mod_inv(a, p).unwrap();
            assert_eq!((a * inv) % p, 1);
        }
        assert_eq!(mod_inv(0, 7), None);
    }

    #[test]
    fn test_extended_gcd_bezout() {
        let p = 5;
        // g0 = x + 2, h0 = x + 3 are coprime mod 5.
        let g0 = vec![2, 1];
        let h0 = vec![3, 1];
        let (g, s, t) = extended_gcd(&g0, &h0, p);
        assert_eq!(g, vec![1]); // coprime
        // s*g0 + t*h0 == 1
        let lhs = add(&mul(&s, &g0, p), &mul(&t, &h0, p), p);
        assert_eq!(lhs, vec![1]);
    }

    #[test]
    fn test_factor_split_quadratic() {
        let p = 5;
        // x^2 + 1 = (x + 2)(x + 3) over F_5.
        let f = vec![1, 0, 1];
        let mut fac = factor(&f, p);
        fac.sort();
        assert_eq!(fac.len(), 2);
        assert_eq!(product(&fac, p), make_monic(&f, p));
        // each factor is linear
        for fct in &fac {
            assert_eq!(degree(fct), 1);
        }
    }

    #[test]
    fn test_factor_irreducible() {
        let p = 5;
        // x^2 + 2 is irreducible over F_5 (no square root of -2 = 3 mod 5).
        let f = vec![2, 0, 1];
        let fac = factor(&f, p);
        assert_eq!(fac.len(), 1);
        assert_eq!(fac[0], make_monic(&f, p));
        assert!(distinct_degree_factor(&make_monic(&f, p), p)
            .iter()
            .all(|(d, _)| *d == 2));
    }

    #[test]
    fn test_factor_repeated_root_squarefree() {
        let p = 7;
        // (x + 1)^2 (x + 3) — squarefree part drops the multiplicity.
        let lin1 = vec![1, 1];
        let lin3 = vec![3, 1];
        let f = mul(&mul(&lin1, &lin1, p), &lin3, p);
        let mut fac = factor(&f, p);
        fac.sort();
        // distinct factors only: (x+1), (x+3)
        assert_eq!(fac, vec![vec![1, 1], vec![3, 1]]);
    }

    #[test]
    fn test_factor_cubic_product() {
        let p = 11;
        // (x+1)(x+4)(x^2 + x + 1)  — mixed degrees, two irreducible degrees.
        let lin1 = vec![1, 1];
        let lin4 = vec![4, 1];
        let quad = vec![1, 1, 1];
        let f = mul(&mul(&lin1, &lin4, p), &quad, p);
        let fac = factor(&f, p);
        // product of returned factors equals monic f
        assert_eq!(product(&fac, p), make_monic(&f, p));
        // degrees present: 1,1,2
        let mut degs: Vec<i64> = fac.iter().map(|g| degree(g)).collect();
        degs.sort();
        assert_eq!(degs, vec![1, 1, 2]);
    }

    #[test]
    fn test_factor_gf2_trace_split() {
        // p = 2 exercises the trace-form equal-degree split.
        let p = 2;
        // x^2 + x = x(x+1) over F_2.
        let f = vec![0, 1, 1];
        let mut fac = factor(&f, p);
        fac.sort();
        assert_eq!(fac, vec![vec![0, 1], vec![1, 1]]);
        // x^2 + x + 1 is irreducible over F_2.
        let irr = vec![1, 1, 1];
        assert_eq!(factor(&irr, p), vec![vec![1, 1, 1]]);
    }

    // Multiply factor^mult pairs mod p (monic) — for the self-certifying
    // "product of returned factors equals the input" gate.
    fn product_with_mult(factors: &[(Vec<i64>, u32)], p: i64) -> Vec<i64> {
        let mut acc = vec![1i64];
        for (f, m) in factors {
            for _ in 0..*m {
                acc = mul(&acc, f, p);
            }
        }
        make_monic(&acc, p)
    }

    // Assert factor_with_multiplicity(f) == expected (sorted), plus the
    // self-certifying product check, plus factor() giving the distinct parts.
    fn check_factorization(f: &[i64], p: i64, expected: &[(&[i64], u32)]) {
        let mut fac = factor_with_multiplicity(f, p);
        fac.sort();
        let mut exp: Vec<(Vec<i64>, u32)> =
            expected.iter().map(|(g, m)| (g.to_vec(), *m)).collect();
        exp.sort();
        assert_eq!(fac, exp, "factorization of {f:?} mod {p}");
        // Self-certifying: multiply everything back together.
        assert_eq!(product_with_mult(&fac, p), make_monic(f, p));
        // factor() returns exactly the distinct factors.
        let mut distinct = factor(f, p);
        distinct.sort();
        let mut exp_distinct: Vec<Vec<i64>> = exp.iter().map(|(g, _)| g.clone()).collect();
        exp_distinct.sort();
        assert_eq!(distinct, exp_distinct);
    }

    #[test]
    fn test_factor_inseparable_gf2() {
        // gp: factormod(x^2+1, 2) = (x+1)^2 — the old code returned x^2+1 whole.
        check_factorization(&[1, 0, 1], 2, &[(&[1, 1], 2)]);
        // gp: factormod(x^4+x^2+1, 2) = (x^2+x+1)^2.
        check_factorization(&[1, 0, 1, 0, 1], 2, &[(&[1, 1, 1], 2)]);
    }

    #[test]
    fn test_factor_inseparable_gf5() {
        // gp: factormod(x^5-2, 5) = (x+3)^5. (x-c)^5 = x^5 - c^5 = x^5 - c over
        // F_5 by Fermat, so c = 2, i.e. the factor is x - 2 = x + 3.
        check_factorization(&[3, 0, 0, 0, 0, 1], 5, &[(&[3, 1], 5)]);
    }

    #[test]
    fn test_factor_inseparable_gf3() {
        // gp: factormod(x^6+1, 3) = (x^2+1)^3 (x^2+1 irreducible over F_3).
        check_factorization(&[1, 0, 0, 0, 0, 0, 1], 3, &[(&[1, 0, 1], 3)]);
    }

    #[test]
    fn test_factor_p_divides_multiplicity_but_derivative_nonzero() {
        // f = (x+1)^3 (x+2) over F_3 = x^4 + 2x^3 + x + 2 (gp-expanded); f' != 0
        // but gcd(f, f') = (x+1)^3 swallows (x+1) entirely, so the old radical
        // f/gcd(f,f') = x+2 silently LOST the factor x+1.
        check_factorization(&[2, 1, 0, 2, 1], 3, &[(&[1, 1], 3), (&[2, 1], 1)]);
        // gp: factormod of x^8+2x^7+x^6+x^5+2x^4+x^3+2x^2+x+2 over F_3
        //   = (x+1)^2 (x^2+x+2)^3 — mixed p|e and p∤e multiplicities.
        check_factorization(
            &[2, 1, 2, 1, 2, 1, 1, 2, 1],
            3,
            &[(&[1, 1], 2), (&[2, 1, 1], 3)],
        );
    }

    #[test]
    fn test_factor_separable_regression_gp_derived() {
        // Old behavior must be preserved exactly on separable inputs.
        // gp: factormod(x^8+x^4+x^2+x+1, 7)
        //   = (x^4+x^3+x^2+x+1)(x^4+6x^3+1).
        check_factorization(
            &[1, 1, 1, 0, 1, 0, 0, 0, 1],
            7,
            &[(&[1, 1, 1, 1, 1], 1), (&[1, 0, 0, 6, 1], 1)],
        );
        // gp: factormod(x^5+4x+1, 11) = (x+6)(x^2+2x+10)(x^2+3x+9).
        check_factorization(
            &[1, 4, 0, 0, 0, 1],
            11,
            &[(&[6, 1], 1), (&[10, 2, 1], 1), (&[9, 3, 1], 1)],
        );
    }

    #[test]
    fn test_squarefree_factor_is_the_radical() {
        // rad((x+1)^3 (x+2)) = (x+1)(x+2) = x^2 + 2 mod 3... derive:
        // (x+1)(x+2) = x^2 + 3x + 2 = x^2 + 2 over F_3.
        let f = [2, 1, 0, 2, 1]; // (x+1)^3 (x+2) over F_3
        assert_eq!(squarefree_factor(&f, 3), vec![2, 0, 1]);
        // rad(x^6+1) over F_3 = x^2+1.
        assert_eq!(squarefree_factor(&[1, 0, 0, 0, 0, 0, 1], 3), vec![1, 0, 1]);
        // Separable input: radical is the input itself (monic).
        assert_eq!(squarefree_factor(&[1, 1, 1], 5), vec![1, 1, 1]);
    }

    // ---------------------------------------------------------------------
    // Equal-degree factorization (X1a): reference rows.
    //
    // Each row is (id, p, d, f, PARI factors in PARI's printed order).
    // Coefficients are little-endian. Reference: PARI/GP 2.15, for every row
    //   gp> le(g) = my(v = Vec(lift(g))); vector(#v, i, v[#v + 1 - i]);
    //   gp> apply(le, factormod(Pol(Vecrev(f)), p)[,1]~)
    // (every multiplicity is 1 and every degree is d). The inputs are
    //   A1 x^2+x, A2 (x^3+x+1)(x^3+x^2+1), A3 (x^16-x)/(x^4-x),
    //   A4 (x^32-x)/(x^2-x) over F_2; B1 x^3-x, B2 (x^9-x)/(x^3-x),
    //   B3 (x^3+2x+1)(x^3+2x+2) over F_3; C1 x^5-x, C2 x^4+1,
    //   C3 (x^3+x+1)(x^3+x+4)(x^3+2x+1) over F_5; D1 (x-1)...(x-5),
    //   D2 (x^2+1)(x^2+2)(x^2+4), D3 (x^3+x+4)(x^3+x+13) over F_(2^31-1);
    //   E1 (x^49-x)/(x^7-x) over F_7 (all 21 monic irreducible quadratics);
    //   D4 (x-1)(x-2)(x-3), D5 (x^2+2)(x^2+3) over F_3037000493
    //   (gp: precprime(3037000499) = 3037000493, the largest prime <= MAX_P;
    //   kronecker(-2, p) = kronecker(-3, p) = -1).
    // ---------------------------------------------------------------------
    #[allow(clippy::type_complexity)]
    fn edf_reference_rows() -> Vec<(&'static str, i64, i64, Vec<i64>, Vec<Vec<i64>>)> {
        let mut e1 = vec![0i64; 43];
        for k in 0..8 {
            e1[6 * k] = 1;
        }
        vec![
            ("A1", 2, 1, vec![0, 1, 1], vec![vec![0, 1], vec![1, 1]]),
            (
                "A2",
                2,
                3,
                vec![1, 1, 1, 1, 1, 1, 1],
                vec![vec![1, 1, 0, 1], vec![1, 0, 1, 1]],
            ),
            (
                "A3",
                2,
                4,
                vec![1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                vec![
                    vec![1, 1, 0, 0, 1],
                    vec![1, 0, 0, 1, 1],
                    vec![1, 1, 1, 1, 1],
                ],
            ),
            (
                "A4",
                2,
                5,
                vec![1; 31],
                vec![
                    vec![1, 0, 1, 0, 0, 1],
                    vec![1, 0, 0, 1, 0, 1],
                    vec![1, 1, 1, 1, 0, 1],
                    vec![1, 1, 1, 0, 1, 1],
                    vec![1, 1, 0, 1, 1, 1],
                    vec![1, 0, 1, 1, 1, 1],
                ],
            ),
            (
                "B1",
                3,
                1,
                vec![0, 2, 0, 1],
                vec![vec![0, 1], vec![1, 1], vec![2, 1]],
            ),
            (
                "B2",
                3,
                2,
                vec![1, 0, 1, 0, 1, 0, 1],
                vec![vec![1, 0, 1], vec![2, 1, 1], vec![2, 2, 1]],
            ),
            (
                "B3",
                3,
                3,
                vec![2, 0, 1, 0, 1, 0, 1],
                vec![vec![1, 2, 0, 1], vec![2, 2, 0, 1]],
            ),
            (
                "C1",
                5,
                1,
                vec![0, 4, 0, 0, 0, 1],
                vec![vec![0, 1], vec![1, 1], vec![2, 1], vec![3, 1], vec![4, 1]],
            ),
            (
                "C2",
                5,
                2,
                vec![1, 0, 0, 0, 1],
                vec![vec![2, 0, 1], vec![3, 0, 1]],
            ),
            (
                "C3",
                5,
                3,
                vec![4, 3, 1, 1, 2, 0, 1, 4, 0, 1],
                vec![vec![1, 1, 0, 1], vec![4, 1, 0, 1], vec![1, 2, 0, 1]],
            ),
            (
                "D1",
                2147483647,
                1,
                vec![2147483527, 274, 2147483422, 85, 2147483632, 1],
                vec![
                    vec![2147483642, 1],
                    vec![2147483643, 1],
                    vec![2147483644, 1],
                    vec![2147483645, 1],
                    vec![2147483646, 1],
                ],
            ),
            (
                "D2",
                2147483647,
                2,
                vec![8, 0, 14, 0, 7, 0, 1],
                vec![vec![1, 0, 1], vec![2, 0, 1], vec![4, 0, 1]],
            ),
            (
                "D3",
                2147483647,
                3,
                vec![52, 17, 1, 17, 2, 0, 1],
                vec![vec![4, 1, 0, 1], vec![13, 1, 0, 1]],
            ),
            (
                "E1",
                7,
                2,
                e1,
                vec![
                    vec![1, 0, 1],
                    vec![2, 0, 1],
                    vec![4, 0, 1],
                    vec![3, 1, 1],
                    vec![4, 1, 1],
                    vec![6, 1, 1],
                    vec![2, 2, 1],
                    vec![3, 2, 1],
                    vec![5, 2, 1],
                    vec![1, 3, 1],
                    vec![5, 3, 1],
                    vec![6, 3, 1],
                    vec![1, 4, 1],
                    vec![5, 4, 1],
                    vec![6, 4, 1],
                    vec![2, 5, 1],
                    vec![3, 5, 1],
                    vec![5, 5, 1],
                    vec![3, 6, 1],
                    vec![4, 6, 1],
                    vec![6, 6, 1],
                ],
            ),
            (
                "D4",
                3037000493,
                1,
                vec![3037000487, 11, 3037000487, 1],
                vec![
                    vec![3037000490, 1],
                    vec![3037000491, 1],
                    vec![3037000492, 1],
                ],
            ),
            (
                "D5",
                3037000493,
                2,
                vec![6, 0, 5, 0, 1],
                vec![vec![2, 0, 1], vec![3, 0, 1]],
            ),
        ]
    }

    /// T2: the exact, unsorted output order of `factor` and of
    /// `equal_degree_factor` is PARI's `factormod` order (degree, then
    /// coefficients from the top down). gp, for every row (inputs at
    /// `edf_reference_rows`, plus three mixed-degree rows):
    ///   gp> le(g) = my(v = Vec(lift(g))); vector(#v, i, v[#v + 1 - i]);
    ///   gp> apply(le, factormod(f, p)[,1]~)
    /// The extra rows are factormod(x^5+4*x+1, 11),
    /// factormod(x^8+x^4+x^2+x+1, 7) and factormod((x-1)*(x^2+2)*(x^2+3), 5).
    #[test]
    fn factor_order_is_pari_order() {
        let rows = edf_reference_rows();
        for (id, p, d, f, pari) in &rows {
            // The reference lists are written in PARI's printed order.
            let via_factor = factor(f, *p);
            let via_edf = equal_degree_factor(f, *d, *p);
            assert_eq!(&via_factor, pari, "factor order, row {id}");
            assert_eq!(&via_edf, pari, "equal_degree_factor order, row {id}");
            let mut sorted = pari.clone();
            sorted.sort_by(|a, b| factor_order(a, b));
            assert_eq!(&sorted, pari, "row {id}: PARI order is factor_order");
        }
        let mixed: Vec<(i64, Vec<i64>, Vec<Vec<i64>>)> = vec![
            (11, vec![1, 4, 0, 0, 0, 1], vec![vec![6, 1], vec![10, 2, 1], vec![9, 3, 1]]),
            (
                7,
                vec![1, 1, 1, 0, 1, 0, 0, 0, 1],
                vec![vec![1, 1, 1, 1, 1], vec![1, 0, 0, 6, 1]],
            ),
            // (x-1)(x^2+2)(x^2+3) = (x-1)(x^4+1) mod 5.
            (5, vec![4, 1, 0, 0, 4, 1], vec![vec![4, 1], vec![2, 0, 1], vec![3, 0, 1]]),
        ];
        for (p, f, pari) in mixed {
            assert_eq!(factor(&f, p), pari, "factor order of {f:?} mod {p}");
            let with_mult: Vec<Vec<i64>> =
                factor_with_multiplicity(&f, p).into_iter().map(|(g, _)| g).collect();
            assert_eq!(with_mult, pari, "factor_with_multiplicity order of {f:?} mod {p}");
        }
    }

    /// T1: the public entry on every reference row equals PARI `factormod`
    /// as a set (both sides sorted; gp commands at `edf_reference_rows`).
    #[test]
    fn equal_degree_factor_matches_pari() {
        for (id, p, d, f, mut pari) in edf_reference_rows() {
            pari.sort();
            let mut got = equal_degree_factor(&f, d, p);
            got.sort();
            assert_eq!(got, pari, "row {id}: equal_degree_factor(f, {d}, {p})");
            let mut via_factor = factor(&f, p);
            via_factor.sort();
            assert_eq!(via_factor, pari, "row {id}: factor(f, {p})");
        }
    }

    /// T3: the equal-degree recursion on its own (uniform vzGG Algorithm 14.8
    /// tries, lazy certificate) equals PARI `factormod` as a set on every
    /// reference row (E1: 21 quadratics).
    #[test]
    fn edf_rec_matches_pari() {
        for (id, p, d, f, mut pari) in edf_reference_rows() {
            pari.sort();
            let mut out = Vec::new();
            edf_rec(&make_monic(&f, p), d, p, &mut out);
            out.sort();
            assert_eq!(out, pari, "row {id}: edf_rec");
        }
    }

    /// T4: exhaustive count of the `a` (all p^n coefficient vectors, deg a < n)
    /// for which one phase-2 try splits the block, against PARI:
    ///   gp> cnt(f, d, p) = { my(n = poldegree(f), c = 0, F = f*Mod(1,p),
    ///         e = (p^d-1)/2); forvec(v = vector(n, i, [0, p-1]),
    ///         my(a = Pol(Vecrev(v))*Mod(1,p), g, t); if(a == 0, next);
    ///         g = gcd(a, F); if(poldegree(g) > 0 && poldegree(g) < n, c++; next);
    ///         if(p == 2, t = sum(i=0, d-1, lift(Mod(a, F)^(2^i))),
    ///                    t = lift(Mod(a, F)^e) - 1);
    ///         g = gcd(t*Mod(1,p), F); if(poldegree(g) > 0 && poldegree(g) < n, c++)); c }
    /// The counts equal the closed forms p^n - 1 - 2((Q-1)/2)^r (odd p) and
    /// 2^n - 1 - (Q/2-1)^r - (Q/2)^r (p = 2), Q = p^d, r = n/d, and are at
    /// least p^n / 2 (vzGG Theorem 14.9 with step 3).
    #[test]
    fn split_with_exhaustive_counts() {
        let a3 = vec![1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1];
        let rows: Vec<(i64, i64, Vec<i64>, u64)> = vec![
            (2, 1, vec![0, 1, 1], 2),
            (2, 3, vec![1, 1, 1, 1, 1, 1, 1], 38),
            (2, 4, a3, 3240),
            (3, 1, vec![0, 2, 0, 1], 24),
            (3, 2, vec![1, 0, 1, 0, 1, 0, 1], 600),
            (3, 3, vec![2, 0, 1, 0, 1, 0, 1], 390),
            (5, 2, vec![1, 0, 0, 0, 1], 336),
            (5, 1, vec![0, 4, 0, 0, 0, 1], 3060),
            (5, 1, vec![4, 0, 1], 16),         // (x-1)(x-4)
            (7, 1, vec![2, 4, 1], 30),         // (x-1)(x-2)
            (7, 2, vec![2, 0, 3, 0, 1], 1248), // (x^2+1)(x^2+2)
        ];
        for (p, d, f, expected) in rows {
            let n = degree(&f) as u32;
            let e = if p == 2 {
                Integer::zero()
            } else {
                (&Integer::from(p).pow(d as u32) - &Integer::one()) / Integer::from(2i64)
            };
            let total = (p as u64).pow(n);
            let mut count = 0u64;
            for idx in 0..total {
                let mut v = idx;
                let a: Vec<i64> = (0..n)
                    .map(|_| {
                        let c = (v % p as u64) as i64;
                        v /= p as u64;
                        c
                    })
                    .collect();
                if let Some(g) = split_with(&trim(&a), &f, d, p, &e) {
                    // Every reported split is a proper monic divisor.
                    assert!(degree(&g) > 0 && degree(&g) < n as i64);
                    assert!(is_zero(&div_mod(&f, &g, p).1));
                    count += 1;
                }
            }
            assert_eq!(count, expected, "split count for {f:?} mod {p}, d = {d}");
            assert!(
                2 * count >= total,
                "success probability below 1/2 for {f:?} mod {p}"
            );
        }
        // Contrast: a monic sampler h = x + c (the old equal-degree stage drew
        // monic h of degree n - 1) on (x-1)(x-4) mod 5 splits for only 2 of the
        // 5 h (below 1/2), so it carries no probability bound. This is why the
        // equal-degree stage draws a uniformly. gp:
        //   F = Mod(1,5)*(x^2+4);
        //   sum(c=0,4, poldegree(gcd(lift(Mod(x+c,F)^2)-1, F)) == 1)  \\ = 2
        let f = vec![4, 0, 1];
        let e = Integer::from(2i64);
        let legacy = (0..5)
            .filter(|&c| degree(&gcd(&sub(&pow_mod(&[c, 1], &e, &f, 5), &[1], 5), &f, 5)) == 1)
            .count();
        assert_eq!(legacy, 2);
    }

    // T5: contract violations of the public entry panic with the reason.
    // gp: factormod(x^2+3*x+2, 5) = (x+1)(x+2);
    //     polisirreducible(Mod(1,3)*(x^4+x^3+2)) = 1;
    //     (x^2+1)^2*Mod(1,3) = x^4+2x^2+1; isprime(4) = 0;
    //     nextprime(3037000500) = 3037000507 (isprime = 1, above MAX_P).

    #[test]
    #[should_panic(expected = "degree dividing 1")]
    fn equal_degree_factor_rejects_reducible_block_of_degree_d() {
        // (x+1)(x+2) over F_5 with d = 2: legacy code returned it as irreducible.
        equal_degree_factor(&[2, 3, 1], 2, 5);
    }

    #[test]
    #[should_panic(expected = "does not divide 2")]
    fn equal_degree_factor_rejects_irreducible_quartic_for_d2() {
        // x^4 + x^3 + 2 is irreducible over F_3: no try can split it with d = 2.
        equal_degree_factor(&[2, 0, 0, 1, 1], 2, 3);
    }

    #[test]
    #[should_panic(expected = "not squarefree")]
    fn equal_degree_factor_rejects_non_squarefree() {
        // (x^2+1)^2 over F_3.
        equal_degree_factor(&[1, 0, 2, 0, 1], 2, 3);
    }

    #[test]
    #[should_panic(expected = "d = 2 does not divide deg f = 3")]
    fn equal_degree_factor_rejects_d_not_dividing_degree() {
        equal_degree_factor(&[0, 2, 0, 1], 2, 3);
    }

    #[test]
    #[should_panic(expected = "d = 0 must be at least 1")]
    fn equal_degree_factor_rejects_d_zero() {
        equal_degree_factor(&[0, 2, 0, 1], 0, 3);
    }

    #[test]
    #[should_panic(expected = "p = 4 is not a prime <= 3037000499")]
    fn equal_degree_factor_rejects_composite_p() {
        equal_degree_factor(&[3, 0, 1], 1, 4);
    }

    #[test]
    #[should_panic(expected = "p = 3037000507 is not a prime <= 3037000499")]
    fn equal_degree_factor_rejects_prime_above_max_p() {
        equal_degree_factor(&[1, 1], 1, 3037000507);
    }

    #[test]
    #[should_panic(expected = "f = 0 has no factorization")]
    fn equal_degree_factor_rejects_zero() {
        equal_degree_factor(&[0], 1, 5);
    }

    #[test]
    fn equal_degree_factor_constant_and_boundary_prime() {
        // A nonzero constant is the empty product.
        assert_eq!(equal_degree_factor(&[3], 1, 5), Vec::<Vec<i64>>::new());
        // x - 1 at the largest prime <= MAX_P (gp: precprime(3037000499) =
        // 3037000493): the certificate accepts it.
        assert_eq!(MAX_P, 3_037_000_499);
        assert_eq!(
            equal_degree_factor(&[3037000492, 1], 1, 3037000493),
            vec![vec![3037000492, 1]]
        );
    }

    /// T6: the internal path reports a bad block instead of looping or
    /// returning it unsplit (x^4 + x^3 + 2 is irreducible over F_3): no try
    /// splits it, and after 32 failed tries the certificate panics.
    #[test]
    #[should_panic(
        expected = "violates its contract (f has an irreducible factor whose degree does not divide 2)"
    )]
    fn edf_rec_certificate_panics_on_bad_block() {
        let mut out = Vec::new();
        edf_rec(&[2, 0, 0, 1, 1], 2, 3, &mut out);
    }

    /// The uniform draw is exactly uniform by rejection: 2^64 mod n words are
    /// rejected. Scripted check of the acceptance bound for n = 10
    /// (2^64 mod 10 = 6, so u64::MAX - 5 ..= u64::MAX are rejected).
    #[test]
    fn splitmix_below_rejection_bound() {
        let n: u64 = 10;
        let r = (u64::MAX % n + 1) % n;
        assert_eq!(r, 6);
        assert_eq!((u64::MAX - r + 1) % n, 0); // accepted range is a multiple of n
        let mut rng = SplitMix64(12345);
        for _ in 0..1000 {
            let v = rng.below(7);
            assert!((0..7).contains(&v));
        }
    }

    #[test]
    fn test_pow_mod() {
        let p = 7;
        // x^7 mod (x^2 + 1) over F_7. (Frobenius: x^p.)
        let modulus = vec![1, 0, 1];
        let r = pow_mod(&[0, 1], &Integer::from(7i64), &modulus, p);
        // x^2 ≡ -1, so x^7 = x*(x^2)^3 ≡ x*(-1)^3 = -x ≡ 6x.
        assert_eq!(r, vec![0, 6]);
    }

    // ---------------------------------------------------------------------
    // Checked entries (follow-up unit P1). References, PARI/GP 2.15:
    //   gp> Vecrev(lift(Mod(1,5)*(x+1)^2*(x^2+2)*(x^2+3)))  \\ [1,2,1,0,1,2,1]
    //   gp> factormod(x^5+x^4+x+1, 5)   \\ (x+1)(x^2+2)(x^2+3)
    //   gp> factormod(x-4, 5)           \\ x+1
    //   gp> factormod(x^2-3, 5)         \\ x^2+2, irreducible
    //   gp> factormod(5*x^2+x+1, 5)     \\ x+1
    //   gp> factormod(5*x+1, 5)         \\ empty: a nonzero constant
    //   gp> [isprime(2047), isprime(1373653), isprime(25326001)]  \\ [0,0,0]
    //   gp> [precprime(3037000499), nextprime(3037000500)]
    //       \\ [3037000493, 3037000507]
    // 2047 = 23*89 is a strong pseudoprime to base 2, 1373653 = 829*1657 to
    // the bases 2 and 3, 25326001 = 2251*11251 to 2, 3 and 5, and
    // 3215031751 = 151*751*28351 to 2, 3, 5 and 7 (gp: a strong test by
    // hand, `Mod(a, n)^d` with n - 1 = 2^s d).
    // ---------------------------------------------------------------------

    #[test]
    #[should_panic(expected = "fp_factor::distinct_degree_factor: f is not squarefree")]
    fn ddf_rejects_non_squarefree() {
        distinct_degree_factor(&[1, 2, 1], 5);
    }

    /// (x+1)^2 (x^2+2)(x^2+3) mod 5: the old loop labelled the block
    /// x^5+x^4+x+1 = (x+1)(x^2+2)(x^2+3) with degree 2.
    #[test]
    #[should_panic(expected = "fp_factor::distinct_degree_factor: f is not squarefree")]
    fn ddf_rejects_hidden_repeated_factor() {
        distinct_degree_factor(&[1, 2, 1, 0, 1, 2, 1], 5);
    }

    /// x^2 over F_2 has derivative 0.
    #[test]
    #[should_panic(expected = "fp_factor::distinct_degree_factor: f is not squarefree")]
    fn ddf_rejects_zero_derivative() {
        distinct_degree_factor(&[0, 0, 1], 2);
    }

    #[test]
    #[should_panic(expected = "fp_factor::distinct_degree_factor: f = 0 is not squarefree")]
    fn ddf_rejects_zero() {
        distinct_degree_factor(&[0], 5);
    }

    #[test]
    #[should_panic(
        expected = "fp_factor::distinct_degree_factor: p = 15 is not a prime <= 3037000499"
    )]
    fn ddf_rejects_composite_p() {
        distinct_degree_factor(&[1, 0, 1], 15);
    }

    #[test]
    fn ddf_constant_and_squarefree() {
        assert_eq!(distinct_degree_factor(&[3], 5), Vec::<(i64, Vec<i64>)>::new());
        // x^5+x^4+x+1 = (x+1)(x^4+1), x^4+1 = (x^2+2)(x^2+3) mod 5.
        assert_eq!(
            distinct_degree_factor(&[1, 1, 0, 0, 1, 1], 5),
            vec![(1, vec![1, 1]), (2, vec![1, 0, 0, 0, 1])]
        );
    }

    #[test]
    fn ddf_reads_coefficients_mod_p() {
        // x - 4 = x + 1 mod 5.
        assert_eq!(distinct_degree_factor(&[-4, 1], 5), vec![(1, vec![1, 1])]);
    }

    #[test]
    #[should_panic(expected = "fp_factor::factor: p = 15 is not a prime <= 3037000499")]
    fn factor_rejects_composite_p() {
        factor(&[1, 0, 1], 15);
    }

    #[test]
    #[should_panic(expected = "fp_factor::factor: p = 1 is not a prime <= 3037000499")]
    fn factor_rejects_p_one() {
        factor(&[1, 0, 1], 1);
    }

    #[test]
    #[should_panic(expected = "fp_factor::factor: p = 3037000507 is not a prime <= 3037000499")]
    fn factor_rejects_large_prime() {
        factor(&[1, 1], 3037000507);
    }

    #[test]
    #[should_panic(
        expected = "fp_factor::factor_with_multiplicity: p = 21 is not a prime <= 3037000499"
    )]
    fn factor_with_multiplicity_rejects_composite_p() {
        factor_with_multiplicity(&[1, 0, 1], 21);
    }

    #[test]
    #[should_panic(expected = "fp_factor::squarefree_factor: p = 15 is not a prime <= 3037000499")]
    fn squarefree_factor_rejects_composite_p() {
        squarefree_factor(&[1, 0, 1], 15);
    }

    /// 5 + 10x is 0 mod 5: an empty list would read as "f is a unit".
    #[test]
    #[should_panic(expected = "fp_factor::factor: f = 0 has no factorization")]
    fn factor_rejects_zero_mod_p() {
        factor(&[5, 10], 5);
    }

    #[test]
    #[should_panic(expected = "fp_factor::factor_with_multiplicity: f = 0 has no factorization")]
    fn factor_with_multiplicity_rejects_zero() {
        factor_with_multiplicity(&[0], 7);
    }

    #[test]
    #[should_panic(expected = "fp_factor::squarefree_factor: f = 0 has no factorization")]
    fn squarefree_factor_rejects_zero_mod_p() {
        squarefree_factor(&[3, 0, 3], 3);
    }

    #[test]
    fn factor_reads_coefficients_mod_p() {
        // x^2 - 3 = x^2 + 2 mod 5 (irreducible); 6x^2 + 7 = x^2 + 2 mod 5.
        assert_eq!(factor(&[-3, 0, 1], 5), vec![vec![2, 0, 1]]);
        assert_eq!(factor(&[7, 0, 6], 5), vec![vec![2, 0, 1]]);
    }

    #[test]
    fn factor_reduces_leading_coefficient() {
        // 5x^2 + x + 1 = x + 1 mod 5; 5x + 1 = 1 mod 5 (a nonzero constant).
        assert_eq!(factor(&[1, 1, 5], 5), vec![vec![1, 1]]);
        assert_eq!(factor(&[1, 5], 5), Vec::<Vec<i64>>::new());
    }

    #[test]
    fn factor_with_multiplicity_reduces_leading_coefficient() {
        assert_eq!(factor_with_multiplicity(&[1, 1, 5], 5), vec![(vec![1, 1], 1)]);
    }

    #[test]
    #[should_panic(expected = "fp_factor::equal_degree_factor: f = 0 has no factorization")]
    fn equal_degree_factor_rejects_zero_mod_p() {
        equal_degree_factor(&[5], 1, 5);
    }

    #[test]
    fn check_modulus_boundary() {
        assert_eq!(check_modulus(2), Ok(()));
        assert_eq!(check_modulus(3037000493), Ok(()));
        for p in [0, 1, 4, 15, -7, 2047, 1373653, 25326001, 3037000507, 3215031751] {
            assert_eq!(
                check_modulus(p),
                Err(format!("p = {p} is not a prime <= 3037000499")),
                "p = {p}"
            );
        }
    }

    // Equal-degree certificate on composite d (sparse Frobenius ladder).
    // gp: Vecrev(lift(Mod(1,2)*(x^12+x^3+1)*(x^12+x^6+x^4+x+1)));
    //     factormod(%, 2) lists x^12+x^3+1 first, then x^12+x^6+x^4+x+1;
    //     polisirreducible(Mod(1,2)*g) = 1 for g = x^12+x^3+1,
    //     x^12+x^6+x^4+x+1, x^6+x+1, x^4+x+1, x^2+x+1;
    //     Vecrev(lift(Mod(1,2)*(x^2+x+1)*(x^4+x+1)*(x^6+x+1)))
    //       = [1,1,0,1,0,0,1,1,0,1,1,1,1];
    //     factormod(x^6+x^5+x^4+x^3+x^2+x+1, 2) = (x^3+x+1)(x^3+x^2+1).

    #[test]
    fn certificate_composite_d_accepts() {
        let f = [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1];
        assert_eq!(
            equal_degree_factor(&f, 12, 2),
            vec![
                vec![1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                vec![1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
            ]
        );
    }

    #[test]
    #[should_panic(
        expected = "fp_factor::equal_degree_factor: f has an irreducible factor of degree dividing 6"
    )]
    fn certificate_composite_d_rejects_divisor_degree() {
        equal_degree_factor(&[1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1], 12, 2);
    }

    #[test]
    #[should_panic(
        expected = "fp_factor::equal_degree_factor: f has an irreducible factor of degree dividing 3"
    )]
    fn certificate_d6_two_cubics() {
        equal_degree_factor(&[1, 1, 1, 1, 1, 1, 1], 6, 2);
    }

    #[test]
    fn certificate_d6_irreducible() {
        assert_eq!(
            equal_degree_factor(&[1, 1, 0, 0, 0, 0, 1], 6, 2),
            vec![vec![1, 1, 0, 0, 0, 0, 1]]
        );
    }

    #[test]
    fn prime_divisors_small() {
        assert_eq!(prime_divisors(1), Vec::<i64>::new());
        assert_eq!(prime_divisors(12), vec![2, 3]);
        assert_eq!(prime_divisors(60), vec![2, 3, 5]);
        assert_eq!(prime_divisors(97), vec![97]);
    }

    #[test]
    fn factor_order_compares_degree_then_top_coefficients() {
        assert_eq!(factor_order(&[1, 1], &[0, 0, 1]), Ordering::Less);
        assert_eq!(factor_order(&[1, 1, 0, 1], &[1, 0, 1, 1]), Ordering::Less);
        assert_eq!(factor_order(&[2, 1, 0], &[2, 1]), Ordering::Equal);
    }
}
