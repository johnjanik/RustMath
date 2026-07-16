//! Factorization over `F_p[x]` (distinct-degree + Cantor–Zassenhaus).
//!
//! Ported from the `FpPoly` engine of the p-adelic calculator. Polynomials are
//! little-endian `Vec<i64>` arrays with every coefficient reduced to `[0, p)`.
//! The prime `p` is passed as a parameter rather than stored — the caller is
//! responsible for using a consistent prime across operations.
//!
//! This is the mod-`p` companion of [`crate::zp_hensel`]: factor `f mod p` here,
//! then Hensel-lift the factors to `Z/p^N Z` there.
//!
//! Overflow contract: products `a * b mod p` are formed in `i64`, so the caller
//! must ensure `p * p <= i64::MAX` (i.e. `p < 2^31`). All screen primes are far
//! below that bound.

use rustmath_integers::Integer;

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

/// Radical of `f` over `F_p`: the product of the *distinct* monic irreducible
/// factors of `f`, each exactly once (monic; constants map to `1`-like monic
/// constants). This is what distinct-degree factorization needs as input.
///
/// Characteristic-`p` subtleties handled correctly (both were bugs before):
/// * `f' = 0` means `f(x) = g(x^p)`; over the prime field `g`'s coefficients
///   are `f`'s directly ([`pth_root`]), and `rad(f) = rad(g)` — recurse.
/// * Even when `f' != 0`, factors whose multiplicity is divisible by `p`
///   survive inside `gcd(f, f')` at *full* multiplicity, so `f / gcd(f, f')`
///   alone misses them (e.g. `(x+1)^3 (x+2)` over `F_3` lost `x+1`). The
///   `p`-power part is split off and recursed on.
pub fn squarefree_factor(f: &[i64], p: i64) -> Vec<i64> {
    let f_m = make_monic(f, p);
    if degree(&f_m) <= 0 {
        return f_m;
    }
    let derivative = derivative_of(&f_m, p);
    if is_zero(&derivative) {
        // f = g(x^p): same distinct irreducible factors as g.
        return squarefree_factor(&pth_root(&f_m, p), p);
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
    mul(&w, &squarefree_factor(&c, p), p)
}

/// Distinct-degree factorization of a squarefree polynomial. Returns pairs
/// `(d, g_d)` where `g_d` is the product of all irreducible factors of `f` of
/// degree exactly `d`.
pub fn distinct_degree_factor(f: &[i64], p: i64) -> Vec<(i64, Vec<i64>)> {
    let mut f_rem = make_monic(f, p);
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

/// Small deterministic PRNG (SplitMix64-style) used to sample the random
/// polynomials in Cantor–Zassenhaus. Deterministic so factorization is
/// reproducible across runs; the algorithm tries many samples and converges.
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: i64) -> i64 {
        (self.next_u64() % (n as u64)) as i64
    }
}

fn random_monic_polynomial(d: i64, p: i64, rng: &mut Lcg) -> Vec<i64> {
    assert!(d >= 0);
    if d == 0 {
        return vec![1];
    }
    let d = d as usize;
    let mut coeffs = vec![0i64; d + 1];
    for c in coeffs.iter_mut().take(d) {
        *c = rng.below(p);
    }
    coeffs[d] = 1;
    coeffs
}

/// Equal-degree factorization (Cantor–Zassenhaus). Splits a polynomial `f`
/// whose irreducible factors all have degree `d` into those individual
/// factors. Returns `[f]` if `f` is already irreducible. For `p = 2` the
/// trace-form split is used; for odd `p`, the standard `h^((p^d-1)/2) - 1`.
pub fn equal_degree_factor(f: &[i64], d: i64, p: i64) -> Vec<Vec<i64>> {
    let f_m = make_monic(f, p);
    let n = degree(&f_m);
    if n <= 0 || n == d {
        return vec![f_m];
    }

    // Seed the PRNG deterministically from the input so results are stable.
    let mut seed: u64 = 0x243F_6A88_85A3_08D3 ^ (d as u64).wrapping_mul(0x100_0193);
    seed ^= (p as u64).wrapping_mul(0x85EB_CA6B);
    for (i, &c) in f_m.iter().enumerate() {
        seed = seed
            .wrapping_mul(0x0100_0000_01B3)
            .wrapping_add((c as u64) ^ (i as u64));
    }
    let mut rng = Lcg(seed | 1);

    for _ in 0..200 {
        let h = random_monic_polynomial((n - 1).max(1), p, &mut rng);
        let t: Vec<i64> = if p == 2 {
            // Trace: T(h) = h + h^2 + h^4 + ... + h^(2^(d-1)) mod f.
            let mut sum = h.clone();
            let mut current = h.clone();
            for _ in 1..d {
                current = div_mod(&mul(&current, &current, p), &f_m, p).1;
                sum = div_mod(&add(&sum, &current, p), &f_m, p).1;
            }
            sum
        } else {
            // exp = (p^d - 1) / 2.
            let exp = (&Integer::from(p).pow(d as u32) - &Integer::from(1i64)) / Integer::from(2i64);
            let pow_h = pow_mod(&h, &exp, &f_m, p);
            sub(&pow_h, &[1], p)
        };
        let g = gcd(&t, &f_m, p);
        let dg = degree(&g);
        if dg > 0 && dg < n {
            let g1 = make_monic(&g, p);
            let g2 = make_monic(&div_mod(&f_m, &g1, p).0, p);
            let mut out = equal_degree_factor(&g1, d, p);
            out.extend(equal_degree_factor(&g2, d, p));
            return out;
        }
    }
    // Should not reach for valid inputs; return f as a single block.
    vec![f_m]
}

/// Full irreducible factorization of `f` in `F_p[x]`. The radical
/// ([`squarefree_factor`]) is computed first so repeated factors are coalesced
/// (multiplicities are *not* tracked — every returned factor is distinct and
/// monic; use [`factor_with_multiplicity`] to recover exponents). Correct on
/// inseparable inputs (`x^2 + 1` over `F_2`, `x^5 - 2` over `F_5`, ...) since
/// the radical handles the characteristic-`p` cases.
pub fn factor(f: &[i64], p: i64) -> Vec<Vec<i64>> {
    let f_m = make_monic(f, p);
    if degree(&f_m) <= 0 {
        return Vec::new();
    }
    let squarefree = squarefree_factor(&f_m, p);
    let mut out: Vec<Vec<i64>> = Vec::new();
    for (d, gd) in distinct_degree_factor(&squarefree, p) {
        for piece in equal_degree_factor(&gd, d, p) {
            out.push(make_monic(&piece, p));
        }
    }
    out
}

/// Irreducible factorization of `f` in `F_p[x]` *with multiplicities*:
/// returns `(g_i, e_i)` pairs with each `g_i` monic irreducible and distinct,
/// such that `prod g_i^{e_i}` equals `f` made monic. Multiplicities are found
/// by repeated exact division of `f` by each distinct factor from [`factor`].
pub fn factor_with_multiplicity(f: &[i64], p: i64) -> Vec<(Vec<i64>, u32)> {
    let f_m = make_monic(f, p);
    if degree(&f_m) <= 0 {
        return Vec::new();
    }
    factor(&f_m, p)
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

    #[test]
    fn test_pow_mod() {
        let p = 7;
        // x^7 mod (x^2 + 1) over F_7. (Frobenius: x^p.)
        let modulus = vec![1, 0, 1];
        let r = pow_mod(&[0, 1], &Integer::from(7i64), &modulus, p);
        // x^2 ≡ -1, so x^7 = x*(x^2)^3 ≡ x*(-1)^3 = -x ≡ 6x.
        assert_eq!(r, vec![0, 6]);
    }
}
