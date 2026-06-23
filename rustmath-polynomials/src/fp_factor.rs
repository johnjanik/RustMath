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

/// Squarefree part of `f` — the radical, `f / gcd(f, f')`. The
/// characteristic-`p` edge case (where `f' = 0` because all coefficients are
/// `p`-th powers) returns the input unchanged.
pub fn squarefree_factor(f: &[i64], p: i64) -> Vec<i64> {
    let f_m = make_monic(f, p);
    if degree(&f_m) <= 0 {
        return f_m;
    }
    let derivative = derivative_of(&f_m, p);
    if is_zero(&derivative) {
        return f_m;
    }
    let g = gcd(&f_m, &derivative, p);
    if degree(&g) == 0 {
        return f_m;
    }
    make_monic(&div_mod(&f_m, &g, p).0, p)
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

/// Full irreducible factorization of `f` in `F_p[x]`. Squarefree
/// factorization is performed first so repeated factors are coalesced
/// (multiplicities are *not* tracked — every returned factor is distinct and
/// monic).
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
