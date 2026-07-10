//! Resultant and discriminant over ℤ[x] by a modular (CRT) method.
//!
//! The generic `UnivariatePolynomial::discriminant` handles only degrees 2–3 and
//! `resultant` forms the Sylvester matrix and expands its determinant cofactor-wise
//! (O(n!)) — unusable at degree 24. Here `Res(f, g)` is computed modulo a sequence
//! of primes (each via the Euclidean resultant recurrence over `F_p`) and
//! reconstructed by CRT once the modulus exceeds twice the Hadamard bound. This is
//! fast and exact for the degree-24 IGP24 polynomials, and underlies the field
//! discriminant / small-discriminant work (Phase 3).

use crate::zx;
use rustmath_integers::Integer;

/// ceil(‖f‖₂) = ceil(sqrt(Σ aᵢ²)).
fn l2_norm_ceil(f: &[Integer]) -> Integer {
    let mut s = Integer::zero();
    for c in f {
        s = s + c.clone() * c.clone();
    }
    let mut r = s.sqrt().unwrap_or_else(|_| Integer::zero());
    if r.clone() * r.clone() < s {
        r = r + Integer::one();
    }
    r
}

fn is_prime_i64(n: i64) -> bool {
    if n < 2 {
        return false;
    }
    let mut d = 2i64;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 1;
    }
    true
}

fn mod_pow_i64(base: i64, mut exp: i64, p: i64) -> i64 {
    let mut acc: i128 = 1;
    let mut b = (((base % p) + p) % p) as i128;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = acc * b % p as i128;
        }
        b = b * b % p as i128;
        exp >>= 1;
    }
    acc as i64
}

fn mod_inv_i64(a: i64, p: i64) -> i64 {
    // p prime, a not ≡ 0
    mod_pow_i64(((a % p) + p) % p, p - 2, p)
}

/// Reduce an integer polynomial mod p to `Vec<i64>` in `[0, p)`, trimmed.
fn reduce_mod_p(f: &[Integer], p: i64) -> Vec<i64> {
    let pi = Integer::from(p);
    let mut v: Vec<i64> = f
        .iter()
        .map(|c| {
            let r = (c.clone() % pi.clone()).to_i64();
            ((r % p) + p) % p
        })
        .collect();
    while v.len() > 1 && *v.last().unwrap() == 0 {
        v.pop();
    }
    v
}

fn deg_i64(p: &[i64]) -> i64 {
    let mut n = p.len();
    while n > 0 && p[n - 1] == 0 {
        n -= 1;
    }
    n as i64 - 1
}

/// Remainder of `a` by `b` over `F_p` (ordinary polynomial division, `b ≠ 0`).
fn poly_rem_fp(a: &[i64], b: &[i64], p: i64) -> Vec<i64> {
    let db = deg_i64(b);
    let lcb_inv = mod_inv_i64(b[b.len() - 1], p);
    let mut r: Vec<i64> = a.to_vec();
    while deg_i64(&r) >= db && deg_i64(&r) >= 0 {
        let dr = deg_i64(&r) as usize;
        let coeff = (r[dr] as i128 * lcb_inv as i128 % p as i128) as i64;
        let shift = dr - db as usize;
        for j in 0..b.len() {
            let idx = j + shift;
            let sub = coeff as i128 * b[j] as i128 % p as i128;
            r[idx] = (((r[idx] as i128 - sub) % p as i128 + p as i128) % p as i128) as i64;
        }
        while r.len() > 1 && *r.last().unwrap() == 0 {
            r.pop();
        }
        if deg_i64(&r) < db {
            break;
        }
    }
    r
}

/// Resultant of `a, b` over `F_p` via the Euclidean recurrence
/// `Res(a,b) = (−1)^{deg a · deg b} lc(b)^{deg a − deg r} Res(b, r)`.
fn resultant_mod_p(a0: &[i64], b0: &[i64], p: i64) -> i64 {
    let mut a = a0.to_vec();
    let mut b = b0.to_vec();
    let mut sign: i64 = 1;
    let mut res: i128 = 1;
    if deg_i64(&a) < deg_i64(&b) {
        if (deg_i64(&a) * deg_i64(&b)) & 1 == 1 {
            sign = -sign;
        }
        std::mem::swap(&mut a, &mut b);
    }
    loop {
        let db = deg_i64(&b);
        if db < 0 {
            return 0; // b == 0: nontrivial gcd ⇒ resultant 0
        }
        let da = deg_i64(&a);
        if db == 0 {
            res = res * mod_pow_i64(b[0], da, p) as i128 % p as i128;
            break;
        }
        let r = poly_rem_fp(&a, &b, p);
        let dr = deg_i64(&r);
        if dr < 0 {
            return 0;
        }
        if (da * db) & 1 == 1 {
            sign = -sign;
        }
        let lcb = b[b.len() - 1];
        res = res * mod_pow_i64(lcb, da - dr, p) as i128 % p as i128;
        a = b;
        b = r;
    }
    let out = (sign as i128 * res % p as i128 + p as i128) % p as i128;
    out as i64
}

/// Resultant `Res(f, g) ∈ ℤ` via CRT over primes (exact). Returns 0 if either is
/// the zero polynomial.
pub fn resultant(f: &[Integer], g: &[Integer]) -> Integer {
    let f = zx::trim(f);
    let g = zx::trim(g);
    if zx::is_zero(&f) || zx::is_zero(&g) {
        return Integer::zero();
    }
    let (df, dg) = (zx::degree(&f), zx::degree(&g));
    if df == 0 && dg == 0 {
        return Integer::one();
    }
    // Hadamard bound: |Res(f,g)| ≤ ‖f‖₂^{deg g} · ‖g‖₂^{deg f}.
    let bound = l2_norm_ceil(&f).pow(dg.max(0) as u32) * l2_norm_ceil(&g).pow(df.max(0) as u32);
    let limit = bound * Integer::from(2) + Integer::one();

    let lc_f = f[f.len() - 1].abs();
    let lc_g = g[g.len() - 1].abs();

    let mut acc = Integer::zero(); // residue mod `modulus`
    let mut modulus = Integer::one();
    let mut p: i64 = 1 << 20; // ~30-bit primes keep i128 products safe
    while modulus <= limit {
        p += 1;
        if !is_prime_i64(p) {
            continue;
        }
        // skip primes dividing a leading coefficient (degree would drop)
        if (lc_f.clone() % Integer::from(p)).is_zero() || (lc_g.clone() % Integer::from(p)).is_zero()
        {
            continue;
        }
        let rp = resultant_mod_p(&reduce_mod_p(&f, p), &reduce_mod_p(&g, p), p);
        // CRT-combine (acc mod modulus) with (rp mod p)
        let pi = Integer::from(p);
        let m_mod_p = (modulus.clone() % pi.clone()).to_i64();
        let inv = mod_inv_i64(((m_mod_p % p) + p) % p, p);
        let acc_mod_p = (acc.clone() % pi.clone()).to_i64();
        let mut diff = (rp - acc_mod_p) % p;
        diff = ((diff % p) + p) % p;
        let t = (diff as i128 * inv as i128 % p as i128) as i64;
        acc = acc + modulus.clone() * Integer::from(t);
        modulus = modulus * pi;
    }
    // balanced representative in (−modulus/2, modulus/2]
    let half = modulus.clone() / Integer::from(2);
    if acc > half {
        acc = acc - modulus;
    }
    acc
}

/// Discriminant of `f ∈ ℤ[x]`:
/// `disc(f) = (−1)^{n(n−1)/2} · Res(f, f') / lc(f)`.
pub fn discriminant(f: &[Integer]) -> Integer {
    let f = zx::trim(f);
    let n = zx::degree(&f);
    if n < 1 {
        return Integer::zero();
    }
    let fp = zx::derivative(&f);
    let res = resultant(&f, &fp);
    let lc = f[f.len() - 1].clone();
    let (q, _r) = (res.clone() / lc.clone(), res.clone() % lc.clone());
    let signed = if ((n * (n - 1) / 2) & 1) == 1 { -q } else { q };
    signed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(cs: &[i64]) -> Vec<Integer> {
        cs.iter().map(|&c| Integer::from(c)).collect()
    }

    #[test]
    fn test_disc_quadratic() {
        // x^2 + bx + c → b^2 - 4c. x^2 - 5x + 6 → 25 - 24 = 1
        assert_eq!(discriminant(&p(&[6, -5, 1])), Integer::from(1));
        // x^2 + 1 → -4
        assert_eq!(discriminant(&p(&[1, 0, 1])), Integer::from(-4));
    }

    #[test]
    fn test_disc_cubic() {
        // x^3 + px + q → -4p^3 - 27q^2.  x^3 - 2 → -4·0 - 27·4 = -108
        assert_eq!(discriminant(&p(&[-2, 0, 0, 1])), Integer::from(-108));
        // x^3 - x → disc 4 (roots -1,0,1; disc = 4)
        assert_eq!(discriminant(&p(&[0, -1, 0, 1])), Integer::from(4));
    }

    #[test]
    fn test_resultant_coprime() {
        // Res(x^2+1, x^2+x+1): both irreducible, no common root → nonzero
        let r = resultant(&p(&[1, 0, 1]), &p(&[1, 1, 1]));
        assert_eq!(r, Integer::from(1));
    }

    #[test]
    fn test_resultant_common_factor_zero() {
        // share (x-1): Res = 0
        let f = zx::mul(&p(&[-1, 1]), &p(&[1, 1]));
        let g = zx::mul(&p(&[-1, 1]), &p(&[2, 1]));
        assert_eq!(resultant(&f, &g), Integer::zero());
    }

    #[test]
    fn test_disc_cyclotomic5() {
        // Φ_5 = x^4+x^3+x^2+x+1, disc = 125
        assert_eq!(discriminant(&p(&[1, 1, 1, 1, 1])), Integer::from(125));
    }
}
