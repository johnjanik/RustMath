//! Hensel lifting over `Z` / `Z/p^N Z`.
//!
//! Ported from the `ZpPoly` engine of the p-adelic calculator. Polynomials are
//! little-endian `Vec<Integer>` arrays. Given a factorization
//! `f ≡ g₀ · h₀ (mod p)` with coprime monic factors in `F_p[x]` (produced by
//! [`crate::fp_factor`]), this lifts to `f ≡ G · H (mod p^N)` for any target
//! precision `N` via the linear Hensel construction (one prime-power of
//! precision per iteration, Bezout coefficients computed once at mod `p`).
//!
//! This replaces RustMath's earlier broken Hensel routine; see
//! [`crate::factorization::hensel_lift`], which now delegates here.

use crate::fp_factor;
use rustmath_integers::Integer;

#[inline]
fn zero() -> Integer {
    Integer::from(0i64)
}

/// Strip trailing zeros, keeping at least one coefficient.
pub fn trim(p: &[Integer]) -> Vec<Integer> {
    let mut c = p.to_vec();
    while c.len() > 1 && c.last().unwrap().is_zero() {
        c.pop();
    }
    if c.is_empty() {
        c.push(zero());
    }
    c
}

/// Polynomial degree. Returns `-1` for the zero polynomial.
pub fn degree(p: &[Integer]) -> i64 {
    let t = trim(p);
    if t.len() == 1 && t[0].is_zero() {
        -1
    } else {
        (t.len() - 1) as i64
    }
}

/// `a + b` over `Z`.
pub fn add(a: &[Integer], b: &[Integer]) -> Vec<Integer> {
    let n = a.len().max(b.len());
    let mut out = vec![zero(); n];
    for i in 0..a.len() {
        out[i] = &out[i] + &a[i];
    }
    for i in 0..b.len() {
        out[i] = &out[i] + &b[i];
    }
    trim(&out)
}

/// `a - b` over `Z`.
pub fn sub(a: &[Integer], b: &[Integer]) -> Vec<Integer> {
    let n = a.len().max(b.len());
    let mut out = vec![zero(); n];
    for i in 0..a.len() {
        out[i] = &out[i] + &a[i];
    }
    for i in 0..b.len() {
        out[i] = &out[i] - &b[i];
    }
    trim(&out)
}

/// `a * b` over `Z`, schoolbook.
pub fn mul(a: &[Integer], b: &[Integer]) -> Vec<Integer> {
    if a.is_empty() || b.is_empty() {
        return vec![zero()];
    }
    if (a.len() == 1 && a[0].is_zero()) || (b.len() == 1 && b[0].is_zero()) {
        return vec![zero()];
    }
    let mut out = vec![zero(); a.len() + b.len() - 1];
    for i in 0..a.len() {
        if a[i].is_zero() {
            continue;
        }
        for j in 0..b.len() {
            if b[j].is_zero() {
                continue;
            }
            out[i + j] = &out[i + j] + &(&a[i] * &b[j]);
        }
    }
    trim(&out)
}

/// Reduce every coefficient mod `m` into the canonical range `[0, m)`.
pub fn reduce(a: &[Integer], m: &Integer) -> Vec<Integer> {
    assert!(m > &zero(), "zp_hensel::reduce: modulus must be positive");
    let reduced: Vec<Integer> = a
        .iter()
        .map(|c| {
            let r = c % m;
            &(&r + m) % m
        })
        .collect();
    trim(&reduced)
}

/// Divide every coefficient by `d` (must be exact). Used to extract the
/// Hensel correction `δ = (f - G·H) / p^k`.
pub fn divide_coefficients(a: &[Integer], d: &Integer) -> Vec<Integer> {
    assert!(!d.is_zero(), "zp_hensel::divide_coefficients: divisor must be non-zero");
    let divided: Vec<Integer> = a.iter().map(|c| c / d).collect();
    trim(&divided)
}

/// Lift an `F_p[x]` polynomial (`Vec<i64>`) into `Vec<Integer>`.
pub fn lift(a: &[i64]) -> Vec<Integer> {
    a.iter().map(|&c| Integer::from(c)).collect()
}

/// Reduce an integer polynomial mod `p` to an `F_p[x]` polynomial (`Vec<i64>`)
/// with coefficients in `[0, p)`.
fn to_fp(a: &[Integer], p: i64) -> Vec<i64> {
    let pm = Integer::from(p);
    let v: Vec<i64> = a
        .iter()
        .map(|c| {
            let r = c % &pm;
            (&(&r + &pm) % &pm).to_i64()
        })
        .collect();
    fp_factor::trim(&v)
}

/// Linear Hensel lift. Given `f ∈ Z[x]` with `f ≡ g₀ · h₀ (mod p)` for coprime
/// monic factors `g₀, h₀ ∈ F_p[x]`, return `(G, H)` with
///
/// ```text
/// f ≡ G · H (mod p^N),    G ≡ g₀ (mod p),    H ≡ h₀ (mod p)
/// ```
///
/// `G` is monic if `g₀` is. Returns `None` if `g₀, h₀` are not coprime in
/// `F_p[x]` (Hensel does not apply) or if `f mod p ≠ g₀ · h₀`.
///
/// `n` is the target precision (the power of `p`); `n >= 1`.
pub fn hensel_lift(
    f: &[Integer],
    g0: &[i64],
    h0: &[i64],
    p: i64,
    n: u32,
) -> Option<(Vec<Integer>, Vec<Integer>)> {
    assert!(n >= 1, "zp_hensel::hensel_lift: precision must be >= 1");
    let p_big = Integer::from(p);

    // Sanity check: f mod p == g0 · h0 ?
    let f_mod_p = to_fp(f, p);
    let prod_mod_p = fp_factor::mul(g0, h0, p);
    if fp_factor::trim(&f_mod_p) != fp_factor::trim(&prod_mod_p) {
        return None;
    }

    // Bezout: s·g₀ + t·h₀ = 1 in F_p[x]. Coprime iff the gcd is the constant 1.
    let (g, s, t) = fp_factor::extended_gcd(g0, h0, p);
    if g != vec![1i64] {
        return None;
    }

    let mut big_g = lift(g0);
    let mut big_h = lift(h0);
    let mut p_pow = p_big.clone(); // p^1 currently
    for _ in 1..n {
        // δ = (f - G·H) / p^k
        let prod = mul(&big_g, &big_h);
        let diff = sub(f, &prod);
        let delta = divide_coefficients(&diff, &p_pow);
        // Reduce δ mod p and operate as an F_p polynomial.
        let delta_mod = to_fp(&delta, p);
        // A = (t · δ) mod g₀;  B = (s · δ) mod h₀, both in F_p[x].
        let a = fp_factor::div_mod(&fp_factor::mul(&t, &delta_mod, p), g0, p).1;
        let b = fp_factor::div_mod(&fp_factor::mul(&s, &delta_mod, p), h0, p).1;
        // Update G, H by adding p^k · {A, B}.
        big_g = add(&big_g, &mul(&lift(&a), &[p_pow.clone()]));
        big_h = add(&big_h, &mul(&lift(&b), &[p_pow.clone()]));
        p_pow = &p_pow * &p_big;
    }

    let p_n = p_big.pow(n);
    Some((reduce(&big_g, &p_n), reduce(&big_h, &p_n)))
}

/// Iteratively lift a list of pairwise-coprime mod-`p` factors of `f` to
/// `mod p^N`, returned in the same order. Peels off one factor at a time:
/// lift `f` against `(g₀, ∏ rest)`, then recurse on the lifted rest-product.
pub fn hensel_lift_all(
    f: &[Integer],
    factors_mod_p: &[Vec<i64>],
    p: i64,
    n: u32,
) -> Option<Vec<Vec<Integer>>> {
    if factors_mod_p.is_empty() {
        return None;
    }
    if factors_mod_p.len() == 1 {
        let p_n = Integer::from(p).pow(n);
        return Some(vec![reduce(f, &p_n)]);
    }
    let g0 = &factors_mod_p[0];
    let mut h_product: Vec<i64> = vec![1];
    for factor in factors_mod_p.iter().skip(1) {
        h_product = fp_factor::mul(&h_product, factor, p);
    }
    let (big_g, big_h) = hensel_lift(f, g0, &h_product, p, n)?;
    let rest = &factors_mod_p[1..];
    let lifted = hensel_lift_all(&big_h, rest, p, n)?;
    let mut out = vec![big_g];
    out.extend(lifted);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fp_factor;

    fn ints(v: &[i64]) -> Vec<Integer> {
        v.iter().map(|&c| Integer::from(c)).collect()
    }

    // Reduce a Z[x] polynomial mod m into [0, m) for comparison.
    fn reduce_vec(a: &[Integer], m: i64) -> Vec<i64> {
        let mm = Integer::from(m);
        let v: Vec<i64> = a
            .iter()
            .map(|c| {
                let r = c % &mm;
                (&(&r + &mm) % &mm).to_i64()
            })
            .collect();
        fp_factor::trim(&v)
    }

    #[test]
    fn test_hensel_lift_quadratic_p25() {
        // f = x^2 + 1 ≡ (x+2)(x+3) (mod 5). Lift to 5^2 = 25.
        let f = ints(&[1, 0, 1]);
        let (g, h) = hensel_lift(&f, &[2, 1], &[3, 1], 5, 2).unwrap();
        let prod = mul(&g, &h);
        // g·h ≡ f (mod 25)
        assert_eq!(reduce_vec(&prod, 25), reduce_vec(&f, 25));
        // factors reduce to the originals mod 5
        assert_eq!(reduce_vec(&g, 5), vec![2, 1]);
        assert_eq!(reduce_vec(&h, 5), vec![3, 1]);
    }

    #[test]
    fn test_hensel_lift_high_precision() {
        // x^2 - 2 over Z_7: 2 is a QR mod 7 (3^2=9≡2, 4^2=16≡2), roots ±3.
        // x - 3 and x - 4 = x + 3, x + 4 mod 7.
        let f = ints(&[-2, 0, 1]);
        // mod 7: f = x^2 + 5 = (x+3)(x+4)? (x+3)(x+4)=x^2+7x+12 = x^2+0x+5 ✓
        let n = 6u32;
        let (g, h) = hensel_lift(&f, &[3, 1], &[4, 1], 7, n).unwrap();
        let modulus = 7i64.pow(n);
        let prod = mul(&g, &h);
        assert_eq!(reduce_vec(&prod, modulus), reduce_vec(&f, modulus));
        // The lifted linear root should square to 2 mod 7^n.
        // g = x + c with c the lift of 3; (-c)^2 ≡ 2 (mod 7^n).
        let c = g[0].clone();
        let m = Integer::from(modulus);
        let c2 = &(&c * &c) % &m;
        let two = Integer::from(2i64);
        let two_m = &(&c2 - &two) % &m;
        assert!(two_m.is_zero(), "root^2 != 2 mod 7^{}", n);
    }

    #[test]
    fn test_hensel_lift_rejects_non_coprime() {
        // g0 = h0 = x  → not coprime. f = x^2.
        let f = ints(&[0, 0, 1]);
        assert!(hensel_lift(&f, &[0, 1], &[0, 1], 5, 3).is_none());
    }

    #[test]
    fn test_hensel_lift_rejects_wrong_factorization() {
        // f mod 5 != g0·h0.
        let f = ints(&[1, 0, 1]); // x^2 + 1
        // (x+1)(x+1) = x^2 + 2x + 1 != x^2 + 1 mod 5
        assert!(hensel_lift(&f, &[1, 1], &[1, 1], 5, 2).is_none());
    }

    #[test]
    fn test_hensel_lift_all_three_factors() {
        // f = (x+1)(x+2)(x+4) over Z, all distinct linear mod 7.
        // = (x^2+3x+2)(x+4) = x^3 + 7x^2 + 14x + 8 = x^3 + 0x^2 + 0x + 1 mod 7? check:
        // expand exactly over Z:
        let lin = |c: i64| ints(&[c, 1]);
        let f = mul(&mul(&lin(1), &lin(2)), &lin(4));
        let p = 7;
        let factors_mod_p = vec![vec![1, 1], vec![2, 1], vec![4, 1]];
        let n = 3u32;
        let lifted = hensel_lift_all(&f, &factors_mod_p, p, n).unwrap();
        assert_eq!(lifted.len(), 3);
        // product of lifted factors ≡ f (mod p^n)
        let modulus = (p as i64).pow(n);
        let mut prod = ints(&[1]);
        for fct in &lifted {
            prod = mul(&prod, fct);
        }
        assert_eq!(reduce_vec(&prod, modulus), reduce_vec(&f, modulus));
        // each lifted factor reduces to its mod-p original
        for (lift_fac, orig) in lifted.iter().zip(factors_mod_p.iter()) {
            assert_eq!(&reduce_vec(lift_fac, p as i64), orig);
        }
    }

    #[test]
    fn test_arith_helpers() {
        // (x+1)(x-1) = x^2 - 1 over Z
        let a = ints(&[1, 1]);
        let b = ints(&[-1, 1]);
        assert_eq!(mul(&a, &b), ints(&[-1, 0, 1]));
        // reduce into [0, m)
        assert_eq!(reduce(&ints(&[-1, 0, 1]), &Integer::from(5i64)), ints(&[4, 0, 1]));
        // exact coefficient division
        assert_eq!(divide_coefficients(&ints(&[10, 0, 5]), &Integer::from(5i64)), ints(&[2, 0, 1]));
    }
}
