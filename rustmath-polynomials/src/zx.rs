//! Dense polynomial arithmetic over ℤ[x], coefficients `Vec<Integer>` little-endian
//! (index = power of x). This is the ℤ-layer that sits between `fp_factor`
//! (F_p[x] over `Vec<i64>`) and `zassenhaus` (factorization over ℤ): it supplies
//! the operations the rational-roots-only factorizer lacked — a **subresultant
//! pseudo-remainder GCD** (the Euclidean GCD diverges over ℤ[x] because a leading
//! coefficient need not divide), exact division, and **Yun square-free
//! decomposition with multiplicities**.
//!
//! Ported in the style of `fp_factor`/`zp_hensel`: free functions over slices, no
//! wrapper type. The zero polynomial is the empty slice; [`degree`] returns `-1`.
//!
//! References: Cohen, *A Course in Computational Algebraic Number Theory*,
//! Alg. 3.1.2 (pseudo-division), 3.3.1 (subresultant GCD), 3.4.2 (square-free).

use rustmath_integers::Integer;

#[inline]
fn izero() -> Integer {
    Integer::zero()
}

/// Strip trailing (high-degree) zero coefficients. The zero polynomial becomes `[]`.
pub fn trim(p: &[Integer]) -> Vec<Integer> {
    let mut n = p.len();
    while n > 0 && p[n - 1].is_zero() {
        n -= 1;
    }
    p[..n].to_vec()
}

/// True for the zero polynomial.
pub fn is_zero(p: &[Integer]) -> bool {
    p.iter().all(|c| c.is_zero())
}

/// Degree, or `-1` for the zero polynomial.
pub fn degree(p: &[Integer]) -> i64 {
    let mut n = p.len();
    while n > 0 && p[n - 1].is_zero() {
        n -= 1;
    }
    n as i64 - 1
}

/// Leading coefficient (panics on the zero polynomial).
pub fn leading(p: &[Integer]) -> Integer {
    let t = trim(p);
    t[t.len() - 1].clone()
}

pub fn add(a: &[Integer], b: &[Integer]) -> Vec<Integer> {
    let n = a.len().max(b.len());
    let mut out = vec![izero(); n];
    for (i, c) in a.iter().enumerate() {
        out[i] = out[i].clone() + c.clone();
    }
    for (i, c) in b.iter().enumerate() {
        out[i] = out[i].clone() + c.clone();
    }
    trim(&out)
}

pub fn sub(a: &[Integer], b: &[Integer]) -> Vec<Integer> {
    let n = a.len().max(b.len());
    let mut out = vec![izero(); n];
    for (i, c) in a.iter().enumerate() {
        out[i] = out[i].clone() + c.clone();
    }
    for (i, c) in b.iter().enumerate() {
        out[i] = out[i].clone() - c.clone();
    }
    trim(&out)
}

pub fn neg(a: &[Integer]) -> Vec<Integer> {
    a.iter().map(|c| -c.clone()).collect()
}

pub fn mul(a: &[Integer], b: &[Integer]) -> Vec<Integer> {
    if is_zero(a) || is_zero(b) {
        return Vec::new();
    }
    let mut out = vec![izero(); a.len() + b.len() - 1];
    for (i, ca) in a.iter().enumerate() {
        if ca.is_zero() {
            continue;
        }
        for (j, cb) in b.iter().enumerate() {
            out[i + j] = out[i + j].clone() + ca.clone() * cb.clone();
        }
    }
    trim(&out)
}

/// Multiply every coefficient by the scalar `s`.
pub fn scalar_mul(a: &[Integer], s: &Integer) -> Vec<Integer> {
    if s.is_zero() {
        return Vec::new();
    }
    a.iter().map(|c| c.clone() * s.clone()).collect()
}

/// Divide every coefficient by `s`, which must divide each exactly.
pub fn scalar_div_exact(a: &[Integer], s: &Integer) -> Vec<Integer> {
    a.iter()
        .map(|c| {
            debug_assert!((c.clone() % s.clone()).is_zero(), "scalar_div_exact: not divisible");
            c.clone() / s.clone()
        })
        .collect()
}

/// Multiply by `x^k`.
pub fn shift(a: &[Integer], k: usize) -> Vec<Integer> {
    if is_zero(a) {
        return Vec::new();
    }
    let mut out = vec![izero(); k];
    out.extend_from_slice(a);
    out
}

/// Content: gcd of the coefficients, taken non-negative (`0` for the zero poly).
pub fn content(p: &[Integer]) -> Integer {
    let mut g = izero();
    for c in p {
        if !c.is_zero() {
            g = g.gcd(c);
        }
    }
    g.abs()
}

/// Primitive part: `p / content(p)`, leaving the sign of `p` unchanged.
pub fn primitive_part(p: &[Integer]) -> Vec<Integer> {
    let c = content(p);
    if c.is_zero() || c.is_one() {
        return trim(p);
    }
    scalar_div_exact(&trim(p), &c)
}

/// Normalize: primitive part with a positive leading coefficient. The canonical
/// representative of a polynomial up to a rational scalar (the form a GCD returns).
pub fn normalize(p: &[Integer]) -> Vec<Integer> {
    let pp = primitive_part(p);
    if pp.is_empty() {
        return pp;
    }
    if pp[pp.len() - 1].signum() < 0 {
        neg(&pp)
    } else {
        pp
    }
}

/// Formal derivative.
pub fn derivative(p: &[Integer]) -> Vec<Integer> {
    if p.len() <= 1 {
        return Vec::new();
    }
    let mut out = vec![izero(); p.len() - 1];
    for i in 1..p.len() {
        out[i - 1] = p[i].clone() * Integer::from(i as i64);
    }
    trim(&out)
}

/// Pseudo-remainder: the unique `r` with `lc(b)^(deg a - deg b + 1) · a = q·b + r`
/// and `deg r < deg b`, computed with only ring operations (Cohen Alg. 3.1.2).
pub fn pseudo_rem(a: &[Integer], b: &[Integer]) -> Vec<Integer> {
    let b = trim(b);
    let db = degree(&b);
    assert!(db >= 0, "pseudo_rem: division by zero polynomial");
    let mut r = trim(a);
    if degree(&r) < db {
        return r;
    }
    let lcb = b[b.len() - 1].clone();
    let mut e = (degree(&r) - db + 1) as u32;
    while !is_zero(&r) && degree(&r) >= db {
        let lcr = r[r.len() - 1].clone();
        let s = (degree(&r) - db) as usize;
        // r <- lc(b)·r - lc(r)·x^s·b
        let r_scaled = scalar_mul(&r, &lcb);
        let b_term = scalar_mul(&shift(&b, s), &lcr);
        r = sub(&r_scaled, &b_term);
        e -= 1;
    }
    if e > 0 {
        r = scalar_mul(&r, &lcb.pow(e));
    }
    trim(&r)
}

/// Exact polynomial division over ℤ. Returns `Some(q)` with `a = q·b` exactly,
/// or `None` if `b ∤ a` in ℤ[x].
pub fn try_divide(a: &[Integer], b: &[Integer]) -> Option<Vec<Integer>> {
    let a = trim(a);
    let b = trim(b);
    if b.is_empty() {
        return None;
    }
    if a.is_empty() {
        return Some(Vec::new());
    }
    let db = degree(&b);
    if degree(&a) < db {
        return None;
    }
    let lcb = b[b.len() - 1].clone();
    let mut r = a.clone();
    let mut q = vec![izero(); (degree(&a) - db + 1) as usize];
    while !is_zero(&r) && degree(&r) >= db {
        let lcr = r[r.len() - 1].clone();
        if !(lcr.clone() % lcb.clone()).is_zero() {
            return None;
        }
        let coeff = lcr / lcb.clone();
        let s = (degree(&r) - db) as usize;
        q[s] = coeff.clone();
        let b_term = scalar_mul(&shift(&b, s), &coeff);
        r = sub(&r, &b_term);
    }
    if is_zero(&r) {
        Some(trim(&q))
    } else {
        None
    }
}

/// Exact division, panicking if `b ∤ a`. Use only when divisibility is guaranteed.
pub fn divide_exact(a: &[Integer], b: &[Integer]) -> Vec<Integer> {
    try_divide(a, b).expect("divide_exact: divisor does not divide dividend in ℤ[x]")
}

/// Subresultant GCD over ℤ[x] (Cohen Alg. 3.3.1). Returns the GCD with content
/// `gcd(cont a, cont b)` and a positive leading coefficient. `gcd(p, 0) = normalize(p)`.
pub fn subresultant_gcd(a: &[Integer], b: &[Integer]) -> Vec<Integer> {
    let mut a = trim(a);
    let mut b = trim(b);
    if is_zero(&a) {
        return normalize(&b);
    }
    if is_zero(&b) {
        return normalize(&a);
    }
    if degree(&a) < degree(&b) {
        std::mem::swap(&mut a, &mut b);
    }
    let ca = content(&a);
    let cb = content(&b);
    let d = ca.gcd(&cb); // integer content of the GCD
    let mut a = scalar_div_exact(&a, &ca); // primitive
    let mut b = scalar_div_exact(&b, &cb);

    let mut g = Integer::one();
    let mut h = Integer::one();
    loop {
        let delta = (degree(&a) - degree(&b)) as u32;
        let r = pseudo_rem(&a, &b);
        if is_zero(&r) {
            break;
        }
        if degree(&r) == 0 {
            b = vec![Integer::one()]; // GCD of primitive parts is 1
            break;
        }
        // a <- b ;  b <- r / (g · h^delta)
        let denom = g.clone() * h.pow(delta);
        a = b;
        b = scalar_div_exact(&r, &denom);
        // g <- lc(a) ;  h <- g^delta · h^(1-delta)
        g = a[a.len() - 1].clone();
        if delta == 1 {
            h = g.clone();
        } else if delta >= 2 {
            h = g.pow(delta) / h.pow(delta - 1);
        }
        // delta == 0 leaves h unchanged
    }
    // result = d · pp(b), with positive leading coefficient
    let mut res = scalar_mul(&primitive_part(&b), &d);
    if !res.is_empty() && res[res.len() - 1].signum() < 0 {
        res = neg(&res);
    }
    res
}

/// Square-free decomposition over ℤ via Yun's algorithm. Returns
/// `(g_i, i)` with `f = c · ∏ g_i^i`, each `g_i` primitive, square-free, pairwise
/// coprime, with positive leading coefficient; the content `c` is discarded.
/// Repeated and constant inputs are handled; the empty vector is returned for a
/// constant or zero `f`.
pub fn squarefree_decomposition(f: &[Integer]) -> Vec<(Vec<Integer>, u32)> {
    let f = trim(f);
    if degree(&f) <= 0 {
        return Vec::new();
    }
    let prim = normalize(&f);
    let fp = derivative(&prim);
    let g = subresultant_gcd(&prim, &fp);

    let mut b = divide_exact(&prim, &g);
    let mut c = divide_exact(&fp, &g);
    let mut d = sub(&c, &derivative(&b));

    let mut result = Vec::new();
    let mut i = 1u32;
    while degree(&b) > 0 {
        let a = subresultant_gcd(&b, &d);
        if degree(&a) > 0 {
            result.push((normalize(&a), i));
        }
        let b_next = divide_exact(&b, &a);
        c = divide_exact(&d, &a);
        d = sub(&c, &derivative(&b_next));
        b = b_next;
        i += 1;
        debug_assert!(i < 1_000_000, "squarefree_decomposition runaway");
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(cs: &[i64]) -> Vec<Integer> {
        cs.iter().map(|&c| Integer::from(c)).collect()
    }

    #[test]
    fn test_pseudo_rem_basic() {
        // a = x^2 - 1, b = x - 1  =>  exact, pseudo-remainder 0
        let r = pseudo_rem(&p(&[-1, 0, 1]), &p(&[-1, 1]));
        assert!(is_zero(&r));
    }

    #[test]
    fn test_subresultant_gcd_linear_common() {
        // (x-1)(x-2) and (x-1)(x-3) share x-1
        let f = mul(&p(&[-1, 1]), &p(&[-2, 1]));
        let g = mul(&p(&[-1, 1]), &p(&[-3, 1]));
        assert_eq!(subresultant_gcd(&f, &g), p(&[-1, 1]));
    }

    #[test]
    fn test_subresultant_gcd_coprime() {
        // x^2+1 and x^2+x+1 are coprime => gcd is a unit (constant 1)
        let d = subresultant_gcd(&p(&[1, 0, 1]), &p(&[1, 1, 1]));
        assert_eq!(degree(&d), 0);
    }

    #[test]
    fn test_subresultant_gcd_nonmonic() {
        // 2x^2-2 = 2(x-1)(x+1), 3x-3 = 3(x-1) => gcd content gcd(2,3)=1, pp = x-1
        let d = subresultant_gcd(&p(&[-2, 0, 2]), &p(&[-3, 3]));
        assert_eq!(d, p(&[-1, 1]));
    }

    #[test]
    fn test_try_divide_exact_and_not() {
        let f = mul(&p(&[-1, 1]), &p(&[2, 1])); // (x-1)(x+2)
        assert_eq!(try_divide(&f, &p(&[-1, 1])), Some(p(&[2, 1])));
        assert_eq!(try_divide(&f, &p(&[1, 1])), None); // x+1 ∤ f
    }

    #[test]
    fn test_squarefree_repeated() {
        // x^2 has x with multiplicity 2
        let d = squarefree_decomposition(&p(&[0, 0, 1]));
        assert_eq!(d, vec![(p(&[0, 1]), 2)]);
    }

    #[test]
    fn test_squarefree_mixed() {
        // f = (x-1)^2 (x+2)^3 : multiplicities 2 and 3
        let a = mul(&p(&[-1, 1]), &p(&[-1, 1]));
        let b = mul(&mul(&p(&[2, 1]), &p(&[2, 1])), &p(&[2, 1]));
        let f = mul(&a, &b);
        let mut d = squarefree_decomposition(&f);
        d.sort_by_key(|(_, m)| *m);
        assert_eq!(d, vec![(p(&[-1, 1]), 2), (p(&[2, 1]), 3)]);
    }

    #[test]
    fn test_squarefree_already_squarefree() {
        // x^2+1 is square-free => one factor, multiplicity 1
        let d = squarefree_decomposition(&p(&[1, 0, 1]));
        assert_eq!(d, vec![(p(&[1, 0, 1]), 1)]);
    }
}
