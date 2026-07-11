//! Low-level polynomial arithmetic over Z/p^N and GF(p) shared by the
//! p-adic extension machinery (`unramified`, `eisenstein`).
//!
//! All polynomials are **little-endian** coefficient vectors
//! `[a_0, a_1, ..., a_n]` with coefficients kept canonical in `[0, m)`.
//! Quotient-ring elements of `Z_p[x]/(u)` at precision `N` are vectors of
//! length `deg u`, reduced modulo the **monic** modulus `u` and modulo
//! `m = p^N`.
//!
//! Nothing here is public API; the honest public surface lives in
//! [`super::unramified`] and [`super::eisenstein`].

use rustmath_core::{MathError, Result};
use rustmath_integers::Integer;

/// Canonical representative of `v` in `[0, m)`, `m > 0`.
pub(crate) fn canon(v: Integer, m: &Integer) -> Integer {
    let r = v % m.clone();
    if r.signum() < 0 {
        r + m.clone()
    } else {
        r
    }
}

/// Multiply two elements of `Z[x]/(modulus)` with coefficients mod `m`.
///
/// `modulus` must be **monic** (leading coefficient exactly 1), little-endian,
/// of length `n + 1`; inputs may have any length; the result has length `n`.
pub(crate) fn polymul_mod(
    a: &[Integer],
    b: &[Integer],
    modulus: &[Integer],
    m: &Integer,
) -> Vec<Integer> {
    let n = modulus.len() - 1;
    debug_assert!(modulus[n].is_one(), "polymul_mod requires a monic modulus");
    if a.is_empty() || b.is_empty() {
        return vec![Integer::zero(); n];
    }
    let mut res = vec![Integer::zero(); a.len() + b.len() - 1];
    for (i, ai) in a.iter().enumerate() {
        if ai.is_zero() {
            continue;
        }
        for (j, bj) in b.iter().enumerate() {
            if bj.is_zero() {
                continue;
            }
            res[i + j] = canon(res[i + j].clone() + ai.clone() * bj.clone(), m);
        }
    }
    // reduce x^k for k >= n using the monic modulus: x^k = -sum m_i x^{k-n+i}
    for k in (n..res.len()).rev() {
        if res[k].is_zero() {
            continue;
        }
        let c = std::mem::replace(&mut res[k], Integer::zero());
        for i in 0..n {
            if modulus[i].is_zero() {
                continue;
            }
            res[k - n + i] = canon(
                res[k - n + i].clone() - c.clone() * modulus[i].clone(),
                m,
            );
        }
    }
    res.truncate(n);
    res.resize(n, Integer::zero());
    res
}

/// Evaluate the integer polynomial `pol` (little-endian) at the quotient-ring
/// element `r` (Horner), all arithmetic in `Z[x]/(modulus)` mod `m`.
pub(crate) fn eval_poly_ext(
    pol: &[Integer],
    r: &[Integer],
    modulus: &[Integer],
    m: &Integer,
) -> Vec<Integer> {
    let n = modulus.len() - 1;
    let mut acc = vec![Integer::zero(); n];
    for c in pol.iter().rev() {
        acc = polymul_mod(&acc, r, modulus, m);
        acc[0] = canon(acc[0].clone() + c.clone(), m);
    }
    acc
}

/// `a^e` in `Z[x]/(modulus)` mod `m`, for an arbitrary-size exponent `e >= 0`.
pub(crate) fn polypow_mod(
    a: &[Integer],
    e: &Integer,
    modulus: &[Integer],
    m: &Integer,
) -> Vec<Integer> {
    let n = modulus.len() - 1;
    let mut r = vec![Integer::zero(); n];
    r[0] = canon(Integer::one(), m);
    let mut base = a.to_vec();
    let mut exp = e.clone();
    let two = Integer::from(2);
    while exp.signum() > 0 {
        if exp.is_odd() {
            r = polymul_mod(&r, &base, modulus, m);
        }
        base = polymul_mod(&base, &base, modulus, m);
        exp = exp / two.clone();
    }
    r
}

// ---------------------------------------------------------------------------
// GF(p)[x] helpers (coefficients canonical in [0, p), trailing zeros trimmed)
// ---------------------------------------------------------------------------

/// Trim trailing zeros (the zero polynomial becomes the empty vector).
pub(crate) fn gfp_trim(mut v: Vec<Integer>) -> Vec<Integer> {
    while v.last().is_some_and(|c| c.is_zero()) {
        v.pop();
    }
    v
}

/// Reduce all coefficients mod `p` and trim.
pub(crate) fn gfp_reduce(v: &[Integer], p: &Integer) -> Vec<Integer> {
    gfp_trim(v.iter().map(|c| canon(c.clone(), p)).collect())
}

/// Polynomial division with remainder over GF(p): `a = q*b + r`, `deg r < deg b`.
pub(crate) fn gfp_divrem(
    a: &[Integer],
    b: &[Integer],
    p: &Integer,
) -> Result<(Vec<Integer>, Vec<Integer>)> {
    let b = gfp_reduce(b, p);
    if b.is_empty() {
        return Err(MathError::DivisionByZero);
    }
    let mut r = gfp_reduce(a, p);
    let db = b.len() - 1;
    let lb_inv = b[db]
        .mod_inverse(p)
        .ok_or(MathError::NotInvertible)?;
    let mut q = vec![Integer::zero(); r.len().saturating_sub(db)];
    // invariant: deg r decreases strictly each pass (leading term cancels)
    while r.len() >= b.len() {
        let dr = r.len() - 1;
        let coef = canon(r[dr].clone() * lb_inv.clone(), p);
        q[dr - db] = coef.clone();
        for i in 0..=db {
            let idx = dr - db + i;
            r[idx] = canon(r[idx].clone() - coef.clone() * b[i].clone(), p);
        }
        debug_assert!(r[dr].is_zero());
        r = gfp_trim(r);
    }
    Ok((gfp_trim(q), r))
}

/// Monic gcd over GF(p).
pub(crate) fn gfp_gcd(a: &[Integer], b: &[Integer], p: &Integer) -> Result<Vec<Integer>> {
    let mut r0 = gfp_reduce(a, p);
    let mut r1 = gfp_reduce(b, p);
    while !r1.is_empty() {
        let (_, rem) = gfp_divrem(&r0, &r1, p)?;
        r0 = r1;
        r1 = rem;
    }
    if r0.is_empty() {
        return Ok(r0);
    }
    // make monic
    let d = r0.len() - 1;
    let inv = r0[d].mod_inverse(p).ok_or(MathError::NotInvertible)?;
    Ok(gfp_trim(
        r0.iter().map(|c| canon(c.clone() * inv.clone(), p)).collect(),
    ))
}

/// Inverse of `a` modulo the polynomial `u` over GF(p) (extended Euclid).
///
/// Errors with `NotInvertible` if `gcd(a, u) != 1`.
pub(crate) fn gfp_inverse_mod(
    a: &[Integer],
    u: &[Integer],
    p: &Integer,
) -> Result<Vec<Integer>> {
    let u = gfp_reduce(u, p);
    let a = gfp_reduce(a, p);
    if a.is_empty() {
        return Err(MathError::NotInvertible);
    }
    // (r0, s0) = (u, 0), (r1, s1) = (a, 1); invariant s_i * a = r_i (mod u)
    let mut r0 = u.clone();
    let mut s0: Vec<Integer> = vec![];
    let mut r1 = a;
    let mut s1 = vec![Integer::one()];
    while !r1.is_empty() {
        let (q, rem) = gfp_divrem(&r0, &r1, p)?;
        // s2 = s0 - q * s1  (all mod p)
        let qs1 = gfp_polymul_p(&q, &s1, p);
        let s2 = gfp_polysub_p(&s0, &qs1, p);
        r0 = r1;
        s0 = s1;
        r1 = rem;
        s1 = s2;
    }
    if r0.len() != 1 {
        return Err(MathError::NotInvertible); // gcd has positive degree
    }
    let inv = r0[0].mod_inverse(p).ok_or(MathError::NotInvertible)?;
    Ok(gfp_trim(
        s0.iter().map(|c| canon(c.clone() * inv.clone(), p)).collect(),
    ))
}

fn gfp_polymul_p(a: &[Integer], b: &[Integer], p: &Integer) -> Vec<Integer> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let mut res = vec![Integer::zero(); a.len() + b.len() - 1];
    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            res[i + j] = canon(res[i + j].clone() + ai.clone() * bj.clone(), p);
        }
    }
    gfp_trim(res)
}

fn gfp_polysub_p(a: &[Integer], b: &[Integer], p: &Integer) -> Vec<Integer> {
    let n = a.len().max(b.len());
    let mut res = vec![Integer::zero(); n];
    for i in 0..n {
        let av = a.get(i).cloned().unwrap_or_else(Integer::zero);
        let bv = b.get(i).cloned().unwrap_or_else(Integer::zero);
        res[i] = canon(av - bv, p);
    }
    gfp_trim(res)
}

/// Newton-lift the inverse of a unit `a` of `Z_p[x]/(modulus)` from its
/// GF(p) inverse up to precision `p^n` (`z -> z(2 - az)` doubles precision).
///
/// Self-certifying: verifies `a * z == 1 mod (modulus, p^n)` before returning.
pub(crate) fn ext_inverse(
    a: &[Integer],
    modulus: &[Integer],
    p: &Integer,
    n: u32,
) -> Result<Vec<Integer>> {
    let deg = modulus.len() - 1;
    let z0 = gfp_inverse_mod(a, modulus, p)?;
    let mut z = z0;
    z.resize(deg, Integer::zero());
    let mut k: u32 = 1;
    while k < n {
        k = (2 * k).min(n);
        let mk = p.pow(k);
        let az = polymul_mod(a, &z, modulus, &mk);
        let mut two_minus: Vec<Integer> =
            az.iter().map(|c| canon(-c.clone(), &mk)).collect();
        two_minus[0] = canon(two_minus[0].clone() + Integer::from(2), &mk);
        z = polymul_mod(&z, &two_minus, modulus, &mk);
    }
    // certify
    let mn = p.pow(n);
    let prod = polymul_mod(a, &z, modulus, &mn);
    let mut is_one = prod[0].is_one();
    for c in &prod[1..] {
        is_one &= c.is_zero();
    }
    if !is_one {
        return Err(MathError::NumericalError(
            "ext_inverse: certification failed (a*z != 1); input is not a unit \
             or precision bookkeeping is broken"
                .to_string(),
        ));
    }
    Ok(z)
}

/// Distinct prime factors of `n` (trial division; `n` is a tiny extension degree).
pub(crate) fn distinct_prime_factors(mut n: usize) -> Vec<usize> {
    let mut out = vec![];
    let mut d = 2;
    while d * d <= n {
        if n % d == 0 {
            out.push(d);
            while n % d == 0 {
                n /= d;
            }
        }
        d += 1;
    }
    if n > 1 {
        out.push(n);
    }
    out
}

/// Irreducibility of the reduction of `modulus` mod `p` over GF(p).
///
/// Standard criterion for monic `u` of degree `n >= 1`:
/// `x^(p^n) == x mod (u, p)` and `gcd(x^(p^(n/q)) - x, u) == 1` in GF(p)[x]
/// for every prime `q | n`.
pub(crate) fn is_irreducible_mod_p(modulus: &[Integer], p: &Integer) -> Result<bool> {
    let n = modulus.len() - 1;
    if n == 0 {
        return Ok(false);
    }
    if n == 1 {
        return Ok(true); // linear polynomials are irreducible
    }
    // the reduction must still have degree n (leading coeff a p-unit);
    // callers pass monic moduli so this holds, but check honestly:
    if canon(modulus[n].clone(), p).is_zero() {
        return Err(MathError::InvalidArgument(
            "is_irreducible_mod_p: leading coefficient vanishes mod p".to_string(),
        ));
    }
    let x_vec = {
        let mut v = vec![Integer::zero(); n];
        v[1] = Integer::one();
        v
    };
    // x^(p^n) mod (u, p)
    let pn = {
        let mut e = Integer::one();
        for _ in 0..n {
            e = e * p.clone();
        }
        e
    };
    let xpn = polypow_mod(&x_vec, &pn, modulus, p);
    if gfp_reduce(&xpn, p) != gfp_reduce(&x_vec, p) {
        return Ok(false);
    }
    for q in distinct_prime_factors(n) {
        let mut e = Integer::one();
        for _ in 0..(n / q) {
            e = e * p.clone();
        }
        let t = polypow_mod(&x_vec, &e, modulus, p);
        // t - x
        let mut diff = t;
        diff[1] = canon(diff[1].clone() - Integer::one(), p);
        let g = gfp_gcd(&diff, modulus, p)?;
        if g.len() != 1 {
            return Ok(false); // gcd nontrivial (or diff == 0 => g == u)
        }
    }
    Ok(true)
}

/// Exact determinant of a square integer matrix via Bareiss's fraction-free
/// elimination (all divisions exact).
pub(crate) fn det_bareiss(mut m: Vec<Vec<Integer>>) -> Integer {
    let n = m.len();
    if n == 0 {
        return Integer::one();
    }
    debug_assert!(m.iter().all(|row| row.len() == n));
    let mut sign = 1i8;
    let mut prev = Integer::one();
    for k in 0..n - 1 {
        if m[k][k].is_zero() {
            // find a pivot row below
            let Some(swap) = (k + 1..n).find(|&i| !m[i][k].is_zero()) else {
                return Integer::zero();
            };
            m.swap(k, swap);
            sign = -sign;
        }
        for i in k + 1..n {
            for j in k + 1..n {
                let num =
                    m[i][j].clone() * m[k][k].clone() - m[i][k].clone() * m[k][j].clone();
                m[i][j] = num / prev.clone(); // exact by Bareiss
            }
            m[i][k] = Integer::zero();
        }
        prev = m[k][k].clone();
    }
    let det = m[n - 1][n - 1].clone();
    if sign < 0 {
        -det
    } else {
        det
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ints(v: &[i64]) -> Vec<Integer> {
        v.iter().map(|&c| Integer::from(c)).collect()
    }

    #[test]
    fn test_polymul_mod_basic() {
        // (x)(x) mod (x^2 - 5, 7^4): x^2 = 5
        let modulus = ints(&[-5, 0, 1]);
        let m = Integer::from(7).pow(4);
        let x = ints(&[0, 1]);
        let r = polymul_mod(&x, &x, &modulus, &m);
        assert_eq!(r, ints(&[5, 0]));
    }

    #[test]
    fn test_gfp_inverse_mod() {
        // invert x mod (x^2 + 4x + 2) over GF(5): x * z = 1
        let p = Integer::from(5);
        let u = ints(&[2, 4, 1]);
        let a = ints(&[0, 1]);
        let z = gfp_inverse_mod(&a, &u, &p).unwrap();
        let mut zfull = z.clone();
        zfull.resize(2, Integer::zero());
        let prod = polymul_mod(&a, &zfull, &u, &p);
        assert_eq!(prod, ints(&[1, 0]));
    }

    #[test]
    fn test_ext_inverse_certifies() {
        let p = Integer::from(5);
        let u = ints(&[2, 4, 1]); // Conway(5,2)
        let a = ints(&[3, 2]);
        let z = ext_inverse(&a, &u, &p, 8).unwrap();
        let m = p.pow(8);
        let prod = polymul_mod(&a, &z, &u, &m);
        assert_eq!(prod, ints(&[1, 0]));
        // p itself is not a unit
        assert!(ext_inverse(&ints(&[5, 0]), &u, &p, 8).is_err());
    }

    #[test]
    fn test_irreducibility_mod_p() {
        let p = Integer::from(5);
        // Conway(5,2) = x^2 + 4x + 2 irreducible mod 5
        assert!(is_irreducible_mod_p(&ints(&[2, 4, 1]), &p).unwrap());
        // x^2 - 6 = (x-1)(x+1) mod 5: the old placeholder bug
        assert!(!is_irreducible_mod_p(&ints(&[-6, 0, 1]), &p).unwrap());
        // Conway(5,3) = x^3 + 3x + 3 irreducible mod 5 (sympy-verified)
        assert!(is_irreducible_mod_p(&ints(&[3, 3, 0, 1]), &p).unwrap());
        // x^3 - x factors
        assert!(!is_irreducible_mod_p(&ints(&[0, -1, 0, 1]), &p).unwrap());
        // x^4 + x + 1 over GF(2) is irreducible; x^4 + 1 is not
        let two = Integer::from(2);
        assert!(is_irreducible_mod_p(&ints(&[1, 1, 0, 0, 1]), &two).unwrap());
        assert!(!is_irreducible_mod_p(&ints(&[1, 0, 0, 0, 1]), &two).unwrap());
    }

    #[test]
    fn test_det_bareiss() {
        // det [[1,2],[3,4]] = -2
        let m = vec![ints(&[1, 2]), ints(&[3, 4])];
        assert_eq!(det_bareiss(m), Integer::from(-2));
        // det of companion of x^3 - 2 (mult-by-pi matrix) = 2
        let m = vec![ints(&[0, 0, 2]), ints(&[1, 0, 0]), ints(&[0, 1, 0])];
        assert_eq!(det_bareiss(m), Integer::from(2));
        // singular with zero leading pivot
        let m = vec![ints(&[0, 1]), ints(&[0, 5])];
        assert_eq!(det_bareiss(m), Integer::from(0));
        // needs a row swap: [[0,1],[1,0]] det = -1
        let m = vec![ints(&[0, 1]), ints(&[1, 0])];
        assert_eq!(det_bareiss(m), Integer::from(-1));
    }
}
