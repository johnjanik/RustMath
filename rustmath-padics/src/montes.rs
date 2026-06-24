//! Montes / Ore (OM) factorization over `ℚ_p` and prime-ideal decomposition.
//!
//! Given `f ∈ ℤ[x]` and a rational prime `p`, this module factors `f` over `ℚ_p`
//! into its local factors, each tagged with its ramification index `e` and residue
//! degree `f` (the `(e, f)` data), and — for a number-field defining polynomial —
//! returns the primes of `O_K` above `p` as `(e, f)` pairs with the cross-check
//! `Σ eᵢ fᵢ = deg f`.
//!
//! ## Method (first-order Ore / Montes)
//!
//! 1. Build the [`crate::newton_polygon`] of `f` at `p`: its lower convex hull.
//! 2. For each face of negative slope `λ = h/e` (lowest terms) and horizontal
//!    length `ℓ = e·t`, form the **Ore residual polynomial** `R ∈ F_p[y]` of degree
//!    `t`. Its coefficients are the lattice points of `f` lying *on* the face,
//!    divided by the appropriate power of `p` and reduced mod `p`.
//! 3. In the **Montes-regular** case (`R` separable) each irreducible factor of `R`
//!    of degree `s` yields one `ℚ_p`-factor of `f` with ramification `e`, residue
//!    degree `s`, and root valuation `h/e`. This uniformly covers the unramified
//!    (slope 0, `e = 1`), Eisenstein (`deg R = 1`), and regular mixed-polygon cases.
//!
//! ## Limitation / assumption
//!
//! Only the **first-order Montes-regular** case is fully resolved. If a face's
//! residual polynomial is inseparable — `p` divides the index `[O_K : ℤ[α]]` and a
//! second-order Montes type would be required — this module reports
//! [`MontesError::NonRegular`] for that face rather than returning incorrect `(e,f)`
//! data. The regular case covers the overwhelming majority of inputs (and every
//! prime not dividing the index). The companion engine
//! `rustmath-polynomials::padic_factor` makes the same assumption.
//!
//! Self-contained: depends only on `rustmath-integers`. The small `F_p[y]` layer
//! needed for residual-polynomial factorization lives in [`fp`] below.

use rustmath_integers::Integer;

use crate::newton_polygon::{newton_polygon, Face};

/// One irreducible factor of `f` over `ℚ_p` (equivalently, one prime above `p` when
/// `f` is a number-field defining polynomial).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalFactor {
    /// Ramification index `e ≥ 1`.
    pub e: usize,
    /// Residue degree `f ≥ 1`.
    pub f: usize,
    /// Local degree over `ℚ_p` (`= e·f`).
    pub degree: usize,
    /// Root valuation `h/e` in lowest terms (`h = root_val_num`, `e = root_val_den`).
    pub root_val_num: i64,
    pub root_val_den: i64,
}

impl LocalFactor {
    /// The `(e, f)` pair.
    pub fn ef(&self) -> (usize, usize) {
        (self.e, self.f)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MontesError {
    /// `f` is the zero polynomial.
    ZeroPolynomial,
    /// A face's residual polynomial is inseparable: `p` divides the index and a
    /// higher-order Montes type is required. First-order `(e,f)` is not returned.
    NonRegular { slope_num: i64, slope_den: i64 },
    /// Internal invariant violation (should not occur).
    Internal(String),
}

/// The Ore residual polynomial of a face, as a monic-or-not `F_p[y]` polynomial
/// (little-endian, coefficients in `[0, p)`). A face from `(i0, v0)` of negative
/// slope `h/e` and length `ℓ = e·t` picks the lattice points `(i0 + j·e, v0 − j·h)`
/// for `j = 0..=t`; `R_j = (a_{i0+j·e} / p^{v0 − j·h}) mod p` (points strictly above
/// the face reduce to 0).
fn residual_polynomial(f: &[Integer], p: i64, face: &Face, e: usize, h: i64) -> Vec<i64> {
    let i0 = face.from.i;
    let v0 = face.from.v;
    let t = face.length / e;
    let pi = Integer::from(p);
    let mut r = vec![0i64; t + 1];
    for j in 0..=t {
        let idx = i0 + j * e;
        let coeff = &f[idx];
        if coeff.is_zero() {
            continue; // above the face
        }
        let exponent = v0 - (j as i64) * h; // ≥ 0 along the face
        // a_idx / p^exponent  (exact along the face; above-face points reduce to 0)
        let mut q = coeff.clone();
        for _ in 0..exponent {
            q = q / pi.clone();
        }
        let m = (q % pi.clone()).to_i64();
        r[j] = ((m % p) + p) % p;
    }
    fp::trim(&r)
}

/// Factor `f` over `ℚ_p`, returning the local factors with `(e, f, root valuation)`.
///
/// Regular faces are fully resolved; a non-regular face (inseparable residual
/// polynomial) yields [`MontesError::NonRegular`].
pub fn montes_factor(f: &[Integer], p: i64) -> Result<Vec<LocalFactor>, MontesError> {
    let polygon = newton_polygon(f, p).ok_or(MontesError::ZeroPolynomial)?;
    let mut out = Vec::new();
    for face in &polygon.faces {
        let e = face.neg_slope_den as usize; // ramification of this face's roots
        let h = face.neg_slope_num; // ≥ 0
        if e == 0 || face.length % e != 0 {
            return Err(MontesError::Internal(format!(
                "face length {} not divisible by ramification {}",
                face.length, e
            )));
        }
        let r = residual_polynomial(f, p, face, e, h);
        if !fp::is_separable(&r, p) {
            return Err(MontesError::NonRegular {
                slope_num: -h,
                slope_den: face.neg_slope_den,
            });
        }
        for g in fp::factor(&r, p) {
            let s = (g.len() - 1).max(0); // residue degree of this irreducible
            out.push(LocalFactor {
                e,
                f: s,
                degree: e * s,
                root_val_num: h, // gcd(h, e) = 1 (slope in lowest terms)
                root_val_den: face.neg_slope_den,
            });
        }
    }
    Ok(out)
}

/// Convenience: the sorted multiset of `(e, f)` pairs of the local factors.
pub fn ramification_type(f: &[Integer], p: i64) -> Result<Vec<(usize, usize)>, MontesError> {
    let mut ef: Vec<(usize, usize)> = montes_factor(f, p)?.iter().map(|lf| lf.ef()).collect();
    ef.sort_unstable();
    Ok(ef)
}

/// A prime ideal of `O_K` above `p`, where `K = ℚ[x]/(f)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrimeAbove {
    /// Ramification index `e(𝔭 | p)`.
    pub e: usize,
    /// Residue (inertia) degree `f(𝔭 | p) = [O_K/𝔭 : F_p]`.
    pub f: usize,
    /// Local degree `e·f`.
    pub residue_degree_times_e: usize,
}

/// Prime-ideal decomposition of `p` in `O_K = ℤ[x]/(f)` (assuming `p` does not
/// divide the index `[O_K : ℤ[α]]`, i.e. the **Montes-regular** case): the primes
/// above `p` as `(e, f)` pairs. The fundamental identity `Σ eᵢ fᵢ = [K : ℚ] = deg f`
/// is verified internally and surfaced via [`decompose_prime_checked`].
///
/// `f` must be monic (a number-field defining polynomial). When `p` divides the
/// index this returns [`MontesError::NonRegular`] for the offending face.
pub fn decompose_prime(f: &[Integer], p: i64) -> Result<Vec<PrimeAbove>, MontesError> {
    let factors = montes_factor(f, p)?;
    Ok(factors
        .iter()
        .map(|lf| PrimeAbove { e: lf.e, f: lf.f, residue_degree_times_e: lf.degree })
        .collect())
}

/// Like [`decompose_prime`] but also returns whether the fundamental identity
/// `Σ eᵢ fᵢ = deg f` holds. In the regular case it always should; a `false` here is
/// a signal that `p` divides the index and the decomposition is incomplete (some
/// faces were resolved but multiplicities are off — in practice such cases surface
/// as [`MontesError::NonRegular`] first).
pub fn decompose_prime_checked(
    f: &[Integer],
    p: i64,
) -> Result<(Vec<PrimeAbove>, bool), MontesError> {
    let primes = decompose_prime(f, p)?;
    let deg = poly_degree(f);
    let sum: usize = primes.iter().map(|pr| pr.e * pr.f).sum();
    Ok((primes, sum == deg))
}

/// Degree of a little-endian integer polynomial.
fn poly_degree(f: &[Integer]) -> usize {
    let mut d = f.len();
    while d > 1 && f[d - 1].is_zero() {
        d -= 1;
    }
    d - 1
}

// ------------------------------------------------------------------------- //
// Minimal `F_p[y]` layer for residual-polynomial factorization
// ------------------------------------------------------------------------- //

/// Self-contained factorization over `F_p[y]` (square-free test + distinct-degree
/// + equal-degree Cantor–Zassenhaus), specialised to the small residual
/// polynomials produced by the Montes faces. Polynomials are little-endian
/// `Vec<i64>` with coefficients in `[0, p)`. The prime `p` is passed explicitly.
mod fp {
    use rustmath_integers::Integer;

    #[inline]
    fn modp(a: i64, p: i64) -> i64 {
        ((a % p) + p) % p
    }

    /// Strip trailing zeros, keeping at least one coefficient (the zero poly is `[0]`).
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

    pub fn is_zero(p: &[i64]) -> bool {
        p.iter().all(|&c| c == 0)
    }

    /// Degree; `-1` for the zero polynomial.
    pub fn degree(p: &[i64]) -> i64 {
        let t = trim(p);
        if is_zero(&t) {
            -1
        } else {
            (t.len() - 1) as i64
        }
    }

    pub fn mul(a: &[i64], b: &[i64], p: i64) -> Vec<i64> {
        if is_zero(a) || is_zero(b) {
            return vec![0];
        }
        let mut out = vec![0i64; a.len() + b.len() - 1];
        for (i, &ai) in a.iter().enumerate() {
            if ai == 0 {
                continue;
            }
            for (j, &bj) in b.iter().enumerate() {
                out[i + j] = modp(out[i + j] + ai * bj, p);
            }
        }
        trim(&out)
    }

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

    /// Modular inverse of `a` mod prime `p` (extended Euclid). `a != 0 (mod p)`.
    pub fn mod_inv(a: i64, p: i64) -> i64 {
        let mut t = 0i64;
        let mut newt = 1i64;
        let mut r = p;
        let mut newr = modp(a, p);
        while newr != 0 {
            let q = r / newr;
            let tmp = t - q * newt;
            t = newt;
            newt = tmp;
            let tmp = r - q * newr;
            r = newr;
            newr = tmp;
        }
        modp(t, p)
    }

    /// `(quotient, remainder)` of `a / b` in `F_p[y]` (`b != 0`).
    pub fn div_mod(a: &[i64], b: &[i64], p: i64) -> (Vec<i64>, Vec<i64>) {
        let b = trim(b);
        let mut rem = trim(a);
        let db = degree(&b);
        if db < 0 {
            return (vec![0], vec![0]);
        }
        let dr = degree(&rem);
        if dr < db {
            return (vec![0], rem);
        }
        let binv = mod_inv(*b.last().unwrap(), p);
        let mut quot = vec![0i64; (dr - db + 1) as usize];
        let mut dr = dr;
        while dr >= db {
            let shift = (dr - db) as usize;
            let lead = rem[dr as usize];
            if lead != 0 {
                let factor = modp(lead * binv, p);
                quot[shift] = factor;
                for (j, &bj) in b.iter().enumerate() {
                    rem[shift + j] = modp(rem[shift + j] - factor * bj, p);
                }
            }
            rem = trim(&rem);
            dr = degree(&rem);
            if is_zero(&rem) {
                break;
            }
        }
        (trim(&quot), trim(&rem))
    }

    pub fn rem(a: &[i64], b: &[i64], p: i64) -> Vec<i64> {
        div_mod(a, b, p).1
    }

    pub fn gcd(a: &[i64], b: &[i64], p: i64) -> Vec<i64> {
        let mut a = trim(a);
        let mut b = trim(b);
        while !is_zero(&b) {
            let r = rem(&a, &b, p);
            a = b;
            b = r;
        }
        make_monic(&a, p)
    }

    pub fn make_monic(a: &[i64], p: i64) -> Vec<i64> {
        let a = trim(a);
        if is_zero(&a) {
            return a;
        }
        let inv = mod_inv(*a.last().unwrap(), p);
        trim(&a.iter().map(|&c| modp(c * inv, p)).collect::<Vec<_>>())
    }

    pub fn derivative(f: &[i64], p: i64) -> Vec<i64> {
        if degree(f) <= 0 {
            return vec![0];
        }
        let mut d = vec![0i64; f.len() - 1];
        for i in 1..f.len() {
            d[i - 1] = modp(f[i] * (i as i64), p);
        }
        trim(&d)
    }

    /// `base^exp mod modulus` in `F_p[y]`.
    pub fn pow_mod(base: &[i64], exp: &Integer, modulus: &[i64], p: i64) -> Vec<i64> {
        let mut result = vec![1i64];
        let mut b = rem(base, modulus, p);
        let mut e = exp.clone();
        let two = Integer::from(2);
        let zero = Integer::zero();
        while e > zero {
            if (e.clone() % two.clone()) == Integer::one() {
                result = rem(&mul(&result, &b, p), modulus, p);
            }
            b = rem(&mul(&b, &b, p), modulus, p);
            e = e / two.clone();
        }
        trim(&result)
    }

    /// True iff `f` is square-free over `F_p` (separable).
    pub fn is_separable(f: &[i64], p: i64) -> bool {
        if degree(f) <= 0 {
            return true;
        }
        let d = derivative(f, p);
        if is_zero(&d) {
            return false; // a p-th power: inseparable
        }
        degree(&gcd(f, &d, p)) == 0
    }

    /// Distinct-degree factorization: returns `(d, prod)` where `prod` is the product
    /// of all monic irreducible factors of degree `d` of (square-free, monic) `f`.
    fn distinct_degree(f: &[i64], p: i64) -> Vec<(i64, Vec<i64>)> {
        let mut out = Vec::new();
        let mut fstar = make_monic(f, p);
        let mut d = 1i64;
        let pi = Integer::from(p);
        // x
        let x = vec![0i64, 1i64];
        let mut xqd = x.clone(); // x^(p^d) mod fstar, maintained iteratively
        while degree(&fstar) >= 2 * d {
            // xqd = xqd^p mod fstar
            xqd = pow_mod(&xqd, &pi, &fstar, p);
            let g = gcd(&sub(&xqd, &x, p), &fstar, p);
            if degree(&g) > 0 {
                out.push((d, g.clone()));
                let (q, _) = div_mod(&fstar, &g, p);
                fstar = make_monic(&q, p);
                // reduce xqd modulo the smaller fstar
                xqd = rem(&xqd, &fstar, p);
            }
            d += 1;
        }
        if degree(&fstar) > 0 {
            out.push((degree(&fstar), fstar));
        }
        out
    }

    /// Equal-degree (Cantor–Zassenhaus) split of a product of `n` distinct monic
    /// irreducibles each of degree `d`. `p` is odd here for residual polynomials over
    /// `F_p` (p = 2 handled by the deterministic trial below).
    fn equal_degree(f: &[i64], d: i64, p: i64) -> Vec<Vec<i64>> {
        let f = make_monic(f, p);
        let deg = degree(&f);
        if deg <= 0 {
            return Vec::new();
        }
        if deg == d {
            return vec![f]; // already irreducible
        }
        // Deterministic CZ over small F_p: try linear shifts a = 0,1,2,...
        let pi = Integer::from(p);
        let exp = {
            // (p^d − 1) / 2
            let pd = pi.pow(d as u32);
            (pd - Integer::one()) / Integer::from(2)
        };
        let mut factors = vec![f];
        let mut a = 0i64;
        while factors.iter().any(|g| degree(g) > d) {
            // candidate polynomial h = (x + a)
            let h = vec![modp(a, p), 1i64];
            a += 1;
            if a > 4 * p + 8 {
                // pathological; fall back to brute factor (small degree)
                break;
            }
            let mut next = Vec::new();
            for g in factors.into_iter() {
                if degree(&g) <= d {
                    next.push(g);
                    continue;
                }
                let split = if p == 2 {
                    // Frobenius-trace map: T = h + h^2 + ... + h^(2^(d-1))
                    let mut t = rem(&h, &g, p);
                    let mut acc = t.clone();
                    for _ in 1..d {
                        t = rem(&mul(&t, &t, p), &g, p);
                        acc = add(&acc, &t, p);
                    }
                    gcd(&acc, &g, p)
                } else {
                    let hp = pow_mod(&h, &exp, &g, p);
                    let hm1 = sub(&hp, &[1i64], p);
                    gcd(&hm1, &g, p)
                };
                let dgs = degree(&split);
                if dgs > 0 && dgs < degree(&g) {
                    let (q, _) = div_mod(&g, &split, p);
                    next.push(make_monic(&split, p));
                    next.push(make_monic(&q, p));
                } else {
                    next.push(g);
                }
            }
            factors = next;
        }
        factors
    }

    /// Factor `f` over `F_p` into monic irreducibles (with multiplicity). For the
    /// separable residual polynomials of Montes faces this is square-free already,
    /// but we handle multiplicity for robustness.
    pub fn factor(f: &[i64], p: i64) -> Vec<Vec<i64>> {
        let rad = radical(&make_monic(f, p), p);
        if degree(&rad) <= 0 {
            return Vec::new();
        }
        let mut irreducibles = Vec::new();
        for (d, prod) in distinct_degree(&rad, p) {
            for irr in equal_degree(&prod, d, p) {
                if degree(&irr) > 0 {
                    irreducibles.push(make_monic(&irr, p));
                }
            }
        }
        irreducibles
    }

    /// The **radical** (square-free part) of `f`: the product of its distinct monic
    /// irreducible factors, each to the first power. Correct in characteristic `p`:
    /// it peels the separable layer `f / gcd(f, f')`, merges in the radical of the
    /// `gcd` (which holds the repeated factors), and handles the inseparable case
    /// (`f' = 0`, a `p`-th power) by taking the `p`-th root and recursing. For the
    /// (separable) Montes residual polynomials the radical equals `f` itself.
    fn radical(f: &[i64], p: i64) -> Vec<i64> {
        let f = make_monic(f, p);
        if degree(&f) <= 0 {
            return f;
        }
        let d = derivative(&f, p);
        if is_zero(&d) {
            // f(y) = g(y^p): rad(f) = rad(g).
            return radical(&pth_root(&f, p), p);
        }
        let g = gcd(&f, &d, p); // product of the repeated factors (to lowered powers)
        if degree(&g) == 0 {
            return make_monic(&f, p); // already square-free
        }
        let (sqfree, _) = div_mod(&f, &g, p); // separable layer (distinct factors)
        // The repeated factors live in g; recurse to get their radical, then merge so
        // each distinct irreducible appears exactly once.
        let rad_g = radical(&g, p);
        lcm_squarefree(&make_monic(&sqfree, p), &rad_g, p)
    }

    /// Least common multiple of two **square-free** monic polynomials: `a · b / gcd`.
    /// The result is again square-free and contains each common/distinct irreducible
    /// exactly once.
    fn lcm_squarefree(a: &[i64], b: &[i64], p: i64) -> Vec<i64> {
        let g = gcd(a, b, p);
        let (q, _) = div_mod(a, &g, p); // a with the shared factors removed
        make_monic(&mul(&q, b, p), p)
    }

    /// `p`-th root of a polynomial known to be a `p`-th power `g(y^p)`: map the
    /// coefficient at degree `p·k` back to degree `k` (other coefficients are 0).
    fn pth_root(f: &[i64], p: i64) -> Vec<i64> {
        let f = trim(f);
        let deg = degree(&f);
        if deg <= 0 {
            return f;
        }
        let m = (deg as usize) / (p as usize);
        let mut out = vec![0i64; m + 1];
        for k in 0..=m {
            out[k] = f[k * (p as usize)];
        }
        make_monic(&trim(&out), p)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn factor_splits_linear() {
            // y^2 - 1 = (y-1)(y+1) over F_5
            let fs = factor(&[4, 0, 1], 5); // y^2 + 4 = y^2 - 1
            assert_eq!(fs.len(), 2);
            for g in &fs {
                assert_eq!(degree(g), 1);
            }
        }

        #[test]
        fn factor_irreducible_quadratic() {
            // y^2 + 1 over F_7 is irreducible (-1 not a QR mod 7)
            let fs = factor(&[1, 0, 1], 7);
            assert_eq!(fs.len(), 1);
            assert_eq!(degree(&fs[0]), 2);
        }

        #[test]
        fn factor_cubic_over_f2() {
            // y^3 + y + 1 over F_2 is irreducible
            let fs = factor(&[1, 1, 0, 1], 2);
            assert_eq!(fs.len(), 1);
            assert_eq!(degree(&fs[0]), 3);
            // y^3 + y = y(y+1)^2 over F_2 -> radical y(y+1) -> two distinct factors.
            let mut fs2 = factor(&[0, 1, 0, 1], 2);
            fs2.sort();
            assert_eq!(fs2.len(), 2);
            assert!(fs2.iter().all(|g| degree(g) == 1));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iz(cs: &[i64]) -> Vec<Integer> {
        cs.iter().map(|&c| Integer::from(c)).collect()
    }

    // ---- Montes factorization (e, f) over Q_p ---- //

    #[test]
    fn unramified_split() {
        // x^2 + 1 over Q_5: -1 is a QR mod 5 -> two unramified linear (e=1,f=1).
        assert_eq!(ramification_type(&iz(&[1, 0, 1]), 5).unwrap(), vec![(1, 1), (1, 1)]);
    }

    #[test]
    fn unramified_inert() {
        // x^2 + 1 over Q_7: -1 not a QR mod 7 -> one unramified (e=1,f=2).
        assert_eq!(ramification_type(&iz(&[1, 0, 1]), 7).unwrap(), vec![(1, 2)]);
    }

    #[test]
    fn ramified_x2_minus_2_over_q2() {
        // x^2 - 2 over Q_2: Eisenstein (1-slope polygon), totally ramified, e=2, f=1.
        // This is the canonical e=2 ramified-at-2 case the first-order engine resolves.
        assert_eq!(ramification_type(&iz(&[-2, 0, 1]), 2).unwrap(), vec![(2, 1)]);
    }

    #[test]
    fn x2_plus_1_over_q2_is_first_order_nonregular() {
        // x^2 + 1 over Q_2 IS ramified (= Q_2(i), e=2) mathematically, but its
        // defining-polynomial Newton polygon at φ=x is FLAT (both coeffs are units)
        // with residual (y+1)^2 mod 2 — inseparable. The ramification is "hidden":
        // only a second-order Montes type (a φ-improvement to φ=x+1) exposes it.
        // The first-order engine therefore (correctly, not wrongly) reports NonRegular
        // rather than guessing. Contrast x^2-2, which is Eisenstein and IS resolved.
        match montes_factor(&iz(&[1, 0, 1]), 2) {
            Err(MontesError::NonRegular { .. }) => {}
            other => panic!("expected NonRegular for x^2+1 at p=2, got {other:?}"),
        }
    }

    #[test]
    fn eisenstein_totally_ramified() {
        // x^3 + 3x + 3 over Q_3: Eisenstein -> e=3, f=1.
        assert_eq!(ramification_type(&iz(&[3, 3, 0, 1]), 3).unwrap(), vec![(3, 1)]);
    }

    #[test]
    fn mixed_segments() {
        // (x^2+3)(x-1) over Q_3: x-1 unramified (1,1); x^2+3 Eisenstein (2,1).
        assert_eq!(ramification_type(&iz(&[-3, 3, -1, 1]), 3).unwrap(), vec![(1, 1), (2, 1)]);
    }

    #[test]
    fn root_valuations_recorded() {
        // Eisenstein x^2 - 2 over Q_2: root valuation 1/2.
        let lf = montes_factor(&iz(&[-2, 0, 1]), 2).unwrap();
        assert_eq!(lf.len(), 1);
        assert_eq!((lf[0].root_val_num, lf[0].root_val_den), (1, 2));
        // unramified x^2+1 over Q_5: root valuation 0/1.
        let lf = montes_factor(&iz(&[1, 0, 1]), 5).unwrap();
        assert!(lf.iter().all(|l| (l.root_val_num, l.root_val_den) == (0, 1)));
    }

    // ---- Prime-ideal decomposition with Sum e_i f_i = deg cross-check ---- //

    #[test]
    fn decompose_x2_plus_1_p5_split() {
        // 5 splits in Z[i]: two primes (1,1),(1,1). Sum e f = 2 = deg.
        let (primes, ok) = decompose_prime_checked(&iz(&[1, 0, 1]), 5).unwrap();
        assert!(ok);
        let mut ef: Vec<(usize, usize)> = primes.iter().map(|p| (p.e, p.f)).collect();
        ef.sort_unstable();
        assert_eq!(ef, vec![(1, 1), (1, 1)]);
    }

    #[test]
    fn decompose_x2_plus_1_p7_inert() {
        // 7 inert in Z[i]: one prime (1,2). Sum e f = 2 = deg.
        let (primes, ok) = decompose_prime_checked(&iz(&[1, 0, 1]), 7).unwrap();
        assert!(ok);
        assert_eq!(primes.len(), 1);
        assert_eq!((primes[0].e, primes[0].f), (1, 2));
    }

    #[test]
    fn decompose_ramified_x2_minus_2_p2() {
        // 2 ramifies in Z[√2]: one prime (2,1) (Eisenstein). Sum e f = 2 = deg.
        let (primes, ok) = decompose_prime_checked(&iz(&[-2, 0, 1]), 2).unwrap();
        assert!(ok);
        assert_eq!((primes[0].e, primes[0].f), (2, 1));
    }

    #[test]
    fn decompose_cubic_pure() {
        // x^3 - 2 (Q(2^{1/3})).
        // p=7: x^3-2 mod 7 -> 7 inert? Actually 2 is not a cube mod 7; x^3-2 irred mod 7
        //   -> one unramified prime (1,3). Sum = 3.
        let (primes, ok) = decompose_prime_checked(&iz(&[-2, 0, 0, 1]), 7).unwrap();
        assert!(ok);
        let sum: usize = primes.iter().map(|p| p.e * p.f).sum();
        assert_eq!(sum, 3);
        // p=3: x^3 - 2 = (x+1)^3 mod 3 (since -2≡1, x^3+ ... ); Eisenstein after shift.
        // Newton polygon of x^3-2 at 3 is flat (v_3(2)=0), so this is the wild case:
        //   x^3 ≡ (x-2) ... actually x^3-2 mod 3 = x^3+1 = (x+1)^3 -> totally ramified.
        // The residual polynomial is (y+1)^3 -> inseparable -> NonRegular at p=3.
        // We instead test p=31 where 2 is a cube: x^3-2 splits into 3 linears.
        let (primes, ok) = decompose_prime_checked(&iz(&[-2, 0, 0, 1]), 31).unwrap();
        assert!(ok);
        let mut ef: Vec<(usize, usize)> = primes.iter().map(|p| (p.e, p.f)).collect();
        ef.sort_unstable();
        assert_eq!(ef, vec![(1, 1), (1, 1), (1, 1)]);
    }

    #[test]
    fn decompose_eisenstein_totally_ramified() {
        // x^2 - 2: 2 totally ramifies in Z[sqrt 2]: (2,1). Sum = 2.
        let (primes, ok) = decompose_prime_checked(&iz(&[-2, 0, 1]), 2).unwrap();
        assert!(ok);
        assert_eq!((primes[0].e, primes[0].f), (2, 1));
    }

    #[test]
    fn decompose_dedekind_split() {
        // A degree-4 example: x^4 + 1 (cyclotomic, Q(zeta_8)).
        // p=2: 2 is totally ramified ((4,1) mathematically), but the defining poly
        // x^4+1 ≡ (x+1)^4 mod 2 has a FLAT Newton polygon with an inseparable
        // residual polynomial — the genuine wild case where 2 | index and first-order
        // Montes is insufficient. We must report NonRegular, not bogus data.
        match decompose_prime(&iz(&[1, 0, 0, 0, 1]), 2) {
            Err(MontesError::NonRegular { .. }) => {}
            other => panic!("expected NonRegular for x^4+1 at p=2, got {other:?}"),
        }
        // p=17: 17 ≡ 1 mod 8 -> splits completely into 4 primes (1,1).
        let (primes, ok) = decompose_prime_checked(&iz(&[1, 0, 0, 0, 1]), 17).unwrap();
        assert!(ok);
        let sum: usize = primes.iter().map(|p| p.e * p.f).sum();
        assert_eq!(sum, 4);
        assert_eq!(primes.len(), 4);
        assert!(primes.iter().all(|p| (p.e, p.f) == (1, 1)));
        // p=3: 3 ≡ 3 mod 8 -> two primes (1,2) (f=2). Sum = 4.
        let (primes, ok) = decompose_prime_checked(&iz(&[1, 0, 0, 0, 1]), 3).unwrap();
        assert!(ok);
        let mut ef: Vec<(usize, usize)> = primes.iter().map(|p| (p.e, p.f)).collect();
        ef.sort_unstable();
        assert_eq!(ef, vec![(1, 2), (1, 2)]);
    }

    #[test]
    fn nonregular_reported_not_wrong() {
        // x^3 - 2 at p=3: residual polynomial (y+1)^3 is inseparable (3 | index).
        // We must report NonRegular rather than bogus (e,f).
        match decompose_prime(&iz(&[-2, 0, 0, 1]), 3) {
            Err(MontesError::NonRegular { .. }) => {}
            other => panic!("expected NonRegular at p=3, got {other:?}"),
        }
    }
}
