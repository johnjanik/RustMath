//! Recognize an approximation of an algebraic number as exact algebraic data.
//!
//! Ported from dessin_engine `src/recognize.rs` (rational reconstruction + the
//! `p`-adic-residue → minimal-polynomial LLL search) **plus**
//! `src/exactification.rs::recognize_complex_algebraic` (the high-precision
//! complex-float route). The exact-rational LLL that both rely on is ported
//! locally (dessin_engine `src/lll.rs`) so this module is self-contained.
//!
//! Three routes:
//! - [`recognize_rational`]: rational reconstruction (`a ∈ Q` from `a mod m`),
//!   the "defined over `Q`" / CRT case;
//! - [`recognize_algebraic`]: minimal polynomial from a `p`-adic residue
//!   `a mod m`, via an LLL integer-relation search on `(1, a, …, a^d)`;
//! - [`recognize_complex_algebraic`]: minimal polynomial from a high-precision
//!   complex float `re + i·im`, via LLL on `(1, z, …, z^d)` with the shortness
//!   gate. **This is the Wave-2 interface contract** — keep the signature stable:
//!   `recognize_complex_algebraic(re: f64, im: f64, max_deg: usize) -> Option<Vec<BigInt>>`.
//!
//! Numerical/`p`-adic recognition is heuristic; downstream callers must certify
//! the result (exact back-substitution) before treating it as decided.

use num_bigint::BigInt;
use num_integer::{Integer, Roots};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use rustmath_rationals::Rational;

// ---------------------------------------------------------------------------
// Exact-rational LLL (ported from dessin_engine src/lll.rs)
// ---------------------------------------------------------------------------

type IVec = Vec<BigInt>;

fn to_rat_vec(v: &IVec) -> Vec<BigRational> {
    v.iter().map(|x| BigRational::from(x.clone())).collect()
}

fn dot(a: &[BigRational], b: &[BigRational]) -> BigRational {
    a.iter()
        .zip(b)
        .fold(BigRational::zero(), |acc, (x, y)| acc + x * y)
}

/// Returns `(mu, bnorm)`: Gram–Schmidt coefficients and squared norms `‖b*_i‖²`.
fn gram_schmidt(basis: &[IVec]) -> (Vec<Vec<BigRational>>, Vec<BigRational>) {
    let n = basis.len();
    let mut bstar: Vec<Vec<BigRational>> = Vec::with_capacity(n);
    let mut mu = vec![vec![BigRational::zero(); n]; n];
    let mut bnorm = vec![BigRational::zero(); n];
    for i in 0..n {
        let bi = to_rat_vec(&basis[i]);
        let mut vi = bi.clone();
        for j in 0..i {
            let mij = dot(&bi, &bstar[j]) / &bnorm[j];
            mu[i][j] = mij.clone();
            for k in 0..vi.len() {
                vi[k] = &vi[k] - &(&mij * &bstar[j][k]);
            }
        }
        bnorm[i] = dot(&vi, &vi);
        bstar.push(vi);
    }
    (mu, bnorm)
}

/// Nearest integer to a rational (round half away from zero).
fn round_rat(r: &BigRational) -> BigInt {
    r.round().to_integer()
}

/// `|r| > 1/2`?
fn abs_gt_half(r: &BigRational) -> bool {
    // denom is always positive in a normalized Ratio
    (r.numer().abs() * 2) > *r.denom()
}

/// `a ≥ b`?
fn rat_ge(a: &BigRational, b: &BigRational) -> bool {
    a >= b
}

/// LLL-reduce an integer lattice basis; returns the reduced basis. Tiny lattices
/// (degree + a few columns), so Gram–Schmidt is recomputed each step for
/// clarity — exactness matters here, not speed.
pub fn lll_reduce(mut basis: Vec<IVec>) -> Vec<IVec> {
    let n = basis.len();
    if n <= 1 {
        return basis;
    }
    let delta = BigRational::new(BigInt::from(3), BigInt::from(4));
    let mut k = 1usize;
    while k < n {
        // size-reduce b_k against b_{k-1..0}
        for j in (0..k).rev() {
            let (mu, _) = gram_schmidt(&basis);
            if abs_gt_half(&mu[k][j]) {
                let q = round_rat(&mu[k][j]);
                for t in 0..basis[k].len() {
                    basis[k][t] = &basis[k][t] - &q * &basis[j][t];
                }
            }
        }
        // Lovász condition
        let (mu, bnorm) = gram_schmidt(&basis);
        let mu2 = &mu[k][k - 1] * &mu[k][k - 1];
        let rhs = (&delta - &mu2) * &bnorm[k - 1];
        if rat_ge(&bnorm[k], &rhs) {
            k += 1;
        } else {
            basis.swap(k, k - 1);
            k = if k > 1 { k - 1 } else { 1 };
        }
    }
    basis
}

// ---------------------------------------------------------------------------
// Modular helpers
// ---------------------------------------------------------------------------

/// `a mod m` in `[0, m)`.
pub fn modpos(a: &BigInt, m: &BigInt) -> BigInt {
    let r = a % m;
    if r.is_negative() {
        r + m
    } else {
        r
    }
}

/// `a^e mod m`.
pub fn mod_pow(a: &BigInt, mut e: u32, m: &BigInt) -> BigInt {
    let mut base = modpos(a, m);
    let mut acc = BigInt::one() % m;
    while e > 0 {
        if e & 1 == 1 {
            acc = (&acc * &base) % m;
        }
        base = (&base * &base) % m;
        e >>= 1;
    }
    modpos(&acc, m)
}

/// Modular inverse of `a` mod `m`, if it exists.
pub fn mod_inv(a: &BigInt, m: &BigInt) -> Option<BigInt> {
    let eg = a.extended_gcd(m);
    if eg.gcd != BigInt::one() {
        return None;
    }
    Some(modpos(&eg.x, m))
}

// ---------------------------------------------------------------------------
// Rational reconstruction (CRT / p-adic → Q)
// ---------------------------------------------------------------------------

/// Rational reconstruction: the unique `num/den` with `num/den ≡ a (mod m)` and
/// `|num|, den ≤ √(m/2)`, if it exists.
pub fn rational_reconstruct(a: &BigInt, m: &BigInt) -> Option<(BigInt, BigInt)> {
    let bound = (m / 2u8).sqrt();
    let (mut r0, mut r1) = (m.clone(), modpos(a, m));
    let (mut t0, mut t1) = (BigInt::zero(), BigInt::from(1));
    while r1 > bound {
        let q = &r0 / &r1;
        let r2 = &r0 - &q * &r1;
        r0 = std::mem::replace(&mut r1, r2);
        let t2 = &t0 - &q * &t1;
        t0 = std::mem::replace(&mut t1, t2);
    }
    let (mut num, mut den) = (r1, t1);
    if den.is_negative() {
        num = -num;
        den = -den;
    }
    if den.is_zero() || num.abs() > bound || den > bound || num.gcd(&den) != BigInt::from(1) {
        return None;
    }
    Some((num, den))
}

/// Rational reconstruction as a [`Rational`].
pub fn recognize_rational(a: &BigInt, m: &BigInt) -> Option<Rational> {
    let (num, den) = rational_reconstruct(a, m)?;
    Rational::new(num, den).ok()
}

// ---------------------------------------------------------------------------
// Minimal-polynomial recovery from a p-adic residue
// ---------------------------------------------------------------------------

/// Minimal polynomial (coefficients ascending, primitive, positive leading) of
/// an algebraic number approximated by `a (mod m)`, searching degree `1..=max_deg`.
/// Returns the smallest-degree integer relation found that is genuinely short.
pub fn recognize_algebraic(a: &BigInt, m: &BigInt, max_deg: usize) -> Option<Vec<BigInt>> {
    for d in 1..=max_deg {
        if let Some(poly) = relation_of_degree(a, m, d) {
            return Some(poly);
        }
    }
    None
}

fn relation_of_degree(a: &BigInt, m: &BigInt, d: usize) -> Option<Vec<BigInt>> {
    let dim = d + 2;
    let weight = m.clone(); // penalize a nonzero modular residue
    let mut basis: Vec<Vec<BigInt>> = Vec::with_capacity(dim);
    // rows e_i ++ [W * (a^i mod m)]
    for i in 0..=d {
        let mut row = vec![BigInt::zero(); dim];
        row[i] = BigInt::from(1);
        row[d + 1] = &weight * mod_pow(a, i as u32, m);
        basis.push(row);
    }
    // the modulus row: [0,…,0, W*m]
    let mut u = vec![BigInt::zero(); dim];
    u[d + 1] = &weight * m;
    basis.push(u);

    let reduced = lll_reduce(basis);

    // A genuine degree-`d` relation has tiny coefficients (the min-poly height);
    // a spurious one (true degree > d) has coefficients ≈ m^{1/(d+1)}. Accept
    // only relations comfortably below that, i.e. ≤ m^{1/(d+2)}.
    let thresh = m.nth_root((d as u32) + 2);
    let mut best: Option<Vec<BigInt>> = None;
    let mut best_norm: Option<BigInt> = None;
    for v in &reduced {
        if !v[d + 1].is_zero() {
            continue;
        }
        let coeffs = &v[0..=d];
        if coeffs.iter().all(|c| c.is_zero()) {
            continue;
        }
        if coeffs.iter().any(|c| c.abs() > thresh) {
            continue;
        }
        let norm2: BigInt = coeffs.iter().map(|c| c * c).sum();
        if best_norm.as_ref().map(|b| &norm2 < b).unwrap_or(true) {
            best_norm = Some(norm2);
            best = Some(coeffs.to_vec());
        }
    }
    best.map(|c| normalize_poly(&c))
}

// ---------------------------------------------------------------------------
// Minimal-polynomial recovery from a high-precision complex float
// (interface contract for Wave 2)
// ---------------------------------------------------------------------------

/// Recognize a complex algebraic number `z = re + i·im` of degree `≤ max_deg` by
/// LLL on `(1, z, …, z^d)`; returns the minimal polynomial (integer coeffs,
/// ascending, primitive, positive leading) verified to annihilate `z`
/// numerically, or `None` if no short relation within the degree bound.
///
/// **Interface contract — keep this signature stable (Wave 2 depends on it).**
pub fn recognize_complex_algebraic(re: f64, im: f64, max_deg: usize) -> Option<Vec<BigInt>> {
    let weight = 1e10_f64;
    for d in 1..=max_deg {
        // powers z^i
        let (mut pr, mut pi) = (vec![0.0f64; d + 1], vec![0.0f64; d + 1]);
        pr[0] = 1.0;
        for i in 1..=d {
            pr[i] = pr[i - 1] * re - pi[i - 1] * im;
            pi[i] = pr[i - 1] * im + pi[i - 1] * re;
        }
        // lattice: rows e_i ++ [round(W·Re z^i), round(W·Im z^i)]
        let dim = d + 3;
        let mut basis: Vec<Vec<BigInt>> = Vec::with_capacity(d + 1);
        for i in 0..=d {
            let mut row = vec![BigInt::zero(); dim];
            row[i] = BigInt::from(1);
            row[d + 1] = BigInt::from((weight * pr[i]).round() as i128);
            row[d + 2] = BigInt::from((weight * pi[i]).round() as i128);
            basis.push(row);
        }
        let reduced = lll_reduce(basis);

        // A genuine degree-`d` relation has small coefficients; a spurious one at
        // insufficient degree has coefficients ≈ weight^{1/(d+1)}. Accept only
        // relations comfortably below that (≤ weight^{1/(d+2)}) that annihilate z.
        let coeff_bound = weight.powf(1.0 / ((d + 2) as f64));
        let mut best: Option<(f64, Vec<BigInt>)> = None;
        for v in &reduced {
            let coeffs = &v[0..=d];
            if coeffs.iter().all(|c| c.is_zero()) {
                continue;
            }
            let cf: Vec<f64> = coeffs.iter().map(bigint_to_f64).collect();
            if cf.iter().any(|c| c.abs() > coeff_bound) {
                continue;
            }
            let (mut er, mut ei) = (0.0f64, 0.0f64);
            for (i, &c) in cf.iter().enumerate() {
                er += c * pr[i];
                ei += c * pi[i];
            }
            let resid = (er * er + ei * ei).sqrt();
            let cnorm: f64 = cf.iter().map(|x| x * x).sum::<f64>().sqrt();
            if resid < 1e-4 && best.as_ref().map(|(b, _)| cnorm < *b).unwrap_or(true) {
                best = Some((cnorm, coeffs.to_vec()));
            }
        }
        if let Some((_, c)) = best {
            return Some(normalize_poly(&c));
        }
    }
    None
}

fn bigint_to_f64(x: &BigInt) -> f64 {
    x.to_string().parse::<f64>().unwrap_or(f64::NAN)
}

// ---------------------------------------------------------------------------
// High-precision (rug::Float) complex algebraic recognizer
// ---------------------------------------------------------------------------

/// `rug::Integer` → `num_bigint::BigInt` (via the exact decimal string; these
/// are one-shot recognitions, not a hot loop).
fn rug_int_to_bigint(z: &rug::Integer) -> BigInt {
    z.to_string().parse::<BigInt>().expect("rug Integer decimal parses")
}

/// `num_bigint::BigInt` → exact `rug::Float` at `prec` bits (via decimal string).
fn bigint_to_rug_float(c: &BigInt, prec: u32) -> rug::Float {
    let z = rug::Integer::from_str_radix(&c.to_string(), 10).expect("BigInt decimal parses");
    rug::Float::with_val(prec, &z)
}

/// High-precision analogue of [`recognize_complex_algebraic`]: the identical LLL
/// integer-relation search on `(1, z, …, z^d)`, but ingesting arbitrary-precision
/// `rug::Float` real/imaginary parts and computing every power `z^i` in HP —
/// never truncating to `f64`. Returns the minimal polynomial (integer coeffs,
/// ascending, primitive, positive leading) verified in HP to annihilate `z`, or
/// `None` if no genuinely short relation exists within the degree bound.
///
/// Unlike the `f64` route (whose fixed `W = 1e10` caps recoverable coefficient
/// height at `W^{1/(d+2)} ≈ 46` for `d = 4`), the penalty weight here scales with
/// precision: `W = 2^floor(prec_bits · FRAC)` with `FRAC = 0.6`.
///
/// Why `0.6`: to separate the *true* relation (whose penalty column is only the
/// ½-ulp integer-rounding residue `~ Σ|c_i|/2`, i.e. on the height scale `H`,
/// because `W·|Σ c_i z^i| = 2^{0.6·prec}·2^{-prec+O(1)} = 2^{-0.4·prec} ≪ 1`)
/// from any *spurious* relation (which must pay a penalty `≳ W^{1/(d+1)}`), the
/// weight must be large enough that the acceptance floor `W^{1/(d+2)} =
/// 2^{0.6·prec/(d+2)}` exceeds any realistic height, yet small enough that the
/// scaled residual stays below `1`. `frac ∈ [0.5, 0.7]` all satisfy both; `0.6`
/// centers the trade-off and leaves `~0.4·prec` bits of head-room for the HP
/// residual gate at `2^{-0.8·prec}` (achievable since each `z^i` carries
/// `prec_bits` bits, so `|Σ c_i z^i|` bottoms out near `2^{-prec}`).
pub fn recognize_complex_algebraic_hp(
    re: &rug::Float,
    im: &rug::Float,
    prec_bits: u32,
    max_deg: usize,
) -> Option<Vec<BigInt>> {
    use rug::float::Round;
    use rug::{Float, Integer};

    if prec_bits < 8 || max_deg == 0 {
        return None;
    }

    // W = 2^wexp, an exact power of two.
    let wexp: u32 = ((prec_bits as f64) * 0.6).floor() as u32;
    let w_int: Integer = Integer::from(1) << wexp;
    // Product precision: enough to hold the integer part of W·z^i (≈ wexp bits)
    // plus prec_bits of fractional accuracy, so rounding to the nearest integer
    // is faithful.
    let wp: u32 = prec_bits + wexp + 16;
    let w_float = Float::with_val(wp, &w_int);

    // HP residual acceptance bound: |Σ c_i z^i| < 2^{-0.8·prec_bits}. Compared
    // squared to avoid a sqrt: resid² < 2^{-1.6·prec_bits}.
    let resid_exp: u32 = ((prec_bits as f64) * 0.8).floor() as u32;
    let bound_den: Integer = Integer::from(1) << (2 * resid_exp);
    let bound2 = Float::with_val(prec_bits, 1) / Float::with_val(2 * resid_exp + 2, &bound_den);

    for d in 1..=max_deg {
        // Powers z^i carried as (re, im) Float pairs, all at prec_bits.
        // z^i = z^{i-1}·z ; (a+bi)(re+im·i) = (a·re − b·im) + (a·im + b·re)·i
        let mut pr: Vec<Float> = Vec::with_capacity(d + 1);
        let mut pi: Vec<Float> = Vec::with_capacity(d + 1);
        pr.push(Float::with_val(prec_bits, 1));
        pi.push(Float::with_val(prec_bits, 0));
        for i in 1..=d {
            let a = &pr[i - 1];
            let b = &pi[i - 1];
            let nr = Float::with_val(prec_bits, a * re) - Float::with_val(prec_bits, b * im);
            let ni = Float::with_val(prec_bits, a * im) + Float::with_val(prec_bits, b * re);
            pr.push(nr);
            pi.push(ni);
        }

        // Lattice rows: e_i ++ [round(W·Re z^i), round(W·Im z^i)].
        let dim = d + 3;
        let mut basis: Vec<Vec<BigInt>> = Vec::with_capacity(d + 1);
        for i in 0..=d {
            let mut row = vec![BigInt::zero(); dim];
            row[i] = BigInt::from(1);
            let sr = Float::with_val(wp, &w_float * &pr[i]);
            let si = Float::with_val(wp, &w_float * &pi[i]);
            row[d + 1] = rug_int_to_bigint(&sr.to_integer_round(Round::Nearest)?.0);
            row[d + 2] = rug_int_to_bigint(&si.to_integer_round(Round::Nearest)?.0);
            basis.push(row);
        }

        let reduced = lll_reduce(basis);

        // Genuine relation: coefficients ≤ W^{1/(d+2)} = floor((2^wexp)^{1/(d+2)}).
        let coeff_bound = BigInt::from(2).pow(wexp).nth_root((d as u32) + 2);
        let mut best: Option<(BigInt, Vec<BigInt>)> = None;
        for v in &reduced {
            let coeffs = &v[0..=d];
            if coeffs.iter().all(|c| c.is_zero()) {
                continue;
            }
            if coeffs.iter().any(|c| c.abs() > coeff_bound) {
                continue;
            }
            // Residual |Σ c_i z^i| verified in HP arithmetic.
            let mut er = Float::with_val(prec_bits, 0);
            let mut ei = Float::with_val(prec_bits, 0);
            for (i, c) in coeffs.iter().enumerate() {
                let cf = bigint_to_rug_float(c, prec_bits);
                er += Float::with_val(prec_bits, &cf * &pr[i]);
                ei += Float::with_val(prec_bits, &cf * &pi[i]);
            }
            let resid2 = Float::with_val(prec_bits, &er * &er) + Float::with_val(prec_bits, &ei * &ei);
            if resid2 < bound2 {
                let cnorm2: BigInt = coeffs.iter().map(|c| c * c).sum();
                if best.as_ref().map(|(b, _)| &cnorm2 < b).unwrap_or(true) {
                    best = Some((cnorm2, coeffs.to_vec()));
                }
            }
        }
        if let Some((_, c)) = best {
            return Some(normalize_poly(&c));
        }
    }
    None
}

/// Make primitive (content 1) and force a positive leading coefficient.
fn normalize_poly(coeffs: &[BigInt]) -> Vec<BigInt> {
    let mut g = BigInt::zero();
    for c in coeffs {
        g = g.gcd(c);
    }
    if g.is_zero() {
        return coeffs.to_vec();
    }
    let mut out: Vec<BigInt> = coeffs.iter().map(|c| c / &g).collect();
    // leading (highest-degree nonzero) coefficient positive
    if let Some(lead) = out.iter().rev().find(|c| !c.is_zero()) {
        if lead.is_negative() {
            out.iter_mut().for_each(|c| *c = -&*c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iv(v: &[i64]) -> Vec<BigInt> {
        v.iter().map(|&x| BigInt::from(x)).collect()
    }

    // --- exact-rational LLL (from dessin_engine src/lll.rs tests) ---

    #[test]
    fn lll_reduces_to_shorter_basis() {
        let basis = vec![iv(&[1, 1, 1]), iv(&[-1, 0, 2]), iv(&[3, 5, 6])];
        let red = lll_reduce(basis);
        let norm2 = |v: &Vec<BigInt>| v.iter().map(|x| x * x).sum::<BigInt>();
        for v in &red {
            assert!(norm2(v) <= BigInt::from(70));
        }
        assert!(norm2(&red[0]) <= BigInt::from(6));
    }

    #[test]
    fn lll_finds_short_planted_vector() {
        let basis = vec![iv(&[1, 0, 0]), iv(&[1000, 1, 0]), iv(&[2000, 0, 1])];
        let red = lll_reduce(basis);
        let has_unit = red
            .iter()
            .any(|v| v == &iv(&[1, 0, 0]) || v == &iv(&[-1, 0, 0]));
        assert!(has_unit);
    }

    // --- rational reconstruction (CRT residue) ---

    #[test]
    fn reconstructs_rational_from_crt_residue() {
        // 3/5 mod 7^10
        let m = BigInt::from(7).pow(10);
        let a = modpos(
            &(&BigInt::from(3) * mod_inv(&BigInt::from(5), &m).unwrap()),
            &m,
        );
        assert_eq!(
            recognize_rational(&a, &m).unwrap(),
            Rational::new(3, 5).unwrap()
        );
    }

    #[test]
    fn reconstructs_negative_rational() {
        // -7/11 mod 13^12
        let m = BigInt::from(13).pow(12);
        let a = modpos(
            &(&BigInt::from(-7) * mod_inv(&BigInt::from(11), &m).unwrap()),
            &m,
        );
        assert_eq!(
            recognize_rational(&a, &m).unwrap(),
            Rational::new(-7, 11).unwrap()
        );
    }

    // --- minimal polynomial from a p-adic residue ---

    /// Hensel-lift a root of `x^2 - 2` starting at `3 (mod 7)` to `7^target`.
    fn padic_sqrt2(target: u32) -> (BigInt, BigInt) {
        let p = BigInt::from(7);
        let mut x = BigInt::from(3); // 3^2 = 9 ≡ 2 (mod 7)
        let mut prec = 1u32;
        while prec < target {
            let newprec = (prec * 2).min(target);
            let m = p.pow(newprec);
            let f = modpos(&(&x * &x - 2), &m);
            let df = modpos(&(2 * &x), &m);
            let inv = mod_inv(&df, &m).unwrap();
            x = modpos(&(&x - &f * &inv), &m);
            prec = newprec;
        }
        (x, p.pow(target))
    }

    #[test]
    fn recognizes_sqrt2_from_padic_residue() {
        let (x, m) = padic_sqrt2(40);
        // sanity: x^2 ≡ 2 (mod m)
        assert_eq!(modpos(&(&x * &x), &m), BigInt::from(2));
        let poly = recognize_algebraic(&x, &m, 4).unwrap();
        assert_eq!(poly, iv(&[-2, 0, 1])); // x^2 - 2
    }

    // --- minimal polynomial from a high-precision complex float
    //     (interface contract) ---

    #[test]
    fn recognizes_sqrt2_from_float() {
        // Known quadratic irrational from a high-precision float.
        let p = recognize_complex_algebraic(2.0_f64.sqrt(), 0.0, 4).unwrap();
        assert_eq!(p, iv(&[-2, 0, 1])); // x^2 - 2
    }

    #[test]
    fn recognizes_golden_ratio_and_cbrt2() {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        assert_eq!(
            recognize_complex_algebraic(phi, 0.0, 4).unwrap(),
            iv(&[-1, -1, 1])
        );
        let cbrt2 = 2.0_f64.powf(1.0 / 3.0);
        assert_eq!(
            recognize_complex_algebraic(cbrt2, 0.0, 4).unwrap(),
            iv(&[-2, 0, 0, 1])
        );
    }

    #[test]
    fn recognizes_imaginary_unit() {
        // i is a root of x^2 + 1
        let p = recognize_complex_algebraic(0.0, 1.0, 4).unwrap();
        assert_eq!(p, iv(&[1, 0, 1]));
    }

    // --- high-precision (rug::Float) complex algebraic recognition ---
    //
    // Oracle: the irreducible quartic  x^4 - 3x^3 + 1000000x^2 - 7x + 500000
    // (height 1e6), one complex root z ≈ 2.75e-6 - 0.70710695796i computed in
    // mpmath at 220 decimal digits, |p(z)| there ≈ 5.8e-216. The (re, im)
    // decimal strings below are that root to 170 significant digits (> 512 bits).
    // Ascending primitive minpoly: [500000, -7, 1000000, -3, 1].

    const Q4_RE: &str = "0.0000027500023750363750724695832213186932200485078396289406597418941003714852989093892143400447457337702240700325576957531002453391864652344340588916338452575138581802281855524";
    const Q4_IM: &str = "-0.70710695796388363047596972716231546307955287477076945830204133661959833849986102308537200699381945319621505998216738913869748026937548558275334087721642369874561214002376";

    fn f512(s: &str) -> rug::Float {
        rug::Float::with_val(512, rug::Float::parse(s).expect("parse 512-bit float"))
    }

    /// Independent HP recomputation of |Σ c_i z^i| (squared), used to certify the
    /// accepted residual — derived from the coefficients, not from the recognizer.
    fn hp_resid2(coeffs: &[BigInt], re: &rug::Float, im: &rug::Float, prec: u32) -> rug::Float {
        use rug::Float;
        let mut zr = Float::with_val(prec, 1);
        let mut zi = Float::with_val(prec, 0);
        let mut sr = Float::with_val(prec, 0);
        let mut si = Float::with_val(prec, 0);
        for c in coeffs {
            let cf = bigint_to_rug_float(c, prec);
            sr += Float::with_val(prec, &cf * &zr);
            si += Float::with_val(prec, &cf * &zi);
            let nzr = Float::with_val(prec, &zr * re) - Float::with_val(prec, &zi * im);
            let nzi = Float::with_val(prec, &zr * im) + Float::with_val(prec, &zi * re);
            zr = nzr;
            zi = nzi;
        }
        Float::with_val(prec, &sr * &sr) + Float::with_val(prec, &si * &si)
    }

    #[test]
    fn hp_recognizes_height_1e6_quartic_where_f64_fails() {
        let re = f512(Q4_RE);
        let im = f512(Q4_IM);
        let expected = iv(&[500000, -7, 1000000, -3, 1]);

        // The HP recognizer recovers the byte-identical minimal polynomial.
        let got = recognize_complex_algebraic_hp(&re, &im, 512, 4)
            .expect("HP recognizer must find the height-1e6 quartic");
        assert_eq!(got, expected);

        // The f64 route CANNOT: its W = 1e10 caps recoverable height at
        // 1e10^(1/6) ≈ 46, far below the true coefficients (~1e6). It must not
        // return the correct minpoly (it returns None or a spurious relation).
        let f64_ans = recognize_complex_algebraic(re.to_f64(), im.to_f64(), 4);
        assert_ne!(
            f64_ans,
            Some(expected.clone()),
            "f64 route unexpectedly recovered a height-1e6 minpoly"
        );

        // Certify the accepted residual is genuinely tiny (independently of the
        // recognizer): |p(z)|^2 in 512-bit HP must be below (2^-400)^2 = 2^-800.
        let resid2 = hp_resid2(&got, &re, &im, 512);
        let tiny2 = rug::Float::with_val(512, 1)
            / rug::Float::with_val(802, &(rug::Integer::from(1) << 800));
        assert!(
            resid2 < tiny2,
            "accepted residual not tiny: resid^2 = {:?}",
            resid2
        );
    }

    #[test]
    fn hp_rejects_transcendental_input() {
        // ln(2) + i·π to 512 bits is not algebraic of degree ≤ 4.
        const LN2: &str = "0.69314718055994530941723212145817656807550013436025525412068000949339362196969471560586332699641868754200148102057068573368552023575813055703267075163507596193072757082837";
        const PI: &str = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865132823066470938446095505822317253594081284811174502841027019";
        let re = f512(LN2);
        let im = f512(PI);
        assert_eq!(recognize_complex_algebraic_hp(&re, &im, 512, 4), None);
    }
}
