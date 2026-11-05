//! Factorization of polynomials over finite fields (MAGMA Handbook ch. 21).
//!
//! MAGMA source: Chapter 21 §21.2.2 / §21.5 (`IsIrreducible`, `Factorization`,
//! `IrreduciblePolynomial`, `RandomIrreduciblePolynomial`, `Roots`).
//!
//! Two layers:
//!   * a lightweight `F_p[x]` layer on `Vec<Integer>` (little-endian, coeffs in
//!     `[0, p)`) used to **bootstrap** field construction — it must not build a
//!     [`crate::FiniteField`] itself, to avoid recursion; and
//!   * a general [`FqPoly`] layer over a [`crate::FiniteField`] `GF(q)` providing
//!     Cantor–Zassenhaus factorization: **distinct-degree factorization** (DDF)
//!     followed by randomized **equal-degree factorization** (EDF), plus
//!     square-free factorization (handling `char p` p-th powers), a Rabin
//!     irreducibility test, and root finding.

use std::fmt;

use rustmath_core::{Field, MathError, NumericConversion, Result, Ring};
use rustmath_integers::prime::factor as int_factor;
use rustmath_integers::Integer;

use crate::finite_field::{FiniteField, FiniteFieldElement};

// ===========================================================================
// Layer 1: F_p[x] on Vec<Integer> (bootstrap; no FiniteField required)
// ===========================================================================

fn redp(x: Integer, p: &Integer) -> Integer {
    let r = x % p.clone();
    if r.signum() < 0 {
        r + p.clone()
    } else {
        r
    }
}

fn fp_inv(a: &Integer, p: &Integer) -> Integer {
    let (g, x, _) = redp(a.clone(), p).extended_gcd(p);
    debug_assert!(g.is_one(), "not invertible mod p");
    redp(x, p)
}

fn fp_trim(mut v: Vec<Integer>) -> Vec<Integer> {
    while v.last().map(|c| c.is_zero()).unwrap_or(false) {
        v.pop();
    }
    v
}

fn fp_deg(v: &[Integer]) -> Option<usize> {
    if v.is_empty() {
        None
    } else {
        Some(v.len() - 1)
    }
}

fn fp_mul(a: &[Integer], b: &[Integer], p: &Integer) -> Vec<Integer> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![Integer::zero(); a.len() + b.len() - 1];
    for (i, ai) in a.iter().enumerate() {
        if ai.is_zero() {
            continue;
        }
        for (j, bj) in b.iter().enumerate() {
            out[i + j] = redp(out[i + j].clone() + ai.clone() * bj.clone(), p);
        }
    }
    fp_trim(out)
}

fn fp_sub(a: &[Integer], b: &[Integer], p: &Integer) -> Vec<Integer> {
    let n = a.len().max(b.len());
    let mut out = vec![Integer::zero(); n];
    for i in 0..n {
        let ai = a.get(i).cloned().unwrap_or_else(Integer::zero);
        let bi = b.get(i).cloned().unwrap_or_else(Integer::zero);
        out[i] = redp(ai - bi, p);
    }
    fp_trim(out)
}

/// Divide `a` by `b` (b nonzero) over `F_p`, returning `(quotient, remainder)`.
fn fp_div_rem(a: &[Integer], b: &[Integer], p: &Integer) -> (Vec<Integer>, Vec<Integer>) {
    let mut r = fp_trim(a.to_vec());
    let b = fp_trim(b.to_vec());
    let bdeg = fp_deg(&b).expect("division by zero polynomial");
    let binv = fp_inv(&b[bdeg], p);
    if fp_deg(&r).map(|d| d < bdeg).unwrap_or(true) {
        return (Vec::new(), r);
    }
    let mut q = vec![Integer::zero(); fp_deg(&r).unwrap() - bdeg + 1];
    while let Some(rdeg) = fp_deg(&r) {
        if rdeg < bdeg {
            break;
        }
        let shift = rdeg - bdeg;
        let factor = redp(r[rdeg].clone() * binv.clone(), p);
        q[shift] = factor.clone();
        // r -= factor * x^shift * b
        for i in 0..b.len() {
            let idx = i + shift;
            r[idx] = redp(r[idx].clone() - factor.clone() * b[i].clone(), p);
        }
        r = fp_trim(r);
    }
    (fp_trim(q), r)
}

fn fp_gcd(a: &[Integer], b: &[Integer], p: &Integer) -> Vec<Integer> {
    let mut a = fp_trim(a.to_vec());
    let mut b = fp_trim(b.to_vec());
    while !b.is_empty() {
        let (_, r) = fp_div_rem(&a, &b, p);
        a = b;
        b = r;
    }
    fp_make_monic(a, p)
}

fn fp_make_monic(v: Vec<Integer>, p: &Integer) -> Vec<Integer> {
    let v = fp_trim(v);
    match fp_deg(&v) {
        None => v,
        Some(d) => {
            let inv = fp_inv(&v[d], p);
            fp_trim(v.iter().map(|c| redp(c.clone() * inv.clone(), p)).collect())
        }
    }
}

/// `base^exp mod modulus` in `F_p[x]`.
fn fp_pow_mod(base: &[Integer], exp: &Integer, modulus: &[Integer], p: &Integer) -> Vec<Integer> {
    let (_, mut b) = fp_div_rem(base, modulus, p);
    let mut result = vec![Integer::one()]; // 1
    let mut e = exp.clone();
    let two = Integer::from(2);
    while e > Integer::zero() {
        if redp(e.clone(), &two).is_one() {
            result = fp_div_rem(&fp_mul(&result, &b, p), modulus, p).1;
        }
        e = e / two.clone();
        if e > Integer::zero() {
            b = fp_div_rem(&fp_mul(&b, &b, p), modulus, p).1;
        }
    }
    fp_trim(result)
}

/// Rabin irreducibility test for a monic `f` over `F_p` (as `Vec<Integer>`).
pub fn is_irreducible_fp(f: &[Integer], p: &Integer) -> bool {
    let f = fp_trim(f.to_vec());
    let m = match fp_deg(&f) {
        None => return false,
        Some(0) => return false,
        Some(1) => return true,
        Some(d) => d,
    };
    let x = vec![Integer::zero(), Integer::one()];
    // x^{p^k} mod f for k = 1..=m via repeated p-th powering.
    let mut xpk = x.clone();
    // For each prime r | m: gcd(x^{p^{m/r}} - x, f) must be 1.
    let m_int = Integer::from(m as i64);
    let prime_divisors: Vec<usize> = int_factor(&m_int)
        .into_iter()
        .filter_map(|(pr, _)| pr.to_usize())
        .collect();
    // We need x^{p^{m/r}}. Compute the whole ladder up to m, checking as we go.
    let mut current = x.clone();
    for k in 1..=m {
        current = fp_pow_mod(&current, p, &f, p); // now current = x^{p^k} mod f
        if k == m {
            xpk = current.clone();
        }
        if prime_divisors.iter().any(|&r| m / r == k) {
            let diff = fp_sub(&current, &x, p);
            let g = fp_gcd(&diff, &f, p);
            if fp_deg(&g).map(|d| d >= 1).unwrap_or(false) {
                return false;
            }
        }
    }
    // Final: x^{p^m} == x mod f.
    fp_sub(&xpk, &x, p).is_empty()
}

/// Find a monic irreducible polynomial of degree `n` over `F_p`, returned as a
/// little-endian `Vec<Integer>` of length `n + 1`. Systematic lexicographic
/// search over monic candidates (one is guaranteed to exist).
pub fn find_irreducible(p: &Integer, n: usize) -> Result<Vec<Integer>> {
    if n == 0 {
        return Err(MathError::InvalidArgument("degree must be >= 1".into()));
    }
    if n == 1 {
        return Ok(vec![Integer::zero(), Integer::one()]);
    }
    // Enumerate the n low coefficients as a base-p counter; leading coeff = 1.
    let mut low = vec![Integer::zero(); n];
    loop {
        let mut cand = low.clone();
        cand.push(Integer::one());
        if is_irreducible_fp(&cand, p) {
            return Ok(cand);
        }
        // increment base-p counter over `low`
        let mut i = 0;
        loop {
            if i == n {
                return Err(MathError::NotSupported(format!(
                    "no irreducible of degree {n} found over F_{p} (search exhausted)"
                )));
            }
            low[i] = low[i].clone() + Integer::one();
            if low[i] < *p {
                break;
            }
            low[i] = Integer::zero();
            i += 1;
        }
    }
}

/// Random monic irreducible polynomial of degree `n` over `F_p` (MAGMA
/// `RandomIrreduciblePolynomial`): draws random monic candidates and tests
/// them with the Rabin irreducibility test ([`is_irreducible_fp`]).
/// Deterministic for a given `seed`. A random monic polynomial of degree `n`
/// over `F_q` is irreducible with probability roughly `1/n`, so the expected
/// number of trials is about `n`.
pub fn random_irreducible(p: &Integer, n: usize, seed: u64) -> Result<Vec<Integer>> {
    if n == 0 {
        return Err(MathError::InvalidArgument("degree must be >= 1".into()));
    }
    if n == 1 {
        // x + c for a random c is always irreducible.
        let mut rng = Rng(seed);
        let c = redp(Integer::from(rng.next_u64() as i64).abs(), p);
        return Ok(vec![c, Integer::one()]);
    }
    let mut rng = Rng(seed ^ 0xA5A5_5A5A_0F0F_F0F0);
    // Failure probability after 512*n trials is ~ (1 - 1/n)^(512 n) ≈ e^-512.
    for _ in 0..(512 * n) {
        let mut cand: Vec<Integer> = (0..n)
            .map(|_| redp(Integer::from(rng.next_u64() as i64).abs(), p))
            .collect();
        cand.push(Integer::one());
        if is_irreducible_fp(&cand, p) {
            return Ok(cand);
        }
    }
    Err(MathError::NotSupported(format!(
        "no irreducible of degree {n} found over F_{p} within the random search bound"
    )))
}

// ===========================================================================
// Layer 2: FqPoly over a FiniteField GF(q)  (Cantor–Zassenhaus)
// ===========================================================================

/// A univariate polynomial over a [`FiniteField`] `GF(q)`, little-endian and
/// trimmed (no trailing zero coefficients; the empty vector is the zero poly).
#[derive(Clone)]
pub struct FqPoly {
    coeffs: Vec<FiniteFieldElement>,
    field: FiniteField,
}

impl fmt::Debug for FqPoly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FqPoly(deg={:?}) over {}", self.degree(), self.field)
    }
}

impl FqPoly {
    /// Build from little-endian coefficients (trims trailing zeros).
    pub fn new(field: FiniteField, coeffs: Vec<FiniteFieldElement>) -> Self {
        let mut p = FqPoly { coeffs, field };
        p.trim();
        p
    }

    /// The zero polynomial.
    pub fn zero(field: FiniteField) -> Self {
        FqPoly {
            coeffs: Vec::new(),
            field,
        }
    }

    /// The constant polynomial `1`.
    pub fn one(field: FiniteField) -> Self {
        let one = field.one();
        FqPoly {
            coeffs: vec![one],
            field,
        }
    }

    /// The monomial `x` (`X`).
    pub fn x(field: FiniteField) -> Self {
        FqPoly {
            coeffs: vec![field.zero(), field.one()],
            field,
        }
    }

    fn trim(&mut self) {
        while self.coeffs.last().map(|c| c.is_zero()).unwrap_or(false) {
            self.coeffs.pop();
        }
    }

    /// Degree, or `None` for the zero polynomial.
    pub fn degree(&self) -> Option<usize> {
        if self.coeffs.is_empty() {
            None
        } else {
            Some(self.coeffs.len() - 1)
        }
    }

    /// Whether this is the zero polynomial.
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// Coefficients (little-endian, trimmed).
    pub fn coefficients(&self) -> &[FiniteFieldElement] {
        &self.coeffs
    }

    fn leading(&self) -> FiniteFieldElement {
        self.coeffs
            .last()
            .cloned()
            .unwrap_or_else(|| self.field.zero())
    }

    /// Make monic (divide by the leading coefficient).
    pub fn make_monic(&self) -> FqPoly {
        if self.is_zero() {
            return self.clone();
        }
        let inv = self.leading().inverse().unwrap();
        let coeffs = self.coeffs.iter().map(|c| c.clone() * inv.clone()).collect();
        FqPoly::new(self.field.clone(), coeffs)
    }

    fn add(&self, other: &FqPoly) -> FqPoly {
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let a = self
                .coeffs
                .get(i)
                .cloned()
                .unwrap_or_else(|| self.field.zero());
            let b = other
                .coeffs
                .get(i)
                .cloned()
                .unwrap_or_else(|| self.field.zero());
            out.push(a + b);
        }
        FqPoly::new(self.field.clone(), out)
    }

    fn sub(&self, other: &FqPoly) -> FqPoly {
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let a = self
                .coeffs
                .get(i)
                .cloned()
                .unwrap_or_else(|| self.field.zero());
            let b = other
                .coeffs
                .get(i)
                .cloned()
                .unwrap_or_else(|| self.field.zero());
            out.push(a - b);
        }
        FqPoly::new(self.field.clone(), out)
    }

    fn mul(&self, other: &FqPoly) -> FqPoly {
        if self.is_zero() || other.is_zero() {
            return FqPoly::zero(self.field.clone());
        }
        let mut out = vec![self.field.zero(); self.coeffs.len() + other.coeffs.len() - 1];
        for (i, a) in self.coeffs.iter().enumerate() {
            if a.is_zero() {
                continue;
            }
            for (j, b) in other.coeffs.iter().enumerate() {
                out[i + j] = out[i + j].clone() + a.clone() * b.clone();
            }
        }
        FqPoly::new(self.field.clone(), out)
    }

    /// Division with remainder: `self = q*other + r`, `deg r < deg other`.
    pub fn div_rem(&self, other: &FqPoly) -> (FqPoly, FqPoly) {
        assert!(!other.is_zero(), "division by zero polynomial");
        let bdeg = other.degree().unwrap();
        let binv = other.leading().inverse().unwrap();
        let mut r = self.clone();
        let mut q = vec![self.field.zero(); self.coeffs.len().saturating_sub(bdeg).max(1)];
        while let Some(rdeg) = r.degree() {
            if rdeg < bdeg {
                break;
            }
            let shift = rdeg - bdeg;
            let factor = r.leading() * binv.clone();
            if shift >= q.len() {
                q.resize(shift + 1, self.field.zero());
            }
            q[shift] = factor.clone();
            // r -= factor * x^shift * other
            let mut sub = vec![self.field.zero(); shift];
            for c in &other.coeffs {
                sub.push(factor.clone() * c.clone());
            }
            r = r.sub(&FqPoly::new(self.field.clone(), sub));
        }
        (FqPoly::new(self.field.clone(), q), r)
    }

    fn rem(&self, other: &FqPoly) -> FqPoly {
        self.div_rem(other).1
    }

    /// Monic gcd.
    pub fn gcd(&self, other: &FqPoly) -> FqPoly {
        let mut a = self.clone();
        let mut b = other.clone();
        while !b.is_zero() {
            let r = a.rem(&b);
            a = b;
            b = r;
        }
        if a.is_zero() {
            a
        } else {
            a.make_monic()
        }
    }

    /// Formal derivative.
    pub fn derivative(&self) -> FqPoly {
        if self.coeffs.len() <= 1 {
            return FqPoly::zero(self.field.clone());
        }
        let mut out = Vec::with_capacity(self.coeffs.len() - 1);
        for i in 1..self.coeffs.len() {
            let scalar = self.field.from_int(Integer::from(i as i64));
            out.push(scalar * self.coeffs[i].clone());
        }
        FqPoly::new(self.field.clone(), out)
    }

    /// `self^exp mod modulus`.
    pub fn pow_mod(&self, exp: &Integer, modulus: &FqPoly) -> FqPoly {
        let mut result = FqPoly::one(self.field.clone()).rem(modulus);
        let mut base = self.rem(modulus);
        let mut e = exp.clone();
        let two = Integer::from(2);
        while e > Integer::zero() {
            if (e.clone() % two.clone()).is_one() {
                result = result.mul(&base).rem(modulus);
            }
            e = e / two.clone();
            if e > Integer::zero() {
                base = base.mul(&base).rem(modulus);
            }
        }
        result
    }

    /// The `p`-th root, valid when every nonzero coefficient sits at an exponent
    /// divisible by `p` (i.e. `self` is a `p`-th power). Uses the inverse
    /// Frobenius on coefficients.
    fn pth_root(&self) -> FqPoly {
        let p = self.field.characteristic().clone();
        let p_usize = p.to_usize().expect("characteristic too large for p-th root");
        let k = self.field.degree();
        let mut out = vec![self.field.zero(); self.coeffs.len() / p_usize + 1];
        for (i, c) in self.coeffs.iter().enumerate() {
            if c.is_zero() {
                continue;
            }
            // inverse Frobenius: c^{1/p} = c^{p^{k-1}}
            let root = c.frobenius_pow(k - 1);
            out[i / p_usize] = root;
        }
        FqPoly::new(self.field.clone(), out)
    }
}

impl PartialEq for FqPoly {
    fn eq(&self, other: &Self) -> bool {
        self.field.same_field(other.field()) && self.coeffs == other.coeffs
    }
}

impl FqPoly {
    fn field(&self) -> &FiniteField {
        &self.field
    }
}

// -- deterministic PRNG (splitmix64) so factorization is reproducible ---------

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn field_elt(&mut self, field: &FiniteField) -> FiniteFieldElement {
        let n = field.degree();
        let p = field.characteristic().clone();
        let coeffs = (0..n)
            .map(|_| Integer::from(self.next_u64() as i64).abs() % p.clone())
            .collect();
        field.element(coeffs)
    }
}

// -- Square-free / distinct-degree / equal-degree factorization ---------------

/// Square-free factorization: returns `(g_i, i)` with `f = prod g_i^i`, each
/// `g_i` monic and square-free. Handles `char p` p-th powers.
pub fn square_free_factorization(f: &FqPoly) -> Vec<(FqPoly, usize)> {
    let field = f.field().clone();
    let p_usize = field.characteristic().to_usize();
    let mut result: Vec<(FqPoly, usize)> = Vec::new();
    let f = f.make_monic();
    if f.degree().unwrap_or(0) == 0 {
        return result;
    }
    let fp = f.derivative();
    if fp.is_zero() {
        // f = g(x)^p
        let g = f.pth_root();
        let p = p_usize.expect("char too large");
        for (h, e) in square_free_factorization(&g) {
            result.push((h, e * p));
        }
        return result;
    }
    let mut c = f.gcd(&fp);
    let mut w = f.div_rem(&c).0; // f / c  (square-free part, exponents not div by p)
    let mut i = 1usize;
    while w.degree().unwrap_or(0) >= 1 {
        let y = w.gcd(&c);
        let z = w.div_rem(&y).0;
        if z.degree().unwrap_or(0) >= 1 {
            result.push((z.make_monic(), i));
        }
        i += 1;
        w = y;
        c = c.div_rem(&c.gcd(&w)).0.make_monic(); // c := c / gcd(c,w) keeps c shrinking
        // NOTE: standard algorithm sets c := c / y; we recompute robustly below.
    }
    // The residual c collects the p-th-power part.
    if c.degree().unwrap_or(0) >= 1 {
        let g = c.pth_root();
        let p = p_usize.expect("char too large");
        for (h, e) in square_free_factorization(&g) {
            result.push((h, e * p));
        }
    }
    result
}

/// Distinct-degree factorization of a **monic square-free** `f`: returns
/// `(g_d, d)` where `g_d` is the product of all monic irreducible factors of
/// `f` of degree `d`.
pub fn distinct_degree_factorization(f: &FqPoly) -> Vec<(FqPoly, usize)> {
    let field = f.field().clone();
    let q = field.order();
    let x = FqPoly::x(field.clone());
    let mut result = Vec::new();
    let mut fstar = f.make_monic();
    let mut xqi = x.clone(); // will hold x^{q^i} mod fstar
    let mut d = 1usize;
    while fstar.degree().unwrap_or(0) >= 2 * d {
        xqi = xqi.pow_mod(&q, &fstar); // x^{q^d} mod fstar
        let g = fstar.gcd(&xqi.sub(&x));
        if g.degree().unwrap_or(0) >= 1 {
            let gm = g.make_monic();
            result.push((gm.clone(), d));
            fstar = fstar.div_rem(&gm).0;
            xqi = xqi.rem(&fstar);
        }
        d += 1;
    }
    if fstar.degree().unwrap_or(0) >= 1 {
        let deg = fstar.degree().unwrap();
        result.push((fstar.make_monic(), deg));
    }
    result
}

/// Equal-degree factorization (Cantor–Zassenhaus): split a **monic square-free**
/// `f` whose irreducible factors all have degree `d` into those factors.
pub fn equal_degree_factorization(f: &FqPoly, d: usize) -> Vec<FqPoly> {
    let field = f.field().clone();
    let mut out = Vec::new();
    // seed the PRNG deterministically from f
    let seed = 0x1234_5678_9abc_def0u64
        ^ (f.degree().unwrap_or(0) as u64).wrapping_mul(0x100000001b3)
        ^ field.order().to_u64().unwrap_or(7);
    let mut rng = Rng(seed);
    edf_rec(f, d, &mut rng, &mut out);
    out.sort_by(|a, b| a.coeffs_key().cmp(&b.coeffs_key()));
    out
}

impl FqPoly {
    /// A comparable key (little-endian coefficient sequences) for deterministic
    /// ordering of factor lists.
    fn coeffs_key(&self) -> Vec<Vec<Integer>> {
        self.coeffs.iter().map(|c| c.eltseq().to_vec()).collect()
    }
}

fn edf_rec(f: &FqPoly, d: usize, rng: &mut Rng, out: &mut Vec<FqPoly>) {
    let field = f.field().clone();
    let deg = f.degree().unwrap_or(0);
    if deg == 0 {
        return;
    }
    if deg == d {
        out.push(f.make_monic());
        return;
    }
    let q = field.order();
    let p = field.characteristic().clone();
    loop {
        // random polynomial of degree < deg
        let mut coeffs: Vec<FiniteFieldElement> = (0..deg).map(|_| rng.field_elt(&field)).collect();
        // ensure nonconstant
        if coeffs.len() > 1 {
            coeffs[1] = coeffs[1].clone() + field.one();
        }
        let a = FqPoly::new(field.clone(), coeffs);
        if a.degree().unwrap_or(0) == 0 {
            continue;
        }
        let g0 = f.gcd(&a);
        if g0.degree().unwrap_or(0) >= 1 && g0.degree().unwrap() < deg {
            edf_rec(&g0, d, rng, out);
            edf_rec(&f.div_rem(&g0).0, d, rng, out);
            return;
        }
        let g = if p == Integer::from(2) {
            // char 2: trace map T = a + a^q + a^{q^2} + ... + a^{q^{d-1}} (over F_2, s*d terms)
            let s = field.degree(); // q = 2^s
            let mut t = a.clone();
            let mut cur = a.clone();
            for _ in 1..(s * d) {
                cur = cur.pow_mod(&Integer::from(2), f); // square mod f
                t = t.add(&cur).rem(f);
            }
            f.gcd(&t)
        } else {
            // odd char: b = a^{(q^d - 1)/2} mod f; gcd(b - 1, f)
            let exp = (q.pow(d as u32) - Integer::one()) / Integer::from(2);
            let b = a.pow_mod(&exp, f);
            let b_minus_1 = b.sub(&FqPoly::one(field.clone()));
            f.gcd(&b_minus_1)
        };
        let gdeg = g.degree().unwrap_or(0);
        if gdeg >= 1 && gdeg < deg {
            edf_rec(&g, d, rng, out);
            edf_rec(&f.div_rem(&g).0, d, rng, out);
            return;
        }
    }
}

/// Full factorization of `f` over `GF(q)`: returns `(g_i, e_i)` with distinct
/// monic irreducible `g_i` and `f = c * prod g_i^{e_i}` (the unit `c` is
/// discarded). Combines square-free + distinct-degree + equal-degree steps.
pub fn factor(f: &FqPoly) -> Vec<(FqPoly, usize)> {
    let mut result = Vec::new();
    for (sqfree, mult) in square_free_factorization(f) {
        for (g, d) in distinct_degree_factorization(&sqfree) {
            for irr in equal_degree_factorization(&g, d) {
                result.push((irr.make_monic(), mult));
            }
        }
    }
    result.sort_by(|a, b| a.0.coeffs_key().cmp(&b.0.coeffs_key()));
    result
}

/// Rabin irreducibility test for a monic `f` over `GF(q)`.
pub fn is_irreducible(f: &FqPoly) -> bool {
    let field = f.field().clone();
    let m = match f.degree() {
        None | Some(0) => return false,
        Some(1) => return true,
        Some(d) => d,
    };
    let q = field.order();
    let x = FqPoly::x(field.clone());
    let m_int = Integer::from(m as i64);
    let prime_divisors: Vec<usize> = int_factor(&m_int)
        .into_iter()
        .filter_map(|(pr, _)| pr.to_usize())
        .collect();
    let mut current = x.clone();
    let mut xqm = x.clone();
    for k in 1..=m {
        current = current.pow_mod(&q, f); // x^{q^k} mod f
        if k == m {
            xqm = current.clone();
        }
        if prime_divisors.iter().any(|&r| m / r == k) {
            let g = f.gcd(&current.sub(&x));
            if g.degree().unwrap_or(0) >= 1 {
                return false;
            }
        }
    }
    xqm.sub(&x).is_zero()
}

/// Roots of `f` in its coefficient field `GF(q)`, as a de-duplicated list.
pub fn roots(f: &FqPoly) -> Vec<FiniteFieldElement> {
    let mut out = Vec::new();
    for (g, _e) in factor(f) {
        if g.degree() == Some(1) {
            // g = x + c0  (monic) => root = -c0
            let c0 = g.coefficients()[0].clone();
            out.push(-c0);
        }
    }
    out
}

/// Search for a monic irreducible polynomial of degree `n` over `GF(q)`,
/// returned as an [`FqPoly`]. For prime fields this reuses the fast `F_p` search.
pub fn irreducible_polynomial(field: &FiniteField, n: usize) -> Result<FqPoly> {
    if field.is_prime_field() {
        let coeffs = find_irreducible(field.characteristic(), n)?;
        let fq: Vec<FiniteFieldElement> = coeffs
            .into_iter()
            .map(|c| field.from_int(c))
            .collect();
        return Ok(FqPoly::new(field.clone(), fq));
    }
    // General GF(q): randomized search.
    let seed = 0xDEAD_BEEF_CAFE_0000u64
        ^ (n as u64)
        ^ field.order().to_u64().unwrap_or(11);
    let mut rng = Rng(seed);
    for _ in 0..100_000 {
        let mut coeffs: Vec<FiniteFieldElement> = (0..n).map(|_| rng.field_elt(field)).collect();
        coeffs.push(field.one()); // monic, degree n
        let cand = FqPoly::new(field.clone(), coeffs);
        if cand.degree() == Some(n) && is_irreducible(&cand) {
            return Ok(cand);
        }
    }
    Err(MathError::NotSupported(format!(
        "no irreducible of degree {n} found over {field} within search bound"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gf(p: i64, n: usize) -> FiniteField {
        FiniteField::new(Integer::from(p), n).unwrap()
    }

    fn poly(field: &FiniteField, ints: &[i64]) -> FqPoly {
        FqPoly::new(
            field.clone(),
            ints.iter().map(|&c| field.from_int(Integer::from(c))).collect(),
        )
    }

    #[test]
    fn fp_irreducibility_bootstrap() {
        let p = Integer::from(2);
        // x^2 + x + 1 irreducible over F_2
        assert!(is_irreducible_fp(&[Integer::from(1), Integer::from(1), Integer::from(1)], &p));
        // x^2 + 1 = (x+1)^2 reducible over F_2
        assert!(!is_irreducible_fp(&[Integer::from(1), Integer::from(0), Integer::from(1)], &p));
        // x^2 + x reducible
        assert!(!is_irreducible_fp(&[Integer::from(0), Integer::from(1), Integer::from(1)], &p));
    }

    #[test]
    fn find_irreducible_degrees() {
        for &pp in &[2i64, 3, 5] {
            let p = Integer::from(pp);
            for n in 2..=4 {
                let f = find_irreducible(&p, n).unwrap();
                assert_eq!(f.len(), n + 1);
                assert!(is_irreducible_fp(&f, &p));
            }
        }
    }

    #[test]
    fn factor_over_f2() {
        let f2 = gf(2, 1);
        // (x^2+x+1)(x+1) = x^3 + 0x^2 + 0x + 1 = x^3 + 1 over F_2
        let a = poly(&f2, &[1, 1, 1]); // x^2+x+1
        let b = poly(&f2, &[1, 1]); // x+1
        let prod = a.mul(&b);
        assert_eq!(prod, poly(&f2, &[1, 0, 0, 1])); // x^3+1
        let fac = factor(&prod);
        // two distinct irreducible factors, each multiplicity 1
        assert_eq!(fac.len(), 2);
        assert!(fac.iter().all(|(_, e)| *e == 1));
        // product of factors reconstructs
        let mut recon = FqPoly::one(f2.clone());
        for (g, e) in &fac {
            for _ in 0..*e {
                recon = recon.mul(g);
            }
        }
        assert_eq!(recon.make_monic(), prod.make_monic());
    }

    #[test]
    fn factor_with_multiplicity() {
        let f5 = gf(5, 1);
        // (x+1)^2 (x+2) over F_5
        let x1 = poly(&f5, &[1, 1]);
        let x2 = poly(&f5, &[2, 1]);
        let f = x1.mul(&x1).mul(&x2);
        let fac = factor(&f);
        // recover multiplicities: (x+1) exp 2, (x+2) exp 1
        let mut recon = FqPoly::one(f5.clone());
        for (g, e) in &fac {
            for _ in 0..*e {
                recon = recon.mul(g);
            }
        }
        assert_eq!(recon.make_monic(), f.make_monic());
        assert!(fac.iter().any(|(_, e)| *e == 2));
    }

    #[test]
    fn distinct_degree_split() {
        let f2 = gf(2, 1);
        // (x)(x+1)(x^2+x+1) = square-free, degrees {1,1,2}
        let f = poly(&f2, &[0, 1]).mul(&poly(&f2, &[1, 1])).mul(&poly(&f2, &[1, 1, 1]));
        let ddf = distinct_degree_factorization(&f);
        // degree-1 block has degree 2 (two linear factors), degree-2 block deg 2
        let deg1 = ddf.iter().find(|(_, d)| *d == 1).unwrap();
        assert_eq!(deg1.0.degree(), Some(2));
        let deg2 = ddf.iter().find(|(_, d)| *d == 2).unwrap();
        assert_eq!(deg2.0.degree(), Some(2));
    }

    #[test]
    fn equal_degree_cantor_zassenhaus_odd() {
        let f5 = gf(5, 1);
        // product of two distinct linear factors (x+1)(x+2)
        let f = poly(&f5, &[1, 1]).mul(&poly(&f5, &[2, 1]));
        let factors = equal_degree_factorization(&f, 1);
        assert_eq!(factors.len(), 2);
        assert!(factors.iter().all(|g| g.degree() == Some(1)));
    }

    #[test]
    fn factor_over_gf4() {
        // factor over an extension field GF(4)
        let f4 = gf(2, 2);
        let a = f4.generator();
        // (x - a)(x - a^2) has coefficients in F_2 subfield => x^2 + x + 1
        let root1 = FqPoly::new(f4.clone(), vec![-a.clone(), f4.one()]); // x - a
        let root2 = FqPoly::new(f4.clone(), vec![-(a.clone() * a.clone()), f4.one()]); // x - a^2
        let f = root1.mul(&root2);
        let fac = factor(&f);
        assert_eq!(fac.len(), 2);
        assert!(fac.iter().all(|(g, e)| g.degree() == Some(1) && *e == 1));
    }

    #[test]
    fn irreducibility_and_roots() {
        let f7 = gf(7, 1);
        // x^2 + 1 over F_7: 7 ≡ 3 mod 4 so -1 is a non-residue => irreducible
        let f = poly(&f7, &[1, 0, 1]);
        assert!(is_irreducible(&f));
        assert!(roots(&f).is_empty());
        // x^2 - 2 over F_7: 2 is a QR (3^2=9=2), roots exist
        let g = poly(&f7, &[-2, 0, 1]);
        assert!(!is_irreducible(&g));
        let r = roots(&g);
        assert_eq!(r.len(), 2);
        for root in r {
            // root^2 = 2
            assert_eq!(root.clone() * root, f7.from_int(Integer::from(2)));
        }
    }

    #[test]
    fn search_irreducible_fqpoly() {
        let f4 = gf(2, 2);
        let irr = irreducible_polynomial(&f4, 2).unwrap();
        assert_eq!(irr.degree(), Some(2));
        assert!(is_irreducible(&irr));
    }

    #[test]
    fn random_irreducible_various_p_n() {
        for (p, n) in [(2i64, 8usize), (3, 5), (5, 4), (7, 3), (19, 4), (10007, 2)] {
            let pi = Integer::from(p);
            let f = random_irreducible(&pi, n, 42).unwrap();
            assert_eq!(f.len(), n + 1, "wrong degree for p={p}, n={n}");
            assert!(f[n].is_one(), "not monic for p={p}, n={n}");
            assert!(is_irreducible_fp(&f, &pi), "reducible result for p={p}, n={n}");
            // Deterministic for a fixed seed.
            assert_eq!(f, random_irreducible(&pi, n, 42).unwrap());
        }
        assert!(random_irreducible(&Integer::from(7), 0, 1).is_err());
        // degree 1 always works
        let lin = random_irreducible(&Integer::from(13), 1, 7).unwrap();
        assert_eq!(lin.len(), 2);
    }

    // --- factorizations cross-checked against sympy (Poly(..., modulus=p)) ---

    /// Convert a factor list over a prime field GF(p) into sorted
    /// `(little-endian i64 coefficients, multiplicity)` pairs.
    fn fac_to_i64(fac: &[(FqPoly, usize)]) -> Vec<(Vec<i64>, usize)> {
        let mut out: Vec<(Vec<i64>, usize)> = fac
            .iter()
            .map(|(g, e)| {
                (
                    g.coefficients()
                        .iter()
                        .map(|c| c.eltseq()[0].to_i64())
                        .collect(),
                    *e,
                )
            })
            .collect();
        out.sort();
        out
    }

    #[test]
    fn factor_x15_plus_1_gf2_matches_sympy() {
        // sympy: factor_list(x^15 + 1, modulus=2) =
        //   (x+1)(x^2+x+1)(x^4+x+1)(x^4+x^3+1)(x^4+x^3+x^2+x+1)
        // (the classical splitting of x^15 - 1 into the minimal polynomials
        // of the 15th roots of unity over F_2).
        let f2 = gf(2, 1);
        let mut coeffs = vec![0i64; 16];
        coeffs[0] = 1;
        coeffs[15] = 1;
        let f = poly(&f2, &coeffs);
        let fac = factor(&f);
        let expected: Vec<(Vec<i64>, usize)> = vec![
            (vec![1, 1], 1),
            (vec![1, 1, 1], 1),
            (vec![1, 1, 0, 0, 1], 1),
            (vec![1, 0, 0, 1, 1], 1),
            (vec![1, 1, 1, 1, 1], 1),
        ];
        let mut expected = expected;
        expected.sort();
        assert_eq!(fac_to_i64(&fac), expected);
    }

    #[test]
    fn factor_gf5_matches_sympy() {
        // sympy: factor_list(x^7+3x^5+x^4+2x^3+4x^2+x+3, modulus=5) =
        //   (x + 4)(x^2 + 3)(x^4 + x^3 + x^2 + 2x + 4)
        let f5 = gf(5, 1);
        let f = poly(&f5, &[3, 1, 4, 2, 1, 3, 0, 1]);
        let fac = factor(&f);
        let mut expected: Vec<(Vec<i64>, usize)> = vec![
            (vec![4, 1], 1),
            (vec![3, 0, 1], 1),
            (vec![4, 2, 1, 1, 1], 1),
        ];
        expected.sort();
        assert_eq!(fac_to_i64(&fac), expected);
    }

    #[test]
    fn factor_x9_minus_x_gf3_matches_sympy() {
        // x^9 - x over F_3 is the product of ALL monic irreducibles of degree
        // dividing 2: x, x+1, x+2, x^2+1, x^2+x+2, x^2+2x+2 (sympy-verified).
        let f3 = gf(3, 1);
        let f = poly(&f3, &[0, 2, 0, 0, 0, 0, 0, 0, 0, 1]); // -x = 2x
        let fac = factor(&f);
        let mut expected: Vec<(Vec<i64>, usize)> = vec![
            (vec![0, 1], 1),
            (vec![1, 1], 1),
            (vec![2, 1], 1),
            (vec![1, 0, 1], 1),
            (vec![2, 1, 1], 1),
            (vec![2, 2, 1], 1),
        ];
        expected.sort();
        assert_eq!(fac_to_i64(&fac), expected);
    }

    #[test]
    fn factor_pth_power_multiplicity_gf3_matches_sympy() {
        // (x+1)^3 (x^2+1) over F_3: multiplicity 3 = p exercises the p-th-root
        // branch of square-free factorization. sympy: (x+1)^3 * (x^2+1).
        let f3 = gf(3, 1);
        let x1 = poly(&f3, &[1, 1]);
        let q = poly(&f3, &[1, 0, 1]);
        let f = x1.mul(&x1).mul(&x1).mul(&q);
        let fac = factor(&f);
        let mut expected: Vec<(Vec<i64>, usize)> = vec![(vec![1, 1], 3), (vec![1, 0, 1], 1)];
        expected.sort();
        assert_eq!(fac_to_i64(&fac), expected);
    }

    #[test]
    fn aes_modulus_irreducible_not_primitive() {
        // x^8 + x^4 + x^3 + x + 1 (the AES / Rijndael modulus) is irreducible
        // over F_2 (sympy-verified) but NOT primitive: ord(x) = 51 != 255.
        let f2 = gf(2, 1);
        let f = poly(&f2, &[1, 1, 0, 1, 1, 0, 0, 0, 1]);
        assert!(is_irreducible(&f));
        let coeffs: Vec<Integer> = [1i64, 1, 0, 1, 1, 0, 0, 0, 1]
            .iter()
            .map(|&c| Integer::from(c))
            .collect();
        assert!(is_irreducible_fp(&coeffs, &Integer::from(2)));
        // ord(x) mod f is 51 (verified independently with sympy galoistools).
        let x = FqPoly::x(f2.clone());
        assert_eq!(x.pow_mod(&Integer::from(51), &f), FqPoly::one(f2.clone()));
        assert_ne!(x.pow_mod(&Integer::from(17), &f), FqPoly::one(f2.clone()));
        assert_ne!(x.pow_mod(&Integer::from(3), &f), FqPoly::one(f2.clone()));
    }

    #[test]
    fn factor_reconstructs_over_gf9() {
        // Factor a product with repeated and extension-field factors over
        // GF(9) and check exact reconstruction: f = (x - a)^2 (x^2 + a) with
        // a the generator of GF(9).
        let f9 = gf(3, 2);
        let a = f9.generator();
        let lin = FqPoly::new(f9.clone(), vec![-a.clone(), f9.one()]);
        let quad = FqPoly::new(f9.clone(), vec![a.clone(), f9.zero(), f9.one()]);
        let f = lin.mul(&lin).mul(&quad);
        let fac = factor(&f);
        // multiplicities must include a 2, degrees must sum correctly
        let total: usize = fac
            .iter()
            .map(|(g, e)| g.degree().unwrap() * e)
            .sum();
        assert_eq!(total, 4);
        assert!(fac.iter().any(|(_, e)| *e == 2));
        let mut recon = FqPoly::one(f9.clone());
        for (g, e) in &fac {
            for _ in 0..*e {
                recon = recon.mul(g);
            }
        }
        assert_eq!(recon.make_monic(), f.make_monic());
        // every reported factor must be irreducible
        for (g, _) in &fac {
            assert!(is_irreducible(g));
        }
    }
}
