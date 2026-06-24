//! High-precision, deterministically-labeled root computation for an integer
//! polynomial — both **complex** and **p-adic** — feeding generic Stauduhar
//! descent (WP-STAUDHAR).
//!
//! # What this module provides
//!
//! Given `f ∈ Z[x]` of degree `n` (squarefree, up to ~24) this module computes:
//!
//! * its `n` complex roots to a controllable binary precision in C
//!   ([`complex_roots`]), Newton-refined with a returned accuracy bound;
//! * its roots in `Z_p` (the degree-1 factors over `Q_p`) lifted from `F_p`
//!   via Hensel to precision `p^k` ([`padic_roots`]);
//! * a *stable, deterministic* labeling of the complex roots
//!   ([`complex_roots`] always returns roots in the same canonical order for the
//!   same `f`), so downstream resolvent evaluation can index roots reliably;
//! * a "round-to-integer-if-close" / rational-reconstruction decision helper
//!   ([`round_to_integer_if_close`], [`rational_reconstruction`]) used to decide
//!   when an evaluated resolvent factor is integral.
//!
//! # Arbitrary precision without new dependencies
//!
//! The crate does not depend on a binary-float bignum library, so this module
//! carries its own *fixed-point* arbitrary-precision real ([`BigFloat`]) and
//! complex ([`BigComplex`]) types: a [`BigFloat`] of precision `prec` bits
//! represents the exact rational `mantissa / 2^prec` where `mantissa` is a
//! `rustmath_integers::Integer`. All arithmetic is exact integer arithmetic on
//! the mantissa with an explicit rescale step, so the only error is the final
//! truncation, which is tracked.
//!
//! # Root-labeling scheme (deterministic, documented)
//!
//! [`complex_roots`] returns roots sorted by the total order:
//! `(round(re · 2^cmp_bits), round(im · 2^cmp_bits))` ascending — i.e. by real
//! part, then by imaginary part, compared at a coarse `cmp_bits` resolution so
//! that two roots agreeing to within `2^-cmp_bits` never swap order due to
//! noise in the last bits. The comparison resolution is deliberately coarser
//! than the working precision. Because the underlying root set of a fixed `f`
//! is fixed and the sort is a total order on stable coarse keys, **the same `f`
//! always yields roots in the same positions** — this *is* the stable labeling
//! Stauduhar relies on. p-adic roots ([`padic_roots`]) are returned sorted by
//! their canonical residue in `[0, p^k)`, likewise deterministic.

use crate::univariate::UnivariatePolynomial;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;

// ---------------------------------------------------------------------------
// Small integer-polynomial helpers (self-contained; little-endian Vec<Integer>)
// ---------------------------------------------------------------------------

/// Strip trailing zero coefficients, keeping at least one element.
fn poly_trim(p: &[Integer]) -> Vec<Integer> {
    let mut c = p.to_vec();
    while c.len() > 1 && c.last().map(|x| x.is_zero()).unwrap_or(false) {
        c.pop();
    }
    if c.is_empty() {
        c.push(izero());
    }
    c
}

/// Degree of a little-endian integer polynomial; `-1` for the zero polynomial.
fn poly_degree(p: &[Integer]) -> i64 {
    let t = poly_trim(p);
    if t.len() == 1 && t[0].is_zero() {
        -1
    } else {
        (t.len() - 1) as i64
    }
}

/// Schoolbook multiply of two little-endian integer polynomials.
/// (Used for test-fixture construction and as a public-ish building block.)
#[allow(dead_code)]
fn poly_mul(a: &[Integer], b: &[Integer]) -> Vec<Integer> {
    if a.is_empty() || b.is_empty() {
        return vec![izero()];
    }
    let mut out = vec![izero(); a.len() + b.len() - 1];
    for (i, ai) in a.iter().enumerate() {
        if ai.is_zero() {
            continue;
        }
        for (j, bj) in b.iter().enumerate() {
            if bj.is_zero() {
                continue;
            }
            out[i + j] = &out[i + j] + &(ai * bj);
        }
    }
    poly_trim(&out)
}

// ---------------------------------------------------------------------------
// Small Integer helpers
// ---------------------------------------------------------------------------

#[inline]
fn izero() -> Integer {
    Integer::from(0i64)
}

#[inline]
fn ione() -> Integer {
    Integer::from(1i64)
}

/// `2^k` as an [`Integer`].
#[inline]
fn pow2(k: u32) -> Integer {
    Integer::from(2i64).pow(k)
}

/// Round-half-away division of an integer by a *positive* power of two: returns
/// `round(n / 2^k)`. Exact integer arithmetic.
fn round_div_pow2(n: &Integer, k: u32) -> Integer {
    if k == 0 {
        return n.clone();
    }
    let d = pow2(k);
    let half = pow2(k - 1);
    if n.signum() >= 0 {
        &(n + &half) / &d
    } else {
        // symmetric rounding away from zero at .5
        -(&(&n.abs() + &half) / &d)
    }
}

// ---------------------------------------------------------------------------
// BigFloat: fixed-point real of `prec` fractional bits, value = mantissa / 2^prec
// ---------------------------------------------------------------------------

/// Fixed-point arbitrary-precision real number.
///
/// Represents the exact value `mantissa / 2^prec`. All operations keep the
/// operands at a common precision and the result is truncated/rounded back to
/// `prec` fractional bits; the truncation error of a single rescale is at most
/// `2^-prec`.
#[derive(Clone, Debug)]
pub struct BigFloat {
    mantissa: Integer,
    prec: u32,
}

impl BigFloat {
    /// The exact zero at precision `prec`.
    pub fn zero(prec: u32) -> Self {
        BigFloat { mantissa: izero(), prec }
    }

    /// Construct from an integer value at precision `prec`.
    pub fn from_integer(n: &Integer, prec: u32) -> Self {
        BigFloat { mantissa: n * &pow2(prec), prec }
    }

    /// Construct from an `i64` value at precision `prec`.
    pub fn from_i64(n: i64, prec: u32) -> Self {
        BigFloat::from_integer(&Integer::from(n), prec)
    }

    /// Construct from an `f64` value at precision `prec`.
    pub fn from_f64(x: f64, prec: u32) -> Self {
        if x == 0.0 || !x.is_finite() {
            return BigFloat::zero(prec);
        }
        let neg = x < 0.0;
        let mut ax = x.abs();
        // Decompose ax = m * 2^e with m in [1,2).
        let mut e: i32 = 0;
        while ax >= 2.0 {
            ax /= 2.0;
            e += 1;
        }
        while ax < 1.0 {
            ax *= 2.0;
            e -= 1;
        }
        // Now ax in [1,2); take 52 significant fractional bits.
        let sig_bits = 52i32;
        let scaled = (ax * (1u64 << sig_bits) as f64).round();
        let scaled_i = if scaled < 0.0 { 0i64 } else { scaled as i64 };
        let mant53 = Integer::from(scaled_i);
        // value = mant53 * 2^(e - sig_bits); want mantissa = value * 2^prec.
        let shift = e - sig_bits + prec as i32;
        let mut mantissa = if shift >= 0 {
            &mant53 * &pow2(shift as u32)
        } else {
            round_div_pow2(&mant53, (-shift) as u32)
        };
        if neg {
            mantissa = -mantissa;
        }
        BigFloat { mantissa, prec }
    }

    /// The fixed-point precision (fractional bits).
    pub fn prec(&self) -> u32 {
        self.prec
    }

    /// The raw mantissa (value = `mantissa / 2^prec`).
    pub fn mantissa(&self) -> &Integer {
        &self.mantissa
    }

    /// Rescale to a new precision (rounding).
    pub fn with_prec(&self, prec: u32) -> Self {
        if prec == self.prec {
            self.clone()
        } else if prec > self.prec {
            BigFloat { mantissa: &self.mantissa * &pow2(prec - self.prec), prec }
        } else {
            BigFloat { mantissa: round_div_pow2(&self.mantissa, self.prec - prec), prec }
        }
    }

    fn at(a: &BigFloat, b: &BigFloat) -> (Integer, Integer, u32) {
        let p = a.prec.max(b.prec);
        (a.with_prec(p).mantissa, b.with_prec(p).mantissa, p)
    }

    /// Exact sign: `-1`, `0`, or `1`.
    pub fn signum(&self) -> i8 {
        self.mantissa.signum()
    }

    /// Absolute value.
    pub fn abs(&self) -> Self {
        BigFloat { mantissa: self.mantissa.abs(), prec: self.prec }
    }

    /// `self + other`.
    pub fn add(&self, other: &BigFloat) -> BigFloat {
        let (a, b, p) = BigFloat::at(self, other);
        BigFloat { mantissa: &a + &b, prec: p }
    }

    /// `self - other`.
    pub fn sub(&self, other: &BigFloat) -> BigFloat {
        let (a, b, p) = BigFloat::at(self, other);
        BigFloat { mantissa: &a - &b, prec: p }
    }

    /// `self * other`, rounded back to the common precision.
    pub fn mul(&self, other: &BigFloat) -> BigFloat {
        let (a, b, p) = BigFloat::at(self, other);
        // (a/2^p)·(b/2^p) = (a·b)/2^{2p}; round to /2^p.
        let prod = &a * &b;
        BigFloat { mantissa: round_div_pow2(&prod, p), prec: p }
    }

    /// `self / other`, rounded. Panics on division by zero.
    pub fn div(&self, other: &BigFloat) -> BigFloat {
        let (a, b, p) = BigFloat::at(self, other);
        assert!(b.signum() != 0, "BigFloat::div by zero");
        // a/2^p ÷ b/2^p = a/b; want mantissa = round(a/b · 2^p) = round(a·2^p / b).
        let num = &a * &pow2(p);
        let q = round_div_signed(&num, &b);
        BigFloat { mantissa: q, prec: p }
    }

    /// Compare two big-floats (exact on the common precision).
    pub fn cmp(&self, other: &BigFloat) -> std::cmp::Ordering {
        let (a, b, _) = BigFloat::at(self, other);
        a.cmp(&b)
    }

    /// Lossy conversion to `f64` (for diagnostics / seeding).
    pub fn to_f64(&self) -> f64 {
        let m = &self.mantissa;
        if m.is_zero() {
            return 0.0;
        }
        let bits = m.bit_length() as i64;
        let drop = (bits - 60).max(0) as u32; // keep ~60 significant bits
        let reduced = round_div_pow2(m, drop);
        let mant_f = reduced.to_f64().unwrap_or(0.0);
        // value = mant_f * 2^drop / 2^prec = mant_f * 2^(drop - prec)
        let exp = drop as i64 - self.prec as i64;
        mant_f * 2f64.powi(exp as i32)
    }

    /// Round to the nearest integer.
    pub fn round_to_integer(&self) -> Integer {
        round_div_pow2(&self.mantissa, self.prec)
    }
}

/// Round-half-away division of integers `n / d` (d ≠ 0).
fn round_div_signed(n: &Integer, d: &Integer) -> Integer {
    let two = Integer::from(2i64);
    let (q, r) = (n / d, n % d);
    if r.is_zero() {
        return q;
    }
    let twice_r = &r.abs() * &two;
    if twice_r >= d.abs() {
        if (n.signum() as i32) * (d.signum() as i32) >= 0 {
            &q + &ione()
        } else {
            &q - &ione()
        }
    } else {
        q
    }
}

// ---------------------------------------------------------------------------
// BigComplex
// ---------------------------------------------------------------------------

/// Arbitrary-precision complex number = pair of [`BigFloat`].
#[derive(Clone, Debug)]
pub struct BigComplex {
    /// Real part.
    pub re: BigFloat,
    /// Imaginary part.
    pub im: BigFloat,
}

impl BigComplex {
    /// Construct `re + im·i`.
    pub fn new(re: BigFloat, im: BigFloat) -> Self {
        BigComplex { re, im }
    }

    /// Zero at precision `prec`.
    pub fn zero(prec: u32) -> Self {
        BigComplex { re: BigFloat::zero(prec), im: BigFloat::zero(prec) }
    }

    /// From an `f64` pair at precision `prec`.
    pub fn from_f64(re: f64, im: f64, prec: u32) -> Self {
        BigComplex { re: BigFloat::from_f64(re, prec), im: BigFloat::from_f64(im, prec) }
    }

    /// Common precision of the parts.
    pub fn prec(&self) -> u32 {
        self.re.prec.max(self.im.prec)
    }

    /// `self + other`.
    pub fn add(&self, other: &BigComplex) -> BigComplex {
        BigComplex { re: self.re.add(&other.re), im: self.im.add(&other.im) }
    }

    /// `self - other`.
    pub fn sub(&self, other: &BigComplex) -> BigComplex {
        BigComplex { re: self.re.sub(&other.re), im: self.im.sub(&other.im) }
    }

    /// `self * other`.
    pub fn mul(&self, other: &BigComplex) -> BigComplex {
        let ac = self.re.mul(&other.re);
        let bd = self.im.mul(&other.im);
        let ad = self.re.mul(&other.im);
        let bc = self.im.mul(&other.re);
        BigComplex { re: ac.sub(&bd), im: ad.add(&bc) }
    }

    /// `self / other`. Panics if `other == 0`.
    pub fn div(&self, other: &BigComplex) -> BigComplex {
        // (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c²+d²)
        let denom = other.re.mul(&other.re).add(&other.im.mul(&other.im));
        let re = self.re.mul(&other.re).add(&self.im.mul(&other.im)).div(&denom);
        let im = self.im.mul(&other.re).sub(&self.re.mul(&other.im)).div(&denom);
        BigComplex { re, im }
    }

    /// Squared modulus `re² + im²` as a [`BigFloat`].
    pub fn norm_sqr(&self) -> BigFloat {
        self.re.mul(&self.re).add(&self.im.mul(&self.im))
    }

    /// Lossy `(re, im)` as `f64`.
    pub fn to_f64(&self) -> (f64, f64) {
        (self.re.to_f64(), self.im.to_f64())
    }

    /// Rescale both parts to `prec`.
    pub fn with_prec(&self, prec: u32) -> Self {
        BigComplex { re: self.re.with_prec(prec), im: self.im.with_prec(prec) }
    }
}

// ---------------------------------------------------------------------------
// Complex roots
// ---------------------------------------------------------------------------

/// Result of [`complex_roots`]: the labeled roots plus an accuracy bound.
#[derive(Clone, Debug)]
pub struct ComplexRoots {
    /// The `n` roots in canonical (deterministic) order — the stable labeling.
    pub roots: Vec<BigComplex>,
    /// Working precision in fractional bits.
    pub prec: u32,
    /// Upper bound (in absolute value, as `f64`) on the distance of each
    /// returned approximation from a true root, valid for simple roots
    /// (Newton residual `|f(z)| / |f'(z)|`).
    pub accuracy_bound: f64,
}

/// Evaluate `f` (integer coeffs, little-endian) at a [`BigComplex`] via Horner.
fn eval_complex(coeffs: &[Integer], x: &BigComplex, prec: u32) -> BigComplex {
    let mut acc = BigComplex::zero(prec);
    for c in coeffs.iter().rev() {
        let cf = BigComplex::new(BigFloat::from_integer(c, prec), BigFloat::zero(prec));
        acc = acc.mul(x).add(&cf);
    }
    acc
}

/// Evaluate the derivative `f'` at a [`BigComplex`] via Horner on the
/// derivative coefficients.
fn eval_complex_deriv(coeffs: &[Integer], x: &BigComplex, prec: u32) -> BigComplex {
    if coeffs.len() <= 1 {
        return BigComplex::zero(prec);
    }
    let dcoeffs: Vec<Integer> =
        (1..coeffs.len()).map(|i| &coeffs[i] * &Integer::from(i as i64)).collect();
    eval_complex(&dcoeffs, x, prec)
}

/// f64 Durand–Kerner (Weierstrass) iteration to isolate all `n` roots
/// approximately. Used only to *seed* the high-precision Newton refinement.
fn durand_kerner_f64(coeffs_f64: &[f64], iters: usize) -> Vec<(f64, f64)> {
    let n = coeffs_f64.len() - 1;
    let lead = coeffs_f64[n];
    // Monic version.
    let monic: Vec<f64> = coeffs_f64.iter().map(|c| c / lead).collect();
    let eval = |zr: f64, zi: f64| -> (f64, f64) {
        let (mut ar, mut ai) = (0.0f64, 0.0f64);
        for c in monic.iter().rev() {
            // acc = acc*z + c
            let nr = ar * zr - ai * zi + c;
            let ni = ar * zi + ai * zr;
            ar = nr;
            ai = ni;
        }
        (ar, ai)
    };
    // Spread initial guesses as powers of (0.4 + 0.9i).
    let mut roots: Vec<(f64, f64)> = Vec::with_capacity(n);
    let (br, bi) = (0.4f64, 0.9f64);
    let (mut cr, mut ci) = (1.0f64, 0.0f64);
    for _ in 0..n {
        let nr = cr * br - ci * bi;
        let ni = cr * bi + ci * br;
        cr = nr;
        ci = ni;
        roots.push((cr, ci));
    }
    for _ in 0..iters {
        let mut max_step = 0.0f64;
        for i in 0..n {
            let (zr, zi) = roots[i];
            let (fr, fi) = eval(zr, zi);
            // denom = prod_{j != i} (z_i - z_j)
            let (mut dr, mut di) = (1.0f64, 0.0f64);
            for j in 0..n {
                if j == i {
                    continue;
                }
                let (xr, xi) = (zr - roots[j].0, zi - roots[j].1);
                let nr = dr * xr - di * xi;
                let ni = dr * xi + di * xr;
                dr = nr;
                di = ni;
            }
            let den = dr * dr + di * di;
            if den == 0.0 {
                continue;
            }
            let sr = (fr * dr + fi * di) / den;
            let si = (fi * dr - fr * di) / den;
            roots[i] = (zr - sr, zi - si);
            let step = (sr * sr + si * si).sqrt();
            if step > max_step {
                max_step = step;
            }
        }
        if max_step < 1e-13 {
            break;
        }
    }
    roots
}

/// Newton-refine a single complex root to the working precision.
fn newton_refine(coeffs: &[Integer], seed: &BigComplex, prec: u32, max_iters: usize) -> BigComplex {
    let mut z = seed.with_prec(prec);
    for _ in 0..max_iters {
        let fz = eval_complex(coeffs, &z, prec);
        let dfz = eval_complex_deriv(coeffs, &z, prec);
        // Guard: derivative ~ 0 → stop (multiple-root or bad seed).
        let dnorm = dfz.norm_sqr();
        if dnorm.signum() == 0 {
            break;
        }
        let step = fz.div(&dfz);
        let znext = z.sub(&step);
        let sn = step.norm_sqr();
        z = znext;
        // step² < 2^-(2 prec - 8)  ⇒ |step| < 2^-(prec-4)
        let thresh = BigFloat { mantissa: ione(), prec: (2 * prec).saturating_sub(8).max(1) };
        if sn.cmp(&thresh) == std::cmp::Ordering::Less {
            break;
        }
    }
    z
}

/// Coarse comparison key for deterministic labeling: `round(value · 2^cmp_bits)`.
fn cmp_key(b: &BigFloat, cmp_bits: u32) -> Integer {
    b.with_prec(cmp_bits).round_to_integer()
}

/// Compute all `n` complex roots of an integer polynomial `f` (little-endian
/// coefficients) to `prec` fractional bits, returned in a **stable,
/// deterministic** order (see the module docs for the labeling scheme).
///
/// `f` should be squarefree for full accuracy; multiple roots converge linearly
/// and may not reach the target bound.
///
/// # Returns
/// A [`ComplexRoots`] with the labeled roots, the working precision, and an
/// accuracy bound (max residual distance, as `f64`).
pub fn complex_roots(coeffs: &[Integer], prec: u32) -> ComplexRoots {
    let trimmed = poly_trim(coeffs);
    let deg = poly_degree(&trimmed);
    if deg <= 0 {
        return ComplexRoots { roots: vec![], prec, accuracy_bound: 0.0 };
    }
    let coeffs_f64: Vec<f64> = trimmed.iter().map(|c| c.to_f64().unwrap_or(0.0)).collect();
    // Seed with f64 Durand–Kerner.
    let seeds = durand_kerner_f64(&coeffs_f64, 200);
    // Working precision with guard bits for the refinement.
    let work = prec + 16;
    let mut roots: Vec<BigComplex> = seeds
        .iter()
        .map(|&(r, i)| {
            let seed = BigComplex::from_f64(r, i, work);
            newton_refine(&trimmed, &seed, work, 200)
        })
        .collect();

    // Accuracy bound: max |f(z)| / |f'(z)| over the roots.
    let mut bound = 0.0f64;
    for z in &roots {
        let fz = eval_complex(&trimmed, z, work);
        let dfz = eval_complex_deriv(&trimmed, z, work);
        let (fr, fi) = fz.to_f64();
        let (dr, di) = dfz.to_f64();
        let fmag = (fr * fr + fi * fi).sqrt();
        let dmag = (dr * dr + di * di).sqrt();
        if dmag > 0.0 {
            let est = fmag / dmag;
            if est > bound {
                bound = est;
            }
        }
    }

    // Truncate back to the requested precision and sort into canonical order.
    for z in roots.iter_mut() {
        *z = z.with_prec(prec);
    }
    // Coarse comparison resolution: below working precision so noise in the last
    // bits cannot reorder roots. Use min(prec/2, prec-8).
    let cmp_bits = (prec / 2).min(prec.saturating_sub(8)).max(1);
    roots.sort_by(|a, b| {
        let ka = cmp_key(&a.re, cmp_bits);
        let kb = cmp_key(&b.re, cmp_bits);
        match ka.cmp(&kb) {
            std::cmp::Ordering::Equal => {
                cmp_key(&a.im, cmp_bits).cmp(&cmp_key(&b.im, cmp_bits))
            }
            other => other,
        }
    });

    ComplexRoots { roots, prec, accuracy_bound: bound }
}

// ---------------------------------------------------------------------------
// p-adic roots
// ---------------------------------------------------------------------------

/// A p-adic root of `f`: the residue `value` in `[0, p^prec_power)` with
/// `f(value) ≡ 0 (mod p^prec_power)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PadicRoot {
    /// Canonical lift of the root in `[0, p^prec_power)`.
    pub value: Integer,
    /// The prime.
    pub p: i64,
    /// The precision (power of `p`).
    pub prec_power: u32,
}

/// Find the roots of `f` in `Z_p` to precision `p^n`, by lifting the *simple*
/// linear factors of `f mod p` via Hensel/Newton.
///
/// Returns the roots whose mod-`p` reduction is a *simple* root of `f mod p`
/// (these are exactly the roots that lift uniquely by Hensel's lemma). Roots at
/// a repeated factor mod `p` are not returned (they require Newton-polygon /
/// ramified handling, out of this WP's scope). The result is sorted by
/// canonical residue, hence deterministic.
///
/// # Arguments
/// * `coeffs` — `f` little-endian integer coefficients.
/// * `p` — the prime.
/// * `n` — target precision (power of `p`), `n ≥ 1`.
pub fn padic_roots(coeffs: &[Integer], p: i64, n: u32) -> Vec<PadicRoot> {
    assert!(n >= 1, "padic_roots: precision must be >= 1");
    let trimmed = poly_trim(coeffs);
    let deg = poly_degree(&trimmed);
    if deg <= 0 {
        return vec![];
    }
    let p_big = Integer::from(p);
    let p_n = p_big.pow(n);

    // Reduce f mod p to F_p[x] (as i64 coefficients in [0, p)).
    let f_fp = to_fp(&trimmed, p);
    if poly_degree_fp(&f_fp) < 0 {
        // f ≡ 0 mod p: cannot Hensel-lift cleanly. Out of scope.
        return vec![];
    }

    // f' mod p.
    let deriv_fp = derivative_fp(&f_fp, p);
    // Simple roots a of f mod p: f(a) ≡ 0 and f'(a) ≢ 0 mod p.
    let mut simple_roots_mod_p: Vec<i64> = Vec::new();
    for a in 0..p {
        if eval_fp(&f_fp, a, p) == 0 && eval_fp(&deriv_fp, a, p) != 0 {
            simple_roots_mod_p.push(a);
        }
    }

    let mut out: Vec<PadicRoot> = Vec::new();
    for a in simple_roots_mod_p {
        if let Some(root) = lift_simple_root(&trimmed, a, p, n) {
            let val = &(&root % &p_n) + &p_n;
            let val = &val % &p_n;
            out.push(PadicRoot { value: val, p, prec_power: n });
        }
    }
    out.sort_by(|a, b| a.value.cmp(&b.value));
    out
}

/// Newton-lift a simple root `a` of `f mod p` to a root mod `p^n`.
fn lift_simple_root(coeffs: &[Integer], a0: i64, p: i64, n: u32) -> Option<Integer> {
    let p_big = Integer::from(p);
    let p_n = p_big.pow(n);
    let dcoeffs: Vec<Integer> = if coeffs.len() <= 1 {
        vec![izero()]
    } else {
        (1..coeffs.len()).map(|i| &coeffs[i] * &Integer::from(i as i64)).collect()
    };

    let mut x = Integer::from(a0);
    let mut cur: u32 = 1;
    // f'(a0)^{-1} mod p (the simple-root condition guarantees this exists).
    let dval0 = eval_int_mod(&dcoeffs, &Integer::from(a0), &p_big);
    let dinv_p = mod_inverse(&dval0, &p_big)?;

    // Linear Hensel: raise precision one power of p at a time. With the fixed
    // mod-p derivative inverse this converges linearly, which is robust.
    while cur < n {
        let next = (cur + 1).min(n);
        let mod_next = p_big.pow(next);
        let fval = eval_int_mod(coeffs, &x, &mod_next);
        // f(x) is divisible by p^cur; correction = f(x) · f'(a0)^{-1} mod p^next.
        let corr = &(&fval * &dinv_p) % &mod_next;
        x = &(&(&x - &corr) % &mod_next) + &mod_next;
        x = &x % &mod_next;
        cur = next;
    }
    // Verify f(x) ≡ 0 mod p^n.
    if eval_int_mod(coeffs, &x, &p_n).is_zero() {
        Some(x)
    } else {
        None
    }
}

/// Evaluate integer-coeff poly at `x`, reduced mod `m` (result in `[0, m)`).
fn eval_int_mod(coeffs: &[Integer], x: &Integer, m: &Integer) -> Integer {
    let mut acc = izero();
    for c in coeffs.iter().rev() {
        acc = &(&(&acc * x) + c) % m;
    }
    let r = &(&acc % m) + m;
    &r % m
}

/// Modular inverse of `a` mod `m` (m prime here), via extended gcd.
fn mod_inverse(a: &Integer, m: &Integer) -> Option<Integer> {
    let a_red = &(&(a % m) + m) % m;
    let (g, x, _) = a_red.extended_gcd(m);
    if !g.is_one() {
        return None;
    }
    let inv = &(&(&x % m) + m) % m;
    Some(inv)
}

/// Strip trailing zeros of an `F_p[x]` poly (`Vec<i64>`), keeping one element.
fn trim_fp(p: &[i64]) -> Vec<i64> {
    let mut c = p.to_vec();
    while c.len() > 1 && *c.last().unwrap() == 0 {
        c.pop();
    }
    if c.is_empty() {
        c.push(0);
    }
    c
}

/// Degree of an `F_p[x]` poly; `-1` for zero.
fn poly_degree_fp(p: &[i64]) -> i64 {
    let t = trim_fp(p);
    if t.len() == 1 && t[0] == 0 {
        -1
    } else {
        (t.len() - 1) as i64
    }
}

/// Derivative of an `F_p[x]` poly.
fn derivative_fp(coeffs: &[i64], p: i64) -> Vec<i64> {
    if coeffs.len() <= 1 {
        return vec![0];
    }
    let d: Vec<i64> = (1..coeffs.len())
        .map(|i| {
            let v = (coeffs[i] % p) * ((i as i64) % p) % p;
            ((v % p) + p) % p
        })
        .collect();
    trim_fp(&d)
}

/// Reduce an integer polynomial mod p to `F_p[x]` (`Vec<i64>`) in `[0,p)`.
fn to_fp(a: &[Integer], p: i64) -> Vec<i64> {
    let pm = Integer::from(p);
    let v: Vec<i64> = a
        .iter()
        .map(|c| {
            let r = c % &pm;
            (&(&r + &pm) % &pm).to_i64()
        })
        .collect();
    trim_fp(&v)
}

/// Evaluate an `F_p[x]` poly at `a` mod p.
fn eval_fp(coeffs: &[i64], a: i64, p: i64) -> i64 {
    let mut acc = 0i64;
    for &c in coeffs.iter().rev() {
        acc = (acc % p * (a % p) % p + c % p) % p;
        acc = ((acc % p) + p) % p;
    }
    acc % p
}

// ---------------------------------------------------------------------------
// Round-to-integer / rational reconstruction
// ---------------------------------------------------------------------------

/// If the [`BigFloat`] `x` is within `2^-tol_bits` of an integer, return that
/// integer; otherwise `None`. Used by Stauduhar to decide whether an evaluated
/// resolvent factor is integral.
///
/// `tol_bits` should be chosen comfortably below the working precision (so the
/// truncation error of `x` is well inside the tolerance).
pub fn round_to_integer_if_close(x: &BigFloat, tol_bits: u32) -> Option<Integer> {
    let nearest = x.round_to_integer();
    let nearest_bf = BigFloat::from_integer(&nearest, x.prec);
    let err = x.sub(&nearest_bf).abs();
    let tol = if tol_bits <= x.prec {
        BigFloat { mantissa: pow2(x.prec - tol_bits), prec: x.prec }
    } else {
        // tolerance smaller than ulp: require exact.
        BigFloat { mantissa: izero(), prec: x.prec }
    };
    if err.cmp(&tol) != std::cmp::Ordering::Greater {
        Some(nearest)
    } else {
        None
    }
}

/// Decide whether a [`BigComplex`] `z` is (within tolerance) a rational
/// **integer** — imaginary part ≈ 0 and real part ≈ an integer — and if so
/// return it. This is the resolvent-root identification test Stauduhar uses to
/// recognise when a Lagrange resolvent evaluates to an algebraic integer in `Z`.
pub fn complex_round_to_integer_if_close(z: &BigComplex, tol_bits: u32) -> Option<Integer> {
    if round_to_integer_if_close(&z.im, tol_bits) != Some(izero()) {
        return None;
    }
    round_to_integer_if_close(&z.re, tol_bits)
}

/// Rational reconstruction by continued fractions: find the rational `p/q` with
/// `|q| ≤ max_denom` closest to the [`BigFloat`] `x`, within `2^-tol_bits`.
/// Returns `None` if no such rational is found.
///
/// Used to recognise when an approximate value is a small-denominator rational
/// (e.g. resolvent coefficients after clearing leading coefficients).
pub fn rational_reconstruction(
    x: &BigFloat,
    max_denom: &Integer,
    tol_bits: u32,
) -> Option<Rational> {
    // Continued-fraction expansion of the exact rational mantissa / 2^prec.
    let mut num = x.mantissa.clone();
    let mut den = pow2(x.prec);

    // Convergent recurrences.
    let mut h_prev = izero();
    let mut h_cur = ione();
    let mut k_prev = ione();
    let mut k_cur = izero();

    let mut best: Option<Rational> = None;

    for _ in 0..400 {
        if den.is_zero() {
            break;
        }
        let a = floor_div(&num, &den);
        let h_next = &(&a * &h_cur) + &h_prev;
        let k_next = &(&a * &k_cur) + &k_prev;
        h_prev = h_cur;
        h_cur = h_next;
        k_prev = k_cur;
        k_cur = k_next;

        let r = &num - &(&a * &den);
        num = den;
        den = r;

        if k_cur.abs() > *max_denom {
            break;
        }
        if k_cur.signum() == 0 {
            continue;
        }
        if let Ok(cand) = Rational::new(h_cur.clone(), k_cur.clone()) {
            let cand_bf = rational_to_bigfloat(&cand, x.prec + tol_bits + 8);
            let err = x.with_prec(cand_bf.prec).sub(&cand_bf).abs();
            let tol = if tol_bits <= err.prec {
                BigFloat { mantissa: pow2(err.prec - tol_bits), prec: err.prec }
            } else {
                BigFloat { mantissa: izero(), prec: err.prec }
            };
            if err.cmp(&tol) != std::cmp::Ordering::Greater {
                best = Some(cand);
                break;
            }
        }
        if den.is_zero() {
            break;
        }
    }
    best
}

/// Floor division (toward −∞) of integers.
fn floor_div(n: &Integer, d: &Integer) -> Integer {
    let q = n / d;
    let r = n % d;
    if !r.is_zero() && (r.signum() as i32) * (d.signum() as i32) < 0 {
        &q - &ione()
    } else {
        q
    }
}

/// Convert a [`Rational`] to a [`BigFloat`] at `prec` bits (rounded).
fn rational_to_bigfloat(q: &Rational, prec: u32) -> BigFloat {
    let num = q.numerator();
    let den = q.denominator();
    let scaled = num * &pow2(prec);
    let mantissa = round_div_signed(&scaled, den);
    BigFloat { mantissa, prec }
}

/// Convenience: complex roots of a [`UnivariatePolynomial<Integer>`].
pub fn complex_roots_of(poly: &UnivariatePolynomial<Integer>, prec: u32) -> ComplexRoots {
    complex_roots(poly.coefficients(), prec)
}

/// Convenience: p-adic roots of a [`UnivariatePolynomial<Integer>`].
pub fn padic_roots_of(poly: &UnivariatePolynomial<Integer>, p: i64, n: u32) -> Vec<PadicRoot> {
    padic_roots(poly.coefficients(), p, n)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ints(v: &[i64]) -> Vec<Integer> {
        v.iter().map(|&c| Integer::from(c)).collect()
    }

    // ---- BigFloat sanity ----

    #[test]
    fn bigfloat_arith() {
        let prec = 64;
        let a = BigFloat::from_i64(3, prec);
        let b = BigFloat::from_i64(7, prec);
        let s = a.add(&b);
        assert_eq!(s.round_to_integer(), Integer::from(10));
        let p = a.mul(&b);
        assert_eq!(p.round_to_integer(), Integer::from(21));
        // 7 / 2 = 3.5 → rounds away to 4
        let q = b.div(&BigFloat::from_i64(2, prec));
        assert_eq!(q.round_to_integer(), Integer::from(4));
        // 1/3 then *3 ≈ 1
        let third = BigFloat::from_i64(1, prec).div(&BigFloat::from_i64(3, prec));
        let one = third.mul(&BigFloat::from_i64(3, prec));
        assert_eq!(one.round_to_integer(), Integer::from(1));
        // negative division: -7/2 = -3.5 → -4
        let qn = BigFloat::from_i64(-7, prec).div(&BigFloat::from_i64(2, prec));
        assert_eq!(qn.round_to_integer(), Integer::from(-4));
    }

    // ---- complex roots: x^2 - 2 -> ±sqrt2 to high precision ----

    #[test]
    fn complex_roots_sqrt2() {
        let prec = 200;
        let cr = complex_roots(&ints(&[-2, 0, 1]), prec);
        assert_eq!(cr.roots.len(), 2);
        for z in &cr.roots {
            assert_eq!(round_to_integer_if_close(&z.im, prec - 20), Some(Integer::from(0)));
        }
        // sorted ascending by real part: -sqrt2, +sqrt2
        assert!(cr.roots[0].re.signum() < 0);
        assert!(cr.roots[1].re.signum() > 0);
        // r1^2 ≈ 2
        let sq = cr.roots[1].re.mul(&cr.roots[1].re);
        assert_eq!(round_to_integer_if_close(&sq, prec - 20), Some(Integer::from(2)));
        assert!(cr.accuracy_bound < 1e-10, "bound={}", cr.accuracy_bound);
    }

    #[test]
    fn complex_roots_imaginary_unit() {
        // x^2 + 1 -> ±i
        let prec = 160;
        let cr = complex_roots(&ints(&[1, 0, 1]), prec);
        assert_eq!(cr.roots.len(), 2);
        for z in &cr.roots {
            assert_eq!(round_to_integer_if_close(&z.re, prec - 20), Some(Integer::from(0)));
        }
        assert_eq!(round_to_integer_if_close(&cr.roots[0].im, prec - 20), Some(Integer::from(-1)));
        assert_eq!(round_to_integer_if_close(&cr.roots[1].im, prec - 20), Some(Integer::from(1)));
    }

    #[test]
    fn complex_roots_cyclotomic_phi5() {
        // Φ_5 = x^4 + x^3 + x^2 + x + 1: 4 roots on the unit circle, |z| = 1.
        let prec = 160;
        let cr = complex_roots(&ints(&[1, 1, 1, 1, 1]), prec);
        assert_eq!(cr.roots.len(), 4);
        for z in &cr.roots {
            let ns = z.norm_sqr();
            assert_eq!(round_to_integer_if_close(&ns, prec - 24), Some(Integer::from(1)));
        }
        // determinism: recompute and compare coarse labels
        let cr2 = complex_roots(&ints(&[1, 1, 1, 1, 1]), prec);
        for (a, b) in cr.roots.iter().zip(cr2.roots.iter()) {
            assert_eq!(cmp_key(&a.re, 40), cmp_key(&b.re, 40));
            assert_eq!(cmp_key(&a.im, 40), cmp_key(&b.im, 40));
        }
    }

    #[test]
    fn complex_roots_degree12_real_count() {
        // L(X) = Π_{i=1}^{6}(X^2 - i^2): degree 12, 12 real roots ±1..±6.
        let mut l = ints(&[1]);
        for i in 1..=6i64 {
            let factor = ints(&[-i * i, 0, 1]);
            l = poly_mul(&l, &factor);
        }
        let prec = 160;
        let cr = complex_roots(&l, prec);
        assert_eq!(cr.roots.len(), 12);
        let mut real_count = 0;
        for z in &cr.roots {
            if round_to_integer_if_close(&z.im, prec - 30) == Some(Integer::from(0)) {
                real_count += 1;
            }
        }
        assert_eq!(real_count, 12);
        // roots are exactly ±1..±6 (sorted ascending): -6..-1, 1..6
        let expected = [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6];
        for (z, &e) in cr.roots.iter().zip(expected.iter()) {
            assert_eq!(
                round_to_integer_if_close(&z.re, prec - 30),
                Some(Integer::from(e)),
                "root mismatch"
            );
        }
    }

    // ---- p-adic roots ----

    #[test]
    fn padic_roots_sqrt2_in_z7() {
        // x^2 - 2 over Z_7: 2 is a QR mod 7 (3^2=2, 4^2=2). Two roots.
        let n = 6u32;
        let roots = padic_roots(&ints(&[-2, 0, 1]), 7, n);
        assert_eq!(roots.len(), 2);
        let p_n = Integer::from(7).pow(n);
        for r in &roots {
            let v = &r.value;
            let val = &(&(v * v) - &Integer::from(2)) % &p_n;
            let val = &(&val + &p_n) % &p_n;
            assert!(val.is_zero(), "root^2 != 2 mod 7^{}", n);
            let r7 = v.to_i64() % 7;
            assert!(r7 == 3 || r7 == 4);
        }
        assert!(roots[0].value < roots[1].value);
    }

    #[test]
    fn padic_roots_split_cubic() {
        // f = (x-1)(x-2)(x-4): three distinct simple roots mod 7.
        let lin = |c: i64| ints(&[-c, 1]);
        let f = poly_mul(&poly_mul(&lin(1), &lin(2)), &lin(4));
        let n = 4u32;
        let roots = padic_roots(&f, 7, n);
        assert_eq!(roots.len(), 3);
        let p_n = Integer::from(7).pow(n);
        for r in &roots {
            assert!(eval_int_mod(&f, &r.value, &p_n).is_zero(), "f(root) != 0 mod 7^{}", n);
        }
        let mut res: Vec<i64> = roots.iter().map(|r| r.value.to_i64() % 7).collect();
        res.sort();
        assert_eq!(res, vec![1, 2, 4]);
    }

    #[test]
    fn padic_roots_no_root_mod_p() {
        // x^2 + 1 mod 7: -1 is not a QR mod 7 → no Z_7 roots.
        let roots = padic_roots(&ints(&[1, 0, 1]), 7, 4);
        assert_eq!(roots.len(), 0);
    }

    #[test]
    fn padic_roots_skip_repeated_factor() {
        // f = x^2 (double root 0 mod p) → not simple, skipped.
        let roots = padic_roots(&ints(&[0, 0, 1]), 5, 3);
        assert_eq!(roots.len(), 0);
    }

    // ---- round-to-integer helper, borderline cases ----

    #[test]
    fn round_to_integer_borderline() {
        let prec = 100;
        let five = BigFloat::from_i64(5, prec);
        assert_eq!(round_to_integer_if_close(&five, 80), Some(Integer::from(5)));
        // 5 + 2^-90: within 2^-80 tolerance → 5
        let tiny = BigFloat { mantissa: ione(), prec: 90 };
        assert_eq!(round_to_integer_if_close(&five.add(&tiny), 80), Some(Integer::from(5)));
        // 5 + 2^-50: NOT within 2^-80 tolerance → None
        let big = BigFloat { mantissa: ione(), prec: 50 };
        assert_eq!(round_to_integer_if_close(&five.add(&big), 80), None);
        // 5.5: clearly not an integer
        let half = BigFloat::from_i64(1, prec).div(&BigFloat::from_i64(2, prec));
        assert_eq!(round_to_integer_if_close(&five.add(&half), 80), None);
    }

    #[test]
    fn complex_round_to_integer() {
        let prec = 120;
        let z = BigComplex::new(
            BigFloat::from_i64(3, prec).add(&BigFloat { mantissa: ione(), prec: 100 }),
            BigFloat { mantissa: ione(), prec: 100 },
        );
        assert_eq!(complex_round_to_integer_if_close(&z, 80), Some(Integer::from(3)));
        // 3 + 1i → not an integer (nonzero im)
        let z2 = BigComplex::new(BigFloat::from_i64(3, prec), BigFloat::from_i64(1, prec));
        assert_eq!(complex_round_to_integer_if_close(&z2, 80), None);
    }

    #[test]
    fn rational_reconstruction_basic() {
        let prec = 120;
        let q = Rational::new(22i64, 7i64).unwrap();
        let bf = rational_to_bigfloat(&q, prec);
        let rec = rational_reconstruction(&bf, &Integer::from(100), 60).unwrap();
        assert_eq!(rec.numerator(), &Integer::from(22));
        assert_eq!(rec.denominator(), &Integer::from(7));
        // 1/3
        let q2 = Rational::new(1i64, 3i64).unwrap();
        let bf2 = rational_to_bigfloat(&q2, prec);
        assert_eq!(rational_reconstruction(&bf2, &Integer::from(50), 60).unwrap(), q2);
        // integer 5 → 5/1
        let bf3 = BigFloat::from_i64(5, prec);
        let rec3 = rational_reconstruction(&bf3, &Integer::from(10), 60).unwrap();
        assert_eq!(rec3.numerator(), &Integer::from(5));
        assert_eq!(rec3.denominator(), &Integer::from(1));
    }

    #[test]
    fn rational_reconstruction_denominator_too_small() {
        let prec = 120;
        // 1/7 but only allow denom up to 3 → should fail.
        let q = Rational::new(1i64, 7i64).unwrap();
        let bf = rational_to_bigfloat(&q, prec);
        assert!(rational_reconstruction(&bf, &Integer::from(3), 60).is_none());
    }

    #[test]
    fn convenience_wrappers() {
        let poly = UnivariatePolynomial::new(ints(&[-2, 0, 1]));
        let cr = complex_roots_of(&poly, 120);
        assert_eq!(cr.roots.len(), 2);
        let pr = padic_roots_of(&poly, 7, 3);
        assert_eq!(pr.len(), 2);
    }
}
