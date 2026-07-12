//! Division polynomials ψ_n, φ_n, ω_n on a general Weierstrass curve over Q,
//! and the fibre of multiplication-by-n ([`EllipticCurve::divide_point`]).
//!
//! # The polynomials
//!
//! For E: y² + a₁xy + a₃y = x³ + a₂x² + a₄x + a₆ the division polynomials
//! live in the coordinate ring Z[x, y]/(E). Writing
//!
//! ```text
//! ψ_2 = 2y + a₁x + a₃,      F(x) := ψ_2² = 4x³ + b₂x² + 2b₄x + b₆,
//! ```
//!
//! every ψ_n is either a polynomial in x alone (n odd) or ψ_2 times one
//! (n even). We therefore store the **reduced** polynomials f_n ∈ Z[x],
//!
//! ```text
//! ψ_n = f_n(x)             (n odd),        ψ_n = f_n(x)·ψ_2   (n even),
//! ```
//!
//! with f_0 = 0, f_1 = f_2 = 1,
//! f_3 = 3x⁴ + b₂x³ + 3b₄x² + 3b₆x + b₈,
//! f_4 = 2x⁶ + b₂x⁵ + 5b₄x⁴ + 10b₆x³ + 10b₈x² + (b₂b₈ − b₄b₆)x + (b₄b₈ − b₆²),
//! and the two recursions, rewritten for f (the ψ_2-powers are absorbed
//! into F = ψ_2², which is why the odd case splits on the parity of m):
//!
//! ```text
//! f_{2m+1} = f_{m+2}f_m³·F² − f_{m−1}f_{m+1}³        (m even)
//! f_{2m+1} = f_{m+2}f_m³    − f_{m−1}f_{m+1}³·F²     (m odd)
//! f_{2m}   = f_m·( f_{m+2}f_{m−1}² − f_{m−2}f_{m+1}² )   (either parity)
//! ```
//!
//! Then, entirely inside Z[x],
//!
//! ```text
//! D_n := ψ_n² = f_n²        (n odd)   or   f_n²·F   (n even)      deg n²−1
//! φ_n  = x·D_n − ψ_{n+1}ψ_{n−1}
//!      = x·D_n − f_{n+1}f_{n−1}·F  (n odd)  or  x·D_n − f_{n+1}f_{n−1}  (n even)
//!                                                                 deg n², monic
//! ```
//!
//! and [n]P = ( φ_n(x)/ψ_n(x)² , ω_n(x,y)/ψ_n(x,y)³ ) with
//! ω_n = ψ_{2n}/(2ψ_n) − (a₁φ_n + a₃ψ_n²)·ψ_n/2 (Silverman/Sage normalization;
//! `omega` returns it in the a(x) + b(x)·y representation over Q — the halves
//! are genuinely needed on models with a₁ or a₃ odd).
//!
//! Every one of these is gated in the tests against `scalar_mul` on actual
//! points, and against the classical short-Weierstrass ψ_2, ψ_3, ψ_4.
//!
//! # Dividing a point
//!
//! x(Q) for the Q with [n]Q = P are exactly the rational roots of
//! v·φ_n(X) − u·D_n(X), where x(P) = u/v in lowest terms — a degree-n²
//! polynomial with leading coefficient v. Each root is lifted to y (the
//! Weierstrass quadratic; `rational_sqrt` decides solvability exactly) and
//! then **re-multiplied**: only points with `is_on_curve(Q)` *and*
//! `[n]Q == P` are returned. That re-multiplication is a self-certifying
//! gate and it lives in the release path, not in a test.
//!
//! # The rational-root search (rigorous, and factorization-free)
//!
//! `rustmath-polynomials`' `rational_roots` applies the rational-root
//! theorem literally: it calls `divisors()` on the constant term. Here the
//! constant term of v·φ_n − u·D_n routinely has 30+ digits, and a single
//! large semiprime factor would hang the search. So this module uses a
//! different, factorization-free identity.
//!
//! Let F = Σ c_i X^i be primitive with leading coefficient L = c_d > 0. Any
//! rational root x = a/b in lowest terms has b | L (rational-root theorem),
//! hence **m := L·x is an integer**, and m is a root of the *monic* integer
//! polynomial H(t) = L^{d−1}·F(t/L) = Σ_i c_i L^{d−1−i} t^i. Two rigorous
//! facts then pin every rational root with no factoring at all:
//!
//! * **Bound.** Cauchy: |x| ≤ 1 + max_{i<d}|c_i|/L, so |m| ≤ L + max_{i<d}|c_i| =: B.
//! * **Sieve.** For any prime q ∤ L, m mod q is a root of H mod q. Root sets
//!   S_q are found by brute force (q ≈ 2^16, machine arithmetic); an empty S_q
//!   proves there is no rational root at all, which is the common case and
//!   makes saturation cheap.
//!
//! CRT the S_q for enough primes that ∏q > 2B, take the symmetric
//! representative m of each surviving residue, keep those with |m| ≤ B, and
//! test F(m/L) = 0 **exactly** in Z. Nothing here is heuristic: the candidate
//! set provably contains every rational root, and the final test is exact.

use crate::curve::{EllipticCurve, Point};
use rustmath_integers::Integer;
use rustmath_rationals::Rational;

// ---------------------------------------------------------------------------
// Dense little-endian polynomials over Z
// ---------------------------------------------------------------------------

type IPoly = Vec<Integer>;

fn ip_trim(mut p: IPoly) -> IPoly {
    while p.last().map(|c| c.is_zero()).unwrap_or(false) {
        p.pop();
    }
    p
}

fn ip_add(a: &IPoly, b: &IPoly) -> IPoly {
    let n = a.len().max(b.len());
    let mut r = Vec::with_capacity(n);
    for i in 0..n {
        let x = a.get(i).cloned().unwrap_or_else(Integer::zero);
        let y = b.get(i).cloned().unwrap_or_else(Integer::zero);
        r.push(x + y);
    }
    ip_trim(r)
}

fn ip_sub(a: &IPoly, b: &IPoly) -> IPoly {
    let n = a.len().max(b.len());
    let mut r = Vec::with_capacity(n);
    for i in 0..n {
        let x = a.get(i).cloned().unwrap_or_else(Integer::zero);
        let y = b.get(i).cloned().unwrap_or_else(Integer::zero);
        r.push(x - y);
    }
    ip_trim(r)
}

fn ip_mul(a: &IPoly, b: &IPoly) -> IPoly {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut r = vec![Integer::zero(); a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        if x.is_zero() {
            continue;
        }
        for (j, y) in b.iter().enumerate() {
            if y.is_zero() {
                continue;
            }
            r[i + j] = r[i + j].clone() + x * y;
        }
    }
    ip_trim(r)
}

fn ip_scale(a: &IPoly, k: &Integer) -> IPoly {
    if k.is_zero() {
        return Vec::new();
    }
    ip_trim(a.iter().map(|c| c * k).collect())
}

/// Multiply by x.
fn ip_shift1(a: &IPoly) -> IPoly {
    if a.is_empty() {
        return Vec::new();
    }
    let mut r = vec![Integer::zero()];
    r.extend_from_slice(a);
    ip_trim(r)
}

fn ip_pow(a: &IPoly, e: u32) -> IPoly {
    let mut r: IPoly = vec![Integer::one()];
    for _ in 0..e {
        r = ip_mul(&r, a);
    }
    r
}

#[cfg(test)]
fn ip_eval(a: &IPoly, x: &Rational) -> Rational {
    let mut acc = Rational::from_integer(Integer::zero());
    for c in a.iter().rev() {
        acc = acc * x.clone() + Rational::from_integer(c.clone());
    }
    acc
}

// ---------------------------------------------------------------------------
// The a(x) + b(x)·y representation of an element of Q[x, y]/(E)
// ---------------------------------------------------------------------------

/// An element a(x) + b(x)·y of the coordinate ring of E over Q, with dense
/// little-endian coefficient vectors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurvePoly {
    /// Coefficients of a(x): `a[i]` multiplies x^i.
    pub a: Vec<Rational>,
    /// Coefficients of b(x): `b[i]` multiplies x^i·y.
    pub b: Vec<Rational>,
}

impl CurvePoly {
    /// Evaluate at an affine point (x, y). No curve check is performed — the
    /// value is only meaningful on E.
    pub fn eval(&self, x: &Rational, y: &Rational) -> Rational {
        let ev = |c: &[Rational]| {
            let mut acc = Rational::from_integer(Integer::zero());
            for k in c.iter().rev() {
                acc = acc * x.clone() + k.clone();
            }
            acc
        };
        ev(&self.a) + ev(&self.b) * y.clone()
    }

    fn from_ipoly(p: &IPoly) -> Self {
        CurvePoly {
            a: p.iter()
                .map(|c| Rational::from_integer(c.clone()))
                .collect(),
            b: Vec::new(),
        }
    }
}

fn rp_scale(p: &IPoly, k: &Rational) -> Vec<Rational> {
    p.iter()
        .map(|c| Rational::from_integer(c.clone()) * k.clone())
        .collect()
}

// ---------------------------------------------------------------------------
// Division polynomials
// ---------------------------------------------------------------------------

/// The reduced division polynomials f_0 … f_N of a curve (see the module
/// docs), together with F = ψ_2². Built once and reused; building for N
/// costs O(N) polynomial multiplications of degree O(N²).
#[derive(Debug, Clone)]
pub struct DivisionPolynomials {
    f: Vec<IPoly>,
    fx: IPoly,
    a1: Integer,
    a3: Integer,
    nmax: u32,
}

impl EllipticCurve {
    /// Build the division polynomials up to index `nmax` (the tables
    /// internally reach nmax + 2, which [`DivisionPolynomials::omega`] needs).
    pub fn division_polynomials(&self, nmax: u32) -> DivisionPolynomials {
        let (b2, b4, b6, b8) = self.b_invariants();
        let top = (nmax.saturating_add(2)).max(4) as usize;
        let fx: IPoly = ip_trim(vec![
            b6.clone(),
            Integer::from(2) * b4.clone(),
            b2.clone(),
            Integer::from(4),
        ]);

        let mut f: Vec<IPoly> = Vec::with_capacity(top + 1);
        f.push(Vec::new()); // f_0 = 0
        f.push(vec![Integer::one()]); // f_1 = 1
        f.push(vec![Integer::one()]); // f_2 = 1
        f.push(ip_trim(vec![
            b8.clone(),
            Integer::from(3) * b6.clone(),
            Integer::from(3) * b4.clone(),
            b2.clone(),
            Integer::from(3),
        ])); // f_3
        f.push(ip_trim(vec![
            b4.clone() * b8.clone() - b6.clone() * b6.clone(),
            b2.clone() * b8.clone() - b4.clone() * b6.clone(),
            Integer::from(10) * b8.clone(),
            Integer::from(10) * b6.clone(),
            Integer::from(5) * b4.clone(),
            b2.clone(),
            Integer::from(2),
        ])); // f_4

        let f2sq = ip_mul(&fx, &fx);
        for k in 5..=top {
            let next = if k % 2 == 1 {
                let m = (k - 1) / 2;
                let left = ip_mul(&f[m + 2], &ip_pow(&f[m], 3));
                let right = ip_mul(&f[m - 1], &ip_pow(&f[m + 1], 3));
                // ψ_m carries a ψ_2 exactly when m is even; the surplus ψ_2⁴ = F²
                // therefore rides on whichever term has the even indices.
                if m % 2 == 0 {
                    ip_sub(&ip_mul(&left, &f2sq), &right)
                } else {
                    ip_sub(&left, &ip_mul(&right, &f2sq))
                }
            } else {
                let m = k / 2;
                let inner = ip_sub(
                    &ip_mul(&f[m + 2], &ip_pow(&f[m - 1], 2)),
                    &ip_mul(&f[m - 2], &ip_pow(&f[m + 1], 2)),
                );
                ip_mul(&f[m], &inner)
            };
            f.push(next);
        }

        DivisionPolynomials {
            f,
            fx,
            a1: self.a1.clone(),
            a3: self.a3.clone(),
            nmax,
        }
    }
}

impl DivisionPolynomials {
    fn check(&self, n: u32) {
        assert!(
            n <= self.nmax,
            "division polynomial index {} exceeds the table built for nmax = {}",
            n,
            self.nmax
        );
    }

    /// D_n = ψ_n² ∈ Z[x] (degree n² − 1 for n ≥ 1).
    pub fn psi_squared(&self, n: u32) -> Vec<Integer> {
        self.check(n);
        let fk = &self.f[n as usize];
        let sq = ip_mul(fk, fk);
        if n.is_multiple_of(2) {
            ip_mul(&sq, &self.fx)
        } else {
            sq
        }
    }

    /// φ_n ∈ Z[x], monic of degree n²: x([n]P) = φ_n(x)/ψ_n(x)².
    pub fn phi(&self, n: u32) -> Vec<Integer> {
        self.check(n);
        assert!(n >= 1, "phi_n needs n >= 1");
        let d = self.psi_squared(n);
        let cross = ip_mul(&self.f[n as usize + 1], &self.f[n as usize - 1]);
        let cross = if n % 2 == 1 {
            ip_mul(&cross, &self.fx)
        } else {
            cross
        };
        ip_sub(&ip_shift1(&d), &cross)
    }

    /// ψ_n as an element a(x) + b(x)·y of the coordinate ring.
    pub fn psi(&self, n: u32) -> CurvePoly {
        self.check(n);
        let fk = &self.f[n as usize];
        if n % 2 == 1 {
            CurvePoly::from_ipoly(fk)
        } else {
            // f_n · (2y + a₁x + a₃)
            let lin: IPoly = ip_trim(vec![self.a3.clone(), self.a1.clone()]);
            CurvePoly {
                a: ip_mul(fk, &lin)
                    .iter()
                    .map(|c| Rational::from_integer(c.clone()))
                    .collect(),
                b: rp_scale(fk, &Rational::from_i64(2)),
            }
        }
    }

    /// ω_n = ψ_{2n}/(2ψ_n) − (a₁φ_n + a₃ψ_n²)·ψ_n/2, so that
    /// y([n]P) = ω_n(x, y)/ψ_n(x, y)³.
    pub fn omega(&self, n: u32) -> CurvePoly {
        self.check(n);
        assert!(n >= 2, "omega_n is built from f_{{n±2}}; use n >= 2");
        let k = n as usize;
        // U = f_{n+2}f_{n−1}² − f_{n−2}f_{n+1}²  (so ψ_{2n}/ψ_n = ψ_2·U or U·F/ψ_2)
        let u = ip_sub(
            &ip_mul(&self.f[k + 2], &ip_pow(&self.f[k - 1], 2)),
            &ip_mul(&self.f[k - 2], &ip_pow(&self.f[k + 1], 2)),
        );
        let half = Rational::new(Integer::one(), Integer::from(2)).expect("2 != 0");
        let phi = self.phi(n);
        let dsq = self.psi_squared(n);
        let lin: IPoly = ip_trim(vec![self.a3.clone(), self.a1.clone()]); // a₁x + a₃
                                                                          // C := a₁φ_n + a₃ψ_n²   (a polynomial in x)
        let c = ip_add(&ip_scale(&phi, &self.a1), &ip_scale(&dsq, &self.a3));

        if n % 2 == 1 {
            // ψ_{2n}/(2ψ_n) = ψ_2·U/2 = (y + (a₁x+a₃)/2)·U ; ψ_n = f_n
            // ω_n = U·y + ½[(a₁x+a₃)U − C·f_n]
            let cf = ip_mul(&c, &self.f[k]);
            let arest = ip_sub(&ip_mul(&lin, &u), &cf);
            CurvePoly {
                a: rp_scale(&arest, &half),
                b: rp_scale(&u, &Rational::from_i64(1)),
            }
        } else {
            // ψ_{2n}/(2ψ_n) = U/2 ; ψ_n = f_n·ψ_2 = f_n(2y + a₁x + a₃)
            // ω_n = ½U − ½·C·f_n·(2y + a₁x + a₃)
            //     = −C·f_n·y + ½[U − C·f_n·(a₁x + a₃)]
            let cf = ip_mul(&c, &self.f[k]);
            let arest = ip_sub(&u, &ip_mul(&cf, &lin));
            CurvePoly {
                a: rp_scale(&arest, &half),
                b: rp_scale(&cf, &Rational::from_i64(-1)),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Rational roots of an integer polynomial: rigorous, factorization-free
// ---------------------------------------------------------------------------

/// Guard on the CRT candidate set: exceeding it is an honest `Err`, never a
/// silently truncated (and therefore possibly incomplete) root list.
const MAX_CANDIDATES: usize = 400_000;
/// Guard on the number of sieve primes (the bound B would have to exceed
/// 65536^32 ≈ 10^154 for this to bite).
const MAX_SIEVE_PRIMES: usize = 32;

fn is_small_prime(q: u64) -> bool {
    if q < 2 {
        return false;
    }
    let mut d = 2u64;
    while d * d <= q {
        if q.is_multiple_of(d) {
            return false;
        }
        d += 1;
    }
    true
}

/// Primes just below 2^16, descending.
fn sieve_primes(count: usize) -> Vec<u64> {
    let mut out = Vec::with_capacity(count);
    let mut q = 65535u64;
    while out.len() < count && q > 3 {
        if is_small_prime(q) {
            out.push(q);
        }
        q -= 2;
    }
    out
}

fn imod(a: &Integer, q: u64) -> u64 {
    let m = Integer::from(q as i64);
    let r = (a % &m).to_i64();
    if r < 0 {
        (r + q as i64) as u64
    } else {
        r as u64
    }
}

fn powmod(mut b: u64, mut e: u64, q: u64) -> u64 {
    let mut r = 1u64;
    b %= q;
    while e > 0 {
        if e & 1 == 1 {
            r = r * b % q;
        }
        b = b * b % q;
        e >>= 1;
    }
    r
}

fn invmod(a: u64, q: u64) -> u64 {
    powmod(a % q, q - 2, q)
}

/// All rational roots of the integer polynomial `coeffs` (little-endian).
///
/// Rigorous and factorization-free — see the module docs. `Err` only when the
/// input is the zero polynomial (every rational is a root) or when a guard
/// (candidate-set size, sieve-prime count) is hit; never a truncated list.
pub fn rational_roots_of_integer_poly(coeffs: &[Integer]) -> Result<Vec<Rational>, String> {
    let mut c = ip_trim(coeffs.to_vec());
    if c.is_empty() {
        return Err("rational_roots: the zero polynomial has every rational as a root".to_string());
    }
    let mut roots: Vec<Rational> = Vec::new();

    // Split off the x^k factor.
    let mut k0 = 0usize;
    while k0 < c.len() && c[k0].is_zero() {
        k0 += 1;
    }
    if k0 > 0 {
        roots.push(Rational::from_integer(Integer::zero()));
        c = c[k0..].to_vec();
    }
    let d = c.len() - 1;
    if d == 0 {
        return Ok(roots);
    }

    // Primitive part, positive leading coefficient.
    let mut g = Integer::zero();
    for x in &c {
        g = g.gcd(x);
    }
    if !g.is_zero() && !g.is_one() {
        for x in c.iter_mut() {
            *x = x.clone() / g.clone();
        }
    }
    if c[d].signum() < 0 {
        for x in c.iter_mut() {
            *x = -x.clone();
        }
    }
    let lead = c[d].clone();

    // Cauchy: |x| <= 1 + max_{i<d}|c_i|/L, hence |m| = L|x| <= L + max_{i<d}|c_i|.
    let mut mx = Integer::zero();
    for ci in c.iter().take(d) {
        let a = ci.abs();
        if a > mx {
            mx = a;
        }
    }
    let bound = lead.clone() + mx;
    let need = Integer::from(2) * bound.clone() + Integer::one();

    // Sieve: roots of the monic H(t) = sum_i c_i L^{d-1-i} t^i modulo q.
    let primes = sieve_primes(MAX_SIEVE_PRIMES);
    let mut sets: Vec<(u64, Vec<u64>)> = Vec::new();
    let mut modulus = Integer::one();
    for &q in &primes {
        if modulus >= need {
            break;
        }
        if imod(&lead, q) == 0 {
            continue;
        }
        let lq = imod(&lead, q);
        // h_i = c_i * L^{d-1-i} mod q for i < d ; h_d = 1
        let mut h = vec![0u64; d + 1];
        for (i, hi) in h.iter_mut().enumerate().take(d) {
            *hi = imod(&c[i], q) * powmod(lq, (d - 1 - i) as u64, q) % q;
        }
        h[d] = 1;
        let mut s: Vec<u64> = Vec::new();
        for t in 0..q {
            let mut acc = 1u64; // h[d]
            for i in (0..d).rev() {
                acc = (acc * t + h[i]) % q;
            }
            if acc == 0 {
                s.push(t);
            }
        }
        if s.is_empty() {
            // No root mod q ⇒ no rational root at all.
            return Ok(roots);
        }
        modulus = modulus * Integer::from(q as i64);
        sets.push((q, s));
    }
    if modulus < need {
        return Err(format!(
            "rational_roots: ran out of sieve primes before the CRT modulus covered the \
             root bound ({} digits) — refusing (guard)",
            bound.to_string().len()
        ));
    }

    // Cheapest sets first keeps the intermediate CRT product small.
    sets.sort_by_key(|(_, s)| s.len());

    let mut cur_mod = Integer::one();
    let mut cands: Vec<Integer> = vec![Integer::zero()];
    for (q, s) in &sets {
        let qi = Integer::from(*q as i64);
        let minv = invmod(imod(&cur_mod, *q), *q);
        // Checked BEFORE materializing: `cands.len() * s.len()` Integers would otherwise be
        // allocated in full before any guard could fire.
        let width = cands.len().saturating_mul(s.len());
        if width > MAX_CANDIDATES {
            return Err(format!(
                "rational_roots: CRT candidate set would exceed {} — refusing (guard)",
                MAX_CANDIDATES
            ));
        }
        let mut next = Vec::with_capacity(width);
        for r in &cands {
            let rq = imod(r, *q);
            for &t in s {
                let delta = (t + *q - rq) % *q;
                let k = delta * minv % *q;
                next.push(r.clone() + cur_mod.clone() * Integer::from(k as i64));
            }
        }
        cands = next;
        cur_mod = cur_mod * qi;
    }

    // Symmetric representatives inside the bound, then an EXACT test:
    // F(m/L) = 0  <=>  sum_i c_i m^i L^{d-i} = 0.
    let half = cur_mod.clone() / Integer::from(2);
    for r in cands {
        let m = if r > half { r - cur_mod.clone() } else { r };
        if m.abs() > bound {
            continue;
        }
        let mut acc = Integer::zero();
        let mut mpow = Integer::one();
        for (i, ci) in c.iter().enumerate() {
            acc = acc + ci.clone() * mpow.clone() * lead.pow((d - i) as u32);
            mpow = mpow * m.clone();
        }
        if acc.is_zero() {
            let root = Rational::new(m, lead.clone()).expect("leading coefficient is nonzero");
            if !roots.contains(&root) {
                roots.push(root);
            }
        }
    }
    Ok(roots)
}

// ---------------------------------------------------------------------------
// Point division
// ---------------------------------------------------------------------------

impl EllipticCurve {
    /// The affine points of E(Q) with the given x-coordinate (0, 1 or 2 of
    /// them): the rational roots of y² + (a₁x + a₃)y − (x³+a₂x²+a₄x+a₆).
    pub fn lift_x(&self, x: &Rational) -> Vec<Point> {
        let q = Rational::from_integer;
        let s = q(self.a1.clone()) * x.clone() + q(self.a3.clone());
        let f = x.clone() * x.clone() * x.clone()
            + q(self.a2.clone()) * x.clone() * x.clone()
            + q(self.a4.clone()) * x.clone()
            + q(self.a6.clone());
        let disc = s.clone() * s.clone() + Rational::from_i64(4) * f;
        let Some(root) = crate::rank::rational_sqrt(&disc) else {
            return Vec::new();
        };
        let two = Rational::from_i64(2);
        let y1 = (-s.clone() + root.clone()) / two.clone();
        let y2 = (-s - root) / two;
        let mut out = vec![Point::new(x.clone(), y1.clone())];
        if y2 != y1 {
            out.push(Point::new(x.clone(), y2));
        }
        out
    }

    /// All rational Q with [n]Q = P — the fibre of multiplication-by-n over P.
    ///
    /// The x-coordinates are the rational roots of φ_n(X) − x(P)·ψ_n(X)²
    /// (degree n²); every candidate is lifted to y and then **verified by
    /// re-multiplication**: `is_on_curve(Q) && [n]Q == P`. That gate is in the
    /// release path, so a wrong root cannot be returned.
    ///
    /// For P = O this is E(Q)[n], read off the exact torsion subgroup.
    ///
    /// # Panics
    ///
    /// Panics if n = 0, or if the rational-root search hits an internal guard
    /// (see [`try_divide_point`](Self::try_divide_point) for the `Result` form).
    pub fn divide_point(&self, p: &Point, n: u32) -> Vec<Point> {
        self.try_divide_point(p, n)
            .expect("divide_point: rational-root search hit an internal guard")
    }

    /// [`divide_point`](Self::divide_point) with the guard failures surfaced
    /// as `Err` instead of a panic. An `Err` is never an empty fibre — it is
    /// "I could not decide", and callers (e.g. saturation) must propagate it.
    pub fn try_divide_point(&self, p: &Point, n: u32) -> Result<Vec<Point>, String> {
        assert!(n >= 1, "divide_point: n must be >= 1");
        assert!(self.is_on_curve(p), "divide_point: P is not on the curve");
        if n == 1 {
            return Ok(vec![p.clone()]);
        }
        if p.infinity {
            let t = self.torsion_subgroup();
            let mut out = vec![Point::infinity()];
            for (q, ord) in &t.points {
                if n.is_multiple_of(*ord) {
                    out.push(q.clone());
                }
            }
            return Ok(out);
        }

        let dp = self.division_polynomials(n);
        let phi = dp.phi(n);
        let dsq = dp.psi_squared(n);
        let u = p.x.numerator().clone();
        let v = p.x.denominator().clone();
        let poly = ip_sub(&ip_scale(&phi, &v), &ip_scale(&dsq, &u));
        let xs = rational_roots_of_integer_poly(&poly)?;

        let nn = Integer::from(n as i64);
        let mut out: Vec<Point> = Vec::new();
        for x in xs {
            for q in self.lift_x(&x) {
                // Self-certifying gate: on-curve AND [n]Q == P. Release path.
                if self.is_on_curve(&q) && self.scalar_mul(&nn, &q) == *p && !out.contains(&q) {
                    out.push(q);
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve(a: [i64; 5]) -> EllipticCurve {
        EllipticCurve::new(
            Integer::from(a[0]),
            Integer::from(a[1]),
            Integer::from(a[2]),
            Integer::from(a[3]),
            Integer::from(a[4]),
        )
    }

    fn ipoly(v: &[i64]) -> IPoly {
        ip_trim(v.iter().map(|&c| Integer::from(c)).collect())
    }

    /// The classical short-Weierstrass division polynomials for
    /// y² = x³ + ax + b (Silverman AEC ex. 3.7 / Washington 3.6):
    ///   ψ_2 = 2y, ψ_3 = 3x⁴ + 6ax² + 12bx − a²,
    ///   ψ_4 = 4y(x⁶ + 5ax⁴ + 20bx³ − 5a²x² − 4abx − 8b² − a³).
    /// Derived from the textbook formulas, not from the code below.
    #[test]
    fn test_short_weierstrass_psi_tables() {
        // y² = x³ − 2x + 3  (a = −2, b = 3)
        let e = curve([0, 0, 0, -2, 3]);
        let dp = e.division_polynomials(4);
        assert_eq!(dp.f[1], ipoly(&[1]));
        assert_eq!(dp.f[2], ipoly(&[1]));
        // 3x⁴ + 6(−2)x² + 12(3)x − 4 = 3x⁴ − 12x² + 36x − 4
        assert_eq!(dp.f[3], ipoly(&[-4, 36, -12, 0, 3]));
        // ψ_4/ψ_2 = 2(x⁶ + 5ax⁴ + 20bx³ − 5a²x² − 4abx − 8b² − a³)
        //         = 2x⁶ − 20x⁴ + 120x³ − 40x² + 48x − 128
        assert_eq!(dp.f[4], ipoly(&[-128, 48, -40, 120, -20, 0, 2]));
        // φ_2 = x⁴ − b4x² − 2b6x − b8, with b4 = 2a = −4, b6 = 4b = 12, b8 = −a² = −4
        assert_eq!(dp.phi(2), ipoly(&[4, -24, 4, 0, 1]));
    }

    /// The degree ladder: deg φ_n = n², deg ψ_n² = n² − 1.
    #[test]
    fn test_division_polynomial_degrees() {
        let e = curve([0, 1, 1, -2, 0]); // 389a1, a general model
        let dp = e.division_polynomials(7);
        for n in 1..=7u32 {
            assert_eq!(
                dp.phi(n).len() - 1,
                (n * n) as usize,
                "deg phi_{} should be {}",
                n,
                n * n
            );
            assert!(dp.phi(n)[(n * n) as usize].is_one(), "phi_{} is monic", n);
            assert_eq!(
                dp.psi_squared(n).len() - 1,
                (n * n - 1) as usize,
                "deg psi_{}^2 should be {}",
                n,
                n * n - 1
            );
        }
    }

    /// THE GATE ON φ, ψ, ω: for actual points P on general models,
    /// [n]P == ( φ_n(x)/ψ_n(x,y)², ω_n(x,y)/ψ_n(x,y)³ ), checked against the
    /// independently-implemented group law `scalar_mul`. Any sign slip in the
    /// a₁/a₃ terms (the classical trap on non-short models) shows up here.
    #[test]
    fn test_division_polynomials_reproduce_the_group_law() {
        #[allow(clippy::type_complexity)]
        let cases: [([i64; 5], [(i64, i64); 2]); 4] = [
            ([0, 1, 1, -2, 0], [(0, 0), (-1, 1)]), // 389a1
            ([0, 0, 1, -1, 0], [(0, 0), (1, 0)]),  // 37a1
            ([1, 0, 0, -1, 0], [(1, 0), (0, 0)]),  // 65a1 (a₁ = 1)
            // 15a1 (a₁ = a₂ = a₃ = 1); (−2,3) has order 4, so ψ_4 must vanish
            // there and ψ_2, ψ_3, ψ_5, ψ_6 must not — the O-case of the gate.
            ([1, 1, 1, -10, -10], [(8, 18), (-2, 3)]),
        ];
        for (model, pts) in &cases {
            let e = curve(*model);
            let dp = e.division_polynomials(6);
            for (px, py) in pts {
                let p = Point::from_integers(*px, *py);
                assert!(e.is_on_curve(&p), "{:?}: ({},{}) off curve", model, px, py);
                for n in 2..=6u32 {
                    let np = e.scalar_mul(&Integer::from(n as i64), &p);
                    let psi = dp.psi(n).eval(&p.x, &p.y);
                    if psi.numerator().is_zero() {
                        // ψ_n(P) = 0 <=> [n]P = O
                        assert!(np.infinity, "psi_{} vanishes but [n]P != O", n);
                        continue;
                    }
                    assert!(!np.infinity, "[{}]P = O but psi_{} != 0", n, n);
                    let x = ip_eval(&dp.phi(n), &p.x) / (psi.clone() * psi.clone());
                    let y = dp.omega(n).eval(&p.x, &p.y) / (psi.clone() * psi.clone() * psi);
                    assert_eq!(x, np.x, "{:?}: x([{}]P)", model, n);
                    assert_eq!(y, np.y, "{:?}: y([{}]P)", model, n);
                }
            }
        }
    }

    /// The rational-root engine against hand-built polynomials, including a
    /// non-monic case, a repeated root, an irrational-root case, and a
    /// deliberately large constant term whose factorization is out of reach
    /// of trial division (2^61 − 1 is prime; the point is that we never try).
    #[test]
    fn test_rational_roots_engine() {
        // 6x³ − 5x² − 17x + 6 = (2x + 3)(3x − 1)(x − 2)
        let r = rational_roots_of_integer_poly(&ipoly(&[6, -17, -5, 6])).unwrap();
        assert_eq!(r.len(), 3);
        for (n, d) in [(-3i64, 2i64), (1, 3), (2, 1)] {
            let q = Rational::new(Integer::from(n), Integer::from(d)).unwrap();
            assert!(r.contains(&q), "missing root {}/{}", n, d);
        }

        // x² − 2: no rational roots
        assert!(rational_roots_of_integer_poly(&ipoly(&[-2, 0, 1]))
            .unwrap()
            .is_empty());

        // x³ (triple root at 0) — reported once
        assert_eq!(
            rational_roots_of_integer_poly(&ipoly(&[0, 0, 0, 1])).unwrap(),
            vec![Rational::from_i64(0)]
        );

        // (x − 3)·(x² + 2305843009213693951)  — huge constant term, one root
        let big = Integer::from(2305843009213693951i64);
        let poly = ip_mul(
            &ipoly(&[-3, 1]),
            &vec![big, Integer::zero(), Integer::one()],
        );
        assert_eq!(
            rational_roots_of_integer_poly(&poly).unwrap(),
            vec![Rational::from_i64(3)]
        );

        assert!(rational_roots_of_integer_poly(&[]).is_err());
    }

    /// divide_point on 37a1 (torsion trivial, so E[n](Q) = {O} for every n and
    /// the fibre of [n] over [n]P is the single point P). P = (0,0) is the
    /// Mordell–Weil generator, so it is itself divisible by nothing.
    #[test]
    fn test_divide_point_37a() {
        let e = curve([0, 0, 1, -1, 0]);
        assert_eq!(e.torsion_subgroup().order, 1);
        let p = Point::from_integers(0, 0);
        let two_p = e.double_point(&p);
        let fibre = e.divide_point(&two_p, 2);
        assert_eq!(
            fibre,
            vec![p.clone()],
            "fibre of [2] over 2P is P + E[2] = {{P}}"
        );

        assert!(
            e.divide_point(&p, 2).is_empty(),
            "the generator of 37a1 is not divisible by 2"
        );
        assert!(e.divide_point(&p, 3).is_empty());
        assert!(e.divide_point(&p, 5).is_empty());

        let three_p = e.scalar_mul(&Integer::from(3), &p);
        assert_eq!(e.divide_point(&three_p, 3), vec![p.clone()]);
        // −P is NOT in the fibre: [n](−P) = −[n]P ≠ [n]P for a non-2-torsion P.
        assert!(!e.divide_point(&two_p, 2).contains(&e.negate_point(&p)));
    }

    /// divide_point over O is E(Q)[n]. 15a1 has torsion Z/4 × Z/2 (order 8;
    /// PARI `elltors(ellinit([1,1,1,-10,-10]))` = [8, [4,2], …]), so E[2](Q)
    /// is the full (Z/2)² with 4 points, E[4](Q) is everything (the exponent
    /// is 4), and E[3](Q) = {O}. The order and the structure are re-derived
    /// here from the crate's exact torsion machinery.
    #[test]
    fn test_divide_point_over_infinity() {
        let e = curve([1, 1, 1, -10, -10]);
        let t = e.torsion_subgroup();
        assert_eq!(t.order, 8);
        assert_eq!(t.structure.invariants(), vec![2, 4]); // crate order is ascending
        assert_eq!(e.divide_point(&Point::infinity(), 2).len(), 4);
        assert_eq!(e.divide_point(&Point::infinity(), 4).len(), 8);
        assert_eq!(e.divide_point(&Point::infinity(), 8).len(), 8);
        assert_eq!(e.divide_point(&Point::infinity(), 3).len(), 1);
    }

    /// Torsion interacts with divisibility: on 65a1 (|T| = 2, generator
    /// (1,0) of infinite order) the fibre of [2] over 2P is P + E[2](Q) =
    /// {P, P + T} — two points, not one. Divisibility lives in E(Q), not in
    /// E(Q)/tors, which is exactly why the saturator sweeps the torsion coset.
    #[test]
    fn test_divide_point_sees_the_torsion_coset() {
        let e = curve([1, 0, 0, -1, 0]);
        let t = e.torsion_subgroup();
        assert_eq!(t.order, 2);
        let tors = t.points[0].0.clone();
        let p = Point::from_integers(1, 0);
        let two_p = e.double_point(&p);
        let fibre = e.divide_point(&two_p, 2);
        assert_eq!(fibre.len(), 2, "P + E[2](Q) has 2 elements");
        for q in &fibre {
            assert_eq!(e.double_point(q), two_p);
        }
        assert!(fibre.contains(&p));
        assert!(fibre.contains(&e.add_points(&p, &tors)));
    }

    /// 389a1: the generator (−1,1) is not divisible by 2, 3 or 5, and
    /// 5·(−1,1) divided by 5 returns it (torsion is trivial, so the fibre is
    /// the single point). Exercises n = 5 — a degree-25 rational-root search
    /// on a general model, with x(5P) = −22625407/11397376 (PARI `ellmul`)
    /// forcing a leading coefficient of 11397376 = 3376².
    #[test]
    fn test_divide_point_389a_degree_25() {
        let e = curve([0, 1, 1, -2, 0]);
        assert_eq!(e.torsion_subgroup().order, 1);
        let p = Point::from_integers(-1, 1);
        for n in [2u32, 3, 5] {
            assert!(
                e.divide_point(&p, n).is_empty(),
                "(-1,1) must not be divisible by {}",
                n
            );
        }
        let five_p = e.scalar_mul(&Integer::from(5), &p);
        assert_eq!(e.divide_point(&five_p, 5), vec![p]);
    }
}
