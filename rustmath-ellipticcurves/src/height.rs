//! Canonical (Néron–Tate) heights on E/Q, computed over
//! [`rustmath_reals::bigfloat::BigFloat`] at a requested precision.
//!
//! # Normalization
//!
//! We use the "x-coordinate" normalization used by Sage, PARI and the LMFDB:
//!
//! ```text
//! ĥ(P) = lim_{n→∞} 4^{−n} h(x([2ⁿ]P)),    h(p/q) = log max(|p|, |q|),
//! ```
//!
//! so that ĥ is a quadratic form (ĥ(mP) = m²ĥ(P)), ĥ(P) = 0 iff P is
//! torsion, and the regulator det(⟨P_i, P_j⟩) with
//! ⟨P, Q⟩ = (ĥ(P+Q) − ĥ(P) − ĥ(Q))/2 matches LMFDB regulators. (This is
//! twice the normalization of Silverman's book.)
//!
//! # Algorithm
//!
//! Let E_min be the global minimal model ([`crate::minimal`]) and map P
//! there (ĥ is invariant under Weierstrass isomorphism). Let
//! M = lcm_p c_p over the bad primes (Tamagawa numbers from Tate's
//! algorithm) and Q = [M]P_min, so that Q lies in E⁰(Q_p) (nonsingular
//! reduction) at *every* prime. For such a point the non-archimedean part
//! of the canonical height is exactly log(denominator(x(Q))) [Silverman,
//! *Advanced Topics*, VI.4.1: at a prime of nonsingular reduction on a
//! minimal model the local height is max(0, −v_p(x)) log p in our
//! normalization], hence
//!
//! ```text
//! ĥ(P) = ( λ̂_∞(Q) + log den(x(Q)) ) / M².
//! ```
//!
//! # The archimedean local height λ̂_∞ and its derived error bound
//!
//! Write b2, b4, b6, b8 for the b-invariants of E_min and for real x set
//!
//! ```text
//! F(x) = 4x³ + b2x² + 2b4x + b6        ( = (2y + a1x + a3)² )
//! G(x) = x⁴ − b4x² − 2b6x − b8,
//! ```
//!
//! so that x(2P) = G(x)/F(x). λ̂_∞ is the unique function on E(R) \ {O}
//! with λ̂_∞(2P) = 4λ̂_∞(P) − log|F(x(P))| and λ̂_∞ − μ bounded, where
//! μ(P) = log max(|x(P)|, 1). Iterating the functional equation,
//!
//! ```text
//! λ̂_∞(P) = Σ_{n=0}^{N−1} 4^{−(n+1)} log|F(xₙ)| + 4^{−N} λ̂_∞(2^N P),
//! xₙ = x([2ⁿ]P),
//! ```
//!
//! and replacing λ̂_∞(2^N P) by μ(x_N) gives a truncation error of at most
//! 4^{−N}·(4/3)·M_E, where M_E = sup_{E(R)} |e| for the one-step defect
//!
//! ```text
//! e(P) = (1/4)·log [ max(|G(x)|, |F(x)|) / max(|x|, 1)⁴ ].
//! ```
//!
//! (Proof: g = λ̂_∞ − μ satisfies g = (1/4) g∘[2] + e, so ‖g‖ ≤ (4/3)‖e‖.)
//! M_E is bounded *per curve*, fully rigorously, as follows:
//!
//! * upper side: max(|G|, |F|) ≤ C_up · max(|x|,1)⁴ with
//!   C_up = max(1 + |b4| + 2|b6| + |b8|, 4 + |b2| + 2|b4| + |b6|);
//! * lower side, |x| ≥ R₀ := max(1, 2(|b4| + 2|b6| + |b8|)): |G(x)| ≥ x⁴/2;
//! * lower side, |x| ≤ R₀: G and F are coprime in Q[x] (their resultant is
//!   a nonzero multiple of Δ²), so extended Euclid gives A, B ∈ Q[x] with
//!   A·G + B·F = 1, whence max(|G|, |F|) ≥ 1/(S_A + S_B) on [−R₀, R₀] with
//!   S_A = Σ|A_i|R₀^i, S_B = Σ|B_i|R₀^i.
//!
//! Therefore M_E ≤ max( (1/4)log C_up, (1/4)(4 log R₀ + log(S_A+S_B)),
//! (1/4)log 2 ). N is chosen so that 4^{−N}(4/3)M_E ≤ 2^{−(prec+1)}.
//! The bound is evaluated in f64 with a ×1.01 safety factor; the series
//! itself runs in BigFloat with wp = prec + 2N + 64 bits (2 guard bits per
//! doubling step — the doubling map expands by ≈ 2 per step in the
//! uniformizing coordinate — plus margin), which controls rounding error
//! far below the truncation bound.
//!
//! # Verification
//!
//! The whole pipeline was verified against an independent exact-rational
//! duplication-limit computation in Python (n = 10 exact doublings with
//! tail estimate) on 37a1, 389a1 (both generators, their sum and
//! difference), y² = x³ − 2, y² = x³ − x + 1 and y² = x³ − 4x + 4, and
//! against a 60-digit mpmath prototype of this very algorithm; the
//! self-certifying gates (quadraticity, parallelogram law, vanishing on
//! torsion) are asserted in the tests at working precision.

use crate::curve::{EllipticCurve, Point};
use rustmath_core::analytic::RealField;
use rustmath_core::ordering::OrderedRing;
use rustmath_core::Ring;
use rustmath_integers::prime::factor;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use rustmath_reals::bigfloat::BigFloat;

impl EllipticCurve {
    /// The naive (Weil) height h(x(P)) = log max(|num x|, den x) of the
    /// x-coordinate, exact from the `Rational` and rounded to `prec_bits`.
    /// h(O) = 0.
    pub fn naive_height(p: &Point, prec_bits: u64) -> BigFloat {
        if p.infinity {
            return BigFloat::zero_prec(prec_bits);
        }
        let num = p.x.numerator().abs();
        let den = p.x.denominator().abs();
        let m = if num > den { num } else { den };
        if m.is_one() || m.is_zero() {
            return BigFloat::zero_prec(prec_bits);
        }
        BigFloat::from_integer(&m, prec_bits + 8)
            .ln()
            .with_precision(prec_bits)
    }

    /// The canonical (Néron–Tate) height ĥ(P) in the Sage/LMFDB
    /// normalization (see module docs), with absolute truncation error at
    /// most 2^{−prec_bits} (plus floating-point rounding controlled by
    /// generous guard bits). Exactly zero for torsion points.
    ///
    /// # Panics
    ///
    /// Panics if the curve is singular or `p` is not on the curve.
    pub fn canonical_height(&self, p: &Point, prec_bits: u64) -> BigFloat {
        assert!(
            !self.is_singular(),
            "canonical_height: curve is singular (discriminant 0)"
        );
        assert!(self.is_on_curve(p), "canonical_height: point not on curve");
        if self.point_order(p).is_some() {
            // Torsion (including O): ĥ = 0 exactly.
            return BigFloat::zero_prec(prec_bits);
        }

        // Work on the global minimal model.
        let (emin, iso) = self.minimal_model();
        let pmin = iso.map_point(p);

        // M = lcm of the Tamagawa numbers kills every component group, so
        // Q = [M]P_min has nonsingular reduction at every prime.
        let mut m = Integer::one();
        for (prime, _) in factor(&emin.discriminant.abs()) {
            let c = emin.local_data(&prime).tamagawa_number;
            m = m.lcm(&Integer::from(c as i64));
        }
        let q = emin.scalar_mul(&m, &pmin);
        assert!(!q.infinity, "canonical_height: non-torsion point hit O");

        let lam = archimedean_lambda(&emin, &q.x, prec_bits + 4);
        let wp = lam.prec();
        let den = q.x.denominator().abs();
        let nonarch = if den.is_one() {
            BigFloat::zero_prec(wp)
        } else {
            BigFloat::from_integer(&den, wp).ln()
        };
        let m2 = BigFloat::from_integer(&(&m * &m), wp);
        ((lam + nonarch) / m2).with_precision(prec_bits)
    }

    /// The Néron–Tate height pairing
    /// ⟨P, Q⟩ = (ĥ(P+Q) − ĥ(P) − ĥ(Q)) / 2.
    pub fn height_pairing(&self, p: &Point, q: &Point, prec_bits: u64) -> BigFloat {
        let wp = prec_bits + 8;
        let s = self.add_points(p, q);
        let hs = self.canonical_height(&s, wp);
        let hp = self.canonical_height(p, wp);
        let hq = self.canonical_height(q, wp);
        let two = BigFloat::from_integer(&Integer::from(2), wp);
        ((hs - hp - hq) / two).with_precision(prec_bits)
    }

    /// The regulator det(⟨P_i, P_j⟩) of a set of points (1 for the empty
    /// set). The points are *not* checked for independence; a dependent set
    /// yields a regulator ≈ 0.
    pub fn regulator(&self, points: &[Point], prec_bits: u64) -> BigFloat {
        let n = points.len();
        if n == 0 {
            return BigFloat::one_prec(prec_bits);
        }
        let wp = prec_bits + 16;
        let mut g = vec![vec![BigFloat::zero_prec(wp); n]; n];
        for i in 0..n {
            for j in i..n {
                let v = if i == j {
                    self.canonical_height(&points[i], wp)
                } else {
                    self.height_pairing(&points[i], &points[j], wp)
                };
                g[i][j] = v.clone();
                g[j][i] = v;
            }
        }
        det_bigfloat(g, wp).with_precision(prec_bits)
    }
}

/// Determinant by Gaussian elimination with partial pivoting.
pub(crate) fn det_bigfloat(mut a: Vec<Vec<BigFloat>>, wp: u64) -> BigFloat {
    let n = a.len();
    let mut det = BigFloat::one_prec(wp);
    for k in 0..n {
        // partial pivot
        let mut piv = k;
        for i in (k + 1)..n {
            if a[i][k].abs() > a[piv][k].abs() {
                piv = i;
            }
        }
        if a[piv][k].is_zero() {
            return BigFloat::zero_prec(wp);
        }
        if piv != k {
            a.swap(piv, k);
            det = -det;
        }
        det = det * a[k][k].clone();
        let row_k = a[k].clone();
        for row in a.iter_mut().skip(k + 1) {
            let f = row[k].clone() / row_k[k].clone();
            for (j, pivot_val) in row_k.iter().enumerate().skip(k) {
                row[j] = row[j].clone() - f.clone() * pivot_val.clone();
            }
        }
    }
    det
}

// ---------------------------------------------------------------------------
// Archimedean local height
// ---------------------------------------------------------------------------

/// λ̂_∞ evaluated at the real point with x-coordinate `x0` on the (minimal)
/// model `e`, with truncation error ≤ 2^{−(prec_bits+1)}; see module docs
/// for the derivation of the error bound.
fn archimedean_lambda(e: &EllipticCurve, x0: &Rational, prec_bits: u64) -> BigFloat {
    let (b2, b4, b6, b8) = e.b_invariants();

    // --- derived truncation bound (conservative f64; see module docs) ---
    let m_e = one_step_defect_bound(&b2, &b4, &b6, &b8);
    assert!(
        m_e.is_finite(),
        "archimedean_lambda: defect bound overflowed f64 (coefficients too large)"
    );
    // 4^{-N} (4/3) M_E <= 2^{-(prec+1)}  <=>  2N >= prec + 1 + log2(4 M_E / 3)
    let n_steps =
        (((prec_bits as f64) + 1.0 + (4.0 * m_e / 3.0).max(1.0).log2()) / 2.0).ceil() as u64 + 1;
    let wp = prec_bits + 2 * n_steps + 64;

    // --- the series ---
    let bf = |n: &Integer| BigFloat::from_integer(n, wp);
    let b2f = bf(&b2);
    let b4f2 = bf(&(Integer::from(2) * b4.clone()));
    let b4f = bf(&b4);
    let b6f = bf(&b6);
    let b6f2 = bf(&(Integer::from(2) * b6.clone()));
    let b8f = bf(&b8);
    let four = bf(&Integer::from(4));
    let one = BigFloat::one_prec(wp);
    let quarter = one.clone() / four.clone();

    let mut x = BigFloat::from_rational(x0, wp);
    let mut acc = BigFloat::zero_prec(wp);
    let mut scale = one.clone();
    for _ in 0..n_steps {
        // F = 4x³ + b2x² + 2b4x + b6 (Horner), G = x⁴ − b4x² − 2b6x − b8
        let f = ((four.clone() * x.clone() + b2f.clone()) * x.clone() + b4f2.clone()) * x.clone()
            + b6f.clone();
        let x2 = x.clone() * x.clone();
        let g = (x2.clone() - b4f.clone()) * x2 - b6f2.clone() * x.clone() - b8f.clone();
        assert!(
            !f.is_zero(),
            "archimedean_lambda: hit a 2-torsion x exactly (torsion point?)"
        );
        scale = scale * quarter.clone();
        acc = acc + scale.clone() * f.abs().ln();
        x = g / f;
    }
    // + 4^{-N} · log max(|x_N|, 1)
    let ax = x.abs();
    if ax > one {
        acc = acc + scale * ax.ln();
    }
    acc
}

/// Conservative f64 upper bound for M_E = sup |e| (see module docs).
fn one_step_defect_bound(b2: &Integer, b4: &Integer, b6: &Integer, b8: &Integer) -> f64 {
    let f = |n: &Integer| n.abs().to_f64().unwrap_or(f64::INFINITY);
    let (ab2, ab4, ab6, ab8) = (f(b2), f(b4), f(b6), f(b8));
    let k1 = ab4 + 2.0 * ab6 + ab8;
    let r0 = (2.0 * k1).max(1.0);
    let c_up = (1.0 + ab4 + 2.0 * ab6 + ab8).max(4.0 + ab2 + 2.0 * ab4 + ab6);

    // Bezout: A·G + B·F = 1 in Q[x].
    let q = |n: &Integer| Rational::from_integer(n.clone());
    let g_poly = vec![
        -q(b8),
        Rational::from_i64(-2) * q(b6),
        -q(b4),
        Rational::zero(),
        Rational::one(),
    ];
    let f_poly = vec![
        q(b6),
        Rational::from_i64(2) * q(b4),
        q(b2),
        Rational::from_i64(4),
    ];
    let (pa, pb) = poly_bezout(&g_poly, &f_poly);
    let sup_norm = |p: &[Rational]| -> f64 {
        let mut s = 0.0f64;
        let mut rk = 1.0f64;
        for c in p {
            s += c.abs().to_f64().unwrap_or(f64::INFINITY) * rk;
            rk *= r0;
        }
        s
    };
    let s_ab = sup_norm(&pa) + sup_norm(&pb);

    let cand = [
        0.25 * c_up.ln(),
        0.25 * (4.0 * r0.ln() + s_ab.ln()),
        0.25 * 2.0f64.ln(),
    ];
    let m = cand.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    // ×1.01 + ε safety for the f64 evaluation of the bound itself.
    m * 1.01 + 1e-9
}

// ---------------------------------------------------------------------------
// Tiny dense polynomial helpers over Q (degree ≤ 4), for the Bezout bound
// ---------------------------------------------------------------------------

type QPoly = Vec<Rational>; // little-endian coefficients, trimmed

fn ptrim(mut p: QPoly) -> QPoly {
    while p.last().is_some_and(|c| c.is_zero()) {
        p.pop();
    }
    p
}

fn psub(a: &[Rational], b: &[Rational]) -> QPoly {
    let n = a.len().max(b.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let av = a.get(i).cloned().unwrap_or_else(Rational::zero);
        let bv = b.get(i).cloned().unwrap_or_else(Rational::zero);
        out.push(av - bv);
    }
    ptrim(out)
}

fn pmul(a: &[Rational], b: &[Rational]) -> QPoly {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let mut out = vec![Rational::zero(); a.len() + b.len() - 1];
    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            out[i + j] = out[i + j].clone() + ai.clone() * bj.clone();
        }
    }
    ptrim(out)
}

/// (quotient, remainder) of a ÷ b in Q[x]; b non-zero.
fn pdivmod(a: &[Rational], b: &[Rational]) -> (QPoly, QPoly) {
    assert!(!b.is_empty(), "pdivmod: division by zero polynomial");
    let mut r: QPoly = a.to_vec();
    r = ptrim(r);
    let db = b.len() - 1;
    let lb = b[db].clone();
    if r.len() < b.len() {
        return (vec![], r);
    }
    let mut q = vec![Rational::zero(); r.len() - db];
    while r.len() >= b.len() && !r.is_empty() {
        let dr = r.len() - 1;
        let coef = r[dr].clone() / lb.clone();
        let shift = dr - db;
        q[shift] = coef.clone();
        // r -= coef * x^shift * b
        let mut sub = vec![Rational::zero(); shift];
        sub.extend(b.iter().map(|c| coef.clone() * c.clone()));
        r = psub(&r, &sub);
    }
    (ptrim(q), r)
}

/// For coprime a, b ∈ Q[x], return (s, t) with s·a + t·b = 1.
fn poly_bezout(a: &[Rational], b: &[Rational]) -> (QPoly, QPoly) {
    let mut r0: QPoly = ptrim(a.to_vec());
    let mut r1: QPoly = ptrim(b.to_vec());
    let mut s0: QPoly = vec![Rational::one()];
    let mut s1: QPoly = vec![];
    let mut t0: QPoly = vec![];
    let mut t1: QPoly = vec![Rational::one()];
    while !r1.is_empty() {
        let (q, r2) = pdivmod(&r0, &r1);
        let s2 = psub(&s0, &pmul(&q, &s1));
        let t2 = psub(&t0, &pmul(&q, &t1));
        r0 = std::mem::take(&mut r1);
        r1 = r2;
        s0 = std::mem::take(&mut s1);
        s1 = s2;
        t0 = std::mem::take(&mut t1);
        t1 = t2;
    }
    assert!(
        r0.len() == 1,
        "poly_bezout: inputs not coprime (gcd degree {})",
        r0.len().saturating_sub(1)
    );
    let c = r0[0].clone();
    let s = s0.into_iter().map(|v| v / c.clone()).collect();
    let t = t0.into_iter().map(|v| v / c.clone()).collect();
    (s, t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve(a1: i64, a2: i64, a3: i64, a4: i64, a6: i64) -> EllipticCurve {
        EllipticCurve::new(
            Integer::from(a1),
            Integer::from(a2),
            Integer::from(a3),
            Integer::from(a4),
            Integer::from(a6),
        )
    }

    fn tol_bits(bits: u32) -> BigFloat {
        BigFloat::one_prec(64) / BigFloat::from_integer(&Integer::from(2).pow(bits), 64)
    }

    fn assert_close_f64(v: &BigFloat, expected: f64, tol: f64, what: &str) {
        let d = (v.to_f64() - expected).abs();
        assert!(
            d < tol,
            "{}: got {}, expected {} (|diff| = {:.3e} ≥ {:.1e})",
            what,
            v.to_f64(),
            expected,
            d,
            tol
        );
    }

    // Expected values below were computed independently in Python before
    // being asserted: (a) an exact-rational duplication-limit computation
    // (n = 10 doublings, tail-estimated), and (b) a 60-digit mpmath
    // prototype of this algorithm; (a) and (b) agree to ~1e-7 and (b) is
    // quoted to 15+ digits. The 37a1 value also matches the published
    // Sage/LMFDB height 0.0511114082399688.

    #[test]
    fn height_37a1_generator() {
        let e = curve(0, 0, 1, -1, 0);
        let p = Point::from_integers(0, 0);
        let h = e.canonical_height(&p, 128);
        assert_close_f64(&h, 0.0511114082399688, 1e-13, "hhat 37a1 (0,0)");
    }

    #[test]
    fn height_of_torsion_is_exactly_zero() {
        let e11 = curve(0, -1, 1, -10, -20);
        let t = Point::from_integers(5, 5);
        assert!(e11.canonical_height(&t, 64).is_zero());
        assert!(e11.canonical_height(&Point::infinity(), 64).is_zero());
        // full 2-torsion curve
        let e = curve(0, 0, 0, -1, 0);
        assert!(e
            .canonical_height(&Point::from_integers(1, 0), 64)
            .is_zero());
    }

    #[test]
    fn height_quadraticity_gates() {
        // ĥ(2P) = 4ĥ(P) and ĥ(3P) = 9ĥ(P) to ~2^{-100} at 128-bit precision.
        // These exercise the archimedean series and the non-archimedean
        // decomposition across changing denominators.
        let cases = [
            (curve(0, 0, 1, -1, 0), Point::from_integers(0, 0)), // 37a1, M=1
            (curve(0, 0, 0, 0, -2), Point::from_integers(3, 5)), // Mordell, additive at 2,3
        ];
        let thr = tol_bits(100);
        for (e, p) in cases {
            let h1 = e.canonical_height(&p, 160);
            let four = BigFloat::from_integer(&Integer::from(4), 160);
            let nine = BigFloat::from_integer(&Integer::from(9), 160);
            let p2 = e.scalar_mul(&Integer::from(2), &p);
            let p3 = e.scalar_mul(&Integer::from(3), &p);
            let h2 = e.canonical_height(&p2, 160);
            let h3 = e.canonical_height(&p3, 160);
            let d2 = (h2 - four * h1.clone()).abs();
            let d3 = (h3 - nine * h1.clone()).abs();
            assert!(d2 < thr, "hhat(2P) != 4 hhat(P): diff {}", d2.to_f64());
            assert!(d3 < thr, "hhat(3P) != 9 hhat(P): diff {}", d3.to_f64());
        }
    }

    #[test]
    fn height_parallelogram_law_389a() {
        // 389a1 (rank 2), generators P = (0,0), Q = (−1,1):
        // ĥ(P+Q) + ĥ(P−Q) = 2ĥ(P) + 2ĥ(Q) to ~2^{-100}.
        let e = curve(0, 1, 1, -2, 0);
        let p = Point::from_integers(0, 0);
        let q = Point::from_integers(-1, 1);
        assert!(e.is_on_curve(&p) && e.is_on_curve(&q));
        let s = e.add_points(&p, &q);
        let d = e.add_points(&p, &e.negate_point(&q));
        let (hp, hq) = (e.canonical_height(&p, 160), e.canonical_height(&q, 160));
        let (hs, hd) = (e.canonical_height(&s, 160), e.canonical_height(&d, 160));
        let two = BigFloat::from_integer(&Integer::from(2), 160);
        let lhs = hs + hd;
        let rhs = two.clone() * hp + two * hq;
        let diff = (lhs - rhs).abs();
        assert!(
            diff < tol_bits(100),
            "parallelogram law violated: diff {}",
            diff.to_f64()
        );
    }

    #[test]
    fn height_389a_generators_and_regulator() {
        // Python ground truth (duplication limit + mpmath prototype):
        // ĥ(P) = 0.327000773651605, ĥ(Q) = 0.686667083305587,
        // regulator = 0.152460177943144.
        let e = curve(0, 1, 1, -2, 0);
        let p = Point::from_integers(0, 0);
        let q = Point::from_integers(-1, 1);
        let hp = e.canonical_height(&p, 128);
        let hq = e.canonical_height(&q, 128);
        assert_close_f64(&hp, 0.327000773651605, 1e-13, "hhat 389a P");
        assert_close_f64(&hq, 0.686667083305587, 1e-13, "hhat 389a Q");
        let reg = e.regulator(&[p, q], 128);
        assert_close_f64(&reg, 0.152460177943144, 1e-12, "389a regulator");
    }

    #[test]
    fn height_with_nontrivial_tamagawa_numbers() {
        // y² = x³ − x + 1, P = (1,1): c₂ = 3 (type IV at 2), so this
        // exercises the M = lcm(c_p) > 1 path. Python ground truth
        // 0.0498083972980648266 (M = 12 and M = 24 agree to 1e-57).
        let e = curve(0, 0, 0, -1, 1);
        let p = Point::from_integers(1, 1);
        let h = e.canonical_height(&p, 128);
        assert_close_f64(&h, 0.0498083972980648266, 1e-14, "hhat x3-x+1 (1,1)");

        // y² = x³ − 4x + 4, P = (1,1): Python ground truth 0.644229829390090287.
        let e2 = curve(0, 0, 0, -4, 4);
        let p2 = Point::from_integers(1, 1);
        let h2 = e2.canonical_height(&p2, 128);
        assert_close_f64(&h2, 0.644229829390090287, 1e-13, "hhat x3-4x+4 (1,1)");

        // y² = x³ − 2, P = (3,5): Python ground truth 1.34957683568011805.
        let e3 = curve(0, 0, 0, 0, -2);
        let p3 = Point::from_integers(3, 5);
        let h3 = e3.canonical_height(&p3, 128);
        assert_close_f64(&h3, 1.34957683568011805, 1e-13, "hhat mordell-2 (3,5)");
    }

    #[test]
    fn height_invariant_under_nonminimal_models() {
        // 37a1 scaled by λ = 2 (non-minimal model); the corresponding point
        // is (4·0, 8·0) = (0,0). ĥ must be model-independent.
        let e = curve(0, 0, 8, -16, 0);
        let p = Point::from_integers(0, 0);
        assert!(e.is_on_curve(&p));
        let h = e.canonical_height(&p, 128);
        assert_close_f64(&h, 0.0511114082399688, 1e-13, "hhat scaled 37a1");
    }

    #[test]
    fn naive_height_exact() {
        // h(x = 0) = 0; h(x = 3) = ln 3; h(x = 5/4) = ln 5.
        let p0 = Point::from_integers(0, 0);
        assert!(EllipticCurve::naive_height(&p0, 64).is_zero());
        let p3 = Point::from_integers(3, 5);
        assert_close_f64(
            &EllipticCurve::naive_height(&p3, 64),
            3.0f64.ln(),
            1e-15,
            "naive height x=3",
        );
        let pq = Point::new(
            Rational::new(Integer::from(5), Integer::from(4)).unwrap(),
            Rational::zero(),
        );
        assert_close_f64(
            &EllipticCurve::naive_height(&pq, 64),
            5.0f64.ln(),
            1e-15,
            "naive height x=5/4",
        );
        assert!(EllipticCurve::naive_height(&Point::infinity(), 64).is_zero());
    }

    #[test]
    fn height_pairing_matrix_consistency() {
        // ⟨P, P⟩ = ĥ(P) (pairing definition with S = 2P uses quadraticity).
        let e = curve(0, 1, 1, -2, 0);
        let p = Point::from_integers(0, 0);
        let hp = e.canonical_height(&p, 96);
        let pp = e.height_pairing(&p, &p, 96);
        let diff = (hp - pp).abs();
        assert!(diff < tol_bits(80), "<P,P> != hhat(P): {}", diff.to_f64());
        // regulator of an empty set is 1; of a dependent pair ~ 0.
        assert!(e.regulator(&[], 64).is_one());
        let p2 = e.scalar_mul(&Integer::from(2), &p);
        let reg = e.regulator(&[p.clone(), p2], 96);
        assert!(reg.abs() < tol_bits(60), "dependent regulator not ~0");
    }

    #[test]
    fn poly_bezout_certifies() {
        // A·G + B·F = 1 for 37a1's quartic/cubic pair.
        let e = curve(0, 0, 1, -1, 0);
        let (b2, b4, b6, b8) = e.b_invariants();
        let q = |n: &Integer| Rational::from_integer(n.clone());
        let g = vec![
            -q(&b8),
            Rational::from_i64(-2) * q(&b6),
            -q(&b4),
            Rational::zero(),
            Rational::one(),
        ];
        let f = vec![
            q(&b6),
            Rational::from_i64(2) * q(&b4),
            q(&b2),
            Rational::from_i64(4),
        ];
        let (s, t) = poly_bezout(&g, &f);
        let one = psub(
            &pmul(&s, &g),
            &pmul(&pmul(&t, &f), &[Rational::from_i64(-1)]),
        );
        assert_eq!(one, vec![Rational::one()]);
    }
}
