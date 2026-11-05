//! KMSV §4 (hypergeometric part): the triangle-group uniformizer φ as an exact
//! power series.
//!
//! For Δ = Δ(a,b,c) the quotient X(Δ) = Δ\ℍ is a genus-0 orbifold, and the
//! uniformizer φ : X(Δ) ≅ P¹ (elliptic points ↦ 0,1,∞, an a-to-1 map at w_a = 0)
//! has an explicit expansion around z_a. The functional inverse ψ = φ⁻¹ is a ratio
//! of solutions of the hypergeometric ₂F₁ differential equation (eq 4.9–4.14):
//!
//!   ψ(t)/κ = t^{1/a} · F(A,B,C;t) / F(1+A−C, 1+B−C, 2−C; t),  t = φ(w),
//!
//! with A,B,C rational (eq 4.10) and κ (eq 4.13) a ratio of Γ-values. Reverting and
//! raising to the a-th power gives (eq 4.16)
//!
//!   φ(w) = (w/κ)^a + c₂ (w/κ)^{2a} + …  ∈ ℚ[[(w/κ)^a]]   — RATIONAL in u = w/κ.
//!
//! Since κ factors out, the whole computation is exact rational power-series
//! arithmetic. The transcendental κ is only needed later (§5) to fix the scale.

use rug::{Assign, Float};
use rustmath_core::traits::Field;
use rustmath_powerseries::series::PowerSeries;
use rustmath_rationals::Rational;

fn rat(n: i64) -> Rational {
    Rational::from_i64(n)
}
fn ratio(n: i64, d: i64) -> Rational {
    Rational::new(n, d).expect("nonzero denominator")
}

/// The hypergeometric parameters (A, B, C) of eq (4.10) for Δ(a,b,c).
pub fn abc_params(a: i64, b: i64, c: i64) -> (Rational, Rational, Rational) {
    let one = rat(1);
    let half = ratio(1, 2);
    let big_a = half.clone() * (one.clone() + ratio(1, a) - ratio(1, b) - ratio(1, c));
    let big_b = half * (one.clone() + ratio(1, a) - ratio(1, b) + ratio(1, c));
    let big_c = one + ratio(1, a);
    (big_a, big_b, big_c)
}

/// The Gaussian hypergeometric series F(A,B,C;t) = Σ (A)_n(B)_n/((C)_n n!) tⁿ,
/// truncated to `prec` terms, over ℚ (requires C ∉ ℤ_{≤0}).
pub fn hyp_series(
    big_a: &Rational,
    big_b: &Rational,
    big_c: &Rational,
    prec: usize,
) -> PowerSeries<Rational> {
    let mut coeffs = Vec::with_capacity(prec);
    let mut c = rat(1);
    coeffs.push(c.clone());
    for n in 1..prec {
        let nm1 = rat((n - 1) as i64);
        // c_n = c_{n-1} · (A+n-1)(B+n-1) / ((C+n-1)·n)
        let num = (big_a.clone() + nm1.clone()) * (big_b.clone() + nm1.clone());
        let den = (big_c.clone() + nm1) * rat(n as i64);
        c = c.clone() * num / den;
        coeffs.push(c.clone());
    }
    PowerSeries::new(coeffs, prec)
}

/// Series reversion by Newton iteration: given `f` with f(0)=0 and f'(0)≠0, return
/// `g` with f(g(u)) = u to the tracked precision. Uses `compose`/`derivative`/
/// `inverse`, doubling correct terms each step.
pub fn revert(f: &PowerSeries<Rational>) -> PowerSeries<Rational> {
    let prec = f.precision();
    let a1_inv = f.coeff(1).inverse().expect("f'(0) ≠ 0 for reversion");
    // g₀ = u / a₁
    let mut g = PowerSeries::new(vec![rat(0), a1_inv], prec);
    let fp = f.derivative();
    let id = PowerSeries::new(vec![rat(0), rat(1)], prec); // the series u
    let iters = (prec as f64).log2().ceil() as usize + 2;
    for _ in 0..iters {
        // g ← g − (f∘g − u)/(f'∘g)
        let fg = f.compose(&g);
        let num = fg - id.clone();
        let denom = fp.compose(&g).inverse().expect("f'∘g invertible (const ≠ 0)");
        let corr = num * denom;
        g = (g.clone() - corr).truncate(prec);
    }
    g
}

fn pow_series(f: &PowerSeries<Rational>, k: usize) -> PowerSeries<Rational> {
    let prec = f.precision();
    let mut r = PowerSeries::new(vec![rat(1)], prec);
    for _ in 0..k {
        r = r * f.clone();
    }
    r
}

/// The uniformizer φ(w) of Δ(a,b,c) expanded around z_a as a rational power series
/// in `u = w/κ` (eq 4.16): φ = u^a + c₂ u^{2a} + …. Returns the series in `u`
/// (only exponents divisible by a are nonzero). `prec` = number of `u`-terms tracked.
pub fn phi_in_u(a: i64, b: i64, c: i64, prec: usize) -> PowerSeries<Rational> {
    let (big_a, big_b, big_c) = abc_params(a, b, c);
    let one = rat(1);
    // F₁ = F(A,B,C;t),  F₂ʰʸᵖ = F(1+A−C, 1+B−C, 2−C; t)
    let f1 = hyp_series(&big_a, &big_b, &big_c, prec);
    let a2 = one.clone() + big_a.clone() - big_c.clone();
    let b2 = one.clone() + big_b.clone() - big_c.clone();
    let c2 = rat(2) - big_c.clone();
    let f2 = hyp_series(&a2, &b2, &c2, prec);
    // R(t) = F₁/F₂ʰʸᵖ  (so ψ/κ = t^{1/a} R(t))
    let r = f1 * f2.inverse().expect("F₂(0)=1 invertible");
    // u(τ) = τ·R(τ^a): coefficient of τ^{a·n+1} is R_n.
    let mut ucoeffs = vec![rat(0); prec];
    for n in 0..prec {
        let idx = a as usize * n + 1;
        if idx < prec {
            ucoeffs[idx] = r.coeff(n).clone();
        }
    }
    let u_of_tau = PowerSeries::new(ucoeffs, prec);
    // revert to τ(u), then φ = τ(u)^a = t, a series in u = w/κ.
    let tau = revert(&u_of_tau);
    pow_series(&tau, a as usize)
}

/// R(t) = F₁/F₂ʰʸᵖ, i.e. the coefficients of ψ(t)/(κ t^{1/a}) — useful for validation
/// against the paper's intermediate `w/κ = t^{1/a}(1 + t/10 + …)` expansion.
pub fn psi_over_kappa_reduced(a: i64, b: i64, c: i64, prec: usize) -> PowerSeries<Rational> {
    let (big_a, big_b, big_c) = abc_params(a, b, c);
    let one = rat(1);
    let f1 = hyp_series(&big_a, &big_b, &big_c, prec);
    let a2 = one.clone() + big_a.clone() - big_c.clone();
    let b2 = one.clone() + big_b.clone() - big_c.clone();
    let c2 = rat(2) - big_c;
    let f2 = hyp_series(&a2, &b2, &c2, prec);
    f1 * f2.inverse().expect("F₂(0)=1 invertible")
}

// ---------------------------------------------------------------------------
// E3: the high-precision-FLOAT φ preamble
// ---------------------------------------------------------------------------
//
// Ported from the campaign's CERTIFIED driver
// `/home/john/inverse_galois/M23/route_a/belyi_shakedown/src/phi_hp.rs`
// (measured there: 31.5 s at len = 1501 vs > 13 h 47 m for the exact path,
// which was killed unfinished; max rel err vs exact 3.3e-75 at len = 201).
//
// The exact-rational [`phi_in_u`] has superquadratic bit-cost (coefficient
// heights grow linearly, GCD on every op). This path computes the SAME series
// in `rug::Float`:
//
//   F₁ = F(A,B,C;t), F₂ = F(1+A−C, 1+B−C, 2−C; t),  R(t) = F₁/F₂.
//   u(τ) = τ·R(τ^a)  ⇒  raising to the a-th power with t = τ^a = φ:
//       x := u^a = t·R(t)^a =: g(t).
//   So φ, as a series in u, is t(u^a) where t(x) is the REVERSION of the
//   single series g — computed to M = floor((len−1)/a)+1 t-terms instead of
//   reverting a length-len series (a = 2 halves the length).
//   Coefficient of u^{a·m} in φ_in_u is t_m; all other u-exponents vanish.
//
// CONDITIONING (measured driver-side, len=101 vs exact): the reversion
// identity g(t(x)) = x hides exponentially growing cancellation mass —
// computing at the output precision loses ~0.9 decimal digits PER ORDER
// (rel. err 1.8e-34 at t_50 from 77-digit arithmetic). The fix mirrors the
// exact path's "infinite internal precision, then round": all series
// arithmetic runs at
//   prec_internal = prec_out + 64 + ceil(BITS_PER_ORDER · M),
// (BITS_PER_ORDER default 4.0, env override PHI_HP_BITS_PER_ORDER) and the
// final coefficients are rounded to prec_out. An a-posteriori Newton bound
// (‖δt_j/t_j‖ from the final residual) gates the output: it must beat
// 2^-(prec_out+20) or the run aborts. The exact path is kept untouched for
// cross-validation (see the G1-mirror gate in the tests).

/// Exact small fraction num/den as an hp Float.
fn frac(prec: u32, num: i64, den: i64) -> Float {
    Float::with_val(prec, num) / den
}

/// F(A,B,C;t) to `len` terms via c_{n} = c_{n-1}·(A+n-1)(B+n-1)/((C+n-1)·n),
/// all in hp floats (mirrors [`hyp_series`]).
fn hyp_series_hp(af: &Float, bf: &Float, cf: &Float, prec: u32, len: usize) -> Vec<Float> {
    let mut out = Vec::with_capacity(len);
    let mut c = Float::with_val(prec, 1);
    out.push(c.clone());
    for n in 1..len {
        let k = (n - 1) as u32;
        let fa = Float::with_val(prec, af + k);
        let fb = Float::with_val(prec, bf + k);
        let fc = Float::with_val(prec, cf + k);
        c *= fa;
        c *= fb;
        c /= fc;
        c /= n as u32;
        out.push(c.clone());
    }
    out
}

/// Truncated product of two hp series to `len` terms (naive O(n²), fused
/// accumulate — no per-op allocation).
fn mul_trunc(a: &[Float], b: &[Float], len: usize, prec: u32) -> Vec<Float> {
    let mut out = vec![Float::with_val(prec, 0); len];
    for i in 0..a.len().min(len) {
        if a[i].is_zero() {
            continue;
        }
        let jmax = b.len().min(len - i);
        for j in 0..jmax {
            out[i + j] += &a[i] * &b[j];
        }
    }
    out
}

/// Reciprocal 1/s of a unit series (s[0] ≠ 0) to `len` terms:
/// b₀ = 1/s₀, b_n = −(Σ_{j=1..n} s_j b_{n−j})/s₀.
fn recip_trunc(s: &[Float], len: usize, prec: u32) -> Vec<Float> {
    assert!(!s.is_empty() && !s[0].is_zero(), "recip_trunc: constant term must be nonzero");
    let s0_inv = Float::with_val(prec, 1) / &s[0];
    let mut r = vec![Float::with_val(prec, 0); len];
    r[0] = s0_inv.clone();
    for n in 1..len {
        let mut acc = Float::with_val(prec, 0);
        let jmax = n.min(s.len().saturating_sub(1));
        for j in 1..=jmax {
            acc += &s[j] * &r[n - j];
        }
        acc *= &s0_inv;
        r[n] = -acc;
    }
    r
}

/// Formal derivative (length len−1).
fn deriv(s: &[Float], prec: u32) -> Vec<Float> {
    (1..s.len()).map(|m| Float::with_val(prec, &s[m] * (m as u32))).collect()
}

/// Series composition g(t(x)) truncated to `len` terms, Paterson–Stockmeyer:
/// g = Σ_j G_j(y)·y^{K·j} with K ≈ √deg(g) — (K−1) + deg/K ≈ 2√deg series
/// products instead of deg (matters at the large internal precision).
/// Requires t(0) = 0.
fn compose_trunc(g: &[Float], t: &[Float], len: usize, prec: u32) -> Vec<Float> {
    debug_assert!(t.is_empty() || t[0].is_zero(), "compose_trunc: t must have zero constant term");
    let mmax = g.len().min(len);
    let mut acc = vec![Float::with_val(prec, 0); len];
    if mmax == 0 {
        return acc;
    }
    if mmax == 1 {
        acc[0].assign(&g[0]);
        return acc;
    }
    let k = ((mmax as f64).sqrt().ceil() as usize).max(1);
    // powers t^1..t^K, each truncated to len
    let mut pows: Vec<Vec<Float>> = Vec::with_capacity(k + 1);
    pows.push(Vec::new()); // index 0 unused
    pows.push(t[..t.len().min(len)].to_vec());
    for i in 2..=k {
        let p = mul_trunc(&pows[i - 1], &pows[1], len, prec);
        pows.push(p);
    }
    let nblocks = mmax.div_ceil(k);
    for jb in (0..nblocks).rev() {
        if jb != nblocks - 1 {
            acc = mul_trunc(&acc, &pows[k], len, prec);
        }
        let base = jb * k;
        for i in 0..k {
            let gi = base + i;
            if gi >= mmax {
                break;
            }
            if g[gi].is_zero() {
                continue;
            }
            if i == 0 {
                acc[0] += &g[gi];
            } else {
                for (jj, pv) in pows[i].iter().enumerate() {
                    if jj >= len {
                        break;
                    }
                    acc[jj] += &g[gi] * pv;
                }
            }
        }
    }
    acc
}

/// Reversion of g (g[0]=0, g[1]≠0) by Newton iteration with order doubling.
/// Returns (t, out_rel_bound, step_residuals): g(t(x)) = x + O(x^len), where
/// out_rel_bound is the a-posteriori FIRST-ORDER OUTPUT ERROR — after the loop
/// one extra composition forms num = g∘t − x, δt = num/(g'∘t), and the bound is
/// max_j |δt_j| / max(|t_j|, 1), the relative accuracy of the returned
/// coefficients at the working precision. `step_residuals[i]` is the i-th
/// step's PRE-step relative residual (a recorded diagnostic measuring the
/// previous step's output — never gated here; see [`PhiHpU`]):
/// max over already-converged orders j ≤ order of |[x^j](g∘t − x)| / (|t_j|+1)
/// — relative to the local coefficient magnitude, because t_m grows
/// exponentially when the reversion radius is < 1 and absolute residuals
/// mislead there.
fn revert_newton(g: &[Float], len: usize, prec: u32) -> (Vec<Float>, Float, Vec<f64>) {
    assert!(len >= 2 && g.len() >= 2, "revert_newton: need at least 2 terms");
    assert!(g[0].is_zero(), "revert_newton: g(0) must be 0");
    assert!(!g[1].is_zero(), "revert_newton: g'(0) must be nonzero");
    let target_order = len - 1;
    let mut t = vec![Float::with_val(prec, 0), Float::with_val(prec, 1) / &g[1]];
    let mut order = 1usize;
    let mut step_residuals = Vec::new();
    let t_all = std::time::Instant::now();
    while order < target_order {
        let new_order = (2 * order + 1).min(target_order);
        let nlen = new_order + 1;
        t.resize(nlen, Float::with_val(prec, 0));
        let ts = std::time::Instant::now();
        let gt = compose_trunc(&g[..g.len().min(nlen)], &t, nlen, prec);
        // num = g∘t − x
        let mut num = gt.clone();
        num[1] -= 1u32;
        // validation of the PREVIOUS step: coefficients 0..=order must vanish
        // relative to the local coefficient magnitude
        let mut pre_resid = 0f64;
        for (j, coef) in num.iter().enumerate().take(order.min(nlen - 1) + 1) {
            let scale = Float::with_val(prec, &t[j].clone().abs() + 1u32);
            let v = Float::with_val(prec, &coef.clone().abs() / &scale).to_f64();
            if v > pre_resid {
                pre_resid = v;
            }
        }
        step_residuals.push(pre_resid);
        // g'∘t = (g∘t)' / t'  — one composition per step
        let dgt = deriv(&gt, prec);
        let dt = deriv(&t, prec);
        let inv_dt = recip_trunc(&dt, nlen - 1, prec);
        let gpt = mul_trunc(&dgt, &inv_dt, nlen - 1, prec);
        let inv_gpt = recip_trunc(&gpt, nlen - 1, prec);
        let corr = mul_trunc(&num, &inv_gpt, nlen, prec);
        for j in 0..nlen {
            t[j] -= &corr[j];
        }
        eprintln!(
            "[phi-hp] revert: order {order} -> {new_order} in {:.2}s (pre-step rel residual {:.3e}, cumulative {:.1}s)",
            ts.elapsed().as_secs_f64(),
            pre_resid,
            t_all.elapsed().as_secs_f64()
        );
        order = new_order;
    }
    // a-posteriori output bound: one extra composition, δt = (g∘t − x)/(g'∘t)
    let ts = std::time::Instant::now();
    let gt = compose_trunc(&g[..g.len().min(len)], &t, len, prec);
    let mut num = gt.clone();
    num[1] -= 1u32;
    let dgt = deriv(&gt, prec);
    let dt = deriv(&t, prec);
    let inv_dt = recip_trunc(&dt, len - 1, prec);
    let gpt = mul_trunc(&dgt, &inv_dt, len - 1, prec);
    let inv_gpt = recip_trunc(&gpt, len - 1, prec);
    let dt_vec = mul_trunc(&num, &inv_gpt, len, prec);
    // Float-native maxima: these can under/overflow f64 (e.g. 1e-677 at 3324
    // internal bits), so keep them as Floats and report log10.
    let mut bound = Float::with_val(prec, 0);
    let mut max_t = Float::with_val(prec, 0);
    for j in 0..len {
        let scale = Float::with_val(prec, &t[j].clone().abs() + 1u32);
        let v = Float::with_val(prec, &dt_vec[j].clone().abs() / &scale);
        if v > bound {
            bound = v;
        }
        let tv = t[j].clone().abs();
        if tv > max_t {
            max_t = tv;
        }
    }
    let log10 = |f: &Float| -> f64 {
        if f.is_zero() { f64::NEG_INFINITY } else { f.clone().abs().log10().to_f64() }
    };
    eprintln!(
        "[phi-hp] revert certificate: max |dt_j|/max(|t_j|,1) = 1e{:.1} over {} terms (max |t_m| = 1e{:.1}) ({:.2}s)",
        log10(&bound),
        len,
        log10(&max_t),
        ts.elapsed().as_secs_f64()
    );
    (t, bound, step_residuals)
}

/// Internal working precision for a reversion of M terms emitting at prec_out:
/// prec_out + 64 guard bits + BITS_PER_ORDER·M for the measured ~0.9-digit/order
/// conditioning loss. Override the rate with env PHI_HP_BITS_PER_ORDER.
fn internal_prec(prec_out: u32, mlen: usize) -> u32 {
    let rate: f64 = std::env::var("PHI_HP_BITS_PER_ORDER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4.0);
    prec_out + 64 + (rate * mlen as f64).ceil() as u32
}

/// The hp φ_in_u result with its accuracy certificate.
///
/// The certificate layer (ported faithfully from the campaign driver):
/// * `step_residuals` — DIAGNOSTICS ONLY: each Newton order-doubling step
///   re-evaluates g∘t − x over the PREVIOUSLY converged orders relative to the
///   local coefficient magnitude, and the values are recorded here and printed
///   to stderr. They are never compared against a threshold — a corrupted step
///   would show up in them but would NOT abort the run at that point;
/// * `rel_bound` — THE ONLY HARD GATE: after the final step, ONE extra
///   composition forms the first-order Newton correction δt = (g∘t − x)/(g'∘t);
///   the bound max_j |δt_j|/max(|t_j|,1) is an a-posteriori estimate of the
///   relative coefficient error, independent of how t was produced (so it would
///   catch corruption from any earlier step). [`phi_in_u_hp`] hard-gates it
///   against 2^-(prec_out+20) and panics on failure — it never returns
///   coefficients that are less accurate than the emitted precision.
///
/// NOTE (honesty): the bound is FIRST-ORDER (the standard Newton a-posteriori
/// estimate), not a rigorous interval enclosure; it certifies the residual of
/// the reversion identity at working precision, which is the same class of
/// certificate the exact path gets from exact arithmetic plus final rounding.
pub struct PhiHpU {
    /// coeffs[p] = [u^p] φ_in_u rounded to `prec_out`; exactly zero unless a | p.
    pub coeffs: Vec<Float>,
    /// t_m = [u^{a·m}] φ_in_u at FULL internal precision — for downstream
    /// κ-scaling without an intermediate rounding (the driver scales at
    /// internal precision and rounds once at the end).
    pub t_internal: Vec<Float>,
    /// The internal working precision actually used.
    pub prec_internal: u32,
    /// The output precision the certificate was gated against.
    pub prec_out: u32,
    /// a-posteriori first-order relative output bound (see type docs).
    pub rel_bound: Float,
    /// Per-Newton-step pre-step relative residuals (see type docs).
    pub step_residuals: Vec<f64>,
}

/// φ_in_u (the SAME series as [`phi_in_u`]) computed in hp floats via the
/// reduction u^a = t·R(t)^a — one Newton order-doubling reversion of a
/// half-length series — with per-step residual certificates and a hard
/// a-posteriori output gate (panics if the gate fails; see [`PhiHpU`]).
///
/// Cost: minutes-scale where the exact path is hours/days-scale (driver
/// measurements: 31.5 s at len = 1501 / 256-bit out, 496 s at len = 3001 /
/// 400-bit out, vs 68 min for exact at len = 201 and > 13 h 47 m killed
/// unfinished at len = 1501).
pub fn phi_in_u_hp(a: i64, b: i64, c: i64, prec_out: u32, len: usize) -> PhiHpU {
    let t0 = std::time::Instant::now();
    let au = a as usize;
    assert!(a >= 2 && len > au, "phi_in_u_hp: need a >= 2 and len > a");
    // t-terms needed: u-exponent a·m ≤ len−1
    let mlen = (len - 1) / au + 1;
    let prec = internal_prec(prec_out, mlen);
    eprintln!(
        "[phi-hp] config: len={len} mlen={mlen} prec_out={prec_out} prec_internal={prec} ({:.2} guard bits/order margin)",
        (prec - prec_out - 64) as f64 / mlen as f64
    );
    // Hypergeometric parameters, exact fractions (mirrors abc_params):
    //   A = (abc + bc − ac − ab)/(2abc), B = (abc + bc − ac + ab)/(2abc), C = (a+1)/a
    //   A₂ = 1+A−C = (abc − bc − ac − ab)/(2abc), B₂ = 1+B−C = (abc − bc − ac + ab)/(2abc),
    //   C₂ = 2−C = (a−1)/a
    let (abc, ab, ac, bc) = (a * b * c, a * b, a * c, b * c);
    let af = frac(prec, abc + bc - ac - ab, 2 * abc);
    let bf = frac(prec, abc + bc - ac + ab, 2 * abc);
    let cf = frac(prec, a + 1, a);
    let a2 = frac(prec, abc - bc - ac - ab, 2 * abc);
    let b2 = frac(prec, abc - bc - ac + ab, 2 * abc);
    let c2 = frac(prec, a - 1, a);
    let f1 = hyp_series_hp(&af, &bf, &cf, prec, mlen);
    let f2 = hyp_series_hp(&a2, &b2, &c2, prec, mlen);
    let r = mul_trunc(&f1, &recip_trunc(&f2, mlen, prec), mlen, prec);
    eprintln!(
        "[phi-hp] R(t) = F1/F2 done: {} t-terms at {} bits in {:.1}s",
        mlen,
        prec,
        t0.elapsed().as_secs_f64()
    );
    // g(t) = t·R(t)^a
    let mut ra = r.clone();
    for _ in 1..au {
        ra = mul_trunc(&ra, &r, mlen, prec);
    }
    let mut g = vec![Float::with_val(prec, 0); mlen];
    for m in 1..mlen {
        g[m].assign(&ra[m - 1]);
    }
    // t(x): reversion of g — φ_in_u coefficient of u^{a·m} is t_m
    let (t, bound, step_residuals) = revert_newton(&g, mlen, prec);
    // OUTPUT GATE: coefficients must be accurate beyond the emitted precision
    let gate = Float::with_val(prec, Float::i_exp(1, -((prec_out + 20) as i32)));
    let blog = if bound.is_zero() { f64::NEG_INFINITY } else { bound.clone().abs().log10().to_f64() };
    assert!(
        bound < gate,
        "[phi-hp] OUTPUT GATE FAILED: a-posteriori relative error 1e{blog:.1} >= 2^-(prec_out+20); \
         raise PHI_HP_BITS_PER_ORDER (default 4.0) and rerun"
    );
    eprintln!("[phi-hp] OUTPUT GATE: PASS (1e{:.1} < 2^-{})", blog, prec_out + 20);
    let mut coeffs = vec![Float::with_val(prec_out, 0); len];
    for (p, slot) in coeffs.iter_mut().enumerate() {
        if p % au == 0 {
            slot.assign(&t[p / au]);
        }
    }
    eprintln!("[phi-hp] total: {} u-terms in {:.1}s", len, t0.elapsed().as_secs_f64());
    PhiHpU {
        coeffs,
        t_internal: t,
        prec_internal: prec,
        prec_out,
        rel_bound: bound,
        step_residuals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abc_5_3_3() {
        let (a, b, c) = abc_params(5, 3, 3);
        assert_eq!(a, ratio(4, 15));
        assert_eq!(b, ratio(3, 5));
        assert_eq!(c, ratio(6, 5));
    }

    #[test]
    fn abc_2_12_5() {
        let (a, b, c) = abc_params(2, 12, 5);
        assert_eq!(a, ratio(73, 120));
        assert_eq!(b, ratio(97, 120));
        assert_eq!(c, ratio(3, 2));
    }

    // Paper (5,3,3): w/κ = t^{1/5}(1 + t/10 + 3943/89100 t² + 2161/81000 t³ + …).
    #[test]
    fn psi_reduced_5_3_3() {
        let r = psi_over_kappa_reduced(5, 3, 3, 6);
        assert_eq!(*r.coeff(0), rat(1));
        assert_eq!(*r.coeff(1), ratio(1, 10));
        assert_eq!(*r.coeff(2), ratio(3943, 89100));
        assert_eq!(*r.coeff(3), ratio(2161, 81000));
    }

    // Paper (5,3,3), eq near (5.10):
    //   φ(w) = (w/κ)^5 − 1/2 (w/κ)^10 + 637/3564 (w/κ)^15 − 383/7128 (w/κ)^20 + O(w^25).
    #[test]
    fn phi_5_3_3_matches_paper() {
        let phi = phi_in_u(5, 3, 3, 25);
        assert_eq!(*phi.coeff(5), rat(1));
        assert_eq!(*phi.coeff(10), ratio(-1, 2));
        assert_eq!(*phi.coeff(15), ratio(637, 3564));
        assert_eq!(*phi.coeff(20), ratio(-383, 7128));
        // exponents not divisible by a=5 vanish
        for k in [1, 2, 3, 4, 6, 7, 11, 13, 16, 19] {
            assert_eq!(*phi.coeff(k), rat(0), "phi coeff {k} should vanish");
        }
    }

    /// Exact `Rational` → hp Float via decimal strings (same route as
    /// `genus0_map::rat_to_float`).
    fn rat_to_float(r: &Rational, prec: u32) -> Float {
        let num = Float::with_val(prec, Float::parse(r.numerator().to_string()).unwrap());
        let den = Float::with_val(prec, Float::parse(r.denominator().to_string()).unwrap());
        Float::with_val(prec, num / den)
    }

    // The G1-mirror agreement gate (E3): the hp φ_in_u must match the exact
    // rational φ_in_u to ≤ 1e-40 RELATIVE per coefficient, with the zero
    // pattern (u-exponents not divisible by a) EXACT and the support (nonzero
    // positions) EXACT — mirroring the campaign's certified G1 gate, which
    // measured max rel err 3.3e-75 at len = 201 driver-side.
    //
    // GATE LENGTHS: derived from what actually runs in seconds ON THIS BOX —
    // the exact path at (2,12,5) scales like ~len^4.3 (measured release-mode:
    // 4.8 s at len 41, 11.5 s at 51, 23.4 s at 61, 47.7 s at 71, which
    // extrapolates to 69 min at len 201, matching the production 68-min
    // measurement) and runs ~6.6× slower again under the debug profile tests
    // use (len 51: 75 s measured debug; len 101 was still unfinished at
    // 17 min debug when killed). Hence gates at len 41 and 51 for (2,12,5)
    // (~30 s and ~75 s debug) — larger lengths do NOT run in seconds and are
    // unusable in the suite. The hp path costs milliseconds at these sizes.
    fn g1_gate(a: i64, b: i64, c: i64, len: usize) {
        let prec_out = 256u32;
        let t0 = std::time::Instant::now();
        let exact = phi_in_u(a, b, c, len);
        let t_exact = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        let hp = phi_in_u_hp(a, b, c, prec_out, len);
        let t_hp = t1.elapsed().as_secs_f64();
        eprintln!("[g1-gate] ({a},{b},{c}) len={len}: exact {t_exact:.2}s, hp {t_hp:.3}s");
        // certificate: the a-posteriori bound is far below the 1e-40 gate
        let gate = Float::with_val(hp.prec_internal, Float::parse("1e-40").unwrap());
        assert!(hp.rel_bound < gate, "certificate bound above the G1 gate");
        let cmp_prec = 2 * prec_out;
        let mut worst = Float::with_val(cmp_prec, 0);
        let mut worst_p = 0usize;
        for p in 0..len {
            let ex = exact.coeff(p);
            let ex_zero = *ex == rat(0);
            if p % (a as usize) != 0 {
                // zeros exact: both sides identically zero off the support lattice
                assert!(ex_zero, "exact coeff {p} not zero off lattice");
                assert!(hp.coeffs[p].is_zero(), "hp coeff {p} not exactly zero off lattice");
                continue;
            }
            // support exact: nonzero positions agree exactly
            assert_eq!(
                ex_zero,
                hp.coeffs[p].is_zero(),
                "support mismatch at p = {p}: exact zero = {ex_zero}, hp zero = {}",
                hp.coeffs[p].is_zero()
            );
            if ex_zero {
                continue;
            }
            let exf = rat_to_float(ex, cmp_prec);
            let diff = Float::with_val(cmp_prec, &hp.coeffs[p] - &exf).abs();
            let rel = Float::with_val(cmp_prec, diff / exf.abs());
            if rel > worst {
                worst = rel;
                worst_p = p;
            }
        }
        let wlog = if worst.is_zero() { f64::NEG_INFINITY } else { worst.clone().log10().to_f64() };
        eprintln!("[g1-gate] ({a},{b},{c}) len={len}: worst rel err 1e{wlog:.1} at p={worst_p}");
        let g1 = Float::with_val(cmp_prec, Float::parse("1e-40").unwrap());
        assert!(worst < g1, "G1 gate FAILED: worst rel err 1e{wlog:.1} at p={worst_p} (gate 1e-40)");
    }

    // Production triple (2,12,5), first gate length.
    #[test]
    fn g1_gate_2_12_5_len_41() {
        g1_gate(2, 12, 5, 41);
    }

    // Production triple (2,12,5), second gate length.
    #[test]
    fn g1_gate_2_12_5_len_51() {
        g1_gate(2, 12, 5, 51);
    }

    // Cross-validation triple (5,3,3): exercises the a = 5 branch of the
    // u^a = t·R(t)^a reduction (the driver was built for a = 2; the port is
    // generic and must stay correct for the (5,3,3) reference geometry).
    #[test]
    fn g1_gate_5_3_3_len_51() {
        g1_gate(5, 3, 3, 51);
    }

    // E3 acceptance probe: the hp φ at the campaign's len = 1501 shakedown
    // size, prec_out = 256. Driver-side measurement on this box: 31.5 s (the
    // exact path was killed unfinished after 13 h 47 m at this length).
    // #[ignore]d: run explicitly with
    //   cargo test -p rustmath-curves --lib -- --ignored phi_hp_timing_probe_len_1501 --nocapture
    #[test]
    #[ignore]
    fn phi_hp_timing_probe_len_1501() {
        let t0 = std::time::Instant::now();
        let hp = phi_in_u_hp(2, 12, 5, 256, 1501);
        let dt = t0.elapsed().as_secs_f64();
        let blog = hp.rel_bound.clone().log10().to_f64();
        eprintln!("[phi-hp-probe] len=1501 prec_out=256: {dt:.1}s wall, certificate bound 1e{blog:.1}");
        // the output gate inside phi_in_u_hp already panicked if the
        // certificate failed; re-assert here so the probe is self-reporting
        let gate = Float::with_val(hp.prec_internal, Float::i_exp(1, -276));
        assert!(hp.rel_bound < gate);
    }

    // Our target: φ for (2,12,5). Leading term is u^2; verify structure (rational,
    // even exponents only) and that it is computable to reasonable depth.
    #[test]
    fn phi_2_12_5_structure() {
        let phi = phi_in_u(2, 12, 5, 20);
        assert_eq!(*phi.coeff(0), rat(0));
        assert_eq!(*phi.coeff(2), rat(1)); // (w/κ)^a leading coefficient is 1
        // a = 2 ⇒ all odd-degree coefficients vanish
        for k in (1..20).step_by(2) {
            assert_eq!(*phi.coeff(k), rat(0), "odd phi coeff {k} must vanish");
        }
        // and at least one higher even coefficient is present (nontrivial series)
        assert_ne!(*phi.coeff(4), rat(0));
    }
}
