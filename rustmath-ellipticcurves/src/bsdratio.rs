//! # The BSD ratio L(E,1)/Ω_E and the analytic order of Ш
//!
//! ## What is unconditional and what is not (read this first)
//!
//! * The **numeric statements** produced here are unconditional: a
//!   certified value of L(E,1) (or L′(E,1)) with a rigorous tail bound and
//!   documented rounding allowance, a certified Ω_E, a certified canonical
//!   height, and the resulting closeness certificate
//!   |numeric − recognized| < bound. None of them certify an exact
//!   equality; they certify an explicit interval.
//! * The **rationality of L(E,1)/Ω_E** is itself a theorem (modular
//!   symbols + Manin–Drinfeld: L(f,1)/Ω is rational for the weight-2
//!   newform f attached to E by modularity), so recognizing a rational is
//!   mathematically meaningful, not numerology.
//! * The **denominator bound used for recognition** (denominator divides
//!   |E(Q)_tors|²) and the **interpretation** of the assembled quantity as
//!   the order of the Tate–Shafarevich group are exactly the Birch–
//!   Swinnerton-Dyer conjecture. Everything conditional is labeled
//!   "assuming BSD" ([`AnalyticShaAssumingBSD`]); the closeness
//!   certificates themselves assume nothing.
//!
//! ## The assemblies
//!
//! Rank 0 (ε = +1, L(1) certified nonzero):
//!
//! ```text
//! L(E,1)/Ω_E  =  (∏_p c_p) · |Ш| / |T|²      (BSD prediction)
//! ```
//!
//! so the ratio is recognized as the best rational with denominator
//! ≤ |T|² (scan of all denominators — equivalent to the continued-fraction
//! best approximation bounded by that denominator, since the certified
//! bound pins the value to an interval shorter than the gap 1/(n·|T|²)
//! between any two candidates; uniqueness is asserted, see
//! [`BSDRatio::certification_bound`]), and
//! Ш_an = (L/Ω) · |T|² / ∏c_p.
//!
//! Rank 1 (ε = −1 so L(1) = 0 exactly, L′(1) certified nonzero):
//!
//! ```text
//! Ш_an = L′(E,1) · |T|² / (Ω_E · Reg · ∏_p c_p),
//! ```
//!
//! where Reg = ĥ(P) is the rank-1 regulator of the supplied generator P of
//! E(Q)/tors ([`crate::height`], Sage/LMFDB normalization — the same
//! normalization whose product with our Ω reproduces the LMFDB BSD
//! invariants). **The caller asserts generatorhood**: this crate's descent
//! finds infinite-order witness points but does not saturate, so if P is
//! m·(generator) then ĥ(P) = m²·ĥ(generator) and the assembly evaluates to
//! (true Ш_an)/m² — typically a non-integer, caught by the integrality
//! check (it can pass silently only when m² divides the true Ш_an); the
//! provenance of P is recorded in the output. Ш_an is asserted to be certified-close to a
//! positive perfect-square integer (Cassels: |Ш| is a square when finite,
//! given that the pairing is alternating and Ш finite under BSD).
//!
//! ## Error accounting (documented model)
//!
//! Numeric legs carry: L-values — rigorous tail + documented rounding
//! allowance ([`LValue::error_budget`]); Ω — relative error ≤ 2^{−prec}
//! ([`crate::period`], derived bound); ĥ — absolute truncation error
//! ≤ 2^{−prec} plus rounding under generous guard bits
//! ([`crate::height`]). These are combined by first-order interval
//! propagation with an explicit ×2 headroom factor absorbing the
//! second-order terms and the O(2^{−wp}) division/multiplication roundings
//! of the combination itself (wp ≥ 128 everywhere here, while the combined
//! budgets are ~10^{−digits} with digits ≤ 40 — the second-order terms are
//! smaller than the first-order ones by factors ≥ 2^{60}, so the ×2 is
//! extremely conservative). The certified statement is:
//! |numeric − true| < bound, with `bound` returned in the output.

use crate::curve::{EllipticCurve, Point};
use crate::lfunction::{CurveLSeries, LValue};
use rustmath_core::ordering::OrderedRing;
use rustmath_integers::prime::factor;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use rustmath_reals::bigfloat::BigFloat;
use std::cmp::Ordering;

/// log2(10), rounded up a hair (sizes working precisions).
const LOG2_10: f64 = 3.321928094887363;

/// Fail-closed strict comparison: true ONLY when `a < b` holds; an
/// incomparable pair (impossible for finite BigFloats, but the type is
/// only partially ordered) counts as NOT certified.
fn certified_less(a: &BigFloat, b: &BigFloat) -> bool {
    matches!(a.partial_cmp(b), Some(Ordering::Less))
}

/// The certified rank-0 BSD ratio L(E,1)/Ω_E with its rational
/// recognition. The recognition target (denominator | |T|²) is the BSD
/// prediction; the closeness certificate is unconditional numerics.
#[derive(Debug, Clone)]
pub struct BSDRatio {
    /// Certified numeric L(E,1) (rigorous tail + documented rounding).
    pub l1: LValue,
    /// Ω_E of the global minimal model, relative error ≤ 2^{−prec}.
    pub omega: BigFloat,
    /// The numeric ratio l1.value / omega.
    pub ratio: BigFloat,
    /// The recognized rational (denominator ≤ |T|², in lowest terms).
    pub recognized: Rational,
    /// CERTIFIED: |ratio_true − recognized| < certification_bound, where
    /// ratio_true = L(E,1)/Ω_E (under the documented error model), AND
    /// 2·certification_bound·|T|⁴ < 1, so `recognized` is the UNIQUE
    /// rational with denominator ≤ |T|² within the bound.
    pub certification_bound: BigFloat,
    /// |E(Q)_tors| (exact, Mazur-classified).
    pub torsion_order: u32,
    /// ∏_p c_p over the bad primes (exact, Tate's algorithm).
    pub tamagawa_product: u32,
}

/// The analytic order of Ш, ASSUMING BSD (the name says so): the
/// unconditional content is the closeness certificate of the numeric
/// assembly to the returned integer; calling that integer "the order of
/// the Tate–Shafarevich group" is the BSD conjecture (plus, at rank 1,
/// the caller's assertion that the supplied point generates E(Q)/tors).
#[derive(Debug, Clone)]
pub struct AnalyticShaAssumingBSD {
    /// Ш_an as an exact positive integer.
    pub order: u32,
    /// Its integer square root (Cassels: the order is a perfect square;
    /// asserted during assembly).
    pub sqrt_order: u32,
    /// Which assembly produced it (0 or 1 = the analytic rank used).
    pub rank: u32,
    /// CERTIFIED closeness of the numeric assembly to `order` (exact
    /// rational arithmetic in the rank-0 path after recognition, so there
    /// the bound is the recognition bound scaled by |T|²/∏c_p).
    pub certification_bound: BigFloat,
    /// Provenance of every leg (L-value, Ω, regulator/generator, c_p, |T|).
    pub provenance: String,
}

/// ∏_p c_p over the bad primes (primes where the — possibly non-minimal —
/// model's discriminant loses its content are skipped via the conductor
/// exponent, exactly as in Tate's algorithm).
fn tamagawa_product(e: &EllipticCurve) -> u32 {
    let mut prod: u32 = 1;
    for (p, _) in factor(&e.discriminant.abs()) {
        let ld = e.local_data(&p);
        if ld.conductor_exponent > 0 {
            prod *= ld.tamagawa_number;
        }
    }
    prod
}

/// 2^{−k} at working precision wp.
fn pow2_neg(k: u64, wp: u64) -> BigFloat {
    BigFloat::one_prec(wp) / BigFloat::from_integer(&Integer::from(2).pow(k as u32), wp)
}

impl EllipticCurve {
    /// The rank-0 BSD ratio L(E,1)/Ω_E, recognized as a rational with
    /// denominator ≤ |T|² and certified as in [`BSDRatio`]. `digits`
    /// controls the L-value precision (Ω and the combination run with
    /// matching guard bits).
    ///
    /// Honest errors, never guesses:
    /// * ε = −1 (then L(1) = 0 exactly — this ratio is 0; use the rank-1
    ///   assembly [`EllipticCurve::analytic_sha_rank1`]);
    /// * L(1) not certified nonzero at this precision (possible analytic
    ///   rank ≥ 2 — numerics can never certify a zero);
    /// * the numeric ratio not within the certified bound of any rational
    ///   with denominator ≤ |T|², or the bound too coarse to pin it
    ///   uniquely (raise `digits`).
    ///
    /// # Panics
    ///
    /// Panics if the curve is singular.
    pub fn bsd_ratio_rank0(&self, digits: usize) -> Result<BSDRatio, String> {
        let ls = CurveLSeries::new(self)?;
        if ls.root_number() == -1 {
            return Err(
                "bsd_ratio_rank0: epsilon = -1, so L(E,1) = 0 exactly and the rank-0 \
                 ratio is 0; use analytic_sha_rank1 with a generator"
                    .to_string(),
            );
        }
        let lv = ls.l1(digits);
        if !lv.certified_nonzero() {
            return Err(format!(
                "bsd_ratio_rank0: L(E,1) = {} is not certified nonzero at {} digits \
                 (budget {}); the analytic rank may be >= 2 — refusing to recognize \
                 a rational from an uncertified value",
                lv.value.to_decimal_string(12),
                digits,
                lv.error_budget().to_decimal_string(6)
            ));
        }
        let wp = (digits as f64 * LOG2_10).ceil() as u64 + 64;
        let omega = self.real_period(wp);
        let omega_w = omega.with_precision(wp);
        let ratio = lv.value.clone().with_precision(wp) / omega_w.clone();

        // Error propagation (see the module docs): with L = value ± e_L and
        // Ω = omega·(1 ± 2^{−wp}),
        //   |L/Ω − value/omega| ≤ [e_L + (|value| + e_L)·2^{−wp+1}] / omega
        // up to second-order terms; ×2 headroom absorbs those and the
        // division rounding of `ratio` itself.
        let e_l = lv.error_budget().with_precision(wp);
        let abs_v = OrderedRing::abs(&lv.value).with_precision(wp);
        let two = BigFloat::from_integer(&Integer::from(2), wp);
        let bound =
            two.clone() * ((e_l.clone() + (abs_v + e_l) * pow2_neg(wp - 1, wp)) / omega_w.clone());

        let t = self.torsion_subgroup().order;
        let d_max = t * t; // denominator bound |T|^2 (BSD-predicted)

        // Best rational with denominator <= |T|^2: scan every denominator
        // (D <= 256 by Mazur, so this is exact and total — equivalent to
        // the bounded-denominator continued-fraction best approximation).
        let mut best: Option<(Rational, BigFloat)> = None;
        for den in 1..=d_max {
            let den_bf = BigFloat::from_integer(&Integer::from(den as i64), wp);
            let num = (ratio.clone() * den_bf.clone()).round_int();
            let cand =
                Rational::new(num, Integer::from(den as i64)).expect("denominator is nonzero");
            let diff = OrderedRing::abs(&(ratio.clone() - BigFloat::from_rational(&cand, wp)));
            if best.as_ref().is_none_or(|(_, d)| diff < *d) {
                best = Some((cand, diff));
            }
        }
        let (recognized, diff) = best.expect("denominator range is nonempty");

        // Certification: the true ratio lies within `bound` of `ratio`, and
        // `recognized` within `diff` of `ratio`.
        let total = diff.clone() + bound.clone();
        if !certified_less(&diff, &bound) {
            return Err(format!(
                "bsd_ratio_rank0: numeric ratio {} differs from the best rational {}/{} \
                 (denominator <= |T|^2 = {}) by {}, which EXCEEDS the certified error \
                 bound {} — refusing to recognize (raise digits, or the denominator \
                 hypothesis fails)",
                ratio.to_decimal_string(20),
                recognized.numerator(),
                recognized.denominator(),
                d_max,
                diff.to_decimal_string(6),
                bound.to_decimal_string(6)
            ));
        }
        // Uniqueness: two distinct rationals with denominators <= D differ
        // by >= 1/D^2, so 2·total·D^2 < 1 pins `recognized` uniquely.
        let d2 = BigFloat::from_integer(&Integer::from((d_max as i64) * (d_max as i64)), wp);
        if !certified_less(&(two * total.clone() * d2), &BigFloat::one_prec(wp)) {
            return Err(format!(
                "bsd_ratio_rank0: certified bound {} too coarse to pin a unique \
                 rational with denominator <= {} (raise digits)",
                total.to_decimal_string(6),
                d_max
            ));
        }

        Ok(BSDRatio {
            l1: lv,
            omega,
            ratio,
            recognized,
            certification_bound: total,
            torsion_order: t,
            tamagawa_product: tamagawa_product(self),
        })
    }

    /// The analytic order of Ш for an analytic-rank-0 curve, ASSUMING BSD
    /// (see [`AnalyticShaAssumingBSD`]): Ш_an = (L/Ω)·|T|²/∏c_p, evaluated
    /// in exact rational arithmetic on the recognized ratio, then asserted
    /// to be a positive perfect-square integer (Cassels). Honest `Err`
    /// when any leg fails (including a non-integer or non-square result,
    /// which would witness either a too-coarse recognition or a BSD
    /// violation).
    pub fn analytic_sha_rank0(&self, digits: usize) -> Result<AnalyticShaAssumingBSD, String> {
        let ratio = self.bsd_ratio_rank0(digits)?;
        let t2 = Integer::from((ratio.torsion_order as i64) * (ratio.torsion_order as i64));
        let cp = Integer::from(ratio.tamagawa_product as i64);
        let sha_q = &(&ratio.recognized * &Rational::from_integer(t2.clone()))
            / &Rational::from_integer(cp.clone());
        if !sha_q.denominator().is_one() {
            return Err(format!(
                "analytic_sha_rank0: (L/Omega)·|T|²/∏c_p = {}/{} is not an integer — \
                 recognition inconsistent with the BSD shape (or ∏c_p carries the \
                 denominator differently); refusing",
                sha_q.numerator(),
                sha_q.denominator()
            ));
        }
        let sha_int = sha_q.numerator().clone();
        if sha_int.signum() <= 0 {
            return Err(format!(
                "analytic_sha_rank0: Sha_an = {} is not positive — the L(1) sign or \
                 the recognition is wrong (bug or BSD violation)",
                sha_int
            ));
        }
        let root = sha_int.sqrt().expect("nonnegative");
        if &root * &root != sha_int {
            return Err(format!(
                "analytic_sha_rank0: Sha_an = {} is not a perfect square (Cassels \
                 requires a square order); recognition or hypotheses inconsistent",
                sha_int
            ));
        }
        // scale the ratio-side bound to the Sha side (exact integers scale)
        let wp = ratio.certification_bound.prec();
        let scale = BigFloat::from_integer(&t2, wp) / BigFloat::from_integer(&cp, wp);
        let bound = ratio.certification_bound.clone() * scale;
        let provenance = format!(
            "rank-0 assembly ASSUMING BSD: L(E,1) certified numerically (epsilon = +1, \
             {} digits; tail + rounding budget {}), Omega from the AGM on the global \
             minimal model, ratio recognized as {}/{} (denominator <= |T|^2 = {}, \
             unique within the certified bound), |T| = {} (exact torsion), \
             prod c_p = {} (Tate). Sha_an = ratio·|T|²/∏c_p in exact rationals.",
            digits,
            ratio.l1.error_budget().to_decimal_string(6),
            ratio.recognized.numerator(),
            ratio.recognized.denominator(),
            ratio.torsion_order as u64 * ratio.torsion_order as u64,
            ratio.torsion_order,
            ratio.tamagawa_product,
        );
        Ok(AnalyticShaAssumingBSD {
            order: sha_int.to_i64() as u32,
            sqrt_order: root.to_i64() as u32,
            rank: 0,
            certification_bound: bound,
            provenance,
        })
    }

    /// The analytic order of Ш for an analytic-rank-1 curve, ASSUMING BSD
    /// **and assuming `generator` generates E(Q)/tors** (this crate's
    /// descent produces infinite-order witnesses but does not saturate; if
    /// the point is m·(generator), the assembly evaluates to Ш_an/m² and
    /// generally fails the integrality check with an honest `Err` — the
    /// caller owns the generatorhood assertion, which is recorded in the
    /// provenance):
    ///
    /// ```text
    /// Ш_an = L'(E,1) · |T|² / (Ω_E · ĥ(P) · ∏c_p),
    /// ```
    ///
    /// asserted certified-close to a positive perfect-square integer.
    /// Honest `Err` when ε ≠ −1, L′(1) is not certified nonzero, the point
    /// is torsion / off the curve, or the certified interval around the
    /// assembly contains no (unique) positive square integer.
    pub fn analytic_sha_rank1(
        &self,
        generator: &Point,
        digits: usize,
    ) -> Result<AnalyticShaAssumingBSD, String> {
        let ls = CurveLSeries::new(self)?;
        if ls.root_number() != -1 {
            return Err(
                "analytic_sha_rank1: epsilon = +1 (L(1) need not vanish); use the \
                 rank-0 assembly"
                    .to_string(),
            );
        }
        let dv = ls.l1_derivative(digits)?;
        if !dv.certified_nonzero() {
            return Err(format!(
                "analytic_sha_rank1: L'(E,1) = {} not certified nonzero at {} digits \
                 (analytic rank may be >= 3); refusing",
                dv.value.to_decimal_string(12),
                digits
            ));
        }
        if !self.is_on_curve(generator) {
            return Err("analytic_sha_rank1: the supplied point is not on the curve".to_string());
        }
        if self.point_order(generator).is_some() {
            return Err(
                "analytic_sha_rank1: the supplied point is torsion — it generates \
                 nothing of infinite order; a rank-1 regulator needs a non-torsion \
                 generator"
                    .to_string(),
            );
        }

        let wp = (digits as f64 * LOG2_10).ceil() as u64 + 64;
        let omega = self.real_period(wp);
        let reg = self.canonical_height(generator, wp);
        assert!(
            reg > BigFloat::zero_prec(wp),
            "canonical height of a non-torsion point must be positive (bug)"
        );
        let t = self.torsion_subgroup().order;
        let cp = tamagawa_product(self);

        let t2_bf = BigFloat::from_integer(&Integer::from((t as i64) * (t as i64)), wp);
        let cp_bf = BigFloat::from_integer(&Integer::from(cp as i64), wp);
        let denom = omega.clone().with_precision(wp) * reg.clone().with_precision(wp) * cp_bf;
        let s = dv.value.clone().with_precision(wp) * t2_bf.clone() / denom.clone();

        // Relative-error combination (module docs): L' contributes
        // e_L/|L'|, Ω contributes 2^{−wp}, ĥ contributes 2^{−wp}/ĥ
        // (absolute-to-relative; ĥ ≥ the curve's height floor, and here
        // simply ĥ as computed, which is >> 2^{−wp}); ×2 headroom.
        let e_l = dv.error_budget().with_precision(wp);
        let abs_v = OrderedRing::abs(&dv.value).with_precision(wp);
        let two = BigFloat::from_integer(&Integer::from(2), wp);
        let rel =
            e_l / abs_v + pow2_neg(wp - 1, wp) + pow2_neg(wp, wp) / reg.clone().with_precision(wp);
        let bound = two * s.clone() * rel;

        let k = s.round_int();
        let k_bf = BigFloat::from_integer(&k, wp);
        let diff = OrderedRing::abs(&(s.clone() - k_bf));
        if !certified_less(&diff, &bound) {
            return Err(format!(
                "analytic_sha_rank1: assembly = {} differs from the nearest integer {} \
                 by {}, exceeding the certified bound {} — refusing (raise digits, \
                 verify the generator, or BSD fails here)",
                s.to_decimal_string(20),
                k,
                diff.to_decimal_string(6),
                bound.to_decimal_string(6)
            ));
        }
        // uniqueness of the integer: bound must be < 1/2
        if !certified_less(
            &bound,
            &BigFloat::from_rational(&Rational::new(1, 2).unwrap(), wp),
        ) {
            return Err(
                "analytic_sha_rank1: certified bound >= 1/2 cannot pin an integer \
                 (raise digits)"
                    .to_string(),
            );
        }
        if k.signum() <= 0 {
            return Err(format!(
                "analytic_sha_rank1: Sha_an = {} is not positive (bug or wrong inputs)",
                k
            ));
        }
        let root = k.sqrt().expect("nonnegative");
        if &root * &root != k {
            return Err(format!(
                "analytic_sha_rank1: Sha_an = {} is not a perfect square (Cassels); \
                 either the supplied point is not a full generator (an index-m point \
                 deflates the assembly to Sha_an/m²) or BSD fails here",
                k
            ));
        }
        let provenance = format!(
            "rank-1 assembly ASSUMING BSD and ASSUMING the supplied point P = ({}, {}) \
             generates E(Q)/tors (not saturated by this crate): L'(E,1) certified \
             numerically (epsilon = -1 exact, {} digits, budget {}), Omega from the \
             AGM (minimal model), Reg = hhat(P) = {} (Sage/LMFDB normalization, \
             rank-1 regulator), |T| = {} (exact), prod c_p = {} (Tate).",
            generator.x,
            generator.y,
            digits,
            dv.error_budget().to_decimal_string(6),
            reg.to_decimal_string(20),
            t,
            cp,
        );
        Ok(AnalyticShaAssumingBSD {
            order: k.to_i64() as u32,
            sqrt_order: root.to_i64() as u32,
            rank: 1,
            certification_bound: bound,
            provenance,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_core::analytic::RealField;

    fn curve(a: [i64; 5]) -> EllipticCurve {
        EllipticCurve::new(
            Integer::from(a[0]),
            Integer::from(a[1]),
            Integer::from(a[2]),
            Integer::from(a[3]),
            Integer::from(a[4]),
        )
    }

    /// THE RANK-0 BSD RATIO GATES. Every constant was derived
    /// independently BEFORE this test, twice: PARI/GP (ellL1, E.omega,
    /// elltors, elllocalred) and a from-scratch mpmath pipeline
    /// (point-counted a_n → L(1); AGM → Ω; both at 40+ digits), agreeing
    /// with each other to ~40 digits; |T| and c_p additionally re-derived
    /// here from the crate's own exact torsion/Tate machinery. Includes
    /// the wild-additive curves unblocked by the Kraus/Halberstadt tables
    /// (x³−x N=32, x³+1 N=36, 27a1, 20a) alongside the multiplicative
    /// gates demanded by the stage-2 spec.
    #[test]
    #[allow(clippy::type_complexity)] // flat gate table, clearest as-is
    fn test_bsd_ratio_rank0_gates() {
        // (label, model, |T|, prod c_p, ratio num/den)
        let gates: [(&str, [i64; 5], u32, u32, i64, i64); 11] = [
            ("11a1", [0, -1, 1, -10, -20], 5, 5, 1, 5),
            ("14a1", [1, 0, 1, 4, -6], 6, 6, 1, 6),
            ("15a1", [1, 1, 1, -10, -10], 8, 8, 1, 8),
            ("26a1", [1, 0, 1, -5, -8], 3, 3, 1, 3),
            ("26b1", [1, -1, 1, -3, 3], 7, 7, 1, 7),
            ("37b1", [0, 1, 1, -23, -50], 3, 3, 1, 3),
            ("49a1", [1, -1, 0, -2, -1], 2, 2, 1, 2),
            ("x3-x(32)", [0, 0, 0, -1, 0], 4, 2, 1, 8),
            ("x3+1(36)", [0, 0, 0, 0, 1], 6, 6, 1, 6),
            ("27a1", [0, 0, 1, 0, -7], 3, 3, 1, 3),
            ("20a", [0, 1, 0, -1, 0], 6, 3, 1, 12),
        ];
        for (label, a, t, cp, num, den) in &gates {
            let e = curve(*a);
            let r = e
                .bsd_ratio_rank0(30)
                .unwrap_or_else(|err| panic!("bsd_ratio_rank0({}) failed: {}", label, err));
            assert_eq!(r.torsion_order, *t, "{}: |T|", label);
            assert_eq!(r.tamagawa_product, *cp, "{}: prod c_p", label);
            assert_eq!(
                r.recognized,
                Rational::new(*num, *den).unwrap(),
                "{}: recognized L/Omega",
                label
            );
            // the certificate itself: recognized is inside the envelope
            // and the envelope pins it uniquely (both asserted inside; we
            // re-check the headline numbers here)
            let diff = OrderedRing::abs(
                &(r.ratio.clone() - BigFloat::from_rational(&r.recognized, r.ratio.prec())),
            );
            assert!(
                diff < r.certification_bound.clone(),
                "{}: certificate",
                label
            );
            assert!(r.omega > BigFloat::zero_prec(64), "{}: Omega > 0", label);
        }
    }

    /// The INDIVIDUAL Tamagawa numbers entering the gates, re-derived here
    /// from the crate's Tate machinery and pinned against the PARI
    /// `elllocalred` values obtained during gate derivation (two
    /// independent sources; nothing recalled): 11a1 c₁₁ = 5; 14a1 c₂ = 2,
    /// c₇ = 3; 26a1 c₂ = 1, c₁₃ = 3; 26b1 c₂ = 7, c₁₃ = 1; 37b1 c₃₇ = 3;
    /// 49a1 c₇ = 2; 20a c₂ = 3, c₅ = 1; 65a1 c₅ = c₁₃ = 1; 37a1 c₃₇ = 1.
    #[test]
    #[allow(clippy::type_complexity)] // flat gate table, clearest as-is
    fn test_gate_tamagawa_numbers_individually() {
        let cases: [([i64; 5], &[(i64, u32)]); 9] = [
            ([0, -1, 1, -10, -20], &[(11, 5)]),
            ([1, 0, 1, 4, -6], &[(2, 2), (7, 3)]),
            ([1, 0, 1, -5, -8], &[(2, 1), (13, 3)]),
            ([1, -1, 1, -3, 3], &[(2, 7), (13, 1)]),
            ([0, 1, 1, -23, -50], &[(37, 3)]),
            ([1, -1, 0, -2, -1], &[(7, 2)]),
            ([0, 1, 0, -1, 0], &[(2, 3), (5, 1)]),
            ([1, 0, 0, -1, 0], &[(5, 1), (13, 1)]),
            ([0, 0, 1, -1, 0], &[(37, 1)]),
        ];
        for (a, cps) in &cases {
            let e = curve(*a);
            for (p, c) in cps.iter() {
                assert_eq!(
                    e.local_data(&Integer::from(*p)).tamagawa_number,
                    *c,
                    "c_{} of {:?}",
                    p,
                    a
                );
            }
        }
    }

    /// Rank-0 analytic Sha = 1 across the gate battery (assuming BSD; the
    /// expected value 1 was PARI- and mpmath-derived for every curve).
    #[test]
    fn test_analytic_sha_rank0_gates() {
        for (label, a) in [
            ("11a1", [0i64, -1, 1, -10, -20]),
            ("14a1", [1, 0, 1, 4, -6]),
            ("15a1", [1, 1, 1, -10, -10]),
            ("26a1", [1, 0, 1, -5, -8]),
            ("26b1", [1, -1, 1, -3, 3]),
            ("37b1", [0, 1, 1, -23, -50]),
            ("49a1", [1, -1, 0, -2, -1]),
            ("x3-x(32)", [0, 0, 0, -1, 0]),
            ("x3+1(36)", [0, 0, 0, 0, 1]),
        ] {
            let e = curve(a);
            let sha = e
                .analytic_sha_rank0(30)
                .unwrap_or_else(|err| panic!("Sha({}) failed: {}", label, err));
            assert_eq!(sha.order, 1, "{}: Sha_an", label);
            assert_eq!(sha.sqrt_order, 1);
            assert_eq!(sha.rank, 0);
            assert!(sha.provenance.contains("ASSUMING BSD"));
        }
    }

    /// THE RANK-1 SHOWPIECE (37a1): L'(37a,1) / (Ω · ĥ((0,0))) is
    /// certified equal to 1 — this single assertion ties together the
    /// exact ε = −1 root number, the certified L′-series, the AGM period,
    /// the canonical height, exact torsion and Tate's algorithm. The
    /// generator (0,0) is the classical Mordell–Weil generator of 37a1
    /// (its Heegner point; PARI ellheegner returns exactly [0,0], and
    /// ĥ((0,0)) = 0.05111140823996884… matches the published LMFDB
    /// regulator, re-derived by exact-rational duplication in mpmath
    /// before this test). c_37 = 1, |T| = 1, expected Ш_an = 1 (PARI: 1 to
    /// 60 digits).
    #[test]
    fn test_analytic_sha_rank1_37a() {
        let e = curve([0, 0, 1, -1, 0]);
        let gen = Point::from_integers(0, 0);
        let sha = e.analytic_sha_rank1(&gen, 26).expect("37a1 assembly");
        assert_eq!(sha.order, 1, "Sha_an(37a1) = 1");
        assert_eq!(sha.rank, 1);
        assert!(sha.provenance.contains("ASSUMING BSD"));
        assert!(sha.provenance.contains("generates E(Q)/tors"));
        // the certified bound is genuinely tight (< 10^-15 at 26 digits)
        let tiny = BigFloat::from_decimal_str("0.000000000000001", 128).unwrap();
        assert!(sha.certification_bound < tiny, "bound is tight");
    }

    /// The 65a1 rank-1 analogue: generator (1, 0) (infinite order —
    /// ĥ = 0.3755140986612663218… — while the 2-torsion point of 65a1 is
    /// (0,0)); c_5 = c_13 = 1 (non-split I₁ at both), |T| = 2, expected
    /// Ш_an = 1 (PARI + mpmath duplication-limit derivation, both before
    /// this test).
    #[test]
    fn test_analytic_sha_rank1_65a() {
        let e = curve([1, 0, 0, -1, 0]);
        // sanity: derived local data (c_p re-derived from crate Tate here)
        assert_eq!(tamagawa_product(&e), 1, "65a1: prod c_p");
        assert_eq!(e.torsion_subgroup().order, 2, "65a1: |T|");
        let gen = Point::from_integers(1, 0);
        let sha = e.analytic_sha_rank1(&gen, 24).expect("65a1 assembly");
        assert_eq!(sha.order, 1, "Sha_an(65a1) = 1");
        assert_eq!(sha.sqrt_order, 1);
    }

    /// Non-generator input honesty: feeding 2·P instead of the generator P
    /// of 37a1 deflates the assembly to Ш_an/4 = 1/4, which fails the
    /// integrality check with an honest `Err` (the documented m² caveat);
    /// and a torsion point is refused outright.
    #[test]
    fn test_analytic_sha_rank1_non_generator_caveat() {
        let e = curve([0, 0, 1, -1, 0]);
        let gen = Point::from_integers(0, 0);
        let twop = e.double_point(&gen);
        let r = e.analytic_sha_rank1(&twop, 26);
        assert!(
            r.is_err(),
            "2P in place of P gives Sha_an/4 = 1/4: must be refused, got {:?}",
            r.map(|s| s.order)
        );
        // 65a1's 2-torsion point (0,0) is refused as a regulator input
        let e65 = curve([1, 0, 0, -1, 0]);
        let tors = Point::from_integers(0, 0);
        let r = e65.analytic_sha_rank1(&tors, 20);
        assert!(r.is_err() && r.unwrap_err().contains("torsion"));
    }

    /// Honest refusals of the wrong-rank assemblies: 37a1 (ε = −1) refuses
    /// the rank-0 path; 11a1 (ε = +1) refuses the rank-1 path; 389a1
    /// (true analytic rank 2, ε = +1 with L(1) ≈ 0) refuses BOTH — the
    /// rank-0 path because L(1) is not certified nonzero, never a
    /// fabricated recognition.
    #[test]
    fn test_bsd_ratio_honest_refusals() {
        let e37 = curve([0, 0, 1, -1, 0]);
        let r = e37.bsd_ratio_rank0(20);
        assert!(r.is_err() && r.unwrap_err().contains("epsilon = -1"));

        let e11 = curve([0, -1, 1, -10, -20]);
        let p = Point::from_integers(0, 0); // arbitrary; refused before use
        let r = e11.analytic_sha_rank1(&p, 20);
        assert!(r.is_err() && r.unwrap_err().contains("epsilon = +1"));

        let e389 = curve([0, 1, 1, -2, 0]);
        let r = e389.bsd_ratio_rank0(20);
        assert!(r.is_err(), "389a1 rank-0 ratio must refuse");
        assert!(r.unwrap_err().contains("not certified nonzero"));
    }

    /// Model invariance: the BSD ratio of a non-minimal (u = 2)-scaled
    /// model of 11a1 equals the minimal one — every leg (L, Ω, |T|, c_p)
    /// normalizes to the minimal model internally.
    #[test]
    fn test_bsd_ratio_model_invariance() {
        let e = curve([0, -4, 8, -160, -1280]); // 11a1 scaled by u = 2
        assert_eq!(e.compute_conductor(), Integer::from(11));
        let r = e.bsd_ratio_rank0(26).expect("scaled 11a1");
        assert_eq!(r.recognized, Rational::new(1, 5).unwrap());
        assert_eq!(r.torsion_order, 5);
        assert_eq!(r.tamagawa_product, 5);
    }

    /// The certified L(1)/Ω values also sit inside their envelopes as raw
    /// numerics against the independently derived 40-digit truths (PARI +
    /// mpmath agreed): spot-check 11a1's pieces to 25 digits.
    #[test]
    fn test_bsd_ratio_numeric_envelope_11a() {
        let e = curve([0, -1, 1, -10, -20]);
        let r = e.bsd_ratio_rank0(32).unwrap();
        let prec = RealField::precision(&r.ratio).max(256);
        let truth =
            BigFloat::from_decimal_str("0.2000000000000000000000000000000000000", prec).unwrap();
        let d = OrderedRing::abs(&(r.ratio.clone() - truth));
        let tol = BigFloat::from_decimal_str("0.0000000000000000000000000001", prec).unwrap();
        assert!(
            d < tol,
            "L/Omega of 11a1 to 28 digits, got {}",
            r.ratio.to_decimal_string(32)
        );
    }
}
