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
//! Rank r ≥ 2 ([`EllipticCurve::analytic_sha_rank_r`], the general case —
//! the rank-1 formula above is its r = 1 instance):
//!
//! ```text
//! Ш_an = ( L^(r)(E,1)/r! ) · |T|² / (Ω_E · Reg · ∏_p c_p),
//! ```
//!
//! with Reg the r × r Néron–Tate Gram determinant, taken from
//! [`EllipticCurve::regulator_checked`] so that the generators' independence
//! is a CERTIFIED interval statement (|det| > its own error bound) and not a
//! float compared against zero. Independence gives rank E(Q) ≥ r
//! unconditionally, and a certified-nonzero L^(r)(1)/r! gives ord ≤ r; BSD
//! (rank = ord) then forces rank = ord = r. What remains assumed, always, is
//! BSD itself and that the generators have index 1 in E(Q)/tors — p-saturation
//! ([`crate::mordellweil`]) bounds but never eliminates that gap.
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
use crate::mordellweil::MordellWeilSubgroup;
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
    /// The rank r the assembly used — 0 (rational recognition of L(1)/Ω), 1,
    /// or any r ≥ 2 via [`EllipticCurve::analytic_sha_rank_r`]. Under BSD
    /// (and with certified-independent generators) this is also the algebraic
    /// rank and the order of vanishing; see the provenance.
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

    /// The analytic order of Ш at **arbitrary rank r = `generators.len()`**,
    /// ASSUMING BSD and ASSUMING the generators generate E(Q)/tors:
    ///
    /// ```text
    /// Ш_an  =  ( L^(r)(E,1)/r! ) · |T|²  /  ( Ω_E · Reg · ∏_p c_p ).
    /// ```
    ///
    /// The legs: `l_derivative(r, digits)` ([`crate::ltaylor`], rigorous tail
    /// bound), `real_period` (AGM on the minimal model, all real components),
    /// [`EllipticCurve::regulator_checked`] (Gram determinant WITH an error
    /// bound — dependent generators are refused, never silently accepted),
    /// the exact torsion order and the exact Tamagawa product.
    ///
    /// # What r = the analytic rank rests on
    ///
    /// * The r generators are certified independent, so **rank E(Q) ≥ r**
    ///   unconditionally.
    /// * L^(r)(E,1)/r! is certified nonzero, so **ord_{s=1} L ≤ r**
    ///   unconditionally (given modularity, a theorem).
    /// * ASSUMING BSD (rank = ord) these force rank = ord = r. That is the
    ///   *only* place the rank enters; the function does not need — and never
    ///   claims — an independent certificate that L^(k)(1) = 0 for k < r.
    ///   (For 389a1 such a certificate does exist and is exercised in the
    ///   showpiece test; nothing here depends on it.)
    ///
    /// The root number is checked against the parity of r first (ε = (−1)^r),
    /// which is exact and rules out an r of the wrong parity outright.
    ///
    /// # Honest errors, never guesses
    ///
    /// ε of the wrong parity; L^(r)(1)/r! not certified nonzero; a generator
    /// off the curve or torsion; generators not certified independent; the
    /// certified interval around the assembly containing no unique positive
    /// integer; that integer not a perfect square.
    ///
    /// **Cassels' square check is a genuine (though not decisive) guard on
    /// unsaturated input**: if the supplied points generate only an index-m
    /// subgroup of E(Q)/tors then Reg is m² too large and the assembly comes
    /// out as Ш_an/m², which is generally not a positive square integer and is
    /// then refused. It can pass silently only when m² divides Ш_an and the
    /// quotient is still square — so a pass is evidence, not a proof, of
    /// saturation. Use [`crate::mordellweil::MordellWeilSubgroup::saturate`]
    /// and [`Self::analytic_sha_from_mordell_weil`] to record what was
    /// actually proved.
    pub fn analytic_sha_rank_r(
        &self,
        generators: &[Point],
        digits: usize,
    ) -> Result<AnalyticShaAssumingBSD, String> {
        let r = generators.len();
        if r == 0 {
            return self.analytic_sha_rank0(digits);
        }
        let ls = CurveLSeries::new(self)?;
        let eps_expected: i8 = if r.is_multiple_of(2) { 1 } else { -1 };
        if ls.root_number() != eps_expected {
            return Err(format!(
                "analytic_sha_rank_r: the exact root number is epsilon = {}, so the order of \
                 vanishing is {} — it cannot equal r = {}. Refusing (the generators, the \
                 curve, or r is wrong).",
                ls.root_number(),
                if ls.root_number() == 1 { "even" } else { "odd" },
                r
            ));
        }
        for g in generators {
            if !self.is_on_curve(g) {
                return Err(format!(
                    "analytic_sha_rank_r: the supplied point ({}, {}) is not on the curve",
                    g.x, g.y
                ));
            }
            if self.point_order(g).is_some() {
                return Err(
                    "analytic_sha_rank_r: a supplied generator is torsion — it contributes \
                     nothing to the regulator (which would then be singular)"
                        .to_string(),
                );
            }
        }

        let lv = ls.l_derivative(r as u32, digits)?;
        if !lv.certified_nonzero() {
            return Err(format!(
                "analytic_sha_rank_r: L^({})(E,1)/{}! = {} is NOT certified nonzero at {} \
                 digits (budget {}) — the analytic rank may exceed {}, and numerics can never \
                 certify a zero. Refusing.",
                r,
                r,
                lv.value.to_decimal_string(12),
                digits,
                lv.error_budget().to_decimal_string(6),
                r
            ));
        }

        let wp = (digits as f64 * LOG2_10).ceil() as u64 + 64;
        let reg = self.regulator_checked(generators, wp)?;
        let omega = self.real_period(wp);
        let t = self.torsion_subgroup().order;
        let cp = tamagawa_product(self);

        let t2_bf = BigFloat::from_integer(&Integer::from((t as i64) * (t as i64)), wp);
        let cp_bf = BigFloat::from_integer(&Integer::from(cp as i64), wp);
        let reg_w = reg.value.clone().with_precision(wp);
        let omega_w = omega.clone().with_precision(wp);
        let denom = omega_w * reg_w.clone() * cp_bf;
        let s = lv.value.clone().with_precision(wp) * t2_bf / denom;

        // Relative-error combination (module docs): L^(r) contributes
        // e_L/|L^(r)|, Omega contributes 2^{-wp}, Reg contributes
        // err_Reg/|Reg| (both certified intervals); a x2 headroom absorbs the
        // second-order terms and the roundings of the combination itself.
        let e_l = lv.error_budget().with_precision(wp);
        let abs_v = OrderedRing::abs(&lv.value).with_precision(wp);
        let two = BigFloat::from_integer(&Integer::from(2), wp);
        let rel = e_l / abs_v
            + pow2_neg(wp - 1, wp)
            + reg.error_bound.clone().with_precision(wp) / OrderedRing::abs(&reg_w);
        let bound = two * OrderedRing::abs(&s) * rel;

        let k = s.round_int();
        let k_bf = BigFloat::from_integer(&k, wp);
        let diff = OrderedRing::abs(&(s.clone() - k_bf));
        if !certified_less(&diff, &bound) {
            return Err(format!(
                "analytic_sha_rank_r: the rank-{} assembly = {} differs from the nearest \
                 integer {} by {}, EXCEEDING the certified bound {} — refusing (raise digits, \
                 or the generators do not generate E(Q)/tors, or BSD fails here)",
                r,
                s.to_decimal_string(20),
                k,
                diff.to_decimal_string(6),
                bound.to_decimal_string(6)
            ));
        }
        if !certified_less(
            &bound,
            &BigFloat::from_rational(&Rational::new(1, 2).unwrap(), wp),
        ) {
            return Err(format!(
                "analytic_sha_rank_r: certified bound {} >= 1/2 cannot pin an integer \
                 (raise digits)",
                bound.to_decimal_string(6)
            ));
        }
        if k.signum() <= 0 {
            return Err(format!(
                "analytic_sha_rank_r: Sha_an = {} is not positive (bug or wrong inputs)",
                k
            ));
        }
        let root = k.sqrt().expect("nonnegative");
        if &root * &root != k {
            return Err(format!(
                "analytic_sha_rank_r: Sha_an = {} is NOT a perfect square (Cassels: |Sha| is \
                 a square when finite). Either the supplied points generate only an index-m \
                 subgroup of E(Q)/tors — which inflates Reg by m² and deflates this assembly \
                 to Sha_an/m² — or BSD fails here. Refusing.",
                k
            ));
        }

        let provenance = format!(
            "rank-{} assembly, ASSUMING BSD *and* ASSUMING the {} supplied points generate \
             E(Q)/tors (index 1 — NOT proved here; this crate computes no height-difference \
             bound and no index bound). Sha_an = L^({})(E,1)/{}! · |T|² / (Omega · Reg · \
             prod c_p). Legs: L^({})(E,1)/{}! = {} certified nonzero ({} digits, tail + \
             rounding budget {}; exact root number epsilon = {} has the right parity); \
             Omega = {} (AGM, global minimal model, all real components); Reg = {} \
             (Neron-Tate Gram determinant, CERTIFIED nonsingular: |det| > error bound {}, so \
             the points are independent and rank E(Q) >= {} unconditionally); |T| = {} \
             (exact); prod c_p = {} (Tate). Under BSD, rank = ord <= {} (L^({}) != 0) and \
             rank >= {} (independence), hence rank = ord = {}. Cassels' square check passed \
             (a guard against unsaturated input, not a proof of saturation).",
            r,
            r,
            r,
            r,
            r,
            r,
            lv.value.to_decimal_string(20),
            digits,
            lv.error_budget().to_decimal_string(6),
            ls.root_number(),
            omega.to_decimal_string(20),
            reg.value.to_decimal_string(20),
            reg.error_bound.to_decimal_string(6),
            r,
            t,
            cp,
            r,
            r,
            r,
            r,
        );
        Ok(AnalyticShaAssumingBSD {
            order: k.to_i64() as u32,
            sqrt_order: root.to_i64() as u32,
            rank: r as u32,
            certification_bound: bound,
            provenance,
        })
    }

    /// [`Self::analytic_sha_rank_r`] driven by a [`MordellWeilSubgroup`], so
    /// that whatever saturation was actually proved lands in the provenance.
    ///
    /// This still ASSUMES BSD and still ASSUMES index 1: `saturated_up_to =
    /// Some(B)` proves only p-saturation for p ≤ B, never a basis.
    pub fn analytic_sha_from_mordell_weil(
        &self,
        mw: &MordellWeilSubgroup,
        digits: usize,
    ) -> Result<AnalyticShaAssumingBSD, String> {
        let mut sha = self.analytic_sha_rank_r(&mw.generators, digits)?;
        sha.provenance = format!(
            "{} GENERATORS: {} SATURATION: {}",
            sha.provenance,
            mw.provenance,
            match mw.saturated_up_to {
                Some(b) => format!(
                    "p-saturated for every prime p <= {} (proved, via division polynomials). \
                     Primes above {} are UNTESTED, so index 1 remains an ASSUMPTION.",
                    b, b
                ),
                None => "NONE proved — index 1 is a bare assumption.".to_string(),
            }
        );
        Ok(sha)
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

    // -----------------------------------------------------------------------
    // Rank >= 2
    // -----------------------------------------------------------------------

    fn close_to(a: &BigFloat, decimal: &str, k: usize) -> bool {
        let prec = RealField::precision(a).max(256);
        let b = BigFloat::from_decimal_str(decimal, prec).unwrap();
        let tol_str = format!("0.{}1", "0".repeat(k - 1));
        let tol = BigFloat::from_decimal_str(&tol_str, prec).unwrap();
        OrderedRing::abs(&(a.clone() - b)) < tol
    }

    /// The rank-r entry point reproduces the rank-1 one where they overlap:
    /// 37a1 with its generator, through `analytic_sha_rank_r(&[P], ·)`, must
    /// give the same Ш_an = 1 as the (unchanged) `analytic_sha_rank1`.
    #[test]
    fn test_rank_r_agrees_with_rank1_on_37a() {
        let e = curve([0, 0, 1, -1, 0]);
        let gen = Point::from_integers(0, 0);
        let a = e.analytic_sha_rank1(&gen, 26).unwrap();
        let b = e.analytic_sha_rank_r(&[gen], 26).unwrap();
        assert_eq!((a.order, a.rank), (1, 1));
        assert_eq!((b.order, b.rank), (1, 1));
        assert!(b.provenance.contains("ASSUMING BSD"));
        // r = 0 delegates to the rank-0 assembly
        let e11 = curve([0, -1, 1, -10, -20]);
        let z = e11.analytic_sha_rank_r(&[], 26).unwrap();
        assert_eq!((z.order, z.rank), (1, 0));
    }

    /// *** THE RANK-2 SHOWPIECE — 389a1 ***
    ///
    /// Generators FOUND by naive search (not seeded), certified independent,
    /// p-saturated for p ≤ 5, then run through the rank-2 BSD assembly:
    ///
    /// ```text
    /// L''(389a,1)/2! · |T|² / (Omega · Reg · prod c_p)  ==  1
    /// ```
    ///
    /// Every constant PARI-derived BEFORE the assertions (realprecision 40):
    ///   ellglobalred → N = 389, prod c_p = 1 ; elltors → |T| = 1
    ///   2·E.omega[1] = 4.98042512171011015064271558388460492  (Delta > 0 ⇒
    ///     TWO real components; the crate's `real_period` integrates over all
    ///     of E(R), which is the Omega that BSD wants)
    ///   matdet(ellheightmatrix) = 0.152460177943143751624324757049455823
    ///   lfun(E,1,2)/2!          = 0.759316500288426770230192607894722019
    ///   and PARI's own BSD quotient of these is 1.000000000000000000000000000000000000000.
    ///
    /// This one test exercises: naive point search, canonical heights, the
    /// certified regulator, division polynomials + p-saturation, exact Tate
    /// local data (c_p), the exact torsion, the exact root number, the AGM
    /// period and the order-2 Taylor coefficient of L.
    #[test]
    fn test_389a_rank_two_bsd_showpiece() {
        use crate::descent::TwoDescent;
        use crate::mordellweil::MordellWeilSubgroup;

        let e = curve([0, 1, 1, -2, 0]);
        assert_eq!(e.compute_conductor(), Integer::from(389));
        assert_eq!(tamagawa_product(&e), 1, "389a: prod c_p");
        assert_eq!(e.torsion_subgroup().order, 1, "389a: |T|");
        assert_eq!(crate::global_root_number(&e), Ok(1), "389a: epsilon = +1");

        // 1. FIND the generators (no seeding).
        let mut affine: Vec<Point> = TwoDescent::new(&e)
            .find_rational_points(4)
            .into_iter()
            .filter(|p| !p.infinity)
            .collect();
        affine.sort_by(|a, b| {
            EllipticCurve::naive_height(a, 64)
                .partial_cmp(&EllipticCurve::naive_height(b, 64))
                .unwrap()
        });
        let mut gens: Vec<Point> = Vec::new();
        for p in affine {
            let mut trial = gens.clone();
            trial.push(p.clone());
            if e.regulator_checked(&trial, 96).is_ok() {
                gens.push(p);
            }
            if gens.len() == 2 {
                break;
            }
        }
        assert_eq!(gens.len(), 2, "the search must find 2 independent points");

        // 2. SATURATE for every prime p <= 5 (the run is asserted below).
        let mut mw = MordellWeilSubgroup::new(
            &e,
            &gens,
            160,
            "found by naive x = m/n search over |m|,n <= 4 (descent::find_rational_points)",
        )
        .unwrap();
        let rep = mw.saturate(&e, 5, 160).expect("389a saturation");
        assert_eq!(rep.primes_tested, vec![2, 3, 5]);
        assert_eq!(rep.classes_tested, 13, "the saturation actually ran");
        assert!(
            rep.steps.is_empty(),
            "the found pair is already 2,3,5-saturated"
        );
        assert_eq!(mw.saturated_up_to, Some(5));

        let reg = mw.regulator(&e, 160).unwrap();
        assert!(
            close_to(&reg.value, "0.152460177943143751624324757049455823", 20),
            "Reg(389a) = {}",
            reg.value.to_decimal_string(28)
        );
        assert!(
            close_to(
                &e.real_period(256),
                "4.98042512171011015064271558388460492",
                20
            ),
            "Omega(389a) = {}",
            e.real_period(256).to_decimal_string(28)
        );

        // 3. The analytic rank is CERTIFIED EXACTLY 2 (exact + certified, no
        //    conjecture): epsilon = +1 is exact, so ord is EVEN; L(389a,1) = 0
        //    is EXACT via the Manin-Birch winding element in rustmath-modular
        //    (rational linear algebra, not a small number), so ord >= 1 and
        //    hence ord >= 2 by parity; and L''(1)/2! is certified nonzero, so
        //    ord <= 2.  ==> ord = 2, assuming only modularity (a theorem).
        {
            use crate::lfunction::{AnalyticRank, LFunction};
            use rustmath_modular::modsym::decomposition::{HeckeEigenvalue, SummandHeckeAction};
            use rustmath_modular::modsym::ModularSymbolsGamma0;

            let ms = ModularSymbolsGamma0::new(389);
            let dec = ms.cuspidal_hecke_decomposition().unwrap();
            let want =
                |p: i64| Rational::from_integer(Integer::from(e.compute_a_p(&Integer::from(p))));
            let mut vanishes = None;
            for w in dec.summands() {
                let hit = [2i64, 3, 5].iter().all(|&p| {
                    matches!(
                        ms.hecke_action_on_summand(w, p as u64),
                        Ok(SummandHeckeAction::Eigenvalue(HeckeEigenvalue::Rational(ref a)))
                            if *a == want(p)
                    )
                });
                if hit {
                    assert!(vanishes.is_none(), "389a's summand must be unique");
                    vanishes = Some(ms.l1_vanishes(w).expect("winding projection"));
                }
            }
            assert!(
                vanishes.expect("the 389a newform appears in the decomposition"),
                "Manin-Birch: the winding projection of the 389a newform must vanish"
            );
            let lf = LFunction::new(e.clone());
            let ar = lf.analytic_rank_with_exact_vanishing(
                22,
                1,
                "Manin-Birch winding element: pi_f(e) = 0 in the level-389 modular symbol \
                 space (exact rational linear algebra, rustmath-modular)",
            );
            assert!(matches!(
                ar,
                AnalyticRank::RankCertifiedModuloRounding { rank: 2, .. }
            ));
            assert_eq!(ar.certified_value(), Some(2));
            // and WITHOUT the exact certificate nothing is certified
            assert_eq!(lf.analytic_rank(22).certified_value(), None);
        }

        // 4. THE BSD RATIO. L''(1)/2! · |T|² / (Omega · Reg · prod c_p) = 1.
        let sha = e
            .analytic_sha_from_mordell_weil(&mw, 26)
            .expect("389a rank-2 assembly");
        assert_eq!(sha.order, 1, "Sha_an(389a) = 1");
        assert_eq!(sha.sqrt_order, 1);
        assert_eq!(sha.rank, 2);
        let tiny = BigFloat::from_decimal_str("0.000000000000001", 256).unwrap();
        assert!(
            sha.certification_bound < tiny,
            "certification bound must be < 1e-15 at 26 digits, got {}",
            sha.certification_bound.to_decimal_string(8)
        );
        assert!(sha.provenance.contains("ASSUMING BSD"));
        assert!(sha.provenance.contains("generate E(Q)/tors"));
        assert!(sha
            .provenance
            .contains("p-saturated for every prime p <= 5"));
        assert!(sha.provenance.contains("index 1 remains an ASSUMPTION"));
        // the leading coefficient itself, against PARI's lfun(E,1,2)/2!
        let l2 = CurveLSeries::new(&e).unwrap().l_derivative(2, 26).unwrap();
        assert!(l2.certified_nonzero());
        assert!(
            close_to(&l2.value, "0.759316500288426770230192607894722019078", 20),
            "L''(389a,1)/2! = {}",
            l2.value.to_decimal_string(28)
        );
    }

    /// *** RANK 3 — 5077a1, and an HONEST label on it ***
    ///
    /// The BSD assembly at r = 3 gives Ш_an = 1. But the analytic rank of
    /// 5077a1 is **not certifiable** by anything in this workspace: ε = −1
    /// hands us L(1) = 0 for free, and the next coefficient that must vanish,
    /// L'(1), has the RIGHT parity — parity says nothing about it, the winding
    /// element certifies only L(1), and numerics can never certify a zero.
    /// So this Ш_an is conditional on rank = ord = 3, which here follows from
    /// BSD + the certified independence of the three generators (rank ≥ 3) +
    /// L'''(1)/3! ≠ 0 (ord ≤ 3) — exactly what the provenance says, and no
    /// more. `test_5077a_rank_three_is_NOT_certifiable` in `lfunction` records
    /// the unconditional non-result.
    ///
    /// The generators are SEEDED (a naive search to the height needed here is
    /// far too slow for a unit test) and then VERIFIED: on the curve, of
    /// infinite order, certified independent, and p-saturated for p ≤ 5 (51
    /// projective classes decided exactly). Note the *dependent* triple
    /// (−1,3), (0,2), (2,0) — (−1,3)+(0,2)+(2,0) = O — which `regulator_checked`
    /// catches (see the mordellweil tests). The true basis (PARI
    /// `ellgenerators`) is (1,0), (2,0), (0,2).
    ///
    /// PARI (realprecision 40), all derived BEFORE the assertions:
    ///   N = 5077, prod c_p = 1, |T| = 1
    ///   2·E.omega[1] = 4.15168798308693304988417568350728630
    ///   matdet(ellheightmatrix) = 0.417143558758383969817119544618093397
    ///   lfun(E,1,3)/3! = 1.73184990011930068979197508506015284
    ///   PARI's own BSD quotient of these = 1.000000000000000000000000000000000000000
    #[test]
    fn test_5077a_rank_three_bsd_conditional() {
        use crate::mordellweil::MordellWeilSubgroup;

        let e = curve([0, 0, 1, -7, 6]);
        assert_eq!(e.compute_conductor(), Integer::from(5077));
        assert_eq!(tamagawa_product(&e), 1);
        assert_eq!(e.torsion_subgroup().order, 1);
        assert_eq!(crate::global_root_number(&e), Ok(-1), "5077a: epsilon = -1");

        let gens = [
            Point::from_integers(1, 0),
            Point::from_integers(2, 0),
            Point::from_integers(0, 2),
        ];
        for g in &gens {
            assert!(
                e.is_on_curve(g),
                "seeded point {:?} must be on the curve",
                g
            );
            assert!(
                e.point_order(g).is_none(),
                "seeded point must be non-torsion"
            );
        }

        let mut mw = MordellWeilSubgroup::new(
            &e,
            &gens,
            160,
            "SEEDED from PARI ellgenerators (a naive search to this height is too slow for a \
             unit test); each point re-verified here: on the curve, infinite order, and the \
             triple CERTIFIED independent",
        )
        .expect("(1,0), (2,0), (0,2) are independent");
        let rep = mw.saturate(&e, 5, 160).expect("5077a saturation");
        assert_eq!(rep.primes_tested, vec![2, 3, 5]);
        assert_eq!(
            rep.classes_tested, 51,
            "(2³−1)/1 + (3³−1)/2 + (5³−1)/4 = 7 + 13 + 31 projective classes"
        );
        assert!(
            rep.steps.is_empty(),
            "the seeded triple is already 2,3,5-saturated"
        );
        let reg = mw.regulator(&e, 160).unwrap();
        assert!(
            close_to(&reg.value, "0.417143558758383969817119544618093397", 20),
            "Reg(5077a) = {}",
            reg.value.to_decimal_string(28)
        );
        assert!(
            close_to(
                &e.real_period(256),
                "4.15168798308693304988417568350728630",
                20
            ),
            "Omega(5077a) = {}",
            e.real_period(256).to_decimal_string(28)
        );

        let sha = e
            .analytic_sha_from_mordell_weil(&mw, 26)
            .expect("5077a rank-3 assembly");
        assert_eq!(sha.order, 1, "Sha_an(5077a) = 1 (CONDITIONAL on rank 3)");
        assert_eq!(sha.rank, 3);
        let tiny = BigFloat::from_decimal_str("0.000000000000001", 256).unwrap();
        assert!(
            sha.certification_bound < tiny,
            "bound {}",
            sha.certification_bound.to_decimal_string(8)
        );
        assert!(sha.provenance.contains("ASSUMING BSD"));
        assert!(sha.provenance.contains("NOT proved here"));
        assert!(sha
            .provenance
            .contains("p-saturated for every prime p <= 5"));
        assert!(sha.provenance.contains("index 1 remains an ASSUMPTION"));
        // The rank is NOT certified: the L-machinery refuses 5077a outright.
        assert_eq!(
            crate::lfunction::LFunction::new(e)
                .analytic_rank(20)
                .certified_value(),
            None,
            "5077a's analytic rank must stay UNRESOLVED — this Sha_an is conditional on rank 3"
        );
    }

    /// Rank-r honest refusals: wrong parity, dependent generators, and an
    /// index-m (unsaturated) input caught by Cassels' square condition.
    #[test]
    fn test_analytic_sha_rank_r_refusals() {
        // 389a with THREE points (epsilon = +1 has even parity, so r = 3 is
        // impossible) — refused on parity alone, before any numerics.
        let e389 = curve([0, 1, 1, -2, 0]);
        let r = e389.analytic_sha_rank_r(
            &[
                Point::from_integers(0, 0),
                Point::from_integers(-1, 1),
                Point::from_integers(1, 0),
            ],
            20,
        );
        assert!(r.is_err() && r.unwrap_err().contains("epsilon"));

        // 5077a fed the DEPENDENT triple: refused by regulator_checked.
        let e5077 = curve([0, 0, 1, -7, 6]);
        let r = e5077.analytic_sha_rank_r(
            &[
                Point::from_integers(-1, 3),
                Point::from_integers(0, 2),
                Point::from_integers(2, 0),
            ],
            20,
        );
        assert!(
            r.is_err() && r.clone().unwrap_err().contains("NOT certified nonzero"),
            "dependent triple must be refused: {:?}",
            r.map(|s| s.order)
        );

        // 389a fed {2·P1, P2}: an index-2 subgroup. Reg is 4x too big, so the
        // assembly lands on 1/4 — not a positive square integer, so Cassels
        // refuses it. This is the guard, and it is honest about being only a
        // guard (a pass would not prove saturation).
        let p1 = Point::from_integers(0, 0);
        let p2 = Point::from_integers(-1, 1);
        let r = e389.analytic_sha_rank_r(&[e389.double_point(&p1), p2], 26);
        assert!(
            r.is_err(),
            "an index-2 subgroup deflates Sha_an to 1/4 and must be refused, got {:?}",
            r.map(|s| s.order)
        );
    }
}
