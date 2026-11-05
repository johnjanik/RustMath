//! The hp → exact connector: snap a converged [`NewtonHpReport`] to certified
//! exact data ([`ExactifyOutcome`]).
//!
//! This is the previously missing consumer of
//! [`NewtonHpReport::solution_decimals`] (documented there as "for LLL/PSLQ
//! recognition"). Pipeline position:
//!
//! ```text
//! refine_hp (newton_hp)  →  snap_hp_solution  →  ExactifyOutcome
//!                                                    → bridge::SolvedCover::from_exactify
//! ```
//!
//! # The gauge ([`Gauge2_12_5`]) — derivation
//!
//! `refine_hp` packs the `[2,12,5]` unknowns as 56 reals: `a`-roots `0..16`,
//! `b`-roots `16..32`, `r`-roots `32..40`, `s`-roots `40..48`, `u`-roots
//! `48..52`, `λ` `52..54`, `c` `54..56`. The default frozen indices
//! `[0,1,48,49,50,51]` (see [`NewtonHpConfig::default`]) pin the full complex
//! positions of `u₀`, `u₁` (both order-12 white points, the whole fibre over
//! `φ = 1`) and `a₀` (one black double zero).
//!
//! PGL₂(ℂ) — real dimension 6 — acts *simply transitively* on ordered triples
//! of distinct points of ℙ¹. Freezing three full complex positions is therefore
//! a 6-real slice meeting each Möbius orbit exactly once near the solution: the
//! residual gauge group is trivial, every remaining packed coordinate is an
//! isolated algebraic number, and LLL recognition is well-posed on the raw
//! packed coordinates **as-is**. The honest gauge step is consequently the
//! **identity**, carrying the frozen indices as documentation and layout
//! validation — there is no leftover Möbius freedom to normalize away.
//!
//! Relation to [`pinned`](super::pinned): the pinned frame (`u` at `{0, ∞}`,
//! one order-5 pole at `1`, identity `A²B − λR⁵S = c·x¹²`) is a *different
//! section* of the same Möbius quotient; `newton_hp` works in the
//! both-`u`-finite chart. Moving a snapped solution between the frames is a
//! Möbius substitution plus monic renormalization on exact data and is
//! deliberately deferred, exactly as `bridge.rs` defers the common-field
//! embedding. Note also that the report packs *root* coordinates (28 complex),
//! while [`pinned_system_2_12_5`](super::pinned::pinned_system_2_12_5) has 25
//! *coefficient* unknowns: the two do not pair directly. Roots of a ℚ-rational
//! cover are generically algebraic of higher degree, so on real campaign data
//! this connector is expected to return [`ExactifyOutcome::AlgebraicCoordinates`];
//! the elementary-symmetric roots → coefficients bridge into the pinned frame is
//! the documented follow-up.
//!
//! Arithmetic caveat: recognition can only succeed over ℚ̄ ∩ (small height) if
//! the frozen values themselves are algebraic of small height — the campaign
//! must freeze at rational (or low-height algebraic) positions; a transcendental
//! pin makes every coordinate transcendental and recognition will honestly fail.
//!
//! # The stability-under-doubling gate — honest deviation
//!
//! The build spec asks the gate to "re-run `refine_hp` at 2× precision". That is
//! not implementable from `&NewtonHpReport` alone: the report carries neither
//! the seed nor the config, and `refine_hp` is hardwired to the 56-real
//! `[2,12,5]` residual while this connector must also serve other packed
//! layouts. The gate is therefore realized in two honest forms:
//!
//! * [`snap_hp_solution`] (the mandated signature) recognizes every coordinate
//!   **twice** — at the report's full effective precision `P` and again at
//!   `P/2` — and accepts only if the minimal polynomials are byte-identical.
//!   This is the same inequality read downward: the report's `P` bits *are* the
//!   "2×" run relative to `P/2`. A relation fitted to noise at one precision
//!   does not survive the other.
//! * [`snap_hp_solution_doubled`] is the literal cross-run gate: the caller
//!   re-runs `refine_hp` with `prec_bits` doubled and passes both reports; the
//!   per-coordinate minimal polynomials must be byte-identical across the two
//!   independent runs (each run also passing the internal half-precision gate).
//!
//! # Precision discipline
//!
//! The decimals are parsed as `rug::Float` — never through `f64`. The effective
//! precision is derived from the strings themselves: `refine_hp` prints
//! `prec_bits/4` significant digits, i.e. only `≈ 0.83·prec_bits` bits of
//! information, so the recognizer weight is computed from the actual digit
//! count (`⌊digits · log₂10⌋`), not from the nominal Newton precision.
//!
//! Recognition is heuristic; the *certificate* is exact: rational points are
//! back-substituted with [`PolySystem::is_exact_solution`], and each rational
//! coordinate is additionally cross-checked against the independent
//! continued-fraction recognizer
//! [`from_real`](rustmath_rationals::continued_fraction::from_real).
//!
//! [`NewtonHpConfig::default`]: super::newton_hp::NewtonHpConfig

use num_bigint::BigInt;
use rug::Float;
use rustmath_integers::Integer;
use rustmath_numberfields::recognize::recognize_complex_algebraic_hp;
use rustmath_numerical::exactify::ExactifyOutcome;
use rustmath_polynomials::poly_system::PolySystem;
use rustmath_rationals::continued_fraction::from_real;
use rustmath_rationals::Rational;

use super::newton_hp::NewtonHpReport;

const LOG2_10: f64 = std::f64::consts::LOG2_10;

/// Below this effective precision (in bits, derived from the printed digit
/// count) the recognizer's spurious-relation gates are meaningless; the
/// connector refuses to recognize rather than guess.
const MIN_EFF_BITS: u32 = 64;

/// Number of packed reals in the `[2,12,5]` layout of `newton_hp`
/// (`2·(8 + 8 + 4 + 4 + 2 + 1 + 1)`).
const PACKED_REALS_2_12_5: usize = 56;

/// The Möbius gauge of the `[2,12,5]` high-precision Newton frame.
///
/// The frozen packed indices pin the full complex positions of `u₀`, `u₁` and
/// `a₀`; since PGL₂(ℂ) acts simply transitively on ordered triples of distinct
/// points, this kills *all* Möbius freedom and the gauge transformation is the
/// identity (see the module docs for the derivation). The struct carries the
/// frozen indices and the expected packed length purely for layout validation
/// and documentation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gauge2_12_5 {
    /// Packed real indices frozen during `refine_hp` (the gauge fix); must
    /// match the `NewtonHpConfig::frozen` used to produce the report.
    pub frozen: Vec<usize>,
    /// Expected number of packed reals in the report (`Some(56)` for the
    /// production `[2,12,5]` layout; `None` skips the check).
    pub expected_reals: Option<usize>,
}

impl Gauge2_12_5 {
    /// The production gauge: the default `newton_hp` freeze
    /// `[0,1,48,49,50,51]` = `Re/Im(a₀), Re/Im(u₀), Re/Im(u₁)` on the 56-real
    /// `[2,12,5]` packing.
    pub fn pinned_default() -> Self {
        Self {
            frozen: vec![0, 1, 48, 49, 50, 51],
            expected_reals: Some(PACKED_REALS_2_12_5),
        }
    }

    /// A gauge carrying explicit frozen indices for a non-standard packed
    /// layout (synthetic systems, tests). The transformation is still the
    /// identity; no layout-length check is performed.
    pub fn with_frozen(frozen: Vec<usize>) -> Self {
        Self {
            frozen,
            expected_reals: None,
        }
    }
}

impl Default for Gauge2_12_5 {
    fn default() -> Self {
        Self::pinned_default()
    }
}

/// Count the significant decimal digits of one printed mantissa (leading zeros
/// are not significant; trailing zeros are — `to_string_radix` emits a fixed
/// digit count). A string representing exact zero yields 0.
fn sig_digits(s: &str) -> usize {
    let mantissa = s
        .split(['e', 'E', '@'])
        .next()
        .unwrap_or("")
        .trim_start_matches(['+', '-']);
    let digits: String = mantissa.chars().filter(|c| c.is_ascii_digit()).collect();
    digits.trim_start_matches('0').len()
}

/// Effective precision (bits) carried by the report's decimal strings:
/// `⌊max_digits · log₂10⌋`. `None` if every string is zero/empty.
fn effective_prec_bits(decimals: &[String]) -> Option<u32> {
    let max_digits = decimals.iter().map(|s| sig_digits(s)).max()?;
    if max_digits == 0 {
        return None;
    }
    Some(((max_digits as f64) * LOG2_10).floor() as u32)
}

/// Parse one decimal string to a finite `rug::Float` at `bits` precision —
/// never through `f64`.
fn parse_finite(s: &str, bits: u32) -> Option<Float> {
    let inc = Float::parse(s).ok()?;
    let f = Float::with_val(bits, inc);
    if f.is_finite() {
        Some(f)
    } else {
        None
    }
}

/// Recognize one complex coordinate with the two-precision stability gate:
/// the minimal polynomial found at full effective precision must be
/// byte-identical to the one found at half precision, else the coordinate is
/// rejected (this is the difference between recognition and wishful thinking).
/// Returns the minpoly plus the full-precision parse of `(re, im)`.
fn recognize_coord_stable(
    re_s: &str,
    im_s: &str,
    bits: u32,
    max_deg: usize,
) -> Option<(Vec<BigInt>, Float, Float)> {
    let re = parse_finite(re_s, bits)?;
    let im = parse_finite(im_s, bits)?;
    let full = recognize_complex_algebraic_hp(&re, &im, bits, max_deg)?;

    let half_bits = bits / 2;
    let re_h = parse_finite(re_s, half_bits)?;
    let im_h = parse_finite(im_s, half_bits)?;
    let half = recognize_complex_algebraic_hp(&re_h, &im_h, half_bits, max_deg)?;

    if full != half {
        return None;
    }
    Some((full, re, im))
}

/// A fully recognized report: per-coordinate minimal polynomials (ascending,
/// primitive, positive leading) plus the full-precision parsed coordinates.
struct RecognizedReport {
    minpolys: Vec<Vec<BigInt>>,
    coords: Vec<(Float, Float)>,
    eff_bits: u32,
}

fn recognize_report(decimals: &[String], max_deg: usize) -> Option<RecognizedReport> {
    let bits = effective_prec_bits(decimals)?;
    if bits < MIN_EFF_BITS {
        return None;
    }
    let n = decimals.len() / 2;
    let mut minpolys = Vec::with_capacity(n);
    let mut coords = Vec::with_capacity(n);
    for k in 0..n {
        let (p, re, im) =
            recognize_coord_stable(&decimals[2 * k], &decimals[2 * k + 1], bits, max_deg)?;
        minpolys.push(p);
        coords.push((re, im));
    }
    Some(RecognizedReport {
        minpolys,
        coords,
        eff_bits: bits,
    })
}

/// Assemble the outcome from a recognized report. Degree-1 minpolys become
/// rationals (cross-checked against the independent continued-fraction
/// recognizer); an all-rational point is certified by exact back-substitution.
///
/// # Panics
/// Panics if the all-rational branch is reached with a point whose arity does
/// not equal `system.num_variables()` — the caller paired a report with a
/// system of a different layout (e.g. the 28-complex-root packing against the
/// 25-coefficient pinned system), which must never be certified silently.
fn assemble(rec: RecognizedReport, system: &PolySystem) -> ExactifyOutcome {
    if rec.minpolys.iter().all(|p| p.len() == 2) {
        let mut pt: Vec<Rational> = Vec::with_capacity(rec.minpolys.len());
        for (p, (re, _im)) in rec.minpolys.iter().zip(&rec.coords) {
            let num = Integer::from(-p[0].clone());
            let den = Integer::from(p[1].clone());
            let r = Rational::new(num, den).expect("degree-1 leading coefficient is nonzero");
            // Independent second opinion: the continued-fraction best rational
            // with the same denominator bound must agree with the LLL answer
            // (|z − p/q| < 2^{-0.8·eff_bits} ≪ 1/(2q²) makes agreement forced
            // for a true rational; disagreement means a recognizer fault).
            let cf = from_real(re, r.denominator());
            if cf != r {
                return ExactifyOutcome::RecognitionFailed;
            }
            pt.push(r);
        }
        assert_eq!(
            pt.len(),
            system.num_variables(),
            "rational point arity {} does not match system arity {} — report/system layout mismatch",
            pt.len(),
            system.num_variables()
        );
        return if system.is_exact_solution(&pt) {
            ExactifyOutcome::CertifiedRational(pt)
        } else {
            ExactifyOutcome::SubstitutionFailed
        };
    }
    // Higher-degree coordinates: return per-coordinate minimal polynomials.
    // The common-field embedding is deliberately deferred (bridge.rs defers it
    // too) — a common field must never be fabricated here.
    ExactifyOutcome::AlgebraicCoordinates(rec.minpolys)
}

fn validate_layout(report: &NewtonHpReport, gauge: &Gauge2_12_5) {
    let n = report.solution_decimals.len();
    assert!(
        n % 2 == 0,
        "solution_decimals must interleave re/im pairs (got {n} strings)"
    );
    if let Some(expected) = gauge.expected_reals {
        assert_eq!(
            n, expected,
            "report packs {n} reals but the gauge expects {expected}"
        );
    }
    for &k in &gauge.frozen {
        assert!(
            k < n,
            "frozen index {k} out of range for {n} packed reals"
        );
    }
}

/// Snap a converged high-precision Newton report to exact data.
///
/// Steps: (1) parse `solution_decimals` into `rug::Float` re/im pairs at the
/// report's *effective* precision (derived from the printed digit count, never
/// via `f64`); (2) gauge-normalize — the documented identity, since the frozen
/// placement kills all Möbius freedom (see [`Gauge2_12_5`] and module docs);
/// (3) recognize each coordinate with
/// [`recognize_complex_algebraic_hp`] at the effective precision; (4) accept a
/// coordinate only if its minimal polynomial is byte-identical when recognized
/// again at half precision (the stability gate — see module docs for why the
/// doubling is realized downward); (5) all-rational points are certified by
/// exact back-substitution ([`PolySystem::is_exact_solution`]) plus an
/// independent continued-fraction cross-check per coordinate; higher-degree
/// coordinates are returned as [`ExactifyOutcome::AlgebraicCoordinates`]
/// without fabricating a common field.
///
/// A non-converged report is refused outright ([`ExactifyOutcome::RecognitionFailed`]):
/// its decimals do not carry the precision they typographically claim.
///
/// # Panics
/// Panics on structural misuse: odd `solution_decimals` length, a report whose
/// packed length contradicts `gauge.expected_reals`, a frozen index out of
/// range, or (in the all-rational branch) a point arity different from
/// `system.num_variables()`.
pub fn snap_hp_solution(
    report: &NewtonHpReport,
    gauge: &Gauge2_12_5,
    system: &PolySystem,
    max_deg: usize,
) -> ExactifyOutcome {
    validate_layout(report, gauge);
    if !report.converged {
        return ExactifyOutcome::RecognitionFailed;
    }
    // (2) gauge normalization: identity — no residual Möbius freedom to fix.
    match recognize_report(&report.solution_decimals, max_deg) {
        Some(rec) => assemble(rec, system),
        None => ExactifyOutcome::RecognitionFailed,
    }
}

/// The literal stability-under-doubling gate across two independent Newton
/// runs: `report` from `refine_hp` at some `prec_bits`, `report_2x` from a
/// re-run of `refine_hp` with `prec_bits` doubled (same seed, same frozen
/// gauge). Every coordinate's minimal polynomial must be byte-identical across
/// the two runs (each run also passing the internal half-precision gate of
/// [`snap_hp_solution`]); the outcome is then assembled from the
/// higher-precision report.
///
/// # Panics
/// Panics on structural misuse (see [`snap_hp_solution`]), on reports of
/// different packed lengths, or if `report_2x` does not actually carry at
/// least 3/2 the effective precision of `report` (it is supposed to carry 2×).
pub fn snap_hp_solution_doubled(
    report: &NewtonHpReport,
    report_2x: &NewtonHpReport,
    gauge: &Gauge2_12_5,
    system: &PolySystem,
    max_deg: usize,
) -> ExactifyOutcome {
    validate_layout(report, gauge);
    validate_layout(report_2x, gauge);
    assert_eq!(
        report.solution_decimals.len(),
        report_2x.solution_decimals.len(),
        "the two reports must pack the same layout"
    );
    if !report.converged || !report_2x.converged {
        return ExactifyOutcome::RecognitionFailed;
    }
    let (Some(bits_lo), Some(bits_hi)) = (
        effective_prec_bits(&report.solution_decimals),
        effective_prec_bits(&report_2x.solution_decimals),
    ) else {
        return ExactifyOutcome::RecognitionFailed;
    };
    assert!(
        2 * bits_hi >= 3 * bits_lo,
        "report_2x carries {bits_hi} effective bits — not a doubled-precision re-run of {bits_lo}"
    );
    let Some(lo) = recognize_report(&report.solution_decimals, max_deg) else {
        return ExactifyOutcome::RecognitionFailed;
    };
    let Some(hi) = recognize_report(&report_2x.solution_decimals, max_deg) else {
        return ExactifyOutcome::RecognitionFailed;
    };
    if lo.minpolys != hi.minpolys {
        return ExactifyOutcome::RecognitionFailed;
    }
    debug_assert!(hi.eff_bits > lo.eff_bits);
    assemble(hi, system)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belyi::newton_hp::NewtonHpConfig;

    /// Mock a `NewtonHpReport` from exact coordinates, rendered exactly the way
    /// `refine_hp` renders them (`to_string_radix(10, Some(prec_bits/4))`).
    ///
    /// HONESTY NOTE: the report is mocked, not produced by `refine_hp`, because
    /// `refine_hp` is hardwired to the 56-real `[2,12,5]` residual and no true
    /// `[2,12,5]` seed is available — a synthetic small system cannot drive it.
    /// The build spec explicitly sanctions this.
    fn mock_report(coords: &[(Float, Float)], digits: usize) -> NewtonHpReport {
        let mut solution_decimals = Vec::with_capacity(2 * coords.len());
        for (re, im) in coords {
            solution_decimals.push(re.to_string_radix(10, Some(digits)));
            solution_decimals.push(im.to_string_radix(10, Some(digits)));
        }
        NewtonHpReport {
            initial_residual: 0.0,
            final_residual: 0.0,
            iterations: 0,
            converged: true,
            history: vec![],
            solution_decimals,
        }
    }

    /// Synthetic system with the known exact solution (x, y, z) = (1/3, −7/2, 0):
    /// 3x − 1 = 0, 2y + 7 = 0, z = 0, 6xy + 7 = 0 (hand-check: 6·(1/3)·(−7/2) = −7).
    fn synthetic_system() -> PolySystem {
        PolySystem::from_terms(
            3,
            &[
                vec![(vec![1, 0, 0], 3), (vec![0, 0, 0], -1)],
                vec![(vec![0, 1, 0], 2), (vec![0, 0, 0], 7)],
                vec![(vec![0, 0, 1], 1)],
                vec![(vec![1, 1, 0], 6), (vec![0, 0, 0], 7)],
            ],
        )
    }

    fn zero512() -> Float {
        Float::with_val(512, 0)
    }

    fn known_solution_coords(prec: u32) -> Vec<(Float, Float)> {
        let third = Float::with_val(prec, 1) / 3u32;
        let y = Float::with_val(prec, -7) / 2u32;
        vec![
            (third, Float::with_val(prec, 0)),
            (y, Float::with_val(prec, 0)),
            (Float::with_val(prec, 0), Float::with_val(prec, 0)),
        ]
    }

    fn expected_solution() -> Vec<Rational> {
        vec![
            Rational::new(1, 3).unwrap(),
            Rational::new(-7, 2).unwrap(),
            Rational::from_i64(0),
        ]
    }

    #[test]
    fn gauge_matches_newton_hp_default_freeze() {
        let g = Gauge2_12_5::pinned_default();
        assert_eq!(g.frozen, NewtonHpConfig::default().frozen);
        assert_eq!(g.expected_reals, Some(56));
        assert_eq!(Gauge2_12_5::default(), g);
    }

    #[test]
    fn effective_precision_from_digit_count() {
        // 128 significant digits carry ⌊128·log₂10⌋ = 425 bits — independent
        // check: log₂(10¹²⁸) ≈ 425.21.
        let third = Float::with_val(512, 1) / 3u32;
        let s = third.to_string_radix(10, Some(128));
        assert_eq!(sig_digits(&s), 128);
        assert_eq!(effective_prec_bits(&[s]), Some(425));
        // Exact zero carries no digits and must not dominate.
        let z = zero512().to_string_radix(10, Some(128));
        assert_eq!(sig_digits(&z), 0);
    }

    #[test]
    fn snaps_known_rational_solution() {
        let report = mock_report(&known_solution_coords(512), 128);
        let system = synthetic_system();
        let gauge = Gauge2_12_5::with_frozen(vec![0, 1]);
        let outcome = snap_hp_solution(&report, &gauge, &system, 4);
        let expected = expected_solution();
        assert_eq!(outcome, ExactifyOutcome::CertifiedRational(expected.clone()));
        // The certificate re-checked directly (exact back-substitution).
        assert!(system.is_exact_solution(&expected));
    }

    /// Poison one decimal at the 2⁻⁶⁰ level: the recognizer must reject
    /// (RecognitionFailed), never fabricate a nearby rational — the residual
    /// gate at the effective precision (2⁻³⁴⁰ for 425 bits) sits far below the
    /// 2⁻⁶⁰ contamination.
    #[test]
    fn poisoned_decimal_is_rejected_not_snapped() {
        let mut coords = known_solution_coords(512);
        let eps = Float::with_val(512, 1) >> 60u32; // exact 2^-60
        coords[0].0 += &eps;
        let report = mock_report(&coords, 128);
        let outcome = snap_hp_solution(
            &report,
            &Gauge2_12_5::with_frozen(vec![0, 1]),
            &synthetic_system(),
            4,
        );
        assert_eq!(outcome, ExactifyOutcome::RecognitionFailed);
    }

    /// The stability gate itself must be the operative rejector for a rational
    /// whose height fits the full-precision acceptance bound but not the
    /// half-precision one: p/2⁶⁰ with p odd. At 425 effective bits the degree-1
    /// coefficient bound is 2^⌊0.6·425⌋⌄³ = 2⁸⁵ ≥ 2⁶⁰ (accepted); at 212 bits
    /// it is 2⁴² < 2⁶⁰ (rejected) ⇒ byte-mismatch ⇒ RecognitionFailed. This is
    /// deliberately conservative: certification demands precision headroom.
    #[test]
    fn stability_gate_rejects_precision_marginal_height() {
        let p: i64 = 987_654_321_987_654_321; // odd, < 2^60, gcd(p, 2^60) = 1
        let q: i64 = 1 << 60;
        let v = Float::with_val(512, p) / Float::with_val(512, q); // exact (÷2^60)
        let coords = vec![(v, Float::with_val(512, 0))];
        let report = mock_report(&coords, 128);

        // Premise 1: full effective precision (425 bits) recognizes [−p, 2^60].
        let bits = effective_prec_bits(&report.solution_decimals).unwrap();
        assert_eq!(bits, 425);
        let re = parse_finite(&report.solution_decimals[0], bits).unwrap();
        let im = parse_finite(&report.solution_decimals[1], bits).unwrap();
        let full = recognize_complex_algebraic_hp(&re, &im, bits, 4).unwrap();
        assert_eq!(full, vec![BigInt::from(-p), BigInt::from(q)]);

        // Premise 2: half precision does NOT reproduce it (coefficient bound
        // 2^42 < 2^60, and no smaller-height relation exists).
        let re_h = parse_finite(&report.solution_decimals[0], bits / 2).unwrap();
        let im_h = parse_finite(&report.solution_decimals[1], bits / 2).unwrap();
        let half = recognize_complex_algebraic_hp(&re_h, &im_h, bits / 2, 4);
        assert_ne!(half, Some(full));

        // Therefore the gate rejects — no fabricated value.
        let system = PolySystem::from_terms(1, &[vec![(vec![1], q), (vec![0], -p)]]);
        let outcome = snap_hp_solution(&report, &Gauge2_12_5::with_frozen(vec![0]), &system, 4);
        assert_eq!(outcome, ExactifyOutcome::RecognitionFailed);
    }

    /// Higher-degree coordinates come back as per-coordinate minimal
    /// polynomials — no common field is fabricated. Oracles: √2 ⇒ x² − 2 ⇒
    /// [−2, 0, 1]; (1+i)/2 ⇒ (2z−1)² = −1 ⇒ 2z² − 2z + 1 ⇒ [1, −2, 2].
    #[test]
    fn algebraic_coordinates_pass_through_with_stable_minpolys() {
        let sqrt2 = Float::with_val(512, 2).sqrt();
        let half = Float::with_val(512, 1) / 2u32;
        let coords = vec![(sqrt2, Float::with_val(512, 0)), (half.clone(), half)];
        let report = mock_report(&coords, 128);
        // Dummy 2-var system: never consulted on the algebraic branch.
        let system = PolySystem::from_terms(2, &[vec![(vec![1, 0], 1)]]);
        let outcome = snap_hp_solution(&report, &Gauge2_12_5::with_frozen(vec![0]), &system, 4);
        let iv = |v: &[i64]| v.iter().map(|&x| BigInt::from(x)).collect::<Vec<_>>();
        assert_eq!(
            outcome,
            ExactifyOutcome::AlgebraicCoordinates(vec![iv(&[-2, 0, 1]), iv(&[1, -2, 2])])
        );
    }

    /// Rational coordinates that are not an exact zero of the system must be
    /// reported as SubstitutionFailed (a spurious path), never certified.
    #[test]
    fn spurious_rational_point_is_substitution_failed() {
        let mut coords = known_solution_coords(512);
        coords[2].0 = Float::with_val(512, 1); // z = 1 breaks the equation z = 0
        let report = mock_report(&coords, 128);
        let outcome = snap_hp_solution(
            &report,
            &Gauge2_12_5::with_frozen(vec![0, 1]),
            &synthetic_system(),
            4,
        );
        assert_eq!(outcome, ExactifyOutcome::SubstitutionFailed);
    }

    #[test]
    fn refuses_non_converged_and_low_precision_reports() {
        let mut report = mock_report(&known_solution_coords(512), 128);
        report.converged = false;
        let system = synthetic_system();
        let gauge = Gauge2_12_5::with_frozen(vec![0, 1]);
        assert_eq!(
            snap_hp_solution(&report, &gauge, &system, 4),
            ExactifyOutcome::RecognitionFailed
        );
        // 15 digits ⇒ 49 effective bits < MIN_EFF_BITS ⇒ refused.
        let short = mock_report(&known_solution_coords(512), 15);
        assert_eq!(
            snap_hp_solution(&short, &gauge, &system, 4),
            ExactifyOutcome::RecognitionFailed
        );
    }

    #[test]
    #[should_panic(expected = "packs 6 reals but the gauge expects 56")]
    fn production_gauge_rejects_wrong_layout() {
        let report = mock_report(&known_solution_coords(512), 128);
        let _ = snap_hp_solution(
            &report,
            &Gauge2_12_5::pinned_default(),
            &synthetic_system(),
            4,
        );
    }

    /// The literal cross-run doubling gate: a 512-bit report plus its 1024-bit
    /// re-run certify; if the doubled run is contaminated at 2⁻²⁰⁰ (below the
    /// low run's resolution but far above the high run's residual gate), the
    /// pair is rejected.
    #[test]
    fn doubled_reports_certify_and_reject_instability() {
        let system = synthetic_system();
        let gauge = Gauge2_12_5::with_frozen(vec![0, 1]);
        let lo = mock_report(&known_solution_coords(512), 128);
        let hi = mock_report(&known_solution_coords(1024), 256);
        assert_eq!(
            snap_hp_solution_doubled(&lo, &hi, &gauge, &system, 4),
            ExactifyOutcome::CertifiedRational(expected_solution())
        );

        let mut poisoned = known_solution_coords(1024);
        let eps = Float::with_val(1024, 1) >> 200u32; // exact 2^-200
        poisoned[0].0 += &eps;
        let hi_bad = mock_report(&poisoned, 256);
        assert_eq!(
            snap_hp_solution_doubled(&lo, &hi_bad, &gauge, &system, 4),
            ExactifyOutcome::RecognitionFailed
        );
    }
}
