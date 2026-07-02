//! The end-to-end decide pipeline, the **G6 Müller gate**, the M23 uniform-law
//! aggregator, and the `[2,12,5]` parameter-homotopy solve assembler.
//!
//! Ported and folded from `dessin_engine/src/pipeline.rs` and
//! `dessin_engine/src/portal.rs` in `/home/john/inverse_galois/M23/dessin_engine`
//! ("everything into existing crates"): the `PortalConic`/`UniformLaw`/
//! [`uniform_law`] decision aggregator lives here rather than beside the genus-0
//! ansatz in [`crate::belyi::portal`]. The `dessin_engine` solver stack
//! (`RealIsolatedEngine`/Gröbner) is **not** ported — RustMath solves via the
//! parameter homotopy (Julia), so `solve_and_bridge` is replaced by:
//!
//! * [`assemble_2_12_5_homotopy`] — build the `ParameterHomotopyJob` that tracks
//!   `p₀ = ψ(seed) → p* = pinned::p_star()` for the 24 solving variables; and
//! * [`decide_from_solved_cover`] — the plumbing that, *once the emitted Julia job
//!   runs*, chains exactify → cover-verify → bad-locus → bridge → descent →
//!   conic → portal-conic, leaving "run the Julia job" as the only open step.
//!
//! **G6** is the validation gate: Müller's M23-subcover carries the anisotropic
//! Hamilton conic `(-1,-1)`, ramified `{2,∞}` — so [`g6_mueller_gate`] must report
//! it as `LocallyEmpty` (a correct negative: no M23/Q through Müller).

use crate::belyi::audit::{self, GroupKind, GroupVerdict};
use crate::belyi::bad_locus::{BadLocusStatus, GenusZeroBelyiFactorizationQ, P1PointQ};
use crate::belyi::bridge::{bridge_and_read, read_explicit_conic, SolvedCover};
use crate::belyi::descent::{
    certify_phi_sigma_over_L, descent_conic, g_sigma_from_solved_cover, Gl2Quad, LCover,
    SigmaCorrespondence,
};
use crate::belyi::pinned::{self, pinned_system_2_12_5};
use crate::belyi::verify::{verify_2_12_5_cover, ExactBelyiCover, VerificationReport};
use rustmath_groups::transitive23::Group23;
use rustmath_integers::Integer;
use rustmath_numerical::exactify::{exactify, ExactifyOutcome};
use rustmath_numerical::homotopy::{ComplexDecimal, NumericalSolution, ParameterHomotopyJob};
use rustmath_polynomials::multivariate::{Monomial, MultivariatePolynomial};
use rustmath_polynomials::poly_system::PolySystem;
use rustmath_quadraticforms::conic::{ConicBrauerReport, Verdict, VerdictKind};
use rustmath_quadraticforms::hilbert::Place;
use rustmath_quadraticforms::ternary::TernaryForm;
use rustmath_rationals::Rational;
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// The G6 Müller gate
// ---------------------------------------------------------------------------

/// The Müller conic, as the descent (or hand-descent) produces it: `x²+y²+z²`.
pub fn mueller_m23_conic() -> TernaryForm {
    let z = Rational::from_i64(0);
    TernaryForm::from_coeffs(
        Rational::from_i64(1),
        Rational::from_i64(1),
        Rational::from_i64(1),
        z.clone(),
        z.clone(),
        z,
    )
    .expect("x^2+y^2+z^2 is a nondegenerate ternary form")
}

/// The G6 gate: read Müller's M23-conic and verify it is the anisotropic
/// `(-1,-1)` ramified `{2,∞}` — i.e. **not** M23/Q (a correct negative).
pub fn g6_mueller_gate() -> Verdict<ConicBrauerReport> {
    read_explicit_conic(&mueller_m23_conic(), false)
}

// ---------------------------------------------------------------------------
// The M23 uniform-law aggregator (folded from dessin_engine/src/portal.rs)
// ---------------------------------------------------------------------------

/// The conic outcome of one portal, once read by D4.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PortalConic {
    /// Split — a rational point exists (M23/Q realized through this portal).
    Split,
    /// Anisotropic over Q, ramified at the given (nonempty) place set.
    Obstructed(Vec<Place>),
    /// Not yet decided (solver / descent pending).
    Unresolved,
}

impl PortalConic {
    pub fn from_report(report: &ConicBrauerReport) -> Self {
        if report.has_rational_point {
            PortalConic::Split
        } else {
            let mut places = report.ramified_places.clone();
            places.sort_by_key(|p| match p {
                Place::Real => (0u64, 0u64),
                Place::Finite(q) => (1, *q),
            });
            PortalConic::Obstructed(places)
        }
    }
}

/// The batch verdict over a set of portal conics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UniformLaw {
    /// Some portal splits: M23/Q is realized (names the portal).
    Realized(String),
    /// Every resolved portal is obstructed at the *same* place set — the
    /// conjectural uniform law, with that common ramification.
    UniformObstruction(Vec<Place>),
    /// All resolved, obstructed, but at differing place sets — no uniform law.
    Mixed,
    /// At least one portal is still unresolved — the batch is incomplete.
    Incomplete,
}

/// Aggregate the portal conics into the uniform-law verdict.
pub fn uniform_law(outcomes: &[(String, PortalConic)]) -> UniformLaw {
    if let Some((name, _)) = outcomes.iter().find(|(_, c)| *c == PortalConic::Split) {
        return UniformLaw::Realized(name.clone());
    }
    if outcomes.iter().any(|(_, c)| *c == PortalConic::Unresolved) {
        return UniformLaw::Incomplete;
    }
    let mut sets = outcomes.iter().filter_map(|(_, c)| match c {
        PortalConic::Obstructed(s) => Some(s.clone()),
        _ => None,
    });
    let Some(first) = sets.next() else {
        return UniformLaw::Incomplete;
    };
    if sets.all(|s| s == first) {
        UniformLaw::UniformObstruction(first)
    } else {
        UniformLaw::Mixed
    }
}

// ---------------------------------------------------------------------------
// The [2,12,5] parameter-homotopy solve assembler
// ---------------------------------------------------------------------------

/// A concrete rational seed `z₀ = (A₀, B₀, R₀, S₀, λ₀)` of the 24 solving
/// variables — the free start solution of the parameter homotopy.
///
/// Coefficient conventions match [`pinned::psi`] (ascending in `x`, monic leading
/// term implicit): `a` = `[a0..a7]` (8), `b` = `[b0..b7]` (8),
/// `r` = `[r0,r1,r2]` (3, for `R = (x−1)(x³ + r₂x² + r₁x + r₀)`),
/// `s` = `[s0..s3]` (4), and `lambda` = `λ`.
#[derive(Debug, Clone)]
pub struct HomotopySeed {
    pub a: Vec<Rational>,
    pub b: Vec<Rational>,
    pub r: Vec<Rational>,
    pub s: Vec<Rational>,
    pub lambda: Rational,
}

// Variable-index layout of the parameter system:
//   a0..a7 -> 0..=7, b0..b7 -> 8..=15, r0..r2 -> 16..=18, s0..s3 -> 19..=22,
//   lambda -> 23, parameters p0..p24 -> 24..=48, and the polynomial variable
//   x -> 49 (stripped after coefficient extraction).
const A_BASE: usize = 0;
const B_BASE: usize = 8;
const R_BASE: usize = 16;
const S_BASE: usize = 19;
const LAMBDA: usize = 23;
const PARAM_BASE: usize = 24;
const X: usize = 49;
/// Number of solving variables (`a×8, b×8, r×3, s×4, λ`).
pub const NUM_SOLVING_VARS: usize = 24;
/// Number of parameters (`= 25` coefficients `x⁰..x²⁴` of `A²B − λR⁵S`).
pub const NUM_PARAMS: usize = 25;

fn mono(pairs: &[(usize, u32)]) -> Monomial {
    let mut map = BTreeMap::new();
    for &(v, e) in pairs {
        if e > 0 {
            map.insert(v, e);
        }
    }
    Monomial::from_exponents(map)
}

fn poly_pow(base: &MultivariatePolynomial<Integer>, e: u32) -> MultivariatePolynomial<Integer> {
    let mut acc = MultivariatePolynomial::<Integer>::constant(Integer::one());
    for _ in 0..e {
        acc = acc * base.clone();
    }
    acc
}

/// Monic degree-`deg` form `x^deg + Σ_{i<deg} v_{base+i} · x^i`.
fn monic_form(base: usize, deg: usize) -> MultivariatePolynomial<Integer> {
    let mut f = MultivariatePolynomial::<Integer>::zero();
    f.add_term(mono(&[(X, deg as u32)]), Integer::one());
    for i in 0..deg {
        f.add_term(mono(&[(base + i, 1), (X, i as u32)]), Integer::one());
    }
    f
}

/// The parameter system `F(z; p) = 0` with equation `k`
/// `[xᵏ](A²B − λR⁵S) − pₖ = 0`, `k = 0..24`. Solve variables occupy slots
/// `0..24`; the 25 parameters `p₀..p₂₄` occupy slots `24..49`, so the whole
/// system has 49 variables (as [`ParameterHomotopyJob::from_system`] expects for
/// `variables(24) + parameters(25)`).
fn parameter_system_2_12_5() -> PolySystem {
    let a = monic_form(A_BASE, 8);
    let b = monic_form(B_BASE, 8);
    let s = monic_form(S_BASE, 4);

    // cubic = x^3 + r2 x^2 + r1 x + r0 ; R = (x - 1) * cubic
    let mut cubic = MultivariatePolynomial::<Integer>::zero();
    cubic.add_term(mono(&[(X, 3)]), Integer::one());
    cubic.add_term(mono(&[(R_BASE + 2, 1), (X, 2)]), Integer::one());
    cubic.add_term(mono(&[(R_BASE + 1, 1), (X, 1)]), Integer::one());
    cubic.add_term(mono(&[(R_BASE, 1)]), Integer::one());
    let mut x_minus_1 = MultivariatePolynomial::<Integer>::zero();
    x_minus_1.add_term(mono(&[(X, 1)]), Integer::one());
    x_minus_1.add_term(mono(&[]), Integer::from(-1));
    let r = x_minus_1 * cubic;

    // LHS = A^2 B - lambda * (R^5 S)
    let lhs_full = poly_pow(&a, 2) * b;
    let lambda = MultivariatePolynomial::<Integer>::variable(LAMBDA);
    let q = lambda * (poly_pow(&r, 5) * s);
    let expr = lhs_full - q;

    // Bucket by x-degree, drop the x variable.
    let mut lhs: Vec<MultivariatePolynomial<Integer>> =
        vec![MultivariatePolynomial::<Integer>::zero(); NUM_PARAMS];
    for (m, coeff) in expr.terms() {
        let d = m.exponent(X) as usize;
        debug_assert!(d < NUM_PARAMS, "x-degree {d} exceeds {}", NUM_PARAMS - 1);
        let mut stripped = BTreeMap::new();
        for (&v, &e) in m.iter_exponents() {
            if v != X {
                stripped.insert(v, e);
            }
        }
        lhs[d].add_term(Monomial::from_exponents(stripped), coeff.clone());
    }

    // Equation k: LHS_k(z) - p_k = 0.
    let eqs: Vec<MultivariatePolynomial<Integer>> = (0..NUM_PARAMS)
        .map(|k| lhs[k].clone() - MultivariatePolynomial::<Integer>::variable(PARAM_BASE + k))
        .collect();

    PolySystem::new(NUM_SOLVING_VARS + NUM_PARAMS, eqs)
}

/// Ordered names of the 25 parameters `p0..p24`.
fn parameter_names() -> Vec<String> {
    (0..NUM_PARAMS).map(|k| format!("p{k}")).collect()
}

/// Deterministic exact decimal rendering of a rational to `frac_digits` places —
/// a Julia-parseable float literal, used for both the seed (start solution) and
/// the parameter points.
fn rat_to_decimal(r: &Rational, frac_digits: usize) -> String {
    use num_bigint::BigInt;
    use num_traits::{Signed, Zero};
    let num: BigInt = r.numerator().to_string().parse().expect("integer parses");
    let den: BigInt = r.denominator().to_string().parse().expect("integer parses");
    let neg = (num.is_negative()) ^ (den.is_negative());
    let n = num.abs();
    let d = den.abs();
    let int_part = &n / &d;
    let mut rem = &n - &int_part * &d;
    let mut s = String::new();
    if neg && !(int_part.is_zero() && rem.is_zero()) {
        s.push('-');
    }
    s.push_str(&int_part.to_string());
    s.push('.');
    let ten = BigInt::from(10);
    for _ in 0..frac_digits {
        rem *= &ten;
        let digit = &rem / &d;
        s.push_str(&digit.to_string());
        rem -= &digit * &d;
    }
    s
}

fn rat_to_complex(r: &Rational) -> ComplexDecimal {
    ComplexDecimal::real(rat_to_decimal(r, 40))
}

/// Assemble the end-to-end `[2,12,5]` solve as a parameter-homotopy job.
///
/// Given a rational seed `z₀`, computes the free start parameters
/// `p₀ = ψ(z₀)` ([`pinned::psi`]) — so `z₀` is an exact start solution of
/// `F(z; p₀) = 0` — and the target `p* = ` [`pinned::p_star`] (the monomial
/// `c·x¹²` with `c` absorbed to 1). The returned [`ParameterHomotopyJob`] tracks
/// `p₀ → p*` over the 24 solving variables. No numerical code is run here; the
/// only remaining step is to render + execute the emitted Julia job.
///
/// # Panics
/// Panics if any seed slice has the wrong length (via [`pinned::psi`]).
pub fn assemble_2_12_5_homotopy(seed: &HomotopySeed) -> ParameterHomotopyJob {
    let system = parameter_system_2_12_5();
    let variables = pinned::solving_var_names();
    let parameters = parameter_names();

    // Free start solution z0 = seed, in solving-variable order.
    let mut start_solution: Vec<ComplexDecimal> = Vec::with_capacity(NUM_SOLVING_VARS);
    for r in seed
        .a
        .iter()
        .chain(seed.b.iter())
        .chain(seed.r.iter())
        .chain(seed.s.iter())
        .chain(std::iter::once(&seed.lambda))
    {
        start_solution.push(rat_to_complex(r));
    }

    // Start parameters p0 = psi(seed); target p* = pinned::p_star().
    let p0 = pinned::psi(&seed.a, &seed.b, &seed.r, &seed.s, &seed.lambda);
    let start_parameters: Vec<ComplexDecimal> = p0.iter().map(rat_to_complex).collect();
    let target_parameters: Vec<ComplexDecimal> =
        pinned::p_star().iter().map(rat_to_complex).collect();

    ParameterHomotopyJob::from_system(
        "belyi_2_12_5",
        &system,
        &variables,
        &parameters,
        start_solution,
        start_parameters,
        target_parameters,
        "belyi_2_12_5_result.json",
    )
    .expect("parameter system shape matches variables(24) + parameters(25)")
}

// ---------------------------------------------------------------------------
// The decide pipeline (post-solve plumbing)
// ---------------------------------------------------------------------------

/// The full record of running a solved cover through the decide chain.
#[derive(Debug, Clone)]
pub struct DecideReport {
    pub name: String,
    /// Stage 1 — exactification of the numerical candidate.
    pub exactify: ExactifyOutcome,
    /// Stage 2 — the independent cover verification (hard constructibility gate).
    pub verification: VerificationReport,
    /// Stage 3 — bad-locus classification of the produced point (if supplied).
    pub bad_locus: Option<BadLocusStatus>,
    /// Stages 4–6 — the conic verdict (bridge, or descent when a cocycle is given).
    pub conic: Verdict<ConicBrauerReport>,
    /// The aggregated portal-conic outcome, ready for [`uniform_law`].
    pub portal_conic: PortalConic,
}

/// Chain the whole decide half, *once the emitted Julia job has produced a
/// candidate* (`sol`): exactify → cover-verify → bad-locus → bridge → (descent) →
/// conic → portal-conic.
///
/// * `system` is the exact solving system the candidate is checked against
///   (`rustmath_numerical::exactify`).
/// * `cover` supplies the independent constructibility audit
///   ([`ExactBelyiCover`]); a `Constructed` conic is **only** promoted to a
///   `Split` portal-conic when the cover is affirmatively constructible *and* the
///   produced point is `Z_C`-clear — an honest refusal of false positives.
/// * `bad_locus` optionally supplies `(factorization, point)` for the S4 `Z_C`
///   classification (the predicate never emits `Clear`, so a clearance flag stays
///   `false` until an independent monodromy certifier upgrades it).
/// * `descent_cocycle` optionally supplies the `PGL₂(L)` cocycle for an
///   extension-field cover; when present the conic is read via
///   [`descent_conic`] rather than the split bridge.
pub fn decide_from_solved_cover<C: ExactBelyiCover>(
    name: &str,
    sol: &NumericalSolution,
    system: &PolySystem,
    max_deg: usize,
    cover: &C,
    bad_locus: Option<(&GenusZeroBelyiFactorizationQ, &P1PointQ)>,
    descent_cocycle: Option<&Gl2Quad>,
) -> DecideReport {
    // Stage 1: exactify the numerical candidate against the exact system.
    let outcome = exactify(sol, system, max_deg);

    // Stage 2: independent cover verification (never a pass without monodromy).
    let verification = verify_2_12_5_cover(cover);

    // Stage 3: bad-locus classification of the produced point.
    let bad_locus_status = bad_locus.map(|(f, p)| f.classify(p));
    let bad_locus_clear = matches!(bad_locus_status, Some(BadLocusStatus::Clear));

    // Stages 4–5: bridge to a conic (or run descent for an extension-field cover).
    let conic = if let Some(g) = descent_cocycle {
        descent_conic(g, bad_locus_clear)
    } else {
        match SolvedCover::from_exactify(&outcome) {
            Some(c) => bridge_and_read(&c),
            None => Verdict::unresolved("exactification did not yield a solved cover"),
        }
    };

    // Stage 6: read into a portal-conic; a Split is only asserted on a full pass.
    let portal_conic = match (&conic.value, conic.kind) {
        (Some(report), VerdictKind::Constructed) => {
            if verification.is_constructible() && bad_locus_clear {
                PortalConic::from_report(report)
            } else {
                PortalConic::Unresolved
            }
        }
        (Some(report), VerdictKind::LocallyEmpty) => PortalConic::from_report(report),
        _ => PortalConic::Unresolved,
    };

    DecideReport {
        name: name.to_string(),
        exactify: outcome,
        verification,
        bad_locus: bad_locus_status,
        conic,
        portal_conic,
    }
}

// ---------------------------------------------------------------------------
// The NON-PROVISIONAL decision — the audit-backed gate.
// ---------------------------------------------------------------------------

/// The audit-backed, non-provisional verdict for a solved cover.
#[derive(Debug, Clone)]
pub enum NonProvisionalVerdict {
    /// **The payoff.** The cover is over `Q`, its monodromy is confirmed `M24`,
    /// and a rational `1+23` split yields an irreducible degree-23 residual
    /// classified as `M23`. `residual23` **is** the M23/`Q` witnessing polynomial.
    M23QRealized {
        residual23: Vec<Integer>,
        group: Group23,
    },
    /// The cover is over a quadratic `L = Q(√δ)`; the gluing `φ^σ = φ ∘ g_σ` is
    /// certified exactly (B3), and the descent conic `(δ, β)` gives the Hasse
    /// obstruction verdict *through this cover*.
    ObstructedThroughCover {
        conic: Verdict<ConicBrauerReport>,
        portal: PortalConic,
    },
    /// Honest refusal: the audit does not force the group (parasitic/degenerate
    /// solution, unforced M24, reducible residual, uncertified gluing, or missing
    /// inputs). Never asserts a group without the evidence.
    Unresolved(String),
}

/// The full record of a non-provisional decision.
#[derive(Debug, Clone)]
pub struct DecideReportNonProvisional {
    pub name: String,
    /// Stage 1 — exactification (the field of definition).
    pub exactify: ExactifyOutcome,
    /// Stage 2 — the independent M24 monodromy audit (over `Q`; `Unresolved` for
    /// the `L`-coordinatized branch, where B3 is the certificate instead).
    pub m24: GroupVerdict,
    /// The bad-locus (S4) classification, when a point was supplied.
    pub bad_locus: Option<BadLocusStatus>,
    /// The decisive verdict.
    pub verdict: NonProvisionalVerdict,
    pub notes: Vec<String>,
}

impl DecideReportNonProvisional {
    /// True exactly when the M23/`Q` realization is witnessed (the payoff).
    pub fn is_m23_q_realized(&self) -> bool {
        matches!(self.verdict, NonProvisionalVerdict::M23QRealized { .. })
    }
}

fn portal_from_conic(conic: &Verdict<ConicBrauerReport>) -> PortalConic {
    match (&conic.value, conic.kind) {
        (Some(report), VerdictKind::Constructed) | (Some(report), VerdictKind::LocallyEmpty) => {
            PortalConic::from_report(report)
        }
        _ => PortalConic::Unresolved,
    }
}

/// The **non-provisional decision gate** for the `[2,12,5]` solve.
///
/// Chains, once a candidate `sol` is in hand:
/// 1. `exactify` against [`pinned_system_2_12_5`] → the field of definition;
/// 2. run [`audit::audit_m24`]; a verdict is only trustworthy when the monodromy
///    is **Confirmed(M24)** — a parasitic/degenerate solution fails here;
/// 3. **over `Q`** (`CertifiedRational`): run [`audit::audit_m23_residual`]; an
///    irreducible degree-23 residual classified `M23` ⇒
///    [`NonProvisionalVerdict::M23QRealized`] with the witnessing polynomial —
///    *the* M23/`Q` realization;
/// 4. **over `L`** (`AlgebraicCoordinates`, quadratic): recover `g_σ` (B5), certify
///    `φ^σ = φ ∘ g_σ` exactly (B3), then read the descent conic `(δ, β)` →
///    [`NonProvisionalVerdict::ObstructedThroughCover`];
/// 5. anything ambiguous ⇒ honest [`NonProvisionalVerdict::Unresolved`]. The
///    bad-locus (S4) must be clear before a conic is promoted.
///
/// `x0_candidates` are rational source points to try for the `1+23` split (the
/// true `x₀` are rational roots of the solved numerator). `l_descent` supplies the
/// `L`-cover and the labelled `σ`-correspondence for the quadratic branch (absent
/// until the solved `L`-coordinates are available).
///
/// NOTE (before publication): follow this native statistical audit with an exact
/// OSCAR `galois_group` cross-check of the degree-24 fibre and the degree-23
/// residual.
#[allow(clippy::too_many_arguments)]
pub fn decide_nonprovisional(
    name: &str,
    sol: &NumericalSolution,
    max_deg: usize,
    primes: &[i64],
    x0_candidates: &[Rational],
    l_descent: Option<(&LCover, &SigmaCorrespondence)>,
    bad_locus: Option<(&GenusZeroBelyiFactorizationQ, &P1PointQ)>,
) -> DecideReportNonProvisional {
    let system = pinned_system_2_12_5();
    let outcome = exactify(sol, &system, max_deg);

    let bad_locus_status = bad_locus.map(|(f, p)| f.classify(p));
    let bad_locus_clear = matches!(bad_locus_status, Some(BadLocusStatus::Clear));

    let mut notes: Vec<String> =
        vec!["non-provisional: audit-backed (OSCAR galois_group cross-check to follow)".into()];

    let (m24, verdict) = match &outcome {
        ExactifyOutcome::CertifiedRational(pt) => {
            // Over Q: the decisive M24 gate, then the M23/Q realization.
            let m24 = audit::audit_m24(pt, primes);
            if m24 == GroupVerdict::Confirmed(GroupKind::M24) {
                let w = audit::audit_m23_residual(pt, x0_candidates, primes);
                notes.extend(w.notes.iter().cloned());
                match (w.is_m23_realized(), w.residual, w.group23) {
                    (true, Some(residual23), Some(group)) => (
                        m24,
                        NonProvisionalVerdict::M23QRealized { residual23, group },
                    ),
                    _ => (
                        m24,
                        NonProvisionalVerdict::Unresolved(
                            "M24 confirmed but no irreducible degree-23 residual classified M23 \
                             (try more rational x0 source points / more primes)"
                                .into(),
                        ),
                    ),
                }
            } else {
                (
                    m24.clone(),
                    NonProvisionalVerdict::Unresolved(format!(
                        "audit_m24 = {m24:?} (not Confirmed(M24)): parasitic/degenerate or \
                         under-sampled — verdict withheld"
                    )),
                )
            }
        }
        ExactifyOutcome::AlgebraicCoordinates(_) => {
            // Over L: the Q-Frobenius M24 audit does not apply to the L-coordinate
            // tuple; the exact B3 gluing identity is the certificate instead.
            notes.push(
                "cover over quadratic L: M24 confirmed via the exact φ^σ=φ∘g_σ gluing (B3), \
                 not the Q-Frobenius audit"
                    .into(),
            );
            match l_descent {
                Some((cover, corr)) => match g_sigma_from_solved_cover(corr) {
                    Ok(g) => {
                        if certify_phi_sigma_over_L(cover, &g) {
                            let conic = descent_conic(&g, bad_locus_clear);
                            let portal = portal_from_conic(&conic);
                            (
                                GroupVerdict::Unresolved,
                                NonProvisionalVerdict::ObstructedThroughCover { conic, portal },
                            )
                        } else {
                            (
                                GroupVerdict::Unresolved,
                                NonProvisionalVerdict::Unresolved(
                                    "B3 FAILED: φ^σ = φ∘g_σ is not an exact identity over L — \
                                     gluing uncertified"
                                        .into(),
                                ),
                            )
                        }
                    }
                    Err(e) => (
                        GroupVerdict::Unresolved,
                        NonProvisionalVerdict::Unresolved(format!(
                            "B5 g_σ recovery failed: {e:?}"
                        )),
                    ),
                },
                None => (
                    GroupVerdict::Unresolved,
                    NonProvisionalVerdict::Unresolved(
                        "cover over L but B5/B3 inputs (L-cover + σ-correspondence) not supplied \
                         — awaiting solved L-coordinates"
                            .into(),
                    ),
                ),
            }
        }
        ExactifyOutcome::RecognitionFailed => (
            GroupVerdict::Unresolved,
            NonProvisionalVerdict::Unresolved(
                "exactification RecognitionFailed (raise max_deg?)".into(),
            ),
        ),
        ExactifyOutcome::SubstitutionFailed => (
            GroupVerdict::Unresolved,
            NonProvisionalVerdict::Unresolved(
                "exactification SubstitutionFailed (spurious homotopy path)".into(),
            ),
        ),
    };

    DecideReportNonProvisional {
        name: name.to_string(),
        exactify: outcome,
        m24,
        bad_locus: bad_locus_status,
        verdict,
        notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ri(n: i64) -> Rational {
        Rational::from_i64(n)
    }

    fn sample_seed() -> HomotopySeed {
        HomotopySeed {
            a: vec![ri(1), ri(-2), ri(3), ri(0), ri(-1), ri(2), ri(1), ri(-3)],
            b: vec![ri(2), ri(1), ri(-1), ri(3), ri(0), ri(-2), ri(1), ri(1)],
            r: vec![ri(-1), ri(2), ri(1)],
            s: vec![ri(3), ri(-2), ri(1), ri(2)],
            lambda: Rational::new(3, 2).unwrap(),
        }
    }

    #[test]
    fn g6_mueller_gate_is_locally_empty() {
        // The gate: Müller's conic must read as anisotropic (-1,-1) at {2,∞}.
        let v = g6_mueller_gate();
        assert_eq!(v.kind, VerdictKind::LocallyEmpty);
        let report = v.value.unwrap();
        assert!(!report.has_rational_point);
        assert!(report.ramified_places.contains(&Place::Finite(2)));
        assert!(report.ramified_places.contains(&Place::Real));
        assert_eq!(report.ramified_places.len(), 2);
    }

    #[test]
    fn uniform_law_detects_common_obstruction() {
        let m2inf = || PortalConic::Obstructed(vec![Place::Real, Place::Finite(2)]);
        let all_same = vec![
            ("[2,12,5]a".into(), m2inf()),
            ("[2,12,5]b".into(), m2inf()),
            ("[2,8,10]".into(), m2inf()),
        ];
        assert_eq!(
            uniform_law(&all_same),
            UniformLaw::UniformObstruction(vec![Place::Real, Place::Finite(2)])
        );
    }

    #[test]
    fn uniform_law_realized_and_mixed_and_incomplete() {
        let obstr = PortalConic::Obstructed(vec![Place::Finite(2), Place::Real]);
        assert_eq!(
            uniform_law(&[
                ("a".into(), PortalConic::Split),
                ("b".into(), obstr.clone())
            ]),
            UniformLaw::Realized("a".into())
        );
        assert_eq!(
            uniform_law(&[
                ("a".into(), obstr.clone()),
                ("b".into(), PortalConic::Unresolved)
            ]),
            UniformLaw::Incomplete
        );
        let other = PortalConic::Obstructed(vec![Place::Finite(3), Place::Finite(5)]);
        assert_eq!(
            uniform_law(&[("a".into(), obstr), ("b".into(), other)]),
            UniformLaw::Mixed
        );
    }

    #[test]
    fn homotopy_job_shape_and_endpoints() {
        let seed = sample_seed();
        let job = assemble_2_12_5_homotopy(&seed);

        // (b) 24 solving variables, 25 parameters.
        assert_eq!(job.variables.len(), NUM_SOLVING_VARS);
        assert_eq!(job.variables, pinned::solving_var_names());
        assert_eq!(job.parameters.len(), NUM_PARAMS);
        assert_eq!(job.start_solution.len(), NUM_SOLVING_VARS);

        // start parameters equal psi(seed); target equals p_star.
        let psi = pinned::psi(&seed.a, &seed.b, &seed.r, &seed.s, &seed.lambda);
        let expected_start: Vec<ComplexDecimal> = psi.iter().map(rat_to_complex).collect();
        let expected_target: Vec<ComplexDecimal> =
            pinned::p_star().iter().map(rat_to_complex).collect();
        assert_eq!(job.start_parameters, expected_start);
        assert_eq!(job.target_parameters, expected_target);

        // one Julia equation per parameter coordinate.
        assert_eq!(job.equations_julia.len(), NUM_PARAMS);
    }

    #[test]
    fn homotopy_start_solution_is_a_zero_of_the_start_system() {
        // Cross-check the assembler's contract algebraically: with p0 = psi(seed),
        // the seed is an exact solution of F(z; p0) = 0. We verify this on the
        // exact parameter system, feeding (seed ++ p0) as the point.
        let seed = sample_seed();
        let psi = pinned::psi(&seed.a, &seed.b, &seed.r, &seed.s, &seed.lambda);
        let mut point: Vec<Rational> = Vec::new();
        point.extend(seed.a.iter().cloned());
        point.extend(seed.b.iter().cloned());
        point.extend(seed.r.iter().cloned());
        point.extend(seed.s.iter().cloned());
        point.push(seed.lambda.clone());
        point.extend(psi.iter().cloned());

        let sys = parameter_system_2_12_5();
        assert!(sys.is_exact_solution(&point));
    }

    #[test]
    fn decide_nonprovisional_is_honest_on_a_non_solution() {
        // A rational 25-vector that is NOT a zero of the pinned system exactifies
        // to SubstitutionFailed ⇒ the non-provisional gate withholds a verdict
        // (Unresolved), never asserting M24/M23.
        let coords: Vec<rustmath_numerical::homotopy::CoordinateReIm> = (0..pinned::NUM_UNKNOWNS)
            .map(|i| rustmath_numerical::homotopy::CoordinateReIm {
                re: format!("{}.0", (i as i64 % 3) + 1),
                im: "0.0".into(),
            })
            .collect();
        let sol = NumericalSolution {
            coordinates_re_im_decimal: coords,
            residual_norm_decimal: "0.0".into(),
            path_status: "candidate".into(),
        };
        let primes = [7i64, 11, 13, 17, 19, 23];
        let report =
            decide_nonprovisional("smoke", &sol, 4, &primes, &[ri(2), ri(-1)], None, None);
        assert_eq!(report.m24, GroupVerdict::Unresolved);
        assert!(!report.is_m23_q_realized());
        assert!(matches!(
            report.verdict,
            NonProvisionalVerdict::Unresolved(_)
        ));
    }

    #[test]
    fn decide_from_split_cover_refuses_false_positive() {
        // A minimal end-to-end smoke test of the decide plumbing: a rational
        // candidate solving x - 2 = 0 exactifies to a rational point ⇒ split
        // frame ⇒ Constructed conic; but without Z_C clearance the portal-conic
        // must stay Unresolved (no false Split).
        struct DummyCover;
        impl ExactBelyiCover for DummyCover {
            fn verify_identity(&self) -> Result<(), String> {
                Ok(())
            }
            fn verify_branch_locus_0_1_infty(&self) -> bool {
                true
            }
            fn verify_ramification_2_12_5(&self) -> bool {
                true
            }
            fn observed_genus(&self) -> Option<i64> {
                Some(0)
            }
            fn verify_monodromy_independent(&self) -> Option<bool> {
                Some(true)
            }
        }

        let sys = PolySystem::from_terms(1, &[vec![(vec![1], 1), (vec![0], -2)]]);
        let sol = NumericalSolution {
            coordinates_re_im_decimal: vec![rustmath_numerical::homotopy::CoordinateReIm {
                re: "2.0".into(),
                im: "0.0".into(),
            }],
            residual_norm_decimal: "0.0".into(),
            path_status: "candidate".into(),
        };
        let report = decide_from_solved_cover("smoke", &sol, &sys, 2, &DummyCover, None, None);
        // The cover verifies, but Z_C clearance is not established ⇒ Unresolved.
        assert!(report.verification.is_constructible());
        assert_eq!(report.portal_conic, PortalConic::Unresolved);
    }
}
