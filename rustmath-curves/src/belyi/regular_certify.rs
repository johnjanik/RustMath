//! Certification **bookkeeping** for the regular `Gal(R/Q(u)) ≅ M23` claim —
//! the endpoint gate of the M23/Q campaign, with an honest verdict type.
//!
//! ## The honest scope (why the verdict is an enum, never a `bool`)
//!
//! No in-tree CAS can *prove* that the Galois group of the degree-23
//! deleted-sheet resolvent over `Q(u)` is M23, let alone regularly
//! ([`super::pipeline::decide_nonprovisional`] itself defers to an external
//! OSCAR `galois_group` cross-check). What this module can honestly do:
//!
//! * **Frobenius-statistical fibre sweep** ([`certify_regular_gal_m23_over_qu`]):
//!   specialize `u → u₀` at many rational points, classify each fibre with the
//!   same machinery as [`super::audit::audit_m23_residual`], and report *exact
//!   counts* — [`RegularM23Verdict::StatisticalOnly`], never "proven".
//! * **Decisive counter-evidence**: for a *good* specialization (leading
//!   coefficient nonvanishing, fibre squarefree) the specialized group embeds
//!   into the generic group as a permutation group (the decomposition group at
//!   the place `u = u₀`; Serre, *Topics in Galois Theory*, §1). So one good
//!   fibre whose group provably is **not** contained in M23 — a non-square
//!   discriminant (`M23 ⊆ A23`), or an observed cycle type outside M23's
//!   fingerprint — refutes the regular claim:
//!   [`RegularM23Verdict::CounterEvidence`].
//! * **Thin-set honesty** (Hilbert irreducibility): the specialized group can
//!   be a *proper subgroup* of the generic group on a thin set of `u₀`. Hence a
//!   degenerate fibre (degree drop, non-squarefree, reducible) or an
//!   irreducible fibre classified as a *subgroup* of M23 (`C23`, `F23`) is
//!   never counter-evidence — those fibres are skipped or counted as
//!   consistent-but-unforced.
//! * **The external bridge** ([`emit_external_certificate_request`] /
//!   [`attach_external_certificate`]): the only path to
//!   [`RegularM23Verdict::ProvenExternally`] is parsing an explicit,
//!   well-formed confirmation artifact from an external tool (OSCAR/Sage);
//!   anything else is refused with a precise reason. The sweep itself can
//!   never produce `ProvenExternally`.
//!
//! The sweep/verdict machinery is group-agnostic (the fibre classifier is a
//! parameter), so the acceptance tests exercise the identical bookkeeping on
//! degree-4 analogues whose fibre groups were derived independently (sympy
//! `galois_group` + PARI `polgalois` agree: see the test-local oracles).

use super::audit::frobenius_types;
use super::deleted_sheet::DeletedSheetResolvent;
use rustmath_groups::transitive23::{self, Group23};
use rustmath_integers::Integer;
use rustmath_polynomials::disc::discriminant;
use rustmath_polynomials::zassenhaus;
use rustmath_polynomials::zx;
use rustmath_rationals::Rational;
use std::collections::BTreeMap;
use std::io;

/// `|M23| = 10 200 960` — required verbatim in an external confirmation.
pub const M23_ORDER: u64 = 10_200_960;

/// The exact claim string used by the emit/attach bridge.
pub const M23_REGULAR_CLAIM: &str = "M23-regular-over-Q(u)";

/// The verdict of the regular-`Gal(M23/Q(u))` certification — honest about the
/// strength of the evidence. Never a bare `bool`: Frobenius statistics over
/// specialized fibres are evidence, not proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegularM23Verdict {
    /// An external computer-algebra run (OSCAR/Sage `galois_group` over
    /// `Q(u)`) explicitly confirmed the claim; only
    /// [`attach_external_certificate`] constructs this, and it is exactly as
    /// trustworthy as the artifact it parsed.
    ProvenExternally {
        tool: String,
        artifact_path: String,
        summary: String,
    },
    /// Fibre-sweep evidence only — the labelled-statistical outcome. Exact
    /// counts; `fibres_confirmed_m23` of `fibres_tested` classifiable fibres
    /// had Frobenius data forcing M23 (the rest were consistent-but-unforced).
    StatisticalOnly {
        fibres_tested: usize,
        fibres_confirmed_m23: usize,
        primes_per_fibre: usize,
        details: String,
    },
    /// A good (non-thin-symptom) fibre whose group is decisively **not**
    /// contained in M23; by the specialization embedding this refutes the
    /// regular claim. `t0` is the specialized value of `u`.
    CounterEvidence { t0: Rational, what: String },
    /// Not enough usable evidence either way (too few usable fibres, no
    /// Frobenius data, degree mismatch, …) — with the precise reason.
    Insufficient { reason: String },
}

/// Per-fibre outcome of the sweep — the bookkeeping unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FibreStatus {
    /// A degenerate specialization (degree drop, `disc = 0`, reducible):
    /// a thin-set symptom under Hilbert irreducibility, where the fibre group
    /// may legitimately be smaller — skipped, **never** counter-evidence.
    ThinSkipped { symptom: String },
    /// Irreducible, squarefree, and the Frobenius data *forced* the target
    /// group.
    Confirmed,
    /// Irreducible and consistent with (a subgroup of) the target group, but
    /// the data did not force it — an unsaturated prime sample or a thin-set
    /// subgroup fibre; not confirmation, not counter-evidence.
    ConsistentUnforced { classified: String },
    /// A good fibre whose group is decisively not contained in the target —
    /// counter-evidence for the regular claim.
    Counter { why: String },
    /// The fibre could not be classified (no unramified primes with
    /// full-degree reduction, factorization failure, contradictory data) —
    /// counts as consumed evidence budget but proves nothing.
    Unusable { why: String },
}

/// One sweep record: the specialized `u`-value and what happened there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FibreRecord {
    pub u0: Rational,
    pub status: FibreStatus,
}

// ---------------------------------------------------------------------------
// Candidate enumeration and fibre preparation.
// ---------------------------------------------------------------------------

fn gcd_i64(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Deterministic rational specialization points spread in height and sign:
/// for height `h = |n| + d = 2, 3, …` yield every reduced `±n/d`
/// (`1, −1, 2, −2, 1/2, −1/2, 3, −3, 1/3, −1/3, 4, −4, 3/2, …`).
fn candidate_u0s(count: usize) -> Vec<Rational> {
    let mut out = Vec::with_capacity(count + 8);
    let mut h: i64 = 2;
    while out.len() < count {
        for d in 1..h {
            let n = h - d;
            if gcd_i64(n, d) == 1 {
                out.push(Rational::new(n, d).expect("d >= 1"));
                out.push(Rational::new(-n, d).expect("d >= 1"));
            }
        }
        h += 1;
    }
    out.truncate(count);
    out
}

/// Clear an ascending rational polynomial to its primitive integer form
/// (positive leading coefficient, content 1, trailing zeros trimmed) — the
/// same normalization [`super::audit`] uses, so cross-route comparisons are
/// exact equalities.
fn clear_primitive(f: &[Rational]) -> Vec<Integer> {
    let mut den = Integer::one();
    for c in f {
        den = den.lcm(c.denominator());
    }
    let ints: Vec<Integer> = f
        .iter()
        .map(|c| {
            let scale = den.clone() / c.denominator().clone(); // exact: den is a multiple
            c.numerator().clone() * scale
        })
        .collect();
    zx::normalize(&ints)
}

// ---------------------------------------------------------------------------
// The sweep — group-agnostic bookkeeping over specialized fibres.
// ---------------------------------------------------------------------------

/// Sweep rational specializations of `resolvent`, recording a [`FibreRecord`]
/// per candidate until `fibres` non-thin fibres have been examined (or a
/// bounded candidate pool is exhausted). `classify_fibre` sees only primitive
/// integer fibre polynomials that are full-degree, squarefree, and
/// irreducible; degenerate specializations are recorded as
/// [`FibreStatus::ThinSkipped`] without consuming the fibre budget.
///
/// A [`FibreStatus::Counter`] stops the sweep immediately: by the
/// specialization embedding one decisive good fibre already settles the
/// verdict, and the remaining budget would only repeat the refutation.
pub fn sweep_resolvent_fibres(
    resolvent: &DeletedSheetResolvent,
    fibres: usize,
    classify_fibre: &dyn Fn(&[Integer]) -> FibreStatus,
) -> Vec<FibreRecord> {
    let deg = resolvent.deg_x() as i64;
    let pool = candidate_u0s(4 * fibres + 24);
    let mut records: Vec<FibreRecord> = Vec::new();
    let mut examined = 0usize;

    for u0 in pool {
        if examined >= fibres {
            break;
        }
        let g = clear_primitive(&resolvent.specialize(&u0));
        if zx::degree(&g) != deg {
            records.push(FibreRecord {
                u0,
                status: FibreStatus::ThinSkipped {
                    symptom: format!(
                        "degree drop to {} (leading X-coefficient vanishes at this u0)",
                        zx::degree(&g)
                    ),
                },
            });
            continue;
        }
        let status = match zassenhaus::factor(&g) {
            Err(()) => {
                examined += 1;
                FibreStatus::Unusable {
                    why: "Zassenhaus factorization failed on the fibre polynomial".into(),
                }
            }
            Ok((_, factors)) => {
                if factors.iter().any(|(_, m)| *m > 1) {
                    FibreStatus::ThinSkipped {
                        symptom: "not squarefree (disc = 0): u0 meets a branch/critical value"
                            .into(),
                    }
                } else if factors.len() != 1 {
                    FibreStatus::ThinSkipped {
                        symptom: format!(
                            "reducible ({} irreducible factors): thin-set specialization — \
                             the fibre group may legitimately be smaller, NOT counter-evidence",
                            factors.len()
                        ),
                    }
                } else {
                    examined += 1;
                    classify_fibre(&g)
                }
            }
        };
        let decisive = matches!(status, FibreStatus::Counter { .. });
        records.push(FibreRecord { u0, status });
        if decisive {
            break;
        }
    }
    records
}

/// Map a degree-23 fibre's exact disc-square bit and observed Frobenius cycle
/// types to a [`FibreStatus`], with the thin-set distinction built in:
///
/// * non-square disc ⇒ `Gal ⊄ A23 ⊇ M23` — decisive [`FibreStatus::Counter`];
/// * `classify = M23` ⇒ [`FibreStatus::Confirmed`];
/// * `classify ∈ {C23, F23}` (both subgroups of M23: `C23 ⊂ F23 = N_{M23}(P23)`)
///   ⇒ [`FibreStatus::ConsistentUnforced`] — a thin-set fibre or an
///   unsaturated prime sample, never counter-evidence;
/// * `classify = A23` ⇒ an observed type outside M23's fingerprint (Frobenius
///   types are genuine group elements) — decisive [`FibreStatus::Counter`];
/// * contradictory or empty data ⇒ [`FibreStatus::Unusable`].
fn status_from_deg23_types(observed: &[Vec<usize>], disc_is_square: bool) -> FibreStatus {
    if !disc_is_square {
        return FibreStatus::Counter {
            why: "disc(fibre) is not a perfect square, so Gal(fibre/Q) ⊄ A23; \
                  every subgroup of M23 lies in A23 — decisive exclusion of M23"
                .into(),
        };
    }
    if observed.is_empty() {
        return FibreStatus::Unusable {
            why: "no offered prime was unramified with full-degree reduction — \
                  no Frobenius data to classify"
                .into(),
        };
    }
    match transitive23::classify(observed, true) {
        Some(Group23::M23) => FibreStatus::Confirmed,
        Some(g @ (Group23::C23 | Group23::F23)) => FibreStatus::ConsistentUnforced {
            classified: g.label().to_string(),
        },
        Some(Group23::A23) => FibreStatus::Counter {
            why: "an observed Frobenius cycle type lies outside M23's fingerprint \
                  (classification forces A23 ⊋ M23) — decisive exclusion of M23"
                .into(),
        },
        Some(other) => FibreStatus::Unusable {
            why: format!(
                "internal inconsistency: odd-chain group {} classified despite a square \
                 discriminant",
                other.label()
            ),
        },
        None => FibreStatus::Unusable {
            why: "contradictory Frobenius data: square discriminant but an odd cycle \
                  type observed"
                .into(),
        },
    }
}

/// The degree-23 fibre classifier — the exact machinery of
/// [`super::audit::audit_m23_residual`] (disc-square bit, [`frobenius_types`],
/// [`transitive23::classify`]) restated per fibre.
fn classify_fibre_m23(g: &[Integer], primes: &[i64]) -> FibreStatus {
    let disc_is_square = discriminant(g).is_perfect_square();
    if !disc_is_square {
        return status_from_deg23_types(&[], false);
    }
    let observed = frobenius_types(g, primes, 23);
    status_from_deg23_types(&observed, true)
}

fn describe_status(s: &FibreStatus) -> String {
    match s {
        FibreStatus::ThinSkipped { symptom } => format!("thin-skipped ({symptom})"),
        FibreStatus::Confirmed => "confirmed".into(),
        FibreStatus::ConsistentUnforced { classified } => {
            format!("consistent-unforced (classified {classified})")
        }
        FibreStatus::Counter { why } => format!("COUNTER ({why})"),
        FibreStatus::Unusable { why } => format!("unusable ({why})"),
    }
}

/// Assemble the honest verdict from sweep records:
///
/// * any [`FibreStatus::Counter`] ⇒ [`RegularM23Verdict::CounterEvidence`]
///   (one decisive good fibre outweighs any number of confirmations — the
///   generic group then strictly contains M23 or misses it);
/// * otherwise, ≥ 1 confirmed fibre ⇒ [`RegularM23Verdict::StatisticalOnly`]
///   with exact counts (`fibres_tested` counts classifiable fibres:
///   confirmed + consistent-unforced) — **never** a proven claim;
/// * otherwise ⇒ [`RegularM23Verdict::Insufficient`] with the per-fibre log.
pub fn verdict_from_fibre_records(
    records: &[FibreRecord],
    primes_per_fibre: usize,
) -> RegularM23Verdict {
    let details = records
        .iter()
        .map(|r| format!("u0={}: {}", r.u0, describe_status(&r.status)))
        .collect::<Vec<_>>()
        .join("; ");

    for r in records {
        if let FibreStatus::Counter { why } = &r.status {
            return RegularM23Verdict::CounterEvidence {
                t0: r.u0.clone(),
                what: format!(
                    "good specialization at u0 = {}: {} [sweep log: {}]",
                    r.u0, why, details
                ),
            };
        }
    }
    let confirmed = records
        .iter()
        .filter(|r| matches!(r.status, FibreStatus::Confirmed))
        .count();
    let tested = confirmed
        + records
            .iter()
            .filter(|r| matches!(r.status, FibreStatus::ConsistentUnforced { .. }))
            .count();
    if confirmed >= 1 {
        RegularM23Verdict::StatisticalOnly {
            fibres_tested: tested,
            fibres_confirmed_m23: confirmed,
            primes_per_fibre,
            details,
        }
    } else {
        RegularM23Verdict::Insufficient {
            reason: format!(
                "no fibre's Frobenius data forced the target group ({} classifiable \
                 fibre(s), 0 confirmed): {}",
                tested,
                if details.is_empty() {
                    "no candidates examined".to_string()
                } else {
                    details
                }
            ),
        }
    }
}

/// The endpoint gate: sweep `fibres` non-degenerate rational specializations
/// of the degree-23 deleted-sheet resolvent, classify each with the
/// [`super::audit`] machinery over the offered `primes`, and return the
/// honest bookkeeping verdict.
///
/// This function **cannot** return [`RegularM23Verdict::ProvenExternally`]:
/// Frobenius statistics never prove `Gal(R/Q(u)) = M23`. The proof path is
/// [`emit_external_certificate_request`] → external OSCAR/Sage run →
/// [`attach_external_certificate`].
///
/// `primes_per_fibre` in the verdict is the number of primes *offered*; per
/// fibre, ramified primes and leading-coefficient drops are screened out by
/// [`frobenius_types`] exactly as in `audit`.
pub fn certify_regular_gal_m23_over_qu(
    resolvent: &DeletedSheetResolvent,
    fibres: usize,
    primes: &[i64],
) -> RegularM23Verdict {
    if resolvent.deg_x() != 23 {
        return RegularM23Verdict::Insufficient {
            reason: format!(
                "resolvent has deg_X = {}; the regular Gal(M23/Q(u)) statement is about \
                 the degree-23 deleted-sheet resolvent of a degree-24 map",
                resolvent.deg_x()
            ),
        };
    }
    if fibres == 0 {
        return RegularM23Verdict::Insufficient {
            reason: "fibres = 0 requested: no evidence gathered".into(),
        };
    }
    let records = sweep_resolvent_fibres(resolvent, fibres, &|g| classify_fibre_m23(g, primes));
    verdict_from_fibre_records(&records, primes.len())
}

// ---------------------------------------------------------------------------
// The external bridge: emit a request OSCAR/Sage can consume, attach a
// completed run. The ONLY path to ProvenExternally.
// ---------------------------------------------------------------------------

fn fmt_rat(r: &Rational) -> String {
    if r.denominator() == &Integer::one() {
        format!("{}", r.numerator())
    } else {
        format!("{}/{}", r.numerator(), r.denominator())
    }
}

/// The canonical `coeff_x` block — the single serialization shared by
/// [`emit_external_certificate_request`] and [`resolvent_fingerprint`], so the
/// fingerprint is by construction a hash of exactly what the external CAS saw.
fn coeff_rows(resolvent: &DeletedSheetResolvent) -> String {
    let mut out = String::new();
    for (i, c) in resolvent.coeffs().iter().enumerate() {
        let row = if c.is_zero() {
            "0".to_string()
        } else {
            c.coefficients()
                .iter()
                .map(fmt_rat)
                .collect::<Vec<_>>()
                .join(" ")
        };
        out.push_str(&format!("coeff_x {i}: {row}\n"));
    }
    out
}

/// A stable fingerprint of the resolvent: FNV-1a (64-bit) over the canonical
/// `coeff_x` rows, rendered as 16 lowercase hex digits. FNV-1a is implemented
/// inline (not `DefaultHasher`, whose output is unstable across Rust releases)
/// so a certificate remains attachable across toolchain upgrades. This is an
/// integrity binder, not a cryptographic commitment — it stops the real
/// failure mode (attaching a confirmation issued for a DIFFERENT resolvent),
/// not a deliberate forgery, which no offline hash could stop anyway.
pub fn resolvent_fingerprint(resolvent: &DeletedSheetResolvent) -> String {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = FNV_OFFSET;
    for b in coeff_rows(resolvent).as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    format!("{h:016x}")
}

/// Write the resolvent and verification instructions to `path` in a small
/// text format an external CAS operator can consume directly. The embedded
/// OSCAR snippet computes `galois_group` over `Q(u)`; the response format
/// required by [`attach_external_certificate`] is spelled out verbatim.
pub fn emit_external_certificate_request(
    resolvent: &DeletedSheetResolvent,
    path: &str,
) -> io::Result<()> {
    let mut out = String::new();
    out.push_str("# RustMath external certificate REQUEST v1\n");
    out.push_str(&format!("# claim: {M23_REGULAR_CLAIM}\n"));
    out.push_str(&format!(
        "# fingerprint: {}\n",
        resolvent_fingerprint(resolvent)
    ));
    out.push_str(
        "# Verify: the Galois group of R(X, u) below over Q(u) is the Mathieu group M23\n\
         # (order 10200960), and the extension is regular over Q (the constant field of\n\
         # the splitting field is Q). Since M23 is simple, Gal(R/Q(u)) = M23 plus a\n\
         # nontrivial geometric monodromy group already forces regularity: the geometric\n\
         # group is a nontrivial normal subgroup, hence all of M23, hence the constant\n\
         # field extension Gal(R/Q(u))/Gal(R/QQbar(u)) is trivial.\n",
    );
    out.push_str(&format!("# deg_X: {}\n", resolvent.deg_x()));
    out.push_str(&format!("# deg_u: {}\n", resolvent.deg_u()));
    out.push_str(
        "# encoding: line `coeff_x <i>: c0 c1 c2 ...` gives [X^i] R as a polynomial in u,\n\
         # ascending powers of u, each c a rational `n` or `n/d`.\n\
         #\n\
         # OSCAR (Julia) verification sketch:\n\
         #   using Oscar\n\
         #   Qu, u = rational_function_field(QQ, \"u\")\n\
         #   Rx, X = polynomial_ring(Qu, \"X\")\n\
         #   R = sum(c_i(u) * X^i for the rows below)\n\
         #   G, C = galois_group(R)\n\
         #   describe(G), order(G)   # expect M 23, 10200960\n\
         #\n\
         # Response file format required by attach_external_certificate (key: value\n\
         # lines; '#' comments ignored; every key required, each exactly once):\n\
         #   tool: <name and version of the CAS that ran the check>\n",
    );
    out.push_str(&format!("#   claim: {M23_REGULAR_CLAIM}\n"));
    out.push_str(&format!(
        "#   fingerprint: <the fingerprint from this request, echoed verbatim —\n\
         #                 it binds the certificate to THIS resolvent>\n\
         #   verdict: CONFIRMED\n\
         #   group_order: {M23_ORDER}\n\
         #   summary: <what was computed, including the regularity argument>\n",
    ));
    out.push_str(&coeff_rows(resolvent));
    std::fs::write(path, out)
}

/// Parse a completed external run into
/// [`RegularM23Verdict::ProvenExternally`], refusing (`Err`, with the precise
/// reason) anything that is not an explicit, unambiguous M23-regular
/// confirmation **for this resolvent**: required keys `tool`, `claim`,
/// `fingerprint`, `verdict`, `group_order`, `summary`, each exactly once;
/// `claim` must equal [`M23_REGULAR_CLAIM`], `fingerprint` must equal
/// [`resolvent_fingerprint`]`(resolvent)` (a certificate issued for a
/// different resolvent must not attach — the adversarial audit flagged
/// exactly this hole), `verdict` must be `CONFIRMED`, and `group_order`
/// must be `|M23| = 10200960`.
pub fn attach_external_certificate(
    resolvent: &DeletedSheetResolvent,
    verdict_path: &str,
) -> Result<RegularM23Verdict, String> {
    let text = std::fs::read_to_string(verdict_path)
        .map_err(|e| format!("cannot read external certificate {verdict_path}: {e}"))?;
    let mut kv: BTreeMap<String, String> = BTreeMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (k, v) = line
            .split_once(':')
            .ok_or_else(|| format!("unparseable line (expected `key: value`): {line:?}"))?;
        let (k, v) = (k.trim().to_string(), v.trim().to_string());
        if kv.contains_key(&k) {
            return Err(format!(
                "ambiguous certificate: key {k:?} appears more than once — refusing"
            ));
        }
        kv.insert(k, v);
    }
    let get = |k: &str| -> Result<&String, String> {
        kv.get(k)
            .ok_or_else(|| format!("missing required key {k:?} in external certificate"))
    };
    let tool = get("tool")?;
    if tool.is_empty() {
        return Err("empty `tool` in external certificate".into());
    }
    let claim = get("claim")?;
    if claim != M23_REGULAR_CLAIM {
        return Err(format!(
            "claim mismatch: certificate says {claim:?}, this gate certifies only \
             {M23_REGULAR_CLAIM:?}"
        ));
    }
    let expected_fp = resolvent_fingerprint(resolvent);
    let fp = get("fingerprint")?;
    if *fp != expected_fp {
        return Err(format!(
            "fingerprint mismatch: certificate is bound to {fp:?}, this resolvent is \
             {expected_fp:?} — a confirmation of a DIFFERENT resolvent is not a \
             confirmation of this one"
        ));
    }
    let verdict = get("verdict")?;
    if verdict != "CONFIRMED" {
        return Err(format!(
            "external verdict is {verdict:?}, not CONFIRMED — refusing to attach"
        ));
    }
    let order = get("group_order")?;
    if *order != M23_ORDER.to_string() {
        return Err(format!(
            "group_order mismatch: certificate says {order:?}, |M23| = {M23_ORDER} — \
             refusing (a confirmation of some other group is not an M23 confirmation)"
        ));
    }
    let summary = get("summary")?;
    if summary.is_empty() {
        return Err("empty `summary` in external certificate".into());
    }
    Ok(RegularM23Verdict::ProvenExternally {
        tool: tool.clone(),
        artifact_path: verdict_path.to_string(),
        summary: summary.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belyi::deleted_sheet::deleted_sheet_resolvent;

    fn ri(n: i64) -> Rational {
        Rational::from_i64(n)
    }
    fn rq(n: i64, d: i64) -> Rational {
        Rational::new(n, d).unwrap()
    }
    fn rvec(v: &[i64]) -> Vec<Rational> {
        v.iter().map(|&n| ri(n)).collect()
    }

    /// R(X, u) = (P(X) − P(u))/(X − u) for a polynomial map φ = P (Q = 1).
    fn poly_family(p: &[i64]) -> DeletedSheetResolvent {
        deleted_sheet_resolvent(&rvec(p), &rvec(&[1])).unwrap()
    }

    // ------------------------------------------------------------------
    // Degree-4 analogue classifiers, mirroring the transitive23 shape.
    // Transitive quartic groups and their cycle-type sets (textbook; the
    // per-fibre groups below were independently cross-checked with sympy
    // galois_group AND PARI polgalois — see the derivation notes):
    //   even chain (⊆ A4):  V4 {1⁴, 2²} ⊂ A4 {1⁴, 2², 1·3}
    //   odd chain:          C4 {1⁴, 2², 4} ⊂ D4 (+1²·2) ⊂ S4 (all five)
    // ------------------------------------------------------------------
    fn classify4(observed: &[Vec<usize>], disc_is_square: bool) -> Option<&'static str> {
        let v4: &[&[usize]] = &[&[1, 1, 1, 1], &[2, 2]];
        let a4: &[&[usize]] = &[&[1, 1, 1, 1], &[2, 2], &[1, 3]];
        let c4: &[&[usize]] = &[&[1, 1, 1, 1], &[2, 2], &[4]];
        let d4: &[&[usize]] = &[&[1, 1, 1, 1], &[1, 1, 2], &[2, 2], &[4]];
        let s4: &[&[usize]] = &[&[1, 1, 1, 1], &[1, 1, 2], &[1, 3], &[2, 2], &[4]];
        let chain: &[(&str, &[&[usize]])] = if disc_is_square {
            &[("V4", v4), ("A4", a4)]
        } else {
            &[("C4", c4), ("D4", d4), ("S4", s4)]
        };
        chain
            .iter()
            .find(|(_, set)| {
                observed
                    .iter()
                    .all(|t| set.iter().any(|s| s == &t.as_slice()))
            })
            .map(|(name, _)| *name)
    }

    /// Target S4 (odd chain, like S23): forced only when classify4 pins S4;
    /// a square disc excludes S4 decisively (Gal ⊆ A4).
    fn classify_fibre_s4(g: &[Integer], primes: &[i64]) -> FibreStatus {
        let disc_is_square = discriminant(g).is_perfect_square();
        if disc_is_square {
            return FibreStatus::Counter {
                why: "square disc: Gal ⊆ A4, decisively not S4".into(),
            };
        }
        let observed = frobenius_types(g, primes, 4);
        if observed.is_empty() {
            return FibreStatus::Unusable {
                why: "no Frobenius data".into(),
            };
        }
        match classify4(&observed, false) {
            Some("S4") => FibreStatus::Confirmed,
            Some(smaller) => FibreStatus::ConsistentUnforced {
                classified: smaller.into(),
            },
            None => FibreStatus::Unusable {
                why: "contradictory quartic data".into(),
            },
        }
    }

    /// Target A4 (even chain, the structural mirror of M23 ⊆ A23): a
    /// non-square disc excludes it decisively.
    fn classify_fibre_a4(g: &[Integer], primes: &[i64]) -> FibreStatus {
        let disc_is_square = discriminant(g).is_perfect_square();
        if !disc_is_square {
            return FibreStatus::Counter {
                why: "non-square disc: Gal ⊄ A4, decisively not A4".into(),
            };
        }
        let observed = frobenius_types(g, primes, 4);
        if observed.is_empty() {
            return FibreStatus::Unusable {
                why: "no Frobenius data".into(),
            };
        }
        match classify4(&observed, true) {
            Some("A4") => FibreStatus::Confirmed,
            Some(smaller) => FibreStatus::ConsistentUnforced {
                classified: smaller.into(),
            },
            None => FibreStatus::Unusable {
                why: "contradictory quartic data".into(),
            },
        }
    }

    // ------------------------------------------------------------------
    // Deg-4 analogue, family A: P = x⁵ − x − 1, so
    // R(X, u) = X⁴ + uX³ + u²X² + u³X + (u⁴ − 1).
    // Independent oracle (sympy galois_group + PARI polgalois agree):
    //   u0 = 1, −1: fibre reducible (X⁵ − X splits) — thin-skipped;
    //   u0 = 2, −2, 1/2: irreducible, disc non-square, group S4, and mod-p
    //   factorization types over {3,5,7,11,13} include 1·3 (p = 7), so the
    //   odd-chain classification forces S4 → Confirmed.
    // ------------------------------------------------------------------
    #[test]
    fn deg4_sweep_confirms_s4_family_with_exact_bookkeeping() {
        let r = poly_family(&[-1, -1, 0, 0, 0, 1]);
        assert_eq!(r.deg_x(), 4);
        let primes = [3i64, 5, 7, 11, 13];
        let records = sweep_resolvent_fibres(&r, 3, &|g| classify_fibre_s4(g, &primes));

        assert_eq!(records.len(), 5, "1, −1 thin-skipped, then 3 examined fibres");
        assert_eq!(records[0].u0, ri(1));
        assert!(
            matches!(&records[0].status, FibreStatus::ThinSkipped { symptom } if symptom.contains("reducible")),
            "u0=1: X(X+1)(X²+1) — got {:?}",
            records[0].status
        );
        assert_eq!(records[1].u0, ri(-1));
        assert!(matches!(
            &records[1].status,
            FibreStatus::ThinSkipped { symptom } if symptom.contains("reducible")
        ));
        for (rec, u0) in records[2..].iter().zip([ri(2), ri(-2), rq(1, 2)]) {
            assert_eq!(rec.u0, u0);
            assert_eq!(
                rec.status,
                FibreStatus::Confirmed,
                "sympy+PARI oracle: S4 forced at u0={u0}"
            );
        }

        let v = verdict_from_fibre_records(&records, primes.len());
        assert_eq!(
            v,
            RegularM23Verdict::StatisticalOnly {
                fibres_tested: 3,
                fibres_confirmed_m23: 3,
                primes_per_fibre: 5,
                details: match &v {
                    RegularM23Verdict::StatisticalOnly { details, .. } => details.clone(),
                    _ => String::new(),
                },
            },
            "exact counts, statistical labelling — never a proven claim"
        );
    }

    // ------------------------------------------------------------------
    // Deg-4 analogue, family B: P = x⁵, so R(X, u) = X⁴ + uX³ + u²X² + u³X + u⁴
    // (the scaled Φ₅ family). Oracle: every fibre at u0 ≠ 0 has group C4,
    // disc = 125·u0¹² (non-square). Against target S4 the data is consistent
    // but never forcing → 0 confirmed → honest Insufficient, NOT counter.
    // ------------------------------------------------------------------
    #[test]
    fn deg4_sweep_is_honest_when_nothing_forces_the_target() {
        let r = poly_family(&[0, 0, 0, 0, 0, 1]);
        // p = 5 is ramified for the Φ₅ family and screened automatically.
        let primes = [3i64, 7, 11, 19];
        let records = sweep_resolvent_fibres(&r, 2, &|g| classify_fibre_s4(g, &primes));
        assert_eq!(records.len(), 2);
        for (rec, u0) in records.iter().zip([ri(1), ri(-1)]) {
            assert_eq!(rec.u0, u0);
            assert_eq!(
                rec.status,
                FibreStatus::ConsistentUnforced {
                    classified: "C4".into()
                },
                "oracle: Φ₅-fibre types {{1⁴, 2², 4}} classify to C4"
            );
        }
        let v = verdict_from_fibre_records(&records, primes.len());
        assert!(
            matches!(&v, RegularM23Verdict::Insufficient { reason } if reason.contains("0 confirmed")),
            "no forced fibre ⇒ Insufficient, got {v:?}"
        );
    }

    // ------------------------------------------------------------------
    // Deg-4/5 analogue, family C: P = x⁶ — R(X, u) = (X⁶ − u⁶)/(X − u) factors
    // as (X + u)(X² + uX + u²)(X² − uX + u²) for every u0, so ALL fibres are
    // thin symptoms: reducibility is never counter-evidence, and the sweep
    // exhausts its pool → Insufficient.
    // ------------------------------------------------------------------
    #[test]
    fn all_reducible_family_yields_insufficient_never_counter() {
        let r = poly_family(&[0, 0, 0, 0, 0, 0, 1]);
        assert_eq!(r.deg_x(), 5);
        let primes = [7i64, 11];
        let records = sweep_resolvent_fibres(&r, 1, &|g| classify_fibre_s4(g, &primes));
        assert!(!records.is_empty());
        for rec in &records {
            assert!(
                matches!(&rec.status, FibreStatus::ThinSkipped { symptom } if symptom.contains("reducible")),
                "u0={}: every specialization splits 1+2+2 — got {:?}",
                rec.u0,
                rec.status
            );
        }
        let v = verdict_from_fibre_records(&records, primes.len());
        assert!(matches!(v, RegularM23Verdict::Insufficient { .. }));
    }

    // ------------------------------------------------------------------
    // Counter-evidence arm on the deg-4 analogue: family A against target A4.
    // Oracle: disc(fibre at u0 = 2) = 405584, not a square ⇒ Gal ⊄ A4 —
    // decisive. Thin fibres before it must be skipped, and the sweep stops on
    // the first counter.
    // ------------------------------------------------------------------
    #[test]
    fn deg4_sweep_detects_counter_evidence_and_stops() {
        let r = poly_family(&[-1, -1, 0, 0, 0, 1]);
        let primes = [3i64, 5, 7];
        let records = sweep_resolvent_fibres(&r, 4, &|g| classify_fibre_a4(g, &primes));
        assert_eq!(records.len(), 3, "1, −1 thin, then the counter stops the sweep");
        assert!(matches!(records[0].status, FibreStatus::ThinSkipped { .. }));
        assert!(matches!(records[1].status, FibreStatus::ThinSkipped { .. }));
        assert_eq!(records[2].u0, ri(2));
        assert!(matches!(records[2].status, FibreStatus::Counter { .. }));

        let v = verdict_from_fibre_records(&records, primes.len());
        match v {
            RegularM23Verdict::CounterEvidence { t0, what } => {
                assert_eq!(t0, ri(2));
                assert!(what.contains("disc"), "the reason names the disc bit: {what}");
            }
            other => panic!("expected CounterEvidence, got {other:?}"),
        }
    }

    // ------------------------------------------------------------------
    // The deg-23 status mapping (the M23-specific thin-set logic), fed
    // synthetic classification inputs.
    // ------------------------------------------------------------------
    #[test]
    fn deg23_status_mapping_is_honest() {
        // Full M23 fingerprint + square disc ⇒ Confirmed.
        let full: Vec<Vec<usize>> = Group23::M23.type_set().unwrap().into_iter().collect();
        assert_eq!(status_from_deg23_types(&full, true), FibreStatus::Confirmed);

        // Only {1²³, 23¹} ⇒ classify C23 ⊂ M23: consistent, NOT forced, NOT counter.
        let weak = vec![vec![1usize; 23], vec![23usize]];
        assert!(matches!(
            status_from_deg23_types(&weak, true),
            FibreStatus::ConsistentUnforced { .. }
        ));

        // An even type outside M23's fingerprint (a plain 3-cycle: 3·1²⁰;
        // M23's 3A is 3⁶1⁵) forces A23 ⇒ decisive counter.
        let mut three_cycle = vec![1usize; 20];
        three_cycle.push(3);
        three_cycle.sort_unstable();
        assert!(matches!(
            status_from_deg23_types(&[three_cycle], true),
            FibreStatus::Counter { .. }
        ));

        // Non-square disc ⇒ counter regardless of types.
        assert!(matches!(
            status_from_deg23_types(&full, false),
            FibreStatus::Counter { .. }
        ));

        // Square disc + an odd type (2·1²¹) is contradictory ⇒ Unusable.
        let mut odd = vec![1usize; 21];
        odd.push(2);
        odd.sort_unstable();
        assert!(matches!(
            status_from_deg23_types(&[odd], true),
            FibreStatus::Unusable { .. }
        ));

        // No data ⇒ Unusable, never a classification.
        assert!(matches!(
            status_from_deg23_types(&[], true),
            FibreStatus::Unusable { .. }
        ));
    }

    // ------------------------------------------------------------------
    // The real sweep shape: deg-24 synthetic (non-solution) Belyi map, same
    // sample 25-vector as audit/deleted_sheet tests. Independent sympy oracle
    // for the first candidates:
    //   u0 = 1:  R(X,1) = −P(1)·Q(X)/(X−1), shape (x−1)⁴·cubic⁵·S — disc 0;
    //   u0 = −1: t0 = φ(−1) = 0 (B(−1) = 0), fibre = A²·(B/(X+1)) — disc 0;
    //   u0 = 2:  irreducible degree 23, disc NOT a perfect square.
    // So the honest verdict is CounterEvidence at u0 = 2 — CORRECT, because
    // this synthetic vector is not the M24 cover; the sweep detects the
    // impostor exactly as designed. (The two disc-0 fibres are thin-skipped,
    // never counter-evidence.) Nothing here can be "proven".
    // ------------------------------------------------------------------
    fn rat_mul(a: &[Rational], b: &[Rational]) -> Vec<Rational> {
        let mut c = vec![ri(0); a.len() + b.len() - 1];
        for (i, ai) in a.iter().enumerate() {
            for (j, bj) in b.iter().enumerate() {
                c[i + j] = c[i + j].clone() + ai.clone() * bj.clone();
            }
        }
        c
    }
    fn rat_pow(a: &[Rational], e: u32) -> Vec<Rational> {
        let mut acc = vec![ri(1)];
        for _ in 0..e {
            acc = rat_mul(&acc, a);
        }
        acc
    }

    /// The audit/deleted_sheet sample 25-vector (a non-solution), and the
    /// degree-24 P = A²B, Q = λR⁵S it encodes.
    fn synthetic_deg24_resolvent() -> DeletedSheetResolvent {
        let one = ri(1);
        let mut a = rvec(&[1, -2, 3, 0, -1, 2, 1, -3]);
        a.push(one.clone());
        let mut b = rvec(&[2, 1, -1, 3, 0, -2, 1, 1]);
        b.push(one.clone());
        let cubic = vec![ri(-1), ri(2), ri(1), one.clone()];
        let x_minus_1 = vec![ri(-1), one.clone()];
        let r4 = rat_mul(&x_minus_1, &cubic);
        let mut s = rvec(&[3, -2, 1, 2]);
        s.push(one);
        let p = rat_mul(&rat_pow(&a, 2), &b);
        let q: Vec<Rational> = rat_mul(&rat_pow(&r4, 5), &s)
            .iter()
            .map(|c| c.clone() * rq(3, 2))
            .collect();
        deleted_sheet_resolvent(&p, &q).unwrap()
    }

    #[test]
    fn deg24_synthetic_sweep_honestly_refutes_the_impostor() {
        let resolvent = synthetic_deg24_resolvent();
        assert_eq!(resolvent.deg_x(), 23);
        let v = certify_regular_gal_m23_over_qu(&resolvent, 1, &[5, 7, 11, 13]);
        match &v {
            RegularM23Verdict::CounterEvidence { t0, what } => {
                assert_eq!(*t0, ri(2), "first good fibre is u0 = 2 (1 and −1 are disc-0 thin)");
                assert!(
                    what.contains("perfect square"),
                    "the sympy-verified non-square disc is the stated reason: {what}"
                );
                assert!(
                    what.contains("thin-skipped") && what.contains("not squarefree"),
                    "the sweep log records the two thin fibres: {what}"
                );
            }
            other => panic!("sympy oracle says CounterEvidence at u0=2, got {other:?}"),
        }
        // The bookkeeping can never launder statistics into a proof.
        assert!(!matches!(v, RegularM23Verdict::ProvenExternally { .. }));
        assert!(!matches!(v, RegularM23Verdict::StatisticalOnly { .. }));
    }

    #[test]
    fn certify_guards_degree_and_zero_budget() {
        let quartic_family = poly_family(&[-1, -1, 0, 0, 0, 1]);
        assert!(matches!(
            certify_regular_gal_m23_over_qu(&quartic_family, 2, &[5, 7]),
            RegularM23Verdict::Insufficient { reason } if reason.contains("deg_X = 4")
        ));
        let resolvent = synthetic_deg24_resolvent();
        assert!(matches!(
            certify_regular_gal_m23_over_qu(&resolvent, 0, &[5, 7]),
            RegularM23Verdict::Insufficient { reason } if reason.contains("fibres = 0")
        ));
    }

    // ------------------------------------------------------------------
    // The external bridge: emit is parseable + complete; attach accepts only
    // an explicit, unambiguous M23-regular confirmation.
    // ------------------------------------------------------------------
    /// Per-test temp dir, removed on drop — the first draft's shared pid-keyed
    /// dir leaked artifacts into /tmp (audit finding) and a shared dir cannot
    /// be safely removed while tests run in parallel.
    struct TmpDir(std::path::PathBuf);
    impl TmpDir {
        fn new(test: &str) -> Self {
            let dir = std::env::temp_dir()
                .join(format!("regular_certify_{}_{test}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            TmpDir(dir)
        }
        fn path(&self, name: &str) -> String {
            self.0.join(name).to_str().unwrap().to_string()
        }
    }
    impl Drop for TmpDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn emit_request_contains_claim_snippet_and_all_rows() {
        let tmp = TmpDir::new("emit");
        let resolvent = synthetic_deg24_resolvent();
        let path = tmp.path("request.txt");
        emit_external_certificate_request(&resolvent, &path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains(&format!("# claim: {M23_REGULAR_CLAIM}")));
        assert!(text.contains(&format!(
            "# fingerprint: {}",
            resolvent_fingerprint(&resolvent)
        )));
        assert!(text.contains("galois_group"), "the OSCAR one-liner is embedded");
        assert!(text.contains("rational_function_field(QQ"));
        for i in 0..=23 {
            assert!(
                text.contains(&format!("coeff_x {i}: ")),
                "row for X^{i} present"
            );
        }
        // Spot-check one exactly-known coefficient row: [X²³] R = lc(P)·Q(u) − lc(Q)·P(u)
        // has constant term 1·q₀ − (3/2·(−1)⁵·(−1)·3)·p₀ ... verified structurally:
        // the row must start with a rational and have deg_u + 1 = 24 entries.
        let row23 = text
            .lines()
            .find(|l| l.starts_with("coeff_x 23: "))
            .unwrap();
        assert_eq!(row23.split_whitespace().count(), 2 + 24);
    }

    #[test]
    fn attach_accepts_only_an_explicit_confirmation() {
        let tmp = TmpDir::new("attach");
        let resolvent = synthetic_deg24_resolvent();
        let fp = resolvent_fingerprint(&resolvent);
        // A well-formed confirmation parses to ProvenExternally.
        let good = tmp.path("confirmed.txt");
        std::fs::write(
            &good,
            format!(
                "# completed run\ntool: OSCAR 1.4 (galois_group over QQ(u))\n\
                 claim: {M23_REGULAR_CLAIM}\nfingerprint: {fp}\nverdict: CONFIRMED\n\
                 group_order: {M23_ORDER}\n\
                 summary: galois_group(R) = M23; regularity from simplicity of M23\n"
            ),
        )
        .unwrap();
        match attach_external_certificate(&resolvent, &good) {
            Ok(RegularM23Verdict::ProvenExternally {
                tool,
                artifact_path,
                summary,
            }) => {
                assert!(tool.contains("OSCAR"));
                assert_eq!(artifact_path, good);
                assert!(summary.contains("M23"));
            }
            other => panic!("expected ProvenExternally, got {other:?}"),
        }

        // A certificate issued for a DIFFERENT resolvent must not attach —
        // the exact hole both adversarial audits flagged. Same claim, same
        // group order, valid shape; only the underlying R(X,u) differs.
        let other = poly_family(&[7, 0, 0, 0, -2, 1]);
        let other_fp = resolvent_fingerprint(&other);
        assert_ne!(fp, other_fp, "distinct resolvents must fingerprint apart");

        // Every mutilation is refused with a precise reason.
        let cases: &[(&str, String, &str)] = &[
            (
                "foreign_resolvent.txt",
                format!(
                    "tool: OSCAR\nclaim: {M23_REGULAR_CLAIM}\nfingerprint: {other_fp}\n\
                     verdict: CONFIRMED\ngroup_order: {M23_ORDER}\n\
                     summary: a genuine confirmation, of the wrong resolvent\n"
                ),
                "fingerprint mismatch",
            ),
            (
                "no_fingerprint.txt",
                format!(
                    "tool: OSCAR\nclaim: {M23_REGULAR_CLAIM}\nverdict: CONFIRMED\n\
                     group_order: {M23_ORDER}\nsummary: unbound certificate\n"
                ),
                "missing required key \"fingerprint\"",
            ),
            (
                "denied.txt",
                format!(
                    "tool: OSCAR\nclaim: {M23_REGULAR_CLAIM}\nfingerprint: {fp}\n\
                     verdict: EXCLUDED\ngroup_order: {M23_ORDER}\nsummary: group was S23\n"
                ),
                "not CONFIRMED",
            ),
            (
                "wrong_order.txt",
                format!(
                    "tool: OSCAR\nclaim: {M23_REGULAR_CLAIM}\nfingerprint: {fp}\n\
                     verdict: CONFIRMED\ngroup_order: 244823040\nsummary: oops, that is M24\n"
                ),
                "group_order mismatch",
            ),
            (
                "wrong_claim.txt",
                format!(
                    "tool: OSCAR\nclaim: M24-over-Q(t)\nfingerprint: {fp}\nverdict: CONFIRMED\n\
                     group_order: {M23_ORDER}\nsummary: different statement\n"
                ),
                "claim mismatch",
            ),
            (
                "no_tool.txt",
                format!(
                    "claim: {M23_REGULAR_CLAIM}\nfingerprint: {fp}\nverdict: CONFIRMED\n\
                     group_order: {M23_ORDER}\nsummary: anonymous\n"
                ),
                "missing required key \"tool\"",
            ),
            (
                "duplicate.txt",
                format!(
                    "tool: OSCAR\nclaim: {M23_REGULAR_CLAIM}\nfingerprint: {fp}\n\
                     verdict: EXCLUDED\nverdict: CONFIRMED\ngroup_order: {M23_ORDER}\n\
                     summary: ambiguous\n"
                ),
                "more than once",
            ),
        ];
        for (name, contents, expect) in cases {
            let path = tmp.path(name);
            std::fs::write(&path, contents).unwrap();
            match attach_external_certificate(&resolvent, &path) {
                Err(e) => assert!(
                    e.contains(expect),
                    "{name}: error should mention {expect:?}, got: {e}"
                ),
                Ok(v) => panic!("{name}: must be refused, got {v:?}"),
            }
        }

        // A missing file is an Err, never a verdict.
        assert!(attach_external_certificate(&resolvent, &tmp.path("nonexistent.txt")).is_err());
    }
}
