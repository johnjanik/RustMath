//! The conic-from-cover bridge — turn a solved cover into the conic the Hasse
//! reader (D4) decides.
//!
//! Ported from `dessin_engine/src/conic_bridge.rs` in
//! `/home/john/inverse_galois/M23/dessin_engine` (S3), retyped onto the RustMath
//! foundation: the reference `Rat`/private conic types become
//! [`rustmath_rationals::Rational`] and the Wave-1
//! [`rustmath_quadraticforms::conic`] / [`rustmath_quadraticforms::ternary`]
//! types, and the exactification outcome comes from the Wave-2
//! [`rustmath_numerical::exactify::ExactifyOutcome`].
//!
//! The conic of the genus-0 subcover `X_C` is encoded in *how* the cover solved:
//! - solved over `Q` with the pinned ramification points `Q`-rational ⇒
//!   `X_C ≅ P¹_Q`, the conic **splits** (a rational point exists) and D4 reports
//!   an empty ramification set;
//! - solved only over an extension `L` ⇒ the conic is the **descent obstruction**
//!   (the class in `Br(Q)[2]` of the cocycle `σ ↦ g_σ ∈ PGL₂(L)`); routed here as
//!   a [`DescentPacket`], to be discharged by [`crate::belyi::descent`];
//! - solved over `Q` but with no rational frame ⇒ the anticanonical conic by
//!   Riemann–Roch (Hess).
//!
//! When an explicit ternary form is in hand (descent- or RR-produced, or Müller's
//! hand-descent), [`read_explicit_conic`] diagonalizes it and runs D4.

use rustmath_numerical::exactify::ExactifyOutcome;
use rustmath_quadraticforms::conic::{
    ConicBrauerReport, DiagonalConicQ, MathStatus, Verdict, VerdictKind,
};
use rustmath_quadraticforms::ternary::TernaryForm;
use rustmath_rationals::Rational;

/// The field the cover solved over.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolvedField {
    Rational,
    Extension { degree: usize },
}

/// Minimal description of a solved cover, enough to route the bridge.
#[derive(Debug, Clone)]
pub struct SolvedCover {
    pub field: SolvedField,
    /// The chosen PGL₂ frame pins are `Q`-rational — true exactly when the cover
    /// solved over `Q` in that frame.
    pub q_frame: bool,
}

impl SolvedCover {
    /// Route a Phase-2 exactification outcome to a solved-cover descriptor.
    /// `AlgebraicCoordinates` gives only a lower bound on the field degree (the
    /// true degree is the compositum's, an S2 follow-up).
    pub fn from_exactify(outcome: &ExactifyOutcome) -> Option<Self> {
        match outcome {
            ExactifyOutcome::CertifiedRational(_) => Some(Self {
                field: SolvedField::Rational,
                q_frame: true,
            }),
            ExactifyOutcome::AlgebraicCoordinates(minpolys) => {
                let deg = minpolys
                    .iter()
                    .map(|p| p.len().saturating_sub(1))
                    .max()
                    .unwrap_or(1);
                Some(Self {
                    field: SolvedField::Extension { degree: deg.max(2) },
                    q_frame: false,
                })
            }
            _ => None,
        }
    }
}

/// The descent recipe to run once the explicit `L`-cover is in hand (S3b).
#[derive(Debug, Clone)]
pub struct DescentPacket {
    pub field_degree: usize,
    pub recipe: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum ConicBridge {
    /// `X_C ≅ P¹_Q`: split conic with a rational point.
    Split(DiagonalConicQ),
    /// Cover over `L`: the conic is the descent obstruction.
    NeedsDescent(DescentPacket),
    /// Over `Q` but no rational frame: anticanonical Riemann–Roch.
    NeedsRiemannRoch,
}

/// The standard split conic `x² + y² − z² = 0`, rational point `[1:0:1]`.
pub fn split_conic() -> DiagonalConicQ {
    DiagonalConicQ::new(
        Rational::from_i64(1),
        Rational::from_i64(1),
        Rational::from_i64(-1),
    )
    .expect("split conic 1,1,-1 is nondegenerate")
}

/// Route a solved cover to its conic-determination path.
pub fn bridge(cover: &SolvedCover) -> ConicBridge {
    match (&cover.field, cover.q_frame) {
        (SolvedField::Rational, true) => ConicBridge::Split(split_conic()),
        (SolvedField::Rational, false) => ConicBridge::NeedsRiemannRoch,
        (SolvedField::Extension { degree }, _) => ConicBridge::NeedsDescent(DescentPacket {
            field_degree: *degree,
            recipe: vec![
                "Compute Gal(L/Q) action on the solved coefficient tuple.".into(),
                "Extract the descent cocycle sigma -> g_sigma in PGL2(L).".into(),
                "Its class in Br(Q)[2] is the conic; ramification within {2,3,5,oo}.".into(),
            ],
        }),
    }
}

/// Bridge, then read through D4 when possible; otherwise an honest `Unresolved`
/// carrying the pending recipe. The split case is `Constructed` *pending* the
/// bad-locus (S4) check, recorded in the notes.
pub fn bridge_and_read(cover: &SolvedCover) -> Verdict<ConicBrauerReport> {
    match bridge(cover) {
        ConicBridge::Split(c) => {
            let report = c.brauer_report().expect("split conic is nondegenerate");
            Verdict {
                kind: VerdictKind::Constructed,
                status: MathStatus::CertifyingSemialgorithm,
                value: Some(report),
                notes: vec!["split via rational frame; confirm the point is off Z_C (S4)".into()],
            }
        }
        ConicBridge::NeedsDescent(p) => Verdict::unresolved(format!(
            "conic over degree-{} field; Galois descent pending (S3b)",
            p.field_degree
        ))
        .with_note(p.recipe.join(" ")),
        ConicBridge::NeedsRiemannRoch => {
            Verdict::unresolved("anticanonical Riemann-Roch (Hess) pending (S3b)")
        }
    }
}

/// Read an explicit conic (a descent- or RR-produced ternary form) through D4.
/// `bad_locus_clear` comes from S4. Returns the D4 verdict.
pub fn read_explicit_conic(form: &TernaryForm, bad_locus_clear: bool) -> Verdict<ConicBrauerReport> {
    match DiagonalConicQ::from_ternary_form(form) {
        Ok(c) => c
            .verdict(bad_locus_clear)
            .unwrap_or_else(|e| Verdict::unresolved(format!("conic read failed: {e}"))),
        Err(e) => Verdict::unresolved(format!("degenerate conic form: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use rustmath_quadraticforms::hilbert::Place;

    #[test]
    fn rational_frame_splits_and_reads_via_d4() {
        let cover = SolvedCover {
            field: SolvedField::Rational,
            q_frame: true,
        };
        assert!(matches!(bridge(&cover), ConicBridge::Split(_)));
        let v = bridge_and_read(&cover);
        assert_eq!(v.kind, VerdictKind::Constructed);
        let report = v.value.unwrap();
        assert!(report.has_rational_point);
        assert!(report.ramified_places.is_empty());
    }

    #[test]
    fn extension_routes_to_descent() {
        let cover = SolvedCover {
            field: SolvedField::Extension { degree: 2 },
            q_frame: false,
        };
        assert!(matches!(bridge(&cover), ConicBridge::NeedsDescent(_)));
        let v = bridge_and_read(&cover);
        assert_eq!(v.kind, VerdictKind::Unresolved);
        assert!(v.notes.iter().any(|n| n.contains("cocycle")));
    }

    #[test]
    fn over_q_without_frame_routes_to_riemann_roch() {
        let cover = SolvedCover {
            field: SolvedField::Rational,
            q_frame: false,
        };
        assert!(matches!(bridge(&cover), ConicBridge::NeedsRiemannRoch));
    }

    #[test]
    fn reads_mueller_explicit_conic_via_bridge() {
        // The descent (or hand-descent) yields Müller's conic x²+y²+z² = (-1,-1).
        // The bridge must read it as anisotropic, ramified {2,∞}: LocallyEmpty.
        let z = Rational::from_i64(0);
        let form = TernaryForm::from_coeffs(
            Rational::from_i64(1),
            Rational::from_i64(1),
            Rational::from_i64(1),
            z.clone(),
            z.clone(),
            z,
        )
        .unwrap();
        let v = read_explicit_conic(&form, false);
        assert_eq!(v.kind, VerdictKind::LocallyEmpty);
        let report = v.value.unwrap();
        assert!(!report.has_rational_point);
        assert!(report.ramified_places.contains(&Place::Finite(2)));
        assert!(report.ramified_places.contains(&Place::Real));
        assert_eq!(report.ramified_places.len(), 2);
    }

    #[test]
    fn exactify_outcome_routes_to_solved_cover() {
        let rat = ExactifyOutcome::CertifiedRational(vec![Rational::from_i64(2), Rational::from_i64(3)]);
        let cover = SolvedCover::from_exactify(&rat).unwrap();
        assert_eq!(cover.field, SolvedField::Rational);
        assert!(cover.q_frame);

        let alg = ExactifyOutcome::AlgebraicCoordinates(vec![vec![
            BigInt::from(-2),
            BigInt::from(0),
            BigInt::from(1),
        ]]);
        let cover = SolvedCover::from_exactify(&alg).unwrap();
        assert_eq!(cover.field, SolvedField::Extension { degree: 2 });
    }
}
