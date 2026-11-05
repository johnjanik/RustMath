//! The arithmetic payload of a genus-0 curve over `Q`: a conic, read through its
//! rational point and Brauer class.
//!
//! Ported from `dessin_engine/src/conic.rs`
//! (`/home/john/inverse_galois/M23/dessin_engine/src/conic.rs`), adapted to
//! `rustmath_rationals::Rational` and to this crate's `hilbert`/`ternary`/
//! `quaternion` modules. The `Verdict` discipline (originally
//! `dessin_engine/src/status.rs`) is inlined here so the conic reader keeps its
//! "UNRESOLVED vs decided" honesty without an extra module.
//!
//! For `a X² + b Y² + c Z² = 0` (`c ≠ 0`) the obstruction is the quaternion
//! class `(−a/c, −b/c)`: the conic has a `Q`-point iff that class splits, i.e.
//! every Hilbert symbol is `+1` (Hasse–Minkowski; for conics every global
//! failure is local).

use crate::hilbert::{rat_is_zero, HilbertError, Place};
use crate::quaternion::QuaternionAlgebra;
use crate::ternary::{FormError, TernaryForm};
use rustmath_rationals::Rational;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConicError {
    #[error("hilbert symbol error: {0}")]
    Hilbert(#[from] HilbertError),
    #[error("quadratic form error: {0}")]
    Form(#[from] FormError),
    #[error("zero diagonal coefficient: conic is degenerate")]
    DegenerateDiagonal,
}

// --- Verdict discipline (ported from dessin_engine/src/status.rs) ---

/// How rigorous a produced claim is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathStatus {
    /// Backed by a theorem (exact, unconditional).
    Theorem,
    /// A terminating algorithm correct on all inputs.
    Algorithm,
    /// Correct when it succeeds, but may return `Unresolved` instead.
    CertifyingSemialgorithm,
    /// Correct conditional on a stated hypothesis.
    Conditional,
    /// Heuristic / conjectural.
    Speculative,
}

/// The mathematical conclusion of a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerdictKind {
    /// A witness was produced (e.g. a rational point off the bad locus).
    Constructed,
    /// No global object, detected by a local obstruction.
    LocallyEmpty,
    /// No global object, with everywhere-local existence (a genuine global gap).
    GloballyEmpty,
    /// Obstructed for a stated reason short of the above.
    Obstructed,
    /// Undecided within the configured budget.
    Unresolved,
}

/// A status-tagged result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verdict<T> {
    pub kind: VerdictKind,
    pub status: MathStatus,
    pub value: Option<T>,
    pub notes: Vec<String>,
}

impl<T> Verdict<T> {
    pub fn constructed(value: T) -> Self {
        Self {
            kind: VerdictKind::Constructed,
            status: MathStatus::CertifyingSemialgorithm,
            value: Some(value),
            notes: vec![],
        }
    }

    pub fn unresolved(note: impl Into<String>) -> Self {
        Self {
            kind: VerdictKind::Unresolved,
            status: MathStatus::CertifyingSemialgorithm,
            value: None,
            notes: vec![note.into()],
        }
    }

    pub fn locally_empty(value: T, note: impl Into<String>) -> Self {
        Self {
            kind: VerdictKind::LocallyEmpty,
            status: MathStatus::Theorem,
            value: Some(value),
            notes: vec![note.into()],
        }
    }

    pub fn obstructed(value: T, note: impl Into<String>) -> Self {
        Self {
            kind: VerdictKind::Obstructed,
            status: MathStatus::CertifyingSemialgorithm,
            value: Some(value),
            notes: vec![note.into()],
        }
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }
}

// --- The conic reader ---

/// Diagonal ternary conic `a x² + b y² + c z² = 0` over `Q`.
#[derive(Debug, Clone)]
pub struct DiagonalConicQ {
    pub a: Rational,
    pub b: Rational,
    pub c: Rational,
}

#[derive(Debug, Clone)]
pub struct ConicBrauerReport {
    /// The quaternion class `(−a/c, −b/c)`.
    pub quaternion_a: Rational,
    pub quaternion_b: Rational,
    pub ramified_places: Vec<Place>,
    pub has_rational_point: bool,
}

impl DiagonalConicQ {
    pub fn new(a: Rational, b: Rational, c: Rational) -> Result<Self, ConicError> {
        if rat_is_zero(&a) || rat_is_zero(&b) || rat_is_zero(&c) {
            return Err(ConicError::DegenerateDiagonal);
        }
        Ok(Self { a, b, c })
    }

    /// Build from a general ternary form by exact diagonalization (entry point
    /// from a non-diagonal conic). Requires the form to be nondegenerate.
    pub fn from_ternary_form(form: &TernaryForm) -> Result<Self, ConicError> {
        let (diag, _p) = form.diagonalize()?;
        Self::new(diag[0].clone(), diag[1].clone(), diag[2].clone())
    }

    /// `(−a/c, −b/c)`.
    pub fn quaternion_class(&self) -> Result<QuaternionAlgebra, ConicError> {
        let qa = -(&self.a / &self.c);
        let qb = -(&self.b / &self.c);
        Ok(QuaternionAlgebra::new(qa, qb))
    }

    /// Exact Hasse–Minkowski reading: the ramified places and whether a point exists.
    pub fn brauer_report(&self) -> Result<ConicBrauerReport, ConicError> {
        let quat = self.quaternion_class()?;
        let ramified = quat.ramified_places()?;
        let has_rational_point = ramified.is_empty();
        Ok(ConicBrauerReport {
            quaternion_a: quat.a,
            quaternion_b: quat.b,
            ramified_places: ramified,
            has_rational_point,
        })
    }

    /// The portal verdict. A rational point realises `M23/Q` only if it lies off
    /// the bad locus `Z_C`; the caller supplies whether the produced point is
    /// `Z_C`-clear (branch / cusp / singular / monodromy-drop check).
    pub fn verdict(&self, bad_locus_clear: bool) -> Result<Verdict<ConicBrauerReport>, ConicError> {
        let report = self.brauer_report()?;
        let places = report
            .ramified_places
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(",");
        Ok(if !report.has_rational_point {
            Verdict::locally_empty(
                report,
                format!("conic anisotropic; ramified at {{{places}}}"),
            )
        } else if bad_locus_clear {
            Verdict::constructed(report).with_note("rational point off the bad locus Z_C")
        } else {
            Verdict::unresolved("conic has a Q-point, but Z_C-clearance not established")
                .with_note("the point may lie in the bad locus; M23/Q not yet realised")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hilbert::Place;
    use crate::ternary::TernaryForm;

    fn r(n: i64) -> Rational {
        Rational::from_i64(n)
    }

    #[test]
    fn x2_plus_y2_plus_z2_has_no_q_point() {
        // Müller's M23 conic shape: (-1,-1), ramified {2, ∞}, anisotropic.
        let c = DiagonalConicQ::new(r(1), r(1), r(1)).unwrap();
        let report = c.brauer_report().unwrap();
        assert!(!report.has_rational_point);
        assert!(report.ramified_places.contains(&Place::Real));
        assert!(report.ramified_places.contains(&Place::Finite(2)));
        assert_eq!(report.ramified_places.len(), 2);

        let v = c.verdict(false).unwrap();
        assert_eq!(v.kind, VerdictKind::LocallyEmpty);
    }

    #[test]
    fn x2_plus_y2_minus_z2_has_q_point() {
        let c = DiagonalConicQ::new(r(1), r(1), r(-1)).unwrap();
        let report = c.brauer_report().unwrap();
        assert!(report.has_rational_point);
        assert!(report.ramified_places.is_empty());

        assert_eq!(c.verdict(true).unwrap().kind, VerdictKind::Constructed);
        assert_eq!(c.verdict(false).unwrap().kind, VerdictKind::Unresolved);
    }

    #[test]
    fn split_conic_has_empty_ramification_set() {
        let c = DiagonalConicQ::new(r(1), r(1), r(-1)).unwrap();
        let report = c.brauer_report().unwrap();
        assert!(report.has_rational_point);
        assert!(report.ramified_places.is_empty());
    }

    #[test]
    fn mueller_gate_conic_hamilton_quaternion() {
        // x² + y² + z² = 0 is (-1,-1), ramified exactly at {2, oo}.
        let c = DiagonalConicQ::new(r(1), r(1), r(1)).unwrap();
        let report = c.brauer_report().unwrap();
        assert!(!report.has_rational_point);
        assert!(report.ramified_places.contains(&Place::Finite(2)));
        assert!(report.ramified_places.contains(&Place::Real));
        assert_eq!(report.ramified_places.len(), 2);
    }

    #[test]
    fn reads_conic_from_nondiagonal_form() {
        // 2xy - z^2 = 0 is isotropic; diagonalize then read.
        let form = TernaryForm::from_coeffs(r(0), r(0), r(-1), r(2), r(0), r(0)).unwrap();
        let conic = DiagonalConicQ::from_ternary_form(&form).unwrap();
        let report = conic.brauer_report().unwrap();
        assert!(report.has_rational_point);
    }
}
