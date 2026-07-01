//! The exact cover verifier — a hard gate before a cover counts as constructible.
//!
//! Ported from `dessin_engine/src/cover_verify.rs` in
//! `/home/john/inverse_galois/M23/dessin_engine` (S5). Serde derives are dropped
//! (not needed by this crate).
//!
//! A solved cover counts only after it passes, independently: the exact identity
//! `P − Q − c·U¹² = 0` over `L`, branch locus exactly `{0,1,∞}`, ramification
//! pattern `2⁸1⁸/12²/5⁴1⁴`, Riemann–Hurwitz genus 0, and a recomputed monodromy
//! matching the group-theoretic triple. Missing monodromy is **not** a pass — it
//! is [`VerificationFailure::MonodromyUnresolved`], and
//! [`VerificationReport::is_constructible`] stays `false`.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationFailure {
    IdentityFailed(String),
    WrongBranchLocus,
    WrongRamificationPattern,
    GenusMismatch { expected: i64, observed: i64 },
    MonodromyMismatch,
    MonodromyUnresolved,
}

#[derive(Debug, Clone)]
pub struct VerificationReport {
    pub exact_identity_ok: bool,
    pub branch_locus_ok: bool,
    pub ramification_pattern_ok: bool,
    pub genus_ok: bool,
    pub monodromy_ok: Option<bool>,
    pub failures: Vec<VerificationFailure>,
}

impl VerificationReport {
    /// The full conjunction — every audit must pass, and monodromy must be
    /// affirmatively verified (`Some(true)`), never merely unrefuted.
    pub fn is_constructible(&self) -> bool {
        self.exact_identity_ok
            && self.branch_locus_ok
            && self.ramification_pattern_ok
            && self.genus_ok
            && self.monodromy_ok == Some(true)
            && self.failures.is_empty()
    }
}

/// The audit a solved exact cover must answer. `L` is the field of definition
/// (arithmetic lives in the implementor).
pub trait ExactBelyiCover {
    fn verify_identity(&self) -> Result<(), String>;
    fn verify_branch_locus_0_1_infty(&self) -> bool;
    fn verify_ramification_2_12_5(&self) -> bool;
    fn observed_genus(&self) -> Option<i64>;
    /// `Some(true)` only on an affirmative independent monodromy check; `None`
    /// means "not established", which blocks constructibility.
    fn verify_monodromy_independent(&self) -> Option<bool>;
}

pub fn verify_2_12_5_cover<C: ExactBelyiCover>(cover: &C) -> VerificationReport {
    let mut failures = Vec::new();

    let exact_identity_ok = match cover.verify_identity() {
        Ok(()) => true,
        Err(e) => {
            failures.push(VerificationFailure::IdentityFailed(e));
            false
        }
    };

    let branch_locus_ok = cover.verify_branch_locus_0_1_infty();
    if !branch_locus_ok {
        failures.push(VerificationFailure::WrongBranchLocus);
    }

    let ramification_pattern_ok = cover.verify_ramification_2_12_5();
    if !ramification_pattern_ok {
        failures.push(VerificationFailure::WrongRamificationPattern);
    }

    let observed_genus = cover.observed_genus();
    let genus_ok = observed_genus == Some(0);
    if !genus_ok {
        failures.push(VerificationFailure::GenusMismatch {
            expected: 0,
            observed: observed_genus.unwrap_or(i64::MIN),
        });
    }

    let monodromy_ok = cover.verify_monodromy_independent();
    match monodromy_ok {
        Some(true) => {}
        Some(false) => failures.push(VerificationFailure::MonodromyMismatch),
        None => failures.push(VerificationFailure::MonodromyUnresolved),
    }

    VerificationReport {
        exact_identity_ok,
        branch_locus_ok,
        ramification_pattern_ok,
        genus_ok,
        monodromy_ok,
        failures,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Cover {
        monodromy: Option<bool>,
    }
    impl ExactBelyiCover for Cover {
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
            self.monodromy
        }
    }

    #[test]
    fn no_constructed_without_monodromy() {
        // Every other check passes, but monodromy is unresolved ⇒ not constructible.
        let report = verify_2_12_5_cover(&Cover { monodromy: None });
        assert!(!report.is_constructible());
        assert!(report
            .failures
            .iter()
            .any(|f| matches!(f, VerificationFailure::MonodromyUnresolved)));
    }

    #[test]
    fn constructible_only_with_full_pass() {
        let report = verify_2_12_5_cover(&Cover { monodromy: Some(true) });
        assert!(report.is_constructible());
        // a refuted monodromy is a hard failure
        let bad = verify_2_12_5_cover(&Cover { monodromy: Some(false) });
        assert!(!bad.is_constructible());
        assert!(bad
            .failures
            .iter()
            .any(|f| matches!(f, VerificationFailure::MonodromyMismatch)));
    }
}
