//! Mathematical status labels and status-tagged verdicts.
//!
//! Per the sporadic-Galois-realization specification, every module declares the
//! mathematical status of its output, and the pipeline must be honest about what
//! is *decided* versus merely *searched*. A computation returns a [`Verdict`]
//! carrying both a value (when one is produced) and its [`MathStatus`].

/// The status of an algorithmic claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathStatus {
    /// Established theorem with a known proof.
    Theorem,
    /// Implementable exact algorithm with specified input/output/termination.
    Algorithm,
    /// May not terminate in all cases, but any output produced is certified.
    CertifyingSemialgorithm,
    /// Depends on hypotheses (rank bounds, completeness of a search, GRH, …).
    Conditional,
    /// Research-level pattern or conjectural mechanism.
    Speculative,
}

/// A value together with the mathematical status under which it is asserted.
/// `value == None` means "no conclusion under available methods" (`UNRESOLVED`).
#[derive(Debug, Clone)]
pub struct Verdict<T> {
    pub value: Option<T>,
    pub status: MathStatus,
    pub note: String,
}

impl<T> Verdict<T> {
    /// An exact, certified result.
    pub fn algorithm(value: T) -> Self {
        Verdict { value: Some(value), status: MathStatus::Algorithm, note: String::new() }
    }
    /// A certified result from a semi-algorithm (terminated, output trusted).
    pub fn certifying(value: T, note: impl Into<String>) -> Self {
        Verdict {
            value: Some(value),
            status: MathStatus::CertifyingSemialgorithm,
            note: note.into(),
        }
    }
    /// A result that holds under a stated hypothesis.
    pub fn conditional(value: T, note: impl Into<String>) -> Self {
        Verdict { value: Some(value), status: MathStatus::Conditional, note: note.into() }
    }
    /// No conclusion reached.
    pub fn unresolved(note: impl Into<String>) -> Self {
        Verdict { value: None, status: MathStatus::CertifyingSemialgorithm, note: note.into() }
    }
}
