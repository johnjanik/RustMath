//! Result and identification types for Galois-group computation.
//!
//! References:
//! - MAGMA Handbook, Chapter 38 (Galois Groups): `GaloisGroup(f)` returns an
//!   identified permutation group together with proof data; this port makes the
//!   "identified vs. not yet separated" distinction explicit in the type.
//! - SageMath: `sage.rings.number_field.galois_group` (`GaloisGroup_v2`,
//!   `galois_group()` on number fields / polynomials).
//!
//! The central honesty contract of this crate lives here: a computation either
//! produces [`GaloisGroupResult::Decided`] with an exact certificate trail in
//! [`Evidence`], or [`GaloisGroupResult::Unresolved`] carrying precisely what
//! was ruled out (each removal justified by an exact invariant or an exhibited
//! Frobenius element) and what remains. A bounded search that merely fails to
//! find a distinguishing witness NEVER decides.

use rustmath_integers::Integer;

/// A cycle type of a permutation of `n` points: the partition of `n` given by
/// the cycle lengths, sorted in **descending** order (e.g. `[3, 1, 1]`).
pub type CycleType = Vec<usize>;

/// An identified Galois group, named as an abstract group together with the
/// degree of the root action it was identified in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupId {
    /// Number of roots the group permutes in the identification (the degree of
    /// the polynomial whose splitting field was analysed).
    pub degree: usize,
    /// Group order, when representable in `u128` (`None` only for `S_n`/`A_n`
    /// at degrees where `n!` overflows `u128`; the identification itself is
    /// still exact).
    pub order: Option<u128>,
    /// Conventional name: `"C5"`, `"D4"`, `"F20"`, `"S3xC2"`, `"C2^3"`, ….
    pub name: String,
    /// Transitive-group index `nTk` (Butler–McKay numbering) when the action is
    /// transitive and the degree is covered by the built-in tables (n ≤ 5).
    pub t_number: Option<u32>,
}

impl GroupId {
    pub fn new(degree: usize, order: u128, name: &str, t_number: Option<u32>) -> Self {
        GroupId { degree, order: Some(order), name: name.to_string(), t_number }
    }
}

/// The exact evidence trail accumulated during classification. Everything in
/// here is a *proven* statement about the true Galois group `G` of the input:
/// observed Frobenius cycle types are cycle types of actual elements of `G`
/// (Dedekind), resolvent signatures are exact orbit-length multisets of the
/// `G`-action on root pairs/subsets.
#[derive(Clone, Debug, Default)]
pub struct Evidence {
    /// Degree of the (trimmed) input polynomial.
    pub degree: usize,
    /// Whether the input itself was irreducible over `ℚ` (after removing
    /// content). When false, the classification concerns the splitting field of
    /// the radical; see `notes`.
    pub irreducible: bool,
    /// Discriminant of the monic polynomial that was actually classified
    /// (the monicized unique nonlinear factor in the delegated-reducible case).
    pub discriminant: Option<Integer>,
    /// `disc` is a nonzero perfect square ⟺ `G ⊆ A_n` in the root action.
    pub disc_is_square: Option<bool>,
    /// Distinct Frobenius cycle types observed, each with the smallest good
    /// prime witnessing it. Each entry certifies an element of `G` with that
    /// cycle type.
    pub frobenius_types: Vec<(u64, CycleType)>,
    /// Exact resolvent invariants computed: `(description, factor-degree
    /// multiset)`; the multiset equals the orbit lengths of `G` on the indexed
    /// family (pairs, subsets, …) whenever the resolvent was verified
    /// squarefree — which is a precondition for recording it here.
    pub resolvent_signatures: Vec<(String, Vec<usize>)>,
    /// Human-readable audit trail of every normalization and decision step.
    pub notes: Vec<String>,
}

/// The remaining possibilities of an unresolved computation.
#[derive(Clone, Debug)]
pub enum Candidates {
    /// The true group is provably one of these (complete candidate list).
    Among(Vec<GroupId>),
    /// No complete transitive-group table is built in for this degree; the
    /// group is only constrained by the recorded evidence.
    Unknown {
        degree: usize,
        /// Whether the action on the roots is transitive (input irreducible).
        transitive: bool,
        /// `Some(true)` ⟺ proven `G ⊆ A_n`; `Some(false)` ⟺ proven `G ⊄ A_n`.
        contained_in_alternating: Option<bool>,
    },
}

/// Outcome of a Galois-group computation. See module docs for the honesty
/// contract; in particular `Unresolved` is a first-class answer, not an error.
#[derive(Clone, Debug)]
pub enum GaloisGroupResult {
    /// The group is exactly identified; `evidence` contains the certificate
    /// trail that forces uniqueness.
    Decided { group: GroupId, evidence: Evidence },
    /// The bounded search did not force uniqueness. `ruled_out` lists every
    /// candidate eliminated together with the exact reason; `candidates` is
    /// what remains; `blocked_on` names the missing capability that would
    /// complete the decision.
    Unresolved {
        candidates: Candidates,
        ruled_out: Vec<(GroupId, String)>,
        evidence: Evidence,
        blocked_on: String,
    },
}

impl GaloisGroupResult {
    pub fn is_decided(&self) -> bool {
        matches!(self, GaloisGroupResult::Decided { .. })
    }

    /// Name of the decided group, if decided.
    pub fn decided_name(&self) -> Option<&str> {
        match self {
            GaloisGroupResult::Decided { group, .. } => Some(&group.name),
            GaloisGroupResult::Unresolved { .. } => None,
        }
    }

    pub fn evidence(&self) -> &Evidence {
        match self {
            GaloisGroupResult::Decided { evidence, .. } => evidence,
            GaloisGroupResult::Unresolved { evidence, .. } => evidence,
        }
    }

    /// Names of the remaining candidates: the decided name alone, the explicit
    /// candidate list, or `None` when the candidate set is not enumerable.
    pub fn candidate_names(&self) -> Option<Vec<&str>> {
        match self {
            GaloisGroupResult::Decided { group, .. } => Some(vec![group.name.as_str()]),
            GaloisGroupResult::Unresolved { candidates: Candidates::Among(v), .. } => {
                Some(v.iter().map(|g| g.name.as_str()).collect())
            }
            GaloisGroupResult::Unresolved { candidates: Candidates::Unknown { .. }, .. } => None,
        }
    }
}

/// `n!` as `u128` when representable.
pub(crate) fn factorial_u128(n: usize) -> Option<u128> {
    let mut acc: u128 = 1;
    for k in 2..=n as u128 {
        acc = acc.checked_mul(k)?;
    }
    Some(acc)
}
