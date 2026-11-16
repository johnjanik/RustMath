//! The reduction certificate and the point counts of the normalization.
//!
//! Specification: MAGMA Handbook ch. 114 *Algebraic Curves* (digest
//! `rustmath-lava/docs/Chapter_114_Algebraic_Curves.md`): 114.5.1 `Genus`, `ArithmeticGenus`
//! (the delta-sum `p_a - g`), 114.4.3 `IsNode`, `IsAnalyticallyIrreducible` (the branches above
//! a singular point), 114.8.1 `IsAbsolutelyIrreducible`, `NumberOfPlacesOfDegreeOneECF`,
//! `LPolynomial`.
//!
//! The certificate is Lemma 9.1 of the owner's paper (johnjanik/M23 at `2d11038`,
//! `paper/golay_m23.tex`, `\label{lem:reduction}`). Let `C` be smooth, projective and
//! geometrically connected of genus `g` over `Q`, birational to the curve `C_0` in
//! `P^1 x P^1` given by a primitive form `F` in `Z[X, Y]` of bidegree `(a, b)`. If `F mod p` has
//! bidegree `(a, b)`, is absolutely irreducible, and the delta invariants of the singular points
//! of `F mod p = 0` over `F_p-bar` add up to `p_a - g`, `p_a = (a - 1)(b - 1)`, then `C` has good
//! reduction at `p` and its reduction is the normalization of `F mod p = 0`. So the point counts
//! of that normalization are those of the reduction of `C`. The WP-F design, Theorem 4.5, is the
//! same statement with a proof by Zariski's connectedness theorem (R. Hartshorne, *Algebraic
//! geometry*, III.9, III.11; Q. Liu, *Algebraic geometry and arithmetic curves*, 7.5, 10.1;
//! W. Fulton, *Algebraic curves*, delta invariants).
//!
//! What [`certify`] checks, hypothesis by hypothesis:
//! - primitivity of `F` over `Z`: the caller's (see [`FormModP`];
//!   [`BihomogeneousForm::reduce_primitive`](super::BihomogeneousForm::reduce_primitive) checks it
//!   where the integral form is known). At `p` it means `F mod p != 0`, which
//!   [`FormModP::new`] checks;
//! - the bidegree is kept and `F mod p` is absolutely irreducible:
//!   [`absolute_irreducibility`](super::absolute_irreducibility)
//!   (a line of a ruling, including the lines at infinity, is a refusal of the model; otherwise
//!   a sufficient criterion, and `Unresolved(IrreducibilityNotShown)` when it does not apply);
//! - every delta invariant is computed and they add up to `p_a - g`: the singular locus;
//! - the genus `g`: a trusted input, recorded in the certificate (with a genus below the true
//!   one a bad prime can be certified).
//!
//! The proof of Lemma 9.1 uses only that the base is a discrete valuation ring with perfect
//! residue field (the uniformizer in place of `p`), so it applies at a degree-one prime of a
//! number field, as for the covers over `L` of the M23 workstream.
//!
//! Statuses. Both certificates are `Decided` (the owner's decision on follow-up F-9 of the
//! WP-F report, 2026-11-15), and the type and the label say which one it is:
//! [`Certificate::Nodal`] (every singular point a node) and [`Certificate::DeltaSum`] (some
//! point is an `A_k` or an ordinary m-fold point, classified exactly). Unresolved: an
//! unclassified point, a delta-sum above `p_a - g`, or absolute irreducibility not shown. A
//! delta-sum below `p_a - g` is impossible for a correct genus input (it needs only that
//! `F mod p` is reduced) and is an error.
//!
//! Counts: `N_k = #X0_p(F_q) + sum over closed singular points P with d_P | k of
//! d_P (b_P(k) - 1)`, with `b_P(k)` the rational branches above one geometric point
//! ([`SingularPoint::rational_branches`]). A Decided status always comes with these exact
//! counts. The reference rule of `curve.py` at `d05475b` counts every non-nodal point once
//! ([`BranchRule::AssumedUnibranch`], [`NormalizationCounts::counts_under`]); it is exact for
//! `A_k` with `k` even and wrong otherwise.

use super::fibres::{plane_point_count, Fibration};
use super::irreducible::{
    check_bidegree, check_line_components, irreducibility_search, Irreducibility,
    IrreducibilityEvidence, IrreducibilityGap,
};
use super::local::LocalType;
use super::singular::{singular_locus, SingularLocus, SingularPoint};
use super::{CountError, FormModP};
use crate::weil::{FrobeniusPolynomial, WeilError};
use rustmath_integers::Integer;

/// A proof, by Lemma 9.1, that the curve has good reduction at `p` and that the counts are
/// those of its reduction. Both kinds carry the evidence of absolute irreducibility. Only
/// [`certify`] builds a certificate (the variants are `non_exhaustive`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Certificate {
    /// Every singular point of `V(F mod p)` is a node and there are `p_a - g` of them over
    /// `F_p-bar`. Relative to `genus_hypothesis = g`, a trusted input (with a genus below the
    /// true one a bad prime can be certified).
    #[non_exhaustive]
    Nodal {
        nodes: usize,
        genus_hypothesis: usize,
        irreducibility: IrreducibilityEvidence,
    },
    /// Some singular point is not a node; every point is classified exactly (a node, an `A_k`
    /// or an ordinary m-fold point) and the delta invariants add up to `delta_sum = p_a - g`
    /// over `F_p-bar`. `nodes` counts the nodes over `F_p-bar`, `non_nodal` lists the other
    /// closed points.
    #[non_exhaustive]
    DeltaSum {
        delta_sum: usize,
        nodes: usize,
        genus_hypothesis: usize,
        non_nodal: Vec<SingularPoint>,
        irreducibility: IrreducibilityEvidence,
    },
}

impl Certificate {
    /// The genus input the certificate is relative to.
    pub fn genus_hypothesis(&self) -> usize {
        match self {
            Certificate::Nodal {
                genus_hypothesis, ..
            }
            | Certificate::DeltaSum {
                genus_hypothesis, ..
            } => *genus_hypothesis,
        }
    }

    /// The evidence that `F mod p` is absolutely irreducible.
    pub fn irreducibility(&self) -> &IrreducibilityEvidence {
        match self {
            Certificate::Nodal { irreducibility, .. }
            | Certificate::DeltaSum { irreducibility, .. } => irreducibility,
        }
    }
}

/// Why no certificate is given.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unresolved {
    /// Every singular point is classified and the delta-sum is `p_a - g`, but the absolute
    /// irreducibility of `F mod p` (a hypothesis of Lemma 9.1) is not shown. This does not say
    /// that `F mod p` is reducible.
    IrreducibilityNotShown(IrreducibilityGap),
    /// A singular point the classifier could not identify (its delta is unknown).
    Unclassified(SingularPoint),
    /// The delta-sum exceeds `p_a - g`: the model, or the curve, is bad at p. With unclassified
    /// points `found` is a lower bound (each of them counts at least one).
    DeltaMismatch {
        found: usize,
        expected: usize,
        found_is_lower_bound: bool,
    },
}

/// The status of a count.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReductionStatus {
    Decided(Certificate),
    Unresolved(Unresolved),
}

/// `A2`, `A3 rational branches`, `A3 conjugate branches`, `ordinary 3-fold point [1, 2]`,
/// joined by `; `; a closed point of degree `d > 1` adds ` at a point of degree d`.
fn non_nodal_types(non_nodal: &[SingularPoint]) -> String {
    let types: Vec<String> = non_nodal
        .iter()
        .map(|s| {
            let br = match (&s.local, &s.branch_orbits) {
                (LocalType::A(k), Some(o)) if k % 2 == 1 => {
                    if o == &vec![1, 1] {
                        " rational branches".to_string()
                    } else {
                        " conjugate branches".to_string()
                    }
                }
                (LocalType::OrdinaryMultiple(_), Some(o)) => format!(" {o:?}"),
                _ => String::new(),
            };
            let deg = if s.residue_degree > 1 {
                format!(" at a point of degree {}", s.residue_degree)
            } else {
                String::new()
            };
            format!("{}{br}{deg}", s.local)
        })
        .collect();
    types.join("; ")
}

impl ReductionStatus {
    pub fn is_decided(&self) -> bool {
        matches!(self, ReductionStatus::Decided(_))
    }

    /// The certificate of a Decided status.
    pub fn certificate(&self) -> Option<&Certificate> {
        match self {
            ReductionStatus::Decided(c) => Some(c),
            ReductionStatus::Unresolved(_) => None,
        }
    }

    /// A short label: `Decided: nodal (62 nodes)`,
    /// `Decided: delta-sum, Lemma 9.1 (61 nodes, A2)`, `Unresolved: ...`.
    pub fn label(&self) -> String {
        match self {
            ReductionStatus::Decided(Certificate::Nodal { nodes, .. }) => {
                format!("Decided: nodal ({nodes} nodes)")
            }
            ReductionStatus::Decided(Certificate::DeltaSum {
                nodes, non_nodal, ..
            }) => format!(
                "Decided: delta-sum, Lemma 9.1 ({nodes} nodes, {})",
                non_nodal_types(non_nodal)
            ),
            ReductionStatus::Unresolved(Unresolved::IrreducibilityNotShown(gap)) => {
                format!("Unresolved: absolute irreducibility not shown ({gap})")
            }
            ReductionStatus::Unresolved(Unresolved::Unclassified(s)) => {
                format!("Unresolved: unclassified singular point ({})", s.describe())
            }
            ReductionStatus::Unresolved(Unresolved::DeltaMismatch {
                found,
                expected,
                found_is_lower_bound,
            }) => format!(
                "Unresolved: delta-sum {}{found} above p_a - g = {expected}",
                if *found_is_lower_bound { ">= " } else { "" }
            ),
        }
    }
}

/// The status of `V(F mod p)` under Lemma 9.1 (module header), and its singular locus, for a
/// curve whose normalization over `Q` has genus `genus` (a trusted input; see
/// [`Certificate::Nodal`]).
///
/// The hypotheses that are the caller's: `f` is the reduction of a primitive form of the same
/// bidegree whose zero set is birational to the curve of genus `genus` (see [`FormModP`]).
///
/// Order of the checks: the bidegree (both entries positive); [`singular_locus`] (its errors);
/// `genus <= p_a`; line components ([`absolute_irreducibility`](super::absolute_irreducibility),
/// step 1); the delta tally
/// (`Unclassified`, `DeltaMismatch`, or the error `GenusInconsistent`); then the irreducibility
/// search, which runs only when a certificate is in reach.
///
/// Errors: `Shape` (a bidegree with a zero entry, or `genus > p_a`), the errors of
/// [`singular_locus`], `DegenerateModel(FibreComponent)` for a line component, and
/// `GenusInconsistent` when every point is classified and the delta-sum is below `p_a - genus`
/// (impossible for a correct genus).
///
/// Cost: [`singular_locus`] and [`absolute_irreducibility`](super::absolute_irreducibility)
/// (see theirs).
pub fn certify(f: &FormModP, genus: usize) -> Result<(ReductionStatus, SingularLocus), CountError> {
    check_bidegree(f)?;
    let locus = singular_locus(f)?;
    let arithmetic_genus = f.arithmetic_genus();
    if genus > arithmetic_genus {
        return Err(CountError::Shape(format!(
            "genus {genus} above the arithmetic genus {arithmetic_genus}"
        )));
    }
    check_line_components(f)?;
    let expected = arithmetic_genus - genus;
    let mut known = 0usize;
    let mut unknown: Option<&SingularPoint> = None;
    let mut unknown_min = 0usize;
    for s in &locus.points {
        match s.delta {
            Some(d) => known += d * s.residue_degree,
            None => {
                unknown_min += s.residue_degree;
                if unknown.is_none() {
                    unknown = Some(s);
                }
            }
        }
    }
    if let Some(s) = unknown {
        let status = if known + unknown_min > expected {
            Unresolved::DeltaMismatch {
                found: known + unknown_min,
                expected,
                found_is_lower_bound: true,
            }
        } else {
            Unresolved::Unclassified(s.clone())
        };
        return Ok((ReductionStatus::Unresolved(status), locus));
    }
    if known < expected {
        return Err(CountError::GenusInconsistent {
            found: known,
            expected,
        });
    }
    if known > expected {
        let status = Unresolved::DeltaMismatch {
            found: known,
            expected,
            found_is_lower_bound: false,
        };
        return Ok((ReductionStatus::Unresolved(status), locus));
    }
    let irreducibility = match irreducibility_search(f) {
        Irreducibility::Absolute(ev) => ev,
        Irreducibility::NotShown(gap) => {
            return Ok((
                ReductionStatus::Unresolved(Unresolved::IrreducibilityNotShown(gap)),
                locus,
            ))
        }
    };
    let non_nodal: Vec<SingularPoint> = locus.non_nodal().into_iter().cloned().collect();
    let certificate = if non_nodal.is_empty() {
        Certificate::Nodal {
            nodes: locus.node_count(),
            genus_hypothesis: genus,
            irreducibility,
        }
    } else {
        Certificate::DeltaSum {
            delta_sum: known,
            nodes: locus.node_count(),
            genus_hypothesis: genus,
            non_nodal,
            irreducibility,
        }
    };
    Ok((ReductionStatus::Decided(certificate), locus))
}

/// How the branches of a non-nodal singular point are counted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BranchRule {
    /// From the exact local classification.
    Exact,
    /// Every non-nodal singular point counts once (the rule of the reference `curve.py` at
    /// `d05475b`): "assumed unibranch". Never the rule of the counts of a Decided status; use
    /// [`NormalizationCounts::counts_under`] to see what it gives.
    AssumedUnibranch,
}

/// Options of [`normalization_point_counts`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CountOptions {
    /// The fibration of the plane count (the certificate always fibres in `X`).
    pub fibration: Fibration,
    /// With an unclassified point (status Unresolved): give the counts under the
    /// `AssumedUnibranch` rule (true) or no counts (false, the default).
    pub unibranch_if_unclassified: bool,
    /// Threads for the plane count (`None`: the global rayon pool).
    pub threads: Option<usize>,
}

impl Default for CountOptions {
    fn default() -> Self {
        CountOptions {
            fibration: Fibration::FibresInX,
            unibranch_if_unclassified: false,
            threads: None,
        }
    }
}

/// The result of [`normalization_point_counts`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormalizationCounts {
    pub p: u64,
    /// The genus input (a trusted hypothesis).
    pub genus: usize,
    /// `N_1, ..., N_kmax` of the normalization of `V(F mod p)` under `branch_rule`; empty when no
    /// branch rule applies (an unclassified point without `unibranch_if_unclassified`).
    pub counts: Vec<u64>,
    /// `#X0_p(F_(p^k))`, `k = 1..=kmax`, of the plane model.
    pub plane_counts: Vec<u64>,
    pub status: ReductionStatus,
    pub locus: SingularLocus,
    /// The rule `counts` were made with: `Exact`, except with an unclassified point under
    /// `unibranch_if_unclassified` (status Unresolved). `Exact` whenever the status is Decided.
    pub branch_rule: BranchRule,
}

impl NormalizationCounts {
    /// The Frobenius polynomial from `N_1, ..., N_g` (needs `counts.len() >= g`).
    pub fn frobenius_polynomial(&self) -> Option<Result<FrobeniusPolynomial, WeilError>> {
        if self.counts.len() < self.genus {
            return None;
        }
        Some(FrobeniusPolynomial::from_counts_u64(
            self.p,
            &self.counts[..self.genus],
        ))
    }

    /// The consistency check beyond `g`: whether `N_(g+1), ..., N_kmax` are the counts predicted
    /// by `P` (from `N_1..N_g`). `None` when `kmax <= g` or `P` cannot be formed.
    pub fn predicted_counts_agree(&self) -> Option<bool> {
        if self.counts.len() <= self.genus {
            return None;
        }
        let fp = self.frobenius_polynomial()?.ok()?;
        let c: Vec<Integer> = self.counts.iter().map(|&n| Integer::from(n)).collect();
        Some(fp.predicts(&c))
    }

    /// `N_1..N_kmax` under a branch rule, from `plane_counts` and `locus`; `None` under `Exact`
    /// with an unclassified point. `AssumedUnibranch` reproduces the reference rule of
    /// `curve.py` at `d05475b` (every non-nodal point once): those counts are not certified,
    /// whatever the status.
    pub fn counts_under(&self, rule: BranchRule) -> Option<Vec<u64>> {
        let corr = corrections(&self.locus, self.plane_counts.len(), rule)?;
        Some(apply_corrections(&self.plane_counts, &corr))
    }
}

/// The corrections `N_k - #X0_p(F_(p^k))`, `k = 1..=kmax`, under a rule; `None` if a point has
/// no branch data under `Exact`.
pub(crate) fn corrections(
    locus: &SingularLocus,
    kmax: usize,
    rule: BranchRule,
) -> Option<Vec<i64>> {
    let mut corr = vec![0i64; kmax];
    for s in &locus.points {
        let d = s.residue_degree;
        let nodal = s.local == LocalType::Node;
        for k in 1..=kmax {
            if k % d != 0 {
                continue;
            }
            let b = if nodal || rule == BranchRule::Exact {
                s.rational_branches(k)? as i64
            } else {
                1
            };
            corr[k - 1] += d as i64 * (b - 1);
        }
    }
    Some(corr)
}

fn apply_corrections(plane_counts: &[u64], corr: &[i64]) -> Vec<u64> {
    plane_counts
        .iter()
        .zip(corr.iter())
        .map(|(&n, &d)| {
            let v = n as i64 + d;
            assert!(v >= 0, "negative count");
            v as u64
        })
        .collect()
}

/// Point counts `N_1..N_kmax` of the normalization of `V(F mod p)`, with the reduction status
/// of [`certify`].
///
/// `genus` is the genus of the normalization of `V(F)` over `Q`, a trusted input: the status is
/// relative to it (cite its source at the call site). The other hypotheses of Lemma 9.1 that are
/// the caller's are those of [`certify`]. `F` must have `X`-degree in `2..=4` (transpose
/// otherwise); `p` is the prime of `f`; `kmax <= 8` and `p^kmax < 2^63`.
///
/// Cost: [`certify`] once (see its cost), then [`plane_point_count`] for each `k = 1..=kmax`, so
/// about `p^k/k + 1` fibre root counts for each k. The last k dominates. For the M23 curve in
/// release on 4 threads: milliseconds for the certificate, about 20 s at p = 79 and kmax = 4
/// (WP-F report, Section 6). The bound `p^kmax < 2^63` does not make the call fast: see
/// [`plane_point_count`].
///
/// A Decided status always comes with the exact counts (`branch_rule == Exact`): the
/// `AssumedUnibranch` rule is used only for an unclassified point under
/// `unibranch_if_unclassified`, and then the status is Unresolved.
pub fn normalization_point_counts(
    f: &FormModP,
    genus: usize,
    kmax: u32,
    opts: &CountOptions,
) -> Result<NormalizationCounts, CountError> {
    let p = f.p();
    if kmax > 8 || (p as u128).pow(kmax) >= 1u128 << 63 {
        return Err(CountError::FieldTooLarge { p, k: kmax });
    }
    let (status, locus) = certify(f, genus)?;
    let run = || -> Result<Vec<u64>, CountError> {
        (1..=kmax)
            .map(|k| plane_point_count(f, k, opts.fibration))
            .collect()
    };
    let plane_counts = match opts.threads {
        None => run()?,
        Some(n) => rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .map_err(|e| CountError::Shape(format!("thread pool: {e}")))?
            .install(run)?,
    };
    let kmax = kmax as usize;
    let (rule, corr) = match corrections(&locus, kmax, BranchRule::Exact) {
        Some(c) => (BranchRule::Exact, Some(c)),
        None if opts.unibranch_if_unclassified => (
            BranchRule::AssumedUnibranch,
            corrections(&locus, kmax, BranchRule::AssumedUnibranch),
        ),
        None => (BranchRule::Exact, None),
    };
    let counts = match corr {
        Some(c) => apply_corrections(&plane_counts, &c),
        None => Vec::new(),
    };
    assert!(
        !(status.is_decided() && rule == BranchRule::AssumedUnibranch),
        "a Decided status with assumed-unibranch counts"
    );
    Ok(NormalizationCounts {
        p,
        genus,
        counts,
        plane_counts,
        status,
        locus,
        branch_rule: rule,
    })
}
