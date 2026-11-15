//! The reduction certificate and the point counts of the normalization.
//!
//! Theorem (WP-F design, Theorem 4.5). Let `F` be primitive of bidegree `(dx, dy)`, `C` the
//! normalization of `V(F)` over `Q`, of genus `g`, and `X0_p = V(F mod p)`. If `X0_p` has finitely
//! many singular points over `F_p-bar` and the sum of their delta invariants is
//! `p_a - g`, `p_a = (dx - 1)(dy - 1)`, then `C` has good reduction at p: the normalization of
//! `X0_p` is smooth, projective and geometrically connected of genus `g`, it is the special fibre
//! of a smooth proper model of `C` over `Z_(p)`, and its point counts are those of the reduction
//! of `C`. Proof outline: `Y0 = V(F)` over `Z_(p)` is flat; its normalization `Y` has
//! `p_a(Y_p) = g` by flatness; `Y_p` is reduced (R0 and S1); `Y_p -> X0_p` is finite birational,
//! so `p_a(X0_p) = p_a(normalization) + delta`, which forces `delta(Y_p) = 0`: `Y_p` is normal,
//! hence smooth, and connected by Zariski (R. Hartshorne, *Algebraic geometry*, III.9, III.11;
//! Q. Liu, *Algebraic geometry and arithmetic curves*, 7.5, 10.1; W. Fulton, *Algebraic curves*).
//! The same holds at a degree-one prime of a number field. A delta-sum below `p_a - g` is
//! impossible (`delta(Y_p) >= 0`), so it means a wrong genus input; above it, nothing is
//! certified.
//!
//! Statuses follow the project rule: `Decided` only for the nodal certificate (every singular
//! point a node, `p_a - g` of them). A model whose non-nodal points are classified exactly with
//! delta-sum `p_a - g` is `Unresolved(DeltaSumCertified)`: Theorem 4.5 applies verbatim, but
//! promoting it to `Decided` is the owner's decision (follow-up F-9 of the WP-F report).
//!
//! Counts: `N_k = #X0_p(F_q) + sum over closed singular points P with d_P | k of
//! d_P (b_P(k) - 1)`, with `b_P(k)` the rational branches above one geometric point
//! ([`SingularPoint::rational_branches`]). The reference rule of `curve.py` counts every
//! non-nodal point once ([`BranchRule::AssumedUnibranch`]); it is exact for `A_k` with `k`
//! even and wrong otherwise.

use super::fibres::{plane_point_count, Fibration};
use super::local::LocalType;
use super::singular::{singular_locus, SingularLocus, SingularPoint};
use super::{CountError, FormModP};
use crate::weil::{FrobeniusPolynomial, WeilError};
use rustmath_integers::Integer;

/// A proof that the curve has good reduction at p and that the counts are those of its
/// reduction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Certificate {
    /// Every singular point of `V(F mod p)` is a node and there are `p_a - g` of them over
    /// `F_p-bar`. Relative to `genus_hypothesis = g`, a trusted input (with a genus below the
    /// true one a bad prime can be certified).
    Nodal {
        nodes: usize,
        genus_hypothesis: usize,
    },
}

/// Why no certificate is given.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unresolved {
    /// Non-nodal singularities classified exactly, delta-sum certified: every singular point is
    /// a node, an `A_k` or an ordinary m-fold point and the delta-sum is `p_a - g`. Theorem 4.5
    /// applies, but by the project rule a model with a non-nodal singular point is not Decided.
    /// The counts use the exact branch rule.
    DeltaSumCertified {
        delta_sum: usize,
        genus_hypothesis: usize,
        non_nodal: Vec<SingularPoint>,
    },
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

impl ReductionStatus {
    pub fn is_decided(&self) -> bool {
        matches!(self, ReductionStatus::Decided(_))
    }

    /// A short label: `Nodal (62 nodes)`, `Unresolved: ...`.
    pub fn label(&self) -> String {
        match self {
            ReductionStatus::Decided(Certificate::Nodal { nodes, .. }) => {
                format!("Decided: nodal ({nodes} nodes)")
            }
            ReductionStatus::Unresolved(Unresolved::DeltaSumCertified { non_nodal, .. }) => {
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
                        format!("{}{br}", s.local)
                    })
                    .collect();
                format!(
                    "Unresolved: non-nodal singularities classified exactly, delta-sum certified ({})",
                    types.join("; ")
                )
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

/// The status of `locus` for a curve of arithmetic genus `p_a` whose normalization over `Q` has
/// genus `genus` (a trusted input; see [`Certificate::Nodal`]).
///
/// Errors: `GenusInconsistent` when every point is classified and the delta-sum is below
/// `p_a - genus` (impossible for a correct genus).
pub fn certify(
    locus: &SingularLocus,
    arithmetic_genus: usize,
    genus: usize,
) -> Result<ReductionStatus, CountError> {
    if genus > arithmetic_genus {
        return Err(CountError::Shape(format!(
            "genus {genus} above the arithmetic genus {arithmetic_genus}"
        )));
    }
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
        if known + unknown_min > expected {
            return Ok(ReductionStatus::Unresolved(Unresolved::DeltaMismatch {
                found: known + unknown_min,
                expected,
                found_is_lower_bound: true,
            }));
        }
        return Ok(ReductionStatus::Unresolved(Unresolved::Unclassified(
            s.clone(),
        )));
    }
    if known < expected {
        return Err(CountError::GenusInconsistent {
            found: known,
            expected,
        });
    }
    if known > expected {
        return Ok(ReductionStatus::Unresolved(Unresolved::DeltaMismatch {
            found: known,
            expected,
            found_is_lower_bound: false,
        }));
    }
    let non_nodal: Vec<SingularPoint> = locus.non_nodal().into_iter().cloned().collect();
    if non_nodal.is_empty() {
        Ok(ReductionStatus::Decided(Certificate::Nodal {
            nodes: locus.node_count(),
            genus_hypothesis: genus,
        }))
    } else {
        Ok(ReductionStatus::Unresolved(Unresolved::DeltaSumCertified {
            delta_sum: known,
            genus_hypothesis: genus,
            non_nodal,
        }))
    }
}

/// How the branches of a non-nodal singular point are counted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BranchRule {
    /// From the exact local classification.
    Exact,
    /// Every non-nodal singular point counts once (the rule of the reference `curve.py`):
    /// "assumed unibranch". Never combined with a Decided status.
    AssumedUnibranch,
}

/// Options of [`normalization_point_counts`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CountOptions {
    /// The fibration of the plane count (the certificate always fibres in `X`).
    pub fibration: Fibration,
    /// The branch rule requested for non-nodal points. `AssumedUnibranch` reproduces the
    /// reference tables; with `Exact` (the default) the classification is used.
    pub branch_rule: BranchRule,
    /// With `branch_rule = Exact` and an unclassified point: give the counts with the
    /// `AssumedUnibranch` rule (true) or no counts (false, the default).
    pub unibranch_if_unclassified: bool,
    /// Threads for the plane count (`None`: the global rayon pool).
    pub threads: Option<usize>,
}

impl Default for CountOptions {
    fn default() -> Self {
        CountOptions {
            fibration: Fibration::FibresInX,
            branch_rule: BranchRule::Exact,
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
    /// The rule `counts` were made with (`Exact` when there is no non-nodal point).
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

/// Point counts `N_1..N_kmax` of the normalization of `V(F mod p)`, with the reduction status.
///
/// `genus` is the genus of the normalization of `V(F)` over `Q`, a trusted input: the status is
/// relative to it (cite its source at the call site). `F` must have `X`-degree in `2..=4`
/// (transpose otherwise); `p` is the prime of `f`; `kmax <= 8` and `p^kmax < 2^63`.
///
/// Never returns counts with `BranchRule::AssumedUnibranch` and a Decided status.
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
    let p_a = f.arithmetic_genus();
    let locus = singular_locus(f)?;
    let status = certify(&locus, p_a, genus)?;
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
    let has_non_nodal = !locus.non_nodal().is_empty();
    let kmax = kmax as usize;
    let (rule, corr) = match opts.branch_rule {
        BranchRule::AssumedUnibranch if has_non_nodal => (
            BranchRule::AssumedUnibranch,
            corrections(&locus, kmax, BranchRule::AssumedUnibranch),
        ),
        _ => match corrections(&locus, kmax, BranchRule::Exact) {
            Some(c) => (BranchRule::Exact, Some(c)),
            None if opts.unibranch_if_unclassified => (
                BranchRule::AssumedUnibranch,
                corrections(&locus, kmax, BranchRule::AssumedUnibranch),
            ),
            None => (BranchRule::Exact, None),
        },
    };
    let counts = match corr {
        Some(c) => plane_counts
            .iter()
            .zip(c.iter())
            .map(|(&n, &d)| {
                let v = n as i64 + d;
                assert!(v >= 0, "negative count");
                v as u64
            })
            .collect(),
        None => Vec::new(),
    };
    debug_assert!(!(status.is_decided() && rule == BranchRule::AssumedUnibranch));
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
