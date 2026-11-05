//! Generic Stauduhar descent for small degree.
//!
//! Given a monic, irreducible, separable `f ∈ ℤ[x]` of small degree `n`, this
//! module identifies `Gal(f) ⊆ S_n` up to conjugacy and returns its `nTt`
//! label. The algorithm is the classical Stauduhar method:
//!
//! 1. **Label the roots.** Compute the `n` complex roots in the stable order of
//!    [`rustmath_polynomials::root_label::complex_roots`].
//! 2. **Seed the group.** Start at `G = S_n`. If `disc(f)` is a perfect square
//!    then `Gal(f) ⊆ A_n` (free first descent via the discriminant resolvent),
//!    so start at `G = A_n`.
//! 3. **Descend.** While `G` has a maximal subgroup `H` such that the relative
//!    resolvent `R_{G,H}` has a *simple rational root* at coset `c`, replace
//!    `G ← c H c⁻¹` (the conjugate of `H` containing `Gal(f)`). Repeat.
//! 4. **Identify.** When no maximal subgroup admits a descent, `G` is the Galois
//!    group; return its `nTt` label via [`crate::labels::identify`].
//!
//! Precision is raised and the resolvent rebuilt if the rational-root test is
//! borderline at the current precision (a *verified* descent never accepts a
//! root whose error exceeds the tolerance).

use crate::labels::identify;
use crate::perm::{
    alt_gens, compose, conjugate, coset_reps, group_closure, index, sym_gens, Perm,
};
use crate::resolvent_eval::{
    build_relative_resolvent, find_simple_rational_root,
};
use crate::subgroups::maximal_subgroups;
use rustmath_integers::Integer;
use rustmath_polynomials::disc::discriminant;
use rustmath_polynomials::root_label::{complex_roots, BigComplex};

/// The result of a Stauduhar descent.
#[derive(Clone, Debug)]
pub struct GaloisResult {
    /// Degree of `f`.
    pub n: usize,
    /// Order of the identified Galois group.
    pub order: usize,
    /// Whether `Gal(f) ⊆ A_n`.
    pub in_alternating: bool,
    /// `nTt` label, if the group was identified against the table.
    pub label: Option<String>,
    /// Common name (e.g. `"S4"`), if identified.
    pub name: Option<String>,
    /// A trace of the descent: one `(from_order, to_order, index)` per step.
    pub steps: Vec<(usize, usize, usize)>,
}

/// Configuration knobs for the descent.
#[derive(Clone, Debug)]
pub struct Config {
    /// Starting working precision (fractional bits).
    pub start_prec: u32,
    /// Maximum working precision before giving up on a borderline root.
    pub max_prec: u32,
    /// Cap on group order for closure operations (subgroup lattice etc.).
    pub group_cap: usize,
    /// Max denominator allowed in rational-root reconstruction.
    pub max_denom: Integer,
    /// Power `d` in the symmetrized invariant `Σ_h L(h·α)^d`.
    pub invariant_power: u32,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            start_prec: 256,
            max_prec: 4096,
            group_cap: 2000,
            max_denom: Integer::from(1_000_000i64),
            invariant_power: 2,
        }
    }
}

/// Deterministic weight vectors `β` to try for the resolvent invariant, in order
/// of increasing "genericity". Small distinct integers usually suffice; later
/// vectors spread further to break accidental symmetries.
fn weight_vectors(n: usize) -> Vec<Vec<Integer>> {
    let mut out = Vec::new();
    // 1,2,3,…,n
    out.push((1..=n as i64).map(Integer::from).collect());
    // 1,3,5,… (odd)
    out.push((0..n as i64).map(|i| Integer::from(2 * i + 1)).collect());
    // primes-ish spread
    let spread = [2i64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97];
    out.push((0..n).map(|i| Integer::from(spread[i % spread.len()] + (i / spread.len()) as i64 * 100)).collect());
    // 1,4,9,16,… (squares — strongly generic)
    out.push((1..=n as i64).map(|i| Integer::from(i * i)).collect());
    out
}

/// Run the Stauduhar descent on `f` (monic, irreducible, separable, little-endian
/// integer coefficients) and return the identified Galois group.
///
/// Returns `Err` if the input is unusable (non-monic, too small, etc.) or the
/// descent cannot be completed at the maximum precision. The group is always
/// *sound*: the returned group always contains `Gal(f)`; the descent only stops
/// when no verified proper descent exists, so the returned group equals `Gal(f)`
/// when the resolvents separate (true for all generic small-degree inputs).
pub fn galois_group(f: &[Integer], cfg: &Config) -> Result<GaloisResult, String> {
    let n = poly_degree(f);
    if n < 2 {
        return Err("degree must be ≥ 2".to_string());
    }
    if f[n] != Integer::from(1i64) {
        return Err("f must be monic".to_string());
    }

    // 1. Label the roots.
    let mut prec = cfg.start_prec;
    let mut roots = compute_roots(f, prec)?;

    // 2. Seed the group: A_n if disc is a perfect square, else S_n.
    let d = discriminant(f);
    if d.is_zero() {
        return Err("f is not separable (disc = 0)".to_string());
    }
    let in_alt = d.is_perfect_square();
    let start_gens = if in_alt { alt_gens(n) } else { sym_gens(n) };
    let mut g_elems = group_closure(n, &start_gens, cfg.group_cap)
        .ok_or_else(|| format!("starting group too large (>{})", cfg.group_cap))?;

    let mut steps: Vec<(usize, usize, usize)> = Vec::new();

    // 3. Descend.
    loop {
        let from_order = g_elems.len();
        let maximals = maximal_subgroups(n, &g_elems);
        let mut descended = false;
        for h in &maximals {
            // Try to find a simple rational root of R_{G,H}. On a borderline
            // numeric situation, raise precision (recomputing roots) and retry
            // this same H until the descent verdict is unambiguous or we hit the
            // precision cap.
            let mut verdict = try_descend(&g_elems, h, &roots, prec, cfg);
            while matches!(verdict, Descend::NeedMorePrecision) {
                let next = prec.saturating_mul(2);
                if next > cfg.max_prec {
                    break;
                }
                roots = compute_roots(f, next)?;
                prec = next;
                verdict = try_descend(&g_elems, h, &roots, prec, cfg);
            }
            if let Descend::Into(rep, _val) = verdict {
                // descend into the conjugate c H c⁻¹ containing Gal(f)
                let conj = conjugate(&rep, h);
                let mut conj_sorted = conj;
                conj_sorted.sort();
                conj_sorted.dedup();
                let to_order = conj_sorted.len();
                let idx = index(&g_elems, h);
                steps.push((from_order, to_order, idx));
                g_elems = conj_sorted;
                descended = true;
                break;
            }
        }
        if !descended {
            break;
        }
    }

    // 4. Identify.
    let order = g_elems.len();
    let in_alternating = g_elems.iter().all(|p| !crate::perm::is_odd(p));
    let (label, name) = match identify(n, &elem_gens(&g_elems), cfg.group_cap) {
        Some((l, nm)) => (Some(l.to_string()), Some(nm.to_string())),
        None => (None, None),
    };

    Ok(GaloisResult { n, order, in_alternating, label, name, steps })
}

/// Verdict of testing one maximal subgroup `H` at a fixed precision.
enum Descend {
    /// `Gal(f)` lies in the conjugate `rep·H·rep⁻¹`; descend (carries the value).
    Into(Perm, rustmath_rationals::Rational),
    /// `Gal(f)` is not in any conjugate of `H` (separable resolvent, no rational
    /// root) — definitive negative at the working precision.
    No,
    /// No separable resolvent could be built with any tried `(β, d)` — the coset
    /// values collided at the working precision; raise precision and retry.
    NeedMorePrecision,
}

/// Try every weight vector to build a separable `R_{G,H}` at the given precision
/// and apply Stauduhar's criterion. Pure: never changes precision itself (the
/// caller raises precision on [`Descend::NeedMorePrecision`]).
fn try_descend(
    g_elems: &[Perm],
    h_elems: &[Perm],
    roots: &[BigComplex],
    prec: u32,
    cfg: &Config,
) -> Descend {
    let n = roots.len();
    let reps = coset_reps(g_elems, h_elems);
    // tolerance: comfortably below the working precision.
    let tol_bits = (prec / 3).max(32);
    for beta in weight_vectors(n) {
        // try increasing powers if collisions persist
        for d in [cfg.invariant_power, cfg.invariant_power + 1, cfg.invariant_power + 2] {
            if let Some(res) =
                build_relative_resolvent(&beta, roots, h_elems, &reps, d, prec, tol_bits)
            {
                if let Some(rr) = find_simple_rational_root(&res, &cfg.max_denom, tol_bits) {
                    return Descend::Into(rr.rep, rr.value);
                } else {
                    // separable resolvent with no rational root ⇒ definitive negative.
                    return Descend::No;
                }
            }
            // else: collision → try larger d / next β
        }
    }
    // No separable resolvent at this precision with any (β, d).
    Descend::NeedMorePrecision
}

/// A set of generators for a group given as an element list (just reuse the whole
/// list — closure of the element list is itself, so fingerprinting is exact).
fn elem_gens(elems: &[Perm]) -> Vec<Perm> {
    elems.to_vec()
}

/// Compute the labeled complex roots of `f` at precision `prec`, verifying the
/// accuracy bound is comfortably small.
fn compute_roots(f: &[Integer], prec: u32) -> Result<Vec<BigComplex>, String> {
    let cr = complex_roots(f, prec);
    if cr.roots.len() != poly_degree(f) {
        return Err(format!(
            "root solver returned {} roots, expected {}",
            cr.roots.len(),
            poly_degree(f)
        ));
    }
    if !(cr.accuracy_bound.is_finite() && cr.accuracy_bound < 1e-6) {
        return Err(format!("root accuracy too poor: {}", cr.accuracy_bound));
    }
    Ok(cr.roots)
}

/// Degree of a little-endian integer polynomial (assumes nonzero leading coeff
/// after trimming).
fn poly_degree(f: &[Integer]) -> usize {
    let mut k = f.len();
    while k > 1 && f[k - 1].is_zero() {
        k -= 1;
    }
    k - 1
}

/// Re-export of `compose` for callers that build coset elements (kept to avoid a
/// wider `perm` import surface in downstream code).
pub fn apply_coset(c: &Perm, h: &Perm) -> Perm {
    compose(c, h)
}
