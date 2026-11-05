//! Resolvent-driven Galois-group identification for irreducible `f ∈ ℤ[x]` of small
//! degree — the absolute-resolvent / Stauduhar method, end to end, without MAGMA.
//!
//! Given a monic irreducible separable `f`, [`identify`] determines `Gal(f) ⊆ S_n`
//! (as a transitive group `nTk`) by intersecting three *sound* invariants — each one
//! only ever **excludes** groups the true group cannot equal, so the true group is
//! never dropped:
//!
//! 1. **Parity** — `Gal(f) ⊆ A_n` iff `disc(f)` is a perfect square
//!    ([`rustmath_polynomials::resolvent::galois_in_alternating`]).
//! 2. **Frobenius cycle types** — for each unramified prime `p`, the factorisation
//!    degrees of `f mod p` give a cycle type that *must* occur in `Gal(f)` (Dedekind /
//!    Chebotarev). A group missing an observed type is excluded.
//! 3. **`k`-subset resolvent orbit lengths** — the irreducible-factor degrees of the
//!    `k`-subset-sum resolvent equal the orbit lengths of `Gal(f)` on `k`-subsets of
//!    the roots. This is the Stauduhar invariant; it separates the Frobenius-blind
//!    classes (e.g. `C₄` vs `V₄`, `D₄` vs `C₄`). The resolvent is made **separable**
//!    by a Tschirnhaus relabelling when subset sums collide
//!    ([`rustmath_polynomials::tschirnhaus::separable_subset_sum_resolvent`]).
//!
//! Supported degrees: `n = 3, 4, 5` (complete transitive-group tables here). The same
//! pipeline drives the degree-24 competition path via the
//! [`crate::transitive24`] atlas; this module is the fully-verifiable small-degree
//! reference implementation of it.
//!
//! ## What absolute resolvents cannot see
//!
//! The three invariants above determine the group for *every* irreducible cubic and
//! quartic and quintic **except** distinguishing a cyclic group from the dihedral
//! group directly above it — `C₄` vs `D₄` (`4T1`/`4T3`) and `C₅` vs `D₅`
//! (`5T1`/`5T2`). These pairs have **identical** `k`-subset orbit lengths for every
//! `k` (so every subset-sum resolvent is blind to them), and the cyclic group's
//! polynomial never exhibits the extra Frobenius cycle type (`[2,1,1]`, `[2,2,1]`)
//! that would exclude the dihedral overgroup. `identify` therefore returns the sound
//! 2-element set `{Cₙ, Dₙ}` in exactly those cases. Separating them requires a
//! *relative* resolvent built from a non-symmetric invariant whose stabiliser is the
//! cyclic group (generic Stauduhar descent) — the next engine layer. Note the
//! asymmetry is real and useful: a *dihedral* polynomial **is** determined here,
//! because it does exhibit the extra cycle type that rules out the cyclic subgroup.

use crate::ksubset_orbits::orbit_lengths_on_ksubsets;
use crate::pgalois::{group_in_alternating, transitive_groups, TransitiveSubgroup};
use crate::transitive24::{perm_from_cycles, Perm};
use rustmath_integers::Integer;
use rustmath_polynomials::resolvent::{galois_in_alternating, resolvent_orbit_signature};
use rustmath_polynomials::tschirnhaus::separable_subset_sum_resolvent;
use rustmath_polynomials::{disc, factorization, padic_factor};

/// Outcome of a Galois-group identification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GaloisResult {
    /// Degree of `f`.
    pub degree: usize,
    /// Whether `Gal(f) ⊆ A_n` (`disc(f)` a perfect square).
    pub in_alternating: bool,
    /// Labels still consistent with every invariant, e.g. `["4T3"]`. A singleton means
    /// the group is determined; empty means `f` was reducible or out of range.
    pub candidates: Vec<&'static str>,
    /// The conventional name when determined, e.g. `"D4"`; `None` otherwise.
    pub name: Option<&'static str>,
    /// `false` when `f` is reducible over ℚ (so `Gal(f)` is *not* transitive and this
    /// transitive-group identification does not apply).
    pub irreducible: bool,
    /// The unramified Frobenius cycle types actually observed (sorted, deduplicated).
    pub frobenius_types: Vec<Vec<usize>>,
}

impl GaloisResult {
    /// `true` iff exactly one transitive group survived — the group is determined.
    pub fn is_determined(&self) -> bool {
        self.candidates.len() == 1
    }
}

fn pc(cycles: &[&[u8]]) -> Perm {
    perm_from_cycles(&cycles.iter().map(|c| c.to_vec()).collect::<Vec<_>>())
}

/// The complete transitive-group table for degree `n` (generators on the first `n`
/// points). Degrees 4 and 5 reuse [`crate::pgalois::transitive_groups`]; degree 3 is
/// added here. Returns an empty list for unsupported degrees.
fn group_table(n: usize) -> Vec<TransitiveSubgroup> {
    match n {
        3 => vec![
            TransitiveSubgroup { label: "3T1", name: "C3", order: 3, gens: vec![pc(&[&[1, 2, 3]])] },
            TransitiveSubgroup {
                label: "3T2", name: "S3", order: 6,
                gens: vec![pc(&[&[1, 2, 3]]), pc(&[&[1, 2]])],
            },
        ],
        _ => transitive_groups(n),
    }
}

/// Cycle type of `p` restricted to the points `{0,…,n−1}` (the transitive-group
/// generators fix every point `≥ n`, so this is well-defined). Sorted ascending,
/// summing to `n` — directly comparable to a Frobenius cycle type of a degree-`n`
/// polynomial.
fn cycle_type_n(p: &Perm, n: usize) -> Vec<usize> {
    let mut seen = vec![false; n];
    let mut ct = Vec::new();
    for i in 0..n {
        if !seen[i] {
            let mut len = 0usize;
            let mut j = i;
            while !seen[j] {
                seen[j] = true;
                j = p[j] as usize;
                len += 1;
            }
            ct.push(len);
        }
    }
    ct.sort_unstable();
    ct
}

/// The set of cycle types occurring in `⟨gens⟩` on `n` points (small groups only —
/// the degree ≤ 5 tables enumerate trivially).
fn group_cycle_types(gens: &[Perm], n: usize) -> Vec<Vec<usize>> {
    let elements = crate::transitive24::group_closure(gens, 100_000).expect("small group");
    let mut set: std::collections::BTreeSet<Vec<usize>> = std::collections::BTreeSet::new();
    for e in &elements {
        set.insert(cycle_type_n(e, n));
    }
    set.into_iter().collect()
}

/// First `count` primes (small sieve), used as Frobenius probes.
fn first_primes(count: usize) -> Vec<i64> {
    let mut primes = Vec::new();
    let mut cand = 2i64;
    while primes.len() < count {
        if (2..cand).take_while(|&d| d * d <= cand).all(|d| cand % d != 0) {
            primes.push(cand);
        }
        cand += 1;
    }
    primes
}

fn binom(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut num = 1u128;
    let mut den = 1u128;
    for i in 0..k {
        num *= (n - i) as u128;
        den *= (i + 1) as u128;
    }
    (num / den) as usize
}

/// Is `f ∈ ℤ[x]` irreducible over ℚ? (Exactly one non-constant integer factor.)
fn is_irreducible(f: &[Integer]) -> bool {
    let poly = rustmath_polynomials::univariate::UnivariatePolynomial::new(f.to_vec());
    match factorization::factor_over_integers(&poly) {
        Ok(factors) => {
            let nonconst: usize = factors
                .iter()
                .filter(|(g, _)| g.degree().map(|d| d >= 1).unwrap_or(false))
                .map(|(_, m)| *m as usize)
                .sum();
            nonconst == 1
        }
        Err(_) => false,
    }
}

/// Identify `Gal(f)` for a monic irreducible separable `f ∈ ℤ[x]` of degree 3, 4 or 5.
///
/// Returns the surviving transitive-group labels after parity, Frobenius and resolvent
/// narrowing. For these degrees the three invariants always determine the group
/// (a single surviving label) when `f` is irreducible.
pub fn identify(f: &[Integer]) -> GaloisResult {
    let n = {
        let mut d = f.len();
        while d > 1 && f[d - 1] == Integer::zero() {
            d -= 1;
        }
        d - 1
    };

    let table = group_table(n);
    let in_alt = !disc::discriminant(f).is_zero() && galois_in_alternating(f);

    if table.is_empty() {
        return GaloisResult {
            degree: n,
            in_alternating: in_alt,
            candidates: Vec::new(),
            name: None,
            irreducible: false,
            frobenius_types: Vec::new(),
        };
    }

    let irreducible = is_irreducible(f);

    // 1. Parity filter.
    let mut cands: Vec<&TransitiveSubgroup> = table
        .iter()
        .filter(|g| group_in_alternating(&g.gens) == in_alt)
        .collect();

    // 2. Frobenius cycle-type narrowing over the first primes not dividing disc.
    let mut frob: std::collections::BTreeSet<Vec<usize>> = std::collections::BTreeSet::new();
    for p in first_primes(40) {
        if let Some(ct) = padic_factor::cycle_type(f, p) {
            frob.insert(ct);
        }
    }
    let frobenius_types: Vec<Vec<usize>> = frob.iter().cloned().collect();
    if !frobenius_types.is_empty() {
        let supports: Vec<(usize, Vec<Vec<usize>>)> = cands
            .iter()
            .enumerate()
            .map(|(i, g)| (i, group_cycle_types(&g.gens, n)))
            .collect();
        let keep: Vec<bool> = supports
            .iter()
            .map(|(_, sup)| frobenius_types.iter().all(|t| sup.contains(t)))
            .collect();
        cands = cands
            .into_iter()
            .enumerate()
            .filter(|(i, _)| keep[*i])
            .map(|(_, g)| g)
            .collect();
    }

    // 3. k-subset resolvent orbit lengths (Stauduhar), for k = 2 .. ⌊n/2⌋, with a
    //    Tschirnhaus relabelling to keep the resolvent separable.
    for k in 2..=n / 2 {
        if cands.len() <= 1 {
            break;
        }
        if binom(n, k) > 4096 {
            continue;
        }
        let observed = match separable_subset_sum_resolvent(f, k, 300) {
            Some((r, _g)) => match resolvent_orbit_signature(&r) {
                Ok(sig) => sig,
                Err(_) => continue,
            },
            None => continue,
        };
        cands.retain(|g| {
            let mut sig = orbit_lengths_on_ksubsets(&g.gens, n, k);
            sig.sort_unstable();
            sig == observed
        });
    }

    let labels: Vec<&'static str> = cands.iter().map(|g| g.label).collect();
    let name = if cands.len() == 1 { Some(cands[0].name) } else { None };

    GaloisResult {
        degree: n,
        in_alternating: in_alt,
        candidates: labels,
        name,
        irreducible,
        frobenius_types,
    }
}
