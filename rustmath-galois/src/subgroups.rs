//! Subgroup lattice utilities for the Stauduhar descent at *small* degree.
//!
//! Generic Stauduhar needs, at each node of the descent, the **maximal
//! subgroups** of the current group `G ⊆ S_n`. For small `n` (`|G|` up to a few
//! hundred) we compute the full subgroup lattice directly from the element list
//! and read off the maximal ones. This is exponential in general but entirely
//! adequate for the known-answer validation degrees (3, 4, 5; `|S_5| = 120`).
//!
//! For large degrees (24) the full lattice is infeasible; the degree-24 descent
//! in [`crate::deg24`] does **not** use this module — it descends through the
//! transitive-group atlas using resolvent orbit signatures instead.

use crate::perm::{compose, group_closure, identity, is_subset, Perm};
use std::collections::{BTreeSet, HashSet};

/// Closure of a set of elements under composition (they must be drawn from a
/// finite group, so this terminates). Returns a sorted, deduped element list.
pub fn closure_of(n: usize, seed: &[Perm], cap: usize) -> Option<Vec<Perm>> {
    group_closure(n, seed, cap)
}

/// Enumerate **all** subgroups of `G` (given as its full element list), each as
/// a sorted element list. Uses cyclic extension: start from the trivial
/// subgroup, repeatedly adjoin one group element to an existing subgroup and
/// close, collecting every distinct subgroup reached. Correct for finite
/// groups; intended for `|G|` up to a few hundred.
pub fn all_subgroups(n: usize, group: &[Perm]) -> Vec<Vec<Perm>> {
    let cap = group.len() + 1;
    let id = vec![identity(n)];
    let mut found: HashSet<Vec<Perm>> = HashSet::new();
    found.insert(id.clone());
    // worklist of subgroups to extend
    let mut work: Vec<Vec<Perm>> = vec![id];
    while let Some(h) = work.pop() {
        if h.len() == group.len() {
            continue; // G itself: nothing strictly larger to find from here
        }
        let hset: HashSet<&Perm> = h.iter().collect();
        for g in group {
            if hset.contains(g) {
                continue;
            }
            // adjoin g
            let mut seed = h.clone();
            seed.push(g.clone());
            if let Some(mut bigger) = group_closure(n, &seed, cap) {
                bigger.sort();
                bigger.dedup();
                if found.insert(bigger.clone()) {
                    work.push(bigger);
                }
            }
        }
    }
    let mut out: Vec<Vec<Perm>> = found.into_iter().collect();
    out.sort_by_key(|s| s.len());
    out
}

/// The **maximal** subgroups of `G` (proper subgroups not contained in any other
/// proper subgroup), each as a sorted element list. Conjugate maximal subgroups
/// are all returned (the descent dedupes by conjugacy implicitly through the
/// resolvent test).
pub fn maximal_subgroups(n: usize, group: &[Perm]) -> Vec<Vec<Perm>> {
    let subs = all_subgroups(n, group);
    let g_order = group.len();
    let proper: Vec<&Vec<Perm>> = subs.iter().filter(|s| s.len() < g_order).collect();
    let mut maximal: Vec<Vec<Perm>> = Vec::new();
    for (i, s) in proper.iter().enumerate() {
        // s is maximal iff no other proper subgroup strictly contains it.
        let mut is_max = true;
        for (j, t) in proper.iter().enumerate() {
            if i == j {
                continue;
            }
            if t.len() > s.len() && is_subset(s, t) {
                is_max = false;
                break;
            }
        }
        if is_max {
            maximal.push((*s).clone());
        }
    }
    // deterministic order: by descending size then lexicographically
    maximal.sort_by(|a, b| b.len().cmp(&a.len()).then(a.cmp(b)));
    maximal
}

/// Group multiplication helper used in tests/diagnostics: the element list of
/// `⟨a, b⟩`.
pub fn generated_by(n: usize, a: &Perm, b: &Perm, cap: usize) -> Option<Vec<Perm>> {
    closure_of(n, &[a.clone(), b.clone()], cap)
}

/// The distinct cycle-type multiset of an element list (diagnostic).
pub fn cycle_types(elems: &[Perm]) -> BTreeSet<Vec<usize>> {
    elems.iter().map(crate::perm::cycle_type).collect()
}

/// Compose all of `a` with a fixed `g` on the left (the coset `g·H`).
pub fn left_coset(g: &Perm, h: &[Perm]) -> Vec<Perm> {
    h.iter().map(|hh| compose(g, hh)).collect()
}
