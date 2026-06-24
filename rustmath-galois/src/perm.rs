//! Generic (arbitrary-degree) permutations of `{0, …, n−1}` for the Stauduhar
//! engine.
//!
//! The degree-24 atlas in `rustmath_groups::transitive24` uses a fixed `[u8; 24]`
//! permutation type, which is perfect for that one degree. The Stauduhar descent
//! must work for *any* small degree (3, 4, 5, …) as well as 24, so this module
//! carries its own heap-backed `Vec<usize>` permutation type and the small
//! amount of permutation-group machinery the descent needs: composition,
//! inversion, group closure, parity, cycle types, coset enumeration, and a few
//! named groups (`S_n`, `A_n`, generators for cyclic / dihedral / Klein groups).
//!
//! Convention (matching `rustmath_groups::transitive24::compose`):
//! `compose(a, b)[i] = a[b[i]]` — apply `b` first, then `a`. A permutation `p`
//! is the *image list*: point `i` maps to `p[i]`.

use std::collections::{BTreeSet, HashMap, HashSet};

/// A permutation of `{0, …, n−1}` as an image list (`p[i]` = image of `i`).
pub type Perm = Vec<usize>;

/// The identity permutation on `n` points.
pub fn identity(n: usize) -> Perm {
    (0..n).collect()
}

/// Compose two permutations of the same degree: `(a ∘ b)[i] = a[b[i]]`
/// (apply `b`, then `a`).
pub fn compose(a: &Perm, b: &Perm) -> Perm {
    debug_assert_eq!(a.len(), b.len());
    b.iter().map(|&bi| a[bi]).collect()
}

/// The inverse permutation.
pub fn inverse(p: &Perm) -> Perm {
    let mut inv = vec![0usize; p.len()];
    for (i, &pi) in p.iter().enumerate() {
        inv[pi] = i;
    }
    inv
}

/// Build a permutation of degree `n` from disjoint cycles given in **0-indexed**
/// points. Points not mentioned are fixed.
pub fn from_cycles(n: usize, cycles: &[Vec<usize>]) -> Perm {
    let mut p = identity(n);
    for cyc in cycles {
        let m = cyc.len();
        for k in 0..m {
            p[cyc[k]] = cyc[(k + 1) % m];
        }
    }
    p
}

/// Cycle type: sorted (ascending) multiset of cycle lengths, including fixed
/// points (1-cycles); sums to `n`.
pub fn cycle_type(p: &Perm) -> Vec<usize> {
    let n = p.len();
    let mut seen = vec![false; n];
    let mut ct = Vec::new();
    for i in 0..n {
        if !seen[i] {
            let mut len = 0usize;
            let mut j = i;
            while !seen[j] {
                seen[j] = true;
                j = p[j];
                len += 1;
            }
            ct.push(len);
        }
    }
    ct.sort_unstable();
    ct
}

/// Parity: `true` if `p` is an odd permutation. `sign = (−1)^(n − #cycles)`.
pub fn is_odd(p: &Perm) -> bool {
    let n = p.len();
    let mut seen = vec![false; n];
    let mut cycles = 0usize;
    for i in 0..n {
        if !seen[i] {
            cycles += 1;
            let mut j = i;
            while !seen[j] {
                seen[j] = true;
                j = p[j];
            }
        }
    }
    (n - cycles) % 2 == 1
}

/// Enumerate `⟨gens⟩` (BFS closure). Returns `None` if it exceeds `cap` elements
/// (so callers can detect groups too large to materialise). The identity is
/// always present even for an empty generator list.
pub fn group_closure(n: usize, gens: &[Perm], cap: usize) -> Option<Vec<Perm>> {
    let id = identity(n);
    let mut set: HashSet<Perm> = HashSet::new();
    set.insert(id.clone());
    let mut frontier = vec![id];
    while let Some(g) = frontier.pop() {
        for s in gens {
            let h = compose(s, &g);
            if set.insert(h.clone()) {
                if set.len() > cap {
                    return None;
                }
                frontier.push(h);
            }
        }
    }
    let mut out: Vec<Perm> = set.into_iter().collect();
    out.sort();
    Some(out)
}

/// Order of `⟨gens⟩` (number of elements), or `None` if it exceeds `cap`.
pub fn group_order(n: usize, gens: &[Perm], cap: usize) -> Option<usize> {
    group_closure(n, gens, cap).map(|g| g.len())
}

/// Symmetric group `S_n` generators: the transposition `(0 1)` and the
/// `n`-cycle `(0 1 … n−1)`.
pub fn sym_gens(n: usize) -> Vec<Perm> {
    if n <= 1 {
        return vec![identity(n)];
    }
    let transp = from_cycles(n, &[vec![0, 1]]);
    let cyc = from_cycles(n, &[(0..n).collect()]);
    vec![transp, cyc]
}

/// Alternating group `A_n` generators: all 3-cycles `(0 1 k)` for `k = 2..n`.
/// (For `n ≤ 2`, `A_n` is trivial.)
pub fn alt_gens(n: usize) -> Vec<Perm> {
    if n <= 2 {
        return vec![identity(n)];
    }
    (2..n).map(|k| from_cycles(n, &[vec![0, 1, k]])).collect()
}

/// Full element list of `S_n` (Heap-free lexicographic enumeration). Only call
/// for small `n` (`n! ≤ a few thousand`); used for small-degree descent.
pub fn sym_elements(n: usize) -> Vec<Perm> {
    let mut out = Vec::new();
    let mut p: Perm = (0..n).collect();
    loop {
        out.push(p.clone());
        // next lexicographic permutation
        if n < 2 {
            break;
        }
        let mut i = n - 1;
        while i > 0 && p[i - 1] >= p[i] {
            i -= 1;
        }
        if i == 0 {
            break;
        }
        let mut j = n - 1;
        while p[j] <= p[i - 1] {
            j -= 1;
        }
        p.swap(i - 1, j);
        p[i..].reverse();
    }
    out
}

/// All left cosets `g·H` of `H = elements(subgroup)` inside
/// `G = elements(group)`, returned as one representative per coset. The
/// representative is the lexicographically smallest element of the coset (so the
/// list is deterministic).
///
/// `subgroup` must actually be a subgroup of `group` (closed, contained); this
/// is not re-verified here for speed but is guaranteed by how the descent calls
/// it.
pub fn coset_reps(group: &[Perm], subgroup: &[Perm]) -> Vec<Perm> {
    let h: BTreeSet<Perm> = subgroup.iter().cloned().collect();
    let mut assigned: HashSet<Perm> = HashSet::new();
    let mut reps: Vec<Perm> = Vec::new();
    for g in group {
        if assigned.contains(g) {
            continue;
        }
        // coset g·H
        let coset: Vec<Perm> = h.iter().map(|hh| compose(g, hh)).collect();
        // representative = lexicographically smallest member
        let rep = coset.iter().min().unwrap().clone();
        for c in &coset {
            assigned.insert(c.clone());
        }
        reps.push(rep);
    }
    reps.sort();
    reps
}

/// Conjugate subgroup `g H g⁻¹` from a generator/element list (returns the same
/// kind of list as the input).
pub fn conjugate(g: &Perm, h_elems: &[Perm]) -> Vec<Perm> {
    let gi = inverse(g);
    h_elems.iter().map(|h| compose(&compose(g, h), &gi)).collect()
}

/// The set of cycle types occurring in `⟨gens⟩` (deduplicated, sorted). `None`
/// if the group exceeds `cap`.
pub fn cycle_type_set(n: usize, gens: &[Perm], cap: usize) -> Option<BTreeSet<Vec<usize>>> {
    let elems = group_closure(n, gens, cap)?;
    Some(elems.iter().map(cycle_type).collect())
}

/// Test whether every element of `sub` lies in `whole` (membership by lookup).
pub fn is_subset(sub: &[Perm], whole: &[Perm]) -> bool {
    let set: HashSet<&Perm> = whole.iter().collect();
    sub.iter().all(|s| set.contains(s))
}

/// Index `[G : H]` = `|G| / |H|` from explicit element lists.
pub fn index(group: &[Perm], subgroup: &[Perm]) -> usize {
    group.len() / subgroup.len()
}

/// A cheap canonical key for a subgroup (its sorted element list) — used to
/// dedupe conjugate maximal subgroups during descent.
pub fn subgroup_key(elems: &[Perm]) -> Vec<Perm> {
    let mut v: Vec<Perm> = elems.to_vec();
    v.sort();
    v.dedup();
    v
}

/// Build a lookup from element to index for fast membership tests.
pub fn index_map(elems: &[Perm]) -> HashMap<Perm, usize> {
    elems.iter().cloned().enumerate().map(|(i, p)| (p, i)).collect()
}
