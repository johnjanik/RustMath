//! Small-degree transitive-group tables and a conjugacy-invariant identifier.
//!
//! For degrees 3, 4, 5 we tabulate every transitive subgroup of `S_n` with its
//! standard `nTt` label and a set of generators (in this crate's generic
//! `Vec<usize>` permutation type). The Stauduhar descent stabilises `Gal(f)` at
//! a concrete subgroup `G ⊆ S_n`; [`identify`] maps that group to its `nTt`
//! label by a **conjugacy-invariant fingerprint** — order, parity, the full
//! cycle-type set, and the `k`-subset orbit-length signatures — which separates
//! every transitive group of these degrees.
//!
//! (`rustmath_groups::pgalois::transitive_groups` tabulates the same degrees but
//! in the fixed `[u8; 24]` atlas permutation type; here we keep an independent,
//! degree-`n`-native copy so the descent never has to pad to 24 points.)

use crate::perm::{cycle_type, from_cycles, group_closure, is_odd, Perm};
use std::collections::BTreeSet;

/// A named transitive group `nTt`.
#[derive(Clone, Debug)]
pub struct NamedGroup {
    /// LMFDB-style label, e.g. `"4T3"`.
    pub label: &'static str,
    /// Common name, e.g. `"D4"`.
    pub name: &'static str,
    /// Group order.
    pub order: usize,
    /// Generators (degree-`n` permutations).
    pub gens: Vec<Perm>,
}

/// The transitive subgroups of `S_n` for the tabulated degrees (3, 4, 5).
pub fn transitive_groups(n: usize) -> Vec<NamedGroup> {
    let c = |cycles: &[&[usize]]| -> Perm {
        from_cycles(n, &cycles.iter().map(|x| x.to_vec()).collect::<Vec<_>>())
    };
    match n {
        3 => vec![
            NamedGroup { label: "3T1", name: "C3", order: 3, gens: vec![c(&[&[0, 1, 2]])] },
            NamedGroup {
                label: "3T2",
                name: "S3",
                order: 6,
                gens: vec![c(&[&[0, 1, 2]]), c(&[&[0, 1]])],
            },
        ],
        4 => vec![
            NamedGroup { label: "4T1", name: "C4", order: 4, gens: vec![c(&[&[0, 1, 2, 3]])] },
            NamedGroup {
                label: "4T2",
                name: "V4",
                order: 4,
                gens: vec![c(&[&[0, 1], &[2, 3]]), c(&[&[0, 2], &[1, 3]])],
            },
            NamedGroup {
                label: "4T3",
                name: "D4",
                order: 8,
                gens: vec![c(&[&[0, 1, 2, 3]]), c(&[&[0, 2]])],
            },
            NamedGroup {
                label: "4T4",
                name: "A4",
                order: 12,
                gens: vec![c(&[&[0, 1, 2]]), c(&[&[1, 2, 3]])],
            },
            NamedGroup {
                label: "4T5",
                name: "S4",
                order: 24,
                gens: vec![c(&[&[0, 1, 2, 3]]), c(&[&[0, 1]])],
            },
        ],
        5 => vec![
            NamedGroup { label: "5T1", name: "C5", order: 5, gens: vec![c(&[&[0, 1, 2, 3, 4]])] },
            NamedGroup {
                label: "5T2",
                name: "D5",
                order: 10,
                gens: vec![c(&[&[0, 1, 2, 3, 4]]), c(&[&[1, 4], &[2, 3]])],
            },
            NamedGroup {
                label: "5T3",
                name: "F20",
                order: 20,
                gens: vec![c(&[&[0, 1, 2, 3, 4]]), c(&[&[1, 2, 4, 3]])],
            },
            NamedGroup {
                label: "5T4",
                name: "A5",
                order: 60,
                gens: vec![c(&[&[0, 1, 2, 3, 4]]), c(&[&[2, 3, 4]])],
            },
            NamedGroup {
                label: "5T5",
                name: "S5",
                order: 120,
                gens: vec![c(&[&[0, 1, 2, 3, 4]]), c(&[&[0, 1]])],
            },
        ],
        _ => Vec::new(),
    }
}

/// A conjugacy-invariant fingerprint of a permutation group, used to match an
/// abstractly-stabilised group to a tabulated `nTt`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Fingerprint {
    pub n: usize,
    pub order: usize,
    pub in_alternating: bool,
    pub cycle_types: BTreeSet<Vec<usize>>,
    /// `k`-subset orbit-length signatures for `k = 1 .. ⌊n/2⌋`.
    pub ksubset_sigs: Vec<Vec<usize>>,
}

/// Compute the fingerprint of `⟨gens⟩` on `n` points. `cap` bounds the closure.
pub fn fingerprint(n: usize, gens: &[Perm], cap: usize) -> Option<Fingerprint> {
    let elems = group_closure(n, gens, cap)?;
    let order = elems.len();
    let in_alternating = elems.iter().all(|p| !is_odd(p));
    let cycle_types: BTreeSet<Vec<usize>> = elems.iter().map(cycle_type).collect();
    let ksubset_sigs: Vec<Vec<usize>> = (1..=n / 2)
        .map(|k| ksubset_orbit_lengths(n, gens, k))
        .collect();
    Some(Fingerprint { n, order, in_alternating, cycle_types, ksubset_sigs })
}

/// Sorted orbit lengths of `⟨gens⟩` on `k`-subsets of `{0,…,n−1}` (small `n`).
fn ksubset_orbit_lengths(n: usize, gens: &[Perm], k: usize) -> Vec<usize> {
    use std::collections::HashSet;
    // enumerate k-subsets as sorted index vectors
    let mut subsets: Vec<Vec<usize>> = Vec::new();
    let mut idx: Vec<usize> = (0..k).collect();
    if k == 0 || k > n {
        return vec![];
    }
    loop {
        subsets.push(idx.clone());
        // next combination
        let mut i = k;
        while i > 0 {
            i -= 1;
            if idx[i] != i + n - k {
                idx[i] += 1;
                for j in (i + 1)..k {
                    idx[j] = idx[j - 1] + 1;
                }
                break;
            }
            if i == 0 {
                // done
                i = usize::MAX;
                break;
            }
        }
        if i == usize::MAX {
            break;
        }
    }
    let apply = |g: &Perm, s: &[usize]| -> Vec<usize> {
        let mut t: Vec<usize> = s.iter().map(|&x| g[x]).collect();
        t.sort_unstable();
        t
    };
    let mut visited: HashSet<Vec<usize>> = HashSet::new();
    let mut lengths = Vec::new();
    for start in &subsets {
        if visited.contains(start) {
            continue;
        }
        let mut stack = vec![start.clone()];
        visited.insert(start.clone());
        let mut count = 0usize;
        while let Some(cur) = stack.pop() {
            count += 1;
            for g in gens {
                let nxt = apply(g, &cur);
                if visited.insert(nxt.clone()) {
                    stack.push(nxt);
                }
            }
        }
        lengths.push(count);
    }
    lengths.sort_unstable();
    lengths
}

/// Identify the transitive group `⟨gens⟩ ⊆ S_n` as a `(label, name)` pair by
/// matching its conjugacy-invariant fingerprint against the table. Returns
/// `None` if `n` is not tabulated, the group is not transitive, or no table
/// entry matches.
pub fn identify(n: usize, gens: &[Perm], cap: usize) -> Option<(&'static str, &'static str)> {
    let fp = fingerprint(n, gens, cap)?;
    for g in transitive_groups(n) {
        if let Some(gfp) = fingerprint(n, &g.gens, cap) {
            if gfp == fp {
                return Some((g.label, g.name));
            }
        }
    }
    None
}
