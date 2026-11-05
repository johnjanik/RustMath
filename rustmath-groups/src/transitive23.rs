//! Degree-23 transitive-group classifier ("galois23") via Burnside's prime-degree
//! dichotomy — the identifier for the `1 + 23` residual after specialization.
//!
//! Because 23 is prime, a transitive group `G ≤ S₂₃` is either **solvable** (hence a
//! subgroup of the affine group `AGL(1,23) = 23:22`, a Frobenius group `C₂₃ ⋊ C_d`
//! with `d ∣ 22`) or **2-transitive** (Burnside). The seven transitive groups of
//! degree 23 are therefore:
//!
//! | group        | order      | ⊆ A₂₃ | distinctive cycle types                       |
//! |--------------|-----------:|:-----:|-----------------------------------------------|
//! | `C23`        | 23         |  yes  | `1²³`, `23`                                    |
//! | `D23 = 23:2` | 46         |  no   | `+ 2¹¹1`                                       |
//! | `F23 = 23:11`| 253        |  yes  | `+ 11²1`                                       |
//! | `AGL(1,23)`  | 506        |  no   | `+ 2¹¹1, 11²1, 22·1`                           |
//! | `M23`        | 10 200 960 |  yes  | rich 12-type fingerprint (below)              |
//! | `A23`        | 23!/2      |  yes  | every **even** cycle type                     |
//! | `S23`        | 23!        |  no   | every cycle type                              |
//!
//! Their cycle-type **sets** are pairwise distinct and, within each parity class,
//! form a containment chain:
//! `C23 ⊂ F23 ⊂ M23 ⊂ A23` (all even) and `D23 ⊂ AGL(1,23) ⊂ S23` (odd present).
//!
//! Classification (given the observed Frobenius cycle types and the
//! discriminant-square bit `Gal ⊆ A₂₃ ⇔ disc is a square`):
//! 1. the square bit selects the parity class (the even chain or the odd chain);
//! 2. within that chain, the group is the **smallest** whose cycle-type set contains
//!    every observed type (sound: the true group always contains its Frobenius types;
//!    sharp once Chebotarev has saturated the sample).
//!
//! Modelled on [`crate::transitive24`] (Perm = `[u8; DEG]`, `cycle_type`,
//! `is_odd_type`, `cycle_type_set`, `candidates`) but for `DEG = 23`. The M23
//! fingerprint is the documented ATLAS cycle-type list on 23 points; the four small
//! affine groups' sets are cross-checked against `group_closure` in the tests.

use std::collections::{BTreeSet, HashSet};

pub const DEG: usize = 23;
/// A permutation of `{0,…,22}` (image list).
pub type Perm = [u8; DEG];

pub fn identity() -> Perm {
    let mut p = [0u8; DEG];
    for i in 0..DEG {
        p[i] = i as u8;
    }
    p
}

/// Compose: `(compose(a,b))[i] = a[b[i]]` (apply `b`, then `a`).
pub fn compose(a: &Perm, b: &Perm) -> Perm {
    let mut p = [0u8; DEG];
    for i in 0..DEG {
        p[i] = a[b[i] as usize];
    }
    p
}

/// Build a permutation from disjoint cycles given in **1-indexed** points.
pub fn perm_from_cycles(cycles: &[Vec<u8>]) -> Perm {
    let mut p = identity();
    for cyc in cycles {
        let m = cyc.len();
        for k in 0..m {
            let from = (cyc[k] - 1) as usize;
            let to = (cyc[(k + 1) % m] - 1) as usize;
            p[from] = to as u8;
        }
    }
    p
}

/// Cycle type: sorted (ascending) multiset of cycle lengths, including 1-cycles;
/// sums to 23.
pub fn cycle_type(p: &Perm) -> Vec<usize> {
    let mut seen = [false; DEG];
    let mut ct = Vec::new();
    for i in 0..DEG {
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

/// True if the cycle type is an odd permutation (`sign = (−1)^(23−#cycles)`).
pub fn is_odd_type(ct: &[usize]) -> bool {
    (DEG - ct.len()) % 2 == 1
}

/// Enumerate the group `⟨gens⟩` (BFS closure). `None` if it exceeds `cap` elements.
pub fn group_closure(gens: &[Perm], cap: usize) -> Option<HashSet<Perm>> {
    let mut set = HashSet::new();
    let id = identity();
    set.insert(id);
    let mut frontier = vec![id];
    while let Some(g) = frontier.pop() {
        for s in gens {
            let h = compose(s, &g);
            if set.insert(h) {
                if set.len() > cap {
                    return None;
                }
                frontier.push(h);
            }
        }
    }
    Some(set)
}

/// The set of cycle types occurring in `⟨gens⟩`. `None` if the group exceeds `cap`.
pub fn cycle_type_set(gens: &[Perm], cap: usize) -> Option<BTreeSet<Vec<usize>>> {
    let elements = group_closure(gens, cap)?;
    Some(elements.iter().map(cycle_type).collect())
}

/// Expand `(part, count)` pairs into an ascending sorted cycle type summing to 23.
fn ct(parts: &[(usize, usize)]) -> Vec<usize> {
    let mut v = Vec::new();
    for &(p, c) in parts {
        for _ in 0..c {
            v.push(p);
        }
    }
    v.sort_unstable();
    debug_assert_eq!(v.iter().sum::<usize>(), DEG, "cycle type must sum to 23");
    v
}

/// The seven transitive groups of degree 23.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group23 {
    /// cyclic, `C₂₃`
    C23,
    /// dihedral, `23:2`
    D23,
    /// Frobenius `23:11`
    F23,
    /// affine group `AGL(1,23) = 23:22`
    AGL23,
    /// Mathieu group `M₂₃`
    M23,
    /// alternating `A₂₃`
    A23,
    /// symmetric `S₂₃`
    S23,
}

impl Group23 {
    /// Short label.
    pub fn label(self) -> &'static str {
        match self {
            Group23::C23 => "C23",
            Group23::D23 => "D23 (23:2)",
            Group23::F23 => "F23 (23:11)",
            Group23::AGL23 => "AGL(1,23) (23:22)",
            Group23::M23 => "M23",
            Group23::A23 => "A23",
            Group23::S23 => "S23",
        }
    }

    /// Group order.
    pub fn order(self) -> u128 {
        match self {
            Group23::C23 => 23,
            Group23::D23 => 46,
            Group23::F23 => 253,
            Group23::AGL23 => 506,
            Group23::M23 => 10_200_960,
            Group23::A23 => 12_926_008_369_442_488_320_000, // 23!/2
            Group23::S23 => 25_852_016_738_884_976_640_000, // 23!
        }
    }

    /// Whether `G ⊆ A₂₃` (all elements even ⇔ the discriminant is a square).
    pub fn in_a23(self) -> bool {
        matches!(self, Group23::C23 | Group23::F23 | Group23::M23 | Group23::A23)
    }

    /// The finite cycle-type fingerprint for the "small" groups
    /// (`C23,D23,F23,AGL23,M23`); `None` for `A23,S23` which are handled by a parity
    /// predicate in [`Group23::contains_type`].
    fn finite_type_set(self) -> Option<BTreeSet<Vec<usize>>> {
        let types: Vec<Vec<usize>> = match self {
            Group23::C23 => vec![ct(&[(1, 23)]), ct(&[(23, 1)])],
            Group23::D23 => vec![
                ct(&[(1, 23)]),
                ct(&[(23, 1)]),
                ct(&[(2, 11), (1, 1)]),
            ],
            Group23::F23 => vec![
                ct(&[(1, 23)]),
                ct(&[(23, 1)]),
                ct(&[(11, 2), (1, 1)]),
            ],
            Group23::AGL23 => vec![
                ct(&[(1, 23)]),
                ct(&[(23, 1)]),
                ct(&[(2, 11), (1, 1)]),
                ct(&[(11, 2), (1, 1)]),
                ct(&[(22, 1), (1, 1)]),
            ],
            Group23::M23 => vec![
                ct(&[(1, 23)]),                       // 1A
                ct(&[(2, 8), (1, 7)]),                // 2A
                ct(&[(3, 6), (1, 5)]),                // 3A
                ct(&[(4, 4), (2, 2), (1, 3)]),        // 4A
                ct(&[(5, 4), (1, 3)]),                // 5A
                ct(&[(6, 2), (3, 2), (2, 2), (1, 1)]),// 6A
                ct(&[(7, 3), (1, 2)]),                // 7A/7B
                ct(&[(8, 2), (4, 1), (2, 1), (1, 1)]),// 8A
                ct(&[(11, 2), (1, 1)]),               // 11A/11B
                ct(&[(14, 1), (7, 1), (2, 1)]),       // 14A/14B
                ct(&[(15, 1), (5, 1), (3, 1)]),       // 15A/15B
                ct(&[(23, 1)]),                       // 23A/23B
            ],
            Group23::A23 | Group23::S23 => return None,
        };
        Some(types.into_iter().collect())
    }

    /// Whether a cycle type of 23 occurs in this group.
    pub fn contains_type(self, cyc: &[usize]) -> bool {
        match self {
            Group23::S23 => true,
            Group23::A23 => !is_odd_type(cyc),
            _ => self
                .finite_type_set()
                .map(|s| s.contains(cyc))
                .unwrap_or(false),
        }
    }

    /// The full cycle-type set (`None` for `A23`/`S23`, which are infinite families
    /// captured by the parity predicate).
    pub fn type_set(self) -> Option<BTreeSet<Vec<usize>>> {
        self.finite_type_set()
    }
}

/// The parity-class chain (increasing order) selected by the discriminant-square bit.
fn chain(disc_is_square: bool) -> &'static [Group23] {
    if disc_is_square {
        &[Group23::C23, Group23::F23, Group23::M23, Group23::A23]
    } else {
        &[Group23::D23, Group23::AGL23, Group23::S23]
    }
}

/// Every transitive group of degree 23 in the parity class selected by
/// `disc_is_square` whose cycle-type set contains **all** the `observed` types.
///
/// Sound: the true Galois group is always present (its Frobenius cycle types lie in
/// its own cycle-type set). The result is a suffix of the containment chain (the
/// true group and everything above it).
pub fn candidates(observed: &[Vec<usize>], disc_is_square: bool) -> Vec<Group23> {
    chain(disc_is_square)
        .iter()
        .copied()
        .filter(|g| observed.iter().all(|t| g.contains_type(t)))
        .collect()
}

/// Classify the degree-23 Galois group from its observed Frobenius cycle types and
/// the discriminant-square bit. Returns the **smallest** group in the selected
/// parity class whose cycle-type set contains every observed type — the sharp answer
/// once the prime sample is Chebotarev-saturated. `None` iff no group is consistent
/// (e.g. an odd observed type while `disc_is_square = true`, which is contradictory).
pub fn classify(observed: &[Vec<usize>], disc_is_square: bool) -> Option<Group23> {
    // The chain is ordered by increasing order; the first consistent member is the
    // smallest (most specific) sound identification.
    candidates(observed, disc_is_square).into_iter().next()
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- affine-group generators on residues {0,…,22} (point i = residue i) ---- //
    /// Translation `x -> x+1 mod 23` (a 23-cycle).
    fn translation() -> Perm {
        let mut p = [0u8; DEG];
        for i in 0..DEG {
            p[i] = ((i + 1) % DEG) as u8;
        }
        p
    }
    /// Multiplier `x -> a·x mod 23` (fixes 0).
    fn multiplier(a: u64) -> Perm {
        let mut p = [0u8; DEG];
        for i in 0..DEG {
            p[i] = ((a * i as u64) % DEG as u64) as u8;
        }
        p
    }

    #[test]
    fn small_affine_group_sets_match_closure() {
        // 2 has order 11 mod 23 (2^11 = 2048 = 89*23 + 1); 5 is a primitive root.
        let t = translation();
        // C23 = <t>
        assert_eq!(
            cycle_type_set(&[t], 100).unwrap(),
            Group23::C23.type_set().unwrap()
        );
        // D23 = <t, x->-x>  (-1 mod 23 = 22)
        assert_eq!(
            cycle_type_set(&[t, multiplier(22)], 100).unwrap(),
            Group23::D23.type_set().unwrap()
        );
        // F23 = <t, x->2x>  (2 has order 11)
        assert_eq!(
            cycle_type_set(&[t, multiplier(2)], 1000).unwrap(),
            Group23::F23.type_set().unwrap()
        );
        // AGL(1,23) = <t, x->5x>  (5 is a primitive root, order 22)
        assert_eq!(
            cycle_type_set(&[t, multiplier(5)], 1000).unwrap(),
            Group23::AGL23.type_set().unwrap()
        );
    }

    /// Full observed set = the group's own cycle-type set (Chebotarev-saturated).
    fn full(g: Group23) -> Vec<Vec<usize>> {
        g.type_set().unwrap().into_iter().collect()
    }

    #[test]
    fn classify_even_chain() {
        assert_eq!(classify(&full(Group23::C23), true), Some(Group23::C23));
        assert_eq!(classify(&full(Group23::F23), true), Some(Group23::F23));
        assert_eq!(classify(&full(Group23::M23), true), Some(Group23::M23));
        // A23: an even type not in M23 (a single 3-cycle: 3·1^20, #parts 21 = even perm)
        let a23_sample = vec![ct(&[(3, 1), (1, 20)])];
        assert_eq!(classify(&a23_sample, true), Some(Group23::A23));
    }

    #[test]
    fn classify_odd_chain() {
        assert_eq!(classify(&full(Group23::D23), false), Some(Group23::D23));
        assert_eq!(classify(&full(Group23::AGL23), false), Some(Group23::AGL23));
        // S23: an odd type not in AGL (a transposition: 2·1^21, #parts 22 = odd perm)
        let s23_sample = vec![ct(&[(2, 1), (1, 21)])];
        assert_eq!(classify(&s23_sample, false), Some(Group23::S23));
    }

    #[test]
    fn m23_vs_a23_vs_s23() {
        // M23: its exact fingerprint, disc a square -> M23.
        assert_eq!(classify(&full(Group23::M23), true), Some(Group23::M23));

        // M23 fingerprint plus an even type M23 lacks, disc square -> A23.
        let mut a = full(Group23::M23);
        a.push(ct(&[(3, 1), (1, 20)])); // even, not an M23 type
        assert_eq!(classify(&a, true), Some(Group23::A23));

        // M23 fingerprint plus an odd type => disc not a square -> S23.
        let mut s = full(Group23::M23);
        s.push(ct(&[(2, 1), (1, 21)])); // odd transposition
        assert_eq!(classify(&s, false), Some(Group23::S23));
    }

    #[test]
    fn candidates_are_a_sound_suffix() {
        // C23's types are contained in every even-chain group.
        let cands = candidates(&full(Group23::C23), true);
        assert_eq!(
            cands,
            vec![Group23::C23, Group23::F23, Group23::M23, Group23::A23]
        );
        // F23's distinctive 11^2 1 excludes C23.
        let cands = candidates(&full(Group23::F23), true);
        assert_eq!(cands, vec![Group23::F23, Group23::M23, Group23::A23]);
    }

    #[test]
    fn odd_type_contradicts_square_bit() {
        // An odd type cannot occur in any subgroup of A23.
        let odd = vec![ct(&[(2, 1), (1, 21)])];
        assert_eq!(classify(&odd, true), None);
    }

    #[test]
    fn distinct_cycle_type_sets() {
        // The five finite groups have pairwise-distinct cycle-type sets.
        let finite = [
            Group23::C23,
            Group23::D23,
            Group23::F23,
            Group23::AGL23,
            Group23::M23,
        ];
        for (i, &a) in finite.iter().enumerate() {
            for &b in &finite[i + 1..] {
                assert_ne!(a.type_set(), b.type_set(), "{} vs {}", a.label(), b.label());
            }
        }
    }
}
