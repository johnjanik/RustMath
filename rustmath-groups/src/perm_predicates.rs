//! Free-function permutation-group predicates missing from `permutation_group.rs`.
//!
//! These operate directly on a list of generators, each a permutation given as an
//! **image list** `Vec<usize>` on the points `{0,…,n-1}` (so `g[i]` is the image of
//! `i`). This representation interoperates with [`crate::transitive24::Perm`] and
//! [`crate::transitive23::Perm`] (convert `[u8; N]` → `Vec<usize>`) and with
//! `rustmath_combinatorics::Permutation` (via its `to_vec`/image accessors).
//!
//! Provided: [`orbits`], [`is_transitive`], [`block_systems`] (the nontrivial block
//! systems of a transitive group, via Atkinson's minimal-block algorithm),
//! [`stabilizer`] (point-stabilizer generators via Schreier's lemma), and
//! [`is_primitive`]. All are computed from the generators alone — no group closure —
//! so they are cheap even for enormous groups (e.g. deciding primitivity of a
//! degree-23 group from two generators).
//!
//! Ported/adapted for RustMath (Agent G, dessin→RustMath refactor). The algorithms
//! (orbit BFS, Atkinson minimal blocks, Schreier generators) are standard
//! computational-group-theory routines; no dessin_engine source has an equivalent,
//! so this is a fresh implementation beside the existing crate code.

use std::collections::{BTreeSet, HashSet, VecDeque};

/// A permutation as an image list on `{0,…,n-1}`.
pub type Perm = Vec<usize>;

/// Compose two permutations: `compose(a, b)[i] = a[b[i]]` (apply `b`, then `a`).
fn compose(a: &[usize], b: &[usize]) -> Perm {
    b.iter().map(|&i| a[i]).collect()
}

/// Inverse permutation: `inverse(a)[a[i]] = i`.
fn inverse(a: &[usize]) -> Perm {
    let mut inv = vec![0usize; a.len()];
    for (i, &ai) in a.iter().enumerate() {
        inv[ai] = i;
    }
    inv
}

/// The orbit of `start` under `⟨gens⟩`, returned sorted ascending.
pub fn orbit(gens: &[Perm], n: usize, start: usize) -> Vec<usize> {
    let mut seen = vec![false; n];
    let mut order = Vec::new();
    let mut queue = VecDeque::new();
    seen[start] = true;
    queue.push_back(start);
    while let Some(p) = queue.pop_front() {
        order.push(p);
        for g in gens {
            let q = g[p];
            if !seen[q] {
                seen[q] = true;
                queue.push_back(q);
            }
        }
    }
    order.sort_unstable();
    order
}

/// All orbits of `⟨gens⟩` on `{0,…,n-1}`, each sorted, in ascending order of least
/// element.
pub fn orbits(gens: &[Perm], n: usize) -> Vec<Vec<usize>> {
    let mut seen = vec![false; n];
    let mut out = Vec::new();
    for i in 0..n {
        if !seen[i] {
            let orb = orbit(gens, n, i);
            for &p in &orb {
                seen[p] = true;
            }
            out.push(orb);
        }
    }
    out
}

/// True iff `⟨gens⟩` is transitive on `{0,…,n-1}` (one orbit). Empty degree `n=0`
/// is vacuously non-transitive; `n=1` is transitive.
pub fn is_transitive(gens: &[Perm], n: usize) -> bool {
    if n == 0 {
        return false;
    }
    orbit(gens, n, 0).len() == n
}

// --- union-find for Atkinson's minimal-block algorithm --------------------- //

struct Uf {
    parent: Vec<usize>,
}
impl Uf {
    fn new(n: usize) -> Uf {
        Uf { parent: (0..n).collect() }
    }
    fn find(&mut self, x: usize) -> usize {
        let mut r = x;
        while self.parent[r] != r {
            r = self.parent[r];
        }
        // path compression
        let mut c = x;
        while self.parent[c] != r {
            let next = self.parent[c];
            self.parent[c] = r;
            c = next;
        }
        r
    }
    /// Merge the classes of `a` and `b`; return `true` if they were distinct.
    fn union(&mut self, a: usize, b: usize) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            false
        } else {
            self.parent[ra] = rb;
            true
        }
    }
}

/// The **minimal block** of the transitive group `⟨gens⟩` containing the pair
/// `{a, b}` (Atkinson 1975): the smallest block of imprimitivity that contains both.
/// Returned sorted. If it equals all of `{0,…,n-1}` the pair forces the trivial
/// (whole-set) block; a singleton is impossible for `a != b`.
pub fn minimal_block(gens: &[Perm], n: usize, a: usize, b: usize) -> Vec<usize> {
    let mut uf = Uf::new(n);
    let mut queue = VecDeque::new();
    if uf.union(a, b) {
        queue.push_back((a, b));
    }
    while let Some((x, y)) = queue.pop_front() {
        for g in gens {
            let gx = g[x];
            let gy = g[y];
            if uf.union(gx, gy) {
                queue.push_back((gx, gy));
            }
        }
    }
    let ra = uf.find(a);
    let mut block: Vec<usize> = (0..n).filter(|&i| uf.find(i) == ra).collect();
    block.sort_unstable();
    block
}

/// Canonicalize a block system: sort each block, then sort the blocks.
fn canonical_system(mut sys: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    for b in sys.iter_mut() {
        b.sort_unstable();
    }
    sys.sort();
    sys
}

/// Grow a block into its full block system: the orbit of the set `block` under the
/// group generators (the images `g(block)` partition `{0,…,n-1}`).
fn system_from_block(gens: &[Perm], block: &[usize]) -> Vec<Vec<usize>> {
    let mut known: HashSet<Vec<usize>> = HashSet::new();
    let mut frontier: Vec<Vec<usize>> = Vec::new();
    let mut start = block.to_vec();
    start.sort_unstable();
    known.insert(start.clone());
    frontier.push(start);
    while let Some(blk) = frontier.pop() {
        for g in gens {
            let mut img: Vec<usize> = blk.iter().map(|&x| g[x]).collect();
            img.sort_unstable();
            if known.insert(img.clone()) {
                frontier.push(img);
            }
        }
    }
    canonical_system(known.into_iter().collect())
}

/// All **nontrivial block systems** of the transitive group `⟨gens⟩` — the systems
/// of imprimitivity whose blocks have size strictly between `1` and `n`. Returns an
/// empty vector when the group is primitive (or not transitive). Each system is
/// canonicalized (blocks sorted, blocks in sorted order); duplicates removed.
pub fn block_systems(gens: &[Perm], n: usize) -> Vec<Vec<Vec<usize>>> {
    if !is_transitive(gens, n) {
        return Vec::new();
    }
    let mut seen: HashSet<Vec<Vec<usize>>> = HashSet::new();
    let mut out = Vec::new();
    for b in 1..n {
        let block = minimal_block(gens, n, 0, b);
        if block.len() > 1 && block.len() < n {
            let sys = system_from_block(gens, &block);
            if seen.insert(sys.clone()) {
                out.push(sys);
            }
        }
    }
    out
}

/// True iff `⟨gens⟩` is **primitive**: transitive with no nontrivial block system.
/// (Every transitive group of prime degree is primitive.)
pub fn is_primitive(gens: &[Perm], n: usize) -> bool {
    is_transitive(gens, n) && block_systems(gens, n).is_empty()
}

/// Generators of the **point stabilizer** `Stab(point)` in `⟨gens⟩`, via Schreier's
/// lemma. Computes the orbit of `point` with a Schreier transversal, then returns
/// the nonidentity Schreier generators (each fixing `point`). Works from the
/// generators alone — no full closure — so it applies to large groups. The returned
/// set generates the stabilizer but is not reduced to a minimal generating set.
pub fn stabilizer(gens: &[Perm], n: usize, point: usize) -> Vec<Perm> {
    // Schreier transversal: transversal[beta] maps `point` -> beta.
    let mut transversal: Vec<Option<Perm>> = vec![None; n];
    let identity: Perm = (0..n).collect();
    let mut queue = VecDeque::new();
    transversal[point] = Some(identity.clone());
    queue.push_back(point);
    let mut orbit_pts = vec![point];
    while let Some(beta) = queue.pop_front() {
        let ub = transversal[beta].clone().unwrap();
        for g in gens {
            let gamma = g[beta];
            if transversal[gamma].is_none() {
                transversal[gamma] = Some(compose(g, &ub));
                orbit_pts.push(gamma);
                queue.push_back(gamma);
            }
        }
    }
    let mut gen_set: BTreeSet<Perm> = BTreeSet::new();
    for &beta in &orbit_pts {
        let ub = transversal[beta].as_ref().unwrap();
        for g in gens {
            let gamma = g[beta];
            let ug = transversal[gamma].as_ref().unwrap();
            // s = ug^{-1} * g * ub  (fixes `point`)
            let s = compose(&inverse(ug), &compose(g, ub));
            if s != identity {
                gen_set.insert(s);
            }
        }
    }
    gen_set.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small BFS closure for verifying stabilizer orders in tests.
    fn closure_order(gens: &[Perm], n: usize) -> usize {
        let identity: Perm = (0..n).collect();
        let mut set: HashSet<Perm> = HashSet::new();
        set.insert(identity.clone());
        let mut frontier = vec![identity];
        while let Some(g) = frontier.pop() {
            for s in gens {
                let h = compose(s, &g);
                if set.insert(h.clone()) {
                    frontier.push(h);
                }
            }
        }
        set.len()
    }

    fn cycle(n: usize) -> Perm {
        (0..n).map(|i| (i + 1) % n).collect()
    }

    #[test]
    fn c23_is_transitive_and_primitive() {
        let g = cycle(23);
        let gens = vec![g];
        assert!(is_transitive(&gens, 23));
        assert_eq!(orbits(&gens, 23).len(), 1);
        assert!(block_systems(&gens, 23).is_empty());
        assert!(is_primitive(&gens, 23));
    }

    #[test]
    fn c6_is_imprimitive_with_two_systems() {
        // C6 = <(0 1 2 3 4 5)> on 6 points: blocks {0,3}{1,4}{2,5} and {0,2,4}{1,3,5}.
        let gens = vec![cycle(6)];
        assert!(is_transitive(&gens, 6));
        assert!(!is_primitive(&gens, 6));
        let sys = block_systems(&gens, 6);
        assert_eq!(sys.len(), 2, "C6 has exactly two nontrivial block systems");
        // one system has blocks of size 2, the other size 3
        let sizes: BTreeSet<usize> = sys.iter().map(|s| s[0].len()).collect();
        assert_eq!(sizes, BTreeSet::from([2, 3]));
        assert!(sys.contains(&vec![vec![0, 3], vec![1, 4], vec![2, 5]]));
        assert!(sys.contains(&vec![vec![0, 2, 4], vec![1, 3, 5]]));
    }

    #[test]
    fn degree6_wreath_style_imprimitive() {
        // <(0 1 2)(3 4 5), (0 3)(1 4)(2 5)>: transitive, blocks {0,1,2}{3,4,5}.
        let a = vec![1, 2, 0, 4, 5, 3];
        let b = vec![3, 4, 5, 0, 1, 2];
        let gens = vec![a, b];
        assert!(is_transitive(&gens, 6));
        assert!(!is_primitive(&gens, 6));
        let sys = block_systems(&gens, 6);
        assert!(sys.contains(&vec![vec![0, 1, 2], vec![3, 4, 5]]));
    }

    #[test]
    fn sn_is_primitive() {
        // S4 via adjacent transpositions.
        let t01 = vec![1, 0, 2, 3];
        let t12 = vec![0, 2, 1, 3];
        let t23 = vec![0, 1, 3, 2];
        let gens = vec![t01, t12, t23];
        assert!(is_transitive(&gens, 4));
        assert!(is_primitive(&gens, 4));
        assert!(block_systems(&gens, 4).is_empty());
    }

    #[test]
    fn orbits_of_intransitive() {
        // <(0 1)> on 4 points: orbits {0,1},{2},{3}.
        let gens = vec![vec![1, 0, 2, 3]];
        assert!(!is_transitive(&gens, 4));
        let orbs = orbits(&gens, 4);
        assert_eq!(orbs, vec![vec![0, 1], vec![2], vec![3]]);
    }

    #[test]
    fn stabilizer_of_point_in_s4_is_s3() {
        let t01 = vec![1, 0, 2, 3];
        let t12 = vec![0, 2, 1, 3];
        let t23 = vec![0, 1, 3, 2];
        let gens = vec![t01, t12, t23];
        let stab = stabilizer(&gens, 4, 0);
        // every returned generator fixes 0
        for s in &stab {
            assert_eq!(s[0], 0);
        }
        // Stab_{S4}(0) = S3 on {1,2,3}, order 6.
        assert_eq!(closure_order(&stab, 4), 6);
    }

    #[test]
    fn stabilizer_in_c6_is_trivial() {
        let gens = vec![cycle(6)];
        let stab = stabilizer(&gens, 6, 0);
        // regular action: point stabilizer is trivial (no nonidentity Schreier gens)
        assert!(stab.is_empty());
    }
}
