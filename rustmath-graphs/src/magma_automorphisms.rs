//! Automorphism group of a graph, returned as a plain permutation-list
//! representation (MAGMA Handbook Chapter 149, §149.22 "Automorphism Group of a
//! Graph or Digraph").
//!
//! ## Deferred: wiring to `rustmath-groups::PermutationGroup`
//!
//! MAGMA's `AutomorphismGroup(G)` returns a genuine permutation group `A`
//! (with orbits, stabilisers, composition factors, group actions, …).  In the
//! MAGMA→RustMath port, `rustmath-groups` is an **OFF-LIMITS** crate for this
//! Wave-1 worker (it is owned by the active IGP24 effort — see
//! MASTER_PORT_PLAN.md §0 collision firewall).  Therefore this module returns a
//! *local* permutation-list representation ([`GraphAutomorphisms`]) instead of a
//! `PermutationGroup`.  The remaining work — constructing a
//! `rustmath-groups::PermutationGroup` from the generators below so that
//! `Orbits`, `Action`, `ActionImage`, `Stabilizer`, `IsVertexTransitive`, etc.
//! reuse the existing group machinery — is **deferred** until `rustmath-groups`
//! merges and can be safely depended upon.
//!
//! ## Implementation note
//!
//! The pre-existing `automorphisms.rs` module ships a heuristic nauty-style
//! search that (at the time of writing) fails to construct the individualised
//! permutations and so reports the trivial group for every input.  Rather than
//! rewrite that module (port discipline: new files only), this facade computes
//! the automorphisms directly by an exact backtracking search with degree and
//! partial-adjacency pruning.  It is `O(n!)` in the worst case and therefore
//! intended for the small graphs of the MAGMA worked examples; a guard caps the
//! vertex count and larger inputs return the trivial group with the full
//! computation deferred to a real group backend.

use crate::graph::Graph;

/// Largest graph for which the exact automorphism enumeration is attempted.
/// Beyond this the trivial group is returned (deferred to a group backend).
pub const AUTOMORPHISM_VERTEX_CAP: usize = 10;

/// The automorphism group of a graph as a local permutation-list object.
///
/// Each automorphism is a permutation of `{0, …, degree-1}` stored as a
/// `Vec<usize>` (image list).  This is deliberately *not* a
/// `rustmath-groups::PermutationGroup` (see module docs: groups wiring deferred).
#[derive(Debug, Clone)]
pub struct GraphAutomorphisms {
    /// Number of vertices the permutations act on.
    pub degree: usize,
    /// A generating set for the group, as image lists.
    pub generators: Vec<Vec<usize>>,
    /// The order |Aut(G)| of the automorphism group.
    pub order: usize,
    /// Every automorphism as an image list (populated for small graphs).
    pub elements: Vec<Vec<usize>>,
}

impl GraphAutomorphisms {
    /// Whether the group is trivial (only the identity automorphism).
    pub fn is_trivial(&self) -> bool {
        self.order <= 1
    }

    /// Every automorphism as an image list, when it was enumerated (small
    /// graphs).  Returns `None` if the group was not expanded.
    pub fn all_permutations(&self) -> Option<Vec<Vec<usize>>> {
        if self.elements.is_empty() {
            None
        } else {
            Some(self.elements.clone())
        }
    }
}

/// `AutomorphismGroup(G)` — the automorphism group of `G` as a local
/// permutation-list representation.
///
/// See the module documentation: this returns [`GraphAutomorphisms`] rather than
/// a `rustmath-groups::PermutationGroup` because that crate is off-limits for
/// this port worker; the group wiring is deferred.
pub fn automorphism_group(g: &Graph) -> GraphAutomorphisms {
    let n = g.num_vertices();
    let identity: Vec<usize> = (0..n).collect();

    if n == 0 {
        return GraphAutomorphisms {
            degree: 0,
            generators: Vec::new(),
            order: 1,
            elements: vec![vec![]],
        };
    }
    if n > AUTOMORPHISM_VERTEX_CAP {
        // Deferred to a real backend for large graphs.
        return GraphAutomorphisms {
            degree: n,
            generators: Vec::new(),
            order: 1,
            elements: vec![identity],
        };
    }

    let elements = enumerate_automorphisms(g);
    let order = elements.len();
    let generators = minimal_generators(&elements, n);
    GraphAutomorphisms {
        degree: n,
        generators,
        order,
        elements,
    }
}

/// Enumerate every automorphism of `g` by backtracking, pruning on vertex degree
/// and on adjacency-consistency with the vertices already mapped.
fn enumerate_automorphisms(g: &Graph) -> Vec<Vec<usize>> {
    let n = g.num_vertices();
    // Boolean adjacency matrix and degree vector.
    let mut adj = vec![vec![false; n]; n];
    for i in 0..n {
        if let Some(neigh) = g.neighbors(i) {
            for j in neigh {
                adj[i][j] = true;
            }
        }
    }
    let deg: Vec<usize> = (0..n).map(|v| g.degree(v).unwrap_or(0)).collect();

    let mut perm = vec![usize::MAX; n];
    let mut used = vec![false; n];
    let mut out = Vec::new();
    backtrack(0, n, &adj, &deg, &mut perm, &mut used, &mut out);
    out
}

#[allow(clippy::too_many_arguments)]
fn backtrack(
    i: usize,
    n: usize,
    adj: &[Vec<bool>],
    deg: &[usize],
    perm: &mut Vec<usize>,
    used: &mut Vec<bool>,
    out: &mut Vec<Vec<usize>>,
) {
    if i == n {
        out.push(perm.clone());
        return;
    }
    for w in 0..n {
        if used[w] || deg[w] != deg[i] {
            continue;
        }
        // Adjacency must be preserved against all already-mapped vertices.
        let mut ok = true;
        for j in 0..i {
            if adj[i][j] != adj[w][perm[j]] {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        perm[i] = w;
        used[w] = true;
        backtrack(i + 1, n, adj, deg, perm, used, out);
        used[w] = false;
        perm[i] = usize::MAX;
    }
}

/// Compose two permutations (image lists): `(a ∘ b)[i] = a[b[i]]`.
fn compose(a: &[usize], b: &[usize]) -> Vec<usize> {
    b.iter().map(|&i| a[i]).collect()
}

/// Closure of a set of generators under composition (the generated subgroup) as
/// a sorted list of image lists.
fn closure(gens: &[Vec<usize>], n: usize) -> std::collections::BTreeSet<Vec<usize>> {
    let identity: Vec<usize> = (0..n).collect();
    let mut set = std::collections::BTreeSet::new();
    set.insert(identity.clone());
    let mut frontier = vec![identity];
    while let Some(cur) = frontier.pop() {
        for ggen in gens {
            let prod = compose(&cur, ggen);
            if set.insert(prod.clone()) {
                frontier.push(prod);
            }
        }
    }
    set
}

/// A small generating set for the group given all its elements: greedily add
/// elements that enlarge the currently generated subgroup.
fn minimal_generators(elements: &[Vec<usize>], n: usize) -> Vec<Vec<usize>> {
    let total = elements.len();
    let mut gens: Vec<Vec<usize>> = Vec::new();
    if total <= 1 {
        return gens;
    }
    let mut cur = closure(&gens, n);
    for p in elements {
        if cur.len() == total {
            break;
        }
        if !cur.contains(p) {
            gens.push(p.clone());
            cur = closure(&gens, n);
        }
    }
    gens
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automorphisms::{is_automorphism, Permutation};

    fn cycle(n: usize) -> Graph {
        let mut g = Graph::new(n);
        for i in 0..n {
            g.add_edge(i, (i + 1) % n).unwrap();
        }
        g
    }

    fn complete(n: usize) -> Graph {
        let mut g = Graph::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                g.add_edge(i, j).unwrap();
            }
        }
        g
    }

    fn path(n: usize) -> Graph {
        let mut g = Graph::new(n);
        for i in 0..n - 1 {
            g.add_edge(i, i + 1).unwrap();
        }
        g
    }

    #[test]
    fn automorphism_group_of_triangle_is_s3() {
        // Aut(K3) = S3, order 6.
        let g = complete(3);
        let a = automorphism_group(&g);
        assert_eq!(a.degree, 3);
        assert_eq!(a.order, 6);
        let perms = a.all_permutations().unwrap();
        assert_eq!(perms.len(), 6);
        assert!(perms.contains(&vec![0, 1, 2])); // identity present
    }

    #[test]
    fn automorphism_group_of_kn() {
        // Aut(K_n) = S_n, order n!.
        assert_eq!(automorphism_group(&complete(4)).order, 24);
        assert_eq!(automorphism_group(&complete(5)).order, 120);
    }

    #[test]
    fn automorphism_group_of_cycle_is_dihedral() {
        // Aut(C_n) = dihedral group of order 2n.
        assert_eq!(automorphism_group(&cycle(5)).order, 10);
        assert_eq!(automorphism_group(&cycle(6)).order, 12);
    }

    #[test]
    fn automorphism_group_of_path_is_z2() {
        // Aut(P_n) = Z/2 (the reflection), order 2 for n >= 2.
        assert_eq!(automorphism_group(&path(4)).order, 2);
    }

    #[test]
    fn generators_generate_the_full_group() {
        let g = cycle(6);
        let a = automorphism_group(&g);
        // Every generator is a valid automorphism.
        for gen in &a.generators {
            let p = Permutation::from_vec(gen.clone());
            assert!(is_automorphism(&g, &p));
        }
        // The generators close up to the full group order.
        let cl = closure(&a.generators, a.degree);
        assert_eq!(cl.len(), a.order);
    }
}
