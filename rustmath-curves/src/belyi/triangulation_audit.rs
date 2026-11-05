//! Triangulation audit — the gate that turns "is this packable?" into a checked
//! property instead of a design assumption.
//!
//! The 4E flag triangulation has degree-2 leaf vertices (σ0's 8 fixed points in the
//! `[2,12,5]` dessin), which cannot satisfy the Euclidean circle-packing angle
//! equation except in the `r→0` limit. Any regularization (subdivision, leaf cap,
//! …) must *prove* it removed those vertices — the label ("6E", "full barycentric")
//! is not evidence, this audit is.
//!
//! Adapted from the P2 review note (`inverse_galois/P2_note.md`) to RustMath's
//! plain-`usize` vertex convention.

use crate::belyi::flags::FlagTriangulation;
use crate::belyi::packing::PackingComplex;
use std::collections::{BTreeMap, BTreeSet};

/// A simplicial triangulation as a vertex count + triangle (face) list.
#[derive(Debug, Clone)]
pub struct Triangulation {
    pub n_vertices: usize,
    pub triangles: Vec<[usize; 3]>,
}

/// The result of auditing a triangulation for circle-packability.
#[derive(Debug, Clone)]
pub struct TriangulationAudit {
    pub n_vertices: usize,
    pub n_edges: usize,
    pub n_faces: usize,
    pub euler_characteristic: i64,
    pub min_valence: usize,
    /// Vertices with exactly two distinct neighbours — not Euclidean-packable.
    pub degree_two_vertices: Vec<usize>,
    pub isolated_vertices: Vec<usize>,
    /// Edges shared by a number of triangles other than 2 `(u, v, count)`.
    pub nonmanifold_edges: Vec<(usize, usize, usize)>,
}

fn ordered_edge(a: usize, b: usize) -> (usize, usize) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

impl Triangulation {
    pub fn new(n_vertices: usize, triangles: Vec<[usize; 3]>) -> Self {
        Self {
            n_vertices,
            triangles,
        }
    }

    /// The 4E flag triangulation as a face list (each flag is a triangle
    /// `[vertex, midpoint, face]`). This is the input the audit judges.
    pub fn from_flags(tri: &FlagTriangulation) -> Self {
        let triangles = tri
            .corners
            .iter()
            .map(|c| [c.vertex, c.midpoint, c.face])
            .collect();
        Self::new(tri.n_vertices(), triangles)
    }

    /// Convert to a [`PackingComplex`] for relaxation/layout.
    pub fn to_packing_complex(&self) -> PackingComplex {
        PackingComplex::new(self.n_vertices, self.triangles.clone())
    }

    pub fn audit(&self) -> TriangulationAudit {
        let mut neighbors: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); self.n_vertices];
        let mut edge_counts: BTreeMap<(usize, usize), usize> = BTreeMap::new();

        for &[a, b, c] in &self.triangles {
            for (u, v) in [(a, b), (b, c), (c, a)] {
                *edge_counts.entry(ordered_edge(u, v)).or_insert(0) += 1;
                if u != v {
                    neighbors[u].insert(v);
                    neighbors[v].insert(u);
                }
            }
        }

        let mut degree_two_vertices = Vec::new();
        let mut isolated_vertices = Vec::new();
        let mut min_valence = if self.n_vertices == 0 { 0 } else { usize::MAX };

        for (i, ns) in neighbors.iter().enumerate() {
            let d = ns.len();
            min_valence = min_valence.min(d);
            if d == 0 {
                isolated_vertices.push(i);
            }
            if d == 2 {
                degree_two_vertices.push(i);
            }
        }

        let nonmanifold_edges = edge_counts
            .iter()
            .filter(|(_, &count)| count != 2)
            .map(|(&(u, v), &count)| (u, v, count))
            .collect();

        let v = self.n_vertices as i64;
        let e = edge_counts.len() as i64;
        let f = self.triangles.len() as i64;

        TriangulationAudit {
            n_vertices: self.n_vertices,
            n_edges: edge_counts.len(),
            n_faces: self.triangles.len(),
            euler_characteristic: v - e + f,
            min_valence,
            degree_two_vertices,
            isolated_vertices,
            nonmanifold_edges,
        }
    }

    /// A closed genus-0 surface: `χ = 2`, every edge in exactly two triangles, no
    /// isolated vertices.
    pub fn is_sphere_candidate(&self) -> bool {
        let a = self.audit();
        a.euler_characteristic == 2 && a.nonmanifold_edges.is_empty() && a.isolated_vertices.is_empty()
    }

    /// Packable by the valence gate: a sphere candidate with every vertex of
    /// valence ≥ 3 (so the Euclidean angle equation is solvable everywhere).
    pub fn is_packable_by_valence_gate(&self) -> bool {
        self.is_sphere_candidate() && self.audit().min_valence >= 3
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belyi::flags::flag_triangulation;
    use crate::belyi::monodromy::Permutation;

    #[test]
    fn tetrahedron_is_packable() {
        // The boundary of a tetrahedron: 4 vertices, 4 triangles, all valence 3.
        let tri = Triangulation::new(
            4,
            vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
        );
        let a = tri.audit();
        assert_eq!(a.euler_characteristic, 2);
        assert_eq!(a.min_valence, 3);
        assert!(a.degree_two_vertices.is_empty());
        assert!(a.nonmanifold_edges.is_empty());
        assert!(tri.is_packable_by_valence_gate());
    }

    #[test]
    fn leaf_free_hexagon_cycle_flags_are_a_sphere() {
        // The bipartite 6-cycle: leaf-free, so its flag triangulation is a sphere
        // candidate (χ = 2) even though the 4E model still has some low valence.
        let s0 = Permutation::from_cycles(6, &[vec![5, 0], vec![1, 2], vec![3, 4]]).unwrap();
        let s1 = Permutation::from_cycles(6, &[vec![0, 1], vec![2, 3], vec![4, 5]]).unwrap();
        let ft = flag_triangulation(&s0, &s1).unwrap();
        let tri = Triangulation::from_flags(&ft);
        let a = tri.audit();
        assert!(a.degree_two_vertices.is_empty(), "no leaves ⇒ no degree-2: {a:?}");
    }

    #[test]
    fn star_flags_expose_degree_two_leaves() {
        // The star S3 has three valence-1 white leaves ⇒ degree-2 packing vertices.
        let s0 = Permutation::from_cycles(3, &[vec![0, 1, 2]]).unwrap();
        let s1 = Permutation::identity(3);
        let ft = flag_triangulation(&s0, &s1).unwrap();
        let tri = Triangulation::from_flags(&ft);
        let a = tri.audit();
        assert!(a.min_valence <= 2, "leaves present: {a:?}");
        assert!(!a.degree_two_vertices.is_empty());
        assert!(!tri.is_packable_by_valence_gate());
    }
}
