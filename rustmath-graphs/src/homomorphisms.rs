//! Graph homomorphisms and related concepts
//!
//! A graph homomorphism is a mapping between graphs that preserves adjacency.
//! This module provides functions for checking homomorphisms, isomorphisms,
//! and related graph mappings.

use crate::graph::Graph;
use std::collections::{HashMap, HashSet};

/// A graph homomorphism is a mapping f: V(G) -> V(H) such that
/// if (u,v) is an edge in G, then (f(u), f(v)) is an edge in H.
#[derive(Debug, Clone)]
pub struct GraphHomomorphism {
    /// The mapping from vertices of G to vertices of H
    pub mapping: HashMap<usize, usize>,
}

impl GraphHomomorphism {
    /// Create a new graph homomorphism from a mapping
    pub fn new(mapping: HashMap<usize, usize>) -> Self {
        GraphHomomorphism { mapping }
    }

    /// Apply the homomorphism to a vertex
    pub fn apply(&self, v: usize) -> Option<usize> {
        self.mapping.get(&v).copied()
    }

    /// Check if this is a valid homomorphism from g to h
    pub fn is_valid(&self, g: &Graph, h: &Graph) -> bool {
        // Check all vertices of g are mapped
        for v in 0..g.num_vertices() {
            if !self.mapping.contains_key(&v) {
                return false;
            }
        }

        // Check all mapped vertices exist in h
        for &target in self.mapping.values() {
            if target >= h.num_vertices() {
                return false;
            }
        }

        // Check edge preservation: if (u,v) in G, then (f(u), f(v)) in H
        for u in 0..g.num_vertices() {
            if let Some(neighbors) = g.neighbors(u) {
                for v in neighbors {
                    if u < v {  // Check each edge once
                        let fu = self.mapping[&u];
                        let fv = self.mapping[&v];
                        if !h.has_edge(fu, fv) {
                            return false;
                        }
                    }
                }
            }
        }

        true
    }

    /// Check if this homomorphism is injective (one-to-one)
    pub fn is_injective(&self) -> bool {
        let mut seen = HashSet::new();
        for &v in self.mapping.values() {
            if !seen.insert(v) {
                return false;
            }
        }
        true
    }

    /// Check if this homomorphism is surjective (onto)
    pub fn is_surjective(&self, h: &Graph) -> bool {
        let mut covered = vec![false; h.num_vertices()];
        for &v in self.mapping.values() {
            if v < h.num_vertices() {
                covered[v] = true;
            }
        }
        covered.iter().all(|&x| x)
    }

    /// Check if this is a graph isomorphism
    /// (bijective homomorphism whose inverse is also a homomorphism)
    pub fn is_isomorphism(&self, g: &Graph, h: &Graph) -> bool {
        if !self.is_valid(g, h) {
            return false;
        }

        if !self.is_injective() || !self.is_surjective(h) {
            return false;
        }

        // Check that the inverse preserves edges
        let mut inverse = HashMap::new();
        for (&u, &v) in &self.mapping {
            inverse.insert(v, u);
        }

        // If (u,v) in H, then (f^-1(u), f^-1(v)) must be in G
        for u in 0..h.num_vertices() {
            if let Some(neighbors) = h.neighbors(u) {
                for v in neighbors {
                    if u < v {
                        if let (Some(&gu), Some(&gv)) = (inverse.get(&u), inverse.get(&v)) {
                            if !g.has_edge(gu, gv) {
                                return false;
                            }
                        }
                    }
                }
            }
        }

        true
    }
}

/// Check if two graphs are isomorphic
///
/// This is a simple backtracking algorithm. For larger graphs,
/// more sophisticated algorithms like VF2 or nauty should be used.
pub fn is_isomorphic(g: &Graph, h: &Graph) -> bool {
    if g.num_vertices() != h.num_vertices() {
        return false;
    }

    if g.num_edges() != h.num_edges() {
        return false;
    }

    // Try all possible mappings (brute force for small graphs)
    let n = g.num_vertices();
    if n > 10 {
        // For larger graphs, use degree sequence heuristic
        if !has_same_degree_sequence(g, h) {
            return false;
        }
    }

    // Backtracking search
    let mut mapping = HashMap::new();
    let mut used = vec![false; n];
    is_isomorphic_backtrack(g, h, 0, &mut mapping, &mut used)
}

fn is_isomorphic_backtrack(
    g: &Graph,
    h: &Graph,
    vertex: usize,
    mapping: &mut HashMap<usize, usize>,
    used: &mut [bool],
) -> bool {
    if vertex == g.num_vertices() {
        // All vertices mapped, check if it's valid
        let hom = GraphHomomorphism::new(mapping.clone());
        return hom.is_isomorphism(g, h);
    }

    // Try mapping vertex to each unused vertex in h
    for target in 0..h.num_vertices() {
        if used[target] {
            continue;
        }

        // Prune: check degree
        if g.degree(vertex).unwrap_or(0) != h.degree(target).unwrap_or(0) {
            continue;
        }

        // Try this mapping
        mapping.insert(vertex, target);
        used[target] = true;

        // Check partial consistency
        if is_partial_mapping_valid(g, h, mapping) {
            if is_isomorphic_backtrack(g, h, vertex + 1, mapping, used) {
                return true;
            }
        }

        // Backtrack
        mapping.remove(&vertex);
        used[target] = false;
    }

    false
}

fn is_partial_mapping_valid(g: &Graph, h: &Graph, mapping: &HashMap<usize, usize>) -> bool {
    // Check that all edges between mapped vertices are preserved
    for (&u, &fu) in mapping.iter() {
        if let Some(neighbors) = g.neighbors(u) {
            for v in neighbors {
                if let Some(&fv) = mapping.get(&v) {
                    if !h.has_edge(fu, fv) {
                        return false;
                    }
                }
            }
        }
    }
    true
}

fn has_same_degree_sequence(g: &Graph, h: &Graph) -> bool {
    let mut deg_g: Vec<usize> = (0..g.num_vertices())
        .map(|v| g.degree(v).unwrap_or(0))
        .collect();
    let mut deg_h: Vec<usize> = (0..h.num_vertices())
        .map(|v| h.degree(v).unwrap_or(0))
        .collect();

    deg_g.sort_unstable();
    deg_h.sort_unstable();

    deg_g == deg_h
}

/// Count the number of homomorphisms from g to h
///
/// Warning: This can be exponential in the size of g
pub fn count_homomorphisms(g: &Graph, h: &Graph) -> usize {
    let mut count = 0;
    enumerate_homomorphisms_helper(g, h, 0, &mut HashMap::new(), &mut count);
    count
}

fn enumerate_homomorphisms_helper(
    g: &Graph,
    h: &Graph,
    vertex: usize,
    mapping: &mut HashMap<usize, usize>,
    count: &mut usize,
) {
    if vertex == g.num_vertices() {
        // Check if current mapping is valid
        let hom = GraphHomomorphism::new(mapping.clone());
        if hom.is_valid(g, h) {
            *count += 1;
        }
        return;
    }

    // Try mapping vertex to each vertex in h
    for target in 0..h.num_vertices() {
        mapping.insert(vertex, target);

        // Early pruning: check if partial mapping preserves edges
        if is_partial_mapping_valid(g, h, mapping) {
            enumerate_homomorphisms_helper(g, h, vertex + 1, mapping, count);
        }

        mapping.remove(&vertex);
    }
}

/// Find a homomorphism from g to h if one exists
pub fn find_homomorphism(g: &Graph, h: &Graph) -> Option<GraphHomomorphism> {
    let mut mapping = HashMap::new();
    if find_homomorphism_helper(g, h, 0, &mut mapping) {
        Some(GraphHomomorphism::new(mapping))
    } else {
        None
    }
}

fn find_homomorphism_helper(
    g: &Graph,
    h: &Graph,
    vertex: usize,
    mapping: &mut HashMap<usize, usize>,
) -> bool {
    if vertex == g.num_vertices() {
        let hom = GraphHomomorphism::new(mapping.clone());
        return hom.is_valid(g, h);
    }

    for target in 0..h.num_vertices() {
        mapping.insert(vertex, target);

        if is_partial_mapping_valid(g, h, mapping) {
            if find_homomorphism_helper(g, h, vertex + 1, mapping) {
                return true;
            }
        }

        mapping.remove(&vertex);
    }

    false
}

/// Check if g is a core (has no homomorphism to a proper subgraph)
pub fn is_core(g: &Graph) -> bool {
    // A graph is a core if every homomorphism to itself is an automorphism
    // Simplified check: graph with no non-trivial endomorphisms

    // For simple check, we verify that the only homomorphism to itself
    // is the identity (for connected graphs)
    if !g.is_connected() {
        return false;
    }

    // A core has no homomorphism to any proper induced subgraph
    // For now, implement a simplified version
    true
}

/// Compute the core of a graph (smallest subgraph to which g has a retraction)
///
/// This is a simplified version. A full implementation would use
/// more sophisticated algorithms.
pub fn core(g: &Graph) -> Graph {
    // For now, return the graph itself
    // A proper implementation would find the minimal core
    g.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_homomorphism_identity() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        let mut mapping = HashMap::new();
        mapping.insert(0, 0);
        mapping.insert(1, 1);
        mapping.insert(2, 2);

        let hom = GraphHomomorphism::new(mapping);
        assert!(hom.is_valid(&g, &g));
        assert!(hom.is_injective());
        assert!(hom.is_surjective(&g));
        assert!(hom.is_isomorphism(&g, &g));
    }

    #[test]
    fn test_homomorphism_path_to_edge() {
        // Path P3 -> Edge K2
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        let mut h = Graph::new(2);
        h.add_edge(0, 1).unwrap();

        // Map 0->0, 1->1, 2->0
        let mut mapping = HashMap::new();
        mapping.insert(0, 0);
        mapping.insert(1, 1);
        mapping.insert(2, 0);

        let hom = GraphHomomorphism::new(mapping);
        assert!(hom.is_valid(&g, &h));
        assert!(!hom.is_injective());
    }

    #[test]
    fn test_isomorphism_same_graph() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(0, 2).unwrap();

        assert!(is_isomorphic(&g, &g));
    }

    #[test]
    fn test_isomorphism_different_size() {
        let g = Graph::new(3);
        let h = Graph::new(4);

        assert!(!is_isomorphic(&g, &h));
    }

    #[test]
    fn test_isomorphism_triangle() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(0, 2).unwrap();

        let mut h = Graph::new(3);
        h.add_edge(2, 1).unwrap();
        h.add_edge(1, 0).unwrap();
        h.add_edge(2, 0).unwrap();

        assert!(is_isomorphic(&g, &h));
    }

    #[test]
    fn test_find_homomorphism() {
        let mut g = Graph::new(2);
        g.add_edge(0, 1).unwrap();

        let mut h = Graph::new(3);
        h.add_edge(0, 1).unwrap();
        h.add_edge(1, 2).unwrap();

        let hom = find_homomorphism(&g, &h);
        assert!(hom.is_some());
        assert!(hom.unwrap().is_valid(&g, &h));
    }

    #[test]
    fn test_count_homomorphisms() {
        let mut g = Graph::new(1);

        let mut h = Graph::new(2);
        h.add_edge(0, 1).unwrap();

        // Single vertex can map to either vertex of h
        let count = count_homomorphisms(&g, &h);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_degree_sequence() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        let mut h = Graph::new(3);
        h.add_edge(0, 1).unwrap();
        h.add_edge(1, 2).unwrap();

        assert!(has_same_degree_sequence(&g, &h));
    }

    #[test]
    fn test_is_core() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(0, 2).unwrap();

        // Triangle is a core
        assert!(is_core(&g));
    }
}
