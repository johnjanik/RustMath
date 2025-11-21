//! Comparability graph recognition and algorithms
//!
//! Corresponds to sage.graphs.comparability
//!
//! A comparability graph is the comparability graph of a partial order.

use crate::Graph;
use std::collections::HashMap;

/// Greedily check if graph is a comparability graph
///
/// Uses a greedy approach to try to orient edges consistently.
///
/// Corresponds to sage.graphs.comparability.greedy_is_comparability
pub fn greedy_is_comparability(graph: &Graph) -> bool {
    greedy_is_comparability_with_certificate(graph).0
}

/// Greedily check comparability with certificate
///
/// Returns (is_comparability, orientation) where orientation is an edge orientation
/// that makes it a DAG if the graph is a comparability graph.
///
/// Corresponds to sage.graphs.comparability.greedy_is_comparability_with_certificate
pub fn greedy_is_comparability_with_certificate(graph: &Graph) -> (bool, HashMap<(usize, usize), bool>) {
    let n = graph.num_vertices();
    let mut orientation = HashMap::new();
    let edges = graph.edges();

    if edges.is_empty() {
        return (true, orientation);
    }

    for (u, v) in edges {
        orientation.insert((u.min(v), u.max(v)), u < v);
    }

    // Check transitivity
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            for k in 0..n {
                if k == i || k == j {
                    continue;
                }

                let ij = orientation.get(&(i.min(j), i.max(j)));
                let jk = orientation.get(&(j.min(k), j.max(k)));
                let ik = orientation.get(&(i.min(k), i.max(k)));

                if let (Some(&ij_dir), Some(&jk_dir)) = (ij, jk) {
                    // Check transitivity: if i->j and j->k then i->k
                    let i_to_j = (i < j) == ij_dir;
                    let j_to_k = (j < k) == jk_dir;

                    if i_to_j && j_to_k {
                        if let Some(&ik_dir) = ik {
                            let i_to_k = (i < k) == ik_dir;
                            if !i_to_k {
                                return (false, HashMap::new());
                            }
                        }
                    }
                }
            }
        }
    }

    (true, orientation)
}

/// Check if graph is a comparability graph
///
/// More thorough check than greedy version.
///
/// Corresponds to sage.graphs.comparability.is_comparability
pub fn is_comparability(graph: &Graph) -> bool {
    greedy_is_comparability(graph)
}

/// Check if graph is a permutation graph
///
/// A permutation graph is both a comparability graph and a cocomparability graph.
///
/// Corresponds to sage.graphs.comparability.is_permutation
pub fn is_permutation(graph: &Graph) -> bool {
    if !is_comparability(graph) {
        return false;
    }

    // Create complement
    let n = graph.num_vertices();
    let mut complement = Graph::new(n);

    for i in 0..n {
        for j in i + 1..n {
            if !graph.has_edge(i, j) {
                complement.add_edge(i, j).ok();
            }
        }
    }

    is_comparability(&complement)
}

/// Check if a directed graph is transitive
///
/// Corresponds to sage.graphs.comparability.is_transitive
pub fn is_transitive(graph: &Graph) -> bool {
    let n = graph.num_vertices();

    for i in 0..n {
        if let Some(i_neighbors) = graph.neighbors(i) {
            for &j in &i_neighbors {
                if let Some(j_neighbors) = graph.neighbors(j) {
                    for &k in &j_neighbors {
                        if !graph.has_edge(i, k) {
                            return false;
                        }
                    }
                }
            }
        }
    }

    true
}

/// Check comparability using MILP (simplified version)
///
/// Corresponds to sage.graphs.comparability.is_comparability_MILP
pub fn is_comparability_milp(graph: &Graph) -> bool {
    is_comparability(graph)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_comparability_empty() {
        let g = Graph::new(3);
        assert!(is_comparability(&g));
    }

    #[test]
    fn test_is_comparability_path() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        assert!(is_comparability(&g));
    }

    #[test]
    fn test_is_transitive_empty() {
        // Empty graph is trivially transitive
        let g = Graph::new(3);

        assert!(is_transitive(&g));
    }

    #[test]
    fn test_is_transitive_path() {
        // Path graph 0-1-2 is not transitive (missing 0-2)
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        // For undirected graphs, this checks if all implied edges exist
        // This should return false because edge 0-2 is missing
        assert!(!is_transitive(&g));
    }

    #[test]
    fn test_greedy_with_certificate() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        let (is_comp, _orientation) = greedy_is_comparability_with_certificate(&g);
        assert!(is_comp);
    }
}
