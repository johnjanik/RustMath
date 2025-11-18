//! Clique finding algorithms
//!
//! Corresponds to sage.graphs.cliquer
//!
//! Provides algorithms for finding cliques (complete subgraphs) in graphs.

use crate::Graph;
use std::collections::HashSet;

/// Find a maximum clique in the graph
///
/// A maximum clique is a largest possible complete subgraph.
///
/// Corresponds to sage.graphs.cliquer.max_clique
pub fn max_clique(graph: &Graph) -> Vec<usize> {
    let all_max = all_max_clique(graph);
    all_max.into_iter().next().unwrap_or_else(Vec::new)
}

/// Find all maximum cliques in the graph
///
/// Returns all cliques that have the maximum size.
///
/// Corresponds to sage.graphs.cliquer.all_max_clique
pub fn all_max_clique(graph: &Graph) -> Vec<Vec<usize>> {
    let mut max_size = 0;
    let mut max_cliques = Vec::new();

    let all_cliques_list = all_cliques(graph, 0, graph.num_vertices());

    for clique in all_cliques_list {
        if clique.len() > max_size {
            max_size = clique.len();
            max_cliques = vec![clique];
        } else if clique.len() == max_size {
            max_cliques.push(clique);
        }
    }

    max_cliques
}

/// Find all cliques within a size range
///
/// Returns all maximal cliques with size in [min_size, max_size].
///
/// Corresponds to sage.graphs.cliquer.all_cliques
pub fn all_cliques(graph: &Graph, min_size: usize, max_size: usize) -> Vec<Vec<usize>> {
    let n = graph.num_vertices();
    let mut cliques = Vec::new();
    let mut current = Vec::new();
    let mut candidates: Vec<usize> = (0..n).collect();

    bron_kerbosch(graph, &mut current, &mut candidates, &mut cliques, min_size, max_size);

    cliques
}

/// Compute the clique number (size of maximum clique)
///
/// Corresponds to sage.graphs.cliquer.clique_number
pub fn clique_number(graph: &Graph) -> usize {
    let max_cl = max_clique(graph);
    max_cl.len()
}

/// Bron-Kerbosch algorithm for finding all maximal cliques
fn bron_kerbosch(
    graph: &Graph,
    current: &mut Vec<usize>,
    candidates: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
    min_size: usize,
    max_size: usize,
) {
    if candidates.is_empty() {
        if current.len() >= min_size && current.len() <= max_size {
            result.push(current.clone());
        }
        return;
    }

    if current.len() + candidates.len() < min_size {
        return; // Pruning
    }

    let candidates_copy = candidates.clone();
    for (i, &v) in candidates_copy.iter().enumerate() {
        current.push(v);

        // Filter candidates to neighbors of v
        let v_neighbors: HashSet<usize> = graph.neighbors(v)
            .unwrap_or_else(Vec::new)
            .into_iter()
            .collect();

        let mut new_candidates: Vec<usize> = candidates_copy[i + 1..]
            .iter()
            .filter(|&&u| v_neighbors.contains(&u))
            .copied()
            .collect();

        bron_kerbosch(graph, current, &mut new_candidates, result, min_size, max_size);

        current.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_clique_complete() {
        let mut g = Graph::new(4);
        for i in 0..4 {
            for j in i + 1..4 {
                g.add_edge(i, j).unwrap();
            }
        }

        let clique = max_clique(&g);
        assert_eq!(clique.len(), 4);
    }

    #[test]
    fn test_clique_number() {
        let mut g = Graph::new(5);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 0).unwrap();

        assert_eq!(clique_number(&g), 3);
    }

    #[test]
    fn test_all_cliques() {
        let mut g = Graph::new(4);
        for i in 0..3 {
            for j in i + 1..3 {
                g.add_edge(i, j).unwrap();
            }
        }

        let cliques = all_cliques(&g, 3, 3);
        assert_eq!(cliques.len(), 1);
    }
}
