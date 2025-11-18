//! Static dense graph algorithms
//!
//! Corresponds to sage.graphs.base.static_dense_graph
//!
//! This module provides algorithms that work on immutable (static) dense graphs,
//! optimized for query performance.

use super::dense_graph::DenseGraph;
use super::generic_backend::GenericGraphBackend;
use std::collections::{HashSet, VecDeque};

/// Check if a graph is strongly regular
///
/// A graph G is (n,k,λ,μ)-strongly regular if:
/// - It has n vertices
/// - Every vertex has degree k
/// - Every pair of adjacent vertices has λ common neighbors
/// - Every pair of non-adjacent vertices has μ common neighbors
///
/// # Returns
/// Returns `Some((k, lambda, mu))` if the graph is strongly regular, None otherwise
///
/// Corresponds to sage.graphs.base.static_dense_graph.is_strongly_regular
pub fn is_strongly_regular(graph: &DenseGraph) -> Option<(usize, usize, usize)> {
    let n = graph.num_vertices();

    if n == 0 {
        return None;
    }

    // Get degree of first vertex
    let k = match graph.out_degree(0) {
        Some(d) => d,
        None => return None,
    };

    // Check all vertices have the same degree
    for v in 1..n {
        if graph.out_degree(v) != Some(k) {
            return None; // Not regular
        }
    }

    // Find λ (common neighbors for adjacent pairs)
    let mut lambda = None;
    let mut mu = None;

    for u in 0..n {
        let u_neighbors: HashSet<usize> = graph.neighbors(u).unwrap().into_iter().collect();

        for v in u + 1..n {
            let v_neighbors: HashSet<usize> = graph.neighbors(v).unwrap().into_iter().collect();

            // Count common neighbors
            let common_count = u_neighbors.intersection(&v_neighbors).count();

            if graph.has_edge(u, v) {
                // Adjacent vertices
                if let Some(lam) = lambda {
                    if common_count != lam {
                        return None; // Not strongly regular
                    }
                } else {
                    lambda = Some(common_count);
                }
            } else {
                // Non-adjacent vertices
                if let Some(m) = mu {
                    if common_count != m {
                        return None; // Not strongly regular
                    }
                } else {
                    mu = Some(common_count);
                }
            }
        }
    }

    match (lambda, mu) {
        (Some(lam), Some(m)) => Some((k, lam, m)),
        _ => None,
    }
}

/// Check if a graph is triangle-free
///
/// A graph is triangle-free if it contains no cycles of length 3.
///
/// Corresponds to sage.graphs.base.static_dense_graph.is_triangle_free
pub fn is_triangle_free(graph: &DenseGraph) -> bool {
    let n = graph.num_vertices();

    // Check all possible triangles
    for u in 0..n {
        let u_neighbors: HashSet<usize> = graph.neighbors(u).unwrap().into_iter().collect();

        for &v in &u_neighbors {
            if v <= u {
                continue; // Avoid checking the same triangle twice
            }

            let v_neighbors: HashSet<usize> = graph.neighbors(v).unwrap().into_iter().collect();

            // Check if u and v have a common neighbor w where w > v
            for &w in &v_neighbors {
                if w > v && u_neighbors.contains(&w) {
                    return false; // Found triangle u-v-w
                }
            }
        }
    }

    true
}

/// Count the number of triangles in a graph
///
/// Returns the total number of triangles (3-cycles) in the graph.
///
/// Corresponds to sage.graphs.base.static_dense_graph.triangles_count
pub fn triangles_count(graph: &DenseGraph) -> usize {
    let n = graph.num_vertices();
    let mut count = 0;

    // Check all possible triangles
    for u in 0..n {
        let u_neighbors: HashSet<usize> = graph.neighbors(u).unwrap().into_iter().collect();

        for &v in &u_neighbors {
            if v <= u {
                continue;
            }

            let v_neighbors: HashSet<usize> = graph.neighbors(v).unwrap().into_iter().collect();

            // Count common neighbors w where w > v
            for &w in &v_neighbors {
                if w > v && u_neighbors.contains(&w) {
                    count += 1;
                }
            }
        }
    }

    count
}

/// Find all connected full subgraphs (cliques) of a given size
///
/// Returns all maximal cliques in the graph that have exactly `size` vertices.
///
/// Corresponds to sage.graphs.base.static_dense_graph.connected_full_subgraphs
pub fn connected_full_subgraphs(graph: &DenseGraph, size: usize) -> Vec<Vec<usize>> {
    if size == 0 {
        return vec![vec![]];
    }

    let n = graph.num_vertices();
    if size > n {
        return vec![];
    }

    let mut cliques = Vec::new();
    let mut current_clique = Vec::new();
    let mut candidates: Vec<usize> = (0..n).collect();

    find_cliques_of_size(
        graph,
        &mut current_clique,
        &mut candidates,
        size,
        &mut cliques,
    );

    cliques
}

/// Helper function for finding cliques of a specific size
fn find_cliques_of_size(
    graph: &DenseGraph,
    current: &mut Vec<usize>,
    candidates: &mut Vec<usize>,
    target_size: usize,
    result: &mut Vec<Vec<usize>>,
) {
    if current.len() == target_size {
        result.push(current.clone());
        return;
    }

    if current.len() + candidates.len() < target_size {
        return; // Not enough candidates left
    }

    let candidates_copy = candidates.clone();
    for (i, &v) in candidates_copy.iter().enumerate() {
        // Add v to current clique
        current.push(v);

        // Filter candidates to only those adjacent to v
        let v_neighbors: HashSet<usize> = graph.neighbors(v).unwrap().into_iter().collect();
        let new_candidates: Vec<usize> = candidates_copy[i + 1..]
            .iter()
            .filter(|&&u| v_neighbors.contains(&u))
            .copied()
            .collect();

        let mut new_cands = new_candidates;
        find_cliques_of_size(graph, current, &mut new_cands, target_size, result);

        // Backtrack
        current.pop();
    }
}

/// Iterator over all connected subgraphs
///
/// Generates all connected induced subgraphs of the graph.
///
/// Corresponds to sage.graphs.base.static_dense_graph.connected_subgraph_iterator
pub fn connected_subgraph_iterator(graph: &DenseGraph) -> ConnectedSubgraphIterator {
    ConnectedSubgraphIterator::new(graph.clone())
}

/// Iterator for connected subgraphs
pub struct ConnectedSubgraphIterator {
    graph: DenseGraph,
    current_size: usize,
    current_subsets: Vec<Vec<usize>>,
    subset_index: usize,
}

impl ConnectedSubgraphIterator {
    fn new(graph: DenseGraph) -> Self {
        ConnectedSubgraphIterator {
            graph,
            current_size: 1,
            current_subsets: Vec::new(),
            subset_index: 0,
        }
    }

    /// Generate all subsets of given size
    fn generate_subsets(&mut self) {
        let n = self.graph.num_vertices();
        if self.current_size > n {
            self.current_subsets = Vec::new();
            return;
        }

        self.current_subsets = Vec::new();
        let mut current = Vec::new();
        self.generate_subsets_helper(0, &mut current);
        self.subset_index = 0;
    }

    fn generate_subsets_helper(&mut self, start: usize, current: &mut Vec<usize>) {
        if current.len() == self.current_size {
            // Check if this subset induces a connected subgraph
            if self.is_connected_subset(current) {
                self.current_subsets.push(current.clone());
            }
            return;
        }

        let n = self.graph.num_vertices();
        for v in start..n {
            current.push(v);
            self.generate_subsets_helper(v + 1, current);
            current.pop();
        }
    }

    fn is_connected_subset(&self, vertices: &[usize]) -> bool {
        if vertices.is_empty() {
            return false;
        }

        let vertex_set: HashSet<usize> = vertices.iter().copied().collect();

        // BFS from first vertex
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(vertices[0]);
        visited.insert(vertices[0]);

        while let Some(v) = queue.pop_front() {
            if let Some(neighbors) = self.graph.neighbors(v) {
                for neighbor in neighbors {
                    if vertex_set.contains(&neighbor) && !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        visited.len() == vertices.len()
    }
}

impl Iterator for ConnectedSubgraphIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_subsets.is_empty() || self.subset_index >= self.current_subsets.len() {
                self.current_size += 1;
                if self.current_size > self.graph.num_vertices() {
                    return None;
                }
                self.generate_subsets();
            }

            if self.subset_index < self.current_subsets.len() {
                let result = self.current_subsets[self.subset_index].clone();
                self.subset_index += 1;
                return Some(result);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_triangle_free_empty() {
        let graph = DenseGraph::new(false);
        assert!(is_triangle_free(&graph));
    }

    #[test]
    fn test_is_triangle_free_no_triangle() {
        let mut graph = DenseGraph::new(false);
        graph.add_vertex();
        graph.add_vertex();
        graph.add_vertex();

        graph.add_edge(0, 1, None, None).unwrap();
        graph.add_edge(1, 2, None, None).unwrap();
        // No edge 0-2, so no triangle

        assert!(is_triangle_free(&graph));
    }

    #[test]
    fn test_is_triangle_free_with_triangle() {
        let mut graph = DenseGraph::new(false);
        graph.add_vertex();
        graph.add_vertex();
        graph.add_vertex();

        graph.add_edge(0, 1, None, None).unwrap();
        graph.add_edge(1, 2, None, None).unwrap();
        graph.add_edge(2, 0, None, None).unwrap();

        assert!(!is_triangle_free(&graph));
    }

    #[test]
    fn test_triangles_count() {
        let mut graph = DenseGraph::new(false);
        graph.add_vertex();
        graph.add_vertex();
        graph.add_vertex();
        graph.add_vertex();

        // Create two triangles: 0-1-2 and 1-2-3
        graph.add_edge(0, 1, None, None).unwrap();
        graph.add_edge(1, 2, None, None).unwrap();
        graph.add_edge(2, 0, None, None).unwrap();
        graph.add_edge(2, 3, None, None).unwrap();
        graph.add_edge(3, 1, None, None).unwrap();

        assert_eq!(triangles_count(&graph), 2);
    }

    #[test]
    fn test_triangles_count_k4() {
        let mut graph = DenseGraph::new(false);
        for _ in 0..4 {
            graph.add_vertex();
        }

        // Create complete graph K4 (has 4 triangles)
        for i in 0..4 {
            for j in i + 1..4 {
                graph.add_edge(i, j, None, None).unwrap();
            }
        }

        assert_eq!(triangles_count(&graph), 4);
    }

    #[test]
    fn test_is_strongly_regular_empty() {
        let graph = DenseGraph::new(false);
        assert_eq!(is_strongly_regular(&graph), None);
    }

    #[test]
    fn test_is_strongly_regular_petersen() {
        // Petersen graph is (10,3,0,1)-strongly regular
        // For simplicity, test a smaller regular graph: cycle C5
        let mut graph = DenseGraph::new(false);
        for _ in 0..5 {
            graph.add_vertex();
        }

        // Create cycle C5 (5-cycle)
        for i in 0..5 {
            graph.add_edge(i, (i + 1) % 5, None, None).unwrap();
        }

        // C5 is (5,2,0,1)-strongly regular
        let result = is_strongly_regular(&graph);
        assert!(result.is_some());
        let (k, lambda, mu) = result.unwrap();
        assert_eq!(k, 2); // Each vertex has degree 2
        assert_eq!(lambda, 0); // Adjacent vertices share 0 neighbors
        assert_eq!(mu, 1); // Non-adjacent vertices share 1 neighbor
    }

    #[test]
    fn test_connected_full_subgraphs() {
        let mut graph = DenseGraph::new(false);
        for _ in 0..4 {
            graph.add_vertex();
        }

        // Create complete graph K4
        for i in 0..4 {
            for j in i + 1..4 {
                graph.add_edge(i, j, None, None).unwrap();
            }
        }

        // Find all triangles (size 3 cliques)
        let triangles = connected_full_subgraphs(&graph, 3);
        assert_eq!(triangles.len(), 4); // K4 has 4 triangles

        // Find all edges (size 2 cliques)
        let edges = connected_full_subgraphs(&graph, 2);
        assert_eq!(edges.len(), 6); // K4 has 6 edges

        // Find all size-4 cliques
        let cliques4 = connected_full_subgraphs(&graph, 4);
        assert_eq!(cliques4.len(), 1); // K4 itself
    }

    #[test]
    fn test_connected_full_subgraphs_empty() {
        let graph = DenseGraph::new(false);
        let cliques = connected_full_subgraphs(&graph, 1);
        assert_eq!(cliques.len(), 0);
    }

    #[test]
    fn test_connected_full_subgraphs_single_vertex() {
        let mut graph = DenseGraph::new(false);
        graph.add_vertex();
        graph.add_vertex();
        graph.add_vertex();

        let vertices = connected_full_subgraphs(&graph, 1);
        assert_eq!(vertices.len(), 3); // Three isolated vertices
    }

    #[test]
    fn test_connected_subgraph_iterator() {
        let mut graph = DenseGraph::new(false);
        graph.add_vertex();
        graph.add_vertex();
        graph.add_vertex();

        graph.add_edge(0, 1, None, None).unwrap();
        graph.add_edge(1, 2, None, None).unwrap();
        graph.add_edge(0, 2, None, None).unwrap();

        let iter = connected_subgraph_iterator(&graph);

        // Should generate connected subgraphs
        // This is a triangle, so all subsets are connected
        let subgraphs: Vec<Vec<usize>> = iter.take(10).collect();

        // Should generate some subgraphs
        assert!(subgraphs.len() > 0);

        // All subgraphs should be non-empty
        for subgraph in &subgraphs {
            assert!(subgraph.len() > 0);
        }
    }
}
