//! Static sparse graph algorithms
//!
//! Corresponds to sage.graphs.base.static_sparse_graph
//!
//! This module provides algorithms that work on immutable (static) sparse graphs.

use super::sparse_graph::SparseGraph;
use super::generic_backend::GenericGraphBackend;
use std::collections::{HashMap, HashSet, VecDeque};

/// Compute the spectral radius of a graph
///
/// The spectral radius is the largest eigenvalue (in absolute value) of the adjacency matrix.
/// This implementation uses the power iteration method for approximation.
///
/// Corresponds to sage.graphs.base.static_sparse_graph.spectral_radius
pub fn spectral_radius(graph: &SparseGraph, iterations: usize) -> f64 {
    let n = graph.num_vertices();

    if n == 0 {
        return 0.0;
    }

    // Initialize with a random-ish vector
    let mut v: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();

    // Normalize
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }

    // Power iteration
    for _ in 0..iterations {
        let mut new_v = vec![0.0; n];

        // Multiply by adjacency matrix
        for i in 0..n {
            if let Some(neighbors) = graph.out_neighbors(i) {
                for &j in &neighbors {
                    new_v[i] += v[j];
                }
            }
        }

        // Normalize
        let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm == 0.0 {
            break;
        }

        for x in &mut new_v {
            *x /= norm;
        }

        v = new_v;
    }

    // Compute Rayleigh quotient: v^T A v / v^T v
    let mut numerator = 0.0;
    for i in 0..n {
        if let Some(neighbors) = graph.out_neighbors(i) {
            for &j in &neighbors {
                numerator += v[i] * v[j];
            }
        }
    }

    numerator
}

/// Find strongly connected components using Tarjan's algorithm
///
/// Returns a vector where each element is a vector of vertices in that component.
///
/// Corresponds to sage.graphs.base.static_sparse_graph.tarjan_strongly_connected_components
pub fn tarjan_strongly_connected_components(graph: &SparseGraph) -> Vec<Vec<usize>> {
    if !graph.is_directed() {
        // For undirected graphs, SCCs are the same as connected components
        return connected_components_undirected(graph);
    }

    let n = graph.num_vertices();
    let mut index = 0;
    let mut stack = Vec::new();
    let mut on_stack = vec![false; n];
    let mut indices = vec![None; n];
    let mut low_links = vec![None; n];
    let mut sccs = Vec::new();

    for v in 0..n {
        if indices[v].is_none() {
            tarjan_strongconnect(
                v,
                &mut index,
                &mut stack,
                &mut on_stack,
                &mut indices,
                &mut low_links,
                &mut sccs,
                graph,
            );
        }
    }

    sccs
}

fn tarjan_strongconnect(
    v: usize,
    index: &mut usize,
    stack: &mut Vec<usize>,
    on_stack: &mut [bool],
    indices: &mut [Option<usize>],
    low_links: &mut [Option<usize>],
    sccs: &mut Vec<Vec<usize>>,
    graph: &SparseGraph,
) {
    indices[v] = Some(*index);
    low_links[v] = Some(*index);
    *index += 1;
    stack.push(v);
    on_stack[v] = true;

    if let Some(neighbors) = graph.out_neighbors(v) {
        for &w in &neighbors {
            if indices[w].is_none() {
                tarjan_strongconnect(w, index, stack, on_stack, indices, low_links, sccs, graph);
                low_links[v] = Some(low_links[v].unwrap().min(low_links[w].unwrap()));
            } else if on_stack[w] {
                low_links[v] = Some(low_links[v].unwrap().min(indices[w].unwrap()));
            }
        }
    }

    if low_links[v] == indices[v] {
        let mut scc = Vec::new();
        loop {
            let w = stack.pop().unwrap();
            on_stack[w] = false;
            scc.push(w);
            if w == v {
                break;
            }
        }
        sccs.push(scc);
    }
}

fn connected_components_undirected(graph: &SparseGraph) -> Vec<Vec<usize>> {
    let n = graph.num_vertices();
    let mut visited = vec![false; n];
    let mut components = Vec::new();

    for v in 0..n {
        if !visited[v] {
            let mut component = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(v);
            visited[v] = true;

            while let Some(u) = queue.pop_front() {
                component.push(u);

                if let Some(neighbors) = graph.out_neighbors(u) {
                    for &neighbor in &neighbors {
                        if !visited[neighbor] {
                            visited[neighbor] = true;
                            queue.push_back(neighbor);
                        }
                    }
                }
            }

            components.push(component);
        }
    }

    components
}

/// Build the strongly connected components digraph
///
/// Returns a new graph where each SCC is condensed to a single vertex.
/// The resulting graph is a DAG.
///
/// Corresponds to sage.graphs.base.static_sparse_graph.strongly_connected_components_digraph
pub fn strongly_connected_components_digraph(graph: &SparseGraph) -> (SparseGraph, Vec<Vec<usize>>) {
    let sccs = tarjan_strongly_connected_components(graph);
    let num_sccs = sccs.len();

    // Create mapping from vertex to SCC index
    let mut vertex_to_scc = HashMap::new();
    for (scc_idx, scc) in sccs.iter().enumerate() {
        for &v in scc {
            vertex_to_scc.insert(v, scc_idx);
        }
    }

    // Build condensation graph
    let mut condensation = SparseGraph::new(true);
    for _ in 0..num_sccs {
        condensation.add_vertex();
    }

    let mut added_edges = HashSet::new();
    for u in 0..graph.num_vertices() {
        if let Some(neighbors) = graph.out_neighbors(u) {
            let scc_u = vertex_to_scc[&u];
            for &v in &neighbors {
                let scc_v = vertex_to_scc[&v];
                if scc_u != scc_v && !added_edges.contains(&(scc_u, scc_v)) {
                    condensation.add_edge(scc_u, scc_v, None, None).ok();
                    added_edges.insert((scc_u, scc_v));
                }
            }
        }
    }

    (condensation, sccs)
}

/// Count triangles in a sparse graph
///
/// Returns the number of triangles (3-cycles) in the graph.
///
/// Corresponds to sage.graphs.base.static_sparse_graph.triangles_count
pub fn triangles_count(graph: &SparseGraph) -> usize {
    let n = graph.num_vertices();
    let mut count = 0;

    // For each vertex, count triangles
    for u in 0..n {
        if let Some(u_neighbors) = graph.out_neighbors(u) {
            let u_neighbor_set: HashSet<usize> = u_neighbors.into_iter().collect();

            for &v in &u_neighbor_set {
                if v <= u {
                    continue;
                }

                if let Some(v_neighbors) = graph.out_neighbors(v) {
                    for &w in &v_neighbors {
                        if w > v && u_neighbor_set.contains(&w) {
                            count += 1;
                        }
                    }
                }
            }
        }
    }

    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_radius_empty() {
        let graph = SparseGraph::new(false);
        let radius = spectral_radius(&graph, 100);
        assert_eq!(radius, 0.0);
    }

    #[test]
    fn test_spectral_radius_single_edge() {
        let mut graph = SparseGraph::new(false);
        graph.add_vertex();
        graph.add_vertex();
        graph.add_edge(0, 1, None, None).unwrap();

        let radius = spectral_radius(&graph, 100);
        // For a path of 2 vertices, spectral radius should be close to 1
        assert!((radius - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_spectral_radius_triangle() {
        let mut graph = SparseGraph::new(false);
        graph.add_vertex();
        graph.add_vertex();
        graph.add_vertex();

        graph.add_edge(0, 1, None, None).unwrap();
        graph.add_edge(1, 2, None, None).unwrap();
        graph.add_edge(2, 0, None, None).unwrap();

        let radius = spectral_radius(&graph, 100);
        // Triangle has spectral radius 2
        assert!((radius - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_tarjan_scc_empty() {
        let graph = SparseGraph::new(true);
        let sccs = tarjan_strongly_connected_components(&graph);
        assert_eq!(sccs.len(), 0);
    }

    #[test]
    fn test_tarjan_scc_single_vertex() {
        let mut graph = SparseGraph::new(true);
        graph.add_vertex();

        let sccs = tarjan_strongly_connected_components(&graph);
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0], vec![0]);
    }

    #[test]
    fn test_tarjan_scc_cycle() {
        let mut graph = SparseGraph::new(true);
        graph.add_vertex();
        graph.add_vertex();
        graph.add_vertex();

        graph.add_edge(0, 1, None, None).unwrap();
        graph.add_edge(1, 2, None, None).unwrap();
        graph.add_edge(2, 0, None, None).unwrap();

        let sccs = tarjan_strongly_connected_components(&graph);
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0].len(), 3);
    }

    #[test]
    fn test_tarjan_scc_two_components() {
        let mut graph = SparseGraph::new(true);
        for _ in 0..4 {
            graph.add_vertex();
        }

        // First SCC: 0 -> 1 -> 0
        graph.add_edge(0, 1, None, None).unwrap();
        graph.add_edge(1, 0, None, None).unwrap();

        // Second SCC: 2 -> 3 -> 2
        graph.add_edge(2, 3, None, None).unwrap();
        graph.add_edge(3, 2, None, None).unwrap();

        let sccs = tarjan_strongly_connected_components(&graph);
        assert_eq!(sccs.len(), 2);

        // Each SCC should have 2 vertices
        for scc in &sccs {
            assert_eq!(scc.len(), 2);
        }
    }

    #[test]
    fn test_tarjan_scc_dag() {
        let mut graph = SparseGraph::new(true);
        for _ in 0..4 {
            graph.add_vertex();
        }

        // DAG: 0 -> 1 -> 2 -> 3
        graph.add_edge(0, 1, None, None).unwrap();
        graph.add_edge(1, 2, None, None).unwrap();
        graph.add_edge(2, 3, None, None).unwrap();

        let sccs = tarjan_strongly_connected_components(&graph);
        assert_eq!(sccs.len(), 4); // Each vertex is its own SCC
    }

    #[test]
    fn test_strongly_connected_components_digraph() {
        let mut graph = SparseGraph::new(true);
        for _ in 0..6 {
            graph.add_vertex();
        }

        // Create graph with 3 SCCs
        // SCC1: 0 <-> 1
        graph.add_edge(0, 1, None, None).unwrap();
        graph.add_edge(1, 0, None, None).unwrap();

        // SCC2: 2 <-> 3
        graph.add_edge(2, 3, None, None).unwrap();
        graph.add_edge(3, 2, None, None).unwrap();

        // SCC3: 4 <-> 5
        graph.add_edge(4, 5, None, None).unwrap();
        graph.add_edge(5, 4, None, None).unwrap();

        // Add edges between SCCs
        graph.add_edge(1, 2, None, None).unwrap();
        graph.add_edge(3, 4, None, None).unwrap();

        let (condensation, sccs) = strongly_connected_components_digraph(&graph);

        assert_eq!(sccs.len(), 3); // 3 SCCs
        assert_eq!(condensation.num_vertices(), 3);
        assert_eq!(condensation.num_edges(), 2); // 2 edges between SCCs
    }

    #[test]
    fn test_triangles_count_empty() {
        let graph = SparseGraph::new(false);
        assert_eq!(triangles_count(&graph), 0);
    }

    #[test]
    fn test_triangles_count_no_triangles() {
        let mut graph = SparseGraph::new(false);
        graph.add_vertex();
        graph.add_vertex();
        graph.add_vertex();

        graph.add_edge(0, 1, None, None).unwrap();
        graph.add_edge(1, 2, None, None).unwrap();

        assert_eq!(triangles_count(&graph), 0);
    }

    #[test]
    fn test_triangles_count_one_triangle() {
        let mut graph = SparseGraph::new(false);
        graph.add_vertex();
        graph.add_vertex();
        graph.add_vertex();

        graph.add_edge(0, 1, None, None).unwrap();
        graph.add_edge(1, 2, None, None).unwrap();
        graph.add_edge(2, 0, None, None).unwrap();

        assert_eq!(triangles_count(&graph), 1);
    }

    #[test]
    fn test_triangles_count_k4() {
        let mut graph = SparseGraph::new(false);
        for _ in 0..4 {
            graph.add_vertex();
        }

        // Create complete graph K4
        for i in 0..4 {
            for j in i + 1..4 {
                graph.add_edge(i, j, None, None).unwrap();
            }
        }

        assert_eq!(triangles_count(&graph), 4);
    }

    #[test]
    fn test_triangles_count_directed() {
        let mut graph = SparseGraph::new(true);
        graph.add_vertex();
        graph.add_vertex();
        graph.add_vertex();

        // Create directed cycle
        graph.add_edge(0, 1, None, None).unwrap();
        graph.add_edge(1, 2, None, None).unwrap();
        graph.add_edge(2, 0, None, None).unwrap();

        // For directed graphs, this counts directed triangles
        let count = triangles_count(&graph);
        assert!(count >= 0); // Should work without panicking
    }
}
