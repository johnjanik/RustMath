//! Static sparse backend implementation
//!
//! Corresponds to sage.graphs.base.static_sparse_backend
//!
//! This module provides an immutable (static) sparse graph representation
//! optimized for query performance.

use super::generic_backend::GenericGraphBackend;
use std::collections::HashMap;

/// Static sparse C-level graph
///
/// An immutable sparse graph representation using compressed storage.
/// Corresponds to SageMath's StaticSparseCGraph.
#[derive(Debug, Clone)]
pub struct StaticSparseCGraph {
    /// Whether the graph is directed
    directed: bool,
    /// Number of vertices
    num_vertices: usize,
    /// Number of edges
    num_edges: usize,
    /// Compressed row storage: indices where each vertex's neighbors start
    row_ptr: Vec<usize>,
    /// Column indices (neighbor list)
    col_ind: Vec<usize>,
    /// Edge labels (optional)
    edge_labels: HashMap<(usize, usize), String>,
}

impl StaticSparseCGraph {
    /// Create a new static sparse graph from edge list
    pub fn from_edges(directed: bool, num_vertices: usize, edges: &[(usize, usize)]) -> Self {
        let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); num_vertices];

        // Build neighbor lists
        for &(u, v) in edges {
            if u < num_vertices && v < num_vertices {
                neighbors[u].push(v);
                if !directed {
                    neighbors[v].push(u);
                }
            }
        }

        // Sort and deduplicate neighbors
        for neighbor_list in &mut neighbors {
            neighbor_list.sort_unstable();
            neighbor_list.dedup();
        }

        // Build compressed storage
        let mut row_ptr = vec![0];
        let mut col_ind = Vec::new();

        for neighbor_list in &neighbors {
            col_ind.extend(neighbor_list);
            row_ptr.push(col_ind.len());
        }

        let num_edges = if directed {
            edges.len()
        } else {
            edges.len() * 2 / 2
        };

        StaticSparseCGraph {
            directed,
            num_vertices,
            num_edges,
            row_ptr,
            col_ind,
            edge_labels: HashMap::new(),
        }
    }

    /// Get neighbors of a vertex
    pub fn neighbors(&self, v: usize) -> Option<&[usize]> {
        if v >= self.num_vertices {
            return None;
        }

        let start = self.row_ptr[v];
        let end = self.row_ptr[v + 1];
        Some(&self.col_ind[start..end])
    }

    /// Get the degree of a vertex
    pub fn degree(&self, v: usize) -> Option<usize> {
        if v >= self.num_vertices {
            return None;
        }

        let start = self.row_ptr[v];
        let end = self.row_ptr[v + 1];
        Some(end - start)
    }

    /// Check if an edge exists
    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        if let Some(neighbors) = self.neighbors(u) {
            neighbors.binary_search(&v).is_ok()
        } else {
            false
        }
    }

    /// Get the number of vertices
    pub fn num_vertices(&self) -> usize {
        self.num_vertices
    }

    /// Get the number of edges
    pub fn num_edges(&self) -> usize {
        self.num_edges
    }

    /// Check if the graph is directed
    pub fn is_directed(&self) -> bool {
        self.directed
    }
}

/// Static sparse graph backend
///
/// Wrapper around StaticSparseCGraph that implements GenericGraphBackend.
/// This backend is immutable after creation.
/// Corresponds to SageMath's StaticSparseBackend.
#[derive(Debug, Clone)]
pub struct StaticSparseBackend {
    graph: StaticSparseCGraph,
}

impl StaticSparseBackend {
    /// Create a new static sparse backend from edge list
    pub fn from_edges(directed: bool, num_vertices: usize, edges: &[(usize, usize)]) -> Self {
        StaticSparseBackend {
            graph: StaticSparseCGraph::from_edges(directed, num_vertices, edges),
        }
    }

    /// Get a reference to the underlying static graph
    pub fn inner(&self) -> &StaticSparseCGraph {
        &self.graph
    }
}

impl GenericGraphBackend for StaticSparseBackend {
    fn new(directed: bool) -> Self {
        StaticSparseBackend {
            graph: StaticSparseCGraph::from_edges(directed, 0, &[]),
        }
    }

    fn is_directed(&self) -> bool {
        self.graph.is_directed()
    }

    fn num_vertices(&self) -> usize {
        self.graph.num_vertices()
    }

    fn num_edges(&self) -> usize {
        self.graph.num_edges()
    }

    fn add_vertex(&mut self) -> usize {
        // Static graphs don't support modifications
        panic!("Cannot add vertex to static graph");
    }

    fn add_edge(&mut self, _u: usize, _v: usize, _label: Option<String>, _directed: Option<bool>) -> Result<(), String> {
        // Static graphs don't support modifications
        Err("Cannot add edge to static graph".to_string())
    }

    fn del_edge(&mut self, _u: usize, _v: usize) -> Result<(), String> {
        // Static graphs don't support modifications
        Err("Cannot delete edge from static graph".to_string())
    }

    fn has_edge(&self, u: usize, v: usize) -> bool {
        self.graph.has_edge(u, v)
    }

    fn neighbors(&self, v: usize) -> Option<Vec<usize>> {
        self.graph.neighbors(v).map(|slice| slice.to_vec())
    }

    fn in_degree(&self, v: usize) -> Option<usize> {
        if !self.graph.is_directed() {
            return self.graph.degree(v);
        }

        // For directed graphs, count in-edges
        let mut count = 0;
        for u in 0..self.graph.num_vertices() {
            if self.graph.has_edge(u, v) {
                count += 1;
            }
        }
        Some(count)
    }

    fn out_degree(&self, v: usize) -> Option<usize> {
        self.graph.degree(v)
    }

    fn edges(&self) -> Vec<(usize, usize)> {
        let mut result = Vec::new();
        for u in 0..self.graph.num_vertices() {
            if let Some(neighbors) = self.graph.neighbors(u) {
                for &v in neighbors {
                    if self.graph.is_directed() || u <= v {
                        result.push((u, v));
                    }
                }
            }
        }
        result
    }

    fn allows_loops(&self) -> bool {
        true // Static graphs can represent any structure
    }

    fn allows_multiple_edges(&self) -> bool {
        false // Compressed format deduplicates edges
    }

    fn get_edge_label(&self, u: usize, v: usize) -> Option<String> {
        self.graph.edge_labels.get(&(u, v)).cloned()
    }

    fn set_edge_label(&mut self, _u: usize, _v: usize, _label: Option<String>) -> Result<(), String> {
        // Static graphs don't support modifications
        Err("Cannot set edge label on static graph".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_sparse_cgraph_creation() {
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let graph = StaticSparseCGraph::from_edges(false, 3, &edges);

        assert_eq!(graph.num_vertices(), 3);
        assert!(graph.num_edges() > 0);
    }

    #[test]
    fn test_static_sparse_cgraph_neighbors() {
        let edges = vec![(0, 1), (1, 2), (0, 2)];
        let graph = StaticSparseCGraph::from_edges(false, 3, &edges);

        let neighbors = graph.neighbors(0).unwrap();
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&1));
        assert!(neighbors.contains(&2));
    }

    #[test]
    fn test_static_sparse_cgraph_has_edge() {
        let edges = vec![(0, 1), (1, 2)];
        let graph = StaticSparseCGraph::from_edges(false, 3, &edges);

        assert!(graph.has_edge(0, 1));
        assert!(graph.has_edge(1, 0)); // Undirected
        assert!(graph.has_edge(1, 2));
        assert!(!graph.has_edge(0, 2));
    }

    #[test]
    fn test_static_sparse_cgraph_directed() {
        let edges = vec![(0, 1), (1, 2)];
        let graph = StaticSparseCGraph::from_edges(true, 3, &edges);

        assert!(graph.is_directed());
        assert!(graph.has_edge(0, 1));
        assert!(!graph.has_edge(1, 0)); // Directed
    }

    #[test]
    fn test_static_sparse_cgraph_degree() {
        let edges = vec![(0, 1), (0, 2), (1, 2)];
        let graph = StaticSparseCGraph::from_edges(false, 3, &edges);

        assert_eq!(graph.degree(0), Some(2));
        assert_eq!(graph.degree(1), Some(2));
        assert_eq!(graph.degree(2), Some(2));
    }

    #[test]
    fn test_static_sparse_backend() {
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let backend = StaticSparseBackend::from_edges(false, 3, &edges);

        assert_eq!(backend.num_vertices(), 3);
        assert!(backend.has_edge(0, 1));
        assert!(backend.has_edge(1, 2));
        assert!(backend.has_edge(2, 0));
    }

    #[test]
    fn test_static_sparse_backend_neighbors() {
        let edges = vec![(0, 1), (0, 2)];
        let backend = StaticSparseBackend::from_edges(false, 3, &edges);

        let neighbors = backend.neighbors(0).unwrap();
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&1));
        assert!(neighbors.contains(&2));
    }

    #[test]
    fn test_static_sparse_backend_immutable() {
        let edges = vec![(0, 1)];
        let mut backend = StaticSparseBackend::from_edges(false, 2, &edges);

        // Should not be able to add edges
        assert!(backend.add_edge(0, 2, None, None).is_err());
    }

    #[test]
    fn test_static_sparse_backend_edges() {
        let edges = vec![(0, 1), (1, 2), (0, 2)];
        let backend = StaticSparseBackend::from_edges(false, 3, &edges);

        let edge_list = backend.edges();
        assert_eq!(edge_list.len(), 3);
    }

    #[test]
    fn test_static_sparse_cgraph_empty() {
        let graph = StaticSparseCGraph::from_edges(false, 5, &[]);

        assert_eq!(graph.num_vertices(), 5);
        assert_eq!(graph.num_edges(), 0);

        for v in 0..5 {
            assert_eq!(graph.degree(v), Some(0));
        }
    }

    #[test]
    fn test_static_sparse_cgraph_complete() {
        let mut edges = Vec::new();
        for i in 0..5 {
            for j in i + 1..5 {
                edges.push((i, j));
            }
        }

        let graph = StaticSparseCGraph::from_edges(false, 5, &edges);

        assert_eq!(graph.num_vertices(), 5);

        // All vertices should have degree 4 in K5
        for v in 0..5 {
            assert_eq!(graph.degree(v), Some(4));
        }
    }
}
