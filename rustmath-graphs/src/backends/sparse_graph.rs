//! Sparse graph implementation using adjacency lists
//!
//! Corresponds to sage.graphs.base.sparse_graph

use super::generic_backend::GenericGraphBackend;
use std::collections::{HashMap, HashSet};

/// Sparse graph using adjacency list representation
///
/// This is efficient for sparse graphs where most vertex pairs don't have edges.
/// Corresponds to SageMath's SparseGraph class.
#[derive(Debug, Clone)]
pub struct SparseGraph {
    /// Whether the graph is directed
    directed: bool,
    /// Number of vertices
    num_vertices: usize,
    /// Out-adjacency lists
    out_neighbors: Vec<HashSet<usize>>,
    /// In-adjacency lists (for directed graphs)
    in_neighbors: Vec<HashSet<usize>>,
    /// Edge labels
    edge_labels: HashMap<(usize, usize), String>,
    /// Whether loops are allowed
    loops: bool,
    /// Whether multiple edges are allowed
    multiedges: bool,
}

impl SparseGraph {
    /// Create a new sparse graph with options
    pub fn new_with_options(directed: bool, loops: bool, multiedges: bool) -> Self {
        SparseGraph {
            directed,
            num_vertices: 0,
            out_neighbors: Vec::new(),
            in_neighbors: if directed { Vec::new() } else { Vec::new() },
            edge_labels: HashMap::new(),
            loops,
            multiedges,
        }
    }

    /// Create with specified initial capacity
    pub fn with_capacity(directed: bool, capacity: usize) -> Self {
        SparseGraph {
            directed,
            num_vertices: capacity,
            out_neighbors: vec![HashSet::new(); capacity],
            in_neighbors: if directed { vec![HashSet::new(); capacity] } else { Vec::new() },
            edge_labels: HashMap::new(),
            loops: false,
            multiedges: false,
        }
    }

    /// Get all in-neighbors of a vertex (for directed graphs)
    pub fn in_neighbors(&self, v: usize) -> Option<Vec<usize>> {
        if v >= self.num_vertices {
            return None;
        }
        if self.directed {
            Some(self.in_neighbors[v].iter().copied().collect())
        } else {
            // For undirected graphs, in-neighbors = out-neighbors
            Some(self.out_neighbors[v].iter().copied().collect())
        }
    }

    /// Get all out-neighbors of a vertex
    pub fn out_neighbors(&self, v: usize) -> Option<Vec<usize>> {
        if v >= self.num_vertices {
            return None;
        }
        Some(self.out_neighbors[v].iter().copied().collect())
    }
}

impl GenericGraphBackend for SparseGraph {
    fn new(directed: bool) -> Self {
        Self::new_with_options(directed, false, false)
    }

    fn is_directed(&self) -> bool {
        self.directed
    }

    fn num_vertices(&self) -> usize {
        self.num_vertices
    }

    fn num_edges(&self) -> usize {
        if self.directed {
            self.out_neighbors.iter().map(|n| n.len()).sum()
        } else {
            self.out_neighbors.iter().map(|n| n.len()).sum::<usize>() / 2
        }
    }

    fn add_vertex(&mut self) -> usize {
        let idx = self.num_vertices;
        self.num_vertices += 1;
        self.out_neighbors.push(HashSet::new());
        if self.directed {
            self.in_neighbors.push(HashSet::new());
        }
        idx
    }

    fn add_edge(&mut self, u: usize, v: usize, label: Option<String>, _directed: Option<bool>) -> Result<(), String> {
        if u >= self.num_vertices || v >= self.num_vertices {
            return Err(format!("Vertex out of bounds"));
        }

        if !self.loops && u == v {
            return Err("Loops not allowed".to_string());
        }

        self.out_neighbors[u].insert(v);
        if self.directed {
            self.in_neighbors[v].insert(u);
        } else {
            self.out_neighbors[v].insert(u);
        }

        if let Some(lbl) = label {
            self.edge_labels.insert((u, v), lbl);
        }

        Ok(())
    }

    fn del_edge(&mut self, u: usize, v: usize) -> Result<(), String> {
        if u >= self.num_vertices || v >= self.num_vertices {
            return Err("Vertex out of bounds".to_string());
        }

        self.out_neighbors[u].remove(&v);
        if self.directed {
            self.in_neighbors[v].remove(&u);
        } else {
            self.out_neighbors[v].remove(&u);
        }

        self.edge_labels.remove(&(u, v));

        Ok(())
    }

    fn has_edge(&self, u: usize, v: usize) -> bool {
        if u >= self.num_vertices || v >= self.num_vertices {
            return false;
        }
        self.out_neighbors[u].contains(&v)
    }

    fn neighbors(&self, v: usize) -> Option<Vec<usize>> {
        self.out_neighbors(v)
    }

    fn in_degree(&self, v: usize) -> Option<usize> {
        if v >= self.num_vertices {
            return None;
        }
        if self.directed {
            Some(self.in_neighbors[v].len())
        } else {
            Some(self.out_neighbors[v].len())
        }
    }

    fn out_degree(&self, v: usize) -> Option<usize> {
        if v >= self.num_vertices {
            return None;
        }
        Some(self.out_neighbors[v].len())
    }

    fn edges(&self) -> Vec<(usize, usize)> {
        let mut result = Vec::new();
        for u in 0..self.num_vertices {
            for &v in &self.out_neighbors[u] {
                if self.directed || u <= v {
                    result.push((u, v));
                }
            }
        }
        result
    }

    fn allows_loops(&self) -> bool {
        self.loops
    }

    fn allows_multiple_edges(&self) -> bool {
        self.multiedges
    }

    fn get_edge_label(&self, u: usize, v: usize) -> Option<String> {
        self.edge_labels.get(&(u, v)).cloned()
    }

    fn set_edge_label(&mut self, u: usize, v: usize, label: Option<String>) -> Result<(), String> {
        if !self.has_edge(u, v) {
            return Err("Edge does not exist".to_string());
        }

        if let Some(lbl) = label {
            self.edge_labels.insert((u, v), lbl);
        } else {
            self.edge_labels.remove(&(u, v));
        }

        Ok(())
    }
}

/// Sparse graph backend wrapper
///
/// Wraps SparseGraph to provide compatibility with CGraphBackend interface.
/// Corresponds to SageMath's SparseGraphBackend.
#[derive(Debug, Clone)]
pub struct SparseGraphBackend {
    inner: SparseGraph,
}

impl SparseGraphBackend {
    /// Create a new sparse graph backend
    pub fn new(directed: bool) -> Self {
        SparseGraphBackend {
            inner: SparseGraph::new(directed),
        }
    }

    /// Create with specified initial capacity
    pub fn with_capacity(directed: bool, capacity: usize) -> Self {
        SparseGraphBackend {
            inner: SparseGraph::with_capacity(directed, capacity),
        }
    }

    /// Create with specific options
    pub fn new_with_options(directed: bool, loops: bool, multiedges: bool) -> Self {
        SparseGraphBackend {
            inner: SparseGraph::new_with_options(directed, loops, multiedges),
        }
    }

    /// Get a reference to the underlying sparse graph
    pub fn inner(&self) -> &SparseGraph {
        &self.inner
    }
}

impl GenericGraphBackend for SparseGraphBackend {
    fn new(directed: bool) -> Self {
        Self::new(directed)
    }

    fn is_directed(&self) -> bool {
        self.inner.is_directed()
    }

    fn num_vertices(&self) -> usize {
        self.inner.num_vertices()
    }

    fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    fn add_vertex(&mut self) -> usize {
        self.inner.add_vertex()
    }

    fn add_edge(&mut self, u: usize, v: usize, label: Option<String>, directed: Option<bool>) -> Result<(), String> {
        self.inner.add_edge(u, v, label, directed)
    }

    fn del_edge(&mut self, u: usize, v: usize) -> Result<(), String> {
        self.inner.del_edge(u, v)
    }

    fn has_edge(&self, u: usize, v: usize) -> bool {
        self.inner.has_edge(u, v)
    }

    fn neighbors(&self, v: usize) -> Option<Vec<usize>> {
        self.inner.neighbors(v)
    }

    fn in_degree(&self, v: usize) -> Option<usize> {
        self.inner.in_degree(v)
    }

    fn out_degree(&self, v: usize) -> Option<usize> {
        self.inner.out_degree(v)
    }

    fn edges(&self) -> Vec<(usize, usize)> {
        self.inner.edges()
    }

    fn allows_loops(&self) -> bool {
        self.inner.allows_loops()
    }

    fn allows_multiple_edges(&self) -> bool {
        self.inner.allows_multiple_edges()
    }

    fn get_edge_label(&self, u: usize, v: usize) -> Option<String> {
        self.inner.get_edge_label(u, v)
    }

    fn set_edge_label(&mut self, u: usize, v: usize, label: Option<String>) -> Result<(), String> {
        self.inner.set_edge_label(u, v, label)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_graph_creation() {
        let graph = SparseGraph::new(false);
        assert!(!graph.is_directed());
        assert_eq!(graph.num_vertices(), 0);
        assert_eq!(graph.num_edges(), 0);
    }

    #[test]
    fn test_sparse_graph_add_vertices() {
        let mut graph = SparseGraph::new(false);
        let v0 = graph.add_vertex();
        let v1 = graph.add_vertex();
        let v2 = graph.add_vertex();

        assert_eq!(v0, 0);
        assert_eq!(v1, 1);
        assert_eq!(v2, 2);
        assert_eq!(graph.num_vertices(), 3);
    }

    #[test]
    fn test_sparse_graph_add_edges() {
        let mut graph = SparseGraph::new(false);
        graph.add_vertex();
        graph.add_vertex();
        graph.add_vertex();

        graph.add_edge(0, 1, None, None).unwrap();
        graph.add_edge(1, 2, None, None).unwrap();

        assert_eq!(graph.num_edges(), 2);
        assert!(graph.has_edge(0, 1));
        assert!(graph.has_edge(1, 0)); // Undirected
        assert!(graph.has_edge(1, 2));
        assert!(!graph.has_edge(0, 2));
    }

    #[test]
    fn test_sparse_graph_directed() {
        let mut graph = SparseGraph::new(true);
        graph.add_vertex();
        graph.add_vertex();

        graph.add_edge(0, 1, None, None).unwrap();

        assert!(graph.has_edge(0, 1));
        assert!(!graph.has_edge(1, 0)); // Directed
        assert_eq!(graph.num_edges(), 1);
    }

    #[test]
    fn test_sparse_graph_degrees() {
        let mut graph = SparseGraph::new(false);
        graph.add_vertex();
        graph.add_vertex();
        graph.add_vertex();

        graph.add_edge(0, 1, None, None).unwrap();
        graph.add_edge(1, 2, None, None).unwrap();

        assert_eq!(graph.out_degree(0), Some(1));
        assert_eq!(graph.out_degree(1), Some(2));
        assert_eq!(graph.out_degree(2), Some(1));
    }

    #[test]
    fn test_sparse_graph_neighbors() {
        let mut graph = SparseGraph::new(false);
        graph.add_vertex();
        graph.add_vertex();
        graph.add_vertex();

        graph.add_edge(0, 1, None, None).unwrap();
        graph.add_edge(0, 2, None, None).unwrap();

        let neighbors = graph.neighbors(0).unwrap();
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&1));
        assert!(neighbors.contains(&2));
    }

    #[test]
    fn test_sparse_graph_in_neighbors() {
        let mut graph = SparseGraph::new(true);
        graph.add_vertex();
        graph.add_vertex();
        graph.add_vertex();

        graph.add_edge(0, 1, None, None).unwrap();
        graph.add_edge(2, 1, None, None).unwrap();

        let in_neighbors = graph.in_neighbors(1).unwrap();
        assert_eq!(in_neighbors.len(), 2);
        assert!(in_neighbors.contains(&0));
        assert!(in_neighbors.contains(&2));

        let out_neighbors = graph.out_neighbors(1).unwrap();
        assert_eq!(out_neighbors.len(), 0);
    }

    #[test]
    fn test_sparse_graph_with_capacity() {
        let graph = SparseGraph::with_capacity(false, 10);
        assert_eq!(graph.num_vertices(), 10);
        assert_eq!(graph.num_edges(), 0);
    }

    #[test]
    fn test_sparse_graph_backend() {
        let mut backend = SparseGraphBackend::new(false);
        backend.add_vertex();
        backend.add_vertex();
        backend.add_vertex();

        backend.add_edge(0, 1, None, None).unwrap();
        backend.add_edge(1, 2, None, None).unwrap();

        assert_eq!(backend.num_edges(), 2);
        assert!(backend.has_edge(0, 1));
        assert!(backend.has_edge(1, 2));
    }

    #[test]
    fn test_sparse_graph_edge_deletion() {
        let mut graph = SparseGraph::new(false);
        graph.add_vertex();
        graph.add_vertex();

        graph.add_edge(0, 1, None, None).unwrap();
        assert!(graph.has_edge(0, 1));

        graph.del_edge(0, 1).unwrap();
        assert!(!graph.has_edge(0, 1));
        assert_eq!(graph.num_edges(), 0);
    }

    #[test]
    fn test_sparse_graph_loops() {
        let mut graph = SparseGraph::new_with_options(false, false, false);
        graph.add_vertex();

        // Loops not allowed
        assert!(graph.add_edge(0, 0, None, None).is_err());

        // Create graph with loops allowed
        let mut graph2 = SparseGraph::new_with_options(false, true, false);
        graph2.add_vertex();

        assert!(graph2.add_edge(0, 0, None, None).is_ok());
        assert!(graph2.has_edge(0, 0));
    }

    #[test]
    fn test_sparse_graph_labels() {
        let mut graph = SparseGraph::new(false);
        graph.add_vertex();
        graph.add_vertex();

        graph.add_edge(0, 1, Some("edge01".to_string()), None).unwrap();

        assert_eq!(graph.get_edge_label(0, 1), Some("edge01".to_string()));

        graph.set_edge_label(0, 1, Some("new_label".to_string())).unwrap();
        assert_eq!(graph.get_edge_label(0, 1), Some("new_label".to_string()));
    }
}
