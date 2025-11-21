//! Dense graph implementation using adjacency matrix
//!
//! Corresponds to sage.graphs.base.dense_graph

use super::generic_backend::GenericGraphBackend;
use std::collections::HashMap;

/// Dense graph using adjacency matrix representation
///
/// This is efficient for dense graphs where most vertex pairs have edges.
/// Corresponds to SageMath's DenseGraph class.
#[derive(Debug, Clone)]
pub struct DenseGraph {
    /// Whether the graph is directed
    directed: bool,
    /// Number of vertices
    num_vertices: usize,
    /// Adjacency matrix (stored as flat vector for efficiency)
    /// For vertex i and j: adj_matrix[i * num_vertices + j] indicates if edge exists
    adj_matrix: Vec<bool>,
    /// Edge labels
    edge_labels: HashMap<(usize, usize), String>,
    /// Whether loops are allowed
    loops: bool,
}

impl DenseGraph {
    /// Create a new dense graph with specified capacity
    pub fn with_capacity(directed: bool, capacity: usize, loops: bool) -> Self {
        DenseGraph {
            directed,
            num_vertices: capacity,
            adj_matrix: vec![false; capacity * capacity],
            edge_labels: HashMap::new(),
            loops,
        }
    }

    /// Get the index in the flat adjacency matrix
    #[inline]
    fn matrix_index(&self, u: usize, v: usize) -> usize {
        u * self.num_vertices + v
    }

    /// Resize the adjacency matrix to accommodate more vertices
    fn resize(&mut self, new_size: usize) {
        if new_size <= self.num_vertices {
            return;
        }

        let mut new_matrix = vec![false; new_size * new_size];

        // Copy existing edges
        for i in 0..self.num_vertices {
            for j in 0..self.num_vertices {
                let old_idx = i * self.num_vertices + j;
                let new_idx = i * new_size + j;
                new_matrix[new_idx] = self.adj_matrix[old_idx];
            }
        }

        self.adj_matrix = new_matrix;
        self.num_vertices = new_size;
    }
}

impl GenericGraphBackend for DenseGraph {
    fn new(directed: bool) -> Self {
        Self::with_capacity(directed, 0, false)
    }

    fn is_directed(&self) -> bool {
        self.directed
    }

    fn num_vertices(&self) -> usize {
        self.num_vertices
    }

    fn num_edges(&self) -> usize {
        let mut count = 0;
        for i in 0..self.num_vertices {
            for j in 0..self.num_vertices {
                if self.adj_matrix[self.matrix_index(i, j)] {
                    if self.directed || i <= j {
                        count += 1;
                    }
                }
            }
        }
        count
    }

    fn add_vertex(&mut self) -> usize {
        let new_size = self.num_vertices + 1;
        self.resize(new_size);
        new_size - 1
    }

    fn add_edge(&mut self, u: usize, v: usize, label: Option<String>, _directed: Option<bool>) -> Result<(), String> {
        if u >= self.num_vertices || v >= self.num_vertices {
            return Err(format!("Vertex out of bounds"));
        }

        if !self.loops && u == v {
            return Err("Loops not allowed".to_string());
        }

        let idx = self.matrix_index(u, v);
        self.adj_matrix[idx] = true;

        if !self.directed {
            let idx_rev = self.matrix_index(v, u);
            self.adj_matrix[idx_rev] = true;
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

        let idx = self.matrix_index(u, v);
        self.adj_matrix[idx] = false;

        if !self.directed {
            let idx_rev = self.matrix_index(v, u);
            self.adj_matrix[idx_rev] = false;
        }

        self.edge_labels.remove(&(u, v));

        Ok(())
    }

    fn has_edge(&self, u: usize, v: usize) -> bool {
        if u >= self.num_vertices || v >= self.num_vertices {
            return false;
        }
        self.adj_matrix[self.matrix_index(u, v)]
    }

    fn neighbors(&self, v: usize) -> Option<Vec<usize>> {
        if v >= self.num_vertices {
            return None;
        }

        let mut neighbors = Vec::new();
        for i in 0..self.num_vertices {
            if self.adj_matrix[self.matrix_index(v, i)] {
                neighbors.push(i);
            }
        }
        Some(neighbors)
    }

    fn in_degree(&self, v: usize) -> Option<usize> {
        if v >= self.num_vertices {
            return None;
        }

        let mut count = 0;
        for i in 0..self.num_vertices {
            if self.adj_matrix[self.matrix_index(i, v)] {
                count += 1;
            }
        }
        Some(count)
    }

    fn out_degree(&self, v: usize) -> Option<usize> {
        if v >= self.num_vertices {
            return None;
        }

        let mut count = 0;
        for i in 0..self.num_vertices {
            if self.adj_matrix[self.matrix_index(v, i)] {
                count += 1;
            }
        }
        Some(count)
    }

    fn edges(&self) -> Vec<(usize, usize)> {
        let mut result = Vec::new();
        for i in 0..self.num_vertices {
            for j in 0..self.num_vertices {
                if self.adj_matrix[self.matrix_index(i, j)] {
                    if self.directed || i <= j {
                        result.push((i, j));
                    }
                }
            }
        }
        result
    }

    fn allows_loops(&self) -> bool {
        self.loops
    }

    fn allows_multiple_edges(&self) -> bool {
        false // Dense adjacency matrix doesn't support multiple edges
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

/// Dense graph backend wrapper
///
/// Wraps DenseGraph to provide compatibility with CGraphBackend interface.
/// Corresponds to SageMath's DenseGraphBackend.
#[derive(Debug, Clone)]
pub struct DenseGraphBackend {
    inner: DenseGraph,
}

impl DenseGraphBackend {
    /// Create a new dense graph backend
    pub fn new(directed: bool) -> Self {
        DenseGraphBackend {
            inner: DenseGraph::new(directed),
        }
    }

    /// Create with specified initial capacity
    pub fn with_capacity(directed: bool, capacity: usize) -> Self {
        DenseGraphBackend {
            inner: DenseGraph::with_capacity(directed, capacity, false),
        }
    }

    /// Get a reference to the underlying dense graph
    pub fn inner(&self) -> &DenseGraph {
        &self.inner
    }
}

impl GenericGraphBackend for DenseGraphBackend {
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
    fn test_dense_graph_creation() {
        let graph = DenseGraph::new(false);
        assert!(!graph.is_directed());
        assert_eq!(graph.num_vertices(), 0);
        assert_eq!(graph.num_edges(), 0);
    }

    #[test]
    fn test_dense_graph_add_vertices() {
        let mut graph = DenseGraph::new(false);
        let v0 = graph.add_vertex();
        let v1 = graph.add_vertex();
        let v2 = graph.add_vertex();

        assert_eq!(v0, 0);
        assert_eq!(v1, 1);
        assert_eq!(v2, 2);
        assert_eq!(graph.num_vertices(), 3);
    }

    #[test]
    fn test_dense_graph_add_edges() {
        let mut graph = DenseGraph::new(false);
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
    fn test_dense_graph_directed() {
        let mut graph = DenseGraph::new(true);
        graph.add_vertex();
        graph.add_vertex();

        graph.add_edge(0, 1, None, None).unwrap();

        assert!(graph.has_edge(0, 1));
        assert!(!graph.has_edge(1, 0)); // Directed
        assert_eq!(graph.num_edges(), 1);
    }

    #[test]
    fn test_dense_graph_degrees() {
        let mut graph = DenseGraph::new(false);
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
    fn test_dense_graph_neighbors() {
        let mut graph = DenseGraph::new(false);
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
    fn test_dense_graph_with_capacity() {
        let graph = DenseGraph::with_capacity(false, 10, false);
        assert_eq!(graph.num_vertices(), 10);
        assert_eq!(graph.num_edges(), 0);
    }

    #[test]
    fn test_dense_graph_backend() {
        let mut backend = DenseGraphBackend::new(false);
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
    fn test_dense_graph_edge_deletion() {
        let mut graph = DenseGraph::new(false);
        graph.add_vertex();
        graph.add_vertex();

        graph.add_edge(0, 1, None, None).unwrap();
        assert!(graph.has_edge(0, 1));

        graph.del_edge(0, 1).unwrap();
        assert!(!graph.has_edge(0, 1));
        assert_eq!(graph.num_edges(), 0);
    }

    #[test]
    fn test_dense_graph_complete() {
        let mut graph = DenseGraph::new(false);
        for _ in 0..5 {
            graph.add_vertex();
        }

        // Create complete graph K5
        for i in 0..5 {
            for j in i + 1..5 {
                graph.add_edge(i, j, None, None).unwrap();
            }
        }

        assert_eq!(graph.num_edges(), 10); // K5 has 10 edges

        // Every vertex should have degree 4
        for i in 0..5 {
            assert_eq!(graph.out_degree(i), Some(4));
        }
    }
}
