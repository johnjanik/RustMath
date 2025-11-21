//! Generic graph backend trait
//!
//! Corresponds to sage.graphs.base.graph_backends.GenericGraphBackend


/// Generic graph backend trait
///
/// This trait defines the interface that all graph backend implementations must provide.
/// It supports both directed and undirected graphs with optional loops and multiple edges.
pub trait GenericGraphBackend: Clone {
    /// Create a new empty graph backend
    fn new(directed: bool) -> Self;

    /// Check if the graph is directed
    fn is_directed(&self) -> bool;

    /// Get the number of vertices
    fn num_vertices(&self) -> usize;

    /// Get the number of edges
    fn num_edges(&self) -> usize;

    /// Add a vertex to the graph
    ///
    /// Returns the index of the newly added vertex
    fn add_vertex(&mut self) -> usize;

    /// Add multiple vertices to the graph
    ///
    /// Returns the indices of the newly added vertices
    fn add_vertices(&mut self, count: usize) -> Vec<usize> {
        (0..count).map(|_| self.add_vertex()).collect()
    }

    /// Add an edge between two vertices
    ///
    /// # Arguments
    /// * `u` - Source vertex (or one endpoint for undirected)
    /// * `v` - Target vertex (or other endpoint for undirected)
    /// * `label` - Optional edge label
    /// * `directed` - Whether this specific edge is directed (for mixed graphs)
    fn add_edge(&mut self, u: usize, v: usize, label: Option<String>, directed: Option<bool>) -> Result<(), String>;

    /// Remove an edge between two vertices
    fn del_edge(&mut self, u: usize, v: usize) -> Result<(), String>;

    /// Check if there's an edge between two vertices
    fn has_edge(&self, u: usize, v: usize) -> bool;

    /// Get all neighbors of a vertex
    ///
    /// For directed graphs, this returns out-neighbors
    fn neighbors(&self, v: usize) -> Option<Vec<usize>>;

    /// Get the in-degree of a vertex
    ///
    /// For undirected graphs, this equals the degree
    fn in_degree(&self, v: usize) -> Option<usize>;

    /// Get the out-degree of a vertex
    ///
    /// For undirected graphs, this equals the degree
    fn out_degree(&self, v: usize) -> Option<usize>;

    /// Get the degree of a vertex
    fn degree(&self, v: usize) -> Option<usize> {
        if self.is_directed() {
            // For directed graphs, degree = in_degree + out_degree
            match (self.in_degree(v), self.out_degree(v)) {
                (Some(ind), Some(outd)) => Some(ind + outd),
                _ => None,
            }
        } else {
            self.out_degree(v)
        }
    }

    /// Get all vertices in the graph
    fn vertices(&self) -> Vec<usize> {
        (0..self.num_vertices()).collect()
    }

    /// Get all edges in the graph
    ///
    /// Returns a vector of (source, target) tuples
    fn edges(&self) -> Vec<(usize, usize)>;

    /// Check if the graph allows loops (self-edges)
    fn allows_loops(&self) -> bool;

    /// Check if the graph allows multiple edges between the same vertices
    fn allows_multiple_edges(&self) -> bool;

    /// Get the edge label between two vertices
    fn get_edge_label(&self, u: usize, v: usize) -> Option<String>;

    /// Set the edge label between two vertices
    fn set_edge_label(&mut self, u: usize, v: usize, label: Option<String>) -> Result<(), String>;
}

/// Unpickle a graph backend (deserialization helper)
///
/// Corresponds to sage.graphs.base.graph_backends.unpickle_graph_backend
///
/// Note: This returns a concrete SparseGraphBackend type rather than a trait object,
/// since GenericGraphBackend requires Clone which makes it non-dyn-compatible.
pub fn unpickle_graph_backend(_data: &[u8]) -> Result<crate::backends::sparse_graph::SparseGraphBackend, String> {
    // In a real implementation, this would deserialize the graph from bytes
    // For now, we just return an error
    Err("Unpickling not yet implemented".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::sparse_graph::SparseGraphBackend;

    #[test]
    fn test_generic_backend_interface() {
        let mut backend = SparseGraphBackend::new(false);

        // Add vertices
        let v0 = backend.add_vertex();
        let v1 = backend.add_vertex();
        let v2 = backend.add_vertex();

        assert_eq!(backend.num_vertices(), 3);
        assert_eq!(v0, 0);
        assert_eq!(v1, 1);
        assert_eq!(v2, 2);

        // Add edges
        backend.add_edge(0, 1, None, None).unwrap();
        backend.add_edge(1, 2, None, None).unwrap();

        assert_eq!(backend.num_edges(), 2);
        assert!(backend.has_edge(0, 1));
        assert!(backend.has_edge(1, 2));
        assert!(!backend.has_edge(0, 2));

        // Check degrees
        assert_eq!(backend.degree(0), Some(1));
        assert_eq!(backend.degree(1), Some(2));
        assert_eq!(backend.degree(2), Some(1));
    }

    #[test]
    fn test_add_multiple_vertices() {
        let mut backend = SparseGraphBackend::new(false);
        let vertices = backend.add_vertices(5);

        assert_eq!(vertices.len(), 5);
        assert_eq!(backend.num_vertices(), 5);
        assert_eq!(vertices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_directed_graph() {
        let mut backend = SparseGraphBackend::new(true);

        backend.add_vertex();
        backend.add_vertex();
        backend.add_vertex();

        backend.add_edge(0, 1, None, None).unwrap();
        backend.add_edge(1, 2, None, None).unwrap();

        assert!(backend.is_directed());

        // In directed graph, edge 0->1 exists but 1->0 doesn't
        assert!(backend.has_edge(0, 1));
        assert!(!backend.has_edge(1, 0));
    }
}
