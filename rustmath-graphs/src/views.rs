//! Graph views for edges and vertices
//!
//! This module provides view objects for iterating over graph elements
//! without copying data. Views provide a read-only interface to the
//! underlying graph structure.

use crate::graph::Graph;

/// A view of the edges in a graph
///
/// `EdgesView` provides a lazy iterator over the edges of a graph without
/// copying the graph data. It can be used to efficiently iterate over all
/// edges or filter edges based on specific criteria.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::{Graph, views::EdgesView};
///
/// let mut g = Graph::new(3);
/// g.add_edge(0, 1).unwrap();
/// g.add_edge(1, 2).unwrap();
///
/// let edges = EdgesView::new(&g);
/// assert_eq!(edges.len(), 2);
/// ```
#[derive(Debug)]
pub struct EdgesView<'a> {
    graph: &'a Graph,
    current_u: usize,
    current_v: usize,
}

impl<'a> EdgesView<'a> {
    /// Create a new edges view for a graph
    ///
    /// # Arguments
    ///
    /// * `graph` - The graph to create a view for
    ///
    /// # Returns
    ///
    /// A new `EdgesView` instance
    pub fn new(graph: &'a Graph) -> Self {
        EdgesView {
            graph,
            current_u: 0,
            current_v: 0,
        }
    }

    /// Get the number of edges in the graph
    ///
    /// # Returns
    ///
    /// The total number of edges
    pub fn len(&self) -> usize {
        self.graph.num_edges()
    }

    /// Check if the view is empty (no edges)
    ///
    /// # Returns
    ///
    /// `true` if there are no edges, `false` otherwise
    pub fn is_empty(&self) -> bool {
        self.graph.num_edges() == 0
    }

    /// Collect all edges into a vector
    ///
    /// Returns a vector of tuples (u, v) representing edges.
    /// For undirected graphs, each edge appears once with u < v.
    ///
    /// # Returns
    ///
    /// A vector of edge tuples
    pub fn edges(&self) -> Vec<(usize, usize)> {
        let mut result = Vec::new();
        let n = self.graph.num_vertices();

        for u in 0..n {
            if let Some(neighbors) = self.graph.neighbors(u) {
                for &v in &neighbors {
                    // For undirected graphs, only include each edge once
                    if u < v {
                        result.push((u, v));
                    }
                }
            }
        }

        result
    }

    /// Check if a specific edge exists
    ///
    /// # Arguments
    ///
    /// * `u` - First vertex
    /// * `v` - Second vertex
    ///
    /// # Returns
    ///
    /// `true` if the edge (u, v) exists, `false` otherwise
    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        self.graph.has_edge(u, v)
    }

    /// Get edges incident to a specific vertex
    ///
    /// # Arguments
    ///
    /// * `vertex` - The vertex to get incident edges for
    ///
    /// # Returns
    ///
    /// A vector of edges incident to the vertex
    pub fn incident_edges(&self, vertex: usize) -> Vec<(usize, usize)> {
        let mut result = Vec::new();

        if let Some(neighbors) = self.graph.neighbors(vertex) {
            for &v in &neighbors {
                if vertex < v {
                    result.push((vertex, v));
                } else {
                    result.push((v, vertex));
                }
            }
        }

        result
    }
}

impl<'a> Iterator for EdgesView<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        let n = self.graph.num_vertices();

        while self.current_u < n {
            if let Some(neighbors) = self.graph.neighbors(self.current_u) {
                while self.current_v < neighbors.len() {
                    let v = neighbors[self.current_v];
                    self.current_v += 1;

                    // Only return each edge once (u < v)
                    if self.current_u < v {
                        return Some((self.current_u, v));
                    }
                }
            }

            self.current_u += 1;
            self.current_v = 0;
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Graph;

    #[test]
    fn test_empty_graph() {
        let g = Graph::new(3);
        let view = EdgesView::new(&g);

        assert_eq!(view.len(), 0);
        assert!(view.is_empty());
        assert_eq!(view.edges().len(), 0);
    }

    #[test]
    fn test_single_edge() {
        let mut g = Graph::new(2);
        g.add_edge(0, 1).unwrap();

        let view = EdgesView::new(&g);

        assert_eq!(view.len(), 1);
        assert!(!view.is_empty());
        assert!(view.has_edge(0, 1));
        assert!(view.has_edge(1, 0)); // Undirected

        let edges = view.edges();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0], (0, 1));
    }

    #[test]
    fn test_multiple_edges() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();
        g.add_edge(0, 3).unwrap();

        let view = EdgesView::new(&g);

        assert_eq!(view.len(), 4);
        assert!(!view.is_empty());

        let edges = view.edges();
        assert_eq!(edges.len(), 4);

        // Check all edges exist
        assert!(view.has_edge(0, 1));
        assert!(view.has_edge(1, 2));
        assert!(view.has_edge(2, 3));
        assert!(view.has_edge(0, 3));
    }

    #[test]
    fn test_complete_graph() {
        let mut g = Graph::new(4);
        for i in 0..4 {
            for j in (i+1)..4 {
                g.add_edge(i, j).unwrap();
            }
        }

        let view = EdgesView::new(&g);

        // K4 has 6 edges
        assert_eq!(view.len(), 6);

        let edges = view.edges();
        assert_eq!(edges.len(), 6);
    }

    #[test]
    fn test_incident_edges() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 2).unwrap();
        g.add_edge(0, 3).unwrap();
        g.add_edge(1, 2).unwrap();

        let view = EdgesView::new(&g);

        let incident = view.incident_edges(0);
        assert_eq!(incident.len(), 3);

        let incident = view.incident_edges(1);
        assert_eq!(incident.len(), 2);

        let incident = view.incident_edges(3);
        assert_eq!(incident.len(), 1);
    }

    #[test]
    fn test_iterator() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(0, 2).unwrap();

        let view = EdgesView::new(&g);
        let collected: Vec<_> = view.collect();

        assert_eq!(collected.len(), 3);
    }

    #[test]
    fn test_iterator_empty() {
        let g = Graph::new(3);
        let view = EdgesView::new(&g);
        let collected: Vec<_> = view.collect();

        assert_eq!(collected.len(), 0);
    }

    #[test]
    fn test_no_self_loops() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        let view = EdgesView::new(&g);

        assert!(!view.has_edge(0, 0));
        assert!(!view.has_edge(1, 1));
        assert!(!view.has_edge(2, 2));
    }
}
