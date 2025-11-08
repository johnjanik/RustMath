//! Multigraph data structures - graphs with multiple edges between vertices

use std::collections::HashMap;

/// A multigraph allowing multiple edges between the same pair of vertices
#[derive(Debug, Clone)]
pub struct MultiGraph {
    /// Number of vertices
    num_vertices: usize,
    /// Adjacency list: vertex -> list of (neighbor, edge_id) pairs
    /// Multiple edges to the same neighbor are allowed
    adj: Vec<Vec<(usize, usize)>>,
    /// Edge multiplicity: counts how many edges exist between each pair
    edge_count: HashMap<(usize, usize), usize>,
    /// Next edge ID
    next_edge_id: usize,
}

impl MultiGraph {
    /// Create a new multigraph with n vertices
    pub fn new(n: usize) -> Self {
        MultiGraph {
            num_vertices: n,
            adj: vec![Vec::new(); n],
            edge_count: HashMap::new(),
            next_edge_id: 0,
        }
    }

    /// Get the number of vertices
    pub fn num_vertices(&self) -> usize {
        self.num_vertices
    }

    /// Get the total number of edges (counting multiplicities)
    pub fn num_edges(&self) -> usize {
        self.adj.iter().map(|neighbors| neighbors.len()).sum::<usize>() / 2
    }

    /// Add an edge between vertices u and v
    ///
    /// Returns the edge ID. Multiple edges between the same vertices are allowed.
    pub fn add_edge(&mut self, u: usize, v: usize) -> Result<usize, String> {
        if u >= self.num_vertices || v >= self.num_vertices {
            return Err(format!("Vertex out of bounds"));
        }

        let edge_id = self.next_edge_id;
        self.next_edge_id += 1;

        self.adj[u].push((v, edge_id));
        self.adj[v].push((u, edge_id));

        // Update edge count
        let key = if u < v { (u, v) } else { (v, u) };
        *self.edge_count.entry(key).or_insert(0) += 1;

        Ok(edge_id)
    }

    /// Get the multiplicity of edges between two vertices
    pub fn edge_multiplicity(&self, u: usize, v: usize) -> usize {
        let key = if u < v { (u, v) } else { (v, u) };
        *self.edge_count.get(&key).unwrap_or(&0)
    }

    /// Get all edges as (u, v, edge_id) tuples
    pub fn edges(&self) -> Vec<(usize, usize, usize)> {
        let mut edges = Vec::new();
        for u in 0..self.num_vertices {
            for &(v, edge_id) in &self.adj[u] {
                if u < v {
                    edges.push((u, v, edge_id));
                }
            }
        }
        edges
    }

    /// Get all vertices
    pub fn vertices(&self) -> Vec<usize> {
        (0..self.num_vertices).collect()
    }

    /// Get neighbors of a vertex (with multiplicities)
    pub fn neighbors(&self, v: usize) -> Vec<(usize, usize)> {
        if v >= self.num_vertices {
            return vec![];
        }
        self.adj[v].clone()
    }

    /// Get the degree of a vertex (counting multiple edges)
    pub fn degree(&self, v: usize) -> Option<usize> {
        if v >= self.num_vertices {
            return None;
        }
        Some(self.adj[v].len())
    }

    /// Check if the multigraph is connected
    pub fn is_connected(&self) -> bool {
        if self.num_vertices == 0 {
            return true;
        }

        let mut visited = vec![false; self.num_vertices];
        let mut stack = vec![0];
        visited[0] = true;
        let mut count = 1;

        while let Some(v) = stack.pop() {
            for &(neighbor, _) in &self.adj[v] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    stack.push(neighbor);
                    count += 1;
                }
            }
        }

        count == self.num_vertices
    }

    /// Get the simple underlying graph (removing multiple edges)
    pub fn to_simple_graph(&self) -> crate::Graph {
        let mut simple = crate::Graph::new(self.num_vertices);

        for &(u, v) in self.edge_count.keys() {
            simple.add_edge(u, v).ok();
        }

        simple
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multigraph_creation() {
        let g = MultiGraph::new(5);
        assert_eq!(g.num_vertices(), 5);
        assert_eq!(g.num_edges(), 0);
    }

    #[test]
    fn test_add_multiple_edges() {
        let mut g = MultiGraph::new(3);

        // Add multiple edges between same vertices
        let e1 = g.add_edge(0, 1).unwrap();
        let e2 = g.add_edge(0, 1).unwrap();
        let e3 = g.add_edge(0, 1).unwrap();

        assert_ne!(e1, e2);
        assert_ne!(e2, e3);
        assert_eq!(g.num_edges(), 3);
        assert_eq!(g.edge_multiplicity(0, 1), 3);
        assert_eq!(g.edge_multiplicity(1, 0), 3);
    }

    #[test]
    fn test_multigraph_degree() {
        let mut g = MultiGraph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 2).unwrap();

        assert_eq!(g.degree(0), Some(3)); // Two edges to 1, one to 2
        assert_eq!(g.degree(1), Some(2)); // Two edges to 0
        assert_eq!(g.degree(2), Some(1)); // One edge to 0
    }

    #[test]
    fn test_multigraph_to_simple() {
        let mut mg = MultiGraph::new(3);
        mg.add_edge(0, 1).unwrap();
        mg.add_edge(0, 1).unwrap();
        mg.add_edge(1, 2).unwrap();

        let simple = mg.to_simple_graph();
        assert_eq!(simple.num_edges(), 2); // Only 2 edges in simple graph
    }

    #[test]
    fn test_multigraph_connectivity() {
        let mut g = MultiGraph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        assert!(g.is_connected());

        // Disconnected multigraph
        let mut g2 = MultiGraph::new(4);
        g2.add_edge(0, 1).unwrap();
        g2.add_edge(2, 3).unwrap();

        assert!(!g2.is_connected());
    }
}
