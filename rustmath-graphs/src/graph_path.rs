//! Path enumeration and generating functions for directed graphs
//!
//! This module provides algorithms for:
//! - Enumerating all paths between vertices
//! - Counting paths of specific lengths
//! - Computing generating functions for path counts
//! - Transfer matrix methods for path analysis

use crate::DiGraph;
use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_matrix::Matrix;
use rustmath_powerseries::PowerSeries;
use std::collections::HashSet;

/// Result type for paths: a path is a sequence of vertices
pub type Path = Vec<usize>;

/// Path enumeration and analysis for directed graphs
pub struct GraphPath<'a> {
    graph: &'a DiGraph,
}

impl<'a> GraphPath<'a> {
    /// Create a new path analyzer for the given graph
    pub fn new(graph: &'a DiGraph) -> Self {
        GraphPath { graph }
    }

    /// Find all simple paths from start to end with at most max_length edges
    ///
    /// A simple path visits each vertex at most once.
    ///
    /// # Arguments
    /// * `start` - Starting vertex
    /// * `end` - Ending vertex
    /// * `max_length` - Maximum path length (number of edges)
    ///
    /// # Returns
    /// Vector of all simple paths, where each path is a vector of vertices
    pub fn all_simple_paths(&self, start: usize, end: usize, max_length: usize) -> Vec<Path> {
        if start >= self.graph.num_vertices() || end >= self.graph.num_vertices() {
            return Vec::new();
        }

        let mut paths = Vec::new();
        let mut current_path = vec![start];
        let mut visited = HashSet::new();
        visited.insert(start);

        self.dfs_paths(start, end, max_length, &mut current_path, &mut visited, &mut paths);

        paths
    }

    /// DFS helper for finding all simple paths
    fn dfs_paths(
        &self,
        current: usize,
        end: usize,
        remaining: usize,
        path: &mut Path,
        visited: &mut HashSet<usize>,
        results: &mut Vec<Path>,
    ) {
        if current == end {
            results.push(path.clone());
            return;
        }

        if remaining == 0 {
            return;
        }

        // Get neighbors
        let neighbors: Vec<usize> = self
            .graph
            .edges()
            .iter()
            .filter(|(u, _)| *u == current)
            .map(|(_, v)| *v)
            .collect();

        for neighbor in neighbors {
            if !visited.contains(&neighbor) {
                path.push(neighbor);
                visited.insert(neighbor);

                self.dfs_paths(neighbor, end, remaining - 1, path, visited, results);

                path.pop();
                visited.remove(&neighbor);
            }
        }
    }

    /// Find all paths from start to end with exactly the specified length
    ///
    /// Unlike all_simple_paths, this allows revisiting vertices.
    ///
    /// # Arguments
    /// * `start` - Starting vertex
    /// * `end` - Ending vertex
    /// * `length` - Exact path length (number of edges)
    ///
    /// # Returns
    /// Vector of all paths with the specified length
    pub fn all_paths_of_length(&self, start: usize, end: usize, length: usize) -> Vec<Path> {
        if start >= self.graph.num_vertices() || end >= self.graph.num_vertices() {
            return Vec::new();
        }

        if length == 0 {
            return if start == end {
                vec![vec![start]]
            } else {
                Vec::new()
            };
        }

        let mut paths = Vec::new();
        let mut current_path = vec![start];

        self.dfs_exact_length(start, end, length, &mut current_path, &mut paths);

        paths
    }

    /// DFS helper for finding paths of exact length
    fn dfs_exact_length(
        &self,
        current: usize,
        end: usize,
        remaining: usize,
        path: &mut Path,
        results: &mut Vec<Path>,
    ) {
        if remaining == 0 {
            if current == end {
                results.push(path.clone());
            }
            return;
        }

        // Get neighbors
        let neighbors: Vec<usize> = self
            .graph
            .edges()
            .iter()
            .filter(|(u, _)| *u == current)
            .map(|(_, v)| *v)
            .collect();

        for neighbor in neighbors {
            path.push(neighbor);
            self.dfs_exact_length(neighbor, end, remaining - 1, path, results);
            path.pop();
        }
    }

    /// Count the number of paths from start to end with exactly k edges
    ///
    /// Uses matrix exponentiation for efficient counting.
    /// The count is given by A^k[start][end] where A is the adjacency matrix.
    ///
    /// # Arguments
    /// * `start` - Starting vertex
    /// * `end` - Ending vertex
    /// * `k` - Number of edges in the path
    ///
    /// # Returns
    /// Number of paths of length k
    pub fn count_paths(&self, start: usize, end: usize, k: usize) -> Integer {
        if start >= self.graph.num_vertices() || end >= self.graph.num_vertices() {
            return Integer::zero();
        }

        if k == 0 {
            return if start == end {
                Integer::one()
            } else {
                Integer::zero()
            };
        }

        let adj_matrix = self.adjacency_matrix();

        // Matrix::pow takes u32 and returns Result
        match adj_matrix.pow(k as u32) {
            Ok(power) => match power.get(start, end) {
                Ok(value) => value.clone(),
                Err(_) => Integer::zero(),
            },
            Err(_) => Integer::zero(),
        }
    }

    /// Get the adjacency matrix of the graph
    fn adjacency_matrix(&self) -> Matrix<Integer> {
        let n = self.graph.num_vertices();
        let mut matrix = Matrix::zeros(n, n);

        for (u, v) in self.graph.edges() {
            let _ = matrix.set(u, v, Integer::one());
        }

        matrix
    }

    /// Compute the generating function for paths from start to end
    ///
    /// Returns a power series where the coefficient of x^k is the number of
    /// paths of length k from start to end.
    ///
    /// # Arguments
    /// * `start` - Starting vertex
    /// * `end` - Ending vertex
    /// * `precision` - Number of terms to compute in the power series
    ///
    /// # Returns
    /// Power series with path counts as coefficients
    pub fn path_generating_function(
        &self,
        start: usize,
        end: usize,
        precision: usize,
    ) -> PowerSeries<Integer> {
        let mut coeffs = Vec::with_capacity(precision);

        for k in 0..precision {
            coeffs.push(self.count_paths(start, end, k));
        }

        PowerSeries::new(coeffs, precision)
    }

    /// Compute the transfer matrix for the graph
    ///
    /// The transfer matrix can be used to analyze path properties and
    /// compute generating functions efficiently.
    ///
    /// # Returns
    /// The adjacency matrix as a transfer matrix
    pub fn transfer_matrix(&self) -> Matrix<Integer> {
        self.adjacency_matrix()
    }

    /// Compute the total number of paths of length k in the graph
    ///
    /// Sums over all possible start and end vertices.
    ///
    /// # Arguments
    /// * `k` - Path length (number of edges)
    ///
    /// # Returns
    /// Total number of paths of length k
    pub fn total_paths_of_length(&self, k: usize) -> Integer {
        if k == 0 {
            return Integer::from(self.graph.num_vertices() as u64);
        }

        let adj_matrix = self.adjacency_matrix();

        // Matrix::pow takes u32 and returns Result
        match adj_matrix.pow(k as u32) {
            Ok(power) => {
                let mut total = Integer::zero();
                for i in 0..self.graph.num_vertices() {
                    for j in 0..self.graph.num_vertices() {
                        if let Ok(value) = power.get(i, j) {
                            total = total + value.clone();
                        }
                    }
                }
                total
            }
            Err(_) => Integer::zero(),
        }
    }

    /// Compute the generating function for total paths in the graph
    ///
    /// Returns a power series where the coefficient of x^k is the total
    /// number of paths of length k in the graph.
    ///
    /// # Arguments
    /// * `precision` - Number of terms to compute
    ///
    /// # Returns
    /// Power series with total path counts
    pub fn total_paths_generating_function(&self, precision: usize) -> PowerSeries<Integer> {
        let mut coeffs = Vec::with_capacity(precision);

        for k in 0..precision {
            coeffs.push(self.total_paths_of_length(k));
        }

        PowerSeries::new(coeffs, precision)
    }

    /// Check if there exists a path from start to end
    ///
    /// # Arguments
    /// * `start` - Starting vertex
    /// * `end` - Ending vertex
    ///
    /// # Returns
    /// true if a path exists, false otherwise
    pub fn has_path(&self, start: usize, end: usize) -> bool {
        self.graph.shortest_path(start, end).unwrap_or(None).is_some()
    }

    /// Get the length of the shortest path from start to end
    ///
    /// # Arguments
    /// * `start` - Starting vertex
    /// * `end` - Ending vertex
    ///
    /// # Returns
    /// Some(length) if a path exists, None otherwise
    pub fn shortest_path_length(&self, start: usize, end: usize) -> Option<usize> {
        match self.graph.shortest_path(start, end) {
            Ok(Some(path)) => Some(path.len() - 1), // Convert vertex count to edge count
            _ => None,
        }
    }

    /// Compute the path polynomial for paths from start to end
    ///
    /// This is similar to the generating function but returns it as a polynomial
    /// representation showing the count of paths for each length.
    ///
    /// # Arguments
    /// * `start` - Starting vertex
    /// * `end` - Ending vertex
    /// * `max_length` - Maximum path length to consider
    ///
    /// # Returns
    /// Vector where index k contains the number of paths of length k
    pub fn path_counts_by_length(&self, start: usize, end: usize, max_length: usize) -> Vec<Integer> {
        let mut counts = Vec::with_capacity(max_length + 1);

        for k in 0..=max_length {
            counts.push(self.count_paths(start, end, k));
        }

        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_path_enumeration() {
        // Create a simple directed graph: 0 -> 1 -> 2
        //                                  \-> 2
        let mut g = DiGraph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(0, 2).unwrap();

        let gp = GraphPath::new(&g);
        let paths = gp.all_simple_paths(0, 2, 3);

        // Should find two paths: [0, 2] and [0, 1, 2]
        assert_eq!(paths.len(), 2);
        assert!(paths.contains(&vec![0, 2]));
        assert!(paths.contains(&vec![0, 1, 2]));
    }

    #[test]
    fn test_paths_of_exact_length() {
        // Graph: 0 -> 1 -> 2
        //        ^---------|
        let mut g = DiGraph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 0).unwrap();

        let gp = GraphPath::new(&g);

        // Paths of length 2 from 0 to 2
        let paths = gp.all_paths_of_length(0, 2, 2);
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], vec![0, 1, 2]);

        // Paths of length 3 from 0 to 0 (cycle)
        let paths = gp.all_paths_of_length(0, 0, 3);
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], vec![0, 1, 2, 0]);
    }

    #[test]
    fn test_path_counting() {
        // Create a simple directed graph: 0 -> 1 -> 2
        let mut g = DiGraph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        let gp = GraphPath::new(&g);

        // No paths of length 0 from 0 to 2
        assert_eq!(gp.count_paths(0, 2, 0), Integer::zero());

        // No paths of length 1 from 0 to 2
        assert_eq!(gp.count_paths(0, 2, 1), Integer::zero());

        // One path of length 2 from 0 to 2
        assert_eq!(gp.count_paths(0, 2, 2), Integer::one());
    }

    #[test]
    fn test_path_counting_with_multiple_paths() {
        // Create graph with multiple paths: 0 -> 1 -> 3
        //                                    0 -> 2 -> 3
        let mut g = DiGraph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 2).unwrap();
        g.add_edge(1, 3).unwrap();
        g.add_edge(2, 3).unwrap();

        let gp = GraphPath::new(&g);

        // Two paths of length 2 from 0 to 3
        assert_eq!(gp.count_paths(0, 3, 2), Integer::from(2));
    }

    #[test]
    fn test_path_counting_with_cycles() {
        // Graph with self-loop: 0 -> 1
        //                       1 -> 1 (self-loop)
        //                       1 -> 2
        let mut g = DiGraph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        let gp = GraphPath::new(&g);

        // Paths from 0 to 2:
        // Length 2: 0 -> 1 -> 2 (1 path)
        assert_eq!(gp.count_paths(0, 2, 2), Integer::one());

        // Length 3: 0 -> 1 -> 1 -> 2 (1 path)
        assert_eq!(gp.count_paths(0, 2, 3), Integer::one());

        // Length 4: 0 -> 1 -> 1 -> 1 -> 2 (1 path)
        assert_eq!(gp.count_paths(0, 2, 4), Integer::one());
    }

    #[test]
    fn test_generating_function() {
        // Simple path graph: 0 -> 1 -> 2
        let mut g = DiGraph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        let gp = GraphPath::new(&g);
        let gf = gp.path_generating_function(0, 2, 5);

        // Check coefficients
        assert_eq!(gf.coeff(0), &Integer::zero()); // No path of length 0
        assert_eq!(gf.coeff(1), &Integer::zero()); // No path of length 1
        assert_eq!(gf.coeff(2), &Integer::one());  // One path of length 2
        assert_eq!(gf.coeff(3), &Integer::zero()); // No path of length 3
    }

    #[test]
    fn test_total_paths() {
        // Triangle: 0 -> 1, 1 -> 2, 2 -> 0
        let mut g = DiGraph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 0).unwrap();

        let gp = GraphPath::new(&g);

        // Length 0: 3 paths (each vertex to itself)
        assert_eq!(gp.total_paths_of_length(0), Integer::from(3));

        // Length 1: 3 paths (the 3 edges)
        assert_eq!(gp.total_paths_of_length(1), Integer::from(3));

        // Length 2: 3 paths
        assert_eq!(gp.total_paths_of_length(2), Integer::from(3));

        // Length 3: 3 paths (each vertex back to itself)
        assert_eq!(gp.total_paths_of_length(3), Integer::from(3));
    }

    #[test]
    fn test_has_path() {
        let mut g = DiGraph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        let gp = GraphPath::new(&g);

        assert!(gp.has_path(0, 1));
        assert!(gp.has_path(0, 2));
        assert!(!gp.has_path(2, 0));
        assert!(!gp.has_path(1, 0));
    }

    #[test]
    fn test_shortest_path_length() {
        let mut g = DiGraph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(0, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        let gp = GraphPath::new(&g);

        assert_eq!(gp.shortest_path_length(0, 1), Some(1));
        assert_eq!(gp.shortest_path_length(0, 2), Some(1)); // Direct edge
        assert_eq!(gp.shortest_path_length(0, 3), Some(2)); // 0 -> 2 -> 3
        assert_eq!(gp.shortest_path_length(3, 0), None);    // No path
    }

    #[test]
    fn test_path_counts_by_length() {
        // Simple chain: 0 -> 1 -> 2
        let mut g = DiGraph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        let gp = GraphPath::new(&g);
        let counts = gp.path_counts_by_length(0, 2, 5);

        assert_eq!(counts[0], Integer::zero());
        assert_eq!(counts[1], Integer::zero());
        assert_eq!(counts[2], Integer::one());
        assert_eq!(counts[3], Integer::zero());
        assert_eq!(counts[4], Integer::zero());
        assert_eq!(counts[5], Integer::zero());
    }

    #[test]
    fn test_adjacency_matrix() {
        let mut g = DiGraph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(0, 2).unwrap();

        let gp = GraphPath::new(&g);
        let adj = gp.adjacency_matrix();

        assert_eq!(adj.get(0, 1).unwrap(), &Integer::one());
        assert_eq!(adj.get(1, 2).unwrap(), &Integer::one());
        assert_eq!(adj.get(0, 2).unwrap(), &Integer::one());
        assert_eq!(adj.get(1, 0).unwrap(), &Integer::zero());
        assert_eq!(adj.get(2, 1).unwrap(), &Integer::zero());
    }

    #[test]
    fn test_empty_graph() {
        let g = DiGraph::new(3);
        let gp = GraphPath::new(&g);

        // No paths of length > 0 in empty graph
        assert_eq!(gp.count_paths(0, 1, 1), Integer::zero());
        assert_eq!(gp.total_paths_of_length(1), Integer::zero());

        // Each vertex has a trivial path to itself
        assert_eq!(gp.count_paths(0, 0, 0), Integer::one());
        assert_eq!(gp.total_paths_of_length(0), Integer::from(3));
    }

    #[test]
    fn test_complete_directed_graph() {
        // Complete directed graph on 3 vertices
        let mut g = DiGraph::new(3);
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    g.add_edge(i, j).unwrap();
                }
            }
        }

        let gp = GraphPath::new(&g);

        // From vertex 0 to any other vertex, there's 1 path of length 1
        assert_eq!(gp.count_paths(0, 1, 1), Integer::one());
        assert_eq!(gp.count_paths(0, 2, 1), Integer::one());

        // From vertex 0 to vertex 1 with length 2:
        // 0 -> 2 -> 1
        assert_eq!(gp.count_paths(0, 1, 2), Integer::one());

        // Total paths of length 1: 6 (each of 3 vertices to each of 2 others)
        assert_eq!(gp.total_paths_of_length(1), Integer::from(6));
    }
}
