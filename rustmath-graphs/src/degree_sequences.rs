//! Degree sequences and graph realization
//!
//! This module provides functionality for working with degree sequences,
//! including verification using the Erdős-Gallai theorem and graph
//! realization using the Havel-Hakimi algorithm.

use crate::graph::Graph;

/// Represents a degree sequence for a graph
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DegreeSequence {
    degrees: Vec<usize>,
}

impl DegreeSequence {
    /// Create a new degree sequence from a vector of degrees
    ///
    /// # Arguments
    /// * `degrees` - A vector of non-negative integers representing vertex degrees
    ///
    /// # Examples
    /// ```
    /// use rustmath_graphs::degree_sequences::DegreeSequence;
    ///
    /// let seq = DegreeSequence::new(vec![3, 3, 2, 2, 2]);
    /// ```
    pub fn new(mut degrees: Vec<usize>) -> Self {
        // Sort in non-increasing order
        degrees.sort_by(|a, b| b.cmp(a));
        DegreeSequence { degrees }
    }

    /// Create a degree sequence from a graph
    ///
    /// # Arguments
    /// * `graph` - The graph to extract the degree sequence from
    ///
    /// # Examples
    /// ```
    /// use rustmath_graphs::{Graph, degree_sequences::DegreeSequence};
    ///
    /// let mut g = Graph::new(3);
    /// g.add_edge(0, 1).unwrap();
    /// g.add_edge(1, 2).unwrap();
    /// let seq = DegreeSequence::from_graph(&g);
    /// ```
    pub fn from_graph(graph: &Graph) -> Self {
        let degrees: Vec<usize> = (0..graph.num_vertices())
            .map(|v| graph.degree(v).unwrap_or(0))
            .collect();
        DegreeSequence::new(degrees)
    }

    /// Get the degree sequence as a slice
    pub fn as_slice(&self) -> &[usize] {
        &self.degrees
    }

    /// Get the number of vertices
    pub fn len(&self) -> usize {
        self.degrees.len()
    }

    /// Check if the sequence is empty
    pub fn is_empty(&self) -> bool {
        self.degrees.is_empty()
    }

    /// Verify if the degree sequence is graphical using the Erdős-Gallai theorem
    ///
    /// A sequence d₁ ≥ d₂ ≥ ... ≥ dₙ is graphical if and only if:
    /// 1. The sum of all degrees is even
    /// 2. For each k = 1, 2, ..., n:
    ///    Σᵢ₌₁ᵏ dᵢ ≤ k(k-1) + Σᵢ₌ₖ₊₁ⁿ min(dᵢ, k)
    ///
    /// # Returns
    /// `true` if the sequence can be realized as a simple graph, `false` otherwise
    ///
    /// # Examples
    /// ```
    /// use rustmath_graphs::degree_sequences::DegreeSequence;
    ///
    /// let seq = DegreeSequence::new(vec![3, 3, 2, 2, 2]);
    /// assert!(seq.is_graphical());
    ///
    /// let invalid = DegreeSequence::new(vec![3, 3, 3]);
    /// assert!(!invalid.is_graphical());
    /// ```
    pub fn is_graphical(&self) -> bool {
        if self.degrees.is_empty() {
            return true;
        }

        // Check if sum is even
        let sum: usize = self.degrees.iter().sum();
        if sum % 2 != 0 {
            return false;
        }

        // Check if any degree is >= n (would require self-loops or multiple edges)
        let n = self.degrees.len();
        if self.degrees.iter().any(|&d| d >= n) {
            return false;
        }

        // Apply Erdős-Gallai theorem
        let mut left_sum = 0;
        for k in 1..=n {
            left_sum += self.degrees[k - 1];

            let k_times_k_minus_1 = k * (k - 1);

            // Calculate right side: Σᵢ₌ₖ₊₁ⁿ min(dᵢ, k)
            let right_sum: usize = self.degrees
                .iter()
                .skip(k)
                .map(|&d| d.min(k))
                .sum();

            if left_sum > k_times_k_minus_1 + right_sum {
                return false;
            }
        }

        true
    }

    /// Realize the degree sequence as a graph using the Havel-Hakimi algorithm
    ///
    /// This constructs a simple graph with the given degree sequence if possible.
    ///
    /// # Returns
    /// * `Ok(Graph)` - A graph with the specified degree sequence
    /// * `Err(String)` - If the sequence is not graphical
    ///
    /// # Examples
    /// ```
    /// use rustmath_graphs::degree_sequences::DegreeSequence;
    ///
    /// let seq = DegreeSequence::new(vec![3, 3, 2, 2, 2]);
    /// let graph = seq.realize().unwrap();
    /// assert_eq!(graph.num_vertices(), 5);
    /// ```
    pub fn realize(&self) -> Result<Graph, String> {
        if !self.is_graphical() {
            return Err("Degree sequence is not graphical".to_string());
        }

        let n = self.degrees.len();
        if n == 0 {
            return Ok(Graph::new(0));
        }

        // Use Havel-Hakimi algorithm
        // Keep track of remaining degrees and vertex indices
        let mut remaining: Vec<(usize, usize)> = self.degrees
            .iter()
            .enumerate()
            .map(|(i, &d)| (d, i))
            .collect();

        let mut graph = Graph::new(n);

        while !remaining.is_empty() {
            // Sort by degree (descending)
            remaining.sort_by(|a, b| b.0.cmp(&a.0));

            // Remove vertices with degree 0
            remaining.retain(|&(d, _)| d > 0);

            if remaining.is_empty() {
                break;
            }

            // Take the vertex with highest degree
            let (degree, vertex) = remaining[0];

            if degree > remaining.len() - 1 {
                return Err("Invalid degree sequence during realization".to_string());
            }

            // Connect to the next 'degree' vertices
            for i in 1..=degree {
                let neighbor = remaining[i].1;
                graph.add_edge(vertex, neighbor)?;
                remaining[i].0 -= 1;
            }

            // Remove the processed vertex
            remaining.remove(0);
        }

        Ok(graph)
    }

    /// Check if this degree sequence is the same as another (ignoring order)
    ///
    /// # Arguments
    /// * `other` - Another degree sequence to compare with
    ///
    /// # Examples
    /// ```
    /// use rustmath_graphs::degree_sequences::DegreeSequence;
    ///
    /// let seq1 = DegreeSequence::new(vec![3, 2, 2]);
    /// let seq2 = DegreeSequence::new(vec![2, 3, 2]);
    /// assert!(seq1.same_as(&seq2));
    /// ```
    pub fn same_as(&self, other: &DegreeSequence) -> bool {
        self.degrees == other.degrees
    }

    /// Get the maximum degree in the sequence
    pub fn max_degree(&self) -> Option<usize> {
        self.degrees.first().copied()
    }

    /// Get the minimum degree in the sequence
    pub fn min_degree(&self) -> Option<usize> {
        self.degrees.last().copied()
    }

    /// Check if the sequence is regular (all degrees are equal)
    pub fn is_regular(&self) -> bool {
        if self.degrees.is_empty() {
            return true;
        }
        let first = self.degrees[0];
        self.degrees.iter().all(|&d| d == first)
    }
}

impl From<Vec<usize>> for DegreeSequence {
    fn from(degrees: Vec<usize>) -> Self {
        DegreeSequence::new(degrees)
    }
}

impl std::fmt::Display for DegreeSequence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, &degree) in self.degrees.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", degree)?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_degree_sequence_new() {
        let seq = DegreeSequence::new(vec![2, 3, 1, 3]);
        assert_eq!(seq.as_slice(), &[3, 3, 2, 1]);
    }

    #[test]
    fn test_from_graph() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 2).unwrap();
        g.add_edge(1, 2).unwrap();

        let seq = DegreeSequence::from_graph(&g);
        assert_eq!(seq.as_slice(), &[2, 2, 2, 0]);
    }

    #[test]
    fn test_is_graphical_valid() {
        // Triangle: each vertex has degree 2
        let seq = DegreeSequence::new(vec![2, 2, 2]);
        assert!(seq.is_graphical());

        // Complete graph K4
        let seq = DegreeSequence::new(vec![3, 3, 3, 3]);
        assert!(seq.is_graphical());

        // Path graph P5
        let seq = DegreeSequence::new(vec![2, 2, 2, 1, 1]);
        assert!(seq.is_graphical());
    }

    #[test]
    fn test_is_graphical_invalid() {
        // Odd sum
        let seq = DegreeSequence::new(vec![3, 3, 3]);
        assert!(!seq.is_graphical());

        // Too high degree
        let seq = DegreeSequence::new(vec![4, 3, 2, 1]);
        assert!(!seq.is_graphical());

        // Violates Erdős-Gallai condition
        let seq = DegreeSequence::new(vec![3, 2, 1]);
        assert!(!seq.is_graphical());
    }

    #[test]
    fn test_is_graphical_edge_cases() {
        // Empty sequence
        let seq = DegreeSequence::new(vec![]);
        assert!(seq.is_graphical());

        // All zeros
        let seq = DegreeSequence::new(vec![0, 0, 0]);
        assert!(seq.is_graphical());

        // Single vertex
        let seq = DegreeSequence::new(vec![0]);
        assert!(seq.is_graphical());
    }

    #[test]
    fn test_realize_triangle() {
        let seq = DegreeSequence::new(vec![2, 2, 2]);
        let graph = seq.realize().unwrap();

        assert_eq!(graph.num_vertices(), 3);
        assert_eq!(graph.num_edges(), 3);

        // Verify all vertices have degree 2
        for v in 0..3 {
            assert_eq!(graph.degree(v), Some(2));
        }
    }

    #[test]
    fn test_realize_k4() {
        let seq = DegreeSequence::new(vec![3, 3, 3, 3]);
        let graph = seq.realize().unwrap();

        assert_eq!(graph.num_vertices(), 4);
        assert_eq!(graph.num_edges(), 6);

        // Verify all vertices have degree 3
        for v in 0..4 {
            assert_eq!(graph.degree(v), Some(3));
        }
    }

    #[test]
    fn test_realize_path() {
        let seq = DegreeSequence::new(vec![2, 2, 2, 1, 1]);
        let graph = seq.realize().unwrap();

        assert_eq!(graph.num_vertices(), 5);
        assert_eq!(graph.num_edges(), 4);
    }

    #[test]
    fn test_realize_invalid() {
        let seq = DegreeSequence::new(vec![3, 3, 3]);
        assert!(seq.realize().is_err());
    }

    #[test]
    fn test_realize_preserves_degree_sequence() {
        let original = DegreeSequence::new(vec![3, 3, 2, 2, 2]);
        let graph = original.realize().unwrap();
        let realized = DegreeSequence::from_graph(&graph);

        assert!(original.same_as(&realized));
    }

    #[test]
    fn test_same_as() {
        let seq1 = DegreeSequence::new(vec![3, 2, 2, 1]);
        let seq2 = DegreeSequence::new(vec![1, 2, 3, 2]);
        assert!(seq1.same_as(&seq2));

        let seq3 = DegreeSequence::new(vec![3, 3, 2, 1]);
        assert!(!seq1.same_as(&seq3));
    }

    #[test]
    fn test_max_min_degree() {
        let seq = DegreeSequence::new(vec![5, 3, 3, 2, 1]);
        assert_eq!(seq.max_degree(), Some(5));
        assert_eq!(seq.min_degree(), Some(1));

        let empty = DegreeSequence::new(vec![]);
        assert_eq!(empty.max_degree(), None);
        assert_eq!(empty.min_degree(), None);
    }

    #[test]
    fn test_is_regular() {
        let regular = DegreeSequence::new(vec![3, 3, 3, 3]);
        assert!(regular.is_regular());

        let not_regular = DegreeSequence::new(vec![3, 2, 2, 1]);
        assert!(!not_regular.is_regular());
    }

    #[test]
    fn test_display() {
        let seq = DegreeSequence::new(vec![3, 2, 1]);
        assert_eq!(format!("{}", seq), "[3, 2, 1]");
    }

    #[test]
    fn test_petersen_graph() {
        // Petersen graph: 10 vertices, each with degree 3
        let seq = DegreeSequence::new(vec![3; 10]);
        assert!(seq.is_graphical());

        let graph = seq.realize().unwrap();
        assert_eq!(graph.num_vertices(), 10);
        assert_eq!(graph.num_edges(), 15);
    }

    #[test]
    fn test_star_graph() {
        // Star graph with 5 vertices: one center with degree 4, four leaves with degree 1
        let seq = DegreeSequence::new(vec![4, 1, 1, 1, 1]);
        assert!(seq.is_graphical());

        let graph = seq.realize().unwrap();
        assert_eq!(graph.num_vertices(), 5);
        assert_eq!(graph.num_edges(), 4);
    }

    #[test]
    fn test_bipartite_complete_k33() {
        // Complete bipartite graph K₃,₃: 6 vertices, each with degree 3
        let seq = DegreeSequence::new(vec![3, 3, 3, 3, 3, 3]);
        assert!(seq.is_graphical());

        let graph = seq.realize().unwrap();
        assert_eq!(graph.num_vertices(), 6);
        assert_eq!(graph.num_edges(), 9);
    }
}
