//! Path semigroup - the semigroup of all paths in a quiver

use crate::{Quiver, QuiverPath};
use std::collections::HashMap;

/// The path semigroup of a quiver
///
/// This structure manages all paths in a quiver and provides
/// efficient operations for path enumeration and composition.
pub struct PathSemigroup {
    quiver: Quiver,
    /// Cache of paths by length and start vertex
    paths_cache: HashMap<(usize, usize), Vec<QuiverPath>>,
}

impl PathSemigroup {
    /// Create a new path semigroup for a quiver
    pub fn new(quiver: Quiver) -> Self {
        PathSemigroup {
            quiver,
            paths_cache: HashMap::new(),
        }
    }

    /// Get a reference to the underlying quiver
    pub fn quiver(&self) -> &Quiver {
        &self.quiver
    }

    /// Get all trivial paths (idempotents)
    ///
    /// One for each vertex in the quiver
    pub fn idempotents(&self) -> Vec<QuiverPath> {
        (0..self.quiver.num_vertices())
            .map(|v| QuiverPath::trivial(v))
            .collect()
    }

    /// Get all paths of length 1 (arrows/edges)
    pub fn arrows(&self) -> Vec<QuiverPath> {
        self.quiver
            .edges()
            .iter()
            .enumerate()
            .map(|(idx, edge)| {
                QuiverPath::new(&self.quiver, edge.source, edge.target, vec![idx])
                    .expect("Edge should form valid path")
            })
            .collect()
    }

    /// Generate all paths of a given length starting from a vertex
    ///
    /// This is done recursively:
    /// - Length 0: trivial path at the vertex
    /// - Length n: extend all paths of length n-1 by one edge
    pub fn paths_by_length_and_start(&mut self, length: usize, start: usize) -> Vec<QuiverPath> {
        if start >= self.quiver.num_vertices() {
            return Vec::new();
        }

        // Check cache
        if let Some(cached) = self.paths_cache.get(&(length, start)) {
            return cached.clone();
        }

        let paths = if length == 0 {
            vec![QuiverPath::trivial(start)]
        } else {
            // Get paths of length n-1
            let prev_paths = self.paths_by_length_and_start(length - 1, start);
            let mut new_paths = Vec::new();

            for path in prev_paths {
                // Extend by each outgoing edge from the end vertex
                if let Some(outgoing) = self.quiver.outgoing_edges(path.end) {
                    for &edge_idx in outgoing {
                        let edge = &self.quiver.edges()[edge_idx];
                        let mut new_edge_seq = path.edges.clone();
                        new_edge_seq.push(edge_idx);

                        if let Ok(new_path) =
                            QuiverPath::new(&self.quiver, start, edge.target, new_edge_seq)
                        {
                            new_paths.push(new_path);
                        }
                    }
                }
            }

            new_paths
        };

        // Cache the result
        self.paths_cache.insert((length, start), paths.clone());
        paths
    }

    /// Generate all paths of a given length
    pub fn paths_by_length(&mut self, length: usize) -> Vec<QuiverPath> {
        let mut all_paths = Vec::new();

        for start in 0..self.quiver.num_vertices() {
            all_paths.extend(self.paths_by_length_and_start(length, start));
        }

        all_paths
    }

    /// Generate all paths up to a given length
    pub fn paths_up_to_length(&mut self, max_length: usize) -> Vec<QuiverPath> {
        let mut all_paths = Vec::new();

        for length in 0..=max_length {
            all_paths.extend(self.paths_by_length(length));
        }

        all_paths
    }

    /// Find all paths from start to end vertex
    ///
    /// For acyclic quivers, this enumerates all paths.
    /// For cyclic quivers, this returns None.
    pub fn all_paths(&self, start: usize, end: usize) -> Option<Vec<QuiverPath>> {
        if !self.quiver.is_acyclic() {
            return None; // Cannot enumerate all paths in cyclic quiver
        }

        let edge_paths = self.quiver.all_paths(start, end)?;

        Some(
            edge_paths
                .into_iter()
                .map(|edges| {
                    QuiverPath::new(&self.quiver, start, end, edges)
                        .expect("Valid edge sequence from quiver")
                })
                .collect(),
        )
    }

    /// Count the number of paths of a given length starting from a vertex
    ///
    /// This is more efficient than generating all paths if you only need the count
    pub fn count_paths(&mut self, length: usize, start: usize) -> usize {
        self.paths_by_length_and_start(length, start).len()
    }

    /// Generate the basis of the path algebra of a given degree
    ///
    /// The basis consists of all paths of the specified length
    pub fn algebra_basis(&mut self, degree: usize) -> Vec<QuiverPath> {
        self.paths_by_length(degree)
    }

    /// Clear the path cache
    ///
    /// Useful if the underlying quiver is modified
    pub fn clear_cache(&mut self) {
        self.paths_cache.clear();
    }

    /// Get statistics about the path semigroup
    pub fn statistics(&mut self, max_length: usize) -> PathStatistics {
        let mut total_paths = 0;
        let mut paths_by_length = Vec::new();

        for length in 0..=max_length {
            let count = self.paths_by_length(length).len();
            paths_by_length.push(count);
            total_paths += count;
        }

        PathStatistics {
            num_vertices: self.quiver.num_vertices(),
            num_edges: self.quiver.num_edges(),
            is_acyclic: self.quiver.is_acyclic(),
            max_length,
            total_paths,
            paths_by_length,
        }
    }
}

/// Statistics about a path semigroup
#[derive(Debug, Clone)]
pub struct PathStatistics {
    /// Number of vertices in the quiver
    pub num_vertices: usize,
    /// Number of edges in the quiver
    pub num_edges: usize,
    /// Whether the quiver is acyclic
    pub is_acyclic: bool,
    /// Maximum path length computed
    pub max_length: usize,
    /// Total number of paths up to max_length
    pub total_paths: usize,
    /// Number of paths at each length
    pub paths_by_length: Vec<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_linear_quiver() -> Quiver {
        // Linear quiver: 0 -a-> 1 -b-> 2 -c-> 3
        let mut q = Quiver::new(4);
        q.add_edge(0, 1, "a").unwrap();
        q.add_edge(1, 2, "b").unwrap();
        q.add_edge(2, 3, "c").unwrap();
        q
    }

    fn setup_diamond_quiver() -> Quiver {
        // Diamond: 0 -a-> 1 -c-> 3
        //          0 -b-> 2 -d-> 3
        let mut q = Quiver::new(4);
        q.add_edge(0, 1, "a").unwrap();
        q.add_edge(0, 2, "b").unwrap();
        q.add_edge(1, 3, "c").unwrap();
        q.add_edge(2, 3, "d").unwrap();
        q
    }

    #[test]
    fn test_idempotents() {
        let q = setup_linear_quiver();
        let ps = PathSemigroup::new(q);

        let idem = ps.idempotents();
        assert_eq!(idem.len(), 4);

        for (i, path) in idem.iter().enumerate() {
            assert_eq!(path.start, i);
            assert_eq!(path.end, i);
            assert!(path.is_trivial());
        }
    }

    #[test]
    fn test_arrows() {
        let q = setup_linear_quiver();
        let ps = PathSemigroup::new(q);

        let arrows = ps.arrows();
        assert_eq!(arrows.len(), 3); // a, b, c

        for arrow in &arrows {
            assert_eq!(arrow.length(), 1);
        }
    }

    #[test]
    fn test_paths_by_length() {
        let q = setup_linear_quiver();
        let mut ps = PathSemigroup::new(q);

        // Length 0: 4 trivial paths (one per vertex)
        let paths0 = ps.paths_by_length(0);
        assert_eq!(paths0.len(), 4);

        // Length 1: 3 arrows
        let paths1 = ps.paths_by_length(1);
        assert_eq!(paths1.len(), 3);

        // Length 2: 2 paths (a*b, b*c)
        let paths2 = ps.paths_by_length(2);
        assert_eq!(paths2.len(), 2);

        // Length 3: 1 path (a*b*c)
        let paths3 = ps.paths_by_length(3);
        assert_eq!(paths3.len(), 1);

        // Length 4: 0 paths (graph is too short)
        let paths4 = ps.paths_by_length(4);
        assert_eq!(paths4.len(), 0);
    }

    #[test]
    fn test_paths_by_length_and_start() {
        let q = setup_linear_quiver();
        let mut ps = PathSemigroup::new(q);

        // Paths of length 2 starting from vertex 0
        let paths = ps.paths_by_length_and_start(2, 0);
        assert_eq!(paths.len(), 1); // Only a*b

        let path = &paths[0];
        assert_eq!(path.start, 0);
        assert_eq!(path.end, 2);
        assert_eq!(path.length(), 2);
    }

    #[test]
    fn test_all_paths_acyclic() {
        let q = setup_diamond_quiver();
        let ps = PathSemigroup::new(q);

        // All paths from 0 to 3
        let paths = ps.all_paths(0, 3);
        assert!(paths.is_some());
        let p = paths.unwrap();
        assert_eq!(p.len(), 2); // a*c and b*d
    }

    #[test]
    fn test_all_paths_cyclic() {
        // Cyclic quiver: 0 -> 1 -> 2 -> 0
        let mut q = Quiver::new(3);
        q.add_edge(0, 1, "a").unwrap();
        q.add_edge(1, 2, "b").unwrap();
        q.add_edge(2, 0, "c").unwrap();

        let ps = PathSemigroup::new(q);
        let paths = ps.all_paths(0, 2);
        assert!(paths.is_none()); // Cyclic quiver
    }

    #[test]
    fn test_count_paths() {
        let q = setup_linear_quiver();
        let mut ps = PathSemigroup::new(q);

        assert_eq!(ps.count_paths(0, 0), 1); // Trivial path at 0
        assert_eq!(ps.count_paths(1, 0), 1); // a
        assert_eq!(ps.count_paths(2, 0), 1); // a*b
        assert_eq!(ps.count_paths(3, 0), 1); // a*b*c
        assert_eq!(ps.count_paths(4, 0), 0); // No path of length 4
    }

    #[test]
    fn test_statistics() {
        let q = setup_diamond_quiver();
        let mut ps = PathSemigroup::new(q);

        let stats = ps.statistics(3);
        assert_eq!(stats.num_vertices, 4);
        assert_eq!(stats.num_edges, 4);
        assert!(stats.is_acyclic);
        assert_eq!(stats.max_length, 3);

        // Length 0: 4 trivial paths
        // Length 1: 4 arrows
        // Length 2: 2 paths (a*c, b*d)
        // Length 3: 0 paths
        assert_eq!(stats.paths_by_length, vec![4, 4, 2, 0]);
        assert_eq!(stats.total_paths, 10);
    }

    #[test]
    fn test_cache() {
        let q = setup_linear_quiver();
        let mut ps = PathSemigroup::new(q);

        // First call - computes
        let paths1 = ps.paths_by_length_and_start(2, 0);
        assert_eq!(paths1.len(), 1);

        // Second call - should use cache
        let paths2 = ps.paths_by_length_and_start(2, 0);
        assert_eq!(paths2.len(), 1);

        // Clear cache
        ps.clear_cache();

        // After clearing, should recompute
        let paths3 = ps.paths_by_length_and_start(2, 0);
        assert_eq!(paths3.len(), 1);
    }

    #[test]
    fn test_algebra_basis() {
        let q = setup_linear_quiver();
        let mut ps = PathSemigroup::new(q);

        // Basis of degree 1 (arrows)
        let basis1 = ps.algebra_basis(1);
        assert_eq!(basis1.len(), 3);

        // Basis of degree 2
        let basis2 = ps.algebra_basis(2);
        assert_eq!(basis2.len(), 2);
    }

    #[test]
    fn test_paths_up_to_length() {
        let q = setup_linear_quiver();
        let mut ps = PathSemigroup::new(q);

        // All paths up to length 2
        let paths = ps.paths_up_to_length(2);
        // 4 (length 0) + 3 (length 1) + 2 (length 2) = 9
        assert_eq!(paths.len(), 9);
    }
}
