//! Tree generation and iteration
//!
//! This module provides functionality for generating and iterating over
//! all labeled trees of a given size. Trees are fundamental structures
//! in graph theory.

use crate::graph::Graph;

/// Iterator over all labeled trees with n vertices
///
/// `TreeIterator` generates all non-isomorphic labeled trees with a specified
/// number of vertices. It uses Prüfer sequences to enumerate trees efficiently.
///
/// A tree with n vertices has n-2 elements in its Prüfer sequence, and there
/// are n^(n-2) distinct labeled trees on n vertices (Cayley's formula).
///
/// # Examples
///
/// ```
/// use rustmath_graphs::trees::TreeIterator;
///
/// // Iterate over all trees with 4 vertices
/// let iter = TreeIterator::new(4);
/// let trees: Vec<_> = iter.collect();
///
/// // There are 4^(4-2) = 16 labeled trees with 4 vertices
/// assert_eq!(trees.len(), 16);
/// ```
pub struct TreeIterator {
    n: usize,
    current: Vec<usize>,
    done: bool,
}

impl TreeIterator {
    /// Create a new tree iterator for trees with n vertices
    ///
    /// # Arguments
    ///
    /// * `n` - Number of vertices in the trees to generate
    ///
    /// # Returns
    ///
    /// A new `TreeIterator` instance
    ///
    /// # Panics
    ///
    /// Panics if n < 2, as trees must have at least 2 vertices
    pub fn new(n: usize) -> Self {
        assert!(n >= 2, "Trees must have at least 2 vertices");

        TreeIterator {
            n,
            current: vec![0; n.saturating_sub(2)],
            done: false,
        }
    }

    /// Get the number of vertices in the trees being generated
    pub fn num_vertices(&self) -> usize {
        self.n
    }

    /// Convert a Prüfer sequence to a tree (graph)
    ///
    /// The Prüfer sequence uniquely identifies a labeled tree.
    /// This method reconstructs the tree from the sequence.
    fn prufer_to_tree(sequence: &[usize], n: usize) -> Graph {
        let mut graph = Graph::new(n);

        if n < 2 {
            return graph;
        }

        if n == 2 {
            graph.add_edge(0, 1).ok();
            return graph;
        }

        // Count degree of each vertex (starts at 1, add 1 for each appearance in sequence)
        let mut degree = vec![1; n];
        for &v in sequence {
            degree[v] += 1;
        }

        // Build tree by repeatedly connecting lowest degree-1 vertex to next sequence element
        for &v in sequence {
            // Find the smallest leaf (degree 1 vertex)
            for i in 0..n {
                if degree[i] == 1 {
                    graph.add_edge(i, v).ok();
                    degree[i] -= 1;
                    degree[v] -= 1;
                    break;
                }
            }
        }

        // Connect the last two remaining degree-1 vertices
        let mut remaining = Vec::new();
        for i in 0..n {
            if degree[i] == 1 {
                remaining.push(i);
            }
        }

        if remaining.len() == 2 {
            graph.add_edge(remaining[0], remaining[1]).ok();
        }

        graph
    }

    /// Increment the Prüfer sequence to the next one (like incrementing a base-n number)
    fn increment_sequence(&mut self) -> bool {
        let n = self.n;
        let len = self.current.len();

        if len == 0 {
            return false;
        }

        // Increment from rightmost position
        let mut pos = len - 1;
        loop {
            self.current[pos] += 1;

            if self.current[pos] < n {
                return true;
            }

            // Carry over
            self.current[pos] = 0;

            if pos == 0 {
                return false; // Overflow - we've generated all sequences
            }

            pos -= 1;
        }
    }
}

impl Iterator for TreeIterator {
    type Item = Graph;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        // Generate tree from current Prüfer sequence
        let tree = Self::prufer_to_tree(&self.current, self.n);

        // Move to next sequence
        if !self.increment_sequence() {
            self.done = true;
        }

        Some(tree)
    }
}

/// Count the number of labeled trees with n vertices
///
/// Uses Cayley's formula: n^(n-2)
///
/// # Arguments
///
/// * `n` - Number of vertices
///
/// # Returns
///
/// The number of distinct labeled trees with n vertices
///
/// # Examples
///
/// ```
/// use rustmath_graphs::trees::count_labeled_trees;
///
/// assert_eq!(count_labeled_trees(1), 1);
/// assert_eq!(count_labeled_trees(2), 1);
/// assert_eq!(count_labeled_trees(3), 3);
/// assert_eq!(count_labeled_trees(4), 16);
/// assert_eq!(count_labeled_trees(5), 125);
/// ```
pub fn count_labeled_trees(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }

    // Cayley's formula: n^(n-2)
    n.pow((n - 2) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_iterator_basic() {
        let iter = TreeIterator::new(3);
        let trees: Vec<_> = iter.collect();

        // There are 3^(3-2) = 3 labeled trees with 3 vertices
        assert_eq!(trees.len(), 3);

        // All should be valid trees
        for tree in &trees {
            assert_eq!(tree.num_vertices(), 3);
            assert_eq!(tree.num_edges(), 2); // Tree with 3 vertices has 2 edges
        }
    }

    #[test]
    fn test_tree_iterator_4_vertices() {
        let iter = TreeIterator::new(4);
        let trees: Vec<_> = iter.collect();

        // There are 4^(4-2) = 16 labeled trees with 4 vertices
        assert_eq!(trees.len(), 16);

        for tree in &trees {
            assert_eq!(tree.num_vertices(), 4);
            assert_eq!(tree.num_edges(), 3); // Tree with 4 vertices has 3 edges
        }
    }

    #[test]
    fn test_tree_iterator_2_vertices() {
        let iter = TreeIterator::new(2);
        let trees: Vec<_> = iter.collect();

        // Only one tree with 2 vertices (an edge)
        assert_eq!(trees.len(), 1);
        assert_eq!(trees[0].num_vertices(), 2);
        assert_eq!(trees[0].num_edges(), 1);
    }

    #[test]
    fn test_num_vertices() {
        let iter = TreeIterator::new(5);
        assert_eq!(iter.num_vertices(), 5);
    }

    #[test]
    fn test_count_labeled_trees() {
        assert_eq!(count_labeled_trees(1), 1);
        assert_eq!(count_labeled_trees(2), 1);
        assert_eq!(count_labeled_trees(3), 3);
        assert_eq!(count_labeled_trees(4), 16);
        assert_eq!(count_labeled_trees(5), 125);
        assert_eq!(count_labeled_trees(6), 1296);
    }

    #[test]
    fn test_prufer_to_tree_basic() {
        // Empty sequence for n=2
        let tree = TreeIterator::prufer_to_tree(&[], 2);
        assert_eq!(tree.num_vertices(), 2);
        assert_eq!(tree.num_edges(), 1);
        assert!(tree.has_edge(0, 1));
    }

    #[test]
    fn test_prufer_to_tree_3_vertices() {
        // Sequence [0] creates a star with center at 0
        let tree = TreeIterator::prufer_to_tree(&[0], 3);
        assert_eq!(tree.num_vertices(), 3);
        assert_eq!(tree.num_edges(), 2);
    }

    #[test]
    fn test_trees_are_connected() {
        let iter = TreeIterator::new(4);

        for tree in iter {
            // Check that tree is connected by verifying it has n-1 edges
            assert_eq!(tree.num_edges(), tree.num_vertices() - 1);
        }
    }

    #[test]
    #[should_panic(expected = "Trees must have at least 2 vertices")]
    fn test_invalid_size() {
        TreeIterator::new(1);
    }

    #[test]
    fn test_iterator_exhaustion() {
        let mut iter = TreeIterator::new(3);

        // Consume all trees
        let count = iter.by_ref().count();
        assert_eq!(count, 3);

        // Iterator should be exhausted
        assert!(iter.next().is_none());
        assert!(iter.next().is_none());
    }
}
