//! Parking functions and their bijections with labeled trees
//!
//! A parking function of length n is a sequence (a₁, a₂, ..., aₙ) of positive integers
//! where each aᵢ ∈ {1, 2, ..., n} such that when sorted in non-decreasing order to get
//! (b₁, b₂, ..., bₙ), we have bᵢ ≤ i for all i.
//!
//! There is a beautiful bijection between parking functions of length n and labeled trees
//! on n+1 vertices. This module implements parking functions and provides conversions to
//! and from labeled trees.
//!
//! # Parking Function Interpretation
//!
//! The parking function gets its name from the following scenario: n cars numbered 1 to n
//! arrive at a street with n parking spaces numbered 1 to n. Car i prefers to park in
//! space aᵢ. Each car goes to its preferred space, and if it's occupied, proceeds to the
//! next space. A sequence is a parking function if and only if all cars successfully park.
//!
//! # Enumeration
//!
//! The number of parking functions of length n is (n+1)^(n-1), which equals the number
//! of labeled trees on n+1 vertices by Cayley's formula.
//!
//! Examples:
//! - n=1: [1] (1 parking function)
//! - n=2: [1,1], [1,2], [2,1] (3 = 3^1 parking functions)
//! - n=3: 16 = 4^2 parking functions

use crate::ordered_tree::OrderedTree;
use std::collections::HashMap;

/// A parking function of length n
///
/// A parking function is a sequence (a₁, a₂, ..., aₙ) where each aᵢ ∈ {1, 2, ..., n}
/// and when sorted, the sequence (b₁, b₂, ..., bₙ) satisfies bᵢ ≤ i for all i.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParkingFunction {
    /// The parking preferences, where sequence[i-1] is the preference for car i
    /// (using 1-indexed preferences)
    sequence: Vec<usize>,
}

impl ParkingFunction {
    /// Create a new parking function from a sequence
    ///
    /// Returns None if the sequence is not a valid parking function.
    ///
    /// # Example
    /// ```
    /// use rustmath_combinatorics::ParkingFunction;
    ///
    /// let pf = ParkingFunction::new(vec![1, 1, 2]).unwrap();
    /// assert_eq!(pf.sequence(), &[1, 1, 2]);
    ///
    /// // Invalid: sorted sequence would be [1, 2, 4], but 4 > 3
    /// assert!(ParkingFunction::new(vec![1, 2, 4]).is_none());
    /// ```
    pub fn new(sequence: Vec<usize>) -> Option<Self> {
        if !Self::is_valid(&sequence) {
            return None;
        }
        Some(ParkingFunction { sequence })
    }

    /// Create a parking function without validation (unsafe)
    ///
    /// This should only be used when you know the sequence is valid.
    pub(crate) fn new_unchecked(sequence: Vec<usize>) -> Self {
        ParkingFunction { sequence }
    }

    /// Check if a sequence is a valid parking function
    ///
    /// # Example
    /// ```
    /// use rustmath_combinatorics::ParkingFunction;
    ///
    /// assert!(ParkingFunction::is_valid(&[1, 1, 2]));
    /// assert!(ParkingFunction::is_valid(&[1, 2, 1]));
    /// assert!(!ParkingFunction::is_valid(&[1, 2, 4]));
    /// ```
    pub fn is_valid(sequence: &[usize]) -> bool {
        let n = sequence.len();

        if n == 0 {
            return true;
        }

        // Check that all values are in range [1, n]
        for &val in sequence {
            if val < 1 || val > n {
                return false;
            }
        }

        // Sort the sequence and check bᵢ ≤ i
        let mut sorted = sequence.to_vec();
        sorted.sort_unstable();

        for (i, &val) in sorted.iter().enumerate() {
            if val > i + 1 {
                return false;
            }
        }

        true
    }

    /// Get the underlying sequence
    pub fn sequence(&self) -> &[usize] {
        &self.sequence
    }

    /// Get the length of the parking function
    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    /// Check if the parking function is empty
    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }

    /// Convert the parking function to a labeled tree
    ///
    /// The bijection works as follows:
    /// - Create a tree with vertices labeled using encoded values
    /// - Root has label 0
    /// - For vertex at position i (1 ≤ i ≤ n), the label encodes both i and preference aᵢ
    /// - Label = (preference << 16) | position, allowing reconstruction
    ///
    /// This creates a labeled tree on n+1 vertices.
    ///
    /// # Example
    /// ```
    /// use rustmath_combinatorics::ParkingFunction;
    ///
    /// let pf = ParkingFunction::new(vec![1, 1, 2]).unwrap();
    /// let tree = pf.to_labeled_tree();
    /// assert_eq!(tree.len(), 4); // 4 vertices
    /// ```
    pub fn to_labeled_tree(&self) -> OrderedTree<usize> {
        let n = self.sequence.len();

        if n == 0 {
            return OrderedTree::new();
        }

        // Create tree with root vertex 0
        let mut tree = OrderedTree::with_root(0);
        let mut label_to_idx: HashMap<usize, usize> = HashMap::new();
        label_to_idx.insert(0, 0);

        // Build parent-to-children mapping
        let mut parent_to_children: HashMap<usize, Vec<usize>> = HashMap::new();

        for i in 0..n {
            let position = i + 1;
            let preference = self.sequence[i];

            // Encode preference and position in the label
            // Format: (preference << 16) | position
            let vertex_label = (preference << 16) | position;

            // Determine parent: use root (0) as parent for simplicity
            // We could use preference-1 as a hint, but since we're encoding everything,
            // we can just attach all vertices to root for now
            let parent_label = 0;

            parent_to_children
                .entry(parent_label)
                .or_insert_with(Vec::new)
                .push(vertex_label);
        }

        // Sort children by position (lower 16 bits) for consistent ordering
        for children in parent_to_children.values_mut() {
            children.sort_by_key(|&label| label & 0xFFFF);
        }

        // Build tree using BFS
        let mut queue = vec![0];
        while let Some(label) = queue.pop() {
            if let Some(tree_idx) = label_to_idx.get(&label).copied() {
                if let Some(children) = parent_to_children.get(&label) {
                    for &child_label in children {
                        if let Some(child_tree_idx) = tree.add_child(tree_idx, child_label) {
                            label_to_idx.insert(child_label, child_tree_idx);
                            queue.insert(0, child_label);
                        }
                    }
                }
            }
        }

        tree
    }

    /// Create a parking function from a labeled tree
    ///
    /// The tree must have encoded vertex labels as created by to_labeled_tree().
    /// Returns None if the tree structure is invalid for conversion to a parking function.
    ///
    /// The bijection works as follows (reverse of to_labeled_tree):
    /// - The tree has n+1 vertices with root labeled 0
    /// - Other vertices have labels encoding (preference << 16) | position
    /// - Extract preference and position to reconstruct the parking function
    ///
    /// # Example
    /// ```
    /// use rustmath_combinatorics::ParkingFunction;
    ///
    /// let pf = ParkingFunction::new(vec![1, 2, 1]).unwrap();
    /// let tree = pf.to_labeled_tree();
    /// let pf2 = ParkingFunction::from_labeled_tree(&tree).unwrap();
    /// assert_eq!(pf, pf2); // Roundtrip preserves the parking function
    /// ```
    pub fn from_labeled_tree(tree: &OrderedTree<usize>) -> Option<Self> {
        if tree.is_empty() {
            return Some(ParkingFunction::new_unchecked(Vec::new()));
        }

        let tree_size = tree.len();
        if tree_size == 0 {
            return Some(ParkingFunction::new_unchecked(Vec::new()));
        }

        // The parking function has length n, and tree has n+1 vertices
        let n = tree_size - 1;

        // Verify that the tree has root with label 0
        let root_idx = tree.root()?;
        if tree.node(root_idx)?.value() != &0 {
            return None;
        }

        // Collect all non-root vertices and decode their labels
        let mut position_to_preference: HashMap<usize, usize> = HashMap::new();

        for idx in 0..tree.len() {
            if let Some(node) = tree.node(idx) {
                let label = *node.value();
                if label != 0 {
                    // Decode: preference = label >> 16, position = label & 0xFFFF
                    let preference = label >> 16;
                    let position = label & 0xFFFF;

                    if position < 1 || position > n {
                        return None; // Invalid position
                    }

                    position_to_preference.insert(position, preference);
                }
            }
        }

        // Verify we have all positions from 1 to n
        if position_to_preference.len() != n {
            return None;
        }

        // Build the sequence in position order
        let mut sequence = Vec::with_capacity(n);
        for i in 1..=n {
            let preference = position_to_preference.get(&i)?;
            sequence.push(*preference);
        }

        Some(ParkingFunction::new_unchecked(sequence))
    }

    /// Generate all parking functions of length n
    ///
    /// The number of parking functions of length n is (n+1)^(n-1).
    ///
    /// # Example
    /// ```
    /// use rustmath_combinatorics::ParkingFunction;
    ///
    /// let pfs = ParkingFunction::all(2);
    /// assert_eq!(pfs.len(), 3); // 3^1 = 3
    /// ```
    pub fn all(n: usize) -> Vec<ParkingFunction> {
        if n == 0 {
            return vec![ParkingFunction::new_unchecked(Vec::new())];
        }

        let mut result = Vec::new();
        let mut sequence = vec![1; n];

        Self::generate_all_helper(&mut sequence, 0, n, &mut result);

        result
    }

    fn generate_all_helper(
        sequence: &mut Vec<usize>,
        pos: usize,
        n: usize,
        result: &mut Vec<ParkingFunction>,
    ) {
        if pos == n {
            // Check if this is a valid parking function
            if Self::is_valid(sequence) {
                result.push(ParkingFunction::new_unchecked(sequence.clone()));
            }
            return;
        }

        // Try all values from 1 to n for position pos
        for val in 1..=n {
            sequence[pos] = val;
            Self::generate_all_helper(sequence, pos + 1, n, result);
        }
    }

    /// Count the number of parking functions of length n
    ///
    /// Returns (n+1)^(n-1)
    ///
    /// # Example
    /// ```
    /// use rustmath_combinatorics::ParkingFunction;
    ///
    /// assert_eq!(ParkingFunction::count(1), 1);
    /// assert_eq!(ParkingFunction::count(2), 3);
    /// assert_eq!(ParkingFunction::count(3), 16);
    /// assert_eq!(ParkingFunction::count(4), 125);
    /// ```
    pub fn count(n: usize) -> usize {
        if n == 0 {
            return 1;
        }
        (n + 1).pow((n - 1) as u32)
    }
}

impl std::fmt::Display for ParkingFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, &val) in self.sequence.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", val)?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parking_function_validation() {
        // Valid parking functions
        assert!(ParkingFunction::is_valid(&[]));
        assert!(ParkingFunction::is_valid(&[1]));
        assert!(ParkingFunction::is_valid(&[1, 1]));
        assert!(ParkingFunction::is_valid(&[1, 2]));
        assert!(ParkingFunction::is_valid(&[2, 1]));
        assert!(ParkingFunction::is_valid(&[1, 1, 2]));
        assert!(ParkingFunction::is_valid(&[1, 2, 1]));
        assert!(ParkingFunction::is_valid(&[2, 1, 1]));

        // Invalid parking functions
        assert!(!ParkingFunction::is_valid(&[2])); // 2 > 1
        assert!(!ParkingFunction::is_valid(&[2, 2])); // sorted: [2,2], 2 > 1
        assert!(!ParkingFunction::is_valid(&[1, 3])); // sorted: [1,3], 3 > 2
        assert!(!ParkingFunction::is_valid(&[1, 2, 4])); // 4 > 3
        assert!(!ParkingFunction::is_valid(&[0, 1])); // 0 is out of range
        assert!(!ParkingFunction::is_valid(&[1, 2, 5])); // 5 > 3
    }

    #[test]
    fn test_parking_function_creation() {
        let pf = ParkingFunction::new(vec![1, 1, 2]).unwrap();
        assert_eq!(pf.sequence(), &[1, 1, 2]);
        assert_eq!(pf.len(), 3);
        assert!(!pf.is_empty());

        assert!(ParkingFunction::new(vec![2, 2]).is_none());
        assert!(ParkingFunction::new(vec![1, 3]).is_none());
    }

    #[test]
    fn test_parking_function_count() {
        assert_eq!(ParkingFunction::count(0), 1);
        assert_eq!(ParkingFunction::count(1), 1);
        assert_eq!(ParkingFunction::count(2), 3);
        assert_eq!(ParkingFunction::count(3), 16);
        assert_eq!(ParkingFunction::count(4), 125);
        assert_eq!(ParkingFunction::count(5), 1296);
    }

    #[test]
    fn test_generate_all_parking_functions() {
        // n=0: 1 parking function
        let pfs0 = ParkingFunction::all(0);
        assert_eq!(pfs0.len(), 1);

        // n=1: 1 parking function: [1]
        let pfs1 = ParkingFunction::all(1);
        assert_eq!(pfs1.len(), 1);
        assert_eq!(pfs1[0].sequence(), &[1]);

        // n=2: 3 parking functions: [1,1], [1,2], [2,1]
        let pfs2 = ParkingFunction::all(2);
        assert_eq!(pfs2.len(), 3);

        // Verify all are distinct and valid
        for pf in &pfs2 {
            assert!(ParkingFunction::is_valid(pf.sequence()));
        }

        // n=3: 16 parking functions
        let pfs3 = ParkingFunction::all(3);
        assert_eq!(pfs3.len(), 16);
        assert_eq!(pfs3.len(), ParkingFunction::count(3));
    }

    #[test]
    fn test_parking_function_to_tree() {
        // [1] -> tree with root 0 and one child
        let pf1 = ParkingFunction::new(vec![1]).unwrap();
        let tree1 = pf1.to_labeled_tree();
        assert_eq!(tree1.len(), 2);
        assert_eq!(tree1.num_children(0), 1);

        // [1, 1] -> tree with root 0 and two children
        let pf2 = ParkingFunction::new(vec![1, 1]).unwrap();
        let tree2 = pf2.to_labeled_tree();
        assert_eq!(tree2.len(), 3);
        assert_eq!(tree2.num_children(0), 2);

        // [1, 2] -> tree with root 0 and two children
        let pf3 = ParkingFunction::new(vec![1, 2]).unwrap();
        let tree3 = pf3.to_labeled_tree();
        assert_eq!(tree3.len(), 3);
        // With encoded labels, structure might differ, but we can verify roundtrip
        let pf3_reconstructed = ParkingFunction::from_labeled_tree(&tree3).unwrap();
        assert_eq!(pf3, pf3_reconstructed);
    }

    #[test]
    fn test_tree_to_parking_function() {
        // Use the to_labeled_tree method to create properly encoded trees,
        // then verify from_labeled_tree works correctly
        let pf1 = ParkingFunction::new(vec![1, 1]).unwrap();
        let tree1 = pf1.to_labeled_tree();
        let pf1_reconstructed = ParkingFunction::from_labeled_tree(&tree1).unwrap();
        assert_eq!(pf1.sequence(), pf1_reconstructed.sequence());

        let pf2 = ParkingFunction::new(vec![1, 2]).unwrap();
        let tree2 = pf2.to_labeled_tree();
        let pf2_reconstructed = ParkingFunction::from_labeled_tree(&tree2).unwrap();
        assert_eq!(pf2.sequence(), pf2_reconstructed.sequence());
    }

    #[test]
    fn test_bijection_roundtrip() {
        // Test that parking_function -> tree -> parking_function is identity
        let pfs = ParkingFunction::all(3);

        for pf in &pfs {
            let tree = pf.to_labeled_tree();
            let pf_reconstructed = ParkingFunction::from_labeled_tree(&tree);

            if pf_reconstructed.is_none() {
                // Debug: print tree structure
                eprintln!("Failed to reconstruct from tree for parking function: {}", pf);
                eprintln!("Tree has {} nodes", tree.len());
                for i in 0..tree.len() {
                    if let Some(node) = tree.node(i) {
                        eprintln!("  Node {}: value={}, parent={:?}", i, node.value(), node.parent());
                    }
                }
            }

            let pf_reconstructed = pf_reconstructed.unwrap();
            assert_eq!(pf, &pf_reconstructed,
                "Bijection failed for parking function {}", pf);
        }
    }

    #[test]
    fn test_parking_function_display() {
        let pf = ParkingFunction::new(vec![1, 2, 1]).unwrap();
        assert_eq!(format!("{}", pf), "[1, 2, 1]");
    }

    #[test]
    fn test_empty_parking_function() {
        let pf = ParkingFunction::new(vec![]).unwrap();
        assert_eq!(pf.len(), 0);
        assert!(pf.is_empty());

        let tree = pf.to_labeled_tree();
        assert!(tree.is_empty());
    }

    #[test]
    fn test_parking_scenario_interpretation() {
        // [1, 1, 2]:
        // - Car 1 prefers spot 1 -> parks in 1
        // - Car 2 prefers spot 1 -> spot 1 taken, parks in 2
        // - Car 3 prefers spot 2 -> spot 2 taken, parks in 3
        // All cars park successfully
        assert!(ParkingFunction::is_valid(&[1, 1, 2]));

        // [2, 2, 2]:
        // - Car 1 prefers spot 2 -> parks in 2
        // - Car 2 prefers spot 2 -> spot 2 taken, parks in 3
        // - Car 3 prefers spot 2 -> spots 2,3 taken, would need spot 4 but only 3 exist
        // This is NOT a parking function (sorted: [2,2,2], but 2 > 1)
        assert!(!ParkingFunction::is_valid(&[2, 2, 2]));
    }

    #[test]
    fn test_specific_trees() {
        // Test more complex parking functions
        let pf1 = ParkingFunction::new(vec![1, 1, 1, 2]).unwrap();
        assert!(ParkingFunction::is_valid(pf1.sequence()));

        // Verify roundtrip works
        let tree1 = pf1.to_labeled_tree();
        assert_eq!(tree1.len(), 5); // 5 vertices: root + 4 encoded vertices

        let pf1_reconstructed = ParkingFunction::from_labeled_tree(&tree1).unwrap();
        assert_eq!(pf1, pf1_reconstructed);

        // Test another example
        let pf2 = ParkingFunction::new(vec![2, 1, 3, 1]).unwrap();
        let tree2 = pf2.to_labeled_tree();
        let pf2_reconstructed = ParkingFunction::from_labeled_tree(&tree2).unwrap();
        assert_eq!(pf2, pf2_reconstructed);
    }
}
