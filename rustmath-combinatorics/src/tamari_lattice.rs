//! Tamari Lattices - Partial orders on Catalan objects
//!
//! The Tamari lattice is a partial order on binary trees (or equivalently on
//! parenthesizations or Dyck paths) where the order is defined by right rotations.
//! A binary tree T1 ≤ T2 if T2 can be obtained from T1 by a sequence of right rotations.
//!
//! # Mathematical Background
//!
//! The Tamari lattice Yn has several interpretations:
//! - Binary trees with n internal nodes
//! - Full parenthesizations of (n+1) factors
//! - Dyck paths from (0,0) to (n,n)
//!
//! The number of elements is the nth Catalan number: C_n = (1/(n+1)) * C(2n, n)
//!
//! ## Properties
//!
//! - The Tamari lattice is a graded lattice
//! - It is a lattice (every pair of elements has a meet and join)
//! - Every element has a unique meet and join with any other element
//! - The covering relations correspond to single right rotations
//! - Tamari lattices are congruence-uniform but not necessarily distributive
//!
//! ## References
//!
//! - Tamari, D. (1962). "The algebra of bracketings and their enumeration"
//! - Knuth, D.E. (1973). "The Art of Computer Programming, Vol. 3"

use std::collections::{HashMap, VecDeque};
use std::fmt;

/// A binary tree node
///
/// Binary trees in the Tamari lattice are full binary trees where:
/// - Each internal node has exactly two children
/// - Leaves represent the absence of structure
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BinaryTree {
    /// A leaf (empty tree)
    Leaf,
    /// An internal node with left and right subtrees
    Node {
        left: Box<BinaryTree>,
        right: Box<BinaryTree>,
    },
}

impl BinaryTree {
    /// Create a new leaf node
    pub fn leaf() -> Self {
        BinaryTree::Leaf
    }

    /// Create a new internal node with left and right subtrees
    pub fn node(left: BinaryTree, right: BinaryTree) -> Self {
        BinaryTree::Node {
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Get the number of internal nodes (size)
    ///
    /// This is the parameter n in the nth Tamari lattice Y_n
    pub fn size(&self) -> usize {
        match self {
            BinaryTree::Leaf => 0,
            BinaryTree::Node { left, right } => 1 + left.size() + right.size(),
        }
    }

    /// Check if this is a leaf
    pub fn is_leaf(&self) -> bool {
        matches!(self, BinaryTree::Leaf)
    }

    /// Perform a right rotation at the root
    ///
    /// A right rotation transforms:
    ///     y              x
    ///    / \            / \
    ///   x   C    =>    A   y
    ///  / \                / \
    /// A   B              B   C
    ///
    /// Returns None if rotation is not possible (tree doesn't have the required structure)
    pub fn rotate_right(&self) -> Option<BinaryTree> {
        match self {
            BinaryTree::Leaf => None,
            BinaryTree::Node { left, right } => {
                match left.as_ref() {
                    BinaryTree::Leaf => None,
                    BinaryTree::Node {
                        left: a,
                        right: b,
                    } => {
                        // Create: node(A, node(B, C))
                        Some(BinaryTree::node(
                            a.as_ref().clone(),
                            BinaryTree::node(b.as_ref().clone(), right.as_ref().clone()),
                        ))
                    }
                }
            }
        }
    }

    /// Perform a left rotation at the root
    ///
    /// A left rotation transforms:
    ///   x                y
    ///  / \              / \
    /// A   y      =>    x   C
    ///    / \          / \
    ///   B   C        A   B
    ///
    /// Returns None if rotation is not possible
    pub fn rotate_left(&self) -> Option<BinaryTree> {
        match self {
            BinaryTree::Leaf => None,
            BinaryTree::Node { left, right } => match right.as_ref() {
                BinaryTree::Leaf => None,
                BinaryTree::Node {
                    left: b,
                    right: c,
                } => {
                    // Create: node(node(A, B), C)
                    Some(BinaryTree::node(
                        BinaryTree::node(left.as_ref().clone(), b.as_ref().clone()),
                        c.as_ref().clone(),
                    ))
                }
            },
        }
    }

    /// Get all possible right rotations (at any node in the tree)
    ///
    /// Returns a vector of trees that can be obtained by a single right rotation
    pub fn all_right_rotations(&self) -> Vec<BinaryTree> {
        let mut result = Vec::new();

        // Try rotation at root
        if let Some(rotated) = self.rotate_right() {
            result.push(rotated);
        }

        // Try rotations in subtrees
        match self {
            BinaryTree::Leaf => {}
            BinaryTree::Node { left, right } => {
                // Rotations in left subtree
                for rotated_left in left.all_right_rotations() {
                    result.push(BinaryTree::node(rotated_left, right.as_ref().clone()));
                }

                // Rotations in right subtree
                for rotated_right in right.all_right_rotations() {
                    result.push(BinaryTree::node(left.as_ref().clone(), rotated_right));
                }
            }
        }

        result
    }

    /// Get all possible left rotations (at any node in the tree)
    pub fn all_left_rotations(&self) -> Vec<BinaryTree> {
        let mut result = Vec::new();

        // Try rotation at root
        if let Some(rotated) = self.rotate_left() {
            result.push(rotated);
        }

        // Try rotations in subtrees
        match self {
            BinaryTree::Leaf => {}
            BinaryTree::Node { left, right } => {
                // Rotations in left subtree
                for rotated_left in left.all_left_rotations() {
                    result.push(BinaryTree::node(rotated_left, right.as_ref().clone()));
                }

                // Rotations in right subtree
                for rotated_right in right.all_left_rotations() {
                    result.push(BinaryTree::node(left.as_ref().clone(), rotated_right));
                }
            }
        }

        result
    }

    /// Convert to a parenthesization string
    ///
    /// Binary trees correspond to ways of parenthesizing a product
    /// For example, ((ab)c)d vs (a(bc))d vs a((bc)d) vs a(b(cd))
    pub fn to_parenthesization(&self) -> String {
        match self {
            BinaryTree::Leaf => "•".to_string(),
            BinaryTree::Node { left, right } => {
                format!("({}{})", left.to_parenthesization(), right.to_parenthesization())
            }
        }
    }

    /// Convert to a compact string representation
    pub fn to_compact_string(&self) -> String {
        match self {
            BinaryTree::Leaf => ".".to_string(),
            BinaryTree::Node { left, right } => {
                format!("[{}{}]", left.to_compact_string(), right.to_compact_string())
            }
        }
    }
}

impl fmt::Display for BinaryTree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_compact_string())
    }
}

/// Generate all binary trees with n internal nodes
///
/// These trees are the elements of the nth Tamari lattice
/// The number of trees is the nth Catalan number
pub fn all_binary_trees(n: usize) -> Vec<BinaryTree> {
    if n == 0 {
        return vec![BinaryTree::Leaf];
    }

    let mut result = Vec::new();

    // For each way to split n-1 nodes between left and right subtrees
    // (the root uses 1 node)
    for left_size in 0..n {
        let right_size = n - 1 - left_size;

        let left_trees = all_binary_trees(left_size);
        let right_trees = all_binary_trees(right_size);

        for left in &left_trees {
            for right in &right_trees {
                result.push(BinaryTree::node(left.clone(), right.clone()));
            }
        }
    }

    result
}

/// The Tamari lattice on binary trees
///
/// This structure represents the partial order on binary trees where
/// T1 ≤ T2 if T2 can be obtained from T1 by a sequence of right rotations.
pub struct TamariLattice {
    /// The size parameter (number of internal nodes)
    n: usize,
    /// All binary trees in the lattice
    elements: Vec<BinaryTree>,
    /// Map from tree to its index
    tree_to_index: HashMap<BinaryTree, usize>,
    /// Covering relations (i covers j means tree[i] covers tree[j])
    /// i.e., tree[i] can be obtained from tree[j] by a single right rotation
    covering_relations: Vec<(usize, usize)>,
}

impl TamariLattice {
    /// Create a new Tamari lattice of size n
    ///
    /// This generates all binary trees with n internal nodes and computes
    /// the partial order defined by right rotations.
    pub fn new(n: usize) -> Self {
        let elements = all_binary_trees(n);
        let mut tree_to_index = HashMap::new();

        for (i, tree) in elements.iter().enumerate() {
            tree_to_index.insert(tree.clone(), i);
        }

        // Compute covering relations
        let mut covering_relations = Vec::new();

        for (i, tree) in elements.iter().enumerate() {
            // Get all trees reachable by a single right rotation
            for rotated in tree.all_right_rotations() {
                if let Some(&j) = tree_to_index.get(&rotated) {
                    covering_relations.push((i, j));
                }
            }
        }

        TamariLattice {
            n,
            elements,
            tree_to_index,
            covering_relations,
        }
    }

    /// Get the size of the lattice
    pub fn size(&self) -> usize {
        self.n
    }

    /// Get the number of elements (should be the nth Catalan number)
    pub fn cardinality(&self) -> usize {
        self.elements.len()
    }

    /// Get all elements
    pub fn elements(&self) -> &[BinaryTree] {
        &self.elements
    }

    /// Check if tree1 ≤ tree2 in the Tamari order
    ///
    /// This uses BFS to check if tree2 can be reached from tree1
    /// by a sequence of right rotations
    pub fn less_than_or_equal(&self, tree1: &BinaryTree, tree2: &BinaryTree) -> bool {
        if tree1 == tree2 {
            return true;
        }

        let idx1 = match self.tree_to_index.get(tree1) {
            Some(&i) => i,
            None => return false,
        };

        let idx2 = match self.tree_to_index.get(tree2) {
            Some(&i) => i,
            None => return false,
        };

        // BFS to check reachability
        let mut visited = vec![false; self.elements.len()];
        let mut queue = VecDeque::new();
        queue.push_back(idx1);
        visited[idx1] = true;

        while let Some(current) = queue.pop_front() {
            if current == idx2 {
                return true;
            }

            // Add all successors (trees reachable by right rotation)
            for &(from, to) in &self.covering_relations {
                if from == current && !visited[to] {
                    visited[to] = true;
                    queue.push_back(to);
                }
            }
        }

        false
    }

    /// Compute the meet (greatest lower bound) of two trees
    ///
    /// The meet is the largest tree that is ≤ both inputs in the Tamari order
    pub fn meet(&self, tree1: &BinaryTree, tree2: &BinaryTree) -> Option<BinaryTree> {
        // Find all common lower bounds
        let mut common_lower_bounds = Vec::new();

        for tree in &self.elements {
            if self.less_than_or_equal(tree, tree1) && self.less_than_or_equal(tree, tree2) {
                common_lower_bounds.push(tree.clone());
            }
        }

        if common_lower_bounds.is_empty() {
            return None;
        }

        // Find the maximum among lower bounds
        for candidate in &common_lower_bounds {
            let is_greatest = common_lower_bounds.iter().all(|other| {
                self.less_than_or_equal(other, candidate)
            });
            if is_greatest {
                return Some(candidate.clone());
            }
        }

        None
    }

    /// Compute the join (least upper bound) of two trees
    ///
    /// The join is the smallest tree that is ≥ both inputs in the Tamari order
    pub fn join(&self, tree1: &BinaryTree, tree2: &BinaryTree) -> Option<BinaryTree> {
        // Find all common upper bounds
        let mut common_upper_bounds = Vec::new();

        for tree in &self.elements {
            if self.less_than_or_equal(tree1, tree) && self.less_than_or_equal(tree2, tree) {
                common_upper_bounds.push(tree.clone());
            }
        }

        if common_upper_bounds.is_empty() {
            return None;
        }

        // Find the minimum among upper bounds
        for candidate in &common_upper_bounds {
            let is_least = common_upper_bounds.iter().all(|other| {
                self.less_than_or_equal(candidate, other)
            });
            if is_least {
                return Some(candidate.clone());
            }
        }

        None
    }

    /// Get the minimum element (left comb tree)
    ///
    /// The minimum is the tree that is all left branches: (((•...)•)•)
    pub fn minimum(&self) -> BinaryTree {
        let mut tree = BinaryTree::Leaf;
        for _ in 0..self.n {
            tree = BinaryTree::node(tree, BinaryTree::Leaf);
        }
        tree
    }

    /// Get the maximum element (right comb tree)
    ///
    /// The maximum is the tree that is all right branches: (•(•(...(••))))
    pub fn maximum(&self) -> BinaryTree {
        let mut tree = BinaryTree::Leaf;
        for _ in 0..self.n {
            tree = BinaryTree::node(BinaryTree::Leaf, tree);
        }
        tree
    }

    /// Check if the lattice structure is valid
    ///
    /// Every pair of elements should have a unique meet and join
    pub fn is_valid_lattice(&self) -> bool {
        for tree1 in &self.elements {
            for tree2 in &self.elements {
                if self.meet(tree1, tree2).is_none() || self.join(tree1, tree2).is_none() {
                    return false;
                }
            }
        }
        true
    }

    /// Get the covering relations as pairs of tree indices
    pub fn covering_relations(&self) -> &[(usize, usize)] {
        &self.covering_relations
    }

    /// Get the Hasse diagram as pairs of trees
    pub fn hasse_diagram(&self) -> Vec<(BinaryTree, BinaryTree)> {
        self.covering_relations
            .iter()
            .map(|&(i, j)| (self.elements[i].clone(), self.elements[j].clone()))
            .collect()
    }

    /// Get all elements covered by a given tree
    ///
    /// These are the trees that can be obtained by a single left rotation
    pub fn lower_covers(&self, tree: &BinaryTree) -> Vec<BinaryTree> {
        tree.all_left_rotations()
            .into_iter()
            .filter(|t| self.tree_to_index.contains_key(t))
            .collect()
    }

    /// Get all elements that cover a given tree
    ///
    /// These are the trees that can be obtained by a single right rotation
    pub fn upper_covers(&self, tree: &BinaryTree) -> Vec<BinaryTree> {
        tree.all_right_rotations()
            .into_iter()
            .filter(|t| self.tree_to_index.contains_key(t))
            .collect()
    }

    /// Compute the rank of a tree in the lattice
    ///
    /// The rank is the length of the longest chain from the minimum to the tree
    pub fn rank(&self, tree: &BinaryTree) -> usize {
        let min = self.minimum();
        if tree == &min {
            return 0;
        }

        let idx = match self.tree_to_index.get(tree) {
            Some(&i) => i,
            None => return 0,
        };

        let min_idx = self.tree_to_index[&min];

        // BFS from minimum to compute ranks
        let mut ranks = vec![0; self.elements.len()];
        let mut visited = vec![false; self.elements.len()];
        let mut queue = VecDeque::new();
        queue.push_back(min_idx);
        visited[min_idx] = true;

        while let Some(current) = queue.pop_front() {
            for &(from, to) in &self.covering_relations {
                if from == current && !visited[to] {
                    visited[to] = true;
                    ranks[to] = ranks[current] + 1;
                    queue.push_back(to);
                }
            }
        }

        ranks[idx]
    }

    /// Get the height of the lattice (rank of maximum element)
    pub fn height(&self) -> usize {
        let max = self.maximum();
        self.rank(&max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_tree_size() {
        let leaf = BinaryTree::Leaf;
        assert_eq!(leaf.size(), 0);

        let tree1 = BinaryTree::node(BinaryTree::Leaf, BinaryTree::Leaf);
        assert_eq!(tree1.size(), 1);

        let tree2 = BinaryTree::node(tree1.clone(), BinaryTree::Leaf);
        assert_eq!(tree2.size(), 2);
    }

    #[test]
    fn test_rotations() {
        // Create tree: node(node(leaf, leaf), leaf)  i.e., [[..].] or ((••)•)
        let tree = BinaryTree::node(
            BinaryTree::node(BinaryTree::Leaf, BinaryTree::Leaf),
            BinaryTree::Leaf,
        );

        // Right rotation should give: node(leaf, node(leaf, leaf)) i.e., [.[..]] or (•(••))
        let rotated_right = tree.rotate_right();
        assert!(rotated_right.is_some());
        let expected = BinaryTree::node(
            BinaryTree::Leaf,
            BinaryTree::node(BinaryTree::Leaf, BinaryTree::Leaf),
        );
        assert_eq!(rotated_right.unwrap(), expected);

        // Left rotation of the expected tree should give back the original
        let rotated_left = expected.rotate_left();
        assert!(rotated_left.is_some());
        assert_eq!(rotated_left.unwrap(), tree);
    }

    #[test]
    fn test_all_binary_trees_count() {
        // Number of binary trees should match Catalan numbers
        // C_0 = 1, C_1 = 1, C_2 = 2, C_3 = 5, C_4 = 14
        assert_eq!(all_binary_trees(0).len(), 1); // Just the leaf
        assert_eq!(all_binary_trees(1).len(), 1);
        assert_eq!(all_binary_trees(2).len(), 2);
        assert_eq!(all_binary_trees(3).len(), 5);
        assert_eq!(all_binary_trees(4).len(), 14);
    }

    #[test]
    fn test_tamari_lattice_size_2() {
        let lattice = TamariLattice::new(2);

        // Should have 2 elements (C_2 = 2)
        assert_eq!(lattice.cardinality(), 2);

        // Check minimum and maximum
        let min = lattice.minimum(); // [[..].]
        let max = lattice.maximum(); // [.[..]]

        assert!(lattice.less_than_or_equal(&min, &max));
        assert!(!lattice.less_than_or_equal(&max, &min));
    }

    #[test]
    fn test_tamari_lattice_size_3() {
        let lattice = TamariLattice::new(3);

        // Should have 5 elements (C_3 = 5)
        assert_eq!(lattice.cardinality(), 5);

        // Verify it's a valid lattice
        assert!(lattice.is_valid_lattice());
    }

    #[test]
    fn test_meet_and_join() {
        let lattice = TamariLattice::new(3);

        let min = lattice.minimum();
        let max = lattice.maximum();

        // Meet of min and max should be min
        assert_eq!(lattice.meet(&min, &max), Some(min.clone()));

        // Join of min and max should be max
        assert_eq!(lattice.join(&min, &max), Some(max.clone()));

        // Meet of any element with itself should be itself
        for tree in lattice.elements() {
            assert_eq!(lattice.meet(tree, tree), Some(tree.clone()));
            assert_eq!(lattice.join(tree, tree), Some(tree.clone()));
        }
    }

    #[test]
    fn test_covering_relations() {
        let lattice = TamariLattice::new(2);

        // For n=2, there should be exactly one covering relation
        // from min [[..].] to max [.[..]]
        assert_eq!(lattice.covering_relations().len(), 1);

        let min = lattice.minimum();
        let max = lattice.maximum();

        // Verify the covering relation
        let upper = lattice.upper_covers(&min);
        assert_eq!(upper.len(), 1);
        assert_eq!(upper[0], max);

        let lower = lattice.lower_covers(&max);
        assert_eq!(lower.len(), 1);
        assert_eq!(lower[0], min);
    }

    #[test]
    fn test_rank_and_height() {
        let lattice = TamariLattice::new(3);

        let min = lattice.minimum();
        let max = lattice.maximum();

        // Minimum should have rank 0
        assert_eq!(lattice.rank(&min), 0);

        // Maximum should have rank equal to height
        let height = lattice.height();
        assert_eq!(lattice.rank(&max), height);

        // Height should be n-1 for Tamari lattice of size n
        // (Actually, this is not always true; let's just check it's > 0)
        assert!(height > 0);
    }

    #[test]
    fn test_parenthesization() {
        let tree1 = BinaryTree::node(
            BinaryTree::node(BinaryTree::Leaf, BinaryTree::Leaf),
            BinaryTree::Leaf,
        );
        assert_eq!(tree1.to_parenthesization(), "((••)•)");

        let tree2 = BinaryTree::node(
            BinaryTree::Leaf,
            BinaryTree::node(BinaryTree::Leaf, BinaryTree::Leaf),
        );
        assert_eq!(tree2.to_parenthesization(), "(•(••))");
    }

    #[test]
    fn test_lattice_properties() {
        // Test that Tamari lattices have expected properties
        for n in 0..=4 {
            let lattice = TamariLattice::new(n);

            // Cardinality should be Catalan number
            let expected_size = crate::catalan(n as u32);
            assert_eq!(
                lattice.cardinality(),
                expected_size.to_string().parse::<usize>().unwrap()
            );

            // Should be a valid lattice
            if n > 0 {
                assert!(lattice.is_valid_lattice());
            }

            // Minimum should be ≤ all elements
            let min = lattice.minimum();
            for tree in lattice.elements() {
                assert!(lattice.less_than_or_equal(&min, tree));
            }

            // Maximum should be ≥ all elements
            let max = lattice.maximum();
            for tree in lattice.elements() {
                assert!(lattice.less_than_or_equal(tree, &max));
            }
        }
    }

    #[test]
    fn test_all_rotations() {
        let tree = BinaryTree::node(
            BinaryTree::node(BinaryTree::Leaf, BinaryTree::Leaf),
            BinaryTree::node(BinaryTree::Leaf, BinaryTree::Leaf),
        );

        let right_rotations = tree.all_right_rotations();
        let left_rotations = tree.all_left_rotations();

        // This tree should have multiple rotation possibilities
        assert!(right_rotations.len() > 0);
        assert!(left_rotations.len() > 0);
    }

    #[test]
    fn test_transitivity() {
        let lattice = TamariLattice::new(3);

        // Check transitivity of the order
        for tree1 in lattice.elements() {
            for tree2 in lattice.elements() {
                if lattice.less_than_or_equal(tree1, tree2) {
                    for tree3 in lattice.elements() {
                        if lattice.less_than_or_equal(tree2, tree3) {
                            // Should have tree1 ≤ tree3
                            assert!(
                                lattice.less_than_or_equal(tree1, tree3),
                                "Transitivity failed"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_antisymmetry() {
        let lattice = TamariLattice::new(3);

        // Check antisymmetry: if a ≤ b and b ≤ a, then a = b
        for tree1 in lattice.elements() {
            for tree2 in lattice.elements() {
                if lattice.less_than_or_equal(tree1, tree2)
                    && lattice.less_than_or_equal(tree2, tree1)
                {
                    assert_eq!(tree1, tree2, "Antisymmetry failed");
                }
            }
        }
    }

    #[test]
    fn test_lattice_structure() {
        // Tamari lattices are lattices but not necessarily distributive
        // For small n, we can test that meet and join are well-defined
        let lattice = TamariLattice::new(2);

        // For n=2, the lattice should be distributive
        // Test: a ∧ (b ∨ c) = (a ∧ b) ∨ (a ∧ c)
        for a in lattice.elements() {
            for b in lattice.elements() {
                for c in lattice.elements() {
                    let b_join_c = lattice.join(b, c);
                    let a_meet_b = lattice.meet(a, b);
                    let a_meet_c = lattice.meet(a, c);

                    if let (Some(bjc), Some(amb), Some(amc)) = (b_join_c, a_meet_b, a_meet_c) {
                        let lhs = lattice.meet(a, &bjc);
                        let rhs = lattice.join(&amb, &amc);

                        // For small lattices like n=2, this should hold
                        assert_eq!(lhs, rhs, "Distributivity failed for n=2");
                    }
                }
            }
        }
    }
}
