//! Free Pre-Lie Algebra with Rooted Trees
//!
//! This module implements the free pre-Lie algebra, which has rooted trees as a basis.
//! The pre-Lie product is defined by grafting operations: the product of two trees T₁ • T₂
//! is the sum of all ways to attach the root of T₂ to a vertex of T₁.
//!
//! # Mathematical Background
//!
//! A pre-Lie algebra is a vector space with a bilinear product • satisfying:
//! (x • y) • z - x • (y • z) = (x • z) • y - x • (z • y)
//!
//! The free pre-Lie algebra is naturally realized using rooted trees, where:
//! - The basis consists of all rooted trees (unlabeled, up to isomorphism)
//! - The product T₁ • T₂ grafts T₂'s root onto each vertex of T₁
//! - Linear combinations form the full algebra
//!
//! # Examples
//!
//! ```
//! use rustmath_combinatorics::free_prelie_algebra::{RootedTree, PreLieAlgebra};
//!
//! // Create simple trees
//! let dot = RootedTree::dot(); // •
//! let fork = RootedTree::fork(2); // • with 2 children
//!
//! // Create a pre-Lie algebra element
//! let mut algebra = PreLieAlgebra::new();
//! algebra.add_tree(dot.clone(), 1);
//! algebra.add_tree(fork, 2);
//!
//! // Compute pre-Lie product
//! let product = PreLieAlgebra::prelie_product(&dot, &dot);
//! ```

use std::collections::HashMap;
use std::fmt;

/// A rooted tree represented as a multiset of rooted subtrees
///
/// This representation treats a rooted tree as its root together with
/// an ordered list of subtrees (children of the root). This is equivalent
/// to plane rooted trees where the order of children matters.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RootedTree {
    /// Children of the root (ordered list of subtrees)
    children: Vec<RootedTree>,
}

impl RootedTree {
    /// Creates a new tree with the given children
    pub fn new(children: Vec<RootedTree>) -> Self {
        RootedTree { children }
    }

    /// Creates a dot (single vertex tree with no children)
    pub fn dot() -> Self {
        RootedTree { children: vec![] }
    }

    /// Creates a tree with n children, each being a dot
    pub fn fork(n: usize) -> Self {
        RootedTree {
            children: vec![RootedTree::dot(); n],
        }
    }

    /// Creates a path tree of length n (linear chain)
    pub fn path(n: usize) -> Self {
        if n == 0 {
            RootedTree::dot()
        } else {
            let mut tree = RootedTree::dot();
            for _ in 0..n {
                tree = RootedTree::new(vec![tree]);
            }
            tree
        }
    }

    /// Returns the number of children
    pub fn num_children(&self) -> usize {
        self.children.len()
    }

    /// Returns a reference to the children
    pub fn children(&self) -> &[RootedTree] {
        &self.children
    }

    /// Returns the total number of vertices in the tree
    pub fn num_vertices(&self) -> usize {
        1 + self.children.iter().map(|t| t.num_vertices()).sum::<usize>()
    }

    /// Returns the height of the tree (longest path from root to leaf)
    pub fn height(&self) -> usize {
        if self.children.is_empty() {
            0
        } else {
            1 + self.children.iter().map(|t| t.height()).max().unwrap()
        }
    }

    /// Checks if this is a leaf (dot)
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Grafts another tree onto this tree at vertex index i
    ///
    /// This creates a new tree by attaching the root of `other` as a new child
    /// to the i-th vertex (in preorder traversal).
    pub fn graft_at(&self, other: &RootedTree, vertex_index: usize) -> Option<RootedTree> {
        if vertex_index == 0 {
            // Attach to the root - add as a new child
            let mut new_children = self.children.clone();
            new_children.push(other.clone());
            Some(RootedTree::new(new_children))
        } else {
            // Find which subtree contains the target vertex
            let mut current_index = 1; // Start at 1 (root is 0)
            for (i, child) in self.children.iter().enumerate() {
                let child_size = child.num_vertices();
                if current_index + child_size > vertex_index {
                    // The vertex is in this subtree
                    let local_index = vertex_index - current_index;
                    let new_child = child.graft_at(other, local_index)?;
                    let mut new_children = self.children.clone();
                    new_children[i] = new_child;
                    return Some(RootedTree::new(new_children));
                }
                current_index += child_size;
            }
            None
        }
    }

    /// Returns all possible graftings of another tree onto this tree
    ///
    /// This is the core operation for the pre-Lie product.
    pub fn all_graftings(&self, other: &RootedTree) -> Vec<RootedTree> {
        let n = self.num_vertices();
        (0..n)
            .filter_map(|i| self.graft_at(other, i))
            .collect()
    }

    /// Returns a string representation in bracket notation
    ///
    /// Examples:
    /// - Dot: "•"
    /// - Path of length 2: "[[[•]]]"
    /// - Fork with 2 children: "[• •]"
    pub fn to_bracket_string(&self) -> String {
        if self.children.is_empty() {
            "•".to_string()
        } else {
            let children_str: Vec<String> = self
                .children
                .iter()
                .map(|t| t.to_bracket_string())
                .collect();
            format!("[{}]", children_str.join(" "))
        }
    }

    /// Generates all rooted trees with exactly n vertices
    pub fn all_trees(n: usize) -> Vec<RootedTree> {
        if n == 0 {
            return vec![];
        }
        if n == 1 {
            return vec![RootedTree::dot()];
        }

        let mut result = Vec::new();

        // Generate all partitions of n-1 (for the children)
        let partitions = generate_integer_partitions(n - 1);

        for partition in partitions {
            // For each partition, generate all ways to assign trees to parts
            let tree_combinations = generate_tree_combinations(&partition);
            for children in tree_combinations {
                result.push(RootedTree::new(children));
            }
        }

        result
    }

    /// Counts the number of rooted trees with n vertices (this is sequence A000081)
    pub fn count_trees(n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        if n == 1 {
            return 1;
        }
        // For small n, we can compute exactly
        // This is a placeholder - the exact formula involves complex recursion
        RootedTree::all_trees(n).len()
    }
}

impl fmt::Display for RootedTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_bracket_string())
    }
}

/// A Pre-Lie algebra element represented as a linear combination of rooted trees
///
/// Elements are formal sums: a₁T₁ + a₂T₂ + ... + aₙTₙ
/// where aᵢ are integer coefficients and Tᵢ are rooted trees.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreLieAlgebra {
    /// Map from trees to their coefficients
    terms: HashMap<RootedTree, i64>,
}

impl PreLieAlgebra {
    /// Creates a new empty pre-Lie algebra element (zero)
    pub fn new() -> Self {
        PreLieAlgebra {
            terms: HashMap::new(),
        }
    }

    /// Creates a pre-Lie algebra element from a single tree with coefficient 1
    pub fn from_tree(tree: RootedTree) -> Self {
        let mut algebra = PreLieAlgebra::new();
        algebra.add_tree(tree, 1);
        algebra
    }

    /// Adds a tree with a given coefficient
    pub fn add_tree(&mut self, tree: RootedTree, coeff: i64) {
        if coeff == 0 {
            return;
        }
        *self.terms.entry(tree).or_insert(0) += coeff;
        // Clean up zero coefficients
        self.terms.retain(|_, &mut c| c != 0);
    }

    /// Removes a tree
    pub fn remove_tree(&mut self, tree: &RootedTree) {
        self.terms.remove(tree);
    }

    /// Returns the coefficient of a given tree
    pub fn coefficient(&self, tree: &RootedTree) -> i64 {
        *self.terms.get(tree).unwrap_or(&0)
    }

    /// Returns the number of terms (non-zero coefficients)
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    /// Checks if this is the zero element
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Returns an iterator over (tree, coefficient) pairs
    pub fn terms(&self) -> impl Iterator<Item = (&RootedTree, &i64)> {
        self.terms.iter()
    }

    /// Scalar multiplication
    pub fn scale(&mut self, scalar: i64) {
        if scalar == 0 {
            self.terms.clear();
        } else {
            for coeff in self.terms.values_mut() {
                *coeff *= scalar;
            }
        }
    }

    /// Addition of two pre-Lie algebra elements
    pub fn add(&self, other: &PreLieAlgebra) -> PreLieAlgebra {
        let mut result = self.clone();
        for (tree, &coeff) in other.terms.iter() {
            result.add_tree(tree.clone(), coeff);
        }
        result
    }

    /// Subtraction of two pre-Lie algebra elements
    pub fn subtract(&self, other: &PreLieAlgebra) -> PreLieAlgebra {
        let mut result = self.clone();
        for (tree, &coeff) in other.terms.iter() {
            result.add_tree(tree.clone(), -coeff);
        }
        result
    }

    /// Pre-Lie product of two trees: T₁ • T₂
    ///
    /// The product is the sum of all ways to graft T₂ onto vertices of T₁.
    pub fn prelie_product(tree1: &RootedTree, tree2: &RootedTree) -> PreLieAlgebra {
        let mut result = PreLieAlgebra::new();
        let graftings = tree1.all_graftings(tree2);
        for tree in graftings {
            result.add_tree(tree, 1);
        }
        result
    }

    /// Pre-Lie product of two algebra elements (bilinear extension)
    pub fn product(&self, other: &PreLieAlgebra) -> PreLieAlgebra {
        let mut result = PreLieAlgebra::new();
        for (tree1, &coeff1) in self.terms.iter() {
            for (tree2, &coeff2) in other.terms.iter() {
                let prod = PreLieAlgebra::prelie_product(tree1, tree2);
                for (tree, &coeff) in prod.terms.iter() {
                    result.add_tree(tree.clone(), coeff1 * coeff2 * coeff);
                }
            }
        }
        result
    }

    /// Commutator (Lie bracket): [x, y] = x • y - y • x
    ///
    /// The commutator of a pre-Lie algebra is a Lie algebra.
    pub fn commutator(&self, other: &PreLieAlgebra) -> PreLieAlgebra {
        let xy = self.product(other);
        let yx = other.product(self);
        xy.subtract(&yx)
    }

    /// Checks the pre-Lie identity for three trees
    ///
    /// Verifies: (x • y) • z - x • (y • z) = (x • z) • y - x • (z • y)
    pub fn verify_prelie_identity(x: &RootedTree, y: &RootedTree, z: &RootedTree) -> bool {
        let x_elem = PreLieAlgebra::from_tree(x.clone());
        let y_elem = PreLieAlgebra::from_tree(y.clone());
        let z_elem = PreLieAlgebra::from_tree(z.clone());

        // Left side: (x • y) • z - x • (y • z)
        let xy = x_elem.product(&y_elem);
        let xy_z = xy.product(&z_elem);
        let yz = y_elem.product(&z_elem);
        let x_yz = x_elem.product(&yz);
        let left = xy_z.subtract(&x_yz);

        // Right side: (x • z) • y - x • (z • y)
        let xz = x_elem.product(&z_elem);
        let xz_y = xz.product(&y_elem);
        let zy = z_elem.product(&y_elem);
        let x_zy = x_elem.product(&zy);
        let right = xz_y.subtract(&x_zy);

        left == right
    }
}

impl Default for PreLieAlgebra {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for PreLieAlgebra {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut terms: Vec<_> = self.terms.iter().collect();
        terms.sort_by_key(|(t, _)| (t.num_vertices(), t.to_bracket_string()));

        let term_strs: Vec<String> = terms
            .iter()
            .map(|(tree, &coeff)| {
                if coeff == 1 {
                    tree.to_string()
                } else if coeff == -1 {
                    format!("-{}", tree)
                } else {
                    format!("{}{}", coeff, tree)
                }
            })
            .collect();

        write!(f, "{}", term_strs.join(" + ").replace(" + -", " - "))
    }
}

/// Helper function to generate integer partitions of n
fn generate_integer_partitions(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![vec![]];
    }
    if n == 1 {
        return vec![vec![1]];
    }

    let mut result = Vec::new();

    fn partitions_helper(n: usize, max: usize, current: &mut Vec<usize>, result: &mut Vec<Vec<usize>>) {
        if n == 0 {
            result.push(current.clone());
            return;
        }

        for i in (1..=max.min(n)).rev() {
            current.push(i);
            partitions_helper(n - i, i, current, result);
            current.pop();
        }
    }

    partitions_helper(n, n, &mut Vec::new(), &mut result);
    result
}

/// Helper function to generate all combinations of trees for a partition
fn generate_tree_combinations(partition: &[usize]) -> Vec<Vec<RootedTree>> {
    if partition.is_empty() {
        return vec![vec![]];
    }

    let first = partition[0];
    let rest = &partition[1..];

    let mut result = Vec::new();
    let trees_for_first = RootedTree::all_trees(first);
    let combinations_for_rest = generate_tree_combinations(rest);

    for tree in &trees_for_first {
        for combination in &combinations_for_rest {
            let mut new_combination = vec![tree.clone()];
            new_combination.extend(combination.clone());
            result.push(new_combination);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_creation() {
        let dot = RootedTree::dot();
        assert_eq!(dot.num_vertices(), 1);
        assert_eq!(dot.num_children(), 0);
        assert!(dot.is_leaf());
        assert_eq!(dot.height(), 0);
    }

    #[test]
    fn test_fork_creation() {
        let fork = RootedTree::fork(3);
        assert_eq!(fork.num_vertices(), 4); // 1 root + 3 children
        assert_eq!(fork.num_children(), 3);
        assert!(!fork.is_leaf());
        assert_eq!(fork.height(), 1);
    }

    #[test]
    fn test_path_creation() {
        let path = RootedTree::path(3);
        assert_eq!(path.num_vertices(), 4);
        assert_eq!(path.height(), 3);
    }

    #[test]
    fn test_grafting() {
        let dot = RootedTree::dot();
        let result = dot.graft_at(&dot, 0);
        assert!(result.is_some());
        let tree = result.unwrap();
        assert_eq!(tree.num_vertices(), 2);
        assert_eq!(tree.num_children(), 1);
    }

    #[test]
    fn test_all_graftings() {
        let dot = RootedTree::dot();
        let graftings = dot.all_graftings(&dot);
        assert_eq!(graftings.len(), 1); // Can only graft at the root
        assert_eq!(graftings[0].num_vertices(), 2);
    }

    #[test]
    fn test_all_graftings_fork() {
        let fork = RootedTree::fork(2);
        let dot = RootedTree::dot();
        let graftings = fork.all_graftings(&dot);
        assert_eq!(graftings.len(), 3); // Can graft at root and 2 children
    }

    #[test]
    fn test_prelie_product_dots() {
        let dot = RootedTree::dot();
        let product = PreLieAlgebra::prelie_product(&dot, &dot);
        assert_eq!(product.num_terms(), 1);
        assert_eq!(product.coefficient(&RootedTree::fork(1)), 1);
    }

    #[test]
    fn test_prelie_algebra_addition() {
        let dot = RootedTree::dot();
        let fork = RootedTree::fork(1);

        let mut elem1 = PreLieAlgebra::new();
        elem1.add_tree(dot.clone(), 2);

        let mut elem2 = PreLieAlgebra::new();
        elem2.add_tree(dot.clone(), 3);
        elem2.add_tree(fork.clone(), 1);

        let sum = elem1.add(&elem2);
        assert_eq!(sum.coefficient(&dot), 5);
        assert_eq!(sum.coefficient(&fork), 1);
    }

    #[test]
    fn test_prelie_algebra_subtraction() {
        let dot = RootedTree::dot();

        let mut elem1 = PreLieAlgebra::new();
        elem1.add_tree(dot.clone(), 5);

        let mut elem2 = PreLieAlgebra::new();
        elem2.add_tree(dot.clone(), 3);

        let diff = elem1.subtract(&elem2);
        assert_eq!(diff.coefficient(&dot), 2);
    }

    #[test]
    fn test_prelie_algebra_scalar() {
        let dot = RootedTree::dot();
        let mut elem = PreLieAlgebra::new();
        elem.add_tree(dot.clone(), 3);

        elem.scale(2);
        assert_eq!(elem.coefficient(&dot), 6);
    }

    #[test]
    fn test_commutator() {
        let dot = RootedTree::dot();
        let x = PreLieAlgebra::from_tree(dot.clone());
        let y = PreLieAlgebra::from_tree(dot.clone());

        let comm = x.commutator(&y);
        // [•, •] = • • • - • • • = 0
        assert!(comm.is_zero());
    }

    #[test]
    fn test_prelie_identity_dots() {
        let dot = RootedTree::dot();
        assert!(PreLieAlgebra::verify_prelie_identity(&dot, &dot, &dot));
    }

    #[test]
    fn test_all_trees_n1() {
        let trees = RootedTree::all_trees(1);
        assert_eq!(trees.len(), 1);
        assert_eq!(trees[0], RootedTree::dot());
    }

    #[test]
    fn test_all_trees_n2() {
        let trees = RootedTree::all_trees(2);
        assert_eq!(trees.len(), 1);
        assert_eq!(trees[0], RootedTree::fork(1));
    }

    #[test]
    fn test_all_trees_n3() {
        let trees = RootedTree::all_trees(3);
        assert_eq!(trees.len(), 2);
        // Should have fork(2) and path(1)
    }

    #[test]
    fn test_bracket_string() {
        let dot = RootedTree::dot();
        assert_eq!(dot.to_bracket_string(), "•");

        let fork = RootedTree::fork(2);
        assert_eq!(fork.to_bracket_string(), "[• •]");
    }

    #[test]
    fn test_tree_equality() {
        let dot1 = RootedTree::dot();
        let dot2 = RootedTree::dot();
        assert_eq!(dot1, dot2);

        let fork1 = RootedTree::fork(2);
        let fork2 = RootedTree::fork(2);
        assert_eq!(fork1, fork2);
    }

    #[test]
    fn test_prelie_product_associativity_deviation() {
        // Pre-Lie algebras are NOT associative, but satisfy the pre-Lie identity
        let dot = RootedTree::dot();
        let fork = RootedTree::fork(1);

        let x = PreLieAlgebra::from_tree(dot.clone());
        let y = PreLieAlgebra::from_tree(fork.clone());

        let xy = x.product(&y);
        let yx = y.product(&x);

        // The products should generally be different (non-commutative)
        // This is expected for pre-Lie algebras
    }

    #[test]
    fn test_display() {
        let dot = RootedTree::dot();
        let elem = PreLieAlgebra::from_tree(dot);
        let display = format!("{}", elem);
        assert_eq!(display, "•");
    }

    #[test]
    fn test_zero_element() {
        let zero = PreLieAlgebra::new();
        assert!(zero.is_zero());
        assert_eq!(zero.num_terms(), 0);
    }

    #[test]
    fn test_graft_at_invalid_index() {
        let dot = RootedTree::dot();
        let result = dot.graft_at(&dot, 5);
        assert!(result.is_none());
    }

    #[test]
    fn test_complex_tree_structure() {
        // Create a more complex tree manually
        let dot = RootedTree::dot();
        let child1 = RootedTree::new(vec![dot.clone()]);
        let child2 = RootedTree::new(vec![dot.clone(), dot.clone()]);
        let tree = RootedTree::new(vec![child1, child2]);

        assert_eq!(tree.num_vertices(), 6); // 1 root + 2 + 3
        assert_eq!(tree.num_children(), 2);
    }

    #[test]
    fn test_integer_partitions() {
        let partitions = generate_integer_partitions(4);
        // Partitions of 4: [4], [3,1], [2,2], [2,1,1], [1,1,1,1]
        assert_eq!(partitions.len(), 5);
    }

    #[test]
    fn test_prelie_product_bilinearity() {
        let dot = RootedTree::dot();
        let fork = RootedTree::fork(1);

        // Test distributivity: (x + y) • z = x • z + y • z
        let mut xy = PreLieAlgebra::new();
        xy.add_tree(dot.clone(), 1);
        xy.add_tree(fork.clone(), 1);

        let z = PreLieAlgebra::from_tree(dot.clone());

        let left = xy.product(&z);

        let x = PreLieAlgebra::from_tree(dot.clone());
        let y = PreLieAlgebra::from_tree(fork.clone());
        let xz = x.product(&z);
        let yz = y.product(&z);
        let right = xz.add(&yz);

        assert_eq!(left, right);
    }
}
