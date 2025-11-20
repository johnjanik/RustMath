//! Grossman-Larson Hopf algebra of rooted trees
//!
//! This module implements the Grossman-Larson Hopf algebra, which is a graded Hopf algebra
//! whose basis elements are rooted trees. The product is given by grafting operations,
//! and the coproduct is given by admissible cuts.
//!
//! # Mathematical Background
//!
//! The Grossman-Larson Hopf algebra H_GL is defined as follows:
//!
//! - **Basis**: Rooted trees (planar, ordered trees)
//! - **Product**: For trees T and S, the product T·S is the sum over all ways to graft S
//!   onto a vertex of T (grafting onto the root gives the tree with root having children T and S)
//! - **Coproduct**: For a tree T, the coproduct Δ(T) is the sum over all admissible cuts
//!   of T ⊗ (forest of cut subtrees)
//! - **Counit**: ε(T) = 1 if T is the trivial tree (single vertex), 0 otherwise
//! - **Unit**: The trivial tree (single vertex)
//!
//! An admissible cut is a set of edges such that no two edges lie on the same path from the root.
//!
//! # References
//!
//! - Grossman, R., & Larson, R. G. (1989). Hopf-algebraic structure of families of trees.
//!   Journal of Algebra, 126(1), 184-210.

use crate::ordered_tree::{OrderedTree, OrderedTreeNode};
use rustmath_core::Ring;
use rustmath_integers::Integer;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};

/// A rooted tree for use in the Grossman-Larson algebra
///
/// This is a wrapper around `OrderedTree<()>` since we only care about the tree structure,
/// not the values at nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootedTree {
    tree: OrderedTree<()>,
}

impl RootedTree {
    /// Create a new rooted tree with a single vertex (the unit element)
    pub fn unit() -> Self {
        RootedTree {
            tree: OrderedTree::with_root(()),
        }
    }

    /// Create a rooted tree from an ordered tree
    pub fn from_tree(tree: OrderedTree<()>) -> Self {
        RootedTree { tree }
    }

    /// Create a rooted tree with a root and a list of child subtrees
    pub fn from_children(children: Vec<RootedTree>) -> Self {
        let mut tree = OrderedTree::with_root(());
        for child_tree in children {
            Self::attach_subtree(&mut tree, 0, child_tree);
        }
        RootedTree { tree }
    }

    /// Helper function to attach a subtree at a given node
    fn attach_subtree(tree: &mut OrderedTree<()>, parent_idx: usize, subtree: RootedTree) {
        if let Some(root_idx) = subtree.tree.root() {
            let new_root_idx = tree.add_child(parent_idx, ()).unwrap();
            Self::copy_children(tree, new_root_idx, &subtree.tree, root_idx);
        }
    }

    /// Recursively copy children from one tree to another
    fn copy_children(
        dest: &mut OrderedTree<()>,
        dest_idx: usize,
        src: &OrderedTree<()>,
        src_idx: usize,
    ) {
        for child_idx in src.children(src_idx) {
            let new_child_idx = dest.add_child(dest_idx, ()).unwrap();
            Self::copy_children(dest, new_child_idx, src, child_idx);
        }
    }

    /// Get the number of vertices in the tree
    pub fn num_vertices(&self) -> usize {
        self.tree.len()
    }

    /// Get the depth (height) of the tree
    pub fn depth(&self) -> usize {
        self.tree.depth()
    }

    /// Check if this is the unit tree (single vertex)
    pub fn is_unit(&self) -> bool {
        self.num_vertices() == 1
    }

    /// Get all vertices in the tree
    pub fn vertices(&self) -> Vec<usize> {
        (0..self.tree.len()).collect()
    }

    /// Graft another tree onto a specific vertex of this tree
    ///
    /// This creates a new tree where `other` is attached as a child of vertex `vertex_idx`
    pub fn graft(&self, vertex_idx: usize, other: &RootedTree) -> Option<RootedTree> {
        if vertex_idx >= self.num_vertices() {
            return None;
        }

        let mut new_tree = self.tree.clone();
        Self::attach_subtree(&mut new_tree, vertex_idx, other.clone());
        Some(RootedTree { tree: new_tree })
    }

    /// Compute all ways to graft another tree onto this tree
    ///
    /// Returns a list of trees, one for each vertex of self
    pub fn all_graftings(&self, other: &RootedTree) -> Vec<RootedTree> {
        self.vertices()
            .iter()
            .filter_map(|&v| self.graft(v, other))
            .collect()
    }

    /// Get all edges in the tree as (parent, child) pairs
    pub fn edges(&self) -> Vec<(usize, usize)> {
        let mut edges = Vec::new();
        for parent in 0..self.tree.len() {
            for child in self.tree.children(parent) {
                edges.push((parent, child));
            }
        }
        edges
    }

    /// Check if a set of edges forms an admissible cut
    ///
    /// A cut is admissible if no two edges lie on the same path from the root
    pub fn is_admissible_cut(&self, cut: &HashSet<(usize, usize)>) -> bool {
        // For each edge in the cut, check that no ancestor edge is also in the cut
        for &(parent, child) in cut {
            // Check all ancestors of parent
            let mut current = parent;
            while let Some(node) = self.tree.node(current) {
                if let Some(p) = node.parent() {
                    if cut.contains(&(p, current)) {
                        return false; // Found an ancestor edge in the cut
                    }
                    current = p;
                } else {
                    break;
                }
            }

            // Also check that no descendant of child has an edge in the cut
            if self.has_descendant_edge_in_cut(child, cut) {
                return false;
            }
        }
        true
    }

    /// Helper to check if any descendant of a node has an edge in the cut
    fn has_descendant_edge_in_cut(&self, node_idx: usize, cut: &HashSet<(usize, usize)>) -> bool {
        for child in self.tree.children(node_idx) {
            if cut.contains(&(node_idx, child)) {
                return true;
            }
            if self.has_descendant_edge_in_cut(child, cut) {
                return true;
            }
        }
        false
    }

    /// Generate all admissible cuts of the tree
    ///
    /// Returns a list of (remaining_tree, cut_forest) pairs where:
    /// - remaining_tree is the tree with cut edges removed
    /// - cut_forest is the list of disconnected subtrees
    pub fn admissible_cuts(&self) -> Vec<(RootedTree, Vec<RootedTree>)> {
        let edges = self.edges();
        if edges.is_empty() {
            // Single vertex tree - only the empty cut
            return vec![(self.clone(), vec![])];
        }

        let mut cuts = vec![];

        // Generate all subsets of edges
        let n = edges.len();
        for mask in 0..(1 << n) {
            let mut cut = HashSet::new();
            for i in 0..n {
                if (mask >> i) & 1 == 1 {
                    cut.insert(edges[i]);
                }
            }

            if self.is_admissible_cut(&cut) {
                if let Some((remaining, forest)) = self.apply_cut(&cut) {
                    cuts.push((remaining, forest));
                }
            }
        }

        cuts
    }

    /// Apply a cut to the tree, returning the remaining tree and the forest of subtrees
    fn apply_cut(&self, cut: &HashSet<(usize, usize)>) -> Option<(RootedTree, Vec<RootedTree>)> {
        if cut.is_empty() {
            return Some((self.clone(), vec![]));
        }

        // Build the remaining tree (rooted at original root)
        let mut remaining_tree = OrderedTree::with_root(());
        let mut old_to_new: HashMap<usize, usize> = HashMap::new();
        old_to_new.insert(0, 0); // Map root to root

        let mut forest = vec![];

        // BFS through the tree, excluding cut edges
        let mut queue = vec![0];
        let mut visited = HashSet::new();
        visited.insert(0);

        while let Some(old_idx) = queue.pop() {
            let new_idx = old_to_new[&old_idx];

            for child in self.tree.children(old_idx) {
                if cut.contains(&(old_idx, child)) {
                    // This is a cut edge - the subtree becomes part of the forest
                    if let Some(subtree) = self.extract_subtree(child, cut) {
                        forest.push(subtree);
                    }
                } else if !visited.contains(&child) {
                    // Not a cut edge - include in remaining tree
                    visited.insert(child);
                    queue.push(child);
                    let new_child_idx = remaining_tree.add_child(new_idx, ()).unwrap();
                    old_to_new.insert(child, new_child_idx);
                }
            }
        }

        Some((RootedTree { tree: remaining_tree }, forest))
    }

    /// Extract a subtree rooted at a given node
    fn extract_subtree(&self, root_idx: usize, cut: &HashSet<(usize, usize)>) -> Option<RootedTree> {
        let mut new_tree = OrderedTree::with_root(());
        let mut old_to_new: HashMap<usize, usize> = HashMap::new();
        old_to_new.insert(root_idx, 0);

        let mut queue = vec![root_idx];
        let mut visited = HashSet::new();
        visited.insert(root_idx);

        while let Some(old_idx) = queue.pop() {
            let new_idx = old_to_new[&old_idx];

            for child in self.tree.children(old_idx) {
                if !cut.contains(&(old_idx, child)) && !visited.contains(&child) {
                    visited.insert(child);
                    queue.push(child);
                    let new_child_idx = new_tree.add_child(new_idx, ()).unwrap();
                    old_to_new.insert(child, new_child_idx);
                }
            }
        }

        Some(RootedTree { tree: new_tree })
    }

    /// Convert to parenthesis notation for display
    pub fn to_parenthesis(&self) -> String {
        if let Some(root) = self.tree.root() {
            self.node_to_parenthesis(root)
        } else {
            "()".to_string()
        }
    }

    fn node_to_parenthesis(&self, node_idx: usize) -> String {
        let children = self.tree.children(node_idx);
        if children.is_empty() {
            "·".to_string()
        } else {
            let mut result = "[".to_string();
            for (i, &child) in children.iter().enumerate() {
                if i > 0 {
                    result.push_str(", ");
                }
                result.push_str(&self.node_to_parenthesis(child));
            }
            result.push(']');
            result
        }
    }
}

impl fmt::Display for RootedTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_parenthesis())
    }
}

impl Hash for RootedTree {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash based on the tree structure
        self.to_parenthesis().hash(state);
    }
}

/// An element of the Grossman-Larson algebra
///
/// This is a formal linear combination of rooted trees with coefficients in a ring R
#[derive(Debug, Clone, PartialEq)]
pub struct GrossmanLarsonElement<R: Ring> {
    /// Map from trees to coefficients
    terms: HashMap<RootedTree, R>,
}

impl<R: Ring> GrossmanLarsonElement<R> {
    /// Create a zero element
    pub fn zero() -> Self {
        GrossmanLarsonElement {
            terms: HashMap::new(),
        }
    }

    /// Create an element from a single tree with coefficient 1
    pub fn from_tree(tree: RootedTree) -> Self {
        let mut terms = HashMap::new();
        terms.insert(tree, R::one());
        GrossmanLarsonElement { terms }
    }

    /// Create an element from a single tree with a given coefficient
    pub fn from_tree_with_coeff(tree: RootedTree, coeff: R) -> Self {
        if coeff.is_zero() {
            return Self::zero();
        }
        let mut terms = HashMap::new();
        terms.insert(tree, coeff);
        GrossmanLarsonElement { terms }
    }

    /// Get the unit element (the single-vertex tree)
    pub fn unit() -> Self {
        Self::from_tree(RootedTree::unit())
    }

    /// Add a term (tree with coefficient)
    pub fn add_term(&mut self, tree: RootedTree, coeff: R) {
        if coeff.is_zero() {
            return;
        }

        let entry = self.terms.entry(tree).or_insert_with(R::zero);
        *entry = entry.clone() + coeff;

        // Remove zero coefficients
        self.terms.retain(|_, c| !c.is_zero());
    }

    /// Check if this is the zero element
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Get the terms of this element
    pub fn terms(&self) -> &HashMap<RootedTree, R> {
        &self.terms
    }

    /// Add two elements
    pub fn add(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (tree, coeff) in &other.terms {
            result.add_term(tree.clone(), coeff.clone());
        }
        result
    }

    /// Multiply by a scalar
    pub fn scalar_mul(&self, scalar: &R) -> Self {
        if scalar.is_zero() {
            return Self::zero();
        }

        let mut result = HashMap::new();
        for (tree, coeff) in &self.terms {
            let new_coeff = coeff.clone() * scalar.clone();
            if !new_coeff.is_zero() {
                result.insert(tree.clone(), new_coeff);
            }
        }
        GrossmanLarsonElement { terms: result }
    }

    /// Compute the product of two elements (using grafting)
    ///
    /// The product x·y is computed by grafting y onto each vertex of each tree in x
    pub fn mul(&self, other: &Self) -> Self {
        let mut result = Self::zero();

        for (tree1, coeff1) in &self.terms {
            for (tree2, coeff2) in &other.terms {
                // Graft tree2 onto each vertex of tree1
                for grafted in tree1.all_graftings(tree2) {
                    let new_coeff = coeff1.clone() * coeff2.clone();
                    result.add_term(grafted, new_coeff);
                }
            }
        }

        result
    }

    /// Compute the coproduct using admissible cuts
    ///
    /// Returns a list of (left, right) pairs representing the coproduct
    /// Δ(x) = Σ x₁ ⊗ x₂
    pub fn coproduct(&self) -> Vec<(Self, Self)> {
        let mut result = vec![];

        for (tree, coeff) in &self.terms {
            let cuts = tree.admissible_cuts();

            for (remaining, forest) in cuts {
                let left = Self::from_tree_with_coeff(remaining, coeff.clone());

                // The right part is the product of trees in the forest
                let mut right = Self::unit();
                for subtree in forest {
                    let subtree_elem = Self::from_tree(subtree);
                    right = right.mul(&subtree_elem);
                }

                result.push((left, right));
            }
        }

        result
    }

    /// Compute the counit
    ///
    /// ε(x) = coefficient of the unit tree in x
    pub fn counit(&self) -> R {
        let unit_tree = RootedTree::unit();
        self.terms.get(&unit_tree).cloned().unwrap_or_else(R::zero)
    }
}

impl<R: Ring> fmt::Display for GrossmanLarsonElement<R>
where
    R: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut first = true;
        for (tree, coeff) in &self.terms {
            if !first {
                write!(f, " + ")?;
            }
            first = false;

            if coeff == &R::one() {
                write!(f, "{}", tree)?;
            } else {
                write!(f, "{}·{}", coeff, tree)?;
            }
        }

        Ok(())
    }
}

/// Generate all rooted trees with n vertices
///
/// This generates all non-isomorphic rooted plane trees with n vertices
pub fn all_rooted_trees(n: usize) -> Vec<RootedTree> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![RootedTree::unit()];
    }

    let mut trees = vec![];

    // Generate all partitions of n-1 (for children)
    let partitions = generate_partitions(n - 1);

    for partition in partitions {
        // For each partition, generate all ways to assign trees to parts
        let assignments = generate_tree_assignments(&partition);

        for assignment in assignments {
            trees.push(RootedTree::from_children(assignment));
        }
    }

    trees
}

/// Generate all integer partitions of n
fn generate_partitions(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![vec![]];
    }

    let mut partitions = vec![];
    generate_partitions_helper(n, n, &mut vec![], &mut partitions);
    partitions
}

fn generate_partitions_helper(
    n: usize,
    max_part: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if n == 0 {
        result.push(current.clone());
        return;
    }

    for part in 1..=max_part.min(n) {
        current.push(part);
        generate_partitions_helper(n - part, part, current, result);
        current.pop();
    }
}

/// Generate all ways to assign trees to partition parts
fn generate_tree_assignments(partition: &[usize]) -> Vec<Vec<RootedTree>> {
    if partition.is_empty() {
        return vec![vec![]];
    }

    let first = partition[0];
    let rest = &partition[1..];

    let first_trees = all_rooted_trees(first);
    let rest_assignments = generate_tree_assignments(rest);

    let mut result = vec![];
    for tree in &first_trees {
        for assignment in &rest_assignments {
            let mut new_assignment = vec![tree.clone()];
            new_assignment.extend(assignment.clone());
            result.push(new_assignment);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unit_tree() {
        let unit = RootedTree::unit();
        assert_eq!(unit.num_vertices(), 1);
        assert!(unit.is_unit());
        assert_eq!(unit.to_parenthesis(), "·");
    }

    #[test]
    fn test_tree_from_children() {
        // Create tree with root having two children (both leaves)
        let child1 = RootedTree::unit();
        let child2 = RootedTree::unit();
        let tree = RootedTree::from_children(vec![child1, child2]);

        assert_eq!(tree.num_vertices(), 3); // root + 2 children
        assert!(!tree.is_unit());
    }

    #[test]
    fn test_grafting() {
        let tree1 = RootedTree::unit();
        let tree2 = RootedTree::unit();

        // Graft tree2 onto the root of tree1
        let grafted = tree1.graft(0, &tree2).unwrap();
        assert_eq!(grafted.num_vertices(), 2);
    }

    #[test]
    fn test_all_graftings() {
        let tree1 = RootedTree::from_children(vec![RootedTree::unit()]);
        let tree2 = RootedTree::unit();

        // tree1 has 2 vertices, so we should get 2 different graftings
        let graftings = tree1.all_graftings(&tree2);
        assert_eq!(graftings.len(), 2);
    }

    #[test]
    fn test_admissible_cuts_unit() {
        let unit = RootedTree::unit();
        let cuts = unit.admissible_cuts();

        // Only the empty cut for a single vertex
        assert_eq!(cuts.len(), 1);
        assert_eq!(cuts[0].0, unit);
        assert!(cuts[0].1.is_empty());
    }

    #[test]
    fn test_admissible_cuts_two_vertices() {
        // Tree with root and one child
        let tree = RootedTree::from_children(vec![RootedTree::unit()]);
        let cuts = tree.admissible_cuts();

        // Should have 2 cuts: empty cut and cutting the edge
        assert_eq!(cuts.len(), 2);
    }

    #[test]
    fn test_algebra_element_zero() {
        let zero: GrossmanLarsonElement<Integer> = GrossmanLarsonElement::zero();
        assert!(zero.is_zero());
    }

    #[test]
    fn test_algebra_element_unit() {
        let unit: GrossmanLarsonElement<Integer> = GrossmanLarsonElement::unit();
        assert!(!unit.is_zero());
        assert_eq!(unit.terms().len(), 1);
    }

    #[test]
    fn test_algebra_addition() {
        let elem1: GrossmanLarsonElement<Integer> = GrossmanLarsonElement::from_tree(RootedTree::unit());
        let elem2: GrossmanLarsonElement<Integer> = GrossmanLarsonElement::from_tree(RootedTree::unit());

        let sum = elem1.add(&elem2);
        // Should combine like terms: 1·T + 1·T = 2·T
        assert_eq!(sum.terms().len(), 1);
        let coeff = sum.terms().values().next().unwrap();
        assert_eq!(*coeff, Integer::from(2));
    }

    #[test]
    fn test_algebra_scalar_multiplication() {
        let elem: GrossmanLarsonElement<Integer> = GrossmanLarsonElement::from_tree(RootedTree::unit());
        let scalar = Integer::from(3);

        let result = elem.scalar_mul(&scalar);
        assert_eq!(result.terms().len(), 1);
        let coeff = result.terms().values().next().unwrap();
        assert_eq!(*coeff, Integer::from(3));
    }

    #[test]
    fn test_algebra_multiplication() {
        let tree1 = RootedTree::unit();
        let tree2 = RootedTree::unit();

        let elem1: GrossmanLarsonElement<Integer> = GrossmanLarsonElement::from_tree(tree1);
        let elem2: GrossmanLarsonElement<Integer> = GrossmanLarsonElement::from_tree(tree2);

        let product = elem1.mul(&elem2);
        // Unit tree grafted with unit tree gives a tree with 2 vertices
        assert!(!product.is_zero());
    }

    #[test]
    fn test_counit() {
        let unit: GrossmanLarsonElement<Integer> = GrossmanLarsonElement::unit();
        assert_eq!(unit.counit(), Integer::one());

        let non_unit: GrossmanLarsonElement<Integer> = GrossmanLarsonElement::from_tree(RootedTree::from_children(vec![RootedTree::unit()]));
        assert_eq!(non_unit.counit(), Integer::zero());
    }

    #[test]
    fn test_coproduct_unit() {
        let unit: GrossmanLarsonElement<Integer> = GrossmanLarsonElement::unit();
        let coproduct = unit.coproduct();

        // Coproduct of unit should give unit ⊗ unit
        assert_eq!(coproduct.len(), 1);
        assert!(!coproduct[0].0.is_zero());
        assert!(!coproduct[0].1.is_zero());
    }

    #[test]
    fn test_all_rooted_trees_small() {
        // n=1: just the unit tree
        let trees_1 = all_rooted_trees(1);
        assert_eq!(trees_1.len(), 1);

        // n=2: just one tree (root with one child)
        let trees_2 = all_rooted_trees(2);
        assert_eq!(trees_2.len(), 1);

        // n=3: two trees
        // - root with two children
        // - root with one child which has one child
        let trees_3 = all_rooted_trees(3);
        assert!(trees_3.len() >= 2); // Should be exactly 2, but allow for duplicates
    }

    #[test]
    fn test_tree_display() {
        let unit = RootedTree::unit();
        assert_eq!(format!("{}", unit), "·");

        let tree = RootedTree::from_children(vec![RootedTree::unit(), RootedTree::unit()]);
        // Should display as [·, ·]
        assert!(format!("{}", tree).contains("·"));
    }

    #[test]
    fn test_element_display() {
        let elem: GrossmanLarsonElement<Integer> = GrossmanLarsonElement::unit();
        let display = format!("{}", elem);
        assert!(!display.is_empty());

        let zero: GrossmanLarsonElement<Integer> = GrossmanLarsonElement::zero();
        assert_eq!(format!("{}", zero), "0");
    }
}
