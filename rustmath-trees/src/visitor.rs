//! Visitor pattern for tree operations
//!
//! This module provides traits for implementing custom tree visitors and mutators.

use crate::{BinaryTree, NaryTree};

/// Trait for visiting tree nodes (read-only)
///
/// Implement this trait to perform custom analysis or computations on trees
/// without modifying them.
///
/// # Examples
///
/// ```
/// use rustmath_trees::{NaryTree, TreeVisitor};
///
/// struct SumVisitor {
///     sum: i32,
/// }
///
/// impl TreeVisitor<i32> for SumVisitor {
///     type Output = ();
///
///     fn visit_value(&mut self, value: &i32) {
///         self.sum += value;
///     }
/// }
///
/// let mut root = NaryTree::new(1);
/// root.add_child(NaryTree::new(2));
/// root.add_child(NaryTree::new(3));
///
/// let mut visitor = SumVisitor { sum: 0 };
/// visitor.visit_nary(&root);
/// assert_eq!(visitor.sum, 6);
/// ```
pub trait TreeVisitor<T> {
    /// Output type for visit operations
    type Output;

    /// Visit a single value
    ///
    /// This method is called for each node's value during traversal
    fn visit_value(&mut self, value: &T) -> Self::Output;

    /// Visit an n-ary tree node
    ///
    /// Default implementation visits the node's value, then recursively
    /// visits all children
    fn visit_nary(&mut self, node: &NaryTree<T>) -> Self::Output {
        let result = self.visit_value(&node.value);
        for child in node.children() {
            self.visit_nary(child);
        }
        result
    }

    /// Visit a binary tree node
    ///
    /// Default implementation visits the node's value, then recursively
    /// visits left and right children
    fn visit_binary(&mut self, node: &BinaryTree<T>) -> Self::Output {
        let result = self.visit_value(&node.value);
        if let Some(left) = node.left() {
            self.visit_binary(left);
        }
        if let Some(right) = node.right() {
            self.visit_binary(right);
        }
        result
    }
}

/// Trait for mutating tree nodes
///
/// Implement this trait to transform trees or modify their values in place.
///
/// # Examples
///
/// ```
/// use rustmath_trees::{NaryTree, TreeMutator};
///
/// struct DoubleVisitor;
///
/// impl TreeMutator<i32> for DoubleVisitor {
///     fn mutate_value(&mut self, value: &mut i32) {
///         *value *= 2;
///     }
/// }
///
/// let mut root = NaryTree::new(1);
/// root.add_child(NaryTree::new(2));
/// root.add_child(NaryTree::new(3));
///
/// let mut mutator = DoubleVisitor;
/// mutator.mutate_nary(&mut root);
/// assert_eq!(root.value(), &2);
/// assert_eq!(root.child(0).unwrap().value(), &4);
/// assert_eq!(root.child(1).unwrap().value(), &6);
/// ```
pub trait TreeMutator<T> {
    /// Mutate a single value
    ///
    /// This method is called for each node's value during traversal
    fn mutate_value(&mut self, value: &mut T);

    /// Mutate an n-ary tree node
    ///
    /// Default implementation mutates the node's value, then recursively
    /// mutates all children
    fn mutate_nary(&mut self, node: &mut NaryTree<T>) {
        self.mutate_value(&mut node.value);
        for child in node.children_mut() {
            self.mutate_nary(child);
        }
    }

    /// Mutate a binary tree node
    ///
    /// Default implementation mutates the node's value, then recursively
    /// mutates left and right children
    fn mutate_binary(&mut self, node: &mut BinaryTree<T>) {
        self.mutate_value(&mut node.value);
        if let Some(left) = node.left_mut() {
            self.mutate_binary(left);
        }
        if let Some(right) = node.right_mut() {
            self.mutate_binary(right);
        }
    }
}

// ============================================================================
// Concrete Visitor Implementations
// ============================================================================

/// Counts the total number of nodes in a tree
pub struct NodeCounter {
    count: usize,
}

impl NodeCounter {
    /// Create a new node counter
    pub fn new() -> Self {
        NodeCounter { count: 0 }
    }

    /// Get the current count
    pub fn count(&self) -> usize {
        self.count
    }
}

impl Default for NodeCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> TreeVisitor<T> for NodeCounter {
    type Output = ();

    fn visit_value(&mut self, _value: &T) {
        self.count += 1;
    }
}

/// Collects all values from a tree into a vector
pub struct ValueCollector<T> {
    values: Vec<T>,
}

impl<T> ValueCollector<T> {
    /// Create a new value collector
    pub fn new() -> Self {
        ValueCollector { values: Vec::new() }
    }

    /// Get the collected values
    pub fn values(self) -> Vec<T> {
        self.values
    }

    /// Get a reference to the collected values
    pub fn values_ref(&self) -> &[T] {
        &self.values
    }
}

impl<T> Default for ValueCollector<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> TreeVisitor<T> for ValueCollector<T> {
    type Output = ();

    fn visit_value(&mut self, value: &T) {
        self.values.push(value.clone());
    }
}

/// Finds the maximum value in a tree
pub struct MaxFinder<T> {
    max: Option<T>,
}

impl<T> MaxFinder<T> {
    /// Create a new max finder
    pub fn new() -> Self {
        MaxFinder { max: None }
    }

    /// Get the maximum value found
    pub fn max(&self) -> Option<&T> {
        self.max.as_ref()
    }

    /// Take the maximum value
    pub fn take_max(self) -> Option<T> {
        self.max
    }
}

impl<T> Default for MaxFinder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Ord + Clone> TreeVisitor<T> for MaxFinder<T> {
    type Output = ();

    fn visit_value(&mut self, value: &T) {
        match &self.max {
            None => self.max = Some(value.clone()),
            Some(current_max) => {
                if value > current_max {
                    self.max = Some(value.clone());
                }
            }
        }
    }
}

/// Finds the minimum value in a tree
pub struct MinFinder<T> {
    min: Option<T>,
}

impl<T> MinFinder<T> {
    /// Create a new min finder
    pub fn new() -> Self {
        MinFinder { min: None }
    }

    /// Get the minimum value found
    pub fn min(&self) -> Option<&T> {
        self.min.as_ref()
    }

    /// Take the minimum value
    pub fn take_min(self) -> Option<T> {
        self.min
    }
}

impl<T> Default for MinFinder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Ord + Clone> TreeVisitor<T> for MinFinder<T> {
    type Output = ();

    fn visit_value(&mut self, value: &T) {
        match &self.min {
            None => self.min = Some(value.clone()),
            Some(current_min) => {
                if value < current_min {
                    self.min = Some(value.clone());
                }
            }
        }
    }
}

/// Searches for a specific value in the tree
pub struct ValueSearcher<T> {
    target: T,
    found: bool,
}

impl<T> ValueSearcher<T> {
    /// Create a new value searcher
    pub fn new(target: T) -> Self {
        ValueSearcher {
            target,
            found: false,
        }
    }

    /// Check if the value was found
    pub fn found(&self) -> bool {
        self.found
    }
}

impl<T: PartialEq> TreeVisitor<T> for ValueSearcher<T> {
    type Output = ();

    fn visit_value(&mut self, value: &T) {
        if value == &self.target {
            self.found = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tree;

    #[test]
    fn test_node_counter() {
        let mut root = NaryTree::new(1);
        root.add_child(NaryTree::new(2));
        root.add_child(NaryTree::new(3));

        let mut counter = NodeCounter::new();
        counter.visit_nary(&root);

        assert_eq!(counter.count(), 3);
    }

    #[test]
    fn test_value_collector() {
        let mut root = NaryTree::new(1);
        root.add_child(NaryTree::new(2));
        root.add_child(NaryTree::new(3));

        let mut collector = ValueCollector::new();
        collector.visit_nary(&root);

        let values = collector.values();
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn test_max_finder() {
        let mut root = NaryTree::new(5);
        root.add_child(NaryTree::new(10));
        root.add_child(NaryTree::new(3));

        let mut finder = MaxFinder::new();
        finder.visit_nary(&root);

        assert_eq!(finder.max(), Some(&10));
    }

    #[test]
    fn test_min_finder() {
        let mut root = NaryTree::new(5);
        root.add_child(NaryTree::new(10));
        root.add_child(NaryTree::new(3));

        let mut finder = MinFinder::new();
        finder.visit_nary(&root);

        assert_eq!(finder.min(), Some(&3));
    }

    #[test]
    fn test_value_searcher() {
        let mut root = NaryTree::new(1);
        root.add_child(NaryTree::new(2));
        root.add_child(NaryTree::new(3));

        let mut searcher = ValueSearcher::new(2);
        searcher.visit_nary(&root);
        assert!(searcher.found());

        let mut searcher2 = ValueSearcher::new(99);
        searcher2.visit_nary(&root);
        assert!(!searcher2.found());
    }

    #[test]
    fn test_tree_mutator() {
        struct DoubleVisitor;

        impl TreeMutator<i32> for DoubleVisitor {
            fn mutate_value(&mut self, value: &mut i32) {
                *value *= 2;
            }
        }

        let mut root = NaryTree::new(1);
        root.add_child(NaryTree::new(2));
        root.add_child(NaryTree::new(3));

        let mut mutator = DoubleVisitor;
        mutator.mutate_nary(&mut root);

        assert_eq!(root.value(), &2);
        assert_eq!(root.child(0).unwrap().value(), &4);
        assert_eq!(root.child(1).unwrap().value(), &6);
    }

    #[test]
    fn test_binary_tree_visitor() {
        let mut root = BinaryTree::new(1);
        root.set_left(BinaryTree::new(2));
        root.set_right(BinaryTree::new(3));

        let mut counter = NodeCounter::new();
        counter.visit_binary(&root);

        assert_eq!(counter.count(), 3);
    }

    #[test]
    fn test_binary_tree_mutator() {
        struct IncrementVisitor;

        impl TreeMutator<i32> for IncrementVisitor {
            fn mutate_value(&mut self, value: &mut i32) {
                *value += 1;
            }
        }

        let mut root = BinaryTree::new(1);
        root.set_left(BinaryTree::new(2));
        root.set_right(BinaryTree::new(3));

        let mut mutator = IncrementVisitor;
        mutator.mutate_binary(&mut root);

        assert_eq!(root.value(), &2);
        assert_eq!(root.left().unwrap().value(), &3);
        assert_eq!(root.right().unwrap().value(), &4);
    }
}
