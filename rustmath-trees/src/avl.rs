//! AVL tree implementation
//!
//! This module provides an AVL (Adelson-Velsky and Landis) tree, which is a self-balancing
//! binary search tree. The heights of the two child subtrees of any node differ by at most one.
//! If at any time they differ by more than one, rebalancing is done via rotations.
//!
//! # Examples
//!
//! ```
//! use rustmath_trees::AvlTree;
//!
//! let mut tree = AvlTree::new();
//! tree.insert(5);
//! tree.insert(3);
//! tree.insert(7);
//!
//! assert!(tree.search(&5));
//! assert_eq!(tree.height(), 2);
//! ```

use std::cmp::{max, Ordering};
use std::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A node in an AVL tree
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
struct AvlNode<T> {
    value: T,
    height: i32,
    left: Option<Box<AvlNode<T>>>,
    right: Option<Box<AvlNode<T>>>,
}

impl<T> AvlNode<T> {
    fn new(value: T) -> Self {
        AvlNode {
            value,
            height: 1,
            left: None,
            right: None,
        }
    }

    fn update_height(&mut self) {
        let left_height = self.left.as_ref().map_or(0, |n| n.height);
        let right_height = self.right.as_ref().map_or(0, |n| n.height);
        self.height = 1 + max(left_height, right_height);
    }

    fn balance_factor(&self) -> i32 {
        let left_height = self.left.as_ref().map_or(0, |n| n.height);
        let right_height = self.right.as_ref().map_or(0, |n| n.height);
        left_height - right_height
    }

    fn rotate_right(mut self: Box<Self>) -> Box<Self> {
        let mut new_root = self.left.take().unwrap();
        self.left = new_root.right.take();
        self.update_height();
        new_root.right = Some(self);
        new_root.update_height();
        new_root
    }

    fn rotate_left(mut self: Box<Self>) -> Box<Self> {
        let mut new_root = self.right.take().unwrap();
        self.right = new_root.left.take();
        self.update_height();
        new_root.left = Some(self);
        new_root.update_height();
        new_root
    }

    fn rebalance(mut self: Box<Self>) -> Box<Self> {
        self.update_height();
        let balance = self.balance_factor();

        // Left heavy
        if balance > 1 {
            if let Some(ref left) = self.left {
                if left.balance_factor() < 0 {
                    // Left-Right case
                    let left = self.left.take().unwrap();
                    self.left = Some(left.rotate_left());
                }
            }
            // Left-Left case
            return self.rotate_right();
        }

        // Right heavy
        if balance < -1 {
            if let Some(ref right) = self.right {
                if right.balance_factor() > 0 {
                    // Right-Left case
                    let right = self.right.take().unwrap();
                    self.right = Some(right.rotate_right());
                }
            }
            // Right-Right case
            return self.rotate_left();
        }

        self
    }

    fn insert(node: Option<Box<Self>>, value: T) -> Box<Self>
    where
        T: Ord,
    {
        match node {
            None => Box::new(AvlNode::new(value)),
            Some(mut n) => {
                match value.cmp(&n.value) {
                    Ordering::Less => {
                        n.left = Some(Self::insert(n.left.take(), value));
                    }
                    Ordering::Greater => {
                        n.right = Some(Self::insert(n.right.take(), value));
                    }
                    Ordering::Equal => {
                        // Duplicate value, don't insert
                        return n;
                    }
                }
                n.rebalance()
            }
        }
    }

    fn find_min(&self) -> &T {
        match &self.left {
            None => &self.value,
            Some(left) => left.find_min(),
        }
    }

    fn delete(node: Option<Box<Self>>, value: &T) -> Option<Box<Self>>
    where
        T: Ord + Clone,
    {
        let mut n = match node {
            None => return None,
            Some(n) => n,
        };

        match value.cmp(&n.value) {
            Ordering::Less => {
                n.left = Self::delete(n.left.take(), value);
            }
            Ordering::Greater => {
                n.right = Self::delete(n.right.take(), value);
            }
            Ordering::Equal => {
                // Node to be deleted found
                match (n.left.take(), n.right.take()) {
                    (None, None) => return None,
                    (Some(left), None) => return Some(left),
                    (None, Some(right)) => return Some(right),
                    (Some(left), Some(right)) => {
                        // Node has two children
                        // Find the minimum value in the right subtree and clone it
                        let successor_value = (*right.find_min()).clone();
                        n.value = successor_value.clone();
                        n.left = Some(left);
                        n.right = Self::delete(Some(right), &successor_value);
                    }
                }
            }
        }

        Some(n.rebalance())
    }
}

/// An AVL tree - a self-balancing binary search tree
///
/// The AVL tree maintains balance by ensuring that for any node, the heights of its
/// left and right subtrees differ by at most 1. This guarantees O(log n) time complexity
/// for search, insertion, and deletion operations.
///
/// # Examples
///
/// ```
/// use rustmath_trees::AvlTree;
///
/// let mut tree = AvlTree::new();
/// tree.insert(10);
/// tree.insert(5);
/// tree.insert(15);
/// tree.insert(3);
/// tree.insert(7);
///
/// assert!(tree.search(&7));
/// assert!(!tree.search(&20));
/// assert_eq!(tree.len(), 5);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AvlTree<T> {
    root: Option<Box<AvlNode<T>>>,
    size: usize,
}

impl<T> AvlTree<T> {
    /// Create a new empty AVL tree
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_trees::AvlTree;
    ///
    /// let tree: AvlTree<i32> = AvlTree::new();
    /// assert!(tree.is_empty());
    /// ```
    pub fn new() -> Self {
        AvlTree {
            root: None,
            size: 0,
        }
    }

    /// Check if the tree is empty
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// Get the number of nodes in the tree
    pub fn len(&self) -> usize {
        self.size
    }

    /// Get the height of the tree
    ///
    /// The height is the number of edges on the longest path from the root to a leaf.
    /// An empty tree has height 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_trees::AvlTree;
    ///
    /// let mut tree = AvlTree::new();
    /// assert_eq!(tree.height(), 0);
    ///
    /// tree.insert(5);
    /// assert_eq!(tree.height(), 1);
    ///
    /// tree.insert(3);
    /// tree.insert(7);
    /// assert_eq!(tree.height(), 2);
    /// ```
    pub fn height(&self) -> usize {
        self.root.as_ref().map_or(0, |n| n.height as usize)
    }

    /// Insert a value into the AVL tree
    ///
    /// The tree automatically rebalances after insertion to maintain the AVL property.
    /// Duplicate values are not inserted.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_trees::AvlTree;
    ///
    /// let mut tree = AvlTree::new();
    /// tree.insert(5);
    /// tree.insert(3);
    /// tree.insert(7);
    /// assert_eq!(tree.len(), 3);
    /// ```
    pub fn insert(&mut self, value: T)
    where
        T: Ord,
    {
        let old_size = self.size;
        self.root = Some(AvlNode::insert(self.root.take(), value));
        // Only increment size if a new node was actually inserted
        if self.root.is_some() {
            self.size = old_size + 1;
        }
    }

    /// Search for a value in the tree
    ///
    /// Returns true if the value is found, false otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_trees::AvlTree;
    ///
    /// let mut tree = AvlTree::new();
    /// tree.insert(5);
    /// tree.insert(3);
    ///
    /// assert!(tree.search(&5));
    /// assert!(tree.search(&3));
    /// assert!(!tree.search(&7));
    /// ```
    pub fn search(&self, value: &T) -> bool
    where
        T: Ord,
    {
        let mut current = &self.root;
        while let Some(node) = current {
            match value.cmp(&node.value) {
                Ordering::Equal => return true,
                Ordering::Less => current = &node.left,
                Ordering::Greater => current = &node.right,
            }
        }
        false
    }

    /// Delete a value from the AVL tree
    ///
    /// The tree automatically rebalances after deletion to maintain the AVL property.
    /// Returns true if the value was found and deleted, false otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_trees::AvlTree;
    ///
    /// let mut tree = AvlTree::new();
    /// tree.insert(5);
    /// tree.insert(3);
    /// tree.insert(7);
    ///
    /// assert!(tree.delete(&3));
    /// assert_eq!(tree.len(), 2);
    /// assert!(!tree.search(&3));
    /// ```
    pub fn delete(&mut self, value: &T) -> bool
    where
        T: Ord + Clone,
    {
        if !self.search(value) {
            return false;
        }

        self.root = AvlNode::delete(self.root.take(), value);
        self.size -= 1;
        true
    }

    /// Get the minimum value in the tree
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_trees::AvlTree;
    ///
    /// let mut tree = AvlTree::new();
    /// tree.insert(5);
    /// tree.insert(3);
    /// tree.insert(7);
    ///
    /// assert_eq!(tree.min(), Some(&3));
    /// ```
    pub fn min(&self) -> Option<&T> {
        self.root.as_ref().map(|n| n.find_min())
    }

    /// Get the maximum value in the tree
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_trees::AvlTree;
    ///
    /// let mut tree = AvlTree::new();
    /// tree.insert(5);
    /// tree.insert(3);
    /// tree.insert(7);
    ///
    /// assert_eq!(tree.max(), Some(&7));
    /// ```
    pub fn max(&self) -> Option<&T> {
        let mut current = self.root.as_ref()?;
        while let Some(ref right) = current.right {
            current = right;
        }
        Some(&current.value)
    }

    /// Perform an inorder traversal of the tree
    ///
    /// Returns a vector of references to the values in sorted order.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_trees::AvlTree;
    ///
    /// let mut tree = AvlTree::new();
    /// tree.insert(5);
    /// tree.insert(3);
    /// tree.insert(7);
    /// tree.insert(1);
    /// tree.insert(9);
    ///
    /// let values = tree.inorder();
    /// assert_eq!(values, vec![&1, &3, &5, &7, &9]);
    /// ```
    pub fn inorder(&self) -> Vec<&T> {
        let mut result = Vec::new();
        self.inorder_helper(&self.root, &mut result);
        result
    }

    fn inorder_helper<'a>(&'a self, node: &'a Option<Box<AvlNode<T>>>, result: &mut Vec<&'a T>) {
        if let Some(n) = node {
            self.inorder_helper(&n.left, result);
            result.push(&n.value);
            self.inorder_helper(&n.right, result);
        }
    }

    /// Check if the tree satisfies the AVL balance property
    ///
    /// Returns true if all nodes have a balance factor between -1 and 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_trees::AvlTree;
    ///
    /// let mut tree = AvlTree::new();
    /// tree.insert(5);
    /// tree.insert(3);
    /// tree.insert(7);
    ///
    /// assert!(tree.is_balanced());
    /// ```
    pub fn is_balanced(&self) -> bool {
        self.check_balanced(&self.root)
    }

    fn check_balanced(&self, node: &Option<Box<AvlNode<T>>>) -> bool {
        match node {
            None => true,
            Some(n) => {
                let balance = n.balance_factor();
                balance >= -1 && balance <= 1 && self.check_balanced(&n.left) && self.check_balanced(&n.right)
            }
        }
    }
}

impl<T> Default for AvlTree<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: fmt::Display> fmt::Display for AvlTree<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.root.is_none() {
            return write!(f, "Empty AVL tree");
        }
        self.fmt_helper(&self.root, f, "", "")
    }
}

impl<T: fmt::Display> AvlTree<T> {
    fn fmt_helper(
        &self,
        node: &Option<Box<AvlNode<T>>>,
        f: &mut fmt::Formatter<'_>,
        prefix: &str,
        child_prefix: &str,
    ) -> fmt::Result {
        if let Some(n) = node {
            writeln!(f, "{}{} (h:{})", prefix, n.value, n.height)?;

            let has_right = n.right.is_some();
            let has_left = n.left.is_some();

            if has_left {
                let new_prefix = format!("{}├── ", child_prefix);
                let new_child_prefix = if has_right {
                    format!("{}│   ", child_prefix)
                } else {
                    format!("{}    ", child_prefix)
                };
                self.fmt_helper(&n.left, f, &new_prefix, &new_child_prefix)?;
            }

            if has_right {
                let new_prefix = format!("{}└── ", child_prefix);
                let new_child_prefix = format!("{}    ", child_prefix);
                self.fmt_helper(&n.right, f, &new_prefix, &new_child_prefix)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let tree: AvlTree<i32> = AvlTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert_eq!(tree.height(), 0);
    }

    #[test]
    fn test_insert_single() {
        let mut tree = AvlTree::new();
        tree.insert(5);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.height(), 1);
        assert!(!tree.is_empty());
    }

    #[test]
    fn test_insert_balanced() {
        let mut tree = AvlTree::new();
        tree.insert(5);
        tree.insert(3);
        tree.insert(7);
        tree.insert(2);
        tree.insert(4);
        tree.insert(6);
        tree.insert(8);

        assert_eq!(tree.len(), 7);
        assert!(tree.is_balanced());
        assert_eq!(tree.height(), 3);
    }

    #[test]
    fn test_insert_right_rotation() {
        let mut tree = AvlTree::new();
        tree.insert(3);
        tree.insert(2);
        tree.insert(1);

        assert!(tree.is_balanced());
        assert_eq!(tree.inorder(), vec![&1, &2, &3]);
    }

    #[test]
    fn test_insert_left_rotation() {
        let mut tree = AvlTree::new();
        tree.insert(1);
        tree.insert(2);
        tree.insert(3);

        assert!(tree.is_balanced());
        assert_eq!(tree.inorder(), vec![&1, &2, &3]);
    }

    #[test]
    fn test_insert_left_right_rotation() {
        let mut tree = AvlTree::new();
        tree.insert(3);
        tree.insert(1);
        tree.insert(2);

        assert!(tree.is_balanced());
        assert_eq!(tree.inorder(), vec![&1, &2, &3]);
    }

    #[test]
    fn test_insert_right_left_rotation() {
        let mut tree = AvlTree::new();
        tree.insert(1);
        tree.insert(3);
        tree.insert(2);

        assert!(tree.is_balanced());
        assert_eq!(tree.inorder(), vec![&1, &2, &3]);
    }

    #[test]
    fn test_search() {
        let mut tree = AvlTree::new();
        tree.insert(5);
        tree.insert(3);
        tree.insert(7);
        tree.insert(1);
        tree.insert(9);

        assert!(tree.search(&5));
        assert!(tree.search(&3));
        assert!(tree.search(&7));
        assert!(tree.search(&1));
        assert!(tree.search(&9));
        assert!(!tree.search(&2));
        assert!(!tree.search(&10));
    }

    #[test]
    fn test_delete_leaf() {
        let mut tree = AvlTree::new();
        tree.insert(5);
        tree.insert(3);
        tree.insert(7);

        assert!(tree.delete(&3));
        assert_eq!(tree.len(), 2);
        assert!(!tree.search(&3));
        assert!(tree.is_balanced());
    }

    #[test]
    fn test_delete_one_child() {
        let mut tree = AvlTree::new();
        tree.insert(5);
        tree.insert(3);
        tree.insert(7);
        tree.insert(1);

        assert!(tree.delete(&3));
        assert_eq!(tree.len(), 3);
        assert!(!tree.search(&3));
        assert!(tree.search(&1));
        assert!(tree.is_balanced());
    }

    #[test]
    fn test_delete_two_children() {
        let mut tree = AvlTree::new();
        tree.insert(5);
        tree.insert(3);
        tree.insert(7);
        tree.insert(1);
        tree.insert(4);

        assert!(tree.delete(&3));
        assert_eq!(tree.len(), 4);
        assert!(!tree.search(&3));
        assert!(tree.is_balanced());
    }

    #[test]
    fn test_delete_root() {
        let mut tree = AvlTree::new();
        tree.insert(5);
        tree.insert(3);
        tree.insert(7);

        assert!(tree.delete(&5));
        assert_eq!(tree.len(), 2);
        assert!(!tree.search(&5));
        assert!(tree.is_balanced());
    }

    #[test]
    fn test_delete_nonexistent() {
        let mut tree = AvlTree::new();
        tree.insert(5);
        tree.insert(3);

        assert!(!tree.delete(&7));
        assert_eq!(tree.len(), 2);
    }

    #[test]
    fn test_delete_maintains_balance() {
        let mut tree = AvlTree::new();
        for i in 1..=10 {
            tree.insert(i);
        }

        tree.delete(&5);
        tree.delete(&6);
        tree.delete(&7);

        assert!(tree.is_balanced());
        assert_eq!(tree.len(), 7);
    }

    #[test]
    fn test_min_max() {
        let mut tree = AvlTree::new();
        tree.insert(5);
        tree.insert(3);
        tree.insert(7);
        tree.insert(1);
        tree.insert(9);

        assert_eq!(tree.min(), Some(&1));
        assert_eq!(tree.max(), Some(&9));
    }

    #[test]
    fn test_min_max_empty() {
        let tree: AvlTree<i32> = AvlTree::new();
        assert_eq!(tree.min(), None);
        assert_eq!(tree.max(), None);
    }

    #[test]
    fn test_inorder_traversal() {
        let mut tree = AvlTree::new();
        tree.insert(5);
        tree.insert(3);
        tree.insert(7);
        tree.insert(1);
        tree.insert(9);
        tree.insert(4);
        tree.insert(6);

        let values = tree.inorder();
        assert_eq!(values, vec![&1, &3, &4, &5, &6, &7, &9]);
    }

    #[test]
    fn test_clone() {
        let mut tree = AvlTree::new();
        tree.insert(5);
        tree.insert(3);
        tree.insert(7);

        let cloned = tree.clone();
        assert_eq!(cloned.len(), tree.len());
        assert_eq!(cloned.inorder(), tree.inorder());
    }

    #[test]
    fn test_large_tree() {
        let mut tree = AvlTree::new();
        for i in 1..=100 {
            tree.insert(i);
        }

        assert_eq!(tree.len(), 100);
        assert!(tree.is_balanced());
        assert!(tree.height() <= 7); // log2(100) ~ 6.64

        for i in 1..=100 {
            assert!(tree.search(&i));
        }
    }
}
