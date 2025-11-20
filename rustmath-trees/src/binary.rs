//! Binary tree implementation
//!
//! This module provides a binary tree where each node has at most two children (left and right).

use crate::traits::{Tree, TreeNode};
use std::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A binary tree where each node has at most a left and right child
///
/// # Examples
///
/// ```
/// use rustmath_trees::BinaryTree;
///
/// let mut root = BinaryTree::new(1);
/// root.set_left(BinaryTree::new(2));
/// root.set_right(BinaryTree::new(3));
///
/// assert_eq!(root.value(), &1);
/// assert!(root.left().is_some());
/// assert!(root.right().is_some());
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BinaryTree<T> {
    pub(crate) value: T,
    pub(crate) left: Option<Box<BinaryTree<T>>>,
    pub(crate) right: Option<Box<BinaryTree<T>>>,
}

impl<T> BinaryTree<T> {
    /// Create a new binary tree node with the given value
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_trees::BinaryTree;
    ///
    /// let node = BinaryTree::new(42);
    /// assert_eq!(node.value(), &42);
    /// assert!(node.is_leaf());
    /// ```
    pub fn new(value: T) -> Self {
        BinaryTree {
            value,
            left: None,
            right: None,
        }
    }

    /// Create a new binary tree with the given value, left child, and right child
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_trees::BinaryTree;
    ///
    /// let left = BinaryTree::new(2);
    /// let right = BinaryTree::new(3);
    /// let root = BinaryTree::with_children(1, Some(left), Some(right));
    ///
    /// assert_eq!(root.value(), &1);
    /// assert!(root.left().is_some());
    /// assert!(root.right().is_some());
    /// ```
    pub fn with_children(
        value: T,
        left: Option<BinaryTree<T>>,
        right: Option<BinaryTree<T>>,
    ) -> Self {
        BinaryTree {
            value,
            left: left.map(Box::new),
            right: right.map(Box::new),
        }
    }

    /// Set the left child
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_trees::BinaryTree;
    ///
    /// let mut root = BinaryTree::new(1);
    /// root.set_left(BinaryTree::new(2));
    /// assert!(root.left().is_some());
    /// ```
    pub fn set_left(&mut self, child: BinaryTree<T>) {
        self.left = Some(Box::new(child));
    }

    /// Set the right child
    pub fn set_right(&mut self, child: BinaryTree<T>) {
        self.right = Some(Box::new(child));
    }

    /// Get a reference to the left child
    pub fn left(&self) -> Option<&BinaryTree<T>> {
        self.left.as_deref()
    }

    /// Get a reference to the right child
    pub fn right(&self) -> Option<&BinaryTree<T>> {
        self.right.as_deref()
    }

    /// Get a mutable reference to the left child
    pub fn left_mut(&mut self) -> Option<&mut BinaryTree<T>> {
        self.left.as_deref_mut()
    }

    /// Get a mutable reference to the right child
    pub fn right_mut(&mut self) -> Option<&mut BinaryTree<T>> {
        self.right.as_deref_mut()
    }

    /// Take the left child, leaving None in its place
    pub fn take_left(&mut self) -> Option<BinaryTree<T>> {
        self.left.take().map(|boxed| *boxed)
    }

    /// Take the right child, leaving None in its place
    pub fn take_right(&mut self) -> Option<BinaryTree<T>> {
        self.right.take().map(|boxed| *boxed)
    }

    /// Check if this node has a left child
    pub fn has_left(&self) -> bool {
        self.left.is_some()
    }

    /// Check if this node has a right child
    pub fn has_right(&self) -> bool {
        self.right.is_some()
    }

    /// Map a function over all values in the tree
    ///
    /// Creates a new tree with the function applied to each value
    pub fn map<U, F>(self, f: &F) -> BinaryTree<U>
    where
        F: Fn(T) -> U,
    {
        BinaryTree {
            value: f(self.value),
            left: self.left.map(|child| Box::new((*child).map(f))),
            right: self.right.map(|child| Box::new((*child).map(f))),
        }
    }

    /// Apply a function to each value in the tree (in-place)
    pub fn map_mut<F>(&mut self, f: &F)
    where
        F: Fn(&mut T),
    {
        f(&mut self.value);
        if let Some(left) = &mut self.left {
            left.map_mut(f);
        }
        if let Some(right) = &mut self.right {
            right.map_mut(f);
        }
    }

    /// Find a node with a specific value (depth-first search)
    ///
    /// Returns a reference to the first node found with the given value
    pub fn find(&self, target: &T) -> Option<&BinaryTree<T>>
    where
        T: PartialEq,
    {
        if &self.value == target {
            return Some(self);
        }

        if let Some(left) = &self.left {
            if let Some(found) = left.find(target) {
                return Some(found);
            }
        }

        if let Some(right) = &self.right {
            if let Some(found) = right.find(target) {
                return Some(found);
            }
        }

        None
    }

    /// Get all leaf nodes
    pub fn leaves(&self) -> Vec<&T>
    where
        T: Clone + fmt::Debug,
    {
        if self.is_leaf() {
            vec![&self.value]
        } else {
            let mut result = Vec::new();
            if let Some(left) = &self.left {
                result.extend(left.leaves());
            }
            if let Some(right) = &self.right {
                result.extend(right.leaves());
            }
            result
        }
    }

    /// Check if the tree is balanced
    ///
    /// A binary tree is balanced if the heights of the two subtrees of any node
    /// never differ by more than one.
    pub fn is_balanced(&self) -> bool {
        self.check_balance().is_some()
    }

    fn check_balance(&self) -> Option<usize> {
        let left_height = if let Some(left) = &self.left {
            left.check_balance()?
        } else {
            0
        };

        let right_height = if let Some(right) = &self.right {
            right.check_balance()?
        } else {
            0
        };

        if left_height.abs_diff(right_height) > 1 {
            None
        } else {
            Some(1 + left_height.max(right_height))
        }
    }

    /// Check if the tree is a complete binary tree
    ///
    /// A complete binary tree is a binary tree in which every level, except
    /// possibly the last, is completely filled, and all nodes are as far left
    /// as possible.
    pub fn is_complete(&self) -> bool
    where
        T: Clone + fmt::Debug,
    {
        self.is_complete_helper(0, self.size())
    }

    fn is_complete_helper(&self, index: usize, node_count: usize) -> bool
    where
        T: Clone + fmt::Debug,
    {
        if index >= node_count {
            return false;
        }

        if self.is_leaf() {
            return true;
        }

        let left_ok = self
            .left
            .as_ref()
            .map(|l| l.is_complete_helper(2 * index + 1, node_count))
            .unwrap_or(true);

        let right_ok = self
            .right
            .as_ref()
            .map(|r| r.is_complete_helper(2 * index + 2, node_count))
            .unwrap_or(true);

        left_ok && right_ok
    }

    /// Check if the tree is a full binary tree
    ///
    /// A full binary tree is a tree in which every node has either 0 or 2 children.
    pub fn is_full(&self) -> bool {
        match (&self.left, &self.right) {
            (None, None) => true,
            (Some(left), Some(right)) => left.is_full() && right.is_full(),
            _ => false,
        }
    }

    /// Check if the tree is a perfect binary tree
    ///
    /// A perfect binary tree is a binary tree in which all interior nodes have
    /// two children and all leaves have the same depth.
    pub fn is_perfect(&self) -> bool
    where
        T: Clone + fmt::Debug,
    {
        let height = self.height();
        self.is_perfect_helper(height, 0)
    }

    fn is_perfect_helper(&self, height: usize, level: usize) -> bool
    where
        T: Clone + fmt::Debug,
    {
        if self.is_leaf() {
            return level == height;
        }

        match (&self.left, &self.right) {
            (Some(left), Some(right)) => {
                left.is_perfect_helper(height, level + 1)
                    && right.is_perfect_helper(height, level + 1)
            }
            _ => false,
        }
    }
}

impl<T: Clone + fmt::Debug> Tree for BinaryTree<T> {
    type Value = T;

    fn value(&self) -> &Self::Value {
        &self.value
    }

    fn value_mut(&mut self) -> &mut Self::Value {
        &mut self.value
    }

    fn height(&self) -> usize {
        let left_height = self.left.as_ref().map(|l| l.height()).unwrap_or(0);
        let right_height = self.right.as_ref().map(|r| r.height()).unwrap_or(0);

        if left_height == 0 && right_height == 0 {
            0
        } else {
            1 + left_height.max(right_height)
        }
    }

    fn size(&self) -> usize {
        let left_size = self.left.as_ref().map(|l| l.size()).unwrap_or(0);
        let right_size = self.right.as_ref().map(|r| r.size()).unwrap_or(0);
        1 + left_size + right_size
    }

    fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }

    fn depth_to(&self, descendant: &Self) -> Option<usize>
    where
        Self::Value: PartialEq,
    {
        if self == descendant {
            return Some(0);
        }

        if let Some(left) = &self.left {
            if let Some(depth) = left.depth_to(descendant) {
                return Some(depth + 1);
            }
        }

        if let Some(right) = &self.right {
            if let Some(depth) = right.depth_to(descendant) {
                return Some(depth + 1);
            }
        }

        None
    }
}

impl<T: Clone + fmt::Debug> TreeNode for BinaryTree<T> {
    fn num_children(&self) -> usize {
        let mut count = 0;
        if self.left.is_some() {
            count += 1;
        }
        if self.right.is_some() {
            count += 1;
        }
        count
    }
}

impl<T: fmt::Display> fmt::Display for BinaryTree<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_helper_with_prefix(f, "", "", "")
    }
}

impl<T: fmt::Display> BinaryTree<T> {
    fn fmt_helper_with_prefix(
        &self,
        f: &mut fmt::Formatter<'_>,
        prefix: &str,
        child_prefix: &str,
        label: &str,
    ) -> fmt::Result {
        writeln!(f, "{}{}{}", prefix, label, self.value)?;

        let has_right = self.right.is_some();

        if let Some(left) = &self.left {
            let new_prefix = format!("{}├── ", child_prefix);
            let new_child_prefix = if has_right {
                format!("{}│   ", child_prefix)
            } else {
                format!("{}    ", child_prefix)
            };
            left.fmt_helper_with_prefix(f, &new_prefix, &new_child_prefix, "L: ")?;
        }

        if let Some(right) = &self.right {
            let new_prefix = format!("{}└── ", child_prefix);
            let new_child_prefix = format!("{}    ", child_prefix);
            right.fmt_helper_with_prefix(f, &new_prefix, &new_child_prefix, "R: ")?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let tree: BinaryTree<i32> = BinaryTree::new(42);
        assert_eq!(tree.value(), &42);
        assert!(tree.is_leaf());
        assert_eq!(tree.height(), 0);
        assert_eq!(tree.size(), 1);
    }

    #[test]
    fn test_set_children() {
        let mut root = BinaryTree::new(1);
        root.set_left(BinaryTree::new(2));
        root.set_right(BinaryTree::new(3));

        assert!(root.has_left());
        assert!(root.has_right());
        assert_eq!(root.num_children(), 2);
        assert_eq!(root.height(), 1);
        assert_eq!(root.size(), 3);
    }

    #[test]
    fn test_with_children() {
        let left = BinaryTree::new(2);
        let right = BinaryTree::new(3);
        let root = BinaryTree::with_children(1, Some(left), Some(right));

        assert_eq!(root.value(), &1);
        assert!(root.has_left());
        assert!(root.has_right());
    }

    #[test]
    fn test_height() {
        let mut root = BinaryTree::new(1);
        let mut left = BinaryTree::new(2);
        left.set_left(BinaryTree::new(4));
        root.set_left(left);
        root.set_right(BinaryTree::new(3));

        assert_eq!(root.height(), 2);
    }

    #[test]
    fn test_size() {
        let mut root = BinaryTree::new(1);
        let mut left = BinaryTree::new(2);
        left.set_left(BinaryTree::new(4));
        left.set_right(BinaryTree::new(5));
        root.set_left(left);
        root.set_right(BinaryTree::new(3));

        assert_eq!(root.size(), 5);
    }

    #[test]
    fn test_map() {
        let mut root = BinaryTree::new(1);
        root.set_left(BinaryTree::new(2));
        root.set_right(BinaryTree::new(3));

        let doubled = root.map(&|x| x * 2);
        assert_eq!(doubled.value(), &2);
        assert_eq!(doubled.left().unwrap().value(), &4);
        assert_eq!(doubled.right().unwrap().value(), &6);
    }

    #[test]
    fn test_find() {
        let mut root = BinaryTree::new(1);
        let mut left = BinaryTree::new(2);
        left.set_left(BinaryTree::new(4));
        root.set_left(left);
        root.set_right(BinaryTree::new(3));

        assert!(root.find(&4).is_some());
        assert!(root.find(&99).is_none());
    }

    #[test]
    fn test_leaves() {
        let mut root = BinaryTree::new(1);
        let mut left = BinaryTree::new(2);
        left.set_left(BinaryTree::new(4));
        left.set_right(BinaryTree::new(5));
        root.set_left(left);
        root.set_right(BinaryTree::new(3));

        let leaves = root.leaves();
        assert_eq!(leaves.len(), 3);
        assert!(leaves.contains(&&4));
        assert!(leaves.contains(&&5));
        assert!(leaves.contains(&&3));
    }

    #[test]
    fn test_is_balanced() {
        let mut balanced = BinaryTree::new(1);
        balanced.set_left(BinaryTree::new(2));
        balanced.set_right(BinaryTree::new(3));
        assert!(balanced.is_balanced());

        let mut unbalanced = BinaryTree::new(1);
        let mut left = BinaryTree::new(2);
        left.set_left(BinaryTree::new(4));
        unbalanced.set_left(left);
        assert!(unbalanced.is_balanced()); // Still balanced (diff = 1)

        let mut very_unbalanced = BinaryTree::new(1);
        let mut left = BinaryTree::new(2);
        let mut left_left = BinaryTree::new(3);
        left_left.set_left(BinaryTree::new(4));
        left.set_left(left_left);
        very_unbalanced.set_left(left);
        assert!(!very_unbalanced.is_balanced());
    }

    #[test]
    fn test_is_full() {
        let mut full = BinaryTree::new(1);
        full.set_left(BinaryTree::new(2));
        full.set_right(BinaryTree::new(3));
        assert!(full.is_full());

        let mut not_full = BinaryTree::new(1);
        not_full.set_left(BinaryTree::new(2));
        assert!(!not_full.is_full());
    }

    #[test]
    fn test_is_perfect() {
        let mut perfect = BinaryTree::new(1);
        perfect.set_left(BinaryTree::new(2));
        perfect.set_right(BinaryTree::new(3));
        assert!(perfect.is_perfect());

        let mut not_perfect = BinaryTree::new(1);
        let mut left = BinaryTree::new(2);
        left.set_left(BinaryTree::new(4));
        not_perfect.set_left(left);
        not_perfect.set_right(BinaryTree::new(3));
        assert!(!not_perfect.is_perfect());
    }

    #[test]
    fn test_take_children() {
        let mut root = BinaryTree::new(1);
        root.set_left(BinaryTree::new(2));
        root.set_right(BinaryTree::new(3));

        let left = root.take_left().unwrap();
        assert_eq!(left.value(), &2);
        assert!(!root.has_left());

        let right = root.take_right().unwrap();
        assert_eq!(right.value(), &3);
        assert!(!root.has_right());
    }

    #[test]
    fn test_clone() {
        let mut root = BinaryTree::new(1);
        root.set_left(BinaryTree::new(2));
        root.set_right(BinaryTree::new(3));

        let cloned = root.clone();
        assert_eq!(root, cloned);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serialization() {
        let mut root = BinaryTree::new(1);
        root.set_left(BinaryTree::new(2));
        root.set_right(BinaryTree::new(3));

        let json = serde_json::to_string(&root).unwrap();
        let deserialized: BinaryTree<i32> = serde_json::from_str(&json).unwrap();
        assert_eq!(root, deserialized);
    }
}
