//! N-ary tree implementation
//!
//! This module provides a tree where each node can have any number of children.

use crate::traits::{Tree, TreeNode};
use std::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// An n-ary tree where each node can have any number of children
///
/// # Examples
///
/// ```
/// use rustmath_trees::NaryTree;
///
/// let mut root = NaryTree::new("root");
/// root.add_child(NaryTree::new("child1"));
/// root.add_child(NaryTree::new("child2"));
///
/// assert_eq!(root.num_children(), 2);
/// assert_eq!(root.value(), &"root");
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NaryTree<T> {
    pub(crate) value: T,
    pub(crate) children: Vec<NaryTree<T>>,
}

impl<T> NaryTree<T> {
    /// Create a new n-ary tree node with the given value
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_trees::NaryTree;
    ///
    /// let node = NaryTree::new(42);
    /// assert_eq!(node.value(), &42);
    /// assert!(node.is_leaf());
    /// ```
    pub fn new(value: T) -> Self {
        NaryTree {
            value,
            children: Vec::new(),
        }
    }

    /// Create a new n-ary tree node with the given value and children
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_trees::NaryTree;
    ///
    /// let child1 = NaryTree::new(1);
    /// let child2 = NaryTree::new(2);
    /// let root = NaryTree::with_children(0, vec![child1, child2]);
    ///
    /// assert_eq!(root.num_children(), 2);
    /// ```
    pub fn with_children(value: T, children: Vec<NaryTree<T>>) -> Self {
        NaryTree { value, children }
    }

    /// Add a child to this node
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_trees::NaryTree;
    ///
    /// let mut root = NaryTree::new(0);
    /// root.add_child(NaryTree::new(1));
    /// assert_eq!(root.num_children(), 1);
    /// ```
    pub fn add_child(&mut self, child: NaryTree<T>) {
        self.children.push(child);
    }

    /// Get a reference to the children
    pub fn children(&self) -> &[NaryTree<T>] {
        &self.children
    }

    /// Get a mutable reference to the children
    pub fn children_mut(&mut self) -> &mut Vec<NaryTree<T>> {
        &mut self.children
    }

    /// Get a reference to a specific child by index
    ///
    /// Returns None if the index is out of bounds
    pub fn child(&self, index: usize) -> Option<&NaryTree<T>> {
        self.children.get(index)
    }

    /// Get a mutable reference to a specific child by index
    ///
    /// Returns None if the index is out of bounds
    pub fn child_mut(&mut self, index: usize) -> Option<&mut NaryTree<T>> {
        self.children.get_mut(index)
    }

    /// Remove and return a child at the given index
    ///
    /// # Panics
    ///
    /// Panics if index is out of bounds
    pub fn remove_child(&mut self, index: usize) -> NaryTree<T> {
        self.children.remove(index)
    }

    /// Remove all children
    pub fn clear_children(&mut self) {
        self.children.clear();
    }

    /// Map a function over all values in the tree
    ///
    /// Creates a new tree with the function applied to each value
    pub fn map<U, F>(self, f: &F) -> NaryTree<U>
    where
        F: Fn(T) -> U,
    {
        NaryTree {
            value: f(self.value),
            children: self
                .children
                .into_iter()
                .map(|child| child.map(f))
                .collect(),
        }
    }

    /// Apply a function to each value in the tree (in-place)
    pub fn map_mut<F>(&mut self, f: &F)
    where
        F: Fn(&mut T),
    {
        f(&mut self.value);
        for child in &mut self.children {
            child.map_mut(f);
        }
    }

    /// Find a node with a specific value (depth-first search)
    ///
    /// Returns a reference to the first node found with the given value
    pub fn find(&self, target: &T) -> Option<&NaryTree<T>>
    where
        T: PartialEq,
    {
        if &self.value == target {
            return Some(self);
        }

        for child in &self.children {
            if let Some(found) = child.find(target) {
                return Some(found);
            }
        }

        None
    }

    /// Find a node with a specific value (mutable)
    pub fn find_mut(&mut self, target: &T) -> Option<&mut NaryTree<T>>
    where
        T: PartialEq,
    {
        if &self.value == target {
            return Some(self);
        }

        for child in &mut self.children {
            if let Some(found) = child.find_mut(target) {
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
            self.children
                .iter()
                .flat_map(|child| child.leaves())
                .collect()
        }
    }

    /// Get the level (distance from root) of each node as a vector of (level, value) pairs
    pub fn levels(&self) -> Vec<(usize, &T)> {
        let mut result = Vec::new();
        self.levels_helper(0, &mut result);
        result
    }

    fn levels_helper<'a>(&'a self, level: usize, result: &mut Vec<(usize, &'a T)>) {
        result.push((level, &self.value));
        for child in &self.children {
            child.levels_helper(level + 1, result);
        }
    }
}

impl<T: Clone + fmt::Debug> Tree for NaryTree<T> {
    type Value = T;

    fn value(&self) -> &Self::Value {
        &self.value
    }

    fn value_mut(&mut self) -> &mut Self::Value {
        &mut self.value
    }

    fn height(&self) -> usize {
        if self.children.is_empty() {
            0
        } else {
            1 + self
                .children
                .iter()
                .map(|child| child.height())
                .max()
                .unwrap_or(0)
        }
    }

    fn size(&self) -> usize {
        1 + self.children.iter().map(|child| child.size()).sum::<usize>()
    }

    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    fn depth_to(&self, descendant: &Self) -> Option<usize>
    where
        Self::Value: PartialEq,
    {
        if self == descendant {
            return Some(0);
        }

        for child in &self.children {
            if let Some(depth) = child.depth_to(descendant) {
                return Some(depth + 1);
            }
        }

        None
    }
}

impl<T: Clone + fmt::Debug> TreeNode for NaryTree<T> {
    fn num_children(&self) -> usize {
        self.children.len()
    }
}

impl<T: fmt::Display> fmt::Display for NaryTree<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_helper(f, 0)
    }
}

impl<T: fmt::Display> NaryTree<T> {
    fn fmt_helper(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        writeln!(f, "{}{}", "  ".repeat(indent), self.value)?;
        for child in &self.children {
            child.fmt_helper(f, indent + 1)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let tree: NaryTree<i32> = NaryTree::new(42);
        assert_eq!(tree.value(), &42);
        assert!(tree.is_leaf());
        assert_eq!(tree.height(), 0);
        assert_eq!(tree.size(), 1);
    }

    #[test]
    fn test_add_child() {
        let mut root = NaryTree::new(1);
        root.add_child(NaryTree::new(2));
        root.add_child(NaryTree::new(3));

        assert_eq!(root.num_children(), 2);
        assert!(!root.is_leaf());
        assert_eq!(root.height(), 1);
        assert_eq!(root.size(), 3);
    }

    #[test]
    fn test_with_children() {
        let child1 = NaryTree::new(2);
        let child2 = NaryTree::new(3);
        let root = NaryTree::with_children(1, vec![child1, child2]);

        assert_eq!(root.value(), &1);
        assert_eq!(root.num_children(), 2);
    }

    #[test]
    fn test_height() {
        let mut root = NaryTree::new(1);
        let mut child1 = NaryTree::new(2);
        child1.add_child(NaryTree::new(4));
        root.add_child(child1);
        root.add_child(NaryTree::new(3));

        assert_eq!(root.height(), 2);
    }

    #[test]
    fn test_size() {
        let mut root = NaryTree::new(1);
        let mut child1 = NaryTree::new(2);
        child1.add_child(NaryTree::new(4));
        child1.add_child(NaryTree::new(5));
        root.add_child(child1);
        root.add_child(NaryTree::new(3));

        assert_eq!(root.size(), 5);
    }

    #[test]
    fn test_map() {
        let mut root = NaryTree::new(1);
        root.add_child(NaryTree::new(2));
        root.add_child(NaryTree::new(3));

        let doubled = root.map(&|x| x * 2);
        assert_eq!(doubled.value(), &2);
        assert_eq!(doubled.child(0).unwrap().value(), &4);
        assert_eq!(doubled.child(1).unwrap().value(), &6);
    }

    #[test]
    fn test_find() {
        let mut root = NaryTree::new(1);
        let mut child1 = NaryTree::new(2);
        child1.add_child(NaryTree::new(4));
        root.add_child(child1);
        root.add_child(NaryTree::new(3));

        assert!(root.find(&4).is_some());
        assert!(root.find(&99).is_none());
    }

    #[test]
    fn test_leaves() {
        let mut root = NaryTree::new(1);
        let mut child1 = NaryTree::new(2);
        child1.add_child(NaryTree::new(4));
        child1.add_child(NaryTree::new(5));
        root.add_child(child1);
        root.add_child(NaryTree::new(3));

        let leaves = root.leaves();
        assert_eq!(leaves.len(), 3);
        assert!(leaves.contains(&&4));
        assert!(leaves.contains(&&5));
        assert!(leaves.contains(&&3));
    }

    #[test]
    fn test_clone() {
        let mut root = NaryTree::new(1);
        root.add_child(NaryTree::new(2));

        let cloned = root.clone();
        assert_eq!(root, cloned);
    }

    #[test]
    fn test_remove_child() {
        let mut root = NaryTree::new(1);
        root.add_child(NaryTree::new(2));
        root.add_child(NaryTree::new(3));

        let removed = root.remove_child(0);
        assert_eq!(removed.value(), &2);
        assert_eq!(root.num_children(), 1);
    }

    #[test]
    fn test_levels() {
        let mut root = NaryTree::new(1);
        let mut child1 = NaryTree::new(2);
        child1.add_child(NaryTree::new(4));
        root.add_child(child1);
        root.add_child(NaryTree::new(3));

        let levels = root.levels();
        assert_eq!(levels.len(), 4);
        assert_eq!(levels[0], (0, &1));
        assert_eq!(levels[1], (1, &2));
        assert_eq!(levels[2], (2, &4));
        assert_eq!(levels[3], (1, &3));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serialization() {
        let mut root = NaryTree::new(1);
        root.add_child(NaryTree::new(2));
        root.add_child(NaryTree::new(3));

        let json = serde_json::to_string(&root).unwrap();
        let deserialized: NaryTree<i32> = serde_json::from_str(&json).unwrap();
        assert_eq!(root, deserialized);
    }
}
