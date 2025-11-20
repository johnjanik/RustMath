//! Core tree traits
//!
//! This module defines the fundamental traits for tree data structures.

use std::fmt::Debug;

/// Core trait for tree nodes
///
/// This trait defines the essential operations that any tree node must support.
pub trait Tree: Clone + Debug {
    /// The type of value stored in the tree node
    type Value;

    /// Get a reference to the value stored in this node
    fn value(&self) -> &Self::Value;

    /// Get a mutable reference to the value stored in this node
    fn value_mut(&mut self) -> &mut Self::Value;

    /// Get the height of the tree (longest path to a leaf)
    ///
    /// A leaf node has height 0, and a node with children has height
    /// 1 + max(height of children).
    fn height(&self) -> usize;

    /// Get the size of the tree (total number of nodes)
    fn size(&self) -> usize;

    /// Check if this is a leaf node (has no children)
    fn is_leaf(&self) -> bool;

    /// Get the depth from this node to a descendant
    ///
    /// Returns None if the descendant is not found
    fn depth_to(&self, descendant: &Self) -> Option<usize>
    where
        Self::Value: PartialEq;
}

/// Trait for tree nodes with parent-child relationships
///
/// This trait extends Tree with methods for navigating the tree structure.
pub trait TreeNode: Tree {
    /// Get the number of children
    fn num_children(&self) -> usize;

    /// Check if this node has any children
    fn has_children(&self) -> bool {
        self.num_children() > 0
    }
}
