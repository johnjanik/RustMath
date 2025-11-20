//! Tree traversal algorithms
//!
//! This module provides various tree traversal strategies as iterators.

use crate::{BinaryTree, NaryTree};
use std::collections::VecDeque;

/// Traversal order for tree iteration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraversalOrder {
    /// Visit the root, then the left subtree, then the right subtree
    PreOrder,
    /// Visit the left subtree, then the root, then the right subtree (binary trees only)
    InOrder,
    /// Visit the left subtree, then the right subtree, then the root
    PostOrder,
    /// Visit nodes level by level (breadth-first)
    LevelOrder,
}

/// Trait for tree traversal
///
/// This trait provides iterator-based traversal for trees.
pub trait TreeTraversal {
    /// The type of values in the tree
    type Value;

    /// Iterate over the tree in pre-order
    fn preorder(&self) -> PreOrderIter<'_, Self::Value>;

    /// Iterate over the tree in post-order
    fn postorder(&self) -> PostOrderIter<'_, Self::Value>;

    /// Iterate over the tree in level-order (breadth-first)
    fn levelorder(&self) -> LevelOrderIter<'_, Self::Value>;
}

// ============================================================================
// N-ary Tree Traversal Implementations
// ============================================================================

impl<T> TreeTraversal for NaryTree<T> {
    type Value = T;

    fn preorder(&self) -> PreOrderIter<'_, T> {
        PreOrderIter::new_nary(self)
    }

    fn postorder(&self) -> PostOrderIter<'_, T> {
        PostOrderIter::new_nary(self)
    }

    fn levelorder(&self) -> LevelOrderIter<'_, T> {
        LevelOrderIter::new_nary(self)
    }
}

/// Pre-order iterator for trees
///
/// Visits nodes in the order: root, left subtree, right subtree
pub struct PreOrderIter<'a, T> {
    stack: Vec<NodeRef<'a, T>>,
}

impl<'a, T> PreOrderIter<'a, T> {
    fn new_nary(root: &'a NaryTree<T>) -> Self {
        PreOrderIter {
            stack: vec![NodeRef::Nary(root)],
        }
    }

    fn new_binary(root: &'a BinaryTree<T>) -> Self {
        PreOrderIter {
            stack: vec![NodeRef::Binary(root)],
        }
    }
}

impl<'a, T> Iterator for PreOrderIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.stack.pop()?;

        match node {
            NodeRef::Nary(n) => {
                // Push children in reverse order so they're visited left-to-right
                for child in n.children().iter().rev() {
                    self.stack.push(NodeRef::Nary(child));
                }
                Some(&n.value)
            }
            NodeRef::Binary(b) => {
                // Push right first, then left (so left is processed first)
                if let Some(right) = b.right() {
                    self.stack.push(NodeRef::Binary(right));
                }
                if let Some(left) = b.left() {
                    self.stack.push(NodeRef::Binary(left));
                }
                Some(&b.value)
            }
        }
    }
}

/// Post-order iterator for trees
///
/// Visits nodes in the order: left subtree, right subtree, root
pub struct PostOrderIter<'a, T> {
    stack: Vec<(NodeRef<'a, T>, bool)>,
}

impl<'a, T> PostOrderIter<'a, T> {
    fn new_nary(root: &'a NaryTree<T>) -> Self {
        PostOrderIter {
            stack: vec![(NodeRef::Nary(root), false)],
        }
    }

    fn new_binary(root: &'a BinaryTree<T>) -> Self {
        PostOrderIter {
            stack: vec![(NodeRef::Binary(root), false)],
        }
    }
}

impl<'a, T> Iterator for PostOrderIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, visited)) = self.stack.pop() {
            if visited {
                return Some(match node {
                    NodeRef::Nary(n) => &n.value,
                    NodeRef::Binary(b) => &b.value,
                });
            }

            // Push children to be visited first, then push current node as visited
            match node {
                NodeRef::Nary(n) => {
                    // Mark this node as visited and push it back
                    self.stack.push((node, true));
                    for child in n.children().iter().rev() {
                        self.stack.push((NodeRef::Nary(child), false));
                    }
                }
                NodeRef::Binary(b) => {
                    // Mark this node as visited and push it back
                    self.stack.push((node, true));
                    if let Some(right) = b.right() {
                        self.stack.push((NodeRef::Binary(right), false));
                    }
                    if let Some(left) = b.left() {
                        self.stack.push((NodeRef::Binary(left), false));
                    }
                }
            }
        }
        None
    }
}

/// Level-order (breadth-first) iterator for trees
pub struct LevelOrderIter<'a, T> {
    queue: VecDeque<NodeRef<'a, T>>,
}

impl<'a, T> LevelOrderIter<'a, T> {
    fn new_nary(root: &'a NaryTree<T>) -> Self {
        let mut queue = VecDeque::new();
        queue.push_back(NodeRef::Nary(root));
        LevelOrderIter { queue }
    }

    fn new_binary(root: &'a BinaryTree<T>) -> Self {
        let mut queue = VecDeque::new();
        queue.push_back(NodeRef::Binary(root));
        LevelOrderIter { queue }
    }
}

impl<'a, T> Iterator for LevelOrderIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.queue.pop_front()?;

        match node {
            NodeRef::Nary(n) => {
                for child in n.children() {
                    self.queue.push_back(NodeRef::Nary(child));
                }
                Some(&n.value)
            }
            NodeRef::Binary(b) => {
                if let Some(left) = b.left() {
                    self.queue.push_back(NodeRef::Binary(left));
                }
                if let Some(right) = b.right() {
                    self.queue.push_back(NodeRef::Binary(right));
                }
                Some(&b.value)
            }
        }
    }
}

// ============================================================================
// Binary Tree Specific Traversal
// ============================================================================

/// Additional traversal methods for binary trees
pub trait BinaryTreeTraversal: TreeTraversal {
    /// Iterate over the tree in in-order (left, root, right)
    ///
    /// This is only available for binary trees
    fn inorder(&self) -> InOrderIter<'_, Self::Value>;
}

impl<T> BinaryTreeTraversal for BinaryTree<T> {
    fn inorder(&self) -> InOrderIter<'_, T> {
        InOrderIter::new(self)
    }
}

impl<T> TreeTraversal for BinaryTree<T> {
    type Value = T;

    fn preorder(&self) -> PreOrderIter<'_, T> {
        PreOrderIter::new_binary(self)
    }

    fn postorder(&self) -> PostOrderIter<'_, T> {
        PostOrderIter::new_binary(self)
    }

    fn levelorder(&self) -> LevelOrderIter<'_, T> {
        LevelOrderIter::new_binary(self)
    }
}

/// In-order iterator for binary trees
///
/// Visits nodes in the order: left subtree, root, right subtree
pub struct InOrderIter<'a, T> {
    stack: Vec<(&'a BinaryTree<T>, bool)>,
}

impl<'a, T> InOrderIter<'a, T> {
    fn new(root: &'a BinaryTree<T>) -> Self {
        InOrderIter {
            stack: vec![(root, false)],
        }
    }
}

impl<'a, T> Iterator for InOrderIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, visited)) = self.stack.pop() {
            if visited {
                return Some(&node.value);
            }

            // Push right child (to be processed last)
            if let Some(right) = node.right() {
                self.stack.push((right, false));
            }

            // Push current node (to be visited after left)
            self.stack.push((node, true));

            // Push left child (to be processed first)
            if let Some(left) = node.left() {
                self.stack.push((left, false));
            }
        }
        None
    }
}

// Helper enum to handle both tree types in iterators
#[derive(Clone, Copy)]
enum NodeRef<'a, T> {
    Nary(&'a NaryTree<T>),
    Binary(&'a BinaryTree<T>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nary_preorder() {
        let mut root = NaryTree::new(1);
        let mut child1 = NaryTree::new(2);
        child1.add_child(NaryTree::new(4));
        child1.add_child(NaryTree::new(5));
        root.add_child(child1);
        root.add_child(NaryTree::new(3));

        let values: Vec<i32> = root.preorder().copied().collect();
        assert_eq!(values, vec![1, 2, 4, 5, 3]);
    }

    #[test]
    fn test_nary_postorder() {
        let mut root = NaryTree::new(1);
        let mut child1 = NaryTree::new(2);
        child1.add_child(NaryTree::new(4));
        child1.add_child(NaryTree::new(5));
        root.add_child(child1);
        root.add_child(NaryTree::new(3));

        let values: Vec<i32> = root.postorder().copied().collect();
        assert_eq!(values, vec![4, 5, 2, 3, 1]);
    }

    #[test]
    fn test_nary_levelorder() {
        let mut root = NaryTree::new(1);
        let mut child1 = NaryTree::new(2);
        child1.add_child(NaryTree::new(4));
        child1.add_child(NaryTree::new(5));
        root.add_child(child1);
        root.add_child(NaryTree::new(3));

        let values: Vec<i32> = root.levelorder().copied().collect();
        assert_eq!(values, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_binary_preorder() {
        let mut root = BinaryTree::new(1);
        let mut left = BinaryTree::new(2);
        left.set_left(BinaryTree::new(4));
        left.set_right(BinaryTree::new(5));
        root.set_left(left);
        root.set_right(BinaryTree::new(3));

        let values: Vec<i32> = root.preorder().copied().collect();
        assert_eq!(values, vec![1, 2, 4, 5, 3]);
    }

    #[test]
    fn test_binary_inorder() {
        let mut root = BinaryTree::new(1);
        let mut left = BinaryTree::new(2);
        left.set_left(BinaryTree::new(4));
        left.set_right(BinaryTree::new(5));
        root.set_left(left);
        root.set_right(BinaryTree::new(3));

        let values: Vec<i32> = root.inorder().copied().collect();
        assert_eq!(values, vec![4, 2, 5, 1, 3]);
    }

    #[test]
    fn test_binary_postorder() {
        let mut root = BinaryTree::new(1);
        let mut left = BinaryTree::new(2);
        left.set_left(BinaryTree::new(4));
        left.set_right(BinaryTree::new(5));
        root.set_left(left);
        root.set_right(BinaryTree::new(3));

        let values: Vec<i32> = root.postorder().copied().collect();
        assert_eq!(values, vec![4, 5, 2, 3, 1]);
    }

    #[test]
    fn test_binary_levelorder() {
        let mut root = BinaryTree::new(1);
        let mut left = BinaryTree::new(2);
        left.set_left(BinaryTree::new(4));
        left.set_right(BinaryTree::new(5));
        root.set_left(left);
        root.set_right(BinaryTree::new(3));

        let values: Vec<i32> = root.levelorder().copied().collect();
        assert_eq!(values, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_single_node_traversal() {
        let tree = NaryTree::new(42);

        let preorder: Vec<i32> = tree.preorder().copied().collect();
        let postorder: Vec<i32> = tree.postorder().copied().collect();
        let levelorder: Vec<i32> = tree.levelorder().copied().collect();

        assert_eq!(preorder, vec![42]);
        assert_eq!(postorder, vec![42]);
        assert_eq!(levelorder, vec![42]);
    }
}
