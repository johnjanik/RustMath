//! RustMath Trees - Abstract tree data structures with trait-based interfaces
//!
//! This crate provides generic tree data structures with support for:
//! - Multiple tree types (n-ary trees, binary trees, threaded trees, AVL trees)
//! - Tree traversal (pre-order, in-order, post-order, level-order)
//! - Self-balancing trees (AVL)
//! - Threaded binary trees for efficient traversal
//! - Tree generation (complete, full, perfect binary trees)
//! - Visitor pattern for custom tree operations
//! - Serialization support (with `serde` feature)
//! - Cloneable tree operations
//!
//! # Examples
//!
//! ```
//! use rustmath_trees::{NaryTree, TreeTraversal};
//!
//! let mut root = NaryTree::new(1);
//! root.add_child(NaryTree::new(2));
//! root.add_child(NaryTree::new(3));
//!
//! let values: Vec<i32> = root.preorder().copied().collect();
//! assert_eq!(values, vec![1, 2, 3]);
//! ```
//!
//! # AVL Trees
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
//! assert!(tree.is_balanced());
//! ```
//!
//! # Threaded Binary Trees
//!
//! ```
//! use rustmath_trees::ThreadedBinaryTree;
//!
//! let mut tree = ThreadedBinaryTree::new();
//! tree.insert(5);
//! tree.insert(3);
//! tree.insert(7);
//!
//! let values: Vec<i32> = tree.inorder_traversal().collect();
//! assert_eq!(values, vec![3, 5, 7]);
//! ```

pub mod traits;
pub mod nary;
pub mod binary;
pub mod traversal;
pub mod visitor;
pub mod threaded;
pub mod avl;

pub use traits::{Tree, TreeNode};
pub use nary::NaryTree;
pub use binary::BinaryTree;
pub use traversal::{TreeTraversal, TraversalOrder};
pub use visitor::{TreeVisitor, TreeMutator};
pub use threaded::{ThreadedBinaryTree, InorderIterator};
pub use avl::AvlTree;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_nary_tree() {
        let mut root = NaryTree::new(1);
        root.add_child(NaryTree::new(2));
        root.add_child(NaryTree::new(3));

        assert_eq!(root.value(), &1);
        assert_eq!(root.num_children(), 2);
    }

    #[test]
    fn basic_binary_tree() {
        let mut root = BinaryTree::new(1);
        root.set_left(BinaryTree::new(2));
        root.set_right(BinaryTree::new(3));

        assert_eq!(root.value(), &1);
        assert!(root.left().is_some());
        assert!(root.right().is_some());
    }

    #[test]
    fn test_tree_traversal() {
        let mut root = NaryTree::new(1);
        let mut child1 = NaryTree::new(2);
        child1.add_child(NaryTree::new(4));
        child1.add_child(NaryTree::new(5));
        root.add_child(child1);
        root.add_child(NaryTree::new(3));

        let values: Vec<i32> = root.preorder().copied().collect();
        assert_eq!(values, vec![1, 2, 4, 5, 3]);
    }
}
