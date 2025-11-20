//! Threaded binary tree implementation
//!
//! This module provides a threaded binary tree where null pointers are replaced with
//! threads pointing to the inorder predecessor or successor. This allows for efficient
//! traversal without recursion or an explicit stack.
//!
//! # Examples
//!
//! ```
//! use rustmath_trees::ThreadedBinaryTree;
//!
//! let mut tree = ThreadedBinaryTree::new();
//! tree.insert(5);
//! tree.insert(3);
//! tree.insert(7);
//!
//! let inorder: Vec<i32> = tree.inorder_traversal().collect();
//! assert_eq!(inorder, vec![3, 5, 7]);
//! ```

use std::cell::RefCell;
use std::cmp::Ordering;
use std::fmt;
use std::rc::Rc;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

type Link<T> = Option<Rc<RefCell<ThreadedNode<T>>>>;

/// A node in a threaded binary tree
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
struct ThreadedNode<T> {
    value: T,
    left: Link<T>,
    right: Link<T>,
    left_thread: bool,  // true if left is a thread, false if it's a child
    right_thread: bool, // true if right is a thread, false if it's a child
}

impl<T> ThreadedNode<T> {
    fn new(value: T) -> Self {
        ThreadedNode {
            value,
            left: None,
            right: None,
            left_thread: true,
            right_thread: true,
        }
    }
}

/// A threaded binary search tree
///
/// In a threaded binary tree, null pointers are replaced with threads:
/// - Left threads point to the inorder predecessor
/// - Right threads point to the inorder successor
///
/// This allows for efficient traversal without recursion or stack.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ThreadedBinaryTree<T> {
    root: Link<T>,
    size: usize,
}

impl<T> ThreadedBinaryTree<T> {
    /// Create a new empty threaded binary tree
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_trees::ThreadedBinaryTree;
    ///
    /// let tree: ThreadedBinaryTree<i32> = ThreadedBinaryTree::new();
    /// assert!(tree.is_empty());
    /// ```
    pub fn new() -> Self {
        ThreadedBinaryTree {
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

    /// Insert a value into the threaded binary search tree
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_trees::ThreadedBinaryTree;
    ///
    /// let mut tree = ThreadedBinaryTree::new();
    /// tree.insert(5);
    /// tree.insert(3);
    /// tree.insert(7);
    /// assert_eq!(tree.len(), 3);
    /// ```
    pub fn insert(&mut self, value: T)
    where
        T: Ord,
    {
        if self.root.is_none() {
            self.root = Some(Rc::new(RefCell::new(ThreadedNode::new(value))));
            self.size = 1;
            return;
        }

        let mut current = self.root.clone().unwrap();
        loop {
            let cmp = value.cmp(&current.borrow().value);
            match cmp {
                Ordering::Less => {
                    if current.borrow().left_thread {
                        // Insert as left child
                        let new_node = Rc::new(RefCell::new(ThreadedNode::new(value)));

                        // Set threading for new node
                        new_node.borrow_mut().left = current.borrow().left.clone();
                        new_node.borrow_mut().left_thread = true;
                        new_node.borrow_mut().right = Some(current.clone());
                        new_node.borrow_mut().right_thread = true;

                        // Update current node
                        current.borrow_mut().left = Some(new_node);
                        current.borrow_mut().left_thread = false;

                        self.size += 1;
                        break;
                    } else {
                        let next = current.borrow().left.clone().unwrap();
                        current = next;
                    }
                }
                Ordering::Greater => {
                    if current.borrow().right_thread {
                        // Insert as right child
                        let new_node = Rc::new(RefCell::new(ThreadedNode::new(value)));

                        // Set threading for new node
                        new_node.borrow_mut().left = Some(current.clone());
                        new_node.borrow_mut().left_thread = true;
                        new_node.borrow_mut().right = current.borrow().right.clone();
                        new_node.borrow_mut().right_thread = true;

                        // Update current node
                        current.borrow_mut().right = Some(new_node);
                        current.borrow_mut().right_thread = false;

                        self.size += 1;
                        break;
                    } else {
                        let next = current.borrow().right.clone().unwrap();
                        current = next;
                    }
                }
                Ordering::Equal => {
                    // Duplicate value, don't insert
                    break;
                }
            }
        }
    }

    /// Search for a value in the tree
    ///
    /// Returns true if the value is found, false otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_trees::ThreadedBinaryTree;
    ///
    /// let mut tree = ThreadedBinaryTree::new();
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
        if self.root.is_none() {
            return false;
        }

        let mut current = self.root.clone();
        while let Some(node) = current {
            let cmp = value.cmp(&node.borrow().value);
            match cmp {
                Ordering::Equal => return true,
                Ordering::Less => {
                    if node.borrow().left_thread {
                        return false;
                    }
                    current = node.borrow().left.clone();
                }
                Ordering::Greater => {
                    if node.borrow().right_thread {
                        return false;
                    }
                    current = node.borrow().right.clone();
                }
            }
        }
        false
    }

    /// Find the leftmost (minimum) node in the tree
    fn leftmost(&self) -> Link<T> {
        if self.root.is_none() {
            return None;
        }

        let mut current = self.root.clone().unwrap();
        while !current.borrow().left_thread {
            let next = current.borrow().left.clone().unwrap();
            current = next;
        }
        Some(current)
    }

    /// Find the inorder successor of a node
    fn inorder_successor(node: &Rc<RefCell<ThreadedNode<T>>>) -> Link<T> {
        if node.borrow().right_thread {
            // If right is a thread, it points to the successor
            return node.borrow().right.clone();
        }

        // Otherwise, find the leftmost node in the right subtree
        let mut current = node.borrow().right.clone().unwrap();
        while !current.borrow().left_thread {
            let next = current.borrow().left.clone().unwrap();
            current = next;
        }
        Some(current)
    }

    /// Perform an inorder traversal of the tree
    ///
    /// Returns an iterator over the values in sorted order.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_trees::ThreadedBinaryTree;
    ///
    /// let mut tree = ThreadedBinaryTree::new();
    /// tree.insert(5);
    /// tree.insert(3);
    /// tree.insert(7);
    /// tree.insert(1);
    /// tree.insert(9);
    ///
    /// let values: Vec<i32> = tree.inorder_traversal().collect();
    /// assert_eq!(values, vec![1, 3, 5, 7, 9]);
    /// ```
    pub fn inorder_traversal(&self) -> InorderIterator<T> {
        InorderIterator {
            current: self.leftmost(),
        }
    }

    /// Get the minimum value in the tree
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_trees::ThreadedBinaryTree;
    ///
    /// let mut tree = ThreadedBinaryTree::new();
    /// tree.insert(5);
    /// tree.insert(3);
    /// tree.insert(7);
    ///
    /// assert_eq!(tree.min(), Some(&3));
    /// ```
    pub fn min(&self) -> Option<&T> {
        self.leftmost().as_ref().map(|node| {
            // SAFETY: We need to convert Rc<RefCell<T>> to &T
            // This is safe because we're just reading the value
            unsafe { &(*node.as_ptr()).value }
        })
    }

    /// Get the maximum value in the tree
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_trees::ThreadedBinaryTree;
    ///
    /// let mut tree = ThreadedBinaryTree::new();
    /// tree.insert(5);
    /// tree.insert(3);
    /// tree.insert(7);
    ///
    /// assert_eq!(tree.max(), Some(&7));
    /// ```
    pub fn max(&self) -> Option<&T> {
        if self.root.is_none() {
            return None;
        }

        let mut current = self.root.clone().unwrap();
        while !current.borrow().right_thread {
            let next = current.borrow().right.clone().unwrap();
            current = next;
        }

        // SAFETY: We need to convert Rc<RefCell<T>> to &T
        // This is safe because we're just reading the value
        unsafe { Some(&(*current.as_ptr()).value) }
    }
}

impl<T> Default for ThreadedBinaryTree<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator for inorder traversal of a threaded binary tree
pub struct InorderIterator<T> {
    current: Link<T>,
}

impl<T: Clone> Iterator for InorderIterator<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(node) = &self.current {
            let value = node.borrow().value.clone();
            self.current = ThreadedBinaryTree::inorder_successor(node);
            Some(value)
        } else {
            None
        }
    }
}

impl<T: fmt::Display> fmt::Display for ThreadedBinaryTree<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.root.is_none() {
            return write!(f, "Empty tree");
        }
        self.fmt_helper(&self.root, f, "", "")
    }
}

impl<T: fmt::Display> ThreadedBinaryTree<T> {
    fn fmt_helper(
        &self,
        node: &Link<T>,
        f: &mut fmt::Formatter<'_>,
        prefix: &str,
        child_prefix: &str,
    ) -> fmt::Result {
        if let Some(n) = node {
            writeln!(f, "{}{}", prefix, n.borrow().value)?;

            let has_right = !n.borrow().right_thread;
            let has_left = !n.borrow().left_thread;

            if has_left {
                let new_prefix = format!("{}├── ", child_prefix);
                let new_child_prefix = if has_right {
                    format!("{}│   ", child_prefix)
                } else {
                    format!("{}    ", child_prefix)
                };
                self.fmt_helper(&n.borrow().left, f, &new_prefix, &new_child_prefix)?;
            }

            if has_right {
                let new_prefix = format!("{}└── ", child_prefix);
                let new_child_prefix = format!("{}    ", child_prefix);
                self.fmt_helper(&n.borrow().right, f, &new_prefix, &new_child_prefix)?;
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
        let tree: ThreadedBinaryTree<i32> = ThreadedBinaryTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn test_insert_single() {
        let mut tree = ThreadedBinaryTree::new();
        tree.insert(5);
        assert_eq!(tree.len(), 1);
        assert!(!tree.is_empty());
    }

    #[test]
    fn test_insert_multiple() {
        let mut tree = ThreadedBinaryTree::new();
        tree.insert(5);
        tree.insert(3);
        tree.insert(7);
        tree.insert(1);
        tree.insert(9);

        assert_eq!(tree.len(), 5);
    }

    #[test]
    fn test_insert_duplicates() {
        let mut tree = ThreadedBinaryTree::new();
        tree.insert(5);
        tree.insert(5);
        tree.insert(5);

        assert_eq!(tree.len(), 1); // Duplicates not inserted
    }

    #[test]
    fn test_search() {
        let mut tree = ThreadedBinaryTree::new();
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
    fn test_inorder_traversal() {
        let mut tree = ThreadedBinaryTree::new();
        tree.insert(5);
        tree.insert(3);
        tree.insert(7);
        tree.insert(1);
        tree.insert(9);
        tree.insert(4);
        tree.insert(6);

        let values: Vec<i32> = tree.inorder_traversal().collect();
        assert_eq!(values, vec![1, 3, 4, 5, 6, 7, 9]);
    }

    #[test]
    fn test_inorder_traversal_empty() {
        let tree: ThreadedBinaryTree<i32> = ThreadedBinaryTree::new();
        let values: Vec<i32> = tree.inorder_traversal().collect();
        assert_eq!(values, vec![]);
    }

    #[test]
    fn test_inorder_traversal_single() {
        let mut tree = ThreadedBinaryTree::new();
        tree.insert(5);

        let values: Vec<i32> = tree.inorder_traversal().collect();
        assert_eq!(values, vec![5]);
    }

    #[test]
    fn test_min_max() {
        let mut tree = ThreadedBinaryTree::new();
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
        let tree: ThreadedBinaryTree<i32> = ThreadedBinaryTree::new();
        assert_eq!(tree.min(), None);
        assert_eq!(tree.max(), None);
    }

    #[test]
    fn test_clone() {
        let mut tree = ThreadedBinaryTree::new();
        tree.insert(5);
        tree.insert(3);
        tree.insert(7);

        let cloned = tree.clone();
        assert_eq!(cloned.len(), tree.len());

        let values: Vec<i32> = cloned.inorder_traversal().collect();
        assert_eq!(values, vec![3, 5, 7]);
    }
}
