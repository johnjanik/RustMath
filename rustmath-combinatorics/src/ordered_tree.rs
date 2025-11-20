//! Ordered plane trees with left-child right-sibling representation
//!
//! This module provides an implementation of ordered plane trees using the
//! left-child right-sibling (LCRS) representation. In this representation,
//! each node stores a pointer to its leftmost child and to its right sibling.
//!
//! Properties:
//! - Ordered: The children of each node have a specific order
//! - Plane: The tree is embedded in the plane (children have left-to-right order)
//! - LCRS representation: Space-efficient and allows for easy tree traversal

use std::fmt;

/// A node in an ordered tree using left-child right-sibling representation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderedTreeNode<T> {
    /// The value stored in this node
    value: T,
    /// Index of the leftmost child (None if no children)
    left_child: Option<usize>,
    /// Index of the right sibling (None if this is the rightmost sibling)
    right_sibling: Option<usize>,
    /// Index of the parent node (None for root)
    parent: Option<usize>,
}

impl<T> OrderedTreeNode<T> {
    /// Create a new tree node with the given value
    pub fn new(value: T) -> Self {
        OrderedTreeNode {
            value,
            left_child: None,
            right_sibling: None,
            parent: None,
        }
    }

    /// Get the value of this node
    pub fn value(&self) -> &T {
        &self.value
    }

    /// Get a mutable reference to the value of this node
    pub fn value_mut(&mut self) -> &mut T {
        &mut self.value
    }

    /// Get the index of the leftmost child
    pub fn left_child(&self) -> Option<usize> {
        self.left_child
    }

    /// Get the index of the right sibling
    pub fn right_sibling(&self) -> Option<usize> {
        self.right_sibling
    }

    /// Get the index of the parent
    pub fn parent(&self) -> Option<usize> {
        self.parent
    }
}

/// An ordered plane tree with left-child right-sibling representation
///
/// The tree stores all nodes in a vector, and nodes reference each other by index.
/// This provides cache-friendly access and avoids pointer-based structures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderedTree<T> {
    /// All nodes in the tree, stored in a vector
    nodes: Vec<OrderedTreeNode<T>>,
    /// Index of the root node (0 for non-empty trees)
    root: Option<usize>,
}

impl<T> OrderedTree<T> {
    /// Create an empty tree
    pub fn new() -> Self {
        OrderedTree {
            nodes: Vec::new(),
            root: None,
        }
    }

    /// Create a tree with a single root node
    pub fn with_root(value: T) -> Self {
        let mut tree = OrderedTree {
            nodes: vec![OrderedTreeNode::new(value)],
            root: Some(0),
        };
        tree
    }

    /// Check if the tree is empty
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// Get the number of nodes in the tree
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Get the index of the root node
    pub fn root(&self) -> Option<usize> {
        self.root
    }

    /// Get a reference to a node by index
    pub fn node(&self, index: usize) -> Option<&OrderedTreeNode<T>> {
        self.nodes.get(index)
    }

    /// Get a mutable reference to a node by index
    pub fn node_mut(&mut self, index: usize) -> Option<&mut OrderedTreeNode<T>> {
        self.nodes.get_mut(index)
    }

    /// Add a child to the given parent node
    ///
    /// The new child is added as the rightmost child of the parent.
    /// Returns the index of the newly added node.
    pub fn add_child(&mut self, parent_idx: usize, value: T) -> Option<usize> {
        if parent_idx >= self.nodes.len() {
            return None;
        }

        let new_idx = self.nodes.len();
        let mut new_node = OrderedTreeNode::new(value);
        new_node.parent = Some(parent_idx);

        // Find the rightmost child of the parent
        if let Some(first_child) = self.nodes[parent_idx].left_child {
            // Parent has children, traverse to find the rightmost
            let mut current = first_child;
            while let Some(next_sibling) = self.nodes[current].right_sibling {
                current = next_sibling;
            }
            // Add as right sibling of the rightmost child
            self.nodes[current].right_sibling = Some(new_idx);
        } else {
            // Parent has no children, this becomes the leftmost child
            self.nodes[parent_idx].left_child = Some(new_idx);
        }

        self.nodes.push(new_node);
        Some(new_idx)
    }

    /// Add a child as the leftmost child of the given parent
    ///
    /// The new child is inserted at the beginning of the children list.
    /// Returns the index of the newly added node.
    pub fn add_leftmost_child(&mut self, parent_idx: usize, value: T) -> Option<usize> {
        if parent_idx >= self.nodes.len() {
            return None;
        }

        let new_idx = self.nodes.len();
        let mut new_node = OrderedTreeNode::new(value);
        new_node.parent = Some(parent_idx);

        // If parent has existing children, the new node points to them as sibling
        if let Some(first_child) = self.nodes[parent_idx].left_child {
            new_node.right_sibling = Some(first_child);
        }

        // Update parent to point to new node as leftmost child
        self.nodes[parent_idx].left_child = Some(new_idx);
        self.nodes.push(new_node);
        Some(new_idx)
    }

    /// Get all children of a node as a vector of indices
    pub fn children(&self, parent_idx: usize) -> Vec<usize> {
        if parent_idx >= self.nodes.len() {
            return Vec::new();
        }

        let mut result = Vec::new();
        let mut current = self.nodes[parent_idx].left_child;

        while let Some(child_idx) = current {
            result.push(child_idx);
            current = self.nodes[child_idx].right_sibling;
        }

        result
    }

    /// Get the number of children of a node
    pub fn num_children(&self, parent_idx: usize) -> usize {
        self.children(parent_idx).len()
    }

    /// Get the depth (height) of the tree
    pub fn depth(&self) -> usize {
        if let Some(root_idx) = self.root {
            self.node_depth(root_idx)
        } else {
            0
        }
    }

    /// Get the depth of a subtree rooted at the given node
    fn node_depth(&self, node_idx: usize) -> usize {
        let children = self.children(node_idx);
        if children.is_empty() {
            1
        } else {
            1 + children
                .iter()
                .map(|&child_idx| self.node_depth(child_idx))
                .max()
                .unwrap_or(0)
        }
    }

    /// Perform a pre-order traversal of the tree
    ///
    /// Returns a vector of node indices in pre-order (parent before children)
    pub fn preorder(&self) -> Vec<usize> {
        let mut result = Vec::new();
        if let Some(root_idx) = self.root {
            self.preorder_helper(root_idx, &mut result);
        }
        result
    }

    fn preorder_helper(&self, node_idx: usize, result: &mut Vec<usize>) {
        result.push(node_idx);
        for child_idx in self.children(node_idx) {
            self.preorder_helper(child_idx, result);
        }
    }

    /// Perform a post-order traversal of the tree
    ///
    /// Returns a vector of node indices in post-order (children before parent)
    pub fn postorder(&self) -> Vec<usize> {
        let mut result = Vec::new();
        if let Some(root_idx) = self.root {
            self.postorder_helper(root_idx, &mut result);
        }
        result
    }

    fn postorder_helper(&self, node_idx: usize, result: &mut Vec<usize>) {
        for child_idx in self.children(node_idx) {
            self.postorder_helper(child_idx, result);
        }
        result.push(node_idx);
    }

    /// Perform a level-order (breadth-first) traversal of the tree
    ///
    /// Returns a vector of node indices in level-order
    pub fn level_order(&self) -> Vec<usize> {
        let mut result = Vec::new();
        if let Some(root_idx) = self.root {
            let mut queue = vec![root_idx];
            while !queue.is_empty() {
                let node_idx = queue.remove(0);
                result.push(node_idx);
                queue.extend(self.children(node_idx));
            }
        }
        result
    }

    /// Check if a node is a leaf (has no children)
    pub fn is_leaf(&self, node_idx: usize) -> bool {
        if let Some(node) = self.nodes.get(node_idx) {
            node.left_child.is_none()
        } else {
            false
        }
    }

    /// Get all leaf nodes in the tree
    pub fn leaves(&self) -> Vec<usize> {
        (0..self.nodes.len())
            .filter(|&idx| self.is_leaf(idx))
            .collect()
    }

    /// Convert the tree to a parenthesis representation
    ///
    /// For example, a tree with root 'a' and children 'b', 'c' where 'b' has child 'd'
    /// would be represented as "a(b(d)c)"
    pub fn to_parenthesis(&self) -> String
    where
        T: fmt::Display,
    {
        if let Some(root_idx) = self.root {
            self.node_to_parenthesis(root_idx)
        } else {
            String::new()
        }
    }

    fn node_to_parenthesis(&self, node_idx: usize) -> String
    where
        T: fmt::Display,
    {
        let node = &self.nodes[node_idx];
        let mut result = format!("{}", node.value);

        let children = self.children(node_idx);
        if !children.is_empty() {
            result.push('(');
            for (i, &child_idx) in children.iter().enumerate() {
                if i > 0 {
                    result.push(',');
                }
                result.push_str(&self.node_to_parenthesis(child_idx));
            }
            result.push(')');
        }

        result
    }
}

impl<T> Default for OrderedTree<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: fmt::Display> fmt::Display for OrderedTree<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_parenthesis())
    }
}

/// Iterator over the nodes of a tree in pre-order
pub struct PreorderIterator<'a, T> {
    tree: &'a OrderedTree<T>,
    stack: Vec<usize>,
}

impl<'a, T> PreorderIterator<'a, T> {
    fn new(tree: &'a OrderedTree<T>) -> Self {
        let stack = if let Some(root_idx) = tree.root {
            vec![root_idx]
        } else {
            Vec::new()
        };
        PreorderIterator { tree, stack }
    }
}

impl<'a, T> Iterator for PreorderIterator<'a, T> {
    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(node_idx) = self.stack.pop() {
            let node = &self.tree.nodes[node_idx];

            // Push children in reverse order so leftmost is processed first
            let children = self.tree.children(node_idx);
            for &child_idx in children.iter().rev() {
                self.stack.push(child_idx);
            }

            Some((node_idx, &node.value))
        } else {
            None
        }
    }
}

impl<T> OrderedTree<T> {
    /// Create an iterator over the tree nodes in pre-order
    pub fn iter_preorder(&self) -> PreorderIterator<'_, T> {
        PreorderIterator::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_tree() {
        let tree: OrderedTree<i32> = OrderedTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert_eq!(tree.root(), None);
    }

    #[test]
    fn test_single_node_tree() {
        let tree = OrderedTree::with_root(42);
        assert!(!tree.is_empty());
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.root(), Some(0));
        assert_eq!(tree.node(0).unwrap().value(), &42);
        assert!(tree.is_leaf(0));
    }

    #[test]
    fn test_add_child() {
        let mut tree = OrderedTree::with_root(1);
        let child1 = tree.add_child(0, 2);
        let child2 = tree.add_child(0, 3);

        assert_eq!(child1, Some(1));
        assert_eq!(child2, Some(2));
        assert_eq!(tree.len(), 3);

        let children = tree.children(0);
        assert_eq!(children, vec![1, 2]);
        assert_eq!(tree.node(1).unwrap().value(), &2);
        assert_eq!(tree.node(2).unwrap().value(), &3);
    }

    #[test]
    fn test_add_leftmost_child() {
        let mut tree = OrderedTree::with_root(1);
        tree.add_child(0, 2);
        tree.add_child(0, 3);
        tree.add_leftmost_child(0, 0);

        let children = tree.children(0);
        assert_eq!(children, vec![3, 1, 2]);
        assert_eq!(tree.node(3).unwrap().value(), &0);
    }

    #[test]
    fn test_nested_tree() {
        // Build tree:
        //       1
        //      / \
        //     2   3
        //    /   / \
        //   4   5   6
        let mut tree = OrderedTree::with_root(1);
        let n2 = tree.add_child(0, 2).unwrap();
        let n3 = tree.add_child(0, 3).unwrap();
        tree.add_child(n2, 4);
        tree.add_child(n3, 5);
        tree.add_child(n3, 6);

        assert_eq!(tree.len(), 6);
        assert_eq!(tree.num_children(0), 2);
        assert_eq!(tree.num_children(n2), 1);
        assert_eq!(tree.num_children(n3), 2);
    }

    #[test]
    fn test_depth() {
        // Single node
        let tree = OrderedTree::with_root(1);
        assert_eq!(tree.depth(), 1);

        // Two levels
        let mut tree = OrderedTree::with_root(1);
        tree.add_child(0, 2);
        assert_eq!(tree.depth(), 2);

        // Three levels
        let mut tree = OrderedTree::with_root(1);
        let n2 = tree.add_child(0, 2).unwrap();
        tree.add_child(n2, 3);
        assert_eq!(tree.depth(), 3);
    }

    #[test]
    fn test_preorder_traversal() {
        // Build tree:
        //       1
        //      / \
        //     2   3
        //    /
        //   4
        let mut tree = OrderedTree::with_root(1);
        let n2 = tree.add_child(0, 2).unwrap();
        tree.add_child(0, 3);
        tree.add_child(n2, 4);

        let preorder = tree.preorder();
        let values: Vec<_> = preorder.iter().map(|&i| *tree.node(i).unwrap().value()).collect();
        assert_eq!(values, vec![1, 2, 4, 3]);
    }

    #[test]
    fn test_postorder_traversal() {
        // Build tree:
        //       1
        //      / \
        //     2   3
        //    /
        //   4
        let mut tree = OrderedTree::with_root(1);
        let n2 = tree.add_child(0, 2).unwrap();
        tree.add_child(0, 3);
        tree.add_child(n2, 4);

        let postorder = tree.postorder();
        let values: Vec<_> = postorder.iter().map(|&i| *tree.node(i).unwrap().value()).collect();
        assert_eq!(values, vec![4, 2, 3, 1]);
    }

    #[test]
    fn test_level_order_traversal() {
        // Build tree:
        //       1
        //      / \
        //     2   3
        //    /   / \
        //   4   5   6
        let mut tree = OrderedTree::with_root(1);
        let n2 = tree.add_child(0, 2).unwrap();
        let n3 = tree.add_child(0, 3).unwrap();
        tree.add_child(n2, 4);
        tree.add_child(n3, 5);
        tree.add_child(n3, 6);

        let level_order = tree.level_order();
        let values: Vec<_> = level_order.iter().map(|&i| *tree.node(i).unwrap().value()).collect();
        assert_eq!(values, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_leaves() {
        // Build tree:
        //       1
        //      / \
        //     2   3
        //    /   / \
        //   4   5   6
        let mut tree = OrderedTree::with_root(1);
        let n2 = tree.add_child(0, 2).unwrap();
        let n3 = tree.add_child(0, 3).unwrap();
        tree.add_child(n2, 4);
        tree.add_child(n3, 5);
        tree.add_child(n3, 6);

        let leaves = tree.leaves();
        let leaf_values: Vec<_> = leaves.iter().map(|&i| *tree.node(i).unwrap().value()).collect();
        assert_eq!(leaf_values, vec![4, 5, 6]);
    }

    #[test]
    fn test_parenthesis_representation() {
        // Build tree:
        //       a
        //      / \
        //     b   c
        //    /
        //   d
        let mut tree = OrderedTree::with_root('a');
        let nb = tree.add_child(0, 'b').unwrap();
        tree.add_child(0, 'c');
        tree.add_child(nb, 'd');

        assert_eq!(tree.to_parenthesis(), "a(b(d),c)");
    }

    #[test]
    fn test_iterator() {
        // Build tree:
        //       1
        //      / \
        //     2   3
        let mut tree = OrderedTree::with_root(1);
        tree.add_child(0, 2);
        tree.add_child(0, 3);

        let values: Vec<_> = tree.iter_preorder().map(|(_, &v)| v).collect();
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn test_parent_links() {
        let mut tree = OrderedTree::with_root(1);
        let n2 = tree.add_child(0, 2).unwrap();
        let n3 = tree.add_child(0, 3).unwrap();

        assert_eq!(tree.node(0).unwrap().parent(), None);
        assert_eq!(tree.node(n2).unwrap().parent(), Some(0));
        assert_eq!(tree.node(n3).unwrap().parent(), Some(0));
    }

    #[test]
    fn test_sibling_links() {
        let mut tree = OrderedTree::with_root(1);
        let n2 = tree.add_child(0, 2).unwrap();
        let n3 = tree.add_child(0, 3).unwrap();
        let n4 = tree.add_child(0, 4).unwrap();

        // Check sibling links
        assert_eq!(tree.node(n2).unwrap().right_sibling(), Some(n3));
        assert_eq!(tree.node(n3).unwrap().right_sibling(), Some(n4));
        assert_eq!(tree.node(n4).unwrap().right_sibling(), None);
    }

    #[test]
    fn test_lcrs_structure() {
        // Build tree:
        //       1
        //      /|\
        //     2 3 4
        //    /
        //   5
        let mut tree = OrderedTree::with_root(1);
        let n2 = tree.add_child(0, 2).unwrap();
        let n3 = tree.add_child(0, 3).unwrap();
        let n4 = tree.add_child(0, 4).unwrap();
        let n5 = tree.add_child(n2, 5).unwrap();

        // Root's left child should be first child (2)
        assert_eq!(tree.node(0).unwrap().left_child(), Some(n2));

        // Children should be linked as siblings
        assert_eq!(tree.node(n2).unwrap().right_sibling(), Some(n3));
        assert_eq!(tree.node(n3).unwrap().right_sibling(), Some(n4));
        assert_eq!(tree.node(n4).unwrap().right_sibling(), None);

        // Node 2's left child should be 5
        assert_eq!(tree.node(n2).unwrap().left_child(), Some(n5));
        assert_eq!(tree.node(n5).unwrap().right_sibling(), None);
    }
}
