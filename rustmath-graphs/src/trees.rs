//! Tree generation and iteration
//!
//! This module provides functionality for generating and iterating over
//! all labeled trees of a given size. Trees are fundamental structures
//! in graph theory.

use crate::graph::Graph;

/// Iterator over all labeled trees with n vertices
///
/// `TreeIterator` generates all non-isomorphic labeled trees with a specified
/// number of vertices. It uses Prüfer sequences to enumerate trees efficiently.
///
/// A tree with n vertices has n-2 elements in its Prüfer sequence, and there
/// are n^(n-2) distinct labeled trees on n vertices (Cayley's formula).
///
/// # Examples
///
/// ```
/// use rustmath_graphs::trees::TreeIterator;
///
/// // Iterate over all trees with 4 vertices
/// let iter = TreeIterator::new(4);
/// let trees: Vec<_> = iter.collect();
///
/// // There are 4^(4-2) = 16 labeled trees with 4 vertices
/// assert_eq!(trees.len(), 16);
/// ```
pub struct TreeIterator {
    n: usize,
    current: Vec<usize>,
    done: bool,
}

impl TreeIterator {
    /// Create a new tree iterator for trees with n vertices
    ///
    /// # Arguments
    ///
    /// * `n` - Number of vertices in the trees to generate
    ///
    /// # Returns
    ///
    /// A new `TreeIterator` instance
    ///
    /// # Panics
    ///
    /// Panics if n < 2, as trees must have at least 2 vertices
    pub fn new(n: usize) -> Self {
        assert!(n >= 2, "Trees must have at least 2 vertices");

        TreeIterator {
            n,
            current: vec![0; n.saturating_sub(2)],
            done: false,
        }
    }

    /// Get the number of vertices in the trees being generated
    pub fn num_vertices(&self) -> usize {
        self.n
    }

    /// Convert a Prüfer sequence to a tree (graph)
    ///
    /// The Prüfer sequence uniquely identifies a labeled tree.
    /// This method reconstructs the tree from the sequence.
    fn prufer_to_tree(sequence: &[usize], n: usize) -> Graph {
        let mut graph = Graph::new(n);

        if n < 2 {
            return graph;
        }

        if n == 2 {
            graph.add_edge(0, 1).ok();
            return graph;
        }

        // Count degree of each vertex (starts at 1, add 1 for each appearance in sequence)
        let mut degree = vec![1; n];
        for &v in sequence {
            degree[v] += 1;
        }

        // Build tree by repeatedly connecting lowest degree-1 vertex to next sequence element
        for &v in sequence {
            // Find the smallest leaf (degree 1 vertex)
            for i in 0..n {
                if degree[i] == 1 {
                    graph.add_edge(i, v).ok();
                    degree[i] -= 1;
                    degree[v] -= 1;
                    break;
                }
            }
        }

        // Connect the last two remaining degree-1 vertices
        let mut remaining = Vec::new();
        for i in 0..n {
            if degree[i] == 1 {
                remaining.push(i);
            }
        }

        if remaining.len() == 2 {
            graph.add_edge(remaining[0], remaining[1]).ok();
        }

        graph
    }

    /// Increment the Prüfer sequence to the next one (like incrementing a base-n number)
    fn increment_sequence(&mut self) -> bool {
        let n = self.n;
        let len = self.current.len();

        if len == 0 {
            return false;
        }

        // Increment from rightmost position
        let mut pos = len - 1;
        loop {
            self.current[pos] += 1;

            if self.current[pos] < n {
                return true;
            }

            // Carry over
            self.current[pos] = 0;

            if pos == 0 {
                return false; // Overflow - we've generated all sequences
            }

            pos -= 1;
        }
    }
}

impl Iterator for TreeIterator {
    type Item = Graph;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        // Generate tree from current Prüfer sequence
        let tree = Self::prufer_to_tree(&self.current, self.n);

        // Move to next sequence
        if !self.increment_sequence() {
            self.done = true;
        }

        Some(tree)
    }
}

/// Count the number of labeled trees with n vertices
///
/// Uses Cayley's formula: n^(n-2)
///
/// # Arguments
///
/// * `n` - Number of vertices
///
/// # Returns
///
/// The number of distinct labeled trees with n vertices
///
/// # Examples
///
/// ```
/// use rustmath_graphs::trees::count_labeled_trees;
///
/// assert_eq!(count_labeled_trees(1), 1);
/// assert_eq!(count_labeled_trees(2), 1);
/// assert_eq!(count_labeled_trees(3), 3);
/// assert_eq!(count_labeled_trees(4), 16);
/// assert_eq!(count_labeled_trees(5), 125);
/// ```
pub fn count_labeled_trees(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }

    // Cayley's formula: n^(n-2)
    n.pow((n - 2) as u32)
}

/// A rooted tree with parent pointers and depth tracking
///
/// `RootedTree` represents a tree with a designated root vertex,
/// storing parent relationships and depth information for efficient
/// tree operations and queries.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::trees::RootedTree;
///
/// // Create a tree with 5 vertices, rooted at vertex 0
/// let mut tree = RootedTree::new(5, 0);
/// tree.add_child(0, 1).unwrap();
/// tree.add_child(0, 2).unwrap();
/// tree.add_child(1, 3).unwrap();
/// tree.add_child(1, 4).unwrap();
///
/// assert_eq!(tree.depth(3), Some(2));
/// assert_eq!(tree.children(1), Some(vec![3, 4]));
/// ```
#[derive(Debug, Clone)]
pub struct RootedTree {
    /// Number of vertices in the tree
    n: usize,
    /// Root vertex of the tree
    root: usize,
    /// Parent of each vertex (None for root)
    parent: Vec<Option<usize>>,
    /// Depth of each vertex (distance from root)
    depth: Vec<usize>,
    /// Children of each vertex
    children: Vec<Vec<usize>>,
}

impl RootedTree {
    /// Create a new rooted tree with n vertices
    ///
    /// # Arguments
    ///
    /// * `n` - Number of vertices
    /// * `root` - The vertex to use as the root
    ///
    /// # Returns
    ///
    /// A new `RootedTree` with only the root vertex initialized
    ///
    /// # Panics
    ///
    /// Panics if root >= n
    pub fn new(n: usize, root: usize) -> Self {
        assert!(root < n, "Root vertex must be in range [0, n)");

        let mut parent = vec![None; n];
        let mut depth = vec![0; n];
        let children = vec![Vec::new(); n];

        // Root has no parent and depth 0
        parent[root] = None;
        depth[root] = 0;

        RootedTree {
            n,
            root,
            parent,
            depth,
            children,
        }
    }

    /// Create a rooted tree from an undirected tree (Graph)
    ///
    /// Performs BFS from the specified root to build parent-child relationships
    ///
    /// # Arguments
    ///
    /// * `graph` - The undirected tree as a Graph
    /// * `root` - The vertex to use as the root
    ///
    /// # Returns
    ///
    /// A new `RootedTree` or an error if the graph is not a tree
    pub fn from_graph(graph: &Graph, root: usize) -> Result<Self, String> {
        let n = graph.num_vertices();

        if root >= n {
            return Err("Root vertex out of bounds".to_string());
        }

        // Check that graph is a tree (n-1 edges and connected)
        if graph.num_edges() != n.saturating_sub(1) {
            return Err("Graph is not a tree (wrong number of edges)".to_string());
        }

        let mut tree = RootedTree::new(n, root);
        let mut visited = vec![false; n];
        let mut queue = std::collections::VecDeque::new();

        queue.push_back(root);
        visited[root] = true;

        while let Some(v) = queue.pop_front() {
            if let Some(neighbors) = graph.neighbors(v) {
                for &neighbor in &neighbors {
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        tree.parent[neighbor] = Some(v);
                        tree.depth[neighbor] = tree.depth[v] + 1;
                        tree.children[v].push(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        // Check that all vertices were visited (connected)
        if visited.iter().any(|&v| !v) {
            return Err("Graph is not connected".to_string());
        }

        Ok(tree)
    }

    /// Add a child to a parent vertex
    ///
    /// # Arguments
    ///
    /// * `parent` - The parent vertex
    /// * `child` - The child vertex to add
    ///
    /// # Returns
    ///
    /// Ok(()) on success, or an error message
    pub fn add_child(&mut self, parent_vertex: usize, child: usize) -> Result<(), String> {
        if parent_vertex >= self.n || child >= self.n {
            return Err("Vertex out of bounds".to_string());
        }

        if self.parent[child].is_some() {
            return Err("Child already has a parent".to_string());
        }

        self.parent[child] = Some(parent_vertex);
        self.depth[child] = self.depth[parent_vertex] + 1;
        self.children[parent_vertex].push(child);

        Ok(())
    }

    /// Get the root vertex
    pub fn root(&self) -> usize {
        self.root
    }

    /// Get the number of vertices
    pub fn num_vertices(&self) -> usize {
        self.n
    }

    /// Get the parent of a vertex
    ///
    /// Returns None for the root, Some(parent) for other vertices
    pub fn parent(&self, v: usize) -> Option<Option<usize>> {
        if v >= self.n {
            return None;
        }
        Some(self.parent[v])
    }

    /// Get the depth of a vertex (distance from root)
    pub fn depth(&self, v: usize) -> Option<usize> {
        if v >= self.n {
            return None;
        }
        Some(self.depth[v])
    }

    /// Get the children of a vertex
    pub fn children(&self, v: usize) -> Option<Vec<usize>> {
        if v >= self.n {
            return None;
        }
        Some(self.children[v].clone())
    }

    /// Check if vertex u is an ancestor of vertex v
    pub fn is_ancestor(&self, u: usize, v: usize) -> bool {
        if u >= self.n || v >= self.n {
            return false;
        }

        let mut current = v;
        while let Some(p) = self.parent[current] {
            if p == u {
                return true;
            }
            current = p;
        }

        u == v // A vertex is considered its own ancestor
    }

    /// Get all ancestors of a vertex (from parent to root)
    ///
    /// Returns the path from the vertex to the root (excluding the vertex itself)
    pub fn ancestors(&self, v: usize) -> Option<Vec<usize>> {
        if v >= self.n {
            return None;
        }

        let mut result = Vec::new();
        let mut current = v;

        while let Some(p) = self.parent[current] {
            result.push(p);
            current = p;
        }

        Some(result)
    }

    /// Get the path from the root to a vertex
    pub fn path_from_root(&self, v: usize) -> Option<Vec<usize>> {
        if v >= self.n {
            return None;
        }

        let mut path = vec![v];
        let mut current = v;

        while let Some(p) = self.parent[current] {
            path.push(p);
            current = p;
        }

        path.reverse();
        Some(path)
    }

    /// Find the lowest common ancestor (LCA) of two vertices
    pub fn lowest_common_ancestor(&self, u: usize, v: usize) -> Option<usize> {
        if u >= self.n || v >= self.n {
            return None;
        }

        let path_u = self.path_from_root(u)?;
        let path_v = self.path_from_root(v)?;

        let mut lca = self.root;
        for i in 0..path_u.len().min(path_v.len()) {
            if path_u[i] == path_v[i] {
                lca = path_u[i];
            } else {
                break;
            }
        }

        Some(lca)
    }

    /// Get the size of the subtree rooted at vertex v
    ///
    /// Returns the number of vertices in the subtree (including v itself)
    pub fn subtree_size(&self, v: usize) -> Option<usize> {
        if v >= self.n {
            return None;
        }

        let mut size = 1; // Count the vertex itself

        for &child in &self.children[v] {
            if let Some(child_size) = self.subtree_size(child) {
                size += child_size;
            }
        }

        Some(size)
    }

    /// Get all vertices in the subtree rooted at v (including v)
    ///
    /// Returns vertices in pre-order traversal
    pub fn subtree_vertices(&self, v: usize) -> Option<Vec<usize>> {
        if v >= self.n {
            return None;
        }

        let mut result = vec![v];

        for &child in &self.children[v] {
            if let Some(mut child_vertices) = self.subtree_vertices(child) {
                result.append(&mut child_vertices);
            }
        }

        Some(result)
    }

    /// Check if the vertex is a leaf (has no children)
    pub fn is_leaf(&self, v: usize) -> Option<bool> {
        if v >= self.n {
            return None;
        }
        Some(self.children[v].is_empty())
    }

    /// Get all leaf vertices in the tree
    pub fn leaves(&self) -> Vec<usize> {
        (0..self.n)
            .filter(|&v| self.children[v].is_empty())
            .collect()
    }

    /// Get the height of the tree (maximum depth)
    pub fn height(&self) -> usize {
        *self.depth.iter().max().unwrap_or(&0)
    }

    /// Get all vertices at a given depth
    pub fn vertices_at_depth(&self, d: usize) -> Vec<usize> {
        (0..self.n)
            .filter(|&v| self.depth[v] == d)
            .collect()
    }

    /// Compute the distance between two vertices
    ///
    /// Returns the number of edges in the path between u and v
    pub fn distance(&self, u: usize, v: usize) -> Option<usize> {
        if u >= self.n || v >= self.n {
            return None;
        }

        let lca = self.lowest_common_ancestor(u, v)?;
        let dist = self.depth[u] + self.depth[v] - 2 * self.depth[lca];

        Some(dist)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_iterator_basic() {
        let iter = TreeIterator::new(3);
        let trees: Vec<_> = iter.collect();

        // There are 3^(3-2) = 3 labeled trees with 3 vertices
        assert_eq!(trees.len(), 3);

        // All should be valid trees
        for tree in &trees {
            assert_eq!(tree.num_vertices(), 3);
            assert_eq!(tree.num_edges(), 2); // Tree with 3 vertices has 2 edges
        }
    }

    #[test]
    fn test_tree_iterator_4_vertices() {
        let iter = TreeIterator::new(4);
        let trees: Vec<_> = iter.collect();

        // There are 4^(4-2) = 16 labeled trees with 4 vertices
        assert_eq!(trees.len(), 16);

        for tree in &trees {
            assert_eq!(tree.num_vertices(), 4);
            assert_eq!(tree.num_edges(), 3); // Tree with 4 vertices has 3 edges
        }
    }

    #[test]
    fn test_tree_iterator_2_vertices() {
        let iter = TreeIterator::new(2);
        let trees: Vec<_> = iter.collect();

        // Only one tree with 2 vertices (an edge)
        assert_eq!(trees.len(), 1);
        assert_eq!(trees[0].num_vertices(), 2);
        assert_eq!(trees[0].num_edges(), 1);
    }

    #[test]
    fn test_num_vertices() {
        let iter = TreeIterator::new(5);
        assert_eq!(iter.num_vertices(), 5);
    }

    #[test]
    fn test_count_labeled_trees() {
        assert_eq!(count_labeled_trees(1), 1);
        assert_eq!(count_labeled_trees(2), 1);
        assert_eq!(count_labeled_trees(3), 3);
        assert_eq!(count_labeled_trees(4), 16);
        assert_eq!(count_labeled_trees(5), 125);
        assert_eq!(count_labeled_trees(6), 1296);
    }

    #[test]
    fn test_prufer_to_tree_basic() {
        // Empty sequence for n=2
        let tree = TreeIterator::prufer_to_tree(&[], 2);
        assert_eq!(tree.num_vertices(), 2);
        assert_eq!(tree.num_edges(), 1);
        assert!(tree.has_edge(0, 1));
    }

    #[test]
    fn test_prufer_to_tree_3_vertices() {
        // Sequence [0] creates a star with center at 0
        let tree = TreeIterator::prufer_to_tree(&[0], 3);
        assert_eq!(tree.num_vertices(), 3);
        assert_eq!(tree.num_edges(), 2);
    }

    #[test]
    fn test_trees_are_connected() {
        let iter = TreeIterator::new(4);

        for tree in iter {
            // Check that tree is connected by verifying it has n-1 edges
            assert_eq!(tree.num_edges(), tree.num_vertices() - 1);
        }
    }

    #[test]
    #[should_panic(expected = "Trees must have at least 2 vertices")]
    fn test_invalid_size() {
        TreeIterator::new(1);
    }

    #[test]
    fn test_iterator_exhaustion() {
        let mut iter = TreeIterator::new(3);

        // Consume all trees
        let count = iter.by_ref().count();
        assert_eq!(count, 3);

        // Iterator should be exhausted
        assert!(iter.next().is_none());
        assert!(iter.next().is_none());
    }

    // RootedTree tests
    #[test]
    fn test_rooted_tree_new() {
        let tree = RootedTree::new(5, 0);
        assert_eq!(tree.root(), 0);
        assert_eq!(tree.num_vertices(), 5);
        assert_eq!(tree.depth(0), Some(0));
        assert_eq!(tree.parent(0), Some(None));
    }

    #[test]
    #[should_panic(expected = "Root vertex must be in range [0, n)")]
    fn test_rooted_tree_invalid_root() {
        RootedTree::new(5, 5);
    }

    #[test]
    fn test_rooted_tree_add_child() {
        let mut tree = RootedTree::new(5, 0);

        tree.add_child(0, 1).unwrap();
        tree.add_child(0, 2).unwrap();
        tree.add_child(1, 3).unwrap();
        tree.add_child(1, 4).unwrap();

        assert_eq!(tree.parent(1), Some(Some(0)));
        assert_eq!(tree.parent(2), Some(Some(0)));
        assert_eq!(tree.parent(3), Some(Some(1)));
        assert_eq!(tree.parent(4), Some(Some(1)));

        assert_eq!(tree.depth(0), Some(0));
        assert_eq!(tree.depth(1), Some(1));
        assert_eq!(tree.depth(2), Some(1));
        assert_eq!(tree.depth(3), Some(2));
        assert_eq!(tree.depth(4), Some(2));

        assert_eq!(tree.children(0), Some(vec![1, 2]));
        assert_eq!(tree.children(1), Some(vec![3, 4]));
        assert_eq!(tree.children(2), Some(vec![]));
    }

    #[test]
    fn test_rooted_tree_add_child_errors() {
        let mut tree = RootedTree::new(5, 0);

        // Out of bounds
        assert!(tree.add_child(0, 10).is_err());
        assert!(tree.add_child(10, 1).is_err());

        // Child already has a parent
        tree.add_child(0, 1).unwrap();
        assert!(tree.add_child(0, 1).is_err());
    }

    #[test]
    fn test_rooted_tree_from_graph() {
        let mut graph = Graph::new(5);
        graph.add_edge(0, 1).unwrap();
        graph.add_edge(0, 2).unwrap();
        graph.add_edge(1, 3).unwrap();
        graph.add_edge(1, 4).unwrap();

        let tree = RootedTree::from_graph(&graph, 0).unwrap();

        assert_eq!(tree.root(), 0);
        assert_eq!(tree.num_vertices(), 5);
        assert_eq!(tree.depth(0), Some(0));
        assert_eq!(tree.depth(1), Some(1));
        assert_eq!(tree.depth(3), Some(2));
    }

    #[test]
    fn test_rooted_tree_from_graph_invalid() {
        // Graph with cycle (not a tree)
        let mut graph = Graph::new(3);
        graph.add_edge(0, 1).unwrap();
        graph.add_edge(1, 2).unwrap();
        graph.add_edge(2, 0).unwrap();

        assert!(RootedTree::from_graph(&graph, 0).is_err());

        // Disconnected graph
        let mut graph = Graph::new(4);
        graph.add_edge(0, 1).unwrap();
        graph.add_edge(2, 3).unwrap();

        assert!(RootedTree::from_graph(&graph, 0).is_err());

        // Root out of bounds
        let graph = Graph::new(3);
        assert!(RootedTree::from_graph(&graph, 5).is_err());
    }

    #[test]
    fn test_rooted_tree_is_ancestor() {
        let mut tree = RootedTree::new(5, 0);
        tree.add_child(0, 1).unwrap();
        tree.add_child(0, 2).unwrap();
        tree.add_child(1, 3).unwrap();
        tree.add_child(1, 4).unwrap();

        // Direct parent-child
        assert!(tree.is_ancestor(0, 1));
        assert!(tree.is_ancestor(1, 3));

        // Transitive ancestor
        assert!(tree.is_ancestor(0, 3));
        assert!(tree.is_ancestor(0, 4));

        // Self is ancestor of self
        assert!(tree.is_ancestor(1, 1));

        // Not ancestors
        assert!(!tree.is_ancestor(1, 0));
        assert!(!tree.is_ancestor(2, 3));
        assert!(!tree.is_ancestor(3, 4));
    }

    #[test]
    fn test_rooted_tree_ancestors() {
        let mut tree = RootedTree::new(5, 0);
        tree.add_child(0, 1).unwrap();
        tree.add_child(0, 2).unwrap();
        tree.add_child(1, 3).unwrap();
        tree.add_child(1, 4).unwrap();

        assert_eq!(tree.ancestors(0), Some(vec![]));
        assert_eq!(tree.ancestors(1), Some(vec![0]));
        assert_eq!(tree.ancestors(3), Some(vec![1, 0]));
        assert_eq!(tree.ancestors(4), Some(vec![1, 0]));
    }

    #[test]
    fn test_rooted_tree_path_from_root() {
        let mut tree = RootedTree::new(5, 0);
        tree.add_child(0, 1).unwrap();
        tree.add_child(0, 2).unwrap();
        tree.add_child(1, 3).unwrap();
        tree.add_child(1, 4).unwrap();

        assert_eq!(tree.path_from_root(0), Some(vec![0]));
        assert_eq!(tree.path_from_root(1), Some(vec![0, 1]));
        assert_eq!(tree.path_from_root(3), Some(vec![0, 1, 3]));
        assert_eq!(tree.path_from_root(2), Some(vec![0, 2]));
    }

    #[test]
    fn test_rooted_tree_lowest_common_ancestor() {
        let mut tree = RootedTree::new(7, 0);
        tree.add_child(0, 1).unwrap();
        tree.add_child(0, 2).unwrap();
        tree.add_child(1, 3).unwrap();
        tree.add_child(1, 4).unwrap();
        tree.add_child(2, 5).unwrap();
        tree.add_child(2, 6).unwrap();

        // LCA of siblings
        assert_eq!(tree.lowest_common_ancestor(3, 4), Some(1));
        assert_eq!(tree.lowest_common_ancestor(5, 6), Some(2));

        // LCA across subtrees
        assert_eq!(tree.lowest_common_ancestor(3, 5), Some(0));
        assert_eq!(tree.lowest_common_ancestor(4, 6), Some(0));

        // LCA with ancestor
        assert_eq!(tree.lowest_common_ancestor(1, 3), Some(1));
        assert_eq!(tree.lowest_common_ancestor(0, 6), Some(0));

        // LCA of same vertex
        assert_eq!(tree.lowest_common_ancestor(3, 3), Some(3));
    }

    #[test]
    fn test_rooted_tree_subtree_size() {
        let mut tree = RootedTree::new(7, 0);
        tree.add_child(0, 1).unwrap();
        tree.add_child(0, 2).unwrap();
        tree.add_child(1, 3).unwrap();
        tree.add_child(1, 4).unwrap();
        tree.add_child(2, 5).unwrap();
        tree.add_child(2, 6).unwrap();

        assert_eq!(tree.subtree_size(0), Some(7)); // Entire tree
        assert_eq!(tree.subtree_size(1), Some(3)); // Vertex 1 and its children
        assert_eq!(tree.subtree_size(2), Some(3)); // Vertex 2 and its children
        assert_eq!(tree.subtree_size(3), Some(1)); // Leaf
        assert_eq!(tree.subtree_size(4), Some(1)); // Leaf
    }

    #[test]
    fn test_rooted_tree_subtree_vertices() {
        let mut tree = RootedTree::new(7, 0);
        tree.add_child(0, 1).unwrap();
        tree.add_child(0, 2).unwrap();
        tree.add_child(1, 3).unwrap();
        tree.add_child(1, 4).unwrap();
        tree.add_child(2, 5).unwrap();
        tree.add_child(2, 6).unwrap();

        let subtree_1 = tree.subtree_vertices(1).unwrap();
        assert_eq!(subtree_1.len(), 3);
        assert!(subtree_1.contains(&1));
        assert!(subtree_1.contains(&3));
        assert!(subtree_1.contains(&4));

        let subtree_2 = tree.subtree_vertices(2).unwrap();
        assert_eq!(subtree_2.len(), 3);
        assert!(subtree_2.contains(&2));
        assert!(subtree_2.contains(&5));
        assert!(subtree_2.contains(&6));

        let subtree_3 = tree.subtree_vertices(3).unwrap();
        assert_eq!(subtree_3, vec![3]);
    }

    #[test]
    fn test_rooted_tree_is_leaf() {
        let mut tree = RootedTree::new(5, 0);
        tree.add_child(0, 1).unwrap();
        tree.add_child(0, 2).unwrap();
        tree.add_child(1, 3).unwrap();
        tree.add_child(1, 4).unwrap();

        assert_eq!(tree.is_leaf(0), Some(false));
        assert_eq!(tree.is_leaf(1), Some(false));
        assert_eq!(tree.is_leaf(2), Some(true));
        assert_eq!(tree.is_leaf(3), Some(true));
        assert_eq!(tree.is_leaf(4), Some(true));
    }

    #[test]
    fn test_rooted_tree_leaves() {
        let mut tree = RootedTree::new(5, 0);
        tree.add_child(0, 1).unwrap();
        tree.add_child(0, 2).unwrap();
        tree.add_child(1, 3).unwrap();
        tree.add_child(1, 4).unwrap();

        let leaves = tree.leaves();
        assert_eq!(leaves.len(), 3);
        assert!(leaves.contains(&2));
        assert!(leaves.contains(&3));
        assert!(leaves.contains(&4));
    }

    #[test]
    fn test_rooted_tree_height() {
        let mut tree = RootedTree::new(5, 0);
        tree.add_child(0, 1).unwrap();
        tree.add_child(0, 2).unwrap();
        tree.add_child(1, 3).unwrap();
        tree.add_child(1, 4).unwrap();

        assert_eq!(tree.height(), 2);

        // Single vertex tree
        let single = RootedTree::new(1, 0);
        assert_eq!(single.height(), 0);

        // Add deeper level
        let mut tree2 = RootedTree::new(6, 0);
        tree2.add_child(0, 1).unwrap();
        tree2.add_child(1, 2).unwrap();
        tree2.add_child(2, 3).unwrap();
        tree2.add_child(3, 4).unwrap();
        tree2.add_child(4, 5).unwrap();
        assert_eq!(tree2.height(), 5);
    }

    #[test]
    fn test_rooted_tree_vertices_at_depth() {
        let mut tree = RootedTree::new(7, 0);
        tree.add_child(0, 1).unwrap();
        tree.add_child(0, 2).unwrap();
        tree.add_child(1, 3).unwrap();
        tree.add_child(1, 4).unwrap();
        tree.add_child(2, 5).unwrap();
        tree.add_child(2, 6).unwrap();

        let depth_0 = tree.vertices_at_depth(0);
        assert_eq!(depth_0, vec![0]);

        let depth_1 = tree.vertices_at_depth(1);
        assert_eq!(depth_1.len(), 2);
        assert!(depth_1.contains(&1));
        assert!(depth_1.contains(&2));

        let depth_2 = tree.vertices_at_depth(2);
        assert_eq!(depth_2.len(), 4);
        assert!(depth_2.contains(&3));
        assert!(depth_2.contains(&4));
        assert!(depth_2.contains(&5));
        assert!(depth_2.contains(&6));

        let depth_3 = tree.vertices_at_depth(3);
        assert_eq!(depth_3.len(), 0);
    }

    #[test]
    fn test_rooted_tree_distance() {
        let mut tree = RootedTree::new(7, 0);
        tree.add_child(0, 1).unwrap();
        tree.add_child(0, 2).unwrap();
        tree.add_child(1, 3).unwrap();
        tree.add_child(1, 4).unwrap();
        tree.add_child(2, 5).unwrap();
        tree.add_child(2, 6).unwrap();

        // Distance between siblings
        assert_eq!(tree.distance(3, 4), Some(2));
        assert_eq!(tree.distance(5, 6), Some(2));

        // Distance across subtrees
        assert_eq!(tree.distance(3, 5), Some(4));
        assert_eq!(tree.distance(4, 6), Some(4));

        // Distance to ancestor
        assert_eq!(tree.distance(0, 3), Some(2));
        assert_eq!(tree.distance(1, 4), Some(1));

        // Distance to self
        assert_eq!(tree.distance(3, 3), Some(0));
    }

    #[test]
    fn test_rooted_tree_from_graph_different_roots() {
        let mut graph = Graph::new(5);
        graph.add_edge(0, 1).unwrap();
        graph.add_edge(0, 2).unwrap();
        graph.add_edge(1, 3).unwrap();
        graph.add_edge(1, 4).unwrap();

        // Root at vertex 0
        let tree0 = RootedTree::from_graph(&graph, 0).unwrap();
        assert_eq!(tree0.depth(4), Some(2));

        // Root at vertex 3 (different structure)
        let tree3 = RootedTree::from_graph(&graph, 3).unwrap();
        assert_eq!(tree3.depth(0), Some(2));
        assert_eq!(tree3.depth(1), Some(1));
        assert_eq!(tree3.depth(4), Some(2));
    }
}
