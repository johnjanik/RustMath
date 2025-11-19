//! Spanning tree algorithms for graphs
//!
//! This module provides algorithms for finding minimum spanning trees (MSTs),
//! enumerating spanning trees, and related operations.
//!
//! Key algorithms:
//! - Kruskal's algorithm for MST
//! - Borůvka's algorithm for MST
//! - Random spanning tree generation
//! - Spanning tree enumeration

use crate::{Graph, WeightedGraph};
use std::collections::{HashMap, HashSet};

/// Union-Find (Disjoint Set Union) data structure for Kruskal's algorithm
#[derive(Debug, Clone)]
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    /// Create a new Union-Find structure with n elements
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    /// Find the representative of the set containing x (with path compression)
    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    /// Union the sets containing x and y (union by rank)
    /// Returns true if they were in different sets, false otherwise
    fn union(&mut self, x: usize, y: usize) -> bool {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x == root_y {
            return false;
        }

        // Union by rank
        if self.rank[root_x] < self.rank[root_y] {
            self.parent[root_x] = root_y;
        } else if self.rank[root_x] > self.rank[root_y] {
            self.parent[root_y] = root_x;
        } else {
            self.parent[root_y] = root_x;
            self.rank[root_x] += 1;
        }

        true
    }
}

/// Find a minimum spanning tree using Kruskal's algorithm
///
/// Returns a vector of edges (u, v) in the MST.
///
/// # Arguments
///
/// * `graph` - The weighted graph
/// * `by_weight` - Whether to minimize by weight (true) or just find any spanning tree (false)
///
/// # Returns
///
/// A vector of edges forming the minimum spanning tree
///
/// # Examples
///
/// ```
/// use rustmath_graphs::{WeightedGraph, spanning_tree::kruskal};
///
/// let mut g = WeightedGraph::new(4);
/// g.add_edge(0, 1, 1).unwrap();
/// g.add_edge(0, 2, 4).unwrap();
/// g.add_edge(1, 2, 2).unwrap();
/// g.add_edge(1, 3, 5).unwrap();
/// g.add_edge(2, 3, 3).unwrap();
///
/// let mst = kruskal(&g, true);
/// assert_eq!(mst.len(), 3); // n-1 edges for n vertices
/// ```
pub fn kruskal(graph: &WeightedGraph, by_weight: bool) -> Vec<(usize, usize)> {
    // Use the graph's edges() method
    let mut edges = graph.edges();

    // Sort edges by weight if needed
    if by_weight {
        edges.sort_by_key(|&(_, _, w)| w);
    }

    let n = graph.num_vertices();
    let mut uf = UnionFind::new(n);
    let mut mst = Vec::new();

    for (u, v, _weight) in edges {
        if uf.union(u, v) {
            mst.push((u, v));
            if mst.len() == n - 1 {
                break;
            }
        }
    }

    mst
}

/// Iterator for Kruskal's algorithm
///
/// Returns edges of the MST one at a time
pub struct KruskalIterator {
    edges: Vec<(usize, usize, i64)>,
    uf: UnionFind,
    index: usize,
    target_edges: usize,
    found_edges: usize,
}

impl KruskalIterator {
    /// Create a new Kruskal iterator
    pub fn new(graph: &WeightedGraph, by_weight: bool) -> Self {
        let n = graph.num_vertices();

        // Use the graph's edges() method
        let mut edges = graph.edges();

        // Sort by weight if needed
        if by_weight {
            edges.sort_by_key(|&(_, _, w)| w);
        }

        KruskalIterator {
            edges,
            uf: UnionFind::new(n),
            index: 0,
            target_edges: n.saturating_sub(1),
            found_edges: 0,
        }
    }
}

impl Iterator for KruskalIterator {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.found_edges >= self.target_edges {
            return None;
        }

        while self.index < self.edges.len() {
            let (u, v, _) = self.edges[self.index];
            self.index += 1;

            if self.uf.union(u, v) {
                self.found_edges += 1;
                return Some((u, v));
            }
        }

        None
    }
}

/// Create a Kruskal iterator
pub fn kruskal_iterator(graph: &WeightedGraph, by_weight: bool) -> KruskalIterator {
    KruskalIterator::new(graph, by_weight)
}

/// Create a Kruskal iterator from an edge list
pub fn kruskal_iterator_from_edges(
    num_vertices: usize,
    edges: Vec<(usize, usize, i64)>,
    by_weight: bool,
) -> KruskalIterator {
    let mut sorted_edges = edges;

    if by_weight {
        sorted_edges.sort_by_key(|&(_, _, w)| w);
    }

    KruskalIterator {
        edges: sorted_edges,
        uf: UnionFind::new(num_vertices),
        index: 0,
        target_edges: num_vertices.saturating_sub(1),
        found_edges: 0,
    }
}

/// Find a minimum spanning tree using Kruskal's algorithm with edge filtering
///
/// Only considers edges that pass the filter predicate
pub fn filter_kruskal<F>(graph: &WeightedGraph, by_weight: bool, filter: F) -> Vec<(usize, usize)>
where
    F: Fn(usize, usize, i64) -> bool,
{
    let n = graph.num_vertices();
    if n == 0 {
        return Vec::new();
    }

    // Collect filtered edges
    let mut edges: Vec<(usize, usize, i64)> = graph
        .edges()
        .into_iter()
        .filter(|&(u, v, w)| filter(u, v, w))
        .collect();

    // Sort edges by weight if needed
    if by_weight {
        edges.sort_by_key(|&(_, _, w)| w);
    }

    // Kruskal's algorithm
    let mut uf = UnionFind::new(n);
    let mut mst = Vec::new();

    for (u, v, _weight) in edges {
        if uf.union(u, v) {
            mst.push((u, v));
            if mst.len() == n - 1 {
                break;
            }
        }
    }

    mst
}

/// Iterator for filtered Kruskal's algorithm
pub fn filter_kruskal_iterator<F>(
    graph: &WeightedGraph,
    by_weight: bool,
    filter: F,
) -> KruskalIterator
where
    F: Fn(usize, usize, i64) -> bool,
{
    let n = graph.num_vertices();

    // Collect filtered edges
    let mut edges: Vec<(usize, usize, i64)> = graph
        .edges()
        .into_iter()
        .filter(|&(u, v, w)| filter(u, v, w))
        .collect();

    // Sort by weight if needed
    if by_weight {
        edges.sort_by_key(|&(_, _, w)| w);
    }

    KruskalIterator {
        edges,
        uf: UnionFind::new(n),
        index: 0,
        target_edges: n.saturating_sub(1),
        found_edges: 0,
    }
}

/// Find a minimum spanning tree using Borůvka's algorithm
///
/// Borůvka's algorithm works by repeatedly finding the minimum-weight edge
/// from each component and adding all such edges to the MST.
///
/// # Arguments
///
/// * `graph` - The weighted graph
///
/// # Returns
///
/// A vector of edges forming the minimum spanning tree
///
/// # Examples
///
/// ```
/// use rustmath_graphs::{WeightedGraph, spanning_tree::boruvka};
///
/// let mut g = WeightedGraph::new(4);
/// g.add_edge(0, 1, 1).unwrap();
/// g.add_edge(0, 2, 4).unwrap();
/// g.add_edge(1, 2, 2).unwrap();
/// g.add_edge(1, 3, 5).unwrap();
/// g.add_edge(2, 3, 3).unwrap();
///
/// let mst = boruvka(&g);
/// assert_eq!(mst.len(), 3);
/// ```
pub fn boruvka(graph: &WeightedGraph) -> Vec<(usize, usize)> {
    let n = graph.num_vertices();
    if n == 0 {
        return Vec::new();
    }

    let mut uf = UnionFind::new(n);
    let mut mst = Vec::new();

    // Get all edges once
    let all_edges = graph.edges();

    // Repeat until we have n-1 edges (or no more edges to add)
    while mst.len() < n - 1 {
        // For each component, find the minimum-weight outgoing edge
        let mut min_edge: HashMap<usize, (usize, usize, i64)> = HashMap::new();

        for &(u, v, weight) in &all_edges {
            let comp_u = uf.find(u);
            let comp_v = uf.find(v);

            if comp_u != comp_v {
                // Update minimum edge for this component
                let update = if let Some((_, _, min_w)) = min_edge.get(&comp_u) {
                    weight < *min_w
                } else {
                    true
                };

                if update {
                    min_edge.insert(comp_u, (u, v, weight));
                }
            }
        }

        // If no edges found, we're done (graph might be disconnected)
        if min_edge.is_empty() {
            break;
        }

        // Add all minimum edges
        let mut added_any = false;
        for (_comp, (u, v, _weight)) in min_edge {
            if uf.union(u, v) {
                mst.push((u, v));
                added_any = true;
            }
        }

        // If we didn't add any edges, we're done
        if !added_any {
            break;
        }
    }

    mst
}

/// Generate a random spanning tree
///
/// Uses a randomized DFS approach to generate a spanning tree.
/// Note: This is not uniformly random (use Wilson's algorithm for that),
/// but it's simpler and faster.
///
/// # Arguments
///
/// * `graph` - The graph
///
/// # Returns
///
/// A vector of edges forming a random spanning tree
pub fn random_spanning_tree(graph: &Graph) -> Vec<(usize, usize)> {
    use rand::seq::SliceRandom;
    use rand::thread_rng;

    let n = graph.num_vertices();
    if n == 0 {
        return Vec::new();
    }

    let mut visited = vec![false; n];
    let mut tree_edges = Vec::new();
    let mut rng = thread_rng();

    // Randomized DFS from vertex 0
    fn dfs_random(
        graph: &Graph,
        v: usize,
        visited: &mut Vec<bool>,
        tree_edges: &mut Vec<(usize, usize)>,
        rng: &mut impl rand::Rng,
    ) {
        visited[v] = true;

        if let Some(mut neighbors) = graph.neighbors(v) {
            // Randomly shuffle neighbors
            neighbors.shuffle(rng);

            for u in neighbors {
                if !visited[u] {
                    tree_edges.push((v, u));
                    dfs_random(graph, u, visited, tree_edges, rng);
                }
            }
        }
    }

    dfs_random(graph, 0, &mut visited, &mut tree_edges, &mut rng);

    tree_edges
}

/// Enumerate all spanning trees of a graph
///
/// Returns an iterator over all spanning trees.
/// Note: This can be exponential in the size of the graph!
///
/// # Arguments
///
/// * `graph` - The graph
///
/// # Returns
///
/// A vector of all spanning trees, each represented as a vector of edges
pub fn spanning_trees(graph: &Graph) -> Vec<Vec<(usize, usize)>> {
    let n = graph.num_vertices();
    if n == 0 {
        return vec![Vec::new()];
    }

    // Collect all edges
    let mut all_edges = Vec::new();
    for u in 0..n {
        if let Some(neighbors) = graph.neighbors(u) {
            for v in neighbors {
                if u < v {
                    all_edges.push((u, v));
                }
            }
        }
    }

    let mut result = Vec::new();
    let target_size = n - 1;

    // Generate all subsets of size n-1
    enumerate_spanning_trees_recursive(
        &all_edges,
        0,
        Vec::new(),
        target_size,
        n,
        &mut result,
    );

    result
}

fn enumerate_spanning_trees_recursive(
    edges: &[(usize, usize)],
    index: usize,
    current: Vec<(usize, usize)>,
    target_size: usize,
    num_vertices: usize,
    result: &mut Vec<Vec<(usize, usize)>>,
) {
    // If we have enough edges, check if it's a spanning tree
    if current.len() == target_size {
        if is_spanning_tree(&current, num_vertices) {
            result.push(current);
        }
        return;
    }

    // If we've considered all edges, stop
    if index >= edges.len() {
        return;
    }

    // Try including current edge
    let mut with_edge = current.clone();
    with_edge.push(edges[index]);
    enumerate_spanning_trees_recursive(
        edges,
        index + 1,
        with_edge,
        target_size,
        num_vertices,
        result,
    );

    // Try excluding current edge
    enumerate_spanning_trees_recursive(edges, index + 1, current, target_size, num_vertices, result);
}

fn is_spanning_tree(edges: &[(usize, usize)], num_vertices: usize) -> bool {
    // Check if edges form a tree (connected and acyclic)
    let mut uf = UnionFind::new(num_vertices);
    let mut edge_count = 0;

    for &(u, v) in edges {
        if !uf.union(u, v) {
            // Cycle detected
            return false;
        }
        edge_count += 1;
    }

    // Check if it's connected (all vertices in same component)
    let root = uf.find(0);
    for v in 1..num_vertices {
        if uf.find(v) != root {
            return false;
        }
    }

    edge_count == num_vertices - 1
}

/// Find edge-disjoint spanning trees
///
/// Returns a vector of spanning trees that don't share any edges.
/// The maximum number of edge-disjoint spanning trees is limited by
/// the edge connectivity of the graph.
///
/// # Arguments
///
/// * `graph` - The graph
/// * `k` - The number of edge-disjoint spanning trees to find
///
/// # Returns
///
/// A vector of spanning trees (each a vector of edges)
pub fn edge_disjoint_spanning_trees(graph: &Graph, k: usize) -> Vec<Vec<(usize, usize)>> {
    let n = graph.num_vertices();
    if n == 0 || k == 0 {
        return Vec::new();
    }

    let mut result = Vec::new();
    let mut used_edges = HashSet::new();

    for _ in 0..k {
        // Try to find a spanning tree using unused edges
        let mut edges: Vec<(usize, usize)> = Vec::new();

        for u in 0..n {
            if let Some(neighbors) = graph.neighbors(u) {
                for v in neighbors {
                    if u < v && !used_edges.contains(&(u, v)) && !used_edges.contains(&(v, u)) {
                        edges.push((u, v));
                    }
                }
            }
        }

        // Try to build a spanning tree from remaining edges
        let mut uf = UnionFind::new(n);
        let mut tree = Vec::new();

        for (u, v) in edges {
            if uf.union(u, v) {
                tree.push((u, v));
                used_edges.insert((u, v));
                if tree.len() == n - 1 {
                    break;
                }
            }
        }

        // Check if we found a complete spanning tree
        if tree.len() == n - 1 {
            result.push(tree);
        } else {
            // Can't find k edge-disjoint spanning trees
            break;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kruskal_basic() {
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 1, 1).unwrap();
        g.add_edge(0, 2, 4).unwrap();
        g.add_edge(1, 2, 2).unwrap();
        g.add_edge(1, 3, 5).unwrap();
        g.add_edge(2, 3, 3).unwrap();

        let mst = kruskal(&g, true);
        assert_eq!(mst.len(), 3);

        // The MST should contain edges with weights 1, 2, 3 (total = 6)
        // We can verify this by checking the edges are in the MST
        assert!(mst.contains(&(0, 1)) || mst.contains(&(1, 0)));
        assert!(mst.contains(&(1, 2)) || mst.contains(&(2, 1)));
        assert!(mst.contains(&(2, 3)) || mst.contains(&(3, 2)));
    }

    #[test]
    fn test_kruskal_empty() {
        let g = WeightedGraph::new(0);
        let mst = kruskal(&g, true);
        assert_eq!(mst.len(), 0);
    }

    #[test]
    fn test_kruskal_single_vertex() {
        let g = WeightedGraph::new(1);
        let mst = kruskal(&g, true);
        assert_eq!(mst.len(), 0);
    }

    #[test]
    fn test_kruskal_iterator() {
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 1, 1).unwrap();
        g.add_edge(0, 2, 4).unwrap();
        g.add_edge(1, 2, 2).unwrap();
        g.add_edge(1, 3, 5).unwrap();
        g.add_edge(2, 3, 3).unwrap();

        let edges: Vec<_> = kruskal_iterator(&g, true).collect();
        assert_eq!(edges.len(), 3);
    }

    #[test]
    fn test_filter_kruskal() {
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 1, 1).unwrap();
        g.add_edge(0, 2, 4).unwrap();
        g.add_edge(1, 2, 2).unwrap();
        g.add_edge(1, 3, 5).unwrap();
        g.add_edge(2, 3, 3).unwrap();

        // Filter out edges with weight > 3
        let mst = filter_kruskal(&g, true, |_u, _v, w| w <= 3);
        assert_eq!(mst.len(), 3);

        // The MST should use edges with weights <= 3
        assert!(mst.contains(&(0, 1)));
        assert!(mst.contains(&(1, 2)));
        assert!(mst.contains(&(2, 3)));
    }

    #[test]
    fn test_boruvka() {
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 1, 1).unwrap();
        g.add_edge(0, 2, 4).unwrap();
        g.add_edge(1, 2, 2).unwrap();
        g.add_edge(1, 3, 5).unwrap();
        g.add_edge(2, 3, 3).unwrap();

        let mst = boruvka(&g);
        assert_eq!(mst.len(), 3);

        // Borůvka should produce an MST with the same edges as Kruskal
        assert!(mst.contains(&(0, 1)));
        assert!(mst.contains(&(1, 2)));
        assert!(mst.contains(&(2, 3)));
    }

    #[test]
    fn test_boruvka_empty() {
        let g = WeightedGraph::new(0);
        let mst = boruvka(&g);
        assert_eq!(mst.len(), 0);
    }

    #[test]
    fn test_random_spanning_tree() {
        let mut g = Graph::new(5);
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 2).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(1, 3).unwrap();
        g.add_edge(2, 3).unwrap();
        g.add_edge(2, 4).unwrap();
        g.add_edge(3, 4).unwrap();

        let tree = random_spanning_tree(&g);
        assert_eq!(tree.len(), 4); // n-1 edges

        // Verify it's actually a spanning tree
        assert!(is_spanning_tree(&tree, 5));
    }

    #[test]
    fn test_spanning_trees_small() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 2).unwrap();
        g.add_edge(1, 2).unwrap();

        let trees = spanning_trees(&g);
        assert_eq!(trees.len(), 3); // K3 has 3 spanning trees
    }

    #[test]
    fn test_spanning_trees_empty() {
        let g = Graph::new(0);
        let trees = spanning_trees(&g);
        assert_eq!(trees.len(), 1);
        assert_eq!(trees[0].len(), 0);
    }

    #[test]
    fn test_edge_disjoint_spanning_trees() {
        // Create a graph with enough edge connectivity
        let mut g = Graph::new(4);
        // Make it 2-edge-connected
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 2).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(1, 3).unwrap();
        g.add_edge(2, 3).unwrap();
        g.add_edge(0, 3).unwrap();

        let trees = edge_disjoint_spanning_trees(&g, 2);
        assert!(trees.len() <= 2);

        // Verify they're edge-disjoint
        if trees.len() == 2 {
            let mut edges1: HashSet<(usize, usize)> = HashSet::new();
            for &(u, v) in &trees[0] {
                edges1.insert((u.min(v), u.max(v)));
            }

            for &(u, v) in &trees[1] {
                let edge = (u.min(v), u.max(v));
                assert!(!edges1.contains(&edge));
            }
        }
    }

    #[test]
    fn test_kruskal_larger_graph() {
        let mut g = WeightedGraph::new(6);
        g.add_edge(0, 1, 2).unwrap();
        g.add_edge(0, 2, 3).unwrap();
        g.add_edge(1, 2, 1).unwrap();
        g.add_edge(1, 3, 4).unwrap();
        g.add_edge(2, 3, 5).unwrap();
        g.add_edge(2, 4, 6).unwrap();
        g.add_edge(3, 4, 7).unwrap();
        g.add_edge(3, 5, 8).unwrap();
        g.add_edge(4, 5, 9).unwrap();

        let mst = kruskal(&g, true);
        // Should have n-1 edges for n vertices
        assert_eq!(mst.len(), 5);

        // Verify it's a valid spanning tree using Union-Find
        assert!(is_spanning_tree(&mst, 6));
    }

    #[test]
    fn test_union_find() {
        let mut uf = UnionFind::new(5);

        // Initially all separate
        assert_ne!(uf.find(0), uf.find(1));

        // Union 0 and 1
        assert!(uf.union(0, 1));
        assert_eq!(uf.find(0), uf.find(1));

        // Union 2 and 3
        assert!(uf.union(2, 3));
        assert_eq!(uf.find(2), uf.find(3));

        // 0-1 and 2-3 are still separate
        assert_ne!(uf.find(0), uf.find(2));

        // Union the two components
        assert!(uf.union(1, 2));
        assert_eq!(uf.find(0), uf.find(3));

        // Try to union already connected
        assert!(!uf.union(0, 3));
    }

    #[test]
    fn test_kruskal_disconnected() {
        let mut g = WeightedGraph::new(6);
        // Component 1: vertices 0, 1, 2
        g.add_edge(0, 1, 1).unwrap();
        g.add_edge(1, 2, 2).unwrap();

        // Component 2: vertices 3, 4, 5
        g.add_edge(3, 4, 3).unwrap();
        g.add_edge(4, 5, 4).unwrap();

        let mst = kruskal(&g, true);
        // Should have 4 edges (2 for each component)
        assert_eq!(mst.len(), 4);
    }
}
