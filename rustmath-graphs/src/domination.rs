//! Graph domination theory
//!
//! This module provides functions for analyzing dominating sets in graphs.
//! A dominating set is a subset D of vertices such that every vertex is either
//! in D or adjacent to a vertex in D.

use std::collections::{HashSet, VecDeque};
use crate::graph::Graph;

/// Check if a set of vertices is a dominating set
///
/// A set D is dominating if every vertex not in D is adjacent to at least one vertex in D.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `vertices` - The set of vertices to check
///
/// # Returns
/// `true` if the set is dominating
///
/// # Examples
/// ```
/// use rustmath_graphs::graph::Graph;
/// use rustmath_graphs::domination::is_dominating;
///
/// let mut g = Graph::new(4);
/// g.add_edge(0, 1).unwrap();
/// g.add_edge(1, 2).unwrap();
/// g.add_edge(2, 3).unwrap();
///
/// // Vertex 1 and 2 dominate the path
/// assert!(is_dominating(&g, &[1, 2]));
/// ```
pub fn is_dominating(graph: &Graph, vertices: &[usize]) -> bool {
    let n = graph.num_vertices();
    let vertex_set: HashSet<usize> = vertices.iter().copied().collect();

    for v in 0..n {
        if vertex_set.contains(&v) {
            continue;
        }

        // Check if v has a neighbor in the dominating set
        let neighbors = graph.neighbors(v).unwrap_or_default();
        let has_dominating_neighbor = neighbors.iter().any(|n| vertex_set.contains(n));

        if !has_dominating_neighbor {
            return false;
        }
    }

    true
}

/// Check if a vertex is redundant in a dominating set
///
/// A vertex v is redundant in dominating set D if D \ {v} is still dominating.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `dominating_set` - The dominating set
/// * `vertex` - The vertex to check for redundancy
///
/// # Returns
/// `true` if the vertex is redundant
pub fn is_redundant(graph: &Graph, dominating_set: &[usize], vertex: usize) -> bool {
    if !dominating_set.contains(&vertex) {
        return false;
    }

    let reduced_set: Vec<usize> = dominating_set
        .iter()
        .copied()
        .filter(|&v| v != vertex)
        .collect();

    is_dominating(graph, &reduced_set)
}

/// Find the private neighbors of a vertex with respect to a dominating set
///
/// The private neighbors of v in D are vertices that are dominated only by v
/// (not by any other vertex in D).
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `dominating_set` - The dominating set
/// * `vertex` - The vertex to find private neighbors for
///
/// # Returns
/// Set of private neighbors
pub fn private_neighbors(graph: &Graph, dominating_set: &[usize], vertex: usize) -> HashSet<usize> {
    let dom_set: HashSet<usize> = dominating_set.iter().copied().collect();

    if !dom_set.contains(&vertex) {
        return HashSet::new();
    }

    let mut private = HashSet::new();
    let neighbors = graph.neighbors(vertex).unwrap_or_default();

    for &neighbor in &neighbors {
        if dom_set.contains(&neighbor) {
            continue;
        }

        // Check if neighbor is dominated only by vertex
        let neighbor_neighbors = graph.neighbors(neighbor).unwrap_or_default();
        let dominating_neighbors: Vec<usize> = neighbor_neighbors
            .iter()
            .copied()
            .filter(|n| dom_set.contains(n))
            .collect();

        if dominating_neighbors.len() == 1 && dominating_neighbors[0] == vertex {
            private.insert(neighbor);
        }
    }

    // The vertex itself can be a private neighbor if it's in the set
    if dom_set.contains(&vertex) {
        let vertex_neighbors = graph.neighbors(vertex).unwrap_or_default();
        let has_other_dominator = vertex_neighbors
            .iter()
            .any(|n| dom_set.contains(n) && *n != vertex);

        if !has_other_dominator {
            private.insert(vertex);
        }
    }

    private
}

/// Find all dominating sets of a given size
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `size` - The size of dominating sets to find (None for all sizes)
///
/// # Returns
/// Vector of dominating sets
pub fn dominating_sets(graph: &Graph, size: Option<usize>) -> Vec<Vec<usize>> {
    let n = graph.num_vertices();
    let mut result = Vec::new();

    let sizes: Vec<usize> = match size {
        Some(k) => vec![k],
        None => (1..=n).collect(),
    };

    for k in sizes {
        let sets = enumerate_dominating_sets_of_size(graph, k);
        result.extend(sets);
    }

    result
}

fn enumerate_dominating_sets_of_size(graph: &Graph, size: usize) -> Vec<Vec<usize>> {
    let n = graph.num_vertices();
    let mut result = Vec::new();

    if size > n {
        return result;
    }

    // Generate all combinations of 'size' vertices
    let mut indices = vec![0; size];
    for i in 0..size {
        indices[i] = i;
    }

    loop {
        if is_dominating(graph, &indices) {
            result.push(indices.clone());
        }

        // Generate next combination
        let mut i = size;
        while i > 0 {
            i -= 1;
            if indices[i] < n - size + i {
                indices[i] += 1;
                for j in i + 1..size {
                    indices[j] = indices[j - 1] + 1;
                }
                break;
            }
            if i == 0 {
                return result;
            }
        }
    }
}

/// Find all minimal dominating sets
///
/// A dominating set is minimal if no proper subset is dominating.
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// Vector of minimal dominating sets
pub fn minimal_dominating_sets(graph: &Graph) -> Vec<Vec<usize>> {
    let n = graph.num_vertices();
    let mut result = Vec::new();

    // Try all sizes from 1 to n
    for size in 1..=n {
        let sets = enumerate_dominating_sets_of_size(graph, size);

        for set in sets {
            // Check if minimal (no proper subset is dominating)
            let mut is_minimal = true;

            for &v in &set {
                let reduced: Vec<usize> = set.iter().copied().filter(|&u| u != v).collect();
                if is_dominating(graph, &reduced) {
                    is_minimal = false;
                    break;
                }
            }

            if is_minimal {
                result.push(set);
            }
        }
    }

    result
}

/// Find a minimum dominating set (smallest size)
///
/// Uses exhaustive search to find the smallest dominating set.
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// A minimum dominating set, or empty vector if graph has no vertices
///
/// # Examples
/// ```
/// use rustmath_graphs::graph::Graph;
/// use rustmath_graphs::domination::dominating_set;
///
/// let mut g = Graph::new(5);
/// g.add_edge(0, 1).unwrap();
/// g.add_edge(0, 2).unwrap();
/// g.add_edge(0, 3).unwrap();
/// g.add_edge(0, 4).unwrap();
///
/// // Star graph: center dominates all
/// let dom_set = dominating_set(&g);
/// assert_eq!(dom_set.len(), 1);
/// ```
pub fn dominating_set(graph: &Graph) -> Vec<usize> {
    let n = graph.num_vertices();

    if n == 0 {
        return vec![];
    }

    // Try increasing sizes
    for size in 1..=n {
        let sets = enumerate_dominating_sets_of_size(graph, size);
        if !sets.is_empty() {
            return sets[0].clone();
        }
    }

    // Fallback: all vertices
    (0..n).collect()
}

/// Find a dominating set using a greedy algorithm
///
/// Greedily selects vertices that dominate the most undominated vertices.
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// A dominating set (not necessarily minimum)
///
/// # Examples
/// ```
/// use rustmath_graphs::graph::Graph;
/// use rustmath_graphs::domination::greedy_dominating_set;
///
/// let mut g = Graph::new(6);
/// g.add_edge(0, 1).unwrap();
/// g.add_edge(0, 2).unwrap();
/// g.add_edge(3, 4).unwrap();
/// g.add_edge(3, 5).unwrap();
///
/// let dom_set = greedy_dominating_set(&g);
/// assert!(dom_set.len() <= 2); // Should find at most 2 vertices
/// ```
pub fn greedy_dominating_set(graph: &Graph) -> Vec<usize> {
    let n = graph.num_vertices();
    let mut dominating_set = Vec::new();
    let mut dominated = HashSet::new();

    while dominated.len() < n {
        // Find vertex that dominates the most undominated vertices
        let mut best_vertex = None;
        let mut best_count = 0;

        for v in 0..n {
            if dominating_set.contains(&v) {
                continue;
            }

            let mut count = 0;

            // Count v itself if not dominated
            if !dominated.contains(&v) {
                count += 1;
            }

            // Count neighbors that aren't dominated
            for &neighbor in graph.neighbors(v).unwrap_or_default().iter() {
                if !dominated.contains(&neighbor) {
                    count += 1;
                }
            }

            if count > best_count {
                best_count = count;
                best_vertex = Some(v);
            }
        }

        if let Some(v) = best_vertex {
            dominating_set.push(v);
            dominated.insert(v);

            for &neighbor in graph.neighbors(v).unwrap_or_default().iter() {
                dominated.insert(neighbor);
            }
        } else {
            // No improvement possible, add any undominated vertex
            for v in 0..n {
                if !dominated.contains(&v) {
                    dominating_set.push(v);
                    dominated.insert(v);
                    for &neighbor in graph.neighbors(v).unwrap_or_default().iter() {
                        dominated.insert(neighbor);
                    }
                    break;
                }
            }
        }
    }

    dominating_set
}

/// Find the maximum leaf number of a spanning tree
///
/// The maximum leaf number is the maximum number of leaves in any spanning tree.
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// The maximum leaf number, or None if graph is not connected
pub fn maximum_leaf_number(graph: &Graph) -> Option<usize> {
    if !graph.is_connected() {
        return None;
    }

    let n = graph.num_vertices();

    if n <= 2 {
        return Some(1);
    }

    // For connected graphs, the maximum leaf number is at least 2
    // We'll use a heuristic: BFS from different starting points
    let mut max_leaves = 0;

    for start in 0..n {
        let leaves = count_leaves_in_bfs_tree(graph, start);
        if leaves > max_leaves {
            max_leaves = leaves;
        }
    }

    Some(max_leaves)
}

fn count_leaves_in_bfs_tree(graph: &Graph, start: usize) -> usize {
    let n = graph.num_vertices();
    let mut visited = vec![false; n];
    let mut parent = vec![None; n];
    let mut queue = VecDeque::new();

    queue.push_back(start);
    visited[start] = true;

    while let Some(v) = queue.pop_front() {
        for &neighbor in graph.neighbors(v).unwrap_or_default().iter() {
            if !visited[neighbor] {
                visited[neighbor] = true;
                parent[neighbor] = Some(v);
                queue.push_back(neighbor);
            }
        }
    }

    // Count leaves (vertices with no children in the BFS tree)
    let mut leaf_count = 0;

    for v in 0..n {
        if v == start {
            continue;
        }

        // Check if v has any children
        let mut has_children = false;
        for u in 0..n {
            if parent[u] == Some(v) {
                has_children = true;
                break;
            }
        }

        if !has_children {
            leaf_count += 1;
        }
    }

    leaf_count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_dominating() {
        let mut g = Graph::new(5);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();
        g.add_edge(3, 4).unwrap();

        // {1, 3} dominates the path
        assert!(is_dominating(&g, &[1, 3]));

        // {2} does NOT dominate a 5-vertex path (0 and 4 are not adjacent to 2)
        assert!(!is_dominating(&g, &[2]));

        // {0, 2, 4} dominates the path
        assert!(is_dominating(&g, &[0, 2, 4]));

        // {0, 4} does not dominate (2 is not dominated)
        assert!(!is_dominating(&g, &[0, 4]));

        // For a 3-vertex path, middle vertex dominates
        let mut g2 = Graph::new(3);
        g2.add_edge(0, 1).unwrap();
        g2.add_edge(1, 2).unwrap();
        assert!(is_dominating(&g2, &[1]));
    }

    #[test]
    fn test_is_redundant() {
        let mut g = Graph::new(5);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();
        g.add_edge(3, 4).unwrap();

        // {1, 2, 3} is dominating, but 2 is redundant
        assert!(is_dominating(&g, &[1, 2, 3]));
        assert!(is_redundant(&g, &[1, 2, 3], 2));
        assert!(!is_redundant(&g, &[1, 3], 1));
    }

    #[test]
    fn test_private_neighbors() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        let dom_set = vec![1, 3];
        let private = private_neighbors(&g, &dom_set, 1);

        // Vertex 1's private neighbors should include 0 (only dominated by 1)
        assert!(private.contains(&0));
    }

    #[test]
    fn test_dominating_set_star() {
        // Star graph: center dominates all
        let mut g = Graph::new(5);
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 2).unwrap();
        g.add_edge(0, 3).unwrap();
        g.add_edge(0, 4).unwrap();

        let dom_set = dominating_set(&g);
        assert_eq!(dom_set.len(), 1);
        assert!(is_dominating(&g, &dom_set));
    }

    #[test]
    fn test_greedy_dominating_set() {
        let mut g = Graph::new(6);
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 2).unwrap();
        g.add_edge(3, 4).unwrap();
        g.add_edge(3, 5).unwrap();

        let dom_set = greedy_dominating_set(&g);
        assert!(is_dominating(&g, &dom_set));
        assert!(dom_set.len() <= 3);
    }

    #[test]
    fn test_dominating_sets() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        let sets = dominating_sets(&g, Some(1));
        // Only vertex 1 can dominate alone
        assert_eq!(sets.len(), 1);
        assert_eq!(sets[0], vec![1]);
    }

    #[test]
    fn test_minimal_dominating_sets() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        let minimal = minimal_dominating_sets(&g);
        // Should find: {1}, {0, 2}
        assert!(minimal.len() >= 1);

        // All should be minimal
        for set in &minimal {
            assert!(is_dominating(&g, set));
            for &v in set {
                let reduced: Vec<usize> = set.iter().copied().filter(|&u| u != v).collect();
                assert!(!is_dominating(&g, &reduced));
            }
        }
    }

    #[test]
    fn test_maximum_leaf_number() {
        // Path graph
        let mut g = Graph::new(5);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();
        g.add_edge(3, 4).unwrap();

        let max_leaves = maximum_leaf_number(&g);
        assert!(max_leaves.is_some());
        assert!(max_leaves.unwrap() >= 2);
    }

    #[test]
    fn test_complete_graph_domination() {
        // Complete graph K4: any single vertex dominates
        let mut g = Graph::new(4);
        for i in 0..4 {
            for j in i + 1..4 {
                g.add_edge(i, j).unwrap();
            }
        }

        let dom_set = dominating_set(&g);
        assert_eq!(dom_set.len(), 1);
    }

    #[test]
    fn test_disconnected_graph() {
        // Disconnected graph
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(2, 3).unwrap();

        // Should still find dominating set
        let dom_set = dominating_set(&g);
        assert!(is_dominating(&g, &dom_set));
        assert_eq!(dom_set.len(), 2);
    }
}
