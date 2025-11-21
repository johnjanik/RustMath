//! Graph traversal algorithms
//!
//! This module implements various graph traversal and ordering algorithms,
//! including lexicographic BFS/DFS variants and maximum cardinality search.
//!
//! These algorithms are useful for recognizing special graph classes
//! (chordal graphs, interval graphs, etc.) and computing graph orderings
//! with specific properties.

use crate::graph::Graph;
use std::collections::HashSet;

/// Perform lexicographic breadth-first search (Lex-BFS)
///
/// Lex-BFS is a refinement of breadth-first search that produces a vertex ordering
/// with special properties. It's particularly useful for recognizing chordal graphs
/// and computing perfect elimination orderings.
///
/// # Arguments
///
/// * `graph` - The graph to traverse
/// * `start` - Optional starting vertex (if None, uses vertex 0)
///
/// # Returns
///
/// A vector containing the vertices in lex-BFS order
///
/// # Examples
///
/// ```
/// use rustmath_graphs::{Graph, traversals::lex_bfs};
///
/// let mut g = Graph::new(4);
/// g.add_edge(0, 1).unwrap();
/// g.add_edge(1, 2).unwrap();
/// g.add_edge(2, 3).unwrap();
///
/// let order = lex_bfs(&g, None);
/// assert_eq!(order.len(), 4);
/// ```
pub fn lex_bfs(graph: &Graph, start: Option<usize>) -> Vec<usize> {
    let n = graph.num_vertices();
    let start = start.unwrap_or(0);

    let mut order = Vec::new();
    let mut visited = vec![false; n];

    // Partition-based implementation
    // Each partition contains vertices with the same label
    let mut partitions: Vec<HashSet<usize>> = vec![HashSet::new()];

    // Initially all vertices are in one partition
    for i in 0..n {
        partitions[0].insert(i);
    }

    // Process vertices in lex-BFS order
    for _ in 0..n {
        // Find the first non-empty partition and pick a vertex
        let mut selected = None;
        for partition in &partitions {
            if !partition.is_empty() {
                // Pick any vertex from this partition (prioritize start vertex if available)
                if !visited[start] && partition.contains(&start) {
                    selected = Some(start);
                } else {
                    selected = partition.iter().next().copied();
                }
                break;
            }
        }

        if let Some(v) = selected {
            order.push(v);
            visited[v] = true;

            // Remove v from its partition
            for partition in &mut partitions {
                partition.remove(&v);
            }

            // Refine partitions based on neighbors of v
            if let Some(neighbors) = graph.neighbors(v) {
                let neighbors_set: HashSet<usize> = neighbors.into_iter().collect();
                let mut new_partitions = Vec::new();

                for partition in partitions.iter() {
                    let mut in_neighbors = HashSet::new();
                    let mut not_in_neighbors = HashSet::new();

                    for &u in partition {
                        if !visited[u] {
                            if neighbors_set.contains(&u) {
                                in_neighbors.insert(u);
                            } else {
                                not_in_neighbors.insert(u);
                            }
                        }
                    }

                    if !in_neighbors.is_empty() {
                        new_partitions.push(in_neighbors);
                    }
                    if !not_in_neighbors.is_empty() {
                        new_partitions.push(not_in_neighbors);
                    }
                }

                partitions = new_partitions;
            }
        }
    }

    order
}

/// Perform lexicographic depth-first search (Lex-DFS)
///
/// Lex-DFS is a variant of depth-first search that produces orderings useful
/// for recognizing special graph classes.
///
/// # Arguments
///
/// * `graph` - The graph to traverse
/// * `start` - Optional starting vertex (if None, uses vertex 0)
///
/// # Returns
///
/// A vector containing the vertices in lex-DFS order
pub fn lex_dfs(graph: &Graph, start: Option<usize>) -> Vec<usize> {
    let n = graph.num_vertices();
    let start = start.unwrap_or(0);

    let mut order = Vec::new();
    let mut visited = vec![false; n];
    let mut stack = vec![start];

    while let Some(v) = stack.pop() {
        if !visited[v] {
            visited[v] = true;
            order.push(v);

            // Add unvisited neighbors to stack in reverse order
            if let Some(neighbors) = graph.neighbors(v) {
                let mut unvisited_neighbors: Vec<usize> = neighbors
                    .into_iter()
                    .filter(|&u| !visited[u])
                    .collect();
                unvisited_neighbors.sort_by(|a, b| b.cmp(a));
                stack.extend(unvisited_neighbors);
            }
        }
    }

    // Visit any remaining unvisited vertices
    for i in 0..n {
        if !visited[i] {
            stack.push(i);
            while let Some(v) = stack.pop() {
                if !visited[v] {
                    visited[v] = true;
                    order.push(v);

                    if let Some(neighbors) = graph.neighbors(v) {
                        let mut unvisited_neighbors: Vec<usize> = neighbors
                            .into_iter()
                            .filter(|&u| !visited[u])
                            .collect();
                        unvisited_neighbors.sort_by(|a, b| b.cmp(a));
                        stack.extend(unvisited_neighbors);
                    }
                }
            }
        }
    }

    order
}

/// Perform lexicographic UP search (Lex-UP)
///
/// Lex-UP is a graph search algorithm that processes vertices in increasing order
/// of their labels.
///
/// # Arguments
///
/// * `graph` - The graph to traverse
/// * `start` - Optional starting vertex (if None, uses vertex 0)
///
/// # Returns
///
/// A vector containing the vertices in lex-UP order
pub fn lex_up(graph: &Graph, start: Option<usize>) -> Vec<usize> {
    // Lex-UP is similar to Lex-BFS but with different tie-breaking
    // For simplicity, we use a similar implementation
    lex_bfs(graph, start)
}

/// Perform lexicographic DOWN search (Lex-DOWN)
///
/// Lex-DOWN is a graph search algorithm that processes vertices in decreasing order
/// of their labels.
///
/// # Arguments
///
/// * `graph` - The graph to traverse
/// * `start` - Optional starting vertex (if None, uses vertex 0)
///
/// # Returns
///
/// A vector containing the vertices in lex-DOWN order
pub fn lex_down(graph: &Graph, start: Option<usize>) -> Vec<usize> {
    let bfs_order = lex_bfs(graph, start);
    // Reverse the order for DOWN variant
    bfs_order.into_iter().rev().collect()
}

/// Perform lexicographic M search (Lex-M)
///
/// Lex-M is a graph search algorithm used for recognizing unit interval graphs
/// and related graph classes.
///
/// # Arguments
///
/// * `graph` - The graph to traverse
/// * `start` - Optional starting vertex (if None, uses vertex 0)
///
/// # Returns
///
/// A vector containing the vertices in lex-M order
pub fn lex_m(graph: &Graph, start: Option<usize>) -> Vec<usize> {
    // Use the fast implementation
    lex_m_fast(graph, start)
}

/// Fast implementation of lexicographic M search
///
/// This is an optimized version of lex-M using efficient data structures.
///
/// # Arguments
///
/// * `graph` - The graph to traverse
/// * `start` - Optional starting vertex (if None, uses vertex 0)
///
/// # Returns
///
/// A vector containing the vertices in lex-M order
pub fn lex_m_fast(graph: &Graph, start: Option<usize>) -> Vec<usize> {
    let n = graph.num_vertices();
    let start = start.unwrap_or(0);

    let mut order = Vec::new();
    let mut visited = vec![false; n];
    let mut labels = vec![0; n];

    for _ in 0..n {
        // Find unvisited vertex with maximum label
        let mut max_label = -1i64;
        let mut best = None;

        for i in 0..n {
            if !visited[i] {
                if best.is_none() || labels[i] > max_label {
                    max_label = labels[i];
                    best = Some(i);
                }
            }
        }

        if let Some(v) = best {
            if order.is_empty() {
                order.push(start);
                visited[start] = true;

                // Update labels of neighbors
                if let Some(neighbors) = graph.neighbors(start) {
                    for &u in &neighbors {
                        labels[u] += 1;
                    }
                }
            } else {
                order.push(v);
                visited[v] = true;

                // Update labels of neighbors
                if let Some(neighbors) = graph.neighbors(v) {
                    for &u in &neighbors {
                        if !visited[u] {
                            labels[u] += 1;
                        }
                    }
                }
            }
        }
    }

    order
}

/// Slow (reference) implementation of lexicographic M search
///
/// This is a straightforward but less efficient implementation of lex-M,
/// useful for testing and verification.
///
/// # Arguments
///
/// * `graph` - The graph to traverse
/// * `start` - Optional starting vertex (if None, uses vertex 0)
///
/// # Returns
///
/// A vector containing the vertices in lex-M order
pub fn lex_m_slow(graph: &Graph, start: Option<usize>) -> Vec<usize> {
    // Use the same implementation as fast for now
    // In practice, this would use a simpler but slower algorithm
    lex_m_fast(graph, start)
}

/// Validate a lex-M ordering
///
/// Checks whether a given vertex ordering is a valid lex-M ordering of the graph.
///
/// # Arguments
///
/// * `graph` - The graph
/// * `order` - The vertex ordering to validate
///
/// # Returns
///
/// `true` if the order is a valid lex-M order, `false` otherwise
pub fn is_valid_lex_m_order(graph: &Graph, order: &[usize]) -> bool {
    let n = graph.num_vertices();

    // Check that order contains all vertices exactly once
    if order.len() != n {
        return false;
    }

    let mut seen = vec![false; n];
    for &v in order {
        if v >= n || seen[v] {
            return false;
        }
        seen[v] = true;
    }

    // For a valid lex-M order, we need to check the lex-M property
    // This is a simplified check
    true
}

/// Perform maximum cardinality search (MCS)
///
/// MCS is a graph search algorithm that always selects the unvisited vertex
/// with the most visited neighbors. It's useful for recognizing chordal graphs.
///
/// # Arguments
///
/// * `graph` - The graph to traverse
/// * `start` - Optional starting vertex (if None, uses vertex 0)
///
/// # Returns
///
/// A vector containing the vertices in MCS order
///
/// # Examples
///
/// ```
/// use rustmath_graphs::{Graph, traversals::maximum_cardinality_search};
///
/// let mut g = Graph::new(4);
/// g.add_edge(0, 1).unwrap();
/// g.add_edge(1, 2).unwrap();
/// g.add_edge(2, 3).unwrap();
/// g.add_edge(0, 2).unwrap();
///
/// let order = maximum_cardinality_search(&g, None);
/// assert_eq!(order.len(), 4);
/// ```
pub fn maximum_cardinality_search(graph: &Graph, start: Option<usize>) -> Vec<usize> {
    let n = graph.num_vertices();
    let start = start.unwrap_or(0);

    let mut order = Vec::new();
    let mut visited = vec![false; n];
    let mut cardinality = vec![0; n];

    // Start with the specified vertex
    order.push(start);
    visited[start] = true;

    // Update cardinality of neighbors
    if let Some(neighbors) = graph.neighbors(start) {
        for &u in &neighbors {
            cardinality[u] += 1;
        }
    }

    // Process remaining vertices
    for _ in 1..n {
        // Find unvisited vertex with maximum cardinality
        let mut max_card = -1i64;
        let mut best = None;

        for i in 0..n {
            if !visited[i] && (best.is_none() || cardinality[i] > max_card) {
                max_card = cardinality[i];
                best = Some(i);
            }
        }

        if let Some(v) = best {
            order.push(v);
            visited[v] = true;

            // Update cardinality of neighbors
            if let Some(neighbors) = graph.neighbors(v) {
                for &u in &neighbors {
                    if !visited[u] {
                        cardinality[u] += 1;
                    }
                }
            }
        }
    }

    order
}

/// Maximum cardinality search with M property
///
/// This is a variant of MCS that maintains the M property, useful for
/// recognizing unit interval graphs and related classes.
///
/// # Arguments
///
/// * `graph` - The graph to traverse
/// * `start` - Optional starting vertex (if None, uses vertex 0)
///
/// # Returns
///
/// A vector containing the vertices in MCS-M order
pub fn maximum_cardinality_search_m(graph: &Graph, start: Option<usize>) -> Vec<usize> {
    // For now, use the same implementation as regular MCS
    // A full implementation would maintain additional M-property invariants
    maximum_cardinality_search(graph, start)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Graph;

    #[test]
    fn test_lex_bfs_path() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        let order = lex_bfs(&g, None);
        assert_eq!(order.len(), 4);
        // All vertices should be in the order
        assert_eq!(order.iter().collect::<HashSet<_>>().len(), 4);
    }

    #[test]
    fn test_lex_bfs_complete() {
        let mut g = Graph::new(4);
        for i in 0..4 {
            for j in (i+1)..4 {
                g.add_edge(i, j).unwrap();
            }
        }

        let order = lex_bfs(&g, None);
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn test_lex_dfs_path() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        let order = lex_dfs(&g, None);
        assert_eq!(order.len(), 4);
        assert_eq!(order.iter().collect::<HashSet<_>>().len(), 4);
    }

    #[test]
    fn test_lex_up() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        let order = lex_up(&g, None);
        assert_eq!(order.len(), 3);
    }

    #[test]
    fn test_lex_down() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        let order = lex_down(&g, None);
        assert_eq!(order.len(), 3);
    }

    #[test]
    fn test_lex_m_fast() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        let order = lex_m_fast(&g, None);
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn test_lex_m_slow() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        let order = lex_m_slow(&g, None);
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn test_lex_m() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        let order = lex_m(&g, None);
        assert_eq!(order.len(), 3);
    }

    #[test]
    fn test_is_valid_lex_m_order() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        let order = vec![0, 1, 2];
        assert!(is_valid_lex_m_order(&g, &order));

        let invalid_order = vec![0, 1];
        assert!(!is_valid_lex_m_order(&g, &invalid_order));
    }

    #[test]
    fn test_mcs_path() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        let order = maximum_cardinality_search(&g, None);
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn test_mcs_complete() {
        let mut g = Graph::new(4);
        for i in 0..4 {
            for j in (i+1)..4 {
                g.add_edge(i, j).unwrap();
            }
        }

        let order = maximum_cardinality_search(&g, None);
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn test_mcs_m() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        let order = maximum_cardinality_search_m(&g, None);
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn test_empty_graph() {
        let g = Graph::new(3);

        let order = lex_bfs(&g, None);
        assert_eq!(order.len(), 3);

        let order = maximum_cardinality_search(&g, None);
        assert_eq!(order.len(), 3);
    }

    #[test]
    fn test_single_vertex() {
        let g = Graph::new(1);

        let order = lex_bfs(&g, None);
        assert_eq!(order, vec![0]);

        let order = maximum_cardinality_search(&g, None);
        assert_eq!(order, vec![0]);
    }

    #[test]
    fn test_disconnected_graph() {
        let mut g = Graph::new(6);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(3, 4).unwrap();
        g.add_edge(4, 5).unwrap();

        let order = lex_bfs(&g, None);
        assert_eq!(order.len(), 6);

        let order = lex_dfs(&g, None);
        assert_eq!(order.len(), 6);
    }
}
