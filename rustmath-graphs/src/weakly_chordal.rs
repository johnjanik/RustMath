//! Weakly chordal graph recognition algorithms
//!
//! A graph is weakly chordal if it is both long-hole-free and long-antihole-free:
//! - A hole is an induced cycle of length at least 4
//! - An antihole is the complement of a hole
//! - A long hole is an induced cycle of length at least 5
//! - A long antihole is the complement of an induced cycle of length at least 5
//!
//! This module implements efficient algorithms for testing these properties.

use crate::graph::Graph;
use std::collections::{HashSet, VecDeque};

/// Check if a graph is free of long holes (induced cycles of length >= 5)
///
/// A hole is an induced cycle. A long hole has length at least 5.
/// This function returns true if the graph contains no induced cycles
/// of length 5 or more.
///
/// # Arguments
///
/// * `graph` - The graph to check
///
/// # Returns
///
/// `true` if the graph is long-hole-free, `false` otherwise
///
/// # Examples
///
/// ```
/// use rustmath_graphs::{Graph, weakly_chordal::is_long_hole_free};
///
/// let mut g = Graph::new(5);
/// // Create a 5-cycle (pentagon)
/// g.add_edge(0, 1).unwrap();
/// g.add_edge(1, 2).unwrap();
/// g.add_edge(2, 3).unwrap();
/// g.add_edge(3, 4).unwrap();
/// g.add_edge(4, 0).unwrap();
///
/// // A 5-cycle has a long hole
/// assert_eq!(is_long_hole_free(&g), false);
/// ```
pub fn is_long_hole_free(graph: &Graph) -> bool {
    let n = graph.num_vertices();

    // Check each vertex as a potential start of an induced cycle
    for start in 0..n {
        if has_long_induced_cycle_from(graph, start, 5) {
            return false;
        }
    }

    true
}

/// Check if a graph is free of long antiholes (complements of induced cycles of length >= 5)
///
/// An antihole is the complement of a hole. A long antihole is the complement
/// of an induced cycle of length at least 5. This function returns true if
/// the graph contains no such structures.
///
/// # Arguments
///
/// * `graph` - The graph to check
///
/// # Returns
///
/// `true` if the graph is long-antihole-free, `false` otherwise
///
/// # Examples
///
/// ```
/// use rustmath_graphs::{Graph, weakly_chordal::is_long_antihole_free};
///
/// let mut g = Graph::new(5);
/// // Create the complement of a 5-cycle
/// for i in 0..5 {
///     for j in (i+1)..5 {
///         if (j - i != 1) && (i != 0 || j != 4) {
///             g.add_edge(i, j).unwrap();
///         }
///     }
/// }
///
/// // This should have a long antihole
/// assert_eq!(is_long_antihole_free(&g), false);
/// ```
pub fn is_long_antihole_free(graph: &Graph) -> bool {
    // Create the complement graph
    let complement = graph_complement(graph);

    // An antihole in the original graph is a hole in the complement
    is_long_hole_free(&complement)
}

/// Check if a graph is weakly chordal
///
/// A graph is weakly chordal if and only if it is both long-hole-free
/// and long-antihole-free. This is equivalent to saying that neither
/// the graph nor its complement contains an induced cycle of length
/// at least 5.
///
/// Weakly chordal graphs include:
/// - Chordal graphs (triangle-free graphs with no induced cycles >= 4)
/// - Interval graphs
/// - Permutation graphs
///
/// # Arguments
///
/// * `graph` - The graph to check
///
/// # Returns
///
/// `true` if the graph is weakly chordal, `false` otherwise
///
/// # Examples
///
/// ```
/// use rustmath_graphs::{Graph, weakly_chordal::is_weakly_chordal};
///
/// let mut g = Graph::new(4);
/// // Create a path graph (which is weakly chordal)
/// g.add_edge(0, 1).unwrap();
/// g.add_edge(1, 2).unwrap();
/// g.add_edge(2, 3).unwrap();
///
/// assert_eq!(is_weakly_chordal(&g), true);
///
/// // Create a 5-cycle (not weakly chordal)
/// let mut h = Graph::new(5);
/// h.add_edge(0, 1).unwrap();
/// h.add_edge(1, 2).unwrap();
/// h.add_edge(2, 3).unwrap();
/// h.add_edge(3, 4).unwrap();
/// h.add_edge(4, 0).unwrap();
///
/// assert_eq!(is_weakly_chordal(&h), false);
/// ```
pub fn is_weakly_chordal(graph: &Graph) -> bool {
    is_long_hole_free(graph) && is_long_antihole_free(graph)
}

// Helper function: Create the complement of a graph
fn graph_complement(graph: &Graph) -> Graph {
    let n = graph.num_vertices();
    let mut complement = Graph::new(n);

    for i in 0..n {
        for j in (i+1)..n {
            if !graph.has_edge(i, j) {
                complement.add_edge(i, j).unwrap();
            }
        }
    }

    complement
}

// Helper function: Check if there's an induced cycle of given minimum length from a start vertex
fn has_long_induced_cycle_from(graph: &Graph, start: usize, min_length: usize) -> bool {
    let n = graph.num_vertices();

    // Use BFS to find paths from start vertex
    // Track path to each vertex and check for cycles
    let mut visited = vec![false; n];
    let mut parent = vec![None; n];
    let mut distance = vec![0; n];
    let mut queue = VecDeque::new();

    queue.push_back(start);
    visited[start] = true;

    while let Some(u) = queue.pop_front() {
        if let Some(neighbors) = graph.neighbors(u) {
            for &v in &neighbors {
                if !visited[v] {
                    visited[v] = true;
                    parent[v] = Some(u);
                    distance[v] = distance[u] + 1;
                    queue.push_back(v);
                } else if parent[u] != Some(v) && distance[v] + distance[u] + 1 >= min_length {
                    // Found a cycle, check if it's induced
                    if is_induced_cycle(graph, start, u, v, &parent) {
                        return true;
                    }
                }
            }
        }
    }

    false
}

// Helper function: Check if a cycle is induced (no chords)
fn is_induced_cycle(
    graph: &Graph,
    start: usize,
    u: usize,
    v: usize,
    parent: &[Option<usize>]
) -> bool {
    // Reconstruct the cycle
    let mut cycle = Vec::new();

    // Path from u back to start
    let mut current = u;
    while current != start {
        cycle.push(current);
        if let Some(p) = parent[current] {
            current = p;
        } else {
            return false;
        }
    }
    cycle.push(start);

    // Reverse to get correct order
    cycle.reverse();

    // Add path from start to v
    let mut path_to_v = Vec::new();
    current = v;
    while current != start {
        path_to_v.push(current);
        if let Some(p) = parent[current] {
            current = p;
        } else {
            return false;
        }
    }

    // Complete the cycle
    cycle.extend(path_to_v);

    // Check if cycle has at least min length
    if cycle.len() < 5 {
        return false;
    }

    // Check for chords (edges between non-adjacent vertices in cycle)
    for i in 0..cycle.len() {
        for j in (i+2)..cycle.len() {
            // Skip adjacent edges in the cycle
            if j == (i + 1) % cycle.len() || i == (j + 1) % cycle.len() {
                continue;
            }

            // If there's an edge between non-adjacent cycle vertices, it's not induced
            if graph.has_edge(cycle[i], cycle[j]) {
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Graph;

    #[test]
    fn test_path_is_long_hole_free() {
        // A path has no cycles at all
        let mut g = Graph::new(6);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();
        g.add_edge(3, 4).unwrap();
        g.add_edge(4, 5).unwrap();

        assert!(is_long_hole_free(&g));
    }

    #[test]
    fn test_triangle_is_long_hole_free() {
        // A triangle (3-cycle) is not a long hole
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 0).unwrap();

        assert!(is_long_hole_free(&g));
    }

    #[test]
    fn test_square_is_long_hole_free() {
        // A 4-cycle is not a long hole (long means >= 5)
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();
        g.add_edge(3, 0).unwrap();

        assert!(is_long_hole_free(&g));
    }

    #[test]
    fn test_pentagon_has_long_hole() {
        // A 5-cycle (pentagon) is a long hole
        let mut g = Graph::new(5);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();
        g.add_edge(3, 4).unwrap();
        g.add_edge(4, 0).unwrap();

        assert!(!is_long_hole_free(&g));
    }

    #[test]
    fn test_hexagon_has_long_hole() {
        // A 6-cycle (hexagon) is a long hole
        let mut g = Graph::new(6);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();
        g.add_edge(3, 4).unwrap();
        g.add_edge(4, 5).unwrap();
        g.add_edge(5, 0).unwrap();

        assert!(!is_long_hole_free(&g));
    }

    #[test]
    fn test_path_is_weakly_chordal() {
        // A path is weakly chordal
        let mut g = Graph::new(5);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();
        g.add_edge(3, 4).unwrap();

        assert!(is_weakly_chordal(&g));
    }

    #[test]
    fn test_complete_graph_is_weakly_chordal() {
        // A complete graph has no induced cycles of any length
        let mut g = Graph::new(5);
        for i in 0..5 {
            for j in (i+1)..5 {
                g.add_edge(i, j).unwrap();
            }
        }

        assert!(is_weakly_chordal(&g));
    }

    #[test]
    fn test_pentagon_not_weakly_chordal() {
        // A 5-cycle is not weakly chordal (has a long hole)
        let mut g = Graph::new(5);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();
        g.add_edge(3, 4).unwrap();
        g.add_edge(4, 0).unwrap();

        assert!(!is_weakly_chordal(&g));
    }

    #[test]
    fn test_complement_of_pentagon_not_weakly_chordal() {
        // The complement of a 5-cycle has a long antihole
        let mut g = Graph::new(5);
        // Add all edges except the pentagon edges
        for i in 0..5 {
            for j in (i+1)..5 {
                let is_pentagon_edge = (j == i + 1) || (i == 0 && j == 4);
                if !is_pentagon_edge {
                    g.add_edge(i, j).unwrap();
                }
            }
        }

        assert!(!is_weakly_chordal(&g));
    }

    #[test]
    fn test_tree_is_weakly_chordal() {
        // Trees are weakly chordal (no cycles at all)
        let mut g = Graph::new(7);
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 2).unwrap();
        g.add_edge(1, 3).unwrap();
        g.add_edge(1, 4).unwrap();
        g.add_edge(2, 5).unwrap();
        g.add_edge(2, 6).unwrap();

        assert!(is_weakly_chordal(&g));
    }

    #[test]
    fn test_graph_complement() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        let comp = graph_complement(&g);

        // Check complement has correct edges
        assert!(!comp.has_edge(0, 1));
        assert!(!comp.has_edge(1, 2));
        assert!(comp.has_edge(0, 2));
        assert!(comp.has_edge(0, 3));
        assert!(comp.has_edge(1, 3));
        assert!(comp.has_edge(2, 3));
    }

    #[test]
    fn test_empty_graph_is_weakly_chordal() {
        let g = Graph::new(5);
        assert!(is_weakly_chordal(&g));
    }

    #[test]
    fn test_single_vertex_is_weakly_chordal() {
        let g = Graph::new(1);
        assert!(is_weakly_chordal(&g));
    }

    #[test]
    fn test_single_edge_is_weakly_chordal() {
        let mut g = Graph::new(2);
        g.add_edge(0, 1).unwrap();
        assert!(is_weakly_chordal(&g));
    }
}
