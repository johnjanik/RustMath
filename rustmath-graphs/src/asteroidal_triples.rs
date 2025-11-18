//! # Asteroidal Triples
//!
//! This module provides functions for detecting asteroidal triples in graphs.
//!
//! ## Definition
//!
//! An **asteroidal triple** is a set of three independent (non-adjacent) vertices
//! {u, v, w} in a graph such that for each pair of vertices from the triple, there
//! exists a path connecting them that avoids the closed neighborhood of the third vertex.
//!
//! A graph is **asteroidal triple-free** (AT-free) if it contains no asteroidal triples.
//!
//! ## Properties
//!
//! AT-free graphs are important in graph theory because:
//! - Interval graphs are AT-free
//! - AT-free graphs have nice algorithmic properties
//! - They generalize interval graphs and cocomparability graphs
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_graphs::{Graph, is_asteroidal_triple_free};
//!
//! // Create a path graph P4: 0-1-2-3
//! let mut g = Graph::new(4);
//! g.add_edge(0, 1).unwrap();
//! g.add_edge(1, 2).unwrap();
//! g.add_edge(2, 3).unwrap();
//!
//! // P4 is AT-free (it's an interval graph)
//! assert!(is_asteroidal_triple_free(&g));
//! ```

use crate::Graph;
use std::collections::{HashSet, VecDeque};

/// Checks if a graph is asteroidal triple-free.
///
/// A graph is AT-free if it contains no asteroidal triple: three independent vertices
/// such that between each pair there is a path avoiding the neighborhood of the third.
///
/// # Arguments
///
/// * `graph` - The graph to test
///
/// # Returns
///
/// `true` if the graph is AT-free, `false` if it contains an asteroidal triple
///
/// # Algorithm
///
/// Uses the straightforward O(n³) algorithm:
/// 1. For each vertex v, compute connected components of G - N[v] (graph minus closed neighborhood of v)
/// 2. Check all triples (u, v, w) of independent vertices
/// 3. For each triple, verify if u and w are in same component when v's neighborhood is removed
///
/// # Examples
///
/// ```rust
/// use rustmath_graphs::{Graph, is_asteroidal_triple_free};
///
/// // Triangle graph: 0-1, 1-2, 2-0
/// let mut triangle = Graph::new(3);
/// triangle.add_edge(0, 1).unwrap();
/// triangle.add_edge(1, 2).unwrap();
/// triangle.add_edge(2, 0).unwrap();
///
/// // No independent triples exist, so it's AT-free
/// assert!(is_asteroidal_triple_free(&triangle));
///
/// // Path graph: 0-1-2-3-4
/// let mut path = Graph::new(5);
/// path.add_edge(0, 1).unwrap();
/// path.add_edge(1, 2).unwrap();
/// path.add_edge(2, 3).unwrap();
/// path.add_edge(3, 4).unwrap();
///
/// // Path graphs are AT-free (they're interval graphs)
/// assert!(is_asteroidal_triple_free(&path));
/// ```
pub fn is_asteroidal_triple_free(graph: &Graph) -> bool {
    let n = graph.num_vertices();

    if n < 3 {
        return true; // Trivially AT-free
    }

    // Build component matrix: components[v][u] = component ID of u in G - N[v]
    let components = build_component_matrix(graph);

    // Check all potential triples (u, v, w) of independent vertices
    for u in 0..n {
        for v in (u + 1)..n {
            // Skip if u and v are adjacent
            if graph.has_edge(u, v) {
                continue;
            }

            for w in (v + 1)..n {
                // Skip if any pair is adjacent (not independent)
                if graph.has_edge(u, w) || graph.has_edge(v, w) {
                    continue;
                }

                // Check if (u, v, w) form an asteroidal triple
                // u and w must be in same component of G - N[v]
                // v and w must be in same component of G - N[u]
                // u and v must be in same component of G - N[w]
                if let (Some(comp_u_v), Some(comp_w_v)) = (components[v][u], components[v][w]) {
                    if comp_u_v == comp_w_v {
                        // u and w are in same component when v removed
                        if let (Some(comp_v_u), Some(comp_w_u)) = (components[u][v], components[u][w]) {
                            if comp_v_u == comp_w_u {
                                // v and w are in same component when u removed
                                if let (Some(comp_u_w), Some(comp_v_w)) = (components[w][u], components[w][v]) {
                                    if comp_u_w == comp_v_w {
                                        // Found an asteroidal triple!
                                        return false;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    true // No asteroidal triple found
}

/// Builds the component matrix for AT-free testing.
///
/// For each vertex v, computes connected components in G - N[v]
/// (the graph with v and its neighbors removed).
///
/// Returns: n×n matrix where components[v][u] = component ID of u in G - N[v]
/// (None if u is in N[v] or u == v)
fn build_component_matrix(graph: &Graph) -> Vec<Vec<Option<usize>>> {
    let n = graph.num_vertices();
    let mut components = vec![vec![None; n]; n];

    for v in 0..n {
        // Compute closed neighborhood of v: N[v] = {v} ∪ N(v)
        let mut closed_neighborhood = HashSet::new();
        closed_neighborhood.insert(v);
        if let Some(neighbors) = graph.neighbors(v) {
            for neighbor in neighbors {
                closed_neighborhood.insert(neighbor);
            }
        }

        // Compute connected components in G - N[v]
        let mut visited = vec![false; n];
        let mut component_id = 0;

        for u in 0..n {
            if closed_neighborhood.contains(&u) {
                visited[u] = true; // Skip vertices in N[v]
                components[v][u] = None;
            } else if !visited[u] {
                // Start BFS from u to find its component
                bfs_component(graph, u, &mut visited, &closed_neighborhood, &mut components[v], component_id);
                component_id += 1;
            }
        }
    }

    components
}

/// Performs BFS to mark all vertices in a component.
///
/// Starting from `start`, marks all reachable vertices that are not in `excluded`
/// with the given `component_id`.
fn bfs_component(
    graph: &Graph,
    start: usize,
    visited: &mut [bool],
    excluded: &HashSet<usize>,
    component: &mut [Option<usize>],
    component_id: usize,
) {
    let mut queue = VecDeque::new();
    queue.push_back(start);
    visited[start] = true;
    component[start] = Some(component_id);

    while let Some(v) = queue.pop_front() {
        if let Some(neighbors) = graph.neighbors(v) {
            for neighbor in neighbors {
                if !excluded.contains(&neighbor) && !visited[neighbor] {
                    visited[neighbor] = true;
                    component[neighbor] = Some(component_id);
                    queue.push_back(neighbor);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph_is_at_free() {
        let g = Graph::new(0);
        assert!(is_asteroidal_triple_free(&g));
    }

    #[test]
    fn test_single_vertex_is_at_free() {
        let g = Graph::new(1);
        assert!(is_asteroidal_triple_free(&g));
    }

    #[test]
    fn test_two_vertices_is_at_free() {
        let mut g = Graph::new(2);
        g.add_edge(0, 1).unwrap();
        assert!(is_asteroidal_triple_free(&g));
    }

    #[test]
    fn test_triangle_is_at_free() {
        // Triangle: 0-1, 1-2, 2-0
        // No independent triple exists
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 0).unwrap();
        assert!(is_asteroidal_triple_free(&g));
    }

    #[test]
    fn test_path_p3_is_at_free() {
        // Path: 0-1-2
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        assert!(is_asteroidal_triple_free(&g));
    }

    #[test]
    fn test_path_p4_is_at_free() {
        // Path: 0-1-2-3
        // This is an interval graph, so it's AT-free
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();
        assert!(is_asteroidal_triple_free(&g));
    }

    #[test]
    fn test_cycle_c4_is_not_at_free() {
        // Cycle: 0-1-2-3-0
        // {0, 1, 2} should form an asteroidal triple
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();
        g.add_edge(3, 0).unwrap();
        assert!(!is_asteroidal_triple_free(&g));
    }

    #[test]
    fn test_independent_set_of_three_is_not_at_free() {
        // Three independent vertices with no edges
        // They form an asteroidal triple
        let g = Graph::new(3);
        assert!(!is_asteroidal_triple_free(&g));
    }

    #[test]
    fn test_complete_graph_k4_is_at_free() {
        // Complete graph has no independent vertices
        let mut g = Graph::new(4);
        for u in 0..4 {
            for v in (u + 1)..4 {
                g.add_edge(u, v).unwrap();
            }
        }
        assert!(is_asteroidal_triple_free(&g));
    }

    #[test]
    fn test_star_graph_is_at_free() {
        // Star: center vertex 0 connected to 1, 2, 3
        // Interval graph, so AT-free
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 2).unwrap();
        g.add_edge(0, 3).unwrap();
        assert!(is_asteroidal_triple_free(&g));
    }

    #[test]
    fn test_claw_graph_is_at_free() {
        // Same as star graph - AT-free
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 2).unwrap();
        g.add_edge(0, 3).unwrap();
        assert!(is_asteroidal_triple_free(&g));
    }
}
