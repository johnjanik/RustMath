//! Intersection graph generators
//!
//! This module provides generators for intersection graphs, where vertices
//! represent mathematical objects (intervals, sets, permutations) and edges
//! represent intersections or relationships between these objects.

use crate::graph::Graph;
use std::collections::{HashMap, HashSet};
use std::cmp::{min, max};
use std::hash::Hash;

/// Generate an interval graph from a collection of intervals
///
/// Vertices represent intervals, and two vertices are adjacent if their
/// corresponding intervals intersect (overlap).
///
/// # Arguments
///
/// * `intervals` - A vector of pairs (a, b) representing closed intervals [a, b]
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::intersection::interval_graph;
///
/// let intervals = vec![(0, 2), (1, 3), (4, 6)];
/// let g = interval_graph(&intervals);
/// assert_eq!(g.num_vertices(), 3);
/// assert_eq!(g.num_edges(), 1); // Only (0,2) and (1,3) overlap
/// ```
pub fn interval_graph(intervals: &[(i64, i64)]) -> Graph {
    let n = intervals.len();
    let mut g = Graph::new(n);

    // Normalize intervals and check for intersections
    for i in 0..n {
        for j in (i + 1)..n {
            let (l1, r1) = (min(intervals[i].0, intervals[i].1), max(intervals[i].0, intervals[i].1));
            let (l2, r2) = (min(intervals[j].0, intervals[j].1), max(intervals[j].0, intervals[j].1));

            // Two intervals intersect if neither is completely to the left of the other
            if r1 >= l2 && r2 >= l1 {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate a permutation graph from one or two permutations
///
/// Given a permutation σ, two vertices i < j are adjacent if and only if
/// σ⁻¹(i) > σ⁻¹(j). This corresponds to inversions in the inverse permutation.
///
/// # Arguments
///
/// * `permutation` - A permutation represented as a vector
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::intersection::permutation_graph;
///
/// let perm = vec![2, 4, 1, 3];
/// let g = permutation_graph(&perm);
/// assert_eq!(g.num_vertices(), 4);
/// ```
pub fn permutation_graph(permutation: &[usize]) -> Graph {
    let n = permutation.len();
    let mut g = Graph::new(n);

    // Create inverse permutation
    let mut inv = vec![0; n];
    for (i, &p) in permutation.iter().enumerate() {
        if p < n {
            inv[p] = i;
        }
    }

    // Add edges for inversions in the inverse permutation
    for i in 0..n {
        for j in (i + 1)..n {
            if inv[i] > inv[j] {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate a tolerance graph from tolerance representations
///
/// A tolerance graph where vertices i and j are adjacent when the length
/// of their interval intersection is at least min(t_i, t_j), where t_i
/// and t_j are tolerance values.
///
/// # Arguments
///
/// * `tolerances` - A vector of triples (left, right, tolerance)
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::intersection::tolerance_graph;
///
/// let tol = vec![(0, 4, 2), (1, 3, 1), (5, 8, 2)];
/// let g = tolerance_graph(&tol);
/// assert_eq!(g.num_vertices(), 3);
/// ```
pub fn tolerance_graph(tolerances: &[(i64, i64, i64)]) -> Graph {
    let n = tolerances.len();
    let mut g = Graph::new(n);

    for i in 0..n {
        for j in (i + 1)..n {
            let (l1, r1, t1) = tolerances[i];
            let (l2, r2, t2) = tolerances[j];

            // Validate tolerances are positive
            if t1 <= 0 || t2 <= 0 {
                continue;
            }

            // Calculate intersection length
            let intersection_left = max(l1, l2);
            let intersection_right = min(r1, r2);
            let intersection_length = max(0, intersection_right - intersection_left);

            // Add edge if intersection length >= min(t1, t2)
            if intersection_length >= min(t1, t2) {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate an orthogonal array block graph
///
/// Creates a graph from an orthogonal array where vertices represent blocks
/// and edges connect blocks that share elements.
///
/// This is a simplified implementation that generates a graph based on the
/// structure of an orthogonal array OA(k, n).
///
/// # Arguments
///
/// * `k` - Strength parameter
/// * `n` - Alphabet size
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::intersection::orthogonal_array_block_graph;
///
/// let g = orthogonal_array_block_graph(3, 4);
/// assert!(g.num_vertices() > 0);
/// ```
pub fn orthogonal_array_block_graph(k: usize, n: usize) -> Graph {
    // For a simplified implementation, we construct OA(k, n) as rows
    // Each row is a k-tuple from {0, 1, ..., n-1}

    if k == 0 || n == 0 {
        return Graph::new(0);
    }

    // For OA(k, n), we have n^k rows
    let num_vertices = n.pow(k as u32);
    let mut g = Graph::new(num_vertices);

    // Generate all k-tuples as vertices
    let mut rows = Vec::new();
    generate_tuples(k, n, &mut vec![], &mut rows);

    // For each position and value, create a clique among all rows
    // that have that value at that position
    for pos in 0..k {
        for val in 0..n {
            let mut matching_rows = Vec::new();
            for (idx, row) in rows.iter().enumerate() {
                if row[pos] == val {
                    matching_rows.push(idx);
                }
            }

            // Add edges between all pairs in this clique
            for i in 0..matching_rows.len() {
                for j in (i + 1)..matching_rows.len() {
                    g.add_edge(matching_rows[i], matching_rows[j]).unwrap();
                }
            }
        }
    }

    g
}

/// Helper function to generate all k-tuples from {0, 1, ..., n-1}
fn generate_tuples(k: usize, n: usize, current: &mut Vec<usize>, result: &mut Vec<Vec<usize>>) {
    if current.len() == k {
        result.push(current.clone());
        return;
    }

    for i in 0..n {
        current.push(i);
        generate_tuples(k, n, current, result);
        current.pop();
    }
}

/// Generate a general intersection graph from a collection of sets
///
/// Vertices represent sets, and two vertices are adjacent if and only if
/// their corresponding sets have a non-empty intersection.
///
/// # Arguments
///
/// * `sets` - A vector of sets (represented as vectors of elements)
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::intersection::intersection_graph;
///
/// let sets = vec![
///     vec![1, 2, 3],
///     vec![2, 3, 4],
///     vec![5, 6, 7],
/// ];
/// let g = intersection_graph(&sets);
/// assert_eq!(g.num_vertices(), 3);
/// assert_eq!(g.num_edges(), 1); // Sets 0 and 1 intersect
/// ```
pub fn intersection_graph<T: Eq + Hash + Clone>(sets: &[Vec<T>]) -> Graph {
    let n = sets.len();
    let mut g = Graph::new(n);

    // Convert each vector to a HashSet for efficient intersection checking
    let hash_sets: Vec<HashSet<_>> = sets
        .iter()
        .map(|s| s.iter().collect::<HashSet<_>>())
        .collect();

    // Check each pair of sets for intersection
    for i in 0..n {
        for j in (i + 1)..n {
            // Check if sets have non-empty intersection
            if hash_sets[i].iter().any(|item| hash_sets[j].contains(item)) {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_graph() {
        let intervals = vec![(0, 2), (1, 3), (4, 6)];
        let g = interval_graph(&intervals);
        assert_eq!(g.num_vertices(), 3);
        // (0,2) and (1,3) overlap
        assert_eq!(g.num_edges(), 1);

        // Test with overlapping intervals
        let intervals2 = vec![(0, 3), (1, 4), (2, 5)];
        let g2 = interval_graph(&intervals2);
        assert_eq!(g2.num_vertices(), 3);
        assert_eq!(g2.num_edges(), 3); // All pairs overlap
    }

    #[test]
    fn test_permutation_graph() {
        let perm = vec![2, 0, 3, 1];
        let g = permutation_graph(&perm);
        assert_eq!(g.num_vertices(), 4);
        assert!(g.num_edges() > 0);
    }

    #[test]
    fn test_tolerance_graph() {
        let tol = vec![(0, 4, 2), (1, 3, 1), (5, 8, 2)];
        let g = tolerance_graph(&tol);
        assert_eq!(g.num_vertices(), 3);

        // (0,4) and (1,3) intersect over [1,3] with length 2
        // min(2,1) = 1, so they should be adjacent
        assert!(g.num_edges() >= 1);
    }

    #[test]
    fn test_orthogonal_array_block_graph() {
        let g = orthogonal_array_block_graph(2, 3);
        assert_eq!(g.num_vertices(), 9); // 3^2 = 9
        assert!(g.num_edges() > 0);
    }

    #[test]
    fn test_intersection_graph() {
        let sets = vec![
            vec![1, 2, 3],
            vec![2, 3, 4],
            vec![5, 6, 7],
        ];
        let g = intersection_graph(&sets);
        assert_eq!(g.num_vertices(), 3);
        assert_eq!(g.num_edges(), 1); // Only sets 0 and 1 intersect
    }
}
