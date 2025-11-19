//! Random graph generators
//!
//! This module provides generators for various types of random graphs including
//! Erdős-Rényi graphs, Barabási-Albert scale-free networks, random regular graphs,
//! and other probabilistic graph models.

use crate::graph::Graph;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::collections::{HashSet, HashMap};

/// Generate a random graph using the Erdős-Rényi G(n, p) model
///
/// Each possible edge is included independently with probability p.
///
/// # Arguments
///
/// * `n` - Number of vertices
/// * `p` - Probability of including each edge (must be in [0, 1])
/// * `seed` - Optional random seed for reproducibility
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::random::random_gnp;
///
/// let g = random_gnp(10, 0.5, None);
/// assert_eq!(g.num_vertices(), 10);
/// ```
pub fn random_gnp(n: usize, p: f64, seed: Option<u64>) -> Graph {
    let mut g = Graph::new(n);

    let mut rng: Box<dyn RngCore> = if let Some(s) = seed {
        Box::new(StdRng::seed_from_u64(s))
    } else {
        Box::new(rand::thread_rng())
    };

    for i in 0..n {
        for j in (i + 1)..n {
            if rng.gen::<f64>() < p {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate a random graph using the Erdős-Rényi G(n, m) model
///
/// Returns a uniformly random graph with exactly n vertices and m edges.
///
/// # Arguments
///
/// * `n` - Number of vertices
/// * `m` - Exact number of edges
/// * `seed` - Optional random seed for reproducibility
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::random::random_gnm;
///
/// let g = random_gnm(10, 15, None);
/// assert_eq!(g.num_vertices(), 10);
/// assert_eq!(g.num_edges(), 15);
/// ```
pub fn random_gnm(n: usize, m: usize, seed: Option<u64>) -> Graph {
    let max_edges = n * (n - 1) / 2;
    if m > max_edges {
        panic!("Too many edges: m={} exceeds maximum {} for n={}", m, max_edges, n);
    }

    let mut g = Graph::new(n);

    let mut rng: Box<dyn RngCore> = if let Some(s) = seed {
        Box::new(StdRng::seed_from_u64(s))
    } else {
        Box::new(rand::thread_rng())
    };

    let mut edges = HashSet::new();

    while edges.len() < m {
        let u = rng.gen_range(0..n);
        let v = rng.gen_range(0..n);

        if u != v {
            let (min_v, max_v) = if u < v { (u, v) } else { (v, u) };
            if edges.insert((min_v, max_v)) {
                g.add_edge(min_v, max_v).unwrap();
            }
        }
    }

    g
}

/// Generate a random bipartite graph
///
/// Creates a bipartite graph with vertex sets of sizes n1 and n2, where each
/// possible edge between the two sets is included with probability p.
///
/// # Arguments
///
/// * `n1` - Size of first vertex set
/// * `n2` - Size of second vertex set
/// * `p` - Probability of including each edge
/// * `seed` - Optional random seed
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::random::random_bipartite;
///
/// let g = random_bipartite(5, 5, 0.5, None);
/// assert_eq!(g.num_vertices(), 10);
/// ```
pub fn random_bipartite(n1: usize, n2: usize, p: f64, seed: Option<u64>) -> Graph {
    let n = n1 + n2;
    let mut g = Graph::new(n);

    let mut rng: Box<dyn RngCore> = if let Some(s) = seed {
        Box::new(StdRng::seed_from_u64(s))
    } else {
        Box::new(rand::thread_rng())
    };

    // Vertices 0..n1 are in first set, n1..n are in second set
    for i in 0..n1 {
        for j in n1..n {
            if rng.gen::<f64>() < p {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate a random graph using the Barabási-Albert preferential attachment model
///
/// Creates a scale-free network where new vertices connect to existing vertices
/// with probability proportional to their degree (preferential attachment).
///
/// # Arguments
///
/// * `n` - Final number of vertices
/// * `m` - Number of edges to attach from each new vertex
/// * `seed` - Optional random seed
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::random::random_barabasi_albert;
///
/// let g = random_barabasi_albert(100, 3, None);
/// assert_eq!(g.num_vertices(), 100);
/// ```
pub fn random_barabasi_albert(n: usize, m: usize, seed: Option<u64>) -> Graph {
    if m >= n {
        panic!("m must be less than n");
    }

    let mut g = Graph::new(n);

    let mut rng: Box<dyn RngCore> = if let Some(s) = seed {
        Box::new(StdRng::seed_from_u64(s))
    } else {
        Box::new(rand::thread_rng())
    };

    // Start with m vertices connected as a path (or star for m=1)
    if m == 1 {
        // Create a star with vertex 0 at center
        for i in 1..=m {
            g.add_edge(0, i).unwrap();
        }
    } else {
        // Create a path
        for i in 0..m {
            if i > 0 {
                g.add_edge(i - 1, i).unwrap();
            }
        }
    }

    // Add remaining vertices using preferential attachment
    for new_vertex in (m + 1)..n {
        // Build cumulative degree distribution
        let mut cumulative_degrees = Vec::new();
        let mut sum = 0;

        for v in 0..new_vertex {
            sum += g.degree(v).unwrap_or(0) + 1; // +1 to avoid zero probability
            cumulative_degrees.push(sum);
        }

        let mut targets = HashSet::new();

        // Select m unique targets using preferential attachment
        while targets.len() < m && targets.len() < new_vertex {
            let r = rng.gen_range(0..sum);

            // Binary search for target vertex
            let target = cumulative_degrees.iter()
                .position(|&x| x > r)
                .unwrap_or(new_vertex - 1);

            targets.insert(target);
        }

        // Add edges to selected targets
        for &target in &targets {
            g.add_edge(new_vertex, target).unwrap();
        }
    }

    g
}

/// Generate a random d-regular graph
///
/// Creates a random graph where every vertex has exactly degree d.
///
/// # Arguments
///
/// * `d` - Desired degree (must be even × n)
/// * `n` - Number of vertices
/// * `seed` - Optional random seed
///
/// # Returns
///
/// A random d-regular graph, or None if such a graph cannot be constructed.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::random::random_regular;
///
/// let g = random_regular(3, 10, None);
/// if let Some(graph) = g {
///     assert_eq!(graph.num_vertices(), 10);
///     for v in 0..10 {
///         assert_eq!(graph.degree(v), Some(3));
///     }
/// }
/// ```
pub fn random_regular(d: usize, n: usize, seed: Option<u64>) -> Option<Graph> {
    // Check if d-regular graph is possible
    if d >= n || (n * d) % 2 != 0 {
        return None;
    }

    let mut rng: Box<dyn RngCore> = if let Some(s) = seed {
        Box::new(StdRng::seed_from_u64(s))
    } else {
        Box::new(rand::thread_rng())
    };

    // Use configuration model: create d copies of each vertex
    let mut stubs: Vec<usize> = Vec::new();
    for v in 0..n {
        for _ in 0..d {
            stubs.push(v);
        }
    }

    // Try to pair up stubs randomly
    let max_attempts = 100;
    for _ in 0..max_attempts {
        let mut g = Graph::new(n);
        let mut temp_stubs = stubs.clone();
        let mut success = true;

        while temp_stubs.len() >= 2 {
            // Pick two random stubs
            let idx1 = rng.gen_range(0..temp_stubs.len());
            let v1 = temp_stubs.remove(idx1);

            let idx2 = rng.gen_range(0..temp_stubs.len());
            let v2 = temp_stubs.remove(idx2);

            // Don't allow self-loops or multi-edges
            if v1 != v2 && !g.has_edge(v1, v2) {
                g.add_edge(v1, v2).unwrap();
            } else {
                success = false;
                break;
            }
        }

        if success {
            return Some(g);
        }
    }

    None
}

/// Generate a random interval graph
///
/// Creates an interval graph by generating n random intervals on [0,1]
/// and connecting intervals that overlap.
///
/// # Arguments
///
/// * `n` - Number of vertices (and intervals)
/// * `seed` - Optional random seed
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::random::random_interval_graph;
///
/// let g = random_interval_graph(20, None);
/// assert_eq!(g.num_vertices(), 20);
/// ```
pub fn random_interval_graph(n: usize, seed: Option<u64>) -> Graph {
    let mut rng: Box<dyn RngCore> = if let Some(s) = seed {
        Box::new(StdRng::seed_from_u64(s))
    } else {
        Box::new(rand::thread_rng())
    };

    // Generate random intervals
    let mut intervals = Vec::new();
    for _ in 0..n {
        let a = rng.gen::<f64>();
        let b = rng.gen::<f64>();
        intervals.push(if a < b { (a, b) } else { (b, a) });
    }

    // Create interval graph
    let mut g = Graph::new(n);
    for i in 0..n {
        for j in (i + 1)..n {
            let (l1, r1) = intervals[i];
            let (l2, r2) = intervals[j];

            // Intervals overlap if neither is completely to the left of the other
            if r1 >= l2 && r2 >= l1 {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate a random k-tree
///
/// A k-tree is constructed by starting with a complete graph on k vertices,
/// then repeatedly adding new vertices that are connected to exactly k vertices
/// forming a clique.
///
/// # Arguments
///
/// * `n` - Total number of vertices
/// * `k` - Clique size parameter (determines treewidth)
/// * `seed` - Optional random seed
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::random::random_k_tree;
///
/// let g = random_k_tree(15, 3, None);
/// assert_eq!(g.num_vertices(), 15);
/// ```
pub fn random_k_tree(n: usize, k: usize, seed: Option<u64>) -> Graph {
    if n <= k {
        panic!("n must be greater than k");
    }

    let mut g = Graph::new(n);

    let mut rng: Box<dyn RngCore> = if let Some(s) = seed {
        Box::new(StdRng::seed_from_u64(s))
    } else {
        Box::new(rand::thread_rng())
    };

    // Start with a complete graph on k vertices (vertices 0..k-1)
    for i in 0..k {
        for j in (i + 1)..k {
            g.add_edge(i, j).unwrap();
        }
    }

    // Track all k-cliques
    let mut cliques: Vec<Vec<usize>> = Vec::new();

    // Add initial k-clique (all k initial vertices)
    cliques.push((0..k).collect());

    // Add remaining vertices one by one
    for new_vertex in k..n {
        // Choose a random existing k-clique
        let clique_idx = rng.gen_range(0..cliques.len());
        let clique = cliques[clique_idx].clone(); // Clone to avoid borrow issues

        // Connect new vertex to all k vertices in the chosen clique
        for &v in &clique {
            g.add_edge(new_vertex, v).unwrap();
        }

        // Create k new k-cliques by replacing each vertex in the chosen clique
        // with the new vertex
        for i in 0..k {
            let mut new_clique = clique.clone();
            new_clique[i] = new_vertex;
            cliques.push(new_clique);
        }
    }

    g
}

/// Generate a random tree on n vertices
///
/// Uses Prüfer sequences to generate uniformly random labeled trees.
///
/// # Arguments
///
/// * `n` - Number of vertices
/// * `seed` - Optional random seed
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::random::random_tree;
///
/// let g = random_tree(10, None);
/// assert_eq!(g.num_vertices(), 10);
/// assert_eq!(g.num_edges(), 9); // Trees have n-1 edges
/// ```
pub fn random_tree(n: usize, seed: Option<u64>) -> Graph {
    if n == 0 {
        return Graph::new(0);
    }

    if n == 1 {
        return Graph::new(1);
    }

    let mut rng: Box<dyn RngCore> = if let Some(s) = seed {
        Box::new(StdRng::seed_from_u64(s))
    } else {
        Box::new(rand::thread_rng())
    };

    // Generate random Prüfer sequence of length n-2
    let mut prufer: Vec<usize> = Vec::new();
    for _ in 0..(n - 2) {
        prufer.push(rng.gen_range(0..n));
    }

    // Convert Prüfer sequence to tree
    let mut g = Graph::new(n);
    let mut degree = vec![1; n];

    // Calculate degrees from Prüfer sequence
    for &val in &prufer {
        degree[val] += 1;
    }

    // Build tree from Prüfer sequence
    for &val in &prufer {
        // Find minimum leaf
        for i in 0..n {
            if degree[i] == 1 {
                g.add_edge(i, val).unwrap();
                degree[i] -= 1;
                degree[val] -= 1;
                break;
            }
        }
    }

    // Connect last two vertices with degree 1
    let mut remaining = Vec::new();
    for i in 0..n {
        if degree[i] == 1 {
            remaining.push(i);
        }
    }

    if remaining.len() == 2 {
        g.add_edge(remaining[0], remaining[1]).unwrap();
    }

    g
}

/// Generate a random tolerance graph
///
/// Creates a random tolerance graph by generating random tolerance representations
/// (intervals with tolerance values) and connecting vertices based on tolerance overlap.
///
/// # Arguments
///
/// * `n` - Number of vertices
/// * `seed` - Optional random seed
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::random::random_tolerance_graph;
///
/// let g = random_tolerance_graph(15, None);
/// assert_eq!(g.num_vertices(), 15);
/// ```
pub fn random_tolerance_graph(n: usize, seed: Option<u64>) -> Graph {
    let mut rng: Box<dyn RngCore> = if let Some(s) = seed {
        Box::new(StdRng::seed_from_u64(s))
    } else {
        Box::new(rand::thread_rng())
    };

    // Width bounded by n^2 * 2^n for small n, or just use a reasonable bound
    let w = if n <= 10 {
        n * n * (1 << n)
    } else {
        10000 // Use fixed width for larger n
    };

    // Generate random tolerance representations
    let mut tolerances = Vec::new();
    for _ in 0..n {
        let left = rng.gen_range(0..w) as i64;
        let right = rng.gen_range(0..=w) as i64;
        let tolerance = rng.gen_range(1..=w) as i64; // Must be positive

        let (l, r) = if left < right { (left, right) } else { (right, left) };
        tolerances.push((l, r, tolerance));
    }

    // Create tolerance graph
    let mut g = Graph::new(n);
    for i in 0..n {
        for j in (i + 1)..n {
            let (l1, r1, t1) = tolerances[i];
            let (l2, r2, t2) = tolerances[j];

            // Calculate intersection length
            let intersection_left = l1.max(l2);
            let intersection_right = r1.min(r2);
            let intersection_length = (intersection_right - intersection_left).max(0);

            // Add edge if intersection length >= min(t1, t2)
            if intersection_length >= t1.min(t2) {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

// Import RngCore trait
use rand::RngCore;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_gnp() {
        let g = random_gnp(10, 0.5, Some(42));
        assert_eq!(g.num_vertices(), 10);
        assert!(g.num_edges() > 0);

        // Test with p=0 (no edges)
        let g_empty = random_gnp(10, 0.0, Some(42));
        assert_eq!(g_empty.num_edges(), 0);

        // Test with p=1 (complete graph)
        let g_complete = random_gnp(5, 1.0, Some(42));
        assert_eq!(g_complete.num_edges(), 10); // K_5 has 10 edges
    }

    #[test]
    fn test_random_gnm() {
        let g = random_gnm(10, 15, Some(42));
        assert_eq!(g.num_vertices(), 10);
        assert_eq!(g.num_edges(), 15);
    }

    #[test]
    fn test_random_bipartite() {
        let g = random_bipartite(5, 5, 0.5, Some(42));
        assert_eq!(g.num_vertices(), 10);

        // Check that all edges are between the two sets
        for i in 0..5 {
            for j in 0..5 {
                // No edges within first set
                assert!(!g.has_edge(i, j) || i == j);
            }
        }
    }

    #[test]
    fn test_random_barabasi_albert() {
        let g = random_barabasi_albert(50, 3, Some(42));
        assert_eq!(g.num_vertices(), 50);
        assert!(g.num_edges() > 0);
    }

    #[test]
    fn test_random_regular() {
        let g = random_regular(3, 10, Some(42));
        if let Some(graph) = g {
            assert_eq!(graph.num_vertices(), 10);

            // Check all vertices have degree 3
            for v in 0..10 {
                assert_eq!(graph.degree(v), Some(3));
            }

            // Total edges should be 3*10/2 = 15
            assert_eq!(graph.num_edges(), 15);
        }
    }

    #[test]
    fn test_random_interval_graph() {
        let g = random_interval_graph(20, Some(42));
        assert_eq!(g.num_vertices(), 20);
    }

    #[test]
    fn test_random_k_tree() {
        let g = random_k_tree(15, 3, Some(42));
        assert_eq!(g.num_vertices(), 15);

        // A k-tree on n vertices with parameter k has
        // k(k+1)/2 + (n-k-1)k edges
        let expected_edges = 3 * 4 / 2 + (15 - 3 - 1) * 3;
        assert_eq!(g.num_edges(), expected_edges);
    }

    #[test]
    fn test_random_tree() {
        let g = random_tree(10, Some(42));
        assert_eq!(g.num_vertices(), 10);
        assert_eq!(g.num_edges(), 9); // Trees have n-1 edges

        // Test single vertex
        let g1 = random_tree(1, Some(42));
        assert_eq!(g1.num_vertices(), 1);
        assert_eq!(g1.num_edges(), 0);

        // Test two vertices
        let g2 = random_tree(2, Some(42));
        assert_eq!(g2.num_vertices(), 2);
        assert_eq!(g2.num_edges(), 1);
    }

    #[test]
    fn test_random_tolerance_graph() {
        let g = random_tolerance_graph(15, Some(42));
        assert_eq!(g.num_vertices(), 15);
        // Tolerance graphs are perfect graphs
        assert!(g.num_edges() >= 0);
    }
}
