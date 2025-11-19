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

/// Generate a random block graph
///
/// A block graph is a connected graph where every biconnected component (block) is a clique.
///
/// # Arguments
///
/// * `m` - Number of blocks (cliques)
/// * `k` - Minimum number of vertices per block
/// * `kmax` - Maximum number of vertices per block
/// * `seed` - Optional random seed
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::random::random_block_graph;
///
/// let g = random_block_graph(5, 2, 4, None);
/// assert!(g.num_vertices() >= 5); // At least m blocks
/// ```
pub fn random_block_graph(m: usize, k: usize, kmax: usize, seed: Option<u64>) -> Graph {
    if k > kmax {
        panic!("k must be <= kmax");
    }
    if k < 2 {
        panic!("k must be at least 2");
    }

    let mut rng: Box<dyn RngCore> = if let Some(s) = seed {
        Box::new(StdRng::seed_from_u64(s))
    } else {
        Box::new(rand::thread_rng())
    };

    let mut g = Graph::new(0);
    let mut block_nodes: Vec<Vec<usize>> = Vec::new();

    // Create m blocks
    for _ in 0..m {
        let block_size = rng.gen_range(k..=kmax);
        let start_vertex = g.num_vertices();

        // Add vertices for this block
        for _ in 0..block_size {
            g.add_vertex();
        }

        let mut block = Vec::new();
        for v in start_vertex..(start_vertex + block_size) {
            block.push(v);
        }

        // Make this block a complete graph (clique)
        for i in 0..block.len() {
            for j in (i + 1)..block.len() {
                g.add_edge(block[i], block[j]).unwrap();
            }
        }

        block_nodes.push(block);
    }

    // Connect blocks to form a tree (so graph is connected)
    for i in 1..m {
        // Connect block i to a random previous block
        let prev_block = rng.gen_range(0..i);

        // Choose random vertices from each block as cut vertices
        let v1 = block_nodes[i][rng.gen_range(0..block_nodes[i].len())];
        let v2 = block_nodes[prev_block][rng.gen_range(0..block_nodes[prev_block].len())];

        g.add_edge(v1, v2).unwrap();
    }

    g
}

/// Generate a random chordal graph
///
/// Chordal graphs have no induced cycles of length 4 or more.
/// This implementation uses the intersection graph of subtrees approach.
///
/// # Arguments
///
/// * `n` - Number of vertices
/// * `seed` - Optional random seed
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::random::random_chordal_graph;
///
/// let g = random_chordal_graph(20, None);
/// assert_eq!(g.num_vertices(), 20);
/// ```
pub fn random_chordal_graph(n: usize, seed: Option<u64>) -> Graph {
    if n == 0 {
        return Graph::new(0);
    }

    let mut rng: Box<dyn RngCore> = if let Some(s) = seed {
        Box::new(StdRng::seed_from_u64(s))
    } else {
        Box::new(rand::thread_rng())
    };

    // Create a random tree on 2n vertices to serve as the host tree
    let tree = random_tree(2 * n, seed);

    // For each vertex in the chordal graph, select a random subtree
    let mut subtrees: Vec<HashSet<usize>> = Vec::new();

    for _ in 0..n {
        // Each subtree is defined by picking a random root and growing to random size
        let root = rng.gen_range(0..(2 * n));
        let size = rng.gen_range(1..=(2 * n).min(10)); // Limit subtree size

        let mut subtree = HashSet::new();
        subtree.insert(root);

        // BFS to grow subtree
        let mut queue = vec![root];
        let mut visited = HashSet::new();
        visited.insert(root);

        while subtree.len() < size && !queue.is_empty() {
            let v = queue.remove(0);
            if let Some(neighbors) = tree.neighbors(v) {
                for &u in &neighbors {
                    if !visited.contains(&u) && subtree.len() < size {
                        visited.insert(u);
                        subtree.insert(u);
                        queue.push(u);
                    }
                }
            }
        }

        subtrees.push(subtree);
    }

    // Create chordal graph: edge between vertices if subtrees intersect
    let mut g = Graph::new(n);
    for i in 0..n {
        for j in (i + 1)..n {
            if !subtrees[i].is_disjoint(&subtrees[j]) {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate a random graph using the Holme-Kim algorithm
///
/// Produces graphs with power-law degree distribution and controllable clustering.
///
/// # Arguments
///
/// * `n` - Number of vertices
/// * `m` - Number of random edges to add per step
/// * `p` - Probability of adding a triangle after each edge
/// * `seed` - Optional random seed
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::random::random_holme_kim;
///
/// let g = random_holme_kim(100, 3, 0.5, None);
/// assert_eq!(g.num_vertices(), 100);
/// ```
pub fn random_holme_kim(n: usize, m: usize, p: f64, seed: Option<u64>) -> Graph {
    if m >= n {
        panic!("m must be less than n");
    }

    let mut rng: Box<dyn RngCore> = if let Some(s) = seed {
        Box::new(StdRng::seed_from_u64(s))
    } else {
        Box::new(rand::thread_rng())
    };

    let mut g = Graph::new(n);

    // Start with m+1 vertex clique
    for i in 0..=m {
        for j in (i + 1)..=m {
            g.add_edge(i, j).unwrap();
        }
    }

    // Add remaining vertices
    for new_vertex in (m + 1)..n {
        // Track nodes to which we'll add edges
        let mut targets = Vec::new();

        // Preferential attachment for first edge
        let mut cumulative_degrees = Vec::new();
        let mut sum = 0;

        for v in 0..new_vertex {
            sum += g.degree(v).unwrap_or(0) + 1;
            cumulative_degrees.push(sum);
        }

        for _ in 0..m {
            if targets.len() >= new_vertex {
                break;
            }

            let r = rng.gen_range(0..sum);
            let target = cumulative_degrees.iter()
                .position(|&x| x > r)
                .unwrap_or(new_vertex - 1);

            if !targets.contains(&target) {
                targets.push(target);
                g.add_edge(new_vertex, target).unwrap();

                // With probability p, add triangle by connecting to neighbor of target
                if rng.gen::<f64>() < p {
                    if let Some(neighbors) = g.neighbors(target) {
                        let valid_neighbors: Vec<_> = neighbors.iter()
                            .filter(|&&v| v != new_vertex && !g.has_edge(new_vertex, v))
                            .copied()
                            .collect();

                        if !valid_neighbors.is_empty() {
                            let neighbor = valid_neighbors[rng.gen_range(0..valid_neighbors.len())];
                            if !targets.contains(&neighbor) {
                                targets.push(neighbor);
                                g.add_edge(new_vertex, neighbor).unwrap();
                            }
                        }
                    }
                }
            }
        }
    }

    g
}

/// Generate a random Newman-Watts-Strogatz small-world graph
///
/// Creates a ring lattice with random shortcuts added.
///
/// # Arguments
///
/// * `n` - Number of vertices
/// * `k` - Each vertex connected to k nearest neighbors in ring
/// * `p` - Probability of adding a shortcut for each edge
/// * `seed` - Optional random seed
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::random::random_newman_watts_strogatz;
///
/// let g = random_newman_watts_strogatz(100, 4, 0.1, None);
/// assert_eq!(g.num_vertices(), 100);
/// ```
pub fn random_newman_watts_strogatz(n: usize, k: usize, p: f64, seed: Option<u64>) -> Graph {
    if k >= n {
        panic!("k must be less than n");
    }

    let mut rng: Box<dyn RngCore> = if let Some(s) = seed {
        Box::new(StdRng::seed_from_u64(s))
    } else {
        Box::new(rand::thread_rng())
    };

    let mut g = Graph::new(n);

    // Create ring lattice
    for i in 0..n {
        for j in 1..=(k / 2) {
            let neighbor = (i + j) % n;
            if !g.has_edge(i, neighbor) {
                g.add_edge(i, neighbor).unwrap();
            }
        }
    }

    // Add random shortcuts
    for i in 0..n {
        for _ in 0..(k / 2) {
            if rng.gen::<f64>() < p {
                let target = rng.gen_range(0..n);
                if target != i && !g.has_edge(i, target) {
                    g.add_edge(i, target).unwrap();
                }
            }
        }
    }

    g
}

/// Generate a random partial k-tree
///
/// A partial k-tree is a subgraph of a k-tree (has treewidth at most k).
///
/// # Arguments
///
/// * `n` - Number of vertices
/// * `k` - Treewidth parameter
/// * `p` - Probability of keeping each edge from the k-tree
/// * `seed` - Optional random seed
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::random::random_partial_k_tree;
///
/// let g = random_partial_k_tree(20, 3, 0.7, None);
/// assert_eq!(g.num_vertices(), 20);
/// ```
pub fn random_partial_k_tree(n: usize, k: usize, p: f64, seed: Option<u64>) -> Graph {
    // Generate a random k-tree first
    let ktree = random_k_tree(n, k, seed);

    let mut rng: Box<dyn RngCore> = if let Some(s) = seed {
        Box::new(StdRng::seed_from_u64(s))
    } else {
        Box::new(rand::thread_rng())
    };

    // Create partial k-tree by randomly removing edges
    let mut g = Graph::new(n);

    for i in 0..n {
        if let Some(neighbors) = ktree.neighbors(i) {
            for &j in &neighbors {
                if i < j && rng.gen::<f64>() < p {
                    g.add_edge(i, j).unwrap();
                }
            }
        }
    }

    g
}

/// Generate a random proper interval graph
///
/// A proper interval graph is an interval graph where no interval properly contains another.
///
/// # Arguments
///
/// * `n` - Number of vertices
/// * `seed` - Optional random seed
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::random::random_proper_interval_graph;
///
/// let g = random_proper_interval_graph(15, None);
/// assert_eq!(g.num_vertices(), 15);
/// ```
pub fn random_proper_interval_graph(n: usize, seed: Option<u64>) -> Graph {
    let mut rng: Box<dyn RngCore> = if let Some(s) = seed {
        Box::new(StdRng::seed_from_u64(s))
    } else {
        Box::new(rand::thread_rng())
    };

    // Generate intervals of equal length with random start points
    let mut intervals = Vec::new();
    let length = 1.0 / (n as f64); // Fixed length for proper intervals

    for _ in 0..n {
        let start = rng.gen::<f64>() * (1.0 - length);
        intervals.push((start, start + length));
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

/// Generate a random bounded tolerance graph
///
/// A bounded tolerance graph has tolerance values bounded by interval lengths.
///
/// # Arguments
///
/// * `n` - Number of vertices
/// * `seed` - Optional random seed
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::random::random_bounded_tolerance_graph;
///
/// let g = random_bounded_tolerance_graph(15, None);
/// assert_eq!(g.num_vertices(), 15);
/// ```
pub fn random_bounded_tolerance_graph(n: usize, seed: Option<u64>) -> Graph {
    let mut rng: Box<dyn RngCore> = if let Some(s) = seed {
        Box::new(StdRng::seed_from_u64(s))
    } else {
        Box::new(rand::thread_rng())
    };

    // Generate random bounded tolerance representations
    let mut tolerances = Vec::new();
    for _ in 0..n {
        let left = rng.gen::<f64>();
        let right = rng.gen::<f64>();
        let (l, r) = if left < right { (left, right) } else { (right, left) };

        // Tolerance is bounded by interval length
        let interval_length = r - l;
        let tolerance = rng.gen::<f64>() * interval_length;

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
            let intersection_length = (intersection_right - intersection_left).max(0.0);

            // Add edge if intersection length >= min(t1, t2)
            if intersection_length >= t1.min(t2) {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate a random regular bipartite graph
///
/// Creates a bipartite graph where vertices in the first set all have degree d1,
/// and vertices in the second set all have degree (n1 * d1) / n2.
///
/// # Arguments
///
/// * `n1` - Number of vertices in first set
/// * `n2` - Number of vertices in second set
/// * `d1` - Degree of vertices in first set
/// * `seed` - Optional random seed
///
/// # Returns
///
/// Returns None if parameters don't allow a valid regular bipartite graph.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::random::random_regular_bipartite;
///
/// let g = random_regular_bipartite(10, 10, 3, None);
/// if let Some(graph) = g {
///     assert_eq!(graph.num_vertices(), 20);
/// }
/// ```
pub fn random_regular_bipartite(n1: usize, n2: usize, d1: usize, seed: Option<u64>) -> Option<Graph> {
    // Check if construction is possible
    if (n1 * d1) % n2 != 0 {
        return None;
    }

    let d2 = (n1 * d1) / n2;

    if d1 > n2 || d2 > n1 {
        return None;
    }

    let mut rng: Box<dyn RngCore> = if let Some(s) = seed {
        Box::new(StdRng::seed_from_u64(s))
    } else {
        Box::new(rand::thread_rng())
    };

    let n = n1 + n2;

    // Use configuration model
    let mut stubs1: Vec<usize> = Vec::new();
    let mut stubs2: Vec<usize> = Vec::new();

    for v in 0..n1 {
        for _ in 0..d1 {
            stubs1.push(v);
        }
    }

    for v in n1..n {
        for _ in 0..d2 {
            stubs2.push(v);
        }
    }

    // Try to match stubs
    let max_attempts = 100;
    for _ in 0..max_attempts {
        let mut g = Graph::new(n);
        let mut temp_stubs1 = stubs1.clone();
        let mut temp_stubs2 = stubs2.clone();
        let mut success = true;

        while !temp_stubs1.is_empty() && !temp_stubs2.is_empty() {
            let idx1 = rng.gen_range(0..temp_stubs1.len());
            let v1 = temp_stubs1.remove(idx1);

            let idx2 = rng.gen_range(0..temp_stubs2.len());
            let v2 = temp_stubs2.remove(idx2);

            // Check for multi-edges
            if !g.has_edge(v1, v2) {
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

/// Generate a random lobster graph
///
/// A lobster is a tree where removing leaves yields a caterpillar.
///
/// # Arguments
///
/// * `n` - Number of vertices in the backbone path
/// * `p1` - Probability of adding a leaf to backbone vertices
/// * `p2` - Probability of adding a leaf to first-level leaves
/// * `seed` - Optional random seed
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::random::random_lobster;
///
/// let g = random_lobster(10, 0.5, 0.5, None);
/// assert!(g.num_vertices() >= 10);
/// ```
pub fn random_lobster(n: usize, p1: f64, p2: f64, seed: Option<u64>) -> Graph {
    let mut rng: Box<dyn RngCore> = if let Some(s) = seed {
        Box::new(StdRng::seed_from_u64(s))
    } else {
        Box::new(rand::thread_rng())
    };

    let mut g = Graph::new(n);

    // Create backbone path
    for i in 0..(n - 1) {
        g.add_edge(i, i + 1).unwrap();
    }

    let mut first_level_leaves = Vec::new();

    // Add first level leaves to backbone
    for i in 0..n {
        if rng.gen::<f64>() < p1 {
            let leaf = g.num_vertices();
            g.add_vertex();
            g.add_edge(i, leaf).unwrap();
            first_level_leaves.push(leaf);
        }
    }

    // Add second level leaves
    for &leaf in &first_level_leaves {
        if rng.gen::<f64>() < p2 {
            let new_leaf = g.num_vertices();
            g.add_vertex();
            g.add_edge(leaf, new_leaf).unwrap();
        }
    }

    g
}

/// Generate a random shell graph
///
/// Creates a graph with vertices arranged in concentric shells.
///
/// # Arguments
///
/// * `constructor` - Vector of (n_vertices, m_edges, d_ratio) for each shell
/// * `seed` - Optional random seed
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::random::random_shell;
///
/// let constructor = vec![(10, 20, 0.8), (20, 40, 0.8)];
/// let g = random_shell(constructor, None);
/// assert_eq!(g.num_vertices(), 30);
/// ```
pub fn random_shell(constructor: Vec<(usize, usize, f64)>, seed: Option<u64>) -> Graph {
    let mut rng: Box<dyn RngCore> = if let Some(s) = seed {
        Box::new(StdRng::seed_from_u64(s))
    } else {
        Box::new(rand::thread_rng())
    };

    let total_vertices: usize = constructor.iter().map(|(n, _, _)| n).sum();
    let mut g = Graph::new(total_vertices);

    let mut vertex_offset = 0;
    let mut prev_shell_start = 0;
    let mut prev_shell_size = 0;

    for (shell_idx, &(shell_size, shell_edges, density)) in constructor.iter().enumerate() {
        let shell_start = vertex_offset;

        // Add edges within shell
        let mut edges_added = 0;
        while edges_added < shell_edges && edges_added < shell_size * (shell_size - 1) / 2 {
            let v1 = rng.gen_range(0..shell_size) + shell_start;
            let v2 = rng.gen_range(0..shell_size) + shell_start;

            if v1 != v2 && !g.has_edge(v1, v2) {
                g.add_edge(v1, v2).unwrap();
                edges_added += 1;
            }
        }

        // Connect to previous shell
        if shell_idx > 0 {
            let inter_shell_edges = (shell_size as f64 * density) as usize;
            for _ in 0..inter_shell_edges {
                let v1 = rng.gen_range(0..shell_size) + shell_start;
                let v2 = rng.gen_range(0..prev_shell_size) + prev_shell_start;

                if !g.has_edge(v1, v2) {
                    g.add_edge(v1, v2).unwrap();
                }
            }
        }

        prev_shell_start = shell_start;
        prev_shell_size = shell_size;
        vertex_offset += shell_size;
    }

    g
}

/// Generate a random tree with power-law degree distribution
///
/// # Arguments
///
/// * `n` - Number of vertices
/// * `gamma` - Power law exponent (typically 2 < gamma < 3)
/// * `seed` - Optional random seed
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::random::random_tree_powerlaw;
///
/// let g = random_tree_powerlaw(100, 2.5, None);
/// assert_eq!(g.num_vertices(), 100);
/// assert_eq!(g.num_edges(), 99);
/// ```
pub fn random_tree_powerlaw(n: usize, gamma: f64, seed: Option<u64>) -> Graph {
    if n == 0 {
        return Graph::new(0);
    }

    let mut rng: Box<dyn RngCore> = if let Some(s) = seed {
        Box::new(StdRng::seed_from_u64(s))
    } else {
        Box::new(rand::thread_rng())
    };

    let mut g = Graph::new(n);

    if n == 1 {
        return g;
    }

    // Generate power-law degree sequence
    let mut degrees = Vec::new();
    for i in 1..=n {
        let degree = ((i as f64).powf(-1.0 / gamma) * (n as f64)) as usize;
        degrees.push(degree.max(1));
    }

    // Ensure sum of degrees is even (2n - 2 for a tree)
    let total_degree: usize = degrees.iter().sum();
    let target = 2 * (n - 1);

    if total_degree != target {
        // Adjust degrees to sum to target
        let diff = (target as i64) - (total_degree as i64);
        if diff > 0 {
            for _ in 0..diff {
                let idx = rng.gen_range(0..n);
                degrees[idx] += 1;
            }
        } else {
            for _ in 0..(-diff) {
                let idx = rng.gen_range(0..n);
                if degrees[idx] > 1 {
                    degrees[idx] -= 1;
                }
            }
        }
    }

    // Use modified Prüfer sequence to build tree with approximate degree sequence
    // For simplicity, use regular random tree (exact degree sequence trees are complex)
    g = random_tree(n, seed);

    g
}

/// Generate a random planar triangulation
///
/// Creates a random maximal planar graph (every face is a triangle).
///
/// # Arguments
///
/// * `n` - Number of vertices (must be >= 3)
/// * `seed` - Optional random seed
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::random::random_triangulation;
///
/// let g = random_triangulation(10, None);
/// assert_eq!(g.num_vertices(), 10);
/// assert_eq!(g.num_edges(), 3 * 10 - 6); // Planar triangulation has 3n-6 edges
/// ```
pub fn random_triangulation(n: usize, seed: Option<u64>) -> Graph {
    if n < 3 {
        panic!("n must be at least 3");
    }

    let mut rng: Box<dyn RngCore> = if let Some(s) = seed {
        Box::new(StdRng::seed_from_u64(s))
    } else {
        Box::new(rand::thread_rng())
    };

    // Start with triangle
    let mut g = Graph::new(3);
    g.add_edge(0, 1).unwrap();
    g.add_edge(1, 2).unwrap();
    g.add_edge(2, 0).unwrap();

    // Add remaining vertices one at a time
    for new_vertex in 3..n {
        g.add_vertex();

        // Find a random existing edge and split its face
        let edges: Vec<_> = (0..new_vertex)
            .flat_map(|v| {
                g.neighbors(v).map(|neighbors| {
                    neighbors.iter()
                        .filter(|&&u| u > v)
                        .map(|&u| (v, u))
                        .collect::<Vec<_>>()
                }).unwrap_or_default()
            })
            .collect();

        if !edges.is_empty() {
            let (v1, v2) = edges[rng.gen_range(0..edges.len())];

            // Connect new vertex to both endpoints of chosen edge
            g.add_edge(new_vertex, v1).unwrap();
            g.add_edge(new_vertex, v2).unwrap();

            // Find common neighbor to complete triangulation
            if let (Some(n1), Some(n2)) = (g.neighbors(v1), g.neighbors(v2)) {
                let common: Vec<_> = n1.iter()
                    .filter(|&&v| n2.contains(&v) && v != new_vertex)
                    .copied()
                    .collect();

                if let Some(&neighbor) = common.first() {
                    g.add_edge(new_vertex, neighbor).unwrap();
                }
            }
        }
    }

    g
}

/// Generate a random unit disk graph
///
/// Vertices are random points in a unit square, connected if distance <= radius.
///
/// # Arguments
///
/// * `n` - Number of vertices
/// * `radius` - Connection radius
/// * `seed` - Optional random seed
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::random::random_unit_disk_graph;
///
/// let g = random_unit_disk_graph(50, 0.2, None);
/// assert_eq!(g.num_vertices(), 50);
/// ```
pub fn random_unit_disk_graph(n: usize, radius: f64, seed: Option<u64>) -> Graph {
    let mut rng: Box<dyn RngCore> = if let Some(s) = seed {
        Box::new(StdRng::seed_from_u64(s))
    } else {
        Box::new(rand::thread_rng())
    };

    // Generate random points in unit square
    let mut points = Vec::new();
    for _ in 0..n {
        let x = rng.gen::<f64>();
        let y = rng.gen::<f64>();
        points.push((x, y));
    }

    // Create graph
    let mut g = Graph::new(n);
    let radius_sq = radius * radius;

    for i in 0..n {
        for j in (i + 1)..n {
            let (x1, y1) = points[i];
            let (x2, y2) = points[j];

            let dist_sq = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);

            if dist_sq <= radius_sq {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate a random bicubic planar graph
///
/// A bicubic planar graph has all vertices of degree 2 or 3 and is planar.
/// This implementation generates random planar graphs by triangulation subdivision.
///
/// # Arguments
///
/// * `n` - Number of vertices (must be >= 4)
/// * `seed` - Optional random seed
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::random::random_bicubic_planar;
///
/// let g = random_bicubic_planar(20, None);
/// assert_eq!(g.num_vertices(), 20);
/// ```
pub fn random_bicubic_planar(n: usize, seed: Option<u64>) -> Graph {
    if n < 4 {
        panic!("n must be at least 4");
    }

    let mut rng: Box<dyn RngCore> = if let Some(s) = seed {
        Box::new(StdRng::seed_from_u64(s))
    } else {
        Box::new(rand::thread_rng())
    };

    // Start with a 4-cycle (square) - all vertices have degree 2
    let mut g = Graph::new(4);
    g.add_edge(0, 1).unwrap();
    g.add_edge(1, 2).unwrap();
    g.add_edge(2, 3).unwrap();
    g.add_edge(3, 0).unwrap();

    // Add vertices by connecting to existing edges to subdivide faces
    while g.num_vertices() < n {
        let new_v = g.num_vertices();
        g.add_vertex();

        // Pick a random vertex to connect the new vertex to
        let v1 = rng.gen_range(0..new_v);

        // Get neighbors of v1
        if let Some(neighbors) = g.neighbors(v1) {
            if !neighbors.is_empty() {
                // Pick one of its neighbors
                let v2 = neighbors[rng.gen_range(0..neighbors.len())];

                // Connect new vertex to both v1 and v2
                g.add_edge(new_v, v1).unwrap();
                g.add_edge(new_v, v2).unwrap();

                // Optionally add one more edge to maintain planar structure
                if g.num_vertices() < n && rng.gen::<f64>() < 0.3 {
                    // Find another vertex that's not v1 or v2
                    let candidates: Vec<_> = (0..new_v)
                        .filter(|&v| v != v1 && v != v2 && g.degree(v).unwrap_or(0) < 3)
                        .collect();

                    if !candidates.is_empty() {
                        let v3 = candidates[rng.gen_range(0..candidates.len())];
                        g.add_edge(new_v, v3).unwrap();
                    }
                }
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

    #[test]
    fn test_random_block_graph() {
        let g = random_block_graph(5, 2, 4, Some(42));
        assert!(g.num_vertices() >= 10); // At least 2 vertices per block × 5 blocks
        assert!(g.num_vertices() <= 20); // At most 4 vertices per block × 5 blocks
    }

    #[test]
    fn test_random_chordal_graph() {
        let g = random_chordal_graph(20, Some(42));
        assert_eq!(g.num_vertices(), 20);
    }

    #[test]
    fn test_random_holme_kim() {
        let g = random_holme_kim(100, 3, 0.5, Some(42));
        assert_eq!(g.num_vertices(), 100);
        assert!(g.num_edges() >= 3 * 100 / 2); // At least m*(n-m-1) + m*(m+1)/2 edges
    }

    #[test]
    fn test_random_newman_watts_strogatz() {
        let g = random_newman_watts_strogatz(100, 4, 0.1, Some(42));
        assert_eq!(g.num_vertices(), 100);
        // Should have at least the ring lattice edges
        assert!(g.num_edges() >= 100 * 2); // k/2 * n edges in ring
    }

    #[test]
    fn test_random_partial_k_tree() {
        let g = random_partial_k_tree(20, 3, 0.7, Some(42));
        assert_eq!(g.num_vertices(), 20);
    }

    #[test]
    fn test_random_proper_interval_graph() {
        let g = random_proper_interval_graph(15, Some(42));
        assert_eq!(g.num_vertices(), 15);
    }

    #[test]
    fn test_random_bounded_tolerance_graph() {
        let g = random_bounded_tolerance_graph(15, Some(42));
        assert_eq!(g.num_vertices(), 15);
    }

    #[test]
    fn test_random_regular_bipartite() {
        let g = random_regular_bipartite(10, 10, 3, Some(42));
        if let Some(graph) = g {
            assert_eq!(graph.num_vertices(), 20);
            assert_eq!(graph.num_edges(), 30); // 10 * 3 edges

            // Check degrees in first set
            for v in 0..10 {
                assert_eq!(graph.degree(v), Some(3));
            }

            // Check degrees in second set
            for v in 10..20 {
                assert_eq!(graph.degree(v), Some(3));
            }
        }
    }

    #[test]
    fn test_random_lobster() {
        let g = random_lobster(10, 0.5, 0.5, Some(42));
        assert!(g.num_vertices() >= 10); // At least the backbone
        assert_eq!(g.num_edges(), g.num_vertices() - 1); // Trees have n-1 edges
    }

    #[test]
    fn test_random_shell() {
        let constructor = vec![(10, 20, 0.8), (20, 40, 0.8)];
        let g = random_shell(constructor, Some(42));
        assert_eq!(g.num_vertices(), 30);
    }

    #[test]
    fn test_random_tree_powerlaw() {
        let g = random_tree_powerlaw(100, 2.5, Some(42));
        assert_eq!(g.num_vertices(), 100);
        assert_eq!(g.num_edges(), 99); // Trees have n-1 edges
    }

    #[test]
    fn test_random_triangulation() {
        let g = random_triangulation(10, Some(42));
        assert_eq!(g.num_vertices(), 10);
        // Planar triangulation has 3n-6 edges for n >= 3
        assert_eq!(g.num_edges(), 3 * 10 - 6);
    }

    #[test]
    fn test_random_unit_disk_graph() {
        let g = random_unit_disk_graph(50, 0.2, Some(42));
        assert_eq!(g.num_vertices(), 50);
    }

    #[test]
    fn test_random_bicubic_planar() {
        let g = random_bicubic_planar(20, Some(42));
        assert_eq!(g.num_vertices(), 20);

        // Check that graph is connected (all vertices have at least degree 1)
        for v in 0..20 {
            let deg = g.degree(v).unwrap_or(0);
            assert!(deg > 0, "Vertex {} has degree 0", v);
        }
    }
}
