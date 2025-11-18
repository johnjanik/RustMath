//! Graph centrality measures
//!
//! Corresponds to sage.graphs.centrality
//!
//! This module provides algorithms for computing various centrality metrics
//! that measure the importance of vertices in a graph.

use crate::Graph;
use std::collections::{HashMap, VecDeque};
use rand::seq::SliceRandom;
use rand::thread_rng;

/// Compute betweenness centrality for all vertices
///
/// Betweenness centrality measures the number of shortest paths passing through each vertex.
/// For a vertex v, it's defined as:
///   C_B(v) = Σ(σ_st(v) / σ_st)
/// where σ_st is the number of shortest paths from s to t, and σ_st(v) is the number
/// of those paths passing through v.
///
/// Uses Brandes' algorithm for efficient computation.
///
/// Corresponds to sage.graphs.centrality.centrality_betweenness
///
/// # Returns
/// HashMap mapping vertex index to betweenness centrality score
pub fn centrality_betweenness(graph: &Graph) -> HashMap<usize, f64> {
    let n = graph.num_vertices();
    let mut betweenness = vec![0.0; n];

    // Brandes' algorithm
    for s in 0..n {
        let mut stack = Vec::new();
        let mut paths: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut sigma = vec![0.0; n];
        sigma[s] = 1.0;
        let mut dist = vec![-1i32; n];
        dist[s] = 0;

        let mut queue = VecDeque::new();
        queue.push_back(s);

        // BFS to find shortest paths
        while let Some(v) = queue.pop_front() {
            stack.push(v);

            if let Some(neighbors) = graph.neighbors(v) {
                for &w in &neighbors {
                    // First time we see w?
                    if dist[w] < 0 {
                        dist[w] = dist[v] + 1;
                        queue.push_back(w);
                    }

                    // Shortest path to w via v?
                    if dist[w] == dist[v] + 1 {
                        sigma[w] += sigma[v];
                        paths.entry(w).or_insert_with(Vec::new).push(v);
                    }
                }
            }
        }

        // Accumulation
        let mut delta = vec![0.0; n];
        while let Some(w) = stack.pop() {
            if let Some(predecessors) = paths.get(&w) {
                for &v in predecessors {
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                }
            }
            if w != s {
                betweenness[w] += delta[w];
            }
        }
    }

    // Normalize for undirected graphs
    if !graph.is_directed() {
        for b in &mut betweenness {
            *b /= 2.0;
        }
    }

    betweenness.into_iter().enumerate().map(|(i, b)| (i, b)).collect()
}

/// Compute closeness centrality for the top k vertices
///
/// Closeness centrality of a vertex is defined as the reciprocal of the average
/// shortest path distance to all other vertices:
///   C_C(v) = (n-1) / Σ d(v,u)
///
/// This function computes closeness for all vertices and returns the top k.
///
/// Corresponds to sage.graphs.centrality.centrality_closeness_top_k
///
/// # Returns
/// Vector of (vertex, closeness_score) pairs for the top k vertices
pub fn centrality_closeness_top_k(graph: &Graph, k: usize) -> Vec<(usize, f64)> {
    let n = graph.num_vertices();
    if n == 0 {
        return vec![];
    }

    let mut closeness = Vec::new();

    for v in 0..n {
        let distances = compute_distances_from(graph, v);

        // Sum of distances to all reachable vertices
        let mut sum_dist = 0.0;
        let mut reachable = 0;

        for &d in &distances {
            if d >= 0 {
                sum_dist += d as f64;
                reachable += 1;
            }
        }

        // Closeness centrality
        let cc = if sum_dist > 0.0 && reachable > 1 {
            (reachable - 1) as f64 / sum_dist
        } else {
            0.0
        };

        closeness.push((v, cc));
    }

    // Sort by closeness (descending) and take top k
    closeness.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    closeness.truncate(k);

    closeness
}

/// Compute closeness centrality for k random vertices
///
/// This is more efficient than computing closeness for all vertices when
/// k << n. It randomly samples k vertices and computes their closeness centrality.
///
/// Corresponds to sage.graphs.centrality.centrality_closeness_random_k
///
/// # Returns
/// Vector of (vertex, closeness_score) pairs for k random vertices
pub fn centrality_closeness_random_k(graph: &Graph, k: usize) -> Vec<(usize, f64)> {
    let n = graph.num_vertices();
    if n == 0 {
        return vec![];
    }

    // Sample k random vertices
    let mut vertices: Vec<usize> = (0..n).collect();
    let mut rng = thread_rng();
    vertices.shuffle(&mut rng);

    let sample_size = k.min(n);
    let sampled = &vertices[..sample_size];

    let mut closeness = Vec::new();

    for &v in sampled {
        let distances = compute_distances_from(graph, v);

        // Sum of distances to all reachable vertices
        let mut sum_dist = 0.0;
        let mut reachable = 0;

        for &d in &distances {
            if d >= 0 {
                sum_dist += d as f64;
                reachable += 1;
            }
        }

        // Closeness centrality
        let cc = if sum_dist > 0.0 && reachable > 1 {
            (reachable - 1) as f64 / sum_dist
        } else {
            0.0
        };

        closeness.push((v, cc));
    }

    closeness
}

/// Helper: Compute distances from a source vertex using BFS
fn compute_distances_from(graph: &Graph, start: usize) -> Vec<i32> {
    let n = graph.num_vertices();
    let mut distances = vec![-1; n];
    let mut queue = VecDeque::new();

    queue.push_back(start);
    distances[start] = 0;

    while let Some(v) = queue.pop_front() {
        if let Some(neighbors) = graph.neighbors(v) {
            for &u in &neighbors {
                if distances[u] < 0 {
                    distances[u] = distances[v] + 1;
                    queue.push_back(u);
                }
            }
        }
    }

    distances
}

/// Compute degree centrality for all vertices
///
/// Degree centrality is simply the degree of each vertex normalized by (n-1).
///
/// # Returns
/// HashMap mapping vertex index to degree centrality
pub fn centrality_degree(graph: &Graph) -> HashMap<usize, f64> {
    let n = graph.num_vertices();
    if n <= 1 {
        return HashMap::new();
    }

    let mut centrality = HashMap::new();
    let norm = (n - 1) as f64;

    for v in 0..n {
        if let Some(deg) = graph.degree(v) {
            centrality.insert(v, deg as f64 / norm);
        }
    }

    centrality
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_betweenness_centrality_path() {
        // Path graph: 0-1-2-3-4
        let mut g = Graph::new(5);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();
        g.add_edge(3, 4).unwrap();

        let bc = centrality_betweenness(&g);

        // Vertex 2 (middle) should have highest betweenness
        assert!(bc[&2] > bc[&0]);
        assert!(bc[&2] > bc[&1]);
        assert!(bc[&2] > bc[&3]);
        assert!(bc[&2] > bc[&4]);

        // Endpoints should have 0 betweenness
        assert_eq!(bc[&0], 0.0);
        assert_eq!(bc[&4], 0.0);
    }

    #[test]
    fn test_betweenness_centrality_star() {
        // Star graph: center connected to all others
        let mut g = Graph::new(5);
        for i in 1..5 {
            g.add_edge(0, i).unwrap();
        }

        let bc = centrality_betweenness(&g);

        // Center should have high betweenness
        assert!(bc[&0] > 0.0);

        // Leaves should have 0 betweenness
        for i in 1..5 {
            assert_eq!(bc[&i], 0.0);
        }
    }

    #[test]
    fn test_closeness_centrality_complete() {
        // Complete graph K4
        let mut g = Graph::new(4);
        for i in 0..4 {
            for j in i + 1..4 {
                g.add_edge(i, j).unwrap();
            }
        }

        let top_k = centrality_closeness_top_k(&g, 4);

        // All vertices should have same closeness in complete graph
        assert_eq!(top_k.len(), 4);
        for (_, score) in &top_k {
            assert!(score > &0.0);
        }

        // All scores should be equal
        let first_score = top_k[0].1;
        for (_, score) in &top_k {
            assert!((score - first_score).abs() < 1e-10);
        }
    }

    #[test]
    fn test_closeness_centrality_path() {
        // Path graph: 0-1-2-3-4
        let mut g = Graph::new(5);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();
        g.add_edge(3, 4).unwrap();

        let top_k = centrality_closeness_top_k(&g, 5);

        // Middle vertex should have highest closeness
        assert_eq!(top_k[0].0, 2);

        // Closeness should decrease towards endpoints
        assert!(top_k[0].1 > top_k[1].1);
    }

    #[test]
    fn test_closeness_centrality_random() {
        let mut g = Graph::new(10);
        for i in 0..9 {
            g.add_edge(i, i + 1).unwrap();
        }

        let random_k = centrality_closeness_random_k(&g, 5);

        assert_eq!(random_k.len(), 5);

        // All scores should be positive
        for (_, score) in &random_k {
            assert!(score > &0.0);
        }
    }

    #[test]
    fn test_degree_centrality() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 2).unwrap();
        g.add_edge(0, 3).unwrap();

        let dc = centrality_degree(&g);

        // Vertex 0 is connected to all others
        assert_eq!(dc[&0], 1.0); // degree 3 / (4-1) = 1.0

        // Other vertices have degree 1
        assert_eq!(dc[&1], 1.0 / 3.0);
        assert_eq!(dc[&2], 1.0 / 3.0);
        assert_eq!(dc[&3], 1.0 / 3.0);
    }

    #[test]
    fn test_betweenness_empty_graph() {
        let g = Graph::new(0);
        let bc = centrality_betweenness(&g);
        assert_eq!(bc.len(), 0);
    }

    #[test]
    fn test_closeness_empty_graph() {
        let g = Graph::new(0);
        let top_k = centrality_closeness_top_k(&g, 5);
        assert_eq!(top_k.len(), 0);
    }

    #[test]
    fn test_betweenness_disconnected() {
        // Two disconnected components
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(2, 3).unwrap();

        let bc = centrality_betweenness(&g);

        // All vertices should have 0 betweenness (no paths between components)
        for i in 0..4 {
            assert_eq!(bc[&i], 0.0);
        }
    }
}
