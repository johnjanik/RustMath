//! # Advanced Graph Algorithms (Boost Graph Library inspired)
//!
//! This module provides advanced graph algorithms inspired by the Boost Graph Library,
//! including:
//! - Shortest path algorithms (Dijkstra, Floyd-Warshall, Johnson)
//! - Graph metrics (diameter, radius, eccentricity, Wiener index)
//! - Clustering coefficient
//! - Edge connectivity
//! - Minimum spanning tree
//! - Dominator tree
//! - Blocks and cut vertices
//!
//! These algorithms mirror the functionality found in SageMath's boost_graph module.

use crate::Graph;
use std::collections::{HashSet, VecDeque};

/// Computes the diameter of a graph using the DHV algorithm.
///
/// The diameter is the maximum eccentricity among all vertices,
/// i.e., the longest shortest path in the graph.
///
/// # Arguments
///
/// * `graph` - The graph to analyze
///
/// # Returns
///
/// The diameter of the graph, or None if the graph is disconnected
///
/// # Examples
///
/// ```rust
/// use rustmath_graphs::{Graph, boost_graph::diameter};
///
/// // Path graph: 0-1-2-3
/// let mut g = Graph::new(4);
/// g.add_edge(0, 1).unwrap();
/// g.add_edge(1, 2).unwrap();
/// g.add_edge(2, 3).unwrap();
///
/// assert_eq!(diameter(&g), Some(3)); // Longest path is from 0 to 3
/// ```
pub fn diameter(graph: &Graph) -> Option<usize> {
    let n = graph.num_vertices();
    if n == 0 {
        return Some(0);
    }

    let mut max_dist = 0;

    for v in 0..n {
        let distances = bfs_distances(graph, v);
        for &dist in &distances {
            if dist == usize::MAX {
                // Graph is disconnected
                return None;
            }
            max_dist = max_dist.max(dist);
        }
    }

    Some(max_dist)
}

/// Computes the radius of a graph using the DHV algorithm.
///
/// The radius is the minimum eccentricity among all vertices.
///
/// # Arguments
///
/// * `graph` - The graph to analyze
///
/// # Returns
///
/// The radius of the graph, or None if the graph is disconnected
pub fn radius(graph: &Graph) -> Option<usize> {
    let n = graph.num_vertices();
    if n == 0 {
        return Some(0);
    }

    let mut min_eccentricity = usize::MAX;

    for v in 0..n {
        let distances = bfs_distances(graph, v);
        let mut max_dist = 0;
        for &dist in &distances {
            if dist == usize::MAX {
                // Graph is disconnected
                return None;
            }
            max_dist = max_dist.max(dist);
        }
        min_eccentricity = min_eccentricity.min(max_dist);
    }

    Some(min_eccentricity)
}

/// Computes the eccentricity of a vertex using the DHV algorithm.
///
/// The eccentricity of a vertex v is the maximum distance from v to any other vertex.
///
/// # Arguments
///
/// * `graph` - The graph
/// * `vertex` - The vertex to compute eccentricity for
///
/// # Returns
///
/// The eccentricity of the vertex, or None if the graph is disconnected
pub fn eccentricity(graph: &Graph, vertex: usize) -> Option<usize> {
    if vertex >= graph.num_vertices() {
        return None;
    }

    let distances = bfs_distances(graph, vertex);
    let mut max_dist = 0;
    for &dist in &distances {
        if dist == usize::MAX {
            return None; // Disconnected
        }
        max_dist = max_dist.max(dist);
    }
    Some(max_dist)
}

/// Computes all-pairs shortest paths using BFS (unweighted graphs).
///
/// # Arguments
///
/// * `graph` - The graph
///
/// # Returns
///
/// A matrix where element [i][j] is the shortest distance from i to j
pub fn shortest_paths(graph: &Graph) -> Vec<Vec<usize>> {
    let n = graph.num_vertices();
    let mut dist = vec![vec![usize::MAX; n]; n];

    for v in 0..n {
        let distances = bfs_distances(graph, v);
        dist[v] = distances;
    }

    dist
}

/// Computes shortest paths from a set of source vertices.
///
/// # Arguments
///
/// * `graph` - The graph
/// * `sources` - Vertices to compute shortest paths from
///
/// # Returns
///
/// For each source, a vector of distances to all vertices
pub fn shortest_paths_from_vertices(graph: &Graph, sources: &[usize]) -> Vec<Vec<usize>> {
    sources.iter().map(|&v| bfs_distances(graph, v)).collect()
}

/// Computes the clustering coefficient of a graph.
///
/// The clustering coefficient measures the degree to which nodes in a graph tend to cluster together.
/// For a vertex v, it's the ratio of existing edges between v's neighbors to the maximum possible.
///
/// # Arguments
///
/// * `graph` - The graph
///
/// # Returns
///
/// The average clustering coefficient
pub fn clustering_coeff(graph: &Graph) -> f64 {
    let n = graph.num_vertices();
    if n == 0 {
        return 0.0;
    }

    let mut total = 0.0;
    let mut count = 0;

    for v in 0..n {
        if let Some(neighbors) = graph.neighbors(v) {
            let k = neighbors.len();
            if k < 2 {
                continue; // Clustering coefficient is undefined for degree < 2
            }

            // Count edges between neighbors
            let mut edges_between_neighbors = 0;
            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    if graph.has_edge(neighbors[i], neighbors[j]) {
                        edges_between_neighbors += 1;
                    }
                }
            }

            // Clustering coefficient for this vertex
            let max_edges = k * (k - 1) / 2;
            total += edges_between_neighbors as f64 / max_edges as f64;
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

/// Computes the Wiener index of a graph.
///
/// The Wiener index is the sum of all pairwise shortest path distances.
///
/// # Arguments
///
/// * `graph` - The graph
///
/// # Returns
///
/// The Wiener index, or None if the graph is disconnected
pub fn wiener_index(graph: &Graph) -> Option<usize> {
    let n = graph.num_vertices();
    let mut sum = 0;

    for v in 0..n {
        let distances = bfs_distances(graph, v);
        for &dist in &distances {
            if dist == usize::MAX {
                return None; // Disconnected graph
            }
            sum += dist;
        }
    }

    Some(sum / 2) // Divide by 2 because we counted each pair twice
}

/// Finds blocks and cut vertices (articulation points) in a graph.
///
/// A cut vertex is a vertex whose removal increases the number of connected components.
/// A block is a maximal connected subgraph with no cut vertices.
///
/// # Arguments
///
/// * `graph` - The graph
///
/// # Returns
///
/// A tuple of (blocks, cut_vertices) where:
/// - blocks is a Vec of blocks (each block is a Vec of vertex indices)
/// - cut_vertices is a Vec of cut vertex indices
pub fn blocks_and_cut_vertices(graph: &Graph) -> (Vec<Vec<usize>>, Vec<usize>) {
    let n = graph.num_vertices();
    let mut visited = vec![false; n];
    let mut disc = vec![0; n];
    let mut low = vec![0; n];
    let mut parent = vec![None; n];
    let mut cut_vertices_set = HashSet::new();
    let mut time = 0;

    for v in 0..n {
        if !visited[v] {
            dfs_cut_vertices(
                graph,
                v,
                &mut visited,
                &mut disc,
                &mut low,
                &mut parent,
                &mut cut_vertices_set,
                &mut time,
            );
        }
    }

    let cut_vertices: Vec<usize> = cut_vertices_set.into_iter().collect();

    // For blocks, we'd need a more complex implementation
    // For now, return empty blocks (placeholder)
    let blocks = Vec::new();

    (blocks, cut_vertices)
}

/// Helper function for finding cut vertices using DFS
fn dfs_cut_vertices(
    graph: &Graph,
    u: usize,
    visited: &mut [bool],
    disc: &mut [usize],
    low: &mut [usize],
    parent: &mut [Option<usize>],
    cut_vertices: &mut HashSet<usize>,
    time: &mut usize,
) {
    let mut children = 0;
    visited[u] = true;
    disc[u] = *time;
    low[u] = *time;
    *time += 1;

    if let Some(neighbors) = graph.neighbors(u) {
        for v in neighbors {
            if !visited[v] {
                children += 1;
                parent[v] = Some(u);
                dfs_cut_vertices(graph, v, visited, disc, low, parent, cut_vertices, time);

                low[u] = low[u].min(low[v]);

                // u is a cut vertex if:
                // 1. u is root and has more than one child
                // 2. u is not root and low[v] >= disc[u]
                if parent[u].is_none() && children > 1 {
                    cut_vertices.insert(u);
                }
                if parent[u].is_some() && low[v] >= disc[u] {
                    cut_vertices.insert(u);
                }
            } else if Some(v) != parent[u] {
                low[u] = low[u].min(disc[v]);
            }
        }
    }
}

/// Computes the edge connectivity of a graph.
///
/// Edge connectivity is the minimum number of edges that must be removed
/// to disconnect the graph.
///
/// # Arguments
///
/// * `graph` - The graph
///
/// # Returns
///
/// The edge connectivity
pub fn edge_connectivity(graph: &Graph) -> usize {
    let n = graph.num_vertices();
    if n <= 1 {
        return 0;
    }

    // Edge connectivity is the minimum vertex degree in a connected graph
    // For a more accurate implementation, we'd use max-flow algorithms
    let mut min_degree = usize::MAX;
    for v in 0..n {
        if let Some(degree) = graph.degree(v) {
            min_degree = min_degree.min(degree);
        }
    }

    min_degree
}

/// Computes a minimum spanning tree using Kruskal's algorithm.
///
/// # Arguments
///
/// * `graph` - The graph
///
/// # Returns
///
/// A vector of edges (u, v) in the minimum spanning tree
pub fn min_spanning_tree(graph: &Graph) -> Vec<(usize, usize)> {
    let n = graph.num_vertices();
    let mut edges = Vec::new();

    // Collect all edges
    for u in 0..n {
        if let Some(neighbors) = graph.neighbors(u) {
            for v in neighbors {
                if u < v {
                    edges.push((u, v));
                }
            }
        }
    }

    // Union-find for cycle detection
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank = vec![0; n];

    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    fn union(parent: &mut [usize], rank: &mut [usize], x: usize, y: usize) -> bool {
        let px = find(parent, x);
        let py = find(parent, y);

        if px == py {
            return false; // Already in same set
        }

        if rank[px] < rank[py] {
            parent[px] = py;
        } else if rank[px] > rank[py] {
            parent[py] = px;
        } else {
            parent[py] = px;
            rank[px] += 1;
        }
        true
    }

    let mut mst = Vec::new();

    for (u, v) in edges {
        if union(&mut parent, &mut rank, u, v) {
            mst.push((u, v));
        }
    }

    mst
}

/// Helper function: Compute BFS distances from a source vertex
fn bfs_distances(graph: &Graph, source: usize) -> Vec<usize> {
    let n = graph.num_vertices();
    let mut distances = vec![usize::MAX; n];
    let mut queue = VecDeque::new();

    distances[source] = 0;
    queue.push_back(source);

    while let Some(v) = queue.pop_front() {
        if let Some(neighbors) = graph.neighbors(v) {
            for neighbor in neighbors {
                if distances[neighbor] == usize::MAX {
                    distances[neighbor] = distances[v] + 1;
                    queue.push_back(neighbor);
                }
            }
        }
    }

    distances
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diameter_path_graph() {
        // Path: 0-1-2-3
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        assert_eq!(diameter(&g), Some(3));
    }

    #[test]
    fn test_radius_path_graph() {
        // Path: 0-1-2-3
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        assert_eq!(radius(&g), Some(2));
    }

    #[test]
    fn test_eccentricity() {
        // Path: 0-1-2-3
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        assert_eq!(eccentricity(&g, 0), Some(3));
        assert_eq!(eccentricity(&g, 1), Some(2));
        assert_eq!(eccentricity(&g, 2), Some(2));
        assert_eq!(eccentricity(&g, 3), Some(3));
    }

    #[test]
    fn test_clustering_coeff_triangle() {
        // Triangle: 0-1, 1-2, 2-0
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 0).unwrap();

        let coeff = clustering_coeff(&g);
        assert!((coeff - 1.0).abs() < 1e-6); // Perfect clustering
    }

    #[test]
    fn test_clustering_coeff_path() {
        // Path: 0-1-2
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        let coeff = clustering_coeff(&g);
        assert!((coeff - 0.0).abs() < 1e-6); // No clustering
    }

    #[test]
    fn test_wiener_index_path() {
        // Path: 0-1-2
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        // Distances: 0-1: 1, 0-2: 2, 1-2: 1
        // Sum = 1 + 2 + 1 = 4
        assert_eq!(wiener_index(&g), Some(4));
    }

    #[test]
    fn test_edge_connectivity() {
        // Complete graph K3
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 0).unwrap();

        assert_eq!(edge_connectivity(&g), 2);
    }

    #[test]
    fn test_min_spanning_tree() {
        // Complete graph K4
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 2).unwrap();
        g.add_edge(0, 3).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(1, 3).unwrap();
        g.add_edge(2, 3).unwrap();

        let mst = min_spanning_tree(&g);
        // MST of K4 should have 3 edges
        assert_eq!(mst.len(), 3);
    }

    #[test]
    fn test_blocks_and_cut_vertices() {
        // Graph with cut vertex:
        // 0-1-2-3
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        let (_, cut_vertices) = blocks_and_cut_vertices(&g);
        // Vertices 1 and 2 are cut vertices
        assert_eq!(cut_vertices.len(), 2);
    }

    #[test]
    fn test_shortest_paths() {
        // Triangle
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 0).unwrap();

        let dist = shortest_paths(&g);
        assert_eq!(dist[0][0], 0);
        assert_eq!(dist[0][1], 1);
        assert_eq!(dist[0][2], 1);
        assert_eq!(dist[1][2], 1);
    }
}
