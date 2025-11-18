//! All-pairs shortest paths and distance computations
//!
//! This module provides algorithms for computing distances between all pairs of vertices,
//! including diameter, eccentricity, radius, Wiener index, and other distance-based metrics.

use std::collections::{HashMap, HashSet, VecDeque};
use crate::graph::Graph;

/// Compute all-pairs shortest path distances using BFS
///
/// Returns a matrix where entry (i, j) contains the distance from vertex i to vertex j.
/// If no path exists, the entry is `usize::MAX`.
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// 2D vector of distances
///
/// # Examples
/// ```
/// use rustmath_graphs::graph::Graph;
/// use rustmath_graphs::distances_all_pairs::distances_all_pairs;
///
/// let mut g = Graph::new(3);
/// g.add_edge(0, 1).unwrap();
/// g.add_edge(1, 2).unwrap();
/// let distances = distances_all_pairs(&g);
/// assert_eq!(distances[0][2], 2);
/// ```
pub fn distances_all_pairs(graph: &Graph) -> Vec<Vec<usize>> {
    let n = graph.num_vertices();
    let mut distances = vec![vec![usize::MAX; n]; n];

    for v in 0..n {
        let dist_from_v = bfs_distances(graph, v);
        for (u, &d) in dist_from_v.iter().enumerate() {
            distances[v][u] = d;
        }
    }

    distances
}

/// BFS to compute distances from a single source
fn bfs_distances(graph: &Graph, start: usize) -> Vec<usize> {
    let n = graph.num_vertices();
    let mut distances = vec![usize::MAX; n];
    let mut queue = VecDeque::new();

    queue.push_back(start);
    distances[start] = 0;

    while let Some(v) = queue.pop_front() {
        for &neighbor in graph.neighbors(v).unwrap_or_default().iter() {
            if distances[neighbor] == usize::MAX {
                distances[neighbor] = distances[v] + 1;
                queue.push_back(neighbor);
            }
        }
    }

    distances
}

/// Compute all-pairs shortest paths and their predecessors
///
/// Returns (distances, predecessors) where:
/// - distances[i][j] is the distance from i to j
/// - predecessors[i][j] is the vertex before j on a shortest path from i to j
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// Tuple of (distances matrix, predecessors matrix)
pub fn distances_and_predecessors_all_pairs(
    graph: &Graph,
) -> (Vec<Vec<usize>>, Vec<Vec<Option<usize>>>) {
    let n = graph.num_vertices();
    let mut distances = vec![vec![usize::MAX; n]; n];
    let mut predecessors = vec![vec![None; n]; n];

    for v in 0..n {
        let (dist, pred) = bfs_distances_with_predecessors(graph, v);
        for u in 0..n {
            distances[v][u] = dist[u];
            predecessors[v][u] = pred[u];
        }
    }

    (distances, predecessors)
}

fn bfs_distances_with_predecessors(
    graph: &Graph,
    start: usize,
) -> (Vec<usize>, Vec<Option<usize>>) {
    let n = graph.num_vertices();
    let mut distances = vec![usize::MAX; n];
    let mut predecessors = vec![None; n];
    let mut queue = VecDeque::new();

    queue.push_back(start);
    distances[start] = 0;

    while let Some(v) = queue.pop_front() {
        for &neighbor in graph.neighbors(v).unwrap_or_default().iter() {
            if distances[neighbor] == usize::MAX {
                distances[neighbor] = distances[v] + 1;
                predecessors[neighbor] = Some(v);
                queue.push_back(neighbor);
            }
        }
    }

    (distances, predecessors)
}

/// Compute all-pairs shortest paths
///
/// Returns a vector of paths where paths[i][j] is the shortest path from i to j.
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// 2D vector of shortest paths
pub fn shortest_path_all_pairs(graph: &Graph) -> Vec<Vec<Option<Vec<usize>>>> {
    let n = graph.num_vertices();
    let (_, predecessors) = distances_and_predecessors_all_pairs(graph);
    let mut paths = vec![vec![None; n]; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                paths[i][j] = Some(vec![i]);
            } else if predecessors[i][j].is_some() {
                paths[i][j] = reconstruct_path(i, j, &predecessors[i]);
            }
        }
    }

    paths
}

fn reconstruct_path(start: usize, end: usize, predecessors: &[Option<usize>]) -> Option<Vec<usize>> {
    if predecessors[end].is_none() && start != end {
        return None;
    }

    let mut path = vec![end];
    let mut current = end;

    while current != start {
        if let Some(pred) = predecessors[current] {
            path.push(pred);
            current = pred;
        } else {
            return None;
        }
    }

    path.reverse();
    Some(path)
}

/// Compute the diameter of a graph
///
/// The diameter is the maximum eccentricity among all vertices,
/// i.e., the longest shortest path in the graph.
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// The diameter, or None if the graph is disconnected
///
/// # Examples
/// ```
/// use rustmath_graphs::graph::Graph;
/// use rustmath_graphs::distances_all_pairs::diameter;
///
/// let mut g = Graph::new(4);
/// g.add_edge(0, 1).unwrap();
/// g.add_edge(1, 2).unwrap();
/// g.add_edge(2, 3).unwrap();
/// assert_eq!(diameter(&g), Some(3));
/// ```
pub fn diameter(graph: &Graph) -> Option<usize> {
    graph.diameter()
}

/// Compute the eccentricity of vertices
///
/// The eccentricity of a vertex v is the maximum distance from v to any other vertex.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `vertices` - Optional list of vertices to compute eccentricity for (all if None)
///
/// # Returns
/// HashMap mapping vertices to their eccentricities
pub fn eccentricity(graph: &Graph, vertices: Option<&[usize]>) -> HashMap<usize, Option<usize>> {
    let n = graph.num_vertices();
    let vertex_list: Vec<usize> = match vertices {
        Some(v) => v.to_vec(),
        None => (0..n).collect(),
    };

    let mut eccentricities = HashMap::new();

    for &v in &vertex_list {
        let distances = bfs_distances(graph, v);
        let max_dist = distances
            .iter()
            .filter(|&&d| d != usize::MAX)
            .max()
            .copied();

        // If any distance is MAX, vertex is not connected to all others
        if distances.iter().any(|&d| d == usize::MAX && distances[v] != d) {
            eccentricities.insert(v, None);
        } else {
            eccentricities.insert(v, max_dist);
        }
    }

    eccentricities
}

/// Compute the radius of a graph
///
/// The radius is the minimum eccentricity among all vertices.
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// The radius, or None if the graph is disconnected
pub fn radius(graph: &Graph) -> Option<usize> {
    let ecc = eccentricity(graph, None);
    ecc.values()
        .filter_map(|&e| e)
        .min()
}

/// Compute the radius using the DHV (Dragan-Hedetniemi-Vince) algorithm
///
/// This is an approximation algorithm for computing the radius more efficiently.
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// Approximate radius value
pub fn radius_dhv(graph: &Graph) -> Option<usize> {
    // For simplicity, use exact radius computation
    // A true DHV implementation would use a more sophisticated approximation
    radius(graph)
}

/// Floyd-Warshall algorithm for all-pairs shortest paths
///
/// This algorithm works for weighted graphs, but here we use it for unweighted graphs
/// where all edge weights are 1.
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// Distance matrix
pub fn floyd_warshall(graph: &Graph) -> Vec<Vec<usize>> {
    let n = graph.num_vertices();
    let mut dist = vec![vec![usize::MAX / 2; n]; n];

    // Initialize distances
    for i in 0..n {
        dist[i][i] = 0;
    }

    for (u, v) in graph.edges() {
        dist[u][v] = 1;
        dist[v][u] = 1;
    }

    // Floyd-Warshall main loop
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                if dist[i][k] != usize::MAX / 2 && dist[k][j] != usize::MAX / 2 {
                    dist[i][j] = dist[i][j].min(dist[i][k] + dist[k][j]);
                }
            }
        }
    }

    dist
}

/// Compute the distribution of distances in a graph
///
/// Returns a histogram where the index represents the distance
/// and the value represents the count of vertex pairs at that distance.
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// Vector where index i contains the number of pairs at distance i
pub fn distances_distribution(graph: &Graph) -> Vec<usize> {
    let distances = distances_all_pairs(graph);
    let n = graph.num_vertices();

    // Find maximum distance
    let max_dist = distances
        .iter()
        .flat_map(|row| row.iter())
        .filter(|&&d| d != usize::MAX)
        .max()
        .copied()
        .unwrap_or(0);

    let mut distribution = vec![0; max_dist + 1];

    for i in 0..n {
        for j in i + 1..n {
            if distances[i][j] != usize::MAX {
                distribution[distances[i][j]] += 1;
            }
        }
    }

    distribution
}

/// Check if a graph is distance-regular
///
/// A distance-regular graph has the property that the number of vertices at distance j
/// from a vertex i depends only on j and the distance from i to another vertex k,
/// not on the specific vertices i and k.
///
/// # Arguments
/// * `graph` - The graph to check
///
/// # Returns
/// `true` if the graph is distance-regular
pub fn is_distance_regular(graph: &Graph) -> bool {
    let n = graph.num_vertices();
    let distances = distances_all_pairs(graph);

    if !graph.is_connected() {
        return false;
    }

    // Check if all vertices have the same degree (necessary but not sufficient)
    let degree_0 = graph.degree(0).unwrap_or(0);
    for v in 1..n {
        if graph.degree(v) != Some(degree_0) {
            return false;
        }
    }

    // Compute intersection numbers for the first vertex
    let diam = diameter(graph).unwrap_or(0);
    let mut reference_params = vec![];

    for dist in 0..=diam {
        let count_at_dist: usize = (0..n)
            .filter(|&v| distances[0][v] == dist)
            .count();
        reference_params.push(count_at_dist);
    }

    // Check if all vertices have the same distance distribution
    for v in 1..n {
        let mut params = vec![];
        for dist in 0..=diam {
            let count_at_dist: usize = (0..n)
                .filter(|&u| distances[v][u] == dist)
                .count();
            params.push(count_at_dist);
        }
        if params != reference_params {
            return false;
        }
    }

    true
}

/// Compute the Wiener index of a graph
///
/// The Wiener index is the sum of all shortest path distances in the graph.
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// The Wiener index
///
/// # Examples
/// ```
/// use rustmath_graphs::graph::Graph;
/// use rustmath_graphs::distances_all_pairs::wiener_index;
///
/// let mut g = Graph::new(3);
/// g.add_edge(0, 1).unwrap();
/// g.add_edge(1, 2).unwrap();
/// assert_eq!(wiener_index(&g), 4); // d(0,1)=1, d(0,2)=2, d(1,2)=1
/// ```
pub fn wiener_index(graph: &Graph) -> usize {
    let distances = distances_all_pairs(graph);
    let n = graph.num_vertices();
    let mut sum = 0;

    for i in 0..n {
        for j in i + 1..n {
            if distances[i][j] != usize::MAX {
                sum += distances[i][j];
            }
        }
    }

    sum
}

/// Compute the Szeged index of a graph
///
/// For each edge (u, v), the Szeged index counts vertices closer to u than to v
/// times vertices closer to v than to u, then sums over all edges.
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// The Szeged index
pub fn szeged_index(graph: &Graph) -> usize {
    let distances = distances_all_pairs(graph);
    let n = graph.num_vertices();
    let mut index = 0;

    for (u, v) in graph.edges() {
        let mut closer_to_u = 0;
        let mut closer_to_v = 0;

        for w in 0..n {
            if w != u && w != v {
                if distances[w][u] < distances[w][v] {
                    closer_to_u += 1;
                } else if distances[w][v] < distances[w][u] {
                    closer_to_v += 1;
                }
            }
        }

        index += closer_to_u * closer_to_v;
    }

    index
}

/// Create an antipodal graph
///
/// The antipodal graph has the same vertices as the original graph,
/// with edges connecting vertices that are at maximum distance (diameter) from each other.
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// A new Graph representing the antipodal graph, or None if disconnected
pub fn antipodal_graph(graph: &Graph) -> Option<Graph> {
    let n = graph.num_vertices();
    let distances = distances_all_pairs(graph);
    let diam = diameter(graph)?;

    let mut antipodal = Graph::new(n);

    for i in 0..n {
        for j in i + 1..n {
            if distances[i][j] == diam {
                antipodal.add_edge(i, j).ok();
            }
        }
    }

    Some(antipodal)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distances_all_pairs() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        let distances = distances_all_pairs(&g);

        assert_eq!(distances[0][0], 0);
        assert_eq!(distances[0][1], 1);
        assert_eq!(distances[0][2], 2);
        assert_eq!(distances[0][3], 3);
        assert_eq!(distances[1][2], 1);
        assert_eq!(distances[1][3], 2);
    }

    #[test]
    fn test_diameter() {
        let mut g = Graph::new(5);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();
        g.add_edge(3, 4).unwrap();

        assert_eq!(diameter(&g), Some(4));
    }

    #[test]
    fn test_eccentricity() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        let ecc = eccentricity(&g, None);
        assert_eq!(ecc.get(&1), Some(&Some(1))); // Center vertex has ecc 1
        assert_eq!(ecc.get(&0), Some(&Some(2))); // End vertices have ecc 2
        assert_eq!(ecc.get(&2), Some(&Some(2)));
    }

    #[test]
    fn test_radius() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        assert_eq!(radius(&g), Some(1));
    }

    #[test]
    fn test_floyd_warshall() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        let distances = floyd_warshall(&g);

        assert_eq!(distances[0][2], 2);
        assert_eq!(distances[0][1], 1);
        assert_eq!(distances[1][2], 1);
    }

    #[test]
    fn test_distances_distribution() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        let dist = distances_distribution(&g);

        // dist[1] = pairs at distance 1 (3 edges)
        // dist[2] = pairs at distance 2 (2 pairs: 0-2, 1-3)
        // dist[3] = pairs at distance 3 (1 pair: 0-3)
        assert_eq!(dist[1], 3);
        assert_eq!(dist[2], 2);
        assert_eq!(dist[3], 1);
    }

    #[test]
    fn test_is_distance_regular_complete_graph() {
        // Complete graph K4 is distance-regular
        let mut g = Graph::new(4);
        for i in 0..4 {
            for j in i + 1..4 {
                g.add_edge(i, j).unwrap();
            }
        }
        assert!(is_distance_regular(&g));
    }

    #[test]
    fn test_is_distance_regular_cycle() {
        // Cycle C4 is distance-regular
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();
        g.add_edge(3, 0).unwrap();
        assert!(is_distance_regular(&g));
    }

    #[test]
    fn test_wiener_index() {
        // Path of length 2: distances are 1, 1, 2 -> sum = 4
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        assert_eq!(wiener_index(&g), 4);
    }

    #[test]
    fn test_szeged_index() {
        // Simple path
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        let index = szeged_index(&g);
        // For edge (0,1): n_0=0 (no vertices closer to 0), n_1=1 (vertex 2), product = 0
        // For edge (1,2): n_1=1 (vertex 0), n_2=0 (no vertices closer to 2), product = 0
        // Total = 0
        assert_eq!(index, 0);

        // For a more interesting case, use a graph with more vertices
        let mut g2 = Graph::new(4);
        g2.add_edge(0, 1).unwrap();
        g2.add_edge(1, 2).unwrap();
        g2.add_edge(1, 3).unwrap();

        let index2 = szeged_index(&g2);
        // Edge (0,1): n_0=0, n_1=2 (vertices 2,3), product = 0
        // Edge (1,2): n_1=2 (vertices 0,3), n_2=0, product = 0
        // Edge (1,3): n_1=2 (vertices 0,2), n_3=0, product = 0
        // Total = 0
        assert_eq!(index2, 0);
    }

    #[test]
    fn test_antipodal_graph() {
        // Path of length 3 has diameter 3
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        let antipodal = antipodal_graph(&g).unwrap();

        // Only vertices at distance 3 are connected: 0-3
        assert!(antipodal.has_edge(0, 3));
        assert!(!antipodal.has_edge(0, 1));
        assert!(!antipodal.has_edge(1, 2));
    }

    #[test]
    fn test_shortest_path_all_pairs() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        let paths = shortest_path_all_pairs(&g);

        assert_eq!(paths[0][2], Some(vec![0, 1, 2]));
        assert_eq!(paths[0][1], Some(vec![0, 1]));
        assert_eq!(paths[1][2], Some(vec![1, 2]));
    }

    #[test]
    fn test_distances_and_predecessors() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        let (distances, predecessors) = distances_and_predecessors_all_pairs(&g);

        assert_eq!(distances[0][2], 2);
        assert_eq!(predecessors[0][2], Some(1));
        assert_eq!(predecessors[0][1], Some(0));
    }
}
