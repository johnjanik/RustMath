//! Graph convexity properties
//!
//! This module provides functions for analyzing convexity properties of graphs,
//! particularly geodetic convexity where a set S is convex if it contains all
//! shortest paths between any two vertices in S.

use std::collections::{HashMap, HashSet, VecDeque};
use crate::graph::Graph;

/// Convexity properties analyzer for graphs
///
/// Provides methods to analyze geodetic convexity and related properties.
#[derive(Debug, Clone)]
pub struct ConvexityProperties {
    graph: Graph,
    // Cache for all-pairs shortest paths
    distances: Option<HashMap<(usize, usize), usize>>,
}

impl ConvexityProperties {
    /// Create a new convexity properties analyzer for a graph
    ///
    /// # Arguments
    /// * `graph` - The graph to analyze
    ///
    /// # Returns
    /// A new ConvexityProperties instance
    pub fn new(graph: Graph) -> Self {
        ConvexityProperties {
            graph,
            distances: None,
        }
    }

    /// Compute all-pairs shortest path distances
    fn compute_distances(&mut self) {
        if self.distances.is_some() {
            return;
        }

        let n = self.graph.num_vertices();
        let mut dist_map = HashMap::new();

        for u in 0..n {
            let distances = self.bfs_distances(u);
            for (v, d) in distances.iter().enumerate() {
                if *d < usize::MAX {
                    dist_map.insert((u, v), *d);
                }
            }
        }

        self.distances = Some(dist_map);
    }

    fn bfs_distances(&self, start: usize) -> Vec<usize> {
        let n = self.graph.num_vertices();
        let mut distances = vec![usize::MAX; n];
        let mut queue = VecDeque::new();

        queue.push_back(start);
        distances[start] = 0;

        while let Some(v) = queue.pop_front() {
            for &neighbor in self.graph.neighbors(v).unwrap_or_default().iter() {
                if distances[neighbor] == usize::MAX {
                    distances[neighbor] = distances[v] + 1;
                    queue.push_back(neighbor);
                }
            }
        }

        distances
    }

    /// Get the distance between two vertices
    ///
    /// # Arguments
    /// * `u` - First vertex
    /// * `v` - Second vertex
    ///
    /// # Returns
    /// The shortest path distance, or None if no path exists
    pub fn distance(&mut self, u: usize, v: usize) -> Option<usize> {
        if u >= self.graph.num_vertices() || v >= self.graph.num_vertices() {
            return None;
        }

        if u == v {
            return Some(0);
        }

        self.compute_distances();
        self.distances.as_ref()?.get(&(u, v)).copied()
    }

    /// Check if a vertex is on a shortest path between two other vertices
    ///
    /// # Arguments
    /// * `u` - Start vertex
    /// * `v` - End vertex
    /// * `w` - Vertex to check
    ///
    /// # Returns
    /// `true` if w is on some shortest path from u to v
    pub fn is_on_shortest_path(&mut self, u: usize, v: usize, w: usize) -> bool {
        if let (Some(d_uv), Some(d_uw), Some(d_wv)) = (
            self.distance(u, v),
            self.distance(u, w),
            self.distance(w, v),
        ) {
            d_uw + d_wv == d_uv
        } else {
            false
        }
    }

    /// Compute the geodetic interval between two vertices
    ///
    /// The geodetic interval I[u,v] is the set of all vertices that lie on
    /// some shortest path between u and v.
    ///
    /// # Arguments
    /// * `u` - First vertex
    /// * `v` - Second vertex
    ///
    /// # Returns
    /// Set of vertices in the geodetic interval
    pub fn geodetic_interval(&mut self, u: usize, v: usize) -> HashSet<usize> {
        let n = self.graph.num_vertices();
        let mut interval = HashSet::new();

        for w in 0..n {
            if self.is_on_shortest_path(u, v, w) {
                interval.insert(w);
            }
        }

        interval
    }

    /// Compute the geodetic closure of a set of vertices
    ///
    /// The geodetic closure is the smallest geodetically convex set containing
    /// the given vertices. A set is geodetically convex if it contains all
    /// shortest paths between any two of its vertices.
    ///
    /// # Arguments
    /// * `vertices` - Initial set of vertices
    ///
    /// # Returns
    /// The geodetic closure as a set of vertices
    pub fn geodetic_closure(&mut self, vertices: &[usize]) -> HashSet<usize> {
        let mut closure: HashSet<usize> = vertices.iter().copied().collect();
        let mut changed = true;

        while changed {
            changed = false;
            let current_closure: Vec<usize> = closure.iter().copied().collect();

            for &u in &current_closure {
                for &v in &current_closure {
                    if u != v {
                        let interval = self.geodetic_interval(u, v);
                        let old_size = closure.len();
                        closure.extend(interval);
                        if closure.len() > old_size {
                            changed = true;
                        }
                    }
                }
            }
        }

        closure
    }

    /// Check if a set of vertices is geodetically convex
    ///
    /// # Arguments
    /// * `vertices` - Set of vertices to check
    ///
    /// # Returns
    /// `true` if the set is geodetically convex
    pub fn is_convex(&mut self, vertices: &[usize]) -> bool {
        let vertex_set: HashSet<usize> = vertices.iter().copied().collect();

        for &u in vertices {
            for &v in vertices {
                if u != v {
                    let interval = self.geodetic_interval(u, v);
                    // Check if all vertices in the interval are in the set
                    if !interval.is_subset(&vertex_set) {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Check if the entire graph is geodetic
    ///
    /// A graph is geodetic if there is exactly one shortest path between
    /// every pair of vertices.
    ///
    /// # Returns
    /// `true` if the graph is geodetic
    pub fn is_geodetic(&mut self) -> bool {
        let n = self.graph.num_vertices();

        for u in 0..n {
            for v in u + 1..n {
                if self.count_shortest_paths(u, v) != 1 {
                    return false;
                }
            }
        }

        true
    }

    /// Count the number of shortest paths between two vertices
    ///
    /// # Arguments
    /// * `u` - Start vertex
    /// * `v` - End vertex
    ///
    /// # Returns
    /// Number of shortest paths from u to v
    fn count_shortest_paths(&mut self, u: usize, v: usize) -> usize {
        if u == v {
            return 1;
        }

        let Some(target_dist) = self.distance(u, v) else {
            return 0;
        };

        let mut count = 0;
        let mut paths = Vec::new();
        let mut current_path = vec![u];
        let mut visited = vec![false; self.graph.num_vertices()];
        visited[u] = true;

        self.find_all_shortest_paths(
            u,
            v,
            target_dist,
            &mut current_path,
            &mut visited,
            &mut paths,
        );

        paths.len()
    }

    fn find_all_shortest_paths(
        &mut self,
        current: usize,
        target: usize,
        remaining_dist: usize,
        current_path: &mut Vec<usize>,
        visited: &mut [bool],
        paths: &mut Vec<Vec<usize>>,
    ) {
        if current == target && remaining_dist == 0 {
            paths.push(current_path.clone());
            return;
        }

        if remaining_dist == 0 {
            return;
        }

        for &neighbor in self.graph.neighbors(current).unwrap_or_default().iter() {
            if !visited[neighbor] {
                if let Some(dist_to_target) = self.distance(neighbor, target) {
                    if dist_to_target == remaining_dist - 1 {
                        current_path.push(neighbor);
                        visited[neighbor] = true;

                        self.find_all_shortest_paths(
                            neighbor,
                            target,
                            remaining_dist - 1,
                            current_path,
                            visited,
                            paths,
                        );

                        current_path.pop();
                        visited[neighbor] = false;
                    }
                }
            }
        }
    }

    /// Compute the geodetic number of the graph
    ///
    /// The geodetic number is the minimum size of a set whose geodetic closure
    /// is the entire vertex set.
    ///
    /// # Returns
    /// The geodetic number
    pub fn geodetic_number(&mut self) -> usize {
        let n = self.graph.num_vertices();

        // Try all subset sizes starting from 1
        for size in 1..=n {
            if self.has_geodetic_set_of_size(size) {
                return size;
            }
        }

        n
    }

    fn has_geodetic_set_of_size(&mut self, size: usize) -> bool {
        let n = self.graph.num_vertices();

        if size > n {
            return false;
        }

        // Try all combinations of 'size' vertices
        let mut indices = vec![0; size];
        for i in 0..size {
            indices[i] = i;
        }

        loop {
            let closure = self.geodetic_closure(&indices);
            if closure.len() == n {
                return true;
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
                    return false;
                }
            }
        }
    }

    /// Compute the hull number of the graph
    ///
    /// The hull number is the minimum size of a set whose geodetic closure
    /// equals itself (i.e., a minimal geodetically convex set that generates itself).
    ///
    /// # Returns
    /// The hull number
    pub fn hull_number(&mut self) -> usize {
        let n = self.graph.num_vertices();
        let mut min_hull = n;

        // Try all subsets
        for size in 1..=n {
            if self.find_hull_of_size(size, &mut min_hull) {
                return min_hull;
            }
        }

        min_hull
    }

    fn find_hull_of_size(&mut self, size: usize, min_hull: &mut usize) -> bool {
        let n = self.graph.num_vertices();

        if size > n {
            return false;
        }

        let mut indices = vec![0; size];
        for i in 0..size {
            indices[i] = i;
        }

        loop {
            let closure = self.geodetic_closure(&indices);
            if closure.len() == size {
                *min_hull = size;
                return true;
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
                    return false;
                }
            }
        }
    }
}

/// Compute the geodetic closure of a set of vertices in a graph
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `vertices` - Initial set of vertices
///
/// # Returns
/// The geodetic closure as a set of vertices
pub fn geodetic_closure(graph: &Graph, vertices: &[usize]) -> HashSet<usize> {
    let mut props = ConvexityProperties::new(graph.clone());
    props.geodetic_closure(vertices)
}

/// Check if a graph is geodetic
///
/// A graph is geodetic if there is exactly one shortest path between
/// every pair of vertices.
///
/// # Arguments
/// * `graph` - The graph to check
///
/// # Returns
/// `true` if the graph is geodetic
pub fn is_geodetic(graph: &Graph) -> bool {
    let mut props = ConvexityProperties::new(graph.clone());
    props.is_geodetic()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convexity_properties_creation() {
        let g = Graph::new(5);
        let props = ConvexityProperties::new(g);
        assert_eq!(props.graph.num_vertices(), 5);
    }

    #[test]
    fn test_distance() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        let mut props = ConvexityProperties::new(g);

        assert_eq!(props.distance(0, 0), Some(0));
        assert_eq!(props.distance(0, 1), Some(1));
        assert_eq!(props.distance(0, 2), Some(2));
        assert_eq!(props.distance(0, 3), Some(3));
    }

    #[test]
    fn test_is_on_shortest_path() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        let mut props = ConvexityProperties::new(g);

        // Vertex 1 is on shortest path from 0 to 2
        assert!(props.is_on_shortest_path(0, 2, 1));
        // Vertex 1 is on shortest path from 0 to 3
        assert!(props.is_on_shortest_path(0, 3, 1));
        // Vertex 0 is not on shortest path from 1 to 3
        assert!(!props.is_on_shortest_path(1, 3, 0));
    }

    #[test]
    fn test_geodetic_interval() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        let mut props = ConvexityProperties::new(g);

        let interval = props.geodetic_interval(0, 3);
        assert_eq!(interval.len(), 4); // All vertices are on path 0-1-2-3
        assert!(interval.contains(&0));
        assert!(interval.contains(&1));
        assert!(interval.contains(&2));
        assert!(interval.contains(&3));
    }

    #[test]
    fn test_geodetic_closure_simple() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        let mut props = ConvexityProperties::new(g);

        // Closure of {0, 3} should be the entire path
        let closure = props.geodetic_closure(&[0, 3]);
        assert_eq!(closure.len(), 4);

        // Closure of {0, 2} should be {0, 1, 2}
        let closure2 = props.geodetic_closure(&[0, 2]);
        assert_eq!(closure2.len(), 3);
        assert!(closure2.contains(&0));
        assert!(closure2.contains(&1));
        assert!(closure2.contains(&2));
    }

    #[test]
    fn test_is_convex() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        let mut props = ConvexityProperties::new(g);

        // The entire path is convex
        assert!(props.is_convex(&[0, 1, 2, 3]));

        // {0, 1, 2} is convex
        assert!(props.is_convex(&[0, 1, 2]));

        // {0, 2} is NOT convex (missing 1)
        assert!(!props.is_convex(&[0, 2]));
    }

    #[test]
    fn test_is_geodetic_path() {
        // Path graph is geodetic (unique shortest paths)
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        let mut props = ConvexityProperties::new(g);
        assert!(props.is_geodetic());
    }

    #[test]
    fn test_is_geodetic_cycle() {
        // Cycle is NOT geodetic (multiple shortest paths for opposite vertices)
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();
        g.add_edge(3, 0).unwrap();

        let mut props = ConvexityProperties::new(g);
        assert!(!props.is_geodetic());
    }

    #[test]
    fn test_geodetic_closure_function() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        let closure = geodetic_closure(&g, &[0, 3]);
        assert_eq!(closure.len(), 4);
    }

    #[test]
    fn test_is_geodetic_function() {
        // Path graph
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        assert!(is_geodetic(&g));

        // Triangle IS geodetic (all pairs have unique shortest paths via direct edges)
        let mut g2 = Graph::new(3);
        g2.add_edge(0, 1).unwrap();
        g2.add_edge(1, 2).unwrap();
        g2.add_edge(2, 0).unwrap();
        assert!(is_geodetic(&g2));

        // Square is NOT geodetic (opposite corners have 2 paths)
        let mut g3 = Graph::new(4);
        g3.add_edge(0, 1).unwrap();
        g3.add_edge(1, 2).unwrap();
        g3.add_edge(2, 3).unwrap();
        g3.add_edge(3, 0).unwrap();
        assert!(!is_geodetic(&g3));
    }

    #[test]
    fn test_geodetic_number_simple() {
        // Path graph: geodetic number is 2 (endpoints)
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        let mut props = ConvexityProperties::new(g);
        assert_eq!(props.geodetic_number(), 2);
    }

    #[test]
    fn test_convexity_on_disconnected_graph() {
        let mut g = Graph::new(5);
        g.add_edge(0, 1).unwrap();
        g.add_edge(2, 3).unwrap();

        let mut props = ConvexityProperties::new(g);

        // Each component is convex
        assert!(props.is_convex(&[0, 1]));
        assert!(props.is_convex(&[2, 3]));

        // Union is also convex (no paths between components)
        assert!(props.is_convex(&[0, 1, 2, 3]));
    }
}
