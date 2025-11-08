//! Weighted graph data structures and algorithms

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

/// A weighted undirected graph
#[derive(Debug, Clone)]
pub struct WeightedGraph {
    /// Number of vertices
    num_vertices: usize,
    /// Adjacency list: vertex -> list of (neighbor, weight) pairs
    adj: Vec<Vec<(usize, i64)>>,
}

/// A node in the priority queue for Dijkstra's algorithm
#[derive(Debug, Clone, Eq, PartialEq)]
struct DijkstraNode {
    vertex: usize,
    distance: i64,
}

impl Ord for DijkstraNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap
        other.distance.cmp(&self.distance)
            .then_with(|| self.vertex.cmp(&other.vertex))
    }
}

impl PartialOrd for DijkstraNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl WeightedGraph {
    /// Create a new weighted graph with n vertices
    pub fn new(n: usize) -> Self {
        WeightedGraph {
            num_vertices: n,
            adj: vec![Vec::new(); n],
        }
    }

    /// Get the number of vertices
    pub fn num_vertices(&self) -> usize {
        self.num_vertices
    }

    /// Get the number of edges
    pub fn num_edges(&self) -> usize {
        self.adj.iter().map(|neighbors| neighbors.len()).sum::<usize>() / 2
    }

    /// Add a weighted edge between vertices u and v
    pub fn add_edge(&mut self, u: usize, v: usize, weight: i64) -> Result<(), String> {
        if u >= self.num_vertices || v >= self.num_vertices {
            return Err(format!("Vertex out of bounds"));
        }
        self.adj[u].push((v, weight));
        self.adj[v].push((u, weight));
        Ok(())
    }

    /// Get all edges with weights
    pub fn edges(&self) -> Vec<(usize, usize, i64)> {
        let mut edges = Vec::new();
        for u in 0..self.num_vertices {
            for &(v, weight) in &self.adj[u] {
                if u < v {
                    edges.push((u, v, weight));
                }
            }
        }
        edges
    }

    /// Find shortest paths from a source using Dijkstra's algorithm
    ///
    /// Returns distances to all vertices and the parent map for path reconstruction.
    /// Uses a min-heap for O((V + E) log V) complexity.
    pub fn dijkstra(&self, source: usize) -> Result<(Vec<Option<i64>>, HashMap<usize, usize>), String> {
        if source >= self.num_vertices {
            return Err("Source vertex out of bounds".to_string());
        }

        let mut distances = vec![None; self.num_vertices];
        let mut parent: HashMap<usize, usize> = HashMap::new();
        let mut heap = BinaryHeap::new();

        distances[source] = Some(0);
        heap.push(DijkstraNode {
            vertex: source,
            distance: 0,
        });

        while let Some(DijkstraNode { vertex: u, distance: dist_u }) = heap.pop() {
            // Skip if we've found a better path already
            if let Some(current_dist) = distances[u] {
                if dist_u > current_dist {
                    continue;
                }
            }

            // Explore neighbors
            for &(v, weight) in &self.adj[u] {
                let new_dist = dist_u + weight;

                if distances[v].is_none() || new_dist < distances[v].unwrap() {
                    distances[v] = Some(new_dist);
                    parent.insert(v, u);
                    heap.push(DijkstraNode {
                        vertex: v,
                        distance: new_dist,
                    });
                }
            }
        }

        Ok((distances, parent))
    }

    /// Get the shortest path between two vertices using Dijkstra
    pub fn shortest_path(&self, start: usize, end: usize) -> Result<Option<(Vec<usize>, i64)>, String> {
        let (distances, parent) = self.dijkstra(start)?;

        match distances[end] {
            None => Ok(None),
            Some(dist) => {
                // Reconstruct path
                let mut path = vec![end];
                let mut current = end;

                while current != start {
                    if let Some(&prev) = parent.get(&current) {
                        path.push(prev);
                        current = prev;
                    } else {
                        return Ok(None);
                    }
                }

                path.reverse();
                Ok(Some((path, dist)))
            }
        }
    }

    /// Find minimum spanning tree using Prim's algorithm
    ///
    /// Returns edges in the MST or None if graph is disconnected.
    pub fn prim_mst(&self) -> Option<Vec<(usize, usize, i64)>> {
        if self.num_vertices == 0 {
            return Some(vec![]);
        }

        let mut mst = Vec::new();
        let mut in_mst = vec![false; self.num_vertices];
        let mut heap = BinaryHeap::new();

        // Start from vertex 0
        in_mst[0] = true;
        for &(v, weight) in &self.adj[0] {
            heap.push(PrimEdge {
                from: 0,
                to: v,
                weight: -weight, // Negative for min-heap
            });
        }

        while let Some(PrimEdge { from: u, to: v, weight }) = heap.pop() {
            if in_mst[v] {
                continue;
            }

            // Add edge to MST
            mst.push((u, v, -weight));
            in_mst[v] = true;

            // Add new edges from v
            for &(w, edge_weight) in &self.adj[v] {
                if !in_mst[w] {
                    heap.push(PrimEdge {
                        from: v,
                        to: w,
                        weight: -edge_weight,
                    });
                }
            }
        }

        // Check if all vertices are in MST
        if in_mst.iter().all(|&x| x) {
            Some(mst)
        } else {
            None
        }
    }

    /// Find all-pairs shortest paths using Floyd-Warshall algorithm
    ///
    /// Returns a matrix of shortest distances.
    /// Returns None for cells with no path.
    pub fn floyd_warshall(&self) -> Vec<Vec<Option<i64>>> {
        let n = self.num_vertices;
        let mut dist = vec![vec![None; n]; n];

        // Initialize with direct edges
        for i in 0..n {
            dist[i][i] = Some(0);
        }

        for u in 0..n {
            for &(v, weight) in &self.adj[u] {
                dist[u][v] = Some(weight);
            }
        }

        // Floyd-Warshall
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if let (Some(d_ik), Some(d_kj)) = (dist[i][k], dist[k][j]) {
                        let new_dist = d_ik + d_kj;
                        if dist[i][j].is_none() || new_dist < dist[i][j].unwrap() {
                            dist[i][j] = Some(new_dist);
                        }
                    }
                }
            }
        }

        dist
    }

    /// Check if the graph is connected
    pub fn is_connected(&self) -> bool {
        if self.num_vertices == 0 {
            return true;
        }

        let mut visited = vec![false; self.num_vertices];
        let mut stack = vec![0];
        visited[0] = true;
        let mut count = 1;

        while let Some(v) = stack.pop() {
            for &(neighbor, _) in &self.adj[v] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    stack.push(neighbor);
                    count += 1;
                }
            }
        }

        count == self.num_vertices
    }
}

/// Edge for Prim's algorithm priority queue
#[derive(Debug, Clone, Eq, PartialEq)]
struct PrimEdge {
    from: usize,
    to: usize,
    weight: i64, // Negative for min-heap behavior
}

impl Ord for PrimEdge {
    fn cmp(&self, other: &Self) -> Ordering {
        self.weight.cmp(&other.weight)
            .then_with(|| self.from.cmp(&other.from))
            .then_with(|| self.to.cmp(&other.to))
    }
}

impl PartialOrd for PrimEdge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weighted_graph_creation() {
        let g = WeightedGraph::new(5);
        assert_eq!(g.num_vertices(), 5);
        assert_eq!(g.num_edges(), 0);
    }

    #[test]
    fn test_add_weighted_edge() {
        let mut g = WeightedGraph::new(3);
        g.add_edge(0, 1, 5).unwrap();
        g.add_edge(1, 2, 3).unwrap();

        assert_eq!(g.num_edges(), 2);
        let edges = g.edges();
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn test_dijkstra() {
        let mut g = WeightedGraph::new(5);
        g.add_edge(0, 1, 4).unwrap();
        g.add_edge(0, 2, 1).unwrap();
        g.add_edge(1, 3, 1).unwrap();
        g.add_edge(2, 1, 2).unwrap();
        g.add_edge(2, 3, 5).unwrap();
        g.add_edge(3, 4, 3).unwrap();

        let (distances, _) = g.dijkstra(0).unwrap();
        assert_eq!(distances[0], Some(0));
        assert_eq!(distances[1], Some(3)); // 0->2->1
        assert_eq!(distances[2], Some(1)); // 0->2
        assert_eq!(distances[3], Some(4)); // 0->2->1->3
        assert_eq!(distances[4], Some(7)); // 0->2->1->3->4
    }

    #[test]
    fn test_shortest_path() {
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 1, 1).unwrap();
        g.add_edge(1, 2, 2).unwrap();
        g.add_edge(2, 3, 3).unwrap();
        g.add_edge(0, 3, 10).unwrap();

        let result = g.shortest_path(0, 3).unwrap();
        assert!(result.is_some());
        let (path, dist) = result.unwrap();
        assert_eq!(dist, 6); // 0->1->2->3
        assert_eq!(path, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_prim_mst() {
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 1, 1).unwrap();
        g.add_edge(0, 2, 4).unwrap();
        g.add_edge(1, 2, 2).unwrap();
        g.add_edge(1, 3, 5).unwrap();
        g.add_edge(2, 3, 3).unwrap();

        let mst = g.prim_mst().unwrap();
        assert_eq!(mst.len(), 3); // MST has n-1 edges

        // Total weight should be 6 (edges 0-1, 1-2, 2-3)
        let total_weight: i64 = mst.iter().map(|(_, _, w)| w).sum();
        assert_eq!(total_weight, 6);
    }

    #[test]
    fn test_floyd_warshall() {
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 1, 1).unwrap();
        g.add_edge(1, 2, 2).unwrap();
        g.add_edge(2, 3, 3).unwrap();
        g.add_edge(0, 3, 10).unwrap();

        let dist = g.floyd_warshall();
        assert_eq!(dist[0][3], Some(6)); // Shortest path 0->1->2->3
        assert_eq!(dist[0][0], Some(0)); // Distance to self
        assert_eq!(dist[1][3], Some(5)); // 1->2->3
    }

    #[test]
    fn test_disconnected_graph() {
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 1, 1).unwrap();
        g.add_edge(2, 3, 1).unwrap();

        assert!(!g.is_connected());

        let mst = g.prim_mst();
        assert!(mst.is_none()); // No MST for disconnected graph
    }
}
