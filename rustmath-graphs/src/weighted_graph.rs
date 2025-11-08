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

/// A node in the priority queue for A* search
#[derive(Debug, Clone, Eq, PartialEq)]
struct AStarNode {
    vertex: usize,
    f_score: i64,  // g_score + heuristic
    g_score: i64,  // actual cost from start
}

impl Ord for AStarNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (prioritize lower f_score)
        other.f_score.cmp(&self.f_score)
            .then_with(|| self.vertex.cmp(&other.vertex))
    }
}

impl PartialOrd for AStarNode {
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

    /// A* search algorithm for pathfinding with heuristic
    ///
    /// Finds shortest path from start to goal using a heuristic function.
    /// The heuristic should be admissible (never overestimate) for optimal results.
    ///
    /// # Arguments
    /// * `start` - Starting vertex
    /// * `goal` - Goal vertex
    /// * `heuristic` - Function mapping vertex to estimated distance to goal
    ///
    /// # Returns
    /// Returns Some((path, cost)) if path exists, None otherwise
    pub fn astar<F>(&self, start: usize, goal: usize, heuristic: F) -> Result<Option<(Vec<usize>, i64)>, String>
    where
        F: Fn(usize) -> i64,
    {
        if start >= self.num_vertices || goal >= self.num_vertices {
            return Err("Vertex out of bounds".to_string());
        }

        if start == goal {
            return Ok(Some((vec![start], 0)));
        }

        // g_score: cost from start to vertex
        let mut g_score = vec![None; self.num_vertices];
        g_score[start] = Some(0);

        // Parent tracking for path reconstruction
        let mut parent: HashMap<usize, usize> = HashMap::new();

        // Priority queue ordered by f_score = g_score + heuristic
        let mut heap = BinaryHeap::new();
        heap.push(AStarNode {
            vertex: start,
            f_score: heuristic(start),
            g_score: 0,
        });

        let mut visited = HashSet::new();

        while let Some(AStarNode { vertex: current, g_score: current_g, .. }) = heap.pop() {
            // Found goal
            if current == goal {
                let mut path = vec![goal];
                let mut v = goal;
                while v != start {
                    if let Some(&prev) = parent.get(&v) {
                        path.push(prev);
                        v = prev;
                    } else {
                        break;
                    }
                }
                path.reverse();
                return Ok(Some((path, current_g)));
            }

            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);

            // Explore neighbors
            for &(neighbor, weight) in &self.adj[current] {
                if visited.contains(&neighbor) {
                    continue;
                }

                let tentative_g = current_g + weight;

                if g_score[neighbor].is_none() || tentative_g < g_score[neighbor].unwrap() {
                    g_score[neighbor] = Some(tentative_g);
                    parent.insert(neighbor, current);

                    let f = tentative_g + heuristic(neighbor);
                    heap.push(AStarNode {
                        vertex: neighbor,
                        f_score: f,
                        g_score: tentative_g,
                    });
                }
            }
        }

        Ok(None)
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

    /// Hungarian algorithm for maximum weight bipartite matching
    ///
    /// Solves the assignment problem for a weighted bipartite graph.
    /// Assumes the graph is bipartite with partitions [0..n/2) and [n/2..n).
    ///
    /// # Returns
    /// Returns Some(matching, total_weight) if a perfect matching exists,
    /// None if the graph is not bipartite or no perfect matching exists.
    pub fn hungarian(&self) -> Option<(Vec<(usize, usize)>, i64)> {
        if self.num_vertices == 0 || self.num_vertices % 2 != 0 {
            return None;
        }

        let n = self.num_vertices / 2;

        // Build cost matrix (negate for max weight matching)
        let mut cost = vec![vec![i64::MIN / 2; n]; n];
        for u in 0..n {
            for &(v, weight) in &self.adj[u] {
                if v >= n && v < 2 * n {
                    cost[u][v - n] = -weight; // Negate for minimization
                }
            }
        }

        // Hungarian algorithm implementation
        let assignment = hungarian_solve(&cost)?;

        // Build matching and calculate total weight
        let mut matching = Vec::new();
        let mut total_weight = 0i64;

        for (u, v) in assignment.iter().enumerate() {
            if let Some(v_idx) = v {
                if cost[u][*v_idx] != i64::MIN / 2 {
                    matching.push((u, v_idx + n));
                    total_weight -= cost[u][*v_idx]; // Negate back
                }
            }
        }

        if matching.len() == n {
            Some((matching, total_weight))
        } else {
            None
        }
    }

    /// Find shortest paths using Bellman-Ford algorithm
    ///
    /// Handles negative edge weights and detects negative cycles.
    /// Returns distances or None if a negative cycle is detected.
    /// Time complexity: O(VE)
    pub fn bellman_ford(&self, source: usize) -> Result<Option<Vec<Option<i64>>>, String> {
        if source >= self.num_vertices {
            return Err("Source vertex out of bounds".to_string());
        }

        let mut distances = vec![None; self.num_vertices];
        distances[source] = Some(0);

        // Relax edges V-1 times
        for _ in 0..self.num_vertices - 1 {
            let mut updated = false;

            for u in 0..self.num_vertices {
                if let Some(dist_u) = distances[u] {
                    for &(v, weight) in &self.adj[u] {
                        let new_dist = dist_u + weight;

                        if distances[v].is_none() || new_dist < distances[v].unwrap() {
                            distances[v] = Some(new_dist);
                            updated = true;
                        }
                    }
                }
            }

            // Early termination if no updates
            if !updated {
                break;
            }
        }

        // Check for negative cycles
        for u in 0..self.num_vertices {
            if let Some(dist_u) = distances[u] {
                for &(v, weight) in &self.adj[u] {
                    let new_dist = dist_u + weight;

                    if distances[v].is_none() || new_dist < distances[v].unwrap() {
                        // Negative cycle detected
                        return Ok(None);
                    }
                }
            }
        }

        Ok(Some(distances))
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

/// Solve the assignment problem using the Hungarian algorithm
///
/// Takes a cost matrix and returns an assignment vector where
/// assignment[i] = Some(j) means row i is matched to column j.
fn hungarian_solve(cost: &[Vec<i64>]) -> Option<Vec<Option<usize>>> {
    let n = cost.len();
    if n == 0 {
        return Some(vec![]);
    }

    // Initialize
    let mut u = vec![0i64; n + 1]; // Potentials for rows
    let mut v = vec![0i64; n + 1]; // Potentials for columns
    let mut match_col = vec![None; n + 1]; // match_col[j] = row matched to column j
    let mut match_row = vec![None; n + 1]; // match_row[i] = column matched to row i

    for i in 0..n {
        let mut min_val = vec![i64::MAX; n + 1];
        let mut visited = vec![false; n + 1];
        let mut prev = vec![None; n + 1];

        let mut col = n; // Start with dummy column
        match_col[col] = Some(i);
        visited[col] = true;

        loop {
            let row = match_col[col].unwrap();
            let mut delta = i64::MAX;
            let mut next_col = None;

            // Find minimum slack edge
            for j in 0..n {
                if !visited[j] {
                    let cur = if cost[row][j] == i64::MIN / 2 {
                        i64::MAX
                    } else {
                        cost[row][j] - u[row] - v[j]
                    };

                    if cur < min_val[j] {
                        min_val[j] = cur;
                        prev[j] = Some(col);
                    }

                    if min_val[j] < delta {
                        delta = min_val[j];
                        next_col = Some(j);
                    }
                }
            }

            // Update potentials
            for j in 0..=n {
                if visited[j] {
                    u[match_col[j].unwrap()] += delta;
                    v[j] -= delta;
                } else {
                    min_val[j] = min_val[j].saturating_sub(delta);
                }
            }

            col = next_col?;
            visited[col] = true;

            if match_col[col].is_none() {
                // Augmenting path found
                while col != n {
                    let prev_col = prev[col]?;
                    match_col[col] = match_col[prev_col];
                    col = prev_col;
                }
                break;
            }
        }
    }

    // Build result
    for j in 0..n {
        if let Some(i) = match_col[j] {
            match_row[i] = Some(j);
        }
    }

    Some(match_row[..n].to_vec())
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

    #[test]
    fn test_bellman_ford() {
        // Note: In undirected graphs, ANY negative edge creates a negative cycle
        // (since u->v->u with negative weight = 2*negative_weight < 0)
        // So we test with non-negative weights here
        let mut g = WeightedGraph::new(5);
        g.add_edge(0, 1, 4).unwrap();
        g.add_edge(0, 2, 1).unwrap();
        g.add_edge(1, 2, 2).unwrap();
        g.add_edge(1, 3, 1).unwrap();
        g.add_edge(2, 3, 5).unwrap();
        g.add_edge(3, 4, 3).unwrap();

        let result = g.bellman_ford(0).unwrap();
        assert!(result.is_some());
        let distances = result.unwrap();

        assert_eq!(distances[0], Some(0));
        assert_eq!(distances[1], Some(3)); // 0->2->1
        assert_eq!(distances[2], Some(1)); // 0->2
        assert_eq!(distances[3], Some(4)); // 0->2->1->3
        assert_eq!(distances[4], Some(7)); // 0->2->1->3->4
    }

    #[test]
    fn test_bellman_ford_negative_cycle() {
        // Graph with a negative cycle
        let mut g = WeightedGraph::new(3);
        g.add_edge(0, 1, 1).unwrap();
        g.add_edge(1, 2, -1).unwrap();
        g.add_edge(2, 0, -1).unwrap(); // Creates negative cycle: 0->1->2->0 = 1-1-1 = -1

        let result = g.bellman_ford(0).unwrap();
        assert!(result.is_none()); // Should detect negative cycle
    }

    #[test]
    fn test_astar() {
        // Grid-like graph where A* should benefit from heuristic
        let mut g = WeightedGraph::new(9);

        // Create a 3x3 grid graph
        // 0-1-2
        // | | |
        // 3-4-5
        // | | |
        // 6-7-8

        // Horizontal edges
        g.add_edge(0, 1, 1).unwrap();
        g.add_edge(1, 2, 1).unwrap();
        g.add_edge(3, 4, 1).unwrap();
        g.add_edge(4, 5, 1).unwrap();
        g.add_edge(6, 7, 1).unwrap();
        g.add_edge(7, 8, 1).unwrap();

        // Vertical edges
        g.add_edge(0, 3, 1).unwrap();
        g.add_edge(1, 4, 1).unwrap();
        g.add_edge(2, 5, 1).unwrap();
        g.add_edge(3, 6, 1).unwrap();
        g.add_edge(4, 7, 1).unwrap();
        g.add_edge(5, 8, 1).unwrap();

        // Manhattan distance heuristic for grid
        let heuristic = |v: usize| -> i64 {
            let goal = 8;
            let vx = (v % 3) as i64;
            let vy = (v / 3) as i64;
            let gx = (goal % 3) as i64;
            let gy = (goal / 3) as i64;
            (vx - gx).abs() + (vy - gy).abs()
        };

        let result = g.astar(0, 8, heuristic).unwrap();
        assert!(result.is_some());

        let (path, cost) = result.unwrap();
        assert_eq!(cost, 4); // Shortest path from corner to corner is 4
        assert_eq!(path[0], 0);
        assert_eq!(path[path.len() - 1], 8);
    }

    #[test]
    fn test_astar_no_path() {
        // Disconnected graph
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 1, 1).unwrap();
        g.add_edge(2, 3, 1).unwrap();

        let heuristic = |_: usize| -> i64 { 0 };
        let result = g.astar(0, 3, heuristic).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_hungarian() {
        // Simple bipartite graph: vertices 0,1 on left, 2,3 on right
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 2, 10).unwrap(); // Left 0 to Right 0 with weight 10
        g.add_edge(0, 3, 15).unwrap(); // Left 0 to Right 1 with weight 15
        g.add_edge(1, 2, 20).unwrap(); // Left 1 to Right 0 with weight 20
        g.add_edge(1, 3, 18).unwrap(); // Left 1 to Right 1 with weight 18

        let result = g.hungarian();
        assert!(result.is_some());

        let (matching, total_weight) = result.unwrap();
        assert_eq!(matching.len(), 2);
        // Optimal matching: 0->3 (15) and 1->2 (20) = 35
        assert_eq!(total_weight, 35);
    }

    #[test]
    fn test_hungarian_simple() {
        // Smaller example: 2 vertices on each side
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 2, 5).unwrap();
        g.add_edge(1, 3, 7).unwrap();

        let result = g.hungarian();
        assert!(result.is_some());

        let (matching, total_weight) = result.unwrap();
        assert_eq!(matching.len(), 2);
        assert_eq!(total_weight, 12); // 5 + 7
    }
}
