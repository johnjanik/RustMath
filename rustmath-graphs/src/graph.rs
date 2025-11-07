//! Graph data structures and basic operations

use std::collections::{HashMap, HashSet, VecDeque};

/// An undirected graph using adjacency list representation
#[derive(Debug, Clone)]
pub struct Graph {
    /// Number of vertices
    num_vertices: usize,
    /// Adjacency list: vertex -> list of neighbors
    adj: Vec<HashSet<usize>>,
}

impl Graph {
    /// Create a new graph with n vertices
    pub fn new(n: usize) -> Self {
        Graph {
            num_vertices: n,
            adj: vec![HashSet::new(); n],
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

    /// Add an edge between vertices u and v
    pub fn add_edge(&mut self, u: usize, v: usize) -> Result<(), String> {
        if u >= self.num_vertices || v >= self.num_vertices {
            return Err(format!("Vertex out of bounds"));
        }
        self.adj[u].insert(v);
        self.adj[v].insert(u);
        Ok(())
    }

    /// Check if there's an edge between u and v
    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        if u >= self.num_vertices || v >= self.num_vertices {
            return false;
        }
        self.adj[u].contains(&v)
    }

    /// Get the degree of a vertex
    pub fn degree(&self, v: usize) -> Option<usize> {
        if v >= self.num_vertices {
            return None;
        }
        Some(self.adj[v].len())
    }

    /// Get neighbors of a vertex
    pub fn neighbors(&self, v: usize) -> Option<Vec<usize>> {
        if v >= self.num_vertices {
            return None;
        }
        Some(self.adj[v].iter().copied().collect())
    }

    /// Perform breadth-first search from a starting vertex
    pub fn bfs(&self, start: usize) -> Result<Vec<usize>, String> {
        if start >= self.num_vertices {
            return Err("Start vertex out of bounds".to_string());
        }

        let mut visited = vec![false; self.num_vertices];
        let mut order = Vec::new();
        let mut queue = VecDeque::new();

        queue.push_back(start);
        visited[start] = true;

        while let Some(v) = queue.pop_front() {
            order.push(v);

            for &neighbor in &self.adj[v] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }

        Ok(order)
    }

    /// Perform depth-first search from a starting vertex
    pub fn dfs(&self, start: usize) -> Result<Vec<usize>, String> {
        if start >= self.num_vertices {
            return Err("Start vertex out of bounds".to_string());
        }

        let mut visited = vec![false; self.num_vertices];
        let mut order = Vec::new();

        self.dfs_recursive(start, &mut visited, &mut order);

        Ok(order)
    }

    fn dfs_recursive(&self, v: usize, visited: &mut [bool], order: &mut Vec<usize>) {
        visited[v] = true;
        order.push(v);

        for &neighbor in &self.adj[v] {
            if !visited[neighbor] {
                self.dfs_recursive(neighbor, visited, order);
            }
        }
    }

    /// Check if the graph is connected
    pub fn is_connected(&self) -> bool {
        if self.num_vertices == 0 {
            return true;
        }

        let visited = self.bfs(0).unwrap();
        visited.len() == self.num_vertices
    }

    /// Find shortest path between two vertices using BFS
    pub fn shortest_path(&self, start: usize, end: usize) -> Result<Option<Vec<usize>>, String> {
        if start >= self.num_vertices || end >= self.num_vertices {
            return Err("Vertex out of bounds".to_string());
        }

        if start == end {
            return Ok(Some(vec![start]));
        }

        let mut visited = vec![false; self.num_vertices];
        let mut parent: HashMap<usize, usize> = HashMap::new();
        let mut queue = VecDeque::new();

        queue.push_back(start);
        visited[start] = true;

        while let Some(v) = queue.pop_front() {
            if v == end {
                // Reconstruct path
                let mut path = vec![end];
                let mut current = end;

                while current != start {
                    current = parent[&current];
                    path.push(current);
                }

                path.reverse();
                return Ok(Some(path));
            }

            for &neighbor in &self.adj[v] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    parent.insert(neighbor, v);
                    queue.push_back(neighbor);
                }
            }
        }

        Ok(None) // No path found
    }

    /// Find all connected components
    ///
    /// Returns a vector where each element is a vector of vertices in that component
    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let mut visited = vec![false; self.num_vertices];
        let mut components = Vec::new();

        for v in 0..self.num_vertices {
            if !visited[v] {
                let mut component = Vec::new();
                let mut queue = VecDeque::new();
                queue.push_back(v);
                visited[v] = true;

                while let Some(u) = queue.pop_front() {
                    component.push(u);

                    for &neighbor in &self.adj[u] {
                        if !visited[neighbor] {
                            visited[neighbor] = true;
                            queue.push_back(neighbor);
                        }
                    }
                }

                components.push(component);
            }
        }

        components
    }

    /// Check if the graph has a cycle
    pub fn has_cycle(&self) -> bool {
        let mut visited = vec![false; self.num_vertices];
        let mut parent = vec![None; self.num_vertices];

        for v in 0..self.num_vertices {
            if !visited[v] && self.has_cycle_dfs(v, &mut visited, &mut parent) {
                return true;
            }
        }

        false
    }

    fn has_cycle_dfs(
        &self,
        v: usize,
        visited: &mut [bool],
        parent: &mut [Option<usize>],
    ) -> bool {
        visited[v] = true;

        for &neighbor in &self.adj[v] {
            if !visited[neighbor] {
                parent[neighbor] = Some(v);
                if self.has_cycle_dfs(neighbor, visited, parent) {
                    return true;
                }
            } else if parent[v] != Some(neighbor) {
                // Found a back edge (cycle)
                return true;
            }
        }

        false
    }

    /// Find a spanning tree using BFS
    ///
    /// Returns edges (as pairs of vertices) that form a spanning tree
    /// Returns None if graph is not connected
    pub fn spanning_tree_bfs(&self) -> Option<Vec<(usize, usize)>> {
        if self.num_vertices == 0 {
            return Some(Vec::new());
        }

        let mut edges = Vec::new();
        let mut visited = vec![false; self.num_vertices];
        let mut queue = VecDeque::new();

        queue.push_back(0);
        visited[0] = true;

        while let Some(v) = queue.pop_front() {
            for &neighbor in &self.adj[v] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    edges.push((v, neighbor));
                    queue.push_back(neighbor);
                }
            }
        }

        // Check if all vertices were visited (graph is connected)
        if visited.iter().all(|&v| v) {
            Some(edges)
        } else {
            None
        }
    }

    /// Check if the graph is bipartite
    ///
    /// A graph is bipartite if its vertices can be divided into two sets
    /// such that all edges connect vertices from different sets
    pub fn is_bipartite(&self) -> bool {
        let mut color = vec![None; self.num_vertices];

        for start in 0..self.num_vertices {
            if color[start].is_none() {
                let mut queue = VecDeque::new();
                queue.push_back(start);
                color[start] = Some(0);

                while let Some(v) = queue.pop_front() {
                    let current_color = color[v].unwrap();

                    for &neighbor in &self.adj[v] {
                        if let Some(neighbor_color) = color[neighbor] {
                            if neighbor_color == current_color {
                                return false; // Adjacent vertices have same color
                            }
                        } else {
                            color[neighbor] = Some(1 - current_color);
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }

        true
    }

    /// Greedy graph coloring
    ///
    /// Returns a coloring (vertex -> color mapping) using a greedy algorithm
    /// The number of colors used may not be optimal
    pub fn greedy_coloring(&self) -> Vec<usize> {
        let mut colors = vec![None; self.num_vertices];

        for v in 0..self.num_vertices {
            // Find colors used by neighbors
            let mut used_colors = HashSet::new();
            for &neighbor in &self.adj[v] {
                if let Some(color) = colors[neighbor] {
                    used_colors.insert(color);
                }
            }

            // Find smallest available color
            let mut color = 0;
            while used_colors.contains(&color) {
                color += 1;
            }

            colors[v] = Some(color);
        }

        colors.into_iter().map(|c| c.unwrap()).collect()
    }

    /// Get the chromatic number (minimum number of colors needed)
    ///
    /// Uses greedy coloring as an approximation
    pub fn chromatic_number(&self) -> usize {
        let coloring = self.greedy_coloring();
        if coloring.is_empty() {
            0
        } else {
            *coloring.iter().max().unwrap() + 1
        }
    }

    /// Find the diameter of the graph (longest shortest path)
    ///
    /// Returns None if graph is not connected
    pub fn diameter(&self) -> Option<usize> {
        if !self.is_connected() {
            return None;
        }

        let mut max_distance = 0;

        for v in 0..self.num_vertices {
            let distances = self.bfs_distances(v);
            if let Some(&max_dist) = distances.iter().max() {
                if max_dist > max_distance {
                    max_distance = max_dist;
                }
            }
        }

        Some(max_distance)
    }

    fn bfs_distances(&self, start: usize) -> Vec<usize> {
        let mut distances = vec![usize::MAX; self.num_vertices];
        let mut queue = VecDeque::new();

        queue.push_back(start);
        distances[start] = 0;

        while let Some(v) = queue.pop_front() {
            for &neighbor in &self.adj[v] {
                if distances[neighbor] == usize::MAX {
                    distances[neighbor] = distances[v] + 1;
                    queue.push_back(neighbor);
                }
            }
        }

        distances
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let g = Graph::new(5);
        assert_eq!(g.num_vertices(), 5);
        assert_eq!(g.num_edges(), 0);
    }

    #[test]
    fn test_add_edge() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        assert!(g.has_edge(0, 1));
        assert!(g.has_edge(1, 2));
        assert!(!g.has_edge(0, 2));
        assert_eq!(g.num_edges(), 2);
    }

    #[test]
    fn test_degree() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 2).unwrap();

        assert_eq!(g.degree(0), Some(2));
        assert_eq!(g.degree(1), Some(1));
        assert_eq!(g.degree(2), Some(1));
    }

    #[test]
    fn test_bfs() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 2).unwrap();
        g.add_edge(1, 3).unwrap();

        let order = g.bfs(0).unwrap();
        assert_eq!(order.len(), 4);
        assert_eq!(order[0], 0); // Start vertex first
    }

    #[test]
    fn test_is_connected() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        assert!(g.is_connected());

        let mut g2 = Graph::new(4);
        g2.add_edge(0, 1).unwrap();
        g2.add_edge(2, 3).unwrap();

        assert!(!g2.is_connected());
    }

    #[test]
    fn test_shortest_path() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();
        g.add_edge(0, 3).unwrap();

        let path = g.shortest_path(0, 3).unwrap().unwrap();
        assert_eq!(path, vec![0, 3]); // Direct path is shorter
    }

    #[test]
    fn test_connected_components() {
        let mut g = Graph::new(6);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(3, 4).unwrap();
        // Vertex 5 is isolated

        let components = g.connected_components();
        assert_eq!(components.len(), 3); // Three components: {0,1,2}, {3,4}, {5}
    }

    #[test]
    fn test_has_cycle() {
        // Tree - no cycle
        let mut g1 = Graph::new(4);
        g1.add_edge(0, 1).unwrap();
        g1.add_edge(0, 2).unwrap();
        g1.add_edge(0, 3).unwrap();
        assert!(!g1.has_cycle());

        // Graph with cycle
        let mut g2 = Graph::new(4);
        g2.add_edge(0, 1).unwrap();
        g2.add_edge(1, 2).unwrap();
        g2.add_edge(2, 3).unwrap();
        g2.add_edge(3, 0).unwrap();
        assert!(g2.has_cycle());
    }

    #[test]
    fn test_spanning_tree() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 2).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        let tree = g.spanning_tree_bfs().unwrap();
        assert_eq!(tree.len(), 3); // Tree has n-1 edges for n vertices
    }

    #[test]
    fn test_is_bipartite() {
        // Bipartite graph (complete bipartite K_{2,2})
        let mut g1 = Graph::new(4);
        g1.add_edge(0, 2).unwrap();
        g1.add_edge(0, 3).unwrap();
        g1.add_edge(1, 2).unwrap();
        g1.add_edge(1, 3).unwrap();
        assert!(g1.is_bipartite());

        // Triangle (odd cycle) - not bipartite
        let mut g2 = Graph::new(3);
        g2.add_edge(0, 1).unwrap();
        g2.add_edge(1, 2).unwrap();
        g2.add_edge(2, 0).unwrap();
        assert!(!g2.is_bipartite());
    }

    #[test]
    fn test_greedy_coloring() {
        // Triangle needs 3 colors
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 0).unwrap();

        let coloring = g.greedy_coloring();
        // Check that adjacent vertices have different colors
        assert_ne!(coloring[0], coloring[1]);
        assert_ne!(coloring[1], coloring[2]);
        assert_ne!(coloring[2], coloring[0]);
    }

    #[test]
    fn test_chromatic_number() {
        // Complete graph K_3 has chromatic number 3
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 0).unwrap();

        assert_eq!(g.chromatic_number(), 3);
    }

    #[test]
    fn test_diameter() {
        // Path graph 0-1-2-3 has diameter 3
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        assert_eq!(g.diameter(), Some(3));

        // Disconnected graph has no diameter
        let mut g2 = Graph::new(4);
        g2.add_edge(0, 1).unwrap();
        g2.add_edge(2, 3).unwrap();
        assert_eq!(g2.diameter(), None);
    }
}
