//! Directed graph data structures and algorithms

use std::collections::{HashSet, VecDeque};

/// A directed graph using adjacency list representation
#[derive(Debug, Clone)]
pub struct DiGraph {
    /// Number of vertices
    num_vertices: usize,
    /// Adjacency list: vertex -> list of outgoing neighbors
    adj: Vec<HashSet<usize>>,
}

impl DiGraph {
    /// Create a new directed graph with n vertices
    pub fn new(n: usize) -> Self {
        DiGraph {
            num_vertices: n,
            adj: vec![HashSet::new(); n],
        }
    }

    /// Get the number of vertices
    pub fn num_vertices(&self) -> usize {
        self.num_vertices
    }

    /// Get the number of directed edges
    pub fn num_edges(&self) -> usize {
        self.adj.iter().map(|neighbors| neighbors.len()).sum()
    }

    /// Add a directed edge from u to v
    pub fn add_edge(&mut self, u: usize, v: usize) -> Result<(), String> {
        if u >= self.num_vertices || v >= self.num_vertices {
            return Err(format!("Vertex out of bounds"));
        }
        self.adj[u].insert(v);
        Ok(())
    }

    /// Check if there's a directed edge from u to v
    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        if u >= self.num_vertices || v >= self.num_vertices {
            return false;
        }
        self.adj[u].contains(&v)
    }

    /// Get the out-degree of a vertex (number of outgoing edges)
    pub fn out_degree(&self, v: usize) -> Option<usize> {
        if v >= self.num_vertices {
            return None;
        }
        Some(self.adj[v].len())
    }

    /// Get the in-degree of a vertex (number of incoming edges)
    pub fn in_degree(&self, v: usize) -> Option<usize> {
        if v >= self.num_vertices {
            return None;
        }

        let mut count = 0;
        for u in 0..self.num_vertices {
            if self.adj[u].contains(&v) {
                count += 1;
            }
        }
        Some(count)
    }

    /// Get all vertices
    pub fn vertices(&self) -> Vec<usize> {
        (0..self.num_vertices).collect()
    }

    /// Get all directed edges as (from, to) pairs
    pub fn edges(&self) -> Vec<(usize, usize)> {
        let mut edges = Vec::new();
        for u in 0..self.num_vertices {
            for &v in &self.adj[u] {
                edges.push((u, v));
            }
        }
        edges
    }

    /// Perform topological sort (Kahn's algorithm)
    ///
    /// Returns None if the graph contains a cycle
    pub fn topological_sort(&self) -> Option<Vec<usize>> {
        let mut in_degree = vec![0; self.num_vertices];

        // Compute in-degrees
        for u in 0..self.num_vertices {
            for &v in &self.adj[u] {
                in_degree[v] += 1;
            }
        }

        // Queue of vertices with in-degree 0
        let mut queue: VecDeque<usize> = in_degree
            .iter()
            .enumerate()
            .filter(|(_, &deg)| deg == 0)
            .map(|(v, _)| v)
            .collect();

        let mut result = Vec::new();

        while let Some(u) = queue.pop_front() {
            result.push(u);

            for &v in &self.adj[u] {
                in_degree[v] -= 1;
                if in_degree[v] == 0 {
                    queue.push_back(v);
                }
            }
        }

        // If all vertices are in result, we have a valid topological sort
        if result.len() == self.num_vertices {
            Some(result)
        } else {
            None // Cycle detected
        }
    }

    /// Check if the graph is strongly connected
    ///
    /// A directed graph is strongly connected if there's a path from every vertex to every other vertex
    pub fn is_strongly_connected(&self) -> bool {
        if self.num_vertices == 0 {
            return true;
        }

        // Check if all vertices are reachable from vertex 0
        if !self.all_reachable_from(0) {
            return false;
        }

        // Check if all vertices can reach vertex 0 (by checking transpose)
        let transposed = self.transpose();
        transposed.all_reachable_from(0)
    }

    /// Check if all vertices are reachable from a given start vertex
    fn all_reachable_from(&self, start: usize) -> bool {
        let mut visited = vec![false; self.num_vertices];
        let mut stack = vec![start];
        visited[start] = true;
        let mut count = 1;

        while let Some(v) = stack.pop() {
            for &neighbor in &self.adj[v] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    stack.push(neighbor);
                    count += 1;
                }
            }
        }

        count == self.num_vertices
    }

    /// Get the transpose (reverse) of the graph
    fn transpose(&self) -> DiGraph {
        let mut transposed = DiGraph::new(self.num_vertices);

        for u in 0..self.num_vertices {
            for &v in &self.adj[u] {
                transposed.add_edge(v, u).unwrap();
            }
        }

        transposed
    }

    /// Find strongly connected components using Kosaraju's algorithm
    pub fn strongly_connected_components(&self) -> Vec<Vec<usize>> {
        // First DFS to get finish times
        let mut visited = vec![false; self.num_vertices];
        let mut finish_stack = Vec::new();

        for v in 0..self.num_vertices {
            if !visited[v] {
                self.dfs_finish_time(v, &mut visited, &mut finish_stack);
            }
        }

        // Second DFS on transposed graph in reverse finish time order
        let transposed = self.transpose();
        let mut visited = vec![false; self.num_vertices];
        let mut components = Vec::new();

        while let Some(v) = finish_stack.pop() {
            if !visited[v] {
                let mut component = Vec::new();
                transposed.dfs_collect(v, &mut visited, &mut component);
                components.push(component);
            }
        }

        components
    }

    /// DFS helper to collect finish times
    fn dfs_finish_time(&self, v: usize, visited: &mut Vec<bool>, finish_stack: &mut Vec<usize>) {
        visited[v] = true;

        for &neighbor in &self.adj[v] {
            if !visited[neighbor] {
                self.dfs_finish_time(neighbor, visited, finish_stack);
            }
        }

        finish_stack.push(v);
    }

    /// DFS helper to collect component vertices
    fn dfs_collect(&self, v: usize, visited: &mut Vec<bool>, component: &mut Vec<usize>) {
        visited[v] = true;
        component.push(v);

        for &neighbor in &self.adj[v] {
            if !visited[neighbor] {
                self.dfs_collect(neighbor, visited, component);
            }
        }
    }

    /// Check if the graph is a DAG (directed acyclic graph)
    pub fn is_dag(&self) -> bool {
        self.topological_sort().is_some()
    }

    /// Check if the graph has a cycle
    pub fn has_cycle(&self) -> bool {
        !self.is_dag()
    }

    /// Find a shortest path from start to end (BFS for unweighted directed graph)
    pub fn shortest_path(&self, start: usize, end: usize) -> Result<Option<Vec<usize>>, String> {
        if start >= self.num_vertices || end >= self.num_vertices {
            return Err("Vertex out of bounds".to_string());
        }

        if start == end {
            return Ok(Some(vec![start]));
        }

        let mut visited = vec![false; self.num_vertices];
        let mut parent = vec![None; self.num_vertices];
        let mut queue = VecDeque::new();

        queue.push_back(start);
        visited[start] = true;

        while let Some(v) = queue.pop_front() {
            for &neighbor in &self.adj[v] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    parent[neighbor] = Some(v);
                    queue.push_back(neighbor);

                    if neighbor == end {
                        // Reconstruct path
                        let mut path = vec![end];
                        let mut current = end;

                        while let Some(prev) = parent[current] {
                            path.push(prev);
                            current = prev;
                        }

                        path.reverse();
                        return Ok(Some(path));
                    }
                }
            }
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_digraph_creation() {
        let g = DiGraph::new(5);
        assert_eq!(g.num_vertices(), 5);
        assert_eq!(g.num_edges(), 0);
    }

    #[test]
    fn test_add_directed_edge() {
        let mut g = DiGraph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        assert!(g.has_edge(0, 1));
        assert!(g.has_edge(1, 2));
        assert!(!g.has_edge(1, 0)); // Not bidirectional
        assert_eq!(g.num_edges(), 2);
    }

    #[test]
    fn test_degrees() {
        let mut g = DiGraph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 2).unwrap();
        g.add_edge(1, 2).unwrap();

        assert_eq!(g.out_degree(0), Some(2));
        assert_eq!(g.out_degree(1), Some(1));
        assert_eq!(g.out_degree(2), Some(0));

        assert_eq!(g.in_degree(0), Some(0));
        assert_eq!(g.in_degree(1), Some(1));
        assert_eq!(g.in_degree(2), Some(2));
    }

    #[test]
    fn test_topological_sort() {
        // DAG: 0 -> 1 -> 3, 0 -> 2 -> 3
        let mut g = DiGraph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 2).unwrap();
        g.add_edge(1, 3).unwrap();
        g.add_edge(2, 3).unwrap();

        let topo = g.topological_sort();
        assert!(topo.is_some());
        let order = topo.unwrap();
        assert_eq!(order.len(), 4);
        // 0 must come before 1 and 2, and both must come before 3
        let pos: std::collections::HashMap<usize, usize> =
            order.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        assert!(pos[&0] < pos[&1]);
        assert!(pos[&0] < pos[&2]);
        assert!(pos[&1] < pos[&3]);
        assert!(pos[&2] < pos[&3]);
    }

    #[test]
    fn test_topological_sort_with_cycle() {
        // Graph with cycle: 0 -> 1 -> 2 -> 0
        let mut g = DiGraph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 0).unwrap();

        assert!(g.topological_sort().is_none());
        assert!(g.has_cycle());
        assert!(!g.is_dag());
    }

    #[test]
    fn test_strongly_connected_components() {
        // Graph with two SCCs: {0,1,2} and {3,4}
        let mut g = DiGraph::new(5);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 0).unwrap();
        g.add_edge(1, 3).unwrap();
        g.add_edge(3, 4).unwrap();
        g.add_edge(4, 3).unwrap();

        let sccs = g.strongly_connected_components();
        assert_eq!(sccs.len(), 2);

        // Check that each SCC is correct
        for scc in &sccs {
            if scc.contains(&0) {
                assert_eq!(scc.len(), 3);
                assert!(scc.contains(&1) && scc.contains(&2));
            } else {
                assert_eq!(scc.len(), 2);
                assert!(scc.contains(&3) && scc.contains(&4));
            }
        }
    }

    #[test]
    fn test_is_strongly_connected() {
        // Strongly connected: cycle 0 -> 1 -> 2 -> 0
        let mut g1 = DiGraph::new(3);
        g1.add_edge(0, 1).unwrap();
        g1.add_edge(1, 2).unwrap();
        g1.add_edge(2, 0).unwrap();
        assert!(g1.is_strongly_connected());

        // Not strongly connected: 0 -> 1 -> 2
        let mut g2 = DiGraph::new(3);
        g2.add_edge(0, 1).unwrap();
        g2.add_edge(1, 2).unwrap();
        assert!(!g2.is_strongly_connected());
    }

    #[test]
    fn test_shortest_path() {
        let mut g = DiGraph::new(5);
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 2).unwrap();
        g.add_edge(1, 3).unwrap();
        g.add_edge(2, 3).unwrap();
        g.add_edge(3, 4).unwrap();

        let path = g.shortest_path(0, 4).unwrap();
        assert!(path.is_some());
        let p = path.unwrap();
        assert_eq!(p[0], 0);
        assert_eq!(p[p.len() - 1], 4);
        assert_eq!(p.len(), 4); // 0 -> 1 -> 3 -> 4 or 0 -> 2 -> 3 -> 4
    }
}
