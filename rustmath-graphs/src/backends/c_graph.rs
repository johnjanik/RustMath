//! C-level graph backend implementations
//!
//! Corresponds to sage.graphs.base.c_graph

use super::generic_backend::GenericGraphBackend;
use std::collections::{HashMap, HashSet, VecDeque};

/// C-level graph backend base class
///
/// This provides a common interface for graph backends with efficient
/// C-level (Rust-level) implementations. Corresponds to SageMath's CGraphBackend.
#[derive(Debug, Clone)]
pub struct CGraphBackend {
    /// Whether the graph is directed
    directed: bool,
    /// Number of vertices
    num_vertices: usize,
    /// Adjacency structure (depends on specific backend)
    /// For this base implementation, we use adjacency lists
    out_neighbors: Vec<HashSet<usize>>,
    /// In-neighbors for directed graphs
    in_neighbors: Vec<HashSet<usize>>,
    /// Edge labels
    edge_labels: HashMap<(usize, usize), String>,
    /// Whether loops are allowed
    loops: bool,
    /// Whether multiple edges are allowed
    multiedges: bool,
}

impl CGraphBackend {
    /// Create a new C-level graph backend
    pub fn new_with_options(directed: bool, loops: bool, multiedges: bool) -> Self {
        CGraphBackend {
            directed,
            num_vertices: 0,
            out_neighbors: Vec::new(),
            in_neighbors: if directed { Vec::new() } else { Vec::new() },
            edge_labels: HashMap::new(),
            loops,
            multiedges,
        }
    }

    /// Get all in-neighbors of a vertex (for directed graphs)
    pub fn in_neighbors(&self, v: usize) -> Option<Vec<usize>> {
        if v >= self.num_vertices {
            return None;
        }
        if self.directed {
            Some(self.in_neighbors[v].iter().copied().collect())
        } else {
            // For undirected graphs, in-neighbors = out-neighbors
            Some(self.out_neighbors[v].iter().copied().collect())
        }
    }

    /// Get all out-neighbors of a vertex
    pub fn out_neighbors(&self, v: usize) -> Option<Vec<usize>> {
        if v >= self.num_vertices {
            return None;
        }
        Some(self.out_neighbors[v].iter().copied().collect())
    }
}

impl GenericGraphBackend for CGraphBackend {
    fn new(directed: bool) -> Self {
        Self::new_with_options(directed, false, false)
    }

    fn is_directed(&self) -> bool {
        self.directed
    }

    fn num_vertices(&self) -> usize {
        self.num_vertices
    }

    fn num_edges(&self) -> usize {
        if self.directed {
            self.out_neighbors.iter().map(|n| n.len()).sum()
        } else {
            self.out_neighbors.iter().map(|n| n.len()).sum::<usize>() / 2
        }
    }

    fn add_vertex(&mut self) -> usize {
        let idx = self.num_vertices;
        self.num_vertices += 1;
        self.out_neighbors.push(HashSet::new());
        if self.directed {
            self.in_neighbors.push(HashSet::new());
        }
        idx
    }

    fn add_edge(&mut self, u: usize, v: usize, label: Option<String>, _directed: Option<bool>) -> Result<(), String> {
        if u >= self.num_vertices || v >= self.num_vertices {
            return Err(format!("Vertex out of bounds: u={}, v={}, num_vertices={}", u, v, self.num_vertices));
        }

        // Check for loops
        if !self.loops && u == v {
            return Err("Loops not allowed in this graph".to_string());
        }

        self.out_neighbors[u].insert(v);
        if self.directed {
            self.in_neighbors[v].insert(u);
        } else {
            self.out_neighbors[v].insert(u);
        }

        if let Some(lbl) = label {
            self.edge_labels.insert((u, v), lbl);
        }

        Ok(())
    }

    fn del_edge(&mut self, u: usize, v: usize) -> Result<(), String> {
        if u >= self.num_vertices || v >= self.num_vertices {
            return Err("Vertex out of bounds".to_string());
        }

        self.out_neighbors[u].remove(&v);
        if self.directed {
            self.in_neighbors[v].remove(&u);
        } else {
            self.out_neighbors[v].remove(&u);
        }

        self.edge_labels.remove(&(u, v));

        Ok(())
    }

    fn has_edge(&self, u: usize, v: usize) -> bool {
        if u >= self.num_vertices || v >= self.num_vertices {
            return false;
        }
        self.out_neighbors[u].contains(&v)
    }

    fn neighbors(&self, v: usize) -> Option<Vec<usize>> {
        self.out_neighbors(v)
    }

    fn in_degree(&self, v: usize) -> Option<usize> {
        if v >= self.num_vertices {
            return None;
        }
        if self.directed {
            Some(self.in_neighbors[v].len())
        } else {
            Some(self.out_neighbors[v].len())
        }
    }

    fn out_degree(&self, v: usize) -> Option<usize> {
        if v >= self.num_vertices {
            return None;
        }
        Some(self.out_neighbors[v].len())
    }

    fn edges(&self) -> Vec<(usize, usize)> {
        let mut result = Vec::new();
        for u in 0..self.num_vertices {
            for &v in &self.out_neighbors[u] {
                if self.directed || u <= v {
                    result.push((u, v));
                }
            }
        }
        result
    }

    fn allows_loops(&self) -> bool {
        self.loops
    }

    fn allows_multiple_edges(&self) -> bool {
        self.multiedges
    }

    fn get_edge_label(&self, u: usize, v: usize) -> Option<String> {
        self.edge_labels.get(&(u, v)).cloned()
    }

    fn set_edge_label(&mut self, u: usize, v: usize, label: Option<String>) -> Result<(), String> {
        if !self.has_edge(u, v) {
            return Err("Edge does not exist".to_string());
        }

        if let Some(lbl) = label {
            self.edge_labels.insert((u, v), lbl);
        } else {
            self.edge_labels.remove(&(u, v));
        }

        Ok(())
    }
}

/// Search iterator for graph traversal
///
/// Provides an iterator interface for BFS and DFS traversals.
/// Corresponds to SageMath's Search_iterator.
#[derive(Debug, Clone)]
pub struct SearchIterator {
    /// The graph being traversed
    graph: CGraphBackend,
    /// Traversal mode: "bfs" or "dfs"
    mode: String,
    /// Starting vertex
    start: usize,
    /// Current position in traversal
    visited: HashSet<usize>,
    /// Queue (for BFS) or stack (for DFS)
    queue: VecDeque<usize>,
}

impl SearchIterator {
    /// Create a new BFS iterator
    pub fn bfs(graph: CGraphBackend, start: usize) -> Result<Self, String> {
        if start >= graph.num_vertices() {
            return Err("Start vertex out of bounds".to_string());
        }

        let mut queue = VecDeque::new();
        queue.push_back(start);

        let mut visited = HashSet::new();
        visited.insert(start);

        Ok(SearchIterator {
            graph,
            mode: "bfs".to_string(),
            start,
            visited,
            queue,
        })
    }

    /// Create a new DFS iterator
    pub fn dfs(graph: CGraphBackend, start: usize) -> Result<Self, String> {
        if start >= graph.num_vertices() {
            return Err("Start vertex out of bounds".to_string());
        }

        let mut queue = VecDeque::new();
        queue.push_back(start);

        let mut visited = HashSet::new();
        visited.insert(start);

        Ok(SearchIterator {
            graph,
            mode: "dfs".to_string(),
            start,
            visited,
            queue,
        })
    }

    /// Get the next vertex in the traversal
    pub fn next(&mut self) -> Option<usize> {
        if let Some(current) = self.queue.pop_front() {
            // Add unvisited neighbors to queue/stack
            if let Some(neighbors) = self.graph.out_neighbors(current) {
                for &neighbor in &neighbors {
                    if !self.visited.contains(&neighbor) {
                        self.visited.insert(neighbor);
                        if self.mode == "bfs" {
                            self.queue.push_back(neighbor);
                        } else {
                            // DFS: add to front
                            self.queue.push_front(neighbor);
                        }
                    }
                }
            }
            Some(current)
        } else {
            None
        }
    }

    /// Collect all remaining vertices in the traversal
    pub fn collect_all(mut self) -> Vec<usize> {
        let mut result = Vec::new();
        while let Some(v) = self.next() {
            result.push(v);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cgraph_backend_creation() {
        let backend = CGraphBackend::new(false);
        assert!(!backend.is_directed());
        assert_eq!(backend.num_vertices(), 0);
        assert_eq!(backend.num_edges(), 0);
    }

    #[test]
    fn test_cgraph_backend_add_vertices() {
        let mut backend = CGraphBackend::new(false);
        let v0 = backend.add_vertex();
        let v1 = backend.add_vertex();

        assert_eq!(v0, 0);
        assert_eq!(v1, 1);
        assert_eq!(backend.num_vertices(), 2);
    }

    #[test]
    fn test_cgraph_backend_add_edges() {
        let mut backend = CGraphBackend::new(false);
        backend.add_vertex();
        backend.add_vertex();
        backend.add_vertex();

        backend.add_edge(0, 1, None, None).unwrap();
        backend.add_edge(1, 2, None, None).unwrap();

        assert_eq!(backend.num_edges(), 2);
        assert!(backend.has_edge(0, 1));
        assert!(backend.has_edge(1, 0)); // Undirected
        assert!(backend.has_edge(1, 2));
        assert!(!backend.has_edge(0, 2));
    }

    #[test]
    fn test_cgraph_backend_directed() {
        let mut backend = CGraphBackend::new(true);
        backend.add_vertex();
        backend.add_vertex();

        backend.add_edge(0, 1, None, None).unwrap();

        assert!(backend.has_edge(0, 1));
        assert!(!backend.has_edge(1, 0)); // Directed
    }

    #[test]
    fn test_cgraph_backend_degrees() {
        let mut backend = CGraphBackend::new(false);
        backend.add_vertex();
        backend.add_vertex();
        backend.add_vertex();

        backend.add_edge(0, 1, None, None).unwrap();
        backend.add_edge(1, 2, None, None).unwrap();

        assert_eq!(backend.out_degree(0), Some(1));
        assert_eq!(backend.out_degree(1), Some(2));
        assert_eq!(backend.out_degree(2), Some(1));
    }

    #[test]
    fn test_cgraph_backend_edge_labels() {
        let mut backend = CGraphBackend::new(false);
        backend.add_vertex();
        backend.add_vertex();

        backend.add_edge(0, 1, Some("edge01".to_string()), None).unwrap();

        assert_eq!(backend.get_edge_label(0, 1), Some("edge01".to_string()));

        backend.set_edge_label(0, 1, Some("new_label".to_string())).unwrap();
        assert_eq!(backend.get_edge_label(0, 1), Some("new_label".to_string()));
    }

    #[test]
    fn test_cgraph_backend_loops() {
        let mut backend = CGraphBackend::new_with_options(false, false, false);
        backend.add_vertex();

        // Loops not allowed
        assert!(backend.add_edge(0, 0, None, None).is_err());

        // Create backend with loops allowed
        let mut backend2 = CGraphBackend::new_with_options(false, true, false);
        backend2.add_vertex();

        assert!(backend2.add_edge(0, 0, None, None).is_ok());
        assert!(backend2.has_edge(0, 0));
    }

    #[test]
    fn test_search_iterator_bfs() {
        let mut backend = CGraphBackend::new(false);
        for _ in 0..4 {
            backend.add_vertex();
        }

        backend.add_edge(0, 1, None, None).unwrap();
        backend.add_edge(0, 2, None, None).unwrap();
        backend.add_edge(1, 3, None, None).unwrap();

        let iter = SearchIterator::bfs(backend, 0).unwrap();
        let order = iter.collect_all();

        assert_eq!(order.len(), 4);
        assert_eq!(order[0], 0); // Start vertex first

        // BFS should visit 0, then its neighbors (1, 2), then 3
        assert!(order.iter().position(|&x| x == 1).unwrap() < order.iter().position(|&x| x == 3).unwrap());
    }

    #[test]
    fn test_search_iterator_dfs() {
        let mut backend = CGraphBackend::new(false);
        for _ in 0..4 {
            backend.add_vertex();
        }

        backend.add_edge(0, 1, None, None).unwrap();
        backend.add_edge(0, 2, None, None).unwrap();
        backend.add_edge(1, 3, None, None).unwrap();

        let iter = SearchIterator::dfs(backend, 0).unwrap();
        let order = iter.collect_all();

        assert_eq!(order.len(), 4);
        assert_eq!(order[0], 0); // Start vertex first
    }

    #[test]
    fn test_in_neighbors_directed() {
        let mut backend = CGraphBackend::new(true);
        for _ in 0..3 {
            backend.add_vertex();
        }

        backend.add_edge(0, 1, None, None).unwrap();
        backend.add_edge(2, 1, None, None).unwrap();

        let in_neighbors = backend.in_neighbors(1).unwrap();
        assert_eq!(in_neighbors.len(), 2);
        assert!(in_neighbors.contains(&0));
        assert!(in_neighbors.contains(&2));

        let out_neighbors = backend.out_neighbors(1).unwrap();
        assert_eq!(out_neighbors.len(), 0);
    }
}
