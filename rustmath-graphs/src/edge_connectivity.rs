//! Edge connectivity algorithms for graphs
//!
//! This module implements algorithms for computing edge connectivity,
//! including Gabow's algorithm for directed graphs.

use crate::graph::Graph;
use crate::digraph::DiGraph;
use std::collections::VecDeque;

/// Gabow's algorithm for finding edge connectivity of digraphs
///
/// This implementation computes the edge connectivity (the minimum number of edges
/// whose removal disconnects the graph) using Gabow's Round Robin algorithm.
///
/// # References
///
/// H. N. Gabow. "Using expander graphs to find vertex connectivity."
/// Journal of the ACM (JACM) 53.5 (2006): 800-844.
pub struct GabowEdgeConnectivity {
    /// The directed graph to analyze
    graph: DiGraph,
    /// Number of vertices
    n: usize,
    /// Computed edge connectivity value
    connectivity: Option<usize>,
    /// Adjacency list representation
    adj_list: Vec<Vec<usize>>,
    /// Reverse adjacency list
    rev_adj_list: Vec<Vec<usize>>,
}

impl GabowEdgeConnectivity {
    /// Create a new GabowEdgeConnectivity instance
    ///
    /// # Arguments
    ///
    /// * `graph` - The directed graph to analyze
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_graphs::digraph::DiGraph;
    /// use rustmath_graphs::edge_connectivity::GabowEdgeConnectivity;
    ///
    /// let mut g = DiGraph::new(4);
    /// g.add_edge(0, 1);
    /// g.add_edge(1, 2);
    /// g.add_edge(2, 3);
    /// g.add_edge(3, 0);
    ///
    /// let gabow = GabowEdgeConnectivity::new(g);
    /// let ec = gabow.compute();
    /// assert_eq!(ec, 1);
    /// ```
    pub fn new(graph: DiGraph) -> Self {
        let n = graph.num_vertices();
        let adj_list = Self::build_adjacency_list(&graph);
        let rev_adj_list = Self::build_reverse_adjacency_list(&graph);

        Self {
            graph,
            n,
            connectivity: None,
            adj_list,
            rev_adj_list,
        }
    }

    /// Build adjacency list representation
    fn build_adjacency_list(graph: &DiGraph) -> Vec<Vec<usize>> {
        let n = graph.num_vertices();
        let mut adj = vec![Vec::new(); n];

        for (u, v) in graph.edges() {
            adj[u].push(v);
        }

        adj
    }

    /// Build reverse adjacency list representation
    fn build_reverse_adjacency_list(graph: &DiGraph) -> Vec<Vec<usize>> {
        let n = graph.num_vertices();
        let mut rev_adj = vec![Vec::new(); n];

        for (u, v) in graph.edges() {
            rev_adj[v].push(u);
        }

        rev_adj
    }

    /// Compute the edge connectivity of the digraph
    ///
    /// Returns the minimum number of edges whose removal disconnects the graph.
    ///
    /// # Returns
    ///
    /// The edge connectivity value
    pub fn compute(&mut self) -> usize {
        if let Some(ec) = self.connectivity {
            return ec;
        }

        if self.n <= 1 {
            self.connectivity = Some(0);
            return 0;
        }

        // Compute minimum in-degree and out-degree
        let min_degree = self.compute_min_degree();

        if min_degree == 0 {
            self.connectivity = Some(0);
            return 0;
        }

        // Use the Round Robin algorithm to compute exact edge connectivity
        let ec = self.round_robin_algorithm(min_degree);
        self.connectivity = Some(ec);
        ec
    }

    /// Get the edge connectivity value (must call compute first)
    pub fn edge_connectivity(&self) -> Option<usize> {
        self.connectivity
    }

    /// Compute minimum in-degree and out-degree
    fn compute_min_degree(&self) -> usize {
        let mut min_degree = usize::MAX;

        for v in 0..self.n {
            let in_deg = self.rev_adj_list[v].len();
            let out_deg = self.adj_list[v].len();

            min_degree = min_degree.min(in_deg).min(out_deg);
        }

        min_degree
    }

    /// Round Robin algorithm for computing edge connectivity
    ///
    /// This is the core algorithm that computes the exact edge connectivity
    /// by finding the minimum local edge connectivity between all pairs.
    fn round_robin_algorithm(&self, upper_bound: usize) -> usize {
        let mut min_connectivity = upper_bound;

        // Find minimum local edge connectivity over all pairs
        for s in 0..self.n {
            for t in 0..self.n {
                if s != t {
                    let local_ec = self.find_max_edge_disjoint_paths(s, t);
                    min_connectivity = min_connectivity.min(local_ec);

                    // Early termination if we find connectivity 1
                    if min_connectivity == 1 {
                        return 1;
                    }
                }
            }
        }

        min_connectivity
    }

    /// Find maximum number of edge-disjoint paths between s and t using max flow
    fn find_max_edge_disjoint_paths(&self, s: usize, t: usize) -> usize {
        // Use Ford-Fulkerson algorithm to find maximum flow
        let mut residual = self.build_residual_graph();
        let mut max_flow = 0;

        while let Some(path) = self.find_augmenting_path(s, t, &residual) {
            // Find minimum capacity along the path
            let mut min_cap = usize::MAX;
            for i in 0..path.len() - 1 {
                let u = path[i];
                let v = path[i + 1];
                min_cap = min_cap.min(residual[u][v]);
            }

            // Update residual graph
            for i in 0..path.len() - 1 {
                let u = path[i];
                let v = path[i + 1];
                residual[u][v] -= min_cap;
                residual[v][u] += min_cap;
            }

            max_flow += min_cap;
        }

        max_flow
    }

    /// Build residual graph for max flow computation
    fn build_residual_graph(&self) -> Vec<Vec<usize>> {
        let mut residual = vec![vec![0; self.n]; self.n];

        for u in 0..self.n {
            for &v in &self.adj_list[u] {
                residual[u][v] = 1; // Unit capacity for edge-disjoint paths
            }
        }

        residual
    }

    /// Find an augmenting path using BFS
    fn find_augmenting_path(&self, s: usize, t: usize, residual: &[Vec<usize>]) -> Option<Vec<usize>> {
        let mut visited = vec![false; self.n];
        let mut parent = vec![None; self.n];
        let mut queue = VecDeque::new();

        queue.push_back(s);
        visited[s] = true;

        while let Some(u) = queue.pop_front() {
            if u == t {
                // Reconstruct path
                let mut path = Vec::new();
                let mut current = t;
                path.push(current);

                while let Some(p) = parent[current] {
                    path.push(p);
                    current = p;
                }

                path.reverse();
                return Some(path);
            }

            for v in 0..self.n {
                if !visited[v] && residual[u][v] > 0 {
                    visited[v] = true;
                    parent[v] = Some(u);
                    queue.push_back(v);
                }
            }
        }

        None
    }

    /// Compute k edge-disjoint spanning trees (not yet fully implemented)
    ///
    /// This method would return k edge-disjoint spanning trees if the graph
    /// is k-edge-connected.
    pub fn edge_disjoint_spanning_trees(&self, k: usize) -> Result<Vec<Vec<(usize, usize)>>, String> {
        if let Some(ec) = self.connectivity {
            if k > ec {
                return Err(format!("Graph is only {}-edge-connected, cannot find {} edge-disjoint spanning trees", ec, k));
            }
        } else {
            return Err("Must call compute() first".to_string());
        }

        // This is a placeholder for the full implementation
        // The full algorithm would construct k edge-disjoint spanning trees
        Err("Edge-disjoint spanning tree construction not yet implemented".to_string())
    }
}

/// Compute edge connectivity of an undirected graph
///
/// For undirected graphs, this is simpler than the directed case.
/// It returns the minimum number of edges whose removal disconnects the graph.
///
/// # Arguments
///
/// * `graph` - The undirected graph
///
/// # Returns
///
/// The edge connectivity value
///
/// # Examples
///
/// ```
/// use rustmath_graphs::Graph;
/// use rustmath_graphs::edge_connectivity::edge_connectivity;
///
/// let mut g = Graph::new(4);
/// g.add_edge(0, 1);
/// g.add_edge(1, 2);
/// g.add_edge(2, 3);
/// g.add_edge(3, 0);
/// g.add_edge(0, 2);
///
/// let ec = edge_connectivity(&g);
/// assert_eq!(ec, 2);
/// ```
pub fn edge_connectivity(graph: &Graph) -> usize {
    let n = graph.num_vertices();

    if n <= 1 {
        return 0;
    }

    // Compute minimum degree as upper bound
    let mut min_degree = usize::MAX;
    for v in 0..n {
        if let Some(deg) = graph.degree(v) {
            min_degree = min_degree.min(deg);
        }
    }

    if min_degree == 0 {
        return 0;
    }

    // Use max flow to compute exact edge connectivity
    // We check between vertex 0 and all other vertices
    let mut min_cut = min_degree;

    for t in 1..n {
        let cut = compute_min_cut(graph, 0, t);
        min_cut = min_cut.min(cut);
    }

    min_cut
}

/// Compute minimum cut between two vertices using max flow
fn compute_min_cut(graph: &Graph, s: usize, t: usize) -> usize {
    let n = graph.num_vertices();

    // Build capacity matrix
    let mut capacity = vec![vec![0; n]; n];
    for u in 0..n {
        if let Some(neighbors) = graph.neighbors(u) {
            for v in neighbors {
                if u < v {
                    capacity[u][v] = 1;
                    capacity[v][u] = 1;
                }
            }
        }
    }

    // Ford-Fulkerson algorithm
    let mut residual = capacity.clone();
    let mut max_flow = 0;

    while let Some(path) = find_path_bfs(s, t, &residual, n) {
        // Find minimum capacity along path
        let mut min_cap = usize::MAX;
        for i in 0..path.len() - 1 {
            let u = path[i];
            let v = path[i + 1];
            min_cap = min_cap.min(residual[u][v]);
        }

        // Update residual graph
        for i in 0..path.len() - 1 {
            let u = path[i];
            let v = path[i + 1];
            residual[u][v] -= min_cap;
            residual[v][u] += min_cap;
        }

        max_flow += min_cap;
    }

    max_flow
}

/// Find a path using BFS
fn find_path_bfs(s: usize, t: usize, residual: &[Vec<usize>], n: usize) -> Option<Vec<usize>> {
    let mut visited = vec![false; n];
    let mut parent = vec![None; n];
    let mut queue = VecDeque::new();

    queue.push_back(s);
    visited[s] = true;

    while let Some(u) = queue.pop_front() {
        if u == t {
            // Reconstruct path
            let mut path = Vec::new();
            let mut current = t;
            path.push(current);

            while let Some(p) = parent[current] {
                path.push(p);
                current = p;
            }

            path.reverse();
            return Some(path);
        }

        for v in 0..n {
            if !visited[v] && residual[u][v] > 0 {
                visited[v] = true;
                parent[v] = Some(u);
                queue.push_back(v);
            }
        }
    }

    None
}

/// Compute local edge connectivity between two vertices
///
/// Returns the maximum number of edge-disjoint paths between u and v.
///
/// # Arguments
///
/// * `graph` - The graph
/// * `u` - Source vertex
/// * `v` - Target vertex
///
/// # Examples
///
/// ```
/// use rustmath_graphs::Graph;
/// use rustmath_graphs::edge_connectivity::local_edge_connectivity;
///
/// let mut g = Graph::new(4);
/// g.add_edge(0, 1);
/// g.add_edge(0, 2);
/// g.add_edge(1, 3);
/// g.add_edge(2, 3);
///
/// let lec = local_edge_connectivity(&g, 0, 3);
/// assert_eq!(lec, 2);
/// ```
pub fn local_edge_connectivity(graph: &Graph, u: usize, v: usize) -> usize {
    if u == v {
        return 0;
    }

    compute_min_cut(graph, u, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_connectivity_cycle() {
        let mut g = Graph::new(5);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();
        g.add_edge(3, 4).unwrap();
        g.add_edge(4, 0).unwrap();

        let ec = edge_connectivity(&g);
        assert_eq!(ec, 2);
    }

    #[test]
    fn test_edge_connectivity_complete() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 2).unwrap();
        g.add_edge(0, 3).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(1, 3).unwrap();
        g.add_edge(2, 3).unwrap();

        let ec = edge_connectivity(&g);
        assert_eq!(ec, 3); // K4 is 3-edge-connected
    }

    #[test]
    fn test_edge_connectivity_path() {
        let mut g = Graph::new(5);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();
        g.add_edge(3, 4).unwrap();

        let ec = edge_connectivity(&g);
        assert_eq!(ec, 1); // Path graph has edge connectivity 1
    }

    #[test]
    fn test_edge_connectivity_disconnected() {
        let mut g = Graph::new(6);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(3, 4).unwrap();
        g.add_edge(4, 5).unwrap();

        let ec = edge_connectivity(&g);
        assert_eq!(ec, 0); // Disconnected graph
    }

    #[test]
    fn test_local_edge_connectivity() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 2).unwrap();
        g.add_edge(1, 3).unwrap();
        g.add_edge(2, 3).unwrap();

        let lec = local_edge_connectivity(&g, 0, 3);
        assert_eq!(lec, 2);
    }

    #[test]
    fn test_gabow_digraph() {
        // Create a digraph with multiple paths
        // This forms a cycle with two shortcuts
        let mut g = DiGraph::new(4);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        g.add_edge(3, 0);
        g.add_edge(0, 2);
        g.add_edge(1, 3);

        let mut gabow = GabowEdgeConnectivity::new(g);
        let ec = gabow.compute();
        // Edge connectivity is 1 because vertex 3 can only reach vertex 1 via 3→0→1
        assert_eq!(ec, 1);
    }

    #[test]
    fn test_gabow_digraph_higher_connectivity() {
        // Create a more connected digraph
        let mut g = DiGraph::new(4);
        // Complete directed graph (all edges)
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    g.add_edge(i, j);
                }
            }
        }

        let mut gabow = GabowEdgeConnectivity::new(g);
        let ec = gabow.compute();
        assert_eq!(ec, 3); // Minimum in-degree is 3
    }

    #[test]
    fn test_gabow_cycle_digraph() {
        let mut g = DiGraph::new(5);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        g.add_edge(3, 4);
        g.add_edge(4, 0);

        let mut gabow = GabowEdgeConnectivity::new(g);
        let ec = gabow.compute();
        assert_eq!(ec, 1);
    }

    #[test]
    fn test_gabow_empty_graph() {
        let g = DiGraph::new(3);

        let mut gabow = GabowEdgeConnectivity::new(g);
        let ec = gabow.compute();
        assert_eq!(ec, 0);
    }

    #[test]
    fn test_edge_connectivity_bridge() {
        let mut g = Graph::new(6);
        // Two triangles connected by a bridge
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 0).unwrap();
        g.add_edge(2, 3).unwrap(); // Bridge
        g.add_edge(3, 4).unwrap();
        g.add_edge(4, 5).unwrap();
        g.add_edge(5, 3).unwrap();

        let ec = edge_connectivity(&g);
        assert_eq!(ec, 1); // Bridge limits connectivity
    }
}
