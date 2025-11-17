//! Quiver data structure - directed graph with labeled edges

use std::collections::HashMap;

/// An edge in a quiver: (source, target, label)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Edge {
    pub source: usize,
    pub target: usize,
    pub label: String,
}

impl Edge {
    /// Create a new edge
    pub fn new(source: usize, target: usize, label: impl Into<String>) -> Self {
        Edge {
            source,
            target,
            label: label.into(),
        }
    }
}

/// A quiver (directed graph with labeled edges)
///
/// Vertices are labeled by integers from 0 to n-1.
/// Edges are labeled by unique strings that:
/// - Must be non-empty
/// - Cannot start with "e_" (reserved for idempotents)
/// - Cannot contain "*" (reserved for path concatenation)
#[derive(Debug, Clone)]
pub struct Quiver {
    /// Number of vertices
    num_vertices: usize,
    /// All edges in the quiver
    edges: Vec<Edge>,
    /// Map from edge label to edge index
    label_to_index: HashMap<String, usize>,
    /// Adjacency lists: vertex -> outgoing edges (as indices into edges vec)
    outgoing: Vec<Vec<usize>>,
    /// Adjacency lists: vertex -> incoming edges (as indices into edges vec)
    incoming: Vec<Vec<usize>>,
}

impl Quiver {
    /// Create a new quiver with n vertices
    pub fn new(n: usize) -> Self {
        Quiver {
            num_vertices: n,
            edges: Vec::new(),
            label_to_index: HashMap::new(),
            outgoing: vec![Vec::new(); n],
            incoming: vec![Vec::new(); n],
        }
    }

    /// Get the number of vertices
    pub fn num_vertices(&self) -> usize {
        self.num_vertices
    }

    /// Get the number of edges
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Add a labeled edge from source to target
    ///
    /// Returns an error if:
    /// - Vertices are out of bounds
    /// - Label is empty
    /// - Label starts with "e_"
    /// - Label contains "*"
    /// - Label already exists
    pub fn add_edge(&mut self, source: usize, target: usize, label: impl Into<String>) -> Result<(), String> {
        if source >= self.num_vertices {
            return Err(format!("Source vertex {} out of bounds", source));
        }
        if target >= self.num_vertices {
            return Err(format!("Target vertex {} out of bounds", target));
        }

        let label = label.into();

        // Validate label
        if label.is_empty() {
            return Err("Edge label cannot be empty".to_string());
        }
        if label.starts_with("e_") {
            return Err("Edge label cannot start with 'e_' (reserved for idempotents)".to_string());
        }
        if label.contains('*') {
            return Err("Edge label cannot contain '*' (reserved for path concatenation)".to_string());
        }
        if self.label_to_index.contains_key(&label) {
            return Err(format!("Edge label '{}' already exists", label));
        }

        let edge_index = self.edges.len();
        let edge = Edge::new(source, target, label.clone());

        self.edges.push(edge);
        self.label_to_index.insert(label, edge_index);
        self.outgoing[source].push(edge_index);
        self.incoming[target].push(edge_index);

        Ok(())
    }

    /// Get an edge by its label
    pub fn get_edge(&self, label: &str) -> Option<&Edge> {
        self.label_to_index.get(label).map(|&idx| &self.edges[idx])
    }

    /// Get all edges
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// Get all edge labels sorted
    pub fn labels(&self) -> Vec<String> {
        let mut labels: Vec<String> = self.label_to_index.keys().cloned().collect();
        labels.sort();
        labels
    }

    /// Get all vertices
    pub fn vertices(&self) -> Vec<usize> {
        (0..self.num_vertices).collect()
    }

    /// Get outgoing edges from a vertex (as edge indices)
    pub fn outgoing_edges(&self, vertex: usize) -> Option<&[usize]> {
        if vertex >= self.num_vertices {
            return None;
        }
        Some(&self.outgoing[vertex])
    }

    /// Get incoming edges to a vertex (as edge indices)
    pub fn incoming_edges(&self, vertex: usize) -> Option<&[usize]> {
        if vertex >= self.num_vertices {
            return None;
        }
        Some(&self.incoming[vertex])
    }

    /// Get the out-degree of a vertex
    pub fn out_degree(&self, vertex: usize) -> Option<usize> {
        if vertex >= self.num_vertices {
            return None;
        }
        Some(self.outgoing[vertex].len())
    }

    /// Get the in-degree of a vertex
    pub fn in_degree(&self, vertex: usize) -> Option<usize> {
        if vertex >= self.num_vertices {
            return None;
        }
        Some(self.incoming[vertex].len())
    }

    /// Check if there's an edge with the given label
    pub fn has_edge_label(&self, label: &str) -> bool {
        self.label_to_index.contains_key(label)
    }

    /// Check if the quiver is acyclic (a DAG)
    ///
    /// Uses topological sort to detect cycles
    pub fn is_acyclic(&self) -> bool {
        let mut in_degree = vec![0; self.num_vertices];

        // Compute in-degrees
        for edge in &self.edges {
            in_degree[edge.target] += 1;
        }

        // Queue of vertices with in-degree 0
        let mut queue: Vec<usize> = in_degree
            .iter()
            .enumerate()
            .filter(|(_, &deg)| deg == 0)
            .map(|(v, _)| v)
            .collect();

        let mut processed = 0;

        while let Some(v) = queue.pop() {
            processed += 1;

            for &edge_idx in &self.outgoing[v] {
                let target = self.edges[edge_idx].target;
                in_degree[target] -= 1;
                if in_degree[target] == 0 {
                    queue.push(target);
                }
            }
        }

        processed == self.num_vertices
    }

    /// Check if the quiver has a cycle
    pub fn has_cycle(&self) -> bool {
        !self.is_acyclic()
    }

    /// Find all paths from start to end vertex (only works for acyclic quivers)
    ///
    /// Returns None if the quiver has a cycle
    pub fn all_paths(&self, start: usize, end: usize) -> Option<Vec<Vec<usize>>> {
        if start >= self.num_vertices || end >= self.num_vertices {
            return Some(Vec::new());
        }

        if !self.is_acyclic() {
            return None; // Cannot enumerate paths in cyclic quivers
        }

        let mut all_paths = Vec::new();
        let mut current_path = Vec::new();
        self.dfs_paths(start, end, &mut current_path, &mut all_paths);

        Some(all_paths)
    }

    /// DFS helper to find all paths
    fn dfs_paths(&self, current: usize, end: usize, path: &mut Vec<usize>, all_paths: &mut Vec<Vec<usize>>) {
        if current == end {
            all_paths.push(path.clone());
            return;
        }

        for &edge_idx in &self.outgoing[current] {
            let edge = &self.edges[edge_idx];
            path.push(edge_idx);
            self.dfs_paths(edge.target, end, path, all_paths);
            path.pop();
        }
    }

    /// Get the reverse (opposite) quiver
    ///
    /// Each edge (u, v, "label") becomes (v, u, "label")
    pub fn reverse(&self) -> Quiver {
        let mut reversed = Quiver::new(self.num_vertices);

        for edge in &self.edges {
            // For reversed edges, we could add a prefix or suffix to avoid label conflicts
            let rev_label = format!("rev_{}", edge.label);
            reversed.add_edge(edge.target, edge.source, rev_label).unwrap();
        }

        reversed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quiver_creation() {
        let q = Quiver::new(5);
        assert_eq!(q.num_vertices(), 5);
        assert_eq!(q.num_edges(), 0);
    }

    #[test]
    fn test_add_edge() {
        let mut q = Quiver::new(3);
        assert!(q.add_edge(0, 1, "a").is_ok());
        assert!(q.add_edge(1, 2, "b").is_ok());
        assert_eq!(q.num_edges(), 2);

        let edge = q.get_edge("a");
        assert!(edge.is_some());
        let e = edge.unwrap();
        assert_eq!(e.source, 0);
        assert_eq!(e.target, 1);
        assert_eq!(e.label, "a");
    }

    #[test]
    fn test_duplicate_label() {
        let mut q = Quiver::new(3);
        assert!(q.add_edge(0, 1, "a").is_ok());
        assert!(q.add_edge(1, 2, "a").is_err()); // Duplicate label
    }

    #[test]
    fn test_invalid_labels() {
        let mut q = Quiver::new(3);

        // Empty label
        assert!(q.add_edge(0, 1, "").is_err());

        // Label starting with "e_"
        assert!(q.add_edge(0, 1, "e_bad").is_err());

        // Label containing "*"
        assert!(q.add_edge(0, 1, "a*b").is_err());
    }

    #[test]
    fn test_out_of_bounds() {
        let mut q = Quiver::new(3);
        assert!(q.add_edge(0, 5, "a").is_err());
        assert!(q.add_edge(5, 0, "b").is_err());
    }

    #[test]
    fn test_degrees() {
        let mut q = Quiver::new(4);
        q.add_edge(0, 1, "a").unwrap();
        q.add_edge(0, 2, "b").unwrap();
        q.add_edge(1, 2, "c").unwrap();
        q.add_edge(1, 3, "d").unwrap();

        assert_eq!(q.out_degree(0), Some(2));
        assert_eq!(q.out_degree(1), Some(2));
        assert_eq!(q.out_degree(2), Some(0));
        assert_eq!(q.out_degree(3), Some(0));

        assert_eq!(q.in_degree(0), Some(0));
        assert_eq!(q.in_degree(1), Some(1));
        assert_eq!(q.in_degree(2), Some(2));
        assert_eq!(q.in_degree(3), Some(1));
    }

    #[test]
    fn test_acyclic() {
        // Acyclic quiver: 0 -> 1 -> 2
        let mut q1 = Quiver::new(3);
        q1.add_edge(0, 1, "a").unwrap();
        q1.add_edge(1, 2, "b").unwrap();
        assert!(q1.is_acyclic());
        assert!(!q1.has_cycle());

        // Cyclic quiver: 0 -> 1 -> 2 -> 0
        let mut q2 = Quiver::new(3);
        q2.add_edge(0, 1, "a").unwrap();
        q2.add_edge(1, 2, "b").unwrap();
        q2.add_edge(2, 0, "c").unwrap();
        assert!(!q2.is_acyclic());
        assert!(q2.has_cycle());
    }

    #[test]
    fn test_all_paths_acyclic() {
        // Diamond graph: 0 -> {1, 2} -> 3
        let mut q = Quiver::new(4);
        q.add_edge(0, 1, "a").unwrap();
        q.add_edge(0, 2, "b").unwrap();
        q.add_edge(1, 3, "c").unwrap();
        q.add_edge(2, 3, "d").unwrap();

        let paths = q.all_paths(0, 3);
        assert!(paths.is_some());
        let p = paths.unwrap();
        assert_eq!(p.len(), 2); // Two paths: [a,c] and [b,d]
    }

    #[test]
    fn test_all_paths_cyclic() {
        // Cyclic quiver
        let mut q = Quiver::new(3);
        q.add_edge(0, 1, "a").unwrap();
        q.add_edge(1, 2, "b").unwrap();
        q.add_edge(2, 0, "c").unwrap();

        let paths = q.all_paths(0, 2);
        assert!(paths.is_none()); // Cannot enumerate paths in cyclic quivers
    }

    #[test]
    fn test_labels_sorted() {
        let mut q = Quiver::new(3);
        q.add_edge(0, 1, "z").unwrap();
        q.add_edge(1, 2, "a").unwrap();
        q.add_edge(0, 2, "m").unwrap();

        let labels = q.labels();
        assert_eq!(labels, vec!["a", "m", "z"]);
    }

    #[test]
    fn test_reverse_quiver() {
        let mut q = Quiver::new(3);
        q.add_edge(0, 1, "a").unwrap();
        q.add_edge(1, 2, "b").unwrap();

        let rev = q.reverse();
        assert_eq!(rev.num_vertices(), 3);
        assert_eq!(rev.num_edges(), 2);

        let edge = rev.get_edge("rev_a");
        assert!(edge.is_some());
        let e = edge.unwrap();
        assert_eq!(e.source, 1);
        assert_eq!(e.target, 0);
    }

    #[test]
    fn test_outgoing_incoming_edges() {
        let mut q = Quiver::new(3);
        q.add_edge(0, 1, "a").unwrap();
        q.add_edge(0, 2, "b").unwrap();
        q.add_edge(1, 2, "c").unwrap();

        let out0 = q.outgoing_edges(0);
        assert!(out0.is_some());
        assert_eq!(out0.unwrap().len(), 2);

        let in2 = q.incoming_edges(2);
        assert!(in2.is_some());
        assert_eq!(in2.unwrap().len(), 2);
    }
}
