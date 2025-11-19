//! Tutte polynomial computation for graphs
//!
//! The Tutte polynomial is a fundamental graph invariant that encodes
//! many properties of a graph including its chromatic polynomial,
//! flow polynomial, and reliability polynomial.
//!
//! For a graph G, the Tutte polynomial T(x,y) can be computed using
//! the deletion-contraction recurrence.

use crate::graph::Graph;
use std::collections::HashMap;

/// Edge selection strategy for Tutte polynomial computation
///
/// Different strategies affect the efficiency of the deletion-contraction
/// algorithm for computing the Tutte polynomial.
pub trait EdgeSelection {
    /// Select the next edge to process
    ///
    /// # Arguments
    ///
    /// * `graph` - The graph to select an edge from
    ///
    /// # Returns
    ///
    /// An optional edge (u, v) to process next
    fn select_edge(&self, graph: &Graph) -> Option<(usize, usize)>;
}

/// Vertex order-based edge selection
///
/// Selects edges based on a fixed vertex ordering
pub struct VertexOrder {
    order: Vec<usize>,
}

impl VertexOrder {
    /// Create a new vertex order strategy
    pub fn new(order: Vec<usize>) -> Self {
        VertexOrder { order }
    }
}

impl EdgeSelection for VertexOrder {
    fn select_edge(&self, graph: &Graph) -> Option<(usize, usize)> {
        for &u in &self.order {
            if let Some(neighbors) = graph.neighbors(u) {
                for &v in &neighbors {
                    if u < v {
                        return Some((u, v));
                    }
                }
            }
        }
        None
    }
}

/// Minimize single degree edge selection
///
/// Selects edges incident to vertices of minimum degree
pub struct MinimizeSingleDegree;

impl EdgeSelection for MinimizeSingleDegree {
    fn select_edge(&self, graph: &Graph) -> Option<(usize, usize)> {
        let n = graph.num_vertices();
        let mut min_degree = usize::MAX;
        let mut min_vertex = None;

        for v in 0..n {
            if let Some(neighbors) = graph.neighbors(v) {
                let degree = neighbors.len();
                if degree > 0 && degree < min_degree {
                    min_degree = degree;
                    min_vertex = Some(v);
                }
            }
        }

        if let Some(v) = min_vertex {
            if let Some(neighbors) = graph.neighbors(v) {
                if !neighbors.is_empty() {
                    return Some((v, neighbors[0]));
                }
            }
        }

        None
    }
}

/// Minimize degree edge selection
///
/// Selects edges between vertices with minimum total degree
pub struct MinimizeDegree;

impl EdgeSelection for MinimizeDegree {
    fn select_edge(&self, graph: &Graph) -> Option<(usize, usize)> {
        let n = graph.num_vertices();
        let mut min_sum = usize::MAX;
        let mut best_edge = None;

        for u in 0..n {
            if let Some(neighbors_u) = graph.neighbors(u) {
                let deg_u = neighbors_u.len();
                for &v in &neighbors_u {
                    if u < v {
                        let deg_v = graph.neighbors(v).map(|n| n.len()).unwrap_or(0);
                        let sum = deg_u + deg_v;
                        if sum < min_sum {
                            min_sum = sum;
                            best_edge = Some((u, v));
                        }
                    }
                }
            }
        }

        best_edge
    }
}

/// Maximize degree edge selection
///
/// Selects edges between vertices with maximum total degree
pub struct MaximizeDegree;

impl EdgeSelection for MaximizeDegree {
    fn select_edge(&self, graph: &Graph) -> Option<(usize, usize)> {
        let n = graph.num_vertices();
        let mut max_sum = 0;
        let mut best_edge = None;

        for u in 0..n {
            if let Some(neighbors_u) = graph.neighbors(u) {
                let deg_u = neighbors_u.len();
                for &v in &neighbors_u {
                    if u < v {
                        let deg_v = graph.neighbors(v).map(|n| n.len()).unwrap_or(0);
                        let sum = deg_u + deg_v;
                        if sum > max_sum {
                            max_sum = sum;
                            best_edge = Some((u, v));
                        }
                    }
                }
            }
        }

        best_edge
    }
}

/// An ear in an ear decomposition
///
/// Used in Tutte polynomial computation
pub struct Ear {
    pub path: Vec<usize>,
}

impl Ear {
    /// Create a new ear
    pub fn new(path: Vec<usize>) -> Self {
        Ear { path }
    }
}

/// Remove an edge from a graph
///
/// Creates a new graph with the specified edge removed
///
/// # Arguments
///
/// * `graph` - The original graph
/// * `u` - First vertex of edge
/// * `v` - Second vertex of edge
///
/// # Returns
///
/// A new graph with the edge removed
pub fn removed_edge(graph: &Graph, u: usize, v: usize) -> Graph {
    let n = graph.num_vertices();
    let mut result = Graph::new(n);

    for i in 0..n {
        if let Some(neighbors) = graph.neighbors(i) {
            for &j in &neighbors {
                if i < j && !((i == u && j == v) || (i == v && j == u)) {
                    result.add_edge(i, j).ok();
                }
            }
        }
    }

    result
}

/// Contract an edge in a graph
///
/// Creates a new graph with the specified edge contracted.
/// Contracting an edge merges its two endpoints into a single vertex.
///
/// # Arguments
///
/// * `graph` - The original graph
/// * `u` - First vertex of edge
/// * `v` - Second vertex of edge
///
/// # Returns
///
/// A new graph with the edge contracted
pub fn contracted_edge(graph: &Graph, u: usize, v: usize) -> Graph {
    let n = graph.num_vertices();
    let mut result = Graph::new(n - 1);

    // Map old vertices to new vertices (merge u and v)
    let mut vertex_map = vec![0; n];
    let mut new_idx = 0;
    for i in 0..n {
        if i == v {
            vertex_map[i] = vertex_map[u];
        } else {
            vertex_map[i] = new_idx;
            new_idx += 1;
        }
    }

    // Add edges to contracted graph
    for i in 0..n {
        if let Some(neighbors) = graph.neighbors(i) {
            for &j in &neighbors {
                if i < j {
                    let new_i = vertex_map[i];
                    let new_j = vertex_map[j];
                    // Skip self-loops
                    if new_i != new_j {
                        result.add_edge(new_i, new_j).ok();
                    }
                }
            }
        }
    }

    result
}

/// Remove all loop edges from a graph
///
/// # Arguments
///
/// * `graph` - The graph to process
///
/// # Returns
///
/// A new graph with all loops removed
pub fn removed_loops(graph: &Graph) -> Graph {
    // Simple graphs don't have loops, so just return a copy
    let n = graph.num_vertices();
    let mut result = Graph::new(n);

    for i in 0..n {
        if let Some(neighbors) = graph.neighbors(i) {
            for &j in &neighbors {
                if i < j {
                    result.add_edge(i, j).ok();
                }
            }
        }
    }

    result
}

/// Remove one instance of a multi-edge
///
/// For simple graphs, this is the same as removing an edge
///
/// # Arguments
///
/// * `graph` - The graph
/// * `u` - First vertex
/// * `v` - Second vertex
///
/// # Returns
///
/// Graph with one instance of the edge removed
pub fn removed_multiedge(graph: &Graph, u: usize, v: usize) -> Graph {
    removed_edge(graph, u, v)
}

/// Get edge multiplicities in a graph
///
/// For simple graphs, all edges have multiplicity 1
///
/// # Arguments
///
/// * `graph` - The graph
///
/// # Returns
///
/// A map from edges to their multiplicities
pub fn edge_multiplicities(graph: &Graph) -> HashMap<(usize, usize), usize> {
    let mut result = HashMap::new();
    let n = graph.num_vertices();

    for i in 0..n {
        if let Some(neighbors) = graph.neighbors(i) {
            for &j in &neighbors {
                if i < j {
                    *result.entry((i, j)).or_insert(0) += 1;
                }
            }
        }
    }

    result
}

/// Get the underlying simple graph
///
/// Removes loops and reduces multi-edges to simple edges
///
/// # Arguments
///
/// * `graph` - The graph
///
/// # Returns
///
/// The underlying simple graph
pub fn underlying_graph(graph: &Graph) -> Graph {
    removed_loops(graph)
}

/// Compute the Tutte polynomial of a graph
///
/// The Tutte polynomial T(x,y) is computed using the deletion-contraction
/// recurrence:
/// - T(G) = T(G-e) + T(G/e) for non-loop, non-bridge edge e
/// - T(G) = x * T(G/e) for bridge e
/// - T(G) = y * T(G-e) for loop e
/// - T(empty graph) = 1
///
/// # Arguments
///
/// * `graph` - The graph to compute the polynomial for
///
/// # Returns
///
/// A tuple (coefficients, x_powers, y_powers) representing the polynomial
/// This is a simplified representation - a full implementation would return
/// a polynomial object
///
/// # Examples
///
/// ```
/// use rustmath_graphs::{Graph, tutte_polynomial::tutte_polynomial};
///
/// let mut g = Graph::new(3);
/// g.add_edge(0, 1).unwrap();
/// g.add_edge(1, 2).unwrap();
///
/// let poly = tutte_polynomial(&g);
/// // For a path graph, the Tutte polynomial has specific structure
/// ```
pub fn tutte_polynomial(graph: &Graph) -> Vec<(i64, usize, usize)> {
    // Base case: empty graph (no edges)
    if graph.num_edges() == 0 {
        return vec![(1, 0, 0)]; // T(x,y) = 1
    }

    // Find an edge to process
    let edge = {
        let n = graph.num_vertices();
        let mut found = None;
        for i in 0..n {
            if let Some(neighbors) = graph.neighbors(i) {
                for &j in &neighbors {
                    if i < j {
                        found = Some((i, j));
                        break;
                    }
                }
                if found.is_some() {
                    break;
                }
            }
        }
        found
    };

    if let Some((u, v)) = edge {
        // Check if edge is a bridge
        let is_bridge = {
            let temp = removed_edge(graph, u, v);
            !is_connected_component(&temp, u, v)
        };

        if is_bridge {
            // T(G) = x * T(G/e)
            let contracted = contracted_edge(graph, u, v);
            let sub_poly = tutte_polynomial(&contracted);
            // Multiply by x
            sub_poly.iter().map(|&(coef, xp, yp)| (coef, xp + 1, yp)).collect()
        } else {
            // T(G) = T(G-e) + T(G/e)
            let deleted = removed_edge(graph, u, v);
            let contracted = contracted_edge(graph, u, v);

            let poly1 = tutte_polynomial(&deleted);
            let poly2 = tutte_polynomial(&contracted);

            // Add the polynomials
            add_polynomials(&poly1, &poly2)
        }
    } else {
        vec![(1, 0, 0)]
    }
}

// Helper: Check if two vertices are in the same connected component
fn is_connected_component(graph: &Graph, u: usize, v: usize) -> bool {
    let n = graph.num_vertices();
    let mut visited = vec![false; n];
    let mut stack = vec![u];

    while let Some(current) = stack.pop() {
        if current == v {
            return true;
        }

        if !visited[current] {
            visited[current] = true;

            if let Some(neighbors) = graph.neighbors(current) {
                for &neighbor in &neighbors {
                    if !visited[neighbor] {
                        stack.push(neighbor);
                    }
                }
            }
        }
    }

    false
}

// Helper: Add two polynomials represented as lists of (coef, x_power, y_power)
fn add_polynomials(
    p1: &[(i64, usize, usize)],
    p2: &[(i64, usize, usize)]
) -> Vec<(i64, usize, usize)> {
    let mut result = HashMap::new();

    for &(coef, xp, yp) in p1 {
        *result.entry((xp, yp)).or_insert(0) += coef;
    }

    for &(coef, xp, yp) in p2 {
        *result.entry((xp, yp)).or_insert(0) += coef;
    }

    result.into_iter()
        .filter(|(_, coef)| *coef != 0)
        .map(|((xp, yp), coef)| (coef, xp, yp))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_removed_edge() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(0, 2).unwrap();

        let h = removed_edge(&g, 1, 2);
        assert_eq!(h.num_edges(), 2);
        assert!(h.has_edge(0, 1));
        assert!(h.has_edge(0, 2));
        assert!(!h.has_edge(1, 2));
    }

    #[test]
    fn test_contracted_edge() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        let h = contracted_edge(&g, 0, 1);
        assert_eq!(h.num_vertices(), 2);
    }

    #[test]
    fn test_removed_loops() {
        let g = Graph::new(3);
        let h = removed_loops(&g);
        assert_eq!(h.num_vertices(), 3);
    }

    #[test]
    fn test_edge_multiplicities() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        let mult = edge_multiplicities(&g);
        assert_eq!(mult.len(), 2);
    }

    #[test]
    fn test_tutte_polynomial_empty() {
        let g = Graph::new(3);
        let poly = tutte_polynomial(&g);
        assert_eq!(poly, vec![(1, 0, 0)]);
    }

    #[test]
    fn test_tutte_polynomial_single_edge() {
        let mut g = Graph::new(2);
        g.add_edge(0, 1).unwrap();

        let poly = tutte_polynomial(&g);
        // Single edge is a bridge: T(x,y) = x
        assert!(poly.contains(&(1, 1, 0)));
    }

    #[test]
    fn test_edge_selection_minimize_single() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        let selector = MinimizeSingleDegree;
        let edge = selector.select_edge(&g);
        assert!(edge.is_some());
    }

    #[test]
    fn test_edge_selection_minimize_degree() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        let selector = MinimizeDegree;
        let edge = selector.select_edge(&g);
        assert!(edge.is_some());
    }

    #[test]
    fn test_edge_selection_maximize_degree() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 2).unwrap();
        g.add_edge(0, 3).unwrap();
        g.add_edge(1, 2).unwrap();

        let selector = MaximizeDegree;
        let edge = selector.select_edge(&g);
        assert!(edge.is_some());
    }

    #[test]
    fn test_vertex_order_selection() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        let selector = VertexOrder::new(vec![0, 1, 2]);
        let edge = selector.select_edge(&g);
        assert!(edge.is_some());
    }

    #[test]
    fn test_ear_creation() {
        let ear = Ear::new(vec![0, 1, 2]);
        assert_eq!(ear.path.len(), 3);
    }

    #[test]
    fn test_underlying_graph() {
        let g = Graph::new(3);
        let h = underlying_graph(&g);
        assert_eq!(h.num_vertices(), 3);
    }

    #[test]
    fn test_is_connected_component() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        assert!(is_connected_component(&g, 0, 2));
        assert!(!is_connected_component(&g, 0, 2 + 1000)); // Non-existent vertex
    }
}
