//! Dynkin Diagrams
//!
//! A Dynkin diagram is a graphical representation of a Cartan matrix / root system.
//! Nodes represent simple roots, and edges encode the inner products between them:
//!
//! - No edge: A_ij = A_ji = 0 (orthogonal roots)
//! - Single edge: A_ij = A_ji = -1 (simply-laced, angle 120°)
//! - Double edge with arrow: |A_ij| = 2, |A_ji| = 1 (angle 135°)
//! - Triple edge with arrow: |A_ij| = 3, |A_ji| = 1 (angle 150°)
//!
//! The arrow points from the longer root to the shorter root.
//!
//! Corresponds to sage.combinat.root_system.dynkin_diagram

use crate::cartan_type::{CartanType, CartanLetter, Affinity};
use crate::cartan_matrix::CartanMatrix;
use rustmath_integers::Integer;
use std::fmt::{self, Display};
use std::collections::{HashMap, HashSet};

/// Type of edge in a Dynkin diagram
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeType {
    /// No edge (orthogonal roots)
    None,
    /// Single edge (simply-laced, A_ij = A_ji = -1)
    Single,
    /// Double edge (A_ij * A_ji = 2)
    /// Tuple is (from, to, multiplicity)
    /// Arrow points from 'from' to 'to' (longer to shorter root)
    Double { from: usize, to: usize },
    /// Triple edge (A_ij * A_ji = 3, only for G_2)
    Triple { from: usize, to: usize },
}

/// An edge in the Dynkin diagram
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynkinEdge {
    /// Source node (0-indexed)
    pub source: usize,
    /// Target node (0-indexed)
    pub target: usize,
    /// Edge type
    pub edge_type: EdgeType,
}

impl DynkinEdge {
    /// Create a new edge
    pub fn new(source: usize, target: usize, edge_type: EdgeType) -> Self {
        DynkinEdge {
            source,
            target,
            edge_type,
        }
    }
}

/// A Dynkin diagram representing a root system
///
/// The diagram consists of nodes (representing simple roots) and edges
/// (representing inner products between roots).
#[derive(Clone, Debug)]
pub struct DynkinDiagram {
    /// The Cartan type
    pub cartan_type: CartanType,
    /// The Cartan matrix
    pub cartan_matrix: CartanMatrix,
    /// Number of nodes (equal to rank)
    pub num_nodes: usize,
    /// Edges in the diagram
    pub edges: Vec<DynkinEdge>,
    /// Adjacency list representation
    pub adjacency: HashMap<usize, Vec<usize>>,
}

impl DynkinDiagram {
    /// Create a Dynkin diagram from a Cartan type
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_liealgebras::cartan_type::{CartanType, CartanLetter};
    /// use rustmath_liealgebras::dynkin_diagram::DynkinDiagram;
    ///
    /// let ct = CartanType::new(CartanLetter::A, 3).unwrap();
    /// let dd = DynkinDiagram::new(ct);
    /// assert_eq!(dd.num_nodes(), 3);
    /// ```
    pub fn new(cartan_type: CartanType) -> Self {
        let num_nodes = cartan_type.rank;
        let cartan_matrix = CartanMatrix::new(cartan_type);
        let edges = Self::compute_edges(&cartan_matrix, num_nodes);
        let adjacency = Self::build_adjacency(&edges, num_nodes);

        DynkinDiagram {
            cartan_type,
            cartan_matrix,
            num_nodes,
            edges,
            adjacency,
        }
    }

    /// Compute edges from the Cartan matrix
    fn compute_edges(cartan_matrix: &CartanMatrix, num_nodes: usize) -> Vec<DynkinEdge> {
        let mut edges = Vec::new();
        let mut processed: HashSet<(usize, usize)> = HashSet::new();

        for i in 0..num_nodes {
            for j in i + 1..num_nodes {
                let a_ij = cartan_matrix.entry(i, j);
                let a_ji = cartan_matrix.entry(j, i);

                // Compute edge type based on Cartan matrix entries
                let product = a_ij * a_ji;

                let edge_type = if a_ij.is_zero() && a_ji.is_zero() {
                    EdgeType::None
                } else if *a_ij == Integer::from(-1) && *a_ji == Integer::from(-1) {
                    EdgeType::Single
                } else if product == Integer::from(2) {
                    // Double edge: arrow points from longer to shorter root
                    // The longer root has the more negative entry in its column
                    if a_ij.abs() > a_ji.abs() {
                        EdgeType::Double { from: i, to: j }
                    } else {
                        EdgeType::Double { from: j, to: i }
                    }
                } else if product == Integer::from(3) {
                    // Triple edge (G_2 only)
                    if a_ij.abs() > a_ji.abs() {
                        EdgeType::Triple { from: i, to: j }
                    } else {
                        EdgeType::Triple { from: j, to: i }
                    }
                } else {
                    EdgeType::None
                };

                if !matches!(edge_type, EdgeType::None) {
                    edges.push(DynkinEdge::new(i, j, edge_type));
                }
            }
        }

        edges
    }

    /// Build adjacency list from edges
    fn build_adjacency(edges: &[DynkinEdge], num_nodes: usize) -> HashMap<usize, Vec<usize>> {
        let mut adjacency: HashMap<usize, Vec<usize>> = HashMap::new();

        for i in 0..num_nodes {
            adjacency.insert(i, Vec::new());
        }

        for edge in edges {
            adjacency.get_mut(&edge.source).unwrap().push(edge.target);
            adjacency.get_mut(&edge.target).unwrap().push(edge.source);
        }

        adjacency
    }

    /// Get the number of nodes
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Get neighbors of a node
    pub fn neighbors(&self, node: usize) -> &[usize] {
        self.adjacency.get(&node).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Get the degree of a node
    pub fn degree(&self, node: usize) -> usize {
        self.neighbors(node).len()
    }

    /// Check if the diagram is connected
    pub fn is_connected(&self) -> bool {
        if self.num_nodes == 0 {
            return true;
        }

        let mut visited = vec![false; self.num_nodes];
        let mut stack = vec![0];
        visited[0] = true;
        let mut count = 1;

        while let Some(node) = stack.pop() {
            for &neighbor in self.neighbors(node) {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    stack.push(neighbor);
                    count += 1;
                }
            }
        }

        count == self.num_nodes
    }

    /// Check if the diagram is simply-laced (all edges are single)
    pub fn is_simply_laced(&self) -> bool {
        self.edges.iter().all(|e| matches!(e.edge_type, EdgeType::Single))
    }

    /// Get the automorphism group order
    ///
    /// The automorphism group consists of symmetries of the Dynkin diagram
    pub fn automorphism_group_order(&self) -> usize {
        match self.cartan_type.letter {
            CartanLetter::A if self.num_nodes >= 2 => 2, // Reflection symmetry
            CartanLetter::D if self.num_nodes >= 4 => 2, // Swap the two ends of the fork
            CartanLetter::E if self.num_nodes == 6 => 2, // Reflection through the branch
            _ => 1, // No non-trivial automorphisms
        }
    }

    /// Get ASCII art representation of the diagram
    pub fn to_ascii(&self) -> String {
        match self.cartan_type.letter {
            CartanLetter::A => self.ascii_type_a(),
            CartanLetter::B => self.ascii_type_b(),
            CartanLetter::C => self.ascii_type_c(),
            CartanLetter::D => self.ascii_type_d(),
            CartanLetter::E => self.ascii_type_e(),
            CartanLetter::F => self.ascii_type_f(),
            CartanLetter::G => self.ascii_type_g(),
        }
    }

    /// ASCII art for type A_n (linear chain)
    fn ascii_type_a(&self) -> String {
        let n = self.num_nodes;
        let mut result = String::new();

        // Node line
        for i in 0..n {
            result.push_str(&format!("●"));
            if i < n - 1 {
                result.push_str("───");
            }
        }
        result.push('\n');

        // Label line
        for i in 0..n {
            result.push_str(&format!("{}", i + 1));
            if i < n - 1 {
                result.push_str("   ");
            }
        }

        result
    }

    /// ASCII art for type B_n
    fn ascii_type_b(&self) -> String {
        let n = self.num_nodes;
        let mut result = String::new();

        // Node line
        for i in 0..n {
            result.push_str("●");
            if i < n - 2 {
                result.push_str("───");
            } else if i == n - 2 {
                result.push_str("═══>"); // Double bond with arrow
            }
        }
        result.push('\n');

        // Label line
        for i in 0..n {
            result.push_str(&format!("{}", i + 1));
            if i < n - 2 {
                result.push_str("   ");
            } else if i == n - 2 {
                result.push_str("    ");
            }
        }

        result
    }

    /// ASCII art for type C_n
    fn ascii_type_c(&self) -> String {
        let n = self.num_nodes;
        let mut result = String::new();

        // Node line
        for i in 0..n {
            result.push_str("●");
            if i < n - 2 {
                result.push_str("───");
            } else if i == n - 2 {
                result.push_str("<═══"); // Double bond with arrow (reversed)
            }
        }
        result.push('\n');

        // Label line
        for i in 0..n {
            result.push_str(&format!("{}", i + 1));
            if i < n - 2 {
                result.push_str("   ");
            } else if i == n - 2 {
                result.push_str("    ");
            }
        }

        result
    }

    /// ASCII art for type D_n (fork at the end)
    fn ascii_type_d(&self) -> String {
        let n = self.num_nodes;
        let mut result = String::new();

        // Top branch
        result.push_str("        ");
        for _ in 0..n - 3 {
            result.push_str("    ");
        }
        result.push_str(&format!("●\n"));
        result.push_str("        ");
        for _ in 0..n - 3 {
            result.push_str("    ");
        }
        result.push_str(&format!("╱ {}\n", n - 1));

        // Main line
        for i in 0..n - 2 {
            result.push_str("●");
            if i < n - 3 {
                result.push_str("───");
            }
        }
        result.push('\n');

        // Labels for main line
        for i in 0..n - 3 {
            result.push_str(&format!("{}", i + 1));
            result.push_str("   ");
        }
        result.push_str(&format!("{}\n", n - 2));

        // Bottom branch
        result.push_str("        ");
        for _ in 0..n - 3 {
            result.push_str("    ");
        }
        result.push_str(&format!("╲ {}\n", n));
        result.push_str("        ");
        for _ in 0..n - 3 {
            result.push_str("    ");
        }
        result.push_str("●");

        result
    }

    /// ASCII art for type E_n
    fn ascii_type_e(&self) -> String {
        let n = self.num_nodes;
        let mut result = String::new();

        // Main line with branch at position 2 (0-indexed)
        // Nodes: 0-1-2-3-4-...
        //            |
        //            5 (or higher for E7, E8)

        // First line: branch node
        result.push_str("        ●\n");
        result.push_str(&format!("        │ {}\n", n));

        // Second line: main chain
        for i in 0..n - 1 {
            result.push_str("●");
            if i < n - 2 {
                result.push_str("───");
            }
        }
        result.push('\n');

        // Labels
        for i in 0..n - 1 {
            result.push_str(&format!("{}", i + 1));
            if i < n - 2 {
                result.push_str("   ");
            }
        }

        result
    }

    /// ASCII art for type F_4
    fn ascii_type_f(&self) -> String {
        String::from("●───●═══>●───●\n1   2    3   4")
    }

    /// ASCII art for type G_2
    fn ascii_type_g(&self) -> String {
        String::from("●≡≡≡>●\n1    2")
    }

    /// Get a detailed description of the diagram
    pub fn describe(&self) -> String {
        let mut desc = format!("Dynkin diagram of type {}\n", self.cartan_type);
        desc.push_str(&format!("Nodes: {}\n", self.num_nodes));
        desc.push_str(&format!("Edges: {}\n", self.edges.len()));
        desc.push_str(&format!("Connected: {}\n", self.is_connected()));
        desc.push_str(&format!("Simply-laced: {}\n", self.is_simply_laced()));

        desc.push_str("\nEdges:\n");
        for edge in &self.edges {
            match edge.edge_type {
                EdgeType::Single => {
                    desc.push_str(&format!("  {} ─ {} (single)\n", edge.source + 1, edge.target + 1));
                }
                EdgeType::Double { from, to } => {
                    desc.push_str(&format!("  {} ══> {} (double)\n", from + 1, to + 1));
                }
                EdgeType::Triple { from, to } => {
                    desc.push_str(&format!("  {} ≡≡> {} (triple)\n", from + 1, to + 1));
                }
                EdgeType::None => {}
            }
        }

        desc
    }
}

impl Display for DynkinDiagram {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_ascii())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynkin_diagram_a3() {
        let ct = CartanType::new(CartanLetter::A, 3).unwrap();
        let dd = DynkinDiagram::new(ct);

        assert_eq!(dd.num_nodes(), 3);
        assert_eq!(dd.edges.len(), 2); // Two edges in linear chain
        assert!(dd.is_connected());
        assert!(dd.is_simply_laced());
    }

    #[test]
    fn test_dynkin_diagram_b3() {
        let ct = CartanType::new(CartanLetter::B, 3).unwrap();
        let dd = DynkinDiagram::new(ct);

        assert_eq!(dd.num_nodes(), 3);
        assert!(!dd.is_simply_laced()); // Has double edge
    }

    #[test]
    fn test_dynkin_diagram_d4() {
        let ct = CartanType::new(CartanLetter::D, 4).unwrap();
        let dd = DynkinDiagram::new(ct);

        assert_eq!(dd.num_nodes(), 4);
        assert!(dd.is_connected());
        assert!(dd.is_simply_laced());

        // Node 1 (index 0) should have degree 3 (the fork point)
        // Actually in D_4, the fork is at the second-to-last node
        assert_eq!(dd.degree(1), 3);
    }

    #[test]
    fn test_dynkin_diagram_g2() {
        let ct = CartanType::new(CartanLetter::G, 2).unwrap();
        let dd = DynkinDiagram::new(ct);

        assert_eq!(dd.num_nodes(), 2);
        assert_eq!(dd.edges.len(), 1);
        assert!(!dd.is_simply_laced());

        // Should have a triple edge
        let edge = &dd.edges[0];
        assert!(matches!(edge.edge_type, EdgeType::Triple { .. }));
    }

    #[test]
    fn test_automorphism_group() {
        let a3 = DynkinDiagram::new(CartanType::new(CartanLetter::A, 3).unwrap());
        assert_eq!(a3.automorphism_group_order(), 2); // Reflection symmetry

        let b3 = DynkinDiagram::new(CartanType::new(CartanLetter::B, 3).unwrap());
        assert_eq!(b3.automorphism_group_order(), 1); // No symmetry

        let d4 = DynkinDiagram::new(CartanType::new(CartanLetter::D, 4).unwrap());
        assert_eq!(d4.automorphism_group_order(), 2); // Swap fork ends
    }

    #[test]
    fn test_ascii_representation() {
        let a3 = DynkinDiagram::new(CartanType::new(CartanLetter::A, 3).unwrap());
        let ascii = a3.to_ascii();
        assert!(ascii.contains("●"));
        assert!(ascii.contains("───"));

        let g2 = DynkinDiagram::new(CartanType::new(CartanLetter::G, 2).unwrap());
        let ascii_g2 = g2.to_ascii();
        assert!(ascii_g2.contains("≡"));
    }

    #[test]
    fn test_all_finite_types() {
        // Test that all finite types produce valid connected diagrams
        for n in 1..=5 {
            let dd = DynkinDiagram::new(CartanType::new(CartanLetter::A, n).unwrap());
            assert!(dd.is_connected());
        }

        for n in 2..=5 {
            let dd = DynkinDiagram::new(CartanType::new(CartanLetter::B, n).unwrap());
            assert!(dd.is_connected());

            let dd = DynkinDiagram::new(CartanType::new(CartanLetter::C, n).unwrap());
            assert!(dd.is_connected());
        }

        for n in 3..=5 {
            let dd = DynkinDiagram::new(CartanType::new(CartanLetter::D, n).unwrap());
            assert!(dd.is_connected());
        }

        let dd = DynkinDiagram::new(CartanType::new(CartanLetter::E, 6).unwrap());
        assert!(dd.is_connected());

        let dd = DynkinDiagram::new(CartanType::new(CartanLetter::E, 7).unwrap());
        assert!(dd.is_connected());

        let dd = DynkinDiagram::new(CartanType::new(CartanLetter::E, 8).unwrap());
        assert!(dd.is_connected());

        let dd = DynkinDiagram::new(CartanType::new(CartanLetter::F, 4).unwrap());
        assert!(dd.is_connected());

        let dd = DynkinDiagram::new(CartanType::new(CartanLetter::G, 2).unwrap());
        assert!(dd.is_connected());
    }

    #[test]
    fn test_affine_diagrams() {
        // Affine diagrams should also be valid
        let a2_aff = DynkinDiagram::new(CartanType::new_affine(CartanLetter::A, 2, 1).unwrap());
        assert_eq!(a2_aff.num_nodes(), 3);
        assert!(a2_aff.is_connected());
    }
}
