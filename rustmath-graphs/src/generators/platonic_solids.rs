//! Platonic Solid Graph Generators
//!
//! This module provides graph generators for the five Platonic solids.
//! Each function creates the 1-skeleton (vertex-edge graph) of the
//! corresponding Platonic solid.

use crate::graph::Graph;


/// Generate a Tetrahedral Graph
///
/// The tetrahedral graph is the complete graph K4, representing the
/// connectivity of vertices in a regular tetrahedron.
///
/// # Properties
///
/// * Vertices: 4
/// * Edges: 6
/// * Degree: 3 (regular)
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::platonic_solids::tetrahedral_graph;
///
/// let g = tetrahedral_graph();
/// assert_eq!(g.num_vertices(), 4);
/// assert_eq!(g.num_edges(), 6);
/// ```
pub fn tetrahedral_graph() -> Graph {
    // The tetrahedral graph is K4 (complete graph on 4 vertices)
    let mut g = Graph::new(4);

    for i in 0..4 {
        for j in (i + 1)..4 {
            g.add_edge(i, j).unwrap();
        }
    }

    g
}

/// Generate a Hexahedral Graph (Cube)
///
/// The hexahedral graph represents the connectivity of vertices in a cube.
/// It is also known as the 3-dimensional hypercube Q3.
///
/// # Properties
///
/// * Vertices: 8
/// * Edges: 12
/// * Degree: 3 (regular)
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::platonic_solids::hexahedral_graph;
///
/// let g = hexahedral_graph();
/// assert_eq!(g.num_vertices(), 8);
/// assert_eq!(g.num_edges(), 12);
/// ```
pub fn hexahedral_graph() -> Graph {
    // Cube vertices can be represented as 3-bit binary numbers
    // Edges connect vertices differing in exactly one bit
    let mut g = Graph::new(8);

    for i in 0..8 {
        for j in (i + 1)..8 {
            // Count number of differing bits
            let xor: usize = i ^ j;
            // Check if exactly one bit differs (power of 2)
            if xor.count_ones() == 1 {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate an Octahedral Graph
///
/// The octahedral graph represents the connectivity of vertices in a
/// regular octahedron. It is equivalent to K6 minus three independent edges.
///
/// # Properties
///
/// * Vertices: 6
/// * Edges: 12
/// * Degree: 4 (regular)
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::platonic_solids::octahedral_graph;
///
/// let g = octahedral_graph();
/// assert_eq!(g.num_vertices(), 6);
/// assert_eq!(g.num_edges(), 12);
/// ```
pub fn octahedral_graph() -> Graph {
    // The octahedral graph is K6 minus three independent edges:
    // {0,1}, {2,3}, {4,5}
    let mut g = Graph::new(6);

    let excluded_edges = [(0, 1), (2, 3), (4, 5)];

    for i in 0..6 {
        for j in (i + 1)..6 {
            // Add edge unless it's one of the excluded pairs
            if !excluded_edges.contains(&(i, j)) {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate an Icosahedral Graph
///
/// The icosahedral graph represents the connectivity of vertices in a
/// regular icosahedron.
///
/// # Properties
///
/// * Vertices: 12
/// * Edges: 30
/// * Degree: 5 (regular)
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::platonic_solids::icosahedral_graph;
///
/// let g = icosahedral_graph();
/// assert_eq!(g.num_vertices(), 12);
/// assert_eq!(g.num_edges(), 30);
/// ```
pub fn icosahedral_graph() -> Graph {
    let mut g = Graph::new(12);

    // Icosahedron with 12 vertices, 30 edges
    // Using standard vertex numbering
    let edges = [
        (0, 1), (0, 5), (0, 7), (0, 8), (0, 11),
        (1, 2), (1, 5), (1, 6), (1, 8),
        (2, 3), (2, 6), (2, 8), (2, 9),
        (3, 4), (3, 6), (3, 9), (3, 10),
        (4, 5), (4, 7), (4, 10), (4, 11),
        (5, 6), (5, 11),
        (6, 10),
        (7, 8), (7, 9), (7, 11),
        (8, 9),
        (9, 10),
        (10, 11),
    ];

    for &(i, j) in &edges {
        g.add_edge(i, j).unwrap();
    }

    g
}

/// Generate a Dodecahedral Graph
///
/// The dodecahedral graph represents the connectivity of vertices in a
/// regular dodecahedron.
///
/// # Properties
///
/// * Vertices: 20
/// * Edges: 30
/// * Degree: 3 (regular)
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::platonic_solids::dodecahedral_graph;
///
/// let g = dodecahedral_graph();
/// assert_eq!(g.num_vertices(), 20);
/// assert_eq!(g.num_edges(), 30);
/// ```
pub fn dodecahedral_graph() -> Graph {
    let mut g = Graph::new(20);

    // Dodecahedron with 20 vertices, 30 edges
    // Each vertex has degree 3
    let edges = [
        (0, 1), (0, 4), (0, 5),
        (1, 2), (1, 6),
        (2, 3), (2, 7),
        (3, 4), (3, 8),
        (4, 9),
        (5, 10), (5, 14),
        (6, 10), (6, 11),
        (7, 11), (7, 12),
        (8, 12), (8, 13),
        (9, 13), (9, 14),
        (10, 15),
        (11, 16),
        (12, 17),
        (13, 18),
        (14, 19),
        (15, 16), (15, 19),
        (16, 17),
        (17, 18),
        (18, 19),
    ];

    for &(i, j) in &edges {
        g.add_edge(i, j).unwrap();
    }

    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tetrahedral_graph() {
        let g = tetrahedral_graph();
        assert_eq!(g.num_vertices(), 4);
        assert_eq!(g.num_edges(), 6);
        // Each vertex should have degree 3
        for v in 0..4 {
            assert_eq!(g.degree(v), Some(3));
        }
    }

    #[test]
    fn test_hexahedral_graph() {
        let g = hexahedral_graph();
        assert_eq!(g.num_vertices(), 8);
        assert_eq!(g.num_edges(), 12);
        // Each vertex should have degree 3
        for v in 0..8 {
            assert_eq!(g.degree(v), Some(3));
        }
    }

    #[test]
    fn test_octahedral_graph() {
        let g = octahedral_graph();
        assert_eq!(g.num_vertices(), 6);
        assert_eq!(g.num_edges(), 12);
        // Each vertex should have degree 4
        for v in 0..6 {
            assert_eq!(g.degree(v), Some(4));
        }
    }

    #[test]
    fn test_icosahedral_graph() {
        let g = icosahedral_graph();
        assert_eq!(g.num_vertices(), 12);
        assert_eq!(g.num_edges(), 30);
        // Each vertex should have degree 5
        for v in 0..12 {
            assert_eq!(g.degree(v), Some(5));
        }
    }

    #[test]
    fn test_dodecahedral_graph() {
        let g = dodecahedral_graph();
        assert_eq!(g.num_vertices(), 20);
        assert_eq!(g.num_edges(), 30);
        // Each vertex should have degree 3
        for v in 0..20 {
            assert_eq!(g.degree(v), Some(3));
        }
    }
}
