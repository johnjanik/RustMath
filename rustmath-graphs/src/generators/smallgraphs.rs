//! Small famous graphs
//!
//! This module provides generators for famous small graphs that appear
//! frequently in graph theory literature, including Chvátal, Clebsch, Coxeter,
//! Desargues, and many other named graphs.

use crate::graph::Graph;

/// Generate the Chvátal graph
///
/// The Chvátal graph is a 4-regular, 4-chromatic graph with 12 vertices and 24 edges.
/// It has radius 2, diameter 2, and girth 4.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::smallgraphs::chvatal_graph;
///
/// let g = chvatal_graph();
/// assert_eq!(g.num_vertices(), 12);
/// assert_eq!(g.num_edges(), 24);
/// ```
pub fn chvatal_graph() -> Graph {
    let mut g = Graph::new(12);

    // Explicit edge list for Chvátal graph
    let edges = vec![
        (0, 1), (0, 4), (0, 6), (0, 9),
        (1, 2), (1, 5), (1, 7),
        (2, 3), (2, 6), (2, 8),
        (3, 4), (3, 7), (3, 9),
        (4, 5), (4, 8),
        (5, 10), (5, 11),
        (6, 10), (6, 11),
        (7, 8), (7, 11),
        (8, 10),
        (9, 10), (9, 11),
    ];

    for (u, v) in edges {
        g.add_edge(u, v).unwrap();
    }

    g
}

/// Generate the Clebsch graph
///
/// The Clebsch graph is a 5-regular graph with 16 vertices and 40 edges.
/// It has diameter 2, girth 4, and chromatic number 4.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::smallgraphs::clebsch_graph;
///
/// let g = clebsch_graph();
/// assert_eq!(g.num_vertices(), 16);
/// assert_eq!(g.num_edges(), 40);
/// ```
pub fn clebsch_graph() -> Graph {
    let mut g = Graph::new(16);

    // Construct using SageMath's pattern
    let mut x = 0;
    for _ in 0..8 {
        g.add_edge(x % 16, (x + 1) % 16).unwrap();
        g.add_edge(x % 16, (x + 6) % 16).unwrap();
        g.add_edge(x % 16, (x + 8) % 16).unwrap();
        x += 1;
        g.add_edge(x % 16, (x + 3) % 16).unwrap();
        g.add_edge(x % 16, (x + 2) % 16).unwrap();
        g.add_edge(x % 16, (x + 8) % 16).unwrap();
        x += 1;
    }

    g
}

/// Generate the Coxeter graph
///
/// The Coxeter graph is a 3-regular graph with 28 vertices and 42 edges.
/// It has girth 7, chromatic number 3, and diameter 4.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::smallgraphs::coxeter_graph;
///
/// let g = coxeter_graph();
/// assert_eq!(g.num_vertices(), 28);
/// assert_eq!(g.num_edges(), 42);
/// ```
pub fn coxeter_graph() -> Graph {
    let mut g = Graph::new(28);

    // Base 24-cycle (vertices 0-23)
    for i in 0..24 {
        g.add_edge(i, (i + 1) % 24).unwrap();
    }

    // Six additional chords in the 24-cycle
    g.add_edge(5, 11).unwrap();
    g.add_edge(9, 20).unwrap();
    g.add_edge(12, 1).unwrap();
    g.add_edge(13, 19).unwrap();
    g.add_edge(17, 4).unwrap();
    g.add_edge(3, 21).unwrap();

    // Connect outer vertices 24-27
    g.add_edge(24, 0).unwrap();
    g.add_edge(24, 7).unwrap();
    g.add_edge(24, 18).unwrap();

    g.add_edge(25, 2).unwrap();
    g.add_edge(25, 8).unwrap();
    g.add_edge(25, 15).unwrap();

    g.add_edge(26, 10).unwrap();
    g.add_edge(26, 16).unwrap();
    g.add_edge(26, 23).unwrap();

    g.add_edge(27, 6).unwrap();
    g.add_edge(27, 14).unwrap();
    g.add_edge(27, 22).unwrap();

    g
}

/// Generate the Desargues graph
///
/// The Desargues graph is a 3-regular bipartite graph with 20 vertices and 30 edges.
/// It is isomorphic to the generalized Petersen graph GP(10, 3).
/// It has diameter 5 and girth 6.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::smallgraphs::desargues_graph;
///
/// let g = desargues_graph();
/// assert_eq!(g.num_vertices(), 20);
/// assert_eq!(g.num_edges(), 30);
/// ```
pub fn desargues_graph() -> Graph {
    // Desargues graph is GP(10, 3)
    crate::generators::families::generalized_petersen_graph(10, 3)
}

/// Generate the Bucky Ball graph (Buckminsterfullerene)
///
/// The Bucky Ball graph represents the carbon skeleton of the buckminsterfullerene
/// molecule (C60). It is a 3-regular planar graph with 60 vertices and 90 edges,
/// consisting of 12 pentagons and 20 hexagons arranged like a soccer ball.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::smallgraphs::bucky_ball;
///
/// let g = bucky_ball();
/// assert_eq!(g.num_vertices(), 60);
/// assert_eq!(g.num_edges(), 90);
/// ```
pub fn bucky_ball() -> Graph {
    let mut g = Graph::new(60);

    // Bucky ball edge list (truncated icosahedron)
    let edges = vec![
        (0, 1), (0, 4), (0, 5), (1, 2), (1, 6), (2, 3), (2, 7), (3, 4), (3, 8), (4, 9),
        (5, 10), (5, 14), (6, 10), (6, 11), (7, 11), (7, 12), (8, 12), (8, 13), (9, 13), (9, 14),
        (10, 15), (11, 16), (12, 17), (13, 18), (14, 19), (15, 20), (15, 24), (16, 20), (16, 21),
        (17, 21), (17, 22), (18, 22), (18, 23), (19, 23), (19, 24), (20, 25), (21, 26), (22, 27),
        (23, 28), (24, 29), (25, 30), (25, 34), (26, 30), (26, 31), (27, 31), (27, 32), (28, 32),
        (28, 33), (29, 33), (29, 34), (30, 35), (31, 36), (32, 37), (33, 38), (34, 39), (35, 40),
        (35, 44), (36, 40), (36, 41), (37, 41), (37, 42), (38, 42), (38, 43), (39, 43), (39, 44),
        (40, 45), (41, 46), (42, 47), (43, 48), (44, 49), (45, 50), (45, 54), (46, 50), (46, 51),
        (47, 51), (47, 52), (48, 52), (48, 53), (49, 53), (49, 54), (50, 55), (51, 56), (52, 57),
        (53, 58), (54, 59), (55, 56), (55, 59), (56, 57), (57, 58), (58, 59),
    ];

    for (u, v) in edges {
        g.add_edge(u, v).unwrap();
    }

    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chvatal_graph() {
        let g = chvatal_graph();
        assert_eq!(g.num_vertices(), 12);
        assert_eq!(g.num_edges(), 24);

        // Chvátal graph is 4-regular
        for v in 0..12 {
            assert_eq!(g.degree(v), Some(4));
        }
    }

    #[test]
    fn test_clebsch_graph() {
        let g = clebsch_graph();
        assert_eq!(g.num_vertices(), 16);
        assert_eq!(g.num_edges(), 40);

        // Clebsch graph is 5-regular
        for v in 0..16 {
            assert_eq!(g.degree(v), Some(5));
        }
    }

    #[test]
    fn test_coxeter_graph() {
        let g = coxeter_graph();
        assert_eq!(g.num_vertices(), 28);
        assert_eq!(g.num_edges(), 42);

        // Coxeter graph is 3-regular
        for v in 0..28 {
            assert_eq!(g.degree(v), Some(3));
        }
    }

    #[test]
    fn test_desargues_graph() {
        let g = desargues_graph();
        assert_eq!(g.num_vertices(), 20);
        assert_eq!(g.num_edges(), 30);

        // Desargues graph is 3-regular
        for v in 0..20 {
            assert_eq!(g.degree(v), Some(3));
        }
    }

    #[test]
    fn test_bucky_ball() {
        let g = bucky_ball();
        assert_eq!(g.num_vertices(), 60);
        assert_eq!(g.num_edges(), 90);

        // Bucky ball is 3-regular
        for v in 0..60 {
            assert_eq!(g.degree(v), Some(3));
        }
    }
}
