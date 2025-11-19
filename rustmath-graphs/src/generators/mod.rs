//! Graph generators for creating various graph structures
//!
//! This module provides functions to generate common graph types organized into submodules:
//! - `basic`: Fundamental graph types (complete, path, cycle, grid, etc.)
//! - `chessboard`: Chessboard-based graphs (rook, bishop, knight, queen, king)
//! - `classical_geometries`: Graphs from classical geometries
//! - `distance_regular`: Distance-regular graphs and generalized polygons
//! - `families`: Common graph families (trees, cubes, Petersen variations, etc.)
//! - `intersection`: Intersection graphs (interval, permutation, tolerance, etc.)
//! - `platonic_solids`: The five Platonic solid graphs (tetrahedron, cube, octahedron, icosahedron, dodecahedron)
//! - `random`: Random graph models (Erdős-Rényi, Barabási-Albert, etc.)

pub mod basic;
pub mod chessboard;
pub mod classical_geometries;
pub mod distance_regular;
pub mod families;
pub mod intersection;
pub mod platonic_solids;
pub mod random;

// Re-export common generators for convenience
pub use basic::*;

use crate::graph::Graph;

/// Generate a wheel graph W_n
///
/// A cycle with an additional central vertex connected to all cycle vertices.
///
/// # Arguments
///
/// * `n` - Number of outer vertices
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::wheel_graph;
///
/// let g = wheel_graph(5);
/// assert_eq!(g.num_vertices(), 6);
/// ```
pub fn wheel_graph(n: usize) -> Graph {
    if n < 3 {
        panic!("Wheel graph requires at least 3 outer vertices");
    }

    let mut g = Graph::new(n + 1);

    // Create the outer cycle
    for i in 1..=n {
        g.add_edge(i, if i == n { 1 } else { i + 1 }).unwrap();
    }

    // Connect center to all outer vertices
    for i in 1..=n {
        g.add_edge(0, i).unwrap();
    }

    g
}

/// Generate a Petersen graph
///
/// The Petersen graph is a famous graph with 10 vertices and 15 edges.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::petersen_graph;
///
/// let g = petersen_graph();
/// assert_eq!(g.num_vertices(), 10);
/// assert_eq!(g.num_edges(), 15);
/// ```
pub fn petersen_graph() -> Graph {
    let mut g = Graph::new(10);

    // Outer pentagon: vertices 0-4
    for i in 0..5 {
        g.add_edge(i, (i + 1) % 5).unwrap();
    }

    // Inner pentagram: vertices 5-9
    for i in 0..5 {
        g.add_edge(5 + i, 5 + (i + 2) % 5).unwrap();
    }

    // Connect outer to inner
    for i in 0..5 {
        g.add_edge(i, 5 + i).unwrap();
    }

    g
}

/// Generate a random graph using the Erdős-Rényi model G(n, p)
///
/// Each possible edge is included independently with probability p.
///
/// # Arguments
///
/// * `n` - Number of vertices
/// * `p` - Probability of including each edge
///
/// # Examples
///
/// ```ignore
/// use rustmath_graphs::generators::random_graph;
///
/// let g = random_graph(10, 0.5);
/// assert_eq!(g.num_vertices(), 10);
/// ```
#[cfg(feature = "random")]
pub fn random_graph(n: usize, p: f64) -> Graph {
    use rand::Rng;

    let mut g = Graph::new(n);
    let mut rng = rand::thread_rng();

    for i in 0..n {
        for j in (i + 1)..n {
            if rng.gen::<f64>() < p {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wheel_graph() {
        let g = wheel_graph(5);
        assert_eq!(g.num_vertices(), 6);
        assert_eq!(g.num_edges(), 10);

        // Center has degree 5
        assert_eq!(g.degree(0), Some(5));

        // Outer vertices have degree 3
        for i in 1..=5 {
            assert_eq!(g.degree(i), Some(3));
        }
    }

    #[test]
    fn test_petersen_graph() {
        let g = petersen_graph();
        assert_eq!(g.num_vertices(), 10);
        assert_eq!(g.num_edges(), 15);

        // All vertices in Petersen graph have degree 3
        for i in 0..10 {
            assert_eq!(g.degree(i), Some(3));
        }

        // Petersen graph has chromatic number 3
        assert_eq!(g.chromatic_number(), 3);
    }
}
