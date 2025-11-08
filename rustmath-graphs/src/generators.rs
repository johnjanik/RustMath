//! Graph generators for creating common graph structures

use crate::graph::Graph;

/// Generate a complete graph K_n
///
/// A complete graph has edges between all pairs of vertices
pub fn complete_graph(n: usize) -> Graph {
    let mut g = Graph::new(n);

    for i in 0..n {
        for j in (i + 1)..n {
            g.add_edge(i, j).unwrap();
        }
    }

    g
}

/// Generate a cycle graph C_n
///
/// A cycle connects vertices in a ring: 0-1-2-...-n-0
pub fn cycle_graph(n: usize) -> Graph {
    if n < 3 {
        panic!("Cycle graph requires at least 3 vertices");
    }

    let mut g = Graph::new(n);

    for i in 0..n {
        g.add_edge(i, (i + 1) % n).unwrap();
    }

    g
}

/// Generate a path graph P_n
///
/// A path connects vertices in a line: 0-1-2-...-n
pub fn path_graph(n: usize) -> Graph {
    let mut g = Graph::new(n);

    for i in 0..(n.saturating_sub(1)) {
        g.add_edge(i, i + 1).unwrap();
    }

    g
}

/// Generate a complete bipartite graph K_{m,n}
///
/// All vertices in one part connect to all vertices in the other part
pub fn complete_bipartite_graph(m: usize, n: usize) -> Graph {
    let mut g = Graph::new(m + n);

    for i in 0..m {
        for j in 0..n {
            g.add_edge(i, m + j).unwrap();
        }
    }

    g
}

/// Generate a star graph S_n
///
/// One central vertex connected to n outer vertices
pub fn star_graph(n: usize) -> Graph {
    let mut g = Graph::new(n + 1);

    for i in 1..=n {
        g.add_edge(0, i).unwrap();
    }

    g
}

/// Generate a wheel graph W_n
///
/// A cycle with an additional central vertex connected to all cycle vertices
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

/// Generate a grid graph (m x n)
///
/// Vertices arranged in a rectangular grid with edges to adjacent vertices
pub fn grid_graph(m: usize, n: usize) -> Graph {
    let mut g = Graph::new(m * n);

    for i in 0..m {
        for j in 0..n {
            let v = i * n + j;

            // Connect to right neighbor
            if j + 1 < n {
                g.add_edge(v, v + 1).unwrap();
            }

            // Connect to bottom neighbor
            if i + 1 < m {
                g.add_edge(v, v + n).unwrap();
            }
        }
    }

    g
}

/// Generate a Petersen graph
///
/// The Petersen graph is a famous graph with 10 vertices and 15 edges
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

/// Generate an empty graph with n vertices and no edges
pub fn empty_graph(n: usize) -> Graph {
    Graph::new(n)
}

/// Generate a random graph using the Erdős-Rényi model G(n, p)
///
/// Each possible edge is included independently with probability p
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
    fn test_complete_graph() {
        let g = complete_graph(4);
        assert_eq!(g.num_vertices(), 4);
        assert_eq!(g.num_edges(), 6); // K_4 has 6 edges

        // Every pair should be connected
        for i in 0..4 {
            for j in (i + 1)..4 {
                assert!(g.has_edge(i, j));
            }
        }
    }

    #[test]
    fn test_cycle_graph() {
        let g = cycle_graph(5);
        assert_eq!(g.num_vertices(), 5);
        assert_eq!(g.num_edges(), 5);

        // Each vertex should have degree 2
        for i in 0..5 {
            assert_eq!(g.degree(i), Some(2));
        }
    }

    #[test]
    fn test_path_graph() {
        let g = path_graph(5);
        assert_eq!(g.num_vertices(), 5);
        assert_eq!(g.num_edges(), 4);

        // End vertices have degree 1
        assert_eq!(g.degree(0), Some(1));
        assert_eq!(g.degree(4), Some(1));

        // Middle vertices have degree 2
        assert_eq!(g.degree(2), Some(2));

        // Path graph is a tree
        assert!(g.is_tree());
    }

    #[test]
    fn test_complete_bipartite() {
        let g = complete_bipartite_graph(3, 2);
        assert_eq!(g.num_vertices(), 5);
        assert_eq!(g.num_edges(), 6); // 3 * 2 = 6

        // Should be bipartite
        assert!(g.is_bipartite());
    }

    #[test]
    fn test_star_graph() {
        let g = star_graph(5);
        assert_eq!(g.num_vertices(), 6);
        assert_eq!(g.num_edges(), 5);

        // Center has degree 5
        assert_eq!(g.degree(0), Some(5));

        // Outer vertices have degree 1
        for i in 1..=5 {
            assert_eq!(g.degree(i), Some(1));
        }

        // Star graph is a tree
        assert!(g.is_tree());
    }

    #[test]
    fn test_wheel_graph() {
        let g = wheel_graph(5);
        assert_eq!(g.num_vertices(), 6);
        assert_eq!(g.num_edges(), 10); // 5 + 5 = 10

        // Center has degree 5
        assert_eq!(g.degree(0), Some(5));

        // Outer vertices have degree 3
        for i in 1..=5 {
            assert_eq!(g.degree(i), Some(3));
        }
    }

    #[test]
    fn test_grid_graph() {
        let g = grid_graph(3, 4);
        assert_eq!(g.num_vertices(), 12);

        // Corner vertices have degree 2
        assert_eq!(g.degree(0), Some(2));
        assert_eq!(g.degree(11), Some(2));

        // Edge vertices (not corners) have degree 3
        assert_eq!(g.degree(1), Some(3));

        // Interior vertices have degree 4
        assert_eq!(g.degree(5), Some(4));
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

    #[test]
    fn test_empty_graph() {
        let g = empty_graph(5);
        assert_eq!(g.num_vertices(), 5);
        assert_eq!(g.num_edges(), 0);

        // All vertices have degree 0
        for i in 0..5 {
            assert_eq!(g.degree(i), Some(0));
        }
    }
}
