//! Basic graph generators for common graph structures
//!
//! This module provides functions to generate fundamental graph types
//! such as complete graphs, paths, cycles, and various named small graphs.

use crate::graph::Graph;

/// Generate a Bull graph
///
/// The bull graph is a planar undirected graph with 5 vertices and 5 edges,
/// in the form of a triangle with two disjoint pendant edges.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::basic::bull_graph;
///
/// let g = bull_graph();
/// assert_eq!(g.num_vertices(), 5);
/// assert_eq!(g.num_edges(), 5);
/// ```
pub fn bull_graph() -> Graph {
    let mut g = Graph::new(5);
    // Triangle: 0-1-2-0
    g.add_edge(0, 1).unwrap();
    g.add_edge(1, 2).unwrap();
    g.add_edge(2, 0).unwrap();
    // Pendant edges
    g.add_edge(1, 3).unwrap();
    g.add_edge(2, 4).unwrap();
    g
}

/// Generate a Butterfly graph (Bowtie graph)
///
/// The butterfly graph consists of two triangles sharing a common vertex.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::basic::butterfly_graph;
///
/// let g = butterfly_graph();
/// assert_eq!(g.num_vertices(), 5);
/// assert_eq!(g.num_edges(), 6);
/// ```
pub fn butterfly_graph() -> Graph {
    let mut g = Graph::new(5);
    // First triangle: 0-1-2-0
    g.add_edge(0, 1).unwrap();
    g.add_edge(1, 2).unwrap();
    g.add_edge(2, 0).unwrap();
    // Second triangle: 0-3-4-0 (sharing vertex 0)
    g.add_edge(0, 3).unwrap();
    g.add_edge(3, 4).unwrap();
    g.add_edge(4, 0).unwrap();
    g
}

/// Generate a Circular Ladder graph (Prism graph)
///
/// A circular ladder graph is the Cartesian product of a cycle and an edge.
///
/// # Arguments
///
/// * `n` - Number of rungs
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::basic::circular_ladder_graph;
///
/// let g = circular_ladder_graph(4);
/// assert_eq!(g.num_vertices(), 8);
/// ```
pub fn circular_ladder_graph(n: usize) -> Graph {
    if n < 3 {
        panic!("Circular ladder graph requires at least 3 rungs");
    }

    let mut g = Graph::new(2 * n);

    // Inner cycle
    for i in 0..n {
        g.add_edge(i, (i + 1) % n).unwrap();
    }

    // Outer cycle
    for i in 0..n {
        g.add_edge(n + i, n + (i + 1) % n).unwrap();
    }

    // Rungs connecting inner and outer cycles
    for i in 0..n {
        g.add_edge(i, n + i).unwrap();
    }

    g
}

/// Generate a Claw graph (Star graph K_{1,3})
///
/// The claw graph is a star graph with one central vertex and 3 leaves.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::basic::claw_graph;
///
/// let g = claw_graph();
/// assert_eq!(g.num_vertices(), 4);
/// assert_eq!(g.num_edges(), 3);
/// ```
pub fn claw_graph() -> Graph {
    star_graph(3)
}

/// Generate a complete graph K_n
///
/// A complete graph has edges between all pairs of vertices.
///
/// # Arguments
///
/// * `n` - Number of vertices
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::basic::complete_graph;
///
/// let g = complete_graph(5);
/// assert_eq!(g.num_vertices(), 5);
/// assert_eq!(g.num_edges(), 10);
/// ```
pub fn complete_graph(n: usize) -> Graph {
    let mut g = Graph::new(n);

    for i in 0..n {
        for j in (i + 1)..n {
            g.add_edge(i, j).unwrap();
        }
    }

    g
}

/// Generate a complete bipartite graph K_{m,n}
///
/// All vertices in one part connect to all vertices in the other part.
///
/// # Arguments
///
/// * `m` - Number of vertices in first part
/// * `n` - Number of vertices in second part
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::basic::complete_bipartite_graph;
///
/// let g = complete_bipartite_graph(3, 2);
/// assert_eq!(g.num_vertices(), 5);
/// assert_eq!(g.num_edges(), 6);
/// ```
pub fn complete_bipartite_graph(m: usize, n: usize) -> Graph {
    let mut g = Graph::new(m + n);

    for i in 0..m {
        for j in 0..n {
            g.add_edge(i, m + j).unwrap();
        }
    }

    g
}

/// Generate a complete multipartite graph
///
/// A complete multipartite graph with partition sizes given by the input vector.
///
/// # Arguments
///
/// * `partition_sizes` - Vector of partition sizes
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::basic::complete_multipartite_graph;
///
/// let g = complete_multipartite_graph(&[2, 3, 2]);
/// assert_eq!(g.num_vertices(), 7);
/// ```
pub fn complete_multipartite_graph(partition_sizes: &[usize]) -> Graph {
    let n: usize = partition_sizes.iter().sum();
    let mut g = Graph::new(n);

    let mut offset = 0;
    for (i, &size_i) in partition_sizes.iter().enumerate() {
        let mut other_offset = offset + size_i;
        for j in (i + 1)..partition_sizes.len() {
            let size_j = partition_sizes[j];
            // Connect all vertices in partition i to all in partition j
            for u in offset..(offset + size_i) {
                for v in other_offset..(other_offset + size_j) {
                    g.add_edge(u, v).unwrap();
                }
            }
            other_offset += size_j;
        }
        offset += size_i;
    }

    g
}

/// Generate a Correlation graph
///
/// Creates a graph based on correlation structure (simplified version).
/// Two vertices are adjacent if their correlation exceeds a threshold.
///
/// # Arguments
///
/// * `correlations` - Matrix of correlations (as flat vector)
/// * `n` - Number of vertices
/// * `threshold` - Correlation threshold
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::basic::correlation_graph;
///
/// let correlations = vec![
///     1.0, 0.9, 0.1,
///     0.9, 1.0, 0.2,
///     0.1, 0.2, 1.0,
/// ];
/// let g = correlation_graph(&correlations, 3, 0.5);
/// assert_eq!(g.num_vertices(), 3);
/// ```
pub fn correlation_graph(correlations: &[f64], n: usize, threshold: f64) -> Graph {
    let mut g = Graph::new(n);

    for i in 0..n {
        for j in (i + 1)..n {
            if correlations[i * n + j].abs() > threshold {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate a cycle graph C_n
///
/// A cycle connects vertices in a ring: 0-1-2-...-n-0.
///
/// # Arguments
///
/// * `n` - Number of vertices
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::basic::cycle_graph;
///
/// let g = cycle_graph(5);
/// assert_eq!(g.num_vertices(), 5);
/// assert_eq!(g.num_edges(), 5);
/// ```
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

/// Generate a Dart graph
///
/// The dart graph is a 5-vertex graph with specific structure.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::basic::dart_graph;
///
/// let g = dart_graph();
/// assert_eq!(g.num_vertices(), 5);
/// ```
pub fn dart_graph() -> Graph {
    let mut g = Graph::new(5);
    // Structure: triangle 0-1-2-0 with pendant edges from 2 to 3 and 4
    g.add_edge(0, 1).unwrap();
    g.add_edge(1, 2).unwrap();
    g.add_edge(2, 0).unwrap();
    g.add_edge(2, 3).unwrap();
    g.add_edge(3, 4).unwrap();
    g
}

/// Generate a Diamond graph
///
/// The diamond graph is K_4 minus one edge.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::basic::diamond_graph;
///
/// let g = diamond_graph();
/// assert_eq!(g.num_vertices(), 4);
/// assert_eq!(g.num_edges(), 5);
/// ```
pub fn diamond_graph() -> Graph {
    let mut g = Graph::new(4);
    g.add_edge(0, 1).unwrap();
    g.add_edge(0, 2).unwrap();
    g.add_edge(1, 2).unwrap();
    g.add_edge(1, 3).unwrap();
    g.add_edge(2, 3).unwrap();
    g
}

/// Generate an empty graph with n vertices and no edges
///
/// # Arguments
///
/// * `n` - Number of vertices
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::basic::empty_graph;
///
/// let g = empty_graph(5);
/// assert_eq!(g.num_vertices(), 5);
/// assert_eq!(g.num_edges(), 0);
/// ```
pub fn empty_graph(n: usize) -> Graph {
    Graph::new(n)
}

/// Generate a Fork graph
///
/// The fork graph is a 5-vertex path-like graph.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::basic::fork_graph;
///
/// let g = fork_graph();
/// assert_eq!(g.num_vertices(), 5);
/// ```
pub fn fork_graph() -> Graph {
    let mut g = Graph::new(5);
    g.add_edge(0, 1).unwrap();
    g.add_edge(1, 2).unwrap();
    g.add_edge(2, 3).unwrap();
    g.add_edge(2, 4).unwrap();
    g
}

/// Generate a Gem graph
///
/// The gem graph is a 5-vertex graph (also known as fan graph).
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::basic::gem_graph;
///
/// let g = gem_graph();
/// assert_eq!(g.num_vertices(), 5);
/// ```
pub fn gem_graph() -> Graph {
    let mut g = Graph::new(5);
    // Path with extra connections
    g.add_edge(0, 1).unwrap();
    g.add_edge(1, 2).unwrap();
    g.add_edge(2, 3).unwrap();
    g.add_edge(3, 4).unwrap();
    g.add_edge(0, 2).unwrap();
    g
}

/// Generate a 2D grid graph (m x n)
///
/// Vertices arranged in a rectangular grid with edges to adjacent vertices.
///
/// # Arguments
///
/// * `m` - Number of rows
/// * `n` - Number of columns
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::basic::grid_2d_graph;
///
/// let g = grid_2d_graph(3, 4);
/// assert_eq!(g.num_vertices(), 12);
/// ```
pub fn grid_2d_graph(m: usize, n: usize) -> Graph {
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

/// Generate a general grid graph
///
/// Creates a grid graph with the given dimensions.
///
/// # Arguments
///
/// * `dims` - Dimensions of the grid
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::basic::grid_graph;
///
/// let g = grid_graph(&[3, 4]);
/// assert_eq!(g.num_vertices(), 12);
/// ```
pub fn grid_graph(dims: &[usize]) -> Graph {
    if dims.is_empty() {
        return Graph::new(0);
    }

    if dims.len() == 1 {
        return path_graph(dims[0]);
    }

    if dims.len() == 2 {
        return grid_2d_graph(dims[0], dims[1]);
    }

    // For higher dimensions, use Cartesian product approach
    let n: usize = dims.iter().product();
    let mut g = Graph::new(n);

    for vertex in 0..n {
        // Convert linear index to multi-dimensional coordinates
        let mut coords = vec![0; dims.len()];
        let mut temp = vertex;
        for (i, &dim) in dims.iter().enumerate().rev() {
            coords[i] = temp % dim;
            temp /= dim;
        }

        // Connect to neighbors in each dimension
        for d in 0..dims.len() {
            if coords[d] + 1 < dims[d] {
                let mut neighbor_coords = coords.clone();
                neighbor_coords[d] += 1;

                // Convert back to linear index
                let mut neighbor = 0;
                let mut multiplier = 1;
                for i in (0..dims.len()).rev() {
                    neighbor += neighbor_coords[i] * multiplier;
                    multiplier *= dims[i];
                }

                g.add_edge(vertex, neighbor).unwrap();
            }
        }
    }

    g
}

/// Generate a House graph
///
/// The house graph is a 5-vertex graph shaped like a house.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::basic::house_graph;
///
/// let g = house_graph();
/// assert_eq!(g.num_vertices(), 5);
/// assert_eq!(g.num_edges(), 6);
/// ```
pub fn house_graph() -> Graph {
    let mut g = Graph::new(5);
    // Square base
    g.add_edge(0, 1).unwrap();
    g.add_edge(1, 2).unwrap();
    g.add_edge(2, 3).unwrap();
    g.add_edge(3, 0).unwrap();
    // Roof
    g.add_edge(2, 4).unwrap();
    g.add_edge(3, 4).unwrap();
    g
}

/// Generate a House-X graph
///
/// The house-X graph is a house graph with two diagonals in the square base.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::basic::house_x_graph;
///
/// let g = house_x_graph();
/// assert_eq!(g.num_vertices(), 5);
/// assert_eq!(g.num_edges(), 8);
/// ```
pub fn house_x_graph() -> Graph {
    let mut g = house_graph();
    // Add diagonals
    g.add_edge(0, 2).unwrap();
    g.add_edge(1, 3).unwrap();
    g
}

/// Generate a Ladder graph
///
/// A ladder graph is the Cartesian product of two path graphs.
///
/// # Arguments
///
/// * `n` - Number of rungs
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::basic::ladder_graph;
///
/// let g = ladder_graph(5);
/// assert_eq!(g.num_vertices(), 10);
/// ```
pub fn ladder_graph(n: usize) -> Graph {
    let mut g = Graph::new(2 * n);

    // Two parallel paths
    for i in 0..(n - 1) {
        g.add_edge(i, i + 1).unwrap();
        g.add_edge(n + i, n + i + 1).unwrap();
    }

    // Rungs connecting the paths
    for i in 0..n {
        g.add_edge(i, n + i).unwrap();
    }

    g
}

/// Generate a Möbius Ladder graph
///
/// A Möbius ladder is a cubic circulant graph.
///
/// # Arguments
///
/// * `n` - Number of rungs (must be even)
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::basic::moebius_ladder_graph;
///
/// let g = moebius_ladder_graph(4);
/// assert_eq!(g.num_vertices(), 8);
/// ```
pub fn moebius_ladder_graph(n: usize) -> Graph {
    if n % 2 != 0 {
        panic!("Möbius ladder requires an even number of rungs");
    }

    let mut g = Graph::new(2 * n);

    // Create two cycles
    for i in 0..n {
        g.add_edge(i, (i + 1) % n).unwrap();
        g.add_edge(n + i, n + (i + 1) % n).unwrap();
    }

    // Add twisted rungs
    for i in 0..n {
        if i < n / 2 {
            g.add_edge(i, n + i).unwrap();
        } else {
            g.add_edge(i, n + (i + n / 2) % n).unwrap();
        }
    }

    g
}

/// Generate a path graph P_n
///
/// A path connects vertices in a line: 0-1-2-...-n.
///
/// # Arguments
///
/// * `n` - Number of vertices
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::basic::path_graph;
///
/// let g = path_graph(5);
/// assert_eq!(g.num_vertices(), 5);
/// assert_eq!(g.num_edges(), 4);
/// ```
pub fn path_graph(n: usize) -> Graph {
    let mut g = Graph::new(n);

    for i in 0..(n.saturating_sub(1)) {
        g.add_edge(i, i + 1).unwrap();
    }

    g
}

/// Generate a star graph S_n
///
/// One central vertex connected to n outer vertices.
///
/// # Arguments
///
/// * `n` - Number of leaves
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::basic::star_graph;
///
/// let g = star_graph(5);
/// assert_eq!(g.num_vertices(), 6);
/// assert_eq!(g.num_edges(), 5);
/// ```
pub fn star_graph(n: usize) -> Graph {
    let mut g = Graph::new(n + 1);

    for i in 1..=n {
        g.add_edge(0, i).unwrap();
    }

    g
}

/// Generate a Toroidal 6-regular 2D grid graph
///
/// Creates a toroidal grid where each vertex has degree 6.
///
/// # Arguments
///
/// * `m` - Number of rows
/// * `n` - Number of columns
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::basic::toroidal_6_regular_grid_2d_graph;
///
/// let g = toroidal_6_regular_grid_2d_graph(3, 3);
/// assert_eq!(g.num_vertices(), 9);
/// ```
pub fn toroidal_6_regular_grid_2d_graph(m: usize, n: usize) -> Graph {
    let mut g = toroidal_grid_2d_graph(m, n);

    // Add diagonal connections to make it 6-regular
    for i in 0..m {
        for j in 0..n {
            let v = i * n + j;
            let v_diag1 = ((i + 1) % m) * n + ((j + 1) % n);
            let v_diag2 = ((i + 1) % m) * n + ((j + n - 1) % n);

            g.add_edge(v, v_diag1).unwrap();
            g.add_edge(v, v_diag2).unwrap();
        }
    }

    g
}

/// Generate a Toroidal 2D grid graph
///
/// A toroidal grid wraps around at the edges.
///
/// # Arguments
///
/// * `m` - Number of rows
/// * `n` - Number of columns
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::basic::toroidal_grid_2d_graph;
///
/// let g = toroidal_grid_2d_graph(3, 4);
/// assert_eq!(g.num_vertices(), 12);
/// ```
pub fn toroidal_grid_2d_graph(m: usize, n: usize) -> Graph {
    let mut g = Graph::new(m * n);

    for i in 0..m {
        for j in 0..n {
            let v = i * n + j;

            // Connect to right neighbor (with wraparound)
            let right = i * n + (j + 1) % n;
            g.add_edge(v, right).unwrap();

            // Connect to bottom neighbor (with wraparound)
            let bottom = ((i + 1) % m) * n + j;
            g.add_edge(v, bottom).unwrap();
        }
    }

    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bull_graph() {
        let g = bull_graph();
        assert_eq!(g.num_vertices(), 5);
        assert_eq!(g.num_edges(), 5);
    }

    #[test]
    fn test_butterfly_graph() {
        let g = butterfly_graph();
        assert_eq!(g.num_vertices(), 5);
        assert_eq!(g.num_edges(), 6);
    }

    #[test]
    fn test_claw_graph() {
        let g = claw_graph();
        assert_eq!(g.num_vertices(), 4);
        assert_eq!(g.num_edges(), 3);
        assert_eq!(g.degree(0), Some(3));
    }

    #[test]
    fn test_complete_graph() {
        let g = complete_graph(5);
        assert_eq!(g.num_vertices(), 5);
        assert_eq!(g.num_edges(), 10);
    }

    #[test]
    fn test_complete_multipartite() {
        let g = complete_multipartite_graph(&[2, 3, 2]);
        assert_eq!(g.num_vertices(), 7);
    }

    #[test]
    fn test_diamond_graph() {
        let g = diamond_graph();
        assert_eq!(g.num_vertices(), 4);
        assert_eq!(g.num_edges(), 5);
    }

    #[test]
    fn test_house_graph() {
        let g = house_graph();
        assert_eq!(g.num_vertices(), 5);
        assert_eq!(g.num_edges(), 6);
    }

    #[test]
    fn test_house_x_graph() {
        let g = house_x_graph();
        assert_eq!(g.num_vertices(), 5);
        assert_eq!(g.num_edges(), 8);
    }

    #[test]
    fn test_ladder_graph() {
        let g = ladder_graph(5);
        assert_eq!(g.num_vertices(), 10);
        assert_eq!(g.num_edges(), 13);
    }

    #[test]
    fn test_grid_graph() {
        let g = grid_graph(&[3, 4]);
        assert_eq!(g.num_vertices(), 12);

        let g3d = grid_graph(&[2, 3, 4]);
        assert_eq!(g3d.num_vertices(), 24);
    }

    #[test]
    fn test_toroidal_grid() {
        let g = toroidal_grid_2d_graph(3, 4);
        assert_eq!(g.num_vertices(), 12);
        // Each vertex should have degree 4 in toroidal grid
        for i in 0..12 {
            assert_eq!(g.degree(i), Some(4));
        }
    }
}
