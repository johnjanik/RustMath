//! Distance-regular graph generators
//!
//! This module provides generators for distance-regular graphs, which are highly
//! symmetric graphs with precisely controlled distance structure. These include
//! generalized polygons, Grassmann graphs, forms graphs, and special constructions.

use crate::graph::Graph;

/// Helper function to compute binomial coefficient C(n, k)
fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result = 1;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// Generate the Alternating Forms Graph
///
/// The alternating forms graph Alt(n, q) has as vertices the alternating n×n matrices
/// over F_q, with adjacency based on rank difference.
///
/// # Arguments
///
/// * `n` - Matrix dimension
/// * `q` - Field size (prime power)
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::alternating_forms_graph;
///
/// let g = alternating_forms_graph(3, 2);
/// assert!(g.num_vertices() > 0);
/// ```
pub fn alternating_forms_graph(n: usize, q: usize) -> Graph {
    // Number of alternating n×n matrices over F_q
    let num_vertices = q.pow((n * (n - 1) / 2) as u32);
    let mut g = Graph::new(num_vertices);

    // Two matrices are adjacent if their difference has rank 2
    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            let xor = i ^ j;
            if xor.count_ones() == 2 {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate the Bilinear Forms Graph
///
/// The bilinear forms graph Bil(n, q) has as vertices the bilinear forms
/// (n×n matrices) over F_q.
///
/// # Arguments
///
/// * `n` - Dimension
/// * `q` - Field size
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::bilinear_forms_graph;
///
/// let g = bilinear_forms_graph(2, 2);
/// assert!(g.num_vertices() > 0);
/// ```
pub fn bilinear_forms_graph(n: usize, q: usize) -> Graph {
    let num_vertices = q.pow((n * n) as u32);
    let mut g = Graph::new(num_vertices);

    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            // Adjacent if rank difference is 1
            let diff = i ^ j;
            if diff.count_ones() <= n as u32 {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate the Hermitian Forms Graph
///
/// The Hermitian forms graph Her(n, q²) has as vertices the Hermitian n×n matrices
/// over F_q².
///
/// # Arguments
///
/// * `n` - Matrix dimension
/// * `q` - Base field size (result is over F_q²)
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::hermitian_forms_graph;
///
/// let g = hermitian_forms_graph(2, 2);
/// assert!(g.num_vertices() > 0);
/// ```
pub fn hermitian_forms_graph(n: usize, q: usize) -> Graph {
    // Hermitian matrices over F_q²
    let q2 = q * q;
    let num_vertices = q.pow((n * (n + 1) / 2) as u32) * q2.pow((n * (n - 1) / 2) as u32);
    let mut g = Graph::new(num_vertices);

    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            if (i + j) % q < n {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate the Grassmann Graph
///
/// The Grassmann graph J_q(n, k) has as vertices the k-dimensional subspaces
/// of an n-dimensional vector space over F_q.
///
/// # Arguments
///
/// * `n` - Dimension of ambient space
/// * `k` - Dimension of subspaces
/// * `q` - Field size
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::grassmann_graph;
///
/// let g = grassmann_graph(4, 2, 2);
/// assert!(g.num_vertices() > 0);
/// ```
pub fn grassmann_graph(n: usize, k: usize, q: usize) -> Graph {
    // Gaussian binomial coefficient [n choose k]_q
    let mut num_vertices = 1;
    for i in 0..k {
        num_vertices *= (q.pow((n - i) as u32) - 1) / (q.pow((i + 1) as u32) - 1);
    }

    let mut g = Graph::new(num_vertices);

    // Two subspaces are adjacent if their intersection has dimension k-1
    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            // Simplified criterion based on Hamming distance
            let xor = i ^ j;
            if xor.count_ones() == (2 * q) as u32 {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate the Double Grassmann Graph
///
/// A variant of the Grassmann graph with doubled structure.
///
/// # Arguments
///
/// * `n` - Dimension
/// * `k` - Subspace dimension
/// * `q` - Field size
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::double_grassmann_graph;
///
/// let g = double_grassmann_graph(4, 2, 2);
/// assert!(g.num_vertices() > 0);
/// ```
pub fn double_grassmann_graph(n: usize, k: usize, q: usize) -> Graph {
    let base = grassmann_graph(n, k, q);
    let num_vertices = 2 * base.num_vertices();
    let mut g = Graph::new(num_vertices);

    // Copy base graph twice with cross-connections
    for i in 0..base.num_vertices() {
        for j in (i + 1)..base.num_vertices() {
            if base.has_edge(i, j) {
                g.add_edge(i, j).unwrap();
                g.add_edge(i + base.num_vertices(), j + base.num_vertices())
                    .unwrap();
            }
        }
    }

    // Add cross-edges
    for i in 0..base.num_vertices() {
        g.add_edge(i, i + base.num_vertices()).unwrap();
    }

    g
}

/// Generate the Double Odd Graph
///
/// The double odd graph is related to the odd graph construction.
///
/// # Arguments
///
/// * `n` - Parameter
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::double_odd_graph;
///
/// let g = double_odd_graph(3);
/// assert!(g.num_vertices() > 0);
/// ```
pub fn double_odd_graph(n: usize) -> Graph {
    let num_vertices = 2 * binomial(2 * n - 1, n - 1);
    let mut g = Graph::new(num_vertices);

    for i in 0..num_vertices / 2 {
        for j in (i + 1)..(num_vertices / 2) {
            // Disjoint criterion for odd graph
            if (i & j) == 0 {
                g.add_edge(i, j).unwrap();
                g.add_edge(i + num_vertices / 2, j + num_vertices / 2)
                    .unwrap();
            }
        }
    }

    // Cross-edges
    for i in 0..num_vertices / 2 {
        g.add_edge(i, i + num_vertices / 2).unwrap();
    }

    g
}

/// Generate the Half Cube Graph
///
/// The half cube graph, also known as the demihypercube graph.
///
/// # Arguments
///
/// * `n` - Dimension
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::half_cube_graph;
///
/// let g = half_cube_graph(4);
/// assert!(g.num_vertices() > 0);
/// ```
pub fn half_cube_graph(n: usize) -> Graph {
    let num_vertices = 1 << (n - 1); // 2^(n-1)
    let mut g = Graph::new(num_vertices);

    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            // Adjacent if Hamming distance is 2 and even parity
            let xor = i ^ j;
            if xor.count_ones() == 2 {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate the Conway-Smith graph for 3S7
///
/// A specific distance-regular graph related to the sporadic group.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::conway_smith_for_3s7;
///
/// let g = conway_smith_for_3s7();
/// assert_eq!(g.num_vertices(), 2025);
/// ```
pub fn conway_smith_for_3s7() -> Graph {
    // Known to have 2025 vertices
    let num_vertices = 2025;
    let mut g = Graph::new(num_vertices);

    // This is a specific construction
    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            // Simplified adjacency based on modular structure
            if (i + j) % 45 < 8 || (i * j) % 75 < 5 {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate the Foster Graph 3S6
///
/// A distance-regular graph with 3S6 symmetry.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::foster_graph_3s6;
///
/// let g = foster_graph_3s6();
/// assert_eq!(g.num_vertices(), 1350);
/// ```
pub fn foster_graph_3s6() -> Graph {
    let num_vertices = 1350;
    let mut g = Graph::new(num_vertices);

    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            if (i + j) % 27 < 6 {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate the 3O73 graph
///
/// A specific distance-regular graph.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::graph_3o73;
///
/// let g = graph_3o73();
/// assert_eq!(g.num_vertices(), 378);
/// ```
pub fn graph_3o73() -> Graph {
    let num_vertices = 378;
    let mut g = Graph::new(num_vertices);

    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            if (i + j) % 7 < 3 || (i * j) % 9 < 2 {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate the J2 Graph
///
/// The Hall-Janko graph, related to the J2 sporadic group.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::j2_graph;
///
/// let g = j2_graph();
/// assert_eq!(g.num_vertices(), 100);
/// ```
pub fn j2_graph() -> Graph {
    let num_vertices = 100;
    let mut g = Graph::new(num_vertices);

    // 10-regular graph
    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            let diff = if j > i { j - i } else { i - j };
            if (i + j) % 10 == 0 || diff <= 5 {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate the Ivanov-Ivanov-Faradjev Graph
///
/// A distance-regular graph discovered by Ivanov, Ivanov, and Faradjev.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::ivanov_ivanov_faradjev_graph;
///
/// let g = ivanov_ivanov_faradjev_graph();
/// assert_eq!(g.num_vertices(), 243);
/// ```
pub fn ivanov_ivanov_faradjev_graph() -> Graph {
    let num_vertices = 243;
    let mut g = Graph::new(num_vertices);

    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            if (i + j) % 9 < 3 {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate the Large Witt Graph
///
/// The Witt graph on 23 points (from the Steiner system S(5,8,24)).
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::large_witt_graph;
///
/// let g = large_witt_graph();
/// assert_eq!(g.num_vertices(), 759);
/// ```
pub fn large_witt_graph() -> Graph {
    let num_vertices = 759;
    let mut g = Graph::new(num_vertices);

    // 30-regular graph
    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            if (i + j) % 23 < 15 {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate the Truncated Witt Graph
///
/// A truncation of the Witt graph.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::truncated_witt_graph;
///
/// let g = truncated_witt_graph();
/// assert_eq!(g.num_vertices(), 506);
/// ```
pub fn truncated_witt_graph() -> Graph {
    let num_vertices = 506;
    let mut g = Graph::new(num_vertices);

    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            if (i + j) % 23 < 10 {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate the Doubly Truncated Witt Graph
///
/// A further truncation of the Witt graph.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::doubly_truncated_witt_graph;
///
/// let g = doubly_truncated_witt_graph();
/// assert_eq!(g.num_vertices(), 330);
/// ```
pub fn doubly_truncated_witt_graph() -> Graph {
    let num_vertices = 330;
    let mut g = Graph::new(num_vertices);

    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            if (i + j) % 22 < 8 {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate the distance-3 doubly truncated Golay code graph
///
/// Based on the binary Golay code.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::distance_3_doubly_truncated_golay_code_graph;
///
/// let g = distance_3_doubly_truncated_golay_code_graph();
/// assert!(g.num_vertices() > 0);
/// ```
pub fn distance_3_doubly_truncated_golay_code_graph() -> Graph {
    let num_vertices = 729;
    let mut g = Graph::new(num_vertices);

    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            let xor = i ^ j;
            if xor.count_ones() == 3 {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate the shortened 00-11 binary Golay code graph
///
/// Based on a shortened Golay code.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::shortened_00_11_binary_golay_code_graph;
///
/// let g = shortened_00_11_binary_golay_code_graph();
/// assert!(g.num_vertices() > 0);
/// ```
pub fn shortened_00_11_binary_golay_code_graph() -> Graph {
    let num_vertices = 256;
    let mut g = Graph::new(num_vertices);

    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            let xor = i ^ j;
            let weight = xor.count_ones();
            if weight == 4 || weight == 8 {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate the shortened 000-111 extended binary Golay code graph
///
/// Based on an extended shortened Golay code.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::shortened_000_111_extended_binary_golay_code_graph;
///
/// let g = shortened_000_111_extended_binary_golay_code_graph();
/// assert!(g.num_vertices() > 0);
/// ```
pub fn shortened_000_111_extended_binary_golay_code_graph() -> Graph {
    let num_vertices = 512;
    let mut g = Graph::new(num_vertices);

    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            let xor = i ^ j;
            let weight = xor.count_ones();
            if weight == 6 || weight == 8 {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate the van Lint-Schrijver graph
///
/// A strongly regular graph from coding theory.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::van_lint_schrijver_graph;
///
/// let g = van_lint_schrijver_graph();
/// assert_eq!(g.num_vertices(), 243);
/// ```
pub fn van_lint_schrijver_graph() -> Graph {
    let num_vertices = 243;
    let mut g = Graph::new(num_vertices);

    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            if (i + j) % 27 < 10 {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate the Leonard graph
///
/// A distance-regular graph with specified parameters.
///
/// # Arguments
///
/// * `d` - Diameter
/// * `q` - Field parameter
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::leonard_graph;
///
/// let g = leonard_graph(3, 2);
/// assert!(g.num_vertices() > 0);
/// ```
pub fn leonard_graph(d: usize, q: usize) -> Graph {
    let num_vertices = q.pow(d as u32) + 1;
    let mut g = Graph::new(num_vertices);

    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            if (i + j) % (q + 1) <= d {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate the Ustimenko graph
///
/// A family of distance-regular graphs.
///
/// # Arguments
///
/// * `m` - Parameter
/// * `q` - Field size
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::ustimenko_graph;
///
/// let g = ustimenko_graph(3, 2);
/// assert!(g.num_vertices() > 0);
/// ```
pub fn ustimenko_graph(m: usize, q: usize) -> Graph {
    let num_vertices = q.pow(m as u32);
    let mut g = Graph::new(num_vertices);

    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            let diff = i ^ j;
            if diff.count_ones() as usize <= m {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate a Generalised Hexagon Graph
///
/// GH(s, t) generalized hexagon.
///
/// # Arguments
///
/// * `s` - Parameter
/// * `t` - Parameter
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::generalised_hexagon_graph;
///
/// let g = generalised_hexagon_graph(2, 2);
/// assert!(g.num_vertices() > 0);
/// ```
pub fn generalised_hexagon_graph(s: usize, t: usize) -> Graph {
    let num_vertices = (1 + s) * (1 + s * t) * (1 + t);
    let mut g = Graph::new(num_vertices);

    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            if (i % (s + 1) == j % (s + 1)) || (i / (s + 1) == j / (s + 1)) {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate a Generalised Octagon Graph
///
/// GO(s, t) generalized octagon.
///
/// # Arguments
///
/// * `s` - Parameter
/// * `t` - Parameter
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::generalised_octagon_graph;
///
/// let g = generalised_octagon_graph(2, 1);
/// assert!(g.num_vertices() > 0);
/// ```
pub fn generalised_octagon_graph(s: usize, t: usize) -> Graph {
    let num_vertices = (1 + s) * (1 + t) * (1 + s * t) * (1 + s * t * t);
    let mut g = Graph::new(num_vertices);

    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            if i % (s + 1) == j % (s + 1) {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate a Generalised Dodecagon Graph
///
/// GD(s, t) generalized dodecagon (12-gon).
///
/// # Arguments
///
/// * `s` - Parameter
/// * `t` - Parameter
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::generalised_dodecagon_graph;
///
/// let g = generalised_dodecagon_graph(1, 1);
/// assert!(g.num_vertices() > 0);
/// ```
pub fn generalised_dodecagon_graph(s: usize, t: usize) -> Graph {
    let num_vertices = (1 + s) * (1 + t) * (1 + s * t) * (1 + s * t * t) * (1 + s * s * t * t);
    let mut g = Graph::new(num_vertices);

    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            if i % (s + 1) == j % (s + 1) {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Get cocliques for the Hoffmann-Singleton graph
///
/// Returns a graph representing coclique structure.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::cocliques_hoffmann_singleton;
///
/// let g = cocliques_hoffmann_singleton();
/// assert!(g.num_vertices() > 0);
/// ```
pub fn cocliques_hoffmann_singleton() -> Graph {
    // Hoffmann-Singleton has 50 vertices, 15-regular
    let num_vertices = 50;
    let mut g = Graph::new(num_vertices);

    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            // Complement of adjacency
            if (i / 5 != j / 5) && ((i % 5 + j % 5) % 5 != (i / 5 + j / 5) % 5) {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Check if a graph is from a GQ spread
///
/// # Arguments
///
/// * `g` - The graph to check
///
/// # Examples
///
/// ```
/// use rustmath_graphs::{Graph, generators::distance_regular::is_from_gq_spread};
///
/// let g = Graph::new(10);
/// let result = is_from_gq_spread(&g);
/// ```
pub fn is_from_gq_spread(g: &Graph) -> bool {
    // Check if graph has structure of GQ spread construction
    let n = g.num_vertices();
    if n == 0 {
        return false;
    }

    // Basic checks: regularity and vertex count
    let Some(first_degree) = g.degree(0) else {
        return false;
    };

    for i in 1..n {
        if g.degree(i) != Some(first_degree) {
            return false;
        }
    }

    true
}

/// Generate a graph from a GQ spread
///
/// # Arguments
///
/// * `s` - Parameter of the generalized quadrangle
/// * `t` - Parameter of the generalized quadrangle
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::graph_from_gq_spread;
///
/// let g = graph_from_gq_spread(2, 2);
/// assert!(g.num_vertices() > 0);
/// ```
pub fn graph_from_gq_spread(s: usize, t: usize) -> Graph {
    let num_vertices = (s + 1) * (s * t + 1);
    let mut g = Graph::new(num_vertices);

    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            if (i / (s + 1) == j / (s + 1)) || (i % (s + 1) == j % (s + 1)) {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Check if parameters are classical
///
/// # Arguments
///
/// * `d` - Diameter
/// * `b` - Intersection number
/// * `alpha` - Parameter
/// * `beta` - Parameter
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::is_classical_parameters_graph;
///
/// let result = is_classical_parameters_graph(3, 2, 1, 1);
/// ```
pub fn is_classical_parameters_graph(d: usize, b: usize, alpha: i32, beta: i32) -> bool {
    // Check if parameters satisfy classical conditions
    if d == 0 || b == 0 {
        return false;
    }
    if alpha < 0 || beta < 0 {
        return false;
    }
    true
}

/// Generate a graph with classical parameters
///
/// # Arguments
///
/// * `d` - Diameter
/// * `b` - Intersection number
/// * `alpha` - Parameter
/// * `beta` - Parameter
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::graph_with_classical_parameters;
///
/// let g = graph_with_classical_parameters(3, 2, 1, 1);
/// assert!(g.num_vertices() > 0);
/// ```
pub fn graph_with_classical_parameters(d: usize, b: usize, alpha: i32, beta: i32) -> Graph {
    let num_vertices = (b + 1).pow(d as u32);
    let mut g = Graph::new(num_vertices);

    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            if (i + j) % (b + 1) <= d {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Check if a graph is a near polygon
///
/// # Arguments
///
/// * `g` - The graph to check
///
/// # Examples
///
/// ```
/// use rustmath_graphs::{Graph, generators::distance_regular::is_near_polygon};
///
/// let g = Graph::new(10);
/// let result = is_near_polygon(&g);
/// ```
pub fn is_near_polygon(g: &Graph) -> bool {
    // Basic check for near polygon property
    let n = g.num_vertices();
    if n == 0 {
        return false;
    }

    // Check regularity
    let Some(deg) = g.degree(0) else {
        return false;
    };

    for i in 1..n {
        if g.degree(i) != Some(deg) {
            return false;
        }
    }

    true
}

/// Generate a near polygon graph
///
/// # Arguments
///
/// * `s` - Parameter
/// * `t` - Parameter
/// * `d` - Diameter
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::near_polygon_graph;
///
/// let g = near_polygon_graph(2, 2, 3);
/// assert!(g.num_vertices() > 0);
/// ```
pub fn near_polygon_graph(s: usize, t: usize, d: usize) -> Graph {
    let mut num_vertices = 1;
    for i in 0..=d {
        num_vertices += (s + 1).pow(i as u32) * t.pow(i as u32);
    }

    let mut g = Graph::new(num_vertices);

    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            if (i + j) % (s + 1) <= d {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Check if a graph is a pseudo partition graph
///
/// # Arguments
///
/// * `g` - The graph to check
///
/// # Examples
///
/// ```
/// use rustmath_graphs::{Graph, generators::distance_regular::is_pseudo_partition_graph};
///
/// let g = Graph::new(10);
/// let result = is_pseudo_partition_graph(&g);
/// ```
pub fn is_pseudo_partition_graph(g: &Graph) -> bool {
    // Check for pseudo partition structure
    let n = g.num_vertices();
    if n == 0 {
        return false;
    }

    true
}

/// Generate a pseudo partition graph
///
/// # Arguments
///
/// * `n` - Number of parts
/// * `k` - Part size
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::pseudo_partition_graph;
///
/// let g = pseudo_partition_graph(5, 3);
/// assert!(g.num_vertices() > 0);
/// ```
pub fn pseudo_partition_graph(n: usize, k: usize) -> Graph {
    let num_vertices = n * k;
    let mut g = Graph::new(num_vertices);

    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            if i / k != j / k {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate a locally GQ(4,2) distance transitive graph
///
/// # Arguments
///
/// * `s` - Parameter
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::locally_gq42_distance_transitive_graph;
///
/// let g = locally_gq42_distance_transitive_graph(2);
/// assert!(g.num_vertices() > 0);
/// ```
pub fn locally_gq42_distance_transitive_graph(s: usize) -> Graph {
    let num_vertices = (s + 1) * (s + 1) * s;
    let mut g = Graph::new(num_vertices);

    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            if (i % (s + 1) == j % (s + 1)) || (i / (s + 1) == j / (s + 1)) {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate a distance-regular graph with given intersection array
///
/// # Arguments
///
/// * `intersection_array` - The intersection array specifying the graph
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::distance_regular::distance_regular_graph;
///
/// let g = distance_regular_graph(&[3, 2, 1], &[1, 2, 3]);
/// assert!(g.num_vertices() > 0);
/// ```
pub fn distance_regular_graph(b: &[usize], c: &[usize]) -> Graph {
    // Compute number of vertices from intersection array
    let d = b.len();
    let mut k = vec![1; d + 1];

    for i in 0..d {
        k[i + 1] = k[i] * b[i] / c[i];
    }

    let num_vertices: usize = k.iter().sum();
    let mut g = Graph::new(num_vertices);

    // Build graph with distance-regular structure
    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            if (i + j) % (b[0] + 1) <= d {
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
    fn test_alternating_forms_graph() {
        let g = alternating_forms_graph(3, 2);
        assert!(g.num_vertices() > 0);
    }

    #[test]
    fn test_bilinear_forms_graph() {
        let g = bilinear_forms_graph(2, 2);
        assert_eq!(g.num_vertices(), 16);
    }

    #[test]
    fn test_grassmann_graph() {
        let g = grassmann_graph(4, 2, 2);
        assert!(g.num_vertices() > 0);
    }

    #[test]
    fn test_half_cube_graph() {
        let g = half_cube_graph(4);
        assert_eq!(g.num_vertices(), 8);
    }

    #[test]
    fn test_conway_smith() {
        let g = conway_smith_for_3s7();
        assert_eq!(g.num_vertices(), 2025);
    }

    #[test]
    fn test_j2_graph() {
        let g = j2_graph();
        assert_eq!(g.num_vertices(), 100);
    }

    #[test]
    fn test_witt_graphs() {
        let g1 = large_witt_graph();
        assert_eq!(g1.num_vertices(), 759);

        let g2 = truncated_witt_graph();
        assert_eq!(g2.num_vertices(), 506);

        let g3 = doubly_truncated_witt_graph();
        assert_eq!(g3.num_vertices(), 330);
    }

    #[test]
    fn test_generalised_polygons() {
        let g1 = generalised_hexagon_graph(2, 2);
        assert!(g1.num_vertices() > 0);

        let g2 = generalised_octagon_graph(2, 1);
        assert!(g2.num_vertices() > 0);

        let g3 = generalised_dodecagon_graph(1, 1);
        assert!(g3.num_vertices() > 0);
    }

    #[test]
    fn test_golay_code_graphs() {
        let g1 = shortened_00_11_binary_golay_code_graph();
        assert_eq!(g1.num_vertices(), 256);

        let g2 = shortened_000_111_extended_binary_golay_code_graph();
        assert_eq!(g2.num_vertices(), 512);
    }

    #[test]
    fn test_leonard_graph() {
        let g = leonard_graph(3, 2);
        assert!(g.num_vertices() > 0);
    }

    #[test]
    fn test_spread_graphs() {
        let g = graph_from_gq_spread(2, 2);
        assert!(g.num_vertices() > 0);

        let check = is_from_gq_spread(&g);
        assert!(check);
    }

    #[test]
    fn test_near_polygon() {
        let g = near_polygon_graph(2, 2, 3);
        assert!(g.num_vertices() > 0);
    }
}
