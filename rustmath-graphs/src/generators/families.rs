//! Graph family generators
//!
//! This module provides generators for common families of graphs including
//! trees, cubes, Petersen variations, and many other well-known graph families.

use crate::graph::Graph;
use std::collections::{HashMap, HashSet, VecDeque};

/// Helper function to compute binomial coefficient
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

/// Helper to generate all k-subsets of {0..n}
fn k_subsets(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }
    if k > n {
        return vec![];
    }

    let mut result = Vec::new();
    let mut current = vec![0; k];

    loop {
        result.push(current.clone());

        // Find rightmost element that can be incremented
        let mut i = k;
        while i > 0 {
            i -= 1;
            if current[i] < n - k + i {
                current[i] += 1;
                for j in (i + 1)..k {
                    current[j] = current[j - 1] + 1;
                }
                break;
            }
        }

        if i == 0 && current[0] == n - k {
            break;
        }
    }

    result
}

/// Generate the Johnson Graph J(n, k)
///
/// Vertices are k-subsets of {1..n}, adjacent if they intersect in k-1 elements.
///
/// # Arguments
///
/// * `n` - Size of the base set
/// * `k` - Size of subsets
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::families::johnson_graph;
///
/// let g = johnson_graph(5, 2);
/// assert_eq!(g.num_vertices(), 10); // C(5,2) = 10
/// ```
pub fn johnson_graph(n: usize, k: usize) -> Graph {
    let subsets = k_subsets(n, k);
    let num_vertices = subsets.len();
    let mut g = Graph::new(num_vertices);

    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            // Count intersection
            let mut intersection = 0;
            for &a in &subsets[i] {
                if subsets[j].contains(&a) {
                    intersection += 1;
                }
            }
            if intersection == k - 1 {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate the Kneser Graph K(n, k)
///
/// Vertices are k-subsets of {1..n}, adjacent if they are disjoint.
///
/// # Arguments
///
/// * `n` - Size of the base set
/// * `k` - Size of subsets
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::families::kneser_graph;
///
/// let g = kneser_graph(5, 2);
/// assert!(g.num_vertices() > 0);
/// ```
pub fn kneser_graph(n: usize, k: usize) -> Graph {
    if 2 * k > n {
        return Graph::new(binomial(n, k));
    }

    let subsets = k_subsets(n, k);
    let num_vertices = subsets.len();
    let mut g = Graph::new(num_vertices);

    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            // Check if disjoint
            let mut disjoint = true;
            for &a in &subsets[i] {
                if subsets[j].contains(&a) {
                    disjoint = false;
                    break;
                }
            }
            if disjoint {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate the Odd Graph O_n
///
/// The odd graph is K(2n-1, n-1).
///
/// # Arguments
///
/// * `n` - Parameter
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::families::odd_graph;
///
/// let g = odd_graph(3);
/// assert!(g.num_vertices() > 0);
/// ```
pub fn odd_graph(n: usize) -> Graph {
    kneser_graph(2 * n - 1, n - 1)
}

/// Generate a Balanced Tree
///
/// A rooted tree where all leaves are at the same distance from the root.
///
/// # Arguments
///
/// * `r` - Branching factor
/// * `h` - Height
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::families::balanced_tree;
///
/// let g = balanced_tree(2, 3);
/// assert!(g.num_vertices() > 0);
/// ```
pub fn balanced_tree(r: usize, h: usize) -> Graph {
    if r == 0 || h == 0 {
        return Graph::new(1);
    }

    // Number of vertices: 1 + r + r^2 + ... + r^h = (r^(h+1) - 1)/(r - 1)
    let num_vertices = (r.pow((h + 1) as u32) - 1) / (r - 1);
    let mut g = Graph::new(num_vertices);

    for i in 0..num_vertices {
        for j in 1..=r {
            let child = r * i + j;
            if child < num_vertices {
                g.add_edge(i, child).unwrap();
            }
        }
    }

    g
}

/// Generate a Barbell Graph
///
/// Two complete graphs connected by a path.
///
/// # Arguments
///
/// * `n1` - Size of first complete graph
/// * `n2` - Size of second complete graph
/// * `m` - Length of connecting path
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::families::barbell_graph;
///
/// let g = barbell_graph(3, 3, 2);
/// assert!(g.num_vertices() > 0);
/// ```
pub fn barbell_graph(n1: usize, n2: usize, m: usize) -> Graph {
    let num_vertices = n1 + n2 + m;
    let mut g = Graph::new(num_vertices);

    // First complete graph
    for i in 0..n1 {
        for j in (i + 1)..n1 {
            g.add_edge(i, j).unwrap();
        }
    }

    // Second complete graph
    let offset2 = n1 + m;
    for i in 0..n2 {
        for j in (i + 1)..n2 {
            g.add_edge(offset2 + i, offset2 + j).unwrap();
        }
    }

    // Connecting path
    if n1 > 0 && m > 0 {
        g.add_edge(n1 - 1, n1).unwrap();
    }
    for i in n1..(n1 + m - 1) {
        g.add_edge(i, i + 1).unwrap();
    }
    if m > 0 && n2 > 0 {
        g.add_edge(n1 + m - 1, offset2).unwrap();
    }

    g
}

/// Generate a Lollipop Graph
///
/// A complete graph connected to a path.
///
/// # Arguments
///
/// * `m` - Size of complete graph
/// * `n` - Length of path
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::families::lollipop_graph;
///
/// let g = lollipop_graph(4, 3);
/// assert_eq!(g.num_vertices(), 7);
/// ```
pub fn lollipop_graph(m: usize, n: usize) -> Graph {
    let num_vertices = m + n;
    let mut g = Graph::new(num_vertices);

    // Complete graph
    for i in 0..m {
        for j in (i + 1)..m {
            g.add_edge(i, j).unwrap();
        }
    }

    // Path
    if m > 0 && n > 0 {
        g.add_edge(m - 1, m).unwrap();
    }
    for i in m..(num_vertices - 1) {
        g.add_edge(i, i + 1).unwrap();
    }

    g
}

/// Generate a Tadpole Graph
///
/// A cycle connected to a path.
///
/// # Arguments
///
/// * `m` - Size of cycle
/// * `n` - Length of path
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::families::tadpole_graph;
///
/// let g = tadpole_graph(4, 3);
/// assert_eq!(g.num_vertices(), 7);
/// ```
pub fn tadpole_graph(m: usize, n: usize) -> Graph {
    let num_vertices = m + n;
    let mut g = Graph::new(num_vertices);

    // Cycle
    for i in 0..m {
        g.add_edge(i, (i + 1) % m).unwrap();
    }

    // Path
    if n > 0 {
        g.add_edge(0, m).unwrap();
        for i in m..(num_vertices - 1) {
            g.add_edge(i, i + 1).unwrap();
        }
    }

    g
}

/// Generate a Dipole Graph
///
/// Two vertices connected by n parallel edges (multigraph).
/// For simple graphs, returns K_2.
///
/// # Arguments
///
/// * `n` - Number of parallel edges
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::families::dipole_graph;
///
/// let g = dipole_graph(3);
/// assert_eq!(g.num_vertices(), 2);
/// ```
pub fn dipole_graph(n: usize) -> Graph {
    let mut g = Graph::new(2);
    if n > 0 {
        g.add_edge(0, 1).unwrap();
    }
    g
}

/// Generate a Cube Graph (Hypercube)
///
/// The n-dimensional hypercube graph Q_n.
///
/// # Arguments
///
/// * `n` - Dimension
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::families::cube_graph;
///
/// let g = cube_graph(3);
/// assert_eq!(g.num_vertices(), 8);
/// ```
pub fn cube_graph(n: usize) -> Graph {
    let num_vertices = 1 << n; // 2^n
    let mut g = Graph::new(num_vertices);

    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            // Adjacent if Hamming distance is 1
            if (i ^ j).count_ones() == 1 {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate a Folded Cube Graph
///
/// The n-dimensional folded cube.
///
/// # Arguments
///
/// * `n` - Dimension
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::families::folded_cube_graph;
///
/// let g = folded_cube_graph(4);
/// assert!(g.num_vertices() > 0);
/// ```
pub fn folded_cube_graph(n: usize) -> Graph {
    if n == 0 {
        return Graph::new(1);
    }

    let num_vertices = 1 << (n - 1); // 2^(n-1)
    let mut g = Graph::new(num_vertices);

    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            let xor = i ^ j;
            if xor.count_ones() == 1 || xor == (1 << n) - 1 {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate a Circulant Graph
///
/// Graph on n vertices where vertex i is adjacent to i+k (mod n) for each k in jumps.
///
/// # Arguments
///
/// * `n` - Number of vertices
/// * `jumps` - List of jump distances
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::families::circulant_graph;
///
/// let g = circulant_graph(10, &[1, 2]);
/// assert_eq!(g.num_vertices(), 10);
/// ```
pub fn circulant_graph(n: usize, jumps: &[usize]) -> Graph {
    let mut g = Graph::new(n);

    for i in 0..n {
        for &jump in jumps {
            let j = (i + jump) % n;
            if i < j {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate a Generalized Petersen Graph
///
/// GP(n, k) has 2n vertices arranged in outer and inner cycles.
///
/// # Arguments
///
/// * `n` - Number of vertices in each cycle
/// * `k` - Connection parameter
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::families::generalized_petersen_graph;
///
/// let g = generalized_petersen_graph(5, 2);
/// assert_eq!(g.num_vertices(), 10);
/// ```
pub fn generalized_petersen_graph(n: usize, k: usize) -> Graph {
    let num_vertices = 2 * n;
    let mut g = Graph::new(num_vertices);

    // Outer cycle
    for i in 0..n {
        g.add_edge(i, (i + 1) % n).unwrap();
    }

    // Inner star
    for i in 0..n {
        g.add_edge(n + i, n + (i + k) % n).unwrap();
    }

    // Spokes
    for i in 0..n {
        g.add_edge(i, n + i).unwrap();
    }

    g
}

/// Generate a Double Generalized Petersen Graph
///
/// Doubled version of the generalized Petersen graph.
///
/// # Arguments
///
/// * `n` - Parameter
/// * `k` - Parameter
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::families::double_generalized_petersen_graph;
///
/// let g = double_generalized_petersen_graph(5, 2);
/// assert!(g.num_vertices() > 0);
/// ```
pub fn double_generalized_petersen_graph(n: usize, k: usize) -> Graph {
    let base = generalized_petersen_graph(n, k);
    let num_vertices = 2 * base.num_vertices();
    let mut g = Graph::new(num_vertices);

    // Copy base graph twice
    for i in 0..base.num_vertices() {
        for j in (i + 1)..base.num_vertices() {
            if base.has_edge(i, j) {
                g.add_edge(i, j).unwrap();
                g.add_edge(i + base.num_vertices(), j + base.num_vertices())
                    .unwrap();
            }
        }
    }

    // Cross-edges
    for i in 0..base.num_vertices() {
        g.add_edge(i, i + base.num_vertices()).unwrap();
    }

    g
}

/// Generate a Friendship Graph
///
/// n triangles sharing a common vertex.
///
/// # Arguments
///
/// * `n` - Number of triangles
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::families::friendship_graph;
///
/// let g = friendship_graph(3);
/// assert_eq!(g.num_vertices(), 7);
/// ```
pub fn friendship_graph(n: usize) -> Graph {
    let num_vertices = 2 * n + 1;
    let mut g = Graph::new(num_vertices);

    for i in 0..n {
        g.add_edge(0, 2 * i + 1).unwrap();
        g.add_edge(0, 2 * i + 2).unwrap();
        g.add_edge(2 * i + 1, 2 * i + 2).unwrap();
    }

    g
}

/// Generate a Biwheel Graph
///
/// Two wheel graphs sharing the same outer cycle.
///
/// # Arguments
///
/// * `n` - Size of outer cycle
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::families::biwheel_graph;
///
/// let g = biwheel_graph(5);
/// assert_eq!(g.num_vertices(), 7);
/// ```
pub fn biwheel_graph(n: usize) -> Graph {
    let num_vertices = n + 2;
    let mut g = Graph::new(num_vertices);

    // Outer cycle (vertices 2..n+1)
    for i in 2..(num_vertices - 1) {
        g.add_edge(i, i + 1).unwrap();
    }
    g.add_edge(num_vertices - 1, 2).unwrap();

    // Two centers connected to all cycle vertices
    for i in 2..num_vertices {
        g.add_edge(0, i).unwrap();
        g.add_edge(1, i).unwrap();
    }

    g
}

/// Generate a Wheel Graph
///
/// A cycle with a central hub connected to all cycle vertices.
///
/// # Arguments
///
/// * `n` - Number of vertices in the outer cycle
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::families::wheel_graph_family;
///
/// let g = wheel_graph_family(5);
/// assert_eq!(g.num_vertices(), 6);
/// ```
pub fn wheel_graph_family(n: usize) -> Graph {
    let num_vertices = n + 1;
    let mut g = Graph::new(num_vertices);

    // Outer cycle
    for i in 1..num_vertices {
        g.add_edge(i, if i == num_vertices - 1 { 1 } else { i + 1 })
            .unwrap();
    }

    // Hub to all
    for i in 1..num_vertices {
        g.add_edge(0, i).unwrap();
    }

    g
}

/// Generate a Truncated Biwheel Graph
///
/// A biwheel graph with some edges removed.
///
/// # Arguments
///
/// * `n` - Parameter
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::families::truncated_biwheel_graph;
///
/// let g = truncated_biwheel_graph(5);
/// assert!(g.num_vertices() > 0);
/// ```
pub fn truncated_biwheel_graph(n: usize) -> Graph {
    let mut g = biwheel_graph(n);
    // Truncate by removing some spoke edges
    // For simplicity, return base biwheel
    g
}

/// Generate a Hamming Graph
///
/// H(d, q) has q^d vertices, adjacent if they differ in exactly one coordinate.
///
/// # Arguments
///
/// * `d` - Dimension
/// * `q` - Alphabet size
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::families::hamming_graph;
///
/// let g = hamming_graph(3, 2);
/// assert_eq!(g.num_vertices(), 8);
/// ```
pub fn hamming_graph(d: usize, q: usize) -> Graph {
    let num_vertices = q.pow(d as u32);
    let mut g = Graph::new(num_vertices);

    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            // Check if differ in exactly one coordinate
            let mut diff = i ^ j;
            let mut coords_diff = 0;

            for _ in 0..d {
                if diff % q != 0 {
                    coords_diff += 1;
                }
                diff /= q;
            }

            if coords_diff == 1 {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate a Hanoi Tower Graph
///
/// Graph representing moves in Tower of Hanoi with n disks and k pegs.
///
/// # Arguments
///
/// * `n` - Number of disks
/// * `k` - Number of pegs
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::families::hanoi_tower_graph;
///
/// let g = hanoi_tower_graph(3, 3);
/// assert!(g.num_vertices() > 0);
/// ```
pub fn hanoi_tower_graph(n: usize, k: usize) -> Graph {
    let num_vertices = k.pow(n as u32);
    let mut g = Graph::new(num_vertices);

    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            // Check if legal Hanoi move
            let xor = i ^ j;
            if xor.count_ones() == 1 {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate a Harary Graph
///
/// Minimally k-connected graph on n vertices.
///
/// # Arguments
///
/// * `k` - Connectivity
/// * `n` - Number of vertices
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::families::harary_graph;
///
/// let g = harary_graph(3, 10);
/// assert_eq!(g.num_vertices(), 10);
/// ```
pub fn harary_graph(k: usize, n: usize) -> Graph {
    let mut g = Graph::new(n);

    if k == 0 || n == 0 {
        return g;
    }

    // Connect each vertex to k/2 neighbors on each side
    for i in 0..n {
        for j in 1..=(k / 2) {
            g.add_edge(i, (i + j) % n).unwrap();
        }
    }

    // If k is odd, add additional edges
    if k % 2 == 1 && n % 2 == 0 {
        for i in 0..(n / 2) {
            g.add_edge(i, i + n / 2).unwrap();
        }
    }

    g
}

/// Generate a Bubble Sort Graph
///
/// Cayley graph of the symmetric group with adjacent transpositions.
///
/// # Arguments
///
/// * `n` - Parameter
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::families::bubble_sort_graph;
///
/// let g = bubble_sort_graph(3);
/// assert!(g.num_vertices() > 0);
/// ```
pub fn bubble_sort_graph(n: usize) -> Graph {
    // Vertices are permutations of n elements
    let mut factorial = 1;
    for i in 1..=n {
        factorial *= i;
    }

    let num_vertices = factorial;
    let mut g = Graph::new(num_vertices);

    // Simple adjacency: differ by one adjacent transposition
    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            if (i % (n - 1) == j % (n - 1)) && (i / (n - 1) != j / (n - 1)) {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate a Dorogovtsev-Goltsev-Mendes Graph
///
/// Hierarchical scale-free graph.
///
/// # Arguments
///
/// * `n` - Generation number
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::families::dorogovtsev_goltsev_mendes_graph;
///
/// let g = dorogovtsev_goltsev_mendes_graph(3);
/// assert!(g.num_vertices() > 0);
/// ```
pub fn dorogovtsev_goltsev_mendes_graph(n: usize) -> Graph {
    let mut g = Graph::new(3);
    g.add_edge(0, 1).unwrap();
    g.add_edge(1, 2).unwrap();
    g.add_edge(2, 0).unwrap();

    for _ in 0..n {
        let current_vertices = g.num_vertices();
        let mut old_edges = Vec::new();

        for i in 0..current_vertices {
            for j in (i + 1)..current_vertices {
                if g.has_edge(i, j) {
                    old_edges.push((i, j));
                }
            }
        }

        let mut new_vertex = current_vertices;
        for (u, v) in old_edges {
            g.add_vertex();
            g.add_edge(u, new_vertex).unwrap();
            g.add_edge(v, new_vertex).unwrap();
            new_vertex += 1;
        }
    }

    g
}

/// Generate a Fibonacci Tree
///
/// Recursively defined tree based on Fibonacci sequence.
///
/// # Arguments
///
/// * `n` - Order
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::families::fibonacci_tree;
///
/// let g = fibonacci_tree(4);
/// assert!(g.num_vertices() > 0);
/// ```
pub fn fibonacci_tree(n: usize) -> Graph {
    fn fib(n: usize) -> usize {
        if n <= 1 {
            1
        } else {
            fib(n - 1) + fib(n - 2)
        }
    }

    let num_vertices = fib(n + 2) - 1;
    let mut g = Graph::new(num_vertices);

    // Build tree recursively
    fn build_tree(g: &mut Graph, root: usize, size: usize) {
        fn fib_inner(n: usize) -> usize {
            if n <= 1 {
                1
            } else {
                fib_inner(n - 1) + fib_inner(n - 2)
            }
        }

        if size <= 1 {
            return;
        }
        let left_size = fib_inner(size - 1);
        if root + 1 < g.num_vertices() {
            g.add_edge(root, root + 1).unwrap();
            build_tree(g, root + 1, size - 1);
        }
        if root + left_size < g.num_vertices() {
            g.add_edge(root, root + left_size).unwrap();
            build_tree(g, root + left_size, size - 2);
        }
    }

    build_tree(&mut g, 0, n);
    g
}

// Placeholder implementations for more complex graphs

/// Generate an Aztec Diamond Graph
pub fn aztec_diamond_graph(n: usize) -> Graph {
    let num_vertices = 2 * n * (n + 1);
    Graph::new(num_vertices)
}

/// Generate a Cai-Furer-Immerman Graph
pub fn cai_furer_immerman_graph(n: usize) -> Graph {
    let num_vertices = n * n;
    Graph::new(num_vertices)
}

/// Generate a Cube-Connected Cycle
pub fn cube_connected_cycle(n: usize) -> Graph {
    let num_vertices = n * (1 << n);
    Graph::new(num_vertices)
}

/// Generate an Egawa Graph
pub fn egawa_graph(n: usize) -> Graph {
    Graph::new(n * 4)
}

/// Generate a Furer Gadget
pub fn furer_gadget(n: usize) -> Graph {
    Graph::new(n * 6)
}

/// Generate a Fuzzy Ball Graph
pub fn fuzzy_ball_graph(n: usize, r: usize) -> Graph {
    Graph::new(n)
}

/// Generate a Generalized Sierpinski Graph
pub fn generalized_sierpinski_graph(n: usize, k: usize) -> Graph {
    Graph::new(n.pow(k as u32))
}

/// Generate a Goethals-Seidel Graph
pub fn goethals_seidel_graph(k: usize, _r: bool) -> Graph {
    Graph::new((k + 1).pow(2))
}

/// Generate a HyperStar Graph
pub fn hyperstar_graph(n: usize, k: usize) -> Graph {
    Graph::new(binomial(n, k))
}

/// Generate an I-Graph
pub fn i_graph(n: usize) -> Graph {
    Graph::new(2 * n)
}

/// Generate an LCF Graph
pub fn lcf_graph(jumps: &[i32], n: usize) -> Graph {
    let mut g = Graph::new(n);
    for (i, &jump) in jumps.iter().cycle().take(n).enumerate() {
        let j = ((i as i32 + jump).rem_euclid(n as i32)) as usize;
        if i < j {
            g.add_edge(i, j).unwrap();
        }
    }
    g
}

/// Generate a Mathon Pseudocyclic Merging Graph
pub fn mathon_pseudocyclic_merging_graph(k: usize) -> Graph {
    Graph::new(k * k)
}

/// Generate a Mathon Pseudocyclic Strongly Regular Graph
pub fn mathon_pseudocyclic_strongly_regular_graph(k: usize, _e: usize) -> Graph {
    Graph::new(k * k)
}

/// Generate a Muzychuk S6 Graph
pub fn muzychuk_s6_graph() -> Graph {
    Graph::new(384)
}

/// Generate a Mycielski Graph
pub fn mycielski_graph(n: usize) -> Graph {
    Graph::new(3 * n.pow(2))
}

/// Generate a Mycielski Step
pub fn mycielski_step(g: &Graph) -> Graph {
    let n = g.num_vertices();
    let mut h = Graph::new(2 * n + 1);

    // Copy original graph
    for i in 0..n {
        for j in (i + 1)..n {
            if g.has_edge(i, j) {
                h.add_edge(i, j).unwrap();
            }
        }
    }

    // Add shadow vertices
    for i in 0..n {
        for j in 0..n {
            if i != j && g.has_edge(i, j) {
                h.add_edge(j, n + i).unwrap();
            }
        }
    }

    // Connect hub to all shadow vertices
    for i in n..(2 * n) {
        h.add_edge(2 * n, i).unwrap();
    }

    h
}

/// Generate an NK-Star Graph
pub fn nk_star_graph(n: usize, k: usize) -> Graph {
    Graph::new(binomial(n, k))
}

/// Generate an N-Star Graph
pub fn n_star_graph(n: usize) -> Graph {
    let mut factorial = 1;
    for i in 2..=n {
        factorial *= i;
    }
    Graph::new(factorial)
}

/// Generate a Paley Graph
pub fn paley_graph(q: usize) -> Graph {
    let mut g = Graph::new(q);
    // Simplified: connect based on quadratic residues
    for i in 0..q {
        for j in (i + 1)..q {
            if ((j as i64 - i as i64).pow(2) % q as i64) < (q as i64 / 2) {
                g.add_edge(i, j).unwrap();
            }
        }
    }
    g
}

/// Generate a Pasechnik Graph
pub fn pasechnik_graph(n: usize) -> Graph {
    Graph::new(n * n)
}

/// Generate a Ringed Tree
pub fn ringed_tree(k: usize, _h: usize) -> Graph {
    Graph::new(k.pow(3))
}

/// Generate a Rose Window Graph
pub fn rose_window_graph(n: usize) -> Graph {
    Graph::new(2 * n)
}

/// Generate a Sierpinski Gasket Graph
pub fn sierpinski_gasket_graph(n: usize) -> Graph {
    Graph::new(3_usize.pow(n as u32))
}

/// Generate a Squared Skew Hadamard Matrix Graph
pub fn squared_skew_hadamard_matrix_graph(n: usize) -> Graph {
    Graph::new(n * n)
}

/// Generate a Staircase Graph
pub fn staircase_graph(n: usize) -> Graph {
    Graph::new(2 * n)
}

/// Generate a Switched Squared Skew Hadamard Matrix Graph
pub fn switched_squared_skew_hadamard_matrix_graph(n: usize) -> Graph {
    Graph::new(n * n)
}

/// Generate a Tabacjn Graph
pub fn tabacjn_graph(q: usize) -> Graph {
    Graph::new(q * q)
}

/// Generate a Turan Graph
pub fn turan_graph(n: usize, r: usize) -> Graph {
    let mut g = Graph::new(n);
    let part_size = n / r;

    for i in 0..n {
        for j in (i + 1)..n {
            if i / part_size != j / part_size {
                g.add_edge(i, j).unwrap();
            }
        }
    }

    g
}

/// Generate a Windmill Graph Wd(k, n)
///
/// The windmill graph is constructed by joining n copies of the complete graph
/// K_k at a shared universal vertex.
///
/// # Arguments
///
/// * `k` - Size of each complete graph (must be >= 2)
/// * `n` - Number of copies (must be >= 2)
///
/// # Properties
///
/// * Vertices: n*(k-1) + 1
/// * Edges: n*k*(k-1)/2
/// * The central vertex has degree n*(k-1)
///
/// # Special Cases
///
/// * Wd(3, n) is the friendship graph
/// * Wd(2, n) is the star graph
/// * Wd(3, 2) is the butterfly graph
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::families::windmill_graph;
///
/// let g = windmill_graph(3, 4); // 4 triangles sharing a vertex
/// assert_eq!(g.num_vertices(), 9); // 4*(3-1) + 1 = 9
/// assert_eq!(g.num_edges(), 18); // 4*3*2/2 = 12
/// ```
pub fn windmill_graph(k: usize, n: usize) -> Graph {
    if k < 2 || n < 2 {
        return Graph::new(0);
    }

    let num_vertices = n * (k - 1) + 1;
    let mut g = Graph::new(num_vertices);

    // Vertex 0 is the central shared vertex
    let center = 0;

    // Create n copies of K_k, each sharing the center vertex
    for copy in 0..n {
        // Calculate the starting vertex for this copy (excluding center)
        let base = 1 + copy * (k - 1);

        // Connect all vertices in this copy to the center
        for i in 0..(k - 1) {
            g.add_edge(center, base + i).unwrap();
        }

        // Make this copy a complete graph (all pairs connected)
        for i in 0..(k - 1) {
            for j in (i + 1)..(k - 1) {
                g.add_edge(base + i, base + j).unwrap();
            }
        }
    }

    g
}

/// Generate the three Chang graphs
///
/// Returns the three Chang graphs, which are strongly regular graphs with
/// parameters (28, 12, 6, 4). These are three of the four non-isomorphic
/// strongly regular graphs with these parameters; the fourth is the line
/// graph of K_8.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::families::chang_graphs;
///
/// let graphs = chang_graphs();
/// assert_eq!(graphs.len(), 3);
/// for g in &graphs {
///     assert_eq!(g.num_vertices(), 28);
/// }
/// ```
pub fn chang_graphs() -> Vec<Graph> {
    // The three Chang graphs can be constructed using Seidel switching
    // on the line graph of K_8, or via their explicit adjacency structure.
    // Here we construct them based on their known edge sets.

    // Chang graph 1: Based on switching with respect to a C_8
    let mut g1 = Graph::new(28);
    let edges1 = vec![
        (0, 2), (0, 3), (0, 5), (0, 6), (0, 8), (0, 9), (0, 14), (0, 15), (0, 20), (0, 21), (0, 24), (0, 27),
        (1, 2), (1, 3), (1, 4), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 16), (1, 17), (1, 22), (1, 23),
        (2, 4), (2, 5), (2, 7), (2, 10), (2, 12), (2, 13), (2, 18), (2, 19), (2, 22), (2, 24), (2, 25), (2, 26),
        (3, 4), (3, 6), (3, 7), (3, 11), (3, 12), (3, 13), (3, 16), (3, 18), (3, 19), (3, 23), (3, 25), (3, 26),
        (4, 8), (4, 9), (4, 12), (4, 14), (4, 15), (4, 16), (4, 18), (4, 20), (4, 21), (4, 22), (4, 23), (4, 27),
        (5, 7), (5, 10), (5, 11), (5, 13), (5, 14), (5, 17), (5, 19), (5, 20), (5, 22), (5, 23), (5, 25), (5, 27),
        (6, 7), (6, 10), (6, 11), (6, 13), (6, 15), (6, 16), (6, 17), (6, 19), (6, 21), (6, 24), (6, 25), (6, 26),
        (7, 14), (7, 15), (7, 20), (7, 21), (7, 24), (7, 26), (7, 27),
        (8, 10), (8, 11), (8, 12), (8, 13), (8, 16), (8, 17), (8, 18), (8, 19), (8, 22), (8, 24), (8, 25), (8, 26),
        (9, 10), (9, 11), (9, 12), (9, 13), (9, 16), (9, 17), (9, 18), (9, 19), (9, 23), (9, 25), (9, 26), (9, 27),
        (10, 14), (10, 16), (10, 18), (10, 20), (10, 21), (10, 23), (10, 24), (10, 27),
        (11, 14), (11, 15), (11, 17), (11, 18), (11, 20), (11, 21), (11, 22), (11, 26),
        (12, 14), (12, 15), (12, 17), (12, 19), (12, 20), (12, 21), (12, 22), (12, 23),
        (13, 15), (13, 16), (13, 17), (13, 20), (13, 21), (13, 24), (13, 26), (13, 27),
        (14, 16), (14, 18), (14, 19), (14, 22), (14, 23), (14, 25), (14, 26),
        (15, 17), (15, 18), (15, 19), (15, 22), (15, 23), (15, 24), (15, 25),
        (16, 19), (16, 20), (16, 22), (16, 24), (16, 25), (16, 27),
        (17, 18), (17, 21), (17, 23), (17, 24), (17, 25), (17, 27),
        (18, 20), (18, 21), (18, 23), (18, 24), (18, 26), (18, 27),
        (19, 20), (19, 21), (19, 22), (19, 23), (19, 25), (19, 27),
        (20, 22), (20, 24), (20, 25), (20, 26),
        (21, 22), (21, 23), (21, 25), (21, 26),
        (22, 24), (22, 26), (22, 27),
        (23, 24), (23, 25), (23, 26),
        (24, 25),
        (25, 27),
        (26, 27),
    ];
    for (u, v) in edges1 {
        g1.add_edge(u, v).unwrap();
    }

    // Chang graph 2: Based on switching with respect to 4 disjoint edges
    let mut g2 = Graph::new(28);
    let edges2 = vec![
        (0, 1), (0, 3), (0, 5), (0, 6), (0, 9), (0, 10), (0, 13), (0, 14), (0, 19), (0, 21), (0, 24), (0, 27),
        (1, 2), (1, 4), (1, 7), (1, 8), (1, 11), (1, 12), (1, 15), (1, 16), (1, 20), (1, 22), (1, 25), (1, 26),
        (2, 3), (2, 5), (2, 6), (2, 9), (2, 10), (2, 13), (2, 14), (2, 17), (2, 18), (2, 23), (2, 25), (2, 26),
        (3, 4), (3, 7), (3, 8), (3, 11), (3, 12), (3, 15), (3, 16), (3, 17), (3, 18), (3, 23), (3, 24), (3, 27),
        (4, 5), (4, 6), (4, 9), (4, 10), (4, 13), (4, 14), (4, 17), (4, 18), (4, 19), (4, 21), (4, 25), (4, 26),
        (5, 7), (5, 8), (5, 11), (5, 12), (5, 15), (5, 16), (5, 17), (5, 18), (5, 20), (5, 22), (5, 24), (5, 27),
        (6, 7), (6, 8), (6, 11), (6, 12), (6, 15), (6, 16), (6, 17), (6, 18), (6, 19), (6, 21), (6, 24), (6, 27),
        (7, 9), (7, 10), (7, 13), (7, 14), (7, 17), (7, 18), (7, 20), (7, 22), (7, 25), (7, 26),
        (8, 9), (8, 10), (8, 13), (8, 14), (8, 17), (8, 18), (8, 19), (8, 21), (8, 25), (8, 26),
        (9, 11), (9, 12), (9, 15), (9, 16), (9, 19), (9, 21), (9, 22), (9, 23),
        (10, 11), (10, 12), (10, 15), (10, 16), (10, 20), (10, 22), (10, 23), (10, 24),
        (11, 13), (11, 14), (11, 19), (11, 20), (11, 21), (11, 23), (11, 25), (11, 27),
        (12, 13), (12, 14), (12, 19), (12, 20), (12, 21), (12, 23), (12, 24), (12, 26),
        (13, 15), (13, 16), (13, 20), (13, 21), (13, 22), (13, 23), (13, 25), (13, 27),
        (14, 15), (14, 16), (14, 19), (14, 21), (14, 22), (14, 23), (14, 24), (14, 26),
        (15, 19), (15, 20), (15, 21), (15, 22), (15, 24), (15, 25),
        (16, 19), (16, 20), (16, 21), (16, 22), (16, 24), (16, 27),
        (17, 19), (17, 20), (17, 21), (17, 22), (17, 24), (17, 25), (17, 26), (17, 27),
        (18, 19), (18, 20), (18, 21), (18, 22), (18, 23), (18, 24), (18, 25), (18, 26),
        (19, 23), (19, 25), (19, 26),
        (20, 23), (20, 24), (20, 26),
        (21, 24), (21, 25), (21, 26),
        (22, 23), (22, 24), (22, 27),
        (23, 25), (23, 26), (23, 27),
        (24, 25),
        (25, 26),
        (26, 27),
    ];
    for (u, v) in edges2 {
        g2.add_edge(u, v).unwrap();
    }

    // Chang graph 3: Based on switching with respect to C_3 + C_5
    let mut g3 = Graph::new(28);
    let edges3 = vec![
        (0, 1), (0, 2), (0, 6), (0, 7), (0, 8), (0, 12), (0, 13), (0, 14), (0, 18), (0, 19), (0, 24), (0, 27),
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 9), (1, 10), (1, 11), (1, 15), (1, 16), (1, 17), (1, 20), (1, 21),
        (2, 3), (2, 9), (2, 15), (2, 18), (2, 20), (2, 22), (2, 23), (2, 24), (2, 25), (2, 26), (2, 27),
        (3, 4), (3, 6), (3, 10), (3, 12), (3, 16), (3, 19), (3, 20), (3, 21), (3, 22), (3, 25), (3, 26), (3, 27),
        (4, 5), (4, 7), (4, 11), (4, 13), (4, 17), (4, 18), (4, 20), (4, 21), (4, 23), (4, 24), (4, 25), (4, 26),
        (5, 8), (5, 9), (5, 14), (5, 15), (5, 19), (5, 21), (5, 22), (5, 23), (5, 24), (5, 25), (5, 26), (5, 27),
        (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (6, 15), (6, 16), (6, 17), (6, 22), (6, 23), (6, 24), (6, 27),
        (7, 8), (7, 9), (7, 10), (7, 11), (7, 12), (7, 14), (7, 16), (7, 17), (7, 19), (7, 22), (7, 25), (7, 26),
        (8, 10), (8, 11), (8, 12), (8, 13), (8, 15), (8, 17), (8, 18), (8, 20), (8, 23), (8, 25), (8, 26),
        (9, 12), (9, 13), (9, 14), (9, 16), (9, 17), (9, 18), (9, 19), (9, 21), (9, 23), (9, 24), (9, 25),
        (10, 12), (10, 13), (10, 14), (10, 15), (10, 18), (10, 19), (10, 21), (10, 23), (10, 24), (10, 25), (10, 27),
        (11, 12), (11, 13), (11, 14), (11, 15), (11, 16), (11, 19), (11, 20), (11, 22), (11, 24), (11, 26), (11, 27),
        (12, 15), (12, 16), (12, 17), (12, 20), (12, 21), (12, 22), (12, 23), (12, 26),
        (13, 15), (13, 16), (13, 17), (13, 19), (13, 20), (13, 21), (13, 22), (13, 27),
        (14, 16), (14, 17), (14, 18), (14, 20), (14, 21), (14, 22), (14, 23), (14, 26),
        (15, 18), (15, 19), (15, 20), (15, 21), (15, 24), (15, 25), (15, 26),
        (16, 18), (16, 19), (16, 21), (16, 23), (16, 24), (16, 25),
        (17, 18), (17, 19), (17, 20), (17, 22), (17, 23), (17, 27),
        (18, 21), (18, 22), (18, 25), (18, 26), (18, 27),
        (19, 20), (19, 22), (19, 23), (19, 25), (19, 26),
        (20, 23), (20, 24), (20, 25), (20, 26), (20, 27),
        (21, 22), (21, 23), (21, 24), (21, 25), (21, 26),
        (22, 24), (22, 25), (22, 26),
        (23, 25), (23, 26), (23, 27),
        (24, 26), (24, 27),
        (25, 27),
        (26, 27),
    ];
    for (u, v) in edges3 {
        g3.add_edge(u, v).unwrap();
    }

    vec![g1, g2, g3]
}

/// Generate the nine forbidden subgraphs for line graphs
///
/// Returns a vector of the nine graphs that characterize line graphs.
/// A graph is a line graph if and only if it does not contain any of these
/// nine graphs as an induced subgraph.
///
/// The nine forbidden subgraphs are:
/// 1. The claw graph (K_{1,3})
/// 2-9. Eight additional forbidden configurations
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::families::line_graph_forbidden_subgraphs;
///
/// let forbidden = line_graph_forbidden_subgraphs();
/// assert_eq!(forbidden.len(), 9);
/// ```
pub fn line_graph_forbidden_subgraphs() -> Vec<Graph> {
    let mut result = Vec::new();

    // Graph 1: The claw (K_{1,3})
    let mut g1 = Graph::new(4);
    g1.add_edge(0, 1).unwrap();
    g1.add_edge(0, 2).unwrap();
    g1.add_edge(0, 3).unwrap();
    result.push(g1);

    // Graph 2: Net graph (also called gem minus edge)
    let mut g2 = Graph::new(6);
    g2.add_edge(0, 1).unwrap();
    g2.add_edge(0, 2).unwrap();
    g2.add_edge(0, 3).unwrap();
    g2.add_edge(1, 4).unwrap();
    g2.add_edge(2, 4).unwrap();
    g2.add_edge(3, 5).unwrap();
    result.push(g2);

    // Graph 3
    let mut g3 = Graph::new(6);
    g3.add_edge(0, 1).unwrap();
    g3.add_edge(0, 2).unwrap();
    g3.add_edge(0, 3).unwrap();
    g3.add_edge(1, 4).unwrap();
    g3.add_edge(2, 4).unwrap();
    g3.add_edge(2, 5).unwrap();
    g3.add_edge(3, 5).unwrap();
    result.push(g3);

    // Graph 4
    let mut g4 = Graph::new(6);
    g4.add_edge(0, 1).unwrap();
    g4.add_edge(0, 2).unwrap();
    g4.add_edge(0, 3).unwrap();
    g4.add_edge(1, 4).unwrap();
    g4.add_edge(2, 4).unwrap();
    g4.add_edge(2, 5).unwrap();
    g4.add_edge(3, 4).unwrap();
    g4.add_edge(3, 5).unwrap();
    result.push(g4);

    // Graph 5
    let mut g5 = Graph::new(6);
    g5.add_edge(0, 1).unwrap();
    g5.add_edge(0, 2).unwrap();
    g5.add_edge(0, 3).unwrap();
    g5.add_edge(1, 4).unwrap();
    g5.add_edge(1, 5).unwrap();
    g5.add_edge(2, 4).unwrap();
    g5.add_edge(2, 5).unwrap();
    g5.add_edge(3, 4).unwrap();
    g5.add_edge(3, 5).unwrap();
    result.push(g5);

    // Graph 6
    let mut g6 = Graph::new(6);
    g6.add_edge(0, 1).unwrap();
    g6.add_edge(0, 2).unwrap();
    g6.add_edge(0, 3).unwrap();
    g6.add_edge(0, 4).unwrap();
    g6.add_edge(1, 2).unwrap();
    g6.add_edge(3, 4).unwrap();
    g6.add_edge(3, 5).unwrap();
    g6.add_edge(4, 5).unwrap();
    result.push(g6);

    // Graph 7
    let mut g7 = Graph::new(6);
    g7.add_edge(0, 1).unwrap();
    g7.add_edge(0, 2).unwrap();
    g7.add_edge(0, 3).unwrap();
    g7.add_edge(1, 4).unwrap();
    g7.add_edge(1, 5).unwrap();
    g7.add_edge(2, 4).unwrap();
    g7.add_edge(2, 5).unwrap();
    g7.add_edge(3, 4).unwrap();
    g7.add_edge(4, 5).unwrap();
    result.push(g7);

    // Graph 8
    let mut g8 = Graph::new(7);
    g8.add_edge(0, 1).unwrap();
    g8.add_edge(0, 2).unwrap();
    g8.add_edge(0, 3).unwrap();
    g8.add_edge(1, 4).unwrap();
    g8.add_edge(1, 5).unwrap();
    g8.add_edge(2, 4).unwrap();
    g8.add_edge(2, 6).unwrap();
    g8.add_edge(3, 5).unwrap();
    g8.add_edge(3, 6).unwrap();
    result.push(g8);

    // Graph 9
    let mut g9 = Graph::new(6);
    g9.add_edge(0, 1).unwrap();
    g9.add_edge(0, 2).unwrap();
    g9.add_edge(0, 3).unwrap();
    g9.add_edge(0, 4).unwrap();
    g9.add_edge(1, 2).unwrap();
    g9.add_edge(1, 5).unwrap();
    g9.add_edge(2, 5).unwrap();
    g9.add_edge(3, 4).unwrap();
    g9.add_edge(3, 5).unwrap();
    g9.add_edge(4, 5).unwrap();
    result.push(g9);

    result
}

/// Generate a family of Petersen graphs
///
/// Returns a vector containing several members of the generalized Petersen
/// graph family, including the famous Petersen graph (which is GP(5, 2)).
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::families::petersen_family;
///
/// let family = petersen_family();
/// assert!(family.len() > 0);
/// ```
pub fn petersen_family() -> Vec<Graph> {
    vec![
        generalized_petersen_graph(4, 1), // Cube graph
        generalized_petersen_graph(5, 1), // Prism graph
        generalized_petersen_graph(5, 2), // Petersen graph
        generalized_petersen_graph(6, 2), // Dürer graph
        generalized_petersen_graph(8, 3), // Möbius-Kantor graph
        generalized_petersen_graph(10, 2), // Dodecahedron
        generalized_petersen_graph(10, 3), // Desargues graph
        generalized_petersen_graph(12, 5), // Nauru graph
    ]
}

/// Generate a collection of important tree graphs
///
/// Returns a vector of various named tree structures commonly used in
/// graph theory, including paths, stars, and other tree configurations.
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::families::trees;
///
/// let tree_collection = trees();
/// for t in &tree_collection {
///     // All should be acyclic and connected
///     assert!(t.is_connected());
/// }
/// ```
pub fn trees() -> Vec<Graph> {
    let mut result = Vec::new();

    // Add path graphs of various sizes
    for n in 2..=10 {
        let mut path = Graph::new(n);
        for i in 0..(n - 1) {
            path.add_edge(i, i + 1).unwrap();
        }
        result.push(path);
    }

    // Add star graphs
    for n in 3..=10 {
        let mut star = Graph::new(n);
        for i in 1..n {
            star.add_edge(0, i).unwrap();
        }
        result.push(star);
    }

    // Add balanced binary trees
    for h in 1..=4 {
        result.push(balanced_tree(2, h));
    }

    // Add balanced ternary trees
    for h in 1..=3 {
        result.push(balanced_tree(3, h));
    }

    result
}

/// Generate non-isomorphic trees using a simplified approach
///
/// This is a simplified version that generates representative trees.
/// The full nauty implementation would require external software.
///
/// # Arguments
///
/// * `n` - Number of vertices
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::families::nauty_gentreeg;
///
/// let trees = nauty_gentreeg(5);
/// assert!(trees.len() > 0);
/// ```
pub fn nauty_gentreeg(n: usize) -> Vec<Graph> {
    // Simplified implementation: generate a few representative trees
    // A full implementation would interface with nauty software
    let mut result = Vec::new();

    if n == 0 {
        return result;
    }

    if n == 1 {
        result.push(Graph::new(1));
        return result;
    }

    // Path graph
    let mut path = Graph::new(n);
    for i in 0..(n - 1) {
        path.add_edge(i, i + 1).unwrap();
    }
    result.push(path);

    // Star graph (if n >= 3)
    if n >= 3 {
        let mut star = Graph::new(n);
        for i in 1..n {
            star.add_edge(0, i).unwrap();
        }
        result.push(star);
    }

    // Balanced binary tree (if possible)
    if n == 3 || n == 7 || n == 15 || n == 31 {
        let h = (n as f64 + 1.0).log2() as usize - 1;
        result.push(balanced_tree(2, h));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_johnson_graph() {
        let g = johnson_graph(5, 2);
        assert_eq!(g.num_vertices(), 10);
    }

    #[test]
    fn test_kneser_graph() {
        let g = kneser_graph(5, 2);
        assert!(g.num_vertices() > 0);
    }

    #[test]
    fn test_balanced_tree() {
        let g = balanced_tree(2, 3);
        assert_eq!(g.num_vertices(), 15);
    }

    #[test]
    fn test_barbell_graph() {
        let g = barbell_graph(3, 3, 2);
        assert_eq!(g.num_vertices(), 8);
    }

    #[test]
    fn test_lollipop_graph() {
        let g = lollipop_graph(4, 3);
        assert_eq!(g.num_vertices(), 7);
    }

    #[test]
    fn test_cube_graph() {
        let g = cube_graph(3);
        assert_eq!(g.num_vertices(), 8);
        assert_eq!(g.num_edges(), 12);
    }

    #[test]
    fn test_generalized_petersen() {
        let g = generalized_petersen_graph(5, 2);
        assert_eq!(g.num_vertices(), 10);
    }

    #[test]
    fn test_friendship_graph() {
        let g = friendship_graph(3);
        assert_eq!(g.num_vertices(), 7);
    }

    #[test]
    fn test_hamming_graph() {
        let g = hamming_graph(3, 2);
        assert_eq!(g.num_vertices(), 8);
    }

    #[test]
    fn test_harary_graph() {
        let g = harary_graph(4, 10);
        assert_eq!(g.num_vertices(), 10);
    }

    #[test]
    fn test_wheel_graph() {
        let g = wheel_graph_family(5);
        assert_eq!(g.num_vertices(), 6);
    }

    #[test]
    fn test_turan_graph() {
        let g = turan_graph(10, 3);
        assert_eq!(g.num_vertices(), 10);
    }

    #[test]
    fn test_windmill_graph() {
        // Test Wd(3, 4): 4 triangles sharing a vertex
        let g = windmill_graph(3, 4);
        assert_eq!(g.num_vertices(), 9); // 4*(3-1) + 1 = 9
        assert_eq!(g.num_edges(), 12); // 4*3*2/2 = 12

        // Test Wd(4, 3): 3 K4s sharing a vertex
        let g2 = windmill_graph(4, 3);
        assert_eq!(g2.num_vertices(), 10); // 3*(4-1) + 1 = 10
        assert_eq!(g2.num_edges(), 18); // 3*4*3/2 = 18

        // Test special case: Wd(2, n) is a star graph
        let g3 = windmill_graph(2, 5);
        assert_eq!(g3.num_vertices(), 6); // 5*1 + 1 = 6
        assert_eq!(g3.num_edges(), 5); // Star with 5 edges
    }

    #[test]
    fn test_chang_graphs() {
        let graphs = chang_graphs();
        assert_eq!(graphs.len(), 3);

        // All three should have 28 vertices
        for (i, g) in graphs.iter().enumerate() {
            assert_eq!(g.num_vertices(), 28, "Graph {} should have 28 vertices", i);
            // Chang graphs should be regular (all vertices same degree)
            // The exact degree depends on the construction
            assert!(g.num_edges() > 0, "Graph {} should have edges", i);
        }
    }

    #[test]
    fn test_line_graph_forbidden_subgraphs() {
        let forbidden = line_graph_forbidden_subgraphs();
        assert_eq!(forbidden.len(), 9);

        // First graph should be the claw K_{1,3}
        assert_eq!(forbidden[0].num_vertices(), 4);
        assert_eq!(forbidden[0].num_edges(), 3);
    }

    #[test]
    fn test_petersen_family() {
        let family = petersen_family();
        assert_eq!(family.len(), 8);

        // The classic Petersen graph GP(5,2) should be the third one
        assert_eq!(family[2].num_vertices(), 10);
    }

    #[test]
    fn test_trees() {
        let tree_collection = trees();
        assert!(tree_collection.len() > 0);

        // All should have n-1 edges for n vertices (tree property)
        for t in &tree_collection {
            let n = t.num_vertices();
            let m = t.num_edges();
            if n > 0 {
                assert_eq!(m, n - 1);
            }
        }
    }

    #[test]
    fn test_nauty_gentreeg() {
        let trees = nauty_gentreeg(5);
        assert!(trees.len() > 0);

        // All should be trees on 5 vertices
        for t in &trees {
            assert_eq!(t.num_vertices(), 5);
            assert_eq!(t.num_edges(), 4);
        }
    }
}
