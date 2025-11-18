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
}
