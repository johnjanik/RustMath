//! Exact integer matrices associated with a graph (MAGMA Handbook Chapter 149,
//! §149.16 "Matrices and Vector Spaces Associated with a Graph or Digraph").
//!
//! MAGMA returns these objects as elements of the matrix ring `M_p(Z)` (or the
//! bimodule `M^{p x q}(Z)` for the incidence matrix), *not* as floating-point
//! arrays.  This module provides the exact `rustmath-matrix::Matrix<Integer>`
//! versions and lives *beside* the pre-existing f64 `spectra.rs` (which is kept
//! for backwards compatibility).
//!
//! Ported intrinsics:
//!   * `AdjacencyMatrix(G)`       -> [`adjacency_matrix_integer`]
//!   * `DistanceMatrix(G)`        -> [`distance_matrix_integer`]
//!   * `IncidenceMatrix(G)`       -> [`incidence_matrix_integer`]
//!   * `IntersectionMatrix(G, P)` -> [`intersection_matrix_integer`]
//!   * `Laplacian` (D - A)        -> [`laplacian_matrix_integer`]  (RustMath extension)
//!
//! Foundation adoption: OUTPUT objects are `Matrix<Integer>` so they compose
//! with the rest of the RustMath linear-algebra tower; the `Graph` container
//! itself stays a plain combinatorial object (as the port plan mandates).

use crate::graph::Graph;
use rustmath_integers::Integer;
use rustmath_matrix::Matrix;

/// Adjacency matrix `A` of `G` as an element of `M_p(Z)` (MAGMA `AdjacencyMatrix`).
///
/// `A[i][j] = 1` if `v_i` and `v_j` are adjacent, otherwise `0`.  For an
/// undirected graph the matrix is symmetric.
pub fn adjacency_matrix_integer(g: &Graph) -> Matrix<Integer> {
    let n = g.num_vertices();
    let mut data = vec![Integer::zero(); n * n];
    for i in 0..n {
        if let Some(neighbors) = g.neighbors(i) {
            for j in neighbors {
                data[i * n + j] = Integer::one();
            }
        }
    }
    Matrix::from_vec(n, n, data).expect("square n x n data")
}

/// Laplacian matrix `L = D - A` of `G` as an element of `M_p(Z)`.
///
/// `D` is the diagonal matrix of vertex degrees.  This is a RustMath extension
/// (not a MAGMA §149.16 intrinsic) and is the exact analogue of the f64
/// `spectra::laplacian_matrix`.
pub fn laplacian_matrix_integer(g: &Graph) -> Matrix<Integer> {
    let n = g.num_vertices();
    let mut data = vec![Integer::zero(); n * n];
    for i in 0..n {
        let deg = g.degree(i).unwrap_or(0) as i64;
        data[i * n + i] = Integer::from(deg);
        if let Some(neighbors) = g.neighbors(i) {
            for j in neighbors {
                // L = D - A : subtract one for each adjacency.
                let idx = i * n + j;
                data[idx] = data[idx].clone() - Integer::one();
            }
        }
    }
    Matrix::from_vec(n, n, data).expect("square n x n data")
}

/// Distance matrix `A` of `G` as an element of `M_p(Z)` (MAGMA `DistanceMatrix`).
///
/// `A[i][j]` is the number of edges on a shortest `v_i`–`v_j` path; the diagonal
/// is `0`.  Pairs lying in different connected components are unreachable; MAGMA
/// leaves these implementation-defined, and here they are recorded as the
/// sentinel `-1` so the result stays inside `M_p(Z)` (exact integers).
pub fn distance_matrix_integer(g: &Graph) -> Matrix<Integer> {
    let n = g.num_vertices();
    let mut data = vec![Integer::from(-1i64); n * n];
    // BFS from every source gives all shortest-path lengths in O(V*(V+E)).
    for s in 0..n {
        let dist = bfs_distances(g, s, n);
        for (t, d) in dist.into_iter().enumerate() {
            if let Some(d) = d {
                data[s * n + t] = Integer::from(d as i64);
            }
        }
    }
    Matrix::from_vec(n, n, data).expect("square n x n data")
}

/// Incidence matrix `M` of `G` as a `p x q` integer matrix (MAGMA
/// `IncidenceMatrix`).  Rows are vertices, columns are edges (in the order
/// returned by [`Graph::edges`]); `M[i][j] = 1` iff vertex `v_i` is an
/// end-vertex of edge `e_j`.
pub fn incidence_matrix_integer(g: &Graph) -> Matrix<Integer> {
    let p = g.num_vertices();
    let edges = g.edges();
    let q = edges.len();
    let mut data = vec![Integer::zero(); p * q];
    for (j, (u, v)) in edges.iter().enumerate() {
        data[u * q + j] = Integer::one();
        // For a genuine edge u != v both endpoints are marked; a loop would
        // still yield a single 1 in that column (simple graphs have no loops).
        data[v * q + j] = Integer::one();
    }
    Matrix::from_vec(p, q, data).expect("p x q data")
}

/// Intersection matrix `T` of `G` with respect to an ordered partition
/// `P = P_1 ∪ … ∪ P_r` of `V(G)` (MAGMA `IntersectionMatrix(G, P)`).
///
/// `T[i][j]` is the number of vertices of `P_j` adjacent to a (representative)
/// vertex of `P_i`.  For an *equitable* partition this count is independent of
/// the chosen representative, which is the intended use; the first vertex of
/// each cell is used as the representative.
///
/// Returns `None` if `P` is not a valid partition of `{0,…,p-1}`.
pub fn intersection_matrix_integer(g: &Graph, partition: &[Vec<usize>]) -> Option<Matrix<Integer>> {
    let p = g.num_vertices();
    // Validate: cells are disjoint, non-empty, and cover every vertex exactly once.
    let mut seen = vec![false; p];
    for cell in partition {
        if cell.is_empty() {
            return None;
        }
        for &v in cell {
            if v >= p || seen[v] {
                return None;
            }
            seen[v] = true;
        }
    }
    if !seen.iter().all(|&b| b) {
        return None;
    }

    let r = partition.len();
    // Cell index for each vertex.
    let mut cell_of = vec![0usize; p];
    for (ci, cell) in partition.iter().enumerate() {
        for &v in cell {
            cell_of[v] = ci;
        }
    }

    let mut data = vec![Integer::zero(); r * r];
    for i in 0..r {
        let rep = partition[i][0];
        let neighbors = g.neighbors(rep).unwrap_or_default();
        for w in neighbors {
            let j = cell_of[w];
            let idx = i * r + j;
            data[idx] = data[idx].clone() + Integer::one();
        }
    }
    Some(Matrix::from_vec(r, r, data).expect("r x r data"))
}

/// BFS shortest-path distances from `source` (in edge counts); `None` for
/// unreachable vertices.
fn bfs_distances(g: &Graph, source: usize, n: usize) -> Vec<Option<usize>> {
    let mut dist = vec![None; n];
    dist[source] = Some(0);
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(source);
    while let Some(u) = queue.pop_front() {
        let du = dist[u].unwrap();
        if let Some(neighbors) = g.neighbors(u) {
            for v in neighbors {
                if dist[v].is_none() {
                    dist[v] = Some(du + 1);
                    queue.push_back(v);
                }
            }
        }
    }
    dist
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cycle(n: usize) -> Graph {
        let mut g = Graph::new(n);
        for i in 0..n {
            g.add_edge(i, (i + 1) % n).unwrap();
        }
        g
    }

    fn complete(n: usize) -> Graph {
        let mut g = Graph::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                g.add_edge(i, j).unwrap();
            }
        }
        g
    }

    #[test]
    fn adjacency_is_symmetric_0_1() {
        let g = cycle(4);
        let a = adjacency_matrix_integer(&g);
        assert_eq!(a.rows(), 4);
        assert_eq!(a.cols(), 4);
        // C4: 0-1, 1-2, 2-3, 3-0
        assert_eq!(*a.get(0, 1).unwrap(), Integer::one());
        assert_eq!(*a.get(0, 2).unwrap(), Integer::zero());
        assert_eq!(*a.get(3, 0).unwrap(), Integer::one());
        // symmetry
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(a.get(i, j).unwrap(), a.get(j, i).unwrap());
            }
        }
    }

    #[test]
    fn laplacian_row_sums_zero() {
        // L * 1 = 0 : every row of the Laplacian sums to zero.
        let g = complete(4);
        let l = laplacian_matrix_integer(&g);
        for i in 0..4 {
            let mut s = Integer::zero();
            for j in 0..4 {
                s = s + l.get(i, j).unwrap().clone();
            }
            assert_eq!(s, Integer::zero());
        }
        // diagonal of K4 Laplacian is the degree 3
        assert_eq!(*l.get(0, 0).unwrap(), Integer::from(3));
        assert_eq!(*l.get(0, 1).unwrap(), Integer::from(-1));
    }

    #[test]
    fn distance_matrix_path() {
        // Path 0-1-2: distances 0,1,2.
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        let d = distance_matrix_integer(&g);
        assert_eq!(*d.get(0, 0).unwrap(), Integer::zero());
        assert_eq!(*d.get(0, 1).unwrap(), Integer::one());
        assert_eq!(*d.get(0, 2).unwrap(), Integer::from(2));
        assert_eq!(*d.get(2, 0).unwrap(), Integer::from(2));
    }

    #[test]
    fn distance_matrix_disconnected_sentinel() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        // vertex 2 isolated
        let d = distance_matrix_integer(&g);
        assert_eq!(*d.get(0, 2).unwrap(), Integer::from(-1));
    }

    #[test]
    fn incidence_dimensions_and_column_sum() {
        let g = cycle(4);
        let m = incidence_matrix_integer(&g);
        assert_eq!(m.rows(), 4); // p vertices
        assert_eq!(m.cols(), 4); // q edges
        // Each column (edge) has exactly two 1's.
        for j in 0..m.cols() {
            let mut s = Integer::zero();
            for i in 0..m.rows() {
                s = s + m.get(i, j).unwrap().clone();
            }
            assert_eq!(s, Integer::from(2));
        }
    }

    #[test]
    fn intersection_matrix_of_bipartition() {
        // C4 with bipartition {0,2},{1,3}: every vertex has both neighbours
        // in the other cell.
        let g = cycle(4);
        let p = vec![vec![0, 2], vec![1, 3]];
        let t = intersection_matrix_integer(&g, &p).unwrap();
        assert_eq!(t.rows(), 2);
        assert_eq!(*t.get(0, 0).unwrap(), Integer::zero());
        assert_eq!(*t.get(0, 1).unwrap(), Integer::from(2));
        assert_eq!(*t.get(1, 0).unwrap(), Integer::from(2));
    }

    #[test]
    fn intersection_matrix_rejects_bad_partition() {
        let g = cycle(4);
        // vertex 3 missing
        assert!(intersection_matrix_integer(&g, &[vec![0, 1], vec![2]]).is_none());
        // duplicate vertex
        assert!(intersection_matrix_integer(&g, &[vec![0, 0], vec![1, 2, 3]]).is_none());
    }
}
