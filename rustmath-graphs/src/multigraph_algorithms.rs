//! Multigraph-specific algorithms (MAGMA Handbook Chapter 150 "Multigraphs").
//!
//! The pre-existing `multigraph.rs` provides the [`MultiGraph`] container
//! (parallel edges + loops) with basic queries.  This module adds, *beside* it,
//! the multigraph-specific traversal and Euler-tour machinery that Chapter 150
//! covers (§150.8 degree functions, and the Eulerian theory that §149.11 defines
//! via `IsEulerian`), computed correctly in the presence of loops and parallel
//! edges:
//!
//!   * handshake lemma check           -> [`handshake_holds`] / [`degree_sum`]
//!   * loop detection                  -> [`has_loops`]
//!   * `IsEulerian` (circuit / path)   -> [`is_eulerian`]
//!   * Euler tour (Hierholzer)         -> [`eulerian_circuit`]
//!   * multigraph BFS / DFS traversal  -> [`multi_bfs`] / [`multi_dfs`]
//!   * degree sequence                 -> [`degree_sequence`]
//!
//! These are free functions taking `&MultiGraph` (the container stays plain).

use crate::multigraph::MultiGraph;
use std::collections::{HashSet, VecDeque};

/// The sum of all vertex degrees.  By the handshake lemma this equals twice the
/// number of edges; a loop contributes `2` to the degree of its vertex.
pub fn degree_sum(g: &MultiGraph) -> usize {
    (0..g.num_vertices())
        .map(|v| g.degree(v).unwrap_or(0))
        .sum()
}

/// The handshake lemma: `Σ_v deg(v) = 2 · |E|`.  Always true for a well-formed
/// multigraph; exposed as a checkable invariant (loops counted with weight 2).
pub fn handshake_holds(g: &MultiGraph) -> bool {
    degree_sum(g) == 2 * g.num_edges()
}

/// Whether the multigraph has any loop (an edge from a vertex to itself).
pub fn has_loops(g: &MultiGraph) -> bool {
    (0..g.num_vertices()).any(|v| g.neighbors(v).iter().any(|&(w, _)| w == v))
}

/// The number of loops incident to `v` (each loop counted once).
pub fn loop_count(g: &MultiGraph, v: usize) -> usize {
    if v >= g.num_vertices() {
        return 0;
    }
    // A loop appears twice in the adjacency list of v with the same edge id.
    let mut ids = HashSet::new();
    for &(w, eid) in &g.neighbors(v) {
        if w == v {
            ids.insert(eid);
        }
    }
    ids.len()
}

/// Degree sequence `D` where `D[i]` is the number of vertices of degree `i`
/// (length `max_degree + 1`).  Mirrors MAGMA `DegreeSequence(G)` (§150.8.1) up to
/// the off-by-one in indexing (MAGMA uses `D[i]` = #vertices of degree `i-1`).
pub fn degree_sequence(g: &MultiGraph) -> Vec<usize> {
    let degs: Vec<usize> = (0..g.num_vertices())
        .map(|v| g.degree(v).unwrap_or(0))
        .collect();
    let max_deg = degs.iter().copied().max().unwrap_or(0);
    let mut seq = vec![0usize; max_deg + 1];
    for d in degs {
        seq[d] += 1;
    }
    seq
}

/// Breadth-first traversal order of the multigraph starting at `start`
/// (parallel edges do not create duplicate visits).
pub fn multi_bfs(g: &MultiGraph, start: usize) -> Vec<usize> {
    let n = g.num_vertices();
    let mut order = Vec::new();
    if start >= n {
        return order;
    }
    let mut visited = vec![false; n];
    let mut queue = VecDeque::new();
    visited[start] = true;
    queue.push_back(start);
    while let Some(v) = queue.pop_front() {
        order.push(v);
        let mut neigh: Vec<usize> = g.neighbors(v).iter().map(|&(w, _)| w).collect();
        neigh.sort_unstable();
        neigh.dedup();
        for w in neigh {
            if !visited[w] {
                visited[w] = true;
                queue.push_back(w);
            }
        }
    }
    order
}

/// Depth-first traversal order of the multigraph starting at `start`.
pub fn multi_dfs(g: &MultiGraph, start: usize) -> Vec<usize> {
    let n = g.num_vertices();
    let mut order = Vec::new();
    if start >= n {
        return order;
    }
    let mut visited = vec![false; n];
    let mut stack = vec![start];
    while let Some(v) = stack.pop() {
        if visited[v] {
            continue;
        }
        visited[v] = true;
        order.push(v);
        let mut neigh: Vec<usize> = g.neighbors(v).iter().map(|&(w, _)| w).collect();
        neigh.sort_unstable();
        neigh.dedup();
        // push in reverse so smaller neighbours are visited first
        for w in neigh.into_iter().rev() {
            if !visited[w] {
                stack.push(w);
            }
        }
    }
    order
}

/// `IsEulerian(G)` (§149.11) for a multigraph, returning
/// `(has_eulerian_circuit, has_eulerian_path)`.
///
/// A connected multigraph (ignoring isolated vertices) has:
///   * an Eulerian **circuit** iff every vertex has even degree;
///   * an Eulerian **path** iff exactly zero or two vertices have odd degree.
pub fn is_eulerian(g: &MultiGraph) -> (bool, bool) {
    let n = g.num_vertices();
    if g.num_edges() == 0 {
        // Vacuously both (empty circuit / path).
        return (true, true);
    }
    let odd = (0..n).filter(|&v| g.degree(v).unwrap_or(0) % 2 == 1).count();
    let connected = edges_connected(g);
    let circuit = connected && odd == 0;
    let path = connected && (odd == 0 || odd == 2);
    (circuit, path)
}

/// An Eulerian circuit of `g` as a vertex sequence (Hierholzer's algorithm), or
/// `None` if none exists.  The returned sequence has length `|E| + 1` and starts
/// and ends at the same vertex; each edge (including loops and parallels) is
/// used exactly once.
pub fn eulerian_circuit(g: &MultiGraph) -> Option<Vec<usize>> {
    let n = g.num_vertices();
    if g.num_edges() == 0 {
        return Some(vec![]);
    }
    // Even-degree + connectivity are necessary and sufficient.
    if !is_eulerian(g).0 {
        return None;
    }
    let start = (0..n).find(|&v| g.degree(v).unwrap_or(0) > 0)?;

    // Working adjacency: per vertex, a list of (neighbour, edge_id) half-edges,
    // consumed via a moving pointer; an edge id is used at most once overall.
    let adj: Vec<Vec<(usize, usize)>> = (0..n).map(|v| g.neighbors(v)).collect();
    let mut ptr = vec![0usize; n];
    let mut used: HashSet<usize> = HashSet::new();

    let mut stack = vec![start];
    let mut circuit = Vec::new();
    while let Some(&v) = stack.last() {
        let mut advanced = false;
        while ptr[v] < adj[v].len() {
            let (w, eid) = adj[v][ptr[v]];
            ptr[v] += 1;
            if used.contains(&eid) {
                continue;
            }
            used.insert(eid);
            stack.push(w);
            advanced = true;
            break;
        }
        if !advanced {
            circuit.push(stack.pop().unwrap());
        }
    }
    circuit.reverse();
    // Sanity: a valid Euler circuit visits |E| + 1 vertices.
    if circuit.len() == g.num_edges() + 1 {
        Some(circuit)
    } else {
        None
    }
}

/// Whether every vertex of positive degree is reachable from the first such
/// vertex (connectivity of the edge-carrying part of the multigraph).
fn edges_connected(g: &MultiGraph) -> bool {
    let n = g.num_vertices();
    let start = match (0..n).find(|&v| g.degree(v).unwrap_or(0) > 0) {
        Some(s) => s,
        None => return true,
    };
    let mut visited = vec![false; n];
    let mut stack = vec![start];
    visited[start] = true;
    while let Some(v) = stack.pop() {
        for &(w, _) in &g.neighbors(v) {
            if !visited[w] {
                visited[w] = true;
                stack.push(w);
            }
        }
    }
    (0..n).all(|v| g.degree(v).unwrap_or(0) == 0 || visited[v])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handshake_with_parallel_edges_and_loop() {
        let mut g = MultiGraph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 1).unwrap(); // parallel
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 2).unwrap(); // loop
        assert!(handshake_holds(&g));
        // degrees: 0->2, 1->3, 2->1+2(loop)=3 ; sum = 8 = 2*4 edges
        assert_eq!(degree_sum(&g), 8);
        assert_eq!(g.num_edges(), 4);
    }

    #[test]
    fn loop_detection() {
        let mut g = MultiGraph::new(2);
        g.add_edge(0, 1).unwrap();
        assert!(!has_loops(&g));
        g.add_edge(1, 1).unwrap();
        assert!(has_loops(&g));
        assert_eq!(loop_count(&g, 1), 1);
        assert_eq!(loop_count(&g, 0), 0);
    }

    #[test]
    fn eulerian_circuit_on_multi_triangle() {
        // Triangle with a doubled edge -> all even degrees, Eulerian circuit.
        let mut g = MultiGraph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 0).unwrap();
        g.add_edge(0, 1).unwrap(); // makes deg(0)=3? no: deg0=2(edges to1,to2)+1=3
        // Actually degrees: with 4 edges 0-1,1-2,2-0,0-1: deg0=3,deg1=3,deg2=2 -> not all even.
        let (circuit, path) = is_eulerian(&g);
        assert!(!circuit);
        assert!(path); // exactly two odd vertices (0 and 1)
        assert!(eulerian_circuit(&g).is_none());
    }

    #[test]
    fn eulerian_circuit_exists_and_uses_all_edges() {
        // Simple triangle: all degrees 2 -> Eulerian circuit of length 4.
        let mut g = MultiGraph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 0).unwrap();
        let (circuit, _) = is_eulerian(&g);
        assert!(circuit);
        let tour = eulerian_circuit(&g).unwrap();
        assert_eq!(tour.len(), g.num_edges() + 1);
        assert_eq!(tour.first(), tour.last());
    }

    #[test]
    fn eulerian_circuit_with_loop() {
        // Two vertices, an edge between them plus a loop at each -> all even.
        let mut g = MultiGraph::new(2);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 0).unwrap(); // parallel back, deg0=2, deg1=2
        g.add_edge(0, 0).unwrap(); // loop, deg0 -> 4
        g.add_edge(1, 1).unwrap(); // loop, deg1 -> 4
        assert!(handshake_holds(&g));
        let (circuit, _) = is_eulerian(&g);
        assert!(circuit);
        let tour = eulerian_circuit(&g).unwrap();
        assert_eq!(tour.len(), g.num_edges() + 1);
    }

    #[test]
    fn traversal_orders() {
        let mut g = MultiGraph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 1).unwrap(); // parallel: must not double-visit
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();
        assert_eq!(multi_bfs(&g, 0), vec![0, 1, 2, 3]);
        assert_eq!(multi_dfs(&g, 0), vec![0, 1, 2, 3]);
    }

    #[test]
    fn degree_sequence_counts() {
        let mut g = MultiGraph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 2).unwrap();
        g.add_edge(0, 3).unwrap();
        // star: deg0=3, others=1
        let seq = degree_sequence(&g);
        assert_eq!(seq[1], 3); // three vertices of degree 1
        assert_eq!(seq[3], 1); // one vertex of degree 3
    }

    #[test]
    fn disconnected_is_not_eulerian() {
        let mut g = MultiGraph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(2, 3).unwrap();
        // two components each a single edge: degrees all odd anyway
        let (circuit, _) = is_eulerian(&g);
        assert!(!circuit);
    }
}
