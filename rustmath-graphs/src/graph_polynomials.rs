//! Graph polynomials as exact `rustmath-polynomials` objects (MAGMA Handbook
//! Chapter 149, §149.19 "Colourings", plus the classical matching and Tutte
//! polynomials).
//!
//! The pre-existing `graph.rs` returns these as ad-hoc coefficient vectors
//! (`Vec<i64>` for the chromatic and matching polynomials) and `Vec<(i64, usize,
//! usize)>` for the Tutte polynomial.  The port plan (backlog item: "chromatic /
//! Tutte / matching polynomial as rustmath-polynomials polynomials instead of
//! Vec<i64>") wants exact algebraic OUTPUT objects that compose with the ring
//! tower.  This module provides them, beside the existing encodings:
//!
//!   * `ChromaticPolynomial(G)` -> [`chromatic_polynomial_integer`] as
//!     `UnivariatePolynomial<Integer>` (variable = number of colours).
//!   * matching polynomial      -> [`matching_polynomial_integer`] as
//!     `UnivariatePolynomial<Integer>` (the signed/characteristic matching poly).
//!   * Tutte polynomial `T(x,y)` -> [`tutte_polynomial_integer`] as a bivariate
//!     `MultivariatePolynomial<Integer>` (variable 0 = x, variable 1 = y).
//!
//! All three use exact deletion–contraction recursions over `Integer`, so they
//! are exponential in the number of edges — intended for the small graphs that
//! appear in the MAGMA worked examples, not for large inputs.

use crate::graph::Graph;
use rustmath_integers::Integer;
use rustmath_polynomials::{MultivariatePolynomial, UnivariatePolynomial};
use std::collections::{BTreeSet, VecDeque};

// ---------------------------------------------------------------------------
// Simple-graph representation (chromatic & matching): alive vertex set + a set
// of unordered edges {u, v} with u < v.  Using an "alive set" avoids relabelling
// on vertex removal / contraction.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct SimpleGraph {
    alive: BTreeSet<usize>,
    edges: BTreeSet<(usize, usize)>,
}

impl SimpleGraph {
    fn from_graph(g: &Graph) -> Self {
        let alive: BTreeSet<usize> = (0..g.num_vertices()).collect();
        let mut edges = BTreeSet::new();
        for (u, v) in g.edges() {
            if u != v {
                edges.insert((u.min(v), u.max(v)));
            }
        }
        SimpleGraph { alive, edges }
    }

    fn any_edge(&self) -> Option<(usize, usize)> {
        self.edges.iter().next().copied()
    }

    /// G - e (delete a single edge).
    fn delete_edge(&self, e: (usize, usize)) -> Self {
        let mut g = self.clone();
        g.edges.remove(&e);
        g
    }

    /// G / e (contract edge {u, v}: merge v into u, drop resulting loops, dedup
    /// parallel edges — the correct operation for the chromatic polynomial).
    fn contract_edge(&self, (u, v): (usize, usize)) -> Self {
        let mut alive = self.alive.clone();
        alive.remove(&v);
        let mut edges = BTreeSet::new();
        for &(a, b) in &self.edges {
            if (a, b) == (u, v) {
                continue; // the contracted edge disappears
            }
            // redirect endpoints equal to v onto u
            let a2 = if a == v { u } else { a };
            let b2 = if b == v { u } else { b };
            if a2 != b2 {
                edges.insert((a2.min(b2), a2.max(b2)));
            }
        }
        SimpleGraph { alive, edges }
    }

    /// G - u - v (delete both vertices and all incident edges).
    fn delete_vertices(&self, u: usize, v: usize) -> Self {
        let mut alive = self.alive.clone();
        alive.remove(&u);
        alive.remove(&v);
        let edges = self
            .edges
            .iter()
            .filter(|&&(a, b)| a != u && a != v && b != u && b != v)
            .copied()
            .collect();
        SimpleGraph { alive, edges }
    }
}

/// `ChromaticPolynomial(G)` — the number of proper vertex colourings of `G` with
/// `x` colours, as an exact `UnivariatePolynomial<Integer>`.
///
/// Computed by the deletion–contraction recurrence
/// `P(G) = P(G - e) - P(G / e)`, with base case `P(empty graph on n vertices) =
/// x^n`.
pub fn chromatic_polynomial_integer(g: &Graph) -> UnivariatePolynomial<Integer> {
    chromatic_rec(&SimpleGraph::from_graph(g))
}

fn chromatic_rec(g: &SimpleGraph) -> UnivariatePolynomial<Integer> {
    match g.any_edge() {
        None => {
            // No edges: x^n where n = number of (alive) vertices.
            let n = g.alive.len();
            x_pow(n)
        }
        Some(e) => {
            let del = chromatic_rec(&g.delete_edge(e));
            let con = chromatic_rec(&g.contract_edge(e));
            del - con
        }
    }
}

/// The (signed) matching polynomial `μ(G, x) = Σ_k (-1)^k m_k x^{n-2k}`, where
/// `m_k` is the number of `k`-edge matchings, as an exact
/// `UnivariatePolynomial<Integer>`.  This is the *characteristic* matching
/// polynomial (it coincides with the characteristic polynomial of the adjacency
/// matrix for forests).
///
/// Computed by `μ(G) = μ(G - e) - μ(G - u - v)` for an edge `e = {u, v}`, with
/// base case `μ(empty graph on n vertices) = x^n`.
pub fn matching_polynomial_integer(g: &Graph) -> UnivariatePolynomial<Integer> {
    matching_rec(&SimpleGraph::from_graph(g))
}

fn matching_rec(g: &SimpleGraph) -> UnivariatePolynomial<Integer> {
    match g.any_edge() {
        None => x_pow(g.alive.len()),
        Some((u, v)) => {
            let without_e = matching_rec(&g.delete_edge((u, v)));
            let use_e = matching_rec(&g.delete_vertices(u, v));
            without_e - use_e
        }
    }
}

/// `x^n` as a `UnivariatePolynomial<Integer>`.
fn x_pow(n: usize) -> UnivariatePolynomial<Integer> {
    let mut coeffs = vec![Integer::zero(); n + 1];
    coeffs[n] = Integer::one();
    UnivariatePolynomial::from_coefficients(coeffs)
}

// ---------------------------------------------------------------------------
// Multigraph representation (Tutte): alive vertex set + a multiset of edges
// (as a Vec of ordered pairs) that keeps loops and parallel edges, exactly as
// the Tutte deletion–contraction requires.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct MultiEdges {
    alive: BTreeSet<usize>,
    edges: Vec<(usize, usize)>, // may contain loops (u, u) and parallels
}

impl MultiEdges {
    fn from_graph(g: &Graph) -> Self {
        let alive: BTreeSet<usize> = (0..g.num_vertices()).collect();
        let edges = g.edges();
        MultiEdges { alive, edges }
    }

    /// Delete the edge at index `i`.
    fn delete_at(&self, i: usize) -> Self {
        let mut edges = self.edges.clone();
        edges.remove(i);
        MultiEdges {
            alive: self.alive.clone(),
            edges,
        }
    }

    /// Contract the edge at index `i` (must be a non-loop): merge `v` into `u`,
    /// keeping loops and parallel edges.
    fn contract_at(&self, i: usize) -> Self {
        let (u, v) = self.edges[i];
        debug_assert!(u != v);
        let mut alive = self.alive.clone();
        alive.remove(&v);
        let mut edges = Vec::with_capacity(self.edges.len() - 1);
        for (j, &(a, b)) in self.edges.iter().enumerate() {
            if j == i {
                continue;
            }
            let a2 = if a == v { u } else { a };
            let b2 = if b == v { u } else { b };
            edges.push((a2, b2)); // loops (a2 == b2) are retained
        }
        MultiEdges { alive, edges }
    }

    /// Whether the edge at index `i` (a non-loop `{u, v}`) is a bridge: is `u`
    /// still connected to `v` after removing this single edge occurrence?
    fn is_bridge(&self, i: usize) -> bool {
        let (u, v) = self.edges[i];
        // BFS from u over all edges except occurrence i.
        let mut adj: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for &w in &self.alive {
            adj.entry(w).or_default();
        }
        for (j, &(a, b)) in self.edges.iter().enumerate() {
            if j == i || a == b {
                continue;
            }
            adj.entry(a).or_default().push(b);
            adj.entry(b).or_default().push(a);
        }
        let mut seen = BTreeSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(u);
        seen.insert(u);
        while let Some(w) = queue.pop_front() {
            if w == v {
                return false; // still connected => not a bridge
            }
            if let Some(ns) = adj.get(&w) {
                for &x in ns {
                    if seen.insert(x) {
                        queue.push_back(x);
                    }
                }
            }
        }
        true
    }
}

/// The Tutte polynomial `T(G; x, y)` as an exact bivariate
/// `MultivariatePolynomial<Integer>` (variable 0 = `x`, variable 1 = `y`).
///
/// Computed by the standard deletion–contraction recurrence:
///   * an edge that is a **loop** contributes a factor `y`: `T = y · T(G - e)`;
///   * an edge that is a **bridge** contributes a factor `x`: `T = x · T(G / e)`;
///   * an **ordinary** edge: `T = T(G - e) + T(G / e)`;
///   * base case (no edges): `T = 1`.
pub fn tutte_polynomial_integer(g: &Graph) -> MultivariatePolynomial<Integer> {
    tutte_rec(&MultiEdges::from_graph(g))
}

fn tutte_rec(g: &MultiEdges) -> MultivariatePolynomial<Integer> {
    if g.edges.is_empty() {
        return MultivariatePolynomial::constant(Integer::one());
    }
    // Pick the last edge (cheap removal); handle loops first among all edges.
    // Loops and bridges give a multiplicative factor.
    // Find any loop.
    if let Some(i) = g.edges.iter().position(|&(a, b)| a == b) {
        let y = MultivariatePolynomial::variable(1);
        return y * tutte_rec(&g.delete_at(i));
    }
    let i = g.edges.len() - 1;
    if g.is_bridge(i) {
        let x = MultivariatePolynomial::variable(0);
        x * tutte_rec(&g.contract_at(i))
    } else {
        tutte_rec(&g.delete_at(i)) + tutte_rec(&g.contract_at(i))
    }
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

    fn path(n: usize) -> Graph {
        let mut g = Graph::new(n);
        for i in 0..n - 1 {
            g.add_edge(i, i + 1).unwrap();
        }
        g
    }

    #[test]
    fn chromatic_of_triangle() {
        // P(K3, x) = x(x-1)(x-2) = x^3 - 3x^2 + 2x.
        let g = complete(3);
        let p = chromatic_polynomial_integer(&g);
        let c = p.coefficients();
        assert_eq!(c[0], Integer::zero());
        assert_eq!(c[1], Integer::from(2));
        assert_eq!(c[2], Integer::from(-3));
        assert_eq!(c[3], Integer::one());
        // Evaluate: 3 colours -> 3*2*1 = 6 proper colourings.
        assert_eq!(p.evaluate(&Integer::from(3)), Integer::from(6));
        // 2 colours -> 0 (triangle is not 2-colourable).
        assert_eq!(p.evaluate(&Integer::from(2)), Integer::zero());
    }

    #[test]
    fn chromatic_of_path() {
        // P(P_n, x) = x (x-1)^{n-1}.  For P3: x(x-1)^2 -> at x=3 gives 3*4 = 12.
        let g = path(3);
        let p = chromatic_polynomial_integer(&g);
        assert_eq!(p.evaluate(&Integer::from(3)), Integer::from(12));
        assert_eq!(p.evaluate(&Integer::from(2)), Integer::from(2));
    }

    #[test]
    fn chromatic_of_c4() {
        // P(C4, x) = (x-1)^4 + (x-1).  At x=3: 16 + 2 = 18.
        let g = cycle(4);
        let p = chromatic_polynomial_integer(&g);
        assert_eq!(p.evaluate(&Integer::from(3)), Integer::from(18));
    }

    #[test]
    fn matching_polynomial_of_path3() {
        // P3 (2 edges): matchings m0=1, m1=2, m2=0.
        // mu = x^3 - 2x.
        let g = path(3);
        let p = matching_polynomial_integer(&g);
        let c = p.coefficients();
        assert_eq!(c[0], Integer::zero());
        assert_eq!(c[1], Integer::from(-2));
        assert_eq!(c[2], Integer::zero());
        assert_eq!(c[3], Integer::one());
    }

    #[test]
    fn matching_polynomial_of_triangle() {
        // K3: m0=1, m1=3.  mu = x^3 - 3x.
        let g = complete(3);
        let p = matching_polynomial_integer(&g);
        let c = p.coefficients();
        assert_eq!(c[1], Integer::from(-3));
        assert_eq!(c[3], Integer::one());
    }

    #[test]
    fn tutte_of_single_edge_is_x() {
        // A bridge: T = x.
        let mut g = Graph::new(2);
        g.add_edge(0, 1).unwrap();
        let t = tutte_polynomial_integer(&g);
        // x -> coefficient of monomial x^1 is 1, everything else 0.
        assert_eq!(t.evaluate(&[Integer::from(5), Integer::from(7)]), Integer::from(5));
    }

    #[test]
    fn tutte_of_triangle() {
        // T(C3; x, y) = x^2 + x + y.
        let g = cycle(3);
        let t = tutte_polynomial_integer(&g);
        // Evaluate at (x, y) = (2, 3): 4 + 2 + 3 = 9.
        assert_eq!(t.evaluate(&[Integer::from(2), Integer::from(3)]), Integer::from(9));
        // T(1,1) = number of spanning trees of C3 = 3.
        assert_eq!(t.evaluate(&[Integer::one(), Integer::one()]), Integer::from(3));
    }

    #[test]
    fn tutte_spanning_tree_count_k4() {
        // T(G; 1, 1) counts spanning trees; K4 has 4^{4-2} = 16 (Cayley).
        let g = complete(4);
        let t = tutte_polynomial_integer(&g);
        assert_eq!(t.evaluate(&[Integer::one(), Integer::one()]), Integer::from(16));
    }

    #[test]
    fn tutte_path_is_x_squared() {
        // P3 has two bridges: T = x^2.
        let g = path(3);
        let t = tutte_polynomial_integer(&g);
        assert_eq!(t.evaluate(&[Integer::from(3), Integer::from(9)]), Integer::from(9));
        assert_eq!(t.evaluate(&[Integer::one(), Integer::one()]), Integer::one());
    }
}
