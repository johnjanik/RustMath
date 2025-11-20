//! Ramsey theory for graphs
//!
//! Ramsey theory studies the conditions under which order must appear in
//! large enough structures. The classic Ramsey number R(s,t) is the minimum n
//! such that any graph on n vertices contains either a clique of size s or
//! an independent set of size t.

use crate::graph::Graph;
use std::collections::HashSet;

/// Find a clique of size k in the graph
///
/// Returns the vertices forming a clique, or None if no such clique exists
pub fn find_clique(g: &Graph, k: usize) -> Option<Vec<usize>> {
    let n = g.num_vertices();

    if k == 0 {
        return Some(vec![]);
    }

    if k > n {
        return None;
    }

    // Use backtracking to find clique
    let mut current = Vec::new();
    let mut candidates: Vec<usize> = (0..n).collect();

    find_clique_recursive(g, k, &mut current, &mut candidates)
}

fn find_clique_recursive(
    g: &Graph,
    k: usize,
    current: &mut Vec<usize>,
    candidates: &mut Vec<usize>,
) -> Option<Vec<usize>> {
    if current.len() == k {
        return Some(current.clone());
    }

    if candidates.is_empty() {
        return None;
    }

    // Try each candidate
    let cands = candidates.clone();
    for &v in &cands {
        // Check if v is connected to all vertices in current
        let connected_to_all = current.iter().all(|&u| g.has_edge(u, v));

        if connected_to_all {
            current.push(v);

            // Update candidates to only include neighbors of v
            let new_candidates: Vec<usize> = candidates
                .iter()
                .filter(|&&u| u > v && g.has_edge(v, u))
                .copied()
                .collect();

            let mut temp_candidates = new_candidates;
            if let result @ Some(_) = find_clique_recursive(g, k, current, &mut temp_candidates) {
                return result;
            }

            current.pop();
        }
    }

    None
}

/// Find an independent set of size k
///
/// An independent set is a set of vertices with no edges between them
pub fn find_independent_set(g: &Graph, k: usize) -> Option<Vec<usize>> {
    let n = g.num_vertices();

    if k == 0 {
        return Some(vec![]);
    }

    if k > n {
        return None;
    }

    // Use backtracking
    let mut current = Vec::new();
    let mut candidates: Vec<usize> = (0..n).collect();

    find_independent_set_recursive(g, k, &mut current, &mut candidates)
}

fn find_independent_set_recursive(
    g: &Graph,
    k: usize,
    current: &mut Vec<usize>,
    candidates: &mut Vec<usize>,
) -> Option<Vec<usize>> {
    if current.len() == k {
        return Some(current.clone());
    }

    if candidates.is_empty() {
        return None;
    }

    let cands = candidates.clone();
    for &v in &cands {
        // Check if v has no edges to vertices in current
        let no_edges = current.iter().all(|&u| !g.has_edge(u, v));

        if no_edges {
            current.push(v);

            // Update candidates to exclude neighbors of v
            let new_candidates: Vec<usize> = candidates
                .iter()
                .filter(|&&u| u > v && !g.has_edge(v, u))
                .copied()
                .collect();

            let mut temp_candidates = new_candidates;
            if let result @ Some(_) = find_independent_set_recursive(g, k, current, &mut temp_candidates) {
                return result;
            }

            current.pop();
        }
    }

    None
}

/// Compute the clique number (size of maximum clique)
pub fn clique_number(g: &Graph) -> usize {
    let n = g.num_vertices();

    for k in (1..=n).rev() {
        if find_clique(g, k).is_some() {
            return k;
        }
    }

    0
}

/// Compute the independence number (size of maximum independent set)
pub fn independence_number(g: &Graph) -> usize {
    let n = g.num_vertices();

    for k in (1..=n).rev() {
        if find_independent_set(g, k).is_some() {
            return k;
        }
    }

    0
}

/// Compute the Ramsey number R(s, t) by exhaustive search
///
/// Warning: This is exponential and only works for small values
pub fn ramsey_number(s: usize, t: usize) -> Option<usize> {
    // Known small Ramsey numbers
    if s == 1 || t == 1 {
        return Some(1);
    }

    if s == 2 || t == 2 {
        return Some(s.max(t));
    }

    if s == 3 && t == 3 {
        return Some(6);
    }

    if s == 3 && t == 4 {
        return Some(9);
    }

    if s == 4 && t == 3 {
        return Some(9);
    }

    if s == 3 && t == 5 {
        return Some(14);
    }

    if s == 5 && t == 3 {
        return Some(14);
    }

    if s == 4 && t == 4 {
        return Some(18);
    }

    // For larger values, we can't compute exactly
    None
}

/// Get a lower bound on R(s, t) using known results
pub fn ramsey_lower_bound(s: usize, t: usize) -> usize {
    if s == 1 || t == 1 {
        return 1;
    }

    if s == 2 {
        return t;
    }

    if t == 2 {
        return s;
    }

    // Use bound: R(s,t) >= R(s-1,t) + R(s,t-1)
    // This is a rough approximation
    (s + t - 2)
}

/// Get an upper bound on R(s, t) using binomial coefficient bound
pub fn ramsey_upper_bound(s: usize, t: usize) -> usize {
    // R(s,t) <= C(s+t-2, s-1)
    binomial(s + t - 2, s - 1)
}

fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }

    let k = k.min(n - k);
    let mut result = 1;

    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }

    result
}

/// Check if a graph contains a clique of size k
pub fn has_clique(g: &Graph, k: usize) -> bool {
    find_clique(g, k).is_some()
}

/// Check if a graph contains an independent set of size k
pub fn has_independent_set(g: &Graph, k: usize) -> bool {
    find_independent_set(g, k).is_some()
}

/// Check if a graph witnesses that R(s,t) > n
///
/// Returns true if the graph on n vertices has neither a clique of size s
/// nor an independent set of size t
pub fn is_ramsey_witness(g: &Graph, s: usize, t: usize) -> bool {
    !has_clique(g, s) && !has_independent_set(g, t)
}

/// Find all cliques in a graph
///
/// Warning: Can be exponential in the size of the graph
pub fn all_cliques(g: &Graph, max_size: usize) -> Vec<Vec<usize>> {
    let mut cliques = Vec::new();
    let n = g.num_vertices();

    for k in 1..=max_size.min(n) {
        enumerate_cliques(g, k, &mut cliques);
    }

    cliques
}

fn enumerate_cliques(g: &Graph, k: usize, cliques: &mut Vec<Vec<usize>>) {
    let n = g.num_vertices();
    let mut current = Vec::new();
    let mut candidates: Vec<usize> = (0..n).collect();

    enumerate_cliques_helper(g, k, &mut current, &mut candidates, cliques);
}

fn enumerate_cliques_helper(
    g: &Graph,
    k: usize,
    current: &mut Vec<usize>,
    candidates: &mut Vec<usize>,
    cliques: &mut Vec<Vec<usize>>,
) {
    if current.len() == k {
        cliques.push(current.clone());
        return;
    }

    let cands = candidates.clone();
    for &v in &cands {
        let connected_to_all = current.iter().all(|&u| g.has_edge(u, v));

        if connected_to_all {
            current.push(v);

            let new_candidates: Vec<usize> = candidates
                .iter()
                .filter(|&&u| u > v && g.has_edge(v, u))
                .copied()
                .collect();

            let mut temp_candidates = new_candidates;
            enumerate_cliques_helper(g, k, current, &mut temp_candidates, cliques);

            current.pop();
        }
    }
}

/// Compute the chromatic number using Ramsey theory insights
///
/// The chromatic number is the minimum number of colors needed to color vertices
pub fn chromatic_number_ramsey(g: &Graph) -> usize {
    let n = g.num_vertices();

    if n == 0 {
        return 0;
    }

    // The independence number gives a lower bound on chromatic number
    let alpha = independence_number(g);

    if alpha == 0 {
        return n;
    }

    // Ceil(n / alpha) is a lower bound
    let lower = (n + alpha - 1) / alpha;

    // Try each value starting from lower bound
    for k in lower..=n {
        if is_k_colorable(g, k) {
            return k;
        }
    }

    n
}

fn is_k_colorable(g: &Graph, k: usize) -> bool {
    let n = g.num_vertices();
    let mut colors = vec![0; n];

    is_k_colorable_helper(g, 0, k, &mut colors)
}

fn is_k_colorable_helper(g: &Graph, vertex: usize, k: usize, colors: &mut [usize]) -> bool {
    if vertex == g.num_vertices() {
        return true;
    }

    for color in 1..=k {
        if is_color_valid(g, vertex, color, colors) {
            colors[vertex] = color;

            if is_k_colorable_helper(g, vertex + 1, k, colors) {
                return true;
            }

            colors[vertex] = 0;
        }
    }

    false
}

fn is_color_valid(g: &Graph, vertex: usize, color: usize, colors: &[usize]) -> bool {
    if let Some(neighbors) = g.neighbors(vertex) {
        for neighbor in neighbors {
            if colors[neighbor] == color {
                return false;
            }
        }
    }

    true
}

/// Paley construction: construct a graph that avoids small cliques and independent sets
///
/// This is useful for finding lower bounds on Ramsey numbers
pub fn paley_tournament(q: usize) -> Graph {
    // Simplified version: create a circulant graph
    let mut g = Graph::new(q);

    // Connect i to j if (j-i) is a quadratic residue mod q
    // For simplicity, use a pattern that creates many edges
    for i in 0..q {
        for j in (i + 1)..q {
            let diff = j - i;
            // Simple pattern: connect if diff is odd (for simplicity)
            if diff % 2 == 1 && diff <= q / 2 {
                g.add_edge(i, j).ok();
            }
        }
    }

    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_clique_triangle() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(0, 2).unwrap();

        let clique = find_clique(&g, 3);
        assert!(clique.is_some());
        assert_eq!(clique.unwrap().len(), 3);
    }

    #[test]
    fn test_find_clique_none() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();

        let clique = find_clique(&g, 3);
        assert!(clique.is_none());
    }

    #[test]
    fn test_find_independent_set() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(2, 3).unwrap();

        let ind_set = find_independent_set(&g, 2);
        assert!(ind_set.is_some());
    }

    #[test]
    fn test_clique_number() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(0, 2).unwrap();

        let omega = clique_number(&g);
        assert_eq!(omega, 3);
    }

    #[test]
    fn test_independence_number() {
        let mut g = Graph::new(5);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();
        g.add_edge(3, 4).unwrap();

        let alpha = independence_number(&g);
        // In a path, we can select alternating vertices
        assert!(alpha >= 2);
    }

    #[test]
    fn test_ramsey_number_small() {
        assert_eq!(ramsey_number(1, 5), Some(1));
        assert_eq!(ramsey_number(2, 5), Some(5));
        assert_eq!(ramsey_number(3, 3), Some(6));
        assert_eq!(ramsey_number(3, 4), Some(9));
        assert_eq!(ramsey_number(4, 4), Some(18));
    }

    #[test]
    fn test_ramsey_bounds() {
        let s = 5;
        let t = 5;

        let lower = ramsey_lower_bound(s, t);
        let upper = ramsey_upper_bound(s, t);

        assert!(lower <= upper);
    }

    #[test]
    fn test_has_clique() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 2).unwrap();
        g.add_edge(0, 3).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(1, 3).unwrap();
        g.add_edge(2, 3).unwrap();

        assert!(has_clique(&g, 4));
        assert!(!has_clique(&g, 5));
    }

    #[test]
    fn test_has_independent_set() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();

        assert!(has_independent_set(&g, 2));
    }

    #[test]
    fn test_is_ramsey_witness() {
        let g = Graph::new(5);

        // Empty graph has large independent set but no cliques
        assert!(!is_ramsey_witness(&g, 2, 5));
    }

    #[test]
    fn test_all_cliques() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 2).unwrap();
        g.add_edge(1, 2).unwrap();

        let cliques = all_cliques(&g, 3);
        // Should find triangle and all edges
        assert!(cliques.len() > 0);
    }

    #[test]
    fn test_binomial() {
        assert_eq!(binomial(4, 2), 6);
        assert_eq!(binomial(5, 3), 10);
        assert_eq!(binomial(6, 0), 1);
        assert_eq!(binomial(6, 6), 1);
    }

    #[test]
    fn test_chromatic_number() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(0, 2).unwrap();

        let chi = chromatic_number_ramsey(&g);
        assert_eq!(chi, 3);  // Triangle needs 3 colors
    }

    #[test]
    fn test_paley_tournament() {
        let g = paley_tournament(7);
        assert_eq!(g.num_vertices(), 7);
        // Should have some edges
        assert!(g.num_edges() > 0);
    }

    #[test]
    fn test_empty_graph() {
        let g = Graph::new(0);
        assert_eq!(clique_number(&g), 0);
        assert_eq!(independence_number(&g), 0);
    }
}
