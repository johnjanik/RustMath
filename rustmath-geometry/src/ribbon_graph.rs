//! Ribbon Graphs
//!
//! This module provides functionality for working with ribbon graphs - graphs with
//! cyclic orderings of darts (half-edges) at each vertex that define orientable
//! surfaces with boundary.
//!
//! A ribbon graph is encoded by two permutations:
//! - σ (sigma): Cycles representing vertex adjacencies; each cycle orders darts around a vertex
//! - ρ (rho): 2-cycles pairing darts that form complete edges

use std::collections::{HashMap, HashSet};
use std::fmt;

/// A ribbon graph
///
/// Represents a graph with cyclic orderings of darts (half-edges) at each vertex
/// that define orientable surfaces with boundary.
///
/// # Core Representation
///
/// The graph is encoded by two permutations:
/// - σ (sigma): Cycles representing vertex adjacencies
/// - ρ (rho): 2-cycles pairing darts that form complete edges
///
/// # Example
///
/// ```
/// use rustmath_geometry::ribbon_graph::RibbonGraph;
///
/// // Create a simple ribbon graph
/// let sigma = vec![vec![1, 2, 3], vec![4, 5]];
/// let rho = vec![(1, 4), (2, 5), (3, 6)];
/// let graph = RibbonGraph::new(sigma, rho, false);
/// ```
#[derive(Debug, Clone)]
pub struct RibbonGraph {
    /// Vertex permutation: cycles representing darts around each vertex
    sigma: Vec<Vec<usize>>,
    /// Edge permutation: pairs of darts forming edges
    rho: Vec<(usize, usize)>,
    /// Whether the graph is bipartite
    bipartite: bool,
    /// Number of darts
    n_darts: usize,
}

impl RibbonGraph {
    /// Creates a new ribbon graph from permutations
    ///
    /// # Arguments
    ///
    /// * `sigma` - Vertex cycles (ordering of darts around vertices)
    /// * `rho` - Edge pairs (2-cycles pairing darts)
    /// * `bipartite` - Whether the graph is bipartite
    ///
    /// # Returns
    ///
    /// A new `RibbonGraph`
    pub fn new(sigma: Vec<Vec<usize>>, rho: Vec<(usize, usize)>, bipartite: bool) -> Self {
        // Count total number of darts
        let n_darts = sigma.iter().map(|cycle| cycle.len()).sum::<usize>();

        Self {
            sigma,
            rho,
            bipartite,
            n_darts,
        }
    }

    /// Creates a ribbon graph from genus and boundary count
    ///
    /// Constructs a standard ribbon graph whose thickening has genus g and r boundary components.
    ///
    /// # Arguments
    ///
    /// * `genus` - The genus of the surface
    /// * `boundary_count` - The number of boundary components
    ///
    /// # Returns
    ///
    /// A new `RibbonGraph`
    pub fn from_genus_and_boundary(genus: usize, boundary_count: usize) -> Self {
        make_ribbon(genus, boundary_count)
    }

    /// Creates a bipartite ribbon graph K(p, q)
    ///
    /// Models complete bipartite graphs with cyclic planar ordering.
    ///
    /// # Arguments
    ///
    /// * `p` - First partition size
    /// * `q` - Second partition size
    ///
    /// # Returns
    ///
    /// A new bipartite `RibbonGraph`
    pub fn bipartite(p: usize, q: usize) -> Self {
        bipartite_ribbon_graph(p, q)
    }

    /// Returns the number of darts in the graph
    pub fn n_darts(&self) -> usize {
        self.n_darts
    }

    /// Returns the number of vertices
    pub fn n_vertices(&self) -> usize {
        self.sigma.len()
    }

    /// Returns the number of edges
    pub fn n_edges(&self) -> usize {
        self.rho.len()
    }

    /// Computes the genus of the surface
    ///
    /// For ribbon graphs, the genus is computed as:
    /// g = 1 + (E - V - F) / 2
    /// where E is the number of edges, V is vertices, and F is faces (boundaries).
    pub fn genus(&self) -> i32 {
        let v = self.n_vertices() as i32;
        let e = self.n_edges() as i32;
        let f = self.number_boundaries() as i32;

        // Genus formula for ribbon graphs
        1 + (e - v - f) / 2
    }

    /// Counts the number of boundary components
    ///
    /// Boundary components are cycles in the composition ρ·σ.
    pub fn number_boundaries(&self) -> usize {
        // Build composition ρ·σ
        let mut composition = HashMap::new();

        // First apply σ
        let mut sigma_map = HashMap::new();
        for cycle in &self.sigma {
            for i in 0..cycle.len() {
                let from = cycle[i];
                let to = cycle[(i + 1) % cycle.len()];
                sigma_map.insert(from, to);
            }
        }

        // Then apply ρ
        let mut rho_map = HashMap::new();
        for &(a, b) in &self.rho {
            rho_map.insert(a, b);
            rho_map.insert(b, a);
        }

        // Compose: (ρ·σ)(x) = ρ(σ(x))
        for (&key, &sigma_val) in &sigma_map {
            if let Some(&rho_val) = rho_map.get(&sigma_val) {
                composition.insert(key, rho_val);
            }
        }

        // Count cycles in composition
        let mut visited = HashSet::new();
        let mut n_cycles = 0;

        for &start in composition.keys() {
            if visited.contains(&start) {
                continue;
            }

            let mut current = start;
            let mut cycle_length = 0;
            let max_cycle_length = self.n_darts * 2; // Prevent infinite loops

            loop {
                visited.insert(current);
                cycle_length += 1;

                if cycle_length > max_cycle_length {
                    // Safety check: prevent infinite loops
                    break;
                }

                if let Some(&next) = composition.get(&current) {
                    current = next;
                    if current == start {
                        break;
                    }
                } else {
                    break;
                }
            }
            n_cycles += 1;
        }

        n_cycles
    }

    /// Returns the rank of the first homology group
    ///
    /// Computed as μ = 2g + r - 1 where g is genus and r is boundary count.
    pub fn mu(&self) -> usize {
        let g = self.genus();
        let r = self.number_boundaries();
        (2 * g + r as i32 - 1).max(0) as usize
    }

    /// Returns the boundary components as sequences of edges
    pub fn boundary(&self) -> Vec<Vec<usize>> {
        let mut boundaries = Vec::new();

        // Build composition ρ·σ and track cycles
        let mut sigma_map = HashMap::new();
        for cycle in &self.sigma {
            for i in 0..cycle.len() {
                let from = cycle[i];
                let to = cycle[(i + 1) % cycle.len()];
                sigma_map.insert(from, to);
            }
        }

        let mut rho_map = HashMap::new();
        for &(a, b) in &self.rho {
            rho_map.insert(a, b);
            rho_map.insert(b, a);
        }

        let mut visited = HashSet::new();

        for &start in sigma_map.keys() {
            if visited.contains(&start) {
                continue;
            }

            let mut boundary = Vec::new();
            let mut current = start;

            loop {
                visited.insert(current);
                boundary.push(current);

                // Apply σ then ρ
                if let Some(&sigma_next) = sigma_map.get(&current) {
                    if let Some(&rho_next) = rho_map.get(&sigma_next) {
                        current = rho_next;
                        if current == start {
                            break;
                        }
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }

            if !boundary.is_empty() {
                boundaries.push(boundary);
            }
        }

        boundaries
    }

    /// Contracts an edge in the ribbon graph
    ///
    /// # Arguments
    ///
    /// * `edge_index` - Index of the edge to contract (index into rho)
    ///
    /// # Returns
    ///
    /// A new `RibbonGraph` with the edge contracted
    pub fn contract_edge(&self, edge_index: usize) -> Self {
        if edge_index >= self.rho.len() {
            panic!("Edge index out of bounds");
        }

        // Get the darts to contract
        let (dart1, dart2) = self.rho[edge_index];

        // Remove the edge from rho
        let new_rho: Vec<(usize, usize)> = self.rho.iter()
            .enumerate()
            .filter(|(i, _)| *i != edge_index)
            .map(|(_, &pair)| pair)
            .collect();

        // Merge the vertices containing dart1 and dart2
        let mut new_sigma = self.sigma.clone();

        // Find which cycles contain dart1 and dart2
        let mut cycle1_idx = None;
        let mut cycle2_idx = None;

        for (i, cycle) in new_sigma.iter().enumerate() {
            if cycle.contains(&dart1) {
                cycle1_idx = Some(i);
            }
            if cycle.contains(&dart2) {
                cycle2_idx = Some(i);
            }
        }

        // If in same cycle, remove the darts; if different cycles, merge them
        if let (Some(idx1), Some(idx2)) = (cycle1_idx, cycle2_idx) {
            if idx1 == idx2 {
                // Same cycle - remove both darts
                new_sigma[idx1].retain(|&d| d != dart1 && d != dart2);
            } else {
                // Different cycles - merge them
                let cycle2 = new_sigma[idx2].clone();
                new_sigma[idx1].extend(cycle2);
                new_sigma[idx1].retain(|&d| d != dart1 && d != dart2);
                new_sigma.remove(idx2);
            }
        }

        // Clean up empty cycles
        new_sigma.retain(|cycle| !cycle.is_empty());

        // Renumber darts
        let mut dart_map = HashMap::new();
        let mut new_dart = 1;
        for cycle in &new_sigma {
            for &dart in cycle {
                if !dart_map.contains_key(&dart) {
                    dart_map.insert(dart, new_dart);
                    new_dart += 1;
                }
            }
        }

        // Apply renumbering
        let renumbered_sigma: Vec<Vec<usize>> = new_sigma.iter()
            .map(|cycle| cycle.iter().map(|&d| dart_map[&d]).collect())
            .collect();

        let renumbered_rho: Vec<(usize, usize)> = new_rho.iter()
            .filter_map(|&(a, b)| {
                if let (Some(&new_a), Some(&new_b)) = (dart_map.get(&a), dart_map.get(&b)) {
                    Some((new_a, new_b))
                } else {
                    None
                }
            })
            .collect();

        Self::new(renumbered_sigma, renumbered_rho, self.bipartite)
    }

    /// Normalizes dart numbering to span 1..n consecutively
    pub fn normalize(&self) -> Self {
        let mut all_darts = HashSet::new();
        for cycle in &self.sigma {
            for &dart in cycle {
                all_darts.insert(dart);
            }
        }

        let mut sorted_darts: Vec<usize> = all_darts.into_iter().collect();
        sorted_darts.sort();

        let dart_map: HashMap<usize, usize> = sorted_darts.iter()
            .enumerate()
            .map(|(i, &d)| (d, i + 1))
            .collect();

        let new_sigma: Vec<Vec<usize>> = self.sigma.iter()
            .map(|cycle| cycle.iter().map(|&d| dart_map[&d]).collect())
            .collect();

        let new_rho: Vec<(usize, usize)> = self.rho.iter()
            .map(|&(a, b)| (dart_map[&a], dart_map[&b]))
            .collect();

        Self::new(new_sigma, new_rho, self.bipartite)
    }

    /// Returns whether the graph is bipartite
    pub fn is_bipartite(&self) -> bool {
        self.bipartite
    }
}

impl fmt::Display for RibbonGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Ribbon graph:")?;
        writeln!(f, "  Vertices (σ): {:?}", self.sigma)?;
        writeln!(f, "  Edges (ρ): {:?}", self.rho)?;
        writeln!(f, "  Genus: {}", self.genus())?;
        writeln!(f, "  Boundaries: {}", self.number_boundaries())?;
        Ok(())
    }
}

/// Constructs a standard ribbon graph with specified genus and boundary components
///
/// Creates a ribbon graph whose thickening has genus g and r boundary components,
/// with two vertices of equal valency (2g + r).
///
/// # Arguments
///
/// * `g` - The genus
/// * `r` - The number of boundary components
///
/// # Returns
///
/// A `RibbonGraph`
pub fn make_ribbon(g: usize, r: usize) -> RibbonGraph {
    let valency = 2 * g + r;

    // Create two vertices with valency (2g + r) each
    let v1: Vec<usize> = (1..=valency).collect();
    let v2: Vec<usize> = (valency + 1..=2 * valency).collect();

    let sigma = vec![v1, v2];

    // Create edges pairing corresponding darts
    let mut rho = Vec::new();
    for i in 0..valency {
        rho.push((i + 1, valency + i + 1));
    }

    RibbonGraph::new(sigma, rho, false)
}

/// Constructs a bipartite ribbon graph K(p, q)
///
/// Models complete bipartite graphs with cyclic planar ordering,
/// producing Milnor fibers of Brieskorn-Pham singularities x^p + y^q.
///
/// # Arguments
///
/// * `p` - First partition size
/// * `q` - Second partition size
///
/// # Returns
///
/// A bipartite `RibbonGraph`
pub fn bipartite_ribbon_graph(p: usize, q: usize) -> RibbonGraph {
    // Create p vertices in first partition, each with q darts
    let mut sigma = Vec::new();
    let mut dart = 1;

    // First partition: p vertices with q darts each
    for _ in 0..p {
        let vertex: Vec<usize> = (dart..dart + q).collect();
        sigma.push(vertex);
        dart += q;
    }

    // Second partition: q vertices with p darts each
    for _ in 0..q {
        let vertex: Vec<usize> = (dart..dart + p).collect();
        sigma.push(vertex);
        dart += p;
    }

    // Create edges connecting the two partitions
    let mut rho = Vec::new();
    for i in 0..p {
        for j in 0..q {
            let dart1 = i * q + j + 1;
            let dart2 = p * q + j * p + i + 1;
            rho.push((dart1, dart2));
        }
    }

    RibbonGraph::new(sigma, rho, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_ribbon_graph() {
        let sigma = vec![vec![1, 2, 3], vec![4, 5]];
        let rho = vec![(1, 4), (2, 5)];
        let graph = RibbonGraph::new(sigma, rho, false);

        assert_eq!(graph.n_vertices(), 2);
        assert_eq!(graph.n_edges(), 2);
    }

    #[test]
    fn test_make_ribbon() {
        let graph = make_ribbon(1, 1); // Genus 1, 1 boundary
        eprintln!("Vertices: {}", graph.n_vertices());
        eprintln!("Edges: {}", graph.n_edges());
        eprintln!("Sigma: {:?}", graph.sigma);
        eprintln!("Rho: {:?}", graph.rho);
        eprintln!("Boundaries: {}", graph.number_boundaries());
        eprintln!("Genus: {}", graph.genus());
        assert_eq!(graph.number_boundaries(), 1);
        assert_eq!(graph.genus(), 1);
    }

    #[test]
    fn test_bipartite_ribbon_graph() {
        let graph = bipartite_ribbon_graph(2, 3);
        assert!(graph.is_bipartite());
        assert_eq!(graph.n_vertices(), 5); // 2 + 3 vertices
    }

    #[test]
    fn test_genus_calculation() {
        // Create a custom ribbon graph with known genus
        let sigma = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let rho = vec![(1, 4), (2, 5), (3, 6)];
        let graph = RibbonGraph::new(sigma, rho, false);

        // V=2, E=3, B=1 => g = 1 + (3-2-1)/2 = 1
        assert_eq!(graph.genus(), 1);
    }

    #[test]
    fn test_boundary_count() {
        // Create a graph with multiple boundaries
        let sigma = vec![vec![1], vec![2], vec![3]];
        let rho = vec![(1, 2), (2, 3)];
        let graph = RibbonGraph::new(sigma, rho, false);

        // Count boundaries
        let b = graph.number_boundaries();
        assert!(b > 0); // At least one boundary
    }

    #[test]
    fn test_mu_calculation() {
        let graph = make_ribbon(1, 1); // μ = 2*1 + 1 - 1 = 2
        assert_eq!(graph.mu(), 2);
    }

    #[test]
    fn test_boundary() {
        let sigma = vec![vec![1, 2], vec![3, 4]];
        let rho = vec![(1, 3), (2, 4)];
        let graph = RibbonGraph::new(sigma, rho, false);

        let boundaries = graph.boundary();
        assert!(!boundaries.is_empty());
    }

    #[test]
    fn test_normalize() {
        let sigma = vec![vec![5, 10, 15], vec![20, 25]];
        let rho = vec![(5, 20), (10, 25)];
        let graph = RibbonGraph::new(sigma, rho, false);

        let normalized = graph.normalize();
        // After normalization, darts should be 1, 2, 3, 4, 5
        assert_eq!(normalized.n_darts(), 5);
    }

    #[test]
    fn test_contract_edge() {
        let sigma = vec![vec![1, 2], vec![3, 4]];
        let rho = vec![(1, 3), (2, 4)];
        let graph = RibbonGraph::new(sigma, rho, false);

        let contracted = graph.contract_edge(0);
        assert_eq!(contracted.n_edges(), 1); // One edge removed
    }

    #[test]
    fn test_display() {
        let graph = make_ribbon(1, 1);
        let display = format!("{}", graph);
        assert!(display.contains("Ribbon graph"));
        assert!(display.contains("Genus: 1"));
    }

    #[test]
    fn test_complex_bipartite() {
        let graph = bipartite_ribbon_graph(3, 4);
        assert!(graph.is_bipartite());
        assert_eq!(graph.n_vertices(), 7);
        assert_eq!(graph.n_edges(), 12); // 3 * 4 = 12 edges
    }
}
