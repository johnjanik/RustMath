//! Right-Angled Artin Groups (RAAGs)
//!
//! This module implements right-angled Artin groups, which are groups defined by a graph
//! where generators commute if and only if their corresponding vertices are adjacent.
//!
//! # Mathematical Structure
//!
//! A Right-Angled Artin Group (RAAG) for a graph Γ is defined as:
//! ```text
//! A_Γ = ⟨v ∈ V(Γ) | [u, v] = 1 if {u, v} ∈ E(Γ)⟩
//! ```
//!
//! That is, generators corresponding to adjacent vertices commute, and there are
//! no other relations. RAAGs encompass:
//! - Free groups (when Γ has no edges)
//! - Free abelian groups (when Γ is complete)
//! - Many intermediate structures
//!
//! # Normal Form
//!
//! Elements are represented in a normal form where the word is organized into
//! "syllables" of mutually commuting generators, with each syllable in canonical order.
//!
//! # Examples
//!
//! ```
//! use rustmath_groups::raag::RightAngledArtinGroup;
//! use rustmath_graphs::Graph;
//!
//! // Create a path graph with 3 vertices
//! let mut graph = Graph::new(3);
//! graph.add_edge(0, 1).unwrap();
//! graph.add_edge(1, 2).unwrap();
//!
//! // Create the corresponding RAAG
//! let raag = RightAngledArtinGroup::new(graph);
//! let g0 = raag.gen(0);
//! let g1 = raag.gen(1);
//!
//! // g0 and g1 commute since they're adjacent
//! assert_eq!(g0.clone() * g1.clone(), g1.clone() * g0.clone());
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops::Mul;
use rustmath_graphs::Graph;
use crate::group_traits::{Group, GroupElement};

/// A Right-Angled Artin Group defined by a graph
///
/// The group is defined by generators corresponding to vertices, with the
/// relation that generators commute if and only if their vertices are adjacent.
#[derive(Debug, Clone)]
pub struct RightAngledArtinGroup {
    /// The defining graph (vertices = generators, edges = commutation relations)
    graph: Graph,
    /// Generator names (indexed by vertex number)
    gen_names: Vec<String>,
}

impl RightAngledArtinGroup {
    /// Create a new Right-Angled Artin Group from a graph
    ///
    /// # Arguments
    /// * `graph` - The defining graph
    ///
    /// # Returns
    /// A new RAAG with generators named g0, g1, g2, ...
    pub fn new(graph: Graph) -> Self {
        let n = graph.num_vertices();
        let gen_names = (0..n).map(|i| format!("g{}", i)).collect();
        Self { graph, gen_names }
    }

    /// Create a new Right-Angled Artin Group with custom generator names
    ///
    /// # Arguments
    /// * `graph` - The defining graph
    /// * `names` - Names for the generators
    ///
    /// # Panics
    /// Panics if the number of names doesn't match the number of vertices
    pub fn with_names(graph: Graph, names: Vec<String>) -> Self {
        assert_eq!(
            graph.num_vertices(),
            names.len(),
            "Number of names must match number of vertices"
        );
        Self {
            graph,
            gen_names: names,
        }
    }

    /// Get the defining graph
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Get the number of generators
    pub fn ngens(&self) -> usize {
        self.graph.num_vertices()
    }

    /// Get the i-th generator
    ///
    /// # Arguments
    /// * `i` - Generator index (0-based)
    ///
    /// # Panics
    /// Panics if i >= ngens()
    pub fn gen(&self, i: usize) -> RightAngledArtinGroupElement {
        assert!(i < self.ngens(), "Generator index out of bounds");
        RightAngledArtinGroupElement {
            group: self.clone(),
            word: vec![(i, 1)],
        }
    }

    /// Get all generators
    pub fn gens(&self) -> Vec<RightAngledArtinGroupElement> {
        (0..self.ngens()).map(|i| self.gen(i)).collect()
    }

    /// Get the name of the i-th generator
    pub fn gen_name(&self, i: usize) -> &str {
        &self.gen_names[i]
    }

    /// Check if two generators commute
    ///
    /// Generators commute if their corresponding vertices are adjacent in the graph.
    pub fn commute(&self, i: usize, j: usize) -> bool {
        if i == j {
            return true;
        }
        self.graph.has_edge(i, j)
    }

    /// Get the identity element
    pub fn identity(&self) -> RightAngledArtinGroupElement {
        RightAngledArtinGroupElement {
            group: self.clone(),
            word: vec![],
        }
    }

    /// Create a RAAG for a free group of rank n
    ///
    /// This is a RAAG with no edges (no generators commute).
    pub fn free_group(n: usize) -> Self {
        Self::new(Graph::new(n))
    }

    /// Create a RAAG for a free abelian group of rank n
    ///
    /// This is a RAAG with a complete graph (all generators commute).
    pub fn free_abelian_group(n: usize) -> Self {
        let mut graph = Graph::new(n);
        for i in 0..n {
            for j in i + 1..n {
                graph.add_edge(i, j).unwrap();
            }
        }
        Self::new(graph)
    }
}

impl fmt::Display for RightAngledArtinGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Right-Angled Artin Group with {} generators",
            self.ngens()
        )
    }
}

impl Group for RightAngledArtinGroup {
    type Element = RightAngledArtinGroupElement;

    fn identity(&self) -> Self::Element {
        self.identity()
    }

    fn is_finite(&self) -> bool {
        false // RAAGs are generally infinite (unless the graph has no vertices)
    }

    fn order(&self) -> Option<usize> {
        if self.ngens() == 0 {
            Some(1) // Trivial group
        } else {
            None // Infinite
        }
    }

    fn contains(&self, element: &Self::Element) -> bool {
        // Check if element belongs to this group (same number of generators)
        element.group().ngens() == self.ngens()
    }
}

/// An element of a Right-Angled Artin Group
///
/// Elements are represented as words in the generators, stored as a list
/// of (generator_index, exponent) pairs in normal form.
#[derive(Debug, Clone)]
pub struct RightAngledArtinGroupElement {
    group: RightAngledArtinGroup,
    /// Word representation: list of (generator index, exponent)
    word: Vec<(usize, i32)>,
}

impl RightAngledArtinGroupElement {
    /// Get the underlying group
    pub fn group(&self) -> &RightAngledArtinGroup {
        &self.group
    }

    /// Get the word representation
    pub fn word(&self) -> &[(usize, i32)] {
        &self.word
    }

    /// Check if this is the identity element
    pub fn is_identity(&self) -> bool {
        self.word.is_empty()
    }

    /// Get the length of the word (sum of absolute values of exponents)
    pub fn length(&self) -> usize {
        self.word.iter().map(|(_, exp)| exp.abs() as usize).sum()
    }

    /// Compute the inverse
    pub fn inverse(&self) -> Self {
        let inv_word: Vec<(usize, i32)> = self
            .word
            .iter()
            .rev()
            .map(|(gen, exp)| (*gen, -exp))
            .collect();
        Self {
            group: self.group.clone(),
            word: inv_word,
        }
    }

    /// Raise to a power
    pub fn pow(&self, n: i32) -> Self {
        if n == 0 {
            return self.group.identity();
        }
        if n < 0 {
            return self.inverse().pow(-n);
        }

        let mut result = self.group.identity();
        for _ in 0..n {
            result = result * self.clone();
        }
        result
    }

    /// Compute normal form of the word
    ///
    /// The normal form groups mutually commuting generators together and
    /// arranges them in canonical order.
    fn normalize(&mut self) {
        if self.word.is_empty() {
            return;
        }

        loop {
            let old_word = self.word.clone();

            // Combine adjacent equal generators
            let mut new_word = Vec::new();
            let mut i = 0;
            while i < self.word.len() {
                let (gen, mut exp) = self.word[i];

                // Look ahead for more of the same generator
                while i + 1 < self.word.len() && self.word[i + 1].0 == gen {
                    exp += self.word[i + 1].1;
                    i += 1;
                }

                if exp != 0 {
                    new_word.push((gen, exp));
                }
                i += 1;
            }
            self.word = new_word;

            // Try to move commuting generators together
            let mut changed = false;
            for i in 0..self.word.len().saturating_sub(1) {
                let (gen1, _) = self.word[i];
                let (gen2, _) = self.word[i + 1];

                if gen1 != gen2 && self.group.commute(gen1, gen2) && gen1 > gen2 {
                    // Swap them to maintain canonical order
                    self.word.swap(i, i + 1);
                    changed = true;
                }
            }

            if !changed && self.word == old_word {
                break;
            }
        }
    }
}

impl Default for RightAngledArtinGroupElement {
    /// Create a default element (identity of a trivial RAAG)
    fn default() -> Self {
        <Self as GroupElement>::identity()
    }
}

impl Mul for &RightAngledArtinGroupElement {
    type Output = RightAngledArtinGroupElement;

    fn mul(self, other: Self) -> Self::Output {
        self.clone() * other.clone()
    }
}

impl std::ops::Mul for RightAngledArtinGroupElement {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        assert_eq!(
            self.group.ngens(),
            other.group.ngens(),
            "Cannot multiply elements from different groups"
        );

        let mut word = self.word.clone();
        word.extend(other.word.clone());

        let mut result = Self {
            group: self.group.clone(),
            word,
        };
        result.normalize();
        result
    }
}

impl PartialEq for RightAngledArtinGroupElement {
    fn eq(&self, other: &Self) -> bool {
        self.word == other.word
    }
}

impl Eq for RightAngledArtinGroupElement {}

impl std::hash::Hash for RightAngledArtinGroupElement {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.word.hash(state);
    }
}

impl GroupElement for RightAngledArtinGroupElement {
    fn identity() -> Self {
        // Note: This is a bit awkward since we need group context
        // In practice, users should call group.identity() instead
        RightAngledArtinGroupElement {
            group: RightAngledArtinGroup::free_group(0),
            word: vec![],
        }
    }

    fn inverse(&self) -> Self {
        Self::inverse(self)
    }

    fn op(&self, other: &Self) -> Self {
        self.clone() * other.clone()
    }
}

impl fmt::Display for RightAngledArtinGroupElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_identity() {
            return write!(f, "1");
        }

        for (i, (gen, exp)) in self.word.iter().enumerate() {
            if i > 0 {
                write!(f, "*")?;
            }
            write!(f, "{}", self.group.gen_name(*gen))?;
            if *exp != 1 {
                write!(f, "^{}", exp)?;
            }
        }
        Ok(())
    }
}

/// Cohomology ring of a Right-Angled Artin Group
///
/// The cohomology ring H*(A_Γ; R) is isomorphic to the exterior algebra
/// over R on generators corresponding to vertices, modulo relations
/// e_i ∧ e_j = 0 when (i,j) is an edge.
///
/// Basis elements are indexed by independent sets in the graph.
#[derive(Debug, Clone)]
pub struct CohomologyRAAG {
    /// The underlying RAAG
    raag: RightAngledArtinGroup,
    /// Cached independent sets (basis elements)
    independent_sets: Vec<Vec<usize>>,
}

impl CohomologyRAAG {
    /// Create the cohomology ring for a RAAG
    pub fn new(raag: RightAngledArtinGroup) -> Self {
        let independent_sets = Self::compute_independent_sets(&raag);
        Self {
            raag,
            independent_sets,
        }
    }

    /// Compute all independent sets in the graph
    ///
    /// An independent set is a set of vertices with no edges between them.
    fn compute_independent_sets(raag: &RightAngledArtinGroup) -> Vec<Vec<usize>> {
        let n = raag.ngens();
        let mut sets = vec![vec![]]; // Start with empty set

        // Generate all subsets and check independence
        for mask in 1..(1 << n) {
            let mut set = Vec::new();
            for i in 0..n {
                if mask & (1 << i) != 0 {
                    set.push(i);
                }
            }

            // Check if it's independent (no edges within the set)
            let mut independent = true;
            for i in 0..set.len() {
                for j in i + 1..set.len() {
                    if raag.graph().has_edge(set[i], set[j]) {
                        independent = false;
                        break;
                    }
                }
                if !independent {
                    break;
                }
            }

            if independent {
                sets.push(set);
            }
        }

        sets
    }

    /// Get the RAAG
    pub fn raag(&self) -> &RightAngledArtinGroup {
        &self.raag
    }

    /// Get the number of basis elements
    pub fn dimension(&self) -> usize {
        self.independent_sets.len()
    }

    /// Get the i-th basis element (as an independent set)
    pub fn basis_element(&self, i: usize) -> Option<&[usize]> {
        self.independent_sets.get(i).map(|v| v.as_slice())
    }

    /// Get the degree of a basis element
    ///
    /// The degree is the size of the independent set.
    pub fn degree(&self, i: usize) -> Option<usize> {
        self.basis_element(i).map(|s| s.len())
    }

    /// Get all basis elements of a given degree
    pub fn basis_elements_of_degree(&self, d: usize) -> Vec<usize> {
        (0..self.dimension())
            .filter(|&i| self.degree(i) == Some(d))
            .collect()
    }

    /// Wedge product of two basis elements
    ///
    /// Returns None if the result is zero (i.e., if the sets overlap
    /// or contain adjacent vertices).
    pub fn wedge_product(&self, i: usize, j: usize) -> Option<usize> {
        let set1 = self.basis_element(i)?;
        let set2 = self.basis_element(j)?;

        // Check for overlap
        for &v1 in set1 {
            if set2.contains(&v1) {
                return None;
            }
        }

        // Check for edges between sets
        for &v1 in set1 {
            for &v2 in set2 {
                if self.raag.graph().has_edge(v1, v2) {
                    return None;
                }
            }
        }

        // Compute union and sort
        let mut union: Vec<usize> = set1.iter().chain(set2.iter()).copied().collect();
        union.sort_unstable();

        // Find the index of this independent set
        self.independent_sets.iter().position(|s| s == &union)
    }
}

impl fmt::Display for CohomologyRAAG {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Cohomology ring of RAAG with {} generators and {} basis elements",
            self.raag.ngens(),
            self.dimension()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_free_group() {
        let raag = RightAngledArtinGroup::free_group(2);
        let g0 = raag.gen(0);
        let g1 = raag.gen(1);

        // Generators don't commute in free group
        assert_ne!(g0.clone() * g1.clone(), g1.clone() * g0.clone());
    }

    #[test]
    fn test_free_abelian_group() {
        let raag = RightAngledArtinGroup::free_abelian_group(2);
        let g0 = raag.gen(0);
        let g1 = raag.gen(1);

        // All generators commute
        assert_eq!(g0.clone() * g1.clone(), g1.clone() * g0.clone());
    }

    #[test]
    fn test_path_graph() {
        // Path graph: 0-1-2
        let mut graph = Graph::new(3);
        graph.add_edge(0, 1).unwrap();
        graph.add_edge(1, 2).unwrap();

        let raag = RightAngledArtinGroup::new(graph);
        let g0 = raag.gen(0);
        let g1 = raag.gen(1);
        let g2 = raag.gen(2);

        // Adjacent generators commute
        assert_eq!(g0.clone() * g1.clone(), g1.clone() * g0.clone());
        assert_eq!(g1.clone() * g2.clone(), g2.clone() * g1.clone());

        // Non-adjacent generators don't commute
        assert_ne!(g0.clone() * g2.clone(), g2.clone() * g0.clone());
    }

    #[test]
    fn test_identity() {
        let raag = RightAngledArtinGroup::free_group(2);
        let g0 = raag.gen(0);
        let id = raag.identity();

        assert_eq!(g0.clone() * id.clone(), g0);
        assert_eq!(id.clone() * g0.clone(), g0);
        assert!(id.is_identity());
    }

    #[test]
    fn test_inverse() {
        let raag = RightAngledArtinGroup::free_group(2);
        let g0 = raag.gen(0);
        let id = raag.identity();

        assert_eq!(g0.clone() * g0.inverse(), id);
        assert_eq!(g0.inverse() * g0.clone(), id);
    }

    #[test]
    fn test_power() {
        let raag = RightAngledArtinGroup::free_group(2);
        let g0 = raag.gen(0);
        let id = raag.identity();

        assert_eq!(g0.pow(0), id);
        assert_eq!(g0.pow(1), g0);
        assert_eq!(g0.pow(-1), g0.inverse());
    }

    #[test]
    fn test_length() {
        let raag = RightAngledArtinGroup::free_group(2);
        let g0 = raag.gen(0);
        let g1 = raag.gen(1);

        assert_eq!(g0.length(), 1);
        assert_eq!((g0.clone() * g1.clone()).length(), 2);
        assert_eq!(raag.identity().length(), 0);
    }

    #[test]
    fn test_cohomology() {
        // Test with a path graph: 0-1-2
        let mut graph = Graph::new(3);
        graph.add_edge(0, 1).unwrap();
        graph.add_edge(1, 2).unwrap();

        let raag = RightAngledArtinGroup::new(graph);
        let cohom = CohomologyRAAG::new(raag);

        // Independent sets: {}, {0}, {1}, {2}, {0,2}
        assert_eq!(cohom.dimension(), 5);

        // Check degrees
        assert_eq!(cohom.degree(0), Some(0)); // empty set
        assert_eq!(cohom.basis_elements_of_degree(1).len(), 3); // {0}, {1}, {2}
        assert_eq!(cohom.basis_elements_of_degree(2).len(), 1); // {0,2}
    }

    #[test]
    fn test_wedge_product() {
        // Test with disconnected graph: 0  1
        let graph = Graph::new(2);
        let raag = RightAngledArtinGroup::new(graph);
        let cohom = CohomologyRAAG::new(raag);

        // Independent sets: {}, {0}, {1}, {0,1}
        assert_eq!(cohom.dimension(), 4);

        // Find indices
        let empty = 0;
        let set0 = cohom.independent_sets.iter().position(|s| s == &vec![0]).unwrap();
        let set1 = cohom.independent_sets.iter().position(|s| s == &vec![1]).unwrap();
        let set01 = cohom.independent_sets.iter().position(|s| s == &vec![0, 1]).unwrap();

        // Test wedge products
        assert_eq!(cohom.wedge_product(empty, set0), Some(set0));
        assert_eq!(cohom.wedge_product(set0, set1), Some(set01));
        assert_eq!(cohom.wedge_product(set0, set0), None); // overlap
    }

    #[test]
    fn test_display() {
        let raag = RightAngledArtinGroup::free_group(2);
        let g0 = raag.gen(0);
        let g1 = raag.gen(1);

        assert_eq!(format!("{}", raag.identity()), "1");
        assert_eq!(format!("{}", g0), "g0");
        assert_eq!(format!("{}", g0.clone() * g1.clone()), "g0*g1");
    }
}
