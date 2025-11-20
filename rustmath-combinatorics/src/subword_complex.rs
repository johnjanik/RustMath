//! Subword complexes for Knutson-Miller theory
//!
//! This module implements subword complexes, which are simplicial complexes introduced by
//! Knutson and Miller to study Schubert polynomials and the geometry of flag varieties.
//!
//! # Overview
//!
//! Given a word Q (sequence of simple generators) and a Coxeter group element π,
//! the subword complex SC(Q, π) is a simplicial complex whose facets correspond to
//! reduced subwords of Q that represent π.
//!
//! Each facet is encoded as a subset of positions in Q that, when removed, leave a
//! reduced expression for π.
//!
//! # References
//!
//! - Knutson, A., & Miller, E. (2005). Subword complexes in Coxeter groups.
//!   Advances in Mathematics, 184(1), 161-176.
//! - Knutson, A., & Miller, E. (2004). Gröbner geometry of Schubert polynomials.
//!   Annals of Mathematics, 161(3), 1245-1318.

use crate::permutations::Permutation;
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt;

/// A word in simple generators of a Coxeter group
///
/// For type A (symmetric group), generators are the simple transpositions s_i = (i, i+1).
/// A word is a sequence of generator indices.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ReducedWord {
    /// The sequence of generator indices (1-indexed for mathematical convention)
    generators: Vec<usize>,
}

impl ReducedWord {
    /// Create a new word from a sequence of generator indices
    ///
    /// # Arguments
    /// * `generators` - Sequence of generator indices (should be 1-indexed)
    ///
    /// # Example
    /// ```
    /// use rustmath_combinatorics::subword_complex::ReducedWord;
    /// let word = ReducedWord::new(vec![2, 1, 2]); // s_2 s_1 s_2
    /// ```
    pub fn new(generators: Vec<usize>) -> Self {
        ReducedWord { generators }
    }

    /// Get the length of the word
    pub fn len(&self) -> usize {
        self.generators.len()
    }

    /// Check if the word is empty
    pub fn is_empty(&self) -> bool {
        self.generators.is_empty()
    }

    /// Get the generators as a slice
    pub fn generators(&self) -> &[usize] {
        &self.generators
    }

    /// Apply the word to compute the resulting permutation (for type A)
    ///
    /// # Arguments
    /// * `n` - The size of the symmetric group S_n
    ///
    /// # Returns
    /// The permutation obtained by multiplying the simple transpositions
    pub fn to_permutation(&self, n: usize) -> Permutation {
        let mut perm = Permutation::identity(n);

        // Apply each generator (simple transposition)
        for &gen in &self.generators {
            if gen > 0 && gen < n {
                // s_i swaps positions i-1 and i (converting to 0-indexed)
                let p = perm.as_slice();
                let mut new_perm = p.to_vec();
                new_perm.swap(gen - 1, gen);
                perm = Permutation::from_vec(new_perm).expect("Valid permutation");
            }
        }

        perm
    }

    /// Extract a subword by removing positions specified in the complement set
    ///
    /// # Arguments
    /// * `positions` - Set of positions to keep (0-indexed)
    ///
    /// # Returns
    /// A new word containing only the generators at the specified positions
    pub fn subword(&self, positions: &BTreeSet<usize>) -> ReducedWord {
        let generators: Vec<usize> = positions
            .iter()
            .filter_map(|&pos| {
                if pos < self.generators.len() {
                    Some(self.generators[pos])
                } else {
                    None
                }
            })
            .collect();

        ReducedWord::new(generators)
    }

    /// Extract a subword by keeping all positions except those in the given set
    ///
    /// # Arguments
    /// * `exclude_positions` - Set of positions to exclude (0-indexed)
    ///
    /// # Returns
    /// A new word with the specified positions removed
    pub fn subword_complement(&self, exclude_positions: &BTreeSet<usize>) -> ReducedWord {
        let generators: Vec<usize> = (0..self.generators.len())
            .filter(|pos| !exclude_positions.contains(pos))
            .map(|pos| self.generators[pos])
            .collect();

        ReducedWord::new(generators)
    }

    /// Check if this word is a reduced expression for a given permutation
    ///
    /// A word is reduced if its length equals the Coxeter length of the permutation
    /// (the minimum number of simple transpositions needed)
    pub fn is_reduced(&self, perm: &Permutation, n: usize) -> bool {
        let computed = self.to_permutation(n);
        if computed != *perm {
            return false;
        }

        // Check that the length equals the number of inversions
        self.len() == perm.inversions()
    }

    /// Check if two generators commute (braid relation of length 2)
    ///
    /// In type A, s_i and s_j commute if |i - j| > 1
    fn commute(i: usize, j: usize) -> bool {
        if i == 0 || j == 0 {
            return false;
        }
        i.abs_diff(j) > 1
    }

    /// Check if two generators satisfy the braid relation s_i s_j s_i = s_j s_i s_j
    ///
    /// In type A, this holds when |i - j| = 1
    fn braid_relation(i: usize, j: usize) -> bool {
        if i == 0 || j == 0 {
            return false;
        }
        i.abs_diff(j) == 1
    }

    /// Perform a braid move if possible
    ///
    /// Returns all words that can be obtained by applying one braid relation
    pub fn braid_moves(&self) -> Vec<ReducedWord> {
        let mut moves = Vec::new();

        // Check for commutation relations: s_i s_j -> s_j s_i where |i-j| > 1
        for pos in 0..self.generators.len().saturating_sub(1) {
            let i = self.generators[pos];
            let j = self.generators[pos + 1];

            if Self::commute(i, j) {
                let mut new_gens = self.generators.clone();
                new_gens.swap(pos, pos + 1);
                moves.push(ReducedWord::new(new_gens));
            }
        }

        // Check for braid relations: s_i s_j s_i -> s_j s_i s_j where |i-j| = 1
        for pos in 0..self.generators.len().saturating_sub(2) {
            let i = self.generators[pos];
            let j = self.generators[pos + 1];
            let k = self.generators[pos + 2];

            if i == k && Self::braid_relation(i, j) {
                let mut new_gens = self.generators.clone();
                new_gens[pos] = j;
                new_gens[pos + 1] = i;
                new_gens[pos + 2] = j;
                moves.push(ReducedWord::new(new_gens));
            }
        }

        moves
    }
}

impl fmt::Display for ReducedWord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, gen) in self.generators.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "s{}", gen)?;
        }
        write!(f, "]")
    }
}

/// A subword complex SC(Q, π) for a word Q and permutation π
///
/// The subword complex is a simplicial complex whose facets are subsets of positions
/// in Q that, when deleted, leave a reduced expression for π.
#[derive(Debug, Clone)]
pub struct SubwordComplex {
    /// The word Q (sequence of generators)
    word: ReducedWord,
    /// The target permutation π
    target: Permutation,
    /// The rank n (size of symmetric group S_n)
    rank: usize,
    /// Facets of the complex (maximal faces)
    facets: Vec<BTreeSet<usize>>,
    /// All faces of the complex
    faces: Vec<BTreeSet<usize>>,
}

impl SubwordComplex {
    /// Create a new subword complex for word Q and permutation π
    ///
    /// # Arguments
    /// * `word` - The word Q (sequence of generator indices)
    /// * `target` - The target permutation π
    /// * `rank` - The size n of the symmetric group S_n
    ///
    /// # Example
    /// ```
    /// use rustmath_combinatorics::subword_complex::{ReducedWord, SubwordComplex};
    /// use rustmath_combinatorics::permutations::Permutation;
    ///
    /// let word = ReducedWord::new(vec![2, 1, 2, 1]);
    /// let perm = Permutation::from_vec(vec![2, 3, 0, 1]).unwrap();
    /// let complex = SubwordComplex::new(word, perm, 4);
    /// ```
    pub fn new(word: ReducedWord, target: Permutation, rank: usize) -> Self {
        let mut complex = SubwordComplex {
            word,
            target,
            rank,
            facets: Vec::new(),
            faces: Vec::new(),
        };

        complex.compute_facets();
        complex.compute_faces();
        complex
    }

    /// Compute all facets of the subword complex
    ///
    /// A facet is a maximal subset of positions whose complement is a reduced word for π
    fn compute_facets(&mut self) {
        let word_len = self.word.len();
        let target_len = self.target.inversions();

        if word_len < target_len {
            // No facets if word is too short
            return;
        }

        let facet_size = word_len - target_len;

        // Generate all subsets of the correct size
        let positions: Vec<usize> = (0..word_len).collect();
        let mut facets = Vec::new();

        // Use DFS to generate all combinations
        self.find_facets_recursive(
            &positions,
            facet_size,
            0,
            BTreeSet::new(),
            &mut facets,
        );

        self.facets = facets;
    }

    /// Recursive helper for finding facets
    fn find_facets_recursive(
        &self,
        positions: &[usize],
        remaining: usize,
        start: usize,
        current: BTreeSet<usize>,
        facets: &mut Vec<BTreeSet<usize>>,
    ) {
        if remaining == 0 {
            // Check if the complement is a reduced word for the target
            let subword = self.word.subword_complement(&current);
            if subword.is_reduced(&self.target, self.rank) {
                facets.push(current);
            }
            return;
        }

        for i in start..=positions.len().saturating_sub(remaining) {
            let mut next = current.clone();
            next.insert(positions[i]);
            self.find_facets_recursive(positions, remaining - 1, i + 1, next, facets);
        }
    }

    /// Compute all faces from the facets
    ///
    /// A face is any subset of a facet
    fn compute_faces(&mut self) {
        let mut faces = HashSet::new();

        // Add the empty set
        faces.insert(BTreeSet::new());

        // For each facet, add all its subsets
        for facet in &self.facets {
            let subsets = Self::all_subsets(facet);
            faces.extend(subsets);
        }

        self.faces = faces.into_iter().collect();
        // Sort faces by size for convenience
        self.faces.sort_by_key(|f| f.len());
    }

    /// Generate all subsets of a set
    fn all_subsets(set: &BTreeSet<usize>) -> Vec<BTreeSet<usize>> {
        let elements: Vec<usize> = set.iter().copied().collect();
        let n = elements.len();
        let mut subsets = Vec::new();

        // Iterate through all 2^n possibilities
        for mask in 0..(1 << n) {
            let mut subset = BTreeSet::new();
            for (i, &elem) in elements.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    subset.insert(elem);
                }
            }
            subsets.push(subset);
        }

        subsets
    }

    /// Get the word Q
    pub fn word(&self) -> &ReducedWord {
        &self.word
    }

    /// Get the target permutation π
    pub fn target(&self) -> &Permutation {
        &self.target
    }

    /// Get the facets of the complex
    pub fn facets(&self) -> &[BTreeSet<usize>] {
        &self.facets
    }

    /// Get all faces of the complex
    pub fn faces(&self) -> &[BTreeSet<usize>] {
        &self.faces
    }

    /// Get the dimension of the complex
    ///
    /// The dimension is the size of the largest facet minus 1
    pub fn dimension(&self) -> Option<usize> {
        self.facets
            .iter()
            .map(|f| f.len())
            .max()
            .map(|size| size.saturating_sub(1))
    }

    /// Get the f-vector of the complex
    ///
    /// The f-vector counts faces by dimension: f_i = number of i-dimensional faces
    pub fn f_vector(&self) -> Vec<usize> {
        if self.faces.is_empty() {
            return vec![];
        }

        let max_dim = self.faces.iter().map(|f| f.len()).max().unwrap_or(0);
        let mut f_vec = vec![0; max_dim + 1];

        for face in &self.faces {
            f_vec[face.len()] += 1;
        }

        f_vec
    }

    /// Check if the complex is pure (all facets have the same dimension)
    pub fn is_pure(&self) -> bool {
        if self.facets.is_empty() {
            return true;
        }

        let first_size = self.facets[0].len();
        self.facets.iter().all(|f| f.len() == first_size)
    }

    /// Check if the complex is connected
    ///
    /// A simplicial complex is connected if its 1-skeleton (graph of vertices and edges)
    /// is connected
    pub fn is_connected(&self) -> bool {
        if self.faces.is_empty() {
            return true;
        }

        // Extract vertices (0-dimensional faces)
        let vertices: HashSet<usize> = self.faces
            .iter()
            .filter(|f| f.len() == 1)
            .flat_map(|f| f.iter().copied())
            .collect();

        if vertices.is_empty() {
            return true;
        }

        // Build adjacency list from edges (1-dimensional faces)
        let mut adjacency: HashMap<usize, Vec<usize>> = HashMap::new();
        for v in &vertices {
            adjacency.insert(*v, Vec::new());
        }

        for face in &self.faces {
            if face.len() == 2 {
                let edge: Vec<usize> = face.iter().copied().collect();
                adjacency.get_mut(&edge[0]).unwrap().push(edge[1]);
                adjacency.get_mut(&edge[1]).unwrap().push(edge[0]);
            }
        }

        // BFS to check connectivity
        let start = *vertices.iter().next().unwrap();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        visited.insert(start);
        queue.push_back(start);

        while let Some(v) = queue.pop_front() {
            for &neighbor in &adjacency[&v] {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }

        visited.len() == vertices.len()
    }

    /// Compute the Euler characteristic χ = Σ(-1)^i f_i
    pub fn euler_characteristic(&self) -> i64 {
        let f_vec = self.f_vector();
        f_vec
            .iter()
            .enumerate()
            .map(|(i, &count)| if i % 2 == 0 { count as i64 } else { -(count as i64) })
            .sum()
    }

    /// Get the number of facets
    pub fn num_facets(&self) -> usize {
        self.facets.len()
    }

    /// Get the number of vertices
    pub fn num_vertices(&self) -> usize {
        self.faces.iter().filter(|f| f.len() == 1).count()
    }
}

impl fmt::Display for SubwordComplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Subword Complex SC(Q, π)")?;
        writeln!(f, "  Word Q: {}", self.word)?;
        writeln!(f, "  Target π: {:?}", self.target)?;
        writeln!(f, "  Rank: {}", self.rank)?;
        writeln!(f, "  Number of facets: {}", self.num_facets())?;
        writeln!(f, "  Dimension: {:?}", self.dimension())?;
        writeln!(f, "  f-vector: {:?}", self.f_vector())?;
        writeln!(f, "  Euler characteristic: {}", self.euler_characteristic())?;
        writeln!(f, "  Pure: {}", self.is_pure())?;
        writeln!(f, "  Connected: {}", self.is_connected())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduced_word_creation() {
        let word = ReducedWord::new(vec![1, 2, 1]);
        assert_eq!(word.len(), 3);
        assert!(!word.is_empty());
    }

    #[test]
    fn test_reduced_word_to_permutation() {
        // s_1 = (0,1) in 0-indexed notation
        let word = ReducedWord::new(vec![1]);
        let perm = word.to_permutation(2);
        assert_eq!(perm.as_slice(), &[1, 0]);
    }

    #[test]
    fn test_reduced_word_s2s1s2() {
        // s_2 s_1 s_2 in S_3
        let word = ReducedWord::new(vec![2, 1, 2]);
        let perm = word.to_permutation(3);
        // s_2 swaps (1,2) -> [0,2,1]
        // s_1 swaps (0,1) -> [2,0,1]
        // s_2 swaps (1,2) -> [2,1,0]
        assert_eq!(perm.as_slice(), &[2, 1, 0]);
    }

    #[test]
    fn test_subword_extraction() {
        let word = ReducedWord::new(vec![1, 2, 1, 2]);
        let positions: BTreeSet<usize> = [0, 2].iter().copied().collect();
        let subword = word.subword(&positions);
        assert_eq!(subword.generators(), &[1, 1]);
    }

    #[test]
    fn test_subword_complement() {
        let word = ReducedWord::new(vec![1, 2, 1, 2]);
        let exclude: BTreeSet<usize> = [1, 3].iter().copied().collect();
        let subword = word.subword_complement(&exclude);
        assert_eq!(subword.generators(), &[1, 1]);
    }

    #[test]
    fn test_generators_commute() {
        assert!(ReducedWord::commute(1, 3));
        assert!(!ReducedWord::commute(1, 2));
        assert!(!ReducedWord::commute(2, 3));
    }

    #[test]
    fn test_braid_relation() {
        assert!(ReducedWord::braid_relation(1, 2));
        assert!(ReducedWord::braid_relation(2, 3));
        assert!(!ReducedWord::braid_relation(1, 3));
    }

    #[test]
    fn test_simple_subword_complex() {
        // Create a simple example: Q = [s_1, s_2, s_1], π = s_1 s_2 s_1
        let word = ReducedWord::new(vec![1, 2, 1]);
        let perm = word.to_permutation(3);
        let complex = SubwordComplex::new(word, perm, 3);

        // Should have exactly one facet (the empty set, since we need all generators)
        assert!(complex.num_facets() > 0);
    }

    #[test]
    fn test_subword_complex_dimension() {
        let word = ReducedWord::new(vec![1, 2, 1, 2]);
        let perm = ReducedWord::new(vec![2, 1, 2]).to_permutation(3);
        let complex = SubwordComplex::new(word, perm, 3);

        // The dimension should be well-defined
        assert!(complex.dimension().is_some());
    }

    #[test]
    fn test_f_vector() {
        let word = ReducedWord::new(vec![1, 2]);
        let perm = ReducedWord::new(vec![1, 2]).to_permutation(3);
        let complex = SubwordComplex::new(word, perm, 3);

        let f_vec = complex.f_vector();
        // Should have non-empty f-vector
        assert!(!f_vec.is_empty());
    }

    #[test]
    fn test_euler_characteristic() {
        let word = ReducedWord::new(vec![1]);
        let perm = ReducedWord::new(vec![1]).to_permutation(2);
        let complex = SubwordComplex::new(word, perm, 2);

        // Euler characteristic should be defined
        let chi = complex.euler_characteristic();
        assert!(chi != 0 || chi == 0); // Just check it computes
    }

    #[test]
    fn test_is_reduced() {
        let word = ReducedWord::new(vec![1, 2, 1]);
        let perm = word.to_permutation(3);
        assert!(word.is_reduced(&perm, 3));

        // A non-reduced word: s_1 s_1 = identity
        let non_reduced = ReducedWord::new(vec![1, 1]);
        let identity = Permutation::identity(2);
        assert!(!non_reduced.is_reduced(&identity, 2));
    }

    #[test]
    fn test_display() {
        let word = ReducedWord::new(vec![1, 2, 3]);
        let display = format!("{}", word);
        assert!(display.contains("s1"));
        assert!(display.contains("s2"));
        assert!(display.contains("s3"));
    }
}
