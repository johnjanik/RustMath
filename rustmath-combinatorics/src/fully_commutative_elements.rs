//! Fully Commutative Elements in Coxeter Groups with Heap Posets
//!
//! This module implements fully commutative (FC) elements in Coxeter groups.
//! An element w in a Coxeter group is fully commutative if any reduced expression
//! for w can be transformed into any other reduced expression using only commutations
//! (i.e., relations of the form s_i s_j = s_j s_i when m_ij = 2).
//!
//! # Heap Posets
//!
//! Each FC element has an associated heap poset that encodes the structure of all
//! its reduced expressions. The heap is a labeled poset where:
//! - Vertices correspond to positions in a reduced word
//! - The partial order encodes which positions must come before others
//! - Labels indicate which generator appears at each position
//!
//! # Mathematical Background
//!
//! - FC elements are characterized by avoiding "braid patterns" - they don't contain
//!   subwords of the form s_i s_j s_i ... (m_ij terms) when m_ij > 2
//! - In type A, FC elements correspond to 321-avoiding permutations
//! - The heap poset fully determines the FC element
//!
//! # Examples
//!
//! ```
//! use rustmath_combinatorics::fully_commutative_elements::{ReducedWord, HeapPoset, SimpleCoxeterMatrix};
//!
//! // Create a Coxeter matrix for type A_3
//! let cox = SimpleCoxeterMatrix::type_a(3);
//!
//! // Create a reduced word: s_0 s_2 s_1
//! let word = ReducedWord::new(vec![0, 2, 1]);
//!
//! // Check if it's fully commutative
//! let is_fc = word.is_fully_commutative(&cox);
//!
//! // Build the heap poset (if FC)
//! if is_fc {
//!     let heap = HeapPoset::from_reduced_word(&word, &cox).unwrap();
//! }
//! ```

use crate::posets::Poset;
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Trait for Coxeter matrices
///
/// This allows the module to work with any Coxeter matrix implementation
pub trait CoxeterMatrix {
    /// Get the rank (number of generators)
    fn rank(&self) -> usize;

    /// Get the entry m_ij (returns 0 for infinity)
    fn get(&self, i: usize, j: usize) -> usize;
}

/// A simple Coxeter matrix implementation
///
/// This is a standalone implementation that doesn't depend on rustmath-groups
#[derive(Debug, Clone)]
pub struct SimpleCoxeterMatrix {
    rank: usize,
    entries: Vec<Vec<usize>>,
}

impl SimpleCoxeterMatrix {
    /// Create a new Coxeter matrix from explicit entries
    ///
    /// # Arguments
    ///
    /// * `entries` - Square matrix where entries[i][j] = m_ij (use 0 for ∞)
    pub fn new(entries: Vec<Vec<usize>>) -> Self {
        let rank = entries.len();
        assert!(rank > 0, "Coxeter matrix must have at least one generator");

        // Verify it's square
        for row in &entries {
            assert_eq!(row.len(), rank, "Coxeter matrix must be square");
        }

        // Verify diagonal is all 1s
        for i in 0..rank {
            assert_eq!(entries[i][i], 1, "Diagonal entries must be 1");
        }

        // Verify symmetry
        for i in 0..rank {
            for j in 0..rank {
                assert_eq!(
                    entries[i][j], entries[j][i],
                    "Coxeter matrix must be symmetric"
                );
            }
        }

        Self { rank, entries }
    }

    /// Create a Coxeter matrix of type A_n
    pub fn type_a(n: usize) -> Self {
        assert!(n >= 1, "Type A requires n >= 1");

        let mut entries = vec![vec![1; n]; n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    entries[i][j] = 1;
                } else if (i as i32 - j as i32).abs() == 1 {
                    entries[i][j] = 3; // Adjacent generators
                } else {
                    entries[i][j] = 2; // Non-adjacent commute
                }
            }
        }

        Self { rank: n, entries }
    }

    /// Create a Coxeter matrix of type B_n / C_n
    pub fn type_b(n: usize) -> Self {
        assert!(n >= 2, "Type B requires n >= 2");

        let mut matrix = Self::type_a(n);
        // Modify the last two generators to have order 4
        matrix.entries[n - 2][n - 1] = 4;
        matrix.entries[n - 1][n - 2] = 4;

        matrix
    }
}

impl CoxeterMatrix for SimpleCoxeterMatrix {
    fn rank(&self) -> usize {
        self.rank
    }

    fn get(&self, i: usize, j: usize) -> usize {
        assert!(i < self.rank && j < self.rank, "Index out of bounds");
        self.entries[i][j]
    }
}

/// A reduced word in a Coxeter group
///
/// Represented as a sequence of generator indices. For a Coxeter group with
/// rank n, valid generator indices are 0, 1, ..., n-1.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ReducedWord {
    /// Sequence of generator indices
    generators: Vec<usize>,
}

impl ReducedWord {
    /// Create a new reduced word from a sequence of generator indices
    pub fn new(generators: Vec<usize>) -> Self {
        Self { generators }
    }

    /// Create the identity element (empty word)
    pub fn identity() -> Self {
        Self {
            generators: Vec::new(),
        }
    }

    /// Get the length of the word
    pub fn length(&self) -> usize {
        self.generators.len()
    }

    /// Get the generator at position i
    pub fn get(&self, i: usize) -> Option<usize> {
        self.generators.get(i).copied()
    }

    /// Get all generators as a slice
    pub fn generators(&self) -> &[usize] {
        &self.generators
    }

    /// Check if this word is fully commutative with respect to a Coxeter matrix
    ///
    /// A word is FC if it contains no braid patterns s_i s_j s_i when m_ij > 2
    pub fn is_fully_commutative<C: CoxeterMatrix>(&self, coxeter_matrix: &C) -> bool {
        let n = self.generators.len();

        if n < 3 {
            return true; // Words of length < 3 are always FC
        }

        // Check for forbidden patterns
        for i in 0..(n - 2) {
            let s1 = self.generators[i];
            let s2 = self.generators[i + 1];

            // Check if we have a pattern s_i s_j s_i where m_ij > 2
            if s1 == self.generators[i + 2] && s1 != s2 {
                let m_ij = coxeter_matrix.get(s1, s2);

                // If m_ij > 2 (not commuting and not ∞), this is a braid pattern
                if m_ij > 2 && m_ij != 0 {
                    return false;
                }
            }
        }

        true
    }

    /// Get all positions where a specific generator appears
    pub fn positions_of(&self, gen: usize) -> Vec<usize> {
        self.generators
            .iter()
            .enumerate()
            .filter_map(|(i, &g)| if g == gen { Some(i) } else { None })
            .collect()
    }

    /// Check if two positions can be swapped (generators commute)
    pub fn can_swap<C: CoxeterMatrix>(&self, i: usize, j: usize, coxeter_matrix: &C) -> bool {
        if i >= self.generators.len() || j >= self.generators.len() || i == j {
            return false;
        }

        let gi = self.generators[i];
        let gj = self.generators[j];

        if gi == gj {
            return false; // Same generator doesn't commute with itself
        }

        // Generators commute if m_ij = 2
        coxeter_matrix.get(gi, gj) == 2
    }

    /// Apply a commutation at positions i and i+1 (if valid)
    pub fn apply_commutation<C: CoxeterMatrix>(&self, i: usize, coxeter_matrix: &C) -> Option<Self> {
        if i + 1 >= self.generators.len() {
            return None;
        }

        if !self.can_swap(i, i + 1, coxeter_matrix) {
            return None;
        }

        let mut new_gens = self.generators.clone();
        new_gens.swap(i, i + 1);
        Some(Self::new(new_gens))
    }

    /// Get all reduced words equivalent to this one via commutations
    ///
    /// Returns all words in the same commutation class (useful for FC elements)
    pub fn commutation_class<C: CoxeterMatrix>(&self, coxeter_matrix: &C) -> Vec<ReducedWord> {
        let mut result = HashSet::new();
        let mut queue = vec![self.clone()];
        result.insert(self.clone());

        while let Some(word) = queue.pop() {
            for i in 0..word.length().saturating_sub(1) {
                if let Some(new_word) = word.apply_commutation(i, coxeter_matrix) {
                    if result.insert(new_word.clone()) {
                        queue.push(new_word);
                    }
                }
            }
        }

        result.into_iter().collect()
    }
}

impl fmt::Display for ReducedWord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.generators.is_empty() {
            write!(f, "1")
        } else {
            write!(f, "s")?;
            for (i, &gen) in self.generators.iter().enumerate() {
                if i > 0 {
                    write!(f, " s")?;
                }
                write!(f, "_{}", gen)?;
            }
            Ok(())
        }
    }
}

/// A heap poset associated with a fully commutative element
///
/// The heap encodes the structure of all reduced expressions for an FC element.
/// Each vertex represents a position in a reduced word, and edges connect
/// positions that are incomparable (whose generators don't commute).
#[derive(Debug, Clone)]
pub struct HeapPoset {
    /// The underlying poset structure
    poset: Poset,
    /// Labels for each element (which generator appears at that position)
    labels: HashMap<usize, usize>,
}

impl HeapPoset {
    /// Create a heap poset from a reduced word
    ///
    /// Returns None if the word is not fully commutative
    pub fn from_reduced_word<C: CoxeterMatrix>(
        word: &ReducedWord,
        coxeter_matrix: &C,
    ) -> Option<Self> {
        if !word.is_fully_commutative(coxeter_matrix) {
            return None;
        }

        let n = word.length();
        if n == 0 {
            // Identity element has empty heap
            return Some(HeapPoset {
                poset: Poset::new(vec![], vec![]),
                labels: HashMap::new(),
            });
        }

        // Elements are positions 0..n
        let elements: Vec<usize> = (0..n).collect();

        // Build covering relations based on the heap structure
        // Position i covers position j if:
        // 1. i < j (i appears before j in the word)
        // 2. Generators at i and j don't commute
        // 3. There's no k with i < k < j such that gen[i] and gen[k] don't commute
        let mut covering_relations = Vec::new();

        for j in 0..n {
            let gen_j = word.generators[j];

            // Find all positions i < j whose generators don't commute with gen_j
            let mut candidates = Vec::new();
            for i in 0..j {
                let gen_i = word.generators[i];
                if gen_i != gen_j && coxeter_matrix.get(gen_i, gen_j) != 2 {
                    candidates.push(i);
                }
            }

            // Among candidates, find maximal ones (covering elements)
            for &i in &candidates {
                let gen_i = word.generators[i];
                let mut is_maximal = true;

                for &k in &candidates {
                    if k > i {
                        let gen_k = word.generators[k];
                        // If k is between i and j and doesn't commute with gen_i,
                        // then i is not maximal
                        if gen_k != gen_i && coxeter_matrix.get(gen_i, gen_k) != 2 {
                            is_maximal = false;
                            break;
                        }
                    }
                }

                if is_maximal {
                    covering_relations.push((i, j));
                }
            }
        }

        let poset = Poset::new(elements, covering_relations);

        // Build label map
        let mut labels = HashMap::new();
        for i in 0..n {
            labels.insert(i, word.generators[i]);
        }

        Some(HeapPoset { poset, labels })
    }

    /// Get the underlying poset
    pub fn poset(&self) -> &Poset {
        &self.poset
    }

    /// Get the label (generator index) for a position
    pub fn label(&self, pos: usize) -> Option<usize> {
        self.labels.get(&pos).copied()
    }

    /// Get all labels
    pub fn labels(&self) -> &HashMap<usize, usize> {
        &self.labels
    }

    /// Get the number of elements in the heap
    pub fn size(&self) -> usize {
        self.poset.elements().len()
    }

    /// Check if two positions are comparable in the heap
    pub fn comparable(&self, i: usize, j: usize) -> bool {
        self.poset.less_than_or_equal(i, j) || self.poset.less_than_or_equal(j, i)
    }

    /// Get all linear extensions of the heap poset
    ///
    /// Each linear extension corresponds to a reduced expression for the FC element
    pub fn linear_extensions(&self) -> Vec<Vec<usize>> {
        self.poset.linear_extensions()
    }

    /// Convert a linear extension to a reduced word
    pub fn linear_extension_to_word(&self, extension: &[usize]) -> ReducedWord {
        let generators: Vec<usize> = extension
            .iter()
            .map(|&pos| self.labels[&pos])
            .collect();
        ReducedWord::new(generators)
    }

    /// Get all reduced words for this FC element
    pub fn all_reduced_words(&self) -> Vec<ReducedWord> {
        self.linear_extensions()
            .iter()
            .map(|ext| self.linear_extension_to_word(ext))
            .collect()
    }
}

impl fmt::Display for HeapPoset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Heap poset with {} elements:", self.size())?;
        for &elem in self.poset.elements() {
            if let Some(label) = self.label(elem) {
                writeln!(f, "  Position {}: s_{}", elem, label)?;
            }
        }
        Ok(())
    }
}

/// A fully commutative element in a Coxeter group
///
/// Combines a reduced word with its heap poset structure
#[derive(Debug, Clone)]
pub struct FullyCommutativeElement {
    /// A representative reduced word
    reduced_word: ReducedWord,
    /// The heap poset
    heap: HeapPoset,
    /// Reference rank (from Coxeter matrix)
    rank: usize,
}

impl FullyCommutativeElement {
    /// Create a new FC element from a reduced word
    ///
    /// Returns None if the word is not fully commutative
    pub fn new<C: CoxeterMatrix>(
        word: ReducedWord,
        coxeter_matrix: &C,
    ) -> Option<Self> {
        let heap = HeapPoset::from_reduced_word(&word, coxeter_matrix)?;
        Some(FullyCommutativeElement {
            reduced_word: word,
            heap,
            rank: coxeter_matrix.rank(),
        })
    }

    /// Create the identity FC element
    pub fn identity(rank: usize) -> Self {
        let word = ReducedWord::identity();
        let heap = HeapPoset {
            poset: Poset::new(vec![], vec![]),
            labels: HashMap::new(),
        };
        FullyCommutativeElement {
            reduced_word: word,
            heap,
            rank,
        }
    }

    /// Get a representative reduced word
    pub fn reduced_word(&self) -> &ReducedWord {
        &self.reduced_word
    }

    /// Get the heap poset
    pub fn heap(&self) -> &HeapPoset {
        &self.heap
    }

    /// Get the length of the element
    pub fn length(&self) -> usize {
        self.reduced_word.length()
    }

    /// Get all reduced expressions for this element
    pub fn all_reduced_expressions(&self) -> Vec<ReducedWord> {
        self.heap.all_reduced_words()
    }

    /// Get the rank of the Coxeter group
    pub fn rank(&self) -> usize {
        self.rank
    }
}

impl fmt::Display for FullyCommutativeElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FC element: {}", self.reduced_word)
    }
}

/// Generate all fully commutative elements up to a given length
///
/// Returns all FC elements in the Coxeter group with length at most max_length
pub fn enumerate_fc_elements<C: CoxeterMatrix>(
    coxeter_matrix: &C,
    max_length: usize,
) -> Vec<FullyCommutativeElement> {
    let rank = coxeter_matrix.rank();
    let mut result = Vec::new();

    // Add identity
    result.push(FullyCommutativeElement::identity(rank));

    if max_length == 0 {
        return result;
    }

    // Use BFS to generate FC elements level by level
    let mut current_words = vec![ReducedWord::identity()];

    for _length in 1..=max_length {
        let mut next_words = Vec::new();

        for word in current_words {
            // Try appending each generator
            for gen in 0..rank {
                let mut new_gens = word.generators.clone();
                new_gens.push(gen);
                let new_word = ReducedWord::new(new_gens);

                // Check if it's FC and reduced
                if new_word.is_fully_commutative(coxeter_matrix) {
                    // Simple reduction check: last two generators shouldn't be the same
                    if word.length() == 0 || word.generators[word.length() - 1] != gen {
                        if let Some(fc_elem) = FullyCommutativeElement::new(new_word.clone(), coxeter_matrix) {
                            // Avoid duplicates by checking if we've seen this commutation class
                            let is_new = !result.iter().any(|existing| {
                                existing.heap.labels == fc_elem.heap.labels
                                    && existing.length() == fc_elem.length()
                            });

                            if is_new {
                                result.push(fc_elem);
                                next_words.push(new_word);
                            }
                        }
                    }
                }
            }
        }

        current_words = next_words;
    }

    result
}

/// Count the number of FC elements of each length
///
/// Returns a vector where result[i] is the number of FC elements of length i
pub fn count_fc_elements_by_length<C: CoxeterMatrix>(
    coxeter_matrix: &C,
    max_length: usize,
) -> Vec<usize> {
    let elements = enumerate_fc_elements(coxeter_matrix, max_length);
    let mut counts = vec![0; max_length + 1];

    for elem in elements {
        counts[elem.length()] += 1;
    }

    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduced_word_creation() {
        let word = ReducedWord::new(vec![0, 1, 2]);
        assert_eq!(word.length(), 3);
        assert_eq!(word.get(0), Some(0));
        assert_eq!(word.get(1), Some(1));
        assert_eq!(word.get(2), Some(2));
    }

    #[test]
    fn test_identity_word() {
        let id = ReducedWord::identity();
        assert_eq!(id.length(), 0);
    }

    #[test]
    fn test_is_fully_commutative_type_a() {
        let cox = SimpleCoxeterMatrix::type_a(3);

        // s_0 s_2 is FC (non-adjacent generators commute)
        let word1 = ReducedWord::new(vec![0, 2]);
        assert!(word1.is_fully_commutative(&cox));

        // s_0 s_1 s_0 is NOT FC (braid pattern with m_01 = 3)
        let word2 = ReducedWord::new(vec![0, 1, 0]);
        assert!(!word2.is_fully_commutative(&cox));

        // s_0 s_2 s_1 is FC
        let word3 = ReducedWord::new(vec![0, 2, 1]);
        assert!(word3.is_fully_commutative(&cox));

        // s_1 s_0 s_1 is NOT FC
        let word4 = ReducedWord::new(vec![1, 0, 1]);
        assert!(!word4.is_fully_commutative(&cox));
    }

    #[test]
    fn test_is_fully_commutative_short_words() {
        let cox = SimpleCoxeterMatrix::type_a(2);

        // All words of length ≤ 2 are FC
        assert!(ReducedWord::new(vec![]).is_fully_commutative(&cox));
        assert!(ReducedWord::new(vec![0]).is_fully_commutative(&cox));
        assert!(ReducedWord::new(vec![1]).is_fully_commutative(&cox));
        assert!(ReducedWord::new(vec![0, 1]).is_fully_commutative(&cox));
    }

    #[test]
    fn test_heap_poset_from_word() {
        let cox = SimpleCoxeterMatrix::type_a(3);

        // s_0 s_2 (commuting generators)
        let word = ReducedWord::new(vec![0, 2]);
        let heap = HeapPoset::from_reduced_word(&word, &cox).unwrap();

        assert_eq!(heap.size(), 2);
        assert_eq!(heap.label(0), Some(0));
        assert_eq!(heap.label(1), Some(2));
    }

    #[test]
    fn test_heap_poset_identity() {
        let cox = SimpleCoxeterMatrix::type_a(2);
        let word = ReducedWord::identity();
        let heap = HeapPoset::from_reduced_word(&word, &cox).unwrap();

        assert_eq!(heap.size(), 0);
    }

    #[test]
    fn test_heap_poset_rejects_non_fc() {
        let cox = SimpleCoxeterMatrix::type_a(2);

        // s_0 s_1 s_0 is not FC
        let word = ReducedWord::new(vec![0, 1, 0]);
        assert!(HeapPoset::from_reduced_word(&word, &cox).is_none());
    }

    #[test]
    fn test_fc_element_creation() {
        let cox = SimpleCoxeterMatrix::type_a(3);

        let word = ReducedWord::new(vec![0, 2]);
        let fc = FullyCommutativeElement::new(word, &cox);

        assert!(fc.is_some());
        let fc = fc.unwrap();
        assert_eq!(fc.length(), 2);
    }

    #[test]
    fn test_fc_element_identity() {
        let id = FullyCommutativeElement::identity(3);
        assert_eq!(id.length(), 0);
        assert_eq!(id.rank(), 3);
    }

    #[test]
    fn test_linear_extensions() {
        let cox = SimpleCoxeterMatrix::type_a(3);

        // s_0 s_2 s_1 - all three commute in pairs
        let word = ReducedWord::new(vec![0, 2, 1]);
        let heap = HeapPoset::from_reduced_word(&word, &cox).unwrap();

        let extensions = heap.linear_extensions();

        // Should have multiple linear extensions since generators commute
        assert!(extensions.len() >= 1);
    }

    #[test]
    fn test_all_reduced_expressions() {
        let cox = SimpleCoxeterMatrix::type_a(3);

        let word = ReducedWord::new(vec![0, 2]);
        let fc = FullyCommutativeElement::new(word, &cox).unwrap();

        let expressions = fc.all_reduced_expressions();

        // s_0 and s_2 commute, so we should get both orderings
        assert_eq!(expressions.len(), 2);

        // Check that we have both s_0 s_2 and s_2 s_0
        let has_02 = expressions.iter().any(|w| w.generators() == &[0, 2]);
        let has_20 = expressions.iter().any(|w| w.generators() == &[2, 0]);
        assert!(has_02 && has_20);
    }

    #[test]
    fn test_enumerate_fc_elements_length_0() {
        let cox = SimpleCoxeterMatrix::type_a(2);
        let elements = enumerate_fc_elements(&cox, 0);

        // Only identity
        assert_eq!(elements.len(), 1);
        assert_eq!(elements[0].length(), 0);
    }

    #[test]
    fn test_enumerate_fc_elements_length_1() {
        let cox = SimpleCoxeterMatrix::type_a(2);
        let elements = enumerate_fc_elements(&cox, 1);

        // Identity + s_0 + s_1 = 3 elements
        assert_eq!(elements.len(), 3);
    }

    #[test]
    fn test_enumerate_fc_elements_type_a2() {
        let cox = SimpleCoxeterMatrix::type_a(2);
        let elements = enumerate_fc_elements(&cox, 2);

        // Should include: identity, s_0, s_1, s_0 s_1, s_1 s_0
        // But NOT s_0 s_1 s_0 or s_1 s_0 s_1 (not FC)
        assert!(elements.len() >= 4);

        // Check that all are FC
        for elem in &elements {
            assert!(elem.reduced_word().is_fully_commutative(&cox));
        }
    }

    #[test]
    fn test_count_fc_elements() {
        let cox = SimpleCoxeterMatrix::type_a(2);
        let counts = count_fc_elements_by_length(&cox, 3);

        // Length 0: identity (1)
        assert_eq!(counts[0], 1);

        // Length 1: s_0, s_1 (2)
        assert_eq!(counts[1], 2);

        // Length 2: should have some FC elements
        assert!(counts[2] >= 1);
    }

    #[test]
    fn test_commutation_class() {
        let cox = SimpleCoxeterMatrix::type_a(3);

        // s_0 s_2 (they commute)
        let word = ReducedWord::new(vec![0, 2]);
        let class = word.commutation_class(&cox);

        // Should contain both s_0 s_2 and s_2 s_0
        assert_eq!(class.len(), 2);
    }

    #[test]
    fn test_can_swap() {
        let cox = SimpleCoxeterMatrix::type_a(3);

        let word = ReducedWord::new(vec![0, 2, 1]);

        // s_0 and s_2 commute (positions 0 and 1)
        assert!(word.can_swap(0, 1, &cox));

        // s_2 and s_1 do not commute (positions 1 and 2)
        assert!(!word.can_swap(1, 2, &cox));
    }

    #[test]
    fn test_apply_commutation() {
        let cox = SimpleCoxeterMatrix::type_a(3);

        let word = ReducedWord::new(vec![0, 2]);
        let swapped = word.apply_commutation(0, &cox).unwrap();

        assert_eq!(swapped.generators(), &[2, 0]);
    }

    #[test]
    fn test_positions_of() {
        let word = ReducedWord::new(vec![0, 1, 0, 2, 0]);
        let positions = word.positions_of(0);

        assert_eq!(positions, vec![0, 2, 4]);
    }

    #[test]
    fn test_display_reduced_word() {
        let word = ReducedWord::new(vec![0, 1, 2]);
        let display = format!("{}", word);
        assert!(display.contains("s_0"));
        assert!(display.contains("s_1"));
        assert!(display.contains("s_2"));
    }

    #[test]
    fn test_display_identity() {
        let id = ReducedWord::identity();
        let display = format!("{}", id);
        assert_eq!(display, "1");
    }

    #[test]
    fn test_fc_element_type_b() {
        let cox = SimpleCoxeterMatrix::type_b(3);

        // Simple FC element in type B
        let word = ReducedWord::new(vec![0, 2]);
        assert!(word.is_fully_commutative(&cox));

        let fc = FullyCommutativeElement::new(word, &cox);
        assert!(fc.is_some());
    }
}
