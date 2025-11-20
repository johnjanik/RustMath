//! Composition tableaux and descent set operations
//!
//! A composition tableau is a filling of the Ferrers diagram of a composition
//! with positive integers such that:
//! - Entries are weakly increasing along rows (left to right)
//! - Entries are strictly increasing down columns
//!
//! Unlike Young tableaux (based on partitions), composition tableaux are based
//! on compositions, which are ordered sequences of positive integers.

use crate::composition::Composition;
use std::collections::BTreeSet;

/// A composition tableau - a filling of a composition's Ferrers diagram
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositionTableau {
    /// The entries of the tableau, organized by rows
    rows: Vec<Vec<usize>>,
    /// The composition (shape) of the tableau
    composition: Composition,
}

impl CompositionTableau {
    /// Create a composition tableau from rows
    ///
    /// Returns None if the rows don't match the composition or don't satisfy
    /// the tableau property (weakly increasing rows, strictly increasing columns)
    pub fn new(rows: Vec<Vec<usize>>) -> Option<Self> {
        if rows.is_empty() {
            return Some(CompositionTableau {
                rows: vec![],
                composition: Composition::new(vec![]).unwrap(),
            });
        }

        // Extract the composition from row lengths
        let parts: Vec<usize> = rows.iter().map(|row| row.len()).collect();
        let composition = Composition::new(parts)?;

        // Verify tableau property
        // 1. Rows are weakly increasing (non-decreasing)
        for row in &rows {
            for i in 1..row.len() {
                if row[i] < row[i - 1] {
                    return None; // Row not weakly increasing
                }
            }
        }

        // 2. Columns are strictly increasing
        for row_idx in 1..rows.len() {
            let min_len = rows[row_idx].len().min(rows[row_idx - 1].len());
            for col_idx in 0..min_len {
                if rows[row_idx][col_idx] <= rows[row_idx - 1][col_idx] {
                    return None; // Column not strictly increasing
                }
            }
        }

        Some(CompositionTableau { rows, composition })
    }

    /// Get the composition (shape) of the tableau
    pub fn composition(&self) -> &Composition {
        &self.composition
    }

    /// Get the rows of the tableau
    pub fn rows(&self) -> &[Vec<usize>] {
        &self.rows
    }

    /// Get the number of entries in the tableau
    pub fn size(&self) -> usize {
        self.rows.iter().map(|row| row.len()).sum()
    }

    /// Get entry at position (row, col), if it exists
    pub fn get(&self, row: usize, col: usize) -> Option<usize> {
        self.rows.get(row)?.get(col).copied()
    }

    /// Get the number of rows
    pub fn num_rows(&self) -> usize {
        self.rows.len()
    }

    /// Check if this is a standard composition tableau
    ///
    /// A standard composition tableau contains each of 1, 2, ..., n exactly once
    pub fn is_standard(&self) -> bool {
        let n = self.size();
        if n == 0 {
            return true;
        }

        // Check that we have exactly the numbers 1..=n
        let mut seen = vec![false; n + 1];
        for row in &self.rows {
            for &entry in row {
                if entry == 0 || entry > n {
                    return false;
                }
                if seen[entry] {
                    return false; // Duplicate
                }
                seen[entry] = true;
            }
        }

        // Check all entries from 1 to n are present
        for i in 1..=n {
            if !seen[i] {
                return false;
            }
        }

        true
    }

    /// Compute the descent set of the tableau
    ///
    /// For a composition tableau T, a descent occurs at position i if:
    /// - The rightmost entry of row i is >= the leftmost entry of row i+1
    ///
    /// The descent set is the set of all positions where descents occur.
    /// These positions correspond to the composition's "cuts" where descents happen.
    pub fn descent_set(&self) -> BTreeSet<usize> {
        let mut descents = BTreeSet::new();

        if self.rows.is_empty() {
            return descents;
        }

        let mut position = 0;

        for row_idx in 0..self.rows.len() - 1 {
            position += self.rows[row_idx].len();

            // Check if there's a descent from row_idx to row_idx+1
            if let (Some(&last_in_row), Some(&first_in_next)) = (
                self.rows[row_idx].last(),
                self.rows[row_idx + 1].first(),
            ) {
                if last_in_row >= first_in_next {
                    descents.insert(position);
                }
            }
        }

        descents
    }

    /// Check if there is a descent at a given position
    ///
    /// Position i means after the i-th entry (1-indexed in the composition sense)
    pub fn has_descent_at(&self, position: usize) -> bool {
        self.descent_set().contains(&position)
    }

    /// Get the number of descents in the tableau
    pub fn number_of_descents(&self) -> usize {
        self.descent_set().len()
    }

    /// Get the descent composition
    ///
    /// The descent composition is formed by the sizes of blocks between descents.
    /// If the descent set is {d1, d2, ..., dk}, then the descent composition
    /// has parts [d1, d2-d1, d3-d2, ..., n-dk]
    pub fn descent_composition(&self) -> Option<Composition> {
        let n = self.size();
        let descents = self.descent_set();

        if descents.is_empty() {
            // No descents - entire tableau is one block
            if n > 0 {
                return Composition::new(vec![n]);
            } else {
                return Composition::new(vec![]);
            }
        }

        let mut parts = Vec::new();
        let mut prev = 0;

        for &d in &descents {
            if d > prev {
                parts.push(d - prev);
            }
            prev = d;
        }

        // Add the final part
        if n > prev {
            parts.push(n - prev);
        }

        Composition::new(parts)
    }

    /// Check if the tableau has no descents (is descent-free)
    ///
    /// A descent-free composition tableau has strictly increasing entries
    /// when read row by row from left to right
    pub fn is_descent_free(&self) -> bool {
        self.descent_set().is_empty()
    }

    /// Get the reading word (concatenation of rows from top to bottom)
    pub fn reading_word(&self) -> Vec<usize> {
        self.rows
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect()
    }

    /// Get the content (multiset of entries) of the tableau
    pub fn content(&self) -> Vec<usize> {
        let mut entries = self.reading_word();
        entries.sort_unstable();
        entries
    }

    /// Display the tableau as a string
    pub fn to_string(&self) -> String {
        self.rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Compute the major index
    ///
    /// The major index is the sum of all descent positions
    pub fn major_index(&self) -> usize {
        self.descent_set().iter().sum()
    }

    /// Create a composition tableau from a composition with consecutive entries
    ///
    /// Fills the composition's shape with 1, 2, 3, ... reading left to right,
    /// top to bottom
    pub fn from_composition_consecutive(comp: &Composition) -> Option<Self> {
        let mut rows = Vec::new();
        let mut counter = 1;

        for &part in comp.parts() {
            let mut row = Vec::new();
            for _ in 0..part {
                row.push(counter);
                counter += 1;
            }
            rows.push(row);
        }

        Self::new(rows)
    }

    /// Check if two composition tableaux have the same descent set
    pub fn has_same_descent_set(&self, other: &CompositionTableau) -> bool {
        self.descent_set() == other.descent_set()
    }

    /// Get the shape as a vector of row lengths
    pub fn shape(&self) -> Vec<usize> {
        self.composition.parts().to_vec()
    }
}

/// Generate all standard composition tableaux for a given composition
///
/// A standard composition tableau contains each of 1, 2, ..., n exactly once,
/// where n is the sum of the composition
pub fn standard_composition_tableaux(comp: &Composition) -> Vec<CompositionTableau> {
    let n = comp.sum();
    if n == 0 {
        return vec![CompositionTableau {
            rows: vec![],
            composition: comp.clone(),
        }];
    }

    let mut result = Vec::new();
    let mut current: Vec<Vec<usize>> = comp
        .parts()
        .iter()
        .map(|&len| vec![0; len])
        .collect();

    generate_standard_comp_tableaux(&mut current, comp, 1, n, &mut result);

    result
}

fn generate_standard_comp_tableaux(
    current: &mut Vec<Vec<usize>>,
    comp: &Composition,
    next_value: usize,
    n: usize,
    result: &mut Vec<CompositionTableau>,
) {
    if next_value > n {
        // All values placed, check if valid and add
        if let Some(tableau) = CompositionTableau::new(current.clone()) {
            result.push(tableau);
        }
        return;
    }

    // Try placing next_value in each valid position
    for r in 0..comp.length() {
        for c in 0..comp.parts()[r] {
            if current[r][c] == 0 && can_place_comp(current, r, c, next_value) {
                current[r][c] = next_value;
                generate_standard_comp_tableaux(current, comp, next_value + 1, n, result);
                current[r][c] = 0;
            }
        }
    }
}

fn can_place_comp(current: &[Vec<usize>], row: usize, col: usize, value: usize) -> bool {
    // Check that this position hasn't been filled
    if current[row][col] != 0 {
        return false;
    }

    // For weakly increasing rows: check left neighbor
    if col > 0 {
        let left = current[row][col - 1];
        if left == 0 {
            return false; // Must fill from left to right
        }
        if left > value {
            return false; // Row must be weakly increasing
        }
    }

    // For strictly increasing columns: check upper neighbor
    if row > 0 && col < current[row - 1].len() {
        let up = current[row - 1][col];
        if up == 0 {
            return false; // Must fill from top to bottom
        }
        if up >= value {
            return false; // Column must be strictly increasing
        }
    }

    true
}

/// Count standard composition tableaux with a given descent set
///
/// Returns the number of standard composition tableaux of size n
/// with the specified descent set
pub fn count_tableaux_with_descent_set(n: usize, descents: &BTreeSet<usize>) -> usize {
    // This is a non-trivial counting problem
    // For now, we use generation and filtering
    // A more efficient approach would use the descent set to determine
    // the composition and then count tableaux of that shape

    if n == 0 {
        return if descents.is_empty() { 1 } else { 0 };
    }

    // Convert descent set to composition
    let comp = descent_set_to_composition(n, descents);
    if comp.is_none() {
        return 0;
    }

    let comp = comp.unwrap();
    let tableaux = standard_composition_tableaux(&comp);

    // Filter by descent set
    tableaux
        .into_iter()
        .filter(|t| &t.descent_set() == descents)
        .count()
}

/// Convert a descent set to its corresponding composition
///
/// Given a descent set D ⊆ {1, 2, ..., n-1}, construct the composition
/// whose parts are determined by the positions of descents
fn descent_set_to_composition(n: usize, descents: &BTreeSet<usize>) -> Option<Composition> {
    if n == 0 {
        return if descents.is_empty() {
            Some(Composition::new(vec![]).unwrap())
        } else {
            None
        };
    }

    // Validate descent set
    for &d in descents {
        if d >= n {
            return None; // Invalid descent position
        }
    }

    let mut parts = Vec::new();
    let mut prev = 0;

    for &d in descents {
        if d > prev {
            parts.push(d - prev);
            prev = d;
        } else {
            return None; // Invalid descent set
        }
    }

    // Add final part
    if n > prev {
        parts.push(n - prev);
    }

    Composition::new(parts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_composition_tableau_creation() {
        // Valid composition tableau with composition [3, 2]
        let t = CompositionTableau::new(vec![vec![1, 2, 3], vec![4, 5]]).unwrap();
        assert_eq!(t.size(), 5);
        assert_eq!(t.composition().parts(), &[3, 2]);
    }

    #[test]
    fn test_composition_tableau_validation() {
        // Valid: weakly increasing rows, strictly increasing columns
        let t1 = CompositionTableau::new(vec![vec![1, 1, 2], vec![3, 4]]);
        assert!(t1.is_some());

        // Invalid: row not weakly increasing
        let t2 = CompositionTableau::new(vec![vec![2, 1, 3], vec![4, 5]]);
        assert!(t2.is_none());

        // Invalid: column not strictly increasing
        let t3 = CompositionTableau::new(vec![vec![1, 2, 3], vec![1, 4]]);
        assert!(t3.is_none());

        // Valid: column can be equal for different length rows
        let t4 = CompositionTableau::new(vec![vec![1, 2], vec![3, 4, 5]]);
        assert!(t4.is_some());
    }

    #[test]
    fn test_standard_composition_tableau() {
        // Standard tableau
        let t = CompositionTableau::new(vec![vec![1, 2, 4], vec![3, 5]]).unwrap();
        assert!(t.is_standard());

        // Not standard - missing entry
        let t2 = CompositionTableau::new(vec![vec![1, 2, 4], vec![5, 6]]).unwrap();
        assert!(!t2.is_standard());

        // Not standard - duplicate entry
        let t3 = CompositionTableau::new(vec![vec![1, 2, 2], vec![3, 4]]).unwrap();
        assert!(!t3.is_standard());
    }

    #[test]
    fn test_descent_set() {
        // Tableau: [[1, 2, 4], [3, 5]]
        // After row 0 (position 3): last=4, first of next=3, 4>=3 -> descent
        let t = CompositionTableau::new(vec![vec![1, 2, 4], vec![3, 5]]).unwrap();
        let descents = t.descent_set();
        assert_eq!(descents.len(), 1);
        assert!(descents.contains(&3));

        // No descents: [[1, 2], [3, 4]]
        let t2 = CompositionTableau::new(vec![vec![1, 2], vec![3, 4]]).unwrap();
        assert!(t2.is_descent_free());
        assert_eq!(t2.number_of_descents(), 0);

        // Multiple descents: [[1, 3], [2, 5], [4]]
        // After row 0 (pos 2): 3>=2 -> descent at 2
        // After row 1 (pos 4): 5>=4 -> descent at 4
        let t3 = CompositionTableau::new(vec![vec![1, 3], vec![2, 5], vec![4]]).unwrap();
        let descents3 = t3.descent_set();
        assert_eq!(descents3.len(), 2);
        assert!(descents3.contains(&2));
        assert!(descents3.contains(&4));
    }

    #[test]
    fn test_descent_composition() {
        // Tableau with descent at position 3: comp should be [3, 2]
        let t = CompositionTableau::new(vec![vec![1, 2, 4], vec![3, 5]]).unwrap();
        let desc_comp = t.descent_composition().unwrap();
        assert_eq!(desc_comp.parts(), &[3, 2]);

        // No descents: composition should be [n]
        let t2 = CompositionTableau::new(vec![vec![1, 2], vec![3, 4]]).unwrap();
        let desc_comp2 = t2.descent_composition().unwrap();
        assert_eq!(desc_comp2.parts(), &[4]);

        // Multiple descents at positions 2, 4: comp should be [2, 2, 1]
        let t3 = CompositionTableau::new(vec![vec![1, 3], vec![2, 5], vec![4]]).unwrap();
        let desc_comp3 = t3.descent_composition().unwrap();
        assert_eq!(desc_comp3.parts(), &[2, 2, 1]);
    }

    #[test]
    fn test_major_index() {
        // Descent at position 3: major index = 3
        let t = CompositionTableau::new(vec![vec![1, 2, 4], vec![3, 5]]).unwrap();
        assert_eq!(t.major_index(), 3);

        // Descents at positions 2, 4: major index = 2 + 4 = 6
        let t2 = CompositionTableau::new(vec![vec![1, 3], vec![2, 5], vec![4]]).unwrap();
        assert_eq!(t2.major_index(), 6);

        // No descents: major index = 0
        let t3 = CompositionTableau::new(vec![vec![1, 2], vec![3, 4]]).unwrap();
        assert_eq!(t3.major_index(), 0);
    }

    #[test]
    fn test_reading_word() {
        let t = CompositionTableau::new(vec![vec![1, 2, 4], vec![3, 5]]).unwrap();
        assert_eq!(t.reading_word(), vec![1, 2, 4, 3, 5]);
    }

    #[test]
    fn test_content() {
        let t = CompositionTableau::new(vec![vec![1, 3, 5], vec![2, 4]]).unwrap();
        assert_eq!(t.content(), vec![1, 2, 3, 4, 5]);

        let t2 = CompositionTableau::new(vec![vec![1, 1, 2], vec![3, 3]]).unwrap();
        assert_eq!(t2.content(), vec![1, 1, 2, 3, 3]);
    }

    #[test]
    fn test_from_composition_consecutive() {
        let comp = Composition::new(vec![3, 2, 1]).unwrap();
        let t = CompositionTableau::from_composition_consecutive(&comp).unwrap();

        assert_eq!(t.rows(), &[vec![1, 2, 3], vec![4, 5], vec![6]]);
        assert!(t.is_standard());
    }

    #[test]
    fn test_generate_standard_composition_tableaux() {
        // Generate all standard composition tableaux for [2, 1]
        let comp = Composition::new(vec![2, 1]).unwrap();
        let tableaux = standard_composition_tableaux(&comp);

        // Should generate multiple tableaux
        assert!(tableaux.len() > 0);

        // All should be standard
        for t in &tableaux {
            assert!(t.is_standard());
            assert_eq!(t.composition(), &comp);
        }

        // Check for specific tableaux
        // [[1, 2], [3]] and [[1, 3], [2]]
        let t1_expected = CompositionTableau::new(vec![vec![1, 2], vec![3]]).unwrap();
        let t2_expected = CompositionTableau::new(vec![vec![1, 3], vec![2]]).unwrap();

        assert!(tableaux.contains(&t1_expected));
        assert!(tableaux.contains(&t2_expected));
    }

    #[test]
    fn test_has_same_descent_set() {
        let t1 = CompositionTableau::new(vec![vec![1, 2, 4], vec![3, 5]]).unwrap();
        let t2 = CompositionTableau::new(vec![vec![1, 3, 5], vec![2, 4]]).unwrap();

        // Both have descent at position 3
        assert!(t1.has_same_descent_set(&t2));
    }

    #[test]
    fn test_descent_set_to_composition() {
        // Descents at {2, 4} with n=5 should give composition [2, 2, 1]
        let mut descents = BTreeSet::new();
        descents.insert(2);
        descents.insert(4);

        let comp = descent_set_to_composition(5, &descents).unwrap();
        assert_eq!(comp.parts(), &[2, 2, 1]);

        // No descents with n=5 should give [5]
        let empty_descents = BTreeSet::new();
        let comp2 = descent_set_to_composition(5, &empty_descents).unwrap();
        assert_eq!(comp2.parts(), &[5]);

        // Descents at {3} with n=5 should give [3, 2]
        let mut descents3 = BTreeSet::new();
        descents3.insert(3);
        let comp3 = descent_set_to_composition(5, &descents3).unwrap();
        assert_eq!(comp3.parts(), &[3, 2]);
    }

    #[test]
    fn test_empty_tableau() {
        let t = CompositionTableau::new(vec![]).unwrap();
        assert_eq!(t.size(), 0);
        assert!(t.is_standard());
        assert!(t.is_descent_free());
        assert_eq!(t.major_index(), 0);
        assert_eq!(t.reading_word(), Vec::<usize>::new());
    }

    #[test]
    fn test_single_row_tableau() {
        // Single row - no descents possible
        let t = CompositionTableau::new(vec![vec![1, 2, 3, 4]]).unwrap();
        assert!(t.is_descent_free());
        assert_eq!(t.number_of_descents(), 0);
        assert_eq!(t.major_index(), 0);
    }

    #[test]
    fn test_weakly_increasing_rows() {
        // Rows can have equal elements (weakly increasing)
        let t = CompositionTableau::new(vec![vec![1, 1, 1], vec![2, 2]]).unwrap();
        assert_eq!(t.size(), 5);
        assert!(!t.is_standard()); // Not standard due to repeated 1s

        // But still a valid composition tableau
        assert!(t.is_descent_free()); // 1 < 2
    }

    #[test]
    fn test_count_tableaux_with_descent_set() {
        // Count standard composition tableaux of size 3 with descent at {2}
        let mut descents = BTreeSet::new();
        descents.insert(2);

        let count = count_tableaux_with_descent_set(3, &descents);
        assert!(count > 0);

        // There should be tableaux with this descent set
        // For n=3, descent at 2 means composition [2, 1]
        let comp = Composition::new(vec![2, 1]).unwrap();
        let all_tableaux = standard_composition_tableaux(&comp);
        let with_descent = all_tableaux
            .iter()
            .filter(|t| &t.descent_set() == &descents)
            .count();

        assert_eq!(count, with_descent);
    }
}
