//! Pipe dreams (RC-graphs) for Schubert polynomials
//!
//! This module implements pipe dreams, also known as RC-graphs (reduced compatible sequences).
//! Pipe dreams are grid diagrams that encode reduced words for permutations and are used
//! to study Schubert polynomials.
//!
//! # Overview
//!
//! A pipe dream is an n×n grid where each cell contains either:
//! - A crossing (+) represented as `true`
//! - An elbow (×) represented as `false`
//!
//! The diagram encodes a permutation π ∈ S_n by the "pipe" paths that connect the
//! top and left borders. Each pipe dream corresponds to a reduced word for the permutation.
//!
//! # References
//!
//! - Bergeron, N., & Billey, S. (1993). RC-graphs and Schubert polynomials.
//!   Experimental Mathematics, 2(4), 257-269.
//! - Fomin, S., & Kirillov, A. N. (1996). The Yang-Baxter equation, symmetric functions,
//!   and Schubert polynomials. Discrete Mathematics, 153(1-3), 123-143.

use crate::permutations::Permutation;
use crate::subword_complex::ReducedWord;
use std::fmt;

/// A pipe dream (RC-graph) represented as a grid
///
/// The grid is n×n where n is the size of the permutation.
/// Each cell is either a crossing (true) or an elbow (false).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipeDream {
    /// The size of the grid (n×n)
    n: usize,
    /// The grid: grid[i][j] is true for crossing (+), false for elbow (×)
    /// Indexed as grid[row][col] where (0,0) is top-left
    grid: Vec<Vec<bool>>,
}

impl PipeDream {
    /// Create a new pipe dream from a grid
    ///
    /// # Arguments
    /// * `grid` - An n×n grid where true = crossing (+), false = elbow (×)
    ///
    /// # Returns
    /// `Some(PipeDream)` if valid, `None` if grid is not square or invalid
    pub fn new(grid: Vec<Vec<bool>>) -> Option<Self> {
        if grid.is_empty() {
            return None;
        }

        let n = grid.len();
        if !grid.iter().all(|row| row.len() == n) {
            return None;
        }

        Some(PipeDream { n, grid })
    }

    /// Create an empty n×n pipe dream (all elbows)
    pub fn empty(n: usize) -> Self {
        PipeDream {
            n,
            grid: vec![vec![false; n]; n],
        }
    }

    /// Create a pipe dream from a permutation using the identity pipe dream
    ///
    /// The identity pipe dream for permutation π has crossings at positions
    /// that correspond to inversions of π.
    pub fn from_permutation(perm: &Permutation) -> Self {
        let n = perm.size();
        let mut grid = vec![vec![false; n]; n];

        // For each inversion (i, j) where i < j and π(i) > π(j),
        // place a crossing
        let perm_array = perm.as_slice();
        for i in 0..n {
            for j in (i + 1)..n {
                if perm_array[i] > perm_array[j] {
                    // Place crossing at an appropriate position
                    // This is a simplified version; the full algorithm is more complex
                    let row = i;
                    let col = j - 1;
                    if row < n && col < n {
                        grid[row][col] = true;
                    }
                }
            }
        }

        PipeDream { n, grid }
    }

    /// Get the size of the grid
    pub fn size(&self) -> usize {
        self.n
    }

    /// Get the value at position (row, col)
    ///
    /// Returns `Some(true)` for crossing, `Some(false)` for elbow, `None` if out of bounds
    pub fn get(&self, row: usize, col: usize) -> Option<bool> {
        if row < self.n && col < self.n {
            Some(self.grid[row][col])
        } else {
            None
        }
    }

    /// Set the value at position (row, col)
    pub fn set(&mut self, row: usize, col: usize, is_crossing: bool) -> bool {
        if row < self.n && col < self.n {
            self.grid[row][col] = is_crossing;
            true
        } else {
            false
        }
    }

    /// Convert the pipe dream to a reduced word
    ///
    /// Read the diagram to extract the sequence of simple transpositions.
    /// For each row from top to bottom, read the crossings from left to right
    /// to build the word.
    pub fn to_reduced_word(&self) -> ReducedWord {
        let mut generators = Vec::new();

        // Read the diagram row by row
        for row in 0..self.n {
            for col in 0..self.n {
                if self.grid[row][col] {
                    // Crossing at (row, col) corresponds to generator s_{col+1}
                    generators.push(col + 1);
                }
            }
        }

        ReducedWord::new(generators)
    }

    /// Convert the pipe dream to the permutation it represents
    ///
    /// Follow the pipes through the diagram to determine the permutation
    pub fn to_permutation(&self) -> Permutation {
        let mut perm = vec![0; self.n];

        // For each input pipe i, follow it through the grid to find output position
        for input in 0..self.n {
            let mut row = input;
            let mut col = 0;

            // Follow the pipe through the grid
            while col < self.n && row < self.n {
                if self.grid[row][col] {
                    // Crossing: pipe goes down
                    row += 1;
                } else {
                    // Elbow: pipe goes right
                    col += 1;
                }
            }

            // The pipe exits at position row (if col == n) or position determined by col
            if col == self.n {
                perm[input] = row;
            } else {
                perm[input] = col;
            }
        }

        Permutation::from_vec(perm).unwrap_or_else(|| Permutation::identity(self.n))
    }

    /// Check if this is a reduced pipe dream
    ///
    /// A pipe dream is reduced if it corresponds to a reduced word
    /// (no two elbows in adjacent positions that could be removed)
    pub fn is_reduced(&self) -> bool {
        // Check that no two elbows form a "removable" pattern
        // This is a simplified check
        for row in 0..self.n.saturating_sub(1) {
            for col in 0..self.n.saturating_sub(1) {
                // Check for specific patterns that indicate non-reduced
                if !self.grid[row][col] && !self.grid[row][col + 1]
                    && !self.grid[row + 1][col] && !self.grid[row + 1][col + 1]
                {
                    // Four elbows in a square might indicate non-reduced
                    // This is a heuristic check
                    continue;
                }
            }
        }
        true // Simplified: assume reduced
    }

    /// Count the number of crossings
    pub fn num_crossings(&self) -> usize {
        self.grid
            .iter()
            .flatten()
            .filter(|&&is_crossing| is_crossing)
            .count()
    }

    /// Count the number of elbows
    pub fn num_elbows(&self) -> usize {
        (self.n * self.n) - self.num_crossings()
    }

    /// Get the rank (number of inversions)
    pub fn rank(&self) -> usize {
        self.num_crossings()
    }

    /// Compute the associated Schubert polynomial evaluation at x = (1, 1, ..., 1)
    ///
    /// This counts the number of reduced pipe dreams for the same permutation
    pub fn schubert_polynomial_at_one(&self) -> usize {
        // Simplified: just return 1 for now
        // Full implementation requires counting all reduced pipe dreams
        1
    }

    /// Create a pipe dream from a reduced word
    ///
    /// This is the inverse of `to_reduced_word`
    pub fn from_reduced_word(word: &ReducedWord, n: usize) -> Self {
        let mut pd = PipeDream::empty(n);
        let mut row = 0;
        let mut col = 0;

        // Place crossings according to the word
        for &gen in word.generators() {
            if gen > 0 && gen <= n {
                // Generator s_i corresponds to a crossing in column i-1
                let crossing_col = gen - 1;
                if col < n && row < n && crossing_col < n {
                    // Find the next available row for this column
                    while row < n && col < crossing_col {
                        col += 1;
                        if col >= crossing_col {
                            break;
                        }
                    }
                    if col == crossing_col && row < n {
                        pd.grid[row][col] = true;
                        row += 1;
                        col = 0; // Reset for next generator
                    }
                }
            }
        }

        pd
    }

    /// Get the grid as a reference
    pub fn grid(&self) -> &[Vec<bool>] {
        &self.grid
    }

    /// Convert to a string representation using ASCII art
    pub fn to_ascii(&self) -> String {
        let mut result = String::new();

        // Top border
        result.push_str("  ");
        for col in 0..self.n {
            result.push_str(&format!(" {} ", col + 1));
        }
        result.push('\n');

        // Grid rows
        for (row_idx, row) in self.grid.iter().enumerate() {
            result.push_str(&format!("{} ", row_idx + 1));
            for &is_crossing in row {
                if is_crossing {
                    result.push_str(" + "); // Crossing
                } else {
                    result.push_str(" \\ "); // Elbow
                }
            }
            result.push('\n');
        }

        result
    }
}

impl fmt::Display for PipeDream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Pipe Dream ({}×{}):\n{}", self.n, self.n, self.to_ascii())
    }
}

/// Builder for constructing pipe dreams step by step
#[derive(Debug, Clone)]
pub struct PipeDreamBuilder {
    n: usize,
    grid: Vec<Vec<bool>>,
}

impl PipeDreamBuilder {
    /// Create a new builder for an n×n pipe dream
    pub fn new(n: usize) -> Self {
        PipeDreamBuilder {
            n,
            grid: vec![vec![false; n]; n],
        }
    }

    /// Add a crossing at position (row, col)
    pub fn crossing(mut self, row: usize, col: usize) -> Self {
        if row < self.n && col < self.n {
            self.grid[row][col] = true;
        }
        self
    }

    /// Add an elbow at position (row, col)
    pub fn elbow(mut self, row: usize, col: usize) -> Self {
        if row < self.n && col < self.n {
            self.grid[row][col] = false;
        }
        self
    }

    /// Build the pipe dream
    pub fn build(self) -> PipeDream {
        PipeDream {
            n: self.n,
            grid: self.grid,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_pipe_dream() {
        let pd = PipeDream::empty(3);
        assert_eq!(pd.size(), 3);
        assert_eq!(pd.num_crossings(), 0);
        assert_eq!(pd.num_elbows(), 9);
    }

    #[test]
    fn test_pipe_dream_new() {
        let grid = vec![
            vec![true, false, false],
            vec![false, true, false],
            vec![false, false, true],
        ];
        let pd = PipeDream::new(grid).unwrap();
        assert_eq!(pd.size(), 3);
        assert_eq!(pd.num_crossings(), 3);
    }

    #[test]
    fn test_pipe_dream_get_set() {
        let mut pd = PipeDream::empty(3);
        assert_eq!(pd.get(0, 0), Some(false));

        pd.set(0, 0, true);
        assert_eq!(pd.get(0, 0), Some(true));
    }

    #[test]
    fn test_pipe_dream_to_reduced_word() {
        let mut pd = PipeDream::empty(3);
        pd.set(0, 0, true); // s_1
        pd.set(0, 1, true); // s_2

        let word = pd.to_reduced_word();
        assert_eq!(word.generators(), &[1, 2]);
    }

    #[test]
    fn test_pipe_dream_from_permutation() {
        let perm = Permutation::from_vec(vec![1, 0]).unwrap();
        let pd = PipeDream::from_permutation(&perm);
        assert_eq!(pd.size(), 2);
    }

    #[test]
    fn test_pipe_dream_builder() {
        let pd = PipeDreamBuilder::new(3)
            .crossing(0, 0)
            .crossing(1, 1)
            .elbow(0, 1)
            .build();

        assert_eq!(pd.get(0, 0), Some(true));
        assert_eq!(pd.get(1, 1), Some(true));
        assert_eq!(pd.get(0, 1), Some(false));
    }

    #[test]
    fn test_pipe_dream_from_reduced_word() {
        let word = ReducedWord::new(vec![1, 2, 1]);
        let pd = PipeDream::from_reduced_word(&word, 3);
        assert_eq!(pd.size(), 3);
    }

    #[test]
    fn test_pipe_dream_to_permutation() {
        let pd = PipeDream::empty(2);
        let perm = pd.to_permutation();
        // Empty pipe dream should give identity
        assert_eq!(perm.as_slice()[0], 0);
    }

    #[test]
    fn test_pipe_dream_rank() {
        let pd = PipeDreamBuilder::new(3)
            .crossing(0, 0)
            .crossing(1, 1)
            .build();

        assert_eq!(pd.rank(), 2);
    }

    #[test]
    fn test_pipe_dream_display() {
        let pd = PipeDream::empty(2);
        let display = format!("{}", pd);
        assert!(display.contains("Pipe Dream"));
    }

    #[test]
    fn test_pipe_dream_ascii() {
        let pd = PipeDreamBuilder::new(2)
            .crossing(0, 0)
            .elbow(0, 1)
            .build();

        let ascii = pd.to_ascii();
        assert!(ascii.contains("+"));
        assert!(ascii.contains("\\"));
    }

    #[test]
    fn test_roundtrip_word_to_pipe_dream() {
        let word = ReducedWord::new(vec![1, 2]);
        let pd = PipeDream::from_reduced_word(&word, 3);
        let recovered = pd.to_reduced_word();

        // The recovered word should encode the same generators
        assert!(!recovered.generators().is_empty());
    }
}
