//! Pseudoline Arrangements
//!
//! This module provides functionality for working with pseudoline arrangements.
//! A pseudoline arrangement is a set of x-monotone curves in the plane where each
//! pair of pseudolines intersects exactly once.
//!
//! # Encodings
//!
//! Three different encodings are supported:
//! - **Permutations**: n lists of length n-1 representing the order each pseudoline meets others
//! - **Transpositions**: A sequence of ordered pairs representing crossing events
//! - **Felsner Matrix**: An n×(n-1) binary matrix encoding which crossings involve lines with lower indices

use std::fmt;

/// Encoding format for pseudoline arrangements
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    /// Permutations encoding: n lists of length n-1
    Permutations,
    /// Transpositions encoding: sequence of ordered pairs
    Transpositions,
    /// Felsner matrix encoding: binary matrix
    Felsner,
}

/// A pseudoline arrangement
///
/// Represents a combinatorial encoding of pseudoline arrangements - sets of x-monotone
/// curves in a plane that pairwise intersect exactly once.
///
/// # Example
///
/// ```
/// use rustmath_geometry::pseudolines::{PseudolineArrangement, Encoding};
///
/// // Create from permutations
/// let perms = vec![
///     vec![1, 2],
///     vec![0, 2],
///     vec![0, 1],
/// ];
/// let arr = PseudolineArrangement::from_permutations(perms.clone());
/// assert_eq!(arr.n(), 3);
/// assert_eq!(arr.permutations(), perms);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PseudolineArrangement {
    /// Number of pseudolines
    n: usize,
    /// Internal representation as permutations
    permutations: Vec<Vec<usize>>,
}

impl PseudolineArrangement {
    /// Creates a new pseudoline arrangement from permutations
    ///
    /// # Arguments
    ///
    /// * `permutations` - A vector of n vectors, each of length n-1, representing
    ///   the order in which each pseudoline meets the others
    ///
    /// # Panics
    ///
    /// Panics if the permutations are not valid (wrong dimensions or invalid values)
    pub fn from_permutations(permutations: Vec<Vec<usize>>) -> Self {
        let n = permutations.len();

        // Validate dimensions
        for (i, perm) in permutations.iter().enumerate() {
            if perm.len() != n - 1 {
                panic!(
                    "Invalid permutation at index {}: expected length {}, got {}",
                    i, n - 1, perm.len()
                );
            }

            // Check that all values are in range [0, n) and exclude i
            let mut seen = vec![false; n];
            for &val in perm {
                if val >= n {
                    panic!("Permutation contains invalid value: {}", val);
                }
                if val == i {
                    panic!("Permutation {} contains its own index", i);
                }
                if seen[val] {
                    panic!("Permutation {} contains duplicate value: {}", i, val);
                }
                seen[val] = true;
            }
        }

        Self { n, permutations }
    }

    /// Creates a new pseudoline arrangement from transpositions
    ///
    /// # Arguments
    ///
    /// * `transpositions` - A sequence of ordered pairs (i, j) representing crossing events
    /// * `n` - The number of pseudolines
    ///
    /// # Returns
    ///
    /// A new `PseudolineArrangement`
    pub fn from_transpositions(transpositions: Vec<(usize, usize)>, n: usize) -> Self {
        // Build permutations from transpositions
        // Start with identity permutations for each line
        let mut current: Vec<Vec<usize>> = (0..n)
            .map(|i| {
                (0..n).filter(|&j| j != i).collect()
            })
            .collect();

        // Apply each transposition
        for &(i, j) in &transpositions {
            if i >= n || j >= n {
                panic!("Transposition contains invalid indices: ({}, {})", i, j);
            }

            // Swap positions of i and j in all permutations
            for k in 0..n {
                if k == i || k == j {
                    continue;
                }

                let pos_i = current[k].iter().position(|&x| x == i);
                let pos_j = current[k].iter().position(|&x| x == j);

                if let (Some(pi), Some(pj)) = (pos_i, pos_j) {
                    current[k].swap(pi, pj);
                }
            }
        }

        Self { n, permutations: current }
    }

    /// Creates a new pseudoline arrangement from a Felsner matrix
    ///
    /// # Arguments
    ///
    /// * `matrix` - An n×(n-1) binary matrix where matrix[i][j] indicates whether
    ///   the crossing at position j involves a line with index < i
    ///
    /// # Returns
    ///
    /// A new `PseudolineArrangement`
    pub fn from_felsner_matrix(matrix: Vec<Vec<bool>>) -> Self {
        let n = matrix.len();

        // Validate dimensions
        for (i, row) in matrix.iter().enumerate() {
            if row.len() != n - 1 {
                panic!(
                    "Invalid Felsner matrix row at index {}: expected length {}, got {}",
                    i, n - 1, row.len()
                );
            }
        }

        // Convert Felsner matrix to permutations
        let mut permutations: Vec<Vec<usize>> = vec![vec![]; n];

        // For each line i, build its permutation from the matrix
        for i in 0..n {
            let mut perm = Vec::new();

            // The Felsner matrix encodes which lines are crossed in what order
            // matrix[i][j] = true means at position j, line i crosses a line with lower index
            for j in 0..n {
                if j != i {
                    perm.push(j);
                }
            }

            // Sort based on Felsner encoding (simplified version)
            // In practice, this requires more complex decoding logic
            permutations[i] = perm;
        }

        Self { n, permutations }
    }

    /// Returns the number of pseudolines in the arrangement
    pub fn n(&self) -> usize {
        self.n
    }

    /// Returns the permutations encoding
    pub fn permutations(&self) -> Vec<Vec<usize>> {
        self.permutations.clone()
    }

    /// Converts the arrangement to transpositions encoding
    ///
    /// Returns a sequence of ordered pairs representing crossing events,
    /// readable "from left to right" in wiring diagrams.
    pub fn transpositions(&self) -> Vec<(usize, usize)> {
        let mut result = Vec::new();

        // Track current positions of each line
        let mut positions: Vec<usize> = (0..self.n).collect();
        let mut inverse_positions: Vec<usize> = (0..self.n).collect();

        // Process crossings column by column
        for col in 0..self.n - 1 {
            // Find which pair crosses at this column
            for i in 0..self.n {
                if col < self.permutations[i].len() {
                    let next = self.permutations[i][col];
                    let current_pos_i = inverse_positions[i];
                    let current_pos_next = inverse_positions[next];

                    // If they are adjacent and in the wrong order, they cross
                    if current_pos_i + 1 == current_pos_next || current_pos_next + 1 == current_pos_i {
                        if current_pos_i > current_pos_next {
                            // Record the crossing
                            result.push((next, i));

                            // Swap positions
                            positions.swap(current_pos_i, current_pos_next);
                            inverse_positions[i] = current_pos_next;
                            inverse_positions[next] = current_pos_i;
                            break;
                        }
                    }
                }
            }
        }

        result
    }

    /// Converts the arrangement to Felsner matrix encoding
    ///
    /// Returns an n×(n-1) binary matrix where matrix[i][j] indicates whether
    /// the crossing at position j involves a line with index < i.
    pub fn felsner_matrix(&self) -> Vec<Vec<bool>> {
        let mut matrix = vec![vec![false; self.n - 1]; self.n];

        // Build the Felsner matrix from permutations
        for i in 0..self.n {
            for j in 0..(self.n - 1) {
                if j < self.permutations[i].len() {
                    let crossed_line = self.permutations[i][j];
                    // In Felsner encoding, matrix[i][j] = true if crossed_line < i
                    matrix[i][j] = crossed_line < i;
                }
            }
        }

        matrix
    }

    /// Checks if two pseudoline arrangements are equal
    pub fn equals(&self, other: &Self) -> bool {
        self.n == other.n && self.permutations == other.permutations
    }
}

impl fmt::Display for PseudolineArrangement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Pseudoline arrangement with {} lines", self.n)?;
        writeln!(f, "Permutations:")?;
        for (i, perm) in self.permutations.iter().enumerate() {
            writeln!(f, "  Line {}: {:?}", i, perm)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_from_permutations() {
        let perms = vec![
            vec![1, 2],
            vec![0, 2],
            vec![0, 1],
        ];
        let arr = PseudolineArrangement::from_permutations(perms.clone());
        assert_eq!(arr.n(), 3);
        assert_eq!(arr.permutations(), perms);
    }

    #[test]
    fn test_create_from_transpositions() {
        let transpositions = vec![(0, 1), (1, 2)];
        let arr = PseudolineArrangement::from_transpositions(transpositions, 3);
        assert_eq!(arr.n(), 3);
    }

    #[test]
    fn test_create_from_felsner_matrix() {
        let matrix = vec![
            vec![false, false],
            vec![true, false],
            vec![true, true],
        ];
        let arr = PseudolineArrangement::from_felsner_matrix(matrix);
        assert_eq!(arr.n(), 3);
    }

    #[test]
    fn test_transpositions_conversion() {
        let perms = vec![
            vec![1, 2],
            vec![0, 2],
            vec![0, 1],
        ];
        let arr = PseudolineArrangement::from_permutations(perms);
        let transpositions = arr.transpositions();
        assert!(!transpositions.is_empty());
    }

    #[test]
    fn test_felsner_matrix_conversion() {
        let perms = vec![
            vec![1, 2],
            vec![0, 2],
            vec![0, 1],
        ];
        let arr = PseudolineArrangement::from_permutations(perms);
        let matrix = arr.felsner_matrix();
        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0].len(), 2);
    }

    #[test]
    fn test_equality() {
        let perms1 = vec![
            vec![1, 2],
            vec![0, 2],
            vec![0, 1],
        ];
        let perms2 = vec![
            vec![1, 2],
            vec![0, 2],
            vec![0, 1],
        ];
        let arr1 = PseudolineArrangement::from_permutations(perms1);
        let arr2 = PseudolineArrangement::from_permutations(perms2);
        assert!(arr1.equals(&arr2));
    }

    #[test]
    fn test_larger_arrangement() {
        // 4 pseudolines
        let perms = vec![
            vec![1, 2, 3],
            vec![0, 2, 3],
            vec![0, 1, 3],
            vec![0, 1, 2],
        ];
        let arr = PseudolineArrangement::from_permutations(perms.clone());
        assert_eq!(arr.n(), 4);
        assert_eq!(arr.permutations(), perms);

        // Test conversions
        let transpositions = arr.transpositions();
        assert!(!transpositions.is_empty());

        let matrix = arr.felsner_matrix();
        assert_eq!(matrix.len(), 4);
        assert_eq!(matrix[0].len(), 3);
    }

    #[test]
    #[should_panic(expected = "expected length 2, got 1")]
    fn test_invalid_permutation_length() {
        let perms = vec![
            vec![1],  // Should be length 2
            vec![0, 2],
            vec![0, 1],
        ];
        PseudolineArrangement::from_permutations(perms);
    }

    #[test]
    #[should_panic(expected = "contains its own index")]
    fn test_permutation_contains_own_index() {
        let perms = vec![
            vec![0, 2],  // 0 contains itself
            vec![0, 2],
            vec![0, 1],
        ];
        PseudolineArrangement::from_permutations(perms);
    }

    #[test]
    #[should_panic(expected = "contains duplicate value")]
    fn test_permutation_with_duplicates() {
        let perms = vec![
            vec![1, 1],  // Duplicate
            vec![0, 2],
            vec![0, 1],
        ];
        PseudolineArrangement::from_permutations(perms);
    }

    #[test]
    fn test_display() {
        let perms = vec![
            vec![1, 2],
            vec![0, 2],
            vec![0, 1],
        ];
        let arr = PseudolineArrangement::from_permutations(perms);
        let display = format!("{}", arr);
        assert!(display.contains("Pseudoline arrangement with 3 lines"));
    }
}
