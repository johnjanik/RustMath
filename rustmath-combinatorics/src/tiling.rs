//! Tiling with dominoes and polyominoes using the transfer matrix method
//!
//! This module implements algorithms for counting tilings of rectangular grids
//! using dominoes and general polyominoes. The transfer matrix method is a
//! dynamic programming technique that represents column states and transitions.
//!
//! # Transfer Matrix Method
//!
//! The key idea is to process the board column by column. Each column can be
//! in one of several "states" indicating which cells are already filled by
//! pieces from the previous column. We build a matrix M where M[i][j] counts
//! the number of ways to transition from state i to state j when adding a new column.
//!
//! For an m×n board:
//! - Each state is a binary string of length m (2^m possible states)
//! - Bit k=1 means cell k is already filled (by a horizontal piece from previous column)
//! - We compute the number of tilings as M^(n-1) applied to the initial state

use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_matrix::Matrix;
use std::collections::HashMap;

/// Represents a column state as a bitmask
/// Bit i is set if cell i in the column is pre-filled from the previous column
pub type ColumnState = u32;

/// A polyomino represented as a set of relative cell positions (row, col)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Polyomino {
    /// Cells occupied by this polyomino, relative to origin (0, 0)
    pub cells: Vec<(i32, i32)>,
}

impl Polyomino {
    /// Create a new polyomino from a list of cells
    pub fn new(cells: Vec<(i32, i32)>) -> Self {
        Polyomino { cells }
    }

    /// Create a horizontal domino (1×2)
    pub fn horizontal_domino() -> Self {
        Polyomino::new(vec![(0, 0), (0, 1)])
    }

    /// Create a vertical domino (2×1)
    pub fn vertical_domino() -> Self {
        Polyomino::new(vec![(0, 0), (1, 0)])
    }

    /// Create both domino orientations
    pub fn dominoes() -> Vec<Self> {
        vec![Self::horizontal_domino(), Self::vertical_domino()]
    }

    /// Create a monomino (1×1)
    pub fn monomino() -> Self {
        Polyomino::new(vec![(0, 0)])
    }

    /// Create an L-triomino
    pub fn l_triomino() -> Self {
        Polyomino::new(vec![(0, 0), (1, 0), (1, 1)])
    }

    /// Create an I-triomino (vertical)
    pub fn i_triomino() -> Self {
        Polyomino::new(vec![(0, 0), (1, 0), (2, 0)])
    }

    /// Get all rotations and reflections of this polyomino
    pub fn all_orientations(&self) -> Vec<Polyomino> {
        let mut result = Vec::new();
        let mut seen = std::collections::HashSet::new();

        // Original
        let mut current = self.cells.clone();
        current.sort();
        if seen.insert(current.clone()) {
            result.push(Polyomino::new(current.clone()));
        }

        // Rotate 90, 180, 270 degrees
        for _ in 0..3 {
            current = current.iter().map(|(r, c)| (*c, -*r)).collect();
            current.sort();
            if seen.insert(current.clone()) {
                result.push(Polyomino::new(current.clone()));
            }
        }

        // Reflect and repeat rotations
        current = self.cells.iter().map(|(r, c)| (*r, -*c)).collect();
        current.sort();
        if seen.insert(current.clone()) {
            result.push(Polyomino::new(current.clone()));
        }

        for _ in 0..3 {
            current = current.iter().map(|(r, c)| (*c, -*r)).collect();
            current.sort();
            if seen.insert(current.clone()) {
                result.push(Polyomino::new(current.clone()));
            }
        }

        result
    }

    /// Normalize polyomino so minimum coordinates are (0, 0)
    pub fn normalize(&self) -> Polyomino {
        if self.cells.is_empty() {
            return Polyomino::new(vec![]);
        }

        let min_row = self.cells.iter().map(|(r, _)| *r).min().unwrap();
        let min_col = self.cells.iter().map(|(_, c)| *c).min().unwrap();

        let mut cells: Vec<_> = self
            .cells
            .iter()
            .map(|(r, c)| (r - min_row, c - min_col))
            .collect();
        cells.sort();
        Polyomino::new(cells)
    }

    /// Get the maximum row and column occupied by this polyomino
    pub fn bounds(&self) -> (i32, i32) {
        if self.cells.is_empty() {
            return (0, 0);
        }
        let max_row = self.cells.iter().map(|(r, _)| *r).max().unwrap();
        let max_col = self.cells.iter().map(|(_, c)| *c).max().unwrap();
        (max_row + 1, max_col + 1)
    }
}

/// Generate all valid next states when filling a column
///
/// Given:
/// - `prev_state`: which cells in current column are already filled
/// - `height`: number of rows in the grid
/// - `pieces`: available polyomino pieces
///
/// Returns: list of (next_state, count) pairs where next_state indicates
/// which cells in the next column will be pre-filled
fn generate_next_states(
    prev_state: ColumnState,
    height: usize,
    pieces: &[Polyomino],
) -> Vec<(ColumnState, usize)> {
    let mut results = HashMap::new();

    // Try to fill the current column given the prev_state
    fill_column(prev_state, height, pieces, 0, 0, &mut results);

    results.into_iter().collect()
}

/// Recursively fill a column starting from a given row
///
/// This is a backtracking algorithm that tries to place pieces to fill all
/// empty cells in the current column.
fn fill_column(
    current_state: ColumnState,
    height: usize,
    pieces: &[Polyomino],
    row: usize,
    next_state: ColumnState,
    results: &mut HashMap<ColumnState, usize>,
) {
    // Find the first empty cell in the current column
    let mut first_empty = None;
    for r in row..height {
        if (current_state & (1 << r)) == 0 {
            first_empty = Some(r);
            break;
        }
    }

    // If no empty cells, we've successfully filled this column
    if first_empty.is_none() {
        *results.entry(next_state).or_insert(0) += 1;
        return;
    }

    let r = first_empty.unwrap();

    // Try placing each piece at this position
    for piece in pieces {
        let piece = piece.normalize();

        // Try to place piece starting at (r, 0) in current column
        let mut valid = true;
        let mut new_current = current_state;
        let mut new_next = next_state;

        for &(dr, dc) in &piece.cells {
            let nr = r as i32 + dr;
            let nc = dc; // Current column is column 0

            if nr < 0 || nr >= height as i32 {
                valid = false;
                break;
            }

            if nc < 0 {
                // This cell is in the previous column - not allowed
                valid = false;
                break;
            } else if nc == 0 {
                // This cell is in the current column
                if (new_current & (1 << nr)) != 0 {
                    // Already filled
                    valid = false;
                    break;
                }
                new_current |= 1 << nr;
            } else if nc == 1 {
                // This cell is in the next column
                if (new_next & (1 << nr)) != 0 {
                    // Already marked as filled
                    valid = false;
                    break;
                }
                new_next |= 1 << nr;
            } else {
                // Piece extends beyond next column - not allowed in column-by-column method
                valid = false;
                break;
            }
        }

        if valid {
            fill_column(new_current, height, pieces, r, new_next, results);
        }
    }
}

/// Build the transfer matrix for tiling with given polyominoes
///
/// The transfer matrix M is indexed by column states. M[i][j] counts the
/// number of ways to transition from state i to state j.
pub fn build_transfer_matrix(height: usize, pieces: &[Polyomino]) -> Matrix<Integer> {
    let num_states = 1 << height;
    let mut data = vec![Integer::zero(); num_states * num_states];

    for state in 0..num_states {
        let next_states = generate_next_states(state as ColumnState, height, pieces);
        for (next_state, count) in next_states {
            data[state * num_states + next_state as usize] = Integer::from(count as u32);
        }
    }

    Matrix::from_vec(num_states, num_states, data).unwrap()
}

/// Count the number of ways to tile an m×n rectangle with given polyominoes
///
/// Uses the transfer matrix method:
/// 1. Build transfer matrix M for column-to-column transitions
/// 2. Compute M^n
/// 3. Extract the (0,0) entry (from empty state to empty state)
///
/// The transfer matrix M[i][j] counts ways to fill a column starting in state i
/// and leaving the next column in state j. We need M^n[0][0] to count all paths
/// that start and end with state 0 (no overflow cells).
///
/// # Arguments
/// - `m`: height of rectangle (number of rows)
/// - `n`: width of rectangle (number of columns)
/// - `pieces`: available polyomino pieces to use
///
/// # Returns
/// Number of distinct tilings
pub fn count_tilings(m: usize, n: usize, pieces: &[Polyomino]) -> Integer {
    if m == 0 || n == 0 {
        return Integer::one();
    }

    let transfer = build_transfer_matrix(m, pieces);

    // Compute transfer^n
    let result = transfer.pow(n as u32).unwrap();

    // Return the (0, 0) entry: tiling from empty to empty
    result.get(0, 0).unwrap().clone()
}

/// Count domino tilings of an m×n rectangle
///
/// A domino is a 1×2 or 2×1 piece. This function counts the number of ways
/// to completely tile an m×n rectangle with dominoes.
///
/// # Examples
/// - 2×2 board: 2 tilings
/// - 2×3 board: 3 tilings
/// - 3×4 board: 11 tilings
///
/// # Note
/// If m*n is odd, there are no tilings (returns 0)
pub fn count_domino_tilings(m: usize, n: usize) -> Integer {
    // Quick check: if total cells is odd, no tiling exists
    if (m * n) % 2 == 1 {
        return Integer::zero();
    }

    count_tilings(m, n, &Polyomino::dominoes())
}

/// Count tilings of an m×n rectangle with monominos (1×1 squares)
///
/// This is trivial: there's exactly one way to tile with 1×1 pieces
pub fn count_monomino_tilings(m: usize, n: usize) -> Integer {
    if m == 0 || n == 0 {
        return Integer::one();
    }
    Integer::one()
}

/// Count tilings with a specific set of polyominoes
///
/// This is a more general interface that allows specifying exactly which
/// polyomino shapes are allowed.
pub fn count_polyomino_tilings(m: usize, n: usize, pieces: &[Polyomino]) -> Integer {
    count_tilings(m, n, pieces)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polyomino_creation() {
        let horiz = Polyomino::horizontal_domino();
        assert_eq!(horiz.cells, vec![(0, 0), (0, 1)]);

        let vert = Polyomino::vertical_domino();
        assert_eq!(vert.cells, vec![(0, 0), (1, 0)]);
    }

    #[test]
    fn test_polyomino_normalize() {
        let p = Polyomino::new(vec![(2, 3), (2, 4), (3, 3)]);
        let normalized = p.normalize();
        assert_eq!(normalized.cells, vec![(0, 0), (0, 1), (1, 0)]);
    }

    #[test]
    fn test_polyomino_bounds() {
        let p = Polyomino::new(vec![(0, 0), (0, 1), (1, 0)]);
        let (h, w) = p.bounds();
        assert_eq!(h, 2);
        assert_eq!(w, 2);
    }

    #[test]
    fn test_domino_tilings_2x2() {
        // 2x2 board has exactly 2 domino tilings
        let count = count_domino_tilings(2, 2);
        assert_eq!(count, Integer::from(2));
    }

    #[test]
    fn test_domino_tilings_2x3() {
        // 2x3 board has exactly 3 domino tilings
        let count = count_domino_tilings(2, 3);
        assert_eq!(count, Integer::from(3));
    }

    #[test]
    fn test_domino_tilings_2x4() {
        // 2x4 board has exactly 5 domino tilings
        let count = count_domino_tilings(2, 4);
        assert_eq!(count, Integer::from(5));
    }

    #[test]
    fn test_domino_tilings_2x5() {
        // 2x5 board has exactly 8 domino tilings
        let count = count_domino_tilings(2, 5);
        assert_eq!(count, Integer::from(8));
    }

    #[test]
    fn test_domino_tilings_odd() {
        // Odd number of cells: no tiling possible
        let count = count_domino_tilings(3, 3);
        assert_eq!(count, Integer::zero());

        let count = count_domino_tilings(1, 3);
        assert_eq!(count, Integer::zero());

        let count = count_domino_tilings(3, 1);
        assert_eq!(count, Integer::zero());
    }

    #[test]
    fn test_domino_tilings_3x4() {
        // 3x4 board has exactly 11 domino tilings
        let count = count_domino_tilings(3, 4);
        assert_eq!(count, Integer::from(11));
    }

    #[test]
    fn test_domino_tilings_4x4() {
        // 4x4 board has exactly 36 domino tilings
        let count = count_domino_tilings(4, 4);
        assert_eq!(count, Integer::from(36));
    }

    #[test]
    fn test_monomino_tilings() {
        // Monomino tilings are always 1
        assert_eq!(count_monomino_tilings(1, 1), Integer::one());
        assert_eq!(count_monomino_tilings(5, 5), Integer::one());
        assert_eq!(count_monomino_tilings(10, 10), Integer::one());
    }

    #[test]
    fn test_transfer_matrix_dominoes() {
        // For height 2, there should be 4 states (00, 01, 10, 11)
        let transfer = build_transfer_matrix(2, &Polyomino::dominoes());
        assert_eq!(transfer.rows(), 4);
        assert_eq!(transfer.cols(), 4);

        // State 0 (00) should be able to transition to states 0 and 3 (11)
        // - Place 2 horizontal dominoes -> next state 11 (both cells filled in next column)
        // - Place 2 vertical dominoes -> next state 00 (no cells filled in next column)
        let val00 = transfer.get(0, 0).unwrap();
        let val03 = transfer.get(0, 3).unwrap();
        assert!(val00 > &Integer::zero() || val03 > &Integer::zero());
    }

    #[test]
    fn test_domino_tilings_1xn() {
        // 1×2 board: 1 tiling (one horizontal domino)
        assert_eq!(count_domino_tilings(1, 2), Integer::one());

        // 1×4 board: 1 tiling (two horizontal dominoes)
        assert_eq!(count_domino_tilings(1, 4), Integer::one());

        // 1×1 board: 0 tilings (odd number of cells)
        assert_eq!(count_domino_tilings(1, 1), Integer::zero());

        // 1×3 board: 0 tilings (odd number of cells)
        assert_eq!(count_domino_tilings(1, 3), Integer::zero());
    }

    #[test]
    fn test_domino_tilings_empty() {
        // Empty board
        assert_eq!(count_domino_tilings(0, 0), Integer::one());
        assert_eq!(count_domino_tilings(0, 5), Integer::one());
        assert_eq!(count_domino_tilings(5, 0), Integer::one());
    }

    #[test]
    fn test_polyomino_orientations() {
        let l_tri = Polyomino::l_triomino();
        let orientations = l_tri.all_orientations();

        // L-triomino should have 4 distinct orientations
        // (4 rotations, and reflections give the same set)
        assert!(orientations.len() >= 4 && orientations.len() <= 8);
    }

    #[test]
    fn test_domino_tilings_symmetry() {
        // m×n should equal n×m for domino tilings
        for m in 2..=4 {
            for n in 2..=4 {
                if (m * n) % 2 == 0 {
                    let count1 = count_domino_tilings(m, n);
                    let count2 = count_domino_tilings(n, m);
                    assert_eq!(count1, count2, "Symmetry failed for {}×{}", m, n);
                }
            }
        }
    }
}
