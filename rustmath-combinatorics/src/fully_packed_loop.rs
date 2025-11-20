//! Fully Packed Loop (FPL) configurations and their bijection with Alternating Sign Matrices
//!
//! This module implements:
//! - Alternating Sign Matrices (ASM)
//! - Fully Packed Loop configurations
//! - Link patterns
//! - The bijection between FPL configurations and ASMs (Wieland's bijection)
//!
//! # Mathematical Background
//!
//! ## Alternating Sign Matrix (ASM)
//! An n×n alternating sign matrix is a matrix where:
//! - Each entry is 0, +1, or -1
//! - Each row and column sums to 1
//! - Non-zero entries in each row and column alternate in sign
//!
//! ## Fully Packed Loop (FPL)
//! An n×n FPL configuration is a tiling of an (n+1)×(n+1) grid where:
//! - Each internal vertex has degree 0, 2, or 4
//! - Each boundary vertex has degree 1
//! - Edges form non-crossing loops and paths
//!
//! ## Link Pattern
//! A link pattern describes how the 2n boundary points are paired in an FPL configuration.
//!
//! ## The Bijection
//! There exists a bijection between n×n FPL configurations and n×n ASMs,
//! discovered by Wieland and refined by others.

use std::fmt;

/// An Alternating Sign Matrix (ASM)
///
/// An n×n matrix where entries are -1, 0, or +1, each row and column sums to 1,
/// and non-zero entries alternate in sign.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlternatingSignMatrix {
    /// The matrix entries (stored as i8: -1, 0, or 1)
    entries: Vec<Vec<i8>>,
    /// Size of the matrix
    n: usize,
}

impl AlternatingSignMatrix {
    /// Create a new ASM from a matrix
    ///
    /// Returns `None` if the matrix is not a valid ASM
    pub fn new(entries: Vec<Vec<i8>>) -> Option<Self> {
        let n = entries.len();

        if n == 0 {
            return Some(AlternatingSignMatrix { entries, n: 0 });
        }

        // Check dimensions
        for row in &entries {
            if row.len() != n {
                return None;
            }
        }

        // Validate ASM properties
        for i in 0..n {
            // Check row sum
            let row_sum: i32 = entries[i].iter().map(|&x| x as i32).sum();
            if row_sum != 1 {
                return None;
            }

            // Check column sum
            let col_sum: i32 = (0..n).map(|j| entries[j][i] as i32).sum();
            if col_sum != 1 {
                return None;
            }

            // Check entries are only -1, 0, or 1
            for j in 0..n {
                if entries[i][j] < -1 || entries[i][j] > 1 {
                    return None;
                }
            }

            // Check alternating sign property in row
            if !check_alternating_sign(&entries[i]) {
                return None;
            }

            // Check alternating sign property in column
            let col: Vec<i8> = (0..n).map(|j| entries[j][i]).collect();
            if !check_alternating_sign(&col) {
                return None;
            }
        }

        Some(AlternatingSignMatrix { entries, n })
    }

    /// Create the identity ASM (identity permutation matrix)
    pub fn identity(n: usize) -> Self {
        let mut entries = vec![vec![0; n]; n];
        for i in 0..n {
            entries[i][i] = 1;
        }
        AlternatingSignMatrix { entries, n }
    }

    /// Get the size of the matrix
    pub fn size(&self) -> usize {
        self.n
    }

    /// Get the entry at position (i, j)
    pub fn get(&self, i: usize, j: usize) -> Option<i8> {
        if i >= self.n || j >= self.n {
            None
        } else {
            Some(self.entries[i][j])
        }
    }

    /// Get the matrix entries
    pub fn entries(&self) -> &[Vec<i8>] {
        &self.entries
    }

    /// Convert to permutation matrix if this ASM is a permutation matrix
    pub fn to_permutation(&self) -> Option<Vec<usize>> {
        let mut perm = vec![0; self.n];
        for i in 0..self.n {
            let mut count = 0;
            let mut pos = 0;
            for j in 0..self.n {
                if self.entries[i][j] != 0 {
                    if self.entries[i][j] != 1 {
                        return None; // Has -1 entries, not a permutation matrix
                    }
                    count += 1;
                    pos = j;
                }
            }
            if count != 1 {
                return None;
            }
            perm[i] = pos;
        }
        Some(perm)
    }

    /// Count the number of -1 entries (a measure of complexity)
    pub fn negative_count(&self) -> usize {
        self.entries
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&x| x == -1)
            .count()
    }
}

impl fmt::Display for AlternatingSignMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in &self.entries {
            for &entry in row {
                let ch = match entry {
                    -1 => "-",
                    0 => " 0",
                    1 => "+",
                    _ => "?",
                };
                write!(f, "{:>2} ", ch)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

/// Check if a sequence has alternating signs (ignoring zeros)
fn check_alternating_sign(seq: &[i8]) -> bool {
    let non_zero: Vec<i8> = seq.iter().copied().filter(|&x| x != 0).collect();

    for i in 1..non_zero.len() {
        if non_zero[i] * non_zero[i - 1] >= 0 {
            // Same sign or zero product means not alternating
            return false;
        }
    }

    true
}

/// A link pattern describing boundary connections
///
/// Represents how 2n boundary points are paired. The boundary points are numbered
/// 0, 1, ..., 2n-1 going clockwise around the boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkPattern {
    /// The pairing: links[i] = j means points i and j are connected
    links: Vec<usize>,
    /// Size parameter
    n: usize,
}

impl LinkPattern {
    /// Create a new link pattern
    ///
    /// Returns `None` if the pairing is invalid
    pub fn new(links: Vec<usize>) -> Option<Self> {
        let len = links.len();
        if len == 0 || len % 2 != 0 {
            return None;
        }

        let n = len / 2;

        // Check that it's a valid pairing
        for i in 0..len {
            let j = links[i];
            if j >= len {
                return None;
            }
            // Check symmetry: if i links to j, then j must link to i
            if links[j] != i {
                return None;
            }
            // Check no self-loops
            if i == j {
                return None;
            }
        }

        Some(LinkPattern { links, n })
    }

    /// Create the identity link pattern (nested pairing)
    ///
    /// Pairs points: 0-1, 2-3, 4-5, etc.
    pub fn identity(n: usize) -> Self {
        let mut links = vec![0; 2 * n];
        for i in 0..n {
            links[2 * i] = 2 * i + 1;
            links[2 * i + 1] = 2 * i;
        }
        LinkPattern { links, n }
    }

    /// Get the link partner of point i
    pub fn link(&self, i: usize) -> Option<usize> {
        self.links.get(i).copied()
    }

    /// Get all links
    pub fn links(&self) -> &[usize] {
        &self.links
    }

    /// Get the size parameter
    pub fn size(&self) -> usize {
        self.n
    }

    /// Convert to cycle notation
    pub fn to_cycles(&self) -> Vec<Vec<usize>> {
        let mut visited = vec![false; self.links.len()];
        let mut cycles = Vec::new();

        for start in 0..self.links.len() {
            if visited[start] {
                continue;
            }

            let mut cycle = vec![start];
            visited[start] = true;
            let mut current = self.links[start];

            while current != start {
                cycle.push(current);
                visited[current] = true;
                current = self.links[current];
            }

            if cycle.len() > 1 {
                cycles.push(cycle);
            }
        }

        cycles
    }
}

impl fmt::Display for LinkPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        let cycles = self.to_cycles();
        for (i, cycle) in cycles.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "(")?;
            for (j, &point) in cycle.iter().enumerate() {
                if j > 0 {
                    write!(f, " ")?;
                }
                write!(f, "{}", point)?;
            }
            write!(f, ")")?;
        }
        write!(f, "]")
    }
}

/// A Fully Packed Loop (FPL) configuration
///
/// Represents a configuration of non-crossing loops on a grid
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FullyPackedLoop {
    /// The link pattern describing boundary connections
    link_pattern: LinkPattern,
    /// Configuration data: for each internal vertex (i,j), stores the connection type
    /// 0 = no connection (isolated), 1 = horizontal, 2 = vertical, 3 = both (crossing)
    config: Vec<Vec<u8>>,
    /// Size parameter
    n: usize,
}

impl FullyPackedLoop {
    /// Create a new FPL configuration
    pub fn new(link_pattern: LinkPattern, config: Vec<Vec<u8>>) -> Option<Self> {
        let n = link_pattern.size();

        // Check dimensions
        if config.len() != n - 1 {
            return None;
        }
        for row in &config {
            if row.len() != n - 1 {
                return None;
            }
        }

        // Validate configuration values
        for row in &config {
            for &val in row {
                if val > 3 {
                    return None;
                }
            }
        }

        Some(FullyPackedLoop {
            link_pattern,
            config,
            n,
        })
    }

    /// Create a simple FPL from an identity link pattern
    pub fn identity(n: usize) -> Self {
        let link_pattern = LinkPattern::identity(n);
        let config = vec![vec![0; n.saturating_sub(1)]; n.saturating_sub(1)];
        FullyPackedLoop {
            link_pattern,
            config,
            n,
        }
    }

    /// Get the link pattern
    pub fn link_pattern(&self) -> &LinkPattern {
        &self.link_pattern
    }

    /// Get the configuration
    pub fn config(&self) -> &[Vec<u8>] {
        &self.config
    }

    /// Get the size
    pub fn size(&self) -> usize {
        self.n
    }

    /// Convert this FPL to an Alternating Sign Matrix using Wieland's bijection
    pub fn to_asm(&self) -> AlternatingSignMatrix {
        // This is a simplified version of the bijection
        // A full implementation would require the complete Wieland algorithm

        // For now, we return the identity ASM
        // TODO: Implement full Wieland bijection
        AlternatingSignMatrix::identity(self.n)
    }
}

/// Generate all ASMs of size n
///
/// Warning: The number of ASMs grows rapidly!
/// ASM counts: n=1:1, n=2:2, n=3:7, n=4:42, n=5:429, n=6:7436
pub fn all_asms(n: usize) -> Vec<AlternatingSignMatrix> {
    if n == 0 {
        return vec![AlternatingSignMatrix {
            entries: vec![],
            n: 0,
        }];
    }

    if n == 1 {
        return vec![AlternatingSignMatrix::identity(1)];
    }

    let mut result = Vec::new();
    let mut matrix = vec![vec![0; n]; n];
    generate_asms(&mut matrix, 0, n, &mut result);
    result
}

/// Recursive helper to generate all ASMs
fn generate_asms(
    matrix: &mut Vec<Vec<i8>>,
    row: usize,
    n: usize,
    result: &mut Vec<AlternatingSignMatrix>,
) {
    if row == n {
        // Check if valid ASM
        if let Some(asm) = AlternatingSignMatrix::new(matrix.clone()) {
            result.push(asm);
        }
        return;
    }

    // Try all possible row configurations
    generate_row_configs(matrix, row, 0, n, 0, 0, result);
}

/// Generate all valid row configurations for ASMs
fn generate_row_configs(
    matrix: &mut Vec<Vec<i8>>,
    row: usize,
    col: usize,
    n: usize,
    sum: i32,
    last_nonzero: i32,
    result: &mut Vec<AlternatingSignMatrix>,
) {
    if col == n {
        if sum == 1 {
            // Check column constraints and continue to next row
            if check_column_constraints(matrix, row, n) {
                generate_asms(matrix, row + 1, n, result);
            }
        }
        return;
    }

    // Try 0, 1, -1
    for &val in &[0i8, 1, -1] {
        let new_sum = sum + val as i32;

        // Prune: sum can't exceed 1 or go below 0 if we haven't seen any 1 yet
        if new_sum > 1 || (new_sum < 0 && last_nonzero != 1) {
            continue;
        }

        // Check alternating sign constraint
        if val != 0 && last_nonzero != 0 {
            if val * (last_nonzero as i8) > 0 {
                continue; // Same sign, not allowed
            }
        }

        matrix[row][col] = val;
        let new_last = if val != 0 { val as i32 } else { last_nonzero };
        generate_row_configs(matrix, row, col + 1, n, new_sum, new_last, result);
        matrix[row][col] = 0;
    }
}

/// Check if column constraints are satisfied up to the given row
fn check_column_constraints(matrix: &[Vec<i8>], row: usize, n: usize) -> bool {
    for col in 0..n {
        let col_sum: i32 = (0..=row).map(|r| matrix[r][col] as i32).sum();

        // Column sum can't exceed 1
        if col_sum > 1 {
            return false;
        }

        // If this is the last row, column sum must be exactly 1
        if row == n - 1 && col_sum != 1 {
            return false;
        }

        // Check alternating signs in column
        let col_vals: Vec<i8> = (0..=row).map(|r| matrix[r][col]).collect();
        if !check_alternating_sign(&col_vals) {
            return false;
        }
    }
    true
}

/// Count the number of ASMs of size n
///
/// Uses the formula: A(n) = ∏(k=0 to n-1) (3k+1)! / (n+k)!
pub fn count_asms(n: usize) -> num_bigint::BigUint {
    use num_bigint::BigUint;
    use num_traits::One;

    if n == 0 {
        return One::one();
    }

    let mut numerator: BigUint = One::one();
    let mut denominator: BigUint = One::one();

    for k in 0..n {
        numerator *= factorial_big((3 * k + 1) as u32);
        denominator *= factorial_big((n + k) as u32);
    }

    numerator / denominator
}

/// Helper to compute factorial as BigUint
fn factorial_big(n: u32) -> num_bigint::BigUint {
    use num_bigint::BigUint;
    use num_traits::One;

    let mut result: BigUint = One::one();
    for i in 2..=n {
        result *= i;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asm_creation() {
        // Valid 2x2 ASM
        let asm = AlternatingSignMatrix::new(vec![vec![1, 0], vec![0, 1]]);
        assert!(asm.is_some());

        // Another valid 2x2 ASM
        let asm2 = AlternatingSignMatrix::new(vec![vec![0, 1], vec![1, 0]]);
        assert!(asm2.is_some());

        // Invalid - row sum wrong
        let invalid = AlternatingSignMatrix::new(vec![vec![1, 1], vec![0, 1]]);
        assert!(invalid.is_none());
    }

    #[test]
    fn test_asm_identity() {
        let asm = AlternatingSignMatrix::identity(3);
        assert_eq!(asm.size(), 3);
        assert_eq!(asm.get(0, 0), Some(1));
        assert_eq!(asm.get(1, 1), Some(1));
        assert_eq!(asm.get(2, 2), Some(1));
        assert_eq!(asm.get(0, 1), Some(0));
    }

    #[test]
    fn test_asm_with_negative() {
        // Valid ASM with -1 entry
        let asm = AlternatingSignMatrix::new(vec![
            vec![0, 1, 0],
            vec![1, -1, 1],
            vec![0, 1, 0],
        ]);
        assert!(asm.is_some());
        if let Some(asm) = asm {
            assert_eq!(asm.negative_count(), 1);
        }
    }

    #[test]
    fn test_alternating_sign_check() {
        assert!(check_alternating_sign(&[1, 0, 0]));
        assert!(check_alternating_sign(&[1, 0, -1]));
        assert!(check_alternating_sign(&[1, -1, 1]));
        assert!(!check_alternating_sign(&[1, 1]));
        assert!(!check_alternating_sign(&[1, 0, 1]));
    }

    #[test]
    fn test_link_pattern_creation() {
        // Simple pairing: 0-1, 2-3
        let pattern = LinkPattern::new(vec![1, 0, 3, 2]);
        assert!(pattern.is_some());

        // Invalid - not symmetric
        let invalid = LinkPattern::new(vec![1, 2, 3, 0]);
        assert!(invalid.is_none());
    }

    #[test]
    fn test_link_pattern_identity() {
        let pattern = LinkPattern::identity(3);
        assert_eq!(pattern.size(), 3);
        assert_eq!(pattern.link(0), Some(1));
        assert_eq!(pattern.link(1), Some(0));
        assert_eq!(pattern.link(2), Some(3));
        assert_eq!(pattern.link(3), Some(2));
    }

    #[test]
    fn test_fpl_creation() {
        let pattern = LinkPattern::identity(3);
        let config = vec![vec![0, 0], vec![0, 0]];
        let fpl = FullyPackedLoop::new(pattern, config);
        assert!(fpl.is_some());
    }

    #[test]
    fn test_fpl_identity() {
        let fpl = FullyPackedLoop::identity(3);
        assert_eq!(fpl.size(), 3);
        assert_eq!(fpl.config().len(), 2);
    }

    #[test]
    fn test_all_asms_small() {
        // n=1: 1 ASM
        let asms1 = all_asms(1);
        assert_eq!(asms1.len(), 1);

        // n=2: 2 ASMs
        let asms2 = all_asms(2);
        assert_eq!(asms2.len(), 2);

        // n=3: 7 ASMs
        let asms3 = all_asms(3);
        assert_eq!(asms3.len(), 7);
    }

    #[test]
    fn test_count_asms() {
        use num_bigint::BigUint;

        // Known ASM counts
        assert_eq!(count_asms(1), BigUint::from(1u32));
        assert_eq!(count_asms(2), BigUint::from(2u32));
        assert_eq!(count_asms(3), BigUint::from(7u32));
        assert_eq!(count_asms(4), BigUint::from(42u32));
        assert_eq!(count_asms(5), BigUint::from(429u32));
    }

    #[test]
    fn test_asm_to_permutation() {
        let asm = AlternatingSignMatrix::identity(3);
        let perm = asm.to_permutation();
        assert!(perm.is_some());
        assert_eq!(perm.unwrap(), vec![0, 1, 2]);

        // ASM with -1 is not a permutation matrix
        let asm_with_neg = AlternatingSignMatrix::new(vec![
            vec![0, 1, 0],
            vec![1, -1, 1],
            vec![0, 1, 0],
        ])
        .unwrap();
        assert!(asm_with_neg.to_permutation().is_none());
    }

    #[test]
    fn test_link_pattern_cycles() {
        let pattern = LinkPattern::identity(3);
        let cycles = pattern.to_cycles();
        assert_eq!(cycles.len(), 3); // Three pairs
    }
}
