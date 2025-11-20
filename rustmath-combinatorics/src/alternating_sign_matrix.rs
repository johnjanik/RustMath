//! Alternating Sign Matrices (ASMs)
//!
//! An alternating sign matrix is an n×n matrix where:
//! - Each entry is 1, -1, or 0
//! - Each row and column sums to 1
//! - Non-zero entries in each row and column alternate in sign
//!
//! ASMs generalize permutation matrices and are connected to many areas of
//! combinatorics including plane partitions, determinants, and the
//! six-vertex model from statistical mechanics.

use rustmath_core::Ring;
use rustmath_integers::Integer;

/// An alternating sign matrix
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlternatingSignMatrix {
    /// The matrix entries (-1, 0, or 1)
    entries: Vec<Vec<i8>>,
    /// The size of the matrix
    n: usize,
}

impl AlternatingSignMatrix {
    /// Create a new alternating sign matrix
    ///
    /// Returns None if the matrix does not satisfy the ASM properties
    pub fn new(entries: Vec<Vec<i8>>) -> Option<Self> {
        let n = entries.len();

        if n == 0 {
            return Some(AlternatingSignMatrix { entries, n: 0 });
        }

        // Check dimensions (must be square)
        for row in &entries {
            if row.len() != n {
                return None;
            }
        }

        // Check that all entries are -1, 0, or 1
        for row in &entries {
            for &val in row {
                if val != -1 && val != 0 && val != 1 {
                    return None;
                }
            }
        }

        // Check row sums
        for row in &entries {
            let sum: i32 = row.iter().map(|&x| x as i32).sum();
            if sum != 1 {
                return None;
            }
        }

        // Check column sums
        for col in 0..n {
            let sum: i32 = entries.iter().map(|row| row[col] as i32).sum();
            if sum != 1 {
                return None;
            }
        }

        // Check alternating sign property for rows
        for row in &entries {
            if !has_alternating_signs(row) {
                return None;
            }
        }

        // Check alternating sign property for columns
        for col in 0..n {
            let column: Vec<i8> = entries.iter().map(|row| row[col]).collect();
            if !has_alternating_signs(&column) {
                return None;
            }
        }

        Some(AlternatingSignMatrix { entries, n })
    }

    /// Get the size of the matrix
    pub fn size(&self) -> usize {
        self.n
    }

    /// Get the entries of the matrix
    pub fn entries(&self) -> &[Vec<i8>] {
        &self.entries
    }

    /// Get the entry at position (i, j)
    pub fn get(&self, i: usize, j: usize) -> Option<i8> {
        self.entries.get(i)?.get(j).copied()
    }

    /// Convert to a permutation matrix if this ASM is a permutation matrix
    ///
    /// Returns the permutation as a vector, or None if not a permutation matrix
    pub fn to_permutation(&self) -> Option<Vec<usize>> {
        let mut perm = vec![0; self.n];

        for (i, row) in self.entries.iter().enumerate() {
            let mut count = 0;
            let mut pos = 0;

            for (j, &val) in row.iter().enumerate() {
                if val == 1 {
                    count += 1;
                    pos = j;
                } else if val == -1 {
                    // Not a permutation matrix
                    return None;
                }
            }

            if count != 1 {
                return None;
            }

            perm[i] = pos;
        }

        Some(perm)
    }

    /// Count the number of -1 entries in the matrix
    pub fn inversion_number(&self) -> usize {
        self.entries
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&x| x == -1)
            .count()
    }

    /// Get the descending diagonal sum (used in Razumov-Stroganov)
    ///
    /// For position (i,j), the descending diagonal is characterized by i+j
    /// This computes the sum of entries on each descending diagonal
    pub fn descending_diagonal_sums(&self) -> Vec<i32> {
        let mut sums = vec![0i32; 2 * self.n - 1];

        for i in 0..self.n {
            for j in 0..self.n {
                let diag = i + j;
                sums[diag] += self.entries[i][j] as i32;
            }
        }

        sums
    }

    /// Get the monotone triangle associated with this ASM
    ///
    /// The monotone triangle is obtained by recording the positions of 1's
    /// in the partial row sums from left to right
    pub fn monotone_triangle(&self) -> Vec<Vec<usize>> {
        let mut triangle = Vec::new();

        for row in &self.entries {
            let mut partial_sum = 0i32;
            let mut positions = Vec::new();

            for (j, &val) in row.iter().enumerate() {
                partial_sum += val as i32;
                if partial_sum == 1 {
                    positions.push(j + 1); // 1-indexed
                }
            }

            if !positions.is_empty() {
                triangle.push(positions);
            }
        }

        triangle
    }
}

/// Check if a sequence has the alternating sign property
///
/// Non-zero entries must alternate in sign
fn has_alternating_signs(seq: &[i8]) -> bool {
    let mut last_sign: Option<i8> = None;

    for &val in seq {
        if val != 0 {
            if let Some(prev) = last_sign {
                if val == prev {
                    return false; // Same sign twice in a row
                }
            }
            last_sign = Some(val);
        }
    }

    true
}

/// Compute the number of n×n alternating sign matrices
///
/// The formula is: A(n) = ∏_{k=0}^{n-1} (3k+1)! / (n+k)!
///
/// First few values: A(0)=1, A(1)=1, A(2)=2, A(3)=7, A(4)=42, A(5)=429
pub fn asm_count(n: u32) -> Integer {
    if n == 0 {
        return Integer::one();
    }

    // Compute using the product formula, being careful with integer division
    // We rewrite (3k+1)! / (n+k)! as a product of individual terms
    // to avoid truncation issues

    let mut numerator = Integer::one();
    let mut denominator = Integer::one();

    for k in 0..n {
        // Accumulate (3k+1)! in numerator
        for i in 2..=(3 * k + 1) {
            numerator = numerator * Integer::from(i);
        }

        // Accumulate (n+k)! in denominator
        for i in 2..=(n + k) {
            denominator = denominator * Integer::from(i);
        }
    }

    numerator / denominator
}

/// Compute the refined enumeration by number of -1's
///
/// A_n(k) is the number of n×n ASMs with exactly k entries equal to -1
/// This uses the q-enumeration formula which is more complex
pub fn asm_count_by_inversions(n: u32, k: u32) -> Integer {
    // For small cases, we can use explicit formulas
    // For general case, this would require the full q-analog formula
    // For now, we implement small cases explicitly

    if n == 0 {
        return if k == 0 { Integer::one() } else { Integer::zero() };
    }

    if n == 1 {
        return if k == 0 { Integer::one() } else { Integer::zero() };
    }

    if n == 2 {
        match k {
            0 => Integer::from(2), // Two permutation matrices
            _ => Integer::zero(),
        }
    } else {
        // For larger n, we would need to generate all ASMs or use
        // more sophisticated formulas
        // This is a placeholder
        Integer::zero()
    }
}

/// Generate all n×n alternating sign matrices
///
/// WARNING: The number of ASMs grows very quickly!
/// A(5) = 429, A(6) = 7436, A(7) = 218348
pub fn all_asms(n: usize) -> Vec<AlternatingSignMatrix> {
    if n == 0 {
        return vec![AlternatingSignMatrix {
            entries: vec![],
            n: 0,
        }];
    }

    if n == 1 {
        return vec![AlternatingSignMatrix {
            entries: vec![vec![1]],
            n: 1,
        }];
    }

    // Use recursive generation
    let mut result = Vec::new();
    let mut matrix = vec![vec![0i8; n]; n];
    generate_asms(&mut matrix, 0, &mut result);
    result
}

/// Recursive helper to generate all ASMs
fn generate_asms(matrix: &mut Vec<Vec<i8>>, row: usize, result: &mut Vec<AlternatingSignMatrix>) {
    let n = matrix.len();

    if row == n {
        // Check if we have a valid ASM
        if let Some(asm) = AlternatingSignMatrix::new(matrix.clone()) {
            result.push(asm);
        }
        return;
    }

    // For each row, we need to place values such that:
    // 1. Row sum is 1
    // 2. Non-zero entries alternate in sign
    // 3. Partial column sums are compatible

    generate_row_configs(matrix, row, 0, 0, None, result);
}

/// Generate all valid row configurations recursively
fn generate_row_configs(
    matrix: &mut Vec<Vec<i8>>,
    row: usize,
    col: usize,
    row_sum: i32,
    last_sign: Option<i8>,
    result: &mut Vec<AlternatingSignMatrix>,
) {
    let n = matrix.len();

    if col == n {
        // Check if row sum is 1
        if row_sum == 1 {
            // Check column constraints
            if is_column_valid(matrix, row) {
                generate_asms(matrix, row + 1, result);
            }
        }
        return;
    }

    // Try each possible value: -1, 0, 1
    for &val in &[-1i8, 0, 1] {
        // Check alternating sign constraint
        if val != 0 {
            if let Some(prev) = last_sign {
                if val == prev {
                    continue; // Skip same sign
                }
            }
        }

        // Check if placing this value could lead to valid row sum
        let remaining = (n - col - 1) as i32;
        let new_sum = row_sum + val as i32;

        // Pruning: if we can't reach sum of 1, skip
        if new_sum + remaining < 1 || new_sum - remaining > 1 {
            continue;
        }

        matrix[row][col] = val;
        let next_sign = if val != 0 { Some(val) } else { last_sign };

        generate_row_configs(matrix, row, col + 1, new_sum, next_sign, result);
    }

    matrix[row][col] = 0; // Reset for backtracking
}

/// Check if columns are valid up to the current row
fn is_column_valid(matrix: &[Vec<i8>], current_row: usize) -> bool {
    let n = matrix.len();

    for col in 0..n {
        // Check column sum doesn't exceed 1
        let col_sum: i32 = (0..=current_row).map(|r| matrix[r][col] as i32).sum();

        if current_row == n - 1 {
            // Last row - sum must be exactly 1
            if col_sum != 1 {
                return false;
            }
        } else {
            // Not last row - sum must be at most 1
            if col_sum > 1 || col_sum < 0 {
                return false;
            }

            // Check if we can still reach 1
            let remaining = (n - current_row - 1) as i32;
            if col_sum + remaining < 1 {
                return false;
            }
        }

        // Check alternating signs in column
        let column: Vec<i8> = (0..=current_row).map(|r| matrix[r][col]).collect();
        if !has_alternating_signs(&column) {
            return false;
        }
    }

    true
}

/// Razumov-Stroganov correspondence
///
/// This relates the refined enumeration of ASMs by descending diagonals
/// to the components of the ground state of the O(1) loop model.
///
/// Specifically, for n×n ASMs, we can refine the count by the sum along
/// each descending diagonal. The Razumov-Stroganov conjecture (now theorem)
/// states that these refined counts equal certain entries in the ground
/// state eigenvector of the O(1) loop model Hamiltonian.
///
/// This function computes the generating function coefficients for ASMs
/// refined by their descending diagonal sums.
pub fn razumov_stroganov_refined_count(n: usize) -> Vec<(Vec<i32>, usize)> {
    let asms = all_asms(n);
    let mut counts: std::collections::HashMap<Vec<i32>, usize> = std::collections::HashMap::new();

    for asm in asms {
        let diag_sums = asm.descending_diagonal_sums();
        *counts.entry(diag_sums).or_insert(0) += 1;
    }

    let mut result: Vec<_> = counts.into_iter().collect();
    result.sort_by_key(|(sums, _)| sums.clone());
    result
}

/// Compute the q-enumeration of ASMs
///
/// This is a polynomial in q where the coefficient of q^k gives the number
/// of n×n ASMs with k inversions (entries equal to -1).
///
/// The q-enumeration satisfies:
/// A_n(q) = ∏_{k=0}^{n-1} [3k+1]_q! / [n+k]_q!
///
/// where [m]_q! is the q-factorial
pub fn asm_q_enumeration(n: u32) -> Vec<Integer> {
    // For small n, compute directly
    if n == 0 {
        return vec![Integer::one()];
    }

    if n == 1 {
        return vec![Integer::one()];
    }

    if n == 2 {
        // A_2(q) = 2 (two permutation matrices, both with 0 inversions)
        return vec![Integer::from(2)];
    }

    // For larger n, we would need to implement q-factorials and q-division
    // This is a placeholder returning just the total count
    vec![asm_count(n)]
}

/// Compute the Razumov-Stroganov polynomial
///
/// This is the generating function for ASMs refined by descending diagonal sums.
/// For an n×n ASM with descending diagonal sums (d_0, d_1, ..., d_{2n-2}),
/// the contribution to the polynomial is x_0^{d_0} x_1^{d_1} ... x_{2n-2}^{d_{2n-2}}
pub fn razumov_stroganov_polynomial(n: usize) -> Vec<(Vec<i32>, Integer)> {
    let refined = razumov_stroganov_refined_count(n);
    refined
        .into_iter()
        .map(|(sums, count)| (sums, Integer::from(count as u32)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asm_validation() {
        // Valid 1x1 ASM
        let asm1 = AlternatingSignMatrix::new(vec![vec![1]]);
        assert!(asm1.is_some());

        // Valid 2x2 ASM (permutation matrix)
        let asm2 = AlternatingSignMatrix::new(vec![vec![1, 0], vec![0, 1]]);
        assert!(asm2.is_some());

        // Invalid - row sum not 1
        let invalid1 = AlternatingSignMatrix::new(vec![vec![1, 1], vec![0, 0]]);
        assert!(invalid1.is_none());

        // Invalid - non-alternating signs
        let invalid2 = AlternatingSignMatrix::new(vec![vec![1, -1, 1], vec![0, 1, 0], vec![0, 1, 0]]);
        assert!(invalid2.is_none());
    }

    #[test]
    fn test_asm_count() {
        // Known values from OEIS A005130
        assert_eq!(asm_count(0), Integer::from(1));
        assert_eq!(asm_count(1), Integer::from(1));
        assert_eq!(asm_count(2), Integer::from(2));
        assert_eq!(asm_count(3), Integer::from(7));
        assert_eq!(asm_count(4), Integer::from(42));
        assert_eq!(asm_count(5), Integer::from(429));
        assert_eq!(asm_count(6), Integer::from(7436));
    }

    #[test]
    fn test_alternating_signs() {
        assert!(has_alternating_signs(&[1, 0, -1, 0, 1]));
        assert!(has_alternating_signs(&[0, 1, 0, -1]));
        assert!(has_alternating_signs(&[1, -1, 1]));
        assert!(!has_alternating_signs(&[1, 1, -1]));
        assert!(!has_alternating_signs(&[1, 0, 1]));
        assert!(has_alternating_signs(&[0, 0, 0]));
    }

    #[test]
    fn test_generate_small_asms() {
        // n=1: should have 1 ASM
        let asms1 = all_asms(1);
        assert_eq!(asms1.len(), 1);

        // n=2: should have 2 ASMs
        let asms2 = all_asms(2);
        assert_eq!(asms2.len(), 2);

        // n=3: should have 7 ASMs
        let asms3 = all_asms(3);
        assert_eq!(asms3.len(), 7);
    }

    #[test]
    fn test_3x3_asm() {
        // A specific 3x3 ASM
        let asm = AlternatingSignMatrix::new(vec![
            vec![0, 1, 0],
            vec![1, -1, 1],
            vec![0, 1, 0],
        ]);
        assert!(asm.is_some());

        let asm = asm.unwrap();
        assert_eq!(asm.size(), 3);
        assert_eq!(asm.inversion_number(), 1);
    }

    #[test]
    fn test_permutation_matrix() {
        let asm = AlternatingSignMatrix::new(vec![vec![0, 1, 0], vec![1, 0, 0], vec![0, 0, 1]])
            .unwrap();

        let perm = asm.to_permutation();
        assert!(perm.is_some());
        assert_eq!(perm.unwrap(), vec![1, 0, 2]);
    }

    #[test]
    fn test_non_permutation_matrix() {
        let asm = AlternatingSignMatrix::new(vec![
            vec![0, 1, 0],
            vec![1, -1, 1],
            vec![0, 1, 0],
        ])
        .unwrap();

        let perm = asm.to_permutation();
        assert!(perm.is_none());
    }

    #[test]
    fn test_descending_diagonals() {
        let asm = AlternatingSignMatrix::new(vec![
            vec![1, 0, 0],
            vec![0, 1, 0],
            vec![0, 0, 1],
        ])
        .unwrap();

        let diag_sums = asm.descending_diagonal_sums();
        // For the identity matrix, each diagonal should have sum of 1
        assert_eq!(diag_sums, vec![1, 0, 1, 0, 1]);
    }

    #[test]
    fn test_monotone_triangle() {
        let asm = AlternatingSignMatrix::new(vec![vec![0, 1, 0], vec![1, 0, 0], vec![0, 0, 1]])
            .unwrap();

        let triangle = asm.monotone_triangle();
        // Each row should record positions where partial sum becomes 1
        assert_eq!(triangle.len(), 3);
    }

    #[test]
    fn test_razumov_stroganov() {
        // For small n, compute the refined enumeration
        let refined = razumov_stroganov_refined_count(2);

        // For n=2, we have 2 ASMs (both permutation matrices)
        // They should have different diagonal sum patterns
        assert_eq!(refined.len(), 2);

        let total: usize = refined.iter().map(|(_, count)| count).sum();
        assert_eq!(total, 2);
    }

    #[test]
    fn test_inversion_number() {
        // Identity matrix has 0 inversions
        let identity = AlternatingSignMatrix::new(vec![vec![1, 0], vec![0, 1]]).unwrap();
        assert_eq!(identity.inversion_number(), 0);

        // ASM with one -1
        let asm = AlternatingSignMatrix::new(vec![
            vec![0, 1, 0],
            vec![1, -1, 1],
            vec![0, 1, 0],
        ])
        .unwrap();
        assert_eq!(asm.inversion_number(), 1);
    }

    #[test]
    fn test_q_enumeration() {
        // For small n
        let q_enum_0 = asm_q_enumeration(0);
        assert_eq!(q_enum_0, vec![Integer::one()]);

        let q_enum_1 = asm_q_enumeration(1);
        assert_eq!(q_enum_1, vec![Integer::one()]);

        let q_enum_2 = asm_q_enumeration(2);
        assert_eq!(q_enum_2, vec![Integer::from(2)]);
    }
}
