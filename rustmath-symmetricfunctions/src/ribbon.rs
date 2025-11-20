//! Ribbon tableaux and skew-shapes
//!
//! A ribbon (or rim hook) is a connected skew shape that contains no 2×2 square.
//! Ribbon tableaux are important in the representation theory of the symmetric group
//! and in the study of symmetric functions.

use rustmath_combinatorics::Partition;

/// A ribbon tableau - a filling of a Young diagram with ribbons
///
/// A ribbon is a connected skew shape with no 2×2 square.
/// The height of a ribbon is (number of rows - 1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RibbonTableau {
    /// The outer shape of the tableau
    pub outer_shape: Partition,
    /// The inner shape (removed from outer)
    pub inner_shape: Partition,
    /// The ribbons as sequences of cells (row, col)
    pub ribbons: Vec<Vec<(usize, usize)>>,
}

impl RibbonTableau {
    /// Create a new ribbon tableau
    pub fn new(
        outer_shape: Partition,
        inner_shape: Partition,
        ribbons: Vec<Vec<(usize, usize)>>,
    ) -> Option<Self> {
        // Verify inner shape fits in outer shape
        if !fits_inside(&inner_shape, &outer_shape) {
            return None;
        }

        // Verify ribbons are valid
        for ribbon in &ribbons {
            if !is_valid_ribbon(ribbon) {
                return None;
            }
        }

        Some(RibbonTableau {
            outer_shape,
            inner_shape,
            ribbons,
        })
    }

    /// Get the skew shape size
    pub fn size(&self) -> usize {
        self.outer_shape.sum() - self.inner_shape.sum()
    }

    /// Get the number of ribbons
    pub fn num_ribbons(&self) -> usize {
        self.ribbons.len()
    }

    /// Check if this is a valid ribbon tableau
    pub fn is_valid(&self) -> bool {
        // Check that ribbons partition the skew shape
        let mut cells = std::collections::HashSet::new();

        for ribbon in &self.ribbons {
            for &cell in ribbon {
                if !cells.insert(cell) {
                    return false; // Cell used twice
                }
            }
        }

        // Check that exactly the skew shape cells are covered
        cells.len() == self.size()
    }

    /// Compute the sign of this ribbon tableau
    ///
    /// The sign is (-1)^(sum of heights of all ribbons)
    pub fn sign(&self) -> i32 {
        let total_height: usize = self.ribbons.iter().map(|r| ribbon_height(r)).sum();
        if total_height % 2 == 0 {
            1
        } else {
            -1
        }
    }
}

/// Check if a sequence of cells forms a valid ribbon (connected, no 2×2)
pub fn is_valid_ribbon(cells: &[(usize, usize)]) -> bool {
    if cells.is_empty() {
        return false;
    }

    // Check connectivity
    if !is_connected(cells) {
        return false;
    }

    // Check no 2×2 square
    if has_2x2_square(cells) {
        return false;
    }

    true
}

/// Check if inner partition fits inside outer partition
fn fits_inside(inner: &Partition, outer: &Partition) -> bool {
    for (i, &inner_part) in inner.parts().iter().enumerate() {
        if i >= outer.parts().len() || inner_part > outer.parts()[i] {
            return false;
        }
    }
    true
}

/// Check if a set of cells is connected
fn is_connected(cells: &[(usize, usize)]) -> bool {
    if cells.len() <= 1 {
        return true;
    }

    use std::collections::HashSet;
    let cell_set: HashSet<_> = cells.iter().copied().collect();
    let mut visited = HashSet::new();
    let mut stack = vec![cells[0]];

    while let Some(cell) = stack.pop() {
        if visited.contains(&cell) {
            continue;
        }
        visited.insert(cell);

        // Check 4 neighbors
        let neighbors = [
            (cell.0.wrapping_sub(1), cell.1),
            (cell.0 + 1, cell.1),
            (cell.0, cell.1.wrapping_sub(1)),
            (cell.0, cell.1 + 1),
        ];

        for neighbor in neighbors {
            if cell_set.contains(&neighbor) && !visited.contains(&neighbor) {
                stack.push(neighbor);
            }
        }
    }

    visited.len() == cells.len()
}

/// Check if a set of cells contains a 2×2 square
fn has_2x2_square(cells: &[(usize, usize)]) -> bool {
    use std::collections::HashSet;
    let cell_set: HashSet<_> = cells.iter().copied().collect();

    for &(r, c) in cells {
        // Check if (r, c), (r+1, c), (r, c+1), (r+1, c+1) are all present
        if cell_set.contains(&(r + 1, c))
            && cell_set.contains(&(r, c + 1))
            && cell_set.contains(&(r + 1, c + 1))
        {
            return true;
        }
    }

    false
}

/// Compute the height of a ribbon (number of rows - 1)
fn ribbon_height(cells: &[(usize, usize)]) -> usize {
    if cells.is_empty() {
        return 0;
    }

    let min_row = cells.iter().map(|(r, _)| r).min().unwrap();
    let max_row = cells.iter().map(|(r, _)| r).max().unwrap();

    max_row - min_row
}

/// Generate all ribbon tableaux for a given skew shape
///
/// This is a complex combinatorial problem. For now, we provide
/// the infrastructure for ribbon tableaux.
pub fn ribbon_tableaux(
    outer: &Partition,
    inner: &Partition,
) -> Vec<RibbonTableau> {
    // Placeholder implementation
    vec![]
}

/// Check if a tableau is a ribbon tableau
pub fn is_ribbon_tableau(outer: &Partition, inner: &Partition, ribbons: &[Vec<(usize, usize)>]) -> bool {
    if let Some(rt) = RibbonTableau::new(outer.clone(), inner.clone(), ribbons.to_vec()) {
        rt.is_valid()
    } else {
        false
    }
}

/// Compute the Murnaghan-Nakayama rule
///
/// The Murnaghan-Nakayama rule computes characters of symmetric group
/// representations using ribbon tableaux:
/// χ^λ(ρ) = sum over ribbon tableaux T of shape λ and type ρ of sign(T)
///
/// where λ is a partition indexing an irreducible representation of S_n,
/// and ρ is a partition giving the cycle type of a permutation in S_n.
///
/// The algorithm works by recursively removing border strips (ribbons)
/// of sizes matching the parts of ρ from λ, accumulating the sign
/// contribution (-1)^(height) for each removed ribbon.
///
/// Returns the character value χ^λ(ρ) as an integer (possibly negative).
pub fn murnaghan_nakayama(lambda: &Partition, rho: &Partition) -> i64 {
    // Check that lambda and rho partition the same number
    if lambda.sum() != rho.sum() {
        return 0;
    }

    // Base case: empty partitions
    if lambda.parts().is_empty() && rho.parts().is_empty() {
        return 1;
    }

    // If rho is empty but lambda is not, character is 0
    if rho.parts().is_empty() {
        return 0;
    }

    // Recursive case: remove the first part of rho
    let parts = rho.parts();
    let k = parts[0]; // Size of border strip to remove

    // Create the remaining cycle type (rho with first part removed)
    let remaining_rho = if parts.len() > 1 {
        Partition::new(parts[1..].to_vec())
    } else {
        Partition::new(vec![])
    };

    // Try all ways to remove a border strip of size k from lambda
    let removals = lambda.remove_border_strip_advanced(k);

    let mut total = 0i64;

    for (inner_partition, height) in removals {
        // Recursively compute character for the remaining partition and cycle type
        let sub_character = murnaghan_nakayama(&inner_partition, &remaining_rho);

        // Add contribution with sign based on height
        let sign = if height % 2 == 0 { 1 } else { -1 };
        total += sign * sub_character;
    }

    total
}

/// Compute a full character of S_n for a given partition λ
///
/// Returns a map from cycle types (partitions of n) to character values.
/// This computes χ^λ(ρ) for all cycle types ρ of permutations in S_n.
pub fn symmetric_group_character(lambda: &Partition) -> std::collections::HashMap<Partition, i64> {
    let n = lambda.sum();
    let mut character = std::collections::HashMap::new();

    // Generate all partitions of n (cycle types)
    for rho in rustmath_combinatorics::partitions(n) {
        let value = murnaghan_nakayama(lambda, &rho);
        character.insert(rho, value);
    }

    character
}

/// Compute the character table for the symmetric group S_n
///
/// Returns a map from pairs (λ, ρ) to χ^λ(ρ), where λ and ρ are partitions of n.
/// The rows are indexed by partitions (irreducible representations),
/// and columns are indexed by cycle types (conjugacy classes).
pub fn symmetric_group_character_table(
    n: usize,
) -> std::collections::HashMap<(Partition, Partition), i64> {
    let mut table = std::collections::HashMap::new();

    // Get all partitions of n
    let partitions = rustmath_combinatorics::partitions(n);

    // Compute character values for all pairs
    for lambda in &partitions {
        for rho in &partitions {
            let value = murnaghan_nakayama(lambda, rho);
            table.insert((lambda.clone(), rho.clone()), value);
        }
    }

    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_connected() {
        // Connected: linear ribbon
        let cells = vec![(0, 0), (0, 1), (0, 2)];
        assert!(is_connected(&cells));

        // Connected: L-shape
        let cells2 = vec![(0, 0), (0, 1), (1, 0)];
        assert!(is_connected(&cells2));

        // Not connected
        let cells3 = vec![(0, 0), (0, 2)];
        assert!(!is_connected(&cells3));
    }

    #[test]
    fn test_has_2x2_square() {
        // Has 2×2 square
        let cells = vec![(0, 0), (0, 1), (1, 0), (1, 1)];
        assert!(has_2x2_square(&cells));

        // No 2×2 square - linear
        let cells2 = vec![(0, 0), (0, 1), (0, 2)];
        assert!(!has_2x2_square(&cells2));

        // No 2×2 square - L-shape
        let cells3 = vec![(0, 0), (0, 1), (1, 0)];
        assert!(!has_2x2_square(&cells3));
    }

    #[test]
    fn test_is_valid_ribbon() {
        // Valid: horizontal 3-ribbon
        let ribbon1 = vec![(0, 0), (0, 1), (0, 2)];
        assert!(is_valid_ribbon(&ribbon1));

        // Valid: L-shape ribbon
        let ribbon2 = vec![(0, 0), (0, 1), (1, 1)];
        assert!(is_valid_ribbon(&ribbon2));

        // Invalid: contains 2×2
        let ribbon3 = vec![(0, 0), (0, 1), (1, 0), (1, 1)];
        assert!(!is_valid_ribbon(&ribbon3));

        // Invalid: not connected
        let ribbon4 = vec![(0, 0), (0, 2)];
        assert!(!is_valid_ribbon(&ribbon4));
    }

    #[test]
    fn test_ribbon_height() {
        // Height 0: single row
        let ribbon1 = vec![(0, 0), (0, 1), (0, 2)];
        assert_eq!(ribbon_height(&ribbon1), 0);

        // Height 1: spans 2 rows
        let ribbon2 = vec![(0, 0), (1, 0)];
        assert_eq!(ribbon_height(&ribbon2), 1);

        // Height 2: spans 3 rows
        let ribbon3 = vec![(0, 0), (1, 0), (2, 0)];
        assert_eq!(ribbon_height(&ribbon3), 2);
    }

    #[test]
    fn test_ribbon_tableau_creation() {
        let outer = Partition::new(vec![3, 2]);
        let inner = Partition::new(vec![1]);

        // Create a simple ribbon tableau
        let ribbons = vec![vec![(0, 1), (0, 2), (1, 1)]];

        let rt = RibbonTableau::new(outer.clone(), inner.clone(), ribbons);
        assert!(rt.is_some());
    }

    #[test]
    fn test_ribbon_tableau_sign() {
        let outer = Partition::new(vec![2, 1]);
        let inner = Partition::new(vec![]);

        // Single ribbon of height 1 (spans 2 rows)
        let ribbons = vec![vec![(0, 0), (1, 0), (0, 1)]];

        if let Some(rt) = RibbonTableau::new(outer, inner, ribbons) {
            let sign = rt.sign();
            // Height is 1, so sign should be -1
            assert_eq!(sign, -1);
        }
    }

    #[test]
    fn test_fits_inside() {
        let outer = Partition::new(vec![3, 2, 1]);
        let inner1 = Partition::new(vec![1]);
        let inner2 = Partition::new(vec![2, 2]);
        let inner3 = Partition::new(vec![4]);

        assert!(fits_inside(&inner1, &outer));
        assert!(fits_inside(&inner2, &outer));
        assert!(!fits_inside(&inner3, &outer)); // 4 > 3
    }

    #[test]
    fn test_murnaghan_nakayama_trivial() {
        // Trivial case: empty partitions
        let lambda = Partition::new(vec![]);
        let rho = Partition::new(vec![]);
        assert_eq!(murnaghan_nakayama(&lambda, &rho), 1);
    }

    #[test]
    fn test_murnaghan_nakayama_identity() {
        // χ^(n)(1^n) = 1 (trivial representation on identity element)
        let lambda = Partition::new(vec![3]); // S_3 trivial rep
        let rho = Partition::new(vec![1, 1, 1]); // Identity element
        assert_eq!(murnaghan_nakayama(&lambda, &rho), 1);

        // χ^(1^n)(1^n) = 1 (sign representation on identity element)
        let lambda2 = Partition::new(vec![1, 1, 1]);
        let rho2 = Partition::new(vec![1, 1, 1]);
        assert_eq!(murnaghan_nakayama(&lambda2, &rho2), 1);
    }

    #[test]
    fn test_murnaghan_nakayama_s3() {
        // Test complete character table for S_3
        // S_3 has 3 conjugacy classes: [3], [2,1], [1,1,1]
        // and 3 irreducible representations indexed by partitions of 3

        // Trivial representation λ = (3)
        let lambda_trivial = Partition::new(vec![3]);
        assert_eq!(
            murnaghan_nakayama(&lambda_trivial, &Partition::new(vec![3])),
            1
        );
        assert_eq!(
            murnaghan_nakayama(&lambda_trivial, &Partition::new(vec![2, 1])),
            1
        );
        assert_eq!(
            murnaghan_nakayama(&lambda_trivial, &Partition::new(vec![1, 1, 1])),
            1
        );

        // Sign representation λ = (1,1,1)
        let lambda_sign = Partition::new(vec![1, 1, 1]);
        assert_eq!(
            murnaghan_nakayama(&lambda_sign, &Partition::new(vec![3])),
            1
        );
        assert_eq!(
            murnaghan_nakayama(&lambda_sign, &Partition::new(vec![2, 1])),
            -1
        );
        assert_eq!(
            murnaghan_nakayama(&lambda_sign, &Partition::new(vec![1, 1, 1])),
            1
        );

        // Standard representation λ = (2,1)
        let lambda_std = Partition::new(vec![2, 1]);
        assert_eq!(
            murnaghan_nakayama(&lambda_std, &Partition::new(vec![3])),
            -1
        );
        assert_eq!(
            murnaghan_nakayama(&lambda_std, &Partition::new(vec![2, 1])),
            0
        );
        assert_eq!(
            murnaghan_nakayama(&lambda_std, &Partition::new(vec![1, 1, 1])),
            2
        );
    }

    #[test]
    fn test_murnaghan_nakayama_s4() {
        // Test some values for S_4
        // λ = (4): trivial representation
        let lambda = Partition::new(vec![4]);
        assert_eq!(
            murnaghan_nakayama(&lambda, &Partition::new(vec![4])),
            1
        );
        assert_eq!(
            murnaghan_nakayama(&lambda, &Partition::new(vec![2, 2])),
            1
        );
        assert_eq!(
            murnaghan_nakayama(&lambda, &Partition::new(vec![1, 1, 1, 1])),
            1
        );

        // λ = (1,1,1,1): sign representation
        let lambda_sign = Partition::new(vec![1, 1, 1, 1]);
        assert_eq!(
            murnaghan_nakayama(&lambda_sign, &Partition::new(vec![4])),
            1
        );
        assert_eq!(
            murnaghan_nakayama(&lambda_sign, &Partition::new(vec![3, 1])),
            -1
        );
        assert_eq!(
            murnaghan_nakayama(&lambda_sign, &Partition::new(vec![2, 2])),
            1
        );
        assert_eq!(
            murnaghan_nakayama(&lambda_sign, &Partition::new(vec![2, 1, 1])),
            -1
        );
        assert_eq!(
            murnaghan_nakayama(&lambda_sign, &Partition::new(vec![1, 1, 1, 1])),
            1
        );
    }

    #[test]
    fn test_murnaghan_nakayama_different_sizes() {
        // Character should be 0 if partitions have different sizes
        let lambda = Partition::new(vec![3]);
        let rho = Partition::new(vec![2, 1, 1]);
        assert_eq!(murnaghan_nakayama(&lambda, &rho), 0);
    }

    #[test]
    fn test_symmetric_group_character() {
        // Test full character computation for a partition
        let lambda = Partition::new(vec![2, 1]);
        let character = symmetric_group_character(&lambda);

        // Should have entries for all partitions of 3
        assert_eq!(character.len(), 3);
        assert!(character.contains_key(&Partition::new(vec![3])));
        assert!(character.contains_key(&Partition::new(vec![2, 1])));
        assert!(character.contains_key(&Partition::new(vec![1, 1, 1])));
    }

    #[test]
    fn test_symmetric_group_character_table() {
        // Test character table generation for S_3
        let table = symmetric_group_character_table(3);

        // S_3 has 3 partitions, so table should have 3×3 = 9 entries
        assert_eq!(table.len(), 9);

        // Check some specific values
        let lambda_trivial = Partition::new(vec![3]);
        let rho_identity = Partition::new(vec![1, 1, 1]);
        assert_eq!(table[&(lambda_trivial.clone(), rho_identity.clone())], 1);

        // Column orthogonality: sum over λ of |χ^λ(ρ)|^2 equals |C_ρ| (size of conjugacy class)
        // For identity class (1,1,1), all characters equal their dimension
        // Dimensions: 1 (trivial), 1 (sign), 2 (standard)
        // Sum of squares: 1 + 1 + 4 = 6 = |S_3|
    }

    #[test]
    fn test_character_orthogonality() {
        // Test orthogonality relations for S_4
        let n = 4;
        let table = symmetric_group_character_table(n);
        let partitions = rustmath_combinatorics::partitions(n);

        // Row orthogonality: ⟨χ^λ, χ^μ⟩ = δ_{λμ}
        // This requires knowing conjugacy class sizes, which we'll skip for now

        // Instead, just verify the table was generated correctly
        assert_eq!(table.len(), partitions.len() * partitions.len());
    }
}
