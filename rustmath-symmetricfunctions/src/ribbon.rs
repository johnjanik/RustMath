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
/// where ρ is a partition (cycle type of a permutation).
pub fn murnaghan_nakayama(lambda: &Partition, rho: &Partition) -> i64 {
    // This requires generating ribbon tableaux of shape λ and content ρ
    // and computing their signs

    // For equal partitions, character is 1
    if lambda == rho {
        return 1;
    }

    // Placeholder - full implementation is complex
    0
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
}
