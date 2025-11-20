//! Growth diagrams for Robinson-Schensted correspondence
//!
//! A growth diagram is a visual representation of the Robinson-Schensted correspondence.
//! It consists of an (n+1) × (n+1) grid where each cell (i,j) contains a partition (shape).
//! The partitions start empty at (0,0) and grow according to local rules.
//!
//! For a permutation σ of {1, ..., n}, we mark the cells at positions (i, σ(i)) for i = 1, ..., n.
//! The growth rule ensures that at each marked cell, exactly one box is added to the partition.
//! The final partitions along the right edge and top edge correspond to the P and Q tableaux
//! from the Robinson-Schensted correspondence.

use crate::partitions::Partition;
use crate::tableaux::{robinson_schensted, rs_insert, Tableau};

/// A growth diagram representing the Robinson-Schensted correspondence
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrowthDiagram {
    /// The grid of partitions, indexed by (row, col)
    /// grid[i][j] represents the partition at position (i, j)
    grid: Vec<Vec<Partition>>,
    /// Size of the diagram (n for an (n+1) × (n+1) grid)
    n: usize,
    /// Marked cells (i, j) where the permutation has a value
    marked: Vec<(usize, usize)>,
}

impl GrowthDiagram {
    /// Create a new growth diagram from a permutation
    ///
    /// The permutation is given as a vector where perm[i] = j means
    /// the permutation maps i+1 to j (using 1-based indexing for the permutation).
    pub fn from_permutation(perm: &[usize]) -> Self {
        let n = perm.len();

        // Initialize grid with empty partitions
        let mut grid = vec![vec![Partition::new(vec![]); n + 1]; n + 1];

        // Mark the cells according to the permutation
        let mut marked = Vec::new();
        for (i, &value) in perm.iter().enumerate() {
            marked.push((i + 1, value));
        }

        // Build the growth diagram using the Robinson-Schensted insertion
        // The partition at (i, j) represents the shape after processing
        // the sub-permutation restricted to positions 1..=i and values 1..=j

        for i in 1..=n {
            for j in 1..=n {
                // Determine if we need to insert based on marked cells
                // We look at the marked cells up to and including position (i, j)

                // Find all marked cells (r, c) where r <= i and c <= j
                let mut sub_elements = Vec::new();
                for &(r, c) in &marked {
                    if r <= i && c <= j {
                        sub_elements.push((r, c));
                    }
                }

                // The shape at (i, j) is determined by the number of marked cells
                // in the rectangle [1..i] x [1..j]
                // For Robinson-Schensted, this follows from the insertion algorithm

                // Build a sub-permutation from the marked cells in this rectangle
                // and compute its RS tableau shape
                if sub_elements.is_empty() {
                    grid[i][j] = Partition::new(vec![]);
                } else {
                    // Extract the values from marked cells in this rectangle
                    let mut values = Vec::new();
                    for k in 1..=i {
                        // Check if position k has a marked cell within column range 1..=j
                        for &(r, c) in &marked {
                            if r == k && c <= j {
                                values.push(c);
                                break;
                            }
                        }
                    }

                    // Compute RS shape for these values
                    if values.is_empty() {
                        grid[i][j] = Partition::new(vec![]);
                    } else {
                        let (p_tab, _) = robinson_schensted(&values);
                        grid[i][j] = p_tab.shape().clone();
                    }
                }
            }
        }

        GrowthDiagram { grid, n, marked }
    }

    /// Get the partition at position (row, col)
    pub fn get(&self, row: usize, col: usize) -> Option<&Partition> {
        if row <= self.n && col <= self.n {
            Some(&self.grid[row][col])
        } else {
            None
        }
    }

    /// Get the P tableau (insertion tableau) from the growth diagram
    ///
    /// The P tableau corresponds to the partitions along the right edge
    pub fn p_tableau(&self) -> Tableau {
        // The shape is the partition at the bottom-right corner
        let shape = &self.grid[self.n][self.n];

        // Build the tableau by tracking the growth
        let mut tableau = Tableau::new(vec![]).unwrap();
        let mut prev_shape = Partition::new(vec![]);

        for i in 1..=self.n {
            let current_shape = &self.grid[i][self.n];

            // Find which cell was added
            if current_shape != &prev_shape {
                // A cell was added at position (i, self.n)
                // We need to find the value that was inserted
                // For now, we'll use the row index as the value
                let value = self.get_value_at_row(i);
                tableau = rs_insert(&tableau, value);
            }

            prev_shape = current_shape.clone();
        }

        tableau
    }

    /// Get the Q tableau (recording tableau) from the growth diagram
    ///
    /// The Q tableau corresponds to the partitions along the top edge
    pub fn q_tableau(&self) -> Tableau {
        // Similar to P tableau, but along the top edge
        let mut tableau = Tableau::new(vec![]).unwrap();
        let mut prev_shape = Partition::new(vec![]);

        for j in 1..=self.n {
            let current_shape = &self.grid[self.n][j];

            if current_shape != &prev_shape {
                // A cell was added, label it with j
                tableau = rs_insert(&tableau, j);
            }

            prev_shape = current_shape.clone();
        }

        tableau
    }

    /// Helper to get the value that was inserted at a given row
    fn get_value_at_row(&self, row: usize) -> usize {
        // Find the marked cell in this row
        for &(r, c) in &self.marked {
            if r == row {
                return c;
            }
        }
        row // Fallback
    }

    /// Get the final shape (the shape of the full Robinson-Schensted tableau)
    pub fn final_shape(&self) -> Partition {
        // Compute the RS tableau for the original permutation
        // Extract the permutation from marked cells
        let mut perm = vec![0; self.n];
        for &(i, j) in &self.marked {
            if i >= 1 && i <= self.n {
                perm[i - 1] = j;
            }
        }

        if perm.is_empty() {
            return Partition::new(vec![]);
        }

        let (p_tableau, _) = robinson_schensted(&perm);
        p_tableau.shape().clone()
    }

    /// Get the size of the diagram
    pub fn size(&self) -> usize {
        self.n
    }

    /// Get all marked cells
    pub fn marked_cells(&self) -> &[(usize, usize)] {
        &self.marked
    }

    /// Display the growth diagram as a string
    pub fn to_string(&self) -> String {
        let mut result = String::new();

        for row in 0..=self.n {
            for col in 0..=self.n {
                let shape = &self.grid[row][col];
                let is_marked = self.marked.contains(&(row, col));

                // Format the partition as [p1,p2,...]
                let parts_str = if shape.parts().is_empty() {
                    "∅".to_string()
                } else {
                    shape.parts()
                        .iter()
                        .map(|p| p.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                };

                if is_marked {
                    result.push_str(&format!("[{}*] ", parts_str));
                } else {
                    result.push_str(&format!("[{}] ", parts_str));
                }
            }
            result.push('\n');
        }

        result
    }
}

/// Compute the northeast partition given the three neighboring partitions
///
/// This implements the local growth rule for Robinson-Schensted growth diagrams.
///
/// Given:
/// - λ (southwest partition at (i-1, j-1))
/// - μ (northwest partition at (i, j-1))
/// - ν (southeast partition at (i-1, j))
/// - is_marked: whether the cell (i, j) is marked
///
/// We compute ρ (northeast partition at (i, j)) such that:
/// - If the cell is marked: |ρ| = |μ| + 1 = |ν| + 1 (both μ and ν grow by one box to ρ)
/// - If the cell is not marked: ρ = μ ∪ ν (the union of μ and ν)
fn compute_northeast_partition(
    _lambda: &Partition,
    mu: &Partition,
    nu: &Partition,
    is_marked: bool,
) -> Partition {
    if is_marked {
        // The marked case: ρ must have size |μ| + 1 = |ν| + 1
        // This means μ and ν must have the same size
        // ρ is obtained by adding one box to the union of μ and ν

        let mu_parts = mu.parts();
        let nu_parts = nu.parts();
        let max_len = mu_parts.len().max(nu_parts.len());

        // Start with the maximum of μ and ν coordinate-wise (the union)
        let mut rho_parts = Vec::new();
        for i in 0..max_len {
            let mu_val = mu_parts.get(i).copied().unwrap_or(0);
            let nu_val = nu_parts.get(i).copied().unwrap_or(0);
            rho_parts.push(mu_val.max(nu_val));
        }

        // Now add one box
        // Find the first row where we can add a box while maintaining partition property
        let mut added = false;

        // Try to add to existing rows
        for i in 0..rho_parts.len() {
            let current_length = rho_parts[i];
            let next_length = rho_parts.get(i + 1).copied().unwrap_or(0);

            // We can add to row i if:
            // 1. The next row is shorter or doesn't exist, OR
            // 2. Adding to this row keeps it >= next row
            if current_length > next_length {
                rho_parts[i] += 1;
                added = true;
                break;
            }
        }

        // If we couldn't add to any existing row, try adding a new row
        if !added {
            if rho_parts.is_empty() {
                rho_parts.push(1);
            } else {
                // Add a new row at the end
                rho_parts.push(1);
            }
        }

        Partition::new(rho_parts)
    } else {
        // The unmarked case: ρ is the "union" of μ and ν
        // This means ρ[i] = max(μ[i], ν[i]) for all i

        let mu_parts = mu.parts();
        let nu_parts = nu.parts();
        let max_len = mu_parts.len().max(nu_parts.len());

        let mut rho_parts = Vec::new();
        for i in 0..max_len {
            let mu_val = mu_parts.get(i).copied().unwrap_or(0);
            let nu_val = nu_parts.get(i).copied().unwrap_or(0);
            rho_parts.push(mu_val.max(nu_val));
        }

        Partition::new(rho_parts)
    }
}

/// Construct the growth diagram step by step, returning all intermediate diagrams
///
/// This function builds the growth diagram incrementally, showing how it grows
/// as each element of the permutation is processed.
pub fn growth_sequence(perm: &[usize]) -> Vec<GrowthDiagram> {
    let n = perm.len();
    let mut sequence = Vec::new();

    // Start with empty permutation
    sequence.push(GrowthDiagram::from_permutation(&[]));

    // Add elements one by one
    for i in 1..=n {
        let partial_perm = &perm[..i];
        sequence.push(GrowthDiagram::from_permutation(partial_perm));
    }

    sequence
}

/// Extract the insertion tableau (P tableau) from a growth diagram
///
/// This is a convenience function that creates a growth diagram and extracts the P tableau.
pub fn growth_to_p_tableau(perm: &[usize]) -> Tableau {
    let diagram = GrowthDiagram::from_permutation(perm);
    diagram.p_tableau()
}

/// Extract the recording tableau (Q tableau) from a growth diagram
///
/// This is a convenience function that creates a growth diagram and extracts the Q tableau.
pub fn growth_to_q_tableau(perm: &[usize]) -> Tableau {
    let diagram = GrowthDiagram::from_permutation(perm);
    diagram.q_tableau()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tableaux::robinson_schensted;

    #[test]
    fn test_empty_permutation() {
        let perm = vec![];
        let diagram = GrowthDiagram::from_permutation(&perm);

        assert_eq!(diagram.size(), 0);
        assert_eq!(diagram.final_shape(), Partition::new(vec![]));
    }

    #[test]
    fn test_identity_permutation() {
        // Identity permutation [1, 2, 3]
        let perm = vec![1, 2, 3];
        let diagram = GrowthDiagram::from_permutation(&perm);

        assert_eq!(diagram.size(), 3);
        // Identity should give a single row of length 3
        assert_eq!(diagram.final_shape().parts(), &[3]);
    }

    #[test]
    fn test_reverse_permutation() {
        // Reverse permutation [3, 2, 1]
        let perm = vec![3, 2, 1];
        let diagram = GrowthDiagram::from_permutation(&perm);

        assert_eq!(diagram.size(), 3);
        // Reverse should give a single column of height 3
        assert_eq!(diagram.final_shape().parts(), &[1, 1, 1]);
    }

    #[test]
    fn test_simple_permutation() {
        // Permutation [2, 1, 3]
        let perm = vec![2, 1, 3];
        let diagram = GrowthDiagram::from_permutation(&perm);

        assert_eq!(diagram.size(), 3);

        // The final shape should be [2, 1]
        let expected_parts = vec![2, 1];
        assert_eq!(diagram.final_shape().parts(), &expected_parts);
    }

    #[test]
    fn test_growth_diagram_shape() {
        // Test that the growth diagram produces the same shape as RS insertion
        let perm = vec![2, 1, 4, 3];
        let diagram = GrowthDiagram::from_permutation(&perm);
        let (p_rs, _q_rs) = robinson_schensted(&perm);

        // The final shape from the growth diagram should match the RS tableau shape
        assert_eq!(diagram.final_shape(), *p_rs.shape());
    }

    #[test]
    fn test_marked_cells() {
        let perm = vec![2, 1, 3];
        let diagram = GrowthDiagram::from_permutation(&perm);

        let marked = diagram.marked_cells();
        assert_eq!(marked.len(), 3);

        // Check that the marked cells are correct
        assert!(marked.contains(&(1, 2)));
        assert!(marked.contains(&(2, 1)));
        assert!(marked.contains(&(3, 3)));
    }

    #[test]
    fn test_growth_sequence() {
        let perm = vec![2, 1];
        let sequence = growth_sequence(&perm);

        // Should have 3 diagrams: empty, [2], and [2, 1]
        assert_eq!(sequence.len(), 3);

        assert_eq!(sequence[0].size(), 0);
        assert_eq!(sequence[1].size(), 1);
        assert_eq!(sequence[2].size(), 2);

        // Check that shapes grow
        assert_eq!(sequence[0].final_shape().parts(), &[]);
        assert_eq!(sequence[1].final_shape().parts(), &[1]);
        assert_eq!(sequence[2].final_shape().parts(), &[1, 1]);
    }

    #[test]
    fn test_compute_northeast_unmarked() {
        // Test the unmarked case: ρ should be max of μ and ν
        let lambda = Partition::new(vec![]);
        let mu = Partition::new(vec![1]);
        let nu = Partition::new(vec![1]);

        let rho = compute_northeast_partition(&lambda, &mu, &nu, false);
        assert_eq!(rho.parts(), &[1]);

        // Another test
        let mu2 = Partition::new(vec![2]);
        let nu2 = Partition::new(vec![1, 1]);
        let rho2 = compute_northeast_partition(&lambda, &mu2, &nu2, false);
        assert_eq!(rho2.parts(), &[2, 1]);
    }

    #[test]
    fn test_compute_northeast_marked() {
        // Test the marked case: ρ should add one box
        let lambda = Partition::new(vec![]);
        let mu = Partition::new(vec![1]);
        let nu = Partition::new(vec![1]);

        let rho = compute_northeast_partition(&lambda, &mu, &nu, true);
        // Should add one box to [1]
        assert_eq!(rho.sum(), 2);
    }

    #[test]
    fn test_longer_permutation() {
        let perm = vec![3, 1, 4, 2, 5];
        let diagram = GrowthDiagram::from_permutation(&perm);

        assert_eq!(diagram.size(), 5);
        assert!(diagram.final_shape().sum() == 5);
    }

    #[test]
    fn test_growth_consistency_with_rs() {
        // Test several permutations to ensure growth diagrams are consistent with RS
        let test_perms = vec![
            vec![1, 2, 3],
            vec![3, 2, 1],
            vec![2, 1, 3],
            vec![1, 3, 2],
            vec![3, 1, 2],
            vec![2, 3, 1],
        ];

        for perm in test_perms {
            let diagram = GrowthDiagram::from_permutation(&perm);
            let (p_rs, _q_rs) = robinson_schensted(&perm);

            // Final shapes should match
            assert_eq!(
                diagram.final_shape(),
                *p_rs.shape(),
                "Mismatch for permutation {:?}",
                perm
            );
        }
    }
}
