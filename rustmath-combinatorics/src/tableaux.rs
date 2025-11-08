//! Young tableaux and tableau algorithms

use crate::partitions::Partition;

/// A Young tableau - a filling of a Young diagram with entries
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tableau {
    /// The entries of the tableau, organized by rows
    rows: Vec<Vec<usize>>,
    /// The shape (partition) of the tableau
    shape: Partition,
}

impl Tableau {
    /// Create a tableau from rows
    ///
    /// Returns None if the rows don't form a valid tableau shape
    pub fn new(rows: Vec<Vec<usize>>) -> Option<Self> {
        if rows.is_empty() {
            return Some(Tableau {
                rows: vec![],
                shape: Partition::new(vec![]),
            });
        }

        // Check that rows are non-increasing in length
        for i in 1..rows.len() {
            if rows[i].len() > rows[i - 1].len() {
                return None;
            }
        }

        // Create the shape
        let shape_parts: Vec<usize> = rows.iter().map(|row| row.len()).collect();
        let shape = Partition::new(shape_parts);

        Some(Tableau { rows, shape })
    }

    /// Get the shape of the tableau
    pub fn shape(&self) -> &Partition {
        &self.shape
    }

    /// Get the rows of the tableau
    pub fn rows(&self) -> &[Vec<usize>] {
        &self.rows
    }

    /// Get the number of entries in the tableau
    pub fn size(&self) -> usize {
        self.rows.iter().map(|row| row.len()).sum()
    }

    /// Check if this is a standard tableau
    ///
    /// A standard tableau has entries 1,2,...,n that are strictly increasing
    /// along rows and down columns
    pub fn is_standard(&self) -> bool {
        let n = self.size();
        if n == 0 {
            return true;
        }

        // Check that we have exactly the numbers 1..=n
        let mut seen = vec![false; n + 1];
        for row in &self.rows {
            for &entry in row {
                if entry == 0 || entry > n || seen[entry] {
                    return false;
                }
                seen[entry] = true;
            }
        }

        // Check rows are strictly increasing
        for row in &self.rows {
            for i in 1..row.len() {
                if row[i] <= row[i - 1] {
                    return false;
                }
            }
        }

        // Check columns are strictly increasing
        for col in 0..self.rows[0].len() {
            for row in 1..self.rows.len() {
                if col < self.rows[row].len() {
                    if self.rows[row][col] <= self.rows[row - 1][col] {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Check if this is a semistandard tableau
    ///
    /// Rows are weakly increasing, columns are strictly increasing
    pub fn is_semistandard(&self) -> bool {
        // Check rows are weakly increasing (non-decreasing)
        for row in &self.rows {
            for i in 1..row.len() {
                if row[i] < row[i - 1] {
                    return false;
                }
            }
        }

        // Check columns are strictly increasing
        for col in 0..self.rows[0].len() {
            for row in 1..self.rows.len() {
                if col < self.rows[row].len() {
                    if self.rows[row][col] <= self.rows[row - 1][col] {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Get the content (multiset of entries) of the tableau
    pub fn content(&self) -> Vec<usize> {
        let mut entries: Vec<usize> = self.rows.iter().flat_map(|row| row.iter().copied()).collect();
        entries.sort_unstable();
        entries
    }

    /// Get the reading word (row reading from bottom to top)
    pub fn reading_word(&self) -> Vec<usize> {
        self.rows
            .iter()
            .rev()
            .flat_map(|row| row.iter().copied())
            .collect()
    }

    /// Get entry at position (row, col), if it exists
    pub fn get(&self, row: usize, col: usize) -> Option<usize> {
        self.rows.get(row)?.get(col).copied()
    }

    /// Get the number of rows
    pub fn num_rows(&self) -> usize {
        self.rows.len()
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
}

/// Generate all standard Young tableaux of a given shape
///
/// Uses recursive backtracking to fill the tableau with 1,2,...,n
pub fn standard_tableaux(shape: &Partition) -> Vec<Tableau> {
    let n = shape.sum();
    if n == 0 {
        return vec![Tableau {
            rows: vec![],
            shape: shape.clone(),
        }];
    }

    let mut result = Vec::new();
    let mut current: Vec<Vec<usize>> = shape
        .parts()
        .iter()
        .map(|&len| vec![0; len])
        .collect();

    generate_standard_tableaux(
        &mut current,
        &shape,
        1,
        n,
        &mut result,
    );

    result
}

fn generate_standard_tableaux(
    current: &mut Vec<Vec<usize>>,
    shape: &Partition,
    next_value: usize,
    n: usize,
    result: &mut Vec<Tableau>,
) {
    if next_value > n {
        // All values placed, add this tableau
        result.push(Tableau {
            rows: current.clone(),
            shape: shape.clone(),
        });
        return;
    }

    // Try placing next_value in each valid position
    for r in 0..shape.length() {
        for c in 0..shape.parts()[r] {
            if current[r][c] == 0 && can_place(current, r, c, next_value) {
                current[r][c] = next_value;
                generate_standard_tableaux(current, shape, next_value + 1, n, result);
                current[r][c] = 0;
            }
        }
    }
}

fn can_place(current: &[Vec<usize>], row: usize, col: usize, value: usize) -> bool {
    // Check that this position hasn't been filled
    if current[row][col] != 0 {
        return false;
    }

    // Check that all positions to the left are filled and smaller
    if col > 0 {
        if current[row][col - 1] == 0 || current[row][col - 1] >= value {
            return false;
        }
    }

    // Check that all positions above are filled and smaller
    if row > 0 {
        if current[row - 1][col] == 0 || current[row - 1][col] >= value {
            return false;
        }
    }

    true
}

/// Robinson-Schensted insertion
///
/// Insert a value into a tableau using the bumping algorithm
pub fn rs_insert(tableau: &Tableau, value: usize) -> Tableau {
    let mut rows = tableau.rows.clone();
    let mut current_value = value;

    for row_idx in 0..rows.len() {
        // Find position to insert in this row
        match rows[row_idx].iter().position(|&x| x > current_value) {
            Some(pos) => {
                // Bump the value at this position
                let bumped = rows[row_idx][pos];
                rows[row_idx][pos] = current_value;
                current_value = bumped;
            }
            None => {
                // Append to end of this row
                rows[row_idx].push(current_value);
                return Tableau::new(rows).unwrap();
            }
        }
    }

    // Create a new row with the bumped value
    rows.push(vec![current_value]);
    Tableau::new(rows).unwrap()
}

/// Robinson-Schensted correspondence
///
/// Convert a permutation to a pair of standard tableaux (P, Q)
pub fn robinson_schensted(permutation: &[usize]) -> (Tableau, Tableau) {
    let mut p_tableau = Tableau::new(vec![]).unwrap();
    let mut q_tableau = Tableau::new(vec![]).unwrap();

    for (i, &value) in permutation.iter().enumerate() {
        // Insert value into P tableau
        let old_p = p_tableau.clone();
        p_tableau = rs_insert(&p_tableau, value);

        // Record insertion position in Q tableau with i+1
        let insertion_label = i + 1;

        // Find where the new cell was added
        let new_cell_pos = find_new_cell(&old_p, &p_tableau);

        // Add corresponding label to Q tableau at same position
        q_tableau = insert_at_position(&q_tableau, new_cell_pos.0, new_cell_pos.1, insertion_label);
    }

    (p_tableau, q_tableau)
}

fn find_new_cell(old_tableau: &Tableau, new_tableau: &Tableau) -> (usize, usize) {
    // Find the position where a new cell was added
    for (row_idx, row) in new_tableau.rows().iter().enumerate() {
        if row_idx >= old_tableau.num_rows() {
            return (row_idx, 0);
        }
        if row.len() > old_tableau.rows()[row_idx].len() {
            return (row_idx, old_tableau.rows()[row_idx].len());
        }
    }
    (0, 0) // Shouldn't happen
}

fn insert_at_position(tableau: &Tableau, row: usize, col: usize, value: usize) -> Tableau {
    let mut rows = tableau.rows.clone();

    // Extend rows vector if needed
    while rows.len() <= row {
        rows.push(vec![]);
    }

    // Extend the specific row if needed
    while rows[row].len() <= col {
        rows[row].push(0);
    }

    rows[row][col] = value;
    Tableau::new(rows).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tableau_creation() {
        let t = Tableau::new(vec![vec![1, 2, 3], vec![4, 5]]).unwrap();
        assert_eq!(t.size(), 5);
        assert_eq!(t.shape().parts(), &[3, 2]);
    }

    #[test]
    fn test_standard_tableau() {
        // Valid standard tableau
        let t = Tableau::new(vec![vec![1, 2, 4], vec![3, 5]]).unwrap();
        assert!(t.is_standard());

        // Invalid - not increasing in row
        let t2 = Tableau::new(vec![vec![1, 3, 2], vec![4, 5]]).unwrap();
        assert!(!t2.is_standard());

        // Invalid - not increasing in column
        let t3 = Tableau::new(vec![vec![1, 2, 3], vec![2, 4]]).unwrap();
        assert!(!t3.is_standard());
    }

    #[test]
    fn test_semistandard_tableau() {
        // Valid semistandard (weakly increasing rows)
        let t = Tableau::new(vec![vec![1, 1, 2], vec![2, 3]]).unwrap();
        assert!(t.is_semistandard());
        assert!(!t.is_standard()); // Not standard due to repeated 1s

        // Invalid - decreasing in row
        let t2 = Tableau::new(vec![vec![2, 1, 3], vec![4, 5]]).unwrap();
        assert!(!t2.is_semistandard());
    }

    #[test]
    fn test_reading_word() {
        let t = Tableau::new(vec![vec![1, 2, 3], vec![4, 5]]).unwrap();
        assert_eq!(t.reading_word(), vec![4, 5, 1, 2, 3]);
    }

    #[test]
    fn test_content() {
        let t = Tableau::new(vec![vec![1, 3, 5], vec![2, 4]]).unwrap();
        assert_eq!(t.content(), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_generate_standard_tableaux() {
        // Shape [2, 1] should have 2 standard tableaux
        let shape = Partition::new(vec![2, 1]);
        let tableaux = standard_tableaux(&shape);

        assert_eq!(tableaux.len(), 2);
        for t in &tableaux {
            assert!(t.is_standard());
            assert_eq!(t.shape(), &shape);
        }
    }

    #[test]
    fn test_rs_insert() {
        // Start with empty tableau
        let t = Tableau::new(vec![]).unwrap();

        // Insert 2
        let t = rs_insert(&t, 2);
        assert_eq!(t.rows(), &[vec![2]]);

        // Insert 1 (should bump the 2, creating new row)
        let t = rs_insert(&t, 1);
        assert_eq!(t.rows(), &[vec![1], vec![2]]);

        // Insert 3 (should append to first row)
        let t = rs_insert(&t, 3);
        assert_eq!(t.rows(), &[vec![1, 3], vec![2]]);

        // Insert 2 (should bump 3, 3 appends to second row)
        let t = rs_insert(&t, 2);
        // After inserting 2: 2 bumps 3 in first row, giving [1,2]
        // Then 3 goes to second row [2], and since 3 > 2, it appends giving [2, 3]
        assert_eq!(t.rows(), &[vec![1, 2], vec![2, 3]]);
    }

    #[test]
    fn test_robinson_schensted() {
        // Simple permutation [2, 1, 3]
        let perm = vec![2, 1, 3];
        let (p, q) = robinson_schensted(&perm);

        // Both should be standard tableaux
        assert!(p.is_standard());
        assert!(q.is_standard());

        // Both should have the same shape
        assert_eq!(p.shape(), q.shape());

        // Both should have size 3
        assert_eq!(p.size(), 3);
        assert_eq!(q.size(), 3);
    }

    #[test]
    fn test_robinson_schensted_identity() {
        // Identity permutation [1, 2, 3] should give single row
        let perm = vec![1, 2, 3];
        let (p, q) = robinson_schensted(&perm);

        assert_eq!(p.num_rows(), 1);
        assert_eq!(q.num_rows(), 1);
        assert_eq!(p.size(), 3);
    }
}
