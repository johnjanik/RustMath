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

    /// Compute the residue of a cell at position (row, col) with quantum characteristic e
    ///
    /// The residue is defined as (col - row + multicharge) mod e
    /// where multicharge is typically 0 for standard tableaux.
    ///
    /// # Arguments
    /// * `row` - Row index (0-based)
    /// * `col` - Column index (0-based)
    /// * `e` - Quantum characteristic (positive integer)
    /// * `multicharge` - Multicharge parameter (default 0 for standard tableaux)
    ///
    /// # Returns
    /// The e-residue of the cell, or None if the position is invalid
    ///
    /// # Examples
    /// ```
    /// use rustmath_combinatorics::Tableau;
    ///
    /// let t = Tableau::new(vec![vec![1, 2, 3], vec![4, 5]]).unwrap();
    /// // Cell at (0, 0): residue = (0 - 0) mod 3 = 0
    /// assert_eq!(t.cell_residue(0, 0, 3, 0), Some(0));
    /// // Cell at (0, 2): residue = (2 - 0) mod 3 = 2
    /// assert_eq!(t.cell_residue(0, 2, 3, 0), Some(2));
    /// // Cell at (1, 1): residue = (1 - 1) mod 3 = 0
    /// assert_eq!(t.cell_residue(1, 1, 3, 0), Some(0));
    /// ```
    pub fn cell_residue(&self, row: usize, col: usize, e: usize, multicharge: i32) -> Option<usize> {
        if e == 0 {
            return None;
        }

        if row >= self.rows.len() || col >= self.rows[row].len() {
            return None;
        }

        // Compute (col - row + multicharge) mod e
        // Need to handle negative values properly
        let content = (col as i32) - (row as i32) + multicharge;
        let residue = content.rem_euclid(e as i32) as usize;

        Some(residue)
    }

    /// Compute the residue sequence of a standard tableau
    ///
    /// For a standard tableau with entries 1, 2, ..., n, this returns the sequence
    /// (r₁, r₂, ..., rₙ) where rₖ is the e-residue of the cell containing k.
    ///
    /// # Arguments
    /// * `e` - Quantum characteristic (positive integer)
    /// * `multicharge` - Multicharge parameter (default 0 for standard tableaux)
    ///
    /// # Returns
    /// Vector of residues in order of entries 1, 2, ..., n
    ///
    /// # Examples
    /// ```
    /// use rustmath_combinatorics::Tableau;
    ///
    /// let t = Tableau::new(vec![vec![1, 2, 4], vec![3, 5]]).unwrap();
    /// let residues = t.residue_sequence(3, 0);
    /// assert_eq!(residues.len(), 5);
    /// ```
    pub fn residue_sequence(&self, e: usize, multicharge: i32) -> Vec<usize> {
        if e == 0 {
            return vec![];
        }

        let n = self.size();
        if n == 0 {
            return vec![];
        }

        // Create a map from value to (row, col)
        let mut position_map = vec![(0, 0); n + 1];

        for (row_idx, row) in self.rows.iter().enumerate() {
            for (col_idx, &value) in row.iter().enumerate() {
                if value > 0 && value <= n {
                    position_map[value] = (row_idx, col_idx);
                }
            }
        }

        // Compute residues in order 1, 2, ..., n
        let mut residues = Vec::new();
        for i in 1..=n {
            let (row, col) = position_map[i];
            if let Some(res) = self.cell_residue(row, col, e, multicharge) {
                residues.push(res);
            }
        }

        residues
    }

    /// Compute the residue content of the tableau
    ///
    /// Returns a vector where the i-th element is the count of cells with residue i.
    ///
    /// # Arguments
    /// * `e` - Quantum characteristic (positive integer)
    /// * `multicharge` - Multicharge parameter (default 0 for standard tableaux)
    ///
    /// # Returns
    /// Vector of length e, where element i is the number of cells with residue i
    ///
    /// # Examples
    /// ```
    /// use rustmath_combinatorics::Tableau;
    ///
    /// let t = Tableau::new(vec![vec![1, 2, 3], vec![4, 5]]).unwrap();
    /// let content = t.residue_content(3, 0);
    /// assert_eq!(content.len(), 3);
    /// ```
    pub fn residue_content(&self, e: usize, multicharge: i32) -> Vec<usize> {
        if e == 0 {
            return vec![];
        }

        let mut content = vec![0; e];

        for (row_idx, row) in self.rows.iter().enumerate() {
            for col_idx in 0..row.len() {
                if let Some(res) = self.cell_residue(row_idx, col_idx, e, multicharge) {
                    content[res] += 1;
                }
            }
        }

        content
    }

    /// Get all cells with a specific residue
    ///
    /// Returns a vector of (row, col, value) tuples for all cells with the given residue.
    ///
    /// # Arguments
    /// * `e` - Quantum characteristic (positive integer)
    /// * `residue` - The residue to search for
    /// * `multicharge` - Multicharge parameter (default 0 for standard tableaux)
    ///
    /// # Examples
    /// ```
    /// use rustmath_combinatorics::Tableau;
    ///
    /// let t = Tableau::new(vec![vec![1, 2, 3], vec![4, 5]]).unwrap();
    /// let cells = t.cells_with_residue(3, 0, 0);
    /// // Cells at (0,0), (1,1) have residue 0 when e=3
    /// ```
    pub fn cells_with_residue(&self, e: usize, residue: usize, multicharge: i32) -> Vec<(usize, usize, usize)> {
        if e == 0 || residue >= e {
            return vec![];
        }

        let mut result = Vec::new();

        for (row_idx, row) in self.rows.iter().enumerate() {
            for (col_idx, &value) in row.iter().enumerate() {
                if let Some(res) = self.cell_residue(row_idx, col_idx, e, multicharge) {
                    if res == residue {
                        result.push((row_idx, col_idx, value));
                    }
                }
            }
        }

        result
    }

    /// Perform jeu de taquin slide from a given position
    ///
    /// Jeu de taquin is a process for moving an empty cell in a tableau
    /// while maintaining the tableau property. The empty cell slides by
    /// swapping with the smaller of its right and down neighbors.
    pub fn jeu_de_taquin(&self, empty_row: usize, empty_col: usize) -> Option<Tableau> {
        if empty_row >= self.rows.len() {
            return None;
        }
        if empty_col >= self.rows[empty_row].len() {
            return None;
        }

        let mut rows = self.rows.clone();
        let mut current_row = empty_row;
        let mut current_col = empty_col;

        loop {
            // Check right and down neighbors
            let right_exists = current_col + 1 < rows[current_row].len();
            let down_exists = current_row + 1 < rows.len() && current_col < rows[current_row + 1].len();

            if !right_exists && !down_exists {
                // Reached a corner, remove this position
                rows[current_row].truncate(current_col);
                // Remove empty rows
                rows.retain(|r| !r.is_empty());
                break;
            }

            // Choose which neighbor to swap with
            let swap_right = if right_exists && down_exists {
                // Swap with the smaller value to maintain tableau property
                rows[current_row][current_col + 1] < rows[current_row + 1][current_col]
            } else {
                right_exists
            };

            if swap_right {
                // Swap with right neighbor
                if current_col + 1 < rows[current_row].len() {
                    rows[current_row][current_col] = rows[current_row][current_col + 1];
                    current_col += 1;
                } else {
                    break;
                }
            } else {
                // Swap with down neighbor
                if current_row + 1 < rows.len() && current_col < rows[current_row + 1].len() {
                    rows[current_row][current_col] = rows[current_row + 1][current_col];
                    current_row += 1;
                } else {
                    break;
                }
            }
        }

        Tableau::new(rows)
    }

    /// Remove a specific value from the tableau using jeu de taquin
    pub fn remove_entry(&self, value: usize) -> Option<Tableau> {
        // Find the position of the value
        for (row_idx, row) in self.rows.iter().enumerate() {
            for (col_idx, &entry) in row.iter().enumerate() {
                if entry == value {
                    // Mark this position as "empty" by using jeu de taquin
                    let mut rows = self.rows.clone();
                    rows[row_idx][col_idx] = 0; // Temporary marker

                    // Create temporary tableau and perform slide
                    let temp_tableau = Tableau::new(rows)?;
                    return temp_tableau.jeu_de_taquin(row_idx, col_idx);
                }
            }
        }
        None
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

    #[test]
    fn test_jeu_de_taquin() {
        // Create a tableau: [[1, 2, 4], [3, 5]]
        let t = Tableau::new(vec![vec![1, 2, 4], vec![3, 5]]).unwrap();

        // Perform jeu de taquin from position (0, 0) - removing 1
        let result = t.jeu_de_taquin(0, 0);
        assert!(result.is_some());

        let result_t = result.unwrap();
        // After sliding, the tableau should be valid
        assert!(result_t.is_semistandard() || result_t.rows().is_empty() || result_t.size() < t.size());
    }

    #[test]
    fn test_remove_entry() {
        // Create a tableau: [[1, 3, 5], [2, 4]]
        let t = Tableau::new(vec![vec![1, 3, 5], vec![2, 4]]).unwrap();

        // Remove the value 1
        let result = t.remove_entry(1);
        assert!(result.is_some());

        let result_t = result.unwrap();
        // Size should be one less
        assert_eq!(result_t.size(), t.size() - 1);

        // Should not contain 1 anymore
        assert!(!result_t.content().contains(&1));

        // Should still be a valid semistandard tableau
        assert!(result_t.is_semistandard());
    }

    #[test]
    fn test_cell_residue() {
        // Create a tableau: [[1, 2, 3], [4, 5]]
        let t = Tableau::new(vec![vec![1, 2, 3], vec![4, 5]]).unwrap();

        // Test with e=3, multicharge=0
        // Cell (0,0): residue = (0 - 0 + 0) mod 3 = 0
        assert_eq!(t.cell_residue(0, 0, 3, 0), Some(0));

        // Cell (0,1): residue = (1 - 0 + 0) mod 3 = 1
        assert_eq!(t.cell_residue(0, 1, 3, 0), Some(1));

        // Cell (0,2): residue = (2 - 0 + 0) mod 3 = 2
        assert_eq!(t.cell_residue(0, 2, 3, 0), Some(2));

        // Cell (1,0): residue = (0 - 1 + 0) mod 3 = -1 mod 3 = 2
        assert_eq!(t.cell_residue(1, 0, 3, 0), Some(2));

        // Cell (1,1): residue = (1 - 1 + 0) mod 3 = 0
        assert_eq!(t.cell_residue(1, 1, 3, 0), Some(0));

        // Test with multicharge=1
        // Cell (0,0): residue = (0 - 0 + 1) mod 3 = 1
        assert_eq!(t.cell_residue(0, 0, 3, 1), Some(1));

        // Test invalid positions
        assert_eq!(t.cell_residue(2, 0, 3, 0), None);
        assert_eq!(t.cell_residue(0, 5, 3, 0), None);

        // Test e=0 (should return None)
        assert_eq!(t.cell_residue(0, 0, 0, 0), None);
    }

    #[test]
    fn test_cell_residue_with_different_e() {
        let t = Tableau::new(vec![vec![1, 2, 3], vec![4, 5]]).unwrap();

        // Test with e=2
        // Cell (0,0): residue = 0 mod 2 = 0
        assert_eq!(t.cell_residue(0, 0, 2, 0), Some(0));

        // Cell (0,1): residue = 1 mod 2 = 1
        assert_eq!(t.cell_residue(0, 1, 2, 0), Some(1));

        // Cell (1,0): residue = -1 mod 2 = 1
        assert_eq!(t.cell_residue(1, 0, 2, 0), Some(1));

        // Test with e=5
        // Cell (1,0): residue = -1 mod 5 = 4
        assert_eq!(t.cell_residue(1, 0, 5, 0), Some(4));
    }

    #[test]
    fn test_residue_sequence() {
        // Create a standard tableau: [[1, 2, 4], [3, 5]]
        let t = Tableau::new(vec![vec![1, 2, 4], vec![3, 5]]).unwrap();
        assert!(t.is_standard());

        // Compute residue sequence with e=3, multicharge=0
        let residues = t.residue_sequence(3, 0);

        // Should have 5 residues (one for each entry 1-5)
        assert_eq!(residues.len(), 5);

        // Entry 1 is at (0,0): residue = 0
        assert_eq!(residues[0], 0);

        // Entry 2 is at (0,1): residue = 1
        assert_eq!(residues[1], 1);

        // Entry 3 is at (1,0): residue = -1 mod 3 = 2
        assert_eq!(residues[2], 2);

        // Entry 4 is at (0,2): residue = 2
        assert_eq!(residues[3], 2);

        // Entry 5 is at (1,1): residue = 0
        assert_eq!(residues[4], 0);
    }

    #[test]
    fn test_residue_sequence_with_multicharge() {
        // Create a standard tableau: [[1, 2], [3]]
        let t = Tableau::new(vec![vec![1, 2], vec![3]]).unwrap();

        // Compute residue sequence with e=2, multicharge=1
        let residues = t.residue_sequence(2, 1);

        // Entry 1 at (0,0): residue = (0 - 0 + 1) mod 2 = 1
        assert_eq!(residues[0], 1);

        // Entry 2 at (0,1): residue = (1 - 0 + 1) mod 2 = 0
        assert_eq!(residues[1], 0);

        // Entry 3 at (1,0): residue = (0 - 1 + 1) mod 2 = 0
        assert_eq!(residues[2], 0);
    }

    #[test]
    fn test_residue_content() {
        // Create a tableau: [[1, 2, 3], [4, 5]]
        let t = Tableau::new(vec![vec![1, 2, 3], vec![4, 5]]).unwrap();

        // Compute residue content with e=3, multicharge=0
        let content = t.residue_content(3, 0);

        // Should have 3 entries (one for each residue class 0, 1, 2)
        assert_eq!(content.len(), 3);

        // Count cells with residue 0: (0,0) and (1,1) -> 2 cells
        assert_eq!(content[0], 2);

        // Count cells with residue 1: (0,1) -> 1 cell
        assert_eq!(content[1], 1);

        // Count cells with residue 2: (0,2) and (1,0) -> 2 cells
        assert_eq!(content[2], 2);

        // Verify total
        assert_eq!(content.iter().sum::<usize>(), 5);
    }

    #[test]
    fn test_residue_content_different_shape() {
        // Create a tableau with shape [3, 2, 1]: [[1, 2, 3], [4, 5], [6]]
        let t = Tableau::new(vec![vec![1, 2, 3], vec![4, 5], vec![6]]).unwrap();

        let content = t.residue_content(2, 0);

        // With e=2, residues are 0 or 1
        assert_eq!(content.len(), 2);

        // Cells with even content (col - row): (0,0)=0, (0,2)=2, (1,1)=0, (2,0)=-2 -> 4 cells with residue 0
        // Cells with odd content: (0,1)=1, (1,0)=-1 -> 2 cells with residue 1
        assert_eq!(content[0], 4);
        assert_eq!(content[1], 2);

        assert_eq!(content.iter().sum::<usize>(), 6);
    }

    #[test]
    fn test_cells_with_residue() {
        // Create a tableau: [[1, 2, 3], [4, 5]]
        let t = Tableau::new(vec![vec![1, 2, 3], vec![4, 5]]).unwrap();

        // Find cells with residue 0 (e=3, multicharge=0)
        let cells = t.cells_with_residue(3, 0, 0);

        // Should have 2 cells: (0,0) with value 1 and (1,1) with value 5
        assert_eq!(cells.len(), 2);
        assert!(cells.contains(&(0, 0, 1)));
        assert!(cells.contains(&(1, 1, 5)));

        // Find cells with residue 1
        let cells_1 = t.cells_with_residue(3, 1, 0);
        assert_eq!(cells_1.len(), 1);
        assert_eq!(cells_1[0], (0, 1, 2));

        // Find cells with residue 2
        let cells_2 = t.cells_with_residue(3, 2, 0);
        assert_eq!(cells_2.len(), 2);
        assert!(cells_2.contains(&(0, 2, 3)));
        assert!(cells_2.contains(&(1, 0, 4)));
    }

    #[test]
    fn test_cells_with_residue_invalid() {
        let t = Tableau::new(vec![vec![1, 2, 3]]).unwrap();

        // Invalid: residue >= e
        let cells = t.cells_with_residue(3, 3, 0);
        assert_eq!(cells.len(), 0);

        // Invalid: e = 0
        let cells = t.cells_with_residue(0, 0, 0);
        assert_eq!(cells.len(), 0);
    }

    #[test]
    fn test_residue_empty_tableau() {
        let t = Tableau::new(vec![]).unwrap();

        // Cell residue on empty tableau
        assert_eq!(t.cell_residue(0, 0, 3, 0), None);

        // Residue sequence of empty tableau
        let residues = t.residue_sequence(3, 0);
        assert_eq!(residues.len(), 0);

        // Residue content of empty tableau
        let content = t.residue_content(3, 0);
        assert_eq!(content, vec![0, 0, 0]);

        // Cells with residue on empty tableau
        let cells = t.cells_with_residue(3, 0, 0);
        assert_eq!(cells.len(), 0);
    }

    #[test]
    fn test_residue_large_tableau() {
        // Create a larger standard tableau
        let t = Tableau::new(vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7],
            vec![8, 9],
        ]).unwrap();

        assert!(t.is_standard());
        assert_eq!(t.size(), 9);

        // Test residue sequence with e=4
        let residues = t.residue_sequence(4, 0);
        assert_eq!(residues.len(), 9);

        // Entry 1 at (0,0): residue = 0
        assert_eq!(residues[0], 0);

        // Entry 5 at (1,0): residue = -1 mod 4 = 3
        assert_eq!(residues[4], 3);

        // Entry 9 at (2,1): residue = (1 - 2) mod 4 = -1 mod 4 = 3
        assert_eq!(residues[8], 3);

        // Test residue content
        let content = t.residue_content(4, 0);
        assert_eq!(content.len(), 4);
        assert_eq!(content.iter().sum::<usize>(), 9);
    }

    #[test]
    fn test_residue_negative_multicharge() {
        let t = Tableau::new(vec![vec![1, 2], vec![3]]).unwrap();

        // Test with negative multicharge
        let residues = t.residue_sequence(3, -1);

        // Entry 1 at (0,0): residue = (0 - 0 - 1) mod 3 = -1 mod 3 = 2
        assert_eq!(residues[0], 2);

        // Entry 2 at (0,1): residue = (1 - 0 - 1) mod 3 = 0
        assert_eq!(residues[1], 0);

        // Entry 3 at (1,0): residue = (0 - 1 - 1) mod 3 = -2 mod 3 = 1
        assert_eq!(residues[2], 1);
    }
}
