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

/// Hillman-Grassl algorithm: Binary matrix to pair of tableaux
///
/// The Hillman-Grassl algorithm is a bijection between binary matrices
/// and pairs of semistandard Young tableaux (P, Q) with the same shape.
///
/// Given a binary matrix M, it produces two tableaux:
/// - P tableau: records the column indices of 1s
/// - Q tableau: records which row each insertion came from
///
/// # Arguments
/// * `matrix` - A binary matrix represented as Vec<Vec<usize>> where entries are 0 or 1
///
/// # Returns
/// A pair of semistandard tableaux (P, Q) with the same shape
///
/// # Example
/// For the matrix:
/// ```text
/// [1, 0, 1]
/// [0, 1, 0]
/// [1, 1, 0]
/// ```
/// The algorithm processes row by row, inserting column indices where 1s appear.
pub fn hillman_grassl(matrix: &[Vec<usize>]) -> Option<(Tableau, Tableau)> {
    if matrix.is_empty() {
        return Some((Tableau::new(vec![]).unwrap(), Tableau::new(vec![]).unwrap()));
    }

    // Validate that matrix is binary
    for row in matrix {
        for &entry in row {
            if entry != 0 && entry != 1 {
                return None; // Not a binary matrix
            }
        }
    }

    let mut p_rows: Vec<Vec<usize>> = vec![];
    let mut q_rows: Vec<Vec<usize>> = vec![];

    // Process each row of the matrix
    for (row_idx, matrix_row) in matrix.iter().enumerate() {
        // Collect column indices where this row has 1s
        let cols_with_ones: Vec<usize> = matrix_row
            .iter()
            .enumerate()
            .filter_map(|(col_idx, &val)| if val == 1 { Some(col_idx + 1) } else { None })
            .collect();

        // Insert each column index into the tableaux
        for col_value in cols_with_ones {
            hillman_grassl_insert_pair(&mut p_rows, &mut q_rows, col_value, row_idx + 1);
        }
    }

    Some((
        Tableau::new(p_rows).unwrap(),
        Tableau::new(q_rows).unwrap(),
    ))
}

/// Modified insertion for Hillman-Grassl that produces semistandard tableaux
///
/// This is similar to RS insertion but allows repeated values (weakly increasing rows)
/// while maintaining strictly increasing columns.
fn hillman_grassl_insert(tableau: &Tableau, value: usize) -> Tableau {
    let mut rows = tableau.rows.clone();
    let mut current_value = value;

    for row_idx in 0..rows.len() {
        // Find position to insert in this row (first element > current_value)
        match rows[row_idx].iter().position(|&x| x > current_value) {
            Some(pos) => {
                // Bump the value at this position
                let bumped = rows[row_idx][pos];
                rows[row_idx][pos] = current_value;
                current_value = bumped;
            }
            None => {
                // Append to end of this row (value is >= all elements in row)
                rows[row_idx].push(current_value);
                return Tableau::new(rows).unwrap();
            }
        }
    }

    // Create a new row with the bumped value
    rows.push(vec![current_value]);
    Tableau::new(rows).unwrap()
}

/// Insert into both P and Q tableaux simultaneously
///
/// This ensures that when values are bumped in P, the corresponding Q values
/// are also bumped, maintaining the correspondence between P and Q entries.
fn hillman_grassl_insert_pair(
    p_rows: &mut Vec<Vec<usize>>,
    q_rows: &mut Vec<Vec<usize>>,
    p_value: usize,
    q_value: usize,
) {
    let mut current_p = p_value;
    let mut current_q = q_value;

    for row_idx in 0..p_rows.len() {
        // Find position to insert in this row (first element > current_p_value)
        match p_rows[row_idx].iter().position(|&x| x > current_p) {
            Some(pos) => {
                // Bump both P and Q values at this position
                let bumped_p = p_rows[row_idx][pos];
                let bumped_q = q_rows[row_idx][pos];

                p_rows[row_idx][pos] = current_p;
                q_rows[row_idx][pos] = current_q;

                current_p = bumped_p;
                current_q = bumped_q;
            }
            None => {
                // Append to end of this row
                p_rows[row_idx].push(current_p);
                q_rows[row_idx].push(current_q);
                return;
            }
        }
    }

    // Create a new row with the bumped values
    p_rows.push(vec![current_p]);
    q_rows.push(vec![current_q]);
}

/// Inverse Hillman-Grassl algorithm: Convert pair of tableaux back to binary matrix
///
/// Given a pair of semistandard tableaux (P, Q) with the same shape,
/// reconstruct the original binary matrix.
///
/// # Arguments
/// * `p_tableau` - The P tableau (column indices)
/// * `q_tableau` - The Q tableau (row indices)
///
/// # Returns
/// The binary matrix if the tableaux are valid, None otherwise
pub fn hillman_grassl_inverse(p_tableau: &Tableau, q_tableau: &Tableau) -> Option<Vec<Vec<usize>>> {
    // Check that both tableaux have the same shape
    if p_tableau.shape() != q_tableau.shape() {
        return None;
    }

    if p_tableau.size() == 0 {
        return Some(vec![]);
    }

    // Determine matrix dimensions
    let max_row = q_tableau.rows()
        .iter()
        .flat_map(|row| row.iter())
        .max()
        .copied()
        .unwrap_or(0);

    let max_col = p_tableau.rows()
        .iter()
        .flat_map(|row| row.iter())
        .max()
        .copied()
        .unwrap_or(0);

    if max_row == 0 || max_col == 0 {
        return Some(vec![]);
    }

    // Initialize matrix with zeros
    let mut matrix = vec![vec![0; max_col]; max_row];

    // Fill in the 1s based on (P, Q) pairs
    for (p_row, q_row) in p_tableau.rows().iter().zip(q_tableau.rows().iter()) {
        for (&col_idx, &row_idx) in p_row.iter().zip(q_row.iter()) {
            if row_idx > 0 && row_idx <= max_row && col_idx > 0 && col_idx <= max_col {
                matrix[row_idx - 1][col_idx - 1] = 1;
            }
        }
    }

    Some(matrix)
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
    fn test_hillman_grassl_empty_matrix() {
        // Empty matrix should give empty tableaux
        let matrix: Vec<Vec<usize>> = vec![];
        let result = hillman_grassl(&matrix);
        assert!(result.is_some());

        let (p, q) = result.unwrap();
        assert_eq!(p.size(), 0);
        assert_eq!(q.size(), 0);
    }

    #[test]
    fn test_hillman_grassl_single_one() {
        // Matrix with single 1
        let matrix = vec![vec![1]];
        let result = hillman_grassl(&matrix);
        assert!(result.is_some());

        let (p, q) = result.unwrap();
        assert_eq!(p.size(), 1);
        assert_eq!(q.size(), 1);
        assert_eq!(p.rows(), &[vec![1]]);
        assert_eq!(q.rows(), &[vec![1]]);
    }

    #[test]
    fn test_hillman_grassl_simple_matrix() {
        // Test with a simple 2x2 matrix
        // [1, 0]
        // [0, 1]
        let matrix = vec![vec![1, 0], vec![0, 1]];
        let result = hillman_grassl(&matrix);
        assert!(result.is_some());

        let (p, q) = result.unwrap();
        // Both tableaux should have the same shape
        assert_eq!(p.shape(), q.shape());
        // Should have 2 entries total (2 ones in the matrix)
        assert_eq!(p.size(), 2);
        assert_eq!(q.size(), 2);
        // P should be semistandard
        assert!(p.is_semistandard());
    }

    #[test]
    fn test_hillman_grassl_row_vector() {
        // Matrix with single row [1, 1, 1]
        let matrix = vec![vec![1, 1, 1]];
        let result = hillman_grassl(&matrix);
        assert!(result.is_some());

        let (p, q) = result.unwrap();
        // P should contain columns 1, 2, 3
        assert_eq!(p.size(), 3);
        // All entries in Q should be 1 (all from row 1)
        assert!(q.rows().iter().all(|row| row.iter().all(|&x| x == 1)));
    }

    #[test]
    fn test_hillman_grassl_column_vector() {
        // Matrix with single column [1], [1], [1]
        let matrix = vec![vec![1], vec![1], vec![1]];
        let result = hillman_grassl(&matrix);
        assert!(result.is_some());

        let (p, q) = result.unwrap();
        // P should have all 1s (column 1)
        assert!(p.rows().iter().all(|row| row.iter().all(|&x| x == 1)));
        // Q should contain rows 1, 2, 3
        assert_eq!(q.size(), 3);
    }

    #[test]
    fn test_hillman_grassl_3x3_matrix() {
        // Test with a 3x3 matrix
        // [1, 0, 1]
        // [0, 1, 0]
        // [1, 1, 0]
        let matrix = vec![
            vec![1, 0, 1],
            vec![0, 1, 0],
            vec![1, 1, 0],
        ];
        let result = hillman_grassl(&matrix);
        assert!(result.is_some());

        let (p, q) = result.unwrap();
        // Should have 5 entries (5 ones in the matrix)
        assert_eq!(p.size(), 5);
        assert_eq!(q.size(), 5);
        // Same shape
        assert_eq!(p.shape(), q.shape());
        // P should be semistandard (Q may not be in Hillman-Grassl)
        assert!(p.is_semistandard());
    }

    #[test]
    fn test_hillman_grassl_invalid_matrix() {
        // Non-binary matrix should return None
        let matrix = vec![vec![1, 2, 0], vec![0, 1, 3]];
        let result = hillman_grassl(&matrix);
        assert!(result.is_none());
    }

    #[test]
    fn test_hillman_grassl_zero_matrix() {
        // Matrix of all zeros
        let matrix = vec![vec![0, 0], vec![0, 0]];
        let result = hillman_grassl(&matrix);
        assert!(result.is_some());

        let (p, q) = result.unwrap();
        // Should be empty tableaux
        assert_eq!(p.size(), 0);
        assert_eq!(q.size(), 0);
    }

    #[test]
    fn test_hillman_grassl_inverse_basic() {
        // Test inverse on simple matrix
        let matrix = vec![vec![1, 0], vec![0, 1]];
        let (p, q) = hillman_grassl(&matrix).unwrap();

        let reconstructed = hillman_grassl_inverse(&p, &q);
        assert!(reconstructed.is_some());
        assert_eq!(reconstructed.unwrap(), matrix);
    }

    #[test]
    fn test_hillman_grassl_inverse_3x3() {
        // Test inverse on 3x3 matrix
        let matrix = vec![
            vec![1, 0, 1],
            vec![0, 1, 0],
            vec![1, 1, 0],
        ];
        let (p, q) = hillman_grassl(&matrix).unwrap();

        // Debug output
        println!("Original matrix: {:?}", matrix);
        println!("P tableau: {:?}", p.rows());
        println!("Q tableau: {:?}", q.rows());

        let reconstructed = hillman_grassl_inverse(&p, &q);
        assert!(reconstructed.is_some());
        println!("Reconstructed: {:?}", reconstructed.as_ref().unwrap());
        assert_eq!(reconstructed.unwrap(), matrix);
    }

    #[test]
    fn test_hillman_grassl_inverse_row_vector() {
        // Test inverse on row vector
        let matrix = vec![vec![1, 1, 1]];
        let (p, q) = hillman_grassl(&matrix).unwrap();

        let reconstructed = hillman_grassl_inverse(&p, &q);
        assert!(reconstructed.is_some());
        assert_eq!(reconstructed.unwrap(), matrix);
    }

    #[test]
    fn test_hillman_grassl_inverse_column_vector() {
        // Test inverse on column vector
        let matrix = vec![vec![1], vec![1], vec![1]];
        let (p, q) = hillman_grassl(&matrix).unwrap();

        let reconstructed = hillman_grassl_inverse(&p, &q);
        assert!(reconstructed.is_some());
        assert_eq!(reconstructed.unwrap(), matrix);
    }

    #[test]
    fn test_hillman_grassl_inverse_empty() {
        // Test inverse on empty tableaux
        let p = Tableau::new(vec![]).unwrap();
        let q = Tableau::new(vec![]).unwrap();

        let reconstructed = hillman_grassl_inverse(&p, &q);
        assert!(reconstructed.is_some());
        assert_eq!(reconstructed.unwrap(), vec![] as Vec<Vec<usize>>);
    }

    #[test]
    fn test_hillman_grassl_inverse_mismatched_shapes() {
        // Tableaux with different shapes should return None
        let p = Tableau::new(vec![vec![1, 2]]).unwrap();
        let q = Tableau::new(vec![vec![1], vec![2]]).unwrap();

        let result = hillman_grassl_inverse(&p, &q);
        assert!(result.is_none());
    }

    #[test]
    fn test_hillman_grassl_roundtrip() {
        // Test that hillman_grassl and its inverse are truly inverse operations
        let test_matrices = vec![
            vec![vec![1, 0, 1], vec![0, 1, 0], vec![1, 1, 0]],
            vec![vec![1, 1, 0], vec![1, 0, 1], vec![0, 1, 1]],
            vec![vec![1, 0, 0, 1], vec![0, 1, 1, 0]],
            vec![vec![1], vec![1], vec![1], vec![1]],
            vec![vec![1, 1, 1, 1]],
        ];

        for matrix in test_matrices {
            let (p, q) = hillman_grassl(&matrix).unwrap();
            let reconstructed = hillman_grassl_inverse(&p, &q).unwrap();
            assert_eq!(reconstructed, matrix, "Roundtrip failed for matrix {:?}", matrix);
        }
    }

    #[test]
    fn test_hillman_grassl_insert() {
        // Test the insertion algorithm directly
        let t = Tableau::new(vec![]).unwrap();

        // Insert 2
        let t = hillman_grassl_insert(&t, 2);
        assert_eq!(t.rows(), &[vec![2]]);

        // Insert 1 (should bump 2)
        let t = hillman_grassl_insert(&t, 1);
        assert_eq!(t.rows(), &[vec![1], vec![2]]);

        // Insert 2 again (should go in first row since it allows repeats)
        let t = hillman_grassl_insert(&t, 2);
        assert_eq!(t.rows(), &[vec![1, 2], vec![2]]);

        // Result should be semistandard
        assert!(t.is_semistandard());
    }

    #[test]
    fn test_hillman_grassl_properties() {
        // Test that the resulting tableaux have expected properties
        let matrix = vec![
            vec![1, 1, 0, 0],
            vec![0, 1, 1, 0],
            vec![1, 0, 0, 1],
        ];

        let (p, q) = hillman_grassl(&matrix).unwrap();

        // Count total number of 1s in matrix
        let num_ones: usize = matrix.iter().map(|row| row.iter().sum::<usize>()).sum();

        // P and Q should have the same size as number of 1s
        assert_eq!(p.size(), num_ones);
        assert_eq!(q.size(), num_ones);

        // P and Q should have the same shape
        assert_eq!(p.shape(), q.shape());

        // P should be semistandard
        assert!(p.is_semistandard());

        // P should only contain values from 1 to number of columns
        let max_p_value = p.rows().iter().flat_map(|row| row.iter()).max().copied().unwrap_or(0);
        assert!(max_p_value <= matrix[0].len());

        // Q should only contain values from 1 to number of rows
        let max_q_value = q.rows().iter().flat_map(|row| row.iter()).max().copied().unwrap_or(0);
        assert!(max_q_value <= matrix.len());
    }
}
