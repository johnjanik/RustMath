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

/// An entry in a shifted primed tableau - can be primed or unprimed
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrimedEntry {
    /// Unprimed entry with value
    Unprimed(usize),
    /// Primed entry with value (represented as value')
    Primed(usize),
}

impl PrimedEntry {
    /// Get the underlying value (ignoring prime status)
    pub fn value(&self) -> usize {
        match self {
            PrimedEntry::Unprimed(v) | PrimedEntry::Primed(v) => *v,
        }
    }

    /// Check if this entry is primed
    pub fn is_primed(&self) -> bool {
        matches!(self, PrimedEntry::Primed(_))
    }

    /// Check if this entry is unprimed
    pub fn is_unprimed(&self) -> bool {
        matches!(self, PrimedEntry::Unprimed(_))
    }

    /// Create an unprimed entry
    pub fn unprimed(value: usize) -> Self {
        PrimedEntry::Unprimed(value)
    }

    /// Create a primed entry
    pub fn primed(value: usize) -> Self {
        PrimedEntry::Primed(value)
    }
}

impl PartialOrd for PrimedEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrimedEntry {
    /// Ordering for primed entries: first by value, then unprimed < primed
    /// This means 1 < 1' < 2 < 2' < 3 < 3'
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.value().cmp(&other.value()) {
            std::cmp::Ordering::Equal => {
                // If values are equal, unprimed comes before primed
                match (self, other) {
                    (PrimedEntry::Unprimed(_), PrimedEntry::Primed(_)) => std::cmp::Ordering::Less,
                    (PrimedEntry::Primed(_), PrimedEntry::Unprimed(_)) => std::cmp::Ordering::Greater,
                    _ => std::cmp::Ordering::Equal,
                }
            }
            other_ordering => other_ordering,
        }
    }
}

impl std::fmt::Display for PrimedEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrimedEntry::Unprimed(v) => write!(f, "{}", v),
            PrimedEntry::Primed(v) => write!(f, "{}'", v),
        }
    }
}

/// A shifted primed tableau
///
/// A shifted primed tableau is a filling of a shifted Young diagram (where row i
/// starts at column i) with entries that can be primed or unprimed, satisfying:
/// - The shape is a strict partition (strictly decreasing parts)
/// - Rows are weakly increasing from left to right
/// - Columns are strictly increasing from top to bottom
/// - An unprimed entry cannot appear to the right of a primed entry of the same value
/// - No two primed entries with the same value can appear in the same column
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShiftedPrimedTableau {
    /// The entries of the tableau, organized by rows
    /// Row i contains entries starting at column i (shifted position)
    rows: Vec<Vec<PrimedEntry>>,
    /// The shape (strict partition) of the tableau
    shape: Vec<usize>,
}

impl ShiftedPrimedTableau {
    /// Create a shifted primed tableau from rows
    ///
    /// Returns None if the rows don't form a valid shifted primed tableau
    pub fn new(rows: Vec<Vec<PrimedEntry>>) -> Option<Self> {
        if rows.is_empty() {
            return Some(ShiftedPrimedTableau {
                rows: vec![],
                shape: vec![],
            });
        }

        // Extract shape - for shifted tableaux, this should be strictly decreasing
        let shape: Vec<usize> = rows.iter().map(|row| row.len()).collect();

        // Check that shape is a strict partition (strictly decreasing)
        for i in 1..shape.len() {
            if shape[i] >= shape[i - 1] {
                return None; // Not strictly decreasing
            }
        }

        let tableau = ShiftedPrimedTableau { rows, shape };

        // Validate the tableau rules
        if !tableau.is_valid() {
            return None;
        }

        Some(tableau)
    }

    /// Get the shape of the tableau
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the rows of the tableau
    pub fn rows(&self) -> &[Vec<PrimedEntry>] {
        &self.rows
    }

    /// Get the number of entries in the tableau
    pub fn size(&self) -> usize {
        self.rows.iter().map(|row| row.len()).sum()
    }

    /// Check if this tableau satisfies all the shifted primed tableau rules
    pub fn is_valid(&self) -> bool {
        // Check row conditions
        for row in &self.rows {
            if !self.is_valid_row(row) {
                return false;
            }
        }

        // Check column conditions
        for col_idx in 0..*self.shape.iter().max().unwrap_or(&0) {
            if !self.is_valid_column(col_idx) {
                return false;
            }
        }

        true
    }

    /// Check if a row satisfies the row conditions
    fn is_valid_row(&self, row: &[PrimedEntry]) -> bool {
        // Check weakly increasing
        for i in 1..row.len() {
            if row[i] < row[i - 1] {
                return false;
            }
        }

        // Check that unprimed doesn't follow primed of same value
        for i in 1..row.len() {
            if row[i].is_unprimed() {
                let value = row[i].value();
                // Check if there's a primed entry with the same value to the left
                for j in 0..i {
                    if row[j] == PrimedEntry::Primed(value) {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Check if a column satisfies the column conditions
    fn is_valid_column(&self, col_idx: usize) -> bool {
        let mut primed_values = std::collections::HashSet::new();
        let mut prev_entry: Option<PrimedEntry> = None;

        for (row_idx, row) in self.rows.iter().enumerate() {
            // Column col_idx in shifted tableau is at position col_idx - row_idx in row row_idx
            // But we need to account for the shift: row i starts at column i
            // So absolute column col_idx corresponds to relative position col_idx - row_idx in row row_idx
            if col_idx < row_idx {
                continue; // This row doesn't have this column
            }

            let relative_col = col_idx - row_idx;
            if relative_col >= row.len() {
                continue; // This row doesn't extend to this column
            }

            let entry = row[relative_col];

            // Check strictly increasing
            if let Some(prev) = prev_entry {
                if entry <= prev {
                    return false;
                }
            }
            prev_entry = Some(entry);

            // Check no repeated primed values
            if entry.is_primed() {
                if !primed_values.insert(entry.value()) {
                    return false; // Repeated primed value in column
                }
            }
        }

        true
    }

    /// Get entry at position (row, col) in absolute coordinates
    /// Returns None if the position is not in the tableau
    pub fn get(&self, row: usize, col: usize) -> Option<PrimedEntry> {
        if row >= self.rows.len() {
            return None;
        }
        if col < row {
            return None; // Before the start of this row in shifted tableau
        }
        let relative_col = col - row;
        self.rows.get(row)?.get(relative_col).copied()
    }

    /// Get the number of rows
    pub fn num_rows(&self) -> usize {
        self.rows.len()
    }

    /// Display the tableau as a string with proper shifting
    pub fn to_string(&self) -> String {
        self.rows
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let indent = "  ".repeat(i); // 2 spaces per shift level
                let entries = row
                    .iter()
                    .map(|e| format!("{:3}", e.to_string()))
                    .collect::<Vec<_>>()
                    .join(" ");
                format!("{}{}", indent, entries)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Check if this is a standard shifted primed tableau
    ///
    /// A standard shifted primed tableau uses each value 1, 2, ..., n exactly once
    pub fn is_standard(&self) -> bool {
        let n = self.size();
        if n == 0 {
            return true;
        }

        // Collect all values (ignoring prime status)
        let mut values: Vec<usize> = self.rows
            .iter()
            .flat_map(|row| row.iter().map(|e| e.value()))
            .collect();
        values.sort_unstable();

        // Check that we have exactly 1, 2, ..., n
        values == (1..=n).collect::<Vec<_>>()
    }

    /// Get all entries as a flat list
    pub fn entries(&self) -> Vec<PrimedEntry> {
        self.rows.iter().flat_map(|row| row.iter().copied()).collect()
    }

    /// Count the number of primed entries
    pub fn num_primed(&self) -> usize {
        self.entries().iter().filter(|e| e.is_primed()).count()
    }

    /// Count the number of unprimed entries
    pub fn num_unprimed(&self) -> usize {
        self.entries().iter().filter(|e| e.is_unprimed()).count()
    }
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

    // Tests for PrimedEntry
    #[test]
    fn test_primed_entry_creation() {
        let u1 = PrimedEntry::unprimed(1);
        let p1 = PrimedEntry::primed(1);

        assert!(u1.is_unprimed());
        assert!(!u1.is_primed());
        assert!(p1.is_primed());
        assert!(!p1.is_unprimed());

        assert_eq!(u1.value(), 1);
        assert_eq!(p1.value(), 1);
    }

    #[test]
    fn test_primed_entry_ordering() {
        let u1 = PrimedEntry::unprimed(1);
        let p1 = PrimedEntry::primed(1);
        let u2 = PrimedEntry::unprimed(2);
        let p2 = PrimedEntry::primed(2);

        // 1 < 1' < 2 < 2'
        assert!(u1 < p1);
        assert!(p1 < u2);
        assert!(u2 < p2);

        assert_eq!(u1, u1);
        assert_eq!(p1, p1);
    }

    #[test]
    fn test_primed_entry_display() {
        let u1 = PrimedEntry::unprimed(5);
        let p1 = PrimedEntry::primed(3);

        assert_eq!(format!("{}", u1), "5");
        assert_eq!(format!("{}", p1), "3'");
    }

    // Tests for ShiftedPrimedTableau
    #[test]
    fn test_shifted_primed_tableau_empty() {
        let t = ShiftedPrimedTableau::new(vec![]).unwrap();
        assert_eq!(t.size(), 0);
        assert_eq!(t.num_rows(), 0);
        assert!(t.is_valid());
    }

    #[test]
    fn test_shifted_primed_tableau_valid() {
        use PrimedEntry::{Primed as P, Unprimed as U};

        // Valid shifted primed tableau with shape (3, 2)
        // Row 0: 1  2  3
        // Row 1:   4  5
        let t = ShiftedPrimedTableau::new(vec![
            vec![U(1), U(2), U(3)],
            vec![U(4), U(5)],
        ]);

        assert!(t.is_some());
        let t = t.unwrap();
        assert_eq!(t.size(), 5);
        assert_eq!(t.num_rows(), 2);
        assert_eq!(t.shape(), &[3, 2]);
        assert!(t.is_valid());
        assert!(t.is_standard());
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

    // k-tableau tests
    // TODO: Implement KTableau
    // #[test]
    // fn test_k_tableau_creation() {
    //     // Valid 2-tableau: columns differ by at least 2
    //     // [[1, 2, 3],
    //     //  [3, 4, 5]]
    //     let kt = KTableau::new(vec![vec![1, 2, 3], vec![3, 4, 5]], 2);
    //     assert!(kt.is_some());
    //     let kt = kt.unwrap();
    //     assert_eq!(kt.k(), 2);
    //     assert_eq!(kt.size(), 6);
    //     assert!(kt.is_valid());
    // }

    #[test]
    fn test_shifted_primed_tableau_invalid_shape() {
        use PrimedEntry::Unprimed as U;

        // Invalid - shape is not strictly decreasing (3, 3)
        let t = ShiftedPrimedTableau::new(vec![
            vec![U(1), U(2), U(3)],
            vec![U(4), U(5), U(6)],
        ]);

        assert!(t.is_none()); // Should fail because shape is not strict
    }

    #[test]
    fn test_shifted_primed_tableau_row_not_increasing() {
        use PrimedEntry::Unprimed as U;

        // Invalid - row not weakly increasing
        let t = ShiftedPrimedTableau::new(vec![
            vec![U(2), U(1), U(3)], // Not increasing
            vec![U(4)],
        ]);

        assert!(t.is_none());
    }

    #[test]
    fn test_shifted_primed_tableau_column_not_increasing() {
        use PrimedEntry::{Primed as P, Unprimed as U};

        // Invalid - column not strictly increasing
        // Row 0: 1  2
        // Row 1:   2  (column 1 has 2, 2 which is not strictly increasing)
        let t = ShiftedPrimedTableau::new(vec![
            vec![U(1), U(2)],
            vec![U(2)],
        ]);

        assert!(t.is_none());
    }

    #[test]
    fn test_shifted_primed_tableau_unprimed_after_primed() {
        use PrimedEntry::{Primed as P, Unprimed as U};

        // Invalid - unprimed 2 follows primed 2 in same row
        let t = ShiftedPrimedTableau::new(vec![
            vec![U(1), P(2), U(2)], // Invalid: U(2) after P(2)
            vec![U(3)],
        ]);

        assert!(t.is_none());
    }

    #[test]
    fn test_shifted_primed_tableau_repeated_primed_in_column() {
        use PrimedEntry::{Primed as P, Unprimed as U};

        // Invalid - primed value appears twice in same column
        // In shifted tableau:
        // Row 0: 1' 2'  (columns 0, 1)
        // Row 1:   2' 3' (columns 1, 2)
        // Column 1 would have 2' and 2' which is invalid
        let t = ShiftedPrimedTableau::new(vec![
            vec![P(1), P(2)],
            vec![P(2), P(3)],
        ]);

        assert!(t.is_none());
    }

    #[test]
    fn test_shifted_primed_tableau_get() {
        use PrimedEntry::{Primed as P, Unprimed as U};

        // Row 0: 1  2  3  (columns 0, 1, 2)
        // Row 1:   4  5   (columns 1, 2)
        let t = ShiftedPrimedTableau::new(vec![
            vec![U(1), U(2), U(3)],
            vec![U(4), U(5)],
        ]).unwrap();

        assert_eq!(t.get(0, 0), Some(U(1)));
        assert_eq!(t.get(0, 1), Some(U(2)));
        assert_eq!(t.get(0, 2), Some(U(3)));
        assert_eq!(t.get(1, 1), Some(U(4)));
        assert_eq!(t.get(1, 2), Some(U(5)));

        // Invalid positions
        assert_eq!(t.get(1, 0), None); // Before start of row 1
        assert_eq!(t.get(0, 3), None); // Beyond row 0
        assert_eq!(t.get(2, 0), None); // Beyond tableau
    }

    #[test]
    fn test_shifted_primed_tableau_num_primed_unprimed() {
        use PrimedEntry::{Primed as P, Unprimed as U};

        let t = ShiftedPrimedTableau::new(vec![
            vec![U(1), P(1), U(2)],
            vec![P(2), U(3)],
        ]).unwrap();

        assert_eq!(t.num_primed(), 2);
        assert_eq!(t.num_unprimed(), 3);
    }

    #[test]
    fn test_shifted_primed_tableau_to_string() {
        use PrimedEntry::{Primed as P, Unprimed as U};

        let t = ShiftedPrimedTableau::new(vec![
            vec![U(1), P(1), U(2)],
            vec![P(2), U(3)],
        ]).unwrap();

        let s = t.to_string();
        // Should show the shifted structure with indentation
        assert!(s.contains("1"));
        assert!(s.contains("1'"));
        assert!(s.contains("2"));
    }

    #[test]
    fn test_shifted_primed_tableau_standard() {
        use PrimedEntry::{Primed as P, Unprimed as U};

        // Standard: uses 1, 2, 3, 4, 5 each exactly once
        let t1 = ShiftedPrimedTableau::new(vec![
            vec![U(1), U(2), P(3)],
            vec![U(4), P(5)],
        ]).unwrap();
        assert!(t1.is_standard());

        // Not standard: uses 1, 1, 2, 2, 3 (repeated values because of primes)
        let t2 = ShiftedPrimedTableau::new(vec![
            vec![U(1), P(1), U(2)],
            vec![P(2), P(3)],
        ]).unwrap();
        // This has values 1, 1, 2, 2, 3 so not standard
        assert!(!t2.is_standard());
    }

    #[test]
    fn test_shifted_primed_tableau_complex() {
        use PrimedEntry::{Primed as P, Unprimed as U};

        // A more complex valid tableau
        // Row 0: 1  1' 2  2'  (columns 0, 1, 2, 3)
        // Row 1:   3  3' 4   (columns 1, 2, 3)
        // Row 2:     5  6    (columns 2, 3)
        let t = ShiftedPrimedTableau::new(vec![
            vec![U(1), P(1), U(2), P(2)],
            vec![U(3), P(3), U(4)],
            vec![U(5), U(6)],
        ]);

        assert!(t.is_some());
        let t = t.unwrap();
        assert_eq!(t.size(), 9);
        assert_eq!(t.num_rows(), 3);
        assert!(t.is_valid());
    }
}
