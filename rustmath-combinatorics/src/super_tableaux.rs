//! Super tableaux - Young tableaux with circled entries
//!
//! A super tableau is a filling of a Young diagram where each entry can be either
//! circled or uncircled. These arise in the theory of symmetric functions and
//! representation theory of Lie superalgebras gl(m|n).
//!
//! ## Ordering of Entries
//!
//! Circled entries are considered to be "between" their uncircled value and the next:
//! 1 < ◯1 < 2 < ◯2 < 3 < ◯3 < ...
//!
//! ## Super-semistandard tableaux
//!
//! A super tableau is super-semistandard if:
//! - Entries are weakly increasing along each row (left to right)
//! - Entries are strictly increasing down each column (top to bottom)
//! - No column contains both circled and uncircled versions of the same number

use crate::partitions::Partition;
use std::cmp::Ordering;
use std::fmt;

/// A single entry in a super tableau
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SuperTableauEntry {
    /// The value of the entry
    pub value: usize,
    /// Whether this entry is circled
    pub circled: bool,
}

impl SuperTableauEntry {
    /// Create a new super tableau entry
    pub fn new(value: usize, circled: bool) -> Self {
        SuperTableauEntry { value, circled }
    }

    /// Create an uncircled entry
    pub fn uncircled(value: usize) -> Self {
        SuperTableauEntry {
            value,
            circled: false,
        }
    }

    /// Create a circled entry
    pub fn circled(value: usize) -> Self {
        SuperTableauEntry {
            value,
            circled: true,
        }
    }
}

impl PartialOrd for SuperTableauEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SuperTableauEntry {
    /// Ordering for super tableau entries
    ///
    /// The ordering is: 1 < ◯1 < 2 < ◯2 < 3 < ◯3 < ...
    /// For equal values, uncircled < circled
    fn cmp(&self, other: &Self) -> Ordering {
        match self.value.cmp(&other.value) {
            Ordering::Equal => {
                // For equal values: uncircled < circled
                match (self.circled, other.circled) {
                    (false, true) => Ordering::Less,
                    (true, false) => Ordering::Greater,
                    _ => Ordering::Equal,
                }
            }
            other_ordering => other_ordering,
        }
    }
}

impl fmt::Display for SuperTableauEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.circled {
            write!(f, "◯{}", self.value)
        } else {
            write!(f, "{}", self.value)
        }
    }
}

/// A super tableau - a filling of a Young diagram with circled and uncircled entries
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuperTableau {
    /// The entries of the tableau, organized by rows
    rows: Vec<Vec<SuperTableauEntry>>,
    /// The shape (partition) of the tableau
    shape: Partition,
}

impl SuperTableau {
    /// Create a super tableau from rows
    ///
    /// Returns None if the rows don't form a valid tableau shape
    pub fn new(rows: Vec<Vec<SuperTableauEntry>>) -> Option<Self> {
        if rows.is_empty() {
            return Some(SuperTableau {
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

        Some(SuperTableau { rows, shape })
    }

    /// Get the shape of the tableau
    pub fn shape(&self) -> &Partition {
        &self.shape
    }

    /// Get the rows of the tableau
    pub fn rows(&self) -> &[Vec<SuperTableauEntry>] {
        &self.rows
    }

    /// Get the number of entries in the tableau
    pub fn size(&self) -> usize {
        self.rows.iter().map(|row| row.len()).sum()
    }

    /// Get the number of rows
    pub fn num_rows(&self) -> usize {
        self.rows.len()
    }

    /// Get entry at position (row, col), if it exists
    pub fn get(&self, row: usize, col: usize) -> Option<SuperTableauEntry> {
        self.rows.get(row)?.get(col).copied()
    }

    /// Check if this is a super-semistandard tableau
    ///
    /// A super tableau is super-semistandard if:
    /// 1. Rows are weakly increasing (non-decreasing)
    /// 2. Columns are strictly increasing
    /// 3. No column has both circled and uncircled versions of the same number
    pub fn is_super_semistandard(&self) -> bool {
        if self.rows.is_empty() {
            return true;
        }

        // Check rows are weakly increasing
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

        // Check that no column has both circled and uncircled versions of same number
        for col in 0..self.rows[0].len() {
            let mut values_circled = std::collections::HashMap::new();

            for row in 0..self.rows.len() {
                if col < self.rows[row].len() {
                    let entry = self.rows[row][col];

                    if let Some(&prev_circled) = values_circled.get(&entry.value) {
                        // We've seen this value before in this column
                        // Check if it was with different circled status
                        if prev_circled != entry.circled {
                            return false; // Both circled and uncircled versions exist
                        }
                    } else {
                        values_circled.insert(entry.value, entry.circled);
                    }
                }
            }
        }

        true
    }

    /// Check if this is a super-standard tableau
    ///
    /// A standard super tableau has entries from a multiset where each value
    /// appears at most once in each form (circled or uncircled), and satisfies
    /// the super-semistandard property with strict increase in rows.
    pub fn is_super_standard(&self) -> bool {
        if !self.is_super_semistandard() {
            return false;
        }

        // Check rows are strictly increasing
        for row in &self.rows {
            for i in 1..row.len() {
                if row[i] <= row[i - 1] {
                    return false;
                }
            }
        }

        // Check that each entry appears at most once
        let mut seen = std::collections::HashSet::new();
        for row in &self.rows {
            for &entry in row {
                if !seen.insert(entry) {
                    return false; // Entry appears more than once
                }
            }
        }

        true
    }

    /// Get the content (multiset of entries) of the tableau
    pub fn content(&self) -> Vec<SuperTableauEntry> {
        let mut entries: Vec<SuperTableauEntry> = self
            .rows
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        entries.sort();
        entries
    }

    /// Get the reading word (row reading from bottom to top)
    pub fn reading_word(&self) -> Vec<SuperTableauEntry> {
        self.rows
            .iter()
            .rev()
            .flat_map(|row| row.iter().copied())
            .collect()
    }

    /// Count the number of circled entries
    pub fn num_circled(&self) -> usize {
        self.rows
            .iter()
            .flat_map(|row| row.iter())
            .filter(|e| e.circled)
            .count()
    }

    /// Count the number of uncircled entries
    pub fn num_uncircled(&self) -> usize {
        self.rows
            .iter()
            .flat_map(|row| row.iter())
            .filter(|e| !e.circled)
            .count()
    }

    /// Convert to a regular tableau by forgetting the circled status
    pub fn forget_circles(&self) -> crate::tableaux::Tableau {
        let regular_rows: Vec<Vec<usize>> = self
            .rows
            .iter()
            .map(|row| row.iter().map(|e| e.value).collect())
            .collect();
        crate::tableaux::Tableau::new(regular_rows).expect("Valid super tableau should give valid regular tableau")
    }

    /// Create a super tableau from a regular tableau with no circled entries
    pub fn from_tableau(tableau: &crate::tableaux::Tableau) -> Self {
        let super_rows: Vec<Vec<SuperTableauEntry>> = tableau
            .rows()
            .iter()
            .map(|row| row.iter().map(|&v| SuperTableauEntry::uncircled(v)).collect())
            .collect();
        SuperTableau::new(super_rows).expect("Valid regular tableau should give valid super tableau")
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

    /// Compute the sign of the super tableau
    ///
    /// The sign is (-1)^(number of circled entries)
    pub fn sign(&self) -> i32 {
        let num_circled = self.num_circled();
        if num_circled % 2 == 0 {
            1
        } else {
            -1
        }
    }
}

impl fmt::Display for SuperTableau {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, row) in self.rows.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            for (j, entry) in row.iter().enumerate() {
                if j > 0 {
                    write!(f, " ")?;
                }
                write!(f, "{}", entry)?;
            }
        }
        Ok(())
    }
}

/// Generate all super-standard tableaux of a given shape
///
/// This generates tableaux where entries come from {1, ◯1, 2, ◯2, ..., n, ◯n}
/// and each entry appears at most once.
pub fn standard_super_tableaux(shape: &Partition, max_value: usize) -> Vec<SuperTableau> {
    let n = shape.sum();
    if n == 0 {
        return vec![SuperTableau {
            rows: vec![],
            shape: shape.clone(),
        }];
    }

    // Generate all possible entries up to max_value
    let mut possible_entries = Vec::new();
    for v in 1..=max_value {
        possible_entries.push(SuperTableauEntry::uncircled(v));
        possible_entries.push(SuperTableauEntry::circled(v));
    }

    let mut result = Vec::new();
    let mut current: Vec<Vec<SuperTableauEntry>> = shape
        .parts()
        .iter()
        .map(|&len| vec![SuperTableauEntry::uncircled(0); len])
        .collect();

    generate_standard_super_tableaux(
        &mut current,
        shape,
        &possible_entries,
        0,
        n,
        &mut result,
    );

    result
}

fn generate_standard_super_tableaux(
    current: &mut Vec<Vec<SuperTableauEntry>>,
    shape: &Partition,
    possible_entries: &[SuperTableauEntry],
    next_entry_idx: usize,
    target_size: usize,
    result: &mut Vec<SuperTableau>,
) {
    // Count how many entries we've placed
    let placed = current
        .iter()
        .flat_map(|row| row.iter())
        .filter(|e| e.value > 0)
        .count();

    if placed == target_size {
        // All positions filled, check if valid and add
        if let Some(tableau) = SuperTableau::new(current.clone()) {
            if tableau.is_super_standard() {
                result.push(tableau);
            }
        }
        return;
    }

    if next_entry_idx >= possible_entries.len() {
        return;
    }

    // Try placing the current entry in each valid position
    for r in 0..shape.length() {
        for c in 0..shape.parts()[r] {
            if current[r][c].value == 0 && can_place_super(current, r, c, possible_entries[next_entry_idx]) {
                current[r][c] = possible_entries[next_entry_idx];
                generate_standard_super_tableaux(
                    current,
                    shape,
                    possible_entries,
                    next_entry_idx + 1,
                    target_size,
                    result,
                );
                current[r][c] = SuperTableauEntry::uncircled(0);
            }
        }
    }

    // Also try skipping this entry
    generate_standard_super_tableaux(
        current,
        shape,
        possible_entries,
        next_entry_idx + 1,
        target_size,
        result,
    );
}

fn can_place_super(
    current: &[Vec<SuperTableauEntry>],
    row: usize,
    col: usize,
    entry: SuperTableauEntry,
) -> bool {
    // Check that this position hasn't been filled
    if current[row][col].value != 0 {
        return false;
    }

    // Check that all positions to the left are filled and smaller
    if col > 0 {
        if current[row][col - 1].value == 0 || current[row][col - 1] >= entry {
            return false;
        }
    }

    // Check that all positions above are filled and smaller
    if row > 0 && col < current[row - 1].len() {
        if current[row - 1][col].value == 0 || current[row - 1][col] >= entry {
            return false;
        }
    }

    // Check column constraint: no column can have both circled and uncircled same value
    if row > 0 && col < current[row - 1].len() {
        let above = current[row - 1][col];
        if above.value == entry.value && above.circled != entry.circled {
            return false;
        }
    }

    true
}

/// Generate all super-semistandard tableaux of a given shape with specified content
///
/// The content is a vector of entries that should appear in the tableau.
/// This generates all valid super-semistandard tableaux using these entries.
pub fn super_semistandard_tableaux(
    shape: &Partition,
    content: Vec<SuperTableauEntry>,
) -> Vec<SuperTableau> {
    let n = shape.sum();
    if content.len() != n {
        return vec![]; // Content size must match shape size
    }

    if n == 0 {
        return vec![SuperTableau {
            rows: vec![],
            shape: shape.clone(),
        }];
    }

    let mut result = Vec::new();
    let mut current: Vec<Vec<SuperTableauEntry>> = shape
        .parts()
        .iter()
        .map(|&len| vec![SuperTableauEntry::uncircled(0); len])
        .collect();

    let mut sorted_content = content;
    sorted_content.sort();

    generate_super_semistandard_tableaux(
        &mut current,
        shape,
        &sorted_content,
        0,
        &mut vec![false; sorted_content.len()],
        &mut result,
    );

    result
}

fn generate_super_semistandard_tableaux(
    current: &mut Vec<Vec<SuperTableauEntry>>,
    shape: &Partition,
    content: &[SuperTableauEntry],
    position: usize,
    used: &mut Vec<bool>,
    result: &mut Vec<SuperTableau>,
) {
    // Calculate total cells from shape
    let total_cells = shape.sum();

    if position == total_cells {
        // All positions filled
        if let Some(tableau) = SuperTableau::new(current.clone()) {
            if tableau.is_super_semistandard() {
                result.push(tableau);
            }
        }
        return;
    }

    // Find current row and column
    let mut pos = 0;
    let mut target_row = 0;
    let mut target_col = 0;
    'outer: for (r, row_len) in shape.parts().iter().enumerate() {
        for c in 0..*row_len {
            if pos == position {
                target_row = r;
                target_col = c;
                break 'outer;
            }
            pos += 1;
        }
    }

    // Try each unused entry from content
    for i in 0..content.len() {
        if !used[i] && can_place_super_content(current, target_row, target_col, content[i]) {
            current[target_row][target_col] = content[i];
            used[i] = true;

            generate_super_semistandard_tableaux(
                current,
                shape,
                content,
                position + 1,
                used,
                result,
            );

            used[i] = false;
            current[target_row][target_col] = SuperTableauEntry::uncircled(0);
        }
    }
}

fn can_place_super_content(
    current: &[Vec<SuperTableauEntry>],
    row: usize,
    col: usize,
    entry: SuperTableauEntry,
) -> bool {
    // Check that all positions to the left are filled and smaller or equal
    if col > 0 {
        let left = current[row][col - 1];
        if left.value == 0 || left > entry {
            return false;
        }
    }

    // Check that all positions above are filled and strictly smaller
    if row > 0 && col < current[row - 1].len() {
        let above = current[row - 1][col];
        if above.value == 0 || above >= entry {
            return false;
        }

        // Column constraint: no both circled and uncircled same value
        if above.value == entry.value && above.circled != entry.circled {
            return false;
        }
    }

    // Check all values above in column for the "no both circled and uncircled" rule
    for r in 0..row {
        if col < current[r].len() {
            let above = current[r][col];
            if above.value == entry.value && above.circled != entry.circled {
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_super_entry_ordering() {
        let e1 = SuperTableauEntry::uncircled(1);
        let e1c = SuperTableauEntry::circled(1);
        let e2 = SuperTableauEntry::uncircled(2);
        let e2c = SuperTableauEntry::circled(2);

        // Test ordering: 1 < ◯1 < 2 < ◯2
        assert!(e1 < e1c);
        assert!(e1c < e2);
        assert!(e2 < e2c);
        assert!(e1 < e2);
    }

    #[test]
    fn test_super_entry_display() {
        let e1 = SuperTableauEntry::uncircled(3);
        assert_eq!(format!("{}", e1), "3");

        let e2 = SuperTableauEntry::circled(5);
        assert_eq!(format!("{}", e2), "◯5");
    }

    #[test]
    fn test_super_tableau_creation() {
        let rows = vec![
            vec![
                SuperTableauEntry::uncircled(1),
                SuperTableauEntry::circled(1),
                SuperTableauEntry::uncircled(2),
            ],
            vec![SuperTableauEntry::circled(2), SuperTableauEntry::uncircled(3)],
        ];
        let t = SuperTableau::new(rows).unwrap();

        assert_eq!(t.size(), 5);
        assert_eq!(t.shape().parts(), &[3, 2]);
    }

    #[test]
    fn test_super_semistandard() {
        // Valid super-semistandard: 1 ◯1 2
        //                           ◯2 3
        let rows = vec![
            vec![
                SuperTableauEntry::uncircled(1),
                SuperTableauEntry::circled(1),
                SuperTableauEntry::uncircled(2),
            ],
            vec![SuperTableauEntry::circled(2), SuperTableauEntry::uncircled(3)],
        ];
        let t = SuperTableau::new(rows).unwrap();
        assert!(t.is_super_semistandard());
    }

    #[test]
    fn test_not_super_semistandard_row_decrease() {
        // Invalid - row decreases: ◯2 1 2
        let rows = vec![vec![
            SuperTableauEntry::circled(2),
            SuperTableauEntry::uncircled(1),
            SuperTableauEntry::uncircled(2),
        ]];
        let t = SuperTableau::new(rows).unwrap();
        assert!(!t.is_super_semistandard());
    }

    #[test]
    fn test_not_super_semistandard_column_not_strict() {
        // Invalid - column not strictly increasing: 1 2
        //                                            1 3
        let rows = vec![
            vec![SuperTableauEntry::uncircled(1), SuperTableauEntry::uncircled(2)],
            vec![SuperTableauEntry::uncircled(1), SuperTableauEntry::uncircled(3)],
        ];
        let t = SuperTableau::new(rows).unwrap();
        assert!(!t.is_super_semistandard());
    }

    #[test]
    fn test_not_super_semistandard_column_both_circled_uncircled() {
        // Invalid - column has both 1 and ◯1: 1 2
        //                                      ◯1 3
        let rows = vec![
            vec![SuperTableauEntry::uncircled(1), SuperTableauEntry::uncircled(2)],
            vec![SuperTableauEntry::circled(1), SuperTableauEntry::uncircled(3)],
        ];
        let t = SuperTableau::new(rows).unwrap();
        assert!(!t.is_super_semistandard());
    }

    #[test]
    fn test_super_standard() {
        // Valid super-standard: 1 ◯1
        //                        2
        let rows = vec![
            vec![SuperTableauEntry::uncircled(1), SuperTableauEntry::circled(1)],
            vec![SuperTableauEntry::uncircled(2)],
        ];
        let t = SuperTableau::new(rows).unwrap();
        assert!(t.is_super_standard());
    }

    #[test]
    fn test_not_super_standard_repeated_entry() {
        // Invalid - has repeated uncircled 1: 1 1 2
        let rows = vec![vec![
            SuperTableauEntry::uncircled(1),
            SuperTableauEntry::uncircled(1),
            SuperTableauEntry::uncircled(2),
        ]];
        let t = SuperTableau::new(rows).unwrap();
        assert!(!t.is_super_standard());
    }

    #[test]
    fn test_content() {
        let rows = vec![
            vec![
                SuperTableauEntry::uncircled(1),
                SuperTableauEntry::circled(1),
                SuperTableauEntry::uncircled(2),
            ],
            vec![SuperTableauEntry::circled(2), SuperTableauEntry::uncircled(3)],
        ];
        let t = SuperTableau::new(rows).unwrap();

        let content = t.content();
        assert_eq!(content.len(), 5);
        // Should be sorted: 1, ◯1, 2, ◯2, 3
        assert_eq!(content[0], SuperTableauEntry::uncircled(1));
        assert_eq!(content[1], SuperTableauEntry::circled(1));
        assert_eq!(content[2], SuperTableauEntry::uncircled(2));
        assert_eq!(content[3], SuperTableauEntry::circled(2));
        assert_eq!(content[4], SuperTableauEntry::uncircled(3));
    }

    #[test]
    fn test_num_circled_uncircled() {
        let rows = vec![
            vec![
                SuperTableauEntry::uncircled(1),
                SuperTableauEntry::circled(1),
                SuperTableauEntry::uncircled(2),
            ],
            vec![SuperTableauEntry::circled(2), SuperTableauEntry::uncircled(3)],
        ];
        let t = SuperTableau::new(rows).unwrap();

        assert_eq!(t.num_circled(), 2);
        assert_eq!(t.num_uncircled(), 3);
    }

    #[test]
    fn test_sign() {
        // 2 circled entries -> even -> sign = 1
        let rows1 = vec![
            vec![SuperTableauEntry::circled(1), SuperTableauEntry::circled(2)],
        ];
        let t1 = SuperTableau::new(rows1).unwrap();
        assert_eq!(t1.sign(), 1);

        // 1 circled entry -> odd -> sign = -1
        let rows2 = vec![
            vec![SuperTableauEntry::uncircled(1), SuperTableauEntry::circled(2)],
        ];
        let t2 = SuperTableau::new(rows2).unwrap();
        assert_eq!(t2.sign(), -1);
    }

    #[test]
    fn test_forget_circles() {
        let rows = vec![
            vec![
                SuperTableauEntry::uncircled(1),
                SuperTableauEntry::circled(1),
                SuperTableauEntry::uncircled(2),
            ],
        ];
        let st = SuperTableau::new(rows).unwrap();
        let t = st.forget_circles();

        assert_eq!(t.rows(), &[vec![1, 1, 2]]);
    }

    #[test]
    fn test_from_tableau() {
        use crate::tableaux::Tableau;

        let regular = Tableau::new(vec![vec![1, 2, 3], vec![4, 5]]).unwrap();
        let super_t = SuperTableau::from_tableau(&regular);

        assert_eq!(super_t.size(), 5);
        assert_eq!(super_t.num_circled(), 0);
        assert_eq!(super_t.num_uncircled(), 5);
    }

    #[test]
    fn test_generate_standard_super_tableaux() {
        // Generate super tableaux of shape [2] with max value 2
        let shape = Partition::new(vec![2]);
        let tableaux = standard_super_tableaux(&shape, 2);

        // Should generate tableaux with entries from {1, ◯1, 2, ◯2}
        // All super-standard tableaux of shape [2]:
        // 1 ◯1, 1 2, 1 ◯2, ◯1 2, ◯1 ◯2, 2 ◯2
        assert!(tableaux.len() > 0);

        for t in &tableaux {
            assert!(t.is_super_standard());
            assert_eq!(t.shape(), &shape);
        }
    }

    #[test]
    fn test_reading_word() {
        let rows = vec![
            vec![SuperTableauEntry::uncircled(1), SuperTableauEntry::uncircled(2)],
            vec![SuperTableauEntry::circled(2)],
        ];
        let t = SuperTableau::new(rows).unwrap();

        let reading = t.reading_word();
        // Bottom to top: [◯2, 1, 2]
        assert_eq!(reading.len(), 3);
        assert_eq!(reading[0], SuperTableauEntry::circled(2));
        assert_eq!(reading[1], SuperTableauEntry::uncircled(1));
        assert_eq!(reading[2], SuperTableauEntry::uncircled(2));
    }

    #[test]
    fn test_super_semistandard_tableaux_generation() {
        let shape = Partition::new(vec![2]);
        let content = vec![
            SuperTableauEntry::uncircled(1),
            SuperTableauEntry::uncircled(2),
        ];

        let tableaux = super_semistandard_tableaux(&shape, content);

        // Should generate: [1 2] only (since 2 1 would not be semistandard)
        assert_eq!(tableaux.len(), 1);
        assert!(tableaux[0].is_super_semistandard());
    }
}
