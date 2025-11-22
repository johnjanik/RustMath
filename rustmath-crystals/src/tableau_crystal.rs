//! Tableau crystals
//!
//! Crystals based on semistandard Young tableaux. These are crystals of type A_n.
//! The crystal operators are defined using the signature rule.

use crate::operators::Crystal;
use crate::weight::Weight;

/// A tableau crystal element
///
/// Represents a semistandard Young tableau (SSYT) where entries
/// are positive integers and increase weakly along rows and strictly down columns.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TableauElement {
    /// The underlying tableau
    pub tableau: Vec<Vec<usize>>,
}

impl TableauElement {
    /// Create a new tableau element
    pub fn new(tableau: Vec<Vec<usize>>) -> Self {
        TableauElement { tableau }
    }

    /// Get the shape of the tableau
    pub fn shape(&self) -> Vec<usize> {
        self.tableau.iter().map(|row| row.len()).collect()
    }

    /// Check if this is a valid semistandard Young tableau
    pub fn is_semistandard(&self) -> bool {
        // Check rows are weakly increasing
        for row in &self.tableau {
            for i in 0..row.len() - 1 {
                if row[i] > row[i + 1] {
                    return false;
                }
            }
        }

        // Check columns are strictly increasing
        for col in 0..self.tableau[0].len() {
            for row in 0..self.tableau.len() - 1 {
                if col < self.tableau[row].len() && col < self.tableau[row + 1].len() {
                    if self.tableau[row][col] >= self.tableau[row + 1][col] {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Compute the reading word (row reading from top to bottom)
    pub fn reading_word(&self) -> Vec<usize> {
        self.tableau.iter().flatten().copied().collect()
    }

    /// Compute the weight of the tableau
    /// Weight coordinates count the number of each letter
    pub fn compute_weight(&self, n: usize) -> Weight {
        let mut coords = vec![0i64; n];
        for row in &self.tableau {
            for &entry in row {
                if entry > 0 && entry <= n {
                    coords[entry - 1] += 1;
                }
            }
        }
        Weight::new(coords)
    }
}

/// A crystal based on semistandard Young tableaux
///
/// This implements the crystal structure on tableaux of a fixed shape,
/// with entries from {1, 2, ..., n}.
#[derive(Debug, Clone)]
pub struct TableauCrystal {
    /// Shape of the tableaux (partition)
    pub shape: Vec<usize>,
    /// Maximum entry (n for type A_{n-1})
    pub n: usize,
}

impl TableauCrystal {
    /// Create a new tableau crystal
    pub fn new(shape: Vec<usize>, n: usize) -> Self {
        TableauCrystal { shape, n }
    }

    /// Compute the signature for entry i
    ///
    /// The signature is computed by scanning the reading word and
    /// marking +1 for i, -1 for i+1, and canceling adjacent pairs.
    fn signature(&self, tableau: &TableauElement, i: usize) -> Vec<i8> {
        let word = tableau.reading_word();
        let mut sig = Vec::new();

        for &entry in &word {
            if entry == i {
                sig.push(1);
            } else if entry == i + 1 {
                sig.push(-1);
            }
        }

        // Cancel adjacent +1, -1 pairs
        loop {
            let mut found = false;
            for j in 0..sig.len().saturating_sub(1) {
                if sig[j] == 1 && sig[j + 1] == -1 {
                    sig.remove(j);
                    sig.remove(j); // Remove twice (indices shift)
                    found = true;
                    break;
                }
            }
            if !found {
                break;
            }
        }

        sig
    }

    /// Find the position to change for e_i
    fn find_ei_position(&self, tableau: &TableauElement, i: usize) -> Option<(usize, usize)> {
        let sig = self.signature(tableau, i);

        // Find the rightmost -1 in the signature (leftmost i+1 to change)
        let mut count = 0;
        let word = tableau.reading_word();

        for idx in (0..word.len()).rev() {
            let entry = word[idx];
            if entry == i + 1 {
                count += 1;
                if count == sig.iter().filter(|&&x| x == -1).count() {
                    // Find this position in the tableau
                    let mut pos = 0;
                    for (row_idx, row) in tableau.tableau.iter().enumerate() {
                        for (col_idx, _) in row.iter().enumerate() {
                            if pos == idx {
                                return Some((row_idx, col_idx));
                            }
                            pos += 1;
                        }
                    }
                }
            }
        }

        None
    }

    /// Find the position to change for f_i
    fn find_fi_position(&self, tableau: &TableauElement, i: usize) -> Option<(usize, usize)> {
        let sig = self.signature(tableau, i);

        // Find the leftmost +1 in the signature (rightmost i to change)
        let mut count = 0;
        let word = tableau.reading_word();

        for (idx, &entry) in word.iter().enumerate().rev() {
            if entry == i {
                count += 1;
                if count == sig.iter().filter(|&&x| x == 1).count() {
                    // Find this position in the tableau
                    let mut pos = 0;
                    for (row_idx, row) in tableau.tableau.iter().enumerate() {
                        for (col_idx, _) in row.iter().enumerate() {
                            if pos == idx {
                                return Some((row_idx, col_idx));
                            }
                            pos += 1;
                        }
                    }
                }
            }
        }

        None
    }

    /// Generate all semistandard tableaux of the given shape
    pub fn all_tableaux(&self) -> Vec<TableauElement> {
        let size: usize = self.shape.iter().sum();
        self.generate_tableaux_recursive(vec![], 0, size)
    }

    fn generate_tableaux_recursive(
        &self,
        current: Vec<Vec<usize>>,
        row_idx: usize,
        remaining: usize,
    ) -> Vec<TableauElement> {
        if row_idx >= self.shape.len() {
            if remaining == 0 {
                return vec![TableauElement::new(current)];
            } else {
                return vec![];
            }
        }

        let row_len = self.shape[row_idx];
        let mut result = Vec::new();

        // Generate all possible rows
        let empty_row = vec![];
        let prev_row = if row_idx > 0 { &current[row_idx - 1] } else { &empty_row };

        self.generate_rows(
            vec![],
            row_len,
            1,
            prev_row,
            &mut |row| {
                let mut next = current.clone();
                next.push(row.clone());
                result.extend(self.generate_tableaux_recursive(
                    next,
                    row_idx + 1,
                    remaining.saturating_sub(row_len),
                ));
            },
        );

        result
    }

    fn generate_rows<F>(
        &self,
        current: Vec<usize>,
        len: usize,
        min_val: usize,
        prev_row: &[usize],
        callback: &mut F,
    ) where
        F: FnMut(&Vec<usize>),
    {
        if current.len() == len {
            callback(&current);
            return;
        }

        let pos = current.len();
        let min_entry = if pos > 0 {
            current[pos - 1] // Weakly increasing in row
        } else {
            min_val
        };

        let max_entry = if pos < prev_row.len() {
            prev_row[pos] - 1 // Strictly less than above
        } else {
            self.n
        };

        for entry in min_entry..=max_entry.min(self.n) {
            if entry == 0 {
                continue;
            }
            let mut next = current.clone();
            next.push(entry);
            self.generate_rows(next, len, min_val, prev_row, callback);
        }
    }
}

impl Crystal for TableauCrystal {
    type Element = TableauElement;

    fn weight(&self, b: &Self::Element) -> Weight {
        b.compute_weight(self.n)
    }

    fn e_i(&self, b: &Self::Element, i: usize) -> Option<Self::Element> {
        if i == 0 || i >= self.n {
            return None;
        }

        // Find position to change i+1 to i
        let pos = self.find_ei_position(b, i)?;
        let mut new_tableau = b.tableau.clone();
        new_tableau[pos.0][pos.1] = i;

        let result = TableauElement::new(new_tableau);
        if result.is_semistandard() {
            Some(result)
        } else {
            None
        }
    }

    fn f_i(&self, b: &Self::Element, i: usize) -> Option<Self::Element> {
        if i == 0 || i >= self.n {
            return None;
        }

        // Find position to change i to i+1
        let pos = self.find_fi_position(b, i)?;
        let mut new_tableau = b.tableau.clone();
        new_tableau[pos.0][pos.1] = i + 1;

        let result = TableauElement::new(new_tableau);
        if result.is_semistandard() {
            Some(result)
        } else {
            None
        }
    }

    fn elements(&self) -> Vec<Self::Element> {
        self.all_tableaux()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tableau_element() {
        let tab = TableauElement::new(vec![vec![1, 2, 2], vec![3, 4]]);
        assert!(tab.is_semistandard());

        let tab2 = TableauElement::new(vec![vec![1, 3, 2], vec![2, 4]]);
        assert!(!tab2.is_semistandard()); // Row not increasing

        let tab3 = TableauElement::new(vec![vec![1, 2, 3], vec![2, 4]]);
        assert!(tab3.is_semistandard()); // This is valid: columns strictly increasing

        let tab4 = TableauElement::new(vec![vec![1, 2], vec![1, 3]]);
        assert!(!tab4.is_semistandard()); // Column not strictly increasing (1 >= 1)
    }

    #[test]
    fn test_tableau_weight() {
        let tab = TableauElement::new(vec![vec![1, 2, 2], vec![3, 4]]);
        let weight = tab.compute_weight(4);
        assert_eq!(weight.coords, vec![1, 2, 1, 1]); // One 1, two 2s, one 3, one 4
    }

    #[test]
    fn test_crystal_operators() {
        let crystal = TableauCrystal::new(vec![2, 1], 3);

        // Simple 2x1 tableau: [[1,2],[3]]
        let tab = TableauElement::new(vec![vec![1, 2], vec![3]]);

        // Try f_1: change 1 to 2
        if let Some(new_tab) = crystal.f_i(&tab, 1) {
            println!("f_1 applied: {:?}", new_tab.tableau);
        }

        // Try e_2: change 3 to 2
        if let Some(new_tab) = crystal.e_i(&tab, 2) {
            println!("e_2 applied: {:?}", new_tab.tableau);
        }
    }
}
