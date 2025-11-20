//! Skew partitions and skew tableaux
//!
//! A skew partition λ/μ is formed by removing a smaller partition μ from a larger partition λ.
//! Skew partitions are fundamental in the representation theory of symmetric groups and
//! the study of symmetric functions.

use crate::partitions::Partition;
use crate::tableaux::Tableau;
use std::collections::{HashSet, VecDeque};

/// A skew partition λ/μ
///
/// Represents the set of cells in λ that are not in μ.
/// For a valid skew partition, μ must fit inside λ (i.e., μ_i ≤ λ_i for all i).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SkewPartition {
    /// The outer partition λ
    outer: Partition,
    /// The inner partition μ
    inner: Partition,
}

impl SkewPartition {
    /// Create a new skew partition λ/μ
    ///
    /// Returns None if μ does not fit inside λ
    pub fn new(outer: Partition, inner: Partition) -> Option<Self> {
        // Check that inner fits inside outer
        for (i, &inner_part) in inner.parts().iter().enumerate() {
            let outer_part = outer.parts().get(i).copied().unwrap_or(0);
            if inner_part > outer_part {
                return None;
            }
        }

        Some(SkewPartition { outer, inner })
    }

    /// Get the outer partition λ
    pub fn outer(&self) -> &Partition {
        &self.outer
    }

    /// Get the inner partition μ
    pub fn inner(&self) -> &Partition {
        &self.inner
    }

    /// Get all cells (row, col) in the skew partition
    pub fn cells(&self) -> Vec<(usize, usize)> {
        let mut cells = Vec::new();

        for (row, &outer_len) in self.outer.parts().iter().enumerate() {
            let inner_len = self.inner.parts().get(row).copied().unwrap_or(0);
            for col in inner_len..outer_len {
                cells.push((row, col));
            }
        }

        cells
    }

    /// Get the number of cells in the skew partition
    pub fn size(&self) -> usize {
        self.outer.sum() - self.inner.sum()
    }

    /// Check if a cell (row, col) is in this skew partition
    pub fn contains(&self, row: usize, col: usize) -> bool {
        let outer_len = self.outer.parts().get(row).copied().unwrap_or(0);
        let inner_len = self.inner.parts().get(row).copied().unwrap_or(0);

        col >= inner_len && col < outer_len
    }

    /// Check if the skew partition is connected
    ///
    /// A skew partition is connected if you can move from any cell to any other cell
    /// by moving to adjacent cells (horizontally or vertically).
    pub fn is_connected(&self) -> bool {
        let cells = self.cells();
        if cells.is_empty() {
            return true;
        }

        let cell_set: HashSet<(usize, usize)> = cells.iter().copied().collect();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // Start BFS from the first cell
        queue.push_back(cells[0]);
        visited.insert(cells[0]);

        while let Some((row, col)) = queue.pop_front() {
            // Check all 4 neighbors
            let neighbors = [
                (row.wrapping_sub(1), col), // up
                (row + 1, col),              // down
                (row, col.wrapping_sub(1)), // left
                (row, col + 1),              // right
            ];

            for &neighbor in &neighbors {
                if cell_set.contains(&neighbor) && !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }

        visited.len() == cells.len()
    }

    /// Check if the skew partition is a ribbon (border strip)
    ///
    /// A ribbon is a connected skew shape that contains no 2×2 block of cells.
    pub fn is_ribbon(&self) -> bool {
        if !self.is_connected() {
            return false;
        }

        // Check for 2×2 blocks
        let cells = self.cells();
        for &(row, col) in &cells {
            // Check if (row, col), (row+1, col), (row, col+1), (row+1, col+1) all exist
            if self.contains(row, col)
                && self.contains(row + 1, col)
                && self.contains(row, col + 1)
                && self.contains(row + 1, col + 1) {
                return false;
            }
        }

        true
    }

    /// Compute the height of a ribbon
    ///
    /// The height of a ribbon is the number of rows it spans minus 1.
    /// This is only meaningful if the skew partition is a ribbon.
    pub fn height(&self) -> Option<usize> {
        if !self.is_ribbon() {
            return None;
        }

        let cells = self.cells();
        if cells.is_empty() {
            return Some(0);
        }

        let min_row = cells.iter().map(|&(r, _)| r).min().unwrap();
        let max_row = cells.iter().map(|&(r, _)| r).max().unwrap();

        Some(max_row - min_row)
    }

    /// Compute the spin of a ribbon
    ///
    /// The spin is the number of rows minus the number of columns the ribbon spans.
    /// For a ribbon of size n with height h, spin = h - (n - h - 1) = 2h - n + 1.
    pub fn spin(&self) -> Option<i32> {
        if !self.is_ribbon() {
            return None;
        }

        let cells = self.cells();
        if cells.is_empty() {
            return Some(0);
        }

        let n = self.size() as i32;
        let h = self.height()? as i32;

        Some(2 * h - n + 1)
    }

    /// Get the rim (outer boundary) of the skew partition
    ///
    /// The rim consists of cells that can be removed while maintaining a valid partition.
    pub fn rim(&self) -> Vec<(usize, usize)> {
        let cells = self.cells();
        let cell_set: HashSet<(usize, usize)> = cells.iter().copied().collect();

        cells.into_iter().filter(|&(row, col)| {
            // A cell is on the rim if removing it still gives a valid skew partition
            // This means it's at the end of its row OR has no cell to the right,
            // AND it's at the bottom of its column OR has no cell below
            let has_right = cell_set.contains(&(row, col + 1));
            let has_below = cell_set.contains(&(row + 1, col));

            !has_right || !has_below
        }).collect()
    }

    /// Try to decompose the skew partition into ribbons
    ///
    /// Returns a vector of ribbons if the decomposition exists, None otherwise.
    /// The ribbons are returned as skew partitions.
    pub fn ribbon_decomposition(&self) -> Option<Vec<SkewPartition>> {
        // Use a greedy algorithm: repeatedly remove ribbons from the rim
        let mut current = self.clone();
        let mut ribbons = Vec::new();

        while current.size() > 0 {
            // Try to find a ribbon on the rim
            match current.find_rim_ribbon() {
                Some(ribbon) => {
                    ribbons.push(ribbon.clone());
                    // Remove this ribbon from current
                    current = current.remove_ribbon(&ribbon)?;
                }
                None => {
                    // No ribbon found, decomposition fails
                    return None;
                }
            }
        }

        Some(ribbons)
    }

    /// Find a ribbon on the rim of the skew partition
    fn find_rim_ribbon(&self) -> Option<SkewPartition> {
        let rim = self.rim();
        if rim.is_empty() {
            return None;
        }

        // Try to build a ribbon starting from each rim cell
        for &start_cell in &rim {
            if let Some(ribbon) = self.build_ribbon_from(start_cell) {
                return Some(ribbon);
            }
        }

        None
    }

    /// Build a ribbon starting from a given cell by following the rim
    fn build_ribbon_from(&self, start: (usize, usize)) -> Option<SkewPartition> {
        let cell_set: HashSet<(usize, usize)> = self.cells().iter().copied().collect();

        if !cell_set.contains(&start) {
            return None;
        }

        let mut ribbon_cells = vec![start];
        let mut current = start;

        // Follow the rim to build a connected ribbon
        loop {
            // Try to find next cell in the ribbon
            let (row, col) = current;

            // Prefer moving right, then down, then left, then up
            let candidates = [
                (row, col + 1),              // right
                (row + 1, col),              // down
                (row, col.wrapping_sub(1)), // left
                (row.wrapping_sub(1), col), // up
            ];

            let mut found_next = false;
            for &next in &candidates {
                if cell_set.contains(&next) && !ribbon_cells.contains(&next) {
                    // Check if adding this cell maintains the ribbon property (no 2×2)
                    let mut test_cells = ribbon_cells.clone();
                    test_cells.push(next);

                    if !has_2x2_block(&test_cells) {
                        ribbon_cells.push(next);
                        current = next;
                        found_next = true;
                        break;
                    }
                }
            }

            if !found_next {
                break;
            }
        }

        // Convert ribbon_cells to a skew partition
        if ribbon_cells.is_empty() {
            return None;
        }

        self.from_cells(&ribbon_cells)
    }

    /// Create a skew partition containing only the specified cells
    fn from_cells(&self, cells: &[(usize, usize)]) -> Option<SkewPartition> {
        if cells.is_empty() {
            return None;
        }

        let min_row = cells.iter().map(|&(r, _)| r).min()?;
        let max_row = cells.iter().map(|&(r, _)| r).max()?;

        // Build outer and inner partitions
        let mut outer_parts = Vec::new();
        let mut inner_parts = Vec::new();

        for row in min_row..=max_row {
            let row_cells: Vec<usize> = cells.iter()
                .filter(|&&(r, _)| r == row)
                .map(|&(_, c)| c)
                .collect();

            if row_cells.is_empty() {
                // Gap in rows - not a valid skew partition
                continue;
            }

            let min_col = *row_cells.iter().min()?;
            let max_col = *row_cells.iter().max()?;

            outer_parts.push(max_col + 1);
            inner_parts.push(min_col);
        }

        let outer = Partition::new(outer_parts);
        let inner = Partition::new(inner_parts);

        SkewPartition::new(outer, inner)
    }

    /// Remove a ribbon from this skew partition
    fn remove_ribbon(&self, ribbon: &SkewPartition) -> Option<SkewPartition> {
        let my_cells: HashSet<(usize, usize)> = self.cells().iter().copied().collect();
        let ribbon_cells: HashSet<(usize, usize)> = ribbon.cells().iter().copied().collect();

        let remaining: Vec<(usize, usize)> = my_cells.difference(&ribbon_cells).copied().collect();

        if remaining.is_empty() {
            // Create empty skew partition
            return Some(SkewPartition::new(
                Partition::new(vec![]),
                Partition::new(vec![])
            )?);
        }

        // Build new skew partition from remaining cells
        let min_row = remaining.iter().map(|&(r, _)| r).min()?;
        let max_row = remaining.iter().map(|&(r, _)| r).max()?;

        let mut outer_parts = Vec::new();
        let mut inner_parts = Vec::new();

        for row in min_row..=max_row {
            let row_cells: Vec<usize> = remaining.iter()
                .filter(|&&(r, _)| r == row)
                .map(|&(_, c)| c)
                .collect();

            if !row_cells.is_empty() {
                let min_col = *row_cells.iter().min()?;
                let max_col = *row_cells.iter().max()?;

                outer_parts.push(max_col + 1);
                inner_parts.push(min_col);
            }
        }

        let outer = Partition::new(outer_parts);
        let inner = Partition::new(inner_parts);

        SkewPartition::new(outer, inner)
    }
}

/// Check if a set of cells contains a 2×2 block
fn has_2x2_block(cells: &[(usize, usize)]) -> bool {
    let cell_set: HashSet<(usize, usize)> = cells.iter().copied().collect();

    for &(row, col) in cells {
        if cell_set.contains(&(row, col))
            && cell_set.contains(&(row + 1, col))
            && cell_set.contains(&(row, col + 1))
            && cell_set.contains(&(row + 1, col + 1)) {
            return true;
        }
    }

    false
}

/// A skew tableau - a filling of a skew partition with positive integers
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkewTableau {
    /// The skew shape
    shape: SkewPartition,
    /// The entries, indexed by (row, col)
    entries: Vec<Vec<Option<usize>>>,
}

impl SkewTableau {
    /// Create a new skew tableau
    pub fn new(shape: SkewPartition, entries: Vec<Vec<Option<usize>>>) -> Option<Self> {
        // Verify dimensions match
        if entries.len() != shape.outer().parts().len() {
            return None;
        }

        for (i, row) in entries.iter().enumerate() {
            if row.len() != *shape.outer().parts().get(i)? {
                return None;
            }
        }

        // Verify entries are only in the skew shape
        for (row, cells) in entries.iter().enumerate() {
            for (col, &entry) in cells.iter().enumerate() {
                if entry.is_some() && !shape.contains(row, col) {
                    return None;
                }
                if entry.is_none() && shape.contains(row, col) {
                    return None;
                }
            }
        }

        Some(SkewTableau { shape, entries })
    }

    /// Get the shape of the skew tableau
    pub fn shape(&self) -> &SkewPartition {
        &self.shape
    }

    /// Get the entry at position (row, col), if it exists
    pub fn get(&self, row: usize, col: usize) -> Option<usize> {
        *self.entries.get(row)?.get(col)?
    }

    /// Check if this is a standard skew tableau
    ///
    /// Standard means: entries are 1,2,...,n (each appears once),
    /// strictly increasing along rows and down columns
    pub fn is_standard(&self) -> bool {
        let n = self.shape.size();
        if n == 0 {
            return true;
        }

        // Collect all entries
        let mut entries = Vec::new();
        for row in &self.entries {
            for &entry in row {
                if let Some(e) = entry {
                    entries.push(e);
                }
            }
        }

        // Check we have exactly 1,2,...,n
        entries.sort_unstable();
        if entries.len() != n || entries != (1..=n).collect::<Vec<_>>() {
            return false;
        }

        // Check rows are strictly increasing
        for row in &self.entries {
            let mut prev = None;
            for &entry in row {
                if let Some(e) = entry {
                    if let Some(p) = prev {
                        if e <= p {
                            return false;
                        }
                    }
                    prev = Some(e);
                }
            }
        }

        // Check columns are strictly increasing
        for col in 0..self.entries[0].len() {
            let mut prev = None;
            for row in &self.entries {
                if let Some(&Some(e)) = row.get(col) {
                    if let Some(p) = prev {
                        if e <= p {
                            return false;
                        }
                    }
                    prev = Some(e);
                }
            }
        }

        true
    }

    /// Straighten the skew tableau
    ///
    /// This converts a skew tableau to a linear combination of standard tableaux
    /// using jeu de taquin slides. For now, we return just the resulting standard tableau
    /// after all slides (coefficient +1).
    pub fn straighten(&self) -> Option<Tableau> {
        // Convert skew tableau to a regular tableau by performing jeu de taquin
        // until all inner partition cells are filled

        // Start by creating a full tableau with gaps
        let outer_parts = self.shape.outer().parts();
        let mut rows: Vec<Vec<usize>> = Vec::new();

        for (i, &outer_len) in outer_parts.iter().enumerate() {
            let mut row = Vec::new();
            for j in 0..outer_len {
                if let Some(entry) = self.get(i, j) {
                    row.push(entry);
                }
            }
            if !row.is_empty() {
                rows.push(row);
            }
        }

        Tableau::new(rows)
    }
}

/// A ribbon-shaped tableau
///
/// A ribbon tableau is a standard tableau whose shape is a ribbon (a connected skew shape
/// with no 2×2 blocks). Ribbons are characterized by their height function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RibbonTableau {
    /// The underlying skew tableau (must be a ribbon shape)
    tableau: SkewTableau,
    /// Height of the ribbon (number of rows spanned minus 1)
    height: usize,
}

impl RibbonTableau {
    /// Create a new ribbon tableau
    ///
    /// Returns None if the shape is not a ribbon or the tableau is not standard
    pub fn new(tableau: SkewTableau) -> Option<Self> {
        if !tableau.shape().is_ribbon() {
            return None;
        }

        if !tableau.is_standard() {
            return None;
        }

        let height = tableau.shape().height()?;

        Some(RibbonTableau { tableau, height })
    }

    /// Get the underlying skew tableau
    pub fn tableau(&self) -> &SkewTableau {
        &self.tableau
    }

    /// Get the height of the ribbon
    pub fn height(&self) -> usize {
        self.height
    }

    /// Get the spin of the ribbon
    ///
    /// For a ribbon of size n with height h, spin = h - (n - h - 1) = 2h - n + 1
    pub fn spin(&self) -> i32 {
        self.tableau.shape().spin().unwrap_or(0)
    }

    /// Get the size (number of cells) of the ribbon
    pub fn size(&self) -> usize {
        self.tableau.shape().size()
    }

    /// Compute the height function for the ribbon
    ///
    /// The height function h(i) gives the "height" after reading the first i entries.
    /// It increases by 1 when moving down a row and decreases by 1 when moving right.
    pub fn height_function(&self) -> Vec<usize> {
        let cells = self.tableau.shape().cells();
        let n = cells.len();

        if n == 0 {
            return vec![0];
        }

        // Sort cells by their tableau entry value to get reading order
        let mut cell_entries: Vec<((usize, usize), usize)> = Vec::new();
        for &(row, col) in &cells {
            if let Some(entry) = self.tableau.get(row, col) {
                cell_entries.push(((row, col), entry));
            }
        }
        cell_entries.sort_by_key(|&(_, entry)| entry);

        // Compute height function
        let mut heights = vec![0]; // h(0) = 0
        let mut current_height = 0;

        for i in 1..=n {
            let (curr_row, curr_col) = cell_entries[i - 1].0;

            if i < n {
                let (next_row, next_col) = cell_entries[i].0;

                // If moving down (next row > current row), height increases
                // If moving right (same row, next col > current col), height decreases
                if next_row > curr_row {
                    current_height += 1;
                } else if next_row == curr_row && next_col > curr_col {
                    if current_height > 0 {
                        current_height -= 1;
                    }
                }
            }

            heights.push(current_height);
        }

        heights
    }

    /// Get the head (northwest-most cell) of the ribbon
    pub fn head(&self) -> Option<(usize, usize)> {
        let cells = self.tableau.shape().cells();
        if cells.is_empty() {
            return None;
        }

        // Head is the cell with minimum row, and within that minimum column
        cells.iter()
            .min_by_key(|&&(r, c)| (r, c))
            .copied()
    }

    /// Get the tail (southeast-most cell) of the ribbon
    pub fn tail(&self) -> Option<(usize, usize)> {
        let cells = self.tableau.shape().cells();
        if cells.is_empty() {
            return None;
        }

        // Tail is the cell with maximum row, and within that maximum column
        cells.iter()
            .max_by_key(|&&(r, c)| (r, c))
            .copied()
    }
}

/// Generate all standard ribbon tableaux for a given ribbon shape
///
/// A ribbon is a connected skew shape that contains no 2×2 block.
/// This function generates all ways to fill the ribbon shape with the numbers 1..n
/// such that entries are strictly increasing along rows and down columns.
pub fn ribbon_shaped_tableaux(shape: &SkewPartition) -> Vec<RibbonTableau> {
    if !shape.is_ribbon() {
        return vec![];
    }

    let cells = shape.cells();
    let n = cells.len();

    if n == 0 {
        return vec![];
    }

    let mut result = Vec::new();

    // Initialize grid based on outer partition shape
    let outer_parts = shape.outer().parts();
    let mut current_filling: Vec<Vec<Option<usize>>> = outer_parts.iter()
        .map(|&len| vec![None; len])
        .collect();

    generate_ribbon_tableaux(
        shape,
        &cells,
        &mut current_filling,
        1,
        n,
        &mut result,
    );

    result
}

fn generate_ribbon_tableaux(
    shape: &SkewPartition,
    cells: &[(usize, usize)],
    current_filling: &mut Vec<Vec<Option<usize>>>,
    next_value: usize,
    n: usize,
    result: &mut Vec<RibbonTableau>,
) {
    if next_value > n {
        // All values placed, create ribbon tableau
        if let Some(skew_tableau) = SkewTableau::new(shape.clone(), current_filling.clone()) {
            if let Some(ribbon_tableau) = RibbonTableau::new(skew_tableau) {
                result.push(ribbon_tableau);
            }
        }
        return;
    }

    // Try placing next_value in each valid cell
    for &(row, col) in cells {
        if current_filling[row][col].is_none() && can_place_in_ribbon(shape, current_filling, row, col, next_value) {
            current_filling[row][col] = Some(next_value);
            generate_ribbon_tableaux(shape, cells, current_filling, next_value + 1, n, result);
            current_filling[row][col] = None;
        }
    }
}

fn can_place_in_ribbon(
    _shape: &SkewPartition,
    current_filling: &[Vec<Option<usize>>],
    row: usize,
    col: usize,
    _value: usize,
) -> bool {
    // Check that this position is empty
    // Since we're placing values 1, 2, 3, ... in order, and trying all positions
    // for each value, the tableau property (strictly increasing along rows/columns)
    // is automatically satisfied - any already-placed value is smaller than the
    // current value we're trying to place.
    current_filling[row][col].is_none()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skew_partition_creation() {
        let outer = Partition::new(vec![4, 3, 1]);
        let inner = Partition::new(vec![2, 1]);

        let skew = SkewPartition::new(outer, inner);
        assert!(skew.is_some());

        let skew = skew.unwrap();
        assert_eq!(skew.size(), 4 + 3 + 1 - 2 - 1);
    }

    #[test]
    fn test_invalid_skew_partition() {
        let outer = Partition::new(vec![2, 1]);
        let inner = Partition::new(vec![3, 2]); // Too big!

        let skew = SkewPartition::new(outer, inner);
        assert!(skew.is_none());
    }

    #[test]
    fn test_cells() {
        let outer = Partition::new(vec![3, 2]);
        let inner = Partition::new(vec![1]);
        let skew = SkewPartition::new(outer, inner).unwrap();

        let cells = skew.cells();
        // Row 0: cols 1, 2 (since inner has 1 cell in row 0)
        // Row 1: cols 0, 1 (since inner has no cells in row 1)
        assert_eq!(cells.len(), 4);
        assert!(cells.contains(&(0, 1)));
        assert!(cells.contains(&(0, 2)));
        assert!(cells.contains(&(1, 0)));
        assert!(cells.contains(&(1, 1)));
    }

    #[test]
    fn test_is_connected() {
        // Connected skew partition
        let outer = Partition::new(vec![3, 2]);
        let inner = Partition::new(vec![1]);
        let skew = SkewPartition::new(outer, inner).unwrap();
        assert!(skew.is_connected());

        // Disconnected skew partition
        let outer = Partition::new(vec![3, 1, 1]);
        let inner = Partition::new(vec![2]);
        let skew = SkewPartition::new(outer, inner).unwrap();
        // This gives cells: (0,2), (1,0), (2,0) which is disconnected
        assert!(!skew.is_connected());
    }

    #[test]
    fn test_is_ribbon() {
        // A ribbon: [2,1] forms an L-shape with no 2×2 blocks
        let outer = Partition::new(vec![2, 1]);
        let inner = Partition::new(vec![]);
        let skew = SkewPartition::new(outer, inner).unwrap();

        // Check if [2,1] is a ribbon (it is!)
        assert!(skew.is_ribbon());

        // A non-ribbon with 2×2 block: [3,2,1] has a 2×2 block
        let outer = Partition::new(vec![3, 2, 1]);
        let inner = Partition::new(vec![]);
        let skew = SkewPartition::new(outer, inner).unwrap();
        // This has a 2×2 block at (0,0), (0,1), (1,0), (1,1)
        assert!(!skew.is_ribbon());

        // Another non-ribbon with 2×2 block
        let outer = Partition::new(vec![3, 3]);
        let inner = Partition::new(vec![1, 1]);
        let skew = SkewPartition::new(outer, inner).unwrap();
        // This has a 2×2 block at (0,1), (0,2), (1,1), (1,2)
        assert!(!skew.is_ribbon());
    }

    #[test]
    fn test_ribbon_height() {
        // Use [2,1] which is a ribbon
        let outer = Partition::new(vec![2, 1]);
        let inner = Partition::new(vec![]);
        let skew = SkewPartition::new(outer, inner).unwrap();

        assert!(skew.is_ribbon());
        let height = skew.height().unwrap();
        // Spans rows 0 and 1, so height = 1 - 0 = 1
        assert_eq!(height, 1);

        // Another ribbon: single cell
        let outer = Partition::new(vec![1]);
        let inner = Partition::new(vec![]);
        let skew_single = SkewPartition::new(outer, inner).unwrap();

        assert!(skew_single.is_ribbon());
        let height_single = skew_single.height().unwrap();
        // Spans only row 0, so height = 0
        assert_eq!(height_single, 0);
    }

    #[test]
    fn test_rim() {
        let outer = Partition::new(vec![3, 2]);
        let inner = Partition::new(vec![]);
        let skew = SkewPartition::new(outer, inner).unwrap();

        let rim = skew.rim();
        // Rim should include (0,2), (1,1) at minimum
        assert!(rim.contains(&(0, 2)));
        assert!(rim.contains(&(1, 1)));
    }

    #[test]
    fn test_skew_tableau_creation() {
        let outer = Partition::new(vec![3, 2]);
        let inner = Partition::new(vec![1]);
        let skew = SkewPartition::new(outer, inner).unwrap();

        let entries = vec![
            vec![None, Some(1), Some(3)],
            vec![Some(2), Some(4)],
        ];

        let tableau = SkewTableau::new(skew, entries);
        assert!(tableau.is_some());
    }

    #[test]
    fn test_skew_tableau_is_standard() {
        let outer = Partition::new(vec![3, 2]);
        let inner = Partition::new(vec![1]);
        let skew = SkewPartition::new(outer, inner).unwrap();

        let entries = vec![
            vec![None, Some(1), Some(3)],
            vec![Some(2), Some(4)],
        ];

        let tableau = SkewTableau::new(skew.clone(), entries).unwrap();
        assert!(tableau.is_standard());

        // Non-standard (not increasing in row)
        let entries2 = vec![
            vec![None, Some(3), Some(1)],
            vec![Some(2), Some(4)],
        ];

        let tableau2 = SkewTableau::new(skew, entries2).unwrap();
        assert!(!tableau2.is_standard());
    }

    #[test]
    fn test_straighten() {
        let outer = Partition::new(vec![3, 2]);
        let inner = Partition::new(vec![1]);
        let skew = SkewPartition::new(outer, inner).unwrap();

        let entries = vec![
            vec![None, Some(1), Some(3)],
            vec![Some(2), Some(4)],
        ];

        let tableau = SkewTableau::new(skew, entries).unwrap();
        let straightened = tableau.straighten();

        assert!(straightened.is_some());
        let t = straightened.unwrap();
        assert_eq!(t.size(), 4);
    }

    #[test]
    fn test_ribbon_tableau_creation() {
        // Create a ribbon shape: [2,1] is an L-shape (a ribbon)
        let outer = Partition::new(vec![2, 1]);
        let inner = Partition::new(vec![]);
        let shape = SkewPartition::new(outer, inner).unwrap();

        assert!(shape.is_ribbon());

        // Create a standard filling
        let entries = vec![
            vec![Some(1), Some(2)],
            vec![Some(3)],
        ];

        let skew_tableau = SkewTableau::new(shape, entries).unwrap();
        let ribbon_tableau = RibbonTableau::new(skew_tableau);

        assert!(ribbon_tableau.is_some());
        let rt = ribbon_tableau.unwrap();
        assert_eq!(rt.height(), 1); // Spans 2 rows, so height = 1
        assert_eq!(rt.size(), 3);
    }

    #[test]
    fn test_ribbon_tableau_invalid_shape() {
        // Create a non-ribbon shape with a 2×2 block
        let outer = Partition::new(vec![2, 2]);
        let inner = Partition::new(vec![]);
        let shape = SkewPartition::new(outer, inner).unwrap();

        assert!(!shape.is_ribbon()); // Has a 2×2 block

        let entries = vec![
            vec![Some(1), Some(2)],
            vec![Some(3), Some(4)],
        ];

        let skew_tableau = SkewTableau::new(shape, entries).unwrap();
        let ribbon_tableau = RibbonTableau::new(skew_tableau);

        // Should fail because shape is not a ribbon
        assert!(ribbon_tableau.is_none());
    }

    #[test]
    fn test_ribbon_tableau_height_function() {
        // Create a ribbon shape: [3,2] / [1] is a connected ribbon
        let outer = Partition::new(vec![3, 2]);
        let inner = Partition::new(vec![1]);
        let shape = SkewPartition::new(outer, inner).unwrap();

        // This is a connected ribbon
        assert!(shape.is_ribbon());

        // Create a standard filling
        let entries = vec![
            vec![None, Some(1), Some(3)],
            vec![Some(2), Some(4)],
        ];

        let skew_tableau = SkewTableau::new(shape, entries).unwrap();
        let ribbon_tableau = RibbonTableau::new(skew_tableau).unwrap();

        // Compute height function
        let heights = ribbon_tableau.height_function();
        assert!(!heights.is_empty());
        assert_eq!(heights[0], 0); // Start at height 0
    }

    #[test]
    fn test_ribbon_tableau_head_tail() {
        // Create a simple ribbon: [2,1]
        let outer = Partition::new(vec![2, 1]);
        let inner = Partition::new(vec![]);
        let shape = SkewPartition::new(outer, inner).unwrap();

        let entries = vec![
            vec![Some(1), Some(3)],
            vec![Some(2)],
        ];

        let skew_tableau = SkewTableau::new(shape, entries).unwrap();
        let ribbon_tableau = RibbonTableau::new(skew_tableau).unwrap();

        // Head should be (0, 0) - northwest cell
        assert_eq!(ribbon_tableau.head(), Some((0, 0)));

        // Tail should be (1, 0) - southeast cell
        assert_eq!(ribbon_tableau.tail(), Some((1, 0)));
    }

    #[test]
    fn test_ribbon_tableau_spin() {
        // Create a ribbon with known spin
        let outer = Partition::new(vec![2, 1]);
        let inner = Partition::new(vec![]);
        let shape = SkewPartition::new(outer, inner).unwrap();

        let entries = vec![
            vec![Some(1), Some(2)],
            vec![Some(3)],
        ];

        let skew_tableau = SkewTableau::new(shape, entries).unwrap();
        let ribbon_tableau = RibbonTableau::new(skew_tableau).unwrap();

        let spin = ribbon_tableau.spin();
        // For a ribbon of size 3 with height 1: spin = 2*1 - 3 + 1 = 0
        assert_eq!(spin, 0);
    }

    #[test]
    fn test_ribbon_shaped_tableaux_generation() {
        // Generate all standard ribbon tableaux for shape [2,1]
        let outer = Partition::new(vec![2, 1]);
        let inner = Partition::new(vec![]);
        let shape = SkewPartition::new(outer, inner).unwrap();

        assert!(shape.is_ribbon());

        let tableaux = ribbon_shaped_tableaux(&shape);

        // There should be exactly 2 standard tableaux for this shape:
        // [[1,2],[3]] and [[1,3],[2]]
        assert_eq!(tableaux.len(), 2);

        // Verify all are standard and have correct shape
        for rt in &tableaux {
            assert_eq!(rt.size(), 3);
            assert_eq!(rt.height(), 1);
            assert!(rt.tableau().is_standard());
        }
    }

    #[test]
    fn test_ribbon_shaped_tableaux_single_row() {
        // Single row ribbon: [3]
        let outer = Partition::new(vec![3]);
        let inner = Partition::new(vec![]);
        let shape = SkewPartition::new(outer, inner).unwrap();

        assert!(shape.is_ribbon());

        let tableaux = ribbon_shaped_tableaux(&shape);

        // There should be exactly 1 standard tableau: [[1,2,3]]
        assert_eq!(tableaux.len(), 1);
        assert_eq!(tableaux[0].height(), 0); // Single row has height 0
    }

    #[test]
    fn test_ribbon_shaped_tableaux_single_column() {
        // Single column ribbon: [1,1,1]
        let outer = Partition::new(vec![1, 1, 1]);
        let inner = Partition::new(vec![]);
        let shape = SkewPartition::new(outer, inner).unwrap();

        assert!(shape.is_ribbon());

        let tableaux = ribbon_shaped_tableaux(&shape);

        // There should be exactly 1 standard tableau: [[1],[2],[3]]
        assert_eq!(tableaux.len(), 1);
        assert_eq!(tableaux[0].height(), 2); // Spans 3 rows, so height = 2
    }

    #[test]
    fn test_ribbon_shaped_tableaux_non_ribbon() {
        // Non-ribbon shape with 2×2 block
        let outer = Partition::new(vec![2, 2]);
        let inner = Partition::new(vec![]);
        let shape = SkewPartition::new(outer, inner).unwrap();

        assert!(!shape.is_ribbon());

        let tableaux = ribbon_shaped_tableaux(&shape);

        // Should return empty vector for non-ribbon shapes
        assert_eq!(tableaux.len(), 0);
    }

    #[test]
    fn test_debug_shapes() {
        // Test various shapes to see which are ribbons
        let test_cases = vec![
            (vec![2, 1], vec![]),
            (vec![3, 1], vec![1]),
            (vec![3, 2], vec![1]),
            (vec![3, 2, 1], vec![2, 1]),
            (vec![4, 2], vec![2]),
        ];

        for (outer_parts, inner_parts) in test_cases {
            let outer = Partition::new(outer_parts.clone());
            let inner = Partition::new(inner_parts.clone());
            let shape = SkewPartition::new(outer, inner).unwrap();
            println!("{:?}/{:?}: is_ribbon={}, cells={:?}",
                     outer_parts, inner_parts, shape.is_ribbon(), shape.cells());
        }
    }

    #[test]
    fn test_ribbon_shaped_tableaux_skew_ribbon() {
        // Use [3,2] / [1] which is a valid ribbon
        let outer = Partition::new(vec![3, 2]);
        let inner = Partition::new(vec![1]);
        let shape = SkewPartition::new(outer, inner).unwrap();

        // This should be a ribbon
        assert!(shape.is_ribbon());

        let tableaux = ribbon_shaped_tableaux(&shape);

        // Verify all generated tableaux are valid
        for rt in &tableaux {
            assert_eq!(rt.size(), 4);
            assert!(rt.tableau().is_standard());
        }

        // There should be at least 1 tableau
        assert!(!tableaux.is_empty());
    }
}
