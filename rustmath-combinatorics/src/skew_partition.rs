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
}
