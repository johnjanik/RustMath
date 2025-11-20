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

    /// Check if this is a semistandard skew tableau
    ///
    /// Semistandard means: rows are weakly increasing (non-decreasing),
    /// columns are strictly increasing
    pub fn is_semistandard(&self) -> bool {
        if self.shape.size() == 0 {
            return true;
        }

        // Check rows are weakly increasing (non-decreasing)
        for row in &self.entries {
            let mut prev = None;
            for &entry in row {
                if let Some(e) = entry {
                    if let Some(p) = prev {
                        if e < p {
                            return false;
                        }
                    }
                    prev = Some(e);
                }
            }
        }

        // Check columns are strictly increasing
        if self.entries.is_empty() {
            return true;
        }

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
        if self.entries.is_empty() {
            return true;
        }

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

    /// Perform a single jeu de taquin forward slide from a specified position
    ///
    /// Schützenberger's jeu de taquin: slide an empty cell towards the outer corner
    /// by repeatedly swapping with the smaller of its right and down neighbors.
    ///
    /// The starting position should be at the boundary of the inner partition (an inner corner).
    /// It will either be empty (part of inner partition) or filled (part of skew shape).
    ///
    /// Returns the new skew tableau after the slide, and the path taken by the empty cell.
    pub fn jeu_de_taquin_slide(&self, start_row: usize, start_col: usize) -> Option<(SkewTableau, Vec<(usize, usize)>)> {
        let mut entries = self.entries.clone();
        let mut path = vec![(start_row, start_col)];
        let mut curr_row = start_row;
        let mut curr_col = start_col;

        // Ensure the entries array is large enough
        while entries.len() <= curr_row {
            entries.push(vec![]);
        }
        while entries[curr_row].len() <= curr_col {
            entries[curr_row].push(None);
        }

        // Mark starting position as empty (it might already be None if in inner partition)
        entries[curr_row][curr_col] = None;

        loop {
            // Check right and down neighbors
            let right_val = if curr_row < entries.len() && curr_col + 1 < entries[curr_row].len() {
                entries[curr_row][curr_col + 1]
            } else {
                None
            };

            let down_val = if curr_row + 1 < entries.len() && curr_col < entries[curr_row + 1].len() {
                entries[curr_row + 1][curr_col]
            } else {
                None
            };

            // Decide which direction to slide based on values
            let slide_direction = match (right_val, down_val) {
                (Some(r), Some(d)) => {
                    // Both exist: choose the smaller value to maintain tableau property
                    if r <= d {
                        Some((0, 1)) // right
                    } else {
                        Some((1, 0)) // down
                    }
                }
                (Some(_), None) => Some((0, 1)), // only right
                (None, Some(_)) => Some((1, 0)), // only down
                (None, None) => None, // reached corner
            };

            match slide_direction {
                Some((dr, dc)) => {
                    let next_row = curr_row + dr;
                    let next_col = curr_col + dc;

                    // Swap the empty cell with the chosen neighbor
                    entries[curr_row][curr_col] = entries[next_row][next_col];
                    entries[next_row][next_col] = None;

                    curr_row = next_row;
                    curr_col = next_col;
                    path.push((curr_row, curr_col));
                }
                None => {
                    // Reached a corner - the slide is complete
                    break;
                }
            }
        }

        // Update the shape after the slide
        // The inner partition shrinks and the outer corner where we ended is removed
        let new_shape = self.compute_shape_after_slide(start_row, start_col, curr_row, curr_col)?;

        // Clean up entries to match new shape
        let cleaned_entries = self.clean_entries_for_shape(&entries, &new_shape);

        Some((SkewTableau::new(new_shape, cleaned_entries)?, path))
    }

    /// Clean up entries array to match a given shape
    fn clean_entries_for_shape(&self, entries: &[Vec<Option<usize>>], shape: &SkewPartition) -> Vec<Vec<Option<usize>>> {
        let mut cleaned = Vec::new();

        for (row_idx, &outer_len) in shape.outer().parts().iter().enumerate() {
            let inner_len = shape.inner().parts().get(row_idx).copied().unwrap_or(0);
            let mut row = vec![None; outer_len];

            // Copy existing values
            if row_idx < entries.len() {
                for col_idx in 0..outer_len {
                    if col_idx < entries[row_idx].len() {
                        if col_idx < inner_len {
                            // This is in the inner partition, should be None
                            row[col_idx] = None;
                        } else {
                            // This is in the skew shape, copy the value
                            row[col_idx] = entries[row_idx][col_idx];
                        }
                    }
                }
            }

            cleaned.push(row);
        }

        cleaned
    }

    /// Compute the new skew shape after a jeu de taquin slide
    fn compute_shape_after_slide(&self, start_row: usize, start_col: usize, end_row: usize, end_col: usize) -> Option<SkewPartition> {
        let mut outer_parts = self.shape.outer().parts().to_vec();
        let mut inner_parts = self.shape.inner().parts().to_vec();

        // Ensure inner_parts is long enough
        while inner_parts.len() <= start_row {
            inner_parts.push(0);
        }

        // Reduce the inner partition at the starting position
        if inner_parts[start_row] > 0 && inner_parts[start_row] == start_col + 1 {
            inner_parts[start_row] -= 1;
        }

        // Reduce the outer partition at the ending position
        if end_row < outer_parts.len() && outer_parts[end_row] > 0 && outer_parts[end_row] == end_col + 1 {
            outer_parts[end_row] -= 1;
        }

        // Remove trailing zeros
        while outer_parts.last() == Some(&0) {
            outer_parts.pop();
        }
        while inner_parts.last() == Some(&0) {
            inner_parts.pop();
        }

        let new_outer = Partition::new(outer_parts);
        let new_inner = Partition::new(inner_parts);

        SkewPartition::new(new_outer, new_inner)
    }

    /// Compute the new skew shape after removing a corner cell via jeu de taquin
    fn compute_shape_after_removal(&self, row: usize, col: usize) -> Option<SkewPartition> {
        let outer_parts = self.shape.outer().parts().to_vec();
        let mut inner_parts = self.shape.inner().parts().to_vec();

        // When jeu de taquin completes, we reduce the inner partition at the position
        // where the slide started, and the outer partition at the position where it ended

        // For now, use a simpler approach: reduce the first non-zero inner partition entry
        for i in 0..inner_parts.len() {
            if inner_parts[i] > 0 {
                inner_parts[i] -= 1;
                break;
            }
        }

        // Remove trailing zeros from inner
        while inner_parts.last() == Some(&0) {
            inner_parts.pop();
        }

        let new_outer = Partition::new(outer_parts);
        let new_inner = Partition::new(inner_parts);

        SkewPartition::new(new_outer, new_inner)
    }

    /// Rectify the skew tableau using jeu de taquin
    ///
    /// Repeatedly perform jeu de taquin slides from the inner corners
    /// until we obtain a standard (non-skew) tableau.
    ///
    /// This is the main operation for Schützenberger's jeu de taquin algorithm.
    pub fn rectify(&self) -> Option<Tableau> {
        // Start with the current tableau
        let mut current = self.clone();
        let initial_inner_size = current.shape.inner().sum();

        // Safety limit to prevent infinite loops
        let max_iterations = initial_inner_size + 10;
        let mut iterations = 0;

        // Keep sliding until the inner partition is empty
        while current.shape.inner().sum() > 0 {
            if iterations >= max_iterations {
                // Something went wrong, bail out
                return None;
            }

            // Find an inner corner to slide from
            let inner_corner = current.find_inner_corner()?;

            // Perform jeu de taquin slide from this corner
            let (new_tableau, _path) = current.jeu_de_taquin_slide(inner_corner.0, inner_corner.1)?;
            current = new_tableau;
            iterations += 1;
        }

        // Convert to a standard Tableau
        current.to_tableau()
    }

    /// Find an inner corner of the skew shape for jeu de taquin
    ///
    /// An inner corner is a removable corner cell of the inner partition.
    /// It should be a cell where removing it maintains a valid partition shape.
    fn find_inner_corner(&self) -> Option<(usize, usize)> {
        let inner_parts = self.shape.inner().parts();

        // If no inner partition, we're done
        if inner_parts.is_empty() || inner_parts.iter().all(|&x| x == 0) {
            return None;
        }

        // Find a removable corner of the inner partition
        // This is the rightmost cell of the bottom-most row that can be removed
        // while maintaining a valid partition
        for (row, &inner_len) in inner_parts.iter().enumerate().rev() {
            if inner_len > 0 {
                let col = inner_len - 1;

                // Check if this is a removable corner
                // It's removable if the row below (if it exists) has fewer cells
                let is_removable = if row + 1 < inner_parts.len() {
                    col >= inner_parts[row + 1]
                } else {
                    true
                };

                if is_removable {
                    return Some((row, col));
                }
            }
        }

        // If we didn't find a proper corner, just use the first non-empty cell
        for (row, &inner_len) in inner_parts.iter().enumerate() {
            if inner_len > 0 {
                return Some((row, inner_len - 1));
            }
        }

        None
    }

    /// Convert to a standard Tableau (only works if inner partition is empty)
    fn to_tableau(&self) -> Option<Tableau> {
        if self.shape.inner().sum() > 0 {
            return None;
        }

        let mut rows: Vec<Vec<usize>> = Vec::new();

        for row in &self.entries {
            let row_entries: Vec<usize> = row.iter()
                .filter_map(|&x| x)
                .collect();

            if !row_entries.is_empty() {
                rows.push(row_entries);
            }
        }

        Tableau::new(rows)
    }

    /// Perform inverse jeu de taquin slide
    ///
    /// This is the reverse operation: start with an empty cell at an outer corner
    /// and slide it towards the inner partition.
    pub fn inverse_jdt_slide(&self, corner_row: usize, corner_col: usize) -> Option<(SkewTableau, Vec<(usize, usize)>)> {
        // Verify this is an outer corner
        if !self.is_outer_corner(corner_row, corner_col) {
            return None;
        }

        let mut entries = self.entries.clone();
        let mut path = vec![(corner_row, corner_col)];
        let mut curr_row = corner_row;
        let mut curr_col = corner_col;

        // Mark current position as empty
        entries[curr_row][curr_col] = None;

        loop {
            // Check left and up neighbors
            let left_val = if curr_col > 0 {
                entries[curr_row].get(curr_col - 1).and_then(|&x| x)
            } else {
                None
            };

            let up_val = if curr_row > 0 {
                entries.get(curr_row - 1)
                    .and_then(|row| row.get(curr_col))
                    .and_then(|&x| x)
            } else {
                None
            };

            // Decide which direction to slide (choose larger to maintain tableau property in reverse)
            let slide_direction = match (left_val, up_val) {
                (Some(l), Some(u)) => {
                    if l >= u {
                        Some((0, -1)) // left
                    } else {
                        Some((-1, 0)) // up
                    }
                }
                (Some(_), None) => Some((0, -1)), // only left
                (None, Some(_)) => Some((-1, 0)), // only up
                (None, None) => None, // reached inner corner
            };

            match slide_direction {
                Some((dr, dc)) => {
                    let next_row = (curr_row as i32 + dr) as usize;
                    let next_col = (curr_col as i32 + dc) as usize;

                    // Swap the empty cell with the chosen neighbor
                    entries[curr_row][curr_col] = entries[next_row][next_col];
                    entries[next_row][next_col] = None;

                    curr_row = next_row;
                    curr_col = next_col;
                    path.push((curr_row, curr_col));
                }
                None => {
                    // Reached inner corner
                    break;
                }
            }
        }

        // Update shape to add an inner corner
        let new_shape = self.compute_shape_after_addition(curr_row, curr_col)?;

        Some((SkewTableau::new(new_shape, entries)?, path))
    }

    /// Check if a position is an outer corner
    fn is_outer_corner(&self, row: usize, col: usize) -> bool {
        if !self.shape.contains(row, col) {
            return false;
        }

        // An outer corner has no cell to the right and no cell below
        let no_right = !self.shape.contains(row, col + 1);
        let no_below = !self.shape.contains(row + 1, col);

        no_right && no_below
    }

    /// Compute the new skew shape after adding an inner corner
    fn compute_shape_after_addition(&self, row: usize, col: usize) -> Option<SkewPartition> {
        let outer_parts = self.shape.outer().parts().to_vec();
        let mut inner_parts = self.shape.inner().parts().to_vec();

        // Extend inner_parts if needed
        while inner_parts.len() <= row {
            inner_parts.push(0);
        }

        // Add to the inner partition
        inner_parts[row] = col + 1;

        let new_outer = Partition::new(outer_parts);
        let new_inner = Partition::new(inner_parts);

        SkewPartition::new(new_outer, new_inner)
    }

    /// Display the skew tableau as a string
    pub fn to_string(&self) -> String {
        let mut result = String::new();

        for (row_idx, row) in self.entries.iter().enumerate() {
            for (col_idx, &entry) in row.iter().enumerate() {
                if let Some(val) = entry {
                    result.push_str(&format!("{:3}", val));
                } else if self.shape.contains(row_idx, col_idx) {
                    result.push_str("  .");
                } else {
                    result.push_str("   ");
                }
            }
            result.push('\n');
        }

        result
    }

    /// Straighten the skew tableau (alias for rectify)
    ///
    /// This converts a skew tableau to a linear combination of standard tableaux
    /// using jeu de taquin slides. For now, we return just the resulting standard tableau
    /// after all slides (coefficient +1).
    #[deprecated(note = "Use rectify() instead")]
    pub fn straighten(&self) -> Option<Tableau> {
        self.rectify()
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
    fn test_skew_tableau_is_semistandard() {
        let outer = Partition::new(vec![3, 2]);
        let inner = Partition::new(vec![1]);
        let skew = SkewPartition::new(outer, inner).unwrap();

        // Semistandard with repeated values
        let entries = vec![
            vec![None, Some(1), Some(1)],
            vec![Some(2), Some(3)],
        ];

        let tableau = SkewTableau::new(skew.clone(), entries).unwrap();
        assert!(tableau.is_semistandard());
        assert!(!tableau.is_standard()); // Not standard due to repeated 1s

        // Not semistandard (decreasing in row)
        let entries2 = vec![
            vec![None, Some(3), Some(1)],
            vec![Some(2), Some(4)],
        ];

        let tableau2 = SkewTableau::new(skew, entries2).unwrap();
        assert!(!tableau2.is_semistandard());
    }

    #[test]
    fn test_jeu_de_taquin_slide() {
        // Create a skew tableau: shape (3,2)/(1,0)
        // Layout:
        //   . 1 3
        //   2 4
        let outer = Partition::new(vec![3, 2]);
        let inner = Partition::new(vec![1]);
        let skew = SkewPartition::new(outer, inner).unwrap();

        let entries = vec![
            vec![None, Some(1), Some(3)],
            vec![Some(2), Some(4)],
        ];

        let tableau = SkewTableau::new(skew, entries).unwrap();

        // Perform jeu de taquin slide from the inner corner (0, 0)
        let result = tableau.jeu_de_taquin_slide(0, 0);
        assert!(result.is_some());

        let (new_tableau, path) = result.unwrap();

        // Check that the path is valid
        assert!(!path.is_empty());
        assert_eq!(path[0], (0, 0));

        // Check that the resulting tableau is still semistandard
        assert!(new_tableau.is_semistandard() || new_tableau.shape.size() < tableau.shape.size());
    }

    #[test]
    fn test_rectify_skew_tableau() {
        // Create a simple skew tableau that should rectify to a standard tableau
        let outer = Partition::new(vec![3, 2]);
        let inner = Partition::new(vec![1]);
        let skew = SkewPartition::new(outer, inner).unwrap();

        let entries = vec![
            vec![None, Some(1), Some(3)],
            vec![Some(2), Some(4)],
        ];

        let tableau = SkewTableau::new(skew, entries).unwrap();
        let rectified = tableau.rectify();

        assert!(rectified.is_some());
        let t = rectified.unwrap();

        // The rectified tableau should have 4 cells
        assert_eq!(t.size(), 4);

        // It should be semistandard
        assert!(t.is_semistandard());
    }

    #[test]
    fn test_find_inner_corner() {
        let outer = Partition::new(vec![4, 3, 1]);
        let inner = Partition::new(vec![2, 1]);
        let skew = SkewPartition::new(outer, inner).unwrap();

        let entries = vec![
            vec![None, None, Some(1), Some(2)],
            vec![None, Some(3), Some(4)],
            vec![Some(5)],
        ];

        let tableau = SkewTableau::new(skew, entries).unwrap();
        let corner = tableau.find_inner_corner();

        assert!(corner.is_some());
        // Should find an inner corner
        let (row, col) = corner.unwrap();
        assert!(row < 2 && col < 2);
    }

    #[test]
    fn test_inverse_jdt_slide() {
        // Create a regular (non-skew) tableau
        let outer = Partition::new(vec![2, 1]);
        let inner = Partition::new(vec![]);
        let skew = SkewPartition::new(outer, inner).unwrap();

        let entries = vec![
            vec![Some(1), Some(3)],
            vec![Some(2)],
        ];

        let tableau = SkewTableau::new(skew, entries).unwrap();

        // Find an outer corner
        // (1, 0) should be an outer corner
        assert!(tableau.is_outer_corner(1, 0));

        // Perform inverse jeu de taquin
        let result = tableau.inverse_jdt_slide(1, 0);
        assert!(result.is_some());

        let (new_tableau, path) = result.unwrap();

        // Check that the path is valid
        assert!(!path.is_empty());
        assert_eq!(path[0], (1, 0));

        // The new tableau should have a larger inner partition
        assert!(new_tableau.shape.inner().sum() > tableau.shape.inner().sum());
    }

    #[test]
    fn test_to_string_display() {
        let outer = Partition::new(vec![3, 2]);
        let inner = Partition::new(vec![1]);
        let skew = SkewPartition::new(outer, inner).unwrap();

        let entries = vec![
            vec![None, Some(1), Some(3)],
            vec![Some(2), Some(4)],
        ];

        let tableau = SkewTableau::new(skew, entries).unwrap();
        let display = tableau.to_string();

        // Should contain the values
        assert!(display.contains('1'));
        assert!(display.contains('2'));
        assert!(display.contains('3'));
        assert!(display.contains('4'));
    }

    #[test]
    fn test_jdt_preserves_semistandard() {
        // Create a semistandard skew tableau
        let outer = Partition::new(vec![4, 3, 2]);
        let inner = Partition::new(vec![2, 1]);
        let skew = SkewPartition::new(outer, inner).unwrap();

        let entries = vec![
            vec![None, None, Some(1), Some(2)],
            vec![None, Some(2), Some(3)],
            vec![Some(3), Some(4)],
        ];

        let tableau = SkewTableau::new(skew, entries).unwrap();
        assert!(tableau.is_semistandard());

        // Perform jeu de taquin slide
        let corner = tableau.find_inner_corner().unwrap();
        let result = tableau.jeu_de_taquin_slide(corner.0, corner.1);

        assert!(result.is_some());
        let (new_tableau, _) = result.unwrap();

        // The result should still be semistandard
        assert!(new_tableau.is_semistandard());
    }

    #[test]
    fn test_rectify_produces_valid_tableau() {
        // Create a skew tableau with a larger inner partition
        let outer = Partition::new(vec![5, 4, 3]);
        let inner = Partition::new(vec![2, 1]);
        let skew = SkewPartition::new(outer, inner).unwrap();

        let entries = vec![
            vec![None, None, Some(1), Some(2), Some(5)],
            vec![None, Some(2), Some(3), Some(6)],
            vec![Some(4), Some(5), Some(7)],
        ];

        let tableau = SkewTableau::new(skew, entries).unwrap();
        let rectified = tableau.rectify();

        assert!(rectified.is_some());
        let t = rectified.unwrap();

        // Should have the correct number of cells
        assert_eq!(t.size(), 9);

        // Should be semistandard
        assert!(t.is_semistandard());

        // Shape should be a valid partition
        assert!(t.shape().parts().len() > 0);
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

        #[allow(deprecated)]
        let straightened = tableau.straighten();

        assert!(straightened.is_some());
        let t = straightened.unwrap();
        assert_eq!(t.size(), 4);
    }
}
