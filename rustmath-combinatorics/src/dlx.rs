//! Dancing Links (DLX) - Implementation of Knuth's Algorithm X for exact cover problems
//!
//! This module implements Donald Knuth's Dancing Links technique for efficiently solving
//! exact cover problems using backtracking with constraint propagation.
//!
//! # Exact Cover Problem
//!
//! Given a matrix of 0s and 1s, find a subset of rows such that each column contains
//! exactly one 1 in the selected rows.
//!
//! # Applications
//!
//! - Sudoku solving
//! - N-Queens problem
//! - Pentomino tiling
//! - Latin square construction
//! - Graph coloring with constraints
//!
//! # Example
//!
//! ```
//! use rustmath_combinatorics::dlx::DancingLinks;
//!
//! // Create a simple exact cover problem:
//! //   A B C D E F G
//! // 1 [1 0 0 1 0 0 1]
//! // 2 [1 0 0 1 0 0 0]
//! // 3 [0 0 0 1 1 0 1]
//! // 4 [0 0 1 0 1 1 0]
//! // 5 [0 1 1 0 0 1 1]
//! // 6 [0 1 0 0 0 0 1]
//!
//! let mut dlx = DancingLinks::new(7);
//! dlx.add_row(vec![0, 3, 6]); // Row 1
//! dlx.add_row(vec![0, 3]);    // Row 2
//! dlx.add_row(vec![3, 4, 6]); // Row 3
//! dlx.add_row(vec![2, 4, 5]); // Row 4
//! dlx.add_row(vec![1, 2, 5, 6]); // Row 5
//! dlx.add_row(vec![1, 6]);    // Row 6
//!
//! let solutions = dlx.solve_all();
//! assert_eq!(solutions.len(), 1);
//! // Solution uses rows 2, 4, 6 (0-indexed: rows 1, 3, 5)
//! ```

use std::cell::RefCell;
use std::rc::Rc;

/// A node in the dancing links structure
#[derive(Debug)]
struct DLXNode {
    /// Row index (for data nodes)
    row: usize,
    /// Column header this node belongs to
    column: Option<Rc<RefCell<ColumnHeader>>>,
    /// Links to neighboring nodes
    left: Option<Rc<RefCell<DLXNode>>>,
    right: Option<Rc<RefCell<DLXNode>>>,
    up: Option<Rc<RefCell<DLXNode>>>,
    down: Option<Rc<RefCell<DLXNode>>>,
}

impl DLXNode {
    /// Create a new DLX node
    fn new(row: usize) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(DLXNode {
            row,
            column: None,
            left: None,
            right: None,
            up: None,
            down: None,
        }))
    }

    /// Create a circular self-referencing node (for headers)
    fn new_circular(row: usize) -> Rc<RefCell<Self>> {
        let node = Rc::new(RefCell::new(DLXNode {
            row,
            column: None,
            left: None,
            right: None,
            up: None,
            down: None,
        }));

        // Make it circular
        node.borrow_mut().left = Some(Rc::clone(&node));
        node.borrow_mut().right = Some(Rc::clone(&node));
        node.borrow_mut().up = Some(Rc::clone(&node));
        node.borrow_mut().down = Some(Rc::clone(&node));

        node
    }
}

/// A column header in the dancing links structure
#[derive(Debug)]
struct ColumnHeader {
    /// Column index
    index: usize,
    /// Number of nodes in this column
    size: usize,
    /// The header node for this column
    node: Rc<RefCell<DLXNode>>,
}

impl ColumnHeader {
    /// Create a new column header
    fn new(index: usize) -> Rc<RefCell<Self>> {
        let node = DLXNode::new_circular(usize::MAX); // Use MAX as sentinel for header
        let header = Rc::new(RefCell::new(ColumnHeader {
            index,
            size: 0,
            node: Rc::clone(&node),
        }));

        node.borrow_mut().column = Some(Rc::clone(&header));
        header
    }
}

/// Dancing Links solver for exact cover problems
///
/// Uses Knuth's Algorithm X with the Dancing Links technique for efficient
/// backtracking and constraint propagation.
pub struct DancingLinks {
    /// The root node (master header)
    root: Rc<RefCell<DLXNode>>,
    /// Column headers
    columns: Vec<Rc<RefCell<ColumnHeader>>>,
    /// Number of columns
    num_columns: usize,
    /// Row data for result reporting
    rows: Vec<Vec<usize>>,
}

impl DancingLinks {
    /// Create a new DancingLinks solver for an exact cover problem with the given number of columns
    ///
    /// # Arguments
    ///
    /// * `num_columns` - The number of columns (constraints) in the exact cover problem
    pub fn new(num_columns: usize) -> Self {
        let root = DLXNode::new_circular(usize::MAX);
        let mut columns = Vec::with_capacity(num_columns);

        // Create column headers and link them horizontally
        let mut prev = Rc::clone(&root);
        for i in 0..num_columns {
            let col = ColumnHeader::new(i);
            let col_node = Rc::clone(&col.borrow().node);

            // Link horizontally
            prev.borrow_mut().right = Some(Rc::clone(&col_node));
            col_node.borrow_mut().left = Some(prev);

            columns.push(col);
            prev = col_node;
        }

        // Close the circular list
        prev.borrow_mut().right = Some(Rc::clone(&root));
        root.borrow_mut().left = Some(prev);

        DancingLinks {
            root,
            columns,
            num_columns,
            rows: Vec::new(),
        }
    }

    /// Add a row to the exact cover problem
    ///
    /// # Arguments
    ///
    /// * `columns` - A vector of column indices where this row has 1s
    ///
    /// # Panics
    ///
    /// Panics if any column index is >= num_columns
    pub fn add_row(&mut self, columns: Vec<usize>) {
        if columns.is_empty() {
            return;
        }

        // Validate column indices
        for &col in &columns {
            assert!(
                col < self.num_columns,
                "Column index {} out of bounds (num_columns={})",
                col,
                self.num_columns
            );
        }

        let row_index = self.rows.len();
        self.rows.push(columns.clone());

        let mut row_nodes = Vec::new();

        // Create nodes for each column in this row
        for &col_idx in &columns {
            let node = DLXNode::new(row_index);
            let col_header = Rc::clone(&self.columns[col_idx]);

            // Link vertically into the column
            node.borrow_mut().column = Some(Rc::clone(&col_header));

            let col_node = Rc::clone(&col_header.borrow().node);
            let up_node = col_node.borrow().up.as_ref().unwrap().clone();

            up_node.borrow_mut().down = Some(Rc::clone(&node));
            node.borrow_mut().up = Some(up_node);
            node.borrow_mut().down = Some(Rc::clone(&col_node));
            col_node.borrow_mut().up = Some(Rc::clone(&node));

            // Increase column size
            col_header.borrow_mut().size += 1;

            row_nodes.push(node);
        }

        // Link row nodes horizontally in a circular list
        for i in 0..row_nodes.len() {
            let curr = &row_nodes[i];
            let next = &row_nodes[(i + 1) % row_nodes.len()];
            let prev = &row_nodes[(i + row_nodes.len() - 1) % row_nodes.len()];

            curr.borrow_mut().right = Some(Rc::clone(next));
            curr.borrow_mut().left = Some(Rc::clone(prev));
        }
    }

    /// Cover a column (remove it and all rows that have 1s in this column)
    fn cover(&self, col_header: &Rc<RefCell<ColumnHeader>>) {
        let col_node = Rc::clone(&col_header.borrow().node);

        // Remove column header from the header list
        let left = col_node.borrow().left.as_ref().unwrap().clone();
        let right = col_node.borrow().right.as_ref().unwrap().clone();

        left.borrow_mut().right = Some(Rc::clone(&right));
        right.borrow_mut().left = Some(left);

        // For each row in this column
        let mut curr_row = col_node.borrow().down.as_ref().unwrap().clone();
        while !Rc::ptr_eq(&curr_row, &col_node) {
            // For each node in this row
            let mut curr_col = curr_row.borrow().right.as_ref().unwrap().clone();
            while !Rc::ptr_eq(&curr_col, &curr_row) {
                // Remove this node from its column
                let up = curr_col.borrow().up.as_ref().unwrap().clone();
                let down = curr_col.borrow().down.as_ref().unwrap().clone();

                up.borrow_mut().down = Some(Rc::clone(&down));
                down.borrow_mut().up = Some(up);

                // Decrease column size
                if let Some(col) = &curr_col.borrow().column {
                    col.borrow_mut().size -= 1;
                }

                let next_col = curr_col.borrow().right.as_ref().unwrap().clone();
                curr_col = next_col;
            }

            let next_row = curr_row.borrow().down.as_ref().unwrap().clone();
            curr_row = next_row;
        }
    }

    /// Uncover a column (restore it and all rows that have 1s in this column)
    fn uncover(&self, col_header: &Rc<RefCell<ColumnHeader>>) {
        let col_node = Rc::clone(&col_header.borrow().node);

        // For each row in this column (in reverse order)
        let mut curr_row = col_node.borrow().up.as_ref().unwrap().clone();
        while !Rc::ptr_eq(&curr_row, &col_node) {
            // For each node in this row (in reverse order)
            let mut curr_col = curr_row.borrow().left.as_ref().unwrap().clone();
            while !Rc::ptr_eq(&curr_col, &curr_row) {
                // Restore this node to its column
                let up = curr_col.borrow().up.as_ref().unwrap().clone();
                let down = curr_col.borrow().down.as_ref().unwrap().clone();

                up.borrow_mut().down = Some(Rc::clone(&curr_col));
                down.borrow_mut().up = Some(Rc::clone(&curr_col));

                // Increase column size
                if let Some(col) = &curr_col.borrow().column {
                    col.borrow_mut().size += 1;
                }

                let next_col = curr_col.borrow().left.as_ref().unwrap().clone();
                curr_col = next_col;
            }

            let next_row = curr_row.borrow().up.as_ref().unwrap().clone();
            curr_row = next_row;
        }

        // Restore column header to the header list
        let left = col_node.borrow().left.as_ref().unwrap().clone();
        let right = col_node.borrow().right.as_ref().unwrap().clone();

        left.borrow_mut().right = Some(Rc::clone(&col_node));
        right.borrow_mut().left = Some(Rc::clone(&col_node));
    }

    /// Choose a column with the minimum number of nodes (heuristic for faster solving)
    fn choose_column(&self) -> Option<Rc<RefCell<ColumnHeader>>> {
        let mut curr = self.root.borrow().right.as_ref()?.clone();

        if Rc::ptr_eq(&curr, &self.root) {
            return None; // No columns left
        }

        let mut min_col = curr.borrow().column.as_ref()?.clone();
        let mut min_size = min_col.borrow().size;

        let next = curr.borrow().right.as_ref()?.clone();
        curr = next;
        while !Rc::ptr_eq(&curr, &self.root) {
            if let Some(col) = &curr.borrow().column {
                let size = col.borrow().size;
                if size < min_size {
                    min_size = size;
                    min_col = Rc::clone(col);
                }
            }

            let next = curr.borrow().right.as_ref()?.clone();
            curr = next;
        }

        Some(min_col)
    }

    /// Recursive search function implementing Algorithm X
    fn search(&self, solution: &mut Vec<usize>, all_solutions: &mut Vec<Vec<usize>>) {
        // If no columns remain, we have a solution
        if Rc::ptr_eq(
            self.root.borrow().right.as_ref().unwrap(),
            &self.root,
        ) {
            all_solutions.push(solution.clone());
            return;
        }

        // Choose a column (preferably with minimum size)
        let col = match self.choose_column() {
            Some(c) => c,
            None => return,
        };

        // If chosen column is empty, no solution exists in this branch
        if col.borrow().size == 0 {
            return;
        }

        self.cover(&col);

        // Try each row in this column
        let col_node = Rc::clone(&col.borrow().node);
        let mut curr_row = col_node.borrow().down.as_ref().unwrap().clone();

        while !Rc::ptr_eq(&curr_row, &col_node) {
            let row_index = curr_row.borrow().row;
            solution.push(row_index);

            // Cover all columns in this row
            let mut curr_col = curr_row.borrow().right.as_ref().unwrap().clone();
            while !Rc::ptr_eq(&curr_col, &curr_row) {
                if let Some(c) = &curr_col.borrow().column {
                    self.cover(c);
                }
                let next_col = curr_col.borrow().right.as_ref().unwrap().clone();
                curr_col = next_col;
            }

            // Recursively search
            self.search(solution, all_solutions);

            // Backtrack: uncover all columns in this row
            solution.pop();
            let mut curr_col = curr_row.borrow().left.as_ref().unwrap().clone();
            while !Rc::ptr_eq(&curr_col, &curr_row) {
                if let Some(c) = &curr_col.borrow().column {
                    self.uncover(c);
                }
                let next_col = curr_col.borrow().left.as_ref().unwrap().clone();
                curr_col = next_col;
            }

            let next_row = curr_row.borrow().down.as_ref().unwrap().clone();
            curr_row = next_row;
        }

        self.uncover(&col);
    }

    /// Solve the exact cover problem and return all solutions
    ///
    /// # Returns
    ///
    /// A vector of solutions, where each solution is a vector of row indices
    pub fn solve_all(&self) -> Vec<Vec<usize>> {
        let mut solution = Vec::new();
        let mut all_solutions = Vec::new();
        self.search(&mut solution, &mut all_solutions);
        all_solutions
    }

    /// Solve the exact cover problem and return the first solution found
    ///
    /// # Returns
    ///
    /// An optional vector of row indices representing the first solution, or None if no solution exists
    pub fn solve_one(&self) -> Option<Vec<usize>> {
        let mut solution = Vec::new();
        let mut all_solutions = Vec::new();
        self.search_one(&mut solution, &mut all_solutions);
        all_solutions.into_iter().next()
    }

    /// Recursive search that stops after finding one solution
    fn search_one(&self, solution: &mut Vec<usize>, all_solutions: &mut Vec<Vec<usize>>) {
        // If we already found a solution, stop
        if !all_solutions.is_empty() {
            return;
        }

        // If no columns remain, we have a solution
        if Rc::ptr_eq(
            self.root.borrow().right.as_ref().unwrap(),
            &self.root,
        ) {
            all_solutions.push(solution.clone());
            return;
        }

        // Choose a column (preferably with minimum size)
        let col = match self.choose_column() {
            Some(c) => c,
            None => return,
        };

        // If chosen column is empty, no solution exists in this branch
        if col.borrow().size == 0 {
            return;
        }

        self.cover(&col);

        // Try each row in this column
        let col_node = Rc::clone(&col.borrow().node);
        let mut curr_row = col_node.borrow().down.as_ref().unwrap().clone();

        while !Rc::ptr_eq(&curr_row, &col_node) && all_solutions.is_empty() {
            let row_index = curr_row.borrow().row;
            solution.push(row_index);

            // Cover all columns in this row
            let mut curr_col = curr_row.borrow().right.as_ref().unwrap().clone();
            while !Rc::ptr_eq(&curr_col, &curr_row) {
                if let Some(c) = &curr_col.borrow().column {
                    self.cover(c);
                }
                let next_col = curr_col.borrow().right.as_ref().unwrap().clone();
                curr_col = next_col;
            }

            // Recursively search
            self.search_one(solution, all_solutions);

            // Backtrack: uncover all columns in this row
            if all_solutions.is_empty() {
                solution.pop();
                let mut curr_col = curr_row.borrow().left.as_ref().unwrap().clone();
                while !Rc::ptr_eq(&curr_col, &curr_row) {
                    if let Some(c) = &curr_col.borrow().column {
                        self.uncover(c);
                    }
                    let next_col = curr_col.borrow().left.as_ref().unwrap().clone();
                    curr_col = next_col;
                }
            }

            let next_row = curr_row.borrow().down.as_ref().unwrap().clone();
            curr_row = next_row;
        }

        self.uncover(&col);
    }

    /// Get the number of columns in the problem
    pub fn num_columns(&self) -> usize {
        self.num_columns
    }

    /// Get the number of rows added to the problem
    pub fn num_rows(&self) -> usize {
        self.rows.len()
    }

    /// Get the column indices for a specific row
    pub fn get_row(&self, row_index: usize) -> Option<&[usize]> {
        self.rows.get(row_index).map(|v| v.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_exact_cover() {
        // Simple exact cover problem from Knuth's paper
        //   A B C D E F G
        // 1 [1 0 0 1 0 0 1]
        // 2 [1 0 0 1 0 0 0]
        // 3 [0 0 0 1 1 0 1]
        // 4 [0 0 1 0 1 1 0]
        // 5 [0 1 1 0 0 1 1]
        // 6 [0 1 0 0 0 0 1]

        let mut dlx = DancingLinks::new(7);
        dlx.add_row(vec![0, 3, 6]); // Row 0
        dlx.add_row(vec![0, 3]);    // Row 1
        dlx.add_row(vec![3, 4, 6]); // Row 2
        dlx.add_row(vec![2, 4, 5]); // Row 3
        dlx.add_row(vec![1, 2, 5, 6]); // Row 4
        dlx.add_row(vec![1, 6]);    // Row 5

        let solutions = dlx.solve_all();
        assert_eq!(solutions.len(), 1);

        // The solution should be rows 1, 3, 5
        let mut solution = solutions[0].clone();
        solution.sort();
        assert_eq!(solution, vec![1, 3, 5]);
    }

    #[test]
    fn test_no_solution() {
        // A problem with no solution
        let mut dlx = DancingLinks::new(3);
        dlx.add_row(vec![0, 1]);
        dlx.add_row(vec![1, 2]);
        // Column 0 and 2 can't both be covered without duplicating column 1

        let solutions = dlx.solve_all();
        assert_eq!(solutions.len(), 0);
    }

    #[test]
    fn test_multiple_solutions() {
        // A problem with multiple solutions
        //   A B C
        // 1 [1 0 0]
        // 2 [0 1 0]
        // 3 [0 0 1]
        // 4 [1 0 0]
        // 5 [0 1 0]
        // 6 [0 0 1]

        let mut dlx = DancingLinks::new(3);
        dlx.add_row(vec![0]); // Row 0
        dlx.add_row(vec![1]); // Row 1
        dlx.add_row(vec![2]); // Row 2
        dlx.add_row(vec![0]); // Row 3
        dlx.add_row(vec![1]); // Row 4
        dlx.add_row(vec![2]); // Row 5

        let solutions = dlx.solve_all();
        // Should have 2^3 = 8 solutions (each column can be covered by 2 different rows)
        assert_eq!(solutions.len(), 8);
    }

    #[test]
    fn test_single_solution() {
        // Test solve_one
        let mut dlx = DancingLinks::new(3);
        dlx.add_row(vec![0, 1]);
        dlx.add_row(vec![2]);

        let solution = dlx.solve_one();
        assert!(solution.is_some());

        let mut sol = solution.unwrap();
        sol.sort();
        assert_eq!(sol, vec![0, 1]);
    }

    #[test]
    fn test_pentomino_subset() {
        // A simplified pentomino-like problem
        // Try to cover a 3x2 grid (6 cells) with pieces
        //   0 1 2 3 4 5
        // A [1 1 1 0 0 0]  // Horizontal piece covering 0,1,2
        // B [0 0 0 1 1 1]  // Horizontal piece covering 3,4,5
        // C [1 1 0 0 1 1]  // L-shaped piece
        // D [1 0 1 1 0 1]  // Scattered piece

        let mut dlx = DancingLinks::new(6);
        dlx.add_row(vec![0, 1, 2]);
        dlx.add_row(vec![3, 4, 5]);
        dlx.add_row(vec![0, 1, 4, 5]);
        dlx.add_row(vec![0, 2, 3, 5]);

        let solutions = dlx.solve_all();
        assert!(solutions.len() > 0);
    }

    #[test]
    fn test_n_queens_4x4() {
        // 4-Queens problem represented as an exact cover problem
        // We need to cover:
        // - 4 rows (one queen per row)
        // - 4 columns (one queen per column)
        // - 7 diagonals (\ direction, no duplicates)
        // - 7 anti-diagonals (/ direction, no duplicates)
        // Total: 4 + 4 + 7 + 7 = 22 constraints

        // However, for simplicity in this test, we'll just verify the structure works
        // A full N-Queens implementation would be more complex

        let mut dlx = DancingLinks::new(8); // Simplified: just rows and columns
        // Row 0: queen at (0,0)
        dlx.add_row(vec![0, 4]); // row 0, col 0
        // Row 1: queen at (0,1)
        dlx.add_row(vec![0, 5]); // row 0, col 1
        // Row 2: queen at (1,2)
        dlx.add_row(vec![1, 6]); // row 1, col 2
        // Row 3: queen at (1,3)
        dlx.add_row(vec![1, 7]); // row 1, col 3
        // ... (would continue for all positions)

        // Just verify the structure is created correctly
        assert_eq!(dlx.num_columns(), 8);
        assert_eq!(dlx.num_rows(), 4);
    }

    #[test]
    fn test_empty_row() {
        // Test that empty rows are ignored (they don't get added)
        let mut dlx = DancingLinks::new(3);
        dlx.add_row(vec![]); // Should be ignored
        dlx.add_row(vec![0, 1, 2]);

        let solutions = dlx.solve_all();
        assert_eq!(solutions.len(), 1);
        // Since empty row was ignored, the non-empty row gets index 0
        assert_eq!(solutions[0], vec![0]);
    }

    #[test]
    fn test_get_row() {
        let mut dlx = DancingLinks::new(5);
        dlx.add_row(vec![0, 2, 4]);
        dlx.add_row(vec![1, 3]);

        assert_eq!(dlx.get_row(0), Some(&[0, 2, 4][..]));
        assert_eq!(dlx.get_row(1), Some(&[1, 3][..]));
        assert_eq!(dlx.get_row(2), None);
    }

    #[test]
    #[should_panic(expected = "Column index 5 out of bounds")]
    fn test_invalid_column_index() {
        let mut dlx = DancingLinks::new(5);
        dlx.add_row(vec![0, 5, 2]); // Column 5 is out of bounds
    }

    #[test]
    fn test_sudoku_subset() {
        // Test a very simplified Sudoku constraint
        // For a 2x2 Sudoku (only 4 cells), we need to satisfy:
        // - Each cell has a number (4 constraints)
        // - Each row has each number once (4 constraints)
        // - Each column has each number once (4 constraints)
        // - Each 2x2 box has each number once (4 constraints)
        // Total: 16 constraints

        // This is just a structural test to verify DLX can handle Sudoku-like problems
        let mut dlx = DancingLinks::new(16);

        // Cell (0,0) = 1: covers cell(0,0), row0-val1, col0-val1, box0-val1
        dlx.add_row(vec![0, 4, 8, 12]);

        // Cell (0,0) = 2: covers cell(0,0), row0-val2, col0-val2, box0-val2
        dlx.add_row(vec![0, 5, 9, 13]);

        // ... (would continue for all 16 possibilities)

        assert_eq!(dlx.num_columns(), 16);
    }
}
