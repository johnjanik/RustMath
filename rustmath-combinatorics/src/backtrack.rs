//! Generic backtracking framework with constraint satisfaction and pruning
//!
//! This module provides a flexible backtracking framework for solving constraint
//! satisfaction problems, combinatorial search, and other problems that can be
//! solved by exploring a search tree with backtracking.
//!
//! # Features
//!
//! - Generic over state and choice types
//! - Constraint checking to validate partial solutions
//! - Pruning callbacks to cut off unpromising branches
//! - Find one solution, all solutions, or a limited number
//! - Both trait-based and functional APIs
//!
//! # Example: N-Queens Problem
//!
//! ```
//! use rustmath_combinatorics::backtrack::{BacktrackProblem, Backtracker};
//!
//! struct NQueens {
//!     n: usize,
//! }
//!
//! impl BacktrackProblem for NQueens {
//!     type State = Vec<usize>; // positions[i] = column of queen in row i
//!     type Choice = usize;     // column to place next queen
//!
//!     fn initial_state(&self) -> Self::State {
//!         Vec::new()
//!     }
//!
//!     fn is_solution(&self, state: &Self::State) -> bool {
//!         state.len() == self.n
//!     }
//!
//!     fn is_valid(&self, state: &Self::State) -> bool {
//!         let row = state.len() - 1;
//!         for prev_row in 0..row {
//!             let col_diff = (state[row] as i32 - state[prev_row] as i32).abs() as usize;
//!             let row_diff = row - prev_row;
//!             // Check column and diagonal conflicts
//!             if state[row] == state[prev_row] || col_diff == row_diff {
//!                 return false;
//!             }
//!         }
//!         true
//!     }
//!
//!     fn can_extend(&self, state: &Self::State) -> bool {
//!         state.len() < self.n
//!     }
//!
//!     fn candidates(&self, _state: &Self::State) -> Vec<Self::Choice> {
//!         (0..self.n).collect()
//!     }
//!
//!     fn apply(&self, state: &Self::State, choice: &Self::Choice) -> Self::State {
//!         let mut new_state = state.clone();
//!         new_state.push(*choice);
//!         new_state
//!     }
//! }
//!
//! let n_queens = NQueens { n: 4 };
//! let solver = Backtracker::new(n_queens);
//! let solutions = solver.solve_all();
//! assert_eq!(solutions.len(), 2); // 4-queens has 2 solutions
//! ```

/// A trait defining a backtracking problem
///
/// Implement this trait to define your problem, then use `Backtracker` to solve it.
pub trait BacktrackProblem {
    /// The type representing a partial or complete solution state
    type State: Clone;

    /// The type representing a choice/candidate at each step
    type Choice;

    /// Generate the initial (empty) state
    fn initial_state(&self) -> Self::State;

    /// Check if the current state represents a complete solution
    fn is_solution(&self, state: &Self::State) -> bool;

    /// Check if the current state satisfies all constraints (is valid)
    ///
    /// This is called after applying each choice. Return false to reject
    /// this branch of the search tree.
    fn is_valid(&self, state: &Self::State) -> bool;

    /// Pruning check: can this state potentially lead to a solution?
    ///
    /// Return false to prune this branch early, even if it's currently valid.
    /// This is an optimization to cut off branches that cannot possibly lead
    /// to a solution.
    ///
    /// Default implementation always returns true (no pruning).
    fn can_extend(&self, _state: &Self::State) -> bool {
        true
    }

    /// Generate candidates for the next choice from the current state
    ///
    /// Each candidate will be tried in order via backtracking.
    fn candidates(&self, state: &Self::State) -> Vec<Self::Choice>;

    /// Apply a choice to the current state to get the next state
    fn apply(&self, state: &Self::State, choice: &Self::Choice) -> Self::State;
}

/// Backtracking solver for problems implementing `BacktrackProblem`
///
/// This solver uses depth-first search with backtracking to explore
/// the solution space, applying constraint checking and pruning at each step.
pub struct Backtracker<P: BacktrackProblem> {
    problem: P,
}

impl<P: BacktrackProblem> Backtracker<P> {
    /// Create a new backtracker for the given problem
    pub fn new(problem: P) -> Self {
        Self { problem }
    }

    /// Find one solution to the problem
    ///
    /// Returns `Some(solution)` if a solution exists, `None` otherwise.
    pub fn solve_one(&self) -> Option<P::State> {
        let initial = self.problem.initial_state();
        self.backtrack_one(initial)
    }

    /// Find all solutions to the problem
    ///
    /// Returns a vector of all valid solutions.
    /// Warning: This may take a long time or use excessive memory for
    /// problems with many solutions.
    pub fn solve_all(&self) -> Vec<P::State> {
        let initial = self.problem.initial_state();
        let mut solutions = Vec::new();
        self.backtrack_all(initial, &mut solutions);
        solutions
    }

    /// Find at most `limit` solutions to the problem
    ///
    /// Returns when `limit` solutions have been found or the search space
    /// is exhausted.
    pub fn solve_limit(&self, limit: usize) -> Vec<P::State> {
        let initial = self.problem.initial_state();
        let mut solutions = Vec::new();
        self.backtrack_limit(initial, &mut solutions, limit);
        solutions
    }

    /// Count the total number of solutions without storing them
    ///
    /// This is more memory-efficient than `solve_all().len()` for problems
    /// with many solutions.
    pub fn count_solutions(&self) -> usize {
        let initial = self.problem.initial_state();
        self.backtrack_count(initial)
    }

    // Internal recursive backtracking for finding one solution
    fn backtrack_one(&self, state: P::State) -> Option<P::State> {
        // Check if current state is a solution
        if self.problem.is_solution(&state) {
            return Some(state);
        }

        // Pruning check
        if !self.problem.can_extend(&state) {
            return None;
        }

        // Try each candidate
        for choice in self.problem.candidates(&state) {
            let next_state = self.problem.apply(&state, &choice);

            // Constraint checking
            if !self.problem.is_valid(&next_state) {
                continue;
            }

            // Recursively search
            if let Some(solution) = self.backtrack_one(next_state) {
                return Some(solution);
            }
        }

        None
    }

    // Internal recursive backtracking for finding all solutions
    fn backtrack_all(&self, state: P::State, solutions: &mut Vec<P::State>) {
        // Check if current state is a solution
        if self.problem.is_solution(&state) {
            solutions.push(state);
            return;
        }

        // Pruning check
        if !self.problem.can_extend(&state) {
            return;
        }

        // Try each candidate
        for choice in self.problem.candidates(&state) {
            let next_state = self.problem.apply(&state, &choice);

            // Constraint checking
            if !self.problem.is_valid(&next_state) {
                continue;
            }

            // Recursively search
            self.backtrack_all(next_state, solutions);
        }
    }

    // Internal recursive backtracking for finding limited solutions
    fn backtrack_limit(&self, state: P::State, solutions: &mut Vec<P::State>, limit: usize) {
        // Stop if we've found enough solutions
        if solutions.len() >= limit {
            return;
        }

        // Check if current state is a solution
        if self.problem.is_solution(&state) {
            solutions.push(state);
            return;
        }

        // Pruning check
        if !self.problem.can_extend(&state) {
            return;
        }

        // Try each candidate
        for choice in self.problem.candidates(&state) {
            if solutions.len() >= limit {
                break;
            }

            let next_state = self.problem.apply(&state, &choice);

            // Constraint checking
            if !self.problem.is_valid(&next_state) {
                continue;
            }

            // Recursively search
            self.backtrack_limit(next_state, solutions, limit);
        }
    }

    // Internal recursive backtracking for counting solutions
    fn backtrack_count(&self, state: P::State) -> usize {
        // Check if current state is a solution
        if self.problem.is_solution(&state) {
            return 1;
        }

        // Pruning check
        if !self.problem.can_extend(&state) {
            return 0;
        }

        let mut count = 0;

        // Try each candidate
        for choice in self.problem.candidates(&state) {
            let next_state = self.problem.apply(&state, &choice);

            // Constraint checking
            if !self.problem.is_valid(&next_state) {
                continue;
            }

            // Recursively count
            count += self.backtrack_count(next_state);
        }

        count
    }
}

/// Functional API for backtracking without implementing a trait
///
/// This provides a simpler interface for one-off backtracking problems
/// where implementing the full `BacktrackProblem` trait is overkill.
pub struct BacktrackFn<State, Choice> {
    initial: State,
    is_solution: Box<dyn Fn(&State) -> bool>,
    is_valid: Box<dyn Fn(&State) -> bool>,
    can_extend: Box<dyn Fn(&State) -> bool>,
    candidates: Box<dyn Fn(&State) -> Vec<Choice>>,
    apply: Box<dyn Fn(&State, &Choice) -> State>,
}

impl<State: Clone + 'static, Choice: 'static> BacktrackFn<State, Choice> {
    /// Create a new functional backtracking problem
    ///
    /// # Arguments
    ///
    /// * `initial` - The initial state
    /// * `is_solution` - Function to check if a state is a complete solution
    /// * `is_valid` - Function to check if a state satisfies constraints
    /// * `can_extend` - Function to check if a state can be extended (pruning)
    /// * `candidates` - Function to generate next choices
    /// * `apply` - Function to apply a choice to a state
    pub fn new<F1, F2, F3, F4, F5>(
        initial: State,
        is_solution: F1,
        is_valid: F2,
        can_extend: F3,
        candidates: F4,
        apply: F5,
    ) -> Self
    where
        F1: Fn(&State) -> bool + 'static,
        F2: Fn(&State) -> bool + 'static,
        F3: Fn(&State) -> bool + 'static,
        F4: Fn(&State) -> Vec<Choice> + 'static,
        F5: Fn(&State, &Choice) -> State + 'static,
    {
        Self {
            initial,
            is_solution: Box::new(is_solution),
            is_valid: Box::new(is_valid),
            can_extend: Box::new(can_extend),
            candidates: Box::new(candidates),
            apply: Box::new(apply),
        }
    }

    /// Find one solution
    pub fn solve_one(&self) -> Option<State> {
        self.backtrack_one(self.initial.clone())
    }

    /// Find all solutions
    pub fn solve_all(&self) -> Vec<State> {
        let mut solutions = Vec::new();
        self.backtrack_all(self.initial.clone(), &mut solutions);
        solutions
    }

    /// Find at most `limit` solutions
    pub fn solve_limit(&self, limit: usize) -> Vec<State> {
        let mut solutions = Vec::new();
        self.backtrack_limit(self.initial.clone(), &mut solutions, limit);
        solutions
    }

    /// Count all solutions
    pub fn count_solutions(&self) -> usize {
        self.backtrack_count(self.initial.clone())
    }

    fn backtrack_one(&self, state: State) -> Option<State> {
        if (self.is_solution)(&state) {
            return Some(state);
        }

        if !(self.can_extend)(&state) {
            return None;
        }

        for choice in (self.candidates)(&state) {
            let next_state = (self.apply)(&state, &choice);

            if !(self.is_valid)(&next_state) {
                continue;
            }

            if let Some(solution) = self.backtrack_one(next_state) {
                return Some(solution);
            }
        }

        None
    }

    fn backtrack_all(&self, state: State, solutions: &mut Vec<State>) {
        if (self.is_solution)(&state) {
            solutions.push(state);
            return;
        }

        if !(self.can_extend)(&state) {
            return;
        }

        for choice in (self.candidates)(&state) {
            let next_state = (self.apply)(&state, &choice);

            if !(self.is_valid)(&next_state) {
                continue;
            }

            self.backtrack_all(next_state, solutions);
        }
    }

    fn backtrack_limit(&self, state: State, solutions: &mut Vec<State>, limit: usize) {
        if solutions.len() >= limit {
            return;
        }

        if (self.is_solution)(&state) {
            solutions.push(state);
            return;
        }

        if !(self.can_extend)(&state) {
            return;
        }

        for choice in (self.candidates)(&state) {
            if solutions.len() >= limit {
                break;
            }

            let next_state = (self.apply)(&state, &choice);

            if !(self.is_valid)(&next_state) {
                continue;
            }

            self.backtrack_limit(next_state, solutions, limit);
        }
    }

    fn backtrack_count(&self, state: State) -> usize {
        if (self.is_solution)(&state) {
            return 1;
        }

        if !(self.can_extend)(&state) {
            return 0;
        }

        let mut count = 0;

        for choice in (self.candidates)(&state) {
            let next_state = (self.apply)(&state, &choice);

            if !(self.is_valid)(&next_state) {
                continue;
            }

            count += self.backtrack_count(next_state);
        }

        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test problem: N-Queens
    struct NQueens {
        n: usize,
    }

    impl BacktrackProblem for NQueens {
        type State = Vec<usize>; // positions[i] = column of queen in row i
        type Choice = usize; // column to place next queen

        fn initial_state(&self) -> Self::State {
            Vec::new()
        }

        fn is_solution(&self, state: &Self::State) -> bool {
            state.len() == self.n
        }

        fn is_valid(&self, state: &Self::State) -> bool {
            if state.is_empty() {
                return true;
            }

            let row = state.len() - 1;
            for prev_row in 0..row {
                let col_diff = (state[row] as i32 - state[prev_row] as i32).abs() as usize;
                let row_diff = row - prev_row;
                // Check column and diagonal conflicts
                if state[row] == state[prev_row] || col_diff == row_diff {
                    return false;
                }
            }
            true
        }

        fn can_extend(&self, state: &Self::State) -> bool {
            state.len() < self.n
        }

        fn candidates(&self, _state: &Self::State) -> Vec<Self::Choice> {
            (0..self.n).collect()
        }

        fn apply(&self, state: &Self::State, choice: &Self::Choice) -> Self::State {
            let mut new_state = state.clone();
            new_state.push(*choice);
            new_state
        }
    }

    #[test]
    fn test_n_queens_4() {
        let n_queens = NQueens { n: 4 };
        let solver = Backtracker::new(n_queens);
        let solutions = solver.solve_all();

        assert_eq!(solutions.len(), 2);

        // Verify each solution
        for solution in &solutions {
            assert_eq!(solution.len(), 4);
            // Check no conflicts
            for i in 0..4 {
                for j in (i + 1)..4 {
                    let col_diff = (solution[i] as i32 - solution[j] as i32).abs() as usize;
                    let row_diff = j - i;
                    assert_ne!(solution[i], solution[j]);
                    assert_ne!(col_diff, row_diff);
                }
            }
        }
    }

    #[test]
    fn test_n_queens_8() {
        let n_queens = NQueens { n: 8 };
        let solver = Backtracker::new(n_queens);
        let count = solver.count_solutions();

        // 8-queens has 92 solutions
        assert_eq!(count, 92);
    }

    #[test]
    fn test_n_queens_solve_one() {
        let n_queens = NQueens { n: 8 };
        let solver = Backtracker::new(n_queens);
        let solution = solver.solve_one();

        assert!(solution.is_some());
        let sol = solution.unwrap();
        assert_eq!(sol.len(), 8);
    }

    #[test]
    fn test_n_queens_solve_limit() {
        let n_queens = NQueens { n: 8 };
        let solver = Backtracker::new(n_queens);
        let solutions = solver.solve_limit(10);

        assert_eq!(solutions.len(), 10);
    }

    // Test problem: Subset Sum
    // Find subsets of numbers that sum to a target
    struct SubsetSum {
        numbers: Vec<i32>,
        target: i32,
    }

    impl BacktrackProblem for SubsetSum {
        type State = (Vec<bool>, usize); // (included[], next_index)
        type Choice = bool; // include next number or not

        fn initial_state(&self) -> Self::State {
            (Vec::new(), 0)
        }

        fn is_solution(&self, state: &Self::State) -> bool {
            state.1 == self.numbers.len() && self.current_sum(state) == self.target
        }

        fn is_valid(&self, state: &Self::State) -> bool {
            let sum = self.current_sum(state);
            // Constraint: sum should not exceed target
            sum <= self.target
        }

        fn can_extend(&self, state: &Self::State) -> bool {
            // Pruning: if current sum already exceeds target, prune
            let sum = self.current_sum(state);
            if sum > self.target {
                return false;
            }

            // Can extend if we haven't considered all numbers
            state.1 < self.numbers.len()
        }

        fn candidates(&self, _state: &Self::State) -> Vec<Self::Choice> {
            vec![false, true] // try excluding, then including
        }

        fn apply(&self, state: &Self::State, choice: &Self::Choice) -> Self::State {
            let mut new_included = state.0.clone();
            new_included.push(*choice);
            (new_included, state.1 + 1)
        }
    }

    impl SubsetSum {
        fn current_sum(&self, state: &(Vec<bool>, usize)) -> i32 {
            state
                .0
                .iter()
                .enumerate()
                .filter(|(_, &included)| included)
                .map(|(i, _)| self.numbers[i])
                .sum()
        }
    }

    #[test]
    fn test_subset_sum() {
        let problem = SubsetSum {
            numbers: vec![1, 2, 3, 4, 5],
            target: 7,
        };

        let solver = Backtracker::new(problem);
        let solutions = solver.solve_all();

        // Solutions: {3,4}, {2,5}, {1,2,4}
        assert_eq!(solutions.len(), 3);

        // Verify each solution
        for (included, _) in &solutions {
            let sum: i32 = included
                .iter()
                .enumerate()
                .filter(|(_, &inc)| inc)
                .map(|(i, _)| [1, 2, 3, 4, 5][i])
                .sum();
            assert_eq!(sum, 7);
        }
    }

    #[test]
    fn test_subset_sum_no_solution() {
        let problem = SubsetSum {
            numbers: vec![2, 4, 6],
            target: 5,
        };

        let solver = Backtracker::new(problem);
        let solution = solver.solve_one();

        assert!(solution.is_none());
    }

    // Test functional API
    #[test]
    fn test_functional_api_permutations() {
        // Generate all permutations of [1, 2, 3] using functional API
        let n = 3;

        let problem = BacktrackFn::new(
            Vec::new(),
            move |state: &Vec<usize>| state.len() == n,
            |_state: &Vec<usize>| true, // all states valid
            move |state: &Vec<usize>| state.len() < n,
            move |state: &Vec<usize>| {
                (1..=n).filter(|x| !state.contains(x)).collect()
            },
            |state: &Vec<usize>, choice: &usize| {
                let mut new_state = state.clone();
                new_state.push(*choice);
                new_state
            },
        );

        let solutions = problem.solve_all();

        // 3! = 6 permutations
        assert_eq!(solutions.len(), 6);

        // Verify each is a valid permutation
        for perm in &solutions {
            assert_eq!(perm.len(), 3);
            assert!(perm.contains(&1));
            assert!(perm.contains(&2));
            assert!(perm.contains(&3));
        }
    }

    // Test graph coloring problem
    struct GraphColoring {
        edges: Vec<(usize, usize)>,
        num_vertices: usize,
        num_colors: usize,
    }

    impl BacktrackProblem for GraphColoring {
        type State = Vec<usize>; // color[i] = color of vertex i
        type Choice = usize; // color to assign

        fn initial_state(&self) -> Self::State {
            Vec::new()
        }

        fn is_solution(&self, state: &Self::State) -> bool {
            state.len() == self.num_vertices
        }

        fn is_valid(&self, state: &Self::State) -> bool {
            if state.is_empty() {
                return true;
            }

            let vertex = state.len() - 1;
            let color = state[vertex];

            // Check if this coloring conflicts with any adjacent vertex
            for &(u, v) in &self.edges {
                if u == vertex && v < state.len() {
                    if state[v] == color {
                        return false;
                    }
                }
                if v == vertex && u < state.len() {
                    if state[u] == color {
                        return false;
                    }
                }
            }

            true
        }

        fn can_extend(&self, state: &Self::State) -> bool {
            state.len() < self.num_vertices
        }

        fn candidates(&self, _state: &Self::State) -> Vec<Self::Choice> {
            (0..self.num_colors).collect()
        }

        fn apply(&self, state: &Self::State, choice: &Self::Choice) -> Self::State {
            let mut new_state = state.clone();
            new_state.push(*choice);
            new_state
        }
    }

    #[test]
    fn test_graph_coloring_triangle() {
        // Triangle graph requires 3 colors
        let problem = GraphColoring {
            edges: vec![(0, 1), (1, 2), (2, 0)],
            num_vertices: 3,
            num_colors: 3,
        };

        let solver = Backtracker::new(problem);
        let count = solver.count_solutions();

        // Each vertex can have any color different from its neighbors
        // 3 * 2 * 1 = 6 colorings for a triangle with 3 colors
        assert_eq!(count, 6);
    }

    #[test]
    fn test_graph_coloring_impossible() {
        // Triangle with only 2 colors - impossible
        let problem = GraphColoring {
            edges: vec![(0, 1), (1, 2), (2, 0)],
            num_vertices: 3,
            num_colors: 2,
        };

        let solver = Backtracker::new(problem);
        let solution = solver.solve_one();

        assert!(solution.is_none());
    }

    // Test Sudoku solver (simplified 4x4)
    struct Sudoku4x4 {
        initial: Vec<Vec<Option<u8>>>, // 4x4 grid, None = empty
    }

    impl BacktrackProblem for Sudoku4x4 {
        type State = Vec<Vec<u8>>; // partial solution
        type Choice = (usize, usize, u8); // (row, col, value)

        fn initial_state(&self) -> Self::State {
            let mut grid = vec![vec![0; 4]; 4];
            for i in 0..4 {
                for j in 0..4 {
                    if let Some(val) = self.initial[i][j] {
                        grid[i][j] = val;
                    }
                }
            }
            grid
        }

        fn is_solution(&self, state: &Self::State) -> bool {
            // Check if all cells are filled (no zeros)
            state.iter().all(|row| row.iter().all(|&cell| cell != 0))
        }

        fn is_valid(&self, state: &Self::State) -> bool {
            // Check rows
            for row in state {
                let non_zero: Vec<_> = row.iter().filter(|&&x| x != 0).copied().collect();
                let mut sorted = non_zero.clone();
                sorted.sort_unstable();
                sorted.dedup();
                if sorted.len() != non_zero.len() {
                    return false;
                }
            }

            // Check columns
            for col in 0..4 {
                let non_zero: Vec<_> = (0..4)
                    .map(|row| state[row][col])
                    .filter(|&x| x != 0)
                    .collect();
                let mut sorted = non_zero.clone();
                sorted.sort_unstable();
                sorted.dedup();
                if sorted.len() != non_zero.len() {
                    return false;
                }
            }

            // Check 2x2 boxes
            for box_row in 0..2 {
                for box_col in 0..2 {
                    let mut non_zero = Vec::new();
                    for i in 0..2 {
                        for j in 0..2 {
                            let val = state[box_row * 2 + i][box_col * 2 + j];
                            if val != 0 {
                                non_zero.push(val);
                            }
                        }
                    }
                    let mut sorted = non_zero.clone();
                    sorted.sort_unstable();
                    sorted.dedup();
                    if sorted.len() != non_zero.len() {
                        return false;
                    }
                }
            }

            true
        }

        fn can_extend(&self, _state: &Self::State) -> bool {
            true
        }

        fn candidates(&self, state: &Self::State) -> Vec<Self::Choice> {
            // Find first empty cell
            for i in 0..4 {
                for j in 0..4 {
                    if state[i][j] == 0 {
                        // This cell was not in initial configuration
                        if self.initial[i][j].is_none() {
                            return (1..=4).map(|val| (i, j, val)).collect();
                        }
                    }
                }
            }
            Vec::new()
        }

        fn apply(&self, state: &Self::State, choice: &Self::Choice) -> Self::State {
            let mut new_state = state.clone();
            new_state[choice.0][choice.1] = choice.2;
            new_state
        }
    }

    #[test]
    fn test_sudoku_4x4() {
        let problem = Sudoku4x4 {
            initial: vec![
                vec![Some(1), None, None, Some(4)],
                vec![None, None, None, None],
                vec![None, None, None, None],
                vec![Some(4), None, None, Some(1)],
            ],
        };

        let solver = Backtracker::new(problem);
        let solution = solver.solve_one();

        assert!(solution.is_some());
        let sol = solution.unwrap();

        // Verify solution
        assert_eq!(sol[0][0], 1);
        assert_eq!(sol[0][3], 4);
        assert_eq!(sol[3][0], 4);
        assert_eq!(sol[3][3], 1);

        // All rows should have 1,2,3,4
        for row in &sol {
            let mut sorted = row.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, vec![1, 2, 3, 4]);
        }
    }
}
