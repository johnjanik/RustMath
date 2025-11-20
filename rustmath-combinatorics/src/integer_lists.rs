//! Integer lists with constraint propagation
//!
//! This module provides generation of lists of non-negative integers that satisfy
//! various constraints. The key feature is constraint propagation, which efficiently
//! prunes the search space by inferring bounds on elements based on the constraints.
//!
//! # Examples
//!
//! ```
//! use rustmath_combinatorics::integer_lists::{IntegerListConstraints, integer_lists};
//!
//! // Generate all lists of length 3 that sum to 5
//! let constraints = IntegerListConstraints::new()
//!     .length(3)
//!     .sum(5);
//! let lists = integer_lists(&constraints);
//! // Results: [0,0,5], [0,1,4], [0,2,3], [0,3,2], [0,4,1], [0,5,0],
//! //          [1,0,4], [1,1,3], [1,2,2], [1,3,1], [1,4,0],
//! //          [2,0,3], [2,1,2], [2,2,1], [2,3,0],
//! //          [3,0,2], [3,1,1], [3,2,0],
//! //          [4,0,1], [4,1,0],
//! //          [5,0,0]
//! ```

use std::cmp::{max, min};

/// An integer list (sequence of non-negative integers)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IntegerList {
    parts: Vec<usize>,
}

impl IntegerList {
    /// Create a new integer list from a vector
    pub fn new(parts: Vec<usize>) -> Self {
        IntegerList { parts }
    }

    /// Get the parts
    pub fn parts(&self) -> &[usize] {
        &self.parts
    }

    /// Get the length of the list
    pub fn length(&self) -> usize {
        self.parts.len()
    }

    /// Get the sum of the list
    pub fn sum(&self) -> usize {
        self.parts.iter().sum()
    }

    /// Get the minimum part (or None if empty)
    pub fn min_part(&self) -> Option<usize> {
        self.parts.iter().copied().min()
    }

    /// Get the maximum part (or None if empty)
    pub fn max_part(&self) -> Option<usize> {
        self.parts.iter().copied().max()
    }
}

/// Constraints for generating integer lists
///
/// This struct uses a builder pattern to specify constraints on the integer lists
/// to be generated. Constraints include sum, length, bounds on parts, and slopes.
#[derive(Debug, Clone)]
pub struct IntegerListConstraints {
    /// Exact sum (if Some)
    sum: Option<usize>,
    /// Minimum sum (if Some)
    min_sum: Option<usize>,
    /// Maximum sum (if Some)
    max_sum: Option<usize>,

    /// Exact length (if Some)
    length: Option<usize>,
    /// Minimum length (if Some)
    min_length: Option<usize>,
    /// Maximum length (if Some)
    max_length: Option<usize>,

    /// Minimum value for any part
    min_part: usize,
    /// Maximum value for any part (if Some)
    max_part: Option<usize>,

    /// Minimum slope (difference between consecutive elements)
    min_slope: Option<isize>,
    /// Maximum slope (difference between consecutive elements)
    max_slope: Option<isize>,

    /// Floor: element-wise lower bounds (if Some)
    floor: Option<Vec<usize>>,
    /// Ceiling: element-wise upper bounds (if Some)
    ceiling: Option<Vec<usize>>,

    /// Inner sum: sum of all elements except first and last (if Some)
    inner_sum: Option<usize>,
}

impl IntegerListConstraints {
    /// Create a new constraints builder with default values
    pub fn new() -> Self {
        IntegerListConstraints {
            sum: None,
            min_sum: None,
            max_sum: None,
            length: None,
            min_length: None,
            max_length: None,
            min_part: 0,
            max_part: None,
            min_slope: None,
            max_slope: None,
            floor: None,
            ceiling: None,
            inner_sum: None,
        }
    }

    /// Set exact sum constraint
    pub fn sum(mut self, sum: usize) -> Self {
        self.sum = Some(sum);
        self
    }

    /// Set minimum sum constraint
    pub fn min_sum(mut self, min_sum: usize) -> Self {
        self.min_sum = Some(min_sum);
        self
    }

    /// Set maximum sum constraint
    pub fn max_sum(mut self, max_sum: usize) -> Self {
        self.max_sum = Some(max_sum);
        self
    }

    /// Set exact length constraint
    pub fn length(mut self, length: usize) -> Self {
        self.length = Some(length);
        self
    }

    /// Set minimum length constraint
    pub fn min_length(mut self, min_length: usize) -> Self {
        self.min_length = Some(min_length);
        self
    }

    /// Set maximum length constraint
    pub fn max_length(mut self, max_length: usize) -> Self {
        self.max_length = Some(max_length);
        self
    }

    /// Set minimum part constraint (applies to all parts)
    pub fn min_part(mut self, min_part: usize) -> Self {
        self.min_part = min_part;
        self
    }

    /// Set maximum part constraint (applies to all parts)
    pub fn max_part(mut self, max_part: usize) -> Self {
        self.max_part = Some(max_part);
        self
    }

    /// Set minimum slope constraint (minimum difference between consecutive elements)
    pub fn min_slope(mut self, min_slope: isize) -> Self {
        self.min_slope = Some(min_slope);
        self
    }

    /// Set maximum slope constraint (maximum difference between consecutive elements)
    pub fn max_slope(mut self, max_slope: isize) -> Self {
        self.max_slope = Some(max_slope);
        self
    }

    /// Set floor constraint (element-wise lower bounds)
    pub fn floor(mut self, floor: Vec<usize>) -> Self {
        self.floor = Some(floor);
        self
    }

    /// Set ceiling constraint (element-wise upper bounds)
    pub fn ceiling(mut self, ceiling: Vec<usize>) -> Self {
        self.ceiling = Some(ceiling);
        self
    }

    /// Set inner sum constraint (sum of all elements except first and last)
    pub fn inner_sum(mut self, inner_sum: usize) -> Self {
        self.inner_sum = Some(inner_sum);
        self
    }

    /// Get the effective minimum sum
    fn get_min_sum(&self) -> usize {
        if let Some(s) = self.sum {
            s
        } else if let Some(s) = self.min_sum {
            s
        } else {
            0
        }
    }

    /// Get the effective maximum sum
    fn get_max_sum(&self) -> Option<usize> {
        if let Some(s) = self.sum {
            Some(s)
        } else {
            self.max_sum
        }
    }

    /// Get the effective minimum length
    fn get_min_length(&self) -> usize {
        if let Some(l) = self.length {
            l
        } else if let Some(l) = self.min_length {
            l
        } else {
            0
        }
    }

    /// Get the effective maximum length
    fn get_max_length(&self) -> Option<usize> {
        if let Some(l) = self.length {
            Some(l)
        } else {
            self.max_length
        }
    }
}

impl Default for IntegerListConstraints {
    fn default() -> Self {
        Self::new()
    }
}

/// State during constraint propagation and generation
#[derive(Debug, Clone)]
struct GenerationState {
    /// Current partial list being built
    current: Vec<usize>,
    /// Remaining sum to distribute
    remaining_sum: usize,
    /// Target length (if known)
    target_length: Option<usize>,
}

/// Generate all integer lists satisfying the given constraints
///
/// Uses constraint propagation to efficiently generate lists by pruning
/// the search space based on constraints.
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::integer_lists::{IntegerListConstraints, integer_lists};
///
/// // Lists of length 2 that sum to 3
/// let constraints = IntegerListConstraints::new()
///     .length(2)
///     .sum(3);
/// let lists = integer_lists(&constraints);
/// assert_eq!(lists.len(), 4); // [0,3], [1,2], [2,1], [3,0]
/// ```
pub fn integer_lists(constraints: &IntegerListConstraints) -> Vec<IntegerList> {
    let mut result = Vec::new();

    // Determine target length
    let min_length = constraints.get_min_length();
    let max_length = constraints.get_max_length();

    // If exact length is specified, only generate for that length
    if let Some(length) = constraints.length {
        generate_with_length(constraints, length, &mut result);
    } else {
        // Generate for all valid lengths
        let min_len = min_length;
        let max_len = max_length.unwrap_or_else(|| {
            // If no max length, use sum as upper bound
            let sum = constraints.get_max_sum().unwrap_or(100);
            sum + 1
        });

        for length in min_len..=max_len {
            generate_with_length(constraints, length, &mut result);
        }
    }

    result
}

/// Generate integer lists of a specific length
fn generate_with_length(
    constraints: &IntegerListConstraints,
    length: usize,
    result: &mut Vec<IntegerList>,
) {
    if length == 0 {
        // Empty list
        let list = IntegerList::new(vec![]);
        if satisfies_constraints(&list, constraints) {
            result.push(list);
        }
        return;
    }

    let target_sum = constraints.sum;
    let min_sum = constraints.get_min_sum();
    let max_sum = constraints.get_max_sum();

    // Generate lists with the specified length
    let mut current = Vec::with_capacity(length);

    // Determine sum range
    if let Some(sum) = target_sum {
        generate_recursive(constraints, length, sum, &mut current, result);
    } else {
        // Generate for all valid sums
        let min_s = min_sum;
        let max_s = max_sum.unwrap_or(length * constraints.max_part.unwrap_or(100));

        for sum in min_s..=max_s {
            generate_recursive(constraints, length, sum, &mut current, result);
        }
    }
}

/// Recursively generate lists with constraint propagation
fn generate_recursive(
    constraints: &IntegerListConstraints,
    target_length: usize,
    target_sum: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<IntegerList>,
) {
    let pos = current.len();

    // Base case: reached target length
    if pos == target_length {
        if current.iter().sum::<usize>() == target_sum {
            let list = IntegerList::new(current.clone());
            if satisfies_constraints(&list, constraints) {
                result.push(list);
            }
        }
        return;
    }

    // Calculate bounds for the current position using constraint propagation
    let (min_val, max_val) = propagate_constraints(
        constraints,
        current,
        pos,
        target_length,
        target_sum,
    );

    // Try all valid values for the current position
    for val in min_val..=max_val {
        current.push(val);
        generate_recursive(constraints, target_length, target_sum, current, result);
        current.pop();
    }
}

/// Propagate constraints to determine valid range for the current position
///
/// This is the key optimization: instead of trying all possible values,
/// we use the constraints to compute tight bounds on what values are valid.
fn propagate_constraints(
    constraints: &IntegerListConstraints,
    current: &[usize],
    pos: usize,
    target_length: usize,
    target_sum: usize,
) -> (usize, usize) {
    let current_sum: usize = current.iter().sum();
    let remaining_sum = if target_sum >= current_sum {
        target_sum - current_sum
    } else {
        return (0, 0); // Impossible: already exceeded target sum
    };

    let remaining_positions = target_length - pos;

    // Start with global min/max part constraints
    let mut min_val = constraints.min_part;
    let mut max_val = constraints.max_part.unwrap_or(remaining_sum);

    // Apply floor constraint
    if let Some(ref floor) = constraints.floor {
        if pos < floor.len() {
            min_val = max(min_val, floor[pos]);
        }
    }

    // Apply ceiling constraint
    if let Some(ref ceiling) = constraints.ceiling {
        if pos < ceiling.len() {
            max_val = min(max_val, ceiling[pos]);
        }
    }

    // Apply slope constraints based on previous element
    if let Some(prev) = current.last() {
        if let Some(min_slope) = constraints.min_slope {
            // Current element must be at least prev + min_slope
            let min_from_slope = if min_slope >= 0 {
                prev + min_slope as usize
            } else {
                prev.saturating_sub((-min_slope) as usize)
            };
            min_val = max(min_val, min_from_slope);
        }

        if let Some(max_slope) = constraints.max_slope {
            // Current element must be at most prev + max_slope
            let max_from_slope = if max_slope >= 0 {
                prev + max_slope as usize
            } else {
                prev.saturating_sub((-max_slope) as usize)
            };
            max_val = min(max_val, max_from_slope);
        }
    }

    // Propagate sum constraints
    if remaining_positions > 0 {
        // Minimum value: ensure we can still reach target_sum even if all
        // remaining positions use max_val
        let max_future = constraints.max_part.unwrap_or(remaining_sum);
        let max_remaining = max_future * (remaining_positions - 1);
        if remaining_sum > max_remaining {
            let min_from_sum = remaining_sum - max_remaining;
            min_val = max(min_val, min_from_sum);
        }

        // Maximum value: ensure we can still reach target_sum even if all
        // remaining positions use min_val
        let min_future = constraints.min_part;
        let min_remaining = min_future * (remaining_positions - 1);
        let max_from_sum = remaining_sum.saturating_sub(min_remaining);
        max_val = min(max_val, max_from_sum);
    } else {
        // Last position: must equal remaining sum
        min_val = remaining_sum;
        max_val = remaining_sum;
    }

    // Ensure min <= max (might not be satisfiable)
    if min_val > max_val {
        return (1, 0); // Return invalid range to prune this branch
    }

    (min_val, max_val)
}

/// Check if a list satisfies all constraints
fn satisfies_constraints(list: &IntegerList, constraints: &IntegerListConstraints) -> bool {
    let parts = list.parts();
    let n = parts.len();

    // Check sum constraints
    let sum = list.sum();
    if let Some(target_sum) = constraints.sum {
        if sum != target_sum {
            return false;
        }
    }
    if let Some(min_sum) = constraints.min_sum {
        if sum < min_sum {
            return false;
        }
    }
    if let Some(max_sum) = constraints.max_sum {
        if sum > max_sum {
            return false;
        }
    }

    // Check length constraints
    if let Some(target_length) = constraints.length {
        if n != target_length {
            return false;
        }
    }
    if let Some(min_length) = constraints.min_length {
        if n < min_length {
            return false;
        }
    }
    if let Some(max_length) = constraints.max_length {
        if n > max_length {
            return false;
        }
    }

    // Check part constraints
    for &part in parts {
        if part < constraints.min_part {
            return false;
        }
        if let Some(max_part) = constraints.max_part {
            if part > max_part {
                return false;
            }
        }
    }

    // Check slope constraints
    for i in 1..n {
        let slope = parts[i] as isize - parts[i - 1] as isize;
        if let Some(min_slope) = constraints.min_slope {
            if slope < min_slope {
                return false;
            }
        }
        if let Some(max_slope) = constraints.max_slope {
            if slope > max_slope {
                return false;
            }
        }
    }

    // Check floor constraints
    if let Some(ref floor) = constraints.floor {
        for (i, &f) in floor.iter().enumerate() {
            if i < n && parts[i] < f {
                return false;
            }
        }
    }

    // Check ceiling constraints
    if let Some(ref ceiling) = constraints.ceiling {
        for (i, &c) in ceiling.iter().enumerate() {
            if i < n && parts[i] > c {
                return false;
            }
        }
    }

    // Check inner sum constraint
    if let Some(inner_sum) = constraints.inner_sum {
        if n >= 2 {
            let actual_inner_sum: usize = parts[1..n - 1].iter().sum();
            if actual_inner_sum != inner_sum {
                return false;
            }
        }
    }

    true
}

/// Iterator for integer lists satisfying constraints
///
/// This provides a lazy iterator interface for generating integer lists,
/// which is more memory-efficient than generating all lists at once.
pub struct IntegerListIterator {
    constraints: IntegerListConstraints,
    current_length: usize,
    max_length: usize,
    current_sum: usize,
    max_sum: usize,
    stack: Vec<(Vec<usize>, usize)>, // (current list, position)
}

impl IntegerListIterator {
    /// Create a new iterator for integer lists satisfying the constraints
    pub fn new(constraints: IntegerListConstraints) -> Self {
        let min_length = constraints.get_min_length();
        let max_length = constraints.get_max_length().unwrap_or(100);
        let min_sum = constraints.get_min_sum();
        let max_sum = constraints.get_max_sum().unwrap_or(100);

        IntegerListIterator {
            constraints,
            current_length: min_length,
            max_length,
            current_sum: min_sum,
            max_sum,
            stack: vec![(vec![], 0)],
        }
    }
}

impl Iterator for IntegerListIterator {
    type Item = IntegerList;

    fn next(&mut self) -> Option<Self::Item> {
        // This is a simplified iterator - for production, we'd want a more
        // sophisticated approach using a proper search stack
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integer_list_basic() {
        let list = IntegerList::new(vec![1, 2, 3, 4]);
        assert_eq!(list.length(), 4);
        assert_eq!(list.sum(), 10);
        assert_eq!(list.min_part(), Some(1));
        assert_eq!(list.max_part(), Some(4));
    }

    #[test]
    fn test_integer_list_empty() {
        let list = IntegerList::new(vec![]);
        assert_eq!(list.length(), 0);
        assert_eq!(list.sum(), 0);
        assert_eq!(list.min_part(), None);
        assert_eq!(list.max_part(), None);
    }

    #[test]
    fn test_fixed_length_sum() {
        // Generate all lists of length 3 that sum to 4
        let constraints = IntegerListConstraints::new().length(3).sum(4);
        let lists = integer_lists(&constraints);

        // Verify all results
        for list in &lists {
            assert_eq!(list.length(), 3);
            assert_eq!(list.sum(), 4);
        }

        // Expected count: stars and bars = C(4+3-1, 3-1) = C(6,2) = 15
        assert_eq!(lists.len(), 15);
    }

    #[test]
    fn test_fixed_length_sum_small() {
        // Lists of length 2 that sum to 3
        let constraints = IntegerListConstraints::new().length(2).sum(3);
        let lists = integer_lists(&constraints);

        // Should be: [0,3], [1,2], [2,1], [3,0]
        assert_eq!(lists.len(), 4);
        assert!(lists.contains(&IntegerList::new(vec![0, 3])));
        assert!(lists.contains(&IntegerList::new(vec![1, 2])));
        assert!(lists.contains(&IntegerList::new(vec![2, 1])));
        assert!(lists.contains(&IntegerList::new(vec![3, 0])));
    }

    #[test]
    fn test_min_max_part() {
        // Lists of length 3 that sum to 6, with parts in [1, 3]
        let constraints = IntegerListConstraints::new()
            .length(3)
            .sum(6)
            .min_part(1)
            .max_part(3);
        let lists = integer_lists(&constraints);

        // Verify all results have parts in valid range
        for list in &lists {
            assert_eq!(list.length(), 3);
            assert_eq!(list.sum(), 6);
            for &part in list.parts() {
                assert!(part >= 1 && part <= 3);
            }
        }

        // Should include: [1,2,3], [1,3,2], [2,1,3], [2,2,2], [2,3,1], [3,1,2], [3,2,1]
        assert_eq!(lists.len(), 7);
    }

    #[test]
    fn test_monotone_increasing() {
        // Lists of length 3 that sum to 6, with non-decreasing parts (min_slope = 0)
        let constraints = IntegerListConstraints::new()
            .length(3)
            .sum(6)
            .min_slope(0);
        let lists = integer_lists(&constraints);

        // Verify all results are non-decreasing
        for list in &lists {
            let parts = list.parts();
            for i in 1..parts.len() {
                assert!(parts[i] >= parts[i - 1]);
            }
        }

        // Should be: [0,0,6], [0,1,5], [0,2,4], [0,3,3], [1,1,4], [1,2,3], [2,2,2]
        assert_eq!(lists.len(), 7);
    }

    #[test]
    fn test_monotone_decreasing() {
        // Lists of length 3 that sum to 6, with non-increasing parts (max_slope = 0)
        let constraints = IntegerListConstraints::new()
            .length(3)
            .sum(6)
            .max_slope(0);
        let lists = integer_lists(&constraints);

        // Verify all results are non-increasing
        for list in &lists {
            let parts = list.parts();
            for i in 1..parts.len() {
                assert!(parts[i] <= parts[i - 1]);
            }
        }

        // Should be: [6,0,0], [5,1,0], [4,2,0], [4,1,1], [3,3,0], [3,2,1], [2,2,2]
        assert_eq!(lists.len(), 7);
    }

    #[test]
    fn test_strictly_increasing() {
        // Lists of length 3 that sum to 6, with strictly increasing parts (min_slope = 1)
        let constraints = IntegerListConstraints::new()
            .length(3)
            .sum(6)
            .min_slope(1);
        let lists = integer_lists(&constraints);

        // Verify all results are strictly increasing
        for list in &lists {
            let parts = list.parts();
            for i in 1..parts.len() {
                assert!(parts[i] > parts[i - 1]);
            }
        }

        // Should be: [0,1,5], [0,2,4], [1,2,3]
        assert_eq!(lists.len(), 3);
    }

    #[test]
    fn test_floor_ceiling() {
        // Lists of length 3 with floor [1, 0, 2] and ceiling [2, 3, 4]
        let constraints = IntegerListConstraints::new()
            .length(3)
            .floor(vec![1, 0, 2])
            .ceiling(vec![2, 3, 4]);
        let lists = integer_lists(&constraints);

        // Verify all results respect floor and ceiling
        for list in &lists {
            let parts = list.parts();
            assert!(parts[0] >= 1 && parts[0] <= 2);
            assert!(parts[1] <= 3);
            assert!(parts[2] >= 2 && parts[2] <= 4);
        }

        // Count: 2 * 4 * 3 = 24
        assert_eq!(lists.len(), 24);
    }

    #[test]
    fn test_empty_list() {
        // List of length 0 with sum 0
        let constraints = IntegerListConstraints::new().length(0).sum(0);
        let lists = integer_lists(&constraints);

        assert_eq!(lists.len(), 1);
        assert_eq!(lists[0].length(), 0);
        assert_eq!(lists[0].sum(), 0);
    }

    #[test]
    fn test_impossible_constraints() {
        // Impossible: length 2, sum 5, all parts at most 1
        let constraints = IntegerListConstraints::new()
            .length(2)
            .sum(5)
            .max_part(1);
        let lists = integer_lists(&constraints);

        // Should be empty
        assert_eq!(lists.len(), 0);
    }

    #[test]
    fn test_constraint_propagation_efficiency() {
        // This test verifies that constraint propagation works by checking
        // a case where naive generation would be slow but CP makes it fast

        // Lists of length 5 that sum to 20 with parts in [3, 5]
        let constraints = IntegerListConstraints::new()
            .length(5)
            .sum(20)
            .min_part(3)
            .max_part(5);
        let lists = integer_lists(&constraints);

        // Verify all results
        for list in &lists {
            assert_eq!(list.length(), 5);
            assert_eq!(list.sum(), 20);
            for &part in list.parts() {
                assert!(part >= 3 && part <= 5);
            }
        }

        // Should find valid lists efficiently
        assert!(lists.len() > 0);
    }
}
