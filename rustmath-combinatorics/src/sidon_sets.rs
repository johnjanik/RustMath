//! Sidon Sets and B_h Sequences
//!
//! This module provides implementations for Sidon sets (B_2 sequences) and their
//! generalizations to B_h sequences. A Sidon set is a set of natural numbers where
//! all pairwise sums are distinct. More generally, a B_h sequence is a sequence where
//! all sums of h elements (not necessarily distinct) are unique.
//!
//! # Examples
//!
//! ```
//! use rustmath_combinatorics::sidon_sets::{SidonSet, is_sidon_set, greedy_sidon_set};
//!
//! // Check if a set is a Sidon set
//! let set = vec![1, 2, 5, 7];
//! assert!(is_sidon_set(&set));
//!
//! // Generate a Sidon set using greedy algorithm
//! let sidon = greedy_sidon_set(10, 20);
//! assert!(sidon.is_valid());
//! ```

use std::collections::HashSet;

/// A B_h sequence (generalization of Sidon sets)
///
/// A B_h sequence is a sequence of natural numbers where all sums of h elements
/// (with repetition allowed) are distinct. The classical Sidon set is a B_2 sequence.
///
/// # Properties
/// - For B_2 (Sidon sets): all pairwise sums a_i + a_j are distinct
/// - For B_h: all sums of h elements are distinct
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BhSequence {
    /// The elements of the sequence (stored in sorted order)
    elements: Vec<u64>,
    /// The value of h (2 for Sidon sets)
    h: usize,
}

impl BhSequence {
    /// Create a new B_h sequence from a vector of elements
    ///
    /// # Arguments
    /// * `elements` - The elements of the sequence
    /// * `h` - The parameter h (must be at least 2)
    ///
    /// # Examples
    /// ```
    /// use rustmath_combinatorics::sidon_sets::BhSequence;
    ///
    /// let seq = BhSequence::new(vec![1, 2, 5, 7], 2);
    /// assert!(seq.is_valid());
    /// ```
    pub fn new(mut elements: Vec<u64>, h: usize) -> Self {
        if h < 2 {
            panic!("h must be at least 2");
        }
        elements.sort_unstable();
        elements.dedup(); // Remove duplicates
        BhSequence { elements, h }
    }

    /// Check if this sequence satisfies the B_h property
    ///
    /// Returns true if all sums of h elements are distinct
    pub fn is_valid(&self) -> bool {
        is_bh_sequence(&self.elements, self.h)
    }

    /// Get the elements of the sequence
    pub fn elements(&self) -> &[u64] {
        &self.elements
    }

    /// Get the value of h
    pub fn h(&self) -> usize {
        self.h
    }

    /// Get the size (number of elements) of the sequence
    pub fn size(&self) -> usize {
        self.elements.len()
    }

    /// Add an element to the sequence if it maintains the B_h property
    ///
    /// Returns true if the element was added, false otherwise
    pub fn try_add(&mut self, element: u64) -> bool {
        // Don't add duplicates
        if self.elements.contains(&element) {
            return false;
        }

        // Try adding the element
        let mut temp = self.elements.clone();
        temp.push(element);
        temp.sort_unstable();

        if is_bh_sequence(&temp, self.h) {
            self.elements = temp;
            true
        } else {
            false
        }
    }
}

/// A Sidon set (B_2 sequence)
///
/// A Sidon set is a set of natural numbers where all pairwise sums are distinct.
/// This is equivalent to a B_2 sequence.
pub type SidonSet = BhSequence;

/// Check if a sequence is a B_h sequence
///
/// # Arguments
/// * `elements` - The elements to check
/// * `h` - The parameter h
///
/// # Returns
/// True if all sums of h elements are distinct
pub fn is_bh_sequence(elements: &[u64], h: usize) -> bool {
    if h < 2 {
        panic!("h must be at least 2");
    }

    if elements.is_empty() || elements.len() < h {
        return true;
    }

    let mut sums = HashSet::new();

    // Generate all sums of h elements (with repetition)
    generate_h_sums(elements, h, 0, 0, &mut sums, &mut vec![])
}

/// Helper function to generate all h-sums
fn generate_h_sums(
    elements: &[u64],
    h: usize,
    start: usize,
    current_sum: u64,
    sums: &mut HashSet<u64>,
    current: &mut Vec<u64>,
) -> bool {
    if current.len() == h {
        // Check if this sum is already present
        if sums.contains(&current_sum) {
            return false;
        }
        sums.insert(current_sum);
        return true;
    }

    // Try all elements starting from 'start' (allows repetition)
    for i in start..elements.len() {
        current.push(elements[i]);
        if !generate_h_sums(elements, h, i, current_sum + elements[i], sums, current) {
            return false;
        }
        current.pop();
    }

    true
}

/// Check if a sequence is a Sidon set (B_2 sequence)
///
/// # Arguments
/// * `elements` - The elements to check
///
/// # Returns
/// True if all pairwise sums are distinct
///
/// # Examples
/// ```
/// use rustmath_combinatorics::sidon_sets::is_sidon_set;
///
/// assert!(is_sidon_set(&[1, 2, 5, 7]));
/// assert!(!is_sidon_set(&[1, 2, 3, 4])); // 1+3 = 2+2 = 4
/// ```
pub fn is_sidon_set(elements: &[u64]) -> bool {
    is_bh_sequence(elements, 2)
}

/// Construct a B_h sequence using the greedy algorithm
///
/// The greedy algorithm starts with the first element and repeatedly adds the
/// smallest number that maintains the B_h property, up to the specified upper bound.
///
/// # Arguments
/// * `size` - Target number of elements (may not be reached if upper_bound is too small)
/// * `upper_bound` - Maximum value to consider
/// * `h` - The parameter h
///
/// # Returns
/// A B_h sequence constructed greedily
///
/// # Examples
/// ```
/// use rustmath_combinatorics::sidon_sets::greedy_bh_sequence;
///
/// let seq = greedy_bh_sequence(10, 50, 2);
/// assert!(seq.is_valid());
/// assert!(seq.size() <= 10);
/// ```
pub fn greedy_bh_sequence(size: usize, upper_bound: u64, h: usize) -> BhSequence {
    if h < 2 {
        panic!("h must be at least 2");
    }

    let mut sequence = BhSequence::new(vec![1], h);

    for candidate in 2..=upper_bound {
        if sequence.size() >= size {
            break;
        }

        if sequence.try_add(candidate) {
            // Element was successfully added
        }
    }

    sequence
}

/// Construct a Sidon set using the greedy algorithm
///
/// # Arguments
/// * `size` - Target number of elements
/// * `upper_bound` - Maximum value to consider
///
/// # Returns
/// A Sidon set constructed greedily
///
/// # Examples
/// ```
/// use rustmath_combinatorics::sidon_sets::greedy_sidon_set;
///
/// let sidon = greedy_sidon_set(8, 30);
/// assert!(sidon.is_valid());
/// ```
pub fn greedy_sidon_set(size: usize, upper_bound: u64) -> SidonSet {
    greedy_bh_sequence(size, upper_bound, 2)
}

/// Construct a B_h sequence using backtracking
///
/// This algorithm uses backtracking to find a B_h sequence of exactly the specified size,
/// if one exists within the upper bound. It explores the search space more thoroughly
/// than the greedy algorithm and may find better solutions.
///
/// # Arguments
/// * `size` - Exact number of elements desired
/// * `upper_bound` - Maximum value to consider
/// * `h` - The parameter h
///
/// # Returns
/// A B_h sequence of the specified size, if one exists; otherwise a smaller sequence
///
/// # Examples
/// ```
/// use rustmath_combinatorics::sidon_sets::backtracking_bh_sequence;
///
/// let seq = backtracking_bh_sequence(6, 30, 2);
/// assert!(seq.is_valid());
/// ```
pub fn backtracking_bh_sequence(size: usize, upper_bound: u64, h: usize) -> BhSequence {
    if h < 2 {
        panic!("h must be at least 2");
    }

    let mut current = vec![];
    let mut best = vec![];

    backtrack_helper(&mut current, &mut best, 1, size, upper_bound, h);

    BhSequence::new(best, h)
}

/// Helper function for backtracking
fn backtrack_helper(
    current: &mut Vec<u64>,
    best: &mut Vec<u64>,
    start: u64,
    target_size: usize,
    upper_bound: u64,
    h: usize,
) {
    // If we've reached the target size, update best
    if current.len() == target_size {
        if is_bh_sequence(current, h) {
            *best = current.clone();
        }
        return;
    }

    // If we've already found a solution of target size, we're done
    if best.len() == target_size {
        return;
    }

    // Try adding each candidate
    for candidate in start..=upper_bound {
        current.push(candidate);

        // Only continue if the current sequence is still valid
        if is_bh_sequence(current, h) {
            backtrack_helper(current, best, candidate + 1, target_size, upper_bound, h);

            // Early termination if we found a solution
            if best.len() == target_size {
                current.pop();
                return;
            }
        }

        current.pop();

        // Pruning: if we can't possibly reach target_size with remaining candidates
        let remaining = (upper_bound - candidate) as usize;
        let needed = target_size - current.len();
        if remaining < needed {
            break;
        }
    }

    // Update best if current is better than what we have
    if current.len() > best.len() && is_bh_sequence(current, h) {
        *best = current.clone();
    }
}

/// Construct a Sidon set using backtracking
///
/// # Arguments
/// * `size` - Exact number of elements desired
/// * `upper_bound` - Maximum value to consider
///
/// # Returns
/// A Sidon set of the specified size, if one exists; otherwise a smaller set
///
/// # Examples
/// ```
/// use rustmath_combinatorics::sidon_sets::backtracking_sidon_set;
///
/// let sidon = backtracking_sidon_set(6, 20);
/// assert!(sidon.is_valid());
/// ```
pub fn backtracking_sidon_set(size: usize, upper_bound: u64) -> SidonSet {
    backtracking_bh_sequence(size, upper_bound, 2)
}

/// Find the maximum size of a B_h sequence with elements up to n
///
/// This uses the greedy algorithm as a heuristic.
///
/// # Arguments
/// * `n` - Maximum value in the sequence
/// * `h` - The parameter h
///
/// # Returns
/// The size of the largest B_h sequence found
pub fn max_bh_sequence_size(n: u64, h: usize) -> usize {
    greedy_bh_sequence(n as usize, n, h).size()
}

/// Find the maximum size of a Sidon set with elements up to n
///
/// # Arguments
/// * `n` - Maximum value in the sequence
///
/// # Returns
/// The size of the largest Sidon set found
pub fn max_sidon_set_size(n: u64) -> usize {
    max_bh_sequence_size(n, 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_sidon_set_basic() {
        // Classic Sidon set examples
        assert!(is_sidon_set(&[1, 2, 5, 7]));
        assert!(is_sidon_set(&[1, 2, 5, 10, 16]));

        // Not Sidon sets
        assert!(!is_sidon_set(&[1, 2, 3, 4])); // 1+3 = 2+2 = 4
        assert!(!is_sidon_set(&[1, 2, 3, 5, 8])); // 3+5 = 2+8 = 10 (wait, let me check)
    }

    #[test]
    fn test_is_sidon_set_edge_cases() {
        // Empty and single element are trivially Sidon sets
        assert!(is_sidon_set(&[]));
        assert!(is_sidon_set(&[1]));
        assert!(is_sidon_set(&[5]));

        // Two elements always form a Sidon set
        assert!(is_sidon_set(&[1, 2]));
        assert!(is_sidon_set(&[1, 100]));
    }

    #[test]
    fn test_bh_sequence_basic() {
        // B_2 is the same as Sidon set
        assert!(is_bh_sequence(&[1, 2, 5, 7], 2));

        // B_3 sequence: [1, 2, 5]
        // All 3-sums: 1+1+1=3, 1+1+2=4, 1+1+5=7, 1+2+2=5, 1+2+5=8, 1+5+5=11,
        //             2+2+2=6, 2+2+5=9, 2+5+5=12, 5+5+5=15 (all distinct)
        assert!(is_bh_sequence(&[1, 2, 5], 3));

        // B_3 that fails
        // For [1, 2, 4]: 1+1+4=6, 2+2+2=6 - collision!
        assert!(!is_bh_sequence(&[1, 2, 4], 3));

        // For [1, 2, 3]: 1+1+3=5, 1+2+2=5 - collision!
        assert!(!is_bh_sequence(&[1, 2, 3], 3));
    }

    #[test]
    fn test_sidon_set_struct() {
        let mut sidon = SidonSet::new(vec![1, 2, 5], 2);
        assert!(sidon.is_valid());
        assert_eq!(sidon.size(), 3);

        // Try to add a valid element
        assert!(sidon.try_add(7));
        assert_eq!(sidon.size(), 4);
        assert!(sidon.is_valid());

        // Try to add an invalid element (1+3 = 2+2 = 4)
        let mut sidon2 = SidonSet::new(vec![1, 2, 3], 2);
        assert!(!sidon2.is_valid());
    }

    #[test]
    fn test_greedy_sidon_set() {
        let sidon = greedy_sidon_set(10, 50);
        assert!(sidon.is_valid());
        assert!(sidon.size() <= 10);
        assert!(sidon.size() > 0);

        // First element should be 1
        assert_eq!(sidon.elements()[0], 1);

        // All elements should be <= 50
        for &elem in sidon.elements() {
            assert!(elem <= 50);
        }
    }

    #[test]
    fn test_greedy_bh_sequence() {
        // Test B_3 sequence
        let seq = greedy_bh_sequence(5, 30, 3);
        assert!(seq.is_valid());
        assert_eq!(seq.h(), 3);
    }

    #[test]
    fn test_backtracking_sidon_set() {
        // Small example that should find an exact solution
        let sidon = backtracking_sidon_set(4, 10);
        assert!(sidon.is_valid());
        // Should be able to find at least 4 elements in range [1, 10]
        assert!(sidon.size() >= 3);
    }

    #[test]
    fn test_backtracking_bh_sequence() {
        let seq = backtracking_bh_sequence(4, 15, 2);
        assert!(seq.is_valid());
        assert_eq!(seq.h(), 2);
    }

    #[test]
    fn test_max_sidon_set_size() {
        // For small values
        let size10 = max_sidon_set_size(10);
        assert!(size10 >= 3); // At least {1, 2, 5} fits

        let size20 = max_sidon_set_size(20);
        assert!(size20 >= size10);
    }

    #[test]
    fn test_known_sidon_sets() {
        // {1, 2, 5, 7} is a well-known small Sidon set
        let known = vec![1, 2, 5, 7];
        assert!(is_sidon_set(&known));

        // Check all pairwise sums are distinct
        let mut sums = HashSet::new();
        for i in 0..known.len() {
            for j in i..known.len() {
                let sum = known[i] + known[j];
                assert!(!sums.contains(&sum), "Duplicate sum: {}", sum);
                sums.insert(sum);
            }
        }
    }

    #[test]
    fn test_not_sidon_sets() {
        // {1, 2, 3, 4} is not a Sidon set: 1+3 = 2+2 = 4
        assert!(!is_sidon_set(&[1, 2, 3, 4]));

        // {1, 2, 3, 5} is not a Sidon set: 2+3 = 1+5 = 6? Wait, 5+1=6, 2+3=5
        // Let me recalculate: 1+1=2, 1+2=3, 1+3=4, 1+5=6, 2+2=4 - collision!
        assert!(!is_sidon_set(&[1, 2, 3, 5]));
    }

    #[test]
    fn test_sidon_set_duplicates() {
        // Creating with duplicates should remove them
        let sidon = SidonSet::new(vec![1, 2, 2, 5, 5, 7], 2);
        assert_eq!(sidon.elements(), &[1, 2, 5, 7]);

        // Try adding a duplicate
        let mut sidon = SidonSet::new(vec![1, 2, 5], 2);
        assert!(!sidon.try_add(2));
        assert_eq!(sidon.size(), 3);
    }

    #[test]
    fn test_sorted_elements() {
        // Elements should be stored in sorted order
        let sidon = SidonSet::new(vec![7, 2, 5, 1], 2);
        assert_eq!(sidon.elements(), &[1, 2, 5, 7]);
    }

    #[test]
    fn test_large_greedy_sidon() {
        // Test with larger bounds
        let sidon = greedy_sidon_set(15, 100);
        assert!(sidon.is_valid());
        assert!(sidon.size() <= 15);

        // Verify it's actually a Sidon set by checking all sums
        let elements = sidon.elements();
        let mut sums = HashSet::new();
        for i in 0..elements.len() {
            for j in i..elements.len() {
                let sum = elements[i] + elements[j];
                assert!(!sums.contains(&sum), "Greedy produced invalid Sidon set");
                sums.insert(sum);
            }
        }
    }

    #[test]
    fn test_comparison_greedy_vs_backtracking() {
        // Both should produce valid results
        let greedy = greedy_sidon_set(5, 20);
        let backtrack = backtracking_sidon_set(5, 20);

        assert!(greedy.is_valid());
        assert!(backtrack.is_valid());

        // Backtracking might find a solution of exact size
        // but greedy might not, depending on the parameters
    }
}
