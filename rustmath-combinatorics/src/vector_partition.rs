//! Vector partitions - partitions of vectors with non-negative integer components
//!
//! A vector partition of a vector v = (v₁, v₂, ..., vₖ) is a multiset of vectors
//! that sum componentwise to v. This is a generalization of integer partitions.
//!
//! This module provides fast counting algorithms using dynamic programming with
//! memoization for efficient computation.

use std::collections::HashMap;
use std::hash::Hash;

/// A vector partition represented as a multiset of vectors
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VectorPartition {
    /// The parts of the partition (each part is a vector)
    parts: Vec<Vec<usize>>,
}

impl VectorPartition {
    /// Create a new vector partition from parts
    ///
    /// The parts will be sorted in reverse lexicographic order for canonical representation
    pub fn new(mut parts: Vec<Vec<usize>>) -> Self {
        // Remove zero vectors
        parts.retain(|p| p.iter().any(|&x| x > 0));

        // Sort in reverse lexicographic order for canonical form
        parts.sort_by(|a, b| b.cmp(a));

        VectorPartition { parts }
    }

    /// Get the parts of this partition
    pub fn parts(&self) -> &[Vec<usize>] {
        &self.parts
    }

    /// Get the number of parts
    pub fn num_parts(&self) -> usize {
        self.parts.len()
    }

    /// Compute the sum (the vector being partitioned)
    pub fn sum(&self) -> Vec<usize> {
        if self.parts.is_empty() {
            return vec![];
        }

        let dim = self.parts[0].len();
        let mut result = vec![0; dim];

        for part in &self.parts {
            for (i, &val) in part.iter().enumerate() {
                result[i] += val;
            }
        }

        result
    }

    /// Check if this partition is valid for the given vector
    pub fn is_partition_of(&self, vector: &[usize]) -> bool {
        self.sum() == vector
    }
}

/// Count vector partitions using dynamic programming with memoization
///
/// This function counts the number of ways to partition a vector into parts,
/// where parts are ordered lexicographically to avoid duplicates.
///
/// # Arguments
/// * `vector` - The vector to partition
///
/// # Returns
/// The number of distinct vector partitions
///
/// # Example
/// ```
/// use rustmath_combinatorics::vector_partition::fast_vector_partitions;
///
/// // Count partitions of (2, 1)
/// let count = fast_vector_partitions(&[2, 1]);
/// assert!(count > 0);
/// ```
pub fn fast_vector_partitions(vector: &[usize]) -> usize {
    // Generate all possible parts once
    let all_parts = generate_all_parts(vector);
    let mut memo = HashMap::new();
    count_partitions_with_parts(vector, &all_parts, 0, &mut memo)
}

/// Count vector partitions with a maximum part constraint
///
/// # Arguments
/// * `vector` - The vector to partition
/// * `max_part` - Maximum part allowed (in lexicographic order), or None for no constraint
///
/// # Returns
/// The number of distinct vector partitions
pub fn fast_vector_partitions_with_max_part(vector: &[usize], max_part: Option<&[usize]>) -> usize {
    let actual_max = max_part.unwrap_or(vector);
    let all_parts = generate_all_parts(actual_max);
    let mut memo = HashMap::new();
    count_partitions_with_parts(vector, &all_parts, 0, &mut memo)
}

/// Count partitions using a fixed list of available parts
///
/// This uses the standard partition counting approach: for each part in the list,
/// count partitions that use it (recursively, allowing repetition) plus partitions
/// that skip it entirely.
fn count_partitions_with_parts(
    vector: &[usize],
    available_parts: &[Vec<usize>],
    part_index: usize,
    memo: &mut HashMap<(Vec<usize>, usize), usize>,
) -> usize {
    // Base case: zero vector
    if vector.iter().all(|&x| x == 0) {
        return 1;
    }

    // No more parts to try
    if part_index >= available_parts.len() {
        return 0;
    }

    // Check memo
    let key = (vector.to_vec(), part_index);
    if let Some(&count) = memo.get(&key) {
        return count;
    }

    let current_part = &available_parts[part_index];
    let mut total = 0;

    // Try using current_part 0, 1, 2, ... times
    let mut k = 0;
    loop {
        // Check if we can subtract k copies of current_part from vector
        let mut can_subtract = true;
        let mut remainder = vector.to_vec();

        for _ in 0..k {
            let mut all_non_negative = true;
            for (i, &part_val) in current_part.iter().enumerate() {
                if remainder[i] < part_val {
                    all_non_negative = false;
                    break;
                }
                remainder[i] -= part_val;
            }

            if !all_non_negative {
                can_subtract = false;
                break;
            }
        }

        if !can_subtract {
            break;
        }

        // Count partitions of remainder using parts from part_index+1 onwards
        total += count_partitions_with_parts(&remainder, available_parts, part_index + 1, memo);

        k += 1;

        // Safety check to avoid infinite loops
        if k > 1000 {
            break;
        }
    }

    memo.insert(key, total);
    total
}

/// Generate all non-zero vectors with components in [0, max_vec[i]]
fn generate_all_parts(max_vec: &[usize]) -> Vec<Vec<usize>> {
    let mut parts = Vec::new();
    let dim = max_vec.len();

    generate_vectors_recursive(max_vec, 0, vec![0; dim], &mut parts);

    // Remove zero vector
    parts.retain(|v| v.iter().any(|&x| x > 0));

    // Sort in reverse lexicographic order
    parts.sort_by(|a, b| b.cmp(a));

    parts
}

/// Generate all non-zero vectors componentwise <= both max_vec and constraint
fn generate_parts_up_to(max_vec: &[usize], constraint: &[usize]) -> Vec<Vec<usize>> {
    let dim = max_vec.len();
    let mut parts = Vec::new();

    // Compute actual maximum for each component
    let actual_max: Vec<usize> = max_vec
        .iter()
        .zip(constraint.iter())
        .map(|(&m, &c)| m.min(c))
        .collect();

    // Generate all vectors with components in range [0, actual_max[i]]
    generate_vectors_recursive(&actual_max, 0, vec![0; dim], &mut parts);

    // Remove the zero vector
    parts.retain(|v| v.iter().any(|&x| x > 0));

    // Sort in reverse lexicographic order
    parts.sort_by(|a, b| b.cmp(a));

    parts
}

/// Recursively generate all vectors with components in specified ranges
fn generate_vectors_recursive(
    max_vals: &[usize],
    index: usize,
    current: Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if index == max_vals.len() {
        result.push(current);
        return;
    }

    for val in 0..=max_vals[index] {
        let mut next = current.clone();
        next[index] = val;
        generate_vectors_recursive(max_vals, index + 1, next, result);
    }
}

/// Generate all vector partitions (not just count them)
///
/// Warning: This can be very slow for large vectors!
///
/// # Arguments
/// * `vector` - The vector to partition
///
/// # Returns
/// A vector of all distinct vector partitions
///
/// # Example
/// ```
/// use rustmath_combinatorics::vector_partition::vector_partitions;
///
/// // Generate all partitions of (2, 1)
/// let partitions = vector_partitions(&[2, 1]);
/// assert!(partitions.len() > 0);
/// ```
pub fn vector_partitions(vector: &[usize]) -> Vec<VectorPartition> {
    let all_parts = generate_all_parts(vector);
    let mut result = Vec::new();
    generate_partitions_with_parts(vector, &all_parts, 0, Vec::new(), &mut result);
    result
}

/// Generate vector partitions with a maximum part constraint
pub fn vector_partitions_with_max_part(
    vector: &[usize],
    max_part: Option<&[usize]>,
) -> Vec<VectorPartition> {
    let actual_max = max_part.unwrap_or(vector);
    let all_parts = generate_all_parts(actual_max);
    let mut result = Vec::new();
    generate_partitions_with_parts(vector, &all_parts, 0, Vec::new(), &mut result);
    result
}

/// Internal recursive generation function
fn generate_partitions_with_parts(
    vector: &[usize],
    available_parts: &[Vec<usize>],
    part_index: usize,
    current_parts: Vec<Vec<usize>>,
    result: &mut Vec<VectorPartition>,
) {
    // Base case: zero vector
    if vector.iter().all(|&x| x == 0) {
        result.push(VectorPartition::new(current_parts));
        return;
    }

    // No more parts to try
    if part_index >= available_parts.len() {
        return;
    }

    let current_part = &available_parts[part_index];

    // Try using current_part 0, 1, 2, ... times
    let mut k = 0;
    loop {
        // Check if we can subtract k copies of current_part
        let mut can_subtract = true;
        let mut remainder = vector.to_vec();

        for _ in 0..k {
            let mut all_non_negative = true;
            for (i, &part_val) in current_part.iter().enumerate() {
                if remainder[i] < part_val {
                    all_non_negative = false;
                    break;
                }
                remainder[i] -= part_val;
            }

            if !all_non_negative {
                can_subtract = false;
                break;
            }
        }

        if !can_subtract {
            break;
        }

        // Build new parts list with k copies of current_part
        let mut next_parts = current_parts.clone();
        for _ in 0..k {
            next_parts.push(current_part.clone());
        }

        // Recurse with parts from part_index+1 onwards
        generate_partitions_with_parts(&remainder, available_parts, part_index + 1, next_parts, result);

        k += 1;

        // Safety check
        if k > 1000 {
            break;
        }
    }
}

/// Count vector partitions with constraint on number of parts
///
/// # Arguments
/// * `vector` - The vector to partition
/// * `max_parts` - Maximum number of parts allowed
///
/// # Returns
/// The number of vector partitions with at most `max_parts` parts
pub fn count_vector_partitions_max_parts(vector: &[usize], max_parts: usize) -> usize {
    if max_parts == 0 {
        return if vector.iter().all(|&x| x == 0) {
            1
        } else {
            0
        };
    }

    let all_parts = generate_all_parts(vector);
    let mut memo = HashMap::new();
    count_with_max_parts_helper(vector, &all_parts, 0, max_parts, &mut memo)
}

fn count_with_max_parts_helper(
    vector: &[usize],
    available_parts: &[Vec<usize>],
    part_index: usize,
    max_parts: usize,
    memo: &mut HashMap<(Vec<usize>, usize, usize), usize>,
) -> usize {
    // Base case
    if vector.iter().all(|&x| x == 0) {
        return 1;
    }

    if max_parts == 0 || part_index >= available_parts.len() {
        return 0;
    }

    // Check memo
    let key = (vector.to_vec(), part_index, max_parts);
    if let Some(&count) = memo.get(&key) {
        return count;
    }

    let current_part = &available_parts[part_index];
    let mut total = 0;

    // Try using current_part k times where k <= max_parts
    for k in 0..=max_parts {
        // Check if we can subtract k copies
        let mut can_subtract = true;
        let mut remainder = vector.to_vec();

        for _ in 0..k {
            let mut all_non_negative = true;
            for (i, &part_val) in current_part.iter().enumerate() {
                if remainder[i] < part_val {
                    all_non_negative = false;
                    break;
                }
                remainder[i] -= part_val;
            }

            if !all_non_negative {
                can_subtract = false;
                break;
            }
        }

        if !can_subtract {
            break;
        }

        // Count partitions of remainder with max_parts - k parts remaining
        total += count_with_max_parts_helper(&remainder, available_parts, part_index + 1, max_parts - k, memo);
    }

    memo.insert(key, total);
    total
}

/// Count vector partitions with exactly k parts
///
/// # Arguments
/// * `vector` - The vector to partition
/// * `k` - Exact number of parts required
///
/// # Returns
/// The number of vector partitions with exactly `k` parts
pub fn count_vector_partitions_exact_parts(vector: &[usize], k: usize) -> usize {
    if k == 0 {
        return if vector.iter().all(|&x| x == 0) {
            1
        } else {
            0
        };
    }

    let all_parts = generate_all_parts(vector);
    let mut memo = HashMap::new();
    count_with_exact_parts_helper(vector, &all_parts, 0, k, &mut memo)
}

fn count_with_exact_parts_helper(
    vector: &[usize],
    available_parts: &[Vec<usize>],
    part_index: usize,
    exact_parts: usize,
    memo: &mut HashMap<(Vec<usize>, usize, usize), usize>,
) -> usize {
    // Base case
    if exact_parts == 0 {
        return if vector.iter().all(|&x| x == 0) {
            1
        } else {
            0
        };
    }

    if vector.iter().all(|&x| x == 0) || part_index >= available_parts.len() {
        return 0;
    }

    // Check memo
    let key = (vector.to_vec(), part_index, exact_parts);
    if let Some(&count) = memo.get(&key) {
        return count;
    }

    let current_part = &available_parts[part_index];
    let mut total = 0;

    // Try using current_part k times where k <= exact_parts
    for k in 0..=exact_parts {
        // Check if we can subtract k copies
        let mut can_subtract = true;
        let mut remainder = vector.to_vec();

        for _ in 0..k {
            let mut all_non_negative = true;
            for (i, &part_val) in current_part.iter().enumerate() {
                if remainder[i] < part_val {
                    all_non_negative = false;
                    break;
                }
                remainder[i] -= part_val;
            }

            if !all_non_negative {
                can_subtract = false;
                break;
            }
        }

        if !can_subtract {
            break;
        }

        // Count partitions of remainder with exactly (exact_parts - k) parts
        total += count_with_exact_parts_helper(&remainder, available_parts, part_index + 1, exact_parts - k, memo);
    }

    memo.insert(key, total);
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_partition_creation() {
        let vp = VectorPartition::new(vec![vec![2, 1], vec![1, 0], vec![0, 1]]);

        // Check that parts are sorted in reverse lex order
        assert_eq!(vp.parts()[0], vec![2, 1]);

        // Check sum
        assert_eq!(vp.sum(), vec![3, 2]);
    }

    #[test]
    fn test_zero_vector() {
        // Zero vector has exactly 1 partition (empty)
        assert_eq!(fast_vector_partitions(&[0, 0]), 1);
        assert_eq!(fast_vector_partitions(&[0, 0, 0]), 1);
    }

    #[test]
    fn test_single_component_reduces_to_integer_partition() {
        // Vector partitions of a 1D vector should match integer partitions
        // For n=4, p(4) = 5: [4], [3,1], [2,2], [2,1,1], [1,1,1,1]
        assert_eq!(fast_vector_partitions(&[4]), 5);

        // For n=5, p(5) = 7
        assert_eq!(fast_vector_partitions(&[5]), 7);
    }

    #[test]
    fn test_small_2d_vectors() {
        // (1, 0) has 1 partition: [(1, 0)]
        assert_eq!(fast_vector_partitions(&[1, 0]), 1);

        // (0, 1) has 1 partition: [(0, 1)]
        assert_eq!(fast_vector_partitions(&[0, 1]), 1);

        // (1, 1) has partitions:
        // [(1, 1)]
        // [(1, 0), (0, 1)]
        assert_eq!(fast_vector_partitions(&[1, 1]), 2);
    }

    #[test]
    fn test_2d_vector_2_1() {
        // Test a slightly larger case
        let count = fast_vector_partitions(&[2, 1]);

        // Verify by enumeration
        let partitions = vector_partitions(&[2, 1]);
        assert_eq!(count, partitions.len());

        // Should have multiple partitions
        assert!(count >= 4);

        // All partitions should sum to (2, 1)
        for p in &partitions {
            assert_eq!(p.sum(), vec![2, 1]);
        }
    }

    #[test]
    fn test_3d_vector() {
        // Test a 3D vector
        let count = fast_vector_partitions(&[1, 1, 1]);
        assert!(count > 0);

        // Verify by enumeration
        let partitions = vector_partitions(&[1, 1, 1]);
        assert_eq!(count, partitions.len());
    }

    #[test]
    fn test_max_parts_constraint() {
        // Partition (2, 2) with at most 1 part
        // Only partition: [(2, 2)]
        assert_eq!(count_vector_partitions_max_parts(&[2, 2], 1), 1);

        // Partition (2, 2) with at most 2 parts
        // Should have more options
        let count_2 = count_vector_partitions_max_parts(&[2, 2], 2);
        assert!(count_2 > 1);

        // With at most 3 parts should be >= with at most 2 parts
        let count_3 = count_vector_partitions_max_parts(&[2, 2], 3);
        assert!(count_3 >= count_2);
    }

    #[test]
    fn test_exact_parts_constraint() {
        // (2, 0) with exactly 1 part: [(2, 0)]
        assert_eq!(count_vector_partitions_exact_parts(&[2, 0], 1), 1);

        // (2, 0) with exactly 2 parts: [(1, 0), (1, 0)]
        assert_eq!(count_vector_partitions_exact_parts(&[2, 0], 2), 1);

        // (1, 1) with exactly 1 part: [(1, 1)]
        assert_eq!(count_vector_partitions_exact_parts(&[1, 1], 1), 1);

        // (1, 1) with exactly 2 parts: [(1, 0), (0, 1)]
        assert_eq!(count_vector_partitions_exact_parts(&[1, 1], 2), 1);
    }

    #[test]
    fn test_memoization_performance() {
        // This should complete quickly due to memoization
        let count1 = fast_vector_partitions(&[3, 3]);
        let count2 = fast_vector_partitions(&[3, 3]);
        assert_eq!(count1, count2);

        // Larger example
        let count3 = fast_vector_partitions(&[4, 3]);
        assert!(count3 > 0);
    }

    #[test]
    fn test_symmetry() {
        // Swapping components should give same count
        let count1 = fast_vector_partitions(&[2, 3]);
        let count2 = fast_vector_partitions(&[3, 2]);

        // They should be equal due to symmetry
        assert_eq!(count1, count2);
    }

    #[test]
    fn test_generate_parts() {
        let parts = generate_parts_up_to(&[2, 1], &[2, 1]);

        // Should contain various vectors
        assert!(parts.contains(&vec![2, 1]));
        assert!(parts.contains(&vec![2, 0]));
        assert!(parts.contains(&vec![1, 1]));
        assert!(parts.contains(&vec![1, 0]));
        assert!(parts.contains(&vec![0, 1]));

        // Should not contain zero vector
        assert!(!parts.contains(&vec![0, 0]));

        // Should be sorted in reverse lex order
        for i in 0..parts.len() - 1 {
            assert!(parts[i] > parts[i + 1]);
        }
    }

    #[test]
    fn test_enumeration_correctness() {
        // For small vectors, verify enumeration matches count
        for &n in &[1, 2, 3] {
            for &m in &[1, 2, 3] {
                let count = fast_vector_partitions(&[n, m]);
                let partitions = vector_partitions(&[n, m]);
                assert_eq!(
                    count,
                    partitions.len(),
                    "Mismatch for ({}, {})",
                    n,
                    m
                );
            }
        }
    }

    #[test]
    fn test_partition_validity() {
        let partitions = vector_partitions(&[2, 2]);

        for p in &partitions {
            // Each partition should sum to (2, 2)
            assert!(
                p.is_partition_of(&[2, 2]),
                "Partition {:?} doesn't sum to (2, 2)",
                p
            );
        }
    }

    #[test]
    fn test_increasing_dimension() {
        // As dimension increases, number of partitions should generally increase
        let count_1d = fast_vector_partitions(&[3]);
        let count_2d = fast_vector_partitions(&[3, 0]);
        let count_3d = fast_vector_partitions(&[3, 0, 0]);

        // 1D case is just integer partitions
        assert_eq!(count_1d, count_2d);
        assert_eq!(count_2d, count_3d);
    }

    #[test]
    fn test_max_part_constraint() {
        // Test with explicit max_part
        let count_unrestricted = fast_vector_partitions(&[2, 2]);
        let count_restricted = fast_vector_partitions_with_max_part(&[2, 2], Some(&[1, 1]));

        // Restricted should have fewer (or equal) partitions
        assert!(count_restricted <= count_unrestricted);
    }
}
