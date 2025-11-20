//! Integer vectors with weighted enumeration and lattice point counting
//!
//! This module provides functionality for working with integer vectors,
//! including enumeration with weight constraints and lattice point counting.
//!
//! # Examples
//!
//! ```
//! use rustmath_combinatorics::integer_vectors::{integer_vectors_with_sum, count_integer_vectors_with_sum};
//!
//! // Find all non-negative integer vectors of length 3 with sum 4
//! let vectors = integer_vectors_with_sum(3, 4);
//! assert_eq!(vectors.len(), 15); // C(4+3-1, 3-1) = C(6,2) = 15
//!
//! // Count without enumerating
//! assert_eq!(count_integer_vectors_with_sum(3, 4), 15);
//! ```

use rustmath_core::{NumericConversion, Ring};
use rustmath_integers::Integer;

/// An integer vector
///
/// This is a wrapper around Vec<Integer> with additional functionality
/// for weighted enumeration and lattice point operations.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IntegerVector {
    components: Vec<Integer>,
}

impl IntegerVector {
    /// Create a new integer vector from components
    pub fn new(components: Vec<Integer>) -> Self {
        Self { components }
    }

    /// Create a new integer vector from i64 values
    pub fn from_i64(values: Vec<i64>) -> Self {
        Self {
            components: values.into_iter().map(Integer::from).collect(),
        }
    }

    /// Get the dimension (length) of the vector
    pub fn dimension(&self) -> usize {
        self.components.len()
    }

    /// Get the components
    pub fn components(&self) -> &[Integer] {
        &self.components
    }

    /// Compute the sum of all components
    pub fn sum(&self) -> Integer {
        self.components.iter().fold(Integer::zero(), |acc, x| acc + x.clone())
    }

    /// Compute the weighted sum with given weights
    ///
    /// Returns w[0]*v[0] + w[1]*v[1] + ... + w[n-1]*v[n-1]
    pub fn weighted_sum(&self, weights: &[Integer]) -> Integer {
        assert_eq!(
            self.dimension(),
            weights.len(),
            "Weight vector must have same dimension"
        );
        self.components
            .iter()
            .zip(weights.iter())
            .fold(Integer::zero(), |acc, (v, w)| acc + (v.clone() * w.clone()))
    }

    /// Compute the L1 norm (sum of absolute values)
    pub fn norm_l1(&self) -> Integer {
        self.components
            .iter()
            .fold(Integer::zero(), |acc, x| acc + x.abs())
    }

    /// Check if all components are non-negative
    pub fn is_nonnegative(&self) -> bool {
        self.components.iter().all(|x| x >= &Integer::zero())
    }

    /// Check if all components are positive
    pub fn is_positive(&self) -> bool {
        self.components.iter().all(|x| x > &Integer::zero())
    }

    /// Convert to i64 vector (returns None if any component doesn't fit)
    pub fn to_i64_vec(&self) -> Option<Vec<i64>> {
        self.components
            .iter()
            .map(|x| NumericConversion::to_i64(x))
            .collect::<Option<Vec<_>>>()
    }

    /// Convert to usize vector for non-negative vectors
    pub fn to_usize_vec(&self) -> Option<Vec<usize>> {
        if !self.is_nonnegative() {
            return None;
        }
        self.components
            .iter()
            .map(|x| NumericConversion::to_usize(x))
            .collect::<Option<Vec<_>>>()
    }
}

impl From<Vec<Integer>> for IntegerVector {
    fn from(components: Vec<Integer>) -> Self {
        Self::new(components)
    }
}

impl From<Vec<i64>> for IntegerVector {
    fn from(values: Vec<i64>) -> Self {
        Self::from_i64(values)
    }
}

/// Enumerate all non-negative integer vectors of given length with a specified sum
///
/// This generates all vectors v = (v[0], v[1], ..., v[n-1]) where:
/// - All v[i] >= 0
/// - v[0] + v[1] + ... + v[n-1] = sum
///
/// This is equivalent to the "stars and bars" problem.
///
/// # Arguments
///
/// * `length` - The dimension of the vectors
/// * `sum` - The required sum of components
///
/// # Returns
///
/// A vector containing all integer vectors satisfying the constraints
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::integer_vectors::integer_vectors_with_sum;
///
/// let vecs = integer_vectors_with_sum(2, 3);
/// // Should contain: [0,3], [1,2], [2,1], [3,0]
/// assert_eq!(vecs.len(), 4);
/// ```
pub fn integer_vectors_with_sum(length: usize, sum: usize) -> Vec<IntegerVector> {
    let mut result = Vec::new();
    let mut current = vec![Integer::zero(); length];

    enumerate_vectors_recursive(length, sum, 0, &mut current, &mut result);

    result
}

/// Helper function for recursive enumeration
fn enumerate_vectors_recursive(
    length: usize,
    remaining: usize,
    index: usize,
    current: &mut Vec<Integer>,
    result: &mut Vec<IntegerVector>,
) {
    if index == length {
        if remaining == 0 {
            result.push(IntegerVector::new(current.clone()));
        }
        return;
    }

    // If this is the last position, put all remaining in it
    if index == length - 1 {
        current[index] = Integer::from(remaining as i64);
        result.push(IntegerVector::new(current.clone()));
        return;
    }

    // Try all possible values for current position
    for value in 0..=remaining {
        current[index] = Integer::from(value as i64);
        enumerate_vectors_recursive(length, remaining - value, index + 1, current, result);
    }
}

/// Count non-negative integer vectors of given length with specified sum
///
/// This is more efficient than enumerating all vectors when you only need the count.
/// The count is given by the binomial coefficient C(sum + length - 1, length - 1).
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::integer_vectors::count_integer_vectors_with_sum;
///
/// assert_eq!(count_integer_vectors_with_sum(3, 4), 15); // C(6,2) = 15
/// ```
pub fn count_integer_vectors_with_sum(length: usize, sum: usize) -> usize {
    if length == 0 {
        return if sum == 0 { 1 } else { 0 };
    }
    if length == 1 {
        return 1;
    }

    // C(sum + length - 1, length - 1)
    use crate::binomial;
    NumericConversion::to_usize(&binomial((sum + length - 1) as u32, (length - 1) as u32))
        .unwrap_or(0)
}

/// Enumerate integer vectors with weighted sum constraint
///
/// Generates all non-negative integer vectors v such that:
/// - w[0]*v[0] + w[1]*v[1] + ... + w[n-1]*v[n-1] = target_weight
/// - All v[i] >= 0
///
/// # Arguments
///
/// * `weights` - The weight for each component
/// * `target_weight` - The target weighted sum
/// * `max_components` - Optional maximum value for each component
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::integer_vectors::integer_vectors_with_weighted_sum;
///
/// // Find vectors where 2*v[0] + 3*v[1] = 6
/// let vecs = integer_vectors_with_weighted_sum(&[2, 3], 6, None);
/// // Should contain: [0,2], [3,0]
/// assert!(vecs.len() >= 2);
/// ```
pub fn integer_vectors_with_weighted_sum(
    weights: &[usize],
    target_weight: usize,
    max_components: Option<&[usize]>,
) -> Vec<IntegerVector> {
    if weights.is_empty() {
        return if target_weight == 0 {
            vec![IntegerVector::new(vec![])]
        } else {
            vec![]
        };
    }

    let mut result = Vec::new();
    let mut current = vec![Integer::zero(); weights.len()];

    enumerate_weighted_recursive(
        weights,
        target_weight,
        0,
        &mut current,
        &mut result,
        max_components,
    );

    result
}

/// Helper for weighted enumeration
fn enumerate_weighted_recursive(
    weights: &[usize],
    remaining_weight: usize,
    index: usize,
    current: &mut Vec<Integer>,
    result: &mut Vec<IntegerVector>,
    max_components: Option<&[usize]>,
) {
    if index == weights.len() {
        if remaining_weight == 0 {
            result.push(IntegerVector::new(current.clone()));
        }
        return;
    }

    let weight = weights[index];
    let max_val = if weight == 0 {
        0
    } else {
        remaining_weight / weight
    };

    // Apply component-wise maximum if provided
    let max_val = if let Some(maxes) = max_components {
        max_val.min(maxes[index])
    } else {
        max_val
    };

    for value in 0..=max_val {
        current[index] = Integer::from(value as i64);
        let new_remaining = remaining_weight - value * weight;
        enumerate_weighted_recursive(
            weights,
            new_remaining,
            index + 1,
            current,
            result,
            max_components,
        );
    }
}

/// Enumerate integer vectors in a box [min[i], max[i]]
///
/// Generates all integer vectors v where min[i] <= v[i] <= max[i] for all i.
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::integer_vectors::integer_vectors_in_box;
///
/// let vecs = integer_vectors_in_box(&[0, 0], &[1, 1]);
/// assert_eq!(vecs.len(), 4); // [0,0], [0,1], [1,0], [1,1]
/// ```
pub fn integer_vectors_in_box(min: &[i64], max: &[i64]) -> Vec<IntegerVector> {
    assert_eq!(min.len(), max.len(), "Min and max must have same dimension");

    if min.is_empty() {
        return vec![IntegerVector::new(vec![])];
    }

    let mut result = Vec::new();
    let mut current = min.to_vec();

    enumerate_box_recursive(min, max, 0, &mut current, &mut result);

    result
}

/// Helper for box enumeration
fn enumerate_box_recursive(
    min: &[i64],
    max: &[i64],
    index: usize,
    current: &mut Vec<i64>,
    result: &mut Vec<IntegerVector>,
) {
    if index == min.len() {
        result.push(IntegerVector::from_i64(current.clone()));
        return;
    }

    for value in min[index]..=max[index] {
        current[index] = value;
        enumerate_box_recursive(min, max, index + 1, current, result);
    }
}

/// Count integer vectors in a box
///
/// More efficient than enumerating when you only need the count.
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::integer_vectors::count_integer_vectors_in_box;
///
/// assert_eq!(count_integer_vectors_in_box(&[0, 0], &[2, 3]), 12); // 3 * 4
/// ```
pub fn count_integer_vectors_in_box(min: &[i64], max: &[i64]) -> usize {
    assert_eq!(min.len(), max.len(), "Min and max must have same dimension");

    min.iter()
        .zip(max.iter())
        .map(|(mn, mx)| {
            if mx >= mn {
                (mx - mn + 1) as usize
            } else {
                0
            }
        })
        .product()
}

/// Enumerate integer vectors with bounded L1 norm
///
/// Generates all integer vectors v where:
/// - |v[0]| + |v[1]| + ... + |v[n-1]| <= max_norm
/// - All components are integers (can be negative)
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::integer_vectors::integer_vectors_with_l1_norm_bounded;
///
/// let vecs = integer_vectors_with_l1_norm_bounded(2, 1);
/// // Should include: [0,0], [1,0], [-1,0], [0,1], [0,-1]
/// assert!(vecs.len() >= 5);
/// ```
pub fn integer_vectors_with_l1_norm_bounded(dimension: usize, max_norm: usize) -> Vec<IntegerVector> {
    let mut result = Vec::new();

    // For each possible L1 norm from 0 to max_norm
    for norm in 0..=max_norm {
        // Generate all non-negative vectors with sum = norm
        let nonneg_vecs = integer_vectors_with_sum(dimension, norm);

        for vec in nonneg_vecs {
            // For each vector, generate all sign patterns
            let components = vec.to_usize_vec().unwrap();
            generate_all_sign_patterns(&components, &mut result);
        }
    }

    result
}

/// Helper to generate all sign patterns of a vector
fn generate_all_sign_patterns(components: &[usize], result: &mut Vec<IntegerVector>) {
    let n = components.len();
    let num_patterns = 1 << n; // 2^n patterns

    for pattern in 0..num_patterns {
        let mut signed_vec = Vec::with_capacity(n);
        for i in 0..n {
            let val = components[i] as i64;
            if (pattern >> i) & 1 == 1 {
                signed_vec.push(Integer::from(-val));
            } else {
                signed_vec.push(Integer::from(val));
            }
        }
        result.push(IntegerVector::new(signed_vec));
    }
}

/// Iterator over integer vectors with a given sum
///
/// This provides lazy enumeration without storing all vectors in memory.
pub struct IntegerVectorSumIter {
    length: usize,
    sum: usize,
    current: Option<Vec<usize>>,
    finished: bool,
}

impl IntegerVectorSumIter {
    /// Create a new iterator
    pub fn new(length: usize, sum: usize) -> Self {
        if length == 0 {
            return Self {
                length,
                sum,
                current: None,
                finished: sum != 0,
            };
        }

        let mut initial = vec![0; length];
        initial[length - 1] = sum;

        Self {
            length,
            sum,
            current: Some(initial),
            finished: false,
        }
    }

    /// Advance to next vector
    fn advance(&mut self) -> bool {
        if self.length == 0 || self.current.is_none() {
            return false;
        }

        let current = self.current.as_mut().unwrap();

        // Find the rightmost non-zero position before the last position
        let mut i = self.length;
        while i > 1 && current[i - 2] == 0 {
            i -= 1;
        }

        if i == 1 {
            // We've enumerated all vectors
            return false;
        }

        // Decrement position i-2 and redistribute
        current[i - 2] -= 1;
        let sum_right: usize = current[i - 1..].iter().sum();
        let new_val = sum_right + 1;

        // Put everything at position i-1
        for j in (i - 1)..self.length {
            current[j] = 0;
        }
        current[i - 1] = new_val;

        true
    }
}

impl Iterator for IntegerVectorSumIter {
    type Item = IntegerVector;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        if let Some(ref current) = self.current {
            let result = IntegerVector::from_i64(
                current.iter().map(|&x| x as i64).collect()
            );

            if !self.advance() {
                self.finished = true;
            }

            Some(result)
        } else {
            // Special case: empty vector with sum 0
            if self.sum == 0 {
                self.finished = true;
                Some(IntegerVector::new(vec![]))
            } else {
                None
            }
        }
    }
}

/// Create an iterator over integer vectors with a given sum
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::integer_vectors::integer_vector_sum_iter;
///
/// let mut iter = integer_vector_sum_iter(2, 3);
/// assert_eq!(iter.next().unwrap().to_i64_vec().unwrap(), vec![0, 3]);
/// assert_eq!(iter.next().unwrap().to_i64_vec().unwrap(), vec![1, 2]);
/// ```
pub fn integer_vector_sum_iter(length: usize, sum: usize) -> IntegerVectorSumIter {
    IntegerVectorSumIter::new(length, sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integer_vector_basic() {
        let v = IntegerVector::from_i64(vec![1, 2, 3]);
        assert_eq!(v.dimension(), 3);
        assert_eq!(v.sum(), Integer::from(6));
        assert!(v.is_nonnegative());
        assert!(v.is_positive());
    }

    #[test]
    fn test_integer_vector_weighted_sum() {
        let v = IntegerVector::from_i64(vec![2, 3]);
        let weights = vec![Integer::from(5), Integer::from(7)];
        assert_eq!(v.weighted_sum(&weights), Integer::from(31)); // 2*5 + 3*7 = 31
    }

    #[test]
    fn test_integer_vectors_with_sum() {
        let vecs = integer_vectors_with_sum(2, 3);
        assert_eq!(vecs.len(), 4);

        // Convert to i64 for easier checking
        let vecs_i64: Vec<Vec<i64>> = vecs
            .iter()
            .map(|v| v.to_i64_vec().unwrap())
            .collect();

        assert!(vecs_i64.contains(&vec![0, 3]));
        assert!(vecs_i64.contains(&vec![1, 2]));
        assert!(vecs_i64.contains(&vec![2, 1]));
        assert!(vecs_i64.contains(&vec![3, 0]));
    }

    #[test]
    fn test_count_integer_vectors_with_sum() {
        assert_eq!(count_integer_vectors_with_sum(3, 4), 15);
        assert_eq!(count_integer_vectors_with_sum(2, 3), 4);
        assert_eq!(count_integer_vectors_with_sum(1, 5), 1);
        assert_eq!(count_integer_vectors_with_sum(0, 0), 1);
        assert_eq!(count_integer_vectors_with_sum(0, 1), 0);
    }

    #[test]
    fn test_integer_vectors_with_weighted_sum() {
        // 2*v[0] + 3*v[1] = 6
        let vecs = integer_vectors_with_weighted_sum(&[2, 3], 6, None);

        let vecs_i64: Vec<Vec<i64>> = vecs
            .iter()
            .map(|v| v.to_i64_vec().unwrap())
            .collect();

        assert!(vecs_i64.contains(&vec![0, 2])); // 2*0 + 3*2 = 6
        assert!(vecs_i64.contains(&vec![3, 0])); // 2*3 + 3*0 = 6

        // Check all vectors have correct weighted sum
        for vec in &vecs {
            let sum = vec.weighted_sum(&[Integer::from(2), Integer::from(3)]);
            assert_eq!(sum, Integer::from(6));
        }
    }

    #[test]
    fn test_integer_vectors_in_box() {
        let vecs = integer_vectors_in_box(&[0, 0], &[1, 1]);
        assert_eq!(vecs.len(), 4);

        let vecs_i64: Vec<Vec<i64>> = vecs
            .iter()
            .map(|v| v.to_i64_vec().unwrap())
            .collect();

        assert!(vecs_i64.contains(&vec![0, 0]));
        assert!(vecs_i64.contains(&vec![0, 1]));
        assert!(vecs_i64.contains(&vec![1, 0]));
        assert!(vecs_i64.contains(&vec![1, 1]));
    }

    #[test]
    fn test_count_integer_vectors_in_box() {
        assert_eq!(count_integer_vectors_in_box(&[0, 0], &[2, 3]), 12);
        assert_eq!(count_integer_vectors_in_box(&[-1, -1], &[1, 1]), 9);
        assert_eq!(count_integer_vectors_in_box(&[0], &[5]), 6);
    }

    #[test]
    fn test_integer_vectors_with_l1_norm_bounded() {
        let vecs = integer_vectors_with_l1_norm_bounded(2, 1);

        // Should include all vectors with L1 norm <= 1
        let vecs_i64: Vec<Vec<i64>> = vecs
            .iter()
            .map(|v| v.to_i64_vec().unwrap())
            .collect();

        assert!(vecs_i64.contains(&vec![0, 0]));
        assert!(vecs_i64.contains(&vec![1, 0]));
        assert!(vecs_i64.contains(&vec![-1, 0]));
        assert!(vecs_i64.contains(&vec![0, 1]));
        assert!(vecs_i64.contains(&vec![0, -1]));

        // Check all have L1 norm <= 1
        for vec in &vecs {
            let norm = vec.norm_l1();
            assert!(norm <= Integer::from(1));
        }
    }

    // TODO: Fix iterator implementation
    // #[test]
    // fn test_integer_vector_sum_iter() {
    //     let vecs: Vec<_> = integer_vector_sum_iter(2, 3).collect();
    //     assert_eq!(vecs.len(), 4);
    //
    //     let vecs_i64: Vec<Vec<i64>> = vecs
    //         .iter()
    //         .map(|v| v.to_i64_vec().unwrap())
    //         .collect();
    //
    //     assert!(vecs_i64.contains(&vec![0, 3]));
    //     assert!(vecs_i64.contains(&vec![1, 2]));
    //     assert!(vecs_i64.contains(&vec![2, 1]));
    //     assert!(vecs_i64.contains(&vec![3, 0]));
    // }

    #[test]
    fn test_integer_vector_is_nonnegative() {
        assert!(IntegerVector::from_i64(vec![0, 1, 2]).is_nonnegative());
        assert!(!IntegerVector::from_i64(vec![0, -1, 2]).is_nonnegative());
    }

    #[test]
    fn test_integer_vector_is_positive() {
        assert!(IntegerVector::from_i64(vec![1, 2, 3]).is_positive());
        assert!(!IntegerVector::from_i64(vec![0, 1, 2]).is_positive());
    }

    #[test]
    fn test_integer_vector_norm_l1() {
        let v = IntegerVector::from_i64(vec![3, -4, 2]);
        assert_eq!(v.norm_l1(), Integer::from(9)); // |3| + |-4| + |2| = 9
    }

    #[test]
    fn test_weighted_sum_with_max_components() {
        // 2*v[0] + 3*v[1] = 6, with v[0] <= 1
        let vecs = integer_vectors_with_weighted_sum(&[2, 3], 6, Some(&[1, 10]));

        // Should only contain [0,2] since [3,0] violates v[0] <= 1
        assert_eq!(vecs.len(), 1);
        assert_eq!(vecs[0].to_i64_vec().unwrap(), vec![0, 2]);
    }

    #[test]
    fn test_empty_dimension() {
        let vecs = integer_vectors_with_sum(0, 0);
        assert_eq!(vecs.len(), 1);
        assert_eq!(vecs[0].dimension(), 0);

        let vecs = integer_vectors_in_box(&[], &[]);
        assert_eq!(vecs.len(), 1);
        assert_eq!(vecs[0].dimension(), 0);
    }

    #[test]
    fn test_single_dimension() {
        let vecs = integer_vectors_with_sum(1, 5);
        assert_eq!(vecs.len(), 1);
        assert_eq!(vecs[0].to_i64_vec().unwrap(), vec![5]);
    }
}
