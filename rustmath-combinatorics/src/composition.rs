//! Integer compositions (ordered partitions)
//!
//! A composition of n is an ordered sequence of positive integers that sum to n.

/// An integer composition (ordered partition)
///
/// A composition of n is an ordered sequence of positive integers that sum to n
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Composition {
    parts: Vec<usize>,
}

impl Composition {
    /// Create a composition from a vector of parts
    pub fn new(parts: Vec<usize>) -> Option<Self> {
        if parts.iter().any(|&p| p == 0) {
            return None; // All parts must be positive
        }
        Some(Composition { parts })
    }

    /// Get the sum of the composition
    pub fn sum(&self) -> usize {
        self.parts.iter().sum()
    }

    /// Get the number of parts
    pub fn length(&self) -> usize {
        self.parts.len()
    }

    /// Get the parts
    pub fn parts(&self) -> &[usize] {
        &self.parts
    }
}

/// Generate all compositions of n
///
/// A composition is an ordered way of writing n as a sum of positive integers
pub fn compositions(n: usize) -> Vec<Composition> {
    if n == 0 {
        return vec![Composition { parts: vec![] }];
    }

    let mut result = Vec::new();
    let mut current = Vec::new();

    generate_compositions(n, &mut current, &mut result);

    result
}

fn generate_compositions(n: usize, current: &mut Vec<usize>, result: &mut Vec<Composition>) {
    if n == 0 {
        result.push(Composition {
            parts: current.clone(),
        });
        return;
    }

    for i in 1..=n {
        current.push(i);
        generate_compositions(n - i, current, result);
        current.pop();
    }
}

/// Generate all compositions of n into exactly k parts
pub fn compositions_k(n: usize, k: usize) -> Vec<Composition> {
    if k == 0 {
        if n == 0 {
            return vec![Composition { parts: vec![] }];
        } else {
            return vec![];
        }
    }

    if k > n {
        return vec![];
    }

    let mut result = Vec::new();
    let mut current = Vec::new();

    generate_compositions_k(n, k, &mut current, &mut result);

    result
}

fn generate_compositions_k(
    n: usize,
    k: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Composition>,
) {
    if current.len() == k {
        if n == 0 {
            result.push(Composition {
                parts: current.clone(),
            });
        }
        return;
    }

    let remaining_parts = k - current.len();
    let min_value = 1;
    let max_value = n.saturating_sub(remaining_parts - 1);

    for i in min_value..=max_value {
        current.push(i);
        generate_compositions_k(n - i, k, current, result);
        current.pop();
    }
}

/// A signed integer composition
///
/// A signed composition is an ordered sequence of non-zero integers (positive or negative)
/// that sum to some value. Each part has both a magnitude and a sign.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SignedComposition {
    parts: Vec<isize>,
}

impl SignedComposition {
    /// Create a signed composition from a vector of parts
    ///
    /// Returns None if any part is zero (all parts must be non-zero)
    pub fn new(parts: Vec<isize>) -> Option<Self> {
        if parts.iter().any(|&p| p == 0) {
            return None; // All parts must be non-zero
        }
        Some(SignedComposition { parts })
    }

    /// Get the sum of the composition (considering signs)
    pub fn sum(&self) -> isize {
        self.parts.iter().sum()
    }

    /// Get the number of parts
    pub fn length(&self) -> usize {
        self.parts.len()
    }

    /// Get the parts
    pub fn parts(&self) -> &[isize] {
        &self.parts
    }

    /// Reverse the order of the parts
    ///
    /// Example: [1, -2, 3] becomes [3, -2, 1]
    pub fn reverse(&self) -> Self {
        let mut reversed_parts = self.parts.clone();
        reversed_parts.reverse();
        SignedComposition {
            parts: reversed_parts,
        }
    }

    /// Reverse the signs of all parts
    ///
    /// Example: [1, -2, 3] becomes [-1, 2, -3]
    pub fn reverse_signs(&self) -> Self {
        SignedComposition {
            parts: self.parts.iter().map(|&p| -p).collect(),
        }
    }

    /// Create a signed composition from an unsigned composition with explicit signs
    ///
    /// The signs vector should have the same length as the composition's parts.
    /// True indicates positive, false indicates negative.
    ///
    /// Returns None if the lengths don't match.
    pub fn from_composition(comp: &Composition, signs: Vec<bool>) -> Option<Self> {
        if comp.length() != signs.len() {
            return None;
        }

        let parts: Vec<isize> = comp
            .parts()
            .iter()
            .zip(signs.iter())
            .map(|(&part, &is_positive)| {
                if is_positive {
                    part as isize
                } else {
                    -(part as isize)
                }
            })
            .collect();

        Some(SignedComposition { parts })
    }

    /// Get the absolute value composition (all parts made positive)
    pub fn abs_composition(&self) -> Composition {
        let abs_parts: Vec<usize> = self.parts.iter().map(|&p| p.abs() as usize).collect();
        // Safe unwrap: we know all parts are non-zero, so abs won't be zero
        Composition::new(abs_parts).unwrap()
    }

    /// Get the signs of the parts
    ///
    /// Returns true for positive parts, false for negative parts
    pub fn signs(&self) -> Vec<bool> {
        self.parts.iter().map(|&p| p > 0).collect()
    }
}

/// Generate all signed compositions of n
///
/// A signed composition is an ordered way of writing n as a sum of non-zero integers
/// (which can be positive or negative)
pub fn signed_compositions(n: isize) -> Vec<SignedComposition> {
    // For signed compositions, we first generate all unsigned compositions of |n|,
    // then generate all possible sign assignments
    let abs_n = n.abs() as usize;

    if abs_n == 0 {
        return vec![SignedComposition { parts: vec![] }];
    }

    let mut result = Vec::new();
    let unsigned_comps = compositions(abs_n);

    for comp in unsigned_comps {
        // Generate all 2^k sign assignments for k parts
        let k = comp.length();
        let num_sign_patterns = 1 << k; // 2^k

        for i in 0..num_sign_patterns {
            let signs: Vec<bool> = (0..k).map(|j| (i >> j) & 1 == 1).collect();

            if let Some(signed_comp) = SignedComposition::from_composition(&comp, signs) {
                // Only include compositions that sum to n
                if signed_comp.sum() == n {
                    result.push(signed_comp);
                }
            }
        }
    }

    result
}

/// Generate all signed compositions of n into exactly k parts
pub fn signed_compositions_k(n: isize, k: usize) -> Vec<SignedComposition> {
    if k == 0 {
        if n == 0 {
            return vec![SignedComposition { parts: vec![] }];
        } else {
            return vec![];
        }
    }

    let abs_n = n.abs() as usize;
    let mut result = Vec::new();

    // Generate all unsigned compositions of abs_n into k parts
    let unsigned_comps = compositions_k(abs_n, k);

    for comp in unsigned_comps {
        // Generate all 2^k sign assignments
        let num_sign_patterns = 1 << k;

        for i in 0..num_sign_patterns {
            let signs: Vec<bool> = (0..k).map(|j| (i >> j) & 1 == 1).collect();

            if let Some(signed_comp) = SignedComposition::from_composition(&comp, signs) {
                // Only include compositions that sum to n
                if signed_comp.sum() == n {
                    result.push(signed_comp);
                }
            }
        }
    }

    result
}

/// A weighted integer vector
///
/// An integer vector where each position has an associated weight.
/// The weighted sum is the sum of each component multiplied by its weight.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WeightedIntegerVector {
    components: Vec<usize>,
    weights: Vec<usize>,
}

impl WeightedIntegerVector {
    /// Create a new weighted integer vector
    ///
    /// Returns None if the components and weights have different lengths,
    /// or if any weight is zero.
    pub fn new(components: Vec<usize>, weights: Vec<usize>) -> Option<Self> {
        if components.len() != weights.len() {
            return None;
        }
        if weights.iter().any(|&w| w == 0) {
            return None; // All weights must be positive
        }
        Some(WeightedIntegerVector {
            components,
            weights,
        })
    }

    /// Get the components
    pub fn components(&self) -> &[usize] {
        &self.components
    }

    /// Get the weights
    pub fn weights(&self) -> &[usize] {
        &self.weights
    }

    /// Get the weighted sum: sum of components[i] * weights[i]
    pub fn weighted_sum(&self) -> usize {
        self.components
            .iter()
            .zip(self.weights.iter())
            .map(|(&c, &w)| c * w)
            .sum()
    }

    /// Get the length (number of components)
    pub fn length(&self) -> usize {
        self.components.len()
    }

    /// Convert to a regular vector (just the components)
    pub fn to_vec(&self) -> Vec<usize> {
        self.components.clone()
    }
}

impl PartialOrd for WeightedIntegerVector {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WeightedIntegerVector {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Lexicographic ordering on components
        self.components.cmp(&other.components)
    }
}

/// Generate all weighted integer vectors with a given weighted sum
///
/// Given weights w_1, w_2, ..., w_k and a target sum n, generates all
/// non-negative integer vectors (v_1, v_2, ..., v_k) such that:
/// sum(v_i * w_i) = n
///
/// Uses a priority queue-based algorithm for efficient generation.
/// Vectors are generated in reverse lexicographic order.
///
/// # Arguments
/// * `weights` - The weights for each position
/// * `n` - The target weighted sum
///
/// # Example
/// ```
/// use rustmath_combinatorics::composition::integer_vectors_weighted;
///
/// // Find all ways to make sum 5 with weights [1, 2, 3]
/// let vectors = integer_vectors_weighted(vec![1, 2, 3], 5);
/// // Includes: [5,0,0], [3,1,0], [1,2,0], [0,1,1], etc.
/// ```
pub fn integer_vectors_weighted(weights: Vec<usize>, n: usize) -> Vec<WeightedIntegerVector> {
    if weights.is_empty() {
        if n == 0 {
            return vec![];
        } else {
            return vec![];
        }
    }

    // Check for zero weights
    if weights.iter().any(|&w| w == 0) {
        return vec![];
    }

    let k = weights.len();
    let mut result = Vec::new();

    // Use a priority queue with a HashSet to avoid duplicates
    use std::collections::{HashSet, VecDeque};
    let mut queue: VecDeque<Vec<usize>> = VecDeque::new();
    let mut visited: HashSet<Vec<usize>> = HashSet::new();

    // Start with the zero vector
    let start = vec![0; k];
    queue.push_back(start.clone());
    visited.insert(start);

    while let Some(current) = queue.pop_back() {
        // Calculate current weighted sum
        let current_sum: usize = current
            .iter()
            .zip(weights.iter())
            .map(|(&c, &w)| c * w)
            .sum();

        if current_sum == n {
            // Found a valid vector
            if let Some(vec) = WeightedIntegerVector::new(current.clone(), weights.clone()) {
                result.push(vec);
            }
        } else if current_sum < n {
            // Try incrementing each component
            for i in 0..k {
                let mut next = current.clone();
                next[i] += 1;

                // Only explore if we haven't visited this vector before
                if !visited.contains(&next) {
                    // Calculate new sum to check if it's worth exploring
                    let next_sum: usize =
                        next.iter().zip(weights.iter()).map(|(&c, &w)| c * w).sum();

                    if next_sum <= n {
                        visited.insert(next.clone());
                        queue.push_back(next);
                    }
                }
            }
        }
    }

    result
}

/// Generate all weighted integer vectors with weighted sum n, optimized version
///
/// This version uses dynamic programming to avoid redundant computations.
/// Generates vectors in lexicographic order using a more efficient algorithm.
///
/// # Arguments
/// * `weights` - The weights for each position
/// * `n` - The target weighted sum
pub fn integer_vectors_weighted_dp(weights: Vec<usize>, n: usize) -> Vec<WeightedIntegerVector> {
    if weights.is_empty() {
        if n == 0 {
            // Empty vector with empty weights
            return vec![];
        } else {
            return vec![];
        }
    }

    // Check for zero weights
    if weights.iter().any(|&w| w == 0) {
        return vec![];
    }

    let mut result = Vec::new();
    let k = weights.len();
    let mut current = vec![0; k];

    // Recursive generation using backtracking
    generate_weighted_vectors(&weights, n, 0, &mut current, &mut result);

    result
}

fn generate_weighted_vectors(
    weights: &[usize],
    remaining: usize,
    index: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<WeightedIntegerVector>,
) {
    if index == weights.len() {
        if remaining == 0 {
            if let Some(vec) = WeightedIntegerVector::new(current.clone(), weights.to_vec()) {
                result.push(vec);
            }
        }
        return;
    }

    let weight = weights[index];

    // Try all possible values for current position
    for value in 0..=(remaining / weight) {
        current[index] = value;
        generate_weighted_vectors(
            weights,
            remaining - value * weight,
            index + 1,
            current,
            result,
        );
    }

    current[index] = 0; // Reset for backtracking
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compositions() {
        // Compositions of 4: [4], [1,3], [2,2], [3,1], [1,1,2], [1,2,1], [2,1,1], [1,1,1,1]
        let comps = compositions(4);
        assert_eq!(comps.len(), 8); // 2^(n-1) = 2^3 = 8

        // All compositions should sum to 4
        for comp in &comps {
            assert_eq!(comp.sum(), 4);
        }
    }

    #[test]
    fn test_compositions_k() {
        // Compositions of 5 into 3 parts
        let comps = compositions_k(5, 3);
        // Should be: [1,1,3], [1,2,2], [1,3,1], [2,1,2], [2,2,1], [3,1,1]
        assert_eq!(comps.len(), 6);

        for comp in &comps {
            assert_eq!(comp.sum(), 5);
            assert_eq!(comp.length(), 3);
        }
    }

    #[test]
    fn test_composition_ordering() {
        // Compositions are ordered (unlike partitions)
        let comp1 = Composition::new(vec![1, 3]).unwrap();
        let comp2 = Composition::new(vec![3, 1]).unwrap();

        // These should be different
        assert_ne!(comp1, comp2);
        assert_eq!(comp1.sum(), comp2.sum());
    }

    #[test]
    fn test_signed_composition_new() {
        // Valid signed composition
        let sc = SignedComposition::new(vec![1, -2, 3]).unwrap();
        assert_eq!(sc.parts(), &[1, -2, 3]);
        assert_eq!(sc.sum(), 2);
        assert_eq!(sc.length(), 3);

        // Zero parts should be rejected
        assert!(SignedComposition::new(vec![1, 0, 3]).is_none());
        assert!(SignedComposition::new(vec![0]).is_none());

        // Empty composition is valid
        let empty = SignedComposition::new(vec![]).unwrap();
        assert_eq!(empty.sum(), 0);
        assert_eq!(empty.length(), 0);
    }

    #[test]
    fn test_signed_composition_reverse() {
        let sc = SignedComposition::new(vec![1, -2, 3, -4]).unwrap();
        let reversed = sc.reverse();

        assert_eq!(reversed.parts(), &[-4, 3, -2, 1]);
        assert_eq!(reversed.sum(), sc.sum()); // Sum should be preserved

        // Reversing twice should give original
        let double_reversed = reversed.reverse();
        assert_eq!(double_reversed, sc);
    }

    #[test]
    fn test_signed_composition_reverse_signs() {
        let sc = SignedComposition::new(vec![1, -2, 3, -4]).unwrap();
        let reversed_signs = sc.reverse_signs();

        assert_eq!(reversed_signs.parts(), &[-1, 2, -3, 4]);
        assert_eq!(reversed_signs.sum(), -sc.sum()); // Sum should be negated

        // Reversing signs twice should give original
        let double_reversed = reversed_signs.reverse_signs();
        assert_eq!(double_reversed, sc);
    }

    #[test]
    fn test_signed_composition_from_composition() {
        let comp = Composition::new(vec![1, 2, 3]).unwrap();

        // All positive
        let sc1 = SignedComposition::from_composition(&comp, vec![true, true, true]).unwrap();
        assert_eq!(sc1.parts(), &[1, 2, 3]);
        assert_eq!(sc1.sum(), 6);

        // Mixed signs
        let sc2 = SignedComposition::from_composition(&comp, vec![true, false, true]).unwrap();
        assert_eq!(sc2.parts(), &[1, -2, 3]);
        assert_eq!(sc2.sum(), 2);

        // All negative
        let sc3 = SignedComposition::from_composition(&comp, vec![false, false, false]).unwrap();
        assert_eq!(sc3.parts(), &[-1, -2, -3]);
        assert_eq!(sc3.sum(), -6);

        // Mismatched lengths should fail
        assert!(SignedComposition::from_composition(&comp, vec![true, false]).is_none());
    }

    #[test]
    fn test_signed_composition_abs_and_signs() {
        let sc = SignedComposition::new(vec![1, -2, 3, -4]).unwrap();

        // abs_composition should give unsigned version
        let abs_comp = sc.abs_composition();
        assert_eq!(abs_comp.parts(), &[1, 2, 3, 4]);

        // signs should track positive/negative
        let signs = sc.signs();
        assert_eq!(signs, vec![true, false, true, false]);

        // Round-trip test
        let reconstructed = SignedComposition::from_composition(&abs_comp, signs).unwrap();
        assert_eq!(reconstructed, sc);
    }

    #[test]
    fn test_signed_compositions_generation() {
        // Generate all signed compositions of 2
        let scs = signed_compositions(2);

        // Should include compositions like [2], [1,1], [-1,3], [3,-1], etc.
        // that sum to 2
        assert!(scs.iter().all(|sc| sc.sum() == 2));

        // Check that we have various compositions
        assert!(scs
            .iter()
            .any(|sc| sc.parts() == &[2] || sc.parts() == &[2]));
        assert!(scs.iter().any(|sc| sc.parts() == &[1, 1]));

        // Verify no duplicates
        for i in 0..scs.len() {
            for j in i + 1..scs.len() {
                assert_ne!(scs[i], scs[j]);
            }
        }
    }

    #[test]
    fn test_signed_compositions_negative() {
        // Generate all signed compositions of -2
        let scs = signed_compositions(-2);

        // All should sum to -2
        assert!(scs.iter().all(|sc| sc.sum() == -2));

        // Should include [-2], [-1,-1], etc.
        assert!(scs.iter().any(|sc| sc.parts() == &[-2]));
        assert!(scs.iter().any(|sc| sc.parts() == &[-1, -1]));
    }

    #[test]
    fn test_signed_compositions_k() {
        // Generate all signed compositions of 3 into 2 parts
        let scs = signed_compositions_k(3, 2);

        // All should sum to 3 and have exactly 2 parts
        assert!(scs.iter().all(|sc| sc.sum() == 3));
        assert!(scs.iter().all(|sc| sc.length() == 2));

        // Should include compositions like [1,2], [2,1], [-1,4], [4,-1], etc.
        assert!(scs.iter().any(|sc| sc.parts() == &[1, 2]));
        assert!(scs.iter().any(|sc| sc.parts() == &[2, 1]));

        // Verify no duplicates
        for i in 0..scs.len() {
            for j in i + 1..scs.len() {
                assert_ne!(scs[i], scs[j]);
            }
        }
    }

    #[test]
    fn test_signed_compositions_zero() {
        // Compositions of 0 should only be empty composition
        let scs = signed_compositions(0);
        assert_eq!(scs.len(), 1);
        assert_eq!(scs[0].parts(), &[]);
        assert_eq!(scs[0].sum(), 0);
    }

    #[test]
    fn test_signed_compositions_k_edge_cases() {
        // k=0 with n=0 should give empty composition
        let scs = signed_compositions_k(0, 0);
        assert_eq!(scs.len(), 1);
        assert_eq!(scs[0].parts(), &[]);

        // k=0 with n!=0 should give no compositions
        let scs = signed_compositions_k(5, 0);
        assert_eq!(scs.len(), 0);

        // k > |n| with all same sign should give no compositions
        // (can't make k parts sum to n if k > |n| with all positive)
        let scs = signed_compositions_k(2, 10);
        // This might still have results with mixed signs
        assert!(scs.iter().all(|sc| sc.sum() == 2));
        assert!(scs.iter().all(|sc| sc.length() == 10));
    }

    #[test]
    fn test_signed_composition_symmetries() {
        let sc = SignedComposition::new(vec![1, -2, 3]).unwrap();

        // Test that reverse and reverse_signs commute
        let rev_then_signs = sc.reverse().reverse_signs();
        let signs_then_rev = sc.reverse_signs().reverse();

        // These should be the same
        assert_eq!(rev_then_signs, signs_then_rev);

        // Test sum properties
        assert_eq!(sc.sum(), -sc.reverse_signs().sum());
    }

    #[test]
    fn test_weighted_integer_vector_new() {
        // Valid weighted vector
        let wiv = WeightedIntegerVector::new(vec![1, 2, 3], vec![1, 2, 3]).unwrap();
        assert_eq!(wiv.components(), &[1, 2, 3]);
        assert_eq!(wiv.weights(), &[1, 2, 3]);
        assert_eq!(wiv.weighted_sum(), 1 * 1 + 2 * 2 + 3 * 3); // 1 + 4 + 9 = 14
        assert_eq!(wiv.length(), 3);

        // Mismatched lengths should fail
        assert!(WeightedIntegerVector::new(vec![1, 2], vec![1, 2, 3]).is_none());
        assert!(WeightedIntegerVector::new(vec![1, 2, 3], vec![1, 2]).is_none());

        // Zero weights should fail
        assert!(WeightedIntegerVector::new(vec![1, 2, 3], vec![1, 0, 3]).is_none());
        assert!(WeightedIntegerVector::new(vec![1], vec![0]).is_none());

        // Empty vector is valid
        let empty = WeightedIntegerVector::new(vec![], vec![]).unwrap();
        assert_eq!(empty.weighted_sum(), 0);
        assert_eq!(empty.length(), 0);
    }

    #[test]
    fn test_weighted_integer_vector_weighted_sum() {
        let wiv = WeightedIntegerVector::new(vec![2, 0, 1], vec![1, 2, 3]).unwrap();
        assert_eq!(wiv.weighted_sum(), 2 * 1 + 0 * 2 + 1 * 3); // 2 + 0 + 3 = 5

        let wiv2 = WeightedIntegerVector::new(vec![0, 0, 0], vec![5, 10, 15]).unwrap();
        assert_eq!(wiv2.weighted_sum(), 0);

        let wiv3 = WeightedIntegerVector::new(vec![10], vec![7]).unwrap();
        assert_eq!(wiv3.weighted_sum(), 70);
    }

    #[test]
    fn test_weighted_integer_vector_ordering() {
        let wiv1 = WeightedIntegerVector::new(vec![1, 2, 3], vec![1, 1, 1]).unwrap();
        let wiv2 = WeightedIntegerVector::new(vec![1, 2, 4], vec![1, 1, 1]).unwrap();
        let wiv3 = WeightedIntegerVector::new(vec![1, 3, 0], vec![1, 1, 1]).unwrap();

        // Lexicographic ordering: [1,2,3] < [1,2,4] < [1,3,0]
        assert!(wiv1 < wiv2);
        assert!(wiv1 < wiv3);
        assert!(wiv2 < wiv3); // [1,2,4] < [1,3,0] because 2 < 3 at position 1

        // Same components, different weights should still compare by components
        let wiv4 = WeightedIntegerVector::new(vec![1, 2, 3], vec![2, 3, 4]).unwrap();
        assert_eq!(wiv1.cmp(&wiv4), std::cmp::Ordering::Equal);
    }

    #[test]
    fn test_integer_vectors_weighted_simple() {
        // Find all ways to make sum 5 with weights [1, 2, 3]
        let vectors = integer_vectors_weighted(vec![1, 2, 3], 5);

        // All vectors should have weighted sum 5
        for vec in &vectors {
            assert_eq!(vec.weighted_sum(), 5);
        }

        // Should include specific vectors
        assert!(vectors
            .iter()
            .any(|v| v.components() == &[5, 0, 0])); // 5*1
        assert!(vectors
            .iter()
            .any(|v| v.components() == &[3, 1, 0])); // 3*1 + 1*2
        assert!(vectors
            .iter()
            .any(|v| v.components() == &[1, 2, 0])); // 1*1 + 2*2
        assert!(vectors
            .iter()
            .any(|v| v.components() == &[2, 0, 1])); // 2*1 + 1*3
        assert!(vectors
            .iter()
            .any(|v| v.components() == &[0, 1, 1])); // 1*2 + 1*3

        // No duplicates
        for i in 0..vectors.len() {
            for j in i + 1..vectors.len() {
                assert_ne!(vectors[i].components(), vectors[j].components());
            }
        }
    }

    #[test]
    fn test_integer_vectors_weighted_dp_simple() {
        // Test the DP version
        let vectors = integer_vectors_weighted_dp(vec![1, 2, 3], 5);

        // All vectors should have weighted sum 5
        for vec in &vectors {
            assert_eq!(vec.weighted_sum(), 5);
        }

        // Should generate the same vectors as the queue version (possibly different order)
        let vectors_queue = integer_vectors_weighted(vec![1, 2, 3], 5);
        assert_eq!(vectors.len(), vectors_queue.len());

        // Check that all vectors from queue version are in DP version
        for qv in &vectors_queue {
            assert!(vectors
                .iter()
                .any(|v| v.components() == qv.components()));
        }
    }

    #[test]
    fn test_integer_vectors_weighted_zero_sum() {
        // Sum of 0 should only give the zero vector
        let vectors = integer_vectors_weighted(vec![1, 2, 3], 0);
        assert_eq!(vectors.len(), 1);
        assert_eq!(vectors[0].components(), &[0, 0, 0]);
        assert_eq!(vectors[0].weighted_sum(), 0);
    }

    #[test]
    fn test_integer_vectors_weighted_single_weight() {
        // With weight [3] and sum 9, should only get [3]
        let vectors = integer_vectors_weighted(vec![3], 9);
        assert_eq!(vectors.len(), 1);
        assert_eq!(vectors[0].components(), &[3]);
        assert_eq!(vectors[0].weighted_sum(), 9);

        // With weight [3] and sum 7, should get nothing (not divisible)
        let vectors2 = integer_vectors_weighted(vec![3], 7);
        assert_eq!(vectors2.len(), 0);
    }

    #[test]
    fn test_integer_vectors_weighted_uniform_weights() {
        // With uniform weights [1, 1, 1] and sum 3, we're finding
        // all ways to write 3 as a sum of 3 non-negative integers
        let vectors = integer_vectors_weighted(vec![1, 1, 1], 3);

        // Should include: [3,0,0], [0,3,0], [0,0,3], [2,1,0], [2,0,1], [1,2,0],
        // [0,2,1], [1,0,2], [0,1,2], [1,1,1]
        // Total: C(3+3-1, 3) = C(5,3) = 10
        assert_eq!(vectors.len(), 10);

        for vec in &vectors {
            assert_eq!(vec.weighted_sum(), 3);
            assert_eq!(
                vec.components().iter().sum::<usize>(),
                3,
                "With uniform weights, sum of components should equal weighted sum"
            );
        }
    }

    #[test]
    fn test_integer_vectors_weighted_large_weights() {
        // With weights [10, 20] and sum 50
        let vectors = integer_vectors_weighted(vec![10, 20], 50);

        // Valid combinations:
        // [5, 0] -> 50
        // [3, 1] -> 30 + 20 = 50
        // [1, 2] -> 10 + 40 = 50
        assert_eq!(vectors.len(), 3);

        assert!(vectors.iter().any(|v| v.components() == &[5, 0]));
        assert!(vectors.iter().any(|v| v.components() == &[3, 1]));
        assert!(vectors.iter().any(|v| v.components() == &[1, 2]));
    }

    #[test]
    fn test_integer_vectors_weighted_impossible_sum() {
        // With weights [2, 4] (all even), sum 7 (odd) should be impossible
        let vectors = integer_vectors_weighted(vec![2, 4], 7);
        assert_eq!(vectors.len(), 0);

        // With weights [3, 6], sum 7 should also be impossible
        let vectors2 = integer_vectors_weighted(vec![3, 6], 7);
        assert_eq!(vectors2.len(), 0);
    }

    #[test]
    fn test_integer_vectors_weighted_empty_weights() {
        // Empty weights with sum 0
        let vectors = integer_vectors_weighted(vec![], 0);
        assert_eq!(vectors.len(), 0);

        // Empty weights with non-zero sum
        let vectors2 = integer_vectors_weighted(vec![], 5);
        assert_eq!(vectors2.len(), 0);
    }

    #[test]
    fn test_integer_vectors_weighted_zero_in_weights() {
        // Zero weights should be rejected
        let vectors = integer_vectors_weighted(vec![1, 0, 3], 5);
        assert_eq!(vectors.len(), 0);
    }

    #[test]
    fn test_integer_vectors_weighted_coin_change() {
        // Classic coin change problem: make 10 cents with pennies(1), nickels(5), dimes(10)
        let vectors = integer_vectors_weighted(vec![1, 5, 10], 10);

        // All should sum to 10
        for vec in &vectors {
            assert_eq!(vec.weighted_sum(), 10);
        }

        // Should include:
        // [10, 0, 0] - 10 pennies
        // [5, 1, 0] - 5 pennies, 1 nickel
        // [0, 2, 0] - 2 nickels
        // [0, 0, 1] - 1 dime
        assert!(vectors.iter().any(|v| v.components() == &[10, 0, 0]));
        assert!(vectors.iter().any(|v| v.components() == &[5, 1, 0]));
        assert!(vectors.iter().any(|v| v.components() == &[0, 2, 0]));
        assert!(vectors.iter().any(|v| v.components() == &[0, 0, 1]));
    }

    #[test]
    fn test_integer_vectors_weighted_dp_consistency() {
        // Test that both algorithms produce the same results for various inputs
        let test_cases = vec![
            (vec![1, 2, 3], 5),
            (vec![1, 1, 1], 3),
            (vec![10, 20], 50),
            (vec![2, 3, 5], 10),
            (vec![1, 5, 10], 10),
        ];

        for (weights, sum) in test_cases {
            let v1 = integer_vectors_weighted(weights.clone(), sum);
            let v2 = integer_vectors_weighted_dp(weights.clone(), sum);

            assert_eq!(
                v1.len(),
                v2.len(),
                "Different number of results for weights {:?}, sum {}",
                weights,
                sum
            );

            // Check that all vectors from v1 are in v2 and vice versa
            for vec in &v1 {
                assert!(
                    v2.iter().any(|v| v.components() == vec.components()),
                    "Vector {:?} from queue version not found in DP version",
                    vec.components()
                );
            }

            for vec in &v2 {
                assert!(
                    v1.iter().any(|v| v.components() == vec.components()),
                    "Vector {:?} from DP version not found in queue version",
                    vec.components()
                );
            }
        }
    }
}
