//! Subsets and efficient subset generation algorithms
//!
//! This module provides comprehensive support for working with subsets, including:
//! - Efficient generation of all subsets or k-subsets
//! - Rank/unrank operations using binary representation
//! - Lexicographic successor/predecessor algorithms
//! - Multiple representation formats (element list, bit vector)

use rustmath_integers::Integer;

/// A subset of {0, 1, 2, ..., n-1}
///
/// Subsets can be represented either as:
/// - A sorted vector of elements (space-efficient for small subsets)
/// - A bit vector (efficient for set operations and dense subsets)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Subset {
    /// The selected elements (in sorted order)
    elements: Vec<usize>,
    /// Total size of the universe set
    n: usize,
}

impl Subset {
    /// Create a subset from a vector of elements
    ///
    /// Returns None if elements are invalid or out of range
    ///
    /// # Example
    /// ```
    /// use rustmath_combinatorics::Subset;
    ///
    /// let subset = Subset::from_elements(vec![0, 2, 3], 5).unwrap();
    /// assert_eq!(subset.elements(), &[0, 2, 3]);
    /// assert_eq!(subset.size(), 3);
    /// ```
    pub fn from_elements(elements: Vec<usize>, n: usize) -> Option<Self> {
        // Validate and sort elements
        let mut sorted_elements = elements.clone();
        sorted_elements.sort_unstable();
        sorted_elements.dedup();

        if sorted_elements.len() != elements.len() {
            return None; // Duplicate elements
        }

        for &elem in &sorted_elements {
            if elem >= n {
                return None; // Element out of range
            }
        }

        Some(Subset {
            elements: sorted_elements,
            n,
        })
    }

    /// Create a subset from a bit vector
    ///
    /// The i-th bit indicates whether element i is in the subset
    ///
    /// # Example
    /// ```
    /// use rustmath_combinatorics::Subset;
    ///
    /// // Binary 1101 = elements {0, 2, 3}
    /// let subset = Subset::from_bitvector(0b1101, 4);
    /// assert_eq!(subset.elements(), &[0, 2, 3]);
    /// ```
    pub fn from_bitvector(bits: usize, n: usize) -> Self {
        let mut elements = Vec::new();
        for i in 0..n {
            if (bits >> i) & 1 == 1 {
                elements.push(i);
            }
        }
        Subset { elements, n }
    }

    /// Create an empty subset
    pub fn empty(n: usize) -> Self {
        Subset {
            elements: Vec::new(),
            n,
        }
    }

    /// Create the full subset containing all elements
    pub fn full(n: usize) -> Self {
        Subset {
            elements: (0..n).collect(),
            n,
        }
    }

    /// Get the elements of this subset
    pub fn elements(&self) -> &[usize] {
        &self.elements
    }

    /// Get the size of this subset
    pub fn size(&self) -> usize {
        self.elements.len()
    }

    /// Get the size of the universe set
    pub fn universe_size(&self) -> usize {
        self.n
    }

    /// Convert to bit vector representation
    ///
    /// Returns None if n is too large (> 64 on most platforms)
    pub fn to_bitvector(&self) -> Option<usize> {
        if self.n > usize::BITS as usize {
            return None;
        }

        let mut bits = 0usize;
        for &elem in &self.elements {
            bits |= 1 << elem;
        }
        Some(bits)
    }

    /// Check if an element is in this subset
    pub fn contains(&self, elem: usize) -> bool {
        self.elements.binary_search(&elem).is_ok()
    }

    /// Compute the rank of this subset among all 2^n subsets
    ///
    /// Uses binary representation: subset {0, 2} of {0,1,2,3} has rank 0101₂ = 5
    ///
    /// # Example
    /// ```
    /// use rustmath_combinatorics::Subset;
    ///
    /// let subset = Subset::from_elements(vec![0, 2], 4).unwrap();
    /// assert_eq!(subset.rank_binary(), 5); // Binary 0101
    /// ```
    pub fn rank_binary(&self) -> usize {
        self.to_bitvector().unwrap_or(0)
    }

    /// Create a subset from its binary rank
    ///
    /// # Example
    /// ```
    /// use rustmath_combinatorics::Subset;
    ///
    /// let subset = Subset::unrank_binary(5, 4); // Binary 0101
    /// assert_eq!(subset.elements(), &[0, 2]);
    /// ```
    pub fn unrank_binary(rank: usize, n: usize) -> Self {
        Self::from_bitvector(rank, n)
    }

    /// Compute the rank of this k-subset among all C(n, k) k-subsets
    ///
    /// Uses the combinatorial number system (same as Combination)
    /// Returns None if this is not a valid k-subset
    pub fn rank_k_subset(&self, k: usize) -> Option<usize> {
        if self.size() != k {
            return None;
        }

        let mut rank = 0;

        for (i, &elem) in self.elements.iter().enumerate() {
            // Count k-subsets that come before this one
            let remaining = k - i;
            let start = if i == 0 { 0 } else { self.elements[i - 1] + 1 };

            for val in start..elem {
                let choices = self.n - val - 1;
                if choices >= remaining - 1 {
                    rank += binomial_usize(choices, remaining - 1);
                }
            }
        }

        Some(rank)
    }

    /// Create a k-subset from its rank among all C(n, k) k-subsets
    ///
    /// Returns None if rank is invalid
    pub fn unrank_k_subset(rank: usize, n: usize, k: usize) -> Option<Self> {
        if k > n {
            return None;
        }

        if k == 0 {
            return Some(Subset::empty(n));
        }

        let mut elements = Vec::new();
        let mut remaining_rank = rank;
        let mut start = 0;

        for i in 0..k {
            let remaining = k - i;

            // Find the smallest value v such that C(n-v-1, remaining-1) <= remaining_rank
            for val in start..n {
                let choices = n - val - 1;
                if choices < remaining - 1 {
                    continue;
                }

                let count = binomial_usize(choices, remaining - 1);

                if remaining_rank < count {
                    elements.push(val);
                    start = val + 1;
                    break;
                } else {
                    remaining_rank -= count;
                }
            }
        }

        if elements.len() != k {
            return None;
        }

        Some(Subset { elements, n })
    }

    /// Get the next subset in lexicographic order (among k-subsets)
    ///
    /// Returns None if this is the last k-subset
    ///
    /// # Example
    /// ```
    /// use rustmath_combinatorics::Subset;
    ///
    /// let subset = Subset::from_elements(vec![0, 1, 2], 5).unwrap();
    /// let next = subset.next_k_subset().unwrap();
    /// assert_eq!(next.elements(), &[0, 1, 3]);
    /// ```
    pub fn next_k_subset(&self) -> Option<Self> {
        let k = self.size();
        if k == 0 || self.n == 0 {
            return None;
        }

        let mut new_elements = self.elements.clone();

        // Find the rightmost element that can be incremented
        for i in (0..k).rev() {
            if new_elements[i] < self.n - k + i {
                new_elements[i] += 1;
                // Reset all elements to the right
                for j in i + 1..k {
                    new_elements[j] = new_elements[j - 1] + 1;
                }
                return Some(Subset {
                    elements: new_elements,
                    n: self.n,
                });
            }
        }

        None // This was the last k-subset
    }

    /// Get the previous subset in lexicographic order (among k-subsets)
    ///
    /// Returns None if this is the first k-subset
    pub fn prev_k_subset(&self) -> Option<Self> {
        let k = self.size();
        if k == 0 || self.n == 0 {
            return None;
        }

        let mut new_elements = self.elements.clone();

        // Find the rightmost element that can be decremented
        for i in (0..k).rev() {
            let min_val = if i == 0 { 0 } else { new_elements[i - 1] + 1 };
            if new_elements[i] > min_val {
                new_elements[i] -= 1;
                // Set all elements to the right to their maximum values
                for j in i + 1..k {
                    new_elements[j] = self.n - k + j;
                }
                return Some(Subset {
                    elements: new_elements,
                    n: self.n,
                });
            }
        }

        None // This was the first k-subset
    }

    /// Get the next subset in binary order (among all 2^n subsets)
    ///
    /// Returns None if this is the full set
    pub fn next_binary(&self) -> Option<Self> {
        let bits = self.to_bitvector()?;
        if bits == (1 << self.n) - 1 {
            return None; // This is the full set
        }
        Some(Self::from_bitvector(bits + 1, self.n))
    }

    /// Get the previous subset in binary order
    ///
    /// Returns None if this is the empty set
    pub fn prev_binary(&self) -> Option<Self> {
        let bits = self.to_bitvector()?;
        if bits == 0 {
            return None; // This is the empty set
        }
        Some(Self::from_bitvector(bits - 1, self.n))
    }

    /// Compute the complement of this subset
    pub fn complement(&self) -> Self {
        let mut elements = Vec::new();
        let mut elem_idx = 0;

        for i in 0..self.n {
            if elem_idx < self.elements.len() && self.elements[elem_idx] == i {
                elem_idx += 1;
            } else {
                elements.push(i);
            }
        }

        Subset {
            elements,
            n: self.n,
        }
    }

    /// Compute the union of two subsets
    pub fn union(&self, other: &Subset) -> Option<Self> {
        if self.n != other.n {
            return None;
        }

        let mut elements = Vec::new();
        let mut i = 0;
        let mut j = 0;

        while i < self.elements.len() && j < other.elements.len() {
            if self.elements[i] < other.elements[j] {
                elements.push(self.elements[i]);
                i += 1;
            } else if self.elements[i] > other.elements[j] {
                elements.push(other.elements[j]);
                j += 1;
            } else {
                elements.push(self.elements[i]);
                i += 1;
                j += 1;
            }
        }

        while i < self.elements.len() {
            elements.push(self.elements[i]);
            i += 1;
        }

        while j < other.elements.len() {
            elements.push(other.elements[j]);
            j += 1;
        }

        Some(Subset {
            elements,
            n: self.n,
        })
    }

    /// Compute the intersection of two subsets
    pub fn intersection(&self, other: &Subset) -> Option<Self> {
        if self.n != other.n {
            return None;
        }

        let mut elements = Vec::new();
        let mut i = 0;
        let mut j = 0;

        while i < self.elements.len() && j < other.elements.len() {
            if self.elements[i] < other.elements[j] {
                i += 1;
            } else if self.elements[i] > other.elements[j] {
                j += 1;
            } else {
                elements.push(self.elements[i]);
                i += 1;
                j += 1;
            }
        }

        Some(Subset {
            elements,
            n: self.n,
        })
    }

    /// Compute the difference of two subsets (self \ other)
    pub fn difference(&self, other: &Subset) -> Option<Self> {
        if self.n != other.n {
            return None;
        }

        let mut elements = Vec::new();
        let mut i = 0;
        let mut j = 0;

        while i < self.elements.len() {
            if j >= other.elements.len() || self.elements[i] < other.elements[j] {
                elements.push(self.elements[i]);
                i += 1;
            } else if self.elements[i] > other.elements[j] {
                j += 1;
            } else {
                i += 1;
                j += 1;
            }
        }

        Some(Subset {
            elements,
            n: self.n,
        })
    }
}

/// Generate all subsets of {0, 1, ..., n-1}
///
/// Returns 2^n subsets in binary order (empty set first, full set last)
///
/// # Example
/// ```
/// use rustmath_combinatorics::all_subsets;
///
/// let subsets = all_subsets(3);
/// assert_eq!(subsets.len(), 8);
/// assert_eq!(subsets[0].elements(), &[]); // Empty set
/// assert_eq!(subsets[7].elements(), &[0, 1, 2]); // Full set
/// ```
pub fn all_subsets(n: usize) -> Vec<Subset> {
    if n > 20 {
        // Avoid generating huge numbers of subsets
        panic!("n too large for all_subsets (n = {}). Use subset iterator instead.", n);
    }

    let count = 1 << n;
    let mut result = Vec::with_capacity(count);

    for bits in 0..count {
        result.push(Subset::from_bitvector(bits, n));
    }

    result
}

/// Generate all k-subsets of {0, 1, ..., n-1}
///
/// Returns C(n, k) subsets in lexicographic order
///
/// # Example
/// ```
/// use rustmath_combinatorics::k_subsets;
///
/// let subsets = k_subsets(4, 2);
/// assert_eq!(subsets.len(), 6);
/// assert_eq!(subsets[0].elements(), &[0, 1]);
/// assert_eq!(subsets[5].elements(), &[2, 3]);
/// ```
pub fn k_subsets(n: usize, k: usize) -> Vec<Subset> {
    if k > n {
        return vec![];
    }

    if k == 0 {
        return vec![Subset::empty(n)];
    }

    let mut result = Vec::new();
    let mut current = Vec::new();

    generate_k_subsets(n, k, 0, &mut current, &mut result);

    result
}

fn generate_k_subsets(
    n: usize,
    k: usize,
    start: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Subset>,
) {
    if current.len() == k {
        result.push(Subset {
            elements: current.clone(),
            n,
        });
        return;
    }

    let remaining = k - current.len();
    for i in start..=(n - remaining) {
        current.push(i);
        generate_k_subsets(n, k, i + 1, current, result);
        current.pop();
    }
}

/// Iterator over all k-subsets in lexicographic order
///
/// More memory-efficient than generating all subsets at once
pub struct KSubsetIterator {
    n: usize,
    k: usize,
    current: Option<Subset>,
}

impl KSubsetIterator {
    /// Create a new k-subset iterator
    pub fn new(n: usize, k: usize) -> Self {
        let current = if k <= n {
            Some(Subset {
                elements: (0..k).collect(),
                n,
            })
        } else {
            None
        };

        KSubsetIterator { n, k, current }
    }
}

impl Iterator for KSubsetIterator {
    type Item = Subset;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.current.clone()?;
        self.current = result.next_k_subset();
        Some(result)
    }
}

/// Create an iterator over all k-subsets
///
/// # Example
/// ```
/// use rustmath_combinatorics::k_subset_iterator;
///
/// let mut iter = k_subset_iterator(5, 3);
/// assert_eq!(iter.next().unwrap().elements(), &[0, 1, 2]);
/// assert_eq!(iter.next().unwrap().elements(), &[0, 1, 3]);
/// ```
pub fn k_subset_iterator(n: usize, k: usize) -> KSubsetIterator {
    KSubsetIterator::new(n, k)
}

/// Iterator over all subsets in binary order
pub struct SubsetIterator {
    n: usize,
    current: Option<usize>,
    max: usize,
}

impl SubsetIterator {
    /// Create a new subset iterator for all 2^n subsets
    pub fn new(n: usize) -> Self {
        if n > 30 {
            panic!("n too large for subset iterator (n = {})", n);
        }

        SubsetIterator {
            n,
            current: Some(0),
            max: 1 << n,
        }
    }
}

impl Iterator for SubsetIterator {
    type Item = Subset;

    fn next(&mut self) -> Option<Self::Item> {
        let bits = self.current?;
        if bits >= self.max {
            return None;
        }

        self.current = Some(bits + 1);
        Some(Subset::from_bitvector(bits, self.n))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.max - self.current.unwrap_or(self.max);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for SubsetIterator {}

/// Create an iterator over all 2^n subsets in binary order
///
/// # Example
/// ```
/// use rustmath_combinatorics::subset_iterator;
///
/// let mut iter = subset_iterator(3);
/// assert_eq!(iter.next().unwrap().elements(), &[]); // Empty set
/// assert_eq!(iter.next().unwrap().elements(), &[0]); // {0}
/// ```
pub fn subset_iterator(n: usize) -> SubsetIterator {
    SubsetIterator::new(n)
}

/// Compute binomial coefficient using usize (for internal use)
fn binomial_usize(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }

    let k = k.min(n - k);
    let mut result = 1usize;

    for i in 0..k {
        result = result.saturating_mul(n - i) / (i + 1);
    }

    result
}

/// Count the number of k-subsets of an n-element set
///
/// Returns C(n, k) = n! / (k! * (n-k)!)
pub fn count_k_subsets(n: usize, k: usize) -> Integer {
    crate::binomial(n as u32, k as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subset_creation() {
        let subset = Subset::from_elements(vec![0, 2, 3], 5).unwrap();
        assert_eq!(subset.elements(), &[0, 2, 3]);
        assert_eq!(subset.size(), 3);
        assert_eq!(subset.universe_size(), 5);

        // Test invalid subsets
        assert!(Subset::from_elements(vec![0, 5], 5).is_none()); // Out of range
        assert!(Subset::from_elements(vec![0, 0, 1], 5).is_none()); // Duplicates
    }

    #[test]
    fn test_bitvector_conversion() {
        let subset = Subset::from_elements(vec![0, 2, 3], 5).unwrap();
        assert_eq!(subset.to_bitvector(), Some(0b01101)); // Binary: 01101

        let subset2 = Subset::from_bitvector(0b01101, 5);
        assert_eq!(subset2.elements(), &[0, 2, 3]);
    }

    #[test]
    fn test_binary_rank_unrank() {
        // Test {0, 2} in universe {0,1,2,3}
        let subset = Subset::from_elements(vec![0, 2], 4).unwrap();
        assert_eq!(subset.rank_binary(), 5); // Binary 0101

        let unranked = Subset::unrank_binary(5, 4);
        assert_eq!(unranked.elements(), &[0, 2]);

        // Test roundtrip for all subsets of size 3
        for bits in 0..8 {
            let subset = Subset::unrank_binary(bits, 3);
            assert_eq!(subset.rank_binary(), bits);
        }
    }

    #[test]
    fn test_k_subset_rank_unrank() {
        // Test C(5, 3) ranking
        let subset = Subset::from_elements(vec![0, 1, 2], 5).unwrap();
        assert_eq!(subset.rank_k_subset(3), Some(0)); // First 3-subset

        let subset_last = Subset::from_elements(vec![2, 3, 4], 5).unwrap();
        assert_eq!(subset_last.rank_k_subset(3), Some(9)); // Last 3-subset (C(5,3) = 10)

        // Test unranking
        let unranked = Subset::unrank_k_subset(0, 5, 3).unwrap();
        assert_eq!(unranked.elements(), &[0, 1, 2]);

        let unranked_last = Subset::unrank_k_subset(9, 5, 3).unwrap();
        assert_eq!(unranked_last.elements(), &[2, 3, 4]);

        // Test roundtrip for all C(6, 3) = 20 subsets
        for rank in 0..20 {
            let subset = Subset::unrank_k_subset(rank, 6, 3).unwrap();
            assert_eq!(subset.rank_k_subset(3), Some(rank));
        }
    }

    #[test]
    fn test_next_k_subset() {
        let subset = Subset::from_elements(vec![0, 1, 2], 5).unwrap();
        let next = subset.next_k_subset().unwrap();
        assert_eq!(next.elements(), &[0, 1, 3]);

        let next2 = next.next_k_subset().unwrap();
        assert_eq!(next2.elements(), &[0, 1, 4]);

        let next3 = next2.next_k_subset().unwrap();
        assert_eq!(next3.elements(), &[0, 2, 3]);

        // Test last subset
        let last = Subset::from_elements(vec![2, 3, 4], 5).unwrap();
        assert!(last.next_k_subset().is_none());
    }

    #[test]
    fn test_prev_k_subset() {
        let subset = Subset::from_elements(vec![0, 1, 3], 5).unwrap();
        let prev = subset.prev_k_subset().unwrap();
        assert_eq!(prev.elements(), &[0, 1, 2]);

        // Test first subset
        let first = Subset::from_elements(vec![0, 1, 2], 5).unwrap();
        assert!(first.prev_k_subset().is_none());

        // Test middle subset
        let middle = Subset::from_elements(vec![1, 2, 4], 5).unwrap();
        let prev = middle.prev_k_subset().unwrap();
        assert_eq!(prev.elements(), &[1, 2, 3]);
    }

    #[test]
    fn test_next_prev_binary() {
        let subset = Subset::from_elements(vec![0], 3).unwrap();
        let next = subset.next_binary().unwrap();
        assert_eq!(next.elements(), &[1]);

        let subset2 = Subset::from_elements(vec![0, 1], 3).unwrap();
        let prev = subset2.prev_binary().unwrap();
        assert_eq!(prev.elements(), &[1]);
    }

    #[test]
    fn test_set_operations() {
        let s1 = Subset::from_elements(vec![0, 1, 3], 5).unwrap();
        let s2 = Subset::from_elements(vec![1, 2, 3], 5).unwrap();

        // Union
        let union = s1.union(&s2).unwrap();
        assert_eq!(union.elements(), &[0, 1, 2, 3]);

        // Intersection
        let intersection = s1.intersection(&s2).unwrap();
        assert_eq!(intersection.elements(), &[1, 3]);

        // Difference
        let diff = s1.difference(&s2).unwrap();
        assert_eq!(diff.elements(), &[0]);

        // Complement
        let comp = s1.complement();
        assert_eq!(comp.elements(), &[2, 4]);
    }

    #[test]
    fn test_all_subsets() {
        let subsets = all_subsets(3);
        assert_eq!(subsets.len(), 8);

        assert_eq!(subsets[0].elements(), &[]); // Empty
        assert_eq!(subsets[1].elements(), &[0]);
        assert_eq!(subsets[2].elements(), &[1]);
        assert_eq!(subsets[3].elements(), &[0, 1]);
        assert_eq!(subsets[4].elements(), &[2]);
        assert_eq!(subsets[5].elements(), &[0, 2]);
        assert_eq!(subsets[6].elements(), &[1, 2]);
        assert_eq!(subsets[7].elements(), &[0, 1, 2]); // Full
    }

    #[test]
    fn test_k_subsets() {
        let subsets = k_subsets(4, 2);
        assert_eq!(subsets.len(), 6); // C(4, 2) = 6

        assert_eq!(subsets[0].elements(), &[0, 1]);
        assert_eq!(subsets[1].elements(), &[0, 2]);
        assert_eq!(subsets[2].elements(), &[0, 3]);
        assert_eq!(subsets[3].elements(), &[1, 2]);
        assert_eq!(subsets[4].elements(), &[1, 3]);
        assert_eq!(subsets[5].elements(), &[2, 3]);
    }

    #[test]
    fn test_k_subset_iterator() {
        let mut iter = k_subset_iterator(5, 3);

        let first = iter.next().unwrap();
        assert_eq!(first.elements(), &[0, 1, 2]);

        let second = iter.next().unwrap();
        assert_eq!(second.elements(), &[0, 1, 3]);

        // Count all subsets
        let count = 2 + iter.count();
        assert_eq!(count, 10); // C(5, 3) = 10
    }

    #[test]
    fn test_subset_iterator() {
        let mut iter = subset_iterator(3);

        assert_eq!(iter.next().unwrap().elements(), &[]);
        assert_eq!(iter.next().unwrap().elements(), &[0]);
        assert_eq!(iter.next().unwrap().elements(), &[1]);
        assert_eq!(iter.next().unwrap().elements(), &[0, 1]);

        // Count remaining
        let count = 4 + iter.count();
        assert_eq!(count, 8); // 2^3 = 8
    }

    #[test]
    fn test_contains() {
        let subset = Subset::from_elements(vec![0, 2, 4], 6).unwrap();
        assert!(subset.contains(0));
        assert!(!subset.contains(1));
        assert!(subset.contains(2));
        assert!(!subset.contains(3));
        assert!(subset.contains(4));
        assert!(!subset.contains(5));
    }

    #[test]
    fn test_empty_and_full() {
        let empty = Subset::empty(5);
        assert_eq!(empty.size(), 0);
        assert_eq!(empty.elements(), &[]);

        let full = Subset::full(5);
        assert_eq!(full.size(), 5);
        assert_eq!(full.elements(), &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_edge_cases() {
        // Empty universe
        let empty_universe = Subset::empty(0);
        assert_eq!(empty_universe.size(), 0);

        // Single element
        let single = k_subsets(1, 1);
        assert_eq!(single.len(), 1);
        assert_eq!(single[0].elements(), &[0]);

        // k = 0
        let zero_subsets = k_subsets(5, 0);
        assert_eq!(zero_subsets.len(), 1);
        assert_eq!(zero_subsets[0].elements(), &[]);

        // k = n
        let full_subsets = k_subsets(3, 3);
        assert_eq!(full_subsets.len(), 1);
        assert_eq!(full_subsets[0].elements(), &[0, 1, 2]);

        // k > n
        let invalid = k_subsets(3, 5);
        assert_eq!(invalid.len(), 0);
    }
}
