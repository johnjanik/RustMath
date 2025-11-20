//! Tuples and tuple generation with lexicographic ordering
//!
//! This module provides support for generating fixed-length tuples from a set
//! of elements, with guaranteed lexicographic ordering.

/// A fixed-length tuple from a set of elements
///
/// This represents a k-tuple where each position can be any element from the base set.
/// Tuples differ from combinations in that:
/// - Order matters: (0,1) ≠ (1,0)
/// - Repetition is allowed: (0,0) is valid
/// - There are n^k tuples from a set of size n
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Tuple {
    /// The tuple values (indices into the base set)
    indices: Vec<usize>,
    /// Size of the base set
    n: usize,
}

impl Tuple {
    /// Create a new tuple from a vector of indices
    ///
    /// Returns None if any index is >= n
    pub fn from_vec(indices: Vec<usize>, n: usize) -> Option<Self> {
        // Validate all indices
        for &idx in &indices {
            if idx >= n {
                return None;
            }
        }

        Some(Tuple { indices, n })
    }

    /// Get the length of the tuple (k)
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Check if the tuple is empty
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Get the size of the base set (n)
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the tuple indices
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Convert tuple to its lexicographic rank
    ///
    /// The rank is the position of this tuple in the lexicographically ordered
    /// list of all n^k tuples. Tuples are ordered like:
    /// (0,0,0), (0,0,1), ..., (0,0,n-1), (0,1,0), ..., (n-1,n-1,n-1)
    ///
    /// The rank formula is: sum(indices[i] * n^(k-1-i)) for i in 0..k
    pub fn rank(&self) -> usize {
        let k = self.len();
        let mut rank = 0;
        let mut power = 1;

        // Process from right to left (least significant to most significant)
        for i in (0..k).rev() {
            rank += self.indices[i] * power;
            power *= self.n;
        }

        rank
    }

    /// Create a tuple from its lexicographic rank
    ///
    /// This is the inverse of rank(). Given a rank r and parameters n, k,
    /// returns the tuple at position r in lexicographic order.
    pub fn unrank(rank: usize, n: usize, k: usize) -> Option<Self> {
        if n == 0 && k > 0 {
            return None;
        }

        // Check if rank is valid
        let total = n.checked_pow(k as u32)?;
        if rank >= total {
            return None;
        }

        let mut indices = vec![0; k];
        let mut remaining = rank;

        // Extract indices from right to left
        for i in (0..k).rev() {
            indices[i] = remaining % n;
            remaining /= n;
        }

        Some(Tuple { indices, n })
    }

    /// Get the next tuple in lexicographic order
    ///
    /// Returns None if this is the last tuple
    pub fn next(&self) -> Option<Self> {
        let k = self.len();
        if k == 0 {
            return None;
        }

        let mut new_indices = self.indices.clone();

        // Find the rightmost position that can be incremented
        for i in (0..k).rev() {
            if new_indices[i] < self.n - 1 {
                new_indices[i] += 1;
                return Some(Tuple {
                    indices: new_indices,
                    n: self.n,
                });
            }
            // Reset this position to 0 and continue to the left
            new_indices[i] = 0;
        }

        // All positions were at maximum - no next tuple
        None
    }

    /// Get the previous tuple in lexicographic order
    ///
    /// Returns None if this is the first tuple
    pub fn prev(&self) -> Option<Self> {
        let k = self.len();
        if k == 0 {
            return None;
        }

        let mut new_indices = self.indices.clone();

        // Find the rightmost position that can be decremented
        for i in (0..k).rev() {
            if new_indices[i] > 0 {
                new_indices[i] -= 1;
                return Some(Tuple {
                    indices: new_indices,
                    n: self.n,
                });
            }
            // Reset this position to max and continue to the left
            new_indices[i] = self.n - 1;
        }

        // All positions were at minimum - no previous tuple
        None
    }
}

/// Generate all k-tuples from n elements in lexicographic order
///
/// Returns all possible k-tuples where each position can be any of n values.
/// The tuples are returned in lexicographic order:
/// (0,0,...,0), (0,0,...,1), ..., (n-1,n-1,...,n-1)
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::tuple::tuples;
///
/// // All 2-tuples from {0, 1, 2}
/// let ts = tuples(3, 2);
/// assert_eq!(ts.len(), 9); // 3^2 = 9
///
/// // First few tuples: (0,0), (0,1), (0,2), (1,0), ...
/// assert_eq!(ts[0].indices(), &[0, 0]);
/// assert_eq!(ts[1].indices(), &[0, 1]);
/// assert_eq!(ts[2].indices(), &[0, 2]);
/// assert_eq!(ts[3].indices(), &[1, 0]);
/// ```
pub fn tuples(n: usize, k: usize) -> Vec<Tuple> {
    if k == 0 {
        return vec![Tuple {
            indices: vec![],
            n,
        }];
    }

    if n == 0 {
        return vec![];
    }

    let mut result = Vec::new();
    let mut current = vec![0; k];

    generate_tuples(n, k, 0, &mut current, &mut result);

    result
}

/// Recursive helper for generating tuples in lexicographic order
fn generate_tuples(
    n: usize,
    k: usize,
    pos: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Tuple>,
) {
    if pos == k {
        result.push(Tuple {
            indices: current.clone(),
            n,
        });
        return;
    }

    for i in 0..n {
        current[pos] = i;
        generate_tuples(n, k, pos + 1, current, result);
    }
}

/// An iterator over tuples in lexicographic order
///
/// This is more memory-efficient than generating all tuples at once.
pub struct TupleIterator {
    n: usize,
    k: usize,
    current: Option<Vec<usize>>,
    done: bool,
}

impl TupleIterator {
    /// Create a new tuple iterator for n^k tuples
    pub fn new(n: usize, k: usize) -> Self {
        let current = if k == 0 || n == 0 {
            None
        } else {
            Some(vec![0; k])
        };

        TupleIterator {
            n,
            k,
            current,
            done: false,
        }
    }

    /// Get the total number of tuples that will be generated
    pub fn count(&self) -> usize {
        if self.k == 0 {
            return 1;
        }
        self.n.saturating_pow(self.k as u32)
    }
}

impl Iterator for TupleIterator {
    type Item = Tuple;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        // Handle k=0 case (empty tuple)
        if self.k == 0 {
            self.done = true;
            return Some(Tuple {
                indices: vec![],
                n: self.n,
            });
        }

        // Handle n=0 case (no elements)
        if self.n == 0 {
            return None;
        }

        let current = self.current.as_ref()?;
        let result = Tuple {
            indices: current.clone(),
            n: self.n,
        };

        // Generate next tuple
        let mut next_indices = current.clone();
        let mut carry = true;

        for i in (0..self.k).rev() {
            if carry {
                if next_indices[i] < self.n - 1 {
                    next_indices[i] += 1;
                    carry = false;
                } else {
                    next_indices[i] = 0;
                }
            }
        }

        if carry {
            // We've generated all tuples
            self.done = true;
            self.current = None;
        } else {
            self.current = Some(next_indices);
        }

        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.done {
            return (0, Some(0));
        }

        let total = self.count();
        if let Some(ref current) = self.current {
            let tuple = Tuple {
                indices: current.clone(),
                n: self.n,
            };
            let remaining = total - tuple.rank();
            (remaining, Some(remaining))
        } else if self.k == 0 {
            (1, Some(1))
        } else {
            (0, Some(0))
        }
    }
}

impl ExactSizeIterator for TupleIterator {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tuple_creation() {
        let tuple = Tuple::from_vec(vec![0, 1, 2], 3).unwrap();
        assert_eq!(tuple.len(), 3);
        assert_eq!(tuple.n(), 3);
        assert_eq!(tuple.indices(), &[0, 1, 2]);

        // Invalid index
        assert!(Tuple::from_vec(vec![0, 1, 3], 3).is_none());
    }

    #[test]
    fn test_tuples_generation() {
        // All 2-tuples from {0, 1}
        let ts = tuples(2, 2);
        assert_eq!(ts.len(), 4); // 2^2

        // Should be in lexicographic order: (0,0), (0,1), (1,0), (1,1)
        assert_eq!(ts[0].indices(), &[0, 0]);
        assert_eq!(ts[1].indices(), &[0, 1]);
        assert_eq!(ts[2].indices(), &[1, 0]);
        assert_eq!(ts[3].indices(), &[1, 1]);
    }

    #[test]
    fn test_tuples_lexicographic_order() {
        // All 2-tuples from {0, 1, 2}
        let ts = tuples(3, 2);
        assert_eq!(ts.len(), 9); // 3^2

        // Verify lexicographic ordering
        let expected = vec![
            vec![0, 0],
            vec![0, 1],
            vec![0, 2],
            vec![1, 0],
            vec![1, 1],
            vec![1, 2],
            vec![2, 0],
            vec![2, 1],
            vec![2, 2],
        ];

        for (i, tuple) in ts.iter().enumerate() {
            assert_eq!(tuple.indices(), &expected[i]);
        }
    }

    #[test]
    fn test_rank_unrank() {
        // Test rank/unrank for all 3-tuples from {0,1,2}
        let ts = tuples(3, 3);

        for (expected_rank, tuple) in ts.iter().enumerate() {
            let rank = tuple.rank();
            assert_eq!(rank, expected_rank, "Rank mismatch for {:?}", tuple.indices());

            let unranked = Tuple::unrank(rank, 3, 3).unwrap();
            assert_eq!(
                unranked.indices(),
                tuple.indices(),
                "Unrank mismatch for rank {}",
                rank
            );
        }
    }

    #[test]
    fn test_rank_formula() {
        // (0,0,0) -> rank 0
        let t1 = Tuple::from_vec(vec![0, 0, 0], 3).unwrap();
        assert_eq!(t1.rank(), 0);

        // (0,0,1) -> rank 1
        let t2 = Tuple::from_vec(vec![0, 0, 1], 3).unwrap();
        assert_eq!(t2.rank(), 1);

        // (0,1,0) -> rank 3
        let t3 = Tuple::from_vec(vec![0, 1, 0], 3).unwrap();
        assert_eq!(t3.rank(), 3);

        // (1,0,0) -> rank 9
        let t4 = Tuple::from_vec(vec![1, 0, 0], 3).unwrap();
        assert_eq!(t4.rank(), 9);

        // (2,2,2) -> rank 26 (last tuple for n=3, k=3)
        let t5 = Tuple::from_vec(vec![2, 2, 2], 3).unwrap();
        assert_eq!(t5.rank(), 26);
    }

    #[test]
    fn test_next_prev() {
        let t1 = Tuple::from_vec(vec![0, 0], 2).unwrap();

        let t2 = t1.next().unwrap();
        assert_eq!(t2.indices(), &[0, 1]);

        let t3 = t2.next().unwrap();
        assert_eq!(t3.indices(), &[1, 0]);

        let t4 = t3.next().unwrap();
        assert_eq!(t4.indices(), &[1, 1]);

        // Last tuple has no next
        assert!(t4.next().is_none());

        // Test prev
        let t3_back = t4.prev().unwrap();
        assert_eq!(t3_back.indices(), &[1, 0]);

        let t2_back = t3_back.prev().unwrap();
        assert_eq!(t2_back.indices(), &[0, 1]);

        let t1_back = t2_back.prev().unwrap();
        assert_eq!(t1_back.indices(), &[0, 0]);

        // First tuple has no prev
        assert!(t1_back.prev().is_none());
    }

    #[test]
    fn test_edge_cases() {
        // Empty tuples (k=0)
        let empty = tuples(5, 0);
        assert_eq!(empty.len(), 1);
        assert_eq!(empty[0].len(), 0);

        // No elements (n=0)
        let none = tuples(0, 3);
        assert_eq!(none.len(), 0);

        // Single element
        let single = tuples(1, 3);
        assert_eq!(single.len(), 1);
        assert_eq!(single[0].indices(), &[0, 0, 0]);

        // Single position
        let positions = tuples(5, 1);
        assert_eq!(positions.len(), 5);
        assert_eq!(positions[0].indices(), &[0]);
        assert_eq!(positions[4].indices(), &[4]);
    }

    #[test]
    fn test_tuple_iterator() {
        let mut iter = TupleIterator::new(2, 2);

        let t1 = iter.next().unwrap();
        assert_eq!(t1.indices(), &[0, 0]);

        let t2 = iter.next().unwrap();
        assert_eq!(t2.indices(), &[0, 1]);

        let t3 = iter.next().unwrap();
        assert_eq!(t3.indices(), &[1, 0]);

        let t4 = iter.next().unwrap();
        assert_eq!(t4.indices(), &[1, 1]);

        assert!(iter.next().is_none());
    }

    #[test]
    fn test_tuple_iterator_count() {
        let iter = TupleIterator::new(2, 2);
        assert_eq!(iter.count(), 4);

        let iter2 = TupleIterator::new(3, 3);
        assert_eq!(iter2.count(), 27);
    }

    #[test]
    fn test_iterator_vs_generation() {
        // Verify iterator produces same results as direct generation
        let generated = tuples(3, 2);
        let iterated: Vec<_> = TupleIterator::new(3, 2).collect();

        assert_eq!(generated.len(), iterated.len());
        for (g, i) in generated.iter().zip(iterated.iter()) {
            assert_eq!(g.indices(), i.indices());
        }
    }

    #[test]
    fn test_iterator_size_hint() {
        let iter = TupleIterator::new(3, 2);
        let (lower, upper) = iter.size_hint();
        assert_eq!(lower, 9);
        assert_eq!(upper, Some(9));
    }

    #[test]
    fn test_large_tuples() {
        // Test that we can handle reasonably large cases
        let ts = tuples(4, 3);
        assert_eq!(ts.len(), 64); // 4^3

        // Verify first and last
        assert_eq!(ts[0].indices(), &[0, 0, 0]);
        assert_eq!(ts[63].indices(), &[3, 3, 3]);

        // Verify all have correct rank
        for (i, tuple) in ts.iter().enumerate() {
            assert_eq!(tuple.rank(), i);
        }
    }

    #[test]
    fn test_unrank_invalid() {
        // Rank out of bounds
        assert!(Tuple::unrank(100, 2, 2).is_none());

        // n=0, k>0
        assert!(Tuple::unrank(0, 0, 2).is_none());

        // Valid edge cases
        assert!(Tuple::unrank(0, 5, 0).is_some());
    }
}
