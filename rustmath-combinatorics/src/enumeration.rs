//! Enumeration utilities for combinatorial objects
//!
//! This module provides general utilities for enumerating and iterating over
//! combinatorial structures, including lazy iterators and generating functions.

use rustmath_core::NumericConversion;
use std::collections::VecDeque;

/// A trait for combinatorial objects that can be enumerated
pub trait Enumerable: Sized {
    /// Generate all objects with the given parameters
    fn enumerate(params: &Self::Params) -> Vec<Self>;

    /// Parameters needed to specify what to enumerate
    type Params;

    /// Count the number of objects (may be more efficient than enumerate().len())
    fn count(params: &Self::Params) -> usize {
        Self::enumerate(params).len()
    }
}

/// A lazy iterator for combinatorial objects
///
/// This generates objects one at a time rather than all at once,
/// which is more memory-efficient for large collections
pub struct LazyEnumerator<T, F>
where
    F: FnMut() -> Option<T>,
{
    generator: F,
}

impl<T, F> LazyEnumerator<T, F>
where
    F: FnMut() -> Option<T>,
{
    /// Create a new lazy enumerator from a generator function
    pub fn new(generator: F) -> Self {
        LazyEnumerator { generator }
    }
}

impl<T, F> Iterator for LazyEnumerator<T, F>
where
    F: FnMut() -> Option<T>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        (self.generator)()
    }
}

/// A Gray code iterator for subsets
///
/// Generates all 2^n subsets of {0, 1, ..., n-1} such that
/// consecutive subsets differ by exactly one element (Gray code order)
pub struct GrayCodeIterator {
    n: usize,
    current: usize,
    max: usize,
}

impl GrayCodeIterator {
    /// Create a new Gray code iterator for n elements
    pub fn new(n: usize) -> Self {
        if n > 30 {
            panic!("Gray code iterator: n too large (max 30)");
        }
        GrayCodeIterator {
            n,
            current: 0,
            max: 1 << n,
        }
    }

    /// Convert Gray code to binary
    fn gray_to_binary(gray: usize) -> usize {
        let mut binary = gray;
        let mut mask = gray >> 1;
        while mask != 0 {
            binary ^= mask;
            mask >>= 1;
        }
        binary
    }

    /// Convert subset index to actual subset
    fn index_to_subset(index: usize, n: usize) -> Vec<usize> {
        (0..n).filter(|&i| (index >> i) & 1 == 1).collect()
    }
}

impl Iterator for GrayCodeIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.max {
            return None;
        }

        // Convert index to Gray code, then to subset
        let gray = self.current ^ (self.current >> 1);
        let subset = Self::index_to_subset(gray, self.n);
        self.current += 1;

        Some(subset)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.max - self.current;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for GrayCodeIterator {}

/// A combination iterator using revolving door algorithm
///
/// Generates combinations in a minimal change order where successive
/// combinations differ by swapping adjacent elements
pub struct RevolvingDoorIterator {
    n: usize,
    k: usize,
    current: Option<Vec<usize>>,
    done: bool,
}

impl RevolvingDoorIterator {
    /// Create a new revolving door iterator for C(n, k)
    pub fn new(n: usize, k: usize) -> Self {
        if k > n {
            return RevolvingDoorIterator {
                n,
                k,
                current: None,
                done: true,
            };
        }

        let initial = if k == 0 {
            Some(vec![])
        } else {
            Some((0..k).collect())
        };

        RevolvingDoorIterator {
            n,
            k,
            current: initial,
            done: false,
        }
    }

    fn next_combination(&mut self) -> Option<Vec<usize>> {
        if self.done || self.current.is_none() {
            return None;
        }

        let result = self.current.clone();
        let comb = self.current.as_mut()?;

        if self.k == 0 {
            self.done = true;
            return result;
        }

        // Find rightmost element that can be incremented
        let mut i = self.k;
        while i > 0 && comb[i - 1] == self.n - self.k + i - 1 {
            i -= 1;
        }

        if i == 0 {
            self.done = true;
        } else {
            comb[i - 1] += 1;
            for j in i..self.k {
                comb[j] = comb[i - 1] + j - i + 1;
            }
        }

        result
    }
}

impl Iterator for RevolvingDoorIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_combination()
    }
}

/// A partition iterator
///
/// Generates all integer partitions of n
pub struct PartitionIterator {
    n: usize,
    stack: VecDeque<(usize, usize, Vec<usize>)>, // (remaining, max_value, current)
}

impl PartitionIterator {
    /// Create a new partition iterator for n
    pub fn new(n: usize) -> Self {
        let mut stack = VecDeque::new();
        stack.push_back((n, n, vec![]));

        PartitionIterator { n, stack }
    }
}

impl Iterator for PartitionIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((remaining, max_value, current)) = self.stack.pop_front() {
            if remaining == 0 {
                return Some(current);
            }

            // Try all values from max_value down to 1
            for value in (1..=max_value.min(remaining)).rev() {
                let mut new_current = current.clone();
                new_current.push(value);
                self.stack
                    .push_back((remaining - value, value, new_current));
            }
        }

        None
    }
}

/// A composition iterator
///
/// Generates all compositions (ordered partitions) of n
pub struct CompositionIterator {
    n: usize,
    current: Option<Vec<usize>>,
    stack: VecDeque<(usize, Vec<usize>)>,
}

impl CompositionIterator {
    /// Create a new composition iterator for n
    pub fn new(n: usize) -> Self {
        let mut stack = VecDeque::new();
        if n == 0 {
            stack.push_back((0, vec![]));
        } else {
            stack.push_back((n, vec![]));
        }

        CompositionIterator {
            n,
            current: None,
            stack,
        }
    }
}

impl Iterator for CompositionIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((remaining, current)) = self.stack.pop_front() {
            if remaining == 0 {
                return Some(current);
            }

            // Try all possible next parts
            for part in (1..=remaining).rev() {
                let mut new_current = current.clone();
                new_current.push(part);
                self.stack.push_back((remaining - part, new_current));
            }
        }

        None
    }
}

/// Generate Cartesian product of multiple sets
pub fn cartesian_product<T: Clone>(sets: &[Vec<T>]) -> Vec<Vec<T>> {
    if sets.is_empty() {
        return vec![vec![]];
    }

    if sets.len() == 1 {
        return sets[0].iter().map(|x| vec![x.clone()]).collect();
    }

    let mut result = Vec::new();
    let rest = cartesian_product(&sets[1..]);

    for elem in &sets[0] {
        for tuple in &rest {
            let mut new_tuple = vec![elem.clone()];
            new_tuple.extend(tuple.iter().cloned());
            result.push(new_tuple);
        }
    }

    result
}

/// Generate all k-tuples from a set (Cartesian power)
pub fn tuples<T: Clone>(set: &[T], k: usize) -> Vec<Vec<T>> {
    if k == 0 {
        return vec![vec![]];
    }

    let sets: Vec<_> = (0..k).map(|_| set.to_vec()).collect();
    cartesian_product(&sets)
}

/// Count the number of ways to distribute n identical objects into k distinct bins
///
/// This is the "stars and bars" problem: C(n+k-1, k-1)
pub fn stars_and_bars(n: usize, k: usize) -> usize {
    if k == 0 {
        return if n == 0 { 1 } else { 0 };
    }
    if k == 1 {
        return 1;
    }

    use crate::binomial;
    binomial((n + k - 1) as u32, (k - 1) as u32)
        .to_usize()
        .unwrap_or(0)
}

/// Generate all weak compositions of n into k parts (parts can be 0)
///
/// These correspond to distributing n identical objects into k distinct bins
pub fn weak_compositions(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return if n == 0 { vec![vec![]] } else { vec![] };
    }

    let mut result = Vec::new();
    let mut current = vec![0; k];

    generate_weak_compositions(n, k, 0, &mut current, &mut result);

    result
}

fn generate_weak_compositions(
    remaining: usize,
    k: usize,
    index: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if index == k {
        if remaining == 0 {
            result.push(current.clone());
        }
        return;
    }

    for value in 0..=remaining {
        current[index] = value;
        generate_weak_compositions(remaining - value, k, index + 1, current, result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gray_code_iterator() {
        let subsets: Vec<_> = GrayCodeIterator::new(3).collect();
        assert_eq!(subsets.len(), 8); // 2^3

        // Check that consecutive subsets differ by exactly one element
        for i in 1..subsets.len() {
            let prev_set: std::collections::HashSet<_> = subsets[i - 1].iter().copied().collect();
            let curr_set: std::collections::HashSet<_> = subsets[i].iter().copied().collect();

            let diff_count = prev_set.symmetric_difference(&curr_set).count();

            // In Gray code, consecutive subsets should differ by exactly 1 element
            assert_eq!(diff_count, 1, "Gray code property violated at index {}: prev={:?}, curr={:?}", i, subsets[i-1], subsets[i]);
        }
    }

    #[test]
    fn test_revolving_door_iterator() {
        let combinations: Vec<_> = RevolvingDoorIterator::new(5, 3).collect();
        assert_eq!(combinations.len(), 10); // C(5,3) = 10

        // First should be [0,1,2]
        assert_eq!(combinations[0], vec![0, 1, 2]);

        // Last should be [2,3,4]
        assert_eq!(combinations[9], vec![2, 3, 4]);
    }

    #[test]
    fn test_partition_iterator() {
        let partitions: Vec<_> = PartitionIterator::new(4).collect();
        assert_eq!(partitions.len(), 5); // p(4) = 5

        // Should include [4], [3,1], [2,2], [2,1,1], [1,1,1,1]
        assert!(partitions.contains(&vec![4]));
        assert!(partitions.contains(&vec![3, 1]));
        assert!(partitions.contains(&vec![2, 2]));
        assert!(partitions.contains(&vec![2, 1, 1]));
        assert!(partitions.contains(&vec![1, 1, 1, 1]));
    }

    #[test]
    fn test_composition_iterator() {
        let compositions: Vec<_> = CompositionIterator::new(3).collect();
        assert_eq!(compositions.len(), 4); // 2^(n-1) = 2^2 = 4

        // Should include [3], [1,2], [2,1], [1,1,1]
        assert!(compositions.contains(&vec![3]));
        assert!(compositions.contains(&vec![1, 2]));
        assert!(compositions.contains(&vec![2, 1]));
        assert!(compositions.contains(&vec![1, 1, 1]));
    }

    #[test]
    fn test_cartesian_product() {
        let set1 = vec![1, 2];
        let set2 = vec![3, 4];
        let set3 = vec![5];

        let product = cartesian_product(&[set1, set2, set3]);
        assert_eq!(product.len(), 4); // 2 * 2 * 1

        assert!(product.contains(&vec![1, 3, 5]));
        assert!(product.contains(&vec![1, 4, 5]));
        assert!(product.contains(&vec![2, 3, 5]));
        assert!(product.contains(&vec![2, 4, 5]));
    }

    #[test]
    fn test_tuples() {
        let set = vec![0, 1];
        let tuples_2 = tuples(&set, 2);

        assert_eq!(tuples_2.len(), 4); // 2^2
        assert!(tuples_2.contains(&vec![0, 0]));
        assert!(tuples_2.contains(&vec![0, 1]));
        assert!(tuples_2.contains(&vec![1, 0]));
        assert!(tuples_2.contains(&vec![1, 1]));
    }

    #[test]
    fn test_stars_and_bars() {
        // Distribute 5 objects into 3 bins: C(5+3-1, 3-1) = C(7, 2) = 21
        assert_eq!(stars_and_bars(5, 3), 21);

        // Distribute 0 objects into 0 bins: 1 way (empty distribution)
        assert_eq!(stars_and_bars(0, 0), 1);

        // Distribute 5 objects into 1 bin: 1 way (all in one bin)
        assert_eq!(stars_and_bars(5, 1), 1);
    }

    #[test]
    fn test_weak_compositions() {
        let weak_comps = weak_compositions(3, 2);
        // Should include: [0,3], [1,2], [2,1], [3,0]
        assert_eq!(weak_comps.len(), 4);

        assert!(weak_comps.contains(&vec![0, 3]));
        assert!(weak_comps.contains(&vec![1, 2]));
        assert!(weak_comps.contains(&vec![2, 1]));
        assert!(weak_comps.contains(&vec![3, 0]));

        // All should sum to 3
        for comp in &weak_comps {
            assert_eq!(comp.iter().sum::<usize>(), 3);
        }
    }

    #[test]
    fn test_weak_compositions_count() {
        // Number of weak compositions of n into k parts should match stars_and_bars
        let n = 4;
        let k = 3;
        let weak_comps = weak_compositions(n, k);
        assert_eq!(weak_comps.len(), stars_and_bars(n, k));
    }
}
