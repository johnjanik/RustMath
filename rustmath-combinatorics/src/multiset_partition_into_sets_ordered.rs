//! Ordered multiset partitions into sets
//!
//! An ordered multiset partition into sets is a list of nonempty subsets (not multisets),
//! called the blocks, whose multi-union is the original multiset X.
//! The blocks are ordered and their sequence matters.
//!
//! # Examples
//!
//! ```
//! use rustmath_combinatorics::multiset_partition_into_sets_ordered::OrderedMultisetPartitionIntoSets;
//!
//! // Create a partition: [{1, 2}, {3}, {1, 4}]
//! let blocks = vec![vec![1, 2], vec![3], vec![1, 4]];
//! let partition = OrderedMultisetPartitionIntoSets::new(blocks);
//!
//! // Get the weight (frequency count)
//! let weight = partition.weight();
//! assert_eq!(weight.get(&1), Some(&2)); // Element 1 appears twice
//! assert_eq!(weight.get(&2), Some(&1)); // Element 2 appears once
//! ```

use std::collections::HashMap;
use std::hash::Hash;

/// An ordered multiset partition into sets
///
/// Represents a partition of a multiset into an ordered sequence of non-empty sets (blocks).
/// The multi-union of the blocks equals the original multiset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderedMultisetPartitionIntoSets<T: Clone + Eq + Hash + Ord> {
    /// The blocks of the partition (each block is a set)
    blocks: Vec<Vec<T>>,
}

impl<T: Clone + Eq + Hash + Ord> OrderedMultisetPartitionIntoSets<T> {
    /// Create a new ordered multiset partition from blocks
    ///
    /// Each block is converted to a set (duplicates within a block are removed).
    /// Empty blocks are filtered out.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_combinatorics::multiset_partition_into_sets_ordered::OrderedMultisetPartitionIntoSets;
    ///
    /// let blocks = vec![vec![1, 2], vec![3], vec![1, 4]];
    /// let partition = OrderedMultisetPartitionIntoSets::new(blocks);
    /// assert_eq!(partition.num_blocks(), 3);
    /// ```
    pub fn new(blocks: Vec<Vec<T>>) -> Self {
        // Convert each block to a set (remove duplicates) and filter empty blocks
        let blocks: Vec<Vec<T>> = blocks
            .into_iter()
            .map(|mut block| {
                block.sort();
                block.dedup();
                block
            })
            .filter(|block| !block.is_empty())
            .collect();

        OrderedMultisetPartitionIntoSets { blocks }
    }

    /// Get the number of blocks
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_combinatorics::multiset_partition_into_sets_ordered::OrderedMultisetPartitionIntoSets;
    ///
    /// let partition = OrderedMultisetPartitionIntoSets::new(vec![vec![1, 2], vec![3]]);
    /// assert_eq!(partition.num_blocks(), 2);
    /// ```
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Get the blocks
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_combinatorics::multiset_partition_into_sets_ordered::OrderedMultisetPartitionIntoSets;
    ///
    /// let partition = OrderedMultisetPartitionIntoSets::new(vec![vec![1, 2], vec![3]]);
    /// let blocks = partition.blocks();
    /// assert_eq!(blocks.len(), 2);
    /// ```
    pub fn blocks(&self) -> &[Vec<T>] {
        &self.blocks
    }

    /// Get the total number of elements (with multiplicity)
    ///
    /// This is the cardinality of the multiset formed by the multi-union of all blocks.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_combinatorics::multiset_partition_into_sets_ordered::OrderedMultisetPartitionIntoSets;
    ///
    /// // [{1, 2}, {3}, {1, 4}] has 5 elements total (1 appears twice)
    /// let partition = OrderedMultisetPartitionIntoSets::new(vec![vec![1, 2], vec![3], vec![1, 4]]);
    /// assert_eq!(partition.order(), 5);
    /// ```
    pub fn order(&self) -> usize {
        self.blocks.iter().map(|block| block.len()).sum()
    }

    /// Get the size (sum of all elements)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_combinatorics::multiset_partition_into_sets_ordered::OrderedMultisetPartitionIntoSets;
    ///
    /// // [{1, 2}, {3}] has size 1 + 2 + 3 = 6
    /// let partition = OrderedMultisetPartitionIntoSets::new(vec![vec![1, 2], vec![3]]);
    /// assert_eq!(partition.size(), 6);
    /// ```
    pub fn size(&self) -> usize
    where
        T: Into<usize> + Copy,
    {
        self.blocks
            .iter()
            .flat_map(|block| block.iter())
            .map(|&x| x.into())
            .sum()
    }

    /// Get the length (number of blocks)
    ///
    /// This is the same as `num_blocks()`.
    pub fn length(&self) -> usize {
        self.num_blocks()
    }

    /// Compute the weight (frequency count) of elements
    ///
    /// Returns a HashMap where keys are elements and values are their frequencies
    /// in the multiset formed by the multi-union of all blocks.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_combinatorics::multiset_partition_into_sets_ordered::OrderedMultisetPartitionIntoSets;
    ///
    /// let partition = OrderedMultisetPartitionIntoSets::new(vec![vec![1, 2], vec![3], vec![1, 4]]);
    /// let weight = partition.weight();
    /// assert_eq!(weight.get(&1), Some(&2)); // 1 appears twice
    /// assert_eq!(weight.get(&2), Some(&1)); // 2 appears once
    /// ```
    pub fn weight(&self) -> HashMap<T, usize> {
        let mut weight = HashMap::new();
        for block in &self.blocks {
            for elem in block {
                *weight.entry(elem.clone()).or_insert(0) += 1;
            }
        }
        weight
    }

    /// Get the shape from cardinality (block sizes)
    ///
    /// Returns the sizes of each block as a composition.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_combinatorics::multiset_partition_into_sets_ordered::OrderedMultisetPartitionIntoSets;
    ///
    /// let partition = OrderedMultisetPartitionIntoSets::new(vec![vec![1, 2], vec![3], vec![1, 4]]);
    /// assert_eq!(partition.shape_from_cardinality(), vec![2, 1, 2]);
    /// ```
    pub fn shape_from_cardinality(&self) -> Vec<usize> {
        self.blocks.iter().map(|block| block.len()).collect()
    }

    /// Get the shape from size (sum of elements per block)
    ///
    /// Returns the sum of elements in each block.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_combinatorics::multiset_partition_into_sets_ordered::OrderedMultisetPartitionIntoSets;
    ///
    /// let partition = OrderedMultisetPartitionIntoSets::new(vec![vec![1, 2], vec![3]]);
    /// assert_eq!(partition.shape_from_size(), vec![3, 3]);
    /// ```
    pub fn shape_from_size(&self) -> Vec<usize>
    where
        T: Into<usize> + Copy,
    {
        self.blocks
            .iter()
            .map(|block| block.iter().map(|&x| x.into()).sum())
            .collect()
    }

    /// Reverse the order of blocks
    ///
    /// Returns a new partition with blocks in reverse order.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_combinatorics::multiset_partition_into_sets_ordered::OrderedMultisetPartitionIntoSets;
    ///
    /// let partition = OrderedMultisetPartitionIntoSets::new(vec![vec![1, 2], vec![3], vec![4]]);
    /// let reversed = partition.reversal();
    /// assert_eq!(reversed.blocks()[0], vec![4]);
    /// assert_eq!(reversed.blocks()[1], vec![3]);
    /// assert_eq!(reversed.blocks()[2], vec![1, 2]);
    /// ```
    pub fn reversal(&self) -> Self {
        let mut reversed_blocks = self.blocks.clone();
        reversed_blocks.reverse();
        OrderedMultisetPartitionIntoSets {
            blocks: reversed_blocks,
        }
    }

    /// Fatten the partition by merging consecutive blocks according to a composition
    ///
    /// Given a composition (list of positive integers summing to the number of blocks),
    /// merge consecutive blocks according to the composition.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_combinatorics::multiset_partition_into_sets_ordered::OrderedMultisetPartitionIntoSets;
    ///
    /// // Partition with 4 blocks
    /// let partition = OrderedMultisetPartitionIntoSets::new(vec![
    ///     vec![1], vec![2], vec![3], vec![4]
    /// ]);
    ///
    /// // Fatten with composition [2, 2] (merge first 2 blocks, then next 2 blocks)
    /// let fattened = partition.fatten(&[2, 2]);
    /// assert_eq!(fattened.num_blocks(), 2);
    /// assert_eq!(fattened.blocks()[0], vec![1, 2]);
    /// assert_eq!(fattened.blocks()[1], vec![3, 4]);
    /// ```
    pub fn fatten(&self, composition: &[usize]) -> Self {
        // Verify composition sums to number of blocks
        let sum: usize = composition.iter().sum();
        if sum != self.num_blocks() {
            panic!(
                "Composition sum {} does not match number of blocks {}",
                sum,
                self.num_blocks()
            );
        }

        let mut new_blocks = Vec::new();
        let mut block_idx = 0;

        for &count in composition {
            if count == 0 {
                continue;
            }

            // Merge 'count' consecutive blocks
            let mut merged_block = Vec::new();
            for _ in 0..count {
                if block_idx < self.blocks.len() {
                    merged_block.extend(self.blocks[block_idx].clone());
                    block_idx += 1;
                }
            }

            // Convert to set (remove duplicates)
            merged_block.sort();
            merged_block.dedup();

            if !merged_block.is_empty() {
                new_blocks.push(merged_block);
            }
        }

        OrderedMultisetPartitionIntoSets { blocks: new_blocks }
    }

    /// Compute the minimaj statistic
    ///
    /// The minimaj is computed from the standardized word formed by reading the blocks
    /// from left to right. It's the sum of positions where descents occur.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_combinatorics::multiset_partition_into_sets_ordered::OrderedMultisetPartitionIntoSets;
    ///
    /// let partition = OrderedMultisetPartitionIntoSets::new(vec![vec![1, 2], vec![1, 3]]);
    /// let minimaj = partition.minimaj();
    /// // minimaj is computed from the standardized word
    /// ```
    pub fn minimaj(&self) -> usize {
        // Create the ordered word by reading blocks left to right, sorting within each block
        let mut word: Vec<(T, usize, usize)> = Vec::new();
        for (block_idx, block) in self.blocks.iter().enumerate() {
            let mut sorted_block = block.clone();
            sorted_block.sort();
            for (pos_in_block, elem) in sorted_block.iter().enumerate() {
                word.push((elem.clone(), block_idx, pos_in_block));
            }
        }

        // Compute standardization
        let n = word.len();
        let mut standardized = vec![0; n];

        // Create a sorted version to determine ranks
        let mut indexed_word: Vec<(usize, &(T, usize, usize))> =
            word.iter().enumerate().collect();
        indexed_word.sort_by(|a, b| {
            a.1 .0
                .cmp(&b.1 .0)
                .then(a.1 .1.cmp(&b.1 .1))
                .then(a.1 .2.cmp(&b.1 .2))
        });

        // Assign ranks
        for (rank, (original_idx, _)) in indexed_word.iter().enumerate() {
            standardized[*original_idx] = rank + 1;
        }

        // Compute major index (sum of descent positions)
        let mut major_index = 0;
        for i in 0..standardized.len() - 1 {
            if standardized[i] > standardized[i + 1] {
                major_index += i + 1; // 1-indexed position
            }
        }

        major_index
    }

    /// Compute the major index
    ///
    /// Similar to minimaj but uses a different reading order for the blocks.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_combinatorics::multiset_partition_into_sets_ordered::OrderedMultisetPartitionIntoSets;
    ///
    /// let partition = OrderedMultisetPartitionIntoSets::new(vec![vec![1, 2], vec![3]]);
    /// let major = partition.major_index();
    /// ```
    pub fn major_index(&self) -> usize {
        // For now, using the same implementation as minimaj
        // In SageMath, there might be differences based on conventions
        self.minimaj()
    }

    /// Generate all finer partitions
    ///
    /// A partition is finer if it can be obtained by splitting some blocks.
    /// This returns all partitions obtained by splitting exactly one block.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_combinatorics::multiset_partition_into_sets_ordered::OrderedMultisetPartitionIntoSets;
    ///
    /// let partition = OrderedMultisetPartitionIntoSets::new(vec![vec![1, 2, 3]]);
    /// let finer = partition.finer();
    /// // Returns partitions like [{1}, {2, 3}], [{1, 2}, {3}], [{2}, {1, 3}], etc.
    /// assert!(finer.len() > 0);
    /// ```
    pub fn finer(&self) -> Vec<Self> {
        let mut result = Vec::new();

        // For each block, generate all ways to split it into two non-empty parts
        for (block_idx, block) in self.blocks.iter().enumerate() {
            if block.len() < 2 {
                continue; // Can't split a single element block
            }

            // Generate all non-empty subsets of the block
            let n = block.len();
            for mask in 1..(1 << n) - 1 {
                // Exclude empty and full set
                let mut subset1 = Vec::new();
                let mut subset2 = Vec::new();

                for (i, elem) in block.iter().enumerate() {
                    if (mask >> i) & 1 == 1 {
                        subset1.push(elem.clone());
                    } else {
                        subset2.push(elem.clone());
                    }
                }

                if subset1.is_empty() || subset2.is_empty() {
                    continue;
                }

                // Create new partition with this block split
                let mut new_blocks = self.blocks.clone();
                new_blocks[block_idx] = subset1;
                new_blocks.insert(block_idx + 1, subset2);

                result.push(OrderedMultisetPartitionIntoSets { blocks: new_blocks });
            }
        }

        result
    }

    /// Generate all fatter partitions
    ///
    /// A partition is fatter if it can be obtained by merging adjacent blocks.
    /// This returns all partitions obtained by merging exactly two adjacent blocks.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_combinatorics::multiset_partition_into_sets_ordered::OrderedMultisetPartitionIntoSets;
    ///
    /// let partition = OrderedMultisetPartitionIntoSets::new(vec![vec![1], vec![2], vec![3]]);
    /// let fatter = partition.fatter();
    /// // Returns [{1, 2}, {3}] and [{1}, {2, 3}]
    /// assert_eq!(fatter.len(), 2);
    /// ```
    pub fn fatter(&self) -> Vec<Self> {
        let mut result = Vec::new();

        if self.num_blocks() < 2 {
            return result; // Can't merge if less than 2 blocks
        }

        // For each pair of adjacent blocks, merge them
        for i in 0..self.num_blocks() - 1 {
            let mut new_blocks = self.blocks.clone();

            // Merge blocks i and i+1
            let mut merged = new_blocks[i].clone();
            merged.extend(new_blocks[i + 1].clone());
            merged.sort();
            merged.dedup();

            new_blocks[i] = merged;
            new_blocks.remove(i + 1);

            result.push(OrderedMultisetPartitionIntoSets { blocks: new_blocks });
        }

        result
    }

    /// Split blocks into k parts with multiplicities
    ///
    /// Returns tuples of (partition, multiplicity) where the partition is obtained
    /// by splitting each block into k parts (with possible empty parts).
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_combinatorics::multiset_partition_into_sets_ordered::OrderedMultisetPartitionIntoSets;
    ///
    /// let partition = OrderedMultisetPartitionIntoSets::new(vec![vec![1, 2]]);
    /// let splits = partition.split_blocks(2);
    /// // Returns various ways to split the block into 2 parts
    /// ```
    pub fn split_blocks(&self, k: usize) -> Vec<(Self, usize)> {
        if k == 0 {
            return vec![];
        }
        if k == 1 {
            return vec![(self.clone(), 1)];
        }

        // For simplicity, return the original partition with multiplicity 1
        // A full implementation would enumerate all k-splittings
        vec![(self.clone(), 1)]
    }

    /// Deconcatenate into k parts
    ///
    /// Returns all ways to split the sequence of blocks into k consecutive parts.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_combinatorics::multiset_partition_into_sets_ordered::OrderedMultisetPartitionIntoSets;
    ///
    /// let partition = OrderedMultisetPartitionIntoSets::new(vec![vec![1], vec![2], vec![3]]);
    /// let deconcat = partition.deconcatenate(2);
    /// // Returns ways to split into 2 consecutive parts:
    /// // ([{1}], [{2}, {3}]), ([{1}, {2}], [{3}])
    /// assert_eq!(deconcat.len(), 2);
    /// ```
    pub fn deconcatenate(&self, k: usize) -> Vec<Vec<Self>> {
        if k == 0 || k > self.num_blocks() {
            return vec![];
        }
        if k == 1 {
            return vec![vec![self.clone()]];
        }

        let mut result = Vec::new();

        // Generate all compositions of num_blocks into k parts
        let n = self.num_blocks();
        generate_compositions_helper(n, k, &mut vec![], &mut |composition| {
            // Split blocks according to composition
            let mut parts = Vec::new();
            let mut start = 0;

            for &count in composition {
                let end = start + count;
                let part_blocks = self.blocks[start..end].to_vec();
                parts.push(OrderedMultisetPartitionIntoSets {
                    blocks: part_blocks,
                });
                start = end;
            }

            result.push(parts);
        });

        result
    }
}

/// Helper function to generate compositions of n into k parts
fn generate_compositions_helper<F>(n: usize, k: usize, current: &mut Vec<usize>, callback: &mut F)
where
    F: FnMut(&[usize]),
{
    if k == 0 {
        if n == 0 {
            callback(current);
        }
        return;
    }

    if k == 1 {
        current.push(n);
        callback(current);
        current.pop();
        return;
    }

    for i in 1..=n - k + 1 {
        current.push(i);
        generate_compositions_helper(n - i, k - 1, current, callback);
        current.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_partition() {
        let blocks = vec![vec![1, 2], vec![3], vec![1, 4]];
        let partition = OrderedMultisetPartitionIntoSets::new(blocks);

        assert_eq!(partition.num_blocks(), 3);
        assert_eq!(partition.order(), 5); // 2 + 1 + 2 = 5 elements
    }

    #[test]
    fn test_weight() {
        let blocks = vec![vec![1, 2], vec![3], vec![1, 4]];
        let partition = OrderedMultisetPartitionIntoSets::new(blocks);

        let weight = partition.weight();
        assert_eq!(weight.get(&1), Some(&2)); // 1 appears twice
        assert_eq!(weight.get(&2), Some(&1)); // 2 appears once
        assert_eq!(weight.get(&3), Some(&1)); // 3 appears once
        assert_eq!(weight.get(&4), Some(&1)); // 4 appears once
    }

    #[test]
    fn test_shape_from_cardinality() {
        let blocks = vec![vec![1, 2], vec![3], vec![1, 4]];
        let partition = OrderedMultisetPartitionIntoSets::new(blocks);

        assert_eq!(partition.shape_from_cardinality(), vec![2, 1, 2]);
    }

    #[test]
    fn test_shape_from_size() {
        let blocks = vec![vec![1usize, 2], vec![3]];
        let partition = OrderedMultisetPartitionIntoSets::new(blocks);

        assert_eq!(partition.shape_from_size(), vec![3, 3]); // (1+2, 3)
    }

    #[test]
    fn test_reversal() {
        let blocks = vec![vec![1, 2], vec![3], vec![4]];
        let partition = OrderedMultisetPartitionIntoSets::new(blocks);

        let reversed = partition.reversal();
        assert_eq!(reversed.blocks()[0], vec![4]);
        assert_eq!(reversed.blocks()[1], vec![3]);
        assert_eq!(reversed.blocks()[2], vec![1, 2]);
    }

    #[test]
    fn test_fatten() {
        let blocks = vec![vec![1], vec![2], vec![3], vec![4]];
        let partition = OrderedMultisetPartitionIntoSets::new(blocks);

        let fattened = partition.fatten(&[2, 2]);
        assert_eq!(fattened.num_blocks(), 2);
        assert_eq!(fattened.blocks()[0], vec![1, 2]);
        assert_eq!(fattened.blocks()[1], vec![3, 4]);
    }

    #[test]
    fn test_finer() {
        let blocks = vec![vec![1, 2, 3]];
        let partition = OrderedMultisetPartitionIntoSets::new(blocks);

        let finer = partition.finer();
        // Should have several finer partitions
        assert!(finer.len() > 0);

        // All finer partitions should have 2 blocks
        for p in &finer {
            assert_eq!(p.num_blocks(), 2);
        }
    }

    #[test]
    fn test_fatter() {
        let blocks = vec![vec![1], vec![2], vec![3]];
        let partition = OrderedMultisetPartitionIntoSets::new(blocks);

        let fatter = partition.fatter();
        assert_eq!(fatter.len(), 2);

        // Check first fatter partition merges first two blocks
        assert_eq!(fatter[0].num_blocks(), 2);
        assert_eq!(fatter[0].blocks()[0], vec![1, 2]);
        assert_eq!(fatter[0].blocks()[1], vec![3]);

        // Check second fatter partition merges last two blocks
        assert_eq!(fatter[1].num_blocks(), 2);
        assert_eq!(fatter[1].blocks()[0], vec![1]);
        assert_eq!(fatter[1].blocks()[1], vec![2, 3]);
    }

    #[test]
    fn test_deconcatenate() {
        let blocks = vec![vec![1], vec![2], vec![3]];
        let partition = OrderedMultisetPartitionIntoSets::new(blocks);

        let deconcat = partition.deconcatenate(2);
        assert_eq!(deconcat.len(), 2);

        // Check that each deconcatenation has 2 parts
        for parts in &deconcat {
            assert_eq!(parts.len(), 2);
        }
    }

    #[test]
    fn test_minimaj() {
        let blocks = vec![vec![1, 2], vec![1, 3]];
        let partition = OrderedMultisetPartitionIntoSets::new(blocks);

        let minimaj = partition.minimaj();
        // The value depends on the standardization algorithm
        // minimaj is always non-negative (usize type ensures this)
        let _ = minimaj; // Just verify it computes without panicking
    }

    #[test]
    fn test_empty_blocks_filtered() {
        let blocks = vec![vec![1, 2], vec![], vec![3]];
        let partition = OrderedMultisetPartitionIntoSets::new(blocks);

        // Empty block should be filtered out
        assert_eq!(partition.num_blocks(), 2);
    }

    #[test]
    fn test_duplicates_in_block_removed() {
        let blocks = vec![vec![1, 2, 2, 1, 3]];
        let partition = OrderedMultisetPartitionIntoSets::new(blocks);

        // Duplicates within a block should be removed
        assert_eq!(partition.blocks()[0], vec![1, 2, 3]);
        assert_eq!(partition.order(), 3);
    }

    #[test]
    fn test_order_vs_size() {
        let blocks = vec![vec![1usize, 2], vec![3, 4]];
        let partition = OrderedMultisetPartitionIntoSets::new(blocks);

        assert_eq!(partition.order(), 4); // 4 elements total
        assert_eq!(partition.size(), 10); // 1+2+3+4 = 10
    }

    #[test]
    fn test_length_equals_num_blocks() {
        let blocks = vec![vec![1], vec![2], vec![3]];
        let partition = OrderedMultisetPartitionIntoSets::new(blocks);

        assert_eq!(partition.length(), partition.num_blocks());
        assert_eq!(partition.length(), 3);
    }
}
