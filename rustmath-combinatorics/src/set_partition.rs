//! Set partitions
//!
//! A set partition is a way of partitioning a set into non-empty disjoint subsets.
//! The number of set partitions of n elements is the Bell number B(n).
//!
//! This module uses Restricted Growth Strings (RGS) for efficient generation and iteration.
//! An RGS is a sequence a[0], a[1], ..., a[n-1] where:
//! - a[0] = 0
//! - a[i] <= max(a[0], ..., a[i-1]) + 1 for all i > 0
//!
//! The value a[i] indicates which block element i belongs to.

/// A set partition - a way of partitioning a set into non-empty disjoint subsets
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetPartition {
    /// Each vector represents one block (subset) in the partition
    blocks: Vec<Vec<usize>>,
    /// Total number of elements
    n: usize,
}

impl SetPartition {
    /// Create a set partition from blocks
    pub fn new(blocks: Vec<Vec<usize>>, n: usize) -> Option<Self> {
        // Verify that blocks are non-empty and disjoint
        let mut seen = vec![false; n];

        for block in &blocks {
            if block.is_empty() {
                return None;
            }
            for &elem in block {
                if elem >= n || seen[elem] {
                    return None;
                }
                seen[elem] = true;
            }
        }

        // All elements must be covered
        if !seen.iter().all(|&x| x) {
            return None;
        }

        Some(SetPartition { blocks, n })
    }

    /// Get the number of blocks
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Get the blocks
    pub fn blocks(&self) -> &[Vec<usize>] {
        &self.blocks
    }

    /// Get the size of the set being partitioned
    pub fn size(&self) -> usize {
        self.n
    }

    /// Create a set partition from a restricted growth string (RGS)
    ///
    /// An RGS is a sequence where rgs[i] indicates which block element i belongs to.
    /// The RGS must satisfy: rgs[0] = 0 and rgs[i] <= max(rgs[0..i]) + 1
    pub fn from_rgs(rgs: &[usize]) -> Option<Self> {
        if rgs.is_empty() {
            return Some(SetPartition {
                blocks: vec![],
                n: 0,
            });
        }

        let n = rgs.len();

        // Verify RGS constraints
        if rgs[0] != 0 {
            return None;
        }

        let mut max_block = 0;
        for i in 1..n {
            if rgs[i] > max_block + 1 {
                return None;
            }
            if rgs[i] > max_block {
                max_block = rgs[i];
            }
        }

        // Convert RGS to blocks
        let num_blocks = max_block + 1;
        let mut blocks = vec![Vec::new(); num_blocks];

        for (elem, &block_id) in rgs.iter().enumerate() {
            blocks[block_id].push(elem);
        }

        Some(SetPartition { blocks, n })
    }

    /// Convert this set partition to a restricted growth string (RGS)
    ///
    /// Returns a vector where result[i] is the block number that element i belongs to.
    /// Block numbers are assigned in the order they first appear (starting from 0).
    pub fn to_rgs(&self) -> Vec<usize> {
        let mut rgs = vec![0; self.n];

        for (block_id, block) in self.blocks.iter().enumerate() {
            for &elem in block {
                rgs[elem] = block_id;
            }
        }

        rgs
    }

    /// Check if this partition is valid (for internal consistency checks)
    pub fn is_valid(&self) -> bool {
        let mut seen = vec![false; self.n];

        for block in &self.blocks {
            if block.is_empty() {
                return false;
            }
            for &elem in block {
                if elem >= self.n || seen[elem] {
                    return false;
                }
                seen[elem] = true;
            }
        }

        seen.iter().all(|&x| x)
    }
}

/// Generate all set partitions of n elements (labeled 0..n)
///
/// The number of set partitions is the Bell number B(n)
pub fn set_partitions(n: usize) -> Vec<SetPartition> {
    if n == 0 {
        return vec![SetPartition {
            blocks: vec![],
            n: 0,
        }];
    }

    let mut result = Vec::new();
    let elements: Vec<usize> = (0..n).collect();
    let mut current_partition: Vec<Vec<usize>> = Vec::new();

    generate_set_partitions(&elements, 0, &mut current_partition, &mut result);

    result
}

fn generate_set_partitions(
    elements: &[usize],
    index: usize,
    current: &mut Vec<Vec<usize>>,
    result: &mut Vec<SetPartition>,
) {
    if index == elements.len() {
        result.push(SetPartition {
            blocks: current.clone(),
            n: elements.len(),
        });
        return;
    }

    let elem = elements[index];

    // Try adding element to each existing block
    for i in 0..current.len() {
        current[i].push(elem);
        generate_set_partitions(elements, index + 1, current, result);
        current[i].pop();
    }

    // Try creating a new block with this element
    current.push(vec![elem]);
    generate_set_partitions(elements, index + 1, current, result);
    current.pop();
}

/// Iterator over all set partitions of n elements using restricted growth strings
///
/// This iterator generates set partitions in lexicographic order of their RGS representation,
/// which is more efficient than the recursive algorithm for large n.
#[derive(Debug, Clone)]
pub struct SetPartitionIterator {
    n: usize,
    rgs: Vec<usize>,
    done: bool,
}

impl SetPartitionIterator {
    /// Create a new iterator for set partitions of n elements
    pub fn new(n: usize) -> Self {
        if n == 0 {
            SetPartitionIterator {
                n: 0,
                rgs: vec![],
                done: false,
            }
        } else {
            SetPartitionIterator {
                n,
                rgs: vec![0; n],
                done: false,
            }
        }
    }

    /// Get the current RGS
    pub fn current_rgs(&self) -> &[usize] {
        &self.rgs
    }

    /// Compute the maximum value allowed at position i
    fn max_allowed(&self, i: usize) -> usize {
        if i == 0 {
            return 0; // First position must be 0
        }
        // Max allowed is max(rgs[0..i]) + 1
        let max_before = self.rgs[0..i].iter().max().unwrap_or(&0);
        max_before + 1
    }
}

impl Iterator for SetPartitionIterator {
    type Item = SetPartition;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        // Handle empty case
        if self.n == 0 {
            self.done = true;
            return Some(SetPartition {
                blocks: vec![],
                n: 0,
            });
        }

        // Generate current partition from RGS
        let current = SetPartition::from_rgs(&self.rgs)?;

        // Generate next RGS
        // Find rightmost position that can be incremented
        let mut i = self.n - 1;
        loop {
            let max_val = self.max_allowed(i);
            if self.rgs[i] < max_val {
                // Can increment this position
                self.rgs[i] += 1;

                // Reset all positions after i to 0
                for j in (i + 1)..self.n {
                    self.rgs[j] = 0;
                }
                break;
            }

            // Can't increment this position, try previous
            if i == 0 {
                // Can't increment any position, we're done
                self.done = true;
                break;
            }
            i -= 1;
        }

        Some(current)
    }
}

/// Create an iterator over all set partitions of n elements
///
/// This is more memory-efficient than `set_partitions()` for large n,
/// as it generates partitions on-demand rather than storing them all.
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::set_partition_iterator;
///
/// let mut count = 0;
/// for partition in set_partition_iterator(4) {
///     count += 1;
///     assert_eq!(partition.size(), 4);
/// }
/// assert_eq!(count, 15); // Bell(4) = 15
/// ```
pub fn set_partition_iterator(n: usize) -> SetPartitionIterator {
    SetPartitionIterator::new(n)
}

/// Compute Bell number B(n) using the Bell triangle (optimized algorithm)
///
/// The Bell number B(n) is the number of ways to partition n elements into non-empty subsets.
/// This implementation uses the Bell triangle, which is more efficient than summing Stirling numbers.
///
/// The Bell triangle is constructed as follows:
/// - Start with B(0) = 1
/// - Each row starts with the last element of the previous row
/// - Each element is the sum of the element to its left and the element above-left
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::bell_number_optimized;
/// use rustmath_integers::Integer;
///
/// assert_eq!(bell_number_optimized(0), Integer::from(1));
/// assert_eq!(bell_number_optimized(3), Integer::from(5));
/// assert_eq!(bell_number_optimized(5), Integer::from(52));
/// ```
pub fn bell_number_optimized(n: u32) -> crate::Integer {
    use crate::Integer;
    use rustmath_core::Ring;

    if n == 0 {
        return Integer::one();
    }

    // We only need to keep the previous row
    let mut prev_row = vec![Integer::one()];

    for i in 1..=n {
        let mut current_row = Vec::with_capacity((i + 1) as usize);
        // First element is the last element of previous row
        current_row.push(prev_row[prev_row.len() - 1].clone());

        // Each subsequent element is sum of left and above-left
        for j in 0..prev_row.len() {
            let sum = current_row[j].clone() + prev_row[j].clone();
            current_row.push(sum);
        }

        prev_row = current_row;
    }

    prev_row[0].clone()
}

/// Compute all Bell numbers from B(0) to B(n) efficiently
///
/// Returns a vector where result[i] = B(i).
/// This is more efficient than calling bell_number_optimized multiple times.
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::bell_numbers_up_to;
/// use rustmath_integers::Integer;
///
/// let bells = bell_numbers_up_to(5);
/// assert_eq!(bells[0], Integer::from(1));
/// assert_eq!(bells[3], Integer::from(5));
/// assert_eq!(bells[5], Integer::from(52));
/// ```
pub fn bell_numbers_up_to(n: u32) -> Vec<crate::Integer> {
    use crate::Integer;
    use rustmath_core::Ring;

    let mut result = vec![Integer::one()];

    if n == 0 {
        return result;
    }

    let mut prev_row = vec![Integer::one()];

    for i in 1..=n {
        let mut current_row = Vec::with_capacity((i + 1) as usize);
        current_row.push(prev_row[prev_row.len() - 1].clone());

        for j in 0..prev_row.len() {
            let sum = current_row[j].clone() + prev_row[j].clone();
            current_row.push(sum);
        }

        result.push(current_row[0].clone());
        prev_row = current_row;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_partitions() {
        // Set partitions of 3 elements should equal Bell(3) = 5
        let parts = set_partitions(3);
        assert_eq!(parts.len(), 5);

        // Verify all partitions are valid
        for part in &parts {
            assert_eq!(part.size(), 3);
            assert!(part.num_blocks() > 0);

            // Check that all elements 0,1,2 appear exactly once
            let mut seen = vec![false; 3];
            for block in part.blocks() {
                for &elem in block {
                    assert!(!seen[elem]);
                    seen[elem] = true;
                }
            }
            assert!(seen.iter().all(|&x| x));
        }
    }

    #[test]
    fn test_set_partition_count() {
        // Number of set partitions should match Bell numbers
        assert_eq!(set_partitions(0).len(), 1); // Bell(0) = 1
        assert_eq!(set_partitions(1).len(), 1); // Bell(1) = 1
        assert_eq!(set_partitions(2).len(), 2); // Bell(2) = 2
        assert_eq!(set_partitions(3).len(), 5); // Bell(3) = 5
        assert_eq!(set_partitions(4).len(), 15); // Bell(4) = 15
    }

    #[test]
    fn test_rgs_conversion() {
        // Test RGS to SetPartition conversion
        let rgs = vec![0, 0, 1, 0, 2];
        let partition = SetPartition::from_rgs(&rgs).unwrap();
        assert_eq!(partition.size(), 5);
        assert_eq!(partition.num_blocks(), 3);

        // Verify the partition has correct blocks
        let blocks = partition.blocks();
        assert!(blocks.contains(&vec![0, 1, 3]));
        assert!(blocks.contains(&vec![2]));
        assert!(blocks.contains(&vec![4]));

        // Test round-trip conversion
        let rgs2 = partition.to_rgs();
        assert_eq!(rgs, rgs2);
    }

    #[test]
    fn test_rgs_validation() {
        // Valid RGS
        assert!(SetPartition::from_rgs(&[0, 0, 1, 2, 1]).is_some());
        assert!(SetPartition::from_rgs(&[0, 1, 1, 2, 0]).is_some());

        // Invalid - doesn't start with 0
        assert!(SetPartition::from_rgs(&[1, 0, 0]).is_none());

        // Invalid - increment too large
        assert!(SetPartition::from_rgs(&[0, 2, 1]).is_none());
        assert!(SetPartition::from_rgs(&[0, 0, 0, 3]).is_none());

        // Valid - can increment by 1
        assert!(SetPartition::from_rgs(&[0, 1, 2, 3]).is_some());
    }

    #[test]
    fn test_set_partition_iterator() {
        // Test iterator for n=0
        let parts0: Vec<_> = set_partition_iterator(0).collect();
        assert_eq!(parts0.len(), 1);
        assert_eq!(parts0[0].size(), 0);

        // Test iterator for n=3
        let parts3: Vec<_> = set_partition_iterator(3).collect();
        assert_eq!(parts3.len(), 5); // Bell(3) = 5

        // Verify all partitions are valid
        for part in &parts3 {
            assert_eq!(part.size(), 3);
            assert!(part.is_valid());
        }

        // Test iterator for n=4
        let parts4: Vec<_> = set_partition_iterator(4).collect();
        assert_eq!(parts4.len(), 15); // Bell(4) = 15

        // Verify all partitions are valid
        for part in &parts4 {
            assert_eq!(part.size(), 4);
            assert!(part.is_valid());
        }
    }

    #[test]
    fn test_iterator_vs_recursive() {
        // Compare iterator results with recursive generation
        for n in 0..=5 {
            let iter_parts: Vec<_> = set_partition_iterator(n).collect();
            let recursive_parts = set_partitions(n);

            assert_eq!(iter_parts.len(), recursive_parts.len());

            // Convert to RGS for comparison (order might differ)
            let mut iter_rgs: Vec<_> = iter_parts.iter().map(|p| p.to_rgs()).collect();
            let mut recursive_rgs: Vec<_> = recursive_parts.iter().map(|p| p.to_rgs()).collect();

            iter_rgs.sort();
            recursive_rgs.sort();

            assert_eq!(iter_rgs, recursive_rgs);
        }
    }

    #[test]
    fn test_bell_number_optimized() {
        use crate::Integer;

        // First few Bell numbers: 1, 1, 2, 5, 15, 52, 203, 877
        assert_eq!(bell_number_optimized(0), Integer::from(1));
        assert_eq!(bell_number_optimized(1), Integer::from(1));
        assert_eq!(bell_number_optimized(2), Integer::from(2));
        assert_eq!(bell_number_optimized(3), Integer::from(5));
        assert_eq!(bell_number_optimized(4), Integer::from(15));
        assert_eq!(bell_number_optimized(5), Integer::from(52));
        assert_eq!(bell_number_optimized(6), Integer::from(203));
        assert_eq!(bell_number_optimized(7), Integer::from(877));
    }

    #[test]
    fn test_bell_number_vs_stirling() {
        use crate::{bell_number, Integer};

        // Verify optimized algorithm matches Stirling-based algorithm
        for n in 0..=8 {
            assert_eq!(bell_number_optimized(n), bell_number(n));
        }
    }

    #[test]
    fn test_bell_numbers_up_to() {
        use crate::Integer;

        let bells = bell_numbers_up_to(7);
        assert_eq!(bells.len(), 8);
        assert_eq!(bells[0], Integer::from(1));
        assert_eq!(bells[1], Integer::from(1));
        assert_eq!(bells[2], Integer::from(2));
        assert_eq!(bells[3], Integer::from(5));
        assert_eq!(bells[4], Integer::from(15));
        assert_eq!(bells[5], Integer::from(52));
        assert_eq!(bells[6], Integer::from(203));
        assert_eq!(bells[7], Integer::from(877));
    }

    #[test]
    fn test_iterator_count_matches_bell() {
        use crate::Integer;

        // Verify iterator produces correct count
        for n in 0..=6 {
            let count = set_partition_iterator(n).count();
            let bell = bell_number_optimized(n as u32);
            assert_eq!(Integer::from(count as u32), bell);
        }
    }

    #[test]
    fn test_specific_partitions() {
        // Test specific known partitions for n=3
        let parts3: Vec<_> = set_partition_iterator(3).collect();

        // Expected RGS for n=3: [0,0,0], [0,0,1], [0,1,0], [0,1,1], [0,1,2]
        let expected_rgs = vec![
            vec![0, 0, 0],
            vec![0, 0, 1],
            vec![0, 1, 0],
            vec![0, 1, 1],
            vec![0, 1, 2],
        ];

        let actual_rgs: Vec<_> = parts3.iter().map(|p| p.to_rgs()).collect();
        assert_eq!(actual_rgs, expected_rgs);
    }

    #[test]
    fn test_partition_validity() {
        let partition = SetPartition::new(vec![vec![0, 2], vec![1, 3]], 4).unwrap();
        assert!(partition.is_valid());

        // Invalid - missing element
        let invalid1 = SetPartition {
            blocks: vec![vec![0, 2], vec![1]],
            n: 4,
        };
        assert!(!invalid1.is_valid());

        // Invalid - duplicate element
        let invalid2 = SetPartition {
            blocks: vec![vec![0, 1], vec![1, 2]],
            n: 3,
        };
        assert!(!invalid2.is_valid());

        // Invalid - empty block
        let invalid3 = SetPartition {
            blocks: vec![vec![0, 1], vec![]],
            n: 2,
        };
        assert!(!invalid3.is_valid());
    }

    #[test]
    fn test_empty_partition() {
        let empty = SetPartition::from_rgs(&[]).unwrap();
        assert_eq!(empty.size(), 0);
        assert_eq!(empty.num_blocks(), 0);
        assert_eq!(empty.blocks().len(), 0);
    }

    #[test]
    fn test_single_element_partition() {
        let single = SetPartition::from_rgs(&[0]).unwrap();
        assert_eq!(single.size(), 1);
        assert_eq!(single.num_blocks(), 1);
        assert_eq!(single.blocks(), &[vec![0]]);
    }
}
