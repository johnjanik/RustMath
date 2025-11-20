//! Restricted growth strings and their bijection with set partitions
//!
//! A restricted growth string (RG string) is a sequence a[0], a[1], ..., a[n-1] where:
//! - a[0] = 0
//! - For each i > 0: a[i] <= 1 + max(a[0], ..., a[i-1])
//!
//! RG strings are in bijection with set partitions. The number of RG strings of length n
//! equals the Bell number B(n).

use crate::set_partition::SetPartition;

/// A restricted growth string
///
/// RG strings are sequences where each element is at most one greater than the maximum
/// of all previous elements. They provide a canonical representation of set partitions.
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::RestrictedGrowth;
///
/// // Valid RG strings
/// let rg1 = RestrictedGrowth::new(vec![0]).unwrap();
/// let rg2 = RestrictedGrowth::new(vec![0, 1, 0, 2, 1]).unwrap();
///
/// // Invalid - doesn't start with 0
/// assert!(RestrictedGrowth::new(vec![1, 2, 3]).is_none());
///
/// // Invalid - jumps by more than 1
/// assert!(RestrictedGrowth::new(vec![0, 2]).is_none());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RestrictedGrowth {
    /// The sequence of values in the RG string
    sequence: Vec<usize>,
}

impl RestrictedGrowth {
    /// Create a new restricted growth string
    ///
    /// Returns `None` if the sequence is not a valid RG string.
    ///
    /// # Requirements
    /// - Non-empty sequence
    /// - First element must be 0
    /// - Each element at position i must be <= 1 + max(sequence[0..i])
    pub fn new(sequence: Vec<usize>) -> Option<Self> {
        if sequence.is_empty() {
            return None;
        }

        if sequence[0] != 0 {
            return None;
        }

        let mut max_so_far = 0;
        for (i, &value) in sequence.iter().enumerate().skip(1) {
            if value > max_so_far + 1 {
                return None;
            }
            max_so_far = max_so_far.max(value);
        }

        Some(RestrictedGrowth { sequence })
    }

    /// Get the length of the RG string
    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    /// Check if the RG string is empty (it never should be if constructed properly)
    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }

    /// Get the sequence as a slice
    pub fn as_slice(&self) -> &[usize] {
        &self.sequence
    }

    /// Get the number of distinct values (blocks) in the RG string
    pub fn num_blocks(&self) -> usize {
        if self.sequence.is_empty() {
            return 0;
        }
        1 + *self.sequence.iter().max().unwrap()
    }

    /// Convert this RG string to a set partition
    ///
    /// The bijection maps element i to the block numbered by sequence[i].
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_combinatorics::RestrictedGrowth;
    ///
    /// let rg = RestrictedGrowth::new(vec![0, 1, 0, 2, 1]).unwrap();
    /// let partition = rg.to_set_partition();
    ///
    /// // This represents the partition {{0, 2}, {1, 4}, {3}}
    /// assert_eq!(partition.num_blocks(), 3);
    /// ```
    pub fn to_set_partition(&self) -> SetPartition {
        let n = self.len();
        let num_blocks = self.num_blocks();

        // Create blocks
        let mut blocks: Vec<Vec<usize>> = vec![Vec::new(); num_blocks];

        for (i, &block_num) in self.sequence.iter().enumerate() {
            blocks[block_num].push(i);
        }

        SetPartition::new(blocks, n).expect("RG string should produce valid partition")
    }

    /// Create an RG string from a set partition
    ///
    /// The bijection assigns each element the number of its block, where blocks
    /// are numbered in the order their first element appears.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_combinatorics::{RestrictedGrowth, SetPartition};
    ///
    /// let partition = SetPartition::new(vec![vec![0, 2], vec![1, 4], vec![3]], 5).unwrap();
    /// let rg = RestrictedGrowth::from_set_partition(&partition);
    ///
    /// assert_eq!(rg.as_slice(), &[0, 1, 0, 2, 1]);
    /// ```
    pub fn from_set_partition(partition: &SetPartition) -> Self {
        let n = partition.size();
        let mut sequence = vec![0; n];

        // Assign block numbers based on the order of first appearance
        for (block_num, block) in partition.blocks().iter().enumerate() {
            for &elem in block {
                sequence[elem] = block_num;
            }
        }

        RestrictedGrowth { sequence }
    }

    /// Get the successor of this RG string in lexicographic order
    ///
    /// Returns `None` if this is the last RG string of its length.
    pub fn next_lexicographic(&self) -> Option<Self> {
        let mut new_seq = self.sequence.clone();
        let n = new_seq.len();

        // Find the rightmost position that can be incremented
        // Note: we start at n-1 and go down to 1 (not 0, since position 0 must be 0)
        for i in (1..n).rev() {
            let max_before_i = *new_seq[0..i].iter().max().unwrap();

            if new_seq[i] < max_before_i + 1 {
                // Can increment this position
                new_seq[i] += 1;
                // Reset all positions after i to 0
                for j in (i + 1)..n {
                    new_seq[j] = 0;
                }
                return Some(RestrictedGrowth { sequence: new_seq });
            }
        }

        // No position can be incremented - we're at the last RG string
        None
    }
}

/// Generate all restricted growth strings of length n
///
/// The number of RG strings of length n equals the Bell number B(n).
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::restricted_growth_strings;
///
/// let rgs = restricted_growth_strings(3);
/// assert_eq!(rgs.len(), 5); // Bell(3) = 5
/// ```
pub fn restricted_growth_strings(n: usize) -> Vec<RestrictedGrowth> {
    if n == 0 {
        return vec![];
    }

    let mut result = Vec::new();
    let mut current = vec![0; n];
    generate_rg_strings(&mut current, 0, 0, &mut result);
    result
}

fn generate_rg_strings(
    current: &mut Vec<usize>,
    position: usize,
    max_so_far: usize,
    result: &mut Vec<RestrictedGrowth>,
) {
    let n = current.len();

    if position == n {
        result.push(RestrictedGrowth {
            sequence: current.clone(),
        });
        return;
    }

    // Position 0 must always be 0
    if position == 0 {
        current[0] = 0;
        generate_rg_strings(current, 1, 0, result);
        return;
    }

    // Try all valid values at this position
    for value in 0..=max_so_far + 1 {
        current[position] = value;
        let new_max = max_so_far.max(value);
        generate_rg_strings(current, position + 1, new_max, result);
    }
}

/// Iterator over restricted growth strings of a given length
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::RestrictedGrowthIterator;
///
/// let mut iter = RestrictedGrowthIterator::new(3);
/// let count = iter.count();
/// assert_eq!(count, 5); // Bell(3) = 5
/// ```
pub struct RestrictedGrowthIterator {
    current: Option<RestrictedGrowth>,
}

impl RestrictedGrowthIterator {
    /// Create a new iterator over RG strings of length n
    pub fn new(n: usize) -> Self {
        if n == 0 {
            RestrictedGrowthIterator { current: None }
        } else {
            RestrictedGrowthIterator {
                current: Some(RestrictedGrowth {
                    sequence: vec![0; n],
                }),
            }
        }
    }
}

impl Iterator for RestrictedGrowthIterator {
    type Item = RestrictedGrowth;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current.clone()?;
        self.current = current.next_lexicographic();
        Some(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rg_string_validation() {
        // Valid RG strings
        assert!(RestrictedGrowth::new(vec![0]).is_some());
        assert!(RestrictedGrowth::new(vec![0, 0]).is_some());
        assert!(RestrictedGrowth::new(vec![0, 1]).is_some());
        assert!(RestrictedGrowth::new(vec![0, 0, 0]).is_some());
        assert!(RestrictedGrowth::new(vec![0, 0, 1]).is_some());
        assert!(RestrictedGrowth::new(vec![0, 1, 0]).is_some());
        assert!(RestrictedGrowth::new(vec![0, 1, 1]).is_some());
        assert!(RestrictedGrowth::new(vec![0, 1, 2]).is_some());
        assert!(RestrictedGrowth::new(vec![0, 1, 0, 2, 1]).is_some());

        // Invalid - empty
        assert!(RestrictedGrowth::new(vec![]).is_none());

        // Invalid - doesn't start with 0
        assert!(RestrictedGrowth::new(vec![1]).is_none());
        assert!(RestrictedGrowth::new(vec![1, 0]).is_none());

        // Invalid - jumps by more than 1
        assert!(RestrictedGrowth::new(vec![0, 2]).is_none());
        assert!(RestrictedGrowth::new(vec![0, 1, 3]).is_none());
    }

    #[test]
    fn test_num_blocks() {
        assert_eq!(RestrictedGrowth::new(vec![0]).unwrap().num_blocks(), 1);
        assert_eq!(RestrictedGrowth::new(vec![0, 1]).unwrap().num_blocks(), 2);
        assert_eq!(RestrictedGrowth::new(vec![0, 0, 0]).unwrap().num_blocks(), 1);
        assert_eq!(
            RestrictedGrowth::new(vec![0, 1, 2]).unwrap().num_blocks(),
            3
        );
        assert_eq!(
            RestrictedGrowth::new(vec![0, 1, 0, 2, 1])
                .unwrap()
                .num_blocks(),
            3
        );
    }

    #[test]
    fn test_to_set_partition() {
        // [0] -> {{0}}
        let rg1 = RestrictedGrowth::new(vec![0]).unwrap();
        let p1 = rg1.to_set_partition();
        assert_eq!(p1.num_blocks(), 1);
        assert_eq!(p1.blocks(), &[vec![0]]);

        // [0, 1] -> {{0}, {1}}
        let rg2 = RestrictedGrowth::new(vec![0, 1]).unwrap();
        let p2 = rg2.to_set_partition();
        assert_eq!(p2.num_blocks(), 2);
        assert_eq!(p2.blocks(), &[vec![0], vec![1]]);

        // [0, 0] -> {{0, 1}}
        let rg3 = RestrictedGrowth::new(vec![0, 0]).unwrap();
        let p3 = rg3.to_set_partition();
        assert_eq!(p3.num_blocks(), 1);
        assert_eq!(p3.blocks(), &[vec![0, 1]]);

        // [0, 1, 0, 2, 1] -> {{0, 2}, {1, 4}, {3}}
        let rg4 = RestrictedGrowth::new(vec![0, 1, 0, 2, 1]).unwrap();
        let p4 = rg4.to_set_partition();
        assert_eq!(p4.num_blocks(), 3);
        assert_eq!(p4.blocks(), &[vec![0, 2], vec![1, 4], vec![3]]);
    }

    #[test]
    fn test_from_set_partition() {
        // {{0}} -> [0]
        let p1 = SetPartition::new(vec![vec![0]], 1).unwrap();
        let rg1 = RestrictedGrowth::from_set_partition(&p1);
        assert_eq!(rg1.as_slice(), &[0]);

        // {{0}, {1}} -> [0, 1]
        let p2 = SetPartition::new(vec![vec![0], vec![1]], 2).unwrap();
        let rg2 = RestrictedGrowth::from_set_partition(&p2);
        assert_eq!(rg2.as_slice(), &[0, 1]);

        // {{0, 1}} -> [0, 0]
        let p3 = SetPartition::new(vec![vec![0, 1]], 2).unwrap();
        let rg3 = RestrictedGrowth::from_set_partition(&p3);
        assert_eq!(rg3.as_slice(), &[0, 0]);

        // {{0, 2}, {1, 4}, {3}} -> [0, 1, 0, 2, 1]
        let p4 = SetPartition::new(vec![vec![0, 2], vec![1, 4], vec![3]], 5).unwrap();
        let rg4 = RestrictedGrowth::from_set_partition(&p4);
        assert_eq!(rg4.as_slice(), &[0, 1, 0, 2, 1]);
    }

    #[test]
    fn test_bijection_roundtrip() {
        // Test that RG -> Partition -> RG is identity
        let rg_strings = vec![
            vec![0],
            vec![0, 0],
            vec![0, 1],
            vec![0, 0, 0],
            vec![0, 0, 1],
            vec![0, 1, 0],
            vec![0, 1, 1],
            vec![0, 1, 2],
            vec![0, 1, 0, 2, 1],
        ];

        for seq in rg_strings {
            let rg1 = RestrictedGrowth::new(seq.clone()).unwrap();
            let partition = rg1.to_set_partition();
            let rg2 = RestrictedGrowth::from_set_partition(&partition);
            assert_eq!(rg1, rg2, "Roundtrip failed for {:?}", seq);
        }
    }

    #[test]
    fn test_partition_to_rg_to_partition() {
        // Test that Partition -> RG -> Partition preserves structure
        let partitions = vec![
            vec![vec![0]],
            vec![vec![0, 1]],
            vec![vec![0], vec![1]],
            vec![vec![0, 1, 2]],
            vec![vec![0, 2], vec![1]],
            vec![vec![0], vec![1, 2]],
            vec![vec![0], vec![1], vec![2]],
        ];

        for (i, blocks) in partitions.iter().enumerate() {
            let n = blocks.iter().map(|b| b.len()).sum();
            let p1 = SetPartition::new(blocks.clone(), n).unwrap();
            let rg = RestrictedGrowth::from_set_partition(&p1);
            let p2 = rg.to_set_partition();

            // Compare block structure (may be in different order)
            assert_eq!(
                p1.num_blocks(),
                p2.num_blocks(),
                "Block count mismatch for partition {}",
                i
            );
            assert_eq!(p1.size(), p2.size(), "Size mismatch for partition {}", i);

            // Check that elements are in the same blocks
            for elem in 0..n {
                let block1 = p1
                    .blocks()
                    .iter()
                    .position(|b| b.contains(&elem))
                    .unwrap();
                let block2 = p2
                    .blocks()
                    .iter()
                    .position(|b| b.contains(&elem))
                    .unwrap();

                // Elements in the same block in p1 should be in the same block in p2
                for other_elem in 0..n {
                    let other_block1 = p1
                        .blocks()
                        .iter()
                        .position(|b| b.contains(&other_elem))
                        .unwrap();
                    let other_block2 = p2
                        .blocks()
                        .iter()
                        .position(|b| b.contains(&other_elem))
                        .unwrap();

                    assert_eq!(
                        block1 == other_block1,
                        block2 == other_block2,
                        "Partition structure not preserved for partition {}, elements {} and {}",
                        i,
                        elem,
                        other_elem
                    );
                }
            }
        }
    }

    #[test]
    fn test_generate_rg_strings() {
        // Number of RG strings should equal Bell numbers
        assert_eq!(restricted_growth_strings(1).len(), 1); // Bell(1) = 1
        assert_eq!(restricted_growth_strings(2).len(), 2); // Bell(2) = 2
        assert_eq!(restricted_growth_strings(3).len(), 5); // Bell(3) = 5
        assert_eq!(restricted_growth_strings(4).len(), 15); // Bell(4) = 15
        assert_eq!(restricted_growth_strings(5).len(), 52); // Bell(5) = 52

        // All generated RG strings should be valid
        for n in 1..=5 {
            let rgs = restricted_growth_strings(n);
            for rg in &rgs {
                assert_eq!(rg.len(), n);
                // Re-validate by creating from the sequence
                assert!(RestrictedGrowth::new(rg.as_slice().to_vec()).is_some());
            }
        }
    }

    #[test]
    fn test_next_lexicographic() {
        // Start with [0, 0, 0]
        let rg = RestrictedGrowth::new(vec![0, 0, 0]).unwrap();

        // Sequence should be: [0,0,0] -> [0,0,1] -> [0,1,0] -> [0,1,1] -> [0,1,2] -> None
        let next1 = rg.next_lexicographic().unwrap();
        assert_eq!(next1.as_slice(), &[0, 0, 1]);

        let next2 = next1.next_lexicographic().unwrap();
        assert_eq!(next2.as_slice(), &[0, 1, 0]);

        let next3 = next2.next_lexicographic().unwrap();
        assert_eq!(next3.as_slice(), &[0, 1, 1]);

        let next4 = next3.next_lexicographic().unwrap();
        assert_eq!(next4.as_slice(), &[0, 1, 2]);

        let next5 = next4.next_lexicographic();
        assert!(next5.is_none());
    }

    #[test]
    fn test_iterator() {
        let rgs: Vec<_> = RestrictedGrowthIterator::new(3).collect();

        assert_eq!(rgs.len(), 5); // Bell(3) = 5

        // Check the sequences
        assert_eq!(rgs[0].as_slice(), &[0, 0, 0]);
        assert_eq!(rgs[1].as_slice(), &[0, 0, 1]);
        assert_eq!(rgs[2].as_slice(), &[0, 1, 0]);
        assert_eq!(rgs[3].as_slice(), &[0, 1, 1]);
        assert_eq!(rgs[4].as_slice(), &[0, 1, 2]);

        // Test empty iterator
        let empty_rgs: Vec<_> = RestrictedGrowthIterator::new(0).collect();
        assert_eq!(empty_rgs.len(), 0);
    }

    #[test]
    fn test_rg_specific_sequences() {
        // Test specific sequences from mathematical literature

        // Length 4: should generate 15 RG strings
        let rgs4 = restricted_growth_strings(4);
        assert_eq!(rgs4.len(), 15);

        // Check some specific expected sequences
        let expected_4 = vec![
            vec![0, 0, 0, 0],
            vec![0, 0, 0, 1],
            vec![0, 0, 1, 0],
            vec![0, 0, 1, 1],
            vec![0, 0, 1, 2],
            vec![0, 1, 0, 0],
            vec![0, 1, 0, 1],
            vec![0, 1, 0, 2],
            vec![0, 1, 1, 0],
            vec![0, 1, 1, 1],
            vec![0, 1, 1, 2],
            vec![0, 1, 2, 0],
            vec![0, 1, 2, 1],
            vec![0, 1, 2, 2],
            vec![0, 1, 2, 3],
        ];

        let actual_sequences: Vec<Vec<usize>> =
            rgs4.iter().map(|rg| rg.as_slice().to_vec()).collect();

        for expected_seq in &expected_4 {
            assert!(
                actual_sequences.contains(expected_seq),
                "Missing expected sequence {:?}",
                expected_seq
            );
        }
    }

    #[test]
    fn test_correspondence_with_set_partitions() {
        // Verify that the number of RG strings matches the number of set partitions
        use crate::set_partition::set_partitions;

        for n in 1..=5 {
            let rgs = restricted_growth_strings(n);
            let partitions = set_partitions(n);
            assert_eq!(
                rgs.len(),
                partitions.len(),
                "Mismatch for n={}: {} RG strings vs {} partitions",
                n,
                rgs.len(),
                partitions.len()
            );
        }
    }
}
