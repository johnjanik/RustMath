//! Set partitions
//!
//! A set partition is a way of partitioning a set into non-empty disjoint subsets.
//! The number of set partitions of n elements is the Bell number B(n).
//!
//! An ordered set partition (also called a composition of a set) is a set partition
//! where the order of the blocks matters. The number of ordered set partitions of n
//! elements is the Fubini number (or ordered Bell number).

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

/// An ordered set partition - a set partition where the order of blocks matters
///
/// Unlike regular set partitions, {{0}, {1, 2}} and {{1, 2}, {0}} are considered
/// different ordered set partitions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderedSetPartition {
    /// Each vector represents one block (subset) in the partition, in order
    blocks: Vec<Vec<usize>>,
    /// Total number of elements
    n: usize,
}

impl OrderedSetPartition {
    /// Create an ordered set partition from blocks
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

        Some(OrderedSetPartition { blocks, n })
    }

    /// Get the number of blocks
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Get the blocks in order
    pub fn blocks(&self) -> &[Vec<usize>] {
        &self.blocks
    }

    /// Get the size of the set being partitioned
    pub fn size(&self) -> usize {
        self.n
    }

    /// Convert to a regular (unordered) set partition
    pub fn to_set_partition(&self) -> SetPartition {
        // Note: This creates a SetPartition with blocks in the current order,
        // but SetPartition doesn't consider order when comparing
        SetPartition {
            blocks: self.blocks.clone(),
            n: self.n,
        }
    }
}

/// Generate all ordered set partitions of n elements (labeled 0..n)
///
/// The number of ordered set partitions is the Fubini number (ordered Bell number).
/// For n = 0, 1, 2, 3, 4, the counts are 1, 1, 3, 13, 75, ...
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::set_partition_ordered;
///
/// let partitions = set_partition_ordered(2);
/// assert_eq!(partitions.len(), 3); // {{0,1}}, {{0},{1}}, {{1},{0}}
/// ```
pub fn set_partition_ordered(n: usize) -> Vec<OrderedSetPartition> {
    if n == 0 {
        return vec![OrderedSetPartition {
            blocks: vec![],
            n: 0,
        }];
    }

    let mut result = Vec::new();
    let elements: Vec<usize> = (0..n).collect();
    let mut current_partition: Vec<Vec<usize>> = Vec::new();

    generate_ordered_set_partitions(&elements, 0, &mut current_partition, &mut result);

    result
}

fn generate_ordered_set_partitions(
    elements: &[usize],
    index: usize,
    current: &mut Vec<Vec<usize>>,
    result: &mut Vec<OrderedSetPartition>,
) {
    if index == elements.len() {
        result.push(OrderedSetPartition {
            blocks: current.clone(),
            n: elements.len(),
        });
        return;
    }

    let elem = elements[index];

    // Try adding element to each existing block (order preserved)
    for i in 0..current.len() {
        current[i].push(elem);
        generate_ordered_set_partitions(elements, index + 1, current, result);
        current[i].pop();
    }

    // Try creating a new block at each possible position
    // This is what makes it "ordered" - we can insert at any position
    for pos in 0..=current.len() {
        current.insert(pos, vec![elem]);
        generate_ordered_set_partitions(elements, index + 1, current, result);
        current.remove(pos);
    }
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
    fn test_ordered_set_partition_basic() {
        // Test creating a valid ordered set partition
        let osp = OrderedSetPartition::new(vec![vec![0, 1], vec![2]], 3);
        assert!(osp.is_some());
        let osp = osp.unwrap();
        assert_eq!(osp.num_blocks(), 2);
        assert_eq!(osp.size(), 3);
        assert_eq!(osp.blocks(), &[vec![0, 1], vec![2]]);
    }

    #[test]
    fn test_ordered_set_partition_validation() {
        // Empty block should fail
        let osp = OrderedSetPartition::new(vec![vec![0], vec![]], 2);
        assert!(osp.is_none());

        // Duplicate element should fail
        let osp = OrderedSetPartition::new(vec![vec![0, 1], vec![1, 2]], 3);
        assert!(osp.is_none());

        // Missing element should fail
        let osp = OrderedSetPartition::new(vec![vec![0], vec![2]], 3);
        assert!(osp.is_none());

        // Out of range element should fail
        let osp = OrderedSetPartition::new(vec![vec![0, 3]], 3);
        assert!(osp.is_none());

        // Valid partition
        let osp = OrderedSetPartition::new(vec![vec![2], vec![0, 1]], 3);
        assert!(osp.is_some());
    }

    #[test]
    fn test_ordered_set_partition_count() {
        // Number of ordered set partitions should match Fubini numbers (ordered Bell numbers)
        // See OEIS A000670
        assert_eq!(set_partition_ordered(0).len(), 1); // Fubini(0) = 1
        assert_eq!(set_partition_ordered(1).len(), 1); // Fubini(1) = 1
        assert_eq!(set_partition_ordered(2).len(), 3); // Fubini(2) = 3
        assert_eq!(set_partition_ordered(3).len(), 13); // Fubini(3) = 13
        assert_eq!(set_partition_ordered(4).len(), 75); // Fubini(4) = 75
    }

    #[test]
    fn test_ordered_set_partition_examples() {
        // For n=1, only one partition: {{0}}
        let partitions = set_partition_ordered(1);
        assert_eq!(partitions.len(), 1);
        assert_eq!(partitions[0].blocks(), &[vec![0]]);

        // For n=2, three partitions:
        // {{0, 1}}, {{0}, {1}}, {{1}, {0}}
        let partitions = set_partition_ordered(2);
        assert_eq!(partitions.len(), 3);

        // Check that we have the expected partitions
        let mut has_both_together = false;
        let mut has_0_then_1 = false;
        let mut has_1_then_0 = false;

        for p in &partitions {
            match p.blocks() {
                [b] if b.len() == 2 => has_both_together = true,
                [b1, b2] if b1 == &[0] && b2 == &[1] => has_0_then_1 = true,
                [b1, b2] if b1 == &[1] && b2 == &[0] => has_1_then_0 = true,
                _ => {}
            }
        }

        assert!(has_both_together, "Should have [[0, 1]]");
        assert!(has_0_then_1, "Should have [[0], [1]]");
        assert!(has_1_then_0, "Should have [[1], [0]]");
    }

    #[test]
    fn test_ordered_vs_unordered() {
        // Ordered set partitions should be >= unordered set partitions
        for n in 0..=4 {
            let ordered_count = set_partition_ordered(n).len();
            let unordered_count = set_partitions(n).len();
            assert!(
                ordered_count >= unordered_count,
                "For n={}, ordered ({}) should be >= unordered ({})",
                n,
                ordered_count,
                unordered_count
            );
        }

        // For n=2: ordered=3, unordered=2
        // The difference is that {{0},{1}} and {{1},{0}} are different when ordered
        assert_eq!(set_partition_ordered(2).len(), 3);
        assert_eq!(set_partitions(2).len(), 2);
    }

    #[test]
    fn test_ordered_set_partition_all_valid() {
        // For n=3, verify all 13 partitions are valid and distinct
        let partitions = set_partition_ordered(3);
        assert_eq!(partitions.len(), 13);

        for (i, part) in partitions.iter().enumerate() {
            // Verify size
            assert_eq!(part.size(), 3, "Partition {} has wrong size", i);
            assert!(part.num_blocks() > 0, "Partition {} has no blocks", i);

            // Check that all elements 0,1,2 appear exactly once
            let mut seen = vec![false; 3];
            for block in part.blocks() {
                assert!(!block.is_empty(), "Partition {} has empty block", i);
                for &elem in block {
                    assert!(
                        elem < 3,
                        "Partition {} has out-of-range element {}",
                        i,
                        elem
                    );
                    assert!(!seen[elem], "Partition {} has duplicate element {}", i, elem);
                    seen[elem] = true;
                }
            }
            assert!(
                seen.iter().all(|&x| x),
                "Partition {} is missing elements",
                i
            );
        }

        // Verify all partitions are distinct
        for i in 0..partitions.len() {
            for j in (i + 1)..partitions.len() {
                assert_ne!(
                    partitions[i],
                    partitions[j],
                    "Partitions {} and {} are equal",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_ordered_set_partition_to_unordered() {
        let osp = OrderedSetPartition::new(vec![vec![1], vec![0, 2]], 3).unwrap();
        let sp = osp.to_set_partition();
        assert_eq!(sp.blocks(), &[vec![1], vec![0, 2]]);
        assert_eq!(sp.size(), 3);
    }
}
