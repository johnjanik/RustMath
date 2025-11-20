//! Set partitions
//!
//! A set partition is a way of partitioning a set into non-empty disjoint subsets.
//! The number of set partitions of n elements is the Bell number B(n).

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
}
