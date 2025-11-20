//! Set partitions
//!
//! A set partition is a way of partitioning a set into non-empty disjoint subsets.
//! The number of set partitions of n elements is the Bell number B(n).
//!
//! Set partitions form a lattice under the refinement partial order:
//! - Partition P refines partition Q (P ≤ Q) if every block of P is contained in some block of Q
//! - The meet (∧) is the finest common coarsening
//! - The join (∨) is the coarsest common refinement

use std::cmp::Ordering;
use std::collections::HashMap;

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

    /// Create the finest partition (all singletons)
    pub fn finest(n: usize) -> Self {
        let blocks: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        SetPartition { blocks, n }
    }

    /// Create the coarsest partition (one block containing all elements)
    pub fn coarsest(n: usize) -> Self {
        let blocks = if n > 0 {
            vec![(0..n).collect()]
        } else {
            vec![]
        };
        SetPartition { blocks, n }
    }

    /// Check if this partition refines another partition (self ≤ other)
    ///
    /// Partition P refines partition Q if every block of P is contained in some block of Q.
    /// In the refinement order, P ≤ Q means P is finer (has smaller blocks) than Q.
    pub fn refines(&self, other: &SetPartition) -> bool {
        if self.n != other.n {
            return false;
        }

        // For each block in self, check if it's contained in some block of other
        for self_block in &self.blocks {
            let mut found = false;
            for other_block in &other.blocks {
                // Check if all elements of self_block are in other_block
                if self_block.iter().all(|elem| other_block.contains(elem)) {
                    found = true;
                    break;
                }
            }
            if !found {
                return false;
            }
        }

        true
    }

    /// Check if this partition is coarser than another (self ≥ other)
    pub fn is_coarser_than(&self, other: &SetPartition) -> bool {
        other.refines(self)
    }

    /// Compute the meet (greatest lower bound) of two partitions
    ///
    /// The meet is the finest partition that both partitions refine.
    /// It's computed by taking all pairwise intersections of blocks.
    pub fn meet(&self, other: &SetPartition) -> Option<Self> {
        if self.n != other.n {
            return None;
        }

        let mut blocks: Vec<Vec<usize>> = Vec::new();

        // For each pair of blocks, compute their intersection
        for self_block in &self.blocks {
            for other_block in &other.blocks {
                let intersection: Vec<usize> = self_block
                    .iter()
                    .filter(|elem| other_block.contains(elem))
                    .copied()
                    .collect();

                if !intersection.is_empty() {
                    blocks.push(intersection);
                }
            }
        }

        SetPartition::new(blocks, self.n)
    }

    /// Compute the join (least upper bound) of two partitions
    ///
    /// The join is the coarsest partition that refines both partitions.
    /// It's computed using a union-find algorithm to merge blocks that share elements
    /// when considering both partitions together.
    pub fn join(&self, other: &SetPartition) -> Option<Self> {
        if self.n != other.n {
            return None;
        }

        // Use union-find to merge blocks
        let mut parent: Vec<usize> = (0..self.n).collect();

        fn find(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }

        fn union(parent: &mut Vec<usize>, x: usize, y: usize) {
            let root_x = find(parent, x);
            let root_y = find(parent, y);
            if root_x != root_y {
                parent[root_x] = root_y;
            }
        }

        // Union elements in the same block for self
        for block in &self.blocks {
            if !block.is_empty() {
                let first = block[0];
                for &elem in &block[1..] {
                    union(&mut parent, first, elem);
                }
            }
        }

        // Union elements in the same block for other
        for block in &other.blocks {
            if !block.is_empty() {
                let first = block[0];
                for &elem in &block[1..] {
                    union(&mut parent, first, elem);
                }
            }
        }

        // Group elements by their root
        let mut blocks_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..self.n {
            let root = find(&mut parent, i);
            blocks_map.entry(root).or_insert_with(Vec::new).push(i);
        }

        let blocks: Vec<Vec<usize>> = blocks_map.into_values().collect();

        SetPartition::new(blocks, self.n)
    }

    /// Find which block contains a given element
    pub fn find_block(&self, elem: usize) -> Option<usize> {
        if elem >= self.n {
            return None;
        }

        for (i, block) in self.blocks.iter().enumerate() {
            if block.contains(&elem) {
                return Some(i);
            }
        }

        None
    }

    /// Check if two elements are in the same block
    pub fn in_same_block(&self, a: usize, b: usize) -> bool {
        if a >= self.n || b >= self.n {
            return false;
        }

        for block in &self.blocks {
            if block.contains(&a) && block.contains(&b) {
                return true;
            }
        }

        false
    }
}

impl PartialOrd for SetPartition {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.n != other.n {
            return None;
        }

        if self == other {
            return Some(Ordering::Equal);
        }

        let self_refines_other = self.refines(other);
        let other_refines_self = other.refines(self);

        match (self_refines_other, other_refines_self) {
            (true, false) => Some(Ordering::Less),    // self is finer
            (false, true) => Some(Ordering::Greater), // self is coarser
            (false, false) => None,                   // incomparable
            (true, true) => Some(Ordering::Equal),    // should not happen if self != other
        }
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
    fn test_finest_coarsest() {
        // Finest partition has all singletons
        let finest = SetPartition::finest(4);
        assert_eq!(finest.num_blocks(), 4);
        assert_eq!(finest.size(), 4);
        for i in 0..4 {
            assert_eq!(finest.blocks()[i], vec![i]);
        }

        // Coarsest partition has one block
        let coarsest = SetPartition::coarsest(4);
        assert_eq!(coarsest.num_blocks(), 1);
        assert_eq!(coarsest.size(), 4);
        assert_eq!(coarsest.blocks()[0], vec![0, 1, 2, 3]);

        // Edge case: n=0
        let finest0 = SetPartition::finest(0);
        assert_eq!(finest0.num_blocks(), 0);
        let coarsest0 = SetPartition::coarsest(0);
        assert_eq!(coarsest0.num_blocks(), 0);
    }

    #[test]
    fn test_refines() {
        // {{0}, {1}, {2}} refines {{0, 1}, {2}}
        let fine = SetPartition::new(vec![vec![0], vec![1], vec![2]], 3).unwrap();
        let coarse = SetPartition::new(vec![vec![0, 1], vec![2]], 3).unwrap();

        assert!(fine.refines(&coarse));
        assert!(!coarse.refines(&fine));

        // Every partition refines itself
        assert!(fine.refines(&fine));
        assert!(coarse.refines(&coarse));

        // Every partition refines the coarsest partition
        let coarsest = SetPartition::coarsest(3);
        assert!(fine.refines(&coarsest));
        assert!(coarse.refines(&coarsest));

        // Finest partition refines every partition
        let finest = SetPartition::finest(3);
        assert!(finest.refines(&fine));
        assert!(finest.refines(&coarse));
        assert!(finest.refines(&coarsest));
    }

    #[test]
    fn test_is_coarser_than() {
        let fine = SetPartition::new(vec![vec![0], vec![1], vec![2]], 3).unwrap();
        let coarse = SetPartition::new(vec![vec![0, 1], vec![2]], 3).unwrap();

        assert!(coarse.is_coarser_than(&fine));
        assert!(!fine.is_coarser_than(&coarse));
    }

    #[test]
    fn test_partial_ord() {
        let finest = SetPartition::finest(3);
        let middle = SetPartition::new(vec![vec![0, 1], vec![2]], 3).unwrap();
        let coarsest = SetPartition::coarsest(3);

        // Test ordering
        assert!(finest < middle);
        assert!(middle < coarsest);
        assert!(finest < coarsest);

        // Test equality
        let middle2 = SetPartition::new(vec![vec![0, 1], vec![2]], 3).unwrap();
        assert_eq!(middle.partial_cmp(&middle2), Some(Ordering::Equal));

        // Test incomparable partitions
        let p1 = SetPartition::new(vec![vec![0, 1], vec![2]], 3).unwrap();
        let p2 = SetPartition::new(vec![vec![0], vec![1, 2]], 3).unwrap();
        assert_eq!(p1.partial_cmp(&p2), None);
    }

    #[test]
    fn test_meet() {
        // Meet of {{0, 1}, {2}} and {{0}, {1, 2}} should be {{0}, {1}, {2}}
        let p1 = SetPartition::new(vec![vec![0, 1], vec![2]], 3).unwrap();
        let p2 = SetPartition::new(vec![vec![0], vec![1, 2]], 3).unwrap();
        let meet = p1.meet(&p2).unwrap();

        // The meet should be the finest partition
        let finest = SetPartition::finest(3);
        assert!(meet.refines(&p1));
        assert!(meet.refines(&p2));

        // Check that the meet has the right structure
        assert_eq!(meet.num_blocks(), 3);

        // Meet with self is self
        let meet_self = p1.meet(&p1).unwrap();
        assert_eq!(meet_self, p1);

        // Meet is commutative
        let meet12 = p1.meet(&p2).unwrap();
        let meet21 = p2.meet(&p1).unwrap();
        // Note: blocks might be in different order, so we check refinement equivalence
        assert!(meet12.refines(&meet21));
        assert!(meet21.refines(&meet12));
    }

    #[test]
    fn test_join() {
        // Join of {{0, 1}, {2}} and {{0}, {1, 2}} should be {{0, 1, 2}}
        let p1 = SetPartition::new(vec![vec![0, 1], vec![2]], 3).unwrap();
        let p2 = SetPartition::new(vec![vec![0], vec![1, 2]], 3).unwrap();
        let join = p1.join(&p2).unwrap();

        // The join should be the coarsest partition
        let coarsest = SetPartition::coarsest(3);
        assert!(p1.refines(&join));
        assert!(p2.refines(&join));
        assert_eq!(join.num_blocks(), 1);

        // Join with self is self
        let join_self = p1.join(&p1).unwrap();
        assert_eq!(join_self, p1);

        // Join is commutative
        let join12 = p1.join(&p2).unwrap();
        let join21 = p2.join(&p1).unwrap();
        assert!(join12.refines(&join21));
        assert!(join21.refines(&join12));
    }

    #[test]
    fn test_lattice_properties() {
        // Test that meet and join satisfy lattice properties
        let p1 = SetPartition::new(vec![vec![0, 1], vec![2, 3]], 4).unwrap();
        let p2 = SetPartition::new(vec![vec![0], vec![1, 2], vec![3]], 4).unwrap();
        let p3 = SetPartition::new(vec![vec![0, 1, 2], vec![3]], 4).unwrap();

        // Idempotent: p ∨ p = p and p ∧ p = p
        let p1_join_p1 = p1.join(&p1).unwrap();
        assert!(p1_join_p1.refines(&p1) && p1.refines(&p1_join_p1));
        assert_eq!(p1.meet(&p1).unwrap(), p1);

        // Commutative: p ∨ q = q ∨ p and p ∧ q = q ∧ p
        let join12 = p1.join(&p2).unwrap();
        let join21 = p2.join(&p1).unwrap();
        assert!(join12.refines(&join21) && join21.refines(&join12));

        let meet12 = p1.meet(&p2).unwrap();
        let meet21 = p2.meet(&p1).unwrap();
        assert!(meet12.refines(&meet21) && meet21.refines(&meet12));

        // Associative: (p ∨ q) ∨ r = p ∨ (q ∨ r)
        let join_assoc1 = p1.join(&p2).unwrap().join(&p3).unwrap();
        let join_assoc2 = p1.join(&p2.join(&p3).unwrap()).unwrap();
        assert!(join_assoc1.refines(&join_assoc2) && join_assoc2.refines(&join_assoc1));

        // Absorption: p ∨ (p ∧ q) = p and p ∧ (p ∨ q) = p
        let meet_pq = p1.meet(&p2).unwrap();
        let join_p_meet = p1.join(&meet_pq).unwrap();
        assert_eq!(join_p_meet, p1);

        let join_pq = p1.join(&p2).unwrap();
        let meet_p_join = p1.meet(&join_pq).unwrap();
        assert_eq!(meet_p_join, p1);
    }

    #[test]
    fn test_find_block() {
        let partition = SetPartition::new(vec![vec![0, 2], vec![1, 3]], 4).unwrap();

        assert_eq!(partition.find_block(0), Some(0));
        assert_eq!(partition.find_block(2), Some(0));
        assert_eq!(partition.find_block(1), Some(1));
        assert_eq!(partition.find_block(3), Some(1));
        assert_eq!(partition.find_block(4), None);
    }

    #[test]
    fn test_in_same_block() {
        let partition = SetPartition::new(vec![vec![0, 2], vec![1, 3]], 4).unwrap();

        assert!(partition.in_same_block(0, 2));
        assert!(partition.in_same_block(2, 0));
        assert!(partition.in_same_block(1, 3));
        assert!(partition.in_same_block(3, 1));

        assert!(!partition.in_same_block(0, 1));
        assert!(!partition.in_same_block(0, 3));
        assert!(!partition.in_same_block(2, 3));

        // Out of bounds
        assert!(!partition.in_same_block(0, 4));
        assert!(!partition.in_same_block(4, 0));
    }

    #[test]
    fn test_meet_join_edge_cases() {
        // Test with finest and coarsest
        let finest = SetPartition::finest(3);
        let coarsest = SetPartition::coarsest(3);
        let middle = SetPartition::new(vec![vec![0, 1], vec![2]], 3).unwrap();

        // finest ∧ anything = finest
        assert_eq!(finest.meet(&middle).unwrap(), finest);
        assert_eq!(finest.meet(&coarsest).unwrap(), finest);

        // coarsest ∨ anything = coarsest
        assert_eq!(coarsest.join(&middle).unwrap(), coarsest);
        assert_eq!(coarsest.join(&finest).unwrap(), coarsest);

        // finest ∨ coarsest = coarsest
        assert_eq!(finest.join(&coarsest).unwrap(), coarsest);

        // finest ∧ coarsest = finest
        assert_eq!(finest.meet(&coarsest).unwrap(), finest);
    }

    #[test]
    fn test_incomparable_partitions() {
        // Create two incomparable partitions
        let p1 = SetPartition::new(vec![vec![0, 1], vec![2, 3]], 4).unwrap();
        let p2 = SetPartition::new(vec![vec![0, 2], vec![1, 3]], 4).unwrap();

        // They should be incomparable
        assert!(!p1.refines(&p2));
        assert!(!p2.refines(&p1));
        assert_eq!(p1.partial_cmp(&p2), None);

        // But their meet and join should exist
        let meet = p1.meet(&p2).unwrap();
        let join = p1.join(&p2).unwrap();

        // Meet should refine both
        assert!(meet.refines(&p1));
        assert!(meet.refines(&p2));

        // Both should refine join
        assert!(p1.refines(&join));
        assert!(p2.refines(&join));

        // Meet should be finest, join should be coarsest
        assert_eq!(meet, SetPartition::finest(4));
        assert_eq!(join, SetPartition::coarsest(4));
    }
}
