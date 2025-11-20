//! Set systems and hypergraphs
//!
//! A set system (also called a hypergraph or set family) is a collection of subsets
//! of a ground set. This module provides data structures and algorithms for working
//! with set systems.

use std::collections::{HashMap, HashSet};

/// A set system - a collection of subsets of a ground set {0, 1, ..., n-1}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetSystem {
    /// The size of the ground set
    n: usize,
    /// The collection of sets (each represented as a sorted vector)
    sets: Vec<Vec<usize>>,
}

impl SetSystem {
    /// Create a new set system on a ground set of size n
    pub fn new(n: usize) -> Self {
        SetSystem {
            n,
            sets: Vec::new(),
        }
    }

    /// Create a set system from a vector of sets
    pub fn from_sets(n: usize, sets: Vec<Vec<usize>>) -> Option<Self> {
        // Validate that all elements are in range
        for set in &sets {
            for &elem in set {
                if elem >= n {
                    return None;
                }
            }
        }

        // Normalize sets (sort and remove duplicates)
        let mut normalized_sets = Vec::new();
        for set in sets {
            let mut s: Vec<_> = set.into_iter().collect::<HashSet<_>>().into_iter().collect();
            s.sort();
            normalized_sets.push(s);
        }

        Some(SetSystem {
            n,
            sets: normalized_sets,
        })
    }

    /// Get the size of the ground set
    pub fn ground_set_size(&self) -> usize {
        self.n
    }

    /// Get the number of sets in the system
    pub fn num_sets(&self) -> usize {
        self.sets.len()
    }

    /// Get the sets
    pub fn sets(&self) -> &[Vec<usize>] {
        &self.sets
    }

    /// Add a set to the system
    pub fn add_set(&mut self, mut set: Vec<usize>) -> bool {
        // Check validity
        if set.iter().any(|&x| x >= self.n) {
            return false;
        }

        // Normalize
        let set_hash: HashSet<_> = set.iter().copied().collect();
        set = set_hash.into_iter().collect();
        set.sort();

        // Check if already present
        if self.sets.contains(&set) {
            return false;
        }

        self.sets.push(set);
        true
    }

    /// Remove a set from the system
    pub fn remove_set(&mut self, set: &[usize]) -> bool {
        let mut sorted_set = set.to_vec();
        sorted_set.sort();

        if let Some(pos) = self.sets.iter().position(|s| s == &sorted_set) {
            self.sets.remove(pos);
            true
        } else {
            false
        }
    }

    /// Check if the system contains a given set
    pub fn contains(&self, set: &[usize]) -> bool {
        let mut sorted_set = set.to_vec();
        sorted_set.sort();
        self.sets.contains(&sorted_set)
    }

    /// Check if the system is empty
    pub fn is_empty(&self) -> bool {
        self.sets.is_empty()
    }

    /// Get the union of all sets
    pub fn union(&self) -> Vec<usize> {
        let mut result = HashSet::new();
        for set in &self.sets {
            result.extend(set.iter().copied());
        }
        let mut union: Vec<_> = result.into_iter().collect();
        union.sort();
        union
    }

    /// Get the intersection of all sets (returns None if system is empty)
    pub fn intersection(&self) -> Option<Vec<usize>> {
        if self.sets.is_empty() {
            return None;
        }

        let mut result: HashSet<_> = self.sets[0].iter().copied().collect();
        for set in &self.sets[1..] {
            let set_hash: HashSet<_> = set.iter().copied().collect();
            result.retain(|x| set_hash.contains(x));
        }

        let mut intersection: Vec<_> = result.into_iter().collect();
        intersection.sort();
        Some(intersection)
    }

    /// Check if the system is a chain (totally ordered by inclusion)
    pub fn is_chain(&self) -> bool {
        for i in 0..self.sets.len() {
            for j in i + 1..self.sets.len() {
                let set_i: HashSet<_> = self.sets[i].iter().copied().collect();
                let set_j: HashSet<_> = self.sets[j].iter().copied().collect();

                let i_subset_j = set_i.is_subset(&set_j);
                let j_subset_i = set_j.is_subset(&set_i);

                if !i_subset_j && !j_subset_i {
                    return false; // Neither is subset of the other
                }
            }
        }
        true
    }

    /// Check if the system is an antichain (no set is a subset of another)
    pub fn is_antichain(&self) -> bool {
        for i in 0..self.sets.len() {
            for j in i + 1..self.sets.len() {
                let set_i: HashSet<_> = self.sets[i].iter().copied().collect();
                let set_j: HashSet<_> = self.sets[j].iter().copied().collect();

                if set_i.is_subset(&set_j) || set_j.is_subset(&set_i) {
                    return false;
                }
            }
        }
        true
    }

    /// Check if the system is uniform (all sets have the same size)
    pub fn is_uniform(&self) -> bool {
        if self.sets.is_empty() {
            return true;
        }

        let size = self.sets[0].len();
        self.sets.iter().all(|s| s.len() == size)
    }

    /// Get the uniformity (common size if uniform, None otherwise)
    pub fn uniformity(&self) -> Option<usize> {
        if self.is_uniform() {
            self.sets.first().map(|s| s.len())
        } else {
            None
        }
    }

    /// Check if the system is a partition of the ground set
    pub fn is_partition(&self) -> bool {
        let mut covered = vec![false; self.n];
        let mut count = 0;

        for set in &self.sets {
            if set.is_empty() {
                return false; // Partitions don't have empty sets
            }

            for &elem in set {
                if covered[elem] {
                    return false; // Element appears twice
                }
                covered[elem] = true;
                count += 1;
            }
        }

        count == self.n // All elements covered exactly once
    }

    /// Check if the system is a cover (all elements appear in at least one set)
    pub fn is_cover(&self) -> bool {
        let union = self.union();
        union.len() == self.n && union.iter().enumerate().all(|(i, &x)| x == i)
    }

    /// Get the degree of an element (number of sets containing it)
    pub fn degree(&self, elem: usize) -> usize {
        if elem >= self.n {
            return 0;
        }

        self.sets.iter().filter(|s| s.contains(&elem)).count()
    }

    /// Get the degree sequence (degrees of all elements)
    pub fn degree_sequence(&self) -> Vec<usize> {
        (0..self.n).map(|i| self.degree(i)).collect()
    }

    /// Compute the dual set system (transpose)
    ///
    /// The dual system has sets corresponding to elements, where the i-th set
    /// contains all indices of sets in the original system that contain element i
    pub fn dual(&self) -> SetSystem {
        let mut dual_sets = vec![Vec::new(); self.n];

        for (set_idx, set) in self.sets.iter().enumerate() {
            for &elem in set {
                dual_sets[elem].push(set_idx);
            }
        }

        SetSystem {
            n: self.sets.len(),
            sets: dual_sets,
        }
    }

    /// Compute the shadow (downward closure)
    ///
    /// Returns a new set system containing all subsets of sets in this system
    pub fn shadow(&self) -> SetSystem {
        let mut shadow_sets = HashSet::new();

        for set in &self.sets {
            // Generate all subsets
            let n = set.len();
            for mask in 0..(1 << n) {
                let subset: Vec<usize> = set
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| mask & (1 << i) != 0)
                    .map(|(_, &x)| x)
                    .collect();

                if !subset.is_empty() {
                    shadow_sets.insert(subset);
                }
            }
        }

        let sets: Vec<_> = shadow_sets.into_iter().collect();
        SetSystem::from_sets(self.n, sets).unwrap()
    }

    /// Compute the shade (upward closure in the power set up to the ground set)
    ///
    /// Returns all supersets of sets in this system that are subsets of the ground set
    pub fn shade(&self) -> SetSystem {
        let mut shade_sets = HashSet::new();
        let ground_set: Vec<usize> = (0..self.n).collect();

        for set in &self.sets {
            // Generate all supersets that are subsets of the ground set
            let set_hash: HashSet<_> = set.iter().copied().collect();
            let remaining: Vec<_> = ground_set
                .iter()
                .copied()
                .filter(|x| !set_hash.contains(x))
                .collect();

            let n = remaining.len();
            for mask in 0..(1 << n) {
                let mut superset = set.clone();
                for (i, &elem) in remaining.iter().enumerate() {
                    if mask & (1 << i) != 0 {
                        superset.push(elem);
                    }
                }
                superset.sort();
                shade_sets.insert(superset);
            }
        }

        let sets: Vec<_> = shade_sets.into_iter().collect();
        SetSystem::from_sets(self.n, sets).unwrap()
    }

    /// Compute the Vapnik-Chervonenkis (VC) dimension
    ///
    /// The VC dimension is the size of the largest subset of the ground set
    /// that is shattered by the set system. A set S is shattered if for every
    /// subset T of S, there exists a set in the system whose intersection with S is T.
    pub fn vc_dimension(&self) -> usize {
        // For small ground sets, we can check all subsets
        if self.n > 20 {
            // Too large to enumerate, return 0 as placeholder
            return 0;
        }

        for size in (0..=self.n).rev() {
            // Check all subsets of given size
            if self.has_shattered_set_of_size(size) {
                return size;
            }
        }

        0
    }

    fn has_shattered_set_of_size(&self, size: usize) -> bool {
        // Generate all subsets of the ground set of given size
        let ground_set: Vec<usize> = (0..self.n).collect();
        Self::check_subsets_shattered(&ground_set, size, &self.sets)
    }

    fn check_subsets_shattered(ground_set: &[usize], k: usize, sets: &[Vec<usize>]) -> bool {
        if k > ground_set.len() {
            return false;
        }

        // Generate all k-subsets and check if any is shattered
        let mut current = Vec::new();
        Self::check_shattered_recursive(ground_set, k, 0, &mut current, sets)
    }

    fn check_shattered_recursive(
        ground_set: &[usize],
        k: usize,
        start: usize,
        current: &mut Vec<usize>,
        sets: &[Vec<usize>],
    ) -> bool {
        if current.len() == k {
            return Self::is_shattered(current, sets);
        }

        for i in start..ground_set.len() {
            current.push(ground_set[i]);
            if Self::check_shattered_recursive(ground_set, k, i + 1, current, sets) {
                return true;
            }
            current.pop();
        }

        false
    }

    fn is_shattered(subset: &[usize], sets: &[Vec<usize>]) -> bool {
        let subset_set: HashSet<_> = subset.iter().copied().collect();
        let n = subset.len();

        // Check all 2^n subsets of subset
        for mask in 0..(1 << n) {
            let target: HashSet<_> = subset
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .map(|(_, &x)| x)
                .collect();

            // Check if any set in the system intersects subset to give target
            let found = sets.iter().any(|set| {
                let set_hash: HashSet<_> = set.iter().copied().collect();
                let intersection: HashSet<_> =
                    set_hash.intersection(&subset_set).copied().collect();
                intersection == target
            });

            if !found {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_system_creation() {
        let sys = SetSystem::new(5);
        assert_eq!(sys.ground_set_size(), 5);
        assert_eq!(sys.num_sets(), 0);
        assert!(sys.is_empty());
    }

    #[test]
    fn test_add_remove_sets() {
        let mut sys = SetSystem::new(5);

        assert!(sys.add_set(vec![0, 1, 2]));
        assert_eq!(sys.num_sets(), 1);

        // Adding duplicate
        assert!(!sys.add_set(vec![0, 1, 2]));
        assert_eq!(sys.num_sets(), 1);

        // Adding with different order (should be same set)
        assert!(!sys.add_set(vec![2, 1, 0]));
        assert_eq!(sys.num_sets(), 1);

        assert!(sys.add_set(vec![3, 4]));
        assert_eq!(sys.num_sets(), 2);

        assert!(sys.remove_set(&[0, 1, 2]));
        assert_eq!(sys.num_sets(), 1);
    }

    #[test]
    fn test_union_intersection() {
        let sys = SetSystem::from_sets(5, vec![vec![0, 1, 2], vec![2, 3], vec![1, 3, 4]]).unwrap();

        let union = sys.union();
        assert_eq!(union, vec![0, 1, 2, 3, 4]);

        let intersection = sys.intersection().unwrap();
        assert!(intersection.is_empty()); // No common element

        let sys2 = SetSystem::from_sets(5, vec![vec![0, 1, 2], vec![1, 2, 3], vec![1, 2]]).unwrap();
        let intersection2 = sys2.intersection().unwrap();
        assert_eq!(intersection2, vec![1, 2]);
    }

    #[test]
    fn test_is_chain() {
        // {1} ⊂ {1,2} ⊂ {1,2,3} is a chain
        let chain = SetSystem::from_sets(5, vec![vec![1], vec![1, 2], vec![1, 2, 3]]).unwrap();
        assert!(chain.is_chain());

        // {1,2} and {2,3} is not a chain
        let not_chain = SetSystem::from_sets(5, vec![vec![1, 2], vec![2, 3]]).unwrap();
        assert!(!not_chain.is_chain());
    }

    #[test]
    fn test_is_antichain() {
        // {1,2} and {3,4} have no inclusion relation
        let antichain = SetSystem::from_sets(5, vec![vec![1, 2], vec![3, 4]]).unwrap();
        assert!(antichain.is_antichain());

        // {1} ⊂ {1,2} is not an antichain
        let not_antichain = SetSystem::from_sets(5, vec![vec![1], vec![1, 2]]).unwrap();
        assert!(!not_antichain.is_antichain());
    }

    #[test]
    fn test_is_uniform() {
        let uniform = SetSystem::from_sets(5, vec![vec![0, 1], vec![2, 3], vec![1, 4]]).unwrap();
        assert!(uniform.is_uniform());
        assert_eq!(uniform.uniformity(), Some(2));

        let not_uniform = SetSystem::from_sets(5, vec![vec![0, 1], vec![2, 3, 4]]).unwrap();
        assert!(!not_uniform.is_uniform());
        assert_eq!(not_uniform.uniformity(), None);
    }

    #[test]
    fn test_is_partition() {
        // Partition of {0,1,2,3,4}
        let partition = SetSystem::from_sets(5, vec![vec![0, 1], vec![2], vec![3, 4]]).unwrap();
        assert!(partition.is_partition());

        // Not a partition (0 appears twice)
        let not_partition =
            SetSystem::from_sets(5, vec![vec![0, 1], vec![0, 2], vec![3, 4]]).unwrap();
        assert!(!not_partition.is_partition());

        // Not a partition (4 is missing)
        let incomplete = SetSystem::from_sets(5, vec![vec![0, 1], vec![2, 3]]).unwrap();
        assert!(!incomplete.is_partition());
    }

    #[test]
    fn test_degree() {
        let sys = SetSystem::from_sets(5, vec![vec![0, 1, 2], vec![1, 3], vec![1, 2, 4]]).unwrap();

        assert_eq!(sys.degree(0), 1); // Appears in 1 set
        assert_eq!(sys.degree(1), 3); // Appears in all 3 sets
        assert_eq!(sys.degree(2), 2); // Appears in 2 sets
        assert_eq!(sys.degree(3), 1);
        assert_eq!(sys.degree(4), 1);

        let deg_seq = sys.degree_sequence();
        assert_eq!(deg_seq, vec![1, 3, 2, 1, 1]);
    }

    #[test]
    fn test_dual() {
        let sys = SetSystem::from_sets(3, vec![vec![0, 1], vec![1, 2], vec![0, 2]]).unwrap();
        let dual = sys.dual();

        // Original has 3 sets, so dual ground set size is 3
        assert_eq!(dual.ground_set_size(), 3);

        // Element 0 appears in sets 0 and 2
        // Element 1 appears in sets 0 and 1
        // Element 2 appears in sets 1 and 2
        assert!(dual.contains(&[0, 2])); // Sets containing element 0
        assert!(dual.contains(&[0, 1])); // Sets containing element 1
        assert!(dual.contains(&[1, 2])); // Sets containing element 2
    }

    #[test]
    fn test_vc_dimension_simple() {
        // Empty system has VC dimension 0
        let sys = SetSystem::new(3);
        assert_eq!(sys.vc_dimension(), 0);

        // A system with all singletons can shatter any single element
        let sys2 = SetSystem::from_sets(3, vec![vec![0], vec![1], vec![2], vec![]]).unwrap();
        assert!(sys2.vc_dimension() >= 1);

        // A complete system (all subsets) has VC dimension equal to ground set size
        let sys3 = SetSystem::from_sets(
            2,
            vec![vec![], vec![0], vec![1], vec![0, 1]],
        )
        .unwrap();
        assert_eq!(sys3.vc_dimension(), 2);
    }
}
