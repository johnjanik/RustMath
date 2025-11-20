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
            // Generate all subsets (including empty set)
            let n = set.len();
            for mask in 0..(1 << n) {
                let subset: Vec<usize> = set
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| mask & (1 << i) != 0)
                    .map(|(_, &x)| x)
                    .collect();

                shadow_sets.insert(subset);
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

    /// Check if the set system is hereditary (downward-closed)
    ///
    /// A set system is hereditary if whenever a set S is in the system,
    /// all subsets of S are also in the system. This is also called a
    /// downward-closed set system or an order ideal.
    ///
    /// # Example
    /// ```
    /// use rustmath_combinatorics::SetSystem;
    ///
    /// // {}, {0}, {1}, {0,1} is hereditary (all subsets of {0,1})
    /// let hereditary = SetSystem::from_sets(2, vec![
    ///     vec![],
    ///     vec![0],
    ///     vec![1],
    ///     vec![0, 1]
    /// ]).unwrap();
    /// assert!(hereditary.is_hereditary());
    ///
    /// // {0,1} without {0} is not hereditary
    /// let not_hereditary = SetSystem::from_sets(2, vec![vec![0, 1]]).unwrap();
    /// assert!(!not_hereditary.is_hereditary());
    /// ```
    pub fn is_hereditary(&self) -> bool {
        // A system is hereditary iff it equals its shadow
        let shadow = self.shadow();

        // Check that all sets in shadow are in this system
        for set in &shadow.sets {
            if !self.contains(set) {
                return false;
            }
        }

        true
    }

    /// Compute the hereditary closure (downward closure)
    ///
    /// Returns the smallest hereditary set system containing all sets in this system.
    /// This is equivalent to the shadow operation, but makes the intent clearer.
    ///
    /// # Example
    /// ```
    /// use rustmath_combinatorics::SetSystem;
    ///
    /// let sys = SetSystem::from_sets(3, vec![vec![0, 1, 2]]).unwrap();
    /// let closure = sys.hereditary_closure();
    ///
    /// // Should contain all subsets of {0,1,2}
    /// assert!(closure.contains(&[]));
    /// assert!(closure.contains(&[0]));
    /// assert!(closure.contains(&[1]));
    /// assert!(closure.contains(&[2]));
    /// assert!(closure.contains(&[0, 1]));
    /// assert!(closure.contains(&[0, 2]));
    /// assert!(closure.contains(&[1, 2]));
    /// assert!(closure.contains(&[0, 1, 2]));
    /// assert!(closure.is_hereditary());
    /// ```
    pub fn hereditary_closure(&self) -> SetSystem {
        self.shadow()
    }

    /// Compute the k-shadow of the set system
    ///
    /// The k-shadow consists of all k-element subsets that can be obtained
    /// by removing one element from a (k+1)-element set in the system.
    ///
    /// For a uniform system of (k+1)-sets, this gives all k-sets that are
    /// "covered" by the system.
    ///
    /// # Example
    /// ```
    /// use rustmath_combinatorics::SetSystem;
    ///
    /// // System of 3-sets
    /// let sys = SetSystem::from_sets(4, vec![
    ///     vec![0, 1, 2],
    ///     vec![0, 1, 3]
    /// ]).unwrap();
    ///
    /// let shadow_2 = sys.k_shadow(2);
    /// // Should contain {0,1}, {0,2}, {1,2}, {0,3}, {1,3}
    /// assert!(shadow_2.contains(&[0, 1]));
    /// assert!(shadow_2.contains(&[0, 2]));
    /// assert!(shadow_2.contains(&[1, 2]));
    /// assert!(shadow_2.contains(&[0, 3]));
    /// assert!(shadow_2.contains(&[1, 3]));
    /// ```
    pub fn k_shadow(&self, k: usize) -> SetSystem {
        let mut shadow_sets = HashSet::new();

        for set in &self.sets {
            // Only consider sets of size k+1
            if set.len() != k + 1 {
                continue;
            }

            // Generate all k-subsets of this (k+1)-set
            for i in 0..set.len() {
                let mut subset = Vec::new();
                for (j, &elem) in set.iter().enumerate() {
                    if j != i {
                        subset.push(elem);
                    }
                }
                shadow_sets.insert(subset);
            }
        }

        let sets: Vec<_> = shadow_sets.into_iter().collect();
        SetSystem::from_sets(self.n, sets).unwrap()
    }

    /// Compute the k-shade of the set system
    ///
    /// The k-shade consists of all k-element supersets that can be obtained
    /// by adding one element from the ground set to a (k-1)-element set in the system.
    ///
    /// For a uniform system of (k-1)-sets, this gives all k-sets that
    /// "cover" sets in the system.
    ///
    /// # Example
    /// ```
    /// use rustmath_combinatorics::SetSystem;
    ///
    /// // System of 2-sets
    /// let sys = SetSystem::from_sets(4, vec![
    ///     vec![0, 1],
    ///     vec![0, 2]
    /// ]).unwrap();
    ///
    /// let shade_3 = sys.k_shade(3);
    /// // Should contain all 3-sets containing {0,1} or {0,2}
    /// assert!(shade_3.contains(&[0, 1, 2]));
    /// assert!(shade_3.contains(&[0, 1, 3]));
    /// assert!(shade_3.contains(&[0, 2, 3]));
    /// ```
    pub fn k_shade(&self, k: usize) -> SetSystem {
        let mut shade_sets = HashSet::new();
        let ground_set: Vec<usize> = (0..self.n).collect();

        for set in &self.sets {
            // Only consider sets of size k-1
            if set.len() != k - 1 {
                continue;
            }

            let set_hash: HashSet<_> = set.iter().copied().collect();

            // Add each possible element from ground set
            for &elem in &ground_set {
                if !set_hash.contains(&elem) {
                    let mut superset = set.clone();
                    superset.push(elem);
                    superset.sort();
                    shade_sets.insert(superset);
                }
            }
        }

        let sets: Vec<_> = shade_sets.into_iter().collect();
        SetSystem::from_sets(self.n, sets).unwrap()
    }

    /// Check if the set system is pairwise disjoint
    ///
    /// A set system is pairwise disjoint if every pair of distinct sets
    /// has empty intersection.
    ///
    /// # Example
    /// ```
    /// use rustmath_combinatorics::SetSystem;
    ///
    /// let disjoint = SetSystem::from_sets(6, vec![
    ///     vec![0, 1],
    ///     vec![2, 3],
    ///     vec![4, 5]
    /// ]).unwrap();
    /// assert!(disjoint.is_pairwise_disjoint());
    ///
    /// let not_disjoint = SetSystem::from_sets(4, vec![
    ///     vec![0, 1],
    ///     vec![1, 2]  // Shares element 1
    /// ]).unwrap();
    /// assert!(!not_disjoint.is_pairwise_disjoint());
    /// ```
    pub fn is_pairwise_disjoint(&self) -> bool {
        for i in 0..self.sets.len() {
            let set_i: HashSet<_> = self.sets[i].iter().copied().collect();

            for j in (i + 1)..self.sets.len() {
                let set_j: HashSet<_> = self.sets[j].iter().copied().collect();

                // Check if intersection is non-empty
                if set_i.iter().any(|x| set_j.contains(x)) {
                    return false;
                }
            }
        }
        true
    }

    /// Compute the core of a collection of sets
    ///
    /// The core is the intersection of all sets in the collection.
    ///
    /// # Example
    /// ```
    /// use rustmath_combinatorics::SetSystem;
    ///
    /// let sets = vec![
    ///     vec![0, 1, 2],
    ///     vec![0, 1, 3],
    ///     vec![0, 1, 4]
    /// ];
    /// let core = SetSystem::compute_core(&sets);
    /// assert_eq!(core, vec![0, 1]);
    /// ```
    pub fn compute_core(sets: &[Vec<usize>]) -> Vec<usize> {
        if sets.is_empty() {
            return Vec::new();
        }

        let mut core: HashSet<_> = sets[0].iter().copied().collect();

        for set in &sets[1..] {
            let set_hash: HashSet<_> = set.iter().copied().collect();
            core.retain(|x| set_hash.contains(x));
        }

        let mut result: Vec<_> = core.into_iter().collect();
        result.sort();
        result
    }

    /// Check if a collection of sets forms a sunflower (Δ-system)
    ///
    /// A sunflower is a collection of sets where every pair of sets has
    /// the same intersection (called the "core").
    ///
    /// # Example
    /// ```
    /// use rustmath_combinatorics::SetSystem;
    ///
    /// // Sets {0,1,2}, {0,1,3}, {0,1,4} form a sunflower with core {0,1}
    /// let sunflower = vec![
    ///     vec![0, 1, 2],
    ///     vec![0, 1, 3],
    ///     vec![0, 1, 4]
    /// ];
    /// assert!(SetSystem::is_sunflower(&sunflower));
    ///
    /// // These don't form a sunflower
    /// let not_sunflower = vec![
    ///     vec![0, 1],
    ///     vec![1, 2],
    ///     vec![2, 3]
    /// ];
    /// assert!(!SetSystem::is_sunflower(&not_sunflower));
    /// ```
    pub fn is_sunflower(sets: &[Vec<usize>]) -> bool {
        if sets.len() <= 1 {
            return true; // Trivially a sunflower
        }

        // Compute the core (intersection of all sets)
        let core = Self::compute_core(sets);
        let core_set: HashSet<_> = core.iter().copied().collect();

        // Check that every pair has exactly this core as intersection
        for i in 0..sets.len() {
            for j in (i + 1)..sets.len() {
                let set_i: HashSet<_> = sets[i].iter().copied().collect();
                let set_j: HashSet<_> = sets[j].iter().copied().collect();

                let intersection: HashSet<_> = set_i
                    .intersection(&set_j)
                    .copied()
                    .collect();

                if intersection != core_set {
                    return false;
                }
            }
        }

        true
    }

    /// Find a sunflower with r petals (r sets) in the system
    ///
    /// Returns the indices of sets forming a sunflower, or None if not found.
    ///
    /// # Example
    /// ```
    /// use rustmath_combinatorics::SetSystem;
    ///
    /// let sys = SetSystem::from_sets(5, vec![
    ///     vec![0, 1, 2],
    ///     vec![0, 1, 3],
    ///     vec![0, 1, 4]
    /// ]).unwrap();
    ///
    /// let sunflower = sys.find_sunflower(3);
    /// assert!(sunflower.is_some());
    /// ```
    pub fn find_sunflower(&self, r: usize) -> Option<Vec<usize>> {
        if r > self.sets.len() {
            return None;
        }

        if r == 0 {
            return Some(Vec::new());
        }

        if r == 1 {
            return Some(vec![0]);
        }

        // Try all combinations of r sets
        let mut indices = Vec::new();
        Self::find_sunflower_recursive(&self.sets, r, 0, &mut indices)
    }

    fn find_sunflower_recursive(
        sets: &[Vec<usize>],
        r: usize,
        start: usize,
        current: &mut Vec<usize>,
    ) -> Option<Vec<usize>> {
        if current.len() == r {
            // Check if current indices form a sunflower
            let selected_sets: Vec<_> = current.iter().map(|&i| sets[i].clone()).collect();
            if Self::is_sunflower(&selected_sets) {
                return Some(current.clone());
            }
            return None;
        }

        for i in start..sets.len() {
            current.push(i);
            if let Some(result) = Self::find_sunflower_recursive(sets, r, i + 1, current) {
                return Some(result);
            }
            current.pop();
        }

        None
    }

    /// Check if the system contains a sunflower with at least r petals
    ///
    /// This is more efficient than find_sunflower when you only need
    /// a yes/no answer.
    pub fn contains_sunflower(&self, r: usize) -> bool {
        self.find_sunflower(r).is_some()
    }

    /// Compute the sunflower bound from the sunflower lemma
    ///
    /// The sunflower lemma states: If a family of k-element sets has more than
    /// k! * (r-1)^k sets, then it contains a sunflower with r petals.
    ///
    /// This function returns the maximum number of k-element sets that can exist
    /// without necessarily containing an r-sunflower.
    ///
    /// # Example
    /// ```
    /// use rustmath_combinatorics::SetSystem;
    ///
    /// // For k=3, r=3: bound is 3! * 2^3 = 6 * 8 = 48
    /// assert_eq!(SetSystem::sunflower_bound(3, 3), 48);
    /// ```
    pub fn sunflower_bound(k: usize, r: usize) -> usize {
        if r <= 1 {
            return usize::MAX;
        }

        // Compute k!
        let mut factorial = 1usize;
        for i in 1..=k {
            factorial = factorial.saturating_mul(i);
        }

        // Compute (r-1)^k
        let base = r.saturating_sub(1);
        let mut power = 1usize;
        for _ in 0..k {
            power = power.saturating_mul(base);
        }

        factorial.saturating_mul(power)
    }

    /// Check if the system exceeds the sunflower bound
    ///
    /// If this returns true and the system is uniform with k-element sets,
    /// then by the sunflower lemma, the system must contain a sunflower
    /// with r petals.
    ///
    /// # Example
    /// ```
    /// use rustmath_combinatorics::SetSystem;
    ///
    /// let sys = SetSystem::from_sets(10, vec![
    ///     vec![0, 1],
    ///     vec![2, 3],
    ///     vec![4, 5],
    ///     vec![6, 7],
    ///     vec![8, 9]
    /// ]).unwrap();
    ///
    /// // For k=2, r=3: bound is 2! * 2^2 = 2 * 4 = 8
    /// // We have 5 sets, which is less than 8, so no guarantee
    /// assert!(!sys.exceeds_sunflower_bound(3));
    /// ```
    pub fn exceeds_sunflower_bound(&self, r: usize) -> bool {
        if !self.is_uniform() {
            return false; // Lemma only applies to uniform systems
        }

        let k = self.uniformity().unwrap_or(0);
        let bound = Self::sunflower_bound(k, r);
        self.num_sets() > bound
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

    #[test]
    fn test_is_hereditary() {
        // Empty system is hereditary
        let empty = SetSystem::new(3);
        assert!(empty.is_hereditary());

        // All subsets of {0,1} - this is hereditary
        let hereditary = SetSystem::from_sets(
            2,
            vec![vec![], vec![0], vec![1], vec![0, 1]],
        )
        .unwrap();
        assert!(hereditary.is_hereditary());

        // Just {0,1} without its subsets - not hereditary
        let not_hereditary = SetSystem::from_sets(2, vec![vec![0, 1]]).unwrap();
        assert!(!not_hereditary.is_hereditary());

        // {0}, {1}, {0,1} without empty set - not hereditary
        let missing_empty = SetSystem::from_sets(2, vec![vec![0], vec![1], vec![0, 1]]).unwrap();
        assert!(!missing_empty.is_hereditary());

        // {}, {0}, {1} without {0,1} - this IS hereditary (downward-closed)
        let partial = SetSystem::from_sets(2, vec![vec![], vec![0], vec![1]]).unwrap();
        assert!(partial.is_hereditary());

        // System with multiple maximal sets - all subsets of {0,1,2}
        let multi_max = SetSystem::from_sets(
            4,
            vec![
                vec![],
                vec![0],
                vec![1],
                vec![2],
                vec![0, 1],
                vec![0, 2],
                vec![1, 2],
                vec![0, 1, 2],
            ],
        )
        .unwrap();
        assert!(multi_max.is_hereditary());

        // Missing a subset in the middle
        let missing_middle = SetSystem::from_sets(
            3,
            vec![
                vec![],
                vec![0],
                vec![1],
                vec![2],
                vec![0, 1],
                // Missing {0,2} and {1,2}
                vec![0, 1, 2],
            ],
        )
        .unwrap();
        assert!(!missing_middle.is_hereditary());
    }

    #[test]
    fn test_hereditary_closure() {
        // Closure of single 3-set should give all 2^3 = 8 subsets
        let sys = SetSystem::from_sets(3, vec![vec![0, 1, 2]]).unwrap();
        let closure = sys.hereditary_closure();

        assert_eq!(closure.num_sets(), 8);
        assert!(closure.contains(&[]));
        assert!(closure.contains(&[0]));
        assert!(closure.contains(&[1]));
        assert!(closure.contains(&[2]));
        assert!(closure.contains(&[0, 1]));
        assert!(closure.contains(&[0, 2]));
        assert!(closure.contains(&[1, 2]));
        assert!(closure.contains(&[0, 1, 2]));
        assert!(closure.is_hereditary());

        // Closure of two disjoint 2-sets
        let sys2 = SetSystem::from_sets(4, vec![vec![0, 1], vec![2, 3]]).unwrap();
        let closure2 = sys2.hereditary_closure();

        // Should have: {}, {0}, {1}, {2}, {3}, {0,1}, {2,3}
        assert_eq!(closure2.num_sets(), 7);
        assert!(closure2.is_hereditary());

        // Closure of already hereditary system should be itself
        let already_hereditary = SetSystem::from_sets(
            2,
            vec![vec![], vec![0], vec![1], vec![0, 1]],
        )
        .unwrap();
        let closure3 = already_hereditary.hereditary_closure();
        assert_eq!(closure3.num_sets(), 4);
    }

    #[test]
    fn test_k_shadow() {
        // Shadow of 3-sets
        let sys = SetSystem::from_sets(
            4,
            vec![vec![0, 1, 2], vec![0, 1, 3]],
        )
        .unwrap();

        let shadow_2 = sys.k_shadow(2);

        // {0,1,2} gives {0,1}, {0,2}, {1,2}
        // {0,1,3} gives {0,1}, {0,3}, {1,3}
        // Combined: {0,1}, {0,2}, {1,2}, {0,3}, {1,3}
        assert_eq!(shadow_2.num_sets(), 5);
        assert!(shadow_2.contains(&[0, 1]));
        assert!(shadow_2.contains(&[0, 2]));
        assert!(shadow_2.contains(&[1, 2]));
        assert!(shadow_2.contains(&[0, 3]));
        assert!(shadow_2.contains(&[1, 3]));

        // k-shadow of k-sets should be empty (no k+1 sets to shadow from)
        let shadow_same = sys.k_shadow(3);
        assert_eq!(shadow_same.num_sets(), 0);

        // Shadow of single 4-set
        let sys2 = SetSystem::from_sets(4, vec![vec![0, 1, 2, 3]]).unwrap();
        let shadow_3 = sys2.k_shadow(3);
        assert_eq!(shadow_3.num_sets(), 4); // All 3-subsets of {0,1,2,3}
    }

    #[test]
    fn test_k_shade() {
        // Shade of 2-sets
        let sys = SetSystem::from_sets(
            4,
            vec![vec![0, 1], vec![0, 2]],
        )
        .unwrap();

        let shade_3 = sys.k_shade(3);

        // {0,1} can be extended to {0,1,2}, {0,1,3}
        // {0,2} can be extended to {0,1,2}, {0,2,3}
        // Combined: {0,1,2}, {0,1,3}, {0,2,3}
        assert_eq!(shade_3.num_sets(), 3);
        assert!(shade_3.contains(&[0, 1, 2]));
        assert!(shade_3.contains(&[0, 1, 3]));
        assert!(shade_3.contains(&[0, 2, 3]));

        // k-shade of k-sets should be empty (need k-1 sets to shade from)
        let shade_same = sys.k_shade(2);
        assert_eq!(shade_same.num_sets(), 0);

        // Shade of single empty set (0-set)
        let sys2 = SetSystem::from_sets(3, vec![vec![]]).unwrap();
        let shade_1 = sys2.k_shade(1);
        assert_eq!(shade_1.num_sets(), 3); // All singletons {0}, {1}, {2}
        assert!(shade_1.contains(&[0]));
        assert!(shade_1.contains(&[1]));
        assert!(shade_1.contains(&[2]));
    }

    #[test]
    fn test_shadow_shade_relationship() {
        // For a uniform k-system, k_shadow gives (k-1)-sets
        let sys = SetSystem::from_sets(
            5,
            vec![
                vec![0, 1, 2],
                vec![0, 1, 3],
                vec![0, 2, 3],
            ],
        )
        .unwrap();

        let shadow = sys.k_shadow(2);
        let shade_back = shadow.k_shade(3);

        // The shade of the shadow should contain at least the original sets
        for set in sys.sets() {
            assert!(shade_back.contains(set));
        }
    }

    #[test]
    fn test_hereditary_edge_cases() {
        // Single element ground set
        let sys1 = SetSystem::from_sets(1, vec![vec![], vec![0]]).unwrap();
        assert!(sys1.is_hereditary());

        let sys2 = SetSystem::from_sets(1, vec![vec![0]]).unwrap();
        assert!(!sys2.is_hereditary()); // Missing empty set

        // Just empty set is hereditary
        let sys3 = SetSystem::from_sets(5, vec![vec![]]).unwrap();
        assert!(sys3.is_hereditary());

        // Empty system is hereditary (vacuously true)
        let sys4 = SetSystem::new(5);
        assert!(sys4.is_hereditary());
    }

    #[test]
    fn test_k_shadow_shade_edge_cases() {
        // Empty system
        let empty = SetSystem::new(3);
        let shadow = empty.k_shadow(2);
        assert_eq!(shadow.num_sets(), 0);
        let shade = empty.k_shade(2);
        assert_eq!(shade.num_sets(), 0);

        // Single empty set
        let single_empty = SetSystem::from_sets(3, vec![vec![]]).unwrap();
        let shadow = single_empty.k_shadow(0);
        assert_eq!(shadow.num_sets(), 0); // No 1-sets to shadow from

        // k-shadow with k=0 (should find all empty sets from 1-sets)
        let singletons = SetSystem::from_sets(3, vec![vec![0], vec![1]]).unwrap();
        let shadow_0 = singletons.k_shadow(0);
        assert_eq!(shadow_0.num_sets(), 1); // Just the empty set
        assert!(shadow_0.contains(&[]));
    }

    #[test]
    fn test_is_pairwise_disjoint() {
        // Pairwise disjoint sets
        let disjoint = SetSystem::from_sets(6, vec![
            vec![0, 1],
            vec![2, 3],
            vec![4, 5]
        ]).unwrap();
        assert!(disjoint.is_pairwise_disjoint());

        // Not pairwise disjoint (share element 1)
        let not_disjoint = SetSystem::from_sets(4, vec![
            vec![0, 1],
            vec![1, 2]
        ]).unwrap();
        assert!(!not_disjoint.is_pairwise_disjoint());

        // Empty system is trivially pairwise disjoint
        let empty = SetSystem::new(5);
        assert!(empty.is_pairwise_disjoint());

        // Single set is trivially pairwise disjoint
        let single = SetSystem::from_sets(3, vec![vec![0, 1, 2]]).unwrap();
        assert!(single.is_pairwise_disjoint());

        // Multiple sets with some overlap
        let overlap = SetSystem::from_sets(5, vec![
            vec![0, 1],
            vec![2, 3],
            vec![3, 4]  // Shares 3 with previous set
        ]).unwrap();
        assert!(!overlap.is_pairwise_disjoint());
    }

    #[test]
    fn test_compute_core() {
        // Core of sets with common elements
        let sets1 = vec![
            vec![0, 1, 2],
            vec![0, 1, 3],
            vec![0, 1, 4]
        ];
        let core1 = SetSystem::compute_core(&sets1);
        assert_eq!(core1, vec![0, 1]);

        // Core of disjoint sets (empty)
        let sets2 = vec![
            vec![0, 1],
            vec![2, 3],
            vec![4, 5]
        ];
        let core2 = SetSystem::compute_core(&sets2);
        assert_eq!(core2, Vec::<usize>::new());

        // Core of identical sets
        let sets3 = vec![
            vec![1, 2, 3],
            vec![1, 2, 3],
            vec![1, 2, 3]
        ];
        let core3 = SetSystem::compute_core(&sets3);
        assert_eq!(core3, vec![1, 2, 3]);

        // Core of single set
        let sets4 = vec![vec![5, 6, 7]];
        let core4 = SetSystem::compute_core(&sets4);
        assert_eq!(core4, vec![5, 6, 7]);

        // Core of empty vector
        let sets5: Vec<Vec<usize>> = vec![];
        let core5 = SetSystem::compute_core(&sets5);
        assert_eq!(core5, Vec::<usize>::new());
    }

    #[test]
    fn test_is_sunflower() {
        // Classic sunflower with non-empty core
        let sunflower1 = vec![
            vec![0, 1, 2],
            vec![0, 1, 3],
            vec![0, 1, 4]
        ];
        assert!(SetSystem::is_sunflower(&sunflower1));

        // Pairwise disjoint sets (sunflower with empty core)
        let sunflower2 = vec![
            vec![0, 1],
            vec![2, 3],
            vec![4, 5]
        ];
        assert!(SetSystem::is_sunflower(&sunflower2));

        // Not a sunflower (different pairwise intersections)
        let not_sunflower = vec![
            vec![0, 1],
            vec![1, 2],
            vec![2, 3]
        ];
        assert!(!SetSystem::is_sunflower(&not_sunflower));

        // Single set is trivially a sunflower
        let single = vec![vec![1, 2, 3]];
        assert!(SetSystem::is_sunflower(&single));

        // Empty collection is trivially a sunflower
        let empty: Vec<Vec<usize>> = vec![];
        assert!(SetSystem::is_sunflower(&empty));

        // Two sets with intersection
        let two_sets = vec![
            vec![0, 1, 2],
            vec![0, 1, 3]
        ];
        assert!(SetSystem::is_sunflower(&two_sets));

        // Complex sunflower pattern
        let complex = vec![
            vec![0, 1, 2, 5],
            vec![0, 1, 3, 6],
            vec![0, 1, 4, 7]
        ];
        assert!(SetSystem::is_sunflower(&complex));

        // Not sunflower - varying intersections
        let varying = vec![
            vec![0, 1, 2],  // Shares {0,1} with second
            vec![0, 1, 3],  // Shares {0,1} with first, {1,3} with third
            vec![1, 3, 4]   // Shares {1,3} with second
        ];
        assert!(!SetSystem::is_sunflower(&varying));
    }

    #[test]
    fn test_find_sunflower() {
        // System with clear sunflower
        let sys1 = SetSystem::from_sets(5, vec![
            vec![0, 1, 2],
            vec![0, 1, 3],
            vec![0, 1, 4]
        ]).unwrap();

        let result = sys1.find_sunflower(3);
        assert!(result.is_some());
        let indices = result.unwrap();
        assert_eq!(indices, vec![0, 1, 2]);

        // System with sunflower of size 2
        let sys2 = SetSystem::from_sets(4, vec![
            vec![0, 1],
            vec![2, 3]
        ]).unwrap();

        let result2 = sys2.find_sunflower(2);
        assert!(result2.is_some());

        // System without 3-sunflower but with 2-sunflower
        let sys3 = SetSystem::from_sets(4, vec![
            vec![0, 1],
            vec![1, 2],
            vec![2, 3]
        ]).unwrap();

        // Any 2 sets form a sunflower
        assert!(sys3.find_sunflower(2).is_some());

        // But not all 3 together
        let result3 = sys3.find_sunflower(3);
        assert!(result3.is_none());

        // Request more sets than available
        let sys4 = SetSystem::from_sets(3, vec![vec![0], vec![1]]).unwrap();
        assert!(sys4.find_sunflower(5).is_none());

        // Pairwise disjoint system forms a sunflower
        let sys5 = SetSystem::from_sets(6, vec![
            vec![0, 1],
            vec![2, 3],
            vec![4, 5]
        ]).unwrap();
        assert!(sys5.find_sunflower(3).is_some());
    }

    #[test]
    fn test_contains_sunflower() {
        let sys = SetSystem::from_sets(5, vec![
            vec![0, 1, 2],
            vec![0, 1, 3],
            vec![0, 1, 4]
        ]).unwrap();

        assert!(sys.contains_sunflower(2));
        assert!(sys.contains_sunflower(3));
        assert!(!sys.contains_sunflower(4)); // Only 3 sets total
    }

    #[test]
    fn test_sunflower_bound() {
        // For k=2, r=3: bound = 2! * 2^2 = 2 * 4 = 8
        assert_eq!(SetSystem::sunflower_bound(2, 3), 8);

        // For k=3, r=3: bound = 3! * 2^3 = 6 * 8 = 48
        assert_eq!(SetSystem::sunflower_bound(3, 3), 48);

        // For k=1, r=4: bound = 1! * 3^1 = 1 * 3 = 3
        assert_eq!(SetSystem::sunflower_bound(1, 4), 3);

        // For k=2, r=2: bound = 2! * 1^2 = 2 * 1 = 2
        assert_eq!(SetSystem::sunflower_bound(2, 2), 2);

        // Edge case: r=1 should give maximum
        assert_eq!(SetSystem::sunflower_bound(5, 1), usize::MAX);

        // k=0 should give 1 (0! * (r-1)^0 = 1 * 1)
        assert_eq!(SetSystem::sunflower_bound(0, 3), 1);
    }

    #[test]
    fn test_exceeds_sunflower_bound() {
        // Small system below bound
        let sys1 = SetSystem::from_sets(10, vec![
            vec![0, 1],
            vec![2, 3],
            vec![4, 5]
        ]).unwrap();

        // For k=2, r=3: bound = 8, we have 3 sets
        assert!(!sys1.exceeds_sunflower_bound(3));

        // Build a system that exceeds the bound for k=2, r=2
        // Bound is 2! * 1^2 = 2
        let sys2 = SetSystem::from_sets(6, vec![
            vec![0, 1],
            vec![2, 3],
            vec![4, 5]
        ]).unwrap();

        // 3 sets > bound of 2
        assert!(sys2.exceeds_sunflower_bound(2));

        // Non-uniform system doesn't exceed
        let sys3 = SetSystem::from_sets(5, vec![
            vec![0, 1],
            vec![2, 3, 4]
        ]).unwrap();
        assert!(!sys3.exceeds_sunflower_bound(3));
    }

    #[test]
    fn test_sunflower_lemma_application() {
        // Create a uniform system of 2-sets that exceeds bound
        // For k=2, r=3: bound = 2! * 2^2 = 8
        // If we have 9 or more 2-sets, we must have a 3-sunflower

        let sys = SetSystem::from_sets(20, vec![
            vec![0, 1],
            vec![2, 3],
            vec![4, 5],
            vec![6, 7],
            vec![8, 9],
            vec![10, 11],
            vec![12, 13],
            vec![14, 15],
            vec![16, 17]  // 9 sets
        ]).unwrap();

        assert!(sys.is_uniform());
        assert_eq!(sys.uniformity(), Some(2));
        assert!(sys.exceeds_sunflower_bound(3));

        // By the sunflower lemma, this system must contain a 3-sunflower
        // Since all sets are disjoint, they form a sunflower with empty core
        assert!(sys.contains_sunflower(3));
    }

    #[test]
    fn test_pairwise_disjoint_is_sunflower() {
        // Pairwise disjoint sets always form a sunflower (with empty core)
        let disjoint_sets = vec![
            vec![0, 1],
            vec![2, 3],
            vec![4, 5],
            vec![6, 7]
        ];

        assert!(SetSystem::is_sunflower(&disjoint_sets));

        let core = SetSystem::compute_core(&disjoint_sets);
        assert_eq!(core, Vec::<usize>::new());
    }

    #[test]
    fn test_sunflower_edge_cases() {
        // Empty sets in sunflower
        let with_empty = vec![
            vec![],
            vec![],
            vec![]
        ];
        assert!(SetSystem::is_sunflower(&with_empty));

        // Mix of empty and non-empty with 2 sets (IS a sunflower - only one pair with empty intersection)
        let mixed = vec![
            vec![],
            vec![1, 2]
        ];
        assert!(SetSystem::is_sunflower(&mixed)); // Two sets always form a sunflower

        // Mix of empty and non-empty with 3 sets (NOT a sunflower - different pairwise intersections)
        let mixed3 = vec![
            vec![],
            vec![1, 2],
            vec![3, 4]
        ];
        // Intersection of empty and {1,2} is empty
        // Intersection of empty and {3,4} is empty
        // Intersection of {1,2} and {3,4} is empty
        // All pairwise intersections are empty, so this IS a sunflower
        assert!(SetSystem::is_sunflower(&mixed3));

        // A real non-sunflower with varying intersections
        let not_sunflower_mixed = vec![
            vec![1],
            vec![1, 2],
            vec![2, 3]
        ];
        // Intersection of {1} and {1,2} is {1}
        // Intersection of {1} and {2,3} is empty
        // Different pairwise intersections, so NOT a sunflower
        assert!(!SetSystem::is_sunflower(&not_sunflower_mixed));

        // All identical sets
        let identical = vec![
            vec![1, 2, 3],
            vec![1, 2, 3],
            vec![1, 2, 3]
        ];
        assert!(SetSystem::is_sunflower(&identical));
    }
}
