//! Disjoint Set (Union-Find) Data Structure
//!
//! This module provides efficient implementations of the disjoint set data structure,
//! also known as union-find or merge-find. It supports near-constant time operations
//! for tracking partitions of elements into disjoint sets.
//!
//! # Implementations
//!
//! - [`DisjointSet_of_integers`]: Optimized for integer indices (0..n)
//! - [`DisjointSet_of_hashables`]: Generic implementation for any hashable type
//!
//! # Time Complexity
//!
//! With path compression and union-by-rank optimizations:
//! - `find()`: O(α(n)) amortized, where α is the inverse Ackermann function
//! - `union()`: O(α(n)) amortized
//! - `same_set()`: O(α(n)) amortized
//!
//! In practice, α(n) ≤ 4 for all reasonable values of n, making these operations
//! effectively constant time.
//!
//! # Examples
//!
//! ```
//! use rustmath_sets::disjoint_set::DisjointSet_of_integers;
//!
//! let mut ds = DisjointSet_of_integers::new(5);
//! ds.union(0, 1);
//! ds.union(2, 3);
//! assert!(ds.same_set(0, 1));
//! assert!(!ds.same_set(0, 2));
//! ```

use std::collections::HashMap;
use std::hash::Hash;

/// Disjoint-set data structure for integer indices.
///
/// Optimized implementation for integers in the range [0, n). Uses Vec-based
/// storage for maximum performance with path compression and union-by-rank.
///
/// Corresponds to `sage.sets.disjoint_set.DisjointSet_of_integers`.
///
/// # Time Complexity
///
/// - `new(n)`: O(n)
/// - `find(x)`: O(α(n)) amortized
/// - `union(x, y)`: O(α(n)) amortized
/// - `same_set(x, y)`: O(α(n)) amortized
/// - `set_size(x)`: O(α(n)) amortized
/// - `num_sets()`: O(n)
///
/// where α(n) is the inverse Ackermann function, effectively constant.
///
/// # Space Complexity
///
/// O(n) for n elements.
#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub struct DisjointSet_of_integers {
    /// Parent pointers: parent[i] is the parent of element i
    parent: Vec<usize>,
    /// Rank of each tree (upper bound on height)
    rank: Vec<usize>,
    /// Size of each set (only accurate for root elements)
    size: Vec<usize>,
}

impl DisjointSet_of_integers {
    /// Creates a new disjoint set with n singleton sets.
    ///
    /// Initially, each element i in [0, n) is in its own set.
    ///
    /// # Time Complexity
    ///
    /// O(n)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_sets::disjoint_set::DisjointSet_of_integers;
    ///
    /// let ds = DisjointSet_of_integers::new(5);
    /// // Creates 5 singleton sets: {0}, {1}, {2}, {3}, {4}
    /// ```
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            size: vec![1; n],
        }
    }

    /// Finds the representative (root) of the set containing x.
    ///
    /// Uses path compression to flatten the tree structure, making future
    /// operations faster.
    ///
    /// # Time Complexity
    ///
    /// O(α(n)) amortized
    ///
    /// # Panics
    ///
    /// Panics if x >= n.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_sets::disjoint_set::DisjointSet_of_integers;
    ///
    /// let mut ds = DisjointSet_of_integers::new(5);
    /// ds.union(0, 1);
    /// let root0 = ds.find(0);
    /// let root1 = ds.find(1);
    /// assert_eq!(root0, root1);
    /// ```
    pub fn find(&mut self, x: usize) -> usize {
        // Path compression: make x point directly to the root
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    /// Unites the sets containing x and y.
    ///
    /// Uses union-by-rank to keep trees balanced. Returns true if the sets
    /// were merged (i.e., x and y were in different sets), false otherwise.
    ///
    /// # Time Complexity
    ///
    /// O(α(n)) amortized
    ///
    /// # Panics
    ///
    /// Panics if x >= n or y >= n.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_sets::disjoint_set::DisjointSet_of_integers;
    ///
    /// let mut ds = DisjointSet_of_integers::new(5);
    /// assert!(ds.union(0, 1)); // Returns true (sets were different)
    /// assert!(!ds.union(0, 1)); // Returns false (already in same set)
    /// ```
    pub fn union(&mut self, x: usize, y: usize) -> bool {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x == root_y {
            return false; // Already in the same set
        }

        // Union by rank: attach smaller rank tree under root of higher rank tree
        if self.rank[root_x] < self.rank[root_y] {
            self.parent[root_x] = root_y;
            self.size[root_y] += self.size[root_x];
        } else if self.rank[root_x] > self.rank[root_y] {
            self.parent[root_y] = root_x;
            self.size[root_x] += self.size[root_y];
        } else {
            // Equal rank: choose arbitrarily and increment rank
            self.parent[root_y] = root_x;
            self.size[root_x] += self.size[root_y];
            self.rank[root_x] += 1;
        }

        true
    }

    /// Checks if x and y are in the same set.
    ///
    /// # Time Complexity
    ///
    /// O(α(n)) amortized
    ///
    /// # Panics
    ///
    /// Panics if x >= n or y >= n.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_sets::disjoint_set::DisjointSet_of_integers;
    ///
    /// let mut ds = DisjointSet_of_integers::new(5);
    /// ds.union(0, 1);
    /// assert!(ds.same_set(0, 1));
    /// assert!(!ds.same_set(0, 2));
    /// ```
    pub fn same_set(&mut self, x: usize, y: usize) -> bool {
        self.find(x) == self.find(y)
    }

    /// Returns the size of the set containing x.
    ///
    /// # Time Complexity
    ///
    /// O(α(n)) amortized
    ///
    /// # Panics
    ///
    /// Panics if x >= n.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_sets::disjoint_set::DisjointSet_of_integers;
    ///
    /// let mut ds = DisjointSet_of_integers::new(5);
    /// ds.union(0, 1);
    /// ds.union(1, 2);
    /// assert_eq!(ds.set_size(0), 3);
    /// assert_eq!(ds.set_size(3), 1);
    /// ```
    pub fn set_size(&mut self, x: usize) -> usize {
        let root = self.find(x);
        self.size[root]
    }

    /// Returns the number of disjoint sets.
    ///
    /// # Time Complexity
    ///
    /// O(n)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_sets::disjoint_set::DisjointSet_of_integers;
    ///
    /// let mut ds = DisjointSet_of_integers::new(5);
    /// assert_eq!(ds.num_sets(), 5);
    /// ds.union(0, 1);
    /// assert_eq!(ds.num_sets(), 4);
    /// ds.union(2, 3);
    /// assert_eq!(ds.num_sets(), 3);
    /// ```
    pub fn num_sets(&mut self) -> usize {
        (0..self.parent.len())
            .filter(|&i| self.find(i) == i)
            .count()
    }

    /// Returns the total number of elements.
    ///
    /// # Time Complexity
    ///
    /// O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_sets::disjoint_set::DisjointSet_of_integers;
    ///
    /// let ds = DisjointSet_of_integers::new(10);
    /// assert_eq!(ds.num_elements(), 10);
    /// ```
    pub fn num_elements(&self) -> usize {
        self.parent.len()
    }

    /// Returns all elements in the same set as x.
    ///
    /// # Time Complexity
    ///
    /// O(n)
    ///
    /// # Panics
    ///
    /// Panics if x >= n.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_sets::disjoint_set::DisjointSet_of_integers;
    ///
    /// let mut ds = DisjointSet_of_integers::new(5);
    /// ds.union(0, 1);
    /// ds.union(1, 2);
    /// let mut elements = ds.get_set_elements(0);
    /// elements.sort();
    /// assert_eq!(elements, vec![0, 1, 2]);
    /// ```
    pub fn get_set_elements(&mut self, x: usize) -> Vec<usize> {
        let root = self.find(x);
        (0..self.parent.len())
            .filter(|&i| self.find(i) == root)
            .collect()
    }

    /// Returns all disjoint sets as a vector of vectors.
    ///
    /// # Time Complexity
    ///
    /// O(n²) in worst case, but typically much faster due to path compression.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_sets::disjoint_set::DisjointSet_of_integers;
    ///
    /// let mut ds = DisjointSet_of_integers::new(5);
    /// ds.union(0, 1);
    /// ds.union(2, 3);
    /// let sets = ds.get_all_sets();
    /// assert_eq!(sets.len(), 3); // {0,1}, {2,3}, {4}
    /// ```
    pub fn get_all_sets(&mut self) -> Vec<Vec<usize>> {
        let mut sets: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..self.parent.len() {
            let root = self.find(i);
            sets.entry(root).or_insert_with(Vec::new).push(i);
        }
        sets.into_values().collect()
    }
}

/// Disjoint-set data structure for hashable elements.
///
/// Generic implementation that works with any type implementing `Eq`, `Hash`, and `Clone`.
/// Internally maps elements to indices and uses [`DisjointSet_of_integers`] for the
/// union-find operations.
///
/// Corresponds to `sage.sets.disjoint_set.DisjointSet_of_hashables`.
///
/// # Time Complexity
///
/// Same as [`DisjointSet_of_integers`] plus O(1) expected time for hash lookups.
///
/// # Space Complexity
///
/// O(n) for n elements, plus HashMap overhead.
///
/// # Examples
///
/// ```
/// use rustmath_sets::disjoint_set::DisjointSet_of_hashables;
///
/// let mut ds = DisjointSet_of_hashables::new(vec!["a", "b", "c", "d"]);
/// ds.union(&"a", &"b");
/// assert!(ds.same_set(&"a", &"b"));
/// assert!(!ds.same_set(&"a", &"c"));
/// ```
#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub struct DisjointSet_of_hashables<T: Eq + Hash + Clone> {
    /// Maps elements to their indices
    element_to_idx: HashMap<T, usize>,
    /// Maps indices back to elements
    idx_to_element: Vec<T>,
    /// The underlying integer-based disjoint set
    disjoint_set: DisjointSet_of_integers,
}

impl<T: Eq + Hash + Clone> DisjointSet_of_hashables<T> {
    /// Creates a new disjoint set from a collection of elements.
    ///
    /// Each element starts in its own singleton set. If there are duplicate
    /// elements, only the first occurrence is kept.
    ///
    /// # Time Complexity
    ///
    /// O(n) expected
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_sets::disjoint_set::DisjointSet_of_hashables;
    ///
    /// let ds = DisjointSet_of_hashables::new(vec!["apple", "banana", "cherry"]);
    /// assert_eq!(ds.num_elements(), 3);
    /// ```
    pub fn new(elements: Vec<T>) -> Self {
        let mut element_to_idx = HashMap::new();
        let mut idx_to_element = Vec::new();

        for element in elements {
            if !element_to_idx.contains_key(&element) {
                let idx = idx_to_element.len();
                element_to_idx.insert(element.clone(), idx);
                idx_to_element.push(element);
            }
        }

        let n = idx_to_element.len();
        Self {
            element_to_idx,
            idx_to_element,
            disjoint_set: DisjointSet_of_integers::new(n),
        }
    }

    /// Finds the representative element of the set containing the given element.
    ///
    /// Returns `None` if the element is not in the disjoint set.
    ///
    /// # Time Complexity
    ///
    /// O(α(n)) amortized (plus O(1) expected for hash lookup)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_sets::disjoint_set::DisjointSet_of_hashables;
    ///
    /// let mut ds = DisjointSet_of_hashables::new(vec!["a", "b", "c"]);
    /// ds.union(&"a", &"b");
    /// assert!(ds.same_set(&"a", &"b"));
    /// ```
    pub fn find(&mut self, element: &T) -> Option<&T> {
        let idx = *self.element_to_idx.get(element)?;
        let root_idx = self.disjoint_set.find(idx);
        Some(&self.idx_to_element[root_idx])
    }

    /// Unites the sets containing x and y.
    ///
    /// Returns `true` if the sets were merged, `false` if they were already
    /// in the same set or if either element is not in the disjoint set.
    ///
    /// # Time Complexity
    ///
    /// O(α(n)) amortized (plus O(1) expected for hash lookups)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_sets::disjoint_set::DisjointSet_of_hashables;
    ///
    /// let mut ds = DisjointSet_of_hashables::new(vec![1, 2, 3, 4]);
    /// assert!(ds.union(&1, &2));
    /// assert!(!ds.union(&1, &2)); // Already in same set
    /// ```
    pub fn union(&mut self, x: &T, y: &T) -> bool {
        if let (Some(&idx_x), Some(&idx_y)) = (self.element_to_idx.get(x), self.element_to_idx.get(y)) {
            self.disjoint_set.union(idx_x, idx_y)
        } else {
            false
        }
    }

    /// Checks if x and y are in the same set.
    ///
    /// Returns `false` if either element is not in the disjoint set.
    ///
    /// # Time Complexity
    ///
    /// O(α(n)) amortized (plus O(1) expected for hash lookups)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_sets::disjoint_set::DisjointSet_of_hashables;
    ///
    /// let mut ds = DisjointSet_of_hashables::new(vec!["x", "y", "z"]);
    /// ds.union(&"x", &"y");
    /// assert!(ds.same_set(&"x", &"y"));
    /// assert!(!ds.same_set(&"x", &"z"));
    /// ```
    pub fn same_set(&mut self, x: &T, y: &T) -> bool {
        if let (Some(&idx_x), Some(&idx_y)) = (self.element_to_idx.get(x), self.element_to_idx.get(y)) {
            self.disjoint_set.same_set(idx_x, idx_y)
        } else {
            false
        }
    }

    /// Returns the size of the set containing the given element.
    ///
    /// Returns `None` if the element is not in the disjoint set.
    ///
    /// # Time Complexity
    ///
    /// O(α(n)) amortized (plus O(1) expected for hash lookup)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_sets::disjoint_set::DisjointSet_of_hashables;
    ///
    /// let mut ds = DisjointSet_of_hashables::new(vec![1, 2, 3, 4]);
    /// ds.union(&1, &2);
    /// ds.union(&2, &3);
    /// assert_eq!(ds.set_size(&1), Some(3));
    /// assert_eq!(ds.set_size(&4), Some(1));
    /// ```
    pub fn set_size(&mut self, element: &T) -> Option<usize> {
        let idx = *self.element_to_idx.get(element)?;
        Some(self.disjoint_set.set_size(idx))
    }

    /// Returns the number of disjoint sets.
    ///
    /// # Time Complexity
    ///
    /// O(n)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_sets::disjoint_set::DisjointSet_of_hashables;
    ///
    /// let mut ds = DisjointSet_of_hashables::new(vec![1, 2, 3, 4]);
    /// assert_eq!(ds.num_sets(), 4);
    /// ds.union(&1, &2);
    /// assert_eq!(ds.num_sets(), 3);
    /// ```
    pub fn num_sets(&mut self) -> usize {
        self.disjoint_set.num_sets()
    }

    /// Returns the total number of elements.
    ///
    /// # Time Complexity
    ///
    /// O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_sets::disjoint_set::DisjointSet_of_hashables;
    ///
    /// let ds = DisjointSet_of_hashables::new(vec!["a", "b", "c"]);
    /// assert_eq!(ds.num_elements(), 3);
    /// ```
    pub fn num_elements(&self) -> usize {
        self.idx_to_element.len()
    }

    /// Returns all elements in the same set as the given element.
    ///
    /// Returns `None` if the element is not in the disjoint set.
    ///
    /// # Time Complexity
    ///
    /// O(n)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_sets::disjoint_set::DisjointSet_of_hashables;
    ///
    /// let mut ds = DisjointSet_of_hashables::new(vec!["a", "b", "c", "d"]);
    /// ds.union(&"a", &"b");
    /// ds.union(&"b", &"c");
    /// let elements = ds.get_set_elements(&"a").unwrap();
    /// assert_eq!(elements.len(), 3);
    /// assert!(elements.contains(&"a"));
    /// assert!(elements.contains(&"b"));
    /// assert!(elements.contains(&"c"));
    /// ```
    pub fn get_set_elements(&mut self, element: &T) -> Option<Vec<T>> {
        let idx = *self.element_to_idx.get(element)?;
        let indices = self.disjoint_set.get_set_elements(idx);
        Some(indices.into_iter().map(|i| self.idx_to_element[i].clone()).collect())
    }

    /// Returns all disjoint sets as a vector of vectors.
    ///
    /// # Time Complexity
    ///
    /// O(n²) in worst case, but typically much faster.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_sets::disjoint_set::DisjointSet_of_hashables;
    ///
    /// let mut ds = DisjointSet_of_hashables::new(vec![1, 2, 3, 4, 5]);
    /// ds.union(&1, &2);
    /// ds.union(&3, &4);
    /// let sets = ds.get_all_sets();
    /// assert_eq!(sets.len(), 3); // {1,2}, {3,4}, {5}
    /// ```
    pub fn get_all_sets(&mut self) -> Vec<Vec<T>> {
        let index_sets = self.disjoint_set.get_all_sets();
        index_sets
            .into_iter()
            .map(|indices| {
                indices.into_iter().map(|i| self.idx_to_element[i].clone()).collect()
            })
            .collect()
    }

    /// Checks if an element is in the disjoint set.
    ///
    /// # Time Complexity
    ///
    /// O(1) expected
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_sets::disjoint_set::DisjointSet_of_hashables;
    ///
    /// let ds = DisjointSet_of_hashables::new(vec![1, 2, 3]);
    /// assert!(ds.contains(&1));
    /// assert!(!ds.contains(&4));
    /// ```
    pub fn contains(&self, element: &T) -> bool {
        self.element_to_idx.contains_key(element)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // DisjointSet_of_integers Tests
    // ========================================================================

    #[test]
    fn test_integers_new() {
        let ds = DisjointSet_of_integers::new(5);
        assert_eq!(ds.num_elements(), 5);
    }

    #[test]
    fn test_integers_singleton_sets() {
        let mut ds = DisjointSet_of_integers::new(5);
        assert_eq!(ds.num_sets(), 5);
        for i in 0..5 {
            assert_eq!(ds.find(i), i);
            assert_eq!(ds.set_size(i), 1);
        }
    }

    #[test]
    fn test_integers_union_basic() {
        let mut ds = DisjointSet_of_integers::new(5);

        assert!(ds.union(0, 1));
        assert!(ds.same_set(0, 1));
        assert_eq!(ds.num_sets(), 4);
        assert_eq!(ds.set_size(0), 2);
        assert_eq!(ds.set_size(1), 2);
    }

    #[test]
    fn test_integers_union_same_set() {
        let mut ds = DisjointSet_of_integers::new(5);
        ds.union(0, 1);

        // Unioning elements in the same set should return false
        assert!(!ds.union(0, 1));
        assert_eq!(ds.num_sets(), 4);
    }

    #[test]
    fn test_integers_union_self() {
        let mut ds = DisjointSet_of_integers::new(5);

        // Unioning an element with itself should return false
        assert!(!ds.union(0, 0));
        assert_eq!(ds.num_sets(), 5);
    }

    #[test]
    fn test_integers_transitive_union() {
        let mut ds = DisjointSet_of_integers::new(5);

        ds.union(0, 1);
        ds.union(1, 2);

        // All three should be in the same set
        assert!(ds.same_set(0, 2));
        assert!(ds.same_set(0, 1));
        assert!(ds.same_set(1, 2));
        assert_eq!(ds.set_size(0), 3);
        assert_eq!(ds.num_sets(), 3);
    }

    #[test]
    fn test_integers_multiple_sets() {
        let mut ds = DisjointSet_of_integers::new(10);

        ds.union(0, 1);
        ds.union(1, 2);
        ds.union(3, 4);
        ds.union(5, 6);
        ds.union(6, 7);

        assert_eq!(ds.num_sets(), 5); // {0,1,2}, {3,4}, {5,6,7}, {8}, {9}
        assert_eq!(ds.set_size(0), 3);
        assert_eq!(ds.set_size(3), 2);
        assert_eq!(ds.set_size(5), 3);
        assert_eq!(ds.set_size(8), 1);
        assert_eq!(ds.set_size(9), 1);
    }

    #[test]
    fn test_integers_path_compression() {
        let mut ds = DisjointSet_of_integers::new(10);

        // Create a long chain
        for i in 0..9 {
            ds.union(i, i + 1);
        }

        // All elements should be in the same set
        assert_eq!(ds.num_sets(), 1);
        for i in 0..10 {
            assert!(ds.same_set(0, i));
        }

        // Path compression should have flattened the structure
        let root = ds.find(0);
        for i in 0..10 {
            assert_eq!(ds.find(i), root);
        }
    }

    #[test]
    fn test_integers_get_set_elements() {
        let mut ds = DisjointSet_of_integers::new(5);

        ds.union(0, 1);
        ds.union(1, 2);

        let mut elements = ds.get_set_elements(0);
        elements.sort();
        assert_eq!(elements, vec![0, 1, 2]);

        let elements = ds.get_set_elements(3);
        assert_eq!(elements, vec![3]);
    }

    #[test]
    fn test_integers_get_all_sets() {
        let mut ds = DisjointSet_of_integers::new(5);

        ds.union(0, 1);
        ds.union(2, 3);

        let mut sets = ds.get_all_sets();
        assert_eq!(sets.len(), 3);

        // Sort for deterministic testing
        for set in &mut sets {
            set.sort();
        }
        sets.sort();

        assert!(sets.contains(&vec![0, 1]));
        assert!(sets.contains(&vec![2, 3]));
        assert!(sets.contains(&vec![4]));
    }

    #[test]
    fn test_integers_large_scale() {
        let n = 1000;
        let mut ds = DisjointSet_of_integers::new(n);

        // Union pairs (0,1), (2,3), (4,5), ...
        for i in (0..n).step_by(2) {
            ds.union(i, i + 1);
        }

        assert_eq!(ds.num_sets(), n / 2);

        // Union even more
        for i in (0..n / 2).step_by(2) {
            ds.union(i, i + 2);
        }

        assert!(ds.num_sets() < n / 2);
    }

    // ========================================================================
    // DisjointSet_of_hashables Tests
    // ========================================================================

    #[test]
    fn test_hashables_new() {
        let ds = DisjointSet_of_hashables::new(vec!["a", "b", "c"]);
        assert_eq!(ds.num_elements(), 3);
        assert!(ds.contains(&"a"));
        assert!(ds.contains(&"b"));
        assert!(ds.contains(&"c"));
        assert!(!ds.contains(&"d"));
    }

    #[test]
    fn test_hashables_duplicates() {
        let ds = DisjointSet_of_hashables::new(vec![1, 2, 2, 3, 3, 3]);
        assert_eq!(ds.num_elements(), 3); // Only unique elements
    }

    #[test]
    fn test_hashables_union_basic() {
        let mut ds = DisjointSet_of_hashables::new(vec!["a", "b", "c", "d"]);

        assert!(ds.union(&"a", &"b"));
        assert!(ds.same_set(&"a", &"b"));
        assert!(!ds.same_set(&"a", &"c"));
        assert_eq!(ds.num_sets(), 3);
    }

    #[test]
    fn test_hashables_union_nonexistent() {
        let mut ds = DisjointSet_of_hashables::new(vec![1, 2, 3]);

        // Unioning with non-existent element should return false
        assert!(!ds.union(&1, &5));
        assert_eq!(ds.num_sets(), 3);
    }

    #[test]
    fn test_hashables_find() {
        let mut ds = DisjointSet_of_hashables::new(vec!["x", "y", "z"]);

        ds.union(&"x", &"y");

        // Clone the results to avoid borrowing issues
        let root_x = ds.find(&"x").cloned();
        let root_y = ds.find(&"y").cloned();
        assert_eq!(root_x, root_y);

        let root_z = ds.find(&"z").cloned();
        assert_ne!(root_x, root_z);

        assert_eq!(ds.find(&"nonexistent"), None);
    }

    #[test]
    fn test_hashables_set_size() {
        let mut ds = DisjointSet_of_hashables::new(vec![1, 2, 3, 4, 5]);

        ds.union(&1, &2);
        ds.union(&2, &3);

        assert_eq!(ds.set_size(&1), Some(3));
        assert_eq!(ds.set_size(&2), Some(3));
        assert_eq!(ds.set_size(&3), Some(3));
        assert_eq!(ds.set_size(&4), Some(1));
        assert_eq!(ds.set_size(&99), None);
    }

    #[test]
    fn test_hashables_strings() {
        let mut ds = DisjointSet_of_hashables::new(vec![
            "apple".to_string(),
            "banana".to_string(),
            "cherry".to_string(),
            "date".to_string(),
        ]);

        ds.union(&"apple".to_string(), &"banana".to_string());
        ds.union(&"cherry".to_string(), &"date".to_string());

        assert!(ds.same_set(&"apple".to_string(), &"banana".to_string()));
        assert!(ds.same_set(&"cherry".to_string(), &"date".to_string()));
        assert!(!ds.same_set(&"apple".to_string(), &"cherry".to_string()));

        assert_eq!(ds.num_sets(), 2);
    }

    #[test]
    fn test_hashables_get_set_elements() {
        let mut ds = DisjointSet_of_hashables::new(vec!["a", "b", "c", "d"]);

        ds.union(&"a", &"b");
        ds.union(&"b", &"c");

        let elements = ds.get_set_elements(&"a").unwrap();
        assert_eq!(elements.len(), 3);
        assert!(elements.contains(&"a"));
        assert!(elements.contains(&"b"));
        assert!(elements.contains(&"c"));

        assert_eq!(ds.get_set_elements(&"nonexistent"), None);
    }

    #[test]
    fn test_hashables_get_all_sets() {
        let mut ds = DisjointSet_of_hashables::new(vec![1, 2, 3, 4, 5]);

        ds.union(&1, &2);
        ds.union(&3, &4);

        let sets = ds.get_all_sets();
        assert_eq!(sets.len(), 3); // {1,2}, {3,4}, {5}

        let sizes: Vec<_> = sets.iter().map(|s| s.len()).collect();
        assert!(sizes.contains(&2));
        assert!(sizes.contains(&2));
        assert!(sizes.contains(&1));
    }

    // ========================================================================
    // Performance and Stress Tests
    // ========================================================================

    #[test]
    fn test_performance_sequential_unions() {
        let n = 10000;
        let mut ds = DisjointSet_of_integers::new(n);

        // Sequential unions: 0-1, 1-2, 2-3, ...
        for i in 0..n - 1 {
            ds.union(i, i + 1);
        }

        assert_eq!(ds.num_sets(), 1);
        assert_eq!(ds.set_size(0), n);

        // All elements should be in the same set
        for i in 0..n {
            assert!(ds.same_set(0, i));
        }
    }

    #[test]
    fn test_performance_random_pattern() {
        let n = 1000;
        let mut ds = DisjointSet_of_integers::new(n);

        // Create various unions in a pseudo-random pattern
        for i in (0..n / 2).step_by(2) {
            ds.union(i, i + 1);
        }

        for i in (0..n / 4).step_by(4) {
            ds.union(i, i + 2);
        }

        let initial_sets = ds.num_sets();

        // Merge some sets
        for i in (0..n / 8).step_by(8) {
            ds.union(i, i + 4);
        }

        assert!(ds.num_sets() < initial_sets);
    }

    #[test]
    fn test_worst_case_chain() {
        let n = 100;
        let mut ds = DisjointSet_of_integers::new(n);

        // Create worst-case chain before path compression
        for i in 0..n - 1 {
            ds.union(i, i + 1);
        }

        // Path compression should handle this efficiently
        for i in 0..n {
            ds.find(i);
        }

        // Verify all are in same set
        for i in 1..n {
            assert!(ds.same_set(0, i));
        }
    }

    #[test]
    fn test_many_small_sets() {
        let n = 10000;
        let set_size = 10;
        let mut ds = DisjointSet_of_integers::new(n);

        // Create many small sets of size 10
        for i in (0..n).step_by(set_size) {
            for j in 0..set_size - 1 {
                if i + j + 1 < n {
                    ds.union(i + j, i + j + 1);
                }
            }
        }

        let num_complete_sets = n / set_size;
        assert!(ds.num_sets() <= num_complete_sets + 1);
    }

    // ========================================================================
    // Edge Cases
    // ========================================================================

    #[test]
    fn test_integers_single_element() {
        let mut ds = DisjointSet_of_integers::new(1);
        assert_eq!(ds.num_sets(), 1);
        assert_eq!(ds.find(0), 0);
        assert_eq!(ds.set_size(0), 1);
    }

    #[test]
    fn test_integers_empty() {
        let mut ds = DisjointSet_of_integers::new(0);
        assert_eq!(ds.num_sets(), 0);
        assert_eq!(ds.num_elements(), 0);
    }

    #[test]
    fn test_hashables_empty() {
        let mut ds: DisjointSet_of_hashables<i32> = DisjointSet_of_hashables::new(vec![]);
        assert_eq!(ds.num_sets(), 0);
        assert_eq!(ds.num_elements(), 0);
        assert!(!ds.contains(&1));
    }

    #[test]
    fn test_hashables_single_element() {
        let mut ds = DisjointSet_of_hashables::new(vec![42]);
        assert_eq!(ds.num_sets(), 1);
        assert_eq!(ds.set_size(&42), Some(1));
        assert!(ds.contains(&42));
    }
}
