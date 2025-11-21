//! RustMath Sets Module
//!
//! This module provides set theory structures corresponding to SageMath's `sage.sets` module.
//! It includes implementations for:
//! - Cartesian products
//! - Condition sets
//! - Disjoint sets (union-find)
//! - Families of sets
//! - Finite enumerated sets
//! - Finite set maps and functions
//! - Integer ranges (with set operations)
//! - Prime sets
//! - Real sets
//! - Set operations and utilities
//!
//! # Relationship to Rust's Standard Collections
//!
//! This crate provides mathematical set abstractions that complement Rust's standard collections:
//!
//! ## Rust's `std::collections`
//!
//! Rust provides several collection types that can represent sets:
//! - **`HashSet<T>`**: Unordered set using hash table, O(1) average lookup/insert/remove
//! - **`BTreeSet<T>`**: Ordered set using B-tree, O(log n) lookup/insert/remove
//! - **`Vec<T>`**: Can represent sets when uniqueness is guaranteed
//!
//! These are optimized for performance and general-purpose use.
//!
//! ## RustMath Sets
//!
//! This crate extends Rust's collections with mathematical abstractions:
//!
//! - **Mathematical Operations**: Union, intersection, difference, complement, symmetric difference
//!   with mathematical semantics rather than just data structure operations
//!
//! - **Infinite Sets**: Represent and reason about infinite sets like ℕ (natural numbers),
//!   ℤ (integers), ℝ (real numbers), and primes
//!
//! - **Structured Sets**: Cartesian products, condition sets (sets defined by predicates),
//!   integer ranges with steps (arithmetic progressions)
//!
//! - **Set Morphisms**: Functions between finite sets with properties like injectivity,
//!   surjectivity, and bijectivity
//!
//! - **Algebraic Structures**: Integration with RustMath's algebraic traits (Ring, Field, etc.)
//!   allowing sets to carry algebraic structure
//!
//! ## When to Use Each
//!
//! **Use `std::collections::{HashSet, BTreeSet}`** when:
//! - You need fast membership testing, insertion, or removal
//! - Working with concrete, finite sets
//! - Performance is critical
//! - You don't need mathematical operations beyond basic set operations
//!
//! **Use `rustmath-sets`** when:
//! - You need mathematical set theory operations
//! - Working with infinite sets or ranges
//! - Need to represent sets symbolically (e.g., "all primes")
//! - Require functions between sets with mathematical properties
//! - Integration with other RustMath algebraic structures
//!
//! ## Interoperability
//!
//! You can easily convert between RustMath sets and standard collections:
//!
//! ```rust
//! use std::collections::HashSet;
//! use rustmath_sets::FiniteEnumeratedSet;
//!
//! // From Vec to HashSet
//! let vec = vec![1, 2, 3, 2, 1];
//! let hash_set: HashSet<_> = vec.iter().cloned().collect();
//!
//! // From FiniteEnumeratedSet to HashSet
//! let enum_set = FiniteEnumeratedSet::new(vec![1, 2, 3, 4, 5]);
//! let hash_set: HashSet<_> = enum_set.iter().cloned().collect();
//!
//! // From HashSet to FiniteEnumeratedSet
//! let hash_set: HashSet<i32> = [1, 2, 3, 4, 5].iter().cloned().collect();
//! let enum_set: FiniteEnumeratedSet<i32> = hash_set.into_iter().collect();
//! ```
//!
//! # Set Theory Traits
//!
//! This crate defines traits for common set theory operations that can be implemented
//! by various set-like structures. See the [`traits`] module for details.

use std::collections::{HashMap, HashSet, BTreeSet};
use std::hash::Hash;
use std::fmt;

// Export the disjoint_set module with optimized union-find implementations
pub mod disjoint_set;

// Export the integer_range module with generic set operations
pub mod integer_range;

// Export the finite_set_maps module for functions between finite sets
pub mod finite_set_maps;

// ============================================================================
// Set Theory Traits
// ============================================================================

/// Trait for sets that support membership testing.
///
/// This trait represents the fundamental operation of checking whether an element
/// belongs to a set. It is the most basic property of any set-like structure.
///
/// # Examples
///
/// ```
/// use rustmath_sets::Set;
/// use std::collections::HashSet;
///
/// // HashSet implements this naturally
/// let mut set = HashSet::new();
/// set.insert(1);
/// set.insert(2);
/// set.insert(3);
///
/// assert!(set.contains(&1));
/// assert!(!set.contains(&4));
/// ```
pub trait Set<T> {
    /// Check if the set contains the given element.
    fn contains(&self, element: &T) -> bool;

    /// Check if this set is empty.
    fn is_empty(&self) -> bool {
        false // Default: most infinite sets override this
    }
}

/// Trait for finite sets that can report their cardinality.
///
/// A finite set has a well-defined number of elements. This trait extends
/// the basic `Set` trait with the ability to query the size of the set.
pub trait FiniteSet<T>: Set<T> {
    /// Get the number of elements in this set.
    fn cardinality(&self) -> usize;

    /// Check if this set is empty.
    ///
    /// Default implementation checks if cardinality is zero.
    fn is_empty(&self) -> bool {
        self.cardinality() == 0
    }
}

/// Trait for enumerable sets that can iterate over their elements.
///
/// An enumerable set can produce all of its elements in sequence. This is
/// only possible for finite sets or sets with a well-defined enumeration.
pub trait EnumerableSet<T>: Set<T> {
    /// The type of iterator over elements.
    type Iter: Iterator<Item = T>;

    /// Create an iterator over all elements in this set.
    fn iter_elements(&self) -> Self::Iter;
}

/// Trait for sets that support standard set operations.
///
/// This trait provides the fundamental operations of set theory:
/// union, intersection, and difference.
pub trait SetOperations<T>: Set<T> + Sized {
    /// Compute the union of this set with another set.
    ///
    /// The union A ∪ B contains all elements in either A or B.
    fn union(&self, other: &Self) -> Self;

    /// Compute the intersection of this set with another set.
    ///
    /// The intersection A ∩ B contains all elements in both A and B.
    fn intersection(&self, other: &Self) -> Self;

    /// Compute the set difference (relative complement).
    ///
    /// The difference A - B contains all elements in A but not in B.
    fn difference(&self, other: &Self) -> Self;

    /// Compute the symmetric difference.
    ///
    /// The symmetric difference A △ B = (A - B) ∪ (B - A) contains elements
    /// in exactly one of the two sets.
    fn symmetric_difference(&self, other: &Self) -> Self {
        let a_minus_b = self.difference(other);
        let b_minus_a = other.difference(self);
        a_minus_b.union(&b_minus_a)
    }

    /// Check if this set is a subset of another set.
    ///
    /// A ⊆ B if every element of A is also in B.
    fn is_subset(&self, _other: &Self) -> bool {
        // Default implementation not provided as it requires iteration
        unimplemented!("is_subset requires a concrete implementation")
    }

    /// Check if this set is a superset of another set.
    ///
    /// A ⊇ B if every element of B is also in A.
    fn is_superset(&self, other: &Self) -> bool {
        other.is_subset(self)
    }

    /// Check if this set is disjoint from another set.
    ///
    /// Two sets are disjoint if they have no elements in common.
    fn is_disjoint(&self, other: &Self) -> bool {
        self.intersection(other).is_empty()
    }
}

/// Trait for sets with a defined complement operation.
///
/// This trait is for sets that exist within a universal set, allowing
/// computation of the complement (all elements not in the set).
pub trait ComplementableSet<T>: Set<T> + Sized {
    /// The type representing the universal set.
    type Universe;

    /// Compute the complement of this set within the universal set.
    ///
    /// The complement Aᶜ contains all elements in the universe that are not in A.
    fn complement(&self, universe: &Self::Universe) -> Self;
}

/// Trait for sets that can be constructed from iterators.
///
/// This allows easy construction of sets from various sources.
pub trait FromElements<T>: Sized {
    /// Create a set from an iterator of elements.
    fn from_elements<I: IntoIterator<Item = T>>(iter: I) -> Self;
}

// Implement Set trait for standard collections
impl<T: Eq + Hash> Set<T> for HashSet<T> {
    fn contains(&self, element: &T) -> bool {
        HashSet::contains(self, element)
    }

    fn is_empty(&self) -> bool {
        HashSet::is_empty(self)
    }
}

impl<T: Ord> Set<T> for BTreeSet<T> {
    fn contains(&self, element: &T) -> bool {
        BTreeSet::contains(self, element)
    }

    fn is_empty(&self) -> bool {
        BTreeSet::is_empty(self)
    }
}

impl<T: PartialEq> Set<T> for Vec<T> {
    fn contains(&self, element: &T) -> bool {
        self.iter().any(|x| x == element)
    }

    fn is_empty(&self) -> bool {
        Vec::is_empty(self)
    }
}

// Implement FiniteSet trait for standard collections
impl<T: Eq + Hash> FiniteSet<T> for HashSet<T> {
    fn cardinality(&self) -> usize {
        self.len()
    }
}

impl<T: Ord> FiniteSet<T> for BTreeSet<T> {
    fn cardinality(&self) -> usize {
        self.len()
    }
}

impl<T: PartialEq> FiniteSet<T> for Vec<T> {
    fn cardinality(&self) -> usize {
        self.len()
    }
}

// ============================================================================
// Cartesian Product
// ============================================================================

/// Represents the Cartesian product of multiple sets.
/// Corresponds to sage.sets.cartesian_product.CartesianProduct
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CartesianProduct<T: Clone> {
    factors: Vec<Vec<T>>,
}

impl<T: Clone> CartesianProduct<T> {
    /// Create a new Cartesian product from a vector of sets
    pub fn new(factors: Vec<Vec<T>>) -> Self {
        Self { factors }
    }

    /// Get the number of factors (dimension)
    pub fn dimension(&self) -> usize {
        self.factors.len()
    }

    /// Get the cardinality (number of elements) if all factors are finite
    pub fn cardinality(&self) -> Option<usize> {
        if self.factors.is_empty() {
            return Some(0);
        }

        let mut result = 1usize;
        for factor in &self.factors {
            result = result.checked_mul(factor.len())?;
        }
        Some(result)
    }

    /// Check if the Cartesian product is empty
    pub fn is_empty(&self) -> bool {
        self.factors.iter().any(|f| f.is_empty())
    }

    /// Get an iterator over all tuples in the Cartesian product
    pub fn iter(&self) -> CartesianProductIterator<'_, T> {
        CartesianProductIterator::new(&self.factors)
    }
}

/// Iterator for Cartesian product elements
pub struct CartesianProductIterator<'a, T: Clone> {
    factors: &'a [Vec<T>],
    indices: Vec<usize>,
    done: bool,
}

impl<'a, T: Clone> CartesianProductIterator<'a, T> {
    fn new(factors: &'a [Vec<T>]) -> Self {
        let done = factors.is_empty() || factors.iter().any(|f| f.is_empty());
        Self {
            factors,
            indices: vec![0; factors.len()],
            done,
        }
    }
}

impl<'a, T: Clone> Iterator for CartesianProductIterator<'a, T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        // Build current tuple
        let result: Vec<T> = self.indices
            .iter()
            .enumerate()
            .map(|(i, &idx)| self.factors[i][idx].clone())
            .collect();

        // Increment indices
        let mut pos = self.indices.len();
        while pos > 0 {
            pos -= 1;
            self.indices[pos] += 1;
            if self.indices[pos] < self.factors[pos].len() {
                break;
            }
            if pos == 0 {
                self.done = true;
                break;
            }
            self.indices[pos] = 0;
        }

        Some(result)
    }
}

// ============================================================================
// Condition Set
// ============================================================================

/// A set defined by a condition (predicate).
/// Corresponds to sage.sets.condition_set.ConditionSet
#[derive(Clone)]
pub struct ConditionSet<T> {
    universe: Vec<T>,
    predicate: fn(&T) -> bool,
}

impl<T> ConditionSet<T> {
    /// Create a new condition set from a universe and a predicate
    pub fn new(universe: Vec<T>, predicate: fn(&T) -> bool) -> Self {
        Self { universe, predicate }
    }

    /// Check if an element satisfies the condition
    pub fn contains(&self, element: &T) -> bool
    where
        T: PartialEq,
    {
        self.universe.contains(element) && (self.predicate)(element)
    }

    /// Get all elements that satisfy the condition
    pub fn elements(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.universe
            .iter()
            .filter(|x| (self.predicate)(x))
            .cloned()
            .collect()
    }

    /// Get the cardinality (number of elements satisfying the condition)
    pub fn cardinality(&self) -> usize {
        self.universe.iter().filter(|x| (self.predicate)(x)).count()
    }
}

impl<T: fmt::Debug> fmt::Debug for ConditionSet<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConditionSet")
            .field("universe_size", &self.universe.len())
            .finish()
    }
}

// ============================================================================
// Disjoint Set (Union-Find)
// ============================================================================

/// Disjoint-set data structure (Union-Find).
/// Corresponds to sage.sets.disjoint_set.DisjointSet
#[derive(Debug, Clone)]
pub struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<usize>,
    size: Vec<usize>,
}

impl DisjointSet {
    /// Create a new disjoint set with n elements
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            size: vec![1; n],
        }
    }

    /// Find the root of the set containing x (with path compression)
    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    /// Union the sets containing x and y (by rank)
    pub fn union(&mut self, x: usize, y: usize) -> bool {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x == root_y {
            return false;
        }

        // Union by rank
        if self.rank[root_x] < self.rank[root_y] {
            self.parent[root_x] = root_y;
            self.size[root_y] += self.size[root_x];
        } else if self.rank[root_x] > self.rank[root_y] {
            self.parent[root_y] = root_x;
            self.size[root_x] += self.size[root_y];
        } else {
            self.parent[root_y] = root_x;
            self.size[root_x] += self.size[root_y];
            self.rank[root_x] += 1;
        }

        true
    }

    /// Check if x and y are in the same set
    pub fn same_set(&mut self, x: usize, y: usize) -> bool {
        self.find(x) == self.find(y)
    }

    /// Get the size of the set containing x
    pub fn set_size(&mut self, x: usize) -> usize {
        let root = self.find(x);
        self.size[root]
    }

    /// Get the number of disjoint sets
    pub fn num_sets(&mut self) -> usize {
        (0..self.parent.len())
            .filter(|&i| self.find(i) == i)
            .count()
    }
}

/// Disjoint-set for hashable elements.
/// Corresponds to sage.sets.disjoint_set.DisjointSet_of_hashables
#[derive(Debug, Clone)]
pub struct DisjointSetOfHashables<T: Eq + Hash + Clone> {
    element_to_idx: HashMap<T, usize>,
    disjoint_set: DisjointSet,
}

impl<T: Eq + Hash + Clone> DisjointSetOfHashables<T> {
    /// Create a new disjoint set from a collection of elements
    pub fn new(elements: Vec<T>) -> Self {
        let n = elements.len();
        let element_to_idx: HashMap<T, usize> =
            elements.into_iter().enumerate().map(|(i, e)| (e, i)).collect();

        Self {
            element_to_idx,
            disjoint_set: DisjointSet::new(n),
        }
    }

    /// Find the representative of the set containing element
    pub fn find(&mut self, element: &T) -> Option<usize> {
        self.element_to_idx.get(element).map(|&idx| self.disjoint_set.find(idx))
    }

    /// Union the sets containing x and y
    pub fn union(&mut self, x: &T, y: &T) -> bool {
        if let (Some(&idx_x), Some(&idx_y)) = (self.element_to_idx.get(x), self.element_to_idx.get(y)) {
            self.disjoint_set.union(idx_x, idx_y)
        } else {
            false
        }
    }

    /// Check if x and y are in the same set
    pub fn same_set(&mut self, x: &T, y: &T) -> bool {
        if let (Some(&idx_x), Some(&idx_y)) = (self.element_to_idx.get(x), self.element_to_idx.get(y)) {
            self.disjoint_set.same_set(idx_x, idx_y)
        } else {
            false
        }
    }
}

// ============================================================================
// Family
// ============================================================================

/// A family of mathematical objects indexed by a set.
/// Corresponds to sage.sets.family.Family
#[derive(Debug, Clone)]
pub struct Family<K, V> {
    data: HashMap<K, V>,
}

impl<K: Eq + Hash, V> Family<K, V> {
    /// Create a new empty family
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Create a family from a HashMap
    pub fn from_map(data: HashMap<K, V>) -> Self {
        Self { data }
    }

    /// Get an element by its key
    pub fn get(&self, key: &K) -> Option<&V> {
        self.data.get(key)
    }

    /// Insert an element with a key
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.data.insert(key, value)
    }

    /// Get the number of elements in the family
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the family is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get an iterator over the keys
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.data.keys()
    }

    /// Get an iterator over the values
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.data.values()
    }

    /// Get an iterator over key-value pairs
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.data.iter()
    }
}

impl<K: Eq + Hash, V> Default for Family<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

/// A finite family of objects.
/// Corresponds to sage.sets.family.FiniteFamily
pub type FiniteFamily<K, V> = Family<K, V>;

/// A lazy family where elements are computed on demand.
/// Corresponds to sage.sets.family.LazyFamily
#[derive(Clone)]
pub struct LazyFamily<K, V> {
    keys: Vec<K>,
    generator: fn(&K) -> V,
}

impl<K, V> LazyFamily<K, V> {
    /// Create a new lazy family
    pub fn new(keys: Vec<K>, generator: fn(&K) -> V) -> Self {
        Self { keys, generator }
    }

    /// Get an element by computing it with the generator
    pub fn get(&self, key: &K) -> Option<V>
    where
        K: PartialEq,
    {
        if self.keys.contains(key) {
            Some((self.generator)(key))
        } else {
            None
        }
    }

    /// Get all keys
    pub fn keys(&self) -> &[K] {
        &self.keys
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

impl<K: fmt::Debug, V> fmt::Debug for LazyFamily<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LazyFamily")
            .field("keys", &self.keys)
            .finish()
    }
}

// ============================================================================
// Finite Enumerated Set
// ============================================================================

/// A finite set with explicit enumeration of elements.
/// Corresponds to sage.sets.finite_enumerated_set.FiniteEnumeratedSet
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FiniteEnumeratedSet<T> {
    elements: Vec<T>,
}

impl<T> FiniteEnumeratedSet<T> {
    /// Create a new finite enumerated set
    pub fn new(elements: Vec<T>) -> Self {
        Self { elements }
    }

    /// Get the cardinality (number of elements)
    pub fn cardinality(&self) -> usize {
        self.elements.len()
    }

    /// Check if the set is empty
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Get an element by index
    pub fn get(&self, index: usize) -> Option<&T> {
        self.elements.get(index)
    }

    /// Get an iterator over the elements
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.elements.iter()
    }

    /// Check if an element is in the set
    pub fn contains(&self, element: &T) -> bool
    where
        T: PartialEq,
    {
        self.elements.contains(element)
    }
}

impl<T> FromIterator<T> for FiniteEnumeratedSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::new(iter.into_iter().collect())
    }
}

// ============================================================================
// Integer Range
// ============================================================================

/// A range of integers, possibly infinite.
/// Corresponds to sage.sets.integer_range.IntegerRange
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntegerRange {
    /// Empty range
    Empty,
    /// Finite range [start, end) with step
    Finite { start: i64, end: i64, step: i64 },
    /// Infinite range starting from start with step
    Infinite { start: i64, step: i64 },
}

impl IntegerRange {
    /// Create a finite integer range [start, end) with step 1
    pub fn finite(start: i64, end: i64) -> Self {
        Self::finite_with_step(start, end, 1)
    }

    /// Create a finite integer range [start, end) with custom step
    pub fn finite_with_step(start: i64, end: i64, step: i64) -> Self {
        if step == 0 || (step > 0 && start >= end) || (step < 0 && start <= end) {
            Self::Empty
        } else {
            Self::Finite { start, end, step }
        }
    }

    /// Create an infinite integer range starting from start with step 1
    pub fn infinite(start: i64) -> Self {
        Self::Infinite { start, step: 1 }
    }

    /// Create an infinite integer range with custom step
    pub fn infinite_with_step(start: i64, step: i64) -> Self {
        if step == 0 {
            Self::Empty
        } else {
            Self::Infinite { start, step }
        }
    }

    /// Check if the range is empty
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    /// Check if the range is finite
    pub fn is_finite(&self) -> bool {
        matches!(self, Self::Finite { .. } | Self::Empty)
    }

    /// Check if the range is infinite
    pub fn is_infinite(&self) -> bool {
        matches!(self, Self::Infinite { .. })
    }

    /// Get the cardinality (number of elements) if finite
    pub fn cardinality(&self) -> Option<usize> {
        match self {
            Self::Empty => Some(0),
            Self::Finite { start, end, step } => {
                let diff = (end - start).abs();
                let step_abs = step.abs();
                Some(((diff + step_abs - 1) / step_abs) as usize)
            }
            Self::Infinite { .. } => None,
        }
    }

    /// Check if the range contains a value
    pub fn contains(&self, value: i64) -> bool {
        match self {
            Self::Empty => false,
            Self::Finite { start, end, step } => {
                if *step > 0 {
                    value >= *start && value < *end && (value - start) % step == 0
                } else {
                    value <= *start && value > *end && (value - start) % step == 0
                }
            }
            Self::Infinite { start, step } => {
                if *step > 0 {
                    value >= *start && (value - start) % step == 0
                } else {
                    value <= *start && (value - start) % step == 0
                }
            }
        }
    }

    /// Get an iterator over the range (only for finite ranges)
    pub fn iter(&self) -> Option<IntegerRangeIterator> {
        match self {
            Self::Empty => Some(IntegerRangeIterator::Empty),
            Self::Finite { start, end, step } => {
                Some(IntegerRangeIterator::Finite {
                    current: *start,
                    end: *end,
                    step: *step,
                })
            }
            Self::Infinite { .. } => None,
        }
    }
}

/// Iterator for finite integer ranges
#[derive(Debug, Clone)]
pub enum IntegerRangeIterator {
    Empty,
    Finite { current: i64, end: i64, step: i64 },
}

impl Iterator for IntegerRangeIterator {
    type Item = i64;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Empty => None,
            Self::Finite { current, end, step } => {
                if (*step > 0 && *current >= *end) || (*step < 0 && *current <= *end) {
                    None
                } else {
                    let result = *current;
                    *current += *step;
                    Some(result)
                }
            }
        }
    }
}

// ============================================================================
// Non-negative Integers
// ============================================================================

/// The set of non-negative integers {0, 1, 2, ...}.
/// Corresponds to sage.sets.non_negative_integers.NonNegativeIntegers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NonNegativeIntegers;

impl NonNegativeIntegers {
    /// Check if a value is a non-negative integer
    pub fn contains(value: i64) -> bool {
        value >= 0
    }

    /// Create an integer range representing non-negative integers
    pub fn as_range() -> IntegerRange {
        IntegerRange::infinite(0)
    }
}

// ============================================================================
// Positive Integers
// ============================================================================

/// The set of positive integers {1, 2, 3, ...}.
/// Corresponds to sage.sets.positive_integers.PositiveIntegers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PositiveIntegers;

impl PositiveIntegers {
    /// Check if a value is a positive integer
    pub fn contains(value: i64) -> bool {
        value > 0
    }

    /// Create an integer range representing positive integers
    pub fn as_range() -> IntegerRange {
        IntegerRange::infinite(1)
    }
}

// ============================================================================
// Primes
// ============================================================================

/// The set of prime numbers.
/// Corresponds to sage.sets.primes.Primes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Primes;

impl Primes {
    /// Check if a number is prime
    pub fn contains(n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }

        let limit = (n as f64).sqrt() as u64 + 1;
        for i in (3..=limit).step_by(2) {
            if n % i == 0 {
                return false;
            }
        }
        true
    }

    /// Generate primes up to n using Sieve of Eratosthenes
    pub fn up_to(n: u64) -> Vec<u64> {
        if n < 2 {
            return vec![];
        }

        let mut is_prime = vec![true; (n + 1) as usize];
        is_prime[0] = false;
        is_prime[1] = false;

        for i in 2..=((n as f64).sqrt() as u64) {
            if is_prime[i as usize] {
                for j in ((i * i)..=n).step_by(i as usize) {
                    is_prime[j as usize] = false;
                }
            }
        }

        is_prime
            .iter()
            .enumerate()
            .filter_map(|(i, &prime)| if prime { Some(i as u64) } else { None })
            .collect()
    }

    /// Get the first n primes
    pub fn first_n(n: usize) -> Vec<u64> {
        let mut primes = Vec::with_capacity(n);
        let mut candidate = 2u64;

        while primes.len() < n {
            if Self::contains(candidate) {
                primes.push(candidate);
            }
            candidate += 1;
        }

        primes
    }
}

// ============================================================================
// Real Set
// ============================================================================

/// Represents a subset of the real numbers as a union of intervals.
/// Corresponds to sage.sets.real_set.RealSet
#[derive(Debug, Clone, PartialEq)]
pub struct RealSet {
    intervals: Vec<RealInterval>,
}

/// An interval in the real numbers.
/// Corresponds to sage.sets.real_set.InternalRealInterval
#[derive(Debug, Clone, PartialEq)]
pub struct RealInterval {
    pub lower: RealBound,
    pub upper: RealBound,
}

/// A bound for a real interval (can be finite or infinite, open or closed).
#[derive(Debug, Clone, PartialEq)]
pub enum RealBound {
    NegativeInfinity,
    PositiveInfinity,
    Closed(f64),
    Open(f64),
}

impl RealInterval {
    /// Create a new interval
    pub fn new(lower: RealBound, upper: RealBound) -> Self {
        Self { lower, upper }
    }

    /// Create a closed interval [a, b]
    pub fn closed(a: f64, b: f64) -> Self {
        Self {
            lower: RealBound::Closed(a),
            upper: RealBound::Closed(b),
        }
    }

    /// Create an open interval (a, b)
    pub fn open(a: f64, b: f64) -> Self {
        Self {
            lower: RealBound::Open(a),
            upper: RealBound::Open(b),
        }
    }

    /// Create a half-open interval [a, b)
    pub fn left_closed(a: f64, b: f64) -> Self {
        Self {
            lower: RealBound::Closed(a),
            upper: RealBound::Open(b),
        }
    }

    /// Create a half-open interval (a, b]
    pub fn right_closed(a: f64, b: f64) -> Self {
        Self {
            lower: RealBound::Open(a),
            upper: RealBound::Closed(b),
        }
    }

    /// Check if a value is in the interval
    pub fn contains(&self, value: f64) -> bool {
        let lower_ok = match self.lower {
            RealBound::NegativeInfinity => true,
            RealBound::Closed(a) => value >= a,
            RealBound::Open(a) => value > a,
            RealBound::PositiveInfinity => false,
        };

        let upper_ok = match self.upper {
            RealBound::PositiveInfinity => true,
            RealBound::Closed(b) => value <= b,
            RealBound::Open(b) => value < b,
            RealBound::NegativeInfinity => false,
        };

        lower_ok && upper_ok
    }

    /// Check if the interval is empty
    pub fn is_empty(&self) -> bool {
        match (&self.lower, &self.upper) {
            (RealBound::Closed(a), RealBound::Closed(b)) => a > b,
            (RealBound::Closed(a), RealBound::Open(b)) => a >= b,
            (RealBound::Open(a), RealBound::Closed(b)) => a >= b,
            (RealBound::Open(a), RealBound::Open(b)) => a >= b,
            _ => false,
        }
    }
}

impl RealSet {
    /// Create an empty real set
    pub fn empty() -> Self {
        Self {
            intervals: Vec::new(),
        }
    }

    /// Create a real set from a single interval
    pub fn from_interval(interval: RealInterval) -> Self {
        if interval.is_empty() {
            Self::empty()
        } else {
            Self {
                intervals: vec![interval],
            }
        }
    }

    /// Create a real set from multiple intervals
    pub fn from_intervals(intervals: Vec<RealInterval>) -> Self {
        let intervals: Vec<_> = intervals.into_iter().filter(|i| !i.is_empty()).collect();
        Self { intervals }
    }

    /// Check if a value is in the set
    pub fn contains(&self, value: f64) -> bool {
        self.intervals.iter().any(|interval| interval.contains(value))
    }

    /// Check if the set is empty
    pub fn is_empty(&self) -> bool {
        self.intervals.is_empty()
    }

    /// Get the number of intervals
    pub fn num_intervals(&self) -> usize {
        self.intervals.len()
    }

    /// Create the union with another real set
    pub fn union(&self, other: &RealSet) -> RealSet {
        let mut intervals = self.intervals.clone();
        intervals.extend(other.intervals.clone());
        RealSet::from_intervals(intervals)
    }
}

// ============================================================================
// Set Operations
// ============================================================================

/// Compute the union of two vectors (treating them as sets)
pub fn set_union<T: Eq + Hash + Clone>(a: &[T], b: &[T]) -> Vec<T> {
    let mut set: HashSet<T> = a.iter().cloned().collect();
    set.extend(b.iter().cloned());
    set.into_iter().collect()
}

/// Compute the intersection of two vectors (treating them as sets)
pub fn set_intersection<T: Eq + Hash + Clone>(a: &[T], b: &[T]) -> Vec<T> {
    let set_b: HashSet<&T> = b.iter().collect();
    a.iter()
        .filter(|x| set_b.contains(x))
        .cloned()
        .collect()
}

/// Compute the difference of two vectors (a - b)
pub fn set_difference<T: Eq + Hash + Clone>(a: &[T], b: &[T]) -> Vec<T> {
    let set_b: HashSet<&T> = b.iter().collect();
    a.iter()
        .filter(|x| !set_b.contains(x))
        .cloned()
        .collect()
}

/// Compute the symmetric difference of two vectors
pub fn set_symmetric_difference<T: Eq + Hash + Clone>(a: &[T], b: &[T]) -> Vec<T> {
    let set_a: HashSet<T> = a.iter().cloned().collect();
    let set_b: HashSet<T> = b.iter().cloned().collect();

    set_a.symmetric_difference(&set_b).cloned().collect()
}

// ============================================================================
// Trait Implementations for RustMath Set Types
// ============================================================================

impl<T: PartialEq> Set<T> for FiniteEnumeratedSet<T> {
    fn contains(&self, element: &T) -> bool {
        self.contains(element)
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }
}

impl<T: PartialEq> FiniteSet<T> for FiniteEnumeratedSet<T> {
    fn cardinality(&self) -> usize {
        self.cardinality()
    }
}

impl<T> FromElements<T> for FiniteEnumeratedSet<T> {
    fn from_elements<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::new(iter.into_iter().collect())
    }
}

// Implement Set trait for IntegerRange
impl Set<i64> for IntegerRange {
    fn contains(&self, element: &i64) -> bool {
        self.contains(*element)
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }
}

// Implement FiniteSet trait for finite IntegerRange only
// Note: This is conditional on the range being finite
impl FiniteSet<i64> for IntegerRange {
    fn cardinality(&self) -> usize {
        self.cardinality().expect("IntegerRange must be finite for cardinality")
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }
}

// Implement Set trait for Primes
impl Set<u64> for Primes {
    fn contains(&self, element: &u64) -> bool {
        Primes::contains(*element)
    }

    fn is_empty(&self) -> bool {
        false // The set of primes is infinite
    }
}

// Implement Set trait for PositiveIntegers
impl Set<i64> for PositiveIntegers {
    fn contains(&self, element: &i64) -> bool {
        PositiveIntegers::contains(*element)
    }

    fn is_empty(&self) -> bool {
        false // Infinite set
    }
}

// Implement Set trait for NonNegativeIntegers
impl Set<i64> for NonNegativeIntegers {
    fn contains(&self, element: &i64) -> bool {
        NonNegativeIntegers::contains(*element)
    }

    fn is_empty(&self) -> bool {
        false // Infinite set
    }
}

// Implement Set trait for RealSet
impl Set<f64> for RealSet {
    fn contains(&self, element: &f64) -> bool {
        self.contains(*element)
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }
}

// Implement Set trait for ConditionSet
impl<T: PartialEq> Set<T> for ConditionSet<T> {
    fn contains(&self, element: &T) -> bool {
        self.contains(element)
    }

    fn is_empty(&self) -> bool {
        self.cardinality() == 0
    }
}

impl<T: PartialEq> FiniteSet<T> for ConditionSet<T> {
    fn cardinality(&self) -> usize {
        self.cardinality()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Cartesian Product Tests
    #[test]
    fn test_cartesian_product_basic() {
        let set1 = vec![1, 2];
        let set2 = vec![3, 4];
        let product = CartesianProduct::new(vec![set1, set2]);

        assert_eq!(product.dimension(), 2);
        assert_eq!(product.cardinality(), Some(4));
        assert!(!product.is_empty());

        let elements: Vec<_> = product.iter().collect();
        assert_eq!(elements.len(), 4);
        assert!(elements.contains(&vec![1, 3]));
        assert!(elements.contains(&vec![1, 4]));
        assert!(elements.contains(&vec![2, 3]));
        assert!(elements.contains(&vec![2, 4]));
    }

    #[test]
    fn test_cartesian_product_empty() {
        let set1 = vec![1, 2];
        let set2: Vec<i32> = vec![];
        let product = CartesianProduct::new(vec![set1, set2]);

        assert!(product.is_empty());
        assert_eq!(product.cardinality(), Some(0));
        assert_eq!(product.iter().count(), 0);
    }

    #[test]
    fn test_cartesian_product_three_factors() {
        let set1 = vec![1, 2];
        let set2 = vec![3];
        let set3 = vec![4, 5];
        let product = CartesianProduct::new(vec![set1, set2, set3]);

        assert_eq!(product.dimension(), 3);
        assert_eq!(product.cardinality(), Some(4));

        let elements: Vec<_> = product.iter().collect();
        assert_eq!(elements.len(), 4);
        assert!(elements.contains(&vec![1, 3, 4]));
        assert!(elements.contains(&vec![1, 3, 5]));
        assert!(elements.contains(&vec![2, 3, 4]));
        assert!(elements.contains(&vec![2, 3, 5]));
    }

    // Condition Set Tests
    #[test]
    fn test_condition_set() {
        let universe = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let is_even = |x: &i32| x % 2 == 0;
        let evens = ConditionSet::new(universe, is_even);

        assert!(evens.contains(&2));
        assert!(evens.contains(&4));
        assert!(!evens.contains(&1));
        assert!(!evens.contains(&3));

        assert_eq!(evens.cardinality(), 5);

        let elements = evens.elements();
        assert_eq!(elements, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_condition_set_empty() {
        let universe = vec![1, 3, 5, 7, 9];
        let is_even = |x: &i32| x % 2 == 0;
        let evens = ConditionSet::new(universe, is_even);

        assert_eq!(evens.cardinality(), 0);
        assert_eq!(evens.elements().len(), 0);
    }

    // Disjoint Set Tests
    #[test]
    fn test_disjoint_set_basic() {
        let mut ds = DisjointSet::new(5);

        assert!(!ds.same_set(0, 1));
        assert_eq!(ds.num_sets(), 5);

        ds.union(0, 1);
        assert!(ds.same_set(0, 1));
        assert_eq!(ds.num_sets(), 4);
        assert_eq!(ds.set_size(0), 2);

        ds.union(2, 3);
        assert!(ds.same_set(2, 3));
        assert!(!ds.same_set(0, 2));
        assert_eq!(ds.num_sets(), 3);

        ds.union(0, 2);
        assert!(ds.same_set(0, 3));
        assert!(ds.same_set(1, 2));
        assert_eq!(ds.num_sets(), 2);
        assert_eq!(ds.set_size(0), 4);
    }

    #[test]
    fn test_disjoint_set_of_hashables() {
        let mut ds = DisjointSetOfHashables::new(vec!["a", "b", "c", "d"]);

        assert!(!ds.same_set(&"a", &"b"));

        ds.union(&"a", &"b");
        assert!(ds.same_set(&"a", &"b"));

        ds.union(&"c", &"d");
        assert!(ds.same_set(&"c", &"d"));
        assert!(!ds.same_set(&"a", &"c"));

        ds.union(&"a", &"c");
        assert!(ds.same_set(&"a", &"d"));
        assert!(ds.same_set(&"b", &"c"));
    }

    // Family Tests
    #[test]
    fn test_family() {
        let mut family = Family::new();
        assert!(family.is_empty());

        family.insert("x", 1);
        family.insert("y", 2);
        family.insert("z", 3);

        assert_eq!(family.len(), 3);
        assert_eq!(family.get(&"x"), Some(&1));
        assert_eq!(family.get(&"y"), Some(&2));
        assert_eq!(family.get(&"z"), Some(&3));
        assert_eq!(family.get(&"w"), None);

        let keys: Vec<_> = family.keys().cloned().collect();
        assert_eq!(keys.len(), 3);
    }

    #[test]
    fn test_lazy_family() {
        let keys = vec![1, 2, 3, 4, 5];
        let square = |x: &i32| x * x;
        let family = LazyFamily::new(keys, square);

        assert_eq!(family.len(), 5);
        assert_eq!(family.get(&3), Some(9));
        assert_eq!(family.get(&4), Some(16));
        assert_eq!(family.get(&10), None);
    }

    // Finite Enumerated Set Tests
    #[test]
    fn test_finite_enumerated_set() {
        let set = FiniteEnumeratedSet::new(vec![1, 2, 3, 4, 5]);

        assert_eq!(set.cardinality(), 5);
        assert!(!set.is_empty());
        assert!(set.contains(&3));
        assert!(!set.contains(&10));

        assert_eq!(set.get(0), Some(&1));
        assert_eq!(set.get(4), Some(&5));
        assert_eq!(set.get(5), None);

        let sum: i32 = set.iter().sum();
        assert_eq!(sum, 15);
    }

    #[test]
    fn test_finite_enumerated_set_from_iter() {
        let set: FiniteEnumeratedSet<i32> = (1..=5).collect();
        assert_eq!(set.cardinality(), 5);
    }

    // Integer Range Tests
    #[test]
    fn test_integer_range_finite() {
        let range = IntegerRange::finite(0, 5);

        assert!(!range.is_empty());
        assert!(range.is_finite());
        assert!(!range.is_infinite());
        assert_eq!(range.cardinality(), Some(5));

        assert!(range.contains(0));
        assert!(range.contains(4));
        assert!(!range.contains(5));
        assert!(!range.contains(-1));

        let elements: Vec<_> = range.iter().unwrap().collect();
        assert_eq!(elements, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_integer_range_with_step() {
        let range = IntegerRange::finite_with_step(0, 10, 2);

        assert_eq!(range.cardinality(), Some(5));

        assert!(range.contains(0));
        assert!(range.contains(2));
        assert!(range.contains(8));
        assert!(!range.contains(1));
        assert!(!range.contains(10));

        let elements: Vec<_> = range.iter().unwrap().collect();
        assert_eq!(elements, vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_integer_range_negative_step() {
        let range = IntegerRange::finite_with_step(10, 0, -2);

        assert_eq!(range.cardinality(), Some(5));

        let elements: Vec<_> = range.iter().unwrap().collect();
        assert_eq!(elements, vec![10, 8, 6, 4, 2]);
    }

    #[test]
    fn test_integer_range_empty() {
        let range = IntegerRange::finite(5, 0);
        assert!(range.is_empty());
        assert_eq!(range.cardinality(), Some(0));
    }

    #[test]
    fn test_integer_range_infinite() {
        let range = IntegerRange::infinite(0);

        assert!(!range.is_empty());
        assert!(!range.is_finite());
        assert!(range.is_infinite());
        assert_eq!(range.cardinality(), None);

        assert!(range.contains(0));
        assert!(range.contains(1000));
        assert!(!range.contains(-1));
    }

    // Non-negative Integers Tests
    #[test]
    fn test_non_negative_integers() {
        assert!(NonNegativeIntegers::contains(0));
        assert!(NonNegativeIntegers::contains(1));
        assert!(NonNegativeIntegers::contains(1000));
        assert!(!NonNegativeIntegers::contains(-1));
        assert!(!NonNegativeIntegers::contains(-100));
    }

    // Positive Integers Tests
    #[test]
    fn test_positive_integers() {
        assert!(!PositiveIntegers::contains(0));
        assert!(PositiveIntegers::contains(1));
        assert!(PositiveIntegers::contains(1000));
        assert!(!PositiveIntegers::contains(-1));
        assert!(!PositiveIntegers::contains(-100));
    }

    // Primes Tests
    #[test]
    fn test_primes_contains() {
        assert!(!Primes::contains(0));
        assert!(!Primes::contains(1));
        assert!(Primes::contains(2));
        assert!(Primes::contains(3));
        assert!(!Primes::contains(4));
        assert!(Primes::contains(5));
        assert!(!Primes::contains(6));
        assert!(Primes::contains(7));
        assert!(!Primes::contains(8));
        assert!(!Primes::contains(9));
        assert!(!Primes::contains(10));
        assert!(Primes::contains(11));
        assert!(Primes::contains(13));
        assert!(Primes::contains(17));
        assert!(Primes::contains(19));
        assert!(Primes::contains(23));
    }

    #[test]
    fn test_primes_up_to() {
        let primes = Primes::up_to(20);
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19]);

        let primes = Primes::up_to(10);
        assert_eq!(primes, vec![2, 3, 5, 7]);

        let primes = Primes::up_to(1);
        assert_eq!(primes, vec![]);
    }

    #[test]
    fn test_primes_first_n() {
        let primes = Primes::first_n(5);
        assert_eq!(primes, vec![2, 3, 5, 7, 11]);

        let primes = Primes::first_n(10);
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);

        let primes = Primes::first_n(0);
        assert_eq!(primes, vec![]);
    }

    // Real Interval Tests
    #[test]
    fn test_real_interval_closed() {
        let interval = RealInterval::closed(0.0, 1.0);

        assert!(interval.contains(0.0));
        assert!(interval.contains(0.5));
        assert!(interval.contains(1.0));
        assert!(!interval.contains(-0.1));
        assert!(!interval.contains(1.1));
        assert!(!interval.is_empty());
    }

    #[test]
    fn test_real_interval_open() {
        let interval = RealInterval::open(0.0, 1.0);

        assert!(!interval.contains(0.0));
        assert!(interval.contains(0.5));
        assert!(!interval.contains(1.0));
        assert!(!interval.contains(-0.1));
        assert!(!interval.contains(1.1));
        assert!(!interval.is_empty());
    }

    #[test]
    fn test_real_interval_half_open() {
        let interval = RealInterval::left_closed(0.0, 1.0);

        assert!(interval.contains(0.0));
        assert!(interval.contains(0.5));
        assert!(!interval.contains(1.0));

        let interval = RealInterval::right_closed(0.0, 1.0);

        assert!(!interval.contains(0.0));
        assert!(interval.contains(0.5));
        assert!(interval.contains(1.0));
    }

    #[test]
    fn test_real_interval_empty() {
        let interval = RealInterval::closed(1.0, 0.0);
        assert!(interval.is_empty());

        let interval = RealInterval::open(1.0, 1.0);
        assert!(interval.is_empty());
    }

    #[test]
    fn test_real_interval_infinite() {
        let interval = RealInterval::new(RealBound::NegativeInfinity, RealBound::PositiveInfinity);

        assert!(interval.contains(0.0));
        assert!(interval.contains(-1000.0));
        assert!(interval.contains(1000.0));
        assert!(!interval.is_empty());
    }

    // Real Set Tests
    #[test]
    fn test_real_set_single_interval() {
        let interval = RealInterval::closed(0.0, 1.0);
        let set = RealSet::from_interval(interval);

        assert!(!set.is_empty());
        assert_eq!(set.num_intervals(), 1);
        assert!(set.contains(0.5));
        assert!(!set.contains(1.5));
    }

    #[test]
    fn test_real_set_multiple_intervals() {
        let intervals = vec![
            RealInterval::closed(0.0, 1.0),
            RealInterval::closed(2.0, 3.0),
        ];
        let set = RealSet::from_intervals(intervals);

        assert_eq!(set.num_intervals(), 2);
        assert!(set.contains(0.5));
        assert!(set.contains(2.5));
        assert!(!set.contains(1.5));
    }

    #[test]
    fn test_real_set_union() {
        let set1 = RealSet::from_interval(RealInterval::closed(0.0, 1.0));
        let set2 = RealSet::from_interval(RealInterval::closed(2.0, 3.0));
        let union = set1.union(&set2);

        assert_eq!(union.num_intervals(), 2);
        assert!(union.contains(0.5));
        assert!(union.contains(2.5));
        assert!(!union.contains(1.5));
    }

    #[test]
    fn test_real_set_empty() {
        let set = RealSet::empty();
        assert!(set.is_empty());
        assert_eq!(set.num_intervals(), 0);
        assert!(!set.contains(0.0));
    }

    // Set Operations Tests
    #[test]
    fn test_set_union() {
        let a = vec![1, 2, 3];
        let b = vec![3, 4, 5];
        let union = set_union(&a, &b);

        assert_eq!(union.len(), 5);
        assert!(union.contains(&1));
        assert!(union.contains(&5));
    }

    #[test]
    fn test_set_intersection() {
        let a = vec![1, 2, 3, 4];
        let b = vec![3, 4, 5, 6];
        let intersection = set_intersection(&a, &b);

        assert_eq!(intersection, vec![3, 4]);
    }

    #[test]
    fn test_set_difference() {
        let a = vec![1, 2, 3, 4];
        let b = vec![3, 4, 5, 6];
        let difference = set_difference(&a, &b);

        assert_eq!(difference, vec![1, 2]);
    }

    #[test]
    fn test_set_symmetric_difference() {
        let a = vec![1, 2, 3, 4];
        let b = vec![3, 4, 5, 6];
        let sym_diff = set_symmetric_difference(&a, &b);

        assert_eq!(sym_diff.len(), 4);
        assert!(sym_diff.contains(&1));
        assert!(sym_diff.contains(&2));
        assert!(sym_diff.contains(&5));
        assert!(sym_diff.contains(&6));
        assert!(!sym_diff.contains(&3));
        assert!(!sym_diff.contains(&4));
    }

    // Additional edge case tests
    #[test]
    fn test_cartesian_product_single_factor() {
        let product = CartesianProduct::new(vec![vec![1, 2, 3]]);
        assert_eq!(product.cardinality(), Some(3));

        let elements: Vec<_> = product.iter().collect();
        assert_eq!(elements, vec![vec![1], vec![2], vec![3]]);
    }

    #[test]
    fn test_disjoint_set_self_union() {
        let mut ds = DisjointSet::new(3);
        assert!(!ds.union(0, 0));
        assert_eq!(ds.num_sets(), 3);
    }

    #[test]
    fn test_integer_range_single_element() {
        let range = IntegerRange::finite(5, 6);
        assert_eq!(range.cardinality(), Some(1));
        assert!(range.contains(5));
        assert!(!range.contains(6));

        let elements: Vec<_> = range.iter().unwrap().collect();
        assert_eq!(elements, vec![5]);
    }

    #[test]
    fn test_primes_large_prime() {
        assert!(Primes::contains(97));
        assert!(Primes::contains(101));
        assert!(!Primes::contains(100));
        assert!(!Primes::contains(102));
    }

    #[test]
    fn test_family_update() {
        let mut family = Family::new();
        family.insert("key", 1);
        assert_eq!(family.get(&"key"), Some(&1));

        family.insert("key", 2);
        assert_eq!(family.get(&"key"), Some(&2));
        assert_eq!(family.len(), 1);
    }

    #[test]
    fn test_condition_set_all_match() {
        let universe = vec![2, 4, 6, 8, 10];
        let is_even = |x: &i32| x % 2 == 0;
        let evens = ConditionSet::new(universe.clone(), is_even);

        assert_eq!(evens.cardinality(), universe.len());
        assert_eq!(evens.elements(), universe);
    }

    #[test]
    fn test_recursively_enumerated_set_simulation() {
        // Simulate a recursively enumerated set using FiniteEnumeratedSet
        // Start with seeds and apply a function iteratively
        let mut current = vec![1];
        let mut seen = HashSet::new();
        seen.insert(1);

        // Generate numbers by doubling and adding 1, up to 100
        for _ in 0..10 {
            let mut next = vec![];
            for &x in &current {
                let doubled = x * 2;
                let plus_one = x * 2 + 1;

                if doubled <= 100 && !seen.contains(&doubled) {
                    next.push(doubled);
                    seen.insert(doubled);
                }
                if plus_one <= 100 && !seen.contains(&plus_one) {
                    next.push(plus_one);
                    seen.insert(plus_one);
                }
            }
            if next.is_empty() {
                break;
            }
            current = next;
        }

        let result: Vec<_> = seen.into_iter().collect();
        assert!(result.len() > 0);
        assert!(result.contains(&1));
        assert!(result.contains(&2));
        assert!(result.contains(&3));
    }

    #[test]
    fn test_totally_ordered_finite_set() {
        // Simulate TotallyOrderedFiniteSet using BTreeSet
        let mut ordered_set = BTreeSet::new();
        ordered_set.insert(3);
        ordered_set.insert(1);
        ordered_set.insert(4);
        ordered_set.insert(1); // Duplicate, should not be added
        ordered_set.insert(5);

        let elements: Vec<_> = ordered_set.iter().cloned().collect();
        assert_eq!(elements, vec![1, 3, 4, 5]);
        assert_eq!(ordered_set.len(), 4);
    }

    #[test]
    fn test_finite_set_maps() {
        // Simulate FiniteSetMaps functionality
        let domain = vec![1, 2, 3];
        let codomain = vec!["a", "b"];

        // Count number of possible functions from domain to codomain
        let num_functions = codomain.len().pow(domain.len() as u32);
        assert_eq!(num_functions, 8);

        // Example map
        let map: HashMap<i32, &str> = [(1, "a"), (2, "b"), (3, "a")].into_iter().collect();
        assert_eq!(map.get(&1), Some(&"a"));
        assert_eq!(map.get(&2), Some(&"b"));
        assert_eq!(map.get(&3), Some(&"a"));
    }

    #[test]
    fn test_disjoint_union_enumerated_sets() {
        // Simulate DisjointUnionEnumeratedSets
        let set1 = vec![("A", 1), ("A", 2), ("A", 3)];
        let set2 = vec![("B", 1), ("B", 2)];

        let mut union = set1.clone();
        union.extend(set2.clone());

        assert_eq!(union.len(), 5);
        assert!(union.contains(&("A", 1)));
        assert!(union.contains(&("B", 1)));
        // Elements from different sets are distinguishable
        assert_ne!(("A", 1), ("B", 1));
    }

    #[test]
    fn test_set_from_iterator_simulation() {
        // Simulate EnumeratedSetFromIterator
        let generator = |n: usize| n * n;
        let squares: Vec<usize> = (0..10).map(generator).collect();

        assert_eq!(squares.len(), 10);
        assert_eq!(squares[0], 0);
        assert_eq!(squares[5], 25);
        assert_eq!(squares[9], 81);
    }
}
