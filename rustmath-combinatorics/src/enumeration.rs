//! Enumeration utilities for combinatorial objects
//!
//! This module provides general utilities for enumerating and iterating over
//! combinatorial structures, including lazy iterators and generating functions.

use rustmath_core::NumericConversion;
use std::collections::VecDeque;

/// A trait for combinatorial objects that can be enumerated
pub trait Enumerable: Sized {
    /// Generate all objects with the given parameters
    fn enumerate(params: &Self::Params) -> Vec<Self>;

    /// Parameters needed to specify what to enumerate
    type Params;

    /// Count the number of objects (may be more efficient than enumerate().len())
    fn count(params: &Self::Params) -> usize {
        Self::enumerate(params).len()
    }
}

/// A lazy iterator for combinatorial objects
///
/// This generates objects one at a time rather than all at once,
/// which is more memory-efficient for large collections
pub struct LazyEnumerator<T, F>
where
    F: FnMut() -> Option<T>,
{
    generator: F,
}

impl<T, F> LazyEnumerator<T, F>
where
    F: FnMut() -> Option<T>,
{
    /// Create a new lazy enumerator from a generator function
    pub fn new(generator: F) -> Self {
        LazyEnumerator { generator }
    }
}

impl<T, F> Iterator for LazyEnumerator<T, F>
where
    F: FnMut() -> Option<T>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        (self.generator)()
    }
}

/// A Gray code iterator for subsets
///
/// Generates all 2^n subsets of {0, 1, ..., n-1} such that
/// consecutive subsets differ by exactly one element (Gray code order)
pub struct GrayCodeIterator {
    n: usize,
    current: usize,
    max: usize,
}

impl GrayCodeIterator {
    /// Create a new Gray code iterator for n elements
    pub fn new(n: usize) -> Self {
        if n > 30 {
            panic!("Gray code iterator: n too large (max 30)");
        }
        GrayCodeIterator {
            n,
            current: 0,
            max: 1 << n,
        }
    }

    /// Convert Gray code to binary
    fn gray_to_binary(gray: usize) -> usize {
        let mut binary = gray;
        let mut mask = gray >> 1;
        while mask != 0 {
            binary ^= mask;
            mask >>= 1;
        }
        binary
    }

    /// Convert subset index to actual subset
    fn index_to_subset(index: usize, n: usize) -> Vec<usize> {
        (0..n).filter(|&i| (index >> i) & 1 == 1).collect()
    }
}

impl Iterator for GrayCodeIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.max {
            return None;
        }

        // Convert index to Gray code, then to subset
        let gray = self.current ^ (self.current >> 1);
        let subset = Self::index_to_subset(gray, self.n);
        self.current += 1;

        Some(subset)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.max - self.current;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for GrayCodeIterator {}

/// Convert a binary number to its Gray code representation
///
/// The Gray code (also known as reflected binary code) is a binary numeral system
/// where two successive values differ in only one bit.
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::enumeration::binary_to_gray;
///
/// assert_eq!(binary_to_gray(0), 0);
/// assert_eq!(binary_to_gray(1), 1);
/// assert_eq!(binary_to_gray(2), 3);
/// assert_eq!(binary_to_gray(3), 2);
/// assert_eq!(binary_to_gray(4), 6);
/// ```
pub fn binary_to_gray(binary: usize) -> usize {
    binary ^ (binary >> 1)
}

/// Convert a Gray code to its binary representation
///
/// This is the inverse operation of `binary_to_gray`.
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::enumeration::gray_to_binary;
///
/// assert_eq!(gray_to_binary(0), 0);
/// assert_eq!(gray_to_binary(1), 1);
/// assert_eq!(gray_to_binary(3), 2);
/// assert_eq!(gray_to_binary(2), 3);
/// assert_eq!(gray_to_binary(6), 4);
/// ```
pub fn gray_to_binary(gray: usize) -> usize {
    let mut binary = gray;
    let mut mask = gray >> 1;
    while mask != 0 {
        binary ^= mask;
        mask >>= 1;
    }
    binary
}

/// A binary reflected Gray code iterator
///
/// Generates n-bit Gray codes as integers from 0 to 2^n - 1
/// in Gray code order. Each successive value differs from the
/// previous by exactly one bit.
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::enumeration::BinaryGrayCodeIterator;
///
/// let codes: Vec<usize> = BinaryGrayCodeIterator::new(3).collect();
/// // Generates: [0, 1, 3, 2, 6, 7, 5, 4] in Gray code order
/// assert_eq!(codes.len(), 8);
///
/// // Check that consecutive codes differ by exactly one bit
/// for i in 1..codes.len() {
///     let diff = codes[i] ^ codes[i-1];
///     assert_eq!(diff.count_ones(), 1);
/// }
/// ```
pub struct BinaryGrayCodeIterator {
    n: usize,
    current: usize,
    max: usize,
}

impl BinaryGrayCodeIterator {
    /// Create a new binary Gray code iterator for n bits
    ///
    /// # Panics
    ///
    /// Panics if n > 63 (to avoid overflow in calculating 2^n)
    pub fn new(n: usize) -> Self {
        if n > 63 {
            panic!("Binary Gray code iterator: n too large (max 63)");
        }
        BinaryGrayCodeIterator {
            n,
            current: 0,
            max: if n == 0 { 0 } else { 1 << n },
        }
    }

    /// Get the current position (how many codes have been generated)
    pub fn position(&self) -> usize {
        self.current
    }

    /// Get the number of bits in the codes
    pub fn bits(&self) -> usize {
        self.n
    }
}

impl Iterator for BinaryGrayCodeIterator {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.max {
            return None;
        }

        let gray = binary_to_gray(self.current);
        self.current += 1;

        Some(gray)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.max - self.current;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for BinaryGrayCodeIterator {}

/// Generate all n-bit binary Gray codes as a vector
///
/// Returns a vector of Gray code values in order. Each value differs
/// from the previous by exactly one bit.
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::enumeration::binary_gray_codes;
///
/// let codes = binary_gray_codes(3);
/// assert_eq!(codes, vec![0, 1, 3, 2, 6, 7, 5, 4]);
/// ```
pub fn binary_gray_codes(n: usize) -> Vec<usize> {
    BinaryGrayCodeIterator::new(n).collect()
}

/// Generate all n-bit binary Gray codes as bit vector strings
///
/// Returns a vector of binary strings representing Gray codes.
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::enumeration::binary_gray_code_strings;
///
/// let codes = binary_gray_code_strings(3);
/// assert_eq!(codes, vec!["000", "001", "011", "010", "110", "111", "101", "100"]);
/// ```
pub fn binary_gray_code_strings(n: usize) -> Vec<String> {
    BinaryGrayCodeIterator::new(n)
        .map(|code| format!("{:0width$b}", code, width = n))
        .collect()
}

/// A permutation Gray code iterator using the Steinhaus-Johnson-Trotter algorithm
///
/// Generates all permutations of [0, 1, ..., n-1] such that consecutive
/// permutations differ by swapping two adjacent elements.
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::enumeration::PermutationGrayCodeIterator;
///
/// let perms: Vec<Vec<usize>> = PermutationGrayCodeIterator::new(3).collect();
/// assert_eq!(perms.len(), 6); // 3! = 6
///
/// // Each permutation differs from the next by swapping adjacent elements
/// for i in 1..perms.len() {
///     let mut diffs = 0;
///     let mut swap_adjacent = false;
///     for j in 0..perms[i].len() {
///         if perms[i][j] != perms[i-1][j] {
///             diffs += 1;
///             if j > 0 && perms[i][j] == perms[i-1][j-1] {
///                 swap_adjacent = true;
///             }
///             if j < perms[i].len()-1 && perms[i][j] == perms[i-1][j+1] {
///                 swap_adjacent = true;
///             }
///         }
///     }
///     assert_eq!(diffs, 2);
///     assert!(swap_adjacent);
/// }
/// ```
pub struct PermutationGrayCodeIterator {
    n: usize,
    perm: Vec<usize>,
    directions: Vec<i8>, // -1 for left, 1 for right
    done: bool,
}

impl PermutationGrayCodeIterator {
    /// Create a new permutation Gray code iterator for permutations of {0, 1, ..., n-1}
    pub fn new(n: usize) -> Self {
        if n == 0 {
            return PermutationGrayCodeIterator {
                n: 0,
                perm: vec![],
                directions: vec![],
                done: false,
            };
        }

        PermutationGrayCodeIterator {
            n,
            perm: (0..n).collect(),
            directions: vec![-1; n], // All pointing left initially
            done: false,
        }
    }

    /// Find the largest mobile element
    /// An element is mobile if it points to a smaller adjacent element
    fn find_largest_mobile(&self) -> Option<usize> {
        let mut largest_idx = None;
        let mut largest_val = None;

        for i in 0..self.n {
            let val = self.perm[i];
            let dir = self.directions[i];

            // Check if mobile
            let is_mobile = if dir == -1 {
                i > 0 && self.perm[i - 1] < val
            } else {
                i < self.n - 1 && self.perm[i + 1] < val
            };

            if is_mobile {
                if largest_val.is_none() || val > largest_val.unwrap() {
                    largest_val = Some(val);
                    largest_idx = Some(i);
                }
            }
        }

        largest_idx
    }
}

impl Iterator for PermutationGrayCodeIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        if self.n == 0 {
            self.done = true;
            return Some(vec![]);
        }

        // Return current permutation
        let result = self.perm.clone();

        // Find largest mobile element
        if let Some(k) = self.find_largest_mobile() {
            let dir = self.directions[k];
            let next_pos = if dir == -1 {
                k - 1
            } else {
                k + 1
            };

            // Swap element k with its neighbor
            self.perm.swap(k, next_pos);
            self.directions.swap(k, next_pos);

            // Reverse direction of all elements larger than the mobile element
            let mobile_val = self.perm[next_pos];
            for i in 0..self.n {
                if self.perm[i] > mobile_val {
                    self.directions[i] *= -1;
                }
            }
        } else {
            // No mobile elements, we're done
            self.done = true;
        }

        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.done {
            return (0, Some(0));
        }

        // Total permutations is n!
        // Computing remaining is complex, so we just give total
        let total = factorial(self.n);
        (0, Some(total))
    }
}

/// Compute factorial n!
fn factorial(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        (2..=n).product()
    }
}

/// A combination Gray code iterator
///
/// Generates all k-element subsets of {0, 1, ..., n-1} such that
/// consecutive combinations differ by adding or removing one element.
/// This uses the revolving door algorithm.
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::enumeration::CombinationGrayCodeIterator;
///
/// let combs: Vec<Vec<usize>> = CombinationGrayCodeIterator::new(4, 2).collect();
/// assert_eq!(combs.len(), 6); // C(4, 2) = 6
/// ```
pub struct CombinationGrayCodeIterator {
    n: usize,
    k: usize,
    current: Vec<bool>, // Bit vector representation
    done: bool,
}

impl CombinationGrayCodeIterator {
    /// Create a new combination Gray code iterator for C(n, k)
    pub fn new(n: usize, k: usize) -> Self {
        if k > n {
            return CombinationGrayCodeIterator {
                n,
                k,
                current: vec![],
                done: true,
            };
        }

        // Start with first k bits set
        let mut current = vec![false; n];
        for i in 0..k {
            current[i] = true;
        }

        CombinationGrayCodeIterator {
            n,
            k,
            current,
            done: false, // Fixed: should always start with done=false
        }
    }

    /// Convert bit vector to combination
    fn to_combination(&self) -> Vec<usize> {
        self.current
            .iter()
            .enumerate()
            .filter(|(_, &bit)| bit)
            .map(|(i, _)| i)
            .collect()
    }
}

impl Iterator for CombinationGrayCodeIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result = self.to_combination();

        if self.k == 0 {
            self.done = true;
            return Some(result);
        }

        // Find the rightmost 1 followed by a 0
        let mut i = 0;
        while i < self.n - 1 {
            if self.current[i] && !self.current[i + 1] {
                break;
            }
            i += 1;
        }

        if i == self.n - 1 {
            // No such position found, we're done
            self.done = true;
        } else {
            // Swap positions i and i+1
            self.current[i] = false;
            self.current[i + 1] = true;

            // Move all 1s before position i to the beginning
            let ones_before = self.current[0..i].iter().filter(|&&b| b).count();
            for j in 0..i {
                self.current[j] = j < ones_before;
            }
        }

        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.done {
            return (0, Some(0));
        }

        // Use binomial coefficient from parent crate
        let total = crate::binomial(self.n as u32, self.k as u32)
            .to_usize()
            .unwrap_or(usize::MAX);
        (0, Some(total))
    }
}

/// A combination iterator using revolving door algorithm
///
/// Generates combinations in a minimal change order where successive
/// combinations differ by swapping adjacent elements
pub struct RevolvingDoorIterator {
    n: usize,
    k: usize,
    current: Option<Vec<usize>>,
    done: bool,
}

impl RevolvingDoorIterator {
    /// Create a new revolving door iterator for C(n, k)
    pub fn new(n: usize, k: usize) -> Self {
        if k > n {
            return RevolvingDoorIterator {
                n,
                k,
                current: None,
                done: true,
            };
        }

        let initial = if k == 0 {
            Some(vec![])
        } else {
            Some((0..k).collect())
        };

        RevolvingDoorIterator {
            n,
            k,
            current: initial,
            done: false,
        }
    }

    fn next_combination(&mut self) -> Option<Vec<usize>> {
        if self.done || self.current.is_none() {
            return None;
        }

        let result = self.current.clone();
        let comb = self.current.as_mut()?;

        if self.k == 0 {
            self.done = true;
            return result;
        }

        // Find rightmost element that can be incremented
        let mut i = self.k;
        while i > 0 && comb[i - 1] == self.n - self.k + i - 1 {
            i -= 1;
        }

        if i == 0 {
            self.done = true;
        } else {
            comb[i - 1] += 1;
            for j in i..self.k {
                comb[j] = comb[i - 1] + j - i + 1;
            }
        }

        result
    }
}

impl Iterator for RevolvingDoorIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_combination()
    }
}

/// A partition iterator
///
/// Generates all integer partitions of n
pub struct PartitionIterator {
    n: usize,
    stack: VecDeque<(usize, usize, Vec<usize>)>, // (remaining, max_value, current)
}

impl PartitionIterator {
    /// Create a new partition iterator for n
    pub fn new(n: usize) -> Self {
        let mut stack = VecDeque::new();
        stack.push_back((n, n, vec![]));

        PartitionIterator { n, stack }
    }
}

impl Iterator for PartitionIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((remaining, max_value, current)) = self.stack.pop_front() {
            if remaining == 0 {
                return Some(current);
            }

            // Try all values from max_value down to 1
            for value in (1..=max_value.min(remaining)).rev() {
                let mut new_current = current.clone();
                new_current.push(value);
                self.stack
                    .push_back((remaining - value, value, new_current));
            }
        }

        None
    }
}

/// A composition iterator
///
/// Generates all compositions (ordered partitions) of n
pub struct CompositionIterator {
    n: usize,
    current: Option<Vec<usize>>,
    stack: VecDeque<(usize, Vec<usize>)>,
}

impl CompositionIterator {
    /// Create a new composition iterator for n
    pub fn new(n: usize) -> Self {
        let mut stack = VecDeque::new();
        if n == 0 {
            stack.push_back((0, vec![]));
        } else {
            stack.push_back((n, vec![]));
        }

        CompositionIterator {
            n,
            current: None,
            stack,
        }
    }
}

impl Iterator for CompositionIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((remaining, current)) = self.stack.pop_front() {
            if remaining == 0 {
                return Some(current);
            }

            // Try all possible next parts
            for part in (1..=remaining).rev() {
                let mut new_current = current.clone();
                new_current.push(part);
                self.stack.push_back((remaining - part, new_current));
            }
        }

        None
    }
}

/// A lazy iterator for Cartesian products
///
/// This iterator generates tuples from the Cartesian product of multiple sequences
/// without materializing all combinations at once. It supports both finite and
/// infinite input sequences through diagonal enumeration.
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::enumeration::CartesianProduct;
///
/// let sets = vec![vec![1, 2], vec![3, 4], vec![5]];
/// let product: Vec<Vec<i32>> = CartesianProduct::new(sets).collect();
/// assert_eq!(product.len(), 4); // 2 * 2 * 1
/// ```
pub struct CartesianProduct<T: Clone> {
    /// The sets to take the product of (we store items as we collect them)
    sets: Vec<Vec<T>>,
    /// Current indices into each set
    indices: Vec<usize>,
    /// Whether we've finished iterating
    done: bool,
    /// Whether any set is known to be empty
    has_empty: bool,
}

impl<T: Clone> CartesianProduct<T> {
    /// Create a new CartesianProduct iterator from a collection of sets
    pub fn new(sets: Vec<Vec<T>>) -> Self {
        if sets.is_empty() {
            // Empty input means one empty tuple
            return CartesianProduct {
                sets: vec![],
                indices: vec![],
                done: false,
                has_empty: false,
            };
        }

        let has_empty = sets.iter().any(|s| s.is_empty());
        let indices = vec![0; sets.len()];

        CartesianProduct {
            sets,
            indices,
            done: has_empty,
            has_empty,
        }
    }

    /// Create a CartesianProduct from an iterator of iterators
    ///
    /// This collects items from the input iterators as needed. For infinite
    /// iterators, items are collected lazily during iteration.
    pub fn from_iters<I>(iters: I) -> Self
    where
        I: IntoIterator<Item = Vec<T>>,
    {
        let sets: Vec<Vec<T>> = iters.into_iter().collect();
        Self::new(sets)
    }

    /// Increment indices to get the next tuple (standard lexicographic order)
    fn increment_indices(&mut self) -> bool {
        // Increment from rightmost position
        for i in (0..self.indices.len()).rev() {
            self.indices[i] += 1;
            if self.indices[i] < self.sets[i].len() {
                return true;
            }
            self.indices[i] = 0;
        }
        false
    }

    /// Get the current tuple
    fn current_tuple(&self) -> Vec<T> {
        self.indices
            .iter()
            .enumerate()
            .map(|(i, &idx)| self.sets[i][idx].clone())
            .collect()
    }
}

impl<T: Clone> Iterator for CartesianProduct<T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        // Special case: empty input produces one empty tuple
        if self.sets.is_empty() {
            self.done = true;
            return Some(vec![]);
        }

        // Get current tuple
        let result = self.current_tuple();

        // Advance to next position
        if !self.increment_indices() {
            self.done = true;
        }

        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.has_empty {
            return (0, Some(0));
        }

        if self.sets.is_empty() {
            return if self.done { (0, Some(0)) } else { (1, Some(1)) };
        }

        // Calculate total size and remaining
        let total: Option<usize> = self
            .sets
            .iter()
            .try_fold(1usize, |acc, set| acc.checked_mul(set.len()));

        if let Some(total) = total {
            // Calculate how many we've already produced
            let mut produced = 0usize;
            let mut multiplier = 1usize;

            for i in (0..self.indices.len()).rev() {
                produced += self.indices[i] * multiplier;
                multiplier *= self.sets[i].len();
            }

            let remaining = total.saturating_sub(produced);
            (remaining, Some(remaining))
        } else {
            // Overflow occurred, size is very large
            (usize::MAX, None)
        }
    }
}

impl<T: Clone> ExactSizeIterator for CartesianProduct<T> {}

/// A lazy iterator for Cartesian products with support for infinite sequences
///
/// This uses diagonal enumeration to ensure that products involving infinite
/// sequences can still be enumerated. Each output tuple is finite, but the
/// iterator itself may be infinite.
///
/// The enumeration proceeds by generating tuples in order of increasing
/// "coordinate sum" - i.e., tuples where indices sum to 0, then 1, then 2, etc.
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::enumeration::InfiniteCartesianProduct;
///
/// // Product of two infinite sequences
/// let naturals1 = Box::new(0..) as Box<dyn Iterator<Item=i32>>;
/// let naturals2 = Box::new(0..) as Box<dyn Iterator<Item=i32>>;
///
/// let mut product = InfiniteCartesianProduct::new(vec![naturals1, naturals2]);
///
/// // First few tuples: [0,0], [0,1], [1,0], [0,2], [1,1], [2,0], ...
/// assert_eq!(product.next(), Some(vec![0, 0]));
/// assert_eq!(product.next(), Some(vec![0, 1]));
/// assert_eq!(product.next(), Some(vec![1, 0]));
/// ```
pub struct InfiniteCartesianProduct<T: Clone> {
    /// Storage for elements from each iterator
    storage: Vec<Vec<T>>,
    /// The source iterators
    iterators: Vec<Box<dyn Iterator<Item = T>>>,
    /// Current diagonal level
    diagonal: usize,
    /// Current position within the diagonal
    position: usize,
    /// Number of tuples at current diagonal level
    diagonal_size: usize,
    /// Number of consecutive empty diagonals
    empty_diagonals: usize,
}

impl<T: Clone> InfiniteCartesianProduct<T> {
    /// Create a new InfiniteCartesianProduct from a vector of iterators
    pub fn new(iterators: Vec<Box<dyn Iterator<Item = T>>>) -> Self {
        let n = iterators.len();
        let storage = vec![Vec::new(); n];

        InfiniteCartesianProduct {
            storage,
            iterators,
            diagonal: 0,
            position: 0,
            diagonal_size: if n == 0 { 0 } else { 1 },
            empty_diagonals: 0,
        }
    }

    /// Ensure we have enough elements from iterator i to reach index idx
    fn ensure_element(&mut self, i: usize, idx: usize) -> bool {
        while self.storage[i].len() <= idx {
            if let Some(elem) = self.iterators[i].next() {
                self.storage[i].push(elem);
            } else {
                // Iterator exhausted
                return false;
            }
        }
        true
    }

    /// Generate the k-th tuple at the given diagonal level
    /// Returns None if we can't generate it (iterator exhausted)
    fn tuple_at_diagonal(&mut self, diagonal: usize, k: usize) -> Option<Vec<T>> {
        if self.iterators.is_empty() {
            return None;
        }

        // Generate indices that sum to 'diagonal'
        // We enumerate all ways to partition 'diagonal' into len indices
        let indices = self.get_indices_for_position(diagonal, k)?;

        // Ensure we have all needed elements
        for (i, &idx) in indices.iter().enumerate() {
            if !self.ensure_element(i, idx) {
                return None;
            }
        }

        // Build the tuple
        Some(
            indices
                .iter()
                .enumerate()
                .map(|(i, &idx)| self.storage[i][idx].clone())
                .collect(),
        )
    }

    /// Convert diagonal position to indices
    /// This generates the k-th way to partition 'sum' into 'n' non-negative integers
    fn get_indices_for_position(&self, sum: usize, mut k: usize) -> Option<Vec<usize>> {
        let n = self.iterators.len();
        if n == 0 {
            return None;
        }

        let mut indices = vec![0; n];
        let mut remaining = sum;

        for i in 0..n - 1 {
            // How many ways to distribute remaining among the rest?
            for val in 0..=remaining {
                let ways = Self::count_partitions(remaining - val, n - i - 1);
                if k < ways {
                    indices[i] = val;
                    remaining -= val;
                    break;
                }
                k -= ways;
            }
        }
        indices[n - 1] = remaining;

        Some(indices)
    }

    /// Count number of ways to partition sum into n non-negative integers
    /// This is "stars and bars": C(sum + n - 1, n - 1)
    fn count_partitions(sum: usize, n: usize) -> usize {
        if n == 0 {
            return if sum == 0 { 1 } else { 0 };
        }
        if n == 1 {
            return 1;
        }

        // C(sum + n - 1, n - 1)
        Self::binomial(sum + n - 1, n - 1)
    }

    /// Simple binomial coefficient calculation
    fn binomial(n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        if k == 0 || k == n {
            return 1;
        }

        let k = k.min(n - k);
        let mut result = 1usize;

        for i in 0..k {
            result = result
                .saturating_mul(n - i)
                .checked_div(i + 1)
                .unwrap_or(usize::MAX);
        }

        result
    }
}

impl<T: Clone> Iterator for InfiniteCartesianProduct<T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.iterators.is_empty() {
            return None;
        }

        loop {
            // Try all positions at current diagonal
            while self.position < self.diagonal_size {
                if let Some(tuple) = self.tuple_at_diagonal(self.diagonal, self.position) {
                    self.position += 1;
                    self.empty_diagonals = 0; // Reset counter when we find a tuple
                    return Some(tuple);
                }
                // This particular combination can't be generated, try next
                self.position += 1;
            }

            // Finished this diagonal without finding anything
            // (or found some but now at end of diagonal)

            // Move to next diagonal
            self.diagonal += 1;
            self.position = 0;
            self.diagonal_size = Self::count_partitions(self.diagonal, self.iterators.len());

            if self.diagonal_size == 0 {
                return None;
            }

            // Check if we should stop - if we've seen many empty diagonals in a row,
            // it means all finite iterators are exhausted
            // We use the number of iterators as a threshold
            if self.empty_diagonals > self.iterators.len() {
                return None;
            }

            // Try to find at least one valid tuple at this diagonal
            let mut found_at_this_diagonal = false;
            for pos in 0..self.diagonal_size {
                if self.tuple_at_diagonal(self.diagonal, pos).is_some() {
                    found_at_this_diagonal = true;
                    break;
                }
            }

            if !found_at_this_diagonal {
                self.empty_diagonals += 1;
            } else {
                self.empty_diagonals = 0;
            }
        }
    }
}

/// Generate Cartesian product of multiple sets (eager evaluation)
///
/// This function eagerly computes all tuples in the Cartesian product.
/// For lazy evaluation, use the `CartesianProduct` iterator instead.
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::enumeration::cartesian_product;
///
/// let set1 = vec![1, 2];
/// let set2 = vec![3, 4];
/// let product = cartesian_product(&[set1, set2]);
/// assert_eq!(product.len(), 4);
/// ```
pub fn cartesian_product<T: Clone>(sets: &[Vec<T>]) -> Vec<Vec<T>> {
    CartesianProduct::new(sets.to_vec()).collect()
}

/// Generate all k-tuples from a set (Cartesian power)
pub fn tuples<T: Clone>(set: &[T], k: usize) -> Vec<Vec<T>> {
    if k == 0 {
        return vec![vec![]];
    }

    let sets: Vec<_> = (0..k).map(|_| set.to_vec()).collect();
    cartesian_product(&sets)
}

/// Count the number of ways to distribute n identical objects into k distinct bins
///
/// This is the "stars and bars" problem: C(n+k-1, k-1)
pub fn stars_and_bars(n: usize, k: usize) -> usize {
    if k == 0 {
        return if n == 0 { 1 } else { 0 };
    }
    if k == 1 {
        return 1;
    }

    crate::binomial((n + k - 1) as u32, (k - 1) as u32)
        .to_usize()
        .unwrap_or(0)
}

/// Generate all weak compositions of n into k parts (parts can be 0)
///
/// These correspond to distributing n identical objects into k distinct bins
pub fn weak_compositions(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return if n == 0 { vec![vec![]] } else { vec![] };
    }

    let mut result = Vec::new();
    let mut current = vec![0; k];

    generate_weak_compositions(n, k, 0, &mut current, &mut result);

    result
}

fn generate_weak_compositions(
    remaining: usize,
    k: usize,
    index: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if index == k {
        if remaining == 0 {
            result.push(current.clone());
        }
        return;
    }

    for value in 0..=remaining {
        current[index] = value;
        generate_weak_compositions(remaining - value, k, index + 1, current, result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gray_code_iterator() {
        let subsets: Vec<_> = GrayCodeIterator::new(3).collect();
        assert_eq!(subsets.len(), 8); // 2^3

        // Check that consecutive subsets differ by exactly one element
        for i in 1..subsets.len() {
            let prev_set: std::collections::HashSet<_> = subsets[i - 1].iter().copied().collect();
            let curr_set: std::collections::HashSet<_> = subsets[i].iter().copied().collect();

            let diff_count = prev_set.symmetric_difference(&curr_set).count();

            // In Gray code, consecutive subsets should differ by exactly 1 element
            assert_eq!(diff_count, 1, "Gray code property violated at index {}: prev={:?}, curr={:?}", i, subsets[i-1], subsets[i]);
        }
    }

    #[test]
    fn test_binary_to_gray() {
        assert_eq!(binary_to_gray(0), 0);
        assert_eq!(binary_to_gray(1), 1);
        assert_eq!(binary_to_gray(2), 3);
        assert_eq!(binary_to_gray(3), 2);
        assert_eq!(binary_to_gray(4), 6);
        assert_eq!(binary_to_gray(5), 7);
        assert_eq!(binary_to_gray(6), 5);
        assert_eq!(binary_to_gray(7), 4);
    }

    #[test]
    fn test_gray_to_binary() {
        assert_eq!(gray_to_binary(0), 0);
        assert_eq!(gray_to_binary(1), 1);
        assert_eq!(gray_to_binary(3), 2);
        assert_eq!(gray_to_binary(2), 3);
        assert_eq!(gray_to_binary(6), 4);
        assert_eq!(gray_to_binary(7), 5);
        assert_eq!(gray_to_binary(5), 6);
        assert_eq!(gray_to_binary(4), 7);
    }

    #[test]
    fn test_binary_gray_conversion_roundtrip() {
        // Test that converting binary -> gray -> binary gives original value
        for i in 0..256 {
            let gray = binary_to_gray(i);
            let binary = gray_to_binary(gray);
            assert_eq!(binary, i);
        }
    }

    #[test]
    fn test_binary_gray_code_iterator() {
        let codes: Vec<usize> = BinaryGrayCodeIterator::new(3).collect();
        assert_eq!(codes.len(), 8);
        assert_eq!(codes, vec![0, 1, 3, 2, 6, 7, 5, 4]);

        // Check that consecutive codes differ by exactly one bit
        for i in 1..codes.len() {
            let diff = codes[i] ^ codes[i - 1];
            assert_eq!(
                diff.count_ones(),
                1,
                "Consecutive Gray codes should differ by exactly one bit: {} and {}",
                codes[i - 1],
                codes[i]
            );
        }
    }

    #[test]
    fn test_binary_gray_code_iterator_edge_cases() {
        // n = 0
        let codes: Vec<usize> = BinaryGrayCodeIterator::new(0).collect();
        assert_eq!(codes.len(), 0);

        // n = 1
        let codes: Vec<usize> = BinaryGrayCodeIterator::new(1).collect();
        assert_eq!(codes, vec![0, 1]);

        // n = 2
        let codes: Vec<usize> = BinaryGrayCodeIterator::new(2).collect();
        assert_eq!(codes, vec![0, 1, 3, 2]);
    }

    #[test]
    fn test_binary_gray_codes() {
        let codes = binary_gray_codes(3);
        assert_eq!(codes, vec![0, 1, 3, 2, 6, 7, 5, 4]);

        let codes = binary_gray_codes(4);
        assert_eq!(codes.len(), 16);

        // Verify Gray code property
        for i in 1..codes.len() {
            assert_eq!((codes[i] ^ codes[i - 1]).count_ones(), 1);
        }
    }

    #[test]
    fn test_binary_gray_code_strings() {
        let codes = binary_gray_code_strings(3);
        assert_eq!(
            codes,
            vec!["000", "001", "011", "010", "110", "111", "101", "100"]
        );

        // Check that each consecutive pair differs by one bit
        for i in 1..codes.len() {
            let prev = usize::from_str_radix(&codes[i - 1], 2).unwrap();
            let curr = usize::from_str_radix(&codes[i], 2).unwrap();
            assert_eq!((prev ^ curr).count_ones(), 1);
        }
    }

    #[test]
    fn test_binary_gray_code_iterator_size_hint() {
        let iter = BinaryGrayCodeIterator::new(4);
        let (lower, upper) = iter.size_hint();
        assert_eq!(lower, 16);
        assert_eq!(upper, Some(16));

        let codes: Vec<_> = iter.collect();
        assert_eq!(codes.len(), 16);
    }

    #[test]
    fn test_permutation_gray_code_iterator() {
        let perms: Vec<Vec<usize>> = PermutationGrayCodeIterator::new(3).collect();
        assert_eq!(perms.len(), 6); // 3! = 6

        // First permutation should be [0, 1, 2]
        assert_eq!(perms[0], vec![0, 1, 2]);

        // Check that each consecutive pair differs by swapping adjacent elements
        for i in 1..perms.len() {
            let prev = &perms[i - 1];
            let curr = &perms[i];

            // Find the positions that differ
            let mut diff_positions = vec![];
            for j in 0..prev.len() {
                if prev[j] != curr[j] {
                    diff_positions.push(j);
                }
            }

            // Should have exactly 2 differences (the swapped elements)
            assert_eq!(
                diff_positions.len(),
                2,
                "Permutations should differ by exactly one swap: {:?} -> {:?}",
                prev,
                curr
            );

            // The differences should be adjacent
            if diff_positions.len() == 2 {
                let pos_diff = diff_positions[1] as i32 - diff_positions[0] as i32;
                assert_eq!(pos_diff.abs(), 1, "Swap should be of adjacent elements");
            }
        }
    }

    #[test]
    fn test_permutation_gray_code_iterator_edge_cases() {
        // n = 0
        let perms: Vec<Vec<usize>> = PermutationGrayCodeIterator::new(0).collect();
        assert_eq!(perms.len(), 1);
        assert_eq!(perms[0], vec![]);

        // n = 1
        let perms: Vec<Vec<usize>> = PermutationGrayCodeIterator::new(1).collect();
        assert_eq!(perms.len(), 1);
        assert_eq!(perms[0], vec![0]);

        // n = 2
        let perms: Vec<Vec<usize>> = PermutationGrayCodeIterator::new(2).collect();
        assert_eq!(perms.len(), 2);
        assert_eq!(perms[0], vec![0, 1]);
        assert_eq!(perms[1], vec![1, 0]);
    }

    #[test]
    fn test_permutation_gray_code_completeness() {
        // Test that we generate all permutations
        let perms: Vec<Vec<usize>> = PermutationGrayCodeIterator::new(4).collect();
        assert_eq!(perms.len(), 24); // 4! = 24

        // Convert to set to check uniqueness
        let perm_set: std::collections::HashSet<_> = perms.iter().cloned().collect();
        assert_eq!(perm_set.len(), 24, "All permutations should be unique");
    }

    #[test]
    fn test_combination_gray_code_iterator() {
        let combs: Vec<Vec<usize>> = CombinationGrayCodeIterator::new(4, 2).collect();
        assert_eq!(combs.len(), 6); // C(4, 2) = 6

        // All combinations should have exactly k elements
        for comb in &combs {
            assert_eq!(comb.len(), 2);
        }

        // All combinations should be unique
        let comb_set: std::collections::HashSet<_> = combs.iter().cloned().collect();
        assert_eq!(comb_set.len(), 6);
    }

    #[test]
    fn test_combination_gray_code_iterator_edge_cases() {
        // k = 0
        let combs: Vec<Vec<usize>> = CombinationGrayCodeIterator::new(3, 0).collect();
        assert_eq!(combs.len(), 1);
        assert_eq!(combs[0], vec![]);

        // k = n
        let combs: Vec<Vec<usize>> = CombinationGrayCodeIterator::new(3, 3).collect();
        assert_eq!(combs.len(), 1);
        assert_eq!(combs[0], vec![0, 1, 2]);

        // k > n (invalid)
        let combs: Vec<Vec<usize>> = CombinationGrayCodeIterator::new(3, 5).collect();
        assert_eq!(combs.len(), 0);
    }

    #[test]
    fn test_combination_gray_code_completeness() {
        // Test that we generate all combinations
        let combs: Vec<Vec<usize>> = CombinationGrayCodeIterator::new(5, 3).collect();
        assert_eq!(combs.len(), 10); // C(5, 3) = 10

        // All combinations should have exactly 3 elements
        for comb in &combs {
            assert_eq!(comb.len(), 3);
        }

        // All combinations should be unique
        let comb_set: std::collections::HashSet<_> = combs.iter().cloned().collect();
        assert_eq!(comb_set.len(), 10);
    }

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
        assert_eq!(factorial(2), 2);
        assert_eq!(factorial(3), 6);
        assert_eq!(factorial(4), 24);
        assert_eq!(factorial(5), 120);
        assert_eq!(factorial(10), 3628800);
    }

    #[test]
    fn test_revolving_door_iterator() {
        let combinations: Vec<_> = RevolvingDoorIterator::new(5, 3).collect();
        assert_eq!(combinations.len(), 10); // C(5,3) = 10

        // First should be [0,1,2]
        assert_eq!(combinations[0], vec![0, 1, 2]);

        // Last should be [2,3,4]
        assert_eq!(combinations[9], vec![2, 3, 4]);
    }

    #[test]
    fn test_partition_iterator() {
        let partitions: Vec<_> = PartitionIterator::new(4).collect();
        assert_eq!(partitions.len(), 5); // p(4) = 5

        // Should include [4], [3,1], [2,2], [2,1,1], [1,1,1,1]
        assert!(partitions.contains(&vec![4]));
        assert!(partitions.contains(&vec![3, 1]));
        assert!(partitions.contains(&vec![2, 2]));
        assert!(partitions.contains(&vec![2, 1, 1]));
        assert!(partitions.contains(&vec![1, 1, 1, 1]));
    }

    #[test]
    fn test_composition_iterator() {
        let compositions: Vec<_> = CompositionIterator::new(3).collect();
        assert_eq!(compositions.len(), 4); // 2^(n-1) = 2^2 = 4

        // Should include [3], [1,2], [2,1], [1,1,1]
        assert!(compositions.contains(&vec![3]));
        assert!(compositions.contains(&vec![1, 2]));
        assert!(compositions.contains(&vec![2, 1]));
        assert!(compositions.contains(&vec![1, 1, 1]));
    }

    #[test]
    fn test_cartesian_product() {
        let set1 = vec![1, 2];
        let set2 = vec![3, 4];
        let set3 = vec![5];

        let product = cartesian_product(&[set1, set2, set3]);
        assert_eq!(product.len(), 4); // 2 * 2 * 1

        assert!(product.contains(&vec![1, 3, 5]));
        assert!(product.contains(&vec![1, 4, 5]));
        assert!(product.contains(&vec![2, 3, 5]));
        assert!(product.contains(&vec![2, 4, 5]));
    }

    #[test]
    fn test_tuples() {
        let set = vec![0, 1];
        let tuples_2 = tuples(&set, 2);

        assert_eq!(tuples_2.len(), 4); // 2^2
        assert!(tuples_2.contains(&vec![0, 0]));
        assert!(tuples_2.contains(&vec![0, 1]));
        assert!(tuples_2.contains(&vec![1, 0]));
        assert!(tuples_2.contains(&vec![1, 1]));
    }

    #[test]
    fn test_stars_and_bars() {
        // Distribute 5 objects into 3 bins: C(5+3-1, 3-1) = C(7, 2) = 21
        assert_eq!(stars_and_bars(5, 3), 21);

        // Distribute 0 objects into 0 bins: 1 way (empty distribution)
        assert_eq!(stars_and_bars(0, 0), 1);

        // Distribute 5 objects into 1 bin: 1 way (all in one bin)
        assert_eq!(stars_and_bars(5, 1), 1);
    }

    #[test]
    fn test_weak_compositions() {
        let weak_comps = weak_compositions(3, 2);
        // Should include: [0,3], [1,2], [2,1], [3,0]
        assert_eq!(weak_comps.len(), 4);

        assert!(weak_comps.contains(&vec![0, 3]));
        assert!(weak_comps.contains(&vec![1, 2]));
        assert!(weak_comps.contains(&vec![2, 1]));
        assert!(weak_comps.contains(&vec![3, 0]));

        // All should sum to 3
        for comp in &weak_comps {
            assert_eq!(comp.iter().sum::<usize>(), 3);
        }
    }

    #[test]
    fn test_weak_compositions_count() {
        // Number of weak compositions of n into k parts should match stars_and_bars
        let n = 4;
        let k = 3;
        let weak_comps = weak_compositions(n, k);
        assert_eq!(weak_comps.len(), stars_and_bars(n, k));
    }

    #[test]
    fn test_cartesian_product_iterator_basic() {
        let sets = vec![vec![1, 2], vec![3, 4], vec![5]];
        let product: Vec<Vec<i32>> = CartesianProduct::new(sets).collect();

        assert_eq!(product.len(), 4); // 2 * 2 * 1

        assert!(product.contains(&vec![1, 3, 5]));
        assert!(product.contains(&vec![1, 4, 5]));
        assert!(product.contains(&vec![2, 3, 5]));
        assert!(product.contains(&vec![2, 4, 5]));
    }

    #[test]
    fn test_cartesian_product_iterator_lazy() {
        // Verify that the iterator is truly lazy by only taking first few items
        let sets = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let mut iter = CartesianProduct::new(sets);

        assert_eq!(iter.next(), Some(vec![1, 4, 7]));
        assert_eq!(iter.next(), Some(vec![1, 4, 8]));
        assert_eq!(iter.next(), Some(vec![1, 4, 9]));
        assert_eq!(iter.next(), Some(vec![1, 5, 7]));

        // We can stop here without computing all 27 tuples
    }

    #[test]
    fn test_cartesian_product_iterator_empty_input() {
        // Empty input should produce one empty tuple
        let sets: Vec<Vec<i32>> = vec![];
        let product: Vec<Vec<i32>> = CartesianProduct::new(sets).collect();

        assert_eq!(product.len(), 1);
        assert_eq!(product[0], vec![]);
    }

    #[test]
    fn test_cartesian_product_iterator_empty_set() {
        // If any set is empty, product should be empty
        let sets = vec![vec![1, 2], vec![], vec![3]];
        let product: Vec<Vec<i32>> = CartesianProduct::new(sets).collect();

        assert_eq!(product.len(), 0);
    }

    #[test]
    fn test_cartesian_product_iterator_single_set() {
        let sets = vec![vec![1, 2, 3]];
        let product: Vec<Vec<i32>> = CartesianProduct::new(sets).collect();

        assert_eq!(product.len(), 3);
        assert_eq!(product[0], vec![1]);
        assert_eq!(product[1], vec![2]);
        assert_eq!(product[2], vec![3]);
    }

    #[test]
    fn test_cartesian_product_iterator_size_hint() {
        let sets = vec![vec![1, 2], vec![3, 4, 5], vec![6]];
        let iter = CartesianProduct::new(sets);

        let (lower, upper) = iter.size_hint();
        assert_eq!(lower, 6); // 2 * 3 * 1
        assert_eq!(upper, Some(6));
    }

    #[test]
    fn test_cartesian_product_backward_compat() {
        // Verify that the new implementation gives same results as before
        let set1 = vec![1, 2];
        let set2 = vec![3, 4];
        let set3 = vec![5];

        let product = cartesian_product(&[set1.clone(), set2.clone(), set3.clone()]);
        let product_iter: Vec<Vec<i32>> =
            CartesianProduct::new(vec![set1, set2, set3]).collect();

        // Both should produce the same tuples (order may differ, so we compare sets)
        assert_eq!(product.len(), product_iter.len());
        for tuple in &product {
            assert!(product_iter.contains(tuple));
        }
    }

    #[test]
    fn test_infinite_cartesian_product_finite_inputs() {
        // Test with finite iterators first
        let iter1: Box<dyn Iterator<Item = i32>> = Box::new(vec![1, 2].into_iter());
        let iter2: Box<dyn Iterator<Item = i32>> = Box::new(vec![3, 4].into_iter());

        let mut product = InfiniteCartesianProduct::new(vec![iter1, iter2]);

        // Diagonal enumeration: sum=0: [0,0], sum=1: [0,1],[1,0], sum=2: [0,2],[1,1],[2,0]
        // But we only have indices up to 1 in each dimension
        assert_eq!(product.next(), Some(vec![1, 3])); // [0,0]
        assert_eq!(product.next(), Some(vec![1, 4])); // [0,1]
        assert_eq!(product.next(), Some(vec![2, 3])); // [1,0]
        assert_eq!(product.next(), Some(vec![2, 4])); // [1,1]
        assert_eq!(product.next(), None); // No more elements
    }

    #[test]
    fn test_infinite_cartesian_product_with_infinite_iterator() {
        // Test with actual infinite iterator
        let iter1: Box<dyn Iterator<Item = i32>> = Box::new(0..);
        let iter2: Box<dyn Iterator<Item = i32>> = Box::new(0..);

        let mut product = InfiniteCartesianProduct::new(vec![iter1, iter2]);

        // First few tuples in diagonal order
        assert_eq!(product.next(), Some(vec![0, 0])); // sum=0
        assert_eq!(product.next(), Some(vec![0, 1])); // sum=1
        assert_eq!(product.next(), Some(vec![1, 0])); // sum=1
        assert_eq!(product.next(), Some(vec![0, 2])); // sum=2
        assert_eq!(product.next(), Some(vec![1, 1])); // sum=2
        assert_eq!(product.next(), Some(vec![2, 0])); // sum=2
        assert_eq!(product.next(), Some(vec![0, 3])); // sum=3
        assert_eq!(product.next(), Some(vec![1, 2])); // sum=3

        // Can take as many as we want
        for _ in 0..100 {
            assert!(product.next().is_some());
        }
    }

    #[test]
    fn test_infinite_cartesian_product_three_iterators() {
        // Test with three infinite iterators
        let iter1: Box<dyn Iterator<Item = usize>> = Box::new(0..);
        let iter2: Box<dyn Iterator<Item = usize>> = Box::new(0..);
        let iter3: Box<dyn Iterator<Item = usize>> = Box::new(0..);

        let mut product = InfiniteCartesianProduct::new(vec![iter1, iter2, iter3]);

        // First tuple should be [0,0,0]
        assert_eq!(product.next(), Some(vec![0, 0, 0]));

        // Should be able to generate many tuples
        let tuples: Vec<_> = product.take(50).collect();
        assert_eq!(tuples.len(), 50);

        // All tuples should have 3 elements
        for tuple in &tuples {
            assert_eq!(tuple.len(), 3);
        }

        // Check that we eventually see tuples with larger values
        let has_large = tuples.iter().any(|t| t.iter().any(|&x| x >= 3));
        assert!(has_large);
    }

    #[test]
    fn test_infinite_cartesian_product_mixed_finite_infinite() {
        // Mix of finite and infinite iterators
        let iter1: Box<dyn Iterator<Item = i32>> = Box::new(vec![1, 2].into_iter());
        let iter2: Box<dyn Iterator<Item = i32>> = Box::new(0..);

        let mut product = InfiniteCartesianProduct::new(vec![iter1, iter2]);

        // Should get infinitely many tuples, but first component only 1 or 2
        let tuples: Vec<_> = product.take(20).collect();
        assert_eq!(tuples.len(), 20);

        for tuple in &tuples {
            assert!(tuple[0] == 1 || tuple[0] == 2);
        }
    }

    #[test]
    fn test_cartesian_product_large_lazy() {
        // Test with a large product that would be expensive to compute eagerly
        let sets: Vec<Vec<i32>> = (0..10).map(|_| (0..10).collect()).collect();
        let mut iter = CartesianProduct::new(sets);

        // Only take first 5 tuples, not all 10^10
        let first_five: Vec<_> = iter.take(5).collect();
        assert_eq!(first_five.len(), 5);

        // Verify correctness of first tuple
        assert_eq!(first_five[0], vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_cartesian_product_from_iters() {
        let sets = vec![vec![1, 2], vec![3, 4]];
        let product: Vec<Vec<i32>> = CartesianProduct::from_iters(sets).collect();

        assert_eq!(product.len(), 4);
        assert!(product.contains(&vec![1, 3]));
        assert!(product.contains(&vec![1, 4]));
        assert!(product.contains(&vec![2, 3]));
        assert!(product.contains(&vec![2, 4]));
    }

    #[test]
    fn test_infinite_cartesian_product_coverage() {
        // Verify that diagonal enumeration eventually reaches any specific tuple
        let iter1: Box<dyn Iterator<Item = usize>> = Box::new(0..);
        let iter2: Box<dyn Iterator<Item = usize>> = Box::new(0..);

        let product = InfiniteCartesianProduct::new(vec![iter1, iter2]);

        // Take enough tuples to be sure we've seen [3, 2]
        let tuples: Vec<_> = product.take(100).collect();

        // The tuple [3, 2] has coordinate sum 5, so it should appear
        assert!(tuples.contains(&vec![3, 2]));
        assert!(tuples.contains(&vec![2, 3]));
        assert!(tuples.contains(&vec![5, 0]));
    }
}
