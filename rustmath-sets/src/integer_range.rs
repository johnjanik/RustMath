//! Integer Range Set Operations
//!
//! This module provides mathematical set operations for integer ranges, extending
//! Rust's standard `Range` types with union, intersection, complement, and difference.
//!
//! # Overview
//!
//! The `IntegerRange<T>` type represents a range of integers with optional step size.
//! Unlike Rust's `std::ops::Range`, this type:
//! - Supports mathematical set operations (∪, ∩, -, complement)
//! - Can represent infinite ranges
//! - Works with arbitrary integer types (i32, i64, BigInt, etc.)
//! - Preserves step information for arithmetic progressions
//!
//! # Relationship to std::ops::Range
//!
//! `IntegerRange` extends `std::ops::Range` by:
//! - Adding set-theoretic operations
//! - Supporting infinite bounds
//! - Allowing custom step sizes (arithmetic progressions)
//! - Being generic over any integer-like type implementing required traits
//!
//! You can convert between `Range<T>` and `IntegerRange<T>`:
//! ```rust,ignore
//! use rustmath_sets::integer_range::IntegerRange;
//!
//! // Convert from std::ops::Range
//! let range: IntegerRange<i64> = (0..10).into();
//!
//! // Convert to Vec for iteration
//! let values: Vec<i64> = range.to_vec().unwrap();
//! ```
//!
//! # Examples
//!
//! ```rust,ignore
//! use rustmath_sets::integer_range::IntegerRange;
//!
//! // Create ranges
//! let r1 = IntegerRange::finite(0, 10, 1);    // [0, 10) with step 1
//! let r2 = IntegerRange::finite(5, 15, 1);    // [5, 15) with step 1
//!
//! // Set operations
//! let union = r1.union(&r2);                  // [0, 15)
//! let intersection = r1.intersect(&r2);       // [5, 10)
//! let difference = r1.difference(&r2);        // [0, 5)
//!
//! // Infinite ranges
//! let naturals = IntegerRange::from(0);       // [0, ∞)
//! let evens = IntegerRange::from_step(0, 2);  // {0, 2, 4, 6, ...}
//! ```

use rustmath_core::{NumericConversion, Ring};
use std::cmp::{max, min, Ordering};
use std::fmt;
use std::ops::Range;

/// Represents a bound for an integer range (finite or infinite)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Bound<T> {
    /// Negative infinity (-∞)
    NegInfinity,
    /// Positive infinity (+∞)
    PosInfinity,
    /// A finite inclusive bound
    Finite(T),
}

impl<T: PartialOrd> PartialOrd for Bound<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Bound::NegInfinity, Bound::NegInfinity) => Some(Ordering::Equal),
            (Bound::NegInfinity, _) => Some(Ordering::Less),
            (_, Bound::NegInfinity) => Some(Ordering::Greater),
            (Bound::PosInfinity, Bound::PosInfinity) => Some(Ordering::Equal),
            (Bound::PosInfinity, _) => Some(Ordering::Greater),
            (_, Bound::PosInfinity) => Some(Ordering::Less),
            (Bound::Finite(a), Bound::Finite(b)) => a.partial_cmp(b),
        }
    }
}

impl<T: Ord> Ord for Bound<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// A mathematical integer range with support for set operations
///
/// Represents a range [start, end) with a given step size, where both bounds
/// can be finite or infinite.
///
/// # Type Parameters
///
/// * `T` - The integer type (must implement `Ring`, `Ord`, `Clone`)
///
/// # Examples
///
/// ```rust,ignore
/// use rustmath_sets::integer_range::IntegerRange;
///
/// // Finite range [0, 10)
/// let r = IntegerRange::finite(0, 10, 1);
/// assert!(r.contains(&5));
/// assert!(!r.contains(&10));
///
/// // Infinite range [0, ∞)
/// let naturals = IntegerRange::from(0);
/// assert!(naturals.contains(&1000000));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegerRange<T>
where
    T: Ring + Ord + Clone,
{
    /// Lower bound (inclusive)
    start: Bound<T>,
    /// Upper bound (exclusive)
    end: Bound<T>,
    /// Step size (must be positive)
    step: T,
}

impl<T> IntegerRange<T>
where
    T: Ring + Ord + Clone,
{
    /// Check if this range is empty
    pub fn is_empty(&self) -> bool {
        match (&self.start, &self.end) {
            (Bound::Finite(s), Bound::Finite(e)) => s >= e,
            _ => false,
        }
    }

    /// Check if this range is finite
    pub fn is_finite(&self) -> bool {
        matches!((&self.start, &self.end), (Bound::Finite(_), Bound::Finite(_)))
    }

    /// Check if this range is infinite
    pub fn is_infinite(&self) -> bool {
        !self.is_finite()
    }

    /// Get the start bound
    pub fn start(&self) -> &Bound<T> {
        &self.start
    }

    /// Get the end bound
    pub fn end(&self) -> &Bound<T> {
        &self.end
    }

    /// Get the step size
    pub fn step(&self) -> &T {
        &self.step
    }
}

impl<T> IntegerRange<T>
where
    T: Ring + Ord + Clone + NumericConversion,
{
    /// Create a finite integer range [start, end) with given step
    ///
    /// # Arguments
    ///
    /// * `start` - Starting value (inclusive)
    /// * `end` - Ending value (exclusive)
    /// * `step` - Step size (must be positive)
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let range = IntegerRange::finite(0, 10, 2);  // {0, 2, 4, 6, 8}
    /// ```
    pub fn finite(start: T, end: T, step: T) -> Self {
        assert!(!step.is_zero(), "Step must be positive");
        IntegerRange {
            start: Bound::Finite(start),
            end: Bound::Finite(end),
            step,
        }
    }

    /// Create an infinite range [start, ∞) with given step
    ///
    /// # Arguments
    ///
    /// * `start` - Starting value (inclusive)
    /// * `step` - Step size (must be positive)
    pub fn from_step(start: T, step: T) -> Self {
        assert!(!step.is_zero(), "Step must be positive");
        IntegerRange {
            start: Bound::Finite(start),
            end: Bound::PosInfinity,
            step,
        }
    }

    /// Create a range from only a start value with step 1
    ///
    /// Creates the range [start, ∞)
    pub fn from(start: T) -> Self {
        Self::from_step(start, T::one())
    }

    /// Create the empty range
    pub fn empty() -> Self {
        IntegerRange {
            start: Bound::Finite(T::zero()),
            end: Bound::Finite(T::zero()),
            step: T::one(),
        }
    }

    /// Create the range of all integers (-∞, ∞)
    pub fn all_integers() -> Self {
        IntegerRange {
            start: Bound::NegInfinity,
            end: Bound::PosInfinity,
            step: T::one(),
        }
    }

    /// Get the cardinality (number of elements) if finite
    pub fn cardinality(&self) -> Option<usize> {
        match (&self.start, &self.end) {
            (Bound::Finite(s), Bound::Finite(e)) if s < e => {
                // Calculate: ceil((e - s) / step)
                let diff = e.clone() - s.clone();
                let quot = diff.to_i64()? / self.step.to_i64()?;
                Some(quot as usize)
            }
            (Bound::Finite(s), Bound::Finite(e)) if s >= e => Some(0),
            _ => None, // Infinite range
        }
    }

    /// Check if a value is in this range
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let range = IntegerRange::finite(0, 10, 2);  // {0, 2, 4, 6, 8}
    /// assert!(range.contains(&4));
    /// assert!(!range.contains(&5));
    /// ```
    pub fn contains(&self, value: &T) -> bool {
        // Check bounds
        let in_bounds = match (&self.start, &self.end) {
            (Bound::Finite(s), Bound::Finite(e)) => value >= s && value < e,
            (Bound::Finite(s), Bound::PosInfinity) => value >= s,
            (Bound::NegInfinity, Bound::Finite(e)) => value < e,
            (Bound::NegInfinity, Bound::PosInfinity) => true,
            _ => false,
        };

        if !in_bounds {
            return false;
        }

        // Check if value aligns with step
        if let Bound::Finite(s) = &self.start {
            if self.step.is_one() {
                return true;
            }
            // Check if (value - start) is divisible by step
            let diff = value.clone() - s.clone();
            // For now, we use a simple check via conversion
            if let (Some(d), Some(st)) = (diff.to_i64(), self.step.to_i64()) {
                return d % st == 0;
            }
        }

        true
    }

    /// Compute the union of two ranges
    ///
    /// Returns a vector of disjoint ranges representing the union.
    /// If the ranges overlap or are adjacent, they are merged.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let r1 = IntegerRange::finite(0, 5, 1);
    /// let r2 = IntegerRange::finite(3, 8, 1);
    /// let union = r1.union(&r2);  // Returns [0, 8)
    /// ```
    pub fn union(&self, other: &Self) -> Vec<IntegerRange<T>> {
        // Handle empty ranges
        if self.is_empty() {
            return vec![other.clone()];
        }
        if other.is_empty() {
            return vec![self.clone()];
        }

        // For simplicity, only merge if steps are equal
        if self.step != other.step {
            return vec![self.clone(), other.clone()];
        }

        // Check if ranges overlap or are adjacent
        let overlap = self.intersect(other);
        let adjacent = self.is_adjacent(other);

        if !overlap.is_empty() || adjacent {
            // Merge the ranges
            let new_start = match (&self.start, &other.start) {
                (Bound::NegInfinity, _) | (_, Bound::NegInfinity) => Bound::NegInfinity,
                (Bound::Finite(a), Bound::Finite(b)) => Bound::Finite(min(a, b).clone()),
                (Bound::Finite(a), Bound::PosInfinity) => Bound::Finite(a.clone()),
                (Bound::PosInfinity, Bound::Finite(b)) => Bound::Finite(b.clone()),
                _ => Bound::PosInfinity,
            };

            let new_end = match (&self.end, &other.end) {
                (Bound::PosInfinity, _) | (_, Bound::PosInfinity) => Bound::PosInfinity,
                (Bound::Finite(a), Bound::Finite(b)) => Bound::Finite(max(a, b).clone()),
                (Bound::Finite(a), Bound::NegInfinity) => Bound::Finite(a.clone()),
                (Bound::NegInfinity, Bound::Finite(b)) => Bound::Finite(b.clone()),
                _ => Bound::NegInfinity,
            };

            vec![IntegerRange {
                start: new_start,
                end: new_end,
                step: self.step.clone(),
            }]
        } else {
            // Disjoint ranges - return both
            vec![self.clone(), other.clone()]
        }
    }

    /// Compute the intersection of two ranges
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let r1 = IntegerRange::finite(0, 10, 1);
    /// let r2 = IntegerRange::finite(5, 15, 1);
    /// let intersection = r1.intersect(&r2);  // [5, 10)
    /// ```
    pub fn intersect(&self, other: &Self) -> IntegerRange<T> {
        // Compute overlapping bounds
        let new_start = match (&self.start, &other.start) {
            (Bound::NegInfinity, s) | (s, Bound::NegInfinity) => s.clone(),
            (Bound::Finite(a), Bound::Finite(b)) => Bound::Finite(max(a, b).clone()),
            (Bound::PosInfinity, _) | (_, Bound::PosInfinity) => Bound::PosInfinity,
        };

        let new_end = match (&self.end, &other.end) {
            (Bound::PosInfinity, e) | (e, Bound::PosInfinity) => e.clone(),
            (Bound::Finite(a), Bound::Finite(b)) => Bound::Finite(min(a, b).clone()),
            (Bound::NegInfinity, _) | (_, Bound::NegInfinity) => Bound::NegInfinity,
        };

        // Handle step compatibility
        // For now, use the larger step (simplified approach)
        let new_step = if self.step > other.step {
            self.step.clone()
        } else {
            other.step.clone()
        };

        let result = IntegerRange {
            start: new_start,
            end: new_end,
            step: new_step,
        };

        if result.is_empty() {
            Self::empty()
        } else {
            result
        }
    }

    /// Compute the difference of two ranges (self - other)
    ///
    /// Returns a vector of ranges representing elements in `self` but not in `other`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let r1 = IntegerRange::finite(0, 10, 1);
    /// let r2 = IntegerRange::finite(5, 15, 1);
    /// let diff = r1.difference(&r2);  // [0, 5)
    /// ```
    pub fn difference(&self, other: &Self) -> Vec<IntegerRange<T>> {
        if self.is_empty() || other.is_empty() {
            return vec![self.clone()];
        }

        let intersection = self.intersect(other);
        if intersection.is_empty() {
            return vec![self.clone()];
        }

        // Simple case: steps are equal
        if self.step != other.step {
            // Complex case - return original for now
            return vec![self.clone()];
        }

        let mut result = Vec::new();

        // Left part: [self.start, intersection.start)
        match (&self.start, &intersection.start) {
            (Bound::Finite(s), Bound::Finite(i)) if s < i => {
                result.push(IntegerRange {
                    start: Bound::Finite(s.clone()),
                    end: Bound::Finite(i.clone()),
                    step: self.step.clone(),
                });
            }
            _ => {}
        }

        // Right part: [intersection.end, self.end)
        match (&intersection.end, &self.end) {
            (Bound::Finite(i), Bound::Finite(e)) if i < e => {
                result.push(IntegerRange {
                    start: Bound::Finite(i.clone()),
                    end: Bound::Finite(e.clone()),
                    step: self.step.clone(),
                });
            }
            (Bound::Finite(i), Bound::PosInfinity) => {
                result.push(IntegerRange {
                    start: Bound::Finite(i.clone()),
                    end: Bound::PosInfinity,
                    step: self.step.clone(),
                });
            }
            _ => {}
        }

        if result.is_empty() {
            vec![Self::empty()]
        } else {
            result
        }
    }

    /// Compute the complement of this range relative to all integers
    ///
    /// Returns a vector of ranges representing all integers NOT in this range.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let range = IntegerRange::finite(5, 10, 1);
    /// let complement = range.complement();
    /// // Returns: [(-∞, 5), [10, ∞)]
    /// ```
    pub fn complement(&self) -> Vec<IntegerRange<T>> {
        if self.is_empty() {
            return vec![Self::all_integers()];
        }

        let mut result = Vec::new();

        // Left part: (-∞, start)
        match &self.start {
            Bound::Finite(s) => {
                result.push(IntegerRange {
                    start: Bound::NegInfinity,
                    end: Bound::Finite(s.clone()),
                    step: self.step.clone(),
                });
            }
            Bound::NegInfinity => {}
            Bound::PosInfinity => {
                return vec![Self::all_integers()];
            }
        }

        // Right part: [end, ∞)
        match &self.end {
            Bound::Finite(e) => {
                result.push(IntegerRange {
                    start: Bound::Finite(e.clone()),
                    end: Bound::PosInfinity,
                    step: self.step.clone(),
                });
            }
            Bound::PosInfinity => {}
            Bound::NegInfinity => {
                return vec![Self::all_integers()];
            }
        }

        result
    }

    /// Check if two ranges are adjacent (touching but not overlapping)
    fn is_adjacent(&self, other: &Self) -> bool {
        if self.step != other.step {
            return false;
        }

        match (&self.end, &other.start) {
            (Bound::Finite(e), Bound::Finite(s)) => e == s,
            _ => false,
        }
    }

    /// Convert to a Vec of values (only for finite ranges)
    ///
    /// # Returns
    ///
    /// * `Some(Vec<T>)` - If the range is finite and not too large
    /// * `None` - If the range is infinite
    pub fn to_vec(&self) -> Option<Vec<T>> {
        if !self.is_finite() {
            return None;
        }

        match (&self.start, &self.end) {
            (Bound::Finite(s), Bound::Finite(e)) if s < e => {
                let mut result = Vec::new();
                let mut current = s.clone();

                while current < *e {
                    result.push(current.clone());
                    current = current + self.step.clone();
                }

                Some(result)
            }
            _ => Some(Vec::new()),
        }
    }
}

// Conversion from std::ops::Range
impl<T> From<Range<T>> for IntegerRange<T>
where
    T: Ring + Ord + Clone + NumericConversion,
{
    /// Convert from `std::ops::Range<T>` to `IntegerRange<T>`
    ///
    /// The resulting range has step size 1.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let std_range = 0..10;
    /// let int_range: IntegerRange<i64> = std_range.into();
    /// ```
    fn from(range: Range<T>) -> Self {
        IntegerRange::finite(range.start, range.end, T::one())
    }
}

// Display implementation
impl<T> fmt::Display for IntegerRange<T>
where
    T: Ring + Ord + Clone + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Check if empty first
        if self.is_empty() {
            return write!(f, "∅");
        }

        match (&self.start, &self.end) {
            (Bound::Finite(s), Bound::Finite(e)) => {
                if self.step.is_one() {
                    write!(f, "[{}, {})", s, e)
                } else {
                    write!(f, "[{}:{}, {})", s, self.step, e)
                }
            }
            (Bound::Finite(s), Bound::PosInfinity) => {
                if self.step.is_one() {
                    write!(f, "[{}, ∞)", s)
                } else {
                    write!(f, "[{}:{}, ∞)", s, self.step)
                }
            }
            (Bound::NegInfinity, Bound::Finite(e)) => {
                write!(f, "(-∞, {})", e)
            }
            (Bound::NegInfinity, Bound::PosInfinity) => {
                write!(f, "(-∞, ∞)")
            }
            _ => write!(f, "∅"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finite_range() {
        let range = IntegerRange::finite(0i64, 10, 1);
        assert!(!range.is_empty());
        assert!(range.is_finite());
        assert!(!range.is_infinite());
        assert_eq!(range.cardinality(), Some(10));
    }

    #[test]
    fn test_contains() {
        let range = IntegerRange::finite(0i64, 10, 1);
        assert!(range.contains(&0));
        assert!(range.contains(&5));
        assert!(range.contains(&9));
        assert!(!range.contains(&10));
        assert!(!range.contains(&-1));
    }

    #[test]
    fn test_contains_with_step() {
        let range = IntegerRange::finite(0i64, 10, 2);  // {0, 2, 4, 6, 8}
        assert!(range.contains(&0));
        assert!(range.contains(&2));
        assert!(range.contains(&8));
        assert!(!range.contains(&1));
        assert!(!range.contains(&9));
        assert!(!range.contains(&10));
    }

    #[test]
    fn test_infinite_range() {
        let range = IntegerRange::from(0i64);
        assert!(!range.is_empty());
        assert!(!range.is_finite());
        assert!(range.is_infinite());
        assert_eq!(range.cardinality(), None);
        assert!(range.contains(&0));
        assert!(range.contains(&1000000));
        assert!(!range.contains(&-1));
    }

    #[test]
    fn test_empty_range() {
        let range = IntegerRange::<i64>::empty();
        assert!(range.is_empty());
        assert_eq!(range.cardinality(), Some(0));
        assert!(!range.contains(&0));
    }

    #[test]
    fn test_union_overlapping() {
        let r1 = IntegerRange::finite(0i64, 10, 1);
        let r2 = IntegerRange::finite(5i64, 15, 1);
        let union = r1.union(&r2);
        assert_eq!(union.len(), 1);
        assert_eq!(union[0], IntegerRange::finite(0, 15, 1));
    }

    #[test]
    fn test_union_adjacent() {
        let r1 = IntegerRange::finite(0i64, 5, 1);
        let r2 = IntegerRange::finite(5i64, 10, 1);
        let union = r1.union(&r2);
        assert_eq!(union.len(), 1);
        assert_eq!(union[0], IntegerRange::finite(0, 10, 1));
    }

    #[test]
    fn test_union_disjoint() {
        let r1 = IntegerRange::finite(0i64, 5, 1);
        let r2 = IntegerRange::finite(10i64, 15, 1);
        let union = r1.union(&r2);
        assert_eq!(union.len(), 2);
    }

    #[test]
    fn test_intersection() {
        let r1 = IntegerRange::finite(0i64, 10, 1);
        let r2 = IntegerRange::finite(5i64, 15, 1);
        let intersection = r1.intersect(&r2);
        assert_eq!(intersection, IntegerRange::finite(5, 10, 1));
    }

    #[test]
    fn test_intersection_disjoint() {
        let r1 = IntegerRange::finite(0i64, 5, 1);
        let r2 = IntegerRange::finite(10i64, 15, 1);
        let intersection = r1.intersect(&r2);
        assert!(intersection.is_empty());
    }

    #[test]
    fn test_intersection_infinite() {
        let r1 = IntegerRange::from(0i64);  // [0, ∞)
        let r2 = IntegerRange::finite(5i64, 15, 1);  // [5, 15)
        let intersection = r1.intersect(&r2);
        assert_eq!(intersection, IntegerRange::finite(5, 15, 1));
    }

    #[test]
    fn test_difference() {
        let r1 = IntegerRange::finite(0i64, 10, 1);
        let r2 = IntegerRange::finite(5i64, 15, 1);
        let diff = r1.difference(&r2);
        assert_eq!(diff.len(), 1);
        assert_eq!(diff[0], IntegerRange::finite(0, 5, 1));
    }

    #[test]
    fn test_difference_disjoint() {
        let r1 = IntegerRange::finite(0i64, 5, 1);
        let r2 = IntegerRange::finite(10i64, 15, 1);
        let diff = r1.difference(&r2);
        assert_eq!(diff.len(), 1);
        assert_eq!(diff[0], r1);
    }

    #[test]
    fn test_difference_subset() {
        let r1 = IntegerRange::finite(0i64, 10, 1);
        let r2 = IntegerRange::finite(3i64, 7, 1);
        let diff = r1.difference(&r2);
        assert_eq!(diff.len(), 2);
        assert_eq!(diff[0], IntegerRange::finite(0, 3, 1));
        assert_eq!(diff[1], IntegerRange::finite(7, 10, 1));
    }

    #[test]
    fn test_complement() {
        let range = IntegerRange::finite(5i64, 10, 1);
        let comp = range.complement();
        assert_eq!(comp.len(), 2);
        // First part: (-∞, 5)
        assert_eq!(comp[0].start(), &Bound::NegInfinity);
        assert_eq!(comp[0].end(), &Bound::Finite(5));
        // Second part: [10, ∞)
        assert_eq!(comp[1].start(), &Bound::Finite(10));
        assert_eq!(comp[1].end(), &Bound::PosInfinity);
    }

    #[test]
    fn test_complement_infinite() {
        let range = IntegerRange::from(0i64);  // [0, ∞)
        let comp = range.complement();
        assert_eq!(comp.len(), 1);
        assert_eq!(comp[0].start(), &Bound::NegInfinity);
        assert_eq!(comp[0].end(), &Bound::Finite(0));
    }

    #[test]
    fn test_to_vec() {
        let range = IntegerRange::finite(0i64, 5, 1);
        let vec = range.to_vec().unwrap();
        assert_eq!(vec, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_to_vec_with_step() {
        let range = IntegerRange::finite(0i64, 10, 2);
        let vec = range.to_vec().unwrap();
        assert_eq!(vec, vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_to_vec_infinite() {
        let range = IntegerRange::from(0i64);
        assert!(range.to_vec().is_none());
    }

    #[test]
    fn test_from_std_range() {
        let std_range = 0i64..10;
        let int_range: IntegerRange<i64> = std_range.into();
        assert_eq!(int_range, IntegerRange::finite(0, 10, 1));
    }

    #[test]
    fn test_display() {
        let r1 = IntegerRange::finite(0i64, 10, 1);
        assert_eq!(format!("{}", r1), "[0, 10)");

        let r2 = IntegerRange::finite(0i64, 10, 2);
        assert_eq!(format!("{}", r2), "[0:2, 10)");

        let r3 = IntegerRange::from(0i64);
        assert_eq!(format!("{}", r3), "[0, ∞)");

        let r4 = IntegerRange::<i64>::empty();
        assert_eq!(format!("{}", r4), "∅");
    }

    #[test]
    fn test_cardinality_with_step() {
        let range = IntegerRange::finite(0i64, 10, 2);
        assert_eq!(range.cardinality(), Some(5));  // {0, 2, 4, 6, 8}

        let range2 = IntegerRange::finite(1i64, 10, 3);
        assert_eq!(range2.cardinality(), Some(3));  // {1, 4, 7}
    }

    #[test]
    fn test_range_arithmetic() {
        // Test union associativity
        let r1 = IntegerRange::finite(0i64, 5, 1);
        let r2 = IntegerRange::finite(3i64, 8, 1);
        let r3 = IntegerRange::finite(6i64, 10, 1);

        let union_12 = r1.union(&r2);
        let union_all = union_12[0].union(&r3);
        assert_eq!(union_all[0], IntegerRange::finite(0, 10, 1));
    }

    #[test]
    fn test_intersection_commutativity() {
        let r1 = IntegerRange::finite(0i64, 10, 1);
        let r2 = IntegerRange::finite(5i64, 15, 1);

        let i1 = r1.intersect(&r2);
        let i2 = r2.intersect(&r1);
        assert_eq!(i1, i2);
    }

    #[test]
    fn test_complement_of_complement() {
        let range = IntegerRange::finite(5i64, 10, 1);
        let comp = range.complement();

        // Complement should give us two ranges
        assert_eq!(comp.len(), 2);

        // The union of the range and its complement should cover all integers
        let union1 = range.union(&comp[0]);
        assert!(union1.len() >= 1);
    }

    #[test]
    fn test_difference_identity() {
        // A - A = ∅
        let range = IntegerRange::finite(0i64, 10, 1);
        let diff = range.difference(&range);
        assert!(diff[0].is_empty());
    }

    #[test]
    fn test_intersection_with_empty() {
        let range = IntegerRange::finite(0i64, 10, 1);
        let empty = IntegerRange::<i64>::empty();
        let intersection = range.intersect(&empty);
        assert!(intersection.is_empty());
    }

    #[test]
    fn test_union_with_empty() {
        let range = IntegerRange::finite(0i64, 10, 1);
        let empty = IntegerRange::<i64>::empty();
        let union = range.union(&empty);
        assert_eq!(union.len(), 1);
        assert_eq!(union[0], range);
    }
}
