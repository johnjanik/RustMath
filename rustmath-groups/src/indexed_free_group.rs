//! Indexed Free Groups
//!
//! This module implements free groups and free abelian groups where generators
//! are indexed by arbitrary types rather than traditional integer indices.
//! This provides flexibility for representing groups with more complex indexing schemes.
//!
//! # Examples
//!
//! ```
//! use rustmath_groups::indexed_free_group::IndexedFreeGroup;
//!
//! // Create a free group with string-indexed generators
//! let indices = vec!["x".to_string(), "y".to_string(), "z".to_string()];
//! let g = IndexedFreeGroup::new(indices);
//! ```

use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::ops::Mul;

use crate::group_traits::Group;

/// An indexed group base trait
///
/// This trait provides the basic interface for groups with indexed generators.
pub trait IndexedGroup {
    /// The type used for indexing generators
    type Index: Clone + Eq + Hash + fmt::Debug;

    /// The element type of this group
    type Element;

    /// Returns the rank (number of generators)
    fn rank(&self) -> Option<usize>;

    /// Returns the generator indexed by the given index
    fn gen(&self, index: &Self::Index) -> Option<Self::Element>;

    /// Returns the identity element
    fn one(&self) -> Self::Element;

    /// Returns the order of the group (None for infinite groups)
    fn order(&self) -> Option<usize>;
}

/// A free group with generators indexed by an arbitrary type
///
/// Unlike traditional free groups where generators are numbered 0, 1, 2, ...,
/// this allows generators to be indexed by any hashable type (strings, tuples, etc.).
#[derive(Debug, Clone)]
pub struct IndexedFreeGroup<I: Clone + Eq + Hash + fmt::Debug> {
    /// The index set for generators
    indices: Vec<I>,
    /// Prefix for printing generators
    prefix: String,
}

impl<I: Clone + Eq + Hash + fmt::Debug> IndexedFreeGroup<I> {
    /// Create a new indexed free group
    ///
    /// # Arguments
    ///
    /// * `indices` - The set of indices for generators
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::indexed_free_group::IndexedFreeGroup;
    ///
    /// let g = IndexedFreeGroup::new(vec!["a", "b", "c"]);
    /// ```
    pub fn new(indices: Vec<I>) -> Self {
        Self {
            indices,
            prefix: "x".to_string(),
        }
    }

    /// Create a new indexed free group with a custom prefix
    pub fn with_prefix(indices: Vec<I>, prefix: String) -> Self {
        Self { indices, prefix }
    }

    /// Returns a reference to the index set
    pub fn indices(&self) -> &[I] {
        &self.indices
    }

    /// Returns the prefix used for printing
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Returns the number of generators
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::indexed_free_group::IndexedFreeGroup;
    ///
    /// let g = IndexedFreeGroup::new(vec!["a", "b", "c"]);
    /// assert_eq!(g.ngens(), 3);
    /// ```
    pub fn ngens(&self) -> usize {
        self.indices.len()
    }
}

impl<I: Clone + Eq + Hash + fmt::Debug + fmt::Display> IndexedGroup for IndexedFreeGroup<I> {
    type Index = I;
    type Element = IndexedFreeGroupElement<I>;

    fn rank(&self) -> Option<usize> {
        Some(self.indices.len())
    }

    fn gen(&self, index: &Self::Index) -> Option<Self::Element> {
        if self.indices.contains(index) {
            Some(IndexedFreeGroupElement {
                parent: self.clone(),
                word: vec![(index.clone(), 1)],
            })
        } else {
            None
        }
    }

    fn one(&self) -> Self::Element {
        IndexedFreeGroupElement {
            parent: self.clone(),
            word: Vec::new(),
        }
    }

    fn order(&self) -> Option<usize> {
        if self.indices.is_empty() {
            Some(1) // Trivial group
        } else {
            None // Infinite
        }
    }
}

impl<I: Clone + Eq + Hash + fmt::Debug> fmt::Display for IndexedFreeGroup<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Free group indexed by {} indices", self.indices.len())
    }
}

impl<I: Clone + Eq + Hash + fmt::Debug + fmt::Display> Group for IndexedFreeGroup<I> {
    type Element = IndexedFreeGroupElement<I>;

    fn identity(&self) -> Self::Element {
        self.one()
    }

    fn is_finite(&self) -> bool {
        // Free groups are infinite (unless they have no generators)
        self.ngens() == 0
    }

    fn order(&self) -> Option<usize> {
        if self.ngens() == 0 {
            Some(1) // Trivial group
        } else {
            None // Infinite
        }
    }

    fn contains(&self, element: &Self::Element) -> bool {
        // Check if the element uses only generators from this group
        // In a full implementation, we'd verify the indices match
        element.word().iter().all(|(idx, _)| self.indices.contains(idx))
    }
}

/// An element of an indexed free group
///
/// Elements are represented as words: sequences of (index, exponent) pairs.
#[derive(Clone, Hash, Debug)]
pub struct IndexedFreeGroupElement<I: Clone + Eq + Hash + fmt::Debug + fmt::Display> {
    parent: IndexedFreeGroup<I>,
    word: Vec<(I, i32)>,
}

impl<I: Clone + Eq + Hash + fmt::Debug + fmt::Display> IndexedFreeGroupElement<I> {
    /// Create a new element from a word
    pub fn new(parent: IndexedFreeGroup<I>, word: Vec<(I, i32)>) -> Self {
        Self { parent, word }.reduce()
    }

    /// Returns a reference to the parent group
    pub fn parent(&self) -> &IndexedFreeGroup<I> {
        &self.parent
    }

    /// Returns a reference to the word representation
    pub fn word(&self) -> &[(I, i32)] {
        &self.word
    }

    /// Returns the length of the word (sum of absolute values of exponents)
    pub fn length(&self) -> usize {
        self.word.iter().map(|(_, exp)| exp.abs() as usize).sum()
    }

    /// Check if this is the identity element
    pub fn is_identity(&self) -> bool {
        self.word.is_empty()
    }

    /// Multiply this element with another
    pub fn multiply(&self, other: &Self) -> Self {
        let mut new_word = self.word.clone();
        new_word.extend(other.word.iter().cloned());
        Self {
            parent: self.parent.clone(),
            word: new_word,
        }
        .reduce()
    }

    /// Compute the inverse
    pub fn inverse(&self) -> Self {
        let mut inv_word: Vec<(I, i32)> = self
            .word
            .iter()
            .rev()
            .map(|(idx, exp)| (idx.clone(), -exp))
            .collect();

        Self {
            parent: self.parent.clone(),
            word: inv_word,
        }
    }

    /// Raise to a power
    pub fn pow(&self, n: i32) -> Self {
        if n == 0 {
            return self.parent.one();
        }

        let mut result = self.parent.one();
        let base = if n > 0 { self.clone() } else { self.inverse() };

        for _ in 0..n.abs() {
            result = result.multiply(&base);
        }

        result
    }

    /// Reduce the word by cancelling consecutive inverse generators
    fn reduce(mut self) -> Self {
        let mut i = 0;
        while i + 1 < self.word.len() {
            if self.word[i].0 == self.word[i + 1].0 {
                // Same index, combine exponents
                let new_exp = self.word[i].1 + self.word[i + 1].1;
                if new_exp == 0 {
                    // Cancel out
                    self.word.remove(i);
                    self.word.remove(i);
                    if i > 0 {
                        i -= 1;
                    }
                } else {
                    self.word[i].1 = new_exp;
                    self.word.remove(i + 1);
                }
            } else {
                i += 1;
            }
        }

        // Remove any zero exponents
        self.word.retain(|(_, exp)| *exp != 0);

        self
    }

    /// Convert to a list representation
    pub fn to_word_list(&self) -> Vec<(I, i32)> {
        self.word.clone()
    }
}

impl<I: Clone + Eq + Hash + fmt::Debug + fmt::Display> fmt::Display for IndexedFreeGroupElement<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.word.is_empty() {
            return write!(f, "1");
        }

        for (i, (idx, exp)) in self.word.iter().enumerate() {
            if i > 0 {
                write!(f, "*")?;
            }
            if *exp == 1 {
                write!(f, "{}", idx)?;
            } else if *exp == -1 {
                write!(f, "{}^-1", idx)?;
            } else {
                write!(f, "{}^{}", idx, exp)?;
            }
        }
        Ok(())
    }
}

impl<I: Clone + Eq + Hash + fmt::Debug + fmt::Display> PartialEq for IndexedFreeGroupElement<I> {
    fn eq(&self, other: &Self) -> bool {
        self.word == other.word
    }
}

impl<I: Clone + Eq + Hash + fmt::Debug + fmt::Display> Eq for IndexedFreeGroupElement<I> {}

impl<I: Clone + Eq + Hash + fmt::Debug + fmt::Display> Default for IndexedFreeGroupElement<I> {
    /// Create a default element (identity of a trivial group)
    fn default() -> Self {
        <Self as crate::group_traits::GroupElement>::identity()
    }
}

impl<I: Clone + Eq + Hash + fmt::Debug + fmt::Display> Mul for IndexedFreeGroupElement<I> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.multiply(&rhs)
    }
}

impl<I: Clone + Eq + Hash + fmt::Debug + fmt::Display> Mul for &IndexedFreeGroupElement<I> {
    type Output = IndexedFreeGroupElement<I>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.multiply(rhs)
    }
}

impl<I: Clone + Eq + Hash + fmt::Debug + fmt::Display> crate::group_traits::GroupElement for IndexedFreeGroupElement<I> {
    fn identity() -> Self {
        // We need a parent group to create an identity element
        // This is a limitation of the design - we'll create a trivial group
        let parent = IndexedFreeGroup::new(vec![]);
        parent.one()
    }

    fn inverse(&self) -> Self {
        self.inverse()
    }

    fn op(&self, other: &Self) -> Self {
        self.multiply(other)
    }

    fn pow(&self, n: i64) -> Self {
        self.pow(n as i32)
    }
}

/// A free abelian group with generators indexed by an arbitrary type
///
/// This is the commutative version of IndexedFreeGroup, where elements
/// are represented as dictionaries mapping indices to exponents.
#[derive(Debug, Clone)]
pub struct IndexedFreeAbelianGroup<I: Clone + Eq + Hash + fmt::Debug> {
    /// The index set for generators
    indices: Vec<I>,
    /// Prefix for printing generators
    prefix: String,
}

impl<I: Clone + Eq + Hash + fmt::Debug> IndexedFreeAbelianGroup<I> {
    /// Create a new indexed free abelian group
    pub fn new(indices: Vec<I>) -> Self {
        Self {
            indices,
            prefix: "x".to_string(),
        }
    }

    /// Create a new indexed free abelian group with a custom prefix
    pub fn with_prefix(indices: Vec<I>, prefix: String) -> Self {
        Self { indices, prefix }
    }

    /// Returns a reference to the index set
    pub fn indices(&self) -> &[I] {
        &self.indices
    }

    /// Returns the prefix used for printing
    pub fn prefix(&self) -> &str {
        &self.prefix
    }
}

impl<I: Clone + Eq + Hash + fmt::Debug + fmt::Display> IndexedGroup for IndexedFreeAbelianGroup<I> {
    type Index = I;
    type Element = IndexedFreeAbelianGroupElement<I>;

    fn rank(&self) -> Option<usize> {
        Some(self.indices.len())
    }

    fn gen(&self, index: &Self::Index) -> Option<Self::Element> {
        if self.indices.contains(index) {
            let mut exponents = HashMap::new();
            exponents.insert(index.clone(), 1);
            Some(IndexedFreeAbelianGroupElement {
                parent: self.clone(),
                exponents,
            })
        } else {
            None
        }
    }

    fn one(&self) -> Self::Element {
        IndexedFreeAbelianGroupElement {
            parent: self.clone(),
            exponents: HashMap::new(),
        }
    }

    fn order(&self) -> Option<usize> {
        if self.indices.is_empty() {
            Some(1) // Trivial group
        } else {
            None // Infinite
        }
    }
}

impl<I: Clone + Eq + Hash + fmt::Debug> fmt::Display for IndexedFreeAbelianGroup<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Free abelian group indexed by {} indices",
            self.indices.len()
        )
    }
}

impl<I: Clone + Eq + Hash + fmt::Debug + fmt::Display> Group for IndexedFreeAbelianGroup<I> {
    type Element = IndexedFreeAbelianGroupElement<I>;

    fn identity(&self) -> Self::Element {
        self.one()
    }

    fn is_finite(&self) -> bool {
        // Free abelian groups are infinite (unless they have no generators)
        self.indices.is_empty()
    }

    fn order(&self) -> Option<usize> {
        if self.indices.is_empty() {
            Some(1) // Trivial group
        } else {
            None // Infinite
        }
    }

    fn contains(&self, element: &Self::Element) -> bool {
        // Check if the element uses only indices from this group
        element.exponents.keys().all(|idx| self.indices.contains(idx))
    }

    fn is_abelian(&self) -> bool {
        // Free abelian groups are always abelian
        true
    }
}

/// An element of an indexed free abelian group
///
/// Elements are represented as dictionaries mapping indices to exponents.
#[derive(Clone, Hash, Debug)]
pub struct IndexedFreeAbelianGroupElement<I: Clone + Eq + Hash + fmt::Debug + fmt::Display> {
    parent: IndexedFreeAbelianGroup<I>,
    exponents: HashMap<I, i32>,
}

impl<I: Clone + Eq + Hash + fmt::Debug + fmt::Display> IndexedFreeAbelianGroupElement<I> {
    /// Create a new element from exponents
    pub fn new(parent: IndexedFreeAbelianGroup<I>, exponents: HashMap<I, i32>) -> Self {
        let mut elem = Self { parent, exponents };
        elem.simplify();
        elem
    }

    /// Returns a reference to the parent group
    pub fn parent(&self) -> &IndexedFreeAbelianGroup<I> {
        &self.parent
    }

    /// Returns a reference to the exponent map
    pub fn exponents(&self) -> &HashMap<I, i32> {
        &self.exponents
    }

    /// Check if this is the identity element
    pub fn is_identity(&self) -> bool {
        self.exponents.is_empty()
    }

    /// Multiply (add in additive notation) with another element
    pub fn multiply(&self, other: &Self) -> Self {
        let mut new_exps = self.exponents.clone();

        for (idx, exp) in &other.exponents {
            *new_exps.entry(idx.clone()).or_insert(0) += exp;
        }

        Self::new(self.parent.clone(), new_exps)
    }

    /// Compute the inverse (negation in additive notation)
    pub fn inverse(&self) -> Self {
        let inv_exps: HashMap<I, i32> = self
            .exponents
            .iter()
            .map(|(idx, exp)| (idx.clone(), -exp))
            .collect();

        Self::new(self.parent.clone(), inv_exps)
    }

    /// Raise to a power (scalar multiplication in additive notation)
    pub fn pow(&self, n: i32) -> Self {
        let scaled_exps: HashMap<I, i32> = self
            .exponents
            .iter()
            .map(|(idx, exp)| (idx.clone(), exp * n))
            .collect();

        Self::new(self.parent.clone(), scaled_exps)
    }

    /// Remove zero exponents
    fn simplify(&mut self) {
        self.exponents.retain(|_, exp| *exp != 0);
    }
}

impl<I: Clone + Eq + Hash + fmt::Debug + fmt::Display> fmt::Display
    for IndexedFreeAbelianGroupElement<I>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.exponents.is_empty() {
            return write!(f, "0");
        }

        let mut first = true;
        for (idx, exp) in &self.exponents {
            if !first {
                write!(f, " + ")?;
            }
            first = false;

            if *exp == 1 {
                write!(f, "{}", idx)?;
            } else {
                write!(f, "{}*{}", exp, idx)?;
            }
        }
        Ok(())
    }
}

impl<I: Clone + Eq + Hash + fmt::Debug + fmt::Display> PartialEq for IndexedFreeAbelianGroupElement<I> {
    fn eq(&self, other: &Self) -> bool {
        self.exponents == other.exponents
    }
}

impl<I: Clone + Eq + Hash + fmt::Debug + fmt::Display> Eq for IndexedFreeAbelianGroupElement<I> {}

impl<I: Clone + Eq + Hash + fmt::Debug + fmt::Display> Default for IndexedFreeAbelianGroupElement<I> {
    /// Create a default element (identity of a trivial group)
    fn default() -> Self {
        <Self as crate::group_traits::GroupElement>::identity()
    }
}

impl<I: Clone + Eq + Hash + fmt::Debug + fmt::Display> Mul for IndexedFreeAbelianGroupElement<I> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.multiply(&rhs)
    }
}

impl<I: Clone + Eq + Hash + fmt::Debug + fmt::Display> Mul for &IndexedFreeAbelianGroupElement<I> {
    type Output = IndexedFreeAbelianGroupElement<I>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.multiply(rhs)
    }
}

impl<I: Clone + Eq + Hash + fmt::Debug + fmt::Display> crate::group_traits::GroupElement for IndexedFreeAbelianGroupElement<I> {
    fn identity() -> Self {
        // We need a parent group to create an identity element
        // This is a limitation of the design - we'll create a trivial group
        let parent = IndexedFreeAbelianGroup::new(vec![]);
        parent.one()
    }

    fn inverse(&self) -> Self {
        self.inverse()
    }

    fn op(&self, other: &Self) -> Self {
        self.multiply(other)
    }

    fn pow(&self, n: i64) -> Self {
        self.pow(n as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexed_free_group_creation() {
        let g = IndexedFreeGroup::new(vec!["a", "b", "c"]);
        assert_eq!(g.rank(), Some(3));
    }

    #[test]
    fn test_indexed_free_group_generators() {
        let g = IndexedFreeGroup::new(vec!["x", "y"]);
        let x = g.gen(&"x").unwrap();
        let y = g.gen(&"y").unwrap();

        assert!(!x.is_identity());
        assert!(!y.is_identity());
    }

    #[test]
    fn test_indexed_free_group_identity() {
        let g = IndexedFreeGroup::new(vec!["a", "b"]);
        let id = g.one();
        assert!(id.is_identity());
    }

    #[test]
    fn test_indexed_free_group_multiplication() {
        let g = IndexedFreeGroup::new(vec!["a", "b"]);
        let a = g.gen(&"a").unwrap();
        let b = g.gen(&"b").unwrap();

        let ab = a.multiply(&b);
        assert_eq!(ab.length(), 2);
    }

    #[test]
    fn test_indexed_free_group_inverse() {
        let g = IndexedFreeGroup::new(vec!["a"]);
        let a = g.gen(&"a").unwrap();
        let a_inv = a.inverse();

        let prod = a.multiply(&a_inv);
        assert!(prod.is_identity());
    }

    #[test]
    fn test_indexed_free_group_cancellation() {
        let g = IndexedFreeGroup::new(vec!["a", "b"]);
        let a = g.gen(&"a").unwrap();
        let b = g.gen(&"b").unwrap();
        let a_inv = a.inverse();

        // a * a^-1 * b should reduce to b
        let result = a.multiply(&a_inv).multiply(&b);
        assert_eq!(result, b);
    }

    #[test]
    fn test_indexed_free_group_power() {
        let g = IndexedFreeGroup::new(vec!["a"]);
        let a = g.gen(&"a").unwrap();

        let a3 = a.pow(3);
        assert_eq!(a3.length(), 3);

        let a_neg2 = a.pow(-2);
        assert_eq!(a_neg2.length(), 2);
    }

    #[test]
    fn test_indexed_free_abelian_group_creation() {
        let g = IndexedFreeAbelianGroup::new(vec![1, 2, 3]);
        assert_eq!(g.rank(), Some(3));
    }

    #[test]
    fn test_indexed_free_abelian_group_generators() {
        let g = IndexedFreeAbelianGroup::new(vec![0, 1]);
        let x = g.gen(&0).unwrap();
        let y = g.gen(&1).unwrap();

        assert!(!x.is_identity());
        assert!(!y.is_identity());
    }

    #[test]
    fn test_indexed_free_abelian_group_commutativity() {
        let g = IndexedFreeAbelianGroup::new(vec!["a", "b"]);
        let a = g.gen(&"a").unwrap();
        let b = g.gen(&"b").unwrap();

        let ab = a.multiply(&b);
        let ba = b.multiply(&a);

        assert_eq!(ab, ba);
    }

    #[test]
    fn test_indexed_free_abelian_group_inverse() {
        let g = IndexedFreeAbelianGroup::new(vec![1]);
        let x = g.gen(&1).unwrap();
        let x_inv = x.inverse();

        let prod = x.multiply(&x_inv);
        assert!(prod.is_identity());
    }

    #[test]
    fn test_indexed_free_abelian_group_power() {
        let g = IndexedFreeAbelianGroup::new(vec!["x"]);
        let x = g.gen(&"x").unwrap();

        let x3 = x.pow(3);
        assert_eq!(x3.exponents().get(&"x"), Some(&3));

        let x_neg2 = x.pow(-2);
        assert_eq!(x_neg2.exponents().get(&"x"), Some(&-2));
    }

    #[test]
    fn test_trivial_group() {
        let g: IndexedFreeGroup<i32> = IndexedFreeGroup::new(vec![]);
        assert_eq!(g.order(), Some(1));
        assert_eq!(g.rank(), Some(0));

        let ag: IndexedFreeAbelianGroup<i32> = IndexedFreeAbelianGroup::new(vec![]);
        assert_eq!(ag.order(), Some(1));
        assert_eq!(ag.rank(), Some(0));
    }
}
