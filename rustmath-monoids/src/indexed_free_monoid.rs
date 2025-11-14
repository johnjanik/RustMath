//! # Indexed Free Monoids
//!
//! This module provides indexed free monoids where generators can be indexed
//! by arbitrary indices (not just 0, 1, 2, ...).

use std::collections::HashMap;
use std::hash::Hash;
use std::fmt::Debug;

/// An element of an indexed monoid
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedMonoidElement<I: Eq + Hash + Clone> {
    /// The word as a sequence of generator indices
    word: Vec<I>,
}

impl<I: Eq + Hash + Clone> IndexedMonoidElement<I> {
    /// Create a new element
    pub fn new(word: Vec<I>) -> Self {
        IndexedMonoidElement { word }
    }

    /// Create the identity element
    pub fn identity() -> Self {
        IndexedMonoidElement { word: Vec::new() }
    }

    /// Get the word
    pub fn word(&self) -> &[I] {
        &self.word
    }

    /// Length of the word
    pub fn len(&self) -> usize {
        self.word.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.word.is_empty()
    }

    /// Multiply two elements
    pub fn mul(&self, other: &Self) -> Self {
        let mut new_word = self.word.clone();
        new_word.extend_from_slice(&other.word);
        IndexedMonoidElement::new(new_word)
    }
}

/// An indexed free monoid element (non-abelian)
pub type IndexedFreeMonoidElement<I> = IndexedMonoidElement<I>;

/// An indexed free abelian monoid element
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedFreeAbelianMonoidElement<I: Eq + Hash + Clone> {
    /// Exponents for each index
    exponents: HashMap<I, usize>,
}

impl<I: Eq + Hash + Clone> IndexedFreeAbelianMonoidElement<I> {
    /// Create a new element
    pub fn new(exponents: HashMap<I, usize>) -> Self {
        // Remove zero exponents
        let mut clean_exponents = HashMap::new();
        for (idx, exp) in exponents {
            if exp > 0 {
                clean_exponents.insert(idx, exp);
            }
        }
        IndexedFreeAbelianMonoidElement {
            exponents: clean_exponents,
        }
    }

    /// Create the identity
    pub fn identity() -> Self {
        IndexedFreeAbelianMonoidElement {
            exponents: HashMap::new(),
        }
    }

    /// Get exponents
    pub fn exponents(&self) -> &HashMap<I, usize> {
        &self.exponents
    }

    /// Multiply two elements
    pub fn mul(&self, other: &Self) -> Self {
        let mut new_exponents = self.exponents.clone();
        for (idx, exp) in &other.exponents {
            *new_exponents.entry(idx.clone()).or_insert(0) += exp;
        }
        IndexedFreeAbelianMonoidElement::new(new_exponents)
    }
}

/// An indexed monoid
#[derive(Debug, Clone)]
pub struct IndexedMonoid<I: Eq + Hash + Clone> {
    /// The set of indices
    indices: Vec<I>,
    /// Whether the monoid is abelian
    abelian: bool,
}

impl<I: Eq + Hash + Clone> IndexedMonoid<I> {
    /// Create a new indexed monoid
    pub fn new(indices: Vec<I>, abelian: bool) -> Self {
        IndexedMonoid { indices, abelian }
    }

    /// Get the indices
    pub fn indices(&self) -> &[I] {
        &self.indices
    }

    /// Check if abelian
    pub fn is_abelian(&self) -> bool {
        self.abelian
    }

    /// Create the identity element
    pub fn identity(&self) -> IndexedMonoidElement<I> {
        IndexedMonoidElement::identity()
    }
}

/// An indexed free monoid
pub type IndexedFreeMonoid<I> = IndexedMonoid<I>;

/// An indexed free abelian monoid
pub type IndexedFreeAbelianMonoid<I> = IndexedMonoid<I>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexed_monoid_element() {
        let elem = IndexedMonoidElement::new(vec!["a", "b", "c"]);
        assert_eq!(elem.len(), 3);
        assert!(!elem.is_empty());
    }

    #[test]
    fn test_indexed_monoid_identity() {
        let id = IndexedMonoidElement::<&str>::identity();
        assert!(id.is_empty());
    }

    #[test]
    fn test_indexed_monoid_mul() {
        let e1 = IndexedMonoidElement::new(vec![1, 2]);
        let e2 = IndexedMonoidElement::new(vec![3, 4]);
        let e3 = e1.mul(&e2);
        assert_eq!(e3.word(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_indexed_abelian_monoid_element() {
        let mut exps = HashMap::new();
        exps.insert("x", 2);
        exps.insert("y", 3);

        let elem = IndexedFreeAbelianMonoidElement::new(exps);
        assert_eq!(elem.exponents().get("x"), Some(&2));
        assert_eq!(elem.exponents().get("y"), Some(&3));
    }

    #[test]
    fn test_indexed_abelian_mul() {
        let mut exps1 = HashMap::new();
        exps1.insert("x", 2);

        let mut exps2 = HashMap::new();
        exps2.insert("x", 3);
        exps2.insert("y", 1);

        let e1 = IndexedFreeAbelianMonoidElement::new(exps1);
        let e2 = IndexedFreeAbelianMonoidElement::new(exps2);
        let e3 = e1.mul(&e2);

        assert_eq!(e3.exponents().get("x"), Some(&5));
        assert_eq!(e3.exponents().get("y"), Some(&1));
    }
}
