//! # Free Monoids
//!
//! This module provides free monoids and their elements.
//!
//! A free monoid on a set of generators is the monoid of all finite words
//! over those generators, with concatenation as the operation.

use std::fmt::{self, Display};
use std::hash::{Hash, Hasher};

/// An element of a free monoid (a word)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreeMonoidElement {
    /// The word as a sequence of generator indices
    word: Vec<usize>,
}

impl FreeMonoidElement {
    /// Create a new word from generator indices
    pub fn new(word: Vec<usize>) -> Self {
        FreeMonoidElement { word }
    }

    /// Create the empty word (identity)
    pub fn identity() -> Self {
        FreeMonoidElement { word: Vec::new() }
    }

    /// Get the underlying word
    pub fn word(&self) -> &[usize] {
        &self.word
    }

    /// Length of the word
    pub fn len(&self) -> usize {
        self.word.len()
    }

    /// Check if this is the empty word
    pub fn is_empty(&self) -> bool {
        self.word.is_empty()
    }

    /// Concatenate two words
    pub fn mul(&self, other: &Self) -> Self {
        let mut new_word = self.word.clone();
        new_word.extend_from_slice(&other.word);
        FreeMonoidElement::new(new_word)
    }

    /// Compute the inverse of the word (for free groups, not monoids)
    /// This is included for compatibility but may not make sense for monoids
    pub fn inverse(&self) -> Self {
        let mut inv_word = self.word.clone();
        inv_word.reverse();
        FreeMonoidElement::new(inv_word)
    }

    /// Get the i-th letter
    pub fn get(&self, index: usize) -> Option<usize> {
        self.word.get(index).copied()
    }
}

impl Hash for FreeMonoidElement {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.word.hash(state);
    }
}

/// A free monoid on a set of generators
#[derive(Debug, Clone)]
pub struct FreeMonoid {
    /// Number of generators
    num_generators: usize,
    /// Names of generators
    generator_names: Vec<String>,
}

impl FreeMonoid {
    /// Create a new free monoid with named generators
    pub fn new(generators: Vec<String>) -> Self {
        let num_generators = generators.len();
        FreeMonoid {
            num_generators,
            generator_names: generators,
        }
    }

    /// Create a new free monoid with n generators (x_0, x_1, ..., x_{n-1})
    pub fn with_rank(n: usize) -> Self {
        let generator_names = (0..n).map(|i| format!("x_{}", i)).collect();
        FreeMonoid {
            num_generators: n,
            generator_names,
        }
    }

    /// Get the number of generators
    pub fn rank(&self) -> usize {
        self.num_generators
    }

    /// Get the generator names
    pub fn generators(&self) -> &[String] {
        &self.generator_names
    }

    /// Create the identity element
    pub fn identity(&self) -> FreeMonoidElement {
        FreeMonoidElement::identity()
    }

    /// Create a generator element
    pub fn gen(&self, index: usize) -> Option<FreeMonoidElement> {
        if index < self.num_generators {
            Some(FreeMonoidElement::new(vec![index]))
        } else {
            None
        }
    }

    /// Create an element from a word
    pub fn element(&self, word: Vec<usize>) -> FreeMonoidElement {
        FreeMonoidElement::new(word)
    }

    /// Format a word as a string
    pub fn word_to_string(&self, word: &FreeMonoidElement) -> String {
        if word.is_empty() {
            return "1".to_string();
        }

        word.word()
            .iter()
            .map(|&i| {
                if i < self.generator_names.len() {
                    self.generator_names[i].clone()
                } else {
                    format!("x_{}", i)
                }
            })
            .collect::<Vec<_>>()
            .join("*")
    }
}

/// Check if an object is a free monoid
pub fn is_free_monoid(_obj: &FreeMonoid) -> bool {
    // In Rust, this is always true if we have the object
    true
}

/// Check if an object is a free monoid element
pub fn is_free_monoid_element(_obj: &FreeMonoidElement) -> bool {
    // In Rust, this is always true if we have the object
    true
}

impl Display for FreeMonoid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Free monoid on {} generators [{}]",
            self.num_generators,
            self.generator_names.join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_free_monoid() {
        let M = FreeMonoid::new(vec!["x".to_string(), "y".to_string()]);
        assert_eq!(M.rank(), 2);
    }

    #[test]
    fn test_identity() {
        let M = FreeMonoid::new(vec!["a".to_string(), "b".to_string()]);
        let id = M.identity();
        assert!(id.is_empty());
        assert_eq!(id.len(), 0);
    }

    #[test]
    fn test_generator() {
        let M = FreeMonoid::new(vec!["x".to_string(), "y".to_string()]);
        let x = M.gen(0).unwrap();
        let y = M.gen(1).unwrap();

        assert_eq!(x.len(), 1);
        assert_eq!(y.len(), 1);
        assert_eq!(x.word(), &[0]);
        assert_eq!(y.word(), &[1]);
    }

    #[test]
    fn test_multiplication() {
        let M = FreeMonoid::new(vec!["x".to_string(), "y".to_string()]);
        let x = M.gen(0).unwrap();
        let y = M.gen(1).unwrap();

        let xy = x.mul(&y);
        assert_eq!(xy.word(), &[0, 1]);
        assert_eq!(xy.len(), 2);
    }

    #[test]
    fn test_word_to_string() {
        let M = FreeMonoid::new(vec!["a".to_string(), "b".to_string()]);
        let a = M.gen(0).unwrap();
        let b = M.gen(1).unwrap();
        let ab = a.mul(&b);

        assert_eq!(M.word_to_string(&ab), "a*b");
        assert_eq!(M.word_to_string(&M.identity()), "1");
    }

    #[test]
    fn test_is_FreeMonoid() {
        let M = FreeMonoid::with_rank(3);
        assert!(is_free_monoid(&M));
    }

    #[test]
    fn test_is_FreeMonoidElement() {
        let elem = FreeMonoidElement::new(vec![0, 1, 0]);
        assert!(is_free_monoid_element(&elem));
    }
}
