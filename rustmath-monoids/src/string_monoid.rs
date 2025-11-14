//! # String Monoids
//!
//! This module provides string monoids, which are monoids of strings over an alphabet.

use std::fmt::{self, Display};

/// An element of a string monoid (a string)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StringMonoidElement {
    /// The string
    string: String,
}

impl StringMonoidElement {
    /// Create a new string element
    pub fn new(s: String) -> Self {
        StringMonoidElement { string: s }
    }

    /// Create the empty string (identity)
    pub fn identity() -> Self {
        StringMonoidElement {
            string: String::new(),
        }
    }

    /// Get the underlying string
    pub fn as_str(&self) -> &str {
        &self.string
    }

    /// Length of the string
    pub fn len(&self) -> usize {
        self.string.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.string.is_empty()
    }

    /// Concatenate two strings
    pub fn mul(&self, other: &Self) -> Self {
        let mut new_string = self.string.clone();
        new_string.push_str(&other.string);
        StringMonoidElement::new(new_string)
    }

    /// Reverse the string
    pub fn reverse(&self) -> Self {
        let reversed: String = self.string.chars().rev().collect();
        StringMonoidElement::new(reversed)
    }

    /// Get the i-th character
    pub fn get(&self, index: usize) -> Option<char> {
        self.string.chars().nth(index)
    }
}

impl Display for StringMonoidElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            write!(f, "ε")
        } else {
            write!(f, "{}", self.string)
        }
    }
}

/// A string monoid over an alphabet
#[derive(Debug, Clone)]
pub struct StringMonoid {
    /// The alphabet
    alphabet: Vec<char>,
}

impl StringMonoid {
    /// Create a new string monoid over an alphabet
    pub fn new(alphabet: Vec<char>) -> Self {
        StringMonoid { alphabet }
    }

    /// Get the alphabet
    pub fn alphabet(&self) -> &[char] {
        &self.alphabet
    }

    /// Create the identity element (empty string)
    pub fn identity(&self) -> StringMonoidElement {
        StringMonoidElement::identity()
    }

    /// Create a single-character string
    pub fn gen(&self, c: char) -> Option<StringMonoidElement> {
        if self.alphabet.contains(&c) {
            Some(StringMonoidElement::new(c.to_string()))
        } else {
            None
        }
    }

    /// Create a string element
    pub fn element(&self, s: String) -> StringMonoidElement {
        StringMonoidElement::new(s)
    }

    /// Check if a string is valid (all characters in alphabet)
    pub fn is_valid(&self, elem: &StringMonoidElement) -> bool {
        elem.as_str().chars().all(|c| self.alphabet.contains(&c))
    }
}

impl Display for StringMonoid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "String monoid over alphabet {{{}}}",
            self.alphabet.iter().collect::<String>()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_monoid_element() {
        let elem = StringMonoidElement::new("abc".to_string());
        assert_eq!(elem.as_str(), "abc");
        assert_eq!(elem.len(), 3);
        assert!(!elem.is_empty());
    }

    #[test]
    fn test_identity() {
        let id = StringMonoidElement::identity();
        assert!(id.is_empty());
        assert_eq!(id.len(), 0);
    }

    #[test]
    fn test_mul() {
        let s1 = StringMonoidElement::new("hello".to_string());
        let s2 = StringMonoidElement::new("world".to_string());
        let s3 = s1.mul(&s2);
        assert_eq!(s3.as_str(), "helloworld");
    }

    #[test]
    fn test_reverse() {
        let s = StringMonoidElement::new("abc".to_string());
        let r = s.reverse();
        assert_eq!(r.as_str(), "cba");
    }

    #[test]
    fn test_string_monoid() {
        let M = StringMonoid::new(vec!['a', 'b', 'c']);
        assert_eq!(M.alphabet(), &['a', 'b', 'c']);

        let elem = M.element("abc".to_string());
        assert!(M.is_valid(&elem));

        let invalid = M.element("abcd".to_string());
        assert!(!M.is_valid(&invalid));
    }

    #[test]
    fn test_gen() {
        let M = StringMonoid::new(vec!['x', 'y']);
        let x = M.gen('x').unwrap();
        assert_eq!(x.as_str(), "x");

        assert!(M.gen('z').is_none());
    }
}
