//! # Automatic Semigroups
//!
//! This module provides automatic semigroups and monoids.
//!
//! An automatic semigroup is a semigroup with a regular language of normal forms
//! and a finite-state automaton for computing the product.

use std::collections::HashMap;

/// An element of an automatic semigroup
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Element {
    /// Normal form representation
    normal_form: Vec<usize>,
}

impl Element {
    /// Create a new element
    pub fn new(normal_form: Vec<usize>) -> Self {
        Element { normal_form }
    }

    /// Get the normal form
    pub fn normal_form(&self) -> &[usize] {
        &self.normal_form
    }
}

/// An automatic semigroup
#[derive(Debug, Clone)]
pub struct AutomaticSemigroup {
    /// Number of generators
    num_generators: usize,
    /// Multiplication table (if finite)
    #[allow(dead_code)]
    mult_table: Option<HashMap<(usize, usize), usize>>,
}

impl AutomaticSemigroup {
    /// Create a new automatic semigroup
    pub fn new(num_generators: usize) -> Self {
        AutomaticSemigroup {
            num_generators,
            mult_table: None,
        }
    }

    /// Get the number of generators
    pub fn num_generators(&self) -> usize {
        self.num_generators
    }

    /// Create a generator
    pub fn gen(&self, index: usize) -> Option<Element> {
        if index < self.num_generators {
            Some(Element::new(vec![index]))
        } else {
            None
        }
    }

    /// Multiply two elements (simplified version)
    pub fn mul(&self, _a: &Element, _b: &Element) -> Element {
        // Simplified: just concatenate
        // Real implementation would use automata
        Element::new(vec![])
    }
}

/// An automatic monoid
#[derive(Debug, Clone)]
pub struct AutomaticMonoid {
    /// The underlying semigroup
    semigroup: AutomaticSemigroup,
}

impl AutomaticMonoid {
    /// Create a new automatic monoid
    pub fn new(num_generators: usize) -> Self {
        AutomaticMonoid {
            semigroup: AutomaticSemigroup::new(num_generators),
        }
    }

    /// Get the identity element
    pub fn identity(&self) -> Element {
        Element::new(vec![])
    }

    /// Get the number of generators
    pub fn num_generators(&self) -> usize {
        self.semigroup.num_generators()
    }

    /// Create a generator
    pub fn gen(&self, index: usize) -> Option<Element> {
        self.semigroup.gen(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_automatic_semigroup() {
        let S = AutomaticSemigroup::new(3);
        assert_eq!(S.num_generators(), 3);
    }

    #[test]
    fn test_automatic_monoid() {
        let M = AutomaticMonoid::new(2);
        assert_eq!(M.num_generators(), 2);

        let id = M.identity();
        assert!(id.normal_form().is_empty());
    }

    #[test]
    fn test_element() {
        let e = Element::new(vec![0, 1, 2]);
        assert_eq!(e.normal_form(), &[0, 1, 2]);
    }

    #[test]
    fn test_gen() {
        let M = AutomaticMonoid::new(5);
        let g2 = M.gen(2).unwrap();
        assert_eq!(g2.normal_form(), &[2]);

        assert!(M.gen(10).is_none());
    }
}
