//! Trace monoid structures

use std::collections::HashMap;

/// A trace monoid (partially commutative monoid)
#[derive(Clone, Debug)]
pub struct TraceMonoid {
    generators: Vec<String>,
    independence: HashMap<(usize, usize), bool>,
}

impl TraceMonoid {
    /// Create a new trace monoid
    pub fn new(generators: Vec<String>) -> Self {
        let independence = HashMap::new();
        Self {
            generators,
            independence,
        }
    }

    /// Add an independence relation between two generators
    pub fn add_independence(&mut self, i: usize, j: usize) {
        self.independence.insert((i, j), true);
        self.independence.insert((j, i), true);
    }

    /// Check if two generators are independent
    pub fn are_independent(&self, i: usize, j: usize) -> bool {
        self.independence.get(&(i, j)).copied().unwrap_or(false)
    }

    /// Get the generators
    pub fn generators(&self) -> &[String] {
        &self.generators
    }
}

/// An element of a trace monoid
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceMonoidElement {
    word: Vec<usize>,
}

impl TraceMonoidElement {
    /// Create a new trace monoid element
    pub fn new(word: Vec<usize>) -> Self {
        Self { word }
    }

    /// Get the word representation
    pub fn word(&self) -> &[usize] {
        &self.word
    }
}
