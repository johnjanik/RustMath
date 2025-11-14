//! Hecke monoid structures

/// A Hecke monoid element
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeckeMonoidElement {
    data: Vec<usize>,
}

impl HeckeMonoidElement {
    /// Create a new Hecke monoid element
    pub fn new(data: Vec<usize>) -> Self {
        Self { data }
    }

    /// Get the data
    pub fn data(&self) -> &[usize] {
        &self.data
    }
}

/// A Hecke monoid
#[derive(Clone, Debug)]
pub struct HeckeMonoid {
    rank: usize,
}

impl HeckeMonoid {
    /// Create a new Hecke monoid of given rank
    pub fn new(rank: usize) -> Self {
        Self { rank }
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        self.rank
    }
}
