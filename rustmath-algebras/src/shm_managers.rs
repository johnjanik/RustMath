//! Shared memory managers for fusion ring computations
//!
//! Provides shared memory access for parallel F-matrix and data computations

use std::collections::HashMap;
use rustmath_rationals::Rational;

/// Handler for F-variables in shared memory
pub struct FvarsHandler {
    /// F-variable storage
    fvars: HashMap<(usize, usize, usize, usize), Rational>,
}

impl FvarsHandler {
    /// Create a new F-variables handler
    pub fn new() -> Self {
        FvarsHandler {
            fvars: HashMap::new(),
        }
    }

    /// Get an F-variable
    pub fn get(&self, a: usize, b: usize, c: usize, d: usize) -> Option<&Rational> {
        self.fvars.get(&(a, b, c, d))
    }

    /// Set an F-variable
    pub fn set(&mut self, a: usize, b: usize, c: usize, d: usize, value: Rational) {
        self.fvars.insert((a, b, c, d), value);
    }

    /// Get number of stored F-variables
    pub fn len(&self) -> usize {
        self.fvars.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.fvars.is_empty()
    }
}

impl Default for FvarsHandler {
    fn default() -> Self {
        Self::new()
    }
}

/// Handler for kernel space data in shared memory
pub struct KSHandler {
    /// Kernel space vectors
    vectors: Vec<Vec<Rational>>,
}

impl KSHandler {
    /// Create a new kernel space handler
    pub fn new() -> Self {
        KSHandler {
            vectors: Vec::new(),
        }
    }

    /// Add a kernel vector
    pub fn add_vector(&mut self, vector: Vec<Rational>) {
        self.vectors.push(vector);
    }

    /// Get a kernel vector by index
    pub fn get_vector(&self, index: usize) -> Option<&Vec<Rational>> {
        self.vectors.get(index)
    }

    /// Get number of kernel vectors
    pub fn num_vectors(&self) -> usize {
        self.vectors.len()
    }
}

impl Default for KSHandler {
    fn default() -> Self {
        Self::new()
    }
}

/// Factory function for creating FvarsHandler
pub fn make_fvars_handler() -> FvarsHandler {
    FvarsHandler::new()
}

/// Factory function for creating KSHandler
pub fn make_ks_handler() -> KSHandler {
    KSHandler::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_core::Ring;

    #[test]
    fn test_fvars_handler() {
        let mut handler = FvarsHandler::new();
        handler.set(0, 1, 2, 3, Rational::one());

        assert_eq!(handler.len(), 1);
        assert_eq!(handler.get(0, 1, 2, 3), Some(&Rational::one()));
    }

    #[test]
    fn test_ks_handler() {
        let mut handler = KSHandler::new();
        handler.add_vector(vec![Rational::one(), Rational::zero()]);

        assert_eq!(handler.num_vectors(), 1);
        assert!(handler.get_vector(0).is_some());
    }

    #[test]
    fn test_factory_functions() {
        let fvars = make_fvars_handler();
        let ks = make_ks_handler();

        assert!(fvars.is_empty());
        assert_eq!(ks.num_vectors(), 0);
    }
}
