//! Base module trait and structures

use rustmath_core::{Ring, Field};
use num_traits::{Zero, One};
use std::fmt;

/// Trait representing a module over a ring
pub trait Module: Clone + fmt::Debug {
    /// The base ring type
    type BaseRing: Ring;

    /// The element type
    type Element: Clone + fmt::Debug;

    /// Get the base ring
    fn base_ring(&self) -> &Self::BaseRing;

    /// Get the rank (dimension) of the module
    fn rank(&self) -> usize;

    /// Create the zero element
    fn zero(&self) -> Self::Element;

    /// Check if an element is zero
    fn is_zero(&self, elem: &Self::Element) -> bool;

    /// Add two elements
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;

    /// Negate an element
    fn negate(&self, a: &Self::Element) -> Self::Element;

    /// Scalar multiplication
    fn scalar_mul(&self, scalar: &Self::BaseRing, elem: &Self::Element) -> Self::Element;
}

/// Abstract module structure
#[derive(Clone, Debug)]
pub struct AbstractModule<R: Ring> {
    base_ring: R,
    rank: usize,
}

impl<R: Ring> AbstractModule<R> {
    /// Create a new abstract module
    pub fn new(base_ring: R, rank: usize) -> Self {
        Self { base_ring, rank }
    }

    /// Get the base ring
    pub fn base_ring(&self) -> &R {
        &self.base_ring
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        self.rank
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    #[test]
    fn test_abstract_module() {
        let module = AbstractModule::new(BigInt::from(5), 3);
        assert_eq!(module.rank(), 3);
    }
}
