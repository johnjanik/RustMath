//! # Valuation Space
//!
//! The space of all valuations on a ring

use rustmath_core::Ring;

/// Discrete pseudo-valuation space
#[derive(Debug, Clone)]
pub struct DiscretePseudoValuationSpace<R: Ring> {
    _phantom: std::marker::PhantomData<R>,
}

impl<R: Ring> DiscretePseudoValuationSpace<R> {
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}

impl<R: Ring> Default for DiscretePseudoValuationSpace<R> {
    fn default() -> Self {
        Self::new()
    }
}

/// Element methods
#[derive(Debug, Clone)]
pub struct ElementMethods;

/// Scale action
#[derive(Debug, Clone)]
pub struct ScaleAction;

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_valuation_space() {
        let _space = DiscretePseudoValuationSpace::<Integer>::new();
    }
}
