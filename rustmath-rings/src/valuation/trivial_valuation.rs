//! # Trivial Valuation
//!
//! The trivial valuation: v(x) = 0 for x ≠ 0, v(0) = ∞

use super::valuation::{DiscretePseudoValuation, DiscreteValuation, ValuationValue};
use rustmath_core::Ring;

/// Trivial discrete valuation
#[derive(Debug, Clone, PartialEq)]
pub struct TrivialDiscreteValuation<R: Ring> {
    _phantom: std::marker::PhantomData<R>,
}

impl<R: Ring> TrivialDiscreteValuation<R> {
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}

impl<R: Ring> Default for TrivialDiscreteValuation<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Ring> DiscretePseudoValuation<R> for TrivialDiscreteValuation<R> {
    fn value(&self, element: &R) -> ValuationValue {
        if element.is_zero() {
            ValuationValue::Infinity
        } else {
            ValuationValue::Finite(0)
        }
    }
}

impl<R: Ring> DiscreteValuation<R> for TrivialDiscreteValuation<R> {
    fn uniformizer(&self) -> Option<R> {
        None // No uniformizer for trivial valuation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_trivial_valuation() {
        let val = TrivialDiscreteValuation::<Integer>::new();
        assert_eq!(val.value(&Integer::from(0)), ValuationValue::Infinity);
        assert_eq!(val.value(&Integer::from(5)), ValuationValue::Finite(0));
        assert_eq!(val.value(&Integer::from(100)), ValuationValue::Finite(0));
    }
}
