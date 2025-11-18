//! # Developing Valuation
//!
//! Intermediate valuations in the MacLane algorithm

use super::valuation::{DiscretePseudoValuation, ValuationValue};
use rustmath_core::Ring;

/// Developing valuation (in construction)
#[derive(Debug, Clone)]
pub struct DevelopingValuation<R: Ring> {
    _phantom: std::marker::PhantomData<R>,
}

impl<R: Ring> DevelopingValuation<R> {
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}

impl<R: Ring> Default for DevelopingValuation<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Ring> DiscretePseudoValuation<R> for DevelopingValuation<R> {
    fn value(&self, element: &R) -> ValuationValue {
        if element.is_zero() {
            ValuationValue::Infinity
        } else {
            ValuationValue::Finite(0)
        }
    }
}
