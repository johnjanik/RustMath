//! # Limit Valuation
//!
//! Limit of a sequence of valuations

use super::valuation::{DiscretePseudoValuation, ValuationValue};
use rustmath_core::Ring;

/// Limit valuation
#[derive(Debug, Clone)]
pub struct LimitValuation<R: Ring> {
    _phantom: std::marker::PhantomData<R>,
}

impl<R: Ring> LimitValuation<R> {
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}

impl<R: Ring> Default for LimitValuation<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Ring> DiscretePseudoValuation<R> for LimitValuation<R> {
    fn value(&self, element: &R) -> ValuationValue {
        if element.is_zero() {
            ValuationValue::Infinity
        } else {
            ValuationValue::Finite(0)
        }
    }
}

/// MacLane limit valuation
pub type MacLaneLimitValuation<R> = LimitValuation<R>;
