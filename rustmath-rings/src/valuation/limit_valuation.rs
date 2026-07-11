//! # Limit Valuation
//!
//! Limit of a sequence of valuations
//!
//! > **Superseded**: this type is an empty facade (`value` is
//! > `unimplemented!`). The concrete representation of a MacLane limit that
//! > is reached in finitely many steps is the infinite augmentation
//! > `[v, v(phi) = +Infinity]` of
//! > [`crate::valuation::maclane::InductiveValuation`] (see
//! > `augment(phi, QVal::Infinity)`, produced by `mac_lane_approximants`
//! > when a key divides the target exactly). True limits requiring
//! > infinitely many augmentations remain unimplemented.

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
        let _ = element;
        unimplemented!(
            "rustmath_rings::valuation::limit_valuation::LimitValuation::value: MacLane limit valuation not yet implemented (facade)"
        )
    }
}

/// MacLane limit valuation
pub type MacLaneLimitValuation<R> = LimitValuation<R>;
