//! # Mapped Valuation
//!
//! Valuation obtained by composition with a ring homomorphism

use super::valuation::{DiscretePseudoValuation, ValuationValue};
use rustmath_core::Ring;

/// Mapped valuation base
#[derive(Debug, Clone)]
pub struct MappedValuationBase<R: Ring> {
    _phantom: std::marker::PhantomData<R>,
}

impl<R: Ring> MappedValuationBase<R> {
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}

impl<R: Ring> Default for MappedValuationBase<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Ring> DiscretePseudoValuation<R> for MappedValuationBase<R> {
    fn value(&self, element: &R) -> ValuationValue {
        let _ = element;
        unimplemented!(
            "rustmath_rings::valuation::mapped_valuation::MappedValuationBase::value: mapped valuation (composition with a ring homomorphism) not yet implemented (facade)"
        )
    }
}

/// Finite extension from infinite valuation
pub type FiniteExtensionFromInfiniteValuation<R> = MappedValuationBase<R>;

/// Finite extension from limit valuation
pub type FiniteExtensionFromLimitValuation<R> = MappedValuationBase<R>;
