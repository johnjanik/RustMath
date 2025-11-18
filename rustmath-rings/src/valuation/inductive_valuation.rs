//! # Inductive Valuation
//!
//! Valuations defined inductively via MacLane's algorithm

use super::valuation::{DiscretePseudoValuation, DiscreteValuation, ValuationValue};
use super::developing_valuation::DevelopingValuation;
use rustmath_core::Ring;

/// Inductive valuation
pub type InductiveValuation<R> = DevelopingValuation<R>;

/// Finite inductive valuation
#[derive(Debug, Clone)]
pub struct FiniteInductiveValuation<R: Ring> {
    base: InductiveValuation<R>,
}

impl<R: Ring> FiniteInductiveValuation<R> {
    pub fn new() -> Self {
        Self { base: InductiveValuation::new() }
    }
}

impl<R: Ring> Default for FiniteInductiveValuation<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Ring> DiscretePseudoValuation<R> for FiniteInductiveValuation<R> {
    fn value(&self, element: &R) -> ValuationValue {
        self.base.value(element)
    }
}

impl<R: Ring> DiscreteValuation<R> for FiniteInductiveValuation<R> {
    fn uniformizer(&self) -> Option<R> {
        None
    }
}

/// Non-final inductive valuation
pub type NonFinalInductiveValuation<R> = FiniteInductiveValuation<R>;

/// Final inductive valuation
pub type FinalInductiveValuation<R> = InductiveValuation<R>;

/// Infinite inductive valuation
pub type InfiniteInductiveValuation<R> = FinalInductiveValuation<R>;
