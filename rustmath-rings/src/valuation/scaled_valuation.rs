//! # Scaled Valuation
//!
//! A valuation scaled by a positive rational number

use super::valuation::{DiscretePseudoValuation, DiscreteValuation, ValuationValue};
use rustmath_core::Ring;

/// Scaled valuation: s·v where s > 0
///
/// Note: Clone is not derived because this struct contains a trait object
/// (Box<dyn DiscretePseudoValuation<R>>), which cannot be automatically cloned.
#[derive(Debug)]
pub struct ScaledValuation<R: Ring> {
    base_valuation: Box<dyn DiscretePseudoValuation<R>>,
    scale: i64,
}

impl<R: Ring> ScaledValuation<R> {
    pub fn new(base_valuation: Box<dyn DiscretePseudoValuation<R>>, scale: i64) -> Self {
        if scale <= 0 {
            panic!("Scale must be positive");
        }
        Self { base_valuation, scale }
    }

    pub fn scale(&self) -> i64 {
        self.scale
    }
}

impl<R: Ring> DiscretePseudoValuation<R> for ScaledValuation<R> {
    fn value(&self, element: &R) -> ValuationValue {
        match self.base_valuation.value(element) {
            ValuationValue::Finite(v) => ValuationValue::Finite(v * self.scale),
            ValuationValue::Infinity => ValuationValue::Infinity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::trivial_valuation::TrivialDiscreteValuation;
    use rustmath_integers::Integer;

    #[test]
    fn test_scaled_valuation() {
        let base = Box::new(TrivialDiscreteValuation::<Integer>::new());
        let scaled = ScaledValuation::new(base, 3);

        assert_eq!(scaled.value(&Integer::from(0)), ValuationValue::Infinity);
        assert_eq!(scaled.value(&Integer::from(5)), ValuationValue::Finite(0)); // 0 * 3 = 0
    }
}
