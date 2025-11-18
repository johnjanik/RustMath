//! # Gauss Valuation
//!
//! Valuation on polynomial rings extending a base valuation

use super::valuation::{DiscretePseudoValuation, ValuationValue};
use rustmath_core::Ring;

/// Gauss valuation on polynomial ring
#[derive(Debug, Clone)]
pub struct GaussValuation<R: Ring> {
    base_valuation: Box<dyn DiscretePseudoValuation<R>>,
}

impl<R: Ring> GaussValuation<R> {
    pub fn new(base_valuation: Box<dyn DiscretePseudoValuation<R>>) -> Self {
        Self { base_valuation }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::trivial_valuation::TrivialDiscreteValuation;
    use rustmath_integers::Integer;

    #[test]
    fn test_gauss_valuation() {
        let base = Box::new(TrivialDiscreteValuation::<Integer>::new());
        let _gauss = GaussValuation::new(base);
    }
}
