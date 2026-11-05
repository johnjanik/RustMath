//! # Gauss Valuation
//!
//! Valuation on polynomial rings extending a base valuation
//!
//! > **Superseded**: this type is a data-only stub (it computes nothing).
//! > The REAL Gauss valuation `v_0(sum a_i x^i) = min_i v(a_i)` (and its
//! > shifted variant `v_0(x) = lambda`) is
//! > [`crate::valuation::maclane::InductiveValuation::gauss`] /
//! > [`gauss_shifted`](crate::valuation::maclane::InductiveValuation::gauss_shifted),
//! > with exact rational values over a p-adic or function-field-place base.

use super::valuation::DiscretePseudoValuation;
use rustmath_core::Ring;

/// Gauss valuation on polynomial ring
///
/// Note: Clone is not derived because this struct contains a trait object
/// (Box<dyn DiscretePseudoValuation<R>>), which cannot be automatically cloned.
#[deprecated(
    note = "data-only stub; use valuation::maclane::InductiveValuation::gauss for the real Gauss valuation"
)]
#[derive(Debug)]
pub struct GaussValuation<R: Ring> {
    #[allow(dead_code)]
    base_valuation: Box<dyn DiscretePseudoValuation<R>>,
}

#[allow(deprecated)]
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
    #[allow(deprecated)]
    fn test_gauss_valuation() {
        let base = Box::new(TrivialDiscreteValuation::<Integer>::new());
        let _gauss = GaussValuation::new(base);
        // The real Gauss valuation is exercised in valuation::maclane.
    }
}
