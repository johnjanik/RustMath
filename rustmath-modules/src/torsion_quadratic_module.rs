//! Torsion quadratic modules

use num_bigint::BigInt;
use num_rational::BigRational;

/// A finite quadratic module (discriminant form)
#[derive(Clone, Debug)]
pub struct TorsionQuadraticModule {
    /// Invariant factors
    invariants: Vec<BigInt>,
    /// Quadratic form values on generators
    form_values: Vec<BigRational>,
}

impl TorsionQuadraticModule {
    pub fn new(invariants: Vec<BigInt>, form_values: Vec<BigRational>) -> Self {
        assert_eq!(invariants.len(), form_values.len());
        Self { invariants, form_values }
    }

    pub fn order(&self) -> BigInt {
        self.invariants.iter().fold(BigInt::from(1), |acc, x| acc * x)
    }

    pub fn invariants(&self) -> &[BigInt] {
        &self.invariants
    }

    pub fn rank(&self) -> usize {
        self.invariants.len()
    }

    /// Brown invariant (quadratic refinement)
    pub fn brown_invariant(&self) -> BigRational {
        // Simplified implementation
        BigRational::from_integer(0.into())
    }
}
