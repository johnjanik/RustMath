//! Cusps for modular curves over number fields

use rustmath_rationals::Rational;

/// A cusp of a modular curve over a number field
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CuspNF {
    /// Numerator and denominator in the number field
    data: (Vec<Rational>, Vec<Rational>),
}

impl CuspNF {
    pub fn new(numerator: Vec<Rational>, denominator: Vec<Rational>) -> Self {
        Self {
            data: (numerator, denominator),
        }
    }

    pub fn infinity() -> Self {
        Self {
            data: (vec![Rational::from_i64(1)], vec![Rational::from_i64(0)]),
        }
    }

    pub fn numerator(&self) -> &[Rational] {
        &self.data.0
    }

    pub fn denominator(&self) -> &[Rational] {
        &self.data.1
    }
}
