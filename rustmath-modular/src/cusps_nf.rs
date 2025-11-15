//! Cusps for modular curves over number fields

use num_rational::BigRational;

/// A cusp of a modular curve over a number field
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CuspNF {
    /// Numerator and denominator in the number field
    data: (Vec<BigRational>, Vec<BigRational>),
}

impl CuspNF {
    pub fn new(numerator: Vec<BigRational>, denominator: Vec<BigRational>) -> Self {
        Self {
            data: (numerator, denominator),
        }
    }

    pub fn infinity() -> Self {
        Self {
            data: (vec![BigRational::from_integer(1.into())], vec![BigRational::from_integer(0.into())]),
        }
    }

    pub fn numerator(&self) -> &[BigRational] {
        &self.data.0
    }

    pub fn denominator(&self) -> &[BigRational] {
        &self.data.1
    }
}
