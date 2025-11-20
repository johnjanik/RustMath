//! The field of algebraic numbers (QQbar)

use crate::algebraic_number::AlgebraicNumber;
use std::fmt;

/// The field of algebraic numbers QQbar
///
/// This is the algebraic closure of the rational numbers Q.
/// It contains all complex numbers that are roots of polynomials
/// with rational coefficients.
#[derive(Debug, Clone, Copy)]
pub struct AlgebraicField;

impl AlgebraicField {
    /// Get the singleton instance of QQbar
    pub fn new() -> Self {
        AlgebraicField
    }

    /// Create an element from a rational number
    pub fn from_rational(&self, r: rustmath_rationals::Rational) -> AlgebraicNumber {
        AlgebraicNumber::from_rational(r)
    }

    /// Create an element from an integer
    pub fn from_i64(&self, n: i64) -> AlgebraicNumber {
        AlgebraicNumber::from_i64(n)
    }

    /// Get the zero element
    pub fn zero(&self) -> AlgebraicNumber {
        AlgebraicNumber::zero()
    }

    /// Get the one element
    pub fn one(&self) -> AlgebraicNumber {
        AlgebraicNumber::one()
    }

    /// Get the imaginary unit i
    pub fn i(&self) -> AlgebraicNumber {
        // TODO: Implement construction of i = sqrt(-1)
        // For now, return zero as placeholder
        AlgebraicNumber::zero()
    }
}

impl Default for AlgebraicField {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for AlgebraicField {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Algebraic Field (QQbar)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_field_construction() {
        let qqbar = AlgebraicField::new();
        let zero = qqbar.zero();
        let one = qqbar.one();

        assert!(zero.is_zero());
        assert!(one.is_one());
    }

    #[test]
    fn test_from_rational() {
        let qqbar = AlgebraicField::new();
        let half = qqbar.from_rational(Rational::new(1, 2).unwrap());

        assert!(half.is_rational());
        assert_eq!(half.to_rational().unwrap(), Rational::new(1, 2).unwrap());
    }
}
