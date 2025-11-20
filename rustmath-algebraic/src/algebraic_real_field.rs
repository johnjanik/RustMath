//! The field of algebraic real numbers (AA)

use crate::algebraic_real::AlgebraicReal;
use std::fmt;

/// The field of algebraic real numbers AA
///
/// This contains all real numbers that are roots of polynomials
/// with rational coefficients.
#[derive(Debug, Clone, Copy)]
pub struct AlgebraicRealField;

impl AlgebraicRealField {
    /// Get the singleton instance of AA
    pub fn new() -> Self {
        AlgebraicRealField
    }

    /// Create an element from a rational number
    pub fn from_rational(&self, r: rustmath_rationals::Rational) -> AlgebraicReal {
        AlgebraicReal::from_rational(r)
    }

    /// Create an element from an integer
    pub fn from_i64(&self, n: i64) -> AlgebraicReal {
        AlgebraicReal::from_i64(n)
    }

    /// Get the zero element
    pub fn zero(&self) -> AlgebraicReal {
        AlgebraicReal::zero()
    }

    /// Get the one element
    pub fn one(&self) -> AlgebraicReal {
        AlgebraicReal::one()
    }

    /// Create sqrt(n)
    pub fn sqrt(&self, n: i64) -> AlgebraicReal {
        AlgebraicReal::sqrt(n)
    }

    /// Create nth root
    pub fn nth_root(&self, n: i64, degree: u32) -> AlgebraicReal {
        AlgebraicReal::nth_root(n, degree)
    }
}

impl Default for AlgebraicRealField {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for AlgebraicRealField {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Algebraic Real Field (AA)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_field_construction() {
        let aa = AlgebraicRealField::new();
        let zero = aa.zero();
        let one = aa.one();

        assert!(zero.is_zero());
        assert!(one.is_one());
    }

    #[test]
    fn test_sqrt() {
        let aa = AlgebraicRealField::new();
        let sqrt2 = aa.sqrt(2);

        let squared = sqrt2.clone() * sqrt2.clone();
        assert_eq!(squared.to_rational().unwrap(), Rational::new(2, 1).unwrap());
    }
}
