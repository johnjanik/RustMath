//! Algebraic number operations
//!
//! This module provides explicit operations on algebraic numbers.

use crate::algebraic_number::AlgebraicNumber;
use rustmath_core::Result;

/// Add two algebraic numbers
pub fn algebraic_add(a: &AlgebraicNumber, b: &AlgebraicNumber) -> AlgebraicNumber {
    a.clone() + b.clone()
}

/// Multiply two algebraic numbers
pub fn algebraic_mul(a: &AlgebraicNumber, b: &AlgebraicNumber) -> AlgebraicNumber {
    a.clone() * b.clone()
}

/// Negate an algebraic number
pub fn algebraic_neg(a: &AlgebraicNumber) -> AlgebraicNumber {
    -a.clone()
}

/// Compute the multiplicative inverse of an algebraic number
pub fn algebraic_inverse(a: &AlgebraicNumber) -> Result<AlgebraicNumber> {
    a.inverse()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_operations() {
        let a = AlgebraicNumber::from_rational(Rational::new(3, 2).unwrap());
        let b = AlgebraicNumber::from_rational(Rational::new(1, 2).unwrap());

        let sum = algebraic_add(&a, &b);
        assert_eq!(sum.to_rational().unwrap(), Rational::new(2, 1).unwrap());

        let prod = algebraic_mul(&a, &b);
        assert_eq!(prod.to_rational().unwrap(), Rational::new(3, 4).unwrap());

        let neg = algebraic_neg(&a);
        assert_eq!(neg.to_rational().unwrap(), Rational::new(-3, 2).unwrap());

        let inv = algebraic_inverse(&a).unwrap();
        assert_eq!(inv.to_rational().unwrap(), Rational::new(2, 3).unwrap());
    }
}
