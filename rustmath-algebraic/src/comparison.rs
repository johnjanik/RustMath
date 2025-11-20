//! Exact comparison of algebraic numbers
//!
//! Implement exact equality and ordering for algebraic numbers.

use crate::algebraic_number::AlgebraicNumber;
use crate::algebraic_real::AlgebraicReal;
use std::cmp::Ordering;

/// Compare two algebraic numbers for exact equality
///
/// # Arguments
/// * `a` - First algebraic number
/// * `b` - Second algebraic number
///
/// # Returns
/// true if a and b represent the same algebraic number
pub fn algebraic_eq(a: &AlgebraicNumber, b: &AlgebraicNumber) -> bool {
    a == b
}

/// Compare two algebraic real numbers
///
/// # Arguments
/// * `a` - First algebraic real
/// * `b` - Second algebraic real
///
/// # Returns
/// Ordering::Less if a < b, Ordering::Equal if a = b, Ordering::Greater if a > b
pub fn algebraic_compare(a: &AlgebraicReal, b: &AlgebraicReal) -> Ordering {
    a.cmp(b)
}

/// Check if an algebraic real is positive
pub fn is_positive(a: &AlgebraicReal) -> bool {
    a.sign() > 0
}

/// Check if an algebraic real is negative
pub fn is_negative(a: &AlgebraicReal) -> bool {
    a.sign() < 0
}

/// Check if an algebraic real is zero
pub fn is_zero(a: &AlgebraicReal) -> bool {
    a.is_zero()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_equality() {
        let a = AlgebraicNumber::from_rational(Rational::new(3, 2).unwrap());
        let b = AlgebraicNumber::from_rational(Rational::new(3, 2).unwrap());
        let c = AlgebraicNumber::from_rational(Rational::new(5, 2).unwrap());

        assert!(algebraic_eq(&a, &b));
        assert!(!algebraic_eq(&a, &c));
    }

    #[test]
    fn test_comparison() {
        let a = AlgebraicReal::from_i64(3);
        let b = AlgebraicReal::from_i64(5);

        assert_eq!(algebraic_compare(&a, &b), Ordering::Less);
        assert_eq!(algebraic_compare(&b, &a), Ordering::Greater);
        assert_eq!(algebraic_compare(&a, &a), Ordering::Equal);
    }

    #[test]
    fn test_sign_tests() {
        let positive = AlgebraicReal::from_i64(5);
        let negative = AlgebraicReal::from_i64(-3);
        let zero = AlgebraicReal::from_i64(0);

        assert!(is_positive(&positive));
        assert!(!is_positive(&negative));
        assert!(!is_positive(&zero));

        assert!(is_negative(&negative));
        assert!(!is_negative(&positive));
        assert!(!is_negative(&zero));

        assert!(is_zero(&zero));
        assert!(!is_zero(&positive));
        assert!(!is_zero(&negative));
    }
}
