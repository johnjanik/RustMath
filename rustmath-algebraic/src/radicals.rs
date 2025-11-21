//! Radical expressions and simplification
//!
//! This module handles nth roots and radical simplification.

use crate::algebraic_real::AlgebraicReal;
use crate::algebraic_number::AlgebraicNumber;
use rustmath_core::Ring;

/// Compute the nth root of an algebraic real number
///
/// # Arguments
/// * `alpha` - An algebraic real number
/// * `n` - The degree of the root
///
/// # Returns
/// The nth root of alpha
pub fn nth_root(alpha: &AlgebraicReal, n: u32) -> AlgebraicReal {
    if n == 0 {
        panic!("Cannot take 0th root");
    }

    if n == 1 {
        return alpha.clone();
    }

    // If alpha is rational, try to simplify
    if let Some(r) = alpha.to_rational() {
        if r.is_zero() {
            return AlgebraicReal::zero();
        }

        // For now, handle integer case
        // TODO: Handle general rational nth roots
        if r.denominator().is_one() {
            let num = r.numerator().to_i64();
            return AlgebraicReal::nth_root(num, n);
        }
    }

    // TODO: Implement general nth root for algebraic numbers
    alpha.clone()
}

/// Compute the square root of an algebraic real number
pub fn sqrt(alpha: &AlgebraicReal) -> AlgebraicReal {
    nth_root(alpha, 2)
}

/// Simplify a radical expression
///
/// This attempts to simplify expressions involving radicals.
/// For example, sqrt(8) = 2*sqrt(2)
pub fn radical_simplify(alpha: &AlgebraicReal) -> AlgebraicReal {
    // TODO: Implement radical simplification
    // This requires:
    // 1. Factoring out perfect powers
    // 2. Combining like radicals
    // 3. Rationalizing denominators
    alpha.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_sqrt_from_rational() {
        let four = AlgebraicReal::from_rational(Rational::new(4, 1).unwrap());
        let sqrt_four = sqrt(&four);

        assert!(sqrt_four.is_rational());
        assert_eq!(sqrt_four.to_rational().unwrap(), Rational::new(2, 1).unwrap());
    }

    #[test]
    fn test_nth_root_identity() {
        let five = AlgebraicReal::from_i64(5);
        let root = nth_root(&five, 1);

        assert_eq!(root.to_rational().unwrap(), Rational::new(5, 1).unwrap());
    }
}
