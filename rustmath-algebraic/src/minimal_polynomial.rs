//! Minimal polynomial computation
//!
//! Compute the minimal polynomial of an algebraic number over Q.

use crate::algebraic_number::AlgebraicNumber;
use crate::descriptor::AlgebraicDescriptor;
use rustmath_polynomials::UnivariatePolynomial;
use rustmath_integers::Integer;

/// Compute the minimal polynomial of an algebraic number
///
/// The minimal polynomial is the monic polynomial of smallest degree
/// with rational coefficients that has this algebraic number as a root.
///
/// # Arguments
/// * `alpha` - An algebraic number
///
/// # Returns
/// The minimal polynomial of alpha over Q
pub fn minimal_polynomial(alpha: &AlgebraicNumber) -> UnivariatePolynomial<Integer> {
    let simplified = alpha.simplify();

    match simplified.descriptor() {
        AlgebraicDescriptor::Rational(r) => {
            // For rational r = p/q, minimal polynomial is qx - p
            let p = r.value.numerator();
            let q = r.value.denominator();

            UnivariatePolynomial::new(vec![-p.clone(), q.clone()])
        }
        AlgebraicDescriptor::Root(root) => {
            // The polynomial stored in the root might not be minimal
            // TODO: Factor the polynomial and find the irreducible factor
            root.polynomial.clone()
        }
        AlgebraicDescriptor::UnaryExpr(_) | AlgebraicDescriptor::BinaryExpr(_) => {
            // TODO: Implement minimal polynomial computation for expressions
            // This requires resultants and polynomial algebra
            // For now, return x as placeholder
            UnivariatePolynomial::new(vec![Integer::zero(), Integer::one()])
        }
    }
}

/// Compute the degree of an algebraic number
///
/// This is the degree of its minimal polynomial.
pub fn degree(alpha: &AlgebraicNumber) -> usize {
    let min_poly = minimal_polynomial(alpha);
    min_poly.degree().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_minimal_poly_rational() {
        let alpha = AlgebraicNumber::from_rational(Rational::new(3, 2).unwrap());
        let min_poly = minimal_polynomial(&alpha);

        // Should be 2x - 3
        assert_eq!(min_poly.degree(), Some(1));
        assert_eq!(min_poly.coeff(0), &Integer::from(-3));
        assert_eq!(min_poly.coeff(1), &Integer::from(2));
    }

    #[test]
    fn test_degree_rational() {
        let alpha = AlgebraicNumber::from_i64(5);
        assert_eq!(degree(&alpha), 1);
    }
}
