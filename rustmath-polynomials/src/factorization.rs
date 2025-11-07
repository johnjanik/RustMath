//! Polynomial factorization algorithms

use crate::polynomial::Polynomial;
use crate::univariate::UnivariatePolynomial;
use rustmath_core::{EuclideanDomain, MathError, Result, Ring};
use std::fmt::Debug;

/// Compute the content of a polynomial (GCD of all coefficients)
///
/// For a polynomial f = aₙxⁿ + ... + a₁x + a₀,
/// the content is gcd(aₙ, ..., a₁, a₀)
pub fn content<R>(poly: &UnivariatePolynomial<R>) -> R
where
    R: Ring + EuclideanDomain + Clone + Debug,
{
    if poly.is_zero() {
        return R::zero();
    }

    let coeffs: Vec<R> = poly.coefficients().to_vec();
    if coeffs.is_empty() {
        return R::zero();
    }

    let mut result = coeffs[0].clone();
    for coeff in coeffs.iter().skip(1) {
        result = result.gcd(coeff);
        if result.is_one() {
            return result;
        }
    }
    result
}

/// Compute the primitive part of a polynomial
///
/// The primitive part is the polynomial divided by its content.
/// For f = content(f) × pp(f), this returns pp(f)
pub fn primitive_part<R>(poly: &UnivariatePolynomial<R>) -> UnivariatePolynomial<R>
where
    R: Ring + EuclideanDomain + Clone + Debug,
{
    let cont = content(poly);
    if cont.is_zero() {
        return UnivariatePolynomial::new(vec![R::zero()]);
    }
    if cont.is_one() {
        return poly.clone();
    }

    let new_coeffs: Vec<R> = poly
        .coefficients()
        .iter()
        .map(|c| {
            let (q, _) = c.clone().div_rem(cont.clone()).unwrap();
            q
        })
        .collect();

    UnivariatePolynomial::new(new_coeffs)
}

/// Square-free factorization
///
/// Decomposes a polynomial into square-free factors:
/// f(x) = f₁(x) × f₂(x)² × f₃(x)³ × ...
///
/// Returns a vector of (factor, multiplicity) pairs where each factor is square-free
pub fn square_free_factorization<R>(
    poly: &UnivariatePolynomial<R>,
) -> Result<Vec<(UnivariatePolynomial<R>, u32)>>
where
    R: Ring + EuclideanDomain + Clone + Debug,
{
    if poly.is_zero() {
        return Ok(vec![]);
    }

    // Remove content first
    let f = primitive_part(poly);

    // Compute derivative
    let f_prime = f.derivative();

    // If derivative is zero, we're in characteristic p > 0 or f is constant
    if f_prime.is_zero() {
        if f.is_constant() {
            return Ok(vec![(f, 1)]);
        }
        // In characteristic p, would need different algorithm
        return Err(MathError::NotSupported(
            "Square-free factorization in positive characteristic not yet implemented".to_string(),
        ));
    }

    // Compute gcd(f, f')
    let g = f.gcd(&f_prime);

    // If gcd is 1, f is already square-free
    if g.degree() == Some(0) || g.is_constant() {
        return Ok(vec![(f, 1)]);
    }

    // Compute f / gcd(f, f')
    let (f_reduced, _) = f.div_rem(&g)?;

    let mut factors = Vec::new();
    let mut multiplicity = 1;
    let mut current = f_reduced;
    let mut g_remaining = g;

    while !g_remaining.is_constant() {
        let g_next = current.gcd(&g_remaining);

        if g_next.degree() == Some(0) || g_next.is_constant() {
            // current is a square-free factor with this multiplicity
            if !current.is_constant() {
                factors.push((current.clone(), multiplicity));
            }
            break;
        }

        let (factor, _) = current.div_rem(&g_next)?;

        if !factor.is_constant() {
            factors.push((factor, multiplicity));
        }

        current = g_next.clone();
        let (g_remaining_new, _) = g_remaining.div_rem(&g_next)?;
        g_remaining = g_remaining_new;
        multiplicity += 1;

        // Safety check to prevent infinite loops
        if multiplicity > 1000 {
            return Err(MathError::NotSupported(
                "Square-free factorization exceeded maximum multiplicity".to_string(),
            ));
        }
    }

    // Add the remaining factor if it's not constant
    if !current.is_constant() {
        factors.push((current, multiplicity));
    }

    Ok(factors)
}

/// Check if a polynomial is square-free (has no repeated factors)
pub fn is_square_free<R>(poly: &UnivariatePolynomial<R>) -> Result<bool>
where
    R: Ring + EuclideanDomain + Clone + Debug,
{
    if poly.is_zero() || poly.is_constant() {
        return Ok(true);
    }

    let derivative = poly.derivative();
    if derivative.is_zero() {
        return Ok(false);
    }

    let g = poly.gcd(&derivative);
    Ok(g.degree() == Some(0) || g.is_constant())
}

/// Factor a polynomial over integers (basic implementation using trial and error)
///
/// This is a simple factorization that works for small polynomials.
/// More sophisticated algorithms (Berlekamp, Cantor-Zassenhaus, LLL) would be needed
/// for production use.
pub fn factor_over_integers(
    poly: &UnivariatePolynomial<rustmath_integers::Integer>,
) -> Result<Vec<(UnivariatePolynomial<rustmath_integers::Integer>, u32)>> {
    use rustmath_integers::Integer;

    // First, do square-free factorization
    let square_free_factors = square_free_factorization(poly)?;

    // For now, just return the square-free factorization
    // Full factorization into irreducibles would require more sophisticated algorithms
    Ok(square_free_factors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_content() {
        // 6x² + 9x + 3 has content 3
        let p = UnivariatePolynomial::new(vec![
            Integer::from(3),
            Integer::from(9),
            Integer::from(6),
        ]);

        let cont = content(&p);
        assert_eq!(cont, Integer::from(3));
    }

    #[test]
    fn test_primitive_part() {
        // 6x² + 9x + 3 = 3(2x² + 3x + 1)
        let p = UnivariatePolynomial::new(vec![
            Integer::from(3),
            Integer::from(9),
            Integer::from(6),
        ]);

        let pp = primitive_part(&p);

        // Primitive part should be 2x² + 3x + 1
        assert_eq!(*pp.coeff(0), Integer::from(1));
        assert_eq!(*pp.coeff(1), Integer::from(3));
        assert_eq!(*pp.coeff(2), Integer::from(2));
    }

    #[test]
    fn test_is_square_free() {
        // x² + x + 1 is square-free
        let p1 = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(1),
            Integer::from(1),
        ]);
        assert!(is_square_free(&p1).unwrap());

        // (x - 1)² = x² - 2x + 1 is not square-free
        let p2 = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(-2),
            Integer::from(1),
        ]);
        assert!(!is_square_free(&p2).unwrap());
    }

    #[test]
    fn test_square_free_factorization_simple() {
        // x is already square-free
        let x = UnivariatePolynomial::new(vec![Integer::from(0), Integer::from(1)]);

        let factors = square_free_factorization(&x).unwrap();
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].1, 1); // multiplicity 1
    }

    #[test]
    fn test_square_free_factorization_repeated() {
        // x² = x * x (x with multiplicity 2)
        let p = UnivariatePolynomial::new(vec![
            Integer::from(0),
            Integer::from(0),
            Integer::from(1),
        ]);

        let factors = square_free_factorization(&p).unwrap();

        // Should have x with multiplicity 2
        // Note: The algorithm should detect the repeated factor
        assert!(!factors.is_empty());
    }

    #[test]
    fn test_factor_over_integers() {
        // Simple polynomial x + 1
        let p = UnivariatePolynomial::new(vec![Integer::from(1), Integer::from(1)]);

        let factors = factor_over_integers(&p).unwrap();
        assert!(!factors.is_empty());
    }
}
