//! Galois conjugates of algebraic numbers
//!
//! Compute the Galois conjugates (other roots of the minimal polynomial).

use crate::algebraic_number::AlgebraicNumber;
use crate::minimal_polynomial::minimal_polynomial;

/// Compute all Galois conjugates of an algebraic number
///
/// The Galois conjugates are the roots of the minimal polynomial of alpha.
///
/// # Arguments
/// * `alpha` - An algebraic number
///
/// # Returns
/// A vector of all Galois conjugates (including alpha itself)
pub fn galois_conjugates(alpha: &AlgebraicNumber) -> Vec<AlgebraicNumber> {
    let min_poly = minimal_polynomial(alpha);

    // TODO: Implement root finding for the minimal polynomial
    // This requires:
    // 1. Factoring the polynomial (if reducible)
    // 2. Finding roots using numerical methods
    // 3. Converting roots back to AlgebraicNumber

    // For now, just return alpha itself
    vec![alpha.clone()]
}

/// Count the number of Galois conjugates
pub fn num_conjugates(alpha: &AlgebraicNumber) -> usize {
    galois_conjugates(alpha).len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_conjugates_rational() {
        let alpha = AlgebraicNumber::from_rational(Rational::new(3, 2).unwrap());
        let conjugates = galois_conjugates(&alpha);

        // A rational number has only itself as conjugate
        assert_eq!(conjugates.len(), 1);
        assert_eq!(conjugates[0], alpha);
    }

    #[test]
    fn test_num_conjugates() {
        let alpha = AlgebraicNumber::from_i64(7);
        assert_eq!(num_conjugates(&alpha), 1);
    }
}
