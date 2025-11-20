//! Complex embeddings of algebraic numbers
//!
//! Compute complex embeddings (approximate complex values) of algebraic numbers.

use crate::algebraic_number::AlgebraicNumber;
use rustmath_complex::Complex;

/// Compute a complex embedding of an algebraic number
///
/// Returns an approximation of the algebraic number as a complex number
/// with the specified precision.
///
/// # Arguments
/// * `alpha` - An algebraic number
/// * `precision` - The number of bits of precision
///
/// # Returns
/// A complex approximation of alpha
pub fn complex_embedding(alpha: &AlgebraicNumber, precision: usize) -> Complex {
    alpha.to_complex(precision)
}

/// Compute all complex embeddings of an algebraic number
///
/// An algebraic number of degree n has n complex embeddings,
/// corresponding to the n roots of its minimal polynomial.
///
/// # Arguments
/// * `alpha` - An algebraic number
/// * `precision` - The number of bits of precision
///
/// # Returns
/// A vector of all complex embeddings (Galois conjugates)
pub fn all_complex_embeddings(alpha: &AlgebraicNumber, precision: usize) -> Vec<Complex> {
    // TODO: Implement computation of all embeddings
    // This requires:
    // 1. Computing the minimal polynomial
    // 2. Finding all roots of the minimal polynomial
    // 3. Identifying which root corresponds to alpha
    vec![complex_embedding(alpha, precision)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_embedding_rational() {
        let alpha = AlgebraicNumber::from_rational(Rational::new(3, 2).unwrap());
        let embedding = complex_embedding(&alpha, 53);

        assert!((embedding.real() - 1.5).abs() < 1e-10);
        assert!(embedding.imag().abs() < 1e-10);
    }

    #[test]
    fn test_all_embeddings_rational() {
        let alpha = AlgebraicNumber::from_i64(5);
        let embeddings = all_complex_embeddings(&alpha, 53);

        assert_eq!(embeddings.len(), 1);
        assert!((embeddings[0].real() - 5.0).abs() < 1e-10);
    }
}
