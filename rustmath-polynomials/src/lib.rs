//! RustMath Polynomials - Polynomial arithmetic
//!
//! This crate provides polynomial arithmetic over various coefficient rings.

pub mod polynomial;
pub mod univariate;

pub use polynomial::Polynomial;
pub use univariate::UnivariatePolynomial;

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn basic_polynomial() {
        // Create polynomial 3x^2 + 2x + 1
        let p = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(2),
            Integer::from(3),
        ]);

        assert_eq!(p.degree(), Some(2));
    }
}
