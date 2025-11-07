//! RustMath Polynomials - Polynomial arithmetic
//!
//! This crate provides polynomial arithmetic over various coefficient rings.

pub mod factorization;
pub mod multivariate;
pub mod polynomial;
pub mod roots;
pub mod univariate;

pub use factorization::{
    content, factor_over_integers, is_square_free, primitive_part, square_free_factorization,
};
pub use multivariate::{Monomial, MultivariatePolynomial};
pub use polynomial::Polynomial;
pub use roots::{rational_roots, solve_quadratic, QuadraticRoots};
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
