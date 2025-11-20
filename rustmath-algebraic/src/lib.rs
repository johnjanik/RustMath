//! RustMath Algebraic Numbers - Algebraic closure of Q
//!
//! This crate provides algebraic numbers (QQbar) and algebraic real numbers (AA).
//! Algebraic numbers are roots of polynomials with rational coefficients.
//!
//! # Features
//!
//! - Exact arithmetic with algebraic numbers
//! - Minimal polynomial computation
//! - Radical expressions and simplification
//! - Complex embeddings
//! - Galois conjugates
//! - Exact comparisons for algebraic reals
//!
//! # Examples
//!
//! ```
//! use rustmath_algebraic::{AlgebraicNumber, AlgebraicReal};
//! use rustmath_rationals::Rational;
//!
//! // Create sqrt(2)
//! let sqrt2 = AlgebraicReal::sqrt(2);
//!
//! // Create algebraic numbers from rationals
//! let half = AlgebraicNumber::from_rational(Rational::new(1, 2).unwrap());
//!
//! // Arithmetic operations
//! let sum = sqrt2.clone() + AlgebraicReal::from_i64(3);
//! ```

pub mod descriptor;
pub mod algebraic_number;
pub mod algebraic_real;
pub mod algebraic_field;
pub mod algebraic_real_field;
pub mod minimal_polynomial;
pub mod operations;
pub mod radicals;
pub mod embeddings;
pub mod conjugates;
pub mod comparison;

pub use descriptor::{AlgebraicDescriptor, ANRational, ANRoot, ANUnaryExpr, ANBinaryExpr};
pub use algebraic_number::AlgebraicNumber;
pub use algebraic_real::AlgebraicReal;
pub use algebraic_field::AlgebraicField;
pub use algebraic_real_field::AlgebraicRealField;
pub use minimal_polynomial::minimal_polynomial;
pub use operations::{algebraic_add, algebraic_mul, algebraic_neg, algebraic_inverse};
pub use radicals::{nth_root, sqrt, radical_simplify};
pub use embeddings::{complex_embedding, all_complex_embeddings};
pub use conjugates::galois_conjugates;
pub use comparison::{algebraic_compare, algebraic_eq};

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_rational_algebraic() {
        let a = AlgebraicNumber::from_rational(Rational::new(3, 2).unwrap());
        let b = AlgebraicNumber::from_rational(Rational::new(1, 2).unwrap());

        let sum = a.clone() + b.clone();
        assert!(sum.is_rational());
        assert_eq!(sum.to_rational().unwrap(), Rational::new(2, 1).unwrap());
    }

    #[test]
    fn test_sqrt_construction() {
        let sqrt2 = AlgebraicReal::sqrt(2);

        // sqrt(2)^2 = 2
        let squared = sqrt2.clone() * sqrt2.clone();
        assert!(squared.is_rational());
        assert_eq!(squared.to_rational().unwrap(), Rational::new(2, 1).unwrap());
    }

    #[test]
    fn test_golden_ratio() {
        // Golden ratio: φ = (1 + sqrt(5)) / 2
        let sqrt5 = AlgebraicReal::sqrt(5);
        let one = AlgebraicReal::from_i64(1);
        let two = AlgebraicReal::from_i64(2);

        let phi = (one + sqrt5) / two;

        // φ² = φ + 1
        let phi_squared = phi.clone() * phi.clone();
        let phi_plus_one = phi.clone() + AlgebraicReal::from_i64(1);

        assert_eq!(phi_squared, phi_plus_one);
    }
}
