//! RustMath Rationals - Rational number arithmetic
//!
//! This crate provides rational number (fraction) arithmetic with automatic
//! simplification to lowest terms.

pub mod continued_fraction;
pub mod rational;

pub use continued_fraction::{ContinuedFraction, PeriodicContinuedFraction};
pub use rational::Rational;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_arithmetic() {
        let a = Rational::new(1, 2).unwrap();
        let b = Rational::new(1, 3).unwrap();

        let sum = a.clone() + b.clone();
        assert_eq!(sum, Rational::new(5, 6).unwrap());

        let product = a.clone() * b.clone();
        assert_eq!(product, Rational::new(1, 6).unwrap());
    }
}
