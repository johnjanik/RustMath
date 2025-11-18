//! RustMath Integers - Arbitrary precision integer arithmetic
//!
//! This crate provides arbitrary precision integer arithmetic with implementations
//! of number-theoretic algorithms, including advanced factorization methods.

pub mod crt;
pub mod ecm;
pub mod factorint;
pub mod fast_arith;
pub mod integer;
pub mod modular;
pub mod prime;
pub mod quadratic_sieve;

pub use crt::{chinese_remainder_theorem, crt_two};
pub use ecm::{ecm_factor, ecm_factor_complete};
pub use fast_arith::{prime_range, ArithInt, ArithLLong};
pub use integer::Integer;
pub use modular::{primitive_roots, ModularInteger};
pub use quadratic_sieve::{quadratic_sieve_factor, quadratic_sieve_factor_complete};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_arithmetic() {
        let a = Integer::from(42);
        let b = Integer::from(17);
        assert_eq!(a.clone() + b.clone(), Integer::from(59));
        assert_eq!(a.clone() - b.clone(), Integer::from(25));
        assert_eq!(a * b, Integer::from(714));
    }
}
