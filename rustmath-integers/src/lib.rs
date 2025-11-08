//! RustMath Integers - Arbitrary precision integer arithmetic
//!
//! This crate provides arbitrary precision integer arithmetic with implementations
//! of number-theoretic algorithms.

pub mod crt;
pub mod integer;
pub mod modular;
pub mod prime;

pub use crt::{chinese_remainder_theorem, crt_two};
pub use integer::Integer;
pub use modular::{primitive_roots, ModularInteger};

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
