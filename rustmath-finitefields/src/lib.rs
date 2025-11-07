//! Finite fields GF(p) and GF(p^n)
//!
//! Provides arithmetic in finite fields (Galois fields).
//! - GF(p) for prime p: integers modulo p
//! - GF(p^n) for prime p and n > 1: extension fields

pub mod prime_field;
pub mod extension_field;

pub use prime_field::PrimeField;
pub use extension_field::ExtensionField;

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn basic_prime_field() {
        let p = Integer::from(7);
        let a = PrimeField::new(Integer::from(3), p.clone()).unwrap();
        let b = PrimeField::new(Integer::from(5), p.clone()).unwrap();

        let sum = a.clone() + b.clone();
        // 3 + 5 = 8 ≡ 1 (mod 7)
        assert_eq!(sum.value(), &Integer::from(1));
    }
}
