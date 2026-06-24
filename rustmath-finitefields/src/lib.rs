//! Finite fields GF(p) and GF(p^n)
//!
//! Provides arithmetic in finite fields (Galois fields).
//! - GF(p) for prime p: integers modulo p
//! - GF(p^n) for prime p and n > 1: extension fields
//! - Conway polynomials for standard field construction

pub mod cantor_zassenhaus;
pub mod conway;
pub mod extension_field;
pub mod ff_poly;
pub mod integer_mod;
pub mod irreducible;
pub mod prime_field;

pub use cantor_zassenhaus::{factor, factor_squarefree};
pub use conway::{available_conway_polynomials, conway_polynomial, has_conway_polynomial};
pub use extension_field::ExtensionField;
pub use ff_poly::{FFPoly, FiniteFieldElement, Gfpn};
pub use integer_mod::{
    is_integer_mod, lucas, lucas_q1, square_root_mod_prime, square_root_mod_prime_power,
    IntegerMod,
};
pub use irreducible::{
    find_irreducible, generate_gfpn_modulus, is_irreducible, random_irreducible,
};
pub use prime_field::PrimeField;

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
