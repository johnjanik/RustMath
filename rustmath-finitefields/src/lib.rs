//! Finite fields GF(p) and GF(p^n)
//!
//! Provides arithmetic in finite fields (Galois fields).
//! - GF(p) for prime p: integers modulo p
//! - GF(p^n) for prime p and n > 1: extension fields
//! - Conway polynomials for standard field construction

pub mod conway;
pub mod extension_field;
pub mod integer_mod;
pub mod prime_field;

// --- MAGMA port Wave 1 (chapters 21, 19, 48) ---
pub mod finite_field; // ch 21: GF(p^n) as a shared Parent (Conway embeddings)
pub mod galois_ring; // ch 48: Galois rings GR(p^a, d) (finite chain rings)
pub mod poly_factor; // ch 21: Cantor–Zassenhaus / DDF / EDF / irreducibility
pub mod residue_ring; // ch 19: canonical Integers(m) = Z/mZ

pub use conway::{available_conway_polynomials, conway_polynomial, has_conway_polynomial};
pub use extension_field::ExtensionField;
pub use integer_mod::{
    is_integer_mod, lucas, lucas_q1, square_root_mod_prime, square_root_mod_prime_power,
    IntegerMod,
};
pub use prime_field::PrimeField;

pub use finite_field::{FiniteField, FiniteFieldElement};
pub use galois_ring::{GaloisRing, GaloisRingElement};
pub use poly_factor::{
    distinct_degree_factorization, equal_degree_factorization, factor as factor_fq,
    find_irreducible, irreducible_polynomial, is_irreducible, is_irreducible_fp, roots,
    square_free_factorization, FqPoly,
};
pub use residue_ring::{Integers, ResidueClassRingElement};

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
