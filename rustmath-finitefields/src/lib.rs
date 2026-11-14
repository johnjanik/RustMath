//! Finite fields GF(p) and GF(p^n)
//!
//! Provides arithmetic in finite fields (Galois fields).
//! - GF(p) for prime p: integers modulo p
//! - GF(p^n) for prime p and n > 1: extension fields
//! - Conway polynomials for standard field construction
//!
//! # Canonical types
//!
//! * `Z/mZ`: the canonical home is [`Zmod`] (parent) / [`IntegerMod`]
//!   (element), with the Sage/MAGMA-style constructor [`Integers`]`(m)`.
//! * `GF(p^n)`: the canonical home is [`FiniteField`] (parent) /
//!   [`FiniteFieldElement`] (element); the legacy per-element
//!   [`ExtensionField`] is a migration target (see the module docs of
//!   [`finite_field`] and [`extension_field`]).

pub mod cantor_zassenhaus;
pub mod conway;
pub mod extension_field;
pub mod ff_poly;
pub mod integer_mod;
pub mod irreducible;
pub mod prime_field;
mod prime_field_factor; // ch 23 §23.8.1: FactorizationField for GF(p) (Berlekamp)
mod ring_hooks; // shared `Ring` element-predicate hooks (ch 17 §17.5.5, ch 19, ch 21)
pub mod zmod;

// --- MAGMA port Wave 1 (chapters 21, 19, 48) ---
pub mod finite_field; // ch 21: GF(p^n) as a shared Parent (Conway embeddings)
pub mod galois_ring; // ch 48: Galois rings GR(p^a, d) (finite chain rings)
mod group_order; // ch 19, 21: orders from the factored group exponent (Cohen Alg. 1.4.3)
pub mod poly_factor; // ch 21: Cantor–Zassenhaus / DDF / EDF / irreducibility
                     // ch 19 (Integers(m) = Z/mZ) lives in `zmod` / `integer_mod`.

// --- MAGMA port Wave 2 (chapter 21 depth) ---
pub mod embedding; // ch 21: norm-compatible embeddings GF(p^m) -> GF(p^n), m | n

// Lava-galois GF(p^n) stack (cantor_zassenhaus / ff_poly / irreducible): its
// `FiniteFieldElement`, `find_irreducible`, `is_irreducible`, `random_irreducible`
// collide with the canonical Wave-1 exports below, so reach those via their module
// paths (e.g. `ff_poly::FiniteFieldElement`, `irreducible::is_irreducible`).
pub use cantor_zassenhaus::{factor, factor_squarefree};
pub use conway::{available_conway_polynomials, conway_polynomial, has_conway_polynomial};
pub use extension_field::ExtensionField;
pub use ff_poly::{FFPoly, Gfpn};
pub use integer_mod::{
    all_square_roots_mod_prime_power, is_integer_mod, lucas, lucas_q1, square_root_mod_prime,
    square_root_mod_prime_power, IntegerMod,
};
pub use irreducible::generate_gfpn_modulus;
pub use prime_field::PrimeField;
pub use zmod::{Integers, Zmod};

pub use embedding::FieldEmbedding;
pub use finite_field::{FiniteField, FiniteFieldElement};
pub use galois_ring::{GaloisRing, GaloisRingElement};
pub use poly_factor::{
    distinct_degree_factorization, equal_degree_factorization, factor as factor_fq,
    find_irreducible, irreducible_polynomial, is_irreducible, is_irreducible_fp,
    random_irreducible, roots, square_free_factorization, FqPoly,
};

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
