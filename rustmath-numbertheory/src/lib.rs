//! RustMath Number Theory - Advanced number-theoretic algorithms
//!
//! This crate provides algorithms for number theory including primality testing,
//! factorization, modular arithmetic, Bernoulli numbers, and quadratic forms.

pub mod bernoulli;
pub mod quadratic_forms;

// Re-export prime functions from integers
pub use rustmath_integers::prime::*;

// Re-export quadratic forms
pub use quadratic_forms::QuadraticForm;

// Re-export Bernoulli number functions
pub use bernoulli::{
    bernoulli_number,
    bernoulli_numbers_vec,
    bernoulli_mod_p,
    bernoulli_mod_p_single,
    bernoulli_multimodular,
    verify_bernoulli_mod_p,
};

// Future modules:
// - Elliptic curves
// - Diophantine equations
// - Modular forms
