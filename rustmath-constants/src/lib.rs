//! RustMath Constants - Mathematical constants, tables, and databases
//!
//! This crate provides:
//! - Extended precision mathematical constants (π, e, φ, etc.)
//! - Mathematical tables (logarithms, trigonometric values, etc.)
//! - Lookup tables for common sequences (Fibonacci, primes, Catalan, etc.)
//! - Precomputed factorizations database
//! - Primality certificates for known primes

pub mod certificates;
pub mod constants;
pub mod factorizations;
pub mod sequences;
pub mod tables;

// Re-export main items for convenience
pub use certificates::{PrimalityCertificate, PrattCertificate};
pub use constants::{
    pi_digits, e_digits, phi_digits, sqrt2_digits, euler_gamma_digits,
    PI, E, PHI, SQRT2, EULER_GAMMA,
};
pub use factorizations::{FactorizationDatabase, get_factorization, is_factored};
pub use sequences::{
    SequenceDatabase, get_sequence, fibonacci_lookup, prime_lookup,
    catalan_lookup, bell_lookup, lucas_lookup,
};
pub use tables::{
    TrigTable, LogTable, get_sin_approx, get_cos_approx, get_log_approx,
};
