//! RustMath Number Theory - Advanced number-theoretic algorithms
//!
//! This crate provides algorithms for number theory including primality testing,
//! factorization, modular arithmetic, and quadratic forms.

pub mod quadratic_forms;

// Re-export prime functions from integers
pub use rustmath_integers::prime::*;

// Re-export quadratic forms
pub use quadratic_forms::QuadraticForm;

// Future modules:
// - Elliptic curves
// - Diophantine equations
// - Modular forms
