//! RustMath Quantum Groups - Quantum groups and q-analogs
//!
//! This crate provides quantum group structures and q-analogs of classical
//! mathematical objects, including q-integers, q-factorials, and q-binomials.
//!
//! Corresponds to sage.algebras.quantum_groups

pub mod q_numbers;

pub use q_numbers::{q_binomial, q_factorial, q_int};
