//! RustMath Quantum Groups - Quantum groups and q-analogs
//!
//! This crate provides quantum group structures and q-analogs of classical
//! mathematical objects, including q-integers, q-factorials, q-binomials,
//! q-Bernoulli numbers, and Fock space representations.
//!
//! Corresponds to sage.algebras.quantum_groups

pub mod fock_space;
pub mod q_bernoulli;
pub mod q_numbers;

pub use fock_space::{FockSpaceElement, FockSpaceParams};
pub use q_bernoulli::{q_bernoulli, RationalFunction};
pub use q_numbers::{q_binomial, q_factorial, q_int};
