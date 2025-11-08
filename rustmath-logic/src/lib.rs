//! RustMath Logic - Boolean logic, SAT solving, and proofs
//!
//! This crate provides implementations of:
//! - Boolean formulas and expressions
//! - CNF/DNF (Conjunctive/Disjunctive Normal Form)
//! - SAT solving (DPLL algorithm)
//! - Logic operations and simplification

pub mod formula;
pub mod cnf;
pub mod sat;
pub mod operations;

pub use formula::{Formula, Variable};
pub use cnf::{Cnf, Dnf, Clause, Literal};
pub use sat::{SatSolver, SatResult};
pub use operations::{simplify, evaluate, substitute};
