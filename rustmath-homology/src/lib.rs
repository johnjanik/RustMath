//! RustMath Homology - Homological algebra structures and computations
//!
//! This crate provides implementations of:
//! - Chain complexes
//! - Homology groups
//! - Cohomology

pub mod chain_complex;
pub mod cochain_complex;

pub use chain_complex::{ChainComplex, HomologyGroup};
pub use cochain_complex::{CochainComplex, CohomologyGroup};
