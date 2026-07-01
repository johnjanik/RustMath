//! RustMath Core - Fundamental algebraic structures and traits
//!
//! This crate provides the core traits and types used throughout the RustMath
//! computer algebra system. It defines fundamental algebraic structures like
//! rings, fields, groups, and modules.

pub mod traits;
pub mod error;
pub mod parent;
pub mod unique_representation;

// --- Wave 0 (MAGMA port) foundation trait vocabulary (purely additive) ---
pub mod ordering;
pub mod valuation;
pub mod analytic;
pub mod morphism;
pub mod nonassoc;
pub mod coercion;
pub mod divisibility;

pub use error::{MathError, Result};
pub use traits::*;
pub use parent::*;
pub use unique_representation::{UniqueCache, UniqueRepresentation};
