//! RustMath Core - Fundamental algebraic structures and traits
//!
//! This crate provides the core traits and types used throughout the RustMath
//! computer algebra system. It defines fundamental algebraic structures like
//! rings, fields, groups, and modules.
//!
//! # Re-export policy for the Wave-0 trait modules
//!
//! The Wave-0 modules below ([`ordering`], [`valuation`], [`analytic`],
//! [`morphism`], [`nonassoc`], [`coercion`], [`divisibility`]) are exposed as
//! `pub mod` **only** — none of their names are re-exported from the crate
//! root, and they must never be glob-re-exported. Rationale: the trait names
//! `analytic::RealField` and `analytic::ComplexField` intentionally collide
//! with the pre-existing *structs* `rustmath_reals::RealField` and
//! `rustmath_complex::ComplexField`; a root re-export would make downstream
//! `use rustmath_core::*;` + `use rustmath_reals::*;` ambiguous. Consumers
//! import these traits path-qualified (optionally aliased), e.g.:
//!
//! ```ignore
//! use rustmath_core::analytic::RealField;            // the trait
//! use rustmath_core::ordering::{OrderedRing, OrderedField};
//! // or, when both names are needed in one scope:
//! use rustmath_core::analytic::RealField as RealFieldTrait;
//! ```
//!
//! Any future re-export from these modules must be selective (named item by
//! named item, never a glob) and checked against every downstream crate's
//! root namespace.

pub mod traits;
pub mod error;
pub mod parent;
pub mod unique_representation;

// --- Wave 0 (MAGMA port) foundation trait vocabulary (purely additive) ---
// NB: `pub mod` only — no root re-exports. See the crate docs above.
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
