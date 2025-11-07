//! RustMath Core - Fundamental algebraic structures and traits
//!
//! This crate provides the core traits and types used throughout the RustMath
//! computer algebra system. It defines fundamental algebraic structures like
//! rings, fields, groups, and modules.

pub mod traits;
pub mod error;

pub use error::{MathError, Result};
pub use traits::*;
