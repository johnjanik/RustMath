//! p-adic rings and fields
//!
//! This module provides factory functions and types for working with p-adic numbers,
//! mirroring SageMath's `sage.rings.padics` functionality.
//!
//! ## Overview
//!
//! p-adic numbers extend the usual notion of integers and rationals by introducing
//! a different metric based on a prime p. The p-adic integers Z_p form a ring, and
//! the p-adic numbers Q_p form a field.
//!
//! ## Modules
//!
//! - `factory`: Factory functions for creating p-adic structures (Zp, Qp, Zq, Qq)
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::padics::factory::{Zp, Qp, PrecisionModel};
//! use rustmath_integers::Integer;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create 5-adic integers with precision 20
//! let zp = Zp(Integer::from(5), 20, PrecisionModel::CappedRelative)?;
//! let x = zp.from_int(42)?;
//!
//! // Create 7-adic field
//! let qp = Qp(Integer::from(7), 15, PrecisionModel::CappedAbsolute)?;
//! let y = qp.from_rational_nums(3, 7)?;
//! # Ok(())
//! # }
//! ```

pub mod factory;

// Re-export commonly used items
pub use factory::{
    PadicField, PadicFieldExtension, PadicIntegerExtension, PadicIntegerRing, PrecisionModel, Qp,
    Qq, Zp, Zq,
};
