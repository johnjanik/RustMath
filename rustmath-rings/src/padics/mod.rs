//! p-adic Power Computers
//!
//! This module provides efficient power computation and caching for p-adic arithmetic.
//!
//! ## Modules
//!
//! - [`pow_computer`]: Base power computer with efficient p-power caching
//! - [`pow_computer_ext`]: Extended power computer for p-adic field extensions
//!   with Frobenius endomorphism support
//!
//! ## Quick Start
//!
//! ```
//! use rustmath_integers::Integer;
//! use rustmath_rings::padics::{PowComputer, PowComputerExt};
//!
//! // Basic power computer
//! let pc = PowComputer::new(Integer::from(5), 10);
//! let p_cubed = pc.pow(3); // Efficiently get 5^3
//!
//! // For extensions with Frobenius
//! let pc_ext = PowComputerExt::unramified(Integer::from(5), 10, 3);
//! let frob_exp = pc_ext.frobenius(2); // Get Frobenius exponent
//! ```

pub mod pow_computer;
pub mod pow_computer_ext;

pub use pow_computer::PowComputer;
pub use pow_computer_ext::{ExtensionType, PowComputerExt};
//! p-adic numbers and rings
//!
//! This module provides implementations of p-adic numbers with various precision models.
//!
//! # Precision Models
//!
//! - **Capped Relative Precision**: Elements track precision relative to their valuation.
//!   This is the most commonly used model and corresponds to Sage's default p-adic implementation.
//!
//! # Examples
//!
//! ```rust
//! use rustmath_integers::Integer;
//! use rustmath_rings::padics::capped_relative::CappedRelativePadicElement;
//!
//! // Create 7 + O(5^10) in Q_5
//! let x = CappedRelativePadicElement::new(
//!     Integer::from(7),
//!     Integer::from(5),
//!     10
//! ).unwrap();
//!
//! // Arithmetic operations track precision correctly
//! let y = CappedRelativePadicElement::new(
//!     Integer::from(3),
//!     Integer::from(5),
//!     10
//! ).unwrap();
//!
//! let sum = x.clone() + y.clone();
//! let prod = x * y;
//! ```

pub mod capped_relative;

pub use capped_relative::CappedRelativePadicElement;
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
