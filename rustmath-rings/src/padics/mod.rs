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
