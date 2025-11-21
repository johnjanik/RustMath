//! p-adic numbers and extensions
//!
//! This module provides p-adic field extensions and related structures,
//! building on the basic p-adic arithmetic from rustmath-padics.
//!
//! # Module Contents
//!
//! - `extension`: p-adic field extensions (unramified, Eisenstein, general)
//!
//! # Re-exports
//!
//! For convenience, we re-export the basic p-adic types from rustmath-padics:
//! - `PadicInteger`: p-adic integers Zp
//! - `PadicRational`: p-adic rationals Qp

pub mod extension;

pub use extension::{
    ExtensionType, GaloisGroup, PadicEmbedding, PadicExtension, PadicExtensionElement,
};

// Re-export basic p-adic types
pub use rustmath_padics::{PadicInteger, PadicRational};

// p-adic Power Computers
//
// This module provides efficient power computation and caching for p-adic arithmetic.

pub mod pow_computer;
pub mod pow_computer_ext;

pub use pow_computer::PowComputer;
pub use pow_computer_ext::{ExtensionType, PowComputerExt};

// p-adic numbers and rings with capped relative precision

pub mod capped_relative;

pub use capped_relative::CappedRelativePadicElement;

// p-adic factory functions (Zp, Qp, Zq, Qq)

pub mod factory;

// Re-export commonly used items
pub use factory::{
    PadicField, PadicFieldExtension, PadicIntegerExtension, PadicIntegerRing, PrecisionModel, Qp,
    Qq, Zp, Zq,
};
