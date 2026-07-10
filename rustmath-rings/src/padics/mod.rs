//! p-adic numbers and extensions
//!
//! This is the canonical home of RustMath's p-adic machinery. The
//! `rustmath-padics` crate is now a thin re-export shim over this module.
//!
//! # Module Contents
//!
//! - `padic_integer`: p-adic integers zp (basic fixed-modulus model)
//! - `padic_rational`: p-adic rationals qp
//! - `capped_relative`: capped-relative-precision elements
//! - `extension`: p-adic field extensions (unramified, Eisenstein, general)
//! - `factory`: constructors `zp`, `qp`, `zq`, `qq`
//! - `pow_computer` / `pow_computer_ext`: cached prime-power arithmetic

pub mod padic_integer;
pub mod padic_rational;

pub use padic_integer::{hensel_lift_root, PadicInteger};
pub use padic_rational::PadicRational;

pub mod extension;

pub use extension::{
    ExtensionType, GaloisGroup, PadicEmbedding, PadicExtension, PadicExtensionElement,
};

// p-adic Power Computers
//
// This module provides efficient power computation and caching for p-adic arithmetic.

pub mod pow_computer;
pub mod pow_computer_ext;

pub use pow_computer::PowComputer;
pub use pow_computer_ext::PowComputerExt;

// p-adic numbers and rings with capped relative precision

pub mod capped_relative;

pub use capped_relative::CappedRelativePadicElement;

// p-adic factory functions (zp, qp, zq, qq)

pub mod factory;

// Re-export commonly used items
pub use factory::{
    PadicField, PadicFieldExtension, PadicIntegerExtension, PadicIntegerRing, PrecisionModel, qp,
    qq, zp, zq,
};
