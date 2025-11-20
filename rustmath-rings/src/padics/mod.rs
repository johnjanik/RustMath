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
