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
//! - `newton_polygon`: Newton polygons over a discrete valuation
//! - `unramified`: real unramified-extension arithmetic (Frobenius, N, Tr)
//! - `eisenstein`: real Eisenstein-extension arithmetic (v_L, N, Tr)
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

// Low-level Z/p^N and GF(p) polynomial arithmetic shared by the extension
// machinery (crate-private).
pub(crate) mod poly_arith;

// Newton polygons over a discrete valuation (lower convex hull; a segment of
// slope s and horizontal length l certifies l roots of valuation -s).

pub mod newton_polygon;

pub use newton_polygon::{NewtonPolygon, NewtonSlope};

// Real p-adic extension arithmetic: unramified (Frobenius, norm, trace) and
// Eisenstein (uniformizer valuation, norm, trace), both self-certifying.
//
// DEFERRED (honestly, not facaded): composite towers (unramified over
// Eisenstein / general e,f > 1 extensions as element rings), and field-level
// (negative-valuation) elements in `unramified`/`eisenstein` — the integral
// rings Z_q and Z_p[pi] are implemented; norms/traces/Frobenius on them are
// real and cross-certified.

pub mod eisenstein;
pub mod unramified;

pub use eisenstein::{EisensteinElement, EisensteinExtension};
pub use unramified::{unramified_modulus, UnramifiedElement, UnramifiedExtension};

// OM (Okutsu-Montes / MacLane) factorization over Q_p: monic squarefree
// p-integral polynomials factored by the MacLane tree (valuation::maclane,
// now real at EVERY augmentation level with residue-field towers), with
// DECIDED factor count / (e, f) data and congruence-certified
// approximations. See the module docs for the honest certificate statement.

pub mod om_factorization;

pub use om_factorization::{om_factorization, OmFactor, OmFactorization};

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
