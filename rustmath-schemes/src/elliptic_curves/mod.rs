//! Elliptic Curves over Schemes
//!
//! This module provides elliptic curve functionality from an algebraic geometry
//! perspective, implementing generic elliptic curves over arbitrary fields.
//!
//! This corresponds to `sage.schemes.elliptic_curves` in SageMath.
//!
//! # Modules
//!
//! - `generic`: Generic elliptic curves over any field (EllipticCurve_generic base class)
//!
//! # Examples
//!
//! ## Working with elliptic curves over different fields
//!
//! ```
//! use rustmath_schemes::elliptic_curves::generic::{EllipticCurve, Point};
//! use rustmath_rationals::Rational;
//!
//! // Curve over the rationals
//! let curve = EllipticCurve::short_weierstrass(
//!     Rational::from_integer(-1),
//!     Rational::from_integer(0),
//! );
//! ```

pub mod generic;

#[cfg(test)]
mod tests;

// Re-export commonly used types
pub use generic::{EllipticCurve, Point};
