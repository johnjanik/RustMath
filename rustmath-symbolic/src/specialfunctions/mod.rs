//! Special Functions Module
//!
//! This module provides implementations of various special functions
//! used in mathematics, physics, and engineering.
//!
//! Corresponds to sage.functions.*
//!
//! # Submodules
//!
//! - `generalized`: Generalized functions (Dirac delta, Heaviside, etc.)
//! - `min_max`: Symbolic minimum and maximum functions
//! - `airy`: Airy functions and their derivatives
//!
//! # Examples
//!
//! ```
//! use rustmath_symbolic::specialfunctions::generalized::*;
//! use rustmath_symbolic::specialfunctions::min_max::*;
//! use rustmath_symbolic::Expr;
//!
//! let h = heaviside(&Expr::from(5));
//! let m = min_symbolic(&[Expr::from(1), Expr::from(2)]);
//! ```

pub mod generalized;
pub mod min_max;
pub mod airy;

pub use generalized::{dirac_delta, heaviside, kronecker_delta, signum, unit_step};
pub use min_max::{max2, max_symbolic, min2, min_symbolic};
pub use airy::{airy_ai, airy_ai_prime, airy_bi, airy_bi_prime};
