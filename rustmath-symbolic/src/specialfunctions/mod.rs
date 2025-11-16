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
//! - `other`: Various mathematical utility functions (abs, ceil, floor, factorial, etc.)
//! - `prime_pi`: Prime counting function and related functions
//!
//! # Examples
//!
//! ```
//! use rustmath_symbolic::specialfunctions::generalized::*;
//! use rustmath_symbolic::specialfunctions::min_max::*;
//! use rustmath_symbolic::specialfunctions::other::*;
//! use rustmath_symbolic::Expr;
//!
//! let h = heaviside(&Expr::from(5));
//! let m = min_symbolic(&[Expr::from(1), Expr::from(2)]);
//! let f = factorial(&Expr::from(5));
//! ```

pub mod airy;
pub mod generalized;
pub mod min_max;
pub mod other;
pub mod prime_pi;

pub use airy::{airy_ai, airy_ai_prime, airy_bi, airy_bi_prime};
pub use generalized::{dirac_delta, heaviside, kronecker_delta, signum, unit_step};
pub use min_max::{max2, max_symbolic, min2, min_symbolic};
pub use other::{
    abs_symbolic, binomial, ceil, conjugate, factorial, floor, frac, imag_part, real_part,
};
pub use prime_pi::{legendre_phi, prime_pi};

