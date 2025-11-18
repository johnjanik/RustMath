//! # Semirings Module
//!
//! This module provides various semiring structures.
//!
//! A semiring (or rig) is an algebraic structure with two binary operations
//! (addition and multiplication) that generalizes rings by not requiring
//! additive inverses.

pub mod non_negative_integer_semiring;
pub mod tropical_semiring;
pub mod tropical_polynomial;
pub mod tropical_mpolynomial;
pub mod tropical_variety;

pub use non_negative_integer_semiring::{NonNegativeInteger, NonNegativeIntegerSemiring};
pub use tropical_semiring::{TropicalSemiring, TropicalSemiringElement};
