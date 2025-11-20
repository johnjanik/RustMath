//! Elliptic Curves in Algebraic Geometry
//!
//! This module provides algebraic geometry perspectives on elliptic curves,
//! including isogenies, moduli spaces, and geometric properties.

pub mod isogeny;

pub use isogeny::{Isogeny, IsogenyGraph, KernelPolynomial};
