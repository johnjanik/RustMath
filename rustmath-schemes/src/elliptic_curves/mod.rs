//! Elliptic Curves as Schemes
//!
//! This module provides elliptic curves as algebraic varieties/schemes,
//! going beyond basic point arithmetic to include:
//! - Minimal models and reduction types
//! - Conductor computation
//! - Torsion group structure
//! - Database integration (Cremona, LMFDB)
//! - BSD conjecture computations

pub mod rational;

pub use rational::{EllipticCurveRational, ReductionType, TorsionGroup};
