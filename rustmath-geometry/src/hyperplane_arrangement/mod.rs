//! Hyperplane arrangements
//!
//! This module provides types and functions for working with hyperplane
//! arrangements - collections of hyperplanes in a vector space.

pub mod affine_subspace;
pub mod hyperplane;

pub use affine_subspace::AffineSubspace;
pub use hyperplane::{AmbientVectorSpace, Hyperplane};
