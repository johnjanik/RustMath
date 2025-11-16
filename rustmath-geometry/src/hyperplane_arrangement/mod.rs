//! Hyperplane arrangements
//!
//! This module provides types and functions for working with hyperplane
//! arrangements - collections of hyperplanes in a vector space.

pub mod affine_subspace;
pub mod arrangement;
pub mod hyperplane;
pub mod library;
pub mod ordered_arrangement;

pub use affine_subspace::AffineSubspace;
pub use arrangement::{HyperplaneArrangementElement, HyperplaneArrangements};
pub use hyperplane::{AmbientVectorSpace, Hyperplane};
pub use library::{HyperplaneArrangementLibrary, make_parent};
pub use ordered_arrangement::{OrderedHyperplaneArrangementElement, OrderedHyperplaneArrangements};
