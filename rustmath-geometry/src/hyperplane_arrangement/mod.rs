//! Hyperplane arrangements
//!
//! This module provides types and functions for working with hyperplane
//! arrangements - collections of hyperplanes in a vector space.

pub mod affine_subspace;
pub mod arrangement;
pub mod check_freeness;
pub mod hyperplane;
pub mod library;
pub mod ordered_arrangement;
pub mod plot;

pub use affine_subspace::AffineSubspace;
pub use arrangement::{HyperplaneArrangementElement, HyperplaneArrangements};
pub use check_freeness::{construct_free_chain, less_generators, is_free};
pub use hyperplane::{AmbientVectorSpace, Hyperplane};
pub use library::{HyperplaneArrangementLibrary, make_parent};
pub use ordered_arrangement::{OrderedHyperplaneArrangementElement, OrderedHyperplaneArrangements};
pub use plot::{PlotData, HyperplanePlotData, plot_arrangement_data, plot_hyperplane_data};
