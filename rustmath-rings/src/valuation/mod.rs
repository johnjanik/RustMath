//! # Valuations Module
//!
//! This module provides discrete valuations on rings and their applications.
//!
//! Valuations are fundamental in algebraic number theory, algebraic geometry,
//! and the study of local fields.

pub mod valuation;
pub mod trivial_valuation;
pub mod scaled_valuation;
pub mod gauss_valuation;
pub mod developing_valuation;
pub mod inductive_valuation;
pub mod limit_valuation;
pub mod mapped_valuation;
pub mod valuation_space;
pub mod value_group;

pub use valuation::{DiscretePseudoValuation, DiscreteValuation, InfiniteDiscretePseudoValuation};
pub use trivial_valuation::TrivialDiscreteValuation;
pub use value_group::{DiscreteValueGroup, DiscreteValueSemigroup};
