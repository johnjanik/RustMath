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
pub mod maclane;
pub mod mapped_valuation;
pub mod residue_tower;
pub mod valuation_space;
pub mod value_group;

pub use maclane::{
    mac_lane_approximants, phi_adic_expansion, Augmentation, BaseValuation, KeyCheck,
    PAdicBaseValuation, PAdicInductiveValuation, PlaceBaseValuation, QVal,
};
pub use residue_tower::{ResidueTower, TowerElt};
pub use valuation::{DiscretePseudoValuation, DiscreteValuation, InfiniteDiscretePseudoValuation};
pub use trivial_valuation::TrivialDiscreteValuation;
pub use value_group::{DiscreteValueGroup, DiscreteValueSemigroup};

// NOTE (trait collision, intentionally NOT resolved in this chunk): the
// crate-local `valuation::valuation::DiscreteValuation` trait coexists with
// `rustmath_core::valuation`'s `DiscreteValuation`. The MacLane machinery in
// `maclane` uses its own rational-valued `BaseValuation` abstraction and does
// not touch either.
