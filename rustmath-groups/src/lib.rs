//! RustMath Groups - Group theory structures and algorithms
//!
//! This crate provides implementations of various types of groups:
//! - Permutation groups (symmetric, alternating)
//! - Matrix groups (GL, SL, O, U)
//! - Abelian groups
//! - Free groups
//! - Generic group operations
//! - Representation theory

pub mod permutation_group;
pub mod matrix_group;
pub mod abelian_group;
pub mod free_group;
pub mod generic;
pub mod representation;

pub use permutation_group::{PermutationGroup, SymmetricGroup, AlternatingGroup};
pub use matrix_group::{MatrixGroup, GLn, SLn};
pub use abelian_group::AbelianGroup;
pub use free_group::{FreeGroup, FreeGroupElement};
pub use representation::{Representation, Character, CharacterTable, direct_sum, tensor_product};
