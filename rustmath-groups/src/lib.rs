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
pub mod additive_abelian_group;
pub mod additive_abelian_wrapper;
pub mod affine_group;
pub mod euclidean_group;
pub mod conjugacy_classes;
pub mod free_group;
pub mod generic;
pub mod group_traits;
pub mod kernel_subgroup;
pub mod representation;

pub use permutation_group::{PermutationGroup, SymmetricGroup, AlternatingGroup};
pub use matrix_group::{MatrixGroup, GLn, SLn};
pub use abelian_group::AbelianGroup;
pub use additive_abelian_group::{AdditiveAbelianGroup, AdditiveAbelianGroupElement, AdditiveAbelianGroupFixedGens, additive_abelian_group};
pub use additive_abelian_wrapper::{AdditiveAbelianGroupWrapper, AdditiveAbelianGroupWrapperElement, UnwrappingMorphism, basis_from_generators};
pub use affine_group::{AffineGroup, AffineGroupElement};
pub use euclidean_group::EuclideanGroup;
pub use conjugacy_classes::{ConjugacyClass, conjugacy_classes, num_conjugacy_classes, GroupElement as ConjugacyGroupElement};
pub use free_group::{FreeGroup, FreeGroupElement};
pub use group_traits::{Group, AbelianGroupTrait, FiniteGroupTrait, GroupElement, AlgebraicGroupTrait, is_group};
pub use kernel_subgroup::{KernelSubgroup, KernelSubgroupElement};
pub use representation::{Representation, Character, CharacterTable, direct_sum, tensor_product};
