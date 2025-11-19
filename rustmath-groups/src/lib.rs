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
pub mod artin;
pub mod cactus_group;
pub mod euclidean_group;
pub mod conjugacy_classes;
pub mod cubic_braid;
pub mod free_group;
pub mod finitely_presented;
pub mod finitely_presented_named;
pub mod semidirect_product;
pub mod imaginary_group;
pub mod generic;
pub mod group_exp;
pub mod group_traits;
pub mod indexed_free_group;
pub mod kernel_subgroup;
pub mod raag;
pub mod representation;
pub mod semimonomial_transformation;
pub mod semimonomial_transformation_group;

pub use permutation_group::{PermutationGroup, SymmetricGroup, AlternatingGroup};
pub use matrix_group::{MatrixGroup, GLn, SLn};
pub use abelian_group::AbelianGroup;
pub use additive_abelian_group::{AdditiveAbelianGroup, AdditiveAbelianGroupElement, AdditiveAbelianGroupFixedGens, additive_abelian_group};
pub use additive_abelian_wrapper::{AdditiveAbelianGroupWrapper, AdditiveAbelianGroupWrapperElement, UnwrappingMorphism, basis_from_generators};
pub use affine_group::{AffineGroup, AffineGroupElement};
pub use artin::{ArtinGroup, ArtinGroupElement, FiniteTypeArtinGroup, FiniteTypeArtinGroupElement, CoxeterMatrix};
pub use cactus_group::{CactusGroup, CactusGroupElement, PureCactusGroup, Interval};
pub use conjugacy_classes::{ConjugacyClass, conjugacy_classes, num_conjugacy_classes, GroupElement as ConjugacyGroupElement};
pub use cubic_braid::{CubicBraidGroup, CubicBraidElement, CubicBraidType};
pub use euclidean_group::EuclideanGroup;
pub use free_group::{FreeGroup, FreeGroupElement};
pub use finitely_presented::{FinitelyPresentedGroup, FinitelyPresentedGroupElement, RewritingSystem};
pub use finitely_presented_named::{
    cyclic_presentation, dihedral_presentation, dicyclic_presentation, quaternion_presentation,
    klein_four_presentation, finitely_generated_abelian_presentation,
    finitely_generated_heisenberg_presentation, binary_dihedral_presentation,
    cactus_presentation, symmetric_presentation, alternating_presentation,
};
pub use semidirect_product::{GroupSemidirectProduct, GroupSemidirectProductElement, direct_product, dihedral_group};
pub use imaginary_group::{ImaginaryGroup, ImaginaryElement};
pub use group_exp::{GroupExp, GroupExpElement};
pub use group_traits::{Group, AbelianGroupTrait, FiniteGroupTrait, GroupElement, AlgebraicGroupTrait, is_group};
pub use indexed_free_group::{IndexedGroup, IndexedFreeGroup, IndexedFreeGroupElement, IndexedFreeAbelianGroup, IndexedFreeAbelianGroupElement};
pub use kernel_subgroup::{KernelSubgroup, KernelSubgroupElement};
pub use raag::{RightAngledArtinGroup, RightAngledArtinGroupElement, CohomologyRAAG};
pub use representation::{Representation, Character, CharacterTable, direct_sum, tensor_product};
pub use semimonomial_transformation::{SemimonomialTransformation, Automorphism, action_on_vector, action_on_matrix, compose};
pub use semimonomial_transformation_group::{SemimonomialTransformationGroup, SemimonomialActionVec, SemimonomialActionMat, group_order};
