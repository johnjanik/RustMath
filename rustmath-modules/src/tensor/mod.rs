//! # Tensor Module
//!
//! This module provides comprehensive tensor algebra functionality for free modules,
//! corresponding to SageMath's `sage.tensor.modules` package.
//!
//! ## Module Organization
//!
//! - **comp**: Tensor component storage with symmetry properties
//! - **format_utilities**: Formatting utilities for tensor display
//! - **free_module_basis**: Bases and cobases for free modules
//! - **free_module_tensor**: Base tensor types on free modules
//! - **alternating_contr_tensor**: Alternating contravariant tensors
//! - **free_module_alt_form**: Alternating forms (differential forms)
//! - **free_module_element**: Elements of free modules
//! - **reflexive_module**: Reflexive module structures
//! - **finite_rank_free_module**: Finite rank free modules
//! - **ext_pow_free_module**: Exterior powers of free modules
//! - **tensor_free_module**: Tensor products of free modules
//! - **free_module_automorphism**: Automorphisms of free modules
//! - **free_module_morphism**: Morphisms between free modules
//! - **free_module_homset**: Homomorphism sets
//! - **free_module_linear_group**: General linear groups
//! - **tensor_free_submodule**: Symmetric tensor submodules
//! - **tensor_free_submodule_basis**: Bases for tensor submodules
//! - **tensor_with_indices**: Index notation for tensors
//!
//! ## Main Concepts
//!
//! ### Tensor Types
//!
//! A tensor of type (k,l) has k contravariant indices and l covariant indices.
//! It represents an element of M^⊗k ⊗ M*^⊗l where M is a free module and M* its dual.
//!
//! ### Component Storage
//!
//! Components are stored efficiently with support for:
//! - No symmetry (general components)
//! - Full symmetry (symmetric tensors)
//! - Full antisymmetry (alternating tensors)
//!
//! ### Alternating Forms
//!
//! Differential forms (p-forms) are fully antisymmetric covariant tensors.
//! They support:
//! - Wedge product (exterior product)
//! - Exterior derivative
//! - Evaluation on vectors
//!
//! ### Index Notation
//!
//! Tensors can be written in index notation for Einstein summation convention.
//!
//! ## Examples
//!
//! ```
//! use rustmath_modules::tensor::comp::Components;
//! use rustmath_modules::tensor::free_module_tensor::{FreeModuleTensor, TensorType};
//! use rustmath_modules::tensor::free_module_alt_form::FreeModuleAltForm;
//!
//! // Create a (1,1) tensor on a 3-dimensional module
//! let mut tensor: FreeModuleTensor<i32, ()> = FreeModuleTensor::new(
//!     TensorType::new(1, 1),
//!     3,
//! );
//!
//! // Set components
//! tensor.set_component(vec![0, 1], 5);
//! tensor.set_component(vec![1, 0], 3);
//!
//! // Create a 2-form
//! let mut form: FreeModuleAltForm<i32, ()> = FreeModuleAltForm::new(2, 3);
//! form.set_component(vec![0, 1], 7);
//! ```

pub mod comp;
pub mod format_utilities;
pub mod free_module_basis;
pub mod free_module_tensor;
pub mod alternating_contr_tensor;
pub mod free_module_alt_form;
pub mod free_module_element;
pub mod reflexive_module;
pub mod finite_rank_free_module;
pub mod ext_pow_free_module;
pub mod tensor_free_module;
pub mod free_module_automorphism;
pub mod free_module_morphism;
pub mod free_module_homset;
pub mod free_module_linear_group;
pub mod tensor_free_submodule;
pub mod tensor_free_submodule_basis;
pub mod tensor_with_indices;

// Re-export commonly used types
pub use comp::{Components, CompWithSym, CompFullySym, CompFullyAntiSym, KroneckerDelta, Symmetry};
pub use format_utilities::{
    is_atomic, format_mul_txt, format_mul_latex, format_unop_txt, format_unop_latex,
    FormattedExpansion,
};
pub use free_module_basis::{Basis, FreeModuleBasis, FreeModuleCoBasis, BasisChange};
pub use free_module_tensor::{FreeModuleTensor, TensorType, tensor_product};
pub use alternating_contr_tensor::{AlternatingContrTensor, wedge_product, interior_product};
pub use free_module_alt_form::{FreeModuleAltForm, wedge, exterior_derivative};
pub use free_module_element::FiniteRankFreeModuleElement;
pub use reflexive_module::{ReflexiveModule, ReflexiveModuleBase, ReflexiveModuleDual, ReflexiveModuleTensor};
pub use finite_rank_free_module::{
    FiniteRankFreeModule, FiniteRankDualFreeModule, FiniteRankFreeModuleAbstract,
};
pub use ext_pow_free_module::{ExtPowerFreeModule, ExtPowerDualFreeModule};
pub use tensor_free_module::TensorFreeModule;
pub use free_module_automorphism::FreeModuleAutomorphism;
pub use free_module_morphism::{FiniteRankFreeModuleMorphism, FiniteRankFreeModuleEndomorphism};
pub use free_module_homset::{FreeModuleHomset, FreeModuleEndset};
pub use free_module_linear_group::FreeModuleLinearGroup;
pub use tensor_free_submodule::TensorFreeSubmoduleSym;
pub use tensor_free_submodule_basis::TensorFreeSubmoduleBasisSym;
pub use tensor_with_indices::{TensorWithIndices, TensorIndex, IndexType, contract};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_module_imports() {
        // Verify all modules are accessible
        let _: Components<i32> = Components::new(2, vec![3, 3]);
        let _: FreeModuleTensor<i32, ()> = FreeModuleTensor::new(TensorType::new(1, 1), 3);
        let _: FreeModuleAltForm<i32, ()> = FreeModuleAltForm::new(2, 3);
        let _: AlternatingContrTensor<i32, ()> = AlternatingContrTensor::new(2, 3);
    }

    #[test]
    fn test_tensor_type_system() {
        let scalar = TensorType::new(0, 0);
        assert!(scalar.is_scalar());

        let vector = TensorType::new(1, 0);
        assert!(vector.is_vector());

        let one_form = TensorType::new(0, 1);
        assert!(one_form.is_covector());

        let endo = TensorType::new(1, 1);
        assert!(endo.is_endomorphism());
    }

    #[test]
    fn test_alternating_forms() {
        let mut omega: FreeModuleAltForm<i32, ()> = FreeModuleAltForm::new(2, 3);
        omega.set_component(vec![0, 1], 5);

        assert_eq!(omega.get_component(&[0, 1]), Some(5));
        assert_eq!(omega.get_component(&[1, 0]), Some(-5));
    }
}
