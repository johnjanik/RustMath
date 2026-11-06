//! Modules with distinguished basis
//!
//! This module provides the ModulesWithBasis category, which includes:
//! - Elements with sparse basis representation
//! - Modules with a distinguished basis
//! - Morphisms defined by basis action
//! - Category operations: products, duals, hom sets, tensor products
//!
//! # Examples
//!
//! ```
//! use rustmath_modules::with_basis::element::ModuleWithBasisElement;
//! use rustmath_modules::with_basis::parent::FreeModuleWithBasis;
//! use rustmath_core::ParentWithBasis;
//! use rustmath_integers::Integer;
//!
//! // Create a free module of rank 3 over Z
//! let module: FreeModuleWithBasis<usize, Integer> =
//!     FreeModuleWithBasis::standard(Integer::from(0), 3);
//! assert_eq!(module.dimension(), Some(3));
//! assert_eq!(module.basis_indices(), vec![0, 1, 2]);
//!
//! // Create an element 2*e_0 + 5*e_1 + 3*e_2
//! let elem = ModuleWithBasisElement::from_terms(vec![
//!     (0, Integer::from(2)),
//!     (1, Integer::from(5)),
//!     (2, Integer::from(3)),
//! ]);
//! assert_eq!(
//!     elem.to_dense(&module.basis_indices()),
//!     vec![Integer::from(2), Integer::from(5), Integer::from(3)]
//! );
//!
//! // The same element built from the module's basis vectors
//! let e = |i: usize| module.basis_element(&i).unwrap();
//! let built = e(0).scalar_mul(&Integer::from(2))
//!     + e(1).scalar_mul(&Integer::from(5))
//!     + e(2).scalar_mul(&Integer::from(3));
//! assert_eq!(built, elem);
//! assert!(module.basis_element(&3).is_none());
//! ```

pub mod element;
pub mod parent;
pub mod morphism;
pub mod cartesian_product;
pub mod dual;
pub mod homsets;
pub mod tensor_product;

// Re-export key types
pub use element::ModuleWithBasisElement;
pub use parent::{ModuleWithBasis, ModuleWithBasisParentMethods, FreeModuleWithBasis};
pub use morphism::{ModuleWithBasisMorphism, ModuleWithBasisMorphismMethods, ModuleMorphismWithBasis};
pub use cartesian_product::{CartesianProduct, ProductIndex, CartesianProductElement};
pub use dual::{DualModule, canonical_pairing};
pub use homsets::{HomSpace, HomIndex};
pub use tensor_product::{TensorProduct, TensorIndex, tensor_power};

// Keep legacy exports for compatibility
pub use all::All;
pub use cell_module::CellModule;
pub use indexed_element::IndexedElement;
pub use representation::Representation;
pub use invariant::Invariant;
pub use subquotient::Subquotient;

// Legacy stub modules (to be expanded later)
pub mod all;
pub mod cell_module;
pub mod indexed_element;
pub mod representation;
pub mod invariant;
pub mod subquotient;
