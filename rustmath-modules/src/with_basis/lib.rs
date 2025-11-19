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
//! use num_bigint::BigInt;
//!
//! // Create a free module of rank 3
//! let base_ring = BigInt::from(0);
//! let module = FreeModuleWithBasis::standard(base_ring, 3);
//!
//! // Create an element 2*e_0 + 5*e_1 + 3*e_2
//! let elem = ModuleWithBasisElement::from_terms(vec![
//!     (0, BigInt::from(2)),
//!     (1, BigInt::from(5)),
//!     (2, BigInt::from(3)),
//! ]);
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
