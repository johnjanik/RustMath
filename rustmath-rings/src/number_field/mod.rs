//! # Number Field Module
//!
//! This module provides functionality for working with algebraic number fields,
//! extending the capabilities in rustmath-numberfields with morphisms and
//! Galois theory computations.
//!
//! ## Submodules
//!
//! - `morphisms`: Homomorphisms, embeddings, automorphisms, and Galois groups
//! - `tests`: Comprehensive test suite for splitting fields and Galois theory

pub mod morphisms;
#[cfg(test)]
mod tests;

// Re-export commonly used types
pub use morphisms::{
    NumberFieldMorphism,
    NumberFieldEmbedding,
    NumberFieldAutomorphism,
    GaloisGroup,
    compute_automorphisms,
    is_galois_extension,
    is_normal_extension,
    is_separable_extension,
    splitting_field,
    galois_group,
    MorphismError,
    Result as MorphismResult,
};
//! Number Field Orders
//!
//! This module implements orders (rings of integers) in algebraic number fields,
//! corresponding to sage.rings.number_field.order.

pub mod order;

pub use order::{Order, OrderElement, OrderIdeal, OrderError};
