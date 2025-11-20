//! RustMath Crystals - Crystal bases and Kashiwara crystals
//!
//! This crate provides crystal bases, which are combinatorial models of representations
//! of quantum groups and Lie algebras. Crystal bases were introduced by Kashiwara and
//! Lusztig independently, and they provide a powerful tool for studying representation
//! theory through combinatorics.
//!
//! # Key Concepts
//!
//! - **Crystal operators**: The raising and lowering operators e_i and f_i
//! - **Weight function**: Maps crystal elements to weights
//! - **Tensor products**: Combining crystals
//! - **Character formulas**: Computing characters from crystals
//! - **Highest weight crystals**: Crystals for irreducible representations
//! - **Affine crystals**: Crystals for affine Lie algebras
//! - **Littelmann paths**: Path model for crystals
//!
//! # Examples
//!
//! ```
//! use rustmath_crystals::{Crystal, TableauCrystal};
//! use rustmath_combinatorics::Tableau;
//!
//! // Create a tableau crystal
//! let shape = vec![3, 2];
//! let crystal = TableauCrystal::new(shape, 2);
//!
//! // Apply crystal operators
//! // let b = crystal.highest_weight_element();
//! // let b_prime = crystal.f_i(&b, 1);
//! ```

pub mod operators;
pub mod tableau_crystal;
pub mod tensor_product;
pub mod character;
pub mod affine;
pub mod highest_weight;
pub mod littelmann;
pub mod root_system;
pub mod weight;

pub use operators::{Crystal, CrystalElement};
pub use tableau_crystal::TableauCrystal;
pub use tensor_product::TensorProductCrystal;
pub use character::character;
pub use affine::AffineCrystal;
pub use highest_weight::HighestWeightCrystal;
pub use littelmann::{LittelmannPath, LittelmannCrystal};
pub use root_system::{RootSystem, RootSystemType};
pub use weight::Weight;
