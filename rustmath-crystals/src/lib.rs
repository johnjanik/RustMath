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
//! - **Tensor products**: Combining crystals using tensor product rules
//! - **Character formulas**: Computing characters from crystals
//! - **Highest weight crystals**: Crystals for irreducible representations B(λ)
//! - **Affine crystals**: Crystals for affine Lie algebras with level structure
//! - **Littelmann paths**: Path model for crystals in weight space
//! - **Spin crystals**: Spin representations for types B and D
//! - **Kirillov-Reshetikhin crystals**: Finite crystals B^{r,s} for affine types
//! - **Elementary crystals**: B(∞), T_λ, R_λ, and B(i)
//! - **Nakajima monomials**: Polyhedral realization of crystals
//! - **Crystal morphisms**: Structure-preserving maps between crystals
//!
//! # Supported Types
//!
//! This crate supports all finite and affine Lie types:
//! - **Classical types**: A_n, B_n, C_n, D_n
//! - **Exceptional types**: E_6, E_7, E_8, F_4, G_2
//! - **Affine types**: A_n^(1), B_n^(1), C_n^(1), D_n^(1), etc.
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
pub mod spin_crystal;
pub mod kr_crystal;
pub mod elementary;
pub mod morphisms;
pub mod nakajima;

pub use operators::{Crystal, CrystalElement};
pub use tableau_crystal::{TableauCrystal, TableauElement};
pub use tensor_product::{TensorProductCrystal, TensorElement};
pub use character::{character, Character};
pub use affine::{AffineCrystal, AffineElement};
pub use highest_weight::{HighestWeightCrystal, HighestWeightElement};
pub use littelmann::{LittelmannPath, LittelmannCrystal};
pub use root_system::{RootSystem, RootSystemType};
pub use weight::Weight;
pub use spin_crystal::{SpinElement, SpinCrystalB, SpinCrystalD};
pub use kr_crystal::{KRCrystal, KRElement, KRTensorProduct};
pub use elementary::{
    InfinityCrystal, InfinityCrystalElement, OneDimCrystal, OneDimElement,
    ElementaryCrystal, ElementaryElement, SeminormalCrystal, SeminormalElement,
};
pub use morphisms::{
    CrystalMorphism, CrystalIsomorphism, CrystalEmbedding,
    IdentityMorphism, ExplicitMorphism, VirtualCrystal,
};
pub use nakajima::{NakajimaCrystal, NakajimaMonomial, PolyhedralData, AVariables};
