//! RustMath Category - Category theory structures
//!
//! This crate provides implementations of:
//! - Categories (via traits in rustmath-core)
//! - Morphisms (structure-preserving maps)
//! - Functors (structure-preserving maps between categories)
//! - Natural transformations

pub mod functor;
pub mod morphism;
pub mod natural_transformation;

pub use functor::{Functor, IdentityFunctor, ForgetfulFunctor};
pub use morphism::{
    is_morphism, CallMorphism, FormalCoercionMorphism, IdentityMorphism, Isomorphism, Morphism,
    SetIsomorphism, SetMorphism,
};
pub use natural_transformation::NaturalTransformation;
