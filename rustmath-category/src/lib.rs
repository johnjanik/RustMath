//! RustMath Category - Category theory structures
//!
//! This crate provides implementations of:
//! - Categories (via traits in rustmath-core)
//! - Functors
//! - Natural transformations

pub mod functor;
pub mod natural_transformation;

pub use functor::{Functor, IdentityFunctor, ForgetfulFunctor};
pub use natural_transformation::NaturalTransformation;
