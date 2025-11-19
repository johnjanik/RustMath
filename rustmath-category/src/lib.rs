//! RustMath Category - Category theory structures
//!
//! This crate provides implementations of:
//! - Categories (base traits for organizing mathematical structures)
//! - Group categories (Groups, Commutative, Topological, CartesianProducts)
//! - Morphisms (structure-preserving maps)
//! - Functors (structure-preserving maps between categories)
//! - Natural transformations
//! - Element and Parent methods for category-aware functionality

pub mod category;
pub mod functor;
pub mod group_category;
pub mod morphism;
pub mod natural_transformation;
pub mod module_category;

pub use category::{
    CartesianProductsCategory, Category, CommutativeCategory, FiniteCategory, TopologicalCategory,
};
pub use functor::{Functor, ForgetfulFunctor, IdentityFunctor};
pub use group_category::{
    CartesianProductElement, CartesianProductGroup, GroupCategory, GroupCategoryCartesianProducts,
    GroupCategoryCommutative, GroupCategoryTopological, GroupElementMethods, GroupParentMethods,
};
pub use morphism::{
    is_morphism, CallMorphism, FormalCoercionMorphism, IdentityMorphism, Isomorphism, Morphism,
    SetIsomorphism, SetMorphism,
};
pub use natural_transformation::NaturalTransformation;
pub use module_category::{
    ModuleCategory, ElementMethods, ParentMethods, SubcategoryMethods,
    CartesianProducts, Homsets, Endset, TensorProducts,
    FiniteDimensional, FinitelyPresented,
};
