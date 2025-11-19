//! RustMath Category - Category theory structures
//!
//! This crate provides comprehensive implementations of category theory:
//!
//! # Core Structures
//! - **Categories**: Base traits for organizing mathematical structures
//! - **Morphisms**: Structure-preserving maps between objects
//! - **Functors**: Structure-preserving maps between categories
//! - **Natural Transformations**: Morphisms between functors
//!
//! # Algebraic Infrastructure
//! - **Axioms**: Traits for associativity, commutativity, unity, identity, inverse
//! - **Coercion System**: Automatic type conversion between algebraic structures
//! - **Algebraic Morphisms**: Ring, field, module, group, and algebra homomorphisms
//! - **Morphism Composition**: Utilities for composing and verifying morphisms
//!
//! # Concrete Categories
//! - **Ring Category**: Category of rings and ring homomorphisms
//! - **Group Category**: Category of groups and group homomorphisms
//! - **Module Category**: Category of modules and module homomorphisms
//! - **Field Category**: Category of fields and field homomorphisms
//!
//! # Element and Parent Methods
//! - SageMath-style category methods for elements and parent structures
//! - Integration with rustmath-core trait hierarchy

pub mod axioms;
pub mod category;
pub mod coercion;
pub mod functor;
pub mod group_category;
pub mod morphism;
pub mod algebraic_morphisms;
pub mod morphism_composition;
pub mod natural_transformation;
pub mod module_category;
pub mod ring_category;

// Re-export core traits and types
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

// Re-export new infrastructure
pub use axioms::{
    Axiom, Associativity, Commutativity, Identity, Unity, Inverse,
    Distributivity, Closure, Idempotence, Absorption, NoZeroDivisors,
    AxiomSet, SatisfiesAxiom,
};
pub use coercion::{
    Coercion, IdentityCoercion, ComposedCoercion, CoercionMap,
    CoerceInto, CoerceFrom, CoercionPath, CoercionDiscovery,
    coercion_to_morphism,
};
pub use algebraic_morphisms::{
    RingMorphism, FieldMorphism, ModuleMorphism, AlgebraMorphism, GroupMorphism,
};
pub use morphism_composition::{
    CompositionResult, compose, verify_associativity, MorphismDiagram,
    square_commutes, triangle_commutes, MorphismPath, CompositionTable,
};
pub use ring_category::{
    RingCategory, CommutativeRingCategory, IntegralDomainCategory,
    RingElementMethods, RingParentMethods, CommutativeRingParentMethods,
    RingWithBasis,
};
