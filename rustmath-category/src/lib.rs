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
//!
//! # Relationship to `rustmath_core::coercion` (decision record, P2-G)
//!
//! This crate is **wired into** core's coercion layer, in the category → core
//! direction only, via the [`core_bridge`] module: the runtime coercion graph
//! ([`CoercionMap`]) drives core's statically-typed
//! [`Pushout`](rustmath_core::coercion::Pushout) /
//! [`Coercible`](rustmath_core::coercion::Coercible) resolution
//! (canonical example: `pushout(Z, Q) = Q`). `rustmath-core` has **no**
//! dependency on this crate — the bridge is additive and one-directional, and
//! type erasure goes through core's object-safe `morphism` module rather than
//! any `dyn Ring` (which would be ill-formed: `Ring` is not dyn-safe). See
//! [`core_bridge`] for the full decision record.

pub mod axioms;
pub mod category;
pub mod coercion;
pub mod core_bridge;
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
pub use core_bridge::{
    coerce_pair_via_graph, coercion_as_morphism, graph_pushout, register_parent_coercion,
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
