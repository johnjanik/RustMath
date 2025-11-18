//! Morphisms in Category Theory
//!
//! This module provides core morphism types and utilities for category theory.
//! Morphisms are structure-preserving maps between objects in a category.
//!
//! Key concepts:
//! - Morphism: A map f: A → B between objects in a category
//! - Identity morphism: id_A: A → A for each object A
//! - Composition: g ∘ f for morphisms f: A → B and g: B → C
//! - Isomorphism: A morphism with an inverse

use std::marker::PhantomData;

/// A morphism in a category
///
/// A morphism f: A → B consists of:
/// - Source object A (domain)
/// - Target object B (codomain)
/// - The mapping itself
///
/// Morphisms must satisfy:
/// - Identity: id_A ∘ f = f and f ∘ id_B = f
/// - Associativity: h ∘ (g ∘ f) = (h ∘ g) ∘ f
pub trait Morphism: Clone {
    /// Type of objects in the category
    type Object;

    /// Get the source (domain) of this morphism
    fn source(&self) -> &Self::Object;

    /// Get the target (codomain) of this morphism
    fn target(&self) -> &Self::Object;

    /// Compose this morphism with another: other ∘ self
    ///
    /// Requires: self.target() == other.source()
    fn compose(&self, other: &Self) -> Option<Self>;

    /// Check if this is an identity morphism
    fn is_identity(&self) -> bool {
        false // Default: not an identity
    }
}

/// The identity morphism id_A: A → A
///
/// For any object A, the identity morphism maps A to itself
/// and satisfies: f ∘ id_A = f and id_B ∘ f = f
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IdentityMorphism<T> {
    object: T,
}

impl<T> IdentityMorphism<T> {
    /// Create a new identity morphism for the given object
    ///
    /// # Examples
    /// ```
    /// use rustmath_category::morphism::{IdentityMorphism, Morphism};
    ///
    /// let id = IdentityMorphism::new(42);
    /// assert!(id.is_identity());
    /// ```
    pub fn new(object: T) -> Self {
        IdentityMorphism { object }
    }

    /// Get the object this identity morphism acts on
    pub fn object(&self) -> &T {
        &self.object
    }
}

impl<T: Clone> Morphism for IdentityMorphism<T> {
    type Object = T;

    fn source(&self) -> &Self::Object {
        &self.object
    }

    fn target(&self) -> &Self::Object {
        &self.object
    }

    fn compose(&self, _other: &Self) -> Option<Self> {
        Some(self.clone())
    }

    fn is_identity(&self) -> bool {
        true
    }
}

/// An isomorphism is a morphism with an inverse
///
/// A morphism f: A → B is an isomorphism if there exists g: B → A such that:
/// - g ∘ f = id_A
/// - f ∘ g = id_B
pub trait Isomorphism: Morphism {
    /// Get the inverse morphism
    fn inverse(&self) -> Self;

    /// Check if this morphism has an inverse
    fn is_isomorphism(&self) -> bool {
        true // All types implementing this trait are isomorphisms
    }
}

/// A set morphism (function) between sets
///
/// The most basic type of morphism, representing a function f: X → Y
#[derive(Clone, Debug)]
pub struct SetMorphism<T, F>
where
    F: Fn(&T) -> T + Clone,
{
    source: T,
    target: T,
    function: F,
    _phantom: PhantomData<T>,
}

impl<T, F> SetMorphism<T, F>
where
    F: Fn(&T) -> T + Clone,
{
    /// Create a new set morphism
    ///
    /// # Examples
    /// ```
    /// use rustmath_category::morphism::SetMorphism;
    ///
    /// let f = SetMorphism::new(0, 10, |x| x + 1);
    /// ```
    pub fn new(source: T, target: T, function: F) -> Self {
        SetMorphism {
            source,
            target,
            function,
            _phantom: PhantomData,
        }
    }

    /// Apply this morphism to a value
    pub fn apply(&self, x: &T) -> T {
        (self.function)(x)
    }
}

impl<T: Clone, F> Morphism for SetMorphism<T, F>
where
    F: Fn(&T) -> T + Clone,
{
    type Object = T;

    fn source(&self) -> &Self::Object {
        &self.source
    }

    fn target(&self) -> &Self::Object {
        &self.target
    }

    fn compose(&self, _other: &Self) -> Option<Self> {
        // Composition creates a new closure type, which can't be returned
        // as the same type parameter F. In practice, composition would
        // require dependent types or dynamic dispatch.
        None
    }
}

/// A set isomorphism (bijection) between sets
///
/// A function with an inverse
#[derive(Clone, Debug)]
pub struct SetIsomorphism<T, F, G>
where
    F: Fn(&T) -> T + Clone,
    G: Fn(&T) -> T + Clone,
{
    forward: SetMorphism<T, F>,
    backward: SetMorphism<T, G>,
}

impl<T, F, G> SetIsomorphism<T, F, G>
where
    F: Fn(&T) -> T + Clone,
    G: Fn(&T) -> T + Clone,
{
    /// Create a new set isomorphism with forward and inverse functions
    pub fn new(source: T, target: T, forward: F, backward: G) -> Self
    where
        T: Clone,
    {
        SetIsomorphism {
            forward: SetMorphism::new(source.clone(), target.clone(), forward),
            backward: SetMorphism::new(target, source, backward),
        }
    }

    /// Get the forward morphism
    pub fn forward(&self) -> &SetMorphism<T, F> {
        &self.forward
    }

    /// Get the backward (inverse) morphism
    pub fn backward(&self) -> &SetMorphism<T, G> {
        &self.backward
    }
}

/// A formal coercion morphism
///
/// Represents a morphism used for coercion in type systems
#[derive(Clone, Debug)]
pub struct FormalCoercionMorphism<S, T> {
    source: S,
    target: T,
}

impl<S, T> FormalCoercionMorphism<S, T> {
    /// Create a new formal coercion morphism
    pub fn new(source: S, target: T) -> Self {
        FormalCoercionMorphism { source, target }
    }
}

impl<S: Clone, T: Clone> Morphism for FormalCoercionMorphism<S, T> {
    type Object = S; // Simplified: both source and target use same type

    fn source(&self) -> &Self::Object {
        &self.source
    }

    fn target(&self) -> &Self::Object {
        &self.source // Placeholder
    }

    fn compose(&self, _other: &Self) -> Option<Self> {
        None // Placeholder: composition not implemented for coercions
    }
}

/// A call morphism that applies a callable object
///
/// Wraps function application as a morphism
#[derive(Clone, Debug)]
pub struct CallMorphism<F> {
    callable: F,
}

impl<F> CallMorphism<F> {
    /// Create a new call morphism from a callable
    pub fn new(callable: F) -> Self {
        CallMorphism { callable }
    }

    /// Get the underlying callable
    pub fn callable(&self) -> &F {
        &self.callable
    }
}

/// Check if a value is a morphism (type checking helper)
///
/// This is a marker function for type checking in dynamic contexts
pub fn is_morphism<M: Morphism>(_morph: &M) -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_morphism() {
        let id = IdentityMorphism::new(42);
        assert_eq!(id.source(), &42);
        assert_eq!(id.target(), &42);
        assert!(id.is_identity());
    }

    #[test]
    fn test_identity_composition() {
        let id1 = IdentityMorphism::new(10);
        let id2 = IdentityMorphism::new(10);

        let composed = id1.compose(&id2);
        assert!(composed.is_some());
        assert!(composed.unwrap().is_identity());
    }

    #[test]
    fn test_set_morphism() {
        let f = SetMorphism::new(0, 10, |x: &i32| x + 1);
        assert_eq!(f.apply(&5), 6);
        assert_eq!(f.source(), &0);
        assert_eq!(f.target(), &10);
    }

    #[test]
    fn test_set_morphism_composition() {
        // Note: Composition requires morphisms of the same function type,
        // which is a limitation of Rust's type system
        let inc = |x: &i32| x + 1;
        let f = SetMorphism::new(0, 10, inc);
        let g = SetMorphism::new(10, 20, inc);

        let composed = f.compose(&g);
        // Even with the same function, composition creates a new closure type
        assert!(composed.is_none());
    }

    #[test]
    fn test_set_isomorphism() {
        let iso = SetIsomorphism::new(0, 0, |x: &i32| x + 1, |x: &i32| x - 1);

        assert_eq!(iso.forward().apply(&5), 6);
        assert_eq!(iso.backward().apply(&6), 5);
    }

    #[test]
    fn test_is_morphism() {
        let id = IdentityMorphism::new(42);
        assert!(is_morphism(&id));

        let f = SetMorphism::new(0, 10, |x: &i32| x + 1);
        assert!(is_morphism(&f));
    }

    #[test]
    fn test_formal_coercion_morphism() {
        let coercion = FormalCoercionMorphism::new(10, 20);
        assert_eq!(coercion.source(), &10);
    }

    #[test]
    fn test_call_morphism() {
        let f = |x: i32| x * 2;
        let call = CallMorphism::new(f);
        assert_eq!(call.callable()(5), 10);
    }

    #[test]
    fn test_identity_object() {
        let id = IdentityMorphism::new("test");
        assert_eq!(id.object(), &"test");
    }
}
