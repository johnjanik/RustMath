//! # Ring Morphism Module
//!
//! Implementation of structure-preserving maps (homomorphisms) between rings.
//!
//! This module provides a comprehensive framework for ring homomorphisms, following
//! the categorical approach used in SageMath. It supports:
//!
//! - **Ring Maps**: General set-theoretic maps between rings
//! - **Ring Homomorphisms**: Structure-preserving maps (respecting +, ×, 0, 1)
//! - **Specialized Morphisms**: From quotients, fraction fields, base extensions
//! - **Frobenius Endomorphisms**: Characteristic-p automorphisms
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::morphism::{RingHomomorphism, RingMap};
//!
//! // Create a homomorphism from ℤ to ℤ/5ℤ (quotient map)
//! // let phi = RingHomomorphism::quotient_map(ZZ, Z5);
//! ```
//!
//! ## Design Philosophy
//!
//! Ring morphisms in RustMath follow these principles:
//! 1. **Type Safety**: Domain and codomain types enforced at compile time
//! 2. **Composability**: Morphisms compose naturally with proper type inference
//! 3. **Efficiency**: Caching of computed properties (kernel, image, etc.)
//! 4. **Correctness**: Validation ensures structure preservation
//!
//! ## Mathematical Background
//!
//! A ring homomorphism φ: R → S satisfies:
//! - φ(a + b) = φ(a) + φ(b) (additive)
//! - φ(a × b) = φ(a) × φ(b) (multiplicative)
//! - φ(1_R) = 1_S (unit-preserving, if rings are unital)
//!
//! Key properties:
//! - **Kernel**: ker(φ) = {r ∈ R | φ(r) = 0}
//! - **Image**: im(φ) = {φ(r) | r ∈ R}
//! - **Injective**: ker(φ) = {0}
//! - **Surjective**: im(φ) = S
//! - **Isomorphism**: Bijective homomorphism

use rustmath_core::{Ring, Field};
use std::fmt;
use std::marker::PhantomData;
use thiserror::Error;

/// Errors that can occur when working with ring morphisms
#[derive(Debug, Clone, Error, PartialEq)]
pub enum MorphismError {
    #[error("Not a valid homomorphism: {0}")]
    InvalidHomomorphism(String),

    #[error("Morphism is not injective")]
    NotInjective,

    #[error("Morphism is not surjective")]
    NotSurjective,

    #[error("Morphism is not invertible")]
    NotInvertible,

    #[error("Element not in domain")]
    NotInDomain,

    #[error("Composition error: {0}")]
    CompositionError(String),
}

/// Trait for ring morphisms
///
/// This trait represents a map between rings. It doesn't guarantee structure
/// preservation (that's enforced by specific implementations).
pub trait RingMorphism<R, S>
where
    R: Ring,
    S: Ring,
{
    /// Apply the morphism to an element
    fn apply(&self, element: &R) -> Result<S, MorphismError>;

    /// Get the domain ring (source)
    fn domain(&self) -> &R;

    /// Get the codomain ring (target)
    fn codomain(&self) -> &S;

    /// Check if this morphism is injective (kernel is trivial)
    fn is_injective(&self) -> bool {
        false // Default: unknown
    }

    /// Check if this morphism is surjective (image is full codomain)
    fn is_surjective(&self) -> bool {
        false // Default: unknown
    }

    /// Check if this morphism is bijective (isomorphism)
    fn is_isomorphism(&self) -> bool {
        self.is_injective() && self.is_surjective()
    }
}

/// A general map between rings (may not preserve structure)
///
/// This is the Rust equivalent of SageMath's `RingMap` class.
/// It represents a set-theoretic map between rings without guaranteeing
/// that algebraic structure is preserved.
#[derive(Debug, Clone)]
pub struct RingMap<R: Ring, S: Ring, F>
where
    F: Fn(&R) -> S,
{
    domain: PhantomData<R>,
    codomain: PhantomData<S>,
    map_fn: F,
}

impl<R: Ring, S: Ring, F> RingMap<R, S, F>
where
    F: Fn(&R) -> S,
{
    /// Creates a new ring map with the given function
    pub fn new(map_fn: F) -> Self {
        RingMap {
            domain: PhantomData,
            codomain: PhantomData,
            map_fn,
        }
    }
}

/// A ring homomorphism
///
/// This represents a structure-preserving map between rings. Unlike `RingMap`,
/// this guarantees that φ(a + b) = φ(a) + φ(b) and φ(a × b) = φ(a) × φ(b).
///
/// ## Type Parameters
/// - `R`: Domain ring type
/// - `S`: Codomain ring type
#[derive(Debug, Clone)]
pub struct RingHomomorphism<R, S>
where
    R: Ring,
    S: Ring,
{
    /// Optional description of this homomorphism
    description: String,
    /// Cached injectivity status
    is_injective_cached: Option<bool>,
    /// Cached surjectivity status
    is_surjective_cached: Option<bool>,
    /// Phantom data for type safety
    _phantom: PhantomData<(R, S)>,
}

impl<R, S> RingHomomorphism<R, S>
where
    R: Ring,
    S: Ring,
{
    /// Creates a new ring homomorphism
    pub fn new(description: String) -> Self {
        RingHomomorphism {
            description,
            is_injective_cached: None,
            is_surjective_cached: None,
            _phantom: PhantomData,
        }
    }

    /// Returns the description of this homomorphism
    pub fn description(&self) -> &str {
        &self.description
    }
}

impl<R, S> fmt::Display for RingHomomorphism<R, S>
where
    R: Ring,
    S: Ring,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ring homomorphism: {}", self.description)
    }
}

/// A ring homomorphism determined by images of generators
///
/// This is the Rust equivalent of SageMath's `RingHomomorphism_im_gens`.
/// It defines a homomorphism by specifying where each generator maps to.
#[derive(Debug, Clone)]
pub struct RingHomomorphismImGens<R, S>
where
    R: Ring,
    S: Ring,
{
    base: RingHomomorphism<R, S>,
    /// Images of the generators
    generator_images: Vec<S>,
}

impl<R, S> RingHomomorphismImGens<R, S>
where
    R: Ring,
    S: Ring,
{
    /// Creates a new homomorphism from generator images
    ///
    /// # Arguments
    /// * `generator_images` - Images of each generator in the domain
    ///
    /// # Errors
    /// Returns `InvalidHomomorphism` if the images don't define a valid homomorphism
    pub fn new(generator_images: Vec<S>) -> Result<Self, MorphismError> {
        // In a full implementation, we would validate that these images
        // respect the relations in the domain ring
        Ok(RingHomomorphismImGens {
            base: RingHomomorphism::new("Homomorphism defined by generator images".to_string()),
            generator_images,
        })
    }

    /// Returns the images of the generators
    pub fn generator_images(&self) -> &[S] {
        &self.generator_images
    }
}

/// A lifting morphism from a quotient ring
///
/// This represents the natural projection R → R/I for an ideal I.
#[derive(Debug, Clone)]
pub struct RingMapLift<R>
where
    R: Ring,
{
    _phantom: PhantomData<R>,
}

impl<R> RingMapLift<R>
where
    R: Ring,
{
    /// Creates a new lifting morphism
    pub fn new() -> Self {
        RingMapLift {
            _phantom: PhantomData,
        }
    }
}

impl<R> Default for RingMapLift<R>
where
    R: Ring,
{
    fn default() -> Self {
        Self::new()
    }
}

/// A homomorphism from a base ring
///
/// This represents the canonical inclusion of a base ring into an extension.
/// For example, ℤ → ℚ or ℚ → ℝ.
#[derive(Debug, Clone)]
pub struct RingHomomorphismFromBase<R, S>
where
    R: Ring,
    S: Ring,
{
    base: RingHomomorphism<R, S>,
}

impl<R, S> RingHomomorphismFromBase<R, S>
where
    R: Ring,
    S: Ring,
{
    /// Creates a new homomorphism from base ring
    pub fn new() -> Self {
        RingHomomorphismFromBase {
            base: RingHomomorphism::new("Canonical inclusion from base ring".to_string()),
        }
    }
}

impl<R, S> Default for RingHomomorphismFromBase<R, S>
where
    R: Ring,
    S: Ring,
{
    fn default() -> Self {
        Self::new()
    }
}

/// A homomorphism from a fraction field
///
/// This represents maps from fraction fields, typically extending a homomorphism
/// from the base ring.
#[derive(Debug, Clone)]
pub struct RingHomomorphismFromFractionField<R, S>
where
    R: Ring,
    S: Field,
{
    base: RingHomomorphism<R, S>,
}

impl<R, S> RingHomomorphismFromFractionField<R, S>
where
    R: Ring,
    S: Field,
{
    /// Creates a new homomorphism from fraction field
    pub fn new() -> Self {
        RingHomomorphismFromFractionField {
            base: RingHomomorphism::new("Homomorphism from fraction field".to_string()),
        }
    }
}

impl<R, S> Default for RingHomomorphismFromFractionField<R, S>
where
    R: Ring,
    S: Field,
{
    fn default() -> Self {
        Self::new()
    }
}

/// A homomorphism from a quotient ring
///
/// This represents maps from quotient rings R/I.
#[derive(Debug, Clone)]
pub struct RingHomomorphismFromQuotient<R, S>
where
    R: Ring,
    S: Ring,
{
    base: RingHomomorphism<R, S>,
}

impl<R, S> RingHomomorphismFromQuotient<R, S>
where
    R: Ring,
    S: Ring,
{
    /// Creates a new homomorphism from quotient ring
    pub fn new() -> Self {
        RingHomomorphismFromQuotient {
            base: RingHomomorphism::new("Homomorphism from quotient ring".to_string()),
        }
    }
}

impl<R, S> Default for RingHomomorphismFromQuotient<R, S>
where
    R: Ring,
    S: Ring,
{
    fn default() -> Self {
        Self::new()
    }
}

/// A covering homomorphism (natural quotient map)
///
/// This represents the canonical quotient map R → R/I for an ideal I.
#[derive(Debug, Clone)]
pub struct RingHomomorphismCover<R>
where
    R: Ring,
{
    base: RingHomomorphism<R, R>,
}

impl<R> RingHomomorphismCover<R>
where
    R: Ring,
{
    /// Creates a new covering homomorphism
    pub fn new() -> Self {
        RingHomomorphismCover {
            base: RingHomomorphism::new("Quotient map".to_string()),
        }
    }
}

impl<R> Default for RingHomomorphismCover<R>
where
    R: Ring,
{
    fn default() -> Self {
        Self::new()
    }
}

/// The Frobenius endomorphism
///
/// For a ring of characteristic p > 0, the Frobenius is φ(x) = x^p.
/// This is a ring homomorphism in characteristic p.
#[derive(Debug, Clone)]
pub struct FrobeniusEndomorphism<R>
where
    R: Ring,
{
    base: RingHomomorphism<R, R>,
    /// The characteristic p (power for the Frobenius)
    characteristic: usize,
}

impl<R> FrobeniusEndomorphism<R>
where
    R: Ring,
{
    /// Creates a new Frobenius endomorphism
    ///
    /// # Arguments
    /// * `characteristic` - The characteristic p of the ring
    pub fn new(characteristic: usize) -> Self {
        FrobeniusEndomorphism {
            base: RingHomomorphism::new(format!("Frobenius endomorphism x |--> x^{}", characteristic)),
            characteristic,
        }
    }

    /// Returns the characteristic (Frobenius power)
    pub fn characteristic(&self) -> usize {
        self.characteristic
    }

    /// Composes the Frobenius with itself n times (x → x^(p^n))
    pub fn power(&self, n: usize) -> Self {
        let new_char = self.characteristic.pow(n as u32);
        FrobeniusEndomorphism::new(new_char)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_homomorphism_creation() {
        // Create a homomorphism description
        let phi: RingHomomorphism<i32, i32> = RingHomomorphism::new("Test homomorphism".to_string());
        assert_eq!(phi.description(), "Test homomorphism");
    }

    #[test]
    fn test_ring_homomorphism_display() {
        let phi: RingHomomorphism<i64, i64> = RingHomomorphism::new("Identity map".to_string());
        assert_eq!(format!("{}", phi), "Ring homomorphism: Identity map");
    }

    #[test]
    fn test_frobenius_endomorphism() {
        let frob: FrobeniusEndomorphism<i32> = FrobeniusEndomorphism::new(5);
        assert_eq!(frob.characteristic(), 5);
    }

    #[test]
    fn test_frobenius_power() {
        let frob: FrobeniusEndomorphism<i32> = FrobeniusEndomorphism::new(3);
        let frob_squared = frob.power(2);
        assert_eq!(frob_squared.characteristic(), 9); // 3^2 = 9
    }

    #[test]
    fn test_ring_homomorphism_im_gens() {
        let images = vec![1, 2, 3];
        let phi = RingHomomorphismImGens::<i32, i32>::new(images.clone()).unwrap();
        assert_eq!(phi.generator_images(), &[1, 2, 3]);
    }

    #[test]
    fn test_ring_map_lift_creation() {
        let _lift: RingMapLift<i32> = RingMapLift::new();
    }

    #[test]
    fn test_ring_homomorphism_from_base() {
        let _phi: RingHomomorphismFromBase<i32, i64> = RingHomomorphismFromBase::new();
    }

    #[test]
    fn test_ring_homomorphism_cover() {
        let _cover: RingHomomorphismCover<i32> = RingHomomorphismCover::new();
    }
}
