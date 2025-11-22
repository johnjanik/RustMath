//! Generic Scheme Infrastructure
//!
//! This module provides the foundational traits and types for working with schemes
//! in algebraic geometry. It defines the abstract interface that both affine and
//! projective schemes implement.
//!
//! # Overview
//!
//! A scheme is a topological space together with a sheaf of rings. In modern algebraic
//! geometry, schemes generalize classical algebraic varieties and provide a unified
//! framework for studying geometric objects.
//!
//! This module implements:
//! - **Scheme Trait**: The core trait defining what it means to be a scheme
//! - **Morphisms**: Structure-preserving maps between schemes
//! - **Points**: Geometric and scheme-theoretic points
//! - **Sheaves**: Structure sheaves and quasi-coherent sheaves
//! - **Dimension**: Krull dimension and dimension computations
//!
//! # Key Concepts
//!
//! ## The Scheme Trait
//!
//! All schemes (affine, projective, general) implement the `Scheme` trait, which
//! provides common operations:
//!
//! - Base ring access
//! - Dimension computation
//! - Point membership testing
//! - Irreducibility and reducedness checks
//!
//! ## Morphisms
//!
//! Morphisms between schemes are continuous maps f: X → Y such that for every open
//! set U ⊆ Y, the pullback f*: O_Y(U) → O_X(f⁻¹(U)) is a ring homomorphism.
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_schemes::generic::{Scheme, SchemeMorphism};
//! use rustmath_schemes::affine::AffineScheme;
//! use rustmath_core::Ring;
//!
//! // Create an affine scheme Spec(R)
//! // let spec_r = AffineScheme::spec(ring);
//!
//! // Check properties
//! // assert!(spec_r.is_affine());
//! // println!("Dimension: {:?}", spec_r.dimension());
//! ```

use rustmath_core::{MathError, Result, Ring};
use std::fmt::Debug;

/// Core trait for all schemes
///
/// A scheme is a ringed space that is locally isomorphic to the spectrum of a ring.
/// This trait provides the essential operations that all schemes must support.
///
/// # Type Parameters
///
/// - `R`: The base ring over which the scheme is defined
pub trait Scheme: Debug + Clone {
    /// The base ring type
    type BaseRing: Ring;

    /// Get the base ring over which this scheme is defined
    fn base_ring(&self) -> &Self::BaseRing;

    /// Compute the Krull dimension of the scheme
    ///
    /// Returns `None` if the dimension is infinite or cannot be computed.
    fn dimension(&self) -> Option<usize>;

    /// Check if this is an affine scheme
    fn is_affine(&self) -> bool {
        false
    }

    /// Check if this is a projective scheme
    fn is_projective(&self) -> bool {
        false
    }

    /// Check if the scheme is irreducible
    ///
    /// A scheme is irreducible if its underlying topological space cannot be
    /// written as the union of two proper closed subsets.
    fn is_irreducible(&self) -> bool;

    /// Check if the scheme is reduced
    ///
    /// A scheme is reduced if its structure sheaf has no nilpotent elements.
    fn is_reduced(&self) -> bool;

    /// Check if the scheme is integral
    ///
    /// A scheme is integral if it is both irreducible and reduced.
    fn is_integral(&self) -> bool {
        self.is_irreducible() && self.is_reduced()
    }

    /// Check if the scheme is noetherian
    fn is_noetherian(&self) -> bool;

    /// Check if the scheme is of finite type over the base
    fn is_finite_type(&self) -> bool;
}

/// A morphism between schemes
///
/// Represents a structure-preserving map f: X → Y between two schemes.
///
/// # Type Parameters
///
/// - `S`: The source scheme type
/// - `T`: The target scheme type
pub trait SchemeMorphism: Debug + Clone {
    /// The source scheme type
    type Source: Scheme;
    /// The target scheme type
    type Target: Scheme;

    /// Get the source scheme
    fn source(&self) -> &Self::Source;

    /// Get the target scheme
    fn target(&self) -> &Self::Target;

    /// Check if the morphism is proper
    ///
    /// A morphism is proper if it is separated, of finite type, and universally closed.
    fn is_proper(&self) -> bool;

    /// Check if the morphism is finite
    ///
    /// A morphism is finite if it is affine and the pushforward of the structure
    /// sheaf is a finite module.
    fn is_finite(&self) -> bool;

    /// Check if the morphism is of finite type
    fn is_finite_type(&self) -> bool;

    /// Check if the morphism is a closed embedding
    fn is_closed_embedding(&self) -> bool;

    /// Check if the morphism is an open embedding
    fn is_open_embedding(&self) -> bool;

    /// Check if the morphism is an isomorphism
    fn is_isomorphism(&self) -> bool {
        false // Conservative default
    }
}

/// A point on a scheme
///
/// Represents a geometric or scheme-theoretic point. In the scheme-theoretic
/// sense, points correspond to prime ideals of the coordinate ring.
pub trait SchemePoint: Debug + Clone + PartialEq {
    /// The scheme this point belongs to
    type Parent: Scheme;

    /// Get the parent scheme
    fn parent(&self) -> &Self::Parent;

    /// Check if this point is closed
    ///
    /// A point is closed if its closure is just itself.
    fn is_closed(&self) -> bool;

    // TODO: Cannot use Box<dyn Ring> as Ring is not dyn compatible
    // Consider using an associated type or generic parameter
    // fn residue_field(&self) -> Result<Box<dyn Ring>>;
}

/// Dimension theory for schemes
pub trait DimensionTheory: Scheme {
    /// Compute the dimension at a specific point
    fn dimension_at_point<P: SchemePoint<Parent = Self>>(&self, point: &P) -> Option<usize>;

    /// Compute the Krull dimension using chain of prime ideals
    fn krull_dimension(&self) -> Option<usize> {
        self.dimension()
    }

    /// Check if the scheme is equidimensional
    ///
    /// A scheme is equidimensional if all irreducible components have the same dimension.
    fn is_equidimensional(&self) -> bool;
}

/// Sheaf of rings on a scheme
///
/// The structure sheaf O_X assigns to each open set U a ring O_X(U) of
/// "regular functions" on U.
pub trait StructureSheaf: Debug {
    /// The base ring type
    type Ring: Ring;
    /// The open set type
    type OpenSet: Debug;

    /// Evaluate the sheaf on an open set
    fn sections(&self, open_set: &Self::OpenSet) -> Result<Self::Ring>;

    /// Restriction map: O_X(U) → O_X(V) for V ⊆ U
    fn restriction(
        &self,
        from: &Self::OpenSet,
        to: &Self::OpenSet,
    ) -> Result<Box<dyn Fn(&Self::Ring) -> Result<Self::Ring>>>;
}

/// Properties of schemes related to separatedness
pub trait Separated: Scheme {
    /// Check if the scheme is separated
    ///
    /// A scheme X is separated if the diagonal morphism Δ: X → X × X is a closed embedding.
    fn is_separated(&self) -> bool;

    /// Check if the scheme is quasi-separated
    fn is_quasi_separated(&self) -> bool {
        true // All schemes we work with are quasi-separated
    }
}

/// Fibered products (fiber products) of schemes
///
/// Given morphisms f: X → S and g: Y → S, the fibered product X ×_S Y
/// represents pairs of points (x, y) where f(x) = g(y).
pub struct FiberedProduct<X, Y, S>
where
    X: Scheme,
    Y: Scheme,
    S: Scheme,
{
    pub x: X,
    pub y: Y,
    pub base: S,
}

impl<X, Y, S> FiberedProduct<X, Y, S>
where
    X: Scheme,
    Y: Scheme,
    S: Scheme,
{
    /// Create a new fibered product
    pub fn new(x: X, y: Y, base: S) -> Self {
        FiberedProduct { x, y, base }
    }
}

/// Algebraic properties of schemes
pub trait AlgebraicScheme: Scheme {
    /// Check if the scheme is smooth
    ///
    /// A scheme is smooth if it is regular and of finite type over a field.
    fn is_smooth(&self) -> bool {
        false // Conservative default
    }

    /// Check if the scheme is regular
    ///
    /// A scheme is regular if all local rings are regular local rings.
    fn is_regular(&self) -> bool {
        false // Conservative default
    }

    /// Check if the scheme is normal
    ///
    /// A scheme is normal if all local rings are integrally closed.
    fn is_normal(&self) -> bool {
        false // Conservative default
    }

    /// Compute the genus (for curves)
    fn genus(&self) -> Option<usize> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheme_properties() {
        // Basic compile test - actual implementations will be in affine/projective modules
        // This ensures the trait definitions are sound
    }

    #[test]
    fn test_fibered_product() {
        // Fibered products will be tested with concrete scheme implementations
    }
}
