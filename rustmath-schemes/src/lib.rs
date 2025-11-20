//! Schemes in Algebraic Geometry
//!
//! This crate provides comprehensive support for schemes in algebraic geometry,
//! the fundamental objects of modern algebraic geometry.
//!
//! # Overview
//!
//! A scheme is a topological space together with a sheaf of rings that generalizes
//! classical algebraic varieties. This crate implements:
//!
//! ## Generic Scheme Infrastructure
//!
//! - **Scheme Trait**: Core abstraction for all schemes
//! - **Morphisms**: Structure-preserving maps between schemes
//! - **Points**: Geometric and scheme-theoretic points
//! - **Dimension Theory**: Krull dimension and related invariants
//!
//! ## Affine Schemes
//!
//! - **Spec Construction**: Spec(R) for commutative rings R
//! - **Affine Space**: 𝔸ⁿ as Spec(k[x₁, ..., xₙ])
//! - **Closed Subschemes**: Varieties defined by ideals
//! - **Distinguished Opens**: D(f) basic open sets
//!
//! ## Projective Schemes
//!
//! - **Graded Rings**: Foundation for Proj construction
//! - **Proj Construction**: Building schemes from graded rings
//! - **Projective Spaces**: ℙⁿ with homogeneous coordinates
//! - **Veronese Embeddings**: νₐ: ℙⁿ → ℙᴺ via degree d monomials
//! - **Segre Embeddings**: ℙⁿ × ℙᵐ → ℙᴺ for products
//! - **Line Bundles**: Locally free sheaves of rank 1
//! - **Divisors and Picard Group**: Linear equivalence classes
//!
//! ## Elliptic Curves
//!
//! - **Weierstrass Models**: Standard and short forms
//! - **Group Law**: Abelian group structure on points
//! - **Isogenies**: Morphisms between elliptic curves
//! - **Torsion Points**: Points of finite order
//! - **Invariants**: j-invariant and discriminant
//!
//! # Key Concepts
//!
//! ## Schemes
//!
//! A scheme generalizes the notion of an algebraic variety. Every scheme is built from
//! affine pieces (affine schemes Spec(R)) glued together. The two fundamental examples are:
//!
//! 1. **Affine Schemes**: Spec(R) for a commutative ring R
//! 2. **Projective Schemes**: Proj(S) for a graded ring S
//!
//! ## Affine vs Projective
//!
//! - **Affine schemes** model "unbounded" geometric objects (e.g., affine space 𝔸ⁿ)
//! - **Projective schemes** are "compact" and include "points at infinity" (e.g., ℙⁿ)
//!
//! ## The Scheme Hierarchy
//!
//! All schemes in this crate implement the `Scheme` trait from the `generic` module,
//! which provides common operations like dimension computation and property checking.
//!
//! # Examples
//!
//! ## Working with Affine Schemes
//!
//! ```rust
//! use rustmath_schemes::affine::{AffineSpace, AffinePoint};
//! use rustmath_schemes::generic::Scheme;
//!
//! // Create 2-dimensional affine space 𝔸²
//! // let a2 = AffineSpace::new(2, base_ring);
//! // assert!(a2.is_affine());
//! // assert_eq!(a2.dimension(), Some(2));
//! ```
//!
//! ## Working with Projective Schemes
//!
//! ```
//! use rustmath_schemes::projective::{ProjectiveSpace, ProjectivePoint};
//!
//! // Create ℙ² (projective plane)
//! let p2: ProjectiveSpace<i32> = ProjectiveSpace::new(2);
//! assert_eq!(p2.dimension(), 2);
//! assert_eq!(p2.num_coordinates(), 3);
//!
//! // Create a point [1 : 2 : 3] in ℙ²
//! let point = ProjectivePoint::new(vec![1, 2, 3]).unwrap();
//! assert!(p2.contains_point(&point));
//! ```
//!
//! ## Proj Construction
//!
//! For a graded ring R = ⊕ Rₙ, the scheme Proj(R) is the set of homogeneous prime ideals
//! not containing the irrelevant ideal. The fundamental example is:
//!
//! Proj(k[x₀, x₁, ..., xₙ]) = ℙⁿ
//!
//! ```
//! use rustmath_schemes::proj;
//!
//! // Create ℙ² as Proj(k[x,y,z])
//! let p2 = proj::projective_space::<i32>(2);
//! assert_eq!(p2.dimension(), Some(2));
//! assert!(p2.is_projective_space());
//! ```
//!
//! ## Veronese Embedding
//!
//! The d-th Veronese embedding νₐ: ℙⁿ → ℙᴺ maps a point to all degree d monomials:
//!
//! ```
//! use rustmath_schemes::veronese::VeroneseEmbedding;
//! use rustmath_schemes::projective_space::ProjectivePoint;
//!
//! // Twisted cubic: ν₃: ℙ¹ → ℙ³
//! let twisted_cubic = VeroneseEmbedding::<i32>::twisted_cubic();
//! assert_eq!(twisted_cubic.source().dimension(), 1);
//! assert_eq!(twisted_cubic.target().dimension(), 3);
//!
//! // Apply to [2:1] → [8:4:2:1]
//! let point = ProjectivePoint::new(vec![2, 1]).unwrap();
//! let image = twisted_cubic.apply(&point).unwrap();
//! assert_eq!(image.coordinates(), &[8, 4, 2, 1]);
//! ```
//!
//! ## Segre Embedding
//!
//! The Segre embedding embeds products of projective spaces:
//!
//! ```
//! use rustmath_schemes::segre::SegreEmbedding;
//! use rustmath_schemes::projective_space::ProjectivePoint;
//!
//! // σ: ℙ¹ × ℙ¹ → ℙ³
//! let segre = SegreEmbedding::<i32>::p1_times_p1();
//!
//! let p1 = ProjectivePoint::new(vec![1, 2]).unwrap();
//! let p2 = ProjectivePoint::new(vec![3, 4]).unwrap();
//! let image = segre.apply(&p1, &p2).unwrap();
//! assert_eq!(image.coordinates(), &[3, 4, 6, 8]);
//! ```
//!
//! ## Line Bundles
//!
//! Line bundles are locally free sheaves of rank 1. On ℙⁿ, the twisting sheaves 𝒪(d)
//! are the fundamental line bundles:
//!
//! ```
//! use rustmath_schemes::line_bundle::LineBundle;
//! use rustmath_schemes::proj;
//!
//! let p2 = proj::projective_space::<i32>(2);
//!
//! // Create 𝒪(3) on ℙ²
//! let o3 = LineBundle::twisting_sheaf(p2, 3);
//! assert_eq!(o3.degree(), 3);
//! assert!(o3.is_ample()); // 𝒪(d) is ample for d > 0
//! assert_eq!(o3.h0(), 10); // dim H⁰(ℙ², 𝒪(3)) = C(5,3) = 10
//! ```
//!
//! # Examples
//!
//! ## Working with Homogeneous Coordinates
//!
//! ```
//! use rustmath_schemes::projective_space::ProjectivePoint;
//!
//! let point = ProjectivePoint::new(vec![2, 4, 6]).unwrap();
//!
//! // Convert to affine coordinates on chart U₀ (x₀ ≠ 0)
//! let affine = point.to_affine(0).unwrap();
//! assert_eq!(affine, vec![4, 6]); // (x₁, x₂) in affine coordinates
//!
//! // Convert back
//! let back = ProjectivePoint::from_affine(affine, 0).unwrap();
//! assert_eq!(back.coordinates(), &[1, 4, 6]); // Normalized
//! ```
//!
//! ## Veronese Surface
//!
//! ```
//! use rustmath_schemes::veronese::{VeroneseEmbedding, VeroneseVariety};
//!
//! // ν₂: ℙ² → ℙ⁵
//! let veronese_surface = VeroneseEmbedding::<i32>::veronese_surface();
//! let variety = VeroneseVariety::new(veronese_surface);
//!
//! assert_eq!(variety.dimension(), 2); // 2-dimensional surface
//! assert_eq!(variety.ambient_space().dimension(), 5); // in ℙ⁵
//! ```
//!
//! ## Canonical Bundle
//!
//! ```
//! use rustmath_schemes::line_bundle::CanonicalBundle;
//!
//! // K_{ℙ²} = 𝒪(-3)
//! let k_p2 = CanonicalBundle::<i32>::of_projective_space(2);
//! assert_eq!(k_p2.line_bundle().degree(), -3);
//! assert!(k_p2.is_fano()); // ℙ² is Fano
//! ```

// Core scheme infrastructure
pub mod generic;
pub mod affine;
pub mod projective;
pub mod elliptic_curves;

// Projective-specific modules (organized under projective)
pub mod graded_ring;
pub mod line_bundle;
pub mod proj;
pub mod projective_morphism;
pub mod projective_space;
pub mod segre;
pub mod veronese;

// Re-export commonly used types from generic module
pub use generic::{
    Scheme, SchemeMorphism, SchemePoint, DimensionTheory,
    Separated, AlgebraicScheme, StructureSheaf, FiberedProduct
};

// Re-export affine scheme types
pub use affine::{
    AffineScheme, AffineSpace, AffinePoint, AffineSchemeMorphism,
    ClosedSubscheme, DistinguishedOpen
};

// Re-export projective scheme types
pub use graded_ring::{GradedRing, HomogeneousElement, HomogeneousIdeal};
pub use line_bundle::{CanonicalBundle, Divisor, LineBundle, PicardGroup};
pub use proj::{AffineChart, Proj, TwistingSheaf};
pub use projective_morphism::{ProjectiveMorphism, ProjMorphism};
pub use projective_space::{Hyperplane, LinearSubspace, ProjectivePoint, ProjectiveSpace};
pub use segre::{MultiSegreEmbedding, SegreEmbedding, SegreVariety};
pub use veronese::{VeroneseEmbedding, VeroneseVariety};

// Re-export elliptic curve types
pub use elliptic_curves::{
    EllipticCurve, EllipticCurvePoint, Isogeny, TorsionSubgroup
};
