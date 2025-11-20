//! Projective Schemes
//!
//! This module organizes all projective scheme functionality, including:
//! - Graded rings and the Proj construction
//! - Projective spaces and points
//! - Projective morphisms
//! - Embeddings (Veronese and Segre)
//! - Line bundles and divisors
//!
//! # Overview
//!
//! Projective schemes are central objects in algebraic geometry. Unlike affine schemes,
//! they are compact (in the appropriate sense) and provide natural settings for studying
//! curves, surfaces, and higher-dimensional varieties.
//!
//! ## Key Components
//!
//! ### Projective Space ℙⁿ
//!
//! The fundamental projective scheme, representing lines through the origin in (n+1)-space.
//! Implemented in [`projective_space`].
//!
//! ### Proj Construction
//!
//! For a graded ring S = ⊕ Sₙ, the scheme Proj(S) generalizes projective space.
//! Implemented in [`proj`].
//!
//! ### Embeddings
//!
//! - **Veronese**: Maps ℙⁿ → ℙᴺ via degree d monomials ([`veronese`])
//! - **Segre**: Embeds products ℙⁿ × ℙᵐ → ℙᴺ ([`segre`])
//!
//! ### Line Bundles
//!
//! Locally free sheaves of rank 1, essential for studying divisors and embeddings.
//! Implemented in [`line_bundle`].
//!
//! # Examples
//!
//! ```rust
//! use rustmath_schemes::projective::{ProjectiveSpace, ProjectivePoint};
//!
//! // Create projective plane ℙ²
//! let p2: ProjectiveSpace<i32> = ProjectiveSpace::new(2);
//!
//! // Create a point [1 : 2 : 3]
//! let point = ProjectivePoint::new(vec![1, 2, 3]).unwrap();
//! assert!(p2.contains_point(&point));
//! ```
//!
//! # Module Organization
//!
//! This module serves as the organizational hub for projective geometry, re-exporting
//! types and functions from the specialized submodules. For detailed documentation,
//! see the individual module pages.

// Re-export projective-related modules from the crate root
pub use crate::graded_ring;
pub use crate::line_bundle;
pub use crate::proj;
pub use crate::projective_morphism;
pub use crate::projective_space;
pub use crate::segre;
pub use crate::veronese;

// Re-export commonly used types for convenience
pub use crate::graded_ring::{GradedRing, HomogeneousElement, HomogeneousIdeal};
pub use crate::line_bundle::{CanonicalBundle, Divisor, LineBundle, PicardGroup};
pub use crate::proj::{AffineChart, Proj, TwistingSheaf};
pub use crate::projective_morphism::{ProjectiveMorphism, ProjMorphism};
pub use crate::projective_space::{Hyperplane, LinearSubspace, ProjectivePoint, ProjectiveSpace};
pub use crate::segre::{MultiSegreEmbedding, SegreEmbedding, SegreVariety};
pub use crate::veronese::{VeroneseEmbedding, VeroneseVariety};

// Implement the Scheme trait for ProjectiveSpace to integrate with the generic framework
use crate::generic::{Scheme, Separated};
use rustmath_core::Ring;

impl<R: Ring> Scheme for ProjectiveSpace<R> {
    type BaseRing = R;

    fn base_ring(&self) -> &Self::BaseRing {
        self.base_ring()
    }

    fn dimension(&self) -> Option<usize> {
        Some(self.dimension())
    }

    fn is_projective(&self) -> bool {
        true
    }

    fn is_irreducible(&self) -> bool {
        true // Projective space is irreducible
    }

    fn is_reduced(&self) -> bool {
        true // Projective space is reduced
    }

    fn is_noetherian(&self) -> bool {
        true
    }

    fn is_finite_type(&self) -> bool {
        true
    }
}

impl<R: Ring> Separated for ProjectiveSpace<R> {
    fn is_separated(&self) -> bool {
        true // All projective schemes are separated
    }
}

impl<R: Ring> Scheme for Proj<R> {
    type BaseRing = R;

    fn base_ring(&self) -> &Self::BaseRing {
        self.base_ring()
    }

    fn dimension(&self) -> Option<usize> {
        self.dimension()
    }

    fn is_projective(&self) -> bool {
        true
    }

    fn is_irreducible(&self) -> bool {
        // Proj is irreducible if the graded ring has a unique minimal graded prime
        false // Conservative default
    }

    fn is_reduced(&self) -> bool {
        // Proj is reduced if the graded ring is reduced
        false // Conservative default
    }

    fn is_noetherian(&self) -> bool {
        true
    }

    fn is_finite_type(&self) -> bool {
        true
    }
}

impl<R: Ring> Separated for Proj<R> {
    fn is_separated(&self) -> bool {
        true // Projective schemes are separated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_projective_module_exports() {
        // Verify that key types are accessible through this module
        // This is a compile-time test
    }

    #[test]
    fn test_scheme_trait_impl() {
        // Test that ProjectiveSpace implements Scheme correctly
        let p2: ProjectiveSpace<i32> = ProjectiveSpace::new(2);
        assert!(p2.is_projective());
        assert!(!p2.is_affine());
        assert_eq!(p2.dimension(), Some(2));
        assert!(p2.is_irreducible());
        assert!(p2.is_reduced());
    }
}
