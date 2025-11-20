//! Projective Morphisms
//!
//! A morphism between projective varieties (or schemes) is given by
//! homogeneous polynomials of the same degree.
//!
//! For a morphism φ: ℙⁿ → ℙᵐ, we need m+1 homogeneous polynomials
//! f₀, f₁, ..., fₘ of the same degree d, and the map is:
//!
//! φ([x₀ : ... : xₙ]) = [f₀(x) : f₁(x) : ... : fₘ(x)]
//!
//! The polynomials must not all vanish simultaneously (except at undefined points).

use crate::proj::Proj;
use crate::projective_space::{ProjectivePoint, ProjectiveSpace};
use rustmath_core::Ring;
use std::fmt;

/// A projective morphism φ: X → Y between projective schemes
///
/// For projective spaces, this is given by homogeneous polynomials of the same degree.
///
/// # Examples
///
/// - Identity: ℙⁿ → ℙⁿ given by [x₀ : ... : xₙ]
/// - Projection: ℙⁿ → ℙⁿ⁻¹ given by [x₀ : ... : xₙ₋₁]
/// - Veronese: ℙⁿ → ℙᴺ given by degree d monomials
/// - Linear maps: Given by linear forms
#[derive(Clone, Debug)]
pub struct ProjectiveMorphism<R: Ring> {
    /// Source projective space
    source: ProjectiveSpace<R>,
    /// Target projective space
    target: ProjectiveSpace<R>,
    /// Homogeneous polynomials defining the map (all same degree)
    /// In this simplified version, we represent them abstractly
    degree: usize,
    /// Description of the morphism
    description: String,
}

impl<R: Ring> ProjectiveMorphism<R> {
    /// Create a new projective morphism
    ///
    /// # Arguments
    /// * `source` - Source projective space
    /// * `target` - Target projective space
    /// * `degree` - Degree of the defining homogeneous polynomials
    /// * `description` - Human-readable description
    pub fn new(
        source: ProjectiveSpace<R>,
        target: ProjectiveSpace<R>,
        degree: usize,
        description: String,
    ) -> Self {
        ProjectiveMorphism {
            source,
            target,
            degree,
            description,
        }
    }

    /// Get the source space
    pub fn source(&self) -> &ProjectiveSpace<R> {
        &self.source
    }

    /// Get the target space
    pub fn target(&self) -> &ProjectiveSpace<R> {
        &self.target
    }

    /// Get the degree of the defining polynomials
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Check if this is a linear morphism (degree 1)
    pub fn is_linear(&self) -> bool {
        self.degree == 1
    }

    /// Check if this morphism is defined at a point
    ///
    /// A morphism is undefined at a point if all defining polynomials vanish there
    /// (this is a base locus point)
    pub fn is_defined_at(&self, _point: &ProjectivePoint<R>) -> bool {
        // In a full implementation, we'd evaluate the defining polynomials
        // For now, assume it's defined everywhere
        true
    }

    /// Compose two projective morphisms
    ///
    /// If φ: X → Y and ψ: Y → Z, then ψ ∘ φ: X → Z
    pub fn compose(&self, other: &ProjectiveMorphism<R>) -> Result<ProjectiveMorphism<R>, String> {
        if self.target.dimension() != other.source.dimension() {
            return Err("Target of first morphism must match source of second".to_string());
        }

        // Degree of composition is product of degrees
        let composed_degree = self.degree * other.degree;

        Ok(ProjectiveMorphism {
            source: self.source.clone(),
            target: other.target.clone(),
            degree: composed_degree,
            description: format!("({}) ∘ ({})", other.description, self.description),
        })
    }
}

impl<R: Ring> fmt::Display for ProjectiveMorphism<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "φ: {} → {} (degree {}) - {}",
            self.source, self.target, self.degree, self.description
        )
    }
}

/// Standard projective morphisms
impl<R: Ring> ProjectiveMorphism<R> {
    /// The identity morphism id: ℙⁿ → ℙⁿ
    ///
    /// Given by [x₀ : ... : xₙ] (degree 1 linear forms)
    pub fn identity(dimension: usize) -> Self {
        let space = ProjectiveSpace::new(dimension);
        ProjectiveMorphism {
            source: space.clone(),
            target: space,
            degree: 1,
            description: "identity".to_string(),
        }
    }

    /// A linear projection ℙⁿ → ℙⁿ⁻¹
    ///
    /// Projects away from a point (e.g., from [0:...:0:1])
    pub fn linear_projection(source_dimension: usize) -> Result<Self, String> {
        if source_dimension == 0 {
            return Err("Cannot project from ℙ⁰".to_string());
        }

        let source = ProjectiveSpace::new(source_dimension);
        let target = ProjectiveSpace::new(source_dimension - 1);

        Ok(ProjectiveMorphism {
            source,
            target,
            degree: 1,
            description: "linear projection".to_string(),
        })
    }

    /// A linear embedding ℙᵐ → ℙⁿ for m < n
    ///
    /// Embeds as a linear subspace
    pub fn linear_embedding(
        source_dimension: usize,
        target_dimension: usize,
    ) -> Result<Self, String> {
        if source_dimension >= target_dimension {
            return Err("Source dimension must be less than target dimension".to_string());
        }

        let source = ProjectiveSpace::new(source_dimension);
        let target = ProjectiveSpace::new(target_dimension);

        Ok(ProjectiveMorphism {
            source,
            target,
            degree: 1,
            description: "linear embedding".to_string(),
        })
    }
}

/// A morphism between Proj schemes
///
/// For Proj(R) → Proj(S), this is induced by a graded ring homomorphism S → R
#[derive(Clone, Debug)]
pub struct ProjMorphism<R: Ring> {
    /// Source Proj scheme
    source: Proj<R>,
    /// Target Proj scheme
    target: Proj<R>,
    /// Description
    description: String,
}

impl<R: Ring> ProjMorphism<R> {
    /// Create a new Proj morphism
    pub fn new(source: Proj<R>, target: Proj<R>, description: String) -> Self {
        ProjMorphism {
            source,
            target,
            description,
        }
    }

    /// Get the source
    pub fn source(&self) -> &Proj<R> {
        &self.source
    }

    /// Get the target
    pub fn target(&self) -> &Proj<R> {
        &self.target
    }
}

impl<R: Ring> fmt::Display for ProjMorphism<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} → {}: {}",
            self.source, self.target, self.description
        )
    }
}

/// Properties of projective morphisms
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MorphismProperty {
    /// Isomorphism (bijective morphism with morphism inverse)
    Isomorphism,
    /// Closed embedding (injective with closed image)
    ClosedEmbedding,
    /// Open embedding
    OpenEmbedding,
    /// Finite morphism
    Finite,
    /// Proper morphism
    Proper,
    /// Flat morphism
    Flat,
    /// Dominant morphism (dense image)
    Dominant,
    /// Birational morphism (isomorphism on open dense subsets)
    Birational,
}

/// Check if a morphism has a given property
pub fn has_property<R: Ring>(
    _morphism: &ProjectiveMorphism<R>,
    property: &MorphismProperty,
) -> bool {
    // In a full implementation, we'd check the actual properties
    // For now, this is a placeholder
    match property {
        MorphismProperty::Proper => true, // Projective morphisms are always proper
        _ => false,
    }
}

/// The base locus of a morphism
///
/// The base locus is the set of points where the morphism is not defined
/// (where all defining polynomials vanish simultaneously)
#[derive(Clone, Debug)]
pub struct BaseLocus<R: Ring> {
    /// The morphism
    morphism: ProjectiveMorphism<R>,
    /// Description of the base locus
    description: String,
}

impl<R: Ring> BaseLocus<R> {
    /// Create a base locus for a morphism
    pub fn new(morphism: ProjectiveMorphism<R>, description: String) -> Self {
        BaseLocus {
            morphism,
            description,
        }
    }

    /// Check if the base locus is empty
    ///
    /// A morphism with empty base locus is defined everywhere (a morphism of schemes)
    pub fn is_empty(&self) -> bool {
        // In a full implementation, we'd check if the defining polynomials
        // have no common zeros
        // For now, assume empty
        true
    }

    /// Get the codimension of the base locus
    ///
    /// For most morphisms, the base locus has codimension ≥ 2
    pub fn codimension(&self) -> Option<usize> {
        if self.is_empty() {
            None
        } else {
            Some(2) // Typical case
        }
    }
}

impl<R: Ring> fmt::Display for BaseLocus<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            write!(f, "Empty base locus")
        } else {
            write!(f, "Base locus: {}", self.description)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_projective_morphism_creation() {
        let source: ProjectiveSpace<i32> = ProjectiveSpace::new(2);
        let target: ProjectiveSpace<i32> = ProjectiveSpace::new(3);

        let morphism = ProjectiveMorphism::new(
            source,
            target,
            2,
            "test morphism".to_string(),
        );

        assert_eq!(morphism.source().dimension(), 2);
        assert_eq!(morphism.target().dimension(), 3);
        assert_eq!(morphism.degree(), 2);
    }

    #[test]
    fn test_identity_morphism() {
        let id: ProjectiveMorphism<i32> = ProjectiveMorphism::identity(3);

        assert_eq!(id.source().dimension(), 3);
        assert_eq!(id.target().dimension(), 3);
        assert_eq!(id.degree(), 1);
        assert!(id.is_linear());
    }

    #[test]
    fn test_linear_projection() {
        let proj: ProjectiveMorphism<i32> = ProjectiveMorphism::linear_projection(3).unwrap();

        assert_eq!(proj.source().dimension(), 3);
        assert_eq!(proj.target().dimension(), 2);
        assert!(proj.is_linear());
    }

    #[test]
    fn test_linear_projection_invalid() {
        let result: Result<ProjectiveMorphism<i32>, _> = ProjectiveMorphism::linear_projection(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_embedding() {
        let emb: ProjectiveMorphism<i32> = ProjectiveMorphism::linear_embedding(2, 5).unwrap();

        assert_eq!(emb.source().dimension(), 2);
        assert_eq!(emb.target().dimension(), 5);
        assert!(emb.is_linear());
    }

    #[test]
    fn test_linear_embedding_invalid() {
        let result: Result<ProjectiveMorphism<i32>, _> =
            ProjectiveMorphism::linear_embedding(5, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_morphism_composition() {
        // φ: ℙ² → ℙ¹ (degree 1)
        let phi: ProjectiveMorphism<i32> = ProjectiveMorphism::new(
            ProjectiveSpace::new(2),
            ProjectiveSpace::new(1),
            1,
            "φ".to_string(),
        );

        // ψ: ℙ¹ → ℙ³ (degree 2)
        let psi: ProjectiveMorphism<i32> = ProjectiveMorphism::new(
            ProjectiveSpace::new(1),
            ProjectiveSpace::new(3),
            2,
            "ψ".to_string(),
        );

        // ψ ∘ φ: ℙ² → ℙ³ (degree 2)
        let composition = phi.compose(&psi).unwrap();

        assert_eq!(composition.source().dimension(), 2);
        assert_eq!(composition.target().dimension(), 3);
        assert_eq!(composition.degree(), 2); // 1 * 2 = 2
    }

    #[test]
    fn test_morphism_composition_invalid() {
        let phi: ProjectiveMorphism<i32> = ProjectiveMorphism::new(
            ProjectiveSpace::new(2),
            ProjectiveSpace::new(1),
            1,
            "φ".to_string(),
        );

        let psi: ProjectiveMorphism<i32> = ProjectiveMorphism::new(
            ProjectiveSpace::new(3),
            ProjectiveSpace::new(4),
            1,
            "ψ".to_string(),
        );

        let result = phi.compose(&psi);
        assert!(result.is_err()); // Dimensions don't match
    }

    #[test]
    fn test_is_defined_at() {
        let morphism: ProjectiveMorphism<i32> = ProjectiveMorphism::identity(2);
        let point = ProjectivePoint::new(vec![1, 2, 3]).unwrap();

        assert!(morphism.is_defined_at(&point));
    }

    #[test]
    fn test_base_locus() {
        let morphism: ProjectiveMorphism<i32> = ProjectiveMorphism::identity(2);
        let base_locus = BaseLocus::new(morphism, "empty".to_string());

        assert!(base_locus.is_empty());
        assert_eq!(base_locus.codimension(), None);
    }

    #[test]
    fn test_morphism_property() {
        let morphism: ProjectiveMorphism<i32> = ProjectiveMorphism::identity(2);

        // Projective morphisms are always proper
        assert!(has_property(&morphism, &MorphismProperty::Proper));
    }

    #[test]
    fn test_proj_morphism() {
        let source: Proj<i32> = crate::proj::projective_space(2);
        let target: Proj<i32> = crate::proj::projective_space(3);

        let morphism = ProjMorphism::new(source, target, "test".to_string());

        assert_eq!(morphism.source().dimension(), Some(2));
        assert_eq!(morphism.target().dimension(), Some(3));
    }

    #[test]
    fn test_morphism_display() {
        let morphism: ProjectiveMorphism<i32> = ProjectiveMorphism::identity(2);
        let display = format!("{}", morphism);

        assert!(display.contains("ℙ^2"));
        assert!(display.contains("degree 1"));
    }

    #[test]
    fn test_base_locus_display() {
        let morphism: ProjectiveMorphism<i32> = ProjectiveMorphism::identity(2);
        let base_locus = BaseLocus::new(morphism, "test".to_string());
        let display = format!("{}", base_locus);

        assert!(display.contains("Empty"));
    }
}
