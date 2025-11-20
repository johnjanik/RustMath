//! Veronese Embeddings
//!
//! The Veronese embedding νₐ: ℙⁿ → ℙᴺ maps projective space to a higher-dimensional
//! projective space by taking degree d monomials.
//!
//! For ℙⁿ with coordinates [x₀ : ... : xₙ], the d-th Veronese embedding is:
//!
//! νₐ([x₀ : ... : xₙ]) = [... : x₀^{a₀}···xₙ^{aₙ} : ...]
//!
//! where the right side includes all monomials of degree d in lexicographic order.
//!
//! The image is the d-th Veronese variety Vₐ,ₙ ⊆ ℙᴺ where N = C(n+d, d) - 1.
//!
//! # Examples
//!
//! **Veronese surface:** ν₂: ℙ² → ℙ⁵
//! - Maps [x : y : z] to [x² : xy : xz : y² : yz : z²]
//! - Image is a surface in ℙ⁵ (the 2nd Veronese surface)
//!
//! **Twisted cubic:** ν₃: ℙ¹ → ℙ³
//! - Maps [s : t] to [s³ : s²t : st² : t³]
//! - Image is the twisted cubic curve

use crate::graded_ring::num_monomials_of_degree;
use crate::projective_space::{ProjectivePoint, ProjectiveSpace};
use rustmath_core::Ring;
use std::fmt;

/// A multi-index representing a monomial x₀^{a₀} ··· xₙ^{aₙ}
///
/// The exponents (a₀, ..., aₙ) where Σ aᵢ = d (total degree d)
type MultiIndex = Vec<usize>;

/// Generate all multi-indices of degree d in n variables
///
/// Returns all tuples (a₀, ..., aₙ₋₁) where Σ aᵢ = d
/// in lexicographic order.
///
/// # Examples
///
/// - degree=2, nvars=2: [(2,0), (1,1), (0,2)]
/// - degree=2, nvars=3: [(2,0,0), (1,1,0), (1,0,1), (0,2,0), (0,1,1), (0,0,2)]
fn multi_indices(degree: usize, nvars: usize) -> Vec<MultiIndex> {
    let mut indices = Vec::new();

    fn generate(
        current: &mut MultiIndex,
        remaining_degree: usize,
        remaining_vars: usize,
        indices: &mut Vec<MultiIndex>,
    ) {
        if remaining_vars == 1 {
            // Last variable gets all remaining degree
            let mut index = current.clone();
            index.push(remaining_degree);
            indices.push(index);
            return;
        }

        // Try all possible degrees for current variable (reverse order for reverse lex)
        for i in (0..=remaining_degree).rev() {
            let mut new_current = current.clone();
            new_current.push(i);
            generate(
                &mut new_current,
                remaining_degree - i,
                remaining_vars - 1,
                indices,
            );
        }
    }

    generate(&mut Vec::new(), degree, nvars, &mut indices);
    indices
}

/// Evaluate monomial x₀^{a₀}···xₙ^{aₙ} at a point
fn evaluate_monomial<R: Ring>(point: &[R], exponents: &MultiIndex) -> R {
    let mut result = R::one();

    for (i, &exp) in exponents.iter().enumerate() {
        if let Some(coord) = point.get(i) {
            // Compute coord^exp
            let mut power = R::one();
            for _ in 0..exp {
                power = power * coord.clone();
            }
            result = result * power;
        }
    }

    result
}

/// The Veronese embedding νₐ: ℙⁿ → ℙᴺ
///
/// Embeds n-dimensional projective space into N-dimensional projective space
/// using degree d monomials, where N = C(n+d, d) - 1.
///
/// # Properties
///
/// - The image is the d-th Veronese variety
/// - The embedding is defined by the complete linear system |𝒪(d)|
/// - Closed embedding (image is a closed subvariety)
/// - Image has dimension n
#[derive(Clone, Debug)]
pub struct VeroneseEmbedding<R: Ring> {
    /// Source projective space ℙⁿ
    source: ProjectiveSpace<R>,
    /// Target projective space ℙᴺ
    target: ProjectiveSpace<R>,
    /// Degree of the embedding
    degree: usize,
    /// Multi-indices for monomials (in order)
    monomials: Vec<MultiIndex>,
}

impl<R: Ring> VeroneseEmbedding<R> {
    /// Create a new Veronese embedding νₐ: ℙⁿ → ℙᴺ
    ///
    /// # Arguments
    /// * `source_dimension` - Dimension n of source space ℙⁿ
    /// * `degree` - Degree d of the embedding
    ///
    /// # Returns
    /// The Veronese embedding, where the target is ℙᴺ with N = C(n+d, d) - 1
    pub fn new(source_dimension: usize, degree: usize) -> Self {
        let num_vars = source_dimension + 1;
        let num_monomials = num_monomials_of_degree(num_vars, degree);
        let target_dimension = num_monomials - 1;

        let source = ProjectiveSpace::new(source_dimension);
        let target = ProjectiveSpace::new(target_dimension);

        let monomials = multi_indices(degree, num_vars);

        VeroneseEmbedding {
            source,
            target,
            degree,
            monomials,
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

    /// Get the degree
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Get the dimension of the image (equals source dimension)
    pub fn image_dimension(&self) -> usize {
        self.source.dimension()
    }

    /// Apply the Veronese embedding to a point
    ///
    /// Maps [x₀ : ... : xₙ] to [... : x₀^{a₀}···xₙ^{aₙ} : ...]
    pub fn apply(&self, point: &ProjectivePoint<R>) -> Result<ProjectivePoint<R>, String> {
        if !self.source.contains_point(point) {
            return Err("Point not in source space".to_string());
        }

        // Compute all degree d monomials
        let mut target_coords = Vec::new();

        for monomial in &self.monomials {
            let value = evaluate_monomial(point.coordinates(), monomial);
            target_coords.push(value);
        }

        ProjectivePoint::new(target_coords)
    }

    /// Get the defining equations of the Veronese variety (image)
    ///
    /// The image is defined by quadratic equations coming from
    /// the relation: (xᵢxⱼ)·(xₖxₗ) = (xᵢxₖ)·(xⱼxₗ)
    ///
    /// Returns a description of the ideal (not actual polynomials in this simplified version)
    pub fn image_ideal_description(&self) -> String {
        format!(
            "Veronese variety V_{},{} defined by 2×2 minors",
            self.degree,
            self.source.dimension()
        )
    }
}

impl<R: Ring> fmt::Display for VeroneseEmbedding<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ν_{}: {} → {}",
            self.degree, self.source, self.target
        )
    }
}

/// Standard Veronese embeddings
impl<R: Ring> VeroneseEmbedding<R> {
    /// The quadratic Veronese embedding ν₂: ℙⁿ → ℙᴺ
    ///
    /// Maps [x₀ : ... : xₙ] to all degree 2 monomials
    ///
    /// For ℙ²: ν₂([x:y:z]) = [x²:xy:xz:y²:yz:z²]
    /// This is ℙ² → ℙ⁵ (the Veronese surface)
    pub fn quadratic(source_dimension: usize) -> Self {
        VeroneseEmbedding::new(source_dimension, 2)
    }

    /// The cubic Veronese embedding ν₃: ℙⁿ → ℙᴺ
    ///
    /// Maps [x₀ : ... : xₙ] to all degree 3 monomials
    ///
    /// For ℙ¹: ν₃([s:t]) = [s³:s²t:st²:t³]
    /// This is ℙ¹ → ℙ³ (the twisted cubic)
    pub fn cubic(source_dimension: usize) -> Self {
        VeroneseEmbedding::new(source_dimension, 3)
    }

    /// The twisted cubic curve: ν₃: ℙ¹ → ℙ³
    ///
    /// This is the most famous Veronese embedding, mapping
    /// [s : t] ↦ [s³ : s²t : st² : t³]
    ///
    /// The image is a non-planar cubic curve in ℙ³
    pub fn twisted_cubic() -> Self {
        VeroneseEmbedding::cubic(1)
    }

    /// The Veronese surface: ν₂: ℙ² → ℙ⁵
    ///
    /// Maps [x : y : z] ↦ [x² : xy : xz : y² : yz : z²]
    ///
    /// The image is a 2-dimensional surface in ℙ⁵
    pub fn veronese_surface() -> Self {
        VeroneseEmbedding::quadratic(2)
    }
}

/// The Veronese variety Vₐ,ₙ ⊆ ℙᴺ
///
/// The image of the Veronese embedding νₐ: ℙⁿ → ℙᴺ
#[derive(Clone, Debug)]
pub struct VeroneseVariety<R: Ring> {
    /// The embedding that defines this variety
    embedding: VeroneseEmbedding<R>,
}

impl<R: Ring> VeroneseVariety<R> {
    /// Create a Veronese variety from an embedding
    pub fn new(embedding: VeroneseEmbedding<R>) -> Self {
        VeroneseVariety { embedding }
    }

    /// Get the ambient space
    pub fn ambient_space(&self) -> &ProjectiveSpace<R> {
        self.embedding.target()
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.embedding.image_dimension()
    }

    /// Get the degree
    pub fn degree(&self) -> usize {
        self.embedding.degree()
    }

    /// Get the underlying embedding
    pub fn embedding(&self) -> &VeroneseEmbedding<R> {
        &self.embedding
    }
}

impl<R: Ring> fmt::Display for VeroneseVariety<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "V_{}_{} ⊆ {}",
            self.degree(),
            self.dimension(),
            self.ambient_space()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_indices_degree_1() {
        let indices = multi_indices(1, 2);
        assert_eq!(indices.len(), 2);
        assert_eq!(indices[0], vec![1, 0]);
        assert_eq!(indices[1], vec![0, 1]);
    }

    #[test]
    fn test_multi_indices_degree_2() {
        let indices = multi_indices(2, 2);
        assert_eq!(indices.len(), 3);
        assert_eq!(indices[0], vec![2, 0]);
        assert_eq!(indices[1], vec![1, 1]);
        assert_eq!(indices[2], vec![0, 2]);
    }

    #[test]
    fn test_multi_indices_degree_2_three_vars() {
        let indices = multi_indices(2, 3);
        assert_eq!(indices.len(), 6); // C(3+2-1, 2) = C(4,2) = 6

        // Should be: x², xy, xz, y², yz, z²
        assert_eq!(indices[0], vec![2, 0, 0]);
        assert_eq!(indices[1], vec![1, 1, 0]);
        assert_eq!(indices[2], vec![1, 0, 1]);
        assert_eq!(indices[3], vec![0, 2, 0]);
        assert_eq!(indices[4], vec![0, 1, 1]);
        assert_eq!(indices[5], vec![0, 0, 2]);
    }

    #[test]
    fn test_evaluate_monomial() {
        let point = vec![2, 3, 5];

        // x²: [2,0,0]
        let result = evaluate_monomial(&point, &vec![2, 0, 0]);
        assert_eq!(result, 4);

        // xy: [1,1,0]
        let result = evaluate_monomial(&point, &vec![1, 1, 0]);
        assert_eq!(result, 6);

        // xyz: [1,1,1]
        let result = evaluate_monomial(&point, &vec![1, 1, 1]);
        assert_eq!(result, 30);

        // z²: [0,0,2]
        let result = evaluate_monomial(&point, &vec![0, 0, 2]);
        assert_eq!(result, 25);
    }

    #[test]
    fn test_veronese_embedding_dimensions() {
        // ν₂: ℙ² → ℙ⁵
        let v: VeroneseEmbedding<i32> = VeroneseEmbedding::quadratic(2);

        assert_eq!(v.source().dimension(), 2);
        assert_eq!(v.target().dimension(), 5);
        assert_eq!(v.degree(), 2);
        assert_eq!(v.image_dimension(), 2);
    }

    #[test]
    fn test_twisted_cubic_dimensions() {
        // ν₃: ℙ¹ → ℙ³
        let tc: VeroneseEmbedding<i32> = VeroneseEmbedding::twisted_cubic();

        assert_eq!(tc.source().dimension(), 1);
        assert_eq!(tc.target().dimension(), 3);
        assert_eq!(tc.degree(), 3);
    }

    #[test]
    fn test_veronese_surface_dimensions() {
        // ν₂: ℙ² → ℙ⁵
        let vs: VeroneseEmbedding<i32> = VeroneseEmbedding::veronese_surface();

        assert_eq!(vs.source().dimension(), 2);
        assert_eq!(vs.target().dimension(), 5);
        assert_eq!(vs.degree(), 2);
    }

    #[test]
    fn test_veronese_apply_twisted_cubic() {
        // ν₃: ℙ¹ → ℙ³, [s:t] ↦ [s³:s²t:st²:t³]
        let v: VeroneseEmbedding<i32> = VeroneseEmbedding::twisted_cubic();

        // Apply to [1:1]
        let point = ProjectivePoint::new(vec![1, 1]).unwrap();
        let image = v.apply(&point).unwrap();

        // Should get [1:1:1:1] since 1³=1, 1²·1=1, 1·1²=1, 1³=1
        assert_eq!(image.coordinates(), &[1, 1, 1, 1]);

        // Apply to [2:1]
        let point2 = ProjectivePoint::new(vec![2, 1]).unwrap();
        let image2 = v.apply(&point2).unwrap();

        // Should get [8:4:2:1] since 2³=8, 2²·1=4, 2·1²=2, 1³=1
        assert_eq!(image2.coordinates(), &[8, 4, 2, 1]);
    }

    #[test]
    fn test_veronese_apply_surface() {
        // ν₂: ℙ² → ℙ⁵, [x:y:z] ↦ [x²:xy:xz:y²:yz:z²]
        let v: VeroneseEmbedding<i32> = VeroneseEmbedding::veronese_surface();

        // Apply to [1:1:1]
        let point = ProjectivePoint::new(vec![1, 1, 1]).unwrap();
        let image = v.apply(&point).unwrap();

        // All monomials equal 1
        assert_eq!(image.coordinates(), &[1, 1, 1, 1, 1, 1]);

        // Apply to [2:0:1]
        let point2 = ProjectivePoint::new(vec![2, 0, 1]).unwrap();
        let image2 = v.apply(&point2).unwrap();

        // [4:0:2:0:0:1] for [x²:xy:xz:y²:yz:z²]
        assert_eq!(image2.coordinates(), &[4, 0, 2, 0, 0, 1]);
    }

    #[test]
    fn test_veronese_variety() {
        let embedding: VeroneseEmbedding<i32> = VeroneseEmbedding::twisted_cubic();
        let variety = VeroneseVariety::new(embedding);

        assert_eq!(variety.dimension(), 1);
        assert_eq!(variety.degree(), 3);
        assert_eq!(variety.ambient_space().dimension(), 3);
    }

    #[test]
    fn test_veronese_display() {
        let v: VeroneseEmbedding<i32> = VeroneseEmbedding::twisted_cubic();
        let display = format!("{}", v);
        assert!(display.contains("ν_3"));
        assert!(display.contains("ℙ^1"));
        assert!(display.contains("ℙ^3"));
    }

    #[test]
    fn test_quadratic_veronese_p1() {
        // ν₂: ℙ¹ → ℙ²
        let v: VeroneseEmbedding<i32> = VeroneseEmbedding::quadratic(1);

        assert_eq!(v.source().dimension(), 1);
        assert_eq!(v.target().dimension(), 2);

        // Maps [s:t] to [s²:st:t²]
        let point = ProjectivePoint::new(vec![3, 2]).unwrap();
        let image = v.apply(&point).unwrap();

        // [9:6:4] for [s²:st:t²]
        assert_eq!(image.coordinates(), &[9, 6, 4]);
    }

    #[test]
    fn test_num_monomials() {
        // Verify our multi_indices generates correct number
        let indices = multi_indices(2, 3);
        assert_eq!(indices.len(), 6);

        let indices3 = multi_indices(3, 2);
        assert_eq!(indices3.len(), 4); // s³, s²t, st², t³
    }
}
