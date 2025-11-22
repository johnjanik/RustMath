//! Segre Embeddings
//!
//! The Segre embedding embeds the product of projective spaces into a single
//! projective space.
//!
//! For ℙⁿ × ℙᵐ, the Segre embedding is:
//!
//! σ: ℙⁿ × ℙᵐ → ℙ⁽ⁿ⁺¹⁾⁽ᵐ⁺¹⁾⁻¹
//!
//! ([x₀ : ... : xₙ], [y₀ : ... : yₘ]) ↦ [... : xᵢyⱼ : ...]
//!
//! where the coordinates on the right are all products xᵢyⱼ in lexicographic order.
//!
//! # Examples
//!
//! **Segre embedding of ℙ¹ × ℙ¹:**
//! - σ: ℙ¹ × ℙ¹ → ℙ³
//! - ([s₀:s₁], [t₀:t₁]) ↦ [s₀t₀ : s₀t₁ : s₁t₀ : s₁t₁]
//! - The image is a quadric surface (2-dimensional) in ℙ³
//!
//! **Segre embedding of ℙ¹ × ℙ²:**
//! - σ: ℙ¹ × ℙ² → ℙ⁵
//! - ([s₀:s₁], [t₀:t₁:t₂]) ↦ [s₀t₀ : s₀t₁ : s₀t₂ : s₁t₀ : s₁t₁ : s₁t₂]

use crate::projective_space::{ProjectivePoint, ProjectiveSpace};
use rustmath_core::Ring;
use num_traits::{Zero, One};
use std::fmt;

/// The Segre embedding σ: ℙⁿ × ℙᵐ → ℙ⁽ⁿ⁺¹⁾⁽ᵐ⁺¹⁾⁻¹
///
/// Embeds the product of two projective spaces into projective space
/// via the map ([x₀:...:xₙ], [y₀:...:yₘ]) ↦ [x₀y₀ : x₀y₁ : ... : xₙyₘ]
///
/// # Properties
///
/// - The image is the Segre variety
/// - Closed embedding (image is a closed subvariety)
/// - Image has dimension n + m
/// - Defined by 2×2 minors of a matrix
#[derive(Clone, Debug)]
pub struct SegreEmbedding<R: Ring> {
    /// First source space ℙⁿ
    source1: ProjectiveSpace<R>,
    /// Second source space ℙᵐ
    source2: ProjectiveSpace<R>,
    /// Target space ℙ⁽ⁿ⁺¹⁾⁽ᵐ⁺¹⁾⁻¹
    target: ProjectiveSpace<R>,
}

impl<R: Ring> SegreEmbedding<R> {
    /// Create a new Segre embedding σ: ℙⁿ × ℙᵐ → ℙᴺ
    ///
    /// # Arguments
    /// * `dim1` - Dimension n of first space ℙⁿ
    /// * `dim2` - Dimension m of second space ℙᵐ
    ///
    /// # Returns
    /// The Segre embedding, where target is ℙᴺ with N = (n+1)(m+1) - 1
    pub fn new(dim1: usize, dim2: usize) -> Self {
        let source1 = ProjectiveSpace::new(dim1);
        let source2 = ProjectiveSpace::new(dim2);

        let num_coords1 = dim1 + 1;
        let num_coords2 = dim2 + 1;
        let target_dimension = num_coords1 * num_coords2 - 1;

        let target = ProjectiveSpace::new(target_dimension);

        SegreEmbedding {
            source1,
            source2,
            target,
        }
    }

    /// Get the first source space
    pub fn source1(&self) -> &ProjectiveSpace<R> {
        &self.source1
    }

    /// Get the second source space
    pub fn source2(&self) -> &ProjectiveSpace<R> {
        &self.source2
    }

    /// Get the target space
    pub fn target(&self) -> &ProjectiveSpace<R> {
        &self.target
    }

    /// Get the dimension of the image (n + m for ℙⁿ × ℙᵐ)
    pub fn image_dimension(&self) -> usize {
        self.source1.dimension() + self.source2.dimension()
    }

    /// Apply the Segre embedding to a pair of points
    ///
    /// Maps ([x₀:...:xₙ], [y₀:...:yₘ]) to [x₀y₀ : x₀y₁ : ... : xₙyₘ]
    pub fn apply(
        &self,
        point1: &ProjectivePoint<R>,
        point2: &ProjectivePoint<R>,
    ) -> Result<ProjectivePoint<R>, String> {
        if !self.source1.contains_point(point1) {
            return Err("First point not in source space".to_string());
        }

        if !self.source2.contains_point(point2) {
            return Err("Second point not in source space".to_string());
        }

        // Compute all products xᵢ·yⱼ in lexicographic order
        let mut target_coords = Vec::new();

        for x in point1.coordinates() {
            for y in point2.coordinates() {
                target_coords.push(x.clone() * y.clone());
            }
        }

        ProjectivePoint::new(target_coords)
    }

    /// Get the index in target coordinates for product xᵢyⱼ
    ///
    /// The coordinates are ordered: x₀y₀, x₀y₁, ..., x₀yₘ, x₁y₀, x₁y₁, ..., xₙyₘ
    pub fn coordinate_index(&self, i: usize, j: usize) -> Option<usize> {
        let m = self.source2.dimension() + 1;
        let n = self.source1.dimension() + 1;

        if i >= n || j >= m {
            return None;
        }

        Some(i * m + j)
    }

    /// Get the defining equations of the Segre variety (image)
    ///
    /// The image is defined by 2×2 minors: zᵢⱼzₖₗ - zᵢₗzₖⱼ = 0
    /// where zᵢⱼ corresponds to xᵢyⱼ
    pub fn image_ideal_description(&self) -> String {
        format!(
            "Segre variety defined by 2×2 minors ({}×{} matrix)",
            self.source1.dimension() + 1,
            self.source2.dimension() + 1
        )
    }
}

impl<R: Ring> fmt::Display for SegreEmbedding<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "σ: {} × {} → {}", self.source1, self.source2, self.target)
    }
}

/// Standard Segre embeddings
impl<R: Ring> SegreEmbedding<R> {
    /// The Segre embedding σ: ℙ¹ × ℙ¹ → ℙ³
    ///
    /// Maps ([s₀:s₁], [t₀:t₁]) to [s₀t₀ : s₀t₁ : s₁t₀ : s₁t₁]
    ///
    /// The image is a quadric surface in ℙ³ defined by z₀z₃ - z₁z₂ = 0
    pub fn p1_times_p1() -> Self {
        SegreEmbedding::new(1, 1)
    }

    /// The Segre embedding σ: ℙ¹ × ℙ² → ℙ⁵
    ///
    /// Maps ([s₀:s₁], [t₀:t₁:t₂]) to products of coordinates
    pub fn p1_times_p2() -> Self {
        SegreEmbedding::new(1, 2)
    }

    /// The Segre embedding σ: ℙ² × ℙ² → ℙ⁸
    ///
    /// Maps ([x₀:x₁:x₂], [y₀:y₁:y₂]) to all products xᵢyⱼ
    pub fn p2_times_p2() -> Self {
        SegreEmbedding::new(2, 2)
    }
}

/// The Segre variety (image of Segre embedding)
///
/// The image of σ: ℙⁿ × ℙᵐ → ℙᴺ
#[derive(Clone, Debug)]
pub struct SegreVariety<R: Ring> {
    /// The embedding that defines this variety
    embedding: SegreEmbedding<R>,
}

impl<R: Ring> SegreVariety<R> {
    /// Create a Segre variety from an embedding
    pub fn new(embedding: SegreEmbedding<R>) -> Self {
        SegreVariety { embedding }
    }

    /// Get the ambient space
    pub fn ambient_space(&self) -> &ProjectiveSpace<R> {
        self.embedding.target()
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.embedding.image_dimension()
    }

    /// Get the underlying embedding
    pub fn embedding(&self) -> &SegreEmbedding<R> {
        &self.embedding
    }
}

impl<R: Ring> fmt::Display for SegreVariety<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Segre({},{}) ⊆ {}",
            self.embedding.source1().dimension(),
            self.embedding.source2().dimension(),
            self.ambient_space()
        )
    }
}

/// Multi-factor Segre embedding for products of more than 2 spaces
///
/// σ: ℙⁿ¹ × ℙⁿ² × ... × ℙⁿᵏ → ℙᴺ
///
/// where N = ∏(nᵢ + 1) - 1
#[derive(Clone, Debug)]
pub struct MultiSegreEmbedding<R: Ring> {
    /// Source spaces [ℙⁿ¹, ℙⁿ², ..., ℙⁿᵏ]
    sources: Vec<ProjectiveSpace<R>>,
    /// Target space ℙᴺ
    target: ProjectiveSpace<R>,
}

impl<R: Ring> MultiSegreEmbedding<R> {
    /// Create a multi-factor Segre embedding
    ///
    /// # Arguments
    /// * `dimensions` - List of dimensions [n₁, n₂, ..., nₖ]
    ///
    /// # Returns
    /// Embedding ℙⁿ¹ × ... × ℙⁿᵏ → ℙᴺ where N = ∏(nᵢ+1) - 1
    pub fn new(dimensions: Vec<usize>) -> Self {
        let sources: Vec<ProjectiveSpace<R>> = dimensions
            .iter()
            .map(|&dim| ProjectiveSpace::new(dim))
            .collect();

        // Compute target dimension: product of all (dim + 1) - 1
        let target_num_coords: usize = dimensions.iter().map(|&d| d + 1).product();
        let target_dimension = target_num_coords - 1;

        let target = ProjectiveSpace::new(target_dimension);

        MultiSegreEmbedding { sources, target }
    }

    /// Get the source spaces
    pub fn sources(&self) -> &[ProjectiveSpace<R>] {
        &self.sources
    }

    /// Get the target space
    pub fn target(&self) -> &ProjectiveSpace<R> {
        &self.target
    }

    /// Get the dimension of the image
    pub fn image_dimension(&self) -> usize {
        self.sources.iter().map(|s| s.dimension()).sum()
    }

    /// Apply the multi-factor Segre embedding
    ///
    /// Maps (p₁, p₂, ..., pₖ) to all products of coordinates
    pub fn apply(&self, points: &[ProjectivePoint<R>]) -> Result<ProjectivePoint<R>, String>
    where
        R: One,
    {
        if points.len() != self.sources.len() {
            return Err("Number of points must match number of source spaces".to_string());
        }

        // Verify each point is in its corresponding space
        for (point, space) in points.iter().zip(&self.sources) {
            if !space.contains_point(point) {
                return Err("Point not in corresponding source space".to_string());
            }
        }

        // Compute all products recursively
        let target_coords = self.compute_products(points, 0, <R as Ring>::one());

        ProjectivePoint::new(target_coords)
    }

    /// Recursively compute all products of coordinates
    fn compute_products(
        &self,
        points: &[ProjectivePoint<R>],
        index: usize,
        current_product: R,
    ) -> Vec<R> {
        if index >= points.len() {
            return vec![current_product];
        }

        let mut results = Vec::new();

        for coord in points[index].coordinates() {
            let new_product = current_product.clone() * coord.clone();
            let mut sub_results = self.compute_products(points, index + 1, new_product);
            results.append(&mut sub_results);
        }

        results
    }
}

impl<R: Ring> fmt::Display for MultiSegreEmbedding<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "σ: ")?;
        for (i, source) in self.sources.iter().enumerate() {
            if i > 0 {
                write!(f, " × ")?;
            }
            write!(f, "{}", source)?;
        }
        write!(f, " → {}", self.target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segre_embedding_dimensions() {
        // σ: ℙ¹ × ℙ¹ → ℙ³
        let s: SegreEmbedding<i32> = SegreEmbedding::p1_times_p1();

        assert_eq!(s.source1().dimension(), 1);
        assert_eq!(s.source2().dimension(), 1);
        assert_eq!(s.target().dimension(), 3);
        assert_eq!(s.image_dimension(), 2); // dim(ℙ¹ × ℙ¹) = 1 + 1 = 2
    }

    #[test]
    fn test_segre_p1_times_p2() {
        // σ: ℙ¹ × ℙ² → ℙ⁵
        let s: SegreEmbedding<i32> = SegreEmbedding::p1_times_p2();

        assert_eq!(s.source1().dimension(), 1);
        assert_eq!(s.source2().dimension(), 2);
        assert_eq!(s.target().dimension(), 5); // (1+1)·(2+1) - 1 = 6 - 1 = 5
        assert_eq!(s.image_dimension(), 3); // 1 + 2 = 3
    }

    #[test]
    fn test_segre_apply_p1_times_p1() {
        // σ: ℙ¹ × ℙ¹ → ℙ³
        let s: SegreEmbedding<i32> = SegreEmbedding::p1_times_p1();

        // Apply to ([1:2], [3:4])
        let p1 = ProjectivePoint::new(vec![1, 2]).unwrap();
        let p2 = ProjectivePoint::new(vec![3, 4]).unwrap();

        let image = s.apply(&p1, &p2).unwrap();

        // Should get [1·3 : 1·4 : 2·3 : 2·4] = [3:4:6:8]
        assert_eq!(image.coordinates(), &[3, 4, 6, 8]);
    }

    #[test]
    fn test_segre_apply_identity() {
        // σ: ℙ¹ × ℙ¹ → ℙ³
        let s: SegreEmbedding<i32> = SegreEmbedding::p1_times_p1();

        // Apply to ([1:0], [1:0])
        let p1 = ProjectivePoint::new(vec![1, 0]).unwrap();
        let p2 = ProjectivePoint::new(vec![1, 0]).unwrap();

        let image = s.apply(&p1, &p2).unwrap();

        // Should get [1:0:0:0]
        assert_eq!(image.coordinates(), &[1, 0, 0, 0]);
    }

    #[test]
    fn test_segre_apply_p1_times_p2() {
        // σ: ℙ¹ × ℙ² → ℙ⁵
        let s: SegreEmbedding<i32> = SegreEmbedding::p1_times_p2();

        // Apply to ([1:2], [3:4:5])
        let p1 = ProjectivePoint::new(vec![1, 2]).unwrap();
        let p2 = ProjectivePoint::new(vec![3, 4, 5]).unwrap();

        let image = s.apply(&p1, &p2).unwrap();

        // [1·3:1·4:1·5:2·3:2·4:2·5] = [3:4:5:6:8:10]
        assert_eq!(image.coordinates(), &[3, 4, 5, 6, 8, 10]);
    }

    #[test]
    fn test_coordinate_index() {
        let s: SegreEmbedding<i32> = SegreEmbedding::p1_times_p1();

        // For ℙ¹ × ℙ¹, coordinates are ordered: x₀y₀, x₀y₁, x₁y₀, x₁y₁
        assert_eq!(s.coordinate_index(0, 0), Some(0)); // x₀y₀
        assert_eq!(s.coordinate_index(0, 1), Some(1)); // x₀y₁
        assert_eq!(s.coordinate_index(1, 0), Some(2)); // x₁y₀
        assert_eq!(s.coordinate_index(1, 1), Some(3)); // x₁y₁
    }

    #[test]
    fn test_segre_variety() {
        let embedding: SegreEmbedding<i32> = SegreEmbedding::p1_times_p1();
        let variety = SegreVariety::new(embedding);

        assert_eq!(variety.dimension(), 2);
        assert_eq!(variety.ambient_space().dimension(), 3);
    }

    #[test]
    fn test_multi_segre_two_factors() {
        // ℙ¹ × ℙ¹ using multi-factor
        let ms: MultiSegreEmbedding<i32> = MultiSegreEmbedding::new(vec![1, 1]);

        assert_eq!(ms.sources().len(), 2);
        assert_eq!(ms.target().dimension(), 3);
        assert_eq!(ms.image_dimension(), 2);
    }

    #[test]
    fn test_multi_segre_three_factors() {
        // ℙ¹ × ℙ¹ × ℙ¹ → ℙ⁷
        let ms: MultiSegreEmbedding<i32> = MultiSegreEmbedding::new(vec![1, 1, 1]);

        assert_eq!(ms.sources().len(), 3);
        assert_eq!(ms.target().dimension(), 7); // 2·2·2 - 1 = 7
        assert_eq!(ms.image_dimension(), 3); // 1+1+1 = 3
    }

    #[test]
    fn test_multi_segre_apply() {
        // ℙ¹ × ℙ¹ × ℙ¹
        let ms: MultiSegreEmbedding<i32> = MultiSegreEmbedding::new(vec![1, 1, 1]);

        let p1 = ProjectivePoint::new(vec![1, 2]).unwrap();
        let p2 = ProjectivePoint::new(vec![3, 4]).unwrap();
        let p3 = ProjectivePoint::new(vec![5, 6]).unwrap();

        let image = ms.apply(&[p1, p2, p3]).unwrap();

        // Should have 2·2·2 = 8 coordinates
        assert_eq!(image.coordinates().len(), 8);

        // First coordinate: 1·3·5 = 15
        assert_eq!(image.coordinates()[0], 15);

        // Last coordinate: 2·4·6 = 48
        assert_eq!(image.coordinates()[7], 48);
    }

    #[test]
    fn test_multi_segre_mixed_dimensions() {
        // ℙ¹ × ℙ² × ℙ³ → ℙ²³ (2·3·4 - 1 = 23)
        let ms: MultiSegreEmbedding<i32> = MultiSegreEmbedding::new(vec![1, 2, 3]);

        assert_eq!(ms.sources().len(), 3);
        assert_eq!(ms.target().dimension(), 23);
        assert_eq!(ms.image_dimension(), 6); // 1+2+3 = 6
    }

    #[test]
    fn test_segre_display() {
        let s: SegreEmbedding<i32> = SegreEmbedding::p1_times_p1();
        let display = format!("{}", s);
        assert!(display.contains("σ"));
        assert!(display.contains("ℙ^1"));
        assert!(display.contains("ℙ^3"));
    }

    #[test]
    fn test_multi_segre_display() {
        let ms: MultiSegreEmbedding<i32> = MultiSegreEmbedding::new(vec![1, 2, 3]);
        let display = format!("{}", ms);
        assert!(display.contains("σ"));
        assert!(display.contains("×"));
    }

    #[test]
    fn test_segre_variety_display() {
        let embedding: SegreEmbedding<i32> = SegreEmbedding::p1_times_p1();
        let variety = SegreVariety::new(embedding);
        let display = format!("{}", variety);
        assert!(display.contains("Segre"));
    }
}
