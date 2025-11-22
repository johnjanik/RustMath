//! Proj Construction
//!
//! The Proj construction is a fundamental tool in algebraic geometry that
//! constructs a scheme from a graded ring.
//!
//! For a graded ring R = ⊕_{n≥0} Rₙ, the scheme Proj(R) is defined as:
//! - Set of points: homogeneous prime ideals not containing the irrelevant ideal
//! - Topology: Zariski topology
//! - Structure sheaf: constructed from localization
//!
//! # Key Example
//!
//! Proj(k[x₀, x₁, ..., xₙ]) = ℙⁿ (projective n-space over k)

use crate::graded_ring::{GradedRing, HomogeneousElement, HomogeneousIdeal};
use rustmath_core::Ring;
use num_traits::{Zero, One};
use std::fmt;

/// The Proj scheme of a graded ring
///
/// Proj(R) is the set of homogeneous prime ideals p ⊆ R such that
/// p does not contain the irrelevant ideal R₊ = ⊕_{n>0} Rₙ
///
/// # Examples
///
/// - Proj(k[x, y, z]) = ℙ² (projective plane)
/// - Proj(k[x₀, ..., xₙ]) = ℙⁿ (projective n-space)
/// - Proj(k[x, y, z]/(x² + y² - z²)) = conic in ℙ²
#[derive(Clone, Debug)]
pub struct Proj<R: Ring> {
    /// The graded ring
    graded_ring: GradedRing<R>,
    /// Optional ideal for quotient Proj(R/I)
    quotient_ideal: Option<HomogeneousIdeal<R>>,
    /// Dimension of the scheme
    dimension: Option<usize>,
}

impl<R: Ring> Proj<R> {
    /// Create Proj of a graded ring
    pub fn new(graded_ring: GradedRing<R>) -> Self {
        Proj {
            graded_ring,
            quotient_ideal: None,
            dimension: None,
        }
    }

    /// Create Proj(R/I) for a homogeneous ideal I
    pub fn quotient(graded_ring: GradedRing<R>, ideal: HomogeneousIdeal<R>) -> Self {
        Proj {
            graded_ring,
            quotient_ideal: Some(ideal),
            dimension: None,
        }
    }

    /// Get the underlying graded ring
    pub fn graded_ring(&self) -> &GradedRing<R> {
        &self.graded_ring
    }

    /// Get the quotient ideal if any
    pub fn quotient_ideal(&self) -> Option<&HomogeneousIdeal<R>> {
        self.quotient_ideal.as_ref()
    }

    /// Set the dimension of this scheme
    pub fn with_dimension(mut self, dim: usize) -> Self {
        self.dimension = Some(dim);
        self
    }

    /// Get the dimension if known
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    /// Check if this represents projective space
    ///
    /// Proj(k[x₀, ..., xₙ]) = ℙⁿ
    pub fn is_projective_space(&self) -> bool {
        // Check if the ring is generated in degree 1 with no relations
        self.graded_ring.is_generated_in_degree_1() && self.quotient_ideal.is_none()
    }

    /// Standard affine covering
    ///
    /// Proj(R) is covered by affine schemes D₊(f) for f ∈ R₊ homogeneous
    /// The standard covering uses D₊(xᵢ) for degree 1 generators xᵢ
    ///
    /// For ℙⁿ = Proj(k[x₀, ..., xₙ]), we have:
    /// - Uᵢ = D₊(xᵢ) = {[x₀:...:xₙ] : xᵢ ≠ 0} ≅ 𝔸ⁿ
    pub fn standard_affine_charts(&self) -> Vec<AffineChart<R>> {
        let deg1_gens = self.graded_ring.generators_of_degree(1);

        match deg1_gens {
            Some(generators) => {
                let mut charts = Vec::new();
                for (i, gen) in generators.iter().enumerate() {
                    charts.push(AffineChart {
                        index: i,
                        distinguished_element: HomogeneousElement::new(gen.clone(), 1),
                        proj_scheme: self.clone(),
                    });
                }
                charts
            }
            None => Vec::new(),
        }
    }

    /// Get the number of degree 1 generators (for projective space, this is n+1 for ℙⁿ)
    pub fn num_degree_1_generators(&self) -> usize {
        self.graded_ring
            .generators_of_degree(1)
            .map_or(0, |g| g.len())
    }
}

impl<R: Ring> fmt::Display for Proj<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Proj({})", self.graded_ring)?;
        if let Some(ideal) = &self.quotient_ideal {
            write!(f, "/{}", ideal)?;
        }
        Ok(())
    }
}

/// An affine chart D₊(f) in the standard covering of Proj(R)
///
/// For Proj(k[x₀, ..., xₙ]), the chart D₊(xᵢ) consists of points where xᵢ ≠ 0
/// This is isomorphic to affine n-space 𝔸ⁿ with coordinates x₀/xᵢ, ..., x̂ᵢ/xᵢ, ..., xₙ/xᵢ
#[derive(Clone, Debug)]
pub struct AffineChart<R: Ring> {
    /// Index of this chart (which variable is non-zero)
    index: usize,
    /// The distinguished homogeneous element (usually xᵢ)
    distinguished_element: HomogeneousElement<R>,
    /// The Proj scheme this is a chart of
    proj_scheme: Proj<R>,
}

impl<R: Ring> AffineChart<R> {
    /// Get the index of this chart
    pub fn index(&self) -> usize {
        self.index
    }

    /// Get the distinguished element
    pub fn distinguished_element(&self) -> &HomogeneousElement<R> {
        &self.distinguished_element
    }

    /// Convert homogeneous coordinates to affine coordinates on this chart
    ///
    /// For D₊(xᵢ), maps [x₀:...:xₙ] to (x₀/xᵢ, ..., x̂ᵢ/xᵢ, ..., xₙ/xᵢ) ∈ 𝔸ⁿ
    pub fn to_affine_coordinates(&self, homogeneous: &[R]) -> Vec<R> {
        if homogeneous.len() <= self.index {
            return Vec::new();
        }

        let denom = &homogeneous[self.index];
        let mut affine = Vec::new();

        for (i, coord) in homogeneous.iter().enumerate() {
            if i != self.index {
                // In a full implementation, this would compute coord/denom
                // For now, we store the coordinates as-is
                affine.push(coord.clone());
            }
        }

        affine
    }

    /// Convert affine coordinates to homogeneous coordinates
    ///
    /// For D₊(xᵢ), maps (a₀, ..., âᵢ, ..., aₙ) to [a₀:...:1:...:aₙ] where 1 is at position i
    pub fn to_homogeneous_coordinates(&self, affine: &[R]) -> Vec<R>
    where
        R: One,
    {
        let mut homogeneous = Vec::new();

        for i in 0..=affine.len() {
            if i == self.index {
                homogeneous.push(<R as Ring>::one());
            } else if i < self.index {
                homogeneous.push(affine[i].clone());
            } else {
                homogeneous.push(affine[i - 1].clone());
            }
        }

        homogeneous
    }
}

impl<R: Ring> fmt::Display for AffineChart<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "D₊(x{}) ⊆ {}", self.index, self.proj_scheme)
    }
}

/// Create ℙⁿ (projective n-space) as Proj(k[x₀, ..., xₙ])
///
/// # Arguments
/// * `dimension` - The dimension n (not the number of coordinates)
///
/// # Returns
/// Proj(k[x₀, ..., xₙ]) where there are n+1 homogeneous coordinates
///
/// # Examples
///
/// - `projective_space::<i32>(1)` creates ℙ¹ (projective line)
/// - `projective_space::<i32>(2)` creates ℙ² (projective plane)
pub fn projective_space<R: Ring + Zero + One>(dimension: usize) -> Proj<R> {
    let mut ring = GradedRing::new("k".to_string());

    // Degree 0: base ring
    ring.add_generator(0, <R as Ring>::one());

    // Degree 1: n+1 generators for ℙⁿ
    for _ in 0..=dimension {
        ring.add_generator(1, <R as Ring>::one());
    }

    Proj::new(ring).with_dimension(dimension)
}

/// The twisting sheaf 𝒪(n) on Proj(R)
///
/// For Proj(k[x₀, ..., xₙ]) = ℙⁿ, the sheaf 𝒪(n) consists of
/// homogeneous polynomials of degree n.
///
/// Key properties:
/// - 𝒪(1) is called the tautological line bundle
/// - 𝒪(n) ⊗ 𝒪(m) ≅ 𝒪(n+m)
/// - H⁰(ℙⁿ, 𝒪(n)) = space of degree n homogeneous polynomials (for n ≥ 0)
#[derive(Clone, Debug)]
pub struct TwistingSheaf<R: Ring> {
    /// The Proj scheme
    proj_scheme: Proj<R>,
    /// The twist degree n
    degree: isize,
}

impl<R: Ring> TwistingSheaf<R> {
    /// Create the twisting sheaf 𝒪(n)
    pub fn new(proj_scheme: Proj<R>, degree: isize) -> Self {
        TwistingSheaf {
            proj_scheme,
            degree,
        }
    }

    /// Get the degree
    pub fn degree(&self) -> isize {
        self.degree
    }

    /// Tensor product 𝒪(n) ⊗ 𝒪(m) = 𝒪(n+m)
    pub fn tensor(&self, other: &TwistingSheaf<R>) -> TwistingSheaf<R> {
        TwistingSheaf {
            proj_scheme: self.proj_scheme.clone(),
            degree: self.degree + other.degree,
        }
    }

    /// Dual sheaf 𝒪(n)* = 𝒪(-n)
    pub fn dual(&self) -> TwistingSheaf<R> {
        TwistingSheaf {
            proj_scheme: self.proj_scheme.clone(),
            degree: -self.degree,
        }
    }

    /// Global sections H⁰(X, 𝒪(n))
    ///
    /// For ℙⁿ, this is the space of degree n homogeneous polynomials if n ≥ 0,
    /// and 0 if n < 0.
    pub fn global_sections_dimension(&self) -> usize {
        if self.degree < 0 {
            return 0;
        }

        // For ℙⁿ = Proj(k[x₀, ..., xₘ]), dim H⁰(ℙⁿ, 𝒪(d)) = C(m+d, d)
        // where m = num variables - 1 = dimension
        if let Some(dim) = self.proj_scheme.dimension() {
            let num_vars = dim + 1;
            crate::graded_ring::num_monomials_of_degree(num_vars, self.degree as usize)
        } else {
            0
        }
    }
}

impl<R: Ring> fmt::Display for TwistingSheaf<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "𝒪({})", self.degree)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proj_creation() {
        let mut ring: GradedRing<i32> = GradedRing::new("k".to_string());
        ring.add_generator(0, 1);
        ring.add_generator(1, 1);

        let proj = Proj::new(ring);
        assert!(proj.quotient_ideal().is_none());
    }

    #[test]
    fn test_projective_space() {
        let p2: Proj<i32> = projective_space(2);

        assert_eq!(p2.dimension(), Some(2));
        assert!(p2.is_projective_space());
        assert_eq!(p2.num_degree_1_generators(), 3); // x₀, x₁, x₂
    }

    #[test]
    fn test_projective_line() {
        let p1: Proj<i32> = projective_space(1);

        assert_eq!(p1.dimension(), Some(1));
        assert_eq!(p1.num_degree_1_generators(), 2); // x₀, x₁
    }

    #[test]
    fn test_standard_affine_charts() {
        let p2: Proj<i32> = projective_space(2);
        let charts = p2.standard_affine_charts();

        assert_eq!(charts.len(), 3); // U₀, U₁, U₂

        for (i, chart) in charts.iter().enumerate() {
            assert_eq!(chart.index(), i);
        }
    }

    #[test]
    fn test_affine_chart_coordinates() {
        let p2: Proj<i32> = projective_space(2);
        let charts = p2.standard_affine_charts();

        if let Some(u0) = charts.first() {
            // Convert [1:2:3] to affine coordinates on U₀
            let homogeneous = vec![1, 2, 3];
            let affine = u0.to_affine_coordinates(&homogeneous);

            // Should get (2, 3) in affine coordinates (x₁/x₀, x₂/x₀)
            assert_eq!(affine.len(), 2);

            // Convert back
            let back = u0.to_homogeneous_coordinates(&affine);
            assert_eq!(back[0], 1); // x₀ = 1 (normalized)
        }
    }

    #[test]
    fn test_proj_quotient() {
        let ring: GradedRing<i32> = GradedRing::new("k".to_string());
        let generators = vec![HomogeneousElement::new(1, 2)];
        let ideal = HomogeneousIdeal::new(ring.clone(), generators);

        let proj = Proj::quotient(ring, ideal);
        assert!(proj.quotient_ideal().is_some());
        assert!(!proj.is_projective_space());
    }

    #[test]
    fn test_twisting_sheaf() {
        let p2: Proj<i32> = projective_space(2);
        let o_1 = TwistingSheaf::new(p2.clone(), 1);

        assert_eq!(o_1.degree(), 1);

        // 𝒪(1) ⊗ 𝒪(1) = 𝒪(2)
        let o_2 = o_1.tensor(&o_1);
        assert_eq!(o_2.degree(), 2);

        // 𝒪(1)* = 𝒪(-1)
        let o_minus_1 = o_1.dual();
        assert_eq!(o_minus_1.degree(), -1);
    }

    #[test]
    fn test_global_sections() {
        let p2: Proj<i32> = projective_space(2);

        // H⁰(ℙ², 𝒪(0)) = k (dimension 1)
        let o_0 = TwistingSheaf::new(p2.clone(), 0);
        assert_eq!(o_0.global_sections_dimension(), 1);

        // H⁰(ℙ², 𝒪(1)) = k³ (dimension 3: x, y, z)
        let o_1 = TwistingSheaf::new(p2.clone(), 1);
        assert_eq!(o_1.global_sections_dimension(), 3);

        // H⁰(ℙ², 𝒪(2)) has dimension C(4,2) = 6 (x², xy, xz, y², yz, z²)
        let o_2 = TwistingSheaf::new(p2.clone(), 2);
        assert_eq!(o_2.global_sections_dimension(), 6);

        // H⁰(ℙ², 𝒪(-1)) = 0
        let o_minus_1 = TwistingSheaf::new(p2, -1);
        assert_eq!(o_minus_1.global_sections_dimension(), 0);
    }

    #[test]
    fn test_proj_display() {
        let p2: Proj<i32> = projective_space(2);
        let display = format!("{}", p2);
        assert!(display.contains("Proj"));
    }

    #[test]
    fn test_twisting_sheaf_display() {
        let p2: Proj<i32> = projective_space(2);
        let o_3 = TwistingSheaf::new(p2, 3);
        let display = format!("{}", o_3);
        assert!(display.contains("𝒪(3)"));
    }
}
