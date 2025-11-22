//! Line Bundles and Sheaves
//!
//! A line bundle L on a scheme X is a locally free sheaf of rank 1.
//! On projective space, the twisting sheaves 𝒪(n) are the fundamental line bundles.
//!
//! # Ample Line Bundles
//!
//! A line bundle L is **ample** if some tensor power L^⊗n gives an embedding into projective space.
//! On ℙⁿ, the line bundle 𝒪(1) is ample, and so is 𝒪(d) for any d > 0.
//!
//! # Very Ample Line Bundles
//!
//! A line bundle L is **very ample** if it gives an embedding into projective space
//! via its global sections. 𝒪(1) on ℙⁿ is very ample.

use crate::proj::{Proj, TwistingSheaf};
use crate::projective_space::ProjectiveSpace;
use rustmath_core::Ring;
use std::fmt;

/// A line bundle on a projective scheme
///
/// A line bundle is a locally free sheaf of rank 1.
/// The canonical example is 𝒪(n) on Proj(R).
#[derive(Clone, Debug)]
pub struct LineBundle<R: Ring> {
    /// The underlying scheme
    scheme: Proj<R>,
    /// The degree (for 𝒪(n), this is n)
    degree: isize,
    /// Name of the line bundle
    name: String,
}

impl<R: Ring> LineBundle<R> {
    /// Create a new line bundle
    pub fn new(scheme: Proj<R>, degree: isize, name: String) -> Self {
        LineBundle {
            scheme,
            degree,
            name,
        }
    }

    /// Create the structure sheaf 𝒪
    pub fn structure_sheaf(scheme: Proj<R>) -> Self {
        LineBundle {
            scheme,
            degree: 0,
            name: "𝒪".to_string(),
        }
    }

    /// Create the twisting sheaf 𝒪(n)
    pub fn twisting_sheaf(scheme: Proj<R>, degree: isize) -> Self {
        LineBundle {
            scheme,
            degree,
            name: format!("𝒪({})", degree),
        }
    }

    /// Get the scheme
    pub fn scheme(&self) -> &Proj<R> {
        &self.scheme
    }

    /// Get the degree
    pub fn degree(&self) -> isize {
        self.degree
    }

    /// Tensor product of line bundles
    ///
    /// L ⊗ M has degree deg(L) + deg(M)
    pub fn tensor(&self, other: &LineBundle<R>) -> LineBundle<R> {
        LineBundle {
            scheme: self.scheme.clone(),
            degree: self.degree + other.degree,
            name: format!("{} ⊗ {}", self.name, other.name),
        }
    }

    /// Dual line bundle L*
    ///
    /// For 𝒪(n), the dual is 𝒪(-n)
    pub fn dual(&self) -> LineBundle<R> {
        LineBundle {
            scheme: self.scheme.clone(),
            degree: -self.degree,
            name: format!("({})^*", self.name),
        }
    }

    /// Tensor power L^⊗n
    pub fn power(&self, n: isize) -> LineBundle<R> {
        LineBundle {
            scheme: self.scheme.clone(),
            degree: self.degree * n,
            name: format!("({})^⊗{}", self.name, n),
        }
    }

    /// Check if this line bundle is ample
    ///
    /// A line bundle L is ample if some power L^⊗n embeds X into projective space
    pub fn is_ample(&self) -> bool {
        // For 𝒪(d) on ℙⁿ, it's ample iff d > 0
        self.degree > 0
    }

    /// Check if this line bundle is very ample
    ///
    /// L is very ample if it embeds X into projective space
    pub fn is_very_ample(&self) -> bool {
        // For 𝒪(d) on ℙⁿ, it's very ample iff d ≥ 1
        self.degree >= 1
    }

    /// Check if this line bundle is anti-ample
    pub fn is_anti_ample(&self) -> bool {
        // Dual of an ample bundle
        self.degree < 0
    }

    /// Compute dimension of global sections H⁰(X, L)
    ///
    /// For 𝒪(d) on ℙⁿ:
    /// - d ≥ 0: dim = C(n+d, d)
    /// - d < 0: dim = 0
    pub fn h0(&self) -> usize {
        if self.degree < 0 {
            return 0;
        }

        if let Some(dim) = self.scheme.dimension() {
            let twisting = TwistingSheaf::new(self.scheme.clone(), self.degree);
            twisting.global_sections_dimension()
        } else {
            0
        }
    }

    /// Compute dimension of H¹(X, L)
    ///
    /// For ℙⁿ, H¹(ℙⁿ, 𝒪(d)) = 0 for all d (by Serre vanishing)
    pub fn h1(&self) -> usize {
        // For projective space, this is always 0
        0
    }

    /// Euler characteristic χ(L) = Σ (-1)ⁱ hⁱ
    ///
    /// For a line bundle on a variety
    pub fn euler_characteristic(&self) -> isize {
        self.h0() as isize - self.h1() as isize
    }
}

impl<R: Ring> fmt::Display for LineBundle<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} on {}", self.name, self.scheme)
    }
}

/// Ampleness criterion for line bundles
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AmplenessCriterion {
    /// Nakai-Moishezon criterion (intersection theory)
    NakaiMoishezon,
    /// Kleiman criterion (numerical class)
    Kleiman,
    /// Seshadri criterion (growth of global sections)
    Seshadri,
    /// Direct: gives embedding to projective space
    Direct,
}

/// Check ampleness using various criteria
pub fn check_ampleness<R: Ring>(
    line_bundle: &LineBundle<R>,
    _criterion: AmplenessCriterion,
) -> bool {
    // In a full implementation, we'd use different criteria
    // For now, just use the basic degree check
    line_bundle.is_ample()
}

/// The canonical bundle K_X
///
/// For ℙⁿ, K_{ℙⁿ} = 𝒪(-n-1)
#[derive(Clone, Debug)]
pub struct CanonicalBundle<R: Ring> {
    /// The underlying line bundle
    line_bundle: LineBundle<R>,
}

impl<R: Ring + num_traits::Zero + num_traits::One> CanonicalBundle<R> {
    /// Create the canonical bundle of a projective space
    ///
    /// For ℙⁿ, K_{ℙⁿ} = 𝒪(-n-1)
    pub fn of_projective_space(dimension: usize) -> Self {
        let scheme: Proj<R> = crate::proj::projective_space(dimension);
        let degree = -(dimension as isize + 1);

        let line_bundle = LineBundle::twisting_sheaf(scheme, degree);

        CanonicalBundle { line_bundle }
    }

    /// Get the underlying line bundle
    pub fn line_bundle(&self) -> &LineBundle<R> {
        &self.line_bundle
    }

    /// Check if the variety is Fano (K_X anti-ample, i.e., -K_X ample)
    ///
    /// ℙⁿ is Fano since K_{ℙⁿ} = 𝒪(-n-1) and -K_{ℙⁿ} = 𝒪(n+1) is ample
    pub fn is_fano(&self) -> bool {
        self.line_bundle.is_anti_ample()
    }

    /// Check if the variety is Calabi-Yau (K_X trivial)
    pub fn is_calabi_yau(&self) -> bool {
        self.line_bundle.degree() == 0
    }

    /// Check if the variety is of general type (K_X ample)
    pub fn is_general_type(&self) -> bool {
        self.line_bundle.is_ample()
    }
}

impl<R: Ring> fmt::Display for CanonicalBundle<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "K_X = {}", self.line_bundle)
    }
}

/// The Picard group Pic(X)
///
/// The Picard group is the group of line bundles under tensor product.
/// For ℙⁿ, Pic(ℙⁿ) ≅ ℤ generated by 𝒪(1).
#[derive(Clone, Debug)]
pub struct PicardGroup<R: Ring> {
    /// The scheme
    scheme: Proj<R>,
    /// Generator (for ℙⁿ, this is 𝒪(1))
    generator: LineBundle<R>,
}

impl<R: Ring + num_traits::Zero + num_traits::One> PicardGroup<R> {
    /// Create the Picard group of a projective space
    ///
    /// Pic(ℙⁿ) ≅ ℤ with generator 𝒪(1)
    pub fn of_projective_space(dimension: usize) -> Self {
        let scheme: Proj<R> = crate::proj::projective_space(dimension);
        let generator = LineBundle::twisting_sheaf(scheme.clone(), 1);

        PicardGroup { scheme, generator }
    }

    /// Get the generator
    pub fn generator(&self) -> &LineBundle<R> {
        &self.generator
    }

    /// Create an element of Pic(X) from an integer
    ///
    /// For ℙⁿ, n ↦ 𝒪(n)
    pub fn element(&self, n: isize) -> LineBundle<R> {
        LineBundle::twisting_sheaf(self.scheme.clone(), n)
    }

    /// Check if Pic(X) is torsion-free
    ///
    /// For ℙⁿ, Pic(ℙⁿ) ≅ ℤ which is torsion-free
    pub fn is_torsion_free(&self) -> bool {
        true
    }
}

impl<R: Ring> fmt::Display for PicardGroup<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Pic({}) ≅ ℤ · {}", self.scheme, self.generator)
    }
}

/// A divisor on a projective scheme
///
/// Divisors are formal linear combinations of subvarieties of codimension 1
#[derive(Clone, Debug)]
pub struct Divisor<R: Ring> {
    /// The ambient scheme
    scheme: Proj<R>,
    /// Coefficient (degree for effective divisors)
    coefficient: isize,
    /// Description
    description: String,
}

impl<R: Ring> Divisor<R> {
    /// Create a new divisor
    pub fn new(scheme: Proj<R>, coefficient: isize, description: String) -> Self {
        Divisor {
            scheme,
            coefficient,
            description,
        }
    }

    /// Check if this divisor is effective (all coefficients ≥ 0)
    pub fn is_effective(&self) -> bool {
        self.coefficient >= 0
    }

    /// Convert divisor to line bundle (divisor class group → Picard group)
    ///
    /// For ℙⁿ, a divisor of degree d gives 𝒪(d)
    pub fn to_line_bundle(&self) -> LineBundle<R> {
        LineBundle::twisting_sheaf(
            self.scheme.clone(),
            self.coefficient,
        )
    }

    /// Add two divisors
    pub fn add(&self, other: &Divisor<R>) -> Divisor<R> {
        Divisor {
            scheme: self.scheme.clone(),
            coefficient: self.coefficient + other.coefficient,
            description: format!("({}) + ({})", self.description, other.description),
        }
    }

    /// Check if divisor is ample
    pub fn is_ample(&self) -> bool {
        self.to_line_bundle().is_ample()
    }
}

impl<R: Ring> fmt::Display for Divisor<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.coefficient == 1 {
            write!(f, "{}", self.description)
        } else {
            write!(f, "{}·{}", self.coefficient, self.description)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_bundle_creation() {
        let scheme: Proj<i32> = crate::proj::projective_space(2);
        let bundle = LineBundle::twisting_sheaf(scheme, 3);

        assert_eq!(bundle.degree(), 3);
    }

    #[test]
    fn test_structure_sheaf() {
        let scheme: Proj<i32> = crate::proj::projective_space(2);
        let o = LineBundle::structure_sheaf(scheme);

        assert_eq!(o.degree(), 0);
    }

    #[test]
    fn test_tensor_product() {
        let scheme: Proj<i32> = crate::proj::projective_space(2);
        let o2 = LineBundle::twisting_sheaf(scheme.clone(), 2);
        let o3 = LineBundle::twisting_sheaf(scheme, 3);

        let o5 = o2.tensor(&o3);
        assert_eq!(o5.degree(), 5);
    }

    #[test]
    fn test_dual() {
        let scheme: Proj<i32> = crate::proj::projective_space(2);
        let o3 = LineBundle::twisting_sheaf(scheme, 3);

        let dual = o3.dual();
        assert_eq!(dual.degree(), -3);
    }

    #[test]
    fn test_power() {
        let scheme: Proj<i32> = crate::proj::projective_space(2);
        let o2 = LineBundle::twisting_sheaf(scheme, 2);

        let o6 = o2.power(3);
        assert_eq!(o6.degree(), 6);

        let o_minus4 = o2.power(-2);
        assert_eq!(o_minus4.degree(), -4);
    }

    #[test]
    fn test_is_ample() {
        let scheme: Proj<i32> = crate::proj::projective_space(2);

        let o1 = LineBundle::twisting_sheaf(scheme.clone(), 1);
        assert!(o1.is_ample());

        let o_minus1 = LineBundle::twisting_sheaf(scheme.clone(), -1);
        assert!(!o_minus1.is_ample());

        let o0 = LineBundle::structure_sheaf(scheme);
        assert!(!o0.is_ample());
    }

    #[test]
    fn test_is_very_ample() {
        let scheme: Proj<i32> = crate::proj::projective_space(2);

        let o1 = LineBundle::twisting_sheaf(scheme.clone(), 1);
        assert!(o1.is_very_ample());

        let o2 = LineBundle::twisting_sheaf(scheme.clone(), 2);
        assert!(o2.is_very_ample());

        let o0 = LineBundle::structure_sheaf(scheme);
        assert!(!o0.is_very_ample());
    }

    #[test]
    fn test_global_sections_dimension() {
        let scheme: Proj<i32> = crate::proj::projective_space(2);

        // H⁰(ℙ², 𝒪(0)) = k (dimension 1)
        let o0 = LineBundle::structure_sheaf(scheme.clone());
        assert_eq!(o0.h0(), 1);

        // H⁰(ℙ², 𝒪(1)) = k³ (dimension 3)
        let o1 = LineBundle::twisting_sheaf(scheme.clone(), 1);
        assert_eq!(o1.h0(), 3);

        // H⁰(ℙ², 𝒪(2)) has dimension C(4,2) = 6
        let o2 = LineBundle::twisting_sheaf(scheme.clone(), 2);
        assert_eq!(o2.h0(), 6);

        // H⁰(ℙ², 𝒪(-1)) = 0
        let o_minus1 = LineBundle::twisting_sheaf(scheme, -1);
        assert_eq!(o_minus1.h0(), 0);
    }

    #[test]
    fn test_euler_characteristic() {
        let scheme: Proj<i32> = crate::proj::projective_space(2);
        let o1 = LineBundle::twisting_sheaf(scheme, 1);

        let chi = o1.euler_characteristic();
        assert_eq!(chi, 3); // h0 = 3, h1 = 0
    }

    #[test]
    fn test_canonical_bundle() {
        let k_p2: CanonicalBundle<i32> = CanonicalBundle::of_projective_space(2);

        // K_{ℙ²} = 𝒪(-3)
        assert_eq!(k_p2.line_bundle().degree(), -3);
    }

    #[test]
    fn test_fano() {
        let k_p2: CanonicalBundle<i32> = CanonicalBundle::of_projective_space(2);

        // ℙ² is Fano (K is anti-ample)
        assert!(k_p2.is_fano());
        assert!(!k_p2.is_calabi_yau());
        assert!(!k_p2.is_general_type());
    }

    #[test]
    fn test_picard_group() {
        let pic: PicardGroup<i32> = PicardGroup::of_projective_space(2);

        assert!(pic.is_torsion_free());

        // Get 𝒪(5)
        let o5 = pic.element(5);
        assert_eq!(o5.degree(), 5);

        // Get 𝒪(-2)
        let o_minus2 = pic.element(-2);
        assert_eq!(o_minus2.degree(), -2);
    }

    #[test]
    fn test_divisor() {
        let scheme: Proj<i32> = crate::proj::projective_space(2);
        let d = Divisor::new(scheme, 3, "H".to_string());

        assert!(d.is_effective());
        assert!(d.is_ample());

        let bundle = d.to_line_bundle();
        assert_eq!(bundle.degree(), 3);
    }

    #[test]
    fn test_divisor_addition() {
        let scheme: Proj<i32> = crate::proj::projective_space(2);
        let d1 = Divisor::new(scheme.clone(), 2, "H1".to_string());
        let d2 = Divisor::new(scheme, 3, "H2".to_string());

        let sum = d1.add(&d2);
        assert_eq!(sum.coefficient, 5);
    }

    #[test]
    fn test_effective_divisor() {
        let scheme: Proj<i32> = crate::proj::projective_space(2);

        let effective = Divisor::new(scheme.clone(), 3, "H".to_string());
        assert!(effective.is_effective());

        let not_effective = Divisor::new(scheme, -1, "D".to_string());
        assert!(!not_effective.is_effective());
    }

    #[test]
    fn test_ampleness_criterion() {
        let scheme: Proj<i32> = crate::proj::projective_space(2);
        let o1 = LineBundle::twisting_sheaf(scheme, 1);

        assert!(check_ampleness(&o1, AmplenessCriterion::Direct));
    }
}
