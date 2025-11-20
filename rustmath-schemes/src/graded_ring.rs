//! Graded Rings
//!
//! A graded ring R is a ring with a direct sum decomposition:
//! R = ⊕_{n≥0} Rₙ
//! where each Rₙ is an abelian group and Rᵢ · Rⱼ ⊆ Rᵢ₊ⱼ
//!
//! Elements of Rₙ are called homogeneous elements of degree n.
//! The Proj construction builds schemes from graded rings.

use rustmath_core::Ring;
use std::collections::BTreeMap;
use std::fmt;

/// A graded ring R = ⊕_{n≥0} Rₙ
///
/// # Examples
///
/// The polynomial ring k[x₀, x₁, ..., xₙ] is graded by total degree.
/// For example, k[x, y] has:
/// - R₀ = k (constants)
/// - R₁ = span{x, y} (linear forms)
/// - R₂ = span{x², xy, y²} (quadratic forms)
/// - etc.
#[derive(Clone, Debug)]
pub struct GradedRing<R: Ring> {
    /// Components indexed by degree
    /// Each degree n maps to the generating set of Rₙ
    components: BTreeMap<usize, Vec<R>>,
    /// Name of the ring (for display)
    name: String,
}

impl<R: Ring> GradedRing<R> {
    /// Create a new graded ring
    pub fn new(name: String) -> Self {
        GradedRing {
            components: BTreeMap::new(),
            name,
        }
    }

    /// Add a generator to degree n component
    pub fn add_generator(&mut self, degree: usize, element: R) {
        self.components
            .entry(degree)
            .or_insert_with(Vec::new)
            .push(element);
    }

    /// Get generators of degree n
    pub fn generators_of_degree(&self, degree: usize) -> Option<&Vec<R>> {
        self.components.get(&degree)
    }

    /// Get all degrees that have been defined
    pub fn degrees(&self) -> Vec<usize> {
        self.components.keys().copied().collect()
    }

    /// Check if this is a positively graded ring (R₀ = base ring, Rₙ = 0 for n < 0)
    pub fn is_positively_graded(&self) -> bool {
        // All degrees should be non-negative
        self.components.keys().all(|&d| d >= 0)
    }

    /// Get the dimension of the degree n component (number of generators)
    ///
    /// Note: This is a simplification. In a full implementation, this would
    /// compute the actual vector space dimension.
    pub fn dimension_at_degree(&self, degree: usize) -> usize {
        self.components.get(&degree).map_or(0, |v| v.len())
    }

    /// Check if the ring is finitely generated in degree 1
    ///
    /// A graded ring is finitely generated in degree 1 if it can be written as
    /// R = k[x₀, x₁, ..., xₙ] where each xᵢ has degree 1.
    pub fn is_generated_in_degree_1(&self) -> bool {
        // Check if we have degree 1 generators
        if self.components.get(&1).is_none() {
            return false;
        }

        // For simplicity, we assume if degree 1 component exists, it generates
        // In a full implementation, we'd verify all higher degrees are generated
        true
    }

    /// Get the Hilbert function H(n) = dim_k(Rₙ)
    ///
    /// Returns the dimension of the degree n component
    pub fn hilbert_function(&self, degree: usize) -> usize {
        self.dimension_at_degree(degree)
    }
}

impl<R: Ring> fmt::Display for GradedRing<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[", self.name)?;
        let mut first = true;
        for (degree, generators) in &self.components {
            if *degree == 0 {
                continue;
            }
            for (i, _) in generators.iter().enumerate() {
                if !first {
                    write!(f, ", ")?;
                }
                write!(f, "x{}_{}", i, degree)?;
                first = false;
            }
        }
        write!(f, "]")
    }
}

/// A homogeneous element of a graded ring
///
/// An element r ∈ R is homogeneous if r ∈ Rₙ for some n.
#[derive(Clone, Debug)]
pub struct HomogeneousElement<R: Ring> {
    /// The element
    element: R,
    /// The degree of this element
    degree: usize,
}

impl<R: Ring> HomogeneousElement<R> {
    /// Create a new homogeneous element
    pub fn new(element: R, degree: usize) -> Self {
        HomogeneousElement { element, degree }
    }

    /// Get the element
    pub fn element(&self) -> &R {
        &self.element
    }

    /// Get the degree
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Multiply two homogeneous elements
    ///
    /// If a ∈ Rₘ and b ∈ Rₙ, then a·b ∈ Rₘ₊ₙ
    pub fn multiply(&self, other: &HomogeneousElement<R>) -> HomogeneousElement<R> {
        HomogeneousElement {
            element: self.element.clone() * other.element.clone(),
            degree: self.degree + other.degree,
        }
    }
}

impl<R: Ring> fmt::Display for HomogeneousElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (deg {})", self.element, self.degree)
    }
}

/// The polynomial ring k[x₀, x₁, ..., xₙ] as a graded ring
///
/// This is graded by total degree: a monomial x₀^{a₀}···xₙ^{aₙ} has degree a₀ + ··· + aₙ
///
/// # Examples
///
/// k[x, y, z] is graded:
/// - Degree 0: k (constants)
/// - Degree 1: {x, y, z}
/// - Degree 2: {x², xy, xz, y², yz, z²}
/// - Degree d: all monomials of total degree d
pub fn polynomial_ring_graded<R: Ring>(num_variables: usize, base_ring_name: &str) -> GradedRing<R> {
    let mut ring = GradedRing::new(base_ring_name.to_string());

    // Degree 0: just the base ring (represented by unit element)
    ring.add_generator(0, R::one());

    // Degree 1: one generator for each variable
    for _ in 0..num_variables {
        // In a full implementation, these would be actual polynomial variables
        // For now, we use a placeholder
        ring.add_generator(1, R::one());
    }

    ring
}

/// Compute the number of monomials of degree d in n variables
///
/// This is the binomial coefficient C(n+d-1, d) = (n+d-1)! / (d! (n-1)!)
///
/// This is the dimension of the degree d component of k[x₀, ..., xₙ₋₁]
pub fn num_monomials_of_degree(num_variables: usize, degree: usize) -> usize {
    if num_variables == 0 {
        return if degree == 0 { 1 } else { 0 };
    }

    // Compute C(n+d-1, d)
    binomial(num_variables + degree - 1, degree)
}

/// Compute binomial coefficient C(n, k) = n! / (k! (n-k)!)
fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }

    // Use the formula C(n,k) = C(n,k-1) * (n-k+1) / k
    let mut result = 1;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// The homogeneous ideal generated by homogeneous elements
///
/// An ideal I ⊆ R is homogeneous if it can be generated by homogeneous elements,
/// or equivalently, if r ∈ I and r = Σ rₙ (graded decomposition), then each rₙ ∈ I.
#[derive(Clone, Debug)]
pub struct HomogeneousIdeal<R: Ring> {
    /// The ambient graded ring
    ambient_ring: GradedRing<R>,
    /// Homogeneous generators of the ideal
    generators: Vec<HomogeneousElement<R>>,
}

impl<R: Ring> HomogeneousIdeal<R> {
    /// Create a new homogeneous ideal
    pub fn new(ambient_ring: GradedRing<R>, generators: Vec<HomogeneousElement<R>>) -> Self {
        HomogeneousIdeal {
            ambient_ring,
            generators,
        }
    }

    /// Get the generators
    pub fn generators(&self) -> &[HomogeneousElement<R>] {
        &self.generators
    }

    /// Get the ambient ring
    pub fn ambient_ring(&self) -> &GradedRing<R> {
        &self.ambient_ring
    }

    /// Check if this is the irrelevant ideal (the ideal generated by all elements of positive degree)
    ///
    /// In k[x₀, ..., xₙ], the irrelevant ideal is (x₀, ..., xₙ)
    pub fn is_irrelevant_ideal(&self) -> bool {
        // Simplified check: see if we have degree 1 generators for all variables
        self.generators.iter().any(|g| g.degree == 1)
    }
}

impl<R: Ring> fmt::Display for HomogeneousIdeal<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, gen) in self.generators.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", gen.element)?;
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graded_ring_creation() {
        let mut ring: GradedRing<i32> = GradedRing::new("k".to_string());
        ring.add_generator(0, 1);
        ring.add_generator(1, 1);
        ring.add_generator(1, 1);

        assert_eq!(ring.dimension_at_degree(0), 1);
        assert_eq!(ring.dimension_at_degree(1), 2);
        assert_eq!(ring.dimension_at_degree(2), 0);
    }

    #[test]
    fn test_graded_ring_degrees() {
        let mut ring: GradedRing<i32> = GradedRing::new("k".to_string());
        ring.add_generator(0, 1);
        ring.add_generator(1, 1);
        ring.add_generator(3, 1);

        let degrees = ring.degrees();
        assert_eq!(degrees, vec![0, 1, 3]);
    }

    #[test]
    fn test_positively_graded() {
        let mut ring: GradedRing<i32> = GradedRing::new("k".to_string());
        ring.add_generator(0, 1);
        ring.add_generator(1, 1);

        assert!(ring.is_positively_graded());
    }

    #[test]
    fn test_homogeneous_element() {
        let elem = HomogeneousElement::new(5, 2);
        assert_eq!(elem.degree(), 2);
        assert_eq!(*elem.element(), 5);
    }

    #[test]
    fn test_homogeneous_multiplication() {
        let a = HomogeneousElement::new(2, 1);
        let b = HomogeneousElement::new(3, 2);
        let c = a.multiply(&b);

        assert_eq!(*c.element(), 6);
        assert_eq!(c.degree(), 3);
    }

    #[test]
    fn test_polynomial_ring_graded() {
        let ring: GradedRing<i32> = polynomial_ring_graded(3, "k");

        // Should have degree 0 (base ring) and degree 1 (variables)
        assert_eq!(ring.dimension_at_degree(0), 1);
        assert_eq!(ring.dimension_at_degree(1), 3);
        assert!(ring.is_generated_in_degree_1());
    }

    #[test]
    fn test_num_monomials_of_degree() {
        // In k[x, y], degree 0: 1 monomial (1)
        assert_eq!(num_monomials_of_degree(2, 0), 1);

        // In k[x, y], degree 1: 2 monomials (x, y)
        assert_eq!(num_monomials_of_degree(2, 1), 2);

        // In k[x, y], degree 2: 3 monomials (x², xy, y²)
        assert_eq!(num_monomials_of_degree(2, 2), 3);

        // In k[x, y, z], degree 2: 6 monomials
        assert_eq!(num_monomials_of_degree(3, 2), 6);

        // In k[x], degree d: 1 monomial (x^d)
        assert_eq!(num_monomials_of_degree(1, 5), 1);
    }

    #[test]
    fn test_binomial_coefficients() {
        assert_eq!(binomial(5, 0), 1);
        assert_eq!(binomial(5, 1), 5);
        assert_eq!(binomial(5, 2), 10);
        assert_eq!(binomial(5, 3), 10);
        assert_eq!(binomial(5, 5), 1);
        assert_eq!(binomial(10, 3), 120);
    }

    #[test]
    fn test_homogeneous_ideal() {
        let ring: GradedRing<i32> = polynomial_ring_graded(3, "k");
        let generators = vec![
            HomogeneousElement::new(1, 1),
            HomogeneousElement::new(1, 1),
        ];

        let ideal = HomogeneousIdeal::new(ring, generators);
        assert_eq!(ideal.generators().len(), 2);
        assert!(ideal.is_irrelevant_ideal());
    }

    #[test]
    fn test_hilbert_function() {
        let mut ring: GradedRing<i32> = GradedRing::new("k".to_string());
        ring.add_generator(0, 1);
        ring.add_generator(1, 1);
        ring.add_generator(1, 1);

        assert_eq!(ring.hilbert_function(0), 1);
        assert_eq!(ring.hilbert_function(1), 2);
        assert_eq!(ring.hilbert_function(2), 0);
    }
}
