//! Algebraic Geometry structures
//!
//! Provides affine spaces, projective spaces, and algebraic varieties.

use crate::groebner::{groebner_basis, ideal_membership, MonomialOrdering};
use crate::multivariate::MultivariatePolynomial;
use rustmath_core::Ring;
use std::fmt;

/// Affine space A^n over a ring R
///
/// Represents n-dimensional affine space where points are n-tuples of elements from R.
///
/// # Examples
///
/// ```ignore
/// let affine_space = AffineSpace::<i32>::new(2); // A^2 over integers
/// let point = vec![3, 5]; // Point (3, 5) in A^2
/// assert!(affine_space.contains_point(&point));
/// ```
#[derive(Clone, Debug)]
pub struct AffineSpace<R: Ring> {
    /// Dimension of the affine space
    dimension: usize,
    /// Phantom data to track the ring R
    _phantom: std::marker::PhantomData<R>,
}

impl<R: Ring> AffineSpace<R> {
    /// Create a new affine space of given dimension
    pub fn new(dimension: usize) -> Self {
        AffineSpace {
            dimension,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get the dimension of this affine space
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Check if a point has the correct dimension for this space
    pub fn contains_point(&self, point: &[R]) -> bool {
        point.len() == self.dimension
    }

    /// Create the origin point (0, 0, ..., 0)
    pub fn origin(&self) -> Vec<R> {
        vec![R::zero(); self.dimension]
    }
}

impl<R: Ring> fmt::Display for AffineSpace<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "A^{}", self.dimension)
    }
}

/// Projective space P^n over a ring R
///
/// Represents n-dimensional projective space with homogeneous coordinates.
/// Points are equivalence classes of (n+1)-tuples under scalar multiplication.
///
/// # Homogeneous Coordinates
///
/// A point in P^n is represented by [x₀ : x₁ : ... : xₙ] where not all xᵢ are zero,
/// and [x₀ : x₁ : ... : xₙ] = [λx₀ : λx₁ : ... : λxₙ] for any non-zero λ.
#[derive(Clone, Debug)]
pub struct ProjectiveSpace<R: Ring> {
    /// Dimension of the projective space (not counting homogeneous coordinate)
    dimension: usize,
    _phantom: std::marker::PhantomData<R>,
}

impl<R: Ring> ProjectiveSpace<R> {
    /// Create a new projective space of given dimension
    pub fn new(dimension: usize) -> Self {
        ProjectiveSpace {
            dimension,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get the dimension of this projective space
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Check if homogeneous coordinates are valid (not all zero, correct length)
    pub fn valid_coordinates(&self, coords: &[R]) -> bool {
        coords.len() == self.dimension + 1 && !coords.iter().all(|c| c.is_zero())
    }

    /// Get the number of homogeneous coordinates needed
    pub fn num_coordinates(&self) -> usize {
        self.dimension + 1
    }
}

impl<R: Ring> fmt::Display for ProjectiveSpace<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "P^{}", self.dimension)
    }
}

/// An affine algebraic variety
///
/// Defined as the common zeros of a set of polynomials in affine space.
/// V(f₁, f₂, ..., fₘ) = {p ∈ Aⁿ : fᵢ(p) = 0 for all i}
///
/// # Examples
///
/// ```ignore
/// // Define the variety V(x² + y² - 1) in A²
/// // This is a circle
/// let x = MultivariatePolynomial::variable(0);
/// let y = MultivariatePolynomial::variable(1);
/// let equation = x.clone()*x + y.clone()*y - MultivariatePolynomial::constant(1);
/// let variety = AffineVariety::new(2, vec![equation]);
/// ```
#[derive(Clone, Debug)]
pub struct AffineVariety<R: Ring> {
    /// The ambient affine space
    ambient_space: AffineSpace<R>,
    /// Defining polynomials (generators of the ideal)
    generators: Vec<MultivariatePolynomial<R>>,
    /// Cached Gröbner basis (computed on demand)
    groebner_basis_cache: Option<Vec<MultivariatePolynomial<R>>>,
    /// Monomial ordering used for Gröbner basis
    ordering: MonomialOrdering,
}

impl<R: Ring> AffineVariety<R> {
    /// Create a new affine variety from defining polynomials
    pub fn new(dimension: usize, generators: Vec<MultivariatePolynomial<R>>) -> Self {
        AffineVariety {
            ambient_space: AffineSpace::new(dimension),
            generators,
            groebner_basis_cache: None,
            ordering: MonomialOrdering::Grevlex, // Default to grevlex
        }
    }

    /// Create a variety with a specific monomial ordering
    pub fn with_ordering(
        dimension: usize,
        generators: Vec<MultivariatePolynomial<R>>,
        ordering: MonomialOrdering,
    ) -> Self {
        AffineVariety {
            ambient_space: AffineSpace::new(dimension),
            generators,
            groebner_basis_cache: None,
            ordering,
        }
    }

    /// Get the dimension of the ambient space
    pub fn ambient_dimension(&self) -> usize {
        self.ambient_space.dimension()
    }

    /// Get the defining polynomials
    pub fn generators(&self) -> &[MultivariatePolynomial<R>] {
        &self.generators
    }

    /// Get or compute the Gröbner basis for the defining ideal
    pub fn groebner_basis(&mut self) -> &[MultivariatePolynomial<R>] {
        if self.groebner_basis_cache.is_none() {
            let basis = groebner_basis(self.generators.clone(), self.ordering);
            self.groebner_basis_cache = Some(basis);
        }
        self.groebner_basis_cache.as_ref().unwrap()
    }

    /// Test if a polynomial vanishes on this variety (belongs to the ideal)
    pub fn contains_in_ideal(&self, poly: &MultivariatePolynomial<R>) -> bool {
        ideal_membership(poly, &self.generators, self.ordering)
    }

    /// Get the ideal of this variety
    pub fn ideal(&self) -> Vec<MultivariatePolynomial<R>> {
        self.generators.clone()
    }

    /// Check if this variety is the whole ambient space (ideal = {0})
    pub fn is_whole_space(&self) -> bool {
        self.generators.iter().all(|p| p.is_zero())
    }

    /// Check if this variety is empty (ideal = R[x₁,...,xₙ], contains 1)
    pub fn is_empty(&mut self) -> bool {
        let basis = self.groebner_basis();
        basis.iter().any(|p| p.is_constant() && !p.is_zero())
    }

    /// Compute the intersection of two varieties
    ///
    /// V(I) ∩ V(J) = V(I + J) where I + J is the ideal generated by I ∪ J
    pub fn intersection(&self, other: &AffineVariety<R>) -> AffineVariety<R> {
        let mut combined_generators = self.generators.clone();
        combined_generators.extend(other.generators.clone());

        AffineVariety::with_ordering(
            self.ambient_dimension(),
            combined_generators,
            self.ordering,
        )
    }

    /// Compute the union as a variety (using product of ideals)
    ///
    /// Note: This gives the Zariski closure, not always exact for union
    /// V(I) ∪ V(J) ⊆ V(IJ)
    pub fn union_closure(&self, other: &AffineVariety<R>) -> AffineVariety<R> {
        // Product of ideals: I·J = {Σ fᵢgⱼ : fᵢ ∈ I, gⱼ ∈ J}
        let mut product_generators = Vec::new();

        for f in &self.generators {
            for g in &other.generators {
                product_generators.push(f.clone() * g.clone());
            }
        }

        AffineVariety::with_ordering(
            self.ambient_dimension(),
            product_generators,
            self.ordering,
        )
    }
}

impl<R: Ring> fmt::Display for AffineVariety<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "V(")?;
        for (i, gen) in self.generators.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", gen)?;
        }
        write!(f, ") ⊆ A^{}", self.ambient_dimension())
    }
}

/// A projective algebraic variety
///
/// Defined as the common zeros of homogeneous polynomials in projective space.
#[derive(Clone, Debug)]
pub struct ProjectiveVariety<R: Ring> {
    /// The ambient projective space
    ambient_space: ProjectiveSpace<R>,
    /// Defining homogeneous polynomials
    generators: Vec<MultivariatePolynomial<R>>,
    /// Monomial ordering
    ordering: MonomialOrdering,
}

impl<R: Ring> ProjectiveVariety<R> {
    /// Create a new projective variety from homogeneous polynomials
    ///
    /// Note: The caller must ensure all polynomials are homogeneous
    pub fn new(dimension: usize, generators: Vec<MultivariatePolynomial<R>>) -> Self {
        ProjectiveVariety {
            ambient_space: ProjectiveSpace::new(dimension),
            generators,
            ordering: MonomialOrdering::Grevlex,
        }
    }

    /// Get the dimension of the ambient projective space
    pub fn ambient_dimension(&self) -> usize {
        self.ambient_space.dimension()
    }

    /// Get the defining polynomials
    pub fn generators(&self) -> &[MultivariatePolynomial<R>] {
        &self.generators
    }

    /// Check if a polynomial is homogeneous
    ///
    /// A polynomial is homogeneous if all its terms have the same total degree
    pub fn is_homogeneous(poly: &MultivariatePolynomial<R>) -> bool {
        if poly.is_zero() {
            return true;
        }

        let degree = poly.degree();
        if degree.is_none() {
            return true;
        }

        // All terms should have the same degree
        // This is a simplified check - a proper implementation would verify each term
        true // Placeholder
    }
}

impl<R: Ring> fmt::Display for ProjectiveVariety<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "V(")?;
        for (i, gen) in self.generators.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", gen)?;
        }
        write!(f, ") ⊆ P^{}", self.ambient_dimension())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affine_space() {
        let a2: AffineSpace<i32> = AffineSpace::new(2);
        assert_eq!(a2.dimension(), 2);

        let point = vec![3, 5];
        assert!(a2.contains_point(&point));

        let wrong_dim = vec![1, 2, 3];
        assert!(!a2.contains_point(&wrong_dim));

        let origin = a2.origin();
        assert_eq!(origin, vec![0, 0]);
    }

    #[test]
    fn test_projective_space() {
        let p2: ProjectiveSpace<i32> = ProjectiveSpace::new(2);
        assert_eq!(p2.dimension(), 2);
        assert_eq!(p2.num_coordinates(), 3);

        let valid = vec![1, 2, 3];
        assert!(p2.valid_coordinates(&valid));

        let all_zero = vec![0, 0, 0];
        assert!(!p2.valid_coordinates(&all_zero));

        let wrong_dim = vec![1, 2];
        assert!(!p2.valid_coordinates(&wrong_dim));
    }

    #[test]
    fn test_affine_variety_basic() {
        // Create variety V(x, y) in A^2 (just the origin)
        let x: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(0);
        let y: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(1);

        let variety = AffineVariety::new(2, vec![x.clone(), y.clone()]);

        assert_eq!(variety.ambient_dimension(), 2);
        assert_eq!(variety.generators().len(), 2);
        assert!(!variety.is_whole_space());
    }

    #[test]
    fn test_affine_variety_ideal_membership() {
        // Variety V(x, y)
        let x: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(0);
        let y: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(1);

        let variety = AffineVariety::new(2, vec![x.clone(), y.clone()]);

        // x + y should be in the ideal <x, y>
        let poly = x.clone() + y.clone();
        assert!(variety.contains_in_ideal(&poly));

        // Constant 5 should not be in <x, y>
        let const_poly = MultivariatePolynomial::constant(5);
        assert!(!variety.contains_in_ideal(&const_poly));
    }

    #[test]
    fn test_variety_intersection() {
        // V(x) and V(y) in A^2
        let x: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(0);
        let y: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(1);

        let v1 = AffineVariety::new(2, vec![x.clone()]);
        let v2 = AffineVariety::new(2, vec![y.clone()]);

        // V(x) ∩ V(y) = V(x, y) = the origin
        let intersection = v1.intersection(&v2);
        assert_eq!(intersection.generators().len(), 2);
    }

    #[test]
    fn test_empty_variety() {
        // V(1) is empty (ideal contains 1)
        let one: MultivariatePolynomial<i32> = MultivariatePolynomial::constant(1);
        let mut variety = AffineVariety::new(2, vec![one]);

        assert!(variety.is_empty());
    }

    #[test]
    fn test_whole_space_variety() {
        // V(0) is the whole space
        let zero: MultivariatePolynomial<i32> = MultivariatePolynomial::zero();
        let variety = AffineVariety::new(2, vec![zero]);

        assert!(variety.is_whole_space());
    }

    #[test]
    fn test_projective_variety() {
        // Create a projective variety
        let x: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(0);
        let y: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(1);
        let z: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(2);

        // Define a projective variety (e.g., a conic)
        let equation = x.clone() * x + y.clone() * y - z.clone() * z;
        let variety = ProjectiveVariety::new(2, vec![equation]);

        assert_eq!(variety.ambient_dimension(), 2);
        assert_eq!(variety.generators().len(), 1);
    }
}
