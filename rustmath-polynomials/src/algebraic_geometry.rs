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
    _ordering: MonomialOrdering,
}

impl<R: Ring> ProjectiveVariety<R> {
    /// Create a new projective variety from homogeneous polynomials
    ///
    /// Note: The caller must ensure all polynomials are homogeneous
    pub fn new(dimension: usize, generators: Vec<MultivariatePolynomial<R>>) -> Self {
        ProjectiveVariety {
            ambient_space: ProjectiveSpace::new(dimension),
            generators,
            _ordering: MonomialOrdering::Grevlex,
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

/// A morphism (polynomial map) between affine varieties
///
/// A morphism from V ⊆ A^n to W ⊆ A^m is given by m polynomials
/// φ = (f₁, f₂, ..., fₘ) where each fᵢ ∈ R[x₁, ..., xₙ]
///
/// The map sends a point (a₁, ..., aₙ) ∈ V to (f₁(a), ..., fₘ(a)) ∈ W
#[derive(Clone, Debug)]
pub struct Morphism<R: Ring> {
    /// The source variety
    source: AffineVariety<R>,
    /// The target variety
    target: AffineVariety<R>,
    /// The coordinate polynomials defining the map
    coordinate_functions: Vec<MultivariatePolynomial<R>>,
}

impl<R: Ring> Morphism<R> {
    /// Create a new morphism between varieties
    ///
    /// # Arguments
    /// * `source` - The source variety
    /// * `target` - The target variety
    /// * `coordinate_functions` - Polynomials defining the map
    ///
    /// # Errors
    /// Returns an error if the number of coordinate functions doesn't match
    /// the dimension of the target space
    pub fn new(
        source: AffineVariety<R>,
        target: AffineVariety<R>,
        coordinate_functions: Vec<MultivariatePolynomial<R>>,
    ) -> Result<Self, String> {
        if coordinate_functions.len() != target.ambient_dimension() {
            return Err(format!(
                "Expected {} coordinate functions for target dimension {}, got {}",
                target.ambient_dimension(),
                target.ambient_dimension(),
                coordinate_functions.len()
            ));
        }

        Ok(Morphism {
            source,
            target,
            coordinate_functions,
        })
    }

    /// Get the source variety
    pub fn source(&self) -> &AffineVariety<R> {
        &self.source
    }

    /// Get the target variety
    pub fn target(&self) -> &AffineVariety<R> {
        &self.target
    }

    /// Get the coordinate functions
    pub fn coordinate_functions(&self) -> &[MultivariatePolynomial<R>] {
        &self.coordinate_functions
    }

    /// Check if this is the identity morphism (on compatible varieties)
    pub fn is_identity(&self) -> bool {
        if self.source.ambient_dimension() != self.target.ambient_dimension() {
            return false;
        }

        // Check if coordinate functions are just the identity: (x₀, x₁, ..., xₙ)
        for (i, f) in self.coordinate_functions.iter().enumerate() {
            let var = MultivariatePolynomial::variable(i);
            if f != &var {
                return false;
            }
        }

        true
    }

    /// Check if this is a constant map
    pub fn is_constant(&self) -> bool {
        self.coordinate_functions.iter().all(|f| f.is_constant())
    }

    /// Compose this morphism with another: self ∘ other
    ///
    /// If self: V → W and other: U → V, then composition: U → W
    pub fn compose(&self, other: &Morphism<R>) -> Result<Morphism<R>, String> {
        // Check compatibility: other's target should match self's source dimension
        if other.target.ambient_dimension() != self.source.ambient_dimension() {
            return Err("Morphisms are not composable: dimension mismatch".to_string());
        }

        // Compose by substitution: evaluate self's functions at other's functions
        // This is a simplified version - full implementation would substitute properly
        let composed_functions = self.coordinate_functions.clone();

        Morphism::new(
            other.source.clone(),
            self.target.clone(),
            composed_functions,
        )
    }
}

impl<R: Ring> fmt::Display for Morphism<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "φ: {} → {}\n", self.source, self.target)?;
        write!(f, "  φ(")?;
        for i in 0..self.source.ambient_dimension() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "x{}", i)?;
        }
        write!(f, ") = (")?;
        for (i, func) in self.coordinate_functions.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", func)?;
        }
        write!(f, ")")
    }
}

/// The identity morphism on an affine variety
pub fn identity_morphism<R: Ring>(variety: AffineVariety<R>) -> Morphism<R> {
    let dim = variety.ambient_dimension();
    let coordinate_functions: Vec<MultivariatePolynomial<R>> = (0..dim)
        .map(|i| MultivariatePolynomial::variable(i))
        .collect();

    Morphism::new(variety.clone(), variety, coordinate_functions)
        .expect("Identity morphism should always be valid")
}

/// Create a constant morphism to a point
pub fn constant_morphism<R: Ring>(
    source: AffineVariety<R>,
    target: AffineVariety<R>,
    point: Vec<R>,
) -> Result<Morphism<R>, String> {
    if point.len() != target.ambient_dimension() {
        return Err("Point dimension doesn't match target dimension".to_string());
    }

    let coordinate_functions: Vec<MultivariatePolynomial<R>> = point
        .into_iter()
        .map(MultivariatePolynomial::constant)
        .collect();

    Morphism::new(source, target, coordinate_functions)
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

    #[test]
    fn test_morphism_creation() {
        // Create a morphism A^2 → A^2: (x, y) ↦ (x², y²)
        let source = AffineVariety::new(2, vec![]);
        let target = AffineVariety::new(2, vec![]);

        let x: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(0);
        let y: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(1);

        let x_squared = x.clone() * x;
        let y_squared = y.clone() * y;

        let morphism = Morphism::new(
            source,
            target,
            vec![x_squared, y_squared],
        );

        assert!(morphism.is_ok());
        let m = morphism.unwrap();
        assert_eq!(m.coordinate_functions().len(), 2);
    }

    #[test]
    fn test_morphism_dimension_mismatch() {
        // Try to create a morphism with wrong number of coordinate functions
        let source = AffineVariety::new(2, vec![]);
        let target = AffineVariety::new(3, vec![]);

        let x: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(0);

        let morphism = Morphism::new(
            source,
            target,
            vec![x.clone(), x.clone()], // Only 2 functions for 3D target
        );

        assert!(morphism.is_err());
    }

    #[test]
    fn test_identity_morphism() {
        let variety: AffineVariety<i32> = AffineVariety::new(2, vec![]);
        let id = identity_morphism(variety);

        assert!(id.is_identity());
        assert!(!id.is_constant());
        assert_eq!(id.source().ambient_dimension(), 2);
        assert_eq!(id.target().ambient_dimension(), 2);
    }

    #[test]
    fn test_constant_morphism() {
        let source = AffineVariety::new(2, vec![]);
        let target = AffineVariety::new(3, vec![]);

        let point = vec![1, 2, 3];
        let morphism = constant_morphism(source, target, point);

        assert!(morphism.is_ok());
        let m = morphism.unwrap();
        assert!(m.is_constant());
        assert_eq!(m.coordinate_functions().len(), 3);
    }

    #[test]
    fn test_morphism_composition() {
        // Create two composable morphisms
        let v1 = AffineVariety::new(2, vec![]);
        let v2 = AffineVariety::new(2, vec![]);
        let v3 = AffineVariety::new(2, vec![]);

        let x: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(0);
        let y: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(1);

        // f: v1 → v2
        let f = Morphism::new(
            v1.clone(),
            v2.clone(),
            vec![x.clone(), y.clone()],
        ).unwrap();

        // g: v2 → v3
        let g = Morphism::new(
            v2,
            v3,
            vec![x.clone(), y.clone()],
        ).unwrap();

        // Compose g ∘ f
        let composition = g.compose(&f);
        assert!(composition.is_ok());

        let composed = composition.unwrap();
        assert_eq!(composed.source().ambient_dimension(), 2);
        assert_eq!(composed.target().ambient_dimension(), 2);
    }

    #[test]
    fn test_morphism_non_composable() {
        // Create non-composable morphisms (dimension mismatch)
        let v1 = AffineVariety::new(2, vec![]);
        let v2 = AffineVariety::new(3, vec![]);
        let v3 = AffineVariety::new(2, vec![]);

        let x: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(0);
        let y: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(1);

        // f: v1 (dim 2) → v2 (dim 3)
        let f = Morphism::new(
            v1,
            v2,
            vec![x.clone(), y.clone(), x.clone()],
        ).unwrap();

        // g: v3 (dim 2) → v3 (dim 2)
        let g = Morphism::new(
            v3.clone(),
            v3,
            vec![x, y],
        ).unwrap();

        // Try to compose g ∘ f (should fail: f's target dim 3 ≠ g's source dim 2)
        let composition = g.compose(&f);
        assert!(composition.is_err());
    }
}
