//! Lie Algebra Homomorphisms and Morphisms
//!
//! This module provides infrastructure for morphisms between Lie algebras.
//! A Lie algebra homomorphism f: L₁ → L₂ is a linear map that preserves
//! the Lie bracket: f([x,y]) = [f(x), f(y)] for all x, y ∈ L₁.
//!
//! Corresponds to sage.algebras.lie_algebras.morphism
//!
//! References:
//! - Humphreys, J. "Introduction to Lie Algebras and Representation Theory" (1972)
//! - Erdmann, K. and Wildon, M. "Introduction to Lie Algebras" (2006)

use rustmath_core::Ring;
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::marker::PhantomData;

/// A homomorphism between Lie algebras
///
/// Represents a map f: L₁ → L₂ that preserves the Lie bracket structure.
/// The morphism is defined by specifying the images of generators.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
/// * `Domain` - The type of elements in the domain algebra
/// * `Codomain` - The type of elements in the codomain algebra
///
/// # Examples
///
/// ```
/// # use rustmath_liealgebras::{AbelianLieAlgebra, AbelianLieAlgebraElement};
/// # use rustmath_liealgebras::morphism::LieAlgebraHomomorphism;
/// # use rustmath_integers::Integer;
/// # use std::collections::HashMap;
///
/// let L1: AbelianLieAlgebra<Integer> = AbelianLieAlgebra::new(2);
/// let L2: AbelianLieAlgebra<Integer> = AbelianLieAlgebra::new(3);
///
/// // Define where generators map
/// let mut images = HashMap::new();
/// images.insert(0, AbelianLieAlgebraElement::basis_element(0, 3));
/// images.insert(1, AbelianLieAlgebraElement::basis_element(1, 3));
///
/// let morphism = LieAlgebraHomomorphism::new(2, 3, images);
/// ```
pub struct LieAlgebraHomomorphism<R: Ring, Domain, Codomain> {
    /// Dimension of the domain algebra
    domain_dim: usize,
    /// Dimension of the codomain algebra
    codomain_dim: usize,
    /// Images of domain generators (index -> codomain element)
    generator_images: HashMap<usize, Codomain>,
    /// Marker for coefficient ring and domain type
    _phantom: PhantomData<(R, Domain)>,
}

impl<R, Domain, Codomain> LieAlgebraHomomorphism<R, Domain, Codomain>
where
    R: Ring + Clone + From<i64>,
    Domain: Clone,
    Codomain: Clone,
{
    /// Create a new Lie algebra homomorphism
    ///
    /// # Arguments
    ///
    /// * `domain_dim` - Dimension of the domain algebra
    /// * `codomain_dim` - Dimension of the codomain algebra
    /// * `generator_images` - Map from domain generator indices to codomain elements
    pub fn new(
        domain_dim: usize,
        codomain_dim: usize,
        generator_images: HashMap<usize, Codomain>,
    ) -> Self {
        LieAlgebraHomomorphism {
            domain_dim,
            codomain_dim,
            generator_images,
            _phantom: PhantomData,
        }
    }

    /// Get the image of a generator
    ///
    /// # Arguments
    ///
    /// * `index` - Index of the generator in the domain
    pub fn image_of_generator(&self, index: usize) -> Option<&Codomain> {
        self.generator_images.get(&index)
    }

    /// Get all generator images
    pub fn generator_images(&self) -> &HashMap<usize, Codomain> {
        &self.generator_images
    }

    /// Get domain dimension
    pub fn domain_dimension(&self) -> usize {
        self.domain_dim
    }

    /// Get codomain dimension
    pub fn codomain_dimension(&self) -> usize {
        self.codomain_dim
    }

    /// Check if this is the zero morphism
    pub fn is_zero(&self) -> bool
    where
        Codomain: IsZero,
    {
        self.generator_images
            .values()
            .all(|img| img.is_zero())
    }
}

impl<R, Domain, Codomain> Clone for LieAlgebraHomomorphism<R, Domain, Codomain>
where
    R: Ring + Clone,
    Domain: Clone,
    Codomain: Clone,
{
    fn clone(&self) -> Self {
        LieAlgebraHomomorphism {
            domain_dim: self.domain_dim,
            codomain_dim: self.codomain_dim,
            generator_images: self.generator_images.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<R, Domain, Codomain> Display for LieAlgebraHomomorphism<R, Domain, Codomain>
where
    R: Ring + Clone,
    Domain: Clone,
    Codomain: Clone + Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Lie algebra morphism from {}-dimensional to {}-dimensional algebra",
            self.domain_dim, self.codomain_dim
        )?;
        if !self.generator_images.is_empty() {
            write!(f, "\nGenerator images:")?;
            for (i, img) in &self.generator_images {
                write!(f, "\n  e{} ↦ {}", i, img)?;
            }
        }
        Ok(())
    }
}

/// Trait for checking if an element is zero
pub trait IsZero {
    fn is_zero(&self) -> bool;
}

/// Evaluate a morphism on linear combinations
pub trait EvaluateMorphism<R: Ring, Domain, Codomain> {
    /// Evaluate the morphism on an element
    fn evaluate(&self, element: &Domain) -> Codomain;
}

/// Check if a morphism preserves the Lie bracket
pub trait PreservesBracket<R: Ring, Domain, Codomain> {
    /// Check if f([x,y]) = [f(x), f(y)]
    fn preserves_bracket(&self, x: &Domain, y: &Domain) -> bool;
}

/// Composition of morphisms
pub trait ComposeMorphism<R: Ring, Domain, Intermediate, Codomain> {
    /// Compose two morphisms: (g ∘ f)(x) = g(f(x))
    fn compose(
        &self,
        other: &LieAlgebraHomomorphism<R, Intermediate, Codomain>,
    ) -> LieAlgebraHomomorphism<R, Domain, Codomain>;
}

/// Kernel computation for morphisms
pub trait Kernel<R: Ring, Domain> {
    /// Compute the kernel of the morphism
    fn kernel(&self) -> Vec<Domain>;
}

/// Image computation for morphisms
pub trait Image<Codomain> {
    /// Compute the image of the morphism (as a list of generators)
    fn image(&self) -> Vec<Codomain>;
}

/// Injectivity and surjectivity tests
pub trait MorphismProperties {
    /// Check if the morphism is injective (kernel is trivial)
    fn is_injective(&self) -> bool;

    /// Check if the morphism is surjective (image spans codomain)
    fn is_surjective(&self) -> bool;

    /// Check if the morphism is an isomorphism
    fn is_isomorphism(&self) -> bool {
        self.is_injective() && self.is_surjective()
    }
}

/// The identity morphism on a Lie algebra
pub struct IdentityMorphism<R: Ring, T> {
    dimension: usize,
    _phantom: PhantomData<(R, T)>,
}

impl<R: Ring + Clone, T> IdentityMorphism<R, T> {
    /// Create the identity morphism
    pub fn new(dimension: usize) -> Self {
        IdentityMorphism {
            dimension,
            _phantom: PhantomData,
        }
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

/// The zero morphism between Lie algebras
pub struct ZeroMorphism<R: Ring, Domain, Codomain> {
    domain_dim: usize,
    codomain_dim: usize,
    _phantom: PhantomData<(R, Domain, Codomain)>,
}

impl<R: Ring + Clone, Domain, Codomain> ZeroMorphism<R, Domain, Codomain> {
    /// Create the zero morphism
    pub fn new(domain_dim: usize, codomain_dim: usize) -> Self {
        ZeroMorphism {
            domain_dim,
            codomain_dim,
            _phantom: PhantomData,
        }
    }

    /// Get domain dimension
    pub fn domain_dimension(&self) -> usize {
        self.domain_dim
    }

    /// Get codomain dimension
    pub fn codomain_dimension(&self) -> usize {
        self.codomain_dim
    }
}

/// A collection of morphisms between two Lie algebras (Homset)
///
/// Represents Hom(L₁, L₂), the set of all Lie algebra homomorphisms from L₁ to L₂.
pub struct LieAlgebraHomset<R: Ring, Domain, Codomain> {
    domain_dim: usize,
    codomain_dim: usize,
    _phantom: PhantomData<(R, Domain, Codomain)>,
}

impl<R: Ring + Clone, Domain, Codomain> LieAlgebraHomset<R, Domain, Codomain> {
    /// Create a new homset
    pub fn new(domain_dim: usize, codomain_dim: usize) -> Self {
        LieAlgebraHomset {
            domain_dim,
            codomain_dim,
            _phantom: PhantomData,
        }
    }

    /// Get the zero morphism in this homset
    pub fn zero(&self) -> ZeroMorphism<R, Domain, Codomain> {
        ZeroMorphism::new(self.domain_dim, self.codomain_dim)
    }

    /// Create a morphism from generator images
    pub fn from_images(
        &self,
        generator_images: HashMap<usize, Codomain>,
    ) -> LieAlgebraHomomorphism<R, Domain, Codomain>
    where
        R: From<i64>,
        Domain: Clone,
        Codomain: Clone,
    {
        LieAlgebraHomomorphism::new(self.domain_dim, self.codomain_dim, generator_images)
    }

    /// Get domain dimension
    pub fn domain_dimension(&self) -> usize {
        self.domain_dim
    }

    /// Get codomain dimension
    pub fn codomain_dimension(&self) -> usize {
        self.codomain_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abelian::{AbelianLieAlgebra, AbelianLieAlgebraElement};
    use rustmath_integers::Integer;

    impl IsZero for AbelianLieAlgebraElement<Integer> {
        fn is_zero(&self) -> bool {
            self.is_zero()
        }
    }

    #[test]
    fn test_create_morphism() {
        let _L1: AbelianLieAlgebra<Integer> = AbelianLieAlgebra::new(2);
        let _L2: AbelianLieAlgebra<Integer> = AbelianLieAlgebra::new(3);

        let mut images: HashMap<usize, AbelianLieAlgebraElement<Integer>> = HashMap::new();
        images.insert(0, AbelianLieAlgebraElement::basis_element(0, 3));
        images.insert(1, AbelianLieAlgebraElement::basis_element(1, 3));

        let morphism: LieAlgebraHomomorphism<Integer, AbelianLieAlgebraElement<Integer>, AbelianLieAlgebraElement<Integer>> =
            LieAlgebraHomomorphism::new(2, 3, images);
        assert_eq!(morphism.domain_dimension(), 2);
        assert_eq!(morphism.codomain_dimension(), 3);
    }

    #[test]
    fn test_generator_images() {
        let mut images: HashMap<usize, AbelianLieAlgebraElement<Integer>> = HashMap::new();
        images.insert(0, AbelianLieAlgebraElement::basis_element(0, 2));
        images.insert(1, AbelianLieAlgebraElement::basis_element(1, 2));

        let morphism: LieAlgebraHomomorphism<Integer, AbelianLieAlgebraElement<Integer>, AbelianLieAlgebraElement<Integer>> =
            LieAlgebraHomomorphism::new(2, 2, images);

        let img0 = morphism.image_of_generator(0).unwrap();
        let expected = AbelianLieAlgebraElement::basis_element(0, 2);
        assert_eq!(*img0, expected);
    }

    #[test]
    fn test_zero_morphism_detection() {
        let mut images: HashMap<usize, AbelianLieAlgebraElement<Integer>> = HashMap::new();
        images.insert(0, AbelianLieAlgebraElement::zero(2));
        images.insert(1, AbelianLieAlgebraElement::zero(2));

        let morphism: LieAlgebraHomomorphism<Integer, AbelianLieAlgebraElement<Integer>, AbelianLieAlgebraElement<Integer>> =
            LieAlgebraHomomorphism::new(2, 2, images);

        assert!(morphism.is_zero());
    }

    #[test]
    fn test_homset() {
        let homset: LieAlgebraHomset<Integer, AbelianLieAlgebraElement<Integer>, AbelianLieAlgebraElement<Integer>> =
            LieAlgebraHomset::new(2, 3);

        assert_eq!(homset.domain_dimension(), 2);
        assert_eq!(homset.codomain_dimension(), 3);

        let zero = homset.zero();
        assert_eq!(zero.domain_dimension(), 2);
        assert_eq!(zero.codomain_dimension(), 3);
    }

    #[test]
    fn test_identity_morphism() {
        let id: IdentityMorphism<Integer, AbelianLieAlgebraElement<Integer>> =
            IdentityMorphism::new(3);
        assert_eq!(id.dimension(), 3);
    }

    #[test]
    fn test_homset_from_images() {
        let homset: LieAlgebraHomset<Integer, AbelianLieAlgebraElement<Integer>, AbelianLieAlgebraElement<Integer>> =
            LieAlgebraHomset::new(2, 2);

        let mut images: HashMap<usize, AbelianLieAlgebraElement<Integer>> = HashMap::new();
        images.insert(0, AbelianLieAlgebraElement::basis_element(1, 2));
        images.insert(1, AbelianLieAlgebraElement::basis_element(0, 2));

        let morphism = homset.from_images(images);
        assert_eq!(morphism.domain_dimension(), 2);
        assert_eq!(morphism.codomain_dimension(), 2);
    }

    #[test]
    fn test_morphism_clone() {
        let mut images: HashMap<usize, AbelianLieAlgebraElement<Integer>> = HashMap::new();
        images.insert(0, AbelianLieAlgebraElement::basis_element(0, 2));

        let morphism: LieAlgebraHomomorphism<Integer, AbelianLieAlgebraElement<Integer>, AbelianLieAlgebraElement<Integer>> =
            LieAlgebraHomomorphism::new(2, 2, images);
        let cloned = morphism.clone();

        assert_eq!(cloned.domain_dimension(), morphism.domain_dimension());
        assert_eq!(cloned.codomain_dimension(), morphism.codomain_dimension());
    }

    #[test]
    fn test_morphism_display() {
        let mut images: HashMap<usize, AbelianLieAlgebraElement<Integer>> = HashMap::new();
        images.insert(0, AbelianLieAlgebraElement::basis_element(0, 2));

        let morphism: LieAlgebraHomomorphism<Integer, AbelianLieAlgebraElement<Integer>, AbelianLieAlgebraElement<Integer>> =
            LieAlgebraHomomorphism::new(2, 2, images);
        let display_str = format!("{}", morphism);
        assert!(display_str.contains("Lie algebra morphism"));
        assert!(display_str.contains("2-dimensional"));
    }

    #[test]
    fn test_zero_morphism() {
        let zero: ZeroMorphism<Integer, AbelianLieAlgebraElement<Integer>, AbelianLieAlgebraElement<Integer>> =
            ZeroMorphism::new(3, 4);
        assert_eq!(zero.domain_dimension(), 3);
        assert_eq!(zero.codomain_dimension(), 4);
    }
}
