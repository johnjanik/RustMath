//! Morphisms for modules with basis
//!
//! This module provides morphism types for modules with a distinguished basis.
//! Morphisms are defined by their action on basis elements and extended linearly.

use rustmath_core::Ring;
use rustmath_category::morphism::Morphism;
use crate::with_basis::element::ModuleWithBasisElement;
use crate::with_basis::parent::ModuleWithBasis;
use std::collections::BTreeMap;
use std::fmt;
use std::marker::PhantomData;

/// A morphism between modules with basis
///
/// The morphism is completely determined by where it sends basis elements.
/// For a morphism f: M → N, we store f(e_i) for each basis element e_i of M.
///
/// # Type Parameters
/// - `I`: Index type for basis elements
/// - `R`: Coefficient ring type
/// - `M`: Source module type
/// - `N`: Target module type
#[derive(Clone, Debug)]
pub struct ModuleWithBasisMorphism<I, R, M, N>
where
    I: Ord + Clone,
    R: Ring,
    M: ModuleWithBasis<BasisIndex = I, BaseRing = R>,
    N: ModuleWithBasis<BasisIndex = I, BaseRing = R>,
{
    source: M,
    target: N,
    /// The action on basis elements: maps source basis index to target element
    basis_action: BTreeMap<I, ModuleWithBasisElement<I, R>>,
    _phantom: PhantomData<R>,
}

impl<I, R, M, N> ModuleWithBasisMorphism<I, R, M, N>
where
    I: Ord + Clone + fmt::Debug,
    R: Ring,
    M: ModuleWithBasis<BasisIndex = I, BaseRing = R, Element = ModuleWithBasisElement<I, R>>,
    N: ModuleWithBasis<BasisIndex = I, BaseRing = R, Element = ModuleWithBasisElement<I, R>>,
{
    /// Create a new morphism from basis action
    ///
    /// # Arguments
    /// - `source`: Source module
    /// - `target`: Target module
    /// - `basis_action`: Map from source basis indices to images in target
    pub fn new(
        source: M,
        target: N,
        basis_action: BTreeMap<I, ModuleWithBasisElement<I, R>>,
    ) -> Self {
        ModuleWithBasisMorphism {
            source,
            target,
            basis_action,
            _phantom: PhantomData,
        }
    }

    /// Create a morphism from a function on basis elements
    pub fn from_function<F>(source: M, target: N, f: F) -> Self
    where
        F: Fn(&I) -> ModuleWithBasisElement<I, R>,
    {
        let mut basis_action = BTreeMap::new();
        for index in source.basis_indices() {
            basis_action.insert(index.clone(), f(&index));
        }
        ModuleWithBasisMorphism {
            source,
            target,
            basis_action,
            _phantom: PhantomData,
        }
    }

    /// Create the zero morphism
    pub fn zero(source: M, target: N) -> Self {
        let mut basis_action = BTreeMap::new();
        for index in source.basis_indices() {
            basis_action.insert(index, ModuleWithBasisElement::zero());
        }
        ModuleWithBasisMorphism {
            source,
            target,
            basis_action,
            _phantom: PhantomData,
        }
    }

    /// Create the identity morphism
    pub fn identity(module: M) -> Self
    where
        N: From<M>,
    {
        let target = N::from(module.clone());
        let mut basis_action = BTreeMap::new();
        for index in module.basis_indices() {
            if let Some(elem) = module.basis_element(&index) {
                basis_action.insert(index, elem);
            }
        }
        ModuleWithBasisMorphism {
            source: module,
            target,
            basis_action,
            _phantom: PhantomData,
        }
    }

    /// Get the action on a basis element
    pub fn on_basis(&self, index: &I) -> Option<&ModuleWithBasisElement<I, R>> {
        self.basis_action.get(index)
    }

    /// Apply the morphism to an element
    pub fn apply(&self, elem: &ModuleWithBasisElement<I, R>) -> ModuleWithBasisElement<I, R> {
        let mut result = ModuleWithBasisElement::zero();

        for (index, coeff) in elem.items() {
            if let Some(image) = self.on_basis(index) {
                let scaled = image.scalar_mul(coeff);
                result = result.add(&scaled);
            }
        }

        result
    }

    /// Get the source module
    pub fn domain(&self) -> &M {
        &self.source
    }

    /// Get the target module
    pub fn codomain(&self) -> &N {
        &self.target
    }

    /// Check if this is the zero morphism
    pub fn is_zero(&self) -> bool {
        self.basis_action.values().all(|v| v.is_zero())
    }

    /// Get the matrix representation of this morphism
    ///
    /// Returns a matrix where column i is the image of the i-th basis element
    pub fn matrix(&self) -> Option<Vec<Vec<R>>> {
        let source_indices = self.source.basis_indices();
        let target_indices = self.target.basis_indices();

        if source_indices.is_empty() || target_indices.is_empty() {
            return None;
        }

        let mut matrix = vec![vec![R::zero(); source_indices.len()]; target_indices.len()];

        for (j, source_idx) in source_indices.iter().enumerate() {
            if let Some(image) = self.on_basis(source_idx) {
                for (i, target_idx) in target_indices.iter().enumerate() {
                    matrix[i][j] = image.coefficient(target_idx);
                }
            }
        }

        Some(matrix)
    }

    /// Compute the kernel of this morphism
    ///
    /// Returns basis elements that span the kernel
    pub fn kernel(&self) -> Vec<ModuleWithBasisElement<I, R>> {
        // TODO: Implement proper kernel computation via linear algebra
        // For now, return empty (indicating trivial kernel)
        vec![]
    }

    /// Compute the image of this morphism
    ///
    /// Returns basis elements that span the image
    pub fn image(&self) -> Vec<ModuleWithBasisElement<I, R>> {
        // The image is spanned by the images of the source basis
        self.basis_action.values().cloned().collect()
    }

    /// Compute the rank (dimension of image)
    pub fn rank(&self) -> usize {
        // TODO: Implement proper rank computation
        // For now, count non-zero basis images
        self.basis_action.values().filter(|v| !v.is_zero()).count()
    }

    /// Check if this morphism is injective
    pub fn is_injective(&self) -> bool {
        // Injective iff kernel is trivial
        self.kernel().is_empty()
    }

    /// Check if this morphism is surjective
    pub fn is_surjective(&self) -> bool {
        // Surjective iff rank equals target dimension
        if let Some(target_dim) = self.target.dimension() {
            self.rank() == target_dim
        } else {
            false
        }
    }

    /// Check if this is an isomorphism
    pub fn is_isomorphism(&self) -> bool {
        self.is_injective() && self.is_surjective()
    }
}

impl<I, R, M> ModuleWithBasisMorphism<I, R, M, M>
where
    I: Ord + Clone + fmt::Debug,
    R: Ring,
    M: ModuleWithBasis<BasisIndex = I, BaseRing = R, Element = ModuleWithBasisElement<I, R>>,
{
    /// Compose two endomorphisms: self ∘ other
    pub fn compose_endo(&self, other: &Self) -> Self {
        // Compute composition: for each basis element e, compute self(other(e))
        let mut basis_action = BTreeMap::new();

        for index in self.source.basis_indices() {
            if let Some(other_image) = other.on_basis(&index) {
                let composed_image = self.apply(other_image);
                basis_action.insert(index, composed_image);
            }
        }

        ModuleWithBasisMorphism {
            source: other.source.clone(),
            target: self.target.clone(),
            basis_action,
            _phantom: PhantomData,
        }
    }

    /// Compute powers of an endomorphism
    pub fn power(&self, n: usize) -> Self {
        if n == 0 {
            return Self::identity(self.source.clone());
        }

        let mut result = self.clone();
        for _ in 1..n {
            result = result.compose_endo(self);
        }
        result
    }

    /// Compute the trace of an endomorphism
    pub fn trace(&self) -> R {
        let mut trace = R::zero();
        for index in self.source.basis_indices() {
            if let Some(image) = self.on_basis(&index) {
                trace = trace + image.coefficient(&index);
            }
        }
        trace
    }
}

// Implement Morphism trait from rustmath-category for endomorphisms (M = N)
impl<I, R, M> Morphism for ModuleWithBasisMorphism<I, R, M, M>
where
    I: Ord + Clone + fmt::Debug,
    R: Ring,
    M: ModuleWithBasis<BasisIndex = I, BaseRing = R, Element = ModuleWithBasisElement<I, R>>,
{
    type Object = M;

    fn source(&self) -> &Self::Object {
        &self.source
    }

    fn target(&self) -> &Self::Object {
        &self.target
    }

    fn compose(&self, other: &Self) -> Option<Self> {
        Some(self.compose_endo(other))
    }

    fn is_identity(&self) -> bool {
        // Check if this maps each basis element to itself
        for index in self.source.basis_indices() {
            if let Some(image) = self.on_basis(&index) {
                if let Some(expected) = self.source.basis_element(&index) {
                    if image != &expected {
                        return false;
                    }
                }
            }
        }
        true
    }
}

/// Morphism methods - additional methods for morphisms
pub trait ModuleWithBasisMorphismMethods<I, R, M, N>
where
    I: Ord + Clone,
    R: Ring,
    M: ModuleWithBasis<BasisIndex = I, BaseRing = R>,
    N: ModuleWithBasis<BasisIndex = I, BaseRing = R>,
{
    /// Get the matrix representation with respect to given bases
    fn matrix_wrt_bases(&self, source_basis: &[I], target_basis: &[I]) -> Vec<Vec<R>>;

    /// Restrict to a submodule
    fn restrict(&self, submodule: M) -> ModuleWithBasisMorphism<I, R, M, N>;

    /// Lift from a quotient module
    fn lift(&self) -> Option<ModuleWithBasisMorphism<I, R, M, N>>;
}

/// Legacy struct for compatibility
#[derive(Clone, Debug)]
pub struct ModuleMorphismWithBasis<I, R>
where
    I: Ord + Clone,
    R: Ring,
{
    /// The action on basis elements
    basis_action: BTreeMap<I, ModuleWithBasisElement<I, R>>,
}

impl<I, R> ModuleMorphismWithBasis<I, R>
where
    I: Ord + Clone,
    R: Ring,
{
    pub fn new(basis_action: BTreeMap<I, ModuleWithBasisElement<I, R>>) -> Self {
        ModuleMorphismWithBasis { basis_action }
    }

    /// Apply to an element
    pub fn apply(&self, elem: &ModuleWithBasisElement<I, R>) -> ModuleWithBasisElement<I, R> {
        let mut result = ModuleWithBasisElement::zero();

        for (index, coeff) in elem.items() {
            if let Some(image) = self.basis_action.get(index) {
                let scaled = image.scalar_mul(coeff);
                result = result.add(&scaled);
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::with_basis::parent::FreeModuleWithBasis;
    use num_bigint::BigInt;

    #[test]
    fn test_zero_morphism() {
        let base_ring = BigInt::from(0);
        let source = FreeModuleWithBasis::standard(base_ring.clone(), 3);
        let target = FreeModuleWithBasis::standard(base_ring, 2);

        let zero = ModuleWithBasisMorphism::zero(source.clone(), target);
        assert!(zero.is_zero());

        let e0 = source.basis_element(&0).unwrap();
        let image = zero.apply(&e0);
        assert!(image.is_zero());
    }

    #[test]
    fn test_morphism_application() {
        let base_ring = BigInt::from(0);
        let source = FreeModuleWithBasis::standard(base_ring.clone(), 2);
        let target = FreeModuleWithBasis::standard(base_ring, 3);

        // Define morphism: e_0 ↦ 2*f_0 + f_1, e_1 ↦ f_2
        let mut basis_action = BTreeMap::new();
        basis_action.insert(
            0,
            ModuleWithBasisElement::from_terms(vec![
                (0, BigInt::from(2)),
                (1, BigInt::from(1)),
            ]),
        );
        basis_action.insert(
            1,
            ModuleWithBasisElement::from_basis_element(2, BigInt::from(1)),
        );

        let f = ModuleWithBasisMorphism::new(source.clone(), target, basis_action);

        // Apply to 3*e_0 + 5*e_1
        let input = ModuleWithBasisElement::from_terms(vec![
            (0, BigInt::from(3)),
            (1, BigInt::from(5)),
        ]);

        let output = f.apply(&input);

        // Should get 3*(2*f_0 + f_1) + 5*f_2 = 6*f_0 + 3*f_1 + 5*f_2
        assert_eq!(output.coefficient(&0), BigInt::from(6));
        assert_eq!(output.coefficient(&1), BigInt::from(3));
        assert_eq!(output.coefficient(&2), BigInt::from(5));
    }

    #[test]
    fn test_identity_morphism() {
        let base_ring = BigInt::from(0);
        let module = FreeModuleWithBasis::standard(base_ring, 3);

        let id = ModuleWithBasisMorphism::identity(module.clone());
        assert!(id.is_identity());

        let e1 = module.basis_element(&1).unwrap();
        let image = id.apply(&e1);
        assert_eq!(image, e1);
    }

    #[test]
    fn test_morphism_composition() {
        let base_ring = BigInt::from(0);
        let module = FreeModuleWithBasis::standard(base_ring, 3);

        // f: e_i ↦ e_{i+1 mod 3} (cyclic permutation)
        let mut f_action = BTreeMap::new();
        for i in 0..3 {
            f_action.insert(
                i,
                ModuleWithBasisElement::from_basis_element((i + 1) % 3, BigInt::from(1)),
            );
        }
        let f = ModuleWithBasisMorphism::new(module.clone(), module.clone(), f_action);

        // f ∘ f should give e_i ↦ e_{i+2 mod 3}
        let f2 = f.compose_endo(&f);

        let e0 = module.basis_element(&0).unwrap();
        let image = f2.apply(&e0);
        assert_eq!(image.coefficient(&2), BigInt::from(1));
        assert_eq!(image.coefficient(&0), BigInt::from(0));
    }

    #[test]
    fn test_morphism_power() {
        let base_ring = BigInt::from(0);
        let module = FreeModuleWithBasis::standard(base_ring, 3);

        // f: e_i ↦ e_{i+1 mod 3}
        let mut f_action = BTreeMap::new();
        for i in 0..3 {
            f_action.insert(
                i,
                ModuleWithBasisElement::from_basis_element((i + 1) % 3, BigInt::from(1)),
            );
        }
        let f = ModuleWithBasisMorphism::new(module.clone(), module.clone(), f_action);

        // f^3 should be identity
        let f3 = f.power(3);
        assert!(f3.is_identity());
    }

    #[test]
    fn test_trace() {
        let base_ring = BigInt::from(0);
        let module = FreeModuleWithBasis::standard(base_ring, 3);

        // Morphism with e_0 ↦ 2*e_0, e_1 ↦ 3*e_1, e_2 ↦ -e_2
        let mut action = BTreeMap::new();
        action.insert(0, ModuleWithBasisElement::from_basis_element(0, BigInt::from(2)));
        action.insert(1, ModuleWithBasisElement::from_basis_element(1, BigInt::from(3)));
        action.insert(2, ModuleWithBasisElement::from_basis_element(2, BigInt::from(-1)));

        let f = ModuleWithBasisMorphism::new(module.clone(), module, action);

        // Trace should be 2 + 3 + (-1) = 4
        assert_eq!(f.trace(), BigInt::from(4));
    }

    #[test]
    fn test_matrix_representation() {
        let base_ring = BigInt::from(0);
        let source = FreeModuleWithBasis::standard(base_ring.clone(), 2);
        let target = FreeModuleWithBasis::standard(base_ring, 3);

        // e_0 ↦ 2*f_0 + f_1, e_1 ↦ 3*f_2
        let mut basis_action = BTreeMap::new();
        basis_action.insert(
            0,
            ModuleWithBasisElement::from_terms(vec![
                (0, BigInt::from(2)),
                (1, BigInt::from(1)),
            ]),
        );
        basis_action.insert(
            1,
            ModuleWithBasisElement::from_basis_element(2, BigInt::from(3)),
        );

        let f = ModuleWithBasisMorphism::new(source, target, basis_action);
        let matrix = f.matrix().unwrap();

        // Matrix should be:
        // [[2, 0],
        //  [1, 0],
        //  [0, 3]]
        assert_eq!(matrix[0][0], BigInt::from(2));
        assert_eq!(matrix[0][1], BigInt::from(0));
        assert_eq!(matrix[1][0], BigInt::from(1));
        assert_eq!(matrix[1][1], BigInt::from(0));
        assert_eq!(matrix[2][0], BigInt::from(0));
        assert_eq!(matrix[2][1], BigInt::from(3));
    }

    #[test]
    fn test_image() {
        let base_ring = BigInt::from(0);
        let source = FreeModuleWithBasis::standard(base_ring.clone(), 2);
        let target = FreeModuleWithBasis::standard(base_ring, 3);

        let mut basis_action = BTreeMap::new();
        basis_action.insert(0, ModuleWithBasisElement::from_basis_element(0, BigInt::from(1)));
        basis_action.insert(1, ModuleWithBasisElement::from_basis_element(0, BigInt::from(2)));

        let f = ModuleWithBasisMorphism::new(source, target, basis_action);
        let image_basis = f.image();

        assert_eq!(image_basis.len(), 2);
    }
}
