//! Homomorphism spaces (Hom sets) for modules with basis
//!
//! This module implements Hom(M, N), the space of module homomorphisms from M to N.
//! For modules with basis, Hom(M, N) is itself a module with basis indexed by pairs (i, j)
//! where i is a basis index of M and j is a basis index of N.

use rustmath_core::{Ring, Parent, ParentWithBasis};
use crate::with_basis::element::ModuleWithBasisElement;
use crate::with_basis::parent::ModuleWithBasis;
use crate::with_basis::morphism::ModuleWithBasisMorphism;
use std::collections::BTreeMap;
use std::fmt::Debug;

/// Index for basis elements of Hom(M, N)
///
/// A basis element is indexed by (i, j) representing the morphism that
/// sends e_i to f_j and all other basis elements to 0.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct HomIndex<I1, I2> {
    pub source_index: I1,
    pub target_index: I2,
}

impl<I1, I2> HomIndex<I1, I2> {
    pub fn new(source_index: I1, target_index: I2) -> Self {
        HomIndex {
            source_index,
            target_index,
        }
    }
}

/// Homomorphism space Hom(M, N)
///
/// The space of all module homomorphisms from M to N.
/// For finite-dimensional modules with bases, this is itself a module
/// with dimension dim(M) × dim(N).
#[derive(Clone, Debug)]
pub struct HomSpace<M, N>
where
    M: ModuleWithBasis,
    N: ModuleWithBasis<BaseRing = M::BaseRing>,
{
    source: M,
    target: N,
}

impl<M, N> HomSpace<M, N>
where
    M: ModuleWithBasis,
    N: ModuleWithBasis<BaseRing = M::BaseRing>,
{
    /// Create a new Hom space
    pub fn new(source: M, target: N) -> Self {
        HomSpace { source, target }
    }

    /// Get the source module
    pub fn source(&self) -> &M {
        &self.source
    }

    /// Get the target module
    pub fn target(&self) -> &N {
        &self.target
    }

    /// Create a morphism from a Hom space element
    ///
    /// The element represents coefficients in the standard basis of Hom(M, N)
    pub fn element_to_morphism(
        &self,
        elem: &ModuleWithBasisElement<HomIndex<M::BasisIndex, N::BasisIndex>, M::BaseRing>,
    ) -> ModuleWithBasisMorphism<M::BasisIndex, M::BaseRing, M, N>
    where
        M::BasisIndex: Ord + Clone + Debug,
        N::BasisIndex: Ord + Clone + Debug,
        M::BaseRing: Ring,
        M::Element: Clone,
        N::Element: Clone,
        N::Element: From<ModuleWithBasisElement<N::BasisIndex, M::BaseRing>>,
    {
        // Build the morphism by specifying where each source basis element goes
        let mut basis_action = BTreeMap::new();

        for source_idx in self.source.basis_indices() {
            let mut target_elem = ModuleWithBasisElement::zero();

            // Collect all terms that affect this source basis element
            for (hom_idx, coeff) in elem.items() {
                if hom_idx.source_index == source_idx {
                    target_elem.add_term(hom_idx.target_index.clone(), coeff.clone());
                }
            }

            basis_action.insert(source_idx, target_elem);
        }

        ModuleWithBasisMorphism::new(self.source.clone(), self.target.clone(), basis_action)
    }

    /// Create a Hom space element from a morphism
    ///
    /// Convert a morphism to its representation in the standard basis of Hom(M, N)
    pub fn morphism_to_element(
        &self,
        morphism: &ModuleWithBasisMorphism<M::BasisIndex, M::BaseRing, M, N>,
    ) -> ModuleWithBasisElement<HomIndex<M::BasisIndex, N::BasisIndex>, M::BaseRing>
    where
        M::BasisIndex: Ord + Clone + Debug,
        N::BasisIndex: Ord + Clone + Debug,
        M::BaseRing: Ring,
    {
        let mut terms = Vec::new();

        for source_idx in self.source.basis_indices() {
            if let Some(image) = morphism.on_basis(&source_idx) {
                for (target_idx, coeff) in image.items() {
                    terms.push((
                        HomIndex::new(source_idx.clone(), target_idx.clone()),
                        coeff.clone(),
                    ));
                }
            }
        }

        ModuleWithBasisElement::from_terms(terms)
    }

    /// Compute the zero morphism
    pub fn zero_morphism(&self) -> ModuleWithBasisMorphism<M::BasisIndex, M::BaseRing, M, N>
    where
        M::BasisIndex: Ord + Clone + Debug,
        N::BasisIndex: Ord + Clone + Debug,
        M::BaseRing: Ring,
        M::Element: Clone,
        N::Element: Clone,
    {
        ModuleWithBasisMorphism::zero(self.source.clone(), self.target.clone())
    }

    /// Create the basis morphism E_ij that sends e_i to f_j
    pub fn basis_morphism(
        &self,
        source_idx: &M::BasisIndex,
        target_idx: &N::BasisIndex,
    ) -> Option<ModuleWithBasisMorphism<M::BasisIndex, M::BaseRing, M, N>>
    where
        M::BasisIndex: Ord + Clone + Debug,
        N::BasisIndex: Ord + Clone + Debug,
        M::BaseRing: Ring,
        M::Element: Clone,
        N::Element: Clone,
    {
        let mut basis_action = BTreeMap::new();

        for idx in self.source.basis_indices() {
            if &idx == source_idx {
                basis_action.insert(
                    idx,
                    ModuleWithBasisElement::from_basis_element(
                        target_idx.clone(),
                        M::BaseRing::one(),
                    ),
                );
            } else {
                basis_action.insert(idx, ModuleWithBasisElement::zero());
            }
        }

        Some(ModuleWithBasisMorphism::new(
            self.source.clone(),
            self.target.clone(),
            basis_action,
        ))
    }

    /// Compose with a morphism on the left: Hom(M, N) → Hom(M, P) via g ↦ h ∘ g
    pub fn compose_left<P>(
        &self,
        h: &ModuleWithBasisMorphism<N::BasisIndex, M::BaseRing, N, P>,
    ) -> HomSpace<M, P>
    where
        P: ModuleWithBasis<BaseRing = M::BaseRing>,
        N::BasisIndex: Ord + Clone + Debug,
        P::BasisIndex: Ord + Clone + Debug,
        M::BaseRing: Ring,
    {
        HomSpace::new(self.source.clone(), h.codomain().clone())
    }

    /// Compose with a morphism on the right: Hom(M, N) → Hom(L, N) via g ↦ g ∘ h
    pub fn compose_right<L>(
        &self,
        _h: &ModuleWithBasisMorphism<L::BasisIndex, M::BaseRing, L, M>,
    ) -> HomSpace<L, N>
    where
        L: ModuleWithBasis<BaseRing = M::BaseRing>,
        L::BasisIndex: Ord + Clone + Debug,
        M::BasisIndex: Ord + Clone + Debug,
        M::BaseRing: Ring,
        L::Element: Clone,
    {
        // Placeholder - need proper L type from morphism
        unimplemented!("compose_right requires access to L")
    }
}

impl<M, N> Parent for HomSpace<M, N>
where
    M: ModuleWithBasis,
    N: ModuleWithBasis<BaseRing = M::BaseRing>,
    M::BasisIndex: Ord + Clone + Debug,
    N::BasisIndex: Ord + Clone + Debug,
    M::BaseRing: Ring,
{
    type Element = ModuleWithBasisElement<HomIndex<M::BasisIndex, N::BasisIndex>, M::BaseRing>;

    fn contains(&self, _element: &Self::Element) -> bool {
        true
    }

    fn zero(&self) -> Option<Self::Element> {
        Some(ModuleWithBasisElement::zero())
    }
}

impl<M, N> ParentWithBasis for HomSpace<M, N>
where
    M: ModuleWithBasis,
    N: ModuleWithBasis<BaseRing = M::BaseRing>,
    M::BasisIndex: Ord + Clone + Debug,
    N::BasisIndex: Ord + Clone + Debug,
    M::BaseRing: Ring,
{
    type BasisIndex = HomIndex<M::BasisIndex, N::BasisIndex>;

    fn dimension(&self) -> Option<usize> {
        match (self.source.dimension(), self.target.dimension()) {
            (Some(d1), Some(d2)) => Some(d1 * d2),
            _ => None,
        }
    }

    fn basis_element(&self, index: &Self::BasisIndex) -> Option<Self::Element> {
        Some(ModuleWithBasisElement::from_basis_element(
            index.clone(),
            M::BaseRing::one(),
        ))
    }

    fn basis_indices(&self) -> Vec<Self::BasisIndex> {
        let mut indices = Vec::new();

        for source_idx in self.source.basis_indices() {
            for target_idx in self.target.basis_indices() {
                indices.push(HomIndex::new(source_idx.clone(), target_idx.clone()));
            }
        }

        indices
    }
}

impl<M, N> ModuleWithBasis for HomSpace<M, N>
where
    M: ModuleWithBasis,
    N: ModuleWithBasis<BaseRing = M::BaseRing>,
    M::BasisIndex: Ord + Clone + Debug,
    N::BasisIndex: Ord + Clone + Debug,
    M::BaseRing: Ring,
    M::Element: Clone,
    N::Element: Clone,
{
    type BaseRing = M::BaseRing;

    fn base_ring(&self) -> &Self::BaseRing {
        self.source.base_ring()
    }

    fn module_zero(&self) -> Self::Element {
        ModuleWithBasisElement::zero()
    }

    fn is_module_zero(&self, elem: &Self::Element) -> bool {
        elem.is_zero()
    }

    fn module_add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.add(b)
    }

    fn module_negate(&self, a: &Self::Element) -> Self::Element {
        a.negate()
    }

    fn module_scalar_mul(&self, scalar: &Self::BaseRing, elem: &Self::Element) -> Self::Element {
        elem.scalar_mul(scalar)
    }

    fn from_basis_element(&self, index: &Self::BasisIndex, coefficient: Self::BaseRing) -> Option<Self::Element> {
        Some(ModuleWithBasisElement::from_basis_element(
            index.clone(),
            coefficient,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::with_basis::parent::FreeModuleWithBasis;
    use num_bigint::BigInt;

    #[test]
    fn test_homspace_creation() {
        let base_ring = BigInt::from(0);
        let m = FreeModuleWithBasis::standard(base_ring.clone(), 2);
        let n = FreeModuleWithBasis::standard(base_ring, 3);

        let hom = HomSpace::new(m, n);

        // dim Hom(M, N) = dim(M) * dim(N) = 2 * 3 = 6
        assert_eq!(hom.dimension(), Some(6));
    }

    #[test]
    fn test_hom_basis_indices() {
        let base_ring = BigInt::from(0);
        let m = FreeModuleWithBasis::standard(base_ring.clone(), 2);
        let n = FreeModuleWithBasis::standard(base_ring, 2);

        let hom = HomSpace::new(m, n);
        let indices = hom.basis_indices();

        assert_eq!(indices.len(), 4);

        // Check that we have all combinations (0,0), (0,1), (1,0), (1,1)
        assert!(indices.contains(&HomIndex::new(0, 0)));
        assert!(indices.contains(&HomIndex::new(0, 1)));
        assert!(indices.contains(&HomIndex::new(1, 0)));
        assert!(indices.contains(&HomIndex::new(1, 1)));
    }

    #[test]
    fn test_morphism_to_element_conversion() {
        let base_ring = BigInt::from(0);
        let source = FreeModuleWithBasis::standard(base_ring.clone(), 2);
        let target = FreeModuleWithBasis::standard(base_ring, 2);

        let hom = HomSpace::new(source.clone(), target.clone());

        // Create morphism: e_0 ↦ 2*f_0 + 3*f_1, e_1 ↦ 5*f_1
        let mut basis_action = BTreeMap::new();
        basis_action.insert(
            0,
            ModuleWithBasisElement::from_terms(vec![
                (0, BigInt::from(2)),
                (1, BigInt::from(3)),
            ]),
        );
        basis_action.insert(
            1,
            ModuleWithBasisElement::from_basis_element(1, BigInt::from(5)),
        );

        let morphism = ModuleWithBasisMorphism::new(source, target, basis_action);

        let elem = hom.morphism_to_element(&morphism);

        // Check coefficients
        assert_eq!(elem.coefficient(&HomIndex::new(0, 0)), BigInt::from(2));
        assert_eq!(elem.coefficient(&HomIndex::new(0, 1)), BigInt::from(3));
        assert_eq!(elem.coefficient(&HomIndex::new(1, 0)), BigInt::from(0));
        assert_eq!(elem.coefficient(&HomIndex::new(1, 1)), BigInt::from(5));
    }

    #[test]
    fn test_zero_morphism() {
        let base_ring = BigInt::from(0);
        let m = FreeModuleWithBasis::standard(base_ring.clone(), 2);
        let n = FreeModuleWithBasis::standard(base_ring, 3);

        let hom = HomSpace::new(m.clone(), n);

        let zero = hom.zero_morphism();
        assert!(zero.is_zero());

        let e0 = m.basis_element(&0).unwrap();
        let image = zero.apply(&e0);
        assert!(image.is_zero());
    }

    #[test]
    fn test_basis_morphism() {
        let base_ring = BigInt::from(0);
        let m = FreeModuleWithBasis::standard(base_ring.clone(), 2);
        let n = FreeModuleWithBasis::standard(base_ring, 3);

        let hom = HomSpace::new(m.clone(), n.clone());

        // Create E_01: e_0 ↦ f_1, other basis elements ↦ 0
        let e_01 = hom.basis_morphism(&0, &1).unwrap();

        let e0 = m.basis_element(&0).unwrap();
        let e1 = m.basis_element(&1).unwrap();

        let image0 = e_01.apply(&e0);
        let image1 = e_01.apply(&e1);

        assert_eq!(image0.coefficient(&1), BigInt::from(1));
        assert_eq!(image0.coefficient(&0), BigInt::from(0));
        assert!(image1.is_zero());
    }

    #[test]
    fn test_hom_module_operations() {
        let base_ring = BigInt::from(0);
        let m = FreeModuleWithBasis::standard(base_ring.clone(), 2);
        let n = FreeModuleWithBasis::standard(base_ring, 2);

        let hom = HomSpace::new(m, n);

        let elem1 = ModuleWithBasisElement::from_terms(vec![
            (HomIndex::new(0, 0), BigInt::from(2)),
            (HomIndex::new(1, 1), BigInt::from(3)),
        ]);

        let elem2 = ModuleWithBasisElement::from_terms(vec![
            (HomIndex::new(0, 1), BigInt::from(5)),
            (HomIndex::new(1, 1), BigInt::from(1)),
        ]);

        // Test addition
        let sum = hom.module_add(&elem1, &elem2);
        assert_eq!(sum.coefficient(&HomIndex::new(0, 0)), BigInt::from(2));
        assert_eq!(sum.coefficient(&HomIndex::new(0, 1)), BigInt::from(5));
        assert_eq!(sum.coefficient(&HomIndex::new(1, 1)), BigInt::from(4));

        // Test scalar multiplication
        let scaled = hom.module_scalar_mul(&BigInt::from(2), &elem1);
        assert_eq!(scaled.coefficient(&HomIndex::new(0, 0)), BigInt::from(4));
        assert_eq!(scaled.coefficient(&HomIndex::new(1, 1)), BigInt::from(6));
    }
}
