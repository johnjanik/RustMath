//! Dual modules for modules with basis
//!
//! This module implements the dual space of a module with basis.
//! For a module M over R with basis {e_i}, the dual module M* consists of
//! linear functionals f: M → R, with dual basis {e_i*} where e_i*(e_j) = δ_ij.

use rustmath_core::{Ring, Parent, ParentWithBasis};
use crate::with_basis::element::ModuleWithBasisElement;
use crate::with_basis::parent::ModuleWithBasis;
use std::fmt::Debug;
use std::collections::BTreeMap;

/// Dual module of a module with basis
///
/// The dual module M* has the same basis indices as M, but represents
/// linear functionals rather than vectors.
#[derive(Clone, Debug)]
pub struct DualModule<M>
where
    M: ModuleWithBasis,
{
    base_module: M,
}

impl<M> DualModule<M>
where
    M: ModuleWithBasis,
{
    /// Create the dual module
    pub fn new(base_module: M) -> Self {
        DualModule { base_module }
    }

    /// Get the base module
    pub fn base(&self) -> &M {
        &self.base_module
    }

    /// Apply a dual element (functional) to a primal element
    ///
    /// For dual element with coefficients α_i and primal element with coefficients x_i,
    /// the result is Σ α_i * x_i (using the canonical pairing)
    pub fn apply(
        &self,
        dual_elem: &ModuleWithBasisElement<M::BasisIndex, M::BaseRing>,
        primal_elem: &M::Element,
    ) -> M::BaseRing
    where
        M::BasisIndex: Ord + Clone,
        M::BaseRing: Ring,
        M::Element: Clone,
    {
        // This is a placeholder - proper implementation would need to extract coefficients
        // from the primal element
        M::BaseRing::zero()
    }

    /// Create the dual element corresponding to a basis element
    ///
    /// The dual basis element e_i* is defined by e_i*(e_j) = δ_ij
    pub fn dual_basis_element(&self, index: &M::BasisIndex) -> ModuleWithBasisElement<M::BasisIndex, M::BaseRing>
    where
        M::BasisIndex: Ord + Clone,
        M::BaseRing: Ring,
    {
        ModuleWithBasisElement::from_basis_element(index.clone(), M::BaseRing::one())
    }

    /// Compute the transpose of a morphism
    ///
    /// For f: M → N, the transpose f*: N* → M* is defined by
    /// f*(α)(x) = α(f(x)) for α ∈ N*, x ∈ M
    pub fn transpose_morphism<N>(
        &self,
        _source: &DualModule<N>,
        _morphism_action: &BTreeMap<M::BasisIndex, ModuleWithBasisElement<N::BasisIndex, M::BaseRing>>,
    ) -> BTreeMap<N::BasisIndex, ModuleWithBasisElement<M::BasisIndex, M::BaseRing>>
    where
        N: ModuleWithBasis<BaseRing = M::BaseRing>,
        M::BasisIndex: Ord + Clone,
        N::BasisIndex: Ord + Clone,
        M::BaseRing: Ring,
    {
        // Placeholder - proper implementation would compute transpose
        BTreeMap::new()
    }
}

impl<M> Parent for DualModule<M>
where
    M: ModuleWithBasis,
    M::BasisIndex: Ord + Clone + Debug,
    M::BaseRing: Ring,
{
    type Element = ModuleWithBasisElement<M::BasisIndex, M::BaseRing>;

    fn contains(&self, _element: &Self::Element) -> bool {
        true
    }

    fn zero(&self) -> Option<Self::Element> {
        Some(ModuleWithBasisElement::zero())
    }
}

impl<M> ParentWithBasis for DualModule<M>
where
    M: ModuleWithBasis,
    M::BasisIndex: Ord + Clone + Debug,
    M::BaseRing: Ring,
{
    type BasisIndex = M::BasisIndex;

    fn dimension(&self) -> Option<usize> {
        self.base_module.dimension()
    }

    fn basis_element(&self, index: &Self::BasisIndex) -> Option<Self::Element> {
        Some(self.dual_basis_element(index))
    }

    fn basis_indices(&self) -> Vec<Self::BasisIndex> {
        self.base_module.basis_indices()
    }
}

impl<M> ModuleWithBasis for DualModule<M>
where
    M: ModuleWithBasis,
    M::BasisIndex: Ord + Clone + Debug,
    M::BaseRing: Ring,
    M::Element: Clone,
{
    type BaseRing = M::BaseRing;

    fn base_ring(&self) -> &Self::BaseRing {
        self.base_module.base_ring()
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

/// Canonical pairing between a module and its dual
///
/// For x ∈ M and α ∈ M*, computes α(x)
pub fn canonical_pairing<M>(
    dual_elem: &ModuleWithBasisElement<M::BasisIndex, M::BaseRing>,
    primal_elem: &ModuleWithBasisElement<M::BasisIndex, M::BaseRing>,
) -> M::BaseRing
where
    M: ModuleWithBasis,
    M::BasisIndex: Ord + Clone,
    M::BaseRing: Ring,
{
    let mut result = M::BaseRing::zero();

    // Compute Σ α_i * x_i
    for (idx, alpha_coeff) in dual_elem.items() {
        let x_coeff = primal_elem.coefficient(idx);
        result = result + (alpha_coeff.clone() * x_coeff);
    }

    result
}

/// Double dual isomorphism
///
/// For finite-dimensional modules, there is a canonical isomorphism M → M**
/// sending x ∈ M to the evaluation functional eval_x ∈ M** defined by
/// eval_x(α) = α(x)
pub fn double_dual_isomorphism<M>(
    module: &M,
    elem: &M::Element,
) -> ModuleWithBasisElement<M::BasisIndex, M::BaseRing>
where
    M: ModuleWithBasis,
    M::BasisIndex: Ord + Clone,
    M::BaseRing: Ring,
    M::Element: Clone,
{
    // For modules with basis, the double dual is canonically identified with M
    // This is a placeholder - proper implementation would create the evaluation functional
    ModuleWithBasisElement::zero()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::with_basis::parent::FreeModuleWithBasis;
    use num_bigint::BigInt;

    #[test]
    fn test_dual_module_creation() {
        let base_ring = BigInt::from(0);
        let module = FreeModuleWithBasis::standard(base_ring, 3);

        let dual = DualModule::new(module);

        assert_eq!(dual.dimension(), Some(3));
    }

    #[test]
    fn test_dual_basis() {
        let base_ring = BigInt::from(0);
        let module = FreeModuleWithBasis::standard(base_ring, 3);

        let dual = DualModule::new(module);

        let e0_star = dual.dual_basis_element(&0);
        assert_eq!(e0_star.coefficient(&0), BigInt::from(1));
        assert_eq!(e0_star.coefficient(&1), BigInt::from(0));
    }

    #[test]
    fn test_canonical_pairing() {
        let base_ring = BigInt::from(0);
        let module = FreeModuleWithBasis::standard(base_ring, 3);

        // Primal element: 3*e_0 + 5*e_1 + 2*e_2
        let primal = ModuleWithBasisElement::from_terms(vec![
            (0, BigInt::from(3)),
            (1, BigInt::from(5)),
            (2, BigInt::from(2)),
        ]);

        // Dual element: 2*e_0* + 1*e_1* + 4*e_2*
        let dual = ModuleWithBasisElement::from_terms(vec![
            (0, BigInt::from(2)),
            (1, BigInt::from(1)),
            (2, BigInt::from(4)),
        ]);

        let result = canonical_pairing::<FreeModuleWithBasis<usize, BigInt>>(&dual, &primal);

        // Result should be 2*3 + 1*5 + 4*2 = 6 + 5 + 8 = 19
        assert_eq!(result, BigInt::from(19));
    }

    #[test]
    fn test_dual_basis_pairing() {
        let base_ring = BigInt::from(0);
        let module = FreeModuleWithBasis::standard(base_ring, 3);
        let dual = DualModule::new(module.clone());

        // Test that e_i*(e_j) = δ_ij
        let e0 = module.basis_element(&0).unwrap();
        let e1 = module.basis_element(&1).unwrap();

        let e0_star = dual.dual_basis_element(&0);
        let e1_star = dual.dual_basis_element(&1);

        assert_eq!(
            canonical_pairing::<FreeModuleWithBasis<usize, BigInt>>(&e0_star, &e0),
            BigInt::from(1)
        );
        assert_eq!(
            canonical_pairing::<FreeModuleWithBasis<usize, BigInt>>(&e0_star, &e1),
            BigInt::from(0)
        );
        assert_eq!(
            canonical_pairing::<FreeModuleWithBasis<usize, BigInt>>(&e1_star, &e0),
            BigInt::from(0)
        );
        assert_eq!(
            canonical_pairing::<FreeModuleWithBasis<usize, BigInt>>(&e1_star, &e1),
            BigInt::from(1)
        );
    }

    #[test]
    fn test_dual_module_operations() {
        let base_ring = BigInt::from(0);
        let module = FreeModuleWithBasis::standard(base_ring, 3);
        let dual = DualModule::new(module);

        let alpha = ModuleWithBasisElement::from_terms(vec![
            (0, BigInt::from(2)),
            (1, BigInt::from(3)),
        ]);

        let beta = ModuleWithBasisElement::from_terms(vec![
            (1, BigInt::from(5)),
            (2, BigInt::from(1)),
        ]);

        // Test addition
        let sum = dual.module_add(&alpha, &beta);
        assert_eq!(sum.coefficient(&0), BigInt::from(2));
        assert_eq!(sum.coefficient(&1), BigInt::from(8));
        assert_eq!(sum.coefficient(&2), BigInt::from(1));

        // Test scalar multiplication
        let scaled = dual.module_scalar_mul(&BigInt::from(3), &alpha);
        assert_eq!(scaled.coefficient(&0), BigInt::from(6));
        assert_eq!(scaled.coefficient(&1), BigInt::from(9));
    }
}
