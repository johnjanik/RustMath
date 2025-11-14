//! Free modules over rings

use rustmath_core::Ring;
use crate::module::Module;
use crate::free_module_element::FreeModuleElement;
use std::marker::PhantomData;

/// A free module over a ring
#[derive(Clone, Debug)]
pub struct FreeModule<R: Ring> {
    base_ring: R,
    rank: usize,
    _phantom: PhantomData<R>,
}

impl<R: Ring> FreeModule<R> {
    /// Create a new free module of given rank over a ring
    pub fn new(base_ring: R, rank: usize) -> Self {
        Self {
            base_ring,
            rank,
            _phantom: PhantomData,
        }
    }

    /// Get the base ring
    pub fn base_ring(&self) -> &R {
        &self.base_ring
    }

    /// Get the rank (number of generators)
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Create the zero element
    pub fn zero(&self) -> FreeModuleElement<R> {
        FreeModuleElement::zero(self.rank)
    }

    /// Create a basis element (i-th standard basis vector)
    pub fn basis_element(&self, i: usize) -> Option<FreeModuleElement<R>> {
        if i < self.rank {
            let mut coords = vec![R::zero(); self.rank];
            coords[i] = R::one();
            Some(FreeModuleElement::new(coords))
        } else {
            None
        }
    }

    /// Get all basis elements
    pub fn basis(&self) -> Vec<FreeModuleElement<R>> {
        (0..self.rank)
            .map(|i| self.basis_element(i).unwrap())
            .collect()
    }
}

impl<R: Ring> Module for FreeModule<R> {
    type BaseRing = R;
    type Element = FreeModuleElement<R>;

    fn base_ring(&self) -> &Self::BaseRing {
        &self.base_ring
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn zero(&self) -> Self::Element {
        self.zero()
    }

    fn is_zero(&self, elem: &Self::Element) -> bool {
        elem.is_zero(&self.base_ring)
    }

    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.add(b, &self.base_ring)
    }

    fn negate(&self, a: &Self::Element) -> Self::Element {
        a.negate(&self.base_ring)
    }

    fn scalar_mul(&self, scalar: &Self::BaseRing, elem: &Self::Element) -> Self::Element {
        elem.scalar_mul(scalar, &self.base_ring)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use num_traits::{Zero, One};

    #[test]
    fn test_free_module_creation() {
        let module = FreeModule::new(BigInt::zero(), 3);
        assert_eq!(module.rank(), 3);
    }

    #[test]
    fn test_basis_element() {
        let module = FreeModule::new(BigInt::zero(), 3);
        let basis = module.basis();
        assert_eq!(basis.len(), 3);
    }
}
