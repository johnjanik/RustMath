//! Quotient modules M/N

use rustmath_core::Ring;
use crate::free_module_element::FreeModuleElement;

/// A quotient module M/N
#[derive(Clone, Debug)]
pub struct QuotientModule<R: Ring> {
    ambient_rank: usize,
    /// Generators of the submodule N we're quotienting by
    submodule_generators: Vec<FreeModuleElement<R>>,
}

impl<R: Ring> QuotientModule<R> {
    pub fn new(ambient_rank: usize, submodule_generators: Vec<FreeModuleElement<R>>) -> Self {
        for gen in &submodule_generators {
            assert_eq!(gen.dimension(), ambient_rank);
        }
        Self {
            ambient_rank,
            submodule_generators,
        }
    }

    pub fn ambient_rank(&self) -> usize {
        self.ambient_rank
    }

    pub fn submodule_generators(&self) -> &[FreeModuleElement<R>] {
        &self.submodule_generators
    }

    /// Lift an element from quotient to ambient module
    /// (Any choice of representative)
    pub fn lift(&self, element: &FreeModuleElement<R>) -> FreeModuleElement<R> {
        element.clone()
    }

    /// Check if two elements are equivalent in the quotient
    pub fn are_equivalent(&self, a: &FreeModuleElement<R>, b: &FreeModuleElement<R>) -> bool {
        // Simplified: would need to check if a-b is in span of generators
        a == b
    }
}
