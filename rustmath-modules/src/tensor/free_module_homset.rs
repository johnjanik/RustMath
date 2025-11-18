//! # Free Module Homomorphism Sets
//!
//! This module provides homomorphism sets (Hom sets) for free modules,
//! corresponding to SageMath's `sage.tensor.modules.free_module_homset`.

use std::marker::PhantomData;

/// Set of homomorphisms between two free modules
///
/// Hom(M, N) is the set of all module homomorphisms from M to N
pub struct FreeModuleHomset<R> {
    domain_rank: usize,
    codomain_rank: usize,
    ring: PhantomData<R>,
}

impl<R> FreeModuleHomset<R> {
    pub fn new(domain_rank: usize, codomain_rank: usize) -> Self {
        Self {
            domain_rank,
            codomain_rank,
            ring: PhantomData,
        }
    }

    pub fn domain_rank(&self) -> usize {
        self.domain_rank
    }

    pub fn codomain_rank(&self) -> usize {
        self.codomain_rank
    }

    /// Dimension of Hom(M, N) as a free module
    pub fn dimension(&self) -> usize {
        self.domain_rank * self.codomain_rank
    }
}

/// Set of endomorphisms of a free module
///
/// End(M) = Hom(M, M)
pub struct FreeModuleEndset<R> {
    rank: usize,
    ring: PhantomData<R>,
}

impl<R> FreeModuleEndset<R> {
    pub fn new(rank: usize) -> Self {
        Self {
            rank,
            ring: PhantomData,
        }
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn dimension(&self) -> usize {
        self.rank * self.rank
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_homset() {
        let hom: FreeModuleHomset<i32> = FreeModuleHomset::new(3, 2);

        assert_eq!(hom.domain_rank(), 3);
        assert_eq!(hom.codomain_rank(), 2);
        assert_eq!(hom.dimension(), 6);
    }

    #[test]
    fn test_endset() {
        let end: FreeModuleEndset<i32> = FreeModuleEndset::new(4);

        assert_eq!(end.rank(), 4);
        assert_eq!(end.dimension(), 16);
    }
}
