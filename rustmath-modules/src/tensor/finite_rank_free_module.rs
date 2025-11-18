//! # Finite Rank Free Modules
//!
//! This module provides finite rank free module structures,
//! corresponding to SageMath's `sage.tensor.modules.finite_rank_free_module`.

use super::free_module_basis::FreeModuleBasis;
use std::marker::PhantomData;

/// Abstract finite rank free module trait
pub trait FiniteRankFreeModuleAbstract {
    type Ring;

    fn rank(&self) -> usize;
}

/// A finite rank free module over a ring
pub struct FiniteRankFreeModule<R> {
    rank: usize,
    basis: Option<FreeModuleBasis<Self>>,
    ring: PhantomData<R>,
}

impl<R> FiniteRankFreeModule<R> {
    pub fn new(rank: usize) -> Self {
        Self {
            rank,
            basis: None,
            ring: PhantomData,
        }
    }

    pub fn with_basis(rank: usize, symbol: String) -> Self {
        Self {
            rank,
            basis: Some(FreeModuleBasis::new(rank, symbol)),
            ring: PhantomData,
        }
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn basis(&self) -> Option<&FreeModuleBasis<Self>> {
        self.basis.as_ref()
    }
}

impl<R> FiniteRankFreeModuleAbstract for FiniteRankFreeModule<R> {
    type Ring = R;

    fn rank(&self) -> usize {
        self.rank
    }
}

/// Dual of a finite rank free module
pub struct FiniteRankDualFreeModule<R> {
    base_rank: usize,
    ring: PhantomData<R>,
}

impl<R> FiniteRankDualFreeModule<R> {
    pub fn new(base_rank: usize) -> Self {
        Self {
            base_rank,
            ring: PhantomData,
        }
    }

    pub fn rank(&self) -> usize {
        self.base_rank
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finite_rank_free_module() {
        let module: FiniteRankFreeModule<i32> = FiniteRankFreeModule::new(3);
        assert_eq!(module.rank(), 3);
    }

    #[test]
    fn test_with_basis() {
        let module: FiniteRankFreeModule<i32> =
            FiniteRankFreeModule::with_basis(4, "e".to_string());
        assert_eq!(module.rank(), 4);
        assert!(module.basis().is_some());
    }

    #[test]
    fn test_dual_module() {
        let dual: FiniteRankDualFreeModule<i32> = FiniteRankDualFreeModule::new(5);
        assert_eq!(dual.rank(), 5);
    }
}
