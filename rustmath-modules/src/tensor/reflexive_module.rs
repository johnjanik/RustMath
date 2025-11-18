//! # Reflexive Modules
//!
//! This module provides reflexive module structures,
//! corresponding to SageMath's `sage.tensor.modules.reflexive_module`.
//!
//! A reflexive module M satisfies M ≅ M** (isomorphic to its double dual).

use std::marker::PhantomData;

/// Abstract reflexive module trait
pub trait ReflexiveModule {
    /// The ring type
    type Ring;

    /// Get the rank of the module
    fn rank(&self) -> usize;

    /// Get the dual module
    fn dual(&self) -> Self where Self: Sized;
}

/// Base reflexive module
pub struct ReflexiveModuleBase<R> {
    rank: usize,
    ring: PhantomData<R>,
}

impl<R> ReflexiveModuleBase<R> {
    pub fn new(rank: usize) -> Self {
        Self {
            rank,
            ring: PhantomData,
        }
    }

    pub fn rank(&self) -> usize {
        self.rank
    }
}

/// Dual of a reflexive module
pub struct ReflexiveModuleDual<R> {
    base_rank: usize,
    ring: PhantomData<R>,
}

impl<R> ReflexiveModuleDual<R> {
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

/// Tensor product of reflexive modules
pub struct ReflexiveModuleTensor<R> {
    rank: usize,
    ring: PhantomData<R>,
}

impl<R> ReflexiveModuleTensor<R> {
    pub fn new(rank: usize) -> Self {
        Self {
            rank,
            ring: PhantomData,
        }
    }

    pub fn rank(&self) -> usize {
        self.rank
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reflexive_module_base() {
        let module: ReflexiveModuleBase<i32> = ReflexiveModuleBase::new(3);
        assert_eq!(module.rank(), 3);
    }

    #[test]
    fn test_reflexive_module_dual() {
        let dual: ReflexiveModuleDual<i32> = ReflexiveModuleDual::new(4);
        assert_eq!(dual.rank(), 4);
    }

    #[test]
    fn test_reflexive_module_tensor() {
        let tensor: ReflexiveModuleTensor<i32> = ReflexiveModuleTensor::new(6);
        assert_eq!(tensor.rank(), 6);
    }
}
