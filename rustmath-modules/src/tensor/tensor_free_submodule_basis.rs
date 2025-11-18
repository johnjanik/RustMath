//! # Tensor Free Submodule Bases
//!
//! This module provides bases for symmetric tensor submodules,
//! corresponding to SageMath's `sage.tensor.modules.tensor_free_submodule_basis`.

use std::marker::PhantomData;

/// Basis for the symmetric submodule of a tensor product
pub struct TensorFreeSubmoduleBasisSym<R> {
    degree: usize,
    base_rank: usize,
    ring: PhantomData<R>,
}

impl<R> TensorFreeSubmoduleBasisSym<R> {
    pub fn new(degree: usize, base_rank: usize) -> Self {
        Self {
            degree,
            base_rank,
            ring: PhantomData,
        }
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn base_rank(&self) -> usize {
        self.base_rank
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric_basis() {
        let basis: TensorFreeSubmoduleBasisSym<i32> =
            TensorFreeSubmoduleBasisSym::new(2, 3);

        assert_eq!(basis.degree(), 2);
        assert_eq!(basis.base_rank(), 3);
    }
}
