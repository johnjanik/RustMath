//! # Tensor Products of Free Modules
//!
//! This module provides tensor product constructions for free modules,
//! corresponding to SageMath's `sage.tensor.modules.tensor_free_module`.

use super::free_module_tensor::TensorType;
use std::marker::PhantomData;

/// Tensor product module
///
/// M^⊗k ⊗ M*^⊗l where M is a free module and M* is its dual
pub struct TensorFreeModule<R> {
    tensor_type: TensorType,
    base_rank: usize,
    ring: PhantomData<R>,
}

impl<R> TensorFreeModule<R> {
    pub fn new(tensor_type: TensorType, base_rank: usize) -> Self {
        Self {
            tensor_type,
            base_rank,
            ring: PhantomData,
        }
    }

    pub fn tensor_type(&self) -> TensorType {
        self.tensor_type
    }

    pub fn base_rank(&self) -> usize {
        self.base_rank
    }

    /// Rank of the tensor module
    pub fn rank(&self) -> usize {
        self.base_rank.pow(self.tensor_type.rank() as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_free_module() {
        let tensor_mod: TensorFreeModule<i32> =
            TensorFreeModule::new(TensorType::new(1, 1), 3);

        assert_eq!(tensor_mod.base_rank(), 3);
        assert_eq!(tensor_mod.tensor_type(), TensorType::new(1, 1));
        assert_eq!(tensor_mod.rank(), 9); // 3^2
    }

    #[test]
    fn test_higher_tensor_product() {
        let tensor_mod: TensorFreeModule<i32> =
            TensorFreeModule::new(TensorType::new(2, 1), 2);

        assert_eq!(tensor_mod.rank(), 8); // 2^3
    }
}
