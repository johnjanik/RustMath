//! # Tensor Free Submodules
//!
//! This module provides submodules of tensor products,
//! corresponding to SageMath's `sage.tensor.modules.tensor_free_submodule`.

use super::free_module_tensor::TensorType;
use std::marker::PhantomData;

/// Symmetric submodule of a tensor product
///
/// The submodule of fully symmetric tensors
pub struct TensorFreeSubmoduleSym<R> {
    tensor_type: TensorType,
    base_rank: usize,
    ring: PhantomData<R>,
}

impl<R> TensorFreeSubmoduleSym<R> {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric_submodule() {
        let submod: TensorFreeSubmoduleSym<i32> =
            TensorFreeSubmoduleSym::new(TensorType::new(2, 0), 3);

        assert_eq!(submod.base_rank(), 3);
        assert_eq!(submod.tensor_type(), TensorType::new(2, 0));
    }
}
