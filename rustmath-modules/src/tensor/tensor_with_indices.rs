//! # Tensors with Index Notation
//!
//! This module provides tensors with explicit index notation for Einstein summation,
//! corresponding to SageMath's `sage.tensor.modules.tensor_with_indices`.

use super::free_module_tensor::{FreeModuleTensor, TensorType};
use std::fmt;
use std::marker::PhantomData;

/// Index type for tensor notation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexType {
    /// Contravariant (upper) index
    Contravariant,
    /// Covariant (lower) index
    Covariant,
}

/// Named index in tensor notation
#[derive(Debug, Clone)]
pub struct TensorIndex {
    pub name: String,
    pub index_type: IndexType,
}

impl TensorIndex {
    pub fn new(name: String, index_type: IndexType) -> Self {
        Self { name, index_type }
    }

    pub fn contravariant(name: String) -> Self {
        Self::new(name, IndexType::Contravariant)
    }

    pub fn covariant(name: String) -> Self {
        Self::new(name, IndexType::Covariant)
    }
}

impl fmt::Display for TensorIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.index_type {
            IndexType::Contravariant => write!(f, "^{}", self.name),
            IndexType::Covariant => write!(f, "_{}", self.name),
        }
    }
}

/// A tensor with explicit index notation
///
/// Allows writing tensors in index notation like T^{ij}_{k} for calculations
pub struct TensorWithIndices<R, M> {
    /// The underlying tensor
    tensor: PhantomData<FreeModuleTensor<R, M>>,
    /// The indices
    indices: Vec<TensorIndex>,
    /// Name of the tensor
    name: String,
}

impl<R, M> TensorWithIndices<R, M> {
    /// Create a tensor with indices
    pub fn new(name: String, indices: Vec<TensorIndex>) -> Self {
        Self {
            tensor: PhantomData,
            indices,
            name,
        }
    }

    /// Get the name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the indices
    pub fn indices(&self) -> &[TensorIndex] {
        &self.indices
    }

    /// Count contravariant indices
    pub fn contravariant_count(&self) -> usize {
        self.indices
            .iter()
            .filter(|idx| idx.index_type == IndexType::Contravariant)
            .count()
    }

    /// Count covariant indices
    pub fn covariant_count(&self) -> usize {
        self.indices
            .iter()
            .filter(|idx| idx.index_type == IndexType::Covariant)
            .count()
    }

    /// Get the tensor type
    pub fn tensor_type(&self) -> TensorType {
        TensorType::new(self.contravariant_count(), self.covariant_count())
    }

    /// Format in index notation
    pub fn index_notation(&self) -> String {
        let mut result = self.name.clone();

        let contra_indices: Vec<String> = self
            .indices
            .iter()
            .filter(|idx| idx.index_type == IndexType::Contravariant)
            .map(|idx| idx.name.clone())
            .collect();

        let cov_indices: Vec<String> = self
            .indices
            .iter()
            .filter(|idx| idx.index_type == IndexType::Covariant)
            .map(|idx| idx.name.clone())
            .collect();

        if !contra_indices.is_empty() {
            result.push_str(&format!("^{{{}}}", contra_indices.join(",")));
        }

        if !cov_indices.is_empty() {
            result.push_str(&format!("_{{{}}}", cov_indices.join(",")));
        }

        result
    }
}

impl<R, M> fmt::Display for TensorWithIndices<R, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.index_notation())
    }
}

/// Contract two tensors along specified indices
///
/// Performs Einstein summation over repeated indices
pub fn contract<R, M>(
    _t1: &TensorWithIndices<R, M>,
    _t2: &TensorWithIndices<R, M>,
) -> TensorType {
    // Simplified - would find matching indices and contract
    TensorType::new(0, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestModule;

    #[test]
    fn test_index_types() {
        let contra = TensorIndex::contravariant("i".to_string());
        let cov = TensorIndex::covariant("j".to_string());

        assert_eq!(contra.index_type, IndexType::Contravariant);
        assert_eq!(cov.index_type, IndexType::Covariant);
    }

    #[test]
    fn test_index_display() {
        let contra = TensorIndex::contravariant("i".to_string());
        let cov = TensorIndex::covariant("j".to_string());

        assert_eq!(format!("{}", contra), "^i");
        assert_eq!(format!("{}", cov), "_j");
    }

    #[test]
    fn test_tensor_with_indices() {
        let indices = vec![
            TensorIndex::contravariant("i".to_string()),
            TensorIndex::contravariant("j".to_string()),
            TensorIndex::covariant("k".to_string()),
        ];

        let tensor: TensorWithIndices<i32, TestModule> =
            TensorWithIndices::new("T".to_string(), indices);

        assert_eq!(tensor.name(), "T");
        assert_eq!(tensor.contravariant_count(), 2);
        assert_eq!(tensor.covariant_count(), 1);
    }

    #[test]
    fn test_tensor_type() {
        let indices = vec![
            TensorIndex::contravariant("i".to_string()),
            TensorIndex::covariant("j".to_string()),
            TensorIndex::covariant("k".to_string()),
        ];

        let tensor: TensorWithIndices<i32, TestModule> =
            TensorWithIndices::new("g".to_string(), indices);

        let tt = tensor.tensor_type();
        assert_eq!(tt.contravariant, 1);
        assert_eq!(tt.covariant, 2);
    }

    #[test]
    fn test_index_notation() {
        let indices = vec![
            TensorIndex::contravariant("i".to_string()),
            TensorIndex::contravariant("j".to_string()),
            TensorIndex::covariant("k".to_string()),
        ];

        let tensor: TensorWithIndices<i32, TestModule> =
            TensorWithIndices::new("R".to_string(), indices);

        let notation = tensor.index_notation();
        assert_eq!(notation, "R^{i,j}_{k}");
    }

    #[test]
    fn test_display() {
        let indices = vec![
            TensorIndex::contravariant("mu".to_string()),
            TensorIndex::covariant("nu".to_string()),
        ];

        let tensor: TensorWithIndices<i32, TestModule> =
            TensorWithIndices::new("F".to_string(), indices);

        assert_eq!(format!("{}", tensor), "F^{mu}_{nu}");
    }

    #[test]
    fn test_pure_covariant() {
        let indices = vec![
            TensorIndex::covariant("i".to_string()),
            TensorIndex::covariant("j".to_string()),
        ];

        let tensor: TensorWithIndices<i32, TestModule> =
            TensorWithIndices::new("g".to_string(), indices);

        assert_eq!(tensor.index_notation(), "g_{i,j}");
    }

    #[test]
    fn test_pure_contravariant() {
        let indices = vec![
            TensorIndex::contravariant("i".to_string()),
            TensorIndex::contravariant("j".to_string()),
        ];

        let tensor: TensorWithIndices<i32, TestModule> =
            TensorWithIndices::new("T".to_string(), indices);

        assert_eq!(tensor.index_notation(), "T^{i,j}");
    }
}
