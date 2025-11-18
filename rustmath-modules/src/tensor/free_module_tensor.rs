//! # Free Module Tensors
//!
//! This module provides the base tensor type for free modules,
//! corresponding to SageMath's `sage.tensor.modules.free_module_tensor`.
//!
//! ## Main Type
//!
//! - `FreeModuleTensor`: Base type for tensors on finite-rank free modules

use super::comp::Components;
use super::free_module_basis::FreeModuleBasis;
use std::fmt;
use std::marker::PhantomData;

/// Type for tensor with contravariant and covariant indices
///
/// A tensor of type (k,l) has k contravariant indices and l covariant indices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorType {
    /// Number of contravariant indices (upper indices)
    pub contravariant: usize,
    /// Number of covariant indices (lower indices)
    pub covariant: usize,
}

impl TensorType {
    /// Create a new tensor type
    pub fn new(contravariant: usize, covariant: usize) -> Self {
        Self {
            contravariant,
            covariant,
        }
    }

    /// Total rank (contravariant + covariant)
    pub fn rank(&self) -> usize {
        self.contravariant + self.covariant
    }

    /// Check if this is a scalar (0,0) tensor
    pub fn is_scalar(&self) -> bool {
        self.contravariant == 0 && self.covariant == 0
    }

    /// Check if this is a vector (1,0) tensor
    pub fn is_vector(&self) -> bool {
        self.contravariant == 1 && self.covariant == 0
    }

    /// Check if this is a covector (0,1) tensor
    pub fn is_covector(&self) -> bool {
        self.contravariant == 0 && self.covariant == 1
    }

    /// Check if this is an endomorphism (1,1) tensor
    pub fn is_endomorphism(&self) -> bool {
        self.contravariant == 1 && self.covariant == 1
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({},{})", self.contravariant, self.covariant)
    }
}

/// A tensor on a finite-rank free module
///
/// This represents a tensor T of type (k,l) on a free module M,
/// i.e., an element of M^⊗k ⊗ M*^⊗l where M* is the dual of M
pub struct FreeModuleTensor<R, M> {
    /// The components of the tensor
    components: Components<R>,
    /// The tensor type (k,l)
    tensor_type: TensorType,
    /// The module this tensor is defined on
    module: PhantomData<M>,
    /// Name of the tensor
    name: Option<String>,
    /// LaTeX name
    latex_name: Option<String>,
}

impl<R: Clone + PartialEq, M> FreeModuleTensor<R, M> {
    /// Create a new tensor
    ///
    /// # Arguments
    ///
    /// * `tensor_type` - The type (k,l) of the tensor
    /// * `dimension` - The dimension of the underlying module
    pub fn new(tensor_type: TensorType, dimension: usize) -> Self {
        let rank = tensor_type.rank();
        let dimensions = vec![dimension; rank];

        Self {
            components: Components::new(rank, dimensions),
            tensor_type,
            module: PhantomData,
            name: None,
            latex_name: None,
        }
    }

    /// Create a named tensor
    pub fn with_name(
        tensor_type: TensorType,
        dimension: usize,
        name: String,
        latex_name: Option<String>,
    ) -> Self {
        let mut tensor = Self::new(tensor_type, dimension);
        tensor.name = Some(name);
        tensor.latex_name = latex_name;
        tensor
    }

    /// Get the tensor type
    pub fn tensor_type(&self) -> TensorType {
        self.tensor_type
    }

    /// Get the rank (total number of indices)
    pub fn rank(&self) -> usize {
        self.tensor_type.rank()
    }

    /// Set a component
    pub fn set_component(&mut self, indices: Vec<usize>, value: R) {
        self.components.set(indices, value);
    }

    /// Get a component
    pub fn get_component(&self, indices: &[usize]) -> Option<&R> {
        self.components.get(indices)
    }

    /// Get all non-zero components
    pub fn non_zero_components(&self) -> Vec<(Vec<usize>, R)> {
        self.components.non_zero_components()
    }

    /// Get the name
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Set the name
    pub fn set_name(&mut self, name: String) {
        self.name = Some(name);
    }

    /// Get the LaTeX name
    pub fn latex_name(&self) -> Option<&str> {
        self.latex_name.as_deref()
    }

    /// Contract with another tensor
    ///
    /// This is a simplified version - full contraction would be more complex
    pub fn contract_simple(&self, index1: usize, index2: usize) -> TensorType
    where
        R: std::ops::Add<Output = R> + std::ops::Mul<Output = R> + Default + Copy,
    {
        // Contraction reduces rank by 2
        let new_contra = if index1 < self.tensor_type.contravariant {
            self.tensor_type.contravariant - 1
        } else {
            self.tensor_type.contravariant
        };

        let new_cov = if index2 >= self.tensor_type.contravariant {
            self.tensor_type.covariant - 1
        } else {
            self.tensor_type.covariant
        };

        TensorType::new(new_contra, new_cov)
    }

    /// Display the tensor in component form
    pub fn display_components(&self) -> String
    where
        R: fmt::Display,
    {
        let mut result = String::new();

        if let Some(name) = &self.name {
            result.push_str(&format!("{} = ", name));
        }

        result.push_str(&format!("Tensor type {} with components:\n", self.tensor_type));

        let components = self.non_zero_components();
        if components.is_empty() {
            result.push_str("  (all zero)\n");
        } else {
            for (indices, value) in components {
                result.push_str(&format!("  {:?}: {}\n", indices, value));
            }
        }

        result
    }
}

impl<R: Clone + PartialEq + fmt::Display, M> fmt::Display for FreeModuleTensor<R, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(name) = &self.name {
            write!(f, "Tensor '{}' of type {}", name, self.tensor_type)
        } else {
            write!(f, "Tensor of type {}", self.tensor_type)
        }
    }
}

/// Tensor product of two tensors
pub fn tensor_product<R, M>(
    t1: &FreeModuleTensor<R, M>,
    t2: &FreeModuleTensor<R, M>,
) -> TensorType
where
    R: Clone + PartialEq + std::ops::Mul<Output = R>,
{
    // The tensor product of a (k1,l1) tensor and a (k2,l2) tensor
    // is a (k1+k2, l1+l2) tensor
    TensorType::new(
        t1.tensor_type.contravariant + t2.tensor_type.contravariant,
        t1.tensor_type.covariant + t2.tensor_type.covariant,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestModule;

    #[test]
    fn test_tensor_type() {
        let tt = TensorType::new(2, 1);
        assert_eq!(tt.contravariant, 2);
        assert_eq!(tt.covariant, 1);
        assert_eq!(tt.rank(), 3);
        assert!(!tt.is_scalar());
        assert!(!tt.is_vector());

        let scalar = TensorType::new(0, 0);
        assert!(scalar.is_scalar());

        let vector = TensorType::new(1, 0);
        assert!(vector.is_vector());

        let covector = TensorType::new(0, 1);
        assert!(covector.is_covector());

        let endo = TensorType::new(1, 1);
        assert!(endo.is_endomorphism());
    }

    #[test]
    fn test_tensor_type_display() {
        let tt = TensorType::new(2, 3);
        assert_eq!(format!("{}", tt), "(2,3)");
    }

    #[test]
    fn test_free_module_tensor_creation() {
        let tensor: FreeModuleTensor<i32, TestModule> = FreeModuleTensor::new(
            TensorType::new(1, 1),
            3,
        );

        assert_eq!(tensor.tensor_type(), TensorType::new(1, 1));
        assert_eq!(tensor.rank(), 2);
    }

    #[test]
    fn test_tensor_with_name() {
        let tensor: FreeModuleTensor<i32, TestModule> = FreeModuleTensor::with_name(
            TensorType::new(2, 0),
            3,
            "T".to_string(),
            Some("\\mathcal{T}".to_string()),
        );

        assert_eq!(tensor.name(), Some("T"));
        assert_eq!(tensor.latex_name(), Some("\\mathcal{T}"));
    }

    #[test]
    fn test_set_get_component() {
        let mut tensor: FreeModuleTensor<i32, TestModule> = FreeModuleTensor::new(
            TensorType::new(1, 1),
            2,
        );

        tensor.set_component(vec![0, 1], 42);
        tensor.set_component(vec![1, 0], -17);

        assert_eq!(tensor.get_component(&[0, 1]), Some(&42));
        assert_eq!(tensor.get_component(&[1, 0]), Some(&-17));
        assert_eq!(tensor.get_component(&[0, 0]), None);
    }

    #[test]
    fn test_non_zero_components() {
        let mut tensor: FreeModuleTensor<i32, TestModule> = FreeModuleTensor::new(
            TensorType::new(0, 2),
            2,
        );

        tensor.set_component(vec![0, 0], 1);
        tensor.set_component(vec![0, 1], 2);
        tensor.set_component(vec![1, 1], 3);

        let components = tensor.non_zero_components();
        assert_eq!(components.len(), 3);
    }

    #[test]
    fn test_tensor_product_type() {
        let t1: FreeModuleTensor<i32, TestModule> = FreeModuleTensor::new(
            TensorType::new(1, 0),
            3,
        );

        let t2: FreeModuleTensor<i32, TestModule> = FreeModuleTensor::new(
            TensorType::new(0, 1),
            3,
        );

        let product_type = tensor_product(&t1, &t2);
        assert_eq!(product_type, TensorType::new(1, 1));
    }

    #[test]
    fn test_contract_simple() {
        let tensor: FreeModuleTensor<i32, TestModule> = FreeModuleTensor::new(
            TensorType::new(1, 1),
            3,
        );

        // Contracting a (1,1) tensor gives a (0,0) tensor (scalar)
        let contracted = tensor.contract_simple(0, 1);
        assert_eq!(contracted, TensorType::new(0, 0));
    }

    #[test]
    fn test_display() {
        let tensor: FreeModuleTensor<i32, TestModule> = FreeModuleTensor::with_name(
            TensorType::new(2, 1),
            3,
            "Riemann".to_string(),
            None,
        );

        let display = format!("{}", tensor);
        assert!(display.contains("Riemann"));
        assert!(display.contains("(2,1)"));
    }

    #[test]
    fn test_display_components() {
        let mut tensor: FreeModuleTensor<i32, TestModule> = FreeModuleTensor::with_name(
            TensorType::new(1, 1),
            2,
            "g".to_string(),
            None,
        );

        tensor.set_component(vec![0, 0], 1);
        tensor.set_component(vec![1, 1], -1);

        let display = tensor.display_components();
        assert!(display.contains("g ="));
        assert!(display.contains("(1,1)"));
    }
}
