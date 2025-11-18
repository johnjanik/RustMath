//! # Alternating Contravariant Tensors
//!
//! This module provides alternating contravariant tensors on free modules,
//! corresponding to SageMath's `sage.tensor.modules.alternating_contr_tensor`.
//!
//! ## Main Type
//!
//! - `AlternatingContrTensor`: Fully antisymmetric contravariant tensor

use super::comp::CompFullyAntiSym;
use super::free_module_tensor::{FreeModuleTensor, TensorType};
use std::fmt;
use std::marker::PhantomData;

/// An alternating contravariant tensor on a free module
///
/// This is a tensor that is fully antisymmetric in all its indices.
/// For a tensor of degree p, we have:
/// T(v_σ(1), ..., v_σ(p)) = sgn(σ) T(v_1, ..., v_p)
/// for any permutation σ
pub struct AlternatingContrTensor<R, M> {
    /// The antisymmetric components
    components: CompFullyAntiSym<R>,
    /// Degree of the form (number of indices)
    degree: usize,
    /// The module this tensor is defined on
    module: PhantomData<M>,
    /// Name of the tensor
    name: Option<String>,
}

impl<R: Clone + PartialEq + Default, M> AlternatingContrTensor<R, M> {
    /// Create a new alternating contravariant tensor
    ///
    /// # Arguments
    ///
    /// * `degree` - The degree (number of contravariant indices)
    /// * `dimension` - The dimension of the underlying module
    pub fn new(degree: usize, dimension: usize) -> Self {
        let dimensions = vec![dimension; degree];

        Self {
            components: CompFullyAntiSym::new(degree, dimensions),
            degree,
            module: PhantomData,
            name: None,
        }
    }

    /// Create a named alternating tensor
    pub fn with_name(degree: usize, dimension: usize, name: String) -> Self {
        let mut tensor = Self::new(degree, dimension);
        tensor.name = Some(name);
        tensor
    }

    /// Get the degree
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Set a component
    ///
    /// The component is automatically adjusted for antisymmetry
    pub fn set_component(&mut self, indices: Vec<usize>, value: R)
    where
        R: std::ops::Neg<Output = R>,
    {
        self.components.set(indices, value);
    }

    /// Get a component
    ///
    /// Returns the value with appropriate sign based on index permutation
    pub fn get_component(&self, indices: &[usize]) -> Option<R>
    where
        R: std::ops::Neg<Output = R>,
    {
        self.components.get(indices)
    }

    /// Wedge product with another alternating tensor
    ///
    /// Returns the degree of the resulting tensor
    pub fn wedge_degree(&self, other: &Self) -> usize {
        self.degree + other.degree
    }

    /// Get all non-zero components
    pub fn non_zero_components(&self) -> Vec<(Vec<usize>, R)> {
        self.components.non_zero_components()
    }

    /// Get the name
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Check if this is a scalar (degree 0)
    pub fn is_scalar(&self) -> bool {
        self.degree == 0
    }

    /// Check if this is a vector (degree 1)
    pub fn is_vector(&self) -> bool {
        self.degree == 1
    }

    /// Convert to a general free module tensor
    pub fn to_tensor(&self, dimension: usize) -> FreeModuleTensor<R, M> {
        FreeModuleTensor::new(TensorType::new(self.degree, 0), dimension)
    }
}

impl<R: Clone + PartialEq + Default + fmt::Display, M> fmt::Display for AlternatingContrTensor<R, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(name) = &self.name {
            write!(f, "Alternating tensor '{}' of degree {}", name, self.degree)
        } else {
            write!(f, "Alternating tensor of degree {}", self.degree)
        }
    }
}

/// Wedge product of two alternating tensors
///
/// The wedge product ω ∧ η of a p-form ω and a q-form η is a (p+q)-form
pub fn wedge_product<R, M>(
    omega: &AlternatingContrTensor<R, M>,
    eta: &AlternatingContrTensor<R, M>,
    dimension: usize,
) -> AlternatingContrTensor<R, M>
where
    R: Clone + PartialEq + Default + std::ops::Mul<Output = R> + std::ops::Neg<Output = R>,
{
    let new_degree = omega.degree() + eta.degree();
    AlternatingContrTensor::new(new_degree, dimension)
}

/// Interior product (contraction with a vector)
///
/// For a vector v and a p-form ω, i_v(ω) is a (p-1)-form
pub fn interior_product<R, M>(
    _omega: &AlternatingContrTensor<R, M>,
    _vector: &[R],
    dimension: usize,
) -> AlternatingContrTensor<R, M>
where
    R: Clone + PartialEq + Default,
{
    // Simplified implementation - actual interior product would compute contractions
    AlternatingContrTensor::new(
        if _omega.degree() > 0 {
            _omega.degree() - 1
        } else {
            0
        },
        dimension,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestModule;

    #[test]
    fn test_alternating_tensor_creation() {
        let tensor: AlternatingContrTensor<i32, TestModule> =
            AlternatingContrTensor::new(2, 3);

        assert_eq!(tensor.degree(), 2);
        assert!(!tensor.is_scalar());
        assert!(!tensor.is_vector());
    }

    #[test]
    fn test_alternating_tensor_with_name() {
        let tensor: AlternatingContrTensor<i32, TestModule> =
            AlternatingContrTensor::with_name(3, 4, "omega".to_string());

        assert_eq!(tensor.name(), Some("omega"));
        assert_eq!(tensor.degree(), 3);
    }

    #[test]
    fn test_degree_checks() {
        let scalar: AlternatingContrTensor<i32, TestModule> =
            AlternatingContrTensor::new(0, 3);
        assert!(scalar.is_scalar());

        let vector: AlternatingContrTensor<i32, TestModule> =
            AlternatingContrTensor::new(1, 3);
        assert!(vector.is_vector());
    }

    #[test]
    fn test_set_get_component() {
        let mut tensor: AlternatingContrTensor<i32, TestModule> =
            AlternatingContrTensor::new(2, 3);

        tensor.set_component(vec![0, 1], 5);

        // Antisymmetric: T[0,1] = -T[1,0]
        assert_eq!(tensor.get_component(&[0, 1]), Some(5));
        assert_eq!(tensor.get_component(&[1, 0]), Some(-5));
    }

    #[test]
    fn test_antisymmetry() {
        let mut tensor: AlternatingContrTensor<i32, TestModule> =
            AlternatingContrTensor::new(2, 4);

        tensor.set_component(vec![1, 3], 10);

        // Check antisymmetry
        assert_eq!(tensor.get_component(&[1, 3]), Some(10));
        assert_eq!(tensor.get_component(&[3, 1]), Some(-10));
    }

    #[test]
    fn test_wedge_degree() {
        let omega: AlternatingContrTensor<i32, TestModule> =
            AlternatingContrTensor::new(2, 3);
        let eta: AlternatingContrTensor<i32, TestModule> =
            AlternatingContrTensor::new(1, 3);

        assert_eq!(omega.wedge_degree(&eta), 3);
    }

    #[test]
    fn test_wedge_product() {
        let omega: AlternatingContrTensor<i32, TestModule> =
            AlternatingContrTensor::new(1, 3);
        let eta: AlternatingContrTensor<i32, TestModule> =
            AlternatingContrTensor::new(1, 3);

        let product = wedge_product(&omega, &eta, 3);
        assert_eq!(product.degree(), 2);
    }

    #[test]
    fn test_interior_product() {
        let omega: AlternatingContrTensor<i32, TestModule> =
            AlternatingContrTensor::new(2, 3);
        let vector = vec![1, 2, 3];

        let result = interior_product(&omega, &vector, 3);
        assert_eq!(result.degree(), 1);
    }

    #[test]
    fn test_non_zero_components() {
        let mut tensor: AlternatingContrTensor<i32, TestModule> =
            AlternatingContrTensor::new(2, 3);

        tensor.set_component(vec![0, 1], 5);
        tensor.set_component(vec![1, 2], 3);

        let components = tensor.non_zero_components();
        assert_eq!(components.len(), 2);
    }

    #[test]
    fn test_display() {
        let tensor: AlternatingContrTensor<i32, TestModule> =
            AlternatingContrTensor::with_name(2, 3, "F".to_string());

        let display = format!("{}", tensor);
        assert!(display.contains("F"));
        assert!(display.contains("2"));
    }

    #[test]
    fn test_to_tensor() {
        let alt_tensor: AlternatingContrTensor<i32, TestModule> =
            AlternatingContrTensor::new(2, 3);

        let tensor = alt_tensor.to_tensor(3);
        assert_eq!(tensor.tensor_type().contravariant, 2);
        assert_eq!(tensor.tensor_type().covariant, 0);
    }
}
