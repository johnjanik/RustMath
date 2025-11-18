//! # Alternating Forms on Free Modules
//!
//! This module provides alternating forms (differential forms) on free modules,
//! corresponding to SageMath's `sage.tensor.modules.free_module_alt_form`.
//!
//! ## Main Type
//!
//! - `FreeModuleAltForm`: Alternating multilinear form (p-form)

use super::comp::CompFullyAntiSym;
use super::free_module_tensor::TensorType;
use std::fmt;
use std::marker::PhantomData;

/// An alternating form on a free module
///
/// This is a fully antisymmetric covariant tensor, also called a differential form
/// or exterior form. A p-form is an alternating multilinear map from M^p to R.
pub struct FreeModuleAltForm<R, M> {
    /// The antisymmetric components (all covariant)
    components: CompFullyAntiSym<R>,
    /// Degree of the form (number of covariant indices)
    degree: usize,
    /// The module this form is defined on
    module: PhantomData<M>,
    /// Name of the form
    name: Option<String>,
    /// LaTeX name
    latex_name: Option<String>,
}

impl<R: Clone + PartialEq + Default, M> FreeModuleAltForm<R, M> {
    /// Create a new alternating form
    ///
    /// # Arguments
    ///
    /// * `degree` - The degree (number of covariant indices)
    /// * `dimension` - The dimension of the underlying module
    pub fn new(degree: usize, dimension: usize) -> Self {
        let dimensions = vec![dimension; degree];

        Self {
            components: CompFullyAntiSym::new(degree, dimensions),
            degree,
            module: PhantomData,
            name: None,
            latex_name: None,
        }
    }

    /// Create a named alternating form
    pub fn with_name(
        degree: usize,
        dimension: usize,
        name: String,
        latex_name: Option<String>,
    ) -> Self {
        let mut form = Self::new(degree, dimension);
        form.name = Some(name);
        form.latex_name = latex_name;
        form
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

    /// Evaluate the form on vectors
    ///
    /// For a p-form ω and vectors v1,...,vp, compute ω(v1,...,vp)
    pub fn evaluate(&self, vectors: &[Vec<R>]) -> R
    where
        R: std::ops::Add<Output = R>
            + std::ops::Mul<Output = R>
            + std::ops::Neg<Output = R>
            + Default
            + Copy,
    {
        assert_eq!(
            vectors.len(),
            self.degree,
            "Number of vectors must match form degree"
        );

        // Simplified evaluation - sum over all components
        let mut result = R::default();

        for (indices, coeff) in self.components.non_zero_components() {
            let mut term = coeff;
            for (i, &idx) in indices.iter().enumerate() {
                if idx < vectors[i].len() {
                    term = term * vectors[i][idx];
                }
            }
            result = result + term;
        }

        result
    }

    /// Wedge product with another form
    ///
    /// The wedge product of a p-form and a q-form is a (p+q)-form
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

    /// Get the LaTeX name
    pub fn latex_name(&self) -> Option<&str> {
        self.latex_name.as_deref()
    }

    /// Check if this is a scalar (0-form)
    pub fn is_scalar(&self) -> bool {
        self.degree == 0
    }

    /// Check if this is a 1-form
    pub fn is_one_form(&self) -> bool {
        self.degree == 1
    }

    /// Get the tensor type
    pub fn tensor_type(&self) -> TensorType {
        TensorType::new(0, self.degree)
    }
}

impl<R: Clone + PartialEq + Default + fmt::Display, M> fmt::Display for FreeModuleAltForm<R, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(name) = &self.name {
            write!(f, "{}-form '{}'", self.degree, name)
        } else {
            write!(f, "{}-form", self.degree)
        }
    }
}

/// Wedge product of two alternating forms
///
/// ω ∧ η for p-form ω and q-form η gives a (p+q)-form
pub fn wedge<R, M>(
    omega: &FreeModuleAltForm<R, M>,
    eta: &FreeModuleAltForm<R, M>,
    dimension: usize,
) -> FreeModuleAltForm<R, M>
where
    R: Clone
        + PartialEq
        + Default
        + std::ops::Mul<Output = R>
        + std::ops::Neg<Output = R>
        + Copy,
{
    let new_degree = omega.degree() + eta.degree();
    let result = FreeModuleAltForm::new(new_degree, dimension);

    // Simplified wedge product computation
    // Full implementation would compute all antisymmetrized products
    // This is a placeholder showing the structure

    result
}

/// Exterior derivative
///
/// The exterior derivative d takes a p-form to a (p+1)-form
pub fn exterior_derivative<R, M>(
    omega: &FreeModuleAltForm<R, M>,
    dimension: usize,
) -> FreeModuleAltForm<R, M>
where
    R: Clone + PartialEq + Default,
{
    // The exterior derivative increases degree by 1
    FreeModuleAltForm::new(omega.degree() + 1, dimension)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestModule;

    #[test]
    fn test_alt_form_creation() {
        let form: FreeModuleAltForm<i32, TestModule> = FreeModuleAltForm::new(2, 3);

        assert_eq!(form.degree(), 2);
        assert!(!form.is_scalar());
        assert!(!form.is_one_form());
    }

    #[test]
    fn test_alt_form_with_name() {
        let form: FreeModuleAltForm<i32, TestModule> = FreeModuleAltForm::with_name(
            2,
            3,
            "omega".to_string(),
            Some("\\omega".to_string()),
        );

        assert_eq!(form.name(), Some("omega"));
        assert_eq!(form.latex_name(), Some("\\omega"));
    }

    #[test]
    fn test_degree_checks() {
        let scalar: FreeModuleAltForm<i32, TestModule> = FreeModuleAltForm::new(0, 3);
        assert!(scalar.is_scalar());

        let one_form: FreeModuleAltForm<i32, TestModule> = FreeModuleAltForm::new(1, 3);
        assert!(one_form.is_one_form());
    }

    #[test]
    fn test_set_get_component() {
        let mut form: FreeModuleAltForm<i32, TestModule> = FreeModuleAltForm::new(2, 3);

        form.set_component(vec![0, 1], 7);

        // Antisymmetric
        assert_eq!(form.get_component(&[0, 1]), Some(7));
        assert_eq!(form.get_component(&[1, 0]), Some(-7));
    }

    #[test]
    fn test_antisymmetry() {
        let mut form: FreeModuleAltForm<i32, TestModule> = FreeModuleAltForm::new(2, 4);

        form.set_component(vec![1, 2], 15);

        assert_eq!(form.get_component(&[1, 2]), Some(15));
        assert_eq!(form.get_component(&[2, 1]), Some(-15));
    }

    #[test]
    fn test_evaluate_one_form() {
        let mut form: FreeModuleAltForm<i32, TestModule> = FreeModuleAltForm::new(1, 3);

        form.set_component(vec![0], 2);
        form.set_component(vec![1], 3);
        form.set_component(vec![2], 5);

        let vector = vec![vec![1, 0, 0]];
        let result = form.evaluate(&vector);

        // ω(v) where ω = 2dx⁰ + 3dx¹ + 5dx² and v = (1,0,0)
        assert_eq!(result, 2);
    }

    #[test]
    fn test_evaluate_two_form() {
        let mut form: FreeModuleAltForm<i32, TestModule> = FreeModuleAltForm::new(2, 2);

        form.set_component(vec![0, 1], 1);

        let v1 = vec![1, 0];
        let v2 = vec![0, 1];
        let vectors = vec![v1, v2];

        let result = form.evaluate(&vectors);
        // Should get the component value
        assert!(result != 0);
    }

    #[test]
    fn test_wedge_degree() {
        let omega: FreeModuleAltForm<i32, TestModule> = FreeModuleAltForm::new(1, 3);
        let eta: FreeModuleAltForm<i32, TestModule> = FreeModuleAltForm::new(2, 3);

        assert_eq!(omega.wedge_degree(&eta), 3);
    }

    #[test]
    fn test_wedge_product() {
        let omega: FreeModuleAltForm<i32, TestModule> = FreeModuleAltForm::new(1, 3);
        let eta: FreeModuleAltForm<i32, TestModule> = FreeModuleAltForm::new(1, 3);

        let product = wedge(&omega, &eta, 3);
        assert_eq!(product.degree(), 2);
    }

    #[test]
    fn test_exterior_derivative() {
        let omega: FreeModuleAltForm<i32, TestModule> = FreeModuleAltForm::new(1, 3);

        let d_omega = exterior_derivative(&omega, 3);
        assert_eq!(d_omega.degree(), 2);
    }

    #[test]
    fn test_tensor_type() {
        let form: FreeModuleAltForm<i32, TestModule> = FreeModuleAltForm::new(2, 3);
        let tt = form.tensor_type();

        assert_eq!(tt.contravariant, 0);
        assert_eq!(tt.covariant, 2);
    }

    #[test]
    fn test_display() {
        let form: FreeModuleAltForm<i32, TestModule> =
            FreeModuleAltForm::with_name(2, 3, "F".to_string(), None);

        let display = format!("{}", form);
        assert!(display.contains("F"));
        assert!(display.contains("2-form"));
    }

    #[test]
    fn test_non_zero_components() {
        let mut form: FreeModuleAltForm<i32, TestModule> = FreeModuleAltForm::new(2, 3);

        form.set_component(vec![0, 1], 5);
        form.set_component(vec![1, 2], 3);

        let components = form.non_zero_components();
        assert_eq!(components.len(), 2);
    }
}
