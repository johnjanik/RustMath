//! Derivations on function fields
//!
//! This module provides derivation structures for function fields, corresponding to
//! SageMath's `sage.rings.function_field.derivations` module.
//!
//! # Mathematical Background
//!
//! A derivation D on a ring R is a map D: R → R satisfying:
//! 1. **Additivity**: D(α + β) = D(α) + D(β)
//! 2. **Leibniz rule**: D(αβ) = β·D(α) + α·D(β)
//!
//! For function fields, derivations generalize the notion of differentiation.
//! The most common example is d/dx on k(x), but more complex derivations
//! exist for algebraic extensions.
//!
//! # Key Types
//!
//! - `FunctionFieldDerivation<F>`: Base derivation type for function fields
//! - Methods for computing derivatives and checking derivation properties
//!
//! # Examples
//!
//! ```ignore
//! use rustmath_rings::derivations_function_field::*;
//! use rustmath_rationals::Rational;
//!
//! // Create a derivation d/dx on Q(x)
//! let deriv = FunctionFieldDerivation::d_dx();
//!
//! // Apply to a function field element
//! let result = deriv.apply(element);
//! ```

use rustmath_core::{Ring, Field};
use std::marker::PhantomData;
use std::fmt;

/// Base trait for derivations
///
/// A derivation is a map satisfying additivity and the Leibniz (product) rule.
pub trait Derivation<R: Ring> {
    /// Apply the derivation to an element
    ///
    /// # Arguments
    ///
    /// * `x` - Element to differentiate
    ///
    /// # Returns
    ///
    /// The derivative D(x)
    fn apply(&self, x: &R) -> R;

    /// Check if this derivation is injective
    ///
    /// Derivations are never injective (constants map to zero).
    fn is_injective(&self) -> bool {
        false
    }

    /// Compose two derivations
    ///
    /// Returns the composition D₂ ∘ D₁
    fn compose<D: Derivation<R>>(&self, other: &D) -> ComposedDerivation<R, Self, D>
    where
        Self: Sized,
    {
        ComposedDerivation::new(self, other)
    }
}

/// Derivation on function fields
///
/// Corresponds to SageMath's `FunctionFieldDerivation` class.
/// This is the base class for derivations on function fields, extending
/// the abstract ring derivation concept.
///
/// # Type Parameters
///
/// - `F`: The base field type
#[derive(Clone, Debug)]
pub struct FunctionFieldDerivation<F: Field> {
    /// Description of the derivation (e.g., "d/dx")
    description: String,
    /// Phantom data for the field type
    _field: PhantomData<F>,
}

impl<F: Field> FunctionFieldDerivation<F> {
    /// Create a new function field derivation
    ///
    /// # Arguments
    ///
    /// * `description` - Human-readable description (e.g., "d/dx")
    pub fn new(description: String) -> Self {
        FunctionFieldDerivation {
            description,
            _field: PhantomData,
        }
    }

    /// Create the standard d/dx derivation
    ///
    /// This is the most common derivation on rational function fields.
    pub fn d_dx() -> Self {
        FunctionFieldDerivation::new("d/dx".to_string())
    }

    /// Get the description of this derivation
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Check if this derivation is injective
    ///
    /// Derivations are never injective (constants map to zero)
    pub fn is_injective(&self) -> bool {
        false
    }

    /// Multiply this derivation by a scalar
    ///
    /// Returns c·D where c is a scalar and D is this derivation.
    ///
    /// # Arguments
    ///
    /// * `scalar` - The scalar to multiply by (as string for now)
    pub fn scalar_multiply(&self, scalar: String) -> ScalarMultipleDerivation<F> {
        ScalarMultipleDerivation::new(scalar, self.clone())
    }

    /// Evaluate the derivation on a function field element
    ///
    /// This is a placeholder that would perform actual differentiation.
    ///
    /// # Arguments
    ///
    /// * `element` - String representation of the element
    pub fn eval(&self, element: &str) -> String {
        format!("{}({})", self.description, element)
    }
}

impl<F: Field> fmt::Display for FunctionFieldDerivation<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description)
    }
}

/// Scalar multiple of a derivation
///
/// Represents c·D where c is a scalar and D is a derivation.
#[derive(Clone, Debug)]
pub struct ScalarMultipleDerivation<F: Field> {
    scalar: String,
    base_derivation: FunctionFieldDerivation<F>,
}

impl<F: Field> ScalarMultipleDerivation<F> {
    /// Create a new scalar multiple derivation
    pub fn new(scalar: String, base: FunctionFieldDerivation<F>) -> Self {
        ScalarMultipleDerivation {
            scalar,
            base_derivation: base,
        }
    }

    /// Get the scalar factor
    pub fn scalar(&self) -> &str {
        &self.scalar
    }

    /// Get the base derivation
    pub fn base_derivation(&self) -> &FunctionFieldDerivation<F> {
        &self.base_derivation
    }

    /// Evaluate on an element
    pub fn eval(&self, element: &str) -> String {
        format!("{}·{}", self.scalar, self.base_derivation.eval(element))
    }
}

impl<F: Field> fmt::Display for ScalarMultipleDerivation<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}·{}", self.scalar, self.base_derivation)
    }
}

/// Composition of two derivations
///
/// Represents D₂ ∘ D₁
pub struct ComposedDerivation<R: Ring, D1: Derivation<R> + ?Sized, D2: Derivation<R> + ?Sized> {
    first: PhantomData<D1>,
    second: PhantomData<D2>,
    _ring: PhantomData<R>,
}

impl<R: Ring, D1: Derivation<R>, D2: Derivation<R>> ComposedDerivation<R, D1, D2> {
    /// Create a new composed derivation
    pub fn new(_first: &D1, _second: &D2) -> Self {
        ComposedDerivation {
            first: PhantomData,
            second: PhantomData,
            _ring: PhantomData,
        }
    }
}

/// Higher-order derivation
///
/// Represents D^n, the n-th power of a derivation.
/// For example, d²/dx² is the second-order derivation.
#[derive(Clone, Debug)]
pub struct HigherDerivation<F: Field> {
    /// Base derivation
    base: FunctionFieldDerivation<F>,
    /// Order (n for D^n)
    order: usize,
}

impl<F: Field> HigherDerivation<F> {
    /// Create a new higher-order derivation
    ///
    /// # Arguments
    ///
    /// * `base` - The base derivation
    /// * `order` - The order (must be > 0)
    pub fn new(base: FunctionFieldDerivation<F>, order: usize) -> Self {
        assert!(order > 0, "Order must be positive");
        HigherDerivation { base, order }
    }

    /// Get the order of this derivation
    pub fn order(&self) -> usize {
        self.order
    }

    /// Get the base derivation
    pub fn base(&self) -> &FunctionFieldDerivation<F> {
        &self.base
    }

    /// Evaluate on an element
    pub fn eval(&self, element: &str) -> String {
        if self.order == 1 {
            self.base.eval(element)
        } else {
            format!("{}^{}({})", self.base.description(), self.order, element)
        }
    }
}

impl<F: Field> fmt::Display for HigherDerivation<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.order == 1 {
            write!(f, "{}", self.base)
        } else {
            write!(f, "{}^{}", self.base, self.order)
        }
    }
}

/// Module of derivations
///
/// The set of all derivations on a function field forms a module over the field.
/// This struct represents that module.
#[derive(Clone, Debug)]
pub struct DerivationModule<F: Field> {
    /// Base function field identifier
    function_field: String,
    _field: PhantomData<F>,
}

impl<F: Field> DerivationModule<F> {
    /// Create a new derivation module
    ///
    /// # Arguments
    ///
    /// * `field` - The function field identifier
    pub fn new(field: String) -> Self {
        DerivationModule {
            function_field: field,
            _field: PhantomData,
        }
    }

    /// Get the underlying function field
    pub fn function_field(&self) -> &str {
        &self.function_field
    }

    /// Get a basis for the derivation module
    ///
    /// For a rational function field k(x), this is typically just {d/dx}.
    /// For extensions, the basis can be more complex.
    pub fn basis(&self) -> Vec<FunctionFieldDerivation<F>> {
        // Placeholder: returns single d/dx derivation
        vec![FunctionFieldDerivation::d_dx()]
    }

    /// Dimension of the derivation module
    ///
    /// For a rational function field, dimension is 1.
    /// For a function field of transcendence degree n, dimension is n.
    pub fn dimension(&self) -> usize {
        self.basis().len()
    }

    /// Create a derivation from coefficients
    ///
    /// Given coefficients c₁, ..., cₙ and basis derivations D₁, ..., Dₙ,
    /// returns c₁D₁ + ... + cₙDₙ.
    pub fn derivation_from_coefficients(&self, coeffs: Vec<String>) -> Option<FunctionFieldDerivation<F>> {
        if coeffs.is_empty() {
            return None;
        }

        // Placeholder: creates a derivation with coefficient description
        Some(FunctionFieldDerivation::new(format!(
            "{}·d/dx",
            coeffs[0]
        )))
    }
}

impl<F: Field> fmt::Display for DerivationModule<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Der({})", self.function_field)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_derivation_creation() {
        let deriv = FunctionFieldDerivation::<Rational>::d_dx();
        assert_eq!(deriv.description(), "d/dx");
        assert!(!deriv.is_injective());
    }

    #[test]
    fn test_custom_derivation() {
        let deriv = FunctionFieldDerivation::<Rational>::new("d/dt".to_string());
        assert_eq!(deriv.description(), "d/dt");
    }

    #[test]
    fn test_derivation_display() {
        let deriv = FunctionFieldDerivation::<Rational>::d_dx();
        assert_eq!(format!("{}", deriv), "d/dx");
    }

    #[test]
    fn test_derivation_eval() {
        let deriv = FunctionFieldDerivation::<Rational>::d_dx();
        let result = deriv.eval("x^2");
        assert!(result.contains("d/dx"));
        assert!(result.contains("x^2"));
    }

    #[test]
    fn test_scalar_multiply() {
        let deriv = FunctionFieldDerivation::<Rational>::d_dx();
        let scaled = deriv.scalar_multiply("3".to_string());

        assert_eq!(scaled.scalar(), "3");
        assert_eq!(scaled.base_derivation().description(), "d/dx");
    }

    #[test]
    fn test_scalar_multiple_eval() {
        let deriv = FunctionFieldDerivation::<Rational>::d_dx();
        let scaled = deriv.scalar_multiply("2".to_string());
        let result = scaled.eval("x");

        assert!(result.contains("2"));
        assert!(result.contains("d/dx"));
    }

    #[test]
    fn test_scalar_multiple_display() {
        let deriv = FunctionFieldDerivation::<Rational>::d_dx();
        let scaled = deriv.scalar_multiply("5".to_string());

        assert_eq!(format!("{}", scaled), "5·d/dx");
    }

    #[test]
    fn test_higher_derivation() {
        let base = FunctionFieldDerivation::<Rational>::d_dx();
        let higher = HigherDerivation::new(base, 2);

        assert_eq!(higher.order(), 2);
        assert_eq!(higher.base().description(), "d/dx");
    }

    #[test]
    #[should_panic(expected = "Order must be positive")]
    fn test_higher_derivation_zero_order() {
        let base = FunctionFieldDerivation::<Rational>::d_dx();
        let _ = HigherDerivation::new(base, 0);
    }

    #[test]
    fn test_higher_derivation_display() {
        let base = FunctionFieldDerivation::<Rational>::d_dx();
        let higher = HigherDerivation::new(base.clone(), 2);

        assert_eq!(format!("{}", higher), "d/dx^2");

        let first = HigherDerivation::new(base, 1);
        assert_eq!(format!("{}", first), "d/dx");
    }

    #[test]
    fn test_higher_derivation_eval() {
        let base = FunctionFieldDerivation::<Rational>::d_dx();
        let higher = HigherDerivation::new(base, 3);
        let result = higher.eval("x^5");

        assert!(result.contains("d/dx^3"));
        assert!(result.contains("x^5"));
    }

    #[test]
    fn test_derivation_module() {
        let module = DerivationModule::<Rational>::new("Q(x)".to_string());

        assert_eq!(module.function_field(), "Q(x)");
        assert_eq!(module.dimension(), 1);
    }

    #[test]
    fn test_derivation_module_display() {
        let module = DerivationModule::<Rational>::new("Q(x)".to_string());
        assert_eq!(format!("{}", module), "Der(Q(x))");
    }

    #[test]
    fn test_derivation_module_basis() {
        let module = DerivationModule::<Rational>::new("Q(x)".to_string());
        let basis = module.basis();

        assert_eq!(basis.len(), 1);
        assert_eq!(basis[0].description(), "d/dx");
    }

    #[test]
    fn test_derivation_from_coefficients() {
        let module = DerivationModule::<Rational>::new("Q(x)".to_string());
        let deriv = module.derivation_from_coefficients(vec!["2".to_string()]);

        assert!(deriv.is_some());
        let d = deriv.unwrap();
        assert!(d.description().contains("2"));
    }

    #[test]
    fn test_derivation_from_empty_coefficients() {
        let module = DerivationModule::<Rational>::new("Q(x)".to_string());
        let deriv = module.derivation_from_coefficients(vec![]);

        assert!(deriv.is_none());
    }

    #[test]
    fn test_is_injective_always_false() {
        let deriv1 = FunctionFieldDerivation::<Rational>::d_dx();
        let deriv2 = FunctionFieldDerivation::<Rational>::new("d/dy".to_string());

        assert!(!deriv1.is_injective());
        assert!(!deriv2.is_injective());
    }

    #[test]
    fn test_clone_derivation() {
        let deriv = FunctionFieldDerivation::<Rational>::d_dx();
        let cloned = deriv.clone();

        assert_eq!(deriv.description(), cloned.description());
    }

    #[test]
    fn test_module_clone() {
        let module = DerivationModule::<Rational>::new("Q(x)".to_string());
        let cloned = module.clone();

        assert_eq!(module.function_field(), cloned.function_field());
        assert_eq!(module.dimension(), cloned.dimension());
    }
}
