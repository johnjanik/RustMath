//! Function field base module
//!
//! This module provides the base FunctionField trait and related functionality,
//! corresponding to SageMath's `sage.rings.function_field.function_field` module.
//!
//! # Mathematical Background
//!
//! A function field K over a field k is a finitely generated field extension of
//! transcendence degree 1. The prototypical example is the rational function field
//! k(x) = Frac(k[x]).
//!
//! ## Key Properties
//!
//! - **Transcendence degree 1**: K/k has exactly one algebraically independent element
//! - **Finitely generated**: K = k(x_1, ..., x_n) for some generators
//! - **Constant field**: The algebraic closure of k in K
//!
//! ## Function Field Tower
//!
//! Function fields form a tower structure:
//! - Base: Rational function field k(x)
//! - Extensions: L = K[y]/(f(y)) for irreducible f ∈ K[y]
//! - Iterations: Can build towers of arbitrary height
//!
//! ## Algebraic-Geometric Connection
//!
//! A function field K/k corresponds to:
//! - An algebraic curve C over k
//! - K = k(C) is the field of rational functions on C
//! - Places of K ↔ Points of C
//! - Divisors on K ↔ Divisors on C
//!
//! # Key Structures
//!
//! - `FunctionField`: Base trait for all function fields
//! - `is_function_field`: Type checking function

use rustmath_core::Field;
use std::marker::PhantomData;
use std::fmt;

/// Base trait for function fields
///
/// This corresponds to SageMath's `FunctionField` class.
///
/// # Type Parameters
///
/// - `F`: The constant field type
///
/// # Requirements
///
/// Implementors must provide:
/// - Constant field access
/// - Genus computation
/// - Place enumeration
/// - Element operations
pub trait FunctionField<F: Field> {
    /// Get the constant field
    ///
    /// # Returns
    ///
    /// String representation of the constant field
    fn constant_field(&self) -> String;

    /// Compute the genus
    ///
    /// # Returns
    ///
    /// Genus of the function field
    fn genus(&self) -> Option<usize>;

    /// Get the number of variables
    ///
    /// # Returns
    ///
    /// Number of generators (transcendence degree usually 1)
    fn num_variables(&self) -> usize;

    /// Get a generator
    ///
    /// # Arguments
    ///
    /// * `index` - Index of the generator
    ///
    /// # Returns
    ///
    /// String representation of the generator
    fn generator(&self, index: usize) -> Option<String>;

    /// Check if this is a rational function field
    ///
    /// # Returns
    ///
    /// True if this is k(x) for some x
    fn is_rational(&self) -> bool;

    /// Check if this is a polynomial extension
    ///
    /// # Returns
    ///
    /// True if this is K[y]/(f(y)) for some K and f
    fn is_extension(&self) -> bool {
        !self.is_rational()
    }

    /// Get the base field (for extensions)
    ///
    /// # Returns
    ///
    /// String representation of the base field
    fn base_field(&self) -> Option<String>;

    /// Get the extension degree (for extensions)
    ///
    /// # Returns
    ///
    /// Degree of the extension
    fn extension_degree(&self) -> Option<usize>;

    /// Get the defining polynomial (for extensions)
    ///
    /// # Returns
    ///
    /// String representation of the defining polynomial
    fn defining_polynomial(&self) -> Option<String>;

    /// Convert element to string
    ///
    /// # Arguments
    ///
    /// * `element` - Generic element representation
    ///
    /// # Returns
    ///
    /// String representation of the element
    fn element_to_string(&self, element: &str) -> String {
        element.to_string()
    }

    /// Compute derivative with respect to main variable
    ///
    /// # Arguments
    ///
    /// * `element` - String representation of the element
    ///
    /// # Returns
    ///
    /// String representation of the derivative
    fn derivative(&self, element: &str) -> String;
}

/// Base structure for function fields
///
/// Generic implementation of common function field properties.
///
/// # Type Parameters
///
/// - `F`: The constant field type
#[derive(Clone, Debug)]
pub struct FunctionFieldBase<F: Field> {
    /// Name of the function field
    name: String,
    /// Constant field name
    constant_field: String,
    /// Generators
    generators: Vec<String>,
    /// Genus (if known)
    genus: Option<usize>,
    /// Whether this is rational
    is_rational: bool,
    /// Field marker
    field_marker: PhantomData<F>,
}

impl<F: Field> FunctionFieldBase<F> {
    /// Create a new function field base
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the function field
    /// * `constant_field` - Name of the constant field
    /// * `generators` - List of generators
    /// * `is_rational` - Whether this is a rational function field
    ///
    /// # Returns
    ///
    /// A new FunctionFieldBase instance
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let qx = FunctionFieldBase::new(
    ///     "Q(x)".to_string(),
    ///     "Q".to_string(),
    ///     vec!["x".to_string()],
    ///     true
    /// );
    /// ```
    pub fn new(
        name: String,
        constant_field: String,
        generators: Vec<String>,
        is_rational: bool,
    ) -> Self {
        FunctionFieldBase {
            name,
            constant_field,
            generators,
            genus: None,
            is_rational,
            field_marker: PhantomData,
        }
    }

    /// Create a rational function field k(x)
    ///
    /// # Arguments
    ///
    /// * `constant_field` - Name of the constant field
    /// * `variable` - Name of the variable
    ///
    /// # Returns
    ///
    /// A rational function field
    pub fn rational(constant_field: String, variable: String) -> Self {
        let name = format!("{}({})", constant_field, variable);
        FunctionFieldBase::new(
            name,
            constant_field,
            vec![variable],
            true,
        )
    }

    /// Get the name
    ///
    /// # Returns
    ///
    /// Name of the function field
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Set the genus
    ///
    /// # Arguments
    ///
    /// * `g` - Genus value
    pub fn set_genus(&mut self, g: usize) {
        self.genus = Some(g);
    }
}

impl<F: Field> FunctionField<F> for FunctionFieldBase<F> {
    fn constant_field(&self) -> String {
        self.constant_field.clone()
    }

    fn genus(&self) -> Option<usize> {
        self.genus
    }

    fn num_variables(&self) -> usize {
        self.generators.len()
    }

    fn generator(&self, index: usize) -> Option<String> {
        self.generators.get(index).cloned()
    }

    fn is_rational(&self) -> bool {
        self.is_rational
    }

    fn base_field(&self) -> Option<String> {
        if self.is_rational {
            None
        } else {
            Some(self.constant_field.clone())
        }
    }

    fn extension_degree(&self) -> Option<usize> {
        if self.is_rational {
            None
        } else {
            Some(1) // Simplified
        }
    }

    fn defining_polynomial(&self) -> Option<String> {
        None // Base class doesn't have defining polynomial
    }

    fn derivative(&self, element: &str) -> String {
        if let Some(var) = self.generator(0) {
            format!("d({})/d{}", element, var)
        } else {
            format!("d({})", element)
        }
    }
}

impl<F: Field> fmt::Display for FunctionFieldBase<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Function field {}", self.name)?;
        if let Some(g) = self.genus {
            write!(f, " (genus {})", g)?;
        }
        Ok(())
    }
}

/// Check if an object represents a function field
///
/// This corresponds to SageMath's `is_FunctionField` function.
///
/// # Arguments
///
/// * `name` - String representation to check
///
/// # Returns
///
/// True if the string represents a function field
///
/// # Examples
///
/// ```ignore
/// assert!(is_function_field("Q(x)"));
/// assert!(is_function_field("k(x,y)"));
/// assert!(!is_function_field("Q[x]"));
/// ```
pub fn is_function_field(name: &str) -> bool {
    // Simple heuristic: contains '(' and ')' suggesting field of fractions
    name.contains('(') && name.contains(')')
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_function_field_base_creation() {
        let ff: FunctionFieldBase<Rational> = FunctionFieldBase::new(
            "Q(x)".to_string(),
            "Q".to_string(),
            vec!["x".to_string()],
            true,
        );

        assert_eq!(ff.name(), "Q(x)");
        assert_eq!(ff.constant_field(), "Q");
        assert_eq!(ff.num_variables(), 1);
        assert!(ff.is_rational());
    }

    #[test]
    fn test_rational_function_field() {
        let qx: FunctionFieldBase<Rational> =
            FunctionFieldBase::rational("Q".to_string(), "x".to_string());

        assert_eq!(qx.name(), "Q(x)");
        assert!(qx.is_rational());
        assert_eq!(qx.num_variables(), 1);
        assert_eq!(qx.generator(0), Some("x".to_string()));
    }

    #[test]
    fn test_genus() {
        let mut ff: FunctionFieldBase<Rational> =
            FunctionFieldBase::rational("Q".to_string(), "x".to_string());

        assert_eq!(ff.genus(), None);
        ff.set_genus(0);
        assert_eq!(ff.genus(), Some(0));
    }

    #[test]
    fn test_generators() {
        let ff: FunctionFieldBase<Rational> = FunctionFieldBase::new(
            "Q(x,y)".to_string(),
            "Q".to_string(),
            vec!["x".to_string(), "y".to_string()],
            false,
        );

        assert_eq!(ff.num_variables(), 2);
        assert_eq!(ff.generator(0), Some("x".to_string()));
        assert_eq!(ff.generator(1), Some("y".to_string()));
        assert_eq!(ff.generator(2), None);
    }

    #[test]
    fn test_is_rational() {
        let rational: FunctionFieldBase<Rational> =
            FunctionFieldBase::rational("k".to_string(), "t".to_string());
        let extension: FunctionFieldBase<Rational> = FunctionFieldBase::new(
            "K(y)".to_string(),
            "k".to_string(),
            vec!["y".to_string()],
            false,
        );

        assert!(rational.is_rational());
        assert!(!extension.is_rational());
        assert!(extension.is_extension());
    }

    #[test]
    fn test_base_field() {
        let rational: FunctionFieldBase<Rational> =
            FunctionFieldBase::rational("Q".to_string(), "x".to_string());
        let extension: FunctionFieldBase<Rational> = FunctionFieldBase::new(
            "L".to_string(),
            "Q".to_string(),
            vec!["y".to_string()],
            false,
        );

        assert_eq!(rational.base_field(), None);
        assert_eq!(extension.base_field(), Some("Q".to_string()));
    }

    #[test]
    fn test_derivative() {
        let ff: FunctionFieldBase<Rational> =
            FunctionFieldBase::rational("Q".to_string(), "x".to_string());

        let deriv = ff.derivative("x^2");
        assert!(deriv.contains("d(x^2)"));
        assert!(deriv.contains("dx"));
    }

    #[test]
    fn test_is_function_field() {
        assert!(is_function_field("Q(x)"));
        assert!(is_function_field("k(x,y)"));
        assert!(is_function_field("F2(t)"));
        assert!(!is_function_field("Q[x]"));
        assert!(!is_function_field("ZZ"));
        assert!(!is_function_field("k"));
    }

    #[test]
    fn test_display() {
        let mut ff: FunctionFieldBase<Rational> =
            FunctionFieldBase::rational("Q".to_string(), "x".to_string());

        let display1 = format!("{}", ff);
        assert!(display1.contains("Q(x)"));

        ff.set_genus(1);
        let display2 = format!("{}", ff);
        assert!(display2.contains("genus 1"));
    }

    #[test]
    fn test_clone() {
        let ff1: FunctionFieldBase<Rational> =
            FunctionFieldBase::rational("Q".to_string(), "x".to_string());
        let ff2 = ff1.clone();

        assert_eq!(ff1.name(), ff2.name());
        assert_eq!(ff1.constant_field(), ff2.constant_field());
    }

    #[test]
    fn test_extension_properties() {
        let ext: FunctionFieldBase<Rational> = FunctionFieldBase::new(
            "L".to_string(),
            "k".to_string(),
            vec!["y".to_string()],
            false,
        );

        assert!(ext.is_extension());
        assert_eq!(ext.extension_degree(), Some(1));
        assert_eq!(ext.defining_polynomial(), None);
    }

    #[test]
    fn test_rational_function_field_genus() {
        let mut qx: FunctionFieldBase<Rational> =
            FunctionFieldBase::rational("Q".to_string(), "x".to_string());

        // Rational function fields have genus 0
        qx.set_genus(0);
        assert_eq!(qx.genus(), Some(0));
    }

    #[test]
    fn test_element_to_string() {
        let ff: FunctionFieldBase<Rational> =
            FunctionFieldBase::rational("Q".to_string(), "x".to_string());

        assert_eq!(ff.element_to_string("x^2 + 1"), "x^2 + 1");
    }
}
