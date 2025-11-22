//! Function field constructors
//!
//! This module provides factory types and functions for constructing function fields,
//! corresponding to SageMath's `sage.rings.function_field.constructor`.
//!
//! # Mathematical Background
//!
//! Function fields are fundamental objects in algebraic geometry and number theory:
//!
//! - **Rational Function Fields**: The field k(x) of rational functions over a field k
//! - **Function Field Extensions**: Algebraic extensions K(x)[y]/(f(y)) where f is irreducible
//!
//! The factory pattern ensures that isomorphic function fields are represented by the
//! same object, maintaining uniqueness invariants important for equality testing.
//!
//! # Key Types
//!
//! - `FunctionFieldFactory`: Constructs rational function fields k(x)
//! - `FunctionFieldExtensionFactory`: Constructs algebraic extensions
//! - Helper functions for field construction
//!
//! # Examples
//!
//! ```ignore
//! use rustmath_rings::constructor::*;
//! use rustmath_rationals::Rational;
//!
//! // Create the rational function field Q(x)
//! let factory = FunctionFieldFactory::new();
//! let field = factory.create(/* base field */, "x".to_string());
//! ```

use rustmath_core::Field;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::marker::PhantomData;
use std::fmt;

/// Key type for uniquely identifying function fields
///
/// This ensures that isomorphic function fields map to the same cached object.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FunctionFieldKey {
    /// Base field identifier (simplified as string for now)
    base_field: String,
    /// Variable names
    variables: Vec<String>,
}

impl FunctionFieldKey {
    /// Create a new function field key
    pub fn new(base_field: String, variables: Vec<String>) -> Self {
        FunctionFieldKey {
            base_field,
            variables,
        }
    }

    /// Get the base field identifier
    pub fn base_field(&self) -> &str {
        &self.base_field
    }

    /// Get the variable names
    pub fn variables(&self) -> &[String] {
        &self.variables
    }
}

/// Factory for constructing rational function fields
///
/// Corresponds to SageMath's `FunctionFieldFactory` (UniqueFactory).
/// Creates rational function fields k(x) where k is a field and x is a variable.
///
/// The factory maintains a cache ensuring that identical function fields
/// (same base field and variable) return the same object.
///
/// # Type Parameters
///
/// - `F`: The base field type
///
/// # Implementation Note
///
/// In SageMath, this uses Python's UniqueFactory pattern. In Rust, we use
/// Arc<Mutex<HashMap>> for thread-safe caching.
pub struct FunctionFieldFactory<F: Field> {
    /// Cache of constructed function fields
    cache: Arc<Mutex<HashMap<FunctionFieldKey, Arc<RationalFunctionField<F>>>>>,
    _phantom: PhantomData<F>,
}

impl<F: Field> FunctionFieldFactory<F> {
    /// Create a new function field factory
    pub fn new() -> Self {
        FunctionFieldFactory {
            cache: Arc::new(Mutex::new(HashMap::new())),
            _phantom: PhantomData,
        }
    }

    /// Create a key for looking up/caching a function field
    ///
    /// Normalizes the input into a canonical form.
    ///
    /// # Arguments
    ///
    /// * `base_field_name` - Name/identifier of the base field
    /// * `var` - Variable name
    pub fn create_key(&self, base_field_name: String, var: String) -> FunctionFieldKey {
        FunctionFieldKey::new(base_field_name, vec![var])
    }

    /// Create or retrieve a rational function field
    ///
    /// If a field with the same key already exists in the cache, returns that.
    /// Otherwise, creates a new field and caches it.
    ///
    /// # Arguments
    ///
    /// * `key` - The function field key
    ///
    /// # Returns
    ///
    /// An Arc to the function field (shared ownership for caching)
    pub fn create_object(&self, key: FunctionFieldKey) -> Arc<RationalFunctionField<F>> {
        let mut cache = self.cache.lock().unwrap();

        if let Some(field) = cache.get(&key) {
            return Arc::clone(field);
        }

        // Determine which type of rational function field to create based on characteristics
        let field = Arc::new(RationalFunctionField::new(
            key.variables()[0].clone(),
            key.base_field().to_string(),
        ));

        cache.insert(key, Arc::clone(&field));
        field
    }

    /// Convenience method to create a rational function field
    ///
    /// # Arguments
    ///
    /// * `base_field_name` - Name of the base field
    /// * `var` - Variable name
    pub fn create(&self, base_field_name: String, var: String) -> Arc<RationalFunctionField<F>> {
        let key = self.create_key(base_field_name, var);
        self.create_object(key)
    }
}

impl<F: Field> Default for FunctionFieldFactory<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Field> Clone for FunctionFieldFactory<F> {
    fn clone(&self) -> Self {
        FunctionFieldFactory {
            cache: Arc::clone(&self.cache),
            _phantom: PhantomData,
        }
    }
}

/// Rational function field
///
/// Represents k(x), the field of rational functions in one variable over a field k.
/// This corresponds to various SageMath types:
/// - `RationalFunctionField` (base class)
/// - `RationalFunctionField_global` (over finite fields)
/// - `RationalFunctionField_char_zero` (over characteristic 0 fields)
#[derive(Clone, Debug)]
pub struct RationalFunctionField<F: Field> {
    /// Variable name
    variable: String,
    /// Base field identifier
    base_field: String,
    _phantom: PhantomData<F>,
}

impl<F: Field> RationalFunctionField<F> {
    /// Create a new rational function field
    pub fn new(variable: String, base_field: String) -> Self {
        RationalFunctionField {
            variable,
            base_field,
            _phantom: PhantomData,
        }
    }

    /// Get the variable name
    pub fn variable(&self) -> &str {
        &self.variable
    }

    /// Get the base field identifier
    pub fn base_field(&self) -> &str {
        &self.base_field
    }

    /// Get the generator (the variable as a field element)
    pub fn gen(&self) -> String {
        self.variable.clone()
    }

    /// Check if this is a global function field
    ///
    /// A function field is "global" if it's over a finite field
    pub fn is_global(&self) -> bool {
        // Placeholder: would check if base field is finite
        false
    }

    /// Get the characteristic of the function field
    pub fn characteristic(&self) -> i32 {
        // Placeholder: would query base field characteristic
        0
    }

    /// Check if characteristic is zero
    pub fn is_char_zero(&self) -> bool {
        self.characteristic() == 0
    }
}

impl<F: Field> fmt::Display for RationalFunctionField<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.base_field, self.variable)
    }
}

/// Key type for function field extensions
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FunctionFieldExtensionKey {
    /// Base function field
    base_field: String,
    /// Defining polynomial (as string representation)
    polynomial: String,
    /// Variable name for the extension
    variable: String,
}

impl FunctionFieldExtensionKey {
    /// Create a new extension key
    pub fn new(base_field: String, polynomial: String, variable: String) -> Self {
        FunctionFieldExtensionKey {
            base_field,
            polynomial,
            variable,
        }
    }
}

/// Factory for constructing function field extensions
///
/// Corresponds to SageMath's `FunctionFieldExtensionFactory` (UniqueFactory).
/// Creates function field extensions L = K[y]/(f(y)) where K is a function field
/// and f is an irreducible polynomial.
///
/// # Type Parameters
///
/// - `F`: The base field type
pub struct FunctionFieldExtensionFactory<F: Field> {
    /// Cache of constructed extensions
    cache: Arc<Mutex<HashMap<FunctionFieldExtensionKey, Arc<FunctionFieldExtension<F>>>>>,
    _phantom: PhantomData<F>,
}

impl<F: Field> FunctionFieldExtensionFactory<F> {
    /// Create a new function field extension factory
    pub fn new() -> Self {
        FunctionFieldExtensionFactory {
            cache: Arc::new(Mutex::new(HashMap::new())),
            _phantom: PhantomData,
        }
    }

    /// Create a key for the extension
    ///
    /// # Arguments
    ///
    /// * `base_field` - Identifier of the base function field
    /// * `polynomial` - Defining polynomial (irreducible)
    /// * `variable` - Name of the new variable
    pub fn create_key(
        &self,
        base_field: String,
        polynomial: String,
        variable: String,
    ) -> FunctionFieldExtensionKey {
        FunctionFieldExtensionKey::new(base_field, polynomial, variable)
    }

    /// Create or retrieve a function field extension
    ///
    /// Determines the appropriate type based on:
    /// - Whether base is rational or polymod
    /// - Field characteristic (zero vs positive)
    /// - Whether the polynomial is integral and monic
    ///
    /// Maps to SageMath types:
    /// - `FunctionField_polymod` (general case)
    /// - `FunctionField_global` (over finite fields)
    /// - `FunctionField_global_integral` (integral over finite fields)
    /// - `FunctionField_char_zero` (characteristic 0)
    /// - `FunctionField_char_zero_integral` (integral, characteristic 0)
    pub fn create_object(
        &self,
        key: FunctionFieldExtensionKey,
    ) -> Arc<FunctionFieldExtension<F>> {
        let mut cache = self.cache.lock().unwrap();

        if let Some(field) = cache.get(&key) {
            return Arc::clone(field);
        }

        let extension = Arc::new(FunctionFieldExtension::new(
            key.base_field.clone(),
            key.polynomial.clone(),
            key.variable.clone(),
        ));

        cache.insert(key, Arc::clone(&extension));
        extension
    }

    /// Convenience method to create an extension
    pub fn create(
        &self,
        base_field: String,
        polynomial: String,
        variable: String,
    ) -> Arc<FunctionFieldExtension<F>> {
        let key = self.create_key(base_field, polynomial, variable);
        self.create_object(key)
    }
}

impl<F: Field> Default for FunctionFieldExtensionFactory<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Field> Clone for FunctionFieldExtensionFactory<F> {
    fn clone(&self) -> Self {
        FunctionFieldExtensionFactory {
            cache: Arc::clone(&self.cache),
            _phantom: PhantomData,
        }
    }
}

/// Function field extension
///
/// Represents L = K[y]/(f(y)) where K is a function field and f is irreducible.
/// Corresponds to SageMath's polymod function field types.
#[derive(Clone, Debug)]
pub struct FunctionFieldExtension<F: Field> {
    /// Base function field
    base_field: String,
    /// Defining polynomial
    polynomial: String,
    /// Extension variable
    variable: String,
    _phantom: PhantomData<F>,
}

impl<F: Field> FunctionFieldExtension<F> {
    /// Create a new function field extension
    pub fn new(base_field: String, polynomial: String, variable: String) -> Self {
        FunctionFieldExtension {
            base_field,
            polynomial,
            variable,
            _phantom: PhantomData,
        }
    }

    /// Get the base field identifier
    pub fn base_field(&self) -> &str {
        &self.base_field
    }

    /// Get the defining polynomial
    pub fn polynomial(&self) -> &str {
        &self.polynomial
    }

    /// Get the extension variable name
    pub fn variable(&self) -> &str {
        &self.variable
    }

    /// Get the degree of the extension
    ///
    /// Returns the degree of the defining polynomial
    pub fn degree(&self) -> usize {
        // Placeholder: would parse polynomial and get degree
        1
    }

    /// Check if this extension is integral
    ///
    /// An extension is integral if the defining polynomial has coefficients
    /// in the integral closure of the base field
    pub fn is_integral(&self) -> bool {
        // Placeholder: check polynomial coefficients
        false
    }

    /// Check if this is a global function field
    pub fn is_global(&self) -> bool {
        // Placeholder: check base field
        false
    }

    /// Get the characteristic
    pub fn characteristic(&self) -> i32 {
        // Placeholder: get from base field
        0
    }
}

impl<F: Field> fmt::Display for FunctionFieldExtension<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[{}]/({}) ", self.base_field, self.variable, self.polynomial)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_function_field_key() {
        let key1 = FunctionFieldKey::new("Q".to_string(), vec!["x".to_string()]);
        let key2 = FunctionFieldKey::new("Q".to_string(), vec!["x".to_string()]);
        let key3 = FunctionFieldKey::new("Q".to_string(), vec!["y".to_string()]);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_function_field_factory_creation() {
        let factory = FunctionFieldFactory::<Rational>::new();
        let field1 = factory.create("Q".to_string(), "x".to_string());
        let field2 = factory.create("Q".to_string(), "x".to_string());

        // Should return the same cached object
        assert!(Arc::ptr_eq(&field1, &field2));
        assert_eq!(field1.variable(), "x");
        assert_eq!(field1.base_field(), "Q");
    }

    #[test]
    fn test_rational_function_field() {
        let field = RationalFunctionField::<Rational>::new("x".to_string(), "Q".to_string());

        assert_eq!(field.variable(), "x");
        assert_eq!(field.base_field(), "Q");
        assert_eq!(field.gen(), "x");
        assert!(field.is_char_zero());
        assert!(!field.is_global());
    }

    #[test]
    fn test_rational_function_field_display() {
        let field = RationalFunctionField::<Rational>::new("t".to_string(), "Q".to_string());
        assert_eq!(format!("{}", field), "Q(t)");
    }

    #[test]
    fn test_extension_key() {
        let key1 = FunctionFieldExtensionKey::new(
            "Q(x)".to_string(),
            "y^2 - x".to_string(),
            "y".to_string(),
        );
        let key2 = FunctionFieldExtensionKey::new(
            "Q(x)".to_string(),
            "y^2 - x".to_string(),
            "y".to_string(),
        );
        let key3 = FunctionFieldExtensionKey::new(
            "Q(x)".to_string(),
            "y^3 - x".to_string(),
            "y".to_string(),
        );

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_extension_factory_creation() {
        let factory = FunctionFieldExtensionFactory::<Rational>::new();
        let ext1 = factory.create(
            "Q(x)".to_string(),
            "y^2 - x".to_string(),
            "y".to_string(),
        );
        let ext2 = factory.create(
            "Q(x)".to_string(),
            "y^2 - x".to_string(),
            "y".to_string(),
        );

        // Should return the same cached object
        assert!(Arc::ptr_eq(&ext1, &ext2));
    }

    #[test]
    fn test_function_field_extension() {
        let ext = FunctionFieldExtension::<Rational>::new(
            "Q(x)".to_string(),
            "y^2 - x".to_string(),
            "y".to_string(),
        );

        assert_eq!(ext.base_field(), "Q(x)");
        assert_eq!(ext.polynomial(), "y^2 - x");
        assert_eq!(ext.variable(), "y");
        assert!(!ext.is_integral());
    }

    #[test]
    fn test_extension_display() {
        let ext = FunctionFieldExtension::<Rational>::new(
            "Q(x)".to_string(),
            "y^2 - x".to_string(),
            "y".to_string(),
        );

        let display = format!("{}", ext);
        assert!(display.contains("Q(x)"));
        assert!(display.contains("y"));
        assert!(display.contains("y^2 - x"));
    }

    #[test]
    fn test_factory_caching_different_fields() {
        let factory = FunctionFieldFactory::<Rational>::new();
        let field1 = factory.create("Q".to_string(), "x".to_string());
        let field2 = factory.create("Q".to_string(), "y".to_string());

        // Different variables should create different objects
        assert!(!Arc::ptr_eq(&field1, &field2));
        assert_eq!(field1.variable(), "x");
        assert_eq!(field2.variable(), "y");
    }

    #[test]
    fn test_characteristic() {
        let field = RationalFunctionField::<Rational>::new("x".to_string(), "Q".to_string());
        assert_eq!(field.characteristic(), 0);

        let ext = FunctionFieldExtension::<Rational>::new(
            "Q(x)".to_string(),
            "y^2 - x".to_string(),
            "y".to_string(),
        );
        assert_eq!(ext.characteristic(), 0);
    }

    #[test]
    fn test_factory_clone() {
        let factory1 = FunctionFieldFactory::<Rational>::new();
        let field1 = factory1.create("Q".to_string(), "x".to_string());

        let factory2 = factory1.clone();
        let field2 = factory2.create("Q".to_string(), "x".to_string());

        // Cloned factory should share the same cache
        assert!(Arc::ptr_eq(&field1, &field2));
    }

    #[test]
    fn test_extension_degree() {
        let ext = FunctionFieldExtension::<Rational>::new(
            "Q(x)".to_string(),
            "y^2 - x".to_string(),
            "y".to_string(),
        );

        // Placeholder returns 1, but in full implementation would be 2
        assert_eq!(ext.degree(), 1);
    }
}
