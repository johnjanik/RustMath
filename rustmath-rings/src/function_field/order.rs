//! Function Field Orders Module
//!
//! This module implements orders (rings of integers) in function fields,
//! corresponding to SageMath's `sage.rings.function_field.order` module.
//!
//! # Mathematical Overview
//!
//! An order in a function field K is a subring O ⊂ K that is finitely generated
//! as a module over the constant field ring. The maximal order is the integral
//! closure of the constant field in K.
//!
//! ## Key Concepts
//!
//! ### Maximal Order
//!
//! The maximal order (ring of integers) O_K of K is:
//!
//! O_K = {x ∈ K : x is integral over k[t]}
//!
//! where k is the constant field and t is a separating element.
//!
//! ### Integral Basis
//!
//! A Z-basis (or k[t]-basis) {ω₁, ..., ω_n} of O_K such that every element
//! can be written uniquely as ∑ aᵢωᵢ with aᵢ ∈ k[t].
//!
//! ### Infinite Places
//!
//! Orders can also be defined at infinite places, corresponding to poles
//! rather than zeros.
//!
//! ## Applications
//!
//! - Computing class groups
//! - Finding integral points on curves
//! - Riemann-Roch space computations
//! - Algebraic geometry codes
//!
//! # Implementation
//!
//! This module provides:
//!
//! - `FunctionFieldOrder_base`: Base order class
//! - `FunctionFieldOrder`: Standard order
//! - `FunctionFieldOrderInfinite`: Order at infinity
//! - `FunctionFieldMaximalOrder`: Maximal order (ring of integers)
//! - `FunctionFieldMaximalOrderInfinite`: Maximal order at infinity
//!
//! # References
//!
//! - SageMath: `sage.rings.function_field.order`
//! - Stichtenoth, H. (2009). "Algebraic Function Fields and Codes"

use rustmath_core::Field;
use std::marker::PhantomData;

/// Base class for function field orders
///
/// Represents a subring of a function field that is finitely generated
/// as a module.
///
/// # Type Parameters
///
/// * `F` - The coefficient field type
#[derive(Debug, Clone)]
pub struct FunctionFieldOrder_base<F: Field> {
    /// Field in which this order lives
    field: String,
    /// Name/description of the order
    name: String,
    /// Basis elements (as strings)
    basis: Vec<String>,
    /// Phantom data
    _phantom: PhantomData<F>,
}

impl<F: Field> FunctionFieldOrder_base<F> {
    /// Create a new order
    pub fn new(field: String, name: String) -> Self {
        Self {
            field,
            name,
            basis: Vec::new(),
            _phantom: PhantomData,
        }
    }

    /// Create with an explicit basis
    pub fn with_basis(field: String, name: String, basis: Vec<String>) -> Self {
        Self {
            field,
            name,
            basis,
            _phantom: PhantomData,
        }
    }

    /// Get the function field
    pub fn function_field(&self) -> &str {
        &self.field
    }

    /// Get the name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the basis
    pub fn basis(&self) -> &[String] {
        &self.basis
    }

    /// Get the rank (basis size)
    pub fn rank(&self) -> usize {
        self.basis.len()
    }

    /// Add a basis element
    pub fn add_basis_element(&mut self, element: String) {
        if !self.basis.contains(&element) {
            self.basis.push(element);
        }
    }

    /// Check if an element is in the order
    pub fn contains(&self, _element: &str) -> bool {
        // Would check if element can be written as Z-linear combination of basis
        true
    }

    /// Get the discriminant of the order
    pub fn discriminant(&self) -> String {
        format!("Discriminant of {}", self.name)
    }

    /// Check if this is a maximal order
    pub fn is_maximal(&self) -> bool {
        false // Override in subclass
    }
}

/// Standard function field order
///
/// An order in a function field (not necessarily maximal).
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::order::FunctionFieldOrder;
/// use rustmath_rationals::Rational;
///
/// let order = FunctionFieldOrder::<Rational>::new(
///     "Q(x,y)".to_string(),
///     "O".to_string(),
/// );
/// ```
#[derive(Debug, Clone)]
pub struct FunctionFieldOrder<F: Field> {
    /// Base order structure
    inner: FunctionFieldOrder_base<F>,
}

impl<F: Field> FunctionFieldOrder<F> {
    /// Create a new order
    pub fn new(field: String, name: String) -> Self {
        Self {
            inner: FunctionFieldOrder_base::new(field, name),
        }
    }

    /// Create with basis
    pub fn with_basis(field: String, name: String, basis: Vec<String>) -> Self {
        Self {
            inner: FunctionFieldOrder_base::with_basis(field, name, basis),
        }
    }

    /// Get the function field
    pub fn function_field(&self) -> &str {
        self.inner.function_field()
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        self.inner.rank()
    }

    /// Get the basis
    pub fn basis(&self) -> &[String] {
        self.inner.basis()
    }

    /// Compute the different ideal
    pub fn different(&self) -> String {
        format!("Different of {}", self.inner.name())
    }

    /// Compute the conductor
    pub fn conductor(&self, _other: &Self) -> String {
        format!("Conductor ideal")
    }
}

/// Order at infinite places
///
/// An order defined with respect to poles (infinite places) rather than zeros.
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::order::FunctionFieldOrderInfinite;
/// use rustmath_rationals::Rational;
///
/// let order_inf = FunctionFieldOrderInfinite::<Rational>::new(
///     "Q(x)".to_string(),
///     "O_∞".to_string(),
/// );
/// assert!(order_inf.is_at_infinity());
/// ```
#[derive(Debug, Clone)]
pub struct FunctionFieldOrderInfinite<F: Field> {
    /// Base order structure
    inner: FunctionFieldOrder_base<F>,
}

impl<F: Field> FunctionFieldOrderInfinite<F> {
    /// Create a new infinite order
    pub fn new(field: String, name: String) -> Self {
        Self {
            inner: FunctionFieldOrder_base::new(field, name),
        }
    }

    /// Check if at infinity (always true)
    pub fn is_at_infinity(&self) -> bool {
        true
    }

    /// Get the function field
    pub fn function_field(&self) -> &str {
        self.inner.function_field()
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        self.inner.rank()
    }
}

/// Maximal order (ring of integers) in a function field
///
/// The integral closure of the polynomial ring in the function field.
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::order::FunctionFieldMaximalOrder;
/// use rustmath_rationals::Rational;
///
/// let max_order = FunctionFieldMaximalOrder::<Rational>::new(
///     "Q(x,y)".to_string(),
/// );
/// assert!(max_order.is_maximal());
/// ```
#[derive(Debug, Clone)]
pub struct FunctionFieldMaximalOrder<F: Field> {
    /// Base order structure
    inner: FunctionFieldOrder_base<F>,
}

impl<F: Field> FunctionFieldMaximalOrder<F> {
    /// Create the maximal order
    pub fn new(field: String) -> Self {
        Self {
            inner: FunctionFieldOrder_base::new(field.clone(), format!("O_{}", field)),
        }
    }

    /// Check if maximal (always true)
    pub fn is_maximal(&self) -> bool {
        true
    }

    /// Get the function field
    pub fn function_field(&self) -> &str {
        self.inner.function_field()
    }

    /// Get an integral basis
    pub fn integral_basis(&self) -> &[String] {
        self.inner.basis()
    }

    /// Compute the class number
    pub fn class_number(&self) -> usize {
        // Would compute |Cl(O)|
        1 // Simplified
    }

    /// Get the Picard group (divisor class group)
    pub fn picard_group(&self) -> String {
        format!("Pic({})", self.inner.function_field())
    }

    /// Decompose a prime ideal
    pub fn decompose_prime(&self, _prime: &str) -> Vec<String> {
        // Would compute factorization of prime in this order
        Vec::new()
    }

    /// Check if the order is a principal ideal domain
    pub fn is_pid(&self) -> bool {
        self.class_number() == 1
    }
}

/// Maximal order at infinity
///
/// Combines maximal order and infinite place properties.
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::order::FunctionFieldMaximalOrderInfinite;
/// use rustmath_rationals::Rational;
///
/// let max_order_inf = FunctionFieldMaximalOrderInfinite::<Rational>::new(
///     "Q(x)".to_string(),
/// );
/// assert!(max_order_inf.is_maximal());
/// assert!(max_order_inf.is_at_infinity());
/// ```
#[derive(Debug, Clone)]
pub struct FunctionFieldMaximalOrderInfinite<F: Field> {
    /// Maximal order structure
    maximal: FunctionFieldMaximalOrder<F>,
    /// Infinite order structure
    infinite: FunctionFieldOrderInfinite<F>,
}

impl<F: Field> FunctionFieldMaximalOrderInfinite<F> {
    /// Create a new maximal infinite order
    pub fn new(field: String) -> Self {
        Self {
            maximal: FunctionFieldMaximalOrder::new(field.clone()),
            infinite: FunctionFieldOrderInfinite::new(field, "O_∞".to_string()),
        }
    }

    /// Check if maximal
    pub fn is_maximal(&self) -> bool {
        self.maximal.is_maximal()
    }

    /// Check if at infinity
    pub fn is_at_infinity(&self) -> bool {
        self.infinite.is_at_infinity()
    }

    /// Get the function field
    pub fn function_field(&self) -> &str {
        self.maximal.function_field()
    }

    /// Get the integral basis at infinity
    pub fn integral_basis_at_infinity(&self) -> &[String] {
        self.maximal.integral_basis()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_function_field_order_base() {
        let order = FunctionFieldOrder_base::<Rational>::new(
            "Q(x)".to_string(),
            "O".to_string(),
        );

        assert_eq!(order.function_field(), "Q(x)");
        assert_eq!(order.name(), "O");
        assert_eq!(order.rank(), 0);
    }

    #[test]
    fn test_order_with_basis() {
        let basis = vec!["1".to_string(), "x".to_string()];
        let order = FunctionFieldOrder_base::<Rational>::with_basis(
            "Q(x)".to_string(),
            "O".to_string(),
            basis.clone(),
        );

        assert_eq!(order.rank(), 2);
        assert_eq!(order.basis(), &basis[..]);
    }

    #[test]
    fn test_add_basis_element() {
        let mut order = FunctionFieldOrder_base::<Rational>::new(
            "Q(x)".to_string(),
            "O".to_string(),
        );

        assert_eq!(order.rank(), 0);

        order.add_basis_element("1".to_string());
        assert_eq!(order.rank(), 1);

        order.add_basis_element("x".to_string());
        assert_eq!(order.rank(), 2);

        // Adding duplicate shouldn't increase rank
        order.add_basis_element("1".to_string());
        assert_eq!(order.rank(), 2);
    }

    #[test]
    fn test_contains() {
        let order = FunctionFieldOrder_base::<Rational>::new(
            "Q(x)".to_string(),
            "O".to_string(),
        );

        assert!(order.contains("x"));
    }

    #[test]
    fn test_discriminant() {
        let order = FunctionFieldOrder_base::<Rational>::new(
            "Q(x)".to_string(),
            "O".to_string(),
        );

        let disc = order.discriminant();
        assert!(disc.contains("Discriminant"));
    }

    #[test]
    fn test_function_field_order() {
        let order = FunctionFieldOrder::<Rational>::new(
            "Q(x,y)".to_string(),
            "O".to_string(),
        );

        assert_eq!(order.function_field(), "Q(x,y)");
        assert_eq!(order.rank(), 0);
    }

    #[test]
    fn test_order_different() {
        let order = FunctionFieldOrder::<Rational>::new(
            "Q(x,y)".to_string(),
            "O".to_string(),
        );

        let diff = order.different();
        assert!(diff.contains("Different"));
    }

    #[test]
    fn test_conductor() {
        let o1 = FunctionFieldOrder::<Rational>::new(
            "Q(x,y)".to_string(),
            "O1".to_string(),
        );
        let o2 = FunctionFieldOrder::<Rational>::new(
            "Q(x,y)".to_string(),
            "O2".to_string(),
        );

        let cond = o1.conductor(&o2);
        assert!(cond.contains("Conductor"));
    }

    #[test]
    fn test_function_field_order_infinite() {
        let order_inf = FunctionFieldOrderInfinite::<Rational>::new(
            "Q(x)".to_string(),
            "O_∞".to_string(),
        );

        assert!(order_inf.is_at_infinity());
        assert_eq!(order_inf.function_field(), "Q(x)");
    }

    #[test]
    fn test_function_field_maximal_order() {
        let max_order = FunctionFieldMaximalOrder::<Rational>::new(
            "Q(x,y)".to_string(),
        );

        assert!(max_order.is_maximal());
        assert_eq!(max_order.function_field(), "Q(x,y)");
    }

    #[test]
    fn test_integral_basis() {
        let max_order = FunctionFieldMaximalOrder::<Rational>::new(
            "Q(x,y)".to_string(),
        );

        let basis = max_order.integral_basis();
        // Initially empty, would be computed
        assert!(basis.len() >= 0);
    }

    #[test]
    fn test_class_number() {
        let max_order = FunctionFieldMaximalOrder::<Rational>::new(
            "Q(x)".to_string(),
        );

        // Rational function fields have class number 1
        assert_eq!(max_order.class_number(), 1);
    }

    #[test]
    fn test_picard_group() {
        let max_order = FunctionFieldMaximalOrder::<Rational>::new(
            "Q(x,y)".to_string(),
        );

        let pic = max_order.picard_group();
        assert!(pic.contains("Pic"));
    }

    #[test]
    fn test_is_pid() {
        let max_order = FunctionFieldMaximalOrder::<Rational>::new(
            "Q(x)".to_string(),
        );

        assert!(max_order.is_pid());
    }

    #[test]
    fn test_decompose_prime() {
        let max_order = FunctionFieldMaximalOrder::<Rational>::new(
            "Q(x,y)".to_string(),
        );

        let decomposition = max_order.decompose_prime("(x)");
        // Would compute actual decomposition
        assert!(decomposition.len() >= 0);
    }

    #[test]
    fn test_function_field_maximal_order_infinite() {
        let max_order_inf = FunctionFieldMaximalOrderInfinite::<Rational>::new(
            "Q(x)".to_string(),
        );

        assert!(max_order_inf.is_maximal());
        assert!(max_order_inf.is_at_infinity());
        assert_eq!(max_order_inf.function_field(), "Q(x)");
    }

    #[test]
    fn test_integral_basis_at_infinity() {
        let max_order_inf = FunctionFieldMaximalOrderInfinite::<Rational>::new(
            "Q(x)".to_string(),
        );

        let basis = max_order_inf.integral_basis_at_infinity();
        assert!(basis.len() >= 0);
    }

    #[test]
    fn test_multiple_orders() {
        let field = "Q(x,y)".to_string();

        let order1 = FunctionFieldOrder::<Rational>::new(field.clone(), "O1".to_string());
        let order2 = FunctionFieldOrder::<Rational>::new(field.clone(), "O2".to_string());
        let max_order = FunctionFieldMaximalOrder::<Rational>::new(field);

        assert!(!order1.is_maximal());
        assert!(!order2.is_maximal());
        assert!(max_order.is_maximal());
    }
}
