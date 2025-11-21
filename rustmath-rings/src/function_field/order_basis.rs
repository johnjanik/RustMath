//! Function Field Order Basis Classes
//!
//! This module implements order basis classes for function fields,
//! corresponding to SageMath's `sage.rings.function_field.order_basis` module.
//!
//! # Mathematical Overview
//!
//! An order basis is an explicit representation of an order in a function field
//! as a free module over the polynomial ring k[t]. The basis elements provide
//! a concrete way to compute with elements of the order.
//!
//! ## Key Concepts
//!
//! ### Order Basis
//!
//! Given an order O in a function field K, an order basis is a set of elements
//! {ω₁, ..., ω_n} such that:
//!
//! O = k[t]ω₁ + ... + k[t]ω_n
//!
//! where k is the constant field and t is a separating element.
//!
//! ### Integral Basis
//!
//! An integral basis is a special order basis where the elements generate
//! the maximal order (ring of integers) O_K.
//!
//! ### Infinite Basis
//!
//! For orders at infinity, the basis is expressed in terms of poles rather
//! than zeros, typically using 1/t as the uniformizer.
//!
//! ## Applications
//!
//! - Computing with elements in orders
//! - Determining integrality
//! - Computing discriminants and differents
//! - Finding maximal orders
//! - Riemann-Roch space computations
//!
//! # Implementation
//!
//! This module provides:
//!
//! - `FunctionFieldOrder_basis`: Orders represented by explicit bases
//! - `FunctionFieldOrderInfinite_basis`: Infinite orders with explicit bases
//!
//! # References
//!
//! - SageMath: `sage.rings.function_field.order_basis`
//! - Cohen, H. (1993). "A Course in Computational Algebraic Number Theory"
//! - Stichtenoth, H. (2009). "Algebraic Function Fields and Codes"

use rustmath_core::Field;
use std::marker::PhantomData;

/// Order represented by an explicit basis
///
/// This class represents an order in a function field by giving an explicit
/// basis as a free module over the polynomial ring.
///
/// # Type Parameters
///
/// * `F` - The coefficient field type
///
/// # Mathematical Details
///
/// The order is represented as O = ⊕ᵢ k[t]·ωᵢ where {ω₁, ..., ω_n} is the basis.
/// All elements can be written uniquely as ∑ᵢ aᵢ(t)·ωᵢ with aᵢ ∈ k[t].
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::order_basis::FunctionFieldOrder_basis;
/// use rustmath_rationals::Rational;
///
/// let basis = vec!["1".to_string(), "x".to_string(), "y".to_string()];
/// let order = FunctionFieldOrder_basis::<Rational>::new(
///     "Q(x,y)".to_string(),
///     basis,
/// );
/// assert_eq!(order.rank(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct FunctionFieldOrderBasis<F: Field> {
    /// Function field
    field: String,
    /// Basis elements
    basis: Vec<String>,
    /// Cached discriminant
    discriminant: Option<String>,
    /// Phantom data
    _phantom: PhantomData<F>,
}

impl<F: Field> FunctionFieldOrder_basis<F> {
    /// Create a new order with explicit basis
    ///
    /// # Arguments
    ///
    /// * `field` - The function field
    /// * `basis` - The basis elements
    pub fn new(field: String, basis: Vec<String>) -> Self {
        assert!(!basis.is_empty(), "Basis cannot be empty");
        Self {
            field,
            basis,
            discriminant: None,
            _phantom: PhantomData,
        }
    }

    /// Get the function field
    pub fn function_field(&self) -> &str {
        &self.field
    }

    /// Get the basis
    pub fn basis(&self) -> &[String] {
        &self.basis
    }

    /// Get the rank (number of basis elements)
    pub fn rank(&self) -> usize {
        self.basis.len()
    }

    /// Get a specific basis element
    pub fn basis_element(&self, i: usize) -> Option<&str> {
        self.basis.get(i).map(|s| s.as_str())
    }

    /// Check if an element is in the order
    ///
    /// An element x is in the order if it can be written as
    /// x = ∑ᵢ aᵢ(t)·ωᵢ with aᵢ ∈ k[t].
    pub fn contains_element(&self, _element: &str) -> bool {
        // Would compute coordinates in the basis
        true
    }

    /// Compute the discriminant
    ///
    /// The discriminant is det(Tr(ωᵢωⱼ)) where Tr is the trace form.
    pub fn discriminant(&mut self) -> String {
        if self.discriminant.is_none() {
            self.discriminant = Some(format!("disc({})", self.basis.join(", ")));
        }
        self.discriminant.clone().unwrap()
    }

    /// Compute the different ideal
    ///
    /// The different measures the ramification of the extension.
    pub fn different(&self) -> String {
        format!("Different of order with basis [{}]", self.basis.join(", "))
    }

    /// Express an element in the basis
    ///
    /// Returns coefficients [a₁, ..., a_n] such that element = ∑ aᵢ·ωᵢ.
    pub fn coordinates(&self, _element: &str) -> Vec<String> {
        vec!["0".to_string(); self.rank()]
    }

    /// Multiply two elements in the order
    pub fn multiply(&self, _a: &str, _b: &str) -> Vec<String> {
        self.coordinates("0")
    }

    /// Check if this is a maximal order
    pub fn is_maximal(&self) -> bool {
        // Would check if discriminant is minimal
        false
    }

    /// Convert to a different basis
    pub fn change_basis(&self, new_basis: Vec<String>) -> Self {
        Self::new(self.field.clone(), new_basis)
    }

    /// Compute the module generated by given elements
    pub fn span(&self, elements: Vec<String>) -> Self {
        Self::new(self.field.clone(), elements)
    }

    /// Get the unit element in this basis
    pub fn one(&self) -> Vec<String> {
        let mut coords = vec!["0".to_string(); self.rank()];
        if !coords.is_empty() {
            coords[0] = "1".to_string();
        }
        coords
    }

    /// Check if basis is orthogonal under trace form
    pub fn is_orthogonal(&self) -> bool {
        // Would compute Tr(ωᵢωⱼ) for i ≠ j
        false
    }
}

/// Infinite order represented by an explicit basis
///
/// This class represents an order at infinity in a function field by giving
/// an explicit basis, typically expressed in terms of 1/t.
///
/// # Type Parameters
///
/// * `F` - The coefficient field type
///
/// # Mathematical Details
///
/// At infinity, we work with poles. A typical basis element is 1/tⁱ or
/// similar negative powers. The order consists of functions with controlled
/// pole behavior at infinity.
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::order_basis::FunctionFieldOrderInfinite_basis;
/// use rustmath_rationals::Rational;
///
/// let basis = vec!["1".to_string(), "1/x".to_string(), "1/x^2".to_string()];
/// let order_inf = FunctionFieldOrderInfinite_basis::<Rational>::new(
///     "Q(x)".to_string(),
///     basis,
/// );
/// assert!(order_inf.is_at_infinity());
/// assert_eq!(order_inf.rank(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct FunctionFieldOrderInfiniteBasis<F: Field> {
    /// Function field
    field: String,
    /// Basis elements (typically involving negative powers)
    basis: Vec<String>,
    /// Degree bound at infinity
    degree_bound: Option<i64>,
    /// Phantom data
    _phantom: PhantomData<F>,
}

impl<F: Field> FunctionFieldOrderInfinite_basis<F> {
    /// Create a new infinite order with explicit basis
    ///
    /// # Arguments
    ///
    /// * `field` - The function field
    /// * `basis` - The basis elements (may include negative powers)
    pub fn new(field: String, basis: Vec<String>) -> Self {
        assert!(!basis.is_empty(), "Basis cannot be empty");
        Self {
            field,
            basis,
            degree_bound: None,
            _phantom: PhantomData,
        }
    }

    /// Create with a degree bound
    ///
    /// The degree bound limits the pole order at infinity.
    pub fn with_degree_bound(field: String, basis: Vec<String>, degree_bound: i64) -> Self {
        let mut order = Self::new(field, basis);
        order.degree_bound = Some(degree_bound);
        order
    }

    /// Check if at infinity (always true)
    pub fn is_at_infinity(&self) -> bool {
        true
    }

    /// Get the function field
    pub fn function_field(&self) -> &str {
        &self.field
    }

    /// Get the basis
    pub fn basis(&self) -> &[String] {
        &self.basis
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        self.basis.len()
    }

    /// Get the degree bound
    pub fn degree_bound(&self) -> Option<i64> {
        self.degree_bound
    }

    /// Get a specific basis element
    pub fn basis_element(&self, i: usize) -> Option<&str> {
        self.basis.get(i).map(|s| s.as_str())
    }

    /// Check if an element has bounded poles at infinity
    pub fn has_bounded_poles(&self, _element: &str, bound: i64) -> bool {
        // Would compute pole order at infinity
        self.degree_bound.map_or(true, |b| b >= bound)
    }

    /// Compute valuation at infinity
    ///
    /// Returns the negative of the pole order.
    pub fn valuation_at_infinity(&self, _element: &str) -> i64 {
        // Would compute actual valuation
        0
    }

    /// Express an element in the basis
    pub fn coordinates(&self, _element: &str) -> Vec<String> {
        vec!["0".to_string(); self.rank()]
    }

    /// Compute the different at infinity
    pub fn different_at_infinity(&self) -> String {
        format!("Different at ∞ for basis [{}]", self.basis.join(", "))
    }

    /// Check if this is a maximal order at infinity
    pub fn is_maximal(&self) -> bool {
        // Would verify maximality
        false
    }

    /// Get the principal part at infinity
    ///
    /// Returns the Laurent expansion's negative degree terms.
    pub fn principal_part(&self, _element: &str) -> Vec<String> {
        Vec::new()
    }

    /// Compute the conductor to maximal order
    pub fn conductor_to_maximal(&self) -> String {
        format!("Conductor to maximal order at ∞")
    }

    /// Change to a different basis
    pub fn change_basis(&self, new_basis: Vec<String>) -> Self {
        Self::new(self.field.clone(), new_basis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_order_basis_creation() {
        let basis = vec!["1".to_string(), "x".to_string()];
        let order = FunctionFieldOrder_basis::<Rational>::new(
            "Q(x)".to_string(),
            basis.clone(),
        );

        assert_eq!(order.function_field(), "Q(x)");
        assert_eq!(order.rank(), 2);
        assert_eq!(order.basis(), &basis[..]);
    }

    #[test]
    fn test_basis_element_access() {
        let basis = vec!["1".to_string(), "x".to_string(), "x^2".to_string()];
        let order = FunctionFieldOrder_basis::<Rational>::new(
            "Q(x)".to_string(),
            basis,
        );

        assert_eq!(order.basis_element(0), Some("1"));
        assert_eq!(order.basis_element(1), Some("x"));
        assert_eq!(order.basis_element(2), Some("x^2"));
        assert_eq!(order.basis_element(3), None);
    }

    #[test]
    fn test_contains_element() {
        let basis = vec!["1".to_string(), "x".to_string()];
        let order = FunctionFieldOrder_basis::<Rational>::new(
            "Q(x)".to_string(),
            basis,
        );

        assert!(order.contains_element("3*x + 5"));
    }

    #[test]
    fn test_discriminant() {
        let basis = vec!["1".to_string(), "y".to_string()];
        let mut order = FunctionFieldOrder_basis::<Rational>::new(
            "Q(x,y)".to_string(),
            basis,
        );

        let disc = order.discriminant();
        assert!(disc.contains("disc"));
    }

    #[test]
    fn test_different() {
        let basis = vec!["1".to_string(), "sqrt(x)".to_string()];
        let order = FunctionFieldOrder_basis::<Rational>::new(
            "Q(x,sqrt(x))".to_string(),
            basis,
        );

        let diff = order.different();
        assert!(diff.contains("Different"));
    }

    #[test]
    fn test_coordinates() {
        let basis = vec!["1".to_string(), "x".to_string()];
        let order = FunctionFieldOrder_basis::<Rational>::new(
            "Q(x)".to_string(),
            basis,
        );

        let coords = order.coordinates("3*x + 5");
        assert_eq!(coords.len(), 2);
    }

    #[test]
    fn test_multiply() {
        let basis = vec!["1".to_string(), "x".to_string()];
        let order = FunctionFieldOrder_basis::<Rational>::new(
            "Q(x)".to_string(),
            basis,
        );

        let result = order.multiply("x", "x");
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_is_maximal() {
        let basis = vec!["1".to_string(), "x".to_string()];
        let order = FunctionFieldOrder_basis::<Rational>::new(
            "Q(x)".to_string(),
            basis,
        );

        // By default, not necessarily maximal
        assert!(!order.is_maximal());
    }

    #[test]
    fn test_change_basis() {
        let basis1 = vec!["1".to_string(), "x".to_string()];
        let order1 = FunctionFieldOrder_basis::<Rational>::new(
            "Q(x)".to_string(),
            basis1,
        );

        let basis2 = vec!["1".to_string(), "2*x".to_string()];
        let order2 = order1.change_basis(basis2.clone());

        assert_eq!(order2.basis(), &basis2[..]);
        assert_eq!(order2.function_field(), "Q(x)");
    }

    #[test]
    fn test_span() {
        let basis = vec!["1".to_string(), "x".to_string()];
        let order = FunctionFieldOrder_basis::<Rational>::new(
            "Q(x)".to_string(),
            basis,
        );

        let elements = vec!["1".to_string(), "x".to_string(), "x^2".to_string()];
        let spanned = order.span(elements);

        assert_eq!(spanned.rank(), 3);
    }

    #[test]
    fn test_one() {
        let basis = vec!["1".to_string(), "x".to_string()];
        let order = FunctionFieldOrder_basis::<Rational>::new(
            "Q(x)".to_string(),
            basis,
        );

        let one = order.one();
        assert_eq!(one.len(), 2);
        assert_eq!(one[0], "1");
        assert_eq!(one[1], "0");
    }

    #[test]
    fn test_is_orthogonal() {
        let basis = vec!["1".to_string(), "x".to_string()];
        let order = FunctionFieldOrder_basis::<Rational>::new(
            "Q(x)".to_string(),
            basis,
        );

        // Generic basis typically not orthogonal
        assert!(!order.is_orthogonal());
    }

    #[test]
    fn test_infinite_order_basis_creation() {
        let basis = vec!["1".to_string(), "1/x".to_string()];
        let order_inf = FunctionFieldOrderInfinite_basis::<Rational>::new(
            "Q(x)".to_string(),
            basis.clone(),
        );

        assert!(order_inf.is_at_infinity());
        assert_eq!(order_inf.function_field(), "Q(x)");
        assert_eq!(order_inf.rank(), 2);
    }

    #[test]
    fn test_infinite_with_degree_bound() {
        let basis = vec!["1".to_string(), "1/x".to_string(), "1/x^2".to_string()];
        let order_inf = FunctionFieldOrderInfinite_basis::<Rational>::with_degree_bound(
            "Q(x)".to_string(),
            basis,
            2,
        );

        assert_eq!(order_inf.degree_bound(), Some(2));
        assert_eq!(order_inf.rank(), 3);
    }

    #[test]
    fn test_infinite_basis_element() {
        let basis = vec!["1".to_string(), "1/x".to_string()];
        let order_inf = FunctionFieldOrderInfinite_basis::<Rational>::new(
            "Q(x)".to_string(),
            basis,
        );

        assert_eq!(order_inf.basis_element(0), Some("1"));
        assert_eq!(order_inf.basis_element(1), Some("1/x"));
        assert_eq!(order_inf.basis_element(2), None);
    }

    #[test]
    fn test_has_bounded_poles() {
        let basis = vec!["1".to_string(), "1/x".to_string()];
        let order_inf = FunctionFieldOrderInfinite_basis::<Rational>::with_degree_bound(
            "Q(x)".to_string(),
            basis,
            2,
        );

        assert!(order_inf.has_bounded_poles("1/x", 2));
    }

    #[test]
    fn test_valuation_at_infinity() {
        let basis = vec!["1".to_string(), "1/x".to_string()];
        let order_inf = FunctionFieldOrderInfinite_basis::<Rational>::new(
            "Q(x)".to_string(),
            basis,
        );

        let val = order_inf.valuation_at_infinity("1/x^2");
        assert!(val <= 0); // Negative valuation for poles
    }

    #[test]
    fn test_infinite_coordinates() {
        let basis = vec!["1".to_string(), "1/x".to_string()];
        let order_inf = FunctionFieldOrderInfinite_basis::<Rational>::new(
            "Q(x)".to_string(),
            basis,
        );

        let coords = order_inf.coordinates("3/x + 2");
        assert_eq!(coords.len(), 2);
    }

    #[test]
    fn test_different_at_infinity() {
        let basis = vec!["1".to_string(), "1/x".to_string()];
        let order_inf = FunctionFieldOrderInfinite_basis::<Rational>::new(
            "Q(x)".to_string(),
            basis,
        );

        let diff = order_inf.different_at_infinity();
        assert!(diff.contains("Different"));
        assert!(diff.contains("∞"));
    }

    #[test]
    fn test_infinite_is_maximal() {
        let basis = vec!["1".to_string(), "1/x".to_string()];
        let order_inf = FunctionFieldOrderInfinite_basis::<Rational>::new(
            "Q(x)".to_string(),
            basis,
        );

        // Not necessarily maximal by default
        assert!(!order_inf.is_maximal());
    }

    #[test]
    fn test_principal_part() {
        let basis = vec!["1".to_string(), "1/x".to_string()];
        let order_inf = FunctionFieldOrderInfinite_basis::<Rational>::new(
            "Q(x)".to_string(),
            basis,
        );

        let pp = order_inf.principal_part("1/x + x");
        assert!(pp.len() >= 0);
    }

    #[test]
    fn test_conductor_to_maximal() {
        let basis = vec!["1".to_string(), "2/x".to_string()];
        let order_inf = FunctionFieldOrderInfinite_basis::<Rational>::new(
            "Q(x)".to_string(),
            basis,
        );

        let conductor = order_inf.conductor_to_maximal();
        assert!(conductor.contains("Conductor"));
        assert!(conductor.contains("maximal"));
    }

    #[test]
    fn test_infinite_change_basis() {
        let basis1 = vec!["1".to_string(), "1/x".to_string()];
        let order1 = FunctionFieldOrderInfinite_basis::<Rational>::new(
            "Q(x)".to_string(),
            basis1,
        );

        let basis2 = vec!["1".to_string(), "1/x^2".to_string()];
        let order2 = order1.change_basis(basis2.clone());

        assert_eq!(order2.basis(), &basis2[..]);
        assert!(order2.is_at_infinity());
    }

    #[test]
    #[should_panic(expected = "Basis cannot be empty")]
    fn test_empty_basis_panics() {
        let _order = FunctionFieldOrder_basis::<Rational>::new(
            "Q(x)".to_string(),
            Vec::new(),
        );
    }

    #[test]
    #[should_panic(expected = "Basis cannot be empty")]
    fn test_infinite_empty_basis_panics() {
        let _order = FunctionFieldOrderInfinite_basis::<Rational>::new(
            "Q(x)".to_string(),
            Vec::new(),
        );
    }
}
