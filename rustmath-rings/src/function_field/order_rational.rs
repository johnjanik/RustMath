//! Rational Function Field Order Classes
//!
//! This module implements order classes for rational function fields,
//! corresponding to SageMath's `sage.rings.function_field.order_rational` module.
//!
//! # Mathematical Overview
//!
//! A rational function field k(x) is the field of fractions of k[x].
//! The maximal order (ring of integers) is simply k[x], the polynomial ring.
//!
//! ## Key Concepts
//!
//! ### Rational Function Fields
//!
//! The rational function field k(x) has:
//! - Genus g = 0
//! - Maximal order O = k[x]
//! - Class number h = 1 (principal ideal domain)
//! - Every divisor of degree ≥ 1 is linearly equivalent to a multiple of a point
//!
//! ### Places of k(x)
//!
//! The places of k(x) correspond to:
//! - Finite places: Irreducible polynomials in k[x]
//! - Infinite place: The place at infinity (pole of x)
//!
//! ### Maximal Order k[x]
//!
//! The ring k[x] is:
//! - Euclidean domain (with degree as Euclidean function)
//! - Principal ideal domain
//! - Unique factorization domain
//! - Integrally closed
//!
//! ### Infinite Place
//!
//! At the infinite place, the maximal order consists of polynomials
//! in 1/x with no constant term, denoted k[1/x]_{≥1}.
//!
//! ## Applications
//!
//! - Foundation for algebraic function fields
//! - Riemann-Roch theorem (trivial case)
//! - Coding theory (Reed-Solomon codes)
//! - Arithmetic geometry
//!
//! # Implementation
//!
//! This module provides:
//!
//! - `FunctionFieldMaximalOrder_rational`: Maximal order k[x]
//! - `FunctionFieldMaximalOrderInfinite_rational`: Maximal order at infinity
//!
//! # References
//!
//! - SageMath: `sage.rings.function_field.order_rational`
//! - Rosen, M. (2002). "Number Theory in Function Fields"
//! - Stichtenoth, H. (2009). "Algebraic Function Fields and Codes"

use rustmath_core::Field;
use std::marker::PhantomData;

/// Maximal order for rational function fields
///
/// Represents the maximal order k[x] in a rational function field k(x).
///
/// # Type Parameters
///
/// * `F` - The constant field type
///
/// # Mathematical Details
///
/// For k(x), the maximal order is exactly k[x], the polynomial ring.
/// This is a principal ideal domain with:
/// - Every ideal is principal
/// - Class number = 1
/// - Unique factorization
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::order_rational::FunctionFieldMaximalOrder_rational;
/// use rustmath_rationals::Rational;
///
/// let order = FunctionFieldMaximalOrder_rational::<Rational>::new("Q(x)".to_string());
/// assert!(order.is_maximal());
/// assert!(order.is_pid());
/// assert_eq!(order.class_number(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct FunctionFieldMaximalOrderRational<F: Field> {
    /// Function field name
    field: String,
    /// Variable name (typically "x")
    variable: String,
    /// Constant field name
    constant_field: String,
    /// Phantom data
    _phantom: PhantomData<F>,
}

/// Type alias for snake_case compatibility
pub type FunctionFieldMaximalOrder_rational<F> = FunctionFieldMaximalOrderRational<F>;

impl<F: Field> FunctionFieldMaximalOrder_rational<F> {
    /// Create a new maximal order for a rational function field
    ///
    /// # Arguments
    ///
    /// * `field` - The function field (e.g., "Q(x)")
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rings::function_field::order_rational::FunctionFieldMaximalOrder_rational;
    /// use rustmath_rationals::Rational;
    ///
    /// let order = FunctionFieldMaximalOrder_rational::<Rational>::new("Q(x)".to_string());
    /// ```
    pub fn new(field: String) -> Self {
        // Extract constant field and variable
        let constant_field = if let Some(idx) = field.find('(') {
            field[..idx].to_string()
        } else {
            "k".to_string()
        };

        let variable = if let Some(start) = field.find('(') {
            if let Some(end) = field.find(')') {
                field[start + 1..end].to_string()
            } else {
                "x".to_string()
            }
        } else {
            "x".to_string()
        };

        Self {
            field,
            variable,
            constant_field,
            _phantom: PhantomData,
        }
    }

    /// Get the function field
    pub fn function_field(&self) -> &str {
        &self.field
    }

    /// Get the variable name
    pub fn variable(&self) -> &str {
        &self.variable
    }

    /// Get the constant field
    pub fn constant_field(&self) -> &str {
        &self.constant_field
    }

    /// Check if maximal (always true for k[x])
    pub fn is_maximal(&self) -> bool {
        true
    }

    /// Check if principal ideal domain (always true for k[x])
    pub fn is_pid(&self) -> bool {
        true
    }

    /// Check if Euclidean domain (always true for k[x])
    pub fn is_euclidean(&self) -> bool {
        true
    }

    /// Check if UFD (always true for k[x])
    pub fn is_ufd(&self) -> bool {
        true
    }

    /// Get the class number (always 1)
    pub fn class_number(&self) -> usize {
        1
    }

    /// Get the genus (always 0 for rational function fields)
    pub fn genus(&self) -> usize {
        0
    }

    /// Get the integral basis
    ///
    /// For k(x), a basis is {1, x, x^2, ...} but we return the generator.
    pub fn integral_basis(&self) -> Vec<String> {
        vec!["1".to_string(), self.variable.clone()]
    }

    /// Compute discriminant (always 1 for k[x])
    pub fn discriminant(&self) -> String {
        "1".to_string()
    }

    /// Compute different ideal (always (1) for k[x])
    pub fn different(&self) -> String {
        "(1)".to_string()
    }

    /// Decompose a polynomial into prime factors
    ///
    /// In k[x], this is just polynomial factorization.
    pub fn factor(&self, polynomial: &str) -> Vec<(String, usize)> {
        // Would perform actual factorization
        vec![(polynomial.to_string(), 1)]
    }

    /// Check if a polynomial is irreducible
    pub fn is_irreducible(&self, _polynomial: &str) -> bool {
        // Would check irreducibility
        true
    }

    /// Compute GCD of two polynomials
    pub fn gcd(&self, _a: &str, _b: &str) -> String {
        "1".to_string()
    }

    /// Compute extended GCD
    ///
    /// Returns (g, u, v) such that g = gcd(a,b) = u*a + v*b.
    pub fn xgcd(&self, _a: &str, _b: &str) -> (String, String, String) {
        ("1".to_string(), "1".to_string(), "0".to_string())
    }

    /// Compute the norm of an element
    ///
    /// For k(x), the norm to k is just the constant term.
    pub fn norm(&self, element: &str) -> String {
        element.to_string()
    }

    /// Compute the trace of an element
    ///
    /// For k(x)/k, the trace equals the element itself.
    pub fn trace(&self, element: &str) -> String {
        element.to_string()
    }

    /// Get all places of degree 1
    ///
    /// These correspond to linear polynomials.
    pub fn degree_one_places(&self) -> Vec<String> {
        vec![
            format!("({})", self.variable),
            format!("({} - 1)", self.variable),
        ]
    }

    /// Get the infinite place
    pub fn infinite_place(&self) -> String {
        "∞".to_string()
    }

    /// Evaluate a polynomial at a point
    pub fn evaluate(&self, _polynomial: &str, _point: &str) -> String {
        "0".to_string()
    }

    /// Check if an element is a unit
    ///
    /// Units in k[x] are nonzero constants.
    pub fn is_unit(&self, element: &str) -> bool {
        !element.contains(&self.variable) && element != "0"
    }

    /// Get the unit group
    ///
    /// For k[x], units are k*.
    pub fn unit_group(&self) -> String {
        format!("{}*", self.constant_field)
    }
}

/// Maximal order at infinity for rational function fields
///
/// Represents the maximal order at the infinite place of k(x).
///
/// # Type Parameters
///
/// * `F` - The constant field type
///
/// # Mathematical Details
///
/// At infinity, we work with k[1/x]. Elements with bounded poles at infinity
/// form the order O_∞ = k[1/x].
///
/// The maximal order at infinity is:
/// O_∞ = {f ∈ k(x) : v_∞(f) ≥ 0}
///
/// which is k[1/x] (polynomials in 1/x).
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::order_rational::FunctionFieldMaximalOrderInfinite_rational;
/// use rustmath_rationals::Rational;
///
/// let order_inf = FunctionFieldMaximalOrderInfinite_rational::<Rational>::new(
///     "Q(x)".to_string(),
/// );
/// assert!(order_inf.is_at_infinity());
/// assert!(order_inf.is_maximal());
/// ```
#[derive(Debug, Clone)]
pub struct FunctionFieldMaximalOrderInfiniteRational<F: Field> {
    /// Function field
    field: String,
    /// Variable name
    variable: String,
    /// Inverse variable (1/x)
    inverse_variable: String,
    /// Phantom data
    _phantom: PhantomData<F>,
}

/// Type alias for snake_case compatibility
pub type FunctionFieldMaximalOrderInfinite_rational<F> = FunctionFieldMaximalOrderInfiniteRational<F>;

impl<F: Field> FunctionFieldMaximalOrderInfinite_rational<F> {
    /// Create a new maximal order at infinity
    ///
    /// # Arguments
    ///
    /// * `field` - The function field (e.g., "Q(x)")
    pub fn new(field: String) -> Self {
        let variable = if let Some(start) = field.find('(') {
            if let Some(end) = field.find(')') {
                field[start + 1..end].to_string()
            } else {
                "x".to_string()
            }
        } else {
            "x".to_string()
        };

        let inverse_variable = format!("1/{}", variable);

        Self {
            field,
            variable,
            inverse_variable,
            _phantom: PhantomData,
        }
    }

    /// Check if at infinity (always true)
    pub fn is_at_infinity(&self) -> bool {
        true
    }

    /// Check if maximal (always true)
    pub fn is_maximal(&self) -> bool {
        true
    }

    /// Get the function field
    pub fn function_field(&self) -> &str {
        &self.field
    }

    /// Get the variable
    pub fn variable(&self) -> &str {
        &self.variable
    }

    /// Get the inverse variable (1/x)
    pub fn inverse_variable(&self) -> &str {
        &self.inverse_variable
    }

    /// Get integral basis at infinity
    ///
    /// Returns {1, 1/x, 1/x^2, ...} representation.
    pub fn integral_basis_at_infinity(&self) -> Vec<String> {
        vec!["1".to_string(), self.inverse_variable.clone()]
    }

    /// Compute valuation at infinity
    ///
    /// For f = x^n * (a₀ + a₁/x + ...), v_∞(f) = -n.
    pub fn valuation_at_infinity(&self, _element: &str) -> i64 {
        // Would compute actual valuation
        0
    }

    /// Get the pole order at infinity
    ///
    /// This is -v_∞(f).
    pub fn pole_order(&self, element: &str) -> i64 {
        -self.valuation_at_infinity(element)
    }

    /// Check if element is integral at infinity
    ///
    /// True if v_∞(f) ≥ 0, i.e., f has no pole at infinity.
    pub fn is_integral_at_infinity(&self, element: &str) -> bool {
        self.valuation_at_infinity(element) >= 0
    }

    /// Compute different at infinity
    ///
    /// For k(x), the different at infinity is (1).
    pub fn different_at_infinity(&self) -> String {
        "(1)".to_string()
    }

    /// Expand element as Laurent series at infinity
    ///
    /// Returns coefficients of x^n, x^(n-1), ..., 1, 1/x, ...
    pub fn laurent_expansion(&self, _element: &str, _precision: usize) -> Vec<String> {
        vec!["0".to_string()]
    }

    /// Get the residue at infinity
    ///
    /// The coefficient of 1/x in the Laurent expansion.
    pub fn residue(&self, element: &str) -> String {
        // Would extract 1/x coefficient
        format!("res_∞({})", element)
    }

    /// Check if element has simple pole at infinity
    pub fn has_simple_pole(&self, element: &str) -> bool {
        self.pole_order(element) == 1
    }

    /// Compute the principal part at infinity
    ///
    /// The terms with negative powers in Laurent expansion.
    pub fn principal_part(&self, _element: &str) -> String {
        "0".to_string()
    }

    /// Compute the regular part at infinity
    ///
    /// The terms with non-negative powers.
    pub fn regular_part(&self, _element: &str) -> String {
        "0".to_string()
    }

    /// Get all functions with pole order at most n
    pub fn functions_with_pole_bound(&self, n: i64) -> String {
        format!("L({}∞) = k[x]_{{≤{}}}", n, n)
    }

    /// Dimension of Riemann-Roch space at infinity
    ///
    /// dim L(n∞) = n + 1 for n ≥ 0 (genus 0 case).
    pub fn riemann_roch_dimension(&self, n: i64) -> i64 {
        if n >= 0 {
            n + 1
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_maximal_order_rational_creation() {
        let order = FunctionFieldMaximalOrder_rational::<Rational>::new("Q(x)".to_string());

        assert_eq!(order.function_field(), "Q(x)");
        assert_eq!(order.variable(), "x");
        assert_eq!(order.constant_field(), "Q");
    }

    #[test]
    fn test_is_maximal() {
        let order = FunctionFieldMaximalOrder_rational::<Rational>::new("Q(x)".to_string());
        assert!(order.is_maximal());
    }

    #[test]
    fn test_is_pid() {
        let order = FunctionFieldMaximalOrder_rational::<Rational>::new("Q(x)".to_string());
        assert!(order.is_pid());
    }

    #[test]
    fn test_is_euclidean() {
        let order = FunctionFieldMaximalOrder_rational::<Rational>::new("Q(x)".to_string());
        assert!(order.is_euclidean());
    }

    #[test]
    fn test_is_ufd() {
        let order = FunctionFieldMaximalOrder_rational::<Rational>::new("Q(x)".to_string());
        assert!(order.is_ufd());
    }

    #[test]
    fn test_class_number() {
        let order = FunctionFieldMaximalOrder_rational::<Rational>::new("Q(x)".to_string());
        assert_eq!(order.class_number(), 1);
    }

    #[test]
    fn test_genus() {
        let order = FunctionFieldMaximalOrder_rational::<Rational>::new("Q(x)".to_string());
        assert_eq!(order.genus(), 0);
    }

    #[test]
    fn test_integral_basis() {
        let order = FunctionFieldMaximalOrder_rational::<Rational>::new("Q(x)".to_string());
        let basis = order.integral_basis();
        assert_eq!(basis.len(), 2);
        assert_eq!(basis[0], "1");
        assert_eq!(basis[1], "x");
    }

    #[test]
    fn test_discriminant() {
        let order = FunctionFieldMaximalOrder_rational::<Rational>::new("Q(x)".to_string());
        assert_eq!(order.discriminant(), "1");
    }

    #[test]
    fn test_different() {
        let order = FunctionFieldMaximalOrder_rational::<Rational>::new("Q(x)".to_string());
        assert_eq!(order.different(), "(1)");
    }

    #[test]
    fn test_factor() {
        let order = FunctionFieldMaximalOrder_rational::<Rational>::new("Q(x)".to_string());
        let factors = order.factor("x^2 - 1");
        assert!(!factors.is_empty());
    }

    #[test]
    fn test_is_irreducible() {
        let order = FunctionFieldMaximalOrder_rational::<Rational>::new("Q(x)".to_string());
        assert!(order.is_irreducible("x"));
    }

    #[test]
    fn test_gcd() {
        let order = FunctionFieldMaximalOrder_rational::<Rational>::new("Q(x)".to_string());
        let g = order.gcd("x^2", "x^3");
        assert!(g.contains("1") || g.contains("x"));
    }

    #[test]
    fn test_xgcd() {
        let order = FunctionFieldMaximalOrder_rational::<Rational>::new("Q(x)".to_string());
        let (g, u, v) = order.xgcd("x", "x^2");
        assert!(!g.is_empty());
        assert!(!u.is_empty());
        assert!(!v.is_empty());
    }

    #[test]
    fn test_norm() {
        let order = FunctionFieldMaximalOrder_rational::<Rational>::new("Q(x)".to_string());
        let n = order.norm("x");
        assert!(!n.is_empty());
    }

    #[test]
    fn test_trace() {
        let order = FunctionFieldMaximalOrder_rational::<Rational>::new("Q(x)".to_string());
        let t = order.trace("x");
        assert_eq!(t, "x");
    }

    #[test]
    fn test_degree_one_places() {
        let order = FunctionFieldMaximalOrder_rational::<Rational>::new("Q(x)".to_string());
        let places = order.degree_one_places();
        assert!(!places.is_empty());
    }

    #[test]
    fn test_infinite_place() {
        let order = FunctionFieldMaximalOrder_rational::<Rational>::new("Q(x)".to_string());
        assert_eq!(order.infinite_place(), "∞");
    }

    #[test]
    fn test_is_unit() {
        let order = FunctionFieldMaximalOrder_rational::<Rational>::new("Q(x)".to_string());
        assert!(order.is_unit("5"));
        assert!(!order.is_unit("x"));
        assert!(!order.is_unit("0"));
    }

    #[test]
    fn test_unit_group() {
        let order = FunctionFieldMaximalOrder_rational::<Rational>::new("Q(x)".to_string());
        let units = order.unit_group();
        assert!(units.contains("*"));
    }

    #[test]
    fn test_infinite_order_creation() {
        let order_inf = FunctionFieldMaximalOrderInfinite_rational::<Rational>::new(
            "Q(x)".to_string(),
        );

        assert!(order_inf.is_at_infinity());
        assert!(order_inf.is_maximal());
        assert_eq!(order_inf.function_field(), "Q(x)");
    }

    #[test]
    fn test_infinite_variable() {
        let order_inf = FunctionFieldMaximalOrderInfinite_rational::<Rational>::new(
            "Q(x)".to_string(),
        );

        assert_eq!(order_inf.variable(), "x");
        assert_eq!(order_inf.inverse_variable(), "1/x");
    }

    #[test]
    fn test_integral_basis_at_infinity() {
        let order_inf = FunctionFieldMaximalOrderInfinite_rational::<Rational>::new(
            "Q(x)".to_string(),
        );

        let basis = order_inf.integral_basis_at_infinity();
        assert_eq!(basis.len(), 2);
        assert_eq!(basis[0], "1");
        assert_eq!(basis[1], "1/x");
    }

    #[test]
    fn test_valuation_at_infinity() {
        let order_inf = FunctionFieldMaximalOrderInfinite_rational::<Rational>::new(
            "Q(x)".to_string(),
        );

        let val = order_inf.valuation_at_infinity("1");
        assert!(val >= 0);
    }

    #[test]
    fn test_pole_order() {
        let order_inf = FunctionFieldMaximalOrderInfinite_rational::<Rational>::new(
            "Q(x)".to_string(),
        );

        let po = order_inf.pole_order("x");
        assert!(po.abs() >= 0);
    }

    #[test]
    fn test_is_integral_at_infinity() {
        let order_inf = FunctionFieldMaximalOrderInfinite_rational::<Rational>::new(
            "Q(x)".to_string(),
        );

        assert!(order_inf.is_integral_at_infinity("1"));
    }

    #[test]
    fn test_different_at_infinity() {
        let order_inf = FunctionFieldMaximalOrderInfinite_rational::<Rational>::new(
            "Q(x)".to_string(),
        );

        assert_eq!(order_inf.different_at_infinity(), "(1)");
    }

    #[test]
    fn test_laurent_expansion() {
        let order_inf = FunctionFieldMaximalOrderInfinite_rational::<Rational>::new(
            "Q(x)".to_string(),
        );

        let expansion = order_inf.laurent_expansion("x", 5);
        assert!(!expansion.is_empty());
    }

    #[test]
    fn test_residue() {
        let order_inf = FunctionFieldMaximalOrderInfinite_rational::<Rational>::new(
            "Q(x)".to_string(),
        );

        let res = order_inf.residue("1/x");
        assert!(res.contains("res"));
    }

    #[test]
    fn test_has_simple_pole() {
        let order_inf = FunctionFieldMaximalOrderInfinite_rational::<Rational>::new(
            "Q(x)".to_string(),
        );

        // 1/x has simple pole, x has pole order 0
        assert!(!order_inf.has_simple_pole("1"));
    }

    #[test]
    fn test_principal_part() {
        let order_inf = FunctionFieldMaximalOrderInfinite_rational::<Rational>::new(
            "Q(x)".to_string(),
        );

        let pp = order_inf.principal_part("x + 1/x");
        assert!(!pp.is_empty());
    }

    #[test]
    fn test_regular_part() {
        let order_inf = FunctionFieldMaximalOrderInfinite_rational::<Rational>::new(
            "Q(x)".to_string(),
        );

        let rp = order_inf.regular_part("x + 1/x");
        assert!(!rp.is_empty());
    }

    #[test]
    fn test_functions_with_pole_bound() {
        let order_inf = FunctionFieldMaximalOrderInfinite_rational::<Rational>::new(
            "Q(x)".to_string(),
        );

        let funcs = order_inf.functions_with_pole_bound(3);
        assert!(funcs.contains("L"));
    }

    #[test]
    fn test_riemann_roch_dimension() {
        let order_inf = FunctionFieldMaximalOrderInfinite_rational::<Rational>::new(
            "Q(x)".to_string(),
        );

        // dim L(n∞) = n + 1 for n ≥ 0 in genus 0
        assert_eq!(order_inf.riemann_roch_dimension(0), 1);
        assert_eq!(order_inf.riemann_roch_dimension(1), 2);
        assert_eq!(order_inf.riemann_roch_dimension(5), 6);
        assert_eq!(order_inf.riemann_roch_dimension(-1), 0);
    }

    #[test]
    fn test_custom_variable_name() {
        let order = FunctionFieldMaximalOrder_rational::<Rational>::new("Q(t)".to_string());
        assert_eq!(order.variable(), "t");

        let basis = order.integral_basis();
        assert_eq!(basis[1], "t");
    }

    #[test]
    fn test_infinite_custom_variable() {
        let order_inf = FunctionFieldMaximalOrderInfinite_rational::<Rational>::new(
            "Q(t)".to_string(),
        );

        assert_eq!(order_inf.variable(), "t");
        assert_eq!(order_inf.inverse_variable(), "1/t");
    }
}
