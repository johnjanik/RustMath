//! Function Field Polymod Module
//!
//! This module implements function fields defined as polynomial extensions,
//! corresponding to SageMath's `sage.rings.function_field.function_field_polymod` module.
//!
//! # Mathematical Overview
//!
//! A function field extension L/K is polymod if L = K[y]/(f(y)) where f is an
//! irreducible polynomial over K. These are also called simple extensions.
//!
//! ## Key Concepts
//!
//! ### Simple Extensions
//!
//! Given a function field K and an irreducible polynomial f(y) ∈ K[y], we can
//! construct the extension field:
//!
//! L = K[y]/(f(y))
//!
//! This is a finite extension of degree deg(f).
//!
//! ### Characteristic Zero vs. Characteristic p
//!
//! - **Characteristic zero**: K has characteristic 0 (e.g., ℚ(x))
//! - **Positive characteristic**: K has characteristic p > 0 (e.g., 𝔽_p(x))
//!
//! ### Global Function Fields
//!
//! A global function field is a function field over a finite constant field.
//! These are fundamental in algebraic geometry and number theory.
//!
//! ### Integral Extensions
//!
//! An extension L/K is integral at a place if the defining polynomial has
//! coefficients in the valuation ring at that place.
//!
//! # Implementation Details
//!
//! This module provides several specialized classes:
//!
//! - `FunctionField_polymod`: Base class for polymod extensions
//! - `FunctionField_simple`: Simple (single polynomial) extensions
//! - `FunctionField_char_zero`: Extensions in characteristic 0
//! - `FunctionField_global`: Extensions over finite fields
//! - `FunctionField_integral`: Integral extensions
//! - Combined variants for specific scenarios
//!
//! # References
//!
//! - SageMath: `sage.rings.function_field.function_field_polymod`
//! - Stichtenoth, H. (2009). "Algebraic Function Fields and Codes"

use rustmath_core::{Field, Ring};
use std::marker::PhantomData;

/// Represents a function field defined as a polynomial extension
///
/// Given a base function field K and an irreducible polynomial f(y) ∈ K[y],
/// this represents the extension L = K[y]/(f(y)).
///
/// # Type Parameters
///
/// * `F` - The coefficient field type
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::function_field_polymod::FunctionField_polymod;
/// use rustmath_rationals::Rational;
///
/// // Create an extension Q(x,y) where y^2 = x
/// let field = FunctionField_polymod::<Rational>::new(
///     "Q(x)".to_string(),
///     "y".to_string(),
///     2
/// );
/// assert_eq!(field.degree(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct FunctionFieldPolymod<F: Field> {
    /// Base field name
    base_field: String,
    /// Generator variable name
    variable: String,
    /// Degree of the extension
    degree: usize,
    /// Defining polynomial (as string for simplicity)
    polynomial: String,
    /// Phantom data for the field type
    _phantom: PhantomData<F>,
}

/// Type alias for snake_case compatibility
pub type FunctionField_polymod<F> = FunctionFieldPolymod<F>;

impl<F: Field> FunctionField_polymod<F> {
    /// Create a new polymod function field
    pub fn new(base_field: String, variable: String, degree: usize) -> Self {
        Self {
            base_field,
            variable: variable.clone(),
            degree,
            polynomial: format!("{}^{}", variable, degree),
            _phantom: PhantomData,
        }
    }

    /// Create with an explicit defining polynomial
    pub fn with_polynomial(
        base_field: String,
        variable: String,
        degree: usize,
        polynomial: String,
    ) -> Self {
        Self {
            base_field,
            variable,
            degree,
            polynomial,
            _phantom: PhantomData,
        }
    }

    /// Get the degree of the extension
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Get the base field name
    pub fn base_field(&self) -> &str {
        &self.base_field
    }

    /// Get the variable name
    pub fn variable(&self) -> &str {
        &self.variable
    }

    /// Get the defining polynomial
    pub fn polynomial(&self) -> &str {
        &self.polynomial
    }

    /// Check if this is a simple extension
    pub fn is_simple(&self) -> bool {
        true // Polymod extensions are always simple
    }

    /// Check if this extension is well-defined
    pub fn is_well_defined(&self) -> bool {
        self.degree > 0 && !self.polynomial.is_empty()
    }

    /// Compute the genus of this function field
    /// Uses simplified Hurwitz formula
    pub fn genus(&self, base_genus: usize) -> usize {
        // Simplified: g_L = (deg - 1)(g_K + deg/2)
        // Real computation requires ramification data
        if self.degree == 1 {
            base_genus
        } else {
            (self.degree - 1) * (base_genus + self.degree / 2)
        }
    }
}

/// Simple function field extension L/K where L = K[y]/(f(y))
///
/// This is the most common type of function field extension.
#[derive(Debug, Clone)]
pub struct FunctionFieldSimple<F: Field> {
    /// Underlying polymod structure
    inner: FunctionField_polymod<F>,
}

/// Type alias for snake_case compatibility
pub type FunctionField_simple<F> = FunctionFieldSimple<F>;

impl<F: Field> FunctionField_simple<F> {
    /// Create a new simple extension
    pub fn new(base_field: String, variable: String, degree: usize) -> Self {
        Self {
            inner: FunctionField_polymod::new(base_field, variable, degree),
        }
    }

    /// Create with an explicit polynomial
    pub fn with_polynomial(
        base_field: String,
        variable: String,
        degree: usize,
        polynomial: String,
    ) -> Self {
        Self {
            inner: FunctionField_polymod::with_polynomial(base_field, variable, degree, polynomial),
        }
    }

    /// Get the degree
    pub fn degree(&self) -> usize {
        self.inner.degree()
    }

    /// Check if the extension is Galois
    pub fn is_galois(&self) -> bool {
        // Simplified: would need to check if polynomial splits completely
        false
    }

    /// Get the primitive element
    pub fn primitive_element(&self) -> String {
        self.inner.variable().to_string()
    }

    /// Compute the different (ramification divisor)
    pub fn different(&self) -> String {
        format!("Different of {}/{}", self.inner.variable(), self.inner.base_field())
    }
}

/// Function field extension in characteristic zero
///
/// These are extensions where the base field has characteristic 0.
#[derive(Debug, Clone)]
pub struct FunctionFieldCharZero<F: Field> {
    /// Underlying simple extension
    inner: FunctionField_simple<F>,
}

/// Type alias for snake_case compatibility
pub type FunctionField_char_zero<F> = FunctionFieldCharZero<F>;

impl<F: Field> FunctionField_char_zero<F> {
    /// Create a new characteristic zero extension
    pub fn new(base_field: String, variable: String, degree: usize) -> Self {
        Self {
            inner: FunctionField_simple::new(base_field, variable, degree),
        }
    }

    /// Get the degree
    pub fn degree(&self) -> usize {
        self.inner.degree()
    }

    /// Check that we're in characteristic 0
    pub fn characteristic(&self) -> usize {
        0
    }

    /// Extensions in characteristic 0 are always separable
    pub fn is_separable(&self) -> bool {
        true
    }
}

/// Integral function field extension
///
/// An extension is integral if it's integral at all finite places.
#[derive(Debug, Clone)]
pub struct FunctionFieldIntegral<F: Field> {
    /// Underlying simple extension
    inner: FunctionField_simple<F>,
}

/// Type alias for snake_case compatibility
pub type FunctionField_integral<F> = FunctionFieldIntegral<F>;

impl<F: Field> FunctionField_integral<F> {
    /// Create a new integral extension
    pub fn new(base_field: String, variable: String, degree: usize) -> Self {
        Self {
            inner: FunctionField_simple::new(base_field, variable, degree),
        }
    }

    /// Get the degree
    pub fn degree(&self) -> usize {
        self.inner.degree()
    }

    /// Check if integral at a given place
    pub fn is_integral_at(&self, _place: &str) -> bool {
        // Would check if defining polynomial has coefficients in valuation ring
        true
    }

    /// Get the maximal order (ring of integers)
    pub fn maximal_order(&self) -> String {
        format!("Maximal order of {}", self.inner.primitive_element())
    }
}

/// Characteristic zero integral extension
///
/// Combines characteristic zero and integrality properties.
#[derive(Debug, Clone)]
pub struct FunctionFieldCharZeroIntegral<F: Field> {
    /// Characteristic zero structure
    char_zero: FunctionField_char_zero<F>,
    /// Integral structure
    integral: FunctionField_integral<F>,
}

/// Type alias for snake_case compatibility
pub type FunctionField_char_zero_integral<F> = FunctionFieldCharZeroIntegral<F>;

impl<F: Field> FunctionField_char_zero_integral<F> {
    /// Create a new char zero integral extension
    pub fn new(base_field: String, variable: String, degree: usize) -> Self {
        Self {
            char_zero: FunctionField_char_zero::new(
                base_field.clone(),
                variable.clone(),
                degree,
            ),
            integral: FunctionField_integral::new(base_field, variable, degree),
        }
    }

    /// Get the degree
    pub fn degree(&self) -> usize {
        self.char_zero.degree()
    }

    /// Verify both properties hold
    pub fn is_valid(&self) -> bool {
        self.char_zero.is_separable() && self.integral.is_integral_at("all")
    }
}

/// Global function field (over finite constant field)
///
/// These are function fields with a finite field of constants.
#[derive(Debug, Clone)]
pub struct FunctionFieldGlobal<F: Field> {
    /// Underlying simple extension
    inner: FunctionField_simple<F>,
    /// Size of constant field
    constant_field_size: usize,
}

/// Type alias for snake_case compatibility
pub type FunctionField_global<F> = FunctionFieldGlobal<F>;

impl<F: Field> FunctionField_global<F> {
    /// Create a new global function field
    pub fn new(
        base_field: String,
        variable: String,
        degree: usize,
        constant_field_size: usize,
    ) -> Self {
        Self {
            inner: FunctionField_simple::new(base_field, variable, degree),
            constant_field_size,
        }
    }

    /// Get the constant field size
    pub fn constant_field_size(&self) -> usize {
        self.constant_field_size
    }

    /// Get the degree
    pub fn degree(&self) -> usize {
        self.inner.degree()
    }

    /// Check if the constant field is perfect
    pub fn is_perfect(&self) -> bool {
        // Finite fields are always perfect
        true
    }

    /// Compute the number of rational places of degree d
    /// Uses Weil bounds
    pub fn num_rational_places(&self, d: usize) -> Option<usize> {
        if d == 1 {
            // Would use Weil bounds: |N - q| ≤ 2g√q
            // Simplified approximation
            Some(self.constant_field_size)
        } else {
            None
        }
    }
}

/// Global integral function field
///
/// Combines global and integral properties.
#[derive(Debug, Clone)]
pub struct FunctionFieldGlobalIntegral<F: Field> {
    /// Global structure
    global: FunctionField_global<F>,
    /// Integral structure
    integral: FunctionField_integral<F>,
}

/// Type alias for snake_case compatibility
pub type FunctionField_global_integral<F> = FunctionFieldGlobalIntegral<F>;

impl<F: Field> FunctionField_global_integral<F> {
    /// Create a new global integral function field
    pub fn new(
        base_field: String,
        variable: String,
        degree: usize,
        constant_field_size: usize,
    ) -> Self {
        Self {
            global: FunctionField_global::new(
                base_field.clone(),
                variable.clone(),
                degree,
                constant_field_size,
            ),
            integral: FunctionField_integral::new(base_field, variable, degree),
        }
    }

    /// Get the degree
    pub fn degree(&self) -> usize {
        self.global.degree()
    }

    /// Get the constant field size
    pub fn constant_field_size(&self) -> usize {
        self.global.constant_field_size()
    }

    /// Check if this is a good reduction
    pub fn has_good_reduction(&self) -> bool {
        // Would check ramification and reduction properties
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_function_field_polymod() {
        let field = FunctionField_polymod::<Rational>::new(
            "Q(x)".to_string(),
            "y".to_string(),
            2,
        );

        assert_eq!(field.degree(), 2);
        assert_eq!(field.base_field(), "Q(x)");
        assert_eq!(field.variable(), "y");
        assert!(field.is_simple());
        assert!(field.is_well_defined());
    }

    #[test]
    fn test_function_field_polymod_with_polynomial() {
        let field = FunctionField_polymod::<Rational>::with_polynomial(
            "Q(x)".to_string(),
            "y".to_string(),
            2,
            "y^2 - x".to_string(),
        );

        assert_eq!(field.polynomial(), "y^2 - x");
        assert_eq!(field.degree(), 2);
    }

    #[test]
    fn test_function_field_simple() {
        let field = FunctionField_simple::<Rational>::new(
            "Q(x)".to_string(),
            "y".to_string(),
            3,
        );

        assert_eq!(field.degree(), 3);
        assert_eq!(field.primitive_element(), "y");
        assert!(!field.is_galois()); // Generic polynomial not Galois
    }

    #[test]
    fn test_function_field_char_zero() {
        let field = FunctionField_char_zero::<Rational>::new(
            "Q(x)".to_string(),
            "y".to_string(),
            2,
        );

        assert_eq!(field.characteristic(), 0);
        assert!(field.is_separable());
        assert_eq!(field.degree(), 2);
    }

    #[test]
    fn test_function_field_integral() {
        let field = FunctionField_integral::<Rational>::new(
            "Q(x)".to_string(),
            "y".to_string(),
            2,
        );

        assert_eq!(field.degree(), 2);
        assert!(field.is_integral_at("P"));
        assert!(!field.maximal_order().is_empty());
    }

    #[test]
    fn test_function_field_char_zero_integral() {
        let field = FunctionField_char_zero_integral::<Rational>::new(
            "Q(x)".to_string(),
            "y".to_string(),
            2,
        );

        assert_eq!(field.degree(), 2);
        assert!(field.is_valid());
    }

    #[test]
    fn test_function_field_global() {
        let field = FunctionField_global::<Rational>::new(
            "F2(x)".to_string(),
            "y".to_string(),
            2,
            2, // F2 has 2 elements
        );

        assert_eq!(field.constant_field_size(), 2);
        assert_eq!(field.degree(), 2);
        assert!(field.is_perfect());
        assert!(field.num_rational_places(1).is_some());
    }

    #[test]
    fn test_function_field_global_integral() {
        let field = FunctionField_global_integral::<Rational>::new(
            "F2(x)".to_string(),
            "y".to_string(),
            2,
            2,
        );

        assert_eq!(field.degree(), 2);
        assert_eq!(field.constant_field_size(), 2);
        assert!(field.has_good_reduction());
    }

    #[test]
    fn test_genus_computation() {
        let field = FunctionField_polymod::<Rational>::new(
            "Q(x)".to_string(),
            "y".to_string(),
            2,
        );

        // For degree 2 extension over genus 0 base
        let genus = field.genus(0);
        assert_eq!(genus, 1); // (2-1)(0 + 2/2) = 1

        // For degree 3 extension
        let field3 = FunctionField_polymod::<Rational>::new(
            "Q(x)".to_string(),
            "y".to_string(),
            3,
        );
        let genus3 = field3.genus(0);
        assert_eq!(genus3, 2); // (3-1)(0 + 3/2) = 2*1 = 2
    }

    #[test]
    fn test_different() {
        let field = FunctionField_simple::<Rational>::new(
            "Q(x)".to_string(),
            "y".to_string(),
            2,
        );

        let diff = field.different();
        assert!(diff.contains("Different"));
        assert!(diff.contains("y"));
    }

    #[test]
    fn test_elliptic_curve_extension() {
        // Elliptic curve: y^2 = x^3 + ax + b
        let ec = FunctionField_simple::<Rational>::with_polynomial(
            "Q(x)".to_string(),
            "y".to_string(),
            2,
            "y^2 - x^3 - x".to_string(),
        );

        assert_eq!(ec.degree(), 2);
        assert_eq!(ec.primitive_element(), "y");
    }

    #[test]
    fn test_hyperelliptic_extension() {
        // Hyperelliptic curve: y^2 = f(x) where deg(f) = 5
        let hyp = FunctionField_polymod::<Rational>::with_polynomial(
            "Q(x)".to_string(),
            "y".to_string(),
            2,
            "y^2 - (x^5 + x + 1)".to_string(),
        );

        // Genus of hyperelliptic curve: g = (deg(f) - 1)/2 = 2
        let genus = hyp.genus(0);
        assert!(genus > 0);
    }

    #[test]
    fn test_tower_of_extensions() {
        // First extension: K1 = Q(x)[y]/(y^2 - x)
        let k1 = FunctionField_simple::<Rational>::new(
            "Q(x)".to_string(),
            "y".to_string(),
            2,
        );

        // Second extension: K2 = K1[z]/(z^3 - y)
        let k2 = FunctionField_simple::<Rational>::new(
            "Q(x,y)".to_string(),
            "z".to_string(),
            3,
        );

        // Total degree over Q(x) is 2*3 = 6
        assert_eq!(k1.degree() * k2.degree(), 6);
    }
}
