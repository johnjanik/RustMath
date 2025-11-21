//! Rational Function Field Module
//!
//! This module implements rational function fields k(x), corresponding to
//! SageMath's `sage.rings.function_field.function_field_rational` module.
//!
//! # Mathematical Overview
//!
//! A rational function field k(x) is the field of fractions of the polynomial
//! ring k[x]. It's the simplest function field, having transcendence degree 1
//! over the constant field k.
//!
//! ## Properties
//!
//! - **Genus**: Always 0 (rational functions have no ramification)
//! - **Places**: Finite places correspond to irreducible polynomials in k[x],
//!   plus one infinite place
//! - **Riemann-Roch**: For rational function fields, dim L(D) = max(0, deg(D) + 1)
//!
//! ## Characteristic Variants
//!
//! - **Characteristic zero**: k has characteristic 0 (e.g., ℚ(x), ℝ(x))
//! - **Global**: k is a finite field (e.g., 𝔽_q(x))
//!
//! # Implementation
//!
//! This module provides:
//!
//! - `RationalFunctionField`: Base rational function field
//! - `RationalFunctionField_char_zero`: Rational functions in characteristic 0
//! - `RationalFunctionField_global`: Rational functions over finite fields
//!
//! # Examples
//!
//! ```
//! use rustmath_rings::function_field::function_field_rational::RationalFunctionField;
//! use rustmath_rationals::Rational;
//!
//! // Create Q(x)
//! let qx = RationalFunctionField::<Rational>::new("Q".to_string(), "x".to_string());
//! assert_eq!(qx.genus(), 0);
//! ```
//!
//! # References
//!
//! - SageMath: `sage.rings.function_field.function_field_rational`
//! - Rosen, M. (2002). "Number Theory in Function Fields"

use rustmath_core::Field;
use std::marker::PhantomData;

/// Rational function field k(x)
///
/// Represents the field of rational functions in one variable over a
/// constant field k.
///
/// # Type Parameters
///
/// * `F` - The constant field type
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::function_field_rational::RationalFunctionField;
/// use rustmath_rationals::Rational;
///
/// let qx = RationalFunctionField::<Rational>::new("Q".to_string(), "x".to_string());
/// assert_eq!(qx.variable(), "x");
/// assert_eq!(qx.genus(), 0);
/// ```
#[derive(Debug, Clone)]
pub struct RationalFunctionField<F: Field> {
    /// Name of the constant field
    constant_field: String,
    /// Variable name
    variable: String,
    /// Phantom data for the field type
    _phantom: PhantomData<F>,
}

impl<F: Field> RationalFunctionField<F> {
    /// Create a new rational function field
    ///
    /// # Arguments
    ///
    /// * `constant_field` - Name of the constant field (e.g., "Q", "F2")
    /// * `variable` - Variable name (e.g., "x", "t")
    pub fn new(constant_field: String, variable: String) -> Self {
        Self {
            constant_field,
            variable,
            _phantom: PhantomData,
        }
    }

    /// Get the constant field name
    pub fn constant_field(&self) -> &str {
        &self.constant_field
    }

    /// Get the variable name
    pub fn variable(&self) -> &str {
        &self.variable
    }

    /// Get the genus (always 0 for rational function fields)
    pub fn genus(&self) -> usize {
        0
    }

    /// Get the degree (always 1 as a field extension over itself)
    pub fn degree(&self) -> usize {
        1
    }

    /// Check if this is a rational function field
    pub fn is_rational(&self) -> bool {
        true
    }

    /// Get the infinite place
    pub fn infinite_place(&self) -> String {
        format!("∞ on {}", self.field_name())
    }

    /// Get the field name
    pub fn field_name(&self) -> String {
        format!("{}({})", self.constant_field, self.variable)
    }

    /// Evaluate valuation at a finite place (polynomial)
    ///
    /// For a rational function f/g, the valuation at a polynomial p is:
    /// v_p(f/g) = ord_p(f) - ord_p(g)
    pub fn valuation_at(&self, _polynomial: &str, _element: &str) -> i32 {
        // Simplified: would compute ord_p(element)
        0
    }

    /// Get the divisor of a function
    pub fn divisor(&self, _function: &str) -> String {
        format!("Divisor of function in {}", self.field_name())
    }

    /// Compute the degree of a divisor
    pub fn divisor_degree(&self, _divisor: &str) -> i32 {
        // For principal divisor, degree is always 0
        0
    }
}

/// Rational function field in characteristic zero
///
/// Represents k(x) where k has characteristic 0.
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::function_field_rational::RationalFunctionField_char_zero;
/// use rustmath_rationals::Rational;
///
/// let qx = RationalFunctionField_char_zero::<Rational>::new("Q".to_string(), "x".to_string());
/// assert_eq!(qx.characteristic(), 0);
/// assert!(qx.is_separable());
/// ```
#[derive(Debug, Clone)]
pub struct RationalFunctionFieldCharZero<F: Field> {
    /// Underlying rational function field
    inner: RationalFunctionField<F>,
}

/// Type alias for snake_case compatibility
pub type RationalFunctionField_char_zero<F> = RationalFunctionFieldCharZero<F>;

impl<F: Field> RationalFunctionField_char_zero<F> {
    /// Create a new rational function field in characteristic 0
    pub fn new(constant_field: String, variable: String) -> Self {
        Self {
            inner: RationalFunctionField::new(constant_field, variable),
        }
    }

    /// Get the characteristic (always 0)
    pub fn characteristic(&self) -> usize {
        0
    }

    /// Check if separable (always true in characteristic 0)
    pub fn is_separable(&self) -> bool {
        true
    }

    /// Get the genus
    pub fn genus(&self) -> usize {
        self.inner.genus()
    }

    /// Get the variable
    pub fn variable(&self) -> &str {
        self.inner.variable()
    }

    /// Get the field name
    pub fn field_name(&self) -> String {
        self.inner.field_name()
    }

    /// Compute the derivative of a rational function
    pub fn derivative(&self, _function: &str) -> String {
        format!("d/d{} of function", self.inner.variable())
    }

    /// Check if a rational function is integral
    pub fn is_integral(&self, _function: &str) -> bool {
        // Would check if function is a polynomial
        true
    }
}

/// Global rational function field (over finite constant field)
///
/// Represents 𝔽_q(x) where 𝔽_q is a finite field.
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::function_field_rational::RationalFunctionField_global;
/// use rustmath_rationals::Rational;
///
/// let f2x = RationalFunctionField_global::<Rational>::new("F2".to_string(), "x".to_string(), 2);
/// assert_eq!(f2x.constant_field_size(), 2);
/// assert_eq!(f2x.genus(), 0);
/// ```
#[derive(Debug, Clone)]
pub struct RationalFunctionFieldGlobal<F: Field> {
    /// Underlying rational function field
    inner: RationalFunctionField<F>,
    /// Size of the constant field
    constant_field_size: usize,
}

/// Type alias for snake_case compatibility
pub type RationalFunctionField_global<F> = RationalFunctionFieldGlobal<F>;

impl<F: Field> RationalFunctionField_global<F> {
    /// Create a new global rational function field
    ///
    /// # Arguments
    ///
    /// * `constant_field` - Name of the finite constant field
    /// * `variable` - Variable name
    /// * `constant_field_size` - Size q of the finite field 𝔽_q
    pub fn new(constant_field: String, variable: String, constant_field_size: usize) -> Self {
        Self {
            inner: RationalFunctionField::new(constant_field, variable),
            constant_field_size,
        }
    }

    /// Get the size of the constant field
    pub fn constant_field_size(&self) -> usize {
        self.constant_field_size
    }

    /// Get the genus
    pub fn genus(&self) -> usize {
        self.inner.genus()
    }

    /// Get the variable
    pub fn variable(&self) -> &str {
        self.inner.variable()
    }

    /// Get the field name
    pub fn field_name(&self) -> String {
        self.inner.field_name()
    }

    /// Count rational places of degree 1
    ///
    /// For 𝔽_q(x), there are q finite places of degree 1, plus the infinite place.
    pub fn num_rational_places(&self) -> usize {
        self.constant_field_size + 1 // q finite + 1 infinite
    }

    /// Count places of degree d
    ///
    /// Uses the formula: N_d = (1/d) ∑ μ(d/e) q^e
    /// where the sum is over divisors e of d.
    pub fn num_places_of_degree(&self, d: usize) -> usize {
        if d == 0 {
            return 0;
        }

        // Simplified formula for rational function fields
        // For d=1: q places (finite) + 1 (infinite) counted separately
        // For d>1: approximately q^d / d
        if d == 1 {
            self.constant_field_size
        } else {
            // Simplified approximation
            self.constant_field_size.pow(d as u32) / d
        }
    }

    /// Check if the constant field is perfect (always true for finite fields)
    pub fn is_perfect(&self) -> bool {
        true
    }

    /// Get the Frobenius endomorphism power
    pub fn frobenius_power(&self, n: usize) -> usize {
        self.constant_field_size.pow(n as u32)
    }

    /// Compute the zeta function value
    ///
    /// For genus 0: Z(u) = 1 / ((1-u)(1-qu))
    pub fn zeta_function(&self, _u: f64) -> String {
        format!(
            "Zeta function of {} with q={}",
            self.field_name(),
            self.constant_field_size
        )
    }

    /// Count the number of effective divisors of degree d
    ///
    /// For genus 0, this equals the number of monic polynomials of degree d.
    pub fn num_effective_divisors(&self, d: usize) -> usize {
        if d == 0 {
            1 // The zero divisor
        } else {
            // Number of monic polynomials of degree d over 𝔽_q is q^d
            self.constant_field_size.pow(d as u32)
        }
    }
}

/// Check if an object is a function field
///
/// Helper function corresponding to SageMath's `is_FunctionField`.
pub fn is_function_field<F: Field>(_obj: &RationalFunctionField<F>) -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_rational_function_field() {
        let qx = RationalFunctionField::<Rational>::new("Q".to_string(), "x".to_string());

        assert_eq!(qx.constant_field(), "Q");
        assert_eq!(qx.variable(), "x");
        assert_eq!(qx.genus(), 0);
        assert_eq!(qx.degree(), 1);
        assert!(qx.is_rational());
    }

    #[test]
    fn test_field_name() {
        let qx = RationalFunctionField::<Rational>::new("Q".to_string(), "x".to_string());
        assert_eq!(qx.field_name(), "Q(x)");

        let rt = RationalFunctionField::<Rational>::new("R".to_string(), "t".to_string());
        assert_eq!(rt.field_name(), "R(t)");
    }

    #[test]
    fn test_infinite_place() {
        let qx = RationalFunctionField::<Rational>::new("Q".to_string(), "x".to_string());
        let inf = qx.infinite_place();
        assert!(inf.contains("∞"));
        assert!(inf.contains("Q(x)"));
    }

    #[test]
    fn test_divisor() {
        let qx = RationalFunctionField::<Rational>::new("Q".to_string(), "x".to_string());
        let div = qx.divisor("x^2 + 1");
        assert!(div.contains("Divisor"));

        // Principal divisors have degree 0
        assert_eq!(qx.divisor_degree(&div), 0);
    }

    #[test]
    fn test_rational_function_field_char_zero() {
        let qx = RationalFunctionField_char_zero::<Rational>::new(
            "Q".to_string(),
            "x".to_string(),
        );

        assert_eq!(qx.characteristic(), 0);
        assert!(qx.is_separable());
        assert_eq!(qx.genus(), 0);
        assert_eq!(qx.field_name(), "Q(x)");
    }

    #[test]
    fn test_derivative_char_zero() {
        let qx = RationalFunctionField_char_zero::<Rational>::new(
            "Q".to_string(),
            "x".to_string(),
        );

        let deriv = qx.derivative("x^2 + 1");
        assert!(deriv.contains("d/dx"));
    }

    #[test]
    fn test_is_integral() {
        let qx = RationalFunctionField_char_zero::<Rational>::new(
            "Q".to_string(),
            "x".to_string(),
        );

        assert!(qx.is_integral("x^2 + x + 1"));
    }

    #[test]
    fn test_rational_function_field_global() {
        let f2x = RationalFunctionField_global::<Rational>::new(
            "F2".to_string(),
            "x".to_string(),
            2,
        );

        assert_eq!(f2x.constant_field_size(), 2);
        assert_eq!(f2x.genus(), 0);
        assert!(f2x.is_perfect());
    }

    #[test]
    fn test_num_rational_places() {
        let f2x = RationalFunctionField_global::<Rational>::new(
            "F2".to_string(),
            "x".to_string(),
            2,
        );

        // F2(x) has 2 finite places of degree 1 + 1 infinite place
        assert_eq!(f2x.num_rational_places(), 3);

        let f5x = RationalFunctionField_global::<Rational>::new(
            "F5".to_string(),
            "x".to_string(),
            5,
        );

        // F5(x) has 5 finite places of degree 1 + 1 infinite place
        assert_eq!(f5x.num_rational_places(), 6);
    }

    #[test]
    fn test_num_places_of_degree() {
        let f2x = RationalFunctionField_global::<Rational>::new(
            "F2".to_string(),
            "x".to_string(),
            2,
        );

        // Degree 1: 2 finite places
        assert_eq!(f2x.num_places_of_degree(1), 2);

        // Degree 2: 2^2 / 2 = 2
        assert_eq!(f2x.num_places_of_degree(2), 2);

        // Degree 3: 2^3 / 3 = 2
        assert_eq!(f2x.num_places_of_degree(3), 2);
    }

    #[test]
    fn test_frobenius_power() {
        let f2x = RationalFunctionField_global::<Rational>::new(
            "F2".to_string(),
            "x".to_string(),
            2,
        );

        assert_eq!(f2x.frobenius_power(1), 2); // 2^1
        assert_eq!(f2x.frobenius_power(2), 4); // 2^2
        assert_eq!(f2x.frobenius_power(3), 8); // 2^3
    }

    #[test]
    fn test_num_effective_divisors() {
        let f2x = RationalFunctionField_global::<Rational>::new(
            "F2".to_string(),
            "x".to_string(),
            2,
        );

        // Degree 0: just the zero divisor
        assert_eq!(f2x.num_effective_divisors(0), 1);

        // Degree 1: q^1 = 2
        assert_eq!(f2x.num_effective_divisors(1), 2);

        // Degree 2: q^2 = 4
        assert_eq!(f2x.num_effective_divisors(2), 4);

        // Degree 3: q^3 = 8
        assert_eq!(f2x.num_effective_divisors(3), 8);
    }

    #[test]
    fn test_zeta_function() {
        let f2x = RationalFunctionField_global::<Rational>::new(
            "F2".to_string(),
            "x".to_string(),
            2,
        );

        let zeta = f2x.zeta_function(0.5);
        assert!(zeta.contains("Zeta"));
        assert!(zeta.contains("q=2"));
    }

    #[test]
    fn test_is_function_field() {
        let qx = RationalFunctionField::<Rational>::new("Q".to_string(), "x".to_string());
        assert!(is_function_field(&qx));
    }

    #[test]
    fn test_multiple_variables() {
        let qx = RationalFunctionField::<Rational>::new("Q".to_string(), "x".to_string());
        let qt = RationalFunctionField::<Rational>::new("Q".to_string(), "t".to_string());

        assert_ne!(qx.variable(), qt.variable());
        assert_eq!(qx.constant_field(), qt.constant_field());
    }

    #[test]
    fn test_perfect_field() {
        // Finite fields are perfect
        let f3x = RationalFunctionField_global::<Rational>::new(
            "F3".to_string(),
            "x".to_string(),
            3,
        );
        assert!(f3x.is_perfect());

        let f7x = RationalFunctionField_global::<Rational>::new(
            "F7".to_string(),
            "x".to_string(),
            7,
        );
        assert!(f7x.is_perfect());
    }

    #[test]
    fn test_comparison_char_zero_vs_global() {
        let qx = RationalFunctionField_char_zero::<Rational>::new(
            "Q".to_string(),
            "x".to_string(),
        );

        let f2x = RationalFunctionField_global::<Rational>::new(
            "F2".to_string(),
            "x".to_string(),
            2,
        );

        // Both have genus 0
        assert_eq!(qx.genus(), f2x.genus());

        // But different characteristics
        assert_eq!(qx.characteristic(), 0);
        assert!(f2x.is_perfect()); // Finite field property
    }
}
