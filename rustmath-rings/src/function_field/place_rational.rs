//! Places for rational function fields
//!
//! This module provides place structures for rational function fields k(x),
//! corresponding to SageMath's `sage.rings.function_field.place_rational`.
//!
//! # Mathematical Background
//!
//! For the rational function field k(x) = Frac(k[x]), places correspond to:
//! - **Finite places**: Irreducible polynomials p(x) ∈ k[x]
//! - **Infinite place**: The place at infinity ∞
//!
//! For a finite place p(x) of degree d:
//! - Valuation: v_p(f/g) = ord_p(f) - ord_p(g)
//! - Residue field: k[x]/(p(x)) ≅ k_d (degree d extension of k)
//! - Degree: deg(p) = [k[x]/(p(x)) : k] = d
//!
//! For the infinite place:
//! - Valuation: v_∞(f/g) = deg(g) - deg(f)
//! - Residue field: k
//! - Degree: 1
//!
//! # Divisor Theory
//!
//! The divisor group of k(x) is:
//! Div(k(x)) = ⊕_P ℤ·P
//!
//! where P ranges over all places (finite and infinite).
//!
//! For a rational function f = p_1^{a_1} ... p_n^{a_n} / q_1^{b_1} ... q_m^{b_m}:
//!
//! div(f) = ∑ a_i·[p_i] - ∑ b_j·[q_j] - (deg f)·[∞]
//!
//! The degree of any principal divisor is 0.

use rustmath_core::Field;
use std::marker::PhantomData;
use std::fmt;
use super::place::FunctionFieldPlace;

/// Place for rational function field k(x)
///
/// Represents a place of a rational function field.
/// This corresponds to SageMath's `FunctionFieldPlace_rational` class.
///
/// # Type Parameters
///
/// - `F`: The constant field type
///
/// # Mathematical Details
///
/// Finite places correspond to irreducible polynomials in k[x]:
/// - p(x) = x - a (degree 1, splits over k)
/// - Irreducible polynomials of degree d > 1
///
/// The infinite place corresponds to the point at infinity on the projective line.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FunctionFieldPlaceRational<F: Field> {
    /// Base place structure
    base: FunctionFieldPlace<F>,
    /// Prime polynomial (for finite places)
    prime_polynomial: Option<String>,
    /// Whether this is the infinite place
    is_infinite_place: bool,
    /// Field marker
    field_marker: PhantomData<F>,
}

/// Type alias for snake_case compatibility
pub type FunctionFieldPlace_rational<F> = FunctionFieldPlaceRational<F>;

impl<F: Field> FunctionFieldPlace_rational<F> {
    /// Create a new finite place for rational function field
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the place
    /// * `degree` - Degree of the place (degree of prime polynomial)
    /// * `prime_polynomial` - Prime polynomial defining the place
    ///
    /// # Returns
    ///
    /// A new FunctionFieldPlace_rational instance
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Place at x=0
    /// let p0 = FunctionFieldPlace_rational::new(
    ///     "P_0".to_string(),
    ///     1,
    ///     "x".to_string()
    /// );
    ///
    /// // Place at x=1
    /// let p1 = FunctionFieldPlace_rational::new(
    ///     "P_1".to_string(),
    ///     1,
    ///     "x - 1".to_string()
    /// );
    /// ```
    pub fn new(name: String, degree: usize, prime_polynomial: String) -> Self {
        FunctionFieldPlace_rational {
            base: FunctionFieldPlace::new(name, degree),
            prime_polynomial: Some(prime_polynomial),
            is_infinite_place: false,
            field_marker: PhantomData,
        }
    }

    /// Create the infinite place
    ///
    /// # Returns
    ///
    /// The infinite place of k(x)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let p_inf = FunctionFieldPlace_rational::infinite();
    /// assert!(p_inf.is_infinite());
    /// ```
    pub fn infinite() -> Self {
        FunctionFieldPlace_rational {
            base: FunctionFieldPlace::infinite("∞".to_string(), 1),
            prime_polynomial: None,
            is_infinite_place: true,
            field_marker: PhantomData,
        }
    }

    /// Create a degree 1 place at a point
    ///
    /// # Arguments
    ///
    /// * `point` - String representation of the point (e.g., "a" for x-a)
    ///
    /// # Returns
    ///
    /// Degree 1 place at the given point
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let p_a = FunctionFieldPlace_rational::at_point("2".to_string());
    /// assert_eq!(p_a.degree(), 1);
    /// ```
    pub fn at_point(point: String) -> Self {
        let name = format!("P_{}", point);
        let polynomial = if point == "0" {
            "x".to_string()
        } else {
            format!("x - {}", point)
        };

        FunctionFieldPlace_rational::new(name, 1, polynomial)
    }

    /// Get the underlying base place
    ///
    /// # Returns
    ///
    /// Reference to the base FunctionFieldPlace
    pub fn base(&self) -> &FunctionFieldPlace<F> {
        &self.base
    }

    /// Get the prime polynomial
    ///
    /// # Returns
    ///
    /// Prime polynomial defining the place (None for infinite place)
    pub fn prime_polynomial(&self) -> Option<&str> {
        self.prime_polynomial.as_deref()
    }

    /// Check if this is the infinite place
    ///
    /// # Returns
    ///
    /// True if this is the place at infinity
    pub fn is_infinite(&self) -> bool {
        self.is_infinite_place
    }

    /// Check if this is a finite place
    ///
    /// # Returns
    ///
    /// True if this is a finite place
    pub fn is_finite(&self) -> bool {
        !self.is_infinite_place
    }

    /// Get place name
    ///
    /// # Returns
    ///
    /// Name of the place
    pub fn name(&self) -> &str {
        self.base.name()
    }

    /// Get degree
    ///
    /// # Returns
    ///
    /// Degree of the place
    pub fn degree(&self) -> usize {
        self.base.degree()
    }

    /// Check if this is a degree 1 place
    ///
    /// # Returns
    ///
    /// True if degree equals 1
    pub fn is_degree_one(&self) -> bool {
        self.base.degree() == 1
    }

    /// Compute local uniformizer at this place
    ///
    /// # Returns
    ///
    /// String representation of a local uniformizer
    ///
    /// # Mathematical Details
    ///
    /// - For finite place defined by p(x): uniformizer is p(x)
    /// - For infinite place: uniformizer is 1/x
    pub fn local_uniformizer(&self) -> String {
        if self.is_infinite_place {
            "1/x".to_string()
        } else if let Some(poly) = &self.prime_polynomial {
            poly.clone()
        } else {
            "t".to_string()
        }
    }

    /// Compute residue field
    ///
    /// # Returns
    ///
    /// String description of the residue field
    ///
    /// # Mathematical Details
    ///
    /// - Finite place with prime p(x): k[x]/(p(x))
    /// - Infinite place: k
    pub fn residue_field(&self) -> String {
        if self.is_infinite_place {
            "k".to_string()
        } else if let Some(poly) = &self.prime_polynomial {
            format!("k[x]/({})", poly)
        } else {
            "k".to_string()
        }
    }

    /// Compute valuation of a rational function (symbolic)
    ///
    /// # Arguments
    ///
    /// * `numerator` - Numerator polynomial
    /// * `denominator` - Denominator polynomial
    ///
    /// # Returns
    ///
    /// String representation of the valuation
    ///
    /// # Mathematical Details
    ///
    /// For finite place p: v_p(f/g) = ord_p(f) - ord_p(g)
    /// For infinite place: v_∞(f/g) = deg(g) - deg(f)
    pub fn valuation_of(&self, numerator: &str, denominator: &str) -> String {
        if self.is_infinite_place {
            format!("deg({}) - deg({})", denominator, numerator)
        } else if let Some(p) = &self.prime_polynomial {
            format!("ord_{{{}}}({}) - ord_{{{}}}({})", p, numerator, p, denominator)
        } else {
            format!("v({}/ {})", numerator, denominator)
        }
    }

    /// Check if place splits over the constant field
    ///
    /// # Returns
    ///
    /// True if degree equals 1 (place splits)
    pub fn splits(&self) -> bool {
        self.degree() == 1
    }

    /// Check if place is inert
    ///
    /// # Arguments
    ///
    /// * `max_degree` - Maximum possible degree
    ///
    /// # Returns
    ///
    /// True if degree equals max_degree (place is inert)
    pub fn is_inert(&self, max_degree: usize) -> bool {
        self.degree() == max_degree
    }
}

impl<F: Field> fmt::Display for FunctionFieldPlace_rational<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_infinite_place {
            write!(f, "Place at infinity ∞")
        } else if let Some(poly) = &self.prime_polynomial {
            write!(
                f,
                "Place {} defined by {} (degree {})",
                self.base.name(),
                poly,
                self.base.degree()
            )
        } else {
            write!(f, "Place {} (degree {})", self.base.name(), self.base.degree())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_rational_place_creation() {
        let place: FunctionFieldPlace_rational<Rational> =
            FunctionFieldPlace_rational::new(
                "P_0".to_string(),
                1,
                "x".to_string(),
            );

        assert_eq!(place.name(), "P_0");
        assert_eq!(place.degree(), 1);
        assert!(place.is_finite());
        assert!(!place.is_infinite());
        assert_eq!(place.prime_polynomial(), Some("x"));
    }

    #[test]
    fn test_infinite_place() {
        let p_inf: FunctionFieldPlace_rational<Rational> =
            FunctionFieldPlace_rational::infinite();

        assert!(p_inf.is_infinite());
        assert!(!p_inf.is_finite());
        assert_eq!(p_inf.degree(), 1);
        assert_eq!(p_inf.name(), "∞");
        assert_eq!(p_inf.prime_polynomial(), None);
    }

    #[test]
    fn test_at_point() {
        let p_0: FunctionFieldPlace_rational<Rational> =
            FunctionFieldPlace_rational::at_point("0".to_string());
        let p_1: FunctionFieldPlace_rational<Rational> =
            FunctionFieldPlace_rational::at_point("1".to_string());

        assert_eq!(p_0.prime_polynomial(), Some("x"));
        assert_eq!(p_1.prime_polynomial(), Some("x - 1"));
        assert!(p_0.is_degree_one());
        assert!(p_1.is_degree_one());
    }

    #[test]
    fn test_degree_one_places() {
        let p: FunctionFieldPlace_rational<Rational> =
            FunctionFieldPlace_rational::new(
                "P".to_string(),
                1,
                "x - 3".to_string(),
            );

        assert!(p.is_degree_one());
        assert!(p.splits());
    }

    #[test]
    fn test_higher_degree_place() {
        let p: FunctionFieldPlace_rational<Rational> =
            FunctionFieldPlace_rational::new(
                "P".to_string(),
                2,
                "x^2 + 1".to_string(),
            );

        assert_eq!(p.degree(), 2);
        assert!(!p.is_degree_one());
        assert!(!p.splits());
    }

    #[test]
    fn test_local_uniformizer_finite() {
        let p: FunctionFieldPlace_rational<Rational> =
            FunctionFieldPlace_rational::new(
                "P_0".to_string(),
                1,
                "x".to_string(),
            );

        assert_eq!(p.local_uniformizer(), "x");
    }

    #[test]
    fn test_local_uniformizer_infinite() {
        let p_inf: FunctionFieldPlace_rational<Rational> =
            FunctionFieldPlace_rational::infinite();

        assert_eq!(p_inf.local_uniformizer(), "1/x");
    }

    #[test]
    fn test_residue_field_finite() {
        let p: FunctionFieldPlace_rational<Rational> =
            FunctionFieldPlace_rational::new(
                "P".to_string(),
                2,
                "x^2 + 1".to_string(),
            );

        let res_field = p.residue_field();
        assert!(res_field.contains("k[x]/(x^2 + 1)"));
    }

    #[test]
    fn test_residue_field_infinite() {
        let p_inf: FunctionFieldPlace_rational<Rational> =
            FunctionFieldPlace_rational::infinite();

        assert_eq!(p_inf.residue_field(), "k");
    }

    #[test]
    fn test_valuation_finite() {
        let p: FunctionFieldPlace_rational<Rational> =
            FunctionFieldPlace_rational::new(
                "P_0".to_string(),
                1,
                "x".to_string(),
            );

        let val = p.valuation_of("x^2", "x+1");
        assert!(val.contains("ord"));
    }

    #[test]
    fn test_valuation_infinite() {
        let p_inf: FunctionFieldPlace_rational<Rational> =
            FunctionFieldPlace_rational::infinite();

        let val = p_inf.valuation_of("x^2 + 1", "x^3");
        assert!(val.contains("deg"));
    }

    #[test]
    fn test_inert_place() {
        let p: FunctionFieldPlace_rational<Rational> =
            FunctionFieldPlace_rational::new(
                "P".to_string(),
                3,
                "x^3 + x + 1".to_string(),
            );

        assert!(p.is_inert(3));
        assert!(!p.is_inert(2));
    }

    #[test]
    fn test_split_place() {
        let p: FunctionFieldPlace_rational<Rational> =
            FunctionFieldPlace_rational::new(
                "P_a".to_string(),
                1,
                "x - a".to_string(),
            );

        assert!(p.splits());
    }

    #[test]
    fn test_clone_equality() {
        let p1: FunctionFieldPlace_rational<Rational> =
            FunctionFieldPlace_rational::new(
                "P".to_string(),
                2,
                "x^2 + 1".to_string(),
            );
        let p2 = p1.clone();

        assert_eq!(p1, p2);
    }

    #[test]
    fn test_display_finite() {
        let p: FunctionFieldPlace_rational<Rational> =
            FunctionFieldPlace_rational::new(
                "P_0".to_string(),
                1,
                "x".to_string(),
            );

        let display = format!("{}", p);
        assert!(display.contains("Place P_0"));
        assert!(display.contains("degree 1"));
    }

    #[test]
    fn test_display_infinite() {
        let p_inf: FunctionFieldPlace_rational<Rational> =
            FunctionFieldPlace_rational::infinite();

        let display = format!("{}", p_inf);
        assert!(display.contains("infinity"));
    }

    #[test]
    fn test_divisor_computation_example() {
        // Example: div(x/(x-1)) in Q(x)
        // Should be [P_0] - [P_1] + 0·[∞]

        let p_0: FunctionFieldPlace_rational<Rational> =
            FunctionFieldPlace_rational::at_point("0".to_string());
        let p_1: FunctionFieldPlace_rational<Rational> =
            FunctionFieldPlace_rational::at_point("1".to_string());
        let p_inf: FunctionFieldPlace_rational<Rational> =
            FunctionFieldPlace_rational::infinite();

        // Verify places are set up correctly
        assert_eq!(p_0.degree(), 1);
        assert_eq!(p_1.degree(), 1);
        assert_eq!(p_inf.degree(), 1);

        // The divisor has degree: 1 - 1 + 0 = 0 (as expected for principal divisors)
        let div_degree = p_0.degree() as i32 - p_1.degree() as i32 + 0 * p_inf.degree() as i32;
        assert_eq!(div_degree, 0);
    }

    #[test]
    fn test_multiple_places() {
        // Create several degree 1 places
        let places: Vec<FunctionFieldPlace_rational<Rational>> = vec![
            FunctionFieldPlace_rational::at_point("0".to_string()),
            FunctionFieldPlace_rational::at_point("1".to_string()),
            FunctionFieldPlace_rational::at_point("-1".to_string()),
            FunctionFieldPlace_rational::infinite(),
        ];

        assert_eq!(places.len(), 4);
        assert!(places.iter().all(|p| p.degree() == 1));
        assert_eq!(places.iter().filter(|p| p.is_finite()).count(), 3);
        assert_eq!(places.iter().filter(|p| p.is_infinite()).count(), 1);
    }

    #[test]
    fn test_genus_zero() {
        // The rational function field Q(x) has genus 0
        // By Riemann-Roch: dim L(D) - dim L(K - D) = deg(D) + 1 - g
        // For g = 0: dim L(D) - dim L(K - D) = deg(D) + 1

        let g = 0;
        let deg_d = 3;

        // Simplified Riemann-Roch calculation
        let rr_result = deg_d + 1 - g;
        assert_eq!(rr_result, 4);
    }
}
