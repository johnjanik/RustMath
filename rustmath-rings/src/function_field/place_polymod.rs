//! Places for polynomial extensions of function fields
//!
//! This module provides place structures for polynomial extensions of function fields,
//! corresponding to SageMath's `sage.rings.function_field.place_polymod`.
//!
//! # Mathematical Background
//!
//! For a function field extension L/K where L = K[y]/(f(y)), places of L arise from:
//! - Extensions of places of K (following prime ideal factorization)
//! - The ramification structure determined by f(y)
//!
//! Key concepts:
//! - Ramification index e(P|p): How many times P divides p
//! - Relative degree f(P|p): Degree of residue field extension
//! - Fundamental identity: ∑ e(P|p)f(P|p) = [L:K]
//!
//! # Riemann-Hurwitz Formula
//!
//! For a separable extension L/K of genus g_L and g_K:
//! 2g_L - 2 = [L:K](2g_K - 2) + deg(R)
//!
//! where R is the ramification divisor.

use rustmath_core::Field;
use std::marker::PhantomData;
use std::fmt;
use super::place::FunctionFieldPlace;

/// Place for polynomial extension of a function field
///
/// Represents a place of a polynomial extension L/K where L = K[y]/(f(y)).
/// This corresponds to SageMath's `FunctionFieldPlace_polymod` class.
///
/// # Type Parameters
///
/// - `F`: The constant field type
///
/// # Mathematical Details
///
/// For a place P of L lying over a place p of K:
/// - Ramification index: e(P|p)
/// - Relative degree: f(P|p)
/// - Absolute degree: deg(P) = deg(p) * f(P|p)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FunctionFieldPlacePolymod<F: Field> {
    /// Base place structure
    base: FunctionFieldPlace<F>,
    /// Prime ideal factorization data
    prime_id: String,
    /// Ramification index e(P|p)
    ramification_index: usize,
    /// Relative degree f(P|p)
    relative_degree: usize,
    /// Name of the defining polynomial
    defining_polynomial: String,
    /// Field marker
    field_marker: PhantomData<F>,
}

/// Type alias for snake_case compatibility
pub type FunctionFieldPlace_polymod<F> = FunctionFieldPlacePolymod<F>;

impl<F: Field> FunctionFieldPlace_polymod<F> {
    /// Create a new place for polynomial extension
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the place
    /// * `degree` - Absolute degree of the place
    /// * `prime_id` - Prime ideal identifier
    /// * `ramification_index` - Ramification index e(P|p)
    /// * `relative_degree` - Relative degree f(P|p)
    /// * `defining_polynomial` - Defining polynomial of the extension
    ///
    /// # Returns
    ///
    /// A new FunctionFieldPlace_polymod instance
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let place = FunctionFieldPlace_polymod::new(
    ///     "P".to_string(),
    ///     2,
    ///     "p1".to_string(),
    ///     1,
    ///     2,
    ///     "y^2 - x".to_string()
    /// );
    /// assert_eq!(place.ramification_index(), 1);
    /// assert_eq!(place.relative_degree(), 2);
    /// ```
    pub fn new(
        name: String,
        degree: usize,
        prime_id: String,
        ramification_index: usize,
        relative_degree: usize,
        defining_polynomial: String,
    ) -> Self {
        assert!(ramification_index > 0, "Ramification index must be positive");
        assert!(relative_degree > 0, "Relative degree must be positive");

        FunctionFieldPlace_polymod {
            base: FunctionFieldPlace::new(name, degree),
            prime_id,
            ramification_index,
            relative_degree,
            defining_polynomial,
            field_marker: PhantomData,
        }
    }

    /// Get the underlying base place
    ///
    /// # Returns
    ///
    /// Reference to the base FunctionFieldPlace
    pub fn base(&self) -> &FunctionFieldPlace<F> {
        &self.base
    }

    /// Get the prime ideal identifier
    ///
    /// # Returns
    ///
    /// String representation of the prime ideal
    pub fn prime_id(&self) -> &str {
        &self.prime_id
    }

    /// Get the ramification index e(P|p)
    ///
    /// # Returns
    ///
    /// Ramification index
    pub fn ramification_index(&self) -> usize {
        self.ramification_index
    }

    /// Get the relative degree f(P|p)
    ///
    /// # Returns
    ///
    /// Relative degree
    pub fn relative_degree(&self) -> usize {
        self.relative_degree
    }

    /// Get the defining polynomial
    ///
    /// # Returns
    ///
    /// String representation of the defining polynomial
    pub fn defining_polynomial(&self) -> &str {
        &self.defining_polynomial
    }

    /// Check if the place is ramified
    ///
    /// # Returns
    ///
    /// True if ramification index > 1
    pub fn is_ramified(&self) -> bool {
        self.ramification_index > 1
    }

    /// Check if the place is unramified
    ///
    /// # Returns
    ///
    /// True if ramification index = 1
    pub fn is_unramified(&self) -> bool {
        self.ramification_index == 1
    }

    /// Check if the place is totally ramified
    ///
    /// # Arguments
    ///
    /// * `extension_degree` - Degree of the extension [L:K]
    ///
    /// # Returns
    ///
    /// True if e(P|p) = [L:K]
    pub fn is_totally_ramified(&self, extension_degree: usize) -> bool {
        self.ramification_index == extension_degree
    }

    /// Compute ramification contribution to Riemann-Hurwitz
    ///
    /// # Returns
    ///
    /// Contribution (e(P|p) - 1) to ramification divisor
    pub fn ramification_contribution(&self) -> usize {
        if self.ramification_index > 0 {
            self.ramification_index - 1
        } else {
            0
        }
    }

    /// Verify fundamental identity
    ///
    /// # Arguments
    ///
    /// * `extension_degree` - Degree of the extension [L:K]
    ///
    /// # Returns
    ///
    /// True if e(P|p) * f(P|p) divides [L:K]
    pub fn verify_fundamental_identity(&self, extension_degree: usize) -> bool {
        let product = self.ramification_index * self.relative_degree;
        extension_degree % product == 0
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

    /// Compute local uniformizer at this place
    ///
    /// # Returns
    ///
    /// String representation of a local uniformizer
    pub fn local_uniformizer(&self) -> String {
        if self.is_ramified() {
            format!("π_{}_ramified", self.base.name())
        } else {
            format!("π_{}", self.base.name())
        }
    }

    /// Compute residue field degree over constant field
    ///
    /// # Arguments
    ///
    /// * `base_degree` - Degree of the place below in the base field
    ///
    /// # Returns
    ///
    /// Absolute degree of residue field
    pub fn absolute_residue_degree(&self, base_degree: usize) -> usize {
        base_degree * self.relative_degree
    }
}

impl<F: Field> fmt::Display for FunctionFieldPlace_polymod<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Polymod place {} (deg {}, e={}, f={}) over {}",
            self.base.name(),
            self.base.degree(),
            self.ramification_index,
            self.relative_degree,
            self.prime_id
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_place_polymod_creation() {
        let place: FunctionFieldPlace_polymod<Rational> =
            FunctionFieldPlace_polymod::new(
                "P".to_string(),
                2,
                "p1".to_string(),
                1,
                2,
                "y^2 - x".to_string(),
            );

        assert_eq!(place.name(), "P");
        assert_eq!(place.degree(), 2);
        assert_eq!(place.ramification_index(), 1);
        assert_eq!(place.relative_degree(), 2);
        assert_eq!(place.defining_polynomial(), "y^2 - x");
    }

    #[test]
    fn test_unramified_place() {
        let place: FunctionFieldPlace_polymod<Rational> =
            FunctionFieldPlace_polymod::new(
                "P".to_string(),
                3,
                "p1".to_string(),
                1,
                3,
                "y^3 - x".to_string(),
            );

        assert!(place.is_unramified());
        assert!(!place.is_ramified());
        assert_eq!(place.ramification_contribution(), 0);
    }

    #[test]
    fn test_ramified_place() {
        let place: FunctionFieldPlace_polymod<Rational> =
            FunctionFieldPlace_polymod::new(
                "P".to_string(),
                1,
                "p1".to_string(),
                2,
                1,
                "y^2 - x".to_string(),
            );

        assert!(place.is_ramified());
        assert!(!place.is_unramified());
        assert_eq!(place.ramification_contribution(), 1);
    }

    #[test]
    fn test_totally_ramified_place() {
        let place: FunctionFieldPlace_polymod<Rational> =
            FunctionFieldPlace_polymod::new(
                "P".to_string(),
                1,
                "p1".to_string(),
                2,
                1,
                "y^2 - x".to_string(),
            );

        assert!(place.is_totally_ramified(2));
        assert!(!place.is_totally_ramified(3));
    }

    #[test]
    fn test_fundamental_identity() {
        // For extension of degree 6, e=2, f=3 works
        let place: FunctionFieldPlace_polymod<Rational> =
            FunctionFieldPlace_polymod::new(
                "P".to_string(),
                3,
                "p1".to_string(),
                2,
                3,
                "y^6 - x".to_string(),
            );

        assert!(place.verify_fundamental_identity(6));
        assert!(!place.verify_fundamental_identity(5));
    }

    #[test]
    fn test_residue_degree() {
        let place: FunctionFieldPlace_polymod<Rational> =
            FunctionFieldPlace_polymod::new(
                "P".to_string(),
                6,
                "p1".to_string(),
                1,
                3,
                "y^3 - x".to_string(),
            );

        // If base place has degree 2, absolute degree is 2*3 = 6
        assert_eq!(place.absolute_residue_degree(2), 6);
    }

    #[test]
    fn test_local_uniformizer() {
        let unramified: FunctionFieldPlace_polymod<Rational> =
            FunctionFieldPlace_polymod::new(
                "P".to_string(),
                2,
                "p1".to_string(),
                1,
                2,
                "y^2 - x".to_string(),
            );

        let ramified: FunctionFieldPlace_polymod<Rational> =
            FunctionFieldPlace_polymod::new(
                "Q".to_string(),
                1,
                "q1".to_string(),
                2,
                1,
                "y^2 - x".to_string(),
            );

        let unif_u = unramified.local_uniformizer();
        let unif_r = ramified.local_uniformizer();

        assert!(unif_u.contains("π_P"));
        assert!(!unif_u.contains("ramified"));
        assert!(unif_r.contains("ramified"));
    }

    #[test]
    fn test_prime_id() {
        let place: FunctionFieldPlace_polymod<Rational> =
            FunctionFieldPlace_polymod::new(
                "P".to_string(),
                1,
                "prime_x".to_string(),
                1,
                1,
                "y - x".to_string(),
            );

        assert_eq!(place.prime_id(), "prime_x");
    }

    #[test]
    fn test_ramification_scenarios() {
        // Example: y^2 = x ramifies at x=0
        let ramified: FunctionFieldPlace_polymod<Rational> =
            FunctionFieldPlace_polymod::new(
                "P_0".to_string(),
                1,
                "p_0".to_string(),
                2,  // e = 2 (total ramification)
                1,  // f = 1
                "y^2 - x".to_string(),
            );

        // Example: y^2 = x^3 - x^2 ramifies at different places
        let partially_ramified: FunctionFieldPlace_polymod<Rational> =
            FunctionFieldPlace_polymod::new(
                "P_1".to_string(),
                1,
                "p_1".to_string(),
                2,
                1,
                "y^2 - x^3 + x^2".to_string(),
            );

        assert!(ramified.is_totally_ramified(2));
        assert!(partially_ramified.is_ramified());
        assert_eq!(ramified.ramification_contribution(), 1);
    }

    #[test]
    fn test_clone_equality() {
        let place1: FunctionFieldPlace_polymod<Rational> =
            FunctionFieldPlace_polymod::new(
                "P".to_string(),
                2,
                "p".to_string(),
                1,
                2,
                "y^2 - x".to_string(),
            );
        let place2 = place1.clone();

        assert_eq!(place1, place2);
        assert_eq!(place1.degree(), place2.degree());
    }

    #[test]
    fn test_display() {
        let place: FunctionFieldPlace_polymod<Rational> =
            FunctionFieldPlace_polymod::new(
                "P".to_string(),
                2,
                "p".to_string(),
                1,
                2,
                "y^2 - x".to_string(),
            );

        let display = format!("{}", place);
        assert!(display.contains("Polymod place P"));
        assert!(display.contains("deg 2"));
        assert!(display.contains("e=1"));
        assert!(display.contains("f=2"));
    }

    #[test]
    #[should_panic(expected = "Ramification index must be positive")]
    fn test_invalid_ramification_index() {
        let _place: FunctionFieldPlace_polymod<Rational> =
            FunctionFieldPlace_polymod::new(
                "P".to_string(),
                1,
                "p".to_string(),
                0,  // Invalid
                1,
                "y - x".to_string(),
            );
    }

    #[test]
    #[should_panic(expected = "Relative degree must be positive")]
    fn test_invalid_relative_degree() {
        let _place: FunctionFieldPlace_polymod<Rational> =
            FunctionFieldPlace_polymod::new(
                "P".to_string(),
                1,
                "p".to_string(),
                1,
                0,  // Invalid
                "y - x".to_string(),
            );
    }

    #[test]
    fn test_complex_extension() {
        // Example: cubic extension with ramification
        // y^3 - x^2 has ramification at x=0 (e=3, f=1) and infinity
        let cubic_ramified: FunctionFieldPlace_polymod<Rational> =
            FunctionFieldPlace_polymod::new(
                "P_0".to_string(),
                1,
                "p_0".to_string(),
                3,
                1,
                "y^3 - x^2".to_string(),
            );

        assert!(cubic_ramified.is_totally_ramified(3));
        assert_eq!(cubic_ramified.ramification_contribution(), 2);
        assert!(cubic_ramified.verify_fundamental_identity(3));
    }
}
