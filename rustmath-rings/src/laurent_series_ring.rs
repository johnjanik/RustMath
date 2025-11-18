//! # Laurent Series Ring
//!
//! This module provides functionality for univariate Laurent series rings, which are
//! fundamental algebraic structures allowing formal power series with both positive
//! and negative exponents of the variable.
//!
//! ## Overview
//!
//! A Laurent series is a formal power series that allows negative exponents:
//!
//! f(x) = ... + a₋₂x⁻² + a₋₁x⁻¹ + a₀ + a₁x + a₂x² + ...
//!
//! The Laurent series ring over a base ring R is denoted R((x)) and consists of all
//! such series with coefficients in R.
//!
//! ## Properties
//!
//! - When R is a field, R((x)) forms a complete discrete valuation field (CDVF)
//! - Laurent series support arithmetic operations (addition, multiplication, division)
//! - Can be constructed from polynomials, power series, and rational functions
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::laurent_series_ring::LaurentSeriesRing;
//! use rustmath_integers::Integer;
//!
//! // Create a Laurent series ring over integers
//! let ring = LaurentSeriesRing::<Integer>::new("x".to_string());
//! assert_eq!(ring.variable_name(), "x");
//! ```
//!
//! ## References
//!
//! - Serre, J.-P. "Local Fields" (1979)
//! - Matsumura, H. "Commutative Ring Theory" (1986)

use rustmath_core::Ring;
use std::fmt;
use std::marker::PhantomData;

/// Represents a univariate Laurent series ring over a base ring
///
/// A Laurent series ring R((x)) consists of formal Laurent series with coefficients in R.
/// Each series can be written as x^n · u(x) where n is an integer (the valuation) and
/// u(x) is a power series with nonzero constant term.
#[derive(Clone, Debug)]
pub struct LaurentSeriesRing<R: Ring> {
    /// Name of the variable
    variable: String,
    /// Phantom data for the coefficient ring
    _phantom: PhantomData<R>,
}

impl<R: Ring> LaurentSeriesRing<R> {
    /// Creates a new Laurent series ring with the specified variable name
    ///
    /// # Arguments
    ///
    /// * `variable` - Name of the variable (e.g., "x", "t")
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rings::laurent_series_ring::LaurentSeriesRing;
    /// use rustmath_integers::Integer;
    ///
    /// let ring = LaurentSeriesRing::<Integer>::new("t".to_string());
    /// assert_eq!(ring.variable_name(), "t");
    /// ```
    pub fn new(variable: String) -> Self {
        LaurentSeriesRing {
            variable,
            _phantom: PhantomData,
        }
    }

    /// Returns the variable name of this Laurent series ring
    pub fn variable_name(&self) -> &str {
        &self.variable
    }

    /// Returns a string representation of the ring
    ///
    /// Format: "Univariate Laurent Series Ring in {variable} over {base_ring}"
    pub fn description(&self) -> String {
        format!("Univariate Laurent Series Ring in {} over base ring", self.variable)
    }

    /// Checks if this ring is a field
    ///
    /// The Laurent series ring is a field if and only if the base ring is a field.
    pub fn is_field(&self) -> bool {
        // This would require trait bounds on R to check if it's a field
        // For now, we return false as a conservative answer
        false
    }

    /// Creates the zero series in this ring
    pub fn zero(&self) -> LaurentSeriesElement<R>
    where
        R: From<i32>,
    {
        LaurentSeriesElement::zero()
    }

    /// Creates the one series in this ring
    pub fn one(&self) -> LaurentSeriesElement<R>
    where
        R: From<i32>,
    {
        LaurentSeriesElement::one()
    }
}

impl<R: Ring> fmt::Display for LaurentSeriesRing<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Laurent Series Ring in {}", self.variable)
    }
}

/// Represents an element of a Laurent series ring
///
/// A Laurent series element is represented as x^valuation · power_series,
/// where the power series has a nonzero constant term (or is zero).
#[derive(Clone, Debug)]
pub struct LaurentSeriesElement<R: Ring> {
    /// The valuation (lowest degree term)
    valuation: i64,
    /// Coefficients of the power series part (starting from the constant term)
    coefficients: Vec<R>,
}

impl<R: Ring> LaurentSeriesElement<R> {
    /// Creates a new Laurent series element
    ///
    /// # Arguments
    ///
    /// * `valuation` - The valuation (lowest degree of x)
    /// * `coefficients` - Coefficients of the power series part
    pub fn new(valuation: i64, coefficients: Vec<R>) -> Self {
        LaurentSeriesElement {
            valuation,
            coefficients,
        }
    }

    /// Creates the zero Laurent series
    pub fn zero() -> Self
    where
        R: From<i32>,
    {
        LaurentSeriesElement {
            valuation: 0,
            coefficients: vec![R::from(0)],
        }
    }

    /// Creates the one Laurent series
    pub fn one() -> Self
    where
        R: From<i32>,
    {
        LaurentSeriesElement {
            valuation: 0,
            coefficients: vec![R::from(1)],
        }
    }

    /// Returns the valuation of this Laurent series
    ///
    /// The valuation is the degree of the lowest term with nonzero coefficient.
    pub fn valuation(&self) -> i64 {
        self.valuation
    }

    /// Returns the coefficients of the power series part
    pub fn coefficients(&self) -> &[R] {
        &self.coefficients
    }

    /// Checks if this series is zero
    pub fn is_zero(&self) -> bool
    where
        R: PartialEq + From<i32>,
    {
        self.coefficients.iter().all(|c| *c == R::from(0))
    }

    /// Checks if this series is a unit (invertible)
    ///
    /// A Laurent series is a unit if and only if its constant term (at valuation) is nonzero.
    pub fn is_unit(&self) -> bool
    where
        R: PartialEq + From<i32>,
    {
        !self.coefficients.is_empty() && self.coefficients[0] != R::from(0)
    }
}

impl<R: Ring> fmt::Display for LaurentSeriesElement<R>
where
    R: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.coefficients.is_empty() {
            return write!(f, "0");
        }

        let mut terms = Vec::new();
        for (i, coeff) in self.coefficients.iter().enumerate() {
            let deg = self.valuation + i as i64;
            if deg == 0 {
                terms.push(format!("{}", coeff));
            } else if deg == 1 {
                terms.push(format!("{}*x", coeff));
            } else if deg > 0 {
                terms.push(format!("{}*x^{}", coeff, deg));
            } else if deg == -1 {
                terms.push(format!("{}*x^(-1)", coeff));
            } else {
                terms.push(format!("{}*x^({})", coeff, deg));
            }
        }

        write!(f, "{}", terms.join(" + "))
    }
}

/// Checks if an object is a Laurent series ring
///
/// # Deprecated
///
/// This function is deprecated. Use `matches!(obj, LaurentSeriesRing::<_> { .. })`
/// or type checking instead.
#[deprecated(since = "0.1.0", note = "Use type checking instead")]
pub fn is_laurent_series_ring<R: Ring>(_obj: &LaurentSeriesRing<R>) -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_laurent_series_ring_creation() {
        let ring = LaurentSeriesRing::<Integer>::new("x".to_string());
        assert_eq!(ring.variable_name(), "x");
    }

    #[test]
    fn test_laurent_series_ring_display() {
        let ring = LaurentSeriesRing::<Integer>::new("t".to_string());
        let display = format!("{}", ring);
        assert!(display.contains("Laurent Series Ring"));
        assert!(display.contains("t"));
    }

    #[test]
    fn test_laurent_series_element_creation() {
        let series = LaurentSeriesElement::new(
            -2,
            vec![Integer::from(1), Integer::from(2), Integer::from(3)],
        );
        assert_eq!(series.valuation(), -2);
        assert_eq!(series.coefficients().len(), 3);
    }

    #[test]
    fn test_laurent_series_zero() {
        let zero = LaurentSeriesElement::<Integer>::zero();
        assert!(zero.is_zero());
        assert_eq!(zero.valuation(), 0);
    }

    #[test]
    fn test_laurent_series_one() {
        let one = LaurentSeriesElement::<Integer>::one();
        assert!(!one.is_zero());
        assert!(one.is_unit());
        assert_eq!(one.valuation(), 0);
    }

    #[test]
    fn test_laurent_series_is_unit() {
        let unit = LaurentSeriesElement::new(
            -1,
            vec![Integer::from(1), Integer::from(2)],
        );
        assert!(unit.is_unit());

        let non_unit = LaurentSeriesElement::new(
            0,
            vec![Integer::from(0), Integer::from(1)],
        );
        assert!(!non_unit.is_unit());
    }

    #[test]
    fn test_laurent_series_display() {
        let series = LaurentSeriesElement::new(
            -1,
            vec![Integer::from(1), Integer::from(2)],
        );
        let display = format!("{}", series);
        assert!(display.contains("x"));
    }

    #[test]
    fn test_ring_zero_one() {
        let ring = LaurentSeriesRing::<Integer>::new("x".to_string());
        let zero = ring.zero();
        let one = ring.one();

        assert!(zero.is_zero());
        assert!(one.is_unit());
    }

    #[test]
    #[allow(deprecated)]
    fn test_is_laurent_series_ring() {
        let ring = LaurentSeriesRing::<Integer>::new("x".to_string());
        assert!(is_laurent_series_ring(&ring));
    }
}
