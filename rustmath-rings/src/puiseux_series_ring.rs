//! # Puiseux Series Ring
//!
//! This module implements rings of Puiseux series over fields.
//!
//! ## Overview
//!
//! A Puiseux series is a generalization of Laurent series allowing fractional exponents.
//! They have the form:
//! ```text
//! f(x) = Σ aₙ x^(n/e)
//! ```
//! where e is a positive integer called the ramification index.
//!
//! ## Theory
//!
//! Puiseux series are important in:
//! - Algebraic geometry (local analysis of algebraic curves)
//! - Resolution of singularities
//! - Newton-Puiseux algorithm for solving polynomial equations
//! - Complex analysis (convergent Puiseux series)
//!
//! ## Implementation
//!
//! The ring of Puiseux series over a field K, denoted K⟨⟨x⟩⟩, is:
//! - The union of all Laurent series rings K((x^(1/n))) for n ≥ 1
//! - Algebraically closed if K is algebraically closed
//! - Contains Laurent series as a subring
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::puiseux_series_ring::PuiseuxSeriesRing;
//! use rustmath_rationals::Rational;
//!
//! // Create a Puiseux series ring over the rationals
//! let ring = PuiseuxSeriesRing::<Rational>::new(10);
//! ```

use crate::puiseux_series_ring_element::PuiseuxSeries;
use rustmath_core::{Field, Ring};
use std::fmt;
use std::marker::PhantomData;

/// Ring of Puiseux series over a field
///
/// Represents the ring K⟨⟨x⟩⟩ of Puiseux series over a field K.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PuiseuxSeriesRing<F: Field> {
    /// Base field
    _field: PhantomData<F>,
    /// Default precision for series in this ring
    default_precision: usize,
    /// Variable name (typically 'x')
    variable_name: String,
}

impl<F: Field> PuiseuxSeriesRing<F> {
    /// Create a new Puiseux series ring with default precision
    ///
    /// # Arguments
    /// * `default_precision` - Default precision for series
    ///
    /// # Examples
    /// ```
    /// use rustmath_rings::puiseux_series_ring::PuiseuxSeriesRing;
    /// use rustmath_rationals::Rational;
    ///
    /// let ring = PuiseuxSeriesRing::<Rational>::new(10);
    /// ```
    pub fn new(default_precision: usize) -> Self {
        Self {
            _field: PhantomData,
            default_precision,
            variable_name: String::from("x"),
        }
    }

    /// Create a new Puiseux series ring with custom variable name
    pub fn new_with_variable(default_precision: usize, variable_name: String) -> Self {
        Self {
            _field: PhantomData,
            default_precision,
            variable_name,
        }
    }

    /// Get the default precision
    pub fn default_precision(&self) -> usize {
        self.default_precision
    }

    /// Get the variable name
    pub fn variable_name(&self) -> &str {
        &self.variable_name
    }

    /// Get the base field (type-level only)
    pub fn base_field(&self) -> PhantomData<F> {
        PhantomData
    }

    /// Create a Puiseux series from Laurent series data
    ///
    /// # Arguments
    /// * `coefficients` - Coefficients of the Laurent series
    /// * `valuation` - Valuation (minimum exponent numerator)
    /// * `ramification` - Ramification index (exponent denominator)
    pub fn create_series(
        &self,
        coefficients: Vec<F>,
        valuation: isize,
        ramification: usize,
    ) -> PuiseuxSeries<F> {
        PuiseuxSeries::new(coefficients, valuation, ramification, self.default_precision)
    }

    /// Create a Puiseux series from a single element (constant)
    pub fn from_constant(&self, value: F) -> PuiseuxSeries<F> {
        PuiseuxSeries::new(vec![value], 0, 1, self.default_precision)
    }

    /// Create the series x^(n/e)
    pub fn monomial(&self, numerator: isize, denominator: usize) -> PuiseuxSeries<F> {
        if denominator == 0 {
            panic!("Denominator cannot be zero");
        }
        let mut coeffs = vec![F::zero(); 1];
        coeffs[0] = F::one();
        PuiseuxSeries::new(coeffs, numerator, denominator, self.default_precision)
    }

    /// Get the Laurent series ring of ramification e
    /// Returns a conceptual reference - in practice, Puiseux series can handle any ramification
    pub fn laurent_series_ring(&self, ramification: usize) -> LaurentSeriesRingView<F> {
        LaurentSeriesRingView {
            parent: self,
            ramification,
        }
    }

    /// Check if this ring contains algebraic closures
    /// Puiseux series over an algebraically closed field are algebraically closed
    pub fn is_algebraically_closed(&self) -> bool {
        // This would depend on the base field F being algebraically closed
        // For now, return false as a conservative estimate
        false
    }
}

/// View of a Laurent series ring within the Puiseux series ring
///
/// Represents the subring of Puiseux series with a fixed ramification index.
#[derive(Debug, Clone)]
pub struct LaurentSeriesRingView<'a, F: Field> {
    parent: &'a PuiseuxSeriesRing<F>,
    ramification: usize,
}

impl<'a, F: Field> LaurentSeriesRingView<'a, F> {
    /// Get the ramification index
    pub fn ramification(&self) -> usize {
        self.ramification
    }

    /// Get the parent Puiseux series ring
    pub fn parent(&self) -> &'a PuiseuxSeriesRing<F> {
        self.parent
    }

    /// Create a series in this Laurent series ring
    pub fn create_series(&self, coefficients: Vec<F>, valuation: isize) -> PuiseuxSeries<F> {
        PuiseuxSeries::new(
            coefficients,
            valuation,
            self.ramification,
            self.parent.default_precision,
        )
    }
}

impl<F: Field + fmt::Display> fmt::Display for PuiseuxSeriesRing<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Puiseux Series Ring in {} over {}",
            self.variable_name,
            std::any::type_name::<F>()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_new_puiseux_series_ring() {
        let ring = PuiseuxSeriesRing::<Rational>::new(10);
        assert_eq!(ring.default_precision(), 10);
        assert_eq!(ring.variable_name(), "x");
    }

    #[test]
    fn test_with_custom_variable() {
        let ring = PuiseuxSeriesRing::<Rational>::new_with_variable(10, String::from("t"));
        assert_eq!(ring.variable_name(), "t");
    }

    #[test]
    fn test_from_constant() {
        let ring = PuiseuxSeriesRing::<Rational>::new(10);
        let series = ring.from_constant(Rational::new(3, 2));

        assert_eq!(series.ramification(), 1);
        assert_eq!(series.valuation(), 0);
    }

    #[test]
    fn test_monomial() {
        let ring = PuiseuxSeriesRing::<Rational>::new(10);
        // Create x^(3/2)
        let series = ring.monomial(3, 2);

        assert_eq!(series.valuation(), 3);
        assert_eq!(series.ramification(), 2);
    }

    #[test]
    #[should_panic(expected = "Denominator cannot be zero")]
    fn test_monomial_zero_denominator() {
        let ring = PuiseuxSeriesRing::<Rational>::new(10);
        let _ = ring.monomial(1, 0);
    }

    #[test]
    fn test_laurent_series_ring_view() {
        let ring = PuiseuxSeriesRing::<Rational>::new(10);
        let laurent = ring.laurent_series_ring(3);

        assert_eq!(laurent.ramification(), 3);

        let series = laurent.create_series(vec![Rational::new(1, 1), Rational::new(2, 1)], 0);
        assert_eq!(series.ramification(), 3);
    }

    #[test]
    fn test_create_series() {
        let ring = PuiseuxSeriesRing::<Rational>::new(10);
        let series = ring.create_series(
            vec![Rational::new(1, 2), Rational::new(3, 4)],
            -1, // x^(-1/2)
            2,  // ramification 2
        );

        assert_eq!(series.valuation(), -1);
        assert_eq!(series.ramification(), 2);
    }

    #[test]
    fn test_is_algebraically_closed() {
        let ring = PuiseuxSeriesRing::<Rational>::new(10);
        // Currently returns false conservatively
        assert_eq!(ring.is_algebraically_closed(), false);
    }

    #[test]
    fn test_display() {
        let ring = PuiseuxSeriesRing::<Rational>::new(10);
        let display = format!("{}", ring);
        assert!(display.contains("Puiseux Series Ring"));
        assert!(display.contains("x"));
    }
}
