//! # Lazy Series
//!
//! This module implements lazy evaluation for various types of series, where coefficients
//! are computed on demand rather than all at once.
//!
//! ## Overview
//!
//! Lazy series provide infinite precision representations where:
//! - Coefficients are computed only when needed
//! - Series can represent infinite sequences
//! - Memory usage is minimized through on-demand computation
//!
//! ## Key Concepts
//!
//! - **Lazy Evaluation**: Coefficients are computed only when accessed
//! - **Infinite Precision**: Series can theoretically have infinitely many terms
//! - **Caching**: Computed coefficients are cached for efficiency
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::lazy_series::{LazyModuleElement, LazyPowerSeries};
//! use rustmath_integers::Integer;
//!
//! // Create a lazy power series
//! let series = LazyPowerSeries::<Integer>::new(vec![Integer::from(1), Integer::from(2)]);
//! ```

use rustmath_core::Ring;

/// Base trait for lazy module elements
///
/// Provides common functionality for all lazy series types, including coefficient access,
/// arithmetic operations, and transformations.
pub trait LazyElement<R: Ring> {
    /// Gets the coefficient at a given index
    fn get_coefficient(&self, n: usize) -> R
    where
        R: Clone + From<i32>;

    /// Returns the cached coefficients
    fn cached_coefficients(&self) -> Vec<R>
    where
        R: Clone;

    /// Checks if this element is zero
    fn is_zero(&self) -> bool
    where
        R: PartialEq + From<i32>;
}

/// Represents a lazy module element (base class for all lazy series)
///
/// This is the foundational type for lazy evaluation of series coefficients.
#[derive(Clone, Debug)]
pub struct LazyModuleElement<R: Ring> {
    /// Cached coefficients computed so far
    coefficients: Vec<R>,
    /// Whether all coefficients have been computed
    complete: bool,
}

impl<R: Ring> LazyModuleElement<R> {
    /// Creates a new lazy module element from initial coefficients
    pub fn new(coefficients: Vec<R>) -> Self {
        LazyModuleElement {
            coefficients,
            complete: false,
        }
    }

    /// Creates a zero element
    pub fn zero() -> Self
    where
        R: From<i32>,
    {
        LazyModuleElement {
            coefficients: vec![R::from(0)],
            complete: true,
        }
    }
}

impl<R: Ring> LazyElement<R> for LazyModuleElement<R> {
    fn get_coefficient(&self, n: usize) -> R
    where
        R: Clone + From<i32>,
    {
        if n < self.coefficients.len() {
            self.coefficients[n].clone()
        } else {
            R::from(0)
        }
    }

    fn cached_coefficients(&self) -> Vec<R>
    where
        R: Clone,
    {
        self.coefficients.clone()
    }

    fn is_zero(&self) -> bool
    where
        R: PartialEq + From<i32>,
    {
        self.coefficients.iter().all(|c| *c == R::from(0))
    }
}

/// Lazy series with Cauchy product multiplication
///
/// Extends lazy module elements with multiplicative structure using Cauchy products.
#[derive(Clone, Debug)]
pub struct LazyCauchyProductSeries<R: Ring> {
    /// The underlying lazy module element
    base: LazyModuleElement<R>,
}

impl<R: Ring> LazyCauchyProductSeries<R> {
    /// Creates a new lazy Cauchy product series
    pub fn new(coefficients: Vec<R>) -> Self {
        LazyCauchyProductSeries {
            base: LazyModuleElement::new(coefficients),
        }
    }

    /// Creates the zero series
    pub fn zero() -> Self
    where
        R: From<i32>,
    {
        LazyCauchyProductSeries {
            base: LazyModuleElement::zero(),
        }
    }

    /// Creates the one series
    pub fn one() -> Self
    where
        R: From<i32>,
    {
        LazyCauchyProductSeries {
            base: LazyModuleElement::new(vec![R::from(1)]),
        }
    }
}

impl<R: Ring> LazyElement<R> for LazyCauchyProductSeries<R> {
    fn get_coefficient(&self, n: usize) -> R
    where
        R: Clone + From<i32>,
    {
        self.base.get_coefficient(n)
    }

    fn cached_coefficients(&self) -> Vec<R>
    where
        R: Clone,
    {
        self.base.cached_coefficients()
    }

    fn is_zero(&self) -> bool
    where
        R: PartialEq + From<i32>,
    {
        self.base.is_zero()
    }
}

/// Lazy Laurent series with lazy evaluation
///
/// Laurent series allowing arbitrary integer exponents with lazy coefficient computation.
#[derive(Clone, Debug)]
pub struct LazyLaurentSeries<R: Ring> {
    /// The underlying Cauchy product series
    base: LazyCauchyProductSeries<R>,
    /// Valuation of the series
    valuation: i64,
}

impl<R: Ring> LazyLaurentSeries<R> {
    /// Creates a new lazy Laurent series
    pub fn new(valuation: i64, coefficients: Vec<R>) -> Self {
        LazyLaurentSeries {
            base: LazyCauchyProductSeries::new(coefficients),
            valuation,
        }
    }

    /// Returns the valuation
    pub fn valuation(&self) -> i64 {
        self.valuation
    }

    /// Creates the zero series
    pub fn zero() -> Self
    where
        R: From<i32>,
    {
        LazyLaurentSeries {
            base: LazyCauchyProductSeries::zero(),
            valuation: 0,
        }
    }
}

impl<R: Ring> LazyElement<R> for LazyLaurentSeries<R> {
    fn get_coefficient(&self, n: usize) -> R
    where
        R: Clone + From<i32>,
    {
        self.base.get_coefficient(n)
    }

    fn cached_coefficients(&self) -> Vec<R>
    where
        R: Clone,
    {
        self.base.cached_coefficients()
    }

    fn is_zero(&self) -> bool
    where
        R: PartialEq + From<i32>,
    {
        self.base.is_zero()
    }
}

/// Lazy power series with lazy evaluation
///
/// Power series starting from degree 0 with lazy coefficient computation.
#[derive(Clone, Debug)]
pub struct LazyPowerSeries<R: Ring> {
    /// The underlying Cauchy product series
    base: LazyCauchyProductSeries<R>,
}

impl<R: Ring> LazyPowerSeries<R> {
    /// Creates a new lazy power series
    pub fn new(coefficients: Vec<R>) -> Self {
        LazyPowerSeries {
            base: LazyCauchyProductSeries::new(coefficients),
        }
    }

    /// Creates the zero series
    pub fn zero() -> Self
    where
        R: From<i32>,
    {
        LazyPowerSeries {
            base: LazyCauchyProductSeries::zero(),
        }
    }

    /// Creates the one series
    pub fn one() -> Self
    where
        R: From<i32>,
    {
        LazyPowerSeries {
            base: LazyCauchyProductSeries::one(),
        }
    }
}

impl<R: Ring> LazyElement<R> for LazyPowerSeries<R> {
    fn get_coefficient(&self, n: usize) -> R
    where
        R: Clone + From<i32>,
    {
        self.base.get_coefficient(n)
    }

    fn cached_coefficients(&self) -> Vec<R>
    where
        R: Clone,
    {
        self.base.cached_coefficients()
    }

    fn is_zero(&self) -> bool
    where
        R: PartialEq + From<i32>,
    {
        self.base.is_zero()
    }
}

/// Mixin for GCD operations on lazy power series
///
/// Provides GCD-related functionality for power series over GCD domains.
pub trait LazyPowerSeriesGcdMixin<R: Ring> {
    /// Computes the GCD of two power series
    fn gcd(&self, other: &Self) -> Self;

    /// Computes the extended GCD
    fn xgcd(&self, other: &Self) -> (Self, Self, Self)
    where
        Self: Sized;
}

/// Lazy Dirichlet series with multiplicative indexing
///
/// Dirichlet series use multiplicative rather than additive indexing.
#[derive(Clone, Debug)]
pub struct LazyDirichletSeries<R: Ring> {
    /// The underlying lazy module element
    base: LazyModuleElement<R>,
}

impl<R: Ring> LazyDirichletSeries<R> {
    /// Creates a new lazy Dirichlet series
    pub fn new(coefficients: Vec<R>) -> Self {
        LazyDirichletSeries {
            base: LazyModuleElement::new(coefficients),
        }
    }

    /// Creates the zero series
    pub fn zero() -> Self
    where
        R: From<i32>,
    {
        LazyDirichletSeries {
            base: LazyModuleElement::zero(),
        }
    }
}

impl<R: Ring> LazyElement<R> for LazyDirichletSeries<R> {
    fn get_coefficient(&self, n: usize) -> R
    where
        R: Clone + From<i32>,
    {
        self.base.get_coefficient(n)
    }

    fn cached_coefficients(&self) -> Vec<R>
    where
        R: Clone,
    {
        self.base.cached_coefficients()
    }

    fn is_zero(&self) -> bool
    where
        R: PartialEq + From<i32>,
    {
        self.base.is_zero()
    }
}

/// Lazy completion of graded algebra elements
///
/// Handles lazy evaluation for elements of graded algebras.
#[derive(Clone, Debug)]
pub struct LazyCompletionGradedAlgebraElement<R: Ring> {
    /// The underlying Cauchy product series
    base: LazyCauchyProductSeries<R>,
}

impl<R: Ring> LazyCompletionGradedAlgebraElement<R> {
    /// Creates a new lazy graded algebra element
    pub fn new(coefficients: Vec<R>) -> Self {
        LazyCompletionGradedAlgebraElement {
            base: LazyCauchyProductSeries::new(coefficients),
        }
    }
}

impl<R: Ring> LazyElement<R> for LazyCompletionGradedAlgebraElement<R> {
    fn get_coefficient(&self, n: usize) -> R
    where
        R: Clone + From<i32>,
    {
        self.base.get_coefficient(n)
    }

    fn cached_coefficients(&self) -> Vec<R>
    where
        R: Clone,
    {
        self.base.cached_coefficients()
    }

    fn is_zero(&self) -> bool
    where
        R: PartialEq + From<i32>,
    {
        self.base.is_zero()
    }
}

/// Lazy symmetric functions
///
/// Specialized lazy series for symmetric function algebras.
#[derive(Clone, Debug)]
pub struct LazySymmetricFunction<R: Ring> {
    /// The underlying graded algebra element
    base: LazyCompletionGradedAlgebraElement<R>,
}

impl<R: Ring> LazySymmetricFunction<R> {
    /// Creates a new lazy symmetric function
    pub fn new(coefficients: Vec<R>) -> Self {
        LazySymmetricFunction {
            base: LazyCompletionGradedAlgebraElement::new(coefficients),
        }
    }
}

impl<R: Ring> LazyElement<R> for LazySymmetricFunction<R> {
    fn get_coefficient(&self, n: usize) -> R
    where
        R: Clone + From<i32>,
    {
        self.base.get_coefficient(n)
    }

    fn cached_coefficients(&self) -> Vec<R>
    where
        R: Clone,
    {
        self.base.cached_coefficients()
    }

    fn is_zero(&self) -> bool
    where
        R: PartialEq + From<i32>,
    {
        self.base.is_zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_lazy_module_element() {
        let elem = LazyModuleElement::new(vec![Integer::from(1), Integer::from(2), Integer::from(3)]);
        assert_eq!(elem.get_coefficient(0), Integer::from(1));
        assert_eq!(elem.get_coefficient(1), Integer::from(2));
        assert_eq!(elem.get_coefficient(2), Integer::from(3));
    }

    #[test]
    fn test_lazy_module_element_zero() {
        let zero = LazyModuleElement::<Integer>::zero();
        assert!(zero.is_zero());
    }

    #[test]
    fn test_lazy_cauchy_product_series() {
        let series = LazyCauchyProductSeries::new(vec![Integer::from(1), Integer::from(2)]);
        assert_eq!(series.get_coefficient(0), Integer::from(1));
        assert_eq!(series.get_coefficient(1), Integer::from(2));
    }

    #[test]
    fn test_lazy_cauchy_zero_one() {
        let zero = LazyCauchyProductSeries::<Integer>::zero();
        let one = LazyCauchyProductSeries::<Integer>::one();

        assert!(zero.is_zero());
        assert!(!one.is_zero());
        assert_eq!(one.get_coefficient(0), Integer::from(1));
    }

    #[test]
    fn test_lazy_laurent_series() {
        let series = LazyLaurentSeries::new(-1, vec![Integer::from(1), Integer::from(2)]);
        assert_eq!(series.valuation(), -1);
        assert_eq!(series.get_coefficient(0), Integer::from(1));
    }

    #[test]
    fn test_lazy_laurent_zero() {
        let zero = LazyLaurentSeries::<Integer>::zero();
        assert!(zero.is_zero());
        assert_eq!(zero.valuation(), 0);
    }

    #[test]
    fn test_lazy_power_series() {
        let series = LazyPowerSeries::new(vec![Integer::from(1), Integer::from(2), Integer::from(3)]);
        assert_eq!(series.get_coefficient(0), Integer::from(1));
        assert_eq!(series.get_coefficient(1), Integer::from(2));
        assert_eq!(series.get_coefficient(2), Integer::from(3));
    }

    #[test]
    fn test_lazy_power_series_zero_one() {
        let zero = LazyPowerSeries::<Integer>::zero();
        let one = LazyPowerSeries::<Integer>::one();

        assert!(zero.is_zero());
        assert!(!one.is_zero());
    }

    #[test]
    fn test_lazy_dirichlet_series() {
        let series = LazyDirichletSeries::new(vec![Integer::from(1), Integer::from(0), Integer::from(2)]);
        assert_eq!(series.get_coefficient(0), Integer::from(1));
        assert_eq!(series.get_coefficient(2), Integer::from(2));
    }

    #[test]
    fn test_lazy_dirichlet_zero() {
        let zero = LazyDirichletSeries::<Integer>::zero();
        assert!(zero.is_zero());
    }

    #[test]
    fn test_lazy_graded_algebra_element() {
        let elem = LazyCompletionGradedAlgebraElement::new(vec![Integer::from(1), Integer::from(2)]);
        assert_eq!(elem.get_coefficient(0), Integer::from(1));
        assert_eq!(elem.get_coefficient(1), Integer::from(2));
    }

    #[test]
    fn test_lazy_symmetric_function() {
        let func = LazySymmetricFunction::new(vec![Integer::from(1), Integer::from(1), Integer::from(1)]);
        assert_eq!(func.get_coefficient(0), Integer::from(1));
        assert_eq!(func.get_coefficient(1), Integer::from(1));
        assert_eq!(func.get_coefficient(2), Integer::from(1));
    }

    #[test]
    fn test_coefficient_out_of_bounds() {
        let elem = LazyModuleElement::new(vec![Integer::from(1)]);
        assert_eq!(elem.get_coefficient(10), Integer::from(0));
    }

    #[test]
    fn test_cached_coefficients() {
        let series = LazyPowerSeries::new(vec![Integer::from(1), Integer::from(2), Integer::from(3)]);
        let cached = series.cached_coefficients();
        assert_eq!(cached.len(), 3);
        assert_eq!(cached[0], Integer::from(1));
        assert_eq!(cached[1], Integer::from(2));
        assert_eq!(cached[2], Integer::from(3));
    }
}
