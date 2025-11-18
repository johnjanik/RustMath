//! # Lazy Series Rings
//!
//! This module provides ring structures for lazy series, where series operations
//! are computed on demand with infinite precision.
//!
//! ## Overview
//!
//! Lazy series rings are parent structures for lazy series elements. They define:
//! - The coefficient ring
//! - The variable name(s)
//! - Construction and coercion rules
//! - Ring operations (product, sum, etc.)
//!
//! ## Ring Types
//!
//! - `LazySeriesRing`: Abstract base for all lazy series rings
//! - `LazyLaurentSeriesRing`: Laurent series with lazy evaluation
//! - `LazyPowerSeriesRing`: Power series with lazy evaluation
//! - `LazyDirichletSeriesRing`: Dirichlet series with multiplicative indexing
//! - `LazyCompletionGradedAlgebra`: Completion of graded algebras
//! - `LazySymmetricFunctions`: Symmetric function rings
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::lazy_series_ring::LazyPowerSeriesRing;
//! use rustmath_integers::Integer;
//!
//! let ring = LazyPowerSeriesRing::<Integer>::new("x".to_string());
//! assert_eq!(ring.variable(), "x");
//! ```

use rustmath_core::Ring;
use std::fmt;
use std::marker::PhantomData;

/// Abstract base trait for lazy series rings
pub trait LazySeriesRingTrait<R: Ring> {
    /// Returns the variable name
    fn variable(&self) -> &str;

    /// Checks if this is a field
    fn is_field(&self) -> bool;

    /// Returns a description of the ring
    fn description(&self) -> String;
}

/// Base structure for lazy series rings
///
/// Provides common functionality for all lazy series ring types.
#[derive(Clone, Debug)]
pub struct LazySeriesRing<R: Ring> {
    /// Variable name
    variable: String,
    /// Phantom data for the coefficient ring
    _phantom: PhantomData<R>,
}

impl<R: Ring> LazySeriesRing<R> {
    /// Creates a new lazy series ring
    pub fn new(variable: String) -> Self {
        LazySeriesRing {
            variable,
            _phantom: PhantomData,
        }
    }

    /// Returns the variable name
    pub fn variable(&self) -> &str {
        &self.variable
    }

    /// Checks if this ring is a field
    pub fn is_field(&self) -> bool {
        false // Conservative default
    }
}

impl<R: Ring> fmt::Display for LazySeriesRing<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Lazy Series Ring in {}", self.variable)
    }
}

/// Ring of lazy Laurent series
///
/// Implements the ring of Laurent series over a base ring with lazy coefficient computation.
/// Supports both positive and negative powers of the variable.
#[derive(Clone, Debug)]
pub struct LazyLaurentSeriesRing<R: Ring> {
    /// The base lazy series ring
    base: LazySeriesRing<R>,
}

impl<R: Ring> LazyLaurentSeriesRing<R> {
    /// Creates a new lazy Laurent series ring
    ///
    /// # Arguments
    ///
    /// * `variable` - Name of the variable
    pub fn new(variable: String) -> Self {
        LazyLaurentSeriesRing {
            base: LazySeriesRing::new(variable),
        }
    }

    /// Returns the variable name
    pub fn variable(&self) -> &str {
        self.base.variable()
    }

    /// Checks if this is a field
    pub fn is_field(&self) -> bool {
        // Laurent series ring over a field is a field
        self.base.is_field()
    }
}

impl<R: Ring> fmt::Display for LazyLaurentSeriesRing<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Lazy Laurent Series Ring in {}", self.variable())
    }
}

/// Ring of lazy power series
///
/// Implements univariate and multivariate Taylor series with lazy coefficient computation.
#[derive(Clone, Debug)]
pub struct LazyPowerSeriesRing<R: Ring> {
    /// The base lazy series ring
    base: LazySeriesRing<R>,
}

impl<R: Ring> LazyPowerSeriesRing<R> {
    /// Creates a new lazy power series ring
    ///
    /// # Arguments
    ///
    /// * `variable` - Name of the variable
    pub fn new(variable: String) -> Self {
        LazyPowerSeriesRing {
            base: LazySeriesRing::new(variable),
        }
    }

    /// Returns the variable name
    pub fn variable(&self) -> &str {
        self.base.variable()
    }
}

impl<R: Ring> fmt::Display for LazyPowerSeriesRing<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Lazy Power Series Ring in {}", self.variable())
    }
}

/// Ring of lazy Dirichlet series
///
/// Represents the ring of lazy Dirichlet series, which use multiplicative
/// rather than additive indexing.
#[derive(Clone, Debug)]
pub struct LazyDirichletSeriesRing<R: Ring> {
    /// The base lazy series ring
    base: LazySeriesRing<R>,
}

impl<R: Ring> LazyDirichletSeriesRing<R> {
    /// Creates a new lazy Dirichlet series ring
    ///
    /// # Arguments
    ///
    /// * `variable` - Name of the variable (typically "s")
    pub fn new(variable: String) -> Self {
        LazyDirichletSeriesRing {
            base: LazySeriesRing::new(variable),
        }
    }

    /// Returns the variable name
    pub fn variable(&self) -> &str {
        self.base.variable()
    }
}

impl<R: Ring> fmt::Display for LazyDirichletSeriesRing<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Lazy Dirichlet Series Ring in {}", self.variable())
    }
}

/// Completion of a graded algebra with lazy evaluation
///
/// Represents the completion of a graded algebra consisting of formal series.
#[derive(Clone, Debug)]
pub struct LazyCompletionGradedAlgebra<R: Ring> {
    /// The base lazy series ring
    base: LazySeriesRing<R>,
    /// Name of the graded algebra
    algebra_name: String,
}

impl<R: Ring> LazyCompletionGradedAlgebra<R> {
    /// Creates a new lazy completion of a graded algebra
    ///
    /// # Arguments
    ///
    /// * `variable` - Name of the variable
    /// * `algebra_name` - Name of the underlying graded algebra
    pub fn new(variable: String, algebra_name: String) -> Self {
        LazyCompletionGradedAlgebra {
            base: LazySeriesRing::new(variable),
            algebra_name,
        }
    }

    /// Returns the variable name
    pub fn variable(&self) -> &str {
        self.base.variable()
    }

    /// Returns the algebra name
    pub fn algebra_name(&self) -> &str {
        &self.algebra_name
    }
}

impl<R: Ring> fmt::Display for LazyCompletionGradedAlgebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Lazy Completion of {} in {}",
            self.algebra_name,
            self.variable()
        )
    }
}

/// Ring of lazy symmetric functions
///
/// Implements the ring of (possibly multivariate) lazy symmetric functions.
#[derive(Clone, Debug)]
pub struct LazySymmetricFunctions<R: Ring> {
    /// The underlying graded algebra
    base: LazyCompletionGradedAlgebra<R>,
}

impl<R: Ring> LazySymmetricFunctions<R> {
    /// Creates a new ring of lazy symmetric functions
    ///
    /// # Arguments
    ///
    /// * `variable` - Name of the variable
    pub fn new(variable: String) -> Self {
        LazySymmetricFunctions {
            base: LazyCompletionGradedAlgebra::new(variable, "Symmetric Functions".to_string()),
        }
    }

    /// Returns the variable name
    pub fn variable(&self) -> &str {
        self.base.variable()
    }
}

impl<R: Ring> fmt::Display for LazySymmetricFunctions<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Lazy Symmetric Functions in {}", self.variable())
    }
}

/// Configuration options for lazy series rings
#[derive(Clone, Debug)]
pub struct LazySeriesOptions {
    /// Display precision
    pub display_precision: usize,
    /// Whether to show coefficients in parentheses
    pub parentheses: bool,
}

impl Default for LazySeriesOptions {
    fn default() -> Self {
        LazySeriesOptions {
            display_precision: 10,
            parentheses: false,
        }
    }
}

/// Global options instance
pub static OPTIONS: LazySeriesOptions = LazySeriesOptions {
    display_precision: 10,
    parentheses: false,
};

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_lazy_series_ring() {
        let ring = LazySeriesRing::<Integer>::new("x".to_string());
        assert_eq!(ring.variable(), "x");
    }

    #[test]
    fn test_lazy_laurent_series_ring() {
        let ring = LazyLaurentSeriesRing::<Integer>::new("t".to_string());
        assert_eq!(ring.variable(), "t");
    }

    #[test]
    fn test_lazy_power_series_ring() {
        let ring = LazyPowerSeriesRing::<Integer>::new("x".to_string());
        assert_eq!(ring.variable(), "x");
    }

    #[test]
    fn test_lazy_dirichlet_series_ring() {
        let ring = LazyDirichletSeriesRing::<Integer>::new("s".to_string());
        assert_eq!(ring.variable(), "s");
    }

    #[test]
    fn test_lazy_graded_algebra() {
        let alg = LazyCompletionGradedAlgebra::<Integer>::new(
            "x".to_string(),
            "Test Algebra".to_string(),
        );
        assert_eq!(alg.variable(), "x");
        assert_eq!(alg.algebra_name(), "Test Algebra");
    }

    #[test]
    fn test_lazy_symmetric_functions() {
        let funcs = LazySymmetricFunctions::<Integer>::new("x".to_string());
        assert_eq!(funcs.variable(), "x");
    }

    #[test]
    fn test_ring_display() {
        let ring = LazyPowerSeriesRing::<Integer>::new("x".to_string());
        let display = format!("{}", ring);
        assert!(display.contains("Lazy Power Series"));
        assert!(display.contains("x"));
    }

    #[test]
    fn test_laurent_ring_display() {
        let ring = LazyLaurentSeriesRing::<Integer>::new("t".to_string());
        let display = format!("{}", ring);
        assert!(display.contains("Laurent"));
        assert!(display.contains("t"));
    }

    #[test]
    fn test_dirichlet_ring_display() {
        let ring = LazyDirichletSeriesRing::<Integer>::new("s".to_string());
        let display = format!("{}", ring);
        assert!(display.contains("Dirichlet"));
    }

    #[test]
    fn test_graded_algebra_display() {
        let alg = LazyCompletionGradedAlgebra::<Integer>::new(
            "x".to_string(),
            "Poly".to_string(),
        );
        let display = format!("{}", alg);
        assert!(display.contains("Completion"));
        assert!(display.contains("Poly"));
    }

    #[test]
    fn test_symmetric_functions_display() {
        let funcs = LazySymmetricFunctions::<Integer>::new("x".to_string());
        let display = format!("{}", funcs);
        assert!(display.contains("Symmetric"));
    }

    #[test]
    fn test_options() {
        assert_eq!(OPTIONS.display_precision, 10);
        assert_eq!(OPTIONS.parentheses, false);
    }

    #[test]
    fn test_default_options() {
        let opts = LazySeriesOptions::default();
        assert_eq!(opts.display_precision, 10);
        assert!(!opts.parentheses);
    }
}
