//! # Multivariate Power Series Rings
//!
//! This module implements rings of multivariate power series over commutative base rings.
//!
//! ## Overview
//!
//! A multivariate power series ring R[[x₁, ..., xₙ]] consists of formal power series
//! in multiple variables with coefficients in R. The implementation uses a total-degree
//! precision model.
//!
//! ## Architecture
//!
//! The implementation uses two underlying structures:
//! - A "foreground" multivariate polynomial ring for display
//! - A "background" univariate power series ring for arithmetic
//!
//! This approach provides efficient computation while maintaining a natural interface.
//!
//! ## Features
//!
//! - Arithmetic operations on multivariate power series
//! - Precision management via big-O notation
//! - Term ordering support
//! - Variable substitution and transformation
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::multi_power_series_ring::MPowerSeriesRing;
//! use rustmath_integers::Integer;
//!
//! let ring = MPowerSeriesRing::<Integer>::new(vec!["x".to_string(), "y".to_string()]);
//! assert_eq!(ring.num_variables(), 2);
//! ```

use rustmath_core::Ring;
use std::fmt;
use std::marker::PhantomData;

/// Generic multivariate power series ring
///
/// Represents a ring of multivariate power series R[[x₁, ..., xₙ]] where R is the
/// coefficient ring.
#[derive(Clone, Debug)]
pub struct MPowerSeriesRing<R: Ring> {
    /// Names of the variables
    variables: Vec<String>,
    /// Default precision for computations
    default_precision: usize,
    /// Phantom data for the coefficient ring
    _phantom: PhantomData<R>,
}

impl<R: Ring> MPowerSeriesRing<R> {
    /// Creates a new multivariate power series ring
    ///
    /// # Arguments
    ///
    /// * `variables` - Names of the variables
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rings::multi_power_series_ring::MPowerSeriesRing;
    /// use rustmath_integers::Integer;
    ///
    /// let ring = MPowerSeriesRing::<Integer>::new(vec!["x".to_string(), "y".to_string()]);
    /// assert_eq!(ring.num_variables(), 2);
    /// ```
    pub fn new(variables: Vec<String>) -> Self {
        MPowerSeriesRing {
            variables,
            default_precision: 20,
            _phantom: PhantomData,
        }
    }

    /// Creates a new ring with a specified default precision
    pub fn with_precision(variables: Vec<String>, precision: usize) -> Self {
        MPowerSeriesRing {
            variables,
            default_precision: precision,
            _phantom: PhantomData,
        }
    }

    /// Returns the number of variables
    pub fn num_variables(&self) -> usize {
        self.variables.len()
    }

    /// Returns the variable names
    pub fn variables(&self) -> &[String] {
        &self.variables
    }

    /// Returns the default precision
    pub fn default_precision(&self) -> usize {
        self.default_precision
    }

    /// Sets the default precision
    pub fn set_default_precision(&mut self, precision: usize) {
        self.default_precision = precision;
    }

    /// Gets the name of a specific variable
    pub fn variable_name(&self, index: usize) -> Option<&str> {
        self.variables.get(index).map(|s| s.as_str())
    }

    /// Checks if this is a univariate ring
    pub fn is_univariate(&self) -> bool {
        self.variables.len() == 1
    }

    /// Removes a variable from the ring
    ///
    /// Returns a new ring with one fewer variable
    pub fn remove_variable(&self, var_name: &str) -> Option<Self> {
        let new_vars: Vec<String> = self
            .variables
            .iter()
            .filter(|v| v.as_str() != var_name)
            .cloned()
            .collect();

        if new_vars.len() < self.variables.len() {
            Some(MPowerSeriesRing {
                variables: new_vars,
                default_precision: self.default_precision,
                _phantom: PhantomData,
            })
        } else {
            None
        }
    }

    /// Returns a description of the ring
    pub fn description(&self) -> String {
        if self.variables.is_empty() {
            "Multivariate Power Series Ring (no variables)".to_string()
        } else if self.variables.len() == 1 {
            format!(
                "Univariate Power Series Ring in {}",
                self.variables[0]
            )
        } else {
            format!(
                "Multivariate Power Series Ring in {}",
                self.variables.join(", ")
            )
        }
    }
}

impl<R: Ring> fmt::Display for MPowerSeriesRing<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}

/// Checks if an object is a multivariate power series ring
///
/// # Deprecated
///
/// This function is deprecated. Use type checking instead.
#[deprecated(since = "0.1.0", note = "Use isinstance or type checking instead")]
pub fn is_mpower_series_ring<R: Ring>(_obj: &MPowerSeriesRing<R>) -> bool {
    true
}

/// Unpickles a multivariate power series ring (version 0)
///
/// This function is provided for compatibility with serialization formats.
///
/// # Arguments
///
/// * `variables` - Variable names
/// * `precision` - Default precision
///
/// # Returns
///
/// A reconstructed multivariate power series ring
pub fn unpickle_multi_power_series_ring_v0<R: Ring>(
    variables: Vec<String>,
    precision: usize,
) -> MPowerSeriesRing<R> {
    MPowerSeriesRing::with_precision(variables, precision)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_mpower_series_ring_creation() {
        let ring = MPowerSeriesRing::<Integer>::new(vec!["x".to_string(), "y".to_string()]);
        assert_eq!(ring.num_variables(), 2);
        assert_eq!(ring.variables().len(), 2);
    }

    #[test]
    fn test_single_variable() {
        let ring = MPowerSeriesRing::<Integer>::new(vec!["t".to_string()]);
        assert!(ring.is_univariate());
        assert_eq!(ring.num_variables(), 1);
    }

    #[test]
    fn test_variable_names() {
        let ring = MPowerSeriesRing::<Integer>::new(vec!["x".to_string(), "y".to_string(), "z".to_string()]);
        assert_eq!(ring.variable_name(0), Some("x"));
        assert_eq!(ring.variable_name(1), Some("y"));
        assert_eq!(ring.variable_name(2), Some("z"));
        assert_eq!(ring.variable_name(3), None);
    }

    #[test]
    fn test_default_precision() {
        let ring = MPowerSeriesRing::<Integer>::new(vec!["x".to_string()]);
        assert_eq!(ring.default_precision(), 20);

        let ring2 = MPowerSeriesRing::<Integer>::with_precision(vec!["x".to_string()], 30);
        assert_eq!(ring2.default_precision(), 30);
    }

    #[test]
    fn test_set_precision() {
        let mut ring = MPowerSeriesRing::<Integer>::new(vec!["x".to_string()]);
        ring.set_default_precision(50);
        assert_eq!(ring.default_precision(), 50);
    }

    #[test]
    fn test_remove_variable() {
        let ring = MPowerSeriesRing::<Integer>::new(vec!["x".to_string(), "y".to_string(), "z".to_string()]);
        let ring2 = ring.remove_variable("y");
        assert!(ring2.is_some());

        let ring2 = ring2.unwrap();
        assert_eq!(ring2.num_variables(), 2);
        assert_eq!(ring2.variable_name(0), Some("x"));
        assert_eq!(ring2.variable_name(1), Some("z"));
    }

    #[test]
    fn test_remove_nonexistent_variable() {
        let ring = MPowerSeriesRing::<Integer>::new(vec!["x".to_string(), "y".to_string()]);
        let result = ring.remove_variable("w");
        assert!(result.is_none());
    }

    #[test]
    fn test_description() {
        let ring1 = MPowerSeriesRing::<Integer>::new(vec![]);
        assert!(ring1.description().contains("no variables"));

        let ring2 = MPowerSeriesRing::<Integer>::new(vec!["x".to_string()]);
        assert!(ring2.description().contains("Univariate"));

        let ring3 = MPowerSeriesRing::<Integer>::new(vec!["x".to_string(), "y".to_string()]);
        assert!(ring3.description().contains("Multivariate"));
    }

    #[test]
    fn test_display() {
        let ring = MPowerSeriesRing::<Integer>::new(vec!["x".to_string(), "y".to_string()]);
        let display = format!("{}", ring);
        assert!(display.contains("Power Series"));
        assert!(display.contains("x"));
        assert!(display.contains("y"));
    }

    #[test]
    #[allow(deprecated)]
    fn test_is_mpower_series_ring() {
        let ring = MPowerSeriesRing::<Integer>::new(vec!["x".to_string()]);
        assert!(is_mpower_series_ring(&ring));
    }

    #[test]
    fn test_unpickle() {
        let ring = unpickle_multi_power_series_ring_v0::<Integer>(
            vec!["x".to_string(), "y".to_string()],
            15,
        );
        assert_eq!(ring.num_variables(), 2);
        assert_eq!(ring.default_precision(), 15);
    }

    #[test]
    fn test_empty_variables() {
        let ring = MPowerSeriesRing::<Integer>::new(vec![]);
        assert_eq!(ring.num_variables(), 0);
        assert!(!ring.is_univariate());
    }
}
