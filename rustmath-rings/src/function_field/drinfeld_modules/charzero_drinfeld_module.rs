//! Drinfeld modules in characteristic zero
//!
//! This module provides Drinfeld module structures for characteristic zero fields,
//! corresponding to SageMath's `sage.rings.function_field.drinfeld_modules.charzero_drinfeld_module`.
//!
//! # Mathematical Background
//!
//! A Drinfeld module in characteristic zero is a special case where the base field
//! has characteristic 0. These modules have simpler structure than their positive
//! characteristic counterparts because there's no Frobenius automorphism.
//!
//! Key properties in characteristic zero:
//! - The Frobenius τ acts as identity on the constant field
//! - Exponential and logarithm functions converge
//! - Rational Drinfeld modules have explicit formulas
//!
//! # Key Types
//!
//! - `DrinfeldModule_charzero`: Base class for characteristic zero Drinfeld modules
//! - `DrinfeldModule_rational`: Specialized for rational function fields
//!
//! # Examples
//!
//! ```ignore
//! use rustmath_rings::function_field::drinfeld_modules::charzero_drinfeld_module::*;
//!
//! // Create a characteristic zero Drinfeld module
//! let module = DrinfeldModule_charzero::new("Q(T)".to_string(), 1);
//!
//! // Create a rational Drinfeld module
//! let rational_module = DrinfeldModule_rational::new("Q".to_string());
//! ```

use rustmath_core::{Ring, Field};
use std::marker::PhantomData;
use std::fmt;

/// Base trait for Drinfeld modules
pub trait DrinfeldModule {
    /// Get the rank of the Drinfeld module
    fn rank(&self) -> usize;

    /// Get the base field
    fn base_field(&self) -> String;

    /// Check if the module is well-defined
    fn is_well_defined(&self) -> bool;
}

/// Drinfeld module in characteristic zero
///
/// Represents a Drinfeld module defined over a field of characteristic zero.
/// This corresponds to SageMath's `DrinfeldModule_charzero` class.
///
/// # Type Parameters
///
/// - `F`: The base field (must have characteristic zero)
///
/// # Mathematical Details
///
/// A Drinfeld module φ: A → K{τ} in characteristic zero satisfies:
/// - char(K) = 0
/// - τ acts as identity on the constant field
/// - The exponential function Exp_φ converges
/// - The logarithm function Log_φ is well-defined
#[derive(Clone, Debug)]
pub struct DrinfeldModuleCharzero<F: Field> {
    /// Name of the base field
    base_field: String,
    /// Rank of the Drinfeld module
    rank: usize,
    /// Field type marker
    field_marker: PhantomData<F>,
}

impl<F: Field> DrinfeldModule_charzero<F> {
    /// Create a new characteristic zero Drinfeld module
    ///
    /// # Arguments
    ///
    /// * `base_field` - Name of the base field
    /// * `rank` - Rank of the module
    ///
    /// # Returns
    ///
    /// A new DrinfeldModule_charzero instance
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rustmath_rings::function_field::drinfeld_modules::charzero_drinfeld_module::*;
    ///
    /// let module = DrinfeldModule_charzero::new("Q(T)".to_string(), 1);
    /// assert_eq!(module.rank(), 1);
    /// ```
    pub fn new(base_field: String, rank: usize) -> Self {
        assert!(rank > 0, "Rank must be positive");
        DrinfeldModule_charzero {
            base_field,
            rank,
            field_marker: PhantomData,
        }
    }

    /// Get the characteristic (always 0)
    ///
    /// # Returns
    ///
    /// The characteristic of the base field (always 0)
    pub fn characteristic(&self) -> usize {
        0
    }

    /// Check if exponential function converges
    ///
    /// # Returns
    ///
    /// True (always converges in characteristic zero)
    pub fn exponential_converges(&self) -> bool {
        true
    }

    /// Check if logarithm is well-defined
    ///
    /// # Returns
    ///
    /// True (always well-defined in characteristic zero)
    pub fn logarithm_defined(&self) -> bool {
        true
    }

    /// Get the height of the module
    ///
    /// # Returns
    ///
    /// The height (equals rank in characteristic zero)
    pub fn height(&self) -> usize {
        self.rank
    }
}

impl<F: Field> DrinfeldModule for DrinfeldModule_charzero<F> {
    fn rank(&self) -> usize {
        self.rank
    }

    fn base_field(&self) -> String {
        self.base_field.clone()
    }

    fn is_well_defined(&self) -> bool {
        self.rank > 0 && !self.base_field.is_empty()
    }
}

impl<F: Field> fmt::Display for DrinfeldModule_charzero<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DrinfeldModule_charzero(field={}, rank={})",
            self.base_field, self.rank
        )
    }
}

/// Rational Drinfeld module in characteristic zero
///
/// Specialized Drinfeld module for rational function fields in characteristic zero.
/// This corresponds to SageMath's `DrinfeldModule_rational` class.
///
/// # Type Parameters
///
/// - `F`: The base field (must have characteristic zero)
///
/// # Mathematical Details
///
/// For the rational function field K(T), rational Drinfeld modules have
/// particularly simple structure:
/// - φ_T = T + g_1·τ + ... + g_r·τ^r where g_i ∈ K
/// - The Carlitz module is the simplest example (rank 1)
#[derive(Clone, Debug)]
pub struct DrinfeldModuleRational<F: Field> {
    /// The underlying characteristic zero module
    base_module: DrinfeldModule_charzero<F>,
    /// Generator element name (typically "T")
    generator: String,
}

impl<F: Field> DrinfeldModule_rational<F> {
    /// Create a new rational Drinfeld module
    ///
    /// # Arguments
    ///
    /// * `base_field` - Name of the base field
    ///
    /// # Returns
    ///
    /// A new DrinfeldModule_rational instance (rank 1 by default)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rustmath_rings::function_field::drinfeld_modules::charzero_drinfeld_module::*;
    ///
    /// let module = DrinfeldModule_rational::new("Q".to_string());
    /// assert_eq!(module.generator(), "T");
    /// ```
    pub fn new(base_field: String) -> Self {
        DrinfeldModule_rational {
            base_module: DrinfeldModule_charzero::new(base_field, 1),
            generator: "T".to_string(),
        }
    }

    /// Create a rational module with specified rank
    ///
    /// # Arguments
    ///
    /// * `base_field` - Name of the base field
    /// * `rank` - Rank of the module
    ///
    /// # Returns
    ///
    /// A new DrinfeldModule_rational instance with specified rank
    pub fn with_rank(base_field: String, rank: usize) -> Self {
        DrinfeldModule_rational {
            base_module: DrinfeldModule_charzero::new(base_field, rank),
            generator: "T".to_string(),
        }
    }

    /// Get the generator name
    ///
    /// # Returns
    ///
    /// The generator element name
    pub fn generator(&self) -> &str {
        &self.generator
    }

    /// Set a custom generator name
    ///
    /// # Arguments
    ///
    /// * `gen` - New generator name
    pub fn set_generator(&mut self, gen: String) {
        self.generator = gen;
    }

    /// Check if this is the Carlitz module
    ///
    /// # Returns
    ///
    /// True if rank is 1 (Carlitz module structure)
    pub fn is_carlitz(&self) -> bool {
        self.base_module.rank() == 1
    }
}

impl<F: Field> DrinfeldModule for DrinfeldModule_rational<F> {
    fn rank(&self) -> usize {
        self.base_module.rank()
    }

    fn base_field(&self) -> String {
        self.base_module.base_field()
    }

    fn is_well_defined(&self) -> bool {
        self.base_module.is_well_defined() && !self.generator.is_empty()
    }
}

impl<F: Field> fmt::Display for DrinfeldModule_rational<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DrinfeldModule_rational(field={}, generator={}, rank={})",
            self.base_field(),
            self.generator,
            self.rank()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_charzero_module_creation() {
        let module: DrinfeldModule_charzero<Rational> =
            DrinfeldModule_charzero::new("Q(T)".to_string(), 1);

        assert_eq!(module.rank(), 1);
        assert_eq!(module.base_field(), "Q(T)");
        assert_eq!(module.characteristic(), 0);
        assert!(module.is_well_defined());
    }

    #[test]
    fn test_charzero_rank2_module() {
        let module: DrinfeldModule_charzero<Rational> =
            DrinfeldModule_charzero::new("Q(T)".to_string(), 2);

        assert_eq!(module.rank(), 2);
        assert_eq!(module.height(), 2);
    }

    #[test]
    #[should_panic(expected = "Rank must be positive")]
    fn test_charzero_invalid_rank() {
        let _module: DrinfeldModule_charzero<Rational> =
            DrinfeldModule_charzero::new("Q(T)".to_string(), 0);
    }

    #[test]
    fn test_charzero_convergence() {
        let module: DrinfeldModule_charzero<Rational> =
            DrinfeldModule_charzero::new("Q(T)".to_string(), 1);

        assert!(module.exponential_converges());
        assert!(module.logarithm_defined());
    }

    #[test]
    fn test_rational_module_creation() {
        let module: DrinfeldModule_rational<Rational> =
            DrinfeldModule_rational::new("Q".to_string());

        assert_eq!(module.rank(), 1);
        assert_eq!(module.base_field(), "Q");
        assert_eq!(module.generator(), "T");
        assert!(module.is_well_defined());
    }

    #[test]
    fn test_rational_module_with_rank() {
        let module: DrinfeldModule_rational<Rational> =
            DrinfeldModule_rational::with_rank("Q".to_string(), 3);

        assert_eq!(module.rank(), 3);
        assert!(!module.is_carlitz());
    }

    #[test]
    fn test_carlitz_module() {
        let module: DrinfeldModule_rational<Rational> =
            DrinfeldModule_rational::new("Q".to_string());

        assert!(module.is_carlitz());
    }

    #[test]
    fn test_custom_generator() {
        let mut module: DrinfeldModule_rational<Rational> =
            DrinfeldModule_rational::new("Q".to_string());

        assert_eq!(module.generator(), "T");
        module.set_generator("X".to_string());
        assert_eq!(module.generator(), "X");
    }

    #[test]
    fn test_module_display() {
        let module: DrinfeldModule_charzero<Rational> =
            DrinfeldModule_charzero::new("Q(T)".to_string(), 2);

        let display = format!("{}", module);
        assert!(display.contains("DrinfeldModule_charzero"));
        assert!(display.contains("Q(T)"));
        assert!(display.contains("rank=2"));
    }

    #[test]
    fn test_rational_display() {
        let module: DrinfeldModule_rational<Rational> =
            DrinfeldModule_rational::new("Q".to_string());

        let display = format!("{}", module);
        assert!(display.contains("DrinfeldModule_rational"));
        assert!(display.contains("generator=T"));
    }

    #[test]
    fn test_module_clone() {
        let module1: DrinfeldModule_charzero<Rational> =
            DrinfeldModule_charzero::new("Q(T)".to_string(), 1);
        let module2 = module1.clone();

        assert_eq!(module1.rank(), module2.rank());
        assert_eq!(module1.base_field(), module2.base_field());
    }
}
