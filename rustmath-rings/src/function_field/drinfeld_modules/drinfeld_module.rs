//! Base Drinfeld module implementation
//!
//! This module provides the base Drinfeld module structure, corresponding to
//! SageMath's `sage.rings.function_field.drinfeld_modules.drinfeld_module`.
//!
//! # Mathematical Background
//!
//! A Drinfeld module is a ring homomorphism φ: A → K{τ} where:
//! - A is a Dedekind domain (typically Fq[T])
//! - K is a field containing Fq
//! - K{τ} is the twisted polynomial ring with τα = α^q·τ for α ∈ Fq
//!
//! Drinfeld modules were introduced by Vladimir Drinfeld in 1974 as function field
//! analogues of elliptic curves, playing a crucial role in the Langlands program
//! for function fields.
//!
//! Key properties:
//! - φ(a + b) = φ(a) + φ(b) (additive homomorphism)
//! - φ(ab) = φ(a) ∘ φ(b) (multiplicative homomorphism)
//! - φ(a) has degree deg(a) when viewing elements as polynomials
//!
//! # Key Types
//!
//! - `DrinfeldModule`: Generic Drinfeld module structure
//!
//! # Examples
//!
//! ```ignore
//! use rustmath_rings::function_field::drinfeld_modules::drinfeld_module::*;
//!
//! // Create a Drinfeld module over Fq[T]
//! let module = DrinfeldModule::new("Fq[T]".to_string(), "Fq2".to_string(), 1);
//!
//! // Check rank and properties
//! assert_eq!(module.rank(), 1);
//! assert!(module.is_separable());
//! ```

use rustmath_core::{Ring, Field};
use std::marker::PhantomData;
use std::fmt;

/// Generic Drinfeld module structure
///
/// Represents a Drinfeld module φ: A → K{τ}.
/// This corresponds to SageMath's `DrinfeldModule` class.
///
/// # Type Parameters
///
/// - `F`: The field type for K
/// - `R`: The ring type for A
///
/// # Mathematical Details
///
/// For a Drinfeld module φ: Fq[T] → K{τ} of rank r, we have:
/// φ(T) = T + g_1·τ + g_2·τ² + ... + g_r·τ^r
/// where g_r ≠ 0 and g_i ∈ K.
///
/// Important invariants:
/// - **Rank**: The degree of φ(T) as a polynomial in τ
/// - **Height**: Related to the inseparability of the module
/// - **Characteristic**: The characteristic of the base field
#[derive(Clone, Debug)]
pub struct DrinfeldModule<F: Field, R: Ring> {
    /// Base ring A (domain of φ)
    base_ring: String,
    /// Coefficient field K (contains constants)
    coefficient_field: String,
    /// Rank of the module
    rank: usize,
    /// Characteristic of the base field
    characteristic: usize,
    /// Coefficients of φ(T) = T + g_1·τ + ... + g_r·τ^r
    coefficients: Vec<String>,
    /// Field marker
    field_marker: PhantomData<F>,
    /// Ring marker
    ring_marker: PhantomData<R>,
}

impl<F: Field, R: Ring> DrinfeldModule<F, R> {
    /// Create a new Drinfeld module
    ///
    /// # Arguments
    ///
    /// * `base_ring` - The base ring A
    /// * `coefficient_field` - The coefficient field K
    /// * `rank` - Rank of the module
    ///
    /// # Returns
    ///
    /// A new DrinfeldModule instance
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let module = DrinfeldModule::new("Fq[T]".to_string(), "Fq2".to_string(), 1);
    /// assert_eq!(module.rank(), 1);
    /// ```
    pub fn new(base_ring: String, coefficient_field: String, rank: usize) -> Self {
        assert!(rank > 0, "Rank must be positive");
        DrinfeldModule {
            base_ring,
            coefficient_field,
            rank,
            characteristic: 0, // To be determined
            coefficients: vec!["0".to_string(); rank + 1],
            field_marker: PhantomData,
            ring_marker: PhantomData,
        }
    }

    /// Create a Drinfeld module with specified characteristic
    ///
    /// # Arguments
    ///
    /// * `base_ring` - The base ring A
    /// * `coefficient_field` - The coefficient field K
    /// * `rank` - Rank of the module
    /// * `characteristic` - Characteristic of the base field
    ///
    /// # Returns
    ///
    /// A new DrinfeldModule instance
    pub fn with_characteristic(
        base_ring: String,
        coefficient_field: String,
        rank: usize,
        characteristic: usize,
    ) -> Self {
        assert!(rank > 0, "Rank must be positive");
        DrinfeldModule {
            base_ring,
            coefficient_field,
            rank,
            characteristic,
            coefficients: vec!["0".to_string(); rank + 1],
            field_marker: PhantomData,
            ring_marker: PhantomData,
        }
    }

    /// Get the rank of the Drinfeld module
    ///
    /// # Returns
    ///
    /// The rank (degree of φ(T) in τ)
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the base ring
    ///
    /// # Returns
    ///
    /// Name of the base ring
    pub fn base_ring(&self) -> &str {
        &self.base_ring
    }

    /// Get the coefficient field
    ///
    /// # Returns
    ///
    /// Name of the coefficient field
    pub fn coefficient_field(&self) -> &str {
        &self.coefficient_field
    }

    /// Get the characteristic
    ///
    /// # Returns
    ///
    /// The characteristic of the base field
    pub fn characteristic(&self) -> usize {
        self.characteristic
    }

    /// Check if the module is separable
    ///
    /// # Returns
    ///
    /// True if the module is separable (rank equals height)
    pub fn is_separable(&self) -> bool {
        // In separable case, rank equals height
        true // Simplified for basic implementation
    }

    /// Get the height of the module
    ///
    /// # Returns
    ///
    /// The height (related to inseparability)
    pub fn height(&self) -> usize {
        self.rank // Simplified: height = rank for separable modules
    }

    /// Set a coefficient
    ///
    /// # Arguments
    ///
    /// * `index` - Index of the coefficient (0 to rank)
    /// * `value` - String representation of the coefficient
    pub fn set_coefficient(&mut self, index: usize, value: String) {
        if index < self.coefficients.len() {
            self.coefficients[index] = value;
        }
    }

    /// Get a coefficient
    ///
    /// # Arguments
    ///
    /// * `index` - Index of the coefficient
    ///
    /// # Returns
    ///
    /// The coefficient at the given index
    pub fn get_coefficient(&self, index: usize) -> Option<&String> {
        self.coefficients.get(index)
    }

    /// Check if the module is well-defined
    ///
    /// # Returns
    ///
    /// True if the module satisfies basic requirements
    pub fn is_well_defined(&self) -> bool {
        self.rank > 0 && !self.base_ring.is_empty() && !self.coefficient_field.is_empty()
    }

    /// Get the action on an element
    ///
    /// # Arguments
    ///
    /// * `element` - String representation of the element
    ///
    /// # Returns
    ///
    /// String representation of φ(element)
    pub fn action(&self, element: &str) -> String {
        format!("φ({})", element)
    }
}

impl<F: Field, R: Ring> fmt::Display for DrinfeldModule<F, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DrinfeldModule(base={}, field={}, rank={})",
            self.base_ring, self.coefficient_field, self.rank
        )
    }
}

/// Factory functions for common Drinfeld modules
pub struct DrinfeldModuleFactory;

impl DrinfeldModuleFactory {
    /// Create the Carlitz module (rank 1 over Fq[T])
    ///
    /// # Arguments
    ///
    /// * `q` - The size of the finite field Fq
    ///
    /// # Returns
    ///
    /// A Drinfeld module representing the Carlitz module
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let carlitz = DrinfeldModuleFactory::carlitz_module(2);
    /// assert_eq!(carlitz.rank(), 1);
    /// ```
    pub fn carlitz_module<F: Field, R: Ring>(q: usize) -> DrinfeldModule<F, R> {
        DrinfeldModule::with_characteristic(
            format!("F{}[T]", q),
            format!("F{}", q),
            1,
            if q == 0 { 0 } else { smallest_prime_factor(q) },
        )
    }

    /// Create a rank r Drinfeld module
    ///
    /// # Arguments
    ///
    /// * `q` - The size of the finite field
    /// * `r` - The rank
    ///
    /// # Returns
    ///
    /// A rank r Drinfeld module
    pub fn rank_r_module<F: Field, R: Ring>(q: usize, r: usize) -> DrinfeldModule<F, R> {
        DrinfeldModule::with_characteristic(
            format!("F{}[T]", q),
            format!("F{}", q),
            r,
            if q == 0 { 0 } else { smallest_prime_factor(q) },
        )
    }
}

/// Helper function to find smallest prime factor
fn smallest_prime_factor(n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    for i in 2..=((n as f64).sqrt() as usize + 1) {
        if n % i == 0 {
            return i;
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;
    use rustmath_integers::Integer;

    #[test]
    fn test_drinfeld_module_creation() {
        let module: DrinfeldModule<Rational, Integer> =
            DrinfeldModule::new("Fq[T]".to_string(), "Fq".to_string(), 1);

        assert_eq!(module.rank(), 1);
        assert_eq!(module.base_ring(), "Fq[T]");
        assert_eq!(module.coefficient_field(), "Fq");
        assert!(module.is_well_defined());
    }

    #[test]
    fn test_drinfeld_module_rank2() {
        let module: DrinfeldModule<Rational, Integer> =
            DrinfeldModule::new("Fq[T]".to_string(), "Fq2".to_string(), 2);

        assert_eq!(module.rank(), 2);
        assert_eq!(module.height(), 2);
    }

    #[test]
    #[should_panic(expected = "Rank must be positive")]
    fn test_invalid_rank() {
        let _module: DrinfeldModule<Rational, Integer> =
            DrinfeldModule::new("Fq[T]".to_string(), "Fq".to_string(), 0);
    }

    #[test]
    fn test_drinfeld_module_with_characteristic() {
        let module: DrinfeldModule<Rational, Integer> =
            DrinfeldModule::with_characteristic("Fq[T]".to_string(), "Fq".to_string(), 1, 2);

        assert_eq!(module.characteristic(), 2);
        assert_eq!(module.rank(), 1);
    }

    #[test]
    fn test_coefficients() {
        let mut module: DrinfeldModule<Rational, Integer> =
            DrinfeldModule::new("Fq[T]".to_string(), "Fq".to_string(), 2);

        module.set_coefficient(0, "1".to_string());
        module.set_coefficient(1, "g1".to_string());
        module.set_coefficient(2, "g2".to_string());

        assert_eq!(module.get_coefficient(0), Some(&"1".to_string()));
        assert_eq!(module.get_coefficient(1), Some(&"g1".to_string()));
        assert_eq!(module.get_coefficient(2), Some(&"g2".to_string()));
    }

    #[test]
    fn test_separability() {
        let module: DrinfeldModule<Rational, Integer> =
            DrinfeldModule::new("Fq[T]".to_string(), "Fq".to_string(), 1);

        assert!(module.is_separable());
    }

    #[test]
    fn test_action() {
        let module: DrinfeldModule<Rational, Integer> =
            DrinfeldModule::new("Fq[T]".to_string(), "Fq".to_string(), 1);

        let result = module.action("T");
        assert!(result.contains("φ"));
        assert!(result.contains("T"));
    }

    #[test]
    fn test_carlitz_module() {
        let carlitz: DrinfeldModule<Rational, Integer> = DrinfeldModuleFactory::carlitz_module(2);

        assert_eq!(carlitz.rank(), 1);
        assert_eq!(carlitz.characteristic(), 2);
        assert!(carlitz.base_ring().contains("F2"));
    }

    #[test]
    fn test_rank_r_module() {
        let module: DrinfeldModule<Rational, Integer> = DrinfeldModuleFactory::rank_r_module(4, 3);

        assert_eq!(module.rank(), 3);
        assert_eq!(module.characteristic(), 2);
    }

    #[test]
    fn test_display() {
        let module: DrinfeldModule<Rational, Integer> =
            DrinfeldModule::new("Fq[T]".to_string(), "Fq".to_string(), 1);

        let display = format!("{}", module);
        assert!(display.contains("DrinfeldModule"));
        assert!(display.contains("Fq[T]"));
        assert!(display.contains("rank=1"));
    }

    #[test]
    fn test_clone() {
        let module1: DrinfeldModule<Rational, Integer> =
            DrinfeldModule::new("Fq[T]".to_string(), "Fq".to_string(), 1);
        let module2 = module1.clone();

        assert_eq!(module1.rank(), module2.rank());
        assert_eq!(module1.base_ring(), module2.base_ring());
    }

    #[test]
    fn test_smallest_prime_factor() {
        assert_eq!(smallest_prime_factor(2), 2);
        assert_eq!(smallest_prime_factor(4), 2);
        assert_eq!(smallest_prime_factor(9), 3);
        assert_eq!(smallest_prime_factor(15), 3);
        assert_eq!(smallest_prime_factor(17), 17);
    }
}
