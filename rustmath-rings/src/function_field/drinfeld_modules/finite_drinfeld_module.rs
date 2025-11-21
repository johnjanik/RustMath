//! Finite Drinfeld modules
//!
//! This module provides Drinfeld module structures over finite fields,
//! corresponding to SageMath's `sage.rings.function_field.drinfeld_modules.finite_drinfeld_module`.
//!
//! # Mathematical Background
//!
//! A finite Drinfeld module is a Drinfeld module φ: A → K{τ} where K is a
//! finite field extension of Fq. These modules have special properties:
//!
//! - The Frobenius endomorphism τ is periodic
//! - Point counting algorithms exist (analogous to counting points on elliptic curves)
//! - The j-invariant classifies isomorphism classes
//! - Torsion points form finite groups
//!
//! # Key Properties
//!
//! For a rank r Drinfeld module over Fq, the number of torsion points of
//! degree d is approximately q^(rd), analogous to the Hasse bound for
//! elliptic curves.
//!
//! # Key Types
//!
//! - `DrinfeldModule_finite`: Drinfeld module over a finite field
//!
//! # Examples
//!
//! ```ignore
//! use rustmath_rings::function_field::drinfeld_modules::finite_drinfeld_module::*;
//!
//! // Create a finite Drinfeld module over F4
//! let module = DrinfeldModule_finite::new("F4".to_string(), 1);
//!
//! // Count points
//! let point_count = module.point_count();
//! ```

use rustmath_core::{Ring, Field};
use std::marker::PhantomData;
use std::fmt;
use super::drinfeld_module::DrinfeldModule;

/// Drinfeld module over a finite field
///
/// Represents a Drinfeld module φ: A → K{τ} where K is finite.
/// This corresponds to SageMath's `DrinfeldModule_finite` class.
///
/// # Type Parameters
///
/// - `F`: The finite base field type
/// - `R`: The base ring type
///
/// # Mathematical Details
///
/// Key invariants for finite Drinfeld modules:
/// - **q**: Size of the base finite field
/// - **r**: Rank of the module
/// - **j-invariant**: Classifies isomorphism classes
/// - **Endomorphism ring**: Determines isogeny class
#[derive(Clone, Debug)]
pub struct DrinfeldModuleFinite<F: Field, R: Ring> {
    /// The underlying Drinfeld module
    base_module: DrinfeldModule<F, R>,
    /// Size of the finite field
    field_size: usize,
    /// j-invariant (for classification)
    j_invariant: Option<String>,
}

impl<F: Field, R: Ring> DrinfeldModule_finite<F, R> {
    /// Create a new finite Drinfeld module
    ///
    /// # Arguments
    ///
    /// * `field_name` - Name of the finite field (e.g., "F4", "F9")
    /// * `rank` - Rank of the module
    ///
    /// # Returns
    ///
    /// A new DrinfeldModule_finite instance
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let module = DrinfeldModule_finite::new("F4".to_string(), 1);
    /// assert_eq!(module.rank(), 1);
    /// assert_eq!(module.field_size(), 4);
    /// ```
    pub fn new(field_name: String, rank: usize) -> Self {
        // Extract field size from name (simplified parsing)
        let field_size = parse_field_size(&field_name);

        DrinfeldModule_finite {
            base_module: DrinfeldModule::with_characteristic(
                format!("{}[T]", field_name),
                field_name.clone(),
                rank,
                smallest_prime_factor(field_size),
            ),
            field_size,
            j_invariant: None,
        }
    }

    /// Create with explicit field size
    ///
    /// # Arguments
    ///
    /// * `field_name` - Name of the finite field
    /// * `field_size` - Size of the finite field (must be a prime power)
    /// * `rank` - Rank of the module
    ///
    /// # Returns
    ///
    /// A new DrinfeldModule_finite instance
    pub fn with_field_size(field_name: String, field_size: usize, rank: usize) -> Self {
        assert!(is_prime_power(field_size), "Field size must be a prime power");

        DrinfeldModule_finite {
            base_module: DrinfeldModule::with_characteristic(
                format!("{}[T]", field_name),
                field_name,
                rank,
                smallest_prime_factor(field_size),
            ),
            field_size,
            j_invariant: None,
        }
    }

    /// Get the rank
    ///
    /// # Returns
    ///
    /// The rank of the Drinfeld module
    pub fn rank(&self) -> usize {
        self.base_module.rank()
    }

    /// Get the field size
    ///
    /// # Returns
    ///
    /// The size of the base finite field
    pub fn field_size(&self) -> usize {
        self.field_size
    }

    /// Get the base field characteristic
    ///
    /// # Returns
    ///
    /// The characteristic of the base field
    pub fn characteristic(&self) -> usize {
        self.base_module.characteristic()
    }

    /// Compute approximate point count for degree d torsion
    ///
    /// # Arguments
    ///
    /// * `degree` - Degree of the torsion subgroup
    ///
    /// # Returns
    ///
    /// Approximate number of d-torsion points
    ///
    /// # Mathematical Details
    ///
    /// For a rank r Drinfeld module, the d-torsion has approximately
    /// q^(r·deg(d)) points, where deg(d) is the degree of d.
    pub fn point_count(&self, degree: usize) -> usize {
        // Approximate: q^(rank * degree)
        self.field_size.pow(self.rank() as u32 * degree as u32)
    }

    /// Set the j-invariant
    ///
    /// # Arguments
    ///
    /// * `j_inv` - String representation of the j-invariant
    pub fn set_j_invariant(&mut self, j_inv: String) {
        self.j_invariant = Some(j_inv);
    }

    /// Get the j-invariant
    ///
    /// # Returns
    ///
    /// The j-invariant if set
    pub fn j_invariant(&self) -> Option<&String> {
        self.j_invariant.as_ref()
    }

    /// Check if the module has CM (complex multiplication analogue)
    ///
    /// # Returns
    ///
    /// True if the endomorphism ring is larger than expected
    pub fn has_cm(&self) -> bool {
        // Simplified: would need to compute endomorphism ring
        false
    }

    /// Check if this is a supersingular module
    ///
    /// # Returns
    ///
    /// True if the module is supersingular (analogue to elliptic curves)
    pub fn is_supersingular(&self) -> bool {
        // Simplified: would need specific criteria
        false
    }

    /// Check if this is an ordinary module
    ///
    /// # Returns
    ///
    /// True if the module is ordinary (not supersingular)
    pub fn is_ordinary(&self) -> bool {
        !self.is_supersingular()
    }

    /// Get the Frobenius period
    ///
    /// # Returns
    ///
    /// The order of the Frobenius endomorphism
    pub fn frobenius_period(&self) -> usize {
        // Simplified: related to field extension degree
        1
    }
}

impl<F: Field, R: Ring> fmt::Display for DrinfeldModule_finite<F, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DrinfeldModule_finite(field_size={}, rank={})",
            self.field_size,
            self.rank()
        )
    }
}

/// Parse field size from field name
fn parse_field_size(field_name: &str) -> usize {
    // Simple parser: "F4" -> 4, "F9" -> 9, etc.
    if field_name.starts_with('F') || field_name.starts_with('f') {
        field_name[1..].parse().unwrap_or(2)
    } else {
        2
    }
}

/// Check if n is a prime power
fn is_prime_power(n: usize) -> bool {
    if n <= 1 {
        return false;
    }
    let p = smallest_prime_factor(n);
    let mut power = p;
    while power < n {
        power *= p;
    }
    power == n
}

/// Find smallest prime factor
fn smallest_prime_factor(n: usize) -> usize {
    if n <= 1 {
        return 2;
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
    fn test_finite_module_creation() {
        let module: DrinfeldModule_finite<Rational, Integer> =
            DrinfeldModule_finite::new("F4".to_string(), 1);

        assert_eq!(module.rank(), 1);
        assert_eq!(module.field_size(), 4);
        assert_eq!(module.characteristic(), 2);
    }

    #[test]
    fn test_finite_module_f9() {
        let module: DrinfeldModule_finite<Rational, Integer> =
            DrinfeldModule_finite::new("F9".to_string(), 2);

        assert_eq!(module.rank(), 2);
        assert_eq!(module.field_size(), 9);
        assert_eq!(module.characteristic(), 3);
    }

    #[test]
    fn test_with_field_size() {
        let module: DrinfeldModule_finite<Rational, Integer> =
            DrinfeldModule_finite::with_field_size("GF(8)".to_string(), 8, 1);

        assert_eq!(module.field_size(), 8);
        assert_eq!(module.characteristic(), 2);
    }

    #[test]
    #[should_panic(expected = "Field size must be a prime power")]
    fn test_invalid_field_size() {
        let _module: DrinfeldModule_finite<Rational, Integer> =
            DrinfeldModule_finite::with_field_size("F6".to_string(), 6, 1);
    }

    #[test]
    fn test_point_count() {
        let module: DrinfeldModule_finite<Rational, Integer> =
            DrinfeldModule_finite::new("F4".to_string(), 1);

        assert_eq!(module.point_count(1), 4); // q^(r*1) = 4^1 = 4
        assert_eq!(module.point_count(2), 16); // q^(r*2) = 4^2 = 16
    }

    #[test]
    fn test_point_count_rank2() {
        let module: DrinfeldModule_finite<Rational, Integer> =
            DrinfeldModule_finite::new("F2".to_string(), 2);

        assert_eq!(module.point_count(1), 4); // q^(r*1) = 2^2 = 4
        assert_eq!(module.point_count(2), 16); // q^(r*2) = 2^4 = 16
    }

    #[test]
    fn test_j_invariant() {
        let mut module: DrinfeldModule_finite<Rational, Integer> =
            DrinfeldModule_finite::new("F4".to_string(), 1);

        assert_eq!(module.j_invariant(), None);

        module.set_j_invariant("j = 1728".to_string());
        assert_eq!(module.j_invariant(), Some(&"j = 1728".to_string()));
    }

    #[test]
    fn test_ordinary() {
        let module: DrinfeldModule_finite<Rational, Integer> =
            DrinfeldModule_finite::new("F4".to_string(), 1);

        assert!(module.is_ordinary());
        assert!(!module.has_cm());
    }

    #[test]
    fn test_frobenius_period() {
        let module: DrinfeldModule_finite<Rational, Integer> =
            DrinfeldModule_finite::new("F4".to_string(), 1);

        assert_eq!(module.frobenius_period(), 1);
    }

    #[test]
    fn test_parse_field_size() {
        assert_eq!(parse_field_size("F2"), 2);
        assert_eq!(parse_field_size("F4"), 4);
        assert_eq!(parse_field_size("F16"), 16);
        assert_eq!(parse_field_size("f9"), 9);
    }

    #[test]
    fn test_is_prime_power() {
        assert!(is_prime_power(2));
        assert!(is_prime_power(3));
        assert!(is_prime_power(4));
        assert!(is_prime_power(8));
        assert!(is_prime_power(9));
        assert!(!is_prime_power(6));
        assert!(!is_prime_power(10));
        assert!(!is_prime_power(12));
    }

    #[test]
    fn test_display() {
        let module: DrinfeldModule_finite<Rational, Integer> =
            DrinfeldModule_finite::new("F4".to_string(), 1);

        let display = format!("{}", module);
        assert!(display.contains("DrinfeldModule_finite"));
        assert!(display.contains("field_size=4"));
        assert!(display.contains("rank=1"));
    }

    #[test]
    fn test_clone() {
        let module1: DrinfeldModule_finite<Rational, Integer> =
            DrinfeldModule_finite::new("F4".to_string(), 1);
        let module2 = module1.clone();

        assert_eq!(module1.rank(), module2.rank());
        assert_eq!(module1.field_size(), module2.field_size());
    }
}
