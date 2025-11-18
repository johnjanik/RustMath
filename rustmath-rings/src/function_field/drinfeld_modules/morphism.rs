//! Morphisms between Drinfeld modules
//!
//! This module provides morphism structures for Drinfeld modules, corresponding to
//! SageMath's `sage.rings.function_field.drinfeld_modules.morphism`.
//!
//! # Mathematical Background
//!
//! A morphism u: φ → ψ between Drinfeld modules φ: A → K{τ} and ψ: A → L{τ}
//! is an element u ∈ L{τ} such that:
//! u ∘ φ(a) = ψ(a) ∘ u for all a ∈ A
//!
//! Key types of morphisms:
//! - **Isomorphism**: Invertible morphism (u has degree 0)
//! - **Isogeny**: Non-zero morphism with finite kernel
//! - **Endomorphism**: Morphism from φ to itself
//! - **Frobenius**: Multiplication by τ
//!
//! # Key Properties
//!
//! - The degree of u equals its degree as a polynomial in τ
//! - Isogenies preserve rank
//! - The kernel of an isogeny is a finite A-module
//! - Composition of morphisms is well-defined
//!
//! # Key Types
//!
//! - `DrinfeldModuleMorphism`: A morphism between Drinfeld modules
//!
//! # Examples
//!
//! ```ignore
//! use rustmath_rings::function_field::drinfeld_modules::morphism::*;
//!
//! // Create a morphism
//! let morphism = DrinfeldModuleMorphism::new("φ".to_string(), "ψ".to_string(), 1);
//!
//! // Check if it's an isogeny
//! if morphism.is_isogeny() {
//!     println!("Degree: {}", morphism.degree());
//! }
//! ```

use rustmath_core::{Ring, Field};
use std::marker::PhantomData;
use std::fmt;

/// Morphism between two Drinfeld modules
///
/// Represents a morphism u: φ → ψ between Drinfeld modules.
/// This corresponds to SageMath's `DrinfeldModuleMorphism` class.
///
/// # Type Parameters
///
/// - `F`: The field type
/// - `R`: The ring type
///
/// # Mathematical Details
///
/// A morphism u ∈ L{τ} satisfies the intertwining property:
/// u·φ(a) = ψ(a)·u for all a ∈ A
///
/// The degree of u determines its kernel size:
/// |ker(u)| = q^d where d = deg(u)
#[derive(Clone, Debug)]
pub struct DrinfeldModuleMorphism<F: Field, R: Ring> {
    /// Name of the source module φ
    source: String,
    /// Name of the target module ψ
    target: String,
    /// Degree of the morphism as a polynomial in τ
    degree: usize,
    /// Coefficients of the morphism (in τ)
    coefficients: Vec<String>,
    /// Field marker
    field_marker: PhantomData<F>,
    /// Ring marker
    ring_marker: PhantomData<R>,
}

impl<F: Field, R: Ring> DrinfeldModuleMorphism<F, R> {
    /// Create a new Drinfeld module morphism
    ///
    /// # Arguments
    ///
    /// * `source` - Name of the source module φ
    /// * `target` - Name of the target module ψ
    /// * `degree` - Degree of the morphism
    ///
    /// # Returns
    ///
    /// A new DrinfeldModuleMorphism instance
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let morphism = DrinfeldModuleMorphism::new("φ".to_string(), "ψ".to_string(), 1);
    /// assert_eq!(morphism.degree(), 1);
    /// ```
    pub fn new(source: String, target: String, degree: usize) -> Self {
        DrinfeldModuleMorphism {
            source,
            target,
            degree,
            coefficients: vec!["0".to_string(); degree + 1],
            field_marker: PhantomData,
            ring_marker: PhantomData,
        }
    }

    /// Create the zero morphism
    ///
    /// # Arguments
    ///
    /// * `source` - Name of the source module
    /// * `target` - Name of the target module
    ///
    /// # Returns
    ///
    /// The zero morphism (degree 0, all coefficients zero)
    pub fn zero(source: String, target: String) -> Self {
        DrinfeldModuleMorphism {
            source,
            target,
            degree: 0,
            coefficients: vec!["0".to_string()],
            field_marker: PhantomData,
            ring_marker: PhantomData,
        }
    }

    /// Create the identity morphism
    ///
    /// # Arguments
    ///
    /// * `module` - Name of the module
    ///
    /// # Returns
    ///
    /// The identity morphism (degree 0, coefficient 1)
    pub fn identity(module: String) -> Self {
        DrinfeldModuleMorphism {
            source: module.clone(),
            target: module,
            degree: 0,
            coefficients: vec!["1".to_string()],
            field_marker: PhantomData,
            ring_marker: PhantomData,
        }
    }

    /// Get the degree of the morphism
    ///
    /// # Returns
    ///
    /// The degree as a polynomial in τ
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Get the source module name
    ///
    /// # Returns
    ///
    /// Name of the source module
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Get the target module name
    ///
    /// # Returns
    ///
    /// Name of the target module
    pub fn target(&self) -> &str {
        &self.target
    }

    /// Check if this is an endomorphism
    ///
    /// # Returns
    ///
    /// True if source equals target
    pub fn is_endomorphism(&self) -> bool {
        self.source == self.target
    }

    /// Check if this is an isogeny
    ///
    /// # Returns
    ///
    /// True if this is a non-zero morphism (potential isogeny)
    pub fn is_isogeny(&self) -> bool {
        self.degree > 0 || (self.degree == 0 && self.coefficients[0] != "0")
    }

    /// Check if this is an isomorphism
    ///
    /// # Returns
    ///
    /// True if degree is 0 and leading coefficient is invertible
    pub fn is_isomorphism(&self) -> bool {
        self.degree == 0 && self.coefficients[0] != "0"
    }

    /// Check if this is the zero morphism
    ///
    /// # Returns
    ///
    /// True if all coefficients are zero
    pub fn is_zero(&self) -> bool {
        self.coefficients.iter().all(|c| c == "0")
    }

    /// Check if this is the identity morphism
    ///
    /// # Returns
    ///
    /// True if this is the identity on a module
    pub fn is_identity(&self) -> bool {
        self.is_endomorphism()
            && self.degree == 0
            && self.coefficients.len() == 1
            && self.coefficients[0] == "1"
    }

    /// Set a coefficient
    ///
    /// # Arguments
    ///
    /// * `index` - Index of the coefficient
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

    /// Get the kernel size (for isogenies over finite fields)
    ///
    /// # Arguments
    ///
    /// * `q` - Size of the base finite field
    ///
    /// # Returns
    ///
    /// Size of the kernel: q^degree
    pub fn kernel_size(&self, q: usize) -> usize {
        if self.is_zero() {
            return usize::MAX; // Infinite kernel
        }
        q.pow(self.degree as u32)
    }

    /// Compose with another morphism
    ///
    /// # Arguments
    ///
    /// * `other` - The morphism to compose with
    ///
    /// # Returns
    ///
    /// A string representation of the composition
    pub fn compose(&self, other: &DrinfeldModuleMorphism<F, R>) -> String {
        format!("{} ∘ {}", self.source, other.target)
    }
}

impl<F: Field, R: Ring> fmt::Display for DrinfeldModuleMorphism<F, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_endomorphism() {
            write!(f, "Endomorphism of {} (degree {})", self.source, self.degree)
        } else {
            write!(
                f,
                "Morphism {} → {} (degree {})",
                self.source, self.target, self.degree
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;
    use rustmath_integers::Integer;

    #[test]
    fn test_morphism_creation() {
        let morphism: DrinfeldModuleMorphism<Rational, Integer> =
            DrinfeldModuleMorphism::new("φ".to_string(), "ψ".to_string(), 1);

        assert_eq!(morphism.degree(), 1);
        assert_eq!(morphism.source(), "φ");
        assert_eq!(morphism.target(), "ψ");
        assert!(!morphism.is_endomorphism());
    }

    #[test]
    fn test_zero_morphism() {
        let morphism: DrinfeldModuleMorphism<Rational, Integer> =
            DrinfeldModuleMorphism::zero("φ".to_string(), "ψ".to_string());

        assert!(morphism.is_zero());
        assert!(!morphism.is_isogeny());
        assert_eq!(morphism.degree(), 0);
    }

    #[test]
    fn test_identity_morphism() {
        let morphism: DrinfeldModuleMorphism<Rational, Integer> =
            DrinfeldModuleMorphism::identity("φ".to_string());

        assert!(morphism.is_identity());
        assert!(morphism.is_endomorphism());
        assert!(morphism.is_isomorphism());
        assert_eq!(morphism.degree(), 0);
    }

    #[test]
    fn test_endomorphism() {
        let morphism: DrinfeldModuleMorphism<Rational, Integer> =
            DrinfeldModuleMorphism::new("φ".to_string(), "φ".to_string(), 2);

        assert!(morphism.is_endomorphism());
        assert_eq!(morphism.source(), morphism.target());
    }

    #[test]
    fn test_isogeny() {
        let morphism: DrinfeldModuleMorphism<Rational, Integer> =
            DrinfeldModuleMorphism::new("φ".to_string(), "ψ".to_string(), 1);

        assert!(morphism.is_isogeny());
        assert!(!morphism.is_isomorphism());
    }

    #[test]
    fn test_isomorphism() {
        let mut morphism: DrinfeldModuleMorphism<Rational, Integer> =
            DrinfeldModuleMorphism::new("φ".to_string(), "ψ".to_string(), 0);

        morphism.set_coefficient(0, "1".to_string());
        assert!(morphism.is_isomorphism());
        assert!(morphism.is_isogeny());
    }

    #[test]
    fn test_coefficients() {
        let mut morphism: DrinfeldModuleMorphism<Rational, Integer> =
            DrinfeldModuleMorphism::new("φ".to_string(), "ψ".to_string(), 2);

        morphism.set_coefficient(0, "a0".to_string());
        morphism.set_coefficient(1, "a1".to_string());
        morphism.set_coefficient(2, "a2".to_string());

        assert_eq!(morphism.get_coefficient(0), Some(&"a0".to_string()));
        assert_eq!(morphism.get_coefficient(1), Some(&"a1".to_string()));
        assert_eq!(morphism.get_coefficient(2), Some(&"a2".to_string()));
    }

    #[test]
    fn test_kernel_size() {
        let morphism: DrinfeldModuleMorphism<Rational, Integer> =
            DrinfeldModuleMorphism::new("φ".to_string(), "ψ".to_string(), 2);

        assert_eq!(morphism.kernel_size(3), 9); // q^degree = 3^2 = 9
        assert_eq!(morphism.kernel_size(2), 4); // q^degree = 2^2 = 4
    }

    #[test]
    fn test_zero_kernel_size() {
        let morphism: DrinfeldModuleMorphism<Rational, Integer> =
            DrinfeldModuleMorphism::zero("φ".to_string(), "ψ".to_string());

        assert_eq!(morphism.kernel_size(3), usize::MAX);
    }

    #[test]
    fn test_compose() {
        let morphism1: DrinfeldModuleMorphism<Rational, Integer> =
            DrinfeldModuleMorphism::new("φ".to_string(), "ψ".to_string(), 1);
        let morphism2: DrinfeldModuleMorphism<Rational, Integer> =
            DrinfeldModuleMorphism::new("ψ".to_string(), "χ".to_string(), 1);

        let composition = morphism1.compose(&morphism2);
        assert!(composition.contains("φ"));
        assert!(composition.contains("χ"));
    }

    #[test]
    fn test_display_morphism() {
        let morphism: DrinfeldModuleMorphism<Rational, Integer> =
            DrinfeldModuleMorphism::new("φ".to_string(), "ψ".to_string(), 1);

        let display = format!("{}", morphism);
        assert!(display.contains("Morphism"));
        assert!(display.contains("φ"));
        assert!(display.contains("ψ"));
        assert!(display.contains("degree 1"));
    }

    #[test]
    fn test_display_endomorphism() {
        let morphism: DrinfeldModuleMorphism<Rational, Integer> =
            DrinfeldModuleMorphism::new("φ".to_string(), "φ".to_string(), 2);

        let display = format!("{}", morphism);
        assert!(display.contains("Endomorphism"));
        assert!(display.contains("φ"));
    }

    #[test]
    fn test_clone() {
        let morphism1: DrinfeldModuleMorphism<Rational, Integer> =
            DrinfeldModuleMorphism::new("φ".to_string(), "ψ".to_string(), 1);
        let morphism2 = morphism1.clone();

        assert_eq!(morphism1.degree(), morphism2.degree());
        assert_eq!(morphism1.source(), morphism2.source());
        assert_eq!(morphism1.target(), morphism2.target());
    }
}
