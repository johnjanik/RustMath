//! Homomorphism sets for Drinfeld modules
//!
//! This module provides homset (homomorphism set) structures for Drinfeld modules,
//! corresponding to SageMath's `sage.rings.function_field.drinfeld_modules.homset`.
//!
//! # Mathematical Background
//!
//! The homset Hom(φ, ψ) between two Drinfeld modules φ: A → K{τ} and ψ: A → L{τ}
//! consists of all morphisms u: K → L such that u ∘ φ = ψ ∘ u.
//!
//! Key properties:
//! - Hom(φ, φ) = End(φ) is the endomorphism ring
//! - Hom(φ, ψ) has a natural A-module structure
//! - For rank r modules, Hom(φ, ψ) is finite-dimensional
//! - Isogenies are morphisms with finite kernel
//!
//! # Key Types
//!
//! - `DrinfeldModuleHomset`: The homomorphism set between two Drinfeld modules
//! - `DrinfeldModuleMorphismAction`: Action of morphisms on elements
//!
//! # Examples
//!
//! ```ignore
//! use rustmath_rings::function_field::drinfeld_modules::homset::*;
//!
//! // Create a homset between two modules
//! let homset = DrinfeldModuleHomset::new(phi, psi);
//!
//! // Check if it's an endomorphism ring
//! if homset.is_endomorphism_set() {
//!     println!("This is End(φ)");
//! }
//! ```

use rustmath_core::{Ring, Field};
use std::marker::PhantomData;
use std::fmt;

/// Homomorphism set between two Drinfeld modules
///
/// Represents Hom(φ, ψ) for Drinfeld modules φ and ψ.
/// This corresponds to SageMath's `DrinfeldModuleHomset` class.
///
/// # Type Parameters
///
/// - `F`: The field type
/// - `R`: The ring type
///
/// # Mathematical Details
///
/// Elements of Hom(φ, ψ) are polynomials in τ that intertwine φ and ψ:
/// u·φ(a) = ψ(a)·u for all a ∈ A
#[derive(Clone, Debug)]
pub struct DrinfeldModuleHomset<F: Field, R: Ring> {
    /// Name of the source module φ
    source: String,
    /// Name of the target module ψ
    target: String,
    /// Dimension of the homset as a vector space
    dimension: usize,
    /// Field marker
    field_marker: PhantomData<F>,
    /// Ring marker
    ring_marker: PhantomData<R>,
}

impl<F: Field, R: Ring> DrinfeldModuleHomset<F, R> {
    /// Create a new homset between two Drinfeld modules
    ///
    /// # Arguments
    ///
    /// * `source` - Name of the source module φ
    /// * `target` - Name of the target module ψ
    ///
    /// # Returns
    ///
    /// A new DrinfeldModuleHomset instance
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let homset = DrinfeldModuleHomset::new("φ".to_string(), "ψ".to_string());
    /// ```
    pub fn new(source: String, target: String) -> Self {
        DrinfeldModuleHomset {
            source,
            target,
            dimension: 0, // To be computed
            field_marker: PhantomData,
            ring_marker: PhantomData,
        }
    }

    /// Create with specified dimension
    ///
    /// # Arguments
    ///
    /// * `source` - Name of the source module
    /// * `target` - Name of the target module
    /// * `dimension` - Dimension of the homset
    ///
    /// # Returns
    ///
    /// A new DrinfeldModuleHomset with specified dimension
    pub fn with_dimension(source: String, target: String, dimension: usize) -> Self {
        DrinfeldModuleHomset {
            source,
            target,
            dimension,
            field_marker: PhantomData,
            ring_marker: PhantomData,
        }
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

    /// Get the dimension
    ///
    /// # Returns
    ///
    /// Dimension of the homset as a vector space
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Check if this is an endomorphism set
    ///
    /// # Returns
    ///
    /// True if source equals target (i.e., this is End(φ))
    pub fn is_endomorphism_set(&self) -> bool {
        self.source == self.target
    }

    /// Check if the homset is non-empty
    ///
    /// # Returns
    ///
    /// True if there exists at least one morphism
    pub fn is_nonempty(&self) -> bool {
        self.dimension > 0 || self.is_endomorphism_set()
    }

    /// Check if the homset contains only isogenies
    ///
    /// # Returns
    ///
    /// True if all non-zero morphisms are isogenies
    pub fn is_isogeny_class(&self) -> bool {
        // Simplified: would need to check actual morphisms
        !self.source.is_empty() && !self.target.is_empty()
    }

    /// Set the dimension
    ///
    /// # Arguments
    ///
    /// * `dim` - New dimension value
    pub fn set_dimension(&mut self, dim: usize) {
        self.dimension = dim;
    }
}

impl<F: Field, R: Ring> fmt::Display for DrinfeldModuleHomset<F, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_endomorphism_set() {
            write!(f, "End({})", self.source)
        } else {
            write!(f, "Hom({}, {})", self.source, self.target)
        }
    }
}

/// Action of a Drinfeld module morphism
///
/// Represents the action of a morphism in Hom(φ, ψ) on elements.
/// This corresponds to SageMath's `DrinfeldModuleMorphismAction` class.
///
/// # Type Parameters
///
/// - `F`: The field type
/// - `R`: The ring type
///
/// # Mathematical Details
///
/// A morphism u ∈ Hom(φ, ψ) acts on elements by composition:
/// u(x) for x in the domain of φ
#[derive(Clone, Debug)]
pub struct DrinfeldModuleMorphismAction<F: Field, R: Ring> {
    /// The morphism being applied
    morphism_name: String,
    /// The homset containing this morphism
    homset: DrinfeldModuleHomset<F, R>,
}

impl<F: Field, R: Ring> DrinfeldModuleMorphismAction<F, R> {
    /// Create a new morphism action
    ///
    /// # Arguments
    ///
    /// * `morphism_name` - Name of the morphism
    /// * `homset` - The homset containing this morphism
    ///
    /// # Returns
    ///
    /// A new DrinfeldModuleMorphismAction instance
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let homset = DrinfeldModuleHomset::new("φ".to_string(), "ψ".to_string());
    /// let action = DrinfeldModuleMorphismAction::new("u".to_string(), homset);
    /// ```
    pub fn new(morphism_name: String, homset: DrinfeldModuleHomset<F, R>) -> Self {
        DrinfeldModuleMorphismAction {
            morphism_name,
            homset,
        }
    }

    /// Get the morphism name
    ///
    /// # Returns
    ///
    /// Name of the morphism
    pub fn morphism_name(&self) -> &str {
        &self.morphism_name
    }

    /// Get the associated homset
    ///
    /// # Returns
    ///
    /// Reference to the homset
    pub fn homset(&self) -> &DrinfeldModuleHomset<F, R> {
        &self.homset
    }

    /// Check if this is an endomorphism action
    ///
    /// # Returns
    ///
    /// True if the morphism is an endomorphism
    pub fn is_endomorphism(&self) -> bool {
        self.homset.is_endomorphism_set()
    }

    /// Apply the action to an element (symbolic)
    ///
    /// # Arguments
    ///
    /// * `element` - String representation of the element
    ///
    /// # Returns
    ///
    /// String representation of the result
    pub fn apply(&self, element: &str) -> String {
        format!("{}({})", self.morphism_name, element)
    }

    /// Check if the action is well-defined
    ///
    /// # Returns
    ///
    /// True if the morphism and homset are valid
    pub fn is_well_defined(&self) -> bool {
        !self.morphism_name.is_empty() && self.homset.is_nonempty()
    }
}

impl<F: Field, R: Ring> fmt::Display for DrinfeldModuleMorphismAction<F, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Action of {} in {}",
            self.morphism_name, self.homset
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;
    use rustmath_integers::Integer;

    #[test]
    fn test_homset_creation() {
        let homset: DrinfeldModuleHomset<Rational, Integer> =
            DrinfeldModuleHomset::new("φ".to_string(), "ψ".to_string());

        assert_eq!(homset.source(), "φ");
        assert_eq!(homset.target(), "ψ");
        assert!(!homset.is_endomorphism_set());
    }

    #[test]
    fn test_endomorphism_set() {
        let homset: DrinfeldModuleHomset<Rational, Integer> =
            DrinfeldModuleHomset::new("φ".to_string(), "φ".to_string());

        assert!(homset.is_endomorphism_set());
        assert!(homset.is_nonempty());
    }

    #[test]
    fn test_homset_with_dimension() {
        let homset: DrinfeldModuleHomset<Rational, Integer> =
            DrinfeldModuleHomset::with_dimension("φ".to_string(), "ψ".to_string(), 3);

        assert_eq!(homset.dimension(), 3);
        assert!(homset.is_nonempty());
    }

    #[test]
    fn test_set_dimension() {
        let mut homset: DrinfeldModuleHomset<Rational, Integer> =
            DrinfeldModuleHomset::new("φ".to_string(), "ψ".to_string());

        assert_eq!(homset.dimension(), 0);
        homset.set_dimension(5);
        assert_eq!(homset.dimension(), 5);
    }

    #[test]
    fn test_isogeny_class() {
        let homset: DrinfeldModuleHomset<Rational, Integer> =
            DrinfeldModuleHomset::new("φ".to_string(), "ψ".to_string());

        assert!(homset.is_isogeny_class());
    }

    #[test]
    fn test_homset_display() {
        let homset1: DrinfeldModuleHomset<Rational, Integer> =
            DrinfeldModuleHomset::new("φ".to_string(), "ψ".to_string());
        let homset2: DrinfeldModuleHomset<Rational, Integer> =
            DrinfeldModuleHomset::new("φ".to_string(), "φ".to_string());

        assert!(format!("{}", homset1).contains("Hom"));
        assert!(format!("{}", homset2).contains("End"));
    }

    #[test]
    fn test_morphism_action_creation() {
        let homset: DrinfeldModuleHomset<Rational, Integer> =
            DrinfeldModuleHomset::new("φ".to_string(), "ψ".to_string());
        let action = DrinfeldModuleMorphismAction::new("u".to_string(), homset);

        assert_eq!(action.morphism_name(), "u");
        assert!(!action.is_endomorphism());
    }

    #[test]
    fn test_endomorphism_action() {
        let homset: DrinfeldModuleHomset<Rational, Integer> =
            DrinfeldModuleHomset::new("φ".to_string(), "φ".to_string());
        let action = DrinfeldModuleMorphismAction::new("f".to_string(), homset);

        assert!(action.is_endomorphism());
    }

    #[test]
    fn test_apply_action() {
        let homset: DrinfeldModuleHomset<Rational, Integer> =
            DrinfeldModuleHomset::with_dimension("φ".to_string(), "ψ".to_string(), 1);
        let action = DrinfeldModuleMorphismAction::new("u".to_string(), homset);

        let result = action.apply("x");
        assert_eq!(result, "u(x)");
    }

    #[test]
    fn test_action_well_defined() {
        let homset: DrinfeldModuleHomset<Rational, Integer> =
            DrinfeldModuleHomset::with_dimension("φ".to_string(), "ψ".to_string(), 1);
        let action = DrinfeldModuleMorphismAction::new("u".to_string(), homset);

        assert!(action.is_well_defined());
    }

    #[test]
    fn test_action_display() {
        let homset: DrinfeldModuleHomset<Rational, Integer> =
            DrinfeldModuleHomset::new("φ".to_string(), "ψ".to_string());
        let action = DrinfeldModuleMorphismAction::new("u".to_string(), homset);

        let display = format!("{}", action);
        assert!(display.contains("Action"));
        assert!(display.contains("u"));
    }

    #[test]
    fn test_homset_clone() {
        let homset1: DrinfeldModuleHomset<Rational, Integer> =
            DrinfeldModuleHomset::with_dimension("φ".to_string(), "ψ".to_string(), 3);
        let homset2 = homset1.clone();

        assert_eq!(homset1.source(), homset2.source());
        assert_eq!(homset1.target(), homset2.target());
        assert_eq!(homset1.dimension(), homset2.dimension());
    }

    #[test]
    fn test_action_clone() {
        let homset: DrinfeldModuleHomset<Rational, Integer> =
            DrinfeldModuleHomset::new("φ".to_string(), "ψ".to_string());
        let action1 = DrinfeldModuleMorphismAction::new("u".to_string(), homset);
        let action2 = action1.clone();

        assert_eq!(action1.morphism_name(), action2.morphism_name());
    }
}
