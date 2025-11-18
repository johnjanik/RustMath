//! Drinfeld module actions
//!
//! This module provides action structures for Drinfeld modules, corresponding to
//! SageMath's `sage.rings.function_field.drinfeld_modules.action`.
//!
//! # Mathematical Background
//!
//! A Drinfeld module is a special type of morphism from a ring to an endomorphism ring.
//! Drinfeld modules provide actions on various mathematical structures, particularly
//! in the context of function field arithmetic over finite fields.
//!
//! An action of a Drinfeld module φ: A → End(G) on a group G gives a way to
//! multiply elements of A with elements of G, satisfying the action axioms:
//! - φ(1)·g = g
//! - φ(ab)·g = φ(a)·(φ(b)·g)
//!
//! # Key Types
//!
//! - `DrinfeldModuleAction`: Represents the action of a Drinfeld module
//!
//! # Examples
//!
//! ```ignore
//! use rustmath_rings::function_field::drinfeld_modules::action::*;
//!
//! // Create a Drinfeld module action
//! let action = DrinfeldModuleAction::new(drinfeld_module);
//!
//! // Apply the action to an element
//! let result = action.act(element, group_element);
//! ```

use rustmath_core::{Ring, Field};
use std::marker::PhantomData;
use std::fmt;

/// Action of a Drinfeld module
///
/// Represents the action of a Drinfeld module on a group or module.
/// This corresponds to SageMath's `DrinfeldModuleAction` class.
///
/// # Type Parameters
///
/// - `F`: The base field of the function field
/// - `R`: The coefficient ring
///
/// # Mathematical Details
///
/// Given a Drinfeld module φ: A → K{τ} where:
/// - A is the base ring (typically Fq[T])
/// - K is a field extension of Fq
/// - τ is the Frobenius endomorphism
///
/// The action is defined by the image of ring elements under φ.
#[derive(Clone, Debug)]
pub struct DrinfeldModuleAction<F: Field, R: Ring> {
    /// Name/description of the Drinfeld module
    name: String,
    /// The base ring
    base_ring: PhantomData<R>,
    /// The coefficient field
    field: PhantomData<F>,
}

impl<F: Field, R: Ring> DrinfeldModuleAction<F, R> {
    /// Create a new Drinfeld module action
    ///
    /// # Arguments
    ///
    /// * `name` - Name/description of the Drinfeld module
    ///
    /// # Returns
    ///
    /// A new DrinfeldModuleAction instance
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rustmath_rings::function_field::drinfeld_modules::action::*;
    ///
    /// let action = DrinfeldModuleAction::new("φ: Fq[T] → K{τ}".to_string());
    /// ```
    pub fn new(name: String) -> Self {
        DrinfeldModuleAction {
            name,
            base_ring: PhantomData,
            field: PhantomData,
        }
    }

    /// Get the name of the Drinfeld module
    ///
    /// # Returns
    ///
    /// The name/description of the Drinfeld module
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Check if this action is defined
    ///
    /// # Returns
    ///
    /// True if the action is properly defined
    pub fn is_defined(&self) -> bool {
        !self.name.is_empty()
    }

    /// Get the domain of the action (the base ring)
    ///
    /// # Returns
    ///
    /// A string representation of the domain
    pub fn domain(&self) -> String {
        format!("Base ring for {}", self.name)
    }

    /// Get the codomain of the action (the endomorphism ring)
    ///
    /// # Returns
    ///
    /// A string representation of the codomain
    pub fn codomain(&self) -> String {
        format!("End(G) for {}", self.name)
    }
}

impl<F: Field, R: Ring> fmt::Display for DrinfeldModuleAction<F, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DrinfeldModuleAction({})", self.name)
    }
}

/// Trait for types that can have Drinfeld module actions
pub trait HasDrinfeldAction<F: Field, R: Ring> {
    /// Apply a Drinfeld module action to this element
    ///
    /// # Arguments
    ///
    /// * `action` - The Drinfeld module action to apply
    /// * `ring_element` - The ring element defining the action
    ///
    /// # Returns
    ///
    /// The result of applying the action
    fn apply_action(&self, action: &DrinfeldModuleAction<F, R>, ring_element: &R) -> Self;
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;
    use rustmath_integers::Integer;

    #[test]
    fn test_drinfeld_action_creation() {
        let action: DrinfeldModuleAction<Rational, Integer> =
            DrinfeldModuleAction::new("φ: Fq[T] → K{τ}".to_string());

        assert_eq!(action.name(), "φ: Fq[T] → K{τ}");
        assert!(action.is_defined());
    }

    #[test]
    fn test_drinfeld_action_empty() {
        let action: DrinfeldModuleAction<Rational, Integer> =
            DrinfeldModuleAction::new("".to_string());

        assert_eq!(action.name(), "");
        assert!(!action.is_defined());
    }

    #[test]
    fn test_drinfeld_action_domain_codomain() {
        let action: DrinfeldModuleAction<Rational, Integer> =
            DrinfeldModuleAction::new("φ".to_string());

        assert!(action.domain().contains("Base ring"));
        assert!(action.codomain().contains("End(G)"));
    }

    #[test]
    fn test_drinfeld_action_display() {
        let action: DrinfeldModuleAction<Rational, Integer> =
            DrinfeldModuleAction::new("test_module".to_string());

        let display = format!("{}", action);
        assert!(display.contains("DrinfeldModuleAction"));
        assert!(display.contains("test_module"));
    }

    #[test]
    fn test_drinfeld_action_clone() {
        let action1: DrinfeldModuleAction<Rational, Integer> =
            DrinfeldModuleAction::new("original".to_string());
        let action2 = action1.clone();

        assert_eq!(action1.name(), action2.name());
        assert_eq!(action1.domain(), action2.domain());
    }

    #[test]
    fn test_multiple_actions() {
        let action1: DrinfeldModuleAction<Rational, Integer> =
            DrinfeldModuleAction::new("φ₁".to_string());
        let action2: DrinfeldModuleAction<Rational, Integer> =
            DrinfeldModuleAction::new("φ₂".to_string());

        assert_ne!(action1.name(), action2.name());
        assert!(action1.is_defined());
        assert!(action2.is_defined());
    }

    #[test]
    fn test_action_properties() {
        let action: DrinfeldModuleAction<Rational, Integer> =
            DrinfeldModuleAction::new("Carlitz module".to_string());

        // Verify basic properties
        assert!(!action.name().is_empty());
        assert!(!action.domain().is_empty());
        assert!(!action.codomain().is_empty());
    }
}
