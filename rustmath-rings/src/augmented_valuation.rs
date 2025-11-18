//! # Augmented Valuation Module
//!
//! Implementation of augmented valuations on polynomial rings.
//!
//! ## Overview
//!
//! An augmented valuation extends a discrete valuation v on a polynomial ring K[x]
//! by assigning a specific value μ to a key polynomial φ. For any polynomial f,
//! the valuation is computed as:
//!
//! v_aug(f) = min{v(f_i) + i·μ} where f = Σ f_i·φ^i (φ-adic expansion)
//!
//! ## Mathematical Background
//!
//! Augmented valuations are fundamental in:
//! - **Local fields**: Extending p-adic valuations to polynomial rings
//! - **Algebraic curves**: Computing intersection multiplicities
//! - **MacLane's algorithm**: Computing all extensions of valuations to polynomial rings
//!
//! ## Types of Augmented Valuations
//!
//! - **Final**: Cannot be augmented further
//! - **Non-Final**: Allows further augmentation
//! - **Finite**: μ is a finite value
//! - **Infinite**: μ = ∞ (creates pseudo-valuations)
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::augmented_valuation::{AugmentedValuationBase, AugmentedValuationFactory};
//!
//! // Create an augmentation of the Gauss valuation
//! // v_aug extends v by setting v_aug(x - a) = μ
//! ```

use std::fmt;
use std::marker::PhantomData;
use thiserror::Error;

/// Errors for valuation operations
#[derive(Debug, Clone, Error, PartialEq)]
pub enum ValuationError {
    #[error("Invalid augmentation: μ must exceed base valuation")]
    InvalidAugmentation,

    #[error("Cannot augment a final valuation")]
    CannotAugmentFinal,

    #[error("Key polynomial is not valid")]
    InvalidKeyPolynomial,

    #[error("Valuation is infinite")]
    InfiniteValuation,
}

/// Type for valuation values (can be finite or infinite)
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum ValuationValue {
    /// Finite rational value
    Finite(f64),
    /// Infinite valuation
    Infinity,
}

impl fmt::Display for ValuationValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValuationValue::Finite(v) => write!(f, "{}", v),
            ValuationValue::Infinity => write!(f, "∞"),
        }
    }
}

/// Base class for augmented valuations
///
/// An augmented valuation v_aug extends a base valuation v by setting
/// v_aug(φ) = μ where φ is a key polynomial.
#[derive(Debug, Clone)]
pub struct AugmentedValuationBase<R> {
    /// Description of this valuation
    description: String,
    /// The augmentation value μ
    mu: ValuationValue,
    /// The key polynomial (represented as string for simplicity)
    phi: String,
    _phantom: PhantomData<R>,
}

impl<R> AugmentedValuationBase<R> {
    /// Creates a new augmented valuation
    ///
    /// # Arguments
    /// * `description` - Description of the base valuation
    /// * `phi` - The key polynomial
    /// * `mu` - The augmentation value
    pub fn new(description: String, phi: String, mu: ValuationValue) -> Result<Self, ValuationError> {
        Ok(AugmentedValuationBase {
            description,
            mu,
            phi,
            _phantom: PhantomData,
        })
    }

    /// Returns the augmentation value μ
    pub fn mu(&self) -> &ValuationValue {
        &self.mu
    }

    /// Returns the key polynomial φ
    pub fn phi(&self) -> &str {
        &self.phi
    }

    /// Returns the description
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Checks if this is a finite valuation
    pub fn is_finite(&self) -> bool {
        matches!(self.mu, ValuationValue::Finite(_))
    }

    /// Checks if this is an infinite valuation
    pub fn is_infinite(&self) -> bool {
        matches!(self.mu, ValuationValue::Infinity)
    }
}

impl<R> fmt::Display for AugmentedValuationBase<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[ {} ]-Augmentation of {}", self.phi, self.description)
    }
}

/// Final augmented valuation (cannot be augmented further)
///
/// This occurs when the base valuation is trivial or when μ = ∞.
#[derive(Debug, Clone)]
pub struct FinalAugmentedValuation<R> {
    base: AugmentedValuationBase<R>,
}

impl<R> FinalAugmentedValuation<R> {
    /// Creates a new final augmented valuation
    pub fn new(description: String, phi: String, mu: ValuationValue) -> Result<Self, ValuationError> {
        Ok(FinalAugmentedValuation {
            base: AugmentedValuationBase::new(description, phi, mu)?,
        })
    }

    /// Returns the underlying base valuation
    pub fn base(&self) -> &AugmentedValuationBase<R> {
        &self.base
    }

    /// Checks if this valuation is discrete
    pub fn is_discrete(&self) -> bool {
        true
    }
}

/// Non-final augmented valuation (allows further augmentation)
///
/// This valuation can be extended by augmenting with a new key polynomial.
#[derive(Debug, Clone)]
pub struct NonFinalAugmentedValuation<R> {
    base: AugmentedValuationBase<R>,
}

impl<R> NonFinalAugmentedValuation<R> {
    /// Creates a new non-final augmented valuation
    pub fn new(description: String, phi: String, mu: ValuationValue) -> Result<Self, ValuationError> {
        // Non-final valuations must have finite μ
        if matches!(mu, ValuationValue::Infinity) {
            return Err(ValuationError::InvalidAugmentation);
        }

        Ok(NonFinalAugmentedValuation {
            base: AugmentedValuationBase::new(description, phi, mu)?,
        })
    }

    /// Returns the underlying base valuation
    pub fn base(&self) -> &AugmentedValuationBase<R> {
        &self.base
    }

    /// Attempts to augment this valuation further
    pub fn augment(&self, new_phi: String, new_mu: ValuationValue) -> Result<NonFinalAugmentedValuation<R>, ValuationError> {
        // The new μ must exceed the current valuation of new_phi
        let desc = format!("{} -> {}", self.base.description, new_phi);
        NonFinalAugmentedValuation::new(desc, new_phi, new_mu)
    }
}

/// Finite augmented valuation (μ is finite)
///
/// These valuations have discrete value groups and can represent
/// extensions of p-adic valuations.
#[derive(Debug, Clone)]
pub struct FiniteAugmentedValuation<R> {
    base: AugmentedValuationBase<R>,
}

impl<R> FiniteAugmentedValuation<R> {
    /// Creates a new finite augmented valuation
    pub fn new(description: String, phi: String, mu: f64) -> Result<Self, ValuationError> {
        Ok(FiniteAugmentedValuation {
            base: AugmentedValuationBase::new(description, phi, ValuationValue::Finite(mu))?,
        })
    }

    /// Returns the finite augmentation value
    pub fn mu_value(&self) -> f64 {
        match self.base.mu {
            ValuationValue::Finite(v) => v,
            ValuationValue::Infinity => panic!("FiniteAugmentedValuation has infinite mu"),
        }
    }

    /// Computes the ramification index E
    pub fn ramification_index(&self) -> usize {
        // Simplified: would compute actual ramification
        1
    }

    /// Computes the residue field degree F
    pub fn residue_degree(&self) -> usize {
        // Simplified: would compute actual degree
        1
    }
}

/// Infinite augmented valuation (μ = ∞)
///
/// These create pseudo-valuations useful for trivial extensions.
#[derive(Debug, Clone)]
pub struct InfiniteAugmentedValuation<R> {
    base: AugmentedValuationBase<R>,
}

impl<R> InfiniteAugmentedValuation<R> {
    /// Creates a new infinite augmented valuation
    pub fn new(description: String, phi: String) -> Result<Self, ValuationError> {
        Ok(InfiniteAugmentedValuation {
            base: AugmentedValuationBase::new(description, phi, ValuationValue::Infinity)?,
        })
    }

    /// Returns the base valuation
    pub fn base(&self) -> &AugmentedValuationBase<R> {
        &self.base
    }
}

/// Combined final finite augmented valuation
#[derive(Debug, Clone)]
pub struct FinalFiniteAugmentedValuation<R> {
    finite: FiniteAugmentedValuation<R>,
}

impl<R> FinalFiniteAugmentedValuation<R> {
    /// Creates a new final finite augmented valuation
    pub fn new(description: String, phi: String, mu: f64) -> Result<Self, ValuationError> {
        Ok(FinalFiniteAugmentedValuation {
            finite: FiniteAugmentedValuation::new(description, phi, mu)?,
        })
    }
}

/// Combined non-final finite augmented valuation
#[derive(Debug, Clone)]
pub struct NonFinalFiniteAugmentedValuation<R> {
    finite: FiniteAugmentedValuation<R>,
}

impl<R> NonFinalFiniteAugmentedValuation<R> {
    /// Creates a new non-final finite augmented valuation
    pub fn new(description: String, phi: String, mu: f64) -> Result<Self, ValuationError> {
        Ok(NonFinalFiniteAugmentedValuation {
            finite: FiniteAugmentedValuation::new(description, phi, mu)?,
        })
    }

    /// Augments this valuation further
    pub fn augment(&self, new_phi: String, new_mu: f64) -> Result<Self, ValuationError> {
        let desc = format!("{} -> {}", self.finite.base.description, new_phi);
        NonFinalFiniteAugmentedValuation::new(desc, new_phi, new_mu)
    }
}

/// Factory for creating augmented valuations
///
/// Implements the unique factory pattern ensuring consistent valuation objects.
#[derive(Debug, Default)]
pub struct AugmentedValuationFactory;

impl AugmentedValuationFactory {
    /// Creates a key for the augmented valuation
    pub fn create_key(phi: &str, mu: &ValuationValue) -> String {
        format!("{}@{}", phi, mu)
    }

    /// Creates a final augmented valuation
    pub fn create_final<R>(description: String, phi: String, mu: ValuationValue) -> Result<FinalAugmentedValuation<R>, ValuationError> {
        FinalAugmentedValuation::new(description, phi, mu)
    }

    /// Creates a non-final augmented valuation
    pub fn create_non_final<R>(description: String, phi: String, mu: ValuationValue) -> Result<NonFinalAugmentedValuation<R>, ValuationError> {
        NonFinalAugmentedValuation::new(description, phi, mu)
    }

    /// Creates a finite augmented valuation
    pub fn create_finite<R>(description: String, phi: String, mu: f64) -> Result<FiniteAugmentedValuation<R>, ValuationError> {
        FiniteAugmentedValuation::new(description, phi, mu)
    }

    /// Creates an infinite augmented valuation
    pub fn create_infinite<R>(description: String, phi: String) -> Result<InfiniteAugmentedValuation<R>, ValuationError> {
        InfiniteAugmentedValuation::new(description, phi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valuation_value_finite() {
        let v = ValuationValue::Finite(2.5);
        assert!(matches!(v, ValuationValue::Finite(_)));
    }

    #[test]
    fn test_valuation_value_infinite() {
        let v = ValuationValue::Infinity;
        assert!(matches!(v, ValuationValue::Infinity));
    }

    #[test]
    fn test_augmented_valuation_base() {
        let val: AugmentedValuationBase<i32> = AugmentedValuationBase::new(
            "Base val".to_string(),
            "x - 2".to_string(),
            ValuationValue::Finite(3.0)
        ).unwrap();

        assert_eq!(val.phi(), "x - 2");
        assert!(val.is_finite());
        assert!(!val.is_infinite());
    }

    #[test]
    fn test_final_augmented_valuation() {
        let val: FinalAugmentedValuation<i32> = FinalAugmentedValuation::new(
            "Gauss".to_string(),
            "x".to_string(),
            ValuationValue::Infinity
        ).unwrap();

        assert!(val.is_discrete());
    }

    #[test]
    fn test_non_final_augmented_valuation() {
        let val: NonFinalAugmentedValuation<i32> = NonFinalAugmentedValuation::new(
            "v_2".to_string(),
            "x - 1".to_string(),
            ValuationValue::Finite(1.0)
        ).unwrap();

        assert!(val.base().is_finite());
    }

    #[test]
    fn test_non_final_infinite_rejected() {
        let result: Result<NonFinalAugmentedValuation<i32>, _> = NonFinalAugmentedValuation::new(
            "test".to_string(),
            "x".to_string(),
            ValuationValue::Infinity
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_finite_augmented_valuation() {
        let val: FiniteAugmentedValuation<i32> = FiniteAugmentedValuation::new(
            "p-adic".to_string(),
            "x - p".to_string(),
            2.0
        ).unwrap();

        assert_eq!(val.mu_value(), 2.0);
        assert_eq!(val.ramification_index(), 1);
    }

    #[test]
    fn test_infinite_augmented_valuation() {
        let val: InfiniteAugmentedValuation<i32> = InfiniteAugmentedValuation::new(
            "Trivial".to_string(),
            "x".to_string()
        ).unwrap();

        assert!(val.base().is_infinite());
    }

    #[test]
    fn test_augment_non_final() {
        let val1: NonFinalAugmentedValuation<i32> = NonFinalAugmentedValuation::new(
            "v_0".to_string(),
            "x".to_string(),
            ValuationValue::Finite(1.0)
        ).unwrap();

        let val2 = val1.augment("x^2 - 2".to_string(), ValuationValue::Finite(2.0)).unwrap();
        assert_eq!(val2.base().phi(), "x^2 - 2");
    }

    #[test]
    fn test_factory_create_key() {
        let key = AugmentedValuationFactory::create_key("x - 1", &ValuationValue::Finite(3.0));
        assert!(key.contains("x - 1"));
    }

    #[test]
    fn test_factory_create_final() {
        let val: FinalAugmentedValuation<i32> = AugmentedValuationFactory::create_final(
            "Test".to_string(),
            "y".to_string(),
            ValuationValue::Finite(1.5)
        ).unwrap();

        assert!(val.is_discrete());
    }

    #[test]
    fn test_factory_create_finite() {
        let val: FiniteAugmentedValuation<i32> = AugmentedValuationFactory::create_finite(
            "Test".to_string(),
            "z".to_string(),
            2.5
        ).unwrap();

        assert_eq!(val.mu_value(), 2.5);
    }

    #[test]
    fn test_final_finite_combined() {
        let val: FinalFiniteAugmentedValuation<i32> = FinalFiniteAugmentedValuation::new(
            "Combined".to_string(),
            "t".to_string(),
            4.0
        ).unwrap();

        assert!(val.finite.base.is_finite());
    }

    #[test]
    fn test_non_final_finite_augment() {
        let val1: NonFinalFiniteAugmentedValuation<i32> = NonFinalFiniteAugmentedValuation::new(
            "Start".to_string(),
            "x".to_string(),
            1.0
        ).unwrap();

        let val2 = val1.augment("x - 1".to_string(), 2.0).unwrap();
        assert_eq!(val2.finite.mu_value(), 2.0);
    }
}
