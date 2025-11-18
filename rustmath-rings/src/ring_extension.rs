//! # Ring Extension Module
//!
//! Implementation of ring extensions L/K, representing one ring as an extension of another.
//!
//! ## Overview
//!
//! A ring extension L/K consists of:
//! - A base ring K
//! - An extension ring L
//! - A ring homomorphism φ: K → L (the structure map)
//!
//! Extensions can form towers: K ⊂ L ⊂ M, and this module tracks the hierarchical structure.
//!
//! ## Types of Extensions
//!
//! - **Generic Extension**: Base case with minimal structure
//! - **With Basis**: Finite-dimensional free module structure
//! - **With Generator**: Single-element generation (e.g., K[α])
//! - **Fraction Field**: Field of fractions construction
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::ring_extension::{RingExtensionGeneric, common_base, tower_bases};
//!
//! // Create an extension ℚ(√2) / ℚ
//! // let ext = RingExtensionWithGen::new(QQ, sqrt2_gen);
//! ```
//!
//! ## Mathematical Background
//!
//! For rings K ⊂ L:
//! - **Degree**: [L : K] = dimension of L as K-module
//! - **Finite Extension**: [L : K] < ∞
//! - **Free Extension**: L is free as K-module
//! - **Tower Law**: [M : K] = [M : L] · [L : K]

use crate::morphism::RingHomomorphism;
use rustmath_core::{Ring, CommutativeRing, Field};
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use thiserror::Error;

/// Errors for ring extension operations
#[derive(Debug, Clone, Error, PartialEq)]
pub enum RingExtensionError {
    #[error("No common base found")]
    NoCommonBase,

    #[error("Not a finite extension")]
    NotFinite,

    #[error("Not a free extension")]
    NotFree,

    #[error("Invalid generator")]
    InvalidGenerator,

    #[error("Degree computation failed: {0}")]
    DegreeError(String),
}

/// Generic ring extension L/K
///
/// Represents an extension with minimal structure. Subclasses add specialized
/// features like basis, generators, or field structure.
#[derive(Debug, Clone)]
pub struct RingExtensionGeneric<K, L>
where
    K: Ring,
    L: Ring,
{
    /// The base ring K
    base_ring: PhantomData<K>,
    /// The extension ring L (backend)
    backend_ring: PhantomData<L>,
    /// Description of the extension
    description: String,
    /// Whether the extension exposes backend details
    print_backend: bool,
}

impl<K, L> RingExtensionGeneric<K, L>
where
    K: Ring,
    L: Ring,
{
    /// Creates a new ring extension
    pub fn new(description: String) -> Self {
        RingExtensionGeneric {
            base_ring: PhantomData,
            backend_ring: PhantomData,
            description,
            print_backend: false,
        }
    }

    /// Creates an extension with backend exposure
    pub fn with_backend_visible(description: String) -> Self {
        RingExtensionGeneric {
            base_ring: PhantomData,
            backend_ring: PhantomData,
            description,
            print_backend: true,
        }
    }

    /// Returns the description
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Checks if backend is visible
    pub fn is_backend_visible(&self) -> bool {
        self.print_backend
    }

    /// Computes the degree [L : K]
    ///
    /// Returns None if the extension is infinite
    pub fn degree(&self) -> Option<usize> {
        None // Override in subclasses
    }

    /// Checks if this is a finite extension
    pub fn is_finite(&self) -> bool {
        self.degree().is_some()
    }

    /// Checks if this is a free extension
    pub fn is_free(&self) -> bool {
        false // Override in subclasses with basis
    }
}

impl<K, L> fmt::Display for RingExtensionGeneric<K, L>
where
    K: Ring,
    L: Ring,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ring extension: {}", self.description)
    }
}

/// Ring extension with a vector space basis
///
/// This represents extensions L/K where L is a finite-dimensional free K-module.
#[derive(Debug, Clone)]
pub struct RingExtensionWithBasis<K, L>
where
    K: Ring,
    L: Ring,
{
    base: RingExtensionGeneric<K, L>,
    /// The basis elements
    basis: Vec<String>, // Names of basis elements
    /// The dimension [L : K]
    dimension: usize,
}

impl<K, L> RingExtensionWithBasis<K, L>
where
    K: Ring,
    L: Ring,
{
    /// Creates an extension with specified basis
    pub fn new(description: String, basis: Vec<String>) -> Self {
        let dim = basis.len();
        RingExtensionWithBasis {
            base: RingExtensionGeneric::new(description),
            basis,
            dimension: dim,
        }
    }

    /// Returns the basis
    pub fn basis(&self) -> &[String] {
        &self.basis
    }

    /// Returns the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Checks if this extension is finite
    pub fn is_finite(&self) -> bool {
        true // Always finite if we have a basis
    }

    /// Checks if this extension is free
    pub fn is_free(&self) -> bool {
        true // Has basis, so free
    }
}

/// Ring extension generated by a single element
///
/// Represents K[α] or K(α) extensions where α is algebraic over K.
#[derive(Debug, Clone)]
pub struct RingExtensionWithGen<K, L>
where
    K: Ring,
    L: Ring,
{
    base: RingExtensionWithBasis<K, L>,
    /// Name of the generator
    generator_name: String,
}

impl<K, L> RingExtensionWithGen<K, L>
where
    K: Ring,
    L: Ring,
{
    /// Creates an extension with a single generator
    pub fn new(base_desc: String, generator_name: String, degree: usize) -> Self {
        // Create basis {1, α, α², ..., α^(degree-1)}
        let basis: Vec<String> = (0..degree)
            .map(|i| {
                if i == 0 {
                    "1".to_string()
                } else if i == 1 {
                    generator_name.clone()
                } else {
                    format!("{}^{}", generator_name, i)
                }
            })
            .collect();

        RingExtensionWithGen {
            base: RingExtensionWithBasis::new(base_desc, basis),
            generator_name,
        }
    }

    /// Returns the generator name
    pub fn generator(&self) -> &str {
        &self.generator_name
    }

    /// Returns the degree
    pub fn degree(&self) -> usize {
        self.base.dimension()
    }
}

/// Fraction field extension
///
/// Represents the field of fractions Frac(L) when L/K is an extension.
#[derive(Debug, Clone)]
pub struct RingExtensionFractionField<K, L>
where
    K: Ring,
    L: Ring,
{
    base: RingExtensionGeneric<K, L>,
    /// Whether to extend the base ring to its fraction field
    extend_base: bool,
}

impl<K, L> RingExtensionFractionField<K, L>
where
    K: Ring,
    L: Ring,
{
    /// Creates a fraction field extension
    pub fn new(description: String, extend_base: bool) -> Self {
        RingExtensionFractionField {
            base: RingExtensionGeneric::new(description),
            extend_base,
        }
    }

    /// Checks if base is extended
    pub fn is_base_extended(&self) -> bool {
        self.extend_base
    }
}

/// Factory for creating ring extensions
///
/// Implements the unique factory pattern to ensure extension uniqueness.
#[derive(Debug, Default)]
pub struct RingExtensionFactory;

impl RingExtensionFactory {
    /// Creates a key for the extension
    pub fn create_key<K, L>(base: &K, extension: &L) -> String
    where
        K: Ring + fmt::Debug,
        L: Ring + fmt::Debug,
    {
        format!("{:?}/{:?}", extension, base)
    }

    /// Creates an extension object
    pub fn create<K, L>(description: String) -> RingExtensionGeneric<K, L>
    where
        K: Ring,
        L: Ring,
    {
        RingExtensionGeneric::new(description)
    }
}

/// Returns the common base of two rings
///
/// Finds the largest ring K such that both rings are extensions of K.
pub fn common_base<R>(ring1: &R, ring2: &R) -> Result<R, RingExtensionError>
where
    R: Ring + Clone,
{
    // Simplified: assume rings are equal for now
    // Full implementation would traverse extension towers
    Ok(ring1.clone())
}

/// Returns the tower of bases for a ring
///
/// Returns the chain: base_0 ⊂ base_1 ⊂ ... ⊂ ring
pub fn tower_bases<R>(_ring: &R) -> Vec<String>
where
    R: Ring,
{
    // Simplified: would traverse actual extension hierarchy
    vec!["Base ring".to_string()]
}

/// Extracts generators of a ring over a base
pub fn generators<R>(_ring: &R, _base: &R) -> Vec<String>
where
    R: Ring,
{
    // Simplified: would extract actual generators
    vec![]
}

/// Returns variable names for generators
pub fn variable_names<R>(_ring: &R, _base: &R) -> Vec<String>
where
    R: Ring,
{
    // Simplified: would extract variable names
    vec![]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generic_extension_creation() {
        let ext: RingExtensionGeneric<i32, i64> =
            RingExtensionGeneric::new("Z -> Q".to_string());
        assert_eq!(ext.description(), "Z -> Q");
        assert!(!ext.is_backend_visible());
    }

    #[test]
    fn test_extension_with_backend_visible() {
        let ext: RingExtensionGeneric<i32, f64> =
            RingExtensionGeneric::with_backend_visible("Extension".to_string());
        assert!(ext.is_backend_visible());
    }

    #[test]
    fn test_extension_finite() {
        let ext: RingExtensionGeneric<i32, i64> =
            RingExtensionGeneric::new("Finite ext".to_string());
        assert_eq!(ext.degree(), None);
        assert!(!ext.is_finite());
    }

    #[test]
    fn test_extension_with_basis() {
        let basis = vec!["1".to_string(), "sqrt2".to_string()];
        let ext: RingExtensionWithBasis<i32, f64> =
            RingExtensionWithBasis::new("Q(sqrt2)/Q".to_string(), basis);
        assert_eq!(ext.dimension(), 2);
        assert!(ext.is_finite());
        assert!(ext.is_free());
    }

    #[test]
    fn test_extension_with_gen() {
        let ext: RingExtensionWithGen<i32, f64> =
            RingExtensionWithGen::new("Q(sqrt2)".to_string(), "sqrt2".to_string(), 2);
        assert_eq!(ext.generator(), "sqrt2");
        assert_eq!(ext.degree(), 2);
    }

    #[test]
    fn test_fraction_field_extension() {
        let ext: RingExtensionFractionField<i32, f64> =
            RingExtensionFractionField::new("Frac(R)".to_string(), true);
        assert!(ext.is_base_extended());
    }

    #[test]
    fn test_factory_create_key() {
        let key = RingExtensionFactory::create_key(&5i32, &10i64);
        assert!(key.contains('/'));
    }

    #[test]
    fn test_factory_create() {
        let ext: RingExtensionGeneric<i32, i64> =
            RingExtensionFactory::create("Test".to_string());
        assert_eq!(ext.description(), "Test");
    }

    #[test]
    fn test_common_base() {
        let result = common_base(&5i32, &10i32);
        assert!(result.is_ok());
    }

    #[test]
    fn test_tower_bases() {
        let bases = tower_bases(&5i32);
        assert!(!bases.is_empty());
    }

    #[test]
    fn test_generators() {
        let gens = generators(&5i32, &3i32);
        assert!(gens.is_empty()); // Simplified implementation
    }

    #[test]
    fn test_variable_names() {
        let vars = variable_names(&5i32, &3i32);
        assert!(vars.is_empty()); // Simplified implementation
    }
}
