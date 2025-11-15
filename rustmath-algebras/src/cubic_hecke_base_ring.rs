//! Cubic Hecke Algebra Base Rings
//!
//! This module implements the foundational ring structures needed to define
//! and work with cubic Hecke algebras.
//!
//! The cubic Hecke algebra H(n) is defined over a specialized base ring that
//! contains the parameters for the cubic relation. This module provides:
//!
//! - `CubicHeckeRingOfDefinition`: The ring of definition containing the
//!   fundamental parameters for the cubic relation
//! - `CubicHeckeExtensionRing`: The splitting algebra for irreducible representations
//! - `GaloisGroupAction`: Galois group actions on polynomial rings
//! - Helper functions for naming conventions and ring homomorphisms
//!
//! # Mathematical Background
//!
//! The cubic Hecke algebra is defined by the cubic relation:
//!
//! s_i³ = u·s_i² - v·s_i + w
//!
//! where u, v, w are elements in the base ring. The ring of definition
//! is constructed to contain these parameters in a generic way, allowing
//! for specializations to specific polynomial rings or number fields.
//!
//! Corresponds to sage.algebras.hecke_algebras.cubic_hecke_base_ring
//!
//! # References
//!
//! - Marin, I. "The cubic Hecke algebra on at most 5 strands" (2015)
//! - Brav, C. and Thomas, H. "Braid groups and Kleinian singularities" (2011)

use rustmath_core::{Ring, Field};
use rustmath_polynomials::{Polynomial, UnivariatePolynomial, MultivariatePolynomial};
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::marker::PhantomData;
use std::hash::Hash;

/// Names for the Markov trace parameters
///
/// The Markov trace provides specializations to classical link invariants:
/// - HOMFLY-PT polynomial
/// - Kauffman polynomial
/// - Links-Gould polynomial
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkovTraceVersion {
    /// Standard cubic Hecke algebra parameters
    Standard,
    /// Markov trace parameters for link invariants
    Markov,
}

/// Normalize variable names for Markov trace
///
/// Processes variable naming conventions, supporting both standard
/// and Markov trace versions with specific length requirements.
///
/// # Arguments
///
/// * `names` - Variable names to normalize
/// * `version` - Whether to use standard or Markov trace naming
/// * `length` - Required number of variables
///
/// # Returns
///
/// Normalized list of variable names
///
/// # Examples
///
/// ```
/// # use rustmath_algebras::cubic_hecke_base_ring::{normalize_names_markov, MarkovTraceVersion};
/// let names = normalize_names_markov(vec!["u".to_string(), "v".to_string(), "w".to_string()],
///                                      MarkovTraceVersion::Standard, 3);
/// assert_eq!(names, vec!["u", "v", "w"]);
/// ```
pub fn normalize_names_markov(
    names: Vec<String>,
    version: MarkovTraceVersion,
    length: usize,
) -> Vec<String> {
    if names.len() == length {
        return names;
    }

    match version {
        MarkovTraceVersion::Standard => {
            // Standard names: u, v, w
            if length == 3 {
                vec!["u".to_string(), "v".to_string(), "w".to_string()]
            } else {
                (0..length).map(|i| format!("x{}", i)).collect()
            }
        }
        MarkovTraceVersion::Markov => {
            // Markov trace names: typically different conventions
            if length == 3 {
                vec!["a".to_string(), "b".to_string(), "c".to_string()]
            } else {
                (0..length).map(|i| format!("m{}", i)).collect()
            }
        }
    }
}

/// Register a ring homomorphism as a conversion map
///
/// Manages compatibility between different ring structures by
/// registering homomorphisms for automatic coercion.
///
/// # Type Parameters
///
/// * `R` - Source ring
/// * `S` - Target ring
///
/// # Arguments
///
/// * `morphism` - The homomorphism to register
pub fn register_ring_hom<R, S, F>(morphism: F) -> RingHomomorphism<R, S, F>
where
    R: Ring,
    S: Ring,
    F: Fn(&R) -> S,
{
    RingHomomorphism {
        morphism,
        _phantom_r: PhantomData,
        _phantom_s: PhantomData,
    }
}

/// A ring homomorphism between two rings
pub struct RingHomomorphism<R, S, F>
where
    R: Ring,
    S: Ring,
    F: Fn(&R) -> S,
{
    morphism: F,
    _phantom_r: PhantomData<R>,
    _phantom_s: PhantomData<S>,
}

impl<R, S, F> RingHomomorphism<R, S, F>
where
    R: Ring,
    S: Ring,
    F: Fn(&R) -> S,
{
    /// Apply the homomorphism
    pub fn apply(&self, x: &R) -> S {
        (self.morphism)(x)
    }
}

/// Galois group action on polynomial rings
///
/// Implements permutation of the generators of multivariate polynomial rings,
/// enabling Galois group operations on ring elements.
///
/// # Type Parameters
///
/// * `R` - Base ring
///
/// # Examples
///
/// ```
/// # use rustmath_algebras::cubic_hecke_base_ring::GaloisGroupAction;
/// # use rustmath_integers::Integer;
/// let action: GaloisGroupAction<Integer> = GaloisGroupAction::new(vec![1, 0, 2]);
/// // This represents swapping the first two variables
/// ```
#[derive(Debug, Clone)]
pub struct GaloisGroupAction<R: Ring> {
    /// Permutation of variable indices
    permutation: Vec<usize>,
    /// Base ring marker
    _phantom: PhantomData<R>,
}

impl<R: Ring> GaloisGroupAction<R> {
    /// Create a new Galois group action
    ///
    /// # Arguments
    ///
    /// * `permutation` - How to permute variable indices
    pub fn new(permutation: Vec<usize>) -> Self {
        GaloisGroupAction {
            permutation,
            _phantom: PhantomData,
        }
    }

    /// Get the permutation
    pub fn permutation(&self) -> &[usize] {
        &self.permutation
    }

    /// Identity action
    pub fn identity(n: usize) -> Self {
        GaloisGroupAction {
            permutation: (0..n).collect(),
            _phantom: PhantomData,
        }
    }

    /// Compose two actions
    pub fn compose(&self, other: &Self) -> Self {
        let permutation = self
            .permutation
            .iter()
            .map(|&i| other.permutation[i])
            .collect();
        GaloisGroupAction {
            permutation,
            _phantom: PhantomData,
        }
    }

    /// Inverse action
    pub fn inverse(&self) -> Self {
        let mut inv_perm = vec![0; self.permutation.len()];
        for (i, &p) in self.permutation.iter().enumerate() {
            inv_perm[p] = i;
        }
        GaloisGroupAction {
            permutation: inv_perm,
            _phantom: PhantomData,
        }
    }
}

/// The ring of definition for the cubic Hecke algebra
///
/// This is the most general base ring for the cubic Hecke algebra.
/// It contains one invertible indeterminate (representing the product of
/// cubic equation roots) and two non-invertible indeterminates representing
/// symmetric functions of these roots.
///
/// The ring is a localization of Z[u,v,w] where we invert u.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring (typically Z or a field)
///
/// # Mathematical Structure
///
/// The cubic relation is: t³ = u·t² - v·t + w
///
/// The ring of definition contains:
/// - u: invertible (the product of roots)
/// - v: non-invertible (sum of products of pairs)
/// - w: non-invertible (elementary symmetric function)
///
/// # Examples
///
/// ```
/// # use rustmath_algebras::cubic_hecke_base_ring::CubicHeckeRingOfDefinition;
/// # use rustmath_integers::Integer;
/// let ring: CubicHeckeRingOfDefinition<Integer> =
///     CubicHeckeRingOfDefinition::new();
/// ```
#[derive(Debug, Clone)]
pub struct CubicHeckeRingOfDefinition<R: Ring> {
    /// The base coefficient ring
    base_ring: PhantomData<R>,
    /// Variable names (u, v, w)
    var_names: Vec<String>,
}

impl<R: Ring> CubicHeckeRingOfDefinition<R> {
    /// Create a new cubic Hecke ring of definition
    pub fn new() -> Self {
        CubicHeckeRingOfDefinition {
            base_ring: PhantomData,
            var_names: vec!["u".to_string(), "v".to_string(), "w".to_string()],
        }
    }

    /// Create with custom variable names
    pub fn with_names(names: Vec<String>) -> Self {
        CubicHeckeRingOfDefinition {
            base_ring: PhantomData,
            var_names: names,
        }
    }

    /// Get the variable names
    pub fn var_names(&self) -> &[String] {
        &self.var_names
    }

    /// Get the u variable (invertible)
    pub fn u(&self) -> String {
        self.var_names[0].clone()
    }

    /// Get the v variable
    pub fn v(&self) -> String {
        self.var_names[1].clone()
    }

    /// Get the w variable
    pub fn w(&self) -> String {
        self.var_names[2].clone()
    }

    /// The cubic equation defining the ring
    ///
    /// Returns the polynomial t³ - u·t² + v·t - w
    pub fn cubic_equation(&self) -> Vec<(i32, String)>
    where
        R: From<i64>,
    {
        vec![
            (1, "t^3".to_string()),
            (-1, format!("{}*t^2", self.u())),
            (1, format!("{}*t", self.v())),
            (-1, self.w()),
        ]
    }

    /// Mirror involution on the ring
    ///
    /// Implements the involution corresponding to the braid group's
    /// mirror operation. This typically involves complex conjugation
    /// or a sign change on certain generators.
    pub fn mirror_involution<T>(&self, element: T) -> T
    where
        T: Clone,
    {
        // In the most basic form, the mirror involution is the identity
        // In specific extensions, it may involve conjugation or sign changes
        element
    }
}

impl<R: Ring> Default for CubicHeckeRingOfDefinition<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Ring> Display for CubicHeckeRingOfDefinition<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Cubic Hecke Ring of Definition over {} in variables {}",
            std::any::type_name::<R>(),
            self.var_names.join(", ")
        )
    }
}

/// The extension ring for cubic Hecke algebra representations
///
/// This is the "generic splitting algebra" for the irreducible
/// representations of the cubic Hecke algebra. It is constructed as a
/// Laurent polynomial ring in three indeterminates over a splitting
/// algebra containing a third root of unity.
///
/// This ring guarantees semisimplicity of the cubic Hecke algebra.
///
/// # Type Parameters
///
/// * `R` - Base ring (must contain a third root of unity)
///
/// # Mathematical Structure
///
/// Constructed as R[a±1, b±1, c±1] where:
/// - a, b, c are the three roots of the cubic equation
/// - They satisfy: abc = u, ab + ac + bc = v, a + b + c = ?
///
/// # Examples
///
/// ```
/// # use rustmath_algebras::cubic_hecke_base_ring::CubicHeckeExtensionRing;
/// # use rustmath_integers::Integer;
/// let ring: CubicHeckeExtensionRing<Integer> =
///     CubicHeckeExtensionRing::new(3);
/// assert_eq!(ring.num_variables(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct CubicHeckeExtensionRing<R: Ring> {
    /// Number of variables
    num_vars: usize,
    /// Variable names (typically a, b, c)
    var_names: Vec<String>,
    /// Base ring marker
    _phantom: PhantomData<R>,
}

impl<R: Ring> CubicHeckeExtensionRing<R> {
    /// Create a new cubic Hecke extension ring
    ///
    /// # Arguments
    ///
    /// * `num_vars` - Number of variables (typically 3 for a, b, c)
    pub fn new(num_vars: usize) -> Self {
        let var_names = if num_vars == 3 {
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        } else {
            (0..num_vars).map(|i| format!("x{}", i)).collect()
        };

        CubicHeckeExtensionRing {
            num_vars,
            var_names,
            _phantom: PhantomData,
        }
    }

    /// Create with custom variable names
    pub fn with_names(names: Vec<String>) -> Self {
        let num_vars = names.len();
        CubicHeckeExtensionRing {
            num_vars,
            var_names: names,
            _phantom: PhantomData,
        }
    }

    /// Get the number of variables
    pub fn num_variables(&self) -> usize {
        self.num_vars
    }

    /// Get the variable names
    pub fn var_names(&self) -> &[String] {
        &self.var_names
    }

    /// Get a specific variable name
    pub fn var(&self, i: usize) -> Option<&String> {
        self.var_names.get(i)
    }

    /// Relations defining the splitting
    ///
    /// Returns the relations that the variables satisfy in terms
    /// of the ring of definition parameters
    pub fn splitting_relations(&self) -> Vec<String>
    where
        R: From<i64>,
    {
        if self.num_vars == 3 {
            let a = &self.var_names[0];
            let b = &self.var_names[1];
            let c = &self.var_names[2];
            vec![
                format!("{}*{}*{} = u", a, b, c),
                format!("{}*{} + {}*{} + {}*{} = v", a, b, a, c, b, c),
            ]
        } else {
            vec![]
        }
    }
}

impl<R: Ring> Display for CubicHeckeExtensionRing<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Cubic Hecke Extension Ring in {} variables: {}",
            self.num_vars,
            self.var_names.join(", ")
        )
    }
}

/// Convert from ring of definition to extension ring
///
/// Implements the natural embedding of the ring of definition into
/// the extension ring by expressing u, v, w in terms of a, b, c.
pub fn embedding_to_extension<R>(
    ring_def: &CubicHeckeRingOfDefinition<R>,
    ext_ring: &CubicHeckeExtensionRing<R>,
) -> HashMap<String, String>
where
    R: Ring,
{
    let mut map = HashMap::new();

    if ext_ring.num_variables() == 3 {
        let a = ext_ring.var(0).unwrap();
        let b = ext_ring.var(1).unwrap();
        let c = ext_ring.var(2).unwrap();

        // u = abc
        map.insert(ring_def.u(), format!("{}*{}*{}", a, b, c));
        // v = ab + ac + bc
        map.insert(
            ring_def.v(),
            format!("{}*{} + {}*{} + {}*{}", a, b, a, c, b, c),
        );
        // For w, we need more context, but it's related to (a+b+c)
        map.insert(ring_def.w(), "w".to_string()); // Placeholder
    }

    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_normalize_names_standard() {
        let names = normalize_names_markov(
            vec![],
            MarkovTraceVersion::Standard,
            3,
        );
        assert_eq!(names, vec!["u", "v", "w"]);
    }

    #[test]
    fn test_normalize_names_markov() {
        let names = normalize_names_markov(
            vec![],
            MarkovTraceVersion::Markov,
            3,
        );
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_normalize_names_custom() {
        let custom = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        let names = normalize_names_markov(
            custom.clone(),
            MarkovTraceVersion::Standard,
            3,
        );
        assert_eq!(names, custom);
    }

    #[test]
    fn test_galois_action_identity() {
        let action: GaloisGroupAction<Integer> = GaloisGroupAction::identity(3);
        assert_eq!(action.permutation(), &[0, 1, 2]);
    }

    #[test]
    fn test_galois_action_compose() {
        let swap_01: GaloisGroupAction<Integer> = GaloisGroupAction::new(vec![1, 0, 2]);
        let swap_12: GaloisGroupAction<Integer> = GaloisGroupAction::new(vec![0, 2, 1]);

        let composed = swap_01.compose(&swap_12);
        assert_eq!(composed.permutation(), &[2, 0, 1]);
    }

    #[test]
    fn test_galois_action_inverse() {
        let perm: GaloisGroupAction<Integer> = GaloisGroupAction::new(vec![2, 0, 1]);
        let inv = perm.inverse();

        // Composing with inverse should give identity
        let id = perm.compose(&inv);
        assert_eq!(id.permutation(), &[0, 1, 2]);
    }

    #[test]
    fn test_ring_of_definition() {
        let ring: CubicHeckeRingOfDefinition<Integer> = CubicHeckeRingOfDefinition::new();
        assert_eq!(ring.u(), "u");
        assert_eq!(ring.v(), "v");
        assert_eq!(ring.w(), "w");
    }

    #[test]
    fn test_ring_of_definition_custom_names() {
        let names = vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()];
        let ring: CubicHeckeRingOfDefinition<Integer> =
            CubicHeckeRingOfDefinition::with_names(names.clone());
        assert_eq!(ring.var_names(), names.as_slice());
    }

    #[test]
    fn test_extension_ring() {
        let ring: CubicHeckeExtensionRing<Integer> = CubicHeckeExtensionRing::new(3);
        assert_eq!(ring.num_variables(), 3);
        assert_eq!(ring.var_names(), &["a", "b", "c"]);
    }

    #[test]
    fn test_extension_ring_variables() {
        let ring: CubicHeckeExtensionRing<Integer> = CubicHeckeExtensionRing::new(3);
        assert_eq!(ring.var(0), Some(&"a".to_string()));
        assert_eq!(ring.var(1), Some(&"b".to_string()));
        assert_eq!(ring.var(2), Some(&"c".to_string()));
        assert_eq!(ring.var(3), None);
    }

    #[test]
    fn test_splitting_relations() {
        let ring: CubicHeckeExtensionRing<Integer> = CubicHeckeExtensionRing::new(3);
        let relations = ring.splitting_relations();
        assert_eq!(relations.len(), 2);
        assert!(relations[0].contains("a*b*c = u"));
    }

    #[test]
    fn test_embedding() {
        let ring_def: CubicHeckeRingOfDefinition<Integer> = CubicHeckeRingOfDefinition::new();
        let ext_ring: CubicHeckeExtensionRing<Integer> = CubicHeckeExtensionRing::new(3);

        let embedding = embedding_to_extension(&ring_def, &ext_ring);

        assert!(embedding.contains_key("u"));
        assert!(embedding.contains_key("v"));
        assert!(embedding.get("u").unwrap().contains("a*b*c"));
    }

    #[test]
    fn test_cubic_equation() {
        let ring: CubicHeckeRingOfDefinition<Integer> = CubicHeckeRingOfDefinition::new();
        let eq = ring.cubic_equation();

        assert_eq!(eq.len(), 4);
        assert_eq!(eq[0], (1, "t^3".to_string()));
    }

    #[test]
    fn test_ring_homomorphism() {
        let hom = register_ring_hom(|x: &Integer| x.clone());
        let value = Integer::from(42);
        let result = hom.apply(&value);
        assert_eq!(result, Integer::from(42));
    }
}
