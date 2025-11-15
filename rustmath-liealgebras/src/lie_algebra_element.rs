//! Lie Algebra Element Types and Wrappers
//!
//! This module provides various element types and wrappers for Lie algebra elements,
//! including specialized implementations for free Lie algebras, structure coefficients,
//! affine Lie algebras, and subalgebras.
//!
//! # Element Hierarchy
//!
//! - `LieObject`: Base trait for Lie algebra element representations
//! - `LieGenerator`: Terminal generators in free Lie algebras
//! - `LieBracket`: Formal bracket expressions (binary trees)
//! - `GradedLieBracket`: Graded brackets with degree tracking
//! - `LyndonBracket`: Lyndon basis brackets using lexicographic ordering
//!
//! # Wrapper Types
//!
//! - `LieAlgebraElementWrapper`: Generic element wrapper
//! - `LieAlgebraMatrixWrapper`: Matrix-specific wrapper
//! - `LieSubalgebraElementWrapper`: Subalgebra element wrapper
//! - `StructureCoefficientsElement`: Elements with explicit structure coefficients
//! - `UntwistedAffineLieAlgebraElement`: Affine Lie algebra elements
//!
//! Corresponds to sage.algebras.lie_algebras.lie_algebra_element.pyx
//!
//! # References
//! - Serre, J-P. "Lie Algebras and Lie Groups" (1992)
//! - Reutenauer, C. "Free Lie Algebras" (1993)
//! - Kac, V. "Infinite Dimensional Lie Algebras" (1990)

use rustmath_core::Ring;
use rustmath_matrix::Matrix;
use std::collections::HashMap;
use std::fmt::{self, Display, Debug};
use std::hash::Hash;
use std::ops::{Add, Sub, Mul, Neg};
use std::marker::PhantomData;
use std::cmp::Ordering;

// ============================================================================
// LieObject: Base for Formal Lie Expressions
// ============================================================================

/// Base trait for formal Lie algebra objects
///
/// This represents abstract Lie expressions that can be generators or
/// formal bracket expressions. Used primarily for free Lie algebras.
pub trait LieObject: Clone + Display + PartialEq + Eq {
    /// Get the word representation (flattened to generators)
    fn word(&self) -> Vec<usize>;

    /// Get the degree (for graded Lie algebras)
    fn degree(&self) -> usize;

    /// Check if this is a generator (not a bracket)
    fn is_generator(&self) -> bool;
}

// ============================================================================
// LieGenerator: Terminal Generators
// ============================================================================

/// A generator of a free Lie algebra
///
/// This represents a terminal symbol (leaf node) in the tree structure
/// of a free Lie algebra element.
///
/// # Fields
///
/// * `index` - The index of this generator
/// * `name` - Optional name for display purposes
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LieGenerator {
    /// Index of the generator
    index: usize,

    /// Optional name
    name: Option<String>,
}

impl LieGenerator {
    /// Create a new generator
    ///
    /// # Arguments
    ///
    /// * `index` - The generator index
    /// * `name` - Optional name for display
    pub fn new(index: usize, name: Option<String>) -> Self {
        LieGenerator { index, name }
    }

    /// Get the index
    pub fn index(&self) -> usize {
        self.index
    }

    /// Get the name (if any)
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

impl LieObject for LieGenerator {
    fn word(&self) -> Vec<usize> {
        vec![self.index]
    }

    fn degree(&self) -> usize {
        1
    }

    fn is_generator(&self) -> bool {
        true
    }
}

impl Display for LieGenerator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.name {
            Some(n) => write!(f, "{}", n),
            None => write!(f, "x_{}", self.index),
        }
    }
}

// ============================================================================
// LieBracket: Formal Bracket Expressions
// ============================================================================

/// A formal Lie bracket [left, right]
///
/// This represents a binary tree node in the free Lie algebra,
/// encoding the bracket operation [left, right].
///
/// # Type Parameters
///
/// * `L` - Type implementing LieObject (can be generators or other brackets)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LieBracket<L: LieObject> {
    /// Left operand
    left: Box<L>,

    /// Right operand
    right: Box<L>,
}

impl<L: LieObject> LieBracket<L> {
    /// Create a new Lie bracket
    ///
    /// # Arguments
    ///
    /// * `left` - Left operand
    /// * `right` - Right operand
    pub fn new(left: L, right: L) -> Self {
        LieBracket {
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Get the left operand
    pub fn left(&self) -> &L {
        &self.left
    }

    /// Get the right operand
    pub fn right(&self) -> &L {
        &self.right
    }

    /// Compute the degree (sum of operand degrees)
    pub fn compute_degree(&self) -> usize {
        self.left.degree() + self.right.degree()
    }
}

impl<L: LieObject> LieObject for LieBracket<L> {
    fn word(&self) -> Vec<usize> {
        let mut result = self.left.word();
        result.extend(self.right.word());
        result
    }

    fn degree(&self) -> usize {
        self.compute_degree()
    }

    fn is_generator(&self) -> bool {
        false
    }
}

impl<L: LieObject> Display for LieBracket<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}]", self.left, self.right)
    }
}

// ============================================================================
// GradedLieBracket: Brackets with Explicit Degree Tracking
// ============================================================================

/// A graded Lie bracket with explicit degree storage
///
/// This extends LieBracket by storing the degree explicitly,
/// which is useful for graded Lie algebras where degree comparisons
/// are frequent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GradedLieBracket<L: LieObject> {
    /// The underlying bracket
    bracket: LieBracket<L>,

    /// Cached degree
    degree: usize,
}

impl<L: LieObject> GradedLieBracket<L> {
    /// Create a new graded bracket
    pub fn new(left: L, right: L) -> Self {
        let bracket = LieBracket::new(left, right);
        let degree = bracket.compute_degree();
        GradedLieBracket { bracket, degree }
    }

    /// Get the underlying bracket
    pub fn bracket(&self) -> &LieBracket<L> {
        &self.bracket
    }
}

impl<L: LieObject> LieObject for GradedLieBracket<L> {
    fn word(&self) -> Vec<usize> {
        self.bracket.word()
    }

    fn degree(&self) -> usize {
        self.degree
    }

    fn is_generator(&self) -> bool {
        false
    }
}

impl<L: LieObject> Display for GradedLieBracket<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.bracket)
    }
}

impl<L: LieObject> PartialOrd for GradedLieBracket<L> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<L: LieObject> Ord for GradedLieBracket<L> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Compare by degree first, then by word
        match self.degree.cmp(&other.degree) {
            Ordering::Equal => self.word().cmp(&other.word()),
            ord => ord,
        }
    }
}

// ============================================================================
// LyndonBracket: Lyndon Basis Brackets
// ============================================================================

/// A Lie bracket using Lyndon word ordering
///
/// Lyndon brackets use lexicographic ordering on the word representation,
/// which is useful for canonical representations in free Lie algebras.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LyndonBracket<L: LieObject> {
    /// The graded bracket
    graded_bracket: GradedLieBracket<L>,

    /// Cached word for ordering
    word_cache: Vec<usize>,
}

impl<L: LieObject> LyndonBracket<L> {
    /// Create a new Lyndon bracket
    pub fn new(left: L, right: L) -> Self {
        let graded_bracket = GradedLieBracket::new(left, right);
        let word_cache = graded_bracket.word();
        LyndonBracket {
            graded_bracket,
            word_cache,
        }
    }

    /// Get the graded bracket
    pub fn graded_bracket(&self) -> &GradedLieBracket<L> {
        &self.graded_bracket
    }
}

impl<L: LieObject> LieObject for LyndonBracket<L> {
    fn word(&self) -> Vec<usize> {
        self.word_cache.clone()
    }

    fn degree(&self) -> usize {
        self.graded_bracket.degree()
    }

    fn is_generator(&self) -> bool {
        false
    }
}

impl<L: LieObject> Display for LyndonBracket<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.graded_bracket)
    }
}

impl<L: LieObject> PartialOrd for LyndonBracket<L> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<L: LieObject> Ord for LyndonBracket<L> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Use lexicographic ordering on words
        self.word_cache.cmp(&other.word_cache)
    }
}

// ============================================================================
// FreeLieAlgebraElement: Elements of Free Lie Algebras
// ============================================================================

/// An element of a free Lie algebra
///
/// Represented as a linear combination of Lie monomials (formal brackets).
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
/// * `L` - The type of Lie objects (generators or brackets)
#[derive(Clone, Debug)]
pub struct FreeLieAlgebraElement<R: Ring + Clone, L: LieObject + Hash> {
    /// Coefficients for each Lie monomial
    monomials: HashMap<L, R>,
}

impl<R: Ring + Clone + PartialEq + From<i64>, L: LieObject + Hash> FreeLieAlgebraElement<R, L> {
    /// Create a new element
    pub fn new(monomials: HashMap<L, R>) -> Self {
        // Remove zero coefficients
        let filtered: HashMap<L, R> = monomials
            .into_iter()
            .filter(|(_, coeff)| coeff != &R::from(0))
            .collect();

        FreeLieAlgebraElement {
            monomials: filtered,
        }
    }

    /// Create the zero element
    pub fn zero() -> Self {
        FreeLieAlgebraElement {
            monomials: HashMap::new(),
        }
    }

    /// Create an element from a single generator
    pub fn from_generator(gen: L, coeff: R) -> Self {
        let mut monomials = HashMap::new();
        if coeff != R::from(0) {
            monomials.insert(gen, coeff);
        }
        FreeLieAlgebraElement { monomials }
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool {
        self.monomials.is_empty()
    }

    /// Get the monomials
    pub fn monomials(&self) -> &HashMap<L, R> {
        &self.monomials
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &R) -> Self {
        if scalar == &R::from(0) {
            return Self::zero();
        }

        let scaled: HashMap<L, R> = self
            .monomials
            .iter()
            .map(|(mon, coeff)| (mon.clone(), coeff.clone() * scalar.clone()))
            .collect();

        FreeLieAlgebraElement::new(scaled)
    }
}

impl<R: Ring + Clone + PartialEq + From<i64>, L: LieObject + Hash> Add for FreeLieAlgebraElement<R, L> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut result = self.monomials.clone();

        for (mon, coeff) in other.monomials {
            let entry = result.entry(mon).or_insert_with(|| R::from(0));
            *entry = entry.clone() + coeff;
        }

        FreeLieAlgebraElement::new(result)
    }
}

impl<R: Ring + Clone + PartialEq + From<i64>, L: LieObject + Hash> Sub for FreeLieAlgebraElement<R, L> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut result = self.monomials.clone();

        for (mon, coeff) in other.monomials {
            let entry = result.entry(mon).or_insert_with(|| R::from(0));
            *entry = entry.clone() - coeff;
        }

        FreeLieAlgebraElement::new(result)
    }
}

impl<R: Ring + Clone + PartialEq + From<i64>, L: LieObject + Hash> Neg for FreeLieAlgebraElement<R, L> {
    type Output = Self;

    fn neg(self) -> Self {
        let negated: HashMap<L, R> = self
            .monomials
            .into_iter()
            .map(|(mon, coeff)| (mon, -coeff))
            .collect();

        FreeLieAlgebraElement::new(negated)
    }
}

// ============================================================================
// StructureCoefficientsElement: Elements with Explicit Structure
// ============================================================================

/// Element of a Lie algebra defined by structure coefficients
///
/// Elements are represented as coordinate vectors with respect to a fixed basis.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructureCoefficientsElement<R: Ring + Clone> {
    /// Coordinates with respect to the basis
    coordinates: Vec<R>,

    /// Basis element names (for display)
    basis_names: Vec<String>,
}

impl<R: Ring + Clone + PartialEq + From<i64>> StructureCoefficientsElement<R> {
    /// Create a new element
    ///
    /// # Arguments
    ///
    /// * `coordinates` - Coefficient vector
    /// * `basis_names` - Names of basis elements
    pub fn new(coordinates: Vec<R>, basis_names: Vec<String>) -> Result<Self, String> {
        if coordinates.len() != basis_names.len() {
            return Err("Coordinates length must match basis size".to_string());
        }

        Ok(StructureCoefficientsElement {
            coordinates,
            basis_names,
        })
    }

    /// Get the coordinates
    pub fn coordinates(&self) -> &[R] {
        &self.coordinates
    }

    /// Get the basis names
    pub fn basis_names(&self) -> &[String] {
        &self.basis_names
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool {
        self.coordinates.iter().all(|c| c == &R::from(0))
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.coordinates.len()
    }
}

impl<R: Ring + Clone + PartialEq + From<i64>> Add for StructureCoefficientsElement<R> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        assert_eq!(self.coordinates.len(), other.coordinates.len());

        let coords: Vec<R> = self
            .coordinates
            .iter()
            .zip(&other.coordinates)
            .map(|(a, b)| a.clone() + b.clone())
            .collect();

        StructureCoefficientsElement {
            coordinates: coords,
            basis_names: self.basis_names,
        }
    }
}

impl<R: Ring + Clone + PartialEq + From<i64>> Sub for StructureCoefficientsElement<R> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        assert_eq!(self.coordinates.len(), other.coordinates.len());

        let coords: Vec<R> = self
            .coordinates
            .iter()
            .zip(&other.coordinates)
            .map(|(a, b)| a.clone() - b.clone())
            .collect();

        StructureCoefficientsElement {
            coordinates: coords,
            basis_names: self.basis_names,
        }
    }
}

impl<R: Ring + Clone + PartialEq + From<i64>> Neg for StructureCoefficientsElement<R> {
    type Output = Self;

    fn neg(self) -> Self {
        let coords: Vec<R> = self
            .coordinates
            .into_iter()
            .map(|c| -c)
            .collect();

        StructureCoefficientsElement {
            coordinates: coords,
            basis_names: self.basis_names,
        }
    }
}

impl<R: Ring + Clone + Display + PartialEq + From<i64>> Display for StructureCoefficientsElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let terms: Vec<String> = self
            .coordinates
            .iter()
            .zip(&self.basis_names)
            .filter(|(coeff, _)| coeff != &&R::from(0))
            .map(|(coeff, name)| format!("{} * {}", coeff, name))
            .collect();

        if terms.is_empty() {
            write!(f, "0")
        } else {
            write!(f, "{}", terms.join(" + "))
        }
    }
}

// ============================================================================
// UntwistedAffineLieAlgebraElement: Affine Lie Algebra Elements
// ============================================================================

/// Element of an untwisted affine Lie algebra
///
/// These algebras have a central extension and derivation, with elements
/// of the form: Σ x_i ⊗ t^i + c*c + d*d where x_i are classical Lie algebra
/// elements, c is the central element, and d is the derivation.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
#[derive(Clone, Debug)]
pub struct UntwistedAffineLieAlgebraElement<R: Ring + Clone> {
    /// Classical Lie algebra part: map from power of t to element
    classical_part: HashMap<i64, Vec<R>>,

    /// Central element coefficient
    central_coefficient: R,

    /// Derivation coefficient
    derivation_coefficient: R,

    /// Dimension of classical part
    classical_dimension: usize,
}

impl<R: Ring + Clone + PartialEq + From<i64>> UntwistedAffineLieAlgebraElement<R> {
    /// Create a new affine element
    ///
    /// # Arguments
    ///
    /// * `classical_part` - Map from t-power to classical element coordinates
    /// * `central_coefficient` - Coefficient of central element
    /// * `derivation_coefficient` - Coefficient of derivation
    /// * `classical_dimension` - Dimension of classical Lie algebra
    pub fn new(
        classical_part: HashMap<i64, Vec<R>>,
        central_coefficient: R,
        derivation_coefficient: R,
        classical_dimension: usize,
    ) -> Result<Self, String> {
        // Verify all classical parts have correct dimension
        for coords in classical_part.values() {
            if coords.len() != classical_dimension {
                return Err("All classical parts must have correct dimension".to_string());
            }
        }

        Ok(UntwistedAffineLieAlgebraElement {
            classical_part,
            central_coefficient,
            derivation_coefficient,
            classical_dimension,
        })
    }

    /// Get the classical part
    pub fn classical_part(&self) -> &HashMap<i64, Vec<R>> {
        &self.classical_part
    }

    /// Get the central coefficient
    pub fn central_coefficient(&self) -> &R {
        &self.central_coefficient
    }

    /// Get the derivation coefficient
    pub fn derivation_coefficient(&self) -> &R {
        &self.derivation_coefficient
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool {
        self.classical_part.is_empty()
            && self.central_coefficient == R::from(0)
            && self.derivation_coefficient == R::from(0)
    }
}

impl<R: Ring + Clone + PartialEq + From<i64>> Add for UntwistedAffineLieAlgebraElement<R> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        assert_eq!(self.classical_dimension, other.classical_dimension);

        let mut result_classical = self.classical_part.clone();

        for (power, coords) in other.classical_part {
            let entry = result_classical
                .entry(power)
                .or_insert_with(|| vec![R::from(0); self.classical_dimension]);

            for (i, c) in coords.iter().enumerate() {
                entry[i] = entry[i].clone() + c.clone();
            }
        }

        // Remove zero entries
        result_classical.retain(|_, coords| {
            coords.iter().any(|c| c != &R::from(0))
        });

        UntwistedAffineLieAlgebraElement {
            classical_part: result_classical,
            central_coefficient: self.central_coefficient + other.central_coefficient,
            derivation_coefficient: self.derivation_coefficient + other.derivation_coefficient,
            classical_dimension: self.classical_dimension,
        }
    }
}

impl<R: Ring + Clone + PartialEq + From<i64>> Neg for UntwistedAffineLieAlgebraElement<R> {
    type Output = Self;

    fn neg(self) -> Self {
        let negated_classical: HashMap<i64, Vec<R>> = self
            .classical_part
            .into_iter()
            .map(|(power, coords)| {
                let neg_coords = coords.into_iter().map(|c| -c).collect();
                (power, neg_coords)
            })
            .collect();

        UntwistedAffineLieAlgebraElement {
            classical_part: negated_classical,
            central_coefficient: -self.central_coefficient,
            derivation_coefficient: -self.derivation_coefficient,
            classical_dimension: self.classical_dimension,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_lie_generator() {
        let gen1 = LieGenerator::new(0, Some("e".to_string()));
        let gen2 = LieGenerator::new(1, Some("f".to_string()));

        assert_eq!(gen1.index(), 0);
        assert_eq!(gen1.name(), Some("e"));
        assert!(gen1.is_generator());
        assert_eq!(gen1.degree(), 1);
        assert_eq!(gen1.word(), vec![0]);

        assert_ne!(gen1, gen2);
    }

    #[test]
    fn test_lie_bracket() {
        let e = LieGenerator::new(0, Some("e".to_string()));
        let f = LieGenerator::new(1, Some("f".to_string()));

        let bracket = LieBracket::new(e.clone(), f.clone());

        assert!(!bracket.is_generator());
        assert_eq!(bracket.degree(), 2);
        assert_eq!(bracket.word(), vec![0, 1]);
        assert_eq!(bracket.left(), &e);
        assert_eq!(bracket.right(), &f);
    }

    #[test]
    fn test_graded_lie_bracket() {
        let e = LieGenerator::new(0, Some("e".to_string()));
        let f = LieGenerator::new(1, Some("f".to_string()));
        let h = LieGenerator::new(2, Some("h".to_string()));

        let ef = GradedLieBracket::new(e.clone(), f.clone());
        let eh = GradedLieBracket::new(e.clone(), h.clone());

        assert_eq!(ef.degree(), 2);
        assert_eq!(eh.degree(), 2);

        // Both have degree 2, so compare by word
        assert!(ef < eh || ef > eh); // They should be comparable
    }

    #[test]
    fn test_lyndon_bracket() {
        let gen0 = LieGenerator::new(0, None);
        let gen1 = LieGenerator::new(1, None);

        let lyndon = LyndonBracket::new(gen0.clone(), gen1.clone());

        assert_eq!(lyndon.degree(), 2);
        assert_eq!(lyndon.word(), vec![0, 1]);
    }

    #[test]
    fn test_free_lie_algebra_element() {
        let e = LieGenerator::new(0, Some("e".to_string()));
        let f = LieGenerator::new(1, Some("f".to_string()));

        let elem1 = FreeLieAlgebraElement::from_generator(e.clone(), Integer::from(2));
        let elem2 = FreeLieAlgebraElement::from_generator(f.clone(), Integer::from(3));

        let sum = elem1.clone() + elem2.clone();
        assert_eq!(sum.monomials().len(), 2);

        let diff = elem1.clone() - elem2.clone();
        assert!(!diff.is_zero());

        let zero = elem1.clone() - elem1.clone();
        assert!(zero.is_zero());
    }

    #[test]
    fn test_structure_coefficients_element() {
        let coords = vec![Integer::from(1), Integer::from(2), Integer::from(0)];
        let basis = vec!["e".to_string(), "f".to_string(), "h".to_string()];

        let elem = StructureCoefficientsElement::new(coords, basis).unwrap();

        assert_eq!(elem.dimension(), 3);
        assert!(!elem.is_zero());
        assert_eq!(elem.coordinates()[0], Integer::from(1));
    }

    #[test]
    fn test_structure_coefficients_arithmetic() {
        let coords1 = vec![Integer::from(1), Integer::from(2)];
        let coords2 = vec![Integer::from(3), Integer::from(4)];
        let basis = vec!["x".to_string(), "y".to_string()];

        let elem1 = StructureCoefficientsElement::new(coords1, basis.clone()).unwrap();
        let elem2 = StructureCoefficientsElement::new(coords2, basis).unwrap();

        let sum = elem1.clone() + elem2.clone();
        assert_eq!(sum.coordinates()[0], Integer::from(4));
        assert_eq!(sum.coordinates()[1], Integer::from(6));

        let diff = elem1 - elem2;
        assert_eq!(diff.coordinates()[0], Integer::from(-2));
        assert_eq!(diff.coordinates()[1], Integer::from(-2));
    }

    #[test]
    fn test_untwisted_affine_element() {
        let mut classical = HashMap::new();
        classical.insert(0, vec![Integer::from(1), Integer::from(0)]);
        classical.insert(1, vec![Integer::from(0), Integer::from(2)]);

        let elem = UntwistedAffineLieAlgebraElement::new(
            classical,
            Integer::from(3),
            Integer::from(0),
            2,
        ).unwrap();

        assert!(!elem.is_zero());
        assert_eq!(elem.central_coefficient(), &Integer::from(3));
        assert_eq!(elem.derivation_coefficient(), &Integer::from(0));
        assert_eq!(elem.classical_part().len(), 2);
    }

    #[test]
    fn test_untwisted_affine_addition() {
        let mut classical1 = HashMap::new();
        classical1.insert(0, vec![Integer::from(1), Integer::from(0)]);

        let mut classical2 = HashMap::new();
        classical2.insert(0, vec![Integer::from(2), Integer::from(3)]);
        classical2.insert(1, vec![Integer::from(1), Integer::from(0)]);

        let elem1 = UntwistedAffineLieAlgebraElement::new(
            classical1,
            Integer::from(1),
            Integer::from(0),
            2,
        ).unwrap();

        let elem2 = UntwistedAffineLieAlgebraElement::new(
            classical2,
            Integer::from(2),
            Integer::from(1),
            2,
        ).unwrap();

        let sum = elem1 + elem2;
        assert_eq!(sum.central_coefficient(), &Integer::from(3));
        assert_eq!(sum.derivation_coefficient(), &Integer::from(1));
    }
}
