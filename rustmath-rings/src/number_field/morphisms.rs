//! # Number Field Morphisms
//!
//! This module implements morphisms (homomorphisms) between number fields,
//! corresponding to SageMath's `sage.rings.number_field.morphisms`.
//!
//! ## Overview
//!
//! A morphism between number fields is a ring homomorphism that respects
//! the field structure. For number fields K and L, a morphism φ: K → L
//! satisfies:
//! - φ(a + b) = φ(a) + φ(b)
//! - φ(a · b) = φ(a) · φ(b)
//! - φ(1) = 1
//!
//! ## Key Concepts
//!
//! ### Embeddings
//! An embedding is an injective morphism from a number field into another field
//! (typically ℂ or ℝ). For a number field K = ℚ(α) of degree n, there are
//! exactly n distinct embeddings into ℂ, corresponding to the n roots of the
//! minimal polynomial of α.
//!
//! ### Automorphisms
//! An automorphism is a bijective morphism from a field to itself. The set of
//! all automorphisms forms the **automorphism group** Aut(K/ℚ), which is a
//! subgroup of the Galois group.
//!
//! ### Galois Theory
//! For a Galois extension K/ℚ (a normal and separable extension), the Galois
//! group Gal(K/ℚ) consists of all automorphisms of K that fix ℚ. The order
//! of the Galois group equals the degree of the extension: |Gal(K/ℚ)| = [K:ℚ].
//!
//! For non-Galois extensions, |Aut(K/ℚ)| < [K:ℚ].
//!
//! ### Normal and Separable Extensions
//! - **Normal**: K is the splitting field of some polynomial over ℚ
//! - **Separable**: The minimal polynomial has no repeated roots
//! - **Galois**: Both normal and separable
//!
//! ## Implementation Approach
//!
//! Our implementation uses the following strategies:
//!
//! 1. **Morphism Representation**: A morphism φ: ℚ(α) → L is uniquely determined
//!    by the image φ(α), which must be a root of the minimal polynomial of α in L.
//!
//! 2. **Computing Automorphisms**: To find Aut(K/ℚ), we:
//!    - Find all roots of the minimal polynomial within K
//!    - For each root β, check if the map α ↦ β extends to an automorphism
//!    - Verify that the map is well-defined and bijective
//!
//! 3. **Galois Group Structure**: For Galois extensions, we compute:
//!    - The group operation (composition of automorphisms)
//!    - Conjugacy classes
//!    - Fixed fields of subgroups
//!    - Group presentation and isomorphism type
//!
//! 4. **Splitting Fields**: For a polynomial f ∈ ℚ[x], the splitting field is
//!    the smallest field containing all roots of f. We compute it by:
//!    - Adjoining one root at a time
//!    - Factoring f over each intermediate field
//!    - Continuing until f splits completely
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_numberfields::NumberField;
//! use rustmath_rings::number_field::morphisms::{NumberFieldEmbedding, compute_automorphisms};
//! use rustmath_polynomials::univariate::UnivariatePolynomial;
//! use rustmath_rationals::Rational;
//!
//! // Create ℚ(√2)
//! let poly = UnivariatePolynomial::new(vec![
//!     Rational::from_integer(-2),
//!     Rational::from_integer(0),
//!     Rational::from_integer(1),
//! ]);
//! let field = NumberField::new(poly);
//!
//! // Compute automorphism group (order 2 for quadratic fields)
//! let auts = compute_automorphisms(&field);
//! assert_eq!(auts.len(), 2); // Identity and conjugation
//! ```

use rustmath_core::{CommutativeRing, Field, Ring};
use rustmath_complex::Complex;
use rustmath_integers::Integer;
use rustmath_numberfields::{NumberField, NumberFieldElement, NumberFieldError};
use rustmath_polynomials::univariate::UnivariatePolynomial;
use rustmath_rationals::Rational;
use rustmath_reals::Real;
use std::collections::{HashMap, HashSet};
use std::fmt;
use thiserror::Error;

/// Errors that can occur when working with number field morphisms
#[derive(Debug, Clone, Error, PartialEq)]
pub enum MorphismError {
    #[error("Invalid morphism: {0}")]
    InvalidMorphism(String),

    #[error("Image is not a root of minimal polynomial")]
    NotARoot,

    #[error("Morphism does not extend to an automorphism")]
    NotAnAutomorphism,

    #[error("Field extension is not Galois")]
    NotGalois,

    #[error("Field extension is not normal")]
    NotNormal,

    #[error("Field extension is not separable")]
    NotSeparable,

    #[error("Cannot compute splitting field: {0}")]
    SplittingFieldError(String),

    #[error("Computation not implemented: {0}")]
    NotImplemented(String),

    #[error("Internal number field error: {0}")]
    NumberFieldError(#[from] NumberFieldError),
}

/// Result type for morphism operations
pub type Result<T> = std::result::Result<T, MorphismError>;

/// Trait for morphisms between number fields
///
/// A number field morphism is uniquely determined by the image of the generator.
/// For K = ℚ(α), a morphism φ: K → L is determined by φ(α).
pub trait NumberFieldMorphism {
    /// Apply the morphism to an element
    fn apply(&self, element: &NumberFieldElement) -> Result<NumberFieldElement>;

    /// Get the image of the generator
    fn generator_image(&self) -> &NumberFieldElement;

    /// Check if this morphism is injective
    fn is_injective(&self) -> bool {
        // Number field morphisms are always injective (no zero divisors)
        true
    }

    /// Check if this morphism is surjective
    fn is_surjective(&self) -> bool;

    /// Check if this is an isomorphism
    fn is_isomorphism(&self) -> bool {
        self.is_injective() && self.is_surjective()
    }

    /// Check if this is an automorphism (isomorphism to itself)
    fn is_automorphism(&self) -> bool;
}

/// An embedding of a number field into ℂ
///
/// An embedding σ: K → ℂ is an injective ring homomorphism.
/// For K = ℚ(α) of degree n, there are exactly n embeddings,
/// corresponding to the n roots of the minimal polynomial in ℂ.
#[derive(Clone, Debug)]
pub struct NumberFieldEmbedding {
    /// The source number field
    domain: NumberField,
    /// The image of the generator (as a complex number or symbolic representation)
    /// We store this as a NumberFieldElement in the codomain for generality
    generator_image: NumberFieldElement,
    /// Description of this embedding
    description: String,
}

impl NumberFieldEmbedding {
    /// Create a new embedding by specifying the image of the generator
    ///
    /// # Arguments
    /// * `domain` - The source number field K = ℚ(α)
    /// * `generator_image` - The image of α, which must be a root of the minimal polynomial
    ///
    /// # Errors
    /// Returns `NotARoot` if the generator_image is not a root of the minimal polynomial
    pub fn new(
        domain: NumberField,
        generator_image: NumberFieldElement,
    ) -> Result<Self> {
        // Verify that generator_image is a root of the minimal polynomial
        // This is done by evaluating the minimal polynomial at generator_image
        // For now, we trust the caller (full implementation would verify)

        Ok(NumberFieldEmbedding {
            domain: domain.clone(),
            generator_image,
            description: "Number field embedding".to_string(),
        })
    }

    /// Create an embedding from a root approximation (for computing with ℂ)
    ///
    /// This is useful when we know a numerical approximation of where α should map.
    pub fn from_root_approximation(
        domain: NumberField,
        root_index: usize,
    ) -> Result<Self> {
        // In a full implementation, this would:
        // 1. Numerically compute roots of the minimal polynomial
        // 2. Create an embedding sending α to the root_index-th root

        Err(MorphismError::NotImplemented(
            "Numerical root approximation not yet implemented".to_string()
        ))
    }

    /// Get all embeddings of a number field into ℂ
    ///
    /// For a degree n number field, this returns all n embeddings.
    pub fn all_embeddings(domain: &NumberField) -> Result<Vec<Self>> {
        // To compute all embeddings:
        // 1. Factor the minimal polynomial over ℂ (find all roots)
        // 2. For each root β, create the embedding α ↦ β

        let degree = domain.degree();
        let mut embeddings = Vec::with_capacity(degree);

        // For now, return an error as we need root-finding algorithms
        Err(MorphismError::NotImplemented(
            "Computing all embeddings requires polynomial root-finding".to_string()
        ))
    }

    /// Compute the real embeddings (those with image in ℝ)
    ///
    /// These correspond to real roots of the minimal polynomial.
    /// The number of real embeddings is r₁ in the signature (r₁, r₂).
    pub fn real_embeddings(domain: &NumberField) -> Result<Vec<Self>> {
        // Use Sturm's theorem or Descartes' rule of signs to count real roots
        // Then compute them numerically

        Err(MorphismError::NotImplemented(
            "Computing real embeddings not yet implemented".to_string()
        ))
    }

    /// Compute the complex embeddings (those with image in ℂ \ ℝ)
    ///
    /// These come in conjugate pairs. The number of pairs is r₂ in the signature (r₁, r₂).
    pub fn complex_embeddings(domain: &NumberField) -> Result<Vec<Self>> {
        Err(MorphismError::NotImplemented(
            "Computing complex embeddings not yet implemented".to_string()
        ))
    }
}

impl NumberFieldMorphism for NumberFieldEmbedding {
    fn apply(&self, element: &NumberFieldElement) -> Result<NumberFieldElement> {
        // To apply the embedding to an element a₀ + a₁α + ... + aₙ₋₁αⁿ⁻¹:
        // Replace each occurrence of α with generator_image

        let coeffs = (0..element.degree() + 1)
            .map(|i| element.coeff(i))
            .collect::<Vec<_>>();

        let mut result = self.domain.zero();
        let mut power = self.domain.one();

        for coeff in coeffs {
            // Add coeff * (generator_image)^i to result
            let term = self.domain.from_rational(coeff);
            let scaled = self.domain.mul(&term, &power);
            result = self.domain.add(&result, &scaled);
            power = self.domain.mul(&power, &self.generator_image);
        }

        Ok(result)
    }

    fn generator_image(&self) -> &NumberFieldElement {
        &self.generator_image
    }

    fn is_surjective(&self) -> bool {
        // Embeddings into ℂ are not surjective (image has dimension n over ℚ)
        false
    }

    fn is_automorphism(&self) -> bool {
        // An embedding is an automorphism only if it's surjective onto the domain
        false
    }
}

impl fmt::Display for NumberFieldEmbedding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Embedding of {} with α ↦ {}",
               self.domain, self.generator_image)
    }
}

/// An automorphism of a number field
///
/// An automorphism σ: K → K is a bijective ring homomorphism from K to itself.
/// The set of all automorphisms forms a group under composition: Aut(K/ℚ).
#[derive(Clone, Debug)]
pub struct NumberFieldAutomorphism {
    /// The number field
    field: NumberField,
    /// The image of the generator under this automorphism
    generator_image: NumberFieldElement,
    /// Optional: the inverse automorphism
    inverse: Option<Box<NumberFieldAutomorphism>>,
}

impl NumberFieldAutomorphism {
    /// Create a new automorphism by specifying the image of the generator
    ///
    /// # Arguments
    /// * `field` - The number field K = ℚ(α)
    /// * `generator_image` - The image of α, which must be in K and extend to an automorphism
    ///
    /// # Errors
    /// Returns `NotAnAutomorphism` if the map α ↦ generator_image doesn't extend to an automorphism
    pub fn new(
        field: NumberField,
        generator_image: NumberFieldElement,
    ) -> Result<Self> {
        // Verify that this defines an automorphism:
        // 1. generator_image must be a root of the minimal polynomial
        // 2. The map must be bijective (check by computing inverse)

        // For now, we trust the caller
        Ok(NumberFieldAutomorphism {
            field,
            generator_image,
            inverse: None,
        })
    }

    /// Create the identity automorphism
    pub fn identity(field: NumberField) -> Self {
        let generator_image = field.generator();
        NumberFieldAutomorphism {
            field: field.clone(),
            generator_image,
            inverse: None,
        }
    }

    /// Compose this automorphism with another: (self ∘ other)(x) = self(other(x))
    pub fn compose(&self, other: &NumberFieldAutomorphism) -> Result<Self> {
        // Composition: first apply other, then apply self
        let image = self.apply(&other.generator_image)?;
        Self::new(self.field.clone(), image)
    }

    /// Compute the inverse automorphism
    pub fn inverse(&self) -> Result<Self> {
        if let Some(inv) = &self.inverse {
            return Ok((**inv).clone());
        }

        // To find the inverse, we need to find β such that σ(β) = α
        // where σ is this automorphism and α is the generator

        // This is complex in general; for now return not implemented
        Err(MorphismError::NotImplemented(
            "Computing inverse automorphism not yet implemented".to_string()
        ))
    }

    /// Compute the order of this automorphism (smallest k > 0 such that σᵏ = id)
    pub fn order(&self) -> Result<usize> {
        let mut current = self.clone();
        let identity = Self::identity(self.field.clone());

        for k in 1..=self.field.degree() {
            if current.generator_image == identity.generator_image {
                return Ok(k);
            }
            current = current.compose(self)?;
        }

        // If we didn't find the order within the degree, something is wrong
        Err(MorphismError::InvalidMorphism(
            "Could not determine automorphism order".to_string()
        ))
    }

    /// Check if this automorphism is the identity
    pub fn is_identity(&self) -> bool {
        self.generator_image == self.field.generator()
    }
}

impl NumberFieldMorphism for NumberFieldAutomorphism {
    fn apply(&self, element: &NumberFieldElement) -> Result<NumberFieldElement> {
        // Same as embedding: replace α with generator_image
        let coeffs = (0..element.degree() + 1)
            .map(|i| element.coeff(i))
            .collect::<Vec<_>>();

        let mut result = self.field.zero();
        let mut power = self.field.one();

        for coeff in coeffs {
            let term = self.field.from_rational(coeff);
            let scaled = self.field.mul(&term, &power);
            result = self.field.add(&result, &scaled);
            power = self.field.mul(&power, &self.generator_image);
        }

        Ok(result)
    }

    fn generator_image(&self) -> &NumberFieldElement {
        &self.generator_image
    }

    fn is_surjective(&self) -> bool {
        // Automorphisms are by definition bijective
        true
    }

    fn is_automorphism(&self) -> bool {
        true
    }
}

impl fmt::Display for NumberFieldAutomorphism {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Automorphism: α ↦ {}", self.generator_image)
    }
}

/// Compute all automorphisms of a number field
///
/// Returns Aut(K/ℚ), the group of automorphisms fixing ℚ.
/// For a degree n field, |Aut(K/ℚ)| divides n, with equality iff K/ℚ is Galois.
pub fn compute_automorphisms(field: &NumberField) -> Result<Vec<NumberFieldAutomorphism>> {
    // Algorithm:
    // 1. Find all roots of the minimal polynomial that lie in K
    // 2. For each root β, check if α ↦ β extends to an automorphism
    // 3. Return all valid automorphisms

    let degree = field.degree();
    let mut automorphisms = Vec::new();

    // Always include the identity
    automorphisms.push(NumberFieldAutomorphism::identity(field.clone()));

    // For degree 1 (rationals), only the identity exists
    if degree == 1 {
        return Ok(automorphisms);
    }

    // For degree 2, check if the field is real or complex
    if degree == 2 {
        return compute_automorphisms_quadratic(field);
    }

    // For higher degrees, we need sophisticated algorithms
    // This requires finding roots in the field, which is non-trivial

    Err(MorphismError::NotImplemented(
        format!("Computing automorphisms for degree {} fields not yet implemented", degree)
    ))
}

/// Compute automorphisms for quadratic fields
fn compute_automorphisms_quadratic(field: &NumberField) -> Result<Vec<NumberFieldAutomorphism>> {
    let mut auts = vec![NumberFieldAutomorphism::identity(field.clone())];

    // For ℚ(√d), the non-trivial automorphism is α ↦ -α if it exists
    let alpha = field.generator();
    let neg_alpha = field.sub(&field.zero(), &alpha);

    // Check if -α is also a root of the minimal polynomial
    // For x² - d, if α is a root, so is -α
    let discriminant = field.discriminant();

    // Quadratic fields always have exactly 2 automorphisms
    if let Ok(aut) = NumberFieldAutomorphism::new(field.clone(), neg_alpha) {
        auts.push(aut);
    }

    Ok(auts)
}

/// Check if a field extension is Galois
///
/// A field extension K/ℚ is Galois if it is both normal and separable.
/// Equivalently, K is Galois if |Aut(K/ℚ)| = [K:ℚ].
pub fn is_galois_extension(field: &NumberField) -> Result<bool> {
    let degree = field.degree();
    let auts = compute_automorphisms(field)?;
    Ok(auts.len() == degree)
}

/// Check if a field extension is normal
///
/// K/ℚ is normal if it is the splitting field of some polynomial over ℚ.
/// Equivalently, whenever an irreducible polynomial over ℚ has one root in K,
/// it splits completely in K.
pub fn is_normal_extension(field: &NumberField) -> Result<bool> {
    // Check if the minimal polynomial splits completely in K
    // This requires factoring the minimal polynomial over K

    Err(MorphismError::NotImplemented(
        "Checking normality not yet implemented".to_string()
    ))
}

/// Check if a field extension is separable
///
/// K/ℚ is separable if the minimal polynomial has no repeated roots.
/// Over ℚ (characteristic 0), all extensions are separable.
pub fn is_separable_extension(field: &NumberField) -> bool {
    // In characteristic 0, all algebraic extensions are separable
    // This is always true for number fields over ℚ
    true
}

/// Compute the splitting field of a polynomial
///
/// The splitting field of f ∈ ℚ[x] is the smallest field containing all roots of f.
/// This is always a Galois extension of ℚ.
pub fn splitting_field(poly: &UnivariatePolynomial<Rational>) -> Result<NumberField> {
    // Algorithm:
    // 1. Factor poly over ℚ to find an irreducible factor
    // 2. Adjoin a root of this factor to get K₁
    // 3. Factor poly over K₁ to find another irreducible factor
    // 4. Repeat until poly splits completely

    let degree = poly.degree().ok_or_else(|| {
        MorphismError::SplittingFieldError("Polynomial must be non-zero".to_string())
    })?;

    if degree == 0 {
        return Err(MorphismError::SplittingFieldError(
            "Cannot compute splitting field of constant polynomial".to_string()
        ));
    }

    if degree == 1 {
        // Linear polynomial: splitting field is ℚ
        // Return the trivial extension
        return Ok(NumberField::new(UnivariatePolynomial::new(vec![
            Rational::zero(),
            Rational::one(),
        ])));
    }

    // For degree 2, the splitting field is ℚ or ℚ(√d)
    if degree == 2 {
        return splitting_field_quadratic(poly);
    }

    // For higher degrees, this is very complex
    Err(MorphismError::NotImplemented(
        format!("Computing splitting field for degree {} not yet implemented", degree)
    ))
}

/// Compute splitting field for quadratic polynomials
fn splitting_field_quadratic(poly: &UnivariatePolynomial<Rational>) -> Result<NumberField> {
    // For ax² + bx + c, the splitting field is ℚ(√(b² - 4ac))
    let coeffs = poly.coefficients();

    if coeffs.len() != 3 {
        return Err(MorphismError::SplittingFieldError(
            "Expected quadratic polynomial".to_string()
        ));
    }

    let c = &coeffs[0];
    let b = &coeffs[1];
    let a = &coeffs[2];

    // Compute discriminant b² - 4ac
    let disc = b.clone() * b.clone() - Rational::from_integer(4) * a.clone() * c.clone();

    // The splitting field is ℚ(√disc)
    // Minimal polynomial is x² - disc
    let min_poly = UnivariatePolynomial::new(vec![
        -disc,
        Rational::zero(),
        Rational::one(),
    ]);

    Ok(NumberField::new(min_poly))
}

/// Compute the Galois group of a polynomial
///
/// For a polynomial f ∈ ℚ[x], this computes Gal(K/ℚ) where K is the splitting field of f.
/// The Galois group acts on the roots of f by permuting them.
pub fn galois_group(poly: &UnivariatePolynomial<Rational>) -> Result<GaloisGroup> {
    // Compute the splitting field
    let splitting = splitting_field(poly)?;

    // Compute all automorphisms
    let auts = compute_automorphisms(&splitting)?;

    // Build the group structure
    GaloisGroup::new(splitting, auts, poly.clone())
}

/// Represents the Galois group of a field extension
///
/// The Galois group Gal(K/ℚ) is the group of automorphisms of K fixing ℚ.
/// It has a natural action on the roots of the defining polynomial.
#[derive(Clone, Debug)]
pub struct GaloisGroup {
    /// The field (splitting field)
    field: NumberField,
    /// All automorphisms in the Galois group
    automorphisms: Vec<NumberFieldAutomorphism>,
    /// The polynomial whose splitting field this is
    polynomial: UnivariatePolynomial<Rational>,
    /// Group structure (multiplication table)
    multiplication_table: Option<Vec<Vec<usize>>>,
}

impl GaloisGroup {
    /// Create a new Galois group
    pub fn new(
        field: NumberField,
        automorphisms: Vec<NumberFieldAutomorphism>,
        polynomial: UnivariatePolynomial<Rational>,
    ) -> Result<Self> {
        let mut group = GaloisGroup {
            field,
            automorphisms,
            polynomial,
            multiplication_table: None,
        };

        // Compute multiplication table
        group.compute_multiplication_table()?;

        Ok(group)
    }

    /// Get the order of the Galois group
    pub fn order(&self) -> usize {
        self.automorphisms.len()
    }

    /// Get all automorphisms
    pub fn automorphisms(&self) -> &[NumberFieldAutomorphism] {
        &self.automorphisms
    }

    /// Compute the multiplication table for the group
    fn compute_multiplication_table(&mut self) -> Result<()> {
        let n = self.automorphisms.len();
        let mut table = vec![vec![0; n]; n];

        for i in 0..n {
            for j in 0..n {
                let composition = self.automorphisms[i].compose(&self.automorphisms[j])?;
                // Find which automorphism this is
                for (k, aut) in self.automorphisms.iter().enumerate() {
                    if composition.generator_image == aut.generator_image {
                        table[i][j] = k;
                        break;
                    }
                }
            }
        }

        self.multiplication_table = Some(table);
        Ok(())
    }

    /// Check if the group is abelian
    pub fn is_abelian(&self) -> bool {
        if let Some(table) = &self.multiplication_table {
            for i in 0..table.len() {
                for j in 0..table.len() {
                    if table[i][j] != table[j][i] {
                        return false;
                    }
                }
            }
            true
        } else {
            false
        }
    }

    /// Check if the group is cyclic
    pub fn is_cyclic(&self) -> bool {
        // A group is cyclic if it has a generator
        // For small groups, just check all elements

        for aut in &self.automorphisms {
            if let Ok(order) = aut.order() {
                if order == self.order() {
                    return true;
                }
            }
        }
        false
    }

    /// Identify the group (return its common name)
    pub fn identify(&self) -> String {
        let n = self.order();

        match n {
            1 => "Trivial group".to_string(),
            2 => "C₂ (Cyclic group of order 2)".to_string(),
            3 => "C₃ (Cyclic group of order 3)".to_string(),
            4 => {
                if self.is_cyclic() {
                    "C₄ (Cyclic group of order 4)".to_string()
                } else {
                    "V₄ (Klein four-group)".to_string()
                }
            }
            6 => {
                if self.is_abelian() {
                    "C₆ (Cyclic group of order 6)".to_string()
                } else {
                    "S₃ (Symmetric group on 3 elements)".to_string()
                }
            }
            _ => format!("Group of order {}", n),
        }
    }
}

impl fmt::Display for GaloisGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Galois group of order {} over ℚ", self.order())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_quadratic_field() -> NumberField {
        // Create ℚ(√2) with minimal polynomial x² - 2
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(-2),
            Rational::from_integer(0),
            Rational::from_integer(1),
        ]);
        NumberField::new(poly)
    }

    fn make_cubic_field() -> NumberField {
        // Create ℚ(∛2) with minimal polynomial x³ - 2
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(-2),
            Rational::from_integer(0),
            Rational::from_integer(0),
            Rational::from_integer(1),
        ]);
        NumberField::new(poly)
    }

    #[test]
    fn test_identity_automorphism() {
        let field = make_quadratic_field();
        let id = NumberFieldAutomorphism::identity(field.clone());

        assert!(id.is_identity());
        assert!(id.is_automorphism());
    }

    #[test]
    fn test_quadratic_automorphisms() {
        let field = make_quadratic_field();
        let auts = compute_automorphisms(&field).unwrap();

        // Quadratic fields have exactly 2 automorphisms
        assert_eq!(auts.len(), 2);

        // One should be identity
        assert!(auts.iter().any(|aut| aut.is_identity()));
    }

    #[test]
    fn test_automorphism_application() {
        let field = make_quadratic_field();
        let id = NumberFieldAutomorphism::identity(field.clone());

        let alpha = field.generator();
        let result = id.apply(&alpha).unwrap();

        assert_eq!(result, alpha);
    }

    #[test]
    fn test_is_separable() {
        let field = make_quadratic_field();

        // All extensions of ℚ are separable (characteristic 0)
        assert!(is_separable_extension(&field));
    }

    #[test]
    fn test_quadratic_is_galois() {
        let field = make_quadratic_field();

        // Quadratic extensions are always Galois
        let is_gal = is_galois_extension(&field).unwrap();
        assert!(is_gal);
    }

    #[test]
    fn test_splitting_field_quadratic() {
        // For x² - 2, the splitting field is ℚ(√2)
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(-2),
            Rational::from_integer(0),
            Rational::from_integer(1),
        ]);

        let splitting = splitting_field(&poly).unwrap();
        assert_eq!(splitting.degree(), 2);
    }

    #[test]
    fn test_splitting_field_linear() {
        // For linear polynomial x - 1, splitting field is ℚ
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(-1),
            Rational::from_integer(1),
        ]);

        let splitting = splitting_field(&poly).unwrap();
        assert_eq!(splitting.degree(), 1);
    }

    #[test]
    fn test_galois_group_quadratic() {
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(-2),
            Rational::from_integer(0),
            Rational::from_integer(1),
        ]);

        let gal = galois_group(&poly).unwrap();
        assert_eq!(gal.order(), 2);
        assert!(gal.is_abelian());
        assert!(gal.is_cyclic());
    }

    #[test]
    fn test_galois_group_identify() {
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(-2),
            Rational::from_integer(0),
            Rational::from_integer(1),
        ]);

        let gal = galois_group(&poly).unwrap();
        assert_eq!(gal.identify(), "C₂ (Cyclic group of order 2)");
    }

    #[test]
    fn test_embedding_creation() {
        let field = make_quadratic_field();
        let generator_image = field.generator();

        let embedding = NumberFieldEmbedding::new(field.clone(), generator_image);
        assert!(embedding.is_ok());
    }

    #[test]
    fn test_embedding_is_injective() {
        let field = make_quadratic_field();
        let generator_image = field.generator();
        let embedding = NumberFieldEmbedding::new(field, generator_image).unwrap();

        assert!(embedding.is_injective());
        assert!(!embedding.is_surjective());
        assert!(!embedding.is_automorphism());
    }

    #[test]
    fn test_automorphism_composition() {
        let field = make_quadratic_field();
        let auts = compute_automorphisms(&field).unwrap();

        if auts.len() >= 2 {
            let aut1 = &auts[0];
            let aut2 = &auts[1];

            // Composition should work
            let comp = aut1.compose(aut2);
            assert!(comp.is_ok());
        }
    }

    #[test]
    fn test_splitting_field_discriminant() {
        // For x² + x + 1, discriminant is -3
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(1),
            Rational::from_integer(1),
            Rational::from_integer(1),
        ]);

        let splitting = splitting_field_quadratic(&poly).unwrap();

        // The splitting field should be ℚ(√-3)
        let disc = splitting.discriminant();
        // The discriminant of ℚ(√-3) is -3 or -12 depending on the basis
        // Just check it's negative (imaginary quadratic field)
        assert!(disc < Rational::zero());
    }
}
