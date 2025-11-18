//! # Term Monoids for Asymptotic Analysis
//!
//! This module implements term monoids, which are the building blocks of asymptotic
//! expansions. A term monoid combines growth elements with coefficients to represent
//! individual terms in asymptotic series.
//!
//! ## Mathematical Background
//!
//! An asymptotic expansion is a sum of terms, where each term consists of:
//! - A growth element (e.g., n², e^n, log(n))
//! - A coefficient from some ring (e.g., rationals, reals)
//!
//! Term monoids provide the algebraic structure for manipulating these terms,
//! including multiplication, absorption, and comparison.
//!
//! ## Types of Terms
//!
//! - **GenericTerm**: Base term type without coefficient
//! - **OTerm**: Big O notation terms (O(n²))
//! - **ExactTerm**: Terms with exact coefficients (3n²)
//! - **BTerm**: Big Theta notation terms (Θ(n²))
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::term_monoid::{ExactTerm, ExactTermMonoid};
//! use rustmath_rings::growth_group::{MonomialGrowthGroup, Variable};
//! use num_rational::BigRational;
//! use num_bigint::BigInt;
//!
//! // Create a term monoid for exact terms over rationals
//! let var = Variable::new("n");
//! let growth_group = MonomialGrowthGroup::new(var);
//!
//! // Terms represent expressions like "3*n^2"
//! ```

use crate::growth_group::{GrowthElement, GrowthGroup, MonomialGrowthElement, Variable};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Zero, One, Signed};
use std::cmp::Ordering;
use std::fmt::{self, Debug, Display};
use std::marker::PhantomData;

// ======================================================================================
// ABSORPTION UTILITIES
// ======================================================================================

/// Determines if one term can absorb another in asymptotic notation.
///
/// In Big O notation, a term can absorb another if it grows at least as fast.
/// For example, O(n²) can absorb O(n) because n² dominates n.
///
/// # Arguments
/// * `growth1` - First growth element
/// * `growth2` - Second growth element
///
/// # Returns
/// True if the first can absorb the second
pub fn can_absorb<T: GrowthElement>(growth1: &T, growth2: &T) -> bool {
    matches!(
        growth1.compare_growth(growth2),
        Ordering::Greater | Ordering::Equal
    )
}

/// Determines the result of absorption between two terms.
///
/// Returns which term survives absorption, or indicates they cannot be combined.
///
/// # Arguments
/// * `growth1` - First growth element
/// * `growth2` - Second growth element
///
/// # Returns
/// * Some(true) if first absorbs second
/// * Some(false) if second absorbs first
/// * None if neither can absorb the other
pub fn absorption<T: GrowthElement>(growth1: &T, growth2: &T) -> Option<bool> {
    match growth1.compare_growth(growth2) {
        Ordering::Greater => Some(true),  // First absorbs second
        Ordering::Less => Some(false),    // Second absorbs first
        Ordering::Equal => Some(true),    // Either can absorb (return first)
    }
}

// ======================================================================================
// GENERIC TERM
// ======================================================================================

/// A generic term in an asymptotic expansion.
///
/// This is the base type representing a term without coefficient considerations.
/// It consists primarily of a growth element.
#[derive(Clone, Debug)]
pub struct GenericTerm<T: GrowthElement> {
    /// The growth element (e.g., n², log(n), e^n)
    growth: T,
}

impl<T: GrowthElement> GenericTerm<T> {
    /// Creates a new generic term.
    pub fn new(growth: T) -> Self {
        GenericTerm { growth }
    }

    /// Returns the growth element.
    pub fn growth(&self) -> &T {
        &self.growth
    }

    /// Multiplies two terms (multiplies their growth elements).
    pub fn multiply(&self, other: &Self) -> Self {
        GenericTerm {
            growth: self.growth.multiply(&other.growth),
        }
    }

    /// Computes the inverse of this term.
    pub fn inverse(&self) -> Self {
        GenericTerm {
            growth: self.growth.inverse(),
        }
    }

    /// Checks if this is the identity term.
    pub fn is_identity(&self) -> bool {
        self.growth.is_identity()
    }

    /// Raises this term to a power.
    pub fn pow(&self, exponent: &BigRational) -> Self {
        GenericTerm {
            growth: self.growth.pow(exponent),
        }
    }
}

impl<T: GrowthElement> PartialEq for GenericTerm<T> {
    fn eq(&self, other: &Self) -> bool {
        self.growth == other.growth
    }
}

impl<T: GrowthElement> Eq for GenericTerm<T> {}

impl<T: GrowthElement> PartialOrd for GenericTerm<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: GrowthElement> Ord for GenericTerm<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.growth.compare_growth(&other.growth)
    }
}

impl<T: GrowthElement> Display for GenericTerm<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.growth)
    }
}

// ======================================================================================
// O-TERM (Big O Notation)
// ======================================================================================

/// A term using Big O notation.
///
/// Represents O(growth), indicating an upper bound on growth rate.
/// For example: O(n²), O(log n), O(e^n).
#[derive(Clone, Debug)]
pub struct OTerm<T: GrowthElement> {
    /// The underlying generic term
    base: GenericTerm<T>,
}

impl<T: GrowthElement> OTerm<T> {
    /// Creates a new O-term.
    pub fn new(growth: T) -> Self {
        OTerm {
            base: GenericTerm::new(growth),
        }
    }

    /// Returns the growth element.
    pub fn growth(&self) -> &T {
        self.base.growth()
    }

    /// Multiplies two O-terms.
    ///
    /// O(f) * O(g) = O(f*g)
    pub fn multiply(&self, other: &Self) -> Self {
        OTerm {
            base: self.base.multiply(&other.base),
        }
    }

    /// Attempts to absorb another O-term.
    ///
    /// O(f) absorbs O(g) if f dominates g.
    /// Returns Some(result) if absorption occurs, None otherwise.
    pub fn absorb(&self, other: &Self) -> Option<Self> {
        match absorption(self.growth(), other.growth()) {
            Some(true) => Some(self.clone()),  // Self absorbs other
            Some(false) => Some(other.clone()), // Other absorbs self
            None => None,                      // Cannot absorb
        }
    }
}

impl<T: GrowthElement> PartialEq for OTerm<T> {
    fn eq(&self, other: &Self) -> bool {
        self.base == other.base
    }
}

impl<T: GrowthElement> Eq for OTerm<T> {}

impl<T: GrowthElement> PartialOrd for OTerm<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: GrowthElement> Ord for OTerm<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.base.cmp(&other.base)
    }
}

impl<T: GrowthElement> Display for OTerm<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "O({})", self.base)
    }
}

// ======================================================================================
// TERM WITH COEFFICIENT
// ======================================================================================

/// A term with an explicit coefficient.
///
/// Represents coefficient * growth, where the coefficient comes from a ring
/// (typically rationals or a field).
#[derive(Clone, Debug)]
pub struct TermWithCoefficient<T: GrowthElement> {
    /// The coefficient
    coefficient: BigRational,
    /// The growth element
    growth: T,
}

impl<T: GrowthElement> TermWithCoefficient<T> {
    /// Creates a new term with coefficient.
    pub fn new(coefficient: BigRational, growth: T) -> Self {
        TermWithCoefficient { coefficient, growth }
    }

    /// Returns the coefficient.
    pub fn coefficient(&self) -> &BigRational {
        &self.coefficient
    }

    /// Returns the growth element.
    pub fn growth(&self) -> &T {
        &self.growth
    }

    /// Checks if the coefficient is zero.
    pub fn is_zero(&self) -> bool {
        self.coefficient.is_zero()
    }

    /// Multiplies two terms with coefficients.
    ///
    /// (c₁ * g₁) * (c₂ * g₂) = (c₁ * c₂) * (g₁ * g₂)
    pub fn multiply(&self, other: &Self) -> Self {
        TermWithCoefficient {
            coefficient: &self.coefficient * &other.coefficient,
            growth: self.growth.multiply(&other.growth),
        }
    }

    /// Scales this term by a scalar.
    pub fn scale(&self, scalar: &BigRational) -> Self {
        TermWithCoefficient {
            coefficient: &self.coefficient * scalar,
            growth: self.growth.clone(),
        }
    }

    /// Computes the inverse of this term.
    pub fn inverse(&self) -> Self {
        TermWithCoefficient {
            coefficient: BigRational::one() / &self.coefficient,
            growth: self.growth.inverse(),
        }
    }
}

impl<T: GrowthElement> PartialEq for TermWithCoefficient<T> {
    fn eq(&self, other: &Self) -> bool {
        self.coefficient == other.coefficient && self.growth == other.growth
    }
}

impl<T: GrowthElement> Eq for TermWithCoefficient<T> {}

impl<T: GrowthElement> PartialOrd for TermWithCoefficient<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: GrowthElement> Ord for TermWithCoefficient<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Compare by growth first
        match self.growth.compare_growth(&other.growth) {
            Ordering::Equal => {
                // If growth is equal, compare coefficients
                self.coefficient.cmp(&other.coefficient)
            }
            ord => ord,
        }
    }
}

impl<T: GrowthElement> Display for TermWithCoefficient<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.coefficient.is_one() && !self.growth.is_identity() {
            write!(f, "{}", self.growth)
        } else if self.growth.is_identity() {
            write!(f, "{}", self.coefficient)
        } else {
            write!(f, "{} * {}", self.coefficient, self.growth)
        }
    }
}

// ======================================================================================
// EXACT TERM
// ======================================================================================

/// An exact term with precise coefficient.
///
/// Represents a term with an exact coefficient value, as opposed to
/// asymptotic bounds. For example: 3n², (1/2)e^n, 5log(n).
pub type ExactTerm<T> = TermWithCoefficient<T>;

// ======================================================================================
// B-TERM (Big Theta Notation)
// ======================================================================================

/// A term using Big Theta notation.
///
/// Represents Θ(growth), indicating a tight asymptotic bound.
/// For example: Θ(n²), Θ(log n), Θ(e^n).
#[derive(Clone, Debug)]
pub struct BTerm<T: GrowthElement> {
    /// The growth element (coefficient is implicitly 1 for Theta notation)
    growth: T,
}

impl<T: GrowthElement> BTerm<T> {
    /// Creates a new B-term (Theta term).
    pub fn new(growth: T) -> Self {
        BTerm { growth }
    }

    /// Returns the growth element.
    pub fn growth(&self) -> &T {
        &self.growth
    }

    /// Multiplies two B-terms.
    ///
    /// Θ(f) * Θ(g) = Θ(f*g)
    pub fn multiply(&self, other: &Self) -> Self {
        BTerm {
            growth: self.growth.multiply(&other.growth),
        }
    }
}

impl<T: GrowthElement> PartialEq for BTerm<T> {
    fn eq(&self, other: &Self) -> bool {
        self.growth == other.growth
    }
}

impl<T: GrowthElement> Eq for BTerm<T> {}

impl<T: GrowthElement> Display for BTerm<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Θ({})", self.growth)
    }
}

// ======================================================================================
// TERM MONOIDS
// ======================================================================================

/// Base trait for term monoids.
///
/// A term monoid is a monoid whose elements are terms (growth + possibly coefficient).
pub trait TermMonoid {
    /// The term type
    type Term;

    /// Returns a description of this monoid.
    fn description(&self) -> String;

    /// Creates the identity element.
    fn identity(&self) -> Self::Term;

    /// Checks if a term belongs to this monoid.
    fn contains(&self, term: &Self::Term) -> bool;
}

/// Generic term monoid.
///
/// The monoid of generic terms over a growth group.
#[derive(Clone, Debug)]
pub struct GenericTermMonoid<T: GrowthElement> {
    /// Description of the monoid
    description: String,
    /// Phantom data for the term type
    _phantom: PhantomData<T>,
}

impl<T: GrowthElement> GenericTermMonoid<T> {
    /// Creates a new generic term monoid.
    pub fn new(description: String) -> Self {
        GenericTermMonoid {
            description,
            _phantom: PhantomData,
        }
    }
}

impl<T: GrowthElement> TermMonoid for GenericTermMonoid<T> {
    type Term = GenericTerm<T>;

    fn description(&self) -> String {
        self.description.clone()
    }

    fn identity(&self) -> Self::Term {
        // Would need access to a growth group to create proper identity
        // This is a simplified implementation
        unimplemented!("GenericTermMonoid::identity requires growth group")
    }

    fn contains(&self, _term: &Self::Term) -> bool {
        true // Generic monoid contains all terms
    }
}

/// O-term monoid for Big O notation terms.
#[derive(Clone, Debug)]
pub struct OTermMonoid<T: GrowthElement> {
    description: String,
    _phantom: PhantomData<T>,
}

impl<T: GrowthElement> OTermMonoid<T> {
    /// Creates a new O-term monoid.
    pub fn new(description: String) -> Self {
        OTermMonoid {
            description,
            _phantom: PhantomData,
        }
    }
}

impl<T: GrowthElement> TermMonoid for OTermMonoid<T> {
    type Term = OTerm<T>;

    fn description(&self) -> String {
        format!("O-Term Monoid: {}", self.description)
    }

    fn identity(&self) -> Self::Term {
        unimplemented!("OTermMonoid::identity requires growth group")
    }

    fn contains(&self, _term: &Self::Term) -> bool {
        true
    }
}

/// Term monoid for terms with coefficients.
#[derive(Clone, Debug)]
pub struct TermWithCoefficientMonoid<T: GrowthElement> {
    description: String,
    /// The coefficient ring name (e.g., "QQ" for rationals)
    coefficient_ring: String,
    _phantom: PhantomData<T>,
}

impl<T: GrowthElement> TermWithCoefficientMonoid<T> {
    /// Creates a new term-with-coefficient monoid.
    pub fn new(description: String, coefficient_ring: String) -> Self {
        TermWithCoefficientMonoid {
            description,
            coefficient_ring,
            _phantom: PhantomData,
        }
    }

    /// Returns the coefficient ring.
    pub fn coefficient_ring(&self) -> &str {
        &self.coefficient_ring
    }
}

impl<T: GrowthElement> TermMonoid for TermWithCoefficientMonoid<T> {
    type Term = TermWithCoefficient<T>;

    fn description(&self) -> String {
        format!(
            "Term Monoid with coefficients in {}: {}",
            self.coefficient_ring, self.description
        )
    }

    fn identity(&self) -> Self::Term {
        unimplemented!("TermWithCoefficientMonoid::identity requires growth group")
    }

    fn contains(&self, _term: &Self::Term) -> bool {
        true
    }
}

/// Exact term monoid for terms with exact coefficients.
pub type ExactTermMonoid<T> = TermWithCoefficientMonoid<T>;

/// B-term monoid for Big Theta notation terms.
#[derive(Clone, Debug)]
pub struct BTermMonoid<T: GrowthElement> {
    description: String,
    _phantom: PhantomData<T>,
}

impl<T: GrowthElement> BTermMonoid<T> {
    /// Creates a new B-term (Theta) monoid.
    pub fn new(description: String) -> Self {
        BTermMonoid {
            description,
            _phantom: PhantomData,
        }
    }
}

impl<T: GrowthElement> TermMonoid for BTermMonoid<T> {
    type Term = BTerm<T>;

    fn description(&self) -> String {
        format!("Θ-Term Monoid: {}", self.description)
    }

    fn identity(&self) -> Self::Term {
        unimplemented!("BTermMonoid::identity requires growth group")
    }

    fn contains(&self, _term: &Self::Term) -> bool {
        true
    }
}

// ======================================================================================
// TERM MONOID FACTORY
// ======================================================================================

/// Factory for creating various types of term monoids.
///
/// Provides a centralized interface for constructing term monoids with
/// different characteristics (O-terms, exact terms, B-terms, etc.).
#[derive(Clone, Debug)]
pub struct TermMonoidFactory {
    /// Default coefficient ring (e.g., "QQ" for rationals)
    default_coefficient_ring: String,
}

impl TermMonoidFactory {
    /// Creates a new term monoid factory.
    pub fn new() -> Self {
        TermMonoidFactory {
            default_coefficient_ring: "QQ".to_string(),
        }
    }

    /// Creates a new factory with a specified coefficient ring.
    pub fn with_coefficient_ring(coefficient_ring: String) -> Self {
        TermMonoidFactory {
            default_coefficient_ring: coefficient_ring,
        }
    }

    /// Creates an O-term monoid.
    pub fn o_term_monoid<T: GrowthElement>(&self, description: String) -> OTermMonoid<T> {
        OTermMonoid::new(description)
    }

    /// Creates an exact term monoid.
    pub fn exact_term_monoid<T: GrowthElement>(
        &self,
        description: String,
    ) -> ExactTermMonoid<T> {
        ExactTermMonoid::new(description, self.default_coefficient_ring.clone())
    }

    /// Creates a B-term monoid.
    pub fn b_term_monoid<T: GrowthElement>(&self, description: String) -> BTermMonoid<T> {
        BTermMonoid::new(description)
    }

    /// Creates a generic term monoid.
    pub fn generic_term_monoid<T: GrowthElement>(
        &self,
        description: String,
    ) -> GenericTermMonoid<T> {
        GenericTermMonoid::new(description)
    }

    /// Returns the default coefficient ring.
    pub fn coefficient_ring(&self) -> &str {
        &self.default_coefficient_ring
    }
}

impl Default for TermMonoidFactory {
    fn default() -> Self {
        Self::new()
    }
}

// ======================================================================================
// TESTS
// ======================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::growth_group::{MonomialGrowthElement, Variable};

    fn create_growth(var: &Variable, exp: i64) -> MonomialGrowthElement {
        MonomialGrowthElement::new(var.clone(), BigRational::from_integer(BigInt::from(exp)))
    }

    #[test]
    fn test_can_absorb() {
        let var = Variable::new("n");
        let g1 = create_growth(&var, 2); // n^2
        let g2 = create_growth(&var, 1); // n

        assert!(can_absorb(&g1, &g2)); // n^2 can absorb n
        assert!(!can_absorb(&g2, &g1)); // n cannot absorb n^2
    }

    #[test]
    fn test_absorption() {
        let var = Variable::new("n");
        let g1 = create_growth(&var, 2);
        let g2 = create_growth(&var, 1);

        assert_eq!(absorption(&g1, &g2), Some(true)); // g1 absorbs g2
        assert_eq!(absorption(&g2, &g1), Some(false)); // g2 doesn't absorb g1
        assert_eq!(absorption(&g1, &g1), Some(true)); // Equal elements
    }

    #[test]
    fn test_generic_term() {
        let var = Variable::new("n");
        let growth = create_growth(&var, 2);
        let term = GenericTerm::new(growth);

        assert!(!term.is_identity());
        assert_eq!(format!("{}", term), "n^2");
    }

    #[test]
    fn test_generic_term_multiplication() {
        let var = Variable::new("n");
        let t1 = GenericTerm::new(create_growth(&var, 2));
        let t2 = GenericTerm::new(create_growth(&var, 3));

        let product = t1.multiply(&t2);
        assert_eq!(product.growth().exponent(), &BigRational::from_integer(BigInt::from(5)));
    }

    #[test]
    fn test_o_term() {
        let var = Variable::new("n");
        let growth = create_growth(&var, 2);
        let oterm = OTerm::new(growth);

        assert_eq!(format!("{}", oterm), "O(n^2)");
    }

    #[test]
    fn test_o_term_absorption() {
        let var = Variable::new("n");
        let o1 = OTerm::new(create_growth(&var, 2)); // O(n^2)
        let o2 = OTerm::new(create_growth(&var, 1)); // O(n)

        let result = o1.absorb(&o2);
        assert!(result.is_some());
        // O(n^2) absorbs O(n), so result should be O(n^2)
    }

    #[test]
    fn test_term_with_coefficient() {
        let var = Variable::new("n");
        let growth = create_growth(&var, 2);
        let coeff = BigRational::from_integer(BigInt::from(3));
        let term = TermWithCoefficient::new(coeff, growth);

        assert_eq!(term.coefficient(), &BigRational::from_integer(BigInt::from(3)));
        assert!(!term.is_zero());
    }

    #[test]
    fn test_term_with_coefficient_multiplication() {
        let var = Variable::new("n");
        let t1 = TermWithCoefficient::new(
            BigRational::from_integer(BigInt::from(3)),
            create_growth(&var, 2),
        );
        let t2 = TermWithCoefficient::new(
            BigRational::from_integer(BigInt::from(2)),
            create_growth(&var, 1),
        );

        let product = t1.multiply(&t2);
        assert_eq!(
            product.coefficient(),
            &BigRational::from_integer(BigInt::from(6))
        );
        assert_eq!(
            product.growth().exponent(),
            &BigRational::from_integer(BigInt::from(3))
        );
    }

    #[test]
    fn test_term_with_coefficient_scale() {
        let var = Variable::new("n");
        let term = TermWithCoefficient::new(
            BigRational::from_integer(BigInt::from(3)),
            create_growth(&var, 2),
        );

        let scaled = term.scale(&BigRational::from_integer(BigInt::from(2)));
        assert_eq!(
            scaled.coefficient(),
            &BigRational::from_integer(BigInt::from(6))
        );
    }

    #[test]
    fn test_b_term() {
        let var = Variable::new("n");
        let growth = create_growth(&var, 2);
        let bterm = BTerm::new(growth);

        assert_eq!(format!("{}", bterm), "Θ(n^2)");
    }

    #[test]
    fn test_b_term_multiplication() {
        let var = Variable::new("n");
        let b1 = BTerm::new(create_growth(&var, 2));
        let b2 = BTerm::new(create_growth(&var, 1));

        let product = b1.multiply(&b2);
        assert_eq!(
            product.growth().exponent(),
            &BigRational::from_integer(BigInt::from(3))
        );
    }

    #[test]
    fn test_term_monoid_factory() {
        let factory = TermMonoidFactory::new();
        assert_eq!(factory.coefficient_ring(), "QQ");

        let o_monoid: OTermMonoid<MonomialGrowthElement> =
            factory.o_term_monoid("test".to_string());
        assert!(o_monoid.description().contains("O-Term"));

        let exact_monoid: ExactTermMonoid<MonomialGrowthElement> =
            factory.exact_term_monoid("test".to_string());
        assert!(exact_monoid.description().contains("coefficients"));
    }

    #[test]
    fn test_term_comparison() {
        let var = Variable::new("n");
        let t1 = GenericTerm::new(create_growth(&var, 2));
        let t2 = GenericTerm::new(create_growth(&var, 3));

        assert!(t1 < t2);
        assert!(t2 > t1);
        assert_eq!(t1, t1);
    }

    #[test]
    fn test_term_display() {
        let var = Variable::new("n");

        let term1 = TermWithCoefficient::new(
            BigRational::one(),
            create_growth(&var, 2),
        );
        assert_eq!(format!("{}", term1), "n^2");

        let term2 = TermWithCoefficient::new(
            BigRational::from_integer(BigInt::from(3)),
            create_growth(&var, 2),
        );
        let display = format!("{}", term2);
        assert!(display.contains("3"));
        assert!(display.contains("n"));
    }
}
