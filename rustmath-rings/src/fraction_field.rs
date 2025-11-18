//! Fraction fields (fields of fractions) for integral domains
//!
//! Given an integral domain R, its fraction field (or field of fractions) is the
//! smallest field containing R, denoted Frac(R). Elements are equivalence classes
//! of pairs (a, b) with a, b ∈ R and b ≠ 0, under the equivalence relation
//! (a, b) ~ (c, d) iff ad = bc.
//!
//! Examples:
//! - Frac(ℤ) = ℚ (rationals are fractions of integers)
//! - Frac(k[x]) = k(x) (rational functions over field k)

use rustmath_core::{Ring, IntegralDomain, Field};
use std::fmt;
use std::marker::PhantomData;
use std::hash::{Hash, Hasher};

/// Element of a fraction field
///
/// Represents the fraction a/b where a, b are elements of an integral domain R
#[derive(Clone, Debug)]
pub struct FractionFieldElement<R: IntegralDomain> {
    numerator: R,
    denominator: R,
}

impl<R: IntegralDomain> FractionFieldElement<R> {
    /// Create a new fraction field element
    ///
    /// # Panics
    /// Panics if denominator is zero
    pub fn new(numerator: R, denominator: R) -> Self {
        if denominator.is_zero() {
            panic!("Denominator cannot be zero");
        }

        // Simplify if possible (for Euclidean domains)
        let mut frac = FractionFieldElement {
            numerator,
            denominator,
        };
        frac.reduce();
        frac
    }

    /// Get the numerator
    pub fn numerator(&self) -> &R {
        &self.numerator
    }

    /// Get the denominator
    pub fn denominator(&self) -> &R {
        &self.denominator
    }

    /// Reduce the fraction to lowest terms (if R is a Euclidean domain)
    fn reduce(&mut self) {
        // For general integral domains, we can't always compute GCD
        // This is a placeholder that would need specialization for Euclidean domains
    }

    /// Check if this fraction represents zero
    pub fn is_zero(&self) -> bool {
        self.numerator.is_zero()
    }

    /// Check if this fraction represents one
    pub fn is_one(&self) -> bool {
        self.numerator == self.denominator
    }
}

impl<R: IntegralDomain> std::ops::Add for FractionFieldElement<R> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        // a/b + c/d = (ad + bc)/(bd)
        let num = self.numerator.clone() * other.denominator.clone()
            + other.numerator.clone() * self.denominator.clone();
        let den = self.denominator.clone() * other.denominator.clone();

        FractionFieldElement::new(num, den)
    }
}

impl<R: IntegralDomain> std::ops::Sub for FractionFieldElement<R> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        // a/b - c/d = (ad - bc)/(bd)
        let num = self.numerator.clone() * other.denominator.clone()
            - other.numerator.clone() * self.denominator.clone();
        let den = self.denominator.clone() * other.denominator.clone();

        FractionFieldElement::new(num, den)
    }
}

impl<R: IntegralDomain> std::ops::Mul for FractionFieldElement<R> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        // (a/b) * (c/d) = (ac)/(bd)
        let num = self.numerator.clone() * other.numerator.clone();
        let den = self.denominator.clone() * other.denominator.clone();

        FractionFieldElement::new(num, den)
    }
}

impl<R: IntegralDomain> std::ops::Div for FractionFieldElement<R> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        if other.numerator.is_zero() {
            panic!("Division by zero");
        }

        // (a/b) / (c/d) = (ad)/(bc)
        let num = self.numerator.clone() * other.denominator.clone();
        let den = self.denominator.clone() * other.numerator.clone();

        FractionFieldElement::new(num, den)
    }
}

impl<R: IntegralDomain> std::ops::Neg for FractionFieldElement<R> {
    type Output = Self;

    fn neg(self) -> Self {
        FractionFieldElement {
            numerator: -self.numerator,
            denominator: self.denominator,
        }
    }
}

impl<R: IntegralDomain> PartialEq for FractionFieldElement<R> {
    fn eq(&self, other: &Self) -> bool {
        // a/b == c/d iff ad == bc
        self.numerator.clone() * other.denominator.clone()
            == other.numerator.clone() * self.denominator.clone()
    }
}

impl<R: IntegralDomain> Eq for FractionFieldElement<R> {}

impl<R: IntegralDomain> fmt::Display for FractionFieldElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denominator.is_one() {
            write!(f, "{}", self.numerator)
        } else {
            write!(f, "{}/{}", self.numerator, self.denominator)
        }
    }
}

/// Generic fraction field structure
///
/// Represents Frac(R) for an integral domain R
#[derive(Clone, Debug)]
pub struct FractionField<R: IntegralDomain> {
    base_ring: PhantomData<R>,
}

impl<R: IntegralDomain> FractionField<R> {
    /// Create a new fraction field over the base ring R
    pub fn new() -> Self {
        FractionField {
            base_ring: PhantomData,
        }
    }

    /// Create a fraction field element from numerator and denominator
    pub fn element(&self, numerator: R, denominator: R) -> FractionFieldElement<R> {
        FractionFieldElement::new(numerator, denominator)
    }

    /// Get the zero element
    pub fn zero(&self) -> FractionFieldElement<R> {
        FractionFieldElement::new(R::zero(), R::one())
    }

    /// Get the one element
    pub fn one(&self) -> FractionFieldElement<R> {
        FractionFieldElement::new(R::one(), R::one())
    }

    /// Embed an element from the base ring into the fraction field
    pub fn embed(&self, x: R) -> FractionFieldElement<R> {
        FractionFieldElement::new(x, R::one())
    }
}

impl<R: IntegralDomain> Default for FractionField<R> {
    fn default() -> Self {
        Self::new()
    }
}

/// Embedding of base ring into fraction field
///
/// The natural embedding R → Frac(R) given by r ↦ r/1
pub struct FractionFieldEmbedding<R: IntegralDomain> {
    field: FractionField<R>,
}

impl<R: IntegralDomain> FractionFieldEmbedding<R> {
    /// Create a new embedding
    pub fn new(field: FractionField<R>) -> Self {
        FractionFieldEmbedding { field }
    }

    /// Apply the embedding to an element
    pub fn apply(&self, x: R) -> FractionFieldElement<R> {
        self.field.embed(x)
    }
}

/// Section of the fraction field embedding
///
/// Attempts to map elements of Frac(R) back to R when possible
/// (i.e., when the denominator divides the numerator in R)
pub struct FractionFieldEmbeddingSection<R: IntegralDomain> {
    _phantom: PhantomData<R>,
}

impl<R: IntegralDomain> FractionFieldEmbeddingSection<R> {
    /// Create a new section
    pub fn new() -> Self {
        FractionFieldEmbeddingSection {
            _phantom: PhantomData,
        }
    }

    /// Attempt to map a fraction field element back to the base ring
    ///
    /// Returns Some(r) if the element is in the image of the embedding,
    /// None otherwise
    pub fn apply(&self, x: &FractionFieldElement<R>) -> Option<R> {
        // Check if denominator is 1 (simplified check)
        if x.denominator.is_one() {
            Some(x.numerator.clone())
        } else {
            None
        }
    }
}

impl<R: IntegralDomain> Default for FractionFieldEmbeddingSection<R> {
    fn default() -> Self {
        Self::new()
    }
}

/// Specialized fraction field for univariate polynomial rings
///
/// Represents k(x) = Frac(k[x]) for a field k
pub struct FractionField1Poly<R: IntegralDomain> {
    base: FractionField<R>,
}

impl<R: IntegralDomain> FractionField1Poly<R> {
    /// Create a new fraction field for polynomials
    pub fn new() -> Self {
        FractionField1Poly {
            base: FractionField::new(),
        }
    }

    /// Create a rational function from numerator and denominator polynomials
    pub fn rational_function(&self, num: R, den: R) -> FractionFieldElement<R> {
        self.base.element(num, den)
    }

    /// Get the base fraction field
    pub fn base(&self) -> &FractionField<R> {
        &self.base
    }
}

impl<R: IntegralDomain> Default for FractionField1Poly<R> {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if a type is a fraction field element
pub fn is_fraction_field_element<R: IntegralDomain>(x: &FractionFieldElement<R>) -> bool {
    true
}

/// Constructor function for creating fraction fields
///
/// This is the main entry point for creating fraction fields
pub fn fraction_field<R: IntegralDomain>() -> FractionField<R> {
    FractionField::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_fraction_field_creation() {
        let field: FractionField<Integer> = FractionField::new();
        let zero = field.zero();
        let one = field.one();

        assert!(zero.is_zero());
        assert!(one.is_one());
    }

    #[test]
    fn test_fraction_element_creation() {
        let frac = FractionFieldElement::new(Integer::from(3), Integer::from(4));

        assert_eq!(frac.numerator(), &Integer::from(3));
        assert_eq!(frac.denominator(), &Integer::from(4));
    }

    #[test]
    #[should_panic(expected = "Denominator cannot be zero")]
    fn test_fraction_zero_denominator() {
        let _ = FractionFieldElement::new(Integer::from(1), Integer::zero());
    }

    #[test]
    fn test_fraction_addition() {
        // 1/2 + 1/3 = 5/6
        let a = FractionFieldElement::new(Integer::from(1), Integer::from(2));
        let b = FractionFieldElement::new(Integer::from(1), Integer::from(3));

        let sum = a + b;

        // Check: numerator should be 5, denominator should be 6
        // Note: might not be reduced, so we check the cross-product
        let expected = FractionFieldElement::new(Integer::from(5), Integer::from(6));
        assert_eq!(sum, expected);
    }

    #[test]
    fn test_fraction_subtraction() {
        // 3/4 - 1/2 = 1/4
        let a = FractionFieldElement::new(Integer::from(3), Integer::from(4));
        let b = FractionFieldElement::new(Integer::from(1), Integer::from(2));

        let diff = a - b;
        let expected = FractionFieldElement::new(Integer::from(1), Integer::from(4));

        assert_eq!(diff, expected);
    }

    #[test]
    fn test_fraction_multiplication() {
        // 2/3 * 3/4 = 1/2
        let a = FractionFieldElement::new(Integer::from(2), Integer::from(3));
        let b = FractionFieldElement::new(Integer::from(3), Integer::from(4));

        let product = a * b;
        let expected = FractionFieldElement::new(Integer::from(1), Integer::from(2));

        assert_eq!(product, expected);
    }

    #[test]
    fn test_fraction_division() {
        // (2/3) / (4/5) = 10/12 = 5/6
        let a = FractionFieldElement::new(Integer::from(2), Integer::from(3));
        let b = FractionFieldElement::new(Integer::from(4), Integer::from(5));

        let quotient = a / b;
        let expected = FractionFieldElement::new(Integer::from(5), Integer::from(6));

        assert_eq!(quotient, expected);
    }

    #[test]
    fn test_fraction_negation() {
        let a = FractionFieldElement::new(Integer::from(3), Integer::from(4));
        let neg = -a;

        assert_eq!(neg.numerator(), &Integer::from(-3));
        assert_eq!(neg.denominator(), &Integer::from(4));
    }

    #[test]
    fn test_fraction_equality() {
        // 2/4 == 1/2
        let a = FractionFieldElement::new(Integer::from(2), Integer::from(4));
        let b = FractionFieldElement::new(Integer::from(1), Integer::from(2));

        assert_eq!(a, b);

        // 1/2 != 1/3
        let c = FractionFieldElement::new(Integer::from(1), Integer::from(3));
        assert_ne!(a, c);
    }

    #[test]
    fn test_fraction_field_embedding() {
        let field: FractionField<Integer> = FractionField::new();
        let embedding = FractionFieldEmbedding::new(field);

        let x = Integer::from(5);
        let embedded = embedding.apply(x.clone());

        assert_eq!(embedded.numerator(), &x);
        assert_eq!(embedded.denominator(), &Integer::one());
    }

    #[test]
    fn test_fraction_field_section() {
        let section: FractionFieldEmbeddingSection<Integer> = FractionFieldEmbeddingSection::new();

        // 5/1 should map back to 5
        let a = FractionFieldElement::new(Integer::from(5), Integer::from(1));
        assert_eq!(section.apply(&a), Some(Integer::from(5)));

        // 5/2 should not map back (denominator != 1)
        let b = FractionFieldElement::new(Integer::from(5), Integer::from(2));
        assert_eq!(section.apply(&b), None);
    }

    #[test]
    fn test_fraction_field_1poly() {
        let field: FractionField1Poly<Integer> = FractionField1Poly::new();

        // Create a "rational function" (here just with integers for simplicity)
        let rf = field.rational_function(Integer::from(3), Integer::from(4));

        assert_eq!(rf.numerator(), &Integer::from(3));
        assert_eq!(rf.denominator(), &Integer::from(4));
    }

    #[test]
    fn test_is_fraction_field_element() {
        let frac = FractionFieldElement::new(Integer::from(1), Integer::from(2));
        assert!(is_fraction_field_element(&frac));
    }

    #[test]
    fn test_fraction_field_function() {
        let field: FractionField<Integer> = fraction_field();
        let elem = field.element(Integer::from(7), Integer::from(11));

        assert_eq!(elem.numerator(), &Integer::from(7));
        assert_eq!(elem.denominator(), &Integer::from(11));
    }

    #[test]
    fn test_fraction_display() {
        let a = FractionFieldElement::new(Integer::from(3), Integer::from(4));
        assert_eq!(format!("{}", a), "3/4");

        let b = FractionFieldElement::new(Integer::from(5), Integer::from(1));
        assert_eq!(format!("{}", b), "5");
    }
}
