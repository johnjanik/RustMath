//! Elements of modules with basis
//!
//! This module provides the element type for modules with a distinguished basis.
//! Elements are represented sparsely as maps from basis indices to coefficients.

use rustmath_core::Ring;
use std::collections::BTreeMap;
use std::fmt;
use std::ops::{Add, Neg, Mul};

/// An element of a module with basis
///
/// Elements are stored sparsely as a map from basis indices to coefficients.
/// Zero coefficients are not stored, making this efficient for sparse vectors.
///
/// # Type Parameters
/// - `I`: Index type for basis elements (must be ordered and cloneable)
/// - `R`: Coefficient ring type
///
/// # Examples
/// ```
/// use rustmath_modules::with_basis::element::ModuleWithBasisElement;
/// use num_bigint::BigInt;
/// use std::collections::BTreeMap;
///
/// // Create element 3*e_0 + 5*e_2
/// let mut coeffs = BTreeMap::new();
/// coeffs.insert(0, BigInt::from(3));
/// coeffs.insert(2, BigInt::from(5));
/// let elem = ModuleWithBasisElement::new(coeffs);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleWithBasisElement<I, R>
where
    I: Ord + Clone,
    R: Ring,
{
    /// Sparse representation: maps basis index to coefficient
    /// Zero coefficients are not stored
    coefficients: BTreeMap<I, R>,
}

impl<I, R> ModuleWithBasisElement<I, R>
where
    I: Ord + Clone,
    R: Ring,
{
    /// Create a new element from coefficients
    pub fn new(coefficients: BTreeMap<I, R>) -> Self {
        let mut elem = ModuleWithBasisElement { coefficients };
        elem.cleanup(); // Remove zero coefficients
        elem
    }

    /// Create a zero element
    pub fn zero() -> Self {
        ModuleWithBasisElement {
            coefficients: BTreeMap::new(),
        }
    }

    /// Create an element from a single basis element with coefficient
    pub fn from_basis_element(index: I, coefficient: R) -> Self {
        if coefficient.is_zero() {
            Self::zero()
        } else {
            let mut coefficients = BTreeMap::new();
            coefficients.insert(index, coefficient);
            ModuleWithBasisElement { coefficients }
        }
    }

    /// Check if this element is zero
    pub fn is_zero(&self) -> bool {
        self.coefficients.is_empty()
    }

    /// Get the coefficient of a basis element
    ///
    /// Returns the zero element if the index is not present
    pub fn coefficient(&self, index: &I) -> R {
        self.coefficients
            .get(index)
            .cloned()
            .unwrap_or_else(|| R::zero())
    }

    /// Get a reference to the coefficient of a basis element
    pub fn coefficient_ref(&self, index: &I) -> Option<&R> {
        self.coefficients.get(index)
    }

    /// Get the support (indices with non-zero coefficients)
    pub fn support(&self) -> Vec<I> {
        self.coefficients.keys().cloned().collect()
    }

    /// Get the number of non-zero coefficients
    pub fn num_nonzero(&self) -> usize {
        self.coefficients.len()
    }

    /// Iterate over (index, coefficient) pairs
    pub fn items(&self) -> impl Iterator<Item = (&I, &R)> {
        self.coefficients.iter()
    }

    /// Iterate over coefficients
    pub fn coefficients_iter(&self) -> impl Iterator<Item = &R> {
        self.coefficients.values()
    }

    /// Get mutable access to coefficients (for internal use)
    pub(crate) fn coefficients_mut(&mut self) -> &mut BTreeMap<I, R> {
        &mut self.coefficients
    }

    /// Get immutable access to coefficients
    pub fn coefficients(&self) -> &BTreeMap<I, R> {
        &self.coefficients
    }

    /// Remove zero coefficients (cleanup after operations)
    pub fn cleanup(&mut self) {
        self.coefficients.retain(|_, coeff| !coeff.is_zero());
    }

    /// Add a term (basis index with coefficient)
    pub fn add_term(&mut self, index: I, coefficient: R) {
        if coefficient.is_zero() {
            return;
        }

        self.coefficients
            .entry(index)
            .and_modify(|c| *c = c.clone() + coefficient.clone())
            .or_insert(coefficient);

        self.cleanup();
    }

    /// Scalar multiplication (in-place)
    pub fn scalar_mul_inplace(&mut self, scalar: &R) {
        if scalar.is_zero() {
            self.coefficients.clear();
            return;
        }

        for coeff in self.coefficients.values_mut() {
            *coeff = coeff.clone() * scalar.clone();
        }

        self.cleanup();
    }

    /// Scalar multiplication (returns new element)
    pub fn scalar_mul(&self, scalar: &R) -> Self {
        if scalar.is_zero() {
            return Self::zero();
        }

        let mut result = self.clone();
        result.scalar_mul_inplace(scalar);
        result
    }

    /// Negate this element
    pub fn negate(&self) -> Self {
        let mut coefficients = BTreeMap::new();
        for (idx, coeff) in &self.coefficients {
            coefficients.insert(idx.clone(), -coeff.clone());
        }
        ModuleWithBasisElement { coefficients }
    }

    /// Add two elements
    pub fn add(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (idx, coeff) in other.items() {
            result.add_term(idx.clone(), coeff.clone());
        }
        result
    }

    /// Subtract two elements
    pub fn sub(&self, other: &Self) -> Self {
        self.add(&other.negate())
    }

    /// Create element from a list of (index, coefficient) pairs
    pub fn from_terms(terms: Vec<(I, R)>) -> Self {
        let mut coefficients = BTreeMap::new();
        for (idx, coeff) in terms {
            if !coeff.is_zero() {
                coefficients
                    .entry(idx)
                    .and_modify(|c| *c = c.clone() + coeff.clone())
                    .or_insert(coeff);
            }
        }
        let mut elem = ModuleWithBasisElement { coefficients };
        elem.cleanup();
        elem
    }

    /// Convert to a dense vector representation (given a list of all indices)
    pub fn to_dense(&self, indices: &[I]) -> Vec<R> {
        indices.iter().map(|idx| self.coefficient(idx)).collect()
    }

    /// Apply a function to all coefficients
    pub fn map_coefficients<F>(&self, f: F) -> Self
    where
        F: Fn(&R) -> R,
    {
        let mut coefficients = BTreeMap::new();
        for (idx, coeff) in &self.coefficients {
            let new_coeff = f(coeff);
            if !new_coeff.is_zero() {
                coefficients.insert(idx.clone(), new_coeff);
            }
        }
        ModuleWithBasisElement { coefficients }
    }
}

// Implement Display for pretty printing
impl<I, R> fmt::Display for ModuleWithBasisElement<I, R>
where
    I: Ord + Clone + fmt::Display,
    R: Ring + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut first = true;
        for (idx, coeff) in &self.coefficients {
            if !first {
                write!(f, " + ")?;
            }
            write!(f, "{}*e_{}", coeff, idx)?;
            first = false;
        }
        Ok(())
    }
}

// Implement Add trait
impl<I, R> Add for ModuleWithBasisElement<I, R>
where
    I: Ord + Clone,
    R: Ring,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::add(&self, &other)
    }
}

impl<I, R> Add for &ModuleWithBasisElement<I, R>
where
    I: Ord + Clone,
    R: Ring,
{
    type Output = ModuleWithBasisElement<I, R>;

    fn add(self, other: Self) -> ModuleWithBasisElement<I, R> {
        ModuleWithBasisElement::add(self, other)
    }
}

// Implement Neg trait
impl<I, R> Neg for ModuleWithBasisElement<I, R>
where
    I: Ord + Clone,
    R: Ring,
{
    type Output = Self;

    fn neg(self) -> Self {
        self.negate()
    }
}

impl<I, R> Neg for &ModuleWithBasisElement<I, R>
where
    I: Ord + Clone,
    R: Ring,
{
    type Output = ModuleWithBasisElement<I, R>;

    fn neg(self) -> ModuleWithBasisElement<I, R> {
        self.negate()
    }
}

// Implement scalar multiplication
impl<I, R> Mul<R> for ModuleWithBasisElement<I, R>
where
    I: Ord + Clone,
    R: Ring,
{
    type Output = Self;

    fn mul(self, scalar: R) -> Self {
        self.scalar_mul(&scalar)
    }
}

impl<I, R> Mul<&R> for &ModuleWithBasisElement<I, R>
where
    I: Ord + Clone,
    R: Ring,
{
    type Output = ModuleWithBasisElement<I, R>;

    fn mul(self, scalar: &R) -> ModuleWithBasisElement<I, R> {
        self.scalar_mul(scalar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    #[test]
    fn test_zero_element() {
        let zero: ModuleWithBasisElement<usize, BigInt> = ModuleWithBasisElement::zero();
        assert!(zero.is_zero());
        assert_eq!(zero.num_nonzero(), 0);
    }

    #[test]
    fn test_basis_element() {
        let elem = ModuleWithBasisElement::from_basis_element(2, BigInt::from(5));
        assert!(!elem.is_zero());
        assert_eq!(elem.coefficient(&2), BigInt::from(5));
        assert_eq!(elem.coefficient(&0), BigInt::from(0));
        assert_eq!(elem.num_nonzero(), 1);
    }

    #[test]
    fn test_addition() {
        // 3*e_0 + 5*e_2
        let elem1 = ModuleWithBasisElement::from_terms(vec![
            (0, BigInt::from(3)),
            (2, BigInt::from(5)),
        ]);

        // 2*e_0 + 7*e_1
        let elem2 = ModuleWithBasisElement::from_terms(vec![
            (0, BigInt::from(2)),
            (1, BigInt::from(7)),
        ]);

        // Result: 5*e_0 + 7*e_1 + 5*e_2
        let sum = &elem1 + &elem2;
        assert_eq!(sum.coefficient(&0), BigInt::from(5));
        assert_eq!(sum.coefficient(&1), BigInt::from(7));
        assert_eq!(sum.coefficient(&2), BigInt::from(5));
        assert_eq!(sum.num_nonzero(), 3);
    }

    #[test]
    fn test_negation() {
        let elem = ModuleWithBasisElement::from_terms(vec![
            (0, BigInt::from(3)),
            (2, BigInt::from(-5)),
        ]);

        let neg = -&elem;
        assert_eq!(neg.coefficient(&0), BigInt::from(-3));
        assert_eq!(neg.coefficient(&2), BigInt::from(5));
    }

    #[test]
    fn test_scalar_multiplication() {
        let elem = ModuleWithBasisElement::from_terms(vec![
            (0, BigInt::from(3)),
            (2, BigInt::from(5)),
        ]);

        let scaled = elem.scalar_mul(&BigInt::from(2));
        assert_eq!(scaled.coefficient(&0), BigInt::from(6));
        assert_eq!(scaled.coefficient(&2), BigInt::from(10));
    }

    #[test]
    fn test_scalar_multiplication_by_zero() {
        let elem = ModuleWithBasisElement::from_terms(vec![
            (0, BigInt::from(3)),
            (2, BigInt::from(5)),
        ]);

        let scaled = elem.scalar_mul(&BigInt::from(0));
        assert!(scaled.is_zero());
    }

    #[test]
    fn test_support() {
        let elem = ModuleWithBasisElement::from_terms(vec![
            (0, BigInt::from(3)),
            (2, BigInt::from(5)),
            (5, BigInt::from(1)),
        ]);

        let support = elem.support();
        assert_eq!(support, vec![0, 2, 5]);
    }

    #[test]
    fn test_items() {
        let elem = ModuleWithBasisElement::from_terms(vec![
            (0, BigInt::from(3)),
            (2, BigInt::from(5)),
        ]);

        let items: Vec<_> = elem.items().collect();
        assert_eq!(items.len(), 2);
        assert_eq!(*items[0].1, BigInt::from(3));
        assert_eq!(*items[1].1, BigInt::from(5));
    }

    #[test]
    fn test_cleanup_removes_zeros() {
        let mut coeffs = BTreeMap::new();
        coeffs.insert(0, BigInt::from(3));
        coeffs.insert(1, BigInt::from(0)); // This should be removed
        coeffs.insert(2, BigInt::from(5));

        let elem = ModuleWithBasisElement::new(coeffs);
        assert_eq!(elem.num_nonzero(), 2);
        assert_eq!(elem.support(), vec![0, 2]);
    }

    #[test]
    fn test_add_term() {
        let mut elem = ModuleWithBasisElement::from_basis_element(0, BigInt::from(3));
        elem.add_term(0, BigInt::from(2)); // Add to existing
        elem.add_term(1, BigInt::from(5)); // Add new

        assert_eq!(elem.coefficient(&0), BigInt::from(5));
        assert_eq!(elem.coefficient(&1), BigInt::from(5));
    }

    #[test]
    fn test_to_dense() {
        let elem = ModuleWithBasisElement::from_terms(vec![
            (1, BigInt::from(3)),
            (3, BigInt::from(5)),
        ]);

        let dense = elem.to_dense(&[0, 1, 2, 3]);
        assert_eq!(dense, vec![
            BigInt::from(0),
            BigInt::from(3),
            BigInt::from(0),
            BigInt::from(5),
        ]);
    }

    #[test]
    fn test_map_coefficients() {
        let elem = ModuleWithBasisElement::from_terms(vec![
            (0, BigInt::from(2)),
            (1, BigInt::from(3)),
        ]);

        let doubled = elem.map_coefficients(|c| c * BigInt::from(2));
        assert_eq!(doubled.coefficient(&0), BigInt::from(4));
        assert_eq!(doubled.coefficient(&1), BigInt::from(6));
    }

    #[test]
    fn test_string_indices() {
        let elem = ModuleWithBasisElement::from_terms(vec![
            ("x".to_string(), BigInt::from(3)),
            ("y".to_string(), BigInt::from(5)),
        ]);

        assert_eq!(elem.coefficient(&"x".to_string()), BigInt::from(3));
        assert_eq!(elem.coefficient(&"y".to_string()), BigInt::from(5));
        assert_eq!(elem.coefficient(&"z".to_string()), BigInt::from(0));
    }
}
