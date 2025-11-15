//! Combinatorial Free Module
//!
//! A free module over a ring with basis indexed by a combinatorial set.
//! This is the Rust equivalent of SageMath's CombinatorialFreeModule.
//!
//! Combinatorial free modules form the foundation for many algebraic structures
//! in combinatorics, including group algebras, symmetric function algebras,
//! and various diagram algebras.

use rustmath_core::{Ring, MathError, Result};
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::hash::Hash;
use std::ops::{Add, Sub, Neg, Mul};

/// An element of a combinatorial free module
///
/// Represented as a formal linear combination of basis elements with coefficients
/// from a ring R. Basis elements are indexed by type I.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
/// * `I` - The index type for basis elements (must be hashable and comparable)
#[derive(Clone, Debug)]
pub struct CombinatorialFreeModuleElement<R: Ring, I: Hash + Eq + Clone> {
    /// Map from basis indices to coefficients
    /// Only stores non-zero coefficients
    pub(crate) terms: HashMap<I, R>,
}

impl<R: Ring, I: Hash + Eq + Clone> CombinatorialFreeModuleElement<R, I> {
    /// Create a zero element
    pub fn zero() -> Self {
        Self {
            terms: HashMap::new(),
        }
    }

    /// Create an element from a single term
    ///
    /// # Arguments
    ///
    /// * `index` - The basis index
    /// * `coeff` - The coefficient
    pub fn monomial(index: I, coeff: R) -> Self {
        let mut terms = HashMap::new();
        if !coeff.is_zero() {
            terms.insert(index, coeff);
        }
        Self { terms }
    }

    /// Create an element from a basis index with coefficient 1
    pub fn from_basis_index(index: I) -> Self {
        Self::monomial(index, R::one())
    }

    /// Create an element from a dictionary of terms
    ///
    /// Automatically removes zero coefficients
    pub fn from_dict(terms: HashMap<I, R>) -> Self {
        let mut result = Self::zero();
        for (index, coeff) in terms {
            if !coeff.is_zero() {
                result.terms.insert(index, coeff);
            }
        }
        result
    }

    /// Create an element as a sum of terms
    ///
    /// # Arguments
    ///
    /// * `terms` - Iterator of (index, coefficient) pairs
    pub fn sum_of_terms<Iter>(terms: Iter) -> Self
    where
        Iter: IntoIterator<Item = (I, R)>,
    {
        let mut result = Self::zero();
        for (index, coeff) in terms {
            result.add_term(index, coeff);
        }
        result
    }

    /// Add a term to this element
    ///
    /// # Arguments
    ///
    /// * `index` - The basis index
    /// * `coeff` - The coefficient to add
    pub fn add_term(&mut self, index: I, coeff: R) {
        if coeff.is_zero() {
            return;
        }

        let entry = self.terms.entry(index).or_insert_with(R::zero);
        *entry = entry.clone() + coeff;

        // Clean up zero coefficients
        self.terms.retain(|_, c| !c.is_zero());
    }

    /// Check if this is the zero element
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Get the coefficient of a basis element
    pub fn coefficient(&self, index: &I) -> R {
        self.terms.get(index).cloned().unwrap_or_else(R::zero)
    }

    /// Get the support (set of basis indices with non-zero coefficients)
    pub fn support(&self) -> Vec<I> {
        self.terms.keys().cloned().collect()
    }

    /// Number of terms (non-zero coefficients)
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &R) -> Self {
        if scalar.is_zero() {
            return Self::zero();
        }

        let mut result = Self::zero();
        for (index, coeff) in &self.terms {
            result.add_term(index.clone(), coeff.clone() * scalar.clone());
        }
        result
    }

    /// Apply a function to all coefficients
    pub fn map_coefficients<F>(&self, f: F) -> Self
    where
        F: Fn(&R) -> R,
    {
        let mut result = Self::zero();
        for (index, coeff) in &self.terms {
            let new_coeff = f(coeff);
            if !new_coeff.is_zero() {
                result.terms.insert(index.clone(), new_coeff);
            }
        }
        result
    }

    /// Apply a function to all basis indices
    pub fn map_support<J: Hash + Eq + Clone, F>(&self, f: F) -> CombinatorialFreeModuleElement<R, J>
    where
        F: Fn(&I) -> J,
    {
        let mut result = CombinatorialFreeModuleElement::zero();
        for (index, coeff) in &self.terms {
            result.add_term(f(index), coeff.clone());
        }
        result
    }

    /// Iterate over all (index, coefficient) pairs
    ///
    /// # Returns
    ///
    /// An iterator over references to (index, coefficient) pairs
    pub fn iter(&self) -> impl Iterator<Item = (&I, &R)> {
        self.terms.iter()
    }
}

impl<R: Ring, I: Hash + Eq + Clone> PartialEq for CombinatorialFreeModuleElement<R, I> {
    fn eq(&self, other: &Self) -> bool {
        if self.terms.len() != other.terms.len() {
            return false;
        }

        for (index, coeff) in &self.terms {
            let other_coeff = other.coefficient(index);
            if *coeff != other_coeff {
                return false;
            }
        }

        true
    }
}

impl<R: Ring, I: Hash + Eq + Clone> Eq for CombinatorialFreeModuleElement<R, I> {}

impl<R: Ring, I: Hash + Eq + Clone + Display> Display for CombinatorialFreeModuleElement<R, I> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut sorted_terms: Vec<_> = self.terms.iter().collect();
        sorted_terms.sort_by(|(a, _), (b, _)| {
            // Default ordering - can be customized
            format!("{}", a).cmp(&format!("{}", b))
        });

        for (i, (index, coeff)) in sorted_terms.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }

            if coeff.is_one() {
                write!(f, "B[{}]", index)?;
            } else {
                write!(f, "{}*B[{}]", coeff, index)?;
            }
        }

        Ok(())
    }
}

impl<R: Ring, I: Hash + Eq + Clone> Add for CombinatorialFreeModuleElement<R, I> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut result = self.clone();
        for (index, coeff) in other.terms {
            result.add_term(index, coeff);
        }
        result
    }
}

impl<R: Ring, I: Hash + Eq + Clone> Sub for CombinatorialFreeModuleElement<R, I> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + (-other)
    }
}

impl<R: Ring, I: Hash + Eq + Clone> Neg for CombinatorialFreeModuleElement<R, I> {
    type Output = Self;

    fn neg(self) -> Self {
        let mut result = Self::zero();
        for (index, coeff) in self.terms {
            result.add_term(index, -coeff);
        }
        result
    }
}

/// A combinatorial free module over a ring
///
/// This structure represents a free R-module with a basis indexed by a
/// combinatorial set. It serves as the foundation for many algebraic
/// structures in combinatorics.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
/// * `I` - The index type for basis elements
#[derive(Clone, Debug)]
pub struct CombinatorialFreeModule<R: Ring, I: Hash + Eq + Clone> {
    /// The base ring
    base_ring: std::marker::PhantomData<R>,
    /// The set of basis indices
    basis_keys: Vec<I>,
    /// Optional prefix for displaying basis elements
    prefix: String,
}

impl<R: Ring, I: Hash + Eq + Clone> CombinatorialFreeModule<R, I> {
    /// Create a new combinatorial free module
    ///
    /// # Arguments
    ///
    /// * `basis_keys` - The indices for the basis elements
    /// * `prefix` - String prefix for displaying basis elements (default: "B")
    pub fn new(basis_keys: Vec<I>, prefix: Option<String>) -> Self {
        Self {
            base_ring: std::marker::PhantomData,
            basis_keys,
            prefix: prefix.unwrap_or_else(|| "B".to_string()),
        }
    }

    /// Get the dimension (rank) of the module
    pub fn dimension(&self) -> usize {
        self.basis_keys.len()
    }

    /// Get the rank (synonym for dimension)
    pub fn rank(&self) -> usize {
        self.dimension()
    }

    /// Get the basis keys
    pub fn basis_keys(&self) -> &[I] {
        &self.basis_keys
    }

    /// Create the zero element
    pub fn zero(&self) -> CombinatorialFreeModuleElement<R, I> {
        CombinatorialFreeModuleElement::zero()
    }

    /// Create a basis element from an index
    pub fn monomial(&self, index: I) -> CombinatorialFreeModuleElement<R, I> {
        CombinatorialFreeModuleElement::from_basis_index(index)
    }

    /// Create a term with a specific coefficient
    pub fn term(&self, index: I, coeff: R) -> CombinatorialFreeModuleElement<R, I> {
        CombinatorialFreeModuleElement::monomial(index, coeff)
    }

    /// Get all basis elements
    pub fn basis(&self) -> Vec<CombinatorialFreeModuleElement<R, I>> {
        self.basis_keys
            .iter()
            .map(|idx| self.monomial(idx.clone()))
            .collect()
    }

    /// Linear combination of basis elements
    ///
    /// # Arguments
    ///
    /// * `coeffs` - Iterator of (index, coefficient) pairs
    pub fn linear_combination<Iter>(&self, coeffs: Iter) -> CombinatorialFreeModuleElement<R, I>
    where
        Iter: IntoIterator<Item = (I, R)>,
    {
        CombinatorialFreeModuleElement::sum_of_terms(coeffs)
    }

    /// Sum of elements
    pub fn sum<Iter>(&self, elements: Iter) -> CombinatorialFreeModuleElement<R, I>
    where
        Iter: IntoIterator<Item = CombinatorialFreeModuleElement<R, I>>,
    {
        elements
            .into_iter()
            .fold(self.zero(), |acc, elem| acc + elem)
    }

    /// Get the prefix used for displaying basis elements
    pub fn prefix(&self) -> &str {
        &self.prefix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_combinatorial_free_module_creation() {
        let basis_keys = vec![0, 1, 2, 3];
        let module: CombinatorialFreeModule<Integer, usize> =
            CombinatorialFreeModule::new(basis_keys, None);

        assert_eq!(module.dimension(), 4);
        assert_eq!(module.rank(), 4);
    }

    #[test]
    fn test_monomial_creation() {
        let module: CombinatorialFreeModule<Integer, usize> =
            CombinatorialFreeModule::new(vec![0, 1, 2], None);

        let b0 = module.monomial(0);
        let b1 = module.monomial(1);

        assert!(!b0.is_zero());
        assert_eq!(b0.coefficient(&0), Integer::one());
        assert_eq!(b0.coefficient(&1), Integer::zero());

        assert!(!b1.is_zero());
        assert_eq!(b1.coefficient(&1), Integer::one());
    }

    #[test]
    fn test_element_addition() {
        let b0 = CombinatorialFreeModuleElement::from_basis_index(0);
        let b1 = CombinatorialFreeModuleElement::from_basis_index(1);

        let sum: CombinatorialFreeModuleElement<Integer, usize> = b0.clone() + b1.clone();

        assert_eq!(sum.coefficient(&0), Integer::one());
        assert_eq!(sum.coefficient(&1), Integer::one());
        assert_eq!(sum.num_terms(), 2);
    }

    #[test]
    fn test_element_scalar_multiplication() {
        let b0 = CombinatorialFreeModuleElement::<Integer, usize>::from_basis_index(0);
        let scaled = b0.scalar_mul(&Integer::from(3));

        assert_eq!(scaled.coefficient(&0), Integer::from(3));
    }

    #[test]
    fn test_element_subtraction() {
        let b0 = CombinatorialFreeModuleElement::<Integer, usize>::from_basis_index(0);
        let sum = b0.clone() + b0.clone(); // 2*B[0]
        let diff = sum - b0; // B[0]

        assert_eq!(diff.coefficient(&0), Integer::one());
    }

    #[test]
    fn test_linear_combination() {
        let module: CombinatorialFreeModule<Integer, usize> =
            CombinatorialFreeModule::new(vec![0, 1, 2], None);

        let coeffs = vec![
            (0, Integer::from(2)),
            (1, Integer::from(3)),
            (2, Integer::from(-1)),
        ];

        let elem = module.linear_combination(coeffs);

        assert_eq!(elem.coefficient(&0), Integer::from(2));
        assert_eq!(elem.coefficient(&1), Integer::from(3));
        assert_eq!(elem.coefficient(&2), Integer::from(-1));
    }

    #[test]
    fn test_zero_cleanup() {
        let b0 = CombinatorialFreeModuleElement::<Integer, usize>::from_basis_index(0);
        let diff = b0.clone() - b0;

        assert!(diff.is_zero());
        assert_eq!(diff.num_terms(), 0);
    }

    #[test]
    fn test_support() {
        let module: CombinatorialFreeModule<Integer, usize> =
            CombinatorialFreeModule::new(vec![0, 1, 2, 3], None);

        let elem = module.linear_combination(vec![
            (0, Integer::from(1)),
            (2, Integer::from(5)),
        ]);

        let support = elem.support();
        assert_eq!(support.len(), 2);
        assert!(support.contains(&0));
        assert!(support.contains(&2));
        assert!(!support.contains(&1));
    }

    #[test]
    fn test_map_coefficients() {
        let b0 = CombinatorialFreeModuleElement::monomial(0, Integer::from(2));
        let b1 = CombinatorialFreeModuleElement::monomial(1, Integer::from(3));
        let elem = b0 + b1;

        let doubled = elem.map_coefficients(|c| c.clone() + c.clone());

        assert_eq!(doubled.coefficient(&0), Integer::from(4));
        assert_eq!(doubled.coefficient(&1), Integer::from(6));
    }
}
