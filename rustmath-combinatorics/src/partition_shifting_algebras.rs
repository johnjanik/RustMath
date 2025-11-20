//! Partition Shifting Algebras
//!
//! This module implements the partition shifting operator algebra, which is isomorphic
//! as an R-algebra to the Laurent polynomial ring R[x₁±, x₂±, x₃±, ...].
//!
//! These operators act on integer sequences by elementwise addition, and can be used
//! to transform partitions and symmetric functions.
//!
//! # Mathematical Background
//!
//! The shifting operator algebra consists of formal expressions that act on sequences
//! of integers. For a monomial s = x₁^a₁ x₂^a₂ ... xᵣ^aᵣ and a sequence
//! λ = (λ₁, λ₂, ..., λₗ), the action is:
//!
//! ```text
//! s·λ = (λ₁ + a₁, λ₂ + a₂, ..., λₘ + aₘ)
//! ```
//!
//! where m = max(r, l) and sequences are padded with zeros as needed.
//!
//! ## Young's Raising Operators
//!
//! A special class of operators R_{ij} (where i < j) that raise part i and lower part j:
//! - R_{01} corresponds to x₁/x₂ (raises first part, lowers second)
//! - R_{12} corresponds to x₂/x₃ (raises second part, lowers third)
//!
//! # Examples
//!
//! ```
//! use rustmath_combinatorics::partition_shifting_algebras::*;
//! use rustmath_rationals::Rational;
//!
//! // Create a shifting sequence (1, -1)
//! let seq = ShiftingSequence::new(vec![1, -1]).unwrap();
//!
//! // Act on a partition [5, 4]
//! let result = seq.act_on_sequence(&[5, 4]);
//! assert_eq!(result, vec![6, 3]);
//! ```

use rustmath_core::{Ring, MathError, Result};
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::hash::{Hash, Hasher};
use std::ops::{Add, Sub};
use crate::partitions::Partition;

/// A shifting sequence: a tuple of integers with no trailing zeros
///
/// This represents a basis element of the shifting operator algebra.
/// Valid sequences are finite tuples of integers where the last element is non-zero,
/// or the empty tuple.
///
/// # Examples
///
/// Valid sequences: `()`, `(1)`, `(1, -1)`, `(1, -1, 0, 9)`
/// Invalid sequences: `(1, -1, 0)` - has trailing zeros
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShiftingSequence {
    /// The sequence of integers (guaranteed to have no trailing zeros)
    components: Vec<i32>,
}

impl ShiftingSequence {
    /// Create a new shifting sequence
    ///
    /// Automatically removes trailing zeros. Returns an error if the input
    /// is invalid (though currently all inputs are valid after normalization).
    ///
    /// # Arguments
    ///
    /// * `components` - The integer sequence
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_combinatorics::partition_shifting_algebras::ShiftingSequence;
    ///
    /// let seq = ShiftingSequence::new(vec![1, -1, 0]).unwrap();
    /// assert_eq!(seq.components(), &[1, -1]);
    /// ```
    pub fn new(components: Vec<i32>) -> Result<Self> {
        let mut seq = ShiftingSequence { components };
        seq.normalize();
        Ok(seq)
    }

    /// Create the empty sequence
    pub fn empty() -> Self {
        ShiftingSequence {
            components: Vec::new(),
        }
    }

    /// Create a sequence from a single value
    pub fn singleton(value: i32) -> Result<Self> {
        if value == 0 {
            Ok(Self::empty())
        } else {
            Ok(ShiftingSequence {
                components: vec![value],
            })
        }
    }

    /// Remove trailing zeros
    fn normalize(&mut self) {
        while let Some(&last) = self.components.last() {
            if last == 0 {
                self.components.pop();
            } else {
                break;
            }
        }
    }

    /// Get the components as a slice
    pub fn components(&self) -> &[i32] {
        &self.components
    }

    /// Get the length of the sequence (number of components)
    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// Check if the sequence is empty
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// Get the i-th component (0-indexed), returning 0 if out of bounds
    pub fn get(&self, index: usize) -> i32 {
        self.components.get(index).copied().unwrap_or(0)
    }

    /// Compute the product of two shifting sequences (elementwise addition)
    ///
    /// This is the multiplication operation in the shifting operator algebra.
    ///
    /// # Arguments
    ///
    /// * `other` - The other sequence to multiply with
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_combinatorics::partition_shifting_algebras::ShiftingSequence;
    ///
    /// let s1 = ShiftingSequence::new(vec![1, -1]).unwrap();
    /// let s2 = ShiftingSequence::new(vec![0, 2, 1]).unwrap();
    /// let product = s1.product(&s2);
    /// assert_eq!(product.components(), &[1, 1, 1]);
    /// ```
    pub fn product(&self, other: &Self) -> Self {
        let max_len = self.len().max(other.len());
        let mut result = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let sum = self.get(i) + other.get(i);
            result.push(sum);
        }

        ShiftingSequence::new(result).unwrap()
    }

    /// Act on an integer sequence by elementwise addition
    ///
    /// This is the fundamental action of shifting operators on sequences.
    ///
    /// # Arguments
    ///
    /// * `sequence` - The sequence to act on
    ///
    /// # Returns
    ///
    /// The result of adding this shifting sequence to the input sequence,
    /// with trailing zeros removed.
    pub fn act_on_sequence(&self, sequence: &[i32]) -> Vec<i32> {
        let max_len = self.len().max(sequence.len());
        let mut result = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let sum = self.get(i) + sequence.get(i).copied().unwrap_or(0);
            result.push(sum);
        }

        // Remove trailing zeros
        while let Some(&last) = result.last() {
            if last == 0 {
                result.pop();
            } else {
                break;
            }
        }

        result
    }

    /// Check if this sequence is a valid partition (non-increasing, non-negative)
    pub fn is_partition(&self) -> bool {
        // Check non-negative
        if self.components.iter().any(|&x| x < 0) {
            return false;
        }

        // Check non-increasing
        for i in 1..self.components.len() {
            if self.components[i] > self.components[i - 1] {
                return false;
            }
        }

        true
    }

    /// Convert to a partition (if valid)
    pub fn to_partition(&self) -> Option<Partition> {
        if !self.is_partition() {
            return None;
        }
        Some(Partition::new(self.components.iter().map(|&x| x as usize).collect()))
    }
}

impl Hash for ShiftingSequence {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.components.hash(state);
    }
}

impl Display for ShiftingSequence {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;
        for (i, &comp) in self.components.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", comp)?;
        }
        write!(f, "]")
    }
}

/// The shifting sequence space
///
/// This is a facade for tuples with entries in ℤ of finite support with no trailing 0's.
pub struct ShiftingSequenceSpace;

impl ShiftingSequenceSpace {
    /// Check if a sequence is a valid member of this space
    pub fn contains(sequence: &[i32]) -> bool {
        // Empty sequence is valid
        if sequence.is_empty() {
            return true;
        }

        // Check that the last element is non-zero
        sequence.last().map(|&x| x != 0).unwrap_or(true)
    }

    /// Validate a sequence, returning an error if invalid
    pub fn check(sequence: &[i32]) -> Result<()> {
        if !Self::contains(sequence) {
            return Err(MathError::InvalidArgument(
                "Shifting sequence must not have trailing zeros".to_string(),
            ));
        }
        Ok(())
    }
}

/// The shifting operator algebra over a ring R
///
/// This algebra is isomorphic to R[x₁±, x₂±, x₃±, ...] and acts on integer sequences
/// by elementwise addition.
///
/// # Type Parameters
///
/// * `R` - The base ring (typically QQ['t'] or similar)
pub struct ShiftingOperatorAlgebra<R: Ring> {
    /// Base ring
    _phantom: std::marker::PhantomData<R>,
}

impl<R: Ring> ShiftingOperatorAlgebra<R> {
    /// Create a new shifting operator algebra over the ring R
    pub fn new() -> Self {
        ShiftingOperatorAlgebra {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a monomial basis element (single shifting sequence with coefficient 1)
    pub fn monomial(&self, sequence: ShiftingSequence) -> ShiftingOperatorElement<R> {
        ShiftingOperatorElement::<R>::from_basis_index(sequence)
    }

    /// Create a monomial from a vector
    pub fn monomial_from_vec(&self, components: Vec<i32>) -> Result<ShiftingOperatorElement<R>> {
        let seq = ShiftingSequence::new(components)?;
        Ok(self.monomial(seq))
    }

    /// Create the identity element (empty sequence)
    pub fn one(&self) -> ShiftingOperatorElement<R> {
        self.monomial(ShiftingSequence::empty())
    }

    /// Create the zero element
    pub fn zero(&self) -> ShiftingOperatorElement<R> {
        ShiftingOperatorElement::<R>::zero()
    }

    /// Create Young's raising operator R_{ij}
    ///
    /// This operator raises part i and lowers part j (using 0-based indexing).
    /// Mathematically: R_{ij} = x_{i+1} / x_{j+1}
    ///
    /// # Arguments
    ///
    /// * `i` - The index to raise (0-based)
    /// * `j` - The index to lower (0-based)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_combinatorics::partition_shifting_algebras::*;
    /// use rustmath_rationals::Rational;
    ///
    /// let algebra = ShiftingOperatorAlgebra::<Rational>::new();
    /// let r01 = algebra.raising_operator(0, 1); // Raises first, lowers second
    /// ```
    pub fn raising_operator(&self, i: usize, j: usize) -> ShiftingOperatorElement<R> {
        let max_index = i.max(j);
        let mut components = vec![0; max_index + 1];
        components[i] = 1;
        components[j] = -1;

        self.monomial_from_vec(components).unwrap()
    }

    /// Multiply two elements in the algebra
    ///
    /// The product of two shifting operators corresponds to composition of their actions.
    pub fn multiply(
        &self,
        left: &ShiftingOperatorElement<R>,
        right: &ShiftingOperatorElement<R>,
    ) -> ShiftingOperatorElement<R> {
        let mut result = ShiftingOperatorElement::zero();

        for (seq1, coeff1) in left.iter() {
            for (seq2, coeff2) in right.iter() {
                let product_seq = seq1.product(seq2);
                let product_coeff = coeff1.clone() * coeff2.clone();
                result.add_term(product_seq, product_coeff);
            }
        }

        result
    }
}

impl<R: Ring> Default for ShiftingOperatorAlgebra<R> {
    fn default() -> Self {
        Self::new()
    }
}

/// An element of the shifting operator algebra
///
/// This is a linear combination of shifting sequences with coefficients from ring R.
pub struct ShiftingOperatorElement<R: Ring> {
    /// Map from basis sequences to coefficients
    /// Only stores non-zero coefficients
    terms: HashMap<ShiftingSequence, R>,
}

impl<R: Ring> ShiftingOperatorElement<R> {
    /// Create the zero element
    pub fn zero() -> Self {
        ShiftingOperatorElement {
            terms: HashMap::new(),
        }
    }

    /// Create an element from a single basis element
    pub fn from_basis_index(sequence: ShiftingSequence) -> Self {
        let mut terms = HashMap::new();
        terms.insert(sequence, R::one());
        ShiftingOperatorElement { terms }
    }

    /// Create an element from a basis element with a coefficient
    pub fn monomial(sequence: ShiftingSequence, coeff: R) -> Self {
        let mut terms = HashMap::new();
        if !coeff.is_zero() {
            terms.insert(sequence, coeff);
        }
        ShiftingOperatorElement { terms }
    }

    /// Add a term to this element
    fn add_term(&mut self, sequence: ShiftingSequence, coeff: R) {
        if coeff.is_zero() {
            return;
        }

        let entry = self.terms.entry(sequence).or_insert_with(R::zero);
        *entry = entry.clone() + coeff;

        // Clean up zero coefficients
        self.terms.retain(|_, c| !c.is_zero());
    }

    /// Check if this is the zero element
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Get the coefficient of a basis element
    pub fn coefficient(&self, sequence: &ShiftingSequence) -> R {
        self.terms.get(sequence).cloned().unwrap_or_else(R::zero)
    }

    /// Get the support (set of basis sequences with non-zero coefficients)
    pub fn support(&self) -> Vec<ShiftingSequence> {
        self.terms.keys().cloned().collect()
    }

    /// Number of terms
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    /// Iterate over (sequence, coefficient) pairs
    pub fn iter(&self) -> impl Iterator<Item = (&ShiftingSequence, &R)> {
        self.terms.iter()
    }

    /// Act on an integer sequence
    ///
    /// Applies this operator to an integer sequence, returning a list of
    /// (result_sequence, coefficient) pairs.
    ///
    /// # Arguments
    ///
    /// * `sequence` - The sequence to act on
    pub fn act_on_sequence(&self, sequence: &[i32]) -> Vec<(Vec<i32>, R)> {
        let mut results = Vec::new();

        for (shift_seq, coeff) in self.iter() {
            let result_seq = shift_seq.act_on_sequence(sequence);
            results.push((result_seq, coeff.clone()));
        }

        results
    }

    /// Act on a partition
    ///
    /// Applies this operator to a partition, returning a list of
    /// (result_partition, coefficient) pairs. Invalid results (negative parts)
    /// are omitted.
    ///
    /// # Arguments
    ///
    /// * `partition` - The partition to act on
    pub fn act_on_partition(&self, partition: &Partition) -> Vec<(Partition, R)> {
        let sequence: Vec<i32> = partition
            .parts()
            .iter()
            .map(|&x| x as i32)
            .collect();

        let mut results = Vec::new();

        for (result_seq, coeff) in self.act_on_sequence(&sequence) {
            // Check if result is a valid partition (non-negative, non-increasing)
            if result_seq.iter().any(|&x| x < 0) {
                continue; // Skip negative parts
            }

            // Check if non-increasing
            let mut is_valid = true;
            for i in 1..result_seq.len() {
                if result_seq[i] > result_seq[i - 1] {
                    is_valid = false;
                    break;
                }
            }

            if is_valid {
                let parts: Vec<usize> = result_seq.iter().map(|&x| x as usize).collect();
                let result_partition = Partition::new(parts);
                results.push((result_partition, coeff));
            }
        }

        results
    }

    /// Add another element to this one
    pub fn add(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (seq, coeff) in other.iter() {
            result.add_term(seq.clone(), coeff.clone());
        }
        result
    }

    /// Subtract another element from this one
    pub fn sub(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (seq, coeff) in other.iter() {
            result.add_term(seq.clone(), -coeff.clone());
        }
        result
    }

    /// Multiply by a scalar
    pub fn scalar_mul(&self, scalar: &R) -> Self {
        if scalar.is_zero() {
            return Self::zero();
        }

        let mut result = Self::zero();
        for (seq, coeff) in self.iter() {
            result.add_term(seq.clone(), coeff.clone() * scalar.clone());
        }
        result
    }
}

impl<R: Ring> Clone for ShiftingOperatorElement<R> {
    fn clone(&self) -> Self {
        ShiftingOperatorElement {
            terms: self.terms.clone(),
        }
    }
}

impl<R: Ring> PartialEq for ShiftingOperatorElement<R> {
    fn eq(&self, other: &Self) -> bool {
        if self.terms.len() != other.terms.len() {
            return false;
        }

        for (seq, coeff) in &self.terms {
            let other_coeff = other.coefficient(seq);
            if *coeff != other_coeff {
                return false;
            }
        }

        true
    }
}

impl<R: Ring> Eq for ShiftingOperatorElement<R> {}

impl<R: Ring> Display for ShiftingOperatorElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut first = true;
        for (seq, coeff) in self.iter() {
            if !first {
                write!(f, " + ")?;
            }
            first = false;

            // Display coefficient if not 1 or if sequence is empty
            if !coeff.is_one() || seq.is_empty() {
                write!(f, "{}", coeff)?;
                if !seq.is_empty() {
                    write!(f, "*")?;
                }
            }

            if !seq.is_empty() {
                write!(f, "S{}", seq)?;
            }
        }

        Ok(())
    }
}

impl<R: Ring> Add for ShiftingOperatorElement<R> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        ShiftingOperatorElement::add(&self, &other)
    }
}

impl<R: Ring> Sub for ShiftingOperatorElement<R> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        ShiftingOperatorElement::sub(&self, &other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_shifting_sequence_creation() {
        // Test normal creation
        let seq = ShiftingSequence::new(vec![1, -1, 2]).unwrap();
        assert_eq!(seq.components(), &[1, -1, 2]);

        // Test trailing zero removal
        let seq = ShiftingSequence::new(vec![1, -1, 0, 0]).unwrap();
        assert_eq!(seq.components(), &[1, -1]);

        // Test empty sequence
        let seq = ShiftingSequence::empty();
        assert_eq!(seq.components(), &[]);
        assert!(seq.is_empty());
    }

    #[test]
    fn test_shifting_sequence_product() {
        let s1 = ShiftingSequence::new(vec![1, -1]).unwrap();
        let s2 = ShiftingSequence::new(vec![0, 2, 1]).unwrap();
        let product = s1.product(&s2);

        assert_eq!(product.components(), &[1, 1, 1]);
    }

    #[test]
    fn test_shifting_sequence_action() {
        let seq = ShiftingSequence::new(vec![1, -1, 2]).unwrap();
        let input = vec![5, 4, 1];
        let result = seq.act_on_sequence(&input);

        assert_eq!(result, vec![6, 3, 3]);
    }

    #[test]
    fn test_shifting_sequence_action_with_padding() {
        let seq = ShiftingSequence::new(vec![1, -1]).unwrap();
        let input = vec![5, 4];
        let result = seq.act_on_sequence(&input);

        assert_eq!(result, vec![6, 3]);
    }

    #[test]
    fn test_shifting_sequence_space() {
        // Valid sequences
        assert!(ShiftingSequenceSpace::contains(&[]));
        assert!(ShiftingSequenceSpace::contains(&[1]));
        assert!(ShiftingSequenceSpace::contains(&[1, -1]));
        assert!(ShiftingSequenceSpace::contains(&[1, -1, 0, 9]));

        // Invalid sequences (trailing zeros)
        assert!(!ShiftingSequenceSpace::contains(&[1, -1, 0]));
        assert!(!ShiftingSequenceSpace::contains(&[0]));
    }

    #[test]
    fn test_algebra_monomial() {
        let algebra = ShiftingOperatorAlgebra::<Rational>::new();
        let m = algebra.monomial_from_vec(vec![1, -1]).unwrap();

        assert!(!m.is_zero());
        assert_eq!(m.num_terms(), 1);
    }

    #[test]
    fn test_algebra_one() {
        let algebra = ShiftingOperatorAlgebra::<Rational>::new();
        let one = algebra.one();

        assert!(!one.is_zero());
        assert_eq!(one.num_terms(), 1);

        let seq = ShiftingSequence::empty();
        assert_eq!(one.coefficient(&seq), Rational::one());
    }

    #[test]
    fn test_raising_operator() {
        let algebra = ShiftingOperatorAlgebra::<Rational>::new();
        let r01 = algebra.raising_operator(0, 1);

        // R_{01} should act as [1, -1]
        let input = vec![5, 4];
        let results = r01.act_on_sequence(&input);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, vec![6, 3]);
        assert_eq!(results[0].1, Rational::one());
    }

    #[test]
    fn test_algebra_multiplication() {
        let algebra = ShiftingOperatorAlgebra::<Rational>::new();
        let m1 = algebra.monomial_from_vec(vec![1, 0]).unwrap();
        let m2 = algebra.monomial_from_vec(vec![0, 1]).unwrap();

        let product = algebra.multiply(&m1, &m2);

        // Should get [1, 1]
        let seq = ShiftingSequence::new(vec![1, 1]).unwrap();
        assert_eq!(product.coefficient(&seq), Rational::one());
    }

    #[test]
    fn test_element_action_on_partition() {
        let algebra = ShiftingOperatorAlgebra::<Rational>::new();
        let op = algebra.monomial_from_vec(vec![1, -1]).unwrap();

        let partition = Partition::new(vec![5, 4]);
        let results = op.act_on_partition(&partition);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, Partition::new(vec![6, 3]));
        assert_eq!(results[0].1, Rational::one());
    }

    #[test]
    fn test_element_action_rejects_invalid_partitions() {
        let algebra = ShiftingOperatorAlgebra::<Rational>::new();
        // This will make the result negative
        let op = algebra.monomial_from_vec(vec![-10, 0]).unwrap();

        let partition = Partition::new(vec![5, 4]);
        let results = op.act_on_partition(&partition);

        // Should be empty because result would have negative parts
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_element_addition() {
        let algebra = ShiftingOperatorAlgebra::<Rational>::new();
        let m1 = algebra.monomial_from_vec(vec![1, -1]).unwrap();
        let m2 = algebra.monomial_from_vec(vec![1, 0]).unwrap();

        let sum = ShiftingOperatorElement::add(&m1, &m2);

        assert_eq!(sum.num_terms(), 2);
    }

    #[test]
    fn test_element_scalar_multiplication() {
        let algebra = ShiftingOperatorAlgebra::<Rational>::new();
        let m = algebra.monomial_from_vec(vec![1, -1]).unwrap();
        let scalar = Rational::new(2, 1).unwrap();

        let result = m.scalar_mul(&scalar);

        let seq = ShiftingSequence::new(vec![1, -1]).unwrap();
        assert_eq!(result.coefficient(&seq), Rational::new(2, 1).unwrap());
    }

    #[test]
    fn test_jacobi_trudi_pattern() {
        // Test a pattern from Jacobi-Trudi identity
        let algebra = ShiftingOperatorAlgebra::<Rational>::new();

        let r1 = algebra.raising_operator(0, 1); // (1, -1)
        let r2 = algebra.raising_operator(0, 2); // (1, 0, -1)
        let r3 = algebra.raising_operator(1, 2); // (0, 1, -1)

        let one = algebra.one();

        // (1 - R_{01})
        let term1 = ShiftingOperatorElement::sub(&one, &r1);

        // Check that this has 2 terms
        assert_eq!(term1.num_terms(), 2);
    }

    #[test]
    fn test_partition_action_preserves_order() {
        let algebra = ShiftingOperatorAlgebra::<Rational>::new();
        // This would make parts increase, which is invalid
        let op = algebra.monomial_from_vec(vec![-2, 1]).unwrap();

        let partition = Partition::new(vec![5, 4]);
        let results = op.act_on_partition(&partition);

        // Result would be [3, 5] which is not non-increasing
        assert_eq!(results.len(), 0);
    }
}
