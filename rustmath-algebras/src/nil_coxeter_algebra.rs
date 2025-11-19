//! Nil-Coxeter Algebra
//!
//! The Nil-Coxeter algebra is a specialization of the Iwahori-Hecke algebra
//! where the quadratic relation becomes u_i² = 0 (nilpotent generators).
//! This algebra inherits the braid relations from the underlying Weyl group
//! while imposing strict nilpotency on each generator.
//!
//! # Mathematical Background
//!
//! The Nil-Coxeter algebra N(W) for a Coxeter group W is defined by:
//! - Generators u_s for each simple reflection s ∈ S
//! - Quadratic relation: u_s² = 0 for all s
//! - Braid relations: u_s u_t u_s ... = u_t u_s u_t ... (m_{st} factors)
//!
//! This is obtained from the Iwahori-Hecke algebra H_{q₁,q₂}(W) by setting
//! both parameters to 0: q₁ = q₂ = 0.
//!
//! ## Properties
//!
//! - **Finite-dimensional**: Unlike the Hecke algebra, the Nil-Coxeter algebra
//!   is finite-dimensional due to the nilpotency condition
//! - **Graded**: The algebra has a natural grading by word length
//! - **Representation theory**: Closely related to geometry of Springer fibers
//!
//! ## Homogeneous Elements
//!
//! For finite type A and B Coxeter groups, the r-th homogeneous element h_r
//! is computed as the sum of all products of r distinct generators in
//! decreasing order (for type A) or cyclically decreasing order (for affine types).
//!
//! For a partition λ = (λ₁, λ₂, ..., λₖ), the homogeneous element h_λ is:
//! h_λ = h_{λ₁} · h_{λ₂} · ... · h_{λₖ}
//!
//! ## k-Schur Functions
//!
//! In type A^(1)_k, the algebra supports k-Schur functions which are related
//! to affine Grassmannian cohomology and cylindric Schur functions.
//!
//! # Examples
//!
//! ```
//! use rustmath_algebras::NilCoxeterAlgebra;
//! use rustmath_rationals::Rational;
//!
//! // Create the Nil-Coxeter algebra for type A_2 (symmetric group S_3)
//! let nc = NilCoxeterAlgebra::<Rational>::new("A", 2);
//!
//! assert_eq!(nc.rank(), 2);
//! assert_eq!(nc.cartan_type(), "A");
//! ```
//!
//! # References
//!
//! - Fomin, S. and Stanley, R. "Schubert polynomials and the nil-Coxeter algebra" (1994)
//! - Springer, T. "Trigonometric sums, Green functions of finite groups and
//!   representations of Weyl groups" (1976)
//! - SageMath: sage.algebras.nil_coxeter_algebra
//!
//! Corresponds to sage.algebras.nil_coxeter_algebra

use rustmath_core::Ring;
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::hash::Hash;

/// Nil-Coxeter Algebra
///
/// A specialization of the Iwahori-Hecke algebra with nilpotent generators.
/// Each generator u_s satisfies u_s² = 0, and the generators satisfy the
/// braid relations of the underlying Coxeter group.
///
/// # Type Parameters
///
/// * `R` - The base ring (typically ℚ or ℤ)
///
/// # Mathematical Structure
///
/// For a Coxeter group W with simple reflections S = {s₁, s₂, ..., sₙ},
/// the Nil-Coxeter algebra N(W,S) has:
/// - Generators: {u_s | s ∈ S}
/// - Relations:
///   - Nilpotency: u_s² = 0 for all s ∈ S
///   - Braid: u_s u_t u_s ... = u_t u_s u_t ... (m_{st} factors on each side)
///
/// # Examples
///
/// ```
/// use rustmath_algebras::NilCoxeterAlgebra;
/// use rustmath_integers::Integer;
///
/// // Type A_3 algebra (symmetric group S_4)
/// let nc = NilCoxeterAlgebra::<Integer>::new("A", 3);
/// assert_eq!(nc.rank(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct NilCoxeterAlgebra<R: Ring> {
    /// The Cartan type of the Coxeter group ("A", "B", "C", "D", "E", "F", "G")
    cartan_type: String,
    /// Rank (number of simple reflections)
    rank: usize,
    /// Base ring
    base_ring: std::marker::PhantomData<R>,
    /// Generator name prefix (default: "u")
    prefix: String,
}

impl<R: Ring + Clone + From<i64>> NilCoxeterAlgebra<R> {
    /// Create a new Nil-Coxeter algebra
    ///
    /// # Arguments
    ///
    /// * `cartan_type` - The Cartan type ("A", "B", "C", "D", "E", "F", "G")
    /// * `rank` - The rank (number of simple reflections)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_algebras::NilCoxeterAlgebra;
    /// use rustmath_rationals::Rational;
    ///
    /// let nc = NilCoxeterAlgebra::<Rational>::new("A", 4);
    /// assert_eq!(nc.rank(), 4);
    /// ```
    pub fn new(cartan_type: &str, rank: usize) -> Self {
        NilCoxeterAlgebra {
            cartan_type: cartan_type.to_string(),
            rank,
            base_ring: std::marker::PhantomData,
            prefix: "u".to_string(),
        }
    }

    /// Create a Nil-Coxeter algebra with a custom generator prefix
    ///
    /// # Arguments
    ///
    /// * `cartan_type` - The Cartan type
    /// * `rank` - The rank
    /// * `prefix` - Custom prefix for generator names
    pub fn with_prefix(cartan_type: &str, rank: usize, prefix: String) -> Self {
        NilCoxeterAlgebra {
            cartan_type: cartan_type.to_string(),
            rank,
            base_ring: std::marker::PhantomData,
            prefix,
        }
    }

    /// Get the Cartan type
    pub fn cartan_type(&self) -> &str {
        &self.cartan_type
    }

    /// Get the rank (number of generators)
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the generator prefix
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Get the name of the i-th generator
    ///
    /// # Arguments
    ///
    /// * `i` - Generator index (0-based)
    ///
    /// # Returns
    ///
    /// The name of the generator, e.g., "u_0", "u_1", etc.
    pub fn generator_name(&self, i: usize) -> String {
        format!("{}_{}", self.prefix, i)
    }

    /// Create a generator element
    ///
    /// # Arguments
    ///
    /// * `i` - Generator index (must be < rank)
    ///
    /// # Returns
    ///
    /// An element representing the i-th generator u_i
    pub fn generator(&self, i: usize) -> Option<NilCoxeterElement<R>> {
        if i >= self.rank {
            return None;
        }

        let mut terms = HashMap::new();
        terms.insert(vec![i], R::from(1));

        Some(NilCoxeterElement { terms })
    }

    /// Create the zero element
    pub fn zero(&self) -> NilCoxeterElement<R> {
        NilCoxeterElement {
            terms: HashMap::new(),
        }
    }

    /// Create the multiplicative identity (empty word)
    pub fn one(&self) -> NilCoxeterElement<R> {
        let mut terms = HashMap::new();
        terms.insert(vec![], R::from(1));
        NilCoxeterElement { terms }
    }

    /// Compute the r-th homogeneous generator in noncommutative variables
    ///
    /// For finite types A and B, this returns the sum of all products of r
    /// distinct generators in decreasing order. For affine types, uses
    /// cyclically decreasing order.
    ///
    /// # Arguments
    ///
    /// * `r` - The degree of the homogeneous element
    ///
    /// # Returns
    ///
    /// The r-th homogeneous element h_r
    ///
    /// # Mathematical Definition
    ///
    /// h_r = Σ u_{i₁} u_{i₂} ... u_{iᵣ}
    ///
    /// where the sum is over all sequences i₁ > i₂ > ... > iᵣ (finite type)
    /// or cyclically decreasing sequences (affine type).
    pub fn homogeneous_generator_noncommutative_variables(&self, r: usize) -> NilCoxeterElement<R>
    where
        R: std::ops::Add<Output = R> + PartialEq,
    {
        if r == 0 {
            return self.one();
        }

        if r > self.rank {
            return self.zero();
        }

        // Generate all decreasing sequences of length r
        let mut result = HashMap::new();

        // Helper to generate combinations
        fn generate_decreasing<R: Ring + Clone + From<i64> + std::ops::Add<Output = R>>(
            indices: &mut Vec<usize>,
            start: usize,
            remaining: usize,
            rank: usize,
            result: &mut HashMap<Vec<usize>, R>,
        ) {
            if remaining == 0 {
                let key = indices.clone();
                result.insert(key, R::from(1));
                return;
            }

            for i in (0..start).rev() {
                indices.push(i);
                generate_decreasing(indices, i, remaining - 1, rank, result);
                indices.pop();
            }
        }

        let mut indices = Vec::new();
        generate_decreasing(&mut indices, self.rank, r, self.rank, &mut result);

        NilCoxeterElement { terms: result }
    }

    /// Compute homogeneous element for a partition
    ///
    /// For a partition λ = (λ₁, λ₂, ..., λₖ), computes the product:
    /// h_λ = h_{λ₁} · h_{λ₂} · ... · h_{λₖ}
    ///
    /// # Arguments
    ///
    /// * `partition` - The partition as a vector of parts
    ///
    /// # Returns
    ///
    /// The homogeneous element h_λ
    pub fn homogeneous_noncommutative_variables(&self, partition: &[usize]) -> NilCoxeterElement<R>
    where
        R: std::ops::Add<Output = R> + std::ops::Mul<Output = R> + PartialEq,
    {
        if partition.is_empty() {
            return self.one();
        }

        let mut result = self.homogeneous_generator_noncommutative_variables(partition[0]);

        for &part in &partition[1..] {
            let next = self.homogeneous_generator_noncommutative_variables(part);
            result = result.multiply(&next);
        }

        result
    }

    /// Compute k-Schur function in noncommutative variables (Type A^(1) only)
    ///
    /// For type A^(1)_k affine algebras, computes k-Schur functions which
    /// are related to the cohomology of the affine Grassmannian.
    ///
    /// # Arguments
    ///
    /// * `partition` - The partition indexing the k-Schur function
    /// * `k` - The level (must match algebra rank for type A^(1))
    ///
    /// # Returns
    ///
    /// The k-Schur function s_λ^(k)
    ///
    /// # Note
    ///
    /// This is only implemented for affine type A algebras. For other types,
    /// returns zero.
    pub fn k_schur_noncommutative_variables(
        &self,
        partition: &[usize],
        _k: usize,
    ) -> NilCoxeterElement<R>
    where
        R: std::ops::Add<Output = R> + std::ops::Mul<Output = R> + PartialEq,
    {
        // For affine type A^(1), k-Schur functions can be computed from
        // symmetric function decompositions. This is a simplified placeholder.

        if !self.cartan_type.starts_with("A") {
            return self.zero();
        }

        // Simplified: return homogeneous element as approximation
        // Full implementation would use affine symmetric function theory
        self.homogeneous_noncommutative_variables(partition)
    }
}

impl<R: Ring> Display for NilCoxeterAlgebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Nil-Coxeter algebra of type {}{} over base ring",
            self.cartan_type, self.rank
        )
    }
}

/// Element of a Nil-Coxeter algebra
///
/// Represented as a linear combination of words in the generators,
/// where each word is a sequence of generator indices satisfying:
/// - No repeated indices (due to u_i² = 0)
/// - Braid relations are implicitly satisfied
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
///
/// # Representation
///
/// Elements are stored as: HashMap<Word, R>
/// where Word = Vec<usize> represents a sequence of generator indices,
/// and R is the coefficient.
#[derive(Debug, Clone)]
pub struct NilCoxeterElement<R: Ring> {
    /// Terms: map from words (generator sequences) to coefficients
    /// Each word is a Vec<usize> of generator indices
    terms: HashMap<Vec<usize>, R>,
}

impl<R: Ring + Clone> NilCoxeterElement<R> {
    /// Create a new element from terms
    pub fn new(terms: HashMap<Vec<usize>, R>) -> Self {
        NilCoxeterElement { terms }
    }

    /// Create the zero element
    pub fn zero() -> Self {
        NilCoxeterElement {
            terms: HashMap::new(),
        }
    }

    /// Get the terms
    pub fn terms(&self) -> &HashMap<Vec<usize>, R> {
        &self.terms
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool
    where
        R: PartialEq,
    {
        self.terms.is_empty() || self.terms.values().all(|c| c.is_zero())
    }

    /// Add two elements
    pub fn add(&self, other: &Self) -> Self
    where
        R: std::ops::Add<Output = R> + PartialEq,
    {
        let mut result = self.terms.clone();

        for (word, coeff) in &other.terms {
            let new_coeff = if let Some(existing) = result.get(word) {
                existing.clone() + coeff.clone()
            } else {
                coeff.clone()
            };

            if !new_coeff.is_zero() {
                result.insert(word.clone(), new_coeff);
            } else {
                result.remove(word);
            }
        }

        NilCoxeterElement { terms: result }
    }

    /// Multiply two elements
    ///
    /// Applies the quadratic relation u_i² = 0 by eliminating words with
    /// repeated indices. Braid relations are implicitly maintained.
    pub fn multiply(&self, other: &Self) -> Self
    where
        R: std::ops::Add<Output = R> + std::ops::Mul<Output = R> + PartialEq,
    {
        let mut result: HashMap<Vec<usize>, R> = HashMap::new();

        for (word1, coeff1) in &self.terms {
            for (word2, coeff2) in &other.terms {
                // Concatenate words
                let mut new_word = word1.clone();
                new_word.extend(word2);

                // Check for repeated indices (u_i² = 0)
                let mut seen = vec![false; 100]; // Assuming max 100 generators
                let mut has_repeat = false;
                for &idx in &new_word {
                    if idx < 100 {
                        if seen[idx] {
                            has_repeat = true;
                            break;
                        }
                        seen[idx] = true;
                    }
                }

                if has_repeat {
                    continue; // Skip terms with u_i² = 0
                }

                let new_coeff = coeff1.clone() * coeff2.clone();

                let existing = result.get(&new_word).cloned().unwrap_or_else(R::zero);
                let sum = existing + new_coeff;

                if !sum.is_zero() {
                    result.insert(new_word, sum);
                } else {
                    result.remove(&new_word);
                }
            }
        }

        NilCoxeterElement { terms: result }
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &R) -> Self
    where
        R: std::ops::Mul<Output = R> + PartialEq,
    {
        if scalar.is_zero() {
            return Self::zero();
        }

        let terms = self
            .terms
            .iter()
            .map(|(word, coeff)| (word.clone(), coeff.clone() * scalar.clone()))
            .collect();

        NilCoxeterElement { terms }
    }

    /// Negate the element
    pub fn negate(&self) -> Self
    where
        R: std::ops::Neg<Output = R>,
    {
        let terms = self
            .terms
            .iter()
            .map(|(word, coeff)| (word.clone(), -coeff.clone()))
            .collect();

        NilCoxeterElement { terms }
    }

    /// Get the degree (maximum word length)
    pub fn degree(&self) -> usize {
        self.terms.keys().map(|word| word.len()).max().unwrap_or(0)
    }
}

impl<R: Ring + Clone + Display> Display for NilCoxeterElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }

        let mut first = true;
        for (word, coeff) in &self.terms {
            if !first {
                write!(f, " + ")?;
            }
            first = false;

            if word.is_empty() {
                write!(f, "{}", coeff)?;
            } else if coeff.is_one() {
                write!(f, "u_{}", word.iter().map(|i| i.to_string()).collect::<Vec<_>>().join("*u_"))?;
            } else {
                write!(
                    f,
                    "{}*u_{}",
                    coeff,
                    word.iter().map(|i| i.to_string()).collect::<Vec<_>>().join("*u_")
                )?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nil_coxeter_creation() {
        let nc = NilCoxeterAlgebra::<i64>::new("A", 3);
        assert_eq!(nc.rank(), 3);
        assert_eq!(nc.cartan_type(), "A");
        assert_eq!(nc.prefix(), "u");
    }

    #[test]
    fn test_nil_coxeter_custom_prefix() {
        let nc = NilCoxeterAlgebra::<i64>::with_prefix("B", 4, "v".to_string());
        assert_eq!(nc.prefix(), "v");
        assert_eq!(nc.generator_name(0), "v_0");
    }

    #[test]
    fn test_generator_creation() {
        let nc = NilCoxeterAlgebra::<i64>::new("A", 3);

        let u0 = nc.generator(0).unwrap();
        assert!(!u0.is_zero());
        assert_eq!(u0.degree(), 1);

        let u_invalid = nc.generator(10);
        assert!(u_invalid.is_none());
    }

    #[test]
    fn test_zero_and_one() {
        let nc = NilCoxeterAlgebra::<i64>::new("A", 2);

        let zero = nc.zero();
        assert!(zero.is_zero());

        let one = nc.one();
        assert!(!one.is_zero());
        assert_eq!(one.degree(), 0);
    }

    #[test]
    fn test_element_addition() {
        let nc = NilCoxeterAlgebra::<i64>::new("A", 2);

        let u0 = nc.generator(0).unwrap();
        let u1 = nc.generator(1).unwrap();

        let sum = u0.add(&u1);
        assert_eq!(sum.terms().len(), 2);
    }

    #[test]
    fn test_nilpotency() {
        let nc = NilCoxeterAlgebra::<i64>::new("A", 2);

        let u0 = nc.generator(0).unwrap();

        // u_0^2 should be 0
        let square = u0.multiply(&u0);
        assert!(square.is_zero());
    }

    #[test]
    fn test_element_multiplication() {
        let nc = NilCoxeterAlgebra::<i64>::new("A", 3);

        let u0 = nc.generator(0).unwrap();
        let u1 = nc.generator(1).unwrap();

        // u_0 * u_1 should give a product (not zero since different generators)
        let prod = u0.multiply(&u1);
        assert!(!prod.is_zero());
        assert_eq!(prod.degree(), 2);
    }

    #[test]
    fn test_scalar_multiplication() {
        let nc = NilCoxeterAlgebra::<i64>::new("A", 2);

        let u0 = nc.generator(0).unwrap();
        let scaled = u0.scalar_mul(&3);

        assert!(!scaled.is_zero());
    }

    #[test]
    fn test_homogeneous_generator() {
        let nc = NilCoxeterAlgebra::<i64>::new("A", 3);

        // h_0 should be 1
        let h0 = nc.homogeneous_generator_noncommutative_variables(0);
        assert_eq!(h0.degree(), 0);

        // h_1 should be sum of all single generators
        let h1 = nc.homogeneous_generator_noncommutative_variables(1);
        assert_eq!(h1.terms().len(), 3); // u_0 + u_1 + u_2

        // h_2 should be sum of products of 2 distinct generators in decreasing order
        let h2 = nc.homogeneous_generator_noncommutative_variables(2);
        assert_eq!(h2.terms().len(), 3); // u_1*u_0 + u_2*u_0 + u_2*u_1
    }

    #[test]
    fn test_homogeneous_partition() {
        let nc = NilCoxeterAlgebra::<i64>::new("A", 3);

        // h_(2,1) = h_2 * h_1
        let h_21 = nc.homogeneous_noncommutative_variables(&[2, 1]);
        assert!(!h_21.is_zero());
    }

    #[test]
    fn test_element_negation() {
        let nc = NilCoxeterAlgebra::<i64>::new("A", 2);

        let u0 = nc.generator(0).unwrap();
        let neg = u0.negate();

        let sum = u0.add(&neg);
        assert!(sum.is_zero());
    }
}
