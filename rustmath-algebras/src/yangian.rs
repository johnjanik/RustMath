//! Yangian Implementation
//!
//! The Yangian Y(g) is a Hopf algebra deformation of the universal enveloping
//! algebra U(g[t]) of the polynomial current Lie algebra g[t], where g is a
//! semisimple Lie algebra.
//!
//! # Mathematical Background
//!
//! The Yangian was introduced by Drinfeld and plays a fundamental role in:
//! - Quantum integrable systems
//! - Representation theory
//! - Mathematical physics
//! - Quantum groups
//!
//! For Y(gl_n), the generators are t^(r)_{ij} where:
//! - r ≥ 0 is the level
//! - 1 ≤ i, j ≤ n are matrix indices
//!
//! The defining relations involve the R-matrix and quantum determinant.
//!
//! # Implementation
//!
//! We implement the Yangian for gl_n with generators organized by level.
//! Elements are represented as polynomials in the generators.
//!
//! # Examples
//!
//! ```
//! use rustmath_algebras::yangian::{Yangian, YangianElement};
//! use rustmath_rationals::Rational;
//!
//! // Create Yangian for gl_2
//! let yang = Yangian::<Rational>::new(2);
//!
//! // Get a generator t^(0)_{11}
//! let t = yang.generator(0, 1, 1);
//! ```

use rustmath_core::{Ring, Field};
use std::collections::HashMap;
use std::fmt;

/// Index for a Yangian generator: (level r, row i, column j)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct YangianIndex {
    /// Level (r ≥ 0)
    pub level: usize,
    /// Row index (1 ≤ i ≤ n)
    pub row: usize,
    /// Column index (1 ≤ j ≤ n)
    pub col: usize,
}

impl YangianIndex {
    /// Create a new Yangian generator index
    pub fn new(level: usize, row: usize, col: usize) -> Self {
        Self { level, row, col }
    }
}

impl fmt::Display for YangianIndex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "t^({})_{{{},{}}}", self.level, self.row, self.col)
    }
}

/// A monomial in the Yangian (product of generators)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct YangianMonomial {
    /// List of generator indices in the product
    factors: Vec<YangianIndex>,
}

impl YangianMonomial {
    /// Create a new monomial
    pub fn new(factors: Vec<YangianIndex>) -> Self {
        Self { factors }
    }

    /// Create the unit monomial (empty product)
    pub fn unit() -> Self {
        Self { factors: Vec::new() }
    }

    /// Create a monomial from a single generator
    pub fn from_generator(index: YangianIndex) -> Self {
        Self { factors: vec![index] }
    }

    /// Check if this is the unit
    pub fn is_unit(&self) -> bool {
        self.factors.is_empty()
    }

    /// Get the degree (number of factors)
    pub fn degree(&self) -> usize {
        self.factors.len()
    }

    /// Get the total level (sum of all levels)
    pub fn total_level(&self) -> usize {
        self.factors.iter().map(|idx| idx.level).sum()
    }

    /// Multiply two monomials
    pub fn multiply(&self, other: &Self) -> Self {
        let mut factors = self.factors.clone();
        factors.extend_from_slice(&other.factors);
        Self { factors }
    }
}

impl fmt::Display for YangianMonomial {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_unit() {
            return write!(f, "1");
        }
        for (i, idx) in self.factors.iter().enumerate() {
            if i > 0 {
                write!(f, "·")?;
            }
            write!(f, "{}", idx)?;
        }
        Ok(())
    }
}

/// An element of the Yangian
#[derive(Debug, Clone, PartialEq)]
pub struct YangianElement<R: Ring> {
    /// Coefficients for each monomial
    terms: HashMap<YangianMonomial, R>,
    /// Rank of gl_n
    rank: usize,
}

impl<R: Ring> YangianElement<R> {
    /// Create a new Yangian element
    pub fn new(rank: usize) -> Self {
        Self {
            terms: HashMap::new(),
            rank,
        }
    }

    /// Create from a single monomial
    pub fn from_monomial(monomial: YangianMonomial, coefficient: R, rank: usize) -> Self {
        let mut terms = HashMap::new();
        if !coefficient.is_zero() {
            terms.insert(monomial, coefficient);
        }
        Self { terms, rank }
    }

    /// Create zero element
    pub fn zero(rank: usize) -> Self {
        Self::new(rank)
    }

    /// Create unit element
    pub fn one(rank: usize) -> Self {
        Self::from_monomial(YangianMonomial::unit(), R::one(), rank)
    }

    /// Check if zero
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Check if unit
    pub fn is_one(&self) -> bool {
        if self.terms.len() != 1 {
            return false;
        }
        if let Some((monomial, coeff)) = self.terms.iter().next() {
            monomial.is_unit() && coeff.is_one()
        } else {
            false
        }
    }

    /// Get rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Add a term
    pub fn add_term(&mut self, monomial: YangianMonomial, coefficient: R) {
        if coefficient.is_zero() {
            return;
        }

        let entry = self.terms.entry(monomial).or_insert_with(R::zero);
        *entry = entry.clone() + coefficient;

        // Remove zero terms
        let keys_to_remove: Vec<_> = self.terms.iter()
            .filter(|(_, v)| v.is_zero())
            .map(|(k, _)| k.clone())
            .collect();
        for key in keys_to_remove {
            self.terms.remove(&key);
        }
    }

    /// Addition
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.rank, other.rank, "Rank mismatch");
        let mut result = self.clone();
        for (monomial, coeff) in &other.terms {
            result.add_term(monomial.clone(), coeff.clone());
        }
        result
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &R) -> Self {
        if scalar.is_zero() {
            return Self::zero(self.rank);
        }
        let mut result = Self::new(self.rank);
        for (monomial, coeff) in &self.terms {
            result.terms.insert(monomial.clone(), coeff.clone() * scalar.clone());
        }
        result
    }

    /// Negation
    pub fn negate(&self) -> Self {
        self.scalar_mul(&R::zero().sub(&R::one()))
    }

    /// Subtraction
    pub fn sub(&self, other: &Self) -> Self {
        self.add(&other.negate())
    }

    /// Naive multiplication (without applying relations)
    ///
    /// Note: The full Yangian multiplication would need to implement
    /// the RTT relations and commutation relations between generators.
    /// This is a simplified version for demonstration.
    pub fn naive_multiply(&self, other: &Self) -> Self {
        assert_eq!(self.rank, other.rank, "Rank mismatch");

        let mut result = Self::new(self.rank);
        for (m1, c1) in &self.terms {
            for (m2, c2) in &other.terms {
                let new_monomial = m1.multiply(m2);
                let new_coeff = c1.clone() * c2.clone();
                result.add_term(new_monomial, new_coeff);
            }
        }
        result
    }
}

impl<R: Ring> fmt::Display for YangianElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut terms: Vec<_> = self.terms.iter().collect();
        terms.sort_by_key(|(m, _)| m.clone());

        let mut first = true;
        for (monomial, coeff) in terms {
            if !first {
                write!(f, " + ")?;
            }
            first = false;

            if monomial.is_unit() {
                write!(f, "{}", coeff)?;
            } else if coeff.is_one() {
                write!(f, "{}", monomial)?;
            } else {
                write!(f, "{}·{}", coeff, monomial)?;
            }
        }
        Ok(())
    }
}

/// The Yangian Y(gl_n)
#[derive(Debug, Clone)]
pub struct Yangian<R: Ring> {
    /// Rank n (for gl_n)
    rank: usize,
    _phantom: std::marker::PhantomData<R>,
}

impl<R: Ring> Yangian<R> {
    /// Create a new Yangian for gl_n
    pub fn new(rank: usize) -> Self {
        assert!(rank > 0, "Rank must be positive");
        Self {
            rank,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get a generator t^(r)_{ij}
    ///
    /// # Arguments
    ///
    /// * `level` - The level r ≥ 0
    /// * `i` - Row index (1 ≤ i ≤ n)
    /// * `j` - Column index (1 ≤ j ≤ n)
    pub fn generator(&self, level: usize, i: usize, j: usize) -> YangianElement<R> {
        assert!(i >= 1 && i <= self.rank, "Row index out of range");
        assert!(j >= 1 && j <= self.rank, "Column index out of range");

        let index = YangianIndex::new(level, i, j);
        YangianElement::from_monomial(
            YangianMonomial::from_generator(index),
            R::one(),
            self.rank,
        )
    }

    /// Get zero element
    pub fn zero(&self) -> YangianElement<R> {
        YangianElement::zero(self.rank)
    }

    /// Get unit element
    pub fn one(&self) -> YangianElement<R> {
        YangianElement::one(self.rank)
    }

    /// Get all level 0 generators (these form gl_n)
    pub fn level_zero_generators(&self) -> Vec<YangianElement<R>> {
        let mut gens = Vec::new();
        for i in 1..=self.rank {
            for j in 1..=self.rank {
                gens.push(self.generator(0, i, j));
            }
        }
        gens
    }
}

/// A level-truncated Yangian (generators up to level L)
#[derive(Debug, Clone)]
pub struct YangianLevel<R: Ring> {
    /// Underlying Yangian
    yangian: Yangian<R>,
    /// Maximum level
    max_level: usize,
}

impl<R: Ring> YangianLevel<R> {
    /// Create a level-truncated Yangian
    pub fn new(rank: usize, max_level: usize) -> Self {
        Self {
            yangian: Yangian::new(rank),
            max_level,
        }
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        self.yangian.rank()
    }

    /// Get maximum level
    pub fn max_level(&self) -> usize {
        self.max_level
    }

    /// Get a generator (must be within level bound)
    pub fn generator(&self, level: usize, i: usize, j: usize) -> YangianElement<R> {
        assert!(level <= self.max_level, "Level exceeds maximum");
        self.yangian.generator(level, i, j)
    }

    /// Get zero element
    pub fn zero(&self) -> YangianElement<R> {
        self.yangian.zero()
    }

    /// Get unit element
    pub fn one(&self) -> YangianElement<R> {
        self.yangian.one()
    }
}

/// Graded Yangian base structure
#[derive(Debug, Clone)]
pub struct GradedYangianBase<R: Ring> {
    /// Underlying Yangian
    yangian: Yangian<R>,
}

impl<R: Ring> GradedYangianBase<R> {
    /// Create a new graded Yangian base
    pub fn new(rank: usize) -> Self {
        Self {
            yangian: Yangian::new(rank),
        }
    }

    /// Get the underlying Yangian
    pub fn yangian(&self) -> &Yangian<R> {
        &self.yangian
    }

    /// Get rank
    pub fn rank(&self) -> usize {
        self.yangian.rank()
    }

    /// Get graded component of given total level
    pub fn graded_component(&self, element: &YangianElement<R>, level: usize) -> YangianElement<R> {
        let mut result = YangianElement::new(self.yangian.rank());
        for (monomial, coeff) in &element.terms {
            if monomial.total_level() == level {
                result.add_term(monomial.clone(), coeff.clone());
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;
    use rustmath_integers::Integer;

    #[test]
    fn test_yangian_index() {
        let idx = YangianIndex::new(0, 1, 1);
        assert_eq!(idx.level, 0);
        assert_eq!(idx.row, 1);
        assert_eq!(idx.col, 1);
    }

    #[test]
    fn test_yangian_monomial() {
        let m1 = YangianMonomial::unit();
        assert!(m1.is_unit());
        assert_eq!(m1.degree(), 0);

        let idx = YangianIndex::new(0, 1, 2);
        let m2 = YangianMonomial::from_generator(idx);
        assert!(!m2.is_unit());
        assert_eq!(m2.degree(), 1);
    }

    #[test]
    fn test_yangian_creation() {
        let yang = Yangian::<Rational>::new(2);
        assert_eq!(yang.rank(), 2);
    }

    #[test]
    fn test_generators() {
        let yang = Yangian::<Integer>::new(2);
        let t_0_11 = yang.generator(0, 1, 1);
        let t_0_12 = yang.generator(0, 1, 2);
        let t_1_11 = yang.generator(1, 1, 1);

        assert!(!t_0_11.is_zero());
        assert!(!t_0_12.is_zero());
        assert!(!t_1_11.is_zero());
    }

    #[test]
    fn test_addition() {
        let yang = Yangian::<Rational>::new(2);
        let t1 = yang.generator(0, 1, 1);
        let t2 = yang.generator(0, 1, 2);

        let sum = t1.add(&t2);
        assert!(!sum.is_zero());

        let diff = sum.sub(&sum);
        assert!(diff.is_zero());
    }

    #[test]
    fn test_scalar_multiplication() {
        let yang = Yangian::<Rational>::new(2);
        let t = yang.generator(0, 1, 1);

        let two = Rational::from(2);
        let scaled = t.scalar_mul(&two);
        assert!(!scaled.is_zero());
    }

    #[test]
    fn test_naive_multiplication() {
        let yang = Yangian::<Integer>::new(2);
        let t1 = yang.generator(0, 1, 1);
        let t2 = yang.generator(0, 1, 2);

        let product = t1.naive_multiply(&t2);
        assert!(!product.is_zero());
    }

    #[test]
    fn test_unit_element() {
        let yang = Yangian::<Rational>::new(2);
        let one = yang.one();
        let t = yang.generator(0, 1, 1);

        // 1 * t = t (naive multiplication)
        let product = one.naive_multiply(&t);
        assert_eq!(product, t);
    }

    #[test]
    fn test_zero_properties() {
        let yang = Yangian::<Rational>::new(2);
        let zero = yang.zero();
        let t = yang.generator(0, 1, 1);

        // 0 + t = t
        assert_eq!(zero.add(&t), t);

        // 0 * t = 0
        let product = zero.naive_multiply(&t);
        assert!(product.is_zero());
    }

    #[test]
    fn test_level_zero_generators() {
        let yang = Yangian::<Integer>::new(2);
        let gens = yang.level_zero_generators();
        assert_eq!(gens.len(), 4); // For gl_2, we have 2×2 = 4 generators at level 0
    }

    #[test]
    fn test_yangian_level() {
        let yang_level = YangianLevel::<Rational>::new(2, 3);
        assert_eq!(yang_level.rank(), 2);
        assert_eq!(yang_level.max_level(), 3);

        let t = yang_level.generator(2, 1, 1);
        assert!(!t.is_zero());
    }

    #[test]
    #[should_panic(expected = "Level exceeds maximum")]
    fn test_yangian_level_exceeds() {
        let yang_level = YangianLevel::<Rational>::new(2, 3);
        let _ = yang_level.generator(5, 1, 1); // Should panic
    }

    #[test]
    fn test_graded_yangian_base() {
        let graded = GradedYangianBase::<Integer>::new(2);
        assert_eq!(graded.rank(), 2);

        let yang = graded.yangian();
        let t0 = yang.generator(0, 1, 1);
        let t1 = yang.generator(1, 1, 1);

        // Create element with mixed levels
        let mixed = t0.add(&t1);

        // Extract level 0 component
        let level0_part = graded.graded_component(&mixed, 0);
        assert!(!level0_part.is_zero());

        // Extract level 1 component
        let level1_part = graded.graded_component(&mixed, 1);
        assert!(!level1_part.is_zero());
    }

    #[test]
    fn test_monomial_multiply() {
        let idx1 = YangianIndex::new(0, 1, 1);
        let idx2 = YangianIndex::new(1, 2, 2);

        let m1 = YangianMonomial::from_generator(idx1);
        let m2 = YangianMonomial::from_generator(idx2);

        let product = m1.multiply(&m2);
        assert_eq!(product.degree(), 2);
        assert_eq!(product.total_level(), 1);
    }

    #[test]
    fn test_distributivity() {
        let yang = Yangian::<Rational>::new(2);
        let t1 = yang.generator(0, 1, 1);
        let t2 = yang.generator(0, 1, 2);
        let t3 = yang.generator(0, 2, 1);

        // t1 * (t2 + t3) = t1*t2 + t1*t3
        let sum = t2.add(&t3);
        let left = t1.naive_multiply(&sum);
        let right = t1.naive_multiply(&t2).add(&t1.naive_multiply(&t3));

        assert_eq!(left, right);
    }
}
