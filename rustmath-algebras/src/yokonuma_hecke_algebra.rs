//! Yokonuma-Hecke Algebra Implementation
//!
//! The Yokonuma-Hecke algebra is a generalization of the Iwahori-Hecke algebra.
//! It is the algebra associated to the complex reflection group G(r, 1, n),
//! which consists of monomial matrices with r-th roots of unity as nonzero entries.
//!
//! # Mathematical Background
//!
//! The Yokonuma-Hecke algebra Y_{r,n}(q) is defined over a ring containing
//! a parameter q and has:
//! - Generators: g_1, ..., g_{n-1} (like the Hecke algebra)
//! - Additional generators: t_1, ..., t_n (the "framing" or "Jucys-Murphy" elements)
//! - Parameter r: the "modular parameter"
//!
//! Relations:
//! 1. (g_i - q)(g_i + 1) = 0 (quadratic relation)
//! 2. g_i g_{i+1} g_i = g_{i+1} g_i g_{i+1} (braid relation)
//! 3. g_i g_j = g_j g_i for |i-j| > 1 (distant generators commute)
//! 4. t_i^r = 1 (r-th roots of unity)
//! 5. t_i t_j = t_j t_i (framing generators commute)
//! 6. g_i t_i g_i = t_{i+1}
//! 7. g_i t_j = t_j g_i for j ≠ i, i+1
//!
//! # Examples
//!
//! ```
//! use rustmath_algebras::yokonuma_hecke_algebra::{YokonumaHeckeAlgebra, YokonumaElement};
//! use rustmath_rationals::Rational;
//!
//! // Create Yokonuma-Hecke algebra for n=3, r=2
//! let yok = YokonumaHeckeAlgebra::<Rational>::new(3, 2);
//!
//! // Get generators
//! let g1 = yok.g_generator(1);
//! let t1 = yok.t_generator(1);
//! ```

use rustmath_core::Ring;
use std::collections::HashMap;
use std::fmt;

/// Generator type for Yokonuma-Hecke algebra
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum YokonumaGenerator {
    /// Hecke generator g_i (1 ≤ i < n)
    G(usize),
    /// Framing generator t_i (1 ≤ i ≤ n)
    T(usize),
}

impl fmt::Display for YokonumaGenerator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            YokonumaGenerator::G(i) => write!(f, "g_{}", i),
            YokonumaGenerator::T(i) => write!(f, "t_{}", i),
        }
    }
}

/// A word (monomial) in the Yokonuma-Hecke algebra
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct YokonumaWord {
    /// Sequence of generators
    generators: Vec<YokonumaGenerator>,
}

impl YokonumaWord {
    /// Create a new word
    pub fn new(generators: Vec<YokonumaGenerator>) -> Self {
        Self { generators }
    }

    /// Create the empty word (identity)
    pub fn identity() -> Self {
        Self { generators: Vec::new() }
    }

    /// Check if this is the identity
    pub fn is_identity(&self) -> bool {
        self.generators.is_empty()
    }

    /// Get length of the word
    pub fn length(&self) -> usize {
        self.generators.len()
    }

    /// Concatenate two words
    pub fn concatenate(&self, other: &Self) -> Self {
        let mut generators = self.generators.clone();
        generators.extend_from_slice(&other.generators);
        Self { generators }
    }

    /// Get the generators
    pub fn generators(&self) -> &[YokonumaGenerator] {
        &self.generators
    }
}

impl fmt::Display for YokonumaWord {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_identity() {
            return write!(f, "1");
        }
        for (i, gen) in self.generators.iter().enumerate() {
            if i > 0 {
                write!(f, "·")?;
            }
            write!(f, "{}", gen)?;
        }
        Ok(())
    }
}

/// An element of the Yokonuma-Hecke algebra
#[derive(Debug, Clone, PartialEq)]
pub struct YokonumaElement<R: Ring> {
    /// Coefficients for each word
    terms: HashMap<YokonumaWord, R>,
    /// Number of strands n
    n: usize,
    /// Modular parameter r
    r: usize,
}

impl<R: Ring> YokonumaElement<R> {
    /// Create a new element
    pub fn new(n: usize, r: usize) -> Self {
        Self {
            terms: HashMap::new(),
            n,
            r,
        }
    }

    /// Create from a single word
    pub fn from_word(word: YokonumaWord, coefficient: R, n: usize, r: usize) -> Self {
        let mut terms = HashMap::new();
        if !coefficient.is_zero() {
            terms.insert(word, coefficient);
        }
        Self { terms, n, r }
    }

    /// Create zero element
    pub fn zero(n: usize, r: usize) -> Self {
        Self::new(n, r)
    }

    /// Create unit element
    pub fn one(n: usize, r: usize) -> Self {
        Self::from_word(YokonumaWord::identity(), R::one(), n, r)
    }

    /// Check if zero
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Check if one
    pub fn is_one(&self) -> bool {
        if self.terms.len() != 1 {
            return false;
        }
        if let Some((word, coeff)) = self.terms.iter().next() {
            word.is_identity() && coeff.is_one()
        } else {
            false
        }
    }

    /// Get n
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get r
    pub fn r(&self) -> usize {
        self.r
    }

    /// Add a term
    pub fn add_term(&mut self, word: YokonumaWord, coefficient: R) {
        if coefficient.is_zero() {
            return;
        }

        let entry = self.terms.entry(word).or_insert_with(R::zero);
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
        assert_eq!((self.n, self.r), (other.n, other.r), "Parameter mismatch");
        let mut result = self.clone();
        for (word, coeff) in &other.terms {
            result.add_term(word.clone(), coeff.clone());
        }
        result
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &R) -> Self {
        if scalar.is_zero() {
            return Self::zero(self.n, self.r);
        }
        let mut result = Self::new(self.n, self.r);
        for (word, coeff) in &self.terms {
            result.terms.insert(word.clone(), coeff.clone() * scalar.clone());
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

    /// Naive multiplication (concatenation without applying relations)
    ///
    /// Note: A full implementation would need to apply the defining relations
    /// to reduce products to a normal form.
    pub fn naive_multiply(&self, other: &Self) -> Self {
        assert_eq!((self.n, self.r), (other.n, other.r), "Parameter mismatch");

        let mut result = Self::new(self.n, self.r);
        for (w1, c1) in &self.terms {
            for (w2, c2) in &other.terms {
                let new_word = w1.concatenate(w2);
                let new_coeff = c1.clone() * c2.clone();
                result.add_term(new_word, new_coeff);
            }
        }
        result
    }
}

impl<R: Ring> fmt::Display for YokonumaElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut terms: Vec<_> = self.terms.iter().collect();
        terms.sort_by(|(w1, _), (w2, _)| {
            (w1.length(), w1.generators()).cmp(&(w2.length(), w2.generators()))
        });

        let mut first = true;
        for (word, coeff) in terms {
            if !first {
                write!(f, " + ")?;
            }
            first = false;

            if word.is_identity() {
                write!(f, "{}", coeff)?;
            } else if coeff.is_one() {
                write!(f, "{}", word)?;
            } else {
                write!(f, "{}·{}", coeff, word)?;
            }
        }
        Ok(())
    }
}

/// The Yokonuma-Hecke algebra Y_{r,n}(q)
#[derive(Debug, Clone)]
pub struct YokonumaHeckeAlgebra<R: Ring> {
    /// Number of strands
    n: usize,
    /// Modular parameter (order of framing generators)
    r: usize,
    /// The parameter q (stored as an element)
    q: R,
}

impl<R: Ring> YokonumaHeckeAlgebra<R> {
    /// Create a new Yokonuma-Hecke algebra
    ///
    /// # Arguments
    ///
    /// * `n` - Number of strands (n ≥ 2)
    /// * `r` - Modular parameter (r ≥ 1)
    ///
    /// The parameter q is set to a formal parameter (1 by default).
    pub fn new(n: usize, r: usize) -> Self {
        assert!(n >= 2, "n must be at least 2");
        assert!(r >= 1, "r must be at least 1");
        Self {
            n,
            r,
            q: R::one(),
        }
    }

    /// Create with a specific q parameter
    pub fn with_q(n: usize, r: usize, q: R) -> Self {
        assert!(n >= 2, "n must be at least 2");
        assert!(r >= 1, "r must be at least 1");
        Self { n, r, q }
    }

    /// Get n
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get r
    pub fn r(&self) -> usize {
        self.r
    }

    /// Get q parameter
    pub fn q(&self) -> &R {
        &self.q
    }

    /// Get the g_i generator (1 ≤ i < n)
    pub fn g_generator(&self, i: usize) -> YokonumaElement<R> {
        assert!(i >= 1 && i < self.n, "Generator index out of range");
        YokonumaElement::from_word(
            YokonumaWord::new(vec![YokonumaGenerator::G(i)]),
            R::one(),
            self.n,
            self.r,
        )
    }

    /// Get the t_i generator (1 ≤ i ≤ n)
    pub fn t_generator(&self, i: usize) -> YokonumaElement<R> {
        assert!(i >= 1 && i <= self.n, "Generator index out of range");
        YokonumaElement::from_word(
            YokonumaWord::new(vec![YokonumaGenerator::T(i)]),
            R::one(),
            self.n,
            self.r,
        )
    }

    /// Get all g generators
    pub fn g_generators(&self) -> Vec<YokonumaElement<R>> {
        (1..self.n).map(|i| self.g_generator(i)).collect()
    }

    /// Get all t generators
    pub fn t_generators(&self) -> Vec<YokonumaElement<R>> {
        (1..=self.n).map(|i| self.t_generator(i)).collect()
    }

    /// Get zero element
    pub fn zero(&self) -> YokonumaElement<R> {
        YokonumaElement::zero(self.n, self.r)
    }

    /// Get unit element
    pub fn one(&self) -> YokonumaElement<R> {
        YokonumaElement::one(self.n, self.r)
    }

    /// Get the quadratic relation element (g_i - q)(g_i + 1)
    ///
    /// This should equal zero in the algebra
    pub fn quadratic_relation(&self, i: usize) -> YokonumaElement<R> {
        let gi = self.g_generator(i);
        let one = self.one();

        // (g_i - q)
        let factor1 = gi.sub(&one.scalar_mul(&self.q));

        // (g_i + 1)
        let factor2 = gi.add(&one);

        // (g_i - q)(g_i + 1)
        factor1.naive_multiply(&factor2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;
    use rustmath_integers::Integer;

    #[test]
    fn test_generator_types() {
        let g1 = YokonumaGenerator::G(1);
        let t1 = YokonumaGenerator::T(1);

        assert_ne!(g1, t1);
    }

    #[test]
    fn test_word_creation() {
        let w1 = YokonumaWord::identity();
        assert!(w1.is_identity());
        assert_eq!(w1.length(), 0);

        let w2 = YokonumaWord::new(vec![YokonumaGenerator::G(1), YokonumaGenerator::T(1)]);
        assert!(!w2.is_identity());
        assert_eq!(w2.length(), 2);
    }

    #[test]
    fn test_word_concatenation() {
        let w1 = YokonumaWord::new(vec![YokonumaGenerator::G(1)]);
        let w2 = YokonumaWord::new(vec![YokonumaGenerator::T(1)]);

        let w3 = w1.concatenate(&w2);
        assert_eq!(w3.length(), 2);
    }

    #[test]
    fn test_algebra_creation() {
        let yok = YokonumaHeckeAlgebra::<Rational>::new(3, 2);
        assert_eq!(yok.n(), 3);
        assert_eq!(yok.r(), 2);
    }

    #[test]
    fn test_generators() {
        let yok = YokonumaHeckeAlgebra::<Integer>::new(4, 2);

        let g1 = yok.g_generator(1);
        let g2 = yok.g_generator(2);
        let t1 = yok.t_generator(1);
        let t2 = yok.t_generator(2);

        assert!(!g1.is_zero());
        assert!(!g2.is_zero());
        assert!(!t1.is_zero());
        assert!(!t2.is_zero());
    }

    #[test]
    fn test_addition() {
        let yok = YokonumaHeckeAlgebra::<Rational>::new(3, 2);
        let g1 = yok.g_generator(1);
        let t1 = yok.t_generator(1);

        let sum = g1.add(&t1);
        assert!(!sum.is_zero());

        let diff = sum.sub(&sum);
        assert!(diff.is_zero());
    }

    #[test]
    fn test_scalar_multiplication() {
        let yok = YokonumaHeckeAlgebra::<Rational>::new(3, 2);
        let g1 = yok.g_generator(1);

        let two = Rational::from(2);
        let scaled = g1.scalar_mul(&two);
        assert!(!scaled.is_zero());
    }

    #[test]
    fn test_naive_multiplication() {
        let yok = YokonumaHeckeAlgebra::<Integer>::new(3, 2);
        let g1 = yok.g_generator(1);
        let g2 = yok.g_generator(2);

        let product = g1.naive_multiply(&g2);
        assert!(!product.is_zero());
    }

    #[test]
    fn test_unit_properties() {
        let yok = YokonumaHeckeAlgebra::<Rational>::new(3, 2);
        let one = yok.one();
        let g1 = yok.g_generator(1);

        // 1 * g1 = g1
        let prod1 = one.naive_multiply(&g1);
        assert_eq!(prod1, g1);

        // g1 * 1 = g1
        let prod2 = g1.naive_multiply(&one);
        assert_eq!(prod2, g1);
    }

    #[test]
    fn test_zero_properties() {
        let yok = YokonumaHeckeAlgebra::<Rational>::new(3, 2);
        let zero = yok.zero();
        let g1 = yok.g_generator(1);

        // 0 + g1 = g1
        assert_eq!(zero.add(&g1), g1);

        // 0 * g1 = 0
        let product = zero.naive_multiply(&g1);
        assert!(product.is_zero());
    }

    #[test]
    fn test_all_generators() {
        let yok = YokonumaHeckeAlgebra::<Integer>::new(4, 2);

        let g_gens = yok.g_generators();
        assert_eq!(g_gens.len(), 3); // g_1, g_2, g_3 for n=4

        let t_gens = yok.t_generators();
        assert_eq!(t_gens.len(), 4); // t_1, t_2, t_3, t_4 for n=4
    }

    #[test]
    fn test_with_q_parameter() {
        let q = Rational::from(2);
        let yok = YokonumaHeckeAlgebra::with_q(3, 2, q.clone());
        assert_eq!(yok.q(), &q);
    }

    #[test]
    #[should_panic(expected = "n must be at least 2")]
    fn test_invalid_n() {
        let _ = YokonumaHeckeAlgebra::<Rational>::new(1, 2);
    }

    #[test]
    #[should_panic(expected = "r must be at least 1")]
    fn test_invalid_r() {
        let _ = YokonumaHeckeAlgebra::<Rational>::new(3, 0);
    }

    #[test]
    #[should_panic(expected = "Generator index out of range")]
    fn test_invalid_g_generator() {
        let yok = YokonumaHeckeAlgebra::<Rational>::new(3, 2);
        let _ = yok.g_generator(5);
    }

    #[test]
    #[should_panic(expected = "Generator index out of range")]
    fn test_invalid_t_generator() {
        let yok = YokonumaHeckeAlgebra::<Rational>::new(3, 2);
        let _ = yok.t_generator(5);
    }

    #[test]
    fn test_distributivity() {
        let yok = YokonumaHeckeAlgebra::<Rational>::new(3, 2);
        let g1 = yok.g_generator(1);
        let t1 = yok.t_generator(1);
        let t2 = yok.t_generator(2);

        // g1 * (t1 + t2) = g1*t1 + g1*t2
        let sum = t1.add(&t2);
        let left = g1.naive_multiply(&sum);
        let right = g1.naive_multiply(&t1).add(&g1.naive_multiply(&t2));

        assert_eq!(left, right);
    }

    #[test]
    fn test_associativity() {
        let yok = YokonumaHeckeAlgebra::<Integer>::new(3, 2);
        let g1 = yok.g_generator(1);
        let g2 = yok.g_generator(2);
        let t1 = yok.t_generator(1);

        // (g1 * g2) * t1 = g1 * (g2 * t1)
        let left = g1.naive_multiply(&g2).naive_multiply(&t1);
        let right = g1.naive_multiply(&g2.naive_multiply(&t1));

        assert_eq!(left, right);
    }
}
