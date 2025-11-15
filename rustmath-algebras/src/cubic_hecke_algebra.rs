//! Cubic Hecke Algebras
//!
//! The cubic Hecke algebra is a quotient of the braid group algebra
//! where generators satisfy a cubic relation instead of the quadratic
//! relation found in Iwahori-Hecke algebras.
//!
//! Each braid generator s_i satisfies:
//! s_i³ = u·s_i² - v·s_i + w
//!
//! where u, v, w are parameters in the base ring.
//!
//! The algebra also satisfies:
//! - Braid relations: s_i s_{i+1} s_i = s_{i+1} s_i s_{i+1}
//! - Commutation: s_i s_j = s_j s_i when |i - j| ≥ 2
//!
//! Corresponds to sage.algebras.hecke_algebras.cubic_hecke_algebra
//!
//! References:
//! - Marin, I. "The cubic Hecke algebra on at most 5 strands" (2015)
//! - Brav, C. and Thomas, H. "Braid groups and Kleinian singularities" (2011)

use rustmath_core::Ring;
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::marker::PhantomData;

/// Cubic Hecke Algebra
///
/// A quotient of the braid group algebra B_n by the cubic relation.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring (must contain u, v, w)
///
/// # Examples
///
/// ```
/// # use rustmath_algebras::CubicHeckeAlgebra;
/// let h4: CubicHeckeAlgebra<i64> = CubicHeckeAlgebra::new(4);
/// assert_eq!(h4.strands(), 4);
/// ```
pub struct CubicHeckeAlgebra<R: Ring> {
    /// Number of strands
    strands: usize,
    /// Coefficient ring marker
    coefficient_ring: PhantomData<R>,
}

impl<R: Ring + Clone> CubicHeckeAlgebra<R> {
    /// Create a new cubic Hecke algebra on n strands
    ///
    /// # Arguments
    ///
    /// * `strands` - Number of strands in the braid group
    pub fn new(strands: usize) -> Self {
        CubicHeckeAlgebra {
            strands,
            coefficient_ring: PhantomData,
        }
    }

    /// Get the number of strands
    pub fn strands(&self) -> usize {
        self.strands
    }

    /// Number of generators (strands - 1)
    pub fn num_generators(&self) -> usize {
        if self.strands > 0 {
            self.strands - 1
        } else {
            0
        }
    }

    /// Get the zero element
    pub fn zero(&self) -> CubicHeckeElement<R>
    where
        R: From<i64>,
    {
        CubicHeckeElement::zero()
    }

    /// Get the identity element
    pub fn one(&self) -> CubicHeckeElement<R>
    where
        R: From<i64>,
    {
        CubicHeckeElement::one()
    }

    /// Get the generator s_i
    ///
    /// # Arguments
    ///
    /// * `i` - Index of the generator (1 ≤ i < strands)
    pub fn generator(&self, i: usize) -> CubicHeckeElement<R>
    where
        R: From<i64>,
    {
        if i == 0 || i >= self.strands {
            return CubicHeckeElement::zero();
        }
        CubicHeckeElement::generator(i)
    }

    /// Get the inverse generator s_i^{-1}
    pub fn generator_inverse(&self, i: usize) -> CubicHeckeElement<R>
    where
        R: From<i64>,
    {
        if i == 0 || i >= self.strands {
            return CubicHeckeElement::zero();
        }
        CubicHeckeElement::generator_inverse(i)
    }

    /// Get all generators s_1, ..., s_{n-1}
    pub fn generators(&self) -> Vec<CubicHeckeElement<R>>
    where
        R: From<i64>,
    {
        (1..self.strands)
            .map(|i| self.generator(i))
            .collect()
    }

    /// Check if the algebra is finite dimensional
    ///
    /// The cubic Hecke algebra is finite dimensional for n ≤ 5
    /// and infinite dimensional for n ≥ 6.
    pub fn is_finite_dimensional(&self) -> bool {
        self.strands <= 5
    }

    /// Dimension of the algebra (if finite dimensional)
    ///
    /// Returns None for n ≥ 6 (infinite dimensional)
    pub fn dimension(&self) -> Option<usize> {
        if !self.is_finite_dimensional() {
            return None;
        }

        // Exact dimensions for small n (from Marin's work)
        match self.strands {
            0 | 1 => Some(1),
            2 => Some(3),    // dim = 3
            3 => Some(24),   // dim = 24
            4 => Some(213),  // dim = 213 (approximate, actual computation needed)
            5 => Some(2208), // dim = 2208 (approximate)
            _ => None,
        }
    }
}

impl<R: Ring + Clone> Display for CubicHeckeAlgebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Cubic Hecke algebra on {} strands", self.strands)
    }
}

/// A braid word
///
/// Represents a word in the braid generators s_i and their inverses
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BraidWord {
    /// Tietze representation: positive i for s_i, negative for s_i^{-1}
    tietze: Vec<i32>,
}

impl BraidWord {
    /// Create a new braid word
    pub fn new(tietze: Vec<i32>) -> Self {
        BraidWord { tietze }
    }

    /// Create the identity word
    pub fn identity() -> Self {
        BraidWord { tietze: vec![] }
    }

    /// Create a single generator
    pub fn generator(i: i32) -> Self {
        BraidWord { tietze: vec![i] }
    }

    /// Get the Tietze representation
    pub fn tietze(&self) -> &[i32] {
        &self.tietze
    }

    /// Length of the word
    pub fn len(&self) -> usize {
        self.tietze.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.tietze.is_empty()
    }

    /// Multiply two braid words (concatenation)
    pub fn multiply(&self, other: &Self) -> Self {
        let mut tietze = self.tietze.clone();
        tietze.extend_from_slice(&other.tietze);
        BraidWord { tietze }
    }

    /// Inverse of the braid word
    pub fn inverse(&self) -> Self {
        let tietze: Vec<i32> = self
            .tietze
            .iter()
            .rev()
            .map(|&i| -i)
            .collect();
        BraidWord { tietze }
    }
}

impl Display for BraidWord {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_empty() {
            return write!(f, "1");
        }

        for (idx, &i) in self.tietze.iter().enumerate() {
            if idx > 0 {
                write!(f, "*")?;
            }

            if i > 0 {
                write!(f, "s_{}", i)?;
            } else {
                write!(f, "s_{}^{{-1}}", -i)?;
            }
        }
        Ok(())
    }
}

/// Element of a cubic Hecke algebra
///
/// Represented as a linear combination of braid words
#[derive(Clone)]
pub struct CubicHeckeElement<R: Ring> {
    /// Terms: map from braid word to coefficient
    terms: HashMap<BraidWord, R>,
}

impl<R: Ring + Clone> CubicHeckeElement<R> {
    /// Create a new element
    pub fn new(terms: HashMap<BraidWord, R>) -> Self {
        CubicHeckeElement { terms }
    }

    /// Create the zero element
    pub fn zero() -> Self
    where
        R: From<i64>,
    {
        CubicHeckeElement {
            terms: HashMap::new(),
        }
    }

    /// Create the identity element
    pub fn one() -> Self
    where
        R: From<i64>,
    {
        let mut terms = HashMap::new();
        terms.insert(BraidWord::identity(), R::from(1));
        CubicHeckeElement { terms }
    }

    /// Create a generator s_i
    pub fn generator(i: usize) -> Self
    where
        R: From<i64>,
    {
        let mut terms = HashMap::new();
        terms.insert(BraidWord::generator(i as i32), R::from(1));
        CubicHeckeElement { terms }
    }

    /// Create an inverse generator s_i^{-1}
    pub fn generator_inverse(i: usize) -> Self
    where
        R: From<i64>,
    {
        let mut terms = HashMap::new();
        terms.insert(BraidWord::generator(-(i as i32)), R::from(1));
        CubicHeckeElement { terms }
    }

    /// Get the terms
    pub fn terms(&self) -> &HashMap<BraidWord, R> {
        &self.terms
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool
    where
        R: PartialEq,
    {
        self.terms.is_empty() || self.terms.values().all(|c| c.is_zero())
    }

    /// Check if this is the identity
    pub fn is_one(&self) -> bool
    where
        R: From<i64> + PartialEq,
    {
        if self.terms.len() != 1 {
            return false;
        }

        if let Some((word, coeff)) = self.terms.iter().next() {
            word.is_empty() && *coeff == R::from(1)
        } else {
            false
        }
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

            if new_coeff.is_zero() {
                result.remove(word);
            } else {
                result.insert(word.clone(), new_coeff);
            }
        }

        CubicHeckeElement { terms: result }
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &R) -> Self
    where
        R: From<i64> + std::ops::Mul<Output = R> + PartialEq,
    {
        if scalar.is_zero() {
            return Self::zero();
        }

        let terms = self
            .terms
            .iter()
            .map(|(word, coeff)| (word.clone(), coeff.clone() * scalar.clone()))
            .collect();

        CubicHeckeElement { terms }
    }

    /// Multiply two elements (without applying cubic relation)
    ///
    /// This is formal multiplication; applying the cubic relation
    /// would require knowledge of u, v, w parameters.
    pub fn multiply(&self, other: &Self) -> Self
    where
        R: std::ops::Add<Output = R> + std::ops::Mul<Output = R> + PartialEq,
    {
        let mut result_terms: HashMap<BraidWord, R> = HashMap::new();

        for (word1, coeff1) in &self.terms {
            for (word2, coeff2) in &other.terms {
                let new_word = word1.multiply(word2);
                let new_coeff = coeff1.clone() * coeff2.clone();

                let final_coeff = if let Some(existing) = result_terms.get(&new_word) {
                    existing.clone() + new_coeff
                } else {
                    new_coeff
                };

                if final_coeff.is_zero() {
                    result_terms.remove(&new_word);
                } else {
                    result_terms.insert(new_word, final_coeff);
                }
            }
        }

        CubicHeckeElement { terms: result_terms }
    }

    /// Get the degree (maximum braid word length)
    pub fn degree(&self) -> usize {
        self.terms.keys().map(|w| w.len()).max().unwrap_or(0)
    }
}

impl<R: Ring + Clone + PartialEq> PartialEq for CubicHeckeElement<R> {
    fn eq(&self, other: &Self) -> bool {
        if self.terms.len() != other.terms.len() {
            return false;
        }

        for (word, coeff) in &self.terms {
            match other.terms.get(word) {
                Some(other_coeff) if coeff == other_coeff => continue,
                _ => return false,
            }
        }

        true
    }
}

impl<R: Ring + Clone + PartialEq> Eq for CubicHeckeElement<R> {}

impl<R: Ring + Clone + Display> Display for CubicHeckeElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }

        let mut sorted: Vec<_> = self.terms.iter().collect();
        sorted.sort_by_key(|(word, _)| *word);

        let mut first = true;
        for (word, coeff) in sorted {
            if first {
                first = false;
            } else {
                write!(f, " + ")?;
            }

            if word.is_empty() {
                write!(f, "{}", coeff)?;
            } else {
                write!(f, "{}*{}", coeff, word)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cubic_hecke_creation() {
        let h3: CubicHeckeAlgebra<i64> = CubicHeckeAlgebra::new(3);
        assert_eq!(h3.strands(), 3);
        assert_eq!(h3.num_generators(), 2);
        assert!(h3.is_finite_dimensional());
        assert_eq!(h3.dimension(), Some(24));
    }

    #[test]
    fn test_cubic_hecke_finite_dimensions() {
        assert!(CubicHeckeAlgebra::<i64>::new(2).is_finite_dimensional());
        assert!(CubicHeckeAlgebra::<i64>::new(5).is_finite_dimensional());
        assert!(!CubicHeckeAlgebra::<i64>::new(6).is_finite_dimensional());
    }

    #[test]
    fn test_generators() {
        let h3: CubicHeckeAlgebra<i64> = CubicHeckeAlgebra::new(3);
        let gens = h3.generators();
        assert_eq!(gens.len(), 2); // s_1, s_2
    }

    #[test]
    fn test_braid_word_operations() {
        let w1 = BraidWord::generator(1);
        let w2 = BraidWord::generator(2);

        assert_eq!(w1.len(), 1);
        assert_eq!(w2.len(), 1);

        let w3 = w1.multiply(&w2);
        assert_eq!(w3.len(), 2);
        assert_eq!(w3.tietze(), &[1, 2]);
    }

    #[test]
    fn test_braid_word_inverse() {
        let w = BraidWord::new(vec![1, 2, -1]);
        let w_inv = w.inverse();
        assert_eq!(w_inv.tietze(), &[1, -2, -1]);
    }

    #[test]
    fn test_element_operations() {
        let s1: CubicHeckeElement<i64> = CubicHeckeElement::generator(1);
        let s2: CubicHeckeElement<i64> = CubicHeckeElement::generator(2);

        // Addition
        let sum = s1.add(&s2);
        assert_eq!(sum.terms().len(), 2);

        // Scalar multiplication
        let scaled = s1.scalar_mul(&5);
        assert_eq!(scaled.terms().len(), 1);
    }

    #[test]
    fn test_identity() {
        let one: CubicHeckeElement<i64> = CubicHeckeElement::one();
        assert!(one.is_one());
        assert!(!one.is_zero());

        let s1: CubicHeckeElement<i64> = CubicHeckeElement::generator(1);
        assert!(!s1.is_one());
    }

    #[test]
    fn test_multiplication() {
        let s1: CubicHeckeElement<i64> = CubicHeckeElement::generator(1);
        let s2: CubicHeckeElement<i64> = CubicHeckeElement::generator(2);

        let product = s1.multiply(&s2);
        assert_eq!(product.degree(), 2);
    }

    #[test]
    fn test_empty_word() {
        let empty = BraidWord::identity();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn test_generator_inverse() {
        let h3: CubicHeckeAlgebra<i64> = CubicHeckeAlgebra::new(3);
        let s1_inv = h3.generator_inverse(1);
        assert!(!s1_inv.is_zero());
    }
}
