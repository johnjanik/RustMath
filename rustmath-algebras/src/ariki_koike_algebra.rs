//! Ariki-Koike Algebras (Hecke Algebras of Type G(r,1,n))
//!
//! The Ariki-Koike algebra is a deformation of the group algebra of the
//! complex reflection group G(r,1,n) = ℤ/rℤ ≀ 𝔖_n (wreath product).
//!
//! This generalizes the Iwahori-Hecke algebra of type A (r=1) and type B (r=2).
//!
//! Generators: T₀, T₁, ..., T_{n-1}
//!
//! Relations:
//! 1. Cyclotomic: ∏_{i=0}^{r-1} (T₀ - u_i) = 0
//! 2. Quadratic: T_i² = (q - 1)T_i + q for 1 ≤ i < n
//! 3. Braid: T₀T₁T₀T₁ = T₁T₀T₁T₀
//!          T_i T_{i+1} T_i = T_{i+1} T_i T_{i+1} for i ≥ 1
//! 4. Commutation: T_i T_j = T_j T_i when |i-j| ≥ 2
//!
//! Dimension: r^n · n!
//!
//! Corresponds to sage.algebras.hecke_algebras.ariki_koike_algebra
//!
//! References:
//! - Ariki, S. "On the decomposition numbers of the Hecke algebra of G(m,1,n)" (1996)
//! - Broué, M. and Malle, G. "Zyklotomische Heckealgebren" (1993)

use rustmath_core::Ring;
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::marker::PhantomData;

/// Ariki-Koike Algebra
///
/// The Hecke algebra of the complex reflection group G(r,1,n).
///
/// # Type Parameters
///
/// * `R` - The coefficient ring (must contain q and u₀, ..., u_{r-1})
///
/// # Examples
///
/// ```
/// # use rustmath_algebras::ArikiKoikeAlgebra;
/// // Type A Hecke algebra: H_n(q) when r=1
/// let h3: ArikiKoikeAlgebra<i64> = ArikiKoikeAlgebra::new(1, 3);
/// assert_eq!(h3.rank(), 1);
/// assert_eq!(h3.degree(), 3);
/// ```
pub struct ArikiKoikeAlgebra<R: Ring> {
    /// Rank r (cyclotomic parameter)
    rank: usize,
    /// Degree n (number of symmetric group generators)
    degree: usize,
    /// Coefficient ring marker
    coefficient_ring: PhantomData<R>,
}

impl<R: Ring + Clone> ArikiKoikeAlgebra<R> {
    /// Create a new Ariki-Koike algebra H_{r,n}
    ///
    /// # Arguments
    ///
    /// * `rank` - The rank r (number of cyclotomic parameters)
    /// * `degree` - The degree n (from symmetric group S_n)
    pub fn new(rank: usize, degree: usize) -> Self {
        ArikiKoikeAlgebra {
            rank,
            degree,
            coefficient_ring: PhantomData,
        }
    }

    /// Get the rank r
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the degree n
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Dimension of the algebra: r^n · n!
    pub fn dimension(&self) -> usize {
        let mut factorial = 1;
        for i in 1..=self.degree {
            factorial *= i;
        }
        self.rank.pow(self.degree as u32) * factorial
    }

    /// Get the zero element
    pub fn zero(&self) -> ArikiKoikeElement<R>
    where
        R: From<i64>,
    {
        ArikiKoikeElement::zero()
    }

    /// Get the identity element
    pub fn one(&self) -> ArikiKoikeElement<R>
    where
        R: From<i64>,
    {
        ArikiKoikeElement::one()
    }

    /// Get the generator T_i
    ///
    /// # Arguments
    ///
    /// * `i` - Index of the generator (0 ≤ i < n)
    pub fn generator(&self, i: usize) -> ArikiKoikeElement<R>
    where
        R: From<i64>,
    {
        if i >= self.degree {
            return ArikiKoikeElement::zero();
        }
        ArikiKoikeElement::generator(i)
    }

    /// Get all generators T₀, T₁, ..., T_{n-1}
    pub fn generators(&self) -> Vec<ArikiKoikeElement<R>>
    where
        R: From<i64>,
    {
        (0..self.degree)
            .map(|i| self.generator(i))
            .collect()
    }

    /// Compute the Jucys-Murphy element L_i
    ///
    /// L_i = q^{-i+1} T_{i-1} ⋯ T₁ T₀ T₁ ⋯ T_{i-1}
    ///
    /// These elements form a commutative subalgebra.
    pub fn jucys_murphy_element(&self, i: usize) -> ArikiKoikeElement<R>
    where
        R: From<i64> + std::ops::Add<Output = R> + std::ops::Mul<Output = R> + PartialEq,
    {
        if i == 0 || i >= self.degree {
            return ArikiKoikeElement::one();
        }

        // Start with T₀
        let mut result = self.generator(0);

        // Multiply by T₁, T₂, ..., T_{i-1}
        for j in 1..i {
            let tj = self.generator(j);
            result = result.multiply(&tj);
        }

        // Multiply by T_{i-1}, ..., T₁ (reverse)
        for j in (1..i).rev() {
            let tj = self.generator(j);
            result = result.multiply(&tj);
        }

        // Would multiply by q^{-i+1}, but we don't have q in our ring
        // In a full implementation, this would involve the parameter q
        result
    }

    /// Check if this is a type A Hecke algebra (rank = 1)
    pub fn is_type_a(&self) -> bool {
        self.rank == 1
    }

    /// Check if this is a type B Hecke algebra (rank = 2)
    pub fn is_type_b(&self) -> bool {
        self.rank == 2
    }
}

impl<R: Ring + Clone> Display for ArikiKoikeAlgebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.rank == 1 {
            write!(f, "Iwahori-Hecke algebra of type A_{}", self.degree - 1)
        } else if self.rank == 2 {
            write!(f, "Iwahori-Hecke algebra of type B_{}", self.degree)
        } else {
            write!(
                f,
                "Ariki-Koike algebra H_{{{}}}(G({},1,{}))",
                self.rank, self.rank, self.degree
            )
        }
    }
}

/// Basis type for Ariki-Koike algebra
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArikiKoikeBasis {
    /// T-basis (generator monomials)
    T,
    /// LT-basis (Jucys-Murphy basis)
    LT,
}

/// A word in the Ariki-Koike generators
///
/// Represents a monomial T_{i₁} T_{i₂} ⋯ T_{iₖ}
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ArikiKoikeWord {
    /// Sequence of generator indices
    indices: Vec<usize>,
}

impl ArikiKoikeWord {
    /// Create a new word
    pub fn new(indices: Vec<usize>) -> Self {
        ArikiKoikeWord { indices }
    }

    /// Create the empty word (identity)
    pub fn identity() -> Self {
        ArikiKoikeWord { indices: vec![] }
    }

    /// Create a single generator
    pub fn generator(i: usize) -> Self {
        ArikiKoikeWord { indices: vec![i] }
    }

    /// Get the indices
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Length of the word
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Multiply two words (concatenation)
    pub fn multiply(&self, other: &Self) -> Self {
        let mut indices = self.indices.clone();
        indices.extend_from_slice(&other.indices);
        ArikiKoikeWord { indices }
    }
}

impl Display for ArikiKoikeWord {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_empty() {
            return write!(f, "1");
        }

        for (i, &idx) in self.indices.iter().enumerate() {
            if i > 0 {
                write!(f, "*")?;
            }
            write!(f, "T_{}", idx)?;
        }
        Ok(())
    }
}

/// Element of an Ariki-Koike algebra
///
/// Represented as a linear combination of words in the generators
#[derive(Clone)]
pub struct ArikiKoikeElement<R: Ring> {
    /// Terms: map from word to coefficient
    terms: HashMap<ArikiKoikeWord, R>,
}

impl<R: Ring + Clone> ArikiKoikeElement<R> {
    /// Create a new element
    pub fn new(terms: HashMap<ArikiKoikeWord, R>) -> Self {
        ArikiKoikeElement { terms }
    }

    /// Create the zero element
    pub fn zero() -> Self
    where
        R: From<i64>,
    {
        ArikiKoikeElement {
            terms: HashMap::new(),
        }
    }

    /// Create the identity element
    pub fn one() -> Self
    where
        R: From<i64>,
    {
        let mut terms = HashMap::new();
        terms.insert(ArikiKoikeWord::identity(), R::from(1));
        ArikiKoikeElement { terms }
    }

    /// Create a generator T_i
    pub fn generator(i: usize) -> Self
    where
        R: From<i64>,
    {
        let mut terms = HashMap::new();
        terms.insert(ArikiKoikeWord::generator(i), R::from(1));
        ArikiKoikeElement { terms }
    }

    /// Get the terms
    pub fn terms(&self) -> &HashMap<ArikiKoikeWord, R> {
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

        if let Some((&ref word, coeff)) = self.terms.iter().next() {
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

        ArikiKoikeElement { terms: result }
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

        ArikiKoikeElement { terms }
    }

    /// Multiply two elements (without applying relations)
    ///
    /// This is just formal multiplication; applying the Hecke relations
    /// would require knowledge of q and u parameters.
    pub fn multiply(&self, other: &Self) -> Self
    where
        R: std::ops::Add<Output = R> + std::ops::Mul<Output = R> + PartialEq,
    {
        let mut result_terms: HashMap<ArikiKoikeWord, R> = HashMap::new();

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

        ArikiKoikeElement { terms: result_terms }
    }

    /// Get the degree (maximum word length)
    pub fn degree(&self) -> usize {
        self.terms.keys().map(|w| w.len()).max().unwrap_or(0)
    }
}

impl<R: Ring + Clone + PartialEq> PartialEq for ArikiKoikeElement<R> {
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

impl<R: Ring + Clone + PartialEq> Eq for ArikiKoikeElement<R> {}

impl<R: Ring + Clone + Display> Display for ArikiKoikeElement<R> {
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
    fn test_ariki_koike_creation() {
        // Type A Hecke algebra H_3
        let h3: ArikiKoikeAlgebra<i64> = ArikiKoikeAlgebra::new(1, 3);
        assert_eq!(h3.rank(), 1);
        assert_eq!(h3.degree(), 3);
        assert_eq!(h3.dimension(), 6); // 1^3 * 3! = 6
        assert!(h3.is_type_a());
        assert!(!h3.is_type_b());
    }

    #[test]
    fn test_ariki_koike_type_b() {
        // Type B Hecke algebra
        let hb: ArikiKoikeAlgebra<i64> = ArikiKoikeAlgebra::new(2, 3);
        assert_eq!(hb.rank(), 2);
        assert_eq!(hb.degree(), 3);
        assert_eq!(hb.dimension(), 48); // 2^3 * 3! = 48
        assert!(!hb.is_type_a());
        assert!(hb.is_type_b());
    }

    #[test]
    fn test_ariki_koike_general() {
        // General case G(3,1,2)
        let h: ArikiKoikeAlgebra<i64> = ArikiKoikeAlgebra::new(3, 2);
        assert_eq!(h.dimension(), 18); // 3^2 * 2! = 18
    }

    #[test]
    fn test_generators() {
        let h3: ArikiKoikeAlgebra<i64> = ArikiKoikeAlgebra::new(1, 3);
        let gens = h3.generators();
        assert_eq!(gens.len(), 3);
    }

    #[test]
    fn test_word_operations() {
        let w1 = ArikiKoikeWord::generator(0);
        let w2 = ArikiKoikeWord::generator(1);

        assert_eq!(w1.len(), 1);
        assert_eq!(w2.len(), 1);

        let w3 = w1.multiply(&w2);
        assert_eq!(w3.len(), 2);
        assert_eq!(w3.indices(), &[0, 1]);
    }

    #[test]
    fn test_element_operations() {
        let t0: ArikiKoikeElement<i64> = ArikiKoikeElement::generator(0);
        let t1: ArikiKoikeElement<i64> = ArikiKoikeElement::generator(1);

        // Addition
        let sum = t0.add(&t1);
        assert_eq!(sum.terms().len(), 2);

        // Scalar multiplication
        let scaled = t0.scalar_mul(&5);
        assert_eq!(scaled.terms().len(), 1);
    }

    #[test]
    fn test_identity() {
        let one: ArikiKoikeElement<i64> = ArikiKoikeElement::one();
        assert!(one.is_one());
        assert!(!one.is_zero());

        let t0: ArikiKoikeElement<i64> = ArikiKoikeElement::generator(0);
        assert!(!t0.is_one());
    }

    #[test]
    fn test_multiplication() {
        let t0: ArikiKoikeElement<i64> = ArikiKoikeElement::generator(0);
        let t1: ArikiKoikeElement<i64> = ArikiKoikeElement::generator(1);

        let product = t0.multiply(&t1);
        assert_eq!(product.degree(), 2);
    }

    #[test]
    fn test_jucys_murphy() {
        let h3: ArikiKoikeAlgebra<i64> = ArikiKoikeAlgebra::new(1, 3);

        // L_0 should be identity
        let l0 = h3.jucys_murphy_element(0);
        assert!(l0.is_one());

        // L_1 = T_0 (after scaling)
        let l1 = h3.jucys_murphy_element(1);
        assert!(!l1.is_zero());
    }

    #[test]
    fn test_empty_word() {
        let empty = ArikiKoikeWord::identity();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
    }
}
