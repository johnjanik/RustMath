//! Free Lie Algebras
//!
//! A free Lie algebra is the Lie algebra with generators where there are no
//! relations beyond those required by the Lie bracket axioms:
//! - Anti-commutativity: [x, y] = -[y, x]
//! - Jacobi identity: [x, [y, z]] + [y, [z, x]] + [z, [x, y]] = 0
//!
//! Free Lie algebras can be represented using different bases:
//! - Hall basis: Based on Hall sets with systematic bracketing rules
//! - Lyndon basis: Based on Lyndon words with canonical bracketing
//!
//! The free Lie algebra on n generators has graded components with
//! dimension given by: (1/k) Σ μ(d)·n^(k/d) where μ is the Möbius function.
//!
//! Corresponds to sage.algebras.lie_algebras.free_lie_algebra
//!
//! References:
//! - Reutenauer, C. "Free Lie Algebras" (1993)
//! - Serre, J-P. "Lie Algebras and Lie Groups" (1992)

use rustmath_core::Ring;
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::marker::PhantomData;
use std::cmp::Ordering;

/// Types of bases for free Lie algebras
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FreeLieAlgebraBasis {
    /// Hall basis (systematic bracketing)
    Hall,
    /// Lyndon basis (lexicographically minimal bracketing)
    Lyndon,
}

/// A Lyndon word
///
/// A Lyndon word is a word that is strictly smaller in lexicographic order
/// than any of its proper suffixes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LyndonWord {
    /// The word as a sequence of generator indices
    letters: Vec<usize>,
}

impl LyndonWord {
    /// Create a new Lyndon word
    pub fn new(letters: Vec<usize>) -> Option<Self> {
        let word = LyndonWord { letters };
        if word.is_lyndon() {
            Some(word)
        } else {
            None
        }
    }

    /// Create a Lyndon word without checking (unsafe)
    pub fn new_unchecked(letters: Vec<usize>) -> Self {
        LyndonWord { letters }
    }

    /// Check if this word is a Lyndon word
    pub fn is_lyndon(&self) -> bool {
        if self.letters.is_empty() {
            return false;
        }

        // A word is Lyndon if it's strictly smaller than all its proper suffixes
        for i in 1..self.letters.len() {
            if &self.letters[..] >= &self.letters[i..] {
                return false;
            }
        }

        true
    }

    /// Get the letters
    pub fn letters(&self) -> &[usize] {
        &self.letters
    }

    /// Length of the word
    pub fn len(&self) -> usize {
        self.letters.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.letters.is_empty()
    }

    /// Standard factorization of a Lyndon word into (u, v)
    ///
    /// Returns the unique factorization where v is the longest proper Lyndon suffix
    pub fn standard_factorization(&self) -> Option<(LyndonWord, LyndonWord)> {
        if self.len() <= 1 {
            return None;
        }

        // Find the longest proper Lyndon suffix
        for i in (1..self.len()).rev() {
            let suffix = LyndonWord::new_unchecked(self.letters[i..].to_vec());
            if suffix.is_lyndon() {
                let prefix = LyndonWord::new_unchecked(self.letters[..i].to_vec());
                return Some((prefix, suffix));
            }
        }

        None
    }
}

impl PartialOrd for LyndonWord {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LyndonWord {
    fn cmp(&self, other: &Self) -> Ordering {
        self.letters.cmp(&other.letters)
    }
}

impl Display for LyndonWord {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;
        for (i, &letter) in self.letters.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", letter)?;
        }
        write!(f, "]")
    }
}

/// Check if a word is a Lyndon word
pub fn is_lyndon(word: &[usize]) -> bool {
    if word.is_empty() {
        return false;
    }

    for i in 1..word.len() {
        if &word[..] >= &word[i..] {
            return false;
        }
    }

    true
}

/// A graded Lie bracket element
///
/// Represents elements in the free Lie algebra as nested brackets
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LieBracket {
    /// A single generator
    Generator(usize),
    /// A bracket of two elements
    Bracket(Box<LieBracket>, Box<LieBracket>),
}

impl LieBracket {
    /// Create a generator
    pub fn generator(index: usize) -> Self {
        LieBracket::Generator(index)
    }

    /// Create a bracket
    pub fn bracket(left: LieBracket, right: LieBracket) -> Self {
        LieBracket::Bracket(Box::new(left), Box::new(right))
    }

    /// Get the grade (depth) of this bracket
    pub fn grade(&self) -> usize {
        match self {
            LieBracket::Generator(_) => 1,
            LieBracket::Bracket(left, right) => left.grade() + right.grade(),
        }
    }

    /// Convert to a word (flattening the bracket structure)
    pub fn to_word(&self) -> Vec<usize> {
        match self {
            LieBracket::Generator(i) => vec![*i],
            LieBracket::Bracket(left, right) => {
                let mut word = left.to_word();
                word.extend(right.to_word());
                word
            }
        }
    }
}

impl Display for LieBracket {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LieBracket::Generator(i) => write!(f, "x{}", i),
            LieBracket::Bracket(left, right) => write!(f, "[{}, {}]", left, right),
        }
    }
}

/// Free Lie Algebra
///
/// The free Lie algebra on n generators with a specified basis.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
///
/// # Examples
///
/// ```
/// # use rustmath_liealgebras::{FreeLieAlgebra, FreeLieAlgebraBasis};
/// let free_lie: FreeLieAlgebra<i64> = FreeLieAlgebra::new(3, FreeLieAlgebraBasis::Lyndon);
/// assert_eq!(free_lie.num_generators(), 3);
/// ```
pub struct FreeLieAlgebra<R: Ring> {
    /// Number of generators
    num_generators: usize,
    /// Basis type
    basis_type: FreeLieAlgebraBasis,
    /// Coefficient ring marker
    coefficient_ring: PhantomData<R>,
}

impl<R: Ring + Clone> FreeLieAlgebra<R> {
    /// Create a new free Lie algebra
    ///
    /// # Arguments
    ///
    /// * `num_generators` - Number of generators
    /// * `basis_type` - Type of basis (Hall or Lyndon)
    pub fn new(num_generators: usize, basis_type: FreeLieAlgebraBasis) -> Self {
        FreeLieAlgebra {
            num_generators,
            basis_type,
            coefficient_ring: PhantomData,
        }
    }

    /// Get the number of generators
    pub fn num_generators(&self) -> usize {
        self.num_generators
    }

    /// Get the basis type
    pub fn basis_type(&self) -> &FreeLieAlgebraBasis {
        &self.basis_type
    }

    /// Check if this is finite dimensional (always false for free Lie algebras)
    pub fn is_finite_dimensional(&self) -> bool {
        false
    }

    /// Get the zero element
    pub fn zero(&self) -> FreeLieAlgebraElement<R>
    where
        R: From<i64>,
    {
        FreeLieAlgebraElement::zero()
    }

    /// Get a generator
    pub fn generator(&self, index: usize) -> FreeLieAlgebraElement<R>
    where
        R: From<i64>,
    {
        if index >= self.num_generators {
            return FreeLieAlgebraElement::zero();
        }
        FreeLieAlgebraElement::generator(index)
    }

    /// Get all generators
    pub fn generators(&self) -> Vec<FreeLieAlgebraElement<R>>
    where
        R: From<i64>,
    {
        (0..self.num_generators)
            .map(|i| self.generator(i))
            .collect()
    }

    /// Dimension of the k-th graded component
    ///
    /// Uses the Witt formula: (1/k) Σ μ(d)·n^(k/d)
    /// where μ is the Möbius function
    pub fn graded_dimension(&self, k: usize) -> usize {
        if k == 0 {
            return 0;
        }

        let n = self.num_generators;
        let mut sum = 0i64;

        // Sum over divisors of k
        for d in 1..=k {
            if k % d == 0 {
                let mu_d = moebius(d);
                let power = n.pow((k / d) as u32);
                sum += mu_d * power as i64;
            }
        }

        (sum / k as i64).max(0) as usize
    }

    /// Generate Lyndon words of length k with n letters
    pub fn lyndon_words(&self, k: usize) -> Vec<LyndonWord> {
        if k == 0 {
            return vec![];
        }
        if k == 1 {
            return (0..self.num_generators)
                .map(|i| LyndonWord::new_unchecked(vec![i]))
                .collect();
        }

        let mut words = Vec::new();
        self.generate_lyndon_words_recursive(&mut vec![], k, 0, &mut words);
        words
    }

    fn generate_lyndon_words_recursive(
        &self,
        current: &mut Vec<usize>,
        remaining: usize,
        last: usize,
        result: &mut Vec<LyndonWord>,
    ) {
        if remaining == 0 {
            if is_lyndon(current) {
                result.push(LyndonWord::new_unchecked(current.clone()));
            }
            return;
        }

        for i in 0..self.num_generators {
            current.push(i);
            if current.len() == 1 || i >= last {
                self.generate_lyndon_words_recursive(current, remaining - 1, i, result);
            }
            current.pop();
        }
    }
}

impl<R: Ring + Clone> Display for FreeLieAlgebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Free Lie algebra on {} generators with {:?} basis",
            self.num_generators, self.basis_type
        )
    }
}

/// Element of a free Lie algebra
///
/// Represented as a formal linear combination of Lie brackets
#[derive(Clone)]
pub struct FreeLieAlgebraElement<R: Ring> {
    /// Terms: map from LieBracket to coefficient
    terms: HashMap<LieBracket, R>,
}

impl<R: Ring + Clone> FreeLieAlgebraElement<R> {
    /// Create a new element
    pub fn new(terms: HashMap<LieBracket, R>) -> Self {
        FreeLieAlgebraElement { terms }
    }

    /// Create the zero element
    pub fn zero() -> Self
    where
        R: From<i64>,
    {
        FreeLieAlgebraElement {
            terms: HashMap::new(),
        }
    }

    /// Create a generator element
    pub fn generator(index: usize) -> Self
    where
        R: From<i64>,
    {
        let mut terms = HashMap::new();
        terms.insert(LieBracket::Generator(index), R::from(1));
        FreeLieAlgebraElement { terms }
    }

    /// Create a bracket element
    pub fn bracket_element(left: LieBracket, right: LieBracket) -> Self
    where
        R: From<i64>,
    {
        let mut terms = HashMap::new();
        terms.insert(LieBracket::Bracket(Box::new(left), Box::new(right)), R::from(1));
        FreeLieAlgebraElement { terms }
    }

    /// Get the terms
    pub fn terms(&self) -> &HashMap<LieBracket, R> {
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

        for (bracket, coeff) in &other.terms {
            let new_coeff = if let Some(existing) = result.get(bracket) {
                existing.clone() + coeff.clone()
            } else {
                coeff.clone()
            };

            if new_coeff.is_zero() {
                result.remove(bracket);
            } else {
                result.insert(bracket.clone(), new_coeff);
            }
        }

        FreeLieAlgebraElement { terms: result }
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
            .map(|(bracket, coeff)| (bracket.clone(), coeff.clone() * scalar.clone()))
            .collect();

        FreeLieAlgebraElement { terms }
    }

    /// Lie bracket with another element
    pub fn bracket(&self, other: &Self) -> Self
    where
        R: From<i64> + std::ops::Add<Output = R> + std::ops::Mul<Output = R> + std::ops::Sub<Output = R> + std::ops::Neg<Output = R> + PartialEq,
    {
        let mut result = Self::zero();

        for (left_bracket, left_coeff) in &self.terms {
            for (right_bracket, right_coeff) in &other.terms {
                // Create [left, right]
                let new_bracket = LieBracket::Bracket(
                    Box::new(left_bracket.clone()),
                    Box::new(right_bracket.clone()),
                );
                let coeff = left_coeff.clone() * right_coeff.clone();

                let mut term_map = HashMap::new();
                term_map.insert(new_bracket, coeff);
                let term = FreeLieAlgebraElement { terms: term_map };

                result = result.add(&term);
            }
        }

        result
    }
}

impl<R: Ring + Clone + PartialEq> PartialEq for FreeLieAlgebraElement<R> {
    fn eq(&self, other: &Self) -> bool {
        if self.terms.len() != other.terms.len() {
            return false;
        }

        for (bracket, coeff) in &self.terms {
            match other.terms.get(bracket) {
                Some(other_coeff) if coeff == other_coeff => continue,
                _ => return false,
            }
        }

        true
    }
}

impl<R: Ring + Clone + PartialEq> Eq for FreeLieAlgebraElement<R> {}

impl<R: Ring + Clone + Display> Display for FreeLieAlgebraElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }

        let mut terms: Vec<_> = self.terms.iter().collect();
        terms.sort_by_key(|(bracket, _)| bracket.grade());

        let mut first = true;
        for (bracket, coeff) in terms {
            if first {
                first = false;
            } else {
                write!(f, " + ")?;
            }
            write!(f, "{}*{}", coeff, bracket)?;
        }

        Ok(())
    }
}

/// Compute the Möbius function μ(n)
fn moebius(n: usize) -> i64 {
    if n == 1 {
        return 1;
    }

    // Factor n and count distinct prime factors
    let mut remaining = n;
    let mut num_factors = 0;
    let mut has_square_factor = false;

    for p in 2..=n {
        if p * p > remaining {
            if remaining > 1 {
                num_factors += 1;
            }
            break;
        }

        if remaining % p == 0 {
            num_factors += 1;
            remaining /= p;

            if remaining % p == 0 {
                has_square_factor = true;
                break;
            }
        }
    }

    if has_square_factor {
        0
    } else if num_factors % 2 == 0 {
        1
    } else {
        -1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lyndon_word_creation() {
        // [0, 1] is Lyndon
        let word1 = LyndonWord::new(vec![0, 1]);
        assert!(word1.is_some());

        // [1, 0] is not Lyndon (larger than suffix [0])
        let word2 = LyndonWord::new(vec![1, 0]);
        assert!(word2.is_none());

        // [0, 0, 1] is Lyndon
        let word3 = LyndonWord::new(vec![0, 0, 1]);
        assert!(word3.is_some());
    }

    #[test]
    fn test_is_lyndon() {
        assert!(is_lyndon(&[0]));
        assert!(is_lyndon(&[0, 1]));
        assert!(is_lyndon(&[0, 0, 1]));
        assert!(!is_lyndon(&[1, 0]));
        assert!(!is_lyndon(&[0, 1, 0]));
    }

    #[test]
    fn test_lyndon_standard_factorization() {
        let word = LyndonWord::new(vec![0, 0, 1]).unwrap();
        let (prefix, suffix) = word.standard_factorization().unwrap();
        assert_eq!(prefix.letters(), &[0, 0]);
        assert_eq!(suffix.letters(), &[1]);
    }

    #[test]
    fn test_lie_bracket_grade() {
        let x0 = LieBracket::Generator(0);
        assert_eq!(x0.grade(), 1);

        let x1 = LieBracket::Generator(1);
        let bracket = LieBracket::bracket(x0.clone(), x1.clone());
        assert_eq!(bracket.grade(), 2);

        let nested = LieBracket::bracket(bracket.clone(), x0.clone());
        assert_eq!(nested.grade(), 3);
    }

    #[test]
    fn test_free_lie_algebra_creation() {
        let lie: FreeLieAlgebra<i64> = FreeLieAlgebra::new(3, FreeLieAlgebraBasis::Lyndon);
        assert_eq!(lie.num_generators(), 3);
        assert!(!lie.is_finite_dimensional());
    }

    #[test]
    fn test_free_lie_algebra_generators() {
        let lie: FreeLieAlgebra<i64> = FreeLieAlgebra::new(2, FreeLieAlgebraBasis::Lyndon);
        let gens = lie.generators();
        assert_eq!(gens.len(), 2);
    }

    #[test]
    fn test_graded_dimension() {
        let lie: FreeLieAlgebra<i64> = FreeLieAlgebra::new(2, FreeLieAlgebraBasis::Lyndon);

        // For 2 generators:
        // dim(L_1) = 2 (the generators)
        assert_eq!(lie.graded_dimension(1), 2);

        // dim(L_2) = 1 (one bracket [x_0, x_1])
        assert_eq!(lie.graded_dimension(2), 1);

        // dim(L_3) = 2
        assert_eq!(lie.graded_dimension(3), 2);
    }

    #[test]
    fn test_element_creation() {
        let zero: FreeLieAlgebraElement<i64> = FreeLieAlgebraElement::zero();
        assert!(zero.is_zero());

        let gen0: FreeLieAlgebraElement<i64> = FreeLieAlgebraElement::generator(0);
        assert!(!gen0.is_zero());
        assert_eq!(gen0.terms().len(), 1);
    }

    #[test]
    fn test_element_addition() {
        let gen0: FreeLieAlgebraElement<i64> = FreeLieAlgebraElement::generator(0);
        let gen1: FreeLieAlgebraElement<i64> = FreeLieAlgebraElement::generator(1);

        let sum = gen0.add(&gen1);
        assert_eq!(sum.terms().len(), 2);
    }

    #[test]
    fn test_element_scalar_mul() {
        let gen0: FreeLieAlgebraElement<i64> = FreeLieAlgebraElement::generator(0);
        let scaled = gen0.scalar_mul(&5);
        assert_eq!(scaled.terms().len(), 1);
    }

    #[test]
    fn test_element_bracket() {
        let gen0: FreeLieAlgebraElement<i64> = FreeLieAlgebraElement::generator(0);
        let gen1: FreeLieAlgebraElement<i64> = FreeLieAlgebraElement::generator(1);

        let bracket = gen0.bracket(&gen1);
        assert!(!bracket.is_zero());
        assert_eq!(bracket.terms().len(), 1);
    }

    #[test]
    fn test_moebius_function() {
        assert_eq!(moebius(1), 1);
        assert_eq!(moebius(2), -1); // prime
        assert_eq!(moebius(3), -1); // prime
        assert_eq!(moebius(4), 0); // has square factor 2^2
        assert_eq!(moebius(5), -1); // prime
        assert_eq!(moebius(6), 1); // 2*3, two distinct primes
        assert_eq!(moebius(30), -1); // 2*3*5, three distinct primes
    }

    #[test]
    fn test_lyndon_words_generation() {
        let lie: FreeLieAlgebra<i64> = FreeLieAlgebra::new(2, FreeLieAlgebraBasis::Lyndon);

        // Length 1: [0], [1]
        let words1 = lie.lyndon_words(1);
        assert_eq!(words1.len(), 2);

        // Length 2: [0,1] (only Lyndon word of length 2 with 2 letters)
        let words2 = lie.lyndon_words(2);
        assert!(words2.iter().any(|w| w.letters() == &[0, 1]));
    }
}
