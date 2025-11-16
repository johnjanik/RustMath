//! Shuffle Algebras
//!
//! This module implements shuffle algebras, which are commutative and associative
//! algebras with basis indexed by words over an alphabet.
//!
//! # Mathematical Background
//!
//! The shuffle algebra Sh(A) over an alphabet A has a basis indexed by words
//! (finite sequences) over A. The product is the shuffle product: for words w₁ and w₂,
//! their product is the sum over all ways to interleave w₁ and w₂ while preserving
//! the internal order of each word.
//!
//! For example, the shuffle product of "ab" and "c" is:
//! ab ⊔ c = abc + acb + cab
//!
//! The shuffle algebra forms a graded Hopf algebra with:
//! - Product: shuffle product
//! - Coproduct: deconcatenation
//! - Unit: empty word
//! - Counit: projection onto empty word
//! - Antipode: reversal with sign (-1)^length
//!
//! # Examples
//!
//! ```
//! use rustmath_algebras::shuffle_algebra::*;
//!
//! // Create a shuffle algebra over {x, y}
//! let shuffle = ShuffleAlgebra::new(vec!["x", "y"]);
//!
//! // Compute shuffle product
//! let result = shuffle.shuffle_product(&vec!["x"], &vec!["y"]);
//! // Result: ["x", "y"] and ["y", "x"]
//! ```

use rustmath_core::Ring;
use std::collections::HashMap;
use std::fmt;

/// A word is a sequence of letters from an alphabet
pub type Word = Vec<String>;

/// Compute all shuffles of two words
///
/// Returns all ways to interleave w1 and w2 while preserving their internal order.
///
/// # Examples
///
/// ```
/// use rustmath_algebras::shuffle_algebra::shuffle_product;
///
/// let w1 = vec!["a".to_string()];
/// let w2 = vec!["b".to_string()];
/// let shuffles = shuffle_product(&w1, &w2);
/// // Result: [["a", "b"], ["b", "a"]]
/// assert_eq!(shuffles.len(), 2);
/// ```
pub fn shuffle_product(w1: &[String], w2: &[String]) -> Vec<Word> {
    if w1.is_empty() {
        return vec![w2.to_vec()];
    }
    if w2.is_empty() {
        return vec![w1.to_vec()];
    }

    let mut result = Vec::new();

    // Take first letter from w1
    let mut rest1 = shuffle_product(&w1[1..], w2);
    for mut word in rest1 {
        word.insert(0, w1[0].clone());
        result.push(word);
    }

    // Take first letter from w2
    let mut rest2 = shuffle_product(w1, &w2[1..]);
    for mut word in rest2 {
        word.insert(0, w2[0].clone());
        result.push(word);
    }

    result
}

/// Element of a shuffle algebra
///
/// An element is represented as a linear combination of basis elements (words).
#[derive(Debug, Clone, PartialEq)]
pub struct ShuffleElement<R: Ring> {
    /// Coefficients for each word
    coefficients: HashMap<Word, R>,
}

impl<R: Ring> ShuffleElement<R> {
    /// Create a zero element
    pub fn zero() -> Self {
        Self {
            coefficients: HashMap::new(),
        }
    }

    /// Create an element from a single word
    pub fn from_word(word: Word, coeff: R) -> Self {
        let mut coefficients = HashMap::new();
        if !coeff.is_zero() {
            coefficients.insert(word, coeff);
        }
        Self { coefficients }
    }

    /// Create the identity element (empty word with coefficient 1)
    pub fn one() -> Self
    where
        R: From<i32>,
    {
        Self::from_word(vec![], R::from(1))
    }

    /// Check if the element is zero
    pub fn is_zero(&self) -> bool {
        self.coefficients.is_empty()
    }

    /// Get the coefficients
    pub fn coefficients(&self) -> &HashMap<Word, R> {
        &self.coefficients
    }
}

impl<R: Ring> fmt::Display for ShuffleElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut terms: Vec<_> = self.coefficients.iter().collect();
        terms.sort_by_key(|(k, _)| k.clone());

        for (i, (word, coeff)) in terms.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }
            if word.is_empty() {
                write!(f, "{}", coeff)?;
            } else {
                write!(f, "{}*{}", coeff, word.join(""))?;
            }
        }
        Ok(())
    }
}

/// Shuffle Algebra
///
/// The shuffle algebra Sh(A) over an alphabet A, with basis indexed by words.
#[derive(Debug, Clone)]
pub struct ShuffleAlgebra {
    /// The alphabet (set of generators)
    alphabet: Vec<String>,
}

impl ShuffleAlgebra {
    /// Create a new shuffle algebra over the given alphabet
    ///
    /// # Arguments
    ///
    /// * `alphabet` - The alphabet (list of generator names)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_algebras::shuffle_algebra::ShuffleAlgebra;
    ///
    /// let shuffle = ShuffleAlgebra::new(vec!["x".to_string(), "y".to_string()]);
    /// assert_eq!(shuffle.rank(), 2);
    /// ```
    pub fn new(alphabet: Vec<&str>) -> Self {
        let alphabet = alphabet.iter().map(|s| s.to_string()).collect();
        Self { alphabet }
    }

    /// Get the alphabet
    pub fn alphabet(&self) -> &[String] {
        &self.alphabet
    }

    /// Get the rank (number of generators)
    pub fn rank(&self) -> usize {
        self.alphabet.len()
    }

    /// Compute the shuffle product of two words
    ///
    /// Returns all shuffles of the two words.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_algebras::shuffle_algebra::ShuffleAlgebra;
    ///
    /// let shuffle = ShuffleAlgebra::new(vec!["x", "y"]);
    /// let w1 = vec!["x".to_string()];
    /// let w2 = vec!["y".to_string()];
    /// let result = shuffle.shuffle_product(&w1, &w2);
    /// assert_eq!(result.len(), 2); // xy and yx
    /// ```
    pub fn shuffle_product(&self, w1: &[String], w2: &[String]) -> Vec<Word> {
        shuffle_product(w1, w2)
    }

    /// Get the identity element (empty word)
    pub fn one<R: Ring + From<i32>>(&self) -> ShuffleElement<R> {
        ShuffleElement::one()
    }

    /// Compute the coproduct of a word
    ///
    /// The coproduct is given by deconcatenation:
    /// Δ(w) = Σ (w[0..i] ⊗ w[i..n]) for i in 0..=n
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_algebras::shuffle_algebra::ShuffleAlgebra;
    ///
    /// let shuffle = ShuffleAlgebra::new(vec!["x", "y"]);
    /// let word = vec!["x".to_string(), "y".to_string()];
    /// let coproduct = shuffle.coproduct(&word);
    /// // Result: [([], [x,y]), ([x], [y]), ([x,y], [])]
    /// assert_eq!(coproduct.len(), 3);
    /// ```
    pub fn coproduct(&self, word: &[String]) -> Vec<(Word, Word)> {
        let n = word.len();
        let mut result = Vec::new();

        for i in 0..=n {
            let left = word[0..i].to_vec();
            let right = word[i..n].to_vec();
            result.push((left, right));
        }

        result
    }

    /// Compute the antipode of a word
    ///
    /// The antipode is S(w) = (-1)^|w| * reverse(w)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_algebras::shuffle_algebra::ShuffleAlgebra;
    ///
    /// let shuffle = ShuffleAlgebra::new(vec!["x", "y"]);
    /// let word = vec!["x".to_string(), "y".to_string()];
    /// let (sign, reversed) = shuffle.antipode(&word);
    /// assert_eq!(sign, 1);  // (-1)^2 = 1
    /// assert_eq!(reversed, vec!["y".to_string(), "x".to_string()]);
    /// ```
    pub fn antipode(&self, word: &[String]) -> (i32, Word) {
        let sign = if word.len() % 2 == 0 { 1 } else { -1 };
        let mut reversed = word.to_vec();
        reversed.reverse();
        (sign, reversed)
    }

    /// Check if a word belongs to this algebra (all letters in alphabet)
    pub fn contains_word(&self, word: &[String]) -> bool {
        word.iter().all(|letter| self.alphabet.contains(letter))
    }
}

impl fmt::Display for ShuffleAlgebra {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Shuffle algebra on {{{}}}", self.alphabet.join(", "))
    }
}

/// Dual PBW Basis
///
/// The dual Poincaré-Birkhoff-Witt basis is an alternative basis for the shuffle algebra
/// defined using Lyndon words and their factorizations.
///
/// This provides a PBW-type basis that is often more convenient for computations
/// in representation theory and combinatorics.
#[derive(Debug, Clone)]
pub struct DualPBWBasis {
    /// The underlying shuffle algebra
    shuffle_algebra: ShuffleAlgebra,
}

impl DualPBWBasis {
    /// Create a new dual PBW basis
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_algebras::shuffle_algebra::*;
    ///
    /// let shuffle = ShuffleAlgebra::new(vec!["x", "y"]);
    /// let pbw = DualPBWBasis::new(shuffle);
    /// ```
    pub fn new(shuffle_algebra: ShuffleAlgebra) -> Self {
        Self { shuffle_algebra }
    }

    /// Get the underlying shuffle algebra
    pub fn shuffle_algebra(&self) -> &ShuffleAlgebra {
        &self.shuffle_algebra
    }

    /// Get the alphabet
    pub fn alphabet(&self) -> &[String] {
        self.shuffle_algebra.alphabet()
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        self.shuffle_algebra.rank()
    }

    /// Convert a word to its PBW basis element
    ///
    /// This is a placeholder that would implement the full Lyndon factorization
    /// and conversion algorithm.
    pub fn to_pbw_basis(&self, word: &[String]) -> String {
        format!("PBW({})", word.join(""))
    }

    /// Convert from PBW basis to shuffle basis
    ///
    /// This would implement the expansion of a PBW basis element in terms
    /// of the standard shuffle basis.
    pub fn to_shuffle_basis(&self, pbw_word: &str) -> String {
        format!("Shuffle({})", pbw_word)
    }
}

impl fmt::Display for DualPBWBasis {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Dual PBW basis of shuffle algebra on {{{}}}",
            self.shuffle_algebra.alphabet().join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shuffle_product_empty() {
        let w1 = vec![];
        let w2 = vec!["a".to_string()];
        let result = shuffle_product(&w1, &w2);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], vec!["a".to_string()]);
    }

    #[test]
    fn test_shuffle_product_single() {
        let w1 = vec!["a".to_string()];
        let w2 = vec!["b".to_string()];
        let result = shuffle_product(&w1, &w2);
        assert_eq!(result.len(), 2);
        assert!(result.contains(&vec!["a".to_string(), "b".to_string()]));
        assert!(result.contains(&vec!["b".to_string(), "a".to_string()]));
    }

    #[test]
    fn test_shuffle_product_longer() {
        let w1 = vec!["a".to_string(), "b".to_string()];
        let w2 = vec!["c".to_string()];
        let result = shuffle_product(&w1, &w2);
        // Should have 3 shuffles: abc, acb, cab
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_shuffle_algebra_creation() {
        let shuffle = ShuffleAlgebra::new(vec!["x", "y"]);
        assert_eq!(shuffle.rank(), 2);
        assert_eq!(shuffle.alphabet().len(), 2);
    }

    #[test]
    fn test_shuffle_algebra_contains_word() {
        let shuffle = ShuffleAlgebra::new(vec!["x", "y"]);
        assert!(shuffle.contains_word(&vec!["x".to_string(), "y".to_string()]));
        assert!(!shuffle.contains_word(&vec!["z".to_string()]));
    }

    #[test]
    fn test_shuffle_algebra_product() {
        let shuffle = ShuffleAlgebra::new(vec!["x", "y"]);
        let w1 = vec!["x".to_string()];
        let w2 = vec!["y".to_string()];
        let result = shuffle.shuffle_product(&w1, &w2);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_coproduct() {
        let shuffle = ShuffleAlgebra::new(vec!["x", "y"]);
        let word = vec!["x".to_string(), "y".to_string()];
        let coproduct = shuffle.coproduct(&word);

        assert_eq!(coproduct.len(), 3);
        assert!(coproduct.contains(&(vec![], vec!["x".to_string(), "y".to_string()])));
        assert!(coproduct.contains(&(vec!["x".to_string()], vec!["y".to_string()])));
        assert!(coproduct.contains(&(vec!["x".to_string(), "y".to_string()], vec![])));
    }

    #[test]
    fn test_antipode() {
        let shuffle = ShuffleAlgebra::new(vec!["x", "y"]);

        // Even length: sign = 1
        let word = vec!["x".to_string(), "y".to_string()];
        let (sign, reversed) = shuffle.antipode(&word);
        assert_eq!(sign, 1);
        assert_eq!(reversed, vec!["y".to_string(), "x".to_string()]);

        // Odd length: sign = -1
        let word2 = vec!["x".to_string()];
        let (sign2, reversed2) = shuffle.antipode(&word2);
        assert_eq!(sign2, -1);
        assert_eq!(reversed2, vec!["x".to_string()]);
    }

    #[test]
    fn test_shuffle_element_zero() {
        let elem: ShuffleElement<i32> = ShuffleElement::zero();
        assert!(elem.is_zero());
    }

    #[test]
    fn test_shuffle_element_one() {
        let elem: ShuffleElement<i32> = ShuffleElement::one();
        assert!(!elem.is_zero());
        assert_eq!(elem.coefficients().len(), 1);
        assert_eq!(elem.coefficients().get(&vec![]), Some(&1));
    }

    #[test]
    fn test_shuffle_element_from_word() {
        let word = vec!["x".to_string(), "y".to_string()];
        let elem = ShuffleElement::from_word(word.clone(), 3);
        assert!(!elem.is_zero());
        assert_eq!(elem.coefficients().len(), 1);
        assert_eq!(elem.coefficients().get(&word), Some(&3));
    }

    #[test]
    fn test_dual_pbw_basis() {
        let shuffle = ShuffleAlgebra::new(vec!["x", "y"]);
        let pbw = DualPBWBasis::new(shuffle);
        assert_eq!(pbw.rank(), 2);
    }

    #[test]
    fn test_dual_pbw_conversions() {
        let shuffle = ShuffleAlgebra::new(vec!["x", "y"]);
        let pbw = DualPBWBasis::new(shuffle);

        let word = vec!["x".to_string(), "y".to_string()];
        let pbw_elem = pbw.to_pbw_basis(&word);
        assert!(pbw_elem.contains("PBW"));

        let shuffle_elem = pbw.to_shuffle_basis("xy");
        assert!(shuffle_elem.contains("Shuffle"));
    }

    #[test]
    fn test_shuffle_algebra_display() {
        let shuffle = ShuffleAlgebra::new(vec!["x", "y"]);
        let display = format!("{}", shuffle);
        assert!(display.contains("Shuffle algebra"));
        assert!(display.contains("x"));
        assert!(display.contains("y"));
    }

    #[test]
    fn test_dual_pbw_display() {
        let shuffle = ShuffleAlgebra::new(vec!["a", "b"]);
        let pbw = DualPBWBasis::new(shuffle);
        let display = format!("{}", pbw);
        assert!(display.contains("Dual PBW"));
        assert!(display.contains("shuffle algebra"));
    }

    #[test]
    fn test_element_display() {
        let word = vec!["x".to_string()];
        let elem = ShuffleElement::from_word(word, 2);
        let display = format!("{}", elem);
        assert!(display.contains("2"));
        assert!(display.contains("x"));

        let zero: ShuffleElement<i32> = ShuffleElement::zero();
        assert_eq!(format!("{}", zero), "0");
    }

    #[test]
    fn test_empty_word_display() {
        let elem: ShuffleElement<i32> = ShuffleElement::from_word(vec![], 5);
        let display = format!("{}", elem);
        assert_eq!(display, "5");
    }

    #[test]
    fn test_multi_letter_word() {
        let word = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let elem = ShuffleElement::from_word(word.clone(), 1);
        let display = format!("{}", elem);
        assert!(display.contains("abc"));
    }
}
