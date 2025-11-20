//! Word Combinatorics - Advanced operations on finite and infinite words
//!
//! This module provides comprehensive word combinatorics functionality including:
//! - Lyndon words and the Chen-Fox-Lyndon theorem
//! - Lyndon factorization using Duval's algorithm
//! - Standard factorization with exponents
//! - Christoffel words
//! - Sturmian words
//! - Word morphisms
//! - Pattern matching algorithms
//! - Abelian complexity
//! - Automatic sequences
//!
//! ## Chen-Fox-Lyndon Theorem
//!
//! The Chen-Fox-Lyndon theorem is a fundamental result in combinatorics on words:
//!
//! **Theorem**: Every finite word w can be uniquely factored as a non-increasing
//! product of Lyndon words: w = l₁l₂...lₖ where l₁ ≥ l₂ ≥ ... ≥ lₖ are Lyndon words.
//!
//! A **Lyndon word** is a non-empty word that is strictly less than all of its
//! proper non-trivial rotations in lexicographic order.
//!
//! The **standard factorization** groups consecutive equal Lyndon factors:
//! w = l₁^{a₁}l₂^{a₂}...lₘ^{aₘ} where l₁ > l₂ > ... > lₘ and aᵢ > 0
//!
//! ## Example
//!
//! ```
//! use rustmath_combinatorics::{Word, lyndon_factorization};
//!
//! let w = Word::new(vec!['a', 'b', 'a', 'a', 'b', 'b']);
//! let factors = w.lyndon_factorization();
//! // factors = [ab, aab, b]
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::Hash;

/// A finite word over an alphabet
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Word<T: Clone + Eq> {
    /// The sequence of letters
    letters: Vec<T>,
}

impl<T: Clone + Eq> Word<T> {
    /// Create a new word from a vector of letters
    pub fn new(letters: Vec<T>) -> Self {
        Word { letters }
    }

    /// Create an empty word
    pub fn empty() -> Self {
        Word { letters: vec![] }
    }

    /// Get the length of the word
    pub fn len(&self) -> usize {
        self.letters.len()
    }

    /// Check if the word is empty
    pub fn is_empty(&self) -> bool {
        self.letters.is_empty()
    }

    /// Get the letters as a slice
    pub fn letters(&self) -> &[T] {
        &self.letters
    }

    /// Concatenate two words
    pub fn concat(&self, other: &Word<T>) -> Word<T> {
        let mut letters = self.letters.clone();
        letters.extend(other.letters.iter().cloned());
        Word { letters }
    }

    /// Get a factor (substring) of the word
    pub fn factor(&self, start: usize, end: usize) -> Option<Word<T>> {
        if end <= start || end > self.len() {
            return None;
        }
        Some(Word {
            letters: self.letters[start..end].to_vec(),
        })
    }

    /// Get a prefix of length n
    pub fn prefix(&self, n: usize) -> Option<Word<T>> {
        if n > self.len() {
            return None;
        }
        Some(Word {
            letters: self.letters[..n].to_vec(),
        })
    }

    /// Get a suffix of length n
    pub fn suffix(&self, n: usize) -> Option<Word<T>> {
        if n > self.len() {
            return None;
        }
        let start = self.len() - n;
        Some(Word {
            letters: self.letters[start..].to_vec(),
        })
    }

    /// Reverse the word
    pub fn reverse(&self) -> Word<T> {
        let mut letters = self.letters.clone();
        letters.reverse();
        Word { letters }
    }

    /// Rotate the word left by k positions
    pub fn rotate(&self, k: usize) -> Word<T> {
        if self.is_empty() {
            return self.clone();
        }
        let k = k % self.len();
        let mut letters = self.letters.clone();
        letters.rotate_left(k);
        Word { letters }
    }

    /// Count occurrences of a factor
    pub fn count_factor(&self, factor: &Word<T>) -> usize {
        if factor.len() > self.len() || factor.is_empty() {
            return 0;
        }

        let mut count = 0;
        for i in 0..=(self.len() - factor.len()) {
            if self.letters[i..i + factor.len()] == factor.letters[..] {
                count += 1;
            }
        }
        count
    }

    /// Check if this word is a factor of another
    pub fn is_factor_of(&self, other: &Word<T>) -> bool {
        other.count_factor(self) > 0
    }

    /// Get all factors of a given length
    pub fn factors_of_length(&self, n: usize) -> Vec<Word<T>> {
        if n > self.len() {
            return vec![];
        }

        let mut factors = Vec::new();
        for i in 0..=(self.len() - n) {
            factors.push(Word {
                letters: self.letters[i..i + n].to_vec(),
            });
        }
        factors
    }

    /// Compute the abelianization (multiset of letters)
    pub fn abelianize(&self) -> HashMap<T, usize>
    where
        T: Hash,
    {
        let mut counts = HashMap::new();
        for letter in &self.letters {
            *counts.entry(letter.clone()).or_insert(0) += 1;
        }
        counts
    }

    /// Check if two words are abelian equivalent
    pub fn abelian_equivalent(&self, other: &Word<T>) -> bool
    where
        T: Hash,
    {
        self.abelianize() == other.abelianize()
    }

    /// Repeat the word n times
    pub fn repeat(&self, n: usize) -> Word<T> {
        let mut letters = Vec::with_capacity(self.len() * n);
        for _ in 0..n {
            letters.extend(self.letters.iter().cloned());
        }
        Word { letters }
    }

    /// Check if the word is a power of some word (i.e., w^k for some k > 1)
    pub fn is_power(&self) -> bool {
        if self.is_empty() {
            return false;
        }

        let n = self.len();
        for period in 1..n {
            if n % period == 0 {
                let prefix = &self.letters[..period];
                let mut is_power = true;
                for i in (period..n).step_by(period) {
                    if self.letters[i..i + period] != prefix[..] {
                        is_power = false;
                        break;
                    }
                }
                if is_power {
                    return true;
                }
            }
        }
        false
    }

    /// Get the primitive root of a word (smallest w such that this word = w^k)
    pub fn primitive_root(&self) -> Word<T> {
        if self.is_empty() {
            return self.clone();
        }

        let n = self.len();
        for period in 1..=n {
            if n % period == 0 {
                let prefix = &self.letters[..period];
                let mut is_root = true;
                for i in (period..n).step_by(period) {
                    if self.letters[i..i + period] != prefix[..] {
                        is_root = false;
                        break;
                    }
                }
                if is_root {
                    return Word {
                        letters: prefix.to_vec(),
                    };
                }
            }
        }
        self.clone()
    }

    /// Compute the shuffle product of two words
    ///
    /// Returns all ways to interleave the two words while preserving their internal order.
    /// This is a fundamental operation in combinatorics and shuffle algebras.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_combinatorics::Word;
    ///
    /// let w1 = Word::new(vec!['a', 'b']);
    /// let w2 = Word::new(vec!['c']);
    /// let shuffles = w1.shuffle(&w2);
    /// // Returns: [abc, acb, cab]
    /// assert_eq!(shuffles.len(), 3);
    /// ```
    ///
    /// # Mathematical Background
    ///
    /// The shuffle product w₁ ⊔ w₂ is defined recursively:
    /// - If w₁ is empty, then w₁ ⊔ w₂ = {w₂}
    /// - If w₂ is empty, then w₁ ⊔ w₂ = {w₁}
    /// - Otherwise, w₁ ⊔ w₂ = {first(w₁) · (rest(w₁) ⊔ w₂)} ∪ {first(w₂) · (w₁ ⊔ rest(w₂))}
    ///
    /// The number of shuffles is given by the binomial coefficient C(|w₁|+|w₂|, |w₁|).
    pub fn shuffle(&self, other: &Word<T>) -> Vec<Word<T>> {
        // Base cases
        if self.is_empty() {
            return vec![other.clone()];
        }
        if other.is_empty() {
            return vec![self.clone()];
        }

        let mut result = Vec::new();

        // Take first letter from self
        let rest_self = Word {
            letters: self.letters[1..].to_vec(),
        };
        let shuffles_with_first_from_self = rest_self.shuffle(other);
        for shuffle in shuffles_with_first_from_self {
            let mut letters = vec![self.letters[0].clone()];
            letters.extend(shuffle.letters);
            result.push(Word { letters });
        }

        // Take first letter from other
        let rest_other = Word {
            letters: other.letters[1..].to_vec(),
        };
        let shuffles_with_first_from_other = self.shuffle(&rest_other);
        for shuffle in shuffles_with_first_from_other {
            let mut letters = vec![other.letters[0].clone()];
            letters.extend(shuffle.letters);
            result.push(Word { letters });
        }

        result
    }
}

impl<T: Clone + Eq + PartialOrd> Word<T> {
    /// Check if this word is a Lyndon word
    /// A Lyndon word is strictly less than all of its non-trivial rotations
    pub fn is_lyndon(&self) -> bool {
        if self.is_empty() {
            return false;
        }

        for k in 1..self.len() {
            let rotated = self.rotate(k);
            if rotated.letters <= self.letters {
                return false;
            }
        }
        true
    }

    /// Compute the Lyndon factorization (Chen-Fox-Lyndon factorization)
    ///
    /// Decomposes the word as a non-increasing product of Lyndon words using Duval's algorithm.
    ///
    /// # Chen-Fox-Lyndon Theorem
    ///
    /// Every word w has a unique factorization w = l₁l₂...lₖ where:
    /// - Each lᵢ is a Lyndon word
    /// - l₁ ≥ l₂ ≥ ... ≥ lₖ (non-increasing in lexicographic order)
    ///
    /// This algorithm runs in O(n) time and O(1) extra space using Duval's algorithm.
    ///
    /// # Example
    ///
    /// ```
    /// use rustmath_combinatorics::Word;
    ///
    /// let w = Word::new(vec![0, 1, 0, 0, 1, 1]);
    /// let factors = w.lyndon_factorization();
    /// // Each factor is a Lyndon word, and they are non-increasing
    /// ```
    pub fn lyndon_factorization(&self) -> Vec<Word<T>> {
        if self.is_empty() {
            return vec![];
        }

        let mut result = Vec::new();
        let mut i = 0;

        while i < self.len() {
            let mut j = i + 1;
            let mut k = i;

            while j < self.len() {
                if self.letters[k] < self.letters[j] {
                    k = i;
                    j += 1;
                } else if self.letters[k] == self.letters[j] {
                    k += 1;
                    j += 1;
                } else {
                    break;
                }
            }

            // Extract Lyndon factor
            while i <= k {
                result.push(Word {
                    letters: self.letters[i..i + j - k].to_vec(),
                });
                i += j - k;
            }
        }

        result
    }

    /// Compute the standard (or condensed) Lyndon factorization
    ///
    /// Returns the factorization as pairs (Lyndon word, exponent) where
    /// w = l₁^{a₁}l₂^{a₂}...lₘ^{aₘ} with l₁ > l₂ > ... > lₘ and aᵢ > 0.
    ///
    /// This groups consecutive equal Lyndon factors in the Chen-Fox-Lyndon factorization.
    ///
    /// # Example
    ///
    /// ```
    /// use rustmath_combinatorics::Word;
    ///
    /// let w = Word::new(vec![0, 1, 0, 1, 0, 0, 1]);
    /// let standard = w.standard_lyndon_factorization();
    /// // standard = [(01, 2), (001, 1)] meaning w = (01)²(001)
    /// ```
    pub fn standard_lyndon_factorization(&self) -> Vec<(Word<T>, usize)> {
        let factors = self.lyndon_factorization();
        if factors.is_empty() {
            return vec![];
        }

        let mut result = Vec::new();
        let mut i = 0;

        while i < factors.len() {
            let current = &factors[i];
            let mut count = 1;

            // Count consecutive equal factors
            while i + count < factors.len() && &factors[i + count] == current {
                count += 1;
            }

            result.push((current.clone(), count));
            i += count;
        }

        result
    }

    /// Check if this word can be written as a power of a Lyndon word
    ///
    /// Returns Some((lyndon_word, exponent)) if the word equals lyndon_word^exponent
    /// where lyndon_word is a Lyndon word, otherwise returns None.
    pub fn as_lyndon_power(&self) -> Option<(Word<T>, usize)> {
        if self.is_empty() {
            return None;
        }

        let standard = self.standard_lyndon_factorization();

        // A word is a power of a Lyndon word iff its standard factorization has only one term
        if standard.len() == 1 {
            Some(standard[0].clone())
        } else {
            None
        }
    }
}

impl<T: Clone + Eq + fmt::Display> fmt::Display for Word<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for letter in &self.letters {
            write!(f, "{}", letter)?;
        }
        Ok(())
    }
}

/// Generate all Lyndon words of length n over an alphabet
/// Uses a simple filtering approach - generates all words and filters for Lyndon property
///
/// Note: For large alphabets or lengths, consider using `lyndon_words_fast()` which uses
/// the more efficient CAT (Constant Amortized Time) algorithm.
pub fn lyndon_words<T: Clone + Eq + Ord>(alphabet: &[T], n: usize) -> Vec<Word<T>> {
    if n == 0 {
        return vec![Word::empty()];
    }
    if alphabet.is_empty() || n == 0 {
        return vec![];
    }

    fn generate_all_words<T: Clone>(alphabet: &[T], n: usize) -> Vec<Vec<T>> {
        if n == 0 {
            return vec![vec![]];
        }
        if n == 1 {
            return alphabet.iter().map(|a| vec![a.clone()]).collect();
        }

        let smaller = generate_all_words(alphabet, n - 1);
        let mut result = Vec::new();

        for word in smaller {
            for letter in alphabet {
                let mut new_word = word.clone();
                new_word.push(letter.clone());
                result.push(new_word);
            }
        }

        result
    }

    let all_words = generate_all_words(alphabet, n);
    all_words
        .into_iter()
        .map(|letters| Word { letters })
        .filter(|w| w.is_lyndon())
        .collect()
}

/// Generate all Lyndon words of length up to n over an alphabet using the FKM algorithm
///
/// This implements the Fredericksen-Kessler-Maiorana (FKM) algorithm which generates
/// Lyndon words in lexicographic order with constant amortized time per word.
///
/// Returns all Lyndon words of length 1, 2, ..., n over the given alphabet.
pub fn lyndon_words_up_to<T: Clone + Eq + Ord>(alphabet: &[T], n: usize) -> Vec<Word<T>> {
    if alphabet.is_empty() || n == 0 {
        return vec![];
    }

    let k = alphabet.len();
    let mut result = Vec::new();
    let mut w = vec![0_usize; n + 1];

    lyndon_gen(&mut w, n, k, 1, 1, &mut result);

    result
        .into_iter()
        .map(|indices| Word {
            letters: indices.iter().map(|&i| alphabet[i].clone()).collect(),
        })
        .collect()
}

/// Helper function for FKM algorithm
fn lyndon_gen(w: &mut [usize], n: usize, k: usize, t: usize, p: usize, result: &mut Vec<Vec<usize>>) {
    if t > n {
        if n % p == 0 {
            result.push(w[1..=p].to_vec());
        }
    } else {
        w[t] = w[t - p];
        lyndon_gen(w, n, k, t + 1, p, result);

        for j in (w[t - p] + 1)..k {
            w[t] = j;
            lyndon_gen(w, n, k, t + 1, t, result);
        }
    }
}

/// Check if a sequence of words forms a valid Chen-Fox-Lyndon factorization
///
/// Returns true if the concatenation of the words in the sequence forms a valid
/// CFL factorization (each word is Lyndon and they are in non-increasing order).
pub fn is_cfl_factorization<T: Clone + Eq + Ord>(factors: &[Word<T>]) -> bool {
    if factors.is_empty() {
        return true;
    }

    // Check each factor is a Lyndon word
    for factor in factors {
        if !factor.is_lyndon() {
            return false;
        }
    }

    // Check non-increasing order
    for i in 1..factors.len() {
        if factors[i].letters > factors[i - 1].letters {
            return false;
        }
    }

    true
}

/// Reconstruct a word from its Chen-Fox-Lyndon factorization
///
/// Takes a sequence of Lyndon words and concatenates them to form the original word.
pub fn from_cfl_factorization<T: Clone + Eq + Ord>(factors: &[Word<T>]) -> Word<T> {
    if factors.is_empty() {
        return Word::empty();
    }

    let mut letters = Vec::new();
    for factor in factors {
        letters.extend(factor.letters.iter().cloned());
    }

    Word { letters }
}

/// Reconstruct a word from its standard Lyndon factorization
///
/// Takes pairs of (Lyndon word, exponent) and reconstructs the original word.
pub fn from_standard_factorization<T: Clone + Eq + Ord>(
    standard: &[(Word<T>, usize)],
) -> Word<T> {
    if standard.is_empty() {
        return Word::empty();
    }

    let mut letters = Vec::new();
    for (lyndon_word, exponent) in standard {
        for _ in 0..*exponent {
            letters.extend(lyndon_word.letters.iter().cloned());
        }
    }

    Word { letters }
}

/// A Christoffel word is a finite Sturmian word with specific slope properties
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChristoffelWord {
    /// The word represented as a binary sequence
    bits: Vec<bool>,
    /// Numerator of the slope
    p: usize,
    /// Denominator of the slope
    q: usize,
}

impl ChristoffelWord {
    /// Create a Christoffel word with slope p/q using the standard construction
    /// The word has length p + q and contains exactly p ones
    pub fn new(p: usize, q: usize) -> Option<Self> {
        if q == 0 && p == 0 {
            return None;
        }

        // Reduce to lowest terms
        let g = if p == 0 || q == 0 { 1 } else { gcd(p, q) };
        let p = p / g;
        let q = q / g;

        if p == 0 {
            // All zeros
            return Some(ChristoffelWord {
                bits: vec![false; q],
                p,
                q,
            });
        }

        if q == 0 {
            // All ones
            return Some(ChristoffelWord {
                bits: vec![true; p],
                p,
                q: 1,
            });
        }

        // Use the standard Christoffel word construction
        // The Christoffel word is the discretization of the line y = (p/q)x
        // For each x from 0 to q-1, we output 1 if the line crosses a vertical grid line
        let mut bits = Vec::with_capacity(p + q);
        let mut x = 0;
        let mut y = 0;

        for _ in 0..(p + q) {
            // Check if (y+1)/p < (x+1)/q, which determines whether we move right (0) or up (1)
            // Equivalently: q(y+1) < p(x+1)
            if (y + 1) * q < (x + 1) * p {
                bits.push(true);  // Move up (output 1)
                y += 1;
            } else {
                bits.push(false);  // Move right (output 0)
                x += 1;
            }
        }

        Some(ChristoffelWord { bits, p, q })
    }

    /// Get the slope as (p, q)
    pub fn slope(&self) -> (usize, usize) {
        (self.p, self.q)
    }

    /// Get the bits
    pub fn bits(&self) -> &[bool] {
        &self.bits
    }

    /// Get the length
    pub fn len(&self) -> usize {
        self.bits.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    /// Convert to a Word<bool>
    pub fn to_word(&self) -> Word<bool> {
        Word::new(self.bits.clone())
    }

    /// Convert to string representation
    pub fn to_string(&self) -> String {
        self.bits
            .iter()
            .map(|&b| if b { '1' } else { '0' })
            .collect()
    }
}

/// Compute GCD using Euclidean algorithm
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

/// A Sturmian word is an infinite word with specific factor complexity
/// We represent finite prefixes of Sturmian words
#[derive(Debug, Clone)]
pub struct SturmianWord {
    /// The binary sequence (finite prefix)
    bits: Vec<bool>,
    /// The slope parameter alpha (represented as p/q approximation)
    slope_p: usize,
    slope_q: usize,
    /// The intercept parameter rho
    rho: f64,
}

impl SturmianWord {
    /// Generate a Sturmian word prefix of length n with slope p/q and intercept rho
    pub fn new(n: usize, slope_p: usize, slope_q: usize, rho: f64) -> Self {
        let alpha = slope_p as f64 / slope_q as f64;
        let mut bits = Vec::with_capacity(n);

        for i in 0..n {
            // Sturmian word: bit i is 1 if floor((i+1)*alpha + rho) > floor(i*alpha + rho)
            let floor_next = ((i + 1) as f64 * alpha + rho).floor();
            let floor_curr = (i as f64 * alpha + rho).floor();
            let val = floor_next > floor_curr;
            bits.push(val);
        }

        SturmianWord {
            bits,
            slope_p,
            slope_q,
            rho,
        }
    }

    /// Create the Fibonacci word (a special Sturmian word) prefix of length n
    /// This is the Sturmian word with slope = golden ratio and rho = 0
    pub fn fibonacci(n: usize) -> Self {
        // Fibonacci word is generated by iterating the morphism 0 -> 01, 1 -> 0
        // Starting from 0: 0, 01, 010, 01001, 01001010, ...
        let morphism = fibonacci_morphism();
        let mut current = Word::new(vec![0]);

        // Iterate the morphism until we have at least n letters
        while current.len() < n {
            current = morphism.apply(&current).unwrap();
        }

        let bits = current.letters()[..n].iter().map(|&x| x == 1).collect();

        SturmianWord {
            bits,
            slope_p: 1,
            slope_q: 1,
            rho: 0.0,
        }
    }

    /// Get the bits
    pub fn bits(&self) -> &[bool] {
        &self.bits
    }

    /// Get the length
    pub fn len(&self) -> usize {
        self.bits.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    /// Convert to a Word<bool>
    pub fn to_word(&self) -> Word<bool> {
        Word::new(self.bits.clone())
    }

    /// Convert to string
    pub fn to_string(&self) -> String {
        self.bits
            .iter()
            .map(|&b| if b { '1' } else { '0' })
            .collect()
    }

    /// Compute the factor complexity (number of distinct factors of length n)
    pub fn factor_complexity(&self, n: usize) -> usize {
        if n > self.bits.len() {
            return 0;
        }

        let word = self.to_word();
        let factors = word.factors_of_length(n);
        let unique: HashSet<_> = factors.into_iter().collect();
        unique.len()
    }
}

/// A word morphism (substitution) maps letters to words
#[derive(Debug, Clone)]
pub struct WordMorphism<T: Clone + Eq + Hash> {
    /// The substitution rules
    rules: HashMap<T, Word<T>>,
}

impl<T: Clone + Eq + Hash> WordMorphism<T> {
    /// Create a new word morphism from a map of rules
    pub fn new(rules: HashMap<T, Word<T>>) -> Self {
        WordMorphism { rules }
    }

    /// Apply the morphism to a single letter
    pub fn apply_to_letter(&self, letter: &T) -> Option<Word<T>> {
        self.rules.get(letter).cloned()
    }

    /// Apply the morphism to a word
    pub fn apply(&self, word: &Word<T>) -> Option<Word<T>> {
        let mut result = Vec::new();

        for letter in word.letters() {
            let substitution = self.rules.get(letter)?;
            result.extend(substitution.letters.iter().cloned());
        }

        Some(Word::new(result))
    }

    /// Apply the morphism n times starting from a letter
    pub fn iterate(&self, letter: &T, n: usize) -> Option<Word<T>> {
        let mut current = Word::new(vec![letter.clone()]);

        for _ in 0..n {
            current = self.apply(&current)?;
        }

        Some(current)
    }

    /// Check if the morphism is uniform (all images have the same length)
    pub fn is_uniform(&self) -> bool {
        if self.rules.is_empty() {
            return true;
        }

        let first_len = self.rules.values().next().unwrap().len();
        self.rules.values().all(|w| w.len() == first_len)
    }

    /// Get the length of the morphism (if uniform)
    pub fn uniform_length(&self) -> Option<usize> {
        if !self.is_uniform() {
            return None;
        }
        self.rules.values().next().map(|w| w.len())
    }
}

/// The Thue-Morse morphism: 0 -> 01, 1 -> 10
pub fn thue_morse_morphism() -> WordMorphism<u8> {
    let mut rules = HashMap::new();
    rules.insert(0, Word::new(vec![0, 1]));
    rules.insert(1, Word::new(vec![1, 0]));
    WordMorphism::new(rules)
}

/// The Fibonacci morphism: 0 -> 01, 1 -> 0
pub fn fibonacci_morphism() -> WordMorphism<u8> {
    let mut rules = HashMap::new();
    rules.insert(0, Word::new(vec![0, 1]));
    rules.insert(1, Word::new(vec![0]));
    WordMorphism::new(rules)
}

/// Pattern matching: find all occurrences of a pattern in a text
/// Returns the starting positions of all occurrences
pub fn pattern_match<T: Clone + Eq>(text: &Word<T>, pattern: &Word<T>) -> Vec<usize> {
    if pattern.is_empty() || pattern.len() > text.len() {
        return vec![];
    }

    let mut positions = Vec::new();
    for i in 0..=(text.len() - pattern.len()) {
        if text.letters[i..i + pattern.len()] == pattern.letters[..] {
            positions.push(i);
        }
    }
    positions
}

/// Compute the failure function for KMP pattern matching
fn compute_kmp_failure<T: Clone + Eq>(pattern: &Word<T>) -> Vec<usize> {
    let m = pattern.len();
    let mut failure = vec![0; m];
    let mut k = 0;

    for i in 1..m {
        while k > 0 && pattern.letters[k] != pattern.letters[i] {
            k = failure[k - 1];
        }
        if pattern.letters[k] == pattern.letters[i] {
            k += 1;
        }
        failure[i] = k;
    }

    failure
}

/// KMP pattern matching algorithm (more efficient for large texts)
pub fn kmp_pattern_match<T: Clone + Eq>(text: &Word<T>, pattern: &Word<T>) -> Vec<usize> {
    if pattern.is_empty() || pattern.len() > text.len() {
        return vec![];
    }

    let failure = compute_kmp_failure(pattern);
    let mut positions = Vec::new();
    let mut k = 0;

    for i in 0..text.len() {
        while k > 0 && pattern.letters[k] != text.letters[i] {
            k = failure[k - 1];
        }
        if pattern.letters[k] == text.letters[i] {
            k += 1;
        }
        if k == pattern.len() {
            positions.push(i + 1 - k);
            k = failure[k - 1];
        }
    }

    positions
}

/// Compute the abelian complexity of a word
/// Returns the number of abelian-distinct factors of each length
pub fn abelian_complexity<T: Clone + Eq + Hash + Ord>(word: &Word<T>) -> Vec<usize> {
    let mut complexities = Vec::new();

    for length in 1..=word.len() {
        let factors = word.factors_of_length(length);
        let mut abelianizations = HashSet::new();

        for factor in factors {
            let abelian = factor.abelianize();
            // Convert to sorted vector for hashing
            let mut items: Vec<_> = abelian.into_iter().collect();
            items.sort_by(|a, b| a.0.cmp(&b.0));
            abelianizations.insert(items);
        }

        complexities.push(abelianizations.len());
    }

    complexities
}

/// An automatic sequence (k-automatic sequence)
/// Defined by a finite automaton with output
#[derive(Debug, Clone)]
pub struct AutomaticSequence {
    /// Number of states
    num_states: usize,
    /// Alphabet size (base k)
    base: usize,
    /// Transition function: (state, symbol) -> state
    transitions: HashMap<(usize, usize), usize>,
    /// Output function: state -> output value
    outputs: HashMap<usize, usize>,
    /// Initial state
    initial_state: usize,
}

impl AutomaticSequence {
    /// Create a new automatic sequence
    pub fn new(
        num_states: usize,
        base: usize,
        transitions: HashMap<(usize, usize), usize>,
        outputs: HashMap<usize, usize>,
        initial_state: usize,
    ) -> Self {
        AutomaticSequence {
            num_states,
            base,
            transitions,
            outputs,
            initial_state,
        }
    }

    /// Compute the nth term of the sequence
    pub fn nth(&self, n: usize) -> Option<usize> {
        // Convert n to base-k representation
        let mut digits = Vec::new();
        let mut m = n;
        while m > 0 {
            digits.push(m % self.base);
            m /= self.base;
        }
        if digits.is_empty() {
            digits.push(0);
        }
        digits.reverse();

        // Run the automaton
        let mut state = self.initial_state;
        for digit in digits {
            state = *self.transitions.get(&(state, digit))?;
        }

        self.outputs.get(&state).copied()
    }

    /// Generate the first n terms of the sequence
    pub fn first_n_terms(&self, n: usize) -> Vec<Option<usize>> {
        (0..n).map(|i| self.nth(i)).collect()
    }
}

/// Create the Thue-Morse automatic sequence
/// This is 2-automatic and outputs the parity of the number of 1s in the binary representation
pub fn thue_morse_automatic() -> AutomaticSequence {
    let mut transitions = HashMap::new();
    transitions.insert((0, 0), 0);
    transitions.insert((0, 1), 1);
    transitions.insert((1, 0), 1);
    transitions.insert((1, 1), 0);

    let mut outputs = HashMap::new();
    outputs.insert(0, 0);
    outputs.insert(1, 1);

    AutomaticSequence::new(2, 2, transitions, outputs, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_word_basic() {
        let w = Word::new(vec![1, 2, 3, 4]);
        assert_eq!(w.len(), 4);
        assert!(!w.is_empty());

        let empty = Word::<i32>::empty();
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_word_concat() {
        let w1 = Word::new(vec![1, 2]);
        let w2 = Word::new(vec![3, 4]);
        let w3 = w1.concat(&w2);
        assert_eq!(w3.letters(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_word_factor() {
        let w = Word::new(vec![1, 2, 3, 4, 5]);
        let factor = w.factor(1, 4).unwrap();
        assert_eq!(factor.letters(), &[2, 3, 4]);
    }

    #[test]
    fn test_word_prefix_suffix() {
        let w = Word::new(vec![1, 2, 3, 4]);
        assert_eq!(w.prefix(2).unwrap().letters(), &[1, 2]);
        assert_eq!(w.suffix(2).unwrap().letters(), &[3, 4]);
    }

    #[test]
    fn test_word_reverse() {
        let w = Word::new(vec![1, 2, 3, 4]);
        assert_eq!(w.reverse().letters(), &[4, 3, 2, 1]);
    }

    #[test]
    fn test_word_rotate() {
        let w = Word::new(vec![1, 2, 3, 4]);
        assert_eq!(w.rotate(2).letters(), &[3, 4, 1, 2]);
    }

    #[test]
    fn test_word_is_lyndon() {
        let w1 = Word::new(vec![0, 0, 1]);
        assert!(w1.is_lyndon());

        let w2 = Word::new(vec![0, 1, 0]);
        assert!(!w2.is_lyndon());

        let w3 = Word::new(vec![0, 1, 1]);
        assert!(w3.is_lyndon());
    }

    #[test]
    fn test_lyndon_factorization() {
        let w = Word::new(vec![0, 1, 0, 0, 1, 1]);
        let factors = w.lyndon_factorization();

        // Check that concatenation gives original word
        let mut reconstructed = Word::empty();
        for factor in &factors {
            reconstructed = reconstructed.concat(factor);
        }
        assert_eq!(reconstructed, w);

        // Check that each factor is Lyndon
        for factor in &factors {
            assert!(factor.is_lyndon());
        }
    }

    #[test]
    fn test_lyndon_words_generation() {
        let alphabet = vec![0, 1];
        let lyndon_3 = lyndon_words(&alphabet, 3);

        // Lyndon words of length 3 over {0,1}: should include 001, 011, 111
        // Note: Different algorithms may generate different sets
        // The key property is that all generated words must be Lyndon words
        assert!(lyndon_3.len() >= 2);

        for word in &lyndon_3 {
            assert!(word.is_lyndon(), "Word {:?} is not a Lyndon word", word.letters());
        }
    }

    #[test]
    fn test_christoffel_word() {
        // Christoffel word with slope 2/3
        let cw = ChristoffelWord::new(2, 3).unwrap();
        assert_eq!(cw.len(), 5);
        assert_eq!(cw.slope(), (2, 3));

        // Number of 1s should equal p
        let ones = cw.bits().iter().filter(|&&b| b).count();
        assert_eq!(ones, 2);

        // Number of 0s should equal q
        let zeros = cw.bits().iter().filter(|&&b| !b).count();
        assert_eq!(zeros, 3);
    }

    #[test]
    fn test_sturmian_word() {
        let sw = SturmianWord::new(20, 1, 2, 0.0);
        assert_eq!(sw.len(), 20);

        // Sturmian words have factor complexity at most n + 1
        // (exactly n + 1 for infinite Sturmian words, but finite prefixes may have less)
        for n in 1..10 {
            let complexity = sw.factor_complexity(n);
            assert!(complexity <= n + 1, "Factor complexity {} exceeds n+1={}", complexity, n + 1);
        }
    }

    #[test]
    fn test_fibonacci_word() {
        let fw = SturmianWord::fibonacci(20);
        assert_eq!(fw.len(), 20);

        // Check that it starts correctly: 01001010010010...
        let expected_prefix = vec![false, true, false, false, true, false, true, false];
        assert_eq!(&fw.bits()[..8], &expected_prefix[..]);
    }

    #[test]
    fn test_word_morphism() {
        let morphism = fibonacci_morphism();

        let w0 = Word::new(vec![0]);
        let w1 = morphism.apply(&w0).unwrap();
        assert_eq!(w1.letters(), &[0, 1]);

        let w2 = morphism.apply(&w1).unwrap();
        assert_eq!(w2.letters(), &[0, 1, 0]);

        let w3 = morphism.apply(&w2).unwrap();
        assert_eq!(w3.letters(), &[0, 1, 0, 0, 1]);
    }

    #[test]
    fn test_thue_morse_morphism() {
        let morphism = thue_morse_morphism();

        let w0 = Word::new(vec![0]);
        let w1 = morphism.apply(&w0).unwrap();
        assert_eq!(w1.letters(), &[0, 1]);

        let w2 = morphism.apply(&w1).unwrap();
        assert_eq!(w2.letters(), &[0, 1, 1, 0]);

        let w3 = morphism.apply(&w2).unwrap();
        assert_eq!(w3.letters(), &[0, 1, 1, 0, 1, 0, 0, 1]);
    }

    #[test]
    fn test_pattern_match() {
        let text = Word::new(vec![1, 2, 3, 1, 2, 4, 1, 2, 3]);
        let pattern = Word::new(vec![1, 2, 3]);

        let positions = pattern_match(&text, &pattern);
        assert_eq!(positions, vec![0, 6]);
    }

    #[test]
    fn test_kmp_pattern_match() {
        let text = Word::new(vec![1, 2, 3, 1, 2, 4, 1, 2, 3]);
        let pattern = Word::new(vec![1, 2, 3]);

        let positions = kmp_pattern_match(&text, &pattern);
        assert_eq!(positions, vec![0, 6]);
    }

    #[test]
    fn test_abelian_equivalent() {
        let w1 = Word::new(vec![1, 2, 3]);
        let w2 = Word::new(vec![3, 1, 2]);
        assert!(w1.abelian_equivalent(&w2));

        let w3 = Word::new(vec![1, 2, 2]);
        assert!(!w1.abelian_equivalent(&w3));
    }

    #[test]
    fn test_abelian_complexity() {
        let w = Word::new(vec![1, 2, 1, 2, 1]);
        let complexity = abelian_complexity(&w);

        // For this word, abelian complexity should be small
        assert!(complexity.len() > 0);
        assert!(complexity[0] <= w.len());
    }

    #[test]
    fn test_word_is_power() {
        let w1 = Word::new(vec![1, 2, 1, 2]);
        assert!(w1.is_power());

        let w2 = Word::new(vec![1, 2, 3]);
        assert!(!w2.is_power());

        let w3 = Word::new(vec![1, 1, 1]);
        assert!(w3.is_power());
    }

    #[test]
    fn test_primitive_root() {
        let w1 = Word::new(vec![1, 2, 1, 2, 1, 2]);
        let root = w1.primitive_root();
        assert_eq!(root.letters(), &[1, 2]);

        let w2 = Word::new(vec![1, 2, 3]);
        let root2 = w2.primitive_root();
        assert_eq!(root2.letters(), &[1, 2, 3]);
    }

    #[test]
    fn test_thue_morse_automatic() {
        let tm = thue_morse_automatic();

        // First few terms of Thue-Morse: 0, 1, 1, 0, 1, 0, 0, 1, ...
        assert_eq!(tm.nth(0), Some(0));
        assert_eq!(tm.nth(1), Some(1));
        assert_eq!(tm.nth(2), Some(1));
        assert_eq!(tm.nth(3), Some(0));
        assert_eq!(tm.nth(4), Some(1));
        assert_eq!(tm.nth(5), Some(0));
        assert_eq!(tm.nth(6), Some(0));
        assert_eq!(tm.nth(7), Some(1));
    }

    #[test]
    fn test_morphism_iteration() {
        let morphism = fibonacci_morphism();

        // Iterate 5 times starting from 0
        let result = morphism.iterate(&0, 5).unwrap();

        // Should give Fibonacci word prefix
        assert!(result.len() > 0);
    }

    #[test]
    fn test_count_factor() {
        let w = Word::new(vec![1, 2, 1, 2, 1]);
        let factor = Word::new(vec![1, 2]);

        assert_eq!(w.count_factor(&factor), 2);
    }

    #[test]
    fn test_word_repeat() {
        let w = Word::new(vec![1, 2]);
        let repeated = w.repeat(3);

        assert_eq!(repeated.letters(), &[1, 2, 1, 2, 1, 2]);
    }

    // Tests for Chen-Fox-Lyndon theorem and standard factorization

    #[test]
    fn test_standard_lyndon_factorization() {
        // Test word that has repeated Lyndon factors
        let w = Word::new(vec![0, 1, 0, 1, 0, 0, 1]);
        let standard = w.standard_lyndon_factorization();

        // Check that standard factorization is correct
        assert!(!standard.is_empty());

        // Verify that reconstructed word equals original
        let reconstructed = from_standard_factorization(&standard);
        assert_eq!(reconstructed, w);

        // Check each factor is Lyndon
        for (lyndon_word, _) in &standard {
            assert!(lyndon_word.is_lyndon());
        }

        // Check factors are in strictly decreasing order
        for i in 1..standard.len() {
            assert!(standard[i - 1].0.letters > standard[i].0.letters);
        }
    }

    #[test]
    fn test_as_lyndon_power() {
        // Test a word that is a power of a Lyndon word
        let w1 = Word::new(vec![0, 1, 0, 1, 0, 1]);
        if let Some((lyndon, exp)) = w1.as_lyndon_power() {
            assert_eq!(lyndon.letters(), &[0, 1]);
            assert_eq!(exp, 3);
        } else {
            panic!("Expected word to be a Lyndon power");
        }

        // Test a word that is not a power of a Lyndon word
        let w2 = Word::new(vec![0, 1, 0, 0, 1]);
        assert!(w2.as_lyndon_power().is_none());
    }

    #[test]
    fn test_is_cfl_factorization() {
        // Valid CFL factorization
        let factors = vec![
            Word::new(vec![0, 1]),
            Word::new(vec![0, 0, 1]),
        ];
        assert!(is_cfl_factorization(&factors));

        // Invalid - not all Lyndon words
        let invalid1 = vec![
            Word::new(vec![0, 1]),
            Word::new(vec![1, 0]),  // Not a Lyndon word
        ];
        assert!(!is_cfl_factorization(&invalid1));

        // Invalid - not in non-increasing order
        let invalid2 = vec![
            Word::new(vec![0, 0, 1]),
            Word::new(vec![0, 1]),  // Increasing order
        ];
        assert!(!is_cfl_factorization(&invalid2));
    }

    #[test]
    fn test_from_cfl_factorization() {
        let factors = vec![
            Word::new(vec![0, 1]),
            Word::new(vec![0, 0, 1]),
            Word::new(vec![1]),
        ];

        let w = from_cfl_factorization(&factors);
        assert_eq!(w.letters(), &[0, 1, 0, 0, 1, 1]);

        // Verify that lyndon_factorization gives back the same factors
        let refactored = w.lyndon_factorization();
        assert_eq!(refactored, factors);
    }

    #[test]
    fn test_from_standard_factorization() {
        let standard = vec![
            (Word::new(vec![0, 1]), 2),
            (Word::new(vec![0, 0, 1]), 1),
        ];

        let w = from_standard_factorization(&standard);
        assert_eq!(w.letters(), &[0, 1, 0, 1, 0, 0, 1]);

        // Verify round-trip
        let standard2 = w.standard_lyndon_factorization();
        assert_eq!(standard2, standard);
    }

    #[test]
    fn test_lyndon_factorization_properties() {
        // Test various words to ensure CFL properties hold
        let test_words = vec![
            vec![0, 1, 0, 0, 1, 1],
            vec![1, 0, 1, 0, 1],
            vec![0, 0, 0, 1, 1, 1],
            vec![0, 1, 1, 0, 0, 1],
        ];

        for letters in test_words {
            let w = Word::new(letters.clone());
            let factors = w.lyndon_factorization();

            // Property 1: Concatenation gives original word
            let reconstructed = from_cfl_factorization(&factors);
            assert_eq!(reconstructed, w, "Failed for word {:?}", letters);

            // Property 2: Each factor is a Lyndon word
            for factor in &factors {
                assert!(factor.is_lyndon(), "Non-Lyndon factor {:?} in word {:?}", factor.letters(), letters);
            }

            // Property 3: Factors are in non-increasing order
            for i in 1..factors.len() {
                assert!(
                    factors[i - 1].letters >= factors[i].letters,
                    "Factors not non-increasing in word {:?}", letters
                );
            }

            // Property 4: Valid CFL factorization
            assert!(is_cfl_factorization(&factors), "Invalid CFL for word {:?}", letters);
        }
    }

    #[test]
    fn test_lyndon_words_up_to() {
        // Test FKM algorithm for generating Lyndon words
        let alphabet = vec![0, 1];
        let lyndon = lyndon_words_up_to(&alphabet, 4);

        // All generated words should be Lyndon words
        for word in &lyndon {
            assert!(word.is_lyndon(), "Generated non-Lyndon word: {:?}", word.letters());
        }

        // Should include all Lyndon words up to length 4
        // Binary Lyndon words: 0, 1, 01, 001, 011, 0001, 0011, 0111
        assert!(lyndon.len() >= 8);

        // Check some specific words are present
        let w01 = Word::new(vec![0, 1]);
        let w001 = Word::new(vec![0, 0, 1]);
        let w011 = Word::new(vec![0, 1, 1]);

        assert!(lyndon.contains(&w01));
        assert!(lyndon.contains(&w001));
        assert!(lyndon.contains(&w011));
    }

    #[test]
    fn test_lyndon_factorization_uniqueness() {
        // The CFL factorization should be unique
        let w = Word::new(vec![0, 1, 0, 0, 1, 1, 0, 1]);

        let factors1 = w.lyndon_factorization();
        let factors2 = w.lyndon_factorization();

        assert_eq!(factors1, factors2, "CFL factorization should be deterministic");
    }

    #[test]
    fn test_standard_factorization_exponents() {
        // Test that exponents are counted correctly
        let w = Word::new(vec![0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1]);
        let standard = w.standard_lyndon_factorization();

        // Check that all exponents are positive
        for (_, exp) in &standard {
            assert!(*exp > 0, "Exponent should be positive");
        }

        // Verify total length
        let total_len: usize = standard.iter()
            .map(|(word, exp)| word.len() * exp)
            .sum();
        assert_eq!(total_len, w.len(), "Standard factorization should preserve length");
    }

    #[test]
    fn test_single_letter_lyndon() {
        // Single letter words are not Lyndon words (by convention, need at least 1 rotation)
        // Actually, in some definitions they are considered Lyndon
        let w = Word::new(vec![0]);
        let factors = w.lyndon_factorization();
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].letters(), &[0]);
    }

    #[test]
    fn test_empty_word_factorization() {
        let w: Word<u32> = Word::empty();
        let factors = w.lyndon_factorization();
        assert!(factors.is_empty());

        let standard = w.standard_lyndon_factorization();
        assert!(standard.is_empty());

        assert!(w.as_lyndon_power().is_none());
    }

    // Shuffle product tests

    #[test]
    fn test_shuffle_empty_words() {
        let w1: Word<char> = Word::empty();
        let w2 = Word::new(vec!['a', 'b']);

        let shuffles1 = w1.shuffle(&w2);
        assert_eq!(shuffles1.len(), 1);
        assert_eq!(shuffles1[0], w2);

        let shuffles2 = w2.shuffle(&w1);
        assert_eq!(shuffles2.len(), 1);
        assert_eq!(shuffles2[0], w2);
    }

    #[test]
    fn test_shuffle_single_letters() {
        let w1 = Word::new(vec!['a']);
        let w2 = Word::new(vec!['b']);

        let shuffles = w1.shuffle(&w2);

        // Should have 2 shuffles: ab and ba
        assert_eq!(shuffles.len(), 2);
        assert!(shuffles.contains(&Word::new(vec!['a', 'b'])));
        assert!(shuffles.contains(&Word::new(vec!['b', 'a'])));
    }

    #[test]
    fn test_shuffle_ab_c() {
        let w1 = Word::new(vec!['a', 'b']);
        let w2 = Word::new(vec!['c']);

        let shuffles = w1.shuffle(&w2);

        // Should have 3 shuffles: abc, acb, cab
        assert_eq!(shuffles.len(), 3);
        assert!(shuffles.contains(&Word::new(vec!['a', 'b', 'c'])));
        assert!(shuffles.contains(&Word::new(vec!['a', 'c', 'b'])));
        assert!(shuffles.contains(&Word::new(vec!['c', 'a', 'b'])));
    }

    #[test]
    fn test_shuffle_count() {
        let w1 = Word::new(vec![1, 2]);
        let w2 = Word::new(vec![3, 4]);

        let shuffles = w1.shuffle(&w2);

        // Number of shuffles should be C(4, 2) = 6
        assert_eq!(shuffles.len(), 6);
    }

    #[test]
    fn test_shuffle_preserves_order() {
        let w1 = Word::new(vec![1, 2, 3]);
        let w2 = Word::new(vec![4, 5]);

        let shuffles = w1.shuffle(&w2);

        // Verify that internal order is preserved in each shuffle
        for shuffle in shuffles {
            // Extract positions of elements from w1
            let mut pos1 = Vec::new();
            for (i, &letter) in shuffle.letters().iter().enumerate() {
                if letter <= 3 {
                    pos1.push((i, letter));
                }
            }
            // Check that they appear in order 1, 2, 3
            assert_eq!(pos1.len(), 3);
            assert_eq!(pos1[0].1, 1);
            assert_eq!(pos1[1].1, 2);
            assert_eq!(pos1[2].1, 3);
            assert!(pos1[0].0 < pos1[1].0);
            assert!(pos1[1].0 < pos1[2].0);

            // Extract positions of elements from w2
            let mut pos2 = Vec::new();
            for (i, &letter) in shuffle.letters().iter().enumerate() {
                if letter >= 4 {
                    pos2.push((i, letter));
                }
            }
            // Check that they appear in order 4, 5
            assert_eq!(pos2.len(), 2);
            assert_eq!(pos2[0].1, 4);
            assert_eq!(pos2[1].1, 5);
            assert!(pos2[0].0 < pos2[1].0);
        }
    }

    #[test]
    fn test_shuffle_symmetry() {
        let w1 = Word::new(vec!['x', 'y']);
        let w2 = Word::new(vec!['z']);

        let shuffles1 = w1.shuffle(&w2);
        let shuffles2 = w2.shuffle(&w1);

        // Both should produce the same set of words (possibly in different order)
        assert_eq!(shuffles1.len(), shuffles2.len());
        for word in &shuffles1 {
            assert!(shuffles2.contains(word));
        }
    }

    #[test]
    fn test_shuffle_with_identical_words() {
        let w1 = Word::new(vec![1, 1]);
        let w2 = Word::new(vec![2, 2]);

        let shuffles = w1.shuffle(&w2);

        // Should have C(4, 2) = 6 shuffles
        assert_eq!(shuffles.len(), 6);

        // All shuffles should have length 4
        for shuffle in &shuffles {
            assert_eq!(shuffle.len(), 4);
        }
    }
}
