//! Words over arbitrary alphabets and word combinatorics
//!
//! This module provides comprehensive word combinatorics including:
//! - Lyndon words and the Chen-Fox-Lyndon theorem
//! - Lyndon factorization using Duval's algorithm
//! - Standard factorization with exponents
//! - Christoffel words
//! - Sturmian words
//! - Word morphisms
//! - Pattern matching algorithms (KMP, Boyer-Moore)
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
//! let factors = lyndon_factorization(&w);
//! // factors = [ab, aab, b]
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::Hash;

/// A word over an arbitrary alphabet
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Word<T: Clone + Ord + Hash> {
    /// The letters of the word
    letters: Vec<T>,
}

impl<T: Clone + Ord + Hash> Word<T> {
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

    /// Concatenate with another word
    pub fn concat(&self, other: &Word<T>) -> Word<T> {
        let mut letters = self.letters.clone();
        letters.extend(other.letters.clone());
        Word { letters }
    }

    /// Get a substring/factor
    pub fn factor(&self, start: usize, end: usize) -> Option<Word<T>> {
        if end <= self.len() && start <= end {
            Some(Word {
                letters: self.letters[start..end].to_vec(),
            })
        } else {
            None
        }
    }

    /// Get a prefix of given length
    pub fn prefix(&self, len: usize) -> Option<Word<T>> {
        self.factor(0, len)
    }

    /// Get a suffix of given length
    pub fn suffix(&self, len: usize) -> Option<Word<T>> {
        if len <= self.len() {
            self.factor(self.len() - len, self.len())
        } else {
            None
        }
    }

    /// Reverse the word
    pub fn reverse(&self) -> Word<T> {
        let mut letters = self.letters.clone();
        letters.reverse();
        Word { letters }
    }

    /// Repeat the word n times
    pub fn repeat(&self, n: usize) -> Word<T> {
        let mut letters = Vec::with_capacity(self.len() * n);
        for _ in 0..n {
            letters.extend(self.letters.clone());
        }
        Word { letters }
    }

    /// Check if this word is a prefix of another
    pub fn is_prefix_of(&self, other: &Word<T>) -> bool {
        if self.len() > other.len() {
            return false;
        }
        self.letters == other.letters[..self.len()]
    }

    /// Check if this word is a suffix of another
    pub fn is_suffix_of(&self, other: &Word<T>) -> bool {
        if self.len() > other.len() {
            return false;
        }
        self.letters == other.letters[other.len() - self.len()..]
    }

    /// Check if this word is a factor (substring) of another
    pub fn is_factor_of(&self, other: &Word<T>) -> bool {
        if self.len() > other.len() {
            return false;
        }
        for i in 0..=(other.len() - self.len()) {
            if &other.letters[i..i + self.len()] == &self.letters[..] {
                return true;
            }
        }
        false
    }

    /// Find all occurrences of this word in another (returns starting positions)
    pub fn occurrences_in(&self, text: &Word<T>) -> Vec<usize> {
        let mut positions = Vec::new();
        if self.len() > text.len() {
            return positions;
        }
        for i in 0..=(text.len() - self.len()) {
            if &text.letters[i..i + self.len()] == &self.letters[..] {
                positions.push(i);
            }
        }
        positions
    }

    /// Count occurrences of this word in another
    pub fn count_in(&self, text: &Word<T>) -> usize {
        self.occurrences_in(text).len()
    }

    /// Get the period of the word (smallest p such that w[i] = w[i+p] for all valid i)
    pub fn period(&self) -> usize {
        let n = self.len();
        if n == 0 {
            return 0;
        }

        // Try each possible period
        for p in 1..=n {
            let mut valid = true;
            for i in 0..(n - p) {
                if self.letters[i] != self.letters[i + p] {
                    valid = false;
                    break;
                }
            }
            if valid {
                return p;
            }
        }
        n
    }

    /// Check if the word is periodic (period < length)
    pub fn is_periodic(&self) -> bool {
        self.period() < self.len()
    }

    /// Check if the word is a Lyndon word
    /// A Lyndon word is strictly less than all its non-trivial rotations
    pub fn is_lyndon(&self) -> bool {
        if self.is_empty() {
            return false;
        }

        let n = self.len();
        for k in 1..n {
            let rotation = self.rotate_left(k);
            if rotation.letters <= self.letters {
                return false;
            }
        }
        true
    }

    /// Rotate left by k positions
    pub fn rotate_left(&self, k: usize) -> Word<T> {
        if self.is_empty() {
            return self.clone();
        }
        let k = k % self.len();
        let mut letters = Vec::with_capacity(self.len());
        letters.extend(self.letters[k..].iter().cloned());
        letters.extend(self.letters[..k].iter().cloned());
        Word { letters }
    }

    /// Get the Parikh vector (letter frequencies)
    /// Returns a map from letter to count
    pub fn parikh_vector(&self) -> HashMap<T, usize> {
        let mut counts = HashMap::new();
        for letter in &self.letters {
            *counts.entry(letter.clone()).or_insert(0) += 1;
        }
        counts
    }

    /// Check if two words are abelian equivalent (same Parikh vector)
    pub fn abelian_equivalent(&self, other: &Word<T>) -> bool {
        self.parikh_vector() == other.parikh_vector()
    }
}

impl Word<char> {
    /// Create a word from a string
    pub fn from_str(s: &str) -> Self {
        Word {
            letters: s.chars().collect(),
        }
    }

    /// Convert word to string
    pub fn to_string(&self) -> String {
        self.letters.iter().collect()
    }
}

impl<T: Clone + Ord + Hash + fmt::Display> fmt::Display for Word<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for letter in &self.letters {
            write!(f, "{}", letter)?;
        }
        Ok(())
    }
}

/// Lyndon factorization using Duval's algorithm (Chen-Fox-Lyndon factorization)
///
/// Returns the unique factorization of a word into non-increasing Lyndon words.
///
/// # Chen-Fox-Lyndon Theorem
///
/// Every word w has a unique factorization w = l₁l₂...lₖ where:
/// - Each lᵢ is a Lyndon word
/// - l₁ ≥ l₂ ≥ ... ≥ lₖ (non-increasing in lexicographic order)
///
/// This algorithm runs in O(n) time and O(1) extra space using Duval's algorithm.
pub fn lyndon_factorization<T: Clone + Ord + Hash>(word: &Word<T>) -> Vec<Word<T>> {
    let n = word.len();
    if n == 0 {
        return vec![];
    }

    let mut factors = Vec::new();
    let mut i = 0;

    while i < n {
        let mut j = i + 1;
        let mut k = i;

        while j < n && word.letters[k] <= word.letters[j] {
            if word.letters[k] < word.letters[j] {
                k = i;
            } else {
                k += 1;
            }
            j += 1;
        }

        // Extract Lyndon factors from [i..j)
        while i <= k {
            let factor = Word {
                letters: word.letters[i..i + j - k].to_vec(),
            };
            factors.push(factor);
            i += j - k;
        }
    }

    factors
}

/// Compute the standard (or condensed) Lyndon factorization
///
/// Returns the factorization as pairs (Lyndon word, exponent) where
/// w = l₁^{a₁}l₂^{a₂}...lₘ^{aₘ} with l₁ > l₂ > ... > lₘ and aᵢ > 0.
///
/// This groups consecutive equal Lyndon factors in the Chen-Fox-Lyndon factorization.
pub fn standard_lyndon_factorization<T: Clone + Ord + Hash>(word: &Word<T>) -> Vec<(Word<T>, usize)> {
    let factors = lyndon_factorization(word);
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

/// Check if a sequence of words forms a valid Chen-Fox-Lyndon factorization
///
/// Returns true if the concatenation of the words in the sequence forms a valid
/// CFL factorization (each word is Lyndon and they are in non-increasing order).
pub fn is_cfl_factorization<T: Clone + Ord + Hash>(factors: &[Word<T>]) -> bool {
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
pub fn from_cfl_factorization<T: Clone + Ord + Hash>(factors: &[Word<T>]) -> Word<T> {
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
pub fn from_standard_factorization<T: Clone + Ord + Hash>(
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

/// Generate all Lyndon words of length n over alphabet {0, 1, ..., k-1}
/// Simple approach: generate all words and filter for Lyndon property
///
/// Note: For large alphabets or lengths, consider using `lyndon_words_up_to()` which uses
/// the more efficient FKM (Fredericksen-Kessler-Maiorana) algorithm.
pub fn lyndon_words(n: usize, k: usize) -> Vec<Word<usize>> {
    if n == 0 {
        return vec![];
    }
    if k == 0 {
        return vec![];
    }
    if k == 1 {
        // Only one Lyndon word over single-letter alphabet: the word "0...0"
        return vec![Word {
            letters: vec![0; n],
        }];
    }

    let mut result = Vec::new();
    generate_all_words(n, k, &mut vec![], &mut result);

    // Filter for Lyndon words
    result.into_iter().filter(|w| w.is_lyndon()).collect()
}

/// Generate all Lyndon words of length up to n using the FKM algorithm
///
/// This implements the Fredericksen-Kessler-Maiorana (FKM) algorithm which generates
/// Lyndon words in lexicographic order with constant amortized time per word.
///
/// Returns all Lyndon words of length 1, 2, ..., n over alphabet {0, 1, ..., k-1}.
pub fn lyndon_words_up_to(n: usize, k: usize) -> Vec<Word<usize>> {
    if k == 0 || n == 0 {
        return vec![];
    }

    let mut result = Vec::new();
    let mut w = vec![0_usize; n + 1];

    lyndon_gen(&mut w, n, k, 1, 1, &mut result);

    result
}

/// Helper function for FKM algorithm
fn lyndon_gen(w: &mut [usize], n: usize, k: usize, t: usize, p: usize, result: &mut Vec<Word<usize>>) {
    if t > n {
        if n % p == 0 {
            result.push(Word {
                letters: w[1..=p].to_vec(),
            });
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

/// Helper to generate all words of length n over alphabet {0..k-1}
fn generate_all_words(n: usize, k: usize, current: &mut Vec<usize>, result: &mut Vec<Word<usize>>) {
    if current.len() == n {
        result.push(Word {
            letters: current.clone(),
        });
        return;
    }

    for letter in 0..k {
        current.push(letter);
        generate_all_words(n, k, current, result);
        current.pop();
    }
}

/// Christoffel word for slope p/q where gcd(p,q) = 1
/// The Christoffel word is the word with slope p/q
pub fn christoffel_word(p: usize, q: usize) -> Word<usize> {
    if p == 0 && q == 0 {
        return Word::empty();
    }
    if p == 0 {
        return Word {
            letters: vec![0; q],
        };
    }
    if q == 0 {
        return Word {
            letters: vec![1; p],
        };
    }

    // Use iterative construction based on binary expansion
    let mut result = Vec::with_capacity(p + q);
    for i in 1..=(p + q) {
        // s_i = floor(i*p/(p+q)) - floor((i-1)*p/(p+q))
        let curr = (i * p) / (p + q);
        let prev = ((i - 1) * p) / (p + q);
        result.push(curr - prev);
    }

    Word { letters: result }
}

/// Generate a Sturmian word with slope α up to length n
/// For irrational α, this generates an infinite aperiodic word
/// We approximate α as p/q
pub fn sturmian_word(p: usize, q: usize, length: usize) -> Word<usize> {
    if q == 0 || length == 0 {
        return Word::empty();
    }

    let mut letters = Vec::with_capacity(length);

    for n in 0..length {
        // s_n = floor((n+1)*p/q) - floor(n*p/q)
        let val = ((n + 1) * p) / q - (n * p) / q;
        letters.push(val);
    }

    Word { letters }
}

/// Word morphism - a function that maps letters to words
#[derive(Debug, Clone)]
pub struct Morphism<T: Clone + Ord + Hash> {
    /// The mapping from letters to words
    mapping: HashMap<T, Word<T>>,
}

impl<T: Clone + Ord + Hash> Morphism<T> {
    /// Create a new morphism
    pub fn new(mapping: HashMap<T, Word<T>>) -> Self {
        Morphism { mapping }
    }

    /// Apply the morphism to a single letter
    pub fn apply_letter(&self, letter: &T) -> Option<Word<T>> {
        self.mapping.get(letter).cloned()
    }

    /// Apply the morphism to a word
    pub fn apply(&self, word: &Word<T>) -> Option<Word<T>> {
        let mut result = Word::empty();
        for letter in &word.letters {
            match self.apply_letter(letter) {
                Some(image) => result = result.concat(&image),
                None => return None,
            }
        }
        Some(result)
    }

    /// Iterate the morphism n times starting from a word
    pub fn iterate(&self, word: &Word<T>, n: usize) -> Option<Word<T>> {
        let mut current = word.clone();
        for _ in 0..n {
            current = self.apply(&current)?;
        }
        Some(current)
    }

    /// Check if the morphism is prolongable on a letter
    /// (i.e., φ(a) = aω for some word ω)
    pub fn is_prolongable(&self, letter: &T) -> bool {
        if let Some(image) = self.apply_letter(letter) {
            if !image.is_empty() {
                return &image.letters[0] == letter;
            }
        }
        false
    }

    /// Generate the fixed point of a prolongable morphism
    pub fn fixed_point(&self, letter: &T, length: usize) -> Option<Word<T>> {
        if !self.is_prolongable(letter) {
            return None;
        }

        let mut result = Word::new(vec![letter.clone()]);

        while result.len() < length {
            result = self.apply(&result)?;
            if result.len() >= length {
                result.letters.truncate(length);
                break;
            }
        }

        Some(result)
    }
}

/// KMP (Knuth-Morris-Pratt) pattern matching algorithm
/// Returns all occurrences of pattern in text
pub fn kmp_search<T: Clone + Ord + Hash>(text: &Word<T>, pattern: &Word<T>) -> Vec<usize> {
    if pattern.is_empty() || pattern.len() > text.len() {
        return vec![];
    }

    let m = pattern.len();
    let n = text.len();

    // Build failure function
    let failure = kmp_failure_function(pattern);

    let mut matches = Vec::new();
    let mut j = 0; // position in pattern

    for i in 0..n {
        while j > 0 && text.letters[i] != pattern.letters[j] {
            j = failure[j - 1];
        }

        if text.letters[i] == pattern.letters[j] {
            j += 1;
        }

        if j == m {
            matches.push(i + 1 - m);
            j = failure[j - 1];
        }
    }

    matches
}

/// Compute KMP failure function
fn kmp_failure_function<T: Clone + Ord + Hash>(pattern: &Word<T>) -> Vec<usize> {
    let m = pattern.len();
    let mut failure = vec![0; m];

    let mut j = 0;
    for i in 1..m {
        while j > 0 && pattern.letters[i] != pattern.letters[j] {
            j = failure[j - 1];
        }

        if pattern.letters[i] == pattern.letters[j] {
            j += 1;
        }

        failure[i] = j;
    }

    failure
}

/// Boyer-Moore pattern matching algorithm
/// Returns all occurrences of pattern in text
pub fn boyer_moore_search<T: Clone + Ord + Hash>(text: &Word<T>, pattern: &Word<T>) -> Vec<usize> {
    if pattern.is_empty() || pattern.len() > text.len() {
        return vec![];
    }

    let m = pattern.len();
    let n = text.len();

    // Build bad character table - maps character to rightmost occurrence
    let mut bad_char = HashMap::new();
    for i in 0..m {
        bad_char.insert(pattern.letters[i].clone(), i);
    }

    let mut matches = Vec::new();
    let mut s = 0; // shift of pattern relative to text

    while s <= n - m {
        let mut j = m;

        // Reduce j while characters match (scanning right to left)
        while j > 0 && pattern.letters[j - 1] == text.letters[s + j - 1] {
            j -= 1;
        }

        if j == 0 {
            // Pattern found
            matches.push(s);
            // Move to next possible match position
            if s + m < n {
                let next_char = &text.letters[s + m];
                s += if let Some(&pos) = bad_char.get(next_char) {
                    (m - 1 - pos).max(1)
                } else {
                    m
                };
            } else {
                s += 1;
            }
        } else {
            // Mismatch - use bad character rule
            let mismatched_char = &text.letters[s + j - 1];
            let shift = if let Some(&pos) = bad_char.get(mismatched_char) {
                if pos < j - 1 {
                    j - 1 - pos
                } else {
                    1
                }
            } else {
                j
            };
            s += shift;
        }
    }

    matches
}

/// Compute the abelian complexity function
/// Returns the number of abelian-distinct factors of length n
pub fn abelian_complexity<T: Clone + Ord + Hash>(word: &Word<T>, n: usize) -> usize {
    if n > word.len() {
        return 0;
    }

    let mut parikh_vectors = HashSet::new();

    for i in 0..=(word.len() - n) {
        if let Some(factor) = word.factor(i, i + n) {
            let parikh = factor.parikh_vector();
            // Convert to sorted vector for hashing
            let mut sorted: Vec<(T, usize)> = parikh.into_iter().collect();
            sorted.sort_by(|a, b| a.0.cmp(&b.0));
            parikh_vectors.insert(sorted);
        }
    }

    parikh_vectors.len()
}

/// Compute factor complexity (number of distinct factors of each length)
pub fn factor_complexity<T: Clone + Ord + Hash>(word: &Word<T>) -> Vec<usize> {
    let n = word.len();
    let mut complexity = Vec::with_capacity(n + 1);

    for len in 0..=n {
        let mut factors = HashSet::new();
        for i in 0..=(n - len) {
            if let Some(factor) = word.factor(i, i + len) {
                factors.insert(factor);
            }
        }
        complexity.push(factors.len());
    }

    complexity
}

/// Automatic sequence - a sequence generated by a deterministic finite automaton
#[derive(Debug, Clone)]
pub struct AutomaticSequence {
    /// The base k (alphabet size for input)
    k: usize,
    /// The states of the automaton
    states: Vec<String>,
    /// Transition function: state × symbol → state
    transitions: HashMap<(usize, usize), usize>,
    /// Output function: state → output
    outputs: HashMap<usize, usize>,
    /// Initial state
    initial: usize,
}

impl AutomaticSequence {
    /// Create a new automatic sequence
    pub fn new(
        k: usize,
        states: Vec<String>,
        transitions: HashMap<(usize, usize), usize>,
        outputs: HashMap<usize, usize>,
        initial: usize,
    ) -> Self {
        AutomaticSequence {
            k,
            states,
            transitions,
            outputs,
            initial,
        }
    }

    /// Get the nth term of the sequence
    pub fn nth(&self, n: usize) -> Option<usize> {
        // Convert n to base k
        let digits = self.to_base_k(n);

        // Process through automaton
        let mut state = self.initial;
        for digit in digits {
            state = *self.transitions.get(&(state, digit))?;
        }

        self.outputs.get(&state).copied()
    }

    /// Generate the first n terms
    pub fn first_n(&self, n: usize) -> Vec<Option<usize>> {
        (0..n).map(|i| self.nth(i)).collect()
    }

    /// Convert number to base k representation
    fn to_base_k(&self, mut n: usize) -> Vec<usize> {
        if n == 0 {
            return vec![0];
        }

        let mut digits = Vec::new();
        while n > 0 {
            digits.push(n % self.k);
            n /= self.k;
        }
        digits.reverse();
        digits
    }

    /// Create the Thue-Morse sequence automaton (base 2)
    pub fn thue_morse() -> Self {
        let mut transitions = HashMap::new();
        transitions.insert((0, 0), 0);
        transitions.insert((0, 1), 1);
        transitions.insert((1, 0), 1);
        transitions.insert((1, 1), 0);

        let mut outputs = HashMap::new();
        outputs.insert(0, 0);
        outputs.insert(1, 1);

        AutomaticSequence::new(
            2,
            vec!["even".to_string(), "odd".to_string()],
            transitions,
            outputs,
            0,
        )
    }

    /// Create the Rudin-Shapiro sequence automaton (base 2)
    pub fn rudin_shapiro() -> Self {
        let mut transitions = HashMap::new();
        // States: 0=even 11s, 1=odd 11s, 2=even 11s (last was 1), 3=odd 11s (last was 1)
        transitions.insert((0, 0), 0);
        transitions.insert((0, 1), 2);
        transitions.insert((1, 0), 1);
        transitions.insert((1, 1), 3);
        transitions.insert((2, 0), 0);
        transitions.insert((2, 1), 3);
        transitions.insert((3, 0), 1);
        transitions.insert((3, 1), 2);

        let mut outputs = HashMap::new();
        outputs.insert(0, 0);
        outputs.insert(1, 1);
        outputs.insert(2, 0);
        outputs.insert(3, 1);

        AutomaticSequence::new(
            2,
            vec![
                "even_11".to_string(),
                "odd_11".to_string(),
                "even_11_last1".to_string(),
                "odd_11_last1".to_string(),
            ],
            transitions,
            outputs,
            0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_word_basics() {
        let w = Word::new(vec![1, 2, 3, 4]);
        assert_eq!(w.len(), 4);
        assert!(!w.is_empty());

        let empty: Word<usize> = Word::empty();
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_word_from_string() {
        let w = Word::from_str("hello");
        assert_eq!(w.len(), 5);
        assert_eq!(w.to_string(), "hello");
    }

    #[test]
    fn test_concat() {
        let w1 = Word::new(vec![1, 2]);
        let w2 = Word::new(vec![3, 4]);
        let w3 = w1.concat(&w2);
        assert_eq!(w3.letters(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_factor() {
        let w = Word::new(vec![1, 2, 3, 4, 5]);
        let f = w.factor(1, 4).unwrap();
        assert_eq!(f.letters(), &[2, 3, 4]);

        let prefix = w.prefix(3).unwrap();
        assert_eq!(prefix.letters(), &[1, 2, 3]);

        let suffix = w.suffix(2).unwrap();
        assert_eq!(suffix.letters(), &[4, 5]);
    }

    #[test]
    fn test_reverse() {
        let w = Word::from_str("abc");
        assert_eq!(w.reverse().to_string(), "cba");
    }

    #[test]
    fn test_repeat() {
        let w = Word::from_str("ab");
        assert_eq!(w.repeat(3).to_string(), "ababab");
    }

    #[test]
    fn test_is_lyndon() {
        let w1 = Word::from_str("aab");
        assert!(w1.is_lyndon());

        let w2 = Word::from_str("aba");
        assert!(!w2.is_lyndon()); // "aab" < "aba"

        let w3 = Word::from_str("abcd");
        assert!(w3.is_lyndon());
    }

    #[test]
    fn test_lyndon_factorization() {
        let w = Word::from_str("abaabb");
        let factors = lyndon_factorization(&w);

        // Should be factored into Lyndon words
        for factor in &factors {
            assert!(factor.is_lyndon());
        }

        // Factors should be non-increasing
        for i in 1..factors.len() {
            assert!(factors[i - 1].letters >= factors[i].letters);
        }
    }

    #[test]
    fn test_lyndon_words_generation() {
        // Generate all Lyndon words of length 3 over binary alphabet
        let words = lyndon_words(3, 2);

        // Should be: 001, 011
        assert_eq!(words.len(), 2);

        for word in &words {
            assert!(word.is_lyndon());
            assert_eq!(word.len(), 3);
        }
    }

    #[test]
    fn test_christoffel_word() {
        // Christoffel word for slope 2/3
        let w = christoffel_word(2, 3);

        // Should have 2 ones and 3 zeros
        let count_ones = w.letters().iter().filter(|&&x| x == 1).count();
        let count_zeros = w.letters().iter().filter(|&&x| x == 0).count();
        assert_eq!(count_ones, 2);
        assert_eq!(count_zeros, 3);
    }

    #[test]
    fn test_sturmian_word() {
        // Generate Sturmian word with slope 1/2
        let w = sturmian_word(1, 2, 10);
        assert_eq!(w.len(), 10);

        // All letters should be 0 or 1
        for &letter in w.letters() {
            assert!(letter == 0 || letter == 1);
        }
    }

    #[test]
    fn test_morphism() {
        // Fibonacci morphism: 0 → 01, 1 → 0
        let mut mapping = HashMap::new();
        mapping.insert(0, Word::new(vec![0, 1]));
        mapping.insert(1, Word::new(vec![0]));

        let morph = Morphism::new(mapping);

        let w = Word::new(vec![0]);
        let w1 = morph.apply(&w).unwrap();
        assert_eq!(w1.letters(), &[0, 1]);

        let w2 = morph.apply(&w1).unwrap();
        assert_eq!(w2.letters(), &[0, 1, 0]);

        let w3 = morph.apply(&w2).unwrap();
        assert_eq!(w3.letters(), &[0, 1, 0, 0, 1]);
    }

    #[test]
    fn test_morphism_prolongable() {
        let mut mapping = HashMap::new();
        mapping.insert(0, Word::new(vec![0, 1]));
        mapping.insert(1, Word::new(vec![0]));

        let morph = Morphism::new(mapping);

        assert!(morph.is_prolongable(&0));
        assert!(!morph.is_prolongable(&1));
    }

    #[test]
    fn test_morphism_fixed_point() {
        // Fibonacci morphism
        let mut mapping = HashMap::new();
        mapping.insert(0, Word::new(vec![0, 1]));
        mapping.insert(1, Word::new(vec![0]));

        let morph = Morphism::new(mapping);
        let fixed = morph.fixed_point(&0, 20).unwrap();

        // Should start with 0, 1, 0, 0, 1, 0, 1, 0, ...
        assert_eq!(&fixed.letters()[..8], &[0, 1, 0, 0, 1, 0, 1, 0]);
    }

    #[test]
    fn test_kmp_search() {
        let text = Word::from_str("ababcababa");
        let pattern = Word::from_str("aba");

        let matches = kmp_search(&text, &pattern);
        assert_eq!(matches, vec![0, 5, 7]);
    }

    #[test]
    fn test_boyer_moore_search() {
        let text = Word::from_str("ababcababa");
        let pattern = Word::from_str("aba");

        let matches = boyer_moore_search(&text, &pattern);
        assert_eq!(matches, vec![0, 5, 7]);
    }

    #[test]
    fn test_parikh_vector() {
        let w = Word::from_str("banana");
        let parikh = w.parikh_vector();

        assert_eq!(parikh.get(&'b'), Some(&1));
        assert_eq!(parikh.get(&'a'), Some(&3));
        assert_eq!(parikh.get(&'n'), Some(&2));
    }

    #[test]
    fn test_abelian_equivalent() {
        let w1 = Word::from_str("abc");
        let w2 = Word::from_str("cab");
        let w3 = Word::from_str("xyz");

        assert!(w1.abelian_equivalent(&w2));
        assert!(!w1.abelian_equivalent(&w3));
    }

    #[test]
    fn test_abelian_complexity() {
        let w = Word::from_str("aabbaa");

        // Length 1: {a}, {b} → 2 distinct
        assert_eq!(abelian_complexity(&w, 1), 2);

        // Length 2: {aa}, {ab}, {ba}, {bb} → abelian classes: {aa}, {ab=ba}, {bb} → 3 distinct
        assert_eq!(abelian_complexity(&w, 2), 3);
    }

    #[test]
    fn test_factor_complexity() {
        let w = Word::from_str("abc");
        let complexity = factor_complexity(&w);

        // Length 0: 1 (empty word)
        // Length 1: 3 (a, b, c)
        // Length 2: 2 (ab, bc)
        // Length 3: 1 (abc)
        assert_eq!(complexity, vec![1, 3, 2, 1]);
    }

    #[test]
    fn test_automatic_sequence_thue_morse() {
        let tm = AutomaticSequence::thue_morse();
        let first_8 = tm.first_n(8);

        // Thue-Morse: 0, 1, 1, 0, 1, 0, 0, 1, ...
        assert_eq!(
            first_8,
            vec![
                Some(0),
                Some(1),
                Some(1),
                Some(0),
                Some(1),
                Some(0),
                Some(0),
                Some(1)
            ]
        );
    }

    #[test]
    fn test_period() {
        let w1 = Word::from_str("abcabcabc");
        assert_eq!(w1.period(), 3);

        let w2 = Word::from_str("aaaa");
        assert_eq!(w2.period(), 1);

        let w3 = Word::from_str("abcd");
        assert_eq!(w3.period(), 4); // aperiodic
    }

    #[test]
    fn test_pattern_matching_consistency() {
        let text = Word::from_str("the quick brown fox jumps over the lazy dog");
        let pattern = Word::from_str("the");

        let kmp_matches = kmp_search(&text, &pattern);
        let bm_matches = boyer_moore_search(&text, &pattern);
        let naive_matches = pattern.occurrences_in(&text);

        assert_eq!(kmp_matches, bm_matches);
        assert_eq!(kmp_matches, naive_matches);
    }
}
