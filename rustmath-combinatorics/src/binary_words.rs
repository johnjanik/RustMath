//! Binary words and operations
//!
//! A binary word is a sequence of 0s and 1s. This module provides utilities
//! for working with binary words, including generation, manipulation, and
//! various combinatorial properties.

use std::fmt;

/// A binary word of length n
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BinaryWord {
    /// The word as a vector of bits (true = 1, false = 0)
    bits: Vec<bool>,
}

impl BinaryWord {
    /// Create a binary word from a vector of bools
    pub fn new(bits: Vec<bool>) -> Self {
        BinaryWord { bits }
    }

    /// Create a binary word from a vector of 0s and 1s
    pub fn from_u8(values: Vec<u8>) -> Option<Self> {
        if values.iter().any(|&x| x > 1) {
            return None;
        }
        Some(BinaryWord {
            bits: values.into_iter().map(|x| x == 1).collect(),
        })
    }

    /// Create a binary word from an integer (little-endian)
    pub fn from_usize(value: usize, length: usize) -> Self {
        let bits = (0..length).map(|i| (value >> i) & 1 == 1).collect();
        BinaryWord { bits }
    }

    /// Convert to integer (little-endian)
    pub fn to_usize(&self) -> usize {
        self.bits
            .iter()
            .enumerate()
            .map(|(i, &b)| if b { 1 << i } else { 0 })
            .sum()
    }

    /// Create a zero word of given length
    pub fn zeros(length: usize) -> Self {
        BinaryWord {
            bits: vec![false; length],
        }
    }

    /// Create a word of all ones of given length
    pub fn ones(length: usize) -> Self {
        BinaryWord {
            bits: vec![true; length],
        }
    }

    /// Get the length of the word
    pub fn len(&self) -> usize {
        self.bits.len()
    }

    /// Check if the word is empty
    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    /// Get the bits as a slice
    pub fn bits(&self) -> &[bool] {
        &self.bits
    }

    /// Get the Hamming weight (number of 1s)
    pub fn hamming_weight(&self) -> usize {
        self.bits.iter().filter(|&&b| b).count()
    }

    /// Compute Hamming distance to another word
    pub fn hamming_distance(&self, other: &BinaryWord) -> Option<usize> {
        if self.len() != other.len() {
            return None;
        }

        let distance = self
            .bits
            .iter()
            .zip(other.bits.iter())
            .filter(|(a, b)| a != b)
            .count();

        Some(distance)
    }

    /// Bitwise NOT
    pub fn not(&self) -> BinaryWord {
        BinaryWord {
            bits: self.bits.iter().map(|&b| !b).collect(),
        }
    }

    /// Bitwise AND
    pub fn and(&self, other: &BinaryWord) -> Option<BinaryWord> {
        if self.len() != other.len() {
            return None;
        }

        Some(BinaryWord {
            bits: self
                .bits
                .iter()
                .zip(other.bits.iter())
                .map(|(&a, &b)| a && b)
                .collect(),
        })
    }

    /// Bitwise OR
    pub fn or(&self, other: &BinaryWord) -> Option<BinaryWord> {
        if self.len() != other.len() {
            return None;
        }

        Some(BinaryWord {
            bits: self
                .bits
                .iter()
                .zip(other.bits.iter())
                .map(|(&a, &b)| a || b)
                .collect(),
        })
    }

    /// Bitwise XOR
    pub fn xor(&self, other: &BinaryWord) -> Option<BinaryWord> {
        if self.len() != other.len() {
            return None;
        }

        Some(BinaryWord {
            bits: self
                .bits
                .iter()
                .zip(other.bits.iter())
                .map(|(&a, &b)| a != b)
                .collect(),
        })
    }

    /// Concatenate two binary words
    pub fn concat(&self, other: &BinaryWord) -> BinaryWord {
        let mut bits = self.bits.clone();
        bits.extend(other.bits.iter());
        BinaryWord { bits }
    }

    /// Get a substring
    pub fn substring(&self, start: usize, length: usize) -> Option<BinaryWord> {
        if start + length > self.len() {
            return None;
        }

        Some(BinaryWord {
            bits: self.bits[start..start + length].to_vec(),
        })
    }

    /// Reverse the word
    pub fn reverse(&self) -> BinaryWord {
        let mut bits = self.bits.clone();
        bits.reverse();
        BinaryWord { bits }
    }

    /// Rotate left by k positions
    pub fn rotate_left(&self, k: usize) -> BinaryWord {
        if self.is_empty() {
            return self.clone();
        }

        let k = k % self.len();
        let mut bits = self.bits.clone();
        bits.rotate_left(k);
        BinaryWord { bits }
    }

    /// Rotate right by k positions
    pub fn rotate_right(&self, k: usize) -> BinaryWord {
        if self.is_empty() {
            return self.clone();
        }

        let k = k % self.len();
        let mut bits = self.bits.clone();
        bits.rotate_right(k);
        BinaryWord { bits }
    }

    /// Count the number of runs (maximal sequences of same bit)
    pub fn num_runs(&self) -> usize {
        if self.is_empty() {
            return 0;
        }

        let mut count = 1;
        for i in 1..self.len() {
            if self.bits[i] != self.bits[i - 1] {
                count += 1;
            }
        }
        count
    }

    /// Get the lengths of all runs
    pub fn run_lengths(&self) -> Vec<usize> {
        if self.is_empty() {
            return vec![];
        }

        let mut lengths = vec![1];
        for i in 1..self.len() {
            if self.bits[i] == self.bits[i - 1] {
                *lengths.last_mut().unwrap() += 1;
            } else {
                lengths.push(1);
            }
        }
        lengths
    }

    /// Check if the word is a palindrome
    pub fn is_palindrome(&self) -> bool {
        let n = self.len();
        for i in 0..n / 2 {
            if self.bits[i] != self.bits[n - 1 - i] {
                return false;
            }
        }
        true
    }

    /// Check if the word is a necklace (lexicographically minimal rotation)
    pub fn is_necklace(&self) -> bool {
        let n = self.len();
        for k in 1..n {
            let rotated = self.rotate_left(k);
            if rotated.bits < self.bits {
                return false;
            }
        }
        true
    }

    /// Check if the word is a Lyndon word (strictly less than all its non-trivial rotations)
    pub fn is_lyndon(&self) -> bool {
        let n = self.len();
        for k in 1..n {
            let rotated = self.rotate_left(k);
            if rotated.bits <= self.bits {
                return false;
            }
        }
        true
    }

    /// Compute the Lyndon factorization using Duval's algorithm
    ///
    /// Returns the unique factorization of this word into non-increasing Lyndon words.
    /// This is an O(n) algorithm discovered by Duval in 1983.
    ///
    /// # Example
    /// ```
    /// use rustmath_combinatorics::BinaryWord;
    /// let w = BinaryWord::from_string("0010110").unwrap();
    /// let factors = w.lyndon_factorization();
    /// // factors will be ["001", "011", "0"]
    /// ```
    pub fn lyndon_factorization(&self) -> Vec<BinaryWord> {
        let n = self.len();
        if n == 0 {
            return vec![];
        }

        let mut factors = Vec::new();
        let mut i = 0;

        while i < n {
            let mut j = i + 1;
            let mut k = i;

            // Find the next Lyndon word
            while j < n && self.bits[k] <= self.bits[j] {
                if self.bits[k] < self.bits[j] {
                    k = i;
                } else {
                    k += 1;
                }
                j += 1;
            }

            // Extract Lyndon factors from [i..j)
            while i <= k {
                let factor = BinaryWord {
                    bits: self.bits[i..i + j - k].to_vec(),
                };
                factors.push(factor);
                i += j - k;
            }
        }

        factors
    }

    /// Convert to string representation
    pub fn to_string(&self) -> String {
        self.bits.iter().map(|&b| if b { '1' } else { '0' }).collect()
    }

    /// Parse from string
    pub fn from_string(s: &str) -> Option<BinaryWord> {
        let bits: Option<Vec<bool>> = s
            .chars()
            .map(|c| match c {
                '0' => Some(false),
                '1' => Some(true),
                _ => None,
            })
            .collect();

        bits.map(|b| BinaryWord { bits: b })
    }
}

impl fmt::Display for BinaryWord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

/// Generate all binary words of length n
pub fn all_binary_words(n: usize) -> Vec<BinaryWord> {
    let count = 1 << n;
    (0..count)
        .map(|i| BinaryWord::from_usize(i, n))
        .collect()
}

/// Generate all binary words of length n with exactly k ones
pub fn binary_words_with_weight(n: usize, k: usize) -> Vec<BinaryWord> {
    if k > n {
        return vec![];
    }

    let mut result = Vec::new();
    let mut current = vec![false; n];

    generate_words_with_weight(n, k, 0, &mut current, &mut result);

    result
}

fn generate_words_with_weight(
    n: usize,
    k: usize,
    start: usize,
    current: &mut Vec<bool>,
    result: &mut Vec<BinaryWord>,
) {
    let ones_count = current.iter().filter(|&&b| b).count();

    if ones_count == k {
        result.push(BinaryWord::new(current.clone()));
        return;
    }

    if start >= n || ones_count + (n - start) < k {
        return; // Can't reach k ones
    }

    // Try setting current position to 1
    current[start] = true;
    generate_words_with_weight(n, k, start + 1, current, result);

    // Try setting current position to 0
    current[start] = false;
    generate_words_with_weight(n, k, start + 1, current, result);
}

/// Generate all necklaces of length n (equivalence classes under rotation)
pub fn necklaces(n: usize) -> Vec<BinaryWord> {
    all_binary_words(n)
        .into_iter()
        .filter(|w| w.is_necklace())
        .collect()
}

/// Generate all Lyndon words of length n
pub fn lyndon_words(n: usize) -> Vec<BinaryWord> {
    all_binary_words(n)
        .into_iter()
        .filter(|w| w.is_lyndon())
        .collect()
}

/// Generate all necklaces of length n with exactly k ones (fixed density)
///
/// This is more efficient than filtering all necklaces by weight, as it only
/// generates necklaces with the desired number of ones.
///
/// Uses a recursive algorithm based on the FKM (Fredricksen-Kessler-Maiorana)
/// algorithm adapted for fixed content.
///
/// # Example
/// ```
/// use rustmath_combinatorics::binary_words::necklaces_with_weight;
/// let necklaces = necklaces_with_weight(4, 2);
/// // Returns necklaces with exactly 2 ones: "0011", "0101"
/// ```
pub fn necklaces_with_weight(n: usize, k: usize) -> Vec<BinaryWord> {
    if k > n {
        return vec![];
    }
    if n == 0 {
        return vec![BinaryWord::new(vec![])];
    }
    if k == 0 {
        return vec![BinaryWord::zeros(n)];
    }
    if k == n {
        return vec![BinaryWord::ones(n)];
    }

    let mut result = Vec::new();
    let mut current = vec![false; n];
    // Start with position 1, period 1 (1-indexed in the algorithm)
    generate_necklaces_with_weight_helper(n, k, 1, 1, 0, &mut current, &mut result);
    result
}

/// Recursive helper for generating fixed-density necklaces
///
/// This uses a modified FKM algorithm with 1-indexed positions internally.
/// Parameters:
/// - n: length of necklace
/// - k: desired number of ones
/// - t: current position (1-indexed)
/// - p: period of the current prefix
/// - ones_count: number of ones in positions [0..t)
fn generate_necklaces_with_weight_helper(
    n: usize,
    k: usize,
    t: usize,
    p: usize,
    ones_count: usize,
    current: &mut Vec<bool>,
    result: &mut Vec<BinaryWord>,
) {
    if t > n {
        // Check if we have exactly k ones and if this is a necklace (n divisible by period)
        if ones_count == k && n % p == 0 {
            result.push(BinaryWord::new(current.clone()));
        }
        return;
    }

    // Calculate remaining positions and how many more ones we need
    let remaining = n - t + 1;
    let ones_needed = k.saturating_sub(ones_count);

    // Prune: if we can't reach exactly k ones, skip
    if ones_needed > remaining || ones_count > k {
        return;
    }

    // Try extending with the same value as position t-p
    // Note: t and p are 1-indexed, but current is 0-indexed
    let prev_idx = if t > p { t - p - 1 } else { 0 };
    let prev_bit = current[prev_idx];

    current[t - 1] = prev_bit;
    let new_ones = ones_count + if prev_bit { 1 } else { 0 };
    generate_necklaces_with_weight_helper(n, k, t + 1, p, new_ones, current, result);

    // Try extending with 1 if it's lexicographically larger than position t-p
    if !prev_bit {
        current[t - 1] = true;
        generate_necklaces_with_weight_helper(n, k, t + 1, t, ones_count + 1, current, result);
    }
}

/// Generate all Lyndon words of length n with exactly k ones
///
/// A more efficient alternative to filtering all Lyndon words by weight.
pub fn lyndon_words_with_weight(n: usize, k: usize) -> Vec<BinaryWord> {
    necklaces_with_weight(n, k)
        .into_iter()
        .filter(|w| w.is_lyndon())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repeat(word: &BinaryWord, n: usize) -> BinaryWord {
        let mut bits = Vec::with_capacity(word.len() * n);
        for _ in 0..n {
            bits.extend(word.bits());
        }
        BinaryWord::new(bits)
    }

    #[test]
    fn test_binary_word_creation() {
        let w1 = BinaryWord::new(vec![true, false, true, true]);
        assert_eq!(w1.len(), 4);
        assert_eq!(w1.hamming_weight(), 3);

        let w2 = BinaryWord::from_u8(vec![1, 0, 1, 1]).unwrap();
        assert_eq!(w1, w2);

        let w3 = BinaryWord::from_string("1011").unwrap();
        assert_eq!(w1, w3);
    }

    #[test]
    fn test_conversion() {
        let w = BinaryWord::from_usize(11, 4); // 1011 in binary
        assert_eq!(w.to_string(), "1101"); // Little-endian
        assert_eq!(w.to_usize(), 11);
    }

    #[test]
    fn test_hamming_distance() {
        let w1 = BinaryWord::from_string("1011").unwrap();
        let w2 = BinaryWord::from_string("1001").unwrap();
        assert_eq!(w1.hamming_distance(&w2), Some(1));

        let w3 = BinaryWord::from_string("0000").unwrap();
        assert_eq!(w1.hamming_distance(&w3), Some(3));
    }

    #[test]
    fn test_bitwise_operations() {
        let w1 = BinaryWord::from_string("1100").unwrap();
        let w2 = BinaryWord::from_string("1010").unwrap();

        assert_eq!(w1.and(&w2).unwrap().to_string(), "1000");
        assert_eq!(w1.or(&w2).unwrap().to_string(), "1110");
        assert_eq!(w1.xor(&w2).unwrap().to_string(), "0110");
        assert_eq!(w1.not().to_string(), "0011");
    }

    #[test]
    fn test_concat_substring() {
        let w1 = BinaryWord::from_string("110").unwrap();
        let w2 = BinaryWord::from_string("01").unwrap();

        let concat = w1.concat(&w2);
        assert_eq!(concat.to_string(), "11001");

        let sub = concat.substring(1, 3).unwrap();
        assert_eq!(sub.to_string(), "100");
    }

    #[test]
    fn test_reverse() {
        let w = BinaryWord::from_string("1011").unwrap();
        assert_eq!(w.reverse().to_string(), "1101");
    }

    #[test]
    fn test_rotate() {
        let w = BinaryWord::from_string("10110").unwrap();
        assert_eq!(w.rotate_left(2).to_string(), "11010");
        assert_eq!(w.rotate_right(1).to_string(), "01011");
    }

    #[test]
    fn test_runs() {
        let w = BinaryWord::from_string("1110011").unwrap();
        assert_eq!(w.num_runs(), 3);
        assert_eq!(w.run_lengths(), vec![3, 2, 2]);

        let w2 = BinaryWord::from_string("1010101").unwrap();
        assert_eq!(w2.num_runs(), 7);
    }

    #[test]
    fn test_palindrome() {
        let w1 = BinaryWord::from_string("10101").unwrap();
        assert!(w1.is_palindrome());

        let w2 = BinaryWord::from_string("1011").unwrap();
        assert!(!w2.is_palindrome());

        let w3 = BinaryWord::from_string("110011").unwrap();
        assert!(w3.is_palindrome());
    }

    #[test]
    fn test_necklace() {
        // "0011" is a necklace (minimal among rotations)
        let w1 = BinaryWord::from_string("0011").unwrap();
        assert!(w1.is_necklace());

        // "0110" is not a necklace (rotation "0011" is smaller)
        let w2 = BinaryWord::from_string("0110").unwrap();
        assert!(!w2.is_necklace());
    }

    #[test]
    fn test_lyndon() {
        // "0011" is a Lyndon word
        let w1 = BinaryWord::from_string("0011").unwrap();
        assert!(w1.is_lyndon());

        // "0101" is not a Lyndon word (rotation "0101" is equal)
        let w2 = BinaryWord::from_string("0101").unwrap();
        assert!(!w2.is_lyndon());
    }

    #[test]
    fn test_all_binary_words() {
        let words = all_binary_words(3);
        assert_eq!(words.len(), 8);

        // Check a few
        assert!(words.contains(&BinaryWord::from_string("000").unwrap()));
        assert!(words.contains(&BinaryWord::from_string("111").unwrap()));
    }

    #[test]
    fn test_binary_words_with_weight() {
        let words = binary_words_with_weight(4, 2);
        assert_eq!(words.len(), 6); // C(4,2) = 6

        // All should have hamming weight 2
        for word in &words {
            assert_eq!(word.hamming_weight(), 2);
        }
    }

    #[test]
    fn test_necklaces_count() {
        // For n=3, there are 4 necklaces: 000, 001, 011, 111
        let necklaces_3 = necklaces(3);
        assert_eq!(necklaces_3.len(), 4);

        // For n=4, there are 6 necklaces
        let necklaces_4 = necklaces(4);
        assert_eq!(necklaces_4.len(), 6);
    }

    #[test]
    fn test_lyndon_words_count() {
        // For n=3, Lyndon words over binary alphabet: 001, 011
        // (111 is not a Lyndon word because all rotations are equal)
        let lyndon_3 = lyndon_words(3);
        assert_eq!(lyndon_3.len(), 2);

        // For n=4, there should be 3 Lyndon words: 0001, 0011, 0111
        let lyndon_4 = lyndon_words(4);
        assert_eq!(lyndon_4.len(), 3);
    }

    #[test]
    fn test_lyndon_factorization_basic() {
        // Test basic Lyndon factorizations
        let w1 = BinaryWord::from_string("00101100").unwrap();
        let factors = w1.lyndon_factorization();
        assert_eq!(factors.len(), 3);
        assert_eq!(factors[0].to_string(), "001011");
        assert_eq!(factors[1].to_string(), "0");
        assert_eq!(factors[2].to_string(), "0");

        // Verify each factor is a Lyndon word (except possibly repeats)
        for factor in &factors {
            if factor.len() > 0 {
                // Each factor should be Lyndon or a repeat of a Lyndon word
                let is_lyndon_or_repeat = factor.is_lyndon() || {
                    // Check if it's a repeat of a Lyndon word
                    let mut is_repeat = false;
                    for divisor in 1..factor.len() {
                        if factor.len() % divisor == 0 {
                            let period = factor.substring(0, divisor).unwrap();
                            let repetitions = factor.len() / divisor;
                            let repeated = repeat(&period, repetitions);
                            if repeated == *factor && period.is_lyndon() {
                                is_repeat = true;
                                break;
                            }
                        }
                    }
                    is_repeat
                };
                assert!(is_lyndon_or_repeat || factor.to_string() == "0" || factor.to_string() == "1");
            }
        }
    }

    #[test]
    fn test_lyndon_factorization_lyndon_word() {
        // A Lyndon word should factorize to itself
        let w = BinaryWord::from_string("00011").unwrap();
        assert!(w.is_lyndon());
        let factors = w.lyndon_factorization();
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0], w);
    }

    #[test]
    fn test_lyndon_factorization_empty() {
        let w = BinaryWord::new(vec![]);
        let factors = w.lyndon_factorization();
        assert_eq!(factors.len(), 0);
    }

    #[test]
    fn test_lyndon_factorization_single_bit() {
        let w0 = BinaryWord::from_string("0").unwrap();
        let factors0 = w0.lyndon_factorization();
        assert_eq!(factors0.len(), 1);
        assert_eq!(factors0[0].to_string(), "0");

        let w1 = BinaryWord::from_string("1").unwrap();
        let factors1 = w1.lyndon_factorization();
        assert_eq!(factors1.len(), 1);
        assert_eq!(factors1[0].to_string(), "1");
    }

    #[test]
    fn test_lyndon_factorization_all_zeros() {
        let w = BinaryWord::from_string("0000").unwrap();
        let factors = w.lyndon_factorization();
        // "0000" = "0" * 4
        assert_eq!(factors.len(), 4);
        for factor in factors {
            assert_eq!(factor.to_string(), "0");
        }
    }

    #[test]
    fn test_lyndon_factorization_decreasing() {
        // Lyndon factorization should produce non-increasing sequence
        let w = BinaryWord::from_string("001011001").unwrap();
        let factors = w.lyndon_factorization();

        // Check that factors are in non-increasing lexicographic order
        for i in 1..factors.len() {
            assert!(factors[i - 1].bits >= factors[i].bits);
        }
    }

    #[test]
    fn test_necklaces_with_weight_basic() {
        // Necklaces of length 4 with 2 ones
        let necklaces = necklaces_with_weight(4, 2);

        // Should have: "0011", "0101"
        // "0110" is equivalent to "0011" (rotation)
        // "1001" is equivalent to "0011" (rotation)
        // "1010" is equivalent to "0101" (rotation)
        // "1100" is equivalent to "0011" (rotation)
        assert!(necklaces.len() >= 2);

        // All should be necklaces
        for necklace in &necklaces {
            assert!(necklace.is_necklace());
            assert_eq!(necklace.hamming_weight(), 2);
        }
    }

    #[test]
    fn test_necklaces_with_weight_edge_cases() {
        // Length 0
        let n0 = necklaces_with_weight(0, 0);
        assert_eq!(n0.len(), 1);
        assert_eq!(n0[0].len(), 0);

        // All zeros
        let all_zeros = necklaces_with_weight(5, 0);
        assert_eq!(all_zeros.len(), 1);
        assert_eq!(all_zeros[0].to_string(), "00000");

        // All ones
        let all_ones = necklaces_with_weight(5, 5);
        assert_eq!(all_ones.len(), 1);
        assert_eq!(all_ones[0].to_string(), "11111");

        // Impossible case
        let impossible = necklaces_with_weight(3, 5);
        assert_eq!(impossible.len(), 0);
    }

    #[test]
    fn test_necklaces_with_weight_length_3() {
        // Length 3, weight 1: "001"
        let n31 = necklaces_with_weight(3, 1);
        assert_eq!(n31.len(), 1);
        assert_eq!(n31[0].to_string(), "001");

        // Length 3, weight 2: "011"
        let n32 = necklaces_with_weight(3, 2);
        assert_eq!(n32.len(), 1);
        assert_eq!(n32[0].to_string(), "011");
    }

    #[test]
    fn test_necklaces_with_weight_properties() {
        // Test that all generated necklaces have correct weight
        for n in 1..=6 {
            for k in 0..=n {
                let necklaces = necklaces_with_weight(n, k);
                for necklace in &necklaces {
                    assert_eq!(necklace.len(), n);
                    assert_eq!(necklace.hamming_weight(), k);
                    assert!(necklace.is_necklace());
                }

                // Check for duplicates
                for i in 0..necklaces.len() {
                    for j in (i + 1)..necklaces.len() {
                        assert_ne!(necklaces[i], necklaces[j]);
                    }
                }
            }
        }
    }

    #[test]
    fn test_lyndon_words_with_weight() {
        // Lyndon words of length 4 with 2 ones
        let lyndon = lyndon_words_with_weight(4, 2);

        // All should be Lyndon words with weight 2
        for word in &lyndon {
            assert!(word.is_lyndon());
            assert_eq!(word.hamming_weight(), 2);
        }

        // Should include "0011" and "0111" but not "0101" (not Lyndon)
        let strings: Vec<String> = lyndon.iter().map(|w| w.to_string()).collect();
        assert!(strings.contains(&"0011".to_string()));

        // "0101" is not a Lyndon word (rotation "0101" equals itself)
        assert!(!strings.contains(&"0101".to_string()));
    }

    #[test]
    fn test_lyndon_factorization_concatenation() {
        // The concatenation of Lyndon factorization should equal original word
        let words = vec![
            "0010110",
            "11001",
            "0001111",
            "10101010",
            "00011101",
        ];

        for word_str in words {
            let w = BinaryWord::from_string(word_str).unwrap();
            let factors = w.lyndon_factorization();

            // Concatenate all factors
            let mut reconstructed = BinaryWord::new(vec![]);
            for factor in factors {
                reconstructed = reconstructed.concat(&factor);
            }

            assert_eq!(reconstructed, w);
        }
    }
}
