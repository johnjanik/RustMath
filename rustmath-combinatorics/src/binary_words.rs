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

#[cfg(test)]
mod tests {
    use super::*;

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
}
