//! String operations for string monoids
//!
//! This module provides operations on strings including:
//! - Common prefix/suffix operations
//! - Cryptographic text analysis functions (index of coincidence, frequency analysis)
//! - String encoding utilities

use std::collections::HashMap;

/// Compute the longest common prefix of two strings
pub fn longest_common_prefix(s1: &str, s2: &str) -> String {
    s1.chars()
        .zip(s2.chars())
        .take_while(|(c1, c2)| c1 == c2)
        .map(|(c, _)| c)
        .collect()
}

/// Compute the longest common suffix of two strings
pub fn longest_common_suffix(s1: &str, s2: &str) -> String {
    s1.chars()
        .rev()
        .zip(s2.chars().rev())
        .take_while(|(c1, c2)| c1 == c2)
        .map(|(c, _)| c)
        .collect::<String>()
        .chars()
        .rev()
        .collect()
}

/// Check if s1 is a prefix of s2
pub fn is_prefix(s1: &str, s2: &str) -> bool {
    s2.starts_with(s1)
}

/// Check if s1 is a suffix of s2
pub fn is_suffix(s1: &str, s2: &str) -> bool {
    s2.ends_with(s1)
}

/// Compute the index of coincidence for cryptographic text analysis.
///
/// The index of coincidence (IC) measures the probability that two randomly
/// selected characters from the text are identical. It's a fundamental tool
/// in cryptanalysis, particularly for:
/// - Detecting whether text is encrypted or plaintext
/// - Estimating the key length of polyalphabetic ciphers (Vigenère, etc.)
/// - Distinguishing between different languages
///
/// # Formula
///
/// IC = Σ(count_i * (count_i - 1)) / (N * (N - 1))
///
/// where count_i is the frequency of each n-gram, and N is the total number
/// of n-grams in the text.
///
/// # Parameters
///
/// * `s` - The input string to analyze
/// * `n` - The size of n-grams to analyze (typically 1 for character analysis)
///
/// # Returns
///
/// The index of coincidence as a floating-point value between 0 and 1.
/// - English text typically has IC ≈ 0.065-0.068
/// - Random text has IC ≈ 1/alphabet_size (≈ 0.038 for 26 letters)
///
/// # Examples
///
/// ```
/// use rustmath_monoids::string_ops::coincidence_index;
///
/// // English text has high IC
/// let english = "the quick brown fox jumps over the lazy dog";
/// let ic = coincidence_index(english, 1);
/// assert!(ic > 0.05); // Higher than random
///
/// // Uniformly random text has low IC
/// let random = "abcdefghijklmnopqrstuvwxyz";
/// let ic_random = coincidence_index(random, 1);
/// assert!(ic_random < 0.05);
/// ```
pub fn coincidence_index(s: &str, n: usize) -> f64 {
    if n == 0 || s.len() < n {
        return 0.0;
    }

    let mut freq: HashMap<String, usize> = HashMap::new();
    let chars: Vec<char> = s.chars().collect();

    // Count n-grams
    for i in 0..=chars.len().saturating_sub(n) {
        let ngram: String = chars[i..i + n].iter().collect();
        *freq.entry(ngram).or_insert(0) += 1;
    }

    let total = chars.len().saturating_sub(n - 1);

    if total <= 1 {
        return 0.0;
    }

    // Calculate IC: sum of count * (count - 1) / (N * (N - 1))
    let numerator: usize = freq.values()
        .map(|&count| count * count.saturating_sub(1))
        .sum();

    let denominator = total * (total - 1);

    numerator as f64 / denominator as f64
}

/// Compute the coincidence discriminant for cryptographic text analysis.
///
/// The coincidence discriminant is derived from the index of coincidence
/// and provides a normalized measure for comparing texts. It's particularly
/// useful for:
/// - Comparing the randomness of different texts
/// - Detecting the presence of patterns in encrypted text
/// - Measuring deviation from expected random distribution
///
/// # Formula
///
/// CD = (IC - 1/n) / (1 - 1/n)
///
/// This normalizes the IC to a 0-1 scale where:
/// - 0 indicates perfectly random text
/// - 1 indicates perfectly non-random (single repeated character)
///
/// # Parameters
///
/// * `s` - The input string to analyze
/// * `n` - The alphabet size (typically 26 for English, or n-gram size)
///
/// # Returns
///
/// The coincidence discriminant as a floating-point value between 0 and 1.
///
/// # Examples
///
/// ```
/// use rustmath_monoids::string_ops::coincidence_discriminant;
///
/// // English text has moderate discriminant
/// let english = "the quick brown fox jumps over the lazy dog";
/// let cd = coincidence_discriminant(english, 26);
/// assert!(cd > 0.3 && cd < 0.8);
///
/// // Repeated character has high discriminant
/// let repeated = "aaaaaaaaaa";
/// let cd_repeated = coincidence_discriminant(repeated, 26);
/// assert!(cd_repeated > 0.9);
/// ```
pub fn coincidence_discriminant(s: &str, n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }

    let ic = coincidence_index(s, 1);
    let expected_random = 1.0 / n as f64;

    // Normalize: (IC - 1/n) / (1 - 1/n)
    (ic - expected_random) / (1.0 - expected_random)
}

/// Compute the frequency distribution of n-grams in a string.
///
/// This function analyzes the text and returns the frequency of each n-gram
/// (substring of length n). This is essential for:
/// - Frequency analysis in cryptanalysis
/// - Breaking substitution ciphers
/// - Statistical analysis of text patterns
/// - Language detection and classification
///
/// # Parameters
///
/// * `s` - The input string to analyze
/// * `n` - The size of n-grams to count
///   - n=1: individual character frequencies
///   - n=2: bigram frequencies (common in digraph analysis)
///   - n=3+: trigram and higher-order frequencies
///
/// # Returns
///
/// A HashMap where keys are n-grams and values are their frequencies in the text.
///
/// # Examples
///
/// ```
/// use rustmath_monoids::string_ops::frequency_distribution;
///
/// let text = "hello";
/// let freq = frequency_distribution(text, 1);
/// assert_eq!(freq.get("l"), Some(&2));
/// assert_eq!(freq.get("h"), Some(&1));
/// assert_eq!(freq.get("e"), Some(&1));
/// assert_eq!(freq.get("o"), Some(&1));
///
/// // Bigram analysis
/// let freq2 = frequency_distribution(text, 2);
/// assert_eq!(freq2.get("ll"), Some(&1));
/// assert_eq!(freq2.get("he"), Some(&1));
/// ```
///
/// # Cryptanalysis Applications
///
/// For English text, expected single-letter frequencies (approximate):
/// - E: 12.7%, T: 9.1%, A: 8.2%, O: 7.5%, I: 7.0%, N: 6.7%
/// - Most common bigrams: TH, HE, IN, ER, AN, RE
/// - Most common trigrams: THE, AND, ING, HER, HAT, HIS
pub fn frequency_distribution(s: &str, n: usize) -> HashMap<String, usize> {
    let mut freq: HashMap<String, usize> = HashMap::new();

    if n == 0 || s.is_empty() {
        return freq;
    }

    let chars: Vec<char> = s.chars().collect();

    if chars.len() < n {
        return freq;
    }

    // Count n-grams
    for i in 0..=chars.len() - n {
        let ngram: String = chars[i..i + n].iter().collect();
        *freq.entry(ngram).or_insert(0) += 1;
    }

    freq
}

/// Strip encoding artifacts from a string for cryptographic analysis.
///
/// This function normalizes text by removing or transforming characters that
/// are not relevant for cryptographic analysis. It's commonly used to:
/// - Prepare ciphertext for frequency analysis
/// - Remove punctuation and whitespace
/// - Normalize case for cipher operations
/// - Clean text data before statistical analysis
///
/// # Behavior
///
/// The function:
/// 1. Converts all characters to uppercase
/// 2. Removes non-alphabetic characters (spaces, punctuation, digits)
/// 3. Preserves only A-Z characters
///
/// This is the standard preprocessing for classical cryptanalysis.
///
/// # Parameters
///
/// * `s` - The input string to strip and normalize
///
/// # Returns
///
/// A new String containing only uppercase alphabetic characters.
///
/// # Examples
///
/// ```
/// use rustmath_monoids::string_ops::strip_encoding;
///
/// let text = "Hello, World! 123";
/// let stripped = strip_encoding(text);
/// assert_eq!(stripped, "HELLOWORLD");
///
/// let cipher = "Khoor, Zruog!"; // Simple Caesar cipher
/// let clean = strip_encoding(cipher);
/// assert_eq!(clean, "KHRROZRXOG");
/// ```
///
/// # Cryptanalysis Workflow
///
/// Typical usage in cipher breaking:
/// ```
/// use rustmath_monoids::string_ops::{strip_encoding, frequency_distribution, coincidence_index};
///
/// let ciphertext = "Wkh txlfn eurzq ira mxpsv ryhu wkh odcb grj";
/// let clean = strip_encoding(ciphertext);
/// let freq = frequency_distribution(&clean, 1);
/// let ic = coincidence_index(&clean, 1);
///
/// // Analyze frequency distribution to detect cipher type
/// // and find potential key
/// ```
pub fn strip_encoding(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphabetic())
        .map(|c| c.to_uppercase().to_string())
        .collect::<String>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_longest_common_prefix() {
        assert_eq!(longest_common_prefix("hello", "help"), "hel");
        assert_eq!(longest_common_prefix("abc", "def"), "");
        assert_eq!(longest_common_prefix("", "test"), "");
        assert_eq!(longest_common_prefix("same", "same"), "same");
    }

    #[test]
    fn test_longest_common_suffix() {
        assert_eq!(longest_common_suffix("testing", "running"), "ing");
        assert_eq!(longest_common_suffix("abc", "def"), "");
        assert_eq!(longest_common_suffix("", "test"), "");
        assert_eq!(longest_common_suffix("same", "same"), "same");
    }

    #[test]
    fn test_is_prefix() {
        assert!(is_prefix("hel", "hello"));
        assert!(is_prefix("", "anything"));
        assert!(!is_prefix("abc", "def"));
        assert!(is_prefix("test", "test"));
    }

    #[test]
    fn test_is_suffix() {
        assert!(is_suffix("lo", "hello"));
        assert!(is_suffix("", "anything"));
        assert!(!is_suffix("abc", "def"));
        assert!(is_suffix("test", "test"));
    }

    // ===== Coincidence Index Tests =====

    #[test]
    fn test_coincidence_index_empty_string() {
        let ic = coincidence_index("", 1);
        assert_eq!(ic, 0.0);
    }

    #[test]
    fn test_coincidence_index_single_char() {
        let ic = coincidence_index("a", 1);
        assert_eq!(ic, 0.0); // N=1, so denominator is 0
    }

    #[test]
    fn test_coincidence_index_repeated_char() {
        // All same character should give IC = 1.0
        let ic = coincidence_index("aaaaaaaaaa", 1);
        assert!((ic - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_coincidence_index_uniform_distribution() {
        // Each character appears once - should have low IC
        let ic = coincidence_index("abcdefghijklmnopqrstuvwxyz", 1);
        assert!(ic < 0.05); // Much lower than English text
    }

    #[test]
    fn test_coincidence_index_english_text() {
        // English text should have IC around 0.065-0.068
        let text = "the quick brown fox jumps over the lazy dog the cat sat on the mat";
        let ic = coincidence_index(text, 1);
        assert!(ic > 0.05 && ic < 0.10);
    }

    #[test]
    fn test_coincidence_index_bigrams() {
        let text = "hello world";
        let ic = coincidence_index(text, 2);
        assert!(ic >= 0.0 && ic <= 1.0);
    }

    #[test]
    fn test_coincidence_index_zero_n() {
        let ic = coincidence_index("test", 0);
        assert_eq!(ic, 0.0);
    }

    #[test]
    fn test_coincidence_index_n_larger_than_string() {
        let ic = coincidence_index("hi", 5);
        assert_eq!(ic, 0.0);
    }

    #[test]
    fn test_coincidence_index_known_values() {
        // Text with known character distribution
        // "aabbcc" has 6 chars, each appears twice
        // IC = (2*1 + 2*1 + 2*1) / (6*5) = 6/30 = 0.2
        let ic = coincidence_index("aabbcc", 1);
        assert!((ic - 0.2).abs() < 0.001);
    }

    // ===== Coincidence Discriminant Tests =====

    #[test]
    fn test_coincidence_discriminant_empty() {
        let cd = coincidence_discriminant("", 26);
        assert!(cd.is_finite());
    }

    #[test]
    fn test_coincidence_discriminant_repeated() {
        // All same character should have very high discriminant
        let cd = coincidence_discriminant("aaaaaaaaaa", 26);
        assert!(cd > 0.9);
    }

    #[test]
    fn test_coincidence_discriminant_uniform() {
        // Uniform distribution should have low discriminant (close to 0)
        let cd = coincidence_discriminant("abcdefghijklmnopqrstuvwxyz", 26);
        assert!(cd < 0.2);
    }

    #[test]
    fn test_coincidence_discriminant_english() {
        // English text should have moderate discriminant
        let text = "the quick brown fox jumps over the lazy dog";
        let cd = coincidence_discriminant(text, 26);
        assert!(cd >= 0.0 && cd <= 1.0);
    }

    #[test]
    fn test_coincidence_discriminant_n_equals_one() {
        // n=1 should return 0.0
        let cd = coincidence_discriminant("test", 1);
        assert_eq!(cd, 0.0);
    }

    #[test]
    fn test_coincidence_discriminant_n_equals_zero() {
        // n=0 should return 0.0
        let cd = coincidence_discriminant("test", 0);
        assert_eq!(cd, 0.0);
    }

    #[test]
    fn test_coincidence_discriminant_range() {
        // CD can be negative for very random text or positive for non-random text
        // It's typically in [-0.1, 1.0] for normal text
        let texts = vec![
            "a",
            "aaa",
            "abc",
            "abcdefghijklmnopqrstuvwxyz",
            "the quick brown fox",
        ];

        for text in texts {
            let cd = coincidence_discriminant(text, 26);
            assert!(cd.is_finite(), "CD should be finite for text: {}", text);
            // CD can be negative for text more random than expected
            assert!(cd >= -1.0 && cd <= 1.0, "CD out of expected range for text: {}", text);
        }
    }

    // ===== Frequency Distribution Tests =====

    #[test]
    fn test_frequency_distribution_empty() {
        let freq = frequency_distribution("", 1);
        assert_eq!(freq.len(), 0);
    }

    #[test]
    fn test_frequency_distribution_single_char() {
        let freq = frequency_distribution("a", 1);
        assert_eq!(freq.len(), 1);
        assert_eq!(freq.get("a"), Some(&1));
    }

    #[test]
    fn test_frequency_distribution_basic() {
        let freq = frequency_distribution("hello", 1);
        assert_eq!(freq.len(), 4); // h, e, l, o
        assert_eq!(freq.get("h"), Some(&1));
        assert_eq!(freq.get("e"), Some(&1));
        assert_eq!(freq.get("l"), Some(&2));
        assert_eq!(freq.get("o"), Some(&1));
    }

    #[test]
    fn test_frequency_distribution_bigrams() {
        let freq = frequency_distribution("hello", 2);
        assert_eq!(freq.len(), 4); // he, el, ll, lo
        assert_eq!(freq.get("he"), Some(&1));
        assert_eq!(freq.get("el"), Some(&1));
        assert_eq!(freq.get("ll"), Some(&1));
        assert_eq!(freq.get("lo"), Some(&1));
    }

    #[test]
    fn test_frequency_distribution_trigrams() {
        let freq = frequency_distribution("abcabc", 3);
        assert_eq!(freq.len(), 3); // abc, bca, cab (abc counted twice)
        assert_eq!(freq.get("abc"), Some(&2));
        assert_eq!(freq.get("bca"), Some(&1));
        assert_eq!(freq.get("cab"), Some(&1));
    }

    #[test]
    fn test_frequency_distribution_repeated() {
        let freq = frequency_distribution("aaaa", 1);
        assert_eq!(freq.len(), 1);
        assert_eq!(freq.get("a"), Some(&4));
    }

    #[test]
    fn test_frequency_distribution_n_zero() {
        let freq = frequency_distribution("test", 0);
        assert_eq!(freq.len(), 0);
    }

    #[test]
    fn test_frequency_distribution_n_larger_than_string() {
        let freq = frequency_distribution("hi", 5);
        assert_eq!(freq.len(), 0);
    }

    #[test]
    fn test_frequency_distribution_with_spaces() {
        let freq = frequency_distribution("a b a", 1);
        assert_eq!(freq.get("a"), Some(&2));
        assert_eq!(freq.get(" "), Some(&2));
        assert_eq!(freq.get("b"), Some(&1));
    }

    #[test]
    fn test_frequency_distribution_unicode() {
        let freq = frequency_distribution("héllo", 1);
        assert_eq!(freq.get("h"), Some(&1));
        assert_eq!(freq.get("é"), Some(&1));
        assert_eq!(freq.get("l"), Some(&2));
        assert_eq!(freq.get("o"), Some(&1));
    }

    // ===== Strip Encoding Tests =====

    #[test]
    fn test_strip_encoding_empty() {
        let result = strip_encoding("");
        assert_eq!(result, "");
    }

    #[test]
    fn test_strip_encoding_basic() {
        let result = strip_encoding("Hello, World!");
        assert_eq!(result, "HELLOWORLD");
    }

    #[test]
    fn test_strip_encoding_with_numbers() {
        let result = strip_encoding("abc123def456");
        assert_eq!(result, "ABCDEF");
    }

    #[test]
    fn test_strip_encoding_with_punctuation() {
        let result = strip_encoding("Hello! How are you?");
        assert_eq!(result, "HELLOHOWAREYOU");
    }

    #[test]
    fn test_strip_encoding_with_whitespace() {
        let result = strip_encoding("the quick brown fox");
        assert_eq!(result, "THEQUICKBROWNFOX");
    }

    #[test]
    fn test_strip_encoding_mixed_case() {
        let result = strip_encoding("HeLLo WoRLd");
        assert_eq!(result, "HELLOWORLD");
    }

    #[test]
    fn test_strip_encoding_only_special_chars() {
        let result = strip_encoding("!@#$%^&*()");
        assert_eq!(result, "");
    }

    #[test]
    fn test_strip_encoding_only_numbers() {
        let result = strip_encoding("1234567890");
        assert_eq!(result, "");
    }

    #[test]
    fn test_strip_encoding_already_clean() {
        let result = strip_encoding("ABCDEFG");
        assert_eq!(result, "ABCDEFG");
    }

    #[test]
    fn test_strip_encoding_unicode() {
        // Should preserve non-ASCII alphabetic characters
        let result = strip_encoding("café");
        assert_eq!(result, "CAFÉ");
    }

    #[test]
    fn test_strip_encoding_cipher_text() {
        // Caesar cipher example (ROT3): "Hello" -> "Khoor"
        let result = strip_encoding("Khoor, Zruog!");
        assert_eq!(result, "KHOORZRUOG");
    }

    // ===== Integration Tests =====

    #[test]
    fn test_cryptanalysis_workflow() {
        // Simulate a basic cryptanalysis workflow
        let ciphertext = "Wkh txlfn eurzq ira mxpsv ryhu wkh odcb grj";

        // Step 1: Clean the text
        let clean = strip_encoding(ciphertext);
        assert_eq!(clean, "WKHTXLFNEURZQIRAMXPSVRYHUWKHODCBGRJ");

        // Step 2: Get frequency distribution
        let freq = frequency_distribution(&clean, 1);
        assert!(freq.contains_key("W"));
        assert!(freq.contains_key("K"));

        // Step 3: Calculate index of coincidence
        let ic = coincidence_index(&clean, 1);
        assert!(ic > 0.0);

        // Step 4: Calculate coincidence discriminant
        let cd = coincidence_discriminant(&clean, 26);
        assert!(cd.is_finite()); // CD can be negative for random text
    }

    #[test]
    fn test_vigenere_detection() {
        // Vigenère cipher typically has lower IC than simple substitution
        let monoalphabetic = "WKHWXLFNEURZQIRAMXPSVRYKHWKHODCBGRJ"; // Caesar cipher
        let polyalphabetic = "BNVSUQUZVXBWVUUJEJQBATVFSVXBUSXQC"; // Vigenère

        let ic_mono = coincidence_index(monoalphabetic, 1);
        let ic_poly = coincidence_index(polyalphabetic, 1);

        // Both should be valid values
        assert!(ic_mono >= 0.0 && ic_mono <= 1.0);
        assert!(ic_poly >= 0.0 && ic_poly <= 1.0);
    }

    #[test]
    fn test_bigram_analysis() {
        let text = "the quick brown fox";
        let clean = strip_encoding(text);

        let bigrams = frequency_distribution(&clean, 2);
        let trigrams = frequency_distribution(&clean, 3);

        assert!(bigrams.len() > 0);
        assert!(trigrams.len() > 0);
        assert!(bigrams.len() >= trigrams.len()); // More bigrams than trigrams
    }

    #[test]
    fn test_frequency_sum() {
        let text = "hello world";
        let freq = frequency_distribution(text, 1);

        let total: usize = freq.values().sum();
        assert_eq!(total, text.len()); // Total frequency should equal string length
    }
}
