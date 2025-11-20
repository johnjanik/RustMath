//! Dyck words and paths
//!
//! A Dyck word is a sequence of n X's and n Y's such that no initial segment
//! has more Y's than X's. These correspond to balanced parentheses and
//! monotonic lattice paths.

/// A Dyck word - a sequence of n X's and n Y's such that no initial segment
/// has more Y's than X's
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DyckWord {
    /// The word represented as a vector where true = X, false = Y
    word: Vec<bool>,
}

impl DyckWord {
    /// Create a Dyck word from a boolean vector
    ///
    /// Returns None if not a valid Dyck word
    pub fn new(word: Vec<bool>) -> Option<Self> {
        if word.len() % 2 != 0 {
            return None;
        }

        let mut balance = 0i32;
        let mut x_count = 0;
        let mut y_count = 0;

        for &bit in &word {
            if bit {
                balance += 1;
                x_count += 1;
            } else {
                balance -= 1;
                y_count += 1;
            }

            if balance < 0 {
                return None; // More Y's than X's at some point
            }
        }

        if x_count != y_count {
            return None;
        }

        Some(DyckWord { word })
    }

    /// Get the half-length (number of X's = number of Y's)
    pub fn half_length(&self) -> usize {
        self.word.len() / 2
    }

    /// Get the word as a boolean slice
    pub fn as_slice(&self) -> &[bool] {
        &self.word
    }

    /// Convert to string representation (X's and Y's)
    pub fn to_string(&self) -> String {
        self.word
            .iter()
            .map(|&b| if b { 'X' } else { 'Y' })
            .collect()
    }

    /// Convert to parentheses representation
    pub fn to_parens(&self) -> String {
        self.word
            .iter()
            .map(|&b| if b { '(' } else { ')' })
            .collect()
    }
}

/// Generate all Dyck words of half-length n
///
/// The number of Dyck words of half-length n is the Catalan number C_n
pub fn dyck_words(n: usize) -> Vec<DyckWord> {
    let mut result = Vec::new();
    let mut current = Vec::new();

    generate_dyck_words(n, n, &mut current, &mut result);

    result
}

fn generate_dyck_words(
    x_remaining: usize,
    y_remaining: usize,
    current: &mut Vec<bool>,
    result: &mut Vec<DyckWord>,
) {
    if x_remaining == 0 && y_remaining == 0 {
        result.push(DyckWord {
            word: current.clone(),
        });
        return;
    }

    // Can always add an X if we have any left
    if x_remaining > 0 {
        current.push(true);
        generate_dyck_words(x_remaining - 1, y_remaining, current, result);
        current.pop();
    }

    // Can only add a Y if we have more X's than Y's used so far
    // This means: (n - x_remaining) > (n - y_remaining), which simplifies to y_remaining > x_remaining
    if y_remaining > x_remaining {
        current.push(false);
        generate_dyck_words(x_remaining, y_remaining - 1, current, result);
        current.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dyck_words() {
        // Dyck words of length 3 should equal Catalan(3) = 5
        let words = dyck_words(3);
        assert_eq!(words.len(), 5);

        // Verify all are valid Dyck words
        for word in &words {
            assert_eq!(word.half_length(), 3);

            // Check balance property
            let mut balance = 0;
            for &bit in word.as_slice() {
                if bit {
                    balance += 1;
                } else {
                    balance -= 1;
                }
                assert!(balance >= 0);
            }
            assert_eq!(balance, 0);
        }
    }

    #[test]
    fn test_dyck_word_count() {
        // Number of Dyck words should match Catalan numbers
        assert_eq!(dyck_words(0).len(), 1); // C_0 = 1
        assert_eq!(dyck_words(1).len(), 1); // C_1 = 1
        assert_eq!(dyck_words(2).len(), 2); // C_2 = 2
        assert_eq!(dyck_words(3).len(), 5); // C_3 = 5
        assert_eq!(dyck_words(4).len(), 14); // C_4 = 14
    }

    #[test]
    fn test_dyck_word_representations() {
        // Test string conversions
        let word = DyckWord::new(vec![true, false, true, false]).unwrap();
        assert_eq!(word.to_string(), "XYXY");
        assert_eq!(word.to_parens(), "()()");

        let word2 = DyckWord::new(vec![true, true, false, false]).unwrap();
        assert_eq!(word2.to_string(), "XXYY");
        assert_eq!(word2.to_parens(), "(())");
    }

    #[test]
    fn test_invalid_dyck_word() {
        // More Y's than X's at some point
        assert!(DyckWord::new(vec![false, true, true, false]).is_none());

        // Odd length
        assert!(DyckWord::new(vec![true, false, true]).is_none());

        // Unequal counts
        assert!(DyckWord::new(vec![true, true, false, true]).is_none());
    }
}
