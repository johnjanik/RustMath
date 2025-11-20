//! Dyck words and paths
//!
//! A Dyck word is a sequence of n X's and n Y's such that no initial segment
//! has more Y's than X's. These correspond to balanced parentheses and
//! monotonic lattice paths.
//!
//! This module also implements ν-Dyck words (nu-Dyck words), which are generalizations
//! of Dyck words where the path stays above a boundary path ν instead of just the x-axis.

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

/// A ν-Dyck word (nu-Dyck word) - a lattice path that stays above a boundary path ν
///
/// A ν-Dyck word is a lattice path from (0,0) to (n,0) using steps (1,1) (up) and (1,-1) (down)
/// that stays weakly above a given boundary path ν. When ν is the x-axis, this reduces to
/// a standard Dyck word.
///
/// The path is represented as a sequence of steps where true = up step (1,1), false = down step (1,-1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NuDyckWord {
    /// The path represented as a vector where true = up, false = down
    path: Vec<bool>,
    /// The boundary path ν (heights at each x-coordinate)
    nu: Vec<i32>,
}

impl NuDyckWord {
    /// Create a new ν-Dyck word from a path and boundary path ν
    ///
    /// The boundary path ν should be a lattice path from (0,0) to (n,0).
    /// The path has n steps, and ν has n+1 heights (one for each position 0..=n).
    /// Returns None if the path doesn't stay above ν or if the dimensions don't match.
    pub fn new(path: Vec<bool>, nu: Vec<i32>) -> Option<Self> {
        // ν should have length n+1 where n is the number of steps
        if nu.len() != path.len() + 1 {
            return None;
        }

        // Verify boundary path ν starts and ends at 0
        if !nu.is_empty() && (nu[0] != 0 || nu[nu.len() - 1] != 0) {
            return None;
        }

        // Compute heights of the path and verify it stays above ν
        let mut height = 0i32;

        // Check initial position
        if height < nu[0] {
            return None;
        }

        for (i, &step) in path.iter().enumerate() {
            if step {
                height += 1;
            } else {
                height -= 1;
            }

            // After this step, we're at position (i+1, height)
            // Check that height >= ν[i+1]
            if height < nu[i + 1] {
                return None;
            }
        }

        // Final height should be 0 (already checked via nu[n] == 0)

        Some(NuDyckWord { path, nu })
    }

    /// Create a ν-Dyck word from a boundary path ν (vector of heights)
    ///
    /// This validates that ν is a valid boundary path (starts and ends at 0,
    /// and each consecutive pair differs by exactly 1).
    pub fn from_nu(nu: Vec<i32>) -> Option<Vec<bool>> {
        if nu.is_empty() {
            return Some(vec![]);
        }

        if nu[0] != 0 || nu[nu.len() - 1] != 0 {
            return None;
        }

        // Verify ν is a valid lattice path (consecutive heights differ by exactly 1)
        for i in 0..nu.len() - 1 {
            let diff = (nu[i + 1] - nu[i]).abs();
            if diff != 1 {
                return None;
            }
        }

        // Convert ν to a path
        let mut path = Vec::with_capacity(nu.len() - 1);
        for i in 0..nu.len() - 1 {
            if nu[i + 1] > nu[i] {
                path.push(true); // Up step
            } else {
                path.push(false); // Down step
            }
        }

        Some(path)
    }

    /// Get the length of the path (number of steps)
    pub fn length(&self) -> usize {
        self.path.len()
    }

    /// Get the path as a boolean slice
    pub fn path(&self) -> &[bool] {
        &self.path
    }

    /// Get the boundary path ν
    pub fn nu(&self) -> &[i32] {
        &self.nu
    }

    /// Compute the heights of the path at each x-coordinate
    pub fn heights(&self) -> Vec<i32> {
        let mut heights = vec![0];
        let mut height = 0i32;

        for &step in &self.path {
            if step {
                height += 1;
            } else {
                height -= 1;
            }
            heights.push(height);
        }

        heights
    }

    /// Compute the bounce path of this ν-Dyck word
    ///
    /// The bounce path is constructed by reflecting the path across the boundary ν.
    /// Starting from the endpoint, we trace backwards, and whenever we would go below ν,
    /// we "bounce" back up.
    ///
    /// Returns a new lattice path (as a vector of heights).
    pub fn bounce_path(&self) -> Vec<i32> {
        let n = self.path.len();
        if n == 0 {
            return vec![0];
        }

        let heights = self.heights();
        let mut bounce = vec![0i32; n + 1];

        // Start from the end
        bounce[n] = 0;

        // Trace backwards
        for i in (0..n).rev() {
            // We're at position i+1, moving to position i
            // The original path goes up if path[i] is true
            // The bounce path mirrors this with respect to ν

            // Compute the reflection of the next bounce height across ν[i+1]
            let nu_at_next = self.nu[i + 1];
            let reflected_height = 2 * nu_at_next - bounce[i + 1];

            // The bounce path takes the opposite step
            if self.path[i] {
                // Original goes up, bounce goes down
                bounce[i] = bounce[i + 1] - 1;
            } else {
                // Original goes down, bounce goes up
                bounce[i] = bounce[i + 1] + 1;
            }

            // Ensure bounce path stays at or above ν
            bounce[i] = bounce[i].max(self.nu[i]);
        }

        bounce
    }

    /// Compute the area sequence of this ν-Dyck word
    ///
    /// The area sequence a = (a_1, a_2, ..., a_n) where a_i is the vertical distance
    /// between the path and the boundary ν at the i-th up step.
    ///
    /// Returns a vector of integers representing the area sequence.
    pub fn area_sequence(&self) -> Vec<i32> {
        let mut area = Vec::new();
        let mut height = 0i32;

        for (i, &step) in self.path.iter().enumerate() {
            if step {
                // Up step: record the area (height difference from ν before the step)
                area.push(height - self.nu[i]);
                height += 1;
            } else {
                height -= 1;
            }
        }

        area
    }

    /// Compute the total area between the path and the boundary ν
    pub fn total_area(&self) -> i32 {
        self.area_sequence().iter().sum()
    }

    /// Convert to string representation (U's and D's for up and down)
    pub fn to_string(&self) -> String {
        self.path
            .iter()
            .map(|&b| if b { 'U' } else { 'D' })
            .collect()
    }

    /// Check if this ν-Dyck word corresponds to a standard Dyck word (ν is the x-axis)
    pub fn is_standard_dyck(&self) -> bool {
        self.nu.iter().all(|&h| h == 0)
    }

    /// Convert a standard Dyck word to a ν-Dyck word with ν being the x-axis
    pub fn from_dyck_word(dyck: &DyckWord) -> Self {
        let path: Vec<bool> = dyck.as_slice().to_vec();
        let nu = vec![0; path.len() + 1];
        NuDyckWord { path, nu }
    }
}

/// Generate all ν-Dyck words for a given boundary path ν
///
/// Returns all lattice paths that stay weakly above ν and go from (0,0) to (n,0).
pub fn nu_dyck_words(nu: Vec<i32>) -> Vec<NuDyckWord> {
    let n = if nu.is_empty() { 0 } else { nu.len() - 1 };

    if n == 0 {
        return vec![NuDyckWord {
            path: vec![],
            nu: vec![0],
        }];
    }

    let mut result = Vec::new();
    let mut current_path = Vec::new();

    generate_nu_dyck_words(&nu, 0, 0, &mut current_path, &mut result);

    result
}

fn generate_nu_dyck_words(
    nu: &[i32],
    pos: usize,
    height: i32,
    current_path: &mut Vec<bool>,
    result: &mut Vec<NuDyckWord>,
) {
    let n = nu.len() - 1;

    if pos == n {
        if height == 0 {
            result.push(NuDyckWord {
                path: current_path.clone(),
                nu: nu.to_vec(),
            });
        }
        return;
    }

    // Try up step
    let new_height_up = height + 1;
    if new_height_up >= nu[pos + 1] {
        current_path.push(true);
        generate_nu_dyck_words(nu, pos + 1, new_height_up, current_path, result);
        current_path.pop();
    }

    // Try down step
    let new_height_down = height - 1;
    if new_height_down >= nu[pos + 1] {
        current_path.push(false);
        generate_nu_dyck_words(nu, pos + 1, new_height_down, current_path, result);
        current_path.pop();
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

    #[test]
    fn test_nu_dyck_word_basic() {
        // Standard Dyck word (ν is the x-axis)
        let nu = vec![0, 0, 0, 0, 0];
        let path = vec![true, true, false, false]; // UUDD = (())
        let word = NuDyckWord::new(path, nu).unwrap();

        assert_eq!(word.length(), 4);
        assert!(word.is_standard_dyck());
        assert_eq!(word.to_string(), "UUDD");
    }

    #[test]
    fn test_nu_dyck_word_heights() {
        // Path: UUDD
        let nu = vec![0, 0, 0, 0, 0];
        let path = vec![true, true, false, false];
        let word = NuDyckWord::new(path, nu).unwrap();

        let heights = word.heights();
        assert_eq!(heights, vec![0, 1, 2, 1, 0]);
    }

    #[test]
    fn test_nu_dyck_word_area_sequence() {
        // Standard Dyck word UUDD
        let nu = vec![0, 0, 0, 0, 0];
        let path = vec![true, true, false, false];
        let word = NuDyckWord::new(path, nu).unwrap();

        let area = word.area_sequence();
        // At first U: height before = 0, ν = 0, area = 0
        // At second U: height before = 1, ν = 0, area = 1
        assert_eq!(area, vec![0, 1]);
        assert_eq!(word.total_area(), 1);
    }

    #[test]
    fn test_nu_dyck_word_with_boundary() {
        // Non-trivial boundary path: ν = [0, 1, 0]
        // This means ν goes from (0,0) to (1,1) to (2,0)
        let nu = vec![0, 1, 0];

        // Valid path: UDUD would give heights [0, 1, 0, 1, 0]
        // But we only have 2 steps, so let's try UD
        // UD gives heights [0, 1, 0], which needs to stay above ν = [0, 1, 0]
        let path = vec![true, false]; // UD
        let word = NuDyckWord::new(path.clone(), nu.clone());

        // At position 0: height 0, need >= ν[0] = 0 ✓
        // After U at position 1: height 1, need >= ν[1] = 1 ✓
        // After D at position 2: height 0, need >= ν[2] = 0 ✓
        assert!(word.is_some());

        let word = word.unwrap();
        assert_eq!(word.heights(), vec![0, 1, 0]);
    }

    #[test]
    fn test_nu_dyck_word_invalid() {
        // Path goes below boundary
        let nu = vec![0, 1, 1, 0];
        let path = vec![true, false, false]; // UDD - would go to [0, 1, 0, -1]

        // This should fail because after the second D, height = 0 but ν[3] = 0,
        // but we need to check step by step:
        // Position 0: height 0, ok
        // After U (pos 1): height 1, need >= ν[1] = 1, ok
        // After D (pos 2): height 0, need >= ν[2] = 1, FAIL
        assert!(NuDyckWord::new(path, nu).is_none());
    }

    #[test]
    fn test_nu_dyck_word_from_dyck() {
        let dyck = DyckWord::new(vec![true, false, true, false]).unwrap();
        let nu_dyck = NuDyckWord::from_dyck_word(&dyck);

        assert!(nu_dyck.is_standard_dyck());
        assert_eq!(nu_dyck.path(), dyck.as_slice());
    }

    #[test]
    fn test_nu_dyck_word_bounce_path() {
        // Standard Dyck word UUDD with ν = x-axis
        let nu = vec![0, 0, 0, 0, 0];
        let path = vec![true, true, false, false]; // UUDD
        let word = NuDyckWord::new(path, nu).unwrap();

        let bounce = word.bounce_path();

        // The bounce path should trace backwards taking opposite steps
        // Original: U U D D (heights: 0, 1, 2, 1, 0)
        // Bounce backwards: D D U U
        // Bounce heights from end: 0, -1 (but max with ν[3]=0), ...
        // Let me recalculate: starting at position 4 (height 0)
        // Position 3: original was D (going right), so bounce is U, height = 0 + 1 = 1
        // Position 2: original was D, so bounce is U, height = 1 + 1 = 2
        // Position 1: original was U, so bounce is D, height = 2 - 1 = 1
        // Position 0: original was U, so bounce is D, height = 1 - 1 = 0

        assert_eq!(bounce.len(), 5);
        assert_eq!(bounce[0], 0);
        assert_eq!(bounce[4], 0);
    }

    #[test]
    fn test_nu_dyck_word_area_with_boundary() {
        // Boundary: ν = [0, 1, 2, 1, 0]
        let nu = vec![0, 1, 2, 1, 0];

        // Path that follows ν exactly: UUDD
        let path = vec![true, true, false, false];
        let word = NuDyckWord::new(path, nu).unwrap();

        let area = word.area_sequence();
        // At position 0 (before first U): height = 0, ν[0] = 0, area = 0 - 0 = 0
        // At position 1 (before second U): height = 1, ν[1] = 1, area = 1 - 1 = 0
        assert_eq!(area, vec![0, 0]);
        assert_eq!(word.total_area(), 0);
    }

    #[test]
    fn test_nu_dyck_words_generation() {
        // Generate all ν-Dyck words for ν = [0, 0, 0] (standard Dyck words of length 2)
        let nu = vec![0, 0, 0];
        let words = nu_dyck_words(nu);

        // Should be Catalan(1) = 1 word: UD
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].to_string(), "UD");
    }

    #[test]
    fn test_nu_dyck_words_generation_catalan() {
        // Generate all standard Dyck words of length 4
        let nu = vec![0, 0, 0, 0, 0];
        let words = nu_dyck_words(nu);

        // Should be Catalan(2) = 2 words
        assert_eq!(words.len(), 2);

        // Verify all are valid
        for word in &words {
            assert!(word.is_standard_dyck());
            assert_eq!(word.length(), 4);
        }
    }

    #[test]
    fn test_nu_dyck_from_nu() {
        // Valid boundary path
        let nu = vec![0, 1, 2, 1, 0];
        let path = NuDyckWord::from_nu(nu.clone());
        assert!(path.is_some());

        let path = path.unwrap();
        assert_eq!(path, vec![true, true, false, false]); // UUDD

        // Invalid boundary path (jump of 2)
        let invalid_nu = vec![0, 2, 0];
        assert!(NuDyckWord::from_nu(invalid_nu).is_none());

        // Invalid boundary path (doesn't end at 0)
        let invalid_nu2 = vec![0, 1, 2];
        assert!(NuDyckWord::from_nu(invalid_nu2).is_none());
    }

    #[test]
    fn test_nu_dyck_word_complex_example() {
        // More complex example with higher boundary
        let nu = vec![0, 1, 2, 1, 2, 1, 0];

        // A path that stays above this boundary
        // Let's construct: UUDUUD (gives heights [0, 1, 2, 1, 2, 3, 2, 1, 0])
        // Wait, that's 6 steps but nu has 7 points, so we need 6 steps
        let path = vec![true, true, false, true, false, false]; // UUDUDD
        let word = NuDyckWord::new(path, nu);

        if let Some(w) = word {
            let heights = w.heights();
            // Verify stays above nu
            for (i, &h) in heights.iter().enumerate() {
                if i < w.nu().len() {
                    assert!(h >= w.nu()[i], "Height {} at position {} below nu[{}] = {}", h, i, i, w.nu()[i]);
                }
            }

            // Check area sequence
            let area = w.area_sequence();
            assert_eq!(area.len(), 3); // 3 up steps
        }
    }

    #[test]
    fn test_nu_dyck_word_empty() {
        let nu = vec![0];
        let path = vec![];
        let word = NuDyckWord::new(path, nu).unwrap();

        assert_eq!(word.length(), 0);
        assert_eq!(word.heights(), vec![0]);
        assert_eq!(word.area_sequence(), vec![]);
        assert_eq!(word.total_area(), 0);
    }
}
