//! Parking functions and non-decreasing parking functions
//!
//! A parking function is a sequence (a₁, a₂, ..., aₙ) of positive integers such that
//! when sorted in non-decreasing order to get (b₁, b₂, ..., bₙ), we have bᵢ ≤ i for all i.
//!
//! Parking functions are named after a parking scenario: n cars arrive at a street with
//! parking spaces numbered 1, 2, ..., n. Car i has a preferred parking space aᵢ. Each car
//! drives to its preferred space, and if that space is taken, it takes the next available
//! space. A parking function is a sequence where all cars can park.
//!
//! This module provides:
//! - General parking functions
//! - Non-decreasing parking functions (already sorted)
//! - Area statistic computation
//! - Generation and enumeration algorithms

/// A parking function - a sequence of positive integers satisfying the parking condition
///
/// A sequence (a₁, a₂, ..., aₙ) is a parking function if, when sorted to get (b₁, b₂, ..., bₙ),
/// we have bᵢ ≤ i for all i = 1, 2, ..., n.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParkingFunction {
    /// The parking preferences (1-indexed)
    preferences: Vec<usize>,
}

impl ParkingFunction {
    /// Create a parking function from a vector of preferences
    ///
    /// Returns None if the sequence is not a valid parking function
    /// (i.e., if any preference is 0 or if the parking condition is violated)
    pub fn new(preferences: Vec<usize>) -> Option<Self> {
        if preferences.is_empty() {
            return Some(ParkingFunction { preferences });
        }

        // Check that all preferences are positive (1-indexed)
        if preferences.iter().any(|&p| p == 0) {
            return None;
        }

        // Check the parking function condition
        let mut sorted = preferences.clone();
        sorted.sort_unstable();

        for (i, &pref) in sorted.iter().enumerate() {
            if pref > i + 1 {
                return None; // Parking condition violated
            }
        }

        Some(ParkingFunction { preferences })
    }

    /// Get the length of the parking function
    pub fn length(&self) -> usize {
        self.preferences.len()
    }

    /// Get the preferences
    pub fn preferences(&self) -> &[usize] {
        &self.preferences
    }

    /// Get the sorted preferences
    pub fn sorted_preferences(&self) -> Vec<usize> {
        let mut sorted = self.preferences.clone();
        sorted.sort_unstable();
        sorted
    }

    /// Check if this is a non-decreasing parking function
    pub fn is_non_decreasing(&self) -> bool {
        self.preferences.windows(2).all(|w| w[0] <= w[1])
    }

    /// Convert to a non-decreasing parking function if possible
    ///
    /// This simply sorts the preferences. The result is always a valid non-decreasing
    /// parking function if the original was a valid parking function.
    pub fn to_non_decreasing(&self) -> NonDecreasingParkingFunction {
        NonDecreasingParkingFunction {
            preferences: self.sorted_preferences(),
        }
    }

    /// Compute the area statistic of this parking function
    ///
    /// The area is defined as: area = Σᵢ₌₁ⁿ (aᵢ - 1) = Σᵢ₌₁ⁿ aᵢ - n
    ///
    /// This represents the total "displacement" from ideal parking (where car i parks in space i).
    pub fn area(&self) -> usize {
        self.preferences.iter().sum::<usize>().saturating_sub(self.length())
    }

    /// Compute the major index (sum of descents)
    ///
    /// The major index is the sum of positions i where aᵢ > aᵢ₊₁
    pub fn major_index(&self) -> usize {
        self.preferences
            .windows(2)
            .enumerate()
            .filter(|(_, w)| w[0] > w[1])
            .map(|(i, _)| i + 1)
            .sum()
    }
}

/// A non-decreasing parking function
///
/// This is a parking function where the preferences are already in non-decreasing order:
/// a₁ ≤ a₂ ≤ ... ≤ aₙ
///
/// Non-decreasing parking functions are in bijection with Dyck paths and have many
/// interesting properties in algebraic combinatorics.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NonDecreasingParkingFunction {
    /// The parking preferences in non-decreasing order (1-indexed)
    preferences: Vec<usize>,
}

impl NonDecreasingParkingFunction {
    /// Create a non-decreasing parking function from a vector of preferences
    ///
    /// Returns None if:
    /// - The sequence is not in non-decreasing order
    /// - Any preference is 0
    /// - The parking condition is violated
    pub fn new(preferences: Vec<usize>) -> Option<Self> {
        if preferences.is_empty() {
            return Some(NonDecreasingParkingFunction { preferences });
        }

        // Check that all preferences are positive
        if preferences.iter().any(|&p| p == 0) {
            return None;
        }

        // Check non-decreasing order
        if !preferences.windows(2).all(|w| w[0] <= w[1]) {
            return None;
        }

        // Check parking function condition (already sorted)
        for (i, &pref) in preferences.iter().enumerate() {
            if pref > i + 1 {
                return None;
            }
        }

        Some(NonDecreasingParkingFunction { preferences })
    }

    /// Get the length of the parking function
    pub fn length(&self) -> usize {
        self.preferences.len()
    }

    /// Get the preferences
    pub fn preferences(&self) -> &[usize] {
        &self.preferences
    }

    /// Compute the area statistic of this parking function
    ///
    /// The area is defined as: area = Σᵢ₌₁ⁿ (aᵢ - 1) = Σᵢ₌₁ⁿ aᵢ - n
    ///
    /// For non-decreasing parking functions, the area has a special interpretation
    /// related to the corresponding Dyck path.
    pub fn area(&self) -> usize {
        self.preferences.iter().sum::<usize>().saturating_sub(self.length())
    }

    /// Compute the diagonal inversion number
    ///
    /// This is the number of positions i where aᵢ = i
    pub fn diagonal_inversions(&self) -> usize {
        self.preferences
            .iter()
            .enumerate()
            .filter(|(i, &a)| a == i + 1)
            .count()
    }

    /// Convert to a general parking function
    pub fn to_parking_function(&self) -> ParkingFunction {
        ParkingFunction {
            preferences: self.preferences.clone(),
        }
    }

    /// Compute the bounce path
    ///
    /// The bounce path is a sequence of heights that corresponds to the parking process.
    /// For a non-decreasing parking function, this can be computed efficiently.
    pub fn bounce_path(&self) -> Vec<usize> {
        let n = self.length();
        let mut path = Vec::with_capacity(n + 1);
        path.push(0);

        let mut height = 0;
        for (i, &pref) in self.preferences.iter().enumerate() {
            // The bounce path increases by 1 when we can park, and stays level otherwise
            if pref <= i + 1 {
                height += 1;
            }
            path.push(height);
        }

        path
    }
}

/// Generate all parking functions of length n
///
/// The number of parking functions of length n is (n+1)^(n-1).
pub fn parking_functions(n: usize) -> Vec<ParkingFunction> {
    if n == 0 {
        return vec![ParkingFunction {
            preferences: vec![],
        }];
    }

    let mut result = Vec::new();
    let mut current = Vec::new();

    generate_parking_functions(n, &mut current, &mut result);

    result
}

fn generate_parking_functions(
    n: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<ParkingFunction>,
) {
    if current.len() == n {
        // Check if this is a valid parking function
        if let Some(pf) = ParkingFunction::new(current.clone()) {
            result.push(pf);
        }
        return;
    }

    // Try all possible preferences from 1 to n
    for pref in 1..=n {
        current.push(pref);
        generate_parking_functions(n, current, result);
        current.pop();
    }
}

/// Generate all non-decreasing parking functions of length n
///
/// Non-decreasing parking functions of length n are in bijection with Dyck paths
/// of length n, so there are C_n of them (the nth Catalan number).
pub fn non_decreasing_parking_functions(n: usize) -> Vec<NonDecreasingParkingFunction> {
    if n == 0 {
        return vec![NonDecreasingParkingFunction {
            preferences: vec![],
        }];
    }

    let mut result = Vec::new();
    let mut current = Vec::new();

    generate_non_decreasing_parking_functions(n, 1, &mut current, &mut result);

    result
}

fn generate_non_decreasing_parking_functions(
    n: usize,
    min_value: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<NonDecreasingParkingFunction>,
) {
    if current.len() == n {
        // Verify the parking condition (should always be satisfied by construction)
        if let Some(pf) = NonDecreasingParkingFunction::new(current.clone()) {
            result.push(pf);
        }
        return;
    }

    let pos = current.len() + 1; // 1-indexed position
    let max_value = pos; // Parking condition: a_i <= i

    // Generate in non-decreasing order
    for pref in min_value..=max_value {
        current.push(pref);
        generate_non_decreasing_parking_functions(n, pref, current, result);
        current.pop();
    }
}

/// Generate all non-decreasing parking functions of length n with a given area
pub fn non_decreasing_parking_functions_with_area(
    n: usize,
    area: usize,
) -> Vec<NonDecreasingParkingFunction> {
    non_decreasing_parking_functions(n)
        .into_iter()
        .filter(|pf| pf.area() == area)
        .collect()
}

/// Count parking functions of length n
///
/// The number of parking functions of length n is (n+1)^(n-1)
pub fn count_parking_functions(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    (n + 1).pow((n - 1) as u32)
}

/// Count non-decreasing parking functions of length n
///
/// This is the nth Catalan number: C_n = (1/(n+1)) * C(2n, n)
pub fn count_non_decreasing_parking_functions(n: usize) -> usize {
    catalan_number(n)
}

/// Compute the nth Catalan number
fn catalan_number(n: usize) -> usize {
    if n == 0 || n == 1 {
        return 1;
    }

    // Use the recurrence: C_n = sum_{i=0}^{n-1} C_i * C_{n-1-i}
    // Or use the formula: C_n = (2n)! / ((n+1)! * n!)
    // We'll use dynamic programming for efficiency

    let mut catalan = vec![0; n + 1];
    catalan[0] = 1;
    catalan[1] = 1;

    for i in 2..=n {
        for j in 0..i {
            catalan[i] += catalan[j] * catalan[i - 1 - j];
        }
    }

    catalan[n]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parking_function_creation() {
        // Valid parking function
        let pf = ParkingFunction::new(vec![1, 1, 2]);
        assert!(pf.is_some());

        // Invalid: contains 0
        let pf = ParkingFunction::new(vec![0, 1, 2]);
        assert!(pf.is_none());

        // Invalid: parking condition violated
        let pf = ParkingFunction::new(vec![3, 3, 3]);
        assert!(pf.is_none());

        // Valid: another example
        let pf = ParkingFunction::new(vec![2, 1, 1]);
        assert!(pf.is_some());
    }

    #[test]
    fn test_non_decreasing_parking_function_creation() {
        // Valid non-decreasing parking function
        let ndpf = NonDecreasingParkingFunction::new(vec![1, 1, 2]);
        assert!(ndpf.is_some());

        // Invalid: not non-decreasing
        let ndpf = NonDecreasingParkingFunction::new(vec![2, 1, 2]);
        assert!(ndpf.is_none());

        // Invalid: parking condition violated
        let ndpf = NonDecreasingParkingFunction::new(vec![1, 3, 3]);
        assert!(ndpf.is_none());

        // Valid: edge case
        let ndpf = NonDecreasingParkingFunction::new(vec![1, 2, 3]);
        assert!(ndpf.is_some());
    }

    #[test]
    fn test_area_statistic() {
        // [1, 1, 2] has area = (1 + 1 + 2) - 3 = 1
        let pf = ParkingFunction::new(vec![1, 1, 2]).unwrap();
        assert_eq!(pf.area(), 1);

        // [1, 2, 3] has area = (1 + 2 + 3) - 3 = 3
        let pf = ParkingFunction::new(vec![1, 2, 3]).unwrap();
        assert_eq!(pf.area(), 3);

        // [1, 1, 1] has area = (1 + 1 + 1) - 3 = 0
        let pf = ParkingFunction::new(vec![1, 1, 1]).unwrap();
        assert_eq!(pf.area(), 0);
    }

    #[test]
    fn test_non_decreasing_area_statistic() {
        let ndpf = NonDecreasingParkingFunction::new(vec![1, 1, 2]).unwrap();
        assert_eq!(ndpf.area(), 1);

        let ndpf = NonDecreasingParkingFunction::new(vec![1, 2, 3]).unwrap();
        assert_eq!(ndpf.area(), 3);

        let ndpf = NonDecreasingParkingFunction::new(vec![1, 1, 1]).unwrap();
        assert_eq!(ndpf.area(), 0);
    }

    #[test]
    fn test_is_non_decreasing() {
        let pf = ParkingFunction::new(vec![1, 1, 2]).unwrap();
        assert!(pf.is_non_decreasing());

        let pf = ParkingFunction::new(vec![2, 1, 1]).unwrap();
        assert!(!pf.is_non_decreasing());

        let pf = ParkingFunction::new(vec![1, 2, 3]).unwrap();
        assert!(pf.is_non_decreasing());
    }

    #[test]
    fn test_count_parking_functions() {
        // PF(0) = 1
        assert_eq!(count_parking_functions(0), 1);

        // PF(1) = (1+1)^(1-1) = 2^0 = 1
        assert_eq!(count_parking_functions(1), 1);

        // PF(2) = (2+1)^(2-1) = 3^1 = 3
        assert_eq!(count_parking_functions(2), 3);

        // PF(3) = (3+1)^(3-1) = 4^2 = 16
        assert_eq!(count_parking_functions(3), 16);

        // PF(4) = (4+1)^(4-1) = 5^3 = 125
        assert_eq!(count_parking_functions(4), 125);
    }

    #[test]
    fn test_count_non_decreasing_parking_functions() {
        // Catalan numbers: 1, 1, 2, 5, 14, 42, 132, ...
        assert_eq!(count_non_decreasing_parking_functions(0), 1);
        assert_eq!(count_non_decreasing_parking_functions(1), 1);
        assert_eq!(count_non_decreasing_parking_functions(2), 2);
        assert_eq!(count_non_decreasing_parking_functions(3), 5);
        assert_eq!(count_non_decreasing_parking_functions(4), 14);
        assert_eq!(count_non_decreasing_parking_functions(5), 42);
    }

    #[test]
    fn test_generate_parking_functions() {
        let pfs = parking_functions(2);
        assert_eq!(pfs.len(), 3);

        // The 3 parking functions of length 2 are: [1,1], [1,2], [2,1]
        let prefs: Vec<Vec<usize>> = pfs.iter().map(|pf| pf.preferences().to_vec()).collect();
        assert!(prefs.contains(&vec![1, 1]));
        assert!(prefs.contains(&vec![1, 2]));
        assert!(prefs.contains(&vec![2, 1]));
    }

    #[test]
    fn test_generate_non_decreasing_parking_functions() {
        let ndpfs = non_decreasing_parking_functions(3);
        assert_eq!(ndpfs.len(), 5); // C_3 = 5

        // Verify they are all non-decreasing
        for ndpf in &ndpfs {
            assert!(ndpf.preferences().windows(2).all(|w| w[0] <= w[1]));
        }

        // The 5 non-decreasing parking functions of length 3 are:
        // [1,1,1], [1,1,2], [1,1,3], [1,2,2], [1,2,3]
        let prefs: Vec<Vec<usize>> = ndpfs
            .iter()
            .map(|pf| pf.preferences().to_vec())
            .collect();
        assert!(prefs.contains(&vec![1, 1, 1]));
        assert!(prefs.contains(&vec![1, 1, 2]));
        assert!(prefs.contains(&vec![1, 1, 3]));
        assert!(prefs.contains(&vec![1, 2, 2]));
        assert!(prefs.contains(&vec![1, 2, 3]));
    }

    #[test]
    fn test_non_decreasing_parking_functions_with_area() {
        // For n=3, area can range from 0 to 3
        let ndpfs_area_0 = non_decreasing_parking_functions_with_area(3, 0);
        assert_eq!(ndpfs_area_0.len(), 1);
        assert_eq!(ndpfs_area_0[0].preferences(), &[1, 1, 1]);

        let ndpfs_area_1 = non_decreasing_parking_functions_with_area(3, 1);
        assert_eq!(ndpfs_area_1.len(), 1);
        assert_eq!(ndpfs_area_1[0].preferences(), &[1, 1, 2]);

        let ndpfs_area_2 = non_decreasing_parking_functions_with_area(3, 2);
        assert_eq!(ndpfs_area_2.len(), 2);
        // [1,1,3] and [1,2,2] both have area 2

        let ndpfs_area_3 = non_decreasing_parking_functions_with_area(3, 3);
        assert_eq!(ndpfs_area_3.len(), 1);
        assert_eq!(ndpfs_area_3[0].preferences(), &[1, 2, 3]);
    }

    #[test]
    fn test_bounce_path() {
        let ndpf = NonDecreasingParkingFunction::new(vec![1, 1, 2]).unwrap();
        let bounce = ndpf.bounce_path();
        assert_eq!(bounce, vec![0, 1, 2, 3]);

        let ndpf = NonDecreasingParkingFunction::new(vec![1, 2, 3]).unwrap();
        let bounce = ndpf.bounce_path();
        assert_eq!(bounce, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_diagonal_inversions() {
        let ndpf = NonDecreasingParkingFunction::new(vec![1, 2, 3]).unwrap();
        assert_eq!(ndpf.diagonal_inversions(), 3);

        let ndpf = NonDecreasingParkingFunction::new(vec![1, 1, 1]).unwrap();
        assert_eq!(ndpf.diagonal_inversions(), 1);

        let ndpf = NonDecreasingParkingFunction::new(vec![1, 1, 2]).unwrap();
        assert_eq!(ndpf.diagonal_inversions(), 1);
    }

    #[test]
    fn test_major_index() {
        let pf = ParkingFunction::new(vec![2, 1, 1]).unwrap();
        // Descents at position 1 (2 > 1)
        assert_eq!(pf.major_index(), 1);

        let pf = ParkingFunction::new(vec![1, 2, 1]).unwrap();
        // Descents at position 2 (2 > 1)
        assert_eq!(pf.major_index(), 2);

        let pf = ParkingFunction::new(vec![1, 1, 2]).unwrap();
        // No descents
        assert_eq!(pf.major_index(), 0);
    }

    #[test]
    fn test_conversion() {
        let pf = ParkingFunction::new(vec![2, 1, 1]).unwrap();
        let ndpf = pf.to_non_decreasing();
        assert_eq!(ndpf.preferences(), &[1, 1, 2]);

        let ndpf = NonDecreasingParkingFunction::new(vec![1, 1, 2]).unwrap();
        let pf = ndpf.to_parking_function();
        assert_eq!(pf.preferences(), &[1, 1, 2]);
    }
}
