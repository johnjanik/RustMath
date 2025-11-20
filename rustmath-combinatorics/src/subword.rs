//! Subword operations and subword complexes
//!
//! This module provides:
//! - Subword order: a partial order on words where u ≤ v if u is a subword of v
//! - Subword generation and enumeration
//! - Knutson-Miller subword complexes for Schubert calculus
//! - Pipe dreams and related combinatorial structures
//!
//! # Subword Order
//!
//! A **subword** of a word w is obtained by deleting zero or more (possibly non-consecutive)
//! letters from w. For example, the subwords of "abc" include: ε, a, b, c, ab, ac, bc, abc.
//!
//! The **subword order** is a partial order where u ≤ v if u is a subword of v.
//!
//! # Subword Complexes
//!
//! A **subword complex** Δ(Q, W) is a simplicial complex defined for two words Q and W
//! (typically in a Coxeter group). It consists of subsets of positions in Q such that
//! the complement forms a reduced expression containing W as a subword.
//!
//! These complexes are fundamental in:
//! - Schubert calculus
//! - Coxeter group theory
//! - The study of Grassmannians and flag varieties
//!
//! # Example
//!
//! ```
//! use rustmath_combinatorics::subword::{is_subword, all_subwords};
//!
//! let word = vec![1, 2, 3];
//! let sub = vec![1, 3];
//!
//! assert!(is_subword(&sub, &word));
//!
//! let all = all_subwords(&word);
//! // Includes: [], [1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]
//! ```

use std::collections::HashSet;
use std::hash::Hash;

/// Check if `sub` is a subword of `word`
///
/// A subword is obtained by deleting zero or more (not necessarily consecutive) letters.
///
/// # Example
///
/// ```
/// use rustmath_combinatorics::subword::is_subword;
///
/// assert!(is_subword(&[1, 3], &[1, 2, 3]));
/// assert!(is_subword(&[1, 2], &[1, 2, 3]));
/// assert!(!is_subword(&[2, 1], &[1, 2, 3])); // Order matters!
/// ```
pub fn is_subword<T: PartialEq>(sub: &[T], word: &[T]) -> bool {
    if sub.is_empty() {
        return true;
    }
    if sub.len() > word.len() {
        return false;
    }

    let mut sub_idx = 0;
    for item in word {
        if sub_idx < sub.len() && item == &sub[sub_idx] {
            sub_idx += 1;
        }
        if sub_idx == sub.len() {
            return true;
        }
    }

    sub_idx == sub.len()
}

/// Find all positions where sub appears as a subword in word
///
/// Returns a vector of position sets, where each set represents positions in `word`
/// that form the subword `sub`.
///
/// # Example
///
/// ```
/// use rustmath_combinatorics::subword::subword_positions;
///
/// let word = vec![1, 2, 1, 3];
/// let sub = vec![1, 3];
/// let positions = subword_positions(&sub, &word);
///
/// // Returns [[0, 3]] meaning positions 0 and 3 form the subword [1, 3]
/// assert_eq!(positions.len(), 2); // Can use position 0 or 2 for the first 1
/// ```
pub fn subword_positions<T: PartialEq>(sub: &[T], word: &[T]) -> Vec<Vec<usize>> {
    if sub.is_empty() {
        return vec![vec![]];
    }
    if sub.len() > word.len() {
        return vec![];
    }

    let mut results = Vec::new();
    find_subword_positions(sub, word, 0, 0, &mut vec![], &mut results);
    results
}

/// Helper function for finding subword positions recursively
fn find_subword_positions<T: PartialEq>(
    sub: &[T],
    word: &[T],
    sub_idx: usize,
    word_idx: usize,
    current: &mut Vec<usize>,
    results: &mut Vec<Vec<usize>>,
) {
    if sub_idx == sub.len() {
        results.push(current.clone());
        return;
    }
    if word_idx >= word.len() {
        return;
    }
    if word.len() - word_idx < sub.len() - sub_idx {
        return; // Not enough letters remaining
    }

    // Try matching current position
    if word[word_idx] == sub[sub_idx] {
        current.push(word_idx);
        find_subword_positions(sub, word, sub_idx + 1, word_idx + 1, current, results);
        current.pop();
    }

    // Try skipping current position
    find_subword_positions(sub, word, sub_idx, word_idx + 1, current, results);
}

/// Generate all subwords of a word
///
/// Returns all subwords in lexicographic order (by positions, not values).
///
/// # Example
///
/// ```
/// use rustmath_combinatorics::subword::all_subwords;
///
/// let word = vec![1, 2, 3];
/// let subs = all_subwords(&word);
///
/// assert_eq!(subs.len(), 8); // 2^3 subwords including empty word
/// ```
pub fn all_subwords<T: Clone>(word: &[T]) -> Vec<Vec<T>> {
    let n = word.len();
    let mut result = Vec::with_capacity(1 << n);

    // Generate all 2^n subsets
    for mask in 0..(1 << n) {
        let mut subword = Vec::new();
        for (i, item) in word.iter().enumerate() {
            if (mask & (1 << i)) != 0 {
                subword.push(item.clone());
            }
        }
        result.push(subword);
    }

    result
}

/// Count the number of distinct subwords in a word
///
/// This counts the number of distinct subwords (by value, not by position).
///
/// # Example
///
/// ```
/// use rustmath_combinatorics::subword::count_distinct_subwords;
///
/// let word = vec![1, 1, 2];
/// let count = count_distinct_subwords(&word);
///
/// // Subwords: [], [1], [2], [1,1], [1,2], [1,1,2]
/// assert_eq!(count, 6);
/// ```
pub fn count_distinct_subwords<T: Clone + Eq + Hash>(word: &[T]) -> usize {
    let all = all_subwords(word);
    let unique: HashSet<Vec<T>> = all.into_iter().collect();
    unique.len()
}

/// Subword order relation: check if u ≤ v in the subword order
///
/// This is an alias for `is_subword` that emphasizes the order-theoretic interpretation.
pub fn subword_order<T: PartialEq>(u: &[T], v: &[T]) -> bool {
    is_subword(u, v)
}

/// A subword complex Δ(Q, W) for words Q and W
///
/// In the context of Coxeter groups, Q is typically a word in the generators,
/// and W is a reduced word for a Weyl group element. The complex consists of
/// subsets I ⊆ {1,...,|Q|} such that the complement Q \ I contains W as a subword.
///
/// This is also known as a Knutson-Miller subword complex.
#[derive(Debug, Clone)]
pub struct SubwordComplex<T: Clone + PartialEq> {
    /// The ambient word Q
    q: Vec<T>,
    /// The target word W
    w: Vec<T>,
    /// The facets (maximal faces) of the complex
    facets: Vec<Vec<usize>>,
    /// All faces of the complex
    faces: HashSet<Vec<usize>>,
}

impl<T: Clone + PartialEq + Eq + Hash> SubwordComplex<T> {
    /// Create a new subword complex Δ(Q, W)
    ///
    /// # Arguments
    ///
    /// * `q` - The ambient word Q
    /// * `w` - The target word W
    ///
    /// # Example
    ///
    /// ```
    /// use rustmath_combinatorics::subword::SubwordComplex;
    ///
    /// let q = vec![1, 2, 1, 3, 2];
    /// let w = vec![1, 2, 3];
    ///
    /// let complex = SubwordComplex::new(q, w);
    /// ```
    pub fn new(q: Vec<T>, w: Vec<T>) -> Self {
        let mut complex = SubwordComplex {
            q: q.clone(),
            w: w.clone(),
            facets: Vec::new(),
            faces: HashSet::new(),
        };

        complex.compute_faces();
        complex
    }

    /// Compute all faces of the subword complex
    fn compute_faces(&mut self) {
        let n = self.q.len();

        // A subset I of positions is a face if Q \ I contains W as a subword
        for mask in 0..(1 << n) {
            let mut removed_positions: Vec<usize> = Vec::new();
            let mut complement: Vec<T> = Vec::new();

            for i in 0..n {
                if (mask & (1 << i)) != 0 {
                    removed_positions.push(i);
                } else {
                    complement.push(self.q[i].clone());
                }
            }

            // Check if complement contains w as a subword
            if is_subword(&self.w, &complement) {
                removed_positions.sort_unstable();
                self.faces.insert(removed_positions);
            }
        }

        // Find facets (maximal faces)
        self.compute_facets();
    }

    /// Compute the facets (maximal faces) of the complex
    fn compute_facets(&mut self) {
        let faces_vec: Vec<Vec<usize>> = self.faces.iter().cloned().collect();

        for face in &faces_vec {
            let mut is_maximal = true;

            for other_face in &faces_vec {
                if other_face.len() > face.len() && is_subset(face, other_face) {
                    is_maximal = false;
                    break;
                }
            }

            if is_maximal {
                self.facets.push(face.clone());
            }
        }
    }

    /// Get the ambient word Q
    pub fn q(&self) -> &[T] {
        &self.q
    }

    /// Get the target word W
    pub fn w(&self) -> &[T] {
        &self.w
    }

    /// Get all faces of the complex
    pub fn faces(&self) -> Vec<Vec<usize>> {
        self.faces.iter().cloned().collect()
    }

    /// Get the facets (maximal faces) of the complex
    pub fn facets(&self) -> &[Vec<usize>] {
        &self.facets
    }

    /// Get the dimension of the complex
    ///
    /// The dimension is the maximum size of any facet minus 1.
    pub fn dimension(&self) -> Option<usize> {
        self.facets.iter().map(|f| f.len()).max().map(|d| d.saturating_sub(1))
    }

    /// Check if the complex is pure (all facets have the same dimension)
    pub fn is_pure(&self) -> bool {
        if self.facets.is_empty() {
            return true;
        }

        let first_dim = self.facets[0].len();
        self.facets.iter().all(|f| f.len() == first_dim)
    }

    /// Get the f-vector (number of faces in each dimension)
    ///
    /// Returns a vector where entry i is the number of i-dimensional faces.
    pub fn f_vector(&self) -> Vec<usize> {
        if self.faces.is_empty() {
            return vec![];
        }

        let max_dim = self.faces.iter().map(|f| f.len()).max().unwrap();
        let mut f_vec = vec![0; max_dim + 1];

        for face in &self.faces {
            f_vec[face.len()] += 1;
        }

        f_vec
    }

    /// Check if a set of positions is a face of the complex
    pub fn is_face(&self, positions: &[usize]) -> bool {
        let mut sorted = positions.to_vec();
        sorted.sort_unstable();
        self.faces.contains(&sorted)
    }

    /// Compute the h-vector from the f-vector
    ///
    /// The h-vector is a refined invariant of the complex related to the f-vector
    /// by the formula: f(x) = h(x-1) * (1 + x)^{d+1} where d is the dimension.
    pub fn h_vector(&self) -> Vec<i64> {
        let f = self.f_vector();
        if f.is_empty() {
            return vec![];
        }

        let d = f.len() - 1;
        let mut h = vec![0i64; d + 1];

        // Use the formula: h_i = Σ_{j=0}^{i} (-1)^{i-j} * C(d+1-j, i-j) * f_{j-1}
        // where f_{-1} = 1 by convention
        for i in 0..=d {
            let mut sum = 0i64;
            for j in 0..=i {
                let f_val = if j == 0 { 1 } else { f[j - 1] as i64 };
                let binom = binomial_coefficient(d + 1 - j, i - j);
                let sign = if (i - j) % 2 == 0 { 1 } else { -1 };
                sum += sign * binom * f_val;
            }
            h[i] = sum;
        }

        h
    }
}

/// Helper function to check if a is a subset of b
fn is_subset(a: &[usize], b: &[usize]) -> bool {
    a.iter().all(|x| b.contains(x))
}

/// Compute binomial coefficient C(n, k)
fn binomial_coefficient(n: usize, k: usize) -> i64 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }

    let k = k.min(n - k);
    let mut result = 1i64;

    for i in 0..k {
        result = result * (n - i) as i64 / (i + 1) as i64;
    }

    result
}

/// A pipe dream (also called an RC-graph)
///
/// Pipe dreams are combinatorial objects related to Schubert polynomials.
/// They are represented as n×n grids with pipes that connect entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipeDream {
    /// The grid representing the pipe dream
    /// true = elbow pipe, false = crossing pipe
    grid: Vec<Vec<bool>>,
    /// The size of the grid
    n: usize,
}

impl PipeDream {
    /// Create a new pipe dream from a grid
    ///
    /// # Arguments
    ///
    /// * `grid` - An n×n boolean grid where true represents an elbow and false a crossing
    ///
    /// # Example
    ///
    /// ```
    /// use rustmath_combinatorics::subword::PipeDream;
    ///
    /// let grid = vec![
    ///     vec![true, false, false],
    ///     vec![false, true, false],
    ///     vec![false, false, true],
    /// ];
    ///
    /// let pd = PipeDream::new(grid);
    /// ```
    pub fn new(grid: Vec<Vec<bool>>) -> Option<Self> {
        if grid.is_empty() {
            return None;
        }

        let n = grid.len();

        // Check that grid is square
        for row in &grid {
            if row.len() != n {
                return None;
            }
        }

        Some(PipeDream { grid, n })
    }

    /// Get the size of the pipe dream
    pub fn size(&self) -> usize {
        self.n
    }

    /// Get the grid
    pub fn grid(&self) -> &[Vec<bool>] {
        &self.grid
    }

    /// Check if position (i, j) is an elbow
    pub fn is_elbow(&self, i: usize, j: usize) -> Option<bool> {
        if i >= self.n || j >= self.n {
            return None;
        }
        Some(self.grid[i][j])
    }

    /// Convert pipe dream to a permutation
    ///
    /// The permutation is read off by following the pipes from top to bottom.
    pub fn to_permutation(&self) -> Vec<usize> {
        let mut perm = vec![0; self.n];

        for start_col in 0..self.n {
            let mut col = start_col;
            let mut row = 0;

            while row < self.n {
                if self.grid[row][col] {
                    // Elbow: move down
                    row += 1;
                } else {
                    // Crossing: move right
                    col += 1;
                    if col >= self.n {
                        col = self.n - 1;
                        break;
                    }
                }
            }

            perm[start_col] = col;
        }

        perm
    }

    /// Compute the rank of the pipe dream
    ///
    /// The rank is the number of crossings (false entries) in the grid.
    pub fn rank(&self) -> usize {
        self.grid
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&x| !x)
            .count()
    }

    /// Check if this is a reduced pipe dream
    ///
    /// A reduced pipe dream has no 2×2 block of elbows.
    pub fn is_reduced(&self) -> bool {
        for i in 0..self.n.saturating_sub(1) {
            for j in 0..self.n.saturating_sub(1) {
                if self.grid[i][j]
                    && self.grid[i][j + 1]
                    && self.grid[i + 1][j]
                    && self.grid[i + 1][j + 1]
                {
                    return false;
                }
            }
        }
        true
    }
}

/// Generate all reduced pipe dreams of size n
///
/// This generates pipe dreams corresponding to permutations in S_n.
pub fn reduced_pipe_dreams(n: usize) -> Vec<PipeDream> {
    if n == 0 {
        return vec![];
    }

    let mut result = Vec::new();
    generate_reduced_pipe_dreams(n, vec![vec![false; n]; n], 0, 0, &mut result);
    result
}

/// Helper function to generate reduced pipe dreams recursively
fn generate_reduced_pipe_dreams(
    n: usize,
    grid: Vec<Vec<bool>>,
    row: usize,
    col: usize,
    result: &mut Vec<PipeDream>,
) {
    if row == n {
        if let Some(pd) = PipeDream::new(grid.clone()) {
            if pd.is_reduced() {
                result.push(pd);
            }
        }
        return;
    }

    let (next_row, next_col) = if col + 1 < n {
        (row, col + 1)
    } else {
        (row + 1, 0)
    };

    // Try both elbow and crossing
    for &choice in &[false, true] {
        let mut new_grid = grid.clone();
        new_grid[row][col] = choice;
        generate_reduced_pipe_dreams(n, new_grid, next_row, next_col, result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_subword() {
        assert!(is_subword(&[1, 3], &[1, 2, 3]));
        assert!(is_subword(&[1, 2], &[1, 2, 3]));
        assert!(is_subword(&[2, 3], &[1, 2, 3]));
        assert!(is_subword(&[], &[1, 2, 3]));
        assert!(is_subword(&[1, 2, 3], &[1, 2, 3]));

        assert!(!is_subword(&[2, 1], &[1, 2, 3]));
        assert!(!is_subword(&[3, 2], &[1, 2, 3]));
        assert!(!is_subword(&[1, 2, 3, 4], &[1, 2, 3]));
    }

    #[test]
    fn test_subword_positions() {
        let word = vec![1, 2, 1, 3];
        let sub = vec![1, 3];
        let positions = subword_positions(&sub, &word);

        assert!(positions.len() >= 2);
        assert!(positions.contains(&vec![0, 3]));
        assert!(positions.contains(&vec![2, 3]));
    }

    #[test]
    fn test_all_subwords() {
        let word = vec![1, 2, 3];
        let subs = all_subwords(&word);

        assert_eq!(subs.len(), 8); // 2^3

        // Check empty subword
        assert!(subs.contains(&vec![]));

        // Check single elements
        assert!(subs.contains(&vec![1]));
        assert!(subs.contains(&vec![2]));
        assert!(subs.contains(&vec![3]));

        // Check pairs
        assert!(subs.contains(&vec![1, 2]));
        assert!(subs.contains(&vec![1, 3]));
        assert!(subs.contains(&vec![2, 3]));

        // Check full word
        assert!(subs.contains(&vec![1, 2, 3]));
    }

    #[test]
    fn test_count_distinct_subwords() {
        let word = vec![1, 1, 2];
        let count = count_distinct_subwords(&word);

        // Subwords: [], [1], [2], [1,1], [1,2], [1,1,2]
        assert_eq!(count, 6);
    }

    #[test]
    fn test_subword_order() {
        assert!(subword_order(&[1, 3], &[1, 2, 3]));
        assert!(!subword_order(&[3, 1], &[1, 2, 3]));
    }

    #[test]
    fn test_subword_complex_basic() {
        let q = vec![1, 2, 1, 3];
        let w = vec![1, 2];

        let complex = SubwordComplex::new(q, w);

        // Check that complex was created
        assert!(!complex.faces().is_empty());

        // Check that facets exist
        assert!(!complex.facets().is_empty());
    }

    #[test]
    fn test_subword_complex_dimension() {
        let q = vec![1, 2, 3];
        let w = vec![1];

        let complex = SubwordComplex::new(q, w);

        // Should have some dimension
        assert!(complex.dimension().is_some());
    }

    #[test]
    fn test_subword_complex_f_vector() {
        let q = vec![1, 2, 1];
        let w = vec![1, 2];

        let complex = SubwordComplex::new(q, w);
        let f_vec = complex.f_vector();

        // f-vector should be non-empty
        assert!(!f_vec.is_empty());

        // First entry (vertices) should be non-zero
        assert!(f_vec.iter().sum::<usize>() > 0);
    }

    #[test]
    fn test_pipe_dream_creation() {
        let grid = vec![
            vec![true, false, false],
            vec![false, true, false],
            vec![false, false, true],
        ];

        let pd = PipeDream::new(grid);
        assert!(pd.is_some());

        let pd = pd.unwrap();
        assert_eq!(pd.size(), 3);
    }

    #[test]
    fn test_pipe_dream_is_elbow() {
        let grid = vec![
            vec![true, false],
            vec![false, true],
        ];

        let pd = PipeDream::new(grid).unwrap();

        assert_eq!(pd.is_elbow(0, 0), Some(true));
        assert_eq!(pd.is_elbow(0, 1), Some(false));
        assert_eq!(pd.is_elbow(1, 0), Some(false));
        assert_eq!(pd.is_elbow(1, 1), Some(true));
    }

    #[test]
    fn test_pipe_dream_rank() {
        let grid = vec![
            vec![true, false, false],
            vec![false, true, false],
            vec![false, false, true],
        ];

        let pd = PipeDream::new(grid).unwrap();

        // Count false entries (crossings)
        assert_eq!(pd.rank(), 6);
    }

    #[test]
    fn test_pipe_dream_is_reduced() {
        // Reduced pipe dream (no 2×2 block of elbows)
        let grid1 = vec![
            vec![true, false],
            vec![false, true],
        ];
        let pd1 = PipeDream::new(grid1).unwrap();
        assert!(pd1.is_reduced());

        // Non-reduced pipe dream (has 2×2 block of elbows)
        let grid2 = vec![
            vec![true, true],
            vec![true, true],
        ];
        let pd2 = PipeDream::new(grid2).unwrap();
        assert!(!pd2.is_reduced());
    }

    #[test]
    fn test_pipe_dream_to_permutation() {
        let grid = vec![
            vec![true, false, false],
            vec![false, true, false],
            vec![false, false, true],
        ];

        let pd = PipeDream::new(grid).unwrap();
        let perm = pd.to_permutation();

        // Should produce some permutation
        assert_eq!(perm.len(), 3);
    }

    #[test]
    fn test_empty_subword() {
        assert!(is_subword(&[], &[1, 2, 3]));
        assert!(is_subword::<i32>(&[], &[]));
    }

    #[test]
    fn test_identical_subword() {
        assert!(is_subword(&[1, 2, 3], &[1, 2, 3]));
    }

    #[test]
    fn test_subword_positions_empty() {
        let positions = subword_positions(&[], &[1, 2, 3]);
        assert_eq!(positions, vec![vec![]]);
    }

    #[test]
    fn test_binomial_coefficient() {
        assert_eq!(binomial_coefficient(5, 0), 1);
        assert_eq!(binomial_coefficient(5, 1), 5);
        assert_eq!(binomial_coefficient(5, 2), 10);
        assert_eq!(binomial_coefficient(5, 3), 10);
        assert_eq!(binomial_coefficient(5, 4), 5);
        assert_eq!(binomial_coefficient(5, 5), 1);
        assert_eq!(binomial_coefficient(5, 6), 0);
    }

    #[test]
    fn test_subword_complex_is_face() {
        let q = vec![1, 2, 1, 3];
        let w = vec![1, 2];

        let complex = SubwordComplex::new(q, w);

        // Empty set should be a face (if the complex is non-empty)
        if !complex.faces().is_empty() {
            assert!(complex.is_face(&[]));
        }
    }

    #[test]
    fn test_subword_complex_pure() {
        let q = vec![1, 2];
        let w = vec![1];

        let complex = SubwordComplex::new(q, w);

        // Check if complex is pure (may or may not be, depending on Q and W)
        // Just ensure the method doesn't panic
        let _ = complex.is_pure();
    }

    #[test]
    fn test_reduced_pipe_dreams_small() {
        // For n=1, should have exactly one pipe dream
        let pds = reduced_pipe_dreams(1);
        assert!(!pds.is_empty());

        // All should be reduced
        for pd in &pds {
            assert!(pd.is_reduced());
        }
    }
}
