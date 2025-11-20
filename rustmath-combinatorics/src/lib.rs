//! RustMath Combinatorics - Combinatorial objects and algorithms
//!
//! This crate provides combinatorial structures like permutations, combinations,
//! partitions, and algorithms for generating and manipulating them.

pub mod binary_words;
pub mod combinations;
pub mod enumeration;
pub mod partitions;
pub mod permutations;
pub mod posets;
pub mod ranking;
pub mod set_system;
pub mod tableaux;
pub mod word;

pub use combinations::{combinations, Combination};
pub use partitions::{
    count_partitions_with_max_parts, partition_count, partitions, partitions_with_distinct_parts,
    partitions_with_even_parts, partitions_with_exact_parts, partitions_with_max_part,
    partitions_with_max_parts, partitions_with_min_part, partitions_with_odd_parts, Partition,
    PartitionTuple,
};
pub use permutations::{all_permutations, Permutation};
pub use posets::Poset;
pub use tableaux::{robinson_schensted, rs_insert, standard_tableaux, Tableau};

// Re-export new modules
pub use binary_words::{all_binary_words, binary_words_with_weight, lyndon_words, necklaces, BinaryWord};
pub use enumeration::{
    cartesian_product, stars_and_bars, tuples, weak_compositions, CompositionIterator,
    Enumerable, GrayCodeIterator, LazyEnumerator, PartitionIterator, RevolvingDoorIterator,
};
pub use ranking::{CombinationRank, PermutationRank, Rankable, RankingTable};
pub use set_system::SetSystem;
pub use word::{
    abelian_complexity, boyer_moore_search, christoffel_word, factor_complexity, kmp_search,
    lyndon_factorization, lyndon_words as general_lyndon_words, sturmian_word, AutomaticSequence,
    Morphism, Word,
};

// stirling_first, Composition, compositions, and compositions_k are defined in this module

use rustmath_core::Ring;
use rustmath_integers::Integer;

/// Compute factorial
pub fn factorial(n: u32) -> Integer {
    let mut result = Integer::one();
    for i in 2..=n {
        result = result * Integer::from(i);
    }
    result
}

/// Compute binomial coefficient C(n, k) = n! / (k! * (n-k)!)
pub fn binomial(n: u32, k: u32) -> Integer {
    if k > n {
        return Integer::zero();
    }
    if k == 0 || k == n {
        return Integer::one();
    }

    let k = k.min(n - k); // Optimize using symmetry
    let mut result = Integer::one();

    for i in 0..k {
        result = result * Integer::from(n - i);
        result = result / Integer::from(i + 1);
    }

    result
}

/// Compute multinomial coefficient
///
/// multinomial(n, [k1, k2, ..., km]) = n! / (k1! * k2! * ... * km!)
/// where k1 + k2 + ... + km = n
pub fn multinomial(n: u32, ks: &[u32]) -> Integer {
    let sum: u32 = ks.iter().sum();
    if sum != n {
        return Integer::zero();
    }

    let mut result = factorial(n);
    for &k in ks {
        result = result / factorial(k);
    }
    result
}

/// Compute the nth Catalan number
///
/// C_n = (1/(n+1)) * C(2n, n) = (2n)! / ((n+1)! * n!)
pub fn catalan(n: u32) -> Integer {
    if n == 0 {
        return Integer::one();
    }

    binomial(2 * n, n) / Integer::from(n + 1)
}

/// Compute the nth Fibonacci number
pub fn fibonacci(n: u32) -> Integer {
    if n == 0 {
        return Integer::zero();
    }
    if n == 1 {
        return Integer::one();
    }

    let mut a = Integer::zero();
    let mut b = Integer::one();

    for _ in 2..=n {
        let temp = a.clone() + b.clone();
        a = b;
        b = temp;
    }

    b
}

/// Compute the nth Lucas number
///
/// L_0 = 2, L_1 = 1, L_n = L_{n-1} + L_{n-2}
pub fn lucas(n: u32) -> Integer {
    if n == 0 {
        return Integer::from(2);
    }
    if n == 1 {
        return Integer::one();
    }

    let mut a = Integer::from(2);
    let mut b = Integer::one();

    for _ in 2..=n {
        let temp = a.clone() + b.clone();
        a = b;
        b = temp;
    }

    b
}

/// Compute the falling factorial (Pochhammer symbol)
///
/// (n)_k = n * (n-1) * (n-2) * ... * (n-k+1)
pub fn falling_factorial(n: u32, k: u32) -> Integer {
    if k > n {
        return Integer::zero();
    }

    let mut result = Integer::one();
    for i in 0..k {
        result = result * Integer::from(n - i);
    }
    result
}

/// Compute the rising factorial
///
/// n^(k) = n * (n+1) * (n+2) * ... * (n+k-1)
pub fn rising_factorial(n: u32, k: u32) -> Integer {
    let mut result = Integer::one();
    for i in 0..k {
        result = result * Integer::from(n + i);
    }
    result
}

/// Compute Stirling number of the first kind s(n, k) (unsigned)
///
/// Number of permutations of n elements with exactly k cycles
pub fn stirling_first(n: u32, k: u32) -> Integer {
    if n == 0 && k == 0 {
        return Integer::one();
    }
    if n == 0 || k == 0 || k > n {
        return Integer::zero();
    }
    if k == n {
        return Integer::one();
    }

    // Use recurrence: s(n,k) = (n-1)*s(n-1,k) + s(n-1,k-1)
    let mut dp = vec![vec![Integer::zero(); (k + 1) as usize]; (n + 1) as usize];
    dp[0][0] = Integer::one();

    for i in 1..=n as usize {
        for j in 1..=k.min(i as u32) as usize {
            if j == i {
                dp[i][j] = Integer::one();
            } else if j > 0 {
                dp[i][j] = Integer::from((i - 1) as u32) * dp[i - 1][j].clone()
                    + dp[i - 1][j - 1].clone();
            }
        }
    }

    dp[n as usize][k as usize].clone()
}

/// Compute Stirling number of the second kind S(n, k)
///
/// Number of ways to partition n elements into k non-empty subsets
pub fn stirling_second(n: u32, k: u32) -> Integer {
    if n == 0 && k == 0 {
        return Integer::one();
    }
    if n == 0 || k == 0 || k > n {
        return Integer::zero();
    }
    if k == 1 || k == n {
        return Integer::one();
    }

    // Use recurrence: S(n,k) = k*S(n-1,k) + S(n-1,k-1)
    let mut dp = vec![vec![Integer::zero(); (k + 1) as usize]; (n + 1) as usize];
    dp[0][0] = Integer::one();

    for i in 1..=n as usize {
        for j in 1..=k.min(i as u32) as usize {
            if j == 1 {
                dp[i][j] = Integer::one();
            } else if j == i {
                dp[i][j] = Integer::one();
            } else {
                dp[i][j] = Integer::from(j as u32) * dp[i - 1][j].clone() + dp[i - 1][j - 1].clone();
            }
        }
    }

    dp[n as usize][k as usize].clone()
}

/// Compute Bell number B(n)
///
/// Number of ways to partition n elements into any number of non-empty subsets
pub fn bell_number(n: u32) -> Integer {
    let mut sum = Integer::zero();
    for k in 0..=n {
        sum = sum + stirling_second(n, k);
    }
    sum
}

/// An integer composition (ordered partition)
///
/// A composition of n is an ordered sequence of positive integers that sum to n
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Composition {
    parts: Vec<usize>,
}

impl Composition {
    /// Create a composition from a vector of parts
    pub fn new(parts: Vec<usize>) -> Option<Self> {
        if parts.iter().any(|&p| p == 0) {
            return None; // All parts must be positive
        }
        Some(Composition { parts })
    }

    /// Get the sum of the composition
    pub fn sum(&self) -> usize {
        self.parts.iter().sum()
    }

    /// Get the number of parts
    pub fn length(&self) -> usize {
        self.parts.len()
    }

    /// Get the parts
    pub fn parts(&self) -> &[usize] {
        &self.parts
    }
}

/// Generate all compositions of n
///
/// A composition is an ordered way of writing n as a sum of positive integers
pub fn compositions(n: usize) -> Vec<Composition> {
    if n == 0 {
        return vec![Composition { parts: vec![] }];
    }

    let mut result = Vec::new();
    let mut current = Vec::new();

    generate_compositions(n, &mut current, &mut result);

    result
}

fn generate_compositions(n: usize, current: &mut Vec<usize>, result: &mut Vec<Composition>) {
    if n == 0 {
        result.push(Composition {
            parts: current.clone(),
        });
        return;
    }

    for i in 1..=n {
        current.push(i);
        generate_compositions(n - i, current, result);
        current.pop();
    }
}

/// Generate all compositions of n into exactly k parts
pub fn compositions_k(n: usize, k: usize) -> Vec<Composition> {
    if k == 0 {
        if n == 0 {
            return vec![Composition { parts: vec![] }];
        } else {
            return vec![];
        }
    }

    if k > n {
        return vec![];
    }

    let mut result = Vec::new();
    let mut current = Vec::new();

    generate_compositions_k(n, k, &mut current, &mut result);

    result
}

fn generate_compositions_k(
    n: usize,
    k: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Composition>,
) {
    if current.len() == k {
        if n == 0 {
            result.push(Composition {
                parts: current.clone(),
            });
        }
        return;
    }

    let remaining_parts = k - current.len();
    let min_value = 1;
    let max_value = n.saturating_sub(remaining_parts - 1);

    for i in min_value..=max_value {
        current.push(i);
        generate_compositions_k(n - i, k, current, result);
        current.pop();
    }
}

/// A set partition - a way of partitioning a set into non-empty disjoint subsets
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetPartition {
    /// Each vector represents one block (subset) in the partition
    blocks: Vec<Vec<usize>>,
    /// Total number of elements
    n: usize,
}

impl SetPartition {
    /// Create a set partition from blocks
    pub fn new(blocks: Vec<Vec<usize>>, n: usize) -> Option<Self> {
        // Verify that blocks are non-empty and disjoint
        let mut seen = vec![false; n];

        for block in &blocks {
            if block.is_empty() {
                return None;
            }
            for &elem in block {
                if elem >= n || seen[elem] {
                    return None;
                }
                seen[elem] = true;
            }
        }

        // All elements must be covered
        if !seen.iter().all(|&x| x) {
            return None;
        }

        Some(SetPartition { blocks, n })
    }

    /// Get the number of blocks
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Get the blocks
    pub fn blocks(&self) -> &[Vec<usize>] {
        &self.blocks
    }

    /// Get the size of the set being partitioned
    pub fn size(&self) -> usize {
        self.n
    }
}

/// Generate all set partitions of n elements (labeled 0..n)
///
/// The number of set partitions is the Bell number B(n)
pub fn set_partitions(n: usize) -> Vec<SetPartition> {
    if n == 0 {
        return vec![SetPartition {
            blocks: vec![],
            n: 0,
        }];
    }

    let mut result = Vec::new();
    let elements: Vec<usize> = (0..n).collect();
    let mut current_partition: Vec<Vec<usize>> = Vec::new();

    generate_set_partitions(&elements, 0, &mut current_partition, &mut result);

    result
}

fn generate_set_partitions(
    elements: &[usize],
    index: usize,
    current: &mut Vec<Vec<usize>>,
    result: &mut Vec<SetPartition>,
) {
    if index == elements.len() {
        result.push(SetPartition {
            blocks: current.clone(),
            n: elements.len(),
        });
        return;
    }

    let elem = elements[index];

    // Try adding element to each existing block
    for i in 0..current.len() {
        current[i].push(elem);
        generate_set_partitions(elements, index + 1, current, result);
        current[i].pop();
    }

    // Try creating a new block with this element
    current.push(vec![elem]);
    generate_set_partitions(elements, index + 1, current, result);
    current.pop();
}

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

/// A perfect matching on 2n vertices
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerfectMatching {
    /// The matching as pairs of vertices
    pairs: Vec<(usize, usize)>,
    n: usize,
}

impl PerfectMatching {
    /// Create a new perfect matching
    pub fn new(pairs: Vec<(usize, usize)>, n: usize) -> Option<Self> {
        // Verify it's a valid perfect matching
        let mut seen = vec![false; 2 * n];

        for &(a, b) in &pairs {
            if a >= 2 * n || b >= 2 * n || seen[a] || seen[b] {
                return None;
            }
            seen[a] = true;
            seen[b] = true;
        }

        // Check all vertices are matched
        if !seen.iter().all(|&x| x) {
            return None;
        }

        Some(PerfectMatching { pairs, n })
    }

    /// Get the pairs in the matching
    pub fn pairs(&self) -> &[(usize, usize)] {
        &self.pairs
    }

    /// Get the number of pairs
    pub fn size(&self) -> usize {
        self.n
    }
}

/// Generate all perfect matchings on 2n vertices
pub fn perfect_matchings(n: usize) -> Vec<PerfectMatching> {
    if n == 0 {
        return vec![PerfectMatching {
            pairs: vec![],
            n: 0,
        }];
    }

    let mut result = Vec::new();
    let mut current_pairs = Vec::new();
    let mut available: Vec<usize> = (0..2 * n).collect();

    generate_perfect_matchings(&mut current_pairs, &mut available, n, &mut result);

    result
}

fn generate_perfect_matchings(
    current: &mut Vec<(usize, usize)>,
    available: &mut Vec<usize>,
    n: usize,
    result: &mut Vec<PerfectMatching>,
) {
    if available.is_empty() {
        result.push(PerfectMatching {
            pairs: current.clone(),
            n,
        });
        return;
    }

    // Take the first available vertex and try matching it with all others
    let first = available[0];

    for i in 1..available.len() {
        let second = available[i];

        // Create the pair
        current.push((first, second));

        // Remove both from available
        let mut new_available = available.clone();
        new_available.retain(|&x| x != first && x != second);

        generate_perfect_matchings(current, &mut new_available, n, result);

        current.pop();
    }
}

/// A Latin square of order n
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatinSquare {
    /// The square as a 2D grid
    grid: Vec<Vec<usize>>,
    n: usize,
}

impl LatinSquare {
    /// Create a new Latin square
    pub fn new(grid: Vec<Vec<usize>>) -> Option<Self> {
        let n = grid.len();

        if n == 0 {
            return Some(LatinSquare { grid, n: 0 });
        }

        // Check dimensions
        for row in &grid {
            if row.len() != n {
                return None;
            }
        }

        // Check that each row and column contains each symbol exactly once
        for i in 0..n {
            // Check row i
            let mut row_symbols = vec![false; n];
            for j in 0..n {
                if grid[i][j] >= n {
                    return None;
                }
                if row_symbols[grid[i][j]] {
                    return None;
                }
                row_symbols[grid[i][j]] = true;
            }

            // Check column i
            let mut col_symbols = vec![false; n];
            for j in 0..n {
                if col_symbols[grid[j][i]] {
                    return None;
                }
                col_symbols[grid[j][i]] = true;
            }
        }

        Some(LatinSquare { grid, n })
    }

    /// Get the grid
    pub fn grid(&self) -> &[Vec<usize>] {
        &self.grid
    }

    /// Get the order of the square
    pub fn order(&self) -> usize {
        self.n
    }

    /// Get the element at position (i, j)
    pub fn get(&self, i: usize, j: usize) -> Option<usize> {
        self.grid.get(i)?.get(j).copied()
    }
}

/// Generate all Latin squares of order n (warning: grows very quickly!)
pub fn latin_squares(n: usize) -> Vec<LatinSquare> {
    if n == 0 {
        return vec![LatinSquare {
            grid: vec![],
            n: 0,
        }];
    }

    let mut result = Vec::new();
    let mut grid = vec![vec![0; n]; n];

    generate_latin_squares(&mut grid, 0, 0, n, &mut result);

    result
}

fn generate_latin_squares(
    grid: &mut Vec<Vec<usize>>,
    row: usize,
    col: usize,
    n: usize,
    result: &mut Vec<LatinSquare>,
) {
    if row == n {
        result.push(LatinSquare {
            grid: grid.clone(),
            n,
        });
        return;
    }

    let (next_row, next_col) = if col + 1 < n {
        (row, col + 1)
    } else {
        (row + 1, 0)
    };

    // Try each symbol
    for symbol in 0..n {
        // Check if symbol can be placed at (row, col)
        if can_place_latin(grid, row, col, symbol, n) {
            grid[row][col] = symbol;
            generate_latin_squares(grid, next_row, next_col, n, result);
            grid[row][col] = 0; // Reset for backtracking
        }
    }
}

fn can_place_latin(grid: &[Vec<usize>], row: usize, col: usize, symbol: usize, _n: usize) -> bool {
    // Check row
    for j in 0..col {
        if grid[row][j] == symbol {
            return false;
        }
    }

    // Check column
    for i in 0..row {
        if grid[i][col] == symbol {
            return false;
        }
    }

    // Also check the rest of the current row and column for future consistency
    // (This is optional but helps prune search space)
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), Integer::from(1));
        assert_eq!(factorial(5), Integer::from(120));
        assert_eq!(factorial(10), Integer::from(3628800));
    }

    #[test]
    fn test_binomial() {
        assert_eq!(binomial(5, 2), Integer::from(10));
        assert_eq!(binomial(10, 5), Integer::from(252));
        assert_eq!(binomial(5, 0), Integer::from(1));
        assert_eq!(binomial(5, 5), Integer::from(1));
        assert_eq!(binomial(5, 6), Integer::from(0));
    }

    #[test]
    fn test_multinomial() {
        // 3!/(1!*1!*1!) = 6
        assert_eq!(multinomial(3, &[1, 1, 1]), Integer::from(6));

        // 5!/(2!*2!*1!) = 30
        assert_eq!(multinomial(5, &[2, 2, 1]), Integer::from(30));

        // Should be 0 if sum doesn't equal n
        assert_eq!(multinomial(5, &[2, 2]), Integer::from(0));
    }

    #[test]
    fn test_catalan() {
        // First few Catalan numbers: 1, 1, 2, 5, 14, 42, ...
        assert_eq!(catalan(0), Integer::from(1));
        assert_eq!(catalan(1), Integer::from(1));
        assert_eq!(catalan(2), Integer::from(2));
        assert_eq!(catalan(3), Integer::from(5));
        assert_eq!(catalan(4), Integer::from(14));
        assert_eq!(catalan(5), Integer::from(42));
    }

    #[test]
    fn test_fibonacci() {
        // First few Fibonacci numbers: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...
        assert_eq!(fibonacci(0), Integer::from(0));
        assert_eq!(fibonacci(1), Integer::from(1));
        assert_eq!(fibonacci(2), Integer::from(1));
        assert_eq!(fibonacci(3), Integer::from(2));
        assert_eq!(fibonacci(4), Integer::from(3));
        assert_eq!(fibonacci(5), Integer::from(5));
        assert_eq!(fibonacci(8), Integer::from(21));
        assert_eq!(fibonacci(10), Integer::from(55));
    }

    #[test]
    fn test_lucas() {
        // First few Lucas numbers: 2, 1, 3, 4, 7, 11, 18, ...
        assert_eq!(lucas(0), Integer::from(2));
        assert_eq!(lucas(1), Integer::from(1));
        assert_eq!(lucas(2), Integer::from(3));
        assert_eq!(lucas(3), Integer::from(4));
        assert_eq!(lucas(4), Integer::from(7));
        assert_eq!(lucas(5), Integer::from(11));
        assert_eq!(lucas(6), Integer::from(18));
    }

    #[test]
    fn test_falling_factorial() {
        // 5!/(5-3)! = 5*4*3 = 60
        assert_eq!(falling_factorial(5, 3), Integer::from(60));
        assert_eq!(falling_factorial(10, 2), Integer::from(90));
        assert_eq!(falling_factorial(5, 0), Integer::from(1));
        assert_eq!(falling_factorial(5, 6), Integer::from(0));
    }

    #[test]
    fn test_rising_factorial() {
        // 5 * 6 * 7 = 210
        assert_eq!(rising_factorial(5, 3), Integer::from(210));
        assert_eq!(rising_factorial(10, 2), Integer::from(110));
        assert_eq!(rising_factorial(5, 0), Integer::from(1));
    }

    #[test]
    fn test_stirling_second() {
        // S(n, 0) = 0 for n > 0
        assert_eq!(stirling_second(5, 0), Integer::from(0));

        // S(n, 1) = 1
        assert_eq!(stirling_second(5, 1), Integer::from(1));

        // S(n, n) = 1
        assert_eq!(stirling_second(5, 5), Integer::from(1));

        // S(5, 2) = 15
        assert_eq!(stirling_second(5, 2), Integer::from(15));

        // S(5, 3) = 25
        assert_eq!(stirling_second(5, 3), Integer::from(25));
    }

    #[test]
    fn test_bell_number() {
        // First few Bell numbers: 1, 1, 2, 5, 15, 52, ...
        assert_eq!(bell_number(0), Integer::from(1));
        assert_eq!(bell_number(1), Integer::from(1));
        assert_eq!(bell_number(2), Integer::from(2));
        assert_eq!(bell_number(3), Integer::from(5));
        assert_eq!(bell_number(4), Integer::from(15));
        assert_eq!(bell_number(5), Integer::from(52));
    }

    #[test]
    fn test_stirling_first() {
        // s(n, 0) = 0 for n > 0
        assert_eq!(stirling_first(5, 0), Integer::from(0));

        // s(n, n) = 1
        assert_eq!(stirling_first(5, 5), Integer::from(1));

        // s(4, 2) = 11
        assert_eq!(stirling_first(4, 2), Integer::from(11));

        // s(5, 2) = 50
        assert_eq!(stirling_first(5, 2), Integer::from(50));

        // s(5, 3) = 35
        assert_eq!(stirling_first(5, 3), Integer::from(35));
    }

    #[test]
    fn test_compositions() {
        // Compositions of 4: [4], [1,3], [2,2], [3,1], [1,1,2], [1,2,1], [2,1,1], [1,1,1,1]
        let comps = compositions(4);
        assert_eq!(comps.len(), 8); // 2^(n-1) = 2^3 = 8

        // All compositions should sum to 4
        for comp in &comps {
            assert_eq!(comp.sum(), 4);
        }
    }

    #[test]
    fn test_compositions_k() {
        // Compositions of 5 into 3 parts
        let comps = compositions_k(5, 3);
        // Should be: [1,1,3], [1,2,2], [1,3,1], [2,1,2], [2,2,1], [3,1,1]
        assert_eq!(comps.len(), 6);

        for comp in &comps {
            assert_eq!(comp.sum(), 5);
            assert_eq!(comp.length(), 3);
        }
    }

    #[test]
    fn test_composition_ordering() {
        // Compositions are ordered (unlike partitions)
        let comp1 = Composition::new(vec![1, 3]).unwrap();
        let comp2 = Composition::new(vec![3, 1]).unwrap();

        // These should be different
        assert_ne!(comp1, comp2);
        assert_eq!(comp1.sum(), comp2.sum());
    }

    #[test]
    fn test_set_partitions() {
        // Set partitions of 3 elements should equal Bell(3) = 5
        let parts = set_partitions(3);
        assert_eq!(parts.len(), 5);

        // Verify all partitions are valid
        for part in &parts {
            assert_eq!(part.size(), 3);
            assert!(part.num_blocks() > 0);

            // Check that all elements 0,1,2 appear exactly once
            let mut seen = vec![false; 3];
            for block in part.blocks() {
                for &elem in block {
                    assert!(!seen[elem]);
                    seen[elem] = true;
                }
            }
            assert!(seen.iter().all(|&x| x));
        }
    }

    #[test]
    fn test_set_partition_count() {
        // Number of set partitions should match Bell numbers
        assert_eq!(set_partitions(0).len(), 1); // Bell(0) = 1
        assert_eq!(set_partitions(1).len(), 1); // Bell(1) = 1
        assert_eq!(set_partitions(2).len(), 2); // Bell(2) = 2
        assert_eq!(set_partitions(3).len(), 5); // Bell(3) = 5
        assert_eq!(set_partitions(4).len(), 15); // Bell(4) = 15
    }

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
    fn test_perfect_matchings() {
        // Perfect matchings on 2 vertices: just one matching {(0,1)}
        let matchings1 = perfect_matchings(1);
        assert_eq!(matchings1.len(), 1);

        // Perfect matchings on 4 vertices: should be 3 matchings
        // {(0,1),(2,3)}, {(0,2),(1,3)}, {(0,3),(1,2)}
        let matchings2 = perfect_matchings(2);
        assert_eq!(matchings2.len(), 3);

        // Verify each is a valid matching
        for matching in &matchings2 {
            assert_eq!(matching.pairs().len(), 2);
        }
    }

    #[test]
    fn test_perfect_matching_validation() {
        // Valid matching on 4 vertices
        let matching = PerfectMatching::new(vec![(0, 1), (2, 3)], 2);
        assert!(matching.is_some());

        // Invalid - duplicate vertex
        let invalid = PerfectMatching::new(vec![(0, 1), (1, 2)], 2);
        assert!(invalid.is_none());

        // Invalid - missing vertex
        let invalid2 = PerfectMatching::new(vec![(0, 1)], 2);
        assert!(invalid2.is_none());
    }

    #[test]
    fn test_latin_squares() {
        // Latin squares of order 1
        let squares1 = latin_squares(1);
        assert_eq!(squares1.len(), 1);

        // Latin squares of order 2: should be 2 squares
        // [[0,1],[1,0]] and [[1,0],[0,1]]
        let squares2 = latin_squares(2);
        assert_eq!(squares2.len(), 2);

        // Verify each is valid
        for square in &squares2 {
            assert_eq!(square.order(), 2);
        }
    }

    #[test]
    fn test_latin_square_validation() {
        // Valid 3x3 Latin square
        let valid = LatinSquare::new(vec![vec![0, 1, 2], vec![1, 2, 0], vec![2, 0, 1]]);
        assert!(valid.is_some());

        // Invalid - repeated element in row
        let invalid = LatinSquare::new(vec![vec![0, 0, 1], vec![1, 2, 0], vec![2, 1, 2]]);
        assert!(invalid.is_none());

        // Invalid - repeated element in column
        let invalid2 = LatinSquare::new(vec![vec![0, 1, 2], vec![0, 2, 1], vec![2, 0, 1]]);
        assert!(invalid2.is_none());
    }
}
