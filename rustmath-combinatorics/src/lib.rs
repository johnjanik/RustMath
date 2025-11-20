//! RustMath Combinatorics - Combinatorial objects and algorithms
//!
//! This crate provides combinatorial structures like permutations, combinations,
//! partitions, and algorithms for generating and manipulating them.

pub mod binary_words;
pub mod combinations;
pub mod composition;
pub mod designs;
pub mod dyck_word;
pub mod enumeration;
pub mod partitions;
pub mod perfect_matching;
pub mod permutations;
pub mod posets;
pub mod ranking;
pub mod set_partition;
pub mod set_system;
pub mod species;
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
pub use composition::{
    compositions, compositions_k, signed_compositions, signed_compositions_k, Composition,
    SignedComposition,
};
pub use designs::{
    are_latin_squares_orthogonal, mutually_orthogonal_latin_squares, BlockDesign,
    DesignAutomorphism, DifferenceSet, HadamardMatrix, OrthogonalArray, SteinerSystem,
};
pub use dyck_word::{dyck_words, DyckWord};
pub use enumeration::{
    cartesian_product, stars_and_bars, tuples, weak_compositions, CompositionIterator,
    Enumerable, GrayCodeIterator, LazyEnumerator, PartitionIterator, RevolvingDoorIterator,
};
pub use perfect_matching::{perfect_matchings, PerfectMatching};
pub use ranking::{CombinationRank, PermutationRank, Rankable, RankingTable};
pub use set_partition::{set_partitions, SetPartition};
pub use set_system::SetSystem;
pub use word::{
    abelian_complexity, boyer_moore_search, christoffel_word, factor_complexity, kmp_search,
    lyndon_factorization, lyndon_words as general_lyndon_words, sturmian_word, AutomaticSequence,
    Morphism, Word,
};

// Core combinatorial functions (factorials, Stirling numbers, etc.) defined in this module

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
