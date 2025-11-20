//! RustMath Combinatorics - Combinatorial objects and algorithms
//!
//! This crate provides combinatorial structures like permutations, combinations,
//! partitions, and algorithms for generating and manipulating them.

pub mod affine_permutations;
pub mod binary_words;
pub mod combinations;
pub mod composition;
pub mod derangements;
pub mod designs;
pub mod dyck_word;
pub mod enumeration;
pub mod integer_vectors;
pub mod integer_matrices;
pub mod multiset_partition_into_sets_ordered;
pub mod ordered_tree;
pub mod partitions;
pub mod perfect_matching;
pub mod permutations;
pub mod plane_partition;
pub mod permutation_simd;
pub mod posets;
pub mod q_analogue;
pub mod ranking;
pub mod recurrence_sequences;
pub mod restricted_growth;
pub mod set_partition;
pub mod set_system;
pub mod sidon_sets;
pub mod skew_partition;
pub mod species;
pub mod subset;
pub mod superpartitions;
pub mod tableaux;
pub mod tamari;
pub mod tuple;
pub mod vector_partition;
pub mod word;
pub mod wreath_product;

pub use affine_permutations::{AffinePermutation, CoxeterType};
pub use combinations::{combinations, Combination};
pub use partitions::{
    count_partitions_with_max_parts, partition_count, partitions, partitions_with_distinct_parts,
    partitions_with_even_parts, partitions_with_exact_parts, partitions_with_max_part,
    partitions_with_max_parts, partitions_with_min_part, partitions_with_odd_parts, Partition,
    PartitionTuple,
};
pub use permutations::{all_permutations, Permutation};
pub use permutation_simd::{
    batch_compose_simd, compose_simd, cycles_simd, inverse_simd, power_simd, simd_available,
    simd_info,
};
pub use posets::Poset;
pub use tableaux::{robinson_schensted, rs_insert, standard_tableaux, Tableau};
pub use tuple::{tuples as tuple_tuples, Tuple, TupleIterator};

// Re-export new modules
pub use binary_words::{all_binary_words, binary_words_with_weight, lyndon_words, necklaces, BinaryWord};
pub use composition::{
    compositions, compositions_k, integer_vectors_weighted, integer_vectors_weighted_dp,
    signed_compositions, signed_compositions_k, Composition, SignedComposition,
    WeightedIntegerVector,
};
pub use designs::{
    are_latin_squares_orthogonal, mutually_orthogonal_latin_squares, BlockDesign,
    DesignAutomorphism, DifferenceSet, HadamardMatrix, OrthogonalArray, SteinerSystem,
};
pub use dyck_word::{dyck_words, nu_dyck_words, BounceStats, DyckWord, NuDyckWord};
pub use enumeration::{
    cartesian_product, stars_and_bars, tuples, weak_compositions, CartesianProduct,
    CompositionIterator, Enumerable, GrayCodeIterator, InfiniteCartesianProduct, LazyEnumerator,
    PartitionIterator, RevolvingDoorIterator,
};
pub use integer_matrices::{
    count_integer_matrices, integer_matrices, integer_matrices_bounded, IntegerMatrix,
};
pub use perfect_matching::{perfect_matchings, PerfectMatching};
pub use ranking::{CombinationRank, PermutationRank, Rankable, RankingTable};
pub use recurrence_sequences::{
    solve_binary_recurrence, BinaryRecurrence, LinearRecurrence, RecurrenceSequence,
};
pub use set_partition::{
    bell_number_optimized, bell_numbers_up_to, set_partition_iterator, set_partitions,
    SetPartition, SetPartitionIterator,
};
pub use set_system::SetSystem;
pub use multiset_partition_into_sets_ordered::OrderedMultisetPartitionIntoSets;
pub use word::{
    abelian_complexity, boyer_moore_search, christoffel_word, factor_complexity, kmp_search,
    lyndon_factorization, lyndon_words as general_lyndon_words, sturmian_word, AutomaticSequence,
    Morphism, Word,
};
pub use subset::{
    all_subsets, count_k_subsets, k_subset_iterator, k_subsets, subset_iterator, KSubsetIterator,
    Subset, SubsetIterator,
};
pub use q_analogue::{
    gaussian_polynomial, q_binomial, q_binomial_eval, q_factorial, q_integer, q_multinomial,
};
pub use wreath_product::{all_colored_permutations, ColoredPermutation};
pub use plane_partition::{
    count_cyclically_symmetric_plane_partitions, count_plane_partitions_in_box,
    count_self_complementary_plane_partitions, count_totally_symmetric_plane_partitions,
    count_transpose_complement_plane_partitions, plane_partitions, plane_partitions_in_box,
    PlanePartition,
};
pub use ordered_tree::{OrderedTree, OrderedTreeNode, PreorderIterator};
pub use derangements::{
    all_derangements, count_derangements, count_derangements_recurrence, is_derangement,
};
pub use superpartitions::{
    count_superpartitions, strict_superpartitions, superpartitions, superpartitions_with_k_parts,
    superpartitions_with_m_circled, SuperPart, SuperPartition,
};
pub use integer_vectors::{
    count_integer_vectors_in_box, count_integer_vectors_with_sum, integer_vector_sum_iter,
    integer_vectors_in_box, integer_vectors_with_l1_norm_bounded, integer_vectors_with_sum,
    integer_vectors_with_weighted_sum, IntegerVector, IntegerVectorSumIter,
};
pub use vector_partition::{
    count_vector_partitions_exact_parts, count_vector_partitions_max_parts,
    fast_vector_partitions, fast_vector_partitions_with_max_part, vector_partitions,
    vector_partitions_with_max_part, VectorPartition,
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

/// Compute Eulerian number A(n, k)
///
/// Number of permutations of {1, 2, ..., n} with exactly k descents.
/// A descent in a permutation is a position i where π(i) > π(i+1).
///
/// The Eulerian numbers satisfy the recurrence:
/// A(n, k) = (k+1)*A(n-1, k) + (n-k)*A(n-1, k-1)
///
/// Properties:
/// - A(n, 0) = 1 (identity permutation has 0 descents)
/// - A(n, k) = 0 for k >= n
/// - Sum over k of A(n, k) = n!
pub fn eulerian(n: u32, k: u32) -> Integer {
    if n == 0 && k == 0 {
        return Integer::one();
    }
    if k >= n {
        return Integer::zero();
    }
    if k == 0 {
        return Integer::one();
    }

    // Use dynamic programming to compute A(n, k)
    let mut dp = vec![vec![Integer::zero(); (k + 1) as usize]; (n + 1) as usize];
    dp[0][0] = Integer::one();

    for i in 1..=n as usize {
        for j in 0..=k.min((i - 1) as u32) as usize {
            if j == 0 {
                dp[i][j] = Integer::one();
            } else {
                // A(n, k) = (k+1)*A(n-1, k) + (n-k)*A(n-1, k-1)
                let term1 = Integer::from((j + 1) as u32) * dp[i - 1][j].clone();
                let term2 = Integer::from((i as u32) - (j as u32)) * dp[i - 1][j - 1].clone();
                dp[i][j] = term1 + term2;
            }
        }
    }

    dp[n as usize][k as usize].clone()
}

/// Compute Narayana number N(n, k)
///
/// Number of expressions containing n pairs of parentheses which are correctly matched
/// and which contain k distinct nestings. Equivalently, the number of Dyck paths of
/// length 2n with exactly k peaks.
///
/// The Narayana numbers are given by:
/// N(n, k) = (1/n) * C(n, k) * C(n, k-1)
///
/// Properties:
/// - N(n, 0) = 0 for n > 0
/// - N(n, 1) = 1 for n >= 1
/// - Sum over k of N(n, k) = C_n (nth Catalan number)
pub fn narayana(n: u32, k: u32) -> Integer {
    if n == 0 {
        return if k == 0 { Integer::one() } else { Integer::zero() };
    }
    if k == 0 || k > n {
        return Integer::zero();
    }
    if k == 1 {
        return Integer::one();
    }

    // N(n, k) = (1/n) * C(n, k) * C(n, k-1)
    let binom_n_k = binomial(n, k);
    let binom_n_k_minus_1 = binomial(n, k - 1);

    (binom_n_k * binom_n_k_minus_1) / Integer::from(n)
}

/// Compute central Delannoy number D(n)
///
/// Number of paths from (0, 0) to (n, n) using steps (1, 0), (0, 1), and (1, 1).
/// Also the number of ways to place n non-attacking rooks on an n×n chessboard
/// with some squares removed.
///
/// The central Delannoy numbers satisfy:
/// D(n) = Sum_{k=0}^{n} C(n, k)^2 * 2^k
///
/// First few values: 1, 3, 13, 63, 321, 1683, ...
pub fn delannoy_central(n: u32) -> Integer {
    let mut sum = Integer::zero();

    for k in 0..=n {
        let binom_sq = binomial(n, k).clone();
        let term = binom_sq.clone() * binom_sq * Integer::from(2).pow(k);
        sum = sum + term;
    }

    sum
}

/// Compute Delannoy number D(m, n)
///
/// Number of paths from (0, 0) to (m, n) using steps (1, 0), (0, 1), and (1, 1).
///
/// The Delannoy numbers satisfy:
/// D(m, n) = Sum_{k=0}^{min(m,n)} C(m, k) * C(n, k) * 2^k
pub fn delannoy(m: u32, n: u32) -> Integer {
    let min_mn = m.min(n);
    let mut sum = Integer::zero();

    for k in 0..=min_mn {
        let term = binomial(m, k) * binomial(n, k) * Integer::from(2).pow(k);
        sum = sum + term;
    }

    sum
}

/// Compute nth Motzkin number M(n)
///
/// Number of ways to draw non-intersecting chords between n points on a circle,
/// where not all points need to be paired. Also the number of lattice paths from
/// (0, 0) to (n, 0) using steps (1, 1), (1, -1), and (1, 0) that never go below y=0.
///
/// The Motzkin numbers satisfy the recurrence:
/// M(n) = M(n-1) + Sum_{k=0}^{n-2} M(k) * M(n-2-k)
///
/// First few values: 1, 1, 2, 4, 9, 21, 51, 127, ...
pub fn motzkin(n: u32) -> Integer {
    if n == 0 || n == 1 {
        return Integer::one();
    }

    let mut m = vec![Integer::zero(); (n + 1) as usize];
    m[0] = Integer::one();
    m[1] = Integer::one();

    for i in 2..=n as usize {
        // M(i) = M(i-1) + Sum_{k=0}^{i-2} M(k) * M(i-2-k)
        m[i] = m[i - 1].clone();

        for k in 0..=(i - 2) {
            m[i] = m[i].clone() + m[k].clone() * m[i - 2 - k].clone();
        }
    }

    m[n as usize].clone()
}

/// Compute nth large Schröder number S(n)
///
/// Number of lattice paths from (0, 0) to (2n, 0) using steps (1, 1), (1, -1),
/// and (2, 0) that never go below the x-axis. Also counts the number of ways
/// to parenthesize a string of n+1 factors.
///
/// The large Schröder numbers satisfy the recurrence:
/// (n+1)*S(n) = 3*(2n-1)*S(n-1) - (n-2)*S(n-2)
///
/// First few values: 1, 2, 6, 22, 90, 394, 1806, ...
pub fn schroder_large(n: u32) -> Integer {
    if n == 0 {
        return Integer::one();
    }
    if n == 1 {
        return Integer::from(2);
    }

    let mut s = vec![Integer::zero(); (n + 1) as usize];
    s[0] = Integer::one();
    s[1] = Integer::from(2);

    for i in 2..=n as usize {
        // (i+1)*S(i) = 3*(2*i-1)*S(i-1) - (i-2)*S(i-2)
        // S(i) = (3*(2*i-1)*S(i-1) - (i-2)*S(i-2)) / (i+1)
        let term1 = Integer::from(3 * (2 * i - 1) as u32) * s[i - 1].clone();
        let term2 = Integer::from((i - 2) as u32) * s[i - 2].clone();
        s[i] = (term1 - term2) / Integer::from((i + 1) as u32);
    }

    s[n as usize].clone()
}

/// Compute nth small Schröder number (or Schröder-Hipparchus number) s(n)
///
/// Related to large Schröder numbers by: s(n) = S(n) / 2 for n >= 1, and s(0) = 1
/// The small Schröder numbers count lattice paths and bracket structures.
///
/// First few values: 1, 1, 3, 11, 45, 197, 903, ...
pub fn schroder_small(n: u32) -> Integer {
    if n == 0 {
        return Integer::one();
    }

    // s(n) = S(n) / 2 for n >= 1
    schroder_large(n) / Integer::from(2)
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

    #[test]
    fn test_eulerian() {
        // A(n, k) = 0 for k >= n
        assert_eq!(eulerian(5, 5), Integer::from(0));
        assert_eq!(eulerian(5, 6), Integer::from(0));

        // A(n, 0) = 1 (identity permutation has 0 descents)
        assert_eq!(eulerian(0, 0), Integer::from(1));
        assert_eq!(eulerian(5, 0), Integer::from(1));
        assert_eq!(eulerian(10, 0), Integer::from(1));

        // Known values for small n
        // A(3, 0) = 1, A(3, 1) = 4, A(3, 2) = 1
        assert_eq!(eulerian(3, 0), Integer::from(1));
        assert_eq!(eulerian(3, 1), Integer::from(4));
        assert_eq!(eulerian(3, 2), Integer::from(1));

        // A(4, 0) = 1, A(4, 1) = 11, A(4, 2) = 11, A(4, 3) = 1
        assert_eq!(eulerian(4, 0), Integer::from(1));
        assert_eq!(eulerian(4, 1), Integer::from(11));
        assert_eq!(eulerian(4, 2), Integer::from(11));
        assert_eq!(eulerian(4, 3), Integer::from(1));

        // A(5, 1) = 26, A(5, 2) = 66
        assert_eq!(eulerian(5, 1), Integer::from(26));
        assert_eq!(eulerian(5, 2), Integer::from(66));

        // Verify that sum of A(n, k) over k equals n!
        let mut sum = Integer::zero();
        for k in 0..5 {
            sum = sum + eulerian(5, k);
        }
        assert_eq!(sum, factorial(5));
    }

    #[test]
    fn test_narayana() {
        // N(0, 0) = 1
        assert_eq!(narayana(0, 0), Integer::from(1));

        // N(n, 0) = 0 for n > 0
        assert_eq!(narayana(1, 0), Integer::from(0));
        assert_eq!(narayana(5, 0), Integer::from(0));

        // N(n, k) = 0 for k > n
        assert_eq!(narayana(3, 5), Integer::from(0));

        // N(n, 1) = 1 for n >= 1
        assert_eq!(narayana(1, 1), Integer::from(1));
        assert_eq!(narayana(5, 1), Integer::from(1));
        assert_eq!(narayana(10, 1), Integer::from(1));

        // Known values
        // N(3, 1) = 1, N(3, 2) = 3, N(3, 3) = 1
        assert_eq!(narayana(3, 1), Integer::from(1));
        assert_eq!(narayana(3, 2), Integer::from(3));
        assert_eq!(narayana(3, 3), Integer::from(1));

        // N(4, 2) = 6, N(4, 3) = 6
        assert_eq!(narayana(4, 2), Integer::from(6));
        assert_eq!(narayana(4, 3), Integer::from(6));

        // Verify that sum of N(n, k) equals C_n (Catalan number)
        let mut sum = Integer::zero();
        for k in 1..=5 {
            sum = sum + narayana(5, k);
        }
        assert_eq!(sum, catalan(5));

        // Another verification for n=4
        let mut sum4 = Integer::zero();
        for k in 1..=4 {
            sum4 = sum4 + narayana(4, k);
        }
        assert_eq!(sum4, catalan(4));
    }

    #[test]
    fn test_delannoy_central() {
        // First few central Delannoy numbers: 1, 3, 13, 63, 321, 1683
        assert_eq!(delannoy_central(0), Integer::from(1));
        assert_eq!(delannoy_central(1), Integer::from(3));
        assert_eq!(delannoy_central(2), Integer::from(13));
        assert_eq!(delannoy_central(3), Integer::from(63));
        assert_eq!(delannoy_central(4), Integer::from(321));
        assert_eq!(delannoy_central(5), Integer::from(1683));
    }

    #[test]
    fn test_delannoy() {
        // D(m, n) should be symmetric
        assert_eq!(delannoy(3, 5), delannoy(5, 3));
        assert_eq!(delannoy(2, 4), delannoy(4, 2));

        // D(n, n) should equal central Delannoy number
        assert_eq!(delannoy(0, 0), delannoy_central(0));
        assert_eq!(delannoy(3, 3), delannoy_central(3));
        assert_eq!(delannoy(5, 5), delannoy_central(5));

        // D(0, n) = 1 (only one path along the edge)
        assert_eq!(delannoy(0, 0), Integer::from(1));
        assert_eq!(delannoy(0, 5), Integer::from(1));
        assert_eq!(delannoy(5, 0), Integer::from(1));

        // Known values
        assert_eq!(delannoy(2, 3), Integer::from(25));
        assert_eq!(delannoy(3, 4), Integer::from(129));
    }

    #[test]
    fn test_motzkin() {
        // First few Motzkin numbers: 1, 1, 2, 4, 9, 21, 51, 127, 323
        assert_eq!(motzkin(0), Integer::from(1));
        assert_eq!(motzkin(1), Integer::from(1));
        assert_eq!(motzkin(2), Integer::from(2));
        assert_eq!(motzkin(3), Integer::from(4));
        assert_eq!(motzkin(4), Integer::from(9));
        assert_eq!(motzkin(5), Integer::from(21));
        assert_eq!(motzkin(6), Integer::from(51));
        assert_eq!(motzkin(7), Integer::from(127));
        assert_eq!(motzkin(8), Integer::from(323));
    }

    #[test]
    fn test_schroder_large() {
        // First few large Schröder numbers: 1, 2, 6, 22, 90, 394, 1806
        assert_eq!(schroder_large(0), Integer::from(1));
        assert_eq!(schroder_large(1), Integer::from(2));
        assert_eq!(schroder_large(2), Integer::from(6));
        assert_eq!(schroder_large(3), Integer::from(22));
        assert_eq!(schroder_large(4), Integer::from(90));
        assert_eq!(schroder_large(5), Integer::from(394));
        assert_eq!(schroder_large(6), Integer::from(1806));
    }

    #[test]
    fn test_schroder_small() {
        // First few small Schröder numbers: 1, 1, 3, 11, 45, 197, 903
        assert_eq!(schroder_small(0), Integer::from(1));
        assert_eq!(schroder_small(1), Integer::from(1));
        assert_eq!(schroder_small(2), Integer::from(3));
        assert_eq!(schroder_small(3), Integer::from(11));
        assert_eq!(schroder_small(4), Integer::from(45));
        assert_eq!(schroder_small(5), Integer::from(197));
        assert_eq!(schroder_small(6), Integer::from(903));

        // Verify relationship: S(n) = 2*s(n) for n > 0
        for n in 1..=6 {
            assert_eq!(schroder_large(n), schroder_small(n) * Integer::from(2));
        }
    }

    #[test]
    fn test_counting_edge_cases() {
        // Test edge cases for all new functions

        // Eulerian with n=0
        assert_eq!(eulerian(0, 0), Integer::from(1));
        assert_eq!(eulerian(0, 1), Integer::from(0));

        // Narayana edge cases
        assert_eq!(narayana(1, 1), Integer::from(1));
        assert_eq!(narayana(1, 2), Integer::from(0));

        // Delannoy with zero
        assert_eq!(delannoy(0, 0), Integer::from(1));
        assert_eq!(delannoy_central(0), Integer::from(1));

        // Motzkin base cases
        assert_eq!(motzkin(0), Integer::from(1));
        assert_eq!(motzkin(1), Integer::from(1));

        // Schröder base cases
        assert_eq!(schroder_large(0), Integer::from(1));
        assert_eq!(schroder_small(0), Integer::from(1));
    }
}
