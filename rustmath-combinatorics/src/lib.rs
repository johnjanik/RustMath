//! RustMath Combinatorics - Combinatorial objects and algorithms
//!
//! This crate provides combinatorial structures like permutations, combinations,
//! partitions, and algorithms for generating and manipulating them.

pub mod combinations;
pub mod partitions;
pub mod permutations;

pub use combinations::{combinations, Combination};
pub use partitions::{partition_count, partitions, Partition};
pub use permutations::{all_permutations, Permutation};

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
}
