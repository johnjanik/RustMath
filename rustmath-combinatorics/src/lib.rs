//! RustMath Combinatorics - Combinatorial objects and algorithms
//!
//! This crate provides combinatorial structures like permutations, combinations,
//! partitions, and algorithms for generating and manipulating them.

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
    }
}

// Future modules:
// - Permutations
// - Combinations
// - Partitions
// - Young tableaux
// - Posets
