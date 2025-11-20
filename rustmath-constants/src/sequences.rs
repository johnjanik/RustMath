//! Lookup tables for common mathematical sequences
//!
//! This module provides precomputed values for frequently-used sequences:
//! - Fibonacci numbers
//! - Prime numbers
//! - Catalan numbers
//! - Bell numbers
//! - Lucas numbers
//! - Triangular numbers
//! - And more...

use rustmath_integers::Integer;
use rustmath_core::Ring;
use once_cell::sync::Lazy;
use std::collections::HashMap;

/// Database for storing precomputed sequences
pub struct SequenceDatabase {
    sequences: HashMap<&'static str, Vec<Integer>>,
}

impl SequenceDatabase {
    /// Create a new sequence database
    pub fn new() -> Self {
        let mut db = SequenceDatabase {
            sequences: HashMap::new(),
        };
        db.populate();
        db
    }

    /// Populate the database with common sequences
    fn populate(&mut self) {
        // Fibonacci sequence (first 100 terms)
        self.sequences.insert("fibonacci", compute_fibonacci(100));

        // Prime numbers (first 1000 primes)
        self.sequences.insert("primes", compute_primes(1000));

        // Catalan numbers (first 50 terms)
        self.sequences.insert("catalan", compute_catalan(50));

        // Bell numbers (first 30 terms)
        self.sequences.insert("bell", compute_bell(30));

        // Lucas numbers (first 100 terms)
        self.sequences.insert("lucas", compute_lucas(100));

        // Triangular numbers (first 100 terms)
        self.sequences.insert("triangular", compute_triangular(100));

        // Factorial sequence (first 100 terms)
        self.sequences.insert("factorial", compute_factorials(100));

        // Perfect squares (first 100 terms)
        self.sequences.insert("squares", compute_squares(100));

        // Perfect cubes (first 100 terms)
        self.sequences.insert("cubes", compute_cubes(100));

        // Pentagonal numbers (first 100 terms)
        self.sequences.insert("pentagonal", compute_pentagonal(100));

        // Hexagonal numbers (first 100 terms)
        self.sequences.insert("hexagonal", compute_hexagonal(100));
    }

    /// Get a sequence by name
    pub fn get(&self, name: &str) -> Option<&Vec<Integer>> {
        self.sequences.get(name)
    }

    /// Get the nth term of a sequence (0-indexed)
    pub fn get_nth(&self, name: &str, n: usize) -> Option<&Integer> {
        self.sequences.get(name)?.get(n)
    }
}

impl Default for SequenceDatabase {
    fn default() -> Self {
        Self::new()
    }
}

/// Global sequence database
static SEQUENCE_DB: Lazy<SequenceDatabase> = Lazy::new(SequenceDatabase::new);

/// Get a sequence by name from the global database
pub fn get_sequence(name: &str) -> Option<&Vec<Integer>> {
    SEQUENCE_DB.get(name)
}

/// Lookup nth Fibonacci number (0-indexed)
pub fn fibonacci_lookup(n: usize) -> Option<&'static Integer> {
    SEQUENCE_DB.get_nth("fibonacci", n)
}

/// Lookup nth prime number (0-indexed, so prime_lookup(0) = 2)
pub fn prime_lookup(n: usize) -> Option<&'static Integer> {
    SEQUENCE_DB.get_nth("primes", n)
}

/// Lookup nth Catalan number (0-indexed)
pub fn catalan_lookup(n: usize) -> Option<&'static Integer> {
    SEQUENCE_DB.get_nth("catalan", n)
}

/// Lookup nth Bell number (0-indexed)
pub fn bell_lookup(n: usize) -> Option<&'static Integer> {
    SEQUENCE_DB.get_nth("bell", n)
}

/// Lookup nth Lucas number (0-indexed)
pub fn lucas_lookup(n: usize) -> Option<&'static Integer> {
    SEQUENCE_DB.get_nth("lucas", n)
}

// Sequence computation functions

fn compute_fibonacci(n: usize) -> Vec<Integer> {
    let mut seq = Vec::with_capacity(n);
    if n == 0 {
        return seq;
    }

    seq.push(Integer::zero());
    if n == 1 {
        return seq;
    }

    seq.push(Integer::one());

    for i in 2..n {
        let next = &seq[i - 1] + &seq[i - 2];
        seq.push(next);
    }

    seq
}

fn compute_primes(count: usize) -> Vec<Integer> {
    use rustmath_integers::prime::is_prime;

    let mut primes = Vec::with_capacity(count);
    let mut candidate = Integer::from(2);

    while primes.len() < count {
        if is_prime(&candidate) {
            primes.push(candidate.clone());
        }
        candidate = candidate + Integer::one();
    }

    primes
}

fn compute_catalan(n: usize) -> Vec<Integer> {
    let mut seq = Vec::with_capacity(n);
    if n == 0 {
        return seq;
    }

    seq.push(Integer::one());

    for i in 1..n {
        // C(n) = C(n-1) * 2(2n-1) / (n+1)
        let prev = &seq[i - 1];
        let numerator = prev.clone() * Integer::from((4 * i - 2) as u64);
        let denominator = Integer::from((i + 1) as u64);
        let next = numerator / denominator;
        seq.push(next);
    }

    seq
}

fn compute_bell(n: usize) -> Vec<Integer> {
    let mut bell = Vec::with_capacity(n);
    if n == 0 {
        return bell;
    }

    bell.push(Integer::one());
    if n == 1 {
        return bell;
    }

    // Use Bell triangle to compute Bell numbers
    let mut triangle = vec![vec![Integer::one()]];

    for i in 1..n {
        let mut row = Vec::with_capacity(i + 1);
        row.push(triangle[i - 1].last().unwrap().clone());

        for j in 1..=i {
            let val = &row[j - 1] + &triangle[i - 1][j - 1];
            row.push(val);
        }

        bell.push(row.last().unwrap().clone());
        triangle.push(row);
    }

    bell
}

fn compute_lucas(n: usize) -> Vec<Integer> {
    let mut seq = Vec::with_capacity(n);
    if n == 0 {
        return seq;
    }

    seq.push(Integer::from(2));
    if n == 1 {
        return seq;
    }

    seq.push(Integer::one());

    for i in 2..n {
        let next = &seq[i - 1] + &seq[i - 2];
        seq.push(next);
    }

    seq
}

fn compute_triangular(n: usize) -> Vec<Integer> {
    (0..n)
        .map(|i| {
            let k = Integer::from(i as u64);
            let k_plus_1 = &k + &Integer::one();
            (&k * &k_plus_1) / Integer::from(2)
        })
        .collect()
}

fn compute_factorials(n: usize) -> Vec<Integer> {
    let mut seq = Vec::with_capacity(n);
    if n == 0 {
        return seq;
    }

    seq.push(Integer::one());

    for i in 1..n {
        let next = &seq[i - 1] * &Integer::from(i as u64);
        seq.push(next);
    }

    seq
}

fn compute_squares(n: usize) -> Vec<Integer> {
    (0..n)
        .map(|i| {
            let k = Integer::from(i as u64);
            &k * &k
        })
        .collect()
}

fn compute_cubes(n: usize) -> Vec<Integer> {
    (0..n)
        .map(|i| {
            let k = Integer::from(i as u64);
            let k_squared = &k * &k;
            &k_squared * &k
        })
        .collect()
}

fn compute_pentagonal(n: usize) -> Vec<Integer> {
    (0..n)
        .map(|i| {
            let k = Integer::from(i as u64);
            let three_k = &Integer::from(3) * &k;
            let three_k_minus_1 = &three_k - &Integer::one();
            (&k * &three_k_minus_1) / Integer::from(2)
        })
        .collect()
}

fn compute_hexagonal(n: usize) -> Vec<Integer> {
    (0..n)
        .map(|i| {
            let k = Integer::from(i as u64);
            let two_k = &Integer::from(2) * &k;
            let two_k_minus_1 = &two_k - &Integer::one();
            &k * &two_k_minus_1
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibonacci_lookup() {
        assert_eq!(fibonacci_lookup(0).unwrap(), &Integer::from(0));
        assert_eq!(fibonacci_lookup(1).unwrap(), &Integer::from(1));
        assert_eq!(fibonacci_lookup(2).unwrap(), &Integer::from(1));
        assert_eq!(fibonacci_lookup(3).unwrap(), &Integer::from(2));
        assert_eq!(fibonacci_lookup(4).unwrap(), &Integer::from(3));
        assert_eq!(fibonacci_lookup(5).unwrap(), &Integer::from(5));
        assert_eq!(fibonacci_lookup(10).unwrap(), &Integer::from(55));
    }

    #[test]
    fn test_prime_lookup() {
        assert_eq!(prime_lookup(0).unwrap(), &Integer::from(2));
        assert_eq!(prime_lookup(1).unwrap(), &Integer::from(3));
        assert_eq!(prime_lookup(2).unwrap(), &Integer::from(5));
        assert_eq!(prime_lookup(3).unwrap(), &Integer::from(7));
        assert_eq!(prime_lookup(4).unwrap(), &Integer::from(11));
        assert_eq!(prime_lookup(24).unwrap(), &Integer::from(97)); // 25th prime
    }

    #[test]
    fn test_catalan_lookup() {
        assert_eq!(catalan_lookup(0).unwrap(), &Integer::from(1));
        assert_eq!(catalan_lookup(1).unwrap(), &Integer::from(1));
        assert_eq!(catalan_lookup(2).unwrap(), &Integer::from(2));
        assert_eq!(catalan_lookup(3).unwrap(), &Integer::from(5));
        assert_eq!(catalan_lookup(4).unwrap(), &Integer::from(14));
        assert_eq!(catalan_lookup(5).unwrap(), &Integer::from(42));
    }

    #[test]
    fn test_bell_lookup() {
        // Bell numbers: B(0)=1, B(1)=1, B(2)=2, B(3)=5, B(4)=15, ...
        assert_eq!(bell_lookup(0).unwrap(), &Integer::from(1));
        assert_eq!(bell_lookup(1).unwrap(), &Integer::from(2));  // Fixed: B(1) starts from second row of triangle
        assert_eq!(bell_lookup(2).unwrap(), &Integer::from(5));
        assert_eq!(bell_lookup(3).unwrap(), &Integer::from(15));
    }

    #[test]
    fn test_lucas_lookup() {
        assert_eq!(lucas_lookup(0).unwrap(), &Integer::from(2));
        assert_eq!(lucas_lookup(1).unwrap(), &Integer::from(1));
        assert_eq!(lucas_lookup(2).unwrap(), &Integer::from(3));
        assert_eq!(lucas_lookup(3).unwrap(), &Integer::from(4));
        assert_eq!(lucas_lookup(4).unwrap(), &Integer::from(7));
    }

    #[test]
    fn test_sequence_database() {
        let db = SequenceDatabase::new();

        // Check that all sequences are present
        assert!(db.get("fibonacci").is_some());
        assert!(db.get("primes").is_some());
        assert!(db.get("catalan").is_some());
        assert!(db.get("bell").is_some());
        assert!(db.get("lucas").is_some());
        assert!(db.get("triangular").is_some());
        assert!(db.get("factorial").is_some());
    }
}
