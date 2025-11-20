//! Precomputed factorizations database
//!
//! This module provides a database of precomputed prime factorizations
//! for commonly-used numbers and special forms.

use rustmath_integers::Integer;
use rustmath_integers::prime::factor;
use rustmath_core::Ring;
use once_cell::sync::Lazy;
use std::collections::HashMap;

/// Factorization stored as (prime, exponent) pairs
pub type Factorization = Vec<(Integer, u32)>;

/// Database of precomputed factorizations
pub struct FactorizationDatabase {
    /// Map from number to its prime factorization
    factorizations: HashMap<Integer, Factorization>,
}

impl FactorizationDatabase {
    /// Create a new factorization database
    pub fn new() -> Self {
        let mut db = FactorizationDatabase {
            factorizations: HashMap::new(),
        };
        db.populate();
        db
    }

    /// Populate the database with common factorizations
    fn populate(&mut self) {
        // Factorizations up to 1000
        self.add_range_factorizations(2, 1000);

        // Special numbers
        self.add_mersenne_numbers();
        self.add_fermat_numbers();
        self.add_perfect_powers();
        self.add_factorial_products();
    }

    /// Add factorizations for a range of numbers
    fn add_range_factorizations(&mut self, start: u64, end: u64) {
        for n in start..=end {
            let num = Integer::from(n);
            let factors = factor(&num);
            self.factorizations.insert(num, factors);
        }
    }

    /// Add Mersenne numbers (2^p - 1) and their factorizations
    fn add_mersenne_numbers(&mut self) {
        // Helper function to compute 2^n
        let pow2 = |n: u32| -> Integer {
            let mut result = Integer::one();
            let base = Integer::from(2);
            for _ in 0..n {
                result = &result * &base;
            }
            result
        };

        // Small Mersenne numbers
        let mersenne_primes = vec![
            (2, vec![(3, 1)]),
            (3, vec![(7, 1)]),
            (5, vec![(31, 1)]),
            (7, vec![(127, 1)]),
            (13, vec![(8191, 1)]),
            (17, vec![(131071, 1)]),
            (19, vec![(524287, 1)]),
        ];

        for (p, factors) in mersenne_primes {
            let num = pow2(p) - Integer::one();
            let fact: Factorization = factors
                .into_iter()
                .map(|(p, e)| (Integer::from(p), e))
                .collect();
            self.factorizations.insert(num, fact);
        }

        // Some composite Mersenne numbers
        // 2^11 - 1 = 2047 = 23 × 89
        let m11 = pow2(11) - Integer::one();
        self.factorizations
            .insert(m11, vec![(Integer::from(23), 1), (Integer::from(89), 1)]);
    }

    /// Add Fermat numbers (2^(2^n) + 1)
    fn add_fermat_numbers(&mut self) {
        // F0 = 2^1 + 1 = 3
        let f0 = Integer::from(3);
        self.factorizations.insert(f0, vec![(Integer::from(3), 1)]);

        // F1 = 2^2 + 1 = 5
        let f1 = Integer::from(5);
        self.factorizations.insert(f1, vec![(Integer::from(5), 1)]);

        // F2 = 2^4 + 1 = 17
        let f2 = Integer::from(17);
        self.factorizations.insert(f2, vec![(Integer::from(17), 1)]);

        // F3 = 2^8 + 1 = 257
        let f3 = Integer::from(257);
        self.factorizations.insert(f3, vec![(Integer::from(257), 1)]);

        // F4 = 2^16 + 1 = 65537
        let f4 = Integer::from(65537);
        self.factorizations
            .insert(f4, vec![(Integer::from(65537), 1)]);

        // F5 = 2^32 + 1 = 4294967297 = 641 × 6700417 (composite!)
        let mut f5_val = Integer::one();
        let base = Integer::from(2u64);
        for _ in 0..32 {
            f5_val = &f5_val * &base;
        }
        let f5 = f5_val + Integer::one();
        self.factorizations.insert(
            f5,
            vec![(Integer::from(641), 1), (Integer::from(6700417), 1)],
        );
    }

    /// Add perfect powers (n^k for small n and k)
    fn add_perfect_powers(&mut self) {
        for base in 2..=20 {
            for exp in 2..=10 {
                let mut num = Integer::one();
                let base_int = Integer::from(base);
                for _ in 0..exp {
                    num = &num * &base_int;
                }
                if num < Integer::from(1_000_000) {
                    let factors = factor(&num);
                    self.factorizations.insert(num, factors);
                }
            }
        }
    }

    /// Add factorials and their products
    fn add_factorial_products(&mut self) {

        // Factorials up to 20!
        let mut fact = Integer::one();
        for n in 1..=20 {
            fact = fact * Integer::from(n);
            if !self.factorizations.contains_key(&fact) {
                let factors = factor(&fact);
                self.factorizations.insert(fact.clone(), factors);
            }
        }
    }

    /// Get factorization for a number
    pub fn get(&self, n: &Integer) -> Option<&Factorization> {
        self.factorizations.get(n)
    }

    /// Check if a number has a precomputed factorization
    pub fn contains(&self, n: &Integer) -> bool {
        self.factorizations.contains_key(n)
    }

    /// Get the number of entries in the database
    pub fn len(&self) -> usize {
        self.factorizations.len()
    }

    /// Check if the database is empty
    pub fn is_empty(&self) -> bool {
        self.factorizations.is_empty()
    }
}

impl Default for FactorizationDatabase {
    fn default() -> Self {
        Self::new()
    }
}

/// Global factorization database
static FACTORIZATION_DB: Lazy<FactorizationDatabase> =
    Lazy::new(FactorizationDatabase::new);

/// Get precomputed factorization for a number
pub fn get_factorization(n: &Integer) -> Option<&'static Factorization> {
    FACTORIZATION_DB.get(n)
}

/// Check if a number has a precomputed factorization
pub fn is_factored(n: &Integer) -> bool {
    FACTORIZATION_DB.contains(n)
}

/// Convert factorization to string representation
pub fn factorization_to_string(fact: &Factorization) -> String {
    if fact.is_empty() {
        return String::from("1");
    }

    fact.iter()
        .map(|(p, e)| {
            if *e == 1 {
                p.to_string()
            } else {
                format!("{}^{}", p, e)
            }
        })
        .collect::<Vec<_>>()
        .join(" × ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorization_database() {
        let db = FactorizationDatabase::new();

        // Check some basic factorizations
        assert!(db.contains(&Integer::from(12)));
        assert!(db.contains(&Integer::from(100)));

        // 12 = 2^2 × 3
        let fact_12 = db.get(&Integer::from(12)).unwrap();
        assert_eq!(fact_12.len(), 2);
        assert_eq!(fact_12[0], (Integer::from(2), 2));
        assert_eq!(fact_12[1], (Integer::from(3), 1));
    }

    #[test]
    fn test_mersenne_factorizations() {
        let db = FactorizationDatabase::new();

        // M_3 = 2^3 - 1 = 7 (prime)
        let m3 = Integer::from(7);
        let fact = db.get(&m3).unwrap();
        assert_eq!(fact.len(), 1);
        assert_eq!(fact[0].0, Integer::from(7));

        // M_11 = 2^11 - 1 = 2047 = 23 × 89
        let m11 = Integer::from(2047);
        let fact = db.get(&m11).unwrap();
        assert_eq!(fact.len(), 2);
    }

    #[test]
    fn test_fermat_numbers() {
        let db = FactorizationDatabase::new();

        // F_0 = 3 (prime)
        assert!(db.contains(&Integer::from(3)));

        // F_4 = 65537 (prime)
        assert!(db.contains(&Integer::from(65537)));

        // F_5 = 641 × 6700417 (composite)
        let f5 = (Integer::from(2u64).pow(32)) + Integer::one();
        let fact = db.get(&f5).unwrap();
        assert_eq!(fact.len(), 2);
    }

    #[test]
    fn test_global_factorization() {
        // Test global access
        let fact_100 = get_factorization(&Integer::from(100));
        assert!(fact_100.is_some());

        // 100 = 2^2 × 5^2
        let fact = fact_100.unwrap();
        assert_eq!(fact.len(), 2);
        assert_eq!(fact[0], (Integer::from(2), 2));
        assert_eq!(fact[1], (Integer::from(5), 2));

        assert!(is_factored(&Integer::from(42)));
    }

    #[test]
    fn test_factorization_to_string() {
        let fact = vec![
            (Integer::from(2), 3),
            (Integer::from(3), 2),
            (Integer::from(5), 1),
        ];
        let s = factorization_to_string(&fact);
        assert_eq!(s, "2^3 × 3^2 × 5");

        let fact_prime = vec![(Integer::from(17), 1)];
        let s = factorization_to_string(&fact_prime);
        assert_eq!(s, "17");
    }
}
