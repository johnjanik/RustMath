//! Power Computer for p-adic Numbers
//!
//! This module provides efficient caching of powers of primes for p-adic arithmetic.
//! It's inspired by SageMath's `sage.rings.padics.pow_computer` module.
//!
//! ## Overview
//!
//! The `PowComputer` caches powers of a prime p (i.e., 1, p, p², p³, ..., p^cache_limit)
//! to enable fast modular reduction and other p-adic operations. This is crucial for
//! performance when working with p-adic numbers at high precision.
//!
//! ## Cache Invalidation Strategy
//!
//! The cache is **immutable** after creation. Cache invalidation is achieved through
//! one of the following strategies:
//!
//! 1. **Reconstruction**: Create a new `PowComputer` with different parameters
//! 2. **Extension**: Use `extend_cache()` to create a new computer with a higher limit
//! 3. **Scope-based**: Let the `PowComputer` drop when no longer needed
//!
//! The immutable design ensures thread-safety and predictable behavior in concurrent
//! contexts. If you need different cache parameters, create a new `PowComputer` instance.
//!
//! ## Examples
//!
//! ```
//! use rustmath_integers::Integer;
//! use rustmath_rings::padics::PowComputer;
//!
//! // Create a power computer for prime p=5, caching up to p^10
//! let pc = PowComputer::new(Integer::from(5), 10);
//!
//! // Get p^3 = 125
//! let p_cubed = pc.pow(3);
//! assert_eq!(p_cubed, Integer::from(125));
//!
//! // Modular reduction mod p^7
//! let value = Integer::from(123456);
//! let reduced = pc.reduce(value, 7);
//! ```

use rustmath_integers::Integer;
use std::sync::Arc;

/// Power computer for caching powers of a prime p
///
/// This struct efficiently caches powers of a prime: 1, p, p², ..., p^cache_limit.
/// It provides fast access to these powers for p-adic arithmetic operations.
///
/// The cache is immutable after creation, ensuring thread-safety. To change cache
/// parameters, create a new `PowComputer` instance.
#[derive(Clone, Debug)]
pub struct PowComputer {
    /// The prime base
    prime: Integer,

    /// Cached powers: powers[i] = p^i for i = 0..=cache_limit
    powers: Arc<Vec<Integer>>,

    /// Maximum power cached (cache contains p^0 through p^cache_limit)
    cache_limit: usize,

    /// Whether this is for a p-adic field (vs ring)
    in_field: bool,
}

impl PowComputer {
    /// Create a new power computer
    ///
    /// # Arguments
    ///
    /// * `prime` - The prime p to cache powers of
    /// * `cache_limit` - Cache powers up to p^cache_limit
    ///
    /// # Panics
    ///
    /// Panics if prime ≤ 1
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_integers::Integer;
    /// use rustmath_rings::padics::PowComputer;
    ///
    /// let pc = PowComputer::new(Integer::from(5), 10);
    /// assert_eq!(pc.cache_limit(), 10);
    /// ```
    pub fn new(prime: Integer, cache_limit: usize) -> Self {
        Self::with_field_flag(prime, cache_limit, false)
    }

    /// Create a new power computer for a p-adic field
    ///
    /// # Arguments
    ///
    /// * `prime` - The prime p
    /// * `cache_limit` - Cache powers up to p^cache_limit
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_integers::Integer;
    /// use rustmath_rings::padics::PowComputer;
    ///
    /// let pc = PowComputer::for_field(Integer::from(5), 10);
    /// assert!(pc.is_field());
    /// ```
    pub fn for_field(prime: Integer, cache_limit: usize) -> Self {
        Self::with_field_flag(prime, cache_limit, true)
    }

    /// Create a power computer with explicit field flag
    fn with_field_flag(prime: Integer, cache_limit: usize, in_field: bool) -> Self {
        assert!(prime > Integer::one(), "Prime must be > 1");

        // Pre-compute all powers up to cache_limit
        let mut powers = Vec::with_capacity(cache_limit + 1);
        powers.push(Integer::one()); // p^0 = 1

        for i in 1..=cache_limit {
            let prev = &powers[i - 1];
            powers.push(prev.clone() * prime.clone());
        }

        PowComputer {
            prime,
            powers: Arc::new(powers),
            cache_limit,
            in_field,
        }
    }

    /// Get the prime p
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_integers::Integer;
    /// use rustmath_rings::padics::PowComputer;
    ///
    /// let pc = PowComputer::new(Integer::from(7), 5);
    /// assert_eq!(pc.prime(), &Integer::from(7));
    /// ```
    pub fn prime(&self) -> &Integer {
        &self.prime
    }

    /// Get the cache limit
    ///
    /// The cache contains powers p^0 through p^cache_limit.
    pub fn cache_limit(&self) -> usize {
        self.cache_limit
    }

    /// Check if this is for a p-adic field
    pub fn is_field(&self) -> bool {
        self.in_field
    }

    /// Get p^n, using cache if available
    ///
    /// # Arguments
    ///
    /// * `n` - Exponent
    ///
    /// # Panics
    ///
    /// Panics if n is negative
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_integers::Integer;
    /// use rustmath_rings::padics::PowComputer;
    ///
    /// let pc = PowComputer::new(Integer::from(3), 10);
    /// assert_eq!(pc.pow(4), Integer::from(81)); // 3^4
    /// ```
    pub fn pow(&self, n: usize) -> Integer {
        if n <= self.cache_limit {
            // Use cached value
            self.powers[n].clone()
        } else {
            // Compute on the fly using binary exponentiation
            self.prime.pow(n as u32)
        }
    }

    /// Get p^n as a reference (only works if cached)
    ///
    /// Returns `None` if n > cache_limit
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_integers::Integer;
    /// use rustmath_rings::padics::PowComputer;
    ///
    /// let pc = PowComputer::new(Integer::from(5), 10);
    /// assert_eq!(pc.pow_ref(3), Some(&Integer::from(125)));
    /// assert_eq!(pc.pow_ref(20), None);
    /// ```
    pub fn pow_ref(&self, n: usize) -> Option<&Integer> {
        if n <= self.cache_limit {
            Some(&self.powers[n])
        } else {
            None
        }
    }

    /// Reduce a value modulo p^n
    ///
    /// Efficiently computes value mod p^n, using cached powers when available.
    ///
    /// # Arguments
    ///
    /// * `value` - Value to reduce
    /// * `n` - Precision (reduce mod p^n)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_integers::Integer;
    /// use rustmath_rings::padics::PowComputer;
    ///
    /// let pc = PowComputer::new(Integer::from(5), 10);
    /// let result = pc.reduce(Integer::from(777), 3);
    /// // 777 mod 125 = 27
    /// assert_eq!(result, Integer::from(27));
    /// ```
    pub fn reduce(&self, value: Integer, n: usize) -> Integer {
        let modulus = self.pow(n);
        let reduced = value % modulus.clone();

        // Ensure result is in [0, modulus)
        if reduced.signum() < 0 {
            reduced + modulus
        } else {
            reduced
        }
    }

    /// Reduce a value in-place modulo p^n
    ///
    /// More efficient than `reduce()` when you don't need to keep the original value.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_integers::Integer;
    /// use rustmath_rings::padics::PowComputer;
    ///
    /// let pc = PowComputer::new(Integer::from(5), 10);
    /// let mut value = Integer::from(777);
    /// pc.reduce_in_place(&mut value, 3);
    /// assert_eq!(value, Integer::from(27));
    /// ```
    pub fn reduce_in_place(&self, value: &mut Integer, n: usize) {
        let modulus = self.pow(n);
        *value = value.clone() % modulus.clone();

        if value.signum() < 0 {
            *value = value.clone() + modulus;
        }
    }

    /// Compute the p-adic valuation of an integer
    ///
    /// Returns the largest k such that p^k divides value.
    /// Returns `None` for zero (which has infinite valuation).
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_integers::Integer;
    /// use rustmath_rings::padics::PowComputer;
    ///
    /// let pc = PowComputer::new(Integer::from(5), 10);
    /// assert_eq!(pc.valuation(&Integer::from(125)), Some(3)); // 5^3
    /// assert_eq!(pc.valuation(&Integer::from(7)), Some(0));
    /// assert_eq!(pc.valuation(&Integer::from(0)), None);
    /// ```
    pub fn valuation(&self, value: &Integer) -> Option<u32> {
        if value.is_zero() {
            None
        } else {
            Some(value.valuation(&self.prime))
        }
    }

    /// Create a new power computer with extended cache
    ///
    /// Returns a new `PowComputer` with cache_limit increased to `new_limit`.
    /// If `new_limit` ≤ current limit, returns a clone of self.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_integers::Integer;
    /// use rustmath_rings::padics::PowComputer;
    ///
    /// let pc = PowComputer::new(Integer::from(5), 10);
    /// let extended = pc.extend_cache(20);
    /// assert_eq!(extended.cache_limit(), 20);
    /// ```
    pub fn extend_cache(&self, new_limit: usize) -> Self {
        if new_limit <= self.cache_limit {
            return self.clone();
        }

        Self::with_field_flag(self.prime.clone(), new_limit, self.in_field)
    }

    /// Check if a power is cached
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_integers::Integer;
    /// use rustmath_rings::padics::PowComputer;
    ///
    /// let pc = PowComputer::new(Integer::from(5), 10);
    /// assert!(pc.is_cached(5));
    /// assert!(!pc.is_cached(15));
    /// ```
    pub fn is_cached(&self, n: usize) -> bool {
        n <= self.cache_limit
    }

    /// Get the memory usage of the cache in bytes (approximate)
    ///
    /// This is useful for monitoring memory consumption of p-adic computations.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_integers::Integer;
    /// use rustmath_rings::padics::PowComputer;
    ///
    /// let pc = PowComputer::new(Integer::from(5), 100);
    /// let memory = pc.cache_memory_usage();
    /// assert!(memory > 0);
    /// ```
    pub fn cache_memory_usage(&self) -> usize {
        // Rough estimate: size of Vec + size of each Integer
        let vec_overhead = std::mem::size_of::<Vec<Integer>>();
        let integer_overhead = std::mem::size_of::<Integer>() * self.powers.len();

        // Add approximate size of the integer data (number of digits)
        let data_size: usize = self.powers.iter()
            .map(|i| i.to_string().len() * std::mem::size_of::<u8>())
            .sum();

        vec_overhead + integer_overhead + data_size
    }
}

// Implement PartialEq for testing
impl PartialEq for PowComputer {
    fn eq(&self, other: &Self) -> bool {
        self.prime == other.prime
            && self.cache_limit == other.cache_limit
            && self.in_field == other.in_field
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_creation() {
        let pc = PowComputer::new(Integer::from(5), 10);
        assert_eq!(pc.prime(), &Integer::from(5));
        assert_eq!(pc.cache_limit(), 10);
        assert!(!pc.is_field());
    }

    #[test]
    fn test_field_creation() {
        let pc = PowComputer::for_field(Integer::from(7), 5);
        assert!(pc.is_field());
    }

    #[test]
    fn test_cached_powers() {
        let pc = PowComputer::new(Integer::from(3), 5);

        assert_eq!(pc.pow(0), Integer::from(1));
        assert_eq!(pc.pow(1), Integer::from(3));
        assert_eq!(pc.pow(2), Integer::from(9));
        assert_eq!(pc.pow(3), Integer::from(27));
        assert_eq!(pc.pow(4), Integer::from(81));
        assert_eq!(pc.pow(5), Integer::from(243));
    }

    #[test]
    fn test_uncached_powers() {
        let pc = PowComputer::new(Integer::from(2), 5);

        // Request power beyond cache
        assert_eq!(pc.pow(10), Integer::from(1024));
        assert_eq!(pc.pow(20), Integer::from(1048576));
    }

    #[test]
    fn test_pow_ref() {
        let pc = PowComputer::new(Integer::from(5), 10);

        assert_eq!(pc.pow_ref(3), Some(&Integer::from(125)));
        assert_eq!(pc.pow_ref(0), Some(&Integer::from(1)));
        assert_eq!(pc.pow_ref(15), None);
    }

    #[test]
    fn test_reduce() {
        let pc = PowComputer::new(Integer::from(5), 10);

        // 777 mod 125 = 27
        assert_eq!(pc.reduce(Integer::from(777), 3), Integer::from(27));

        // Test with negative values
        let neg = Integer::from(-123);
        let reduced = pc.reduce(neg, 3);
        assert!(reduced >= Integer::zero());
        assert!(reduced < Integer::from(125));
    }

    #[test]
    fn test_reduce_in_place() {
        let pc = PowComputer::new(Integer::from(7), 5);

        let mut value = Integer::from(1000);
        pc.reduce_in_place(&mut value, 3);
        assert_eq!(value, Integer::from(1000) % Integer::from(343));
    }

    #[test]
    fn test_valuation() {
        let pc = PowComputer::new(Integer::from(5), 10);

        assert_eq!(pc.valuation(&Integer::from(125)), Some(3)); // 5^3
        assert_eq!(pc.valuation(&Integer::from(25)), Some(2));  // 5^2
        assert_eq!(pc.valuation(&Integer::from(5)), Some(1));   // 5^1
        assert_eq!(pc.valuation(&Integer::from(1)), Some(0));   // no factors
        assert_eq!(pc.valuation(&Integer::from(7)), Some(0));   // coprime to 5
        assert_eq!(pc.valuation(&Integer::from(0)), None);      // infinite
    }

    #[test]
    fn test_extend_cache() {
        let pc = PowComputer::new(Integer::from(3), 5);
        let extended = pc.extend_cache(10);

        assert_eq!(extended.cache_limit(), 10);
        assert_eq!(extended.prime(), &Integer::from(3));

        // Can access newly cached values
        assert!(extended.is_cached(8));
    }

    #[test]
    fn test_extend_cache_no_op() {
        let pc = PowComputer::new(Integer::from(5), 10);
        let extended = pc.extend_cache(5);

        // Should return clone if new_limit <= current
        assert_eq!(extended.cache_limit(), 10);
    }

    #[test]
    fn test_is_cached() {
        let pc = PowComputer::new(Integer::from(7), 8);

        assert!(pc.is_cached(0));
        assert!(pc.is_cached(5));
        assert!(pc.is_cached(8));
        assert!(!pc.is_cached(9));
        assert!(!pc.is_cached(100));
    }

    #[test]
    fn test_cache_memory_usage() {
        let pc = PowComputer::new(Integer::from(2), 100);
        let memory = pc.cache_memory_usage();

        // Should have some non-zero memory usage
        assert!(memory > 0);

        // Larger cache should use more memory
        let pc_small = PowComputer::new(Integer::from(2), 10);
        assert!(pc.cache_memory_usage() > pc_small.cache_memory_usage());
    }

    #[test]
    fn test_large_prime() {
        let large_prime = Integer::from(104729); // 10000th prime
        let pc = PowComputer::new(large_prime.clone(), 5);

        assert_eq!(pc.pow(1), large_prime);
        assert_eq!(pc.pow(2), large_prime.clone() * large_prime.clone());
    }

    #[test]
    #[should_panic(expected = "Prime must be > 1")]
    fn test_invalid_prime() {
        PowComputer::new(Integer::from(1), 10);
    }

    #[test]
    fn test_clone() {
        let pc1 = PowComputer::new(Integer::from(5), 10);
        let pc2 = pc1.clone();

        assert_eq!(pc1, pc2);
        assert_eq!(pc2.pow(5), Integer::from(3125));
    }
}
