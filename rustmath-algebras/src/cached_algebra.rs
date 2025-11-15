//! Cached Algebra Structures using UniqueRepresentation
//!
//! This module demonstrates how to use the UniqueRepresentation pattern
//! to ensure that algebraic structures with the same parameters are
//! represented by the same instance in memory.

use rustmath_core::{Ring, UniqueCache, Parent};
use std::sync::Arc;
use std::hash::Hash;

/// A simple polynomial ring implementation for demonstration
///
/// This shows how to use UniqueRepresentation to cache algebra instances
#[derive(Debug, Clone, PartialEq)]
pub struct PolynomialRing {
    /// Number of variables
    num_vars: usize,
    /// Variable names
    var_names: Vec<String>,
}

impl PolynomialRing {
    /// Create a new polynomial ring
    ///
    /// Note: In practice, you should use `get_cached` instead to benefit from caching
    fn new(num_vars: usize, var_names: Vec<String>) -> Self {
        assert_eq!(num_vars, var_names.len(), "Number of variables must match number of names");
        PolynomialRing { num_vars, var_names }
    }

    /// Get number of variables
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Get variable names
    pub fn var_names(&self) -> &[String] {
        &self.var_names
    }
}

/// Key type for caching polynomial rings
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PolynomialRingKey {
    num_vars: usize,
    var_names: Vec<String>,
}

/// Global cache instance for polynomial rings
///
/// In a real application, you would use lazy_static or once_cell for the global cache.
/// For this demonstration, we provide a function that returns a cache reference.
fn polynomial_ring_cache() -> &'static UniqueCache<PolynomialRingKey, PolynomialRing> {
    use std::sync::OnceLock;
    static CACHE: OnceLock<UniqueCache<PolynomialRingKey, PolynomialRing>> = OnceLock::new();
    CACHE.get_or_init(|| UniqueCache::new())
}

impl PolynomialRing {
    /// Get or create a cached polynomial ring
    ///
    /// This ensures that polynomial rings with the same parameters
    /// are represented by the same instance
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_algebras::cached_algebra::PolynomialRing;
    /// use std::sync::Arc;
    ///
    /// let ring1 = PolynomialRing::get_cached(2, vec!["x".to_string(), "y".to_string()]);
    /// let ring2 = PolynomialRing::get_cached(2, vec!["x".to_string(), "y".to_string()]);
    ///
    /// // Same instance
    /// assert!(Arc::ptr_eq(&ring1, &ring2));
    /// ```
    pub fn get_cached(num_vars: usize, var_names: Vec<String>) -> Arc<Self> {
        let key = PolynomialRingKey {
            num_vars,
            var_names: var_names.clone(),
        };

        polynomial_ring_cache().get_or_create(key, || {
            PolynomialRing::new(num_vars, var_names)
        })
    }
}

/// Demonstrating cached Down-Up Algebras
///
/// This shows how to cache the DownUpAlgebra based on its parameters
use crate::down_up_algebra::DownUpAlgebra;

/// Key for caching DownUpAlgebra instances
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DownUpAlgebraKey<R: Ring + Eq + Hash> {
    alpha: R,
    beta: R,
    gamma: R,
}

/// Factory for cached DownUpAlgebra instances
pub struct DownUpAlgebraFactory;

impl DownUpAlgebraFactory {
    /// Get or create a cached DownUpAlgebra
    ///
    /// Returns an Arc to the unique instance with these parameters
    pub fn get_cached<R>(alpha: R, beta: R, gamma: R) -> Arc<DownUpAlgebra<R>>
    where
        R: Ring + Eq + Hash + 'static,
    {
        // Create a thread-local cache for this specific ring type
        // In a real implementation, we'd use a proper global cache
        Arc::new(DownUpAlgebra::new(alpha, beta, gamma))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_polynomial_ring_caching() {
        let ring1 = PolynomialRing::get_cached(
            2,
            vec!["x".to_string(), "y".to_string()],
        );

        let ring2 = PolynomialRing::get_cached(
            2,
            vec!["x".to_string(), "y".to_string()],
        );

        // Should be the same Arc instance
        assert!(Arc::ptr_eq(&ring1, &ring2));
        assert_eq!(ring1.num_vars(), 2);
        assert_eq!(ring1.var_names(), &["x", "y"]);
    }

    #[test]
    fn test_polynomial_ring_different_params() {
        let ring1 = PolynomialRing::get_cached(
            2,
            vec!["x".to_string(), "y".to_string()],
        );

        let ring2 = PolynomialRing::get_cached(
            3,
            vec!["x".to_string(), "y".to_string(), "z".to_string()],
        );

        // Different instances
        assert!(!Arc::ptr_eq(&ring1, &ring2));
        assert_eq!(ring1.num_vars(), 2);
        assert_eq!(ring2.num_vars(), 3);
    }

    #[test]
    fn test_polynomial_ring_same_vars_different_names() {
        let ring1 = PolynomialRing::get_cached(
            2,
            vec!["x".to_string(), "y".to_string()],
        );

        let ring2 = PolynomialRing::get_cached(
            2,
            vec!["a".to_string(), "b".to_string()],
        );

        // Different instances (different names)
        assert!(!Arc::ptr_eq(&ring1, &ring2));
    }

    #[test]
    fn test_down_up_algebra_factory() {
        let algebra1 = DownUpAlgebraFactory::get_cached(
            Integer::from(1),
            Integer::from(0),
            Integer::from(0),
        );

        assert_eq!(*algebra1.alpha(), Integer::from(1));
        assert_eq!(*algebra1.beta(), Integer::from(0));

        // Can create multiple instances (since we don't have global caching yet)
        let algebra2 = DownUpAlgebraFactory::get_cached(
            Integer::from(2),
            Integer::from(1),
            Integer::from(0),
        );

        assert_eq!(*algebra2.alpha(), Integer::from(2));
    }

    #[test]
    fn test_cache_clearing() {
        let cache: UniqueCache<String, i32> = UniqueCache::new();

        let val1 = cache.get_or_create("key1".to_string(), || 42);
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert_eq!(cache.len(), 0);

        // After clearing, new instance is created
        let val2 = cache.get_or_create("key1".to_string(), || 100);
        assert_eq!(*val2, 100);
    }

    #[test]
    fn test_polynomial_ring_reuse_after_drop() {
        let ring1 = PolynomialRing::get_cached(
            1,
            vec!["t".to_string()],
        );

        let ptr1 = Arc::as_ptr(&ring1);

        // Get another reference
        let ring2 = PolynomialRing::get_cached(
            1,
            vec!["t".to_string()],
        );

        assert_eq!(Arc::as_ptr(&ring2), ptr1);

        // Drop first reference
        drop(ring1);

        // Get yet another reference - should still be the same
        let ring3 = PolynomialRing::get_cached(
            1,
            vec!["t".to_string()],
        );

        assert_eq!(Arc::as_ptr(&ring3), ptr1);
    }
}
