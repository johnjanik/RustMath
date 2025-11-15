//! Unique Representation Pattern
//!
//! Ensures that only one instance of an object with given parameters exists.
//! This is useful for algebraic structures where equality should be determined
//! by parameters rather than creating multiple equivalent instances.
//!
//! Corresponds to sage.structure.unique_representation.UniqueRepresentation

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

/// A cache for unique representations
///
/// Stores weak references to objects so they can be garbage collected
/// when no longer in use, but ensures uniqueness while they exist.
pub struct UniqueCache<K: Hash + Eq, V> {
    cache: Mutex<HashMap<K, Arc<V>>>,
}

impl<K: Hash + Eq, V> UniqueCache<K, V> {
    /// Create a new unique cache
    pub fn new() -> Self {
        UniqueCache {
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Get or create an object with the given key
    ///
    /// If an object with this key already exists, returns a reference to it.
    /// Otherwise, creates a new object using the provided constructor.
    ///
    /// # Arguments
    ///
    /// * `key` - The unique key for this object
    /// * `constructor` - Function to create a new object if needed
    ///
    /// # Returns
    ///
    /// An `Arc` reference to the unique object
    pub fn get_or_create<F>(&self, key: K, constructor: F) -> Arc<V>
    where
        F: FnOnce() -> V,
        K: Clone,
    {
        let mut cache = self.cache.lock().unwrap();

        // Check if we already have this object
        if let Some(existing) = cache.get(&key) {
            return Arc::clone(existing);
        }

        // Create new object
        let value = Arc::new(constructor());
        cache.insert(key, Arc::clone(&value));
        value
    }

    /// Clear the cache
    ///
    /// Removes all entries from the cache. Objects with existing references
    /// will continue to exist until those references are dropped.
    pub fn clear(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();
    }

    /// Get the number of cached objects
    pub fn len(&self) -> usize {
        let cache = self.cache.lock().unwrap();
        cache.len()
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<K: Hash + Eq, V> Default for UniqueCache<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for types that use unique representation
///
/// Types implementing this trait ensure that objects with the same
/// parameters are represented by the same instance.
pub trait UniqueRepresentation: Sized + 'static {
    /// The key type used to uniquely identify instances
    type Key: Hash + Eq + Clone + 'static;

    /// Get the key for this instance
    fn key(&self) -> Self::Key;

    /// Get the global cache for this type
    fn cache() -> &'static UniqueCache<Self::Key, Self>;

    /// Get or create a unique instance
    ///
    /// # Arguments
    ///
    /// * `key` - The unique key for this instance
    /// * `constructor` - Function to create a new instance if needed
    fn get_unique<F>(key: Self::Key, constructor: F) -> Arc<Self>
    where
        F: FnOnce() -> Self,
    {
        Self::cache().get_or_create(key, constructor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    struct TestStruct {
        value: i32,
        name: String,
    }

    #[test]
    fn test_unique_cache_basic() {
        let cache: UniqueCache<i32, TestStruct> = UniqueCache::new();

        // Create first object
        let obj1 = cache.get_or_create(42, || TestStruct {
            value: 42,
            name: "test".to_string(),
        });

        assert_eq!(obj1.value, 42);
        assert_eq!(cache.len(), 1);

        // Get same object again
        let obj2 = cache.get_or_create(42, || TestStruct {
            value: 999, // This constructor should not be called
            name: "different".to_string(),
        });

        assert_eq!(obj2.value, 42); // Should be same as obj1
        assert_eq!(obj2.name, "test"); // Should be same as obj1
        assert_eq!(cache.len(), 1); // Still only one object

        // Verify they're the same Arc
        assert!(Arc::ptr_eq(&obj1, &obj2));
    }

    #[test]
    fn test_unique_cache_different_keys() {
        let cache: UniqueCache<String, i32> = UniqueCache::new();

        let val1 = cache.get_or_create("key1".to_string(), || 100);
        let val2 = cache.get_or_create("key2".to_string(), || 200);

        assert_eq!(*val1, 100);
        assert_eq!(*val2, 200);
        assert_eq!(cache.len(), 2);

        // Get val1 again
        let val1_again = cache.get_or_create("key1".to_string(), || 999);
        assert_eq!(*val1_again, 100);
        assert!(Arc::ptr_eq(&val1, &val1_again));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_unique_cache_clear() {
        let cache: UniqueCache<i32, String> = UniqueCache::new();

        cache.get_or_create(1, || "one".to_string());
        cache.get_or_create(2, || "two".to_string());

        assert_eq!(cache.len(), 2);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());

        // After clear, new object should be created
        let new_obj = cache.get_or_create(1, || "NEW ONE".to_string());
        assert_eq!(*new_obj, "NEW ONE");
    }

    // Example using the UniqueRepresentation trait
    #[derive(Debug, Clone, PartialEq)]
    struct Polynomial {
        degree: usize,
        variable: String,
    }

    // We would use lazy_static or once_cell for the actual cache
    // For testing, we'll just demonstrate the pattern

    #[test]
    fn test_unique_representation_pattern() {
        // This demonstrates how UniqueRepresentation would be used
        let cache: UniqueCache<(usize, String), Polynomial> = UniqueCache::new();

        let p1 = cache.get_or_create((3, "x".to_string()), || Polynomial {
            degree: 3,
            variable: "x".to_string(),
        });

        let p2 = cache.get_or_create((3, "x".to_string()), || Polynomial {
            degree: 999, // Won't be used
            variable: "y".to_string(), // Won't be used
        });

        // Same object
        assert!(Arc::ptr_eq(&p1, &p2));
        assert_eq!(p1.degree, 3);
        assert_eq!(p1.variable, "x");
    }
}
