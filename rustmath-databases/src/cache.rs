//! Generic caching layer for database queries
//!
//! Provides in-memory and persistent caching with TTL (time-to-live) support.
//!
//! # Example
//!
//! ```
//! use rustmath_databases::cache::{Cache, CacheConfig};
//! use std::time::Duration;
//!
//! let config = CacheConfig {
//!     max_size: 1000,
//!     ttl: Some(Duration::from_secs(3600)),
//!     persist_to_disk: false,
//!     cache_dir: None,
//! };
//!
//! let mut cache = Cache::new(config);
//!
//! // Store a value
//! cache.set("key1".to_string(), "value1".to_string());
//!
//! // Retrieve a value
//! if let Some(value) = cache.get("key1") {
//!     println!("Found: {}", value);
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

/// Cache entry with expiration time
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntry<T> {
    /// The cached value
    value: T,
    /// Timestamp when the entry was created
    created_at: SystemTime,
    /// Timestamp when the entry expires (if TTL is set)
    expires_at: Option<SystemTime>,
}

impl<T> CacheEntry<T> {
    /// Create a new cache entry
    fn new(value: T, ttl: Option<Duration>) -> Self {
        let created_at = SystemTime::now();
        let expires_at = ttl.map(|d| created_at + d);

        CacheEntry {
            value,
            created_at,
            expires_at,
        }
    }

    /// Check if the entry has expired
    fn is_expired(&self) -> bool {
        if let Some(expires) = self.expires_at {
            SystemTime::now() > expires
        } else {
            false
        }
    }
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries to store
    pub max_size: usize,
    /// Time-to-live for cache entries (None = no expiration)
    pub ttl: Option<Duration>,
    /// Whether to persist cache to disk
    pub persist_to_disk: bool,
    /// Directory for persistent cache (if enabled)
    pub cache_dir: Option<PathBuf>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        CacheConfig {
            max_size: 10000,
            ttl: Some(Duration::from_secs(3600)),  // 1 hour default
            persist_to_disk: false,
            cache_dir: None,
        }
    }
}

/// Generic in-memory cache with TTL support
///
/// Stores key-value pairs with optional expiration and disk persistence.
#[derive(Debug)]
pub struct Cache<T>
where
    T: Clone + Serialize + for<'de> Deserialize<'de>,
{
    /// In-memory storage
    entries: HashMap<String, CacheEntry<T>>,
    /// Cache configuration
    config: CacheConfig,
    /// Statistics
    hits: u64,
    misses: u64,
}

impl<T> Cache<T>
where
    T: Clone + Serialize + for<'de> Deserialize<'de>,
{
    /// Create a new cache with the given configuration
    pub fn new(config: CacheConfig) -> Self {
        let mut cache = Cache {
            entries: HashMap::new(),
            config,
            hits: 0,
            misses: 0,
        };

        // Load from disk if persistence is enabled
        if cache.config.persist_to_disk {
            cache.load_from_disk();
        }

        cache
    }

    /// Create a new cache with default configuration
    pub fn default() -> Self {
        Self::new(CacheConfig::default())
    }

    /// Get a value from the cache
    ///
    /// Returns None if the key doesn't exist or the entry has expired
    pub fn get(&mut self, key: &str) -> Option<T> {
        if let Some(entry) = self.entries.get(key) {
            if entry.is_expired() {
                // Entry has expired, remove it
                self.entries.remove(key);
                self.misses += 1;
                None
            } else {
                self.hits += 1;
                Some(entry.value.clone())
            }
        } else {
            self.misses += 1;
            None
        }
    }

    /// Store a value in the cache
    ///
    /// If the cache is full, removes the oldest entry first
    pub fn set(&mut self, key: String, value: T) {
        // Check if we need to make room
        if self.entries.len() >= self.config.max_size && !self.entries.contains_key(&key) {
            self.evict_oldest();
        }

        let entry = CacheEntry::new(value, self.config.ttl);
        self.entries.insert(key, entry);

        // Persist to disk if enabled
        if self.config.persist_to_disk {
            self.save_to_disk();
        }
    }

    /// Remove a key from the cache
    pub fn remove(&mut self, key: &str) {
        self.entries.remove(key);

        if self.config.persist_to_disk {
            self.save_to_disk();
        }
    }

    /// Clear all entries from the cache
    pub fn clear(&mut self) {
        self.entries.clear();

        if self.config.persist_to_disk {
            self.save_to_disk();
        }
    }

    /// Remove expired entries from the cache
    pub fn cleanup_expired(&mut self) {
        let keys_to_remove: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.is_expired())
            .map(|(key, _)| key.clone())
            .collect();

        for key in keys_to_remove {
            self.entries.remove(&key);
        }

        if self.config.persist_to_disk && !self.entries.is_empty() {
            self.save_to_disk();
        }
    }

    /// Get the number of entries in the cache
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            size: self.entries.len(),
            hits: self.hits,
            misses: self.misses,
            hit_rate: self.hit_rate(),
        }
    }

    /// Evict the oldest entry from the cache
    fn evict_oldest(&mut self) {
        if let Some(oldest_key) = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.created_at)
            .map(|(key, _)| key.clone())
        {
            self.entries.remove(&oldest_key);
        }
    }

    /// Load cache from disk
    fn load_from_disk(&mut self) {
        if let Some(cache_dir) = &self.config.cache_dir {
            let cache_file = cache_dir.join("cache.json");
            if cache_file.exists() {
                if let Ok(contents) = std::fs::read_to_string(&cache_file) {
                    if let Ok(entries) = serde_json::from_str::<HashMap<String, CacheEntry<T>>>(&contents) {
                        self.entries = entries;
                        // Clean up expired entries after loading
                        self.cleanup_expired();
                    }
                }
            }
        }
    }

    /// Save cache to disk
    fn save_to_disk(&self) {
        if let Some(cache_dir) = &self.config.cache_dir {
            if let Err(_) = std::fs::create_dir_all(cache_dir) {
                return;
            }

            let cache_file = cache_dir.join("cache.json");
            if let Ok(contents) = serde_json::to_string(&self.entries) {
                let _ = std::fs::write(&cache_file, contents);
            }
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Current number of entries
    pub size: usize,
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Hit rate (0.0 to 1.0)
    pub hit_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic() {
        let config = CacheConfig {
            max_size: 10,
            ttl: None,
            persist_to_disk: false,
            cache_dir: None,
        };

        let mut cache = Cache::<String>::new(config);

        cache.set("key1".to_string(), "value1".to_string());
        cache.set("key2".to_string(), "value2".to_string());

        assert_eq!(cache.get("key1"), Some("value1".to_string()));
        assert_eq!(cache.get("key2"), Some("value2".to_string()));
        assert_eq!(cache.get("key3"), None);
    }

    #[test]
    fn test_cache_expiration() {
        let config = CacheConfig {
            max_size: 10,
            ttl: Some(Duration::from_millis(100)),
            persist_to_disk: false,
            cache_dir: None,
        };

        let mut cache = Cache::<String>::new(config);

        cache.set("key1".to_string(), "value1".to_string());

        // Should be available immediately
        assert_eq!(cache.get("key1"), Some("value1".to_string()));

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(150));

        // Should be expired now
        assert_eq!(cache.get("key1"), None);
    }

    #[test]
    fn test_cache_max_size() {
        let config = CacheConfig {
            max_size: 3,
            ttl: None,
            persist_to_disk: false,
            cache_dir: None,
        };

        let mut cache = Cache::<String>::new(config);

        cache.set("key1".to_string(), "value1".to_string());
        cache.set("key2".to_string(), "value2".to_string());
        cache.set("key3".to_string(), "value3".to_string());

        assert_eq!(cache.len(), 3);

        // Adding a 4th entry should evict the oldest
        cache.set("key4".to_string(), "value4".to_string());

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.get("key1"), None);  // key1 should be evicted
        assert_eq!(cache.get("key4"), Some("value4".to_string()));
    }

    #[test]
    fn test_cache_stats() {
        let mut cache = Cache::<String>::default();

        cache.set("key1".to_string(), "value1".to_string());

        cache.get("key1");  // hit
        cache.get("key2");  // miss
        cache.get("key1");  // hit

        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = Cache::<String>::default();

        cache.set("key1".to_string(), "value1".to_string());
        cache.set("key2".to_string(), "value2".to_string());

        assert_eq!(cache.len(), 2);

        cache.clear();

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_remove() {
        let mut cache = Cache::<String>::default();

        cache.set("key1".to_string(), "value1".to_string());
        cache.set("key2".to_string(), "value2".to_string());

        cache.remove("key1");

        assert_eq!(cache.get("key1"), None);
        assert_eq!(cache.get("key2"), Some("value2".to_string()));
    }

    #[test]
    fn test_cleanup_expired() {
        let config = CacheConfig {
            max_size: 10,
            ttl: Some(Duration::from_millis(100)),
            persist_to_disk: false,
            cache_dir: None,
        };

        let mut cache = Cache::<String>::new(config);

        cache.set("key1".to_string(), "value1".to_string());
        cache.set("key2".to_string(), "value2".to_string());

        assert_eq!(cache.len(), 2);

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(150));

        cache.cleanup_expired();

        assert_eq!(cache.len(), 0);
    }
}
