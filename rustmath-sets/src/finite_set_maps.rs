//! Finite Set Maps and Functions
//!
//! This module provides structures for working with functions between finite sets,
//! including enumerating all possible maps, checking properties like injectivity
//! and surjectivity, and working with specific types of functions.
//!
//! # Overview
//!
//! Corresponds to SageMath's `sage.sets.finite_set_maps` module.
//!
//! # Examples
//!
//! ```
//! use rustmath_sets::finite_set_maps::{FiniteSetMap, FiniteSetMaps};
//! use std::collections::HashMap;
//!
//! // Create a specific map from {1, 2, 3} to {"a", "b"}
//! let mut mapping = HashMap::new();
//! mapping.insert(1, "a");
//! mapping.insert(2, "b");
//! mapping.insert(3, "a");
//!
//! let map = FiniteSetMap::new(vec![1, 2, 3], vec!["a", "b"], mapping);
//!
//! // Check properties
//! assert!(!map.is_injective());
//! assert!(map.is_surjective());
//! ```

use std::collections::HashMap;
use std::hash::Hash;
use std::fmt;

/// A function (map) from a finite domain to a finite codomain.
///
/// Represents a mathematical function f: Domain → Codomain where both sets are finite.
/// The function is represented explicitly as a HashMap mapping domain elements to
/// codomain elements.
///
/// # Type Parameters
///
/// * `D` - Domain element type (must be `Eq + Hash + Clone`)
/// * `C` - Codomain element type (must be `Eq + Clone`)
///
/// # Examples
///
/// ```
/// use rustmath_sets::finite_set_maps::FiniteSetMap;
/// use std::collections::HashMap;
///
/// let domain = vec![1, 2, 3];
/// let codomain = vec!["a", "b", "c"];
/// let mut mapping = HashMap::new();
/// mapping.insert(1, "a");
/// mapping.insert(2, "b");
/// mapping.insert(3, "c");
///
/// let map = FiniteSetMap::new(domain, codomain, mapping);
/// assert_eq!(map.apply(&1), Some(&"a"));
/// assert!(map.is_bijective());
/// ```
#[derive(Debug, Clone)]
pub struct FiniteSetMap<D, C>
where
    D: Eq + Hash + Clone,
    C: Eq + Clone,
{
    domain: Vec<D>,
    codomain: Vec<C>,
    mapping: HashMap<D, C>,
}

impl<D, C> FiniteSetMap<D, C>
where
    D: Eq + Hash + Clone,
    C: Eq + Clone,
{
    /// Create a new finite set map.
    ///
    /// # Panics
    ///
    /// Panics if the mapping is not total (doesn't map every domain element).
    pub fn new(domain: Vec<D>, codomain: Vec<C>, mapping: HashMap<D, C>) -> Self {
        // Verify that the mapping is total
        for d in &domain {
            if !mapping.contains_key(d) {
                panic!("Mapping must be total: missing mapping for domain element");
            }
        }

        // Verify that all mapped values are in the codomain
        for c in mapping.values() {
            if !codomain.contains(c) {
                panic!("Mapped value not in codomain");
            }
        }

        Self {
            domain,
            codomain,
            mapping,
        }
    }

    /// Apply the function to a domain element.
    ///
    /// Returns `None` if the element is not in the domain.
    pub fn apply(&self, element: &D) -> Option<&C> {
        self.mapping.get(element)
    }

    /// Get the domain of this map.
    pub fn domain(&self) -> &[D] {
        &self.domain
    }

    /// Get the codomain of this map.
    pub fn codomain(&self) -> &[C] {
        &self.codomain
    }

    /// Get the underlying mapping.
    pub fn mapping(&self) -> &HashMap<D, C> {
        &self.mapping
    }

    /// Check if this map is injective (one-to-one).
    ///
    /// A function is injective if distinct domain elements map to distinct codomain elements.
    pub fn is_injective(&self) -> bool {
        let mut seen = Vec::new();
        for value in self.mapping.values() {
            if seen.contains(&value) {
                return false;
            }
            seen.push(value);
        }
        true
    }

    /// Check if this map is surjective (onto).
    ///
    /// A function is surjective if every codomain element is the image of at least
    /// one domain element.
    pub fn is_surjective(&self) -> bool {
        for c in &self.codomain {
            if !self.mapping.values().any(|v| v == c) {
                return false;
            }
        }
        true
    }

    /// Check if this map is bijective (one-to-one and onto).
    ///
    /// A function is bijective if it is both injective and surjective.
    pub fn is_bijective(&self) -> bool {
        self.is_injective() && self.is_surjective()
    }

    /// Get the image (range) of this map.
    ///
    /// Returns the set of all elements in the codomain that are mapped to by
    /// at least one element in the domain.
    pub fn image(&self) -> Vec<C> {
        let mut result = Vec::new();
        for value in self.mapping.values() {
            if !result.contains(value) {
                result.push(value.clone());
            }
        }
        result
    }

    /// Get the preimage of a codomain element.
    ///
    /// Returns all domain elements that map to the given codomain element.
    pub fn preimage(&self, element: &C) -> Vec<D> {
        self.mapping
            .iter()
            .filter_map(|(k, v)| if v == element { Some(k.clone()) } else { None })
            .collect()
    }

    /// Get the cardinality of the domain.
    pub fn domain_cardinality(&self) -> usize {
        self.domain.len()
    }

    /// Get the cardinality of the codomain.
    pub fn codomain_cardinality(&self) -> usize {
        self.codomain.len()
    }
}

impl<D, C> PartialEq for FiniteSetMap<D, C>
where
    D: Eq + Hash + Clone,
    C: Eq + Clone,
{
    fn eq(&self, other: &Self) -> bool {
        self.domain == other.domain
            && self.codomain == other.codomain
            && self.mapping == other.mapping
    }
}

impl<D, C> Eq for FiniteSetMap<D, C>
where
    D: Eq + Hash + Clone,
    C: Eq + Clone,
{
}

/// The set of all functions from a finite domain to a finite codomain.
///
/// This structure can enumerate all possible functions f: Domain → Codomain
/// where both sets are finite.
///
/// # Cardinality
///
/// If |Domain| = m and |Codomain| = n, then there are n^m total functions.
///
/// # Examples
///
/// ```
/// use rustmath_sets::finite_set_maps::FiniteSetMaps;
///
/// let domain = vec![1, 2];
/// let codomain = vec!["a", "b"];
/// let maps = FiniteSetMaps::new(domain, codomain);
///
/// assert_eq!(maps.cardinality(), Some(4)); // 2^2 = 4 functions
/// ```
#[derive(Debug, Clone)]
pub struct FiniteSetMaps<D, C>
where
    D: Eq + Hash + Clone,
    C: Eq + Clone,
{
    domain: Vec<D>,
    codomain: Vec<C>,
}

impl<D, C> FiniteSetMaps<D, C>
where
    D: Eq + Hash + Clone,
    C: Eq + Clone,
{
    /// Create a new finite set maps structure.
    ///
    /// This represents the set of all functions from `domain` to `codomain`.
    pub fn new(domain: Vec<D>, codomain: Vec<C>) -> Self {
        Self { domain, codomain }
    }

    /// Get the domain.
    pub fn domain(&self) -> &[D] {
        &self.domain
    }

    /// Get the codomain.
    pub fn codomain(&self) -> &[C] {
        &self.codomain
    }

    /// Get the cardinality (number of functions).
    ///
    /// Returns `codomain.len()^domain.len()`.
    /// Returns `None` if the result would overflow usize.
    pub fn cardinality(&self) -> Option<usize> {
        if self.domain.is_empty() {
            return Some(if self.codomain.is_empty() { 0 } else { 1 });
        }

        if self.codomain.is_empty() {
            return Some(0);
        }

        self.codomain.len().checked_pow(self.domain.len() as u32)
    }

    /// Check if the set of functions is empty.
    ///
    /// The set is empty if and only if the domain is non-empty and the codomain is empty.
    pub fn is_empty(&self) -> bool {
        !self.domain.is_empty() && self.codomain.is_empty()
    }

    /// Create an iterator over all functions.
    ///
    /// Returns `None` if there are no functions (domain is non-empty but codomain is empty).
    pub fn iter(&self) -> Option<FiniteSetMapsIterator<D, C>> {
        if self.is_empty() {
            return None;
        }

        Some(FiniteSetMapsIterator::new(
            self.domain.clone(),
            self.codomain.clone(),
        ))
    }

    /// Count the number of injective functions.
    ///
    /// If |Domain| = m and |Codomain| = n, the number of injective functions is:
    /// - 0 if m > n
    /// - n!/(n-m)! if m ≤ n (falling factorial)
    pub fn count_injective(&self) -> usize {
        let m = self.domain.len();
        let n = self.codomain.len();

        if m > n {
            return 0;
        }

        // Compute falling factorial: n * (n-1) * ... * (n-m+1)
        (0..m).map(|i| n - i).product()
    }

    /// Count the number of surjective functions.
    ///
    /// This uses the inclusion-exclusion principle. The formula is:
    /// Σ(k=0 to n) (-1)^k * C(n,k) * (n-k)^m
    ///
    /// where m = |Domain| and n = |Codomain|.
    pub fn count_surjective(&self) -> i64 {
        let m = self.domain.len() as i64;
        let n = self.codomain.len() as i64;

        if m < n {
            return 0;
        }

        let mut count = 0i64;
        for k in 0..=n {
            let sign = if k % 2 == 0 { 1 } else { -1 };
            let binomial = binomial_coefficient(n, k);
            let term = (n - k).pow(m as u32);
            count += sign * binomial * term;
        }

        count
    }

    /// Count the number of bijective functions.
    ///
    /// A bijection only exists if |Domain| = |Codomain|.
    /// In that case, there are n! bijections.
    pub fn count_bijective(&self) -> usize {
        if self.domain.len() != self.codomain.len() {
            return 0;
        }

        factorial(self.domain.len())
    }
}

/// Iterator over all functions from a finite domain to a finite codomain.
pub struct FiniteSetMapsIterator<D, C>
where
    D: Eq + Hash + Clone,
    C: Eq + Clone,
{
    domain: Vec<D>,
    codomain: Vec<C>,
    indices: Vec<usize>,
    done: bool,
}

impl<D, C> FiniteSetMapsIterator<D, C>
where
    D: Eq + Hash + Clone,
    C: Eq + Clone,
{
    fn new(domain: Vec<D>, codomain: Vec<C>) -> Self {
        let done = domain.is_empty() && codomain.is_empty();
        Self {
            indices: vec![0; domain.len()],
            domain,
            codomain,
            done,
        }
    }
}

impl<D, C> Iterator for FiniteSetMapsIterator<D, C>
where
    D: Eq + Hash + Clone,
    C: Eq + Clone,
{
    type Item = FiniteSetMap<D, C>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        // Build current mapping
        let mut mapping = HashMap::new();
        for (i, d) in self.domain.iter().enumerate() {
            mapping.insert(d.clone(), self.codomain[self.indices[i]].clone());
        }

        let result = FiniteSetMap {
            domain: self.domain.clone(),
            codomain: self.codomain.clone(),
            mapping,
        };

        // Increment indices (like a counter in base codomain.len())
        let mut pos = self.indices.len();
        while pos > 0 {
            pos -= 1;
            self.indices[pos] += 1;
            if self.indices[pos] < self.codomain.len() {
                break;
            }
            if pos == 0 {
                self.done = true;
                break;
            }
            self.indices[pos] = 0;
        }

        Some(result)
    }
}

/// Compute binomial coefficient C(n, k).
fn binomial_coefficient(n: i64, k: i64) -> i64 {
    if k > n || k < 0 {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }

    let k = k.min(n - k); // Optimization: C(n, k) = C(n, n-k)

    let mut result = 1i64;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// Compute factorial n!
fn factorial(n: usize) -> usize {
    (1..=n).product()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finite_set_map_basic() {
        let domain = vec![1, 2, 3];
        let codomain = vec!["a", "b"];
        let mut mapping = HashMap::new();
        mapping.insert(1, "a");
        mapping.insert(2, "b");
        mapping.insert(3, "a");

        let map = FiniteSetMap::new(domain, codomain, mapping);

        assert_eq!(map.apply(&1), Some(&"a"));
        assert_eq!(map.apply(&2), Some(&"b"));
        assert_eq!(map.apply(&3), Some(&"a"));
        assert_eq!(map.apply(&4), None);

        assert_eq!(map.domain_cardinality(), 3);
        assert_eq!(map.codomain_cardinality(), 2);
    }

    #[test]
    fn test_finite_set_map_injective() {
        let domain = vec![1, 2, 3];
        let codomain = vec!["a", "b", "c", "d"];

        // Injective map
        let mut mapping = HashMap::new();
        mapping.insert(1, "a");
        mapping.insert(2, "b");
        mapping.insert(3, "c");
        let map = FiniteSetMap::new(domain.clone(), codomain.clone(), mapping);
        assert!(map.is_injective());
        assert!(!map.is_surjective());
        assert!(!map.is_bijective());

        // Non-injective map
        let mut mapping = HashMap::new();
        mapping.insert(1, "a");
        mapping.insert(2, "a");
        mapping.insert(3, "c");
        let map = FiniteSetMap::new(domain, codomain, mapping);
        assert!(!map.is_injective());
    }

    #[test]
    fn test_finite_set_map_surjective() {
        let domain = vec![1, 2, 3, 4];
        let codomain = vec!["a", "b"];

        // Surjective map
        let mut mapping = HashMap::new();
        mapping.insert(1, "a");
        mapping.insert(2, "b");
        mapping.insert(3, "a");
        mapping.insert(4, "b");
        let map = FiniteSetMap::new(domain.clone(), codomain.clone(), mapping);
        assert!(map.is_surjective());
        assert!(!map.is_injective());

        // Non-surjective map
        let mut mapping = HashMap::new();
        mapping.insert(1, "a");
        mapping.insert(2, "a");
        mapping.insert(3, "a");
        mapping.insert(4, "a");
        let map = FiniteSetMap::new(domain, codomain, mapping);
        assert!(!map.is_surjective());
    }

    #[test]
    fn test_finite_set_map_bijective() {
        let domain = vec![1, 2, 3];
        let codomain = vec!["a", "b", "c"];

        // Bijective map
        let mut mapping = HashMap::new();
        mapping.insert(1, "a");
        mapping.insert(2, "b");
        mapping.insert(3, "c");
        let map = FiniteSetMap::new(domain, codomain, mapping);
        assert!(map.is_bijective());
        assert!(map.is_injective());
        assert!(map.is_surjective());
    }

    #[test]
    fn test_finite_set_map_image() {
        let domain = vec![1, 2, 3];
        let codomain = vec!["a", "b", "c"];
        let mut mapping = HashMap::new();
        mapping.insert(1, "a");
        mapping.insert(2, "a");
        mapping.insert(3, "b");

        let map = FiniteSetMap::new(domain, codomain, mapping);
        let image = map.image();

        assert_eq!(image.len(), 2);
        assert!(image.contains(&"a"));
        assert!(image.contains(&"b"));
        assert!(!image.contains(&"c"));
    }

    #[test]
    fn test_finite_set_map_preimage() {
        let domain = vec![1, 2, 3, 4];
        let codomain = vec!["a", "b"];
        let mut mapping = HashMap::new();
        mapping.insert(1, "a");
        mapping.insert(2, "b");
        mapping.insert(3, "a");
        mapping.insert(4, "b");

        let map = FiniteSetMap::new(domain, codomain, mapping);

        let preimage_a = map.preimage(&"a");
        assert_eq!(preimage_a.len(), 2);
        assert!(preimage_a.contains(&1));
        assert!(preimage_a.contains(&3));

        let preimage_b = map.preimage(&"b");
        assert_eq!(preimage_b.len(), 2);
        assert!(preimage_b.contains(&2));
        assert!(preimage_b.contains(&4));
    }

    #[test]
    fn test_finite_set_maps_cardinality() {
        let domain = vec![1, 2];
        let codomain = vec!["a", "b"];
        let maps = FiniteSetMaps::new(domain, codomain);

        assert_eq!(maps.cardinality(), Some(4)); // 2^2 = 4

        let domain = vec![1, 2, 3];
        let codomain = vec!["a", "b"];
        let maps = FiniteSetMaps::new(domain, codomain);

        assert_eq!(maps.cardinality(), Some(8)); // 2^3 = 8
    }

    #[test]
    fn test_finite_set_maps_empty() {
        // Empty domain
        let domain: Vec<i32> = vec![];
        let codomain = vec!["a", "b"];
        let maps = FiniteSetMaps::new(domain, codomain);
        assert!(!maps.is_empty());
        assert_eq!(maps.cardinality(), Some(1)); // One empty function

        // Empty codomain, non-empty domain
        let domain = vec![1, 2];
        let codomain: Vec<&str> = vec![];
        let maps = FiniteSetMaps::new(domain, codomain);
        assert!(maps.is_empty());
        assert_eq!(maps.cardinality(), Some(0));
    }

    #[test]
    fn test_finite_set_maps_iterator() {
        let domain = vec![1, 2];
        let codomain = vec!["a", "b"];
        let maps = FiniteSetMaps::new(domain, codomain);

        let all_maps: Vec<_> = maps.iter().unwrap().collect();
        assert_eq!(all_maps.len(), 4);

        // Check that we can apply each map
        for map in &all_maps {
            assert!(map.apply(&1).is_some());
            assert!(map.apply(&2).is_some());
        }
    }

    #[test]
    fn test_count_injective() {
        // 2 elements to 3 elements: 3 * 2 = 6 injective functions
        let domain = vec![1, 2];
        let codomain = vec!["a", "b", "c"];
        let maps = FiniteSetMaps::new(domain, codomain);
        assert_eq!(maps.count_injective(), 6);

        // 3 elements to 2 elements: 0 injective functions (pigeonhole)
        let domain = vec![1, 2, 3];
        let codomain = vec!["a", "b"];
        let maps = FiniteSetMaps::new(domain, codomain);
        assert_eq!(maps.count_injective(), 0);

        // 3 elements to 3 elements: 3! = 6 injective functions
        let domain = vec![1, 2, 3];
        let codomain = vec!["a", "b", "c"];
        let maps = FiniteSetMaps::new(domain, codomain);
        assert_eq!(maps.count_injective(), 6);
    }

    #[test]
    fn test_count_surjective() {
        // 3 elements to 2 elements: Surjections exist
        let domain = vec![1, 2, 3];
        let codomain = vec!["a", "b"];
        let maps = FiniteSetMaps::new(domain, codomain);
        assert_eq!(maps.count_surjective(), 6);

        // 2 elements to 3 elements: 0 surjective functions
        let domain = vec![1, 2];
        let codomain = vec!["a", "b", "c"];
        let maps = FiniteSetMaps::new(domain, codomain);
        assert_eq!(maps.count_surjective(), 0);
    }

    #[test]
    fn test_count_bijective() {
        // 3 elements to 3 elements: 3! = 6 bijections
        let domain = vec![1, 2, 3];
        let codomain = vec!["a", "b", "c"];
        let maps = FiniteSetMaps::new(domain, codomain);
        assert_eq!(maps.count_bijective(), 6);

        // Different sizes: 0 bijections
        let domain = vec![1, 2];
        let codomain = vec!["a", "b", "c"];
        let maps = FiniteSetMaps::new(domain, codomain);
        assert_eq!(maps.count_bijective(), 0);
    }

    #[test]
    fn test_binomial_coefficient() {
        assert_eq!(binomial_coefficient(5, 0), 1);
        assert_eq!(binomial_coefficient(5, 1), 5);
        assert_eq!(binomial_coefficient(5, 2), 10);
        assert_eq!(binomial_coefficient(5, 3), 10);
        assert_eq!(binomial_coefficient(5, 4), 5);
        assert_eq!(binomial_coefficient(5, 5), 1);
        assert_eq!(binomial_coefficient(5, 6), 0);
    }

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
        assert_eq!(factorial(2), 2);
        assert_eq!(factorial(3), 6);
        assert_eq!(factorial(4), 24);
        assert_eq!(factorial(5), 120);
    }
}
