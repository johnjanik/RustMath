//! Coercion system for automatic type conversion between algebraic structures
//!
//! This module provides a comprehensive framework for coercing between related
//! mathematical types while preserving structure. Coercions are morphisms in
//! the category-theoretic sense.
//!
//! # Design Philosophy
//!
//! Coercions follow these principles:
//! 1. **Structure Preservation**: Coercions must preserve algebraic structure
//! 2. **Transitivity**: If A coerces to B and B to C, then A coerces to C
//! 3. **Type Safety**: Coercions are checked at compile time
//! 4. **Explicit**: Coercions must be explicitly defined (no implicit magic)
//!
//! # Examples
//!
//! Common coercions:
//! - Integer → Rational (embed Z into Q)
//! - Integer → Real (embed Z into R)
//! - Rational → Real (embed Q into R)
//! - Ring R → Polynomial Ring R[x]
//! - Field F → Extension Field F'

use crate::morphism::{FormalCoercionMorphism, Morphism};
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;

/// A coercion from source type S to target type T
///
/// Coercions are structure-preserving maps that embed one type into another.
pub trait Coercion: Clone + fmt::Debug {
    /// Source type
    type Source;
    /// Target type
    type Target;

    /// Apply the coercion to convert a value
    fn coerce(&self, source: &Self::Source) -> Self::Target;

    /// Check if this coercion is the identity coercion
    fn is_identity(&self) -> bool {
        false
    }

    /// Get the name of this coercion for display purposes
    fn name(&self) -> String {
        format!("Coercion: {} → {}",
            std::any::type_name::<Self::Source>(),
            std::any::type_name::<Self::Target>())
    }
}

/// Identity coercion: S → S
///
/// Maps every element to itself
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdentityCoercion<T> {
    _phantom: PhantomData<T>,
}

impl<T> IdentityCoercion<T> {
    /// Create a new identity coercion
    pub fn new() -> Self {
        IdentityCoercion {
            _phantom: PhantomData,
        }
    }
}

impl<T> Default for IdentityCoercion<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> Coercion for IdentityCoercion<T> {
    type Source = T;
    type Target = T;

    fn coerce(&self, source: &Self::Source) -> Self::Target {
        source.clone()
    }

    fn is_identity(&self) -> bool {
        true
    }

    fn name(&self) -> String {
        format!("Identity coercion on {}", std::any::type_name::<T>())
    }
}

/// Composed coercion: S → T → U
///
/// If f: S → T and g: T → U are coercions, then g ∘ f: S → U
#[derive(Debug, Clone)]
pub struct ComposedCoercion<F, G>
where
    F: Coercion,
    G: Coercion<Source = F::Target>,
{
    first: F,
    second: G,
}

impl<F, G> ComposedCoercion<F, G>
where
    F: Coercion,
    G: Coercion<Source = F::Target>,
{
    /// Create a new composed coercion g ∘ f
    pub fn new(first: F, second: G) -> Self {
        ComposedCoercion { first, second }
    }

    /// Get the first coercion
    pub fn first(&self) -> &F {
        &self.first
    }

    /// Get the second coercion
    pub fn second(&self) -> &G {
        &self.second
    }
}

impl<F, G> Coercion for ComposedCoercion<F, G>
where
    F: Coercion,
    G: Coercion<Source = F::Target>,
{
    type Source = F::Source;
    type Target = G::Target;

    fn coerce(&self, source: &Self::Source) -> Self::Target {
        let intermediate = self.first.coerce(source);
        self.second.coerce(&intermediate)
    }

    fn name(&self) -> String {
        format!("Composition: {} then {}", self.first.name(), self.second.name())
    }
}

/// A coercion map that stores all registered coercions
///
/// This allows dynamic lookup of coercions between types at runtime.
#[derive(Debug)]
pub struct CoercionMap {
    /// Maps (source_type_id, target_type_id) to a description
    /// In a real implementation, this would store function pointers or trait objects
    coercions: HashMap<(String, String), String>,
}

impl CoercionMap {
    /// Create a new empty coercion map
    pub fn new() -> Self {
        CoercionMap {
            coercions: HashMap::new(),
        }
    }

    /// Register a coercion from S to T
    pub fn register<S, T>(&mut self, name: String)
    where
        S: 'static,
        T: 'static,
    {
        let source_id = std::any::type_name::<S>().to_string();
        let target_id = std::any::type_name::<T>().to_string();
        self.coercions.insert((source_id, target_id), name);
    }

    /// Check if a coercion exists from S to T
    pub fn has_coercion<S, T>(&self) -> bool
    where
        S: 'static,
        T: 'static,
    {
        let source_id = std::any::type_name::<S>().to_string();
        let target_id = std::any::type_name::<T>().to_string();
        self.coercions.contains_key(&(source_id, target_id))
    }

    /// Get all registered coercions
    pub fn list_coercions(&self) -> Vec<String> {
        self.coercions.values().cloned().collect()
    }

    /// Number of registered coercions
    pub fn num_coercions(&self) -> usize {
        self.coercions.len()
    }
}

impl Default for CoercionMap {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for types that can be coerced into another type
///
/// This is similar to Rust's `From` and `Into` traits but specifically
/// for mathematical structures with preservation guarantees.
pub trait CoerceInto<T> {
    /// Coerce this value into type T
    fn coerce_into(self) -> T;
}

/// Trait for types that can be created from another type via coercion
pub trait CoerceFrom<T>: Sized {
    /// Create a value from type T via coercion
    fn coerce_from(value: T) -> Self;
}

/// Automatic implementation: if T implements CoerceFrom<U>, then U implements CoerceInto<T>
impl<T, U> CoerceInto<T> for U
where
    T: CoerceFrom<U>,
{
    fn coerce_into(self) -> T {
        T::coerce_from(self)
    }
}

/// A coercion path is a sequence of coercions that can be composed
///
/// Useful for finding indirect coercion routes: A → B → C → D
#[derive(Debug, Clone)]
pub struct CoercionPath {
    /// Names of types in the path
    path: Vec<String>,
    /// Descriptions of coercions
    coercions: Vec<String>,
}

impl CoercionPath {
    /// Create a new coercion path starting with a type
    pub fn new(start_type: String) -> Self {
        CoercionPath {
            path: vec![start_type],
            coercions: vec![],
        }
    }

    /// Add a coercion step to the path
    pub fn add_step(&mut self, coercion: String, target_type: String) {
        self.coercions.push(coercion);
        self.path.push(target_type);
    }

    /// Get the starting type
    pub fn source(&self) -> Option<&str> {
        self.path.first().map(|s| s.as_str())
    }

    /// Get the ending type
    pub fn target(&self) -> Option<&str> {
        self.path.last().map(|s| s.as_str())
    }

    /// Get the length of the path (number of coercion steps)
    pub fn length(&self) -> usize {
        self.coercions.len()
    }

    /// Check if this is a direct coercion (single step)
    pub fn is_direct(&self) -> bool {
        self.length() == 1
    }

    /// Get all types in the path
    pub fn types(&self) -> &[String] {
        &self.path
    }

    /// Get all coercions in the path
    pub fn coercions(&self) -> &[String] {
        &self.coercions
    }
}

impl fmt::Display for CoercionPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.path.is_empty() {
            return write!(f, "Empty path");
        }

        write!(f, "{}", self.path[0])?;
        for (i, coercion) in self.coercions.iter().enumerate() {
            write!(f, " --[{}]--> {}", coercion, self.path[i + 1])?;
        }
        Ok(())
    }
}

/// Coercion discovery system
///
/// Finds coercion paths between types, including indirect paths
pub struct CoercionDiscovery {
    map: CoercionMap,
}

impl CoercionDiscovery {
    /// Create a new coercion discovery system
    pub fn new(map: CoercionMap) -> Self {
        CoercionDiscovery { map }
    }

    /// Find a coercion path from S to T
    ///
    /// Returns None if no path exists, Some(path) if found.
    /// Prefers shorter paths over longer ones.
    pub fn find_path<S, T>(&self) -> Option<CoercionPath>
    where
        S: 'static,
        T: 'static,
    {
        let source = std::any::type_name::<S>().to_string();
        let target = std::any::type_name::<T>().to_string();

        // Check for direct coercion
        if self.map.has_coercion::<S, T>() {
            let mut path = CoercionPath::new(source.clone());
            path.add_step(format!("{} → {}", source, target), target);
            return Some(path);
        }

        // For now, we don't implement BFS for indirect paths
        // This would require a more sophisticated graph structure
        None
    }

    /// Get the underlying coercion map
    pub fn coercion_map(&self) -> &CoercionMap {
        &self.map
    }
}

/// Standard coercions for built-in types
///
/// This module provides common coercions that are always available
pub mod standard {
    use super::*;

    /// Create a coercion map with standard coercions registered
    pub fn create_standard_coercions() -> CoercionMap {
        let mut map = CoercionMap::new();

        // Integer coercions
        map.register::<i32, i64>("i32 → i64".to_string());
        map.register::<i32, f64>("i32 → f64".to_string());
        map.register::<i64, f64>("i64 → f64".to_string());

        // Unsigned integer coercions
        map.register::<u32, u64>("u32 → u64".to_string());
        map.register::<u32, i64>("u32 → i64".to_string());

        map
    }
}

/// Convert a coercion to a morphism
///
/// Every coercion can be viewed as a morphism in a category
pub fn coercion_to_morphism<S, T>(
    coercion: &impl Coercion<Source = S, Target = T>,
    source: S,
    target: T,
) -> FormalCoercionMorphism<S, T> {
    FormalCoercionMorphism::new(source, target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_coercion() {
        let coercion = IdentityCoercion::<i32>::new();
        assert!(coercion.is_identity());
        assert_eq!(coercion.coerce(&42), 42);
    }

    #[test]
    fn test_identity_coercion_name() {
        let coercion = IdentityCoercion::<i32>::new();
        assert!(coercion.name().contains("Identity"));
    }

    // Simple test coercion: i32 → i64
    #[derive(Debug, Clone)]
    struct I32ToI64;

    impl Coercion for I32ToI64 {
        type Source = i32;
        type Target = i64;

        fn coerce(&self, source: &Self::Source) -> Self::Target {
            *source as i64
        }

        fn name(&self) -> String {
            "i32 → i64".to_string()
        }
    }

    // Simple test coercion: i64 → f64
    #[derive(Debug, Clone)]
    struct I64ToF64;

    impl Coercion for I64ToF64 {
        type Source = i64;
        type Target = f64;

        fn coerce(&self, source: &Self::Source) -> Self::Target {
            *source as f64
        }

        fn name(&self) -> String {
            "i64 → f64".to_string()
        }
    }

    #[test]
    fn test_simple_coercion() {
        let coercion = I32ToI64;
        let result = coercion.coerce(&42);
        assert_eq!(result, 42i64);
    }

    #[test]
    fn test_composed_coercion() {
        let f = I32ToI64;
        let g = I64ToF64;
        let composed = ComposedCoercion::new(f, g);

        let result = composed.coerce(&42);
        assert_eq!(result, 42.0f64);
    }

    #[test]
    fn test_composed_coercion_name() {
        let f = I32ToI64;
        let g = I64ToF64;
        let composed = ComposedCoercion::new(f, g);

        let name = composed.name();
        assert!(name.contains("Composition"));
    }

    #[test]
    fn test_coercion_map_creation() {
        let map = CoercionMap::new();
        assert_eq!(map.num_coercions(), 0);
    }

    #[test]
    fn test_coercion_map_register() {
        let mut map = CoercionMap::new();
        map.register::<i32, i64>("i32 → i64".to_string());

        assert_eq!(map.num_coercions(), 1);
        assert!(map.has_coercion::<i32, i64>());
        assert!(!map.has_coercion::<i64, i32>());
    }

    #[test]
    fn test_coercion_map_list() {
        let mut map = CoercionMap::new();
        map.register::<i32, i64>("i32 → i64".to_string());
        map.register::<i32, f64>("i32 → f64".to_string());

        let list = map.list_coercions();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_standard_coercions() {
        let map = standard::create_standard_coercions();
        assert!(map.num_coercions() > 0);
        assert!(map.has_coercion::<i32, i64>());
        assert!(map.has_coercion::<i32, f64>());
    }

    #[test]
    fn test_coercion_path_creation() {
        let path = CoercionPath::new("i32".to_string());
        assert_eq!(path.source(), Some("i32"));
        assert_eq!(path.target(), Some("i32"));
        assert_eq!(path.length(), 0);
    }

    #[test]
    fn test_coercion_path_add_step() {
        let mut path = CoercionPath::new("i32".to_string());
        path.add_step("convert".to_string(), "i64".to_string());

        assert_eq!(path.source(), Some("i32"));
        assert_eq!(path.target(), Some("i64"));
        assert_eq!(path.length(), 1);
        assert!(path.is_direct());
    }

    #[test]
    fn test_coercion_path_multiple_steps() {
        let mut path = CoercionPath::new("i32".to_string());
        path.add_step("to_i64".to_string(), "i64".to_string());
        path.add_step("to_f64".to_string(), "f64".to_string());

        assert_eq!(path.length(), 2);
        assert!(!path.is_direct());
        assert_eq!(path.types().len(), 3);
        assert_eq!(path.coercions().len(), 2);
    }

    #[test]
    fn test_coercion_path_display() {
        let mut path = CoercionPath::new("i32".to_string());
        path.add_step("convert".to_string(), "i64".to_string());

        let display = format!("{}", path);
        assert!(display.contains("i32"));
        assert!(display.contains("i64"));
        assert!(display.contains("convert"));
    }

    #[test]
    fn test_coercion_discovery() {
        let map = standard::create_standard_coercions();
        let discovery = CoercionDiscovery::new(map);

        // Direct path should be found
        let path = discovery.find_path::<i32, i64>();
        assert!(path.is_some());

        // Non-existent path
        let path = discovery.find_path::<f64, i32>();
        assert!(path.is_none());
    }

    #[test]
    fn test_identity_coercion_default() {
        let coercion1 = IdentityCoercion::<i32>::new();
        let coercion2 = IdentityCoercion::<i32>::default();

        assert_eq!(coercion1, coercion2);
    }

    #[test]
    fn test_coercion_to_morphism() {
        let coercion = I32ToI64;
        let morph = coercion_to_morphism(&coercion, 0, 0i64);

        // Just verify it compiles and creates a morphism
        assert_eq!(morph.source(), &0);
    }
}
