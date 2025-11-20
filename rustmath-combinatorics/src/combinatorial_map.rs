//! Combinatorial Maps - A system for registering and working with bijections between combinatorial objects
//!
//! This module provides infrastructure for:
//! - Defining maps between combinatorial objects
//! - Registering bidirectional bijections with metadata
//! - Looking up maps by name or by source/target types
//! - Automatically handling inverse maps
//!
//! # Example
//!
//! ```ignore
//! use rustmath_combinatorics::combinatorial_map::*;
//! use rustmath_combinatorics::{Permutation, Tableau};
//!
//! // Register a bijection using the macro
//! register_combinatorial_map! {
//!     name: "robinson_schensted",
//!     description: "Robinson-Schensted correspondence between permutations and tableau pairs",
//!     from: Permutation => to: (Tableau, Tableau),
//!     bijection: true,
//!     forward: |perm| robinson_schensted(perm),
//!     inverse: |pair| inverse_robinson_schensted(pair)
//! }
//!
//! // Use the registered map
//! let perm = Permutation::from_vec(vec![3, 1, 2]);
//! if let Some(result) = apply_map::<Permutation, (Tableau, Tableau)>("robinson_schensted", &perm) {
//!     println!("Tableau pair: {:?}", result);
//! }
//! ```

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Trait for types that can be used in combinatorial maps
pub trait CombinatorialObject: Clone + 'static {}

// Blanket implementation for any Clone + 'static type
impl<T: Clone + 'static> CombinatorialObject for T {}

/// Metadata about a registered combinatorial map
#[derive(Clone)]
pub struct MapMetadata {
    /// Unique name of the map
    pub name: String,

    /// Description of what the map does
    pub description: String,

    /// Whether this is a bijection (has an inverse)
    pub is_bijection: bool,

    /// Type ID of the source type
    pub source_type: TypeId,

    /// Type ID of the target type
    pub target_type: TypeId,

    /// Name of the source type (for debugging)
    pub source_type_name: &'static str,

    /// Name of the target type (for debugging)
    pub target_type_name: &'static str,

    /// Name of the inverse map (if this is a bijection)
    pub inverse_name: Option<String>,
}

/// A type-erased combinatorial map function
type BoxedMap = Box<dyn Fn(&dyn Any) -> Option<Box<dyn Any>> + Send + Sync>;

/// Global registry of combinatorial maps
pub struct MapRegistry {
    /// Maps from map name to the function
    maps: HashMap<String, BoxedMap>,

    /// Maps from map name to metadata
    metadata: HashMap<String, MapMetadata>,

    /// Index from (source_type, target_type) to map names
    type_index: HashMap<(TypeId, TypeId), Vec<String>>,
}

impl MapRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        MapRegistry {
            maps: HashMap::new(),
            metadata: HashMap::new(),
            type_index: HashMap::new(),
        }
    }

    /// Register a new map
    pub fn register<F, T, U>(
        &mut self,
        name: String,
        description: String,
        is_bijection: bool,
        inverse_name: Option<String>,
        func: F,
    ) where
        F: Fn(&T) -> Option<U> + Send + Sync + 'static,
        T: CombinatorialObject,
        U: CombinatorialObject,
    {
        let source_type = TypeId::of::<T>();
        let target_type = TypeId::of::<U>();

        // Create metadata
        let metadata = MapMetadata {
            name: name.clone(),
            description,
            is_bijection,
            source_type,
            target_type,
            source_type_name: std::any::type_name::<T>(),
            target_type_name: std::any::type_name::<U>(),
            inverse_name,
        };

        // Create type-erased wrapper
        let boxed_func: BoxedMap = Box::new(move |input: &dyn Any| {
            input.downcast_ref::<T>()
                .and_then(|t| func(t))
                .map(|u| Box::new(u) as Box<dyn Any>)
        });

        // Store in registry
        self.maps.insert(name.clone(), boxed_func);
        self.metadata.insert(name.clone(), metadata);

        // Update type index
        self.type_index
            .entry((source_type, target_type))
            .or_insert_with(Vec::new)
            .push(name);
    }

    /// Apply a map by name
    pub fn apply<T, U>(&self, name: &str, input: &T) -> Option<U>
    where
        T: CombinatorialObject,
        U: CombinatorialObject,
    {
        let func = self.maps.get(name)?;
        let result = func(input as &dyn Any)?;
        result.downcast_ref::<U>().cloned()
    }

    /// Get metadata for a map
    pub fn get_metadata(&self, name: &str) -> Option<&MapMetadata> {
        self.metadata.get(name)
    }

    /// Find all maps from type T to type U
    pub fn find_maps<T, U>(&self) -> Vec<&MapMetadata>
    where
        T: 'static,
        U: 'static,
    {
        let source_type = TypeId::of::<T>();
        let target_type = TypeId::of::<U>();

        self.type_index
            .get(&(source_type, target_type))
            .map(|names| {
                names.iter()
                    .filter_map(|name| self.metadata.get(name))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// List all registered maps
    pub fn list_all(&self) -> Vec<&MapMetadata> {
        self.metadata.values().collect()
    }

    /// Check if a map is registered
    pub fn has_map(&self, name: &str) -> bool {
        self.maps.contains_key(name)
    }
}

impl Default for MapRegistry {
    fn default() -> Self {
        Self::new()
    }
}

lazy_static::lazy_static! {
    /// Global map registry
    static ref GLOBAL_REGISTRY: Arc<RwLock<MapRegistry>> = Arc::new(RwLock::new(MapRegistry::new()));
}

/// Register a map in the global registry
pub fn register_map<F, T, U>(
    name: impl Into<String>,
    description: impl Into<String>,
    is_bijection: bool,
    inverse_name: Option<String>,
    func: F,
) where
    F: Fn(&T) -> Option<U> + Send + Sync + 'static,
    T: CombinatorialObject,
    U: CombinatorialObject,
{
    let mut registry = GLOBAL_REGISTRY.write().unwrap();
    registry.register(name.into(), description.into(), is_bijection, inverse_name, func);
}

/// Apply a map from the global registry
pub fn apply_map<T, U>(name: &str, input: &T) -> Option<U>
where
    T: CombinatorialObject,
    U: CombinatorialObject,
{
    let registry = GLOBAL_REGISTRY.read().unwrap();
    registry.apply(name, input)
}

/// Get metadata for a map from the global registry
pub fn get_map_metadata(name: &str) -> Option<MapMetadata> {
    let registry = GLOBAL_REGISTRY.read().unwrap();
    registry.get_metadata(name).cloned()
}

/// Find all maps between two types in the global registry
pub fn find_maps_between<T, U>() -> Vec<MapMetadata>
where
    T: 'static,
    U: 'static,
{
    let registry = GLOBAL_REGISTRY.read().unwrap();
    registry.find_maps::<T, U>().into_iter().cloned().collect()
}

/// List all registered maps in the global registry
pub fn list_all_maps() -> Vec<MapMetadata> {
    let registry = GLOBAL_REGISTRY.read().unwrap();
    registry.list_all().into_iter().cloned().collect()
}

/// Check if a map exists in the global registry
pub fn has_map(name: &str) -> bool {
    let registry = GLOBAL_REGISTRY.read().unwrap();
    registry.has_map(name)
}

/// Macro for registering combinatorial maps with decorator-like syntax
///
/// # Usage
///
/// ```ignore
/// use rustmath_combinatorics::register_combinatorial_map;
///
/// // Register a simple map
/// register_combinatorial_map! {
///     name: "permutation_to_cycles",
///     description: "Convert a permutation to its cycle representation",
///     from: Permutation => to: Vec<Vec<usize>>,
///     bijection: false,
///     forward: |perm| Some(perm.cycles())
/// }
///
/// // Register a bijection with inverse
/// register_combinatorial_map! {
///     name: "robinson_schensted",
///     description: "Robinson-Schensted correspondence",
///     from: Permutation => to: (Tableau, Tableau),
///     bijection: true,
///     forward: |perm| Some(robinson_schensted(perm)),
///     inverse: |pair| Some(inverse_robinson_schensted(pair))
/// }
/// ```
#[macro_export]
macro_rules! register_combinatorial_map {
    // Map without inverse
    (
        name: $name:expr,
        description: $desc:expr,
        from: $from_ty:ty => to: $to_ty:ty,
        bijection: false,
        forward: $forward:expr
    ) => {
        {
            $crate::combinatorial_map::register_map::<_, $from_ty, $to_ty>(
                $name,
                $desc,
                false,
                None,
                $forward,
            );
        }
    };

    // Bijection with inverse
    (
        name: $name:expr,
        description: $desc:expr,
        from: $from_ty:ty => to: $to_ty:ty,
        bijection: true,
        forward: $forward:expr,
        inverse: $inverse:expr
    ) => {
        {
            let forward_name = $name;
            let inverse_name = format!("{}_inverse", forward_name);

            // Register forward map
            $crate::combinatorial_map::register_map::<_, $from_ty, $to_ty>(
                forward_name,
                $desc,
                true,
                Some(inverse_name.clone()),
                $forward,
            );

            // Register inverse map
            $crate::combinatorial_map::register_map::<_, $to_ty, $from_ty>(
                &inverse_name,
                format!("Inverse of {}", $desc),
                true,
                Some(forward_name.to_string()),
                $inverse,
            );
        }
    };
}

/// Helper macro to define a bijection pair
#[macro_export]
macro_rules! define_bijection {
    (
        $forward_name:ident: $from_ty:ty => $to_ty:ty,
        forward: $forward_fn:expr,
        inverse: $inverse_fn:expr,
        description: $desc:expr
    ) => {
        pub fn $forward_name() {
            $crate::register_combinatorial_map! {
                name: stringify!($forward_name),
                description: $desc,
                from: $from_ty => to: $to_ty,
                bijection: true,
                forward: $forward_fn,
                inverse: $inverse_fn
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct TestObjectA(i32);

    #[derive(Clone, Debug, PartialEq)]
    struct TestObjectB(String);

    #[test]
    fn test_basic_map_registration() {
        let mut registry = MapRegistry::new();

        registry.register::<_, TestObjectA, TestObjectB>(
            "a_to_b".to_string(),
            "Convert A to B".to_string(),
            false,
            None,
            |a: &TestObjectA| Some(TestObjectB(a.0.to_string())),
        );

        assert!(registry.has_map("a_to_b"));

        let input = TestObjectA(42);
        let result: Option<TestObjectB> = registry.apply("a_to_b", &input);
        assert_eq!(result, Some(TestObjectB("42".to_string())));
    }

    #[test]
    fn test_bijection_registration() {
        let mut registry = MapRegistry::new();

        // Register forward map
        registry.register::<_, TestObjectA, TestObjectB>(
            "a_to_b_forward".to_string(),
            "Convert A to B (bijection)".to_string(),
            true,
            Some("a_to_b_inverse".to_string()),
            |a: &TestObjectA| Some(TestObjectB(a.0.to_string())),
        );

        // Register inverse map
        registry.register::<_, TestObjectB, TestObjectA>(
            "a_to_b_inverse".to_string(),
            "Convert B to A (inverse)".to_string(),
            true,
            Some("a_to_b_forward".to_string()),
            |b: &TestObjectB| b.0.parse::<i32>().ok().map(TestObjectA),
        );

        let input = TestObjectA(42);
        let intermediate: Option<TestObjectB> = registry.apply("a_to_b_forward", &input);
        assert_eq!(intermediate, Some(TestObjectB("42".to_string())));

        let back: Option<TestObjectA> = registry.apply("a_to_b_inverse", &intermediate.unwrap());
        assert_eq!(back, Some(input));
    }

    #[test]
    fn test_find_maps_by_type() {
        let mut registry = MapRegistry::new();

        registry.register::<_, TestObjectA, TestObjectB>(
            "map1".to_string(),
            "First map".to_string(),
            false,
            None,
            |a: &TestObjectA| Some(TestObjectB(a.0.to_string())),
        );

        registry.register::<_, TestObjectA, TestObjectB>(
            "map2".to_string(),
            "Second map".to_string(),
            false,
            None,
            |a: &TestObjectA| Some(TestObjectB(format!("value: {}", a.0))),
        );

        let maps = registry.find_maps::<TestObjectA, TestObjectB>();
        assert_eq!(maps.len(), 2);
    }

    #[test]
    fn test_list_all_maps() {
        let mut registry = MapRegistry::new();

        registry.register::<_, TestObjectA, TestObjectB>(
            "map1".to_string(),
            "First map".to_string(),
            false,
            None,
            |a: &TestObjectA| Some(TestObjectB(a.0.to_string())),
        );

        let all_maps = registry.list_all();
        assert_eq!(all_maps.len(), 1);
        assert_eq!(all_maps[0].name, "map1");
    }

    #[test]
    fn test_metadata() {
        let mut registry = MapRegistry::new();

        registry.register::<_, TestObjectA, TestObjectB>(
            "test_map".to_string(),
            "A test map".to_string(),
            true,
            Some("test_map_inverse".to_string()),
            |a: &TestObjectA| Some(TestObjectB(a.0.to_string())),
        );

        let metadata = registry.get_metadata("test_map").unwrap();
        assert_eq!(metadata.name, "test_map");
        assert_eq!(metadata.description, "A test map");
        assert!(metadata.is_bijection);
        assert_eq!(metadata.inverse_name, Some("test_map_inverse".to_string()));
    }

    #[test]
    fn test_global_registry() {
        // Register a map in the global registry
        register_map::<_, TestObjectA, TestObjectB>(
            "global_test_map",
            "A global test map",
            false,
            None,
            |a: &TestObjectA| Some(TestObjectB(a.0.to_string())),
        );

        // Check it exists
        assert!(has_map("global_test_map"));

        // Apply it
        let input = TestObjectA(99);
        let result: Option<TestObjectB> = apply_map("global_test_map", &input);
        assert_eq!(result, Some(TestObjectB("99".to_string())));

        // Get metadata
        let metadata = get_map_metadata("global_test_map");
        assert!(metadata.is_some());
    }
}
