//! Morphism composition utilities for algebraic structures
//!
//! This module provides utilities for composing morphisms between algebraic
//! structures, verifying composition rules, and building morphism diagrams.
//!
//! # Mathematical Background
//!
//! Morphisms in category theory must satisfy:
//! 1. **Associativity**: (h ∘ g) ∘ f = h ∘ (g ∘ f)
//! 2. **Identity**: id_B ∘ f = f = f ∘ id_A for f: A → B
//! 3. **Closure**: If f: A → B and g: B → C, then g ∘ f: A → C exists

use crate::morphism::Morphism;
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;

/// A morphism composition result
///
/// Represents either a successful composition or an error
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompositionResult<M> {
    /// Successful composition
    Success(M),
    /// Composition failed due to type mismatch
    TypeMismatch { source: String, target: String },
    /// Composition undefined
    Undefined,
}

impl<M> CompositionResult<M> {
    /// Check if composition was successful
    pub fn is_success(&self) -> bool {
        matches!(self, CompositionResult::Success(_))
    }

    /// Check if composition failed
    pub fn is_failure(&self) -> bool {
        !self.is_success()
    }

    /// Get the morphism if successful
    pub fn morphism(self) -> Option<M> {
        match self {
            CompositionResult::Success(m) => Some(m),
            _ => None,
        }
    }
}

impl<M: fmt::Display> fmt::Display for CompositionResult<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompositionResult::Success(m) => write!(f, "Success: {}", m),
            CompositionResult::TypeMismatch { source, target } => {
                write!(f, "Type mismatch: {} ≠ {}", source, target)
            }
            CompositionResult::Undefined => write!(f, "Composition undefined"),
        }
    }
}

/// Compose two morphisms: g ∘ f
///
/// Requires: codomain(f) = domain(g)
///
/// # Arguments
/// - `f`: First morphism A → B
/// - `g`: Second morphism B → C
///
/// # Returns
/// Composed morphism A → C if composition is valid
pub fn compose<M: Morphism>(f: &M, g: &M) -> Option<M> {
    f.compose(g)
}

/// Verify associativity of morphism composition
///
/// Check that (h ∘ g) ∘ f = h ∘ (g ∘ f)
pub fn verify_associativity<M: Morphism + PartialEq>(f: &M, g: &M, h: &M) -> bool {
    // Compute (h ∘ g) ∘ f
    let g_comp_f = match compose(f, g) {
        Some(gf) => gf,
        None => return false,
    };
    let left = match compose(&g_comp_f, h) {
        Some(result) => result,
        None => return false,
    };

    // Compute h ∘ (g ∘ f)
    let h_comp_g = match compose(g, h) {
        Some(hg) => hg,
        None => return false,
    };
    let right = match compose(f, &h_comp_g) {
        Some(result) => result,
        None => return false,
    };

    left == right
}

/// A morphism diagram
///
/// Represents a collection of objects and morphisms forming a diagram
/// in a category. Useful for checking commutativity and other properties.
///
/// # Example Diagrams
///
/// Triangle:
/// ```text
///     A
///    / \
///   f   g
///  /     \
/// B ---h-> C
/// ```
///
/// Square:
/// ```text
/// A --f--> B
/// |        |
/// g        h
/// |        |
/// v        v
/// C --k--> D
/// ```
#[derive(Debug, Clone)]
pub struct MorphismDiagram<Obj, Morph>
where
    Obj: Eq + Hash + Clone,
{
    /// Objects in the diagram
    objects: Vec<Obj>,
    /// Morphisms: (source, target, morphism)
    morphisms: Vec<(Obj, Obj, Morph)>,
}

impl<Obj, Morph> MorphismDiagram<Obj, Morph>
where
    Obj: Eq + Hash + Clone,
{
    /// Create a new empty diagram
    pub fn new() -> Self {
        MorphismDiagram {
            objects: vec![],
            morphisms: vec![],
        }
    }

    /// Add an object to the diagram
    pub fn add_object(&mut self, obj: Obj) {
        if !self.objects.contains(&obj) {
            self.objects.push(obj);
        }
    }

    /// Add a morphism to the diagram
    pub fn add_morphism(&mut self, source: Obj, target: Obj, morphism: Morph) {
        self.add_object(source.clone());
        self.add_object(target.clone());
        self.morphisms.push((source, target, morphism));
    }

    /// Get all objects in the diagram
    pub fn objects(&self) -> &[Obj] {
        &self.objects
    }

    /// Get all morphisms in the diagram
    pub fn morphisms(&self) -> &[(Obj, Obj, Morph)] {
        &self.morphisms
    }

    /// Get morphisms from a specific source
    pub fn morphisms_from(&self, source: &Obj) -> Vec<&(Obj, Obj, Morph)> {
        self.morphisms
            .iter()
            .filter(|(s, _, _)| s == source)
            .collect()
    }

    /// Get morphisms to a specific target
    pub fn morphisms_to(&self, target: &Obj) -> Vec<&(Obj, Obj, Morph)> {
        self.morphisms
            .iter()
            .filter(|(_, t, _)| t == target)
            .collect()
    }

    /// Number of objects
    pub fn num_objects(&self) -> usize {
        self.objects.len()
    }

    /// Number of morphisms
    pub fn num_morphisms(&self) -> usize {
        self.morphisms.len()
    }
}

impl<Obj, Morph> Default for MorphismDiagram<Obj, Morph>
where
    Obj: Eq + Hash + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Check if a square diagram commutes
///
/// ```text
/// A --f--> B
/// |        |
/// g        h
/// |        |
/// v        v
/// C --k--> D
/// ```
///
/// Commutes if h ∘ f = k ∘ g
pub fn square_commutes<M: Morphism + PartialEq>(f: &M, g: &M, h: &M, k: &M) -> bool {
    // Compute h ∘ f
    let top_right = match compose(f, h) {
        Some(result) => result,
        None => return false,
    };

    // Compute k ∘ g
    let bottom_left = match compose(g, k) {
        Some(result) => result,
        None => return false,
    };

    top_right == bottom_left
}

/// Check if a triangle diagram commutes
///
/// ```text
///     A
///    / \
///   f   g
///  /     \
/// B ---h-> C
/// ```
///
/// Commutes if g = h ∘ f
pub fn triangle_commutes<M: Morphism + PartialEq>(f: &M, g: &M, h: &M) -> bool {
    match compose(f, h) {
        Some(composed) => composed == *g,
        None => false,
    }
}

/// A morphism path: f₁ → f₂ → ... → fₙ
///
/// Represents a sequence of composable morphisms
#[derive(Debug, Clone)]
pub struct MorphismPath<M> {
    morphisms: Vec<M>,
}

impl<M> MorphismPath<M> {
    /// Create a new empty path
    pub fn new() -> Self {
        MorphismPath {
            morphisms: vec![],
        }
    }

    /// Create a path from a single morphism
    pub fn from_morphism(morphism: M) -> Self {
        MorphismPath {
            morphisms: vec![morphism],
        }
    }

    /// Add a morphism to the path
    pub fn add_morphism(&mut self, morphism: M) {
        self.morphisms.push(morphism);
    }

    /// Get all morphisms in the path
    pub fn morphisms(&self) -> &[M] {
        &self.morphisms
    }

    /// Get the length of the path
    pub fn length(&self) -> usize {
        self.morphisms.len()
    }

    /// Check if the path is empty
    pub fn is_empty(&self) -> bool {
        self.morphisms.is_empty()
    }

    /// Compose all morphisms in the path into a single morphism
    pub fn compose_all(&self) -> Option<M>
    where
        M: Morphism,
    {
        if self.is_empty() {
            return None;
        }

        let mut result = self.morphisms[0].clone();
        for morphism in &self.morphisms[1..] {
            result = result.compose(morphism)?;
        }

        Some(result)
    }
}

impl<M> Default for MorphismPath<M> {
    fn default() -> Self {
        Self::new()
    }
}

/// A morphism composition table
///
/// Stores all compositions between a set of morphisms
/// Useful for small categories where all compositions are known
#[derive(Debug)]
pub struct CompositionTable<Obj, Morph>
where
    Obj: Eq + Hash + Clone,
{
    /// Maps (morphism1, morphism2) to their composition
    table: HashMap<(String, String), String>,
    _phantom: std::marker::PhantomData<(Obj, Morph)>,
}

impl<Obj, Morph> CompositionTable<Obj, Morph>
where
    Obj: Eq + Hash + Clone,
{
    /// Create a new empty composition table
    pub fn new() -> Self {
        CompositionTable {
            table: HashMap::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Add a composition: f ∘ g = h
    pub fn add_composition(&mut self, f: String, g: String, h: String) {
        self.table.insert((f, g), h);
    }

    /// Get the composition of two morphisms
    pub fn get_composition(&self, f: &str, g: &str) -> Option<&str> {
        self.table
            .get(&(f.to_string(), g.to_string()))
            .map(|s| s.as_str())
    }

    /// Check if a composition is defined
    pub fn has_composition(&self, f: &str, g: &str) -> bool {
        self.table.contains_key(&(f.to_string(), g.to_string()))
    }

    /// Number of stored compositions
    pub fn size(&self) -> usize {
        self.table.len()
    }
}

impl<Obj, Morph> Default for CompositionTable<Obj, Morph>
where
    Obj: Eq + Hash + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphism::IdentityMorphism;

    #[test]
    fn test_composition_result_success() {
        let result = CompositionResult::Success(42);
        assert!(result.is_success());
        assert!(!result.is_failure());
        assert_eq!(result.morphism(), Some(42));
    }

    #[test]
    fn test_composition_result_type_mismatch() {
        let result: CompositionResult<i32> = CompositionResult::TypeMismatch {
            source: "A".to_string(),
            target: "B".to_string(),
        };
        assert!(!result.is_success());
        assert!(result.is_failure());
        assert_eq!(result.morphism(), None);
    }

    #[test]
    fn test_composition_result_undefined() {
        let result: CompositionResult<i32> = CompositionResult::Undefined;
        assert!(result.is_failure());
    }

    #[test]
    fn test_compose_identity_morphisms() {
        let id1 = IdentityMorphism::new(42);
        let id2 = IdentityMorphism::new(42);

        let composed = compose(&id1, &id2);
        assert!(composed.is_some());
    }

    #[test]
    fn test_morphism_diagram_creation() {
        let mut diagram = MorphismDiagram::<String, i32>::new();
        assert_eq!(diagram.num_objects(), 0);
        assert_eq!(diagram.num_morphisms(), 0);
    }

    #[test]
    fn test_morphism_diagram_add_object() {
        let mut diagram = MorphismDiagram::<String, i32>::new();
        diagram.add_object("A".to_string());
        diagram.add_object("B".to_string());

        assert_eq!(diagram.num_objects(), 2);
    }

    #[test]
    fn test_morphism_diagram_add_morphism() {
        let mut diagram = MorphismDiagram::new();
        diagram.add_morphism("A".to_string(), "B".to_string(), 42);

        assert_eq!(diagram.num_objects(), 2);
        assert_eq!(diagram.num_morphisms(), 1);
    }

    #[test]
    fn test_morphism_diagram_morphisms_from() {
        let mut diagram = MorphismDiagram::new();
        diagram.add_morphism("A".to_string(), "B".to_string(), 1);
        diagram.add_morphism("A".to_string(), "C".to_string(), 2);
        diagram.add_morphism("B".to_string(), "C".to_string(), 3);

        let from_a = diagram.morphisms_from(&"A".to_string());
        assert_eq!(from_a.len(), 2);
    }

    #[test]
    fn test_morphism_diagram_morphisms_to() {
        let mut diagram = MorphismDiagram::new();
        diagram.add_morphism("A".to_string(), "C".to_string(), 1);
        diagram.add_morphism("B".to_string(), "C".to_string(), 2);

        let to_c = diagram.morphisms_to(&"C".to_string());
        assert_eq!(to_c.len(), 2);
    }

    #[test]
    fn test_morphism_path_creation() {
        let path = MorphismPath::<i32>::new();
        assert_eq!(path.length(), 0);
        assert!(path.is_empty());
    }

    #[test]
    fn test_morphism_path_from_morphism() {
        let path = MorphismPath::from_morphism(42);
        assert_eq!(path.length(), 1);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_morphism_path_add_morphism() {
        let mut path = MorphismPath::new();
        path.add_morphism(1);
        path.add_morphism(2);
        path.add_morphism(3);

        assert_eq!(path.length(), 3);
        assert_eq!(path.morphisms(), &[1, 2, 3]);
    }

    #[test]
    fn test_morphism_path_compose_all_identity() {
        let id1 = IdentityMorphism::new(42);
        let id2 = IdentityMorphism::new(42);

        let mut path = MorphismPath::new();
        path.add_morphism(id1);
        path.add_morphism(id2);

        let composed = path.compose_all();
        assert!(composed.is_some());
    }

    #[test]
    fn test_composition_table_creation() {
        let table = CompositionTable::<String, i32>::new();
        assert_eq!(table.size(), 0);
    }

    #[test]
    fn test_composition_table_add() {
        let mut table = CompositionTable::<String, i32>::new();
        table.add_composition("f".to_string(), "g".to_string(), "h".to_string());

        assert_eq!(table.size(), 1);
        assert!(table.has_composition("f", "g"));
    }

    #[test]
    fn test_composition_table_get() {
        let mut table = CompositionTable::<String, i32>::new();
        table.add_composition("f".to_string(), "g".to_string(), "fg".to_string());

        assert_eq!(table.get_composition("f", "g"), Some("fg"));
        assert_eq!(table.get_composition("g", "f"), None);
    }

    #[test]
    fn test_verify_associativity_identity() {
        let id = IdentityMorphism::new(42);
        assert!(verify_associativity(&id, &id, &id));
    }

    #[test]
    fn test_composition_result_display() {
        let result: CompositionResult<&str> = CompositionResult::Success("morphism");
        let display = format!("{}", result);
        assert!(display.contains("Success"));
    }

    #[test]
    fn test_morphism_diagram_default() {
        let diagram = MorphismDiagram::<String, i32>::default();
        assert_eq!(diagram.num_objects(), 0);
    }

    #[test]
    fn test_morphism_path_default() {
        let path = MorphismPath::<i32>::default();
        assert!(path.is_empty());
    }

    #[test]
    fn test_composition_table_default() {
        let table = CompositionTable::<String, i32>::default();
        assert_eq!(table.size(), 0);
    }
}
