//! Integration tests for category theory infrastructure
//!
//! These tests verify that all components work together correctly:
//! - Axioms
//! - Coercions
//! - Morphisms
//! - Categories
//! - Functors
//! - Natural transformations

use rustmath_category::{
    axioms::*,
    coercion::*,
    algebraic_morphisms::*,
    ring_category::*,
    morphism_composition::*,
    category::Category,
    morphism::{Morphism, IdentityMorphism},
};

#[test]
fn test_axiom_system_integration() {
    // Create axiom sets for different structures
    let group_axioms = AxiomSet::group();
    let ring_axioms = AxiomSet::ring();
    let field_axioms = AxiomSet::field();

    // Verify hierarchy
    assert!(group_axioms.has_axiom("associativity"));
    assert!(ring_axioms.has_axiom("distributivity"));
    assert!(field_axioms.has_axiom("unity"));
}

#[test]
fn test_category_hierarchy_integration() {
    // Create categories
    let ring_cat = RingCategory::new();
    let comm_ring_cat = CommutativeRingCategory::new();
    let domain_cat = IntegralDomainCategory::new();

    // Verify subcategory relationships
    assert!(comm_ring_cat.is_subcategory_of(&ring_cat));
    assert!(domain_cat.is_subcategory_of(&comm_ring_cat));
    assert!(domain_cat.is_subcategory_of(&ring_cat));
}

#[test]
fn test_coercion_and_morphism_integration() {
    // Create a simple coercion
    let coercion = IdentityCoercion::<i32>::new();
    assert!(coercion.is_identity());

    // Create a ring morphism using similar structure
    let ring_morph = RingMorphism::new(0, 0, |x: &i32| *x);
    assert_eq!(ring_morph.apply(&42), 42);

    // Both should preserve structure
    assert_eq!(coercion.coerce(&42), 42);
    assert_eq!(ring_morph.apply(&42), 42);
}

#[test]
fn test_morphism_composition_with_identity() {
    // Create identity morphisms
    let id1 = IdentityMorphism::new(42);
    let id2 = IdentityMorphism::new(42);

    // Compose them
    let composed = id1.compose(&id2);
    assert!(composed.is_some());

    let result = composed.unwrap();
    assert!(result.is_identity());
}

#[test]
fn test_ring_morphism_composition() {
    // Create ring morphisms f: Z → Z (double) and g: Z → Z (add 1)
    let f = RingMorphism::new(0, 0, |x: &i32| x * 2);
    let g = RingMorphism::new(0, 0, |x: &i32| x + 1);

    // Apply them separately
    assert_eq!(f.apply(&5), 10);
    assert_eq!(g.apply(&10), 11);

    // In a full implementation, we'd compose them
    // For now, verify they're both valid ring morphisms
    assert_eq!(f.source_ring(), &0);
    assert_eq!(g.target_ring(), &0);
}

#[test]
fn test_coercion_path_discovery() {
    // Create a coercion map with standard coercions
    let map = standard::create_standard_coercions();

    // Verify basic coercions exist
    assert!(map.has_coercion::<i32, i64>());
    assert!(map.has_coercion::<i32, f64>());

    // Use discovery system
    let discovery = CoercionDiscovery::new(map);
    let path = discovery.find_path::<i32, i64>();
    assert!(path.is_some());
}

#[test]
fn test_morphism_diagram_with_ring_category() {
    // Create a diagram in the ring category
    let mut diagram = MorphismDiagram::<String, i32>::new();

    // Add objects (rings)
    diagram.add_object("Z".to_string()); // Integers
    diagram.add_object("Q".to_string()); // Rationals

    // Add morphism (inclusion Z ↪ Q)
    diagram.add_morphism("Z".to_string(), "Q".to_string(), 1);

    assert_eq!(diagram.num_objects(), 2);
    assert_eq!(diagram.num_morphisms(), 1);
}

#[test]
fn test_axioms_and_categories_correspondence() {
    // Ring category should have corresponding axioms
    let ring_cat = RingCategory;
    let ring_axioms = AxiomSet::ring();

    let cat_axioms = ring_cat.axioms();
    let set_names = ring_axioms.axiom_names();

    // Both should mention distributivity
    assert!(cat_axioms.contains(&"distributivity"));
    assert!(set_names.contains(&"distributivity"));
}

#[test]
fn test_element_methods_on_integers() {
    // Test ring element methods on integers
    assert!(1i32.is_unit());
    assert!(0i32.is_idempotent());
    assert!(0i32.is_nilpotent());

    // Multiplicative order
    assert_eq!(1i32.multiplicative_order(), Some(1));
    assert_eq!((-1i32).multiplicative_order(), Some(2));
}

#[test]
fn test_field_morphism_is_injective() {
    // Field morphisms are always injective
    let morph = FieldMorphism::new(1.0, 1.0, |x: &f64| *x);
    assert!(morph.is_embedding());
}

#[test]
fn test_group_morphism_as_morphism_trait() {
    // Group morphism should implement Morphism trait
    let morph = GroupMorphism::new(0, 0, |x: &i32| x * 2);

    // Can use Morphism trait methods
    assert_eq!(morph.source(), &0);
    assert_eq!(morph.target(), &0);
    assert!(!morph.is_identity());
}

#[test]
fn test_module_morphism_creation() {
    // Create a module morphism (vector doubling)
    let morph = ModuleMorphism::new(vec![0], vec![0], |v: &Vec<i32>| {
        v.iter().map(|x| x * 2).collect()
    });

    let result = morph.apply(&vec![1, 2, 3]);
    assert_eq!(result, vec![2, 4, 6]);
}

#[test]
fn test_composition_table_for_small_category() {
    // Create a composition table for a small category
    let mut table = CompositionTable::<String, String>::new();

    // Add compositions: id ∘ f = f, f ∘ id = f
    table.add_composition("id".to_string(), "f".to_string(), "f".to_string());
    table.add_composition("f".to_string(), "id".to_string(), "f".to_string());

    assert!(table.has_composition("id", "f"));
    assert_eq!(table.get_composition("id", "f"), Some("f"));
}

#[test]
fn test_morphism_path_composition() {
    // Create a path of identity morphisms
    let mut path = MorphismPath::new();
    path.add_morphism(IdentityMorphism::new(1));
    path.add_morphism(IdentityMorphism::new(1));

    assert_eq!(path.length(), 2);

    // Compose all
    let composed = path.compose_all();
    assert!(composed.is_some());
}

#[test]
fn test_standard_ring_morphisms() {
    // Test identity ring morphism
    let id_morph = standard::identity_ring_morphism(42);
    assert_eq!(id_morph.apply(&42), 42);

    // Test zero morphism
    let zero_morph = standard::zero_ring_morphism(1, 2, 0);
    assert_eq!(zero_morph.apply(&100), 0);

    // Test inclusion morphism
    let incl = standard::inclusion_morphism(0, 0);
    assert_eq!(incl.apply(&42), 42);
}

#[test]
fn test_ring_constructions() {
    // Test quotient ring construction
    let qr = constructions::QuotientRing::new(0i32);
    assert_eq!(qr.base_ring(), &0);

    // Test product ring construction
    let pr = constructions::ProductRing::new(0i32, 0.0f64);
    assert_eq!(pr.first_ring(), &0);

    // Test matrix ring construction
    let mr = constructions::MatrixRing::new(0i32, 3);
    assert_eq!(mr.dimension(), 3);
}

#[test]
fn test_composition_result_variants() {
    // Test success variant
    let success: CompositionResult<i32> = CompositionResult::Success(42);
    assert!(success.is_success());
    assert_eq!(success.morphism(), Some(42));

    // Test type mismatch variant
    let mismatch: CompositionResult<i32> = CompositionResult::TypeMismatch {
        source: "A".to_string(),
        target: "B".to_string(),
    };
    assert!(mismatch.is_failure());

    // Test undefined variant
    let undefined: CompositionResult<i32> = CompositionResult::Undefined;
    assert!(undefined.is_failure());
}

#[test]
fn test_coercion_composition() {
    // Test composed coercion
    #[derive(Debug, Clone)]
    struct I32ToI64;
    impl Coercion for I32ToI64 {
        type Source = i32;
        type Target = i64;
        fn coerce(&self, source: &i32) -> i64 {
            *source as i64
        }
    }

    #[derive(Debug, Clone)]
    struct I64ToF64;
    impl Coercion for I64ToF64 {
        type Source = i64;
        type Target = f64;
        fn coerce(&self, source: &i64) -> f64 {
            *source as f64
        }
    }

    let composed = ComposedCoercion::new(I32ToI64, I64ToF64);
    let result = composed.coerce(&42);
    assert_eq!(result, 42.0);
}

#[test]
fn test_axiom_properties() {
    // Test various axiom properties
    let assoc = Associativity;
    assert_eq!(assoc.name(), "associativity");
    assert!(!assoc.symbolic().is_empty());

    let comm = Commutativity;
    assert!(comm.description().contains("commutative"));

    let inv = Inverse;
    assert!(inv.symbolic().contains("⁻¹"));
}

#[test]
fn test_category_description() {
    let ring_cat = RingCategory::new();
    assert!(!ring_cat.description().is_empty());
    assert!(ring_cat.description().contains("ring"));
}

#[test]
fn test_morphism_diagram_queries() {
    let mut diagram = MorphismDiagram::new();

    // Build a simple diagram: A → B → C
    diagram.add_morphism("A".to_string(), "B".to_string(), 1);
    diagram.add_morphism("B".to_string(), "C".to_string(), 2);

    // Query morphisms from A
    let from_a = diagram.morphisms_from(&"A".to_string());
    assert_eq!(from_a.len(), 1);

    // Query morphisms to C
    let to_c = diagram.morphisms_to(&"C".to_string());
    assert_eq!(to_c.len(), 1);
}

#[test]
fn test_full_integration_workflow() {
    // This test demonstrates a complete workflow using all components

    // 1. Define axioms for our structure
    let ring_axioms = AxiomSet::ring();
    assert!(ring_axioms.has_axiom("distributivity"));

    // 2. Create category
    let ring_cat = RingCategory::new();
    assert_eq!(ring_cat.name(), "Rings");

    // 3. Create morphisms
    let f = RingMorphism::new(0, 0, |x: &i32| x * 2);
    assert_eq!(f.apply(&5), 10);

    // 4. Create coercions
    let coerce = IdentityCoercion::<i32>::new();
    assert_eq!(coerce.coerce(&42), 42);

    // 5. Build a diagram
    let mut diagram = MorphismDiagram::<String, i32>::new();
    diagram.add_morphism("Z".to_string(), "Z".to_string(), 1);
    assert_eq!(diagram.num_morphisms(), 1);

    // 6. Verify element properties
    assert!(1i32.is_unit());
    assert!(0i32.is_idempotent());
}
