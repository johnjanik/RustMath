//! Category trait and base implementations
//!
//! This module defines the fundamental Category trait that all categories must implement.
//! Categories organize mathematical structures and provide a framework for:
//! - Type classification and hierarchy
//! - Method injection via subcategories
//! - Axiom specification

use std::fmt;

/// Base trait for all categories
///
/// A category consists of:
/// - Objects (mathematical structures)
/// - Morphisms between objects
/// - Axioms that objects must satisfy
///
/// Categories can have subcategories that refine or extend the base category.
pub trait Category: fmt::Debug {
    /// Get the name of this category
    fn name(&self) -> &str;

    /// Get all axioms this category requires
    fn axioms(&self) -> Vec<&str> {
        Vec::new()
    }

    /// Get a description of this category
    fn description(&self) -> String {
        format!("Category: {}", self.name())
    }

    /// Get super categories (parent categories in the hierarchy)
    fn super_categories(&self) -> Vec<Box<dyn Category>> {
        Vec::new()
    }

    /// Check whether this category is a subcategory of `other`.
    ///
    /// A category is a subcategory of itself and of every category in its
    /// (transitive) chain of super categories. Categories are compared by
    /// name, which uniquely identifies each category in the hierarchy.
    ///
    /// Traverses the (possibly malformed) super-category graph with a
    /// visited-names guard, so a cyclic `super_categories()` chain (e.g. A's
    /// supercategory is B and B's supercategory is A) terminates with `false`
    /// instead of recursing forever.
    fn is_subcategory_of(&self, other: &dyn Category) -> bool {
        // Free function operating purely on `&dyn Category` so it never
        // needs to unsize `Self` (which may be `?Sized` in this default
        // trait method); recursion only ever passes along `&dyn Category`
        // references already produced by `super_categories()`.
        fn search(cat: &dyn Category, other: &dyn Category, visited: &mut Vec<String>) -> bool {
            if cat.name() == other.name() {
                return true;
            }
            if visited.iter().any(|seen| seen == cat.name()) {
                // Already visited this node on the current path: it's part
                // of a cycle that never reached `other`, so stop here.
                return false;
            }
            visited.push(cat.name().to_string());
            cat.super_categories()
                .iter()
                .any(|super_cat| search(super_cat.as_ref(), other, visited))
        }

        if self.name() == other.name() {
            return true;
        }
        let mut visited = vec![self.name().to_string()];
        self.super_categories()
            .iter()
            .any(|super_cat| search(super_cat.as_ref(), other, &mut visited))
    }
}

/// Marker trait for categories that support finite structures
pub trait FiniteCategory: Category {
    /// Check if all objects in this category must be finite
    fn requires_finite(&self) -> bool {
        true
    }
}

/// Marker trait for categories with commutative operations
pub trait CommutativeCategory: Category {
    /// Check if all operations in this category are commutative
    fn is_commutative(&self) -> bool {
        true
    }
}

/// Marker trait for categories with topological structure
pub trait TopologicalCategory: Category {
    /// Check if this category has topological structure
    fn has_topology(&self) -> bool {
        true
    }
}

/// Marker trait for categories that support Cartesian products
pub trait CartesianProductsCategory: Category {
    /// Check if this category supports Cartesian products
    fn supports_cartesian_products(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test category
    #[derive(Clone, Debug)]
    struct TestCategory {
        name: String,
    }

    impl TestCategory {
        fn new(name: &str) -> Self {
            TestCategory {
                name: name.to_string(),
            }
        }
    }

    impl Category for TestCategory {
        fn name(&self) -> &str {
            &self.name
        }

        fn axioms(&self) -> Vec<&str> {
            vec!["test_axiom"]
        }
    }

    #[test]
    fn test_category_name() {
        let cat = TestCategory::new("Test");
        assert_eq!(cat.name(), "Test");
    }

    #[test]
    fn test_category_description() {
        let cat = TestCategory::new("Test");
        assert_eq!(cat.description(), "Category: Test");
    }

    #[test]
    fn test_category_axioms() {
        let cat = TestCategory::new("Test");
        let axioms = cat.axioms();
        assert_eq!(axioms.len(), 1);
        assert_eq!(axioms[0], "test_axiom");
    }

    #[test]
    fn test_category_comparison() {
        let cat1 = TestCategory::new("Test");
        let cat2 = TestCategory::new("Test");
        let cat3 = TestCategory::new("Other");

        assert_eq!(cat1.name(), cat2.name());
        assert_ne!(cat1.name(), cat3.name());
    }

    // Categories whose `super_categories()` chain forms a cycle, used to
    // exercise the cycle guard in the default `is_subcategory_of` impl.
    #[derive(Clone, Debug)]
    struct CycleA;
    #[derive(Clone, Debug)]
    struct CycleB;

    impl Category for CycleA {
        fn name(&self) -> &str {
            "CycleA"
        }
        fn super_categories(&self) -> Vec<Box<dyn Category>> {
            vec![Box::new(CycleB)]
        }
    }

    impl Category for CycleB {
        fn name(&self) -> &str {
            "CycleB"
        }
        fn super_categories(&self) -> Vec<Box<dyn Category>> {
            // Points back at CycleA, forming a 2-cycle.
            vec![Box::new(CycleA)]
        }
    }

    #[derive(Clone, Debug)]
    struct SelfCycle;
    impl Category for SelfCycle {
        fn name(&self) -> &str {
            "SelfCycle"
        }
        fn super_categories(&self) -> Vec<Box<dyn Category>> {
            // Points at a fresh copy of itself, forming a 1-cycle.
            vec![Box::new(SelfCycle)]
        }
    }

    #[test]
    fn test_is_subcategory_of_cyclic_chain_terminates_false() {
        let a = CycleA;
        let unrelated = TestCategory::new("Unrelated");
        // Must terminate (not infinite-loop / stack overflow) and report
        // false, since the cycle never reaches `unrelated`.
        assert!(!a.is_subcategory_of(&unrelated));
    }

    #[test]
    fn test_is_subcategory_of_cyclic_chain_still_finds_self() {
        let a = CycleA;
        // Reflexivity must still hold even though the chain is cyclic.
        assert!(a.is_subcategory_of(&a));
        // And it can still find the other side of the cycle.
        assert!(a.is_subcategory_of(&CycleB));
    }

    #[test]
    fn test_is_subcategory_of_self_referential_cycle_terminates_false() {
        let s = SelfCycle;
        let unrelated = TestCategory::new("Unrelated");
        assert!(!s.is_subcategory_of(&unrelated));
        assert!(s.is_subcategory_of(&s));
    }
}
