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
pub trait Category: Clone + fmt::Debug {
    /// Get the name of this category
    fn name(&self) -> &str;

    /// Get the parent category (if any)
    ///
    /// Returns None for top-level categories
    fn super_categories(&self) -> Vec<Box<dyn Category>> {
        Vec::new()
    }

    /// Check if this category is a subcategory of another
    fn is_subcategory_of(&self, other: &dyn Category) -> bool {
        if self.name() == other.name() {
            return true;
        }

        for parent in self.super_categories() {
            if parent.is_subcategory_of(other) {
                return true;
            }
        }

        false
    }

    /// Get all axioms this category requires
    fn axioms(&self) -> Vec<&str> {
        Vec::new()
    }

    /// Get a description of this category
    fn description(&self) -> String {
        format!("Category: {}", self.name())
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
    fn test_is_subcategory_of() {
        let cat1 = TestCategory::new("Test");
        let cat2 = TestCategory::new("Test");
        let cat3 = TestCategory::new("Other");

        assert!(cat1.is_subcategory_of(&cat2));
        assert!(!cat1.is_subcategory_of(&cat3));
    }
}
