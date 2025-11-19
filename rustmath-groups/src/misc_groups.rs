//! Miscellaneous Groups
//!
//! This module is a collection of group constructions and utilities that may
//! not fit into other infinite families described elsewhere in the crate.
//!
//! # Overview
//!
//! This module serves as a catch-all for:
//! - Special group constructions
//! - Named finite groups
//! - Group utilities and helpers
//! - Catalog of well-known groups
//!
//! # Examples
//!
//! ```
//! use rustmath_groups::misc_groups::{klein_four_group, quaternion_group};
//!
//! // The Klein four-group V_4 ≅ Z_2 × Z_2
//! let v4 = klein_four_group();
//!
//! // The quaternion group Q_8
//! let q8 = quaternion_group();
//! ```

use std::collections::HashMap;

use crate::finitely_presented::FinitelyPresentedGroup;
use crate::permutation_group::PermutationGroup;
use crate::group_traits::Group;

/// Convert a number of generators and Tietze-form relations to the new API
fn make_fp_group(num_gens: usize, relations: Vec<Vec<i32>>) -> FinitelyPresentedGroup {
    let generator_names: Vec<String> = (0..num_gens)
        .map(|i| format!("x{}", i))
        .collect();
    FinitelyPresentedGroup::new(generator_names, relations)
}

/// The Klein four-group V_4
///
/// This is the smallest non-cyclic group, isomorphic to Z_2 × Z_2.
/// It has four elements: {e, a, b, ab} with a² = b² = (ab)² = e.
pub fn klein_four_group() -> FinitelyPresentedGroup {
    // Presentation: ⟨a, b | a² = b² = (ab)² = 1⟩
    let mut relations = Vec::new();

    // a² = 1
    relations.push(vec![1, 1]); // x0^2

    // b² = 1
    relations.push(vec![2, 2]); // x1^2

    // (ab)² = 1
    relations.push(vec![1, 2, 1, 2]); // x0 x1 x0 x1

    make_fp_group(2, relations)
}

/// The quaternion group Q_8
///
/// This is the non-abelian group of order 8 with elements {±1, ±i, ±j, ±k}
/// satisfying i² = j² = k² = ijk = -1.
pub fn quaternion_group() -> FinitelyPresentedGroup {
    // Presentation: ⟨i, j | i⁴ = 1, i² = j², j⁻¹ij = i⁻¹⟩
    let mut relations = Vec::new();

    // i⁴ = 1
    relations.push(vec![1, 1, 1, 1]); // x0^4

    // i² = j²  ⟺  i² j⁻² = 1
    relations.push(vec![1, 1, -2, -2]); // x0^2 x1^-2

    // j⁻¹ij = i⁻¹  ⟺  j⁻¹iji = 1
    relations.push(vec![-2, 1, 2, 1]); // x1^-1 x0 x1 x0

    make_fp_group(2, relations)
}

/// The dicyclic group Dic_n
///
/// This is the generalization of the quaternion group, with order 4n.
/// For n=2, this is Q_8.
pub fn dicyclic_group(n: usize) -> FinitelyPresentedGroup {
    assert!(n >= 2, "Dicyclic group requires n ≥ 2");

    // Presentation: ⟨a, x | a^(2n) = 1, x² = a^n, x⁻¹ax = a⁻¹⟩
    let mut relations = Vec::new();

    // a^(2n) = 1
    let mut a_2n = Vec::new();
    for _ in 0..(2 * n) {
        a_2n.push(1); // x0 repeated 2n times
    }
    relations.push(a_2n);

    // x² = a^n  ⟺  x² a^(-n) = 1
    let mut x2_an = vec![2, 2]; // x1^2
    for _ in 0..n {
        x2_an.push(-1); // x0^-1 repeated n times
    }
    relations.push(x2_an);

    // x⁻¹ax = a⁻¹  ⟺  x⁻¹axa = 1
    relations.push(vec![-2, 1, 2, 1]); // x1^-1 x0 x1 x0

    make_fp_group(2, relations)
}

/// The Mathieu group M_11
///
/// This is a simple sporadic group of order 7920, the smallest Mathieu group.
/// Note: This is a stub that would require a more sophisticated implementation.
pub fn mathieu_11() -> PermutationGroup {
    // Stub: Return a permutation group on 11 elements
    // A full implementation would construct the actual Mathieu group
    PermutationGroup::symmetric(11)
}

/// The Mathieu group M_12
///
/// This is a simple sporadic group of order 95040.
/// Note: This is a stub that would require a more sophisticated implementation.
pub fn mathieu_12() -> PermutationGroup {
    PermutationGroup::symmetric(12)
}

/// The trivial group
pub fn trivial_group() -> FinitelyPresentedGroup {
    FinitelyPresentedGroup::new(vec![], vec![])
}

/// Create a group from its order (when possible)
///
/// For small orders, this returns a well-known group of that order.
pub fn group_of_order(n: usize) -> Option<FinitelyPresentedGroup> {
    match n {
        0 => None,
        1 => Some(trivial_group()),
        2 => {
            // Z_2: x^2 = 1
            Some(make_fp_group(1, vec![vec![1, 1]]))
        }
        3 => {
            // Z_3: x^3 = 1
            Some(make_fp_group(1, vec![vec![1, 1, 1]]))
        }
        4 => Some(klein_four_group()), // Could also be Z_4
        5 => {
            // Z_5: x^5 = 1
            Some(make_fp_group(1, vec![vec![1, 1, 1, 1, 1]]))
        }
        8 => Some(quaternion_group()), // Could also be Z_8, Z_4 × Z_2, etc.
        _ => None, // For larger orders, we'd need more sophisticated logic
    }
}

/// Catalog of named groups
#[derive(Debug)]
pub struct GroupCatalog {
    groups: HashMap<String, FinitelyPresentedGroup>,
}

impl GroupCatalog {
    /// Create a new group catalog
    pub fn new() -> Self {
        let mut catalog = Self {
            groups: HashMap::new(),
        };
        catalog.populate();
        catalog
    }

    /// Populate with well-known groups
    fn populate(&mut self) {
        self.groups.insert("V4".to_string(), klein_four_group());
        self.groups.insert("Q8".to_string(), quaternion_group());
        self.groups.insert("Trivial".to_string(), trivial_group());

        // Add some dicyclic groups
        for n in 2..=5 {
            self.groups
                .insert(format!("Dic{}", n), dicyclic_group(n));
        }
    }

    /// Get a group by name
    pub fn get(&self, name: &str) -> Option<&FinitelyPresentedGroup> {
        self.groups.get(name)
    }

    /// List all available group names
    pub fn list_names(&self) -> Vec<&String> {
        self.groups.keys().collect()
    }

    /// Number of groups in catalog
    pub fn len(&self) -> usize {
        self.groups.len()
    }

    /// Check if catalog is empty
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }
}

impl Default for GroupCatalog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_klein_four_group() {
        let v4 = klein_four_group();
        assert_eq!(v4.num_generators(), 2);
    }

    #[test]
    fn test_quaternion_group() {
        let q8 = quaternion_group();
        assert_eq!(q8.num_generators(), 2);
    }

    #[test]
    fn test_dicyclic_group() {
        let dic2 = dicyclic_group(2);
        assert_eq!(dic2.num_generators(), 2);

        let dic3 = dicyclic_group(3);
        assert_eq!(dic3.num_generators(), 2);
    }

    #[test]
    #[should_panic(expected = "Dicyclic group requires n ≥ 2")]
    fn test_dicyclic_group_invalid() {
        dicyclic_group(1);
    }

    #[test]
    fn test_trivial_group() {
        let triv = trivial_group();
        assert_eq!(triv.num_generators(), 0);
    }

    #[test]
    fn test_group_of_order() {
        assert!(group_of_order(0).is_none());
        assert!(group_of_order(1).is_some());
        assert!(group_of_order(2).is_some());
        assert!(group_of_order(3).is_some());
        assert!(group_of_order(4).is_some());
        assert!(group_of_order(8).is_some());
    }

    #[test]
    fn test_group_catalog() {
        let catalog = GroupCatalog::new();
        assert!(!catalog.is_empty());
        assert!(catalog.len() > 0);

        assert!(catalog.get("V4").is_some());
        assert!(catalog.get("Q8").is_some());
        assert!(catalog.get("Trivial").is_some());
        assert!(catalog.get("Dic2").is_some());

        let names = catalog.list_names();
        assert!(names.len() > 0);
    }

    #[test]
    fn test_catalog_contains_named_groups() {
        let catalog = GroupCatalog::new();

        // Check Klein four group
        let v4 = catalog.get("V4");
        assert!(v4.is_some());
        assert_eq!(v4.unwrap().num_generators(), 2);

        // Check quaternion group
        let q8 = catalog.get("Q8");
        assert!(q8.is_some());
        assert_eq!(q8.unwrap().num_generators(), 2);
    }

    #[test]
    fn test_mathieu_groups_stub() {
        let m11 = mathieu_11();
        assert_eq!(m11.degree(), 11);

        let m12 = mathieu_12();
        assert_eq!(m12.degree(), 12);
    }
}
