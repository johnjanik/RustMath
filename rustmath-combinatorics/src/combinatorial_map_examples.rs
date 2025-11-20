//! Example combinatorial maps demonstrating the registration system
//!
//! This module provides example maps between common combinatorial objects
//! to demonstrate the decorator-based map registration system.

use crate::combinatorial_map::*;
use crate::partitions::Partition;
use crate::permutations::Permutation;
use crate::tableaux::{robinson_schensted, Tableau};

/// Initialize all example combinatorial maps
///
/// This function registers several example maps to demonstrate the system:
/// - Permutation to cycles (one-way map)
/// - Partition conjugation (bijection)
/// - Identity permutation check (one-way map)
pub fn init_example_maps() {
    register_permutation_to_cycles();
    register_partition_conjugation();
    register_permutation_inverse();
}

/// Register a map from Permutation to its cycle representation
fn register_permutation_to_cycles() {
    register_map::<_, Permutation, Vec<Vec<usize>>>(
        "permutation_to_cycles",
        "Convert a permutation to its cycle representation",
        false,
        None,
        |perm: &Permutation| Some(perm.cycles()),
    );
}

/// Register the partition conjugation bijection
///
/// The conjugation operation on partitions is an involution (self-inverse),
/// meaning applying it twice returns the original partition.
fn register_partition_conjugation() {
    register_map::<_, Partition, Partition>(
        "partition_conjugate",
        "Conjugate (transpose) a partition",
        true,
        Some("partition_conjugate_inverse".to_string()),
        |p: &Partition| Some(p.conjugate()),
    );

    // Register the inverse (which is the same operation since conjugation is an involution)
    register_map::<_, Partition, Partition>(
        "partition_conjugate_inverse",
        "Inverse of partition conjugation (same as conjugation)",
        true,
        Some("partition_conjugate".to_string()),
        |p: &Partition| Some(p.conjugate()),
    );
}

/// Register a map from Permutation to its inverse
fn register_permutation_inverse() {
    register_map::<_, Permutation, Permutation>(
        "permutation_inverse",
        "Compute the inverse of a permutation",
        true,
        Some("permutation_inverse_inverse".to_string()),
        |p: &Permutation| Some(p.inverse()),
    );

    // The inverse of the inverse operation is itself
    register_map::<_, Permutation, Permutation>(
        "permutation_inverse_inverse",
        "Inverse of permutation inversion (same as inversion)",
        true,
        Some("permutation_inverse".to_string()),
        |p: &Permutation| Some(p.inverse()),
    );
}

/// Example: Register Robinson-Schensted correspondence
///
/// This is one of the most important bijections in combinatorics, relating
/// permutations to pairs of standard Young tableaux of the same shape.
///
/// Note: This is provided as an example but requires inverse_robinson_schensted
/// to be implemented for full bidirectionality.
pub fn register_robinson_schensted_correspondence() {
    register_map::<_, Permutation, (Tableau, Tableau)>(
        "robinson_schensted",
        "Robinson-Schensted correspondence: permutation to pair of tableaux",
        false, // Set to true once inverse is implemented
        None,
        |perm: &Permutation| {
            let cycles = (0..perm.size()).collect::<Vec<_>>();
            Some(robinson_schensted(&cycles))
        },
    );
}

/// Demonstrate using the combinatorial map system
pub fn demonstrate_maps() {
    // Initialize the example maps
    init_example_maps();

    // Example 1: Permutation to cycles
    println!("=== Example 1: Permutation to Cycles ===");
    let perm = Permutation::from_vec(vec![2, 0, 1]).unwrap();
    if let Some(cycles) = apply_map::<Permutation, Vec<Vec<usize>>>("permutation_to_cycles", &perm) {
        println!("Permutation: {:?}", perm);
        println!("Cycles: {:?}", cycles);
    }

    // Example 2: Partition conjugation (bijection)
    println!("\n=== Example 2: Partition Conjugation ===");
    let partition = Partition::new(vec![4, 3, 1]);
    println!("Original partition: {:?}", partition);

    if let Some(conjugate) = apply_map::<Partition, Partition>("partition_conjugate", &partition) {
        println!("Conjugate: {:?}", conjugate);

        // Apply inverse to get back original
        if let Some(back) = apply_map::<Partition, Partition>("partition_conjugate_inverse", &conjugate) {
            println!("After applying inverse: {:?}", back);
            println!("Matches original: {}", back == partition);
        }
    }

    // Example 3: Query maps by type
    println!("\n=== Example 3: Query Maps by Type ===");
    let perm_to_perm_maps = find_maps_between::<Permutation, Permutation>();
    println!("Maps from Permutation to Permutation:");
    for map in perm_to_perm_maps {
        println!("  - {}: {}", map.name, map.description);
        println!("    Bijection: {}", map.is_bijection);
    }

    // Example 4: List all registered maps
    println!("\n=== Example 4: All Registered Maps ===");
    let all_maps = list_all_maps();
    for map in all_maps {
        println!("  - {} ({})", map.name,
                 if map.is_bijection { "bijection" } else { "one-way" });
        println!("    {}", map.description);
        println!("    {} -> {}", map.source_type_name, map.target_type_name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permutation_to_cycles() {
        init_example_maps();

        let perm = Permutation::from_vec(vec![1, 2, 0]).unwrap();
        let cycles: Option<Vec<Vec<usize>>> = apply_map("permutation_to_cycles", &perm);

        assert!(cycles.is_some());
        let cycles = cycles.unwrap();
        assert!(!cycles.is_empty());
    }

    #[test]
    fn test_partition_conjugation_bijection() {
        init_example_maps();

        let original = Partition::new(vec![4, 3, 1]);

        // Apply forward map
        let conjugate: Option<Partition> = apply_map("partition_conjugate", &original);
        assert!(conjugate.is_some());

        // Apply inverse map
        let back: Option<Partition> = apply_map("partition_conjugate_inverse", &conjugate.unwrap());
        assert!(back.is_some());

        // Should get back the original
        assert_eq!(back.unwrap(), original);
    }

    #[test]
    fn test_permutation_inverse_bijection() {
        init_example_maps();

        let original = Permutation::from_vec(vec![2, 0, 1]).unwrap();

        // Apply inverse operation
        let inverse: Option<Permutation> = apply_map("permutation_inverse", &original);
        assert!(inverse.is_some());

        // Apply inverse again (should get back original)
        let back: Option<Permutation> = apply_map("permutation_inverse_inverse", &inverse.unwrap());
        assert!(back.is_some());

        assert_eq!(back.unwrap(), original);
    }

    #[test]
    fn test_find_maps_between_types() {
        init_example_maps();

        // Find all maps from Partition to Partition
        let maps = find_maps_between::<Partition, Partition>();
        assert!(!maps.is_empty());

        // Should find the conjugation bijection
        let has_conjugate = maps.iter().any(|m| m.name == "partition_conjugate");
        assert!(has_conjugate);
    }

    #[test]
    fn test_map_metadata() {
        init_example_maps();

        let metadata = get_map_metadata("partition_conjugate");
        assert!(metadata.is_some());

        let metadata = metadata.unwrap();
        assert_eq!(metadata.name, "partition_conjugate");
        assert!(metadata.is_bijection);
        assert_eq!(metadata.inverse_name, Some("partition_conjugate_inverse".to_string()));
    }

    #[test]
    fn test_list_all_maps() {
        init_example_maps();

        let all_maps = list_all_maps();
        assert!(!all_maps.is_empty());

        // Should have at least the examples we registered
        let map_names: Vec<String> = all_maps.iter().map(|m| m.name.clone()).collect();
        assert!(map_names.contains(&"permutation_to_cycles".to_string()));
        assert!(map_names.contains(&"partition_conjugate".to_string()));
    }

    #[test]
    fn test_bijection_property() {
        init_example_maps();

        // Test that conjugation is truly an involution
        let p1 = Partition::new(vec![5, 3, 2, 1]);
        let p2: Partition = apply_map("partition_conjugate", &p1).unwrap();
        let p3: Partition = apply_map("partition_conjugate", &p2).unwrap();

        assert_eq!(p1, p3);
    }
}
