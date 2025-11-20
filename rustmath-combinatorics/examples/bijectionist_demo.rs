//! Demonstrations of the Bijectionist system for automated bijection discovery.
//!
//! This example shows how to use Burnside's lemma and group actions to:
//! 1. Count orbits under group actions
//! 2. Analyze orbit structures
//! 3. Discover potential bijections between different combinatorial sets
//!
//! Run with: cargo run --example bijectionist_demo

use rustmath_combinatorics::bijectionist::*;
use rustmath_combinatorics::permutations::Permutation;

fn main() {
    println!("=== Bijectionist: Automated Bijection Discovery ===\n");

    demo_burnside_basic();
    println!("\n{}\n", "=".repeat(60));

    demo_subset_action();
    println!("\n{}\n", "=".repeat(60));

    demo_bijection_discovery();
    println!("\n{}\n", "=".repeat(60));

    demo_necklace_counting();
    println!("\n{}\n", "=".repeat(60));

    demo_coloring_problems();
}

/// Basic demonstration of Burnside's lemma
fn demo_burnside_basic() {
    println!("Demo 1: Basic Burnside's Lemma");
    println!("{}", "-".repeat(40));

    // Count colorings of vertices of a square with 2 colors
    // under the dihedral group D_4 (rotations and reflections)

    // Represent colorings as binary sequences of length 4
    let colorings: Vec<Vec<u8>> = (0..16)
        .map(|i| vec![(i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1])
        .collect();

    println!("Total colorings: {}", colorings.len());

    // D_4 has 8 elements: identity, 3 rotations, 4 reflections
    // For simplicity, we'll use rotations only (cyclic group C_4)

    let c4 = vec![
        Permutation::identity(4),
        Permutation::from_vec(vec![1, 2, 3, 0]).unwrap(), // 90° rotation
        Permutation::from_vec(vec![2, 3, 0, 1]).unwrap(), // 180° rotation
        Permutation::from_vec(vec![3, 0, 1, 2]).unwrap(), // 270° rotation
    ];

    let action = PermutationActionOnSequence;
    let (num_orbits, fixed_counts) = burnside_count(&action, &colorings, &c4);

    println!("\nGroup: Cyclic group C_4 (rotations of square)");
    println!("Fixed points by each group element:");
    for (i, count) in fixed_counts.iter().enumerate() {
        println!("  g_{}: {} fixed colorings", i, count);
    }

    println!("\nBy Burnside's Lemma:");
    println!("Number of distinct colorings = {}", num_orbits);
    println!("(Two colorings are the same if related by rotation)");
}

/// Demonstrate group action on subsets
fn demo_subset_action() {
    println!("Demo 2: Group Actions on Subsets");
    println!("{}", "-".repeat(40));

    let n = 4;
    let k = 2;
    let subsets = all_k_subsets(n, k);

    println!("All {}-subsets of {{0,1,2,3}}: {:?}", k, subsets);

    // Symmetric group S_4
    let s4_generators = vec![
        Permutation::from_vec(vec![1, 0, 2, 3]).unwrap(), // (0 1)
        Permutation::from_vec(vec![0, 2, 1, 3]).unwrap(), // (1 2)
        Permutation::from_vec(vec![0, 1, 3, 2]).unwrap(), // (2 3)
    ];

    // Generate full S_4 would have 24 elements; use just generators for demo
    let action = PermutationActionOnSet;

    println!("\nOrbits under transpositions:");
    let orbits = action.orbits(&subsets, &s4_generators);

    for (i, orbit) in orbits.iter().enumerate() {
        println!("Orbit {}: {:?}", i + 1, orbit);
    }

    println!("\nAll 2-subsets are in the same orbit under S_4!");
}

/// Demonstrate automated bijection discovery
fn demo_bijection_discovery() {
    println!("Demo 3: Automated Bijection Discovery");
    println!("{}", "-".repeat(40));

    // Compare two different representations under group actions

    // Set X: 2-element subsets of {0,1,2}
    let subsets_2 = all_k_subsets(3, 2);
    println!("Set X: 2-subsets of {{0,1,2}}");
    println!("  Elements: {:?}", subsets_2);

    // Set Y: 1-element subsets (singletons)
    let subsets_1 = all_k_subsets(3, 1);
    println!("\nSet Y: 1-subsets of {{0,1,2}}");
    println!("  Elements: {:?}", subsets_1);

    // Group: S_3
    let s3 = vec![
        Permutation::identity(3),
        Permutation::from_vec(vec![1, 0, 2]).unwrap(), // (0 1)
        Permutation::from_vec(vec![0, 2, 1]).unwrap(), // (1 2)
        Permutation::from_vec(vec![2, 1, 0]).unwrap(), // (0 2)
        Permutation::from_vec(vec![1, 2, 0]).unwrap(), // (0 1 2)
        Permutation::from_vec(vec![2, 0, 1]).unwrap(), // (0 2 1)
    ];

    let action = PermutationActionOnSet;
    let finder = BijectionFinder::with_threshold(0.7);

    if let Some(correspondence) = finder.find_bijection(
        "2-subsets",
        &subsets_2,
        &action,
        "1-subsets",
        &subsets_1,
        &action,
        &s3,
    ) {
        println!("\n=== Bijection Analysis ===");
        println!("Compatibility Score: {:.2}%", correspondence.compatibility_score * 100.0);
        println!("\nOrbit Structure (2-subsets):");
        println!("  Number of orbits: {}", correspondence.structure_x.num_orbits);
        println!("  Orbit sizes: {:?}", correspondence.structure_x.orbit_sizes);

        println!("\nOrbit Structure (1-subsets):");
        println!("  Number of orbits: {}", correspondence.structure_y.num_orbits);
        println!("  Orbit sizes: {:?}", correspondence.structure_y.orbit_sizes);

        if correspondence.is_plausible(0.7) {
            println!("\n✓ Plausible bijection exists!");
        } else {
            println!("\n✗ No plausible bijection found");
        }
    }

    // Now try sets of the same size
    println!("\n\n--- Comparing Compatible Sets ---");

    let pairs = all_unordered_pairs(3);
    println!("\nSet A: Unordered pairs from {{0,1,2}}: {:?}", pairs);

    let pair_action = PermutationActionOnPairs;

    if let Some(correspondence) = finder.find_bijection(
        "2-subsets",
        &subsets_2,
        &action,
        "pairs",
        &pairs,
        &pair_action,
        &s3,
    ) {
        println!("\n=== Bijection Analysis ===");
        println!("Compatibility Score: {:.2}%", correspondence.compatibility_score * 100.0);

        if let Some(ref bijection) = correspondence.suggested_bijection {
            println!("\n✓ Explicit bijection found!");
            println!("Mapping:");
            for (x, y) in bijection.iter() {
                println!("  {:?} ↦ {:?}", x, y);
            }
        }
    }
}

/// Advanced: Using Burnside's lemma to count necklaces
fn demo_necklace_counting() {
    println!("Demo 4: Counting Necklaces with Burnside's Lemma");
    println!("{}", "-".repeat(40));

    // Count binary necklaces of length n
    // (necklaces = sequences up to rotation)

    let n = 5;
    let necklaces: Vec<Vec<u8>> = (0..(1 << n))
        .map(|i| (0..n).map(|j| ((i >> j) & 1) as u8).collect())
        .collect();

    println!("Counting binary necklaces of length {}", n);
    println!("Total sequences: {}", necklaces.len());

    // Cyclic group C_n
    let mut c_n = vec![Permutation::identity(n)];
    for k in 1..n {
        let perm: Vec<usize> = (0..n).map(|i| (i + k) % n).collect();
        c_n.push(Permutation::from_vec(perm).unwrap());
    }

    let action = PermutationActionOnSequence;
    let (num_necklaces, fixed) = burnside_count(&action, &necklaces, &c_n);

    println!("\nFixed points for each rotation:");
    for (k, count) in fixed.iter().enumerate() {
        println!("  Rotation by {}: {} fixed sequences", k, count);
    }

    println!("\nNumber of distinct necklaces: {}", num_necklaces);

    // The formula is also: (1/n) * Σ φ(d) * 2^(n/d) for d|n
    // For n=5 (prime): (1/5) * (φ(1)*2^5 + φ(5)*2^1) = (1/5)*(1*32 + 4*2) = 8
    println!("Expected (by formula for n=5): 8");
}

/// Coloring problems with different symmetry groups
fn demo_coloring_problems() {
    println!("Demo 5: Coloring Problems");
    println!("{}", "-".repeat(40));

    // Color the edges of a triangle with 2 colors
    // Edges: (0,1), (0,2), (1,2)

    let edges = vec![
        vec![0, 1],
        vec![0, 2],
        vec![1, 2],
    ];

    // All 2^3 = 8 colorings (0=red, 1=blue)
    let edge_colorings: Vec<Vec<u8>> = (0..8)
        .map(|i| vec![(i >> 2) & 1, (i >> 1) & 1, i & 1])
        .collect();

    println!("Coloring edges of a triangle with 2 colors");
    println!("Total colorings: {}", edge_colorings.len());

    // Symmetry group: S_3 acting on vertices induces action on edges
    // Edge (i,j) maps to (σ(i), σ(j))

    // For simplicity, use cyclic group C_3 (rotations only)
    let c3 = vec![
        Permutation::identity(3),
        Permutation::from_vec(vec![1, 2, 0]).unwrap(), // (0 1 2)
        Permutation::from_vec(vec![2, 0, 1]).unwrap(), // (0 2 1)
    ];

    // Map edge colorings under rotation
    // This is tricky - we need to track which edge goes where
    // Under rotation (0 1 2): edge 0-1 -> 1-2, edge 0-2 -> 1-0, edge 1-2 -> 2-0
    // This permutes the edges as: [0,1,2] -> [1,2,0]

    let edge_permutation = vec![
        Permutation::identity(3),
        Permutation::from_vec(vec![1, 2, 0]).unwrap(), // Rotation affects edges
        Permutation::from_vec(vec![2, 0, 1]).unwrap(), // Inverse rotation
    ];

    let action = PermutationActionOnSequence;
    let (num_colorings, fixed) = burnside_count(&action, &edge_colorings, &edge_permutation);

    println!("\nSymmetry group: C_3 (rotations of triangle)");
    println!("Fixed edge-colorings for each rotation:");
    for (i, count) in fixed.iter().enumerate() {
        println!("  Rotation {}: {} fixed colorings", i, count);
    }

    println!("\nDistinct edge-colorings up to rotation: {}", num_colorings);
    println!("\nOrbit structure:");

    let orbits = action.orbits(&edge_colorings, &edge_permutation);
    for (i, orbit) in orbits.iter().enumerate() {
        println!("  Orbit {} (size {}): {:?}", i + 1, orbit.len(), orbit[0]);
        if orbit.len() > 1 {
            println!("    (and {} other colorings)", orbit.len() - 1);
        }
    }
}
