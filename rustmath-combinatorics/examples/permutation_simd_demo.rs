//! Demonstration of SIMD-optimized permutation operations
//!
//! This example shows how to use the high-performance SIMD implementations
//! of permutation multiplication, inversion, and cycle decomposition.
//!
//! Run with: cargo run --example permutation_simd_demo

use rustmath_combinatorics::{
    Permutation, compose_simd, inverse_simd, cycles_simd, power_simd,
    batch_compose_simd, simd_available, simd_info,
};

fn main() {
    println!("=== SIMD Permutation Operations Demo ===\n");

    // Check SIMD availability
    println!("SIMD Support:");
    println!("  Available: {}", simd_available());
    println!("  {}\n", simd_info());

    // Example 1: Basic permutation composition
    println!("Example 1: Permutation Composition");
    println!("-----------------------------------");
    let p1 = Permutation::from_vec(vec![1, 2, 3, 0]).unwrap();
    let p2 = Permutation::from_vec(vec![3, 0, 1, 2]).unwrap();

    println!("p1 = {:?}", p1.as_slice());
    println!("p2 = {:?}", p2.as_slice());

    let composed = compose_simd(&p1, &p2).unwrap();
    println!("p1 ∘ p2 = {:?}\n", composed.as_slice());

    // Example 2: Permutation inversion
    println!("Example 2: Permutation Inversion");
    println!("---------------------------------");
    let perm = Permutation::from_vec(vec![3, 1, 4, 0, 2]).unwrap();
    println!("perm = {:?}", perm.as_slice());

    let inv = inverse_simd(&perm);
    println!("inverse = {:?}", inv.as_slice());

    // Verify: perm ∘ inverse = identity
    let identity_check = compose_simd(&perm, &inv).unwrap();
    println!("perm ∘ inverse = {:?} (should be identity)\n", identity_check.as_slice());

    // Example 3: Cycle decomposition
    println!("Example 3: Cycle Decomposition");
    println!("-------------------------------");
    let perm = Permutation::from_vec(vec![1, 2, 0, 4, 3, 6, 5, 7]).unwrap();
    println!("perm = {:?}", perm.as_slice());

    let cycles = cycles_simd(&perm);
    println!("Cycles:");
    for cycle in &cycles {
        println!("  {:?}", cycle);
    }
    println!();

    // Example 4: Computing powers of a permutation
    println!("Example 4: Permutation Powers");
    println!("------------------------------");
    let perm = Permutation::from_vec(vec![1, 2, 3, 0]).unwrap();
    println!("perm = {:?}", perm.as_slice());

    for k in 0..=5 {
        let perm_k = power_simd(&perm, k);
        println!("perm^{} = {:?}", k, perm_k.as_slice());
    }
    println!();

    // Example 5: Batch composition
    println!("Example 5: Batch Composition");
    println!("-----------------------------");
    let p1 = Permutation::from_vec(vec![1, 0, 2]).unwrap();
    let p2 = Permutation::from_vec(vec![0, 2, 1]).unwrap();
    let p3 = Permutation::from_vec(vec![2, 1, 0]).unwrap();

    println!("p1 = {:?}", p1.as_slice());
    println!("p2 = {:?}", p2.as_slice());
    println!("p3 = {:?}", p3.as_slice());

    let result = batch_compose_simd(&[p1, p2, p3]).unwrap();
    println!("p1 ∘ p2 ∘ p3 = {:?}\n", result.as_slice());

    // Example 6: Performance with larger permutations
    println!("Example 6: Large Permutation Operations");
    println!("----------------------------------------");
    let n = 128;
    let large_perm: Vec<usize> = (0..n).map(|i| (i * 7 + 13) % n).collect();
    let perm = Permutation::from_vec(large_perm).unwrap();

    println!("Computing operations on permutation of size {}...", n);

    use std::time::Instant;

    // Measure inversion
    let start = Instant::now();
    let inv = inverse_simd(&perm);
    let inv_time = start.elapsed();
    println!("  Inversion: {:?}", inv_time);

    // Measure composition
    let start = Instant::now();
    let composed = compose_simd(&perm, &inv).unwrap();
    let comp_time = start.elapsed();
    println!("  Composition: {:?}", comp_time);

    // Verify result is identity
    let is_identity = composed.as_slice().iter().enumerate().all(|(i, &v)| i == v);
    println!("  Result is identity: {}", is_identity);

    // Measure cycle decomposition
    let start = Instant::now();
    let cycles = cycles_simd(&perm);
    let cycle_time = start.elapsed();
    println!("  Cycle decomposition: {:?}", cycle_time);
    println!("  Number of cycles: {}\n", cycles.len());

    println!("=== Demo Complete ===");
}
