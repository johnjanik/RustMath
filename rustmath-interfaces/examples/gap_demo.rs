//! GAP Interface Demo
//!
//! This example demonstrates the GAP interface functionality.
//! Note: Requires GAP to be installed on the system.
//!
//! Run with: cargo run --example gap_demo

use rustmath_interfaces::gap::GapInterface;
use rustmath_interfaces::gap_parser::*;
use rustmath_interfaces::gap_permutation::GapPermutationGroup;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("RustMath GAP Interface Demo\n");

    // Check if GAP is available
    if !rustmath_interfaces::is_gap_available() {
        eprintln!("GAP is not installed or not in PATH");
        eprintln!("Please install GAP:");
        eprintln!("  Ubuntu/Debian: sudo apt-get install gap");
        eprintln!("  macOS: brew install gap");
        eprintln!("  Windows: https://www.gap-system.org/");
        return Ok(());
    }

    // Show GAP version
    if let Some(version) = rustmath_interfaces::gap_version() {
        println!("GAP version: {}\n", version.trim());
    }

    // Basic GAP commands
    println!("=== Basic GAP Commands ===");
    let gap = GapInterface::new()?;

    let result = gap.execute("2 + 2;")?;
    println!("2 + 2 = {}", result);

    let result = gap.execute("Factorial(10);")?;
    println!("10! = {}", result);

    // Group operations
    println!("\n=== Group Operations ===");

    let s5_order = gap.group_order("SymmetricGroup(5)")?;
    println!("Order of S5: {}", s5_order);

    let a5_order = gap.group_order("AlternatingGroup(5)")?;
    println!("Order of A5: {}", a5_order);

    // Group properties
    println!("\n=== Group Properties ===");

    let is_abelian = gap.is_abelian("SymmetricGroup(3)")?;
    println!("Is S3 abelian? {}", is_abelian);

    let is_simple = gap.is_simple("AlternatingGroup(5)")?;
    println!("Is A5 simple? {}", is_simple);

    let is_solvable = gap.is_solvable("SymmetricGroup(4)")?;
    println!("Is S4 solvable? {}", is_solvable);

    // Permutation parsing
    println!("\n=== Permutation Parsing ===");

    let perm = parse_permutation("(1,2,3)(4,5)")?;
    println!("Parsed permutation: {:?}", perm);
    println!("Cycles: {:?}", perm.cycles);
    println!("Order: {}", perm.order());
    println!("Images: {:?}", perm.to_images());

    // High-level permutation group interface
    println!("\n=== High-Level Permutation Group Interface ===");

    let s5 = GapPermutationGroup::symmetric(5)?;
    println!("Created S5");
    println!("Order: {}", s5.order()?);
    println!("Is abelian: {}", s5.is_abelian()?);
    println!("Is transitive: {}", s5.is_transitive()?);

    // Orbit and stabilizer
    println!("\n=== Orbit and Stabilizer ===");

    let orbit = s5.orbit(1)?;
    println!("Orbit of 1 under S5: {:?}", orbit);

    let stab = s5.stabilizer(1)?;
    println!("Order of Stab_S5(1): {}", stab.order()?);

    // Conjugacy classes
    println!("\n=== Conjugacy Classes ===");

    let s4 = GapPermutationGroup::symmetric(4)?;
    let num_classes = s4.num_conjugacy_classes()?;
    println!("Number of conjugacy classes in S4: {}", num_classes);

    // Generators
    println!("\n=== Generators ===");

    let s3 = GapPermutationGroup::symmetric(3)?;
    let gens = s3.generators()?;
    println!("Generators of S3:");
    for (i, gen) in gens.iter().enumerate() {
        println!("  gen {}: {:?}", i + 1, gen.cycles);
    }

    // Orbits
    println!("\n=== Orbits ===");

    let orbits = s3.orbits()?;
    println!("Orbits of S3:");
    for (i, orbit) in orbits.iter().enumerate() {
        println!("  orbit {}: {:?}", i + 1, orbit);
    }

    // Advanced: Base and Strong Generating Set (Schreier-Sims)
    println!("\n=== Schreier-Sims Algorithm ===");

    match s5.base_and_strong_generators() {
        Ok((base, sgs)) => {
            println!("Base: {:?}", base);
            println!("Number of strong generators: {}", sgs.len());
        }
        Err(e) => {
            println!("Error computing base and SGS: {}", e);
        }
    }

    println!("\n=== Demo Complete ===");

    Ok(())
}
