//! OEIS (Online Encyclopedia of Integer Sequences) Demo
//!
//! This example demonstrates the OEIS database interface.
//!
//! Run with: cargo run --example oeis_demo --package rustmath-databases

use rustmath_databases::oeis::OEISClient;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OEIS Database Interface Demo ===\n");

    let client = OEISClient::new();

    // Example 1: Look up Fibonacci sequence
    println!("Example 1: Lookup Fibonacci Sequence (A000045)");
    println!("-----------------------------------------------");
    if let Some(seq) = client.lookup("A000045")? {
        println!("Number: {}", seq.number);
        println!("Name: {}", seq.name);
        println!("First 15 terms: {:?}", &seq.data[..15.min(seq.data.len())]);
        println!();

        if !seq.formula.is_empty() {
            println!("Formulas:");
            for (i, formula) in seq.formula.iter().take(3).enumerate() {
                println!("  {}. {}", i + 1, formula);
            }
            println!();
        }
    }

    // Example 2: Look up Prime Numbers
    println!("Example 2: Lookup Prime Numbers (A000040)");
    println!("------------------------------------------");
    if let Some(seq) = client.lookup("A000040")? {
        println!("Number: {}", seq.number);
        println!("Name: {}", seq.name);
        println!("First 20 primes: {:?}", &seq.data[..20.min(seq.data.len())]);
        println!();
    }

    // Example 3: Look up Catalan Numbers
    println!("Example 3: Lookup Catalan Numbers (A000108)");
    println!("--------------------------------------------");
    if let Some(seq) = client.lookup("108")? {  // Can use just the number
        println!("Number: {}", seq.number);
        println!("Name: {}", seq.name);
        println!("First 10 terms: {:?}", &seq.data[..10.min(seq.data.len())]);
        println!();
    }

    // Example 4: Search by sequence terms
    println!("Example 4: Search by Terms [1, 2, 4, 8, 16, 32]");
    println!("------------------------------------------------");
    let search_results = client.search_by_terms(&[1, 2, 4, 8, 16, 32])?;
    println!("Found {} sequences", search_results.len());
    for (i, seq) in search_results.iter().take(5).enumerate() {
        println!("  {}. {}: {}", i + 1, seq.number, seq.name);
    }
    println!();

    // Example 5: Text search
    println!("Example 5: Text Search for 'triangular'");
    println!("----------------------------------------");
    let text_results = client.search("triangular")?;
    println!("Found {} sequences", text_results.len());
    for (i, seq) in text_results.iter().take(5).enumerate() {
        println!("  {}. {}: {}", i + 1, seq.number, seq.name);
    }
    println!();

    // Example 6: Get specific number of terms
    println!("Example 6: Get First 8 Terms of Fibonacci");
    println!("------------------------------------------");
    let terms = client.get_terms("A000045", 8)?;
    println!("First 8 Fibonacci numbers: {:?}", terms);
    println!();

    // Example 7: Demonstrate flexible A-number input
    println!("Example 7: Flexible A-Number Input");
    println!("-----------------------------------");
    let formats = vec!["A000045", "a000045", "000045", "45"];
    for format in formats {
        if let Some(seq) = client.lookup(format)? {
            println!("  Input: {:8} -> Sequence: {}", format, seq.number);
        }
    }
    println!();

    println!("=== Demo Complete ===");

    Ok(())
}
