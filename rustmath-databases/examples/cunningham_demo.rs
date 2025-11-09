//! Demonstration of the Cunningham Tables interface
//!
//! Run with: cargo run --example cunningham_demo

use rustmath_databases::cunningham::{CunninghamTables, CunninghamNumber, Factor, Factorization};

fn main() {
    println!("=== Cunningham Tables Demo ===\n");

    let tables = CunninghamTables::new();

    // Example 1: Mersenne numbers (2^n - 1)
    println!("1. Mersenne Numbers (2^n - 1):");
    println!("   Factorizations of 2^n - 1 for small n\n");

    for exp in 2..=10 {
        let num = CunninghamNumber::new(2, exp, false);
        match tables.factorization(&num) {
            Ok(fact) => {
                println!("   {}", fact);
            }
            Err(_) => {
                println!("   {} - not in tables", num);
            }
        }
    }

    // Example 2: Fermat numbers (2^n + 1)
    println!("\n2. Fermat Numbers (2^n + 1):");
    println!("   The first few Fermat numbers\n");

    for exp in 2..=8 {
        let num = CunninghamNumber::new(2, exp, true);
        match tables.factorization(&num) {
            Ok(fact) => {
                println!("   {}", fact);
                if fact.factors.len() == 1 {
                    println!("      ↳ This is a Fermat prime!");
                }
            }
            Err(_) => {
                println!("   {} - not in tables", num);
            }
        }
    }

    // Example 3: Base 3 factorizations
    println!("\n3. Base 3 Numbers:");
    println!("   Factorizations of 3^n ± 1\n");

    for exp in 2..=5 {
        let minus = CunninghamNumber::new(3, exp, false);
        let plus = CunninghamNumber::new(3, exp, true);

        if let Ok(fact) = tables.factorization(&minus) {
            println!("   {}", fact);
        }
        if let Ok(fact) = tables.factorization(&plus) {
            println!("   {}", fact);
        }
    }

    // Example 4: Querying by base
    println!("\n4. All Base 2 Factorizations:");
    println!("   All 2^n ± 1 factorizations in the tables\n");

    let base2_facts = tables.factorizations_for_base(2);
    println!("   Found {} factorizations for base 2", base2_facts.len());

    // Example 5: Custom factorizations
    println!("\n5. Adding Custom Factorizations:");
    println!("   Extending the tables with new entries\n");

    let mut custom_tables = CunninghamTables::new();

    // Add 2^11 - 1 = 2047 = 23 × 89
    let num = CunninghamNumber::new(2, 11, false);
    let mut fact = Factorization::new(num.clone());
    fact.add_factor(Factor::new_prime("23".to_string()));
    fact.add_factor(Factor::new_prime("89".to_string()));
    fact.is_complete = true;

    custom_tables.add_factorization(fact).unwrap();

    if let Ok(retrieved) = custom_tables.factorization(&num) {
        println!("   Added: {}", retrieved);
    }

    // Example 6: Notation and display
    println!("\n6. Cunningham Notation:");
    println!("   Standard compact notation for these numbers\n");

    let examples = vec![
        CunninghamNumber::new(2, 10, false),
        CunninghamNumber::new(2, 8, true),
        CunninghamNumber::new(3, 5, false),
        CunninghamNumber::new(5, 3, true),
    ];

    for num in examples {
        println!("   {} → notation: {}", num, num.notation());
    }

    // Example 7: Supported bases
    println!("\n7. Supported Bases:");
    println!("   The Cunningham Project maintains tables for specific bases\n");

    let bases = vec![2, 3, 5, 6, 7, 10, 11, 12];
    println!("   Valid bases: {:?}", bases);

    let valid = CunninghamNumber::new(2, 5, false);
    let invalid = CunninghamNumber::new(13, 5, false);

    match valid.validate() {
        Ok(_) => println!("   {} - Valid base", valid.base),
        Err(e) => println!("   Error: {}", e),
    }

    match invalid.validate() {
        Ok(_) => println!("   {} - Valid base", invalid.base),
        Err(e) => println!("   {} - {}", invalid.base, e),
    }

    // Example 8: Mathematical properties
    println!("\n8. Mathematical Properties:");
    println!("   Interesting facts about Cunningham numbers\n");

    println!("   • Mersenne primes: 2^p - 1 where the result is prime");
    println!("   • Fermat primes: 2^(2^n) + 1 where the result is prime");
    println!("   • Only 5 known Fermat primes: 3, 5, 17, 257, 65537");
    println!("   • Mersenne primes are used in cryptography and perfect numbers");

    println!("\n=== Demo Complete ===");
}
