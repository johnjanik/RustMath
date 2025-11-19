//! Example demonstrating the FallbackOperation trait
//!
//! Run with: cargo run --example fallback_operation

use rustmath_features::Feature;
use rustmath_features::fallback::{
    FallbackOperation, ExecutionContext, SelectionStrategy, OperationBuilder
};

/// Example: Computing factorial with different implementations
struct FactorialOperation;

impl FallbackOperation for FactorialOperation {
    type Input = u64;
    type Output = u64;
    type Error = String;

    fn execute_with_feature(
        &self,
        input: &Self::Input,
        feature: Feature,
    ) -> Result<Self::Output, Self::Error> {
        match feature {
            Feature::Gmp => {
                // In a real implementation, this would use GMP
                println!("  Using GMP implementation");
                Ok((1..=*input).product())
            }
            Feature::Parallel => {
                // In a real implementation, this would use rayon
                println!("  Using parallel implementation");
                Ok((1..=*input).product())
            }
            _ => Err(format!("Feature {:?} not supported for factorial", feature)),
        }
    }

    fn fallback(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> {
        println!("  Using pure Rust fallback implementation");
        Ok((1..=*input).product())
    }
}

fn main() {
    println!("=== Fallback Operation Example ===\n");

    let operation = FactorialOperation;
    let n = 10u64;

    // Example 1: Default execution (tries preferred features in order)
    println!("Example 1: Default execution");
    let ctx = ExecutionContext::default();
    match operation.execute(&n, &ctx) {
        Ok(result) => println!("  {}! = {}\n", n, result),
        Err(e) => println!("  Error: {}\n", e),
    }

    // Example 2: Force fallback
    println!("Example 2: Force fallback");
    let ctx = ExecutionContext::new()
        .with_strategy(SelectionStrategy::AlwaysFallback);
    match operation.execute(&n, &ctx) {
        Ok(result) => println!("  {}! = {}\n", n, result),
        Err(e) => println!("  Error: {}\n", e),
    }

    // Example 3: Prefer specific features
    println!("Example 3: Prefer parallel over external libs");
    let ctx = ExecutionContext::new()
        .with_features(vec![Feature::Parallel, Feature::Gmp]);
    match operation.execute(&n, &ctx) {
        Ok(result) => println!("  {}! = {}\n", n, result),
        Err(e) => println!("  Error: {}\n", e),
    }

    // Example 4: Use builder pattern
    println!("Example 4: Using builder pattern");
    let (op, ctx) = OperationBuilder::new(FactorialOperation)
        .strategy(SelectionStrategy::Fastest)
        .build();

    match op.execute(&n, &ctx) {
        Ok(result) => println!("  {}! = {}\n", n, result),
        Err(e) => println!("  Error: {}\n", e),
    }

    // Example 5: Most accurate implementation
    println!("Example 5: Most accurate implementation (arbitrary precision)");
    let ctx = ExecutionContext::new()
        .with_strategy(SelectionStrategy::MostAccurate);
    match operation.execute(&n, &ctx) {
        Ok(result) => println!("  {}! = {}\n", n, result),
        Err(e) => println!("  Error: {}\n", e),
    }
}
