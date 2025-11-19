//! Basic usage example for rustmath-features
//!
//! Run with: cargo run --example basic_usage

use rustmath_features::{Feature, has_feature, enabled_features, capabilities};

fn main() {
    println!("=== RustMath Feature Detection Demo ===\n");

    // Check specific features
    println!("Feature Detection:");
    for feature in Feature::all() {
        let status = if has_feature(*feature) { "✓" } else { "✗" };
        println!("  {} {}: {}", status, feature.name(), feature.description());
    }

    println!("\n=== Enabled Features ===");
    let enabled = enabled_features();
    if enabled.is_empty() {
        println!("  No optional features enabled (using defaults)");
    } else {
        for feature in enabled {
            println!("  ✓ {}", feature.name());
        }
    }

    #[cfg(feature = "std")]
    {
        println!("\n=== Feature Summary ===");
        println!("  {}", rustmath_features::feature_summary());
    }

    println!("\n=== CPU Capabilities ===");
    let caps = capabilities();
    println!("  CPU Cores: {}", caps.num_cores);
    println!("  AVX: {}", if caps.has_avx { "Yes" } else { "No" });
    println!("  AVX2: {}", if caps.has_avx2 { "Yes" } else { "No" });
    println!("  AVX-512: {}", if caps.has_avx512 { "Yes" } else { "No" });
    println!("  NEON: {}", if caps.has_neon { "Yes" } else { "No" });
    println!("  SIMD Available: {}", if caps.has_simd() { "Yes" } else { "No" });

    println!("\n=== Conditional Code Example ===");
    demonstrate_conditional_execution();
}

fn demonstrate_conditional_execution() {
    // Example: Choose algorithm based on available features
    let n = 1000u64;

    #[cfg(feature = "parallel")]
    {
        println!("  Using parallel algorithm for n={}", n);
        // Would use rayon or similar
    }

    #[cfg(not(feature = "parallel"))]
    {
        println!("  Using sequential algorithm for n={}", n);
        // Use standard sequential approach
    }

    // Using the with_fallback! macro
    use rustmath_features::with_fallback;

    fn optimized_sum(_n: u64) -> u64 {
        42 // Simulated optimized version
    }

    fn fallback_sum(_n: u64) -> u64 {
        42 // Simulated fallback version
    }

    let result = with_fallback!(
        Feature::Simd => optimized_sum(n),
        fallback_sum(n)
    );

    println!("  Result: {}", result);
}
