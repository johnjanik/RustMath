# rustmath-features

Feature detection and optional dependency management for RustMath.

## Overview

`rustmath-features` provides a comprehensive system for managing optional features and dependencies in RustMath. It enables:

- **Compile-time feature detection** using Cargo feature flags
- **Runtime feature queries** to check which features are enabled
- **Automatic fallbacks** when optional dependencies are not available
- **Performance optimization** by selecting the best available implementation
- **CPU capability detection** for SIMD and parallel processing

## Features

### Core Capabilities

- **Feature Detection**: Check at compile-time and runtime which features are enabled
- **Fallback System**: Automatically fall back to pure Rust implementations when optional libraries are unavailable
- **Selection Strategies**: Choose implementations based on speed, accuracy, or availability
- **CPU Introspection**: Detect SIMD capabilities and CPU core count

### Available Feature Flags

#### External Libraries (Future Integration)
- `gmp` - GNU Multiple Precision Arithmetic Library
- `mpfr` - Multiple Precision Floating-Point Reliable Library
- `flint` - Fast Library for Number Theory
- `pari` - PARI/GP computer algebra system

#### Performance
- `parallel` - Enable parallel algorithms
- `simd` - SIMD optimizations
- `native` - Native CPU optimizations

#### Optional Functionality
- `databases` - Database integrations (OEIS, LMFDB, Cremona)
- `plotting` - Plotting and visualization

#### Other
- `experimental` - Experimental and unstable features
- `std` - Standard library support (enabled by default)
- `full` - Enable all features

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
rustmath-features = { path = "../rustmath-features" }
```

### Basic Feature Detection

```rust
use rustmath_features::{Feature, has_feature, enabled_features};

fn main() {
    // Check if a specific feature is available
    if has_feature(Feature::Gmp) {
        println!("GMP is available!");
    }

    // List all enabled features
    for feature in enabled_features() {
        println!("Enabled: {}", feature.name());
    }

    // Get a summary
    #[cfg(feature = "std")]
    println!("{}", rustmath_features::feature_summary());
}
```

### Using Fallbacks

```rust
use rustmath_features::{with_fallback, Feature};

fn gmp_factorial(n: u64) -> u64 {
    // GMP implementation
    (1..=n).product()
}

fn pure_rust_factorial(n: u64) -> u64 {
    // Pure Rust implementation
    (1..=n).product()
}

// Automatically use GMP if available, otherwise fall back
let result = with_fallback!(
    Feature::Gmp => gmp_factorial(10),
    pure_rust_factorial(10)
);
```

### Advanced: Custom Operations

```rust
use rustmath_features::fallback::{
    FallbackOperation, ExecutionContext, SelectionStrategy
};
use rustmath_features::Feature;

struct MyOperation;

impl FallbackOperation for MyOperation {
    type Input = i32;
    type Output = i32;
    type Error = ();

    fn execute_with_feature(
        &self,
        input: &Self::Input,
        feature: Feature,
    ) -> Result<Self::Output, Self::Error> {
        match feature {
            Feature::Simd => Ok(input * 2), // Optimized version
            _ => Err(())
        }
    }

    fn fallback(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> {
        Ok(input * 2) // Pure Rust version
    }
}

// Use with context
let op = MyOperation;
let ctx = ExecutionContext::new()
    .with_strategy(SelectionStrategy::Fastest);

let result = op.execute(&42, &ctx).unwrap();
```

### Conditional Compilation

```rust
use rustmath_features::if_feature;

if_feature! {
    gmp => {
        use gmp::mpz::Mpz;
        type BigInt = Mpz;
    },
    else => {
        use num_bigint::BigInt;
    }
}
```

## Examples

Run the examples to see the feature system in action:

```bash
# Basic usage
cargo run --example basic_usage

# Fallback operations
cargo run --example fallback_operation

# With features enabled
cargo run --example basic_usage --features parallel,simd
```

## Testing

Run the test suite:

```bash
# Basic tests
cargo test

# With specific features
cargo test --features parallel

# All features
cargo test --all-features
```

## Architecture

### Feature Categories

1. **External Libraries**: Optional bindings to high-performance C libraries
2. **Performance**: Optimizations like SIMD and parallel processing
3. **Platform**: Platform-specific optimizations
4. **Optional**: Additional functionality like databases and plotting
5. **Experimental**: Unstable features under development

### Design Principles

- **Zero-cost abstractions**: Feature detection has minimal runtime overhead
- **Graceful degradation**: Always provide pure Rust fallbacks
- **Type safety**: Feature-dependent code is type-safe at compile time
- **Composability**: Features can be combined without conflicts

## Integration with RustMath

Other RustMath crates can depend on `rustmath-features` to:

1. Detect available features at runtime
2. Provide optimized implementations when features are available
3. Fall back gracefully when features are missing
4. Allow users to control which implementations are used

Example integration in another crate:

```rust
use rustmath_features::{has_feature, Feature, with_fallback};

pub fn optimized_multiply(a: i32, b: i32) -> i32 {
    with_fallback!(
        Feature::Simd => simd_multiply(a, b),
        a * b
    )
}

#[cfg(feature = "simd")]
fn simd_multiply(a: i32, b: i32) -> i32 {
    // SIMD implementation
    a * b
}
```

## License

GPL-2.0-or-later

## Contributing

This is part of the RustMath project. See the main repository for contribution guidelines.
