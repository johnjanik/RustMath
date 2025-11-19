//! # RustMath Feature Detection System
//!
//! This crate provides compile-time and runtime feature detection for RustMath.
//! It enables:
//! - Optional dependency management
//! - Runtime feature checking
//! - Fallback implementations when features are not available
//! - Integration with Cargo feature flags
//!
//! ## Architecture
//!
//! The feature system is organized into several categories:
//! - **External Libraries**: GMP, MPFR, FLINT, PARI
//! - **Performance**: Parallel processing, SIMD
//! - **Platform**: Native optimizations
//! - **Optional**: Databases, plotting
//! - **Experimental**: Unstable features
//!
//! ## Example Usage
//!
//! ### Basic Feature Detection
//!
//! ```rust
//! use rustmath_features::{has_feature, Feature};
//!
//! // Check if a feature is available
//! if has_feature(Feature::Gmp) {
//!     println!("GMP support is available");
//! }
//!
//! // Get all enabled features
//! for feature in rustmath_features::enabled_features() {
//!     println!("Feature {} is enabled", feature.name());
//! }
//! ```
//!
//! ### Using Fallbacks
//!
//! ```rust
//! use rustmath_features::{with_fallback, Feature};
//!
//! # fn gmp_factorial(n: u64) -> u64 { (1..=n).product() }
//! # fn pure_rust_factorial(n: u64) -> u64 { (1..=n).product() }
//! // Use feature-specific implementation with fallback
//! let result = with_fallback!(
//!     Feature::Gmp => gmp_factorial(10),
//!     pure_rust_factorial(10)
//! );
//! ```
//!
//! ### Advanced: Custom Fallback Operations
//!
//! ```rust
//! use rustmath_features::fallback::{FallbackOperation, ExecutionContext, SelectionStrategy};
//! use rustmath_features::Feature;
//!
//! struct MyOperation;
//!
//! impl FallbackOperation for MyOperation {
//!     type Input = i32;
//!     type Output = i32;
//!     type Error = ();
//!
//!     fn execute_with_feature(&self, input: &Self::Input, feature: Feature) -> Result<Self::Output, Self::Error> {
//!         match feature {
//!             Feature::Simd => Ok(input * 2), // SIMD implementation
//!             _ => Err(())
//!         }
//!     }
//!
//!     fn fallback(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> {
//!         Ok(input * 2) // Pure Rust fallback
//!     }
//! }
//!
//! let op = MyOperation;
//! let ctx = ExecutionContext::new().with_strategy(SelectionStrategy::Fastest);
//! let result = op.execute(&42, &ctx).unwrap();
//! assert_eq!(result, 84);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

pub mod fallback;
pub mod conditional;

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

use once_cell::sync::Lazy;

#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap as HashMap;

/// Represents a feature that can be enabled or disabled in RustMath
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Feature {
    /// Standard library support
    Std,
    /// GNU Multiple Precision Arithmetic Library
    Gmp,
    /// Multiple Precision Floating-Point Reliable Library
    Mpfr,
    /// Fast Library for Number Theory
    Flint,
    /// PARI/GP computer algebra system
    Pari,
    /// Parallel algorithms (rayon)
    Parallel,
    /// SIMD optimizations
    Simd,
    /// Native CPU optimizations
    Native,
    /// Experimental features
    Experimental,
    /// Database integrations
    Databases,
    /// Plotting capabilities
    Plotting,
}

impl Feature {
    /// Returns the string name of the feature
    pub const fn name(&self) -> &'static str {
        match self {
            Feature::Std => "std",
            Feature::Gmp => "gmp",
            Feature::Mpfr => "mpfr",
            Feature::Flint => "flint",
            Feature::Pari => "pari",
            Feature::Parallel => "parallel",
            Feature::Simd => "simd",
            Feature::Native => "native",
            Feature::Experimental => "experimental",
            Feature::Databases => "databases",
            Feature::Plotting => "plotting",
        }
    }

    /// Returns a description of the feature
    pub const fn description(&self) -> &'static str {
        match self {
            Feature::Std => "Standard library support",
            Feature::Gmp => "GNU Multiple Precision Arithmetic Library bindings",
            Feature::Mpfr => "Multiple Precision Floating-Point Reliable Library bindings",
            Feature::Flint => "Fast Library for Number Theory bindings",
            Feature::Pari => "PARI/GP computer algebra system bindings",
            Feature::Parallel => "Parallel algorithms using rayon",
            Feature::Simd => "SIMD (Single Instruction Multiple Data) optimizations",
            Feature::Native => "Native CPU-specific optimizations",
            Feature::Experimental => "Experimental and unstable features",
            Feature::Databases => "Database integrations (OEIS, LMFDB, Cremona)",
            Feature::Plotting => "Plotting and visualization capabilities",
        }
    }

    /// Check if this feature is enabled at compile time
    pub const fn is_enabled(&self) -> bool {
        match self {
            Feature::Std => cfg!(feature = "std"),
            Feature::Gmp => cfg!(feature = "gmp"),
            Feature::Mpfr => cfg!(feature = "mpfr"),
            Feature::Flint => cfg!(feature = "flint"),
            Feature::Pari => cfg!(feature = "pari"),
            Feature::Parallel => cfg!(feature = "parallel"),
            Feature::Simd => cfg!(feature = "simd"),
            Feature::Native => cfg!(feature = "native"),
            Feature::Experimental => cfg!(feature = "experimental"),
            Feature::Databases => cfg!(feature = "databases"),
            Feature::Plotting => cfg!(feature = "plotting"),
        }
    }

    /// Returns all available features
    pub const fn all() -> &'static [Feature] {
        &[
            Feature::Std,
            Feature::Gmp,
            Feature::Mpfr,
            Feature::Flint,
            Feature::Pari,
            Feature::Parallel,
            Feature::Simd,
            Feature::Native,
            Feature::Experimental,
            Feature::Databases,
            Feature::Plotting,
        ]
    }
}

/// Global feature registry for runtime queries
static FEATURE_REGISTRY: Lazy<FeatureRegistry> = Lazy::new(FeatureRegistry::new);

/// Runtime registry of enabled features
pub struct FeatureRegistry {
    features: HashMap<Feature, bool>,
}

impl FeatureRegistry {
    /// Create a new feature registry by detecting enabled features at compile time
    fn new() -> Self {
        let mut features = HashMap::new();

        for &feature in Feature::all() {
            features.insert(feature, feature.is_enabled());
        }

        Self { features }
    }

    /// Check if a feature is enabled
    pub fn has(&self, feature: Feature) -> bool {
        self.features.get(&feature).copied().unwrap_or(false)
    }

    /// Get all enabled features
    pub fn enabled_features(&self) -> Vec<Feature> {
        self.features
            .iter()
            .filter_map(|(&f, &enabled)| if enabled { Some(f) } else { None })
            .collect()
    }

    /// Get a feature summary as a string
    #[cfg(feature = "std")]
    pub fn summary(&self) -> String {
        let enabled: Vec<_> = self.enabled_features();
        if enabled.is_empty() {
            "No optional features enabled".to_string()
        } else {
            let names: Vec<_> = enabled.iter().map(|f| f.name()).collect();
            format!("Enabled features: {}", names.join(", "))
        }
    }
}

/// Check if a feature is enabled at runtime
///
/// # Example
///
/// ```
/// use rustmath_features::{has_feature, Feature};
///
/// if has_feature(Feature::Parallel) {
///     println!("Parallel processing is available");
/// }
/// ```
pub fn has_feature(feature: Feature) -> bool {
    FEATURE_REGISTRY.has(feature)
}

/// Get all enabled features
///
/// # Example
///
/// ```
/// use rustmath_features::enabled_features;
///
/// for feature in enabled_features() {
///     println!("Feature {} is enabled", feature.name());
/// }
/// ```
pub fn enabled_features() -> Vec<Feature> {
    FEATURE_REGISTRY.enabled_features()
}

/// Get a summary of enabled features
///
/// # Example
///
/// ```
/// use rustmath_features::feature_summary;
///
/// println!("{}", feature_summary());
/// ```
#[cfg(feature = "std")]
pub fn feature_summary() -> String {
    FEATURE_REGISTRY.summary()
}

/// Macro to conditionally compile code based on feature availability
///
/// # Example
///
/// ```ignore
/// use rustmath_features::if_feature;
///
/// if_feature! {
///     gmp => {
///         // Code that uses GMP
///         use gmp::mpz::Mpz;
///     },
///     else => {
///         // Fallback code
///         use num_bigint::BigInt;
///     }
/// }
/// ```
#[macro_export]
macro_rules! if_feature {
    ($feature:ident => $then:block, else => $else:block) => {
        #[cfg(feature = stringify!($feature))]
        $then

        #[cfg(not(feature = stringify!($feature)))]
        $else
    };
    ($feature:ident => $then:block) => {
        #[cfg(feature = stringify!($feature))]
        $then
    };
}

/// Macro to execute code with a fallback based on feature availability
///
/// # Example
///
/// ```ignore
/// use rustmath_features::with_fallback;
///
/// let result = with_fallback!(
///     Feature::Gmp => gmp_factorial(10),
///     pure_rust_factorial(10)
/// );
/// ```
#[macro_export]
macro_rules! with_fallback {
    ($feature:expr => $primary:expr, $fallback:expr) => {
        if $crate::has_feature($feature) {
            $primary
        } else {
            $fallback
        }
    };
}

/// Trait for types that can provide feature-specific implementations with fallbacks
///
/// This trait enables polymorphic behavior where a type can have multiple
/// implementations depending on available features.
///
/// # Example
///
/// ```ignore
/// use rustmath_features::{FeatureDependent, Feature};
///
/// struct BigInteger;
///
/// impl FeatureDependent for BigInteger {
///     type Output = String;
///
///     fn with_feature(&self, feature: Feature) -> Option<Self::Output> {
///         match feature {
///             Feature::Gmp => Some(self.gmp_impl()),
///             _ => None
///         }
///     }
///
///     fn fallback(&self) -> Self::Output {
///         self.pure_rust_impl()
///     }
/// }
/// ```
pub trait FeatureDependent {
    /// The output type of the operation
    type Output;

    /// Execute with a specific feature if available
    fn with_feature(&self, feature: Feature) -> Option<Self::Output>;

    /// Fallback implementation (always available)
    fn fallback(&self) -> Self::Output;

    /// Execute with the best available implementation
    fn execute(&self, preferred: &[Feature]) -> Self::Output {
        for &feature in preferred {
            if has_feature(feature) {
                if let Some(result) = self.with_feature(feature) {
                    return result;
                }
            }
        }
        self.fallback()
    }
}

/// Capability flags for runtime optimization hints
#[derive(Debug, Clone, Copy)]
pub struct Capabilities {
    /// CPU supports AVX instructions
    pub has_avx: bool,
    /// CPU supports AVX2 instructions
    pub has_avx2: bool,
    /// CPU supports AVX-512 instructions
    pub has_avx512: bool,
    /// CPU supports NEON instructions (ARM)
    pub has_neon: bool,
    /// Number of available CPU cores
    pub num_cores: usize,
}

impl Capabilities {
    /// Detect CPU capabilities at runtime
    pub fn detect() -> Self {
        Self {
            has_avx: cfg!(target_feature = "avx"),
            has_avx2: cfg!(target_feature = "avx2"),
            has_avx512: cfg!(target_feature = "avx512f"),
            has_neon: cfg!(target_feature = "neon"),
            num_cores: num_cpus(),
        }
    }

    /// Check if SIMD is available
    pub fn has_simd(&self) -> bool {
        self.has_avx || self.has_avx2 || self.has_avx512 || self.has_neon
    }
}

/// Get the number of available CPU cores
fn num_cpus() -> usize {
    #[cfg(feature = "std")]
    {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }

    #[cfg(not(feature = "std"))]
    {
        1
    }
}

/// Global CPU capabilities
static CAPABILITIES: Lazy<Capabilities> = Lazy::new(Capabilities::detect);

/// Get CPU capabilities
pub fn capabilities() -> &'static Capabilities {
    &CAPABILITIES
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_names() {
        assert_eq!(Feature::Gmp.name(), "gmp");
        assert_eq!(Feature::Mpfr.name(), "mpfr");
        assert_eq!(Feature::Parallel.name(), "parallel");
    }

    #[test]
    fn test_feature_detection() {
        // These should always be available based on default features
        #[cfg(feature = "std")]
        assert!(has_feature(Feature::Std));

        // Check that we can query features
        let _ = has_feature(Feature::Gmp);
        let _ = has_feature(Feature::Parallel);
    }

    #[test]
    fn test_enabled_features() {
        let features = enabled_features();

        #[cfg(feature = "std")]
        assert!(features.contains(&Feature::Std));

        // Should return a valid list (may be empty)
        assert!(features.len() <= Feature::all().len());
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_feature_summary() {
        let summary = feature_summary();
        assert!(!summary.is_empty());
    }

    #[test]
    fn test_capabilities() {
        let caps = capabilities();

        // Cores should be at least 1
        assert!(caps.num_cores >= 1);

        // SIMD detection should work
        let _ = caps.has_simd();
    }

    #[test]
    fn test_feature_descriptions() {
        for feature in Feature::all() {
            assert!(!feature.description().is_empty());
        }
    }
}
