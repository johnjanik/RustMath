//! # RustMath Monoids
//!
//! This crate provides functionality for monoids and semigroups,
//! corresponding to SageMath's sage.monoids module.
//!
//! ## Modules
//!
//! - `monoid`: Base monoid structures
//! - `free_monoid`: Free monoids
//! - `free_abelian_monoid`: Free abelian monoids
//! - `indexed_free_monoid`: Indexed free monoids
//! - `string_monoid`: String monoids
//! - `automatic_semigroup`: Automatic semigroups

pub mod monoid;
pub mod free_monoid;
pub mod free_abelian_monoid;
pub mod indexed_free_monoid;
pub mod string_monoid;
pub mod automatic_semigroup;

// Re-export commonly used types
pub use monoid::Monoid;
pub use free_monoid::{FreeMonoid, FreeMonoidElement};
pub use free_abelian_monoid::{FreeAbelianMonoid, FreeAbelianMonoidElement};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_monoid_imports() {
        // Just verify that the modules exist
        let _ = FreeMonoid::new(vec!["x".to_string(), "y".to_string()]);
    }
}
