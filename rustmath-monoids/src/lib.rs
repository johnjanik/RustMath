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
//! - `string_ops`: String operations
//! - `automatic_semigroup`: Automatic semigroups
//! - `hecke_monoid`: Hecke monoids
//! - `trace_monoid`: Trace monoids (partially commutative)

pub mod monoid;
pub mod free_monoid;
pub mod free_monoid_core_impls;
pub mod free_abelian_monoid;
pub mod indexed_free_monoid;
pub mod string_monoid;
pub mod string_ops;
pub mod automatic_semigroup;
pub mod hecke_monoid;
pub mod trace_monoid;
// MAGMA handbook chapter 77 (finitely presented semigroups) and 78 (monoids
// given by rewrite systems), consuming the Knuth–Bendix engine in
// rustmath-automata.
pub mod free_semigroup;
pub mod fp_semigroup;
pub mod rws_monoid;

// Re-export commonly used types
pub use monoid::Monoid;
pub use free_monoid::{FreeMonoid, FreeMonoidElement};
pub use free_abelian_monoid::{FreeAbelianMonoid, FreeAbelianMonoidElement};
pub use hecke_monoid::{HeckeMonoid, HeckeMonoidElement};
pub use trace_monoid::{TraceMonoid, TraceMonoidElement};
pub use free_semigroup::{FreeSemigroup, FreeSemigroupElement};
pub use fp_semigroup::{FpSemigroup, Relation, SubKind, SubStructure};
pub use rws_monoid::{RwsMonoid, RwsMonoidElement};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_monoid_imports() {
        // Just verify that the modules exist
        let _ = FreeMonoid::new(vec!["x".to_string(), "y".to_string()]);
    }
}
