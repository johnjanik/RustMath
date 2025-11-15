//! Algebra Catalog
//!
//! This module provides convenient access to various algebra implementations.
//! It serves as a centralized discovery point for different algebraic structures
//! available in RustMath.
//!
//! Corresponds to sage.algebras.catalog
//!
//! # Usage
//!
//! ```
//! use rustmath_algebras::catalog;
//! use rustmath_integers::Integer;
//!
//! // Access algebras through the catalog
//! let free_alg = catalog::FreeAlgebra::<Integer>::new(3);
//! ```
//!
//! # Available Algebras
//!
//! - **FreeAlgebra**: Non-commutative polynomial algebras
//! - **QuotientAlgebra**: Quotient algebras A/I
//! - **CliffordAlgebra**: Clifford algebras and exterior algebras
//! - **GroupAlgebra**: Group algebras
//! - **FiniteDimensionalAlgebra**: Finite dimensional algebras
//! - **ArikiKoikeAlgebra**: Ariki-Koike Hecke algebras
//! - **CubicHeckeAlgebra**: Cubic Hecke algebras
//! - **JordanAlgebra**: Jordan algebras
//! - **HallAlgebra**: Hall algebras
//! - **AskeyWilsonAlgebra**: Askey-Wilson algebras
//! - **ClusterAlgebra**: Cluster algebras
//! - **DownUpAlgebra**: Down-Up algebras
//! - **FreeZinbielAlgebra**: Free Zinbiel algebras
//!
//! # Examples
//!
//! ```
//! use rustmath_algebras::catalog;
//! use rustmath_integers::Integer;
//!
//! // Create a free algebra on 2 generators
//! let algebra = catalog::FreeAlgebra::<Integer>::new(2);
//! assert_eq!(algebra.num_generators(), 2);
//! ```

// Re-export all major algebra types for convenient access
pub use crate::free_algebra::FreeAlgebra;
pub use crate::quotient_algebra::{QuotientAlgebra, QuotientAlgebraElement};
pub use crate::clifford_algebra::{CliffordAlgebra, ExteriorAlgebra};
pub use crate::group_algebra::GroupAlgebra;
pub use crate::finite_dimensional_algebra::FiniteDimensionalAlgebra;
pub use crate::ariki_koike_algebra::ArikiKoikeAlgebra;
pub use crate::cubic_hecke_algebra::CubicHeckeAlgebra;
pub use crate::jordan_algebra::JordanAlgebra;
pub use crate::hall_algebra::HallAlgebra;
pub use crate::askey_wilson::AskeyWilsonAlgebra;
pub use crate::cluster_algebra::ClusterAlgebra;
pub use crate::down_up_algebra::DownUpAlgebra;
pub use crate::free_zinbiel_algebra::FreeZinbielAlgebra;
pub use crate::affine_nil_temperley_lieb::AffineNilTemperleyLiebTypeA;
pub use crate::associated_graded::AssociatedGradedAlgebra;
pub use crate::cellular_basis::CellularBasis;
pub use crate::commutative_dga::{GCAlgebra, DifferentialGCAlgebra};
pub use crate::finite_gca::FiniteGCAlgebra;

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_catalog_free_algebra() {
        // Test that we can access FreeAlgebra through catalog
        let algebra = FreeAlgebra::<Integer>::new(3);
        assert_eq!(algebra.rank(), 3);
    }

    #[test]
    fn test_catalog_access() {
        // Verify catalog provides access to key algebra types
        // This is more of a compile-time check that all types are accessible

        // Free algebra
        let _free: Option<FreeAlgebra<Integer>> = None;

        // Quotient algebra
        let _quot: Option<QuotientAlgebra<Integer>> = None;

        // Clifford algebra
        let _cliff: Option<CliffordAlgebra<Integer>> = None;

        // This test passes if it compiles, showing all types are accessible
        assert!(true);
    }
}
