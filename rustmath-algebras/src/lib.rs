//! RustMath Algebras - Advanced algebraic structures
//!
//! This crate provides implementations of various algebraic structures including:
//! - Free algebras
//! - Quotient algebras
//! - Clifford algebras
//! - Finite dimensional algebras
//! - Group algebras
//! - Hecke algebras
//! - Lie algebras
//!
//! Corresponds to the sage.algebras module from SageMath.

pub mod free_algebra;
pub mod finite_dimensional_algebra;
pub mod finite_dimensional_algebra_ideal;
pub mod quotient_algebra;
pub mod clifford_algebra;
pub mod group_algebra;
pub mod traits;
pub mod affine_nil_temperley_lieb;
pub mod down_up_algebra;
pub mod algebra_morphism;
pub mod algebra_with_parent;
pub mod cached_algebra;
pub mod askey_wilson;
pub mod associated_graded;
pub mod cellular_basis;
pub mod cluster_algebra;
pub mod commutative_dga;

pub use free_algebra::*;
pub use finite_dimensional_algebra::*;
pub use finite_dimensional_algebra_ideal::FiniteDimensionalAlgebraIdeal;
pub use quotient_algebra::*;
pub use clifford_algebra::{
    CliffordAlgebra, CliffordAlgebraElement, CliffordBasisElement, CliffordAlgebraIndices,
    ExteriorAlgebra, ExteriorAlgebraDifferential, ExteriorAlgebraBoundary,
    ExteriorAlgebraCoboundary, ExteriorAlgebraIdeal, StructureCoefficients,
};
pub use group_algebra::*;
pub use traits::*;
pub use affine_nil_temperley_lieb::{AffineNilTemperleyLiebTypeA, Element as AffineNilTemperleyLiebElement};
pub use algebra_morphism::{AlgebraMorphism, AlgebraEndomorphism, AlgebraAutomorphism};
pub use askey_wilson::{AskeyWilsonAlgebra, AskeyWilsonIndex};
pub use associated_graded::AssociatedGradedAlgebra;
pub use cellular_basis::{CellularBasis, CellularIndex};
pub use cluster_algebra::{
    ClusterAlgebra, ClusterAlgebraElement, ClusterAlgebraSeed,
    PrincipalClusterAlgebraElement, GVector, DVector, ExchangeMatrix,
};
pub use commutative_dga::{
    GCAlgebra, GCAlgebraMultigraded, Differential, DifferentialMultigraded,
    DifferentialGCAlgebra, DifferentialGCAlgebraMultigraded,
    CohomologyClass, GCAlgebraHomset, GCAlgebraMorphism,
    Degree, Generator, GCAlgebraElement,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_algebra_operations() {
        // Basic smoke tests will be added as we implement structures
        assert!(true);
    }
}
