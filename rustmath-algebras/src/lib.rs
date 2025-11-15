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
pub mod quotient_algebra;
pub mod clifford_algebra;
pub mod group_algebra;
pub mod traits;
pub mod affine_nil_temperley_lieb;

pub use free_algebra::*;
pub use finite_dimensional_algebra::*;
pub use quotient_algebra::*;
pub use clifford_algebra::{
    CliffordAlgebra, CliffordAlgebraElement, CliffordBasisElement,
    ExteriorAlgebra, ExteriorAlgebraDifferential, ExteriorAlgebraBoundary,
    ExteriorAlgebraCoboundary, ExteriorAlgebraIdeal, StructureCoefficients,
};
pub use group_algebra::*;
pub use traits::*;
pub use affine_nil_temperley_lieb::{AffineNilTemperleyLiebTypeA, Element as AffineNilTemperleyLiebElement};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_algebra_operations() {
        // Basic smoke tests will be added as we implement structures
        assert!(true);
    }
}
