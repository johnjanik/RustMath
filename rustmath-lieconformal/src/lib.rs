//! RustMath Lie Conformal Algebras
//!
//! This crate provides infrastructure for working with Lie conformal algebras,
//! which are algebraic structures that arise in conformal field theory and
//! vertex operator algebras.
//!
//! # Mathematical Background
//!
//! A Lie conformal algebra is an R[∂]-module L with a λ-bracket operation
//! [·_λ ·]: L ⊗ L → R[λ] ⊗ L satisfying:
//!
//! 1. Sesquilinearity: [∂a_λ b] = -λ[a_λ b], [a_λ ∂b] = (∂ + λ)[a_λ b]
//! 2. Skew-symmetry: [b_λ a] = -[a_{-λ-∂} b]
//! 3. Jacobi identity: [a_λ [b_μ c]] = [[a_λ b]_{λ+μ} c] + [b_μ [a_λ c]]
//!
//! # Module Structure
//!
//! - `lie_conformal_algebra`: Base traits and infrastructure
//! - `lie_conformal_algebra_element`: Element representation
//! - `graded_lie_conformal_algebra`: Graded structure support
//! - `lie_conformal_algebra_with_basis`: Basis functionality
//! - `abelian_lie_conformal_algebra`: Abelian (trivial λ-bracket) algebras
//! - `virasoro_lie_conformal_algebra`: Virasoro algebra
//! - `weyl_lie_conformal_algebra`: Weyl algebra
//!
//! Corresponds to sage.algebras.lie_conformal_algebras
//!
//! # References
//!
//! - Kac, V. "Vertex Algebras for Beginners" (1998)
//! - Bakalov, B. and Kac, V. "Field algebras" (2003)

pub mod lie_conformal_algebra;
pub mod lie_conformal_algebra_element;
pub mod graded_lie_conformal_algebra;
pub mod lie_conformal_algebra_with_basis;
pub mod freely_generated_lie_conformal_algebra;
pub mod finitely_freely_generated_lca;
pub mod abelian_lie_conformal_algebra;
pub mod virasoro_lie_conformal_algebra;
pub mod weyl_lie_conformal_algebra;
pub mod free_bosons_lie_conformal_algebra;
pub mod free_fermions_lie_conformal_algebra;
pub mod n2_lie_conformal_algebra;
pub mod neveu_schwarz_lie_conformal_algebra;
pub mod affine_lie_conformal_algebra;
pub mod bosonic_ghosts_lie_conformal_algebra;
pub mod fermionic_ghosts_lie_conformal_algebra;

pub use lie_conformal_algebra::{
    LieConformalAlgebra, LambdaBracket, Derivation,
};
pub use lie_conformal_algebra_element::{
    LieConformalAlgebraElement, LCAElementWrapper,
};
pub use graded_lie_conformal_algebra::{
    GradedLieConformalAlgebra, Degree, Weight,
};
pub use lie_conformal_algebra_with_basis::{
    LieConformalAlgebraWithBasis, BasisElement, StructureCoefficients,
};
pub use abelian_lie_conformal_algebra::{
    AbelianLieConformalAlgebra, AbelianLCAElement,
};
pub use virasoro_lie_conformal_algebra::{
    VirasoroLieConformalAlgebra, VirasoroLCAElement, VirasoroGenerator,
};
pub use weyl_lie_conformal_algebra::{
    WeylLieConformalAlgebra, WeylLCAElement,
};
pub use affine_lie_conformal_algebra::{
    AffineLieConformalAlgebra, AffineLCAElement,
};
pub use free_bosons_lie_conformal_algebra::{
    FreeBosonsLieConformalAlgebra, FreeBosonsLCAElement,
};
pub use free_fermions_lie_conformal_algebra::{
    FreeFermionsLieConformalAlgebra, FreeFermionsLCAElement,
};
pub use bosonic_ghosts_lie_conformal_algebra::{
    BosonicGhostsLieConformalAlgebra, BosonicGhostsLCAElement,
};
pub use fermionic_ghosts_lie_conformal_algebra::{
    FermionicGhostsLieConformalAlgebra, FermionicGhostsLCAElement,
};
pub use finitely_freely_generated_lca::{
    FinitelyFreelyGeneratedLCA, FinitelyFreelyGeneratedElement,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_structure() {
        // Basic smoke test
        assert!(true);
    }
}
