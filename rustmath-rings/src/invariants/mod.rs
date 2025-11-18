//! # Invariants Module
//!
//! This module provides comprehensive functionality for classical invariant theory,
//! focusing on algebraic forms and their invariants.
//!
//! ## Submodules
//!
//! - `invariant_theory`: Core invariant theory functionality for algebraic forms
//! - `reconstruction`: Reconstruction of forms from their invariants

pub mod invariant_theory;
pub mod reconstruction;

// Re-export commonly used types
pub use invariant_theory::{
    AlgebraicForm, BinaryQuartic, BinaryQuintic, FormsBase, InvariantTheoryFactory,
    QuadraticForm, SeveralAlgebraicForms, TernaryCubic, TernaryQuadratic, TwoAlgebraicForms,
    TwoQuaternaryQuadratics, TwoTernaryQuadratics, transvectant, INVARIANT_THEORY,
};

pub use reconstruction::{
    binary_cubic_coefficients_from_invariants, binary_quadratic_coefficients_from_invariants,
    binary_quintic_coefficients_from_invariants, ReconstructionError, Scaling,
};
