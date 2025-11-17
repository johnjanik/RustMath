//! RustMath Manifolds - Differential geometry and manifold theory
//!
//! This crate provides structures and operations for working with mathematical manifolds,
//! including topological manifolds, differentiable manifolds, charts, scalar fields,
//! and various geometric objects.
//!
//! # Overview
//!
//! A manifold is a topological space that locally resembles Euclidean space. This crate
//! implements the foundational structures needed for differential geometry:
//!
//! - **ManifoldSubset**: Base type representing subsets of manifolds
//! - **TopologicalManifold**: Manifolds with a topological structure
//! - **DifferentiableManifold**: Smooth manifolds with differentiable structure
//! - **Chart**: Local coordinate systems on manifolds
//! - **Point**: Points on manifolds
//! - **ScalarField**: Smooth scalar-valued functions on manifolds
//!
//! # Examples
//!
//! ```
//! use rustmath_manifolds::{TopologicalManifold, RealLine};
//!
//! // Create a 1-dimensional real line manifold
//! let real_line = RealLine::new();
//! ```

pub mod subset;
pub mod manifold;
pub mod chart;
pub mod point;
pub mod scalar_field;
pub mod differentiable;
pub mod examples;
pub mod errors;
pub mod utilities;
pub mod vector_bundle;
pub mod vector_bundle_fiber;
pub mod transition;
pub mod traits;
pub mod vector_field;
pub mod tensor_field;
pub mod diff_form;
pub mod tangent_space;
pub mod modules;
pub mod riemannian;
pub mod lie_group;
pub mod lie_algebra;
pub mod fiber_bundles;
pub mod symplectic;
pub mod integration;
pub mod topology;
pub mod maps;
pub mod symmetries;
pub mod catalog;

// Phase 5: Advanced geometric structures
pub mod complex_manifold;
pub mod almost_complex;
pub mod kahler;
pub mod finsler;
pub mod subriemannian;
pub mod contact;
pub mod spin;
pub mod dirac;

pub use subset::ManifoldSubset;
pub use manifold::TopologicalManifold;
pub use chart::{Chart, CoordinateFunction};
pub use point::ManifoldPoint;
pub use scalar_field::ScalarFieldEnhanced as ScalarField;
pub use differentiable::DifferentiableManifold;
pub use examples::{RealLine, EuclideanSpace, Circle, Sphere2, Torus2};
pub use errors::{ManifoldError, Result};
pub use utilities::{
    SimplificationChain, simplify_abs_trig, simplify_sqrt_real,
    simplify_chain_real, simplify_chain_generic, exterior_derivative,
    set_axes_labels, xder,
};
pub use vector_bundle::{TopologicalVectorBundle, TangentBundle, CotangentBundle};
pub use vector_bundle_fiber::{VectorBundleFiber, VectorBundleFiberElement};
pub use transition::TransitionFunction;
pub use vector_field::VectorField;
pub use tensor_field::{TensorField, MultiIndex};
pub use diff_form::DiffForm;
pub use tangent_space::{TangentVector, TangentSpace, Covector, CotangentSpace};
pub use modules::{
    DiffScalarFieldAlgebra, VectorFieldModule, VectorFieldFreeModule,
    TensorFieldModule, DiffFormModule,
};
pub use riemannian::{
    RiemannianMetric, AffineConnection, LeviCivitaConnection,
    CovariantDerivative, RiemannTensor, RicciTensor, ScalarCurvature,
    Geodesic, ParallelTransport,
};
pub use lie_group::{
    LieGroup, LeftInvariantVectorField, RightInvariantVectorField, MaurerCartanForm,
};
pub use lie_algebra::{LieAlgebra, ExponentialMap};
pub use fiber_bundles::{
    FiberBundle, Fiber, PrincipalBundle, AssociatedBundle,
    ConnectionForm, CurvatureForm, Trivialization,
};
pub use symplectic::{
    SymplecticManifold, SymplecticForm, HamiltonianVectorField, PoissonBracket,
};
pub use integration::{
    IntegrationOnManifolds, VolumeForm, OrientedManifold, Orientation, StokesTheorem,
};
pub use topology::{
    DeRhamCohomology, BettiNumber, EulerCharacteristic,
    ChernClass, PontryaginClass, EulerClass,
};
pub use maps::{
    SmoothMap, PushForward, PullBack, Immersion, Submersion, Embedding, Diffeomorphism,
};
pub use symmetries::{
    KillingVectorField, ConformallKillingVectorField, IsometryGroup,
};
pub use catalog::{
    Minkowski, Schwarzschild, Kerr,
    RealProjectiveSpace, ComplexProjectiveSpace, Grassmannian,
    SpecialOrthogonalGroup, SpecialUnitaryGroup,
};

// Phase 5: Advanced geometric structures
pub use complex_manifold::{ComplexManifold, ComplexChart, HolomorphicFunction};
pub use almost_complex::{AlmostComplexStructure, AlmostComplexManifold};
pub use kahler::{KahlerManifold, HermitianMetric};
pub use finsler::{FinslerManifold, FinslerFunction, CartanTensor};
pub use subriemannian::{SubRiemannianManifold, Distribution};
pub use contact::{ContactManifold, ContactForm, ContactHamiltonian};
pub use spin::{SpinStructure, SpinorBundle, SpinorField, SpinGroup, CliffordMultiplication};
pub use dirac::{
    DiracOperator, SpinorConnection, DiracSquared, TwistedDiracOperator,
    AtiyahSingerIndexTheorem, DiracHeatKernel,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real_line_creation() {
        let real_line = RealLine::new();
        assert_eq!(real_line.dimension(), 1);
    }

    #[test]
    fn test_euclidean_space_creation() {
        let euclidean_2d = EuclideanSpace::new(2);
        assert_eq!(euclidean_2d.dimension(), 2);

        let euclidean_3d = EuclideanSpace::new(3);
        assert_eq!(euclidean_3d.dimension(), 3);
    }
}
