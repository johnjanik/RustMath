//! Riemannian Manifolds
//!
//! This module provides tools for working with Riemannian manifolds,
//! particularly parametrized surfaces in 3D space.

pub mod parametrized_surface3d;
pub mod surface3d_generators;

pub use parametrized_surface3d::ParametrizedSurface3D;
pub use surface3d_generators::SurfaceGenerators;
