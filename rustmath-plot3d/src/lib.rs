//! 3D plotting infrastructure for RustMath
//!
//! This crate provides the core infrastructure for 3D plotting in RustMath,
//! including base types, transformations, camera system, and lighting.

pub mod base;
pub mod transform;
pub mod camera;
pub mod shapes;
pub mod platonic;
pub mod plots;

// Re-export key types
pub use base::{Graphics3d, Graphics3dPrimitive, IndexFaceSet, BoundingBox3D, Point3D, Vector3D, Graphics3dOptions};
pub use transform::{Transform3D, TransformGroup};
pub use camera::{Camera, Light};
pub use shapes::{Sphere, Box, Cylinder, Cone, Torus};
pub use platonic::{tetrahedron, cube, octahedron, dodecahedron, icosahedron, colored_platonic_solid};
pub use plots::{plot3d, parametric_plot3d, list_plot3d, revolution_plot3d};

// Type aliases for common naming conventions
/// Alias for BoundingBox3D
pub type BoundingBox = BoundingBox3D;

/// Alias for Graphics3dOptions (rendering parameters)
pub type RenderParams = Graphics3dOptions;

use thiserror::Error;

/// Error types for 3D plotting operations
#[derive(Debug, Error)]
pub enum Plot3DError {
    #[error("Invalid mesh: {0}")]
    InvalidMesh(String),

    #[error("Rendering error: {0}")]
    RenderError(String),

    #[error("Transform error: {0}")]
    TransformError(String),

    #[error("Camera error: {0}")]
    CameraError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Plot3DError>;
