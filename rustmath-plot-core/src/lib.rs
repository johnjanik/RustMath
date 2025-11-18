//! Core plotting infrastructure for RustMath
//!
//! This crate provides the foundational traits and types for the RustMath plotting system.
//! It defines the trait system that all graphics primitives implement, along with common
//! types like bounding boxes, plot options, and rendering interfaces.
//!
//! Based on SageMath's sage.plot.primitive and sage.plot.graphics modules.

mod bbox;
mod options;
mod traits;
mod types;

pub use bbox::BoundingBox;
pub use options::{AxesOptions, GraphicsOptions, LineStyle, MarkerStyle, PlotOptions, TextOptions};
pub use traits::{GraphicPrimitive, Renderable, RenderBackend};
pub use types::{Point2D, Point3D, RenderFormat, Vector2D, Vector3D};

/// Errors that can occur during plotting operations
#[derive(Debug, thiserror::Error)]
pub enum PlotError {
    #[error("Invalid bounding box: {0}")]
    InvalidBoundingBox(String),

    #[error("Invalid option value: {0}")]
    InvalidOption(String),

    #[error("Rendering error: {0}")]
    RenderError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, PlotError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point2d_creation() {
        let p = Point2D::new(1.0, 2.0);
        assert_eq!(p.x, 1.0);
        assert_eq!(p.y, 2.0);
    }

    #[test]
    fn test_bounding_box_creation() {
        let bbox = BoundingBox::new(0.0, 10.0, 0.0, 10.0);
        assert_eq!(bbox.width(), 10.0);
        assert_eq!(bbox.height(), 10.0);
    }
}
