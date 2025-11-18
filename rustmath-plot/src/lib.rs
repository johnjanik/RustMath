//! Main plotting library for RustMath
//!
//! This crate provides the core plotting functionality for RustMath, including:
//! - Graphics container for combining multiple primitives
//! - Multi-graphics for arranging plots in grids
//! - Backend utilities for implementing renderers
//!
//! Based on SageMath's sage.plot.graphics and sage.plot.multigraphics modules.
//!
//! # Phase 2 Implementation
//!
//! This is Phase 2 of the plotting system implementation:
//! - Graphics container system
//! - Multi-graphics (GraphicsArray) for subplot layouts
//! - Backend utilities and helpers
//!
//! # Examples
//!
//! ```
//! use rustmath_plot::Graphics;
//!
//! // Create a new graphics container
//! let mut g = Graphics::new();
//! g.set_title("My Plot");
//! g.set_labels("x-axis", "y-axis");
//! ```
//!
//! ```
//! use rustmath_plot::MultiGraphics;
//!
//! // Create a 2x2 grid of plots
//! let mut mg = MultiGraphics::new(2, 2);
//! ```

mod backend;
mod graphics;
mod multigraphics;

// Re-export core types from rustmath-plot-core
pub use rustmath_plot_core::{
    AxesOptions, BoundingBox, GraphicPrimitive, GraphicsOptions,
    LineStyle, MarkerStyle, PlotError, PlotOptions, Point2D, Point3D, Renderable, RenderBackend,
    RenderFormat, Result, TextOptions, Transform2D, Vector2D, Vector3D,
};

// Re-export color types
pub use rustmath_colors::{Color, ColorSpace, Colormap};

// Re-export main types from this crate
pub use graphics::Graphics;
pub use multigraphics::{
    Alignment, GraphicsArray, GridLayout, MultiGraphics, MultiGraphicsOptions,
};

// Re-export backend utilities
pub use backend::{AdaptiveSampler, ColorInterpolator, ViewportTransform};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graphics_creation() {
        let g = Graphics::new();
        assert!(g.is_empty());
    }

    #[test]
    fn test_multigraphics_creation() {
        let mg = MultiGraphics::new(2, 3);
        assert_eq!(mg.layout().rows, 2);
        assert_eq!(mg.layout().cols, 3);
    }

    #[test]
    fn test_graphics_with_options() {
        let mut opts = GraphicsOptions::default();
        opts.title = Some("Test".to_string());

        let g = Graphics::with_options(opts);
        assert_eq!(g.options().title, Some("Test".to_string()));
    }
}
