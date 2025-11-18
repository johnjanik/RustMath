//! Main plotting library for RustMath
//!
//! This crate provides the core plotting functionality for RustMath, including:
//! - Graphics container for combining multiple primitives
//! - Multi-graphics for arranging plots in grids
//! - Basic 2D primitives (point, line, circle, polygon, etc.)
//! - Advanced 2D plotting functions (plot, contour, histogram, etc.)
//! - Backend utilities for implementing renderers
//!
//! Based on SageMath's sage.plot module.
//!
//! # Implementation Phases
//!
//! - **Phase 1** (Complete): Core infrastructure, color system, plot traits
//! - **Phase 2** (Complete): Graphics containers and layout system
//! - **Phase 3** (Complete): Basic 2D primitives
//! - **Phase 4** (Complete): Advanced 2D plotting functions
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
//! ```ignore
//! use rustmath_plot::{Graphics, primitives::*};
//!
//! // Create a plot with primitives
//! let mut g = Graphics::new();
//! g.add(point(vec![(0.0, 0.0), (1.0, 1.0)], None));
//! g.add(line(vec![(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)], None));
//! g.add(circle((0.0, 0.0), 1.0, None));
//! ```
//!
//! ```ignore
//! use rustmath_plot::plots::*;
//!
//! // Plot a mathematical function
//! let g = plot(|x| x.sin(), 0.0, 2.0 * std::f64::consts::PI, None);
//!
//! // Create a scatter plot
//! let data = vec![(0.0, 1.0), (1.0, 2.0), (2.0, 1.5)];
//! let g = scatter_plot(data, None, None, None);
//!
//! // Create a histogram
//! let data = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0];
//! let g = histogram(data, Some(5), None, None, None);
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
pub mod primitives;
pub mod plots;

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
