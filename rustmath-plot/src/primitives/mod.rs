//! Graphics primitives for 2D plotting
//!
//! This module provides the basic building blocks for creating plots.
//! Each primitive corresponds to a SageMath plotting primitive.
//!
//! # Phase 3 Primitives
//!
//! The following 8 primitives are implemented:
//! - **Point**: Individual points or collections of points
//! - **Line**: Connected line segments
//! - **Circle**: Unfilled circles
//! - **Disk**: Filled circles
//! - **Polygon**: Closed polygons (filled or unfilled)
//! - **Text**: Text labels
//! - **Arrow**: Directed arrows
//! - **Arc**: Circular arcs
//!
//! # Examples
//!
//! ```ignore
//! use rustmath_plot::primitives::*;
//! use rustmath_plot::{Graphics, PlotOptions};
//!
//! let mut g = Graphics::new();
//!
//! // Add a point
//! g.add(point(vec![(0.0, 0.0), (1.0, 1.0)], None));
//!
//! // Add a line
//! g.add(line(vec![(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)], None));
//!
//! // Add a circle
//! g.add(circle((0.0, 0.0), 1.0, None));
//! ```

mod arc;
mod arrow;
mod circle;
mod disk;
mod line;
mod point;
mod polygon;
mod text;

// Re-export all primitives
pub use arc::{arc, Arc};
pub use arrow::{arrow, Arrow};
pub use circle::{circle, Circle};
pub use disk::{disk, Disk};
pub use line::{line, Line};
pub use point::{point, Point};
pub use polygon::{polygon, Polygon};
pub use text::{text, Text};

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_plot_core::PlotOptions;

    #[test]
    fn test_all_primitives_compile() {
        // Test that all factory functions work
        let _p = point(vec![(0.0, 0.0)], None);
        let _l = line(vec![(0.0, 0.0), (1.0, 1.0)], None);
        let _c = circle((0.0, 0.0), 1.0, None);
        let _d = disk((0.0, 0.0), 1.0, None);
        let _poly = polygon(vec![(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)], None);
        let _t = text("Hello", (0.0, 0.0), None);
        let _a = arrow((0.0, 0.0), (1.0, 1.0), None);
        let _arc = arc((0.0, 0.0), 1.0, 0.0, std::f64::consts::PI, None);
    }

    #[test]
    fn test_primitives_are_graphic_primitives() {
        use rustmath_plot_core::GraphicPrimitive;

        // Test that all primitives implement GraphicPrimitive
        let p: Box<dyn GraphicPrimitive> = point(vec![(0.0, 0.0)], None);
        let _opts = p.options();

        let l: Box<dyn GraphicPrimitive> = line(vec![(0.0, 0.0), (1.0, 1.0)], None);
        let _opts = l.options();
    }
}
