//! Core traits for the plotting system
//!
//! Based on SageMath's sage.plot.primitive module

use crate::{BoundingBox, PlotOptions, Point2D, Result, TextOptions};
use rustmath_colors::Color;

/// Trait for objects that can be rendered
///
/// This is the base trait that all graphics objects must implement.
/// Based on SageMath's GraphicPrimitive base class.
pub trait Renderable {
    /// Get the bounding box of this renderable object
    ///
    /// The bounding box should encompass all visible parts of the object.
    fn bounding_box(&self) -> BoundingBox;

    /// Render this object using the provided backend
    ///
    /// # Arguments
    /// * `backend` - The rendering backend to use
    fn render(&self, backend: &mut dyn RenderBackend) -> Result<()>;
}

/// Trait for graphics primitives
///
/// Graphics primitives are renderable objects with additional plotting options.
/// Based on SageMath's GraphicPrimitive class.
pub trait GraphicPrimitive: Renderable {
    /// Get the plot options for this primitive
    fn options(&self) -> &PlotOptions;

    /// Get a mutable reference to the plot options
    fn options_mut(&mut self) -> &mut PlotOptions;

    /// Set the plot options for this primitive
    fn set_options(&mut self, options: PlotOptions);

    /// Set the color
    fn set_color(&mut self, color: Color) {
        self.options_mut().color = color;
    }

    /// Set the thickness
    fn set_thickness(&mut self, thickness: f64) {
        self.options_mut().thickness = thickness;
    }

    /// Set the alpha transparency
    fn set_alpha(&mut self, alpha: f64) {
        self.options_mut().alpha = alpha.clamp(0.0, 1.0);
    }

    /// Set the z-order
    fn set_zorder(&mut self, zorder: i32) {
        self.options_mut().zorder = zorder;
    }

    /// Set the label for the legend
    fn set_label(&mut self, label: impl Into<String>) {
        self.options_mut().label = Some(label.into());
    }
}

/// Trait for rendering backends
///
/// This trait defines the interface that all rendering backends must implement.
/// Different backends can render to different formats (SVG, PNG, etc.)
pub trait RenderBackend {
    /// Draw a line connecting a series of points
    ///
    /// # Arguments
    /// * `points` - The points to connect
    /// * `options` - Styling options for the line
    fn draw_line(&mut self, points: &[Point2D], options: &PlotOptions) -> Result<()>;

    /// Draw a polygon (closed shape)
    ///
    /// # Arguments
    /// * `points` - The vertices of the polygon
    /// * `options` - Styling options (including fill)
    fn draw_polygon(&mut self, points: &[Point2D], options: &PlotOptions) -> Result<()>;

    /// Draw a circle
    ///
    /// # Arguments
    /// * `center` - Center point of the circle
    /// * `radius` - Radius of the circle
    /// * `options` - Styling options (including fill)
    fn draw_circle(&mut self, center: Point2D, radius: f64, options: &PlotOptions) -> Result<()>;

    /// Draw an ellipse
    ///
    /// # Arguments
    /// * `center` - Center point of the ellipse
    /// * `rx` - Horizontal radius
    /// * `ry` - Vertical radius
    /// * `rotation` - Rotation angle in degrees
    /// * `options` - Styling options (including fill)
    fn draw_ellipse(
        &mut self,
        center: Point2D,
        rx: f64,
        ry: f64,
        rotation: f64,
        options: &PlotOptions,
    ) -> Result<()>;

    /// Draw a rectangular region
    ///
    /// # Arguments
    /// * `min` - Minimum corner (bottom-left)
    /// * `max` - Maximum corner (top-right)
    /// * `options` - Styling options (including fill)
    fn draw_rectangle(&mut self, min: Point2D, max: Point2D, options: &PlotOptions)
        -> Result<()>;

    /// Draw an arc (portion of a circle)
    ///
    /// # Arguments
    /// * `center` - Center point of the arc
    /// * `radius` - Radius of the arc
    /// * `start_angle` - Starting angle in degrees
    /// * `end_angle` - Ending angle in degrees
    /// * `options` - Styling options
    fn draw_arc(
        &mut self,
        center: Point2D,
        radius: f64,
        start_angle: f64,
        end_angle: f64,
        options: &PlotOptions,
    ) -> Result<()>;

    /// Draw a cubic Bezier curve
    ///
    /// # Arguments
    /// * `p0` - Start point
    /// * `p1` - First control point
    /// * `p2` - Second control point
    /// * `p3` - End point
    /// * `options` - Styling options
    fn draw_bezier(
        &mut self,
        p0: Point2D,
        p1: Point2D,
        p2: Point2D,
        p3: Point2D,
        options: &PlotOptions,
    ) -> Result<()>;

    /// Draw text at a position
    ///
    /// # Arguments
    /// * `position` - Position to draw the text
    /// * `text` - The text to draw
    /// * `options` - Text styling options
    fn draw_text(&mut self, position: Point2D, text: &str, options: &TextOptions) -> Result<()>;

    /// Draw individual points
    ///
    /// # Arguments
    /// * `points` - The points to draw
    /// * `options` - Styling options (marker style, size, color)
    fn draw_points(&mut self, points: &[Point2D], options: &PlotOptions) -> Result<()>;

    /// Draw an arrow from one point to another
    ///
    /// # Arguments
    /// * `start` - Start point of the arrow
    /// * `end` - End point of the arrow (where the arrowhead is)
    /// * `options` - Styling options
    fn draw_arrow(&mut self, start: Point2D, end: Point2D, options: &PlotOptions) -> Result<()>;

    /// Begin a new path (for complex shapes)
    fn begin_path(&mut self) -> Result<()>;

    /// Move to a point without drawing
    fn move_to(&mut self, point: Point2D) -> Result<()>;

    /// Draw a line to a point
    fn line_to(&mut self, point: Point2D) -> Result<()>;

    /// Close the current path
    fn close_path(&mut self) -> Result<()>;

    /// Stroke the current path with the given options
    fn stroke(&mut self, options: &PlotOptions) -> Result<()>;

    /// Fill the current path with the given options
    fn fill(&mut self, options: &PlotOptions) -> Result<()>;

    /// Get the current view/coordinate transformation
    fn get_transform(&self) -> Transform2D;

    /// Set the view/coordinate transformation
    fn set_transform(&mut self, transform: Transform2D);

    /// Finalize the rendering and get the output
    ///
    /// This is called after all drawing operations are complete.
    fn finalize(&mut self) -> Result<Vec<u8>>;
}

/// A 2D affine transformation matrix
///
/// Represents transformations like translation, rotation, and scaling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform2D {
    /// Matrix elements [a, b, c, d, e, f] representing:
    /// | a  c  e |
    /// | b  d  f |
    /// | 0  0  1 |
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub e: f64,
    pub f: f64,
}

impl Transform2D {
    /// Identity transformation (no change)
    pub fn identity() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            e: 0.0,
            f: 0.0,
        }
    }

    /// Create a translation transformation
    pub fn translate(dx: f64, dy: f64) -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            e: dx,
            f: dy,
        }
    }

    /// Create a scaling transformation
    pub fn scale(sx: f64, sy: f64) -> Self {
        Self {
            a: sx,
            b: 0.0,
            c: 0.0,
            d: sy,
            e: 0.0,
            f: 0.0,
        }
    }

    /// Create a rotation transformation (angle in radians)
    pub fn rotate(angle: f64) -> Self {
        let cos = angle.cos();
        let sin = angle.sin();
        Self {
            a: cos,
            b: sin,
            c: -sin,
            d: cos,
            e: 0.0,
            f: 0.0,
        }
    }

    /// Compose this transformation with another
    pub fn compose(&self, other: &Transform2D) -> Self {
        Self {
            a: self.a * other.a + self.c * other.b,
            b: self.b * other.a + self.d * other.b,
            c: self.a * other.c + self.c * other.d,
            d: self.b * other.c + self.d * other.d,
            e: self.a * other.e + self.c * other.f + self.e,
            f: self.b * other.e + self.d * other.f + self.f,
        }
    }

    /// Transform a point
    pub fn transform_point(&self, point: Point2D) -> Point2D {
        Point2D::new(
            self.a * point.x + self.c * point.y + self.e,
            self.b * point.x + self.d * point.y + self.f,
        )
    }
}

impl Default for Transform2D {
    fn default() -> Self {
        Self::identity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_identity() {
        let t = Transform2D::identity();
        let p = Point2D::new(3.0, 4.0);
        let transformed = t.transform_point(p);
        assert_eq!(transformed, p);
    }

    #[test]
    fn test_transform_translate() {
        let t = Transform2D::translate(2.0, 3.0);
        let p = Point2D::new(1.0, 1.0);
        let transformed = t.transform_point(p);
        assert_eq!(transformed, Point2D::new(3.0, 4.0));
    }

    #[test]
    fn test_transform_scale() {
        let t = Transform2D::scale(2.0, 3.0);
        let p = Point2D::new(2.0, 2.0);
        let transformed = t.transform_point(p);
        assert_eq!(transformed, Point2D::new(4.0, 6.0));
    }

    #[test]
    fn test_transform_rotate() {
        use std::f64::consts::PI;
        let t = Transform2D::rotate(PI / 2.0); // 90 degrees
        let p = Point2D::new(1.0, 0.0);
        let transformed = t.transform_point(p);
        // Should be approximately (0, 1)
        assert!((transformed.x - 0.0).abs() < 1e-10);
        assert!((transformed.y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_transform_compose() {
        let t1 = Transform2D::translate(1.0, 2.0);
        let t2 = Transform2D::scale(2.0, 2.0);
        let composed = t1.compose(&t2);

        let p = Point2D::new(1.0, 1.0);
        let result = composed.transform_point(p);

        // First scale (1,1) to (2,2), then translate by (1,2) to (3,4)
        assert_eq!(result, Point2D::new(3.0, 4.0));
    }
}
