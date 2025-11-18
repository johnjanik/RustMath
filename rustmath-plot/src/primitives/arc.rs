//! Arc primitive for plotting
//!
//! Based on SageMath's sage.plot.arc module

use rustmath_colors::Color;
use rustmath_plot_core::{
    BoundingBox, GraphicPrimitive, PlotOptions, Point2D, Renderable, RenderBackend, Result,
};

/// An arc (portion of a circle)
///
/// Based on SageMath's Arc class from sage.plot.arc
pub struct Arc {
    /// Center point of the arc
    center: Point2D,

    /// Radius of the arc
    radius: f64,

    /// Start angle in radians
    start_angle: f64,

    /// End angle in radians
    end_angle: f64,

    /// Plot options (color, thickness, etc.)
    options: PlotOptions,
}

impl Arc {
    /// Create a new Arc primitive
    ///
    /// # Arguments
    /// * `center` - The center point of the arc
    /// * `radius` - The radius of the arc
    /// * `start_angle` - The starting angle in radians
    /// * `end_angle` - The ending angle in radians
    /// * `options` - Plotting options
    ///
    /// # Examples
    /// ```ignore
    /// use std::f64::consts::PI;
    /// let arc = Arc::new((0.0, 0.0), 1.0, 0.0, PI/2.0, PlotOptions::default());
    /// ```
    pub fn new(
        center: impl Into<Point2D>,
        radius: f64,
        start_angle: f64,
        end_angle: f64,
        options: PlotOptions,
    ) -> Self {
        assert!(radius >= 0.0, "radius must be non-negative");
        Self {
            center: center.into(),
            radius,
            start_angle,
            end_angle,
            options,
        }
    }

    /// Create an arc from start and end angles in degrees
    pub fn from_degrees(
        center: impl Into<Point2D>,
        radius: f64,
        start_degrees: f64,
        end_degrees: f64,
        options: PlotOptions,
    ) -> Self {
        Self::new(
            center,
            radius,
            start_degrees.to_radians(),
            end_degrees.to_radians(),
            options,
        )
    }

    /// Get the center point
    pub fn center(&self) -> Point2D {
        self.center
    }

    /// Get the radius
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Get the start angle in radians
    pub fn start_angle(&self) -> f64 {
        self.start_angle
    }

    /// Get the end angle in radians
    pub fn end_angle(&self) -> f64 {
        self.end_angle
    }

    /// Get the start angle in degrees
    pub fn start_angle_degrees(&self) -> f64 {
        self.start_angle.to_degrees()
    }

    /// Get the end angle in degrees
    pub fn end_angle_degrees(&self) -> f64 {
        self.end_angle.to_degrees()
    }

    /// Get the angular span in radians
    pub fn angular_span(&self) -> f64 {
        self.end_angle - self.start_angle
    }

    /// Get the arc length
    pub fn arc_length(&self) -> f64 {
        self.radius * self.angular_span().abs()
    }

    /// Get the start point on the arc
    pub fn start_point(&self) -> Point2D {
        Point2D::new(
            self.center.x + self.radius * self.start_angle.cos(),
            self.center.y + self.radius * self.start_angle.sin(),
        )
    }

    /// Get the end point on the arc
    pub fn end_point(&self) -> Point2D {
        Point2D::new(
            self.center.x + self.radius * self.end_angle.cos(),
            self.center.y + self.radius * self.end_angle.sin(),
        )
    }
}

impl Renderable for Arc {
    fn bounding_box(&self) -> BoundingBox {
        // For simplicity, we'll use the full circle's bounding box
        // A more accurate implementation would calculate the exact bounds of the arc
        BoundingBox::new(
            self.center.x - self.radius,
            self.center.x + self.radius,
            self.center.y - self.radius,
            self.center.y + self.radius,
        )
    }

    fn render(&self, backend: &mut dyn RenderBackend) -> Result<()> {
        // Convert radians to degrees for backend
        backend.draw_arc(
            self.center,
            self.radius,
            self.start_angle.to_degrees(),
            self.end_angle.to_degrees(),
            &self.options,
        )
    }
}

impl GraphicPrimitive for Arc {
    fn options(&self) -> &PlotOptions {
        &self.options
    }

    fn options_mut(&mut self) -> &mut PlotOptions {
        &mut self.options
    }

    fn set_options(&mut self, options: PlotOptions) {
        self.options = options;
    }
}

/// Factory function to create an Arc primitive
///
/// # Arguments
/// * `center` - The center point of the arc
/// * `radius` - The radius of the arc
/// * `start_angle` - The starting angle in radians
/// * `end_angle` - The ending angle in radians
/// * `options` - Optional plot options
///
/// # Examples
/// ```ignore
/// use rustmath_plot::primitives::arc;
/// use std::f64::consts::PI;
///
/// let a1 = arc((0.0, 0.0), 1.0, 0.0, PI/2.0, None);
/// let a2 = arc((1.0, 1.0), 2.0, 0.0, PI, Some(PlotOptions::default().with_color(Color::red_color())));
/// ```
pub fn arc(
    center: impl Into<Point2D>,
    radius: f64,
    start_angle: f64,
    end_angle: f64,
    options: Option<PlotOptions>,
) -> Box<Arc> {
    Box::new(Arc::new(
        center,
        radius,
        start_angle,
        end_angle,
        options.unwrap_or_default(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_arc_creation() {
        let a = Arc::new((0.0, 0.0), 1.0, 0.0, PI / 2.0, PlotOptions::default());
        assert_eq!(a.center(), Point2D::new(0.0, 0.0));
        assert_eq!(a.radius(), 1.0);
        assert_eq!(a.start_angle(), 0.0);
        assert_eq!(a.end_angle(), PI / 2.0);
    }

    #[test]
    fn test_arc_from_degrees() {
        let a = Arc::from_degrees((0.0, 0.0), 1.0, 0.0, 90.0, PlotOptions::default());
        assert!((a.start_angle() - 0.0).abs() < 1e-10);
        assert!((a.end_angle() - PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_arc_angular_span() {
        let a = Arc::new((0.0, 0.0), 1.0, 0.0, PI / 2.0, PlotOptions::default());
        let span = a.angular_span();
        assert!((span - PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_arc_length() {
        let a = Arc::new((0.0, 0.0), 1.0, 0.0, PI / 2.0, PlotOptions::default());
        let length = a.arc_length();
        assert!((length - PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_arc_endpoints() {
        let a = Arc::new((0.0, 0.0), 1.0, 0.0, PI / 2.0, PlotOptions::default());
        let start = a.start_point();
        let end = a.end_point();

        assert!((start.x - 1.0).abs() < 1e-10);
        assert!((start.y - 0.0).abs() < 1e-10);
        assert!((end.x - 0.0).abs() < 1e-10);
        assert!((end.y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_arc_degrees() {
        let a = Arc::from_degrees((0.0, 0.0), 1.0, 0.0, 90.0, PlotOptions::default());
        assert!((a.start_angle_degrees() - 0.0).abs() < 1e-10);
        assert!((a.end_angle_degrees() - 90.0).abs() < 1e-10);
    }

    #[test]
    fn test_arc_bounding_box() {
        let a = Arc::new((1.0, 2.0), 3.0, 0.0, PI, PlotOptions::default());
        let bbox = a.bounding_box();
        assert_eq!(bbox.xmin, -2.0);
        assert_eq!(bbox.xmax, 4.0);
        assert_eq!(bbox.ymin, -1.0);
        assert_eq!(bbox.ymax, 5.0);
    }

    #[test]
    #[should_panic(expected = "radius must be non-negative")]
    fn test_arc_negative_radius() {
        Arc::new((0.0, 0.0), -1.0, 0.0, PI, PlotOptions::default());
    }

    #[test]
    fn test_arc_factory() {
        let a = arc((0.0, 0.0), 1.0, 0.0, PI, None);
        assert_eq!(a.radius(), 1.0);
    }
}
