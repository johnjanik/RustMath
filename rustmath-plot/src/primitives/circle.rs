//! Circle primitive for plotting
//!
//! Based on SageMath's sage.plot.circle module

use rustmath_colors::Color;
use rustmath_plot_core::{
    BoundingBox, GraphicPrimitive, PlotOptions, Point2D, Renderable, RenderBackend, Result,
};

/// A circle (unfilled)
///
/// Based on SageMath's Circle class from sage.plot.circle
pub struct Circle {
    /// Center point of the circle
    center: Point2D,

    /// Radius of the circle
    radius: f64,

    /// Plot options (color, thickness, line style, etc.)
    options: PlotOptions,
}

impl Circle {
    /// Create a new Circle primitive
    ///
    /// # Arguments
    /// * `center` - The center point of the circle
    /// * `radius` - The radius of the circle
    /// * `options` - Plotting options
    ///
    /// # Examples
    /// ```ignore
    /// let circle = Circle::new((0.0, 0.0), 1.0, PlotOptions::default());
    /// ```
    pub fn new(center: impl Into<Point2D>, radius: f64, options: PlotOptions) -> Self {
        assert!(radius >= 0.0, "radius must be non-negative");
        Self {
            center: center.into(),
            radius,
            options,
        }
    }

    /// Get the center point
    pub fn center(&self) -> Point2D {
        self.center
    }

    /// Get the radius
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Get the diameter
    pub fn diameter(&self) -> f64 {
        self.radius * 2.0
    }

    /// Get the circumference
    pub fn circumference(&self) -> f64 {
        2.0 * std::f64::consts::PI * self.radius
    }

    /// Get the area
    pub fn area(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }

    /// Check if a point is inside the circle
    pub fn contains(&self, point: &Point2D) -> bool {
        self.center.distance_to(point) <= self.radius
    }
}

impl Renderable for Circle {
    fn bounding_box(&self) -> BoundingBox {
        BoundingBox::new(
            self.center.x - self.radius,
            self.center.x + self.radius,
            self.center.y - self.radius,
            self.center.y + self.radius,
        )
    }

    fn render(&self, backend: &mut dyn RenderBackend) -> Result<()> {
        backend.draw_circle(self.center, self.radius, &self.options)
    }
}

impl GraphicPrimitive for Circle {
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

/// Factory function to create a Circle primitive
///
/// # Arguments
/// * `center` - The center point of the circle
/// * `radius` - The radius of the circle
/// * `options` - Optional plot options
///
/// # Examples
/// ```ignore
/// use rustmath_plot::primitives::circle;
///
/// let c1 = circle((0.0, 0.0), 1.0, None);
/// let c2 = circle((1.0, 1.0), 2.0, Some(PlotOptions::default().with_color(Color::red_color())));
/// ```
pub fn circle(center: impl Into<Point2D>, radius: f64, options: Option<PlotOptions>) -> Box<Circle> {
    Box::new(Circle::new(center, radius, options.unwrap_or_default()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circle_creation() {
        let c = Circle::new((0.0, 0.0), 1.0, PlotOptions::default());
        assert_eq!(c.center(), Point2D::new(0.0, 0.0));
        assert_eq!(c.radius(), 1.0);
        assert_eq!(c.diameter(), 2.0);
    }

    #[test]
    fn test_circle_geometry() {
        let c = Circle::new((0.0, 0.0), 1.0, PlotOptions::default());
        let circumference = c.circumference();
        let area = c.area();

        // Check approximate values
        assert!((circumference - 2.0 * std::f64::consts::PI).abs() < 1e-10);
        assert!((area - std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_circle_bounding_box() {
        let c = Circle::new((1.0, 2.0), 3.0, PlotOptions::default());
        let bbox = c.bounding_box();
        assert_eq!(bbox.xmin, -2.0);
        assert_eq!(bbox.xmax, 4.0);
        assert_eq!(bbox.ymin, -1.0);
        assert_eq!(bbox.ymax, 5.0);
    }

    #[test]
    fn test_circle_contains() {
        let c = Circle::new((0.0, 0.0), 1.0, PlotOptions::default());
        assert!(c.contains(&Point2D::new(0.0, 0.0))); // Center
        assert!(c.contains(&Point2D::new(0.5, 0.5))); // Inside
        assert!(c.contains(&Point2D::new(1.0, 0.0))); // On circumference
        assert!(!c.contains(&Point2D::new(2.0, 0.0))); // Outside
    }

    #[test]
    #[should_panic(expected = "radius must be non-negative")]
    fn test_circle_negative_radius() {
        Circle::new((0.0, 0.0), -1.0, PlotOptions::default());
    }

    #[test]
    fn test_circle_factory() {
        let c = circle((0.0, 0.0), 1.0, None);
        assert_eq!(c.radius(), 1.0);
    }
}
