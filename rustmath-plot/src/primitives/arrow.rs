//! Arrow primitive for plotting
//!
//! Based on SageMath's sage.plot.arrow module

use rustmath_plot_core::{
    BoundingBox, GraphicPrimitive, PlotOptions, Point2D, Renderable, RenderBackend, Result,
    Vector2D,
};

/// An arrow from one point to another
///
/// Based on SageMath's Arrow class from sage.plot.arrow
pub struct Arrow {
    /// Start point of the arrow
    start: Point2D,

    /// End point of the arrow (where the arrowhead is)
    end: Point2D,

    /// Plot options (color, thickness, etc.)
    options: PlotOptions,
}

impl Arrow {
    /// Create a new Arrow primitive
    ///
    /// # Arguments
    /// * `start` - The start point of the arrow
    /// * `end` - The end point of the arrow (where the arrowhead will be)
    /// * `options` - Plotting options
    ///
    /// # Examples
    /// ```ignore
    /// let arrow = Arrow::new((0.0, 0.0), (1.0, 1.0), PlotOptions::default());
    /// ```
    pub fn new(start: impl Into<Point2D>, end: impl Into<Point2D>, options: PlotOptions) -> Self {
        Self {
            start: start.into(),
            end: end.into(),
            options,
        }
    }

    /// Create an arrow from a start point and a direction vector
    pub fn from_vector(
        start: impl Into<Point2D>,
        direction: Vector2D,
        options: PlotOptions,
    ) -> Self {
        let start = start.into();
        let end = Point2D::new(start.x + direction.x, start.y + direction.y);
        Self {
            start,
            end,
            options,
        }
    }

    /// Get the start point
    pub fn start(&self) -> Point2D {
        self.start
    }

    /// Get the end point
    pub fn end(&self) -> Point2D {
        self.end
    }

    /// Get the direction vector
    pub fn direction(&self) -> Vector2D {
        Vector2D::new(self.end.x - self.start.x, self.end.y - self.start.y)
    }

    /// Get the length of the arrow
    pub fn length(&self) -> f64 {
        self.start.distance_to(&self.end)
    }

    /// Get the angle of the arrow in radians
    pub fn angle(&self) -> f64 {
        let dir = self.direction();
        dir.y.atan2(dir.x)
    }
}

impl Renderable for Arrow {
    fn bounding_box(&self) -> BoundingBox {
        BoundingBox::from_points(&[self.start, self.end]).unwrap_or_else(|_| BoundingBox::empty())
    }

    fn render(&self, backend: &mut dyn RenderBackend) -> Result<()> {
        backend.draw_arrow(self.start, self.end, &self.options)
    }
}

impl GraphicPrimitive for Arrow {
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

/// Factory function to create an Arrow primitive
///
/// # Arguments
/// * `start` - The start point of the arrow
/// * `end` - The end point of the arrow
/// * `options` - Optional plot options
///
/// # Examples
/// ```ignore
/// use rustmath_plot::primitives::arrow;
///
/// let a1 = arrow((0.0, 0.0), (1.0, 1.0), None);
/// let a2 = arrow((0.0, 0.0), (2.0, 0.0), Some(PlotOptions::default().with_color(Color::red_color())));
/// ```
pub fn arrow(
    start: impl Into<Point2D>,
    end: impl Into<Point2D>,
    options: Option<PlotOptions>,
) -> Box<Arrow> {
    Box::new(Arrow::new(start, end, options.unwrap_or_default()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arrow_creation() {
        let a = Arrow::new((0.0, 0.0), (1.0, 1.0), PlotOptions::default());
        assert_eq!(a.start(), Point2D::new(0.0, 0.0));
        assert_eq!(a.end(), Point2D::new(1.0, 1.0));
    }

    #[test]
    fn test_arrow_from_vector() {
        let dir = Vector2D::new(1.0, 1.0);
        let a = Arrow::from_vector((0.0, 0.0), dir, PlotOptions::default());
        assert_eq!(a.start(), Point2D::new(0.0, 0.0));
        assert_eq!(a.end(), Point2D::new(1.0, 1.0));
    }

    #[test]
    fn test_arrow_direction() {
        let a = Arrow::new((0.0, 0.0), (3.0, 4.0), PlotOptions::default());
        let dir = a.direction();
        assert_eq!(dir.x, 3.0);
        assert_eq!(dir.y, 4.0);
    }

    #[test]
    fn test_arrow_length() {
        let a = Arrow::new((0.0, 0.0), (3.0, 4.0), PlotOptions::default());
        assert_eq!(a.length(), 5.0); // 3-4-5 triangle
    }

    #[test]
    fn test_arrow_angle() {
        let a = Arrow::new((0.0, 0.0), (1.0, 0.0), PlotOptions::default());
        let angle = a.angle();
        assert!((angle - 0.0).abs() < 1e-10); // Pointing right is 0 radians

        let a2 = Arrow::new((0.0, 0.0), (0.0, 1.0), PlotOptions::default());
        let angle2 = a2.angle();
        assert!((angle2 - std::f64::consts::PI / 2.0).abs() < 1e-10); // Pointing up is π/2
    }

    #[test]
    fn test_arrow_bounding_box() {
        let a = Arrow::new((0.0, 0.0), (2.0, 3.0), PlotOptions::default());
        let bbox = a.bounding_box();
        assert_eq!(bbox.xmin, 0.0);
        assert_eq!(bbox.xmax, 2.0);
        assert_eq!(bbox.ymin, 0.0);
        assert_eq!(bbox.ymax, 3.0);
    }

    #[test]
    fn test_arrow_factory() {
        let a = arrow((0.0, 0.0), (1.0, 1.0), None);
        assert_eq!(a.length(), 2.0_f64.sqrt());
    }
}
