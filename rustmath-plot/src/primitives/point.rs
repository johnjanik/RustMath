//! Point primitive for plotting
//!
//! Based on SageMath's sage.plot.point module

use rustmath_plot_core::{
    BoundingBox, GraphicPrimitive, PlotOptions, Point2D, Renderable, RenderBackend, Result,
};

/// A collection of points to be plotted
///
/// Based on SageMath's Point class from sage.plot.point
pub struct Point {
    /// The points to plot
    points: Vec<Point2D>,

    /// Plot options (color, marker style, size, etc.)
    options: PlotOptions,
}

impl Point {
    /// Create a new Point primitive
    ///
    /// # Arguments
    /// * `points` - The points to plot
    /// * `options` - Plotting options
    ///
    /// # Examples
    /// ```ignore
    /// let points = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)];
    /// let p = Point::new(points, PlotOptions::default());
    /// ```
    pub fn new(points: Vec<impl Into<Point2D>>, options: PlotOptions) -> Self {
        Self {
            points: points.into_iter().map(|p| p.into()).collect(),
            options,
        }
    }

    /// Create a single point
    pub fn single(x: f64, y: f64, options: PlotOptions) -> Self {
        Self {
            points: vec![Point2D::new(x, y)],
            options,
        }
    }

    /// Get the points
    pub fn points(&self) -> &[Point2D] {
        &self.points
    }

    /// Get the number of points
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if there are no points
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

impl Renderable for Point {
    fn bounding_box(&self) -> BoundingBox {
        if self.points.is_empty() {
            return BoundingBox::empty();
        }

        BoundingBox::from_points(&self.points).unwrap_or_else(|_| BoundingBox::empty())
    }

    fn render(&self, backend: &mut dyn RenderBackend) -> Result<()> {
        backend.draw_points(&self.points, &self.options)
    }
}

impl GraphicPrimitive for Point {
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

/// Factory function to create a Point primitive
///
/// # Arguments
/// * `points` - The points to plot (can be tuples, Point2D, etc.)
/// * `options` - Optional plot options
///
/// # Examples
/// ```ignore
/// use rustmath_plot::primitives::point;
///
/// let p1 = point(vec![(0.0, 0.0), (1.0, 1.0)], None);
/// let p2 = point(vec![(2.0, 3.0)], Some(PlotOptions::default().with_color(Color::red_color())));
/// ```
pub fn point(points: Vec<impl Into<Point2D>>, options: Option<PlotOptions>) -> Box<Point> {
    Box::new(Point::new(points, options.unwrap_or_default()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_creation() {
        let points = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)];
        let p = Point::new(points, PlotOptions::default());
        assert_eq!(p.len(), 3);
        assert!(!p.is_empty());
    }

    #[test]
    fn test_single_point() {
        let p = Point::single(1.0, 2.0, PlotOptions::default());
        assert_eq!(p.len(), 1);
        assert_eq!(p.points()[0], Point2D::new(1.0, 2.0));
    }

    #[test]
    fn test_point_bounding_box() {
        let points = vec![(0.0, 0.0), (2.0, 3.0)];
        let p = Point::new(points, PlotOptions::default());
        let bbox = p.bounding_box();
        assert_eq!(bbox.xmin, 0.0);
        assert_eq!(bbox.xmax, 2.0);
        assert_eq!(bbox.ymin, 0.0);
        assert_eq!(bbox.ymax, 3.0);
    }

    #[test]
    fn test_point_empty() {
        let p = Point::new(Vec::<(f64, f64)>::new(), PlotOptions::default());
        assert!(p.is_empty());
        assert_eq!(p.len(), 0);
    }

    #[test]
    fn test_point_factory() {
        let p = point(vec![(1.0, 2.0)], None);
        assert_eq!(p.len(), 1);
    }
}
