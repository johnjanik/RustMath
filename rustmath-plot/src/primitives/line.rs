//! Line primitive for plotting
//!
//! Based on SageMath's sage.plot.line module

use rustmath_plot_core::{
    BoundingBox, GraphicPrimitive, PlotOptions, Point2D, Renderable, RenderBackend, Result,
};

/// A line connecting a series of points
///
/// Based on SageMath's Line class from sage.plot.line
pub struct Line {
    /// The points defining the line segments
    points: Vec<Point2D>,

    /// Plot options (color, thickness, line style, etc.)
    options: PlotOptions,
}

impl Line {
    /// Create a new Line primitive
    ///
    /// # Arguments
    /// * `points` - The points to connect with line segments
    /// * `options` - Plotting options
    ///
    /// # Examples
    /// ```ignore
    /// let points = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)];
    /// let line = Line::new(points, PlotOptions::default());
    /// ```
    pub fn new(points: Vec<impl Into<Point2D>>, options: PlotOptions) -> Self {
        Self {
            points: points.into_iter().map(|p| p.into()).collect(),
            options,
        }
    }

    /// Create a line segment between two points
    pub fn segment(start: impl Into<Point2D>, end: impl Into<Point2D>, options: PlotOptions) -> Self {
        Self {
            points: vec![start.into(), end.into()],
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

    /// Check if this forms a valid line (at least 2 points)
    pub fn is_valid(&self) -> bool {
        self.points.len() >= 2
    }
}

impl Renderable for Line {
    fn bounding_box(&self) -> BoundingBox {
        if self.points.is_empty() {
            return BoundingBox::empty();
        }

        BoundingBox::from_points(&self.points).unwrap_or_else(|_| BoundingBox::empty())
    }

    fn render(&self, backend: &mut dyn RenderBackend) -> Result<()> {
        if self.points.len() < 2 {
            // Nothing to render
            return Ok(());
        }

        backend.draw_line(&self.points, &self.options)
    }
}

impl GraphicPrimitive for Line {
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

/// Factory function to create a Line primitive
///
/// # Arguments
/// * `points` - The points to connect with line segments
/// * `options` - Optional plot options
///
/// # Examples
/// ```ignore
/// use rustmath_plot::primitives::line;
///
/// let l1 = line(vec![(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)], None);
/// let l2 = line(vec![(0.0, 0.0), (1.0, 0.0)], Some(PlotOptions::default().with_color(Color::red_color())));
/// ```
pub fn line(points: Vec<impl Into<Point2D>>, options: Option<PlotOptions>) -> Box<Line> {
    Box::new(Line::new(points, options.unwrap_or_default()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_creation() {
        let points = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)];
        let l = Line::new(points, PlotOptions::default());
        assert_eq!(l.len(), 3);
        assert!(!l.is_empty());
        assert!(l.is_valid());
    }

    #[test]
    fn test_line_segment() {
        let l = Line::segment((0.0, 0.0), (1.0, 1.0), PlotOptions::default());
        assert_eq!(l.len(), 2);
        assert!(l.is_valid());
        assert_eq!(l.points()[0], Point2D::new(0.0, 0.0));
        assert_eq!(l.points()[1], Point2D::new(1.0, 1.0));
    }

    #[test]
    fn test_line_bounding_box() {
        let points = vec![(0.0, 0.0), (2.0, 3.0), (1.0, 1.0)];
        let l = Line::new(points, PlotOptions::default());
        let bbox = l.bounding_box();
        assert_eq!(bbox.xmin, 0.0);
        assert_eq!(bbox.xmax, 2.0);
        assert_eq!(bbox.ymin, 0.0);
        assert_eq!(bbox.ymax, 3.0);
    }

    #[test]
    fn test_line_invalid() {
        let l = Line::new(vec![(0.0, 0.0)], PlotOptions::default());
        assert!(!l.is_valid());
    }

    #[test]
    fn test_line_factory() {
        let l = line(vec![(0.0, 0.0), (1.0, 1.0)], None);
        assert_eq!(l.len(), 2);
    }
}
