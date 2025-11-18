//! Polygon primitive for plotting
//!
//! Based on SageMath's sage.plot.polygon module

use rustmath_colors::Color;
use rustmath_plot_core::{
    BoundingBox, GraphicPrimitive, PlotOptions, Point2D, Renderable, RenderBackend, Result,
};

/// A closed polygon
///
/// Based on SageMath's Polygon class from sage.plot.polygon
pub struct Polygon {
    /// The vertices of the polygon
    vertices: Vec<Point2D>,

    /// Plot options (fill, edge color, etc.)
    options: PlotOptions,
}

impl Polygon {
    /// Create a new Polygon primitive
    ///
    /// # Arguments
    /// * `vertices` - The vertices of the polygon
    /// * `options` - Plotting options
    ///
    /// # Examples
    /// ```ignore
    /// let vertices = vec![(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)];
    /// let polygon = Polygon::new(vertices, PlotOptions::default());
    /// ```
    pub fn new(vertices: Vec<impl Into<Point2D>>, options: PlotOptions) -> Self {
        Self {
            vertices: vertices.into_iter().map(|p| p.into()).collect(),
            options,
        }
    }

    /// Create a rectangle
    pub fn rectangle(
        min: impl Into<Point2D>,
        max: impl Into<Point2D>,
        options: PlotOptions,
    ) -> Self {
        let min = min.into();
        let max = max.into();
        Self {
            vertices: vec![
                Point2D::new(min.x, min.y),
                Point2D::new(max.x, min.y),
                Point2D::new(max.x, max.y),
                Point2D::new(min.x, max.y),
            ],
            options,
        }
    }

    /// Create a regular polygon with n sides
    pub fn regular(center: impl Into<Point2D>, radius: f64, n_sides: usize, options: PlotOptions) -> Self {
        assert!(n_sides >= 3, "polygon must have at least 3 sides");
        assert!(radius > 0.0, "radius must be positive");

        let center = center.into();
        let angle_step = 2.0 * std::f64::consts::PI / n_sides as f64;

        let vertices = (0..n_sides)
            .map(|i| {
                let angle = i as f64 * angle_step - std::f64::consts::PI / 2.0; // Start at top
                Point2D::new(
                    center.x + radius * angle.cos(),
                    center.y + radius * angle.sin(),
                )
            })
            .collect();

        Self { vertices, options }
    }

    /// Get the vertices
    pub fn vertices(&self) -> &[Point2D] {
        &self.vertices
    }

    /// Get the number of vertices
    pub fn len(&self) -> usize {
        self.vertices.len()
    }

    /// Check if there are no vertices
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// Check if this forms a valid polygon (at least 3 vertices)
    pub fn is_valid(&self) -> bool {
        self.vertices.len() >= 3
    }

    /// Calculate the perimeter
    pub fn perimeter(&self) -> f64 {
        if self.vertices.len() < 2 {
            return 0.0;
        }

        let mut sum = 0.0;
        for i in 0..self.vertices.len() {
            let next = (i + 1) % self.vertices.len();
            sum += self.vertices[i].distance_to(&self.vertices[next]);
        }
        sum
    }
}

impl Renderable for Polygon {
    fn bounding_box(&self) -> BoundingBox {
        if self.vertices.is_empty() {
            return BoundingBox::empty();
        }

        BoundingBox::from_points(&self.vertices).unwrap_or_else(|_| BoundingBox::empty())
    }

    fn render(&self, backend: &mut dyn RenderBackend) -> Result<()> {
        if self.vertices.len() < 3 {
            // Nothing to render
            return Ok(());
        }

        backend.draw_polygon(&self.vertices, &self.options)
    }
}

impl GraphicPrimitive for Polygon {
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

/// Factory function to create a Polygon primitive
///
/// # Arguments
/// * `vertices` - The vertices of the polygon
/// * `options` - Optional plot options
///
/// # Examples
/// ```ignore
/// use rustmath_plot::primitives::polygon;
///
/// let p1 = polygon(vec![(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)], None);
/// let p2 = polygon(vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
///                  Some(PlotOptions::default().with_fill(Color::blue_color())));
/// ```
pub fn polygon(vertices: Vec<impl Into<Point2D>>, options: Option<PlotOptions>) -> Box<Polygon> {
    Box::new(Polygon::new(vertices, options.unwrap_or_default()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polygon_creation() {
        let vertices = vec![(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)];
        let p = Polygon::new(vertices, PlotOptions::default());
        assert_eq!(p.len(), 3);
        assert!(!p.is_empty());
        assert!(p.is_valid());
    }

    #[test]
    fn test_polygon_rectangle() {
        let p = Polygon::rectangle((0.0, 0.0), (2.0, 3.0), PlotOptions::default());
        assert_eq!(p.len(), 4);
        assert!(p.is_valid());
    }

    #[test]
    fn test_polygon_regular() {
        let p = Polygon::regular((0.0, 0.0), 1.0, 6, PlotOptions::default());
        assert_eq!(p.len(), 6);
        assert!(p.is_valid());
    }

    #[test]
    fn test_polygon_bounding_box() {
        let vertices = vec![(0.0, 0.0), (2.0, 1.0), (1.0, 3.0)];
        let p = Polygon::new(vertices, PlotOptions::default());
        let bbox = p.bounding_box();
        assert_eq!(bbox.xmin, 0.0);
        assert_eq!(bbox.xmax, 2.0);
        assert_eq!(bbox.ymin, 0.0);
        assert_eq!(bbox.ymax, 3.0);
    }

    #[test]
    fn test_polygon_perimeter() {
        // Unit square
        let p = Polygon::rectangle((0.0, 0.0), (1.0, 1.0), PlotOptions::default());
        let perimeter = p.perimeter();
        assert!((perimeter - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_polygon_invalid() {
        let p = Polygon::new(vec![(0.0, 0.0), (1.0, 1.0)], PlotOptions::default());
        assert!(!p.is_valid());
    }

    #[test]
    #[should_panic(expected = "polygon must have at least 3 sides")]
    fn test_polygon_regular_too_few_sides() {
        Polygon::regular((0.0, 0.0), 1.0, 2, PlotOptions::default());
    }

    #[test]
    fn test_polygon_factory() {
        let p = polygon(vec![(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)], None);
        assert_eq!(p.len(), 3);
    }
}
