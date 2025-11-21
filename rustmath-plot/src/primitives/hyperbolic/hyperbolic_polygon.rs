//! Hyperbolic polygon primitive
//!
//! Draws polygons with geodesic sides in hyperbolic geometry.

use crate::{
    BoundingBox, GraphicPrimitive, PlotOptions, Point2D, Renderable, RenderBackend, Result,
};
use super::{HyperbolicModel, HyperbolicArc, hyperbolic_arc};

/// A polygon with geodesic sides in hyperbolic geometry
///
/// Based on SageMath's HyperbolicPolygon from sage.plot.hyperbolic_polygon
#[derive(Debug, Clone)]
pub struct HyperbolicPolygon {
    /// Vertices of the polygon
    vertices: Vec<Point2D>,
    /// Hyperbolic model to use
    model: HyperbolicModel,
    /// Rendering options
    options: PlotOptions,
}

impl HyperbolicPolygon {
    /// Create a new hyperbolic polygon
    ///
    /// # Arguments
    ///
    /// * `vertices` - Vertices of the polygon (in order)
    /// * `model` - Hyperbolic model to use
    /// * `options` - Rendering options
    pub fn new(vertices: Vec<Point2D>, model: HyperbolicModel, options: PlotOptions) -> Self {
        Self {
            vertices,
            model,
            options,
        }
    }

    /// Get the vertices
    pub fn vertices(&self) -> &[Point2D] {
        &self.vertices
    }

    /// Get the hyperbolic model
    pub fn model(&self) -> HyperbolicModel {
        self.model
    }

    /// Get the number of sides
    pub fn num_sides(&self) -> usize {
        self.vertices.len()
    }

    /// Get the geodesic edges of the polygon
    pub fn edges(&self) -> Vec<HyperbolicArc> {
        let mut edges = Vec::new();
        let n = self.vertices.len();

        for i in 0..n {
            let start = self.vertices[i];
            let end = self.vertices[(i + 1) % n];
            edges.push(hyperbolic_arc(start, end, Some(self.model), Some(self.options.clone())));
        }

        edges
    }

    /// Sample all edges as a continuous path
    pub fn sample_edges(&self, points_per_edge: usize) -> Vec<Point2D> {
        let mut result = Vec::new();

        for edge in self.edges() {
            let samples = edge.sample(points_per_edge);
            // Don't duplicate the end point (it will be the start of next edge)
            result.extend(&samples[..samples.len() - 1]);
        }

        // Add the last point to close the polygon
        if let Some(&first) = self.vertices.first() {
            result.push(first);
        }

        result
    }

    /// Calculate the area of the hyperbolic polygon
    ///
    /// Uses the Gauss-Bonnet theorem: Area = (n-2)π - Σ(interior angles)
    pub fn area(&self) -> f64 {
        // TODO: Implement hyperbolic area calculation
        // This requires computing the interior angles at each vertex
        // For now, return 0.0 as a placeholder
        0.0
    }
}

impl Renderable for HyperbolicPolygon {
    fn bounding_box(&self) -> BoundingBox {
        if self.vertices.is_empty() {
            return BoundingBox::new(0.0, 0.0, 0.0, 0.0);
        }

        let samples = self.sample_edges(20);
        if samples.is_empty() {
            return BoundingBox::new(0.0, 0.0, 0.0, 0.0);
        }

        let mut xmin = samples[0].x;
        let mut xmax = samples[0].x;
        let mut ymin = samples[0].y;
        let mut ymax = samples[0].y;

        for point in &samples {
            xmin = xmin.min(point.x);
            xmax = xmax.max(point.x);
            ymin = ymin.min(point.y);
            ymax = ymax.max(point.y);
        }

        BoundingBox::new(xmin, xmax, ymin, ymax)
    }

    fn render(&self, backend: &mut dyn RenderBackend) -> Result<()> {
        let samples = self.sample_edges(50);

        // Draw filled polygon if fill color is set
        if self.options.fill {
            backend.draw_polygon(&samples, &self.options)?;
        } else {
            // Just draw the outline
            backend.draw_line(&samples, &self.options)?;
        }

        Ok(())
    }
}

impl GraphicPrimitive for HyperbolicPolygon {
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

/// Create a hyperbolic polygon
///
/// # Arguments
///
/// * `vertices` - Vertices of the polygon (in order)
/// * `model` - Optional hyperbolic model (defaults to Poincaré disk)
/// * `options` - Optional rendering options
///
/// # Examples
///
/// ```ignore
/// use rustmath_plot::primitives::hyperbolic::hyperbolic_polygon;
///
/// let vertices = vec![
///     (0.0, 0.0),
///     (0.5, 0.0),
///     (0.5, 0.5),
///     (0.0, 0.5),
/// ];
/// let polygon = hyperbolic_polygon(vertices, None, None);
/// ```
pub fn hyperbolic_polygon(
    vertices: Vec<Point2D>,
    model: Option<HyperbolicModel>,
    options: Option<PlotOptions>,
) -> HyperbolicPolygon {
    HyperbolicPolygon::new(
        vertices,
        model.unwrap_or(HyperbolicModel::PoincareDisk),
        options.unwrap_or_default(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperbolic_polygon_creation() {
        let vertices = vec![Point2D::new(0.0, 0.0), Point2D::new(0.5, 0.0), Point2D::new(0.25, 0.5)];
        let polygon = hyperbolic_polygon(vertices.clone(), None, None);
        assert_eq!(polygon.vertices(), &vertices);
        assert_eq!(polygon.num_sides(), 3);
    }

    #[test]
    fn test_hyperbolic_polygon_edges() {
        let vertices = vec![Point2D::new(0.0, 0.0), Point2D::new(0.5, 0.0), Point2D::new(0.25, 0.5)];
        let polygon = hyperbolic_polygon(vertices, None, None);
        let edges = polygon.edges();
        assert_eq!(edges.len(), 3);
    }

    #[test]
    fn test_hyperbolic_polygon_sample_edges() {
        let vertices = vec![Point2D::new(0.0, 0.0), Point2D::new(0.5, 0.0), Point2D::new(0.5, 0.5)];
        let polygon = hyperbolic_polygon(vertices, None, None);
        let samples = polygon.sample_edges(10);
        // Should have 3 edges * (10-1) points + 1 closing point = 28
        assert_eq!(samples.len(), 28);
    }

    #[test]
    fn test_hyperbolic_polygon_bounding_box() {
        let vertices = vec![Point2D::new(0.0, 0.0), Point2D::new(0.5, 0.0), Point2D::new(0.5, 0.5), Point2D::new(0.0, 0.5)];
        let polygon = hyperbolic_polygon(vertices, None, None);
        let bbox = polygon.bounding_box();
        assert!(bbox.xmin <= 0.0);
        assert!(bbox.xmax >= 0.5);
        assert!(bbox.ymin <= 0.0);
        assert!(bbox.ymax >= 0.5);
    }
}
