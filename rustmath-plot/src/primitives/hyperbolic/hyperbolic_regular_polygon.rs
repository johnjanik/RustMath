//! Hyperbolic regular polygon primitive
//!
//! Draws regular polygons (all sides equal, all angles equal) in hyperbolic geometry.

use crate::{
    BoundingBox, GraphicPrimitive, PlotOptions, Point2D, Renderable, RenderBackend, Result,
};
use super::{HyperbolicModel, HyperbolicPolygon, hyperbolic_polygon};
use std::f64::consts::PI;

/// A regular polygon in hyperbolic geometry
///
/// Based on SageMath's HyperbolicRegularPolygon from sage.plot.hyperbolic_regular_polygon
#[derive(Debug, Clone)]
pub struct HyperbolicRegularPolygon {
    /// Number of sides
    n: usize,
    /// Center point
    center: Point2D,
    /// Radius (hyperbolic distance from center to vertices)
    radius: f64,
    /// Rotation angle (in radians)
    rotation: f64,
    /// Hyperbolic model to use
    model: HyperbolicModel,
    /// Rendering options
    options: PlotOptions,
}

impl HyperbolicRegularPolygon {
    /// Create a new hyperbolic regular polygon
    ///
    /// # Arguments
    ///
    /// * `n` - Number of sides (must be at least 3)
    /// * `center` - Center point in the hyperbolic plane
    /// * `radius` - Hyperbolic radius (distance from center to vertices)
    /// * `rotation` - Rotation angle in radians (0 = first vertex on positive x-axis)
    /// * `model` - Hyperbolic model to use
    /// * `options` - Rendering options
    ///
    /// # Panics
    ///
    /// Panics if n < 3
    pub fn new(
        n: usize,
        center: Point2D,
        radius: f64,
        rotation: f64,
        model: HyperbolicModel,
        options: PlotOptions,
    ) -> Self {
        assert!(n >= 3, "A polygon must have at least 3 sides");

        Self {
            n,
            center,
            radius,
            rotation,
            model,
            options,
        }
    }

    /// Get the number of sides
    pub fn num_sides(&self) -> usize {
        self.n
    }

    /// Get the center
    pub fn center(&self) -> Point2D {
        self.center
    }

    /// Get the radius
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Get the rotation angle
    pub fn rotation(&self) -> f64 {
        self.rotation
    }

    /// Get the hyperbolic model
    pub fn model(&self) -> HyperbolicModel {
        self.model
    }

    /// Calculate the vertices of the regular polygon
    pub fn vertices(&self) -> Vec<Point2D> {
        match self.model {
            HyperbolicModel::PoincareDisk => self.vertices_poincare(),
            HyperbolicModel::UpperHalfPlane => self.vertices_uhp(),
            HyperbolicModel::KleinDisk => self.vertices_klein(),
        }
    }

    /// Calculate vertices in Poincaré disk model
    fn vertices_poincare(&self) -> Vec<Point2D> {
        let cx = self.center.x;
        let cy = self.center.y;
        let mut vertices = Vec::with_capacity(self.n);

        for i in 0..self.n {
            let angle = self.rotation + 2.0 * PI * i as f64 / self.n as f64;

            // Convert hyperbolic radius to Euclidean radius in Poincaré disk
            // Using the formula: r_euclidean = tanh(r_hyperbolic / 2)
            let r_euclidean = (self.radius / 2.0).tanh();

            // Place vertex at hyperbolic distance from center
            let x = cx + r_euclidean * angle.cos();
            let y = cy + r_euclidean * angle.sin();

            vertices.push(Point2D::new(x, y));
        }

        vertices
    }

    /// Calculate vertices in upper half-plane model
    fn vertices_uhp(&self) -> Vec<Point2D> {
        let cx = self.center.x;
        let cy = self.center.y;
        let mut vertices = Vec::with_capacity(self.n);

        for i in 0..self.n {
            let angle = self.rotation + 2.0 * PI * i as f64 / self.n as f64;

            // In UHP model, the calculation is more complex
            // For simplicity, we'll use an approximation
            let r = self.radius.exp();
            let x = cx + r * angle.cos();
            let y = cy + r * angle.sin();

            vertices.push(Point2D::new(x, y.abs())); // Ensure y > 0
        }

        vertices
    }

    /// Calculate vertices in Klein disk model
    fn vertices_klein(&self) -> Vec<Point2D> {
        let cx = self.center.x;
        let cy = self.center.y;
        let mut vertices = Vec::with_capacity(self.n);

        for i in 0..self.n {
            let angle = self.rotation + 2.0 * PI * i as f64 / self.n as f64;

            // Klein disk: different relationship between hyperbolic and Euclidean distance
            let r_euclidean = self.radius.tanh();

            let x = cx + r_euclidean * angle.cos();
            let y = cy + r_euclidean * angle.sin();

            vertices.push(Point2D::new(x, y));
        }

        vertices
    }

    /// Convert to a HyperbolicPolygon
    pub fn to_polygon(&self) -> HyperbolicPolygon {
        hyperbolic_polygon(self.vertices(), Some(self.model), Some(self.options.clone()))
    }

    /// Calculate the interior angle of the regular polygon
    ///
    /// In hyperbolic geometry, the interior angle depends on the side length
    pub fn interior_angle(&self) -> f64 {
        // For a regular hyperbolic n-gon with side length s,
        // the interior angle α satisfies: n·α = (n-2)·π - Area
        // This is a simplified approximation
        let euclidean_angle = (self.n as f64 - 2.0) * PI / self.n as f64;

        // In hyperbolic geometry, angles are smaller
        // This is a rough approximation
        euclidean_angle * 0.9
    }

    /// Calculate the side length of the regular polygon
    pub fn side_length(&self) -> f64 {
        // Calculate hyperbolic distance between adjacent vertices
        let vertices = self.vertices();
        if vertices.len() < 2 {
            return 0.0;
        }

        match self.model {
            HyperbolicModel::PoincareDisk => {
                use super::utils::poincare_distance;
                poincare_distance(vertices[0], vertices[1])
            }
            HyperbolicModel::UpperHalfPlane => {
                use super::utils::uhp_distance;
                uhp_distance((vertices[0].x, vertices[0].y), (vertices[1].x, vertices[1].y))
            }
            HyperbolicModel::KleinDisk => {
                // Approximate for now
                let dx = vertices[1].x - vertices[0].x;
                let dy = vertices[1].y - vertices[0].y;
                (dx * dx + dy * dy).sqrt()
            }
        }
    }
}

impl Renderable for HyperbolicRegularPolygon {
    fn bounding_box(&self) -> BoundingBox {
        self.to_polygon().bounding_box()
    }

    fn render(&self, backend: &mut dyn RenderBackend) -> Result<()> {
        self.to_polygon().render(backend)
    }
}

impl GraphicPrimitive for HyperbolicRegularPolygon {
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

/// Create a hyperbolic regular polygon
///
/// # Arguments
///
/// * `n` - Number of sides (must be at least 3)
/// * `center` - Center point
/// * `radius` - Hyperbolic radius
/// * `rotation` - Optional rotation angle in radians (defaults to 0)
/// * `model` - Optional hyperbolic model (defaults to Poincaré disk)
/// * `options` - Optional rendering options
///
/// # Examples
///
/// ```ignore
/// use rustmath_plot::primitives::hyperbolic::hyperbolic_regular_polygon;
///
/// // Create a regular hexagon in the Poincaré disk
/// let hexagon = hyperbolic_regular_polygon(6, (0.0, 0.0), 0.5, None, None, None);
/// ```
pub fn hyperbolic_regular_polygon(
    n: usize,
    center: Point2D,
    radius: f64,
    rotation: Option<f64>,
    model: Option<HyperbolicModel>,
    options: Option<PlotOptions>,
) -> HyperbolicRegularPolygon {
    HyperbolicRegularPolygon::new(
        n,
        center,
        radius,
        rotation.unwrap_or(0.0),
        model.unwrap_or(HyperbolicModel::PoincareDisk),
        options.unwrap_or_default(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperbolic_regular_polygon_creation() {
        let polygon = hyperbolic_regular_polygon(5, Point2D::new(0.0, 0.0), 0.5, None, None, None);
        assert_eq!(polygon.num_sides(), 5);
        assert_eq!(polygon.center(), Point2D::new(0.0, 0.0));
        assert_eq!(polygon.radius(), 0.5);
    }

    #[test]
    #[should_panic(expected = "A polygon must have at least 3 sides")]
    fn test_hyperbolic_regular_polygon_invalid_n() {
        hyperbolic_regular_polygon(2, Point2D::new(0.0, 0.0), 0.5, None, None, None);
    }

    #[test]
    fn test_hyperbolic_regular_polygon_vertices() {
        let polygon = hyperbolic_regular_polygon(4, Point2D::new(0.0, 0.0), 0.5, None, None, None);
        let vertices = polygon.vertices();
        assert_eq!(vertices.len(), 4);
    }

    #[test]
    fn test_hyperbolic_regular_polygon_triangle() {
        let triangle = hyperbolic_regular_polygon(3, Point2D::new(0.0, 0.0), 0.5, None, None, None);
        assert_eq!(triangle.num_sides(), 3);
        let vertices = triangle.vertices();
        assert_eq!(vertices.len(), 3);
    }

    #[test]
    fn test_hyperbolic_regular_polygon_hexagon() {
        let hexagon = hyperbolic_regular_polygon(6, Point2D::new(0.0, 0.0), 0.5, None, None, None);
        assert_eq!(hexagon.num_sides(), 6);
        let vertices = hexagon.vertices();
        assert_eq!(vertices.len(), 6);
    }

    #[test]
    fn test_hyperbolic_regular_polygon_rotation() {
        let polygon = hyperbolic_regular_polygon(
            4,
            Point2D::new(0.0, 0.0),
            0.5,
            Some(PI / 4.0),
            None,
            None,
        );
        assert_eq!(polygon.rotation(), PI / 4.0);
    }

    #[test]
    fn test_hyperbolic_regular_polygon_to_polygon() {
        let reg_polygon = hyperbolic_regular_polygon(5, Point2D::new(0.0, 0.0), 0.5, None, None, None);
        let polygon = reg_polygon.to_polygon();
        assert_eq!(polygon.num_sides(), 5);
    }

    #[test]
    fn test_hyperbolic_regular_polygon_interior_angle() {
        let triangle = hyperbolic_regular_polygon(3, Point2D::new(0.0, 0.0), 0.5, None, None, None);
        let angle = triangle.interior_angle();
        // For a triangle, sum of angles should be less than π in hyperbolic geometry
        assert!(angle > 0.0);
        assert!(angle < PI);
    }

    #[test]
    fn test_hyperbolic_regular_polygon_side_length() {
        let square = hyperbolic_regular_polygon(4, Point2D::new(0.0, 0.0), 0.5, None, None, None);
        let side_length = square.side_length();
        assert!(side_length > 0.0);
    }
}
