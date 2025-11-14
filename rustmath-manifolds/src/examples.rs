//! Example manifolds - concrete implementations

use crate::chart::Chart;
use crate::differentiable::DifferentiableManifold;
use crate::errors::Result;
use crate::point::ManifoldPoint;

/// The real line as a 1-dimensional manifold
///
/// The real line ℝ is the simplest example of a differentiable manifold.
#[derive(Clone, Debug)]
pub struct RealLine {
    manifold: DifferentiableManifold,
}

impl RealLine {
    /// Create a new real line manifold
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_manifolds::RealLine;
    ///
    /// let real_line = RealLine::new();
    /// assert_eq!(real_line.dimension(), 1);
    /// ```
    pub fn new() -> Self {
        let mut manifold = DifferentiableManifold::new("ℝ", 1);

        // Add the standard coordinate chart
        let chart = Chart::new("standard", 1, vec!["x"])
            .expect("Failed to create chart");
        manifold.add_chart(chart).expect("Failed to add chart");

        Self { manifold }
    }

    /// Get the dimension (always 1)
    pub fn dimension(&self) -> usize {
        1
    }

    /// Get the underlying manifold
    pub fn manifold(&self) -> &DifferentiableManifold {
        &self.manifold
    }

    /// Create a point on the real line
    pub fn point(&self, x: f64) -> ManifoldPoint {
        ManifoldPoint::new(vec![x])
    }

    /// Create a named point on the real line
    pub fn named_point(&self, name: impl Into<String>, x: f64) -> ManifoldPoint {
        ManifoldPoint::named(name, vec![x])
    }
}

impl Default for RealLine {
    fn default() -> Self {
        Self::new()
    }
}

/// Euclidean space as an n-dimensional manifold
///
/// Euclidean space ℝⁿ is the fundamental example of a differentiable manifold.
#[derive(Clone, Debug)]
pub struct EuclideanSpace {
    manifold: DifferentiableManifold,
    dimension: usize,
}

impl EuclideanSpace {
    /// Create a new Euclidean space of the given dimension
    ///
    /// # Arguments
    ///
    /// * `dimension` - The dimension n of ℝⁿ
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_manifolds::EuclideanSpace;
    ///
    /// let r2 = EuclideanSpace::new(2);
    /// assert_eq!(r2.dimension(), 2);
    ///
    /// let r3 = EuclideanSpace::new(3);
    /// assert_eq!(r3.dimension(), 3);
    /// ```
    pub fn new(dimension: usize) -> Self {
        let name = format!("ℝ^{}", dimension);
        let mut manifold = DifferentiableManifold::new(name, dimension);

        // Add the standard Cartesian coordinate chart
        let coord_names: Vec<String> = match dimension {
            1 => vec!["x".to_string()],
            2 => vec!["x".to_string(), "y".to_string()],
            3 => vec!["x".to_string(), "y".to_string(), "z".to_string()],
            _ => (1..=dimension).map(|i| format!("x{}", i)).collect(),
        };

        let chart = Chart::new("cartesian", dimension, coord_names)
            .expect("Failed to create chart");
        manifold.add_chart(chart).expect("Failed to add chart");

        Self {
            manifold,
            dimension,
        }
    }

    /// Create 2-dimensional Euclidean space (the plane)
    pub fn plane() -> Self {
        Self::new(2)
    }

    /// Create 3-dimensional Euclidean space
    pub fn space_3d() -> Self {
        Self::new(3)
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the underlying manifold
    pub fn manifold(&self) -> &DifferentiableManifold {
        &self.manifold
    }

    /// Create a point in Euclidean space
    pub fn point(&self, coordinates: Vec<f64>) -> Result<ManifoldPoint> {
        self.manifold.topological().point(coordinates)
    }

    /// Create a named point in Euclidean space
    pub fn named_point(
        &self,
        name: impl Into<String>,
        coordinates: Vec<f64>,
    ) -> Result<ManifoldPoint> {
        self.manifold.topological().named_point(name, coordinates)
    }

    /// Create a point in 2D Euclidean space
    pub fn point_2d(&self, x: f64, y: f64) -> Result<ManifoldPoint> {
        if self.dimension != 2 {
            return Err(crate::errors::ManifoldError::DimensionMismatch {
                expected: 2,
                actual: self.dimension,
            });
        }
        Ok(ManifoldPoint::new(vec![x, y]))
    }

    /// Create a point in 3D Euclidean space
    pub fn point_3d(&self, x: f64, y: f64, z: f64) -> Result<ManifoldPoint> {
        if self.dimension != 3 {
            return Err(crate::errors::ManifoldError::DimensionMismatch {
                expected: 3,
                actual: self.dimension,
            });
        }
        Ok(ManifoldPoint::new(vec![x, y, z]))
    }

    /// Compute Euclidean distance between two points
    pub fn distance(p1: &ManifoldPoint, p2: &ManifoldPoint) -> Result<f64> {
        p1.distance_to(p2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real_line_creation() {
        let real_line = RealLine::new();
        assert_eq!(real_line.dimension(), 1);
        assert_eq!(real_line.manifold().name(), "ℝ");
    }

    #[test]
    fn test_real_line_point() {
        let real_line = RealLine::new();
        let point = real_line.point(3.14);
        assert_eq!(point.dimension(), 1);
        assert_eq!(point.coordinate(0).unwrap(), 3.14);
    }

    #[test]
    fn test_real_line_named_point() {
        let real_line = RealLine::new();
        let point = real_line.named_point("P", 2.71);
        assert_eq!(point.name(), Some("P"));
        assert_eq!(point.coordinate(0).unwrap(), 2.71);
    }

    #[test]
    fn test_euclidean_space_creation() {
        let r2 = EuclideanSpace::new(2);
        assert_eq!(r2.dimension(), 2);
        assert_eq!(r2.manifold().name(), "ℝ^2");

        let r3 = EuclideanSpace::new(3);
        assert_eq!(r3.dimension(), 3);
        assert_eq!(r3.manifold().name(), "ℝ^3");
    }

    #[test]
    fn test_euclidean_plane() {
        let plane = EuclideanSpace::plane();
        assert_eq!(plane.dimension(), 2);
    }

    #[test]
    fn test_euclidean_space_3d() {
        let space = EuclideanSpace::space_3d();
        assert_eq!(space.dimension(), 3);
    }

    #[test]
    fn test_euclidean_space_point() {
        let r3 = EuclideanSpace::new(3);
        let point = r3.point(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(point.dimension(), 3);
        assert_eq!(point.coordinates(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_euclidean_space_point_2d() {
        let plane = EuclideanSpace::plane();
        let point = plane.point_2d(1.0, 2.0).unwrap();
        assert_eq!(point.coordinates(), &[1.0, 2.0]);
    }

    #[test]
    fn test_euclidean_space_point_3d() {
        let space = EuclideanSpace::space_3d();
        let point = space.point_3d(1.0, 2.0, 3.0).unwrap();
        assert_eq!(point.coordinates(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_euclidean_space_point_2d_dimension_mismatch() {
        let space_3d = EuclideanSpace::space_3d();
        assert!(space_3d.point_2d(1.0, 2.0).is_err());
    }

    #[test]
    fn test_euclidean_distance() {
        let plane = EuclideanSpace::plane();
        let p1 = plane.point_2d(0.0, 0.0).unwrap();
        let p2 = plane.point_2d(3.0, 4.0).unwrap();

        let distance = EuclideanSpace::distance(&p1, &p2).unwrap();
        assert!((distance - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_space_named_point() {
        let r3 = EuclideanSpace::new(3);
        let point = r3.named_point("origin", vec![0.0, 0.0, 0.0]).unwrap();
        assert_eq!(point.name(), Some("origin"));
    }

    #[test]
    fn test_euclidean_coordinate_names() {
        let r2 = EuclideanSpace::new(2);
        let chart = &r2.manifold().charts()[0];
        assert_eq!(chart.coordinate_names(), &["x", "y"]);

        let r3 = EuclideanSpace::new(3);
        let chart = &r3.manifold().charts()[0];
        assert_eq!(chart.coordinate_names(), &["x", "y", "z"]);

        let r5 = EuclideanSpace::new(5);
        let chart = &r5.manifold().charts()[0];
        assert_eq!(chart.coordinate_names(), &["x1", "x2", "x3", "x4", "x5"]);
    }
}
