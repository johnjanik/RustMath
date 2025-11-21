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

impl Into<DifferentiableManifold> for EuclideanSpace {
    fn into(self) -> DifferentiableManifold {
        self.manifold
    }
}

/// The circle S¹ = {(x,y) ∈ ℝ² : x² + y² = 1}
///
/// The circle is a 1-dimensional compact manifold, often parameterized by angle θ ∈ [0, 2π).
/// We use two charts to cover the circle (stereographic projections or angular coordinates).
#[derive(Clone, Debug)]
pub struct Circle {
    manifold: DifferentiableManifold,
}

impl Circle {
    /// Create a new circle manifold S¹
    ///
    /// The circle is covered by two charts:
    /// - Chart 1: U₁ = S¹ \ {(0,1)}, coordinates θ ∈ (-π, π)
    /// - Chart 2: U₂ = S¹ \ {(0,-1)}, coordinates θ ∈ (0, 2π)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_manifolds::Circle;
    ///
    /// let circle = Circle::new();
    /// assert_eq!(circle.dimension(), 1);
    /// ```
    pub fn new() -> Self {
        let mut manifold = DifferentiableManifold::new("S¹", 1);

        // Chart 1: U₁ covers all but the north pole (0,1)
        let chart1 = Chart::new("chart1", 1, vec!["θ"])
            .expect("Failed to create chart1");
        manifold.add_chart(chart1).expect("Failed to add chart1");

        // Chart 2: U₂ covers all but the south pole (0,-1)
        let chart2 = Chart::new("chart2", 1, vec!["θ"])
            .expect("Failed to create chart2");
        manifold.add_chart(chart2).expect("Failed to add chart2");

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

    /// Create a point on the circle from an angle
    ///
    /// # Arguments
    ///
    /// * `theta` - Angle in radians
    pub fn point(&self, theta: f64) -> ManifoldPoint {
        ManifoldPoint::new(vec![theta])
    }

    /// Create a point from Cartesian coordinates (x, y) on the unit circle
    ///
    /// The point is normalized to lie on the unit circle.
    pub fn from_cartesian(&self, x: f64, y: f64) -> ManifoldPoint {
        let theta = y.atan2(x);
        ManifoldPoint::new(vec![theta])
    }

    /// Convert a point to Cartesian coordinates
    pub fn to_cartesian(&self, point: &ManifoldPoint) -> Result<(f64, f64)> {
        if point.dimension() != 1 {
            return Err(crate::errors::ManifoldError::DimensionMismatch {
                expected: 1,
                actual: point.dimension(),
            });
        }

        let theta = point.coordinate(0)?;
        Ok((theta.cos(), theta.sin()))
    }
}

impl Default for Circle {
    fn default() -> Self {
        Self::new()
    }
}

/// The 2-sphere S² = {(x,y,z) ∈ ℝ³ : x² + y² + z² = 1}
///
/// The 2-sphere is a 2-dimensional compact manifold.
/// We use stereographic projections from the north and south poles as charts.
#[derive(Clone, Debug)]
pub struct Sphere2 {
    manifold: DifferentiableManifold,
}

impl Sphere2 {
    /// Create a new 2-sphere manifold S²
    ///
    /// The sphere is covered by two stereographic charts:
    /// - North chart: Projects from north pole onto the equatorial plane
    /// - South chart: Projects from south pole onto the equatorial plane
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_manifolds::Sphere2;
    ///
    /// let sphere = Sphere2::new();
    /// assert_eq!(sphere.dimension(), 2);
    /// ```
    pub fn new() -> Self {
        let mut manifold = DifferentiableManifold::new("S²", 2);

        // Stereographic projection from north pole
        // (x,y,z) ↦ (u,v) = (x/(1-z), y/(1-z))
        let north_chart = Chart::new("north_stereo", 2, vec!["u", "v"])
            .expect("Failed to create north chart");
        manifold.add_chart(north_chart).expect("Failed to add north chart");

        // Stereographic projection from south pole
        // (x,y,z) ↦ (u,v) = (x/(1+z), y/(1+z))
        let south_chart = Chart::new("south_stereo", 2, vec!["u", "v"])
            .expect("Failed to create south chart");
        manifold.add_chart(south_chart).expect("Failed to add south chart");

        Self { manifold }
    }

    /// Get the dimension (always 2)
    pub fn dimension(&self) -> usize {
        2
    }

    /// Get the underlying manifold
    pub fn manifold(&self) -> &DifferentiableManifold {
        &self.manifold
    }

    /// Create a point from spherical coordinates (theta, phi)
    ///
    /// # Arguments
    ///
    /// * `theta` - Azimuthal angle in radians, θ ∈ [0, 2π)
    /// * `phi` - Polar angle from north pole, φ ∈ [0, π]
    pub fn from_spherical(&self, theta: f64, phi: f64) -> Result<ManifoldPoint> {
        // Convert to stereographic coordinates (north chart)
        // x = sin(φ)cos(θ), y = sin(φ)sin(θ), z = cos(φ)
        // u = x/(1-z), v = y/(1-z)

        if (phi - std::f64::consts::PI).abs() < 1e-10 {
            // South pole - use south chart
            Ok(ManifoldPoint::new(vec![0.0, 0.0]))
        } else {
            let u = (phi.sin() * theta.cos()) / (1.0 - phi.cos());
            let v = (phi.sin() * theta.sin()) / (1.0 - phi.cos());
            Ok(ManifoldPoint::new(vec![u, v]))
        }
    }

    /// Create a point from Cartesian coordinates (x, y, z) on the unit sphere
    ///
    /// The point is normalized to lie on the unit sphere.
    pub fn from_cartesian(&self, x: f64, y: f64, z: f64) -> Result<ManifoldPoint> {
        // Normalize
        let r = (x * x + y * y + z * z).sqrt();
        let (nx, ny, nz) = (x / r, y / r, z / r);

        // Use north stereographic projection if not at north pole
        if (1.0 - nz).abs() > 1e-10 {
            let u = nx / (1.0 - nz);
            let v = ny / (1.0 - nz);
            Ok(ManifoldPoint::new(vec![u, v]))
        } else {
            // At north pole, use south chart
            Ok(ManifoldPoint::new(vec![0.0, 0.0]))
        }
    }

    /// Convert a point to Cartesian coordinates
    pub fn to_cartesian(&self, point: &ManifoldPoint) -> Result<(f64, f64, f64)> {
        if point.dimension() != 2 {
            return Err(crate::errors::ManifoldError::DimensionMismatch {
                expected: 2,
                actual: point.dimension(),
            });
        }

        let u = point.coordinate(0)?;
        let v = point.coordinate(1)?;

        // Inverse of north stereographic projection
        // (u,v) ↦ (x,y,z) = (2u, 2v, u²+v²-1) / (u²+v²+1)
        let denom = u * u + v * v + 1.0;
        let x = 2.0 * u / denom;
        let y = 2.0 * v / denom;
        let z = (u * u + v * v - 1.0) / denom;

        Ok((x, y, z))
    }
}

impl Into<DifferentiableManifold> for Sphere2 {
    fn into(self) -> DifferentiableManifold {
        self.manifold
    }
}

impl Default for Sphere2 {
    fn default() -> Self {
        Self::new()
    }
}

/// The 2-torus T² = S¹ × S¹
///
/// The 2-torus is a 2-dimensional compact manifold, the product of two circles.
/// It can be parameterized by two angular coordinates (φ, ψ) ∈ [0, 2π) × [0, 2π).
#[derive(Clone, Debug)]
pub struct Torus2 {
    manifold: DifferentiableManifold,
}

impl Torus2 {
    /// Create a new 2-torus manifold T²
    ///
    /// The torus is covered by a single chart with angular coordinates.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_manifolds::Torus2;
    ///
    /// let torus = Torus2::new();
    /// assert_eq!(torus.dimension(), 2);
    /// ```
    pub fn new() -> Self {
        Self::with_radii(1.0, 0.5)
    }

    /// Create a torus with specified major and minor radii
    ///
    /// # Arguments
    ///
    /// * `major_radius` - Distance from center to tube center (R)
    /// * `minor_radius` - Radius of the tube (r)
    pub fn with_radii(major_radius: f64, minor_radius: f64) -> Self {
        let mut manifold = DifferentiableManifold::new(
            format!("T²(R={},r={})", major_radius, minor_radius),
            2
        );

        // Standard chart with angular coordinates
        // φ ∈ [0, 2π): angle around major circle
        // ψ ∈ [0, 2π): angle around minor circle
        let chart = Chart::new("angular", 2, vec!["φ", "ψ"])
            .expect("Failed to create chart");
        manifold.add_chart(chart).expect("Failed to add chart");

        Self { manifold }
    }

    /// Get the dimension (always 2)
    pub fn dimension(&self) -> usize {
        2
    }

    /// Get the underlying manifold
    pub fn manifold(&self) -> &DifferentiableManifold {
        &self.manifold
    }

    /// Create a point from angular coordinates
    ///
    /// # Arguments
    ///
    /// * `phi` - Major angle in radians, φ ∈ [0, 2π)
    /// * `psi` - Minor angle in radians, ψ ∈ [0, 2π)
    pub fn point(&self, phi: f64, psi: f64) -> ManifoldPoint {
        ManifoldPoint::new(vec![phi, psi])
    }

    /// Convert a point to 3D Cartesian coordinates in ℝ³
    ///
    /// Using the standard embedding with major radius R and minor radius r:
    /// x = (R + r cos ψ) cos φ
    /// y = (R + r cos ψ) sin φ
    /// z = r sin ψ
    pub fn to_cartesian(&self, point: &ManifoldPoint, major_r: f64, minor_r: f64) -> Result<(f64, f64, f64)> {
        if point.dimension() != 2 {
            return Err(crate::errors::ManifoldError::DimensionMismatch {
                expected: 2,
                actual: point.dimension(),
            });
        }

        let phi = point.coordinate(0)?;
        let psi = point.coordinate(1)?;

        let x = (major_r + minor_r * psi.cos()) * phi.cos();
        let y = (major_r + minor_r * psi.cos()) * phi.sin();
        let z = minor_r * psi.sin();

        Ok((x, y, z))
    }

    /// Standard flat 2-torus (quotient of ℝ² by ℤ²)
    pub fn flat() -> Self {
        Self::with_radii(1.0, 1.0)
    }
}

impl Default for Torus2 {
    fn default() -> Self {
        Self::new()
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

    // Circle tests

    #[test]
    fn test_circle_creation() {
        let circle = Circle::new();
        assert_eq!(circle.dimension(), 1);
        assert_eq!(circle.manifold().name(), "S¹");
    }

    #[test]
    fn test_circle_charts() {
        let circle = Circle::new();
        assert_eq!(circle.manifold().charts().len(), 2);
    }

    #[test]
    fn test_circle_point() {
        let circle = Circle::new();
        let point = circle.point(std::f64::consts::PI / 4.0);
        assert_eq!(point.dimension(), 1);
    }

    #[test]
    fn test_circle_cartesian_conversion() {
        let circle = Circle::new();

        // Point at angle π/4
        let point = circle.point(std::f64::consts::PI / 4.0);
        let (x, y) = circle.to_cartesian(&point).unwrap();

        // Check it's on the unit circle
        assert!((x * x + y * y - 1.0).abs() < 1e-10);

        // Check coordinates are correct
        let expected_x = (std::f64::consts::PI / 4.0).cos();
        let expected_y = (std::f64::consts::PI / 4.0).sin();
        assert!((x - expected_x).abs() < 1e-10);
        assert!((y - expected_y).abs() < 1e-10);
    }

    #[test]
    fn test_circle_from_cartesian() {
        let circle = Circle::new();
        let point = circle.from_cartesian(1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt());

        // Should be at angle π/4
        let theta = point.coordinate(0).unwrap();
        assert!((theta - std::f64::consts::PI / 4.0).abs() < 1e-10);
    }

    // Sphere2 tests

    #[test]
    fn test_sphere2_creation() {
        let sphere = Sphere2::new();
        assert_eq!(sphere.dimension(), 2);
        assert_eq!(sphere.manifold().name(), "S²");
    }

    #[test]
    fn test_sphere2_charts() {
        let sphere = Sphere2::new();
        assert_eq!(sphere.manifold().charts().len(), 2);
    }

    #[test]
    fn test_sphere2_from_spherical() {
        let sphere = Sphere2::new();

        // Equator point at (1, 0, 0)
        let point = sphere.from_spherical(0.0, std::f64::consts::PI / 2.0).unwrap();
        assert_eq!(point.dimension(), 2);

        // Convert to Cartesian
        let (x, y, z) = sphere.to_cartesian(&point).unwrap();

        // Check it's on the unit sphere
        assert!((x * x + y * y + z * z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sphere2_from_cartesian() {
        let sphere = Sphere2::new();

        // Point at north pole vicinity
        let point = sphere.from_cartesian(0.0, 0.0, 1.0).unwrap();

        // Convert back
        let (x, y, z) = sphere.to_cartesian(&point).unwrap();

        // Should be close to original (allowing for numerical errors and chart choice)
        let dist = ((x - 0.0).powi(2) + (y - 0.0).powi(2) + (z - 1.0).powi(2)).sqrt();
        assert!(dist < 1e-6);
    }

    #[test]
    fn test_sphere2_cartesian_roundtrip() {
        let sphere = Sphere2::new();

        // Test point
        let point = sphere.from_cartesian(1.0, 1.0, 1.0).unwrap();
        let (x, y, z) = sphere.to_cartesian(&point).unwrap();

        // Normalize the original point
        let r = (1.0_f64 + 1.0 + 1.0).sqrt();
        let (nx, ny, nz) = (1.0 / r, 1.0 / r, 1.0 / r);

        // Check roundtrip
        assert!((x - nx).abs() < 1e-10);
        assert!((y - ny).abs() < 1e-10);
        assert!((z - nz).abs() < 1e-10);
    }

    // Torus2 tests

    #[test]
    fn test_torus2_creation() {
        let torus = Torus2::new();
        assert_eq!(torus.dimension(), 2);
        assert!(torus.manifold().name().starts_with("T²"));
    }

    #[test]
    fn test_torus2_chart() {
        let torus = Torus2::new();
        assert_eq!(torus.manifold().charts().len(), 1);
    }

    #[test]
    fn test_torus2_point() {
        let torus = Torus2::new();
        let point = torus.point(0.0, 0.0);
        assert_eq!(point.dimension(), 2);
    }

    #[test]
    fn test_torus2_to_cartesian() {
        let torus = Torus2::with_radii(2.0, 1.0);
        let point = torus.point(0.0, 0.0);

        let (x, y, z) = torus.to_cartesian(&point, 2.0, 1.0).unwrap();

        // At φ=0, ψ=0: point is at (R+r, 0, 0) = (3, 0, 0)
        assert!((x - 3.0).abs() < 1e-10);
        assert!(y.abs() < 1e-10);
        assert!(z.abs() < 1e-10);
    }

    #[test]
    fn test_torus2_flat() {
        let torus = Torus2::flat();
        assert_eq!(torus.dimension(), 2);
    }

    #[test]
    fn test_torus2_with_radii() {
        let torus = Torus2::with_radii(3.0, 1.5);
        assert_eq!(torus.dimension(), 2);

        // Check that the Cartesian conversion uses the correct radii
        let point = torus.point(std::f64::consts::PI / 2.0, 0.0);
        let (x, y, z) = torus.to_cartesian(&point, 3.0, 1.5).unwrap();

        // At φ=π/2, ψ=0: point is at (0, R+r, 0) = (0, 4.5, 0)
        assert!(x.abs() < 1e-10);
        assert!((y - 4.5).abs() < 1e-10);
        assert!(z.abs() < 1e-10);
    }
}
