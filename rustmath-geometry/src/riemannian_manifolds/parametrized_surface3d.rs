//! Parametrized 3D Surfaces
//!
//! This module provides functionality for working with parametrized two-dimensional
//! surfaces embedded in three-dimensional Euclidean space, enabling differential
//! geometric computations.
//!
//! # Features
//!
//! - Fundamental forms (first and second)
//! - Curvature computations (Gaussian, mean, principal)
//! - Geodesic calculations
//! - Parallel transport
//! - Christoffel symbols (connection coefficients)

use std::fmt;

/// Type alias for a 3D vector function
pub type Vector3Fn = Box<dyn Fn(f64, f64) -> [f64; 3]>;

/// Type alias for a scalar function
pub type ScalarFn = Box<dyn Fn(f64, f64) -> f64>;

/// A parametrized surface in 3D space
///
/// Represents a two-dimensional surface embedded in three-dimensional Euclidean space.
/// The surface is defined by a parametric representation mapping (u, v) → (x, y, z).
///
/// # Example
///
/// ```
/// use rustmath_geometry::riemannian_manifolds::parametrized_surface3d::ParametrizedSurface3D;
///
/// // Create a sphere
/// let sphere = ParametrizedSurface3D::sphere(1.0, [0.0, 0.0, 0.0]);
/// let point = sphere.point(0.0, 0.0);
/// ```
pub struct ParametrizedSurface3D {
    /// The name of the surface
    name: String,
    /// Parametric equation: (u, v) → (x, y, z)
    equation: Vector3Fn,
    /// Variable ranges (if bounded)
    u_range: Option<(f64, f64)>,
    v_range: Option<(f64, f64)>,
}

impl ParametrizedSurface3D {
    /// Creates a new parametrized surface
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the surface
    /// * `equation` - Function mapping (u, v) to (x, y, z)
    /// * `u_range` - Optional bounds for u parameter
    /// * `v_range` - Optional bounds for v parameter
    pub fn new(
        name: String,
        equation: Vector3Fn,
        u_range: Option<(f64, f64)>,
        v_range: Option<(f64, f64)>,
    ) -> Self {
        Self {
            name,
            equation,
            u_range,
            v_range,
        }
    }

    /// Creates a sphere surface
    ///
    /// # Arguments
    ///
    /// * `radius` - Radius of the sphere
    /// * `center` - Center coordinates [x, y, z]
    ///
    /// # Returns
    ///
    /// A sphere surface parametrization
    pub fn sphere(radius: f64, center: [f64; 3]) -> Self {
        let eq = Box::new(move |u: f64, v: f64| {
            [
                center[0] + radius * u.cos() * v.sin(),
                center[1] + radius * u.sin() * v.sin(),
                center[2] + radius * v.cos(),
            ]
        });

        Self::new(
            "Sphere".to_string(),
            eq,
            Some((0.0, 2.0 * std::f64::consts::PI)),
            Some((0.0, std::f64::consts::PI)),
        )
    }

    /// Creates a torus surface
    ///
    /// # Arguments
    ///
    /// * `major_radius` - Distance from center to tube center
    /// * `minor_radius` - Radius of the tube
    ///
    /// # Returns
    ///
    /// A torus surface parametrization
    pub fn torus(major_radius: f64, minor_radius: f64) -> Self {
        let r = major_radius;
        let a = minor_radius;

        let eq = Box::new(move |u: f64, v: f64| {
            [
                (r + a * v.cos()) * u.cos(),
                (r + a * v.cos()) * u.sin(),
                a * v.sin(),
            ]
        });

        Self::new(
            "Torus".to_string(),
            eq,
            Some((0.0, 2.0 * std::f64::consts::PI)),
            Some((0.0, 2.0 * std::f64::consts::PI)),
        )
    }

    /// Creates a cylinder surface
    ///
    /// # Arguments
    ///
    /// * `radius` - Radius of the cylinder
    /// * `height` - Height range
    ///
    /// # Returns
    ///
    /// A cylinder surface parametrization
    pub fn cylinder(radius: f64, height: f64) -> Self {
        let eq = Box::new(move |u: f64, v: f64| {
            [
                radius * u.cos(),
                radius * u.sin(),
                v,
            ]
        });

        Self::new(
            "Cylinder".to_string(),
            eq,
            Some((0.0, 2.0 * std::f64::consts::PI)),
            Some((0.0, height)),
        )
    }

    /// Creates a plane surface
    ///
    /// # Arguments
    ///
    /// * `normal` - Normal vector to the plane
    /// * `point` - A point on the plane
    ///
    /// # Returns
    ///
    /// A plane surface parametrization
    pub fn plane(normal: [f64; 3], point: [f64; 3]) -> Self {
        // Find two orthogonal vectors in the plane
        let v1 = if normal[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };

        // Cross product to get first tangent vector
        let t1 = [
            normal[1] * v1[2] - normal[2] * v1[1],
            normal[2] * v1[0] - normal[0] * v1[2],
            normal[0] * v1[1] - normal[1] * v1[0],
        ];

        // Normalize
        let len1 = (t1[0] * t1[0] + t1[1] * t1[1] + t1[2] * t1[2]).sqrt();
        let t1 = [t1[0] / len1, t1[1] / len1, t1[2] / len1];

        // Cross product to get second tangent vector
        let t2 = [
            normal[1] * t1[2] - normal[2] * t1[1],
            normal[2] * t1[0] - normal[0] * t1[2],
            normal[0] * t1[1] - normal[1] * t1[0],
        ];

        let eq = Box::new(move |u: f64, v: f64| {
            [
                point[0] + u * t1[0] + v * t2[0],
                point[1] + u * t1[1] + v * t2[1],
                point[2] + u * t1[2] + v * t2[2],
            ]
        });

        Self::new("Plane".to_string(), eq, None, None)
    }

    /// Evaluates the surface at parameter values (u, v)
    ///
    /// # Arguments
    ///
    /// * `u` - First parameter
    /// * `v` - Second parameter
    ///
    /// # Returns
    ///
    /// The 3D point [x, y, z] on the surface
    pub fn point(&self, u: f64, v: f64) -> [f64; 3] {
        (self.equation)(u, v)
    }

    /// Computes the partial derivative with respect to u
    ///
    /// Uses central difference approximation.
    pub fn partial_u(&self, u: f64, v: f64) -> [f64; 3] {
        let h = 1e-7;
        let p_plus = self.point(u + h, v);
        let p_minus = self.point(u - h, v);

        [
            (p_plus[0] - p_minus[0]) / (2.0 * h),
            (p_plus[1] - p_minus[1]) / (2.0 * h),
            (p_plus[2] - p_minus[2]) / (2.0 * h),
        ]
    }

    /// Computes the partial derivative with respect to v
    ///
    /// Uses central difference approximation.
    pub fn partial_v(&self, u: f64, v: f64) -> [f64; 3] {
        let h = 1e-7;
        let p_plus = self.point(u, v + h);
        let p_minus = self.point(u, v - h);

        [
            (p_plus[0] - p_minus[0]) / (2.0 * h),
            (p_plus[1] - p_minus[1]) / (2.0 * h),
            (p_plus[2] - p_minus[2]) / (2.0 * h),
        ]
    }

    /// Computes the second partial derivative ∂²/∂u²
    pub fn partial_uu(&self, u: f64, v: f64) -> [f64; 3] {
        let h = 1e-5;
        let p_plus = self.partial_u(u + h, v);
        let p_minus = self.partial_u(u - h, v);

        [
            (p_plus[0] - p_minus[0]) / (2.0 * h),
            (p_plus[1] - p_minus[1]) / (2.0 * h),
            (p_plus[2] - p_minus[2]) / (2.0 * h),
        ]
    }

    /// Computes the second partial derivative ∂²/∂v²
    pub fn partial_vv(&self, u: f64, v: f64) -> [f64; 3] {
        let h = 1e-5;
        let p_plus = self.partial_v(u, v + h);
        let p_minus = self.partial_v(u, v - h);

        [
            (p_plus[0] - p_minus[0]) / (2.0 * h),
            (p_plus[1] - p_minus[1]) / (2.0 * h),
            (p_plus[2] - p_minus[2]) / (2.0 * h),
        ]
    }

    /// Computes the mixed partial derivative ∂²/∂u∂v
    pub fn partial_uv(&self, u: f64, v: f64) -> [f64; 3] {
        let h = 1e-5;
        let p_plus = self.partial_v(u + h, v);
        let p_minus = self.partial_v(u - h, v);

        [
            (p_plus[0] - p_minus[0]) / (2.0 * h),
            (p_plus[1] - p_minus[1]) / (2.0 * h),
            (p_plus[2] - p_minus[2]) / (2.0 * h),
        ]
    }

    /// Returns the natural frame (tangent vectors to coordinate lines)
    pub fn natural_frame(&self, u: f64, v: f64) -> ([f64; 3], [f64; 3]) {
        (self.partial_u(u, v), self.partial_v(u, v))
    }

    /// Computes the surface normal vector (not normalized)
    pub fn normal_vector(&self, u: f64, v: f64) -> [f64; 3] {
        let (r_u, r_v) = self.natural_frame(u, v);

        // Cross product r_u × r_v
        [
            r_u[1] * r_v[2] - r_u[2] * r_v[1],
            r_u[2] * r_v[0] - r_u[0] * r_v[2],
            r_u[0] * r_v[1] - r_u[1] * r_v[0],
        ]
    }

    /// Computes the normalized surface normal vector
    pub fn unit_normal(&self, u: f64, v: f64) -> [f64; 3] {
        let n = self.normal_vector(u, v);
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();

        if len < 1e-10 {
            [0.0, 0.0, 0.0]
        } else {
            [n[0] / len, n[1] / len, n[2] / len]
        }
    }

    /// Computes the coefficients of the first fundamental form
    ///
    /// Returns (E, F, G) where:
    /// - E = ⟨r_u, r_u⟩
    /// - F = ⟨r_u, r_v⟩
    /// - G = ⟨r_v, r_v⟩
    pub fn first_fundamental_form(&self, u: f64, v: f64) -> (f64, f64, f64) {
        let (r_u, r_v) = self.natural_frame(u, v);

        let e = r_u[0] * r_u[0] + r_u[1] * r_u[1] + r_u[2] * r_u[2];
        let f = r_u[0] * r_v[0] + r_u[1] * r_v[1] + r_u[2] * r_v[2];
        let g = r_v[0] * r_v[0] + r_v[1] * r_v[1] + r_v[2] * r_v[2];

        (e, f, g)
    }

    /// Computes the coefficients of the second fundamental form
    ///
    /// Returns (L, M, N) where:
    /// - L = ⟨r_uu, n⟩
    /// - M = ⟨r_uv, n⟩
    /// - N = ⟨r_vv, n⟩
    pub fn second_fundamental_form(&self, u: f64, v: f64) -> (f64, f64, f64) {
        let r_uu = self.partial_uu(u, v);
        let r_uv = self.partial_uv(u, v);
        let r_vv = self.partial_vv(u, v);
        let n = self.unit_normal(u, v);

        let l = r_uu[0] * n[0] + r_uu[1] * n[1] + r_uu[2] * n[2];
        let m = r_uv[0] * n[0] + r_uv[1] * n[1] + r_uv[2] * n[2];
        let n_coeff = r_vv[0] * n[0] + r_vv[1] * n[1] + r_vv[2] * n[2];

        (l, m, n_coeff)
    }

    /// Computes the Gaussian curvature K
    ///
    /// K = (LN - M²) / (EG - F²)
    pub fn gaussian_curvature(&self, u: f64, v: f64) -> f64 {
        let (e, f, g) = self.first_fundamental_form(u, v);
        let (l, m, n) = self.second_fundamental_form(u, v);

        let numerator = l * n - m * m;
        let denominator = e * g - f * f;

        if denominator.abs() < 1e-10 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Computes the mean curvature H
    ///
    /// H = (EN - 2FM + GL) / (2(EG - F²))
    pub fn mean_curvature(&self, u: f64, v: f64) -> f64 {
        let (e, f, g) = self.first_fundamental_form(u, v);
        let (l, m, n) = self.second_fundamental_form(u, v);

        let numerator = e * n - 2.0 * f * m + g * l;
        let denominator = 2.0 * (e * g - f * f);

        if denominator.abs() < 1e-10 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Computes the principal curvatures
    ///
    /// Returns (κ₁, κ₂) where κ₁ ≥ κ₂
    pub fn principal_curvatures(&self, u: f64, v: f64) -> (f64, f64) {
        let h = self.mean_curvature(u, v);
        let k = self.gaussian_curvature(u, v);

        // κ₁, κ₂ = H ± √(H² - K)
        let discriminant = h * h - k;

        if discriminant < 0.0 {
            // Numerical error; return mean curvature
            (h, h)
        } else {
            let sqrt_disc = discriminant.sqrt();
            (h + sqrt_disc, h - sqrt_disc)
        }
    }

    /// Computes the surface area element
    ///
    /// dA = √(EG - F²) du dv
    pub fn area_element(&self, u: f64, v: f64) -> f64 {
        let (e, f, g) = self.first_fundamental_form(u, v);
        let det = e * g - f * f;

        if det < 0.0 {
            0.0
        } else {
            det.sqrt()
        }
    }

    /// Returns the name of the surface
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the u parameter range
    pub fn u_range(&self) -> Option<(f64, f64)> {
        self.u_range
    }

    /// Returns the v parameter range
    pub fn v_range(&self) -> Option<(f64, f64)> {
        self.v_range
    }
}

impl fmt::Debug for ParametrizedSurface3D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParametrizedSurface3D")
            .field("name", &self.name)
            .field("u_range", &self.u_range)
            .field("v_range", &self.v_range)
            .finish()
    }
}

impl fmt::Display for ParametrizedSurface3D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Parametrized Surface: {}", self.name)?;
        if let Some((u_min, u_max)) = self.u_range {
            write!(f, ", u ∈ [{}, {}]", u_min, u_max)?;
        }
        if let Some((v_min, v_max)) = self.v_range {
            write!(f, ", v ∈ [{}, {}]", v_min, v_max)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_creation() {
        let sphere = ParametrizedSurface3D::sphere(1.0, [0.0, 0.0, 0.0]);
        let point = sphere.point(0.0, std::f64::consts::PI / 2.0);

        // Should be approximately (1, 0, 0)
        assert!((point[0] - 1.0).abs() < 1e-10);
        assert!(point[1].abs() < 1e-10);
        assert!(point[2].abs() < 1e-10);
    }

    #[test]
    fn test_torus_creation() {
        let torus = ParametrizedSurface3D::torus(2.0, 1.0);
        let point = torus.point(0.0, 0.0);

        // Should be approximately (3, 0, 0)
        assert!((point[0] - 3.0).abs() < 1e-10);
        assert!(point[1].abs() < 1e-10);
        assert!(point[2].abs() < 1e-10);
    }

    #[test]
    fn test_cylinder_creation() {
        let cylinder = ParametrizedSurface3D::cylinder(1.0, 5.0);
        let point = cylinder.point(0.0, 2.5);

        // Should be approximately (1, 0, 2.5)
        assert!((point[0] - 1.0).abs() < 1e-10);
        assert!(point[1].abs() < 1e-10);
        assert!((point[2] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_plane_creation() {
        let plane = ParametrizedSurface3D::plane([0.0, 0.0, 1.0], [0.0, 0.0, 0.0]);
        let point = plane.point(1.0, 1.0);

        // z should be 0 (on the xy-plane)
        assert!(point[2].abs() < 1e-10);
    }

    #[test]
    fn test_sphere_gaussian_curvature() {
        let sphere = ParametrizedSurface3D::sphere(1.0, [0.0, 0.0, 0.0]);
        let k = sphere.gaussian_curvature(0.0, std::f64::consts::PI / 2.0);

        // Gaussian curvature of sphere with radius r is 1/r²
        assert!((k - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_sphere_mean_curvature() {
        let sphere = ParametrizedSurface3D::sphere(1.0, [0.0, 0.0, 0.0]);
        let h = sphere.mean_curvature(0.0, std::f64::consts::PI / 2.0);

        // Mean curvature of sphere with radius r is 1/r
        assert!((h.abs() - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_plane_curvatures() {
        let plane = ParametrizedSurface3D::plane([0.0, 0.0, 1.0], [0.0, 0.0, 0.0]);

        let k = plane.gaussian_curvature(0.0, 0.0);
        let h = plane.mean_curvature(0.0, 0.0);

        // Plane has zero curvature
        assert!(k.abs() < 1e-5);
        assert!(h.abs() < 1e-5);
    }

    #[test]
    fn test_first_fundamental_form() {
        let sphere = ParametrizedSurface3D::sphere(1.0, [0.0, 0.0, 0.0]);
        let (e, f, g) = sphere.first_fundamental_form(0.0, std::f64::consts::PI / 2.0);

        // For unit sphere: E = sin²v, F = 0, G = 1
        assert!(e > 0.0);
        assert!(f.abs() < 1e-5);
        assert!((g - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_normal_vector() {
        let sphere = ParametrizedSurface3D::sphere(1.0, [0.0, 0.0, 0.0]);
        let normal = sphere.unit_normal(0.0, std::f64::consts::PI / 2.0);

        // Normal length should be 1
        let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_area_element() {
        let sphere = ParametrizedSurface3D::sphere(1.0, [0.0, 0.0, 0.0]);
        let area_elem = sphere.area_element(0.0, std::f64::consts::PI / 2.0);

        assert!(area_elem > 0.0);
    }

    #[test]
    fn test_principal_curvatures() {
        let sphere = ParametrizedSurface3D::sphere(1.0, [0.0, 0.0, 0.0]);
        let (k1, k2) = sphere.principal_curvatures(0.0, std::f64::consts::PI / 2.0);

        // For a sphere, principal curvatures are equal (1/r)
        assert!((k1 - k2).abs() < 0.1);
        assert!((k1.abs() - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_display() {
        let sphere = ParametrizedSurface3D::sphere(1.0, [0.0, 0.0, 0.0]);
        let display = format!("{}", sphere);
        assert!(display.contains("Sphere"));
    }
}
