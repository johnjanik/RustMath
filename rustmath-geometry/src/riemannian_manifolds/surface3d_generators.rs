//! Surface Generators
//!
//! This module provides convenient generators for common parametrized surfaces
//! in three-dimensional space.

use super::parametrized_surface3d::ParametrizedSurface3D;
use std::f64::consts::PI;

/// Collection of surface generators
///
/// Provides static methods to create well-known geometric surfaces used in
/// differential geometry and visualization.
///
/// # Example
///
/// ```
/// use rustmath_geometry::riemannian_manifolds::surface3d_generators::SurfaceGenerators;
///
/// let sphere = SurfaceGenerators::sphere(1.0, [0.0, 0.0, 0.0]);
/// let torus = SurfaceGenerators::torus(2.0, 1.0);
/// ```
pub struct SurfaceGenerators;

impl SurfaceGenerators {
    /// Creates a sphere surface
    ///
    /// # Arguments
    ///
    /// * `radius` - Radius of the sphere
    /// * `center` - Center coordinates [x, y, z]
    ///
    /// # Returns
    ///
    /// A sphere surface with the specified radius and center
    pub fn sphere(radius: f64, center: [f64; 3]) -> ParametrizedSurface3D {
        ParametrizedSurface3D::sphere(radius, center)
    }

    /// Creates a torus surface
    ///
    /// # Arguments
    ///
    /// * `major_radius` - Distance from center to tube center (R)
    /// * `minor_radius` - Radius of the tube (r)
    ///
    /// # Returns
    ///
    /// A torus surface (donut shape)
    pub fn torus(major_radius: f64, minor_radius: f64) -> ParametrizedSurface3D {
        ParametrizedSurface3D::torus(major_radius, minor_radius)
    }

    /// Creates a catenoid surface (minimal surface)
    ///
    /// A catenoid is a minimal surface formed by rotating a catenary curve.
    ///
    /// # Arguments
    ///
    /// * `scale` - Scaling parameter
    ///
    /// # Returns
    ///
    /// A catenoid surface
    pub fn catenoid(scale: f64) -> ParametrizedSurface3D {
        let eq = Box::new(move |u: f64, v: f64| {
            [
                scale * v.cosh() * u.cos(),
                scale * v.cosh() * u.sin(),
                scale * v,
            ]
        });

        ParametrizedSurface3D::new(
            "Catenoid".to_string(),
            eq,
            Some((0.0, 2.0 * PI)),
            Some((-2.0, 2.0)),
        )
    }

    /// Creates a helicoid surface (minimal surface)
    ///
    /// A helicoid is a minimal surface that looks like a spiral ramp.
    ///
    /// # Arguments
    ///
    /// * `pitch` - Pitch of the helix
    ///
    /// # Returns
    ///
    /// A helicoid surface
    pub fn helicoid(pitch: f64) -> ParametrizedSurface3D {
        let eq = Box::new(move |u: f64, v: f64| {
            [
                v * u.cos(),
                v * u.sin(),
                pitch * u,
            ]
        });

        ParametrizedSurface3D::new(
            "Helicoid".to_string(),
            eq,
            Some((0.0, 2.0 * PI)),
            Some((-2.0, 2.0)),
        )
    }

    /// Creates Enneper's surface (minimal surface)
    ///
    /// Enneper's surface is a minimal surface with self-intersections.
    ///
    /// # Returns
    ///
    /// An Enneper surface
    pub fn enneper() -> ParametrizedSurface3D {
        let eq = Box::new(|u: f64, v: f64| {
            [
                u - u.powi(3) / 3.0 + u * v.powi(2),
                v - v.powi(3) / 3.0 + v * u.powi(2),
                u.powi(2) - v.powi(2),
            ]
        });

        ParametrizedSurface3D::new(
            "Enneper's Surface".to_string(),
            eq,
            Some((-2.0, 2.0)),
            Some((-2.0, 2.0)),
        )
    }

    /// Creates an ellipsoid surface
    ///
    /// # Arguments
    ///
    /// * `a` - Semi-axis along x
    /// * `b` - Semi-axis along y
    /// * `c` - Semi-axis along z
    ///
    /// # Returns
    ///
    /// An ellipsoid surface
    pub fn ellipsoid(a: f64, b: f64, c: f64) -> ParametrizedSurface3D {
        let eq = Box::new(move |u: f64, v: f64| {
            [
                a * u.cos() * v.sin(),
                b * u.sin() * v.sin(),
                c * v.cos(),
            ]
        });

        ParametrizedSurface3D::new(
            "Ellipsoid".to_string(),
            eq,
            Some((0.0, 2.0 * PI)),
            Some((0.0, PI)),
        )
    }

    /// Creates an elliptic paraboloid surface
    ///
    /// # Arguments
    ///
    /// * `a` - Scale factor along x
    /// * `b` - Scale factor along y
    ///
    /// # Returns
    ///
    /// An elliptic paraboloid (bowl shape)
    pub fn elliptic_paraboloid(a: f64, b: f64) -> ParametrizedSurface3D {
        let eq = Box::new(move |u: f64, v: f64| {
            [
                u,
                v,
                (u / a).powi(2) + (v / b).powi(2),
            ]
        });

        ParametrizedSurface3D::new(
            "Elliptic Paraboloid".to_string(),
            eq,
            Some((-2.0, 2.0)),
            Some((-2.0, 2.0)),
        )
    }

    /// Creates a hyperbolic paraboloid surface (saddle)
    ///
    /// # Arguments
    ///
    /// * `a` - Scale factor along x
    /// * `b` - Scale factor along y
    ///
    /// # Returns
    ///
    /// A hyperbolic paraboloid (saddle shape)
    pub fn hyperbolic_paraboloid(a: f64, b: f64) -> ParametrizedSurface3D {
        let eq = Box::new(move |u: f64, v: f64| {
            [
                u,
                v,
                (u / a).powi(2) - (v / b).powi(2),
            ]
        });

        ParametrizedSurface3D::new(
            "Hyperbolic Paraboloid".to_string(),
            eq,
            Some((-2.0, 2.0)),
            Some((-2.0, 2.0)),
        )
    }

    /// Creates a monkey saddle surface
    ///
    /// A monkey saddle is a cubic surface with three saddle directions.
    ///
    /// # Returns
    ///
    /// A monkey saddle surface
    pub fn monkey_saddle() -> ParametrizedSurface3D {
        let eq = Box::new(|u: f64, v: f64| {
            [
                u,
                v,
                u.powi(3) - 3.0 * u * v.powi(2),
            ]
        });

        ParametrizedSurface3D::new(
            "Monkey Saddle".to_string(),
            eq,
            Some((-2.0, 2.0)),
            Some((-2.0, 2.0)),
        )
    }

    /// Creates Dini's surface
    ///
    /// Dini's surface is a surface of constant negative curvature.
    ///
    /// # Arguments
    ///
    /// * `a` - Scale parameter
    /// * `b` - Shape parameter
    ///
    /// # Returns
    ///
    /// Dini's surface
    pub fn dini_surface(a: f64, b: f64) -> ParametrizedSurface3D {
        let eq = Box::new(move |u: f64, v: f64| {
            [
                a * u.cos() * v.sin(),
                a * u.sin() * v.sin(),
                a * (v.cos() + (v.tan() / b).ln()) + b * u,
            ]
        });

        ParametrizedSurface3D::new(
            "Dini's Surface".to_string(),
            eq,
            Some((0.0, 4.0 * PI)),
            Some((0.1, 2.0)),
        )
    }

    /// Creates a Klein bottle (immersed in 3D)
    ///
    /// The Klein bottle is a non-orientable surface.
    ///
    /// # Arguments
    ///
    /// * `radius` - Scaling radius
    ///
    /// # Returns
    ///
    /// A Klein bottle surface (immersed in 3D)
    pub fn klein_bottle(radius: f64) -> ParametrizedSurface3D {
        let r = radius;

        let eq = Box::new(move |u: f64, v: f64| {
            let x = if u < PI {
                (6.0 + (2.0 + (u / 2.0).cos()) * v.sin()) * u.cos()
            } else {
                (6.0 + (2.0 - (u / 2.0).cos()) * v.sin()) * u.cos()
            };

            let y = if u < PI {
                (6.0 + (2.0 + (u / 2.0).cos()) * v.sin()) * u.sin()
            } else {
                (6.0 + (2.0 - (u / 2.0).cos()) * v.sin()) * u.sin()
            };

            let z = if u < PI {
                (u / 2.0).sin() * v.sin()
            } else {
                (u / 2.0).sin() * v.sin()
            };

            [x * r / 6.0, y * r / 6.0, z * r / 6.0]
        });

        ParametrizedSurface3D::new(
            "Klein Bottle".to_string(),
            eq,
            Some((0.0, 2.0 * PI)),
            Some((0.0, 2.0 * PI)),
        )
    }

    /// Creates a Möbius strip
    ///
    /// # Arguments
    ///
    /// * `radius` - Radius of the strip
    /// * `width` - Width of the strip
    ///
    /// # Returns
    ///
    /// A Möbius strip surface
    pub fn mobius_strip(radius: f64, width: f64) -> ParametrizedSurface3D {
        let eq = Box::new(move |u: f64, v: f64| {
            [
                (radius + v * (u / 2.0).cos()) * u.cos(),
                (radius + v * (u / 2.0).cos()) * u.sin(),
                v * (u / 2.0).sin(),
            ]
        });

        ParametrizedSurface3D::new(
            "Möbius Strip".to_string(),
            eq,
            Some((0.0, 2.0 * PI)),
            Some((-width, width)),
        )
    }

    /// Creates Whitney's umbrella (cross-cap)
    ///
    /// A singular non-orientable surface.
    ///
    /// # Returns
    ///
    /// Whitney's umbrella surface
    pub fn whitney_umbrella() -> ParametrizedSurface3D {
        let eq = Box::new(|u: f64, v: f64| {
            [
                u * v,
                v,
                u * u,
            ]
        });

        ParametrizedSurface3D::new(
            "Whitney's Umbrella".to_string(),
            eq,
            Some((-2.0, 2.0)),
            Some((-2.0, 2.0)),
        )
    }

    /// Creates a cone surface
    ///
    /// # Arguments
    ///
    /// * `height` - Height of the cone
    /// * `radius` - Base radius of the cone
    ///
    /// # Returns
    ///
    /// A cone surface
    pub fn cone(height: f64, radius: f64) -> ParametrizedSurface3D {
        let eq = Box::new(move |u: f64, v: f64| {
            [
                v * radius * u.cos() / height,
                v * radius * u.sin() / height,
                v,
            ]
        });

        ParametrizedSurface3D::new(
            "Cone".to_string(),
            eq,
            Some((0.0, 2.0 * PI)),
            Some((0.0, height)),
        )
    }

    /// Creates a hyperboloid of one sheet
    ///
    /// # Arguments
    ///
    /// * `a` - Scale along x
    /// * `b` - Scale along y
    /// * `c` - Scale along z
    ///
    /// # Returns
    ///
    /// A hyperboloid of one sheet
    pub fn hyperboloid_one_sheet(a: f64, b: f64, c: f64) -> ParametrizedSurface3D {
        let eq = Box::new(move |u: f64, v: f64| {
            [
                a * v.cosh() * u.cos(),
                b * v.cosh() * u.sin(),
                c * v.sinh(),
            ]
        });

        ParametrizedSurface3D::new(
            "Hyperboloid of One Sheet".to_string(),
            eq,
            Some((0.0, 2.0 * PI)),
            Some((-2.0, 2.0)),
        )
    }

    /// Creates a hyperboloid of two sheets
    ///
    /// # Arguments
    ///
    /// * `a` - Scale along x
    /// * `b` - Scale along y
    /// * `c` - Scale along z
    ///
    /// # Returns
    ///
    /// A hyperboloid of two sheets (upper sheet only)
    pub fn hyperboloid_two_sheets(a: f64, b: f64, c: f64) -> ParametrizedSurface3D {
        let eq = Box::new(move |u: f64, v: f64| {
            [
                a * v.sinh() * u.cos(),
                b * v.sinh() * u.sin(),
                c * v.cosh(),
            ]
        });

        ParametrizedSurface3D::new(
            "Hyperboloid of Two Sheets".to_string(),
            eq,
            Some((0.0, 2.0 * PI)),
            Some((0.0, 2.0)),
        )
    }

    /// Creates a cylinder
    ///
    /// # Arguments
    ///
    /// * `radius` - Radius of the cylinder
    /// * `height` - Height range
    ///
    /// # Returns
    ///
    /// A cylinder surface
    pub fn cylinder(radius: f64, height: f64) -> ParametrizedSurface3D {
        ParametrizedSurface3D::cylinder(radius, height)
    }

    /// Creates a plane
    ///
    /// # Arguments
    ///
    /// * `normal` - Normal vector to the plane
    /// * `point` - A point on the plane
    ///
    /// # Returns
    ///
    /// A plane surface
    pub fn plane(normal: [f64; 3], point: [f64; 3]) -> ParametrizedSurface3D {
        ParametrizedSurface3D::plane(normal, point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_generator() {
        let sphere = SurfaceGenerators::sphere(1.0, [0.0, 0.0, 0.0]);
        assert_eq!(sphere.name(), "Sphere");
        let point = sphere.point(0.0, PI / 2.0);
        assert!((point[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_torus_generator() {
        let torus = SurfaceGenerators::torus(2.0, 1.0);
        assert_eq!(torus.name(), "Torus");
        let point = torus.point(0.0, 0.0);
        assert!((point[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_catenoid_generator() {
        let catenoid = SurfaceGenerators::catenoid(1.0);
        assert_eq!(catenoid.name(), "Catenoid");
        let point = catenoid.point(0.0, 0.0);
        assert!((point[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_helicoid_generator() {
        let helicoid = SurfaceGenerators::helicoid(1.0);
        assert_eq!(helicoid.name(), "Helicoid");
    }

    #[test]
    fn test_enneper_generator() {
        let enneper = SurfaceGenerators::enneper();
        assert_eq!(enneper.name(), "Enneper's Surface");
        let point = enneper.point(0.0, 0.0);
        assert!(point[0].abs() < 1e-10);
        assert!(point[1].abs() < 1e-10);
        assert!(point[2].abs() < 1e-10);
    }

    #[test]
    fn test_ellipsoid_generator() {
        let ellipsoid = SurfaceGenerators::ellipsoid(2.0, 3.0, 1.0);
        assert_eq!(ellipsoid.name(), "Ellipsoid");
    }

    #[test]
    fn test_elliptic_paraboloid_generator() {
        let paraboloid = SurfaceGenerators::elliptic_paraboloid(1.0, 1.0);
        assert_eq!(paraboloid.name(), "Elliptic Paraboloid");
        let point = paraboloid.point(0.0, 0.0);
        assert!(point[2].abs() < 1e-10); // z = 0 at origin
    }

    #[test]
    fn test_hyperbolic_paraboloid_generator() {
        let paraboloid = SurfaceGenerators::hyperbolic_paraboloid(1.0, 1.0);
        assert_eq!(paraboloid.name(), "Hyperbolic Paraboloid");
        let point = paraboloid.point(0.0, 0.0);
        assert!(point[2].abs() < 1e-10); // z = 0 at origin
    }

    #[test]
    fn test_monkey_saddle_generator() {
        let saddle = SurfaceGenerators::monkey_saddle();
        assert_eq!(saddle.name(), "Monkey Saddle");
        let point = saddle.point(0.0, 0.0);
        assert!(point[2].abs() < 1e-10); // z = 0 at origin
    }

    #[test]
    fn test_dini_surface_generator() {
        let dini = SurfaceGenerators::dini_surface(1.0, 0.5);
        assert_eq!(dini.name(), "Dini's Surface");
    }

    #[test]
    fn test_klein_bottle_generator() {
        let klein = SurfaceGenerators::klein_bottle(1.0);
        assert_eq!(klein.name(), "Klein Bottle");
    }

    #[test]
    fn test_mobius_strip_generator() {
        let mobius = SurfaceGenerators::mobius_strip(2.0, 0.5);
        assert_eq!(mobius.name(), "Möbius Strip");
    }

    #[test]
    fn test_whitney_umbrella_generator() {
        let umbrella = SurfaceGenerators::whitney_umbrella();
        assert_eq!(umbrella.name(), "Whitney's Umbrella");
        let point = umbrella.point(0.0, 0.0);
        assert!(point[0].abs() < 1e-10);
        assert!(point[1].abs() < 1e-10);
        assert!(point[2].abs() < 1e-10);
    }

    #[test]
    fn test_cone_generator() {
        let cone = SurfaceGenerators::cone(5.0, 2.0);
        assert_eq!(cone.name(), "Cone");
        let point = cone.point(0.0, 0.0);
        assert!(point[2].abs() < 1e-10); // z = 0 at apex
    }

    #[test]
    fn test_hyperboloid_one_sheet_generator() {
        let hyperboloid = SurfaceGenerators::hyperboloid_one_sheet(1.0, 1.0, 1.0);
        assert_eq!(hyperboloid.name(), "Hyperboloid of One Sheet");
    }

    #[test]
    fn test_hyperboloid_two_sheets_generator() {
        let hyperboloid = SurfaceGenerators::hyperboloid_two_sheets(1.0, 1.0, 1.0);
        assert_eq!(hyperboloid.name(), "Hyperboloid of Two Sheets");
    }

    #[test]
    fn test_cylinder_generator() {
        let cylinder = SurfaceGenerators::cylinder(1.0, 5.0);
        assert_eq!(cylinder.name(), "Cylinder");
        let point = cylinder.point(0.0, 2.5);
        assert!((point[0] - 1.0).abs() < 1e-10);
        assert!(point[1].abs() < 1e-10);
        assert!((point[2] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_plane_generator() {
        let plane = SurfaceGenerators::plane([0.0, 0.0, 1.0], [0.0, 0.0, 0.0]);
        assert_eq!(plane.name(), "Plane");
        let point = plane.point(1.0, 1.0);
        assert!(point[2].abs() < 1e-10); // z = 0 on xy-plane
    }

    #[test]
    fn test_all_surfaces_have_names() {
        let surfaces = vec![
            SurfaceGenerators::sphere(1.0, [0.0, 0.0, 0.0]),
            SurfaceGenerators::torus(2.0, 1.0),
            SurfaceGenerators::catenoid(1.0),
            SurfaceGenerators::helicoid(1.0),
            SurfaceGenerators::enneper(),
            SurfaceGenerators::ellipsoid(1.0, 1.0, 1.0),
            SurfaceGenerators::monkey_saddle(),
        ];

        for surface in surfaces {
            assert!(!surface.name().is_empty());
        }
    }
}
