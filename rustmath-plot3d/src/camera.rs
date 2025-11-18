//! Camera and lighting system for 3D graphics

use crate::base::{Point3D, Vector3D};
use rustmath_colors::Color;

/// Camera for 3D viewing
///
/// Defines the viewpoint and projection for 3D scenes.
#[derive(Debug, Clone, PartialEq)]
pub struct Camera {
    /// Camera position in world space
    pub position: Point3D,

    /// Point the camera is looking at
    pub look_at: Point3D,

    /// Up vector (typically (0, 0, 1) or (0, 1, 0))
    pub up: Vector3D,

    /// Field of view in radians (for perspective projection)
    pub fov: f64,

    /// Near clipping plane distance
    pub near: f64,

    /// Far clipping plane distance
    pub far: f64,

    /// Projection type
    pub projection: ProjectionType,
}

/// Type of projection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionType {
    /// Perspective projection (objects farther away appear smaller)
    Perspective,

    /// Orthographic projection (parallel lines remain parallel)
    Orthographic,
}

impl Camera {
    /// Create a new camera with default settings
    pub fn new(position: Point3D, look_at: Point3D, up: Vector3D) -> Self {
        Self {
            position,
            look_at,
            up,
            fov: std::f64::consts::PI / 3.0, // 60 degrees
            near: 0.1,
            far: 1000.0,
            projection: ProjectionType::Perspective,
        }
    }

    /// Create a default camera positioned to view the origin
    pub fn default_for_scene(center: Point3D, radius: f64) -> Self {
        // Position camera at a distance based on scene size
        let distance = radius * 2.5;

        // Default camera position: looking down from above at an angle
        let position = Point3D::new(
            center.x + distance * 0.7,
            center.y + distance * 0.7,
            center.z + distance * 0.5,
        );

        Self::new(position, center, Vector3D::unit_z())
    }

    /// Set the field of view (in radians)
    pub fn set_fov(&mut self, fov: f64) {
        self.fov = fov;
    }

    /// Set to perspective projection
    pub fn set_perspective(&mut self) {
        self.projection = ProjectionType::Perspective;
    }

    /// Set to orthographic projection
    pub fn set_orthographic(&mut self) {
        self.projection = ProjectionType::Orthographic;
    }

    /// Get the view direction vector
    pub fn view_direction(&self) -> Vector3D {
        Vector3D::new(
            self.look_at.x - self.position.x,
            self.look_at.y - self.position.y,
            self.look_at.z - self.position.z,
        )
        .normalize()
    }

    /// Get the right vector (perpendicular to view direction and up)
    pub fn right_vector(&self) -> Vector3D {
        self.view_direction().cross(&self.up).normalize()
    }

    /// Get the actual up vector (perpendicular to view and right)
    pub fn up_vector(&self) -> Vector3D {
        let view = self.view_direction();
        let right = self.right_vector();
        right.cross(&view).normalize()
    }

    /// Rotate the camera around the look_at point
    pub fn orbit(&mut self, horizontal_angle: f64, vertical_angle: f64) {
        // Get current position relative to look_at
        let rel_pos = Vector3D::new(
            self.position.x - self.look_at.x,
            self.position.y - self.look_at.y,
            self.position.z - self.look_at.z,
        );

        let distance = rel_pos.magnitude();

        // Convert to spherical coordinates
        let theta = rel_pos.y.atan2(rel_pos.x);
        let phi = (rel_pos.z / distance).acos();

        // Apply rotations
        let new_theta = theta + horizontal_angle;
        let new_phi = (phi + vertical_angle).clamp(0.01, std::f64::consts::PI - 0.01);

        // Convert back to Cartesian
        let new_pos = Vector3D::new(
            distance * new_phi.sin() * new_theta.cos(),
            distance * new_phi.sin() * new_theta.sin(),
            distance * new_phi.cos(),
        );

        self.position = Point3D::new(
            self.look_at.x + new_pos.x,
            self.look_at.y + new_pos.y,
            self.look_at.z + new_pos.z,
        );
    }

    /// Zoom in or out (move camera closer/farther from look_at)
    pub fn zoom(&mut self, factor: f64) {
        let direction = self.view_direction();
        let distance = self.position.distance_to(&self.look_at);
        let new_distance = (distance * factor).max(self.near * 2.0);
        let offset = distance - new_distance; // Offset to move along the direction towards look_at

        self.position = Point3D::new(
            self.position.x + direction.x * offset,
            self.position.y + direction.y * offset,
            self.position.z + direction.z * offset,
        );
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new(
            Point3D::new(5.0, 5.0, 5.0),
            Point3D::origin(),
            Vector3D::unit_z(),
        )
    }
}

/// A light source for 3D scene illumination
#[derive(Debug, Clone, PartialEq)]
pub struct Light {
    /// Type of light
    pub light_type: LightType,

    /// Light color
    pub color: Color,

    /// Light intensity (0.0 to 1.0)
    pub intensity: f64,
}

/// Type of light source
#[derive(Debug, Clone, PartialEq)]
pub enum LightType {
    /// Directional light (like sunlight, parallel rays)
    Directional {
        /// Direction the light is coming from
        direction: Vector3D,
    },

    /// Point light (emits in all directions from a point)
    Point {
        /// Position of the light
        position: Point3D,

        /// Attenuation factors (constant, linear, quadratic)
        attenuation: (f64, f64, f64),
    },

    /// Ambient light (uniform illumination from all directions)
    Ambient,
}

impl Light {
    /// Create a directional light
    pub fn directional(direction: Vector3D, color: Color, intensity: f64) -> Self {
        Self {
            light_type: LightType::Directional {
                direction: direction.normalize(),
            },
            color,
            intensity,
        }
    }

    /// Create a point light
    pub fn point(position: Point3D, color: Color, intensity: f64) -> Self {
        Self {
            light_type: LightType::Point {
                position,
                attenuation: (1.0, 0.0, 0.0), // No attenuation by default
            },
            color,
            intensity,
        }
    }

    /// Create a point light with attenuation
    pub fn point_with_attenuation(
        position: Point3D,
        color: Color,
        intensity: f64,
        attenuation: (f64, f64, f64),
    ) -> Self {
        Self {
            light_type: LightType::Point {
                position,
                attenuation,
            },
            color,
            intensity,
        }
    }

    /// Create an ambient light
    pub fn ambient(color: Color, intensity: f64) -> Self {
        Self {
            light_type: LightType::Ambient,
            color,
            intensity,
        }
    }

    /// Create a default directional light (white, from above and to the side)
    pub fn default_directional() -> Self {
        Self::directional(
            Vector3D::new(-1.0, -1.0, -1.0),
            Color::white(),
            0.8,
        )
    }

    /// Create a default ambient light (soft white)
    pub fn default_ambient() -> Self {
        Self::ambient(Color::white(), 0.2)
    }

    /// Compute illumination at a point with a given normal
    ///
    /// Returns the light contribution (RGB) for this light source.
    pub fn illuminate(&self, point: &Point3D, normal: &Vector3D) -> (f64, f64, f64) {
        match &self.light_type {
            LightType::Directional { direction } => {
                // Lambertian shading (diffuse)
                let light_dir = direction.scale(-1.0).normalize();
                let diffuse = normal.dot(&light_dir).max(0.0);

                let r = self.color.red() * self.intensity * diffuse;
                let g = self.color.green() * self.intensity * diffuse;
                let b = self.color.blue() * self.intensity * diffuse;

                (r, g, b)
            }

            LightType::Point {
                position,
                attenuation,
            } => {
                // Direction from point to light
                let light_vec = Vector3D::new(
                    position.x - point.x,
                    position.y - point.y,
                    position.z - point.z,
                );
                let distance = light_vec.magnitude();
                let light_dir = light_vec.scale(1.0 / distance);

                // Attenuation
                let atten = attenuation.0 + attenuation.1 * distance + attenuation.2 * distance * distance;
                let atten_factor = 1.0 / atten.max(1.0);

                // Lambertian shading
                let diffuse = normal.dot(&light_dir).max(0.0);

                let intensity = self.intensity * atten_factor * diffuse;
                let r = self.color.red() * intensity;
                let g = self.color.green() * intensity;
                let b = self.color.blue() * intensity;

                (r, g, b)
            }

            LightType::Ambient => {
                // Uniform illumination
                let r = self.color.red() * self.intensity;
                let g = self.color.green() * self.intensity;
                let b = self.color.blue() * self.intensity;

                (r, g, b)
            }
        }
    }
}

/// Default lighting setup for a 3D scene
pub fn default_lights() -> Vec<Light> {
    vec![
        Light::default_directional(),
        Light::default_ambient(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_creation() {
        let camera = Camera::new(
            Point3D::new(5.0, 5.0, 5.0),
            Point3D::origin(),
            Vector3D::unit_z(),
        );

        assert_eq!(camera.position, Point3D::new(5.0, 5.0, 5.0));
        assert_eq!(camera.look_at, Point3D::origin());
        assert_eq!(camera.projection, ProjectionType::Perspective);
    }

    #[test]
    fn test_camera_view_direction() {
        let camera = Camera::new(
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::origin(),
            Vector3D::unit_z(),
        );

        let view_dir = camera.view_direction();
        assert!((view_dir.x + 1.0).abs() < 1e-10);
        assert!((view_dir.y).abs() < 1e-10);
        assert!((view_dir.z).abs() < 1e-10);
    }

    #[test]
    fn test_camera_default_for_scene() {
        let camera = Camera::default_for_scene(Point3D::origin(), 10.0);
        assert_eq!(camera.look_at, Point3D::origin());
        assert!(camera.position.distance_to(&Point3D::origin()) > 10.0);
    }

    #[test]
    fn test_camera_zoom() {
        let mut camera = Camera::new(
            Point3D::new(10.0, 0.0, 0.0),
            Point3D::origin(),
            Vector3D::unit_z(),
        );

        let initial_distance = camera.position.distance_to(&camera.look_at);
        camera.zoom(0.5);
        let new_distance = camera.position.distance_to(&camera.look_at);

        assert!(new_distance < initial_distance);
    }

    #[test]
    fn test_directional_light() {
        let light = Light::directional(
            Vector3D::new(0.0, 0.0, -1.0),
            Color::white(),
            1.0,
        );

        let point = Point3D::origin();
        let normal = Vector3D::unit_z();

        let (r, g, b) = light.illuminate(&point, &normal);

        // Light pointing down, normal pointing up -> full illumination
        assert!((r - 1.0).abs() < 1e-10);
        assert!((g - 1.0).abs() < 1e-10);
        assert!((b - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_directional_light_angle() {
        let light = Light::directional(
            Vector3D::new(1.0, 0.0, 0.0),
            Color::white(),
            1.0,
        );

        let point = Point3D::origin();
        let normal = Vector3D::unit_z();

        let (r, g, b) = light.illuminate(&point, &normal);

        // Light from side, normal pointing up -> no illumination
        assert!((r).abs() < 1e-10);
        assert!((g).abs() < 1e-10);
        assert!((b).abs() < 1e-10);
    }

    #[test]
    fn test_ambient_light() {
        let light = Light::ambient(Color::white(), 0.3);

        let point = Point3D::origin();
        let normal = Vector3D::unit_z();

        let (r, g, b) = light.illuminate(&point, &normal);

        // Ambient light is uniform
        assert!((r - 0.3).abs() < 1e-10);
        assert!((g - 0.3).abs() < 1e-10);
        assert!((b - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_point_light() {
        let light = Light::point(
            Point3D::new(0.0, 0.0, 10.0),
            Color::white(),
            1.0,
        );

        let point = Point3D::origin();
        let normal = Vector3D::unit_z();

        let (r, g, b) = light.illuminate(&point, &normal);

        // Point light above, normal pointing up -> full illumination
        assert!((r - 1.0).abs() < 1e-10);
        assert!((g - 1.0).abs() < 1e-10);
        assert!((b - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_default_lights() {
        let lights = default_lights();
        assert_eq!(lights.len(), 2);

        // First should be directional
        assert!(matches!(lights[0].light_type, LightType::Directional { .. }));

        // Second should be ambient
        assert!(matches!(lights[1].light_type, LightType::Ambient));
    }
}
