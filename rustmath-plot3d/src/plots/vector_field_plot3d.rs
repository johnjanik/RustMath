//! 3D vector field visualization
//!
//! Visualizes 3D vector fields by drawing arrows at sample points.
//! Each arrow's direction and magnitude represent the vector field at that point.

use crate::base::{Graphics3d, IndexFaceSet, Point3D, Vector3D};
use crate::Result;
use rustmath_colors::Color;

/// Options for 3D vector field plots
#[derive(Debug, Clone)]
pub struct VectorFieldPlot3dOptions {
    /// Number of sample points in x direction
    pub x_samples: usize,
    /// Number of sample points in y direction
    pub y_samples: usize,
    /// Number of sample points in z direction
    pub z_samples: usize,
    /// Color of the vectors
    pub color: Option<Color>,
    /// Scale factor for arrow lengths
    pub scale: f64,
    /// Arrow head size (relative to arrow length)
    pub arrow_head_size: f64,
    /// Arrow shaft radius (for 3D arrows)
    pub arrow_radius: f64,
    /// Normalize vectors to unit length
    pub normalize: bool,
}

impl Default for VectorFieldPlot3dOptions {
    fn default() -> Self {
        Self {
            x_samples: 10,
            y_samples: 10,
            z_samples: 10,
            color: Some(Color::rgb(0.0, 0.0, 1.0)),
            scale: 1.0,
            arrow_head_size: 0.2,
            arrow_radius: 0.02,
            normalize: false,
        }
    }
}

impl VectorFieldPlot3dOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_samples(mut self, x_samples: usize, y_samples: usize, z_samples: usize) -> Self {
        self.x_samples = x_samples;
        self.y_samples = y_samples;
        self.z_samples = z_samples;
        self
    }

    pub fn with_color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }

    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    pub fn with_arrow_head_size(mut self, size: f64) -> Self {
        self.arrow_head_size = size;
        self
    }
}

/// Create a 3D vector field plot
///
/// # Arguments
/// * `fx` - Function that returns the x-component of the vector at (x, y, z)
/// * `fy` - Function that returns the y-component of the vector at (x, y, z)
/// * `fz` - Function that returns the z-component of the vector at (x, y, z)
/// * `x_range` - Range for x values (min, max)
/// * `y_range` - Range for y values (min, max)
/// * `z_range` - Range for z values (min, max)
/// * `options` - Optional plotting options
///
/// # Example
/// ```no_run
/// use rustmath_plot3d::plots::vector_field_plot3d;
/// use rustmath_plot3d::plots::vector_field_plot3d::VectorFieldPlot3dOptions;
///
/// // Rotation field: V(x,y,z) = (-y, x, 0)
/// let fx = |_x: f64, y: f64, _z: f64| -y;
/// let fy = |x: f64, _y: f64, _z: f64| x;
/// let fz = |_x: f64, _y: f64, _z: f64| 0.0;
///
/// let opts = VectorFieldPlot3dOptions::new()
///     .with_samples(8, 8, 8)
///     .with_scale(0.5);
/// let graphics = vector_field_plot3d(fx, fy, fz, (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), Some(opts)).unwrap();
/// ```
pub fn vector_field_plot3d<FX, FY, FZ>(
    fx: FX,
    fy: FY,
    fz: FZ,
    x_range: (f64, f64),
    y_range: (f64, f64),
    z_range: (f64, f64),
    options: Option<VectorFieldPlot3dOptions>,
) -> Result<Graphics3d>
where
    FX: Fn(f64, f64, f64) -> f64,
    FY: Fn(f64, f64, f64) -> f64,
    FZ: Fn(f64, f64, f64) -> f64,
{
    let opts = options.unwrap_or_default();
    let (x_min, x_max) = x_range;
    let (y_min, y_max) = y_range;
    let (z_min, z_max) = z_range;

    let mut graphics = Graphics3d::new();

    // Sample the vector field
    for i in 0..opts.x_samples {
        let t_x = i as f64 / (opts.x_samples - 1).max(1) as f64;
        let x = x_min + t_x * (x_max - x_min);

        for j in 0..opts.y_samples {
            let t_y = j as f64 / (opts.y_samples - 1).max(1) as f64;
            let y = y_min + t_y * (y_max - y_min);

            for k in 0..opts.z_samples {
                let t_z = k as f64 / (opts.z_samples - 1).max(1) as f64;
                let z = z_min + t_z * (z_max - z_min);

                // Get vector at this point
                let vx = fx(x, y, z);
                let vy = fy(x, y, z);
                let vz = fz(x, y, z);

                let mut vector = Vector3D::new(vx, vy, vz);

                // Normalize if requested
                if opts.normalize {
                    vector = vector.normalize();
                }

                // Skip zero-length vectors
                if vector.magnitude() < 1e-10 {
                    continue;
                }

                // Scale vector
                vector = vector.scale(opts.scale);

                // Create arrow mesh
                let start = Point3D::new(x, y, z);
                let arrow_mesh = create_arrow_mesh(
                    start,
                    vector,
                    opts.arrow_head_size,
                    opts.arrow_radius,
                );

                let mut mesh = arrow_mesh;
                mesh.compute_normals();

                // Apply color
                if let Some(ref color) = opts.color {
                    mesh.vertex_colors = Some(vec![color.clone(); mesh.vertices.len()]);
                }

                graphics.add_mesh(mesh);
            }
        }
    }

    Ok(graphics)
}

/// Create a 3D arrow mesh from start point and direction vector
fn create_arrow_mesh(
    start: Point3D,
    direction: Vector3D,
    head_size: f64,
    shaft_radius: f64,
) -> IndexFaceSet {
    let length = direction.magnitude();
    if length < 1e-10 {
        return IndexFaceSet::empty();
    }

    let dir = direction.normalize();

    // End point of arrow
    let end = Point3D::new(
        start.x + direction.x,
        start.y + direction.y,
        start.z + direction.z,
    );

    // Point where head starts
    let head_start_length = length * (1.0 - head_size);
    let head_start = Point3D::new(
        start.x + dir.x * head_start_length,
        start.y + dir.y * head_start_length,
        start.z + dir.z * head_start_length,
    );

    // Create a coordinate system perpendicular to the arrow direction
    let (perp1, perp2) = get_perpendicular_vectors(&dir);

    let mut vertices = Vec::new();
    let mut faces = Vec::new();

    // Create shaft as a cylinder
    let num_sides = 8;

    // Shaft start vertices
    for i in 0..num_sides {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / num_sides as f64;
        let offset = perp1.scale(shaft_radius * angle.cos())
            + perp2.scale(shaft_radius * angle.sin());

        vertices.push(Point3D::new(
            start.x + offset.x,
            start.y + offset.y,
            start.z + offset.z,
        ));
    }

    // Shaft end (at head start) vertices
    for i in 0..num_sides {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / num_sides as f64;
        let offset = perp1.scale(shaft_radius * angle.cos())
            + perp2.scale(shaft_radius * angle.sin());

        vertices.push(Point3D::new(
            head_start.x + offset.x,
            head_start.y + offset.y,
            head_start.z + offset.z,
        ));
    }

    // Create shaft faces
    for i in 0..num_sides {
        let next = (i + 1) % num_sides;

        // Two triangles per face
        faces.push([i, i + num_sides, next]);
        faces.push([next, i + num_sides, next + num_sides]);
    }

    // Head base vertices (larger radius)
    let head_radius = shaft_radius * 2.0;
    let head_base_start_idx = vertices.len();

    for i in 0..num_sides {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / num_sides as f64;
        let offset = perp1.scale(head_radius * angle.cos())
            + perp2.scale(head_radius * angle.sin());

        vertices.push(Point3D::new(
            head_start.x + offset.x,
            head_start.y + offset.y,
            head_start.z + offset.z,
        ));
    }

    // Arrow tip
    let tip_idx = vertices.len();
    vertices.push(end);

    // Create head faces (cone)
    for i in 0..num_sides {
        let next = (i + 1) % num_sides;
        faces.push([head_base_start_idx + i, tip_idx, head_base_start_idx + next]);
    }

    IndexFaceSet::new(vertices, faces)
}

/// Get two perpendicular unit vectors to the given vector
fn get_perpendicular_vectors(v: &Vector3D) -> (Vector3D, Vector3D) {
    // Choose a vector that's not parallel to v
    let temp = if v.x.abs() > 0.9 {
        Vector3D::new(0.0, 1.0, 0.0)
    } else {
        Vector3D::new(1.0, 0.0, 0.0)
    };

    let perp1 = v.cross(&temp).normalize();
    let perp2 = v.cross(&perp1).normalize();

    (perp1, perp2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_field_plot3d_simple() {
        // Constant field pointing in +z direction
        let fx = |_x: f64, _y: f64, _z: f64| 0.0;
        let fy = |_x: f64, _y: f64, _z: f64| 0.0;
        let fz = |_x: f64, _y: f64, _z: f64| 1.0;

        let opts = VectorFieldPlot3dOptions::new().with_samples(3, 3, 3);
        let result = vector_field_plot3d(fx, fy, fz, (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), Some(opts));
        assert!(result.is_ok());
        let graphics = result.unwrap();
        // Should have 3x3x3 = 27 arrows
        assert_eq!(graphics.objects.len(), 27);
    }

    #[test]
    fn test_vector_field_plot3d_rotation() {
        // Rotation field around z-axis
        let fx = |_x: f64, y: f64, _z: f64| -y;
        let fy = |x: f64, _y: f64, _z: f64| x;
        let fz = |_x: f64, _y: f64, _z: f64| 0.0;

        let opts = VectorFieldPlot3dOptions::new()
            .with_samples(5, 5, 2)
            .with_normalize(true);
        let result = vector_field_plot3d(fx, fy, fz, (-1.0, 1.0), (-1.0, 1.0), (0.0, 0.5), Some(opts));
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_perpendicular_vectors() {
        let v = Vector3D::new(1.0, 0.0, 0.0);
        let (perp1, perp2) = get_perpendicular_vectors(&v);

        // Check they're perpendicular
        assert!((v.dot(&perp1)).abs() < 1e-10);
        assert!((v.dot(&perp2)).abs() < 1e-10);
        assert!((perp1.dot(&perp2)).abs() < 1e-10);

        // Check they're unit vectors
        assert!((perp1.magnitude() - 1.0).abs() < 1e-10);
        assert!((perp2.magnitude() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_create_arrow_mesh() {
        let start = Point3D::new(0.0, 0.0, 0.0);
        let direction = Vector3D::new(0.0, 0.0, 1.0);
        let mesh = create_arrow_mesh(start, direction, 0.2, 0.05);

        // Should have non-zero vertices and faces
        assert!(mesh.vertices.len() > 0);
        assert!(mesh.faces.len() > 0);
    }
}
