//! 3D scatter plots showing individual points as spheres or markers
//!
//! Scatter plots display discrete 3D points, optionally with different sizes
//! and colors based on data values.

use crate::base::{Graphics3d, IndexFaceSet, Point3D};
use crate::Result;
use rustmath_colors::Color;

/// Options for 3D scatter plots
#[derive(Debug, Clone)]
pub struct ScatterPlot3dOptions {
    /// Color of the points (single color or per-point colors)
    pub color: Option<Color>,
    /// Per-point colors (overrides `color` if set)
    pub colors: Option<Vec<Color>>,
    /// Size of the point markers
    pub marker_size: f64,
    /// Number of subdivisions for sphere markers (higher = smoother)
    pub marker_resolution: usize,
}

impl Default for ScatterPlot3dOptions {
    fn default() -> Self {
        Self {
            color: Some(Color::rgb(0.0, 0.5, 1.0)),
            colors: None,
            marker_size: 0.1,
            marker_resolution: 8,
        }
    }
}

impl ScatterPlot3dOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self.colors = None;
        self
    }

    pub fn with_colors(mut self, colors: Vec<Color>) -> Self {
        self.colors = Some(colors);
        self
    }

    pub fn with_marker_size(mut self, size: f64) -> Self {
        self.marker_size = size;
        self
    }

    pub fn with_marker_resolution(mut self, resolution: usize) -> Self {
        self.marker_resolution = resolution;
        self
    }
}

/// Create a 3D scatter plot from a list of points
///
/// This creates small sphere markers at each data point location.
///
/// # Arguments
/// * `points` - List of (x, y, z) coordinates
/// * `options` - Optional plotting options
///
/// # Example
/// ```no_run
/// use rustmath_plot3d::plots::scatter_plot3d;
/// use rustmath_plot3d::plots::scatter_plot3d::ScatterPlot3dOptions;
///
/// let points = vec![
///     (0.0, 0.0, 0.0),
///     (1.0, 1.0, 1.0),
///     (0.5, 0.5, 0.5),
/// ];
/// let opts = ScatterPlot3dOptions::new().with_marker_size(0.2);
/// let graphics = scatter_plot3d(&points, Some(opts)).unwrap();
/// ```
pub fn scatter_plot3d(
    points: &[(f64, f64, f64)],
    options: Option<ScatterPlot3dOptions>,
) -> Result<Graphics3d> {
    let opts = options.unwrap_or_default();

    let mut graphics = Graphics3d::new();

    for (i, &(x, y, z)) in points.iter().enumerate() {
        let center = Point3D::new(x, y, z);

        // Create a sphere mesh at this point
        let sphere_mesh = create_sphere_mesh(
            center,
            opts.marker_size,
            opts.marker_resolution,
        );

        // Determine color for this point
        let color = if let Some(ref colors) = opts.colors {
            if i < colors.len() {
                colors[i].clone()
            } else {
                opts.color.clone().unwrap_or(Color::rgb(0.0, 0.5, 1.0))
            }
        } else {
            opts.color.clone().unwrap_or(Color::rgb(0.0, 0.5, 1.0))
        };

        let mut mesh = sphere_mesh;
        mesh.vertex_colors = Some(vec![color; mesh.vertices.len()]);
        mesh.compute_normals();

        graphics.add_mesh(mesh);
    }

    Ok(graphics)
}

/// Create a sphere mesh centered at a given point
///
/// Uses UV sphere parameterization
fn create_sphere_mesh(center: Point3D, radius: f64, resolution: usize) -> IndexFaceSet {
    let mut vertices = Vec::new();
    let mut faces = Vec::new();

    // UV sphere parameterization
    // Create vertices
    for i in 0..=resolution {
        let theta = std::f64::consts::PI * (i as f64) / (resolution as f64);

        for j in 0..=resolution {
            let phi = 2.0 * std::f64::consts::PI * (j as f64) / (resolution as f64);

            let x = center.x + radius * theta.sin() * phi.cos();
            let y = center.y + radius * theta.sin() * phi.sin();
            let z = center.z + radius * theta.cos();

            vertices.push(Point3D::new(x, y, z));
        }
    }

    // Create faces
    for i in 0..resolution {
        for j in 0..resolution {
            let idx0 = i * (resolution + 1) + j;
            let idx1 = idx0 + 1;
            let idx2 = (i + 1) * (resolution + 1) + j;
            let idx3 = idx2 + 1;

            // Skip degenerate triangles at poles
            if i > 0 {
                faces.push([idx0, idx2, idx1]);
            }
            if i < resolution - 1 {
                faces.push([idx1, idx2, idx3]);
            }
        }
    }

    IndexFaceSet::new(vertices, faces)
}

/// Create a scatter plot with point colors based on a function of coordinates
///
/// # Arguments
/// * `points` - List of (x, y, z) coordinates
/// * `color_fn` - Function that maps (x, y, z) to a color value in [0, 1]
/// * `colormap` - Function that maps [0, 1] to a Color
/// * `options` - Optional plotting options
pub fn scatter_plot3d_colored<F, C>(
    points: &[(f64, f64, f64)],
    color_fn: F,
    colormap: C,
    options: Option<ScatterPlot3dOptions>,
) -> Result<Graphics3d>
where
    F: Fn(f64, f64, f64) -> f64,
    C: Fn(f64) -> Color,
{
    let mut opts = options.unwrap_or_default();

    // Compute colors for each point
    let colors: Vec<Color> = points
        .iter()
        .map(|&(x, y, z)| {
            let value = color_fn(x, y, z);
            // Clamp value to [0, 1]
            let clamped = value.max(0.0).min(1.0);
            colormap(clamped)
        })
        .collect();

    opts.colors = Some(colors);

    scatter_plot3d(points, Some(opts))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scatter_plot3d_simple() {
        let points = vec![(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)];
        let result = scatter_plot3d(&points, None);
        assert!(result.is_ok());
        let graphics = result.unwrap();
        assert_eq!(graphics.objects.len(), 2); // Two spheres
    }

    #[test]
    fn test_scatter_plot3d_with_options() {
        let points = vec![(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)];
        let opts = ScatterPlot3dOptions::new()
            .with_marker_size(0.3)
            .with_color(Color::rgb(1.0, 0.0, 0.0));
        let result = scatter_plot3d(&points, Some(opts));
        assert!(result.is_ok());
        let graphics = result.unwrap();
        assert_eq!(graphics.objects.len(), 3);
    }

    #[test]
    fn test_scatter_plot3d_with_individual_colors() {
        let points = vec![(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)];
        let colors = vec![Color::rgb(1.0, 0.0, 0.0), Color::rgb(0.0, 1.0, 0.0)];
        let opts = ScatterPlot3dOptions::new().with_colors(colors);
        let result = scatter_plot3d(&points, Some(opts));
        assert!(result.is_ok());
    }

    #[test]
    fn test_scatter_plot3d_colored() {
        let points = vec![
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (2.0, 2.0, 2.0),
        ];

        // Color based on z-coordinate
        let color_fn = |_x: f64, _y: f64, z: f64| z / 2.0;

        // Simple colormap: blue to red
        let colormap = |t: f64| Color::rgb(t, 0.0, 1.0 - t);

        let result = scatter_plot3d_colored(&points, color_fn, colormap, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_create_sphere_mesh() {
        let center = Point3D::new(0.0, 0.0, 0.0);
        let mesh = create_sphere_mesh(center, 1.0, 10);

        // Should have (resolution+1)^2 vertices
        assert_eq!(mesh.vertices.len(), 11 * 11);

        // Should have non-zero faces
        assert!(mesh.faces.len() > 0);
    }
}
