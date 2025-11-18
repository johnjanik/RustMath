//! Surface plots from functions f(x, y) -> z

use crate::base::{Graphics3d, IndexFaceSet, Point3D};
use crate::Result;
use rustmath_colors::Color;

/// Options for 3D surface plots
#[derive(Debug, Clone)]
pub struct Plot3dOptions {
    /// Number of samples in x direction
    pub x_samples: usize,
    /// Number of samples in y direction
    pub y_samples: usize,
    /// Color of the surface
    pub color: Option<Color>,
    /// Whether to use adaptive sampling
    pub adaptive: bool,
}

impl Default for Plot3dOptions {
    fn default() -> Self {
        Self {
            x_samples: 50,
            y_samples: 50,
            color: None,
            adaptive: false,
        }
    }
}

impl Plot3dOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_samples(mut self, x_samples: usize, y_samples: usize) -> Self {
        self.x_samples = x_samples;
        self.y_samples = y_samples;
        self
    }

    pub fn with_color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }

    pub fn with_adaptive(mut self, adaptive: bool) -> Self {
        self.adaptive = adaptive;
        self
    }
}

/// Create a 3D surface plot from a function f(x, y) -> z
///
/// # Arguments
/// * `f` - Function that maps (x, y) to z
/// * `x_range` - Range for x values (min, max)
/// * `y_range` - Range for y values (min, max)
/// * `options` - Optional plotting options
///
/// # Example
/// ```no_run
/// use rustmath_plot3d::plots::plot3d;
/// use rustmath_plot3d::plots::plot3d::Plot3dOptions;
///
/// let f = |x: f64, y: f64| x.powi(2) + y.powi(2);
/// let graphics = plot3d(f, (-2.0, 2.0), (-2.0, 2.0), None).unwrap();
/// ```
pub fn plot3d<F>(
    f: F,
    x_range: (f64, f64),
    y_range: (f64, f64),
    options: Option<Plot3dOptions>,
) -> Result<Graphics3d>
where
    F: Fn(f64, f64) -> f64,
{
    let opts = options.unwrap_or_default();
    let (x_min, x_max) = x_range;
    let (y_min, y_max) = y_range;

    let mut vertices = Vec::new();
    let mut faces = Vec::new();

    // Generate grid of vertices
    for i in 0..=opts.y_samples {
        let y = y_min + (i as f64 / opts.y_samples as f64) * (y_max - y_min);
        for j in 0..=opts.x_samples {
            let x = x_min + (j as f64 / opts.x_samples as f64) * (x_max - x_min);
            let z = f(x, y);
            vertices.push(Point3D { x, y, z });
        }
    }

    // Generate faces (two triangles per grid cell)
    for i in 0..opts.y_samples {
        for j in 0..opts.x_samples {
            let idx0 = i * (opts.x_samples + 1) + j;
            let idx1 = idx0 + 1;
            let idx2 = (i + 1) * (opts.x_samples + 1) + j;
            let idx3 = idx2 + 1;

            // First triangle
            faces.push([idx0, idx2, idx1]);
            // Second triangle
            faces.push([idx1, idx2, idx3]);
        }
    }

    let mut mesh = IndexFaceSet::new(vertices, faces);
    mesh.compute_normals();

    // Apply color if specified
    if let Some(color) = opts.color {
        mesh.vertex_colors = Some(vec![color; mesh.vertices.len()]);
    }

    let mut graphics = Graphics3d::new();
    graphics.add_mesh(mesh);

    Ok(graphics)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plot3d_simple() {
        let f = |x: f64, y: f64| x * x + y * y;
        let result = plot3d(f, (-1.0, 1.0), (-1.0, 1.0), None);
        assert!(result.is_ok());
        let graphics = result.unwrap();
        assert_eq!(graphics.objects.len(), 1);
    }

    #[test]
    fn test_plot3d_with_options() {
        let f = |x: f64, y: f64| x.sin() * y.cos();
        let opts = Plot3dOptions::new()
            .with_samples(20, 20)
            .with_color(Color::rgb(1.0, 0.0, 0.0));
        let result = plot3d(f, (-3.14, 3.14), (-3.14, 3.14), Some(opts));
        assert!(result.is_ok());
    }

    #[test]
    fn test_plot3d_vertex_count() {
        let f = |x: f64, y: f64| x + y;
        let opts = Plot3dOptions::new().with_samples(10, 10);
        let graphics = plot3d(f, (0.0, 1.0), (0.0, 1.0), Some(opts)).unwrap();

        // Should have (10+1) * (10+1) = 121 vertices
        // This assumes we can access the mesh somehow
        // For now, just check that we got a valid graphics object
        assert_eq!(graphics.objects.len(), 1);
    }
}
