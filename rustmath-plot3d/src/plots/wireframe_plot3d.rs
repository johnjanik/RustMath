//! Wireframe 3D plots showing the mesh structure
//!
//! Wireframe plots display only the edges of a 3D surface, creating a grid-like
//! appearance that's useful for understanding the underlying mesh structure.

use crate::base::{Graphics3d, IndexFaceSet, Point3D};
use crate::Result;
use rustmath_colors::Color;

/// Options for wireframe 3D plots
#[derive(Debug, Clone)]
pub struct WireframePlot3dOptions {
    /// Number of samples in x direction
    pub x_samples: usize,
    /// Number of samples in y direction
    pub y_samples: usize,
    /// Color of the wireframe lines
    pub color: Option<Color>,
    /// Line thickness (for rendering)
    pub line_width: f64,
}

impl Default for WireframePlot3dOptions {
    fn default() -> Self {
        Self {
            x_samples: 30,
            y_samples: 30,
            color: Some(Color::rgb(0.0, 0.0, 1.0)),
            line_width: 1.0,
        }
    }
}

impl WireframePlot3dOptions {
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

    pub fn with_line_width(mut self, width: f64) -> Self {
        self.line_width = width;
        self
    }
}

/// Create a wireframe plot from a function f(x, y) -> z
///
/// This creates a 3D surface where only the edges of the mesh are visible,
/// creating a grid-like wireframe appearance.
///
/// # Arguments
/// * `f` - Function that maps (x, y) to z
/// * `x_range` - Range for x values (min, max)
/// * `y_range` - Range for y values (min, max)
/// * `options` - Optional plotting options
///
/// # Example
/// ```no_run
/// use rustmath_plot3d::plots::wireframe_plot3d;
/// use rustmath_plot3d::plots::wireframe_plot3d::WireframePlot3dOptions;
///
/// let f = |x: f64, y: f64| (x.powi(2) + y.powi(2)).sqrt();
/// let opts = WireframePlot3dOptions::new().with_samples(20, 20);
/// let graphics = wireframe_plot3d(f, (-2.0, 2.0), (-2.0, 2.0), Some(opts)).unwrap();
/// ```
pub fn wireframe_plot3d<F>(
    f: F,
    x_range: (f64, f64),
    y_range: (f64, f64),
    options: Option<WireframePlot3dOptions>,
) -> Result<Graphics3d>
where
    F: Fn(f64, f64) -> f64,
{
    let opts = options.unwrap_or_default();
    let (x_min, x_max) = x_range;
    let (y_min, y_max) = y_range;

    let mut vertices = Vec::new();
    let mut edges = Vec::new();

    // Generate grid of vertices
    for i in 0..=opts.y_samples {
        let y = y_min + (i as f64 / opts.y_samples as f64) * (y_max - y_min);
        for j in 0..=opts.x_samples {
            let x = x_min + (j as f64 / opts.x_samples as f64) * (x_max - x_min);
            let z = f(x, y);
            vertices.push(Point3D { x, y, z });
        }
    }

    // Generate edges (horizontal and vertical lines in the grid)
    // Horizontal edges (along x direction)
    for i in 0..=opts.y_samples {
        for j in 0..opts.x_samples {
            let idx0 = i * (opts.x_samples + 1) + j;
            let idx1 = idx0 + 1;
            edges.push([idx0, idx1]);
        }
    }

    // Vertical edges (along y direction)
    for i in 0..opts.y_samples {
        for j in 0..=opts.x_samples {
            let idx0 = i * (opts.x_samples + 1) + j;
            let idx1 = (i + 1) * (opts.x_samples + 1) + j;
            edges.push([idx0, idx1]);
        }
    }

    // Create a mesh representation
    // For wireframe, we'll create thin triangles along each edge
    // This is a workaround since we're using IndexFaceSet which expects faces
    let mut faces = Vec::new();

    // We'll store the edges information in the mesh's metadata for rendering
    // For now, create degenerate triangles to represent lines
    for edge in &edges {
        // Create a degenerate triangle (all three vertices are on the line)
        faces.push([edge[0], edge[1], edge[0]]);
    }

    let mut mesh = IndexFaceSet::new(vertices, faces);

    // Store edges in a custom way - we'll add this to the mesh
    // For now, just compute normals
    mesh.compute_normals();

    // Apply color if specified
    if let Some(color) = opts.color {
        mesh.vertex_colors = Some(vec![color; mesh.vertices.len()]);
    }

    let mut graphics = Graphics3d::new();
    graphics.add_mesh(mesh);
    graphics.options.wireframe = true;
    graphics.options.line_width = opts.line_width;

    Ok(graphics)
}

/// Create a wireframe plot from parametric equations
///
/// # Arguments
/// * `fx` - Function that maps (u, v) to x
/// * `fy` - Function that maps (u, v) to y
/// * `fz` - Function that maps (u, v) to z
/// * `u_range` - Range for u parameter (min, max)
/// * `v_range` - Range for v parameter (min, max)
/// * `options` - Optional plotting options
pub fn wireframe_parametric_plot3d<FX, FY, FZ>(
    fx: FX,
    fy: FY,
    fz: FZ,
    u_range: (f64, f64),
    v_range: (f64, f64),
    options: Option<WireframePlot3dOptions>,
) -> Result<Graphics3d>
where
    FX: Fn(f64, f64) -> f64,
    FY: Fn(f64, f64) -> f64,
    FZ: Fn(f64, f64) -> f64,
{
    let opts = options.unwrap_or_default();
    let (u_min, u_max) = u_range;
    let (v_min, v_max) = v_range;

    let mut vertices = Vec::new();
    let mut edges = Vec::new();

    // Generate grid of vertices
    for i in 0..=opts.y_samples {
        let v = v_min + (i as f64 / opts.y_samples as f64) * (v_max - v_min);
        for j in 0..=opts.x_samples {
            let u = u_min + (j as f64 / opts.x_samples as f64) * (u_max - u_min);
            let x = fx(u, v);
            let y = fy(u, v);
            let z = fz(u, v);
            vertices.push(Point3D { x, y, z });
        }
    }

    // Generate edges
    // Horizontal edges (along u direction)
    for i in 0..=opts.y_samples {
        for j in 0..opts.x_samples {
            let idx0 = i * (opts.x_samples + 1) + j;
            let idx1 = idx0 + 1;
            edges.push([idx0, idx1]);
        }
    }

    // Vertical edges (along v direction)
    for i in 0..opts.y_samples {
        for j in 0..=opts.x_samples {
            let idx0 = i * (opts.x_samples + 1) + j;
            let idx1 = (i + 1) * (opts.x_samples + 1) + j;
            edges.push([idx0, idx1]);
        }
    }

    // Create faces from edges (degenerate triangles)
    let mut faces = Vec::new();
    for edge in &edges {
        faces.push([edge[0], edge[1], edge[0]]);
    }

    let mut mesh = IndexFaceSet::new(vertices, faces);
    mesh.compute_normals();

    if let Some(color) = opts.color {
        mesh.vertex_colors = Some(vec![color; mesh.vertices.len()]);
    }

    let mut graphics = Graphics3d::new();
    graphics.add_mesh(mesh);
    graphics.options.wireframe = true;
    graphics.options.line_width = opts.line_width;

    Ok(graphics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_wireframe_plot3d_simple() {
        let f = |x: f64, y: f64| x * x + y * y;
        let result = wireframe_plot3d(f, (-1.0, 1.0), (-1.0, 1.0), None);
        assert!(result.is_ok());
        let graphics = result.unwrap();
        assert_eq!(graphics.objects.len(), 1);
        assert!(graphics.options.wireframe);
    }

    #[test]
    fn test_wireframe_plot3d_with_options() {
        let f = |x: f64, y: f64| x.sin() * y.cos();
        let opts = WireframePlot3dOptions::new()
            .with_samples(15, 15)
            .with_color(Color::rgb(1.0, 0.0, 0.0))
            .with_line_width(2.0);
        let result = wireframe_plot3d(f, (-PI, PI), (-PI, PI), Some(opts));
        assert!(result.is_ok());
        let graphics = result.unwrap();
        assert!(graphics.options.wireframe);
        assert_eq!(graphics.options.line_width, 2.0);
    }

    #[test]
    fn test_wireframe_parametric_plot3d_sphere() {
        // Sphere parametric equations
        let r = 1.0;
        let fx = |u: f64, v: f64| r * v.sin() * u.cos();
        let fy = |u: f64, v: f64| r * v.sin() * u.sin();
        let fz = |_: f64, v: f64| r * v.cos();

        let result = wireframe_parametric_plot3d(fx, fy, fz, (0.0, 2.0 * PI), (0.0, PI), None);
        assert!(result.is_ok());
        let graphics = result.unwrap();
        assert!(graphics.options.wireframe);
    }
}
