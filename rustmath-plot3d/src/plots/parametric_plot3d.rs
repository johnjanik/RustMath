//! Parametric 3D plots from functions (u, v) -> (x, y, z)

use crate::base::{Graphics3d, IndexFaceSet, Point3D};
use crate::Result;
use rustmath_colors::Color;

/// Options for parametric 3D plots
#[derive(Debug, Clone)]
pub struct ParametricPlot3dOptions {
    /// Number of samples in u direction
    pub u_samples: usize,
    /// Number of samples in v direction
    pub v_samples: usize,
    /// Color of the surface
    pub color: Option<Color>,
}

impl Default for ParametricPlot3dOptions {
    fn default() -> Self {
        Self {
            u_samples: 50,
            v_samples: 50,
            color: None,
        }
    }
}

impl ParametricPlot3dOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_samples(mut self, u_samples: usize, v_samples: usize) -> Self {
        self.u_samples = u_samples;
        self.v_samples = v_samples;
        self
    }

    pub fn with_color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }
}

/// Create a parametric 3D surface plot from parametric equations
///
/// # Arguments
/// * `fx` - Function that maps (u, v) to x
/// * `fy` - Function that maps (u, v) to y
/// * `fz` - Function that maps (u, v) to z
/// * `u_range` - Range for u parameter (min, max)
/// * `v_range` - Range for v parameter (min, max)
/// * `options` - Optional plotting options
///
/// # Example
/// ```no_run
/// use rustmath_plot3d::plots::parametric_plot3d;
/// use std::f64::consts::PI;
///
/// // Torus parametric equations
/// let r = 2.0; // Major radius
/// let a = 0.5; // Minor radius
/// let fx = |u: f64, v: f64| (r + a * v.cos()) * u.cos();
/// let fy = |u: f64, v: f64| (r + a * v.cos()) * u.sin();
/// let fz = |_: f64, v: f64| a * v.sin();
///
/// let graphics = parametric_plot3d(fx, fy, fz, (0.0, 2.0 * PI), (0.0, 2.0 * PI), None).unwrap();
/// ```
pub fn parametric_plot3d<FX, FY, FZ>(
    fx: FX,
    fy: FY,
    fz: FZ,
    u_range: (f64, f64),
    v_range: (f64, f64),
    options: Option<ParametricPlot3dOptions>,
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
    let mut faces = Vec::new();

    // Generate grid of vertices
    for i in 0..=opts.v_samples {
        let v = v_min + (i as f64 / opts.v_samples as f64) * (v_max - v_min);
        for j in 0..=opts.u_samples {
            let u = u_min + (j as f64 / opts.u_samples as f64) * (u_max - u_min);
            let x = fx(u, v);
            let y = fy(u, v);
            let z = fz(u, v);
            vertices.push(Point3D { x, y, z });
        }
    }

    // Generate faces (two triangles per grid cell)
    for i in 0..opts.v_samples {
        for j in 0..opts.u_samples {
            let idx0 = i * (opts.u_samples + 1) + j;
            let idx1 = idx0 + 1;
            let idx2 = (i + 1) * (opts.u_samples + 1) + j;
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
    use std::f64::consts::PI;

    #[test]
    fn test_parametric_plot3d_sphere() {
        // Sphere parametric equations
        let r = 1.0;
        let fx = |u: f64, v: f64| r * v.sin() * u.cos();
        let fy = |u: f64, v: f64| r * v.sin() * u.sin();
        let fz = |_: f64, v: f64| r * v.cos();

        let result = parametric_plot3d(fx, fy, fz, (0.0, 2.0 * PI), (0.0, PI), None);
        assert!(result.is_ok());
        let graphics = result.unwrap();
        assert_eq!(graphics.objects.len(), 1);
    }

    #[test]
    fn test_parametric_plot3d_torus() {
        // Torus parametric equations
        let r = 2.0; // Major radius
        let a = 0.5; // Minor radius
        let fx = |u: f64, v: f64| (r + a * v.cos()) * u.cos();
        let fy = |u: f64, v: f64| (r + a * v.cos()) * u.sin();
        let fz = |_: f64, v: f64| a * v.sin();

        let opts = ParametricPlot3dOptions::new()
            .with_samples(20, 10)
            .with_color(Color::rgb(0.0, 1.0, 0.0));

        let result = parametric_plot3d(fx, fy, fz, (0.0, 2.0 * PI), (0.0, 2.0 * PI), Some(opts));
        assert!(result.is_ok());
    }

    #[test]
    fn test_parametric_plot3d_helicoid() {
        // Helicoid parametric equations
        let fx = |u: f64, v: f64| v * u.cos();
        let fy = |u: f64, v: f64| v * u.sin();
        let fz = |u: f64, _: f64| u;

        let result = parametric_plot3d(fx, fy, fz, (0.0, 2.0 * PI), (-1.0, 1.0), None);
        assert!(result.is_ok());
    }
}
