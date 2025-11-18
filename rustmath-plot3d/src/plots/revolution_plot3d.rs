//! Surfaces of revolution from 2D curves

use crate::base::{Graphics3d, IndexFaceSet, Point3D};
use crate::Result;
use rustmath_colors::Color;
use std::f64::consts::PI;

/// Options for revolution plots
#[derive(Debug, Clone)]
pub struct RevolutionPlot3dOptions {
    /// Number of samples along the curve
    pub t_samples: usize,
    /// Number of rotational samples (around the axis)
    pub theta_samples: usize,
    /// Color of the surface
    pub color: Option<Color>,
    /// Axis of revolution (0=x, 1=y, 2=z)
    pub axis: usize,
}

impl Default for RevolutionPlot3dOptions {
    fn default() -> Self {
        Self {
            t_samples: 50,
            theta_samples: 50,
            color: None,
            axis: 2, // Default to z-axis
        }
    }
}

impl RevolutionPlot3dOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_samples(mut self, t_samples: usize, theta_samples: usize) -> Self {
        self.t_samples = t_samples;
        self.theta_samples = theta_samples;
        self
    }

    pub fn with_color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }

    pub fn with_axis(mut self, axis: usize) -> Self {
        self.axis = axis;
        self
    }
}

/// Create a surface of revolution by rotating a 2D curve around an axis
///
/// # Arguments
/// * `curve` - Function that maps t to (r, z), where r is the radial distance and z is the height
/// * `t_range` - Range for parameter t along the curve (min, max)
/// * `theta_range` - Optional range for rotation angle (default: 0 to 2π)
/// * `options` - Optional plotting options
///
/// # Example
/// ```no_run
/// use rustmath_plot3d::plots::revolution_plot3d;
/// use std::f64::consts::PI;
///
/// // Create a vase shape
/// let curve = |t: f64| {
///     let r = 1.0 + 0.5 * (2.0 * PI * t).sin();
///     let z = t;
///     (r, z)
/// };
///
/// let graphics = revolution_plot3d(curve, (0.0, 2.0), None, None).unwrap();
/// ```
pub fn revolution_plot3d<F>(
    curve: F,
    t_range: (f64, f64),
    theta_range: Option<(f64, f64)>,
    options: Option<RevolutionPlot3dOptions>,
) -> Result<Graphics3d>
where
    F: Fn(f64) -> (f64, f64),
{
    let opts = options.unwrap_or_default();
    let (t_min, t_max) = t_range;
    let (theta_min, theta_max) = theta_range.unwrap_or((0.0, 2.0 * PI));

    let mut vertices = Vec::new();
    let mut faces = Vec::new();

    // Generate vertices by rotating the curve
    for i in 0..=opts.t_samples {
        let t = t_min + (i as f64 / opts.t_samples as f64) * (t_max - t_min);
        let (r, z) = curve(t);

        for j in 0..=opts.theta_samples {
            let theta = theta_min + (j as f64 / opts.theta_samples as f64) * (theta_max - theta_min);

            let (x, y) = match opts.axis {
                0 => {
                    // Revolve around x-axis: (r, z) -> (z, r*cos(θ), r*sin(θ))
                    // Return (y, z) for consistency with other functions
                    (r * theta.cos(), r * theta.sin())
                }
                1 => {
                    // Revolve around y-axis: (r, z) -> (r*cos(θ), z, r*sin(θ))
                    // Return (x, z) for consistency
                    (r * theta.cos(), r * theta.sin())
                }
                2 | _ => {
                    // Revolve around z-axis: (r, z) -> (r*cos(θ), r*sin(θ), z)
                    (r * theta.cos(), r * theta.sin())
                }
            };

            let point = match opts.axis {
                0 => Point3D { x: z, y: x, z: y },
                1 => Point3D { x, y: z, z: y },
                2 | _ => Point3D { x, y, z },
            };

            vertices.push(point);
        }
    }

    // Generate faces (two triangles per grid cell)
    for i in 0..opts.t_samples {
        for j in 0..opts.theta_samples {
            let idx0 = i * (opts.theta_samples + 1) + j;
            let idx1 = idx0 + 1;
            let idx2 = (i + 1) * (opts.theta_samples + 1) + j;
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
    fn test_revolution_plot3d_cylinder() {
        // Straight vertical line creates a cylinder
        let curve = |t: f64| (1.0, t);
        let result = revolution_plot3d(curve, (0.0, 2.0), None, None);
        assert!(result.is_ok());
        let graphics = result.unwrap();
        assert_eq!(graphics.objects.len(), 1);
    }

    #[test]
    fn test_revolution_plot3d_cone() {
        // Linear taper creates a cone
        let curve = |t: f64| (t, t);
        let opts = RevolutionPlot3dOptions::new()
            .with_samples(30, 30)
            .with_color(Color::rgb(1.0, 0.5, 0.0));
        let result = revolution_plot3d(curve, (0.0, 1.0), None, Some(opts));
        assert!(result.is_ok());
    }

    #[test]
    fn test_revolution_plot3d_sphere() {
        // Semicircle creates a sphere
        let curve = |t: f64| {
            let r = t.sin();
            let z = t.cos();
            (r, z)
        };
        let result = revolution_plot3d(curve, (0.0, PI), None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_revolution_plot3d_partial_revolution() {
        // Partial revolution (not full 360 degrees)
        let curve = |t: f64| (1.0, t);
        let result = revolution_plot3d(curve, (0.0, 1.0), Some((0.0, PI)), None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_revolution_plot3d_vase() {
        // Wavy curve creates a vase
        let curve = |t: f64| {
            let r = 1.0 + 0.3 * (3.0 * t).sin();
            let z = t;
            (r, z)
        };
        let result = revolution_plot3d(curve, (0.0, 2.0), None, None);
        assert!(result.is_ok());
    }
}
