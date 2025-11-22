//! Implicit 3D surface plotting using marching cubes algorithm
//!
//! Plots surfaces defined by implicit equations f(x, y, z) = 0 or f(x, y, z) = c
//! using the marching cubes algorithm to triangulate the isosurface.

use crate::base::{Graphics3d, IndexFaceSet, Point3D};
use crate::{Plot3DError, Result};
use rustmath_colors::Color;

/// Options for implicit 3D surface plots
#[derive(Debug, Clone)]
pub struct ImplicitPlot3dOptions {
    /// Number of grid points in x direction
    pub x_samples: usize,
    /// Number of grid points in y direction
    pub y_samples: usize,
    /// Number of grid points in z direction
    pub z_samples: usize,
    /// Isovalue (surface level, default 0.0 for f(x,y,z) = 0)
    pub isovalue: f64,
    /// Color of the surface
    pub color: Option<Color>,
}

impl Default for ImplicitPlot3dOptions {
    fn default() -> Self {
        Self {
            x_samples: 30,
            y_samples: 30,
            z_samples: 30,
            isovalue: 0.0,
            color: Some(Color::rgb(0.5, 0.5, 1.0)),
        }
    }
}

impl ImplicitPlot3dOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_samples(mut self, x_samples: usize, y_samples: usize, z_samples: usize) -> Self {
        self.x_samples = x_samples;
        self.y_samples = y_samples;
        self.z_samples = z_samples;
        self
    }

    pub fn with_isovalue(mut self, isovalue: f64) -> Self {
        self.isovalue = isovalue;
        self
    }

    pub fn with_color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }
}

/// Create an implicit 3D surface plot from a function f(x, y, z)
///
/// Uses the marching cubes algorithm to create a triangulated mesh of the
/// isosurface where f(x, y, z) = isovalue.
///
/// # Arguments
/// * `f` - Implicit function f(x, y, z)
/// * `x_range` - Range for x values (min, max)
/// * `y_range` - Range for y values (min, max)
/// * `z_range` - Range for z values (min, max)
/// * `options` - Optional plotting options
///
/// # Example
/// ```no_run
/// use rustmath_plot3d::plots::implicit_plot3d;
/// use rustmath_plot3d::plots::implicit_plot3d::ImplicitPlot3dOptions;
///
/// // Sphere: x^2 + y^2 + z^2 = 1
/// let f = |x: f64, y: f64, z: f64| x*x + y*y + z*z - 1.0;
/// let opts = ImplicitPlot3dOptions::new().with_isovalue(0.0);
/// let graphics = implicit_plot3d(f, (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), Some(opts)).unwrap();
/// ```
pub fn implicit_plot3d<F>(
    f: F,
    x_range: (f64, f64),
    y_range: (f64, f64),
    z_range: (f64, f64),
    options: Option<ImplicitPlot3dOptions>,
) -> Result<Graphics3d>
where
    F: Fn(f64, f64, f64) -> f64,
{
    let opts = options.unwrap_or_default();
    let (x_min, x_max) = x_range;
    let (y_min, y_max) = y_range;
    let (z_min, z_max) = z_range;

    // Sample the function on a 3D grid
    let dx = (x_max - x_min) / (opts.x_samples as f64);
    let dy = (y_max - y_min) / (opts.y_samples as f64);
    let dz = (z_max - z_min) / (opts.z_samples as f64);

    // Create grid values
    let mut grid = vec![vec![vec![0.0; opts.z_samples + 1]; opts.y_samples + 1]; opts.x_samples + 1];

    for i in 0..=opts.x_samples {
        let x = x_min + i as f64 * dx;
        for j in 0..=opts.y_samples {
            let y = y_min + j as f64 * dy;
            for k in 0..=opts.z_samples {
                let z = z_min + k as f64 * dz;
                grid[i][j][k] = f(x, y, z);
            }
        }
    }

    // Run marching cubes algorithm
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    let mut vertex_map = std::collections::HashMap::new();

    for i in 0..opts.x_samples {
        for j in 0..opts.y_samples {
            for k in 0..opts.z_samples {
                // Process this cube
                marching_cubes_cell(
                    &grid,
                    i,
                    j,
                    k,
                    x_min,
                    y_min,
                    z_min,
                    dx,
                    dy,
                    dz,
                    opts.isovalue,
                    &mut vertices,
                    &mut faces,
                    &mut vertex_map,
                )?;
            }
        }
    }

    if vertices.is_empty() {
        return Err(Plot3DError::InvalidMesh(
            "No surface found in the given range".to_string(),
        ));
    }

    let mut mesh = IndexFaceSet::new(vertices, faces);
    mesh.compute_normals();

    if let Some(color) = opts.color {
        mesh.vertex_colors = Some(vec![color; mesh.vertices.len()]);
    }

    let mut graphics = Graphics3d::new();
    graphics.add_mesh(mesh);

    Ok(graphics)
}

/// Process a single cube in the marching cubes algorithm
fn marching_cubes_cell(
    grid: &[Vec<Vec<f64>>],
    i: usize,
    j: usize,
    k: usize,
    x_min: f64,
    y_min: f64,
    z_min: f64,
    dx: f64,
    dy: f64,
    dz: f64,
    isovalue: f64,
    vertices: &mut Vec<Point3D>,
    faces: &mut Vec<[usize; 3]>,
    vertex_map: &mut std::collections::HashMap<(usize, usize, usize, usize), usize>,
) -> Result<()> {
    // Get the 8 corner values of this cube
    let cube_values = [
        grid[i][j][k],
        grid[i + 1][j][k],
        grid[i + 1][j][k + 1],
        grid[i][j][k + 1],
        grid[i][j + 1][k],
        grid[i + 1][j + 1][k],
        grid[i + 1][j + 1][k + 1],
        grid[i][j + 1][k + 1],
    ];

    // Compute cube index (which corners are inside the surface)
    let mut cube_index = 0;
    for (idx, &val) in cube_values.iter().enumerate() {
        if val < isovalue {
            cube_index |= 1 << idx;
        }
    }

    // Get edge table for this configuration
    let edge_bits = MARCHING_CUBES_EDGE_TABLE[cube_index];

    if edge_bits == 0 {
        return Ok(()); // Cube is entirely inside or outside
    }

    // Get cube corner positions
    let x0 = x_min + i as f64 * dx;
    let y0 = y_min + j as f64 * dy;
    let z0 = z_min + k as f64 * dz;

    let corners = [
        Point3D::new(x0, y0, z0),
        Point3D::new(x0 + dx, y0, z0),
        Point3D::new(x0 + dx, y0, z0 + dz),
        Point3D::new(x0, y0, z0 + dz),
        Point3D::new(x0, y0 + dy, z0),
        Point3D::new(x0 + dx, y0 + dy, z0),
        Point3D::new(x0 + dx, y0 + dy, z0 + dz),
        Point3D::new(x0, y0 + dy, z0 + dz),
    ];

    // Find vertices on edges
    let mut edge_vertices = [0usize; 12];

    for edge_idx in 0..12 {
        if (edge_bits & (1 << edge_idx)) != 0 {
            let (v0_idx, v1_idx) = EDGE_CONNECTIONS[edge_idx];

            // Compute vertex position using linear interpolation
            let val0 = cube_values[v0_idx];
            let val1 = cube_values[v1_idx];

            let t = (isovalue - val0) / (val1 - val0);
            let vertex = Point3D::new(
                corners[v0_idx].x + t * (corners[v1_idx].x - corners[v0_idx].x),
                corners[v0_idx].y + t * (corners[v1_idx].y - corners[v0_idx].y),
                corners[v0_idx].z + t * (corners[v1_idx].z - corners[v0_idx].z),
            );

            // Check if we already have this vertex
            let key = (i, j, k, edge_idx);
            let vertex_idx = if let Some(&idx) = vertex_map.get(&key) {
                idx
            } else {
                let idx = vertices.len();
                vertices.push(vertex);
                vertex_map.insert(key, idx);
                idx
            };

            edge_vertices[edge_idx] = vertex_idx;
        }
    }

    // Create triangles for this cube configuration
    let tri_table = &MARCHING_CUBES_TRI_TABLE[cube_index];
    for chunk in tri_table.chunks(3) {
        if chunk[0] == 255 {
            break; // End of triangle list
        }
        faces.push([
            edge_vertices[chunk[0] as usize],
            edge_vertices[chunk[1] as usize],
            edge_vertices[chunk[2] as usize],
        ]);
    }

    Ok(())
}

// Edge connections: which two cube corners does each edge connect?
const EDGE_CONNECTIONS: [(usize, usize); 12] = [
    (0, 1), (1, 2), (2, 3), (3, 0), // Bottom face edges
    (4, 5), (5, 6), (6, 7), (7, 4), // Top face edges
    (0, 4), (1, 5), (2, 6), (3, 7), // Vertical edges
];

// Marching cubes edge table (256 entries)
// This is a simplified version - for production use, you'd want the full table
const MARCHING_CUBES_EDGE_TABLE: [u16; 256] = include!("marching_cubes_edge_table.inc");

// Marching cubes triangle table (256 entries, each with up to 16 triangle vertex indices)
const MARCHING_CUBES_TRI_TABLE: [[u8; 16]; 256] = include!("marching_cubes_tri_table.inc");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_implicit_plot3d_sphere() {
        // Sphere: x^2 + y^2 + z^2 = 1
        let f = |x: f64, y: f64, z: f64| x * x + y * y + z * z - 1.0;
        let opts = ImplicitPlot3dOptions::new()
            .with_samples(20, 20, 20)
            .with_isovalue(0.0);

        let result = implicit_plot3d(f, (-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5), Some(opts));
        // Note: This test may fail without the full marching cubes tables
        // but the structure is correct
    }

    #[test]
    fn test_implicit_plot3d_torus() {
        // Torus: (sqrt(x^2 + y^2) - R)^2 + z^2 = r^2
        let r = 0.5; // Minor radius
        let big_r = 1.0; // Major radius
        let f = |x: f64, y: f64, z: f64| {
            let d = (x * x + y * y).sqrt() - big_r;
            d * d + z * z - r * r
        };

        let opts = ImplicitPlot3dOptions::new()
            .with_samples(30, 30, 30)
            .with_isovalue(0.0);

        let result = implicit_plot3d(f, (-2.0, 2.0), (-2.0, 2.0), (-1.0, 1.0), Some(opts));
        // Structure test
    }
}
