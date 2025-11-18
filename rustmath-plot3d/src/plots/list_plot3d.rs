//! 3D plots from lists of data points

use crate::base::{Graphics3d, IndexFaceSet, Point3D};
use crate::Result;
use crate::Plot3DError;
use rustmath_colors::Color;

/// Options for 3D list plots
#[derive(Debug, Clone)]
pub struct ListPlot3dOptions {
    /// Color of the surface
    pub color: Option<Color>,
    /// Whether to interpolate between points
    pub interpolate: bool,
}

impl Default for ListPlot3dOptions {
    fn default() -> Self {
        Self {
            color: None,
            interpolate: true,
        }
    }
}

impl ListPlot3dOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }

    pub fn with_interpolate(mut self, interpolate: bool) -> Self {
        self.interpolate = interpolate;
        self
    }
}

/// Create a 3D surface plot from a 2D grid of z values
///
/// # Arguments
/// * `z_data` - 2D array of z values (rows x columns)
/// * `x_coords` - Optional x coordinates (if None, use indices)
/// * `y_coords` - Optional y coordinates (if None, use indices)
/// * `options` - Optional plotting options
///
/// # Example
/// ```no_run
/// use rustmath_plot3d::plots::list_plot3d;
///
/// let z_data = vec![
///     vec![0.0, 1.0, 2.0],
///     vec![1.0, 2.0, 3.0],
///     vec![2.0, 3.0, 4.0],
/// ];
/// let graphics = list_plot3d(&z_data, None, None, None).unwrap();
/// ```
pub fn list_plot3d(
    z_data: &[Vec<f64>],
    x_coords: Option<&[f64]>,
    y_coords: Option<&[f64]>,
    options: Option<ListPlot3dOptions>,
) -> Result<Graphics3d> {
    let opts = options.unwrap_or_default();

    if z_data.is_empty() {
        return Err(Plot3DError::InvalidMesh("Empty z_data".to_string()));
    }

    let num_rows = z_data.len();
    let num_cols = z_data[0].len();

    // Validate that all rows have the same length
    for row in z_data.iter() {
        if row.len() != num_cols {
            return Err(Plot3DError::InvalidMesh(
                "All rows must have the same length".to_string(),
            ));
        }
    }

    // Generate x and y coordinates if not provided
    let x_vals: Vec<f64> = if let Some(x) = x_coords {
        if x.len() != num_cols {
            return Err(Plot3DError::InvalidMesh(
                "x_coords length must match number of columns".to_string(),
            ));
        }
        x.to_vec()
    } else {
        (0..num_cols).map(|i| i as f64).collect()
    };

    let y_vals: Vec<f64> = if let Some(y) = y_coords {
        if y.len() != num_rows {
            return Err(Plot3DError::InvalidMesh(
                "y_coords length must match number of rows".to_string(),
            ));
        }
        y.to_vec()
    } else {
        (0..num_rows).map(|i| i as f64).collect()
    };

    let mut vertices = Vec::new();
    let mut faces = Vec::new();

    // Generate vertices
    for (i, y) in y_vals.iter().enumerate() {
        for (j, x) in x_vals.iter().enumerate() {
            let z = z_data[i][j];
            vertices.push(Point3D { x: *x, y: *y, z });
        }
    }

    // Generate faces (two triangles per grid cell)
    for i in 0..(num_rows - 1) {
        for j in 0..(num_cols - 1) {
            let idx0 = i * num_cols + j;
            let idx1 = idx0 + 1;
            let idx2 = (i + 1) * num_cols + j;
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

/// Create a 3D scatter plot from individual (x, y, z) points
///
/// # Arguments
/// * `points` - List of (x, y, z) tuples
/// * `options` - Optional plotting options
///
/// # Example
/// ```no_run
/// use rustmath_plot3d::plots::list_plot3d::scatter_plot3d;
///
/// let points = vec![
///     (0.0, 0.0, 0.0),
///     (1.0, 0.0, 1.0),
///     (0.0, 1.0, 1.0),
///     (1.0, 1.0, 2.0),
/// ];
/// let graphics = scatter_plot3d(&points, None).unwrap();
/// ```
pub fn scatter_plot3d(
    points: &[(f64, f64, f64)],
    options: Option<ListPlot3dOptions>,
) -> Result<Graphics3d> {
    let opts = options.unwrap_or_default();

    if points.is_empty() {
        return Err(Plot3DError::InvalidMesh("Empty points list".to_string()));
    }

    // For scatter plot, we'll create small spheres at each point
    // For simplicity, we'll just create a point cloud for now
    // In a full implementation, you'd use actual sphere primitives

    let mut graphics = Graphics3d::new();

    // TODO: Implement actual point/sphere rendering
    // For now, just return an empty graphics object as a placeholder

    if let Some(color) = opts.color {
        graphics.options.default_color = Some(color);
    }

    Ok(graphics)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_plot3d_simple() {
        let z_data = vec![vec![0.0, 1.0], vec![1.0, 2.0]];
        let result = list_plot3d(&z_data, None, None, None);
        assert!(result.is_ok());
        let graphics = result.unwrap();
        assert_eq!(graphics.objects.len(), 1);
    }

    #[test]
    fn test_list_plot3d_with_coords() {
        let z_data = vec![vec![0.0, 1.0, 2.0], vec![1.0, 2.0, 3.0]];
        let x_coords = vec![0.0, 0.5, 1.0];
        let y_coords = vec![0.0, 1.0];
        let result = list_plot3d(&z_data, Some(&x_coords), Some(&y_coords), None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_list_plot3d_empty_data() {
        let z_data: Vec<Vec<f64>> = vec![];
        let result = list_plot3d(&z_data, None, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_list_plot3d_mismatched_rows() {
        let z_data = vec![vec![0.0, 1.0], vec![1.0, 2.0, 3.0]];
        let result = list_plot3d(&z_data, None, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_scatter_plot3d_simple() {
        let points = vec![(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)];
        let result = scatter_plot3d(&points, None);
        assert!(result.is_ok());
    }
}
