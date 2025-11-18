//! Matrix plotting - visualizing matrices as heatmaps
//!
//! Provides matrix visualization (heatmap) functionality.

use crate::backend::ColorInterpolator;
use crate::primitives::polygon;
use crate::Graphics;
use rustmath_colors::{Color, Colormap};
use rustmath_plot_core::{PlotOptions, Point2D, Result};

/// Create a heatmap visualization of a matrix
///
/// Visualizes a 2D matrix using color intensity, where each matrix element
/// is represented as a colored cell.
///
/// # Arguments
/// * `matrix` - 2D vector representing the matrix (row-major order)
/// * `colormap` - Optional colormap (if None, uses default blue-white-red)
/// * `options` - Optional plot options
///
/// # Returns
/// A `Graphics` object containing the matrix heatmap
///
/// # Examples
/// ```ignore
/// use rustmath_plot::plots::matrix_plot;
///
/// // Simple 3x3 matrix
/// let matrix = vec![
///     vec![1.0, 2.0, 3.0],
///     vec![4.0, 5.0, 6.0],
///     vec![7.0, 8.0, 9.0],
/// ];
/// let g = matrix_plot(matrix, None, None);
///
/// // Correlation matrix
/// let corr_matrix = vec![
///     vec![1.0, 0.8, 0.3],
///     vec![0.8, 1.0, 0.5],
///     vec![0.3, 0.5, 1.0],
/// ];
/// let g = matrix_plot(corr_matrix, None, None);
/// ```
pub fn matrix_plot(
    matrix: Vec<Vec<f64>>,
    colormap: Option<Colormap>,
    options: Option<PlotOptions>,
) -> Graphics {
    if matrix.is_empty() || matrix[0].is_empty() {
        return Graphics::new();
    }

    let rows = matrix.len();
    let cols = matrix[0].len();

    // Find min and max values for normalization
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;

    for row in &matrix {
        for &val in row {
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }
    }

    let mut g = Graphics::new();

    // Create colored cells for each matrix element
    for (i, row) in matrix.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            // Normalize value to [0, 1]
            let normalized = if (max_val - min_val).abs() < 1e-10 {
                0.5
            } else {
                (val - min_val) / (max_val - min_val)
            };

            // Map to color
            let color = value_to_color(normalized);

            // Create rectangle for this cell
            // Matrix convention: (0, 0) is top-left
            // Plot convention: (0, 0) is bottom-left
            // So we flip the i-coordinate
            let x0 = j as f64;
            let y0 = (rows - i - 1) as f64;
            let x1 = x0 + 1.0;
            let y1 = y0 + 1.0;

            let rect_points = vec![
                Point2D::new(x0, y0),
                Point2D::new(x1, y0),
                Point2D::new(x1, y1),
                Point2D::new(x0, y1),
            ];

            let mut opts = PlotOptions::default();
            opts.color = color;
            opts.fill = true;

            g.add(polygon(rect_points, Some(opts)));
        }
    }

    g
}

/// Create a matrix heatmap with custom value range
///
/// Similar to `matrix_plot` but allows specifying the value range for color mapping.
///
/// # Arguments
/// * `matrix` - 2D vector representing the matrix
/// * `vmin` - Minimum value for color mapping
/// * `vmax` - Maximum value for color mapping
/// * `colormap` - Optional colormap
/// * `options` - Optional plot options
pub fn matrix_plot_range(
    matrix: Vec<Vec<f64>>,
    vmin: f64,
    vmax: f64,
    colormap: Option<Colormap>,
    options: Option<PlotOptions>,
) -> Graphics {
    if matrix.is_empty() || matrix[0].is_empty() {
        return Graphics::new();
    }

    let rows = matrix.len();
    let cols = matrix[0].len();

    let mut g = Graphics::new();

    for (i, row) in matrix.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            // Normalize value to [0, 1] using specified range
            let normalized = ((val - vmin) / (vmax - vmin)).clamp(0.0, 1.0);

            let color = value_to_color(normalized);

            let x0 = j as f64;
            let y0 = (rows - i - 1) as f64;
            let x1 = x0 + 1.0;
            let y1 = y0 + 1.0;

            let rect_points = vec![
                Point2D::new(x0, y0),
                Point2D::new(x1, y0),
                Point2D::new(x1, y1),
                Point2D::new(x0, y1),
            ];

            let mut opts = PlotOptions::default();
            opts.color = color;
            opts.fill = true;

            g.add(polygon(rect_points, Some(opts)));
        }
    }

    g
}

/// Create a binary matrix plot (0/1 values)
///
/// Optimized for visualizing binary matrices, using only two colors.
///
/// # Arguments
/// * `matrix` - 2D vector of boolean or 0/1 values
/// * `color_false` - Color for false/0 values
/// * `color_true` - Color for true/1 values
pub fn matrix_plot_binary(
    matrix: Vec<Vec<bool>>,
    color_false: Color,
    color_true: Color,
) -> Graphics {
    if matrix.is_empty() || matrix[0].is_empty() {
        return Graphics::new();
    }

    let rows = matrix.len();
    let cols = matrix[0].len();

    let mut g = Graphics::new();

    for (i, row) in matrix.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            let color = if val { color_true.clone() } else { color_false.clone() };

            let x0 = j as f64;
            let y0 = (rows - i - 1) as f64;
            let x1 = x0 + 1.0;
            let y1 = y0 + 1.0;

            let rect_points = vec![
                Point2D::new(x0, y0),
                Point2D::new(x1, y0),
                Point2D::new(x1, y1),
                Point2D::new(x0, y1),
            ];

            let mut opts = PlotOptions::default();
            opts.color = color;
            opts.fill = true;

            g.add(polygon(rect_points, Some(opts)));
        }
    }

    g
}

/// Create a spy plot showing matrix sparsity pattern
///
/// Visualizes the sparsity pattern of a matrix by showing only non-zero entries.
///
/// # Arguments
/// * `matrix` - 2D vector representing the matrix
/// * `threshold` - Values with absolute value below this are considered zero (default: 1e-10)
/// * `color` - Color for non-zero entries
pub fn matrix_plot_spy(matrix: Vec<Vec<f64>>, threshold: Option<f64>, color: Color) -> Graphics {
    if matrix.is_empty() || matrix[0].is_empty() {
        return Graphics::new();
    }

    let rows = matrix.len();
    let _cols = matrix[0].len();
    let thresh = threshold.unwrap_or(1e-10);

    let mut g = Graphics::new();

    for (i, row) in matrix.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if val.abs() > thresh {
                let x0 = j as f64;
                let y0 = (rows - i - 1) as f64;
                let x1 = x0 + 1.0;
                let y1 = y0 + 1.0;

                let rect_points = vec![
                    Point2D::new(x0, y0),
                    Point2D::new(x1, y0),
                    Point2D::new(x1, y1),
                    Point2D::new(x0, y1),
                ];

                let mut opts = PlotOptions::default();
                opts.color = color.clone();
                opts.fill = true;

                g.add(polygon(rect_points, Some(opts)));
            }
        }
    }

    g
}

/// Map a normalized value [0, 1] to a color using blue-white-red colormap
fn value_to_color(t: f64) -> Color {
    let t = ColorInterpolator::clamp01(t);

    if t < 0.5 {
        // Blue to white
        let s = t * 2.0;
        Color::rgb(s, s, 1.0)
    } else {
        // White to red
        let s = (t - 0.5) * 2.0;
        Color::rgb(1.0, 1.0 - s, 1.0 - s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_plot() {
        let matrix = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]];
        let g = matrix_plot(matrix, None, None);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_matrix_plot_single_value() {
        let matrix = vec![vec![5.0, 5.0], vec![5.0, 5.0]];
        let g = matrix_plot(matrix, None, None);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_matrix_plot_range() {
        let matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let g = matrix_plot_range(matrix, 0.0, 10.0, None, None);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_matrix_plot_binary() {
        let matrix = vec![
            vec![true, false, true],
            vec![false, true, false],
            vec![true, true, false],
        ];
        let g = matrix_plot_binary(matrix, Color::rgb(1.0, 1.0, 1.0), Color::rgb(0.0, 0.0, 0.0));
        assert!(!g.is_empty());
    }

    #[test]
    fn test_matrix_plot_spy() {
        let matrix = vec![
            vec![1.0, 0.0, 0.0, 2.0],
            vec![0.0, 3.0, 0.0, 0.0],
            vec![0.0, 0.0, 4.0, 0.0],
            vec![5.0, 0.0, 0.0, 6.0],
        ];
        let g = matrix_plot_spy(matrix, None, Color::rgb(0.0, 0.0, 0.0));
        assert!(!g.is_empty());
    }

    #[test]
    fn test_matrix_plot_empty() {
        let matrix: Vec<Vec<f64>> = vec![];
        let g = matrix_plot(matrix, None, None);
        assert!(g.is_empty());
    }

    #[test]
    fn test_value_to_color() {
        let c0 = value_to_color(0.0);
        let c1 = value_to_color(1.0);
        let c_mid = value_to_color(0.5);

        // Blue at 0
        assert_eq!(c0.blue(), 1.0);
        // Red at 1
        assert_eq!(c1.red(), 1.0);
        // White at 0.5
        assert_eq!(c_mid.red(), 1.0);
        assert_eq!(c_mid.green(), 1.0);
        assert_eq!(c_mid.blue(), 1.0);
    }
}
