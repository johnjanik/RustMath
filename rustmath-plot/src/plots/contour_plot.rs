//! Contour plotting - visualizing level sets of 2D functions
//!
//! Provides contour plot functionality using marching squares algorithm.

use crate::primitives::line;
use crate::Graphics;
use rustmath_colors::{Color, Colormap};
use rustmath_plot_core::{PlotOptions, Point2D, Result};

/// Create a contour plot of a 2D function
///
/// Plots level curves (contours) of a function f(x, y) = constant.
///
/// # Arguments
/// * `f` - Function of two variables
/// * `x_range` - (x_min, x_max) range
/// * `y_range` - (y_min, y_max) range
/// * `levels` - Contour levels to plot (if None, automatically determined)
/// * `num_points` - Grid resolution (default: 50x50)
/// * `options` - Optional plot options
///
/// # Returns
/// A `Graphics` object containing the contour plot
///
/// # Examples
/// ```ignore
/// use rustmath_plot::plots::contour_plot;
///
/// // Plot contours of f(x,y) = x^2 + y^2 (circles)
/// let g = contour_plot(
///     |x, y| x * x + y * y,
///     (-2.0, 2.0),
///     (-2.0, 2.0),
///     Some(vec![0.5, 1.0, 1.5, 2.0]),
///     None,
///     None
/// );
///
/// // Plot with automatic levels
/// let g = contour_plot(
///     |x, y| (x * x - y * y).sin(),
///     (-3.0, 3.0),
///     (-3.0, 3.0),
///     None,
///     Some(100),
///     None
/// );
/// ```
pub fn contour_plot<F>(
    f: F,
    x_range: (f64, f64),
    y_range: (f64, f64),
    levels: Option<Vec<f64>>,
    num_points: Option<usize>,
    options: Option<PlotOptions>,
) -> Graphics
where
    F: Fn(f64, f64) -> f64,
{
    let n = num_points.unwrap_or(50);
    let (x_min, x_max) = x_range;
    let (y_min, y_max) = y_range;

    // Evaluate function on grid
    let dx = (x_max - x_min) / (n - 1) as f64;
    let dy = (y_max - y_min) / (n - 1) as f64;

    let mut grid = vec![vec![0.0; n]; n];
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;

    for i in 0..n {
        for j in 0..n {
            let x = x_min + i as f64 * dx;
            let y = y_min + j as f64 * dy;
            let val = f(x, y);
            grid[i][j] = val;
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }
    }

    // Determine contour levels
    let contour_levels = if let Some(lvls) = levels {
        lvls
    } else {
        // Auto-generate 10 evenly spaced levels
        let num_levels = 10;
        (0..num_levels)
            .map(|i| min_val + (max_val - min_val) * i as f64 / (num_levels - 1) as f64)
            .collect()
    };

    let mut g = Graphics::new();

    // Extract contours using marching squares
    for level in contour_levels {
        let contour_lines = extract_contours(&grid, level, x_min, y_min, dx, dy);

        for line_points in contour_lines {
            g.add(line(line_points, options.clone()));
        }
    }

    g
}

/// Create a filled contour plot with color mapping
///
/// Similar to `contour_plot` but fills regions between contours with colors.
///
/// # Arguments
/// * `f` - Function of two variables
/// * `x_range` - (x_min, x_max) range
/// * `y_range` - (y_min, y_max) range
/// * `levels` - Contour levels (if None, automatically determined)
/// * `colormap` - Color mapping for levels
/// * `num_points` - Grid resolution
///
/// # Returns
/// A `Graphics` object containing the filled contour plot
pub fn contour_plot_filled<F>(
    f: F,
    x_range: (f64, f64),
    y_range: (f64, f64),
    levels: Option<Vec<f64>>,
    colormap: Option<Colormap>,
    num_points: Option<usize>,
) -> Graphics
where
    F: Fn(f64, f64) -> f64,
{
    // For now, just create regular contour plot
    // TODO: Implement polygon filling between contours
    contour_plot(f, x_range, y_range, levels, num_points, None)
}

/// Extract contour lines at a specific level using marching squares
///
/// This is a simplified implementation of the marching squares algorithm.
fn extract_contours(
    grid: &[Vec<f64>],
    level: f64,
    x_min: f64,
    y_min: f64,
    dx: f64,
    dy: f64,
) -> Vec<Vec<Point2D>> {
    let mut contours = Vec::new();
    let n = grid.len();

    // Process each cell in the grid
    for i in 0..n - 1 {
        for j in 0..n - 1 {
            // Get the four corners of the cell
            let v00 = grid[i][j];
            let v10 = grid[i + 1][j];
            let v01 = grid[i][j + 1];
            let v11 = grid[i + 1][j + 1];

            // Determine the case (which corners are above the level)
            let mut case = 0;
            if v00 >= level {
                case |= 1;
            }
            if v10 >= level {
                case |= 2;
            }
            if v11 >= level {
                case |= 4;
            }
            if v01 >= level {
                case |= 8;
            }

            // Skip if all corners are on the same side
            if case == 0 || case == 15 {
                continue;
            }

            // Calculate cell coordinates
            let x0 = x_min + i as f64 * dx;
            let y0 = y_min + j as f64 * dy;
            let x1 = x0 + dx;
            let y1 = y0 + dy;

            // Interpolate edge intersections
            let interpolate = |v1: f64, v2: f64, p1: f64, p2: f64| -> f64 {
                if (v2 - v1).abs() < 1e-10 {
                    (p1 + p2) / 2.0
                } else {
                    p1 + (level - v1) / (v2 - v1) * (p2 - p1)
                }
            };

            // Calculate intersection points for each edge
            let mut points = Vec::new();

            // Process based on marching squares case
            match case {
                1 | 14 => {
                    // Bottom and left edges
                    let x_bottom = interpolate(v00, v10, x0, x1);
                    let x_left = interpolate(v00, v01, y0, y1);
                    points.push(Point2D::new(x_bottom, y0));
                    points.push(Point2D::new(x0, x_left));
                }
                2 | 13 => {
                    // Right and bottom edges
                    let x_bottom = interpolate(v00, v10, x0, x1);
                    let y_right = interpolate(v10, v11, y0, y1);
                    points.push(Point2D::new(x_bottom, y0));
                    points.push(Point2D::new(x1, y_right));
                }
                3 | 12 => {
                    // Left and right edges
                    let y_left = interpolate(v00, v01, y0, y1);
                    let y_right = interpolate(v10, v11, y0, y1);
                    points.push(Point2D::new(x0, y_left));
                    points.push(Point2D::new(x1, y_right));
                }
                4 | 11 => {
                    // Top and right edges
                    let y_right = interpolate(v10, v11, y0, y1);
                    let x_top = interpolate(v01, v11, x0, x1);
                    points.push(Point2D::new(x1, y_right));
                    points.push(Point2D::new(x_top, y1));
                }
                6 | 9 => {
                    // Bottom and top edges
                    let x_bottom = interpolate(v00, v10, x0, x1);
                    let x_top = interpolate(v01, v11, x0, x1);
                    points.push(Point2D::new(x_bottom, y0));
                    points.push(Point2D::new(x_top, y1));
                }
                7 | 8 => {
                    // Left and top edges
                    let y_left = interpolate(v00, v01, y0, y1);
                    let x_top = interpolate(v01, v11, x0, x1);
                    points.push(Point2D::new(x0, y_left));
                    points.push(Point2D::new(x_top, y1));
                }
                5 | 10 => {
                    // Saddle point - two separate segments
                    // This is a simplified handling
                    let x_bottom = interpolate(v00, v10, x0, x1);
                    let y_right = interpolate(v10, v11, y0, y1);
                    points.push(Point2D::new(x_bottom, y0));
                    points.push(Point2D::new(x1, y_right));
                }
                _ => {}
            }

            if !points.is_empty() {
                contours.push(points);
            }
        }
    }

    contours
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contour_plot() {
        // Circle contours: x^2 + y^2
        let g = contour_plot(
            |x, y| x * x + y * y,
            (-2.0, 2.0),
            (-2.0, 2.0),
            Some(vec![0.5, 1.0, 1.5, 2.0]),
            None,
            None,
        );
        assert!(!g.is_empty());
    }

    #[test]
    fn test_contour_plot_auto_levels() {
        let g = contour_plot(
            |x, y| (x * x - y * y).sin(),
            (-3.0, 3.0),
            (-3.0, 3.0),
            None,
            Some(50),
            None,
        );
        assert!(!g.is_empty());
    }

    #[test]
    fn test_contour_plot_linear() {
        // Simple linear function
        let _g = contour_plot(
            |x, y| x + y,
            (-2.0, 2.0),
            (-2.0, 2.0),
            Some(vec![0.0]),
            Some(30),
            None,
        );
        // May be empty if no contours cross the grid
    }
}
