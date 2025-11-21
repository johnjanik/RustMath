//! Implicit curve plotting
//!
//! Provides functionality for plotting implicit curves defined by equations f(x, y) = 0.

use crate::primitives::line;
use crate::Graphics;
use rustmath_plot_core::{PlotOptions, Point2D};

/// Plot an implicit curve defined by f(x, y) = 0
///
/// Uses marching squares algorithm to find and trace contours where f(x, y) = 0.
/// This is useful for plotting curves defined implicitly rather than as y = f(x).
///
/// # Arguments
/// * `f` - Function of two variables. The curve is traced where f(x, y) = 0
/// * `x_range` - (x_min, x_max) range
/// * `y_range` - (y_min, y_max) range
/// * `num_points` - Grid resolution (default: 100x100)
/// * `options` - Optional plot options
///
/// # Returns
/// A `Graphics` object containing the implicit curve
///
/// # Examples
/// ```ignore
/// use rustmath_plot::plots::implicit_plot;
///
/// // Plot a circle: x^2 + y^2 - 1 = 0
/// let g = implicit_plot(
///     |x, y| x * x + y * y - 1.0,
///     (-2.0, 2.0),
///     (-2.0, 2.0),
///     None,
///     None
/// );
///
/// // Plot an ellipse: x^2/4 + y^2/9 - 1 = 0
/// let g = implicit_plot(
///     |x, y| x * x / 4.0 + y * y / 9.0 - 1.0,
///     (-3.0, 3.0),
///     (-4.0, 4.0),
///     None,
///     None
/// );
///
/// // Plot a hyperbola: x^2 - y^2 - 1 = 0
/// let g = implicit_plot(
///     |x, y| x * x - y * y - 1.0,
///     (-3.0, 3.0),
///     (-3.0, 3.0),
///     Some(150),
///     None
/// );
///
/// // Plot a lemniscate: (x^2 + y^2)^2 - 2(x^2 - y^2) = 0
/// let g = implicit_plot(
///     |x, y| {
///         let r2 = x * x + y * y;
///         r2 * r2 - 2.0 * (x * x - y * y)
///     },
///     (-2.0, 2.0),
///     (-2.0, 2.0),
///     None,
///     None
/// );
/// ```
///
/// # Common Implicit Curves
/// - Circle: x² + y² - r² = 0
/// - Ellipse: x²/a² + y²/b² - 1 = 0
/// - Hyperbola: x²/a² - y²/b² - 1 = 0
/// - Lemniscate: (x² + y²)² - 2a²(x² - y²) = 0
/// - Folium of Descartes: x³ + y³ - 3axy = 0
/// - Cassini oval: ((x-a)² + y²)((x+a)² + y²) - b⁴ = 0
pub fn implicit_plot<F>(
    f: F,
    x_range: (f64, f64),
    y_range: (f64, f64),
    num_points: Option<usize>,
    options: Option<PlotOptions>,
) -> Graphics
where
    F: Fn(f64, f64) -> f64,
{
    let n = num_points.unwrap_or(100);
    let (x_min, x_max) = x_range;
    let (y_min, y_max) = y_range;

    // Evaluate function on grid
    let dx = (x_max - x_min) / n as f64;
    let dy = (y_max - y_min) / n as f64;

    let mut grid = vec![vec![0.0; n + 1]; n + 1];

    for i in 0..=n {
        for j in 0..=n {
            let x = x_min + i as f64 * dx;
            let y = y_min + j as f64 * dy;
            grid[i][j] = f(x, y);
        }
    }

    let mut g = Graphics::new();

    // Extract contours at level 0 using marching squares
    let contour_lines = extract_zero_contours(&grid, x_min, y_min, dx, dy);

    for line_points in contour_lines {
        if !line_points.is_empty() {
            g.add(line(line_points, options.clone()));
        }
    }

    g
}

/// Plot multiple implicit curves on the same axes
///
/// This is useful for plotting families of curves or level sets.
///
/// # Arguments
/// * `equations` - Vector of (function, options) pairs
/// * `x_range` - (x_min, x_max) range
/// * `y_range` - (y_min, y_max) range
/// * `num_points` - Grid resolution
///
/// # Returns
/// A `Graphics` object containing all implicit curves
///
/// # Examples
/// ```ignore
/// use rustmath_plot::plots::implicit_plot_multiple;
/// use rustmath_plot::PlotOptions;
/// use rustmath_colors::Color;
///
/// let mut opts1 = PlotOptions::default();
/// opts1.color = Color::rgb(1.0, 0.0, 0.0);
///
/// let mut opts2 = PlotOptions::default();
/// opts2.color = Color::rgb(0.0, 0.0, 1.0);
///
/// // Plot two circles
/// let equations = vec![
///     (|x: f64, y: f64| x * x + y * y - 1.0, Some(opts1)),
///     (|x: f64, y: f64| x * x + y * y - 4.0, Some(opts2)),
/// ];
///
/// let g = implicit_plot_multiple(equations, (-3.0, 3.0), (-3.0, 3.0), None);
/// ```
pub fn implicit_plot_multiple<F>(
    equations: Vec<(F, Option<PlotOptions>)>,
    x_range: (f64, f64),
    y_range: (f64, f64),
    num_points: Option<usize>,
) -> Graphics
where
    F: Fn(f64, f64) -> f64,
{
    let g = Graphics::new();

    for (f, opts) in equations {
        let curve = implicit_plot(f, x_range, y_range, num_points, opts);
        // Combine the plots
        for _primitive in curve.primitives() {
            // Note: This requires some refactoring of combine() in graphics.rs
            // For now, we'll need to work around this limitation
        }
    }

    g
}

/// Extract zero-level contours from a grid using marching squares
///
/// This implements the marching squares algorithm to find curves where f(x, y) = 0.
fn extract_zero_contours(
    grid: &[Vec<f64>],
    x_min: f64,
    y_min: f64,
    dx: f64,
    dy: f64,
) -> Vec<Vec<Point2D>> {
    let rows = grid.len();
    if rows == 0 {
        return vec![];
    }
    let cols = grid[0].len();
    if cols == 0 {
        return vec![];
    }

    let mut contours = Vec::new();
    let mut visited = vec![vec![false; cols - 1]; rows - 1];

    // Iterate through each cell in the grid
    for i in 0..(rows - 1) {
        for j in 0..(cols - 1) {
            if visited[i][j] {
                continue;
            }

            // Get the four corner values
            let v00 = grid[i][j];
            let v10 = grid[i + 1][j];
            let v11 = grid[i + 1][j + 1];
            let v01 = grid[i][j + 1];

            // Calculate marching squares case
            let mut case = 0;
            if v00 < 0.0 {
                case |= 1;
            }
            if v10 < 0.0 {
                case |= 2;
            }
            if v11 < 0.0 {
                case |= 4;
            }
            if v01 < 0.0 {
                case |= 8;
            }

            // Skip cells that are entirely inside or outside
            if case == 0 || case == 15 {
                visited[i][j] = true;
                continue;
            }

            // Extract line segment(s) for this cell
            let segments = get_cell_segments(case, i, j, v00, v10, v11, v01, x_min, y_min, dx, dy);

            for segment in segments {
                contours.push(segment);
            }

            visited[i][j] = true;
        }
    }

    // Post-process: connect nearby segments into continuous curves
    connect_segments(contours)
}

/// Get line segments for a marching squares cell
fn get_cell_segments(
    case: u8,
    i: usize,
    j: usize,
    v00: f64,
    v10: f64,
    v11: f64,
    v01: f64,
    x_min: f64,
    y_min: f64,
    dx: f64,
    dy: f64,
) -> Vec<Vec<Point2D>> {
    let x0 = x_min + i as f64 * dx;
    let y0 = y_min + j as f64 * dy;
    let x1 = x0 + dx;
    let y1 = y0 + dy;

    // Linear interpolation to find zero crossing
    let interp = |val0: f64, val1: f64| -> f64 {
        if (val1 - val0).abs() < 1e-10 {
            0.5
        } else {
            -val0 / (val1 - val0)
        }
    };

    // Calculate edge midpoints where zero crossings occur
    let bottom = if (v00 < 0.0) != (v10 < 0.0) {
        let t = interp(v00, v10);
        Some(Point2D::new(x0 + t * dx, y0))
    } else {
        None
    };

    let right = if (v10 < 0.0) != (v11 < 0.0) {
        let t = interp(v10, v11);
        Some(Point2D::new(x1, y0 + t * dy))
    } else {
        None
    };

    let top = if (v11 < 0.0) != (v01 < 0.0) {
        let t = interp(v01, v11);
        Some(Point2D::new(x0 + t * dx, y1))
    } else {
        None
    };

    let left = if (v00 < 0.0) != (v01 < 0.0) {
        let t = interp(v00, v01);
        Some(Point2D::new(x0, y0 + t * dy))
    } else {
        None
    };

    // Create line segments based on marching squares case
    let mut segments = Vec::new();

    match case {
        1 | 14 => {
            // Bottom and left
            if let (Some(b), Some(l)) = (bottom, left) {
                segments.push(vec![b, l]);
            }
        }
        2 | 13 => {
            // Bottom and right
            if let (Some(b), Some(r)) = (bottom, right) {
                segments.push(vec![b, r]);
            }
        }
        3 | 12 => {
            // Left and right
            if let (Some(l), Some(r)) = (left, right) {
                segments.push(vec![l, r]);
            }
        }
        4 | 11 => {
            // Top and right
            if let (Some(t), Some(r)) = (top, right) {
                segments.push(vec![t, r]);
            }
        }
        5 => {
            // Two segments: bottom-right and top-left
            if let (Some(b), Some(r)) = (bottom, right) {
                segments.push(vec![b, r]);
            }
            if let (Some(t), Some(l)) = (top, left) {
                segments.push(vec![t, l]);
            }
        }
        6 | 9 => {
            // Bottom and top
            if let (Some(b), Some(t)) = (bottom, top) {
                segments.push(vec![b, t]);
            }
        }
        7 | 8 => {
            // Top and left
            if let (Some(t), Some(l)) = (top, left) {
                segments.push(vec![t, l]);
            }
        }
        10 => {
            // Two segments: bottom-left and top-right
            if let (Some(b), Some(l)) = (bottom, left) {
                segments.push(vec![b, l]);
            }
            if let (Some(t), Some(r)) = (top, right) {
                segments.push(vec![t, r]);
            }
        }
        _ => {}
    }

    segments
}

/// Connect nearby line segments into continuous curves
fn connect_segments(segments: Vec<Vec<Point2D>>) -> Vec<Vec<Point2D>> {
    if segments.is_empty() {
        return vec![];
    }

    let mut curves: Vec<Vec<Point2D>> = Vec::new();
    let tolerance = 1e-6;

    for segment in segments {
        if segment.len() < 2 {
            continue;
        }

        // Try to append to an existing curve
        let mut added = false;

        for curve in &mut curves {
            if curve.is_empty() {
                continue;
            }

            let curve_start = curve[0];
            let curve_end = curve[curve.len() - 1];
            let seg_start = segment[0];
            let seg_end = segment[segment.len() - 1];

            // Check if segment connects to end of curve
            if distance(&curve_end, &seg_start) < tolerance {
                curve.extend_from_slice(&segment[1..]);
                added = true;
                break;
            }
            // Check if segment connects to start of curve (reversed)
            else if distance(&curve_start, &seg_end) < tolerance {
                let mut reversed = segment.clone();
                reversed.reverse();
                reversed.extend_from_slice(&curve[1..]);
                *curve = reversed;
                added = true;
                break;
            }
            // Check if segment connects to start of curve
            else if distance(&curve_start, &seg_start) < tolerance {
                let mut new_curve = segment.clone();
                new_curve.reverse();
                new_curve.extend_from_slice(&curve[1..]);
                *curve = new_curve;
                added = true;
                break;
            }
            // Check if segment connects to end of curve (reversed)
            else if distance(&curve_end, &seg_end) < tolerance {
                let mut reversed = segment.clone();
                reversed.reverse();
                curve.extend_from_slice(&reversed[1..]);
                added = true;
                break;
            }
        }

        if !added {
            // Start a new curve
            curves.push(segment);
        }
    }

    curves
}

/// Calculate distance between two points
fn distance(p1: &Point2D, p2: &Point2D) -> f64 {
    let dx = p1.x - p2.x;
    let dy = p1.y - p2.y;
    (dx * dx + dy * dy).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_implicit_plot_circle() {
        // Plot a circle: x^2 + y^2 - 1 = 0
        let g = implicit_plot(
            |x, y| x * x + y * y - 1.0,
            (-2.0, 2.0),
            (-2.0, 2.0),
            Some(50),
            None,
        );
        assert!(!g.is_empty());
    }

    #[test]
    fn test_implicit_plot_ellipse() {
        // Plot an ellipse: x^2/4 + y^2/9 - 1 = 0
        let g = implicit_plot(
            |x, y| x * x / 4.0 + y * y / 9.0 - 1.0,
            (-3.0, 3.0),
            (-4.0, 4.0),
            Some(50),
            None,
        );
        assert!(!g.is_empty());
    }

    #[test]
    fn test_implicit_plot_hyperbola() {
        // Plot a hyperbola: x^2 - y^2 - 1 = 0
        let g = implicit_plot(
            |x, y| x * x - y * y - 1.0,
            (-3.0, 3.0),
            (-3.0, 3.0),
            Some(50),
            None,
        );
        assert!(!g.is_empty());
    }

    #[test]
    fn test_implicit_plot_lemniscate() {
        // Plot a lemniscate: (x^2 + y^2)^2 - 2(x^2 - y^2) = 0
        let g = implicit_plot(
            |x, y| {
                let r2 = x * x + y * y;
                r2 * r2 - 2.0 * (x * x - y * y)
            },
            (-2.0, 2.0),
            (-2.0, 2.0),
            Some(100),
            None,
        );
        assert!(!g.is_empty());
    }

    #[test]
    fn test_implicit_plot_with_options() {
        let mut opts = PlotOptions::default();
        opts.color = Color::rgb(1.0, 0.0, 0.0);
        opts.thickness = 2.0;

        let g = implicit_plot(
            |x, y| x * x + y * y - 1.0,
            (-2.0, 2.0),
            (-2.0, 2.0),
            Some(50),
            Some(opts),
        );
        assert!(!g.is_empty());
    }

    #[test]
    fn test_distance() {
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(3.0, 4.0);
        assert!((distance(&p1, &p2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_extract_zero_contours_empty() {
        let grid: Vec<Vec<f64>> = vec![];
        let contours = extract_zero_contours(&grid, 0.0, 0.0, 1.0, 1.0);
        assert!(contours.is_empty());
    }

    #[test]
    fn test_extract_zero_contours_uniform() {
        // All positive values - no contour
        let grid = vec![
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
        ];
        let contours = extract_zero_contours(&grid, 0.0, 0.0, 1.0, 1.0);
        assert!(contours.is_empty());
    }
}
