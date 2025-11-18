//! Density plotting - visualizing 2D functions with color intensity
//!
//! Provides density plot (heatmap) functionality for 2D functions.

use crate::backend::ColorInterpolator;
use crate::primitives::polygon;
use crate::Graphics;
use rustmath_colors::{Color, Colormap};
use rustmath_plot_core::{PlotOptions, Point2D, Result};

/// Create a density plot of a 2D function
///
/// Visualizes a function f(x, y) using color intensity, creating a heatmap.
///
/// # Arguments
/// * `f` - Function of two variables
/// * `x_range` - (x_min, x_max) range
/// * `y_range` - (y_min, y_max) range
/// * `num_points` - Grid resolution (default: 50x50)
/// * `colormap` - Color mapping (if None, uses default blue-white-red)
/// * `options` - Optional plot options
///
/// # Returns
/// A `Graphics` object containing the density plot
///
/// # Examples
/// ```ignore
/// use rustmath_plot::plots::density_plot;
///
/// // Plot density of f(x,y) = sin(x) * cos(y)
/// let g = density_plot(
///     |x, y| (x.sin() * y.cos()),
///     (-3.0, 3.0),
///     (-3.0, 3.0),
///     None,
///     None,
///     None
/// );
///
/// // High-resolution density plot
/// let g = density_plot(
///     |x, y| (-x * x - y * y).exp(),
///     (-2.0, 2.0),
///     (-2.0, 2.0),
///     Some(100),
///     None,
///     None
/// );
/// ```
pub fn density_plot<F>(
    f: F,
    x_range: (f64, f64),
    y_range: (f64, f64),
    num_points: Option<usize>,
    colormap: Option<Colormap>,
    options: Option<PlotOptions>,
) -> Graphics
where
    F: Fn(f64, f64) -> f64,
{
    let n = num_points.unwrap_or(50);
    let (x_min, x_max) = x_range;
    let (y_min, y_max) = y_range;

    // Evaluate function on grid
    let dx = (x_max - x_min) / n as f64;
    let dy = (y_max - y_min) / n as f64;

    let mut grid = vec![vec![0.0; n]; n];
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;

    for i in 0..n {
        for j in 0..n {
            let x = x_min + (i as f64 + 0.5) * dx;
            let y = y_min + (j as f64 + 0.5) * dy;
            let val = f(x, y);
            grid[i][j] = val;
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }
    }

    let mut g = Graphics::new();

    // Create colored rectangles for each grid cell
    for i in 0..n {
        for j in 0..n {
            let val = grid[i][j];

            // Normalize value to [0, 1]
            let normalized = if (max_val - min_val).abs() < 1e-10 {
                0.5
            } else {
                (val - min_val) / (max_val - min_val)
            };

            // Map to color
            let color = value_to_color(normalized);

            // Create rectangle for this cell
            let x0 = x_min + i as f64 * dx;
            let y0 = y_min + j as f64 * dy;
            let x1 = x0 + dx;
            let y1 = y0 + dy;

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

/// Map a normalized value [0, 1] to a color
///
/// Uses a simple blue-white-red colormap:
/// - 0.0 → Blue (cold)
/// - 0.5 → White (neutral)
/// - 1.0 → Red (hot)
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

/// Create a density plot with a custom color mapping function
///
/// # Arguments
/// * `f` - Function of two variables
/// * `x_range` - (x_min, x_max) range
/// * `y_range` - (y_min, y_max) range
/// * `num_points` - Grid resolution
/// * `color_fn` - Function mapping values to colors
/// * `options` - Optional plot options
pub fn density_plot_custom<F, C>(
    f: F,
    x_range: (f64, f64),
    y_range: (f64, f64),
    num_points: Option<usize>,
    color_fn: C,
    options: Option<PlotOptions>,
) -> Graphics
where
    F: Fn(f64, f64) -> f64,
    C: Fn(f64) -> Color,
{
    let n = num_points.unwrap_or(50);
    let (x_min, x_max) = x_range;
    let (y_min, y_max) = y_range;

    let dx = (x_max - x_min) / n as f64;
    let dy = (y_max - y_min) / n as f64;

    let mut grid = vec![vec![0.0; n]; n];
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;

    for i in 0..n {
        for j in 0..n {
            let x = x_min + (i as f64 + 0.5) * dx;
            let y = y_min + (j as f64 + 0.5) * dy;
            let val = f(x, y);
            grid[i][j] = val;
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }
    }

    let mut g = Graphics::new();

    for i in 0..n {
        for j in 0..n {
            let val = grid[i][j];

            // Normalize value
            let normalized = if (max_val - min_val).abs() < 1e-10 {
                0.5
            } else {
                (val - min_val) / (max_val - min_val)
            };

            let color = color_fn(normalized);

            let x0 = x_min + i as f64 * dx;
            let y0 = y_min + j as f64 * dy;
            let x1 = x0 + dx;
            let y1 = y0 + dy;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_density_plot() {
        let g = density_plot(
            |x, y| x.sin() * y.cos(),
            (-3.0, 3.0),
            (-3.0, 3.0),
            Some(20),
            None,
            None,
        );
        assert!(!g.is_empty());
    }

    #[test]
    fn test_density_plot_gaussian() {
        // 2D Gaussian
        let g = density_plot(
            |x, y| (-x * x - y * y).exp(),
            (-2.0, 2.0),
            (-2.0, 2.0),
            Some(30),
            None,
            None,
        );
        assert!(!g.is_empty());
    }

    #[test]
    fn test_density_plot_constant() {
        // Constant function should still work
        let g = density_plot(|_x, _y| 1.0, (-1.0, 1.0), (-1.0, 1.0), Some(10), None, None);
        assert!(!g.is_empty());
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

    #[test]
    fn test_density_plot_custom() {
        // Custom grayscale colormap
        let g = density_plot_custom(
            |x, y| x * x + y * y,
            (-2.0, 2.0),
            (-2.0, 2.0),
            Some(20),
            |t| Color::rgb(t, t, t),
            None,
        );
        assert!(!g.is_empty());
    }
}
