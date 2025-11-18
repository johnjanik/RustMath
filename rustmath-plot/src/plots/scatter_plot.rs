//! Scatter plot - visualizing point data without connecting lines
//!
//! Provides scatter plot functionality for visualizing discrete data points.

use crate::primitives::point;
use crate::Graphics;
use rustmath_colors::Color;
use rustmath_plot_core::{MarkerStyle, PlotOptions, Point2D, Result};

/// Create a scatter plot from data points
///
/// A scatter plot displays data as a collection of points, with optional marker styles.
/// Unlike `list_plot`, scatter plots emphasize individual points rather than trends.
///
/// # Arguments
/// * `data` - Vector of (x, y) coordinates
/// * `marker` - Optional marker style (default: circle)
/// * `size` - Optional marker size (default: 5.0)
/// * `options` - Optional plot options
///
/// # Returns
/// A `Graphics` object containing the scatter plot
///
/// # Examples
/// ```ignore
/// use rustmath_plot::plots::scatter_plot;
/// use rustmath_plot::MarkerStyle;
///
/// // Basic scatter plot
/// let data = vec![(0.0, 1.2), (1.0, 2.3), (2.0, 1.8), (3.0, 3.1)];
/// let g = scatter_plot(data, None, None, None);
///
/// // Scatter plot with custom marker
/// let data = vec![(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)];
/// let g = scatter_plot(data, Some(MarkerStyle::Square), Some(8.0), None);
///
/// // Scatter plot with colors
/// use rustmath_colors::Color;
/// use rustmath_plot::PlotOptions;
///
/// let mut opts = PlotOptions::default();
/// opts.color = Color::rgb(1.0, 0.0, 0.0);
///
/// let data = vec![(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)];
/// let g = scatter_plot(data, None, None, Some(opts));
/// ```
pub fn scatter_plot(
    data: Vec<(f64, f64)>,
    marker: Option<MarkerStyle>,
    size: Option<f64>,
    options: Option<PlotOptions>,
) -> Graphics {
    let points: Vec<Point2D> = data.iter().map(|(x, y)| Point2D::new(*x, *y)).collect();

    let mut opts = options.unwrap_or_default();
    opts.marker = marker.unwrap_or(MarkerStyle::Circle);
    opts.marker_size = size.unwrap_or(5.0);

    let mut g = Graphics::new();
    g.add(point(points, Some(opts)));

    g
}

/// Create a scatter plot with automatic x-coordinates
///
/// Creates a scatter plot from y-values, using 0, 1, 2, ... as x-coordinates.
///
/// # Arguments
/// * `y_values` - Vector of y-values
/// * `marker` - Optional marker style
/// * `size` - Optional marker size
/// * `options` - Optional plot options
///
/// # Returns
/// A `Graphics` object containing the scatter plot
pub fn scatter_plot_y(
    y_values: Vec<f64>,
    marker: Option<MarkerStyle>,
    size: Option<f64>,
    options: Option<PlotOptions>,
) -> Graphics {
    let data: Vec<(f64, f64)> = y_values
        .iter()
        .enumerate()
        .map(|(i, &y)| (i as f64, y))
        .collect();

    scatter_plot(data, marker, size, options)
}

/// Create a scatter plot with per-point colors
///
/// Creates a scatter plot where each point can have its own color.
///
/// # Arguments
/// * `data` - Vector of (x, y, color) tuples
/// * `marker` - Optional marker style
/// * `size` - Optional marker size
///
/// # Returns
/// A `Graphics` object containing the scatter plot
///
/// # Examples
/// ```ignore
/// use rustmath_plot::plots::scatter_plot_colored;
/// use rustmath_colors::Color;
///
/// let data = vec![
///     (0.0, 1.0, Color::rgb(1.0, 0.0, 0.0)), // Red
///     (1.0, 2.0, Color::rgb(0.0, 1.0, 0.0)), // Green
///     (2.0, 1.5, Color::rgb(0.0, 0.0, 1.0)), // Blue
/// ];
/// let g = scatter_plot_colored(data, None, None);
/// ```
pub fn scatter_plot_colored(
    data: Vec<(f64, f64, Color)>,
    marker: Option<MarkerStyle>,
    size: Option<f64>,
) -> Graphics {
    let mut g = Graphics::new();

    // Plot each point individually with its color
    for (x, y, color) in data {
        let mut opts = PlotOptions::default();
        opts.color = color;
        opts.marker = marker.unwrap_or(MarkerStyle::Circle);
        opts.marker_size = size.unwrap_or(5.0);

        g.add(point(vec![Point2D::new(x, y)], Some(opts)));
    }

    g
}

/// Create a scatter plot with size-varying markers
///
/// Creates a scatter plot where marker size represents a third dimension.
///
/// # Arguments
/// * `data` - Vector of (x, y, size) tuples
/// * `marker` - Optional marker style
/// * `options` - Optional plot options (color, etc.)
///
/// # Returns
/// A `Graphics` object containing the scatter plot
///
/// # Examples
/// ```ignore
/// use rustmath_plot::plots::scatter_plot_sized;
///
/// // Bubble chart - size represents a third variable
/// let data = vec![
///     (0.0, 1.0, 5.0),
///     (1.0, 2.0, 10.0),
///     (2.0, 1.5, 15.0),
/// ];
/// let g = scatter_plot_sized(data, None, None);
/// ```
pub fn scatter_plot_sized(
    data: Vec<(f64, f64, f64)>,
    marker: Option<MarkerStyle>,
    options: Option<PlotOptions>,
) -> Graphics {
    let mut g = Graphics::new();
    let base_opts = options.unwrap_or_default();

    // Plot each point with its specific size
    for (x, y, size) in data {
        let mut opts = base_opts.clone();
        opts.marker = marker.unwrap_or(MarkerStyle::Circle);
        opts.marker_size = size;

        g.add(point(vec![Point2D::new(x, y)], Some(opts)));
    }

    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scatter_plot() {
        let data = vec![(0.0, 1.2), (1.0, 2.3), (2.0, 1.8), (3.0, 3.1)];
        let g = scatter_plot(data, None, None, None);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_scatter_plot_with_marker() {
        let data = vec![(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)];
        let g = scatter_plot(data, Some(MarkerStyle::Square), Some(8.0), None);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_scatter_plot_with_options() {
        let mut opts = PlotOptions::default();
        opts.color = Color::rgb(1.0, 0.0, 0.0);

        let data = vec![(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)];
        let g = scatter_plot(data, None, None, Some(opts));
        assert!(!g.is_empty());
    }

    #[test]
    fn test_scatter_plot_y() {
        let y_values = vec![1.0, 2.0, 1.5, 3.0, 2.5];
        let g = scatter_plot_y(y_values, None, None, None);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_scatter_plot_colored() {
        let data = vec![
            (0.0, 1.0, Color::rgb(1.0, 0.0, 0.0)),
            (1.0, 2.0, Color::rgb(0.0, 1.0, 0.0)),
            (2.0, 1.5, Color::rgb(0.0, 0.0, 1.0)),
        ];
        let g = scatter_plot_colored(data, None, None);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_scatter_plot_sized() {
        let data = vec![(0.0, 1.0, 5.0), (1.0, 2.0, 10.0), (2.0, 1.5, 15.0)];
        let g = scatter_plot_sized(data, None, None);
        assert!(!g.is_empty());
    }
}
