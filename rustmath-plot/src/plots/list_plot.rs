//! List plotting - visualizing discrete data points
//!
//! Provides functions for plotting lists of data points.

use crate::primitives::{line, point};
use crate::Graphics;
use rustmath_plot_core::{PlotOptions, Point2D, Result};

/// Plot a list of data points
///
/// Creates a plot from a list of (x, y) coordinates. By default, points are connected
/// with lines, but this can be controlled with the `plotjoined` parameter.
///
/// # Arguments
/// * `data` - Vector of (x, y) coordinates
/// * `plotjoined` - If true, connect points with lines (default: true)
/// * `options` - Optional plot options
///
/// # Returns
/// A `Graphics` object containing the plotted data
///
/// # Examples
/// ```ignore
/// use rustmath_plot::plots::list_plot;
///
/// // Plot discrete points connected by lines
/// let data = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 4.0), (3.0, 9.0)];
/// let g = list_plot(data, None, None);
///
/// // Plot points without connecting lines
/// let data = vec![(0.0, 1.0), (1.0, 2.0), (2.0, 1.5), (3.0, 3.0)];
/// let g = list_plot(data, Some(false), None);
///
/// // Plot with custom options
/// use rustmath_plot::PlotOptions;
/// use rustmath_colors::Color;
///
/// let mut opts = PlotOptions::default();
/// opts.color = Color::rgb(1.0, 0.0, 0.0);
/// opts.thickness = 2.0;
///
/// let data = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)];
/// let g = list_plot(data, None, Some(opts));
/// ```
pub fn list_plot(
    data: Vec<(f64, f64)>,
    plotjoined: Option<bool>,
    options: Option<PlotOptions>,
) -> Graphics {
    let mut g = Graphics::new();

    // Don't add anything if data is empty
    if data.is_empty() {
        return g;
    }

    let joined = plotjoined.unwrap_or(true);
    let points: Vec<Point2D> = data.iter().map(|(x, y)| Point2D::new(*x, *y)).collect();

    if joined {
        // Connect points with lines
        g.add(line(points, options));
    } else {
        // Plot as discrete points
        g.add(point(points, options));
    }

    g
}

/// Plot a list of y-values with automatic x-coordinates
///
/// Creates a plot from a list of y-values, using 0, 1, 2, ... as x-coordinates.
///
/// # Arguments
/// * `y_values` - Vector of y-values
/// * `plotjoined` - If true, connect points with lines (default: true)
/// * `options` - Optional plot options
///
/// # Returns
/// A `Graphics` object containing the plotted data
///
/// # Examples
/// ```ignore
/// use rustmath_plot::plots::list_plot_y;
///
/// // Plot y-values with automatic x-coordinates
/// let y_values = vec![1.0, 4.0, 2.0, 8.0, 5.0, 7.0];
/// let g = list_plot_y(y_values, None, None);
/// ```
pub fn list_plot_y(
    y_values: Vec<f64>,
    plotjoined: Option<bool>,
    options: Option<PlotOptions>,
) -> Graphics {
    let data: Vec<(f64, f64)> = y_values
        .iter()
        .enumerate()
        .map(|(i, &y)| (i as f64, y))
        .collect();

    list_plot(data, plotjoined, options)
}

/// Plot multiple lists on the same axes
///
/// Creates a plot with multiple data series.
///
/// # Arguments
/// * `datasets` - Vector of (data, plotjoined, options) tuples
///
/// # Returns
/// A `Graphics` object containing all plotted datasets
///
/// # Examples
/// ```ignore
/// use rustmath_plot::plots::list_plot_multiple;
/// use rustmath_plot::PlotOptions;
/// use rustmath_colors::Color;
///
/// let mut opts1 = PlotOptions::default();
/// opts1.color = Color::rgb(1.0, 0.0, 0.0);
///
/// let mut opts2 = PlotOptions::default();
/// opts2.color = Color::rgb(0.0, 0.0, 1.0);
///
/// let data1 = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)];
/// let data2 = vec![(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)];
///
/// let datasets = vec![
///     (data1, None, Some(opts1)),
///     (data2, None, Some(opts2)),
/// ];
///
/// let g = list_plot_multiple(datasets);
/// ```
pub fn list_plot_multiple(
    datasets: Vec<(Vec<(f64, f64)>, Option<bool>, Option<PlotOptions>)>,
) -> Graphics {
    let mut g = Graphics::new();

    for (data, plotjoined, options) in datasets {
        let joined = plotjoined.unwrap_or(true);
        let points: Vec<Point2D> = data.iter().map(|(x, y)| Point2D::new(*x, *y)).collect();

        if joined {
            g.add(line(points, options));
        } else {
            g.add(point(points, options));
        }
    }

    g
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_colors::Color;

    #[test]
    fn test_list_plot_joined() {
        let data = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 4.0), (3.0, 9.0)];
        let g = list_plot(data, None, None);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_list_plot_not_joined() {
        let data = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 4.0), (3.0, 9.0)];
        let g = list_plot(data, Some(false), None);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_list_plot_with_options() {
        let mut opts = PlotOptions::default();
        opts.color = Color::rgb(1.0, 0.0, 0.0);
        opts.thickness = 2.0;

        let data = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)];
        let g = list_plot(data, None, Some(opts));
        assert!(!g.is_empty());
    }

    #[test]
    fn test_list_plot_y() {
        let y_values = vec![1.0, 4.0, 2.0, 8.0, 5.0, 7.0];
        let g = list_plot_y(y_values, None, None);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_list_plot_empty() {
        let data: Vec<(f64, f64)> = vec![];
        let g = list_plot(data, None, None);
        // Empty data should still create a graphics object
        assert!(g.is_empty());
    }

    #[test]
    fn test_list_plot_multiple() {
        let mut opts1 = PlotOptions::default();
        opts1.color = Color::rgb(1.0, 0.0, 0.0);

        let mut opts2 = PlotOptions::default();
        opts2.color = Color::rgb(0.0, 0.0, 1.0);

        let data1 = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)];
        let data2 = vec![(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)];

        let datasets = vec![(data1, None, Some(opts1)), (data2, None, Some(opts2))];

        let g = list_plot_multiple(datasets);
        assert!(!g.is_empty());
    }
}
