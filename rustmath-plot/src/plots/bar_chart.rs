//! Bar chart plotting - visualizing categorical data
//!
//! Provides bar chart functionality for visualizing categorical or discrete data.

use crate::primitives::polygon;
use crate::Graphics;
use rustmath_colors::Color;
use rustmath_plot_core::{PlotOptions, Point2D};

/// Create a vertical bar chart
///
/// Visualizes data as vertical bars, useful for comparing categorical data.
///
/// # Arguments
/// * `data` - Vector of (label_position, value) pairs
/// * `bar_width` - Width of each bar (default: 0.8)
/// * `options` - Optional plot options
///
/// # Returns
/// A `Graphics` object containing the bar chart
///
/// # Examples
/// ```ignore
/// use rustmath_plot::plots::bar_chart;
///
/// // Simple bar chart
/// let data = vec![(0.0, 10.0), (1.0, 15.0), (2.0, 7.0), (3.0, 20.0)];
/// let g = bar_chart(data, None, None);
///
/// // Bar chart with custom bar width
/// let data = vec![(0.0, 5.0), (1.0, 8.0), (2.0, 3.0)];
/// let g = bar_chart(data, Some(0.6), None);
/// ```
pub fn bar_chart(
    data: Vec<(f64, f64)>,
    bar_width: Option<f64>,
    options: Option<PlotOptions>,
) -> Graphics {
    let width = bar_width.unwrap_or(0.8);
    let half_width = width / 2.0;

    let mut g = Graphics::new();

    for (x, height) in data {
        let rect_points = vec![
            Point2D::new(x - half_width, 0.0),
            Point2D::new(x + half_width, 0.0),
            Point2D::new(x + half_width, height),
            Point2D::new(x - half_width, height),
        ];

        let mut opts = options.clone().unwrap_or_default();
        opts.fill = true;

        g.add(polygon(rect_points, Some(opts)));
    }

    g
}

/// Create a horizontal bar chart
///
/// Similar to `bar_chart` but with horizontal bars.
///
/// # Arguments
/// * `data` - Vector of (label_position, value) pairs
/// * `bar_width` - Width of each bar (default: 0.8)
/// * `options` - Optional plot options
///
/// # Examples
/// ```ignore
/// use rustmath_plot::plots::bar_chart_horizontal;
///
/// let data = vec![(0.0, 10.0), (1.0, 15.0), (2.0, 7.0)];
/// let g = bar_chart_horizontal(data, None, None);
/// ```
pub fn bar_chart_horizontal(
    data: Vec<(f64, f64)>,
    bar_width: Option<f64>,
    options: Option<PlotOptions>,
) -> Graphics {
    let width = bar_width.unwrap_or(0.8);
    let half_width = width / 2.0;

    let mut g = Graphics::new();

    for (y, length) in data {
        let rect_points = vec![
            Point2D::new(0.0, y - half_width),
            Point2D::new(length, y - half_width),
            Point2D::new(length, y + half_width),
            Point2D::new(0.0, y + half_width),
        ];

        let mut opts = options.clone().unwrap_or_default();
        opts.fill = true;

        g.add(polygon(rect_points, Some(opts)));
    }

    g
}

/// Create a bar chart with per-bar colors
///
/// # Arguments
/// * `data` - Vector of (position, value, color) tuples
/// * `bar_width` - Width of each bar
///
/// # Examples
/// ```ignore
/// use rustmath_plot::plots::bar_chart_colored;
/// use rustmath_colors::Color;
///
/// let data = vec![
///     (0.0, 10.0, Color::rgb(1.0, 0.0, 0.0)), // Red
///     (1.0, 15.0, Color::rgb(0.0, 1.0, 0.0)), // Green
///     (2.0, 7.0, Color::rgb(0.0, 0.0, 1.0)),  // Blue
/// ];
/// let g = bar_chart_colored(data, None);
/// ```
pub fn bar_chart_colored(data: Vec<(f64, f64, Color)>, bar_width: Option<f64>) -> Graphics {
    let width = bar_width.unwrap_or(0.8);
    let half_width = width / 2.0;

    let mut g = Graphics::new();

    for (x, height, color) in data {
        let rect_points = vec![
            Point2D::new(x - half_width, 0.0),
            Point2D::new(x + half_width, 0.0),
            Point2D::new(x + half_width, height),
            Point2D::new(x - half_width, height),
        ];

        let mut opts = PlotOptions::default();
        opts.color = color;
        opts.fill = true;

        g.add(polygon(rect_points, Some(opts)));
    }

    g
}

/// Create a grouped bar chart
///
/// Visualizes multiple data series side by side.
///
/// # Arguments
/// * `groups` - Vector of group data, where each group is (position, values, colors)
/// * `bar_width` - Width of individual bars
/// * `group_spacing` - Spacing between groups (default: 1.0)
///
/// # Examples
/// ```ignore
/// use rustmath_plot::plots::bar_chart_grouped;
/// use rustmath_colors::Color;
///
/// let groups = vec![
///     (vec![10.0, 15.0], vec![Color::rgb(1.0, 0.0, 0.0), Color::rgb(0.0, 1.0, 0.0)]),
///     (vec![12.0, 18.0], vec![Color::rgb(1.0, 0.0, 0.0), Color::rgb(0.0, 1.0, 0.0)]),
/// ];
/// let g = bar_chart_grouped(groups, None, None);
/// ```
pub fn bar_chart_grouped(
    groups: Vec<(Vec<f64>, Vec<Color>)>,
    bar_width: Option<f64>,
    group_spacing: Option<f64>,
) -> Graphics {
    let width = bar_width.unwrap_or(0.2);
    let spacing = group_spacing.unwrap_or(1.0);
    let half_width = width / 2.0;

    let mut g = Graphics::new();

    for (group_idx, (values, colors)) in groups.iter().enumerate() {
        let group_center = group_idx as f64 * spacing;
        let num_bars = values.len();

        for (bar_idx, (&height, color)) in values.iter().zip(colors.iter()).enumerate() {
            // Calculate x position for this bar
            let offset = if num_bars == 1 {
                0.0
            } else {
                let total_width = (num_bars - 1) as f64 * width;
                bar_idx as f64 * width - total_width / 2.0
            };

            let x = group_center + offset;

            let rect_points = vec![
                Point2D::new(x - half_width, 0.0),
                Point2D::new(x + half_width, 0.0),
                Point2D::new(x + half_width, height),
                Point2D::new(x - half_width, height),
            ];

            let mut opts = PlotOptions::default();
            opts.color = color.clone();
            opts.fill = true;

            g.add(polygon(rect_points, Some(opts)));
        }
    }

    g
}

/// Create a stacked bar chart
///
/// Visualizes cumulative data by stacking bars on top of each other.
///
/// # Arguments
/// * `data` - Vector of (position, values, colors) where values are stacked
/// * `bar_width` - Width of bars
///
/// # Examples
/// ```ignore
/// use rustmath_plot::plots::bar_chart_stacked;
/// use rustmath_colors::Color;
///
/// let data = vec![
///     (0.0, vec![5.0, 3.0, 2.0], vec![
///         Color::rgb(1.0, 0.0, 0.0),
///         Color::rgb(0.0, 1.0, 0.0),
///         Color::rgb(0.0, 0.0, 1.0)
///     ]),
/// ];
/// let g = bar_chart_stacked(data, None);
/// ```
pub fn bar_chart_stacked(data: Vec<(f64, Vec<f64>, Vec<Color>)>, bar_width: Option<f64>) -> Graphics {
    let width = bar_width.unwrap_or(0.8);
    let half_width = width / 2.0;

    let mut g = Graphics::new();

    for (x, values, colors) in data {
        let mut y_bottom = 0.0;

        for (&height, color) in values.iter().zip(colors.iter()) {
            let y_top = y_bottom + height;

            let rect_points = vec![
                Point2D::new(x - half_width, y_bottom),
                Point2D::new(x + half_width, y_bottom),
                Point2D::new(x + half_width, y_top),
                Point2D::new(x - half_width, y_top),
            ];

            let mut opts = PlotOptions::default();
            opts.color = color.clone();
            opts.fill = true;

            g.add(polygon(rect_points, Some(opts)));

            y_bottom = y_top;
        }
    }

    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bar_chart() {
        let data = vec![(0.0, 10.0), (1.0, 15.0), (2.0, 7.0), (3.0, 20.0)];
        let g = bar_chart(data, None, None);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_bar_chart_custom_width() {
        let data = vec![(0.0, 5.0), (1.0, 8.0), (2.0, 3.0)];
        let g = bar_chart(data, Some(0.6), None);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_bar_chart_horizontal() {
        let data = vec![(0.0, 10.0), (1.0, 15.0), (2.0, 7.0)];
        let g = bar_chart_horizontal(data, None, None);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_bar_chart_colored() {
        let data = vec![
            (0.0, 10.0, Color::rgb(1.0, 0.0, 0.0)),
            (1.0, 15.0, Color::rgb(0.0, 1.0, 0.0)),
            (2.0, 7.0, Color::rgb(0.0, 0.0, 1.0)),
        ];
        let g = bar_chart_colored(data, None);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_bar_chart_grouped() {
        let groups = vec![
            (
                vec![10.0, 15.0],
                vec![Color::rgb(1.0, 0.0, 0.0), Color::rgb(0.0, 1.0, 0.0)],
            ),
            (
                vec![12.0, 18.0],
                vec![Color::rgb(1.0, 0.0, 0.0), Color::rgb(0.0, 1.0, 0.0)],
            ),
        ];
        let g = bar_chart_grouped(groups, None, None);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_bar_chart_stacked() {
        let data = vec![(
            0.0,
            vec![5.0, 3.0, 2.0],
            vec![
                Color::rgb(1.0, 0.0, 0.0),
                Color::rgb(0.0, 1.0, 0.0),
                Color::rgb(0.0, 0.0, 1.0),
            ],
        )];
        let g = bar_chart_stacked(data, None);
        assert!(!g.is_empty());
    }
}
