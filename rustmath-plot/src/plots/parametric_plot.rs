//! Parametric curve plotting
//!
//! Provides functions for plotting parametric curves in 2D.

use crate::primitives::line;
use crate::Graphics;
use rustmath_plot_core::{PlotOptions, Point2D, Result};

/// Plot a parametric curve
///
/// Creates a plot of a parametric curve defined by (x(t), y(t)) over a parameter range.
///
/// # Arguments
/// * `x_func` - Function for x coordinate as a function of parameter t
/// * `y_func` - Function for y coordinate as a function of parameter t
/// * `t_min` - Minimum parameter value
/// * `t_max` - Maximum parameter value
/// * `num_points` - Number of points to sample (default: 200)
/// * `options` - Optional plot options
///
/// # Returns
/// A `Graphics` object containing the parametric curve
///
/// # Examples
/// ```ignore
/// use rustmath_plot::plots::parametric_plot;
/// use std::f64::consts::PI;
///
/// // Plot a circle: x = cos(t), y = sin(t)
/// let g = parametric_plot(
///     |t| t.cos(),
///     |t| t.sin(),
///     0.0,
///     2.0 * PI,
///     None,
///     None
/// );
///
/// // Plot a spiral
/// let g = parametric_plot(
///     |t| t * t.cos(),
///     |t| t * t.sin(),
///     0.0,
///     4.0 * PI,
///     Some(500),
///     None
/// );
///
/// // Plot a Lissajous curve
/// let g = parametric_plot(
///     |t| (3.0 * t).sin(),
///     |t| (2.0 * t).sin(),
///     0.0,
///     2.0 * PI,
///     None,
///     None
/// );
/// ```
///
/// # Common Parametric Curves
/// - Circle: x = r*cos(t), y = r*sin(t)
/// - Ellipse: x = a*cos(t), y = b*sin(t)
/// - Spiral: x = t*cos(t), y = t*sin(t)
/// - Lissajous: x = sin(a*t), y = sin(b*t)
/// - Cycloid: x = r*(t - sin(t)), y = r*(1 - cos(t))
pub fn parametric_plot<F, G>(
    x_func: F,
    y_func: G,
    t_min: f64,
    t_max: f64,
    num_points: Option<usize>,
    options: Option<PlotOptions>,
) -> Graphics
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
{
    let n = num_points.unwrap_or(200);
    let dt = (t_max - t_min) / (n - 1) as f64;

    // Sample the parametric curve
    let points: Vec<Point2D> = (0..n)
        .map(|i| {
            let t = t_min + i as f64 * dt;
            Point2D::new(x_func(t), y_func(t))
        })
        .collect();

    // Create graphics object and add line
    let mut g = Graphics::new();
    g.add(line(points, options));

    g
}

/// Plot a parametric curve with adaptive sampling
///
/// Similar to `parametric_plot()`, but uses adaptive sampling to add more points
/// where the curve has higher curvature.
///
/// # Arguments
/// * `x_func` - Function for x coordinate
/// * `y_func` - Function for y coordinate
/// * `t_min` - Minimum parameter value
/// * `t_max` - Maximum parameter value
/// * `tolerance` - Tolerance for curvature detection (smaller = more detail)
/// * `options` - Optional plot options
///
/// # Returns
/// A `Graphics` object containing the parametric curve
pub fn parametric_plot_adaptive<F, G>(
    x_func: F,
    y_func: G,
    t_min: f64,
    t_max: f64,
    tolerance: f64,
    options: Option<PlotOptions>,
) -> Graphics
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
{
    // Start with initial sampling
    let initial_samples = 50;
    let mut points = Vec::new();

    // Helper function to check if three points are approximately collinear
    let is_linear = |p1: &Point2D, p2: &Point2D, p3: &Point2D| -> bool {
        let dx = p3.x - p1.x;
        let dy = p3.y - p1.y;
        let len = (dx * dx + dy * dy).sqrt();

        if len == 0.0 {
            return true;
        }

        let dist = ((p2.x - p1.x) * dy - (p2.y - p1.y) * dx).abs() / len;
        dist < tolerance
    };

    // Recursive subdivision function
    fn subdivide<F, G, IsLinear>(
        x_func: &F,
        y_func: &G,
        is_linear: &IsLinear,
        t1: f64,
        t2: f64,
        depth: usize,
        max_depth: usize,
    ) -> Vec<Point2D>
    where
        F: Fn(f64) -> f64,
        G: Fn(f64) -> f64,
        IsLinear: Fn(&Point2D, &Point2D, &Point2D) -> bool,
    {
        let p1 = Point2D::new(x_func(t1), y_func(t1));
        let p3 = Point2D::new(x_func(t2), y_func(t2));

        if depth >= max_depth {
            return vec![p1, p3];
        }

        let t_mid = (t1 + t2) / 2.0;
        let p2 = Point2D::new(x_func(t_mid), y_func(t_mid));

        if is_linear(&p1, &p2, &p3) {
            vec![p1, p3]
        } else {
            let mut left = subdivide(x_func, y_func, is_linear, t1, t_mid, depth + 1, max_depth);
            let right = subdivide(x_func, y_func, is_linear, t_mid, t2, depth + 1, max_depth);

            left.pop(); // Remove duplicate midpoint
            left.extend(right);
            left
        }
    }

    // Perform adaptive sampling
    let dt = (t_max - t_min) / (initial_samples - 1) as f64;
    for i in 0..initial_samples - 1 {
        let t1 = t_min + i as f64 * dt;
        let t2 = t_min + (i + 1) as f64 * dt;

        let mut segment = subdivide(&x_func, &y_func, &is_linear, t1, t2, 0, 10);
        if i > 0 {
            segment.remove(0); // Remove duplicate point
        }
        points.extend(segment);
    }

    // Create graphics object and add line
    let mut g = Graphics::new();
    g.add(line(points, options));

    g
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_colors::Color;
    use std::f64::consts::PI;

    #[test]
    fn test_parametric_plot_circle() {
        let g = parametric_plot(|t| t.cos(), |t| t.sin(), 0.0, 2.0 * PI, None, None);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_parametric_plot_ellipse() {
        // Ellipse with semi-axes a=2, b=1
        let g = parametric_plot(|t| 2.0 * t.cos(), |t| t.sin(), 0.0, 2.0 * PI, None, None);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_parametric_plot_spiral() {
        let g = parametric_plot(
            |t| t * t.cos(),
            |t| t * t.sin(),
            0.0,
            4.0 * PI,
            Some(500),
            None,
        );
        assert!(!g.is_empty());
    }

    #[test]
    fn test_parametric_plot_lissajous() {
        // Lissajous curve with a=3, b=2
        let g = parametric_plot(
            |t| (3.0 * t).sin(),
            |t| (2.0 * t).sin(),
            0.0,
            2.0 * PI,
            None,
            None,
        );
        assert!(!g.is_empty());
    }

    #[test]
    fn test_parametric_plot_with_options() {
        let mut opts = PlotOptions::default();
        opts.color = Color::rgb(1.0, 0.0, 0.0);
        opts.thickness = 2.0;

        let g = parametric_plot(|t| t.cos(), |t| t.sin(), 0.0, 2.0 * PI, None, Some(opts));
        assert!(!g.is_empty());
    }

    #[test]
    fn test_parametric_plot_adaptive() {
        let g = parametric_plot_adaptive(|t| t.cos(), |t| t.sin(), 0.0, 2.0 * PI, 0.01, None);
        assert!(!g.is_empty());
    }
}
