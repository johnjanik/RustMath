//! Basic function plotting
//!
//! Provides the fundamental `plot()` function for visualizing mathematical functions.

use crate::backend::AdaptiveSampler;
use crate::primitives::line;
use crate::Graphics;
use rustmath_plot_core::PlotOptions;

/// Plot a function over a given range
///
/// This is the fundamental plotting function in RustMath, similar to SageMath's `plot()`.
/// It uses adaptive sampling to intelligently sample the function at varying levels of detail.
///
/// # Arguments
/// * `f` - The function to plot (must implement `Fn(f64) -> f64`)
/// * `x_min` - Minimum x value
/// * `x_max` - Maximum x value
/// * `options` - Optional plot options (color, thickness, etc.)
///
/// # Returns
/// A `Graphics` object containing the plotted function
///
/// # Examples
/// ```ignore
/// use rustmath_plot::plots::plot;
///
/// // Plot sin(x) from 0 to 2π
/// let g = plot(|x| x.sin(), 0.0, 2.0 * std::f64::consts::PI, None);
///
/// // Plot with custom options
/// use rustmath_plot::PlotOptions;
/// use rustmath_colors::Color;
///
/// let mut opts = PlotOptions::default();
/// opts.color = Color::rgb(1.0, 0.0, 0.0); // Red
/// opts.thickness = 2.0;
///
/// let g = plot(|x| x * x, -5.0, 5.0, Some(opts));
/// ```
///
/// # Implementation Details
/// - Uses adaptive sampling with a tolerance of 0.01
/// - Maximum recursion depth of 10
/// - Initial uniform sampling of 100 points
/// - Automatically handles discontinuities and sharp features
pub fn plot<F>(f: F, x_min: f64, x_max: f64, options: Option<PlotOptions>) -> Graphics
where
    F: Fn(f64) -> f64,
{
    // Create adaptive sampler with reasonable defaults
    let sampler = AdaptiveSampler::new(10, 0.01);

    // Sample the function
    let points = sampler.sample_range(&f, x_min, x_max, 100);

    // Create a Graphics object and add the line primitive
    let mut g = Graphics::new();
    g.add(line(points, options));

    g
}

/// Plot multiple functions on the same axes
///
/// This is a convenience function for plotting multiple functions together.
///
/// # Arguments
/// * `functions` - A vector of (function, options) pairs
/// * `x_min` - Minimum x value
/// * `x_max` - Maximum x value
///
/// # Returns
/// A `Graphics` object containing all plotted functions
///
/// # Examples
/// ```ignore
/// use rustmath_plot::plots::plot_multiple;
/// use rustmath_plot::PlotOptions;
/// use rustmath_colors::Color;
///
/// let mut opts1 = PlotOptions::default();
/// opts1.color = Color::rgb(1.0, 0.0, 0.0); // Red
///
/// let mut opts2 = PlotOptions::default();
/// opts2.color = Color::rgb(0.0, 0.0, 1.0); // Blue
///
/// let functions = vec![
///     (|x: f64| x.sin() as Box<dyn Fn(f64) -> f64>, Some(opts1)),
///     (|x: f64| x.cos() as Box<dyn Fn(f64) -> f64>, Some(opts2)),
/// ];
///
/// let g = plot_multiple(functions, 0.0, 2.0 * std::f64::consts::PI);
/// ```
pub fn plot_multiple<F>(
    functions: Vec<(F, Option<PlotOptions>)>,
    x_min: f64,
    x_max: f64,
) -> Graphics
where
    F: Fn(f64) -> f64,
{
    let mut g = Graphics::new();

    for (f, opts) in functions {
        let sampler = AdaptiveSampler::new(10, 0.01);
        let points = sampler.sample_range(&f, x_min, x_max, 100);
        g.add(line(points, opts));
    }

    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plot_linear() {
        let g = plot(|x| 2.0 * x + 1.0, 0.0, 10.0, None);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_plot_quadratic() {
        let g = plot(|x| x * x, -5.0, 5.0, None);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_plot_sine() {
        let g = plot(|x| x.sin(), 0.0, 2.0 * std::f64::consts::PI, None);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_plot_with_options() {
        let mut opts = PlotOptions::default();
        opts.color = Color::rgb(1.0, 0.0, 0.0);
        opts.thickness = 2.0;

        let g = plot(|x| x.exp(), 0.0, 2.0, Some(opts));
        assert!(!g.is_empty());
    }

    #[test]
    fn test_plot_discontinuous() {
        // Test with a function that has a discontinuity
        let g = plot(|x| if x < 0.0 { -1.0 } else { 1.0 }, -5.0, 5.0, None);
        assert!(!g.is_empty());
    }
}
