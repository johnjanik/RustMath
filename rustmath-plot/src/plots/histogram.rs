//! Histogram plotting - visualizing data distributions
//!
//! Provides histogram functionality for visualizing frequency distributions.

use crate::primitives::polygon;
use crate::Graphics;
use rustmath_plot_core::{PlotOptions, Point2D, Result};

/// Create a histogram from data
///
/// Visualizes the distribution of data values by dividing the range into bins
/// and counting the frequency in each bin.
///
/// # Arguments
/// * `data` - Vector of data values
/// * `bins` - Number of bins (if None, uses Sturges' formula)
/// * `range` - Optional (min, max) range (if None, uses data range)
/// * `normalized` - If true, normalize to probability density (default: false)
/// * `options` - Optional plot options
///
/// # Returns
/// A `Graphics` object containing the histogram
///
/// # Examples
/// ```ignore
/// use rustmath_plot::plots::histogram;
///
/// // Basic histogram with auto bins
/// let data = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0];
/// let g = histogram(data, None, None, None, None);
///
/// // Histogram with 20 bins
/// let data: Vec<f64> = (0..1000).map(|_| rand::random()).collect();
/// let g = histogram(data, Some(20), None, None, None);
///
/// // Normalized histogram (probability density)
/// let data = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0];
/// let g = histogram(data, Some(10), None, Some(true), None);
/// ```
pub fn histogram(
    data: Vec<f64>,
    bins: Option<usize>,
    range: Option<(f64, f64)>,
    normalized: Option<bool>,
    options: Option<PlotOptions>,
) -> Graphics {
    if data.is_empty() {
        return Graphics::new();
    }

    // Determine data range
    let (min_val, max_val) = if let Some((min, max)) = range {
        (min, max)
    } else {
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (min, max)
    };

    // Determine number of bins using Sturges' formula if not provided
    let num_bins = bins.unwrap_or_else(|| {
        let n = data.len() as f64;
        (n.log2().ceil() + 1.0) as usize
    });

    // Create bins
    let bin_width = (max_val - min_val) / num_bins as f64;
    let mut counts = vec![0; num_bins];

    // Count data in bins
    for &value in &data {
        if value >= min_val && value <= max_val {
            let bin_idx = ((value - min_val) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(num_bins - 1); // Handle edge case for max_val
            counts[bin_idx] += 1;
        }
    }

    // Normalize if requested
    let normalize = normalized.unwrap_or(false);
    let heights: Vec<f64> = if normalize {
        let total: f64 = counts.iter().sum::<usize>() as f64;
        counts
            .iter()
            .map(|&count| count as f64 / (total * bin_width))
            .collect()
    } else {
        counts.iter().map(|&count| count as f64).collect()
    };

    // Create histogram bars
    let mut g = Graphics::new();

    for (i, &height) in heights.iter().enumerate() {
        if height > 0.0 {
            let x_left = min_val + i as f64 * bin_width;
            let x_right = x_left + bin_width;

            let rect_points = vec![
                Point2D::new(x_left, 0.0),
                Point2D::new(x_right, 0.0),
                Point2D::new(x_right, height),
                Point2D::new(x_left, height),
            ];

            let mut opts = options.clone().unwrap_or_default();
            opts.fill = true;

            g.add(polygon(rect_points, Some(opts)));
        }
    }

    g
}

/// Create a histogram with custom bin edges
///
/// # Arguments
/// * `data` - Vector of data values
/// * `bin_edges` - Vector of bin edge values (length n+1 for n bins)
/// * `normalized` - If true, normalize to probability density
/// * `options` - Optional plot options
///
/// # Examples
/// ```ignore
/// use rustmath_plot::plots::histogram_custom_bins;
///
/// let data = vec![1.0, 2.5, 3.0, 4.5, 6.0, 7.5];
/// let bin_edges = vec![0.0, 2.0, 5.0, 10.0]; // 3 bins with different widths
/// let g = histogram_custom_bins(data, bin_edges, None, None);
/// ```
pub fn histogram_custom_bins(
    data: Vec<f64>,
    bin_edges: Vec<f64>,
    normalized: Option<bool>,
    options: Option<PlotOptions>,
) -> Graphics {
    if data.is_empty() || bin_edges.len() < 2 {
        return Graphics::new();
    }

    let num_bins = bin_edges.len() - 1;
    let mut counts = vec![0; num_bins];

    // Count data in bins
    for &value in &data {
        for i in 0..num_bins {
            if value >= bin_edges[i] && value < bin_edges[i + 1] {
                counts[i] += 1;
                break;
            } else if i == num_bins - 1 && value == bin_edges[i + 1] {
                // Include right edge in last bin
                counts[i] += 1;
                break;
            }
        }
    }

    // Normalize if requested
    let normalize = normalized.unwrap_or(false);
    let heights: Vec<f64> = if normalize {
        let total: f64 = counts.iter().sum::<usize>() as f64;
        counts
            .iter()
            .enumerate()
            .map(|(i, &count)| {
                let bin_width = bin_edges[i + 1] - bin_edges[i];
                count as f64 / (total * bin_width)
            })
            .collect()
    } else {
        counts.iter().map(|&count| count as f64).collect()
    };

    // Create histogram bars
    let mut g = Graphics::new();

    for (i, &height) in heights.iter().enumerate() {
        if height > 0.0 {
            let x_left = bin_edges[i];
            let x_right = bin_edges[i + 1];

            let rect_points = vec![
                Point2D::new(x_left, 0.0),
                Point2D::new(x_right, 0.0),
                Point2D::new(x_right, height),
                Point2D::new(x_left, height),
            ];

            let mut opts = options.clone().unwrap_or_default();
            opts.fill = true;

            g.add(polygon(rect_points, Some(opts)));
        }
    }

    g
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_colors::Color;

    #[test]
    fn test_histogram_basic() {
        let data = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0];
        let g = histogram(data, Some(5), None, None, None);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_histogram_auto_bins() {
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let g = histogram(data, None, None, None, None);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_histogram_normalized() {
        let data = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0];
        let g = histogram(data, Some(4), None, Some(true), None);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_histogram_with_range() {
        let data = vec![0.0, 1.0, 2.0, 5.0, 10.0, 15.0];
        let g = histogram(data, Some(3), Some((0.0, 10.0)), None, None);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_histogram_empty() {
        let data: Vec<f64> = vec![];
        let g = histogram(data, Some(5), None, None, None);
        assert!(g.is_empty());
    }

    #[test]
    fn test_histogram_with_options() {
        let mut opts = PlotOptions::default();
        opts.color = Color::rgb(0.0, 0.5, 1.0);

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let g = histogram(data, Some(5), None, None, Some(opts));
        assert!(!g.is_empty());
    }

    #[test]
    fn test_histogram_custom_bins() {
        let data = vec![1.0, 2.5, 3.0, 4.5, 6.0, 7.5];
        let bin_edges = vec![0.0, 2.0, 5.0, 10.0];
        let g = histogram_custom_bins(data, bin_edges, None, None);
        assert!(!g.is_empty());
    }
}
