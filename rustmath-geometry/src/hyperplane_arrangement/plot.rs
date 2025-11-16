//! Plotting support for hyperplane arrangements
//!
//! This module provides data structures and functions to generate plot data
//! for hyperplane arrangements. Since Rust doesn't have built-in plotting
//! capabilities like SageMath, this module focuses on preparing the data
//! that can be consumed by external plotting libraries (e.g., plotters, plotly).
//!
//! # Design Philosophy
//!
//! Rather than directly rendering graphics, this module:
//! 1. Extracts geometric data from hyperplane arrangements
//! 2. Computes intersection points, line segments, and plane regions
//! 3. Generates color schemes and labels
//! 4. Returns structured data suitable for visualization
//!
//! External plotting libraries can then consume this data to create actual visualizations.

use crate::point::Point2D;
use rustmath_rationals::Rational;
use std::collections::HashMap;

/// RGB color representation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Color {
    /// Create a new color from RGB components (0-255)
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Color { r, g, b }
    }

    /// Create a color from HSV (Hue: 0-360, Saturation: 0-1, Value: 0-1)
    /// This is useful for generating color palettes
    pub fn from_hsv(h: f64, s: f64, v: f64) -> Self {
        let c = v * s;
        let h_prime = h / 60.0;
        let x = c * (1.0 - ((h_prime % 2.0) - 1.0).abs());
        let m = v - c;

        let (r, g, b) = if h_prime < 1.0 {
            (c, x, 0.0)
        } else if h_prime < 2.0 {
            (x, c, 0.0)
        } else if h_prime < 3.0 {
            (0.0, c, x)
        } else if h_prime < 4.0 {
            (0.0, x, c)
        } else if h_prime < 5.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };

        Color {
            r: ((r + m) * 255.0) as u8,
            g: ((g + m) * 255.0) as u8,
            b: ((b + m) * 255.0) as u8,
        }
    }

    /// Convert to hex string (e.g., "#FF0000" for red)
    pub fn to_hex(&self) -> String {
        format!("#{:02X}{:02X}{:02X}", self.r, self.g, self.b)
    }
}

/// Plot data for a single hyperplane
#[derive(Debug, Clone)]
pub struct HyperplanePlotData {
    /// Label for this hyperplane
    pub label: String,

    /// Color for this hyperplane
    pub color: Color,

    /// Opacity (0.0 = fully transparent, 1.0 = fully opaque)
    pub opacity: f64,

    /// For 1D (points): the point coordinates
    pub point: Option<Vec<f64>>,

    /// For 2D (lines): parametric line data (start_point, direction, parameter_range)
    pub line_data: Option<(Vec<f64>, Vec<f64>, (f64, f64))>,

    /// For 3D (planes): normal vector and offset
    pub plane_data: Option<(Vec<f64>, f64)>,

    /// Label position offset
    pub label_offset: (f64, f64),

    /// Label font size
    pub label_fontsize: f64,

    /// Label color
    pub label_color: Color,
}

impl HyperplanePlotData {
    /// Create default plot data for a hyperplane
    pub fn new(label: String, color: Color) -> Self {
        HyperplanePlotData {
            label,
            color,
            opacity: 0.7,
            point: None,
            line_data: None,
            plane_data: None,
            label_offset: (0.0, 0.0),
            label_fontsize: 12.0,
            label_color: Color::new(0, 0, 0), // Black
        }
    }
}

/// Complete plot data for a hyperplane arrangement
#[derive(Debug, Clone)]
pub struct PlotData {
    /// Dimension of the ambient space (1, 2, or 3)
    pub dimension: usize,

    /// Plot data for each hyperplane
    pub hyperplanes: Vec<HyperplanePlotData>,

    /// Optional viewing ranges (min, max) for each axis
    pub ranges: Option<Vec<(f64, f64)>>,

    /// Whether to show a legend
    pub show_legend: bool,

    /// Legend entries (for 3D plots)
    pub legend_entries: Vec<(String, Color)>,
}

impl PlotData {
    /// Create new plot data for an arrangement
    pub fn new(dimension: usize) -> Self {
        PlotData {
            dimension,
            hyperplanes: Vec::new(),
            ranges: None,
            show_legend: false,
            legend_entries: Vec::new(),
        }
    }

    /// Add a hyperplane to the plot data
    pub fn add_hyperplane(&mut self, data: HyperplanePlotData) {
        self.hyperplanes.push(data);
    }

    /// Set viewing ranges
    pub fn set_ranges(&mut self, ranges: Vec<(f64, f64)>) {
        self.ranges = Some(ranges);
    }

    /// Enable legend
    pub fn enable_legend(&mut self) {
        self.show_legend = true;
    }
}

/// Generate a color palette using HSV color space
///
/// This generates evenly spaced colors around the color wheel,
/// which is useful for distinguishing multiple hyperplanes.
///
/// # Arguments
///
/// * `n` - Number of colors to generate
/// * `saturation` - Saturation value (0.0 to 1.0)
/// * `value` - Value/brightness (0.0 to 1.0)
///
/// # Returns
///
/// A vector of `n` colors
///
/// # Examples
///
/// ```
/// use rustmath_geometry::hyperplane_arrangement::plot::generate_color_palette;
///
/// let colors = generate_color_palette(5, 0.8, 0.9);
/// assert_eq!(colors.len(), 5);
/// ```
pub fn generate_color_palette(n: usize, saturation: f64, value: f64) -> Vec<Color> {
    if n == 0 {
        return Vec::new();
    }

    let mut colors = Vec::new();
    for i in 0..n {
        let hue = (i as f64 * 360.0) / (n as f64);
        colors.push(Color::from_hsv(hue, saturation, value));
    }
    colors
}

/// Generate plot data for a single hyperplane
///
/// This function extracts the geometric information from a hyperplane
/// and prepares it for plotting.
///
/// # Arguments
///
/// * `normal` - Normal vector of the hyperplane (as coefficients)
/// * `offset` - Constant term in the hyperplane equation
/// * `dimension` - Ambient space dimension
/// * `label` - Label for the hyperplane
/// * `color` - Color for the hyperplane
/// * `ranges` - Optional viewing ranges for each axis
///
/// # Returns
///
/// Plot data for the hyperplane
///
/// # Mathematical Background
///
/// A hyperplane in n-dimensional space is defined by the equation:
/// a₁x₁ + a₂x₂ + ... + aₙxₙ + c = 0
///
/// where (a₁, a₂, ..., aₙ) is the normal vector and c is the offset.
///
/// # Examples
///
/// ```
/// use rustmath_geometry::hyperplane_arrangement::plot::{plot_hyperplane_data, Color};
///
/// // Plot a line in 2D: x + y = 1
/// let normal = vec![1.0, 1.0];
/// let offset = -1.0;
/// let ranges = vec![(-2.0, 2.0), (-2.0, 2.0)];
///
/// let data = plot_hyperplane_data(
///     &normal,
///     offset,
///     2,
///     "x + y = 1".to_string(),
///     Color::new(255, 0, 0),
///     Some(ranges)
/// );
/// ```
pub fn plot_hyperplane_data(
    normal: &[f64],
    offset: f64,
    dimension: usize,
    label: String,
    color: Color,
    ranges: Option<Vec<(f64, f64)>>,
) -> HyperplanePlotData {
    let mut plot_data = HyperplanePlotData::new(label, color);

    match dimension {
        1 => {
            // 0-dimensional hyperplane in 1D space = a point
            // Equation: a*x + c = 0, so x = -c/a
            if normal.len() >= 1 && normal[0].abs() > 1e-10 {
                let x = -offset / normal[0];
                plot_data.point = Some(vec![x]);
            }
        }
        2 => {
            // 1-dimensional hyperplane in 2D space = a line
            // Equation: a*x + b*y + c = 0
            if normal.len() >= 2 {
                let a = normal[0];
                let b = normal[1];

                // Compute parametric form
                // Direction vector: perpendicular to normal
                let direction = vec![-b, a];

                // Find a point on the line
                let point = if a.abs() > 1e-10 {
                    vec![-offset / a, 0.0]
                } else if b.abs() > 1e-10 {
                    vec![0.0, -offset / b]
                } else {
                    vec![0.0, 0.0]
                };

                // Determine parameter range from viewing bounds
                let param_range = if let Some(ref r) = ranges {
                    let span = ((r[0].1 - r[0].0).powi(2) + (r[1].1 - r[1].0).powi(2)).sqrt();
                    (-span, span)
                } else {
                    (-10.0, 10.0)
                };

                plot_data.line_data = Some((point, direction, param_range));
            }
        }
        3 => {
            // 2-dimensional hyperplane in 3D space = a plane
            // Equation: a*x + b*y + c*z + d = 0
            if normal.len() >= 3 {
                plot_data.plane_data = Some((normal.to_vec(), offset));
            }
        }
        _ => {
            // Higher dimensions not supported for plotting
        }
    }

    plot_data
}

/// Generate plot data for a complete hyperplane arrangement
///
/// # Arguments
///
/// * `normals` - Normal vectors for each hyperplane
/// * `offsets` - Constant terms for each hyperplane
/// * `dimension` - Ambient space dimension
/// * `options` - Optional plotting options (colors, labels, ranges)
///
/// # Returns
///
/// Complete plot data for the arrangement
///
/// # Examples
///
/// ```
/// use rustmath_geometry::hyperplane_arrangement::plot::{plot_arrangement_data, PlotOptions};
///
/// // Plot a simple arrangement of two lines in 2D
/// let normals = vec![
///     vec![1.0, 0.0],  // x = 0
///     vec![0.0, 1.0],  // y = 0
/// ];
/// let offsets = vec![0.0, 0.0];
///
/// let options = PlotOptions::default();
/// let data = plot_arrangement_data(&normals, &offsets, 2, options);
///
/// assert_eq!(data.hyperplanes.len(), 2);
/// assert_eq!(data.dimension, 2);
/// ```
pub fn plot_arrangement_data(
    normals: &[Vec<f64>],
    offsets: &[f64],
    dimension: usize,
    options: PlotOptions,
) -> PlotData {
    let n = normals.len();
    let mut plot_data = PlotData::new(dimension);

    // Generate colors if not provided
    let colors = if let Some(c) = options.colors {
        c
    } else {
        generate_color_palette(n, 0.8, 0.9)
    };

    // Generate labels if not provided
    let labels = if let Some(l) = options.labels {
        l
    } else {
        (0..n).map(|i| format!("H{}", i)).collect()
    };

    // Generate plot data for each hyperplane
    for i in 0..n {
        let color = colors.get(i).copied().unwrap_or(Color::new(100, 100, 100));
        let label = labels.get(i).cloned().unwrap_or_else(|| format!("H{}", i));

        let hp_data = plot_hyperplane_data(
            &normals[i],
            offsets[i],
            dimension,
            label.clone(),
            color,
            options.ranges.clone(),
        );

        plot_data.add_hyperplane(hp_data);

        // Add to legend if enabled
        if options.show_legend {
            plot_data.legend_entries.push((label, color));
        }
    }

    if let Some(r) = options.ranges {
        plot_data.set_ranges(r);
    }

    if options.show_legend {
        plot_data.enable_legend();
    }

    plot_data
}

/// Plotting options for hyperplane arrangements
#[derive(Debug, Clone, Default)]
pub struct PlotOptions {
    /// Colors for each hyperplane
    pub colors: Option<Vec<Color>>,

    /// Labels for each hyperplane
    pub labels: Option<Vec<String>>,

    /// Label colors
    pub label_colors: Option<Vec<Color>>,

    /// Label font sizes
    pub label_fontsize: Option<f64>,

    /// Label position offsets
    pub label_offsets: Option<Vec<(f64, f64)>>,

    /// Viewing ranges for each axis
    pub ranges: Option<Vec<(f64, f64)>>,

    /// Opacities for each hyperplane
    pub opacities: Option<Vec<f64>>,

    /// Whether to show a legend
    pub show_legend: bool,

    /// Legend format ("short" or "long")
    pub legend_format: String,
}

impl PlotOptions {
    /// Create default plotting options
    pub fn new() -> Self {
        PlotOptions {
            colors: None,
            labels: None,
            label_colors: None,
            label_fontsize: Some(12.0),
            label_offsets: None,
            ranges: None,
            opacities: None,
            show_legend: false,
            legend_format: "short".to_string(),
        }
    }
}

/// Generate 3D legend data
///
/// Creates legend entries for a 3D hyperplane arrangement.
///
/// # Arguments
///
/// * `labels` - Labels for each hyperplane
/// * `colors` - Colors for each hyperplane
/// * `format` - "short" for abbreviated labels, "long" for full labels
///
/// # Returns
///
/// Vector of legend entries (label, color)
///
/// # Examples
///
/// ```
/// use rustmath_geometry::hyperplane_arrangement::plot::{legend_3d, Color};
///
/// let labels = vec!["Plane 1".to_string(), "Plane 2".to_string()];
/// let colors = vec![Color::new(255, 0, 0), Color::new(0, 255, 0)];
///
/// let legend = legend_3d(&labels, &colors, "short");
/// assert_eq!(legend.len(), 2);
/// ```
pub fn legend_3d(
    labels: &[String],
    colors: &[Color],
    format: &str,
) -> Vec<(String, Color)> {
    let mut entries = Vec::new();

    for (i, (label, color)) in labels.iter().zip(colors.iter()).enumerate() {
        let entry_label = if format == "short" {
            format!("H{}", i)
        } else {
            label.clone()
        };

        entries.push((entry_label, *color));
    }

    entries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_from_hsv() {
        // Red (H=0)
        let red = Color::from_hsv(0.0, 1.0, 1.0);
        assert_eq!(red.r, 255);
        assert_eq!(red.g, 0);
        assert_eq!(red.b, 0);

        // Green (H=120)
        let green = Color::from_hsv(120.0, 1.0, 1.0);
        assert_eq!(green.r, 0);
        assert_eq!(green.g, 255);
        assert_eq!(green.b, 0);

        // Blue (H=240)
        let blue = Color::from_hsv(240.0, 1.0, 1.0);
        assert_eq!(blue.r, 0);
        assert_eq!(blue.g, 0);
        assert_eq!(blue.b, 255);
    }

    #[test]
    fn test_color_to_hex() {
        let red = Color::new(255, 0, 0);
        assert_eq!(red.to_hex(), "#FF0000");

        let green = Color::new(0, 255, 0);
        assert_eq!(green.to_hex(), "#00FF00");

        let blue = Color::new(0, 0, 255);
        assert_eq!(blue.to_hex(), "#0000FF");
    }

    #[test]
    fn test_generate_color_palette() {
        let colors = generate_color_palette(3, 0.8, 0.9);
        assert_eq!(colors.len(), 3);

        // Colors should be different
        assert_ne!(colors[0], colors[1]);
        assert_ne!(colors[1], colors[2]);
        assert_ne!(colors[0], colors[2]);
    }

    #[test]
    fn test_plot_hyperplane_data_1d() {
        // Point at x = 2
        let normal = vec![1.0];
        let offset = -2.0;

        let data = plot_hyperplane_data(
            &normal,
            offset,
            1,
            "x = 2".to_string(),
            Color::new(255, 0, 0),
            None,
        );

        assert!(data.point.is_some());
        let point = data.point.unwrap();
        assert_eq!(point.len(), 1);
        assert!((point[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_plot_hyperplane_data_2d() {
        // Line: x + y = 0
        let normal = vec![1.0, 1.0];
        let offset = 0.0;

        let data = plot_hyperplane_data(
            &normal,
            offset,
            2,
            "x + y = 0".to_string(),
            Color::new(255, 0, 0),
            None,
        );

        assert!(data.line_data.is_some());
    }

    #[test]
    fn test_plot_hyperplane_data_3d() {
        // Plane: x + y + z = 1
        let normal = vec![1.0, 1.0, 1.0];
        let offset = -1.0;

        let data = plot_hyperplane_data(
            &normal,
            offset,
            3,
            "x + y + z = 1".to_string(),
            Color::new(255, 0, 0),
            None,
        );

        assert!(data.plane_data.is_some());
    }

    #[test]
    fn test_plot_arrangement_data() {
        let normals = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let offsets = vec![0.0, 0.0];

        let options = PlotOptions::default();
        let data = plot_arrangement_data(&normals, &offsets, 2, options);

        assert_eq!(data.dimension, 2);
        assert_eq!(data.hyperplanes.len(), 2);
    }

    #[test]
    fn test_legend_3d() {
        let labels = vec!["Plane 1".to_string(), "Plane 2".to_string()];
        let colors = vec![Color::new(255, 0, 0), Color::new(0, 255, 0)];

        let legend = legend_3d(&labels, &colors, "short");
        assert_eq!(legend.len(), 2);
        assert_eq!(legend[0].0, "H0");
        assert_eq!(legend[1].0, "H1");

        let legend_long = legend_3d(&labels, &colors, "long");
        assert_eq!(legend_long[0].0, "Plane 1");
        assert_eq!(legend_long[1].0, "Plane 2");
    }

    #[test]
    fn test_plot_options_default() {
        let options = PlotOptions::default();
        assert!(options.colors.is_none());
        assert!(options.labels.is_none());
        assert!(!options.show_legend);
    }

    #[test]
    fn test_plot_data_new() {
        let mut plot_data = PlotData::new(3);
        assert_eq!(plot_data.dimension, 3);
        assert_eq!(plot_data.hyperplanes.len(), 0);

        let hp_data = HyperplanePlotData::new("Test".to_string(), Color::new(255, 0, 0));
        plot_data.add_hyperplane(hp_data);
        assert_eq!(plot_data.hyperplanes.len(), 1);
    }
}
