//! Integration tests for 2D plotting functionality
//!
//! This test suite demonstrates all the features from tracker 11:
//! - Line plots
//! - Scatter plots
//! - Bar charts
//! - Histograms
//! - Contour plots
//! - Density plots
//! - Parametric curves
//! - Implicit plots
//! - SVG/PNG export

use rustmath_colors::Color;
use rustmath_plot::plots::*;
use rustmath_plot::{Graphics, PlotOptions, RenderFormat};
use std::f64::consts::PI;

#[test]
fn test_line_plot_creation() {
    // Create a line plot of sin(x)
    let g = plot(|x| x.sin(), 0.0, 2.0 * PI, None);
    assert!(!g.is_empty());
}

#[test]
fn test_line_plot_with_options() {
    let mut opts = PlotOptions::default();
    opts.color = Color::rgb(1.0, 0.0, 0.0);
    opts.thickness = 2.0;

    let g = plot(|x| x * x, -5.0, 5.0, Some(opts));
    assert!(!g.is_empty());
}

#[test]
fn test_scatter_plot_creation() {
    let data = vec![(0.0, 1.2), (1.0, 2.3), (2.0, 1.8), (3.0, 3.1)];
    let g = scatter_plot(data, None, None, None);
    assert!(!g.is_empty());
}

#[test]
fn test_bar_chart_creation() {
    let data = vec![(0.0, 10.0), (1.0, 15.0), (2.0, 7.0), (3.0, 20.0)];
    let g = bar_chart(data, None, None);
    assert!(!g.is_empty());
}

#[test]
fn test_histogram_creation() {
    let data = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0];
    let g = histogram(data, None, None, None, None);
    assert!(!g.is_empty());
}

#[test]
fn test_contour_plot_creation() {
    // Plot contours of f(x,y) = x^2 + y^2
    let g = contour_plot(
        |x, y| x * x + y * y,
        (-2.0, 2.0),
        (-2.0, 2.0),
        Some(vec![0.5, 1.0, 1.5, 2.0]),
        Some(30),
        None,
    );
    assert!(!g.is_empty());
}

#[test]
fn test_density_plot_creation() {
    // Plot density of f(x,y) = sin(x) * cos(y)
    let g = density_plot(
        |x, y| (x.sin() * y.cos()),
        (-PI, PI),
        (-PI, PI),
        Some(30),
        None,
        None,
    );
    assert!(!g.is_empty());
}

#[test]
fn test_parametric_plot_creation() {
    // Plot a circle: x = cos(t), y = sin(t)
    let g = parametric_plot(|t| t.cos(), |t| t.sin(), 0.0, 2.0 * PI, Some(100), None);
    assert!(!g.is_empty());
}

#[test]
fn test_parametric_plot_spiral() {
    // Plot a spiral
    let g = parametric_plot(
        |t| t * t.cos(),
        |t| t * t.sin(),
        0.0,
        4.0 * PI,
        Some(200),
        None,
    );
    assert!(!g.is_empty());
}

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
fn test_graphics_with_title_and_labels() {
    let mut g = plot(|x| x.sin(), 0.0, 2.0 * PI, None);
    g.set_title("Sine Wave");
    g.set_labels("x", "sin(x)");
    g.set_figsize(800, 600);

    assert_eq!(g.options().title, Some("Sine Wave".to_string()));
    assert_eq!(g.options().axes.xlabel, Some("x".to_string()));
    assert_eq!(g.options().axes.ylabel, Some("sin(x)".to_string()));
    assert_eq!(g.options().figsize, (800, 600));
}

#[test]
fn test_svg_export() {
    use std::fs;
    use std::path::PathBuf;

    let g = plot(|x| x * x, -2.0, 2.0, None);

    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_plot.svg");

    // Save to SVG
    let result = g.save(&file_path, RenderFormat::SVG);
    assert!(result.is_ok(), "Failed to save SVG: {:?}", result.err());

    // Verify file exists
    assert!(file_path.exists());

    // Clean up
    let _ = fs::remove_file(file_path);
}

#[test]
fn test_png_export() {
    use std::fs;
    use std::path::PathBuf;

    let g = plot(|x| x.sin(), 0.0, 2.0 * PI, None);

    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_plot.png");

    // Save to PNG
    let result = g.save(&file_path, RenderFormat::PNG);
    assert!(result.is_ok(), "Failed to save PNG: {:?}", result.err());

    // Verify file exists
    assert!(file_path.exists());

    // Clean up
    let _ = fs::remove_file(file_path);
}

#[test]
fn test_multiple_primitives_combined() {
    use rustmath_plot::primitives::{circle, line};
    use rustmath_plot::Point2D;

    let mut g = Graphics::new();
    g.set_title("Combined Primitives");

    // Add a line
    let line_points = vec![
        Point2D::new(-1.0, -1.0),
        Point2D::new(0.0, 0.0),
        Point2D::new(1.0, 1.0),
    ];
    g.add(line(line_points, None));

    // Add a circle
    g.add(circle((0.0, 0.0), 0.5, None));

    assert_eq!(g.len(), 2);
}

#[test]
fn test_all_plot_types_in_sequence() {
    // This test creates all plot types to ensure they work together
    let line_plot = plot(|x| x.sin(), 0.0, 2.0 * PI, None);
    assert!(!line_plot.is_empty());

    let scatter = scatter_plot(vec![(0.0, 1.0), (1.0, 2.0), (2.0, 1.5)], None, None, None);
    assert!(!scatter.is_empty());

    let bar = bar_chart(vec![(0.0, 10.0), (1.0, 15.0), (2.0, 7.0)], None, None);
    assert!(!bar.is_empty());

    let hist = histogram(vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0], None, None, None, None);
    assert!(!hist.is_empty());

    let contour = contour_plot(
        |x, y| x * x + y * y,
        (-2.0, 2.0),
        (-2.0, 2.0),
        Some(vec![1.0]),
        Some(20),
        None,
    );
    assert!(!contour.is_empty());

    let density = density_plot(|x, y| x * y, (-1.0, 1.0), (-1.0, 1.0), Some(20), None, None);
    assert!(!density.is_empty());

    let parametric =
        parametric_plot(|t| t.cos(), |t| t.sin(), 0.0, 2.0 * PI, Some(50), None);
    assert!(!parametric.is_empty());

    let implicit = implicit_plot(
        |x, y| x * x + y * y - 1.0,
        (-2.0, 2.0),
        (-2.0, 2.0),
        Some(30),
        None,
    );
    assert!(!implicit.is_empty());
}

#[test]
fn test_list_plot_creation() {
    let data = vec![(0.0, 1.0), (1.0, 4.0), (2.0, 2.0), (3.0, 5.0)];
    let g = list_plot(data, None, None);
    assert!(!g.is_empty());
}

#[test]
fn test_plot_with_custom_colors() {
    let mut opts1 = PlotOptions::default();
    opts1.color = Color::rgb(1.0, 0.0, 0.0); // Red

    let mut opts2 = PlotOptions::default();
    opts2.color = Color::rgb(0.0, 0.0, 1.0); // Blue

    let g1 = plot(|x| x.sin(), 0.0, 2.0 * PI, Some(opts1));
    let g2 = plot(|x| x.cos(), 0.0, 2.0 * PI, Some(opts2));

    assert!(!g1.is_empty());
    assert!(!g2.is_empty());
}

#[test]
fn test_complex_parametric_curves() {
    // Lissajous curve
    let lissajous = parametric_plot(
        |t| (3.0 * t).sin(),
        |t| (2.0 * t).sin(),
        0.0,
        2.0 * PI,
        Some(200),
        None,
    );
    assert!(!lissajous.is_empty());

    // Epicycloid
    let r1 = 3.0;
    let r2 = 1.0;
    let epicycloid = parametric_plot(
        |t| (r1 + r2) * t.cos() - r2 * ((r1 + r2) / r2 * t).cos(),
        |t| (r1 + r2) * t.sin() - r2 * ((r1 + r2) / r2 * t).sin(),
        0.0,
        2.0 * PI,
        Some(300),
        None,
    );
    assert!(!epicycloid.is_empty());
}

#[test]
fn test_complex_implicit_curves() {
    // Lemniscate: (x^2 + y^2)^2 = 2(x^2 - y^2)
    let lemniscate = implicit_plot(
        |x, y| {
            let r2 = x * x + y * y;
            r2 * r2 - 2.0 * (x * x - y * y)
        },
        (-2.0, 2.0),
        (-2.0, 2.0),
        Some(100),
        None,
    );
    assert!(!lemniscate.is_empty());

    // Folium of Descartes: x^3 + y^3 - 3xy = 0
    let folium = implicit_plot(
        |x, y| x.powi(3) + y.powi(3) - 3.0 * x * y,
        (-2.0, 2.0),
        (-2.0, 2.0),
        Some(100),
        None,
    );
    assert!(!folium.is_empty());
}
