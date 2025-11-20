//! Example demonstrating all 2D plot types in RustMath
//!
//! This example showcases all the plotting features from tracker 11:
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
use rustmath_plot::{PlotOptions, RenderFormat};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("RustMath 2D Plotting Examples");
    println!("=============================\n");

    // 1. Line Plot - Sine and Cosine
    println!("1. Creating line plot (sine and cosine)...");
    let mut sine_plot = plot(|x| x.sin(), 0.0, 2.0 * PI, None);
    sine_plot.set_title("Trigonometric Functions");
    sine_plot.set_labels("x", "y");
    sine_plot.set_figsize(800, 600);
    sine_plot.save("sine_plot.svg", RenderFormat::SVG)?;
    sine_plot.save("sine_plot.png", RenderFormat::PNG)?;
    println!("   ✓ Saved to sine_plot.svg and sine_plot.png");

    // 2. Scatter Plot
    println!("2. Creating scatter plot...");
    let data = vec![
        (0.0, 1.2),
        (0.5, 2.1),
        (1.0, 2.3),
        (1.5, 1.9),
        (2.0, 1.8),
        (2.5, 2.5),
        (3.0, 3.1),
    ];
    let mut scatter = scatter_plot(data, None, Some(8.0), None);
    scatter.set_title("Sample Data Points");
    scatter.set_labels("Time (s)", "Value");
    scatter.save("scatter_plot.svg", RenderFormat::SVG)?;
    println!("   ✓ Saved to scatter_plot.svg");

    // 3. Bar Chart
    println!("3. Creating bar chart...");
    let categories = vec![
        (0.0, 10.0),
        (1.0, 15.0),
        (2.0, 7.0),
        (3.0, 20.0),
        (4.0, 12.0),
    ];
    let mut bar = bar_chart(categories, Some(0.8), None);
    bar.set_title("Category Comparison");
    bar.set_labels("Category", "Value");
    bar.save("bar_chart.svg", RenderFormat::SVG)?;
    println!("   ✓ Saved to bar_chart.svg");

    // 4. Histogram
    println!("4. Creating histogram...");
    // Generate sample data with some repeated values
    let mut samples = Vec::new();
    for i in 0..100 {
        let val = (i as f64 / 10.0).sin() * 5.0 + 5.0;
        samples.push(val);
    }
    let mut hist = histogram(samples, Some(20), None, None, None);
    hist.set_title("Data Distribution");
    hist.set_labels("Value", "Frequency");
    hist.save("histogram.svg", RenderFormat::SVG)?;
    println!("   ✓ Saved to histogram.svg");

    // 5. Contour Plot
    println!("5. Creating contour plot...");
    let mut contour = contour_plot(
        |x, y| (x * x + y * y).sqrt() * (x.atan2(y) * 4.0).sin(),
        (-2.0, 2.0),
        (-2.0, 2.0),
        None, // Auto levels
        Some(80),
        None,
    );
    contour.set_title("Contour Plot: r·sin(4θ)");
    contour.set_labels("x", "y");
    contour.save("contour_plot.svg", RenderFormat::SVG)?;
    println!("   ✓ Saved to contour_plot.svg");

    // 6. Density Plot
    println!("6. Creating density plot...");
    let mut density = density_plot(
        |x, y| (-(x * x + y * y) / 2.0).exp(),
        (-3.0, 3.0),
        (-3.0, 3.0),
        Some(60),
        None,
        None,
    );
    density.set_title("2D Gaussian Distribution");
    density.set_labels("x", "y");
    density.save("density_plot.svg", RenderFormat::SVG)?;
    println!("   ✓ Saved to density_plot.svg");

    // 7. Parametric Curves
    println!("7. Creating parametric curves...");

    // Lissajous curve
    let mut lissajous = parametric_plot(
        |t| (3.0 * t).sin(),
        |t| (2.0 * t).sin(),
        0.0,
        2.0 * PI,
        Some(400),
        None,
    );
    lissajous.set_title("Lissajous Curve (3:2)");
    lissajous.set_labels("x", "y");
    lissajous.save("parametric_lissajous.svg", RenderFormat::SVG)?;
    println!("   ✓ Saved to parametric_lissajous.svg");

    // Spiral
    let mut spiral = parametric_plot(
        |t| t * t.cos() / (2.0 * PI),
        |t| t * t.sin() / (2.0 * PI),
        0.0,
        4.0 * PI,
        Some(400),
        None,
    );
    spiral.set_title("Archimedean Spiral");
    spiral.set_labels("x", "y");
    spiral.save("parametric_spiral.svg", RenderFormat::SVG)?;
    println!("   ✓ Saved to parametric_spiral.svg");

    // 8. Implicit Plots
    println!("8. Creating implicit plots...");

    // Circle
    let mut circle = implicit_plot(
        |x, y| x * x + y * y - 1.0,
        (-1.5, 1.5),
        (-1.5, 1.5),
        Some(100),
        None,
    );
    circle.set_title("Implicit: x² + y² = 1");
    circle.set_labels("x", "y");
    circle.save("implicit_circle.svg", RenderFormat::SVG)?;
    println!("   ✓ Saved to implicit_circle.svg");

    // Lemniscate
    let mut lemniscate = implicit_plot(
        |x, y| {
            let r2 = x * x + y * y;
            r2 * r2 - 2.0 * (x * x - y * y)
        },
        (-2.0, 2.0),
        (-2.0, 2.0),
        Some(150),
        None,
    );
    lemniscate.set_title("Lemniscate of Bernoulli");
    lemniscate.set_labels("x", "y");
    lemniscate.save("implicit_lemniscate.svg", RenderFormat::SVG)?;
    println!("   ✓ Saved to implicit_lemniscate.svg");

    // Folium of Descartes
    let mut folium = implicit_plot(
        |x, y| x.powi(3) + y.powi(3) - 3.0 * x * y,
        (-2.0, 2.0),
        (-2.0, 2.0),
        Some(150),
        None,
    );
    folium.set_title("Folium of Descartes");
    folium.set_labels("x", "y");
    folium.save("implicit_folium.svg", RenderFormat::SVG)?;
    println!("   ✓ Saved to implicit_folium.svg");

    // 9. Customized Plot with Colors
    println!("9. Creating customized colored plots...");
    let mut opts_red = PlotOptions::default();
    opts_red.color = Color::rgb(1.0, 0.0, 0.0);
    opts_red.thickness = 3.0;

    let mut custom = plot(|x| x.sin(), 0.0, 2.0 * PI, Some(opts_red));
    custom.set_title("Custom Styled Plot");
    custom.set_labels("x", "sin(x)");
    custom.set_background_color(Color::rgb(0.95, 0.95, 0.95));
    custom.save("custom_plot.svg", RenderFormat::SVG)?;
    println!("   ✓ Saved to custom_plot.svg");

    println!("\n✅ All examples created successfully!");
    println!("\nGenerated files:");
    println!("  - sine_plot.svg / sine_plot.png");
    println!("  - scatter_plot.svg");
    println!("  - bar_chart.svg");
    println!("  - histogram.svg");
    println!("  - contour_plot.svg");
    println!("  - density_plot.svg");
    println!("  - parametric_lissajous.svg");
    println!("  - parametric_spiral.svg");
    println!("  - implicit_circle.svg");
    println!("  - implicit_lemniscate.svg");
    println!("  - implicit_folium.svg");
    println!("  - custom_plot.svg");

    Ok(())
}
