//! Python bindings for RustMath plotting
//!
//! Provides Jupyter notebook integration via _repr_svg_()

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rustmath_plot::{Graphics, backends::SvgBackend, RenderFormat, RenderBackend};
use rustmath_plot::primitives::{line, point, circle};
use rustmath_colors::Color;
use rustmath_plot_core::PlotOptions;
use std::f64::consts::PI;
use std::cell::RefCell;

/// Python wrapper for Graphics container
/// Note: Graphics uses Box<dyn GraphicPrimitive> which isn't Send, so we use unsendable
#[pyclass(name = "Graphics", unsendable)]
pub struct PyGraphics {
    pub inner: Graphics,
}

#[pymethods]
impl PyGraphics {
    #[new]
    fn new() -> Self {
        PyGraphics {
            inner: Graphics::new(),
        }
    }

    /// Set the plot title
    fn set_title(&mut self, title: &str) {
        self.inner.set_title(title);
    }

    /// Set axis labels
    fn set_labels(&mut self, xlabel: &str, ylabel: &str) {
        self.inner.set_labels(xlabel, ylabel);
    }

    /// Set figure size in pixels
    fn set_figsize(&mut self, width: usize, height: usize) {
        self.inner.set_figsize(width, height);
    }

    /// Set whether to show axes
    fn set_axes(&mut self, show: bool) {
        self.inner.set_axes(show);
    }

    /// Set aspect ratio
    fn set_aspect_ratio(&mut self, ratio: f64) {
        self.inner.set_aspect_ratio(ratio);
    }

    /// Render to SVG string (for Jupyter display)
    fn _repr_svg_(&self) -> PyResult<String> {
        let (width, height) = self.inner.options().figsize;
        let mut backend = SvgBackend::new(width, height);

        self.inner.render(&mut backend)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let svg_bytes = backend.finalize()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        String::from_utf8(svg_bytes)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Save to file
    fn save(&self, path: &str) -> PyResult<()> {
        // Determine format from extension
        let fmt = if path.ends_with(".svg") {
            RenderFormat::SVG
        } else if path.ends_with(".png") {
            RenderFormat::PNG
        } else if path.ends_with(".jpg") || path.ends_with(".jpeg") {
            RenderFormat::JPEG
        } else {
            RenderFormat::SVG
        };

        self.inner.save(path, fmt)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Get SVG as string
    fn svg(&self) -> PyResult<String> {
        self._repr_svg_()
    }

    fn __repr__(&self) -> String {
        format!("Graphics({} primitives)", self.inner.len())
    }
}

// ========== Helper for color parsing ==========

fn parse_color(s: &str) -> Color {
    match s.to_lowercase().as_str() {
        "red" => Color::red_color(),
        "green" => Color::green_color(),
        "blue" => Color::blue_color(),
        "black" => Color::black(),
        "white" => Color::white(),
        "yellow" => Color::yellow_color(),
        "cyan" => Color::cyan_color(),
        "magenta" => Color::magenta_color(),
        "orange" => Color::rgb(1.0, 0.65, 0.0),
        "purple" => Color::rgb(0.5, 0.0, 0.5),
        "gray" | "grey" => Color::gray_color(),
        _ => Color::blue_color(),
    }
}

// ========== Simple plotting functions ==========

/// Plot data points as a line
#[pyfunction]
#[pyo3(signature = (data, color=None, thickness=None))]
fn list_plot(
    data: Vec<(f64, f64)>,
    color: Option<&str>,
    thickness: Option<f64>,
) -> PyResult<PyGraphics> {
    let mut opts = PlotOptions::default();
    if let Some(c) = color {
        opts.color = parse_color(c);
    }
    if let Some(t) = thickness {
        opts.thickness = t;
    }

    let mut g = Graphics::new();
    g.add(line(data, Some(opts)));
    Ok(PyGraphics { inner: g })
}

/// Plot points as scatter
#[pyfunction]
#[pyo3(signature = (data, color=None, size=None))]
fn scatter_plot(
    data: Vec<(f64, f64)>,
    color: Option<&str>,
    size: Option<f64>,
) -> PyResult<PyGraphics> {
    let mut opts = PlotOptions::default();
    if let Some(c) = color {
        opts.color = parse_color(c);
    }
    if let Some(s) = size {
        opts.marker_size = s;
    }

    let mut g = Graphics::new();
    g.add(point(data, Some(opts)));
    Ok(PyGraphics { inner: g })
}

/// Draw a circle
#[pyfunction]
#[pyo3(signature = (center, radius, color=None, thickness=None))]
fn draw_circle(
    center: (f64, f64),
    radius: f64,
    color: Option<&str>,
    thickness: Option<f64>,
) -> PyResult<PyGraphics> {
    let mut opts = PlotOptions::default();
    if let Some(c) = color {
        opts.color = parse_color(c);
    }
    if let Some(t) = thickness {
        opts.thickness = t;
    }

    let mut g = Graphics::new();
    g.add(circle(center, radius, Some(opts)));
    Ok(PyGraphics { inner: g })
}

/// Draw a line between points
#[pyfunction]
#[pyo3(signature = (points, color=None, thickness=None))]
fn draw_line(
    points: Vec<(f64, f64)>,
    color: Option<&str>,
    thickness: Option<f64>,
) -> PyResult<PyGraphics> {
    let mut opts = PlotOptions::default();
    if let Some(c) = color {
        opts.color = parse_color(c);
    }
    if let Some(t) = thickness {
        opts.thickness = t;
    }

    let mut g = Graphics::new();
    g.add(line(points, Some(opts)));
    Ok(PyGraphics { inner: g })
}

/// Plot points
#[pyfunction]
#[pyo3(signature = (points, color=None, size=None))]
fn draw_point(
    points: Vec<(f64, f64)>,
    color: Option<&str>,
    size: Option<f64>,
) -> PyResult<PyGraphics> {
    let mut opts = PlotOptions::default();
    if let Some(c) = color {
        opts.color = parse_color(c);
    }
    if let Some(s) = size {
        opts.marker_size = s;
    }

    let mut g = Graphics::new();
    g.add(point(points, Some(opts)));
    Ok(PyGraphics { inner: g })
}

// ========== Fractal Visualization ==========

/// Generate Mandelbrot set as point data
#[pyfunction]
#[pyo3(signature = (xrange=None, yrange=None, max_iter=None, resolution=None))]
fn mandelbrot_data(
    xrange: Option<(f64, f64)>,
    yrange: Option<(f64, f64)>,
    max_iter: Option<u32>,
    resolution: Option<usize>,
) -> Vec<(f64, f64, f64)> {
    let (x_min, x_max) = xrange.unwrap_or((-2.5, 1.0));
    let (y_min, y_max) = yrange.unwrap_or((-1.5, 1.5));
    let max_it = max_iter.unwrap_or(100);
    let res = resolution.unwrap_or(200);

    let mut data = Vec::with_capacity(res * res);

    for j in 0..res {
        let cy = y_min + (j as f64 + 0.5) * (y_max - y_min) / res as f64;
        for i in 0..res {
            let cx = x_min + (i as f64 + 0.5) * (x_max - x_min) / res as f64;

            let mut x = 0.0;
            let mut y = 0.0;
            let mut iter = 0u32;

            while x * x + y * y <= 4.0 && iter < max_it {
                let x_new = x * x - y * y + cx;
                y = 2.0 * x * y + cy;
                x = x_new;
                iter += 1;
            }

            // Return normalized iteration count
            let value = if iter == max_it {
                0.0
            } else {
                iter as f64 / max_it as f64
            };

            data.push((cx, cy, value));
        }
    }

    data
}

/// Generate Julia set as point data
#[pyfunction]
#[pyo3(signature = (c_re, c_im, xrange=None, yrange=None, max_iter=None, resolution=None))]
fn julia_data(
    c_re: f64,
    c_im: f64,
    xrange: Option<(f64, f64)>,
    yrange: Option<(f64, f64)>,
    max_iter: Option<u32>,
    resolution: Option<usize>,
) -> Vec<(f64, f64, f64)> {
    let (x_min, x_max) = xrange.unwrap_or((-2.0, 2.0));
    let (y_min, y_max) = yrange.unwrap_or((-2.0, 2.0));
    let max_it = max_iter.unwrap_or(100);
    let res = resolution.unwrap_or(200);

    let mut data = Vec::with_capacity(res * res);

    for j in 0..res {
        let y0 = y_min + (j as f64 + 0.5) * (y_max - y_min) / res as f64;
        for i in 0..res {
            let x0 = x_min + (i as f64 + 0.5) * (x_max - x_min) / res as f64;

            let mut x = x0;
            let mut y = y0;
            let mut iter = 0u32;

            while x * x + y * y <= 4.0 && iter < max_it {
                let x_new = x * x - y * y + c_re;
                y = 2.0 * x * y + c_im;
                x = x_new;
                iter += 1;
            }

            let value = if iter == max_it {
                0.0
            } else {
                iter as f64 / max_it as f64
            };

            data.push((x0, y0, value));
        }
    }

    data
}

/// Plot roots in the complex plane
#[pyfunction]
#[pyo3(signature = (roots, color=None, size=None))]
fn plot_roots(
    roots: Vec<(f64, f64)>,
    color: Option<&str>,
    size: Option<f64>,
) -> PyResult<PyGraphics> {
    let mut opts = PlotOptions::default();
    opts.color = parse_color(color.unwrap_or("red"));
    if let Some(s) = size {
        opts.marker_size = s;
    }

    let mut g = Graphics::new();
    g.set_aspect_ratio(1.0);
    g.add(point(roots, Some(opts)));

    // Add unit circle
    let mut circle_opts = PlotOptions::default();
    circle_opts.color = Color::gray_color();
    circle_opts.thickness = 0.5;
    g.add(circle((0.0, 0.0), 1.0, Some(circle_opts)));

    Ok(PyGraphics { inner: g })
}

/// Register plotting module
pub fn register_plot_module(m: &PyModule) -> PyResult<()> {
    m.add_class::<PyGraphics>()?;

    // Basic plotting
    m.add_function(wrap_pyfunction!(list_plot, m)?)?;
    m.add_function(wrap_pyfunction!(scatter_plot, m)?)?;

    // Primitives
    m.add_function(wrap_pyfunction!(draw_circle, m)?)?;
    m.add_function(wrap_pyfunction!(draw_line, m)?)?;
    m.add_function(wrap_pyfunction!(draw_point, m)?)?;

    // Fractals (return data for plotting)
    m.add_function(wrap_pyfunction!(mandelbrot_data, m)?)?;
    m.add_function(wrap_pyfunction!(julia_data, m)?)?;

    // Complex plane
    m.add_function(wrap_pyfunction!(plot_roots, m)?)?;

    Ok(())
}
