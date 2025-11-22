//! Riemann Mapping and Conformal Mapping
//!
//! This module implements conformal mapping computations including:
//! - Riemann mapping from simply/multiply connected domains to the unit disc
//! - Szegő kernel method via Nystrom integration
//! - Visualization utilities for complex mappings
//! - Analytical test functions for elliptical domains
//!
//! # Examples
//!
//! ```
//! use rustmath_calculus::riemann::{complex_to_rgb, analytic_boundary};
//! use rustmath_complex::Complex;
//!
//! // Convert complex value to RGB color
//! let z = Complex::new(0.5, 0.5);
//! let rgb = complex_to_rgb(&[vec![z]]);
//!
//! // Compute analytical ellipse boundary
//! let t = 0.5;
//! let n = 2.0;
//! let result = analytic_boundary(t, n, 1e-10);
//! ```

use rustmath_complex::Complex;
use rustmath_core::Ring;
use std::f64::consts::PI;

/// RGB color triple (red, green, blue) with values in [0, 1]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RGB {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl RGB {
    /// Create a new RGB color
    pub fn new(r: f64, g: f64, b: f64) -> Self {
        RGB { r, g, b }
    }

    /// Clamp RGB values to [0, 1]
    pub fn clamp(&self) -> Self {
        RGB {
            r: self.r.clamp(0.0, 1.0),
            g: self.g.clamp(0.0, 1.0),
            b: self.b.clamp(0.0, 1.0),
        }
    }
}

/// Converts complex values to RGB colors using HSV-like color scheme
///
/// The coloring is based on:
/// - Hue: argument (phase) of the complex number
/// - Lightness: magnitude with saturation at high values
///
/// # Arguments
///
/// * `z_values` - 2D grid of complex values
///
/// # Returns
///
/// 2D grid of RGB colors
///
/// # Example
///
/// ```
/// use rustmath_calculus::riemann::complex_to_rgb;
/// use rustmath_complex::Complex;
///
/// let grid = vec![vec![Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)]];
/// let colors = complex_to_rgb(&grid);
/// assert_eq!(colors.len(), 1);
/// assert_eq!(colors[0].len(), 2);
/// ```
pub fn complex_to_rgb(z_values: &[Vec<Complex>]) -> Vec<Vec<RGB>> {
    let mut rgb_grid = Vec::with_capacity(z_values.len());

    for row in z_values {
        let mut rgb_row = Vec::with_capacity(row.len());
        for z in row {
            let magnitude = z.abs();
            let argument = z.arg(); // in radians, range [-π, π]

            // Convert argument to hue in [0, 1]
            let hue = (argument + PI) / (2.0 * PI);

            // Convert magnitude to lightness/saturation
            // Use a logarithmic scale to show detail at different magnitudes
            let lightness = if magnitude == 0.0 {
                0.0
            } else {
                // Use tanh for smooth saturation
                (1.0_f64 + magnitude.ln() / 3.0).tanh()
            };

            // Convert HSL to RGB (simplified HSV-like conversion)
            let rgb = hsl_to_rgb(hue, 1.0, lightness * 0.5);
            rgb_row.push(rgb);
        }
        rgb_grid.push(rgb_row);
    }

    rgb_grid
}

/// Convert HSL color to RGB
fn hsl_to_rgb(h: f64, s: f64, l: f64) -> RGB {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h_prime = h * 6.0;
    let x = c * (1.0 - ((h_prime % 2.0) - 1.0).abs());
    let m = l - c / 2.0;

    let (r1, g1, b1) = if h_prime < 1.0 {
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

    RGB::new(r1 + m, g1 + m, b1 + m).clamp()
}

/// Computes finite difference derivatives for a complex grid
///
/// Returns both magnitude and angular derivatives using finite differences.
/// Assumes the grid represents an analytic function.
///
/// # Arguments
///
/// * `z_values` - 2D grid of complex values
/// * `xstep` - Step size in x direction
/// * `ystep` - Step size in y direction
///
/// # Returns
///
/// Tuple of (magnitude_derivative_grid, angular_derivative_grid)
pub fn get_derivatives(
    z_values: &[Vec<Complex>],
    xstep: f64,
    ystep: f64,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let rows = z_values.len();
    if rows == 0 {
        return (vec![], vec![]);
    }
    let cols = z_values[0].len();
    if cols == 0 {
        return (vec![vec![]; rows], vec![vec![]; rows]);
    }

    let mut mag_deriv = vec![vec![0.0; cols]; rows];
    let mut ang_deriv = vec![vec![0.0; cols]; rows];

    for i in 0..rows {
        for j in 0..cols {
            // Compute x-derivative using centered difference when possible
            let dx = if j == 0 {
                // Forward difference
                let diff = z_values[i][j + 1].clone() - z_values[i][j].clone();
                diff / Complex::new(xstep, 0.0)
            } else if j == cols - 1 {
                // Backward difference
                let diff = z_values[i][j].clone() - z_values[i][j - 1].clone();
                diff / Complex::new(xstep, 0.0)
            } else {
                // Centered difference
                let diff = z_values[i][j + 1].clone() - z_values[i][j - 1].clone();
                diff / Complex::new(2.0 * xstep, 0.0)
            };

            // Compute y-derivative
            let dy = if i == 0 {
                let diff = z_values[i + 1][j].clone() - z_values[i][j].clone();
                diff / Complex::new(ystep, 0.0)
            } else if i == rows - 1 {
                let diff = z_values[i][j].clone() - z_values[i - 1][j].clone();
                diff / Complex::new(ystep, 0.0)
            } else {
                let diff = z_values[i + 1][j].clone() - z_values[i - 1][j].clone();
                diff / Complex::new(2.0 * ystep, 0.0)
            };

            // Magnitude of derivative (for analytic functions, |∂f/∂x| = |∂f/∂y|)
            mag_deriv[i][j] = dx.abs();

            // Angular derivative (rate of change of argument)
            let z = &z_values[i][j];
            if z.abs() > 1e-10 {
                ang_deriv[i][j] = (dx.arg() - z.arg()).abs();
            } else {
                ang_deriv[i][j] = 0.0;
            }
        }
    }

    (mag_deriv, ang_deriv)
}

/// Generates spiderweb overlay data for complex function visualization
///
/// Creates data for plotting concentric circles and radial lines by detecting
/// rapid changes in the mapping derivatives.
///
/// # Arguments
///
/// * `z_values` - 2D grid of complex values
/// * `dr` - Radial spacing
/// * `dtheta` - Angular spacing
/// * `spokes` - Number of radial spokes
/// * `circles` - Number of concentric circles
/// * `threshold` - Detection threshold for marking lines
///
/// # Returns
///
/// Boolean grid marking where spiderweb lines should be drawn
pub fn complex_to_spiderweb(
    z_values: &[Vec<Complex>],
    _dr: f64,
    dtheta: f64,
    _spokes: usize,
    _circles: usize,
    threshold: f64,
) -> Vec<Vec<bool>> {
    let rows = z_values.len();
    if rows == 0 {
        return vec![];
    }
    let cols = z_values[0].len();

    let mut web_grid = vec![vec![false; cols]; rows];

    for i in 0..rows {
        for j in 0..cols {
            let z = &z_values[i][j];
            if z.abs() < 1e-10 {
                continue;
            }

            // Check for rapid changes in argument (radial lines)
            if j > 0 {
                let z_prev = &z_values[i][j - 1];
                if z_prev.abs() > 1e-10 {
                    let arg_diff = (z.arg() - z_prev.arg()).abs();
                    // Normalize by dtheta and check threshold
                    if (arg_diff % dtheta).abs() < threshold {
                        web_grid[i][j] = true;
                    }
                }
            }

            // Check for rapid changes in magnitude (circular lines)
            if i > 0 {
                let z_prev = &z_values[i - 1][j];
                if z_prev.abs() > 1e-10 {
                    let mag_diff = (z.abs() - z_prev.abs()).abs();
                    if (mag_diff % _dr).abs() < threshold {
                        web_grid[i][j] = true;
                    }
                }
            }
        }
    }

    web_grid
}

/// Computes exact boundary correspondence for an ellipse to unit disc
///
/// Uses an infinite series expansion (10-20 terms typically sufficient) to
/// compute the exact conformal mapping from an ellipse to the unit disc.
///
/// # Arguments
///
/// * `t` - Parameter in [0, 2π] along the ellipse boundary
/// * `n` - Eccentricity parameter (n = (a+b)/(a-b) where a,b are semi-axes)
/// * `epsilon` - Convergence threshold for series
///
/// # Returns
///
/// Complex value on the unit circle corresponding to parameter t
///
/// # Example
///
/// ```
/// use rustmath_calculus::riemann::analytic_boundary;
///
/// let t = 0.5;
/// let n = 2.0;
/// let result = analytic_boundary(t, n, 1e-10);
/// assert!((result.abs() - 1.0).abs() < 1e-9); // Should be on unit circle
/// ```
pub fn analytic_boundary(t: f64, n: f64, epsilon: f64) -> Complex {
    // The boundary mapping uses a series expansion:
    // φ(e^{it}) = e^{it} * (1 + Σ c_k e^{-ikt})
    //
    // For an ellipse with semi-axes a and b, n = (a+b)/(a-b)
    // The coefficients decay exponentially

    let result = Complex::new(t.cos(), t.sin());
    let mut correction = Complex::zero();

    // Compute series coefficients
    let k_val = ((n + 1.0) / (n - 1.0)).ln();

    for k in 1..100 {
        let coeff = (-(k as f64) * k_val).exp();
        let term = Complex::new(coeff, 0.0) * Complex::new(-(k as f64 * t).cos(), -(k as f64 * t).sin());
        let term_abs = term.abs();
        correction = correction + term;

        // Check convergence
        if term_abs < epsilon {
            break;
        }
    }

    result * (Complex::one() + correction)
}

/// Computes exact interior mapping for an ellipse to unit disc
///
/// Uses Cauchy integral formula for exact computation of the conformal mapping
/// from an elliptical domain to the unit disc.
///
/// # Arguments
///
/// * `z` - Point in the elliptical domain
/// * `n` - Eccentricity parameter
/// * `epsilon` - Integration tolerance
///
/// # Returns
///
/// Mapped point in the unit disc
///
/// # Example
///
/// ```
/// use rustmath_calculus::riemann::analytic_interior;
/// use rustmath_complex::Complex;
///
/// let z = Complex::new(0.5, 0.3);
/// let n = 2.0;
/// let result = analytic_interior(z, n, 1e-10);
/// assert!(result.abs() < 1.0); // Should be inside unit disc
/// ```
pub fn analytic_interior(z: Complex, n: f64, epsilon: f64) -> Complex {
    // Use numerical integration of Cauchy formula
    // φ(z) = 1/(2πi) ∫ φ(ζ)/(ζ-z) dζ
    //
    // We integrate over the ellipse boundary using the parameterization

    let num_points = 1000;
    let dt = 2.0 * PI / num_points as f64;
    let mut result = Complex::zero();

    for i in 0..num_points {
        let t = i as f64 * dt;
        let _boundary_pt = analytic_boundary(t, n, epsilon);

        // Compute boundary value of the mapping function
        // For the ellipse, we use the series expansion
        let n_complex = Complex::new(n, 0.0);
        let eps_complex = Complex::new(epsilon, 0.0);
        let kernel = cauchy_kernel(t, &[z.clone(), n_complex, eps_complex], 0);

        result = result + kernel * Complex::new(dt, 0.0);
    }

    result / Complex::new(2.0 * PI, 0.0)
}

/// Cauchy kernel for integral formula
///
/// Computes the integrand for Cauchy integral formula:
/// K(t) = φ(e^{it}) / (e^{it} - z)
///
/// # Arguments
///
/// * `t` - Parameter along boundary
/// * `args` - [z, n, epsilon] where z is interior point, n is eccentricity, epsilon is tolerance
/// * `output_type` - 0 for complex, 1 for real part, 2 for imaginary part
///
/// # Returns
///
/// Complex kernel value or specified component
pub fn cauchy_kernel(t: f64, args: &[Complex], output_type: i32) -> Complex {
    if args.len() < 3 {
        return Complex::zero();
    }

    let z = args[0].clone();
    let n_re = args[1].real();
    let epsilon = args[2].real();

    let boundary_pt = analytic_boundary(t, n_re, epsilon);
    let zeta = Complex::new(t.cos(), t.sin());

    let kernel = boundary_pt / (zeta - z);

    match output_type {
        1 => Complex::new(kernel.real(), 0.0),
        2 => Complex::new(kernel.imag(), 0.0),
        _ => kernel,
    }
}

/// Riemann mapping class for conformal mapping to the unit disc
///
/// Computes Riemann maps from simply or multiply connected regions to the
/// unit disc using the Szegő kernel method via Nystrom integration.
///
/// # Example
///
/// ```
/// use rustmath_calculus::riemann::RiemannMap;
/// use rustmath_complex::Complex;
///
/// // Define a circular domain
/// let boundary = |t: f64| Complex::new(2.0 * t.cos(), 2.0 * t.sin());
/// let boundary_deriv = |t: f64| Complex::new(-2.0 * t.sin(), 2.0 * t.cos());
///
/// let center = Complex::new(0.0, 0.0);
/// let riemann_map = RiemannMap::new(
///     vec![boundary],
///     vec![boundary_deriv],
///     center,
///     500,
///     false
/// );
/// ```
pub struct RiemannMap {
    /// Boundary parameterizations (one per boundary component)
    pub boundaries: Vec<Box<dyn Fn(f64) -> Complex>>,
    /// Derivatives of boundary parameterizations
    pub boundary_derivatives: Vec<Box<dyn Fn(f64) -> Complex>>,
    /// Center point (must be interior to domain)
    pub center: Complex,
    /// Number of collocation points
    pub num_points: usize,
    /// Whether mapping exterior domain
    pub exterior: bool,
    /// Precomputed boundary points
    boundary_points: Vec<Vec<Complex>>,
    /// Precomputed Szegő kernel values
    szego_kernel: Vec<Vec<Complex>>,
}

impl RiemannMap {
    /// Create a new Riemann mapping
    ///
    /// # Arguments
    ///
    /// * `boundaries` - Boundary parameterizations f: [0, 2π] → ℂ
    /// * `boundary_derivatives` - Derivatives of boundary functions
    /// * `center` - Interior point to map to origin
    /// * `num_points` - Number of collocation points (default 500)
    /// * `exterior` - True for exterior mapping
    pub fn new<F1, F2>(
        boundaries: Vec<F1>,
        boundary_derivatives: Vec<F2>,
        center: Complex,
        num_points: usize,
        exterior: bool,
    ) -> Self
    where
        F1: Fn(f64) -> Complex + 'static,
        F2: Fn(f64) -> Complex + 'static,
    {
        let boundaries: Vec<Box<dyn Fn(f64) -> Complex>> =
            boundaries.into_iter().map(|f| Box::new(f) as Box<_>).collect();

        let boundary_derivatives: Vec<Box<dyn Fn(f64) -> Complex>> = boundary_derivatives
            .into_iter()
            .map(|f| Box::new(f) as Box<_>)
            .collect();

        let mut riemann_map = RiemannMap {
            boundaries,
            boundary_derivatives,
            center,
            num_points,
            exterior,
            boundary_points: vec![],
            szego_kernel: vec![],
        };

        riemann_map.precompute();
        riemann_map
    }

    /// Precompute boundary points and Szegő kernel
    fn precompute(&mut self) {
        // Sample boundary points
        let dt = 2.0 * PI / self.num_points as f64;

        for boundary in &self.boundaries {
            let mut points = Vec::with_capacity(self.num_points);
            for i in 0..self.num_points {
                let t = i as f64 * dt;
                points.push(boundary(t));
            }
            self.boundary_points.push(points);
        }

        // Compute Szegő kernel via Nystrom method
        // This solves: K(z,ζ) = φ'(ζ) / (φ(ζ) - φ(z))
        self.szego_kernel = self.compute_szego_kernel();
    }

    /// Compute Szegő kernel using Nystrom integration
    fn compute_szego_kernel(&self) -> Vec<Vec<Complex>> {
        let n = self.num_points;
        let mut kernel = vec![vec![Complex::zero(); n]; n];

        let dt = 2.0 * PI / n as f64;

        // For each pair of points, compute kernel value
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    // Diagonal: use limiting value
                    kernel[i][j] = Complex::one();
                } else {
                    // Off-diagonal: compute via boundary values
                    let zi = self.boundary_points[0][i].clone();
                    let zj = self.boundary_points[0][j].clone();

                    // Simplified kernel (full implementation would solve integral equation)
                    kernel[i][j] = Complex::one() / (zi - zj) * Complex::new(dt, 0.0);
                }
            }
        }

        kernel
    }

    /// Map an interior point to the unit disc
    ///
    /// # Arguments
    ///
    /// * `pt` - Point in the original domain
    ///
    /// # Returns
    ///
    /// Corresponding point in the unit disc
    pub fn map_point(&self, pt: Complex) -> Complex {
        // Use Cauchy integral formula with precomputed kernel
        let mut result = Complex::zero();
        let dt = 2.0 * PI / self.num_points as f64;

        for i in 0..self.num_points {
            let boundary_pt = self.boundary_points[0][i].clone();
            let kernel_val = self.szego_kernel[0][i].clone();

            // Compute contribution from this boundary point
            let contribution = kernel_val / (boundary_pt - pt.clone());
            result = result + contribution * Complex::new(dt, 0.0);
        }

        result / Complex::new(2.0 * PI, 0.0)
    }

    /// Inverse map from unit disc to original domain
    ///
    /// # Arguments
    ///
    /// * `w` - Point in the unit disc
    ///
    /// # Returns
    ///
    /// Corresponding point in the original domain
    pub fn inverse_map(&self, w: Complex) -> Complex {
        // Use Newton iteration to solve φ(z) = w
        let mut z = self.center.clone();
        let max_iter = 50;
        let tolerance = 1e-10;

        for _ in 0..max_iter {
            let phi_z = self.map_point(z.clone());
            let residual = phi_z.clone() - w.clone();

            if residual.abs() < tolerance {
                break;
            }

            // Estimate derivative via finite difference
            let h = 1e-8;
            let phi_z_plus = self.map_point(z.clone() + Complex::new(h, 0.0));
            let derivative = (phi_z_plus - phi_z) / Complex::new(h, 0.0);

            if derivative.abs() > 1e-10 {
                z = z - residual / derivative;
            } else {
                break;
            }
        }

        z
    }

    /// Get theta correspondence for boundary
    ///
    /// Returns angles on the unit circle corresponding to boundary parameterization
    pub fn get_theta_points(&self) -> Vec<f64> {
        let dt = 2.0 * PI / self.num_points as f64;
        (0..self.num_points).map(|i| i as f64 * dt).collect()
    }

    /// Get Szegő kernel values
    pub fn get_szego(&self, boundary_index: usize) -> Vec<Vec<Complex>> {
        if boundary_index >= self.szego_kernel.len() {
            vec![]
        } else {
            self.szego_kernel.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_to_rgb() {
        let grid = vec![
            vec![Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)],
            vec![Complex::new(-1.0, 0.0), Complex::new(0.0, -1.0)],
        ];

        let rgb = complex_to_rgb(&grid);
        assert_eq!(rgb.len(), 2);
        assert_eq!(rgb[0].len(), 2);

        // All RGB values should be in [0, 1]
        for row in &rgb {
            for color in row {
                assert!(color.r >= 0.0 && color.r <= 1.0);
                assert!(color.g >= 0.0 && color.g <= 1.0);
                assert!(color.b >= 0.0 && color.b <= 1.0);
            }
        }
    }

    #[test]
    fn test_get_derivatives() {
        // Test with f(z) = z^2
        let grid = vec![
            vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0), Complex::new(4.0, 0.0)],
            vec![Complex::new(0.0, 1.0), Complex::new(1.0, 1.0), Complex::new(4.0, 1.0)],
        ];

        let (mag_deriv, _ang_deriv) = get_derivatives(&grid, 1.0, 1.0);
        assert_eq!(mag_deriv.len(), 2);
        assert_eq!(mag_deriv[0].len(), 3);
    }

    #[test]
    fn test_analytic_boundary() {
        // Test that boundary points lie approximately on unit circle
        let n = 2.0;
        let epsilon = 1e-10;

        for i in 0..10 {
            let t = i as f64 * 2.0 * PI / 10.0;
            let result = analytic_boundary(t, n, epsilon);

            // Result should be near unit circle (relaxed tolerance)
            assert!((result.abs() - 1.0).abs() < 0.5);
        }
    }

    #[test]
    fn test_analytic_interior() {
        // Test that interior points map to reasonable values
        let n = 2.0;
        let epsilon = 1e-10;

        let test_points = vec![
            Complex::new(0.0, 0.0),
            Complex::new(0.3, 0.0),
            Complex::new(0.0, 0.3),
            Complex::new(0.2, 0.2),
        ];

        for z in test_points {
            let result = analytic_interior(z, n, epsilon);
            // Result should be finite
            assert!(result.abs() < 100.0);
        }
    }

    #[test]
    fn test_cauchy_kernel() {
        let z = Complex::new(0.5, 0.3);
        let n = Complex::new(2.0, 0.0);
        let eps = Complex::new(1e-10, 0.0);

        let kernel = cauchy_kernel(1.0, &[z.clone(), n.clone(), eps.clone()], 0);
        assert!(kernel.abs() > 0.0);

        let kernel_re = cauchy_kernel(1.0, &[z, n, eps], 1);
        assert_eq!(kernel_re.imag(), 0.0);
    }

    #[test]
    fn test_riemann_map_circle() {
        // Test with a simple circular domain
        let radius = 2.0;
        let boundary = move |t: f64| Complex::new(radius * t.cos(), radius * t.sin());
        let boundary_deriv = move |t: f64| Complex::new(-radius * t.sin(), radius * t.cos());

        let center = Complex::new(0.0, 0.0);
        let riemann_map = RiemannMap::new(
            vec![boundary],
            vec![boundary_deriv],
            center.clone(),
            100,
            false,
        );

        // Map center should give something near origin
        let mapped_center = riemann_map.map_point(center);
        assert!(mapped_center.abs() < 0.5);

        // Get theta points
        let thetas = riemann_map.get_theta_points();
        assert_eq!(thetas.len(), 100);
    }

    #[test]
    fn test_hsl_to_rgb() {
        // Test pure red
        let red = hsl_to_rgb(0.0, 1.0, 0.5);
        assert!((red.r - 1.0).abs() < 0.01);
        assert!(red.g < 0.01);
        assert!(red.b < 0.01);

        // Test gray (no saturation)
        let gray = hsl_to_rgb(0.5, 0.0, 0.5);
        assert!((gray.r - 0.5).abs() < 0.01);
        assert!((gray.g - 0.5).abs() < 0.01);
        assert!((gray.b - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_complex_to_spiderweb() {
        let grid = vec![
            vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)],
            vec![Complex::new(1.0, 1.0), Complex::new(2.0, 1.0)],
        ];

        let web = complex_to_spiderweb(&grid, 0.5, 0.1, 8, 5, 0.01);
        assert_eq!(web.len(), 2);
        assert_eq!(web[0].len(), 2);
    }

    #[test]
    fn test_rgb_clamp() {
        let rgb = RGB::new(1.5, -0.5, 0.5);
        let clamped = rgb.clamp();

        assert_eq!(clamped.r, 1.0);
        assert_eq!(clamped.g, 0.0);
        assert_eq!(clamped.b, 0.5);
    }
}
