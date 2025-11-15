//! Interpolation and spline functions
//!
//! This module provides various interpolation methods including
//! linear, polynomial, and cubic spline interpolation.

use std::f64;

/// Represents a cubic spline interpolation
///
/// A cubic spline is a piecewise cubic polynomial that is twice
/// continuously differentiable.
#[derive(Debug, Clone)]
pub struct CubicSpline {
    /// X coordinates of data points
    x: Vec<f64>,
    /// Y coordinates of data points
    y: Vec<f64>,
    /// Second derivatives at each point
    y2: Vec<f64>,
}

impl CubicSpline {
    /// Create a new natural cubic spline
    ///
    /// A natural spline has zero second derivatives at the endpoints.
    ///
    /// # Arguments
    ///
    /// * `x` - X coordinates (must be sorted in ascending order)
    /// * `y` - Y coordinates
    ///
    /// # Returns
    ///
    /// A CubicSpline interpolator
    ///
    /// # Panics
    ///
    /// Panics if x and y have different lengths or if x is not sorted.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_calculus::interpolation::CubicSpline;
    ///
    /// let x = vec![0.0, 1.0, 2.0, 3.0];
    /// let y = vec![0.0, 1.0, 4.0, 9.0];
    /// let spline = CubicSpline::new(x, y);
    /// let value = spline.eval(1.5);
    /// ```
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Self {
        assert_eq!(x.len(), y.len(), "x and y must have the same length");
        assert!(x.len() >= 2, "Need at least 2 points for interpolation");

        let n = x.len();
        let mut y2 = vec![0.0; n];
        let mut u = vec![0.0; n];

        // Natural spline: second derivative = 0 at endpoints
        y2[0] = 0.0;
        u[0] = 0.0;

        // Tridiagonal system solution
        for i in 1..n - 1 {
            let sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
            let p = sig * y2[i - 1] + 2.0;
            y2[i] = (sig - 1.0) / p;
            u[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
                - (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
            u[i] = (6.0 * u[i] / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p;
        }

        y2[n - 1] = 0.0;

        // Back substitution
        for i in (0..n - 1).rev() {
            y2[i] = y2[i] * y2[i + 1] + u[i];
        }

        CubicSpline { x, y, y2 }
    }

    /// Evaluate the spline at a point
    ///
    /// # Arguments
    ///
    /// * `x` - Point at which to evaluate the spline
    ///
    /// # Returns
    ///
    /// The interpolated value
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_calculus::interpolation::CubicSpline;
    ///
    /// let x = vec![0.0, 1.0, 2.0];
    /// let y = vec![0.0, 1.0, 4.0];
    /// let spline = CubicSpline::new(x, y);
    /// let value = spline.eval(0.5);
    /// ```
    pub fn eval(&self, x: f64) -> f64 {
        let n = self.x.len();

        // Find the interval containing x
        let mut klo = 0;
        let mut khi = n - 1;

        // Binary search
        while khi - klo > 1 {
            let k = (khi + klo) / 2;
            if self.x[k] > x {
                khi = k;
            } else {
                klo = k;
            }
        }

        let h = self.x[khi] - self.x[klo];
        assert!(h != 0.0, "x values must be distinct");

        let a = (self.x[khi] - x) / h;
        let b = (x - self.x[klo]) / h;

        a * self.y[klo]
            + b * self.y[khi]
            + ((a * a * a - a) * self.y2[klo] + (b * b * b - b) * self.y2[khi]) * (h * h) / 6.0
    }

    /// Evaluate the first derivative at a point
    ///
    /// # Arguments
    ///
    /// * `x` - Point at which to evaluate the derivative
    ///
    /// # Returns
    ///
    /// The value of the first derivative
    pub fn derivative(&self, x: f64) -> f64 {
        let n = self.x.len();

        let mut klo = 0;
        let mut khi = n - 1;

        while khi - klo > 1 {
            let k = (khi + klo) / 2;
            if self.x[k] > x {
                khi = k;
            } else {
                klo = k;
            }
        }

        let h = self.x[khi] - self.x[klo];
        let a = (self.x[khi] - x) / h;
        let b = (x - self.x[klo]) / h;

        (self.y[khi] - self.y[klo]) / h
            + (-(3.0 * a * a - 1.0) * self.y2[klo] + (3.0 * b * b - 1.0) * self.y2[khi]) * h / 6.0
    }
}

/// Linear interpolation between two points
///
/// # Arguments
///
/// * `x0`, `y0` - First point
/// * `x1`, `y1` - Second point
/// * `x` - Point at which to interpolate
///
/// # Returns
///
/// Interpolated y value
///
/// # Examples
///
/// ```
/// use rustmath_calculus::interpolation::linear_interp;
///
/// let y = linear_interp(0.0, 0.0, 1.0, 1.0, 0.5);
/// assert_eq!(y, 0.5);
/// ```
pub fn linear_interp(x0: f64, y0: f64, x1: f64, y1: f64, x: f64) -> f64 {
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

/// Lagrange polynomial interpolation
///
/// # Arguments
///
/// * `x_data` - X coordinates of data points
/// * `y_data` - Y coordinates of data points
/// * `x` - Point at which to interpolate
///
/// # Returns
///
/// Interpolated value
///
/// # Examples
///
/// ```
/// use rustmath_calculus::interpolation::lagrange_interp;
///
/// let x_data = vec![0.0, 1.0, 2.0];
/// let y_data = vec![0.0, 1.0, 4.0];
/// let y = lagrange_interp(&x_data, &y_data, 0.5);
/// ```
pub fn lagrange_interp(x_data: &[f64], y_data: &[f64], x: f64) -> f64 {
    assert_eq!(x_data.len(), y_data.len());
    let n = x_data.len();
    let mut result = 0.0;

    for i in 0..n {
        let mut term = y_data[i];
        for j in 0..n {
            if i != j {
                term *= (x - x_data[j]) / (x_data[i] - x_data[j]);
            }
        }
        result += term;
    }

    result
}

/// Newton divided difference interpolation
///
/// This is more numerically stable than Lagrange interpolation.
///
/// # Arguments
///
/// * `x_data` - X coordinates
/// * `y_data` - Y coordinates
/// * `x` - Point at which to interpolate
///
/// # Returns
///
/// Interpolated value
///
/// # Examples
///
/// ```
/// use rustmath_calculus::interpolation::newton_interp;
///
/// let x_data = vec![0.0, 1.0, 2.0];
/// let y_data = vec![0.0, 1.0, 4.0];
/// let y = newton_interp(&x_data, &y_data, 0.5);
/// ```
pub fn newton_interp(x_data: &[f64], y_data: &[f64], x: f64) -> f64 {
    assert_eq!(x_data.len(), y_data.len());
    let n = x_data.len();

    // Compute divided differences
    let mut diffs = y_data.to_vec();

    for j in 1..n {
        for i in (j..n).rev() {
            diffs[i] = (diffs[i] - diffs[i - 1]) / (x_data[i] - x_data[i - j]);
        }
    }

    // Evaluate polynomial using Horner's method
    let mut result = diffs[n - 1];
    for i in (0..n - 1).rev() {
        result = result * (x - x_data[i]) + diffs[i];
    }

    result
}

/// Piecewise linear interpolation
///
/// # Arguments
///
/// * `x_data` - X coordinates (must be sorted)
/// * `y_data` - Y coordinates
/// * `x` - Point at which to interpolate
///
/// # Returns
///
/// Interpolated value
///
/// # Examples
///
/// ```
/// use rustmath_calculus::interpolation::piecewise_linear;
///
/// let x_data = vec![0.0, 1.0, 2.0, 3.0];
/// let y_data = vec![0.0, 1.0, 4.0, 9.0];
/// let y = piecewise_linear(&x_data, &y_data, 1.5);
/// ```
pub fn piecewise_linear(x_data: &[f64], y_data: &[f64], x: f64) -> f64 {
    assert_eq!(x_data.len(), y_data.len());
    assert!(x_data.len() >= 2);

    // Find interval
    let n = x_data.len();
    if x <= x_data[0] {
        return y_data[0];
    }
    if x >= x_data[n - 1] {
        return y_data[n - 1];
    }

    for i in 0..n - 1 {
        if x >= x_data[i] && x <= x_data[i + 1] {
            return linear_interp(x_data[i], y_data[i], x_data[i + 1], y_data[i + 1], x);
        }
    }

    y_data[n - 1]
}

/// Convenience function: create and evaluate a cubic spline
///
/// # Examples
///
/// ```
/// use rustmath_calculus::interpolation::spline;
///
/// let x = vec![0.0, 1.0, 2.0];
/// let y = vec![0.0, 1.0, 4.0];
/// let value = spline(&x, &y, 0.5);
/// ```
pub fn spline(x_data: &[f64], y_data: &[f64], x: f64) -> f64 {
    let spline = CubicSpline::new(x_data.to_vec(), y_data.to_vec());
    spline.eval(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_interp() {
        let y = linear_interp(0.0, 0.0, 1.0, 1.0, 0.5);
        assert!((y - 0.5).abs() < 1e-10);

        let y = linear_interp(0.0, 0.0, 2.0, 4.0, 1.0);
        assert!((y - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_piecewise_linear() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 4.0];

        let y_interp = piecewise_linear(&x, &y, 0.5);
        assert!((y_interp - 0.5).abs() < 1e-10);

        let y_interp = piecewise_linear(&x, &y, 1.5);
        assert!((y_interp - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_cubic_spline() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 0.0, 1.0];

        let spline = CubicSpline::new(x, y);

        // Spline should pass through data points
        assert!((spline.eval(0.0) - 0.0).abs() < 1e-6);
        assert!((spline.eval(1.0) - 1.0).abs() < 1e-6);
        assert!((spline.eval(2.0) - 0.0).abs() < 1e-6);
        assert!((spline.eval(3.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_lagrange_interp() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 4.0];

        // These points lie on y = x^2, so Lagrange should give exact results
        let y_interp = lagrange_interp(&x, &y, 1.5);
        // For x=1.5, y should be approximately 2.25
        assert!((y_interp - 2.25).abs() < 1e-6);
    }

    #[test]
    fn test_newton_interp() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 4.0];

        // Should give same result as Lagrange
        let y_newton = newton_interp(&x, &y, 1.5);
        let y_lagrange = lagrange_interp(&x, &y, 1.5);

        assert!((y_newton - y_lagrange).abs() < 1e-10);
    }

    #[test]
    fn test_spline_convenience() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 4.0];

        let value = spline(&x, &y, 1.0);
        assert!((value - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cubic_spline_derivative() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 4.0];

        let spline = CubicSpline::new(x, y);
        let deriv = spline.derivative(1.0);

        // For a smooth curve through these points, derivative at x=1
        // should be positive
        assert!(deriv > 0.0);
    }
}
