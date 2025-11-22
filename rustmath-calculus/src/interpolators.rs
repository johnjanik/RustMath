//! Advanced interpolators
//!
//! This module provides specialized interpolation methods including
//! complex cubic splines and polygon splines.

use crate::interpolation::CubicSpline;

/// Complex cubic spline interpolator
///
/// This provides cubic spline interpolation for complex-valued functions
/// represented as separate real and imaginary parts.
#[derive(Debug, Clone)]
pub struct CCSpline {
    /// Spline for real part
    real_spline: CubicSpline,
    /// Spline for imaginary part
    imag_spline: CubicSpline,
}

impl CCSpline {
    /// Create a new complex cubic spline
    ///
    /// # Arguments
    ///
    /// * `x` - X coordinates (must be sorted)
    /// * `real_y` - Real parts of Y coordinates
    /// * `imag_y` - Imaginary parts of Y coordinates
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_calculus::interpolators::CCSpline;
    ///
    /// let x = vec![0.0, 1.0, 2.0];
    /// let real_y = vec![0.0, 1.0, 0.0];
    /// let imag_y = vec![0.0, 1.0, 2.0];
    /// let spline = CCSpline::new(x, real_y, imag_y);
    /// ```
    pub fn new(x: Vec<f64>, real_y: Vec<f64>, imag_y: Vec<f64>) -> Self {
        assert_eq!(x.len(), real_y.len());
        assert_eq!(x.len(), imag_y.len());

        CCSpline {
            real_spline: CubicSpline::new(x.clone(), real_y),
            imag_spline: CubicSpline::new(x, imag_y),
        }
    }

    /// Evaluate the complex spline at a point
    ///
    /// # Arguments
    ///
    /// * `x` - Point at which to evaluate
    ///
    /// # Returns
    ///
    /// Tuple (real_part, imaginary_part)
    pub fn eval(&self, x: f64) -> (f64, f64) {
        (self.real_spline.eval(x), self.imag_spline.eval(x))
    }

    /// Evaluate the derivative at a point
    ///
    /// # Arguments
    ///
    /// * `x` - Point at which to evaluate
    ///
    /// # Returns
    ///
    /// Tuple (real_derivative, imaginary_derivative)
    pub fn derivative(&self, x: f64) -> (f64, f64) {
        (self.real_spline.derivative(x), self.imag_spline.derivative(x))
    }
}

/// Polygon spline - piecewise linear interpolation
///
/// This is a lightweight structure for piecewise linear (polygon) interpolation,
/// also known as linear spline.
#[derive(Debug, Clone)]
pub struct PSpline {
    /// X coordinates
    x: Vec<f64>,
    /// Y coordinates
    y: Vec<f64>,
}

impl PSpline {
    /// Create a new polygon spline
    ///
    /// # Arguments
    ///
    /// * `x` - X coordinates (must be sorted)
    /// * `y` - Y coordinates
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_calculus::interpolators::PSpline;
    ///
    /// let x = vec![0.0, 1.0, 2.0, 3.0];
    /// let y = vec![0.0, 1.0, 4.0, 9.0];
    /// let spline = PSpline::new(x, y);
    /// let value = spline.eval(1.5);
    /// ```
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Self {
        assert_eq!(x.len(), y.len());
        assert!(x.len() >= 2, "Need at least 2 points");

        PSpline { x, y }
    }

    /// Evaluate the polygon spline at a point
    ///
    /// # Arguments
    ///
    /// * `x` - Point at which to evaluate
    ///
    /// # Returns
    ///
    /// Interpolated value
    pub fn eval(&self, x: f64) -> f64 {
        use crate::interpolation::piecewise_linear;
        piecewise_linear(&self.x, &self.y, x)
    }

    /// Get the derivative (slope) at a point
    ///
    /// Since this is piecewise linear, the derivative is constant within
    /// each interval.
    ///
    /// # Arguments
    ///
    /// * `x` - Point at which to evaluate
    ///
    /// # Returns
    ///
    /// The slope of the line segment containing x
    pub fn derivative(&self, x: f64) -> f64 {
        let n = self.x.len();

        // Find interval
        for i in 0..n - 1 {
            if x >= self.x[i] && x <= self.x[i + 1] {
                return (self.y[i + 1] - self.y[i]) / (self.x[i + 1] - self.x[i]);
            }
        }

        // Outside range, return 0
        0.0
    }
}

/// Create a complex cubic spline
///
/// Convenience function for creating a CCSpline.
///
/// # Examples
///
/// ```
/// use rustmath_calculus::interpolators::complex_cubic_spline;
///
/// let x = vec![0.0, 1.0, 2.0];
/// let real_y = vec![0.0, 1.0, 0.0];
/// let imag_y = vec![0.0, 1.0, 2.0];
/// let spline = complex_cubic_spline(&x, &real_y, &imag_y);
/// let (real, imag) = spline.eval(0.5);
/// ```
pub fn complex_cubic_spline(x: &[f64], real_y: &[f64], imag_y: &[f64]) -> CCSpline {
    CCSpline::new(x.to_vec(), real_y.to_vec(), imag_y.to_vec())
}

/// Create a polygon spline
///
/// Convenience function for creating a PSpline.
///
/// # Examples
///
/// ```
/// use rustmath_calculus::interpolators::polygon_spline;
///
/// let x = vec![0.0, 1.0, 2.0];
/// let y = vec![0.0, 2.0, 4.0];
/// let spline = polygon_spline(&x, &y);
/// let value = spline.eval(0.5);
/// assert_eq!(value, 1.0);
/// ```
pub fn polygon_spline(x: &[f64], y: &[f64]) -> PSpline {
    PSpline::new(x.to_vec(), y.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cc_spline() {
        let x = vec![0.0, 1.0, 2.0];
        let real_y = vec![0.0, 1.0, 0.0];
        let imag_y = vec![0.0, 1.0, 2.0];

        let spline = CCSpline::new(x, real_y, imag_y);

        // Test interpolation at data points
        let (r0, i0) = spline.eval(0.0);
        assert!((r0 - 0.0).abs() < 1e-6);
        assert!((i0 - 0.0).abs() < 1e-6);

        let (r1, i1) = spline.eval(1.0);
        assert!((r1 - 1.0).abs() < 1e-6);
        assert!((i1 - 1.0).abs() < 1e-6);

        let (r2, i2) = spline.eval(2.0);
        assert!((r2 - 0.0).abs() < 1e-6);
        assert!((i2 - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_p_spline() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 2.0, 4.0];

        let spline = PSpline::new(x, y);

        // Test interpolation
        assert_eq!(spline.eval(0.0), 0.0);
        assert_eq!(spline.eval(0.5), 1.0);
        assert_eq!(spline.eval(1.0), 2.0);
        assert_eq!(spline.eval(1.5), 3.0);
        assert_eq!(spline.eval(2.0), 4.0);
    }

    #[test]
    fn test_p_spline_derivative() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 2.0, 4.0];

        let spline = PSpline::new(x, y);

        // Slope should be 2.0 everywhere
        assert_eq!(spline.derivative(0.5), 2.0);
        assert_eq!(spline.derivative(1.5), 2.0);
    }

    #[test]
    fn test_complex_cubic_spline_func() {
        let x = vec![0.0, 1.0, 2.0];
        let real_y = vec![1.0, 0.0, -1.0];
        let imag_y = vec![0.0, 1.0, 0.0];

        let spline = complex_cubic_spline(&x, &real_y, &imag_y);
        let (r, i) = spline.eval(1.0);

        assert!((r - 0.0).abs() < 1e-6);
        assert!((i - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_polygon_spline_func() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 2.0, 4.0];

        let spline = polygon_spline(&x, &y);
        assert_eq!(spline.eval(0.5), 1.0);
    }

    #[test]
    fn test_cc_spline_derivative() {
        let x = vec![0.0, 1.0, 2.0];
        let real_y = vec![0.0, 1.0, 0.0];
        let imag_y = vec![0.0, 0.0, 0.0];

        let spline = CCSpline::new(x, real_y, imag_y);
        let (dr, di) = spline.derivative(1.0);

        // Real part derivative should exist
        assert!(dr.is_finite());
        // Imaginary part derivative should be close to 0
        assert!(di.abs() < 1e-5);
    }
}
