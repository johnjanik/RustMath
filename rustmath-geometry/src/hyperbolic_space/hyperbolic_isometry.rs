//! Isometries of hyperbolic space
//!
//! Isometries are distance-preserving transformations. In the Upper Half Plane
//! model, they are represented by Möbius transformations of the form:
//! z ↦ (az + b)/(cz + d) where ad - bc = 1
//!
//! # Examples
//!
//! ```
//! use rustmath_geometry::hyperbolic_space::hyperbolic_isometry::HyperbolicIsometryUHP;
//!
//! // Identity transformation
//! let iso = HyperbolicIsometryUHP::identity();
//! ```

use super::hyperbolic_point::HyperbolicPointUHP;
use std::fmt;

/// Base trait for hyperbolic isometries
pub trait HyperbolicIsometry {
    /// Apply the isometry to a point (as coordinates)
    fn apply(&self, coords: &[f64]) -> Vec<f64>;

    /// Check if this is the identity isometry
    fn is_identity(&self) -> bool;
}

/// An isometry of the Upper Half Plane
///
/// Represented by a Möbius transformation z ↦ (az + b)/(cz + d)
/// with real coefficients and ad - bc = 1.
#[derive(Clone, Debug)]
pub struct HyperbolicIsometryUHP {
    /// Matrix entries for the Möbius transformation
    a: f64,
    b: f64,
    c: f64,
    d: f64,
}

impl HyperbolicIsometryUHP {
    /// Create a new isometry from matrix coefficients
    ///
    /// The transformation is z ↦ (az + b)/(cz + d)
    ///
    /// # Arguments
    ///
    /// * `a`, `b`, `c`, `d` - Matrix coefficients (must satisfy ad - bc = 1)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::hyperbolic_space::hyperbolic_isometry::HyperbolicIsometryUHP;
    ///
    /// // Translation by 1: z ↦ z + 1
    /// let iso = HyperbolicIsometryUHP::new(1.0, 1.0, 0.0, 1.0);
    /// ```
    pub fn new(a: f64, b: f64, c: f64, d: f64) -> Self {
        // Check that ad - bc = 1 (within tolerance)
        let det = a * d - b * c;
        assert!((det - 1.0).abs() < 1e-6, "Determinant must be 1, got {}", det);

        Self { a, b, c, d }
    }

    /// Create the identity isometry
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::hyperbolic_space::hyperbolic_isometry::HyperbolicIsometryUHP;
    ///
    /// let id = HyperbolicIsometryUHP::identity();
    /// assert!(id.is_identity());
    /// ```
    pub fn identity() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
        }
    }

    /// Create a translation isometry
    ///
    /// Translates by a real number: z ↦ z + t
    pub fn translation(t: f64) -> Self {
        Self {
            a: 1.0,
            b: t,
            c: 0.0,
            d: 1.0,
        }
    }

    /// Create a dilation isometry
    ///
    /// Dilates by a positive factor: z ↦ λz where λ > 0
    pub fn dilation(lambda: f64) -> Self {
        assert!(lambda > 0.0, "Dilation factor must be positive");
        let sqrt_lambda = lambda.sqrt();
        Self {
            a: sqrt_lambda,
            b: 0.0,
            c: 0.0,
            d: 1.0 / sqrt_lambda,
        }
    }

    /// Apply the isometry to a point
    pub fn apply_to_point(&self, point: &HyperbolicPointUHP) -> HyperbolicPointUHP {
        let z_real = point.real();
        let z_imag = point.imag();

        // Compute (az + b)/(cz + d) where z = z_real + i*z_imag
        // Numerator: (a + ic)(z_real + i*z_imag) + (b + id)
        //          = (a*z_real - c*z_imag + b) + i(c*z_real + a*z_imag + d)
        // Wait, c and d are real in UHP model, so:
        // Numerator: a*z + b = a*(z_real + i*z_imag) + b
        //          = (a*z_real + b) + i*(a*z_imag)
        // Denominator: c*z + d = c*(z_real + i*z_imag) + d
        //            = (c*z_real + d) + i*(c*z_imag)

        let num_real = self.a * z_real + self.b;
        let num_imag = self.a * z_imag;
        let den_real = self.c * z_real + self.d;
        let den_imag = self.c * z_imag;

        // (num_real + i*num_imag) / (den_real + i*den_imag)
        // = (num_real + i*num_imag)(den_real - i*den_imag) / (den_real² + den_imag²)
        let den_sq = den_real * den_real + den_imag * den_imag;

        let result_real = (num_real * den_real + num_imag * den_imag) / den_sq;
        let result_imag = (num_imag * den_real - num_real * den_imag) / den_sq;

        HyperbolicPointUHP::new(result_real, result_imag)
    }

    /// Compose with another isometry
    ///
    /// Returns self ∘ other (apply other first, then self)
    pub fn compose(&self, other: &HyperbolicIsometryUHP) -> HyperbolicIsometryUHP {
        // Matrix multiplication
        let a = self.a * other.a + self.b * other.c;
        let b = self.a * other.b + self.b * other.d;
        let c = self.c * other.a + self.d * other.c;
        let d = self.c * other.b + self.d * other.d;

        HyperbolicIsometryUHP { a, b, c, d }
    }
}

impl HyperbolicIsometry for HyperbolicIsometryUHP {
    fn apply(&self, coords: &[f64]) -> Vec<f64> {
        assert_eq!(coords.len(), 2, "Expected 2 coordinates");
        let point = HyperbolicPointUHP::new(coords[0], coords[1]);
        let result = self.apply_to_point(&point);
        vec![result.real(), result.imag()]
    }

    fn is_identity(&self) -> bool {
        (self.a - 1.0).abs() < 1e-10
            && self.b.abs() < 1e-10
            && self.c.abs() < 1e-10
            && (self.d - 1.0).abs() < 1e-10
    }
}

impl fmt::Display for HyperbolicIsometryUHP {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "IsometryUHP(z ↦ ({}z + {})/({}z + {}))",
            self.a, self.b, self.c, self.d
        )
    }
}

/// Placeholder for Poincaré Disk isometry
#[derive(Clone, Debug)]
pub struct HyperbolicIsometryPD {
    _marker: (),
}

impl HyperbolicIsometryPD {
    /// Create a trivial isometry
    pub fn new() -> Self {
        Self { _marker: () }
    }
}

impl Default for HyperbolicIsometryPD {
    fn default() -> Self {
        Self::new()
    }
}

impl HyperbolicIsometry for HyperbolicIsometryPD {
    fn apply(&self, coords: &[f64]) -> Vec<f64> {
        coords.to_vec()
    }

    fn is_identity(&self) -> bool {
        true
    }
}

/// Placeholder for Klein model isometry
#[derive(Clone, Debug)]
pub struct HyperbolicIsometryKM {
    _marker: (),
}

impl HyperbolicIsometryKM {
    /// Create a trivial isometry
    pub fn new() -> Self {
        Self { _marker: () }
    }
}

impl Default for HyperbolicIsometryKM {
    fn default() -> Self {
        Self::new()
    }
}

impl HyperbolicIsometry for HyperbolicIsometryKM {
    fn apply(&self, coords: &[f64]) -> Vec<f64> {
        coords.to_vec()
    }

    fn is_identity(&self) -> bool {
        true
    }
}

/// Create a Möbius transformation
///
/// This is a convenience function for creating isometries.
///
/// # Arguments
///
/// * `a`, `b`, `c`, `d` - Coefficients of the transformation z ↦ (az + b)/(cz + d)
pub fn moebius_transform(a: f64, b: f64, c: f64, d: f64) -> HyperbolicIsometryUHP {
    HyperbolicIsometryUHP::new(a, b, c, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let iso = HyperbolicIsometryUHP::identity();
        assert!(iso.is_identity());

        let p = HyperbolicPointUHP::new(1.0, 2.0);
        let result = iso.apply_to_point(&p);

        assert!((result.real() - 1.0).abs() < 1e-10);
        assert!((result.imag() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_translation() {
        let iso = HyperbolicIsometryUHP::translation(3.0);
        let p = HyperbolicPointUHP::new(1.0, 2.0);
        let result = iso.apply_to_point(&p);

        // Translation: z ↦ z + 3, so (1 + 2i) ↦ (4 + 2i)
        assert!((result.real() - 4.0).abs() < 1e-10);
        assert!((result.imag() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_dilation() {
        let iso = HyperbolicIsometryUHP::dilation(4.0);
        let p = HyperbolicPointUHP::new(1.0, 1.0);
        let result = iso.apply_to_point(&p);

        // Dilation should preserve the upper half plane
        assert!(result.imag() > 0.0);
        assert!(result.real().is_finite() && result.imag().is_finite());
    }

    #[test]
    fn test_compose() {
        let t = HyperbolicIsometryUHP::translation(1.0);
        let d = HyperbolicIsometryUHP::dilation(4.0);

        // Compose: first translate, then dilate
        let composed = d.compose(&t);

        let p = HyperbolicPointUHP::new(0.0, 1.0);
        let result = composed.apply_to_point(&p);

        // Result should be in the upper half plane
        assert!(result.imag() > 0.0);
        assert!(result.real().is_finite() && result.imag().is_finite());
    }

    #[test]
    fn test_moebius_transform() {
        let iso = moebius_transform(1.0, 1.0, 0.0, 1.0);
        assert!(!iso.is_identity());

        let p = HyperbolicPointUHP::new(0.0, 1.0);
        let result = iso.apply_to_point(&p);

        // z ↦ z + 1, so i ↦ 1 + i
        assert!((result.real() - 1.0).abs() < 1e-10);
        assert!((result.imag() - 1.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic]
    fn test_invalid_determinant() {
        // This should panic because ad - bc ≠ 1
        HyperbolicIsometryUHP::new(1.0, 0.0, 0.0, 2.0);
    }
}
