//! Geodesics in hyperbolic space
//!
//! Geodesics are the shortest paths between two points in hyperbolic space.
//! In different models, they have different representations:
//! - UHP: Vertical lines or semicircles orthogonal to the real axis
//! - PD: Diameters or circular arcs orthogonal to the boundary
//!
//! # Examples
//!
//! ```
//! use rustmath_geometry::hyperbolic_space::hyperbolic_geodesic::HyperbolicGeodesicUHP;
//! use rustmath_geometry::hyperbolic_space::hyperbolic_point::HyperbolicPointUHP;
//!
//! let p1 = HyperbolicPointUHP::new(0.0, 1.0);
//! let p2 = HyperbolicPointUHP::new(0.0, 2.0);
//! let geodesic = HyperbolicGeodesicUHP::from_endpoints(&p1, &p2);
//! ```

use super::hyperbolic_point::{HyperbolicPoint, HyperbolicPointUHP, HyperbolicPointPD};
use std::fmt;

/// Base trait for hyperbolic geodesics
pub trait HyperbolicGeodesic {
    /// Get the length of the geodesic
    fn length(&self) -> f64;

    /// Check if a point lies on the geodesic
    fn contains_point(&self, coords: &[f64]) -> bool;
}

/// A geodesic in the Upper Half Plane model
///
/// Geodesics in UHP are either:
/// - Vertical lines (when endpoints have same x-coordinate)
/// - Semicircles orthogonal to the x-axis
#[derive(Clone, Debug)]
pub struct HyperbolicGeodesicUHP {
    /// Starting point
    start: HyperbolicPointUHP,
    /// Ending point
    end: HyperbolicPointUHP,
    /// Type of geodesic: true for vertical line, false for semicircle
    is_vertical: bool,
    /// For semicircles: center x-coordinate
    center_x: Option<f64>,
    /// For semicircles: radius
    radius: Option<f64>,
}

impl HyperbolicGeodesicUHP {
    /// Create a geodesic from two endpoints
    ///
    /// # Arguments
    ///
    /// * `start` - Starting point
    /// * `end` - Ending point
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::hyperbolic_space::hyperbolic_geodesic::HyperbolicGeodesicUHP;
    /// use rustmath_geometry::hyperbolic_space::hyperbolic_point::HyperbolicPointUHP;
    ///
    /// let p1 = HyperbolicPointUHP::new(0.0, 1.0);
    /// let p2 = HyperbolicPointUHP::new(0.0, 2.0);
    /// let geo = HyperbolicGeodesicUHP::from_endpoints(&p1, &p2);
    /// ```
    pub fn from_endpoints(start: &HyperbolicPointUHP, end: &HyperbolicPointUHP) -> Self {
        let is_vertical = (start.real() - end.real()).abs() < 1e-10;

        let (center_x, radius) = if !is_vertical {
            // For a semicircle through (x1, y1) and (x2, y2),
            // the center is on the x-axis at ((x1² + y1² - x2² - y2²) / (2(x1 - x2)), 0)
            let x1 = start.real();
            let y1 = start.imag();
            let x2 = end.real();
            let y2 = end.imag();

            let cx = (x1 * x1 + y1 * y1 - x2 * x2 - y2 * y2) / (2.0 * (x1 - x2));
            let r = ((x1 - cx) * (x1 - cx) + y1 * y1).sqrt();

            (Some(cx), Some(r))
        } else {
            (None, None)
        };

        Self {
            start: start.clone(),
            end: end.clone(),
            is_vertical,
            center_x,
            radius,
        }
    }

    /// Get the starting point
    pub fn start(&self) -> &HyperbolicPointUHP {
        &self.start
    }

    /// Get the ending point
    pub fn end(&self) -> &HyperbolicPointUHP {
        &self.end
    }

    /// Check if this is a vertical geodesic
    pub fn is_vertical(&self) -> bool {
        self.is_vertical
    }

    /// Get the center and radius (for semicircular geodesics)
    pub fn semicircle_params(&self) -> Option<(f64, f64)> {
        if let (Some(cx), Some(r)) = (self.center_x, self.radius) {
            Some((cx, r))
        } else {
            None
        }
    }
}

impl HyperbolicGeodesic for HyperbolicGeodesicUHP {
    fn length(&self) -> f64 {
        self.start.distance_to(&self.end)
    }

    fn contains_point(&self, coords: &[f64]) -> bool {
        if coords.len() != 2 {
            return false;
        }

        let x = coords[0];
        let y = coords[1];

        if self.is_vertical {
            // For vertical line, x must match start.x
            (x - self.start.real()).abs() < 1e-10
        } else if let (Some(cx), Some(r)) = (self.center_x, self.radius) {
            // For semicircle, check distance from center
            let dist = ((x - cx) * (x - cx) + y * y).sqrt();
            (dist - r).abs() < 1e-6 && y > 0.0
        } else {
            false
        }
    }
}

impl fmt::Display for HyperbolicGeodesicUHP {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_vertical {
            write!(
                f,
                "Geodesic(vertical line x = {})",
                self.start.real()
            )
        } else if let (Some(cx), Some(r)) = (self.center_x, self.radius) {
            write!(
                f,
                "Geodesic(semicircle center=({}, 0), radius={})",
                cx, r
            )
        } else {
            write!(f, "Geodesic")
        }
    }
}

/// A geodesic in the Poincaré Disk model
///
/// Geodesics in PD are either:
/// - Diameters (straight lines through origin)
/// - Circular arcs orthogonal to the unit circle
#[derive(Clone, Debug)]
pub struct HyperbolicGeodesicPD {
    /// Starting point
    start: HyperbolicPointPD,
    /// Ending point
    end: HyperbolicPointPD,
}

impl HyperbolicGeodesicPD {
    /// Create a geodesic from two endpoints
    pub fn from_endpoints(start: &HyperbolicPointPD, end: &HyperbolicPointPD) -> Self {
        Self {
            start: start.clone(),
            end: end.clone(),
        }
    }

    /// Get the starting point
    pub fn start(&self) -> &HyperbolicPointPD {
        &self.start
    }

    /// Get the ending point
    pub fn end(&self) -> &HyperbolicPointPD {
        &self.end
    }
}

impl HyperbolicGeodesic for HyperbolicGeodesicPD {
    fn length(&self) -> f64 {
        self.start.distance_to(&self.end)
    }

    fn contains_point(&self, _coords: &[f64]) -> bool {
        // Simplified implementation
        false
    }
}

impl fmt::Display for HyperbolicGeodesicPD {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GeodesicPD from {} to {}", self.start, self.end)
    }
}

/// Placeholder for Klein model geodesic
#[derive(Clone, Debug)]
pub struct HyperbolicGeodesicKM {
    length: f64,
}

impl HyperbolicGeodesicKM {
    /// Create a trivial geodesic
    pub fn new() -> Self {
        Self { length: 0.0 }
    }
}

impl Default for HyperbolicGeodesicKM {
    fn default() -> Self {
        Self::new()
    }
}

impl HyperbolicGeodesic for HyperbolicGeodesicKM {
    fn length(&self) -> f64 {
        self.length
    }

    fn contains_point(&self, _coords: &[f64]) -> bool {
        false
    }
}

/// Placeholder for Hyperboloid model geodesic
#[derive(Clone, Debug)]
pub struct HyperbolicGeodesicHM {
    length: f64,
}

impl HyperbolicGeodesicHM {
    /// Create a trivial geodesic
    pub fn new() -> Self {
        Self { length: 0.0 }
    }
}

impl Default for HyperbolicGeodesicHM {
    fn default() -> Self {
        Self::new()
    }
}

impl HyperbolicGeodesic for HyperbolicGeodesicHM {
    fn length(&self) -> f64 {
        self.length
    }

    fn contains_point(&self, _coords: &[f64]) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertical_geodesic() {
        let p1 = HyperbolicPointUHP::new(1.0, 1.0);
        let p2 = HyperbolicPointUHP::new(1.0, 2.0);
        let geo = HyperbolicGeodesicUHP::from_endpoints(&p1, &p2);

        assert!(geo.is_vertical());
        assert!(geo.contains_point(&[1.0, 1.5]));
        assert!(!geo.contains_point(&[0.0, 1.5]));
    }

    #[test]
    fn test_semicircular_geodesic() {
        let p1 = HyperbolicPointUHP::new(0.0, 1.0);
        let p2 = HyperbolicPointUHP::new(2.0, 1.0);
        let geo = HyperbolicGeodesicUHP::from_endpoints(&p1, &p2);

        assert!(!geo.is_vertical());

        if let Some((cx, r)) = geo.semicircle_params() {
            // Center should be at (1, 0) with radius √2
            assert!((cx - 1.0).abs() < 1e-10);
            assert!((r - std::f64::consts::SQRT_2).abs() < 1e-10);
        } else {
            panic!("Expected semicircle parameters");
        }
    }

    #[test]
    fn test_geodesic_length() {
        let p1 = HyperbolicPointUHP::new(0.0, 1.0);
        let p2 = HyperbolicPointUHP::new(0.0, 2.0);
        let geo = HyperbolicGeodesicUHP::from_endpoints(&p1, &p2);

        let length = geo.length();
        // Should be ln(2) ≈ 0.693
        assert!((length - 0.693).abs() < 0.01);
    }

    #[test]
    fn test_pd_geodesic() {
        let p1 = HyperbolicPointPD::new(0.0, 0.0);
        let p2 = HyperbolicPointPD::new(0.5, 0.0);
        let geo = HyperbolicGeodesicPD::from_endpoints(&p1, &p2);

        let length = geo.length();
        assert!(length > 0.0);
    }
}
