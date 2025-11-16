//! Points in hyperbolic space
//!
//! This module provides representations of points in hyperbolic space using
//! various models. The primary model implemented is the Upper Half Plane (UHP).
//!
//! # Examples
//!
//! ```
//! use rustmath_geometry::hyperbolic_space::hyperbolic_point::HyperbolicPointUHP;
//!
//! // Create a point in the upper half plane
//! let point = HyperbolicPointUHP::new(1.0, 2.0);
//! assert!(point.is_valid());
//! ```

use std::fmt;

/// A point in hyperbolic space (base trait)
pub trait HyperbolicPoint: Clone {
    /// Check if the point is valid for its model
    fn is_valid(&self) -> bool;

    /// Get the coordinates as a vector
    fn coordinates(&self) -> Vec<f64>;

    /// Compute the hyperbolic distance to another point
    fn distance_to(&self, other: &Self) -> f64;
}

/// A point in the Upper Half Plane model
///
/// Represents a point z = x + iy where y > 0.
/// This is one of the standard models for 2-dimensional hyperbolic geometry.
#[derive(Clone, Debug, PartialEq)]
pub struct HyperbolicPointUHP {
    /// Real part (x-coordinate)
    x: f64,
    /// Imaginary part (y-coordinate), must be > 0
    y: f64,
}

impl HyperbolicPointUHP {
    /// Create a new point in the Upper Half Plane
    ///
    /// # Arguments
    ///
    /// * `x` - The real part
    /// * `y` - The imaginary part (must be positive)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::hyperbolic_space::hyperbolic_point::HyperbolicPointUHP;
    ///
    /// let p = HyperbolicPointUHP::new(1.0, 2.0);
    /// assert!(p.is_valid());
    /// ```
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Get the real part (x-coordinate)
    pub fn real(&self) -> f64 {
        self.x
    }

    /// Get the imaginary part (y-coordinate)
    pub fn imag(&self) -> f64 {
        self.y
    }

    /// Get the coordinates as a complex number pair
    pub fn as_complex(&self) -> (f64, f64) {
        (self.x, self.y)
    }
}

impl HyperbolicPoint for HyperbolicPointUHP {
    fn is_valid(&self) -> bool {
        self.y > 0.0 && self.y.is_finite() && self.x.is_finite()
    }

    fn coordinates(&self) -> Vec<f64> {
        vec![self.x, self.y]
    }

    /// Compute hyperbolic distance in the UHP model
    ///
    /// The formula is: d(z1, z2) = arcosh(1 + |z1 - z2|²/(2*y1*y2))
    fn distance_to(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let numerator = dx * dx + dy * dy;
        let denominator = 2.0 * self.y * other.y;

        let ratio = 1.0 + numerator / denominator;
        ratio.acosh()
    }
}

impl fmt::Display for HyperbolicPointUHP {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HyperbolicPoint({} + {}i)", self.x, self.y)
    }
}

/// A point in the Poincaré Disk model
///
/// Represents points inside the unit disk |z| < 1.
#[derive(Clone, Debug, PartialEq)]
pub struct HyperbolicPointPD {
    /// x-coordinate
    x: f64,
    /// y-coordinate
    y: f64,
}

impl HyperbolicPointPD {
    /// Create a new point in the Poincaré Disk
    ///
    /// # Arguments
    ///
    /// * `x` - The x-coordinate
    /// * `y` - The y-coordinate
    ///
    /// The point must satisfy x² + y² < 1.
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Get the x-coordinate
    pub fn x(&self) -> f64 {
        self.x
    }

    /// Get the y-coordinate
    pub fn y(&self) -> f64 {
        self.y
    }

    /// Get the radius from the origin
    pub fn radius(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}

impl HyperbolicPoint for HyperbolicPointPD {
    fn is_valid(&self) -> bool {
        let r_squared = self.x * self.x + self.y * self.y;
        r_squared < 1.0 && self.x.is_finite() && self.y.is_finite()
    }

    fn coordinates(&self) -> Vec<f64> {
        vec![self.x, self.y]
    }

    /// Compute hyperbolic distance in the Poincaré Disk model
    ///
    /// The formula is: d(z1, z2) = arcosh(1 + 2|z1 - z2|²/((1-|z1|²)(1-|z2|²)))
    fn distance_to(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dist_squared = dx * dx + dy * dy;

        let r1_squared = self.x * self.x + self.y * self.y;
        let r2_squared = other.x * other.x + other.y * other.y;

        let numerator = 2.0 * dist_squared;
        let denominator = (1.0 - r1_squared) * (1.0 - r2_squared);

        let ratio = 1.0 + numerator / denominator;
        ratio.acosh()
    }
}

impl fmt::Display for HyperbolicPointPD {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HyperbolicPointPD({}, {})", self.x, self.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uhp_creation() {
        let p = HyperbolicPointUHP::new(1.0, 2.0);
        assert_eq!(p.real(), 1.0);
        assert_eq!(p.imag(), 2.0);
        assert!(p.is_valid());
    }

    #[test]
    fn test_uhp_invalid() {
        let p = HyperbolicPointUHP::new(1.0, -1.0); // negative y is invalid
        assert!(!p.is_valid());

        let p2 = HyperbolicPointUHP::new(1.0, 0.0); // y = 0 is on boundary
        assert!(!p2.is_valid());
    }

    #[test]
    fn test_uhp_coordinates() {
        let p = HyperbolicPointUHP::new(3.0, 4.0);
        assert_eq!(p.coordinates(), vec![3.0, 4.0]);
    }

    #[test]
    fn test_uhp_distance() {
        let p1 = HyperbolicPointUHP::new(0.0, 1.0);
        let p2 = HyperbolicPointUHP::new(0.0, 2.0);

        let dist = p1.distance_to(&p2);
        // Distance should be ln(2) ≈ 0.693
        assert!((dist - 0.693).abs() < 0.01);
    }

    #[test]
    fn test_uhp_distance_to_self() {
        let p = HyperbolicPointUHP::new(1.0, 1.0);
        let dist = p.distance_to(&p);
        assert!(dist.abs() < 1e-10); // Should be approximately 0
    }

    #[test]
    fn test_pd_creation() {
        let p = HyperbolicPointPD::new(0.5, 0.3);
        assert_eq!(p.x(), 0.5);
        assert_eq!(p.y(), 0.3);
        assert!(p.is_valid());
    }

    #[test]
    fn test_pd_invalid() {
        let p = HyperbolicPointPD::new(1.5, 0.0); // outside unit disk
        assert!(!p.is_valid());

        let p2 = HyperbolicPointPD::new(0.8, 0.8); // x²+y² > 1
        assert!(!p2.is_valid());
    }

    #[test]
    fn test_pd_radius() {
        let p = HyperbolicPointPD::new(0.6, 0.8);
        assert!((p.radius() - 1.0).abs() < 1e-10); // Should be exactly 1
    }

    #[test]
    fn test_pd_distance() {
        let p1 = HyperbolicPointPD::new(0.0, 0.0); // center
        let p2 = HyperbolicPointPD::new(0.5, 0.0);

        let dist = p1.distance_to(&p2);
        // Distance should be positive and finite
        assert!(dist > 0.0 && dist.is_finite());
    }
}
