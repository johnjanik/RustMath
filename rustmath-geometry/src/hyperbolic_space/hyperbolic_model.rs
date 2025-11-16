//! Models of hyperbolic space
//!
//! This module provides different coordinate systems (models) for representing
//! hyperbolic geometry. The main models are:
//! - UHP: Upper Half Plane
//! - PD: Poincaré Disk
//! - KM: Klein Model
//! - HM: Hyperboloid Model
//!
//! # Examples
//!
//! ```
//! use rustmath_geometry::hyperbolic_space::hyperbolic_model::{HyperbolicModelUHP, HyperbolicModel};
//!
//! let model = HyperbolicModelUHP::new();
//! assert_eq!(model.dimension(), 2);
//! ```

use std::fmt;

/// Trait for hyperbolic space models
pub trait HyperbolicModel {
    /// Get the dimension of the model
    fn dimension(&self) -> usize;

    /// Get the name of the model
    fn name(&self) -> &str;

    /// Check if a point (given as coordinates) is valid in this model
    fn is_valid_point(&self, coords: &[f64]) -> bool;
}

/// Upper Half Plane model
///
/// Points are represented as z = x + iy with y > 0.
/// This is the classical Poincaré upper half-plane model.
#[derive(Clone, Debug)]
pub struct HyperbolicModelUHP {
    dimension: usize,
}

impl HyperbolicModelUHP {
    /// Create a new Upper Half Plane model
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::hyperbolic_space::hyperbolic_model::HyperbolicModelUHP;
    ///
    /// let model = HyperbolicModelUHP::new();
    /// ```
    pub fn new() -> Self {
        Self { dimension: 2 }
    }
}

impl Default for HyperbolicModelUHP {
    fn default() -> Self {
        Self::new()
    }
}

impl HyperbolicModel for HyperbolicModelUHP {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        "UHP"
    }

    fn is_valid_point(&self, coords: &[f64]) -> bool {
        coords.len() == 2 && coords[1] > 0.0 && coords[0].is_finite() && coords[1].is_finite()
    }
}

impl fmt::Display for HyperbolicModelUHP {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hyperbolic space in Upper Half Plane Model")
    }
}

/// Poincaré Disk model
///
/// Points are represented as (x, y) with x² + y² < 1.
#[derive(Clone, Debug)]
pub struct HyperbolicModelPD {
    dimension: usize,
}

impl HyperbolicModelPD {
    /// Create a new Poincaré Disk model
    pub fn new() -> Self {
        Self { dimension: 2 }
    }
}

impl Default for HyperbolicModelPD {
    fn default() -> Self {
        Self::new()
    }
}

impl HyperbolicModel for HyperbolicModelPD {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        "PD"
    }

    fn is_valid_point(&self, coords: &[f64]) -> bool {
        if coords.len() != 2 {
            return false;
        }
        let r_squared = coords[0] * coords[0] + coords[1] * coords[1];
        r_squared < 1.0 && coords[0].is_finite() && coords[1].is_finite()
    }
}

impl fmt::Display for HyperbolicModelPD {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hyperbolic space in Poincaré Disk Model")
    }
}

/// Klein model (also known as the Beltrami-Klein model)
///
/// Points are represented as (x, y) with x² + y² < 1.
/// Similar to Poincaré disk but with different metric.
#[derive(Clone, Debug)]
pub struct HyperbolicModelKM {
    dimension: usize,
}

impl HyperbolicModelKM {
    /// Create a new Klein model
    pub fn new() -> Self {
        Self { dimension: 2 }
    }
}

impl Default for HyperbolicModelKM {
    fn default() -> Self {
        Self::new()
    }
}

impl HyperbolicModel for HyperbolicModelKM {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        "KM"
    }

    fn is_valid_point(&self, coords: &[f64]) -> bool {
        if coords.len() != 2 {
            return false;
        }
        let r_squared = coords[0] * coords[0] + coords[1] * coords[1];
        r_squared < 1.0 && coords[0].is_finite() && coords[1].is_finite()
    }
}

impl fmt::Display for HyperbolicModelKM {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hyperbolic space in Klein Model")
    }
}

/// Hyperboloid model (also known as the Minkowski model)
///
/// Points are on the upper sheet of the hyperboloid x² + y² - z² = -1 with z > 0.
#[derive(Clone, Debug)]
pub struct HyperbolicModelHM {
    dimension: usize,
}

impl HyperbolicModelHM {
    /// Create a new Hyperboloid model
    pub fn new() -> Self {
        Self { dimension: 2 }
    }
}

impl Default for HyperbolicModelHM {
    fn default() -> Self {
        Self::new()
    }
}

impl HyperbolicModel for HyperbolicModelHM {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        "HM"
    }

    fn is_valid_point(&self, coords: &[f64]) -> bool {
        if coords.len() != 3 {
            return false;
        }
        let x = coords[0];
        let y = coords[1];
        let z = coords[2];

        // Check hyperboloid equation: x² + y² - z² = -1 and z > 0
        let eq = x * x + y * y - z * z;
        (eq + 1.0).abs() < 1e-6 && z > 0.0 && x.is_finite() && y.is_finite() && z.is_finite()
    }
}

impl fmt::Display for HyperbolicModelHM {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hyperbolic space in Hyperboloid Model")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uhp_model() {
        let model = HyperbolicModelUHP::new();
        assert_eq!(model.dimension(), 2);
        assert_eq!(model.name(), "UHP");

        assert!(model.is_valid_point(&[1.0, 2.0])); // valid
        assert!(!model.is_valid_point(&[1.0, -1.0])); // y < 0
        assert!(!model.is_valid_point(&[1.0, 0.0])); // y = 0
    }

    #[test]
    fn test_pd_model() {
        let model = HyperbolicModelPD::new();
        assert_eq!(model.dimension(), 2);
        assert_eq!(model.name(), "PD");

        assert!(model.is_valid_point(&[0.5, 0.5])); // inside unit disk
        assert!(!model.is_valid_point(&[1.0, 1.0])); // outside unit disk
        assert!(!model.is_valid_point(&[1.0, 0.0])); // on boundary
    }

    #[test]
    fn test_km_model() {
        let model = HyperbolicModelKM::new();
        assert_eq!(model.dimension(), 2);
        assert_eq!(model.name(), "KM");

        assert!(model.is_valid_point(&[0.0, 0.0])); // center
        assert!(model.is_valid_point(&[0.5, 0.5])); // inside
        assert!(!model.is_valid_point(&[1.0, 1.0])); // outside
    }

    #[test]
    fn test_hm_model() {
        let model = HyperbolicModelHM::new();
        assert_eq!(model.dimension(), 2);
        assert_eq!(model.name(), "HM");

        // Point (0, 0, 1) is on the hyperboloid: 0² + 0² - 1² = -1
        assert!(model.is_valid_point(&[0.0, 0.0, 1.0]));

        // Point (0, 1, √2) is on the hyperboloid: 0² + 1² - 2 = -1
        assert!(model.is_valid_point(&[0.0, 1.0, std::f64::consts::SQRT_2]));

        // Invalid: not on hyperboloid
        assert!(!model.is_valid_point(&[1.0, 1.0, 1.0]));
    }

    #[test]
    fn test_model_display() {
        let uhp = HyperbolicModelUHP::new();
        let display = format!("{}", uhp);
        assert!(display.contains("Upper Half Plane"));

        let pd = HyperbolicModelPD::new();
        let display = format!("{}", pd);
        assert!(display.contains("Poincaré Disk"));
    }
}
