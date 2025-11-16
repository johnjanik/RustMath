//! Interface for hyperbolic space
//!
//! This module provides the main interface for creating and working with
//! hyperbolic space in various models.
//!
//! # Examples
//!
//! ```
//! use rustmath_geometry::hyperbolic_space::hyperbolic_interface::{HyperbolicSpace, ModelType};
//!
//! let space = HyperbolicSpace::new(ModelType::UHP);
//! assert_eq!(space.dimension(), 2);
//! ```

use super::hyperbolic_model::{
    HyperbolicModel, HyperbolicModelUHP, HyperbolicModelPD,
    HyperbolicModelKM, HyperbolicModelHM,
};
use std::fmt;

/// Type of hyperbolic model
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelType {
    /// Upper Half Plane
    UHP,
    /// Poincaré Disk
    PD,
    /// Klein Model
    KM,
    /// Hyperboloid Model
    HM,
}

/// The hyperbolic plane (2-dimensional hyperbolic space)
///
/// This is a convenience type for 2D hyperbolic geometry.
#[derive(Clone, Debug)]
pub struct HyperbolicPlane {
    model_type: ModelType,
}

impl HyperbolicPlane {
    /// Create a new hyperbolic plane with the specified model
    ///
    /// # Arguments
    ///
    /// * `model_type` - The model to use (UHP, PD, KM, or HM)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::hyperbolic_space::hyperbolic_interface::{HyperbolicPlane, ModelType};
    ///
    /// let plane = HyperbolicPlane::new(ModelType::UHP);
    /// ```
    pub fn new(model_type: ModelType) -> Self {
        Self { model_type }
    }

    /// Get the dimension (always 2 for the plane)
    pub fn dimension(&self) -> usize {
        2
    }

    /// Get the model type
    pub fn model_type(&self) -> ModelType {
        self.model_type
    }
}

impl Default for HyperbolicPlane {
    fn default() -> Self {
        Self::new(ModelType::UHP)
    }
}

impl fmt::Display for HyperbolicPlane {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hyperbolic Plane ({:?} model)", self.model_type)
    }
}

/// Hyperbolic space in arbitrary dimension
///
/// This is a general structure for hyperbolic geometry.
#[derive(Clone, Debug)]
pub struct HyperbolicSpace {
    dimension: usize,
    model_type: ModelType,
}

impl HyperbolicSpace {
    /// Create a new hyperbolic space
    ///
    /// # Arguments
    ///
    /// * `model_type` - The model to use
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::hyperbolic_space::hyperbolic_interface::{HyperbolicSpace, ModelType};
    ///
    /// let space = HyperbolicSpace::new(ModelType::UHP);
    /// ```
    pub fn new(model_type: ModelType) -> Self {
        Self {
            dimension: 2,
            model_type,
        }
    }

    /// Create a hyperbolic space with specified dimension
    ///
    /// Note: Currently only dimension 2 is fully supported.
    pub fn with_dimension(dimension: usize, model_type: ModelType) -> Self {
        assert!(dimension >= 2, "Dimension must be at least 2");
        Self {
            dimension,
            model_type,
        }
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the model type
    pub fn model_type(&self) -> ModelType {
        self.model_type
    }

    /// Create the underlying model object
    pub fn create_model(&self) -> Box<dyn HyperbolicModel> {
        match self.model_type {
            ModelType::UHP => Box::new(HyperbolicModelUHP::new()),
            ModelType::PD => Box::new(HyperbolicModelPD::new()),
            ModelType::KM => Box::new(HyperbolicModelKM::new()),
            ModelType::HM => Box::new(HyperbolicModelHM::new()),
        }
    }
}

impl fmt::Display for HyperbolicSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Hyperbolic Space (dim={}, model={:?})",
            self.dimension, self.model_type
        )
    }
}

/// Collection of different hyperbolic models
///
/// This provides a way to work with multiple models simultaneously.
#[derive(Clone, Debug)]
pub struct HyperbolicModels {
    models: Vec<ModelType>,
}

impl HyperbolicModels {
    /// Create a collection with all available models
    pub fn all() -> Self {
        Self {
            models: vec![ModelType::UHP, ModelType::PD, ModelType::KM, ModelType::HM],
        }
    }

    /// Create a collection with specific models
    pub fn from_types(models: Vec<ModelType>) -> Self {
        Self { models }
    }

    /// Get the number of models
    pub fn count(&self) -> usize {
        self.models.len()
    }

    /// Get an iterator over the model types
    pub fn iter(&self) -> impl Iterator<Item = &ModelType> {
        self.models.iter()
    }
}

/// Parent methods for hyperbolic space
///
/// This provides utility methods for working with hyperbolic geometry.
pub struct ParentMethods;

impl ParentMethods {
    /// Create a point in the specified model
    ///
    /// Returns coordinates if valid, None otherwise.
    pub fn create_point(model: ModelType, coords: Vec<f64>) -> Option<Vec<f64>> {
        let space = HyperbolicSpace::new(model);
        let model_obj = space.create_model();

        if model_obj.is_valid_point(&coords) {
            Some(coords)
        } else {
            None
        }
    }

    /// Check if coordinates are valid for a model
    pub fn is_valid(model: ModelType, coords: &[f64]) -> bool {
        let space = HyperbolicSpace::new(model);
        let model_obj = space.create_model();
        model_obj.is_valid_point(coords)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperbolic_plane() {
        let plane = HyperbolicPlane::new(ModelType::UHP);
        assert_eq!(plane.dimension(), 2);
        assert_eq!(plane.model_type(), ModelType::UHP);
    }

    #[test]
    fn test_hyperbolic_plane_default() {
        let plane = HyperbolicPlane::default();
        assert_eq!(plane.model_type(), ModelType::UHP);
    }

    #[test]
    fn test_hyperbolic_space() {
        let space = HyperbolicSpace::new(ModelType::PD);
        assert_eq!(space.dimension(), 2);
        assert_eq!(space.model_type(), ModelType::PD);
    }

    #[test]
    fn test_hyperbolic_space_with_dimension() {
        let space = HyperbolicSpace::with_dimension(3, ModelType::UHP);
        assert_eq!(space.dimension(), 3);
    }

    #[test]
    fn test_create_model() {
        let space = HyperbolicSpace::new(ModelType::UHP);
        let model = space.create_model();
        assert_eq!(model.name(), "UHP");
        assert_eq!(model.dimension(), 2);
    }

    #[test]
    fn test_hyperbolic_models_all() {
        let models = HyperbolicModels::all();
        assert_eq!(models.count(), 4);
    }

    #[test]
    fn test_hyperbolic_models_specific() {
        let models = HyperbolicModels::from_types(vec![ModelType::UHP, ModelType::PD]);
        assert_eq!(models.count(), 2);
    }

    #[test]
    fn test_parent_methods_create_point() {
        let point = ParentMethods::create_point(ModelType::UHP, vec![1.0, 2.0]);
        assert!(point.is_some());

        let invalid = ParentMethods::create_point(ModelType::UHP, vec![1.0, -1.0]);
        assert!(invalid.is_none());
    }

    #[test]
    fn test_parent_methods_is_valid() {
        assert!(ParentMethods::is_valid(ModelType::UHP, &[1.0, 2.0]));
        assert!(!ParentMethods::is_valid(ModelType::UHP, &[1.0, -1.0]));

        assert!(ParentMethods::is_valid(ModelType::PD, &[0.5, 0.5]));
        assert!(!ParentMethods::is_valid(ModelType::PD, &[2.0, 0.0]));
    }

    #[test]
    fn test_display() {
        let plane = HyperbolicPlane::new(ModelType::UHP);
        let display = format!("{}", plane);
        assert!(display.contains("Hyperbolic Plane"));
        assert!(display.contains("UHP"));

        let space = HyperbolicSpace::new(ModelType::PD);
        let display = format!("{}", space);
        assert!(display.contains("Hyperbolic Space"));
    }
}
