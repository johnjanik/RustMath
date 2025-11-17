//! Differentiable (smooth) manifolds

use crate::chart::Chart;
use crate::errors::Result;
use crate::manifold::TopologicalManifold;
use crate::scalar_field::ScalarFieldEnhanced as ScalarField;
use std::fmt;
use std::sync::Arc;

/// A differentiable (smooth) manifold
///
/// A differentiable manifold is a topological manifold equipped with a
/// differentiable structure, allowing calculus to be performed on it.
#[derive(Clone)]
pub struct DifferentiableManifold {
    /// The underlying topological manifold
    topological: TopologicalManifold,
    /// Differentiability class (e.g., 1 for C^1, infinity for C^∞)
    /// None represents C^∞ (smooth)
    differentiability_class: Option<usize>,
}

impl DifferentiableManifold {
    /// Create a new smooth (C^∞) differentiable manifold
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the manifold
    /// * `dimension` - Dimension of the manifold
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_manifolds::DifferentiableManifold;
    ///
    /// let manifold = DifferentiableManifold::new("M", 2);
    /// assert_eq!(manifold.dimension(), 2);
    /// assert!(manifold.is_smooth());
    /// ```
    pub fn new(name: impl Into<String>, dimension: usize) -> Self {
        Self {
            topological: TopologicalManifold::new(name, dimension),
            differentiability_class: None, // C^∞
        }
    }

    /// Create a new differentiable manifold with specified differentiability class
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the manifold
    /// * `dimension` - Dimension of the manifold
    /// * `diff_class` - Differentiability class (1 for C^1, 2 for C^2, etc.)
    pub fn with_class(
        name: impl Into<String>,
        dimension: usize,
        diff_class: usize,
    ) -> Self {
        Self {
            topological: TopologicalManifold::new(name, dimension),
            differentiability_class: Some(diff_class),
        }
    }

    /// Create from an existing topological manifold
    pub fn from_topological(topological: TopologicalManifold) -> Self {
        Self {
            topological,
            differentiability_class: None,
        }
    }

    /// Get the name of the manifold
    pub fn name(&self) -> &str {
        self.topological.name()
    }

    /// Get the dimension of the manifold
    pub fn dimension(&self) -> usize {
        self.topological.dimension()
    }

    /// Get the differentiability class
    /// Returns None for C^∞ (smooth) manifolds
    pub fn differentiability_class(&self) -> Option<usize> {
        self.differentiability_class
    }

    /// Check if the manifold is smooth (C^∞)
    pub fn is_smooth(&self) -> bool {
        self.differentiability_class.is_none()
    }

    /// Get the underlying topological manifold
    pub fn topological(&self) -> &TopologicalManifold {
        &self.topological
    }

    /// Get mutable reference to the underlying topological manifold
    pub fn topological_mut(&mut self) -> &mut TopologicalManifold {
        &mut self.topological
    }

    /// Add a chart to the atlas
    pub fn add_chart(&mut self, chart: Chart) -> Result<()> {
        self.topological.add_chart(chart)
    }

    /// Get all charts in the atlas
    pub fn charts(&self) -> &[Chart] {
        self.topological.charts()
    }

    /// Get the default chart (first chart in atlas)
    pub fn default_chart(&self) -> Option<&Chart> {
        self.topological.default_chart()
    }

    /// Create a scalar field on this manifold
    pub fn scalar_field(&self, name: impl Into<String>) -> ScalarField {
        ScalarField::named(Arc::new(self.clone()), name)
    }

    /// Create a scalar field with a description
    pub fn scalar_field_with_description(
        &self,
        name: impl Into<String>,
        _description: impl Into<String>,
    ) -> ScalarField {
        // Note: ScalarFieldEnhanced doesn't currently support description field
        // Creating a named field as a workaround
        ScalarField::named(Arc::new(self.clone()), name)
    }
}

impl fmt::Debug for DifferentiableManifold {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let class_str = match self.differentiability_class {
            None => "C^∞".to_string(),
            Some(k) => format!("C^{}", k),
        };

        f.debug_struct("DifferentiableManifold")
            .field("name", &self.topological.name())
            .field("dimension", &self.dimension())
            .field("class", &class_str)
            .finish()
    }
}

impl fmt::Display for DifferentiableManifold {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let class_str = match self.differentiability_class {
            None => "smooth",
            Some(k) => return write!(
                f,
                "C^{} {}-dimensional manifold '{}'",
                k,
                self.dimension(),
                self.name()
            ),
        };

        write!(
            f,
            "{} {}-dimensional manifold '{}'",
            class_str,
            self.dimension(),
            self.name()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_differentiable_manifold_creation() {
        let manifold = DifferentiableManifold::new("M", 3);
        assert_eq!(manifold.name(), "M");
        assert_eq!(manifold.dimension(), 3);
        assert!(manifold.is_smooth());
        assert_eq!(manifold.differentiability_class(), None);
    }

    #[test]
    fn test_differentiable_manifold_with_class() {
        let manifold = DifferentiableManifold::with_class("M", 2, 2);
        assert_eq!(manifold.dimension(), 2);
        assert!(!manifold.is_smooth());
        assert_eq!(manifold.differentiability_class(), Some(2));
    }

    #[test]
    fn test_from_topological() {
        let topo = TopologicalManifold::new("M", 2);
        let diff = DifferentiableManifold::from_topological(topo);
        assert_eq!(diff.dimension(), 2);
        assert!(diff.is_smooth());
    }

    #[test]
    fn test_add_chart() {
        let mut manifold = DifferentiableManifold::new("M", 2);
        let chart = Chart::new("cart", 2, vec!["x", "y"]).unwrap();

        manifold.add_chart(chart).unwrap();
        assert_eq!(manifold.charts().len(), 1);
    }

    #[test]
    fn test_scalar_field_creation() {
        let manifold = DifferentiableManifold::new("M", 3);
        let field = manifold.scalar_field("f");

        assert_eq!(field.name(), "f");
        assert_eq!(field.manifold_dimension(), 3);
    }

    #[test]
    fn test_scalar_field_with_description() {
        let manifold = DifferentiableManifold::new("M", 2);
        let field = manifold.scalar_field_with_description(
            "temperature",
            "Temperature field"
        );

        assert_eq!(field.name(), "temperature");
        assert_eq!(field.description(), Some("Temperature field"));
    }

    #[test]
    fn test_differentiable_manifold_display() {
        let smooth = DifferentiableManifold::new("M", 2);
        assert_eq!(format!("{}", smooth), "smooth 2-dimensional manifold 'M'");

        let c2 = DifferentiableManifold::with_class("N", 3, 2);
        assert_eq!(format!("{}", c2), "C^2 3-dimensional manifold 'N'");
    }
}
