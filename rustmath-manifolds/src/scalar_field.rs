//! Scalar fields on manifolds

use crate::errors::Result;
use crate::point::ManifoldPoint;
use std::fmt;

/// A scalar field on a manifold
///
/// A scalar field is a smooth function that assigns a real number to each
/// point on a manifold.
pub struct ScalarField {
    /// Name of the scalar field
    name: String,
    /// Dimension of the manifold
    manifold_dimension: usize,
    /// Optional description
    description: Option<String>,
}

impl ScalarField {
    /// Create a new scalar field
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the scalar field
    /// * `manifold_dimension` - Dimension of the underlying manifold
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_manifolds::ScalarField;
    ///
    /// let field = ScalarField::new("f", 3);
    /// ```
    pub fn new(name: impl Into<String>, manifold_dimension: usize) -> Self {
        Self {
            name: name.into(),
            manifold_dimension,
            description: None,
        }
    }

    /// Create a scalar field with a description
    pub fn with_description(
        name: impl Into<String>,
        manifold_dimension: usize,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            manifold_dimension,
            description: Some(description.into()),
        }
    }

    /// Get the name of the scalar field
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the dimension of the underlying manifold
    pub fn manifold_dimension(&self) -> usize {
        self.manifold_dimension
    }

    /// Get the description
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// Evaluate the scalar field at a point
    ///
    /// This is a placeholder that would be implemented with actual
    /// function evaluation in a complete system.
    pub fn evaluate(&self, _point: &ManifoldPoint) -> Result<f64> {
        // In a full implementation, this would evaluate the function
        Ok(0.0)
    }
}

impl fmt::Debug for ScalarField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScalarField")
            .field("name", &self.name)
            .field("manifold_dimension", &self.manifold_dimension)
            .finish()
    }
}

impl fmt::Display for ScalarField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ScalarField({})", self.name)?;
        if let Some(desc) = &self.description {
            write!(f, ": {}", desc)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_field_creation() {
        let field = ScalarField::new("f", 2);
        assert_eq!(field.name(), "f");
        assert_eq!(field.manifold_dimension(), 2);
        assert!(field.description().is_none());
    }

    #[test]
    fn test_scalar_field_with_description() {
        let field = ScalarField::with_description(
            "temperature",
            3,
            "Temperature distribution"
        );
        assert_eq!(field.name(), "temperature");
        assert_eq!(field.description(), Some("Temperature distribution"));
    }

    #[test]
    fn test_scalar_field_evaluate() {
        let field = ScalarField::new("f", 2);
        let point = ManifoldPoint::new(vec![1.0, 2.0]);
        // Placeholder evaluation
        let value = field.evaluate(&point).unwrap();
        assert!(value.is_finite());
    }

    #[test]
    fn test_scalar_field_display() {
        let field = ScalarField::new("f", 2);
        assert_eq!(format!("{}", field), "ScalarField(f)");

        let field_with_desc = ScalarField::with_description(
            "temp",
            3,
            "temperature"
        );
        assert_eq!(format!("{}", field_with_desc), "ScalarField(temp): temperature");
    }
}
