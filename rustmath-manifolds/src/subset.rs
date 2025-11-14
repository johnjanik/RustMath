//! Manifold subsets - the base structure for all manifold objects

use crate::errors::Result;
use std::fmt;

/// A subset of a manifold
///
/// This is the base structure for all manifold objects. It represents
/// a subset of a topological or differentiable manifold.
#[derive(Debug, Clone)]
pub struct ManifoldSubset {
    /// Name of the subset
    name: String,
    /// Dimension of the ambient manifold
    ambient_dimension: usize,
    /// Optional description
    description: Option<String>,
}

impl ManifoldSubset {
    /// Create a new manifold subset
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the subset
    /// * `ambient_dimension` - Dimension of the ambient manifold
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_manifolds::ManifoldSubset;
    ///
    /// let subset = ManifoldSubset::new("U", 2);
    /// ```
    pub fn new(name: impl Into<String>, ambient_dimension: usize) -> Self {
        Self {
            name: name.into(),
            ambient_dimension,
            description: None,
        }
    }

    /// Create a new manifold subset with a description
    pub fn with_description(
        name: impl Into<String>,
        ambient_dimension: usize,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            ambient_dimension,
            description: Some(description.into()),
        }
    }

    /// Get the name of the subset
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the dimension of the ambient manifold
    pub fn ambient_dimension(&self) -> usize {
        self.ambient_dimension
    }

    /// Get the description if available
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// Check if this subset is contained in another subset
    pub fn is_subset_of(&self, _other: &ManifoldSubset) -> Result<bool> {
        // This is a simplified implementation
        // In a full implementation, this would check actual subset relationships
        Ok(false)
    }
}

impl fmt::Display for ManifoldSubset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)?;
        if let Some(desc) = &self.description {
            write!(f, " ({})", desc)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subset_creation() {
        let subset = ManifoldSubset::new("U", 3);
        assert_eq!(subset.name(), "U");
        assert_eq!(subset.ambient_dimension(), 3);
        assert!(subset.description().is_none());
    }

    #[test]
    fn test_subset_with_description() {
        let subset = ManifoldSubset::with_description(
            "U",
            2,
            "Open subset of R^2"
        );
        assert_eq!(subset.name(), "U");
        assert_eq!(subset.ambient_dimension(), 2);
        assert_eq!(subset.description(), Some("Open subset of R^2"));
    }

    #[test]
    fn test_subset_display() {
        let subset = ManifoldSubset::new("V", 2);
        assert_eq!(format!("{}", subset), "V");

        let subset_with_desc = ManifoldSubset::with_description(
            "U",
            3,
            "unit ball"
        );
        assert_eq!(format!("{}", subset_with_desc), "U (unit ball)");
    }
}
