//! Topological manifolds

use crate::chart::Chart;
use crate::errors::{ManifoldError, Result};
use crate::point::ManifoldPoint;
use crate::subset::ManifoldSubset;
use std::fmt;

/// A topological manifold
///
/// A topological manifold is a topological space that locally resembles
/// Euclidean space. This structure provides the foundation for differentiable
/// manifolds and other geometric objects.
#[derive(Clone)]
pub struct TopologicalManifold {
    /// Name of the manifold
    name: String,
    /// Dimension of the manifold
    dimension: usize,
    /// The underlying subset structure
    subset: ManifoldSubset,
    /// Atlas of charts covering the manifold
    charts: Vec<Chart>,
    /// Optional description
    description: Option<String>,
}

impl TopologicalManifold {
    /// Create a new topological manifold
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the manifold
    /// * `dimension` - Dimension of the manifold
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_manifolds::TopologicalManifold;
    ///
    /// let manifold = TopologicalManifold::new("M", 2);
    /// assert_eq!(manifold.dimension(), 2);
    /// ```
    pub fn new(name: impl Into<String>, dimension: usize) -> Self {
        let name_str = name.into();
        let subset = ManifoldSubset::new(&name_str, dimension);

        Self {
            name: name_str,
            dimension,
            subset,
            charts: Vec::new(),
            description: None,
        }
    }

    /// Create a new topological manifold with a description
    pub fn with_description(
        name: impl Into<String>,
        dimension: usize,
        description: impl Into<String>,
    ) -> Self {
        let mut manifold = Self::new(name, dimension);
        manifold.description = Some(description.into());
        manifold
    }

    /// Get the name of the manifold
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the dimension of the manifold
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the underlying subset
    pub fn subset(&self) -> &ManifoldSubset {
        &self.subset
    }

    /// Get the description
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// Add a chart to the atlas
    pub fn add_chart(&mut self, chart: Chart) -> Result<()> {
        if chart.dimension() != self.dimension {
            return Err(ManifoldError::DimensionMismatch {
                expected: self.dimension,
                actual: chart.dimension(),
            });
        }

        self.charts.push(chart);
        Ok(())
    }

    /// Get all charts in the atlas
    pub fn charts(&self) -> &[Chart] {
        &self.charts
    }

    /// Get the default chart (first chart in atlas)
    pub fn default_chart(&self) -> Option<&Chart> {
        self.charts.first()
    }

    /// Get a chart by name
    pub fn chart(&self, name: &str) -> Result<&Chart> {
        self.charts
            .iter()
            .find(|c| c.name() == name)
            .ok_or_else(|| ManifoldError::ChartNotFound(name.to_string()))
    }

    /// Get the number of charts in the atlas
    pub fn num_charts(&self) -> usize {
        self.charts.len()
    }

    /// Create a point on the manifold with given coordinates
    pub fn point(&self, coordinates: Vec<f64>) -> Result<ManifoldPoint> {
        if coordinates.len() != self.dimension {
            return Err(ManifoldError::DimensionMismatch {
                expected: self.dimension,
                actual: coordinates.len(),
            });
        }

        Ok(ManifoldPoint::new(coordinates))
    }

    /// Create a named point on the manifold
    pub fn named_point(
        &self,
        name: impl Into<String>,
        coordinates: Vec<f64>,
    ) -> Result<ManifoldPoint> {
        if coordinates.len() != self.dimension {
            return Err(ManifoldError::DimensionMismatch {
                expected: self.dimension,
                actual: coordinates.len(),
            });
        }

        Ok(ManifoldPoint::named(name, coordinates))
    }
}

impl fmt::Debug for TopologicalManifold {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TopologicalManifold")
            .field("name", &self.name)
            .field("dimension", &self.dimension)
            .field("num_charts", &self.charts.len())
            .finish()
    }
}

impl fmt::Display for TopologicalManifold {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-dimensional manifold '{}'", self.dimension, self.name)?;
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
    fn test_manifold_creation() {
        let manifold = TopologicalManifold::new("M", 3);
        assert_eq!(manifold.name(), "M");
        assert_eq!(manifold.dimension(), 3);
        assert_eq!(manifold.num_charts(), 0);
    }

    #[test]
    fn test_manifold_with_description() {
        let manifold = TopologicalManifold::with_description(
            "S2",
            2,
            "2-sphere embedded in R^3"
        );
        assert_eq!(manifold.description(), Some("2-sphere embedded in R^3"));
    }

    #[test]
    fn test_add_chart() {
        let mut manifold = TopologicalManifold::new("M", 2);
        let chart = Chart::new("cart", 2, vec!["x", "y"]).unwrap();

        manifold.add_chart(chart).unwrap();
        assert_eq!(manifold.num_charts(), 1);
    }

    #[test]
    fn test_add_chart_dimension_mismatch() {
        let mut manifold = TopologicalManifold::new("M", 2);
        let chart = Chart::new("cart", 3, vec!["x", "y", "z"]).unwrap();

        assert!(manifold.add_chart(chart).is_err());
    }

    #[test]
    fn test_get_chart_by_name() {
        let mut manifold = TopologicalManifold::new("M", 2);
        let chart = Chart::new("cartesian", 2, vec!["x", "y"]).unwrap();

        manifold.add_chart(chart).unwrap();

        let retrieved = manifold.chart("cartesian").unwrap();
        assert_eq!(retrieved.name(), "cartesian");
    }

    #[test]
    fn test_get_nonexistent_chart() {
        let manifold = TopologicalManifold::new("M", 2);
        assert!(manifold.chart("nonexistent").is_err());
    }

    #[test]
    fn test_create_point() {
        let manifold = TopologicalManifold::new("M", 3);
        let point = manifold.point(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(point.dimension(), 3);
    }

    #[test]
    fn test_create_point_dimension_mismatch() {
        let manifold = TopologicalManifold::new("M", 2);
        assert!(manifold.point(vec![1.0, 2.0, 3.0]).is_err());
    }

    #[test]
    fn test_create_named_point() {
        let manifold = TopologicalManifold::new("M", 2);
        let point = manifold.named_point("P", vec![1.0, 2.0]).unwrap();
        assert_eq!(point.name(), Some("P"));
    }

    #[test]
    fn test_manifold_display() {
        let manifold = TopologicalManifold::new("M", 2);
        assert_eq!(format!("{}", manifold), "2-dimensional manifold 'M'");

        let manifold_with_desc = TopologicalManifold::with_description(
            "S2",
            2,
            "unit sphere"
        );
        assert_eq!(format!("{}", manifold_with_desc), "2-dimensional manifold 'S2': unit sphere");
    }
}
