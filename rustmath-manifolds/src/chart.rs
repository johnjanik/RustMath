//! Charts - coordinate systems on manifolds

use crate::errors::{ManifoldError, Result};
use crate::point::ManifoldPoint;
use std::fmt;

/// A coordinate function that maps points to coordinates
pub type CoordinateFunction = fn(&ManifoldPoint) -> Result<Vec<f64>>;

/// A chart (coordinate system) on a manifold
///
/// A chart provides a local coordinate system for a manifold by mapping
/// points in a subset of the manifold to coordinates in Euclidean space.
#[derive(Clone)]
pub struct Chart {
    /// Name of the chart
    name: String,
    /// Dimension of the chart (number of coordinates)
    dimension: usize,
    /// Names of the coordinates (e.g., ["x", "y", "z"])
    coordinate_names: Vec<String>,
    /// Optional description
    description: Option<String>,
}

impl Chart {
    /// Create a new chart
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the chart
    /// * `dimension` - Dimension of the chart
    /// * `coordinate_names` - Names for each coordinate
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_manifolds::Chart;
    ///
    /// let chart = Chart::new("cartesian", 2, vec!["x", "y"]);
    /// assert_eq!(chart.dimension(), 2);
    /// ```
    pub fn new(
        name: impl Into<String>,
        dimension: usize,
        coordinate_names: Vec<impl Into<String>>,
    ) -> Result<Self> {
        let coord_names: Vec<String> = coordinate_names.into_iter()
            .map(|s| s.into())
            .collect();

        if coord_names.len() != dimension {
            return Err(ManifoldError::InvalidChart(format!(
                "Number of coordinate names ({}) doesn't match dimension ({})",
                coord_names.len(),
                dimension
            )));
        }

        Ok(Self {
            name: name.into(),
            dimension,
            coordinate_names: coord_names,
            description: None,
        })
    }

    /// Create a standard chart with default coordinate names (x1, x2, ...)
    pub fn standard(name: impl Into<String>, dimension: usize) -> Self {
        let coordinate_names = (1..=dimension)
            .map(|i| format!("x{}", i))
            .collect();

        Self {
            name: name.into(),
            dimension,
            coordinate_names,
            description: None,
        }
    }

    /// Set the description for this chart
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Get the name of the chart
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the dimension of the chart
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the coordinate names
    pub fn coordinate_names(&self) -> &[String] {
        &self.coordinate_names
    }

    /// Get a specific coordinate name
    pub fn coordinate_name(&self, index: usize) -> Result<&str> {
        self.coordinate_names.get(index)
            .map(|s| s.as_str())
            .ok_or_else(|| ManifoldError::InvalidCoordinate(format!(
                "Coordinate index {} out of bounds for {}-dimensional chart",
                index, self.dimension
            )))
    }

    /// Get the description
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }
}

impl fmt::Debug for Chart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Chart")
            .field("name", &self.name)
            .field("dimension", &self.dimension)
            .field("coordinates", &self.coordinate_names)
            .finish()
    }
}

impl fmt::Display for Chart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (", self.name)?;
        for (i, name) in self.coordinate_names.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", name)?;
        }
        write!(f, ")")
    }
}

/// Coordinate transformation between two charts
pub struct CoordinateTransformation {
    /// Source chart
    source: Chart,
    /// Target chart
    target: Chart,
}

impl CoordinateTransformation {
    /// Create a new coordinate transformation
    pub fn new(source: Chart, target: Chart) -> Result<Self> {
        if source.dimension() != target.dimension() {
            return Err(ManifoldError::DimensionMismatch {
                expected: source.dimension(),
                actual: target.dimension(),
            });
        }

        Ok(Self { source, target })
    }

    /// Get the source chart
    pub fn source(&self) -> &Chart {
        &self.source
    }

    /// Get the target chart
    pub fn target(&self) -> &Chart {
        &self.target
    }

    /// Get the dimension of the transformation
    pub fn dimension(&self) -> usize {
        self.source.dimension()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chart_creation() {
        let chart = Chart::new("cartesian", 2, vec!["x", "y"]).unwrap();
        assert_eq!(chart.name(), "cartesian");
        assert_eq!(chart.dimension(), 2);
        assert_eq!(chart.coordinate_names(), &["x", "y"]);
    }

    #[test]
    fn test_standard_chart() {
        let chart = Chart::standard("std", 3);
        assert_eq!(chart.dimension(), 3);
        assert_eq!(chart.coordinate_names(), &["x1", "x2", "x3"]);
    }

    #[test]
    fn test_chart_with_description() {
        let chart = Chart::standard("polar", 2)
            .with_description("Polar coordinates on R^2");
        assert_eq!(chart.description(), Some("Polar coordinates on R^2"));
    }

    #[test]
    fn test_coordinate_name_access() {
        let chart = Chart::new("cart", 2, vec!["x", "y"]).unwrap();
        assert_eq!(chart.coordinate_name(0).unwrap(), "x");
        assert_eq!(chart.coordinate_name(1).unwrap(), "y");
        assert!(chart.coordinate_name(2).is_err());
    }

    #[test]
    fn test_chart_invalid_coordinate_count() {
        let result = Chart::new("bad", 2, vec!["x", "y", "z"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_coordinate_transformation() {
        let cart = Chart::new("cartesian", 2, vec!["x", "y"]).unwrap();
        let polar = Chart::new("polar", 2, vec!["r", "theta"]).unwrap();

        let transform = CoordinateTransformation::new(cart, polar).unwrap();
        assert_eq!(transform.dimension(), 2);
    }

    #[test]
    fn test_coordinate_transformation_dimension_mismatch() {
        let chart2d = Chart::new("2d", 2, vec!["x", "y"]).unwrap();
        let chart3d = Chart::new("3d", 3, vec!["x", "y", "z"]).unwrap();

        assert!(CoordinateTransformation::new(chart2d, chart3d).is_err());
    }

    #[test]
    fn test_chart_display() {
        let chart = Chart::new("cartesian", 2, vec!["x", "y"]).unwrap();
        assert_eq!(format!("{}", chart), "cartesian (x, y)");
    }
}
