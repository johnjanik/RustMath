//! Points on manifolds

use crate::errors::{ManifoldError, Result};
use std::fmt;

/// A point on a manifold
///
/// Points are the fundamental objects that live on manifolds. Each point
/// can have coordinates in different charts.
#[derive(Debug, Clone, PartialEq)]
pub struct ManifoldPoint {
    /// Coordinates of the point (in some chart)
    coordinates: Vec<f64>,
    /// Name of the point (optional)
    name: Option<String>,
}

impl ManifoldPoint {
    /// Create a new point with the given coordinates
    ///
    /// # Arguments
    ///
    /// * `coordinates` - The coordinates of the point in some chart
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_manifolds::ManifoldPoint;
    ///
    /// let point = ManifoldPoint::new(vec![1.0, 2.0, 3.0]);
    /// assert_eq!(point.dimension(), 3);
    /// ```
    pub fn new(coordinates: Vec<f64>) -> Self {
        Self {
            coordinates,
            name: None,
        }
    }

    /// Create a point from coordinates (alias for new)
    pub fn from_coordinates(coordinates: Vec<f64>) -> Self {
        Self::new(coordinates)
    }

    /// Create a named point
    pub fn named(name: impl Into<String>, coordinates: Vec<f64>) -> Self {
        Self {
            coordinates,
            name: Some(name.into()),
        }
    }

    /// Get the dimension of the point (number of coordinates)
    pub fn dimension(&self) -> usize {
        self.coordinates.len()
    }

    /// Get the coordinates of the point
    pub fn coordinates(&self) -> &[f64] {
        &self.coordinates
    }

    /// Get a specific coordinate
    pub fn coordinate(&self, index: usize) -> Result<f64> {
        self.coordinates.get(index).copied().ok_or_else(|| {
            ManifoldError::InvalidCoordinate(format!(
                "Index {} out of bounds for point with dimension {}",
                index,
                self.dimension()
            ))
        })
    }

    /// Get the name of the point
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Set the name of the point
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = Some(name.into());
    }

    /// Compute the Euclidean distance to another point
    pub fn distance_to(&self, other: &ManifoldPoint) -> Result<f64> {
        if self.dimension() != other.dimension() {
            return Err(ManifoldError::DimensionMismatch {
                expected: self.dimension(),
                actual: other.dimension(),
            });
        }

        let sum_squares: f64 = self.coordinates.iter()
            .zip(other.coordinates.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        Ok(sum_squares.sqrt())
    }
}

impl fmt::Display for ManifoldPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(name) = &self.name {
            write!(f, "{}: ", name)?;
        }
        write!(f, "(")?;
        for (i, coord) in self.coordinates.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", coord)?;
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_creation() {
        let point = ManifoldPoint::new(vec![1.0, 2.0]);
        assert_eq!(point.dimension(), 2);
        assert_eq!(point.coordinates(), &[1.0, 2.0]);
        assert!(point.name().is_none());
    }

    #[test]
    fn test_named_point() {
        let point = ManifoldPoint::named("P", vec![3.0, 4.0]);
        assert_eq!(point.name(), Some("P"));
        assert_eq!(point.coordinates(), &[3.0, 4.0]);
    }

    #[test]
    fn test_coordinate_access() {
        let point = ManifoldPoint::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(point.coordinate(0).unwrap(), 1.0);
        assert_eq!(point.coordinate(1).unwrap(), 2.0);
        assert_eq!(point.coordinate(2).unwrap(), 3.0);
        assert!(point.coordinate(3).is_err());
    }

    #[test]
    fn test_distance() {
        let p1 = ManifoldPoint::new(vec![0.0, 0.0]);
        let p2 = ManifoldPoint::new(vec![3.0, 4.0]);
        let distance = p1.distance_to(&p2).unwrap();
        assert!((distance - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance_dimension_mismatch() {
        let p1 = ManifoldPoint::new(vec![0.0, 0.0]);
        let p2 = ManifoldPoint::new(vec![1.0, 2.0, 3.0]);
        assert!(p1.distance_to(&p2).is_err());
    }

    #[test]
    fn test_point_display() {
        let point = ManifoldPoint::new(vec![1.0, 2.0]);
        assert_eq!(format!("{}", point), "(1, 2)");

        let named_point = ManifoldPoint::named("P", vec![3.0, 4.0]);
        assert_eq!(format!("{}", named_point), "P: (3, 4)");
    }
}
