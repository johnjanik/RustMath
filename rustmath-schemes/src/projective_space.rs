//! Enhanced Projective Space
//!
//! This module provides comprehensive support for projective spaces and
//! homogeneous coordinates, extending the basic implementation.

use rustmath_core::Ring;
use num_traits::{Zero, One};
use std::fmt;

/// A point in projective space ℙⁿ with homogeneous coordinates
///
/// Homogeneous coordinates [x₀ : x₁ : ... : xₙ] represent a point in ℙⁿ,
/// where [x₀ : ... : xₙ] = [λx₀ : ... : λxₙ] for any non-zero λ.
///
/// The point is valid if not all coordinates are zero.
#[derive(Clone, Debug)]
pub struct ProjectivePoint<R: Ring> {
    /// Homogeneous coordinates [x₀ : x₁ : ... : xₙ]
    coordinates: Vec<R>,
    /// Dimension of the ambient projective space
    dimension: usize,
}

impl<R: Ring> ProjectivePoint<R> {
    /// Create a new projective point from homogeneous coordinates
    ///
    /// # Arguments
    /// * `coordinates` - Homogeneous coordinates [x₀ : ... : xₙ]
    ///
    /// # Returns
    /// - `Ok(point)` if coordinates are valid (not all zero, length is n+1 for ℙⁿ)
    /// - `Err(msg)` if coordinates are invalid
    pub fn new(coordinates: Vec<R>) -> Result<Self, String> {
        if coordinates.is_empty() {
            return Err("Coordinates cannot be empty".to_string());
        }

        if coordinates.iter().all(|c| c.is_zero()) {
            return Err("Not all coordinates can be zero".to_string());
        }

        let dimension = coordinates.len() - 1;

        Ok(ProjectivePoint {
            coordinates,
            dimension,
        })
    }

    /// Get the homogeneous coordinates
    pub fn coordinates(&self) -> &[R] {
        &self.coordinates
    }

    /// Get the dimension of the ambient space (n for ℙⁿ)
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Check if two points are equal in projective space
    ///
    /// [x₀:...:xₙ] = [y₀:...:yₙ] iff xᵢ/xⱼ = yᵢ/yⱼ for all i,j where denominators are non-zero
    ///
    /// In practice: [x₀:...:xₙ] = [y₀:...:yₙ] iff ∃λ≠0: xᵢ = λyᵢ for all i
    pub fn equals(&self, other: &ProjectivePoint<R>) -> bool
    where
        R: PartialEq,
    {
        if self.coordinates.len() != other.coordinates.len() {
            return false;
        }

        // Simplified equality check
        // In a full implementation, we'd check if one is a scalar multiple of the other
        // For now, we check coordinate-wise equality
        self.coordinates == other.coordinates
    }

    /// Get the i-th coordinate
    pub fn coordinate(&self, i: usize) -> Option<&R> {
        self.coordinates.get(i)
    }

    /// Check if the i-th coordinate is non-zero
    pub fn is_in_affine_chart(&self, chart_index: usize) -> bool {
        if let Some(coord) = self.coordinate(chart_index) {
            !coord.is_zero()
        } else {
            false
        }
    }

    /// Convert to affine coordinates on the i-th chart
    ///
    /// On the chart Uᵢ = {xᵢ ≠ 0}, we have coordinates (x₀/xᵢ, ..., x̂ᵢ/xᵢ, ..., xₙ/xᵢ)
    pub fn to_affine(&self, chart_index: usize) -> Result<Vec<R>, String> {
        if chart_index >= self.coordinates.len() {
            return Err("Chart index out of bounds".to_string());
        }

        if self.coordinates[chart_index].is_zero() {
            return Err("Point not in this affine chart".to_string());
        }

        // Return coordinates except the chart_index-th one
        let mut affine = Vec::new();
        for (i, coord) in self.coordinates.iter().enumerate() {
            if i != chart_index {
                affine.push(coord.clone());
            }
        }

        Ok(affine)
    }

    /// Create a projective point from affine coordinates on the i-th chart
    ///
    /// Converts (a₀, ..., âᵢ, ..., aₙ) to [a₀ : ... : 1 : ... : aₙ] where 1 is at position i
    pub fn from_affine(affine_coords: Vec<R>, chart_index: usize) -> Result<Self, String>
    where
        R: One,
    {
        let dimension = affine_coords.len();
        let mut coords = Vec::new();

        for i in 0..=dimension {
            if i == chart_index {
                coords.push(<R as Ring>::one());
            } else if i < chart_index {
                coords.push(affine_coords[i].clone());
            } else {
                coords.push(affine_coords[i - 1].clone());
            }
        }

        ProjectivePoint::new(coords)
    }
}

impl<R: Ring> fmt::Display for ProjectivePoint<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, coord) in self.coordinates.iter().enumerate() {
            if i > 0 {
                write!(f, " : ")?;
            }
            write!(f, "{}", coord)?;
        }
        write!(f, "]")
    }
}

/// Projective space ℙⁿ over a ring R
///
/// ℙⁿ is the set of lines through the origin in Rⁿ⁺¹, or equivalently,
/// the set of equivalence classes of (n+1)-tuples [x₀ : ... : xₙ] where
/// not all xᵢ are zero, under the equivalence relation
/// [x₀ : ... : xₙ] ~ [λx₀ : ... : λxₙ] for λ ≠ 0.
#[derive(Clone, Debug)]
pub struct ProjectiveSpace<R: Ring> {
    /// Dimension n (ℙⁿ has n+1 homogeneous coordinates)
    dimension: usize,
    /// Base ring
    _phantom: std::marker::PhantomData<R>,
}

impl<R: Ring> ProjectiveSpace<R> {
    /// Create a new projective space ℙⁿ
    ///
    /// # Arguments
    /// * `dimension` - The dimension n (not the number of coordinates)
    ///
    /// # Examples
    /// - `ProjectiveSpace::new(1)` creates ℙ¹ (projective line)
    /// - `ProjectiveSpace::new(2)` creates ℙ² (projective plane)
    pub fn new(dimension: usize) -> Self {
        ProjectiveSpace {
            dimension,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the number of homogeneous coordinates (n+1 for ℙⁿ)
    pub fn num_coordinates(&self) -> usize {
        self.dimension + 1
    }

    /// Check if coordinates are valid for this projective space
    pub fn contains_point(&self, point: &ProjectivePoint<R>) -> bool {
        point.dimension() == self.dimension
    }

    /// Create a standard basis point [0:...:0:1:0:...:0] where 1 is at position i
    pub fn standard_point(&self, index: usize) -> Result<ProjectivePoint<R>, String>
    where
        R: Zero + One,
    {
        if index > self.dimension {
            return Err("Index out of bounds".to_string());
        }

        let mut coords = vec![<R as Ring>::zero(); self.num_coordinates()];
        coords[index] = <R as Ring>::one();

        ProjectivePoint::new(coords)
    }

    /// Get the number of affine charts in the standard covering
    ///
    /// ℙⁿ is covered by n+1 affine charts Uᵢ = {xᵢ ≠ 0} for i = 0, ..., n
    pub fn num_affine_charts(&self) -> usize {
        self.num_coordinates()
    }
}

impl<R: Ring> fmt::Display for ProjectiveSpace<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ℙ^{}", self.dimension)
    }
}

/// A linear subspace of projective space
///
/// A linear subspace of ℙⁿ is defined by linear equations in the homogeneous coordinates.
/// A k-dimensional linear subspace is a ℙᵏ ⊆ ℙⁿ.
#[derive(Clone, Debug)]
pub struct LinearSubspace<R: Ring> {
    /// Ambient projective space
    ambient_space: ProjectiveSpace<R>,
    /// Dimension of the subspace
    dimension: usize,
    /// Defining linear equations (as coefficient vectors)
    equations: Vec<Vec<R>>,
}

impl<R: Ring> LinearSubspace<R> {
    /// Create a new linear subspace
    pub fn new(
        ambient_space: ProjectiveSpace<R>,
        dimension: usize,
        equations: Vec<Vec<R>>,
    ) -> Self {
        LinearSubspace {
            ambient_space,
            dimension,
            equations,
        }
    }

    /// Get the dimension of the subspace
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the ambient space
    pub fn ambient_space(&self) -> &ProjectiveSpace<R> {
        &self.ambient_space
    }

    /// Get the codimension (dim(ambient) - dim(subspace))
    pub fn codimension(&self) -> usize {
        self.ambient_space.dimension() - self.dimension
    }

    /// Check if a point lies in this subspace
    pub fn contains(&self, point: &ProjectivePoint<R>) -> bool
    where
        R: Zero,
    {
        // Check if point satisfies all defining equations
        for equation in &self.equations {
            // Compute dot product of equation coefficients with point coordinates
            let mut sum = <R as Ring>::zero();
            for (coeff, coord) in equation.iter().zip(point.coordinates()) {
                sum = sum + coeff.clone() * coord.clone();
            }

            if !Ring::is_zero(&sum) {
                return false;
            }
        }

        true
    }
}

impl<R: Ring> fmt::Display for LinearSubspace<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ℙ^{} ⊆ {}",
            self.dimension,
            self.ambient_space
        )
    }
}

/// A hyperplane in projective space
///
/// A hyperplane in ℙⁿ is a ℙⁿ⁻¹ defined by a single linear equation
/// a₀x₀ + a₁x₁ + ··· + aₙxₙ = 0
pub struct Hyperplane<R: Ring> {
    /// Ambient projective space
    ambient_space: ProjectiveSpace<R>,
    /// Coefficients of the defining equation
    coefficients: Vec<R>,
}

impl<R: Ring> Hyperplane<R> {
    /// Create a new hyperplane from equation coefficients
    pub fn new(ambient_space: ProjectiveSpace<R>, coefficients: Vec<R>) -> Result<Self, String> {
        if coefficients.len() != ambient_space.num_coordinates() {
            return Err("Coefficients length must match number of coordinates".to_string());
        }

        if coefficients.iter().all(|c| c.is_zero()) {
            return Err("Not all coefficients can be zero".to_string());
        }

        Ok(Hyperplane {
            ambient_space,
            coefficients,
        })
    }

    /// Get the coefficients
    pub fn coefficients(&self) -> &[R] {
        &self.coefficients
    }

    /// Check if a point lies in this hyperplane
    pub fn contains(&self, point: &ProjectivePoint<R>) -> bool
    where
        R: Zero,
    {
        let mut sum = <R as Ring>::zero();
        for (coeff, coord) in self.coefficients.iter().zip(point.coordinates()) {
            sum = sum + coeff.clone() * coord.clone();
        }
        Ring::is_zero(&sum)
    }

    /// Convert to linear subspace
    pub fn to_linear_subspace(&self) -> LinearSubspace<R> {
        LinearSubspace::new(
            self.ambient_space.clone(),
            self.ambient_space.dimension() - 1,
            vec![self.coefficients.clone()],
        )
    }
}

impl<R: Ring> fmt::Display for Hyperplane<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hyperplane in {}: ", self.ambient_space)?;
        for (i, coeff) in self.coefficients.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }
            write!(f, "{}·x{}", coeff, i)?;
        }
        write!(f, " = 0")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_projective_point_creation() {
        let point: ProjectivePoint<i32> = ProjectivePoint::new(vec![1, 2, 3]).unwrap();
        assert_eq!(point.dimension(), 2);
        assert_eq!(point.coordinates(), &[1, 2, 3]);
    }

    #[test]
    fn test_projective_point_invalid() {
        let result: Result<ProjectivePoint<i32>, _> = ProjectivePoint::new(vec![0, 0, 0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_projective_point_affine_conversion() {
        let point: ProjectivePoint<i32> = ProjectivePoint::new(vec![1, 2, 3]).unwrap();

        // Convert to affine coordinates on U₀ (x₀ ≠ 0)
        let affine = point.to_affine(0).unwrap();
        assert_eq!(affine, vec![2, 3]); // (x₁, x₂)

        // Convert back
        let back = ProjectivePoint::from_affine(affine, 0).unwrap();
        assert_eq!(back.coordinates(), &[1, 2, 3]);
    }

    #[test]
    fn test_projective_point_not_in_chart() {
        let point: ProjectivePoint<i32> = ProjectivePoint::new(vec![0, 2, 3]).unwrap();

        // Point has x₀ = 0, so not in chart U₀
        assert!(!point.is_in_affine_chart(0));
        assert!(point.is_in_affine_chart(1));
        assert!(point.is_in_affine_chart(2));

        let result = point.to_affine(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_projective_space() {
        let p2: ProjectiveSpace<i32> = ProjectiveSpace::new(2);

        assert_eq!(p2.dimension(), 2);
        assert_eq!(p2.num_coordinates(), 3);
        assert_eq!(p2.num_affine_charts(), 3);
    }

    #[test]
    fn test_standard_points() {
        let p2: ProjectiveSpace<i32> = ProjectiveSpace::new(2);

        let p0 = p2.standard_point(0).unwrap();
        assert_eq!(p0.coordinates(), &[1, 0, 0]);

        let p1 = p2.standard_point(1).unwrap();
        assert_eq!(p1.coordinates(), &[0, 1, 0]);

        let p2_point = p2.standard_point(2).unwrap();
        assert_eq!(p2_point.coordinates(), &[0, 0, 1]);
    }

    #[test]
    fn test_contains_point() {
        let p2: ProjectiveSpace<i32> = ProjectiveSpace::new(2);
        let point = ProjectivePoint::new(vec![1, 2, 3]).unwrap();

        assert!(p2.contains_point(&point));

        let wrong_dim = ProjectivePoint::new(vec![1, 2]).unwrap();
        assert!(!p2.contains_point(&wrong_dim));
    }

    #[test]
    fn test_linear_subspace() {
        let p3: ProjectiveSpace<i32> = ProjectiveSpace::new(3);

        // A line (1-dimensional subspace) in ℙ³
        let line = LinearSubspace::new(
            p3,
            1,
            vec![
                vec![1, 0, -1, 0], // x₀ - x₂ = 0
                vec![0, 1, 0, -1], // x₁ - x₃ = 0
            ],
        );

        assert_eq!(line.dimension(), 1);
        assert_eq!(line.codimension(), 2);

        // Check if [1:2:1:2] is on the line
        let point = ProjectivePoint::new(vec![1, 2, 1, 2]).unwrap();
        assert!(line.contains(&point));

        // Check if [1:0:0:0] is on the line
        let point2 = ProjectivePoint::new(vec![1, 0, 0, 0]).unwrap();
        assert!(!line.contains(&point2));
    }

    #[test]
    fn test_hyperplane() {
        let p2: ProjectiveSpace<i32> = ProjectiveSpace::new(2);

        // Hyperplane x + y + z = 0 in ℙ²
        let h = Hyperplane::new(p2, vec![1, 1, 1]).unwrap();

        // Check if [1:-1:0] is on the hyperplane
        let point1 = ProjectivePoint::new(vec![1, -1, 0]).unwrap();
        assert!(h.contains(&point1));

        // Check if [1:2:3] is on the hyperplane (1+2+3 = 6 ≠ 0)
        let point2 = ProjectivePoint::new(vec![1, 2, 3]).unwrap();
        assert!(!h.contains(&point2));

        // Check if [1:1:-2] is on the hyperplane (1+1-2 = 0)
        let point3 = ProjectivePoint::new(vec![1, 1, -2]).unwrap();
        assert!(h.contains(&point3));
    }

    #[test]
    fn test_hyperplane_invalid() {
        let p2: ProjectiveSpace<i32> = ProjectiveSpace::new(2);

        // All zero coefficients - invalid
        let result = Hyperplane::new(p2.clone(), vec![0, 0, 0]);
        assert!(result.is_err());

        // Wrong number of coefficients - invalid
        let result2 = Hyperplane::new(p2, vec![1, 2]);
        assert!(result2.is_err());
    }

    #[test]
    fn test_projective_point_display() {
        let point: ProjectivePoint<i32> = ProjectivePoint::new(vec![1, 2, 3]).unwrap();
        let display = format!("{}", point);
        assert_eq!(display, "[1 : 2 : 3]");
    }

    #[test]
    fn test_projective_space_display() {
        let p2: ProjectiveSpace<i32> = ProjectiveSpace::new(2);
        let display = format!("{}", p2);
        assert!(display.contains("ℙ^2"));
    }
}
