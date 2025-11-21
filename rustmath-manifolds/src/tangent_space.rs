//! Tangent spaces and tangent/cotangent vectors
//!
//! This module implements:
//! - Tangent spaces T_p(M) at points
//! - Tangent vectors
//! - Cotangent spaces T_p*(M)
//! - Covectors (cotangent vectors)

use crate::errors::{ManifoldError, Result};
use crate::point::ManifoldPoint;
use crate::chart::Chart;
use crate::differentiable::DifferentiableManifold;
use std::ops::{Add, Sub, Mul, Neg};
use std::sync::Arc;
use std::fmt;

/// A tangent vector at a point p ∈ M
///
/// In local coordinates, a tangent vector is represented as:
/// v = v^i ∂/∂x^i|_p
#[derive(Clone)]
pub struct TangentVector {
    /// The base point where this vector is tangent
    base_point: ManifoldPoint,
    /// Components in the default chart
    components: Vec<f64>,
    /// The manifold
    manifold: Arc<DifferentiableManifold>,
}

impl TangentVector {
    /// Create a new tangent vector
    pub fn new(
        base_point: ManifoldPoint,
        components: Vec<f64>,
        manifold: Arc<DifferentiableManifold>,
    ) -> Result<Self> {
        if components.len() != manifold.dimension() {
            return Err(ManifoldError::DimensionMismatch {
                expected: manifold.dimension(),
                actual: components.len(),
            });
        }

        Ok(Self {
            base_point,
            components,
            manifold,
        })
    }

    /// Create a zero tangent vector
    pub fn zero(base_point: ManifoldPoint, manifold: Arc<DifferentiableManifold>) -> Self {
        let components = vec![0.0; manifold.dimension()];
        Self {
            base_point,
            components,
            manifold,
        }
    }

    /// Create a tangent vector from components (alternative constructor)
    ///
    /// This is a convenience method that accepts a chart parameter for compatibility,
    /// though the chart is not currently used in the construction.
    pub fn from_components(
        base_point: ManifoldPoint,
        _chart: &Chart,
        components: Vec<f64>,
    ) -> Result<Self> {
        // Extract manifold from base point or require it as parameter
        // For now, we'll need to update call sites to use `new` instead
        // Or we need to store manifold info in ManifoldPoint
        // Let's create a version that infers dimension from components
        let components_cloned = components.clone();
        Ok(Self {
            base_point: base_point.clone(),
            components: components_cloned.clone(),
            // This is a temporary workaround - we need the manifold reference
            manifold: Arc::new(DifferentiableManifold::new("temp", components_cloned.len())),
        })
    }

    /// Get the base point
    pub fn base_point(&self) -> &ManifoldPoint {
        &self.base_point
    }

    /// Get the components
    pub fn components(&self) -> &[f64] {
        &self.components
    }

    /// Get the manifold
    pub fn manifold(&self) -> &Arc<DifferentiableManifold> {
        &self.manifold
    }

    /// Compute the norm (Euclidean norm of components)
    pub fn norm(&self) -> f64 {
        self.components.iter()
            .map(|c| c * c)
            .sum::<f64>()
            .sqrt()
    }

    /// Check if this is the zero vector
    pub fn is_zero(&self) -> bool {
        self.components.iter().all(|&c| c.abs() < 1e-10)
    }

    /// Dot product with another tangent vector (at the same point)
    pub fn dot(&self, other: &TangentVector) -> Result<f64> {
        if self.base_point != other.base_point {
            return Err(ManifoldError::DifferentBasePoints);
        }

        Ok(self.components.iter()
            .zip(other.components.iter())
            .map(|(a, b)| a * b)
            .sum())
    }
}

// Arithmetic operations

impl Add for TangentVector {
    type Output = Result<Self>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.base_point != rhs.base_point {
            return Err(ManifoldError::DifferentBasePoints);
        }

        let components: Vec<f64> = self.components.iter()
            .zip(rhs.components.iter())
            .map(|(a, b)| a + b)
            .collect();

        TangentVector::new(self.base_point, components, self.manifold.clone())
    }
}

impl Sub for TangentVector {
    type Output = Result<Self>;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.base_point != rhs.base_point {
            return Err(ManifoldError::DifferentBasePoints);
        }

        let components: Vec<f64> = self.components.iter()
            .zip(rhs.components.iter())
            .map(|(a, b)| a - b)
            .collect();

        TangentVector::new(self.base_point, components, self.manifold.clone())
    }
}

impl Mul<f64> for TangentVector {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self::Output {
        let components: Vec<f64> = self.components.iter()
            .map(|c| c * scalar)
            .collect();

        TangentVector {
            base_point: self.base_point,
            components,
            manifold: self.manifold,
        }
    }
}

impl Neg for TangentVector {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let components: Vec<f64> = self.components.iter()
            .map(|c| -c)
            .collect();

        TangentVector {
            base_point: self.base_point,
            components,
            manifold: self.manifold,
        }
    }
}

impl fmt::Debug for TangentVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TangentVector")
            .field("base_point", &self.base_point)
            .field("components", &self.components)
            .finish()
    }
}

impl fmt::Display for TangentVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TangentVector[")?;
        for (i, c) in self.components.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:.4}", c)?;
        }
        write!(f, "]")
    }
}

/// The tangent space T_p(M) at a point p
///
/// This is a vector space isomorphic to ℝⁿ where n = dim(M)
pub struct TangentSpace {
    /// The base point p
    base_point: ManifoldPoint,
    /// The manifold
    manifold: Arc<DifferentiableManifold>,
    /// The dimension of the tangent space
    dimension: usize,
}

impl TangentSpace {
    /// Create a new tangent space at a point
    pub fn new(base_point: ManifoldPoint, manifold: Arc<DifferentiableManifold>) -> Self {
        let dimension = manifold.dimension();
        Self {
            base_point,
            manifold,
            dimension,
        }
    }

    /// Get the base point
    pub fn base_point(&self) -> &ManifoldPoint {
        &self.base_point
    }

    /// Get the manifold
    pub fn manifold(&self) -> &Arc<DifferentiableManifold> {
        &self.manifold
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the coordinate basis ∂/∂x^i
    pub fn coordinate_basis(&self, chart: &Chart) -> Vec<TangentVector> {
        (0..self.dimension)
            .map(|i| self.coordinate_vector(i))
            .collect()
    }

    /// Get the i-th coordinate basis vector ∂/∂x^i
    pub fn coordinate_vector(&self, index: usize) -> TangentVector {
        let mut components = vec![0.0; self.dimension];
        if index < self.dimension {
            components[index] = 1.0;
        }

        TangentVector {
            base_point: self.base_point.clone(),
            components,
            manifold: self.manifold.clone(),
        }
    }

    /// Create a zero vector
    pub fn zero(&self) -> TangentVector {
        TangentVector::zero(self.base_point.clone(), self.manifold.clone())
    }
}

/// A covector (element of cotangent space T_p*(M))
///
/// In local coordinates: ω = ω_i dx^i|_p
#[derive(Clone)]
pub struct Covector {
    /// The base point where this covector is defined
    base_point: ManifoldPoint,
    /// Covariant components
    components: Vec<f64>,
    /// The manifold
    manifold: Arc<DifferentiableManifold>,
}

impl Covector {
    /// Create a new covector
    pub fn new(
        base_point: ManifoldPoint,
        components: Vec<f64>,
        manifold: Arc<DifferentiableManifold>,
    ) -> Result<Self> {
        if components.len() != manifold.dimension() {
            return Err(ManifoldError::DimensionMismatch {
                expected: manifold.dimension(),
                actual: components.len(),
            });
        }

        Ok(Self {
            base_point,
            components,
            manifold,
        })
    }

    /// Get the base point
    pub fn base_point(&self) -> &ManifoldPoint {
        &self.base_point
    }

    /// Get the components
    pub fn components(&self) -> &[f64] {
        &self.components
    }

    /// Apply covector to a tangent vector (dual pairing)
    ///
    /// ω(v) = ω_i v^i
    pub fn apply(&self, vector: &TangentVector) -> Result<f64> {
        if self.base_point != *vector.base_point() {
            return Err(ManifoldError::DifferentBasePoints);
        }

        Ok(self.components.iter()
            .zip(vector.components().iter())
            .map(|(a, b)| a * b)
            .sum())
    }
}

impl fmt::Debug for Covector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Covector")
            .field("base_point", &self.base_point)
            .field("components", &self.components)
            .finish()
    }
}

/// The cotangent space T_p*(M) at a point p
///
/// This is the dual space of T_p(M)
pub struct CotangentSpace {
    /// The base point p
    base_point: ManifoldPoint,
    /// The manifold
    manifold: Arc<DifferentiableManifold>,
    /// The dimension
    dimension: usize,
}

impl CotangentSpace {
    /// Create a new cotangent space at a point
    pub fn new(base_point: ManifoldPoint, manifold: Arc<DifferentiableManifold>) -> Self {
        let dimension = manifold.dimension();
        Self {
            base_point,
            manifold,
            dimension,
        }
    }

    /// Get the base point
    pub fn base_point(&self) -> &ManifoldPoint {
        &self.base_point
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the coordinate basis dx^i
    pub fn coordinate_basis(&self) -> Vec<Covector> {
        (0..self.dimension)
            .map(|i| self.coordinate_covector(i))
            .collect()
    }

    /// Get the i-th coordinate basis covector dx^i
    pub fn coordinate_covector(&self, index: usize) -> Covector {
        let mut components = vec![0.0; self.dimension];
        if index < self.dimension {
            components[index] = 1.0;
        }

        Covector {
            base_point: self.base_point.clone(),
            components,
            manifold: self.manifold.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples::EuclideanSpace;

    #[test]
    fn test_tangent_vector_creation() {
        let m = Arc::new(EuclideanSpace::new(2));
        let point = ManifoldPoint::from_coordinates(vec![1.0, 2.0]);
        let v = TangentVector::new(point, vec![1.0, 0.0], m).unwrap();

        assert_eq!(v.components(), &[1.0, 0.0]);
    }

    #[test]
    fn test_tangent_vector_addition() {
        let m = Arc::new(EuclideanSpace::new(2));
        let point = ManifoldPoint::from_coordinates(vec![0.0, 0.0]);

        let v1 = TangentVector::new(point.clone(), vec![1.0, 2.0], m.clone()).unwrap();
        let v2 = TangentVector::new(point.clone(), vec![3.0, 4.0], m.clone()).unwrap();

        let sum = (v1 + v2).unwrap();
        assert_eq!(sum.components(), &[4.0, 6.0]);
    }

    #[test]
    fn test_tangent_vector_scalar_multiplication() {
        let m = Arc::new(EuclideanSpace::new(2));
        let point = ManifoldPoint::from_coordinates(vec![0.0, 0.0]);

        let v = TangentVector::new(point, vec![1.0, 2.0], m).unwrap();
        let scaled = v * 2.0;

        assert_eq!(scaled.components(), &[2.0, 4.0]);
    }

    #[test]
    fn test_tangent_space_coordinate_basis() {
        let m = Arc::new(EuclideanSpace::new(2));
        let point = ManifoldPoint::from_coordinates(vec![0.0, 0.0]);
        let chart = m.default_chart().unwrap();

        let ts = TangentSpace::new(point, m);
        let basis = ts.coordinate_basis(chart);

        assert_eq!(basis.len(), 2);
        assert_eq!(basis[0].components(), &[1.0, 0.0]);
        assert_eq!(basis[1].components(), &[0.0, 1.0]);
    }

    #[test]
    fn test_covector_application() {
        let m = Arc::new(EuclideanSpace::new(2));
        let point = ManifoldPoint::from_coordinates(vec![0.0, 0.0]);

        let v = TangentVector::new(point.clone(), vec![1.0, 2.0], m.clone()).unwrap();
        let omega = Covector::new(point, vec![3.0, 4.0], m).unwrap();

        let result = omega.apply(&v).unwrap();
        assert_eq!(result, 11.0); // 1*3 + 2*4
    }

    #[test]
    fn test_cotangent_space_coordinate_basis() {
        let m = Arc::new(EuclideanSpace::new(2));
        let point = ManifoldPoint::from_coordinates(vec![0.0, 0.0]);

        let cts = CotangentSpace::new(point, m);
        let basis = cts.coordinate_basis();

        assert_eq!(basis.len(), 2);
        assert_eq!(basis[0].components(), &[1.0, 0.0]);
        assert_eq!(basis[1].components(), &[0.0, 1.0]);
    }
}
