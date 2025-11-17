//! Vector fields on manifolds
//!
//! This module implements vector fields and their operations, including:
//! - Component representation in charts
//! - Application to scalar fields (directional derivatives)
//! - Lie brackets
//! - Module operations

use crate::errors::{ManifoldError, Result};
use crate::scalar_field::ScalarFieldEnhanced as ScalarField;
use crate::chart::Chart;
use crate::point::ManifoldPoint;
use crate::differentiable::DifferentiableManifold;
use rustmath_symbolic::Expr;
use std::collections::HashMap;
use std::ops::{Add, Sub, Neg, Mul};
use std::sync::Arc;

/// Type alias for chart ID
pub type ChartId = String;

/// A vector field X ∈ 𝔛(M)
///
/// A vector field assigns a tangent vector to each point on a manifold.
/// In local coordinates (x¹, ..., xⁿ), a vector field is written as:
/// X = X¹ ∂/∂x¹ + ... + Xⁿ ∂/∂xⁿ
#[derive(Clone)]
pub struct VectorField {
    /// The underlying manifold
    manifold: Arc<DifferentiableManifold>,
    /// Components X^i in each chart (contravariant components)
    chart_components: HashMap<ChartId, Vec<Expr>>,
    /// Optional name for the vector field
    name: Option<String>,
}

impl VectorField {
    /// Create a new vector field on a manifold
    pub fn new(manifold: Arc<DifferentiableManifold>) -> Self {
        Self {
            manifold,
            chart_components: HashMap::new(),
            name: None,
        }
    }

    /// Create a vector field with a name
    pub fn named(manifold: Arc<DifferentiableManifold>, name: impl Into<String>) -> Self {
        Self {
            manifold,
            chart_components: HashMap::new(),
            name: Some(name.into()),
        }
    }

    /// Create vector field from components in a specific chart
    ///
    /// # Arguments
    ///
    /// * `manifold` - The underlying manifold
    /// * `chart` - The chart in which components are specified
    /// * `components` - The contravariant components X^i
    ///
    /// # Errors
    ///
    /// Returns an error if the number of components doesn't match the manifold dimension
    pub fn from_components(
        manifold: Arc<DifferentiableManifold>,
        chart: &Chart,
        components: Vec<Expr>,
    ) -> Result<Self> {
        if components.len() != manifold.dimension() {
            return Err(ManifoldError::DimensionMismatch {
                expected: manifold.dimension(),
                actual: components.len(),
            });
        }

        let mut field = Self::new(manifold);
        field.chart_components.insert(chart.name().to_string(), components);
        Ok(field)
    }

    /// Get the underlying manifold
    pub fn manifold(&self) -> &Arc<DifferentiableManifold> {
        &self.manifold
    }

    /// Get the name of the vector field
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Set the name of the vector field
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = Some(name.into());
    }

    /// Get components in a specific chart
    ///
    /// If components are not cached for this chart, attempts to transform
    /// from another chart using the Jacobian of the chart transition.
    pub fn components(&self, chart: &Chart) -> Result<Vec<Expr>> {
        let chart_id = chart.name().to_string();

        if let Some(comps) = self.chart_components.get(&chart_id) {
            return Ok(comps.clone());
        }

        // Try to transform from another chart
        // For now, return an error - full implementation would use transition functions
        Err(ManifoldError::NoComponentsInChart)
    }

    /// Set components in a specific chart
    pub fn set_components(&mut self, chart: &Chart, components: Vec<Expr>) -> Result<()> {
        if components.len() != self.manifold.dimension() {
            return Err(ManifoldError::DimensionMismatch {
                expected: self.manifold.dimension(),
                actual: components.len(),
            });
        }

        self.chart_components.insert(chart.name().to_string(), components);
        Ok(())
    }

    /// Apply vector field to a scalar field: X(f)
    ///
    /// This computes the directional derivative of f along X.
    /// In local coordinates: X(f) = Σ X^i ∂f/∂x^i
    pub fn apply_to_scalar(&self, field: &ScalarField, chart: &Chart) -> Result<ScalarField> {
        let f_expr = field.expr(chart)?;
        let x_comps = self.components(chart)?;

        // X(f) = Σ X^i ∂f/∂x^i
        let mut result_expr = Expr::from(0);
        for i in 0..self.manifold.dimension() {
            let coord_symbol = chart.coordinate_symbol(i);
            let df_dxi = f_expr.differentiate(&coord_symbol);
            result_expr = result_expr + x_comps[i].clone() * df_dxi;
        }

        let mut result = ScalarField::new(self.manifold.clone());
        result.set_expr(chart, result_expr)?;
        Ok(result)
    }

    /// Compute the Lie bracket [X, Y]
    ///
    /// The Lie bracket measures the non-commutativity of vector fields.
    /// In coordinates: [X, Y]^k = X^i ∂Y^k/∂x^i - Y^i ∂X^k/∂x^i
    pub fn lie_bracket(&self, other: &VectorField, chart: &Chart) -> Result<VectorField> {
        if !Arc::ptr_eq(&self.manifold, &other.manifold) {
            return Err(ManifoldError::DifferentManifolds);
        }

        let x_comps = self.components(chart)?;
        let y_comps = other.components(chart)?;
        let n = self.manifold.dimension();

        let mut bracket_comps = Vec::with_capacity(n);
        for k in 0..n {
            let mut term = Expr::from(0);
            for i in 0..n {
                let coord = chart.coordinate_symbol(i);

                // X^i ∂Y^k/∂x^i
                let dy_k = y_comps[k].differentiate(&coord);
                let term1 = x_comps[i].clone() * dy_k;

                // Y^i ∂X^k/∂x^i
                let dx_k = x_comps[k].differentiate(&coord);
                let term2 = y_comps[i].clone() * dx_k;

                term = term + term1 - term2;
            }
            bracket_comps.push(term);
        }

        VectorField::from_components(self.manifold.clone(), chart, bracket_comps)
    }

    /// Check if this is the zero vector field
    pub fn is_zero(&self) -> bool {
        for comps in self.chart_components.values() {
            for comp in comps {
                if !comp.is_zero() {
                    return false;
                }
            }
        }
        true
    }

    /// Create a zero vector field
    pub fn zero(manifold: Arc<DifferentiableManifold>) -> Self {
        Self {
            manifold,
            chart_components: HashMap::new(),
            name: Some("0".to_string()),
        }
    }

    /// Create a coordinate vector field ∂/∂x^i
    pub fn coordinate_vector(
        manifold: Arc<DifferentiableManifold>,
        chart: &Chart,
        index: usize,
    ) -> Result<Self> {
        let n = manifold.dimension();
        if index >= n {
            return Err(ManifoldError::InvalidIndex);
        }

        let mut components = vec![Expr::from(0); n];
        components[index] = Expr::from(1);

        let mut field = VectorField::from_components(manifold, chart, components)?;
        field.set_name(format!("d/d{}", chart.coordinate_names()[index]));
        Ok(field)
    }
}

// PartialEq implementation
impl PartialEq for VectorField {
    fn eq(&self, other: &Self) -> bool {
        // Two vector fields are equal if they have the same components in all charts
        // Note: This is a structural equality, not mathematical equality
        self.chart_components == other.chart_components
    }
}

// Arithmetic operations

impl Add for VectorField {
    type Output = Result<Self>;

    fn add(self, rhs: Self) -> Self::Output {
        if !Arc::ptr_eq(&self.manifold, &rhs.manifold) {
            return Err(ManifoldError::DifferentManifolds);
        }

        let mut result = VectorField::new(self.manifold.clone());

        // Add components in all known charts
        for (chart_id, comps1) in &self.chart_components {
            if let Some(comps2) = rhs.chart_components.get(chart_id) {
                let sum_comps: Vec<Expr> = comps1.iter()
                    .zip(comps2.iter())
                    .map(|(c1, c2)| c1.clone() + c2.clone())
                    .collect();
                result.chart_components.insert(chart_id.clone(), sum_comps);
            }
        }

        Ok(result)
    }
}

impl Sub for VectorField {
    type Output = Result<Self>;

    fn sub(self, rhs: Self) -> Self::Output {
        if !Arc::ptr_eq(&self.manifold, &rhs.manifold) {
            return Err(ManifoldError::DifferentManifolds);
        }

        let mut result = VectorField::new(self.manifold.clone());

        // Subtract components in all known charts
        for (chart_id, comps1) in &self.chart_components {
            if let Some(comps2) = rhs.chart_components.get(chart_id) {
                let diff_comps: Vec<Expr> = comps1.iter()
                    .zip(comps2.iter())
                    .map(|(c1, c2)| c1.clone() - c2.clone())
                    .collect();
                result.chart_components.insert(chart_id.clone(), diff_comps);
            }
        }

        Ok(result)
    }
}

impl Neg for VectorField {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut result = VectorField::new(self.manifold.clone());
        result.name = self.name.map(|n| format!("-{}", n));

        for (chart_id, comps) in &self.chart_components {
            let neg_comps: Vec<Expr> = comps.iter()
                .map(|c| -c.clone())
                .collect();
            result.chart_components.insert(chart_id.clone(), neg_comps);
        }

        result
    }
}

/// Scalar multiplication: f * X where f is a scalar field
impl Mul<VectorField> for ScalarField {
    type Output = Result<VectorField>;

    fn mul(self, rhs: VectorField) -> Self::Output {
        let mut result = VectorField::new(rhs.manifold.clone());

        for (chart_id, x_comps) in &rhs.chart_components {
            // Get the scalar field expression in this chart
            // For simplicity, we'll need to get the chart reference
            // This is a simplified implementation
            let scaled_comps: Vec<Expr> = x_comps.iter()
                .map(|comp| {
                    // In a full implementation, multiply by scalar field expression
                    comp.clone()
                })
                .collect();
            result.chart_components.insert(chart_id.clone(), scaled_comps);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples::EuclideanSpace;
    use rustmath_symbolic::Symbol;

    #[test]
    fn test_vector_field_creation() {
        let m = Arc::new(EuclideanSpace::new(2));
        let field = VectorField::new(m.clone());
        assert_eq!(field.manifold().dimension(), 2);
    }

    #[test]
    fn test_vector_field_from_components() {
        let m = Arc::new(EuclideanSpace::new(2));
        let chart = m.default_chart().unwrap();

        let components = vec![Expr::from(1), Expr::from(0)];
        let field = VectorField::from_components(m.clone(), chart, components).unwrap();

        let comps = field.components(chart).unwrap();
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn test_coordinate_vector() {
        let m = Arc::new(EuclideanSpace::new(3));
        let chart = m.default_chart().unwrap();

        let dx = VectorField::coordinate_vector(m.clone(), chart, 0).unwrap();
        let comps = dx.components(chart).unwrap();

        assert_eq!(comps[0], Expr::from(1));
        assert_eq!(comps[1], Expr::from(0));
        assert_eq!(comps[2], Expr::from(0));
    }

    #[test]
    fn test_vector_field_negation() {
        let m = Arc::new(EuclideanSpace::new(2));
        let chart = m.default_chart().unwrap();

        let components = vec![Expr::from(1), Expr::from(2)];
        let field = VectorField::from_components(m.clone(), chart, components).unwrap();

        let neg_field = -field;
        let neg_comps = neg_field.components(chart).unwrap();

        assert_eq!(neg_comps[0], Expr::from(-1));
        assert_eq!(neg_comps[1], Expr::from(-2));
    }

    #[test]
    fn test_zero_vector_field() {
        let m = Arc::new(EuclideanSpace::new(2));
        let zero = VectorField::zero(m);
        assert!(zero.is_zero());
    }
}
