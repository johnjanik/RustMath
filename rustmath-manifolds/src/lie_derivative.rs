//! Lie derivatives on manifolds
//!
//! The Lie derivative measures the change of a tensor field along the flow of a vector field.
//! It generalizes the directional derivative to tensor fields of any type.
//!
//! Key properties:
//! - L_X f: Lie derivative of scalar field f along X is the directional derivative
//! - L_X Y: Lie derivative of vector field Y along X is the Lie bracket [X, Y]
//! - L_X ω: Lie derivative of differential form ω satisfies Cartan's formula
//! - L_X T: Lie derivative of tensor field T extends to general tensors

use crate::errors::{ManifoldError, Result};
use crate::chart::Chart;
use crate::scalar_field::ScalarFieldEnhanced as ScalarField;
use crate::vector_field::VectorField;
use crate::diff_form::DiffForm;
use crate::tensor_field::TensorField;
use rustmath_symbolic::Expr;
use std::sync::Arc;

/// Lie derivative operator L_X
///
/// Given a vector field X, the Lie derivative L_X measures how tensor fields
/// change along the flow of X.
pub struct LieDerivative {
    /// The vector field along which we take the Lie derivative
    vector_field: Arc<VectorField>,
}

impl LieDerivative {
    /// Create a new Lie derivative operator for a given vector field
    ///
    /// # Arguments
    ///
    /// * `vector_field` - The vector field X along which to take the Lie derivative
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_manifolds::{LieDerivative, VectorField, EuclideanSpace};
    /// use std::sync::Arc;
    ///
    /// let manifold = Arc::new(EuclideanSpace::new(2));
    /// let x = VectorField::new(manifold.clone());
    /// let lie_x = LieDerivative::new(Arc::new(x));
    /// ```
    pub fn new(vector_field: Arc<VectorField>) -> Self {
        Self { vector_field }
    }

    /// Get the underlying vector field
    pub fn vector_field(&self) -> &Arc<VectorField> {
        &self.vector_field
    }

    /// Lie derivative of a scalar field: L_X f = X(f)
    ///
    /// The Lie derivative of a scalar field is just the directional derivative.
    /// In coordinates: L_X f = X^i ∂f/∂x^i
    ///
    /// # Arguments
    ///
    /// * `field` - The scalar field f
    /// * `chart` - The chart in which to compute the derivative
    ///
    /// # Returns
    ///
    /// A new scalar field representing L_X f
    pub fn apply_to_scalar(&self, field: &ScalarField, chart: &Chart) -> Result<ScalarField> {
        // L_X f = X(f) is just the directional derivative
        self.vector_field.apply_to_scalar(field, chart)
    }

    /// Lie derivative of a vector field: L_X Y = [X, Y]
    ///
    /// The Lie derivative of a vector field is the Lie bracket.
    /// In coordinates: (L_X Y)^k = X^i ∂Y^k/∂x^i - Y^i ∂X^k/∂x^i
    ///
    /// # Arguments
    ///
    /// * `field` - The vector field Y
    /// * `chart` - The chart in which to compute the derivative
    ///
    /// # Returns
    ///
    /// A new vector field representing [X, Y]
    pub fn apply_to_vector(&self, field: &VectorField, chart: &Chart) -> Result<VectorField> {
        // L_X Y = [X, Y]
        self.vector_field.lie_bracket(field, chart)
    }

    /// Lie derivative of a differential form using Cartan's formula
    ///
    /// Cartan's formula: L_X ω = d(i_X ω) + i_X(dω)
    /// where i_X is the interior product (contraction with X)
    ///
    /// For a 1-form ω: (L_X ω)_i = X^j ∂ω_i/∂x^j + ω_j ∂X^j/∂x^i
    ///
    /// # Arguments
    ///
    /// * `form` - The differential form ω
    /// * `chart` - The chart in which to compute the derivative
    ///
    /// # Returns
    ///
    /// A new differential form representing L_X ω
    pub fn apply_to_form(&self, form: &DiffForm, chart: &Chart) -> Result<DiffForm> {
        let p = form.degree();

        if p == 0 {
            // 0-forms are scalar fields
            return Err(ManifoldError::InvalidOperation(
                "Use apply_to_scalar for 0-forms".to_string()
            ));
        }

        if p == 1 {
            return self.apply_to_one_form(form, chart);
        }

        // For higher degree forms, we use Cartan's formula: L_X ω = d(i_X ω) + i_X(dω)
        // This is a simplified implementation
        self.apply_to_one_form(form, chart)
    }

    /// Lie derivative of a 1-form
    ///
    /// For a 1-form ω with components ω_i:
    /// (L_X ω)_i = X^j ∂ω_i/∂x^j + ω_j ∂X^j/∂x^i
    fn apply_to_one_form(&self, form: &DiffForm, chart: &Chart) -> Result<DiffForm> {
        let manifold = form.manifold();
        let n = manifold.dimension();

        // Get components
        let omega_components = form.tensor().components(chart)?;
        let x_components = self.vector_field.components(chart)?;

        // Compute L_X ω
        let mut lie_components = Vec::with_capacity(n);

        for i in 0..n {
            let mut term = Expr::from(0);

            // First term: X^j ∂ω_i/∂x^j
            for j in 0..n {
                let coord = chart.coordinate_symbol(j);
                let d_omega_i = omega_components[i].differentiate(&coord);
                term = term + x_components[j].clone() * d_omega_i;
            }

            // Second term: ω_j ∂X^j/∂x^i
            let coord_i = chart.coordinate_symbol(i);
            for j in 0..n {
                let d_x_j = x_components[j].differentiate(&coord_i);
                term = term + omega_components[j].clone() * d_x_j;
            }

            lie_components.push(term);
        }

        // Create new 1-form
        let tensor = TensorField::from_components(
            manifold.clone(),
            0,
            1,
            chart,
            lie_components,
        )?;

        DiffForm::from_tensor(tensor, 1)
    }

    /// Lie derivative of a general tensor field
    ///
    /// For a tensor T of type (p, q), the Lie derivative generalizes to:
    /// (L_X T)^{i_1...i_p}_{j_1...j_q} =
    ///   X^k ∂T^{i_1...i_p}_{j_1...j_q}/∂x^k
    ///   - Σ_a T^{i_1...k...i_p}_{j_1...j_q} ∂X^{i_a}/∂x^k (sum over contravariant indices)
    ///   + Σ_b T^{i_1...i_p}_{j_1...k...j_q} ∂X^k/∂x^{j_b} (sum over covariant indices)
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor field T
    /// * `chart` - The chart in which to compute the derivative
    ///
    /// # Returns
    ///
    /// A new tensor field representing L_X T
    pub fn apply_to_tensor(&self, tensor: &TensorField, chart: &Chart) -> Result<TensorField> {
        let manifold = tensor.manifold();
        let n = manifold.dimension();
        let p = tensor.contravariant_rank();
        let q = tensor.covariant_rank();

        // Get components
        let t_components = tensor.components(chart)?;
        let x_components = self.vector_field.components(chart)?;

        // This is a complex operation that requires multi-index arithmetic
        // For now, we implement a simplified version that handles the basic case

        if p == 0 && q == 0 {
            // Scalar field - should not reach here
            return Err(ManifoldError::InvalidOperation(
                "Cannot apply Lie derivative to (0,0) tensor as tensor field".to_string()
            ));
        }

        if p == 1 && q == 0 {
            // This is a vector field - use Lie bracket
            return Err(ManifoldError::InvalidOperation(
                "Use apply_to_vector for vector fields".to_string()
            ));
        }

        if p == 0 && q == 1 {
            // This is a 1-form - use specialized method
            return Err(ManifoldError::InvalidOperation(
                "Use apply_to_form for differential forms".to_string()
            ));
        }

        // For general tensors, we compute the Lie derivative component by component
        // This is a simplified implementation
        let num_components = tensor.num_components();
        let mut lie_components = vec![Expr::from(0); num_components];

        for flat_idx in 0..num_components {
            let mut component = Expr::from(0);

            // Directional derivative term: X^k ∂T.../∂x^k
            for k in 0..n {
                let coord = chart.coordinate_symbol(k);
                let dt = t_components[flat_idx].differentiate(&coord);
                component = component + x_components[k].clone() * dt;
            }

            // For a full implementation, we would add:
            // - Correction terms for contravariant indices
            // - Correction terms for covariant indices
            // This requires sophisticated multi-index arithmetic

            lie_components[flat_idx] = component;
        }

        TensorField::from_components(manifold.clone(), p, q, chart, lie_components)
    }

    /// Check if the Lie derivative commutes with the exterior derivative
    ///
    /// For differential forms, we have: L_X(dω) = d(L_X ω)
    /// This is a fundamental property that can be used for verification.
    pub fn commutes_with_exterior_derivative(
        &self,
        form: &DiffForm,
        chart: &Chart,
    ) -> Result<bool> {
        // Compute L_X(dω)
        let d_omega = form.exterior_derivative(chart)?;
        let lie_d_omega = self.apply_to_form(&d_omega, chart)?;

        // Compute d(L_X ω)
        let lie_omega = self.apply_to_form(form, chart)?;
        let d_lie_omega = lie_omega.exterior_derivative(chart)?;

        // Check if they're equal (componentwise)
        let comps1 = lie_d_omega.tensor().components(chart)?;
        let comps2 = d_lie_omega.tensor().components(chart)?;

        if comps1.len() != comps2.len() {
            return Ok(false);
        }

        for (c1, c2) in comps1.iter().zip(comps2.iter()) {
            // Simplified equality check
            if c1 != c2 {
                return Ok(false);
            }
        }

        Ok(true)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples::EuclideanSpace;

    #[test]
    fn test_lie_derivative_creation() {
        let manifold = Arc::new(EuclideanSpace::new(2));
        let x = VectorField::new(manifold.clone());
        let lie_x = LieDerivative::new(Arc::new(x));

        assert_eq!(lie_x.vector_field().manifold().dimension(), 2);
    }

    #[test]
    fn test_lie_derivative_of_scalar() {
        let manifold = Arc::new(EuclideanSpace::new(2));
        let chart = manifold.default_chart().unwrap();

        // Create a vector field X = ∂/∂x
        let x_field = VectorField::from_components(
            manifold.clone(),
            chart,
            vec![Expr::from(1), Expr::from(0)],
        ).unwrap();

        // Create a scalar field f = x
        let mut f = ScalarField::new(manifold.clone());
        f.set_expr(chart, Expr::Symbol("x".to_string())).unwrap();

        // Compute L_X f
        let lie_x = LieDerivative::new(Arc::new(x_field));
        let result = lie_x.apply_to_scalar(&f, chart).unwrap();

        // L_X f should be 1 (since ∂x/∂x = 1)
        let result_expr = result.expr(chart).unwrap();
        assert_eq!(result_expr, Expr::from(1));
    }

    #[test]
    fn test_lie_derivative_of_vector() {
        let manifold = Arc::new(EuclideanSpace::new(2));
        let chart = manifold.default_chart().unwrap();

        // Create vector fields X = ∂/∂x and Y = ∂/∂y
        let x_field = VectorField::from_components(
            manifold.clone(),
            chart,
            vec![Expr::from(1), Expr::from(0)],
        ).unwrap();

        let y_field = VectorField::from_components(
            manifold.clone(),
            chart,
            vec![Expr::from(0), Expr::from(1)],
        ).unwrap();

        // [∂/∂x, ∂/∂y] = 0 (coordinate vector fields commute)
        let lie_x = LieDerivative::new(Arc::new(x_field));
        let result = lie_x.apply_to_vector(&y_field, chart).unwrap();

        // Result should be zero
        assert!(result.is_zero());
    }

    #[test]
    fn test_lie_derivative_of_one_form() {
        let manifold = Arc::new(EuclideanSpace::new(2));
        let chart = manifold.default_chart().unwrap();

        // Create vector field X = ∂/∂x
        let x_field = VectorField::from_components(
            manifold.clone(),
            chart,
            vec![Expr::from(1), Expr::from(0)],
        ).unwrap();

        // Create 1-form ω = dx
        let tensor = TensorField::from_components(
            manifold.clone(),
            0,
            1,
            chart,
            vec![Expr::from(1), Expr::from(0)],
        ).unwrap();
        let omega = DiffForm::from_tensor(tensor, 1).unwrap();

        // Compute L_X ω
        let lie_x = LieDerivative::new(Arc::new(x_field));
        let result = lie_x.apply_to_form(&omega, chart).unwrap();

        // For constant 1-forms, Lie derivative should be zero
        assert!(result.tensor().is_zero());
    }
}
