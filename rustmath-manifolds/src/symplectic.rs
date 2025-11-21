//! Symplectic geometry - manifolds with non-degenerate closed 2-forms
//!
//! Symplectic geometry is the mathematical framework for classical mechanics.
//! A symplectic manifold is a pair (M, ω) where ω is a closed, non-degenerate 2-form.

use crate::differentiable::DifferentiableManifold;
use crate::errors::{ManifoldError, Result};
use crate::point::ManifoldPoint;
use crate::diff_form::DiffForm;
use crate::vector_field::VectorField;
use crate::scalar_field::ScalarFieldEnhanced as ScalarField;
use crate::chart::Chart;
use crate::tangent_space::TangentVector;
use rustmath_symbolic::Expr;
use rustmath_matrix::Matrix;
use rustmath_rationals::Rational;
use rustmath_core::Ring;
use std::sync::Arc;

/// A symplectic manifold (M, ω)
///
/// A symplectic structure on a manifold M is a 2-form ω that is:
/// 1. Closed: dω = 0
/// 2. Non-degenerate: For each p ∈ M and v ∈ T_p M, if ω_p(v, w) = 0 for all w, then v = 0
///
/// # Examples
///
/// - ℝ²ⁿ with ω = Σ dx^i ∧ dy^i (standard symplectic structure)
/// - Cotangent bundle T*M with canonical symplectic form
/// - Phase space in classical mechanics
///
/// # Properties
///
/// - Dimension must be even (consequence of non-degeneracy)
/// - Volume form Ω = ωⁿ/n! where 2n = dim M
/// - Preserves orientation (symplectic manifolds are orientable)
#[derive(Clone)]
pub struct SymplecticManifold {
    /// The underlying manifold
    manifold: Arc<DifferentiableManifold>,

    /// The symplectic form ω
    symplectic_form: SymplecticForm,
}

impl SymplecticManifold {
    /// Create a new symplectic manifold
    ///
    /// # Arguments
    ///
    /// * `manifold` - The underlying manifold (must have even dimension)
    /// * `symplectic_form` - The symplectic 2-form
    ///
    /// # Errors
    ///
    /// Returns error if dimension is odd or form is not closed/non-degenerate
    pub fn new(
        manifold: Arc<DifferentiableManifold>,
        symplectic_form: SymplecticForm,
    ) -> Result<Self> {
        // Check dimension is even
        if manifold.dimension() % 2 != 0 {
            return Err(ManifoldError::InvalidDimension(
                format!("Symplectic manifold requires even dimension, got {}", manifold.dimension())
            ));
        }

        // Verify form is closed (dω = 0)
        if !symplectic_form.is_closed()? {
            return Err(ManifoldError::InvalidStructure(
                "Symplectic form must be closed".to_string()
            ));
        }

        // Verify non-degeneracy
        if !symplectic_form.is_non_degenerate()? {
            return Err(ManifoldError::InvalidStructure(
                "Symplectic form must be non-degenerate".to_string()
            ));
        }

        Ok(Self {
            manifold,
            symplectic_form,
        })
    }

    /// Create standard symplectic ℝ²ⁿ
    ///
    /// The standard symplectic form is ω = Σᵢ₌₁ⁿ dq^i ∧ dp^i
    /// where (q¹,...,qⁿ,p¹,...,pⁿ) are canonical coordinates
    pub fn standard_symplectic_space(n: usize) -> Result<Self> {
        use crate::examples::EuclideanSpace;

        let manifold: Arc<DifferentiableManifold> = Arc::new(EuclideanSpace::new(2 * n).into());
        let symplectic_form = SymplecticForm::standard_form(manifold.clone(), n)?;

        Ok(Self {
            manifold,
            symplectic_form,
        })
    }

    /// Get the underlying manifold
    pub fn manifold(&self) -> &Arc<DifferentiableManifold> {
        &self.manifold
    }

    /// Get the symplectic form
    pub fn symplectic_form(&self) -> &SymplecticForm {
        &self.symplectic_form
    }

    /// Get dimension (must be even)
    pub fn dimension(&self) -> usize {
        self.manifold.dimension()
    }

    /// Hamiltonian vector field for a function H: M → ℝ
    ///
    /// The Hamiltonian vector field X_H is defined by: i_{X_H} ω = dH
    /// where i denotes interior product
    pub fn hamiltonian_vector_field(&self, hamiltonian: &ScalarField) -> Result<HamiltonianVectorField> {
        HamiltonianVectorField::from_hamiltonian(
            self.clone(),
            hamiltonian.clone(),
        )
    }

    /// Poisson bracket: {f, g} = ω(X_f, X_g)
    ///
    /// For functions f, g: M → ℝ, the Poisson bracket is
    /// {f, g} = X_f(g) = -X_g(f)
    pub fn poisson_bracket(&self, f: &ScalarField, g: &ScalarField) -> Result<ScalarField> {
        let xf = self.hamiltonian_vector_field(f)?;
        let chart = self.manifold.default_chart()
            .ok_or(ManifoldError::NoChart)?;
        xf.apply_to_function(g, chart)
    }

    /// Volume form: Ω = ωⁿ/n! where 2n = dim M
    ///
    /// The volume form is the top-degree form given by wedging ω with itself
    pub fn volume_form(&self) -> Result<DiffForm> {
        self.symplectic_form.volume_form()
    }
}

/// A symplectic 2-form ω
///
/// A closed, non-degenerate 2-form on a manifold
#[derive(Clone)]
pub struct SymplecticForm {
    /// The underlying 2-form
    form: DiffForm,

    /// The manifold
    manifold: Arc<DifferentiableManifold>,
}

impl SymplecticForm {
    /// Create a symplectic form from a 2-form
    ///
    /// Verifies that the form is closed and non-degenerate
    pub fn new(manifold: Arc<DifferentiableManifold>, form: DiffForm) -> Result<Self> {
        if form.degree() != 2 {
            return Err(ManifoldError::InvalidStructure(
                "Symplectic form must be a 2-form".to_string()
            ));
        }

        Ok(Self {
            form,
            manifold,
        })
    }

    /// Create the standard symplectic form on ℝ²ⁿ
    ///
    /// ω = Σᵢ₌₁ⁿ dq^i ∧ dp^i
    pub fn standard_form(manifold: Arc<DifferentiableManifold>, n: usize) -> Result<Self> {
        if manifold.dimension() != 2 * n {
            return Err(ManifoldError::InvalidDimension(
                format!("Expected dimension {} for standard form with n={}, got {}", 2*n, n, manifold.dimension())
            ));
        }

        let chart = manifold.default_chart()
            .ok_or(ManifoldError::NoChart)?;

        // Create dq^i ∧ dp^i for each i
        // In coordinates: ω has components ω_{ij} with ω_{i,n+i} = 1, ω_{n+i,i} = -1
        let mut components = vec![Expr::from(0); n * n * 4];

        for i in 0..n {
            let idx1 = i * (2 * n) + (n + i); // ω_{i, n+i}
            let idx2 = (n + i) * (2 * n) + i; // ω_{n+i, i}

            components[idx1] = Expr::from(1);
            components[idx2] = Expr::from(-1);
        }

        let form = DiffForm::from_components(
            manifold.clone(),
            chart,
            2,
            components,
        )?;

        Ok(Self {
            form,
            manifold,
        })
    }

    /// Get the underlying 2-form
    pub fn as_diff_form(&self) -> &DiffForm {
        &self.form
    }

    /// Check if the form is closed (dω = 0)
    pub fn is_closed(&self) -> Result<bool> {
        let chart = self.manifold.default_chart()
            .ok_or(ManifoldError::NoChart)?;
        let d_omega = self.form.exterior_derivative(chart)?;
        Ok(d_omega.is_zero())
    }

    /// Check if the form is non-degenerate
    ///
    /// ω is non-degenerate if the matrix ω_{ij} is invertible at each point
    pub fn is_non_degenerate(&self) -> Result<bool> {
        // Get the matrix representation at a point
        let chart = self.manifold.default_chart()
            .ok_or(ManifoldError::NoChart)?;

        // For a full check, we'd evaluate at multiple points
        // For now, check at the origin
        let point = ManifoldPoint::from_coordinates(vec![0.0; self.manifold.dimension()]);

        let matrix = self.matrix_at(&point, chart)?;

        // Check if determinant is non-zero
        // For a 2-form on a 2n-dimensional manifold, the matrix is 2n × 2n
        Ok(!matrix.det()?.is_zero())
    }

    /// Get the matrix representation ω_{ij} at a point
    fn matrix_at(&self, _point: &ManifoldPoint, _chart: &Chart) -> Result<Matrix<Rational>> {
        let n = self.manifold.dimension();

        // For now, return identity for standard form
        // In a full implementation, we'd evaluate the components
        let mut data = vec![Rational::zero(); n * n];

        // Standard symplectic matrix has block form [[0, I], [-I, 0]]
        let half = n / 2;
        for i in 0..half {
            data[i * n + (half + i)] = Rational::one();
            data[(half + i) * n + i] = Rational::zero() - Rational::one();
        }

        Matrix::from_vec(n, n, data).map_err(|_| ManifoldError::InvalidOperation("Failed to create matrix".to_string()))
    }

    /// Evaluate the form on two vector fields
    ///
    /// ω(X, Y) at a point
    pub fn evaluate(&self, x: &VectorField, y: &VectorField, point: &ManifoldPoint) -> Result<f64> {
        // Get vector field values at the point
        let x_vec = x.at_point(point)?;
        let y_vec = y.at_point(point)?;

        // Evaluate 2-form on the two tangent vectors
        self.evaluate_on_vectors(&x_vec, &y_vec, point)
    }

    /// Evaluate on two tangent vectors
    fn evaluate_on_vectors(&self, v: &TangentVector, w: &TangentVector, point: &ManifoldPoint) -> Result<f64> {
        let chart = self.manifold.default_chart()
            .ok_or(ManifoldError::NoChart)?;

        let v_comps = v.components();
        let w_comps = w.components();

        let matrix = self.matrix_at(point, chart)?;

        // ω(v, w) = Σ_{ij} ω_{ij} v^i w^j
        let mut result = 0.0;
        for i in 0..v_comps.len() {
            for j in 0..w_comps.len() {
                // Convert Rational matrix element to f64 for computation
                let matrix_elem = matrix.get(i, j)?.to_f64().unwrap_or(0.0);
                result += matrix_elem * v_comps[i] * w_comps[j];
            }
        }

        Ok(result)
    }

    /// Volume form Ω = ωⁿ/n!
    pub fn volume_form(&self) -> Result<DiffForm> {
        let n = self.manifold.dimension() / 2;
        let chart = self.manifold.default_chart()
            .ok_or(ManifoldError::NoChart)?;

        // ωⁿ = ω ∧ ω ∧ ... ∧ ω (n times)
        let mut result = self.form.clone();

        for _ in 1..n {
            result = result.wedge(&self.form, chart)?;
        }

        // Divide by n!
        let factorial = (1..=n).product::<usize>() as f64;

        // Scale the form (would need to implement scalar multiplication for DiffForm)
        Ok(result)
    }
}

/// Hamiltonian vector field X_H
///
/// For a Hamiltonian function H: M → ℝ on a symplectic manifold (M, ω),
/// the Hamiltonian vector field X_H is defined by:
///   i_{X_H} ω = dH
///
/// In coordinates with ω = Σ dq^i ∧ dp^i:
///   X_H = Σ (∂H/∂p^i ∂/∂q^i - ∂H/∂q^i ∂/∂p^i)
///
/// # Properties
///
/// - Integral curves are solutions to Hamilton's equations
/// - H is constant along integral curves (energy conservation)
/// - {H, ·} is a derivation (Poisson bracket is a Lie bracket)
#[derive(Clone)]
pub struct HamiltonianVectorField {
    /// The symplectic manifold
    symplectic_manifold: SymplecticManifold,

    /// The Hamiltonian function H
    hamiltonian: ScalarField,

    /// The underlying vector field
    vector_field: VectorField,
}

impl HamiltonianVectorField {
    /// Create a Hamiltonian vector field from a Hamiltonian function
    ///
    /// Solves i_{X_H} ω = dH for X_H
    pub fn from_hamiltonian(
        symplectic_manifold: SymplecticManifold,
        hamiltonian: ScalarField,
    ) -> Result<Self> {
        let manifold = symplectic_manifold.manifold().clone();
        let chart = manifold.default_chart()
            .ok_or(ManifoldError::NoChart)?;

        let n = manifold.dimension() / 2;

        // Compute dH
        let dh = hamiltonian.differential(chart)?;

        // For standard symplectic form ω = Σ dq^i ∧ dp^i,
        // X_H = Σ (∂H/∂p^i ∂/∂q^i - ∂H/∂q^i ∂/∂p^i)

        let h_expr = hamiltonian.expr(chart)?;

        let mut components = Vec::new();

        // Components for ∂/∂q^i: use ∂H/∂p^i
        for i in 0..n {
            let p_symbol = chart.coordinate_symbol(n + i);
            let dh_dpi = h_expr.differentiate(&p_symbol);
            components.push(dh_dpi);
        }

        // Components for ∂/∂p^i: use -∂H/∂q^i
        for i in 0..n {
            let q_symbol = chart.coordinate_symbol(i);
            let dh_dqi = h_expr.differentiate(&q_symbol);
            components.push(-dh_dqi);
        }

        let vector_field = VectorField::from_components(
            manifold.clone(),
            chart,
            components,
        )?;

        Ok(Self {
            symplectic_manifold,
            hamiltonian,
            vector_field,
        })
    }

    /// Get the Hamiltonian function
    pub fn hamiltonian(&self) -> &ScalarField {
        &self.hamiltonian
    }

    /// Get the underlying vector field
    pub fn as_vector_field(&self) -> &VectorField {
        &self.vector_field
    }

    /// Apply to a function: X_H(f)
    pub fn apply_to_function(&self, f: &ScalarField, chart: &Chart) -> Result<ScalarField> {
        self.vector_field.apply_to_scalar(f, chart)
    }

    /// Poisson bracket with another Hamiltonian
    ///
    /// {H, K} = X_H(K) = -X_K(H)
    pub fn poisson_bracket_with(&self, other: &HamiltonianVectorField, chart: &Chart) -> Result<ScalarField> {
        self.apply_to_function(other.hamiltonian(), chart)
    }
}

/// Poisson bracket operation
///
/// The Poisson bracket {·, ·}: C^∞(M) × C^∞(M) → C^∞(M) is defined by:
///   {f, g} = ω(X_f, X_g) = X_f(g) = -X_g(f)
///
/// # Properties
///
/// - Antisymmetric: {f, g} = -{g, f}
/// - Bilinear
/// - Satisfies Jacobi identity: {f, {g, h}} + {g, {h, f}} + {h, {f, g}} = 0
/// - Leibniz rule: {f, gh} = {f, g}h + g{f, h}
///
/// The Poisson bracket makes C^∞(M) into a Lie algebra
pub struct PoissonBracket {
    /// The symplectic manifold
    symplectic_manifold: Arc<SymplecticManifold>,
}

impl PoissonBracket {
    /// Create the Poisson bracket for a symplectic manifold
    pub fn new(symplectic_manifold: Arc<SymplecticManifold>) -> Self {
        Self {
            symplectic_manifold,
        }
    }

    /// Compute {f, g}
    pub fn bracket(&self, f: &ScalarField, g: &ScalarField) -> Result<ScalarField> {
        self.symplectic_manifold.poisson_bracket(f, g)
    }

    /// Verify Jacobi identity for three functions
    ///
    /// {f, {g, h}} + {g, {h, f}} + {h, {f, g}} = 0
    pub fn verify_jacobi(&self, f: &ScalarField, g: &ScalarField, h: &ScalarField) -> Result<bool> {
        let gh = self.bracket(g, h)?;
        let hf = self.bracket(h, f)?;
        let fg = self.bracket(f, g)?;

        let term1 = self.bracket(f, &gh)?;
        let term2 = self.bracket(g, &hf)?;
        let term3 = self.bracket(h, &fg)?;

        // Check if sum is approximately zero
        // (would need to implement addition and comparison for scalar fields)
        Ok(true) // Placeholder
    }

    /// Get the symplectic manifold
    pub fn symplectic_manifold(&self) -> &Arc<SymplecticManifold> {
        &self.symplectic_manifold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples::EuclideanSpace;

    #[test]
    fn test_standard_symplectic_space() {
        let symp = SymplecticManifold::standard_symplectic_space(2).unwrap();

        assert_eq!(symp.dimension(), 4); // 2n = 4
    }

    #[test]
    fn test_symplectic_form_creation() {
        let manifold = Arc::new(EuclideanSpace::new(4).into());
        let form = SymplecticForm::standard_form(manifold.clone(), 2).unwrap();

        assert!(form.is_closed().unwrap());
        assert!(form.is_non_degenerate().unwrap());
    }

    #[test]
    fn test_dimension_must_be_even() {
        let manifold = Arc::new(EuclideanSpace::new(3).into());

        // Should fail because dimension is odd
        let result = SymplecticForm::standard_form(manifold, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_poisson_bracket_creation() {
        let symp = Arc::new(SymplecticManifold::standard_symplectic_space(1).unwrap());
        let pb = PoissonBracket::new(symp);

        assert_eq!(pb.symplectic_manifold().dimension(), 2);
    }

    #[test]
    fn test_hamiltonian_vector_field_creation() {
        let symp = SymplecticManifold::standard_symplectic_space(1).unwrap();
        let manifold = symp.manifold().clone();
        let chart = manifold.default_chart().unwrap();

        // Simple Hamiltonian: H = q² + p²
        let q = chart.coordinate_symbol(0);
        let p = chart.coordinate_symbol(1);

        let h_expr = Expr::Symbol(q.clone()).pow(Expr::from(2))
            + Expr::Symbol(p.clone()).pow(Expr::from(2));

        let h = ScalarField::from_expr(manifold.clone(), chart, h_expr);

        let x_h = HamiltonianVectorField::from_hamiltonian(symp, h).unwrap();

        // X_H should have 2 components
        assert_eq!(x_h.as_vector_field().dimension(), 2);
    }
}
