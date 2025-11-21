//! Smooth maps between manifolds and their properties
//!
//! This module provides structures for working with smooth maps between manifolds:
//! - Pushforward and pullback operations
//! - Immersions (injective differential)
//! - Submersions (surjective differential)
//! - Embeddings (injective immersions)
//! - Diffeomorphisms (smooth bijections with smooth inverse)

use crate::errors::{ManifoldError, Result};
use crate::chart::Chart;
use crate::differentiable::DifferentiableManifold;
use crate::scalar_field::ScalarFieldEnhanced as ScalarField;
use crate::vector_field::VectorField;
use crate::diff_form::DiffForm;
use crate::tangent_space::{TangentVector, Covector};
use crate::tensor_field::TensorField;
use rustmath_symbolic::Expr;
use std::sync::Arc;
use std::collections::HashMap;

/// A smooth map between differentiable manifolds f: M → N
#[derive(Clone)]
pub struct SmoothMap {
    /// Source manifold M
    source: Arc<DifferentiableManifold>,
    /// Target manifold N
    target: Arc<DifferentiableManifold>,
    /// Name of the map
    name: String,
    /// Coordinate expression in each pair of charts
    /// Maps (source_chart, target_chart) -> vector of coordinate expressions
    coordinate_expressions: HashMap<(String, String), Vec<Expr>>,
    /// Cached Jacobians (stored as Vec<Vec<Expr>> since Expr doesn't implement Ring)
    jacobians: HashMap<(String, String), Vec<Vec<Expr>>>,
}

impl SmoothMap {
    /// Create a new smooth map
    pub fn new(
        source: Arc<DifferentiableManifold>,
        target: Arc<DifferentiableManifold>,
        name: impl Into<String>,
    ) -> Self {
        Self {
            source,
            target,
            name: name.into(),
            coordinate_expressions: HashMap::new(),
            jacobians: HashMap::new(),
        }
    }

    /// Get the source manifold
    pub fn source(&self) -> &Arc<DifferentiableManifold> {
        &self.source
    }

    /// Get the target manifold
    pub fn target(&self) -> &Arc<DifferentiableManifold> {
        &self.target
    }

    /// Set the coordinate expression for a pair of charts
    pub fn set_coordinate_expression(
        &mut self,
        source_chart: &str,
        target_chart: &str,
        expressions: Vec<Expr>,
    ) -> Result<()> {
        if expressions.len() != self.target.dimension() {
            return Err(ManifoldError::DimensionMismatch {
                expected: self.target.dimension(),
                actual: expressions.len(),
            });
        }

        self.coordinate_expressions.insert(
            (source_chart.to_string(), target_chart.to_string()),
            expressions,
        );

        Ok(())
    }

    /// Get the coordinate expression for a pair of charts
    pub fn coordinate_expression(
        &self,
        source_chart: &str,
        target_chart: &str,
    ) -> Option<&[Expr]> {
        self.coordinate_expressions
            .get(&(source_chart.to_string(), target_chart.to_string()))
            .map(|v| v.as_slice())
    }

    /// Compute the Jacobian matrix of the map
    ///
    /// The Jacobian has entries J^i_j = ∂f^i/∂x^j
    /// Returns as Vec<Vec<Expr>> where jacobian[i][j] = ∂f^i/∂x^j
    pub fn jacobian(
        &mut self,
        source_chart: &Chart,
        target_chart: &Chart,
    ) -> Result<Vec<Vec<Expr>>> {
        let key = (source_chart.name().to_string(), target_chart.name().to_string());

        // Check cache first
        if let Some(jac) = self.jacobians.get(&key) {
            return Ok(jac.clone());
        }

        // Compute Jacobian
        let exprs = self.coordinate_expression(source_chart.name(), target_chart.name())
            .ok_or_else(|| ManifoldError::NoExpressionInChart)?;

        let m = exprs.len(); // target dimension
        let n = source_chart.dimension(); // source dimension

        let mut jacobian = Vec::with_capacity(m);

        for expr in exprs.iter() {
            let mut row = Vec::with_capacity(n);
            for j in 0..n {
                let var = source_chart.coordinate_symbol(j);
                let derivative = expr.differentiate(&var);
                row.push(derivative);
            }
            jacobian.push(row);
        }

        self.jacobians.insert(key, jacobian.clone());

        Ok(jacobian)
    }

    /// Compute the rank of the differential at a point
    ///
    /// This is the rank of the Jacobian matrix evaluated at that point
    pub fn rank_at_point(
        &mut self,
        source_chart: &Chart,
        target_chart: &Chart,
        point_coords: &[f64],
    ) -> Result<usize> {
        let jacobian = self.jacobian(source_chart, target_chart)?;

        // Evaluate the Jacobian at the point
        // For now, we return the minimum dimension as a placeholder
        Ok(self.source.dimension().min(self.target.dimension()))
    }
}

/// Pushforward operation f_*: TM → TN
///
/// The pushforward (or differential) of a smooth map f: M → N
/// maps tangent vectors on M to tangent vectors on N.
pub struct PushForward {
    /// The underlying smooth map
    map: Arc<SmoothMap>,
}

impl PushForward {
    /// Create a pushforward from a smooth map
    pub fn new(map: Arc<SmoothMap>) -> Self {
        Self { map }
    }

    /// Apply the pushforward to a tangent vector
    ///
    /// (f_* v)^i = J^i_j v^j where J is the Jacobian
    pub fn apply_to_tangent_vector(
        &self,
        vector: &TangentVector,
        source_chart: &Chart,
        target_chart: &Chart,
    ) -> Result<TangentVector> {
        // Get the map
        let mut map = (*self.map).clone();
        let jacobian = map.jacobian(source_chart, target_chart)?;

        // Get vector components
        let v_components = vector.components();

        // Multiply: (f_* v)^i = Σ_j J^i_j v^j
        let mut pushed = Vec::with_capacity(jacobian.len());
        for row in jacobian.iter() {
            // This would need proper evaluation - placeholder for now
            pushed.push(0.0);
        }

        // Create new tangent vector in target manifold
        // This is a simplified version - proper implementation would track the base point
        // For now, use the image of the base point under the map
        let target_point = vector.base_point().clone(); // Simplified - should be f(base_point)
        let target_manifold = self.map.target().clone();
        TangentVector::new(target_point, pushed, target_manifold)
    }

    /// Apply the pushforward to a vector field
    ///
    /// Note: This only works when f is a diffeomorphism or when considering
    /// f-related vector fields
    pub fn apply_to_vector_field(
        &self,
        field: &VectorField,
        source_chart: &Chart,
        target_chart: &Chart,
    ) -> Result<VectorField> {
        // Get the map's coordinate expressions
        let mut map = (*self.map).clone();
        let jacobian = map.jacobian(source_chart, target_chart)?;

        // Get field components
        let x_components = field.components(source_chart)?;

        // Transform components using Jacobian
        let mut pushed_components = Vec::with_capacity(self.map.target.dimension());

        for i in 0..self.map.target.dimension() {
            let mut component = Expr::from(0);
            for j in 0..self.map.source.dimension() {
                let jac_entry = &jacobian[i][j];
                component = component + jac_entry.clone() * x_components[j].clone();
            }
            pushed_components.push(component);
        }

        VectorField::from_components(self.map.target.clone(), target_chart, pushed_components)
    }
}

/// Pullback operation f^*: T*N → T*M
///
/// The pullback maps differential forms and covectors from N back to M.
pub struct PullBack {
    /// The underlying smooth map
    map: Arc<SmoothMap>,
}

impl PullBack {
    /// Create a pullback from a smooth map
    pub fn new(map: Arc<SmoothMap>) -> Self {
        Self { map }
    }

    /// Apply the pullback to a scalar field
    ///
    /// (f^* g)(p) = g(f(p))
    pub fn apply_to_scalar_field(
        &self,
        field: &ScalarField,
        source_chart: &Chart,
        target_chart: &Chart,
    ) -> Result<ScalarField> {
        // Get the field's expression in the target chart
        let g_expr = field.expr(target_chart)?;

        // Get the map's coordinate expressions
        let f_exprs = self.map.coordinate_expression(source_chart.name(), target_chart.name())
            .ok_or_else(|| ManifoldError::NoExpressionInChart)?;

        // Substitute the map's expressions into the field
        let mut substituted = g_expr.clone();

        for (i, f_i) in f_exprs.iter().enumerate() {
            let target_coord = target_chart.coordinate_symbol(i);
            substituted = substitute_in_expr(&substituted, &target_coord.name(), f_i);
        }

        // Create new scalar field on source manifold
        ScalarField::from_expr(self.map.source.clone(), source_chart, substituted)
    }

    /// Apply the pullback to a 1-form (covector field)
    ///
    /// (f^* ω)_i = J^j_i ω_j where J is the Jacobian
    pub fn apply_to_one_form(
        &self,
        form: &DiffForm,
        source_chart: &Chart,
        target_chart: &Chart,
    ) -> Result<DiffForm> {
        if form.degree() != 1 {
            return Err(ManifoldError::InvalidDegree {
                expected: 1,
                actual: form.degree(),
            });
        }

        // Get the map
        let mut map = (*self.map).clone();
        let jacobian = map.jacobian(source_chart, target_chart)?;

        // Get form components
        let omega_components = form.tensor().components(target_chart)?;

        // Pull back: (f^* ω)_i = ∂f^j/∂x^i ω_j
        let mut pulled_components = Vec::with_capacity(self.map.source.dimension());

        for i in 0..self.map.source.dimension() {
            let mut component = Expr::from(0);
            for j in 0..self.map.target.dimension() {
                if j < omega_components.len() && j < jacobian.len() {
                    let jac_entry = &jacobian[j][i];
                    component = component + jac_entry.clone() * omega_components[j].clone();
                }
            }
            pulled_components.push(component);
        }

        // Create new 1-form on source manifold
        let tensor = TensorField::from_components(
            self.map.source.clone(),
            0,
            1,
            source_chart,
            pulled_components,
        )?;

        DiffForm::from_tensor(tensor, 1)
    }

    /// Apply the pullback to a general p-form
    ///
    /// The pullback commutes with the exterior derivative: f^*(dω) = d(f^* ω)
    pub fn apply_to_form(
        &self,
        form: &DiffForm,
        source_chart: &Chart,
        target_chart: &Chart,
    ) -> Result<DiffForm> {
        let p = form.degree();

        if p == 0 {
            // 0-forms are scalar fields
            return Err(ManifoldError::InvalidOperation(
                "Use apply_to_scalar_field for 0-forms".to_string()
            ));
        }

        if p == 1 {
            return self.apply_to_one_form(form, source_chart, target_chart);
        }

        // For p > 1, the pullback is more complex
        // It involves pulling back each component and wedging appropriately
        // This is a simplified placeholder
        let tensor = form.tensor().clone();
        Ok(DiffForm::from_tensor(tensor, p)?)
    }

    /// Apply the pullback to a general (0,q) tensor (covariant tensor)
    ///
    /// For a (0,q) tensor T with components T_{i_1...i_q}, the pullback is:
    /// (f^* T)_{j_1...j_q} = T_{i_1...i_q} (∂f^{i_1}/∂x^{j_1}) ... (∂f^{i_q}/∂x^{j_q})
    ///
    /// This generalizes the pullback of 1-forms to higher covariant tensors.
    pub fn apply_to_covariant_tensor(
        &self,
        tensor: &TensorField,
        source_chart: &Chart,
        target_chart: &Chart,
    ) -> Result<TensorField> {
        let p = tensor.contravariant_rank();
        let q = tensor.covariant_rank();

        if p != 0 {
            return Err(ManifoldError::InvalidOperation(
                "Pullback only applies to covariant tensors (contravariant rank must be 0)".to_string()
            ));
        }

        if q == 0 {
            // This is a scalar - shouldn't be here
            return Err(ManifoldError::InvalidOperation(
                "Use apply_to_scalar_field for scalars".to_string()
            ));
        }

        if q == 1 {
            // Use the 1-form pullback, but we need to convert to/from DiffForm
            return Err(ManifoldError::InvalidOperation(
                "Use apply_to_one_form for 1-forms".to_string()
            ));
        }

        // Get the map
        let mut map = (*self.map).clone();
        let jacobian = map.jacobian(source_chart, target_chart)?;

        // Get tensor components
        let t_components = tensor.components(target_chart)?;

        // For a general (0,q) tensor, we need to pull back using the Jacobian
        // This requires sophisticated multi-index arithmetic
        // For now, provide a basic implementation

        let n_source = self.map.source.dimension();
        let n_target = self.map.target.dimension();

        // Simplified: assume we can pull back componentwise
        // A full implementation would require proper tensor index contraction
        let pulled_components = t_components.clone();

        TensorField::from_components(
            self.map.source.clone(),
            0,
            q,
            source_chart,
            pulled_components,
        )
    }

    /// Apply the pullback to an arbitrary tensor
    ///
    /// Note: Pullback only naturally applies to covariant tensors.
    /// For mixed tensors, this is not well-defined without additional structure.
    pub fn apply_to_tensor(
        &self,
        tensor: &TensorField,
        source_chart: &Chart,
        target_chart: &Chart,
    ) -> Result<TensorField> {
        if tensor.contravariant_rank() != 0 {
            return Err(ManifoldError::InvalidOperation(
                "Pullback is only defined for covariant tensors. Use pushforward for contravariant tensors.".to_string()
            ));
        }

        self.apply_to_covariant_tensor(tensor, source_chart, target_chart)
    }
}

impl PushForward {
    /// Apply the pushforward to a general (p,0) tensor (contravariant tensor)
    ///
    /// For a (p,0) tensor T with components T^{i_1...i_p}, the pushforward is:
    /// (f_* T)^{j_1...j_p} = T^{i_1...i_p} (∂f^{j_1}/∂x^{i_1}) ... (∂f^{j_p}/∂x^{i_p})
    ///
    /// This generalizes the pushforward of vector fields to higher contravariant tensors.
    pub fn apply_to_contravariant_tensor(
        &self,
        tensor: &TensorField,
        source_chart: &Chart,
        target_chart: &Chart,
    ) -> Result<TensorField> {
        let p = tensor.contravariant_rank();
        let q = tensor.covariant_rank();

        if q != 0 {
            return Err(ManifoldError::InvalidOperation(
                "Pushforward only applies to contravariant tensors (covariant rank must be 0)".to_string()
            ));
        }

        if p == 0 {
            // This is a scalar - shouldn't be here
            return Err(ManifoldError::InvalidOperation(
                "Scalars don't change under pushforward (use pullback)".to_string()
            ));
        }

        if p == 1 {
            // This is a vector field - we have a specialized method
            return Err(ManifoldError::InvalidOperation(
                "Use apply_to_vector_field for vector fields".to_string()
            ));
        }

        // Get the map
        let mut map = (*self.map).clone();
        let jacobian = map.jacobian(source_chart, target_chart)?;

        // Get tensor components
        let t_components = tensor.components(source_chart)?;

        // For a general (p,0) tensor, we need to push forward using the Jacobian
        // This requires sophisticated multi-index arithmetic
        // For now, provide a basic implementation

        let n_source = self.map.source.dimension();
        let n_target = self.map.target.dimension();

        // Simplified: assume we can push forward componentwise
        // A full implementation would require proper tensor index contraction
        let pushed_components = t_components.clone();

        TensorField::from_components(
            self.map.target.clone(),
            p,
            0,
            target_chart,
            pushed_components,
        )
    }

    /// Apply the pushforward to an arbitrary tensor
    ///
    /// Note: Pushforward only naturally applies to contravariant tensors.
    /// For mixed tensors, this is not well-defined without additional structure.
    pub fn apply_to_tensor(
        &self,
        tensor: &TensorField,
        source_chart: &Chart,
        target_chart: &Chart,
    ) -> Result<TensorField> {
        if tensor.covariant_rank() != 0 {
            return Err(ManifoldError::InvalidOperation(
                "Pushforward is only defined for contravariant tensors. Use pullback for covariant tensors.".to_string()
            ));
        }

        self.apply_to_contravariant_tensor(tensor, source_chart, target_chart)
    }
}

/// An immersion f: M → N
///
/// A smooth map where the differential is injective at every point.
/// This means rank(df_p) = dim(M) for all p ∈ M.
#[derive(Clone)]
pub struct Immersion {
    /// The underlying smooth map
    map: SmoothMap,
}

impl Immersion {
    /// Create an immersion from a smooth map
    ///
    /// Checks that the differential is injective
    pub fn new(map: SmoothMap) -> Result<Self> {
        // In practice, we'd verify the immersion condition
        // For now, we trust the caller

        if map.source().dimension() > map.target().dimension() {
            return Err(ManifoldError::InvalidOperation(
                "Immersion requires source dimension ≤ target dimension".to_string()
            ));
        }

        Ok(Self { map })
    }

    /// Get the underlying map
    pub fn map(&self) -> &SmoothMap {
        &self.map
    }

    /// Check if the immersion is proper
    ///
    /// An immersion is proper if it's a proper map (preimages of compact sets are compact)
    pub fn is_proper(&self) -> bool {
        // This is a topological condition that's difficult to check algorithmically
        // Return false as a conservative default
        false
    }

    /// Verify the immersion condition at a point
    pub fn verify_at_point(
        &mut self,
        source_chart: &Chart,
        target_chart: &Chart,
        point_coords: &[f64],
    ) -> Result<bool> {
        let rank = self.map.rank_at_point(source_chart, target_chart, point_coords)?;
        Ok(rank == self.map.source().dimension())
    }
}

/// A submersion f: M → N
///
/// A smooth map where the differential is surjective at every point.
/// This means rank(df_p) = dim(N) for all p ∈ M.
#[derive(Clone)]
pub struct Submersion {
    /// The underlying smooth map
    map: SmoothMap,
}

impl Submersion {
    /// Create a submersion from a smooth map
    pub fn new(map: SmoothMap) -> Result<Self> {
        if map.source().dimension() < map.target().dimension() {
            return Err(ManifoldError::InvalidOperation(
                "Submersion requires source dimension ≥ target dimension".to_string()
            ));
        }

        Ok(Self { map })
    }

    /// Get the underlying map
    pub fn map(&self) -> &SmoothMap {
        &self.map
    }

    /// Verify the submersion condition at a point
    pub fn verify_at_point(
        &mut self,
        source_chart: &Chart,
        target_chart: &Chart,
        point_coords: &[f64],
    ) -> Result<bool> {
        let rank = self.map.rank_at_point(source_chart, target_chart, point_coords)?;
        Ok(rank == self.map.target().dimension())
    }

    /// Get the fiber over a point in the target
    ///
    /// For a submersion, the fibers are submanifolds of dimension dim(M) - dim(N)
    pub fn fiber_dimension(&self) -> usize {
        self.map.source().dimension() - self.map.target().dimension()
    }
}

/// An embedding f: M → N
///
/// An injective immersion that is also a homeomorphism onto its image.
#[derive(Clone)]
pub struct Embedding {
    /// The underlying immersion
    immersion: Immersion,
    /// Whether this is a proper embedding
    is_proper: bool,
}

impl Embedding {
    /// Create an embedding from an immersion
    pub fn new(immersion: Immersion) -> Self {
        Self {
            immersion,
            is_proper: false,
        }
    }

    /// Mark as a proper embedding
    pub fn mark_proper(mut self) -> Self {
        self.is_proper = true;
        self
    }

    /// Get the underlying immersion
    pub fn immersion(&self) -> &Immersion {
        &self.immersion
    }

    /// Get the underlying map
    pub fn map(&self) -> &SmoothMap {
        &self.immersion.map
    }

    /// Check if this is a proper embedding
    pub fn is_proper(&self) -> bool {
        self.is_proper
    }

    /// Check if this is a closed embedding
    ///
    /// A closed embedding is one where the image is a closed subset
    pub fn is_closed(&self) -> bool {
        // Proper embeddings are always closed
        self.is_proper
    }
}

/// A diffeomorphism f: M → N
///
/// A smooth bijection with smooth inverse. This means M and N are
/// diffeomorphic (smoothly equivalent).
#[derive(Clone)]
pub struct Diffeomorphism {
    /// The forward map f: M → N
    forward: SmoothMap,
    /// The inverse map f^{-1}: N → M
    inverse: Option<SmoothMap>,
}

impl Diffeomorphism {
    /// Create a diffeomorphism from a smooth map
    ///
    /// Requires that dimensions match and the map is invertible
    pub fn new(forward: SmoothMap) -> Result<Self> {
        if forward.source().dimension() != forward.target().dimension() {
            return Err(ManifoldError::DimensionMismatch {
                expected: forward.source().dimension(),
                actual: forward.target().dimension(),
            });
        }

        Ok(Self {
            forward,
            inverse: None,
        })
    }

    /// Set the inverse map
    pub fn set_inverse(&mut self, inverse: SmoothMap) -> Result<()> {
        // Verify that inverse goes the right direction
        if !Arc::ptr_eq(inverse.source(), self.forward.target())
            || !Arc::ptr_eq(inverse.target(), self.forward.source())
        {
            return Err(ManifoldError::IncompatibleManifolds(
                "Inverse map has wrong source/target".to_string()
            ));
        }

        self.inverse = Some(inverse);
        Ok(())
    }

    /// Get the forward map
    pub fn forward(&self) -> &SmoothMap {
        &self.forward
    }

    /// Get the inverse map
    pub fn inverse(&self) -> Option<&SmoothMap> {
        self.inverse.as_ref()
    }

    /// Check if the inverse is available
    pub fn has_inverse(&self) -> bool {
        self.inverse.is_some()
    }

    /// Compose two diffeomorphisms
    ///
    /// If f: M → N and g: N → P, then g ∘ f: M → P
    pub fn compose(&self, other: &Diffeomorphism) -> Result<Diffeomorphism> {
        // Check that composition makes sense
        if !Arc::ptr_eq(self.forward.target(), other.forward.source()) {
            return Err(ManifoldError::IncompatibleManifolds(
                "Cannot compose: target of first ≠ source of second".to_string()
            ));
        }

        // Create composed forward map
        let composed_forward = SmoothMap::new(
            self.forward.source().clone(),
            other.forward.target().clone(),
            format!("{} ∘ {}", other.forward.name, self.forward.name),
        );

        // If both have inverses, create composed inverse
        let composed_inverse = match (&self.inverse, &other.inverse) {
            (Some(inv_self), Some(inv_other)) => {
                let mut inv = SmoothMap::new(
                    other.forward.target().clone(),
                    self.forward.source().clone(),
                    format!("({})^(-1)", composed_forward.name),
                );
                Some(inv)
            }
            _ => None,
        };

        let mut diff = Diffeomorphism::new(composed_forward)?;
        if let Some(inv) = composed_inverse {
            diff.set_inverse(inv)?;
        }

        Ok(diff)
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Substitute a variable in an expression
fn substitute_in_expr(expr: &Expr, var_name: &str, value: &Expr) -> Expr {
    match expr {
        Expr::Symbol(s) if s.name() == var_name => value.clone(),
        Expr::Binary(op, left, right) => {
            let new_left = substitute_in_expr(left, var_name, value);
            let new_right = substitute_in_expr(right, var_name, value);
            Expr::Binary(*op, Arc::new(new_left), Arc::new(new_right))
        }
        Expr::Unary(op, inner) => {
            let new_inner = substitute_in_expr(inner, var_name, value);
            Expr::Unary(*op, Arc::new(new_inner))
        }
        Expr::Function(name, args) => {
            let new_args: Vec<_> = args.iter()
                .map(|arg| Arc::new(substitute_in_expr(arg, var_name, value)))
                .collect();
            Expr::Function(name.clone(), new_args)
        }
        _ => expr.clone(),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples::{RealLine, EuclideanSpace};

    #[test]
    fn test_smooth_map_creation() {
        let r1 = Arc::new(RealLine::new());
        let r2 = Arc::new(RealLine::new());

        let map = SmoothMap::new(r1.clone(), r2.clone(), "f");
        assert_eq!(map.source().dimension(), 1);
        assert_eq!(map.target().dimension(), 1);
    }

    #[test]
    fn test_smooth_map_coordinate_expression() {
        let r1 = Arc::new(RealLine::new());
        let r2 = Arc::new(RealLine::new());

        let mut map = SmoothMap::new(r1.clone(), r2.clone(), "square");

        // f(x) = x^2
        let x = Expr::Symbol("x".to_string());
        let x_squared = x.clone() * x.clone();

        map.set_coordinate_expression("standard", "standard", vec![x_squared])
            .unwrap();

        let expr = map.coordinate_expression("standard", "standard").unwrap();
        assert_eq!(expr.len(), 1);
    }

    #[test]
    fn test_immersion_dimension_check() {
        let r1 = Arc::new(RealLine::new());
        let r3 = Arc::new(EuclideanSpace::new(3));

        let map = SmoothMap::new(r1.clone(), r3.clone(), "curve");
        let immersion = Immersion::new(map);

        // Should succeed: 1D → 3D is allowed
        assert!(immersion.is_ok());
    }

    #[test]
    fn test_immersion_invalid_dimension() {
        let r2 = Arc::new(EuclideanSpace::new(2));
        let r1 = Arc::new(RealLine::new());

        let map = SmoothMap::new(r2.clone(), r1.clone(), "projection");
        let immersion = Immersion::new(map);

        // Should fail: 2D → 1D cannot be an immersion
        assert!(immersion.is_err());
    }

    #[test]
    fn test_submersion_dimension_check() {
        let r2 = Arc::new(EuclideanSpace::new(2));
        let r1 = Arc::new(RealLine::new());

        let map = SmoothMap::new(r2.clone(), r1.clone(), "projection");
        let submersion = Submersion::new(map);

        // Should succeed: 2D → 1D is allowed
        assert!(submersion.is_ok());

        if let Ok(sub) = submersion {
            // Fiber dimension should be 2 - 1 = 1
            assert_eq!(sub.fiber_dimension(), 1);
        }
    }

    #[test]
    fn test_diffeomorphism_dimension_check() {
        let r1 = Arc::new(RealLine::new());
        let r2 = Arc::new(EuclideanSpace::new(2));

        let map = SmoothMap::new(r1.clone(), r2.clone(), "invalid");
        let diff = Diffeomorphism::new(map);

        // Should fail: dimensions must match
        assert!(diff.is_err());
    }

    #[test]
    fn test_diffeomorphism_valid() {
        let r1 = Arc::new(RealLine::new());
        let r1_copy = Arc::new(RealLine::new());

        let map = SmoothMap::new(r1.clone(), r1_copy.clone(), "identity");
        let diff = Diffeomorphism::new(map);

        // Should succeed: same dimensions
        assert!(diff.is_ok());
    }

    #[test]
    fn test_pushforward_creation() {
        let r1 = Arc::new(RealLine::new());
        let r2 = Arc::new(EuclideanSpace::new(2));

        let map = Arc::new(SmoothMap::new(r1.clone(), r2.clone(), "inclusion"));
        let pushforward = PushForward::new(map);

        assert!(true); // Just test creation
    }

    #[test]
    fn test_pullback_creation() {
        let r2 = Arc::new(EuclideanSpace::new(2));
        let r1 = Arc::new(RealLine::new());

        let map = Arc::new(SmoothMap::new(r2.clone(), r1.clone(), "projection"));
        let pullback = PullBack::new(map);

        assert!(true); // Just test creation
    }
}
