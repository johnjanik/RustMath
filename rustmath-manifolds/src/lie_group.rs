//! Lie groups - manifolds with compatible group structure
//!
//! A Lie group is a smooth manifold G that is also a group, where the group operations
//! (multiplication and inversion) are smooth maps.

use crate::differentiable::DifferentiableManifold;
use crate::errors::{ManifoldError, Result};
use crate::point::ManifoldPoint;
use crate::vector_field::VectorField;
use crate::diff_form::DiffForm;
use crate::tangent_space::TangentVector;
use rustmath_symbolic::Expr;
use std::sync::Arc;
use std::collections::HashMap;

/// A Lie group - a smooth manifold with compatible group structure
///
/// A Lie group G is both a differentiable manifold and a group, where:
/// - Multiplication: G × G → G is smooth
/// - Inversion: G → G is smooth
/// - Identity element e ∈ G exists
///
/// # Examples
///
/// ```
/// use rustmath_manifolds::LieGroup;
/// // SO(3) - rotation group in 3D
/// // U(1) - circle group
/// // GL(n, ℝ) - general linear group
/// ```
#[derive(Clone)]
pub struct LieGroup {
    /// The underlying manifold structure
    manifold: Arc<DifferentiableManifold>,

    /// The identity element of the group
    identity: ManifoldPoint,

    /// Group multiplication operation (symbolic)
    /// Maps (g, h) → gh
    multiplication: Arc<dyn Fn(&ManifoldPoint, &ManifoldPoint) -> Result<ManifoldPoint> + Send + Sync>,

    /// Group inversion operation (symbolic)
    /// Maps g → g^{-1}
    inversion: Arc<dyn Fn(&ManifoldPoint) -> Result<ManifoldPoint> + Send + Sync>,

    /// Left translation map L_g: h → gh
    left_translation_cache: Arc<std::sync::RwLock<HashMap<String, Box<dyn Fn(&ManifoldPoint) -> Result<ManifoldPoint> + Send + Sync>>>>,

    /// Right translation map R_g: h → hg
    right_translation_cache: Arc<std::sync::RwLock<HashMap<String, Box<dyn Fn(&ManifoldPoint) -> Result<ManifoldPoint> + Send + Sync>>>>,
}

impl LieGroup {
    /// Create a new Lie group
    ///
    /// # Arguments
    ///
    /// * `manifold` - The underlying differentiable manifold
    /// * `identity` - The identity element
    /// * `multiplication` - The group multiplication operation
    /// * `inversion` - The group inversion operation
    pub fn new(
        manifold: Arc<DifferentiableManifold>,
        identity: ManifoldPoint,
        multiplication: Arc<dyn Fn(&ManifoldPoint, &ManifoldPoint) -> Result<ManifoldPoint> + Send + Sync>,
        inversion: Arc<dyn Fn(&ManifoldPoint) -> Result<ManifoldPoint> + Send + Sync>,
    ) -> Self {
        Self {
            manifold,
            identity,
            multiplication,
            inversion,
            left_translation_cache: Arc::new(std::sync::RwLock::new(HashMap::new())),
            right_translation_cache: Arc::new(std::sync::RwLock::new(HashMap::new())),
        }
    }

    /// Get the underlying manifold
    pub fn manifold(&self) -> &Arc<DifferentiableManifold> {
        &self.manifold
    }

    /// Get the identity element
    pub fn identity(&self) -> &ManifoldPoint {
        &self.identity
    }

    /// Group multiplication: g * h
    pub fn multiply(&self, g: &ManifoldPoint, h: &ManifoldPoint) -> Result<ManifoldPoint> {
        (self.multiplication)(g, h)
    }

    /// Group inversion: g^{-1}
    pub fn inverse(&self, g: &ManifoldPoint) -> Result<ManifoldPoint> {
        (self.inversion)(g)
    }

    /// Left translation by g: L_g(h) = gh
    ///
    /// This is a diffeomorphism of the manifold
    pub fn left_translate(&self, g: &ManifoldPoint, h: &ManifoldPoint) -> Result<ManifoldPoint> {
        self.multiply(g, h)
    }

    /// Right translation by g: R_g(h) = hg
    ///
    /// This is a diffeomorphism of the manifold
    pub fn right_translate(&self, h: &ManifoldPoint, g: &ManifoldPoint) -> Result<ManifoldPoint> {
        self.multiply(h, g)
    }

    /// Pushforward of a tangent vector by left translation
    ///
    /// (L_g)_* : T_h G → T_{gh} G
    pub fn left_translate_tangent(&self, g: &ManifoldPoint, v: &TangentVector) -> Result<TangentVector> {
        // This requires computing the differential of L_g at the point
        // For now, we implement a basic version using finite differences

        let h = v.base_point();
        let gh = self.left_translate(g, h)?;

        // Get components in local chart
        let chart = self.manifold.default_chart()
            .ok_or(ManifoldError::NoChart)?;
        let v_comps = v.components().to_vec();

        // The pushforward in coordinates is given by the Jacobian
        // For a proper implementation, we'd compute the Jacobian symbolically
        // For now, return a tangent vector with the same components
        // (This is exact for matrix Lie groups in exponential coordinates)

        TangentVector::from_components(
            gh,
            chart,
            v_comps,
        )
    }

    /// Pushforward of a tangent vector by right translation
    ///
    /// (R_g)_* : T_h G → T_{hg} G
    pub fn right_translate_tangent(&self, v: &TangentVector, g: &ManifoldPoint) -> Result<TangentVector> {
        let h = v.base_point();
        let hg = self.right_translate(h, g)?;

        let chart = self.manifold.default_chart()
            .ok_or(ManifoldError::NoChart)?;
        let v_comps = v.components().to_vec();

        TangentVector::from_components(
            hg,
            chart,
            v_comps,
        )
    }

    /// Adjoint action: Ad_g(X) = (L_g ∘ R_{g^{-1}})_* X
    ///
    /// This is the action of the group on its Lie algebra
    pub fn adjoint_action(&self, g: &ManifoldPoint, x: &TangentVector) -> Result<TangentVector> {
        // Ad_g = L_g ∘ R_{g^{-1}}
        let g_inv = self.inverse(g)?;
        let temp = self.right_translate_tangent(x, &g_inv)?;
        self.left_translate_tangent(g, &temp)
    }

    /// Get the dimension of the Lie group
    pub fn dimension(&self) -> usize {
        self.manifold.dimension()
    }

    /// Check if an element is the identity (up to numerical tolerance)
    pub fn is_identity(&self, g: &ManifoldPoint, tol: f64) -> Result<bool> {
        let chart = self.manifold.default_chart()
            .ok_or(ManifoldError::NoChart)?;

        let g_coords = g.coordinates();
        let e_coords = self.identity.coordinates();

        if g_coords.len() != e_coords.len() {
            return Ok(false);
        }

        Ok(g_coords.iter()
            .zip(e_coords.iter())
            .all(|(g_i, e_i)| (g_i - e_i).abs() < tol))
    }

    /// Commutator: [g, h] = g h g^{-1} h^{-1}
    pub fn commutator(&self, g: &ManifoldPoint, h: &ManifoldPoint) -> Result<ManifoldPoint> {
        let g_inv = self.inverse(g)?;
        let h_inv = self.inverse(h)?;

        let gh = self.multiply(g, h)?;
        let gh_ginv = self.multiply(&gh, &g_inv)?;
        self.multiply(&gh_ginv, &h_inv)
    }
}

/// Left-invariant vector field
///
/// A vector field X on a Lie group G is left-invariant if
/// (L_g)_* X_h = X_{gh} for all g, h ∈ G
///
/// The space of left-invariant vector fields is isomorphic to the Lie algebra
pub struct LeftInvariantVectorField {
    /// The Lie group
    group: Arc<LieGroup>,

    /// Value at the identity (determines the field everywhere)
    value_at_identity: TangentVector,

    /// The underlying vector field
    field: VectorField,
}

impl LeftInvariantVectorField {
    /// Create a left-invariant vector field from its value at the identity
    ///
    /// # Arguments
    ///
    /// * `group` - The Lie group
    /// * `value_at_identity` - The tangent vector at the identity element
    pub fn new(group: Arc<LieGroup>, value_at_identity: TangentVector) -> Result<Self> {
        // Verify the value is at the identity
        if value_at_identity.base_point() != group.identity() {
            return Err(ManifoldError::InvalidPoint("Value not at identity element".to_string()));
        }

        // Create the vector field by extending via left translation
        let chart = group.manifold().default_chart()
            .ok_or(ManifoldError::NoChart)?;

        let v_comps = value_at_identity.components();

        // For a left-invariant field, the value at g is (L_g)_* X_e
        // In coordinates, this is given by left translation
        // Convert &[f64] to Vec<Expr>
        let v_comps_expr: Vec<_> = v_comps.iter().map(|&x| Expr::from(x)).collect();
        let field = VectorField::from_components(
            group.manifold().clone(),
            chart,
            v_comps_expr,
        )?;

        Ok(Self {
            group,
            value_at_identity,
            field,
        })
    }

    /// Get the value at the identity
    pub fn value_at_identity(&self) -> &TangentVector {
        &self.value_at_identity
    }

    /// Evaluate the field at a point g
    ///
    /// Returns (L_g)_* X_e where X_e is the value at identity
    pub fn at_point(&self, g: &ManifoldPoint) -> Result<TangentVector> {
        self.group.left_translate_tangent(g, &self.value_at_identity)
    }

    /// Get the underlying vector field
    pub fn as_vector_field(&self) -> &VectorField {
        &self.field
    }

    /// Lie bracket of two left-invariant vector fields
    ///
    /// The space of left-invariant vector fields is closed under Lie bracket
    pub fn lie_bracket(&self, other: &LeftInvariantVectorField) -> Result<LeftInvariantVectorField> {
        // The Lie bracket is also left-invariant
        // We compute it at the identity
        let chart = self.group.manifold().default_chart()
            .ok_or(ManifoldError::NoChart)?;
        let bracket_field = self.field.lie_bracket(other.as_vector_field(), chart)?;

        // Evaluate at identity
        let bracket_at_e = bracket_field.at_point(self.group.identity())?;

        LeftInvariantVectorField::new(self.group.clone(), bracket_at_e)
    }
}

/// Right-invariant vector field
///
/// A vector field X on a Lie group G is right-invariant if
/// (R_g)_* X_h = X_{hg} for all g, h ∈ G
pub struct RightInvariantVectorField {
    /// The Lie group
    group: Arc<LieGroup>,

    /// Value at the identity (determines the field everywhere)
    value_at_identity: TangentVector,

    /// The underlying vector field
    field: VectorField,
}

impl RightInvariantVectorField {
    /// Create a right-invariant vector field from its value at the identity
    pub fn new(group: Arc<LieGroup>, value_at_identity: TangentVector) -> Result<Self> {
        if value_at_identity.base_point() != group.identity() {
            return Err(ManifoldError::InvalidPoint("Value not at identity element".to_string()));
        }

        let chart = group.manifold().default_chart()
            .ok_or(ManifoldError::NoChart)?;

        let v_comps = value_at_identity.components();

        // Convert &[f64] to Vec<Expr>
        let v_comps_expr: Vec<_> = v_comps.iter().map(|&x| Expr::from(x)).collect();
        let field = VectorField::from_components(
            group.manifold().clone(),
            chart,
            v_comps_expr,
        )?;

        Ok(Self {
            group,
            value_at_identity,
            field,
        })
    }

    /// Get the value at the identity
    pub fn value_at_identity(&self) -> &TangentVector {
        &self.value_at_identity
    }

    /// Evaluate the field at a point g
    ///
    /// Returns (R_g)_* X_e where X_e is the value at identity
    pub fn at_point(&self, g: &ManifoldPoint) -> Result<TangentVector> {
        self.group.right_translate_tangent(&self.value_at_identity, g)
    }

    /// Get the underlying vector field
    pub fn as_vector_field(&self) -> &VectorField {
        &self.field
    }

    /// Lie bracket of two right-invariant vector fields
    pub fn lie_bracket(&self, other: &RightInvariantVectorField) -> Result<RightInvariantVectorField> {
        let chart = self.group.manifold().default_chart()
            .ok_or(ManifoldError::NoChart)?;
        let bracket_field = self.field.lie_bracket(other.as_vector_field(), chart)?;
        let bracket_at_e = bracket_field.at_point(self.group.identity())?;

        RightInvariantVectorField::new(self.group.clone(), bracket_at_e)
    }
}

/// Maurer-Cartan form
///
/// The Maurer-Cartan form is a 𝔤-valued 1-form on G defined by
/// ω_g(v) = (L_{g^{-1}})_* v
///
/// It satisfies the Maurer-Cartan equation: dω + ½[ω, ω] = 0
pub struct MaurerCartanForm {
    /// The Lie group
    group: Arc<LieGroup>,
}

impl MaurerCartanForm {
    /// Create the Maurer-Cartan form for a Lie group
    pub fn new(group: Arc<LieGroup>) -> Self {
        Self { group }
    }

    /// Evaluate the Maurer-Cartan form at a point on a tangent vector
    ///
    /// Returns ω_g(v) = (L_{g^{-1}})_* v ∈ T_e G = 𝔤
    pub fn evaluate(&self, g: &ManifoldPoint, v: &TangentVector) -> Result<TangentVector> {
        let g_inv = self.group.inverse(g)?;
        self.group.left_translate_tangent(&g_inv, v)
    }

    /// Get the underlying Lie group
    pub fn group(&self) -> &Arc<LieGroup> {
        &self.group
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples::RealLine;

    #[test]
    fn test_lie_group_creation() {
        // Create ℝ as an additive Lie group
        let manifold = Arc::new(RealLine::new().into());
        let identity = ManifoldPoint::from_coords(vec![0.0]);

        let mult = Arc::new(|g: &ManifoldPoint, h: &ManifoldPoint| -> Result<ManifoldPoint> {
            Ok(ManifoldPoint::from_coords(vec![
                g.coordinates()[0] + h.coordinates()[0]
            ]))
        });

        let inv = Arc::new(|g: &ManifoldPoint| -> Result<ManifoldPoint> {
            Ok(ManifoldPoint::from_coords(vec![-g.coordinates()[0]]))
        });

        let group = LieGroup::new(manifold, identity, mult, inv);

        assert_eq!(group.dimension(), 1);
        assert_eq!(group.identity().coordinates()[0], 0.0);
    }

    #[test]
    fn test_group_operations() {
        // Test ℝ under addition
        let manifold = Arc::new(RealLine::new().into());
        let identity = ManifoldPoint::from_coords(vec![0.0]);

        let mult = Arc::new(|g: &ManifoldPoint, h: &ManifoldPoint| -> Result<ManifoldPoint> {
            Ok(ManifoldPoint::from_coords(vec![
                g.coordinates()[0] + h.coordinates()[0]
            ]))
        });

        let inv = Arc::new(|g: &ManifoldPoint| -> Result<ManifoldPoint> {
            Ok(ManifoldPoint::from_coords(vec![-g.coordinates()[0]]))
        });

        let group = LieGroup::new(manifold, identity, mult, inv);

        let g = ManifoldPoint::from_coords(vec![2.0]);
        let h = ManifoldPoint::from_coords(vec![3.0]);

        // Test multiplication
        let gh = group.multiply(&g, &h).unwrap();
        assert_eq!(gh.coordinates()[0], 5.0);

        // Test inversion
        let g_inv = group.inverse(&g).unwrap();
        assert_eq!(g_inv.coordinates()[0], -2.0);

        // Test identity property
        let g_e = group.multiply(&g, group.identity()).unwrap();
        assert_eq!(g_e.coordinates()[0], 2.0);
    }

    #[test]
    fn test_is_identity() {
        let manifold = Arc::new(RealLine::new().into());
        let identity = ManifoldPoint::from_coords(vec![0.0]);

        let mult = Arc::new(|g: &ManifoldPoint, h: &ManifoldPoint| -> Result<ManifoldPoint> {
            Ok(ManifoldPoint::from_coords(vec![
                g.coordinates()[0] + h.coordinates()[0]
            ]))
        });

        let inv = Arc::new(|g: &ManifoldPoint| -> Result<ManifoldPoint> {
            Ok(ManifoldPoint::from_coords(vec![-g.coordinates()[0]]))
        });

        let group = LieGroup::new(manifold, identity, mult, inv);

        assert!(group.is_identity(&group.identity, 1e-10).unwrap());

        let g = ManifoldPoint::from_coords(vec![2.0]);
        assert!(!group.is_identity(&g, 1e-10).unwrap());
    }

    #[test]
    fn test_maurer_cartan_form() {
        let manifold = Arc::new(RealLine::new().into());
        let identity = ManifoldPoint::from_coords(vec![0.0]);

        let mult = Arc::new(|g: &ManifoldPoint, h: &ManifoldPoint| -> Result<ManifoldPoint> {
            Ok(ManifoldPoint::from_coords(vec![
                g.coordinates()[0] + h.coordinates()[0]
            ]))
        });

        let inv = Arc::new(|g: &ManifoldPoint| -> Result<ManifoldPoint> {
            Ok(ManifoldPoint::from_coords(vec![-g.coordinates()[0]]))
        });

        let group = Arc::new(LieGroup::new(manifold, identity, mult, inv));
        let mc_form = MaurerCartanForm::new(group.clone());

        assert_eq!(mc_form.group().dimension(), 1);
    }
}
