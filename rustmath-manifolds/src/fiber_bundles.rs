//! Fiber bundles - bundles with general fiber structure
//!
//! A fiber bundle is a structure (E, B, π, F) where:
//! - E is the total space
//! - B is the base space
//! - π: E → B is the projection map
//! - F is the typical fiber
//!
//! Locally, E looks like B × F

use crate::differentiable::DifferentiableManifold;
use crate::errors::{ManifoldError, Result};
use crate::point::ManifoldPoint;
use crate::chart::Chart;
use crate::lie_group::LieGroup;
use crate::diff_form::DiffForm;
use crate::tangent_space::TangentVector;
use std::sync::Arc;
use std::collections::HashMap;

/// A general fiber bundle π: E → B with fiber F
///
/// Fiber bundles generalize the notion of product spaces E = B × F
/// by allowing the fiber to vary in a "twisted" way over the base.
///
/// # Examples
///
/// - Möbius strip: non-trivial bundle over S¹ with fiber [0,1]
/// - Tangent bundle: TM → M with fiber ℝⁿ
/// - Principal G-bundle: P → M with fiber G
#[derive(Clone)]
pub struct FiberBundle {
    /// Total space E
    total_space: Arc<DifferentiableManifold>,

    /// Base space B
    base_space: Arc<DifferentiableManifold>,

    /// Typical fiber F (may be a manifold or just dimension)
    fiber: Arc<DifferentiableManifold>,

    /// Projection map π: E → B
    projection: Arc<dyn Fn(&ManifoldPoint) -> Result<ManifoldPoint> + Send + Sync>,

    /// Local trivializations (charts on total space compatible with bundle structure)
    /// Each chart maps U × F → π^{-1}(U)
    trivializations: HashMap<String, Trivialization>,
}

/// A local trivialization of a fiber bundle
///
/// Over an open set U ⊂ B, the bundle is trivial: π^{-1}(U) ≅ U × F
#[derive(Clone)]
pub struct Trivialization {
    /// Name of the trivialization
    name: String,

    /// Open set U in base space
    domain_in_base: Vec<ManifoldPoint>,

    /// Chart on the total space
    chart: Chart,

    /// Trivialization map: π^{-1}(U) → U × F
    trivialization_map: Arc<dyn Fn(&ManifoldPoint) -> Result<(ManifoldPoint, ManifoldPoint)> + Send + Sync>,

    /// Inverse: U × F → π^{-1}(U)
    inverse_map: Arc<dyn Fn(&ManifoldPoint, &ManifoldPoint) -> Result<ManifoldPoint> + Send + Sync>,
}

impl FiberBundle {
    /// Create a new fiber bundle
    ///
    /// # Arguments
    ///
    /// * `total_space` - The total space E
    /// * `base_space` - The base space B
    /// * `fiber` - The typical fiber F
    /// * `projection` - The projection map π: E → B
    pub fn new(
        total_space: Arc<DifferentiableManifold>,
        base_space: Arc<DifferentiableManifold>,
        fiber: Arc<DifferentiableManifold>,
        projection: Arc<dyn Fn(&ManifoldPoint) -> Result<ManifoldPoint> + Send + Sync>,
    ) -> Self {
        Self {
            total_space,
            base_space,
            fiber,
            projection,
            trivializations: HashMap::new(),
        }
    }

    /// Get the total space
    pub fn total_space(&self) -> &Arc<DifferentiableManifold> {
        &self.total_space
    }

    /// Get the base space
    pub fn base_space(&self) -> &Arc<DifferentiableManifold> {
        &self.base_space
    }

    /// Get the fiber
    pub fn fiber(&self) -> &Arc<DifferentiableManifold> {
        &self.fiber
    }

    /// Project a point from total space to base space
    pub fn project(&self, point: &ManifoldPoint) -> Result<ManifoldPoint> {
        (self.projection)(point)
    }

    /// Get the fiber over a point in the base space
    ///
    /// Returns π^{-1}(p) = { e ∈ E : π(e) = p }
    pub fn fiber_over(&self, base_point: &ManifoldPoint) -> Result<Fiber> {
        Fiber::new(self.clone(), base_point.clone())
    }

    /// Add a local trivialization
    pub fn add_trivialization(&mut self, name: String, triv: Trivialization) {
        self.trivializations.insert(name, triv);
    }

    /// Check if the bundle is trivial (globally E ≅ B × F)
    pub fn is_trivial(&self) -> bool {
        // A bundle is trivial if it admits a global section
        // For now, we check if there's a trivialization covering the entire base
        // In practice, this is difficult to determine algorithmically
        false
    }

    /// Get dimension of total space
    pub fn total_dimension(&self) -> usize {
        self.total_space.dimension()
    }

    /// Get dimension of base space
    pub fn base_dimension(&self) -> usize {
        self.base_space.dimension()
    }

    /// Get dimension of fiber
    pub fn fiber_dimension(&self) -> usize {
        self.fiber.dimension()
    }
}

/// A fiber π^{-1}(p) over a point p in the base space
pub struct Fiber {
    /// The bundle this fiber belongs to
    bundle: FiberBundle,

    /// The base point p
    base_point: ManifoldPoint,
}

impl Fiber {
    /// Create a fiber over a base point
    pub fn new(bundle: FiberBundle, base_point: ManifoldPoint) -> Result<Self> {
        // Verify the point is in the base space
        // (In practice, we'd check this more carefully)

        Ok(Self {
            bundle,
            base_point,
        })
    }

    /// Get the base point
    pub fn base_point(&self) -> &ManifoldPoint {
        &self.base_point
    }

    /// Get the bundle
    pub fn bundle(&self) -> &FiberBundle {
        &self.bundle
    }

    /// Check if a point in total space belongs to this fiber
    pub fn contains(&self, point: &ManifoldPoint) -> Result<bool> {
        let projected = self.bundle.project(point)?;

        // Check if projected point equals base point
        // (This needs a proper equality check on manifold points)
        Ok(projected.coordinates() == self.base_point.coordinates())
    }
}

/// A principal G-bundle
///
/// A principal bundle is a fiber bundle where:
/// - The fiber is a Lie group G
/// - There is a free right action of G on E
/// - The projection is G-invariant
/// - Locally trivial in a G-equivariant way
///
/// # Examples
///
/// - Frame bundle: Bundle of linear frames on a manifold
/// - SO(n)-bundle over orientable manifolds
/// - U(1)-bundle (circle bundles, related to line bundles)
#[derive(Clone)]
pub struct PrincipalBundle {
    /// The underlying fiber bundle structure
    bundle: FiberBundle,

    /// The structure group G
    structure_group: Arc<LieGroup>,

    /// Right action: E × G → E
    /// r_g(e) = e · g
    right_action: Arc<dyn Fn(&ManifoldPoint, &ManifoldPoint) -> Result<ManifoldPoint> + Send + Sync>,
}

impl PrincipalBundle {
    /// Create a new principal bundle
    ///
    /// # Arguments
    ///
    /// * `bundle` - The underlying fiber bundle
    /// * `structure_group` - The Lie group G
    /// * `right_action` - Right action E × G → E
    pub fn new(
        bundle: FiberBundle,
        structure_group: Arc<LieGroup>,
        right_action: Arc<dyn Fn(&ManifoldPoint, &ManifoldPoint) -> Result<ManifoldPoint> + Send + Sync>,
    ) -> Result<Self> {
        // Verify that fiber has same dimension as structure group
        if bundle.fiber_dimension() != structure_group.dimension() {
            return Err(ManifoldError::InvalidDimension(
                format!("Fiber dimension {} must match structure group dimension {}",
                        bundle.fiber_dimension(), structure_group.dimension())
            ));
        }

        Ok(Self {
            bundle,
            structure_group,
            right_action,
        })
    }

    /// Get the underlying bundle
    pub fn bundle(&self) -> &FiberBundle {
        &self.bundle
    }

    /// Get the structure group
    pub fn structure_group(&self) -> &Arc<LieGroup> {
        &self.structure_group
    }

    /// Right action: e · g
    pub fn act(&self, e: &ManifoldPoint, g: &ManifoldPoint) -> Result<ManifoldPoint> {
        (self.right_action)(e, g)
    }

    /// Project to base space
    pub fn project(&self, e: &ManifoldPoint) -> Result<ManifoldPoint> {
        self.bundle.project(e)
    }

    /// Check if action is free (e·g = e implies g = identity)
    pub fn is_free_action(&self, e: &ManifoldPoint, g: &ManifoldPoint) -> Result<bool> {
        let eg = self.act(e, g)?;

        // If e·g = e, then g should be the identity
        if eg.coordinates() == e.coordinates() {
            Ok(self.structure_group.is_identity(g, 1e-10)?)
        } else {
            Ok(true) // Action moves the point, so it's free
        }
    }
}

/// Associated bundle E ×_G F
///
/// Given a principal G-bundle P → M and a left G-action on F,
/// the associated bundle is E = (P × F) / G
///
/// # Examples
///
/// - Vector bundles from frame bundles
/// - Tangent bundle from frame bundle with GL(n) action on ℝⁿ
#[derive(Clone)]
pub struct AssociatedBundle {
    /// The principal bundle P
    principal_bundle: Arc<PrincipalBundle>,

    /// The fiber F
    fiber: Arc<DifferentiableManifold>,

    /// Left action of G on F: G × F → F
    left_action: Arc<dyn Fn(&ManifoldPoint, &ManifoldPoint) -> Result<ManifoldPoint> + Send + Sync>,
}

impl AssociatedBundle {
    /// Create an associated bundle P ×_G F
    ///
    /// # Arguments
    ///
    /// * `principal_bundle` - The principal G-bundle P → M
    /// * `fiber` - The fiber F
    /// * `left_action` - Left action of G on F
    pub fn new(
        principal_bundle: Arc<PrincipalBundle>,
        fiber: Arc<DifferentiableManifold>,
        left_action: Arc<dyn Fn(&ManifoldPoint, &ManifoldPoint) -> Result<ManifoldPoint> + Send + Sync>,
    ) -> Self {
        Self {
            principal_bundle,
            fiber,
            left_action,
        }
    }

    /// Get the principal bundle
    pub fn principal_bundle(&self) -> &Arc<PrincipalBundle> {
        &self.principal_bundle
    }

    /// Get the fiber
    pub fn fiber(&self) -> &Arc<DifferentiableManifold> {
        &self.fiber
    }

    /// Left action of G on F
    pub fn act_on_fiber(&self, g: &ManifoldPoint, f: &ManifoldPoint) -> Result<ManifoldPoint> {
        (self.left_action)(g, f)
    }

    /// Project to base space
    pub fn base_space(&self) -> &Arc<DifferentiableManifold> {
        self.principal_bundle.bundle().base_space()
    }
}

/// Connection form on a principal bundle
///
/// A connection is a 𝔤-valued 1-form ω on P that:
/// 1. Is G-equivariant: (r_g)^* ω = Ad_{g^{-1}} ω
/// 2. Reproduces generators: ω(X^*) = X for fundamental vector fields
///
/// The connection form defines parallel transport and covariant derivatives
#[derive(Clone)]
pub struct ConnectionForm {
    /// The principal bundle
    principal_bundle: Arc<PrincipalBundle>,

    /// The connection form as a 1-form valued in the Lie algebra
    /// For each point p ∈ P and tangent vector v ∈ T_p P,
    /// ω_p(v) ∈ 𝔤 (Lie algebra of structure group)
    form: Arc<dyn Fn(&ManifoldPoint, &TangentVector) -> Result<Vec<f64>> + Send + Sync>,
}

impl ConnectionForm {
    /// Create a connection form
    ///
    /// # Arguments
    ///
    /// * `principal_bundle` - The principal G-bundle
    /// * `form` - The 𝔤-valued 1-form
    pub fn new(
        principal_bundle: Arc<PrincipalBundle>,
        form: Arc<dyn Fn(&ManifoldPoint, &TangentVector) -> Result<Vec<f64>> + Send + Sync>,
    ) -> Self {
        Self {
            principal_bundle,
            form,
        }
    }

    /// Evaluate the connection form at a point on a tangent vector
    ///
    /// Returns ω_p(v) ∈ 𝔤
    pub fn evaluate(&self, p: &ManifoldPoint, v: &TangentVector) -> Result<Vec<f64>> {
        (self.form)(p, v)
    }

    /// Get the principal bundle
    pub fn principal_bundle(&self) -> &Arc<PrincipalBundle> {
        &self.principal_bundle
    }

    /// Horizontal subspace at a point
    ///
    /// The horizontal subspace H_p ⊂ T_p P is the kernel of ω_p
    /// It's complementary to the vertical subspace (tangent to fibers)
    pub fn horizontal_projection(&self, p: &ManifoldPoint, v: &TangentVector) -> Result<TangentVector> {
        // For a full implementation, we'd project v onto ker(ω_p)
        // This requires computing the vertical component and subtracting

        // Placeholder: return the vector (should implement proper projection)
        Ok(v.clone())
    }

    /// Curvature form: Ω = dω + ½[ω, ω]
    ///
    /// The curvature measures the failure of the connection to be flat
    pub fn curvature(&self) -> Result<CurvatureForm> {
        CurvatureForm::from_connection(self.clone())
    }
}

/// Curvature form on a principal bundle
///
/// The curvature is a 𝔤-valued 2-form Ω = dω + ½[ω, ω]
/// It measures the holonomy (failure of parallel transport to close)
///
/// # Properties
///
/// - Ω is horizontal: Ω(X^*, ·) = 0 for fundamental vector fields
/// - Bianchi identity: DΩ = dΩ + [ω, Ω] = 0
/// - Ω = 0 ⟺ connection is flat
#[derive(Clone)]
pub struct CurvatureForm {
    /// The connection form
    connection: ConnectionForm,

    /// The curvature 2-form
    /// For points p ∈ P and tangent vectors v, w ∈ T_p P,
    /// Ω_p(v, w) ∈ 𝔤
    form: Arc<dyn Fn(&ManifoldPoint, &TangentVector, &TangentVector) -> Result<Vec<f64>> + Send + Sync>,
}

impl CurvatureForm {
    /// Create curvature form from a connection
    pub fn from_connection(connection: ConnectionForm) -> Result<Self> {
        // In a full implementation, we'd compute dω + ½[ω, ω]
        // For now, create a placeholder

        let form = Arc::new(
            move |_p: &ManifoldPoint, _v: &TangentVector, _w: &TangentVector| -> Result<Vec<f64>> {
                // Placeholder: return zero curvature
                let dim = connection.principal_bundle().structure_group().dimension();
                Ok(vec![0.0; dim])
            }
        );

        Ok(Self {
            connection,
            form,
        })
    }

    /// Evaluate the curvature form
    ///
    /// Returns Ω_p(v, w) ∈ 𝔤
    pub fn evaluate(&self, p: &ManifoldPoint, v: &TangentVector, w: &TangentVector) -> Result<Vec<f64>> {
        (self.form)(p, v, w)
    }

    /// Get the connection
    pub fn connection(&self) -> &ConnectionForm {
        &self.connection
    }

    /// Check if the connection is flat (Ω = 0 everywhere)
    pub fn is_flat(&self, tolerance: f64) -> Result<bool> {
        // In practice, we'd check Ω at various points
        // For now, return a placeholder
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples::{RealLine, EuclideanSpace};

    #[test]
    fn test_fiber_bundle_creation() {
        // Create a trivial bundle ℝ × ℝ → ℝ
        let total = Arc::new(EuclideanSpace::new(2).into());
        let base = Arc::new(RealLine::new().into());
        let fiber = Arc::new(RealLine::new().into());

        let proj = Arc::new(|p: &ManifoldPoint| -> Result<ManifoldPoint> {
            Ok(ManifoldPoint::from_coords(vec![p.coordinates()[0]]))
        });

        let bundle = FiberBundle::new(total, base, fiber, proj);

        assert_eq!(bundle.total_dimension(), 2);
        assert_eq!(bundle.base_dimension(), 1);
        assert_eq!(bundle.fiber_dimension(), 1);
    }

    #[test]
    fn test_projection() {
        let total = Arc::new(EuclideanSpace::new(2).into());
        let base = Arc::new(RealLine::new().into());
        let fiber = Arc::new(RealLine::new().into());

        let proj = Arc::new(|p: &ManifoldPoint| -> Result<ManifoldPoint> {
            Ok(ManifoldPoint::from_coords(vec![p.coordinates()[0]]))
        });

        let bundle = FiberBundle::new(total, base, fiber, proj);

        let point = ManifoldPoint::from_coords(vec![3.0, 5.0]);
        let projected = bundle.project(&point).unwrap();

        assert_eq!(projected.coordinates()[0], 3.0);
        assert_eq!(projected.coordinates().len(), 1);
    }

    #[test]
    fn test_fiber_over_point() {
        let total = Arc::new(EuclideanSpace::new(2).into());
        let base = Arc::new(RealLine::new().into());
        let fiber_space = Arc::new(RealLine::new().into());

        let proj = Arc::new(|p: &ManifoldPoint| -> Result<ManifoldPoint> {
            Ok(ManifoldPoint::from_coords(vec![p.coordinates()[0]]))
        });

        let bundle = FiberBundle::new(total, base, fiber_space, proj);

        let base_point = ManifoldPoint::from_coords(vec![2.0]);
        let fiber = bundle.fiber_over(&base_point).unwrap();

        assert_eq!(fiber.base_point().coordinates()[0], 2.0);

        // Point (2.0, 3.0) should be in the fiber over 2.0
        let point_in_fiber = ManifoldPoint::from_coords(vec![2.0, 3.0]);
        assert!(fiber.contains(&point_in_fiber).unwrap());

        // Point (1.0, 3.0) should NOT be in the fiber over 2.0
        let point_not_in_fiber = ManifoldPoint::from_coords(vec![1.0, 3.0]);
        assert!(!fiber.contains(&point_not_in_fiber).unwrap());
    }
}
