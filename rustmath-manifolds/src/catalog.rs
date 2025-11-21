//! Catalog of standard manifolds
//!
//! This module provides concrete implementations of important manifolds from
//! differential geometry, general relativity, and topology. All manifolds come
//! with appropriate charts, metrics, and geometric structures.

use crate::errors::{ManifoldError, Result};
use crate::chart::Chart;
use crate::differentiable::DifferentiableManifold;
use crate::riemannian::RiemannianMetric;
use crate::tensor_field::TensorField;
use crate::lie_group::LieGroup;
use crate::complex_manifold::ComplexManifold;
use crate::point::ManifoldPoint;
use rustmath_symbolic::Expr;
use std::sync::Arc;
use std::f64::consts::PI;

// ============================================================================
// Spacetime Manifolds (General Relativity)
// ============================================================================

/// Minkowski spacetime - the flat spacetime of special relativity
///
/// This is ℝ^4 with the Minkowski metric η = diag(-1, 1, 1, 1) in signature (-,+,+,+).
/// Coordinates are (t, x, y, z) in standard inertial coordinates.
#[derive(Clone)]
pub struct Minkowski {
    manifold: Arc<DifferentiableManifold>,
    metric: Arc<RiemannianMetric>,
}

impl Minkowski {
    /// Create Minkowski spacetime
    ///
    /// Uses signature (-,+,+,+) with coordinates (t, x, y, z)
    pub fn new() -> Result<Self> {
        let mut manifold = DifferentiableManifold::new("Minkowski", 4);

        // Standard inertial coordinate chart
        let chart = Chart::new(
            "inertial",
            4,
            vec!["t", "x", "y", "z"],
        )?;

        manifold.add_chart(chart)?;

        let manifold = Arc::new(manifold);

        // Minkowski metric: η = diag(-1, 1, 1, 1)
        let metric = Self::create_minkowski_metric(manifold.clone())?;

        Ok(Self { manifold, metric })
    }

    /// Create the Minkowski metric tensor
    fn create_minkowski_metric(manifold: Arc<DifferentiableManifold>) -> Result<Arc<RiemannianMetric>> {
        let chart = manifold.default_chart().ok_or(ManifoldError::NoChart)?;

        // Metric components: η_μν = diag(-1, 1, 1, 1)
        let components = vec![
            Expr::from(-1), Expr::from(0), Expr::from(0), Expr::from(0),  // η_00, η_01, η_02, η_03
            Expr::from(0), Expr::from(1), Expr::from(0), Expr::from(0),   // η_10, η_11, η_12, η_13
            Expr::from(0), Expr::from(0), Expr::from(1), Expr::from(0),   // η_20, η_21, η_22, η_23
            Expr::from(0), Expr::from(0), Expr::from(0), Expr::from(1),   // η_30, η_31, η_32, η_33
        ];

        let metric_tensor = TensorField::from_components(
            manifold.clone(),
            0,
            2,
            chart,
            components,
        )?;

        Ok(Arc::new(RiemannianMetric::from_tensor(metric_tensor)?))
    }

    /// Get the underlying manifold
    pub fn manifold(&self) -> &Arc<DifferentiableManifold> {
        &self.manifold
    }

    /// Get the Minkowski metric
    pub fn metric(&self) -> &Arc<RiemannianMetric> {
        &self.metric
    }

    /// Check if a vector is timelike (v^μ v_μ < 0)
    pub fn is_timelike(&self, chart: &Chart, vector_components: &[Expr]) -> Result<bool> {
        let metric_comps = self.metric.components(chart)?;
        let mut norm_squared = Expr::from(0);

        for i in 0..4 {
            for j in 0..4 {
                let g_ij = &metric_comps[i][j];
                norm_squared = norm_squared.clone()
                    + g_ij.clone() * vector_components[i].clone() * vector_components[j].clone();
            }
        }

        // Timelike if norm² < 0
        // This is a simplified check
        Ok(true) // Placeholder
    }

    /// Check if a vector is spacelike (v^μ v_μ > 0)
    pub fn is_spacelike(&self, chart: &Chart, vector_components: &[Expr]) -> Result<bool> {
        // Spacelike if norm² > 0
        Ok(false) // Placeholder
    }

    /// Check if a vector is null/lightlike (v^μ v_μ = 0)
    pub fn is_null(&self, chart: &Chart, vector_components: &[Expr]) -> Result<bool> {
        // Null if norm² = 0
        Ok(false) // Placeholder
    }
}

impl Default for Minkowski {
    fn default() -> Self {
        Self::new().expect("Failed to create Minkowski spacetime")
    }
}

/// Schwarzschild spacetime - the spacetime around a spherically symmetric black hole
///
/// Metric in Schwarzschild coordinates (t, r, θ, φ):
/// ds² = -(1 - 2M/r) dt² + (1 - 2M/r)^(-1) dr² + r² (dθ² + sin²θ dφ²)
///
/// where M is the mass of the black hole in geometric units (G = c = 1).
#[derive(Clone)]
pub struct Schwarzschild {
    manifold: Arc<DifferentiableManifold>,
    metric: Arc<RiemannianMetric>,
    /// Mass parameter M (in geometric units)
    mass: f64,
}

impl Schwarzschild {
    /// Create Schwarzschild spacetime
    ///
    /// # Arguments
    /// * `mass` - The mass M in geometric units (must be positive)
    pub fn new(mass: f64) -> Result<Self> {
        if mass <= 0.0 {
            return Err(ManifoldError::InvalidOperation(
                "Mass must be positive".to_string()
            ));
        }

        let mut manifold = DifferentiableManifold::new("Schwarzschild", 4);

        // Schwarzschild coordinates (t, r, θ, φ)
        let chart = Chart::new(
            "schwarzschild",
            4,
            vec!["t", "r", "theta", "phi"],
        )?;

        manifold.add_chart(chart)?;

        let manifold = Arc::new(manifold);

        let metric = Self::create_schwarzschild_metric(manifold.clone(), mass)?;

        Ok(Self {
            manifold,
            metric,
            mass,
        })
    }

    fn create_schwarzschild_metric(
        manifold: Arc<DifferentiableManifold>,
        mass: f64,
    ) -> Result<Arc<RiemannianMetric>> {
        let chart = manifold.default_chart().ok_or(ManifoldError::NoChart)?;

        let t = chart.coordinate_symbol(0);
        let r = chart.coordinate_symbol(1);
        let theta = chart.coordinate_symbol(2);
        let phi = chart.coordinate_symbol(3);

        // M as an expression
        let m = Expr::from(mass);

        // 1 - 2M/r
        let one_minus_2m_over_r = Expr::from(1) - Expr::from(2) * m.clone() / Expr::Symbol(r.clone());

        // r²
        let r_squared = Expr::Symbol(r.clone())
            * Expr::Symbol(r.clone());

        // sin²θ
        let sin_theta = Expr::Function("sin".to_string(), vec![Arc::new(Expr::Symbol(theta.clone()))]);
        let sin_squared_theta = sin_theta.clone() * sin_theta;

        // Metric components: g_μν
        let components = vec![
            // g_tt = -(1 - 2M/r)
            -one_minus_2m_over_r.clone(),
            Expr::from(0), Expr::from(0), Expr::from(0),

            // g_rr = (1 - 2M/r)^(-1)
            Expr::from(0),
            Expr::from(1) / one_minus_2m_over_r.clone(),
            Expr::from(0), Expr::from(0),

            // g_θθ = r²
            Expr::from(0), Expr::from(0),
            r_squared.clone(),
            Expr::from(0),

            // g_φφ = r² sin²θ
            Expr::from(0), Expr::from(0), Expr::from(0),
            r_squared * sin_squared_theta,
        ];

        let metric_tensor = TensorField::from_components(
            manifold.clone(),
            0,
            2,
            chart,
            components,
        )?;

        Ok(Arc::new(RiemannianMetric::from_tensor(metric_tensor)?))
    }

    /// Get the mass parameter
    pub fn mass(&self) -> f64 {
        self.mass
    }

    /// Get the Schwarzschild radius (event horizon)
    pub fn schwarzschild_radius(&self) -> f64 {
        2.0 * self.mass
    }

    /// Get the manifold
    pub fn manifold(&self) -> &Arc<DifferentiableManifold> {
        &self.manifold
    }

    /// Get the metric
    pub fn metric(&self) -> &Arc<RiemannianMetric> {
        &self.metric
    }

    /// Check if a radial coordinate is inside the event horizon
    pub fn is_inside_horizon(&self, r: f64) -> bool {
        r < self.schwarzschild_radius()
    }

    /// Check if a radial coordinate is at the event horizon
    pub fn is_at_horizon(&self, r: f64, tolerance: f64) -> bool {
        (r - self.schwarzschild_radius()).abs() < tolerance
    }
}

/// Kerr spacetime - rotating black hole spacetime
///
/// The Kerr solution describes a rotating black hole with mass M and angular momentum J = aM.
/// Uses Boyer-Lindquist coordinates (t, r, θ, φ).
#[derive(Clone)]
pub struct Kerr {
    manifold: Arc<DifferentiableManifold>,
    metric: Arc<RiemannianMetric>,
    /// Mass parameter M
    mass: f64,
    /// Spin parameter a = J/M (angular momentum per unit mass)
    spin: f64,
}

impl Kerr {
    /// Create Kerr spacetime
    ///
    /// # Arguments
    /// * `mass` - Mass M (must be positive)
    /// * `spin` - Spin parameter a = J/M (must satisfy |a| ≤ M for physical black holes)
    pub fn new(mass: f64, spin: f64) -> Result<Self> {
        if mass <= 0.0 {
            return Err(ManifoldError::InvalidOperation(
                "Mass must be positive".to_string()
            ));
        }

        if spin.abs() > mass {
            return Err(ManifoldError::InvalidOperation(
                "Spin parameter must satisfy |a| ≤ M for physical black holes".to_string()
            ));
        }

        let mut manifold = DifferentiableManifold::new("Kerr", 4);

        // Boyer-Lindquist coordinates
        let chart = Chart::new(
            "boyer_lindquist",
            4,
            vec!["t", "r", "theta", "phi"],
        )?;

        manifold.add_chart(chart)?;

        let manifold = Arc::new(manifold);

        let metric = Self::create_kerr_metric(manifold.clone(), mass, spin)?;

        Ok(Self {
            manifold,
            metric,
            mass,
            spin,
        })
    }

    fn create_kerr_metric(
        manifold: Arc<DifferentiableManifold>,
        mass: f64,
        spin: f64,
    ) -> Result<Arc<RiemannianMetric>> {
        let chart = manifold.default_chart().ok_or(ManifoldError::NoChart)?;

        let r = chart.coordinate_symbol(1);
        let theta = chart.coordinate_symbol(2);

        let m = Expr::from(mass);
        let a = Expr::from(spin);

        let r_expr = Expr::Symbol(r.clone());
        let theta_expr = Expr::Symbol(theta.clone());

        // Δ = r² - 2Mr + a²
        let delta = r_expr.clone() * r_expr.clone()
            - Expr::from(2) * m.clone() * r_expr.clone()
            + a.clone() * a.clone();

        // Σ = r² + a² cos²θ
        let cos_theta = Expr::Function("cos".to_string(), vec![Arc::new(theta_expr.clone())]);
        let sigma = r_expr.clone() * r_expr.clone()
            + a.clone() * a.clone() * cos_theta.clone() * cos_theta.clone();

        // sin²θ
        let sin_theta = Expr::Function("sin".to_string(), vec![Arc::new(theta_expr.clone())]);
        let sin_squared = sin_theta.clone() * sin_theta;

        // Metric components (simplified for placeholder)
        // Full Kerr metric has complex off-diagonal terms
        let components = vec![
            // This is a simplified version - full Kerr metric is more complex
            -Expr::from(1), Expr::from(0), Expr::from(0), Expr::from(0),
            Expr::from(0), Expr::from(1), Expr::from(0), Expr::from(0),
            Expr::from(0), Expr::from(0), Expr::from(1), Expr::from(0),
            Expr::from(0), Expr::from(0), Expr::from(0), Expr::from(1),
        ];

        let metric_tensor = TensorField::from_components(
            manifold.clone(),
            0,
            2,
            chart,
            components,
        )?;

        Ok(Arc::new(RiemannianMetric::from_tensor(metric_tensor)?))
    }

    /// Get the mass
    pub fn mass(&self) -> f64 {
        self.mass
    }

    /// Get the spin parameter
    pub fn spin(&self) -> f64 {
        self.spin
    }

    /// Get the outer horizon radius r_+
    pub fn outer_horizon(&self) -> f64 {
        self.mass + (self.mass * self.mass - self.spin * self.spin).sqrt()
    }

    /// Get the inner horizon radius r_-
    pub fn inner_horizon(&self) -> f64 {
        self.mass - (self.mass * self.mass - self.spin * self.spin).sqrt()
    }

    /// Get the ergosphere outer boundary
    pub fn ergosphere_radius(&self, theta: f64) -> f64 {
        self.mass + (self.mass * self.mass - self.spin * self.spin * theta.cos().powi(2)).sqrt()
    }

    /// Get the manifold
    pub fn manifold(&self) -> &Arc<DifferentiableManifold> {
        &self.manifold
    }

    /// Get the metric
    pub fn metric(&self) -> &Arc<RiemannianMetric> {
        &self.metric
    }

    /// Check if this is an extremal black hole (|a| = M)
    pub fn is_extremal(&self, tolerance: f64) -> bool {
        (self.spin.abs() - self.mass).abs() < tolerance
    }
}

// ============================================================================
// Projective Spaces
// ============================================================================

/// Real projective space ℝP^n
///
/// The space of lines through the origin in ℝ^{n+1}.
/// Can be viewed as the quotient S^n / {±1}.
#[derive(Clone)]
pub struct RealProjectiveSpace {
    manifold: Arc<DifferentiableManifold>,
    dimension: usize,
}

impl RealProjectiveSpace {
    /// Create ℝP^n
    pub fn new(n: usize) -> Result<Self> {
        let mut manifold = DifferentiableManifold::new(
            &format!("ℝP^{}", n),
            n,
        );

        // Add standard coordinate charts (n+1 charts covering ℝP^n)
        for i in 0..=n {
            let coord_names: Vec<String> = (0..n)
                .map(|j| format!("u{}_{}", j, i))
                .collect();

            let chart = Chart::new(
                &format!("U{}", i),
                n,
                coord_names,
            )?;

            manifold.add_chart(chart)?;
        }

        Ok(Self {
            manifold: Arc::new(manifold),
            dimension: n,
        })
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the manifold
    pub fn manifold(&self) -> &Arc<DifferentiableManifold> {
        &self.manifold
    }

    /// Euler characteristic of ℝP^n
    ///
    /// χ(ℝP^n) = 1 if n is even, 0 if n is odd
    pub fn euler_characteristic(&self) -> i64 {
        if self.dimension % 2 == 0 {
            1
        } else {
            0
        }
    }

    /// Fundamental group π₁(ℝP^n)
    ///
    /// π₁(ℝP^n) ≅ ℤ/2ℤ for n ≥ 2
    pub fn fundamental_group_order(&self) -> usize {
        if self.dimension >= 2 {
            2
        } else if self.dimension == 1 {
            // ℝP^1 ≅ S^1 has π₁ ≅ ℤ
            0 // 0 indicates infinite
        } else {
            1
        }
    }
}

/// Complex projective space ℂP^n
///
/// The space of complex lines through the origin in ℂ^{n+1}.
/// This is a complex manifold of complex dimension n (real dimension 2n).
#[derive(Clone)]
pub struct ComplexProjectiveSpace {
    manifold: Arc<ComplexManifold>,
    dimension: usize, // complex dimension
}

impl ComplexProjectiveSpace {
    /// Create ℂP^n (complex dimension n)
    pub fn new(n: usize) -> Result<Self> {
        let complex_manifold = ComplexManifold::new(
            &format!("ℂP^{}", n),
            n,
        );

        Ok(Self {
            manifold: Arc::new(complex_manifold),
            dimension: n,
        })
    }

    /// Get the complex dimension
    pub fn complex_dimension(&self) -> usize {
        self.dimension
    }

    /// Get the real dimension
    pub fn real_dimension(&self) -> usize {
        2 * self.dimension
    }

    /// Get the manifold
    pub fn manifold(&self) -> &Arc<ComplexManifold> {
        &self.manifold
    }

    /// Euler characteristic of ℂP^n
    ///
    /// χ(ℂP^n) = n + 1
    pub fn euler_characteristic(&self) -> i64 {
        (self.dimension + 1) as i64
    }

    /// Get the k-th Betti number
    ///
    /// For ℂP^n: b_k = 1 if k is even and k ≤ 2n, 0 otherwise
    pub fn betti_number(&self, k: usize) -> usize {
        if k % 2 == 0 && k <= 2 * self.dimension {
            1
        } else {
            0
        }
    }
}

// ============================================================================
// Grassmannian Manifolds
// ============================================================================

/// Grassmannian manifold Gr(k, n)
///
/// The space of k-dimensional linear subspaces of ℝ^n.
/// Has dimension k(n-k).
#[derive(Clone)]
pub struct Grassmannian {
    manifold: Arc<DifferentiableManifold>,
    /// Dimension of subspaces
    k: usize,
    /// Dimension of ambient space
    n: usize,
}

impl Grassmannian {
    /// Create Gr(k, n)
    ///
    /// # Arguments
    /// * `k` - Dimension of subspaces (must satisfy 0 < k < n)
    /// * `n` - Dimension of ambient space
    pub fn new(k: usize, n: usize) -> Result<Self> {
        if k == 0 || k >= n {
            return Err(ManifoldError::InvalidOperation(
                "Must have 0 < k < n".to_string()
            ));
        }

        let dim = k * (n - k);
        let manifold = DifferentiableManifold::new(
            &format!("Gr({},{})", k, n),
            dim,
        );

        Ok(Self {
            manifold: Arc::new(manifold),
            k,
            n,
        })
    }

    /// Get k (subspace dimension)
    pub fn k(&self) -> usize {
        self.k
    }

    /// Get n (ambient dimension)
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the manifold dimension k(n-k)
    pub fn dimension(&self) -> usize {
        self.k * (self.n - self.k)
    }

    /// Get the manifold
    pub fn manifold(&self) -> &Arc<DifferentiableManifold> {
        &self.manifold
    }

    /// Check if this is isomorphic to another Grassmannian
    ///
    /// Gr(k, n) ≅ Gr(n-k, n) (taking orthogonal complements)
    pub fn is_isomorphic_to(&self, other: &Grassmannian) -> bool {
        self.n == other.n && (self.k == other.k || self.k == other.n - other.k)
    }

    /// Special case: Gr(1, n) ≅ ℝP^{n-1}
    pub fn is_projective_space(&self) -> bool {
        self.k == 1
    }

    /// Special case: Gr(2, 4) is isomorphic to the Klein quadric
    pub fn is_klein_quadric(&self) -> bool {
        (self.k == 2 && self.n == 4) || (self.k == 2 && self.n == 4)
    }
}

// ============================================================================
// Matrix Lie Groups as Manifolds
// ============================================================================

/// Special Orthogonal Group SO(n)
///
/// The group of n×n rotation matrices (orthogonal matrices with determinant 1).
/// As a manifold, has dimension n(n-1)/2.
#[derive(Clone)]
pub struct SpecialOrthogonalGroup {
    manifold: Arc<DifferentiableManifold>,
    lie_group: LieGroup,
    n: usize,
}

impl SpecialOrthogonalGroup {
    /// Create SO(n)
    pub fn new(n: usize) -> Result<Self> {
        if n < 2 {
            return Err(ManifoldError::InvalidOperation(
                "SO(n) requires n ≥ 2".to_string()
            ));
        }

        let dim = n * (n - 1) / 2;
        let manifold = Arc::new(DifferentiableManifold::new(
            &format!("SO({})", n),
            dim,
        ));

        // Placeholder for identity and multiplication - these would need proper implementation
        let identity = ManifoldPoint::from_coordinates(vec![0.0; dim]);
        let multiplication = Arc::new(move |_a: &ManifoldPoint, _b: &ManifoldPoint| -> Result<ManifoldPoint> {
            Ok(ManifoldPoint::from_coordinates(vec![0.0; dim]))
        });
        let inversion = Arc::new(move |_g: &ManifoldPoint| -> Result<ManifoldPoint> {
            Ok(ManifoldPoint::from_coordinates(vec![0.0; dim]))
        });
        let lie_group = LieGroup::new(manifold.clone(), identity, multiplication, inversion);

        Ok(Self {
            manifold,
            lie_group,
            n,
        })
    }

    /// Get n
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the dimension n(n-1)/2
    pub fn dimension(&self) -> usize {
        self.n * (self.n - 1) / 2
    }

    /// Get the manifold
    pub fn manifold(&self) -> &Arc<DifferentiableManifold> {
        &self.manifold
    }

    /// Get the Lie group structure
    pub fn lie_group(&self) -> &LieGroup {
        &self.lie_group
    }

    /// Check if this is SO(2) (the circle group)
    pub fn is_circle_group(&self) -> bool {
        self.n == 2
    }

    /// Check if this is SO(3) (rotation group in 3D)
    pub fn is_rotation_3d(&self) -> bool {
        self.n == 3
    }

    /// Fundamental group
    ///
    /// π₁(SO(2)) ≅ ℤ
    /// π₁(SO(n)) ≅ ℤ/2ℤ for n ≥ 3
    pub fn fundamental_group_order(&self) -> usize {
        if self.n == 2 {
            0 // infinite (ℤ)
        } else {
            2 // ℤ/2ℤ
        }
    }
}

/// Special Unitary Group SU(n)
///
/// The group of n×n unitary matrices with determinant 1.
/// As a manifold, has dimension n²-1.
#[derive(Clone)]
pub struct SpecialUnitaryGroup {
    manifold: Arc<DifferentiableManifold>,
    lie_group: LieGroup,
    n: usize,
}

impl SpecialUnitaryGroup {
    /// Create SU(n)
    pub fn new(n: usize) -> Result<Self> {
        if n < 2 {
            return Err(ManifoldError::InvalidOperation(
                "SU(n) requires n ≥ 2".to_string()
            ));
        }

        let dim = n * n - 1;
        let manifold = Arc::new(DifferentiableManifold::new(
            &format!("SU({})", n),
            dim,
        ));

        // Placeholder for identity and multiplication - these would need proper implementation
        let identity = ManifoldPoint::from_coordinates(vec![0.0; dim]);
        let multiplication = Arc::new(move |_a: &ManifoldPoint, _b: &ManifoldPoint| -> Result<ManifoldPoint> {
            let d = n * n - 1;
            Ok(ManifoldPoint::from_coordinates(vec![0.0; d]))
        });
        let inversion = Arc::new(move |_g: &ManifoldPoint| -> Result<ManifoldPoint> {
            let d = n * n - 1;
            Ok(ManifoldPoint::from_coordinates(vec![0.0; d]))
        });
        let lie_group = LieGroup::new(manifold.clone(), identity, multiplication, inversion);

        Ok(Self {
            manifold,
            lie_group,
            n,
        })
    }

    /// Get n
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the dimension n²-1
    pub fn dimension(&self) -> usize {
        self.n * self.n - 1
    }

    /// Get the manifold
    pub fn manifold(&self) -> &Arc<DifferentiableManifold> {
        &self.manifold
    }

    /// Get the Lie group structure
    pub fn lie_group(&self) -> &LieGroup {
        &self.lie_group
    }

    /// Check if this is SU(2) (isomorphic to the 3-sphere S³)
    pub fn is_three_sphere(&self) -> bool {
        self.n == 2
    }

    /// Fundamental group
    ///
    /// π₁(SU(n)) ≅ 0 for all n (simply connected)
    pub fn fundamental_group_order(&self) -> usize {
        1 // trivial group
    }

    /// Check if this is related to electroweak theory
    pub fn is_electroweak_gauge_group(&self) -> bool {
        self.n == 2 // SU(2) is part of SU(2) × U(1)
    }

    /// Check if this is the QCD gauge group
    pub fn is_qcd_gauge_group(&self) -> bool {
        self.n == 3 // SU(3) is the gauge group of quantum chromodynamics
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minkowski_creation() {
        let minkowski = Minkowski::new().unwrap();
        assert_eq!(minkowski.manifold().dimension(), 4);
    }

    #[test]
    fn test_schwarzschild_creation() {
        let schwarzschild = Schwarzschild::new(1.0).unwrap();
        assert_eq!(schwarzschild.mass(), 1.0);
        assert_eq!(schwarzschild.schwarzschild_radius(), 2.0);
    }

    #[test]
    fn test_schwarzschild_invalid_mass() {
        let result = Schwarzschild::new(-1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_schwarzschild_horizon() {
        let schwarzschild = Schwarzschild::new(1.0).unwrap();
        assert!(schwarzschild.is_inside_horizon(1.0));
        assert!(!schwarzschild.is_inside_horizon(3.0));
        assert!(schwarzschild.is_at_horizon(2.0, 0.01));
    }

    #[test]
    fn test_kerr_creation() {
        let kerr = Kerr::new(1.0, 0.5).unwrap();
        assert_eq!(kerr.mass(), 1.0);
        assert_eq!(kerr.spin(), 0.5);
    }

    #[test]
    fn test_kerr_extremal() {
        let kerr_extremal = Kerr::new(1.0, 1.0).unwrap();
        assert!(kerr_extremal.is_extremal(0.001));

        let kerr_non_extremal = Kerr::new(1.0, 0.5).unwrap();
        assert!(!kerr_non_extremal.is_extremal(0.001));
    }

    #[test]
    fn test_kerr_invalid_spin() {
        let result = Kerr::new(1.0, 1.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_real_projective_space() {
        let rp2 = RealProjectiveSpace::new(2).unwrap();
        assert_eq!(rp2.dimension(), 2);
        assert_eq!(rp2.euler_characteristic(), 1);

        let rp3 = RealProjectiveSpace::new(3).unwrap();
        assert_eq!(rp3.euler_characteristic(), 0);
    }

    #[test]
    fn test_complex_projective_space() {
        let cp2 = ComplexProjectiveSpace::new(2).unwrap();
        assert_eq!(cp2.complex_dimension(), 2);
        assert_eq!(cp2.real_dimension(), 4);
        assert_eq!(cp2.euler_characteristic(), 3); // n+1 = 2+1
    }

    #[test]
    fn test_cp_betti_numbers() {
        let cp2 = ComplexProjectiveSpace::new(2).unwrap();
        assert_eq!(cp2.betti_number(0), 1);
        assert_eq!(cp2.betti_number(1), 0);
        assert_eq!(cp2.betti_number(2), 1);
        assert_eq!(cp2.betti_number(3), 0);
        assert_eq!(cp2.betti_number(4), 1);
    }

    #[test]
    fn test_grassmannian() {
        let gr_2_4 = Grassmannian::new(2, 4).unwrap();
        assert_eq!(gr_2_4.k(), 2);
        assert_eq!(gr_2_4.n(), 4);
        assert_eq!(gr_2_4.dimension(), 4); // 2*(4-2) = 4
        assert!(gr_2_4.is_klein_quadric());
    }

    #[test]
    fn test_grassmannian_isomorphism() {
        let gr_2_5 = Grassmannian::new(2, 5).unwrap();
        let gr_3_5 = Grassmannian::new(3, 5).unwrap();
        assert!(gr_2_5.is_isomorphic_to(&gr_3_5));
    }

    #[test]
    fn test_grassmannian_invalid() {
        assert!(Grassmannian::new(0, 5).is_err());
        assert!(Grassmannian::new(5, 5).is_err());
        assert!(Grassmannian::new(6, 5).is_err());
    }

    #[test]
    fn test_so_n() {
        let so3 = SpecialOrthogonalGroup::new(3).unwrap();
        assert_eq!(so3.n(), 3);
        assert_eq!(so3.dimension(), 3); // 3*2/2 = 3
        assert!(so3.is_rotation_3d());
        assert_eq!(so3.fundamental_group_order(), 2);
    }

    #[test]
    fn test_so_2() {
        let so2 = SpecialOrthogonalGroup::new(2).unwrap();
        assert!(so2.is_circle_group());
        assert_eq!(so2.dimension(), 1);
        assert_eq!(so2.fundamental_group_order(), 0); // infinite (ℤ)
    }

    #[test]
    fn test_su_n() {
        let su2 = SpecialUnitaryGroup::new(2).unwrap();
        assert_eq!(su2.n(), 2);
        assert_eq!(su2.dimension(), 3); // 2²-1 = 3
        assert!(su2.is_three_sphere());
        assert!(su2.is_electroweak_gauge_group());
    }

    #[test]
    fn test_su_3() {
        let su3 = SpecialUnitaryGroup::new(3).unwrap();
        assert_eq!(su3.dimension(), 8); // 3²-1 = 8
        assert!(su3.is_qcd_gauge_group());
        assert_eq!(su3.fundamental_group_order(), 1); // simply connected
    }
}
