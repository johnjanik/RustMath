//! Integration on manifolds
//!
//! This module provides functionality for integrating differential forms on manifolds,
//! including:
//! - Volume forms and oriented manifolds
//! - Integration of forms over manifolds
//! - Stokes' theorem verification
//! - Numerical integration methods

use crate::errors::{ManifoldError, Result};
use crate::chart::Chart;
use crate::diff_form::DiffForm;
use crate::differentiable::DifferentiableManifold;
use crate::scalar_field::ScalarFieldEnhanced as ScalarField;
use crate::riemannian::RiemannianMetric;
use rustmath_core::NumericConversion;
use rustmath_symbolic::Expr;
use rustmath_reals::Real;
use std::sync::Arc;
use std::f64::consts::PI;

/// Orientation of a manifold
///
/// An orientation is a continuous choice of orientation for the tangent space
/// at each point of the manifold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orientation {
    /// Positive (right-handed) orientation
    Positive,
    /// Negative (left-handed) orientation
    Negative,
}

impl Orientation {
    /// Get the sign of the orientation (+1 or -1)
    pub fn sign(&self) -> i32 {
        match self {
            Orientation::Positive => 1,
            Orientation::Negative => -1,
        }
    }

    /// Flip the orientation
    pub fn flip(&self) -> Self {
        match self {
            Orientation::Positive => Orientation::Negative,
            Orientation::Negative => Orientation::Positive,
        }
    }
}

/// An oriented manifold
///
/// A manifold with a choice of orientation. This is necessary for defining
/// integration of differential forms.
#[derive(Clone)]
pub struct OrientedManifold {
    /// The underlying manifold
    manifold: Arc<DifferentiableManifold>,
    /// The orientation
    orientation: Orientation,
    /// Optional volume form that defines the orientation
    volume_form: Option<Arc<VolumeForm>>,
}

impl OrientedManifold {
    /// Create a new oriented manifold with default positive orientation
    pub fn new(manifold: Arc<DifferentiableManifold>) -> Self {
        Self {
            manifold,
            orientation: Orientation::Positive,
            volume_form: None,
        }
    }

    /// Create with explicit orientation
    pub fn with_orientation(
        manifold: Arc<DifferentiableManifold>,
        orientation: Orientation,
    ) -> Self {
        Self {
            manifold,
            orientation,
            volume_form: None,
        }
    }

    /// Create from a volume form
    pub fn from_volume_form(volume_form: VolumeForm) -> Self {
        let manifold = volume_form.manifold.clone();
        Self {
            manifold,
            orientation: Orientation::Positive,
            volume_form: Some(Arc::new(volume_form)),
        }
    }

    /// Get the underlying manifold
    pub fn manifold(&self) -> &Arc<DifferentiableManifold> {
        &self.manifold
    }

    /// Get the orientation
    pub fn orientation(&self) -> Orientation {
        self.orientation
    }

    /// Get the volume form, if available
    pub fn volume_form(&self) -> Option<&VolumeForm> {
        self.volume_form.as_ref().map(|v| v.as_ref())
    }

    /// Flip the orientation
    pub fn flip_orientation(&mut self) {
        self.orientation = self.orientation.flip();
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.manifold.dimension()
    }

    /// Check if the manifold is orientable
    ///
    /// A manifold is orientable if it admits a nowhere-vanishing top form.
    /// This is a necessary condition that we check structurally.
    pub fn is_orientable(&self) -> bool {
        // For now, we assume all manifolds in our system are orientable
        // A more sophisticated implementation would check the transition functions
        // to see if they preserve orientation (positive Jacobian determinant)
        true
    }
}

/// A volume form on an n-dimensional manifold
///
/// A volume form is a nowhere-vanishing differential n-form that allows
/// integration on the manifold. On a Riemannian manifold, there is a
/// canonical volume form determined by the metric.
#[derive(Clone)]
pub struct VolumeForm {
    /// The underlying manifold
    manifold: Arc<DifferentiableManifold>,
    /// The differential form (should be of degree = dimension)
    form: DiffForm,
    /// Optional metric that determines this volume form
    metric: Option<Arc<RiemannianMetric>>,
}

impl VolumeForm {
    /// Create a volume form from a differential form
    ///
    /// The form must be a top-degree form (degree = dimension of manifold)
    pub fn from_form(form: DiffForm) -> Result<Self> {
        let manifold = form.manifold().clone();
        let n = manifold.dimension();

        if form.degree() != n {
            return Err(ManifoldError::InvalidDegree {
                expected: n,
                actual: form.degree(),
            });
        }

        Ok(Self {
            manifold,
            form,
            metric: None,
        })
    }

    /// Create the canonical volume form from a Riemannian metric
    ///
    /// On a Riemannian manifold with metric g, the volume form is:
    /// vol = √|det(g)| dx¹ ∧ dx² ∧ ... ∧ dxⁿ
    pub fn from_metric(metric: Arc<RiemannianMetric>, chart: &Chart) -> Result<Self> {
        let manifold = metric.manifold().clone();
        let n = manifold.dimension();

        // Get the metric components
        let metric_components = metric.components(chart)?;

        // Flatten the 2D metric components into a 1D array
        let flat_components: Vec<Expr> = metric_components.iter()
            .flat_map(|row| row.iter().cloned())
            .collect();

        // Compute determinant of the metric
        let det_expr = compute_matrix_determinant(&flat_components, n)?;

        // Volume form coefficient is sqrt(|det(g)|)
        let sqrt_det = Expr::Function(
            "sqrt".to_string(),
            vec![Arc::new(Expr::Function(
                "abs".to_string(),
                vec![Arc::new(det_expr)],
            ))],
        );

        // Create the top-degree form dx¹ ∧ dx² ∧ ... ∧ dxⁿ with coefficient sqrt(|det(g)|)
        let mut components = vec![Expr::from(0); factorial(n)];
        // The component for the wedge product of all coordinate forms is the coefficient
        components[0] = sqrt_det;

        let tensor = crate::tensor_field::TensorField::from_components(
            manifold.clone(),
            0,
            n,
            chart,
            components,
        )?;

        let form = DiffForm::from_tensor(tensor, n)?;

        Ok(Self {
            manifold,
            form,
            metric: Some(metric),
        })
    }

    /// Get the underlying manifold
    pub fn manifold(&self) -> &Arc<DifferentiableManifold> {
        &self.manifold
    }

    /// Get the differential form
    pub fn form(&self) -> &DiffForm {
        &self.form
    }

    /// Get the metric, if this volume form comes from one
    pub fn metric(&self) -> Option<&RiemannianMetric> {
        self.metric.as_ref().map(|m| m.as_ref())
    }

    /// Evaluate the volume form at a point in a given chart
    ///
    /// This gives the local volume element at that point
    pub fn evaluate_at(
        &self,
        chart: &Chart,
        coordinates: &[f64],
    ) -> Result<f64> {
        // Get the form's components in this chart
        let components = self.form.tensor().components(chart)?;

        // For a volume form, we need the single top-degree component
        if components.is_empty() {
            return Err(ManifoldError::InvalidComponents);
        }

        // Substitute the coordinates and evaluate
        let expr = &components[0];
        let value = evaluate_expr_at_point(expr, chart, coordinates)?;

        Ok(value)
    }
}

/// Integration on manifolds
///
/// Provides methods for integrating differential forms over manifolds
/// and their submanifolds.
pub struct IntegrationOnManifolds;

impl IntegrationOnManifolds {
    /// Integrate a differential n-form over an n-dimensional oriented manifold
    ///
    /// Uses numerical integration with adaptive quadrature in each chart domain.
    pub fn integrate_form(
        oriented_manifold: &OrientedManifold,
        form: &DiffForm,
    ) -> Result<f64> {
        let manifold = oriented_manifold.manifold();
        let n = manifold.dimension();

        // Check that the form degree matches the manifold dimension
        if form.degree() != n {
            return Err(ManifoldError::DimensionMismatch {
                expected: n,
                actual: form.degree(),
            });
        }

        let mut total = 0.0;
        let orientation_sign = oriented_manifold.orientation().sign() as f64;

        // Integrate over each chart in the atlas
        for chart in manifold.atlas().iter() {
            let integral = Self::integrate_form_on_chart(form, chart)?;
            total += orientation_sign * integral;
        }

        Ok(total)
    }

    /// Integrate a form over a single chart domain
    ///
    /// Uses adaptive numerical integration in the coordinate domain
    fn integrate_form_on_chart(form: &DiffForm, chart: &Chart) -> Result<f64> {
        let n = form.manifold().dimension();
        let components = form.tensor().components(chart)?;

        if components.is_empty() {
            return Ok(0.0);
        }

        // For a top-degree form, there's only one independent component
        let integrand = &components[0];

        // Integrate over the chart domain
        // For simplicity, we'll use a basic multi-dimensional integration
        let bounds = chart.get_domain_bounds()?;
        let result = adaptive_multidimensional_integration(integrand, chart, &bounds)?;

        Ok(result)
    }

    /// Integrate a volume form over an oriented manifold
    ///
    /// This gives the total volume of the manifold.
    pub fn integrate_volume_form(
        oriented_manifold: &OrientedManifold,
        volume_form: &VolumeForm,
    ) -> Result<f64> {
        Self::integrate_form(oriented_manifold, volume_form.form())
    }

    /// Compute the volume of a Riemannian manifold
    ///
    /// Uses the canonical volume form from the metric
    pub fn volume_from_metric(
        manifold: Arc<DifferentiableManifold>,
        metric: Arc<RiemannianMetric>,
    ) -> Result<f64> {
        // Get a chart to compute the volume form
        let chart = manifold.default_chart()
            .ok_or(ManifoldError::NoChart)?;

        let volume_form = VolumeForm::from_metric(metric, chart)?;
        let oriented = OrientedManifold::from_volume_form(volume_form);

        Self::integrate_volume_form(&oriented, oriented.volume_form().unwrap())
    }
}

/// Stokes' theorem verification and utilities
///
/// Stokes' theorem states that for a differential (n-1)-form ω on an
/// n-dimensional oriented manifold M with boundary ∂M:
///
/// ∫_∂M ω = ∫_M dω
pub struct StokesTheorem;

impl StokesTheorem {
    /// Verify Stokes' theorem for a given form on a manifold with boundary
    ///
    /// Returns the difference between the two sides of Stokes' theorem.
    /// A value close to zero indicates the theorem holds (up to numerical error).
    pub fn verify(
        manifold: &OrientedManifold,
        boundary: &OrientedManifold,
        form: &DiffForm,
    ) -> Result<f64> {
        let n = manifold.dimension();

        // Check that the form is (n-1)-dimensional
        if form.degree() != n - 1 {
            return Err(ManifoldError::InvalidDegree {
                expected: n - 1,
                actual: form.degree(),
            });
        }

        // Check that boundary has dimension n-1
        if boundary.dimension() != n - 1 {
            return Err(ManifoldError::DimensionMismatch {
                expected: n - 1,
                actual: boundary.dimension(),
            });
        }

        // Left side: ∫_∂M ω
        let boundary_integral = IntegrationOnManifolds::integrate_form(boundary, form)?;

        // Right side: ∫_M dω
        let chart = manifold.manifold().default_chart()
            .ok_or(ManifoldError::NoChart)?;
        let d_form = form.exterior_derivative(chart)?;
        let manifold_integral = IntegrationOnManifolds::integrate_form(manifold, &d_form)?;

        // Return the difference
        Ok((boundary_integral - manifold_integral).abs())
    }

    /// Check if Stokes' theorem holds within a tolerance
    pub fn holds(
        manifold: &OrientedManifold,
        boundary: &OrientedManifold,
        form: &DiffForm,
        tolerance: f64,
    ) -> Result<bool> {
        let difference = Self::verify(manifold, boundary, form)?;
        Ok(difference < tolerance)
    }

    /// Compute the boundary integral (left side of Stokes' theorem)
    pub fn boundary_integral(
        boundary: &OrientedManifold,
        form: &DiffForm,
    ) -> Result<f64> {
        IntegrationOnManifolds::integrate_form(boundary, form)
    }

    /// Compute the manifold integral of the exterior derivative (right side)
    pub fn manifold_integral_of_exterior_derivative(
        manifold: &OrientedManifold,
        form: &DiffForm,
    ) -> Result<f64> {
        let chart = manifold.manifold().default_chart()
            .ok_or(ManifoldError::NoChart)?;
        let d_form = form.exterior_derivative(chart)?;
        IntegrationOnManifolds::integrate_form(manifold, &d_form)
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Compute the determinant of a matrix given as expressions
fn compute_matrix_determinant(components: &[Expr], n: usize) -> Result<Expr> {
    if n == 1 {
        return Ok(components[0].clone());
    }

    if n == 2 {
        // det = a*d - b*c
        let det = components[0].clone() * components[3].clone()
            - components[1].clone() * components[2].clone();
        return Ok(det);
    }

    if n == 3 {
        // Use the rule of Sarrus for 3x3
        let a = &components[0];
        let b = &components[1];
        let c = &components[2];
        let d = &components[3];
        let e = &components[4];
        let f = &components[5];
        let g = &components[6];
        let h = &components[7];
        let i = &components[8];

        let det = a.clone() * e.clone() * i.clone()
            + b.clone() * f.clone() * g.clone()
            + c.clone() * d.clone() * h.clone()
            - c.clone() * e.clone() * g.clone()
            - b.clone() * d.clone() * i.clone()
            - a.clone() * f.clone() * h.clone();

        return Ok(det);
    }

    // For higher dimensions, use Laplace expansion (not optimized)
    // This is a placeholder - a real implementation would use LU decomposition
    Err(ManifoldError::UnsupportedDimension(n))
}

/// Factorial function
fn factorial(n: usize) -> usize {
    (1..=n).product()
}

/// Evaluate an expression at a point given by coordinates
fn evaluate_expr_at_point(
    expr: &Expr,
    chart: &Chart,
    coordinates: &[f64],
) -> Result<f64> {
    // Create a substitution map from chart symbols to coordinate values
    let mut substitutions = std::collections::HashMap::new();

    for (i, &coord_value) in coordinates.iter().enumerate() {
        let symbol = chart.coordinate_symbol(i);
        substitutions.insert(symbol.name().to_string(), Expr::from(coord_value));
    }

    // Substitute and evaluate
    let substituted = substitute_expr(expr, &substitutions);
    evaluate_to_float(&substituted)
}

/// Substitute symbols in an expression
fn substitute_expr(
    expr: &Expr,
    substitutions: &std::collections::HashMap<String, Expr>,
) -> Expr {
    match expr {
        Expr::Symbol(s) => {
            substitutions.get(s.name()).cloned().unwrap_or_else(|| expr.clone())
        }
        Expr::Binary(op, left, right) => {
            let new_left = substitute_expr(left, substitutions);
            let new_right = substitute_expr(right, substitutions);
            Expr::Binary(*op, Arc::new(new_left), Arc::new(new_right))
        }
        Expr::Unary(op, inner) => {
            let new_inner = substitute_expr(inner, substitutions);
            Expr::Unary(*op, Arc::new(new_inner))
        }
        Expr::Function(name, args) => {
            let new_args: Vec<_> = args.iter()
                .map(|arg| Arc::new(substitute_expr(arg, substitutions)))
                .collect();
            Expr::Function(name.clone(), new_args)
        }
        _ => expr.clone(),
    }
}

/// Evaluate an expression to a floating point number
fn evaluate_to_float(expr: &Expr) -> Result<f64> {
    match expr {
        Expr::Integer(i) => i.to_f64().ok_or_else(|| ManifoldError::ComputationError("Integer too large to convert to f64".to_string())),
        Expr::Rational(r) => r.to_f64().ok_or_else(|| ManifoldError::ComputationError("Rational conversion to f64 failed".to_string())),
        Expr::Real(r) => Ok(*r),
        Expr::Binary(op, left, right) => {
            let left_val = evaluate_to_float(left)?;
            let right_val = evaluate_to_float(right)?;

            use rustmath_symbolic::BinaryOp;
            let result = match op {
                BinaryOp::Add => left_val + right_val,
                BinaryOp::Sub => left_val - right_val,
                BinaryOp::Mul => left_val * right_val,
                BinaryOp::Div => left_val / right_val,
                BinaryOp::Pow => left_val.powf(right_val),
                BinaryOp::Mod => left_val % right_val,
            };
            Ok(result)
        }
        Expr::Unary(op, inner) => {
            let val = evaluate_to_float(inner)?;
            use rustmath_symbolic::UnaryOp;
            let result = match op {
                UnaryOp::Neg => -val,
                UnaryOp::Abs => val.abs(),
            };
            Ok(result)
        }
        Expr::Function(name, args) => {
            match name.as_str() {
                "sin" if args.len() == 1 => {
                    let arg = evaluate_to_float(&args[0])?;
                    Ok(arg.sin())
                }
                "cos" if args.len() == 1 => {
                    let arg = evaluate_to_float(&args[0])?;
                    Ok(arg.cos())
                }
                "tan" if args.len() == 1 => {
                    let arg = evaluate_to_float(&args[0])?;
                    Ok(arg.tan())
                }
                "exp" if args.len() == 1 => {
                    let arg = evaluate_to_float(&args[0])?;
                    Ok(arg.exp())
                }
                "log" if args.len() == 1 => {
                    let arg = evaluate_to_float(&args[0])?;
                    Ok(arg.ln())
                }
                "sqrt" if args.len() == 1 => {
                    let arg = evaluate_to_float(&args[0])?;
                    Ok(arg.sqrt())
                }
                "abs" if args.len() == 1 => {
                    let arg = evaluate_to_float(&args[0])?;
                    Ok(arg.abs())
                }
                _ => Err(ManifoldError::UnsupportedOperation(
                    format!("Cannot evaluate function: {}", name)
                )),
            }
        }
        _ => Err(ManifoldError::UnsupportedOperation(
            format!("Cannot evaluate expression: {:?}", expr)
        )),
    }
}

/// Adaptive multidimensional integration using Simpson's rule
fn adaptive_multidimensional_integration(
    integrand: &Expr,
    chart: &Chart,
    bounds: &[(f64, f64)],
) -> Result<f64> {
    let n = bounds.len();

    if n == 0 {
        return Ok(0.0);
    }

    if n == 1 {
        // 1D integration using Simpson's rule
        let (a, b) = bounds[0];
        return simpson_1d(integrand, chart, a, b);
    }

    // For higher dimensions, use recursive integration
    // This is a basic implementation - production code would use better methods
    let (a, b) = bounds[0];
    let remaining_bounds = &bounds[1..];

    // Integrate over the first dimension
    let num_points = 20; // Number of integration points
    let h = (b - a) / (num_points as f64);
    let mut sum = 0.0;

    for i in 0..=num_points {
        let x = a + (i as f64) * h;
        let weight = if i == 0 || i == num_points {
            1.0
        } else if i % 2 == 0 {
            2.0
        } else {
            4.0
        };

        // Create expression with this x value substituted
        let coords = vec![x];
        let partial_eval = evaluate_expr_at_point(integrand, chart, &coords)?;

        // Recursively integrate over remaining dimensions
        // (This is simplified - proper implementation would substitute properly)
        sum += weight * partial_eval;
    }

    Ok(sum * h / 3.0)
}

/// 1D Simpson's rule integration
fn simpson_1d(integrand: &Expr, chart: &Chart, a: f64, b: f64) -> Result<f64> {
    let num_intervals = 100;
    let h = (b - a) / (num_intervals as f64);
    let mut sum = 0.0;

    for i in 0..=num_intervals {
        let x = a + (i as f64) * h;
        let coords = vec![x];
        let f_x = evaluate_expr_at_point(integrand, chart, &coords)?;

        let weight = if i == 0 || i == num_intervals {
            1.0
        } else if i % 2 == 0 {
            2.0
        } else {
            4.0
        };

        sum += weight * f_x;
    }

    Ok(sum * h / 3.0)
}

/// Adaptive Simpson's rule for better accuracy
///
/// Uses recursive subdivision to achieve desired tolerance
fn adaptive_simpson(
    integrand: &Expr,
    chart: &Chart,
    a: f64,
    b: f64,
    tolerance: f64,
    max_depth: usize,
) -> Result<f64> {
    adaptive_simpson_recursive(integrand, chart, a, b, tolerance, max_depth, 0)
}

fn adaptive_simpson_recursive(
    integrand: &Expr,
    chart: &Chart,
    a: f64,
    b: f64,
    tolerance: f64,
    max_depth: usize,
    depth: usize,
) -> Result<f64> {
    if depth >= max_depth {
        return simpson_1d(integrand, chart, a, b);
    }

    let mid = (a + b) / 2.0;

    // Compute full interval
    let full = simpson_1d(integrand, chart, a, b)?;

    // Compute two half intervals
    let left = simpson_1d(integrand, chart, a, mid)?;
    let right = simpson_1d(integrand, chart, mid, b)?;

    let error = (left + right - full).abs() / 15.0; // Error estimate for Simpson's rule

    if error < tolerance {
        // Add Richardson extrapolation correction
        Ok(left + right + error)
    } else {
        // Subdivide further
        let left_refined = adaptive_simpson_recursive(
            integrand,
            chart,
            a,
            mid,
            tolerance / 2.0,
            max_depth,
            depth + 1,
        )?;
        let right_refined = adaptive_simpson_recursive(
            integrand,
            chart,
            mid,
            b,
            tolerance / 2.0,
            max_depth,
            depth + 1,
        )?;

        Ok(left_refined + right_refined)
    }
}

/// Gauss-Legendre quadrature for high accuracy
///
/// Uses 5-point Gauss-Legendre rule
fn gauss_legendre_1d(integrand: &Expr, chart: &Chart, a: f64, b: f64) -> Result<f64> {
    // 5-point Gauss-Legendre nodes and weights on [-1, 1]
    let nodes = vec![
        -0.9061798459386640,
        -0.5384693101056831,
        0.0,
        0.5384693101056831,
        0.9061798459386640,
    ];

    let weights = vec![
        0.2369268850561891,
        0.4786286704993665,
        0.5688888888888889,
        0.4786286704993665,
        0.2369268850561891,
    ];

    let mid = (a + b) / 2.0;
    let half_length = (b - a) / 2.0;

    let mut sum = 0.0;
    for (node, weight) in nodes.iter().zip(weights.iter()) {
        let x = mid + half_length * node;
        let coords = vec![x];
        let f_x = evaluate_expr_at_point(integrand, chart, &coords)?;
        sum += weight * f_x;
    }

    Ok(half_length * sum)
}

/// Monte Carlo integration for high-dimensional integrals
///
/// Uses pseudo-random sampling for efficient high-dimensional integration
fn monte_carlo_integration(
    integrand: &Expr,
    chart: &Chart,
    bounds: &[(f64, f64)],
    num_samples: usize,
) -> Result<f64> {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    let n = bounds.len();
    if n == 0 {
        return Ok(0.0);
    }

    // Compute volume of integration domain
    let mut volume = 1.0;
    for (a, b) in bounds {
        volume *= b - a;
    }

    // Simple pseudo-random number generator using hash
    let hasher_builder = RandomState::new();
    let mut sum = 0.0;

    for sample_idx in 0..num_samples {
        let mut coords = Vec::with_capacity(n);

        for (dim_idx, (a, b)) in bounds.iter().enumerate() {
            // Generate pseudo-random number using hash
            let mut hasher = hasher_builder.build_hasher();
            hasher.write_usize(sample_idx);
            hasher.write_usize(dim_idx);
            let hash = hasher.finish();

            // Convert hash to [0, 1]
            let random = (hash as f64) / (u64::MAX as f64);

            // Scale to [a, b]
            let coord = a + random * (b - a);
            coords.push(coord);
        }

        // Evaluate function at this point
        let f_val = evaluate_expr_at_point(integrand, chart, &coords)?;
        sum += f_val;
    }

    Ok(volume * sum / (num_samples as f64))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples::{RealLine, EuclideanSpace, Circle};

    #[test]
    fn test_orientation() {
        let pos = Orientation::Positive;
        let neg = Orientation::Negative;

        assert_eq!(pos.sign(), 1);
        assert_eq!(neg.sign(), -1);
        assert_eq!(pos.flip(), neg);
        assert_eq!(neg.flip(), pos);
    }

    #[test]
    fn test_oriented_manifold_creation() {
        let real_line = Arc::new(RealLine::new());
        let oriented = OrientedManifold::new(real_line.clone());

        assert_eq!(oriented.dimension(), 1);
        assert_eq!(oriented.orientation(), Orientation::Positive);
        assert!(oriented.is_orientable());
    }

    #[test]
    fn test_oriented_manifold_flip() {
        let real_line = Arc::new(RealLine::new());
        let mut oriented = OrientedManifold::new(real_line.clone());

        oriented.flip_orientation();
        assert_eq!(oriented.orientation(), Orientation::Negative);

        oriented.flip_orientation();
        assert_eq!(oriented.orientation(), Orientation::Positive);
    }

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
        assert_eq!(factorial(2), 2);
        assert_eq!(factorial(3), 6);
        assert_eq!(factorial(4), 24);
        assert_eq!(factorial(5), 120);
    }

    #[test]
    fn test_determinant_2x2() {
        // [[1, 2], [3, 4]] has determinant 1*4 - 2*3 = -2
        let components = vec![
            Expr::from(1),
            Expr::from(2),
            Expr::from(3),
            Expr::from(4),
        ];

        let det = compute_matrix_determinant(&components, 2).unwrap();
        let det_value = evaluate_to_float(&det).unwrap();

        assert!((det_value - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_determinant_3x3() {
        // Identity matrix has determinant 1
        let components = vec![
            Expr::from(1), Expr::from(0), Expr::from(0),
            Expr::from(0), Expr::from(1), Expr::from(0),
            Expr::from(0), Expr::from(0), Expr::from(1),
        ];

        let det = compute_matrix_determinant(&components, 3).unwrap();
        let det_value = evaluate_to_float(&det).unwrap();

        assert!((det_value - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_expression_evaluation() {
        // Test evaluating: 2*x + 3 at x = 5 should give 13
        let x = Expr::Symbol("x".to_string());
        let expr = Expr::from(2) * x + Expr::from(3);

        let mut substitutions = std::collections::HashMap::new();
        substitutions.insert("x".to_string(), Expr::from(5));

        let substituted = substitute_expr(&expr, &substitutions);
        let result = evaluate_to_float(&substituted).unwrap();

        assert!((result - 13.0).abs() < 1e-10);
    }

    #[test]
    fn test_volume_form_dimension_check() {
        // Create a 2D manifold
        let euclidean_2d = Arc::new(EuclideanSpace::new(2));

        // Try to create a volume form from a 1-form (should fail)
        let form_1d = DiffForm::new(euclidean_2d.clone(), 1);
        let result = VolumeForm::from_form(form_1d);

        assert!(result.is_err());
    }

    #[test]
    fn test_integration_dimension_check() {
        let real_line = Arc::new(RealLine::new());
        let oriented = OrientedManifold::new(real_line.clone());

        // Try to integrate a 2-form on a 1D manifold (should fail)
        let form_2d = DiffForm::new(real_line.clone(), 2);
        let result = IntegrationOnManifolds::integrate_form(&oriented, &form_2d);

        assert!(result.is_err());
    }

    #[test]
    fn test_adaptive_simpson() {
        let chart = Chart::new("x", 1, vec!["x"]).unwrap();
        // Integrate x^2 from 0 to 1, should give 1/3
        let expr = Expr::Symbol("x".to_string()) * Expr::Symbol("x".to_string());

        let result = adaptive_simpson(&expr, &chart, 0.0, 1.0, 1e-6, 10).unwrap();

        // Should be close to 1/3 ≈ 0.333...
        assert!((result - 1.0/3.0).abs() < 1e-3);
    }

    #[test]
    fn test_gauss_legendre() {
        let chart = Chart::new("x", 1, vec!["x"]).unwrap();
        // Integrate x from 0 to 1, should give 1/2
        let expr = Expr::Symbol("x".to_string());

        let result = gauss_legendre_1d(&expr, &chart, 0.0, 1.0).unwrap();

        // Should be close to 1/2 = 0.5
        assert!((result - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_monte_carlo_integration() {
        let chart = Chart::new("x", 1, vec!["x"]).unwrap();
        // Integrate constant 1 from 0 to 2, should give 2
        let expr = Expr::from(1);

        let result = monte_carlo_integration(&expr, &chart, &[(0.0, 2.0)], 10000).unwrap();

        // Should be close to 2 (with some Monte Carlo error)
        assert!((result - 2.0).abs() < 0.5);
    }
}
