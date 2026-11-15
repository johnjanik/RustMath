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
use crate::diff_form::{check_coordinate_chart, DiffForm};
use crate::expr_zero::{check_derivative, normalize, sum_signed, ZeroTest};
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
    /// let manifold = Arc::new(EuclideanSpace::new(2).manifold().clone());
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
    /// Degree >= 2 goes through [`DiffForm::lie_derivative`] (Cartan's formula with the exact
    /// d, wedge and interior product); degree 1 uses the coordinate formula above.
    ///
    /// # Arguments
    ///
    /// * `form` - The differential form ω
    /// * `chart` - The chart in which to compute the derivative; it must be the chart the
    ///   components were written in
    ///
    /// # Returns
    ///
    /// A new differential form representing L_X ω
    ///
    /// # Errors
    ///
    /// `InvalidOperation` for a 0-form (use `apply_to_scalar`); `NoComponentsInChart`;
    /// `InvalidChart` (the checks of the exterior derivative); `NotImplemented` when a
    /// derivative is needed that rustmath-symbolic's `differentiate` does not implement.
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

        form.lie_derivative(&self.vector_field, chart)
    }

    /// Lie derivative of a 1-form
    ///
    /// For a 1-form ω with components ω_i:
    /// (L_X ω)_i = X^j ∂ω_i/∂x^j + ω_j ∂X^j/∂x^i
    ///
    /// A term whose other factor is a literal 0 is skipped before its derivative is taken.
    /// Every other derivative is checked first (`check_derivative`): rustmath-symbolic's
    /// `differentiate` returns 0 for named functions, gamma and the other kinds listed in
    /// [`crate::expr_zero`]. Each component is a balanced sum, simplified and zero-tested.
    fn apply_to_one_form(&self, form: &DiffForm, chart: &Chart) -> Result<DiffForm> {
        let manifold = form.manifold();
        let n = manifold.dimension();

        // Get components
        let omega_components = form.tensor().components(chart)?;
        let x_components = self.vector_field.components(chart)?;
        if omega_components.len() != n || x_components.len() != n {
            return Err(ManifoldError::DimensionMismatch {
                expected: n,
                actual: x_components.len().min(omega_components.len()),
            });
        }
        check_coordinate_chart(manifold, chart)?;
        let symbols = chart.coordinate_symbols();

        // Compute L_X ω
        let mut lie_components = Vec::with_capacity(n);

        for i in 0..n {
            let mut terms: Vec<(Expr, bool)> = Vec::new();
            for j in 0..n {
                // First term: X^j ∂ω_i/∂x^j
                if !x_components[j].is_zero() {
                    check_derivative(&omega_components[i], &symbols[j])?;
                    let t = (x_components[j].clone() * omega_components[i].differentiate(&symbols[j])).simplify();
                    if !t.is_zero() {
                        terms.push((t, false));
                    }
                }
                // Second term: ω_j ∂X^j/∂x^i
                if !omega_components[j].is_zero() {
                    check_derivative(&x_components[j], &symbols[i])?;
                    let t = (omega_components[j].clone() * x_components[j].differentiate(&symbols[i])).simplify();
                    if !t.is_zero() {
                        terms.push((t, false));
                    }
                }
            }
            lie_components.push(match sum_signed(terms) {
                Some(sum) => normalize(sum)?,
                None => Expr::from(0),
            });
        }

        // Create new 1-form
        DiffForm::from_components(manifold.clone(), chart, 1, lie_components)
    }

    /// Lie derivative of a general tensor field
    ///
    /// For a tensor T of type (p, q), the Lie derivative is:
    /// (L_X T)^{i_1...i_p}_{j_1...j_q} =
    ///   X^k ∂T^{i_1...i_p}_{j_1...j_q}/∂x^k
    ///   - Σ_a T^{i_1...k...i_p}_{j_1...j_q} ∂X^{i_a}/∂x^k (sum over contravariant indices)
    ///   + Σ_b T^{i_1...i_p}_{j_1...k...j_q} ∂X^k/∂x^{j_b} (sum over covariant indices)
    ///
    /// Not implemented for tensors other than the three special types below: the correction
    /// terms are missing, and the directional derivative alone is not the Lie derivative.
    ///
    /// # Errors
    ///
    /// * `InvalidOperation` for types (0, 0), (1, 0) and (0, 1): use `apply_to_scalar`,
    ///   `apply_to_vector` or `apply_to_form`.
    /// * `NotImplemented` for every other type.
    pub fn apply_to_tensor(&self, tensor: &TensorField, _chart: &Chart) -> Result<TensorField> {
        let p = tensor.contravariant_rank();
        let q = tensor.covariant_rank();

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

        Err(ManifoldError::NotImplemented(format!(
            "the Lie derivative of a ({}, {}) tensor field needs the correction terms \
             -Σ_a T^{{..k..}} ∂_k X^{{i_a}} + Σ_b T_{{..k..}} ∂_{{j_b}} X^k (Lee 2013, ch. 12)",
            p, q
        )))
    }

    /// Check if the Lie derivative commutes with the exterior derivative
    ///
    /// For differential forms, we have: L_X(dω) = d(L_X ω)
    /// This is a fundamental property that can be used for verification.
    ///
    /// Decided with the exact zero test on L_X(dω) - d(L_X ω): `Ok(true)` when it vanishes,
    /// `Ok(false)` when it does not, `Err(Undecided)` when the zero test cannot decide. The
    /// errors of [`LieDerivative::apply_to_form`] and of the exterior derivative propagate.
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

        match (lie_d_omega + (-d_lie_omega))?.vanishes(chart)? {
            ZeroTest::Zero => Ok(true),
            ZeroTest::NonZero => Ok(false),
            ZeroTest::Unresolved => Err(ManifoldError::Undecided(
                "L_X(dω) - d(L_X ω) has components the zero test cannot decide".to_string(),
            )),
        }
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
        let manifold = Arc::new(EuclideanSpace::new(2).manifold().clone());
        let x = VectorField::new(manifold.clone());
        let lie_x = LieDerivative::new(Arc::new(x));

        assert_eq!(lie_x.vector_field().manifold().dimension(), 2);
    }

    #[test]
    fn test_lie_derivative_of_scalar() {
        let manifold = Arc::new(EuclideanSpace::new(2).manifold().clone());
        let chart = manifold.default_chart().unwrap();

        // Create a vector field X = ∂/∂x
        let x_field = VectorField::from_components(
            manifold.clone(),
            chart,
            vec![Expr::from(1), Expr::from(0)],
        ).unwrap();

        // Create a scalar field f = x
        let mut f = ScalarField::new(manifold.clone());
        f.set_expr(chart, Expr::symbol("x")).unwrap();

        // Compute L_X f
        let lie_x = LieDerivative::new(Arc::new(x_field));
        let result = lie_x.apply_to_scalar(&f, chart).unwrap();

        // L_X f should be 1 (since ∂x/∂x = 1)
        let result_expr = result.expr(chart).unwrap();
        assert_eq!(result_expr.simplify(), Expr::from(1));
    }

    #[test]
    fn test_lie_derivative_of_vector() {
        let manifold = Arc::new(EuclideanSpace::new(2).manifold().clone());
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

        // Result should be zero. The bracket of constant fields yields
        // unsimplified expressions like (1*0 - 0*0), so compare after
        // constant folding rather than structurally.
        let comps = result.components(chart).unwrap();
        assert!(comps.iter().all(|c| c.simplify().is_zero()));
    }

    #[test]
    fn test_lie_derivative_of_one_form() {
        let manifold = Arc::new(EuclideanSpace::new(2).manifold().clone());
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

        // For constant 1-forms, the Lie derivative is zero; the components are
        // built as unsimplified constant expressions, so simplify before the
        // zero test.
        let comps = result.tensor().components(chart).unwrap();
        assert!(comps.iter().all(|c| c.simplify().is_zero()));
    }

    #[test]
    fn test_apply_to_tensor_not_implemented() {
        // At HEAD the directional derivative alone came back: for T = x dx⊗dx and X = (y, x)
        // it gave [y, 0, 0, 0]; by hand (L_X T)_ij = X^k ∂_k T_ij + T_kj ∂_i X^k + T_ik ∂_j X^k
        // = [y, x, x, 0].
        let manifold = Arc::new(EuclideanSpace::new(2).manifold().clone());
        let chart = manifold.default_chart().unwrap();
        let (x, y) = (Expr::symbol("x"), Expr::symbol("y"));
        let field = VectorField::from_components(manifold.clone(), chart, vec![y.clone(), x.clone()]).unwrap();
        let lie = LieDerivative::new(Arc::new(field));
        let zero4 = || vec![Expr::from(0); 4];
        let mut t02 = zero4();
        t02[0] = x.clone();
        for (p, q, comps) in [(0, 2, t02), (2, 0, zero4()), (1, 1, zero4())] {
            let t = TensorField::from_components(manifold.clone(), p, q, chart, comps).unwrap();
            assert!(matches!(lie.apply_to_tensor(&t, chart), Err(ManifoldError::NotImplemented(_))));
        }
        let v = TensorField::from_components(manifold.clone(), 1, 0, chart, vec![x, y]).unwrap();
        assert!(matches!(lie.apply_to_tensor(&v, chart), Err(ManifoldError::InvalidOperation(_))));
    }

    fn r3() -> (Arc<crate::differentiable::DifferentiableManifold>, Chart) {
        let mut m = crate::differentiable::DifferentiableManifold::new("R3", 3);
        let c = Chart::new("c", 3, vec!["x", "y", "z"]).unwrap();
        m.add_chart(c.clone()).unwrap();
        (Arc::new(m), c)
    }

    fn assert_value(ours: &Expr, gp: &str) {
        let reference = rustmath_symbolic::parse(gp).unwrap();
        assert_eq!(
            crate::expr_zero::zero_test(&(ours.clone() - reference)),
            ZeroTest::Zero,
            "{} is not {}",
            ours,
            gp
        );
    }

    #[test]
    fn test_apply_to_two_form() {
        // gp (refs.gp): d(L_X A) = L_X dA = [-2*y*x^2 - 2*y*x + (-z^2 - z)*y,
        //     2*z*x^2 + (-z*y^2 - z^2), y*x + 2*z*y] for A = (x*y*z, -z*x, y*z),
        //     X = (y*z, -x*z, x*y). HEAD returned a 1-form for a 2-form.
        let (m, c) = r3();
        let p = |s: &str| rustmath_symbolic::parse(s).unwrap();
        let a = DiffForm::from_components(m.clone(), &c, 1, vec![p("x*y*z"), p("-z*x"), p("y*z")]).unwrap();
        let x = VectorField::from_components(m.clone(), &c, vec![p("y*z"), p("-x*z"), p("x*y")]).unwrap();
        let lie = LieDerivative::new(Arc::new(x));
        let r = lie.apply_to_form(&a.exterior_derivative(&c).unwrap(), &c).unwrap();
        assert_eq!(r.degree(), 2);
        let v: Vec<Expr> = r.independent_components(&c).unwrap().into_iter().map(|(_, e)| e).collect();
        assert_value(&v[0], "-2*y*x^2 - 2*y*x + (-z^2 - z)*y");
        assert_value(&v[1], "2*z*x^2 + (-z*y^2 - z^2)");
        assert_value(&v[2], "y*x + 2*z*y");
        // L_X(dA) = d(L_X A): HEAD said false (structural comparison)
        assert!(lie.commutes_with_exterior_derivative(&a, &c).unwrap());
    }

    #[test]
    fn test_lie_one_form_guard() {
        // differentiate returns 0 for f(x), gamma(y) and root_sum(t - x, t) in x; the guard
        // refuses instead. gp (refs2.gp): L_X(xy dx) = [x^2 + y^2, y*x] for X = (y, x);
        // refs4.gp: root_sum(t - x, t) = x, so L_(d/dx)(root_sum(t - x, t) dy) = dy.
        let m = Arc::new(EuclideanSpace::new(2).manifold().clone());
        let c2 = Chart::new("c", 2, vec!["x", "y"]).unwrap();
        let (x, y, t) = (Expr::symbol("x"), Expr::symbol("y"), Expr::symbol("t"));
        let f = Expr::Function("f".into(), vec![Arc::new(x.clone())]);
        let rs = Expr::Function("root_sum".into(), vec![Arc::new(t.clone() - x.clone()), Arc::new(t.clone())]);
        let run = |xv: Vec<Expr>, w: Vec<Expr>, chart: &Chart| {
            let field = VectorField::from_components(m.clone(), &c2, xv).unwrap();
            let form = DiffForm::from_components(m.clone(), &c2, 1, w).unwrap();
            LieDerivative::new(Arc::new(field)).apply_to_form(&form, chart)
        };
        let dx = || vec![Expr::from(1), Expr::from(0)];
        let zero = Expr::from(0);
        assert!(matches!(run(dx(), vec![zero.clone(), f.clone()], &c2), Err(ManifoldError::NotImplemented(_))));
        assert!(matches!(
            run(vec![y.clone().gamma(), zero.clone()], vec![x.clone(), zero.clone()], &c2),
            Err(ManifoldError::NotImplemented(_))
        ));
        let r = run(dx(), vec![zero.clone(), y.clone().gamma()], &c2).unwrap();
        assert_eq!(r.tensor().components(&c2).unwrap(), vec![Expr::from(0), Expr::from(0)]);
        let r = run(vec![y.clone().gamma(), zero.clone()], vec![zero.clone(), Expr::from(1)], &c2).unwrap();
        assert_eq!(r.tensor().components(&c2).unwrap(), vec![Expr::from(0), Expr::from(0)]);
        let r = run(vec![y.clone(), x.clone()], vec![x.clone() * y.clone(), zero.clone()], &c2).unwrap();
        let v = r.tensor().components(&c2).unwrap();
        assert_value(&v[0], "x^2 + y^2");
        assert_value(&v[1], "y*x");
        let c1 = Chart::new("c", 1, vec!["x"]).unwrap();
        assert!(matches!(run(dx(), vec![zero.clone(), x.clone()], &c1), Err(ManifoldError::InvalidChart(_))));
        assert!(matches!(run(dx(), vec![zero.clone(), rs.clone()], &c2), Err(ManifoldError::NotImplemented(_))));
        let r = run(vec![zero.clone(), Expr::from(1)], vec![zero.clone(), rs.clone()], &c2).unwrap();
        let v = r.tensor().components(&c2).unwrap();
        assert_eq!(v[0], Expr::from(0));
        let expected = Expr::Function("root_sum".into(), vec![Arc::new(t - x), Arc::new(Expr::from(0))]);
        assert_eq!(v[1], expected);
    }
}
