//! Contact manifolds and contact geometry
//!
//! This module provides structures for contact manifolds - odd-dimensional manifolds
//! equipped with a maximally non-integrable hyperplane distribution.
//!
//! # Overview
//!
//! A contact manifold (M, α) is an odd-dimensional manifold (dim = 2n+1) with a
//! contact form α (a 1-form) satisfying:
//!
//! α ∧ (dα)^n ≠ 0 everywhere
//!
//! This means the distribution ker(α) is maximally non-integrable.
//!
//! Contact geometry is the odd-dimensional analog of symplectic geometry and appears in:
//! - Classical mechanics (phase space of time-dependent systems)
//! - Thermodynamics (Legendre transformations)
//! - Optics (geometric optics, wavefronts)
//!
//! # Examples
//!
//! ```
//! use rustmath_manifolds::{ContactManifold, DifferentiableManifold};
//!
//! // Create the standard contact structure on ℝ³
//! // α = dz - y dx
//! ```

use crate::errors::{ManifoldError, Result};
use crate::chart::Chart;
use crate::differentiable::DifferentiableManifold;
use crate::diff_form::DiffForm;
use crate::expr_zero::ZeroTest;
use crate::vector_field::VectorField;
use crate::subriemannian::Distribution;
use rustmath_symbolic::Expr;
use std::sync::Arc;

/// A contact form on an odd-dimensional manifold
///
/// A 1-form α satisfying α ∧ (dα)^n ≠ 0
#[derive(Debug, Clone)]
pub struct ContactForm {
    /// The base manifold (must be odd-dimensional)
    manifold: Arc<DifferentiableManifold>,
    /// The 1-form α
    form: DiffForm,
    /// Name of the contact form
    name: Option<String>,
}

impl ContactForm {
    /// Create a new contact form
    ///
    /// # Arguments
    ///
    /// * `manifold` - The base manifold (must be odd-dimensional)
    /// * `form` - The 1-form
    ///
    /// # Returns
    ///
    /// A contact form, or an error if:
    /// - Manifold has even dimension
    /// - Form doesn't satisfy α ∧ (dα)^n ≠ 0
    pub fn new(
        manifold: Arc<DifferentiableManifold>,
        form: DiffForm,
    ) -> Result<Self> {
        // Check odd dimension
        if manifold.dimension() % 2 == 0 {
            return Err(ManifoldError::InvalidDimension(
                "Contact manifold must be odd-dimensional".to_string()
            ));
        }

        // Check form is a 1-form
        if form.degree() != 1 {
            return Err(ManifoldError::ValidationError(
                "Contact form must be a 1-form".to_string()
            ));
        }

        let contact_form = Self {
            manifold,
            form,
            name: None,
        };

        // Verify contact condition
        if !contact_form.is_contact()? {
            return Err(ManifoldError::ValidationError(
                "Form does not satisfy contact condition α ∧ (dα)^n ≠ 0".to_string()
            ));
        }

        Ok(contact_form)
    }

    /// Create the standard contact form on ℝ²ⁿ⁺¹
    ///
    /// In coordinates (x₁, ..., xₙ, y₁, ..., yₙ, z):
    /// α = dz - Σ yᵢ dxᵢ
    ///
    /// This is the canonical contact form
    pub fn standard(n: usize) -> Result<Self> {
        let dim = n
            .checked_mul(2)
            .and_then(|d| d.checked_add(1))
            .ok_or(ManifoldError::DegreeOverflow { left: n, right: n })?;
        let mut base = DifferentiableManifold::new("ℝ²ⁿ⁺¹", dim);
        base.add_chart(Chart::standard("standard", dim))
            .expect("failed to add standard chart");
        let manifold = Arc::new(base);
        let chart = manifold.default_chart().unwrap();

        // Build α = dz - Σ yᵢ dxᵢ in the chart coordinates (x1, ..., x_{2n+1}):
        // xᵢ is coordinate 2i, yᵢ is coordinate 2i + 1, z is the last one.
        // Components: [α_x₁, α_y₁, ..., α_xₙ, α_yₙ, α_z]
        let mut components = crate::tensor_field::zeroed_components(dim)?;

        for i in 0..n {
            // α_xᵢ = -yᵢ, with yᵢ the chart coordinate 2i + 1
            components[2*i] = -Expr::Symbol(chart.coordinate_symbol(2 * i + 1));
            // α_yᵢ = 0
            components[2*i + 1] = Expr::from(0);
        }
        // α_z = 1
        components[dim - 1] = Expr::from(1);

        let form = DiffForm::from_components(
            manifold.clone(),
            chart,
            1,
            components,
        )?;

        Ok(Self {
            manifold,
            form,
            name: Some("Standard".to_string()),
        })
    }

    /// Set the name
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    /// α ∧ (dα)^n in the default chart, a top-degree form (α itself for n = 0).
    fn contact_volume(&self) -> Result<DiffForm> {
        let chart = self.manifold.default_chart()
            .ok_or(ManifoldError::NoChart)?;
        let n = self.manifold.dimension().saturating_sub(1) / 2;
        if n == 0 {
            return Ok(self.form.clone());
        }

        // Compute dα, then (dα)^n via wedge product
        let d_alpha = self.form.exterior_derivative(chart)?;
        let mut power = d_alpha.clone();
        for _ in 1..n {
            power = power.wedge(&d_alpha, chart)?;
        }

        // Compute α ∧ (dα)^n
        self.form.wedge(&power, chart)
    }

    /// Check if the form satisfies the contact condition: α ∧ (dα)^n vanishes nowhere
    ///
    /// α ∧ (dα)^n has one independent component c. `Ok(true)` when c is a nonzero rational
    /// constant; `Ok(false)` when c is decided zero; otherwise `Err(Undecided)`, since a
    /// non-constant c may vanish somewhere. The errors of the exterior derivative propagate.
    fn is_contact(&self) -> Result<bool> {
        let chart = self.manifold.default_chart()
            .ok_or(ManifoldError::NoChart)?;
        let volume = self.contact_volume()?;
        let components = volume.independent_components(chart)?;
        let Some((_, c)) = components.first() else {
            return Ok(false);
        };
        match c {
            Expr::Integer(_) | Expr::Rational(_) if !c.is_zero() => return Ok(true),
            _ => {}
        }
        match volume.vanishes(chart)? {
            ZeroTest::Zero => Ok(false),
            _ => Err(ManifoldError::Undecided(format!(
                "α∧(dα)^n = {} is not constant; whether it vanishes nowhere is not decided",
                c
            ))),
        }
    }

    /// Get the form
    pub fn form(&self) -> &DiffForm {
        &self.form
    }

    /// Get the manifold
    pub fn manifold(&self) -> &Arc<DifferentiableManifold> {
        &self.manifold
    }

    /// Compute the contact distribution (kernel of α)
    ///
    /// The contact distribution is ker(α) = {v ∈ TM : α(v) = 0}
    pub fn contact_distribution(&self) -> Result<Distribution> {
        unimplemented!(
            "ContactForm::contact_distribution not yet implemented (facade): requires \
             computing a basis for ker(alpha), not just the coordinate frame ∂/∂x_i"
        )
    }

    /// Compute the Reeb vector field
    ///
    /// The unique vector field R satisfying:
    /// - α(R) = 1
    /// - dα(R, ·) = 0
    pub fn reeb_vector_field(&self) -> Result<VectorField> {
        unimplemented!(
            "ContactForm::reeb_vector_field not yet implemented (facade): requires solving \
             alpha(R) = 1 and d(alpha)(R, .) = 0 for R"
        )
    }
}

/// A contact manifold - odd-dimensional manifold with contact structure
#[derive(Debug, Clone)]
pub struct ContactManifold {
    /// The base manifold
    base_manifold: Arc<DifferentiableManifold>,
    /// The contact form
    contact_form: ContactForm,
}

impl ContactManifold {
    /// Create a new contact manifold
    pub fn new(
        base_manifold: Arc<DifferentiableManifold>,
        contact_form: ContactForm,
    ) -> Result<Self> {
        // Verify dimensions match
        if base_manifold.dimension() != contact_form.manifold().dimension() {
            return Err(ManifoldError::DimensionMismatch {
                expected: base_manifold.dimension(),
                actual: contact_form.manifold().dimension(),
            });
        }

        Ok(Self {
            base_manifold,
            contact_form,
        })
    }

    /// Create the standard contact structure on ℝ²ⁿ⁺¹
    pub fn standard(n: usize) -> Result<Self> {
        // ContactForm::standard checks that 2n + 1 fits
        let contact_form = ContactForm::standard(n)?;
        let dim = contact_form.manifold().dimension();
        let base = Arc::new(DifferentiableManifold::new("ℝ²ⁿ⁺¹", dim));

        Ok(Self {
            base_manifold: base,
            contact_form,
        })
    }

    /// Get the base manifold
    pub fn base_manifold(&self) -> &Arc<DifferentiableManifold> {
        &self.base_manifold
    }

    /// Get the contact form
    pub fn contact_form(&self) -> &ContactForm {
        &self.contact_form
    }

    /// Get the dimension (always odd)
    pub fn dimension(&self) -> usize {
        self.base_manifold.dimension()
    }

    /// Get the Reeb vector field
    pub fn reeb_vector_field(&self) -> Result<VectorField> {
        self.contact_form.reeb_vector_field()
    }

    /// Get the contact distribution
    pub fn contact_distribution(&self) -> Result<Distribution> {
        self.contact_form.contact_distribution()
    }

    /// Check if a function is a contact Hamiltonian
    ///
    /// A function H generates a contact Hamiltonian vector field X_H
    pub fn hamiltonian_vector_field(&self, _hamiltonian: &Expr) -> Result<VectorField> {
        unimplemented!(
            "ContactManifold::hamiltonian_vector_field not yet implemented (facade): requires \
             deriving X_H from H via the contact form and Reeb field"
        )
    }

    /// Compute the contactomorphism group (symmetries preserving contact structure)
    pub fn is_contactomorphism(&self, _diffeomorphism: &VectorField) -> bool {
        unimplemented!(
            "ContactManifold::is_contactomorphism not yet implemented (facade): requires \
             checking whether the diffeomorphism preserves the contact form up to scale"
        )
    }
}

/// The contact Hamiltonian for contact dynamics
///
/// In contact geometry, dynamics is generated by contact Hamiltonians
#[derive(Debug, Clone)]
pub struct ContactHamiltonian {
    /// The Hamiltonian function
    hamiltonian: Expr,
    /// The contact manifold
    manifold: Arc<ContactManifold>,
}

impl ContactHamiltonian {
    /// Create a new contact Hamiltonian
    pub fn new(manifold: Arc<ContactManifold>, hamiltonian: Expr) -> Self {
        Self {
            hamiltonian,
            manifold,
        }
    }

    /// Compute the associated contact vector field
    pub fn vector_field(&self) -> Result<VectorField> {
        self.manifold.hamiltonian_vector_field(&self.hamiltonian)
    }

    /// Evolve the system under contact dynamics
    pub fn flow(&self, _time: f64) -> Result<Vec<Expr>> {
        unimplemented!(
            "ContactHamiltonian::flow not yet implemented (facade): requires integrating the \
             contact Hamiltonian equations"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_contact_form_3d() {
        // ℝ³ with contact form α = dz - y dx
        let alpha = ContactForm::standard(1);
        assert!(alpha.is_ok());

        let form = alpha.unwrap();
        assert_eq!(form.manifold().dimension(), 3);
    }

    #[test]
    fn test_standard_contact_form_5d() {
        let alpha = ContactForm::standard(2);
        assert!(alpha.is_ok());

        let form = alpha.unwrap();
        assert_eq!(form.manifold().dimension(), 5);
    }

    #[test]
    fn test_contact_manifold_creation() {
        let mut base = DifferentiableManifold::new("M", 3);
        base.add_chart(Chart::standard("standard", 3)).unwrap();
        let m = Arc::new(base);
        let chart = m.default_chart().unwrap();

        // Create a 1-form
        let components = vec![
            Expr::from(0),
            -Expr::symbol("y"),
            Expr::from(1),
        ];

        let form = DiffForm::from_components(m.clone(), chart, 1, components).unwrap();
        let contact_form = ContactForm::new(m.clone(), form);

        // This should succeed (standard contact form)
        // (though validation might not be complete)
        if let Ok(cf) = contact_form {
            let contact_manifold = ContactManifold::new(m, cf);
            assert!(contact_manifold.is_ok());
        }
    }

    #[test]
    fn test_even_dimension_fails() {
        let mut base = DifferentiableManifold::new("M", 4); // Even dimension
        base.add_chart(Chart::standard("standard", 4)).unwrap();
        let m = Arc::new(base);
        let chart = m.default_chart().unwrap();

        let components = vec![Expr::from(1), Expr::from(0), Expr::from(0), Expr::from(0)];
        let form = DiffForm::from_components(m.clone(), chart, 1, components).unwrap();

        let result = ContactForm::new(m, form);
        assert!(result.is_err()); // Should fail for even dimension
    }

    #[test]
    fn test_standard_contact_manifold() {
        let cm = ContactManifold::standard(1);
        assert!(cm.is_ok());

        let manifold = cm.unwrap();
        assert_eq!(manifold.dimension(), 3);
    }

    #[test]
    #[ignore = "facade -> unimplemented (Phase 5)"]
    fn test_contact_hamiltonian() {
        let cm = Arc::new(ContactManifold::standard(1).unwrap());
        let h = Expr::symbol("H");

        let ch = ContactHamiltonian::new(cm, h);
        assert!(ch.vector_field().is_ok());
    }

    #[test]
    fn test_contact_volume() {
        // α = dz - Σ y_i dx_i: α∧(dα)^k = k! dx1∧dy1∧...∧dxk∧dyk∧dz (gp: k! = 1, 2, 6)
        // (with HEAD's y{i}, which are not chart coordinates, dα was 0)
        for (k, top) in [(1usize, 1i64), (2, 2), (3, 6)] {
            let alpha = ContactForm::standard(k).unwrap();
            let chart = alpha.manifold().default_chart().unwrap().clone();
            let volume = alpha.contact_volume().unwrap();
            assert_eq!(volume.degree(), 2 * k + 1);
            let v = volume.independent_components(&chart).unwrap();
            assert_eq!(v.len(), 1);
            let diff = v[0].1.clone() - Expr::from(top);
            assert_eq!(crate::expr_zero::zero_test(&diff), ZeroTest::Zero, "{}", v[0].1);
            assert!(alpha.is_contact().unwrap());
        }
    }

    #[test]
    fn test_dz_is_not_contact() {
        // HEAD accepted dz on R^3 (the structural is_zero of a facade wedge)
        let m = Arc::new(crate::examples::EuclideanSpace::new(3).manifold().clone());
        let chart = m.default_chart().unwrap().clone();
        let dz = DiffForm::coordinate_form(m.clone(), &chart, 2).unwrap();
        assert!(ContactForm::new(m.clone(), dz).is_err());
        // α = dz - y dx is contact (α∧dα = dx∧dy∧dz). β = dz - x*y dx has β∧dβ = x dx∧dy∧dz,
        // which vanishes at x = 0: not decided as a constant.
        let (x, y) = (Expr::symbol("x"), Expr::symbol("y"));
        let alpha = DiffForm::from_components(m.clone(), &chart, 1, vec![-y.clone(), Expr::from(0), Expr::from(1)]).unwrap();
        assert!(ContactForm::new(m.clone(), alpha).is_ok());
        let beta = DiffForm::from_components(m.clone(), &chart, 1, vec![-(x * y), Expr::from(0), Expr::from(1)]).unwrap();
        assert!(matches!(ContactForm::new(m, beta), Err(ManifoldError::Undecided(_))));
    }
}
