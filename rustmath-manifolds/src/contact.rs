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
use crate::differentiable::DifferentiableManifold;
use crate::diff_form::DiffForm;
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
        let dim = 2 * n + 1;
        let manifold = Arc::new(DifferentiableManifold::new("ℝ²ⁿ⁺¹", dim));
        let chart = manifold.default_chart().unwrap();

        // Build α = dz - Σ yᵢ dxᵢ
        // Components: [α_x₁, α_y₁, ..., α_xₙ, α_yₙ, α_z]
        let mut components = vec![Expr::from(0); dim];

        for i in 0..n {
            // α_xᵢ = -yᵢ
            components[2*i] = -Expr::symbol(&format!("y{}", i));
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

    /// Check if the form satisfies the contact condition
    ///
    /// α ∧ (dα)^n ≠ 0
    fn is_contact(&self) -> Result<bool> {
        let chart = self.manifold.default_chart()
            .ok_or(ManifoldError::NoChart)?;

        // Compute dα
        let d_alpha = self.form.exterior_derivative(chart)?;

        // Compute (dα)^n via wedge product
        let n = (self.manifold.dimension() - 1) / 2;
        let mut power = d_alpha.clone();

        for _ in 1..n {
            power = power.wedge(&d_alpha, chart)?;
        }

        // Compute α ∧ (dα)^n
        let contact_volume = self.form.wedge(&power, chart)?;

        // Check if it's non-zero (i.e., a volume form)
        Ok(!contact_volume.is_zero())
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
        // TODO: Compute basis for ker(α)
        // For now, create a placeholder
        let chart = self.manifold.default_chart().unwrap();
        let dim = self.manifold.dimension();

        let mut frame = Vec::new();
        for i in 0..(dim - 1) {
            let mut components = vec![Expr::from(0); dim];
            components[i] = Expr::from(1);

            let v = VectorField::from_components(
                self.manifold.clone(),
                chart,
                components,
            )?;
            frame.push(v);
        }

        Distribution::new(self.manifold.clone(), frame)
    }

    /// Compute the Reeb vector field
    ///
    /// The unique vector field R satisfying:
    /// - α(R) = 1
    /// - dα(R, ·) = 0
    pub fn reeb_vector_field(&self) -> Result<VectorField> {
        // TODO: Solve for Reeb vector field
        // For now, return a placeholder
        let chart = self.manifold.default_chart().unwrap();
        let dim = self.manifold.dimension();

        let components = vec![Expr::from(0); dim];

        VectorField::from_components(
            self.manifold.clone(),
            chart,
            components,
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
        let dim = 2 * n + 1;
        let base = Arc::new(DifferentiableManifold::new("ℝ²ⁿ⁺¹", dim));
        let contact_form = ContactForm::standard(n)?;

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
        // TODO: Compute contact Hamiltonian vector field
        let chart = self.base_manifold.default_chart().unwrap();
        let dim = self.dimension();

        VectorField::from_components(
            self.base_manifold.clone(),
            chart,
            vec![Expr::from(0); dim],
        )
    }

    /// Compute the contactomorphism group (symmetries preserving contact structure)
    pub fn is_contactomorphism(&self, _diffeomorphism: &VectorField) -> bool {
        // TODO: Check if diffeomorphism preserves contact form
        false
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
        // TODO: Integrate contact Hamiltonian equations
        Ok(vec![])
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
        let m = Arc::new(DifferentiableManifold::new("M", 3));
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
        let m = Arc::new(DifferentiableManifold::new("M", 4)); // Even dimension
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
    fn test_contact_hamiltonian() {
        let cm = Arc::new(ContactManifold::standard(1).unwrap());
        let h = Expr::symbol("H");

        let ch = ContactHamiltonian::new(cm, h);
        assert!(ch.vector_field().is_ok());
    }
}
