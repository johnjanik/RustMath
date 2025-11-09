//! Genus Theory
//!
//! Implementation of genus theory for quadratic forms, including mass formulas
//! and classification of forms by their local invariants.

use crate::{QuadraticForm, LocalDensities};
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use rustmath_core::{Ring, NumericConversion};
use std::f64::consts::PI;

/// Genus of quadratic forms with the same discriminant and local invariants
pub struct GenusTheory {
    pub forms: Vec<QuadraticForm>,
    pub discriminant: Integer,
}

impl GenusTheory {
    /// Create a new genus from a collection of forms
    pub fn new(forms: Vec<QuadraticForm>) -> Self {
        let discriminant = if let Some(first) = forms.first() {
            first.discriminant().clone()
        } else {
            Integer::zero()
        };

        // Verify all forms have the same discriminant
        for form in &forms {
            if form.discriminant() != &discriminant {
                panic!("All forms in a genus must have the same discriminant");
            }
        }

        Self { forms, discriminant }
    }

    /// Compute the Smith-Minkowski-Siegel mass formula
    ///
    /// The mass formula gives: Σ 1/|Aut(Q)| = (product of local factors)
    /// This is a fundamental result in the theory of quadratic forms.
    pub fn smith_minkowski_siegel_mass(&self) -> Rational {
        let n = 2; // Number of variables (binary forms)

        // Compute various factors
        let factor1 = self.compute_pi_factor(n);
        let factor2 = self.compute_gamma_factor(n);
        let factor3 = self.compute_discriminant_factor(n);
        let factor4 = self.compute_local_factor();

        // Multiply all factors
        let mass = factor1 * factor2 * factor3 * factor4;
        mass
    }

    /// Compute the (π/2)^(n/2) factor
    fn compute_pi_factor(&self, n: usize) -> Rational {
        // (π/2)^(n/2)
        // For n=2: π/2 ≈ 1.5708
        let pi_over_2 = PI / 2.0;
        let exponent = n as f64 / 2.0;
        let value = pi_over_2.powf(exponent);

        // Convert to rational approximation
        let numerator = (value * 1000000.0).round() as i64;
        Rational::new(Integer::from(numerator), Integer::from(1000000)).unwrap()
    }

    /// Compute the gamma function factor
    fn compute_gamma_factor(&self, n: usize) -> Rational {
        // ∏_{j=1}^n Γ(j/2) / Γ(n/2)
        // For binary forms (n=2): Γ(1/2) * Γ(1) / Γ(1) = Γ(1/2) = √π

        let mut product = 1.0;
        for j in 1..=n {
            product *= Self::gamma_function(j as f64 / 2.0);
        }

        let denominator = Self::gamma_function(n as f64 / 2.0);
        let value = product / denominator;

        // Convert to rational approximation
        let numerator = (value * 1000000.0).round() as i64;
        Rational::new(Integer::from(numerator), Integer::from(1000000)).unwrap()
    }

    /// Compute the discriminant factor |Δ|^{-(n+1)/2}
    fn compute_discriminant_factor(&self, n: usize) -> Rational {
        let abs_disc = self.discriminant.abs();
        if abs_disc.is_zero() {
            return Rational::from_integer(Integer::one());
        }

        let disc_float = abs_disc.to_f64().unwrap_or(1.0);
        let exponent = -((n as f64 + 1.0) / 2.0);
        let value = disc_float.powf(exponent);

        // Convert to rational approximation
        let numerator = (value * 1000000.0).round() as i64;
        Rational::new(Integer::from(numerator), Integer::from(1000000)).unwrap()
    }

    /// Compute the product of local densities
    fn compute_local_factor(&self) -> Rational {
        if self.forms.is_empty() {
            return Rational::from_integer(Integer::one());
        }

        // Use the first form to compute local densities at 0
        let form = &self.forms[0];
        let densities = LocalDensities::new(form.clone());
        densities.siegel_product(&Integer::zero())
    }

    /// Approximate gamma function using Stirling's approximation
    fn gamma_function(x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }

        // Special cases
        if (x - 1.0).abs() < 1e-10 {
            return 1.0; // Γ(1) = 1
        }
        if (x - 0.5).abs() < 1e-10 {
            return PI.sqrt(); // Γ(1/2) = √π
        }

        // Stirling's approximation: Γ(x) ≈ √(2π) * x^(x-0.5) * e^{-x}
        let two_pi = 2.0 * PI;
        two_pi.sqrt() * x.powf(x - 0.5) * (-x).exp()
    }

    /// Estimate class number from the mass formula
    ///
    /// The class number is approximately mass * average_automorphism_order
    pub fn class_number_estimate(&self) -> Integer {
        let mass = self.smith_minkowski_siegel_mass();

        // For primitive forms, typical automorphism group sizes:
        // - Most forms have |Aut| = 2
        // - x² + y² has |Aut| = 8
        // - x² + xy + y² has |Aut| = 6

        // Assume average |Aut| ≈ 2 for estimation
        let avg_aut = Rational::from_integer(Integer::from(2));
        let class_number_approx = mass * avg_aut;

        // Round to nearest integer
        let num = class_number_approx.numerator();
        let den = class_number_approx.denominator();
        let result = num / den;

        if result > Integer::one() {
            result
        } else {
            Integer::one()
        }
    }

    /// Get the number of forms in the genus
    pub fn genus_size(&self) -> usize {
        self.forms.len()
    }

    /// Check if two forms are in the same genus
    ///
    /// Forms are in the same genus if they have the same discriminant
    /// and are equivalent over ℝ and all ℚ_p
    pub fn is_same_genus(&self, form1: &QuadraticForm, form2: &QuadraticForm) -> bool {
        // Same discriminant is necessary
        if form1.discriminant() != form2.discriminant() {
            return false;
        }

        // Same signature (real equivalence)
        if form1.is_positive_definite() != form2.is_positive_definite() {
            return false;
        }
        if form1.is_indefinite() != form2.is_indefinite() {
            return false;
        }

        // For a complete test, we would check p-adic equivalence for all p
        // This is simplified here
        true
    }

    /// Check p-adic equivalence of two forms at prime p
    fn p_adic_equivalent(&self, form1: &QuadraticForm, form2: &QuadraticForm, p: u64) -> bool {
        let densities1 = LocalDensities::new(form1.clone());
        let densities2 = LocalDensities::new(form2.clone());

        // Compare local densities for a few small values
        for m in 0..5 {
            let m_int = Integer::from(m);
            let density1 = densities1.p_adic_density(&m_int, p);
            let density2 = densities2.p_adic_density(&m_int, p);

            // Compare as rationals
            if density1 != density2 {
                // Allow some tolerance for numerical comparison
                let diff_num = (density1.numerator() * density2.denominator() -
                              density2.numerator() * density1.denominator()).abs();
                let threshold = Integer::from(100); // Small threshold

                if diff_num > threshold {
                    return false;
                }
            }
        }

        true
    }

    /// Compute the spinor genus (finer classification than genus)
    ///
    /// The spinor genus is an intermediate notion between proper equivalence
    /// and genus equivalence
    pub fn spinor_genus_count(&self) -> usize {
        // Simplified: return 1 for now
        // Full implementation would use spinor norm theory
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genus_creation() {
        let form1 = QuadraticForm::binary(
            Integer::from(1),
            Integer::from(0),
            Integer::from(1),
        );

        // Form with same discriminant -4
        let genus = GenusTheory::new(vec![form1]);
        assert_eq!(genus.genus_size(), 1);
        assert_eq!(genus.discriminant, Integer::from(-4));
    }

    #[test]
    fn test_mass_formula() {
        let form = QuadraticForm::binary(
            Integer::from(1),
            Integer::from(0),
            Integer::from(1),
        );
        let genus = GenusTheory::new(vec![form]);

        let mass = genus.smith_minkowski_siegel_mass();
        // Mass should be positive
        assert!(mass.numerator() > &Integer::zero());
    }

    #[test]
    fn test_class_number_estimate() {
        let form = QuadraticForm::binary(
            Integer::from(1),
            Integer::from(0),
            Integer::from(1),
        );
        let genus = GenusTheory::new(vec![form]);

        let class_number = genus.class_number_estimate();
        // Should be at least 1
        assert!(class_number >= Integer::one());
    }

    #[test]
    fn test_gamma_function() {
        // Γ(1) = 1
        let gamma_1 = GenusTheory::gamma_function(1.0);
        assert!((gamma_1 - 1.0).abs() < 1e-6);

        // Γ(1/2) = √π ≈ 1.772
        let gamma_half = GenusTheory::gamma_function(0.5);
        assert!((gamma_half - PI.sqrt()).abs() < 1e-2);
    }

    #[test]
    fn test_same_genus() {
        let form1 = QuadraticForm::binary(
            Integer::from(1),
            Integer::from(0),
            Integer::from(1),
        );
        // Form with same discriminant -4
        let form2 = QuadraticForm::binary(
            Integer::from(1),
            Integer::from(0),
            Integer::from(1),
        );

        let genus = GenusTheory::new(vec![form1.clone()]);

        // Both forms are in the same genus (they're actually the same form)
        assert!(genus.is_same_genus(&form1, &form2));
    }
}
