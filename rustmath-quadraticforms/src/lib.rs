//! RustMath Quadratic Forms
//!
//! Comprehensive implementation of quadratic forms theory including:
//! - Binary and unary quadratic forms
//! - Reduction algorithms for positive definite forms
//! - Theta series (representation numbers)
//! - Local densities (p-adic and real)
//! - Genus theory and mass formulas
//! - Connections to modular forms and automorphic representations
//!
//! # Overview
//!
//! A quadratic form is a homogeneous polynomial of degree 2. This crate focuses
//! primarily on binary quadratic forms of the shape:
//!
//! Q(x, y) = ax² + bxy + cy²
//!
//! where a, b, c are integers.
//!
//! # Key Concepts
//!
//! ## Discriminant
//! The discriminant Δ = b² - 4ac determines many properties of the form:
//! - Δ < 0: definite form (positive if a > 0, negative if a < 0)
//! - Δ > 0: indefinite form
//! - Δ = 0: degenerate form
//!
//! ## Reduction
//! A positive definite form [a, b, c] is reduced if:
//! - |b| ≤ a ≤ c
//! - If |b| = a or a = c, then b ≥ 0
//!
//! ## Theta Series
//! The theta series θ_Q(q) = Σ r_Q(n) q^n counts representations:
//! r_Q(n) = #{(x,y) ∈ ℤ² : Q(x,y) = n}
//!
//! ## Local Densities
//! Local densities measure the "density" of representations in p-adic and
//! real completions of ℚ. The Siegel formula relates global representation
//! numbers to products of local densities.
//!
//! # Examples
//!
//! ```
//! use rustmath_quadraticforms::{QuadraticForm, ThetaSeries};
//! use rustmath_integers::Integer;
//!
//! // Create the form x² + y²
//! let form = QuadraticForm::binary(
//!     Integer::from(1),
//!     Integer::from(0),
//!     Integer::from(1),
//! );
//!
//! assert!(form.is_positive_definite());
//! assert!(form.is_primitive());
//!
//! // Compute theta series
//! let theta = ThetaSeries::new(form.clone(), 20);
//!
//! // How many ways to write 5 as x² + y²?
//! let r_5 = theta.representation_count(&Integer::from(5));
//! assert_eq!(r_5, Integer::from(8)); // (±1, ±2), (±2, ±1)
//! ```

pub mod quadratic_form;
pub mod theta_series;
pub mod local_densities;
pub mod genus_theory;
pub mod advanced;

pub use quadratic_form::QuadraticForm;
pub use theta_series::{ThetaSeries, ThetaModularProperties};
pub use local_densities::LocalDensities;
pub use genus_theory::GenusTheory;
pub use advanced::{QuadraticFormAnalytics, AutomorphicProperties};

/// Run comprehensive examples demonstrating the library's capabilities
pub fn run_examples() {
    println!("=== RustMath Quadratic Forms Examples ===\n");

    example_1_basic_forms();
    example_2_theta_series();
    example_3_local_densities();
    example_4_genus_theory();
    example_5_advanced_analytics();
}

fn example_1_basic_forms() {
    use rustmath_integers::Integer;

    println!("Example 1: Basic Quadratic Forms");
    println!("{}", "-".repeat(50));

    let forms = vec![
        ("x² + y²", QuadraticForm::binary(
            Integer::from(1),
            Integer::from(0),
            Integer::from(1),
        )),
        ("x² + xy + y²", QuadraticForm::binary(
            Integer::from(1),
            Integer::from(1),
            Integer::from(1),
        )),
        ("2x² + 3y²", QuadraticForm::binary(
            Integer::from(2),
            Integer::from(0),
            Integer::from(3),
        )),
        ("x² - y²", QuadraticForm::binary(
            Integer::from(1),
            Integer::from(0),
            Integer::from(-1),
        )),
    ];

    for (name, form) in forms {
        println!("\nForm: {} = {}", name, form);
        println!("  Discriminant: {}", form.discriminant());
        println!("  Positive definite: {}", form.is_positive_definite());
        println!("  Indefinite: {}", form.is_indefinite());
        println!("  Primitive: {}", form.is_primitive());

        if form.is_positive_definite() {
            let reduced = form.reduced_form();
            println!("  Reduced form: {}", reduced);

            let class_num = form.class_number_estimate();
            println!("  Class number estimate: {}", class_num);
        }

        // Evaluate at a point
        let value = form.evaluate(&Integer::from(2), &Integer::from(3));
        println!("  Q(2, 3) = {}", value);
    }

    println!();
}

fn example_2_theta_series() {
    use rustmath_integers::Integer;

    println!("\nExample 2: Theta Series");
    println!("{}", "-".repeat(50));

    // x² + y²
    let form = QuadraticForm::binary(
        Integer::from(1),
        Integer::from(0),
        Integer::from(1),
    );
    let theta = ThetaSeries::new(form, 30);

    println!("\nForm: x² + y²");
    println!("Representation counts r(n) for n ≤ 30:");

    let mut count = 0;
    for (n, r_n) in theta.generating_function_coefficients() {
        if &n <= &Integer::from(30) {
            println!("  r({:2}) = {:3}", n, r_n);
            count += 1;
            if count > 15 {
                println!("  ...");
                break;
            }
        }
    }

    let props = theta.modular_form_properties();
    println!("\nModular form properties:");
    println!("  Weight: {}", props.weight);
    println!("  Level: {}", props.level);
    println!("  Character: {}", props.character);

    println!();
}

fn example_3_local_densities() {
    use rustmath_integers::Integer;

    println!("\nExample 3: Local Densities");
    println!("{}", "-".repeat(50));

    let form = QuadraticForm::binary(
        Integer::from(1),
        Integer::from(0),
        Integer::from(1),
    );
    let densities = LocalDensities::new(form);

    println!("\nForm: x² + y²");
    println!("Local densities for representing various integers:\n");

    for m in [0, 1, 2, 3, 4, 5, 7, 10] {
        let m_int = Integer::from(m);
        println!("m = {}:", m);

        let real_dens = densities.real_density(&m_int);
        println!("  Real density: {}/{}", real_dens.numerator(), real_dens.denominator());

        for p in [2, 3, 5] {
            let p_adic_dens = densities.p_adic_density(&m_int, p);
            println!("  {}-adic density: {}/{}",
                p, p_adic_dens.numerator(), p_adic_dens.denominator());
        }

        let siegel = densities.siegel_product(&m_int);
        println!("  Siegel product: {}/{}", siegel.numerator(), siegel.denominator());
        println!();
    }
}

fn example_4_genus_theory() {
    use rustmath_integers::Integer;

    println!("\nExample 4: Genus Theory and Mass Formulas");
    println!("{}", "-".repeat(50));

    // Forms with discriminant -4
    let forms = vec![
        QuadraticForm::binary(Integer::from(1), Integer::from(0), Integer::from(1)),
    ];

    let genus = GenusTheory::new(forms);

    println!("\nGenus with discriminant {}", genus.discriminant);
    println!("Number of forms in genus: {}", genus.genus_size());

    let mass = genus.smith_minkowski_siegel_mass();
    println!("Smith-Minkowski-Siegel mass: {}/{}",
        mass.numerator(), mass.denominator());

    let class_number = genus.class_number_estimate();
    println!("Estimated class number: {}", class_number);

    println!();
}

fn example_5_advanced_analytics() {
    use rustmath_integers::Integer;

    println!("\nExample 5: Advanced Analytics");
    println!("{}", "-".repeat(50));

    let form = QuadraticForm::binary(
        Integer::from(1),
        Integer::from(0),
        Integer::from(1),
    );
    let analytics = QuadraticFormAnalytics::new(form, 50);

    println!("\nForm: x² + y²");

    let props = analytics.automorphic_properties();
    println!("Automorphic properties:");
    println!("  Is modular: {}", props.is_modular);
    println!("  Weight: {}", props.weight);
    println!("  Level: {}", props.level);
    println!("  Is Hecke eigenform: {}", props.is_hecke_eigenform);
    println!("  Has cuspidal part: {}", props.has_cuspidal_part);

    println!("\nSatake parameters at small primes:");
    for p in [2, 3, 5, 7, 11] {
        let satake = analytics.satake_parameters(p);
        println!("  p={}: [{:.4}, {:.4}]", p, satake[0], satake[1]);

        let l_coeff = analytics.l_function_coefficient(p);
        println!("    L-function coefficient: {:.4}", l_coeff);
    }

    let growth = analytics.estimate_growth_rate();
    println!("\nEstimated coefficient growth rate: {:.4}", growth);

    println!("\nSiegel-Weil predictions for small integers:");
    for m in [1, 2, 5, 10] {
        let m_int = Integer::from(m);
        let actual = analytics.theta_series.representation_count(&m_int);
        let prediction = analytics.siegel_weil_prediction(&m_int);
        println!("  m={}: actual={}, prediction={}/{}",
            m, actual, prediction.numerator(), prediction.denominator());
    }

    println!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;
    use rustmath_core::Ring;

    #[test]
    fn test_basic_form_creation() {
        let form = QuadraticForm::binary(
            Integer::from(1),
            Integer::from(0),
            Integer::from(1),
        );

        assert_eq!(form.a, Integer::from(1));
        assert_eq!(form.b, Integer::from(0));
        assert_eq!(form.c, Integer::from(1));
        assert!(form.is_positive_definite());
    }

    #[test]
    fn test_theta_series_sum_of_two_squares() {
        // x² + y²
        let form = QuadraticForm::binary(
            Integer::from(1),
            Integer::from(0),
            Integer::from(1),
        );
        let theta = ThetaSeries::new(form, 20);

        // r(0) = 1: (0,0)
        assert_eq!(theta.coefficient(&Integer::from(0)), Integer::from(1));

        // r(1) = 4: (±1,0), (0,±1)
        assert_eq!(theta.coefficient(&Integer::from(1)), Integer::from(4));

        // r(2) = 4: (±1,±1)
        assert_eq!(theta.coefficient(&Integer::from(2)), Integer::from(4));

        // r(3) = 0: no solutions
        assert_eq!(theta.coefficient(&Integer::from(3)), Integer::from(0));

        // r(5) = 8: (±1,±2), (±2,±1)
        assert_eq!(theta.coefficient(&Integer::from(5)), Integer::from(8));
    }

    #[test]
    fn test_local_densities_positive() {
        let form = QuadraticForm::binary(
            Integer::from(1),
            Integer::from(0),
            Integer::from(1),
        );
        let densities = LocalDensities::new(form);

        let real_dens = densities.real_density(&Integer::from(5));
        assert!(real_dens.numerator() > &Integer::zero());

        let p_adic_dens = densities.p_adic_density(&Integer::from(1), 3);
        assert!(p_adic_dens.numerator() > &Integer::zero());
    }

    #[test]
    fn test_reduced_form_idempotent() {
        // An already reduced form should remain unchanged
        let form = QuadraticForm::binary(
            Integer::from(1),
            Integer::from(0),
            Integer::from(1),
        );
        let reduced = form.reduced_form();
        assert_eq!(form, reduced);
    }

    #[test]
    fn test_genus_theory_basics() {
        let form = QuadraticForm::binary(
            Integer::from(1),
            Integer::from(0),
            Integer::from(1),
        );
        let genus = GenusTheory::new(vec![form]);

        assert_eq!(genus.genus_size(), 1);
        assert_eq!(genus.discriminant, Integer::from(-4));

        let class_num = genus.class_number_estimate();
        assert!(class_num >= Integer::one());
    }

    #[test]
    fn test_analytics_integration() {
        let form = QuadraticForm::binary(
            Integer::from(1),
            Integer::from(0),
            Integer::from(1),
        );
        let analytics = QuadraticFormAnalytics::new(form, 20);

        let props = analytics.automorphic_properties();
        assert!(props.is_modular);
        assert!(props.weight > Integer::zero());
    }
}
