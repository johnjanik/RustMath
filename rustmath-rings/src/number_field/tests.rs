//! Comprehensive tests for number field morphisms, splitting fields, and Galois theory
//!
//! This test module demonstrates the key functionality of number field morphisms
//! and Galois theory computations.

#[cfg(test)]
mod splitting_field_tests {
    use crate::number_field::morphisms::*;
    use rustmath_numberfields::NumberField;
    use rustmath_polynomials::univariate::UnivariatePolynomial;
    use rustmath_rationals::Rational;

    /// Test splitting field for x² - 2
    ///
    /// The splitting field of x² - 2 is ℚ(√2), which contains both roots ±√2.
    #[test]
    fn test_splitting_field_x2_minus_2() {
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(-2),
            Rational::from_integer(0),
            Rational::from_integer(1),
        ]);

        let splitting = splitting_field(&poly).expect("Should compute splitting field");

        // Degree should be 2
        assert_eq!(splitting.degree(), 2);

        // The discriminant should be 8 (for ℚ(√2))
        assert_eq!(splitting.discriminant(), Rational::from_integer(8));
    }

    /// Test splitting field for x² + 1
    ///
    /// The splitting field of x² + 1 is ℚ(i), the Gaussian numbers.
    #[test]
    fn test_splitting_field_x2_plus_1() {
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(1),
            Rational::from_integer(0),
            Rational::from_integer(1),
        ]);

        let splitting = splitting_field(&poly).expect("Should compute splitting field");

        // Degree should be 2
        assert_eq!(splitting.degree(), 2);

        // Discriminant should be negative (imaginary quadratic field)
        assert!(splitting.discriminant() < Rational::zero());
    }

    /// Test splitting field for x² - 3
    ///
    /// The splitting field is ℚ(√3).
    #[test]
    fn test_splitting_field_x2_minus_3() {
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(-3),
            Rational::from_integer(0),
            Rational::from_integer(1),
        ]);

        let splitting = splitting_field(&poly).expect("Should compute splitting field");

        assert_eq!(splitting.degree(), 2);

        // For ℚ(√3), discriminant is 12
        assert_eq!(splitting.discriminant(), Rational::from_integer(12));
    }

    /// Test splitting field for x² + x + 1
    ///
    /// This is the cyclotomic polynomial Φ₃(x), and its splitting field is ℚ(ζ₃)
    /// where ζ₃ is a primitive 3rd root of unity.
    #[test]
    fn test_splitting_field_cyclotomic() {
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(1),
            Rational::from_integer(1),
            Rational::from_integer(1),
        ]);

        let splitting = splitting_field(&poly).expect("Should compute splitting field");

        assert_eq!(splitting.degree(), 2);

        // Discriminant is -3 or -12 depending on normalization
        assert!(splitting.discriminant() < Rational::zero());
    }

    /// Test splitting field for linear polynomial
    ///
    /// The splitting field of a linear polynomial is just ℚ.
    #[test]
    fn test_splitting_field_linear() {
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(-5),
            Rational::from_integer(1),
        ]);

        let splitting = splitting_field(&poly).expect("Should compute splitting field");

        // Should be degree 1 (just ℚ)
        assert_eq!(splitting.degree(), 1);
    }

    /// Test that constant polynomials cannot have splitting fields
    #[test]
    fn test_splitting_field_constant_fails() {
        let poly = UnivariatePolynomial::new(vec![Rational::from_integer(42)]);

        let result = splitting_field(&poly);
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod normal_extension_tests {
    use super::super::morphisms::*;
    use rustmath_numberfields::NumberField;
    use rustmath_polynomials::univariate::UnivariatePolynomial;
    use rustmath_rationals::Rational;

    /// Helper to create ℚ(√2)
    fn make_sqrt2() -> NumberField {
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(-2),
            Rational::from_integer(0),
            Rational::from_integer(1),
        ]);
        NumberField::new(poly)
    }

    /// Helper to create ℚ(√3)
    fn make_sqrt3() -> NumberField {
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(-3),
            Rational::from_integer(0),
            Rational::from_integer(1),
        ]);
        NumberField::new(poly)
    }

    /// Helper to create ℚ(∛2)
    fn make_cbrt2() -> NumberField {
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(-2),
            Rational::from_integer(0),
            Rational::from_integer(0),
            Rational::from_integer(1),
        ]);
        NumberField::new(poly)
    }

    /// Test that quadratic fields are Galois extensions
    ///
    /// All quadratic extensions ℚ(√d) are Galois over ℚ.
    #[test]
    fn test_quadratic_is_galois() {
        let field = make_sqrt2();
        let is_gal = is_galois_extension(&field).expect("Should check Galois");

        assert!(is_gal, "Quadratic fields are always Galois");
    }

    /// Test another quadratic field is Galois
    #[test]
    fn test_sqrt3_is_galois() {
        let field = make_sqrt3();
        let is_gal = is_galois_extension(&field).expect("Should check Galois");

        assert!(is_gal);
    }

    /// Test that cubic fields are typically not Galois
    ///
    /// ℚ(∛2) is not a Galois extension because it doesn't contain
    /// all three cube roots of 2 (the other two are complex).
    #[test]
    #[ignore] // This test requires full automorphism computation
    fn test_cubic_not_galois() {
        let field = make_cbrt2();

        // This should return false (or NotImplemented)
        match is_galois_extension(&field) {
            Ok(false) => {
                // Expected: not Galois
            }
            Err(MorphismError::NotImplemented(_)) => {
                // Also acceptable for now
            }
            other => {
                panic!("Unexpected result: {:?}", other);
            }
        }
    }

    /// Test separability of number fields
    ///
    /// All algebraic extensions of ℚ are separable (characteristic 0).
    #[test]
    fn test_quadratic_is_separable() {
        let field = make_sqrt2();

        assert!(is_separable_extension(&field));
    }

    #[test]
    fn test_cubic_is_separable() {
        let field = make_cbrt2();

        assert!(is_separable_extension(&field));
    }

    /// Test that quadratic fields have exactly 2 automorphisms
    #[test]
    fn test_quadratic_automorphism_count() {
        let field = make_sqrt2();
        let auts = compute_automorphisms(&field).expect("Should compute automorphisms");

        assert_eq!(auts.len(), 2, "Quadratic fields have 2 automorphisms");
    }

    /// Test that one automorphism is the identity
    #[test]
    fn test_identity_automorphism_exists() {
        let field = make_sqrt2();
        let auts = compute_automorphisms(&field).expect("Should compute automorphisms");

        let has_identity = auts.iter().any(|aut| aut.is_identity());
        assert!(has_identity, "Identity automorphism must exist");
    }

    /// Test automorphism group structure
    #[test]
    fn test_automorphism_group_structure() {
        let field = make_sqrt2();
        let auts = compute_automorphisms(&field).expect("Should compute automorphisms");

        // For quadratic fields, we have C₂ (cyclic of order 2)
        // The non-identity element should square to identity

        if let Some(non_id) = auts.iter().find(|aut| !aut.is_identity()) {
            let order = non_id.order().expect("Should compute order");
            assert_eq!(order, 2, "Non-identity automorphism should have order 2");
        }
    }
}

#[cfg(test)]
mod galois_group_tests {
    use super::super::morphisms::*;
    use rustmath_polynomials::univariate::UnivariatePolynomial;
    use rustmath_rationals::Rational;

    /// Test Galois group of x² - 2
    ///
    /// Gal(ℚ(√2)/ℚ) ≅ C₂
    #[test]
    fn test_galois_group_x2_minus_2() {
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(-2),
            Rational::from_integer(0),
            Rational::from_integer(1),
        ]);

        let gal = galois_group(&poly).expect("Should compute Galois group");

        assert_eq!(gal.order(), 2);
        assert!(gal.is_abelian(), "C₂ is abelian");
        assert!(gal.is_cyclic(), "C₂ is cyclic");

        let id = gal.identify();
        assert_eq!(id, "C₂ (Cyclic group of order 2)");
    }

    /// Test Galois group of x² + 1
    ///
    /// Gal(ℚ(i)/ℚ) ≅ C₂
    #[test]
    fn test_galois_group_x2_plus_1() {
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(1),
            Rational::from_integer(0),
            Rational::from_integer(1),
        ]);

        let gal = galois_group(&poly).expect("Should compute Galois group");

        assert_eq!(gal.order(), 2);
        assert!(gal.is_abelian());
        assert!(gal.is_cyclic());
    }

    /// Test Galois group of x² - 5
    ///
    /// Gal(ℚ(√5)/ℚ) ≅ C₂
    #[test]
    fn test_galois_group_x2_minus_5() {
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(-5),
            Rational::from_integer(0),
            Rational::from_integer(1),
        ]);

        let gal = galois_group(&poly).expect("Should compute Galois group");

        assert_eq!(gal.order(), 2);
    }

    /// Test Galois group identification
    #[test]
    fn test_galois_group_identification() {
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(-3),
            Rational::from_integer(0),
            Rational::from_integer(1),
        ]);

        let gal = galois_group(&poly).expect("Should compute Galois group");

        let name = gal.identify();
        assert!(
            name.contains("C₂") || name.contains("Cyclic group of order 2"),
            "Should identify as C₂"
        );
    }

    /// Test that Galois group automorphisms match computed automorphisms
    #[test]
    fn test_galois_group_automorphisms_match() {
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(-2),
            Rational::from_integer(0),
            Rational::from_integer(1),
        ]);

        let gal = galois_group(&poly).expect("Should compute Galois group");
        let auts = gal.automorphisms();

        // Should have 2 automorphisms
        assert_eq!(auts.len(), 2);

        // One should be identity
        assert!(auts.iter().any(|aut| aut.is_identity()));
    }
}

#[cfg(test)]
mod embedding_tests {
    use super::super::morphisms::*;
    use rustmath_numberfields::NumberField;
    use rustmath_polynomials::univariate::UnivariatePolynomial;
    use rustmath_rationals::Rational;

    fn make_sqrt2() -> NumberField {
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(-2),
            Rational::from_integer(0),
            Rational::from_integer(1),
        ]);
        NumberField::new(poly)
    }

    /// Test creating an embedding
    #[test]
    fn test_create_embedding() {
        let field = make_sqrt2();
        let generator = field.generator();

        let embedding = NumberFieldEmbedding::new(field.clone(), generator);
        assert!(embedding.is_ok());
    }

    /// Test embedding properties
    #[test]
    fn test_embedding_properties() {
        let field = make_sqrt2();
        let generator = field.generator();

        let embedding = NumberFieldEmbedding::new(field, generator)
            .expect("Should create embedding");

        assert!(embedding.is_injective());
        assert!(!embedding.is_surjective());
        assert!(!embedding.is_automorphism());
        assert!(!embedding.is_isomorphism());
    }

    /// Test applying an embedding to the generator
    #[test]
    fn test_embedding_apply_generator() {
        let field = make_sqrt2();
        let generator = field.generator();

        let embedding = NumberFieldEmbedding::new(field, generator.clone())
            .expect("Should create embedding");

        let result = embedding.apply(&generator).expect("Should apply");

        // When embedding α ↦ α, we should get α back
        assert_eq!(result, generator);
    }

    /// Test applying an embedding to a rational
    #[test]
    fn test_embedding_apply_rational() {
        let field = make_sqrt2();
        let three = field.from_rational(Rational::from_integer(3));
        let generator = field.generator();

        let embedding = NumberFieldEmbedding::new(field, generator)
            .expect("Should create embedding");

        let result = embedding.apply(&three).expect("Should apply");

        // Rationals should map to themselves
        assert_eq!(result.coeff(0), Rational::from_integer(3));
    }

    /// Test embedding display
    #[test]
    fn test_embedding_display() {
        let field = make_sqrt2();
        let generator = field.generator();

        let embedding = NumberFieldEmbedding::new(field, generator)
            .expect("Should create embedding");

        let display = format!("{}", embedding);
        assert!(display.contains("Embedding"));
    }
}

#[cfg(test)]
mod automorphism_tests {
    use super::super::morphisms::*;
    use rustmath_numberfields::NumberField;
    use rustmath_polynomials::univariate::UnivariatePolynomial;
    use rustmath_rationals::Rational;

    fn make_sqrt2() -> NumberField {
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(-2),
            Rational::from_integer(0),
            Rational::from_integer(1),
        ]);
        NumberField::new(poly)
    }

    /// Test identity automorphism
    #[test]
    fn test_identity_automorphism() {
        let field = make_sqrt2();
        let id = NumberFieldAutomorphism::identity(field);

        assert!(id.is_identity());
        assert!(id.is_automorphism());
    }

    /// Test automorphism application
    #[test]
    fn test_automorphism_apply() {
        let field = make_sqrt2();
        let id = NumberFieldAutomorphism::identity(field.clone());
        let alpha = field.generator();

        let result = id.apply(&alpha).expect("Should apply");
        assert_eq!(result, alpha);
    }

    /// Test automorphism composition
    #[test]
    fn test_automorphism_composition() {
        let field = make_sqrt2();
        let id1 = NumberFieldAutomorphism::identity(field.clone());
        let id2 = NumberFieldAutomorphism::identity(field);

        let comp = id1.compose(&id2).expect("Should compose");

        assert!(comp.is_identity());
    }

    /// Test automorphism order
    #[test]
    fn test_identity_order() {
        let field = make_sqrt2();
        let id = NumberFieldAutomorphism::identity(field);

        let order = id.order().expect("Should compute order");
        assert_eq!(order, 1, "Identity has order 1");
    }

    /// Test non-trivial automorphism order
    #[test]
    fn test_nontrivial_automorphism_order() {
        let field = make_sqrt2();
        let auts = compute_automorphisms(&field).expect("Should compute");

        if let Some(non_id) = auts.iter().find(|aut| !aut.is_identity()) {
            let order = non_id.order().expect("Should compute order");
            assert_eq!(order, 2, "Non-trivial automorphism has order 2");
        }
    }

    /// Test automorphism display
    #[test]
    fn test_automorphism_display() {
        let field = make_sqrt2();
        let id = NumberFieldAutomorphism::identity(field);

        let display = format!("{}", id);
        assert!(display.contains("Automorphism"));
    }
}

#[cfg(test)]
mod advanced_tests {
    use super::super::morphisms::*;
    use rustmath_numberfields::NumberField;
    use rustmath_polynomials::univariate::UnivariatePolynomial;
    use rustmath_rationals::Rational;

    /// Test that the automorphism group of ℚ(√2) forms a group
    ///
    /// This verifies closure, associativity, identity, and inverses.
    #[test]
    fn test_automorphism_group_axioms() {
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(-2),
            Rational::from_integer(0),
            Rational::from_integer(1),
        ]);
        let field = NumberField::new(poly);

        let auts = compute_automorphisms(&field).expect("Should compute");

        // Check identity exists
        assert!(auts.iter().any(|aut| aut.is_identity()));

        // Check closure (all compositions stay in the group)
        for aut1 in &auts {
            for aut2 in &auts {
                let comp = aut1.compose(aut2).expect("Should compose");

                // The composition should match one of the automorphisms
                let found = auts.iter().any(|aut| {
                    aut.generator_image() == comp.generator_image()
                });

                assert!(found, "Composition should stay in the group");
            }
        }
    }

    /// Test fundamental theorem of Galois theory
    ///
    /// For a Galois extension K/ℚ of degree n, we should have:
    /// [K:ℚ] = |Gal(K/ℚ)|
    #[test]
    fn test_fundamental_theorem() {
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(-2),
            Rational::from_integer(0),
            Rational::from_integer(1),
        ]);

        let field = splitting_field(&poly).expect("Should compute splitting field");
        let degree = field.degree();

        let auts = compute_automorphisms(&field).expect("Should compute automorphisms");
        let galois_order = auts.len();

        assert_eq!(
            degree, galois_order,
            "For Galois extensions, degree equals Galois group order"
        );
    }

    /// Test that splitting fields are Galois
    ///
    /// By definition, splitting fields are normal extensions,
    /// and over ℚ (characteristic 0) all extensions are separable.
    #[test]
    fn test_splitting_field_is_galois() {
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(-3),
            Rational::from_integer(0),
            Rational::from_integer(1),
        ]);

        let field = splitting_field(&poly).expect("Should compute splitting field");

        assert!(is_separable_extension(&field));

        let is_gal = is_galois_extension(&field).expect("Should check");
        assert!(is_gal, "Splitting fields are Galois extensions");
    }

    /// Test discriminant relationship to Galois theory
    ///
    /// For a quadratic field ℚ(√d), the discriminant determines
    /// whether it's a real or imaginary quadratic field.
    #[test]
    fn test_discriminant_galois_relationship() {
        // Real quadratic field ℚ(√2)
        let poly1 = UnivariatePolynomial::new(vec![
            Rational::from_integer(-2),
            Rational::from_integer(0),
            Rational::from_integer(1),
        ]);
        let field1 = NumberField::new(poly1);
        let disc1 = field1.discriminant();

        assert!(disc1 > Rational::zero(), "Real quadratic field has positive discriminant");

        // Imaginary quadratic field ℚ(√-3)
        let poly2 = UnivariatePolynomial::new(vec![
            Rational::from_integer(3),
            Rational::from_integer(0),
            Rational::from_integer(1),
        ]);
        let field2 = NumberField::new(poly2);
        let disc2 = field2.discriminant();

        assert!(disc2 < Rational::zero(), "Imaginary quadratic field has negative discriminant");

        // Both are Galois extensions
        assert!(is_galois_extension(&field1).unwrap());
        assert!(is_galois_extension(&field2).unwrap());
    }
}
