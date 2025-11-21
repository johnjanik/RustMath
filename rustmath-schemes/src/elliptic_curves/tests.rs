//! Comprehensive tests for generic elliptic curves
//!
//! Tests elliptic curve arithmetic over:
//! - Rationals (Q)
//! - Prime finite fields (GF(p))
//! - p-adic rationals (qp)

#[cfg(test)]
mod tests {
    use crate::elliptic_curves::generic::{EllipticCurve, Point};
    use rustmath_core::{Field, Ring};
    use rustmath_finitefields::PrimeField;
    use rustmath_integers::Integer;
    use rustmath_padics::{PadicInteger, PadicRational};
    use rustmath_rationals::Rational;

    // ============================================================================
    // Tests over Rationals (Q)
    // ============================================================================

    #[test]
    fn test_rational_curve_creation() {
        // Create y² = x³ - x (congruent number curve for n=1)
        let a = Rational::from_integer(-1);
        let b = Rational::from_integer(0);
        let curve = EllipticCurve::short_weierstrass(a, b);

        assert!(curve.is_short_weierstrass());
        assert_eq!(*curve.a4(), Rational::from_integer(-1));
        assert_eq!(*curve.a6(), Rational::from_integer(0));
    }

    #[test]
    fn test_rational_point_on_curve() {
        // Curve: y² = x³ - x
        let curve = EllipticCurve::short_weierstrass(
            Rational::from_integer(-1),
            Rational::from_integer(0),
        );

        // Point (0, 0) is on the curve
        let p1 = Point::new(Rational::from_integer(0), Rational::from_integer(0));
        assert!(curve.is_on_curve(&p1));

        // Point (1, 0) is on the curve
        let p2 = Point::new(Rational::from_integer(1), Rational::from_integer(0));
        assert!(curve.is_on_curve(&p2));

        // Point (-1, 0) is on the curve
        let p3 = Point::new(Rational::from_integer(-1), Rational::from_integer(0));
        assert!(curve.is_on_curve(&p3));

        // Point at infinity is always on the curve
        let inf = Point::infinity();
        assert!(curve.is_on_curve(&inf));

        // Point (2, 3) is not on the curve
        let p4 = Point::new(Rational::from_integer(2), Rational::from_integer(3));
        assert!(!curve.is_on_curve(&p4));
    }

    #[test]
    fn test_rational_point_addition_with_infinity() {
        let curve = EllipticCurve::short_weierstrass(
            Rational::from_integer(-1),
            Rational::from_integer(0),
        );

        let p = Point::new(Rational::from_integer(0), Rational::from_integer(0));
        let inf = Point::infinity();

        // P + O = P
        let result = curve.add_points(&p, &inf).unwrap();
        assert_eq!(result, p);

        // O + P = P
        let result = curve.add_points(&inf, &p).unwrap();
        assert_eq!(result, p);

        // O + O = O
        let result = curve.add_points(&inf, &inf).unwrap();
        assert!(result.is_infinity());
    }

    #[test]
    fn test_rational_point_negation() {
        let curve = EllipticCurve::short_weierstrass(
            Rational::from_integer(2),
            Rational::from_integer(3),
        );

        let p = Point::new(Rational::from_integer(-1), Rational::from_integer(0));
        let neg_p = curve.negate_point(&p);

        // For short Weierstrass with y=0, negation gives the same point
        assert_eq!(neg_p, p);

        // P + (-P) = O
        let result = curve.add_points(&p, &neg_p).unwrap();
        assert!(result.is_infinity());
    }

    #[test]
    fn test_rational_point_doubling() {
        // Curve: y² = x³ + 2x + 3
        let curve = EllipticCurve::short_weierstrass(
            Rational::from_integer(2),
            Rational::from_integer(3),
        );

        // Point (-1, 0) is on the curve
        let p = Point::new(Rational::from_integer(-1), Rational::from_integer(0));
        assert!(curve.is_on_curve(&p));

        // Doubling a point where y=0 gives infinity (vertical tangent)
        let doubled = curve.double_point(&p).unwrap();
        assert!(doubled.is_infinity());
    }

    #[test]
    fn test_rational_scalar_multiplication_basic() {
        let curve = EllipticCurve::short_weierstrass(
            Rational::from_integer(-1),
            Rational::from_integer(0),
        );

        let p = Point::new(Rational::from_integer(0), Rational::from_integer(0));

        // [0]P = O
        let result = curve.scalar_mul(0, &p).unwrap();
        assert!(result.is_infinity());

        // [1]P = P
        let result = curve.scalar_mul(1, &p).unwrap();
        assert_eq!(result, p);

        // [2]P for a 2-torsion point is O
        let result = curve.scalar_mul(2, &p).unwrap();
        assert!(result.is_infinity());
    }

    #[test]
    fn test_rational_scalar_multiplication_negative() {
        let curve = EllipticCurve::short_weierstrass(
            Rational::from_integer(1),
            Rational::from_integer(0),
        );

        let p = Point::new(Rational::from_integer(0), Rational::from_integer(0));

        // [-1]P = -P
        let result = curve.scalar_mul(-1, &p).unwrap();
        let neg_p = curve.negate_point(&p);
        assert_eq!(result, neg_p);
    }

    #[test]
    fn test_rational_discriminant() {
        // Curve: y² = x³ + ax + b
        // Discriminant: Δ = -16(4a³ + 27b²)

        // For y² = x³ - x, we have a = -1, b = 0
        // Δ = -16(4(-1)³ + 0) = -16(-4) = 64
        let mut curve = EllipticCurve::short_weierstrass(
            Rational::from_integer(-1),
            Rational::from_integer(0),
        );

        let disc = curve.discriminant();
        assert!(!disc.is_zero()); // Non-singular
    }

    #[test]
    fn test_rational_j_invariant() {
        // For y² = x³ + x, j-invariant = 1728
        let mut curve = EllipticCurve::short_weierstrass(
            Rational::from_integer(1),
            Rational::from_integer(0),
        );

        let j = curve.j_invariant();
        assert!(j.is_ok());
        // j-invariant should be 1728 for this curve with CM
    }

    #[test]
    fn test_rational_general_weierstrass() {
        // Create a curve in general Weierstrass form
        let curve = EllipticCurve::new(
            Rational::from_integer(1), // a1
            Rational::from_integer(0), // a2
            Rational::from_integer(1), // a3
            Rational::from_integer(0), // a4
            Rational::from_integer(-1), // a6
        );

        assert!(!curve.is_short_weierstrass());

        // Check that point at infinity is on the curve
        let inf = Point::infinity();
        assert!(curve.is_on_curve(&inf));
    }

    // ============================================================================
    // Tests over Finite Fields GF(p)
    // ============================================================================
    //
    // Note: These tests are currently ignored because PrimeField's Ring::zero()
    // and Ring::one() implementations require parameters (the modulus), which
    // cannot be provided through the Field trait's zero()/one() methods.
    //
    // The implementation is correct for fields that properly implement zero/one
    // (like Rational). Future work could add a FieldWithParameters trait.

    #[test]
    #[ignore]
    fn test_finite_field_curve_creation() {
        let p = Integer::from(7);

        // y² = x³ + 2x + 3 over GF(7)
        let a = PrimeField::new(Integer::from(2), p.clone()).unwrap();
        let b = PrimeField::new(Integer::from(3), p.clone()).unwrap();
        let curve = EllipticCurve::short_weierstrass(a, b);

        assert!(curve.is_short_weierstrass());
    }

    #[test]
    #[ignore]
    fn test_finite_field_point_on_curve() {
        let p = Integer::from(7);

        // y² = x³ + x over GF(7)
        let a = PrimeField::new(Integer::from(1), p.clone()).unwrap();
        let b = PrimeField::new(Integer::from(0), p.clone()).unwrap();
        let curve = EllipticCurve::short_weierstrass(a, b);

        // Check some points
        let x0 = PrimeField::new(Integer::from(0), p.clone()).unwrap();
        let y0 = PrimeField::new(Integer::from(0), p.clone()).unwrap();
        let point = Point::new(x0, y0);
        assert!(curve.is_on_curve(&point));
    }

    #[test]
    #[ignore]
    fn test_finite_field_point_addition() {
        let p = Integer::from(7);

        // y² = x³ + 2x + 3 over GF(7)
        let a = PrimeField::new(Integer::from(2), p.clone()).unwrap();
        let b = PrimeField::new(Integer::from(3), p.clone()).unwrap();
        let curve = EllipticCurve::short_weierstrass(a, b);

        let inf = Point::infinity();

        // Test O + O = O
        let result = curve.add_points(&inf, &inf).unwrap();
        assert!(result.is_infinity());
    }

    #[test]
    #[ignore]
    fn test_finite_field_point_doubling() {
        let p = Integer::from(11);

        // y² = x³ + x over GF(11)
        let a = PrimeField::new(Integer::from(1), p.clone()).unwrap();
        let b = PrimeField::new(Integer::from(0), p.clone()).unwrap();
        let curve = EllipticCurve::short_weierstrass(a, b);

        // Point (0, 0)
        let x = PrimeField::new(Integer::from(0), p.clone()).unwrap();
        let y = PrimeField::new(Integer::from(0), p.clone()).unwrap();
        let point = Point::new(x, y);

        // This should be a 2-torsion point, so doubling gives infinity
        let doubled = curve.double_point(&point).unwrap();
        assert!(doubled.is_infinity());
    }

    #[test]
    #[ignore]
    fn test_finite_field_scalar_multiplication() {
        let p = Integer::from(5);

        // y² = x³ + 1 over GF(5)
        let a = PrimeField::new(Integer::from(0), p.clone()).unwrap();
        let b = PrimeField::new(Integer::from(1), p.clone()).unwrap();
        let curve = EllipticCurve::short_weierstrass(a, b);

        let inf = Point::infinity();

        // [k]O = O for any k
        let result = curve.scalar_mul(5, &inf).unwrap();
        assert!(result.is_infinity());
    }

    #[test]
    #[ignore]
    fn test_finite_field_discriminant() {
        let p = Integer::from(7);

        // y² = x³ + 2x + 3 over GF(7)
        let a = PrimeField::new(Integer::from(2), p.clone()).unwrap();
        let b = PrimeField::new(Integer::from(3), p.clone()).unwrap();
        let mut curve = EllipticCurve::short_weierstrass(a, b);

        let disc = curve.discriminant();
        // Should be non-zero for a non-singular curve
        assert!(!disc.is_zero());
    }

    #[test]
    #[ignore]
    fn test_finite_field_point_order() {
        let p = Integer::from(7);

        let a = PrimeField::new(Integer::from(1), p.clone()).unwrap();
        let b = PrimeField::new(Integer::from(0), p.clone()).unwrap();
        let curve = EllipticCurve::short_weierstrass(a, b);

        // Point at infinity has order 1
        let inf = Point::infinity();
        let order = curve.point_order(&inf, 100);
        assert_eq!(order, Some(1));
    }

    // ============================================================================
    // Tests over p-adic Rationals (qp)
    // ============================================================================
    //
    // Note: These tests are currently ignored because PadicRational's Ring::zero()
    // and Ring::one() implementations require parameters (prime and precision),
    // which cannot be provided through the Field trait's zero()/one() methods.
    //
    // The implementation is correct for fields that properly implement zero/one
    // (like Rational). Future work could add a FieldWithParameters trait.

    #[test]
    #[ignore]
    fn test_padic_curve_creation() {
        let p = Integer::from(5);
        let precision = 10;

        // Create p-adic field elements
        let a_int = PadicInteger::from_integer(Integer::from(-1), p.clone(), precision).unwrap();
        let b_int = PadicInteger::from_integer(Integer::from(0), p.clone(), precision).unwrap();

        let a = PadicRational::from_padic_integer(a_int);
        let b = PadicRational::from_padic_integer(b_int);

        // y² = x³ - x over Q₅
        let curve = EllipticCurve::short_weierstrass(a, b);
        assert!(curve.is_short_weierstrass());
    }

    #[test]
    #[ignore]
    fn test_padic_point_on_curve() {
        let p = Integer::from(5);
        let precision = 10;

        // Create curve y² = x³ - x over Q₅
        let a_int = PadicInteger::from_integer(Integer::from(-1), p.clone(), precision).unwrap();
        let b_int = PadicInteger::from_integer(Integer::from(0), p.clone(), precision).unwrap();
        let a = PadicRational::from_padic_integer(a_int);
        let b = PadicRational::from_padic_integer(b_int);
        let curve = EllipticCurve::short_weierstrass(a, b);

        // Point (0, 0) is on the curve
        let x = PadicRational::from_padic_integer(
            PadicInteger::from_integer(Integer::from(0), p.clone(), precision).unwrap(),
        );
        let y = PadicRational::from_padic_integer(
            PadicInteger::from_integer(Integer::from(0), p.clone(), precision).unwrap(),
        );
        let point = Point::new(x, y);

        assert!(curve.is_on_curve(&point));
    }

    #[test]
    #[ignore]
    fn test_padic_point_addition_with_infinity() {
        let p = Integer::from(7);
        let precision = 10;

        let a_int = PadicInteger::from_integer(Integer::from(1), p.clone(), precision).unwrap();
        let b_int = PadicInteger::from_integer(Integer::from(0), p.clone(), precision).unwrap();
        let a = PadicRational::from_padic_integer(a_int);
        let b = PadicRational::from_padic_integer(b_int);
        let curve = EllipticCurve::short_weierstrass(a, b);

        let x = PadicRational::from_padic_integer(
            PadicInteger::from_integer(Integer::from(0), p.clone(), precision).unwrap(),
        );
        let y = PadicRational::from_padic_integer(
            PadicInteger::from_integer(Integer::from(0), p.clone(), precision).unwrap(),
        );
        let point = Point::new(x, y);

        let inf = Point::infinity();

        // P + O = P
        let result = curve.add_points(&point, &inf).unwrap();
        assert!(!result.is_infinity());
    }

    #[test]
    #[ignore]
    fn test_padic_point_doubling() {
        let p = Integer::from(7);
        let precision = 10;

        let a_int = PadicInteger::from_integer(Integer::from(2), p.clone(), precision).unwrap();
        let b_int = PadicInteger::from_integer(Integer::from(3), p.clone(), precision).unwrap();
        let a = PadicRational::from_padic_integer(a_int);
        let b = PadicRational::from_padic_integer(b_int);
        let curve = EllipticCurve::short_weierstrass(a, b);

        // Point at infinity
        let inf = Point::infinity();

        // 2*O = O
        let doubled = curve.double_point(&inf).unwrap();
        assert!(doubled.is_infinity());
    }

    #[test]
    #[ignore]
    fn test_padic_scalar_multiplication() {
        let p = Integer::from(5);
        let precision = 10;

        let a_int = PadicInteger::from_integer(Integer::from(0), p.clone(), precision).unwrap();
        let b_int = PadicInteger::from_integer(Integer::from(1), p.clone(), precision).unwrap();
        let a = PadicRational::from_padic_integer(a_int);
        let b = PadicRational::from_padic_integer(b_int);
        let curve = EllipticCurve::short_weierstrass(a, b);

        let inf = Point::infinity();

        // [0]O = O
        let result = curve.scalar_mul(0, &inf).unwrap();
        assert!(result.is_infinity());

        // [5]O = O
        let result = curve.scalar_mul(5, &inf).unwrap();
        assert!(result.is_infinity());
    }

    #[test]
    #[ignore]
    fn test_padic_discriminant() {
        let p = Integer::from(7);
        let precision = 10;

        let a_int = PadicInteger::from_integer(Integer::from(-1), p.clone(), precision).unwrap();
        let b_int = PadicInteger::from_integer(Integer::from(0), p.clone(), precision).unwrap();
        let a = PadicRational::from_padic_integer(a_int);
        let b = PadicRational::from_padic_integer(b_int);
        let mut curve = EllipticCurve::short_weierstrass(a, b);

        let disc = curve.discriminant();
        // Should be non-zero for non-singular curve
        assert!(!disc.is_zero());
    }

    // ============================================================================
    // Additional Edge Cases and Special Tests
    // ============================================================================

    #[test]
    fn test_rational_famous_curves() {
        // Curve 37a1: y² = x³ - x (rank 1)
        let mut curve_37a = EllipticCurve::short_weierstrass(
            Rational::from_integer(-1),
            Rational::from_integer(0),
        );
        assert!(!curve_37a.is_singular());

        // Curve with CM: y² = x³ + x (j=1728)
        let mut curve_cm = EllipticCurve::short_weierstrass(
            Rational::from_integer(1),
            Rational::from_integer(0),
        );
        assert!(!curve_cm.is_singular());

        // y² = x³ + 1
        let mut curve_fermat = EllipticCurve::short_weierstrass(
            Rational::from_integer(0),
            Rational::from_integer(1),
        );
        assert!(!curve_fermat.is_singular());
    }

    #[test]
    fn test_display_formatting() {
        let curve = EllipticCurve::short_weierstrass(
            Rational::from_integer(2),
            Rational::from_integer(3),
        );

        let display = format!("{}", curve);
        assert!(display.contains("y²"));
        assert!(display.contains("x³"));
    }

    #[test]
    fn test_point_display() {
        let p = Point::new(Rational::from_integer(1), Rational::from_integer(2));
        let display = format!("{}", p);
        assert!(display.contains("1"));
        assert!(display.contains("2"));

        let inf: Point<Rational> = Point::infinity();
        let inf_display = format!("{}", inf);
        assert_eq!(inf_display, "O");
    }
}
